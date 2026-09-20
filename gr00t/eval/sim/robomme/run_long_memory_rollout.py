#!/usr/bin/env python3
"""Paired-evaluation RoboMME client for the frozen HAMLET + episodic adapter.

Run this file with the *RoboMME simulator environment*, not the model environment.
The server sees observations and actions that were actually sent to ``env.step``;
it never receives future robot actions or the demonstration's oracle actions.
All passive demo endpoints use the exact backwards-aligned cache-training cadence.
The original HAMLET server mode uses this client too, with its adapter disabled.

CSV rows, not videos, are the completion journal. A model/scenario identity guard
prevents accidentally resuming another experiment. Simulator/server exceptions
are journaled and raised, never converted into ordinary failed robot episodes.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Callable
import uuid

import numpy as np


@dataclass
class Config:
    task_id: str
    model_config: str
    output_dir: str
    evaluation_id: str
    policy_client_host: str = "127.0.0.1"
    policy_client_port: int = 5555
    dataset: str = "test"
    n_episodes: int = 50
    max_episode_steps: int = 1300
    n_action_steps: int = 16
    seed: int = 6
    save_videos: bool = False
    request_timeout_seconds: int = 300


CSV_FIELDS = [
    "env_idx", "episode_idx", "episode_seed", "scenario_seed", "success", "status",
    "reward", "steps", "video_path", "task_instruction",
]


def demo_endpoints(n_demo: int, stride: int) -> list[int]:
    """All causal demo endpoints; final K-1 suffix matches original HAMLET.

    Cache event i spans endpoints i -> i+1. The first execution endpoint is
    n_demo and is NOT primed here: that call must also produce the first action.
    Repeated clamped frame zero is handled by HAMLET's original rolling cache.
    """
    if n_demo < 0 or stride <= 0:
        raise ValueError("n_demo must be nonnegative and stride positive")
    if n_demo == 0:
        return []
    return sorted({0, *range(n_demo - stride, -1, -stride)})


def episode_inference_seed(seed: int, task: str, episode: int) -> int:
    """Stable across model names, process restarts, and task enumeration order."""
    payload = json.dumps([int(seed), str(task), int(episode)], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little") % (2**31)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def build_observation(env_obs: dict, index: int, task_goal: str) -> dict:
    front = _numpy(env_obs["front_rgb_list"][index]).astype(np.uint8)
    wrist = _numpy(env_obs["wrist_rgb_list"][index]).astype(np.uint8)
    joint = _numpy(env_obs["joint_state_list"][index]).astype(np.float32).reshape(-1)[:7]
    gripper = _numpy(env_obs["gripper_state_list"][index]).astype(np.float32).reshape(-1)
    if joint.shape != (7,) or not len(gripper):
        raise ValueError(f"Unexpected RoboMME state dimensions: joint={joint.shape}, gripper={gripper.shape}")
    if not np.isfinite(joint).all() or not np.isfinite(gripper[0]):
        raise ValueError("Simulator returned nonfinite robot state")
    return {
        "video.front_view": front[None, None],
        "video.wrist_view": wrist[None, None],
        "state.joint_position": joint[None, None],
        "state.gripper_position": np.asarray([[[gripper[0]]]], dtype=np.float32),
        "annotation.human.action.task_description": [task_goal],
    }


def action_chunk(actions: dict, required: int) -> np.ndarray:
    joint = np.asarray(actions["action.joint_position"], dtype=np.float32)
    gripper = np.asarray(actions["action.gripper_close"], dtype=np.float32)
    if joint.ndim == 3 and joint.shape[0] == 1:
        joint = joint[0]
    if gripper.ndim == 3 and gripper.shape[0] == 1:
        gripper = gripper[0]
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    if joint.ndim != 2 or joint.shape[1] != 7 or gripper.shape != (len(joint), 1):
        raise ValueError(f"Unexpected action shapes: joint={joint.shape}, gripper={gripper.shape}")
    result = np.concatenate([joint, gripper], axis=-1)
    if len(result) < required:
        raise ValueError(f"Policy returned {len(result)} actions; stride requires {required}")
    if not np.isfinite(result).all():
        raise ValueError("Policy returned nonfinite actions")
    return result


def _video_frame(env_obs: dict, index: int = -1, demo: bool = False) -> np.ndarray:
    frame = np.hstack([
        _numpy(env_obs["front_rgb_list"][index]).astype(np.uint8),
        _numpy(env_obs["wrist_rgb_list"][index]).astype(np.uint8),
    ])
    if demo:
        frame = frame.copy()
        width = max(1, min(8, frame.shape[0] // 2, frame.shape[1] // 2))
        frame[:width] = frame[-width:] = [255, 0, 0]
        frame[:, :width] = frame[:, -width:] = [255, 0, 0]
    return frame


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray) or hasattr(value, "detach"):
        return _json_value(_numpy(value).tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_text(path: Path, text: str) -> None:
    """Replace one known result file atomically, keeping interrupted writes out."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def check_manifest(out_dir: Path, identity: dict) -> None:
    manifest = out_dir / "policy_manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous != identity:
            keys = sorted(key for key in previous.keys() | identity.keys() if previous.get(key) != identity.get(key))
            raise RuntimeError(f"Evaluation identity differs for {keys}; use a fresh output directory: {out_dir}")
    else:
        if (out_dir / "simulation_results.csv").exists():
            raise RuntimeError(f"Existing CSV has no identity manifest; refusing unsafe resume: {out_dir}")
        _atomic_text(manifest, json.dumps(identity, indent=2, allow_nan=False) + "\n")


def read_results(csv_path: Path) -> dict[int, dict]:
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(CSV_FIELDS).issubset(reader.fieldnames or []):
            raise RuntimeError(f"Incomplete or incompatible result CSV: {csv_path}")
        rows = {}
        for row in reader:
            episode = int(row["episode_idx"])
            if episode < 0 or episode in rows or row["success"] not in {"0", "1"}:
                raise RuntimeError(f"Invalid or duplicate episode result in {csv_path}: {row}")
            if row["status"] not in {"success", "fail", "timeout", "step_limit"}:
                raise RuntimeError(f"An incomplete/error episode cannot count as an evaluation result: {row}")
            if (row["status"] == "success") != (row["success"] == "1"):
                raise RuntimeError(f"Conflicting success and status in {csv_path}: {row}")
            rows[episode] = row
    return rows


def write_results(csv_path: Path, rows: dict[int, dict]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows[index] for index in sorted(rows))
    _atomic_text(csv_path, stream.getvalue())


def run_episode(
    policy: Any, env: Any, cfg: Config, episode: int, scenario_seed: int,
    journal: Callable[[dict], None],
) -> tuple[dict, list[np.ndarray]]:
    """Run one environment episode; injectable dependencies support CPU tests."""
    session = f"{cfg.task_id}:{episode}:{uuid.uuid4().hex}"
    seed = episode_inference_seed(cfg.seed, cfg.task_id, episode)
    reset_options = {"session_ids": [session], "episode_seed": seed}
    frames: list[np.ndarray] = []
    error: BaseException | None = None
    try:
        policy.reset(options=reset_options)
        env_obs, reset_info = env.reset()  # Builder already sets benchmark metadata seed/difficulty.
        goal = reset_info["task_goal"]
        task_goal = str(goal[0] if isinstance(goal, (list, tuple, np.ndarray)) else goal)
        n_demo = len(env_obs["front_rgb_list"]) - 1
        if n_demo < 0:
            raise ValueError("RoboMME reset returned no observation")
        primes = demo_endpoints(n_demo, cfg.n_action_steps)
        print(f"[rollout] {cfg.task_id} ep={episode} scenario_seed={scenario_seed} inference_seed={seed} demo={n_demo} endpoints={primes}", flush=True)
        if cfg.save_videos:
            frames.extend(_video_frame(env_obs, index, demo=True) for index in range(n_demo))

        first_call = True
        calls = 0
        def request(observation: dict, frame_index: int, *, passive: bool, executed: np.ndarray | None):
            nonlocal first_call, calls
            options = {
                "session_ids": [session], "reset_memory": [first_call],
                "prime_only": passive, "episode_seed": seed,
                "executed_actions": executed, "passive": passive,
                "frame_index": int(frame_index),
            }
            actions, diagnostics = policy.get_action(observation, options=options)
            journal({
                "kind": "policy_call", "episode_idx": episode, "episode_seed": seed,
                "session_id": session, "call": calls, "frame_index": int(frame_index),
                "passive": passive, "executed_action_count": 0 if executed is None else len(executed),
                "info": _json_value(diagnostics),
            })
            first_call = False
            calls += 1
            return actions

        for index in primes:
            request(build_observation(env_obs, index, task_goal), index, passive=True, executed=None)

        steps = 0
        reward_total = 0.0
        status = "step_limit"
        executed_prefix: np.ndarray | None = None
        done = False
        while steps < cfg.max_episode_steps and not done:
            actions = request(build_observation(env_obs, -1, task_goal), n_demo + steps,
                              passive=False, executed=executed_prefix)
            chunk = action_chunk(actions, cfg.n_action_steps)
            executed: list[np.ndarray] = []
            for action in chunk[:min(cfg.n_action_steps, cfg.max_episode_steps - steps)]:
                # Copy before env.step: wrappers may mutate their action argument.
                command = action.copy()
                if cfg.task_id in {"PatternLock", "RouteStick"}:
                    # These stick environments execute seven arm joints and ignore
                    # this dimension. Match the dataset's canonical absent-gripper
                    # command without changing any physically executed motion.
                    command[-1] = -1.0
                env_obs, reward, terminated, truncated, info = env.step(command.copy())
                executed.append(command)
                steps += 1
                step_status = str(info.get("status", ""))
                if step_status == "error":
                    raise RuntimeError(f"RoboMME simulator error at ep={episode} step={steps}: {info}")
                if env_obs is None:
                    raise RuntimeError(f"Simulator returned no observation at ep={episode} step={steps}")
                reward_total += float(_numpy(reward).reshape(-1)[0])
                if cfg.save_videos:
                    frames.append(_video_frame(env_obs))
                done = bool(terminated) or bool(truncated) or step_status in {"success", "fail", "timeout"}
                if done:
                    if step_status not in {"success", "fail", "timeout"}:
                        raise RuntimeError(f"Terminal episode has no benchmark success/failure status: {info}")
                    status = step_status
                    break
            # Only completed execution is sent on the NEXT observation. Predicted
            # but unexecuted action suffixes never enter the episodic memory.
            executed_prefix = np.stack(executed).astype(np.float32)

        row = {
            "env_idx": 0, "episode_idx": episode, "episode_seed": seed,
            "scenario_seed": int(scenario_seed), "success": int(status == "success"),
            "status": status, "reward": reward_total, "steps": steps,
            "video_path": "", "task_instruction": task_goal,
        }
        journal({"kind": "episode_complete", **row, "session_id": session})
        return row, frames
    except BaseException as exc:
        error = exc
        journal({"kind": "episode_error", "episode_idx": episode, "episode_seed": seed,
                 "session_id": session, "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        # Do not leave either memory state or simulator GPU resources alive across
        # episodes. Cleanup errors must not hide the original failure traceback.
        cleanup_errors = []
        for cleanup in (lambda: policy.reset(options=reset_options), env.close):
            try:
                cleanup()
            except Exception as exc:
                cleanup_errors.append(exc)
                journal({"kind": "cleanup_error", "episode_idx": episode, "error": str(exc)})
        if cleanup_errors and error is None:
            raise RuntimeError(f"Episode cleanup failed: {cleanup_errors}") from cleanup_errors[0]


def _resolve_config(cfg: Config) -> tuple[dict, str]:
    if cfg.n_episodes <= 0 or cfg.max_episode_steps <= 0 or cfg.n_action_steps <= 0:
        raise ValueError("Episode count, step limit, and action stride must be positive")
    if not cfg.evaluation_id:
        raise ValueError("An evaluation identity from the paired driver is required")
    config_path = Path(cfg.model_config).resolve()
    if config_path.is_dir():
        config_path /= "config.json"
    payload = config_path.read_bytes()
    model = json.loads(payload)
    if model.get("hamlet_mode") != "finetune":
        raise ValueError("This comparison requires a finetuned HAMLET base checkpoint")
    if int(model["memory_stride"]) != cfg.n_action_steps:
        raise ValueError("n_action_steps must equal the checkpoint's trained memory_stride")
    return model, hashlib.sha256(payload).hexdigest()


def _save_video(path: Path, frames: list[np.ndarray]) -> None:
    import imageio.v2 as imageio
    if not frames:
        raise RuntimeError("No video frames were captured")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".mp4", dir=path.parent)
    os.close(fd)
    try:
        imageio.mimsave(temporary, frames, fps=10)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(cfg: Config) -> None:
    model, config_digest = _resolve_config(cfg)
    # Renderer settings must precede the heavy simulator import. Simulator seed
    # comes from benchmark metadata; cfg.seed controls only policy/noise pairing.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("SAPIEN_RENDER_DEVICE", "cuda")
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    from gr00t.policy.server_client import PolicyClient
    import zmq

    builder = BenchmarkEnvBuilder(env_id=cfg.task_id, dataset=cfg.dataset,
                                  action_space="joint_angle", gui_render=False,
                                  max_steps=cfg.max_episode_steps)
    available = builder.get_episode_num()
    if cfg.n_episodes > available:
        raise ValueError(f"Requested {cfg.n_episodes} episodes, but {cfg.task_id}/{cfg.dataset} has {available}")
    metadata = builder.metadata_index
    scenarios = []
    for episode in range(available):
        if (cfg.task_id, episode) not in metadata:
            raise ValueError(f"Benchmark metadata is missing episode index {episode}")
        scenario_seed, difficulty = builder.resolve_episode(episode)
        if scenario_seed is None:
            raise ValueError(f"No metadata seed for episode {episode}; cannot guarantee paired scenarios")
        scenarios.append({"episode_idx": episode, "seed": int(scenario_seed), "difficulty": difficulty})
    metadata_digest = hashlib.sha256(json.dumps(scenarios, sort_keys=True).encode()).hexdigest()
    out_dir = Path(cfg.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    check_manifest(out_dir, {
        "version": 1, "evaluation_id": cfg.evaluation_id,
        "model_config": str(Path(cfg.model_config).resolve()), "model_config_sha256": config_digest,
        "task_id": cfg.task_id, "dataset": cfg.dataset, "seed": cfg.seed,
        "n_action_steps": cfg.n_action_steps, "max_episode_steps": cfg.max_episode_steps,
        "memory_window": int(model["memory_window"]), "demo_sampling": "backward_aligned_full_history",
        "scenario_metadata_sha256": metadata_digest,
    })
    csv_path = out_dir / "simulation_results.csv"
    rows = read_results(csv_path)
    for episode, row in rows.items():
        if episode >= available or int(row["episode_seed"]) != episode_inference_seed(cfg.seed, cfg.task_id, episode):
            raise RuntimeError(f"Existing result has an incompatible episode identity: {row}")
        if int(row["scenario_seed"]) != scenarios[episode]["seed"]:
            raise RuntimeError(f"Existing result has an incompatible scenario seed: {row}")
    missing = [episode for episode in range(cfg.n_episodes) if episode not in rows]
    print(f"[rollout] {cfg.task_id}: requested={cfg.n_episodes}, completed={cfg.n_episodes-len(missing)}, pending={len(missing)}", flush=True)
    policy = None
    start = time.monotonic()
    with (out_dir / "memory_diagnostics.jsonl").open("a", buffering=1) as journal_file:
        def journal(record: dict) -> None:
            journal_file.write(json.dumps({"timestamp": time.time(), **record}, allow_nan=False) + "\n")
        try:
            if missing:
                policy = PolicyClient(host=cfg.policy_client_host, port=cfg.policy_client_port)
                policy.socket.setsockopt(zmq.RCVTIMEO, cfg.request_timeout_seconds * 1000)
                policy.socket.setsockopt(zmq.SNDTIMEO, cfg.request_timeout_seconds * 1000)
                policy.socket.setsockopt(zmq.LINGER, 0)
            for episode in missing:
                seed = episode_inference_seed(cfg.seed, cfg.task_id, episode)
                random.seed(seed)
                np.random.seed(seed)
                env = builder.make_env_for_episode(episode, max_steps=cfg.max_episode_steps)
                row, frames = run_episode(policy, env, cfg, episode, scenarios[episode]["seed"], journal)
                if cfg.save_videos:
                    filename = f"robomme_{cfg.task_id}_env00-episode_{episode}-{'success' if row['success'] else 'failure'}.mp4"
                    _save_video(out_dir / filename, frames)
                    row["video_path"] = filename
                rows[episode] = row
                write_results(csv_path, rows)
                print(f"[rollout] ep={episode} status={row['status']} success={row['success']} steps={row['steps']}", flush=True)
        finally:
            if policy is not None:
                policy.socket.close(linger=0)
                policy.context.term()
    requested = [rows[episode] for episode in range(cfg.n_episodes)]
    successes = sum(int(row["success"]) for row in requested)
    summary = {
        "task_id": cfg.task_id, "episodes": len(requested), "successes": successes,
        "success_rate": successes / len(requested), "elapsed_seconds_this_invocation": time.monotonic() - start,
    }
    _atomic_text(out_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    _atomic_text(out_dir / "summary.txt", f"Task: {cfg.task_id}\nEpisodes: {len(requested)}\nSuccess rate: {summary['success_rate']:.4f}\n")
    print(f"[rollout] DONE {cfg.task_id}: {successes}/{len(requested)} = {100*summary['success_rate']:.2f}%", flush=True)


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task_id", "model_config", "output_dir", "evaluation_id"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--policy-client-host", default="127.0.0.1")
    parser.add_argument("--policy-client-port", type=int, default=5555)
    parser.add_argument("--dataset", choices=("train", "val", "test"), default="test")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=1300)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--request-timeout-seconds", type=int, default=300)
    return Config(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main(parse_args())
