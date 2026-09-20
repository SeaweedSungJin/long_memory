#!/usr/bin/env python3
"""V13 simulator client: ordinary rollout plus one RGB/text-only demo-tail RPC.

The original run_episode remains the owner of canonical priming, RNG seeds,
action cadence, execution feedback, success criteria and cleanup. Thin proxies
capture only reset RGB and insert a single RPC before the first execution call.
No GT action/state/benchmark solution is sent to the new memory endpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from gr00t.eval.sim.robomme.run_long_memory_rollout import (
    Config as OriginalConfig, _atomic_text, _json_value, _numpy, _resolve_config,
    _save_video, check_manifest, demo_endpoints, episode_inference_seed,
    read_results, run_episode as original_run_episode, write_results,
)

ENDPOINT = "ingest_demo_tail_v13"


@dataclass
class Config(OriginalConfig):
    include_tail: bool = True


class _ResetRGB:
    """Capture only public reset observations; delegate the simulator unchanged."""

    def __init__(self, env):
        self.env, self.observations, self.instruction = env, None, None

    def reset(self):
        observations, info = self.env.reset()
        self.observations = observations
        goal = info["task_goal"]
        self.instruction = str(goal[0] if isinstance(goal, (list, tuple, np.ndarray)) else goal)
        return observations, info

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()


class _TailRPC:
    def __init__(self, policy, reset_rgb, cfg, episode, journal):
        self.policy, self.reset_rgb = policy, reset_rgb
        self.cfg, self.episode, self.journal = cfg, episode, journal
        self.primes, self.executing = [], False

    def reset(self, options=None):
        # Original run_episode owns both pre-episode and finally cleanup calls.
        return self.policy.reset(options=options)

    def get_action(self, observation, options=None):
        options = options or {}
        passive = options.get("passive")
        if type(passive) is not bool:
            raise ValueError("V13 client requires the original explicit passive flag")
        if passive and self.executing:
            raise ValueError("Demo prime after execution")
        if not passive and not self.executing:
            self._ingest(options)
            self.executing = True
        result = self.policy.get_action(observation, options=options)
        if passive:
            self.primes.append(options["frame_index"])
        return result

    def _ingest(self, options):
        env_obs = self.reset_rgb.observations
        if env_obs is None or self.cfg.n_action_steps != 16:
            raise ValueError("V13 requires reset RGB and original stride16")
        n_demo = len(env_obs["front_rgb_list"]) - 1
        if n_demo < 0 or len(env_obs["wrist_rgb_list"]) != n_demo + 1:
            raise ValueError("V13 reset camera lengths differ")
        canonical = demo_endpoints(n_demo, 16)
        if self.primes != canonical or options.get("frame_index") != n_demo:
            raise ValueError("V13 canonical primes/first action changed")
        sid = options.get("session_ids")
        seed = options.get("episode_seed")
        if (not isinstance(sid, list) or len(sid) != 1 or not isinstance(sid[0], str)
                or type(seed) is not int or seed < 0):
            raise ValueError("Missing original session/episode seed")
        last = canonical[-1] if canonical else -1
        frames = list(range(max(0, n_demo - 15, last + 1), n_demo))
        if not frames:
            self.journal({"kind": "demo_tail_skipped", "episode_idx": self.episode,
                "episode_seed": seed, "session_id": sid[0], "n_demo": n_demo,
                "reason": "no_omitted_demo_frames"})
            return
        # Match original uint8 RGB conversion. No state/action column is read.
        images = {camera: np.stack([_numpy(env_obs[key][i]).astype(np.uint8) for i in frames])
                  for camera, key in (("front_view", "front_rgb_list"), ("wrist_view", "wrist_rgb_list"))}
        payload = {"session_id": sid[0], "episode_seed": seed, "n_demo": n_demo,
                   "frames": frames, "images": images,
                   "texts": [self.reset_rgb.instruction] * len(frames)}
        response = self.policy.call_endpoint(ENDPOINT, payload)
        expected = {"session_id": sid[0], "episode_seed": seed, "n_demo": n_demo,
                    "ingested_observations": len(frames), "first_tail_frame": frames[0],
                    "last_tail_frame": frames[-1], "canonical_observations": len(canonical),
                    "canonical_frame": last, "parent_unchanged": True,
                    "rng_preserved": True, "read_performed": False}
        if (not isinstance(response, dict) or set(response) != set(expected)
                or any(type(response[key]) is not type(value) or response[key] != value
                       for key, value in expected.items())):
            raise RuntimeError("V13 tail-ingest response identity/integrity differs")
        self.journal({"kind": "demo_tail_ingest", "episode_idx": self.episode,
                      "episode_seed": seed, "session_id": sid[0], "frames": frames,
                      "info": _json_value(response)})


def run_episode(policy, env, cfg, episode, scenario_seed, journal):
    if type(cfg.include_tail) is not bool:
        raise ValueError("Explicit tail client arm required")
    if not cfg.include_tail:
        return original_run_episode(policy, env, cfg, episode, scenario_seed, journal)
    reset_rgb = _ResetRGB(env)
    proxy = _TailRPC(policy, reset_rgb, cfg, episode, journal)
    return original_run_episode(proxy, reset_rgb, cfg, episode, scenario_seed, journal)


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
        "demo_tail_ingest": cfg.include_tail, "client_variant": "demo_tail_v13",
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
    parser.add_argument("--include-tail", action=argparse.BooleanOptionalAction, default=True)
    return Config(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main(parse_args())
