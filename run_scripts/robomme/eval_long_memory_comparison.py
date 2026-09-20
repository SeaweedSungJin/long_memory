#!/usr/bin/env python3
"""Paired baseline / Stage 1 / Stage 2 RoboMME closed-loop evaluation.

The model and simulator use separate, explicit Python environments. Each model
is served once, tasks run sequentially, and per-episode state/noise is reset by
the rollout client. No checkpoint is modified and no training is performed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from gr00t.eval.sim.robomme.compare_long_memory_results import TASKS, read_results, validate_result_identity, write_report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=Path("checkpoints/author_hamlet_robomme/checkpoint-60000"))
    parser.add_argument("--stage1-checkpoint", type=Path,
                        default=Path("runs/long_memory/stage1_full1600_2k_v1/checkpoint-004300"))
    parser.add_argument("--stage2-checkpoint", type=Path,
                        default=Path("runs/long_memory/stage2_full1600_5k_v1/checkpoint-002900"))
    parser.add_argument("--models", nargs="+", choices=("baseline", "stage1", "stage2"), default=["baseline", "stage1", "stage2"])
    parser.add_argument("--tasks", nargs="+", default=["BinFill", "PatternLock"], help="Task names, or 'all' for all 16 tasks")
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--dataset", choices=("train", "val", "test"), default="test")
    parser.add_argument("--seed", type=int, default=6, help="Same per-task/episode inference-noise seed schedule for every model")
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=1300)
    parser.add_argument("--save-videos", action="store_true", help="Optional mp4s; resume never relies on videos")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/eval/robomme/long-memory-smoke"))
    parser.add_argument("--server-python", type=Path, default=REPO_ROOT / ".venv/bin/python")
    parser.add_argument("--robomme-python", type=Path, default=Path("/home/sjkim/robomme_benchmark/.venv/bin/python"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--server-timeout", type=float, default=300)
    parser.add_argument("--task-timeout", type=float, default=0, help="Wall-clock seconds per task; 0 disables overall task deadline")
    parser.add_argument("--preflight-only", action="store_true", help="Check dependencies/checkpoint provenance without model or simulator loading")
    parser.add_argument("--report-only", action="store_true", help="Only regenerate existing comparison_summary.txt/json")
    return parser


def resolve_repo_path(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def python_path(path: Path) -> Path:
    # DO NOT resolve a venv's python symlink: executing its /usr/bin target loses
    # the virtual environment even though both filenames point to one binary.
    return Path(os.path.abspath(REPO_ROOT / path if not path.is_absolute() else path))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_options(args):
    if args.tasks == ["all"]:
        args.tasks = list(TASKS)
    if not args.tasks or any(task not in TASKS for task in args.tasks) or len(set(args.tasks)) != len(args.tasks):
        raise ValueError("--tasks must be unique RoboMME task names, or exactly 'all'")
    if "baseline" not in args.models or len(set(args.models)) != len(args.models):
        raise ValueError("--models must be unique and include baseline")
    if args.n_episodes < 1 or args.seed < 0 or args.n_action_steps < 1 or args.max_episode_steps < 1:
        raise ValueError("Episode/action/step counts must be positive and seed nonnegative")
    if args.server_timeout <= 0 or args.task_timeout < 0:
        raise ValueError("--server-timeout must be positive and --task-timeout nonnegative")


def check_dependencies(python: Path, imports: str, env: dict, label: str):
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"{label} Python not executable: {python}")
    result = subprocess.run([str(python), "-c", imports], cwd=REPO_ROOT, env=env,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    if result.returncode:
        raise RuntimeError(f"{label} environment check failed ({python}):\n{result.stdout}")
    print(f"[preflight] {label} environment OK: {python}", flush=True)


def benchmark_identity(args):
    """Snapshot installed simulator source and scenario metadata without an env.

    Package versions plus Python source / selected scenario JSON detect normal
    benchmark changes even when already-completed tasks would otherwise skip
    the rollout client's own metadata checks. Large 3-D assets are not hashed.
    """
    probe = ("import json, robomme, importlib.metadata as m\n"
             "versions = {}\n"
             "for name in ['robomme', 'mani_skill', 'mujoco', 'sapien', 'numpy', 'torch']:\n"
             "    try: versions[name] = m.version(name)\n"
             "    except m.PackageNotFoundError: versions[name] = None\n"
             "print('ROBOMME_IDENTITY=' + json.dumps({'roots': list(robomme.__path__), 'versions': versions}))")
    result = subprocess.run([str(python_path(args.robomme_python)), "-c", probe], cwd=REPO_ROOT,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    if result.returncode:
        raise RuntimeError(f"Cannot identify simulator installation:\n{result.stdout}")
    lines = [line.removeprefix("ROBOMME_IDENTITY=") for line in result.stdout.splitlines() if line.startswith("ROBOMME_IDENTITY=")]
    if len(lines) != 1:
        raise RuntimeError("Simulator installation identity probe returned no unique identity")
    identity = json.loads(lines[0])
    signatures = {}
    found_tasks = set()
    for root_name in identity["roots"]:
        root = Path(root_name).resolve()
        for path in sorted(root.rglob("*.py")):
            signatures[str(path)] = file_hash(path)
        for task in args.tasks:
            path = root / "env_metadata" / args.dataset / f"record_dataset_{task}_metadata.json"
            if path.is_file():
                signatures[str(path)] = file_hash(path)
                found_tasks.add(task)
    if set(args.tasks) != found_tasks:
        raise ValueError(f"Simulator scenario metadata missing for {sorted(set(args.tasks) - found_tasks)}")
    identity["source_and_scenario_sha256"] = signatures
    return identity


def build_identity(args):
    """Validate that every add-on was trained against the selected frozen base."""
    from gr00t.long_memory.hamlet import checkpoint_identity

    base = resolve_repo_path(args.base_model)
    base_identity = checkpoint_identity(base)
    config = json.loads((base / "config.json").read_text())
    if (config.get("hamlet_mode") != "finetune" or config.get("mem_cond_type", "cross_attn") != "cross_attn"
            or config.get("memory_type", "moment_token") != "moment_token"):
        raise ValueError("Evaluation requires the trained HAMLET moment-token cross-attention base")
    stride = int(config.get("memory_stride", 16))
    if args.n_action_steps != stride:
        raise ValueError(f"--n-action-steps {args.n_action_steps} must equal trained memory_stride {stride}")
    models = {}
    for name in args.models:
        entry = {"base_model": base_identity, "memory_checkpoint": None, "write_policy": "none"}
        if name != "baseline":
            checkpoint = resolve_repo_path(getattr(args, f"{name}_checkpoint"))
            metadata_path = checkpoint / "checkpoint.json"
            info = json.loads(metadata_path.read_text())
            stage = int(info["config"]["stage"])
            if stage != int(name[-1]):
                raise ValueError(f"{name} points to a Stage {stage} checkpoint: {checkpoint}")
            if info["metadata"]["base_model"] != base_identity:
                raise ValueError(f"{name} was trained against a different/changed base checkpoint; refusing comparison")
            weights = checkpoint / "model.safetensors"
            entry.update(memory_checkpoint=str(checkpoint), checkpoint_sha256=file_hash(metadata_path),
                         weights_sha256=file_hash(weights), stage=stage, step=info["step"],
                         memory_config=info["config"]["memory"], write_policy="all" if stage == 1 else "hard")
        models[name] = entry
    settings = {"tasks": args.tasks, "n_episodes": args.n_episodes, "dataset": args.dataset, "seed": args.seed,
                "n_action_steps": args.n_action_steps, "max_episode_steps": args.max_episode_steps,
                "save_videos": args.save_videos, "device": args.device}
    sources = [Path("run_scripts/robomme/serve_long_memory.py"), Path("run_scripts/robomme/eval_long_memory_comparison.py"),
               Path("gr00t/eval/sim/robomme/run_long_memory_rollout.py"), Path("gr00t/long_memory/online.py"),
               Path("gr00t/long_memory/online_policy.py"), Path("gr00t/long_memory/core.py"),
               Path("gr00t/long_memory/hamlet.py")]
    sources += [path.relative_to(REPO_ROOT) for path in sorted((REPO_ROOT / "gr00t/model").rglob("*.py"))]
    sources += [path.relative_to(REPO_ROOT) for path in sorted((REPO_ROOT / "gr00t/policy").glob("*.py"))]
    code = {str(path): file_hash(REPO_ROOT / path) for path in sources if (REPO_ROOT / path).is_file()}
    for required in ("run_scripts/robomme/serve_long_memory.py", "gr00t/eval/sim/robomme/run_long_memory_rollout.py"):
        if required not in code:
            raise ValueError(f"Evaluation entrypoint is missing: {required}")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors", "numpy")}
    identity = {"format_version": 1, "models": models, "settings": settings, "source_sha256": code,
                "policy_package_versions": versions, "benchmark": benchmark_identity(args),
                "server_python": str(python_path(args.server_python)), "robomme_python": str(python_path(args.robomme_python))}
    identity["evaluation_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def bind_manifest(output: Path, identity: dict):
    manifest = output / "comparison_manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError(f"{output} belongs to different checkpoints/settings/code; use a NEW --output-dir")
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is nonempty without a comparison manifest; use a NEW --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(identity, indent=2) + "\n")


def free_local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def server_ready(process, port: int, timeout: float):
    # Actual protocol ping, not merely a bound TCP socket. Fresh REQ sockets after
    # each timeout prevent ZMQ's send/receive state machine getting stuck.
    import msgpack
    import zmq

    context = zmq.Context()
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Policy server exited with {process.returncode} before readiness")
            request = context.socket(zmq.REQ)
            request.setsockopt(zmq.LINGER, 0)
            request.setsockopt(zmq.SNDTIMEO, 500)
            request.setsockopt(zmq.RCVTIMEO, 500)
            request.connect(f"tcp://127.0.0.1:{port}")
            try:
                request.send(msgpack.packb({"endpoint": "ping"}))
                reply = msgpack.unpackb(request.recv(), raw=False)
                if reply.get("status") == "ok":
                    return
            except zmq.ZMQError:
                pass
            finally:
                request.close()
            time.sleep(.5)
    finally:
        context.term()
    raise TimeoutError(f"Policy server not ready within {timeout}s")


def stop_process(process):
    """Only signal the dedicated process group this driver created."""
    if process is None:
        return
    # Descendants may survive after the group leader exits, so signal the owned
    # group even if poll() already observed the leader's termination.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_client(command, *, env, log_path: Path, server, timeout: float):
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[driver] command: " + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started, last_notice = time.monotonic(), time.monotonic()
        try:
            while process.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError("Policy server died during rollout; see server.log")
                if timeout and time.monotonic() - started > timeout:
                    raise TimeoutError(f"Task exceeded --task-timeout {timeout}s")
                if time.monotonic() - last_notice >= 30:
                    print(f"[eval] running ({int(time.monotonic() - started)}s); log: {log_path}", flush=True)
                    last_notice = time.monotonic()
                time.sleep(.5)
            if process.returncode:
                raise RuntimeError(f"Rollout exited with {process.returncode}; see {log_path}")
        finally:
            stop_process(process)


def run_evaluation(args, identity, env):
    output = resolve_repo_path(args.output_dir)
    bind_manifest(output, identity)
    lock = (output / ".driver.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f"Another evaluation driver already owns {output}; do not launch concurrent writers")
    failures = []
    try:
        for name, model in identity["models"].items():
            pending = []
            for task in args.tasks:
                csv = output / name / task / "simulation_results.csv"
                rows = read_results(csv, expected=args.n_episodes)
                if rows:
                    validate_result_identity(output, name, task, identity)
                if len(rows) == args.n_episodes:
                    print(f"[eval] resume: {name}/{task} already complete", flush=True)
                else:
                    pending.append(task)
            if not pending:
                continue
            folder = output / name
            folder.mkdir(exist_ok=True)
            port = free_local_port()
            server_command = [str(python_path(args.server_python)), "-u", str(REPO_ROOT / "run_scripts/robomme/serve_long_memory.py"),
                              "--base-model", model["base_model"]["path"], "--device", args.device,
                              "--host", "127.0.0.1", "--port", str(port)]
            if model["memory_checkpoint"]:
                server_command += ["--memory-checkpoint", model["memory_checkpoint"]]
            server = None
            with (folder / "server.log").open("a", encoding="utf-8") as log:
                try:
                    print(f"[eval] loading {name}; server log: {folder / 'server.log'}", flush=True)
                    log.write("\n[driver] command: " + json.dumps(server_command) + "\n")
                    log.flush()
                    server = subprocess.Popen(server_command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True)
                    server_ready(server, port, args.server_timeout)
                    for task in pending:
                        task_dir = folder / task
                        task_dir.mkdir(exist_ok=True)
                        command = [str(python_path(args.robomme_python)), "-u", str(REPO_ROOT / "gr00t/eval/sim/robomme/run_long_memory_rollout.py"),
                                   "--task-id", task, "--policy-client-host", "127.0.0.1", "--policy-client-port", str(port),
                                   "--dataset", args.dataset, "--n-episodes", str(args.n_episodes),
                                   "--max-episode-steps", str(args.max_episode_steps), "--n-action-steps", str(args.n_action_steps),
                                   "--model-config", model["base_model"]["path"], "--output-dir", str(task_dir),
                                   "--seed", str(args.seed), "--evaluation-id", identity["evaluation_id"] + ":" + name]
                        if args.save_videos:
                            command.append("--save-videos")
                        print(f"[eval] {name}/{task}: {args.n_episodes} episodes; {task_dir / 'rollout.log'}", flush=True)
                        try:
                            run_client(command, env=env, log_path=task_dir / "rollout.log", server=server, timeout=args.task_timeout)
                            if len(read_results(task_dir / "simulation_results.csv", expected=args.n_episodes)) != args.n_episodes:
                                raise RuntimeError("Rollout exited without all requested CSV episode results")
                            print(f"[eval] completed {name}/{task}", flush=True)
                        except (RuntimeError, TimeoutError, ValueError) as exc:
                            failures.append({"model": name, "task": task, "error": str(exc)})
                            print(f"[eval] ERROR {name}/{task}: {exc}", file=sys.stderr, flush=True)
                        write_report(output, bootstrap_samples=500)
                except (RuntimeError, TimeoutError) as exc:
                    failures.append({"model": name, "error": str(exc)})
                    print(f"[eval] ERROR {name}: {exc}; see {folder / 'server.log'}", file=sys.stderr, flush=True)
                finally:
                    stop_process(server)
    finally:
        try:
            (output / "driver_status.json").write_text(json.dumps({"failures": failures}, indent=2) + "\n")
            result, report = write_report(output)
            print(report, flush=True)
            print(f"[eval] Summary: {output / 'comparison_summary.txt'}", flush=True)
        finally:
            lock.close()
    return 1 if failures or not all(model["complete"] for model in result["models"].values()) else 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.report_only:
        _, report = write_report(resolve_repo_path(args.output_dir))
        print(report)
        return 0
    validate_options(args)
    env = dict(os.environ)
    env.update(PYTHONPATH=str(REPO_ROOT), PYTHONUNBUFFERED="1", GR00T_INFERENCE_SEED=str(args.seed))
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    # Use the model Python explicitly even if the invoking shell activates a
    # different environment. This also makes provenance helpers importable.
    server_python = python_path(args.server_python)
    if python_path(Path(sys.executable)) != server_python:
        return subprocess.call([str(server_python), str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
                               cwd=REPO_ROOT, env=env)
    check_dependencies(server_python, "import torch, cv2, transformers, safetensors, msgpack, zmq", env, "policy")
    check_dependencies(python_path(args.robomme_python), "import robomme, msgpack, zmq, pandas, imageio", env, "simulator")
    identity = build_identity(args)
    print(f"[preflight] compatible checkpoints; models={list(identity['models'])}; tasks={args.tasks}; n={args.n_episodes}", flush=True)
    print(f"[preflight] base metadata hash: {identity['models']['baseline']['base_model']['metadata_sha256']}", flush=True)
    if args.preflight_only:
        print("[preflight] No model, simulator, training, or evaluation has been started.")
        return 0
    return run_evaluation(args, identity, env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[eval] Interrupted. Owned subprocesses stopped; completed CSV rows can be resumed.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[eval] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
