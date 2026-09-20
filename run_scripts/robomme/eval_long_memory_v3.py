#!/usr/bin/env python3
"""Paired RoboMME evaluation of HAMLET / v3 reader / learned replacement.

This is a separate entrypoint: no legacy checkpoint or evaluation is rewritten.
The optional fifo role uses the SAME Stage 2 weights with unconditional FIFO
storage, isolating storage-policy effects from changes to the trained reader.
Completed episodes resume only under identical checkpoint, code and simulator
identities. Failures in the program remain incomplete, never 0%-success rows.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Deliberate dependency on tested process ownership, resume accounting and
# paired statistics. Their sources are included in every v3 manifest below.
from run_scripts.robomme.eval_long_memory_comparison import (
    TASKS, benchmark_identity, bind_manifest, check_dependencies, file_hash,
    free_local_port, python_path, read_results, resolve_repo_path, run_client,
    server_ready, stop_process, validate_result_identity,
)
from gr00t.eval.sim.robomme.compare_long_memory_v3_results import write_v3_report as write_report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path,
                        default=Path("checkpoints/author_hamlet_robomme/checkpoint-60000"))
    parser.add_argument("--reader-checkpoint", type=Path, help="New v3 Stage 1 checkpoint, NOT legacy Stage 1")
    parser.add_argument("--memory-checkpoint", type=Path, help="New v3 Stage 2 checkpoint, NOT legacy Stage 2")
    parser.add_argument("--models", nargs="+", choices=("baseline", "reader", "memory", "fifo"),
                        default=["baseline", "reader", "memory"])
    parser.add_argument("--tasks", nargs="+", default=["BinFill", "PatternLock"], help="Unique task names or all")
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--dataset", choices=("train", "val", "test"), default="test")
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=1300)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/eval/robomme/action_value_v3_smoke"))
    parser.add_argument("--server-python", type=Path, default=REPO_ROOT / ".venv/bin/python")
    parser.add_argument("--robomme-python", type=Path,
                        default=Path("/home/sjkim/robomme_benchmark/.venv/bin/python"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--server-timeout", type=float, default=300)
    parser.add_argument("--task-timeout", type=float, default=0,
                        help="Task wall-clock deadline in seconds; 0 disables")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Read-only checkpoint/dependency validation; no model, simulator or output directory")
    parser.add_argument("--report-only", action="store_true", help="Regenerate saved comparison_summary.txt/json only")
    return parser


def validate_options(args):
    if args.tasks == ["all"]:
        args.tasks = list(TASKS)
    if not args.tasks or any(task not in TASKS for task in args.tasks) or len(set(args.tasks)) != len(args.tasks):
        raise ValueError("--tasks must be unique RoboMME task names or exactly all")
    if "baseline" not in args.models or len(set(args.models)) != len(args.models):
        raise ValueError("--models must be unique and include baseline")
    if "reader" in args.models and args.reader_checkpoint is None:
        raise ValueError("reader requires --reader-checkpoint from v3 Stage 1")
    if any(name in args.models for name in ("memory", "fifo")) and args.memory_checkpoint is None:
        raise ValueError("memory/fifo requires --memory-checkpoint from v3 Stage 2")
    if args.n_episodes < 1 or args.seed < 0 or args.n_action_steps < 1 or args.max_episode_steps < 1:
        raise ValueError("Episode/action/step counts must be positive and seed nonnegative")
    if args.server_timeout <= 0 or args.task_timeout < 0:
        raise ValueError("Server timeout must be positive and task timeout nonnegative")
    if args.preflight_only and args.report_only:
        raise ValueError("Choose preflight-only or report-only, not both")


def build_identity(args):
    """Strict read-only preflight, including new AND reused implementation files."""
    from gr00t.long_memory.checkpoint_v3 import VARIANT, memory_v3_checkpoint_info
    from gr00t.long_memory.hamlet import checkpoint_identity

    base = resolve_repo_path(args.base_model)
    base_identity = checkpoint_identity(base)
    config = json.loads((base / "config.json").read_text())
    if (config.get("hamlet_mode") != "finetune" or config.get("mem_cond_type", "cross_attn") != "cross_attn"
            or config.get("memory_type", "moment_token") != "moment_token"
            or int(config.get("n_moment_tokens", 0)) <= 0):
        raise ValueError("V3 evaluation requires trained HAMLET moment-token/cross-attention weights")
    if args.n_action_steps != int(config.get("memory_stride", 16)):
        raise ValueError("--n-action-steps must equal the base checkpoint memory_stride")
    models, checked, cohorts = {}, {}, []
    for name in args.models:
        entry = {"base_model": base_identity, "memory_checkpoint": None, "write_policy": "none",
                 "write_policy_override": "checkpoint"}
        if name != "baseline":
            checkpoint = resolve_repo_path(args.reader_checkpoint if name == "reader" else args.memory_checkpoint)
            if checkpoint not in checked:
                expected_stage = 1 if name == "reader" else 2
                info = memory_v3_checkpoint_info(base, checkpoint, expected_stage=expected_stage)
                checked[checkpoint] = (info, file_hash(checkpoint / "checkpoint.json"),
                                       file_hash(checkpoint / "model.safetensors"))
            info, metadata_hash, weights_hash = checked[checkpoint]
            if info["config"]["stage"] != (1 if name == "reader" else 2):
                raise ValueError(f"Wrong checkpoint stage for model role {name}")
            cohorts.append((info["metadata"]["cache_fingerprint"], info["config"]["memory"]))
            entry.update(memory_checkpoint=str(checkpoint), checkpoint_sha256=metadata_hash,
                         weights_sha256=weights_hash, stage=info["config"]["stage"], step=info["step"],
                         memory_config=info["config"]["memory"], training_metadata=info["metadata"],
                         trainer_variant=VARIANT, write_policy="hard" if name == "memory" else "all",
                         write_policy_override="all" if name == "fifo" else "checkpoint")
        models[name] = entry
    if cohorts and any(value != cohorts[0] for value in cohorts[1:]):
        raise ValueError("Reader and memory must share the training cache and memory architecture/config")

    settings = {key: getattr(args, key) for key in ("tasks", "n_episodes", "dataset", "seed",
                "n_action_steps", "max_episode_steps", "save_videos", "device")}
    # Hash the whole gr00t Python implementation, including indirect processor,
    # normalizer and legacy adapter dependencies. Missing entrypoints fail closed.
    sources = set(path.relative_to(REPO_ROOT) for path in (REPO_ROOT / "gr00t").rglob("*.py"))
    sources.update(Path(name) for name in (
        "run_scripts/robomme/serve_long_memory_v3.py", "run_scripts/robomme/eval_long_memory_v3.py",
        "run_scripts/robomme/eval_long_memory_comparison.py"))
    for name in ("gr00t/long_memory/core_v3.py", "gr00t/long_memory/online_v3.py",
                 "gr00t/long_memory/online_policy_v3.py", "gr00t/long_memory/checkpoint_v3.py",
                 "gr00t/eval/sim/robomme/run_long_memory_rollout.py"):
        sources.add(Path(name))
    code = {str(path): file_hash(REPO_ROOT / path) for path in sorted(sources)}
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "safetensors", "numpy")}
    print("[preflight] hashing frozen checkpoint files for immutable evaluation provenance ...", flush=True)
    base_files = {str(path.relative_to(base)): file_hash(path) for path in sorted(base.rglob("*"))
                  if path.is_file() and path.suffix in (".json", ".safetensors", ".model", ".txt")}
    identity = {"format_version": 1, "trainer_variant": VARIANT, "models": models, "settings": settings,
                "source_sha256": code, "base_file_sha256": base_files, "policy_package_versions": versions,
                "benchmark": benchmark_identity(args), "server_python": str(python_path(args.server_python)),
                "robomme_python": str(python_path(args.robomme_python))}
    identity["evaluation_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def server_command(args, model, port):
    command = [str(python_path(args.server_python)), "-u", str(REPO_ROOT / "run_scripts/robomme/serve_long_memory_v3.py"),
               "--base-model", model["base_model"]["path"], "--device", args.device,
               "--host", "127.0.0.1", "--port", str(port)]
    if model["memory_checkpoint"]:
        command += ["--memory-checkpoint", model["memory_checkpoint"],
                    "--write-policy", model["write_policy_override"]]
    return command


def run_evaluation(args, identity, env):
    output = resolve_repo_path(args.output_dir)
    bind_manifest(output, identity)
    lock = (output / ".driver.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f"Another evaluation driver owns {output}")
    failures, interrupted = [], False
    try:
        for name, model in identity["models"].items():
            pending = []
            for task in args.tasks:
                rows = read_results(output / name / task / "simulation_results.csv", expected=args.n_episodes)
                if rows:
                    validate_result_identity(output, name, task, identity)
                if len(rows) != args.n_episodes:
                    pending.append(task)
                else:
                    print(f"[v3-eval] resume: {name}/{task} already complete", flush=True)
            if not pending:
                continue
            folder = output / name
            folder.mkdir(exist_ok=True)
            port, server = free_local_port(), None
            command = server_command(args, model, port)
            with (folder / "server.log").open("a", encoding="utf-8") as log:
                try:
                    print(f"[v3-eval] loading {name}; {folder / 'server.log'}", flush=True)
                    log.write("\n[driver] command: " + json.dumps(command) + "\n")
                    log.flush()
                    server = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
                    server_ready(server, port, args.server_timeout)
                    for task in pending:
                        task_dir = folder / task
                        task_dir.mkdir(exist_ok=True)
                        client = [str(python_path(args.robomme_python)), "-u",
                                  str(REPO_ROOT / "gr00t/eval/sim/robomme/run_long_memory_rollout.py"),
                                  "--task-id", task, "--policy-client-host", "127.0.0.1", "--policy-client-port", str(port),
                                  "--dataset", args.dataset, "--n-episodes", str(args.n_episodes),
                                  "--max-episode-steps", str(args.max_episode_steps), "--n-action-steps", str(args.n_action_steps),
                                  "--model-config", model["base_model"]["path"], "--output-dir", str(task_dir),
                                  "--seed", str(args.seed), "--evaluation-id", identity["evaluation_id"] + ":" + name]
                        if args.save_videos:
                            client.append("--save-videos")
                        print(f"[v3-eval] {name}/{task}; {task_dir / 'rollout.log'}", flush=True)
                        try:
                            run_client(client, env=env, log_path=task_dir / "rollout.log", server=server, timeout=args.task_timeout)
                            rows = read_results(task_dir / "simulation_results.csv", expected=args.n_episodes)
                            if len(rows) != args.n_episodes:
                                raise RuntimeError("Rollout exited without all requested completed episode rows")
                            validate_result_identity(output, name, task, identity)
                        except (RuntimeError, TimeoutError, ValueError) as exc:
                            failures.append({"model": name, "task": task, "error": str(exc)})
                            print(f"[v3-eval] ERROR {name}/{task}: {exc}", file=sys.stderr, flush=True)
                        write_report(output, bootstrap_samples=500)
                except (RuntimeError, TimeoutError) as exc:
                    failures.append({"model": name, "error": str(exc)})
                    print(f"[v3-eval] ERROR {name}: {exc}; see server.log", file=sys.stderr, flush=True)
                finally:
                    stop_process(server)
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        try:
            (output / "driver_status.json").write_text(json.dumps({"interrupted": interrupted, "failures": failures}, indent=2) + "\n")
            result, report = write_report(output)
            print(report, flush=True)
            print(f"[v3-eval] Summary: {output / 'comparison_summary.txt'}", flush=True)
        finally:
            lock.close()
    return int(bool(failures) or not all(model["complete"] for model in result["models"].values()))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.report_only:
        if args.preflight_only:
            raise ValueError("Choose report-only or preflight-only, not both")
        output = resolve_repo_path(args.output_dir)
        manifest = json.loads((output / "comparison_manifest.json").read_text())
        if manifest.get("trainer_variant") != "action_value_v3":
            raise ValueError("Not a v3 evaluation; use the legacy report tool for older runs")
        # Do not race the driver's live summary writer. Reading the existing
        # summary.txt is always safe while evaluation is still running.
        with (output / ".driver.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Evaluation is running; read its live comparison_summary.txt instead")
            _, report = write_report(output)
        print(report)
        return 0
    validate_options(args)
    env = dict(os.environ)
    env.update(PYTHONPATH=str(REPO_ROOT), PYTHONUNBUFFERED="1", GR00T_INFERENCE_SEED=str(args.seed))
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    server_python = python_path(args.server_python)
    if python_path(Path(sys.executable)) != server_python:
        return subprocess.call([str(server_python), str(Path(__file__).resolve()),
                                *(sys.argv[1:] if argv is None else argv)], cwd=REPO_ROOT, env=env)
    check_dependencies(server_python, "import torch, cv2, transformers, safetensors, msgpack, zmq", env, "policy")
    check_dependencies(python_path(args.robomme_python), "import robomme, msgpack, zmq, pandas, imageio", env, "simulator")
    identity = build_identity(args)
    print(f"[preflight] compatible v3 models={list(identity['models'])}; tasks={args.tasks}; n={args.n_episodes}", flush=True)
    if args.preflight_only:
        print("[preflight] No model/simulator loaded; no training/evaluation/output started.")
        return 0
    return run_evaluation(args, identity, env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[v3-eval] Interrupted. Owned processes stopped; completed episodes remain resumable.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[v3-eval] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
