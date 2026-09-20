#!/usr/bin/env python3
"""Paired V8 RoboMME evaluation of learned event encoding and retrieval.

Reader/memory-off share the exact V8 checkpoint. AE and short-source controls
are independently trained bundles. Every role uses the same full-demo cadence.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from run_scripts.robomme.eval_long_memory_comparison import (
    TASKS, benchmark_identity, bind_manifest, check_dependencies, file_hash,
    free_local_port, python_path, read_results, resolve_repo_path, run_client,
    server_ready, stop_process, validate_result_identity,
)
from gr00t.eval.sim.robomme.compare_long_memory_v8_results import write_v8_report as write_report
from gr00t.long_memory.safety_v5 import validate_output_scope

VARIANT = "event_memory_v8"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path,
                        default=Path("checkpoints/author_hamlet_robomme/checkpoint-60000"))
    parser.add_argument("--reader-checkpoint", type=Path, help="V8 Stage-1 moment-source event reader")
    parser.add_argument("--ae-checkpoint", type=Path, help="Separately trained V8 mode=none AE control")
    parser.add_argument("--short-checkpoint", type=Path, help="Separately trained V8 short-source event control")
    parser.add_argument("--models", nargs="+",
                        choices=("baseline", "reader", "memory-off", "ae-control", "short-control"),
                        default=["baseline", "reader", "memory-off"])
    parser.add_argument("--tasks", nargs="+", default=["BinFill", "PatternLock"], help="Unique task names or all")
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--dataset", choices=("train", "val", "test"), default="val")
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=1300)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/eval/robomme/event_memory_v8_smoke"))
    parser.add_argument("--server-python", type=Path, default=REPO_ROOT / ".venv/bin/python")
    parser.add_argument("--robomme-python", type=Path, default=Path("/home/sjkim/robomme_benchmark/.venv/bin/python"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--server-timeout", type=float, default=300)
    parser.add_argument("--task-timeout", type=float, default=0, help="Task wall time deadline; 0 disables")
    parser.add_argument("--preflight-only", action="store_true", help="Read-only validation, no output/model/rollout")
    parser.add_argument("--report-only", action="store_true", help="Regenerate saved paired report only")
    return parser


def validate_options(args):
    if args.tasks == ["all"]:
        args.tasks = list(TASKS)
    if not args.tasks or any(task not in TASKS for task in args.tasks) or len(set(args.tasks)) != len(args.tasks):
        raise ValueError("--tasks must be unique RoboMME task names or exactly all")
    if "baseline" not in args.models or len(set(args.models)) != len(args.models):
        raise ValueError("--models must be unique and include baseline")
    if any(name in args.models for name in ("reader", "memory-off")) and args.reader_checkpoint is None:
        raise ValueError("reader/memory-off requires --reader-checkpoint from V8 Stage 1")
    if "ae-control" in args.models and args.ae_checkpoint is None:
        raise ValueError("ae-control requires --ae-checkpoint from separately trained V8 mode=none")
    if "short-control" in args.models and args.short_checkpoint is None:
        raise ValueError("short-control requires --short-checkpoint from separately trained V8 source=short")
    if args.n_episodes < 1 or args.seed < 0 or args.n_action_steps < 1 or args.max_episode_steps < 1:
        raise ValueError("Episode/action/step counts must be positive and seed nonnegative")
    if not all(math.isfinite(value) for value in (args.server_timeout, args.task_timeout)):
        raise ValueError("Timeouts must be finite")
    if args.n_episodes > (100 if args.dataset == "train" else 50):
        raise ValueError("RoboMME supports at most 100 train or 50 val/test episodes per task")
    if args.server_timeout <= 0 or args.task_timeout < 0:
        raise ValueError("Server timeout must be positive and task timeout nonnegative")
    if args.preflight_only and args.report_only:
        raise ValueError("Choose preflight-only or report-only, not both")


def build_identity(args):
    """Read-only two-payload validation, source/capacity and paired identity."""
    from safetensors.torch import load_file
    from gr00t.long_memory.checkpoint_v8 import actor_state_sha256, v8_checkpoint_info
    from gr00t.long_memory.checkpoint_v4 import _state_sha256
    from gr00t.long_memory.hamlet import checkpoint_identity

    base = resolve_repo_path(args.base_model)
    base_identity = checkpoint_identity(base)
    config = json.loads((base / "config.json").read_text())
    if (config.get("hamlet_mode") != "finetune" or config.get("mem_cond_type", "cross_attn") != "cross_attn"
            or config.get("memory_type", "moment_token") != "moment_token"
            or int(config.get("n_moment_tokens", 0)) <= 0):
        raise ValueError("V8 requires trained HAMLET moment-token/cross-attention weights")
    if args.n_action_steps != int(config.get("memory_stride", 16)):
        raise ValueError("--n-action-steps must equal original HAMLET memory_stride")
    roles = {
        "reader": (args.reader_checkpoint, "event", "moment"),
        "memory-off": (args.reader_checkpoint, "event", "moment"),
        "ae-control": (args.ae_checkpoint, "none", None),
        "short-control": (args.short_checkpoint, "event", "short"),
    }
    descriptions = {
        "reader": "Stage-1 learned moment-event encoder/retriever + adapted AE; bounded chronological FIFO",
        "ae-control": "Separately trained AE LoRA without memory; distinct from memory-off inference",
        "short-control": "Separately trained short-source event encoder/retriever; check matched training budget",
        "memory-off": "Exact reader checkpoint and adapted AE; READ bypassed while event WRITE continues",
    }
    models, checked, cohorts = {}, {}, []
    inputs = [base]
    for name in args.models:
        entry = {"base_model": base_identity, "memory_checkpoint": None, "mode": "none",
                 "write_policy": "none", "memory_off": False, "source": None,
                 "description": "Original HAMLET; no added memory or adapted AE"}
        if name != "baseline":
            path, mode, source = roles[name]
            checkpoint = resolve_repo_path(path)
            if checkpoint not in checked:
                info = v8_checkpoint_info(base, checkpoint, expected_stage=1)
                hashes = {filename: file_hash(checkpoint / filename) for filename in
                          ("checkpoint.json", "model.safetensors", "expert.safetensors")}
                semantic = {
                    "actor": actor_state_sha256(load_file(str(checkpoint / "model.safetensors"), device="cpu")),
                    "expert": _state_sha256(load_file(str(checkpoint / "expert.safetensors"), device="cpu")),
                }
                checked[checkpoint] = info, hashes, semantic
                cache = info["metadata"].get("cache_dir") or info["config"].get("train", {}).get("cache_dir")
                inputs += [checkpoint, cache]
                if cache and (Path(cache) / "manifest.json").is_file():
                    inputs.append(json.loads((Path(cache) / "manifest.json").read_text()).get("dataset_path"))
            info, hashes, semantic = checked[checkpoint]
            cfg = info["config"]
            if cfg["stage"] != 1 or cfg["mode"] != mode:
                raise ValueError(f"Role {name} requires Stage 1 mode={mode}")
            if source is not None and cfg["memory"]["source"] != source:
                raise ValueError(f"Role {name} requires memory source={source}")
            architecture = {key: value for key, value in cfg["memory"].items() if key != "source"}
            cohorts.append((info["metadata"]["cache_fingerprint"], architecture, cfg["expert"], cfg["expert_targets"]))
            entry.update(memory_checkpoint=str(checkpoint), checkpoint_sha256=hashes["checkpoint.json"],
                         weights_sha256=hashes["model.safetensors"],
                         expert_weights_sha256=hashes["expert.safetensors"], semantic_state_sha256=semantic,
                         stage=1, step=info["step"], mode=mode, source=cfg["memory"]["source"],
                         memory_config=cfg["memory"], capacity_events=cfg["memory"]["capacity"],
                         capacity_tokens=cfg["memory"]["capacity"] * cfg["memory"]["num_short_tokens"],
                         expert_config=cfg["expert"], expert_targets=cfg["expert_targets"],
                         training_metadata=info["metadata"], trainer_variant=VARIANT,
                         training_contract={
                             "plan_sha256": info["metadata"].get("plan_sha256"),
                             "processed_queries": info["metadata"].get("train_state", {}).get("processed_queries"),
                             "seed": cfg.get("train", {}).get("seed"),
                             "expert_learning_rate": cfg.get("train", {}).get("expert_learning_rate"),
                             "initial_checkpoint": info["metadata"].get("initial_checkpoint"),
                         },
                         write_policy="append_fifo" if mode == "event" else "none",
                         memory_off=name == "memory-off", description=descriptions[name])
        models[name] = entry
    if cohorts and any(value != cohorts[0] for value in cohorts[1:]):
        raise ValueError("Compared V8 models must share training cache and architecture (apart from source)")
    validate_output_scope(resolve_repo_path(args.output_dir), *inputs)
    settings = {key: getattr(args, key) for key in ("tasks", "n_episodes", "dataset", "seed",
                "n_action_steps", "max_episode_steps", "save_videos", "device")}
    sources = set(path.relative_to(REPO_ROOT) for path in (REPO_ROOT / "gr00t").rglob("*.py"))
    sources.update(Path(name) for name in ("run_scripts/robomme/serve_long_memory_v8.py",
                   "run_scripts/robomme/eval_long_memory_v8.py", "run_scripts/robomme/eval_long_memory_comparison.py"))
    code = {str(path): file_hash(REPO_ROOT / path) for path in sorted(sources)}
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors", "numpy")}
    print("[preflight] hashing original HAMLET + V8 event memory + adapted AE ...", flush=True)
    base_files = {str(path.relative_to(base)): file_hash(path) for path in sorted(base.rglob("*"))
                  if path.is_file() and path.suffix in (".json", ".safetensors", ".model", ".txt")}
    identity = {"format_version": 1, "trainer_variant": VARIANT, "models": models, "settings": settings,
                "source_sha256": code, "base_file_sha256": base_files, "policy_package_versions": versions,
                "benchmark": benchmark_identity(args), "server_python": str(python_path(args.server_python)),
                "robomme_python": str(python_path(args.robomme_python)),
                "memory_input": "normalized pre-HAMLET moment tokens (short-control: post-HAMLET short), state/rawframe/demo",
                "storage_policy": "append observed events; evict oldest complete event on capacity overflow",
                "learned_storage_admission": False}
    identity["evaluation_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def server_command(args, model, port):
    command = [str(python_path(args.server_python)), "-u", str(REPO_ROOT / "run_scripts/robomme/serve_long_memory_v8.py"),
               "--base-model", model["base_model"]["path"], "--device", args.device,
               "--host", "127.0.0.1", "--port", str(port)]
    if model["memory_checkpoint"]:
        command += ["--memory-checkpoint", model["memory_checkpoint"]]
    if model["memory_off"]:
        command.append("--memory-off")
    return command


def run_evaluation(args, identity, env):
    """Own subprocess groups and journals; reuse tested low-level rollout helper."""
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
                    print(f"[v8-eval] resume: {name}/{task} already complete", flush=True)
            if not pending:
                continue
            folder = output / name
            folder.mkdir(exist_ok=True)
            port, server = free_local_port(), None
            command = server_command(args, model, port)
            with (folder / "server.log").open("a", encoding="utf-8") as log:
                try:
                    print(f"[v8-eval] loading {name}; {folder / 'server.log'}", flush=True)
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
                        print(f"[v8-eval] {name}/{task}; {task_dir / 'rollout.log'}", flush=True)
                        try:
                            run_client(client, env=env, log_path=task_dir / "rollout.log", server=server, timeout=args.task_timeout)
                            rows = read_results(task_dir / "simulation_results.csv", expected=args.n_episodes)
                            if len(rows) != args.n_episodes:
                                raise RuntimeError("Rollout exited without all requested completed episode rows")
                            validate_result_identity(output, name, task, identity)
                        except (RuntimeError, TimeoutError, ValueError) as exc:
                            failures.append({"model": name, "task": task, "error": str(exc)})
                            print(f"[v8-eval] ERROR {name}/{task}: {exc}", file=sys.stderr, flush=True)
                        write_report(output, bootstrap_samples=500)
                except (RuntimeError, TimeoutError) as exc:
                    failures.append({"model": name, "error": str(exc)})
                    print(f"[v8-eval] ERROR {name}: {exc}; see server.log", file=sys.stderr, flush=True)
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
            print(f"[v8-eval] Summary: {output / 'comparison_summary.txt'}", flush=True)
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
        if manifest.get("trainer_variant") != VARIANT:
            raise ValueError("Not a v8 evaluation; use the matching older report tool")
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
    print(f"[preflight] compatible v8 models={list(identity['models'])}; tasks={args.tasks}; n={args.n_episodes}", flush=True)
    if args.preflight_only:
        print("[preflight] No model/simulator loaded; no training/evaluation/output started.")
        return 0
    # Serialize even initial manifest creation outside the still-empty output.
    output = resolve_repo_path(args.output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ("." + output.name + ".v8-launch.lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another v8 evaluation owns this output directory") from exc
        return run_evaluation(args, identity, env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[v8-eval] Interrupted. Owned processes stopped; completed episodes remain resumable.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[v8-eval] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

