#!/usr/bin/env python3
"""One reproducible V8 training -> action audit -> small VAL rollout iteration.

Never declares the >=30% research goal achieved. Test evaluation is deliberately
not automatically started: inspect val evidence before fixing a final candidate.
All run/workflow/eval directories must be new; no checkpoints are overwritten.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.safety_v5 import validate_output_scope


def run_step(name, command, workflow):
    log = workflow / f"{name}.log"
    record = dict(phase=name, command=command, controller_pid=os.getpid(),
                  started_at=datetime.now(timezone.utc).isoformat(), log=str(log))
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        record.update(status="running", child_pid=process.pid)
        _atomic_json(workflow / "status.json", record)
        print(f"[v8-workflow] {name} PID={process.pid}; {log}", flush=True)
        try:
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f"[v8-workflow] {name} still running PID={process.pid}; {log}", flush=True)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            record.update(status="interrupted", returncode=process.returncode)
            _atomic_json(workflow / "status.json", record)
            raise
    record.update(status="complete" if process.returncode == 0 else "failed", returncode=process.returncode)
    _atomic_json(workflow / f"{name}.json", record)
    _atomic_json(workflow / "status.json", record)
    if process.returncode:
        raise RuntimeError(f"{name} exited {process.returncode}; inspect {log}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="runs/long_memory/v8_event_full_v1")
    p.add_argument("--eval-dir", default="runs/eval/robomme/v8_event_full_v1_val_n10_seed6")
    p.add_argument("--cache-dir", default="runs/long_memory/cache_full1600_v1")
    p.add_argument("--max-epochs", type=int, default=3)
    p.add_argument("--val-episodes", type=int, default=10)
    p.add_argument("--inference-seed", type=int, default=6)
    p.add_argument("--audit-samples", type=int, default=32)
    args = p.parse_args(argv)
    if min(args.max_epochs, args.val_episodes, args.audit_samples) < 1 or args.inference_seed < 0:
        raise ValueError("Invalid epoch/episode/sample count or seed")
    run, evaluation, cache = [(ROOT / path).resolve() for path in (args.run_dir, args.eval_dir, args.cache_dir)]
    workflow = run.with_name(run.name + "_workflow")
    if not run.is_relative_to(ROOT / "runs/long_memory") or not evaluation.is_relative_to(ROOT / "runs/eval/robomme"):
        raise ValueError("Use named runs/long_memory and runs/eval/robomme subdirectories")
    if run == ROOT / "runs/long_memory" or evaluation == ROOT / "runs/eval/robomme":
        raise ValueError("Cannot use a workspace experiment root as a run")
    if any(path.exists() for path in (run, evaluation, workflow)):
        raise FileExistsError("Use NEW run, eval and workflow paths; this launcher never restarts old runs")
    manifest = EpisodeCache(cache).manifest
    for path in (run, evaluation, workflow):
        validate_output_scope(path, cache, manifest.get("dataset_path"), manifest["model_path"])
    python = str(ROOT / ".venv/bin/python")
    training = [python, "run_scripts/robomme/train_long_memory_v8.py", "--stage", "1", "--mode", "event",
        "--source", "moment", "--cache-dir", str(cache), "--output-dir", str(run),
        "--max-epochs", str(args.max_epochs), "--queries-per-prefix", "2", "--query-batch-size", "8",
        "--capacity", "128", "--memory-learning-rate", "3e-5", "--expert-learning-rate", "3e-6",
        "--storage-reconstruction-weight", ".01", "--memory-dropout", ".1", "--residual-scale", ".1",
        "--val-memory-off", "--activation-checkpointing"]
    workflow.mkdir(parents=True, exist_ok=False)
    _atomic_json(workflow / "plan.json", {"args": vars(args), "training": training,
        "scope": "One training and val-screening iteration, not verified >=30% success", "test_started": False})
    run_step("train_preflight", training + ["--preflight-only"], workflow)
    run_step("train", training, workflow)
    status = json.loads((run / "status.json").read_text())
    if status.get("status") != "complete":
        raise RuntimeError("Training process exited without completing its planned horizon")
    pointers = {}
    for name in ("last", "best"):
        pointer = json.loads((run / f"{name}_checkpoint.json").read_text())
        path = Path(pointer["path"])
        path = path if path.is_absolute() else run / path
        # A step-zero best is recorded honestly, not called a trained memory.
        if name == "best" and int(pointer["step"]) == 0:
            continue
        if path.resolve() not in pointers.values():
            pointers[name] = path.resolve()
    evaluation.mkdir(parents=True, exist_ok=False)
    for name, checkpoint in pointers.items():
        audit = [python, "run_scripts/robomme/audit_long_memory_v8_actions.py", "--cache-dir", str(cache),
            "--checkpoint", str(checkpoint), "--samples", str(args.audit_samples), "--noise-samples", "2",
            "--output-dir", str(workflow / f"{name}_action_audit")]
        run_step(f"{name}_action_audit", audit, workflow)
        evaluate = [python, "run_scripts/robomme/eval_long_memory_v8.py", "--models", "baseline", "reader", "memory-off",
            "--reader-checkpoint", str(checkpoint), "--tasks", "all", "--dataset", "val",
            "--n-episodes", str(args.val_episodes), "--seed", str(args.inference_seed),
            "--output-dir", str(evaluation / name)]
        run_step(f"{name}_eval_preflight", evaluate + ["--preflight-only"], workflow)
        run_step(f"{name}_eval", evaluate, workflow)
    _atomic_json(workflow / "status.json", {"status": "iteration_complete", "goal_achieved": False,
        "note": "Inspect val paired success, then fix configuration before test. No >=30% claim.",
        "evaluations": {name: str(evaluation / name / "comparison_summary.txt") for name in pointers}})
    print(f"[v8-workflow] Training/val iteration finished. Results: {evaluation}; 30% goal NOT declared achieved.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
