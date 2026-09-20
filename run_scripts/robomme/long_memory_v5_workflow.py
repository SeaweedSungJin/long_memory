#!/usr/bin/env python3
"""Explicit, reviewable v5 steps; never silently chain research decisions.

Use the bash companion, or invoke this file directly. Each call runs exactly
one named step. ``--dry-run`` prints commands without creating directories.
Preflight flags check metadata only and do NOT run a smoke evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

STEPS = ("plan", "prepare", "probe", "stage1-preflight", "stage1-smoke", "stage1",
         "action-control", "expert-control", "stage2-preflight", "stage2-smoke", "stage2",
         "eval-reader-smoke", "eval-reader-val", "eval-preflight", "eval-smoke", "eval-val", "eval-test")


def _path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _positive(env, name, default):
    value = int(env.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return str(value)


def best_checkpoint(run, *, dry_run=False):
    pointer = run / "best_checkpoint.json"
    if dry_run and not pointer.is_file():
        return run / "<best-checkpoint-after-training>"
    if not pointer.is_file():
        raise FileNotFoundError(f"Finish/review the preceding stage first: {pointer}")
    data = json.loads(pointer.read_text())
    if not isinstance(data.get("path"), str):
        raise ValueError(f"Invalid best pointer: {pointer}")
    result = (run / data["path"]).resolve()
    if result.parent != run.resolve() or not (result / "checkpoint.json").is_file():
        raise ValueError(f"Best pointer escapes run or checkpoint is missing: {pointer}")
    info = json.loads((result / "checkpoint.json").read_text())
    if (type(info.get("step")) is not int or type(data.get("step")) is not int
            or info["step"] < 0 or info["step"] != data["step"]):
        raise ValueError(f"Best pointer step mismatch: {pointer}")
    return result


def build_command(step, env=None, *, dry_run=False):
    env = dict(os.environ if env is None else env)
    run = _path(env.get("V5_RUN_DIR", "runs/long_memory/v5_recall_full_v1"))
    cache = _path(env.get("V5_CACHE_DIR", "runs/long_memory/cache_full1600_v1"))
    labels = _path(env.get("V5_LABEL_DIR", "runs/long_memory/recall_labels_full1600_v5"))
    initial = _path(env.get("V5_INIT_CHECKPOINT", "runs/long_memory/v4_ae_stage1_full/checkpoint-003600"))
    probe_checkpoint = _path(env.get("V5_PROBE_CHECKPOINT", "runs/long_memory/v4_writer_stage2_full/checkpoint-000900"))
    python = str(ROOT / ".venv/bin/python")
    script = lambda name: [python, str(ROOT / "run_scripts/robomme" / name)]
    s1, s2 = run / "stage1", run / "stage2"
    if step == "prepare":
        return script("prepare_recall_labels_v5.py") + ["--cache-dir", str(cache), "--output-dir", str(labels)]
    if step == "probe":
        return script("probe_long_memory_v5.py") + ["--cache-dir", str(cache), "--labels-dir", str(labels),
            "--checkpoint", str(probe_checkpoint), "--output-dir", str(run / "probe_v4"),
            "--samples-per-split", _positive(env, "V5_PROBE_SAMPLES", 512),
            "--epochs", _positive(env, "V5_PROBE_EPOCHS", 100), "--device", "cpu"]
    if step.startswith("stage1") or step in ("action-control", "expert-control"):
        name = "stage1_smoke" if step == "stage1-smoke" else step.replace("-", "_") if step.endswith("control") else "stage1"
        command = script("train_long_memory_v5.py") + ["--stage", "1", "--cache-dir", str(cache),
            "--recall-labels", str(labels), "--init-checkpoint", str(initial), "--output-dir", str(run / name),
            "--max-steps", _positive(env, "V5_STAGE1_STEPS", 2000), "--grad-accum", "4",
            "--reader-learning-rate", "1e-5", "--expert-learning-rate", "1e-5",
            "--recall-learning-rate", "1e-4", "--subgoal-weight", "0.001", "--grounding-weight", "0.01",
            "--delayed-fraction", "0.5", "--val-samples", "64", "--eval-steps", "100",
            "--save-steps", "250", "--plot-steps", "100", "--activation-checkpointing"]
        if step.endswith("control"):
            command += ["--subgoal-weight", "0", "--grounding-weight", "0"]
            if step == "expert-control":
                command += ["--reader-mode", "none"]
        if step == "stage1-preflight":
            command.append("--preflight-only")
        if step == "stage1-smoke":
            command += ["--max-steps", "10", "--grad-accum", "1", "--max-train-episodes", "8",
                        "--max-val-episodes", "4", "--val-samples", "2", "--eval-steps", "5",
                        "--save-steps", "5", "--plot-steps", "5", "--log-steps", "1"]
        return command
    reader = _path(env["V5_READER_CHECKPOINT"]) if env.get("V5_READER_CHECKPOINT") else best_checkpoint(s1, dry_run=dry_run)
    if step.startswith("stage2"):
        command = script("train_long_memory_v5.py") + ["--stage", "2", "--cache-dir", str(cache),
            "--recall-labels", str(labels), "--init-checkpoint", str(reader),
            "--output-dir", str(run / ("stage2_smoke" if step == "stage2-smoke" else "stage2")),
            "--max-steps", _positive(env, "V5_STAGE2_STEPS", 1000), "--writer-learning-rate", "1e-4",
            "--writer-batch-size", "4", "--storage-contexts", "128", "--context-refresh-steps", "250",
            "--future-samples", "2", "--noise-samples", "4", "--val-samples", "64",
            "--val-storage-samples", "8", "--eval-steps", "100", "--save-steps", "250", "--plot-steps", "100"]
        if step == "stage2-preflight":
            command.append("--preflight-only")
        if step == "stage2-smoke":
            command += ["--max-steps", "4", "--writer-batch-size", "1", "--storage-contexts", "4",
                        "--context-refresh-steps", "2", "--noise-samples", "2", "--val-samples", "2",
                        "--val-storage-samples", "1", "--eval-steps", "2", "--save-steps", "2",
                        "--plot-steps", "2", "--log-steps", "1"]
        return command
    if not step.startswith("eval-"):
        raise ValueError(f"Unknown workflow step: {step}")
    only_reader = step.startswith("eval-reader-")
    command = script("eval_long_memory_v5.py") + ["--models", "baseline", "reader"]
    if only_reader:
        command += ["expert-only", "--reader-checkpoint", str(reader)]
    else:
        memory = _path(env["V5_MEMORY_CHECKPOINT"]) if env.get("V5_MEMORY_CHECKPOINT") else best_checkpoint(s2, dry_run=dry_run)
        command += ["memory", "fifo", "--reader-checkpoint", str(reader), "--memory-checkpoint", str(memory)]
    small = step.endswith("smoke") or step == "eval-preflight"
    dataset = "test" if step == "eval-test" else "val"
    command += ["--tasks", *( ["BinFill", "PatternLock"] if small else ["all"] ),
                "--n-episodes", "3" if small else _positive(env, "V5_EVAL_EPISODES", 50),
                "--dataset", dataset, "--seed", str(int(env.get("V5_EVAL_SEED", 6))),
                "--output-dir", str(run / step.replace("-", "_"))]
    if step == "eval-preflight":
        command.append("--preflight-only")
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=STEPS, nargs="?", default="plan")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.step == "plan":
        print("prepare -> probe [review] -> stage1-preflight -> stage1-smoke -> stage1")
        print("-> eval-reader-smoke -> eval-reader-val [review] -> stage2-preflight -> stage2-smoke")
        print("-> stage2 -> eval-preflight -> eval-smoke -> eval-val [freeze experiment] -> eval-test")
        print("Controls: action-control (same sampling/init, no auxiliary loss), expert-control (AE-only).")
        print("Set V5_RUN_DIR for a fresh experiment. Existing runs/checkpoints are never overwritten.")
        print("Guide: docs/LONG_MEMORY_V5_RECALL_CONTINUATION.md")
        return 0
    command = build_command(args.step, dry_run=args.dry_run)
    print("[v5-workflow] " + shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if args.step == "prepare":
        from gr00t.long_memory.cache import EpisodeCache
        from gr00t.long_memory.recall_data_v5 import RecallLabels
        labels = Path(command[command.index("--output-dir") + 1])
        if (labels / "manifest.json").is_file():
            cache = EpisodeCache(command[command.index("--cache-dir") + 1])
            targets = RecallLabels(labels, cache)
            # Loading all small target files also detects an edited payload.
            for record in cache.manifest["episodes"]:
                targets.get(int(record["episode_id"]), 0)
            print(f"[v5-workflow] Existing immutable labels validated: {targets.manifest['fingerprint']}")
            return 0
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"[v5-workflow] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
