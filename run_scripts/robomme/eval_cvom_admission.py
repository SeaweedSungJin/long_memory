#!/usr/bin/env python3
"""Evaluate writer-only CVoM on the fixed development VAL160 panel.

The immutable V19 parent supplies the representation, retrieval/fusion and AE
LoRA for memory, FIFO and READ-off. Only the writer callback differs. Original
HAMLET is strictly referenced at its completed run; source-incompatible old
V19 FIFO results are not reused. FIFO runs once here, and subsequent resumes
require the identical evaluation manifest and completed episode identities.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PARENT_CHECKPOINT = REPO_ROOT / "runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072"
sys.path.insert(0, str(REPO_ROOT))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--writer-checkpoint", type=Path)
    p.add_argument("--models", nargs="+", choices=("memory", "fifo", "memory-off"),
                   default=["memory", "fifo"])
    p.add_argument("--baseline-reference", type=Path,
                   default=Path("runs/eval/robomme/archive_read_best1250_val_n10_seed6"))
    p.add_argument("--base-model", type=Path,
                   default=Path("checkpoints/author_hamlet_robomme/checkpoint-60000"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--server-python", type=Path, default=REPO_ROOT / ".venv/bin/python")
    p.add_argument("--robomme-python", type=Path, default=Path("/home/sjkim/robomme_benchmark/.venv/bin/python"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--server-timeout", type=float, default=300)
    p.add_argument("--task-timeout", type=float, default=0)
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--allow-initialization-checkpoints", action="store_true",
                   help="Explicit diagnostic opt-in; never evidence for a trained writer")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    return p


def evaluator_arguments(args):
    if args.writer_checkpoint is None and not args.report_only:
        raise ValueError("--writer-checkpoint is required for evaluation/preflight")
    if len(set(args.models)) != len(args.models):
        raise ValueError("Candidate roles must be unique")
    argv = ["--cvom-admission", "--checkpoint", str(PARENT_CHECKPOINT), "--models", *args.models,
        "--baseline-reference", str(args.baseline_reference), "--base-model", str(args.base_model),
        "--tasks", "all", "--dataset", "val", "--n-episodes", "10", "--seed", "6",
        "--n-action-steps", "16", "--max-episode-steps", "1300", "--feature-precision", "native",
        "--output-dir", str(args.output_dir), "--server-python", str(args.server_python),
        "--robomme-python", str(args.robomme_python), "--device", args.device,
        "--server-timeout", str(args.server_timeout), "--task-timeout", str(args.task_timeout)]
    if args.writer_checkpoint is not None:
        argv += ["--writer-checkpoint", str(args.writer_checkpoint)]
    for name in ("save_videos", "allow_initialization_checkpoints", "preflight_only", "report_only"):
        if getattr(args, name):
            argv.append("--" + name.replace("_", "-"))
    return argv


def main(argv=None):
    args = build_parser().parse_args(argv)
    from run_scripts.robomme.eval_representation_v18 import main as evaluate
    return evaluate(evaluator_arguments(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[cvom-admission-eval] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
