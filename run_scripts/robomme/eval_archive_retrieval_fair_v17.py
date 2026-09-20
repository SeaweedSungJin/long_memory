#!/usr/bin/env python3
"""Explicit, fixed-checkpoint V17 comparison phases; default only prints a plan.

No training, checkpoint selection, CSV copying, or automatic TEST launch.
Existing VAL results are referenced in place. TEST gets a new namespace whose
protocol locks the chosen checkpoint files and evaluator sources before rollout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_scripts.robomme.compare_archive_retrieval_fair_v17 import validate_checkpoint_family
from run_scripts.robomme.eval_long_memory_comparison import file_hash, python_path
from gr00t.long_memory.safety_v5 import validate_output_scope

PARENT = ROOT / "runs/long_memory/v7_archive_full_v1/checkpoint-001250"
CONTROL = ROOT / "runs/long_memory/v17_retrieval_pilot/control/checkpoint-000512"
GUIDED = ROOT / "runs/long_memory/v17_retrieval_pilot/guided/checkpoint-000512"
OLD_VAL = ROOT / "runs/eval/robomme/archive_read_best1250_val_n10_seed6"
PILOT = ROOT / "runs/eval/robomme/v17_retrieval_pilot"
OLD_TEST = ROOT / "runs/eval/robomme/v4_best_all_n50_seed6"
EVALUATOR = ROOT / "run_scripts/robomme/eval_archive_read_control_v7.py"
REPORTER = ROOT / "run_scripts/robomme/compare_archive_retrieval_fair_v17.py"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", nargs="?", default="plan", choices=("plan", "preflight", "eval-parent",
        "eval-control", "eval-guided", "eval-all", "report-val", "report-test"))
    p.add_argument("--test-root", type=Path, default=ROOT / "runs/eval/robomme/v17_retrieval_fair_test_n50_seed6")
    p.add_argument("--parent-checkpoint", type=Path, default=PARENT)
    p.add_argument("--control-checkpoint", type=Path, default=CONTROL)
    p.add_argument("--guided-checkpoint", type=Path, default=GUIDED)
    p.add_argument("--server-python", type=Path, default=ROOT / ".venv/bin/python")
    p.add_argument("--robomme-python", type=Path, default=Path("/home/sjkim/robomme_benchmark/.venv/bin/python"))
    p.add_argument("--historical", action="store_true", help="Add V4 historical contrast when reporting TEST only")
    p.add_argument("--report-output", type=Path, help="NEW report directory; otherwise print only")
    return p


def eval_command(args, arm, *, preflight=False):
    checkpoint = getattr(args, f"{arm}_checkpoint")
    roles = ["baseline", "archive"] + (["archive-off"] if arm == "guided" else [])
    cmd = [str(args.server_python), str(EVALUATOR), "--archive-checkpoint", str(checkpoint),
           "--models", *roles, "--tasks", "all", "--dataset", "test", "--n-episodes", "50",
           "--seed", "6", "--n-action-steps", "16", "--max-episode-steps", "1300", "--device", "cuda:0",
           "--server-python", str(args.server_python), "--robomme-python", str(args.robomme_python),
           "--output-dir", str(args.test_root / arm)]
    return cmd + (["--preflight-only"] if preflight else [])


def validate_test_root(args):
    root = args.test_root.resolve()
    eval_root = (ROOT / "runs/eval/robomme").resolve()
    # Never permit a broad root or one of the previous experimental namespaces.
    if root.parent != eval_root or not root.name.startswith("v17_retrieval_fair_"):
        raise ValueError("--test-root must be a direct NEW runs/eval/robomme/v17_retrieval_fair_* namespace")
    validate_output_scope(root, args.parent_checkpoint, args.control_checkpoint, args.guided_checkpoint,
                          ROOT / "checkpoints", ROOT / "data", ROOT / "runs/long_memory", PILOT, OLD_VAL, OLD_TEST)
    if root.exists() and not (root / "fair_protocol.json").is_file():
        raise ValueError("Existing test namespace lacks this workflow's immutable protocol; use a NEW namespace")


def protocol(args):
    sources = list((ROOT / "gr00t").rglob("*.py")) + [Path(__file__), REPORTER, EVALUATOR,
        ROOT / "run_scripts/robomme/serve_archive_read_control_v7.py",
        ROOT / "run_scripts/robomme/eval_long_memory_comparison.py",
        ROOT / "run_scripts/robomme/compare_archive_retrieval_v17.py",
        ROOT / "run_scripts/robomme/run_fair_comparison_v17.sh"]
    return {"format_version": 1, "primary": "action-only -> retrieval-guided", "dataset": "test",
        "tasks": "all16", "n_episodes": 50, "seed": 6, "n_action_steps": 16, "max_episode_steps": 1300,
        "checkpoints": validate_checkpoint_family(args.parent_checkpoint, args.control_checkpoint, args.guided_checkpoint),
        "commands": {arm: eval_command(args, arm) for arm in ("parent", "control", "guided")},
        "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(sources)},
        "note": "Fixed models chosen after exploratory VAL; TEST was examined in earlier versions and is not pristine. Do not choose checkpoints after viewing TEST."}


def bind_protocol(args, identity, *, write=False):
    target = args.test_root / "fair_protocol.json"
    if target.exists():
        if json.loads(target.read_text()) != identity:
            raise ValueError("Fair protocol/source/checkpoint changed: keep old results and use a NEW namespace")
    elif write:
        args.test_root.mkdir(parents=True, exist_ok=False)
        # Exclusive creation prevents accidentally overwriting another launch.
        with target.open("x") as handle:
            json.dump(identity, handle, indent=2)


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("test_root", "parent_checkpoint", "control_checkpoint", "guided_checkpoint"):
        setattr(args, name, getattr(args, name).resolve())
    # Preserve virtualenv symlinks: resolving python itself silently selects the
    # bare interpreter and loses packages installed in .venv.
    for name in ("server_python", "robomme_python"):
        setattr(args, name, python_path(getattr(args, name)))
    if args.historical and args.phase != "report-test":
        raise ValueError("--historical is only valid for report-test; VAL cannot be compared to old TEST")
    if args.phase.startswith("report-"):
        if args.phase == "report-val":
            runs = [OLD_VAL, PILOT / "val_control", PILOT / "val_guided"]
        else:
            validate_test_root(args)
            bind_protocol(args, protocol(args))
            runs = [args.test_root / arm for arm in ("parent", "control", "guided")]
        command = [str(args.server_python), str(REPORTER)]
        for arm, run in zip(("parent", "control", "guided"), runs):
            command += [f"--{arm}-run", str(run)]
        if args.historical:
            command += ["--historical-run", str(OLD_TEST)]
        if args.report_output:
            command += ["--output-dir", str(args.report_output.resolve())]
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    validate_test_root(args)
    if args.phase == "plan":
        print("Frozen parent=V7 step1250; control/guided=V17 fixed-final step512. No training required.")
        print("TEST all16 x 50 x (parent2 + control2 + guided3 roles) = 5600 rollouts.")
        print("Original baseline is repeated as an identity/reproducibility anchor, not pooled as independent samples.")
        print("Existing exploratory VAL is reused in place by report-val; no CSV copying.")
        for arm in ("parent", "control", "guided"):
            print(shlex.join(eval_command(args, arm)))
        print("Choose preflight next. eval-* phases are long and only run when explicitly requested.")
        return 0
    identity = protocol(args)
    bind_protocol(args, identity, write=False)
    if args.phase == "preflight":
        for arm in ("parent", "control", "guided"):
            subprocess.run(eval_command(args, arm, preflight=True), cwd=ROOT, check=True)
        print("Read-only preflight passed; no model/simulator loaded or output created.")
        return 0
    bind_protocol(args, identity, write=True)
    arms = ("parent", "control", "guided") if args.phase == "eval-all" else (args.phase.removeprefix("eval-"),)
    for arm in arms:
        # The unchanged evaluator verifies all identities and resumes its own
        # unfinished episodes; it never treats CSVs from another model as its own.
        subprocess.run(eval_command(args, arm), cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
