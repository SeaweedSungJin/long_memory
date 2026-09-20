#!/usr/bin/env python3
"""Read-only, matched full-coverage/tail-weight experiment report.

Uses the unchanged V18 policy evaluator and its complete runtime diagnostics.
The full-chunk control and execution-prefix candidate must differ only in tail
loss weight. It never selects a checkpoint based on rollout or validation score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_scripts.robomme.compare_representation_v18 import load_completed
from run_scripts.robomme.compare_archive_retrieval_v17 import compatible, contrast
from run_scripts.robomme.checkpoint_representation_v18 import checkpoint_info_v18
from run_scripts.robomme.train_archive_deployment_v9 import digest, file_hash

DRIVER = "full_memory_v19"


def validate_final_info(info):
    """Reject intermediate/smoke checkpoints even when otherwise deployable."""
    cfg, step = info["config"], info["step"]
    if cfg.get("driver_variant") != DRIVER:
        raise ValueError("Expected actual full_memory_v19 training, not a renamed V18 pilot")
    train = cfg["train"]
    if type(train.get("epochs")) is not int or train["epochs"] < 1:
        raise ValueError("Missing positive full-epoch training budget")
    if type(step) is not int or step <= 0 or step != train.get("max_steps"):
        raise ValueError("Use the fixed-final full-epoch checkpoint, not an intermediate/best checkpoint")
    if train.get("stop_after_steps") is not None and train["stop_after_steps"] < step:
        raise ValueError("A paused smoke run is not a completed full-epoch experiment")
    rep = cfg["representation"]
    if rep.get("representation") != "short" or rep.get("gate") != "linear" or rep.get("capacity_events") != 32:
        raise ValueError("V19 loss comparison preserves V18 A: short/linear/FIFO32")
    coverage = info["metadata"].get("full_coverage", {})
    count = coverage.get("train_query_count")
    if (type(count) is not int or count < 1 or coverage.get("epochs") != train["epochs"]
            or coverage.get("total_query_presentations") != count * train["epochs"]
            or coverage.get("total_steps") != step):
        raise ValueError("Missing or inconsistent full-TRAIN-query coverage provenance")
    return train


def final_checkpoint(run):
    """Resolve last published checkpoint only after full training completed."""
    root = Path(run).resolve()
    status = json.loads((root / "status.json").read_text())
    last = json.loads((root / "last_checkpoint.json").read_text())
    if status.get("status") != "complete":
        raise ValueError("Training is not complete; paused/running/smoke outputs cannot be the final comparison")
    step = last.get("step")
    if type(step) is not int or step < 1 or last.get("path") != f"checkpoint-{step:06d}":
        raise ValueError("Invalid immutable last_checkpoint.json identity")
    checkpoint = (root / last["path"]).resolve()
    if checkpoint.parent != root:
        raise ValueError("Checkpoint path escapes its training run")
    info = json.loads((checkpoint / "checkpoint.json").read_text())
    train = validate_final_info(info)
    if (step != info["step"] or status.get("step") != step or status.get("max_steps") != step
            or status.get("completed_epochs") != train["epochs"]
            or status.get("processed_queries") != info["metadata"]["full_coverage"]["total_query_presentations"]):
        raise ValueError("Full-epoch status and last checkpoint disagree")
    recorded_plan = json.loads((root / "query_plan.json").read_text())
    plan_hash = recorded_plan.pop("sha256")
    if digest(recorded_plan) != plan_hash or plan_hash != info["metadata"].get("plan_sha256"):
        raise ValueError("Full-coverage query plan identity changed")
    for key, expected in info["metadata"]["full_coverage"].items():
        if recorded_plan.get(key) != expected:
            raise ValueError(f"Full-coverage plan and checkpoint disagree: {key}")
    from run_scripts.robomme.training_plan_v19 import assert_complete_epochs_v19
    assert_complete_epochs_v19(recorded_plan)
    if (root / "failure.json").exists():
        raise ValueError("Run has failure.json; inspect it before treating training as complete")
    return checkpoint


def _info_for_run(run):
    model = run["manifest"]["models"]["memory"]
    if model.get("writer_checkpoint"):
        raise ValueError("CVOM is a separate experiment; this comparison requires unchanged FIFO")
    checkpoint = Path(model["memory_checkpoint"]).resolve()
    if final_checkpoint(checkpoint.parent) != checkpoint:
        raise ValueError("Evaluated checkpoint is not this run's fixed final epoch")
    for filename, expected in model["checkpoint_files_sha256"].items():
        if file_hash(checkpoint / filename) != expected:
            raise ValueError("Evaluated checkpoint was modified")
    info = checkpoint_info_v18(model["base_model"]["path"], checkpoint)
    validate_final_info(info)
    return info


def matched_training(control, prefix):
    """Return independently recorded training details after fail-closed checks."""
    a, b = _info_for_run(control), _info_for_run(prefix)
    for key in ("base_model", "cache_fingerprint", "plan_sha256", "source_sha256", "runtime",
                "initialization", "initial_shared_reader_sha256", "initial_expert_sha256", "full_coverage"):
        if not a["metadata"].get(key) or a["metadata"].get(key) != b["metadata"].get(key):
            raise ValueError(f"Unmatched full-coverage training: {key}")
    if a["step"] != b["step"]:
        raise ValueError("Training budgets differ")
    ca, cb = a["config"], b["config"]
    for key in ca.keys() | cb.keys():
        if key not in {"train", "objective"} and ca.get(key) != cb.get(key):
            raise ValueError(f"More than tail_weight changed: {key}")
    # Exact V19 resume retains the original epoch horizon, query/noise plan,
    # initialization, source/runtime and optimizer/RNG. Its parent/output path
    # can differ between arms without becoming a new experimental factor.
    allowed = {"output_dir", "resume", "preflight_only", "stop_after_steps", "tail_weight"}
    for key in ca["train"].keys() | cb["train"].keys():
        if key not in allowed and ca["train"].get(key) != cb["train"].get(key):
            raise ValueError(f"More than tail_weight changed: train/{key}")
    for key in ca["objective"].keys() | cb["objective"].keys():
        if key != "tail_weight" and ca["objective"].get(key) != cb["objective"].get(key):
            raise ValueError(f"More than tail_weight changed: objective/{key}")
    for cfg, expected in ((ca, 1.), (cb, .25)):
        if cfg["train"].get("tail_weight") != expected or cfg["objective"].get("tail_weight") != expected:
            raise ValueError("Expected full-chunk control tail=1 -> execution-prefix candidate tail=0.25")
    return {"epochs": ca["train"]["epochs"], "updates_per_arm": a["step"],
            "plan_sha256": a["metadata"]["plan_sha256"],
            "tail_weights": {"control": 1., "prefix": .25}}


def report(control_dir, prefix_dir):
    control, prefix = load_completed(control_dir), load_completed(prefix_dir)
    compatible(control, prefix)
    if control["manifest"]["source_sha256"] != prefix["manifest"]["source_sha256"]:
        raise ValueError("V18 evaluation source closure differs")
    training = matched_training(control, prefix)
    comparisons = {
        "full-chunk control -> execution-prefix": contrast(control, "memory", prefix, "memory"),
        "HAMLET -> full-chunk control": contrast(control, "baseline", control, "memory"),
        "HAMLET -> execution-prefix": contrast(prefix, "baseline", prefix, "memory"),
    }
    result = {"driver_variant": DRIVER, "settings": control["manifest"]["settings"],
              "training": training, "comparisons": comparisons,
              "control_run": str(Path(control_dir).resolve()), "prefix_run": str(Path(prefix_dir).resolve())}
    lines = ["V19: matched full-TRAIN-coverage experiment; closed-loop task success",
             f"Each arm: {training['epochs']} full epoch(s), {training['updates_per_arm']} updates.",
             "Only tail loss weight differs; original HAMLET baseline is not retrained."]
    for name, row in comparisons.items():
        lo, hi = row["ci95"]
        lines.append(f"{name}: {100*row['left_macro']:.3f}% -> {100*row['right_macro']:.3f}% "
                     f"({100*row['delta']:+.3f}pp); 95% paired CI [{100*lo:+.3f}, {100*hi:+.3f}]pp; "
                     f"wins/losses={row['wins']}/{row['losses']}; McNemar p={row['mcnemar_p']:.6g}")
    primary = comparisons["full-chunk control -> execution-prefix"]
    lines += ["", "Task                   Control/Prefix success   Wins/Losses"]
    for task, row in primary["tasks"].items():
        lines.append(f"{task:22} {row['left_successes']:3}/{row['n']} -> {row['right_successes']:3}/{row['n']}       {row['wins']}/{row['losses']}")
    lines += ["", "Full TRAIN coverage refers to all cached training endpoints, not every raw video frame; held-out VAL is not trained on.",
              "VAL160 is a repeated-development screen, not proof of TEST improvement.",
              "One success changes VAL160 by 0.625 percentage points; intervals condition on these tasks/scenarios/seed.",
              "Repeated model selection adds optimism. Check SAME-model READ-off before a fixed final larger TEST.",
              "This loss ablation does not by itself prove better memory storage/retrieval."]
    return result, "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run")
    parser.add_argument("--prefix-run")
    parser.add_argument("--checkpoint-run", help="Print complete fixed-final checkpoint path; no model loaded")
    parser.add_argument("--output-dir", help="Optional NEW report directory; otherwise stdout only")
    args = parser.parse_args(argv)
    if args.checkpoint_run:
        if args.control_run or args.prefix_run or args.output_dir:
            parser.error("--checkpoint-run cannot be combined with report options")
        print(final_checkpoint(args.checkpoint_run))
        return 0
    if not args.control_run or not args.prefix_run:
        parser.error("Provide --control-run and --prefix-run, or only --checkpoint-run")
    output = None
    if args.output_dir:
        from gr00t.long_memory.safety_v5 import validate_output_scope
        output = validate_output_scope(args.output_dir, args.control_run, args.prefix_run,
            ROOT / "checkpoints", ROOT / "data", ROOT / "runs/long_memory")
        if output.exists():
            raise FileExistsError("Use a NEW report directory")
    result, text = report(args.control_run, args.prefix_run)
    if output:
        output.mkdir(parents=True, exist_ok=False)
        (output / "comparison_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        (output / "comparison_summary.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
