#!/usr/bin/env python3
"""Strict, read-only matched A/B, A/C or Linear/MLP comparison.

Original baseline CSVs stay in their original run. Reports reject changed
training plans, different initialization, incomplete runs, and multiple changed
experimental factors. A positive small-VAL estimate is not a verified gain.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_scripts.robomme.compare_archive_retrieval_v17 import load_run, compatible, contrast
from run_scripts.robomme.eval_representation_v18 import (validate_manifest_contract,
    completed_read_diagnostics, verify_runtime_inputs)
from run_scripts.robomme.baseline_reference_v18 import validate_reference
from run_scripts.robomme.checkpoint_representation_v18 import checkpoint_info_v18
from run_scripts.robomme.train_archive_deployment_v9 import file_hash


def load_completed(root, role="memory"):
    root = Path(root).resolve()
    manifest = json.loads((root / "comparison_manifest.json").read_text())
    validate_manifest_contract(manifest)
    verify_runtime_inputs(manifest)
    status = json.loads((root / "driver_status.json").read_text())
    if status.get("interrupted") or status.get("failures"):
        raise ValueError("Evaluation driver was interrupted or reported failed tasks")
    candidate = load_run(root, (role,))
    if manifest.get("baseline_reference"):
        ref, _ = validate_reference(manifest["baseline_reference"], manifest)
        baseline = load_run(ref, ("baseline",))
    else:
        baseline = load_run(root, ("baseline",))
    for key in ("rows", "identities"):
        candidate[key]["baseline"] = baseline[key]["baseline"]
    diagnostics = completed_read_diagnostics(root, role, manifest)
    if not diagnostics["complete_evidence"]:
        raise ValueError("Missing complete runtime READ/actor identity diagnostics")
    return candidate


def matched_training(left, right, factor):
    infos = []
    for run in (left, right):
        model = run["manifest"]["models"]["memory"]
        if model.get("writer_checkpoint"):
            raise ValueError("First isolate representation/gate with FIFO, not a changed writer")
        checkpoint = Path(model["memory_checkpoint"])
        for filename, expected in model["checkpoint_files_sha256"].items():
            if file_hash(checkpoint / filename) != expected:
                raise ValueError("Evaluated checkpoint was modified")
        info = checkpoint_info_v18(model["base_model"]["path"], checkpoint)
        if info["step"] != info["config"]["train"]["max_steps"]:
            raise ValueError("Use matched fixed-final checkpoints, not per-arm selected intermediates")
        infos.append(info)
    a, b = infos
    for key in ("base_model", "cache_fingerprint", "plan_sha256", "source_sha256", "runtime",
                "initialization", "initial_shared_reader_sha256", "initial_expert_sha256"):
        if not a["metadata"].get(key) or a["metadata"].get(key) != b["metadata"].get(key):
            raise ValueError(f"Unmatched training: {key}")
    if a["step"] != b["step"]:
        raise ValueError("Training budgets differ")
    for name in ("expert", "expert_targets", "objective", "selection", "storage"):
        if a["config"].get(name) != b["config"].get(name):
            raise ValueError(f"More than {factor} changed: {name}")
    allowed = {"output_dir", "resume", "stop_after_steps", "preflight_only", factor}
    for section, allowed_keys in (("train", allowed), ("representation", {factor})):
        for key in a["config"][section].keys() | b["config"][section].keys():
            if key not in allowed_keys and a["config"][section].get(key) != b["config"][section].get(key):
                raise ValueError(f"More than {factor} changed: {section}/{key}")
    x, y = a["config"]["representation"][factor], b["config"]["representation"][factor]
    if factor == "representation" and (x != "short" or y not in {"adapted_short", "moment"}):
        raise ValueError("Use A(short) -> B(adapted_short) or A(short) -> C(moment), not B -> C")
    if factor == "gate" and (x != "linear" or y != "mlp"):
        raise ValueError("Gate comparison must be Linear -> MLP at identical representation")


def report(left_dir, right_dir, factor):
    left_role = "memory-off" if factor == "read" else "memory"
    left, right = load_completed(left_dir, left_role), load_completed(right_dir)
    compatible(left, right)
    # These are the exact same evaluator, so require its full closure, not just intersection.
    if left["manifest"]["source_sha256"] != right["manifest"]["source_sha256"]:
        raise ValueError("V18 evaluation source closure differs")
    if factor == "read":
        a = left["manifest"]["models"]["memory-off"]
        b = right["manifest"]["models"]["memory"]
        ignored = {"memory_off", "description"}
        if {k: v for k, v in a.items() if k not in ignored} != {k: v for k, v in b.items() if k not in ignored}:
            raise ValueError("READ-off comparison requires the identical checkpoint/AE/short/writer")
    else:
        matched_training(left, right, factor)
    comparisons = {factor + ": left -> right": contrast(left, left_role, right, "memory"),
        "HAMLET -> left": contrast(left, "baseline", left, left_role),
        "HAMLET -> right": contrast(right, "baseline", right, "memory")}
    result = {"factor": factor, "settings": left["manifest"]["settings"], "comparisons": comparisons,
              "left_run": str(Path(left_dir).resolve()), "right_run": str(Path(right_dir).resolve())}
    lines = [f"V18 matched {factor} experiment: closed-loop task success"]
    for name, row in comparisons.items():
        lo, hi = row["ci95"]
        lines.append(f"{name}: {100*row['left_macro']:.3f}% -> {100*row['right_macro']:.3f}% "
            f"({100*row['delta']:+.3f}pp); 95% paired CI [{100*lo:+.3f}, {100*hi:+.3f}]pp; "
            f"wins/losses={row['wins']}/{row['losses']}; McNemar p={row['mcnemar_p']:.6g}")
    primary = comparisons[factor + ": left -> right"]
    lines += ["", "Task                   Left/Right success   Wins/Losses"]
    for task, row in primary["tasks"].items():
        lines.append(f"{task:22} {row['left_successes']:3}/{row['n']} -> {row['right_successes']:3}/{row['n']}       {row['wins']}/{row['losses']}")
    lines += ["", "VAL160 is a repeated-development screen, not proof of TEST improvement.",
        "One success changes overall VAL160 by 0.625 percentage points.",
        "Intervals condition on these fixed tasks/scenarios and inference seed; repeated model selection adds optimism.",
        "For a promising model, use SAME-model READ-off and only then a fixed final larger TEST."]
    return result, "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left-run", required=True)
    p.add_argument("--right-run", required=True)
    p.add_argument("--factor", choices=("representation", "gate", "read"), default="representation")
    p.add_argument("--output-dir", help="Optional NEW report directory; otherwise stdout only")
    args = p.parse_args(argv)
    output = None
    if args.output_dir:
        from gr00t.long_memory.safety_v5 import validate_output_scope
        output = validate_output_scope(args.output_dir, args.left_run, args.right_run,
            ROOT / "checkpoints", ROOT / "data", ROOT / "runs/long_memory")
        if output.exists():
            raise FileExistsError("Use a new report directory")
    result, text = report(args.left_run, args.right_run, args.factor)
    if output:
        output.mkdir(parents=True, exist_ok=False)
        (output / "comparison_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        (output / "comparison_summary.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
