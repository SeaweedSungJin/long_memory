#!/usr/bin/env python3
"""Compare frozen parent, action-only control, guided model and same-AE READ-off.

Reporting only: existing CSVs are read in place, never copied or rewritten.
The prespecified primary contrast is control -> guided (auxiliary-loss effect).
Parent -> guided instead asks whether the entire continuation helped. Neither
contrast identifies a learned writer: all three models use append-only storage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_scripts.robomme.compare_archive_retrieval_v17 import (
    TASKS, compatible, contrast, load_run, matched_training,
)
from run_scripts.robomme.eval_long_memory_comparison import file_hash

PAYLOAD_KEYS = {
    "checkpoint.json": "checkpoint_sha256",
    "model.safetensors": "weights_sha256",
    "expert.safetensors": "expert_weights_sha256",
    "cvom.safetensors": "cvom_weights_sha256",
}
PRIMARY = "action-only -> retrieval-guided"


def checkpoint_model(path):
    """A lightweight, CPU-only identity used before scheduling any rollout."""
    path = Path(path).resolve(strict=True)
    info = json.loads((path / "checkpoint.json").read_text())
    return {"memory_checkpoint": str(path), "step": info["step"],
            **{key: file_hash(path / filename) for filename, key in PAYLOAD_KEYS.items()}}


def verify_recorded_checkpoint(model):
    """Reject post-rollout checkpoint swaps, including AE or memory payloads."""
    actual = checkpoint_model(model["memory_checkpoint"])
    for key in ("step", *PAYLOAD_KEYS.values()):
        if not model.get(key) or model[key] != actual[key]:
            raise ValueError(f"Evaluated checkpoint payload changed: {key}")
    return json.loads((Path(actual["memory_checkpoint"]) / "checkpoint.json").read_text())


def verify_parent(parent, control, guided):
    """Bind the evaluated parent to both arms' actual initialization payloads."""
    parent_model = parent["manifest"]["models"]["archive"]
    info = verify_recorded_checkpoint(parent_model)
    if info["config"].get("mode") != "archive" or info["config"].get("stage") != 1:
        raise ValueError("Parent must be an unchanged Stage-1 archive policy")
    for run in (control, guided):
        candidate = verify_recorded_checkpoint(run["manifest"]["models"]["archive"])
        origin = candidate["metadata"]["original_continuation_parent"]
        if Path(origin["path"]).resolve() != Path(parent_model["memory_checkpoint"]).resolve():
            raise ValueError("Evaluated parent is not the recorded continuation parent")
        if origin["step"] != parent_model["step"]:
            raise ValueError("Continuation parent step differs")
        for filename, key in PAYLOAD_KEYS.items():
            if origin["files"].get(filename) != parent_model[key]:
                raise ValueError(f"Continuation parent payload differs: {filename}")


def validate_checkpoint_family(parent_path, control_path, guided_path):
    """Read-only training-match check before expensive evaluation; no GPU loads."""
    runs = [{"manifest": {"models": {"archive": checkpoint_model(path)}}}
            for path in (parent_path, control_path, guided_path)]
    matched_training(runs[1], runs[2])
    verify_parent(*runs)
    return [run["manifest"]["models"]["archive"] for run in runs]


def build_report(parent_dir, control_dir, guided_dir, historical_dir=None, *, samples=5000):
    # Parent/control need only two roles, avoiding unnecessary READ-off rollouts.
    parent = load_run(parent_dir, ("baseline", "archive"))
    control = load_run(control_dir, ("baseline", "archive"))
    guided = load_run(guided_dir, ("baseline", "archive", "archive-off"))
    compatible(parent, guided)
    compatible(control, guided)
    matched_training(control, guided)
    verify_parent(parent, control, guided)
    comparisons = {
        PRIMARY: contrast(control, "archive", guided, "archive", samples=samples),
        "unchanged parent -> retrieval-guided": contrast(parent, "archive", guided, "archive", samples=samples),
        "same guided AE READ-off -> READ-on": contrast(guided, "archive-off", guided, "archive", samples=samples),
        "original HAMLET -> retrieval-guided": contrast(guided, "baseline", guided, "archive", samples=samples),
    }
    inputs = {"parent": parent, "control": control, "guided": guided}
    if historical_dir:
        s = guided["manifest"]["settings"]
        if s["dataset"] != "test" or s["tasks"] != TASKS or s["n_episodes"] != 50 or s["seed"] != 6:
            raise ValueError("Historical 19.375% comparison requires all16 TEST n50 seed6")
        old = load_run(historical_dir, ("baseline", "memory"))
        compatible(old, guided)
        comparisons["historical V4 -> retrieval-guided (secondary)"] = contrast(
            old, "memory", guided, "archive", samples=samples)
        inputs["historical"] = old
    result = {"format_version": 1, "primary_contrast": PRIMARY,
              "settings": guided["manifest"]["settings"], "comparisons": comparisons,
              "inputs": {name: {"root": run["root"], "evaluation_id": run["manifest"]["evaluation_id"],
                                "result_files_sha256": {
                                    str(Path(role) / task / filename): file_hash(Path(run["root"]) / role / task / filename)
                                    for role in run["rows"] for task in run["rows"][role]
                                    for filename in ("simulation_results.csv", "policy_manifest.json")}}
                         for name, run in inputs.items()},
              "limitations": ["One inference seed and one training seed; intervals condition on these tasks.",
                  "Secondary contrasts are exploratory; p-values are not multiplicity-adjusted.",
                  "VAL has already informed development and is not an untouched confirmatory test.",
                  "RoboMME TEST has also been examined in earlier versions: this remains an exploratory fixed-model comparison.",
                  "READ-off measures the role of retrieval with the SAME adapted AE, not learned WRITE.",
                  "Historical V4 differs in architecture/training; its contrast does not isolate retrieval loss."]}
    lines = ["V17 fair comparison: closed-loop task success", f"PRIMARY: {PRIMARY}",
             f"Dataset={result['settings']['dataset']}; episodes/task={result['settings']['n_episodes']}; "
             f"seed={result['settings']['seed']}"]
    for name, row in comparisons.items():
        lo, hi = row["ci95"]
        total = sum(t["n"] for t in row["tasks"].values())
        a = sum(t["left_successes"] for t in row["tasks"].values())
        b = sum(t["right_successes"] for t in row["tasks"].values())
        lines.append(f"{name}: {a}/{total} ({100*row['left_macro']:.3f}%) -> "
                     f"{b}/{total} ({100*row['right_macro']:.3f}%); {100*row['delta']:+.3f}pp; "
                     f"CI [{100*lo:+.3f}, {100*hi:+.3f}]pp; "
                     f"wins/losses={row['wins']}/{row['losses']}; McNemar p={row['mcnemar_p']:.6g}")
    lines += ["", "Primary contrast by task:", "Task                   Left  Right     Delta   Wins/Losses"]
    for task, row in comparisons[PRIMARY]["tasks"].items():
        delta = 100 * (row["right_successes"] - row["left_successes"]) / row["n"]
        lines.append(f"{task:22s} {row['left_successes']:4d}  {row['right_successes']:5d} "
                     f"{delta:+8.2f}pp   {row['wins']}/{row['losses']}")
    lines += ["", *result["limitations"], "A positive point estimate alone is not proof of a reliable improvement."]
    return result, "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "control", "guided"):
        parser.add_argument(f"--{name}-run", required=True)
    parser.add_argument("--historical-run", help="Optional matched all16 TEST n50 seed6 V4 run")
    parser.add_argument("--output-dir", help="NEW report directory; omitted means stdout only")
    args = parser.parse_args(argv)
    output = Path(args.output_dir).resolve() if args.output_dir else None
    if output:
        if output.exists():
            raise FileExistsError("Use a NEW report directory; existing results are immutable")
        from gr00t.long_memory.safety_v5 import validate_output_scope
        validate_output_scope(output, args.parent_run, args.control_run, args.guided_run, args.historical_run,
                              ROOT / "checkpoints", ROOT / "data", ROOT / "runs/long_memory")
    result, report_text = build_report(args.parent_run, args.control_run, args.guided_run, args.historical_run)
    if output:
        output.mkdir(parents=True, exist_ok=False)
        (output / "comparison.json").write_text(json.dumps(result, indent=2))
        (output / "comparison.txt").write_text(report_text)
    print(report_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
