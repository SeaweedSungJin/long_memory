#!/usr/bin/env python3
"""Strict three-arm VAL160 summary; permits ONLY the intended writer difference.

No manifests or existing results are rewritten. Each evaluation first passes
the existing ownership/reference/runtime checks. Across runs actor, source,
environment, scenarios and settings must match. This is not a general-purpose
escape hatch from strict evaluation provenance.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gr00t.eval.sim.robomme.compare_long_memory_results import (
    read_results, paired_differences, paired_macro_bootstrap, mcnemar_exact,
    validate_result_identity)
from run_scripts.robomme.eval_representation_v18 import validate_manifest_contract, completed_read_diagnostics
from run_scripts.robomme.baseline_reference_v18 import validate_reference


def validate_study(single, coalitional):
    for value in (single, coalitional):
        validate_manifest_contract(value)
        settings = value["settings"]
        if (settings["dataset"] != "val" or settings["n_episodes"] != 10 or settings["seed"] != 6
                or len(settings["tasks"]) != 16 or settings["n_action_steps"] != 16
                or settings["max_episode_steps"] != 1300):
            raise ValueError("This report requires the fixed development VAL160 protocol")
    for key in ("settings", "source_sha256", "base_file_sha256", "policy_package_versions", "benchmark",
                "server_python", "robomme_python", "baseline_reference"):
        if single.get(key) != coalitional.get(key):
            raise ValueError(f"CVoM study mismatch: {key}")
    left, right = single["models"]["memory"], coalitional["models"]["memory"]
    allowed = {"writer_checkpoint", "writer_files_sha256", "cvom_admission"}
    if {k: v for k, v in left.items() if k not in allowed} != {k: v for k, v in right.items() if k not in allowed}:
        raise ValueError("Only writer checkpoint/provenance may differ between learned arms")
    lm, rm = left["cvom_admission"]["manifest"], right["cvom_admission"]["manifest"]
    if lm["arm"] != "single" or rm["arm"] != "coalitional":
        raise ValueError("Single and coalitional checkpoint roles reversed or duplicated")
    for key in ("parent_identity", "config", "step", "metadata"):
        if lm[key] != rm[key]:
            raise ValueError(f"Learned arms do not share actor/training protocol: {key}")
    if left["cvom_admission"]["source_sha256"] != right["cvom_admission"]["source_sha256"]:
        raise ValueError("Runtime writer source changed between learned arms")
    fifo = single["models"].get("fifo")
    if fifo is None or fifo.get("write_policy") != "fifo" or fifo.get("memory_off") is not False:
        raise ValueError("Missing frozen-actor FIFO reference")
    expected = {**left, "write_policy": "fifo"}
    # Current evaluator descriptions are identical; READ remains ON in both.
    if fifo != expected:
        raise ValueError("FIFO must use the exact same actor, writer sidecar and READ settings")


def build_summary(single_root, coalitional_root, bootstrap_samples=5000):
    roots = [Path(single_root).resolve(), Path(coalitional_root).resolve()]
    manifests = [json.loads((p / "comparison_manifest.json").read_text()) for p in roots]
    validate_study(*manifests)
    for manifest in manifests:
        validate_reference(manifest["baseline_reference"], manifest)
    tasks = manifests[0]["settings"]["tasks"]
    arms = {"fifo": (roots[0], "fifo", manifests[0]), "single": (roots[0], "memory", manifests[0]),
            "coalitional": (roots[1], "memory", manifests[1])}
    results, ownership, diagnostics = {}, {}, {}
    for arm, (root, role, manifest) in arms.items():
        results[arm], ownership[arm] = {}, {}
        diagnostics[arm] = completed_read_diagnostics(root, role, manifest)
        if not diagnostics[arm].get("complete_evidence"):
            raise ValueError(f"Incomplete runtime READ/writer evidence: {arm}")
        for task in tasks:
            rows = read_results(root / role / task / "simulation_results.csv", expected=10)
            if len(rows) != 10:
                raise ValueError(f"Incomplete {arm}/{task}; never count missing episodes as failures")
            results[arm][task] = rows
            ownership[arm][task] = validate_result_identity(root, role, task, manifest)
    for task in tasks:
        for arm in ("single", "coalitional"):
            for key in ("scenario_metadata_sha256", "model_config_sha256", "memory_window", "demo_sampling"):
                if ownership["fifo"][task].get(key) != ownership[arm][task].get(key):
                    raise ValueError(f"Scenario/control mismatch {arm}/{task}/{key}")
    per_task = [{"task": task, **{arm: sum(r["success"] for r in results[arm][task].values()) for arm in arms}}
                for task in tasks]
    totals = {arm: sum(row[arm] for row in per_task) for arm in arms}
    contrasts = {}
    for left, right in (("fifo", "single"), ("fifo", "coalitional"), ("single", "coalitional")):
        groups, flips = [], []
        for task in tasks:
            a, b = results[left][task], results[right][task]
            group = paired_differences(a, b)
            groups.append(group)
            for eid in sorted(a):
                if a[eid]["success"] != b[eid]["success"]:
                    flips.append({"task": task, "episode_idx": eid, "episode_seed": a[eid]["episode_seed"],
                                  "left": a[eid]["success"], "right": b[eid]["success"]})
        values = [v for group in groups for v in group]
        wins, losses = values.count(1), values.count(-1)
        contrasts[f"{left}_to_{right}"] = {"paired_n": len(values), "delta": sum(values)/len(values),
            "ci95": paired_macro_bootstrap(groups, samples=bootstrap_samples), "wins": wins, "losses": losses,
            "same": values.count(0), "mcnemar_p": mcnemar_exact(wins, losses), "flips": flips}
    report = {"kind": "writer_only_frozen_actor_VAL160", "totals": totals, "per_task": per_task,
        "contrasts": contrasts, "runtime_evidence": diagnostics,
        "sources": {"single": str(roots[0]), "coalitional": str(roots[1])},
        "evaluation_ids": [m["evaluation_id"] for m in manifests],
        "limitations": "Repeated development VAL160, one inference seed. CI conditional on these tasks. Not final TEST generalization."}
    lines = ["Frozen V19 actor / CVoM writer-only paired VAL160 study", ""]
    lines += [f"{arm}: {count}/160 ({100*count/160:.3f}%)" for arm, count in totals.items()]
    lines += ["", "Task                     FIFO  Single  Coalitional"]
    lines += [f"{row['task']:<24} {row['fifo']:>4} {row['single']:>7} {row['coalitional']:>12}" for row in per_task]
    for name, c in contrasts.items():
        lines += ["", f"{name}: {100*c['delta']:+.3f}pp; wins/losses/same={c['wins']}/{c['losses']}/{c['same']}",
                  f"95% paired CI [{100*c['ci95'][0]:+.3f}, {100*c['ci95'][1]:+.3f}]pp; p={c['mcnemar_p']:.6g}"]
    lines += ["", report["limitations"]]
    return report, "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--single-run", type=Path, required=True)
    p.add_argument("--coalitional-run", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError("Summary must use a NEW output directory")
    report, text = build_summary(args.single_run, args.coalitional_run)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "comparison.txt").write_text(text)
    with (args.output_dir / "tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task", "fifo", "single", "coalitional"))
        writer.writeheader()
        writer.writerows(report["per_task"])
    print(text)


if __name__ == "__main__":
    main()
