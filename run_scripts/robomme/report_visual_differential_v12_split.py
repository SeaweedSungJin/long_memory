#!/usr/bin/env python3
"""Read-only paired report for independently run V12 evaluation arms.

No CSV, policy identity or baseline is copied/relabelled. The two original
manifests remain distinct, while role-aware path adapters let the unchanged
paired-episode statistics read their original locations. Partial completed
episodes are reported honestly; contradictory identities fail closed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_scripts.robomme import eval_visual_differential_v12 as evaluation
from gr00t.long_memory.monitoring import _atomic_json

DRIVER = "visual_differential_v12_split_report"
CONTEXT = ("scenario_metadata_sha256", "model_config_sha256", "memory_window", "demo_sampling")
PROTOCOL = ("settings", "source_sha256", "base_file_sha256", "policy_package_versions", "benchmark",
            "server_python", "robomme_python", "memory_input", "control_contract", "learned_storage_selection")
CONTRASTS = (("visual-off", "differential"), ("current-only", "differential"), ("visual-off", "current-only"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_pair(left, right):
    """Evaluation IDs differ legitimately; experimental protocols must not."""
    for manifest in (left, right):
        evaluation.validate_manifest_contract(manifest)
        if "baseline_reference" not in manifest:
            raise ValueError("Both split runs require explicit REUSED original baseline references")
    if set(left["models"]) != {"baseline", "differential", "visual-off"}:
        raise ValueError("Differential run must contain baseline/differential/visual-off only")
    if set(right["models"]) != {"baseline", "current-only"}:
        raise ValueError("Current-only run must contain baseline/current-only only")
    for key in PROTOCOL:
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f"Split evaluation protocol differs: {key}")
    if left["models"]["baseline"] != right["models"]["baseline"]:
        raise ValueError("Original baseline descriptions/identity differ")
    if left["baseline_reference"] != right["baseline_reference"]:
        raise ValueError("Baseline references must be the SAME original run, IDs and file hashes")
    a, b = left["models"]["differential"], right["models"]["current-only"]
    for key in ("base_model", "frozen_parent", "visual_config", "camera_order", "training_contract"):
        if a[key] != b[key]:
            raise ValueError(f"Split trained-arm comparison is unmatched: {key}")
    contract = a["training_contract"]
    if (contract["train_options"].get("max_steps") != 512
            or contract.get("selection_metric") != "val/generated_observed_prefix_mae"):
        raise ValueError("Requires the declared independently MAE-selected 512-update arms")


def _contained(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Provenance path escapes its declared root")
    return path


def immutable_files(manifests):
    """Resolve expected current bytes without loading a model or simulator."""
    files = {}
    def add(path, sha):
        path = Path(path).resolve()
        if not evaluation._hash(sha) or (str(path) in files and files[str(path)] != sha):
            raise ValueError("Conflicting/missing immutable file hash")
        files[str(path)] = sha
    expected_sources = {str(p.relative_to(ROOT)) for p in (ROOT / "gr00t").rglob("*.py")}
    expected_sources.update(map(str, (evaluation.SERVER, evaluation.BASELINE_SERVER, evaluation.EVALUATOR, *evaluation.DEPENDENCIES)))
    for manifest in manifests:
        if set(manifest["source_sha256"]) != expected_sources:
            raise ValueError("V12 evaluator source closure differs")
        for name, sha in manifest["source_sha256"].items():
            add(_contained(ROOT, name), sha)
        base = Path(manifest["models"]["baseline"]["base_model"]["path"])
        for name, sha in manifest["base_file_sha256"].items():
            add(_contained(base, name), sha)
        benchmark = manifest["benchmark"].get("source_and_scenario_sha256", {})
        if not benchmark:
            raise ValueError("Missing benchmark/source/scenario provenance")
        for name, sha in benchmark.items():
            add(name, sha)
        for name, version in manifest["policy_package_versions"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f"Current policy package version differs: {name}")
        for role, model in manifest["models"].items():
            if role == "baseline":
                continue
            for name, sha in model["checkpoint_files_sha256"].items():
                add(_contained(model["memory_checkpoint"], name), sha)
            parent = model["frozen_parent"]
            for name, sha in parent["files_sha256"].items():
                add(_contained(parent["path"], name), sha)
    return files


def verify_files(expected):
    for path, sha in expected.items():
        if evaluation.file_hash(Path(path)) != sha:
            raise ValueError(f"Immutable provenance file changed: {path}")


def result_snapshot(runs, manifests):
    """Bind raw evidence, including partial/missing files, without driver locks.

    An active driver is allowed. If evidence changes while it is being read,
    report generation fails safely and can be retried into the same absent
    output directory; it never mixes two times or fabricates completed rows.
    """
    files = {}
    for root, manifest in zip(runs, manifests):
        paths = [root / "comparison_manifest.json", root / "driver_status.json"]
        for role in manifest["models"]:
            if role == "baseline":
                continue
            for task in manifest["settings"]["tasks"]:
                folder = root / role / task
                paths.extend(folder / name for name in ("simulation_results.csv", "policy_manifest.json", "memory_diagnostics.jsonl"))
        for path in paths:
            if path.exists() and (not path.is_file() or not path.resolve().is_relative_to(root)):
                raise ValueError("Result evidence is not an in-run regular file")
            files[str(path)] = evaluation.file_hash(path) if path.is_file() else None
    return files


class RoleRoots:
    """Nonmutating adapter for unchanged _contrast(root / role / task / CSV)."""
    def __init__(self, origins):
        self.origins = origins

    def __truediv__(self, role):
        if role not in self.origins:
            raise ValueError("Unknown split-report role")
        return self.origins[role] / role


def build_split_report(differential_run, current_only_run, *, bootstrap_samples=5000):
    if type(bootstrap_samples) is not int or bootstrap_samples <= 0:
        raise ValueError("Positive bootstrap sample count required")
    runs = [Path(differential_run).resolve(), Path(current_only_run).resolve()]
    if runs[0] == runs[1]:
        raise ValueError("Split runs must have separate original manifests")
    manifest_bytes = [(root / "comparison_manifest.json").read_bytes() for root in runs]
    manifests = [json.loads(payload) for payload in manifest_bytes]
    validate_pair(*manifests)
    immutable = immutable_files(manifests)
    verify_files(immutable)
    before = result_snapshot(runs, manifests)
    if any(before[str(root / "comparison_manifest.json")] != hashlib.sha256(payload).hexdigest()
           for root, payload in zip(runs, manifest_bytes)):
        raise ValueError("Evaluation manifest changed during read; retry this snapshot report")
    # These pure existing reporters validate every completed policy ID, paired
    # scenario, explicit baseline reference and completed-session diagnostic.
    reports = [evaluation.build_control_report(root, bootstrap_samples=bootstrap_samples)[0] for root in runs]
    if reports[0]["models"]["baseline"] != reports[1]["models"]["baseline"]:
        raise ValueError("Two reports resolved different baseline outcomes/provenance")
    settings = manifests[0]["settings"]
    tasks, expected = settings["tasks"], settings["n_episodes"]
    origins = {"differential": runs[0], "visual-off": runs[0], "current-only": runs[1]}
    by_role = {role: manifests[index] for index, roles in enumerate((("differential", "visual-off"), ("current-only",))) for role in roles}
    # Cross-run context checks complement existing per-run baseline pairing.
    for task in tasks:
        contexts = []
        for role, root in origins.items():
            if evaluation.read_results(root / role / task / "simulation_results.csv", expected=expected):
                contexts.append(evaluation.validate_result_identity(root, role, task, by_role[role]))
        if contexts and any(any(row.get(key) != contexts[0].get(key) for key in CONTEXT) for row in contexts[1:]):
            raise ValueError(f"Split paired rollout/scenario context differs: {task}")
    models = copy.deepcopy(reports[0]["models"])
    models["current-only"] = copy.deepcopy(reports[1]["models"]["current-only"])
    contrasts = {f"{left}_to_{right}": evaluation._contrast(RoleRoots(origins), left, right, tasks, expected, bootstrap_samples)
                 for left, right in CONTRASTS}
    reference = manifests[0]["baseline_reference"]
    result = {"format_version": 1, "report_variant": DRIVER, "metric": "closed_loop_task_success",
        "settings": settings, "models": models,
        "baseline_comparisons": {**reports[0]["comparisons"], **reports[1]["comparisons"]},
        "additional_comparisons": contrasts, "baseline_reference": reference,
        "source_runs": [{"path": str(root), "evaluation_id": manifest["evaluation_id"],
                         "manifest_sha256": before[str(root / "comparison_manifest.json")]} for root, manifest in zip(runs, manifests)],
        "source_driver_status": {str(root): json.loads((root / "driver_status.json").read_text())
                                 if (root / "driver_status.json").is_file() else None for root in runs},
        "selected_models": {role: by_role[role]["models"][role] for role in origins},
        "storage_diagnostics": {**reports[0]["storage_diagnostics"], **reports[1]["storage_diagnostics"]},
        "visual_diagnostics": {**reports[0]["visual_diagnostics"], **reports[1]["visual_diagnostics"]},
        "visual_diagnostics_complete": all(report["visual_diagnostics_complete"] for report in reports),
        "complete": all(model["complete"] for model in models.values()) and all(c["complete"] for c in contrasts.values()),
        "files_sha256": before, "immutable_files_sha256": immutable,
        "baseline_source_files_sha256": reference["files_sha256"],
        "report_source_sha256": evaluation.file_hash(Path(__file__)),
        "bootstrap_samples": bootstrap_samples,
        "limitations": ["Partial/missing episodes are excluded, never counted as task failures.",
            "Original baseline is REUSED once, not a new rollout or independent replication.",
            "Both arms declare the same planned512 budget and MAE-selection protocol; supplied checkpoint steps may differ, including0. This reporter does not independently verify the final best pointers.",
            "Visual-off preserves the differential checkpoint's original frozen archive1250/AE; it is not all-memory-off.",
            "Bootstrap is conditional on these tasks and inference seed; this is not final-test proof of >=30%."]}
    # Recheck original references, all immutable sources, and the exact evidence
    # snapshot before publishing anything. There is no force-complete override.
    for manifest in manifests:
        evaluation.validate_reference(manifest["baseline_reference"], manifest)
    verify_files(immutable)
    if result_snapshot(runs, manifests) != before:
        raise ValueError("Evaluation evidence changed during read; retry this snapshot report")
    result["report_id"] = digest(result)
    return result, render_report(result)


def render_report(result):
    lines = ["RoboMME V12 split-run paired task-success comparison",
             "COMPLETE" if result["complete"] else "INCOMPLETE — only matched completed episodes included",
             f"Report ID: {result['report_id']}",
             f"Baseline REUSED: {result['baseline_reference']['source_run']} (new rollouts 0)", ""]
    for role in evaluation.ROLES:
        model = result["models"][role]
        n = sum(x["completed"] for x in model["tasks"].values())
        wins = sum(x["successes"] for x in model["tasks"].values())
        expected = sum(x["requested"] for x in model["tasks"].values())
        macro = model["available_task_macro"]
        rate = "--" if macro is None else f"{100 * macro:.2f}%"
        step = "original" if role == "baseline" else str(result["selected_models"][role]["step"])
        state = "" if role == "baseline" else f"; {result['selected_models'][role]['checkpoint_status']}"
        lines.append(f"{role}: {wins}/{n} successes, {n}/{expected} completed; available-task macro={rate}; step={step}{state}")
    for title, comparisons in (("Original baseline comparisons", result["baseline_comparisons"]),
                               ("Historical/current-only controls", result["additional_comparisons"])):
        lines += ["", title]
        for name, comparison in comparisons.items():
            delta = comparison["paired_task_macro_delta"]
            change = "--" if delta is None else f"{100 * delta:+.2f}pp"
            label = name if title.startswith("Historical") else "baseline_to_" + name
            lines.append(f"{label}: {change}; paired N={comparison['paired_n']}; "
                         f"wins/losses/same={comparison['wins']}/{comparison['losses']}/{comparison['same']}; "
                         f"{'COMPLETE' if comparison['complete'] else 'INCOMPLETE'}")
            if comparison["paired_task_macro_bootstrap_ci95"]:
                lo, hi = comparison["paired_task_macro_bootstrap_ci95"]
                lines.append(f"  95% paired within-task bootstrap CI [{100*lo:+.2f}, {100*hi:+.2f}]pp; McNemar p={comparison['mcnemar_exact_p']:.6g}")
    lines += ["", "Runtime visual diagnostics complete: " + str(result["visual_diagnostics_complete"]), *result["limitations"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--differential-run", required=True)
    parser.add_argument("--current-only-run", required=True)
    parser.add_argument("--output-dir", required=True, help="NEW directory; never rewrites source evaluation reports")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args(argv)
    output = Path(args.output_dir).resolve()
    evaluation.validate_output_scope(output, args.differential_run, args.current_only_run)
    if output.exists():
        raise FileExistsError("Use a NEW split-report output directory")
    result, text = build_split_report(args.differential_run, args.current_only_run, bootstrap_samples=args.bootstrap_samples)
    protected = [result["baseline_reference"]["source_run"]]
    for model in result["selected_models"].values():
        protected += [model["base_model"]["path"], model["memory_checkpoint"], model["frozen_parent"]["path"]]
        metadata = model["training_metadata"]
        protected += [metadata.get("cache_dir"), (metadata.get("initialization_reference") or {}).get("path")]
        cache = metadata.get("cache_dir")
        if cache and (Path(cache) / "manifest.json").is_file():
            protected.append(json.loads((Path(cache) / "manifest.json").read_text()).get("dataset_path"))
    evaluation.validate_output_scope(output, *protected)
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "comparison_summary.json", result)
    # Exclusive creation: the newly allocated report directory is never reused.
    with (output / "comparison_summary.txt").open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
