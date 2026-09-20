#!/usr/bin/env python3
"""Read-only paired report for independently run V14 evaluation arms.

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

from run_scripts.robomme import eval_visual_expert_v14 as evaluation
from gr00t.long_memory.monitoring import _atomic_json

DRIVER = "visual_expert_v14_split_report"
CONTEXT = ("scenario_metadata_sha256", "model_config_sha256", "memory_window", "demo_sampling")
PROTOCOL = ("settings", "allow_initialization_checkpoints", "source_sha256", "visual_extraction_assets_sha256", "base_file_sha256", "policy_package_versions", "benchmark",
            "server_python", "robomme_python", "memory_input", "control_contract", "learned_storage_selection")
CONTRASTS = (("visual-off", "visual"),)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_pair(left, right):
    """Evaluation IDs differ legitimately; experimental protocols must not."""
    for manifest in (left, right):
        evaluation.validate_manifest_contract(manifest)
        if "baseline_reference" not in manifest:
            raise ValueError("Both split runs require explicit REUSED original baseline references")
    if set(left["models"]) != {"baseline", "visual"}:
        raise ValueError("Visual run must contain baseline/visual only")
    if set(right["models"]) != {"baseline", "visual-off"}:
        raise ValueError("Visual-off run must contain baseline/visual-off only")
    for key in PROTOCOL:
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f"Split evaluation protocol differs: {key}")
    if left["models"]["baseline"] != right["models"]["baseline"]:
        raise ValueError("Original baseline descriptions/identity differ")
    if left["baseline_reference"] != right["baseline_reference"]:
        raise ValueError("Baseline references must be the SAME original run, IDs and file hashes")
    a, b = left["models"]["visual"], right["models"]["visual-off"]
    ignored = {"description", "visual_read_off"}
    if ({k: v for k, v in a.items() if k not in ignored}
            != {k: v for k, v in b.items() if k not in ignored}):
        raise ValueError("Split on/off must use the SAME V14 bundle, expert, initialization and training contract")
    if (a["step"] != 512 or b["step"] != 512
            or a["training_config"].get("max_steps") != 512
            or a["training_objective"].get("rollout_selection") != "fixed_final_step"):
        raise ValueError("Requires the SAME fixed-final step512, not MAE-best or initialization selections")


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
    expected_sources.update(map(str, (evaluation.SERVER, evaluation.BASELINE_SERVER, evaluation.EVALUATOR,
                                     evaluation.CLIENT, *evaluation.DEPENDENCIES)))
    eagle = ROOT / "gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2"
    expected_assets = {str(p.relative_to(ROOT)) for p in eagle.rglob("*") if p.is_file()
                       and "__pycache__" not in p.parts and p.suffix not in (".py", ".pyc")}
    for manifest in manifests:
        if set(manifest["source_sha256"]) != expected_sources:
            raise ValueError("V14 evaluator source closure differs")
        for name, sha in manifest["source_sha256"].items():
            add(_contained(ROOT, name), sha)
        if set(manifest["visual_extraction_assets_sha256"]) != expected_assets:
            raise ValueError("V14 original image-only extraction assets differ")
        for name, sha in manifest["visual_extraction_assets_sha256"].items():
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
            # Bind declared final-step/header semantics to the ACTUAL immutable
            # bundle, not merely a consistently edited comparison manifest.
            header = json.loads((Path(model["memory_checkpoint"]) / "checkpoint.json").read_text())
            config = header.get("config", {})
            if (header.get("kind") != evaluation.VARIANT or header.get("format_version") != 1
                    or header.get("self_contained") is not False
                    or config.get("trainer_variant") != evaluation.VARIANT
                    or config.get("driver_variant") != evaluation.VARIANT or config.get("stage") != 1
                    or header.get("step") != model["step"]
                    or header.get("metadata") != model["training_metadata"]
                    or any(config.get(key) != model[field] for key, field in (
                        ("train", "training_config"), ("visual", "visual_config"), ("objective", "training_objective"),
                        ("camera_order", "camera_order"), ("mode", "mode"), ("read_mode", "read_mode"),
                        ("include_tail", "include_tail"), ("replay_encoding", "replay_encoding"),
                        ("extraction_rule", "extraction_rule"), ("architecture", "architecture"),
                        ("expert", "expert_config"), ("expert_targets", "expert_targets")))):
                raise ValueError("V14 declared final-step model differs from the actual checkpoint header")
            for name, sha in model["checkpoint_files_sha256"].items():
                add(_contained(model["memory_checkpoint"], name), sha)
            parent = model["initial_parent"]
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


def build_split_report(visual_run, visual_off_run, *, bootstrap_samples=5000):
    if type(bootstrap_samples) is not int or bootstrap_samples <= 0:
        raise ValueError("Positive bootstrap sample count required")
    runs = [Path(visual_run).resolve(), Path(visual_off_run).resolve()]
    if runs[0] == runs[1]:
        raise ValueError("Split runs must have separate original manifests")
    manifest_bytes = [(root / "comparison_manifest.json").read_bytes() for root in runs]
    manifests = [json.loads(payload) for payload in manifest_bytes]
    own_source = evaluation.file_hash(Path(__file__))
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
    origins = {"visual": runs[0], "visual-off": runs[1]}
    by_role = {role: manifests[index] for index, roles in enumerate((("visual",), ("visual-off",))) for role in roles}
    # Cross-run context checks complement existing per-run baseline pairing.
    for task in tasks:
        contexts = []
        for role, root in origins.items():
            if evaluation.read_results(root / role / task / "simulation_results.csv", expected=expected):
                contexts.append(evaluation.validate_task_result(root, role, task, by_role[role]))
        if contexts and any(any(row.get(key) != contexts[0].get(key) for key in CONTEXT) for row in contexts[1:]):
            raise ValueError(f"Split paired rollout/scenario context differs: {task}")
    models = copy.deepcopy(reports[0]["models"])
    models["visual-off"] = copy.deepcopy(reports[1]["models"]["visual-off"])
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
        "tail_diagnostics": {**reports[0]["tail_diagnostics"], **reports[1]["tail_diagnostics"]},
        "visual_diagnostics_complete": all(report["visual_diagnostics_complete"] for report in reports),
        "episode_results_complete": all(model["complete"] for model in models.values()) and all(c["complete"] for c in contrasts.values()),
        "files_sha256": before, "immutable_files_sha256": immutable,
        "baseline_source_files_sha256": reference["files_sha256"],
        "report_source_sha256": own_source,
        "bootstrap_samples": bootstrap_samples,
        "limitations": ["Partial/missing episodes are excluded, never counted as task failures.",
            "Original baseline is REUSED once, not a new rollout or independent replication.",
            "Both roles use the SAME jointly trained visual/expert final step512. MAE-best is diagnostic only.",
            "Visual-off retains the SAME NEW V14 adapted AE, frozen archive memory and observed demo-tail ingestion; only visual READ is bypassed.",
            "Legacy archive1250 is not V14 visual-off. Original V13 RGB-ingest transport is reused with actual V14 expert/visual diagnostics.",
            "Both roles ingest the same tail; READ-off removes visual retrieval, not its encoding/storage cost.",
            "Bootstrap is conditional on these tasks and inference seed; this is not final-test proof of >=30%."]}
    # Recheck original references, all immutable sources, and the exact evidence
    # snapshot before publishing anything. There is no force-complete override.
    for manifest in manifests:
        evaluation.validate_reference(manifest["baseline_reference"], manifest)
    if immutable_files(manifests) != immutable:
        raise ValueError("Split immutable source/runtime/header identity changed during read")
    verify_files(immutable)
    if result_snapshot(runs, manifests) != before:
        raise ValueError("Evaluation evidence changed during read; retry this snapshot report")
    if evaluation.file_hash(Path(__file__)) != own_source:
        raise ValueError("Split-report source changed during read")
    result["source_drivers_complete"] = all(isinstance(status, dict)
        and status.get("interrupted") is False and status.get("failures") == [] and status.get("fatal") is None
        and status.get("inference_files_unchanged") is True for status in result["source_driver_status"].values())
    result["complete"] = (result["episode_results_complete"] and result["visual_diagnostics_complete"]
                          and result["source_drivers_complete"])
    result["report_id"] = digest(result)
    return result, render_report(result)


def render_report(result):
    lines = ["RoboMME V14 split-run paired task-success comparison",
             "COMPLETE" if result["complete"] else "INCOMPLETE — inspect episode, diagnostic, and driver completeness separately",
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
                               ("Same adapted-expert visual READ control", result["additional_comparisons"])):
        lines += ["", title]
        for name, comparison in comparisons.items():
            delta = comparison["paired_task_macro_delta"]
            change = "--" if delta is None else f"{100 * delta:+.2f}pp"
            label = "baseline_to_" + name if title.startswith("Original baseline") else name
            lines.append(f"{label}: {change}; paired N={comparison['paired_n']}; "
                         f"wins/losses/same={comparison['wins']}/{comparison['losses']}/{comparison['same']}; "
                         f"{'COMPLETE' if comparison['complete'] else 'INCOMPLETE'}")
            if comparison["paired_task_macro_bootstrap_ci95"]:
                lo, hi = comparison["paired_task_macro_bootstrap_ci95"]
                lines.append(f"  95% paired within-task bootstrap CI [{100*lo:+.2f}, {100*hi:+.2f}]pp; McNemar p={comparison['mcnemar_exact_p']:.6g}")
    lines += ["", "Episode results complete: " + str(result["episode_results_complete"]),
              "Runtime visual/RPC diagnostics complete: " + str(result["visual_diagnostics_complete"]),
              "Clean terminal source drivers: " + str(result["source_drivers_complete"]), *result["limitations"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-run", required=True)
    parser.add_argument("--visual-off-run", required=True)
    parser.add_argument("--output-dir", required=True, help="NEW directory; never rewrites source evaluation reports")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args(argv)
    output = Path(args.output_dir).resolve()
    evaluation.validate_output_scope(output, args.visual_run, args.visual_off_run)
    if output.exists():
        raise FileExistsError("Use a NEW split-report output directory")
    result, text = build_split_report(args.visual_run, args.visual_off_run, bootstrap_samples=args.bootstrap_samples)
    protected = [result["baseline_reference"]["source_run"]]
    for model in result["selected_models"].values():
        protected += [model["base_model"]["path"], model["memory_checkpoint"], model["initial_parent"]["path"]]
        metadata = model["training_metadata"]
        protected += [metadata.get("cache_dir"), (metadata.get("initialization_reference") or {}).get("path"),
                      metadata["sidecar"]["path"]]
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
