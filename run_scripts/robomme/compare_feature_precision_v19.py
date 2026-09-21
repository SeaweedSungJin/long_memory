#!/usr/bin/env python3
"""Fail-closed, one-variable V19 native/cache-aligned VAL160 A/B audit.

This sidecar never edits an old manifest or copies its episode CSVs. ``audit``
builds the prospective aligned evaluator identity and checks whether an original
native run may be reused. ``report`` requires two clean complete runs. The only
source differences admitted are the three reviewed candidate precision wiring
files and their new feature helper, bound to a completed regression attestation.
The ordinary V18 evaluator and baseline-reference validators are NOT relaxed.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gr00t.eval.sim.robomme.compare_long_memory_results import (
    TASKS, mcnemar_exact, paired_differences, paired_macro_bootstrap,
    read_results, validate_result_identity,
)
from run_scripts.robomme.baseline_reference_v10 import _scenario_context
from run_scripts.robomme.baseline_reference_v18 import validate_reference
from run_scripts.robomme.eval_long_memory_comparison import file_hash, resolve_repo_path
from run_scripts.robomme.eval_representation_v18 import (
    build_identity, build_parser as evaluator_parser, validate_manifest_contract,
    validate_options, verify_runtime_inputs,
)
from run_scripts.robomme.feature_precision_v19 import feature_precision_contract
from gr00t.long_memory.safety_v5 import validate_output_scope

CHANGED_SOURCES = frozenset("run_scripts/robomme/" + name for name in (
    "eval_representation_v18.py", "policy_representation_v18.py", "serve_representation_v18.py"))
ADDED_SOURCES = frozenset({"run_scripts/robomme/feature_precision_v19.py"})
REGRESSION_CHECKS = (
    "native_reproduces_legacy", "cache_aligned_features", "session_invariants", "real_ae_generation",
    "input_artifacts_unchanged")
REGRESSION_EVIDENCE = frozenset({"plan.json", "feature_comparisons.csv", "action_comparisons.csv",
                               "session_invariants.csv", "cadence_and_masks.json"})
CHECKPOINT = "runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072"
DEFAULT_NATIVE = "runs/eval/robomme/v19_fullcoverage_v1/prefix_full"
DEFAULT_BASELINE = "runs/eval/robomme/archive_read_best1250_val_n10_seed6"
DEFAULT_ALIGNED = "runs/eval/robomme/v19_precision_val160_seed6/cache_aligned"


def load_json(path):
    return json.loads(Path(path).read_text())


def fixed_protocol(manifest):
    expected = dict(tasks=TASKS, n_episodes=10, dataset="val", seed=6,
        n_action_steps=16, max_episode_steps=1300, save_videos=False, device="cuda:0")
    if manifest.get("settings") != expected:
        raise ValueError("This precision A/B is fixed to VAL16x10 seed6, action16, limit1300, no videos")
    if set(manifest.get("models", {})) != {"baseline", "memory"}:
        raise ValueError("Only original native baseline and the single prefix candidate are in scope")
    candidate = manifest["models"]["memory"]
    if Path(candidate.get("memory_checkpoint", "")).resolve() != (REPO_ROOT / CHECKPOINT).resolve():
        raise ValueError("This audit is only for the fixed V19 prefix checkpoint-006072")
    if candidate.get("step") != 6072 or candidate.get("writer_checkpoint"):
        raise ValueError("Wrong step or a new writer is outside the precision A/B")


def source_changes(before, after):
    removed = set(before) - set(after)
    added = set(after) - set(before)
    changed = {name for name in set(before) & set(after) if before[name] != after[name]}
    if removed or not added <= ADDED_SOURCES or not changed <= CHANGED_SOURCES:
        raise ValueError(f"Unreviewed source changes: removed={sorted(removed)}, added={sorted(added)}, changed={sorted(changed)}")
    return {"changed": {name: {"native": before[name], "aligned": after[name]} for name in sorted(changed)},
            "added": {name: after[name] for name in sorted(added)}}


def validate_regression(path, native_manifest_path, candidate):
    regression = load_json(path)
    if (regression.get("format_version") != 1 or regression.get("kind") != "v19_feature_precision_regression"
            or regression.get("passed") is not True):
        raise ValueError("A completed passing feature precision regression attestation is required")
    if any(regression.get("checks", {}).get(key) is not True for key in REGRESSION_CHECKS):
        raise ValueError("Regression must pass legacy native, cache features, session invariants, and real AE generation")
    if regression.get("checkpoint_files_sha256") != candidate["models"]["memory"]["checkpoint_files_sha256"]:
        raise ValueError("Regression did not exercise this exact checkpoint")
    # Historical reuse is bound to its immutable original manifest, not a bare
    # total score. Fresh-native runs may instead point to the original historical
    # source via this same attestation; both current branches share exact sources.
    native = load_json(native_manifest_path)
    historical = "feature_precision" not in native["models"]["memory"]
    if historical and regression.get("historical_native_manifest_sha256") != file_hash(native_manifest_path):
        raise ValueError("Regression is not bound to this historical native manifest")
    hashes = regression.get("source_sha256", {})
    for name in CHANGED_SOURCES | ADDED_SOURCES:
        if hashes.get(name) != candidate["source_sha256"].get(name) or hashes.get(name) != file_hash(REPO_ROOT / name):
            raise ValueError(f"Regression/source changed since execution: {name}")
    evidence_hashes = regression.get("evidence_files_sha256", {})
    if not isinstance(evidence_hashes, dict) or not REGRESSION_EVIDENCE <= set(evidence_hashes):
        raise ValueError("Regression must bind nonempty feature/action/session evidence and preregistered panel")
    for name, digest in evidence_hashes.items():
        evidence = (Path(path).parent / name).resolve()
        if not evidence.is_relative_to(Path(path).parent.resolve()) or file_hash(evidence) != digest:
            raise ValueError(f"Regression evidence changed or escapes report directory: {name}")
    return {"path": str(Path(path).resolve()), "sha256": file_hash(path), "checks": regression["checks"]}


def validate_pair(native, aligned):
    for manifest in (native, aligned):
        validate_manifest_contract(manifest)
        fixed_protocol(manifest)
    if native["models"]["memory"].get("feature_precision", "native") != "native":
        raise ValueError("Reference candidate is not native")
    if aligned["models"]["memory"].get("feature_precision") != "cache-aligned":
        raise ValueError("Candidate is not explicitly cache-aligned")
    if aligned["models"]["memory"].get("feature_precision_rules") != feature_precision_contract("cache-aligned"):
        raise ValueError("Aligned contract changed")
    ignored = {"feature_precision", "feature_precision_rules"}
    a = {k: v for k, v in native["models"]["memory"].items() if k not in ignored}
    b = {k: v for k, v in aligned["models"]["memory"].items() if k not in ignored}
    if a != b:
        raise ValueError("Candidate weights/config/memory flags differ beyond feature precision")
    for key in ("settings", "base_file_sha256", "policy_package_versions", "benchmark", "server_python",
                "robomme_python", "selection_protocol", "control_contract", "allow_initialization_checkpoints"):
        if native.get(key) != aligned.get(key) or key not in native:
            raise ValueError(f"Precision comparison changes environment/protocol: {key}")
    if native["models"]["baseline"] != aligned["models"]["baseline"]:
        raise ValueError("Original HAMLET baseline changed")
    return source_changes(native["source_sha256"], aligned["source_sha256"])


def completed_candidate(root, manifest):
    root = Path(root).resolve()
    status = load_json(root / "driver_status.json")
    if status.get("interrupted") is not False or status.get("failures") != []:
        raise ValueError(f"Driver did not complete cleanly: {root}")
    if (root / ".driver.lock").exists():
        with (root / ".driver.lock").open("rb") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Run is still being modified by an evaluator") from exc
    rows, contexts = {}, {}
    artifacts = {name: file_hash(root / name) for name in ("comparison_manifest.json", "driver_status.json")}
    for task in TASKS:
        folder = root / "memory" / task
        task_rows = read_results(folder / "simulation_results.csv", expected=10)
        if set(task_rows) != set(range(10)):
            raise ValueError(f"Incomplete task: {root}/{task}")
        context = validate_result_identity(root, "memory", task, manifest)
        scenarios, digest = _scenario_context(manifest["benchmark"]["source_and_scenario_sha256"], "val", task)
        if (context.get("scenario_metadata_sha256") != digest or context.get("memory_window") != 4
                or context.get("demo_sampling") != "backward_aligned_full_history"
                or context.get("model_config_sha256") != manifest["base_file_sha256"]["config.json"]):
            raise ValueError(f"Scenario/observation/action identity mismatch: {task}")
        for episode, row in task_rows.items():
            # Same standard-library formula as the unchanged rollout. Avoid
            # importing the simulator simply to verify a deterministic ID.
            payload = json.dumps([6, task, episode], separators=(",", ":")).encode()
            expected_seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") % (2**31)
            if row["episode_seed"] != expected_seed or int(row.get("scenario_seed", -1)) != scenarios[episode]["seed"] or not row.get("task_instruction"):
                raise ValueError(f"Scenario/inference seed identity mismatch: {task}/{episode}")
        rows[task], contexts[task] = task_rows, context
        for name in ("simulation_results.csv", "policy_manifest.json"):
            path = folder / name
            artifacts[str(path.relative_to(root))] = file_hash(path)
    if manifest.get("baseline_reference"):
        validate_reference(manifest["baseline_reference"], manifest)
    return {"rows": rows, "contexts": contexts,
            "artifacts": {"root": str(root), "files_sha256": artifacts},
            "successes": sum(row["success"] for task_rows in rows.values() for row in task_rows.values())}


def paired_summary(native, aligned, samples):
    groups, tasks, flips = [], {}, []
    for task in TASKS:
        for key in ("scenario_metadata_sha256", "model_config_sha256", "memory_window", "demo_sampling"):
            if native["contexts"][task].get(key) != aligned["contexts"][task].get(key):
                raise ValueError(f"Paired context differs: {task}/{key}")
        left, right = native["rows"][task], aligned["rows"][task]
        differences = paired_differences(left, right)
        groups.append(differences)
        tasks[task] = dict(native_successes=sum(r["success"] for r in left.values()),
            aligned_successes=sum(r["success"] for r in right.values()), n=10,
            wins=differences.count(1), losses=differences.count(-1), same=differences.count(0),
            delta=sum(differences)/10)
        for episode in range(10):
            if left[episode]["success"] != right[episode]["success"]:
                flips.append(dict(task=task, episode=episode, episode_seed=left[episode]["episode_seed"],
                    scenario_seed=left[episode]["scenario_seed"], task_instruction=left[episode]["task_instruction"],
                    native_success=left[episode]["success"], aligned_success=right[episode]["success"],
                    native_status=left[episode].get("status"), aligned_status=right[episode].get("status"),
                    native_steps=left[episode].get("steps"), aligned_steps=right[episode].get("steps")))
    values = [d for group in groups for d in group]
    wins, losses = values.count(1), values.count(-1)
    return dict(metric="closed_loop_task_success", tasks=tasks, n=160,
        native_successes=native["successes"], aligned_successes=aligned["successes"],
        wins=wins, losses=losses, same=values.count(0), delta=sum(values)/160,
        ci95=paired_macro_bootstrap(groups, samples=samples, seed=190020),
        bootstrap_samples=samples, mcnemar_exact_p=mcnemar_exact(wins, losses), flips=flips)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("audit", "report"))
    p.add_argument("--native-run", type=Path, default=Path(DEFAULT_NATIVE))
    p.add_argument("--aligned-run", type=Path, default=Path(DEFAULT_ALIGNED))
    p.add_argument("--baseline-reference", type=Path, default=Path(DEFAULT_BASELINE))
    p.add_argument("--regression-report", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.bootstrap_samples < 1000:
        raise ValueError("Use at least 1000 paired bootstrap samples")
    output = resolve_repo_path(args.output_dir)
    if output.exists():
        raise ValueError("Use a NEW output directory; existing diagnostic results are preserved")
    native_root, aligned_root = map(resolve_repo_path, (args.native_run, args.aligned_run))
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in (native_root, aligned_root)):
        raise ValueError("Comparison sidecar output must not overlap either evaluation run")
    native_path = native_root / "comparison_manifest.json"
    native_manifest = load_json(native_path)
    candidate = native_manifest.get("models", {}).get("memory", {})
    train = candidate.get("training_config", {}).get("train", {})
    cache = resolve_repo_path(Path(train.get("cache_dir", "runs/long_memory/cache_full1600_v1")))
    protected = [native_root, aligned_root, resolve_repo_path(args.baseline_reference),
        Path(args.regression_report).resolve().parent, REPO_ROOT / CHECKPOINT, cache,
        resolve_repo_path(Path(candidate.get("base_model", {}).get("path", "checkpoints/author_hamlet_robomme/checkpoint-60000")))]
    if (cache / "manifest.json").exists():
        cache_manifest = load_json(cache / "manifest.json")
        raw_data = cache_manifest.get("dataset_path") or cache_manifest.get("config", {}).get("dataset_path")
        if raw_data:
            protected.append(resolve_repo_path(Path(raw_data)))
    validate_output_scope(output, *protected)
    if args.mode == "audit":
        opts = evaluator_parser().parse_args(["--checkpoint", CHECKPOINT, "--baseline-reference", str(args.baseline_reference),
            "--models", "baseline", "memory", "--feature-precision", "cache-aligned", "--tasks", "all",
            "--dataset", "val", "--n-episodes", "10", "--seed", "6", "--output-dir", str(aligned_root)])
        validate_options(opts)
        aligned_manifest = build_identity(opts)
    else:
        aligned_manifest = load_json(aligned_root / "comparison_manifest.json")
    changes = validate_pair(native_manifest, aligned_manifest)
    attestation = validate_regression(args.regression_report, native_path, aligned_manifest)
    # Check live files/packages/protocol before execution; final report checks
    # immutable manifests/results plus file signatures again, without modifying.
    verify_runtime_inputs(aligned_manifest)
    native = completed_candidate(native_root, native_manifest)
    record = dict(format_version=1, kind="v19_precision_paired_audit", native_reuse_eligible=True,
        comparison_script_sha256=file_hash(Path(__file__)),
        native_artifacts=native["artifacts"], native_evaluation_id=native_manifest["evaluation_id"],
        aligned_evaluation_id=aligned_manifest["evaluation_id"], regression=attestation,
        allowed_source_changes=changes, native_successes=native["successes"], requested_episodes=160,
        precision_rules=feature_precision_contract("cache-aligned"),
        limitation="Simulator Python sources/scenario files/package versions are bound; large 3-D asset bytes were not hashed by the historical evaluator.")
    lines = ["# V19 feature precision A/B", "", f"Native reuse audit: PASS; {native['successes']}/160 original completed episode rows.",
        "Only candidate feature precision changes. Original HAMLET baseline remains native."]
    if args.mode == "report":
        aligned = completed_candidate(aligned_root, aligned_manifest)
        result = paired_summary(native, aligned, args.bootstrap_samples)
        record.update(aligned_artifacts=aligned["artifacts"], result=result)
        lines += ["", f"Native: {result['native_successes']}/160; aligned: {result['aligned_successes']}/160.",
            f"Native failure -> aligned success: {result['wins']}; reverse: {result['losses']}; unchanged: {result['same']}.",
            f"Paired delta: {100*result['delta']:+.3f} pp; 95% within-task paired bootstrap CI [{100*result['ci95'][0]:+.3f}, {100*result['ci95'][1]:+.3f}] pp.",
            "", "|Task|Native|Aligned|Wins/losses|", "|---|---:|---:|---:|"]
        lines += [f"|{task}|{r['native_successes']}/10|{r['aligned_successes']}/10|{r['wins']}/{r['losses']}|" for task, r in result["tasks"].items()]
        lines += ["", "Uncertainty is conditional on these 16 tasks and inference seed 6. This repeatedly used VAL160 is a development set, not confirmatory generalization evidence.",
            "A CI containing zero is NOT evidence of no effect or proof precision was not a major cause. Report the full interval and its still-compatible improvement sizes.",
            "Flip CSV contains simulator termination status/step counts, not a validated semantic failure diagnosis. No new videos were requested."]
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    (output / "aligned_manifest.json").write_text(json.dumps(aligned_manifest, indent=2, allow_nan=False) + "\n")
    if "result" in record:
        for name, rows in (("tasks.csv", [dict(task=k, **v) for k, v in record["result"]["tasks"].items()]),
                           ("flipped_episodes.csv", record["result"]["flips"])):
            if rows:
                with (output / name).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    text = "\n".join(lines) + "\n"
    (output / "report.md").write_text(text)
    print(text)
    print(f"Saved immutable sidecar: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[precision A/B] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
