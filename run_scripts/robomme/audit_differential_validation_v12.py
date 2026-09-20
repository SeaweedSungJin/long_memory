#!/usr/bin/env python3
"""Complementary196 cache-VAL audit of two COMPLETED V12 MAE-selected pilots.

Reuse the frozen V12 planner/validator and V11 paired-query statistics without
changing imported globals. Expand each saved128-query plan to324; evaluate only
the remaining196 episodes, retaining q=0 and the original two paired noises.
One frozen parent/AE, two independently loaded visual bests, native Euler4, no
optimizer/learning/model save. This is not robot accuracy or an untouched test.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from gr00t.long_memory.monitoring import _atomic_json
from run_scripts.robomme import train_visual_differential_v12 as trainer
from run_scripts.robomme import validate_visual_holdout_v11 as stats_v11

MODES = ("differential", "current_only")
ROLE = {"differential": "differential", "current_only": "current-only"}
ROLES = ("differential", "current-only", "visual-off")
PRIMARY = "generated_observed_prefix_mae"
METRICS = tuple(f"generated_observed_{part}_{error}" for part in ("prefix", "joint7", "gripper1")
                for error in ("mae", "mse")) + ("flow_action_loss",)
KEYS = ("episode_id", "decision", "repeat", "flow_seed", "generation_seed")
COMPARISONS = (("differential", "current-only"), ("differential", "visual-off"), ("current-only", "visual-off"))
LIMITATIONS = [
    "Offline normalized-action errors, not robot accuracy/success, convergence or correct historical recall.",
    "Complementary196 excludes these V12 selection128 only; V11/earlier audits or base/parent training may have exposed these episodes.",
    "One fixed query per episode with two fixed noise draws; not complete trajectories or every decision.",
    "Query/episode-cluster bootstrap assumes independent episodes and does not model task-level correlation.",
    "Step0 is an honest possible MAE-selected best, not a trained-improvement claim.",
    "No optimizer, model save, sampler change, further checkpoint selection or GPU bitwise backward claim.",
]


def source_hashes():
    result = stats_v11.sources()  # Includes its V11 trainer/transitive source closure.
    result.update(trainer.source_identity())
    result[str(Path(__file__).resolve().relative_to(ROOT))] = trainer.file_hash(__file__)
    return dict(sorted(result.items()))


def completed_run(path, mode):
    """Check completion BEFORE any real model/cache preflight or best selection."""
    if mode not in MODES:
        raise ValueError("Unknown V12 arm")
    root = Path(path).resolve()
    state = json.loads((root / "status.json").read_text())
    if (state.get("status") != "complete" or state.get("optimizer_updates") != 512
            or state.get("window_cursor") != 512 or state.get("processed_queries") != 2048):
        raise ValueError("Both V12 runs must be complete512/2048 queries before audit preflight")
    config = json.loads((root / "run_config.json").read_text())
    args = argparse.Namespace(**config["train"])
    if (config.get("trainer_variant") != trainer.DRIVER or config.get("read_mode") != mode
            or args.read_mode != mode or config.get("objective") != trainer.OBJECTIVE
            or (args.max_steps, args.query_batch_size, args.val_samples, args.val_noise_samples, args.seed) != (512, 4, 128, 2, 9111)
            or args.eval_steps != 128 or args.resume is not None):
        raise ValueError("Requires declared new512 V12 arm,128 queries/two noises/seed9111/MAE selection")
    trainer.validate_options(args)
    saved = json.loads((root / "query_plan.json").read_text())
    if saved.get("sha256") != trainer.digest({k: v for k, v in saved.items() if k != "sha256"}):
        raise ValueError("Saved immutable plan digest differs")
    best = json.loads((root / "best_checkpoint.json").read_text())
    if type(best.get("step")) is not int or best["step"] not in range(0, 513, 128):
        raise ValueError("Best must be an actual declared MAE-validation boundary, including step0")
    checkpoint = (root / best["path"]).resolve()
    if checkpoint != root / f"checkpoint-{best['step']:06d}" or Path(state["best_checkpoint"]).resolve() != checkpoint:
        raise ValueError("Completed status and best checkpoint pointer disagree or escape run")
    scores, files = {}, [root / name for name in ("status.json", "run_config.json", "query_plan.json", "best_checkpoint.json")]
    for step in range(0, 513, 128):
        file = root / f"validation-{step:06d}.json"
        validation = json.loads(file.read_text())
        if (validation["step"] != step or validation["plan_sha256"] != saved["sha256"]
                or validation["selection_metric"] != trainer.SELECTION or validation["read_mode"] != mode):
            raise ValueError("Saved validation checkpoint/mode/MAE-selection identity differs")
        score = validation["summary"]["reader"][PRIMARY]
        if not math.isfinite(score) or score < 0:
            raise ValueError("Invalid MAE selection score")
        scores[step] = score
        files.append(file)
    # Verify the frozen trainer's strict-improvement/earliest-tie decision; do
    # not select or publish any new checkpoint based on complementary196.
    if best["step"] != min(scores, key=scores.get) or scores[best["step"]] != state["best_generated_prefix_mae"]:
        raise ValueError("Pointer is not the recorded MAE-selected best")
    return {"path": root, "args": args, "config": config, "plan": saved, "checkpoint": checkpoint, "status": state,
            "best_step": best["step"], "selection_scores": scores, "files": files}


def verify_run_boundaries(base, run):
    """Bind actual V12 step0 tensors and final512 bundle, not metadata alone."""
    root, selected = run["path"], run["info"]
    last_path = root / "last_checkpoint.json"
    last = json.loads(last_path.read_text())
    if last.get("step") != 512 or Path(last["path"]).resolve() != root / "checkpoint-000512":
        raise ValueError("Last checkpoint must corroborate completed512 status")
    files, initial = [last_path], None
    for step in (0, 512):
        path = root / f"checkpoint-{step:06d}"
        info = trainer.checkpoint_info(base, path)
        if info["step"] != step or info["config"] != run["config"]:
            raise ValueError("V12 boundary bundle step/config/read_mode differs")
        for key in ("frozen_parent", "base_model", "initialization_reference", "source_sha256", "runtime", "plan_sha256", "cache_fingerprint"):
            if info["metadata"][key] != selected["metadata"][key]:
                raise ValueError(f"V12 boundary provenance changed: {key}")
        files += [path / name for name in ("checkpoint.json", "visual.safetensors", "training_state.pt")]
        if step == 0:
            initial = trainer.load_file(str(path / "visual.safetensors"), device="cpu")
        else:
            trainer.validate_resume_state(info, {k: v for k, v in run["plan"].items() if k != "sha256"}, run["plan"]["sha256"])
            if info["metadata"]["train_state"] != run["status"]:
                raise ValueError("Final512 bundle and completed status bookkeeping differ")
    reference = selected["metadata"]["initialization_reference"]
    original = trainer.load_file(str(Path(reference["path"]) / "checkpoint-000000/visual.safetensors"), device="cpu")
    if set(initial) != set(original) or any(initial[key].dtype != original[key].dtype
            or not torch.equal(initial[key], original[key]) for key in original):
        raise ValueError("Actual V12 step0 tensors do not equal the declared original V11 initialization")
    return files, {"actual_v12_step0_equals_v11_reference_tensors": True, "last512_bundle_matches_completed_status": True}


def expanded_plan(args, cache, episodes, saved):
    """Exact original128/256 prefix and entire training structure; q=0 retained."""
    original = {k: v for k, v in saved.items() if k != "sha256"}
    if (args.val_samples, args.val_noise_samples, args.seed) != (128, 2, 9111):
        raise ValueError("Requires saved128/two-noise/seed9111 plan")
    rebuilt, digest = trainer.build_plan(args, cache, episodes)
    if rebuilt != original or digest != saved.get("sha256") or trainer.digest(original) != digest:
        raise ValueError("Original128 plan does not rebuild exactly")
    expanded_args = copy.deepcopy(args)
    expanded_args.val_samples = len(cache.manifest["splits"]["val"])
    expanded, sha = trainer.build_plan(expanded_args, cache, episodes)
    if expanded["validation"][:128] != original["validation"] or expanded["validation_schedule"][:256] != original["validation_schedule"]:
        raise ValueError("Expanded first128 pairs/first256 noise draws changed")
    if any(value != original[key] for key, value in expanded.items() if key not in ("validation", "validation_schedule")):
        raise ValueError("Expanded validation changed training schedule or structural identity")
    extra = {"validation": expanded["validation"][128:], "validation_schedule": expanded["validation_schedule"][256:]}
    selected, excluded = {eid for eid, _ in extra["validation"]}, {eid for eid, _ in original["validation"]}
    if (not selected or len(selected) != len(extra["validation"]) or len(excluded) != 128
            or selected & excluded or selected | excluded != set(cache.manifest["splits"]["val"])
            or selected & set(cache.manifest["splits"]["train"])):
        raise ValueError("Complement must contain every remaining distinct held-out episode")
    if [(r["episode_id"], r["decision"], r["repeat"]) for r in extra["validation_schedule"]] != [
            (eid, q, repeat) for eid, q in extra["validation"] for repeat in range(2)]:
        raise ValueError("Complementary query/noise schedule differs")
    return expanded, sha, extra


def validate_pair_identity(runs):
    left, right = (runs[mode] for mode in MODES)
    if left["plan"] != right["plan"]:
        raise ValueError("Two arms must share exact training/validation/noise/cache plans")
    allowed = {"output_dir", "read_mode"}
    if {k: v for k, v in vars(left["args"]).items() if k not in allowed} != {
            k: v for k, v in vars(right["args"]).items() if k not in allowed}:
        raise ValueError("Two arms differ beyond declared read_mode/output directory")
    for key in ("frozen_parent", "base_model", "initialization_reference", "source_sha256", "runtime", "plan_sha256", "cache_fingerprint"):
        if left["info"]["metadata"][key] != right["info"]["metadata"][key]:
            raise ValueError(f"Two arms have mismatched {key}")


def merge_query_rows(by_mode, schedule):
    """Keep three roles only AFTER both independent off copies match exactly."""
    expected = {tuple(item[k] for k in KEYS) + (role,) for item in schedule for role in ("reader", "visual-off")}
    indexed = {}
    for mode in MODES:
        indexed[mode] = {tuple(row[k] for k in KEYS) + (row["role"],): row for row in by_mode[mode]}
        if len(indexed[mode]) != len(by_mode[mode]) or set(indexed[mode]) != expected:
            raise ValueError("Missing/duplicate/mismatched query, role, repeat or noise")
    records, proofs = [], []
    for item in schedule:
        key = tuple(item[k] for k in KEYS)
        off = indexed[MODES[0]][key + ("visual-off",)]
        other = indexed[MODES[1]][key + ("visual-off",)]
        if off != other:
            raise ValueError("Independent visual-off copies are not exactly equal")
        proofs.append({**item, "off_copies_exact": True, "row_sha256": trainer.digest(off)})
        for mode in MODES:
            row = indexed[mode][key + ("reader",)]
            if row.get("generated_valid_values") != off.get("generated_valid_values"):
                raise ValueError("Paired roles have different observed-coordinate masks/counts")
            if item["decision"] == 0 and any(row.get(metric) != off.get(metric) for metric in METRICS):
                raise ValueError("q=0 must preserve original paired metrics in both modes")
            records.append({**row, "role": ROLE[mode]})
        records.append(off)
    return records, proofs


def paired_statistics(records, plan, *, bootstrap_draws=stats_v11.BOOTSTRAP_DRAWS):
    """Pure adapter to reviewed V11 statistics: left=reader, right=visual-off.

    Each requested error scalar is copied into its required primary scalar
    field solely inside the statistics call, then labeled with its real metric.
    No imported module constant/function is changed. Gain>0 means LEFT better.
    """
    if any(role not in ROLES for role in (row["role"] for row in records)):
        raise ValueError("Unexpected audit role")
    expected = {tuple(row[k] for k in KEYS) + (role,) for row in plan["validation_schedule"] for role in ROLES}
    actual = [tuple(row[k] for k in KEYS) + (row["role"],) for row in records]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Incomplete three-role pairing or noise mismatch")
    def rename(value):
        if isinstance(value, dict):
            return {key.replace("visual_minus_off", "left_minus_right"): rename(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rename(item) for item in value]
        return value
    result = {"primary_metric": PRIMARY, "comparisons": {}}
    for left, right in COMPARISONS:
        compared = {"left_role": left, "right_role": right, "gain_definition": "100*(right_mean-left_mean)/right_mean", "metrics": {}}
        for metric in METRICS:
            if not all(metric in row for row in records):
                raise ValueError(f"Missing required paired metric: {metric}")
            copied = [{**{key: row[key] for key in KEYS}, "role": "reader" if row["role"] == left else "visual-off",
                       stats_v11.METRICS[0]: row[metric]} for row in records if row["role"] in (left, right)]
            values = stats_v11.paired_statistics(copied, plan, bootstrap_draws=bootstrap_draws)
            summary = rename(values["metrics"][stats_v11.METRICS[0]])
            for key in ("equal_query_mean", "query_median"):
                summary[key] = {left: summary[key]["reader"], right: summary[key]["visual-off"]}
            summary["per_query"] = [{"episode_id": row["episode_id"], "decision": row["decision"],
                left: row["metrics"][stats_v11.METRICS[0]]["reader"],
                right: row["metrics"][stats_v11.METRICS[0]]["visual-off"],
                "left_minus_right": row["metrics"][stats_v11.METRICS[0]]["visual_minus_off"]} for row in values["per_query"]]
            compared["metrics"][metric] = summary
        result["comparisons"][left + "_vs_" + right] = compared
    return result


def preflight(args):
    # Status checks precede heavyweight identity/planning work for EITHER run.
    paths = {"differential": args.differential_run, "current_only": args.current_only_run}
    for path in paths.values():
        if json.loads((Path(path) / "status.json").read_text()).get("status") != "complete":
            raise ValueError("Both runs must be complete before real audit preflight")
    runs = {mode: completed_run(path, mode) for mode, path in paths.items()}
    cache = trainer.EpisodeCache(runs[MODES[0]]["args"].cache_dir)
    trainer.validate_cache_checkpoint(cache.manifest)
    base = Path(cache.manifest["model_path"]).resolve()
    episodes = trainer.MappedEpisodes(cache)
    for mode, run in runs.items():
        info = trainer.checkpoint_info(base, run["checkpoint"])
        run["info"] = info
        metadata = info["metadata"]
        if (info["step"] != run["best_step"] or info["config"] != run["config"] or metadata["read_mode"] != mode
                or metadata["selection_metric"] != trainer.SELECTION or metadata["plan_sha256"] != run["plan"]["sha256"]
                or metadata["cache_fingerprint"] != cache.manifest["fingerprint"]
                or metadata["cache_manifest_sha256"] != trainer.digest(cache.manifest)
                or metadata["source_sha256"] != trainer.source_identity() or metadata["runtime"] != trainer.runtime_identity()):
            raise ValueError("Selected bundle mode/config/source/runtime/cache/MAE-plan identity differs")
        original = {k: v for k, v in run["plan"].items() if k != "sha256"}
        reference = trainer.reference_contract(run["args"], base, cache, metadata["frozen_parent"],
            trainer.VisualDifferentialConfig(**info["config"]["visual"]), original)
        if reference != metadata["initialization_reference"]:
            raise ValueError("Original common V11 initialization/reference plan changed")
        boundary_files, run["boundary_checks"] = verify_run_boundaries(base, run)
        run["files"] += boundary_files
        run["expanded"], run["expanded_sha"], run["extra"] = expanded_plan(run["args"], cache, episodes, run["plan"])
    validate_pair_identity(runs)
    left, right = (runs[mode] for mode in MODES)
    if left["expanded"] != right["expanded"] or left["extra"] != right["extra"]:
        raise ValueError("Two complementary196 plans differ")
    extra, expanded = left["extra"], left["expanded"]
    if len(cache.manifest["splits"]["val"]) != 324 or len(extra["validation"]) != 196 or cache.manifest["action_steps"] != 16:
        raise ValueError("Requires real324 minus selection128 equals196,16-step prefix")
    parent = left["info"]["metadata"]["frozen_parent"]
    output = Path(args.output_dir).resolve()
    trainer.validate_output_scope(output, cache.path, cache.manifest.get("dataset_path"), base, parent["path"],
                                  *(run["path"] for run in runs.values()), left["args"].reference_run)
    if output.exists():
        raise FileExistsError("Use a NEW complementary196 audit directory")
    files = [cache.path / "manifest.json"]
    files += [base / name for name in left["info"]["metadata"]["base_model"]["shards"]]
    files += [path for path in base.rglob("*") if path.is_file() and path.suffix in (".json", ".model", ".txt")]
    files += [Path(parent["path"]) / name for name in parent["files_sha256"]]
    files += [Path(expanded["files"][str(eid)][0]) for eid, _ in extra["validation"]]
    for run in runs.values():
        files += run["files"] + [run["checkpoint"] / name for name in ("checkpoint.json", "visual.safetensors", "training_state.pt")]
        ref = run["info"]["metadata"]["initialization_reference"]
        files += [Path(ref["path"]) / name for name in ref["files_sha256"]]
    record = {"diagnostic": "visual_differential_v12_extra196", "selection_metric": trainer.SELECTION,
        "primary_metric": PRIMARY, "selected_bests": {mode: {"run": str(run["path"]), "checkpoint": str(run["checkpoint"]),
            "step": run["best_step"], "read_mode": mode, "selection_scores": run["selection_scores"],
            "boundary_checks": run["boundary_checks"]} for mode, run in runs.items()},
        "excluded_selection128": left["plan"]["validation"], "original_plan_sha256": left["plan"]["sha256"],
        "expanded_plan_sha256": left["expanded_sha"], "validation": extra["validation"],
        "validation_schedule": extra["validation_schedule"], "q0_retained": sum(q == 0 for _, q in extra["validation"]),
        "files_sha256": {str(path): trainer.file_hash(path) for path in sorted(set(files))},
        "source_sha256": source_hashes(), "base_identity": trainer.checkpoint_identity(base), "frozen_parent": parent,
        "initialization_reference": left["info"]["metadata"]["initialization_reference"], "runtime": trainer.runtime_identity(),
        "roles": ROLES, "device": args.device, "sampler": "native_original_Euler4", "optimizer_updates": 0,
        "bootstrap_seed": stats_v11.BOOTSTRAP_SEED, "bootstrap_draws": stats_v11.BOOTSTRAP_DRAWS, "limitations": LIMITATIONS}
    record["audit_id"] = trainer.digest(record)
    return runs, base, episodes, extra, record


@torch.no_grad()
def run_validation(args, runs, base, episodes, extra, result, persist):
    first = runs[MODES[0]]
    parent_path = first["info"]["metadata"]["frozen_parent"]["path"]
    cfg = trainer.v7_checkpoint_info(base, parent_path, expected_stage=1)["config"]
    head = trainer.actual_head(base, args.device)
    with trainer.isolated_seed(9211, args.device):
        parent = trainer.RecurrentMemoryV7(trainer.MemoryV7Config(**cfg["memory"])).to(args.device)
        cvom = trainer.CVOMV7(parent.config).to(args.device)
        trainer.install_expert_lora(head, trainer.LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
        models = {mode: trainer.VisualDifferentialMemoryV12(trainer.VisualDifferentialConfig(**run["config"]["visual"]),
            read_mode=mode).to(args.device) for mode, run in runs.items()}
    trainer.load_checkpoint_v7(parent_path, parent, head, cvom)
    for mode, visual in models.items():
        loaded = trainer.load_checkpoint(runs[mode]["checkpoint"], visual)
        if any(loaded[key] != runs[mode]["info"][key] for key in ("step", "config", "metadata")):
            raise ValueError("Selected visual bundle changed after preflight")
    trainer.set_expert_trainable(head, False)
    modules = {"head": head, "parent": parent, "cvom": cvom, **models}
    for module in modules.values():
        module.eval().requires_grad_(False)
    if head.num_inference_timesteps != 4 or next(head.parameters()).dtype != torch.bfloat16:
        raise ValueError("Original BF16 head/native Euler4 required")
    guard = stats_v11.frozen_guard(tuple(modules.values()))
    before = {name: stats_v11.module_digest(module) for name, module in modules.items()}
    result["module_before_sha256"] = before
    try:
        for index, (eid, query) in enumerate(extra["validation"]):
            schedule = extra["validation_schedule"][index * 2:index * 2 + 2]
            by_mode = {mode: trainer.validate(runs[mode]["args"], model, parent, head, episodes,
                {"validation_schedule": schedule})[1] for mode, model in models.items()}
            rows, proofs = merge_query_rows(by_mode, schedule)
            result["records"].extend(rows)
            result["off_duplicate_checks"].extend(proofs)
            stats_v11.frozen_guard(tuple(modules.values()))
            if any(value._version != version for value, version in guard):
                raise RuntimeError("Frozen module tensor version changed")
            persist()
            print(f"[v12-extra196] {index+1}/{len(extra['validation'])} episode={eid} q={query}; off copies exact", flush=True)
        result["statistics"] = paired_statistics(result["records"], extra)
    finally:
        result["module_after_sha256"] = {name: stats_v11.module_digest(module) for name, module in modules.items()}
        result["checks"].update(frozen_modules_unchanged=before == result["module_after_sha256"],
            frozen_versions_unchanged=all(value._version == version for value, version in guard),
            no_parameter_gradients=all(not p.requires_grad and p.grad is None for module in modules.values() for p in module.parameters()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--differential-run", required=True)
    parser.add_argument("--current-only-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if not args.preflight_only and (args.device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1")):
        raise ValueError("Actual run requires root-authorized one-GPU visibility and cuda:0")
    runs, base, episodes, extra, record = preflight(args)
    if args.preflight_only:
        print(json.dumps({"preflight_passed": True, "audit_id": record["audit_id"], "selected_bests": record["selected_bests"],
            "queries": len(extra["validation"]), "paired_draws": len(extra["validation_schedule"]),
            "q0_retained": record["q0_retained"], "cuda_initialized": torch.cuda.is_initialized(), "output_created": False}, indent=2))
        return 0
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "plan.json", record)
    _atomic_json(output / "expanded_query_plan.json", {"sha256": record["expanded_plan_sha256"], **runs[MODES[0]]["expanded"]})
    result = {"passed": False, "audit_id": record["audit_id"], "records": [], "off_duplicate_checks": [],
              "checks": {}, "optimizer_updates": 0, "limitations": LIMITATIONS}
    persist = lambda: _atomic_json(output / "result.json", result)
    started = time.monotonic()
    try:
        run_validation(args, runs, base, episodes, extra, result, persist)
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(result["error"]["traceback"], flush=True)
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["files_after_sha256"] = {path: trainer.file_hash(path) for path in record["files_sha256"]}
        result["sources_after_sha256"] = source_hashes()
        result["checks"].update(files_unchanged=result["files_after_sha256"] == record["files_sha256"],
            sources_unchanged=result["sources_after_sha256"] == record["source_sha256"],
            complete196_queries392_paired_draws=len(result["records"]) == 1176,
            all_duplicate_off_copies_exact=len(result["off_duplicate_checks"]) == 392 and all(
                row["off_copies_exact"] for row in result["off_duplicate_checks"]))
        result["passed"] = "error" not in result and all(result["checks"].values())
        persist()
    print(json.dumps({"passed": result["passed"], "checks": result["checks"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
