#!/usr/bin/env python3
"""Frozen V11-384 versus visual-off on validation episodes excluded from selection.

Thin wrapper around the unchanged trainer's plan builder and validation. The
original 32 queries/two noises are an exact prefix of an expanded all-validation
plan; only the remaining episodes are evaluated. q=0 is retained. No optimizer,
weight save, epoch, inference configuration change, or target-dependent search.
Query-cluster bootstrap resamples paired two-noise query means, not individual
draws. This is offline normalized-action error, never robot accuracy. Holdout is
relative to this V11 selection; original base/parent exposure is not excluded.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from gr00t.long_memory.monitoring import _atomic_json
from run_scripts.robomme import train_visual_patch_v11 as trainer
from run_scripts.robomme.verify_visual_patch_v11 import module_digest
from run_scripts.robomme.visual_patch_memory_v11 import VisualPatchConfig, VisualPatchMemoryV11

ROLES = ("reader", "visual-off")
METRICS = ("generated_observed_prefix_mse", "generated_observed_prefix_mae",
           "generated_observed_joint7_mse", "generated_observed_gripper1_mse", "flow_action_loss")
BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 108284, 10000
LIMITATIONS = [
    "Offline normalized-action errors, not robot success/accuracy or correct historical recall proof.",
    "Holdout excludes this V11 model-selection32; base/parent pretraining or earlier validation exposure is not excluded.",
    "One fixed query per remaining cache-val episode, with the original two paired noise seeds; not all decisions.",
    "Paired percentile bootstrap treats episodes/query clusters as independent; task-level dependence is not modeled.",
    "No training, checkpoint selection, optimizer, saved weights, simulator, or GPU bitwise-equivalence claim.",
]


def sources():
    result = trainer.source_identity()
    for name in ("validate_visual_holdout_v11.py", "verify_visual_patch_v11.py"):
        path = ROOT / "run_scripts/robomme" / name
        result[str(path.relative_to(ROOT))] = trainer.file_hash(path)
    return dict(sorted(result.items()))


def expanded_holdout(saved_args, cache, episodes, saved):
    """Rebuild exact original plan, then extend only val_samples; retain q=0."""
    if (saved_args.seed, saved_args.val_samples, saved_args.val_noise_samples) != (9111, 32, 2):
        raise ValueError("Requires saved seed9111 /32 queries /2 noises")
    original = {key: value for key, value in saved.items() if key != "sha256"}
    rebuilt, original_sha = trainer.build_plan(saved_args, cache, episodes)
    if saved.get("sha256") != trainer.digest(original) or original_sha != saved["sha256"] or rebuilt != original:
        raise ValueError("Original immutable query/noise/cache plan does not rebuild exactly")
    expanded_args = copy.deepcopy(saved_args)
    expanded_args.val_samples = len(cache.manifest["splits"]["val"])
    expanded, expanded_sha = trainer.build_plan(expanded_args, cache, episodes)
    if expanded["validation"][:32] != original["validation"]:
        raise ValueError("Expanded first32 validation pairs differ")
    if expanded["validation_schedule"][:64] != original["validation_schedule"]:
        raise ValueError("Expanded first64 validation noise draws differ")
    # Also bind the ENTIRE training schedule and all other structural metadata.
    if any(value != original[key] for key, value in expanded.items()
           if key not in ("validation", "validation_schedule")):
        raise ValueError("Expanding validation changed original training schedule/structure")
    pairs = expanded["validation"][32:]
    selected = {eid for eid, _ in pairs}
    excluded = {eid for eid, _ in original["validation"]}
    val = set(cache.manifest["splits"]["val"])
    if (not pairs or len(selected) != len(pairs) or selected & excluded
            or selected | excluded != val or selected & set(cache.manifest["splits"]["train"])):
        raise ValueError("Expanded holdout must be all remaining distinct cache-val episodes")
    holdout = {"validation": pairs, "validation_schedule": expanded["validation_schedule"][64:]}
    if [(r["episode_id"], r["decision"], r["repeat"]) for r in holdout["validation_schedule"]] != [
            (eid, query, repeat) for eid, query in pairs for repeat in range(2)]:
        raise ValueError("Holdout schedule must contain original two ordered paired noises")
    return expanded, expanded_sha, holdout


def paired_statistics(records, plan, *, bootstrap_draws=BOOTSTRAP_DRAWS):
    """Equal query means; gain >0 means lower error with visual history enabled."""
    keys = ("episode_id", "decision", "repeat", "flow_seed", "generation_seed")
    indexed = {}
    for row in records:
        key = tuple(row[name] for name in keys) + (row["role"],)
        if key in indexed:
            raise ValueError("Duplicate paired validation record")
        indexed[key] = row
    expected = {tuple(item[name] for name in keys) + (role,)
                for item in plan["validation_schedule"] for role in ROLES}
    if set(indexed) != expected:
        raise ValueError("Missing/extra role, query, repeat or paired noise seed")
    by_query, summary = [], {}
    for eid, query in plan["validation"]:
        items = [item for item in plan["validation_schedule"] if (item["episode_id"], item["decision"]) == (eid, query)]
        if len(items) != 2 or {item["repeat"] for item in items} != {0, 1}:
            raise ValueError("Each query requires exactly two fixed noises")
        entry = {"episode_id": eid, "decision": query, "metrics": {}}
        for metric in METRICS:
            rows = {role: [indexed[tuple(item[name] for name in keys) + (role,)] for item in items] for role in ROLES}
            if not all(metric in row for values in rows.values() for row in values):
                if metric == METRICS[0]:
                    raise ValueError("Missing primary observed-prefix MSE")
                continue
            values = {role: statistics.mean(row[metric] for row in rows[role]) for role in ROLES}
            if any(not math.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError("Metric must be finite nonnegative")
            entry["metrics"][metric] = {**values, "visual_minus_off": values["reader"] - values["visual-off"]}
        by_query.append(entry)
    for metric in METRICS:
        available = [row for row in by_query if metric in row["metrics"]]
        if not available:
            continue
        visual = torch.tensor([row["metrics"][metric]["reader"] for row in available], dtype=torch.float64)
        off = torch.tensor([row["metrics"][metric]["visual-off"] for row in available], dtype=torch.float64)
        delta = visual - off
        count = len(available)
        def gain(v, o):
            return float((o - v) / o * 100) if o > 0 else None
        loo = [{"excluded_episode_id": row["episode_id"], "excluded_decision": row["decision"],
                "mean_visual_minus_off": float((delta.sum() - delta[index]) / (count - 1)),
                "relative_mean_gain_percent": gain(visual.sum() - visual[index], off.sum() - off[index])}
               for index, row in enumerate(available)] if count > 1 else []
        generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
        index = torch.randint(count, (bootstrap_draws, count), generator=generator)
        vb, ob = visual[index].mean(1), off[index].mean(1)
        db = vb - ob
        relative = (ob - vb) / ob * 100
        ci = lambda values: torch.quantile(values, torch.tensor([.025, .975], dtype=torch.float64)).tolist()
        summary[metric] = {"queries": count, "equal_query_mean": {"reader": float(visual.mean()), "visual-off": float(off.mean())},
            "query_median": {"reader": float(visual.median()) if count % 2 else statistics.median(visual.tolist()),
                             "visual-off": statistics.median(off.tolist())},
            "mean_visual_minus_off": float(delta.mean()), "median_visual_minus_off": statistics.median(delta.tolist()),
            "relative_mean_gain_percent": gain(visual.mean(), off.mean()),
            "improved": int((delta < 0).sum()), "worse": int((delta > 0).sum()), "tied": int((delta == 0).sum()),
            "leave_one_out": loo,
            "leave_one_out_delta_range": [min(row["mean_visual_minus_off"] for row in loo), max(row["mean_visual_minus_off"] for row in loo)] if loo else None,
            "paired_query_bootstrap": {"seed": BOOTSTRAP_SEED, "draws": bootstrap_draws,
                "sampling_unit": "query/episode with both roles and both noises retained",
                "mean_visual_minus_off_95pct_percentile_interval": ci(db),
                "relative_mean_gain_percent_95pct_percentile_interval": ci(relative) if bool(torch.isfinite(relative).all()) else None,
                "fraction_replicates_visual_better": float((db < 0).double().mean()),
                "note": "Percentile uncertainty diagnostic, not a causal test or robot accuracy interval."}}
    return {"per_query": by_query, "metrics": summary}


def frozen_guard(modules):
    if torch.is_grad_enabled() or any(module.training for module in modules):
        raise RuntimeError("All audit execution must be no-grad/eval")
    if any(p.requires_grad or p.grad is not None for module in modules for p in module.parameters()):
        raise RuntimeError("All audit parameters must be frozen and gradient-free")
    return [(value, value._version) for module in modules
            for value in list(module.parameters()) + list(module.buffers())]


def preflight(args):
    run = Path(args.training_run).resolve()
    saved_config = json.loads((run / "run_config.json").read_text())
    saved_plan = json.loads((run / "query_plan.json").read_text())
    saved_args = argparse.Namespace(**saved_config["train"])
    trainer.validate_options(saved_args)
    cache = trainer.EpisodeCache(saved_args.cache_dir)
    trainer.validate_cache_checkpoint(cache.manifest)
    base = Path(cache.manifest["model_path"]).resolve()
    checkpoint = run / "checkpoint-000384"
    info = trainer.checkpoint_info(base, checkpoint)
    if (info["step"] != 384 or info["config"] != saved_config
            or info["metadata"]["plan_sha256"] != saved_plan["sha256"]
            or info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]
            or info["metadata"]["cache_manifest_sha256"] != trainer.digest(cache.manifest)
            or cache.manifest["action_steps"] != 16):
        raise ValueError("Checkpoint384/run/cache/plan protocol differs")
    if info["metadata"]["source_sha256"] != trainer.source_identity() or info["metadata"]["runtime"] != trainer.runtime_identity():
        raise ValueError("Frozen trainer source/runtime differs from actual pilot")
    best = json.loads((run / "best_checkpoint.json").read_text())
    if best.get("step") != 384 or (run / best["path"]).resolve() != checkpoint:
        raise ValueError("Predeclared best384 identity changed")
    output = Path(args.output_dir).resolve()
    trainer.validate_output_scope(output, run, cache.path, cache.manifest.get("dataset_path"), base,
                                  info["metadata"]["frozen_parent"]["path"])
    if output.exists():
        raise FileExistsError("Use a NEW holdout audit output directory")
    episodes = trainer.MappedEpisodes(cache)
    expanded, expanded_sha, holdout = expanded_holdout(saved_args, cache, episodes, saved_plan)
    if len(cache.manifest["splits"]["val"]) != 324 or len(holdout["validation"]) != 292:
        raise ValueError("Predeclared real-cache scope must be324 minus32 equals292 episodes")
    files = [run / name for name in ("run_config.json", "query_plan.json", "best_checkpoint.json")]
    files += [checkpoint / name for name in ("checkpoint.json", "visual.safetensors", "training_state.pt")]
    files += [cache.path / "manifest.json"]
    files += [base / name for name in info["metadata"]["base_model"]["shards"]]
    files += [path for path in base.rglob("*") if path.is_file() and path.suffix in (".json", ".model", ".txt")]
    parent = info["metadata"]["frozen_parent"]
    files += [Path(parent["path"]) / name for name in parent["files_sha256"]]
    # Full selected-cache bytes are bound once before and once after, not fed to memory.
    files += [Path(expanded["files"][str(eid)][0]) for eid, _ in holdout["validation"]]
    hashes = {str(path): trainer.file_hash(path) for path in sorted(set(files))}
    record = {"format_version": 1, "diagnostic": "visual_holdout292_v11", "checkpoint": str(checkpoint),
        "checkpoint_step": 384, "original_plan_sha256": saved_plan["sha256"], "expanded_plan_sha256": expanded_sha,
        "excluded_original_validation": saved_plan["validation"], "validation": holdout["validation"],
        "validation_schedule": holdout["validation_schedule"], "q0_retained": sum(q == 0 for _, q in holdout["validation"]),
        "same_original_first32_pairs_first64_validation_draws_and_all_training_schedule": True,
        "source_sha256": sources(), "files_sha256": hashes, "base_identity": trainer.checkpoint_identity(base),
        "frozen_parent": parent, "runtime": trainer.runtime_identity(), "saved_train_args": vars(saved_args),
        "device": args.device, "roles": list(ROLES), "sampler": "native_original_Euler4", "noise_samples": 2,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "optimizer_updates": 0, "limitations": LIMITATIONS}
    record["audit_id"] = trainer.digest(record)
    return saved_args, cache, base, info, episodes, expanded, holdout, record


@torch.no_grad()
def run_validation(args, train_args, base, info, episodes, holdout, result, persist):
    parent_info = trainer.v7_checkpoint_info(base, info["metadata"]["frozen_parent"]["path"], expected_stage=1)
    head = trainer.actual_head(base, args.device)
    cfg = parent_info["config"]
    with trainer.isolated_seed(train_args.seed + 100, args.device):
        parent = trainer.RecurrentMemoryV7(trainer.MemoryV7Config(**cfg["memory"])).to(args.device)
        cvom = trainer.CVOMV7(parent.config).to(args.device)
        trainer.install_expert_lora(head, trainer.LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
        visual = VisualPatchMemoryV11(VisualPatchConfig(**info["config"]["visual"])).to(args.device)
    trainer.load_checkpoint_v7(info["metadata"]["frozen_parent"]["path"], parent, head, cvom)
    loaded = trainer.load_checkpoint(Path(args.training_run).resolve() / "checkpoint-000384", visual)
    if any(loaded[key] != info[key] for key in ("step", "config", "metadata")):
        raise ValueError("Visual bundle changed after preflight")
    trainer.set_expert_trainable(head, False)
    modules = {"head": head, "parent": parent, "cvom": cvom, "visual": visual}
    for module in modules.values():
        module.eval().requires_grad_(False)
    if head.num_inference_timesteps != 4 or next(head.parameters()).dtype != torch.bfloat16:
        raise ValueError("Actual original BF16 head/native Euler4 required")
    guard = frozen_guard(tuple(modules.values()))
    before = {name: module_digest(module) for name, module in modules.items()}
    result["module_before_sha256"] = before
    try:
        for index, pair in enumerate(holdout["validation"]):
            plan = {"validation_schedule": holdout["validation_schedule"][index * 2:index * 2 + 2]}
            _, rows = trainer.validate(train_args, visual, parent, head, episodes, plan)
            result["records"].extend(rows)
            frozen_guard(tuple(modules.values()))
            if any(value._version != version for value, version in guard):
                raise RuntimeError("Frozen parameter/buffer version changed")
            if pair[1] == 0 and any(row["image_residual_norm"] != 0 or row["image_changed_fraction"] != 0 for row in rows):
                raise RuntimeError("Empty-history image residual must remain zero")
            persist()
            print(f"[visual-holdout] {index+1}/{len(holdout['validation'])} episode={pair[0]} q={pair[1]}", flush=True)
        result["statistics"] = paired_statistics(result["records"], holdout)
    finally:
        result["module_after_sha256"] = {name: module_digest(module) for name, module in modules.items()}
        result["checks"].update(frozen_module_hashes_unchanged=before == result["module_after_sha256"],
            frozen_versions_unchanged=all(value._version == version for value, version in guard),
            no_parameter_gradients=all(p.grad is None and not p.requires_grad for module in modules.values() for p in module.parameters()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if not args.preflight_only and (args.device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != "1"):
        raise ValueError("Actual audit requires explicitly authorized isolated GPU1 as CUDA_VISIBLE_DEVICES=1 / cuda:0")
    train_args, cache, base, info, episodes, expanded, holdout, record = preflight(args)
    if args.preflight_only:
        print(json.dumps({"preflight_passed": True, "audit_id": record["audit_id"], "queries": len(holdout["validation"]),
            "paired_draws": len(holdout["validation_schedule"]), "q0_retained": record["q0_retained"],
            "expanded_plan_sha256": record["expanded_plan_sha256"], "cuda_initialized": torch.cuda.is_initialized(),
            "note": "Read-only; no output, actual head, optimizer or GPU allocation created"}, indent=2))
        return 0
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "plan.json", record)
    _atomic_json(output / "expanded_query_plan.json", {"sha256": record["expanded_plan_sha256"], **expanded})
    result = {"passed": False, "audit_id": record["audit_id"], "records": [], "checks": {}, "optimizer_updates": 0,
              "limitations": LIMITATIONS}
    persist = lambda: _atomic_json(output / "result.json", result)
    started = time.monotonic()
    try:
        run_validation(args, train_args, base, info, episodes, holdout, result, persist)
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(result["error"]["traceback"], flush=True)
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["files_after_sha256"] = {path: trainer.file_hash(path) for path in record["files_sha256"]}
        result["sources_after_sha256"] = sources()
        result["checks"].update(files_unchanged=result["files_after_sha256"] == record["files_sha256"],
            sources_unchanged=result["sources_after_sha256"] == record["source_sha256"],
            complete_292_queries_584_paired_draws=len(result["records"]) == 1168)
        result["passed"] = "error" not in result and all(result["checks"].values())
        persist()
    print(json.dumps({"passed": result["passed"], "checks": result["checks"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
