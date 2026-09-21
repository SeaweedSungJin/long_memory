#!/usr/bin/env python3
"""Fixed-panel V19 precision regression, without training or simulator rollout.

The plan is written BEFORE loading a model. Feature extraction replays all
canonical teacher-trajectory observations; denoising runs only at the two
prespecified valid action endpoints per episode. This is not a success test.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gr00t.long_memory.cache import CacheConfig, EpisodeCache, _episode_records, _extract_episode
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.diagnose_v19_runtime import (
    cached_stages, cpu, gpu_snapshot, write_csv, write_json,
)
from run_scripts.robomme.verify_demo_tail_v13 import numerical_comparison, sha

POLICY = "run_scripts/robomme/policy_representation_v18.py"
SOURCES = (POLICY, "run_scripts/robomme/serve_representation_v18.py",
           "run_scripts/robomme/eval_representation_v18.py",
           "run_scripts/robomme/feature_precision_v19.py")
CHECKPOINT_FILES = ("checkpoint.json", "model.safetensors", "expert.safetensors")
STAGES = ("moment", "features", "short", "state", "encoded", "bank_before", "bank_after", "fused",
          "attention_masks", "image_masks")


def panel_from_metadata(rows):
    """Select by chronology ONLY. Never consult losses, labels or successes."""
    by_id = {row["episode_id"]: row for row in rows}
    if 1355 not in by_id:
        raise ValueError("Prior diagnostic episode 1355 is not in cache-VAL")
    long = [row for row in rows if row["episode_id"] != 1355 and row["demo_events"] > 32]
    if not long:
        raise ValueError("No additional cache-VAL long demo exceeding FIFO32")
    longest = sorted(long, key=lambda row: (-row["demo_events"], -row["events"], row["episode_id"]))[0]
    remaining = [row for row in rows if row["episode_id"] not in (1355, longest["episode_id"])]
    shortest = sorted(remaining, key=lambda row: (row["demo_events"], row["events"], row["episode_id"]))[0]
    return [dict(by_id[1355], selection="previously diagnosed reference"),
            dict(longest, selection="maximum demo events; ties by length then ID; >32 prior events"),
            dict(shortest, selection="minimum demo events; ties by length then ID")]


def make_panel(cache):
    rows = []
    val_ids = set(cache.manifest["splits"]["val"])
    for record in cache.manifest["episodes"]:
        if record["episode_id"] not in val_ids:
            continue
        path = cache.path / record["path"]
        ep = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        decisions = ep["decision_mask"].nonzero().flatten().tolist()
        if not decisions:
            continue
        rows.append({"episode_id": int(record["episode_id"]), "task": record["task"],
                     "demo_events": int(ep["is_demo"].sum()), "events": len(ep["frames"]),
                     "frames": ep["frames"].tolist(),
                     "action_decisions": sorted({decisions[0], decisions[-1]}),
                     "cache_episode": str(path.resolve())})
    return panel_from_metadata(rows)


def legacy_source(ref, manifest_path):
    content = subprocess.check_output(["git", "show", f"{ref}:{POLICY}"], cwd=ROOT)
    digest = hashlib.sha256(content).hexdigest()
    expected = json.loads(Path(manifest_path).read_text())["source_sha256"][POLICY]
    if digest != expected:
        raise ValueError("Historical policy source does not match the original native evaluation manifest")
    return content, digest


def load_legacy_method(content):
    # No second 3B model: execute source into a private module, then bind only
    # its original _get_action implementation to the SAME frozen policy object.
    name = "_v19_historical_policy_regression"
    module = types.ModuleType(name)
    module.__file__ = "git:historical-native-policy"
    sys.modules[name] = module  # dataclasses resolves its module through this map.
    exec(compile(content, module.__file__, "exec"), module.__dict__)
    return module.RepresentationPolicyV18._get_action


def measured(reference, candidate):
    """Compare values in FP32, record original dtype separately.

    Cache stores BF16; the reader correctly promotes these values to FP32.
    That explicit promotion must not be confused with a numerical mismatch.
    """
    result = numerical_comparison(reference, candidate)
    result["values_equal"] = bool(reference.shape == candidate.shape and
                                  torch.equal(reference.float().cpu(), candidate.float().cpu()))
    return result


def action_seed(eid, decision):
    return 190026 + int(eid) * 1009 + int(decision)


@torch.inference_mode()
def replay_policy(policy, loader, record, ep, *, label, method, action_decisions):
    """Actual observation/session path; real AE at fixed endpoints, otherwise stub.

    GT prior controls satisfy the real API but are not V19 encoder inputs.
    Every real generated action is repeated with the exact same conditioning
    and RNG seed. Neither targets nor action masks enter denoising.
    """
    import pandas as pd
    from gr00t.long_memory.hamlet import isolated_seed
    eid, frames = int(record["episode_id"]), ep["frames"].tolist()
    raw_path = loader.dataset_path / loader.data_path_pattern.format(
        episode_chunk=eid // loader.chunk_size, episode_index=eid)
    raw = pd.read_parquet(raw_path, columns=["is_demo"])
    table = loader._load_parquet_data(eid)
    language = policy.language_key
    if language == "task":
        if len(record["metadata"]["tasks"]) != 1:
            raise ValueError("Ambiguous language task")
        table["language.task"] = record["metadata"]["tasks"][0]
    video = loader._load_video_data(eid, np.asarray(frames))
    head, core = policy.model.action_head, policy.representation
    original_process, original_step = head.process_backbone_output, core.step
    original_generate = head.get_action_with_features
    captures, actions, repeated_actions, invariants = [], {}, {}, []
    active, last_backbone = {}, {}
    sid = f"v19-precision-{eid}"  # Same ID reset for every independent pass.

    def process(*args, **kwargs):
        output = original_process(*args, **kwargs)
        last_backbone["output"] = output  # aligned helper modifies same dict after return.
        return output

    def step(short, moment, state, frame, demo, **kwargs):
        d = active["decision"]
        before = kwargs.get("bank")
        before = core.initial_bank() if before is None else before
        expected_tokens = min(d, core.config.capacity_events) * policy.n_q
        if before.shape[1] != expected_tokens:
            raise AssertionError("Read-before-write/FIFO size changed")
        result = original_step(short, moment, state, frame, demo, **kwargs)
        if result["bank"].shape[1] != min(d + 1, core.config.capacity_events) * policy.n_q:
            raise AssertionError("Write-after-read/FIFO size changed")
        if int(frame[0]) != int(ep["frames"][d]) or bool(demo[0]) != bool(ep["is_demo"][d]):
            raise AssertionError("Model time/demo input differs from canonical cache endpoint")
        backbone = last_backbone["output"]
        captures.append({"moment": cpu(moment), "short": cpu(short), "state": cpu(state),
                         "features": cpu(backbone["backbone_features"][0]),
                         "attention_masks": cpu(backbone["backbone_attention_mask"][0]),
                         "image_masks": cpu(backbone["image_mask"][0]),
                         "encoded": cpu(result["encoded_current"]), "bank_before": cpu(before),
                         "bank_after": cpu(result["bank"]), "fused": cpu(result["fused"])})
        invariants.append({"episode_id": eid, "branch": label, "decision": d, "frame": int(frame[0]),
            "read_enabled": bool(kwargs["read_enabled"]), "write_enabled": bool(kwargs["write_enabled"]),
            "bank_before_tokens": before.shape[1], "bank_after_tokens": result["bank"].shape[1],
            "autocast_reader": torch.is_autocast_enabled("cuda")})
        if kwargs["read_enabled"] != (not bool(demo[0]) and d > 0):
            raise AssertionError("Read mask changed")
        if torch.is_autocast_enabled("cuda"):
            raise AssertionError("Autocast escaped feature extraction into memory")
        return result

    def no_ae_autocast(_module, _args):
        if torch.is_autocast_enabled("cuda"):
            raise AssertionError("Autocast escaped feature extraction into AE")

    def generate(features, state_features, emb, backbone):
        d = active["decision"]
        if torch.is_autocast_enabled("cuda"):
            raise AssertionError("Autocast escaped feature extraction into denoising")
        if d not in action_decisions:
            return {"action_pred": features.new_zeros(1, head.action_horizon, head.action_dim)}
        seed = action_seed(eid, d)
        predictions = []
        prior = head._inference_gen
        for _ in range(2):
            head._inference_gen = torch.Generator(device=features.device).manual_seed(seed)
            with isolated_seed(seed, features.device):
                prediction = original_generate(features, state_features, emb, backbone)["action_pred"]
            predictions.append(cpu(prediction))
        head._inference_gen = prior
        actions[d], repeated_actions[d] = predictions
        active["action_conditioning"] = cpu(features)
        return {"action_pred": predictions[0].to(features.device)}

    hooks = [head.state_encoder.register_forward_pre_hook(no_ae_autocast),
             head.model.register_forward_pre_hook(no_ae_autocast)]
    try:
        with patch.object(head, "process_backbone_output", process), patch.object(core, "step", step), \
                patch.object(head, "get_action_with_features", generate):
            for d, frame in enumerate(frames):
                active["decision"] = d
                observation = {
                    "video": {key: video[key][d][None, None] for key in policy.modality_configs["video"].modality_keys},
                    "state": {key: np.asarray(table[f"state.{key}"].iloc[frame], np.float32)[None, None]
                              for key in policy.modality_configs["state"].modality_keys},
                    "language": {language: [[str(table[f"language.{language}"].iloc[frame])]]},
                }
                controls = np.empty((0, 8), np.float32)
                if d and not bool(ep["is_demo"][d - 1]):
                    controls = np.concatenate([
                        np.stack(table[f"action.{key}"].iloc[frames[d - 1]:frame].to_list())
                        for key in policy.modality_configs["action"].modality_keys], axis=-1).astype(np.float32)
                passive = bool(raw["is_demo"].iloc[frame])
                method(policy, observation, {"session_ids": [sid], "reset_memory": [d == 0],
                    "episode_seed": 6, "frame_index": frame, "passive": passive,
                    "prime_only": passive, "executed_actions": controls})
                if d % 16 == 0:
                    print(f"[precision-regression] {label} episode={eid} event={d}/{len(frames)}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    if len(captures) != len(frames):
        raise AssertionError("Endpoint count changed")
    return captures, actions, repeated_actions, invariants


def cache_captures(core, ep):
    stages, batch_shape_rows = cached_stages(core, ep)
    for d, row in enumerate(stages):
        for key in ("features", "attention_masks", "image_masks"):
            row[key] = ep[key][d]
    return stages, batch_shape_rows


def compare_captures(eid, branch, reference, current):
    if len(reference) != len(current):
        raise AssertionError("Different endpoint counts")
    return [{"episode_id": eid, "branch": branch, "decision": d, "stage": key,
             **measured(ref[key], cur[key])}
            for d, (ref, cur) in enumerate(zip(reference, current)) for key in STAGES]


def comparison_pass(rows):
    return bool(rows) and all(row["values_equal"] and row["finite"] for row in rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default="runs/long_memory/cache_full1600_v1")
    p.add_argument("--checkpoint", default="runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072")
    p.add_argument("--historical-native-manifest", default="runs/eval/robomme/v19_fullcoverage_v1/prefix_full/comparison_manifest.json")
    p.add_argument("--historical-ref", default="009ef0cda240901c8d252c89cb96ac312b9e98e6")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--execute-plan", action="store_true", help="Execute an existing plan-only directory; refuses existing results")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.prepare_only and args.execute_plan:
        raise ValueError("Choose prepare-only or execute-plan, not both")
    output, checkpoint = Path(args.output_dir).resolve(), Path(args.checkpoint).resolve()
    cache = EpisodeCache(args.cache_dir)
    validate_output_scope(output, cache.path, checkpoint, cache.manifest["dataset_path"], cache.manifest["model_path"])
    content, old_sha = legacy_source(args.historical_ref, args.historical_native_manifest)
    files_before = {name: sha(checkpoint / name) for name in CHECKPOINT_FILES}
    identifiers = {"checkpoint_files_sha256": files_before, "checkpoint": str(checkpoint),
                   "cache_manifest_sha256": sha(cache.path / "manifest.json"),
                   "cache_fingerprint": cache.manifest["fingerprint"],
                   "historical_native_manifest_sha256": sha(args.historical_native_manifest),
                   "historical_native_policy_sha256": old_sha, "historical_ref": args.historical_ref}
    if args.execute_plan:
        if not output.is_dir() or set(p.name for p in output.iterdir()) != {"plan.json"}:
            raise ValueError("execute-plan accepts only a directory containing plan.json and no results")
        plan = json.loads((output / "plan.json").read_text())
        if any(plan.get(key) != value for key, value in identifiers.items()):
            raise ValueError("Plan artifact identity changed")
    else:
        if output.exists():
            raise FileExistsError("Choose a new output; previous diagnostics are preserved")
        plan = {"format_version": 1, "kind": "v19_feature_precision_fixed_panel", **identifiers,
                "selection_rule": "cache-VAL only, chronology-only deterministic panel; no outcomes read",
                "panel": make_panel(cache), "source_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "device": args.device, "action_seed_rule": "190026 + episode_id * 1009 + decision",
                "pass_contract": "Exact value equality for same-shape execution; batch-shape differences measured separately",
                "limits": ["Teacher-trajectory observations, not simulator rollouts or success rates",
                           "Real AE generation at first and last valid action endpoint only; other endpoints stub denoising",
                           "Cache-style sequential memory reference separates batch-size numerical effects"]}
        output.mkdir(parents=True)
        write_json(output / "plan.json", plan)
    print("[precision-regression] fixed panel: " + json.dumps([(x["episode_id"], x["events"], x["demo_events"]) for x in plan["panel"]]), flush=True)
    if args.prepare_only:
        return 0
    write_json(output / "gpu_before.json", gpu_snapshot(args.device))
    from run_scripts.robomme.policy_representation_v18 import RepresentationPolicyV18
    from run_scripts.robomme.feature_precision_v19 import feature_precision_contract
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.long_memory.action_audit_v8 import generated_action
    from gr00t.eval.sim.robomme.run_long_memory_rollout import demo_endpoints
    start = time.monotonic()
    if os.environ.get("GR00T_INFERENCE_SEED") != "6":
        raise ValueError("Set GR00T_INFERENCE_SEED=6, matching the rollout sampler's seeded-generator branch")
    source_before = {name: sha(ROOT / name) for name in SOURCES}
    policy = RepresentationPolicyV18(cache.manifest["model_path"], str(checkpoint), device=args.device)
    if policy.feature_precision != "native" or inspect.signature(RepresentationPolicyV18).parameters["feature_precision"].default != "native":
        raise AssertionError("Default precision must remain native")
    head, core = policy.model.action_head, policy.representation
    if any(p.requires_grad for p in policy.model.parameters()) or any(p.requires_grad for p in core.parameters()):
        raise AssertionError("Regression must freeze all policy parameters")
    legacy_method = load_legacy_method(content)
    current_method = RepresentationPolicyV18._get_action
    episodes = MappedEpisodes(cache)
    dataset = Path(cache.manifest["dataset_path"])
    _, raw_records = _episode_records(dataset)
    records = {int(row["episode_id"]): row for row in raw_records}
    loader = LeRobotEpisodeLoader(dataset, policy.processor.modality_configs["new_embodiment"], video_backend="opencv")
    config = CacheConfig(model_path=cache.manifest["model_path"], dataset_path=str(dataset),
                         output_dir="UNUSED_NO_CACHE_WRITES", device=args.device)
    rows, action_rows, batch_rows, invariant_rows, cadence = [], [], [], [], []
    for selection in plan["panel"]:
        eid = selection["episode_id"]
        ep = episodes.fetch(eid)
        if ep["frames"].tolist() != selection["frames"]:
            raise AssertionError("Prespecified panel chronology changed")
        record = records[eid]
        recache = _extract_episode(policy.model, policy.processor, loader, record, config)
        if not torch.equal(recache["frames"], ep["frames"]):
            raise AssertionError("Cache producer changed endpoint cadence")
        ref, batch = cache_captures(core, ep)
        reref, _ = cache_captures(core, recache)
        rows += compare_captures(eid, "cache-producer-repeat-vs-saved", ref, reref)
        batch_rows += [dict(row, episode_id=eid) for row in batch]
        for key in ("target_mask", "action_mask", "decision_mask", "transition_valid", "is_demo"):
            if not torch.equal(ep[key], recache[key]):
                raise AssertionError(f"Recomputed cache mask changed: {key}")
        actual_demo = ep["frames"][ep["is_demo"]].tolist()
        n_demo = int(ep["frames"][~ep["is_demo"]][0]) if bool(ep["is_demo"].any()) else 0
        if actual_demo != demo_endpoints(n_demo, policy.stride):
            raise AssertionError("Online/cache demo cadence mismatch")
        cadence.append({"episode_id": eid, "demo_frames": n_demo, "demo_endpoints": actual_demo,
                        "events": len(ep["frames"]), "cache_masks_equal": True,
                        "read_before_write_expected_tokens": "min(decision,32)*4", "reset_each_pass": True})
        branches = {}
        for label, precision, method in (
                ("legacy-native", "native", legacy_method), ("default-native", "native", current_method),
                ("explicit-native", "native", current_method), ("native-repeat", "native", current_method),
                ("cache-aligned", "cache-aligned", current_method), ("aligned-repeat", "cache-aligned", current_method)):
            policy.feature_precision = precision
            policy.feature_precision_rules = feature_precision_contract(precision)
            captures, actions, repeated, invariants = replay_policy(policy, loader, record, ep,
                label=label, method=method, action_decisions=selection["action_decisions"])
            branches[label] = captures, actions
            invariant_rows += invariants
            for d in actions:
                action_rows.append({"episode_id": eid, "decision": d, "branch": label + "-same-conditioning-repeat",
                                    **measured(actions[d], repeated[d])})
            if label in ("default-native", "explicit-native", "native-repeat"):
                refs, refactions = branches["legacy-native"]
            elif label == "aligned-repeat":
                refs, refactions = branches["cache-aligned"]
            elif label == "cache-aligned":
                refs, refactions = ref, {}
            else:
                continue
            rows += compare_captures(eid, label, refs, captures)
            for d in selection["action_decisions"]:
                if label == "cache-aligned":
                    target_action = cpu(generated_action(head, ep, d, ref[d]["fused"].to(args.device), seed=action_seed(eid, d)))
                else:
                    target_action = refactions[d]
                action_rows.append({"episode_id": eid, "decision": d, "branch": label,
                                    **measured(target_action, actions[d])})
            write_csv(output / "feature_comparisons.csv", rows)
            write_csv(output / "action_comparisons.csv", action_rows)
        policy.sessions.clear()
        del branches, recache, ref, reref
    write_csv(output / "batch_shape_comparisons.csv", batch_rows)
    write_csv(output / "session_invariants.csv", invariant_rows)
    write_json(output / "cadence_and_masks.json", cadence)
    unchanged = files_before == {name: sha(checkpoint / name) for name in CHECKPOINT_FILES}
    unchanged = unchanged and source_before == {name: sha(ROOT / name) for name in SOURCES}
    checks = {
        "native_reproduces_legacy": comparison_pass([r for r in rows if r["branch"] in ("default-native", "explicit-native", "native-repeat")]),
        "cache_aligned_features": comparison_pass([r for r in rows if r["branch"] in ("cache-aligned", "aligned-repeat", "cache-producer-repeat-vs-saved")]),
        "session_invariants": bool(invariant_rows) and all(not r["autocast_reader"] for r in invariant_rows),
        "real_ae_generation": comparison_pass(action_rows),
        "input_artifacts_unchanged": unchanged,
    }
    passed = all(checks.values())
    result = {"format_version": 1, "kind": "v19_feature_precision_regression", "passed": passed,
              "checks": checks, "checkpoint_files_sha256": files_before,
              "historical_native_manifest_sha256": identifiers["historical_native_manifest_sha256"],
              "historical_native_policy_sha256": old_sha, "source_sha256": source_before,
              "precision_rules": {mode: feature_precision_contract(mode) for mode in ("native", "cache-aligned")},
              "sampler_environment_seed": os.environ.get("GR00T_INFERENCE_SEED"),
              "sampler_rng_scope": "Explicit fixed per-query generator resets; intervening stubbed calls do not test whole-episode RNG advancement",
              "elapsed_seconds": time.monotonic() - start, "policy_trainable_parameters": 0,
              "feature_comparisons": len(rows), "action_comparisons": len(action_rows),
              "batch_shape_note": "Batched encode_prefix vs sequential step; measured but not assumed exactly equal",
              "batch_shape_nonexact": sum(not row["exact"] for row in batch_rows),
              "evidence_files_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / "completed.json", result)
    print(f"[precision-regression] {'PASS' if passed else 'FAIL'}: {checks}", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
