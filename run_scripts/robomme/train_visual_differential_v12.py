#!/usr/bin/env python3
"""Bounded V12 visual-only pilot over an immutable archive1250 + Action Expert.

All parent memory, LoRA, AE, projector and CVOM parameters remain frozen. Only
the new visual patch encoder/reader/output learns the original flow objective.
The visual-off validation control is the SAME frozen archive1250 policy, not
original HAMLET. Offline errors do not establish retrieval selectivity, robot
success, or convergence. V12 bundles require their hash-bound external parent.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import fcntl
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import load_file

from gr00t.long_memory.action_audit_v8 import generated_action
from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes, validate_decision
from gr00t.long_memory.checkpoint_v7 import load_checkpoint_v7, v7_checkpoint_info
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import LoRAConfig, expert_episode_flow_loss, install_expert_lora, set_expert_trainable
from gr00t.long_memory.hamlet import checkpoint_identity, isolated_seed, validate_cache_checkpoint
from gr00t.long_memory.monitoring import RunLogger, _atomic_json
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from gr00t.long_memory.replay_v7 import replay_queries
from gr00t.long_memory.safety_v5 import validate_output_scope
from gr00t.long_memory.train_v3 import _Tee, _grad_norm, _mean, _seed, runtime_identity
from gr00t.long_memory.train_v7 import lr_factor
from run_scripts.robomme.audit_archive_generation_v7 import prefix_masks
from run_scripts.robomme.checkpoint_visual_differential_v12 import checkpoint_info, load_checkpoint, parent_reference, save_checkpoint
from run_scripts.robomme.checkpoint_visual_patch_v11 import checkpoint_info as reference_checkpoint_info
from run_scripts.robomme.replay_visual_differential_v12 import OBSERVATION_KEYS, replay_visual_differential
from run_scripts.robomme.train_archive_deployment_v9 import (
    assert_finite_optimizer, build_plan as original_build_plan, digest, file_hash, generated_components,
)
from run_scripts.robomme.verify_projector_v10 import actual_head
from run_scripts.robomme.visual_differential_memory_v12 import CAMERA_ORDER, VisualDifferentialConfig, VisualDifferentialMemoryV12

DRIVER = "visual_differential_v12"
SELECTION = "val/generated_observed_prefix_mae"
FILTER = "TRAIN_ONLY_exclude_decision_zero_no_past_observations"
OBJECTIVE = {"flow_weight": 1., "generated_auxiliary_weight": 0., "selection_metric": SELECTION,
             "trainable_scope": "visual_only", "parent_frozen": True}


def source_identity():
    names = ("train_visual_differential_v12.py", "checkpoint_visual_differential_v12.py", "visual_differential_memory_v12.py",
             "replay_visual_differential_v12.py", "train_archive_deployment_v9.py", "verify_projector_v10.py",
             "projector_adapter_v10.py", "deployment_objective_v9.py", "audit_archive_generation_v7.py",
             "checkpoint_visual_patch_v11.py", "visual_patch_memory_v11.py", "replay_visual_patch_v11.py")
    paths = list((ROOT / "gr00t").rglob("*.py")) + [ROOT / "run_scripts/robomme" / name for name in names]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(set(paths))}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir")
    p.add_argument("--output-dir", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-checkpoint", help="Frozen original V7 archive1250, never rewritten")
    source.add_argument("--resume", help="Exact V12 resume into a NEW directory; external parent required")
    p.add_argument("--read-mode", choices=("differential", "current_only"), default="differential")
    p.add_argument("--reference-run", default="runs/long_memory/v11_visual_pilot512_20260916",
                   help="Immutable V11 run providing exact step-zero visual initialization and TRAIN schedule")
    p.add_argument("--max-steps", type=int, default=512)
    p.add_argument("--stop-after-steps", type=int, help="TOTAL pause boundary; no extra off-cadence validation")
    p.add_argument("--query-batch-size", type=int, default=4)
    p.add_argument("--visual-learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--warmup-fraction", type=float, default=.05)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--val-samples", type=int, default=128)
    p.add_argument("--val-noise-samples", type=int, default=2)
    p.add_argument("--eval-steps", type=int, default=128)
    p.add_argument("--save-steps", type=int, default=128)
    p.add_argument("--plot-steps", type=int, default=128)
    p.add_argument("--log-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=9111)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--checkpoint-encoding", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--preflight-only", action="store_true")
    return p


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    explicit = {p._option_string_actions[token.split("=", 1)[0]].dest for token in argv
                if token.split("=", 1)[0] in p._option_string_actions}
    if args.resume:
        info = json.loads((Path(args.resume) / "checkpoint.json").read_text())
        if info["config"].get("trainer_variant") != DRIVER:
            raise ValueError("Exact resume requires a visual_differential_v12 checkpoint")
        for name, value in info["config"]["train"].items():
            if hasattr(args, name) and name not in explicit:
                setattr(args, name, value)
        args.init_checkpoint = None
        args.resume = p.parse_args(argv).resume
        if "stop_after_steps" not in explicit:
            args.stop_after_steps = None
    args.driver_variant, args.stage, args.mode = DRIVER, 1, "visual_differential"
    return args


def validate_options(args):
    if not args.cache_dir:
        raise ValueError("--cache-dir is required")
    for key in ("max_steps", "query_batch_size", "val_noise_samples", "eval_steps", "save_steps", "plot_steps", "log_steps"):
        if type(getattr(args, key)) is not int or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if args.val_samples < 32:
        raise ValueError("Use at least 32 distinct held-out validation episodes")
    if args.seed < 0 or not 0 <= args.warmup_fraction < 1:
        raise ValueError("Invalid seed/warmup fraction")
    for key in ("visual_learning_rate", "max_grad_norm", "weight_decay"):
        value = getattr(args, key)
        if not math.isfinite(value) or (value < 0 if key == "weight_decay" else value <= 0):
            raise ValueError(f"Invalid finite optimizer option {key}")
    if args.stop_after_steps is not None and not 1 <= args.stop_after_steps <= args.max_steps:
        raise ValueError("Pause boundary must lie inside the fixed total horizon")


def build_plan(args, cache, episodes):
    """Apply a structural TRAIN-only filter without modifying cached tensors."""
    train_ids = set(cache.manifest["splits"]["train"])
    excluded = {}
    class EligibleView:
        def fetch(self, eid):
            episode = episodes.fetch(eid)
            if eid not in train_ids:
                return episode
            mask = episode["decision_mask"].clone()
            if len(mask) and bool(mask[0]):
                excluded[int(eid)] = {"episode_id": int(eid), "decision": 0}
                mask[0] = False
            return {**episode, "decision_mask": mask}
    plan, _ = original_build_plan(args, cache, EligibleView())
    identities = [excluded[eid] for eid in sorted(excluded)]
    plan["train_eligibility"] = {"policy": FILTER, "excluded": identities,
        "excluded_count": len(identities), "before_count": plan["train_query_count"] + len(identities),
        "after_count": plan["train_query_count"], "validation_unchanged": True}
    if any(item["decision"] <= 0 for window in plan["schedule"] for item in window["queries"]):
        raise ValueError("No-history TRAIN query escaped structural eligibility filter")
    return plan, digest(plan)


def visual_config(parent_info):
    memory = parent_info["config"]["memory"]
    return VisualDifferentialConfig(feature_dim=memory["feature_dim"], num_short_tokens=memory["num_short_tokens"],
                             time_scale=memory["time_scale"])


def reference_contract(args, base, cache, parent_record, config, plan):
    """Training-only dependency: prove original V11 initialization + query parity.

    Inference does NOT read this run: all new visual tensors live in the V12
    bundle. An exact training resume does revalidate these immutable references.
    A shorter smoke must be a prefix of the declared V11 horizon, not a reshuffle.
    """
    root = Path(args.reference_run).resolve()
    names = ("run_config.json", "query_plan.json", "checkpoint-000000/checkpoint.json",
             "checkpoint-000000/visual.safetensors", "checkpoint-000000/training_state.pt")
    hashes = {name: file_hash(root / name) for name in names}
    reference_config = json.loads((root / "run_config.json").read_text())
    saved = json.loads((root / "query_plan.json").read_text())
    saved_sha = saved.pop("sha256")
    if digest(saved) != saved_sha:
        raise ValueError("Original V11 reference plan digest differs")
    info = reference_checkpoint_info(base, root / "checkpoint-000000")
    if (info["step"] != 0 or info["config"] != reference_config
            or info["metadata"].get("plan_sha256") != saved_sha
            or info["metadata"]["frozen_parent"] != parent_record
            or info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]
            or reference_config["visual"] != asdict(config)):
        raise ValueError("Original V11 reference initialization/config/parent/cache differs")
    # These identities must remain exact even for a shorter CPU/GPU smoke.
    for key in ("train", "files", "train_query_count", "pairing", "validation_sampling", "train_eligibility"):
        if plan.get(key) != saved.get(key):
            raise ValueError(f"Original V11 TRAIN inventory/identity differs: {key}")
    for key in ("schedule", "windows"):
        if not len(plan[key]) <= len(saved[key]) or plan[key] != saved[key][:len(plan[key])]:
            raise ValueError(f"Original V11 TRAIN schedule/noise differs: {key}")
    for key in ("validation", "validation_schedule"):
        if len(plan[key]) < len(saved[key]) or plan[key][:len(saved[key])] != saved[key]:
            raise ValueError(f"Original V11 validation prefix/noise differs: {key}")
    # Training hyperparameters are the original flow/AdamW pilot, not tuned per
    # arm. Validation size/cadence and smoke horizon may explicitly differ.
    for key in ("seed", "query_batch_size", "visual_learning_rate", "weight_decay", "warmup_fraction",
                "max_grad_norm", "val_noise_samples", "activation_checkpointing", "checkpoint_encoding"):
        if getattr(args, key) != reference_config["train"].get(key):
            raise ValueError(f"Original V11 matched-training option differs: {key}")
    return {"path": str(root), "files_sha256": hashes, "plan_sha256": saved_sha,
            "initial_step": 0, "initial_visual_sha256": hashes["checkpoint-000000/visual.safetensors"],
            "train_schedule_relation": "exact" if len(plan["schedule"]) == len(saved["schedule"]) else "prefix",
            "matched_train_updates": len(plan["schedule"]), "preserved_validation_queries": len(saved["validation"]),
            "training_only_dependency": True}


def load_reference_initialization(reference, visual):
    path = Path(reference["path"]) / "checkpoint-000000/visual.safetensors"
    if file_hash(path) != reference["initial_visual_sha256"]:
        raise ValueError("Original V11 initialization changed after preflight")
    state = load_file(str(path), device="cpu")
    expected = visual.state_dict()
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape
            or state[k].dtype != torch.float32 or not bool(torch.isfinite(state[k]).all()) for k in state):
        raise ValueError("Original V11 visual initialization names/shape/dtype/finite differs")
    if bool(state["output_projection.weight"].count_nonzero()):
        raise ValueError("Original V11 step-zero output is not zero")
    visual.load_state_dict(state, strict=True)


def optimizer_groups(args, visual):
    named = sorted(visual.named_parameters())
    return [{"params": [p for _, p in entries], "param_names": [name for name, _ in entries],
             "kind": "visual", "name": name, "lr": args.visual_learning_rate, "weight_decay": decay}
            for name, decay, entries in (
                ("visual_decay", args.weight_decay, [(n, p) for n, p in named if p.ndim >= 2]),
                ("visual_no_decay", 0., [(n, p) for n, p in named if p.ndim < 2])) if entries]


def frozen_guard(parent, head, cvom):
    return [(name, p, p._version) for label, module in (("archive", parent), ("head", head), ("cvom", cvom))
            for name, p in [(label + "." + name, value) for name, value in
                            list(module.named_parameters()) + list(module.named_buffers())]]


def assert_scope(visual, parent, head, cvom, guard):
    if any(module.training for module in (parent, head, cvom)):
        raise RuntimeError("Frozen parent/AE/CVOM must remain eval")
    if any(p.requires_grad or p.grad is not None for module in (parent, head, cvom) for p in module.parameters()):
        raise RuntimeError("Only visual parameters may train; parent/AE/CVOM must have no gradients")
    if not all(p.requires_grad and p.dtype == torch.float32 for p in visual.parameters()):
        raise RuntimeError("All visual master parameters must remain trainable FP32")
    if any(value._version != version for _, value, version in guard):
        raise RuntimeError("Frozen parent/AE/CVOM tensor changed")


def parent_short(parent, episode, query):
    with torch.no_grad():
        return replay_queries(parent, episode, [query], mode="archive", checkpoint_segment=0)[query][0]


def conditioned_episode(args, visual, episode, query, fused_short, *, enabled=True):
    observations = {key: episode[key] for key in OBSERVATION_KEYS}
    features, bank = replay_visual_differential(visual, observations, query, camera_order=CAMERA_ORDER,
        visual_read_enabled=enabled, checkpoint_encoding=args.checkpoint_encoding)
    count = visual.config.num_short_tokens
    # The visual query/bank saw ORIGINAL short tokens. Only the final AE tail
    # receives the immutable parent's fused short, identically in both roles.
    features = torch.cat((features[:, :-count], fused_short.to(features)), dim=1)
    original = episode["features"][query].to(features)[None]
    images = episode["image_masks"][query].to(device=features.device).bool()[None].clone()
    images[:, -count:] = False
    images &= episode["attention_masks"][query].to(device=features.device).bool()[None]
    delta = features.detach()[images].float() - original[images].float()
    metrics = {"visual_bank_observations": float(bank.tokens.shape[1]),
               "visual_bank_tokens": float(bank.tokens.shape[1] * bank.tokens.shape[2]),
               "empty_history": float(query == 0), "image_residual_norm": float(delta.norm()),
               "image_changed_fraction": float((features.detach()[images] != original[images]).float().mean())}
    return {**episode, "features": {query: features[0]}}, metrics


def query_objective(args, visual, parent, head, episode, item):
    query = item["decision"]
    if query <= 0:
        raise ValueError("No-history training queries must be filtered in the immutable plan")
    validate_decision(episode, query)
    fused = parent_short(parent, episode, query)
    view, metrics = conditioned_episode(args, visual, episode, query, fused)
    result = expert_episode_flow_loss(head, view, query, seed=item["flow_seed"],
                                     activation_checkpointing=args.activation_checkpointing)
    loss = result["loss"]
    if not bool(torch.isfinite(loss)) or not loss.requires_grad:
        raise FloatingPointError("Nonfinite/disconnected visual-only training objective")
    metrics.update(loss=float(loss.detach()), action_loss=float(loss.detach()),
                   flow_action_loss=float(loss.detach()), velocity_mae=float(result["velocity_mae"].detach()))
    return loss, metrics


def validation_query_rows(rows):
    """Average noise draws WITHIN query first; each held-out query weighs once."""
    grouped = {}
    excluded = {"role", "episode_id", "decision", "repeat", "flow_seed", "generation_seed"}
    for row in rows:
        key = (row["role"], row["episode_id"], row["decision"])
        grouped.setdefault(key, []).append(row)
    return [{"role": role, "episode_id": eid, "decision": query, "noise_draws": len(values),
             **_mean([{k: v for k, v in row.items() if k not in excluded} for row in values])}
            for (role, eid, query), values in grouped.items()]


def validation_summaries(rows):
    queries = validation_query_rows(rows)
    metrics = {}
    for role in ("reader", "visual-off"):
        values = [row for row in queries if row["role"] == role]
        if not values:
            raise ValueError("Missing fixed validation role")
        metrics[role] = _mean([{k: v for k, v in row.items()
                              if k not in {"role", "episode_id", "decision", "noise_draws"}} for row in values])
    off = {(row["episode_id"], row["decision"]): row for row in queries if row["role"] == "visual-off"}
    for role in ("reader", "visual-off"):
        values = [row for row in queries if row["role"] == role]
        if {(row["episode_id"], row["decision"]) for row in values} != set(off):
            raise ValueError("Validation query roles are not paired")
        for name in ("prefix", "joint7", "gripper1"):
            for error in ("mae", "mse"):
                key = f"generated_observed_{name}_{error}"
                if all(key in row for row in values) and all(key in row for row in off.values()):
                    deltas = [row[key] - off[row["episode_id"], row["decision"]][key] for row in values]
                    metrics[role][f"query_{name}_{error}_median"] = statistics.median(row[key] for row in values)
                    metrics[role][f"query_{name}_{error}_delta_mean"] = statistics.fmean(deltas)
                    metrics[role][f"query_{name}_{error}_improved_fraction"] = statistics.fmean(d < 0 for d in deltas)
    return metrics


@torch.no_grad()
def validate(args, visual, parent, head, episodes, plan):
    rows, previous, views = [], None, None
    for item in plan["validation_schedule"]:
        eid, query = item["episode_id"], item["decision"]
        episode = episodes.fetch(eid)
        validate_decision(episode, query)
        if previous != (eid, query):
            fused = parent_short(parent, episode, query)
            views = {role: conditioned_episode(args, visual, episode, query, fused, enabled=enabled)
                     for role, enabled in (("reader", True), ("visual-off", False))}
            previous = eid, query
        for role, (view, visual_metrics) in views.items():
            flow = expert_episode_flow_loss(head, view, query, seed=item["flow_seed"])
            # Native deployed Euler4, not a teacher-interpolated prediction.
            prediction = generated_action(head, view, query, seed=item["generation_seed"])
            target = episode["targets"][query].to(device=prediction.device, dtype=torch.float32)[None]
            _, observed = prefix_masks(episode["target_mask"][query].to(prediction.device)[None],
                                       episode["action_mask"][query], 16)
            difference = (prediction.float() - target)[observed]
            if not difference.numel() or not bool(torch.isfinite(difference).all()):
                raise FloatingPointError("Missing/nonfinite observed-prefix validation values")
            generated = {"prediction": prediction, "loss": difference.square().mean(),
                         "generated_prefix_mae": difference.abs().mean()}
            row = {**item, "role": role, "action_loss": float(flow["loss"]),
                   "flow_action_loss": float(flow["loss"]), "velocity_mae": float(flow["velocity_mae"]),
                   **generated_components(episode, query, generated, 16), **visual_metrics}
            if any(not math.isfinite(value) for value in row.values() if isinstance(value, float)):
                raise FloatingPointError("Nonfinite validation record")
            rows.append(row)
    return validation_summaries(rows), rows


def resume_options(args, info, sources):
    if info["config"].get("objective") != OBJECTIVE or info["metadata"].get("selection_metric") != SELECTION:
        raise ValueError("Exact V12 objective/selection semantics changed")
    mutable = {"init_checkpoint", "resume", "output_dir", "stop_after_steps", "preflight_only"}
    for key, value in vars(args).items():
        if key not in mutable and info["config"]["train"].get(key) != value:
            raise ValueError(f"Exact resume option changed: {key}")
    if info["metadata"]["source_sha256"] != sources or info["metadata"]["runtime"] != runtime_identity():
        raise ValueError("Source/runtime changed; exact resume rejected")
    if args.stop_after_steps is not None and args.stop_after_steps <= info["step"]:
        raise ValueError("Pause boundary must exceed the resumed step")


def validate_resume_state(info, plan, plan_sha):
    """Validate cursor/selection bookkeeping BEFORE model or output creation."""
    step = info["step"]
    if info["training_state"] != {"driver_variant": DRIVER, "window_cursor": step, "plan_sha256": plan_sha}:
        raise ValueError("Optimizer-boundary extra state differs from exact plan")
    state = info["metadata"].get("train_state", {})
    expected = {"optimizer_updates": step, "window_cursor": step,
                "processed_queries": sum(w["query_count"] for w in plan["windows"][:step])}
    if any(type(state.get(key)) is not int or state[key] != value for key, value in expected.items()):
        raise ValueError("Saved optimizer/query cursor differs from exact schedule")
    if info["metadata"].get("train_eligibility") != plan["train_eligibility"]:
        raise ValueError("Saved TRAIN eligibility record differs from immutable plan")
    for key in ("best_generated_prefix_mae", "elapsed_seconds"):
        value = state.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid resume bookkeeping: {key}")
    if not isinstance(state.get("best_checkpoint"), str) or not state["best_checkpoint"]:
        raise ValueError("Resume requires an honest existing best-checkpoint identity")
    if state.get("status") not in {"initialized", "training", "paused", "complete"}:
        raise ValueError("Invalid saved training status")


def main(argv=None):
    args = parse_args(argv)
    validate_options(args)
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    base = str(Path(cache.manifest["model_path"]).resolve())
    source = Path(args.resume or args.init_checkpoint).resolve()
    initial = checkpoint_info(base, source) if args.resume else v7_checkpoint_info(base, source, expected_stage=1)
    parent_record = initial["metadata"]["frozen_parent"] if args.resume else parent_reference(base, source)
    parent_path = Path(parent_record["path"])
    parent_info = v7_checkpoint_info(base, parent_path, expected_stage=1)
    if parent_info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]:
        raise ValueError("Frozen parent/cache fingerprint differs")
    if args.resume and initial["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]:
        raise ValueError("V12 checkpoint/cache fingerprint differs")
    if int(cache.manifest["action_steps"]) != 16:
        raise ValueError("Requires unchanged 16-step RoboMME execution prefix")
    config = visual_config(parent_info)
    if args.resume and initial["config"]["visual"] != asdict(config):
        raise ValueError("Visual architecture changed on exact resume")
    output = Path(args.output_dir).resolve()
    validate_output_scope(output, cache.path, cache.manifest.get("dataset_path"), base, source, parent_path,
                          Path(args.reference_run).resolve())
    if output.exists():
        raise FileExistsError("Use a NEW output directory")
    sources = source_identity()
    if args.resume:
        resume_options(args, initial, sources)
    episodes = MappedEpisodes(cache)
    plan, plan_sha = build_plan(args, cache, episodes)
    reference = reference_contract(args, base, cache, parent_record, config, plan)
    start = initial["step"] if args.resume else 0
    if start >= args.max_steps:
        raise ValueError("Fixed training horizon already complete")
    if args.resume and initial["metadata"]["plan_sha256"] != plan_sha:
        raise ValueError("Query/noise plan, eligibility or cache file identity changed")
    if args.resume:
        validate_resume_state(initial, plan, plan_sha)
        if initial["metadata"].get("initialization_reference") != reference:
            raise ValueError("Original V11 training initialization/reference provenance changed")
    if args.preflight_only:
        print(json.dumps({"driver_variant": DRIVER, "start_step": start, "planned_updates": args.max_steps,
            "train_eligibility": plan["train_eligibility"], "validation_queries": len(plan["validation"]),
            "plan_sha256": plan_sha, "read_mode": args.read_mode, "visual_config": asdict(config),
            "frozen_parent": parent_record, "initialization_reference": reference,
            "note": "Read-only preflight; no model/output/training created; retrieval selectivity unverified"}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    with (output / ".training.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / "train.log").open("a", buffering=1) as stream:
            with redirect_stdout(_Tee(sys.stdout, stream)), redirect_stderr(_Tee(sys.stderr, stream)):
                try:
                    return run(args, cache, base, source, initial, parent_record, parent_info, episodes,
                               plan, plan_sha, sources, output, config, reference)
                except BaseException as exc:
                    traceback.print_exc()
                    _atomic_json(output / "failure.json", {"error": type(exc).__name__, "message": str(exc),
                        "note": "No partial model published; existing parent/checkpoints retained"})
                    raise


def run(args, cache, base, source, initial, parent_record, parent_info, episodes, plan, plan_sha, sources, output, visual_cfg, reference):
    _seed(args.seed)
    cfg = parent_info["config"]
    head = actual_head(base, args.device)
    with isolated_seed(args.seed + 100, args.device):
        parent = RecurrentMemoryV7(MemoryV7Config(**cfg["memory"])).to(args.device)
        cvom = CVOMV7(parent.config).to(args.device)
        install_expert_lora(head, LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
        visual = VisualDifferentialMemoryV12(visual_cfg, read_mode=args.read_mode).to(args.device)
    load_checkpoint_v7(parent_record["path"], parent, head, cvom)
    parent.eval().requires_grad_(False)
    cvom.eval().requires_grad_(False)
    set_expert_trainable(head, False)
    visual.train()
    if head.num_inference_timesteps != 4:
        raise ValueError("Native Euler4 sampler must remain unchanged")
    guard = frozen_guard(parent, head, cvom)
    assert_scope(visual, parent, head, cvom, guard)
    optimizer = torch.optim.AdamW(optimizer_groups(args, visual))
    start = initial["step"] if args.resume else 0
    state = copy.deepcopy(initial["metadata"]["train_state"]) if args.resume else {
        "optimizer_updates": 0, "window_cursor": 0, "processed_queries": 0, "best_generated_prefix_mae": None,
        "best_checkpoint": None, "elapsed_seconds": 0., "status": "initialized"}
    if args.resume:
        restored = load_checkpoint(source, visual, optimizer)
        if restored["training_state"] != {"driver_variant": DRIVER, "window_cursor": start, "plan_sha256": plan_sha}:
            raise ValueError("Optimizer-boundary extra state differs from exact plan")
        if state["window_cursor"] != start or state["optimizer_updates"] != start:
            raise ValueError("Training cursor differs from checkpoint step")
        if state["processed_queries"] != sum(w["query_count"] for w in plan["windows"][:start]):
            raise ValueError("Saved query count differs from exact schedule")
    else:
        load_reference_initialization(reference, visual)
        _seed(args.seed + 102)
    config = {"trainer_variant": DRIVER, "driver_variant": DRIVER, "stage": 1, "mode": "visual_differential",
        "read_mode": args.read_mode, "visual": asdict(visual_cfg), "camera_order": list(CAMERA_ORDER), "train": vars(args).copy(),
        "objective": copy.deepcopy(OBJECTIVE)}
    metadata = {"driver_variant": DRIVER, "base_model": checkpoint_identity(base), "frozen_parent": parent_record,
        "cache_fingerprint": cache.manifest["fingerprint"], "cache_dir": str(Path(cache.path).resolve()),
        "cache_manifest_sha256": digest(cache.manifest), "source_sha256": sources, "runtime": runtime_identity(),
        "plan_sha256": plan_sha, "train_state": state, "train_eligibility": plan["train_eligibility"],
        "selection_metric": SELECTION, "self_contained": False, "read_mode": args.read_mode,
        "initialization_reference": reference,
        "note": "ONLY visual patch parameters train; frozen external archive+AE required. Offline errors are not robot success; retrieval selectivity unverified."}
    logger = RunLogger(output)
    for name, payload in (("run_config.json", config), ("query_plan.json", {"sha256": plan_sha, **plan}),
                          ("provenance.json", metadata)):
        _atomic_json(output / name, payload)
    if args.resume and state["best_checkpoint"]:
        _atomic_json(output / "best_checkpoint.json", {"path": state["best_checkpoint"], "inherited": True,
                                                      "selection_metric": SELECTION})
    print(f"[visual-v12] ONLY {sum(p.numel() for p in visual.parameters())} visual parameters train; "
          f"archive/LoRA/AE/CVOM frozen; start={start}/{args.max_steps}", flush=True)

    def evaluate(step):
        summaries, records = validate(args, visual, parent, head, episodes, plan)
        assert_scope(visual, parent, head, cvom, guard)
        for role, split in (("reader", "val"), ("visual-off", "comparison/visual-off")):
            logger.log(step, split, {**summaries[role], "queries": float(len(plan["validation"])),
                                    "noise_draws_per_query": float(args.val_noise_samples)})
        _atomic_json(output / f"validation-{step:06d}.json", {"step": step, "plan_sha256": plan_sha,
            "selection_metric": SELECTION, "summary": summaries, "records": records,
            "query_records": validation_query_rows(records), "read_mode": args.read_mode,
            "note": "Native Euler4; visual-off is immutable parent, not original HAMLET; no task accuracy."})
        score = summaries["reader"]["generated_observed_prefix_mae"]
        print(f"[visual-v12][val] step={step} prefix_MAE={score:.8f}; "
              f"parent_MAE={summaries['visual-off']['generated_observed_prefix_mae']:.8f}; "
              f"prefix_MSE={summaries['reader']['generated_observed_prefix_mse']:.8f}; NOT task accuracy", flush=True)
        return score

    def save(step, best=False):
        assert_scope(visual, parent, head, cvom, guard)
        assert_finite_optimizer(optimizer)
        if best:
            state["best_checkpoint"] = str(output / f"checkpoint-{step:06d}")
        path = save_checkpoint(output, step, visual, optimizer, config, metadata, best=best,
            training_state={"driver_variant": DRIVER, "window_cursor": step, "plan_sha256": plan_sha})
        _atomic_json(output / "last_checkpoint.json", {"path": str(path), "step": step})
        _atomic_json(output / "status.json", state)

    if not args.resume:
        state["best_generated_prefix_mae"] = evaluate(0)
        save(0, best=True)
        logger.plot()
    started, elapsed_before = time.monotonic(), state["elapsed_seconds"]
    end = args.stop_after_steps or args.max_steps
    total_queries = sum(w["query_count"] for w in plan["windows"])
    interval = []
    for step in range(start + 1, end + 1):
        tick = time.monotonic()
        if state["window_cursor"] != step - 1:
            raise RuntimeError("Optimizer cursor is not at its planned boundary")
        optimizer.zero_grad(set_to_none=True)
        factor = lr_factor(state["processed_queries"], total_queries, args.warmup_fraction)
        for group in optimizer.param_groups:
            group["lr"] = args.visual_learning_rate * factor
        items, rows = plan["schedule"][step - 1]["queries"], []
        for item in items:
            episode = episodes.fetch(item["episode_id"])
            loss, row = query_objective(args, visual, parent, head, episode, item)
            (loss / len(items)).backward()
            rows.append(row)
            del loss
        assert_scope(visual, parent, head, cvom, guard)
        row = _mean(rows)
        row.update(visual_grad_norm=_grad_norm(visual.parameters()),
                   image_encoder_grad_norm=_grad_norm(visual.image_projection.parameters()),
                   query_grad_norm=_grad_norm(visual.query_projection.parameters()),
                   key_grad_norm=_grad_norm(visual.key_projection.parameters()),
                   value_grad_norm=_grad_norm(visual.value_projection.parameters()),
                   output_grad_norm=_grad_norm(visual.output_projection.parameters()))
        norm = torch.nn.utils.clip_grad_norm_(visual.parameters(), args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        assert_finite_optimizer(optimizer)
        if not all(bool(torch.isfinite(p).all()) for p in visual.parameters()):
            raise FloatingPointError("Nonfinite visual parameters after optimizer update")
        assert_scope(visual, parent, head, cvom, guard)
        state.update(optimizer_updates=step, window_cursor=step, processed_queries=state["processed_queries"] + len(items),
            elapsed_seconds=elapsed_before + time.monotonic() - started,
            status="complete" if step == args.max_steps else "paused" if step == end else "training")
        row.update(grad_norm=float(norm), learning_rate=optimizer.param_groups[0]["lr"],
            processed_queries=float(state["processed_queries"]), elapsed_seconds=state["elapsed_seconds"],
            update_seconds=time.monotonic() - tick)
        interval.append(row)
        if step == 1 or step % args.log_steps == 0 or step == end:
            values = _mean(interval)
            # Counters/LR are the current boundary, not the interval average.
            values.update({key: row[key] for key in ("processed_queries", "elapsed_seconds", "learning_rate")})
            logger.log(step, "train", values)
            interval = []
            print(f"[visual-v12][train] step={step}/{args.max_steps} flow={row['loss']:.8f} grad={float(norm):.6f}", flush=True)
        best = False
        # A pause does not insert an extra validation/best-selection boundary.
        if step % args.eval_steps == 0 or step == args.max_steps:
            score = evaluate(step)
            if score < state["best_generated_prefix_mae"]:
                state["best_generated_prefix_mae"], best = score, True
        if step % args.save_steps == 0 or step == end or best:
            save(step, best)
        if step % args.plot_steps == 0 or step == end:
            logger.plot()
    _atomic_json(output / "status.json", state)
    print(f"[visual-v12] {state['status']}; {output / 'last_checkpoint.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
