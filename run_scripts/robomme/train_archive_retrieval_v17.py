#!/usr/bin/env python3
"""Matched V17 policy continuation: action flow + weak native-retrieval NLL.

New driver, unchanged V7 architecture/evaluator. Both arms see identical action
queries (including the weak-label query) and noise. Only retrieval_weight differs.
Fixed final checkpoint is the preregistered rollout candidate, not TEST-selected.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes, validate_decision
from gr00t.long_memory.checkpoint_v7 import load_checkpoint_v7, save_checkpoint_v7, v7_checkpoint_info
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import (LoRAConfig, adapter_disabled, expert_episode_flow_loss,
    expert_parameters, install_expert_lora, set_expert_trainable)
from gr00t.long_memory.hamlet import checkpoint_identity, isolated_seed, load_frozen_hamlet, validate_cache_checkpoint
from gr00t.long_memory.monitoring import RunLogger, _atomic_json
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from gr00t.long_memory.replay_v7 import replay_queries
from gr00t.long_memory.safety_v5 import validate_output_scope
from gr00t.long_memory.train_v3 import _Tee, _grad_norm, _mean, _seed, runtime_identity
from gr00t.long_memory.train_v7 import coverage_windows, lr_factor, optimizer_groups, query_seed
from run_scripts.robomme.audit_archive_generation_v7 import prefix_masks
from run_scripts.robomme.deployment_objective_v9 import generated_prefix_objective
from run_scripts.robomme.train_archive_deployment_v9 import build_plan as action_plan
from run_scripts.robomme.prepare_segment_targets_v15 import load_manifest
from run_scripts.robomme.archive_retrieval_objective_v17 import positive_mask, retrieval_objective

DRIVER = "archive_retrieval_v17"
ARCHITECTURE = "recurrent_memory_v7"
SELECTION = "fixed_final_step_not_test_selected"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_identity():
    # Explicit NEW-script dependencies avoid changing the live eval code hash
    # scope. All original production code is still included in our own record.
    paths = list((ROOT / "gr00t").rglob("*.py")) + [Path(__file__).resolve(),
        ROOT / "run_scripts/robomme/deployment_objective_v9.py",
        ROOT / "run_scripts/robomme/audit_archive_generation_v7.py",
        ROOT / "run_scripts/robomme/train_archive_deployment_v9.py",
        ROOT / "run_scripts/robomme/prepare_segment_targets_v15.py",
        ROOT / "run_scripts/robomme/archive_retrieval_objective_v17.py"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--targets", help="Published V15 weak TRAIN/VAL segment manifest")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-checkpoint", help="Trained Stage-1 V7 archive; NEW optimizer")
    source.add_argument("--resume", help="Exact V17-driver resume into a NEW directory")
    p.add_argument("--max-steps", type=int, default=512, help="TOTAL planned horizon, including resumed steps")
    p.add_argument("--stop-after-steps", type=int, help="Pause at TOTAL boundary without changing LR horizon")
    p.add_argument("--query-batch-size", type=int, default=4)
    p.add_argument("--retrieval-weight", type=float, default=.01, help="0 = matched action-only control; .01 = exploratory guided arm")
    p.add_argument("--memory-learning-rate", type=float, default=1e-5)
    p.add_argument("--expert-learning-rate", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--warmup-fraction", type=float, default=.05)
    p.add_argument("--checkpoint-segment", type=int, default=8)
    p.add_argument("--val-samples", type=int, default=32)
    p.add_argument("--val-noise-samples", type=int, default=1)
    p.add_argument("--eval-steps", type=int, default=128)
    p.add_argument("--save-steps", type=int, default=128)
    p.add_argument("--log-steps", type=int, default=4)
    p.add_argument("--plot-steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=9171)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    return p


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    explicit = {p._option_string_actions[t.split("=", 1)[0]].dest for t in argv
                if t.split("=", 1)[0] in p._option_string_actions}
    if args.resume:
        info = json.loads((Path(args.resume) / "checkpoint.json").read_text())
        if info["config"].get("driver_variant") != DRIVER:
            raise ValueError("Exact resume requires archive_retrieval_v17, not another trainer")
        for name, value in info["config"]["train"].items():
            if hasattr(args, name) and name not in explicit:
                setattr(args, name, value)
        # The original source argument is never inherited for an exact resume.
        args.init_checkpoint = None
        args.resume = p.parse_args(argv).resume
        if "stop_after_steps" not in explicit:
            args.stop_after_steps = None
    args.driver_variant, args.stage, args.mode = DRIVER, 1, "archive"
    return args


def validate_options(args):
    if not args.cache_dir or not args.targets:
        raise ValueError("--cache-dir and --targets are required")
    if args.query_batch_size < 2:
        raise ValueError("Use at least two queries: one weak-label query plus general action queries")
    for key in ("max_steps", "query_batch_size", "checkpoint_segment", "val_noise_samples",
                "eval_steps", "save_steps", "log_steps", "plot_steps"):
        if type(getattr(args, key)) is not int or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if args.val_samples < 32:
        raise ValueError("Use at least 32 fixed held-out validation queries")
    if args.seed < 0 or not 0 <= args.warmup_fraction < 1:
        raise ValueError("Invalid seed/warmup fraction")
    for key in ("retrieval_weight", "weight_decay"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("memory_learning_rate", "expert_learning_rate", "max_grad_norm"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if args.stop_after_steps is not None and not 1 <= args.stop_after_steps <= args.max_steps:
        raise ValueError("stop-after-steps must lie inside the fixed total horizon")


def resume_options(args, info, sources):
    """Reject silent horizon, objective, query, source or runtime changes."""
    if info["config"].get("driver_variant") != DRIVER or info["metadata"].get("driver_variant") != DRIVER:
        raise ValueError("This is not an exact V17-driver checkpoint")
    mutable = {"init_checkpoint", "resume", "output_dir", "stop_after_steps", "preflight_only"}
    for key, value in vars(args).items():
        if key not in mutable and info["config"]["train"].get(key) != value:
            raise ValueError(f"Exact resume option changed: {key}; use NEW weights-only initialization instead")
    if info["metadata"]["source_sha256"] != sources or info["metadata"]["runtime"] != runtime_identity():
        raise ValueError("Source/runtime changed; exact resume is unsafe")
    if args.stop_after_steps is not None and args.stop_after_steps <= info["step"]:
        raise ValueError("Pause boundary must exceed the resumed step")


def build_plan(args, cache, episodes):
    """Same action/noise schedule in both arms; labels never select VAL/TEST."""
    targets = load_manifest(args.targets, verify_files=True)
    identity = targets["identity"]
    if identity["cache_fingerprint"] != cache.manifest["fingerprint"]:
        raise ValueError("Weak targets belong to a different cache")
    if identity["original_splits"] != cache.manifest["splits"]:
        raise ValueError("Weak target train/val splits changed")
    plan, _ = action_plan(args, cache, episodes)
    inventory = {eid: set(qs) for eid, qs in plan["train"]}
    examples = targets["examples"]
    eligible = {"train": [], "val": []}
    seen = set()
    for index, target in enumerate(examples):
        eid, q, split = target["episode_id"], target["decision"], target["split"]
        if split not in eligible or eid not in cache.manifest["splits"][split] or (eid, q) in seen:
            raise ValueError("Duplicate target or invalid episode split")
        seen.add((eid, q))
        positive_mask(episodes.fetch(eid), q, target)
        if split == "train" and q not in inventory.get(eid, set()):
            raise ValueError("Weak query has no original action supervision")
        eligible[split].append(index)
    if not all(eligible.values()):
        raise ValueError("Need nonempty TRAIN and VAL weak retrieval targets")
    rng = random.Random(args.seed + 177)
    order, cursor = [], 0
    for row, window in zip(plan["schedule"], plan["windows"]):
        if cursor == len(order):
            order, cursor = eligible["train"].copy(), 0
            rng.shuffle(order)
        index = order[cursor]
        cursor += 1
        target = examples[index]
        eid, q = target["episode_id"], target["decision"]
        # Both arms replace EXACTLY the same first query and still apply action
        # loss to it. All remaining queries retain the all-task action schedule.
        row["queries"][0] = {"episode_id": eid, "decision": q, "target_index": index,
            "flow_seed": query_seed(args.seed, row["step"], eid, q, domain="retrieval-v17-flow")}
        window["groups"] = [[item["episode_id"], [item["decision"]]] for item in row["queries"]]
        window["query_count"] = len(row["queries"])
    plan.update(targets_sha256=file_hash(args.targets), targets_fingerprint=targets["fingerprint"],
        weak_examples=examples, eligible_targets=eligible,
        pairing="one_weak_query_plus_general_queries_same_in_both_arms",
        weak_label_limitation="PatternLock unique direction-run proxy, NOT verified causal evidence")
    return plan, digest(plan)


def query_objective(head, episode, decision, fused, *, flow_seed, activation_checkpointing):
    validate_decision(episode, decision)
    result = expert_episode_flow_loss(head, episode, decision, fused, seed=flow_seed,
                                     activation_checkpointing=activation_checkpointing)
    loss = result["loss"]
    if not bool(torch.isfinite(loss)) or (torch.is_grad_enabled() and not loss.requires_grad):
        raise FloatingPointError("Nonfinite or disconnected action objective")
    return loss, {"action_loss": float(loss.detach()), "flow_action_loss": float(loss.detach()),
                  "velocity_mae": float(result["velocity_mae"].detach()), "loss": float(loss.detach())}


def backward_retrieval(args, memory, episodes, plan, items):
    """Separate backward saves AE graph memory; same optimizer boundary.

    Total = mean(action losses) + lambda * mean(eligible-query NLL).
    lambda=0 makes NO extra training forward/backward or RNG call.
    """
    if args.retrieval_weight == 0:
        return {}
    selected = [item for item in items if "target_index" in item]
    if len(selected) != 1:
        raise ValueError("Expected exactly one weak-label query per update")
    rows = []
    for item in selected:
        loss, row = retrieval_objective(memory, episodes.fetch(item["episode_id"]), item["decision"],
            plan["weak_examples"][item["target_index"]], checkpoint_segment=args.checkpoint_segment)
        (args.retrieval_weight * loss / len(selected)).backward()
        rows.append(row)
    result = _mean(rows)
    result["weighted_retrieval_loss"] = args.retrieval_weight * result["retrieval_nll"]
    return result


@torch.no_grad()
def validate_retrieval(args, memory, episodes, plan):
    rows = []
    for index in plan["eligible_targets"]["val"]:
        target = plan["weak_examples"][index]
        _, metrics = retrieval_objective(memory, episodes.fetch(target["episode_id"]), target["decision"],
            target, checkpoint_segment=args.checkpoint_segment)
        rows.append({"episode_id": target["episode_id"], "decision": target["decision"], **metrics})
    summary = _mean([{k: v for k, v in row.items() if k not in ("episode_id", "decision")} for row in rows])
    return summary, rows


def generated_components(episode, decision, result, action_steps):
    prediction = result["prediction"].detach().float()
    target = episode["targets"][decision].to(prediction)[None]
    _, mask = prefix_masks(episode["target_mask"][decision].to(prediction.device)[None],
                           episode["action_mask"][decision], action_steps)
    difference = prediction - target
    metrics = {"generated_observed_prefix_mse": float(result["loss"].detach()),
               "generated_observed_prefix_mae": float(result["generated_prefix_mae"].detach()),
               "generated_valid_values": float(mask.sum())}
    # RoboMME's normalized joint7 + gripper1 coordinates. Padding is masked.
    for name, lo, hi in (("joint7", 0, 7), ("gripper1", 7, 8)):
        component = mask.clone()
        component[:, :, :lo] = False
        component[:, :, hi:] = False
        if bool(component.any()):
            values = difference[component]
            metrics[f"generated_observed_{name}_mse"] = float(values.square().mean())
            metrics[f"generated_observed_{name}_mae"] = float(values.abs().mean())
    return metrics


@torch.no_grad()
def validate(args, memory, head, episodes, plan, action_steps):
    rows = []
    previous, fused = None, None
    for item in plan["validation_schedule"]:
        eid, decision = item["episode_id"], item["decision"]
        ep = episodes.fetch(eid)
        validate_decision(ep, decision)
        if previous != (eid, decision):
            fused = replay_queries(memory, ep, [decision], mode="archive",
                                   checkpoint_segment=args.checkpoint_segment)[decision][0]
            previous = (eid, decision)
        for role in ("reader", "memory-off", "baseline"):
            context = adapter_disabled(head) if role == "baseline" else nullcontext()
            with context:
                conditioning = fused if role == "reader" else None
                flow = expert_episode_flow_loss(head, ep, decision, conditioning, seed=item["flow_seed"])
                generated = generated_prefix_objective(head, ep, decision, conditioning,
                    seed=item["generation_seed"], action_steps=action_steps, activation_checkpointing=False)
            row = {**item, "role": role, "flow_action_loss": float(flow["loss"]),
                   "velocity_mae": float(flow["velocity_mae"]), **generated_components(ep, decision, generated, action_steps)}
            if any(not math.isfinite(v) for k, v in row.items() if isinstance(v, float)):
                raise FloatingPointError("Nonfinite validation result")
            rows.append(row)
    summaries = {role: _mean([{k: v for k, v in row.items() if k not in
        {"role", "episode_id", "decision", "repeat", "flow_seed", "generation_seed"}}
        for row in rows if row["role"] == role]) for role in ("reader", "memory-off", "baseline")}
    return summaries, rows


def assert_trainable_scope(memory, head, cvom):
    adapters = {id(p) for p in expert_parameters(head)}
    if head.training or memory.training or cvom.training:
        raise RuntimeError("Frozen stochastic/stateful layers must stay eval()")
    if any(p.requires_grad for p in cvom.parameters()):
        raise RuntimeError("CVOM is frozen in this same-architecture reader pilot")
    if any(p.requires_grad and id(p) not in adapters for p in head.parameters()):
        raise RuntimeError("Only Action-Expert LoRA may train")
    if not all(p.requires_grad for p in memory.parameters()) or not all(p.requires_grad for p in expert_parameters(head)):
        raise RuntimeError("Memory and installed Expert LoRA must be trainable")


def assert_finite_optimizer(optimizer):
    for values in optimizer.state.values():
        for name, value in values.items():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Nonfinite optimizer state: {name}")


def main(argv=None):
    args = parse_args(argv)
    validate_options(args)
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    base_path = str(Path(cache.manifest["model_path"]).resolve())
    source = Path(args.resume or args.init_checkpoint).resolve()
    initial = v7_checkpoint_info(base_path, source, expected_stage=1)
    if initial["config"]["mode"] != "archive" or initial["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]:
        raise ValueError("Requires same-cache Stage-1 archive checkpoint")
    if not args.resume and initial["step"] <= 0:
        raise ValueError("Weights-only continuation needs a trained archive checkpoint")
    if int(cache.manifest["action_steps"]) != 16:
        raise ValueError("This RoboMME pilot requires the unchanged 16-step execution prefix")
    output = Path(args.output_dir).resolve()
    validate_output_scope(output, cache.path, cache.manifest.get("dataset_path"), base_path, source, Path(args.targets).resolve().parent)
    if output.exists():
        raise FileExistsError("Use a NEW output path; existing directories are never reused")
    sources = source_identity()
    if args.resume:
        resume_options(args, initial, sources)
    episodes = MappedEpisodes(cache)
    plan, plan_sha = build_plan(args, cache, episodes)
    start = initial["step"] if args.resume else 0
    if start >= args.max_steps:
        raise ValueError("Fixed training horizon already complete")
    if args.resume and initial["metadata"]["plan_sha256"] != plan_sha:
        raise ValueError("Exact query/noise plan or immutable cache file records changed")
    if args.preflight_only:
        print(json.dumps({"driver_variant": DRIVER, "architecture": ARCHITECTURE,
            "train_queries": plan["train_query_count"], "planned_updates": args.max_steps,
            "start_step": start, "plan_sha256": plan_sha, "retrieval_weight": args.retrieval_weight,
            "validation_queries": len(plan["validation"]), "selection_metric": SELECTION,
            "weak_queries": {k: len(v) for k, v in plan["eligible_targets"].items()},
            "note": "Read-only: no model, output directory or training created"}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    with (output / ".training.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / "training.log").open("a", encoding="utf-8") as journal:
            with redirect_stdout(_Tee(sys.stdout, journal)), redirect_stderr(_Tee(sys.stderr, journal)):
                try:
                    return run(args, cache, base_path, source, initial, episodes, plan, plan_sha, sources, output)
                except BaseException as exc:
                    traceback.print_exc()
                    _atomic_json(output / "failure.json", {"error": type(exc).__name__, "message": str(exc),
                        "note": "Existing immutable checkpoints remain intact; no partial update published"})
                    raise


def run(args, cache, base_path, source, initial, episodes, plan, plan_sha, sources, output):
    _seed(args.seed)
    cfg, expert_cfg = MemoryV7Config(**initial["config"]["memory"]), LoRAConfig(**initial["config"]["expert"])
    with isolated_seed(args.seed + 100, "cpu"):
        memory, cvom = RecurrentMemoryV7(cfg).to(args.device), CVOMV7(cfg).to(args.device)
    base, processor = load_frozen_hamlet(base_path, args.device)
    head = base.action_head
    del base, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with isolated_seed(args.seed + 101, args.device):
        targets = install_expert_lora(head, expert_cfg, targets=initial["config"]["expert_targets"])
    load_checkpoint_v7(source, memory, head, cvom)
    memory.eval().requires_grad_(True)
    cvom.eval().requires_grad_(False)
    set_expert_trainable(head, True)
    assert_trainable_scope(memory, head, cvom)
    if head.num_inference_timesteps != 4:
        raise ValueError("Requires unchanged deployed Euler4 inference")
    optimizer = torch.optim.AdamW(optimizer_groups(args, memory, head, cvom))
    start = initial["step"] if args.resume else 0
    state = copy.deepcopy(initial["metadata"]["train_state"]) if args.resume else {
        "optimizer_updates": 0, "window_cursor": 0, "processed_queries": 0,
        "elapsed_seconds": 0., "status": "initialized"}
    if args.resume:
        restored = load_checkpoint_v7(source, memory, head, cvom, optimizer)
        if restored["training_state"] != {"driver_variant": DRIVER, "window_cursor": start, "plan_sha256": plan_sha}:
            raise ValueError("Optimizer-boundary extra state differs from exact-resume plan")
        if state["window_cursor"] != start or state["optimizer_updates"] != start:
            raise ValueError("Training cursor differs from checkpoint step")
        if state["processed_queries"] != sum(w["query_count"] for w in plan["windows"][:start]):
            raise ValueError("Saved query coverage differs from fixed optimizer schedule")
    else:
        _seed(args.seed + 102)
    config = {"trainer_variant": ARCHITECTURE, "driver_variant": DRIVER, "stage": 1, "mode": "archive",
        "memory": asdict(cfg), "expert": asdict(expert_cfg), "expert_targets": targets, "train": vars(args).copy(),
        "objective": {"driver_variant": DRIVER, "flow_weight": 1., "weak_retrieval_weight": args.retrieval_weight,
            "action_steps": 16, "weak_labels": "PatternLock_unique_direction_runs",
            "learned_storage_selection": False, "selection_metric": SELECTION}}
    parent = {"path": str(source), "step": initial["step"], "files": {name: file_hash(source / name)
        for name in ("checkpoint.json", "model.safetensors", "expert.safetensors", "cvom.safetensors", "training_state.pt")}}
    metadata = {"driver_variant": DRIVER, "base_model": checkpoint_identity(base_path),
        "cache_fingerprint": cache.manifest["fingerprint"], "cache_dir": str(Path(cache.path).resolve()),
        "cache_manifest_sha256": digest(cache.manifest), "source_sha256": sources, "runtime": runtime_identity(),
        "plan_sha256": plan_sha, "train_state": state, "parent_checkpoint": parent,
        "original_continuation_parent": copy.deepcopy(initial["metadata"].get("original_continuation_parent", parent)) if args.resume else parent,
        "selection_metric": SELECTION, "note": "Same V7 archive architecture; offline validation is NOT task accuracy."}
    logger = RunLogger(output)
    _atomic_json(output / "run_config.json", config)
    _atomic_json(output / "query_plan.json", {"sha256": plan_sha, **plan})
    _atomic_json(output / "provenance.json", metadata)
    print(f"[retrieval-v17] SAME archive architecture; aux={args.retrieval_weight}; planned={args.max_steps}; start={start}", flush=True)
    print(f"[retrieval-v17] trainable memory={sum(p.numel() for p in memory.parameters() if p.requires_grad)} "
          f"expert={sum(p.numel() for p in expert_parameters(head) if p.requires_grad)}; CVOM/base frozen", flush=True)

    def evaluate(step):
        summaries, records = validate(args, memory, head, episodes, plan, 16)
        retrieval_summary, retrieval_records = validate_retrieval(args, memory, episodes, plan)
        logger.log(step, "val/retrieval", retrieval_summary)
        for role, split in (("reader", "val"), ("memory-off", "comparison/memory-off"), ("baseline", "comparison/baseline")):
            logger.log(step, split, {**summaries[role], "queries": float(len(plan["validation"])),
                "noise_draws_per_query": float(args.val_noise_samples)})
        _atomic_json(output / f"validation-{step:06d}.json", {"step": step, "plan_sha256": plan_sha,
            "selection_metric": SELECTION, "summary": summaries, "records": records,
            "retrieval_summary": retrieval_summary, "retrieval_records": retrieval_records})
        score = summaries["reader"]["generated_observed_prefix_mse"]
        print(f"[retrieval-v17][val] step={step} observed-prefix={score:.8f}; "
              f"off={summaries['memory-off']['generated_observed_prefix_mse']:.8f}; NOT task accuracy", flush=True)
        return score

    def save(step):
        assert_trainable_scope(memory, head, cvom)
        assert_finite_optimizer(optimizer)
        path = save_checkpoint_v7(output, step, memory, head, cvom, optimizer, config, metadata, best=False,
            training_state={"driver_variant": DRIVER, "window_cursor": step, "plan_sha256": plan_sha})
        _atomic_json(output / "last_checkpoint.json", {"path": str(path), "step": step})
        _atomic_json(output / "status.json", state)

    if not args.resume:
        evaluate(0)
        save(0)
        logger.plot()
    started, elapsed_before = time.monotonic(), state["elapsed_seconds"]
    end = args.stop_after_steps or args.max_steps
    total_queries = sum(w["query_count"] for w in plan["windows"])
    interval = []
    for step in range(start + 1, end + 1):
        tick = time.monotonic()
        if state["window_cursor"] != step - 1:
            raise RuntimeError("Query cursor must be at an optimizer boundary")
        optimizer.zero_grad(set_to_none=True)
        factor = lr_factor(state["processed_queries"], total_queries, args.warmup_fraction)
        for group in optimizer.param_groups:
            group["lr"] = getattr(args, group["kind"] + "_learning_rate") * factor
        items = plan["schedule"][step - 1]["queries"]
        rows = []
        for item in items:
            eid, decision = item["episode_id"], item["decision"]
            episode = episodes.fetch(eid)
            fused, _ = replay_queries(memory, episode, [decision], mode="archive",
                                      checkpoint_segment=args.checkpoint_segment)[decision]
            loss, row = query_objective(head, episode, decision, fused,
                flow_seed=item["flow_seed"], activation_checkpointing=args.activation_checkpointing)
            (loss / len(items)).backward()
            rows.append(row)
            del loss, fused
        row = _mean(rows)
        aux = backward_retrieval(args, memory, episodes, plan, items)
        row.update(aux)
        row["loss"] += aux.get("weighted_retrieval_loss", 0.)
        row.update(memory_grad_norm=_grad_norm(memory.parameters()), expert_grad_norm=_grad_norm(expert_parameters(head)))
        norm = torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]],
                                             args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        assert_finite_optimizer(optimizer)
        state.update(optimizer_updates=step, window_cursor=step,
            processed_queries=state["processed_queries"] + len(items), elapsed_seconds=elapsed_before + time.monotonic() - started,
            status="complete" if step == args.max_steps else "paused" if step == end else "training")
        row.update(grad_norm=float(norm), learning_rate=optimizer.param_groups[0]["lr"],
            processed_queries=float(state["processed_queries"]), elapsed_seconds=state["elapsed_seconds"],
            update_seconds=time.monotonic() - tick, retrieval_weight=args.retrieval_weight)
        interval.append(row)
        if step == 1 or step % args.log_steps == 0 or step == end:
            logger.log(step, "train", _mean(interval)); interval = []
            print(f"[retrieval-v17][train] step={step}/{args.max_steps} loss={row['loss']:.8f} "
                  f"flow={row['flow_action_loss']:.8f} grad={float(norm):.5f}", flush=True)
        if step % args.eval_steps == 0 or step == end:
            evaluate(step)
        if step % args.save_steps == 0 or step == end:
            save(step)
        if step % args.plot_steps == 0 or step == end:
            logger.plot()
    _atomic_json(output / "status.json", state)
    print(f"[retrieval-v17] {state['status']}; {output / 'last_checkpoint.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
