#!/usr/bin/env python3
"""Matched archive continuation: attention LoRA with/without output-projector adaptation.

The memory structure, APPEND rule, original flow objective and causal query plan
are unchanged. Only the candidate trains an additional zero-initialized full-rank
FP32 output-projector residual. The original BF16 projection is never rewritten;
AdamW decay applies to its delta, not to the original weight. NEW checkpoints
use a strict V10 format: old V7/V9 loaders must reject them, never drop the delta.
Offline validation is diagnostic, NOT RoboMME task accuracy or proof of 30%.
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
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes, validate_decision
from gr00t.long_memory.checkpoint_v7 import load_checkpoint_v7 as load_legacy_checkpoint, v7_checkpoint_info
from run_scripts.robomme.checkpoint_projector_v10 import (
    base_projector_spec, checkpoint_info, load_checkpoint, save_checkpoint)
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import LoRAConfig, expert_parameters, install_expert_lora
from run_scripts.robomme.projector_adapter_v10 import (install_projector, projector_spec,
    projector_parameters, set_trainable, assert_expert_scope, all_adapters_disabled,
    expert_episode_flow_loss, generated_prefix_metrics, trainable_expert_parameters)
from gr00t.long_memory.hamlet import checkpoint_identity, isolated_seed, load_frozen_hamlet, validate_cache_checkpoint
from gr00t.long_memory.monitoring import RunLogger, _atomic_json
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from gr00t.long_memory.replay_v7 import replay_queries
from gr00t.long_memory.safety_v5 import validate_output_scope
from gr00t.long_memory.train_v3 import _Tee, _grad_norm, _mean, _seed, runtime_identity
from gr00t.long_memory.train_v7 import lr_factor, optimizer_groups
from run_scripts.robomme.train_archive_deployment_v9 import build_plan, generated_components

DRIVER = "archive_projector_v10"
ARCHITECTURE = DRIVER
SELECTION = "val/generated_observed_prefix_mse"


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
        ROOT / "run_scripts/robomme/projector_adapter_v10.py",
        ROOT / "run_scripts/robomme/checkpoint_projector_v10.py",
        ROOT / "run_scripts/robomme/train_archive_deployment_v9.py",
        ROOT / "run_scripts/robomme/deployment_objective_v9.py",
        ROOT / "run_scripts/robomme/audit_archive_generation_v7.py"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir")
    p.add_argument("--output-dir", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-checkpoint", help="Trained Stage-1 V7 archive; NEW optimizer")
    source.add_argument("--resume", help="Exact V10-driver resume into a NEW directory")
    p.add_argument("--max-steps", type=int, default=128, help="TOTAL planned horizon, including resumed steps")
    p.add_argument("--stop-after-steps", type=int, help="Pause at TOTAL boundary without changing LR horizon")
    p.add_argument("--query-batch-size", type=int, default=4)
    p.add_argument("--train-projector", action=argparse.BooleanOptionalAction, default=False,
                   help="Candidate: train final-projection delta; control: keep it zero/disabled")
    p.add_argument("--projector-learning-rate", type=float, default=1e-6)
    # A fixed compatibility field for the unchanged plan and objective metadata.
    p.set_defaults(aux_weight=0.)
    p.add_argument("--memory-learning-rate", type=float, default=1e-5)
    p.add_argument("--expert-learning-rate", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--warmup-fraction", type=float, default=.05)
    p.add_argument("--checkpoint-segment", type=int, default=8)
    p.add_argument("--val-samples", type=int, default=32)
    p.add_argument("--val-noise-samples", type=int, default=2)
    p.add_argument("--eval-steps", type=int, default=32)
    p.add_argument("--save-steps", type=int, default=32)
    p.add_argument("--log-steps", type=int, default=4)
    p.add_argument("--plot-steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=9042)
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
            raise ValueError("Exact resume requires archive_projector_v10, not V7 or V9")
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
    if not args.cache_dir:
        raise ValueError("--cache-dir is required")
    for key in ("max_steps", "query_batch_size", "checkpoint_segment", "val_noise_samples",
                "eval_steps", "save_steps", "log_steps", "plot_steps"):
        if type(getattr(args, key)) is not int or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if args.val_samples < 32:
        raise ValueError("Use at least 32 fixed held-out validation queries")
    if args.seed < 0 or not 0 <= args.warmup_fraction < 1:
        raise ValueError("Invalid seed/warmup fraction")
    for key in ("aux_weight", "weight_decay"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("memory_learning_rate", "expert_learning_rate", "projector_learning_rate", "max_grad_norm"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if args.aux_weight != 0 or type(args.train_projector) is not bool:
        raise ValueError("V10 tests only original flow; projector flag must be boolean")
    if args.stop_after_steps is not None and not 1 <= args.stop_after_steps <= args.max_steps:
        raise ValueError("stop-after-steps must lie inside the fixed total horizon")


def resume_options(args, info, sources):
    """Reject silent horizon, objective, query, source or runtime changes."""
    if info["config"].get("driver_variant") != DRIVER or info["metadata"].get("driver_variant") != DRIVER:
        raise ValueError("This is not an exact V10-driver checkpoint")
    mutable = {"init_checkpoint", "resume", "output_dir", "stop_after_steps", "preflight_only"}
    for key, value in vars(args).items():
        if key not in mutable and info["config"]["train"].get(key) != value:
            raise ValueError(f"Exact resume option changed: {key}; this pilot requires unchanged options or a NEW V7-parent experiment")
    if info["metadata"]["source_sha256"] != sources or info["metadata"]["runtime"] != runtime_identity():
        raise ValueError("Source/runtime changed; exact resume is unsafe")
    if args.stop_after_steps is not None and args.stop_after_steps <= info["step"]:
        raise ValueError("Pause boundary must exceed the resumed step")


def query_objective(head, episode, decision, fused, *, flow_seed,
                    generation_seed, aux_weight, action_steps, activation_checkpointing):
    """Same original flow objective in BOTH arms; no auxiliary training."""
    if aux_weight != 0:
        raise ValueError("Output-projector pilot does not alter the loss")
    validate_decision(episode, decision)
    result = expert_episode_flow_loss(head, episode, decision, fused, seed=flow_seed,
                                     activation_checkpointing=activation_checkpointing)
    loss = result["loss"]
    if not bool(torch.isfinite(loss)) or (torch.is_grad_enabled() and not loss.requires_grad):
        raise FloatingPointError("Nonfinite or disconnected original flow objective")
    return loss, {"loss": float(loss.detach()), "action_loss": float(loss.detach()),
                  "flow_action_loss": float(loss.detach()),
                  "velocity_mae": float(result["velocity_mae"].detach())}


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
            context = all_adapters_disabled(head) if role == "baseline" else nullcontext()
            with context:
                conditioning = fused if role == "reader" else None
                flow = expert_episode_flow_loss(head, ep, decision, conditioning, seed=item["flow_seed"])
                generated = generated_prefix_metrics(head, ep, decision, conditioning,
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


def assert_trainable_scope(memory, head, cvom, args):
    assert_expert_scope(head)
    if memory.training or cvom.training or any(p.requires_grad for p in cvom.parameters()):
        raise RuntimeError("CVOM must remain frozen/eval, memory eval")
    if not all(p.requires_grad for p in memory.parameters()):
        raise RuntimeError("Registered memory parameters must be trainable")
    if not all(p.requires_grad for p in expert_parameters(head)):
        raise RuntimeError("Attention LoRA must be trainable in both arms")
    if any(p.requires_grad != args.train_projector for p in projector_parameters(head)):
        raise RuntimeError("Projector trainable scope differs from selected arm")
    if projector_spec(head)["enabled"] != args.train_projector:
        raise RuntimeError("Projector forward policy differs from selected arm")


def make_optimizer(args, memory, head, cvom):
    # Reuse the original groups/order exactly; append only the experimental delta.
    groups = optimizer_groups(args, memory, head, cvom)
    if args.train_projector:
        for p in projector_parameters(head):
            groups.append({"params": [p], "kind": "projector",
                           "name": "projector_" + ("weight" if p.ndim > 1 else "bias"),
                           "lr": args.projector_learning_rate,
                           "weight_decay": args.weight_decay if p.ndim > 1 else 0.})
    return torch.optim.AdamW(groups)


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
    expected_projector = base_projector_spec(base_path, enabled=args.train_projector)
    source = Path(args.resume or args.init_checkpoint).resolve()
    initial = (checkpoint_info if args.resume else v7_checkpoint_info)(base_path, source, expected_stage=1)
    if initial["config"]["mode"] != "archive" or initial["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]:
        raise ValueError("Requires same-cache Stage-1 archive checkpoint")
    if not args.resume and initial["step"] <= 0:
        raise ValueError("Weights-only continuation needs a trained archive checkpoint")
    if int(cache.manifest["action_steps"]) != 16:
        raise ValueError("This RoboMME pilot requires the unchanged 16-step execution prefix")
    output = Path(args.output_dir).resolve()
    validate_output_scope(output, cache.path, cache.manifest.get("dataset_path"), base_path, source)
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
            "start_step": start, "plan_sha256": plan_sha, "train_projector": args.train_projector,
            "projector": expected_projector,
            "validation_queries": len(plan["validation"]), "selection_metric": SELECTION,
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
    if args.resume:
        install_projector(head, enabled=initial["config"]["projector"]["enabled"])
        load_checkpoint(source, memory, head, cvom)
    else:
        load_legacy_checkpoint(source, memory, head, cvom)
        install_projector(head, enabled=args.train_projector)
    if projector_spec(head) != base_projector_spec(base_path, enabled=args.train_projector):
        raise ValueError("Installed projector differs from immutable base header/selected arm")
    memory.eval().requires_grad_(True)
    cvom.eval().requires_grad_(False)
    set_trainable(head, train_projector=args.train_projector, train_lora=True)
    assert_trainable_scope(memory, head, cvom, args)
    if head.num_inference_timesteps != 4:
        raise ValueError("Requires unchanged deployed Euler4 inference")
    optimizer = make_optimizer(args, memory, head, cvom)
    start = initial["step"] if args.resume else 0
    state = copy.deepcopy(initial["metadata"]["train_state"]) if args.resume else {
        "optimizer_updates": 0, "window_cursor": 0, "processed_queries": 0,
        "best_generated_prefix_mse": None, "best_checkpoint": None, "elapsed_seconds": 0., "status": "initialized"}
    if args.resume:
        restored = load_checkpoint(source, memory, head, cvom, optimizer)
        if restored["training_state"] != {"driver_variant": DRIVER, "window_cursor": start, "plan_sha256": plan_sha}:
            raise ValueError("Optimizer-boundary extra state differs from exact-resume plan")
        if state["window_cursor"] != start or state["optimizer_updates"] != start:
            raise ValueError("Training cursor differs from checkpoint step")
        if state["processed_queries"] != sum(w["query_count"] for w in plan["windows"][:start]):
            raise ValueError("Saved query coverage differs from fixed optimizer schedule")
    else:
        _seed(args.seed + 102)
    config = {"trainer_variant": ARCHITECTURE, "driver_variant": DRIVER, "stage": 1, "mode": "archive",
        "memory_architecture": "recurrent_memory_v7", "projector": projector_spec(head),
        "memory": asdict(cfg), "expert": asdict(expert_cfg), "expert_targets": targets, "train": vars(args).copy(),
        "objective": {"driver_variant": DRIVER, "flow_weight": 1., "generated_observed_prefix_weight": args.aux_weight,
            "action_steps": 16, "generation": "unchanged_pure_noise_Euler4", "selection_metric": SELECTION}}
    parent = {"path": str(source), "step": initial["step"], "files": {name: file_hash(source / name)
        for name in (("checkpoint.json", "model.safetensors", "expert.safetensors", "cvom.safetensors", "training_state.pt")
                     + (("projector.safetensors",) if args.resume else ()))}}
    metadata = {"driver_variant": DRIVER, "base_model": checkpoint_identity(base_path),
        "cache_fingerprint": cache.manifest["fingerprint"], "cache_dir": str(Path(cache.path).resolve()),
        "cache_manifest_sha256": digest(cache.manifest), "source_sha256": sources, "runtime": runtime_identity(),
        "plan_sha256": plan_sha, "train_state": state, "parent_checkpoint": parent,
        "original_continuation_parent": copy.deepcopy(initial["metadata"].get("original_continuation_parent", parent)) if args.resume else parent,
        "selection_metric": SELECTION, "note": "Same V7 archive; final-projection delta adaptation is the only intervention. Offline loss is NOT accuracy."}
    logger = RunLogger(output)
    _atomic_json(output / "run_config.json", config)
    _atomic_json(output / "query_plan.json", {"sha256": plan_sha, **plan})
    _atomic_json(output / "provenance.json", metadata)
    if args.resume and state["best_checkpoint"]:
        _atomic_json(output / "best_checkpoint.json", {"path": state["best_checkpoint"], "inherited": True,
            "selection_metric": SELECTION})
    print(f"[projector-v10] SAME archive/original flow; train_projector={args.train_projector}; planned={args.max_steps}; start={start}", flush=True)
    print(f"[projector-v10] trainable memory={sum(p.numel() for p in memory.parameters() if p.requires_grad)} "
          f"expert_total={sum(p.numel() for p in trainable_expert_parameters(head))}; "
          f"projector={sum(p.numel() for p in projector_parameters(head) if p.requires_grad)}; CVOM/base frozen", flush=True)

    def evaluate(step):
        summaries, records = validate(args, memory, head, episodes, plan, 16)
        for role, split in (("reader", "val"), ("memory-off", "comparison/memory-off"), ("baseline", "comparison/baseline")):
            logger.log(step, split, {**summaries[role], "queries": float(len(plan["validation"])),
                "noise_draws_per_query": float(args.val_noise_samples)})
        _atomic_json(output / f"validation-{step:06d}.json", {"step": step, "plan_sha256": plan_sha,
            "selection_metric": SELECTION, "summary": summaries, "records": records})
        score = summaries["reader"]["generated_observed_prefix_mse"]
        print(f"[projector-v10][val] step={step} observed-prefix={score:.8f}; "
              f"off={summaries['memory-off']['generated_observed_prefix_mse']:.8f}; NOT task accuracy", flush=True)
        return score

    def save(step, best=False):
        assert_trainable_scope(memory, head, cvom, args)
        assert_finite_optimizer(optimizer)
        if best:
            state["best_checkpoint"] = str(output / f"checkpoint-{step:06d}")
        path = save_checkpoint(output, step, memory, head, cvom, optimizer, config, metadata, best=best,
            training_state={"driver_variant": DRIVER, "window_cursor": step, "plan_sha256": plan_sha})
        _atomic_json(output / "last_checkpoint.json", {"path": str(path), "step": step})
        _atomic_json(output / "status.json", state)

    if not args.resume:
        state["best_generated_prefix_mse"] = evaluate(0)
        save(0, best=True)
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
                flow_seed=item["flow_seed"], generation_seed=item["generation_seed"], aux_weight=args.aux_weight,
                action_steps=16, activation_checkpointing=args.activation_checkpointing)
            (loss / len(items)).backward()
            rows.append(row)
            del loss, fused
        row = _mean(rows)
        row.update(memory_grad_norm=_grad_norm(memory.parameters()), expert_grad_norm=_grad_norm(expert_parameters(head)),
                   projector_grad_norm=_grad_norm(projector_parameters(head)))
        norm = torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]],
                                             args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        assert_finite_optimizer(optimizer)
        state.update(optimizer_updates=step, window_cursor=step,
            processed_queries=state["processed_queries"] + len(items), elapsed_seconds=elapsed_before + time.monotonic() - started,
            status="complete" if step == args.max_steps else "paused" if step == end else "training")
        row.update(grad_norm=float(norm), learning_rate=optimizer.param_groups[0]["lr"],
            processed_queries=float(state["processed_queries"]), elapsed_seconds=state["elapsed_seconds"],
            update_seconds=time.monotonic() - tick, train_projector=float(args.train_projector))
        interval.append(row)
        if step == 1 or step % args.log_steps == 0 or step == end:
            logger.log(step, "train", _mean(interval)); interval = []
            print(f"[projector-v10][train] step={step}/{args.max_steps} loss={row['loss']:.8f} "
                  f"flow={row['flow_action_loss']:.8f} grad={float(norm):.5f}", flush=True)
        best = False
        # Pausing off-cadence must not add an extra best-selection opportunity.
        if step % args.eval_steps == 0 or step == args.max_steps:
            score = evaluate(step)
            if score < state["best_generated_prefix_mse"]:
                state["best_generated_prefix_mse"], best = score, True
        if step % args.save_steps == 0 or step == end or best:
            save(step, best)
        if step % args.plot_steps == 0 or step == end:
            logger.plot()
    _atomic_json(output / "status.json", state)
    print(f"[projector-v10] {state['status']}; {output / 'last_checkpoint.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
