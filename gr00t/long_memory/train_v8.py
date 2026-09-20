"""V8 Stage 1: train event encoding/retrieval and the original Expert's LoRA.

Past observed features are re-encoded with gradients, while bounded FIFO
storage itself is fixed. Auxiliary reconstruction supervises retained past
events only. No learned admission/eviction or Stage-2 CVOM is implemented.
"""
import argparse
import ast
import copy
from contextlib import redirect_stderr, redirect_stdout
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

import torch

from .cache import EpisodeCache
from .cache_reader_v3 import MappedEpisodes, validate_decision
from .checkpoint_v8 import load_checkpoint_v8, save_checkpoint_v8, v8_checkpoint_info
from .event_v8 import EventMemoryV8, MemoryV8Config
from .expert_v4 import (LoRAConfig, adapter_disabled, expert_episode_flow_loss,
                        expert_parameters, install_expert_lora, set_expert_trainable)
from .hamlet import checkpoint_identity, isolated_seed, load_frozen_hamlet, validate_cache_checkpoint
from .monitoring import RunLogger, _atomic_json
from .replay_v8 import replay_queries
from .safety_v5 import validate_output_scope
from .train_v3 import _Tee, _grad_norm, _mean, _seed, runtime_identity
from .train_v7 import coverage_windows, lr_factor, query_seed

VARIANT = "event_memory_v8"
AUXILIARY = "storage_reconstruction_loss"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, default=1, help="Only Stage 1 is implemented; Stage 2 is rejected")
    p.add_argument("--cache-dir")
    p.add_argument("--output-dir", required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--init-checkpoint", help="Same-mode V8 Stage-1 weights; start a NEW experiment")
    source.add_argument("--resume", help="Exact resume into NEW output, preserving planned coverage/LR horizon")
    p.add_argument("--mode", choices=("event", "none"), default="event")
    p.add_argument("--source", choices=("moment", "short"), default="moment")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, help="Planned TOTAL update cap; changes the LR horizon")
    p.add_argument("--stop-after-steps", type=int, help="Pause at this TOTAL boundary without changing the LR horizon")
    p.add_argument("--queries-per-prefix", type=int, default=2)
    p.add_argument("--query-batch-size", type=int, default=8)
    p.add_argument("--checkpoint-segment", type=int, default=8,
                   help="Compatibility setting; V8 batches independent event encodings without recurrent checkpointing")
    p.add_argument("--memory-learning-rate", type=float, default=3e-5)
    p.add_argument("--expert-learning-rate", type=float, default=3e-6)
    p.add_argument("--storage-reconstruction-weight", type=float, default=.01)
    p.add_argument("--memory-dropout", type=float, default=.1, help="Seeded per-query READ bypass; past reconstruction still trains")
    p.add_argument("--warmup-fraction", type=float, default=.05)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--capacity", type=int, default=128, help="Maximum retained OBSERVATIONS, not tokens")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--residual-scale", type=float, default=.1)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.)
    p.add_argument("--val-samples", type=int, default=64)
    p.add_argument("--val-noise-samples", type=int, default=2)
    p.add_argument("--epoch-val-samples", type=int, default=512, help="0 disables the larger epoch validation")
    p.add_argument("--epoch-val-noise-samples", type=int, default=4)
    p.add_argument("--val-memory-off", action=argparse.BooleanOptionalAction, default=True,
                   help="Compare the same adapted Expert without memory")
    p.add_argument("--max-train-episodes", type=int, default=0)
    p.add_argument("--max-val-episodes", type=int, default=0)
    p.add_argument("--eval-steps", type=int, default=250)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--plot-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
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
        source = args.resume
        info = json.loads((Path(source) / "checkpoint.json").read_text())
        if info["config"].get("trainer_variant") != VARIANT:
            raise ValueError("Exact resume requires event_memory_v8")
        for key, value in info["config"]["train"].items():
            if hasattr(args, key) and key not in explicit:
                setattr(args, key, value)
        args.init_checkpoint, args.resume = None, source
        if "stop_after_steps" not in explicit:
            args.stop_after_steps = None
    args._explicit_options = sorted(explicit)
    return args


def source_identity():
    """Hash V8 and its actual local imports, including reused pure V7 helpers.

    Dynamic frozen-model loading also depends on the model/config packages.
    No filesystem data/checkpoint files are mistaken for source dependencies.
    """
    root = Path(__file__).resolve().parents[2]
    pending = [Path(__file__).resolve(), root / "run_scripts/robomme/train_long_memory_v8.py"]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            targets = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    parent = path.parent
                    for _ in range(node.level - 1):
                        parent = parent.parent
                    targets.append(parent / (node.module or "").replace(".", "/"))
                elif node.module and node.module.startswith("gr00t"):
                    targets.append(root / node.module.replace(".", "/"))
            elif isinstance(node, ast.Import):
                targets.extend(root / x.name.replace(".", "/") for x in node.names if x.name.startswith("gr00t"))
            for target in targets:
                candidate = target.with_suffix(".py") if target.with_suffix(".py").is_file() else target / "__init__.py"
                if candidate.is_file():
                    pending.append(candidate.resolve())
    for directory in ("gr00t/model", "gr00t/configs"):
        seen.update(path.resolve() for path in (root / directory).rglob("*.py"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(seen)}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def preflight(args):
    explicit = vars(args).pop("_explicit_options", [])
    if args.stage != 1:
        raise ValueError("V8 implements Stage 1 only; Stage 2/CVOM is not implemented")
    if not args.cache_dir:
        raise ValueError("--cache-dir is required")
    for key in ("max_epochs", "queries_per_prefix", "query_batch_size", "capacity", "hidden_dim", "num_heads",
                "lora_rank", "val_samples", "val_noise_samples", "epoch_val_noise_samples",
                "eval_steps", "save_steps", "log_steps", "plot_steps"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("max_steps", "stop_after_steps"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive when provided")
    for key in ("seed", "max_train_episodes", "max_val_episodes", "epoch_val_samples", "checkpoint_segment"):
        if getattr(args, key) < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("memory_learning_rate", "expert_learning_rate", "lora_alpha", "max_grad_norm", "residual_scale"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("weight_decay", "storage_reconstruction_weight"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 <= args.warmup_fraction < 1 or not math.isfinite(args.warmup_fraction):
        raise ValueError("warmup-fraction must be in [0,1)")
    if not 0 <= args.memory_dropout <= 1 or not math.isfinite(args.memory_dropout):
        raise ValueError("memory-dropout must be in [0,1]")
    cache = EpisodeCache(args.cache_dir)
    manifest = cache.manifest
    validate_cache_checkpoint(manifest)
    base = str(Path(manifest["model_path"]).resolve())
    source = args.resume or args.init_checkpoint
    validate_output_scope(args.output_dir, getattr(cache, "path", args.cache_dir), manifest.get("dataset_path"), base, source)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a NEW output directory, including for resume; old runs are never overwritten")
    initial = v8_checkpoint_info(base, source) if source else None
    base_cfg = json.loads((Path(base) / "config.json").read_text())
    if initial:
        if initial["metadata"]["cache_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Initial checkpoint cache differs")
        cfg, expert_cfg = MemoryV8Config(**initial["config"]["memory"]), LoRAConfig(**initial["config"]["expert"])
        for key, expected in (("capacity", cfg.capacity), ("hidden_dim", cfg.hidden_dim), ("num_heads", cfg.num_heads),
                              ("source", cfg.source), ("residual_scale", cfg.residual_scale),
                              ("lora_rank", expert_cfg.rank), ("lora_alpha", expert_cfg.alpha)):
            if key in explicit and getattr(args, key) != expected:
                raise ValueError(f"Initial checkpoint architecture differs from explicit {key}")
            setattr(args, key, expected)
        if initial["config"]["stage"] != 1 or initial["config"]["mode"] != args.mode:
            raise ValueError("V8 initialization/resume requires same-mode Stage 1")
        if args.resume:
            mutable = {"resume", "init_checkpoint", "output_dir", "stop_after_steps", "preflight_only"}
            for key, value in vars(args).items():
                if key not in mutable and initial["config"]["train"].get(key) != value:
                    raise ValueError(f"Exact resume option changed: {key}; use a NEW weights-only experiment")
            if initial["metadata"]["source_sha256"] != source_identity() or initial["metadata"]["runtime"] != runtime_identity():
                raise ValueError("Source/runtime changed; exact resume is unsafe")
            if args.stop_after_steps is not None and args.stop_after_steps <= initial["step"]:
                raise ValueError("stop-after-steps must exceed the resumed step")
    else:
        cfg = MemoryV8Config(feature_dim=manifest["feature_dim"], state_dim=manifest["state_dim"],
            num_short_tokens=int(base_cfg["n_moment_tokens"]), hidden_dim=args.hidden_dim, capacity=args.capacity,
            num_heads=args.num_heads, time_scale=float(base_cfg.get("memory_stride", 16)),
            source=args.source, residual_scale=args.residual_scale)
        expert_cfg = LoRAConfig(args.lora_rank, args.lora_alpha)
    if (base_cfg.get("hamlet_mode") != "finetune" or base_cfg.get("memory_type", "moment_token") != "moment_token"
            or base_cfg.get("mem_cond_type", "cross_attn") != "cross_attn"):
        raise ValueError("V8 requires original HAMLET moment-token cross-attention checkpoint")
    for name, expected in (("feature_dim", manifest["feature_dim"]), ("state_dim", manifest["state_dim"]),
                           ("num_short_tokens", int(base_cfg["n_moment_tokens"])),
                           ("time_scale", float(base_cfg.get("memory_stride", 16)))):
        if getattr(cfg, name) != expected:
            raise ValueError(f"Actual V8 {name} differs from cache/base")
    if cfg.feature_dim != int(base_cfg["backbone_embedding_dim"]):
        raise ValueError("Cache feature dimension differs from base backbone")
    return cache, base, cfg, expert_cfg, initial


def make_plans(args, cache, episodes):
    plans = {"train": [], "val": [], "files": {}, "provenance_groups": {}}
    records = {int(r["episode_id"]): r for r in cache.manifest.get("episodes", [])}
    train_ids, val_ids = map(set, (cache.manifest["splits"]["train"], cache.manifest["splits"]["val"]))
    if train_ids & val_ids:
        raise ValueError("Train/validation episode splits overlap")
    for split, limit in (("train", args.max_train_episodes), ("val", args.max_val_episodes)):
        ids = list(cache.manifest["splits"][split])
        for eid in ids[:limit] if limit else ids:
            ep = episodes.fetch(eid)
            if "is_demo" not in ep:
                raise ValueError("V8 requires explicit is_demo metadata")
            decisions = torch.where(ep["decision_mask"])[0].tolist()
            if decisions:
                plans[split].append([int(eid), decisions])
            plans["provenance_groups"][str(eid)] = str(ep.get("task", records.get(eid, {}).get("task", "unknown")))
            if eid in records:
                path = (Path(cache.path) / records[eid]["path"]).resolve()
                stat = path.stat()
                plans["files"][str(eid)] = [str(path), stat.st_size, stat.st_mtime_ns]
        if not plans[split]:
            raise ValueError(f"No valid {split} action queries")
    pool = [[eid, d] for eid, ds in plans["val"] for d in ds]
    plans["validation"] = random.Random(args.seed + 1701).sample(pool, min(len(pool), args.val_samples))
    plans["epoch_validation"] = random.Random(args.seed + 1702).sample(pool, min(len(pool), args.epoch_val_samples))
    plans["train_query_count"] = sum(len(ds) for _, ds in plans["train"])
    plans["pairing"] = "chronologically_adjacent_queries_then_global_group_shuffle"
    return plans


def optimizer_groups(args, memory, head):
    modules = [("memory", list(memory.named_parameters()))] if args.mode != "none" else []
    modules.append(("expert", [(f"lora_{i}", p) for i, p in enumerate(expert_parameters(head))]))
    groups = []
    for kind, parameters in modules:
        decay, no_decay = [], []
        for name, parameter in parameters:
            if not parameter.requires_grad:
                continue
            exempt = parameter.ndim <= 1 or any(s in name.lower() for s in ("norm", "token_id", "position", "embedding"))
            (no_decay if exempt else decay).append(parameter)
        for suffix, values, wd in (("decay", decay, args.weight_decay), ("no_decay", no_decay, 0.)):
            if values:
                groups.append({"params": values, "name": kind + "_" + suffix, "kind": kind,
                               "lr": getattr(args, kind + "_learning_rate"), "weight_decay": wd})
    return groups


def flow(head, ep, decision, fused=None, **kwargs):
    validate_decision(ep, decision)
    result = expert_episode_flow_loss(head, ep, decision, fused, **kwargs)
    metrics = {"conditioning_delta_norm": 0., "conditioning_changed_token_fraction": 0.,
               "conditioning_changed_value_fraction": 0.}
    if fused is not None:
        adapters = {id(p) for p in expert_parameters(head)}
        reference = next(p for p in head.parameters() if id(p) not in adapters)
        q = fused.shape[-2]
        original = (ep["features"][decision][-q:] if "features" in ep else ep["short"][decision]).to(reference.device, reference.dtype)[None]
        actual = fused.detach().to(reference.device, reference.dtype)
        metrics = {"conditioning_delta_norm": float((actual.float() - original.float()).norm()),
                   "conditioning_changed_token_fraction": float((actual != original).any(-1).float().mean()),
                   "conditioning_changed_value_fraction": float((actual != original).float().mean())}
    return {**result, **metrics}


def _scalars(metrics):
    return {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()
            if (torch.is_tensor(v) and v.numel() == 1) or isinstance(v, (int, float))}


def memory_dropped(args, epoch, eid, decision):
    """Decision-local randomness is independent of flow noise and global RNG."""
    return args.mode == "event" and random.Random(query_seed(args.seed, epoch, eid, decision,
        domain="memory-dropout-v8")).random() < args.memory_dropout


def query_objective(args, head, ep, decision, read, epoch):
    fused, diagnostics = read
    diagnostics = dict(diagnostics)
    auxiliary = diagnostics.pop(AUXILIARY, None)
    if args.mode == "event" and auxiliary is None:
        raise ValueError("V8 replay must supply past-event reconstruction loss")
    if auxiliary is None:
        auxiliary = fused.new_zeros(())
    if not torch.is_tensor(auxiliary) or auxiliary.numel() != 1 or not bool(torch.isfinite(auxiliary)):
        raise FloatingPointError("Invalid V8 storage reconstruction loss")
    dropped = memory_dropped(args, epoch, int(ep["episode_id"]), decision)
    result = flow(head, ep, decision, None if dropped else fused,
        seed=query_seed(args.seed, epoch, int(ep["episode_id"]), decision),
        activation_checkpointing=args.activation_checkpointing)
    weighted = args.storage_reconstruction_weight * auxiliary if args.mode == "event" else auxiliary * 0
    loss = result["loss"] + weighted
    if not bool(torch.isfinite(loss)) or not loss.requires_grad:
        raise FloatingPointError("Nonfinite or disconnected V8 training objective")
    row = {"action_loss": float(result["loss"].detach()), "loss": float(loss.detach()),
        "velocity_mae": float(result["velocity_mae"].detach()), AUXILIARY: float(auxiliary.detach()),
        "weighted_storage_reconstruction_loss": float(weighted.detach()), "memory_dropout_rate": float(dropped),
        **_scalars(diagnostics)}
    row.update({k: v for k, v in result.items() if k.startswith("conditioning_")})
    return loss, row


@torch.no_grad()
def validate(args, memory, head, episodes, plan, draws, domain, baseline_cache):
    grouped, rows = {}, []
    for eid, decision in plan:
        grouped.setdefault(eid, []).append(decision)
    for eid, queries in grouped.items():
        ep = episodes.fetch(eid)
        reads = replay_queries(memory, ep, sorted(queries), mode=args.mode, checkpoint_segment=args.checkpoint_segment)
        for decision in queries:
            fused, diagnostics = reads[decision]
            diagnostics = dict(diagnostics)
            # Log held-out reconstruction separately; action loss alone selects
            # checkpoints, regardless of the auxiliary weight or convergence.
            for repeat in range(draws):
                seed = query_seed(args.seed, 0, eid, decision, repeat, domain)
                out = flow(head, ep, decision, fused, seed=seed)
                key = (eid, decision, seed)
                if key not in baseline_cache:
                    with adapter_disabled(head):
                        baseline_cache[key] = float(flow(head, ep, decision, seed=seed)["loss"])
                row = {"action_loss": float(out["loss"]), "loss": float(out["loss"]),
                    "baseline_action_loss": baseline_cache[key], "memory_gain": baseline_cache[key] - float(out["loss"]),
                    "velocity_mae": float(out["velocity_mae"]), **_scalars(diagnostics)}
                row.update({k: v for k, v in out.items() if k.startswith("conditioning_")})
                if args.val_memory_off:
                    off = float(flow(head, ep, decision, seed=seed)["loss"])
                    row.update(expert_no_memory_action_loss=off, memory_gain_over_adapted_expert=off - row["action_loss"])
                rows.append(row)
    return {**_mean(rows), "queries": float(len(plan)), "noise_draws_per_query": float(draws)}


def main(argv=None):
    args = parse_args(argv)
    cache, base, cfg, expert_cfg, initial = preflight(args)
    episodes = MappedEpisodes(cache)
    plans = make_plans(args, cache, episodes)
    windows = coverage_windows(plans["train"], args.max_epochs, args.queries_per_prefix, args.query_batch_size, args.seed)
    if args.max_steps:
        windows = windows[:args.max_steps]
    plan_sha = _digest({"plans": plans, "windows": windows})
    if args.resume and initial["metadata"]["plan_sha256"] != plan_sha:
        raise ValueError("Coverage/validation/cache files changed after checkpoint")
    horizon, start = len(windows), initial["step"] if args.resume else 0
    if start >= horizon:
        raise ValueError("Planned training horizon already completed; use weights-only initialization")
    if args.preflight_only:
        print(json.dumps({"variant": VARIANT, "stage": 1, "mode": args.mode,
            "memory": asdict(cfg), "expert": asdict(expert_cfg), "train_queries_per_epoch": plans["train_query_count"],
            "planned_updates": horizon, "plan_sha256": plan_sha, "note": "Read-only; no model/output/training started"}, indent=2))
        return 0
    output = Path(args.output_dir).resolve()
    # Atomic acquisition avoids racing another invocation for the same output.
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(path.name != ".training.lock" for path in output.iterdir()):
            raise FileExistsError("Output became nonempty before training lock acquisition")
        with (output / "training.log").open("a", encoding="utf-8") as journal:
            with redirect_stdout(_Tee(sys.stdout, journal)), redirect_stderr(_Tee(sys.stderr, journal)):
                try:
                    return _run(args, cache, base, cfg, expert_cfg, initial, episodes, plans, windows,
                                plan_sha, horizon, start, output)
                except Exception:
                    traceback.print_exc()
                    raise


def _run(args, cache, base_path, cfg, expert_cfg, initial, episodes, plans, windows, plan_sha, horizon, start, output):
    _seed(args.seed)
    with isolated_seed(args.seed + 100, "cpu"):
        memory = EventMemoryV8(cfg).to(args.device)
    base, _ = load_frozen_hamlet(base_path, args.device)
    head = base.action_head
    del base
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    with isolated_seed(args.seed + 101, args.device):
        targets = install_expert_lora(head, expert_cfg, targets=initial["config"]["expert_targets"] if initial else None)
    if initial:
        load_checkpoint_v8(args.resume or args.init_checkpoint, memory, head)
    memory.requires_grad_(args.mode != "none").eval()
    set_expert_trainable(head, True)
    config = {"trainer_variant": VARIANT, "stage": 1, "mode": args.mode,
              "memory": asdict(cfg), "expert": asdict(expert_cfg), "expert_targets": targets, "train": vars(args).copy()}
    state = copy.deepcopy(initial["metadata"]["train_state"]) if args.resume else {
        "processed_queries": 0, "optimizer_updates": 0, "epoch": 0, "window_cursor": 0,
        "best_action_loss": None, "best_checkpoint": None, "elapsed_seconds": 0., "status": "initialized"}
    expected_queries = sum(w["query_count"] for w in windows[:start])
    if state["window_cursor"] != start or state["optimizer_updates"] != start or state["processed_queries"] != expected_queries:
        raise ValueError("Saved V8 coverage cursor/query count does not match optimizer boundary")
    metadata = {"base_model": checkpoint_identity(base_path), "cache_fingerprint": cache.manifest["fingerprint"],
        "cache_dir": str(Path(args.cache_dir).resolve()), "source_sha256": source_identity(), "runtime": runtime_identity(),
        "plan_sha256": plan_sha, "plans": plans, "train_state": state,
        "initial_checkpoint": str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None,
        "resumed_from": str(Path(args.resume).resolve()) if args.resume else None,
        "note": "Event FIFO; learned encoding/retrieval; past reconstruction auxiliary; offline loss is not task success."}
    optimizer = torch.optim.AdamW(optimizer_groups(args, memory, head))
    if args.resume:
        restored = load_checkpoint_v8(args.resume, memory, head, optimizer)
        extra = restored.get("training_state", {})
        if extra.get("plan_sha256") != plan_sha or extra.get("window_cursor") != start or extra.get("processed_queries") != expected_queries:
            raise ValueError("Checkpoint extra coverage state disagrees with metadata/query plan")
    else:
        _seed(args.seed + 102)
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    logger, baseline_cache = RunLogger(output), {}
    _atomic_json(output / "run_config.json", config)
    _atomic_json(output / "query_plan.json", {"sha256": plan_sha, "windows": windows, "plans": plans})
    _atomic_json(output / "provenance.json", metadata)
    if args.resume and state["best_checkpoint"]:
        _atomic_json(output / "best_checkpoint.json", {"path": state["best_checkpoint"], "inherited": True})
    print(f"[v8] mode={args.mode} source={cfg.source} planned_updates={horizon} queries/epoch={plans['train_query_count']}", flush=True)
    print(f"[v8] trainable memory={sum(p.numel() for p in memory.parameters() if p.requires_grad)} "
          f"expert={sum(p.numel() for p in expert_parameters(head))}; Stage 1 only", flush=True)
    started, elapsed_before = time.monotonic(), state["elapsed_seconds"]

    def evaluate(step, epoch_end=False):
        metrics = validate(args, memory, head, episodes, plans["validation"], args.val_noise_samples, "monitor-validation", baseline_cache)
        logger.log(step, "val", metrics)
        if epoch_end and plans["epoch_validation"]:
            logger.log(step, "val-epoch", validate(args, memory, head, episodes, plans["epoch_validation"],
                args.epoch_val_noise_samples, "epoch-validation", baseline_cache))
        print(f"[v8][val] step={step} action={metrics['action_loss']:.7f} original={metrics['baseline_action_loss']:.7f}", flush=True)
        return metrics

    def save(step, best=False):
        if best:
            state["best_checkpoint"] = str(output / f"checkpoint-{step:06d}")
        path = save_checkpoint_v8(output, step, memory, head, optimizer, config, metadata, best=best,
            training_state={**copy.deepcopy(state), "plan_sha256": plan_sha})
        _atomic_json(output / "last_checkpoint.json", {"path": str(path), "step": step})

    if not args.resume:
        state["best_action_loss"] = evaluate(0)["action_loss"]
        save(0, best=True)
        logger.plot()
    interval = []
    end = min(horizon, args.stop_after_steps) if args.stop_after_steps else horizon
    planned_queries = sum(w["query_count"] for w in windows)
    try:
        for step in range(start + 1, end + 1):
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            multiplier = lr_factor(state["processed_queries"], planned_queries, args.warmup_fraction)
            for group in optimizer.param_groups:
                group["lr"] = getattr(args, group["kind"] + "_learning_rate") * multiplier
            window, rows = windows[step - 1], []
            if state["window_cursor"] != step - 1:
                raise RuntimeError("Coverage cursor is not at an optimizer boundary")
            for eid, queries in window["groups"]:
                ep = episodes.fetch(eid)
                reads = replay_queries(memory, ep, queries, mode=args.mode, checkpoint_segment=args.checkpoint_segment)
                losses = []
                for decision in queries:
                    loss, row = query_objective(args, head, ep, decision, reads[decision], window["epoch"])
                    losses.append(loss)
                    rows.append(row)
                (torch.stack(losses).sum() / window["query_count"]).backward()
            row = _mean(rows)
            row.update(memory_grad_norm=_grad_norm(memory.parameters()), expert_grad_norm=_grad_norm(expert_parameters(head)))
            for name, pieces in (("event_encoder", ("source_", "state_projection", "encoder_norm", "encoder_ffn")),
                                 ("retrieval", ("attention", "retrieval", "read_blocks")),
                                 ("reconstruction", ("reconstruct", "decoder")), ("fusion", ("fusion",))):
                row[name + "_grad_norm"] = _grad_norm(p for n, p in memory.named_parameters() if any(s in n for s in pieces))
            norm = torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]],
                args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            state.update(processed_queries=state["processed_queries"] + window["query_count"], window_cursor=step,
                epoch=window["epoch"] + int(window["epoch_end"]), optimizer_updates=step,
                elapsed_seconds=elapsed_before + time.monotonic() - started)
            row.update(grad_norm=float(norm), learning_rate=optimizer.param_groups[0]["lr"], coverage_epoch=float(state["epoch"]),
                processed_queries=float(state["processed_queries"]), update_seconds=time.monotonic() - tick,
                elapsed_seconds=state["elapsed_seconds"])
            if str(args.device).startswith("cuda"):
                row["peak_vram_bytes"] = float(torch.cuda.max_memory_allocated(args.device))
            interval.append(row)
            if step == 1 or step % args.log_steps == 0 or step == end:
                logger.log(step, "train", _mean(interval))
                interval = []
                print(f"[v8][train] step={step}/{horizon} queries={state['processed_queries']} action={row['action_loss']:.7f} "
                      f"reconstruction={row[AUXILIARY]:.7f}", flush=True)
            best = False
            due_eval = step % args.eval_steps == 0 or step == end or window["epoch_end"]
            if due_eval:
                metrics = evaluate(step, window["epoch_end"])
                if metrics["action_loss"] < state["best_action_loss"]:
                    state["best_action_loss"], best = metrics["action_loss"], True
            state["status"] = "complete" if step == horizon else "paused" if step == end else "training"
            if step % args.save_steps == 0 or step == end or window["epoch_end"] or best:
                save(step, best)
                if window["epoch_end"]:
                    _atomic_json(output / f"epoch-{state['epoch']:03d}_checkpoint.json", {"path": str(output / f"checkpoint-{step:06d}")})
            _atomic_json(output / "status.json", {"step": step, **state})
            if step % args.plot_steps == 0 or due_eval:
                logger.plot()
    except KeyboardInterrupt:
        print("[v8] Interrupted. Resume only a previously saved optimizer-boundary checkpoint.", flush=True)
        return 130
    print(f"[v8] {state['status']}: {output}; action/reconstruction losses are not RoboMME success", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
