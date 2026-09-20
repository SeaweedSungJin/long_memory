"""V7 full-prefix recurrent memory and same-short archive/AE-only controls.

Stage 1 covers each valid action decision once per planned epoch. Adjacent
within-episode query pairs share a full prefix; groups are shuffled globally.
This is a documented efficiency choice versus randomly paired queries, not
truncated BPTT. Optimizer windows are normalized by their exact query count.
Stage 2 fixes the actor/Expert and fits signed future KEEP/UPDATE gain in two
fixed-policy rounds. No manual cue labels or simulator counterfactuals.
"""
import argparse
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
from safetensors.torch import load_file, save_file

from .cache import EpisodeCache
from .cache_reader_v3 import MappedEpisodes, validate_decision
from .checkpoint_v7 import (actor_state_sha256, cvom_state_sha256, load_checkpoint_v7,
                            save_checkpoint_v7, v7_checkpoint_info)
from .cvom_v7 import CVOMV7
from .expert_v4 import (LoRAConfig, adapter_disabled, expert_episode_flow_loss,
                        expert_parameters, expert_state_sha256, install_expert_lora,
                        set_expert_trainable)
from .hamlet import checkpoint_identity, isolated_seed, load_frozen_hamlet, validate_cache_checkpoint
from .monitoring import RunLogger, _atomic_json
from .objectives_v7 import (LabelV7Config, build_storage_label, make_storage_contexts,
                            storage_loss, summarize_storage_labels)
from .recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from .replay_v7 import replay_queries
from .safety_v5 import validate_output_scope
from .train_v3 import _Tee, _grad_norm, _mean, _seed, runtime_identity
from .train_v4 import source_identity as inherited_source_identity

VARIANT = "recurrent_memory_v7"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, choices=(1, 2), default=1)
    p.add_argument("--cache-dir")
    p.add_argument("--output-dir", required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--init-checkpoint", help="V7 weights-only initialization; Stage 2 requires recurrent Stage 1")
    source.add_argument("--resume", help="Exact resume into NEW output; preserve planned LR/coverage horizon")
    p.add_argument("--mode", choices=("recurrent", "archive", "none"), default="recurrent")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, help="Planned TOTAL cap; changing this changes the LR horizon")
    p.add_argument("--stop-after-steps", type=int, help="Pause at this TOTAL boundary without changing the planned LR horizon")
    p.add_argument("--queries-per-prefix", type=int, default=2)
    p.add_argument("--query-batch-size", type=int, default=8)
    p.add_argument("--checkpoint-segment", type=int, default=8, help="Recomputation segment, NOT a detach/TBPTT length")
    p.add_argument("--memory-learning-rate", type=float, default=3e-5)
    p.add_argument("--expert-learning-rate", type=float, default=3e-6)
    p.add_argument("--warmup-fraction", type=float, default=.05)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--capacity", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.)
    p.add_argument("--cvom-learning-rate", type=float, default=1e-4)
    p.add_argument("--cvom-rounds", type=int, default=2)
    p.add_argument("--cvom-steps-per-round", type=int, default=2000)
    p.add_argument("--cvom-batch-size", type=int, default=4)
    p.add_argument("--cvom-contexts", type=int, default=512)
    p.add_argument("--cvom-val-contexts", type=int, default=128)
    p.add_argument("--future-samples", type=int, default=4)
    p.add_argument("--noise-samples", type=int, default=4)
    p.add_argument("--label-scale", type=float, default=.001)
    p.add_argument("--cvom-threshold", type=float, default=.05)
    p.add_argument("--stage2-opportunity-only", action="store_true")
    p.add_argument("--allow-stage2", action="store_true", help="Explicitly acknowledge Stage-1/audit review before CVOM fitting; start-up opportunity audit still runs")
    p.add_argument("--val-samples", type=int, default=64)
    p.add_argument("--val-noise-samples", type=int, default=2)
    p.add_argument("--epoch-val-samples", type=int, default=512, help="0 disables the larger epoch-boundary validation")
    p.add_argument("--epoch-val-noise-samples", type=int, default=4)
    p.add_argument("--val-memory-off", action="store_true", help="Same adapted Expert, not separately trained AE-only control")
    p.add_argument("--val-last-writes", type=int, default=0, help="Optional suffix-reset ablation; NOT strict no-old")
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
    p, args = parser(), parser().parse_args(argv)
    if args.resume:
        info = json.loads((Path(args.resume) / "checkpoint.json").read_text())
        if info["config"].get("trainer_variant") != VARIANT:
            raise ValueError("Exact resume requires V7; old versions are preserved separately")
        explicit = {p._option_string_actions[t.split("=", 1)[0]].dest for t in argv
                    if t.split("=", 1)[0] in p._option_string_actions}
        for key, value in info["config"]["train"].items():
            if hasattr(args, key) and key not in explicit:
                setattr(args, key, value)
        args.init_checkpoint = None
        args.resume = p.parse_args(argv).resume
        # A previous interruption limit is not the new default horizon.
        if "stop_after_steps" not in explicit:
            args.stop_after_steps = None
    args._explicit_options = sorted({p._option_string_actions[t.split("=", 1)[0]].dest for t in argv
        if t.split("=", 1)[0] in p._option_string_actions})
    return args


def source_identity():
    hashes = inherited_source_identity()
    root = Path(__file__).resolve().parents[2]
    for relative in ("recurrent_v7.py", "cvom_v7.py", "replay_v7.py", "objectives_v7.py",
                     "checkpoint_v7.py", "train_v7.py", "safety_v5.py"):
        path = root / "gr00t/long_memory" / relative
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = root / "run_scripts/robomme/train_long_memory_v7.py"
    hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def preflight(args):
    explicit = vars(args).pop("_explicit_options", [])
    if not args.cache_dir:
        raise ValueError("--cache-dir is required")
    for key in ("max_epochs", "queries_per_prefix", "query_batch_size", "checkpoint_segment", "capacity", "hidden_dim",
                "num_heads", "lora_rank", "cvom_rounds", "cvom_steps_per_round", "cvom_batch_size", "cvom_contexts",
                "cvom_val_contexts", "future_samples", "noise_samples", "val_samples", "val_noise_samples",
                "epoch_val_noise_samples", "eval_steps", "save_steps", "log_steps", "plot_steps"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("max_steps", "stop_after_steps"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive when provided")
    for key in ("seed", "max_train_episodes", "max_val_episodes", "epoch_val_samples", "val_last_writes"):
        if getattr(args, key) < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("memory_learning_rate", "expert_learning_rate", "cvom_learning_rate", "label_scale", "lora_alpha", "max_grad_norm"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 <= args.warmup_fraction < 1 or not math.isfinite(args.warmup_fraction):
        raise ValueError("warmup-fraction must be in [0,1)")
    if any(not math.isfinite(getattr(args, k)) or getattr(args, k) < 0 for k in ("weight_decay", "cvom_threshold")):
        raise ValueError("weight-decay/threshold must be finite and nonnegative")
    if args.noise_samples < 2:
        raise ValueError("Stage-2 paired noise requires at least two draws")
    if args.stage2_opportunity_only and args.stage != 2:
        raise ValueError("Opportunity audit is a Stage-2 operation")
    cache = EpisodeCache(args.cache_dir)
    manifest = cache.manifest
    validate_cache_checkpoint(manifest)
    base = str(Path(manifest["model_path"]).resolve())
    source = args.resume or args.init_checkpoint
    validate_output_scope(args.output_dir, getattr(cache, "path", args.cache_dir), manifest.get("dataset_path"), base, source)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a NEW output directory, including for resume; no old run is overwritten")
    initial = v7_checkpoint_info(base, source) if source else None
    base_cfg = json.loads((Path(base) / "config.json").read_text())
    if initial:
        if initial["metadata"]["cache_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Initial checkpoint cache differs")
        cfg = MemoryV7Config(**initial["config"]["memory"])
        expert_cfg = LoRAConfig(**initial["config"]["expert"])
        for key, expected in (("capacity", cfg.capacity), ("hidden_dim", cfg.hidden_dim),
                ("num_heads", cfg.num_heads), ("lora_rank", expert_cfg.rank), ("lora_alpha", expert_cfg.alpha)):
            if key in explicit and getattr(args, key) != expected:
                raise ValueError(f"Initial checkpoint architecture differs from explicit {key}")
            setattr(args, key, expected)
        if args.resume:
            if initial["config"]["stage"] != args.stage:
                raise ValueError("Resume cannot change stage")
            mutable = {"resume", "init_checkpoint", "output_dir", "stop_after_steps", "preflight_only", "allow_stage2"}
            for key, value in vars(args).items():
                if key not in mutable and initial["config"]["train"].get(key) != value:
                    raise ValueError(f"Exact resume option changed: {key}; start a NEW weights-only experiment instead")
            if initial["metadata"]["source_sha256"] != source_identity() or initial["metadata"]["runtime"] != runtime_identity():
                raise ValueError("Source/runtime changed; exact resume is unsafe")
            if args.stop_after_steps is not None and args.stop_after_steps <= initial["step"]:
                raise ValueError("stop-after-steps must exceed the resumed step")
        elif initial["config"]["stage"] != 1 or initial["config"]["mode"] != args.mode:
            raise ValueError("Weights initialization requires same-mode V7 Stage 1")
    else:
        cfg = MemoryV7Config(feature_dim=manifest["feature_dim"], state_dim=manifest["state_dim"],
            num_short_tokens=int(base_cfg["n_moment_tokens"]), hidden_dim=args.hidden_dim, capacity=args.capacity,
            num_heads=args.num_heads, time_scale=float(base_cfg.get("memory_stride", 16)))
        expert_cfg = LoRAConfig(args.lora_rank, args.lora_alpha)
    if (base_cfg.get("hamlet_mode") != "finetune" or base_cfg.get("memory_type", "moment_token") != "moment_token"
            or base_cfg.get("mem_cond_type", "cross_attn") != "cross_attn"):
        raise ValueError("V7 requires original HAMLET moment-token cross-attention finetuning checkpoint")
    for name, expected in (("feature_dim", manifest["feature_dim"]), ("state_dim", manifest["state_dim"]),
                           ("num_short_tokens", int(base_cfg["n_moment_tokens"])),
                           ("time_scale", float(base_cfg.get("memory_stride", 16)))):
        if getattr(cfg, name) != expected:
            raise ValueError(f"Actual V7 {name} differs from cache/base")
    if cfg.feature_dim != int(base_cfg["backbone_embedding_dim"]):
        raise ValueError("Cache feature dimension differs from base backbone")
    if args.stage == 2:
        if not initial or args.mode != "recurrent" or (not args.resume and initial["step"] == 0):
            raise ValueError("Stage 2 needs a trained recurrent Stage-1 checkpoint (not step 0/archive/none)")
        if not args.stage2_opportunity_only and not (args.allow_stage2 or args.resume):
            raise ValueError("Inspect Stage-1 results/opportunity first, then explicitly pass --allow-stage2")
    train, val = manifest["splits"]["train"], manifest["splits"]["val"]
    if not train or not val or set(train) & set(val):
        raise ValueError("Train and validation must be nonempty, episode-disjoint")
    return cache, base, cfg, expert_cfg, initial


def make_plans(args, cache, episodes):
    plans = {"train": [], "val": [], "files": {}, "provenance_groups": {}}
    records = {int(r["episode_id"]): r for r in cache.manifest.get("episodes", [])}
    for split, limit in (("train", args.max_train_episodes), ("val", args.max_val_episodes)):
        ids = list(cache.manifest["splits"][split])
        if limit:
            ids = ids[:limit]
        for eid in ids:
            ep = episodes.fetch(eid)
            if records and "is_demo" not in ep:
                raise ValueError("Real V7 cache requires explicit is_demo metadata")
            decisions = torch.where(ep["decision_mask"])[0].tolist()
            if decisions:
                plans[split].append([int(eid), decisions])
            # This may be an instruction/provenance string, NOT a task ID.
            plans["provenance_groups"][str(eid)] = str(ep.get("task", records.get(eid, {}).get("task", "unknown")))
            if eid in records:
                path = (Path(cache.path) / records[eid]["path"]).resolve()
                stat = path.stat()
                plans["files"][str(eid)] = [str(path), stat.st_size, stat.st_mtime_ns]
        if not plans[split]:
            raise ValueError(f"No valid {split} action queries")
    pool = [[eid, d] for eid, ds in plans["val"] for d in ds]
    rng = random.Random(args.seed + 1701)
    plans["validation"] = rng.sample(pool, min(len(pool), args.val_samples))
    rng = random.Random(args.seed + 1702)
    plans["epoch_validation"] = rng.sample(pool, min(len(pool), args.epoch_val_samples))
    plans["train_query_count"] = sum(len(ds) for _, ds in plans["train"])
    plans["pairing"] = "chronologically_adjacent_queries_then_global_group_shuffle"
    return plans


def coverage_windows(train, epochs, queries_per_prefix, batch_queries, seed):
    """Exactly-once coverage, including first decisions and final partial windows."""
    if min(epochs, queries_per_prefix, batch_queries) <= 0:
        raise ValueError("Coverage dimensions must be positive")
    flat = [(eid, d) for eid, ds in train for d in ds]
    if not flat or len(set(flat)) != len(flat):
        raise ValueError("Coverage requires nonempty unique episode/decision pairs")
    result = []
    for epoch in range(epochs):
        groups = [[eid, sorted(ds)[i:i + queries_per_prefix]] for eid, ds in train
                  for i in range(0, len(ds), queries_per_prefix)]
        random.Random(seed + epoch * 100003 + 77).shuffle(groups)
        current, count = [], 0
        for eid, decisions in groups:
            pending = list(decisions)
            while pending:
                size = min(len(pending), batch_queries - count)
                current.append([eid, pending[:size]])
                pending = pending[size:]
                count += size
                if count == batch_queries:
                    result.append({"epoch": epoch, "groups": current, "query_count": count, "epoch_end": False})
                    current, count = [], 0
        if count:
            result.append({"epoch": epoch, "groups": current, "query_count": count, "epoch_end": False})
        result[-1]["epoch_end"] = True
    return result


def query_seed(seed, epoch, eid, decision, repeat=0, domain="train"):
    return int.from_bytes(hashlib.sha256(json.dumps([seed, epoch, eid, decision, repeat, domain]).encode()).digest()[:8], "big") % (2**63 - 1)


def lr_factor(processed, total, warmup_fraction):
    warm = max(1, int(total * warmup_fraction)) if warmup_fraction else 0
    next_query = min(processed + 1, total)
    if next_query <= warm:
        return next_query / warm
    ratio = (next_query - warm) / max(1, total - warm)
    return .5 * (1 + math.cos(math.pi * min(1., ratio)))


def optimizer_groups(args, memory, head, cvom):
    modules = [("memory", list(memory.named_parameters()))] if args.stage == 1 and args.mode != "none" else []
    if args.stage == 1:
        modules.append(("expert", [(f"lora_{i}", p) for i, p in enumerate(expert_parameters(head))]))
    else:
        modules.append(("cvom", list(cvom.named_parameters())))
    groups = []
    for kind, parameters in modules:
        decay, no_decay = [], []
        for name, parameter in parameters:
            if not parameter.requires_grad:
                continue
            exempt = parameter.ndim <= 1 or any(term in name.lower() for term in ("norm", "address", "token_id", "position", "slot"))
            (no_decay if exempt else decay).append(parameter)
        for suffix, values, wd in (("decay", decay, args.weight_decay), ("no_decay", no_decay, 0.)):
            if values:
                groups.append({"params": values, "name": kind + "_" + suffix, "kind": kind,
                               "lr": getattr(args, kind + "_learning_rate"), "weight_decay": wd})
    return groups


def flow(head, ep, decision, fused=None, **kwargs):
    validate_decision(ep, decision)
    result = expert_episode_flow_loss(head, ep, decision, fused, **kwargs)
    if fused is None:
        metrics = {"conditioning_delta_norm": 0., "conditioning_changed_token_fraction": 0.,
                   "conditioning_changed_value_fraction": 0.}
    else:
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
    out = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            out[key] = float(value.detach())
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


@torch.no_grad()
def validate(args, memory, head, cvom, episodes, plan, draws, domain, baseline_cache):
    grouped = {}
    for eid, d in plan:
        grouped.setdefault(eid, []).append(d)
    rows = []
    for eid, queries in grouped.items():
        ep = episodes.fetch(eid)
        reads = replay_queries(memory, ep, sorted(queries), mode=args.mode, checkpoint_segment=args.checkpoint_segment,
                               cvom=cvom if args.stage == 2 else None, threshold=args.cvom_threshold)
        fixed = replay_queries(memory, ep, sorted(queries), mode=args.mode) if args.stage == 2 else reads
        for decision in queries:
            fused, diag = reads[decision]
            for repeat in range(draws):
                seed = query_seed(args.seed, 0, eid, decision, repeat, domain)
                out = flow(head, ep, decision, fused, seed=seed)
                key = (eid, decision, seed)
                if key not in baseline_cache:
                    with adapter_disabled(head):
                        baseline_cache[key] = float(flow(head, ep, decision, seed=seed)["loss"])
                row = {"action_loss": float(out["loss"]), "loss": float(out["loss"]),
                    "baseline_action_loss": baseline_cache[key], "memory_gain": baseline_cache[key] - float(out["loss"]),
                    "velocity_mae": float(out["velocity_mae"]), **_scalars(diag)}
                row.update({k: v for k, v in out.items() if k.startswith("conditioning_")})
                if args.stage == 2:
                    fixed_loss = float(flow(head, ep, decision, fixed[decision][0], seed=seed)["loss"])
                    row.update(fixed_action_loss=fixed_loss, cvom_gain_over_fixed=fixed_loss - row["action_loss"])
                if args.val_memory_off:
                    off = float(flow(head, ep, decision, seed=seed)["loss"])
                    row.update(expert_no_memory_action_loss=off, memory_gain_over_adapted_expert=off - row["action_loss"])
                if args.val_last_writes:
                    suffix = replay_queries(memory, ep, [decision], mode=args.mode,
                        reset_before=max(0, decision - args.val_last_writes), cvom=cvom if args.stage == 2 else None,
                        threshold=args.cvom_threshold)[decision][0]
                    row["last_writes_reset_action_loss"] = float(flow(head, ep, decision, suffix, seed=seed)["loss"])
                rows.append(row)
    return {**_mean(rows), "queries": float(len(plan)), "noise_draws_per_query": float(draws)}


def main(argv=None):
    args = parse_args(argv)
    cache, base, cfg, expert_cfg, initial = preflight(args)
    episodes = MappedEpisodes(cache)
    plans = make_plans(args, cache, episodes)
    windows = coverage_windows(plans["train"], args.max_epochs, args.queries_per_prefix, args.query_batch_size, args.seed)
    if args.stage == 1 and args.max_steps:
        windows = windows[:args.max_steps]
    plan_sha = _digest({"plans": plans, "windows": windows})
    if args.resume and initial["metadata"]["plan_sha256"] != plan_sha:
        raise ValueError("Coverage/validation/cache files changed after checkpoint")
    horizon = len(windows) if args.stage == 1 else args.cvom_rounds * args.cvom_steps_per_round
    if args.stage == 2 and args.max_steps:
        horizon = min(horizon, args.max_steps)
    start = initial["step"] if args.resume else 0
    if start >= horizon:
        raise ValueError("Planned training horizon already completed; use weights-only initialization for a NEW experiment")
    if args.preflight_only:
        print(json.dumps({"variant": VARIANT, "stage": args.stage, "mode": args.mode,
            "memory": asdict(cfg), "expert": asdict(expert_cfg), "train_queries_per_epoch": plans["train_query_count"],
            "planned_updates": horizon, "plan_sha256": plan_sha, "note": "Read-only; no model/output/training started"}, indent=2))
        return 0
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
        memory, cvom = RecurrentMemoryV7(cfg).to(args.device), CVOMV7(cfg).to(args.device)
    base, _ = load_frozen_hamlet(base_path, args.device)
    head = base.action_head
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with isolated_seed(args.seed + 101, args.device):
        targets = install_expert_lora(head, expert_cfg, targets=initial["config"]["expert_targets"] if initial else None)
    if initial:
        load_checkpoint_v7(args.resume or args.init_checkpoint, memory, head, cvom)
    memory.requires_grad_(args.stage == 1 and args.mode != "none").eval()
    cvom.requires_grad_(args.stage == 2).eval()
    set_expert_trainable(head, args.stage == 1)
    config = {"trainer_variant": VARIANT, "stage": args.stage, "mode": args.mode,
        "memory": asdict(cfg), "expert": asdict(expert_cfg), "expert_targets": targets, "train": vars(args).copy()}
    state = copy.deepcopy(initial["metadata"]["train_state"]) if args.resume else {
        "processed_queries": 0, "optimizer_updates": 0, "epoch": 0, "window_cursor": 0,
        "best_action_loss": None, "best_checkpoint": None, "elapsed_seconds": 0., "round": -1, "status": "initialized"}
    metadata = {"base_model": checkpoint_identity(base_path), "cache_fingerprint": cache.manifest["fingerprint"],
        "cache_dir": str(Path(args.cache_dir).resolve()), "source_sha256": source_identity(), "runtime": runtime_identity(),
        "plan_sha256": plan_sha, "plans": plans, "train_state": state,
        "initial_checkpoint": str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None,
        "resumed_from": str(Path(args.resume).resolve()) if args.resume else None,
        "note": "Adjacent query pairs/full-prefix recomputation; offline losses are not task success."}
    if args.stage == 2:
        source = Path(args.init_checkpoint or args.resume).resolve()
        metadata["stage1_parent"] = copy.deepcopy(initial["metadata"]["stage1_parent"]) if args.resume else {
            "path": str(source), **{key: hashlib.sha256((source / filename).read_bytes()).hexdigest()
            for key, filename in (("checkpoint_sha256", "checkpoint.json"), ("memory_sha256", "model.safetensors"),
                                  ("expert_sha256", "expert.safetensors"))}}
        metadata.update(frozen_actor_sha256=actor_state_sha256(memory), frozen_expert_sha256=expert_state_sha256(head))
    optimizer = None if args.stage2_opportunity_only else torch.optim.AdamW(optimizer_groups(args, memory, head, cvom))
    if args.resume:
        load_checkpoint_v7(args.resume, memory, head, cvom, optimizer)
    else:
        _seed(args.seed + 102)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    logger, baseline_cache = RunLogger(output), {}
    _atomic_json(output / "run_config.json", config)
    _atomic_json(output / "query_plan.json", {"sha256": plan_sha, "windows": windows, "plans": plans})
    _atomic_json(output / "provenance.json", metadata)
    if args.resume and state["best_checkpoint"]:
        _atomic_json(output / "best_checkpoint.json", {"path": state["best_checkpoint"], "inherited": True})
    print(f"[v7] stage={args.stage} mode={args.mode} planned_updates={horizon} queries/epoch={plans['train_query_count']}", flush=True)
    print(f"[v7] trainable memory={sum(p.numel() for p in memory.parameters() if p.requires_grad)} "
          f"expert={sum(p.numel() for p in expert_parameters(head) if p.requires_grad)} "
          f"cvom={sum(p.numel() for p in cvom.parameters() if p.requires_grad)}", flush=True)
    print("[v7] Chronological adjacent query pairs; each prefix resets/replays from zero; no H2 detach", flush=True)
    started, elapsed_before = time.monotonic(), state["elapsed_seconds"]
    label_cfg = LabelV7Config(future_samples=args.future_samples, noise_samples=args.noise_samples,
                             label_scale=args.label_scale, threshold=args.cvom_threshold,
                             memory_window=int(json.loads((Path(base_path) / "config.json").read_text())["memory_window"]))
    cvom_labels, cvom_val_labels = [], []

    def audit_identity():
        return {"actor": actor_state_sha256(memory), "expert": expert_state_sha256(head),
            "cache": cache.manifest["fingerprint"], "label_config": asdict(label_cfg),
            "train_episodes": [eid for eid, _ in plans["train"]], "val_episodes": [eid for eid, _ in plans["val"]]}

    def prepare_round(round_index, audit=False):
        nonlocal cvom_labels, cvom_val_labels
        folder = output / ("opportunity" if audit else f"round-{round_index:02d}")
        folder.mkdir()
        teacher = None
        if round_index > 0:
            teacher = copy.deepcopy(cvom).eval().requires_grad_(False)
            if args.resume and state.get("round") == round_index:
                teacher_path = Path(state["teacher_snapshot"])
                if hashlib.sha256(teacher_path.read_bytes()).hexdigest() != state["teacher_sha256"]:
                    raise ValueError("Frozen CVOM round teacher changed")
                teacher.load_state_dict(load_file(str(teacher_path)), strict=True)
            else:
                teacher_path = folder / "teacher_cvom.safetensors"
                save_file({k: v.detach().cpu().contiguous() for k, v in teacher.state_dict().items()}, str(teacher_path))
                state.update(teacher_snapshot=str(teacher_path), teacher_sha256=hashlib.sha256(teacher_path.read_bytes()).hexdigest())
        label_lists = []
        for split, count in (("train", args.cvom_contexts), ("val", args.cvom_val_contexts)):
            ids = [eid for eid, _ in plans[split]]
            contexts = make_storage_contexts(episodes, ids, count, args.seed + round_index * 100003 + (0 if split == "train" else 50000),
                                             future_samples=args.future_samples, memory_window=label_cfg.memory_window)
            _atomic_json(folder / f"{split}_contexts.json", contexts)
            labels = []
            label_start = time.monotonic()
            print(f"[v7][labels] round={round_index} split={split} contexts={len(contexts)}", flush=True)
            for index, context in enumerate(contexts):
                label = build_storage_label(memory, teacher, head, episodes.fetch(context["episode_id"]), context, label_cfg, audit=audit or split == "val")
                labels.append(label)
                torch.save(label, folder / f"{split}-label-{index:05d}.pt")
                print(f"[v7][labels] round={round_index} split={split} {index+1}/{len(contexts)} "
                      f"elapsed={time.monotonic()-label_start:.1f}s", flush=True)
            label_lists.append(labels)
        cvom_labels, cvom_val_labels = label_lists
        if not cvom_labels or not cvom_val_labels:
            raise ValueError("No episode-disjoint train/val CVOM contexts with future queries")
        summary = {"identity": audit_identity(), "round": round_index,
            "teacher_policy": "always_update" if teacher is None else cvom_state_sha256(teacher),
            "train": summarize_storage_labels(cvom_labels), "val": summarize_storage_labels(cvom_val_labels),
            "note": "Teacher-forced recorded-trajectory opportunity audit; not task-success improvement."}
        _atomic_json(folder / "summary.json", summary)
        if audit:
            _atomic_json(output / "opportunity_audit.json", summary)
        return summary

    if args.stage == 2:
        if args.stage2_opportunity_only:
            prepare_round(0, audit=True)
            print(f"[v7] Opportunity audit saved; no optimizer/training started: {output / 'opportunity_audit.json'}", flush=True)
            return 0
        summary = prepare_round(start // args.cvom_steps_per_round)
        _atomic_json(output / "opportunity_audit.json", summary)
        state["round"] = start // args.cvom_steps_per_round

    def evaluate(step, epoch_end=False):
        metrics = validate(args, memory, head, cvom, episodes, plans["validation"], args.val_noise_samples,
                           "monitor-validation", baseline_cache)
        logger.log(step, "val", metrics)
        if epoch_end and plans["epoch_validation"]:
            logger.log(step, "val-epoch", validate(args, memory, head, cvom, episodes, plans["epoch_validation"],
                args.epoch_val_noise_samples, "epoch-validation", baseline_cache))
        if args.stage == 2:
            with torch.no_grad():
                rows = []
                for label in cvom_val_labels:
                    result = storage_loss(cvom, memory, label, label_cfg)
                    rows.append({"loss": float(result["loss"]), **_scalars(result["metrics"])})
                logger.log(step, "val-cvom", {**_mean(rows), "round": float(state["round"])})
        print(f"[v7][val] step={step} action={metrics['action_loss']:.7f} original={metrics['baseline_action_loss']:.7f}", flush=True)
        return metrics

    def save(step, best=False):
        if args.stage == 2 and (actor_state_sha256(memory) != metadata["frozen_actor_sha256"] or expert_state_sha256(head) != metadata["frozen_expert_sha256"]):
            raise RuntimeError("Frozen actor/Expert changed during CVOM learning")
        if best:
            state["best_checkpoint"] = str(output / f"checkpoint-{step:06d}")
        path = save_checkpoint_v7(output, step, memory, head, cvom, optimizer, config, metadata,
            best=best, training_state={"window_cursor": state["window_cursor"], "plan_sha256": plan_sha})
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
            if args.stage == 2:
                current_round = (step - 1) // args.cvom_steps_per_round
                if current_round != state["round"]:
                    prepare_round(current_round)
                    state["round"] = current_round
            optimizer.zero_grad(set_to_none=True)
            multiplier = lr_factor(state["processed_queries"], planned_queries, args.warmup_fraction) if args.stage == 1 else 1.
            for group in optimizer.param_groups:
                group["lr"] = getattr(args, group["kind"] + "_learning_rate") * multiplier
            rows, epoch_end = [], False
            if args.stage == 1:
                window = windows[step - 1]
                if state["window_cursor"] != step - 1:
                    raise RuntimeError("Coverage cursor is not at an optimizer boundary")
                for eid, queries in window["groups"]:
                    ep = episodes.fetch(eid)
                    reads = replay_queries(memory, ep, queries, mode=args.mode, checkpoint_segment=args.checkpoint_segment)
                    losses = []
                    for decision in queries:
                        fused, diag = reads[decision]
                        result = flow(head, ep, decision, fused,
                            seed=query_seed(args.seed, window["epoch"], eid, decision),
                            activation_checkpointing=args.activation_checkpointing)
                        if not torch.isfinite(result["loss"]) or not result["loss"].requires_grad:
                            raise FloatingPointError("Nonfinite or disconnected action loss")
                        losses.append(result["loss"])
                        rows.append({"action_loss": float(result["loss"].detach()), "loss": float(result["loss"].detach()),
                            "velocity_mae": float(result["velocity_mae"].detach()), **_scalars(diag)})
                        rows[-1].update({k: v for k, v in result.items() if k.startswith("conditioning_")})
                    (torch.stack(losses).sum() / window["query_count"]).backward()
                state["processed_queries"] += window["query_count"]
                state["window_cursor"] = step
                state["epoch"] = window["epoch"] + int(window["epoch_end"])
                epoch_end = window["epoch_end"]
            else:
                for _ in range(args.cvom_batch_size):
                    label = random.choice(cvom_labels)
                    result = storage_loss(cvom, memory, label, label_cfg)
                    if not torch.isfinite(result["loss"]) or not result["loss"].requires_grad:
                        raise FloatingPointError("Nonfinite or disconnected CVOM loss")
                    (result["loss"] / args.cvom_batch_size).backward()
                    rows.append({"loss": float(result["loss"].detach()), **_scalars(result["metrics"])})
            row = _mean(rows)
            row.update(memory_grad_norm=_grad_norm(memory.parameters()), expert_grad_norm=_grad_norm(expert_parameters(head)),
                       cvom_grad_norm=_grad_norm(cvom.parameters()))
            for name, fragments in (("write_ffn", ("write_ffn", "ffn_w")), ("update_gate", ("update_gate", "gate_mlp")),
                                    ("shared_attention", ("attention", "attn")), ("fusion", ("fusion",))):
                row[name + "_grad_norm"] = _grad_norm(p for n, p in memory.named_parameters() if any(f in n for f in fragments))
            norm = torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], args.max_grad_norm, error_if_nonfinite=True)
            # Zero signed target is legitimate regression, not a missing label.
            optimizer.step()
            state["optimizer_updates"] += 1
            state["elapsed_seconds"] = elapsed_before + time.monotonic() - started
            row.update(grad_norm=float(norm), learning_rate=optimizer.param_groups[0]["lr"], coverage_epoch=float(state["epoch"]),
                processed_queries=float(state["processed_queries"]), update_seconds=time.monotonic() - tick,
                elapsed_seconds=state["elapsed_seconds"])
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                row["peak_vram_bytes"] = float(torch.cuda.max_memory_allocated(args.device))
            interval.append(row)
            if step == 1 or step % args.log_steps == 0 or step == end:
                logger.log(step, "train", _mean(interval)); interval = []
                print(f"[v7][train] stage={args.stage} step={step}/{horizon} queries={state['processed_queries']} loss={row['loss']:.7f}", flush=True)
            best = False
            due_eval = step % args.eval_steps == 0 or step == end or epoch_end
            if due_eval:
                metrics = evaluate(step, epoch_end)
                if metrics["action_loss"] < state["best_action_loss"]:
                    state["best_action_loss"] = metrics["action_loss"]; best = True
            state["status"] = "complete" if step == horizon else "paused" if step == end else "training"
            if step % args.save_steps == 0 or step == end or epoch_end or best:
                save(step, best)
                if epoch_end:
                    _atomic_json(output / f"epoch-{state['epoch']:03d}_checkpoint.json", {"path": str(output / f"checkpoint-{step:06d}")})
            _atomic_json(output / "status.json", {"step": step, **state})
            if step % args.plot_steps == 0 or due_eval:
                logger.plot()
    except KeyboardInterrupt:
        print("[v7] Interrupted. Only previously saved optimizer-boundary checkpoints may be resumed.", flush=True)
        return 130
    print(f"[v7] {state['status']}: {output}; offline loss is not RoboMME success", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
