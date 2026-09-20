"""Contextual Stage-2 v2; the v1 trainer and memory architecture stay unchanged.

This is still Stage 2, with two internal phases: frozen-reader head bootstrap,
then validation-gated low-LR reader training. Hard writes are never treated as
differentiable. Each auxiliary target and prediction use exactly the same past
bank. Frozen teacher snapshots are refreshed periodically, NOT updated by EMA.
See docs/LONG_MEMORY_STAGE2_V2.md before interpreting the offline metrics.
"""

import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .cache import EpisodeCache
from .contextual_cvom import ContextualCVoMConfig, ContextualCVoMLabels, eligible_candidates
from .core import EpisodicMemory, MemoryConfig
from .hamlet import checkpoint_identity, episode_flow_loss, load_frozen_hamlet, validate_cache_checkpoint
from .monitoring import RunLogger, _atomic_json, load_checkpoint, save_checkpoint
from .replay import encode_until, read_bank, replay_bank
from .stage2_objectives import (contextual_predictions, fixed_bank, parameter_groups,
                                prediction_metrics, readiness, set_trainable_phase,
                                weighted_auxiliary_loss)
from .stage2_validation import action_guard, validate_actions


VARIANT = "contextual_stage2_v2"
RAW_KEYS = ("episode_id", "frames", "short", "moment", "state", "actions",
            "action_mask", "transition_valid", "decision_mask")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-checkpoint", help="Unmodified Stage-1 memory checkpoint")
    source.add_argument("--resume", help="V2 checkpoint including optimizer/RNG; max-steps is total")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-model", help="Defaults to immutable checkpoint recorded in cache")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--bootstrap-steps", type=int, default=500)
    p.add_argument("--bootstrap-max-steps", type=int, default=1500)
    p.add_argument("--writer-only", action="store_true", help="Diagnostic head-only run, never unfreeze reader")
    p.add_argument("--head-learning-rate", type=float, default=1e-4)
    p.add_argument("--reader-learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--aux-batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=2, help="Action microbatches per joint update")
    p.add_argument("--joint-aux-weight", type=float, default=0.1)
    p.add_argument("--train-contexts", type=int, default=1024)
    p.add_argument("--val-contexts", type=int, default=256)
    p.add_argument("--val-action-samples", type=int, default=64)
    p.add_argument("--future-samples", type=int, default=4)
    p.add_argument("--noise-samples", type=int, default=4)
    p.add_argument("--utility-scale", type=float, default=0.001)
    p.add_argument("--write-delta", type=float, default=1e-5)
    p.add_argument("--uncertainty-z", type=float, default=2.0)
    p.add_argument("--min-writer-auc", type=float, default=0.55)
    p.add_argument("--min-confident-per-class", type=int, default=8)
    p.add_argument("--action-guard-tolerance", type=float, default=0.05)
    p.add_argument("--teacher-refresh-steps", type=int, default=500)
    p.add_argument("--policy-ramp-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--plot-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--preflight-only", action="store_true", help="Read provenance only; no model, labels, training or output writes")
    return p


def validate_args(args):
    for name in ("max_steps", "bootstrap_steps", "bootstrap_max_steps", "aux_batch_size", "grad_accum",
                 "train_contexts", "val_contexts", "val_action_samples", "future_samples", "noise_samples",
                 "min_confident_per_class", "teacher_refresh_steps", "policy_ramp_steps", "eval_steps",
                 "log_steps", "plot_steps", "save_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.bootstrap_max_steps < args.bootstrap_steps:
        raise ValueError("bootstrap-max-steps must be >= bootstrap-steps")
    for name in ("head_learning_rate", "reader_learning_rate", "max_grad_norm", "utility_scale"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "joint_aux_weight", "write_delta", "uncertainty_z", "action_guard_tolerance"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0.5 <= args.min_writer_auc <= 1 or args.warmup_steps < 0:
        raise ValueError("Require min-writer-auc in [0.5,1] and warmup-steps >= 0")
    if args.noise_samples < 2:
        raise ValueError("noise-samples must be >= 2")


def preflight(args):
    """Cheap, read-only compatibility checks; no scan of large episode tensors."""
    validate_args(args)
    cache = EpisodeCache(args.cache_dir)
    manifest = cache.manifest
    base = str(Path(args.base_model or manifest["model_path"]).resolve())
    if Path(base) != Path(manifest["model_path"]).resolve():
        raise ValueError("Base differs from cached-feature provenance; rebuild cache for a different base")
    validate_cache_checkpoint(manifest)
    identity = checkpoint_identity(base)
    source = Path(args.resume or args.init_checkpoint).resolve()
    info = json.loads((source / "checkpoint.json").read_text())
    if not (source / "model.safetensors").is_file():
        raise ValueError("Memory weights missing")
    if (info.get("format_version") != 1 or info["metadata"]["base_model"] != identity
            or info["metadata"]["cache_fingerprint"] != manifest["fingerprint"]):
        raise ValueError("Memory initialization/resume provenance differs from cache/base")
    if args.resume:
        if info["config"].get("trainer_variant") != VARIANT or info["config"]["stage"] != 2:
            raise ValueError("V1 Stage 2 cannot resume as v2. Start v2 with --init-checkpoint from Stage 1")
        if not (source / "training_state.pt").is_file() or args.max_steps <= info["step"]:
            raise ValueError("Resume needs optimizer/RNG and max-steps > saved step")
        allowed = {"resume", "init_checkpoint", "output_dir", "max_steps", "base_model", "preflight_only"}
        for name, value in vars(args).items():
            if name not in allowed and info["config"]["train"].get(name) != value:
                raise ValueError(f"Exact resume option changed: {name}; use original options")
    elif info["config"]["stage"] != 1:
        raise ValueError("V2 must initialize from a Stage-1 reader, not previously trained v1 heads")
    cfg = MemoryConfig(**info["config"]["memory"])
    if not 1 <= cfg.min_fill < cfg.capacity:
        raise ValueError("Contextual addition training needs 1 <= min_fill < capacity")
    for name in ("feature_dim", "state_dim", "action_dim"):
        if getattr(cfg, name) != manifest[name]:
            raise ValueError(f"Cache/memory dimensions differ: {name}")
    train, val = manifest["splits"]["train"], manifest["splits"]["val"]
    if not train or not val or set(train) & set(val):
        raise ValueError("Require nonempty disjoint episode train/val splits")
    return cache, info, cfg, base, identity


def _mean(rows):
    keys = set().union(*(r.keys() for r in rows))
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}


def make_plans(fetch, splits, cfg, memory_window, args, inherited_validation):
    """Episode-balanced candidates; context sampling never depends on future GT.

    A fixed finite auxiliary set makes repeated-noise labeling affordable. This
    is NOT one pass over every event of every episode. Action updates continue
    to sample from the entire training split. Only learned-write opportunities
    (enough past events to fill min_fill) enter auxiliary supervision.
    """
    pools, action_train, action_val = {}, [], []
    for split, ids in splits.items():
        pool = []
        for index, eid in enumerate(ids):
            ep = fetch(eid)
            candidates = [i for i in eligible_candidates(ep, memory_window)
                          if int(ep["transition_valid"][:i].sum()) >= cfg.min_fill]
            if candidates:
                pool.append((int(eid), candidates))
            decisions = torch.where(ep["decision_mask"])[0].tolist()
            if split == "train":
                decisions = [d for d in decisions if bool(ep["transition_valid"][:d].any())]
                if decisions:
                    action_train.append([int(eid), decisions])
            else:
                action_val.extend([int(eid), d, None] for d in decisions)
            if (index + 1) % 100 == 0:
                print(f"[v2][index] {split} {index+1}/{len(ids)} episodes", flush=True)
        if not pool:
            raise ValueError(f"No old-only post-min-fill candidate in {split}; increase data/episode length")
        pools[split] = pool
    if not action_train or not action_val:
        raise ValueError("No eligible training/validation action queries")
    rng = random.Random(args.seed + 9001)
    contexts = {}
    for split, number in (("train", args.train_contexts), ("val", args.val_contexts)):
        rows, seen = [], set()
        for _ in range(number * 30):
            eid, candidates = rng.choice(pools[split])
            candidate = rng.choice(candidates)
            ep = fetch(eid)
            past_count = int(ep["transition_valid"][:candidate].sum())
            size = rng.randint(cfg.min_fill, min(past_count, cfg.capacity - 1))
            policy = rng.choice(("first", "fifo", "random"))
            bank = fixed_bank(ep, candidate, size, policy=policy, seed=rng.randrange(2**31))
            key = (eid, candidate, tuple(bank))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"episode_id": eid, "candidate": candidate, "bank_ids": bank})
            if len(rows) == number:
                break
        contexts[split] = rows
        print(f"[v2][plan] {split} contexts={len(rows)}/{number}; eligible episodes={len(pools[split])}", flush=True)
    # Preserve the Stage-1 paired seeds/query order on a documented prefix.
    if inherited_validation:
        valid_keys = {(r[0], r[1]) for r in action_val}
        if any((r[0], r[1]) not in valid_keys for r in inherited_validation):
            raise ValueError("Inherited validation plan is not in the current val split")
        plan = inherited_validation[:args.val_action_samples]
    else:
        plan = rng.sample(action_val, min(args.val_action_samples, len(action_val)))
    return {"contexts": contexts, "action_train": action_train, "validation": plan}


@torch.no_grad()
def feature_pack(memory, ep, context):
    """Detached tiny head inputs. Safe to cache ONLY while reader is frozen."""
    output = contextual_predictions(memory, ep, context["candidate"], context["bank_ids"])
    return {k: output[k].detach().cpu() for k in ("event", "short", "read", "novelty")}


def predict_packs(memory, packs):
    device = next(memory.parameters()).device
    batch = {k: torch.cat([p[k] for p in packs], dim=0).to(device) for k in packs[0]}
    utility = memory.utility(batch["event"], batch["short"], batch["read"])
    return utility, memory.write_logits(utility, batch["novelty"])


def prepare_round(memory, teacher, head, fetch, plans, label_config, output, version, fingerprint):
    labels = ContextualCVoMLabels(teacher, head, label_config,
                                 output / "contextual_labels" / f"teacher-{version:06d}",
                                 {"cache_fingerprint": fingerprint, "teacher_version": version})
    records, packs = {}, {}
    for split in ("train", "val"):
        rows = plans["contexts"][split]
        records[split], packs[split] = [], []
        started = time.monotonic()
        for i, context in enumerate(rows):
            ep = fetch(context["episode_id"])
            records[split].append(labels.get(ep, context["candidate"], context["bank_ids"]))
            packs[split].append(feature_pack(memory, ep, context))
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(rows):
                print(f"[v2][labels] teacher={version} {split} {i+1}/{len(rows)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        rows = records[split]
        summary = {"count": len(rows), "confident": sum(r["write_weight"] > 0 for r in rows),
                   "confident_positive": sum(r["write_weight"] > 0 and r["write_target"] == 1 for r in rows),
                   "mean_signed_gain": float(np.mean([r["signed_gain"] for r in rows])),
                   "target_quantiles": np.quantile([r["utility_target"] for r in rows], [0, .5, .9, .99, 1]).tolist()}
        _atomic_json(output / f"labels-{version:06d}-{split}-summary.json", summary)
        print(f"[v2][labels] {split} {summary}", flush=True)
    return records, packs


@torch.no_grad()
def validate_heads(memory, contexts, records, packs, raw_fetch, bootstrap, batch_size):
    was_training = memory.training
    memory.eval()
    utilities, logits = [], []
    try:
        for start in range(0, len(records), batch_size):
            batch = packs[start:start+batch_size] if bootstrap else [
                feature_pack(memory, raw_fetch(c["episode_id"]), c) for c in contexts[start:start+batch_size]]
            u, w = predict_packs(memory, batch)
            utilities.extend(u.cpu().tolist())
            logits.extend(w.cpu().tolist())
        return prediction_metrics(utilities, logits, records, threshold=memory.config.write_threshold)
    finally:
        memory.train(was_training)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def output_lock(output):
    """One writer per run; OS releases the lock on crash or Ctrl+C."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another trainer is writing to {output}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main(argv=None):
    args = parser().parse_args(argv)
    cache, initial, cfg, base_path, identity = preflight(args)
    output = Path(args.output_dir).resolve()
    if args.preflight_only:
        print(json.dumps({"preflight": "PASS (provenance only; tensors not scanned)",
                          "trainer": VARIANT, "base": base_path, "output": str(output),
                          "train_episodes": len(cache.manifest["splits"]["train"]),
                          "val_episodes": len(cache.manifest["splits"]["val"]),
                          "capacity": cfg.capacity, "min_fill": cfg.min_fill,
                          "new_training_started": False}, indent=2))
        return 0
    with output_lock(output):
        return run_training(args, cache, initial, cfg, base_path, identity, output)


def run_training(args, cache, initial, cfg, base_path, identity, output):
    _seed(args.seed)
    config = {"stage": 2, "trainer_variant": VARIANT, "memory": asdict(cfg), "train": vars(args).copy()}
    config["train"]["base_model"] = base_path
    config["train"]["preflight_only"] = False
    # A failed preparation may be rerun with the SAME command. Once a checkpoint
    # exists, --resume is mandatory. No old run is truncated or overwritten.
    run_identity = {"config": config, "base_model": identity,
                    "cache_fingerprint": cache.manifest["fingerprint"],
                    "source_weights_sha256": _file_hash(Path(args.resume or args.init_checkpoint) / "model.safetensors")}
    if any(p.name != ".training.lock" for p in output.iterdir()):
        if args.resume:
            if Path(args.resume).resolve().parent != output:
                raise ValueError("A resume fork requires a NEW empty output directory")
        elif ((output / "run_identity.json").is_file()
              and json.loads((output / "run_identity.json").read_text()) == run_identity
              and not list(output.glob("checkpoint-*"))):
            print("[v2] Resuming interrupted preparation, before any optimizer checkpoint", flush=True)
        else:
            raise ValueError("Use a NEW output directory or explicit --resume; existing runs are preserved")
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        _atomic_json(output / "run_identity.json", run_identity)
    start = initial["step"] if args.resume else 0
    logger = RunLogger(output)
    if any(r["step"] > start for r in logger.records):
        raise ValueError("Logs newer than resume checkpoint; fork to a NEW output directory")

    @lru_cache(maxsize=4)
    def fetch(eid):
        return cache.load(int(eid))

    @lru_cache(maxsize=64)
    def raw_fetch(eid):
        # Keep only event tensors, not large full VL feature lists, in this LRU.
        ep = fetch(eid)
        return {k: ep[k] for k in RAW_KEYS}

    base_config = json.loads((Path(base_path) / "config.json").read_text())
    label_config = ContextualCVoMConfig(memory_window=int(base_config["memory_window"]),
        future_samples=args.future_samples, noise_samples=args.noise_samples,
        utility_scale=args.utility_scale, write_delta=args.write_delta,
        uncertainty_z=args.uncertainty_z, seed=args.seed + 77)
    plans_path = output / "context_plan.json"
    if args.resume:
        plans = initial["metadata"]["plans"]
    elif plans_path.exists():
        plans = json.loads(plans_path.read_text())
    else:
        plans = make_plans(fetch, cache.manifest["splits"], cfg, label_config.memory_window,
                           args, initial["metadata"].get("validation_plan"))
    _atomic_json(plans_path, plans)
    metadata = copy.deepcopy(initial["metadata"]) if args.resume else {
        "base_model": identity, "cache_fingerprint": cache.manifest["fingerprint"],
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "stage1_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "stage1_weights_sha256": run_identity["source_weights_sha256"],
        "utility_target": "log1p-positive-contextual-old-only-gain-v2",
        "note": "Offline validation is not robot task success. Addition utility is not replacement value.",
        "v2_state": {"phase": "bootstrap", "phase_start": None, "teacher_version": 0,
                     "teacher_step": 0, "last_gate_pass": False, "gate_reasons": ["not validated"],
                     "best_joint_action_loss": None}}
    metadata.update(plans=plans, validation_plan=plans["validation"])
    state = metadata["v2_state"]
    memory = EpisodicMemory(cfg).to(args.device)
    load_checkpoint(args.resume or args.init_checkpoint, memory)
    set_trainable_phase(memory, joint=state["phase"] == "joint")
    optimizer = torch.optim.AdamW(parameter_groups(memory, args.head_learning_rate, args.reader_learning_rate),
                                   weight_decay=args.weight_decay)
    base, _ = load_frozen_hamlet(base_path, args.device)
    head = base.action_head
    del base  # Frozen cached VLM is not kept on training GPU.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def teacher_snapshot(version, step, saved_path=None):
        teacher = copy.deepcopy(memory).eval().requires_grad_(False)
        if saved_path:
            saved_path = Path(saved_path)
            if _file_hash(saved_path / "model.safetensors") != state["teacher_weights_sha256"]:
                raise ValueError("Frozen teacher snapshot changed since checkpoint")
            load_checkpoint(saved_path, teacher)
            return teacher
        root = output / "teachers"
        path = root / f"checkpoint-{step:06d}"
        if path.exists():
            # Interrupted label preparation can reuse this immutable snapshot
            # only if its weights equal the intended current teacher exactly.
            original = copy.deepcopy(teacher.state_dict())
            load_checkpoint(path, teacher)
            if any(not torch.equal(original[k], v) for k, v in teacher.state_dict().items()):
                raise ValueError("Existing teacher snapshot differs; use a fresh output directory")
        else:
            save_checkpoint(root, step, teacher, None, config,
                            {"base_model": identity, "teacher_version": version,
                             "cache_fingerprint": cache.manifest["fingerprint"]}, keep_last=None)
        state.update(teacher_checkpoint=str(path), teacher_weights_sha256=_file_hash(path / "model.safetensors"))
        return teacher

    teacher = teacher_snapshot(state["teacher_version"], state["teacher_step"],
                               state["teacher_checkpoint"] if args.resume else None)
    records, packs = prepare_round(memory, teacher, head, fetch, plans, label_config, output,
                                   state["teacher_version"], cache.manifest["fingerprint"])
    if args.resume:
        # Restore all RNG AFTER any model/preparation work, including a fork.
        load_checkpoint(args.resume, memory, optimizer)
    else:
        _seed(args.seed)
    _atomic_json(output / "run_config.json", config)
    _atomic_json(output / "provenance.json", metadata)

    def evaluate(step):
        metrics = validate_heads(memory, plans["contexts"]["val"], records["val"], packs["val"],
                                 raw_fetch, state["phase"] == "bootstrap", args.aux_batch_size)
        metrics.update(validate_actions(memory, head, fetch, plans["validation"], args.seed + 50000))
        ready, reasons = readiness(metrics, args.min_writer_auc, args.min_confident_per_class)
        safe, action_reasons = action_guard(metrics, args.action_guard_tolerance)
        if "stage1_all_action_loss" not in state:
            # First evaluation happens before ANY updates to the Stage-1
            # reader. Keep this fixed reference when both current variants drift.
            if step != 0 or args.resume:
                raise ValueError("Missing frozen Stage-1 validation reference in v2 checkpoint")
            state["stage1_all_action_loss"] = metrics["all_action_loss"]
        metrics["stage1_all_action_loss"] = state["stage1_all_action_loss"]
        if metrics["action_loss"] > state["stage1_all_action_loss"] * (1 + args.action_guard_tolerance):
            safe = False
            action_reasons.append("hard-bank action loss exceeds fixed initial Stage-1 all/FIFO reference")
        # Writer health on synthetic coalitions alone is insufficient: require
        # nontrivial actual hard-replay accepts AND rejects after forced fill.
        accepted, attempts = metrics["learned_write_accepts"], metrics["learned_write_attempts"]
        if not 0 < accepted < attempts:
            reasons.append("actual hard replay has no learned accepts or no learned rejects")
            ready = False
        state.update(last_gate_pass=bool(ready and safe), gate_reasons=reasons + action_reasons)
        metrics.update(phase_joint=float(state["phase"] == "joint"),
                       teacher_version=float(state["teacher_version"]), gate_pass=float(state["last_gate_pass"]))
        logger.log(step, "val", metrics)
        print(f"[v2][val] step={step} phase={state['phase']} action={metrics['action_loss']:.6f} "
              f"write_auc={metrics.get('write_auc', 'undefined')} learned_accepts={accepted}/{attempts} "
              f"gate={state['last_gate_pass']} reasons={state['gate_reasons']}", flush=True)
        return metrics

    if not args.resume:
        evaluate(0)
        save_checkpoint(output, 0, memory, optimizer, config, metadata, keep_last=None)
        logger.plot()
    last_saved, interval, started = start, [], time.monotonic()
    memory.train()
    try:
        for step in range(start + 1, args.max_steps + 1):
            # Refresh at the next update after N JOINT updates. Both labels and
            # their teacher version are replaced; old caches remain on disk.
            joint_steps = (step - 1 - state["phase_start"]) if state["phase"] == "joint" else 0
            if (joint_steps > 0 and joint_steps % args.teacher_refresh_steps == 0
                    and state["teacher_step"] != step - 1):
                state.update(teacher_version=state["teacher_version"] + 1, teacher_step=step - 1)
                teacher = teacher_snapshot(state["teacher_version"], step - 1)
                records, packs = prepare_round(memory, teacher, head, fetch, plans, label_config, output,
                                               state["teacher_version"], cache.manifest["fingerprint"])
            joint = state["phase"] == "joint"
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                base_lr = args.head_learning_rate if group["name"] == "heads" else args.reader_learning_rate
                age = step if group["name"] == "heads" else max(0, step - (state["phase_start"] or step))
                group["lr"] = base_lr * min(1., age / max(1, args.warmup_steps))
            indices = [random.randrange(len(records["train"])) for _ in range(args.aux_batch_size)]
            batch_packs = [feature_pack(memory, raw_fetch(plans["contexts"]["train"][i]["episode_id"]),
                                        plans["contexts"]["train"][i]) if joint else packs["train"][i]
                           for i in indices]
            u, w = predict_packs(memory, batch_packs)
            auxiliary = weighted_auxiliary_loss(u, w, [records["train"][i] for i in indices])
            aux_weight = args.joint_aux_weight if joint else 1.
            (auxiliary["loss"] * aux_weight).backward()
            row = {k: float(v.detach()) for k, v in auxiliary.items()}
            row["loss"] *= aux_weight
            hard_fraction = min(1., (step - state["phase_start"]) / args.policy_ramp_steps) if joint else 0.
            action_rows = []
            if joint:
                for _ in range(args.grad_accum):
                    eid, decisions = random.choice(plans["action_train"])
                    ep, decision = fetch(eid), random.choice(decisions)
                    policy = "hard" if random.random() < hard_fraction else "all"
                    # Selection is no_grad; selected values are recomputed with
                    # autograd. Only event/read/fusion receives action gradients.
                    encoded = encode_until(memory, ep, decision)
                    bank, stats = replay_bank(memory, ep, decision, policy, encoded)
                    read = read_bank(memory, ep, decision, bank, encoded)
                    action = episode_flow_loss(head, ep, decision, read["fused_short"],
                                               activation_checkpointing=args.activation_checkpointing)
                    if not torch.isfinite(action["loss"]) or not action["loss"].requires_grad:
                        raise FloatingPointError("Nonfinite/disconnected action loss")
                    (action["loss"] / args.grad_accum).backward()
                    action_rows.append({"action_loss": float(action["loss"].detach()), **stats,
                                        "velocity_mae": float(action["velocity_mae"].detach())})
                row.update(_mean(action_rows))
                row["loss"] += row["action_loss"]
            norm = torch.nn.utils.clip_grad_norm_(memory.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            row.update(grad_norm=float(norm), phase_joint=float(joint), hard_fraction=hard_fraction,
                       teacher_version=float(state["teacher_version"]),
                       learning_rate=optimizer.param_groups[0]["lr"], reader_learning_rate=optimizer.param_groups[1]["lr"],
                       elapsed_seconds=time.monotonic() - started)
            interval.append(row)
            if step == 1 or step % args.log_steps == 0 or step == args.max_steps:
                logger.log(step, "train", _mean(interval))
                print(f"[v2][train] step={step} phase={state['phase']} loss={row['loss']:.6f} "
                      f"utility={row['utility_loss']:.6f} write={row['write_loss']:.6f}", flush=True)
                interval = []
            is_best, stop_reason, phase_changed = False, None, False
            due_eval = (step % args.eval_steps == 0 or step == args.max_steps
                        or (not joint and step in (args.bootstrap_steps, args.bootstrap_max_steps)))
            if due_eval:
                metrics = evaluate(step)
                if not joint and not args.writer_only and step >= args.bootstrap_steps:
                    if state["last_gate_pass"]:
                        state.update(phase="joint", phase_start=step)
                        set_trainable_phase(memory, joint=True)
                        phase_changed = True
                        print("[v2] Bootstrap checks passed; joint reader updates start NEXT step", flush=True)
                    elif step >= args.bootstrap_max_steps:
                        stop_reason = "bootstrap_not_ready"
                if joint:
                    if not state["last_gate_pass"]:
                        # Do not silently continue if refreshed targets/policy
                        # collapse; retain the checkpoint for diagnosis only.
                        stop_reason = "joint_validation_not_ready"
                    elif (state["best_joint_action_loss"] is None
                          or metrics["action_loss"] < state["best_joint_action_loss"]):
                        state["best_joint_action_loss"] = metrics["action_loss"]
                        is_best = True
                state["status"] = stop_reason or ("joint" if state["phase"] == "joint" else "bootstrap_only")
                _atomic_json(output / "status.json", {"step": step, **state})
            if step % args.save_steps == 0 or step == args.max_steps or is_best or stop_reason or phase_changed:
                save_checkpoint(output, step, memory, optimizer, config, metadata, best=is_best, keep_last=None)
                last_saved = step
            if step % args.plot_steps == 0 or due_eval:
                logger.plot()
            if stop_reason:
                print(f"[v2] STOPPED SAFELY: {stop_reason}; inspect status.json. "
                      "No readiness threshold was silently relaxed; this is not simulator success.", flush=True)
                return 2
    except KeyboardInterrupt:
        print(f"[v2] Interrupted; last saved completed update={last_saved}. No partial update saved.", flush=True)
        return 130
    print(f"[v2] Finished {args.max_steps} updates; phase={state['phase']}; output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
