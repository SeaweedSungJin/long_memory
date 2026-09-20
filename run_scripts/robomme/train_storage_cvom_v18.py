#!/usr/bin/env python3
"""Train a storage-only V18 CVOM, AFTER choosing a validated Stage-1 reader.

Full-bank decisions compare KEEP with INSERT/replace-oldest using paired future
GT action-flow losses. Reader, short adapter, and adapted AE remain frozen.
Future observations/actions enter counterfactual supervision, never the writer.
Use --preflight-only before spending GPU time. This does not run RoboMME.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
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
from torch.nn import functional as F

from run_scripts.robomme.storage_cvom_v18 import (StorageCVOMV18, WriterConfigV18,
    fifo_insert, file_sha256, label_statistics, make_write_policy, parent_identity,
    require_informative_labels, save_storage_writer_v18, storage_features)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reader-checkpoint", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--storage-contexts", type=int, default=128)
    p.add_argument("--val-storage-contexts", type=int, default=32)
    p.add_argument("--context-refresh-steps", type=int, default=250)
    p.add_argument("--future-samples", type=int, default=2)
    p.add_argument("--future-horizon", type=int, default=16, help="Maximum observed endpoint offset, not raw frames")
    p.add_argument("--noise-samples", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--utility-margin", type=float, default=1e-6)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--plot-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=9182)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--preflight-only", action="store_true")
    return p


def validate_options(args):
    for key in ("max_steps", "storage_contexts", "val_storage_contexts", "context_refresh_steps",
                "future_samples", "future_horizon", "noise_samples", "batch_size", "hidden_dim",
                "eval_steps", "save_steps", "plot_steps"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if args.seed < 0 or args.storage_contexts < 8 or args.val_storage_contexts < 4:
        raise ValueError("Use nonnegative seed, >=8 TRAIN and >=4 VAL full-bank contexts")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(args.utility_margin) or args.utility_margin < 0:
        raise ValueError("utility-margin must be finite and nonnegative")


def build_context_plan(cache, episodes, capacity, args):
    """Fixed disjoint TRAIN/VAL episodes; all selected cases require replacement.

    Round-robin shuffled episodes prevents one long episode from supplying all
    labels. All valid future action decisions are retained before subsampling.
    No action target values influence context selection.
    """
    splits = cache.manifest["splits"]
    if not splits["train"] or not splits["val"] or set(splits["train"]) & set(splits["val"]):
        raise ValueError("Storage TRAIN/VAL must be nonempty and episode-disjoint")
    plan = {}
    for split, requested in (("train", args.storage_contexts), ("val", args.val_storage_contexts)):
        rng = random.Random(args.seed + (0 if split == "train" else 101))
        episode_ids = list(cache.manifest["splits"][split])
        rng.shuffle(episode_ids)
        eligible = {}
        for eid in episode_ids:
            ep = episodes.fetch(eid)
            decisions = ep["decision_mask"].nonzero().flatten().tolist()
            rows = []
            for candidate in range(capacity, len(ep["frames"]) - 1):
                future = [q for q in decisions if candidate < q <= candidate + args.future_horizon]
                if future:
                    rng.shuffle(future)
                    rows.append({"episode_id": int(eid), "candidate": candidate,
                        "future": sorted(future[:args.future_samples])})
            rng.shuffle(rows)
            if rows:
                eligible[int(eid)] = rows
        selected = []
        while len(selected) < requested and any(eligible.values()):
            for eid in episode_ids:
                if eligible.get(int(eid)) and len(selected) < requested:
                    selected.append(eligible[int(eid)].pop())
        if len(selected) < requested:
            raise ValueError(f"Only {len(selected)} eligible {split.upper()} full-bank contexts for "
                f"capacity={capacity}, requested {requested}. Lower --storage-contexts/--val-storage-contexts "
                "or rebuild a longer cache. Changing capacity requires a matched Stage-1 rerun, not an "
                "inference-only budget change.")
        plan[split] = selected
    return plan


def _causal_bank(core, encoded, end, write_policy):
    bank = core.initial_bank()
    for index in range(end):
        bank = write_policy(bank, encoded["stored"][index:index + 1], event_index=index,
                            frame=None, is_demo=None)[0]
    return bank


def _flow_loss(head, ep, q, fused, seed):
    from gr00t.long_memory.cache_reader_v3 import validate_decision
    from gr00t.long_memory.expert_v4 import expert_episode_flow_loss
    validate_decision(ep, q)
    value = expert_episode_flow_loss(head, ep, q, fused, seed=seed)["loss"]
    if not bool(torch.isfinite(value)):
        raise FloatingPointError("Nonfinite frozen-actor counterfactual loss")
    return float(value)


def _label_seed(seed, split, row_index, decision, noise_index):
    payload = json.dumps(["v18-storage-paired-flow", seed, split, row_index, decision, noise_index]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") % (2**31 - 1)


@torch.no_grad()
def label_contexts(core, head, episodes, rows, writer, args, *, split, refresh):
    """On-policy causal starting banks, common FIFO future continuation.

    Using FIFO *after* the single counterfactual write isolates its incremental
    benefit. It does not pretend to optimize an entire future learned policy.
    Both alternatives see identical future observations and noise at each query.
    """
    policy, examples = make_write_policy(writer), []
    for row_index, row in enumerate(rows):
        ep = episodes.fetch(row["episode_id"])
        encoded = core.encode_prefix(ep, max(row["future"]) + 1)
        t, capacity = row["candidate"], core.config.capacity_events
        bank = _causal_bank(core, encoded, t, policy)
        candidate = encoded["stored"][t:t + 1]
        if bank.shape[1] != capacity * core.config.num_short_tokens:
            raise ValueError("Selected storage context is not full; reject misleading no-replacement training")
        feature = storage_features(bank, candidate, capacity).detach().cpu()[0]
        keep, insert = bank, fifo_insert(bank, candidate, capacity)
        keep_losses, insert_losses = [], []
        next_index = t + 1
        for q in sorted(row["future"]):
            for j in range(next_index, q):
                keep = core.write_fifo(keep, encoded["stored"][j:j + 1])
                insert = core.write_fifo(insert, encoded["stored"][j:j + 1])
            next_index = q
            short, query = encoded["short"][q:q + 1], encoded["query"][q:q + 1]
            fused_keep, _ = core.read_from_bank(short, query, keep)
            fused_insert, _ = core.read_from_bank(short, query, insert)
            for noise_index in range(args.noise_samples):
                # Stable across refresh and alternatives, with disjoint VAL seeds.
                seed = _label_seed(args.seed, split, row_index, q, noise_index)
                keep_losses.append(_flow_loss(head, ep, q, fused_keep, seed))
                insert_losses.append(_flow_loss(head, ep, q, fused_insert, seed))
        keep_loss = sum(keep_losses) / len(keep_losses)
        insert_loss = sum(insert_losses) / len(insert_losses)
        examples.append({**row, "features": feature, "utility": keep_loss - insert_loss,
            "keep_loss": keep_loss, "insert_loss": insert_loss, "full_bank": True})
        if (row_index + 1) % 8 == 0:
            print(f"[cvom-label] {split} refresh={refresh} {row_index + 1}/{len(rows)} full-bank contexts", flush=True)
    return examples


def examples_to_tensors(examples, device):
    return (torch.stack([row["features"] for row in examples]).to(device),
        torch.tensor([row["utility"] for row in examples], device=device, dtype=torch.float32))


@torch.no_grad()
def writer_validation(writer, examples, scale):
    x, utility = examples_to_tensors(examples, next(writer.parameters()).device)
    score = writer.forward_features(x)
    insert = score >= writer.config.threshold
    keep_loss = torch.tensor([row["keep_loss"] for row in examples], device=x.device)
    fifo_loss = torch.tensor([row["insert_loss"] for row in examples], device=x.device)
    selected_loss = torch.where(insert, fifo_loss, keep_loss)
    return {"utility_loss": float(F.smooth_l1_loss(score, (utility / scale).clamp(-10, 10))),
        "write_accuracy": float((insert == (utility >= 0)).float().mean()),
        "write_rate": float(insert.float().mean()), "positive_label_rate": float((utility > 0).float().mean()),
        "full_bank_coverage": 1.0, "counterfactual_selected_action_loss": float(selected_loss.mean()),
        "counterfactual_fifo_action_loss": float(fifo_loss.mean()),
        "counterfactual_gain_vs_fifo": float((fifo_loss - selected_loss).mean())}


@torch.no_grad()
def policy_validation(core, head, episodes, rows, writer, args):
    """Actual causal CVOM-prefix vs FIFO-prefix offline action loss, same actor."""
    losses = {"writer": [], "fifo": []}
    for index, row in enumerate(rows):
        ep, q = episodes.fetch(row["episode_id"]), row["future"][-1]
        encoded = core.encode_prefix(ep, q + 1)
        writer_bank = _causal_bank(core, encoded, q, make_write_policy(writer))
        fifo_bank = core.initial_bank()
        for j in range(q):
            fifo_bank = core.write_fifo(fifo_bank, encoded["stored"][j:j + 1])
        for name, bank in (("writer", writer_bank), ("fifo", fifo_bank)):
            fused, _ = core.read_from_bank(encoded["short"][q:q + 1], encoded["query"][q:q + 1], bank)
            losses[name].append(_flow_loss(head, ep, q, fused, args.seed + 20_000_000 + index))
    writer_loss, fifo_loss = (sum(losses[name]) / len(rows) for name in ("writer", "fifo"))
    return {"action_loss": writer_loss, "fifo_action_loss": fifo_loss,
        "memory_gain": fifo_loss - writer_loss}


def main(argv=None):
    args = parser().parse_args(argv)
    validate_options(args)
    from gr00t.long_memory.cache import EpisodeCache
    from gr00t.long_memory.cache_reader_v3 import MappedEpisodes
    from gr00t.long_memory.monitoring import RunLogger, _atomic_json
    from gr00t.long_memory.hamlet import validate_cache_checkpoint
    from gr00t.long_memory.safety_v5 import validate_output_scope
    from run_scripts.robomme.checkpoint_representation_v18 import checkpoint_info_v18, load_checkpoint_v18
    from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18, RepresentationMemoryV18
    reader = Path(args.reader_checkpoint).resolve(strict=True)
    preliminary = json.loads((reader / "checkpoint.json").read_text())
    base_model = preliminary["metadata"]["base_model"]["path"]
    info = checkpoint_info_v18(base_model, reader)
    if info["step"] <= 0:
        raise ValueError("CVOM requires a trained actor, not zero-step initialization")
    cfg = RepresentationConfigV18(**info["config"]["representation"])
    cache, output = EpisodeCache(args.cache_dir), Path(args.output_dir).resolve()
    validate_cache_checkpoint(cache.manifest)
    validate_output_scope(output, reader, cache.path, cache.manifest.get("dataset_path"), base_model)
    if output.exists():
        raise ValueError("Use a NEW writer output directory; existing experiments are never overwritten")
    if output == reader or output.is_relative_to(reader) or reader.is_relative_to(output):
        raise ValueError("Writer output must be separate from its immutable Stage-1 checkpoint")
    if cache.manifest["fingerprint"] != info["metadata"]["cache_fingerprint"]:
        raise ValueError("Writer cache differs from Stage-1 cache/episode split")
    episodes = MappedEpisodes(cache, max_cached=2)
    plan = build_context_plan(cache, episodes, cfg.capacity_events, args)
    print(f"[cvom-preflight] parent={reader}; capacity={cfg.capacity_events}; "
          f"TRAIN={len(plan['train'])}, VAL={len(plan['val'])} full-bank contexts", flush=True)
    if args.preflight_only:
        print("[cvom-preflight] No model loaded, labels generated, output written, or training started.")
        return
    from gr00t.long_memory.train_v3 import _Tee
    output.mkdir(parents=True, exist_ok=False)
    with (output / "training.log").open("x", encoding="utf-8") as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            _atomic_json(output / "driver_status.json", {"status": "running", "reader_checkpoint": str(reader)})
            try:
                _run_training(args, reader, base_model, info, cfg, cache, output, episodes, plan)
            except BaseException as error:
                _atomic_json(output / "failure.json", {"type": type(error).__name__,
                    "message": str(error), "traceback": traceback.format_exc(),
                    "interrupted": isinstance(error, KeyboardInterrupt)})
                _atomic_json(output / "driver_status.json", {"status": "failed", "message": str(error)})
                traceback.print_exc()
                raise
            _atomic_json(output / "driver_status.json", {"status": "complete", "max_steps": args.max_steps})


def _run_training(args, reader, base_model, info, cfg, cache, output, episodes, plan):
    from gr00t.long_memory.monitoring import RunLogger, _atomic_json
    from run_scripts.robomme.checkpoint_representation_v18 import load_checkpoint_v18
    from run_scripts.robomme.representation_core_v18 import RepresentationMemoryV18
    from gr00t.long_memory.hamlet import load_frozen_hamlet
    from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora, set_expert_trainable
    torch.manual_seed(args.seed)
    model, _ = load_frozen_hamlet(base_model, args.device)
    head = model.action_head
    install_expert_lora(head, LoRAConfig(**info["config"]["expert"]), info["config"]["expert_targets"])
    core = RepresentationMemoryV18(cfg, head.memory_transformer
        if cfg.representation == "adapted_short" else None).to(args.device)
    load_checkpoint_v18(reader, core, head)
    core.eval().requires_grad_(False)
    set_expert_trainable(head, False)
    if any(p.requires_grad for p in core.parameters()) or any(p.requires_grad for p in head.parameters()):
        raise ValueError("Storage training must freeze the entire actor, short adapter and AE")
    writer = StorageCVOMV18(WriterConfigV18(memory_dim=cfg.hidden_dim,
        capacity_events=cfg.capacity_events, hidden_dim=args.hidden_dim)).to(args.device)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=args.learning_rate, weight_decay=.01)
    _atomic_json(output / "context_plan.json", plan)
    metadata = {"parent": parent_identity(reader), "cache_fingerprint": cache.manifest["fingerprint"],
        "splits": cache.manifest["splits"], "train": vars(args),
        "context_plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in
            (Path(__file__).resolve(), ROOT / "run_scripts/robomme/storage_cvom_v18.py",
             ROOT / "run_scripts/robomme/representation_core_v18.py")},
        "actor_frozen": True, "label": "future_keep_loss_minus_insert_fifo_oldest_loss",
        "label_future_continuation": "common_fifo", "context_prefix": "current_writer_causal",
        "selection": "fixed_final_step_not_test_selected", "metric_limitation": "offline flow loss, NOT robot success"}
    _atomic_json(output / "run_config.json", metadata)
    logger, start = RunLogger(output), time.monotonic()
    # Fixed validation contexts are FIFO contexts, never refreshed using VAL labels.
    val_examples = label_contexts(core, head, episodes, plan["val"], writer, args, split="val", refresh=0)
    train_examples, utility_scale = None, None
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    for step in range(1, args.max_steps + 1):
        if train_examples is None or (step - 1) % args.context_refresh_steps == 0:
            train_examples = label_contexts(core, head, episodes, plan["train"], writer, args,
                                           split="train", refresh=step - 1)
            values = [row["utility"] for row in train_examples]
            audit = {"step": step - 1, "train": label_statistics(values, margin=args.utility_margin),
                "val": label_statistics([row["utility"] for row in val_examples], margin=args.utility_margin),
                "full_bank_coverage": 1.0, "capacity_events": cfg.capacity_events,
                "train_candidate_max_bank_cosine_mean": sum(float(row["features"][-4]) for row in train_examples) / len(train_examples),
                "train_candidate_max_bank_cosine_min": min(float(row["features"][-4]) for row in train_examples),
                "train_candidate_max_bank_cosine_max": max(float(row["features"][-4]) for row in train_examples)}
            _atomic_json(output / "label_audit.json", audit)
            _atomic_json(output / f"label_audit_{step - 1:06d}.json", audit)
            # Weak/constant action sensitivity is a diagnosis, not a trained writer.
            require_informative_labels(values, margin=args.utility_margin)
            x_train, y_train = examples_to_tensors(train_examples, args.device)
            if utility_scale is None:
                utility_scale = max(float(y_train.abs().median()), args.utility_margin, 1e-6)
                metadata["utility_scale"] = utility_scale
                _atomic_json(output / "run_config.json", metadata)
        indices = torch.randint(len(train_examples), (args.batch_size,), generator=generator).to(args.device)
        prediction = writer.forward_features(x_train[indices])
        target = (y_train[indices] / utility_scale).clamp(-10, 10)
        utility_loss = F.smooth_l1_loss(prediction, target)
        confident = y_train[indices].abs() > args.utility_margin
        write_loss = (F.binary_cross_entropy_with_logits(prediction[confident],
            (y_train[indices][confident] > 0).float()) if bool(confident.any()) else prediction.sum() * 0)
        loss = utility_loss + .25 * write_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 10 == 0:
            metrics = {"loss": float(loss), "utility_loss": float(utility_loss), "write_loss": float(write_loss),
                "grad_norm": float(grad_norm), "learning_rate": args.learning_rate,
                "elapsed_seconds": time.monotonic() - start}
            logger.log(step, "train", metrics)
            print(f"[cvom] step={step} loss={float(loss):.6f} utility={float(utility_loss):.6f}", flush=True)
        if step % args.eval_steps == 0 or step == args.max_steps:
            metrics = writer_validation(writer, val_examples, utility_scale)
            metrics.update(policy_validation(core, head, episodes, plan["val"], writer, args))
            logger.log(step, "val", metrics)
            print(f"[cvom-val] step={step} offline_gain_vs_fifo={metrics['memory_gain']:.8f} "
                  f"write_rate={metrics['write_rate']:.3f} (NOT task success)", flush=True)
        if step % args.save_steps == 0 or step == args.max_steps:
            save_storage_writer_v18(output / f"checkpoint-{step:06d}", writer, reader,
                                   step=step, metadata=metadata)
        if step % args.plot_steps == 0 or step == args.max_steps:
            logger.plot()
    _atomic_json(output / "final_checkpoint.json", {"path": f"checkpoint-{args.max_steps:06d}",
        "step": args.max_steps, "selection": "fixed_final_step"})
    print(f"[cvom] Finished; evaluate same-parent FIFO versus CVOM: {output}", flush=True)


if __name__ == "__main__":
    main()
