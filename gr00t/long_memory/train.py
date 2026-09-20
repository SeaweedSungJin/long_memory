"""Two-stage cached HAMLET training; single GPU, small reproducible baseline.

Stage 1: all-write/FIFO, action-supervised event/read/fusion warm-up.
Stage 2: hard-write/FIFO plus continued action training and detached CVoM
utility/write supervision. No exact gradient through hard bank selection.
The original HAMLET checkpoint is immutable and is never saved in our outputs.
"""
import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .cache import EpisodeCache
from .core import EpisodicMemory, MemoryConfig
from .cvom import CVoMConfig, CVoMLabels, candidate_indices
from .hamlet import checkpoint_identity, episode_flow_loss, load_frozen_hamlet, validate_cache_checkpoint
from .monitoring import RunLogger, load_checkpoint, save_checkpoint
from .replay import candidate_predictions, encode_until, event_inputs, read_bank, replay_bank


def _checkpoint_manifest(path):
    return json.loads((Path(path) / "checkpoint.json").read_text())


def _mean(records):
    keys = set().union(*(x.keys() for x in records))
    return {k: float(np.mean([r[k] for r in records if k in r and r[k] is not None]))
            for k in keys if any(k in r and r[k] is not None for r in records)}


def eligible_decisions(episode, for_training=False):
    decisions = torch.where(episode["decision_mask"])[0].tolist()
    if for_training:
        # An empty bank is exact identity and cannot train memory parameters.
        decisions = [d for d in decisions if bool(episode["transition_valid"][:d].any())]
    return decisions


@torch.no_grad()
def validate(memory, head, fetch, plan, policy, seed, labels=None):
    memory.eval()
    metrics, predictions, truths, utilities, targets = [], [], [], [], []
    for index, (eid, decision, wrong_eid) in enumerate(plan):
        ep = fetch(eid)
        bank, stats = replay_bank(memory, ep, decision, policy)
        read = read_bank(memory, ep, decision, bank)
        paired_seed = seed + index * 1009
        output = episode_flow_loss(head, ep, decision, read["fused_short"], seed=paired_seed)
        baseline = episode_flow_loss(head, ep, decision, seed=paired_seed)
        # Wrong CONTENT control, not a permutation of attention slots (which
        # would be mathematically invariant). Only a different VAL episode is used.
        wrong_loss = None
        if wrong_eid is not None:
            other = fetch(wrong_eid)
            other_d = min(decision, len(other["actions"]))
            wrong_bank, _ = replay_bank(memory, other, other_d, policy)
            encoded = encode_until(memory, other, other_d)
            device = next(memory.parameters()).device
            idx = torch.tensor(wrong_bank, device=device, dtype=torch.long)
            wrong = memory.read(ep["short"][decision].to(device).float()[None],
                                ep["state"][decision].to(device).float()[None],
                                encoded["keys"][idx][None], encoded["values"][idx][None],
                                torch.ones((1, len(idx)), device=device, dtype=torch.bool))
            wrong_loss = episode_flow_loss(head, ep, decision, wrong["fused_short"], seed=paired_seed)["loss"].item()
        row = {"action_loss": output["loss"].item(), "velocity_mae": output["velocity_mae"].item(),
               "baseline_action_loss": baseline["loss"].item(),
               "memory_gain": baseline["loss"].item() - output["loss"].item(), **stats}
        row.update({k: float(read[k].mean()) for k in ("gate_mean", "read_norm", "residual_norm", "null_weight")})
        if wrong_loss is not None:
            row["wrong_memory_action_loss"] = wrong_loss
        if labels is not None:
            candidates = candidate_indices(ep)
            if candidates:
                candidate = candidates[index % len(candidates)]
                label = labels.get(ep, candidate)
                utility, logit = candidate_predictions(memory, ep, candidate, policy)
                target = utility.new_tensor([label["utility_target"]])
                truth = logit.new_tensor([label["write_target"]])
                row["utility_loss"] = F.smooth_l1_loss(utility, target).item()
                row["write_loss"] = F.binary_cross_entropy_with_logits(logit, truth).item()
                predictions.append(bool(logit.sigmoid().item() >= memory.config.write_threshold))
                truths.append(bool(label["write_target"]))
                utilities.append(utility.item())
                targets.append(label["utility_target"])
        metrics.append(row)
    result = _mean(metrics)
    if truths:
        pred, truth = np.asarray(predictions), np.asarray(truths)
        tp = int((pred & truth).sum())
        result.update(write_accuracy=float((pred == truth).mean()),
                      write_precision=tp / int(pred.sum()) if pred.any() else None,
                      write_recall=tp / int(truth.sum()) if truth.any() else None,
                      positive_label_rate=float(truth.mean()),
                      utility_corr=float(np.corrcoef(utilities, targets)[0, 1])
                      if len(truths) > 1 and np.std(utilities) > 1e-12 and np.std(targets) > 1e-12 else None)
    memory.train()
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--base-model", help="Defaults to the frozen checkpoint recorded in the cache")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--init-checkpoint", help="Stage 2: Stage-1 memory checkpoint, not a base HAMLET")
    p.add_argument("--resume", help="Same-stage memory checkpoint including optimizer/RNG")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--val-samples", type=int, default=8)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--plot-steps", type=int, default=25)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capacity", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--key-dim", type=int, default=64)
    p.add_argument("--value-dim", type=int, default=128)
    p.add_argument("--write-threshold", type=float, default=0.5)
    p.add_argument("--min-fill", type=int, default=8)
    p.add_argument("--reconstruction-weight", type=float, default=0.0)
    p.add_argument("--utility-weight", type=float, default=0.1)
    p.add_argument("--write-weight", type=float, default=0.1)
    p.add_argument("--utility-every", type=int, default=4, help="Expensive paired teacher labels only on every Nth update (and first)")
    p.add_argument("--utility-scale", type=float, default=0.01, help="Fixed initial scale: calibrate from signed_gain distribution, never batch min/max")
    p.add_argument("--write-delta", type=float, default=0.0)
    p.add_argument("--future-samples", type=int, default=2)
    p.add_argument("--coalitions", type=int, default=1)
    p.add_argument("--activation-checkpointing", action="store_true", help="Recompute frozen expert forward during memory-input backward")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("max_steps", "grad_accum", "eval_steps", "val_samples", "log_steps", "plot_steps", "save_steps", "utility_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.min_fill < 1:
        raise ValueError("This training recipe needs min_fill>=1 to avoid an untrainable empty hard bank")
    if args.stage == 2 and not args.init_checkpoint and not args.resume:
        raise ValueError("Stage 2 requires --init-checkpoint from Stage 1")
    if args.stage == 1 and args.init_checkpoint:
        raise ValueError("Use --resume for Stage 1; --init-checkpoint starts Stage 2")
    for name in ("reconstruction_weight", "utility_weight", "write_weight"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cache = EpisodeCache(args.cache_dir)
    manifest = cache.manifest
    base_path = str(Path(args.base_model or manifest["model_path"]).resolve())
    if Path(base_path) != Path(manifest["model_path"]).resolve():
        raise ValueError("Base checkpoint differs from cache provenance; regenerate cache for a different model")
    validate_cache_checkpoint(manifest)
    base_identity = checkpoint_identity(base_path)
    @lru_cache(maxsize=4)
    def fetch(eid):
        return cache.load(int(eid))

    train_ids, val_ids = manifest["splits"]["train"], manifest["splits"]["val"]
    if not train_ids or not val_ids or set(train_ids) & set(val_ids):
        raise ValueError("Require disjoint, nonempty episode-level train/validation splits")
    train_decisions = {eid: eligible_decisions(fetch(eid), True) for eid in train_ids}
    train_ids = [eid for eid in train_ids if train_decisions[eid]]
    if not train_ids:
        raise ValueError("No action decision has completed past memory in the training split")
    # Sampling is episode-balanced, not dominated by long episodes. Split task
    # grouping and task-balance limitations are documented by cache provenance.
    val_rng = random.Random(args.seed + 9001)
    pool = [(eid, d) for eid in val_ids for d in eligible_decisions(fetch(eid))]
    if not pool:
        raise ValueError("No valid validation action decisions")
    plan = []
    for eid, decision in val_rng.sample(pool, min(args.val_samples, len(pool))):
        others = [x for x in val_ids if x != eid]
        plan.append((eid, decision, val_rng.choice(others) if others else None))

    init_info = _checkpoint_manifest(args.resume or args.init_checkpoint) if (args.resume or args.init_checkpoint) else None
    if init_info:
        saved_stage = init_info["config"]["stage"]
        if args.resume and saved_stage != args.stage:
            raise ValueError("--resume cannot change stages")
        if args.init_checkpoint and not args.resume and saved_stage != 1:
            raise ValueError("Stage-2 initialization must be a Stage-1 checkpoint")
        if init_info["metadata"]["cache_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Initialization/resume cache differs; preserve the exact episode split/cache")
        memory_config = MemoryConfig(**init_info["config"]["memory"])
        # Architecture and bank budget are inherited exactly. Silent command-line
        # overrides would invalidate comparisons/teacher targets.
        for name in ("capacity", "hidden_dim", "key_dim", "value_dim", "min_fill", "write_threshold"):
            if getattr(args, name) != getattr(memory_config, name):
                raise ValueError(f"--{name.replace('_','-')} must match the initial/resume checkpoint ({getattr(memory_config,name)})")
    else:
        memory_config = MemoryConfig(feature_dim=manifest["feature_dim"], state_dim=manifest["state_dim"],
                                     action_dim=manifest["action_dim"], hidden_dim=args.hidden_dim,
                                     key_dim=args.key_dim, value_dim=args.value_dim, capacity=args.capacity,
                                     min_fill=args.min_fill, write_threshold=args.write_threshold)
    for name in ("feature_dim", "state_dim", "action_dim"):
        if getattr(memory_config, name) != manifest[name]:
            raise ValueError(f"Memory/cache dimension mismatch: {name}")
    memory = EpisodicMemory(memory_config).to(args.device)
    memory.utility_head.requires_grad_(args.stage == 2)
    memory.write_head.requires_grad_(args.stage == 2)
    memory.reconstruction.requires_grad_(args.reconstruction_weight > 0)
    optimizer = torch.optim.AdamW([p for p in memory.parameters() if p.requires_grad],
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    start = 0
    if args.resume:
        # Restore RNG AFTER all model loading below, which itself initializes
        # temporary weights and can consume RNG.
        load_checkpoint(args.resume, memory)
        start = init_info["step"]
        if args.max_steps <= start:
            raise ValueError(f"--max-steps is the total update target and must exceed resumed step {start}")
    elif args.init_checkpoint:
        load_checkpoint(args.init_checkpoint, memory)
    config = {"stage": args.stage, "memory": asdict(memory_config), "train": vars(args).copy()}
    config["train"]["base_model"] = base_path
    if args.resume:
        old = init_info["config"]["train"]
        allowed = {"max_steps", "resume", "output_dir", "base_model", "init_checkpoint"}
        for name, value in config["train"].items():
            if name not in allowed and old.get(name) != value:
                raise ValueError(f"Exact resume option changed: {name} ({old.get(name)} -> {value})")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output.iterdir()):
        raise ValueError("Use a new output directory, or --resume with the saved run")
    if args.resume and Path(args.resume).resolve().parent != output and any(output.iterdir()):
        raise ValueError("A resume fork requires a new, empty output directory; preserve existing runs")
    metadata = {"base_model": base_identity, "cache_fingerprint": manifest["fingerprint"],
                "cache_dir": str(Path(args.cache_dir).resolve()), "validation_plan": plan,
                "note": "Offline velocity metrics are not robot success rates; base may have seen these episodes."}
    if init_info and init_info["metadata"]["base_model"] != base_identity:
        raise ValueError("Base checkpoint files changed since the previous stage/checkpoint")
    if args.resume:
        metadata["resumed_from"] = str(Path(args.resume).resolve())
    teacher_path = None
    if args.stage == 2:
        teacher_path = args.init_checkpoint or init_info["metadata"]["teacher_checkpoint"]
        metadata["teacher_checkpoint"] = str(Path(teacher_path).resolve())
        metadata["teacher_sha256"] = hashlib.sha256((Path(teacher_path) / "model.safetensors").read_bytes()).hexdigest()
        teacher_info = _checkpoint_manifest(teacher_path)
        if (teacher_info["config"]["stage"] != 1
                or teacher_info["config"]["memory"] != asdict(memory_config)
                or teacher_info["metadata"]["cache_fingerprint"] != manifest["fingerprint"]
                or teacher_info["metadata"]["base_model"] != base_identity):
            raise ValueError("Stage-2 teacher must be a compatible Stage-1 checkpoint")
        if args.resume and (metadata["teacher_sha256"] != init_info["metadata"]["teacher_sha256"]
                            or metadata["teacher_checkpoint"] != init_info["metadata"]["teacher_checkpoint"]):
            raise ValueError("Stage-2 teacher checkpoint changed since the saved run")
    logger = RunLogger(output)
    if args.resume and any(record["step"] > start for record in logger.records):
        raise ValueError("Logs are newer than resume checkpoint; use the latest checkpoint or a NEW --output-dir to fork safely")
    print(f"[long-memory] stage={args.stage} trainable={sum(p.numel() for p in memory.parameters() if p.requires_grad):,}; base frozen", flush=True)
    base, _ = load_frozen_hamlet(base_path, args.device)
    head = base.action_head
    del base  # No VLM on the training GPU: cache already contains its outputs.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    labels = None
    if args.stage == 2:
        teacher = EpisodicMemory(memory_config).to(args.device)
        load_checkpoint(teacher_path, teacher)
        label_config = CVoMConfig(future_samples=args.future_samples, coalitions=args.coalitions,
                                  far_delay=max(1, int(json.loads((Path(base_path)/"config.json").read_text())["memory_window"])),
                                  utility_scale=args.utility_scale, write_delta=args.write_delta, seed=args.seed + 77)
        labels = CVoMLabels(teacher, head, label_config, output / "cvom_labels",
                            {"teacher_sha256": metadata["teacher_sha256"], "cache": manifest["fingerprint"]})
    # Do not overwrite provenance/config until all resume/teacher checks pass.
    (output / "run_config.json").write_text(json.dumps(config, indent=2))
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2))
    if args.resume:
        load_checkpoint(args.resume, memory, optimizer)
    else:
        # Initialization and teacher construction should not change the intended
        # data/noise sequence of the training experiment.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    policy = "all" if args.stage == 1 else "hard"
    previous_best = init_info["metadata"].get("best_validation_action_loss", float("inf")) if args.resume else float("inf")
    best = min(previous_best, min((r["action_loss"] for r in logger.records if r["split"] == "val" and "action_loss" in r), default=float("inf")))
    if not args.resume:
        initial = validate(memory, head, fetch, plan, policy, args.seed + 50000, labels)
        logger.log(0, "val", initial)
        logger.plot()
        best = initial["action_loss"]
        metadata["best_validation_action_loss"] = best
        # Keep the unmodified reader if subsequent updates only hurt validation.
        save_checkpoint(output, 0, memory, optimizer, config, metadata, best=True, keep_last=None)
    t0 = time.monotonic()
    interval, last_saved, step = [], start, start
    memory.train()
    try:
        for step in range(start + 1, args.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            learning_rate = args.learning_rate * min(1.0, step / max(1, args.warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            batch_rows = []
            for _micro in range(args.grad_accum):
                eid = random.choice(train_ids)
                ep = fetch(eid)
                decision = random.choice(train_decisions[eid])
                bank, stats = replay_bank(memory, ep, decision, policy)
                encoded = encode_until(memory, ep, decision)
                read = read_bank(memory, ep, decision, bank, encoded)
                action = episode_flow_loss(head, ep, decision, read["fused_short"],
                                           activation_checkpointing=args.activation_checkpointing)
                total = action["loss"]
                row = {"action_loss": total.item(), "velocity_mae": action["velocity_mae"].item(), **stats}
                row.update({k: float(read[k].detach().mean()) for k in ("gate_mean", "read_norm", "residual_norm", "null_weight")})
                if args.reconstruction_weight:
                    valid = encoded["valid"]
                    raw = event_inputs(ep, args.device, decision)
                    reconstructed = memory.reconstruct(encoded["event"][valid])
                    dm = (raw["post_moment"] - raw["pre_moment"]).mean(1)[valid]
                    ds = (raw["next_state"] - raw["state"])[valid]
                    rec = F.mse_loss(reconstructed["delta_moment"], dm) + F.mse_loss(reconstructed["delta_state"], ds)
                    total = total + args.reconstruction_weight * rec
                    row["reconstruction_loss"] = rec.item()
                if labels is not None and (step == 1 or step % args.utility_every == 0):
                    candidates = candidate_indices(ep)
                    if candidates:
                        candidate = random.choice(candidates)
                        label = labels.get(ep, candidate)
                        utility, logit = candidate_predictions(memory, ep, candidate, policy)
                        u_loss = F.smooth_l1_loss(utility, utility.new_tensor([label["utility_target"]]))
                        w_loss = F.binary_cross_entropy_with_logits(logit, logit.new_tensor([label["write_target"]]))
                        total = total + args.utility_weight * u_loss + args.write_weight * w_loss
                        row.update(utility_loss=u_loss.item(), write_loss=w_loss.item(),
                                   write_accuracy=float((logit.detach().sigmoid().item() >= memory.config.write_threshold) == bool(label["write_target"])),
                                   positive_label_rate=float(label["write_target"]), signed_gain=label["signed_gain"])
                if not torch.isfinite(total) or not total.requires_grad:
                    raise FloatingPointError("Loss is nonfinite or disconnected from trainable memory")
                row["loss"] = total.item()
                (total / args.grad_accum).backward()
                batch_rows.append(row)
            grad_norm = torch.nn.utils.clip_grad_norm_(memory.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            row = _mean(batch_rows)
            row.update(grad_norm=float(grad_norm), learning_rate=learning_rate, elapsed_seconds=time.monotonic() - t0)
            interval.append(row)
            if step % args.log_steps == 0 or step == 1 or step == args.max_steps:
                summary = _mean(interval)
                logger.log(step, "train", summary)
                print(f"[stage{args.stage}] step={step} action={summary['action_loss']:.6f} grad={summary['grad_norm']:.4f} write_rate={summary['write_rate']:.3f}", flush=True)
                interval = []
            is_best = False
            if step % args.eval_steps == 0 or step == args.max_steps:
                result = validate(memory, head, fetch, plan, policy, args.seed + 50000, labels)
                logger.log(step, "val", result)
                is_best = result["action_loss"] < best
                best = min(best, result["action_loss"])
                metadata["best_validation_action_loss"] = best
                print(f"[val] step={step} action={result['action_loss']:.6f} baseline={result['baseline_action_loss']:.6f} gain={result['memory_gain']:+.6f}", flush=True)
            if step % args.save_steps == 0 or step == args.max_steps or is_best:
                save_checkpoint(output, step, memory, optimizer, config, metadata, best=is_best, keep_last=None)
                last_saved = step
            if step % args.plot_steps == 0 or step % args.eval_steps == 0 or step == args.max_steps:
                logger.plot()
    except KeyboardInterrupt:
        # Only completed optimizer updates are safe resume points. A partial
        # gradient-accumulation update is discarded; retain previous checkpoints.
        print(f"[long-memory] interrupted; last saved completed update={last_saved}. Existing checkpoints preserved.", flush=True)
        logger.plot()
        return
    print(f"[long-memory] complete: {output}; curves.png and metrics.jsonl contain offline validation", flush=True)


if __name__ == "__main__":
    main()
