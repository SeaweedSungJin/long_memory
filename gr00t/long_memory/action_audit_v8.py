"""Paired offline V8 generated-action and fixed-flow-time diagnostics.

This is NOT RoboMME task success. Unlike teacher-conditioned flow loss, action
generation below receives no GT trajectory or target mask. Targets are used
only after generation for normalized-coordinate error measurement.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random

import torch
from transformers.feature_extraction_utils import BatchFeature

from .cache import EpisodeCache
from .cache_reader_v3 import MappedEpisodes, validate_decision
from .checkpoint_v8 import load_checkpoint_v8, v8_checkpoint_info
from .event_v8 import MemoryV8Config, EventMemoryV8
from .expert_v4 import (LoRAConfig, adapter_disabled, expert_flow_loss,
                       install_expert_lora, set_expert_trainable)
from .hamlet import isolated_seed, load_frozen_hamlet, validate_cache_checkpoint
from .monitoring import _atomic_json
from .replay_v8 import replay_queries
from .safety_v5 import validate_output_scope


def cached_inputs(head, episode, decision, fused_short=None):
    """Observation-only conditioning; deliberately never reads targets/masks."""
    ref = next(head.parameters())
    feature = episode["features"][decision].to(device=ref.device, dtype=ref.dtype)[None]
    if fused_short is not None:
        q = fused_short.shape[1]
        feature = torch.cat((feature[:, :-q], fused_short.to(feature)), dim=1)
    state = episode["state"][decision].to(ref).reshape(1, 1, -1)
    emb = torch.tensor([int(episode["embodiment_id"])], device=ref.device, dtype=torch.long)
    masks = BatchFeature({
        "backbone_attention_mask": episode["attention_masks"][decision].to(device=ref.device)[None],
        "image_mask": episode["image_masks"][decision].to(device=ref.device)[None],
    })
    return feature, state, emb, masks


@torch.no_grad()
def generated_action(head, episode, decision, fused_short=None, *, seed):
    feature, state, emb, masks = cached_inputs(head, episode, decision, fused_short)
    previous = getattr(head, "_inference_gen", None)
    try:
        # Both branches of the original generator's optional environment seed
        # use matching noise: explicit generator if enabled, isolated RNG if not.
        head._inference_gen = torch.Generator(device=feature.device).manual_seed(int(seed))
        with isolated_seed(seed, feature.device):
            state_features = head.state_encoder(state, emb)
            prediction = head.get_action_with_features(feature, state_features, emb, masks)["action_pred"]
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("Nonfinite generated normalized action")
        return prediction
    finally:
        head._inference_gen = previous


def action_errors(prediction, target, mask, steps=None):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction/target/mask shapes differ")
    valid = mask.bool()
    if steps is not None:
        valid = valid.clone()
        valid[:, steps:] = False
    if not bool(valid.any()):
        raise ValueError("No valid target elements in generated-action diagnostic")
    difference = (prediction.float() - target.float())[valid]
    if not bool(torch.isfinite(difference).all()):
        raise FloatingPointError("Nonfinite valid generated-action error")
    return {"mse": float(difference.square().mean()), "mae": float(difference.abs().mean())}


@torch.no_grad()
def fixed_time_loss(head, episode, decision, fused_short, *, time, seed):
    if not 0 <= time < 1:
        raise ValueError("Flow time must be in [0,1); t=0 is pure noise")
    feature, state, emb, masks = cached_inputs(head, episode, decision, fused_short)
    target = episode["targets"][decision].to(feature)[None]
    mask = episode["target_mask"][decision].to(device=feature.device)[None]
    with isolated_seed(seed, feature.device):
        noise = torch.randn_like(target)
    result = expert_flow_loss(head, feature, state, target, mask,
        masks.backbone_attention_mask, masks.image_mask, emb, noise=noise,
        time=torch.full((1, 1, 1), float(time), device=feature.device, dtype=feature.dtype))
    return float(result["loss"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--noise-samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=8401)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--times", type=float, nargs="+", default=[0., .25, .75])
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)
    if min(args.samples, args.noise_samples) < 1 or args.seed < 0 or any(not 0 <= t < 1 for t in args.times):
        raise ValueError("Invalid audit sample count/seed/time")
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    base = cache.manifest["model_path"]
    info = v8_checkpoint_info(base, args.checkpoint)
    if info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]:
        raise ValueError("Checkpoint and audit cache differ")
    output = Path(args.output_dir).resolve()
    validate_output_scope(output, cache.path, cache.manifest.get("dataset_path"), base, args.checkpoint)
    if output.exists():
        raise FileExistsError("Use a new audit output; previous evidence is preserved")
    episodes = MappedEpisodes(cache)
    # One decision per episode before any episode is reused: bounds dependence
    # for this small diagnostic without calling the result a significance test.
    ids = list(cache.manifest["splits"]["val"])
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    queries = []
    for eid in ids:
        ep = episodes.fetch(eid)
        valid = torch.where(ep["decision_mask"])[0].tolist()
        if valid:
            queries.append([eid, rng.choice(valid)])
        if len(queries) >= args.samples:
            break
    if not queries:
        raise ValueError("No held-out queries")
    if args.preflight_only:
        print(f"[action-audit] {len(queries)} held-out queries; no model/output created")
        return 0
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "plan.json", {"args": vars(args), "queries": queries,
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_metadata": info["metadata"],
        "note": "Offline normalized action error, NOT simulator success; no targets enter generation"})
    model, processor = load_frozen_hamlet(base, args.device)
    head = model.action_head
    install_expert_lora(head, LoRAConfig(**info["config"]["expert"]), targets=info["config"]["expert_targets"])
    memory = EventMemoryV8(MemoryV8Config(**info["config"]["memory"]))
    load_checkpoint_v8(args.checkpoint, memory, head)
    set_expert_trainable(head, False)
    memory.to(device=args.device, dtype=torch.float32).eval().requires_grad_(False)
    head.eval().requires_grad_(False)
    del model, processor
    rows = []
    for index, (eid, decision) in enumerate(queries):
        ep = episodes.fetch(eid)
        # This validates the DATASET row outside the observation-only policy
        # helper: stale conditioning tails and invalid target masks must fail.
        validate_decision(ep, decision)
        with torch.no_grad():
            fused, _ = replay_queries(memory, ep, [decision], mode=info["config"]["mode"])[decision]
        for repeat in range(args.noise_samples):
            seed = args.seed + index * 100003 + repeat
            for role in ("baseline", "memory-off", "reader"):
                context = adapter_disabled(head) if role == "baseline" else nullcontext()
                current = fused if role == "reader" else None
                with context:
                    prediction = generated_action(head, ep, decision, current, seed=seed)
                    target = ep["targets"][decision].to(device=prediction.device, dtype=torch.float32)[None]
                    mask = ep["target_mask"][decision].to(device=prediction.device)[None]
                    metrics = {"generated_" + k: v for k, v in action_errors(prediction, target, mask).items()}
                    metrics.update({"executed_prefix_" + k: v for k, v in action_errors(
                        prediction, target, mask, int(cache.manifest["action_steps"])).items()})
                    for time in args.times:
                        metrics[f"flow_t{time:g}"] = fixed_time_loss(head, ep, decision, current, time=time, seed=seed)
                rows.append(dict(episode_id=eid, decision=decision, repeat=repeat, role=role, **metrics))
        _atomic_json(output / "records.json", rows)
        print(f"[action-audit] {index + 1}/{len(queries)} episode={eid} decision={decision}", flush=True)
    summary = {"queries": len(queries), "noise_samples": args.noise_samples,
        "note": "Normalized action error, not physical joint MAE or robot task success", "models": {}}
    for role in ("baseline", "memory-off", "reader"):
        selected = [r for r in rows if r["role"] == role]
        keys = [k for k in selected[0] if k not in ("episode_id", "decision", "repeat", "role")]
        summary["models"][role] = {k: sum(r[k] for r in selected) / len(selected) for k in keys}
    _atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
