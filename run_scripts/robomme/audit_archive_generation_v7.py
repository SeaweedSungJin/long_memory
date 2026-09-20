#!/usr/bin/env python3
"""Paired archive/AE generated-action audit; never reports robot success.

Compare fixed noise across READ on/off and original HAMLET, and across Euler
step counts. Unlike the earlier V8 audit, distinguish nominal target-prefix
error from the actually observed-transition prefix and save coordinate arrays.
No existing checkpoint, cache, training source or evaluation is modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import save_file

from gr00t.long_memory.action_audit_v8 import action_errors, generated_action
from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes, validate_decision
from gr00t.long_memory.checkpoint_v7 import load_checkpoint_v7, v7_checkpoint_info
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import LoRAConfig, adapter_disabled, install_expert_lora, set_expert_trainable
from gr00t.long_memory.hamlet import load_frozen_hamlet, validate_cache_checkpoint
from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from gr00t.long_memory.replay_v7 import replay_queries
from gr00t.long_memory.safety_v5 import validate_output_scope

ROLES = ("baseline", "archive-off", "archive")


@contextmanager
def solver_steps(head, count):
    if type(count) is not int or not 1 <= count <= 64:
        raise ValueError("Euler steps must be an integer in [1,64]")
    previous = head.num_inference_timesteps
    head.num_inference_timesteps = count
    try:
        yield
    finally:
        head.num_inference_timesteps = previous


def prefix_masks(target_mask, observed_steps, nominal_steps):
    """Return two named masks without changing source dataset tensors."""
    if target_mask.ndim != 3 or target_mask.shape[0] != 1 or target_mask.dtype != torch.bool:
        raise ValueError("Target mask must be boolean [1,H,A]")
    if observed_steps.ndim != 1 or observed_steps.dtype != torch.bool:
        raise ValueError("Observed transition mask must be boolean [C]")
    if type(nominal_steps) is not int or not 0 < nominal_steps <= target_mask.shape[1]:
        raise ValueError("Invalid nominal control prefix")
    if len(observed_steps) != nominal_steps:
        raise ValueError("Observed prefix width differs from cached action_steps")
    nominal = target_mask.clone()
    nominal[:, nominal_steps:] = False
    observed = torch.zeros_like(nominal)
    observed[:, :nominal_steps] = nominal[:, :nominal_steps] & observed_steps.to(nominal.device)[None, :, None]
    return nominal, observed


def query_plan(cache, episodes, samples, seed):
    ids = list(cache.manifest["splits"]["val"])
    rng = random.Random(seed)
    rng.shuffle(ids)
    queries = []
    for eid in ids:
        ep = episodes.fetch(eid)
        valid = torch.where(ep["decision_mask"])[0].tolist()
        if valid:
            queries.append([eid, rng.choice(valid)])
        if len(queries) == samples:
            break
    if len(queries) != samples:
        raise ValueError("Not enough distinct held-out episodes for requested samples")
    return queries


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--noise-samples", type=int, default=2)
    p.add_argument("--euler-steps", nargs="+", type=int, default=[4, 8, 16])
    p.add_argument("--seed", type=int, default=8401)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)
    if min(args.samples, args.noise_samples) < 1 or args.seed < 0:
        raise ValueError("Invalid sample count/seed")
    if not args.euler_steps or len(set(args.euler_steps)) != len(args.euler_steps) or any(not 1 <= n <= 64 for n in args.euler_steps):
        raise ValueError("Use distinct Euler counts in [1,64]")
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    base = cache.manifest["model_path"]
    checkpoint = Path(args.checkpoint).resolve()
    info = v7_checkpoint_info(base, checkpoint, expected_stage=1)
    if info["config"]["mode"] != "archive" or info["step"] <= 0:
        raise ValueError("Requires trained Stage-1 archive")
    if info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]:
        raise ValueError("Checkpoint/cache mismatch")
    output = Path(args.output_dir).resolve()
    validate_output_scope(output, cache.path, cache.manifest.get("dataset_path"), base, checkpoint)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a NEW audit output")
    episodes = MappedEpisodes(cache)
    queries = query_plan(cache, episodes, args.samples, args.seed)
    if args.preflight_only:
        print(f"[archive-generation] {len(queries)} distinct val episodes, Euler={args.euler_steps}; no model/output created")
        return 0
    sources = list((ROOT / "gr00t").rglob("*.py")) + [Path(__file__).resolve()]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sources)}
    output.mkdir(parents=True, exist_ok=False)
    plan = {"args": vars(args), "queries": queries, "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256((checkpoint / "checkpoint.json").read_bytes()).hexdigest(),
            "checkpoint_metadata": info["metadata"], "source_sha256": hashes,
            "note": "Offline normalized-coordinate errors, NOT simulator success; targets are read after generation only."}
    _atomic_json(output / "plan.json", plan)
    model, processor = load_frozen_hamlet(base, args.device)
    head = model.action_head
    cfg = info["config"]
    install_expert_lora(head, LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
    memory = RecurrentMemoryV7(MemoryV7Config(**cfg["memory"]))
    cvom = CVOMV7(memory.config)
    load_checkpoint_v7(checkpoint, memory, head, cvom)
    memory.to(device=args.device, dtype=torch.float32).eval().requires_grad_(False)
    set_expert_trainable(head, False)
    head.eval().requires_grad_(False)
    del model, processor, cvom
    rows = []
    for index, (eid, decision) in enumerate(queries):
        ep = episodes.fetch(eid)
        validate_decision(ep, decision)
        with torch.no_grad():
            fused, _ = replay_queries(memory, ep, [decision], mode="archive")[decision]
        arrays = {}
        for repeat in range(args.noise_samples):
            seed = args.seed + index * 100003 + repeat
            for steps in args.euler_steps:
                for role in ROLES:
                    with solver_steps(head, steps), adapter_disabled(head) if role == "baseline" else nullcontext():
                        prediction = generated_action(head, ep, decision, fused if role == "archive" else None, seed=seed)
                    # Targets/masks never participate in generation or retrieval.
                    target = ep["targets"][decision].to(device=prediction.device, dtype=torch.float32)[None]
                    mask = ep["target_mask"][decision].to(device=prediction.device)[None]
                    nominal, observed = prefix_masks(mask, ep["action_mask"][decision], int(cache.manifest["action_steps"]))
                    row = {"episode_id": eid, "decision": decision, "repeat": repeat, "seed": seed,
                           "role": role, "euler_steps": steps}
                    for name, valid in (("target", mask), ("nominal_prefix", nominal), ("observed_prefix", observed)):
                        row[name + "_valid_values"] = int(valid.sum())
                        measured = action_errors(prediction, target, valid) if bool(valid.any()) else {"mse": None, "mae": None}
                        row.update({name + "_" + key: value for key, value in measured.items()})
                    arrays[f"prediction.{role}.steps{steps}.repeat{repeat}"] = prediction.detach().float().cpu().contiguous()
                    arrays.update(target=target.cpu().contiguous(), target_mask=mask.cpu().contiguous(),
                                  nominal_prefix_mask=nominal.cpu().contiguous(), observed_prefix_mask=observed.cpu().contiguous())
                    rows.append(row)
        payload = output / f"query-{index:04d}.safetensors"
        save_file(arrays, str(payload))
        _atomic_json(output / "records.json", rows)
        print(f"[archive-generation] {index+1}/{len(queries)} episode={eid} decision={decision}", flush=True)
    summary = {"queries": len(queries), "noise_samples": args.noise_samples,
               "metric": "equal-query/noise mean normalized error; NOT task success", "results": {}}
    for steps in args.euler_steps:
        for role in ROLES:
            chosen = [row for row in rows if row["role"] == role and row["euler_steps"] == steps]
            metrics = {}
            for prefix in ("target", "nominal_prefix", "observed_prefix"):
                valid = [row for row in chosen if row[prefix + "_mse"] is not None]
                metrics[prefix + "_records"] = len(valid)
                for metric in ("mse", "mae"):
                    key = prefix + "_" + metric
                    metrics[key] = sum(row[key] for row in valid) / len(valid) if valid else None
            summary["results"][f"{role}/euler{steps}"] = metrics
    summary["payload_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(output.glob("query-*.safetensors"))}
    _atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
