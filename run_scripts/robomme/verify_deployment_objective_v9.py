#!/usr/bin/env python3
"""Real-checkpoint sampler parity and gradient-scale probe; NO optimizer step."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from gr00t.long_memory.action_audit_v8 import generated_action
from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes, validate_decision
from gr00t.long_memory.checkpoint_v7 import load_checkpoint_v7, v7_checkpoint_info
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import LoRAConfig, expert_episode_flow_loss, expert_parameters, install_expert_lora, set_expert_trainable
from gr00t.long_memory.hamlet import load_frozen_hamlet, sample_noise_time, validate_cache_checkpoint
from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from gr00t.long_memory.replay_v7 import replay_queries
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.audit_archive_generation_v7 import prefix_masks
from run_scripts.robomme.deployment_objective_v9 import generated_prefix_objective


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--noise-samples", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=9021)
    args = p.parse_args(argv)
    if min(args.samples, args.noise_samples) < 1 or args.seed < 0:
        raise ValueError("Invalid samples or seed")
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    base = cache.manifest["model_path"]
    info = v7_checkpoint_info(base, args.checkpoint, expected_stage=1)
    if info["config"]["mode"] != "archive" or info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]:
        raise ValueError("Requires matching-cache Stage-1 archive")
    output = Path(args.output_dir).resolve()
    validate_output_scope(output, cache.path, cache.manifest.get("dataset_path"), base, args.checkpoint)
    if output.exists():
        raise FileExistsError("Use a new diagnostic output")
    episodes = MappedEpisodes(cache)
    ids = list(cache.manifest["splits"]["train"])
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    queries = []
    for eid in ids:
        ep = episodes.fetch(eid)
        eligible = [q for q in torch.where(ep["decision_mask"])[0].tolist() if q >= 3 and bool(ep["action_mask"][q].any())]
        if eligible:
            queries.append([eid, rng.choice(eligible)])
        if len(queries) == args.samples:
            break
    if len(queries) != args.samples:
        raise ValueError("Insufficient train queries")
    output.mkdir(parents=True, exist_ok=False)
    sources = list((ROOT / "gr00t").rglob("*.py")) + [Path(__file__).resolve(),
        ROOT / "run_scripts/robomme/deployment_objective_v9.py",
        ROOT / "run_scripts/robomme/audit_archive_generation_v7.py"]
    _atomic_json(output / "plan.json", {"args": vars(args), "queries": queries,
        "checkpoint_metadata": info["metadata"],
        "checkpoint_sha256": hashlib.sha256((Path(args.checkpoint) / "checkpoint.json").read_bytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sources)},
        "note": "Gradient diagnosis on TRAIN observations only; NO optimizer/training updates."})
    model, processor = load_frozen_hamlet(base, args.device)
    head = model.action_head
    cfg = info["config"]
    install_expert_lora(head, LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
    memory = RecurrentMemoryV7(MemoryV7Config(**cfg["memory"]))
    critic = CVOMV7(memory.config)
    load_checkpoint_v7(args.checkpoint, memory, head, critic)
    memory.to(device=args.device).eval().requires_grad_(True)
    set_expert_trainable(head, True)
    del model, processor, critic
    if head.num_inference_timesteps != 4:
        raise ValueError("Current experiment requires the unchanged deployed Euler4 configuration")
    records = []
    named = [("memory."+name, p) for name, p in memory.named_parameters()] + [("expert."+str(i), p) for i, p in enumerate(expert_parameters(head))]
    for eid, query, repeat in [(eid, query, repeat) for eid, query in queries for repeat in range(args.noise_samples)]:
        noise_seed = args.seed + eid * 100003 + query * 997 + repeat
        cached = episodes.fetch(eid)
        validate_decision(cached, query)
        ref = next(head.parameters())
        _, flow_time = sample_noise_time(head, cached["targets"][query].to(ref)[None], noise_seed)
        flow_time_value = float(flow_time.item())
        with torch.no_grad():
            fused, _ = replay_queries(memory, cached, [query], mode="archive")[query]
            reference = generated_action(head, cached, query, fused, seed=noise_seed).detach()
        flow_gradients, flow_norms = None, None
        for kind in ("flow", "generated_all8", "generated_joint7", "generated_gripper1"):
            ep = dict(cached)
            ep["short"] = cached["short"].float().to(args.device).detach().requires_grad_(True)
            fused, _ = replay_queries(memory, ep, [query], mode="archive")[query]
            if kind == "flow":
                result = expert_episode_flow_loss(head, ep, query, fused, seed=noise_seed, activation_checkpointing=True)
                loss = result["loss"]
                parity = None
            else:
                result = generated_prefix_objective(head, ep, query, fused, seed=noise_seed,
                    action_steps=int(cache.manifest["action_steps"]), activation_checkpointing=True)
                parity = float((reference - result["prediction"]).abs().max())
                if not torch.equal(reference, result["prediction"]):
                    raise RuntimeError(f"Deployed sampler parity failed: max error={parity}")
                target = ep["targets"][query].to(args.device).float()[None]
                mask = ep["target_mask"][query].to(args.device)[None]
                _, valid = prefix_masks(mask, ep["action_mask"][query], int(cache.manifest["action_steps"]))
                if kind == "generated_joint7":
                    valid[:, :, 7:] = False
                elif kind == "generated_gripper1":
                    valid[:, :, :7] = False
                    valid[:, :, 8:] = False
                if not bool(valid.any()):
                    records.append({"episode_id": eid, "decision": query, "repeat": repeat, "noise_seed": noise_seed,
                                    "kind": kind, "missing_supervision": True})
                    continue
                loss = (result["prediction"].float() - target)[valid].square().mean()
            grads = torch.autograd.grad(loss, [p for _, p in named] + [ep["short"]], allow_unused=True)
            norms = {"memory_encoder": 0., "memory_reader": 0., "memory_fusion": 0., "expert": 0.}
            dot_products = dict.fromkeys(norms, 0.)
            nonzero = 0
            for parameter_index, ((name, _), grad) in enumerate(zip(named, grads[:-1])):
                if grad is None:
                    continue
                if not bool(torch.isfinite(grad).all()):
                    raise FloatingPointError(f"Nonfinite gradient {name}")
                group = "expert" if name.startswith("expert.") else "memory_fusion" if name.startswith("memory.fusion_") else "memory_encoder" if name.startswith(("memory.short_", "memory.state_projection", "memory.time_encoder")) else "memory_reader"
                norms[group] += float(grad.float().square().sum())
                if flow_gradients is not None and flow_gradients[parameter_index] is not None:
                    dot_products[group] += float((grad.float().cpu() * flow_gradients[parameter_index]).sum())
                nonzero += int(bool(grad.any()))
            if kind == "flow":
                flow_gradients = [None if grad is None else grad.detach().float().cpu() for grad in grads[:-1]]
                flow_norms = dict(norms)
                cosines = None
            else:
                cosines = {group: dot_products[group] / (norms[group] * flow_norms[group])**.5
                           if norms[group] * flow_norms[group] > 1e-24 else None for group in norms}
            source_grad = grads[-1]
            if source_grad is None or not bool(torch.isfinite(source_grad).all()):
                raise RuntimeError("Past-memory input gradient missing/nonfinite")
            record = {"episode_id": eid, "decision": query, "kind": kind, "loss": float(loss.detach()),
                "repeat": repeat, "noise_seed": noise_seed, "flow_time": flow_time_value,
                "sampler_max_abs_error": parity, "parameter_grad_norms": {k: v**.5 for k, v in norms.items()},
                "gradient_cosine_with_flow": cosines,
                "nonzero_parameter_grads": nonzero, "past_short_grad_norm": float(source_grad[:query].norm()),
                "future_short_grad_norm": float(source_grad[query+1:].norm()),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
            if record["future_short_grad_norm"] != 0:
                raise RuntimeError("Leaked future gradient")
            record["past_gradient_nonzero"] = record["past_short_grad_norm"] > 0
            records.append(record)
            _atomic_json(output / "records.json", records)
            print(json.dumps(record), flush=True)
            del result, loss, grads, source_grad, fused, ep
    _atomic_json(output / "summary.json", {"complete": True, "optimizer_steps": 0, "queries": queries,
        "noise_samples": args.noise_samples, "records": len(records), "all_sampler_checks_exact": True,
        "all_checked_gradients_finite": True,
        "zero_past_gradient_records": sum(not row.get("past_gradient_nonzero", False) for row in records
                                          if not row.get("missing_supervision"))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
