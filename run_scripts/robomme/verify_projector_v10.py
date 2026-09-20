#!/usr/bin/env python3
"""Actual-HAMLET zero-delta parity/gradient probe; no optimizer or saved model.

CPU is deliberately supported while GPUs run immutable simulator evaluations.
This is a one-query numerical implementation check, NOT a learning experiment,
gradient-scale population estimate, or robot-success measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from gr00t.long_memory.action_audit_v8 import generated_action
from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes, validate_decision
from gr00t.long_memory.checkpoint_v7 import load_checkpoint_v7, v7_checkpoint_info
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import (LoRAConfig, expert_episode_flow_loss as old_flow,
    expert_parameters, install_expert_lora, set_expert_trainable)
from gr00t.long_memory.hamlet import isolated_seed, load_frozen_hamlet, validate_cache_checkpoint
from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from gr00t.long_memory.replay_v7 import replay_queries
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.projector_adapter_v10 import (install_projector, set_trainable,
    expert_episode_flow_loss, generated_prefix_metrics, projector_parameters)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def actual_head(base, device):
    """CPU probe loads the real head only; cached inputs already contain VLM output.

The untouched full HAMLET loader constructs a FlashAttention2 backbone, which
requires CUDA even when a diagnostic does not call the backbone. Do not change
that live production loader to make a CPU-only head check pass.
"""
    if torch.device(device).type != "cpu":
        model, processor = load_frozen_hamlet(base, device)
        return model.action_head
    from safetensors import safe_open
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead
    base = Path(base).resolve()
    config = Gr00tN1d6Config.from_pretrained(base)
    with isolated_seed(0, "cpu"):
        head = Gr00tN1d6ActionHead(config).to(device="cpu", dtype=torch.bfloat16)
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "action_head."
    keys = {name[len(prefix):]: shard for name, shard in index.items() if name.startswith(prefix)}
    if set(keys) != set(head.state_dict()):
        raise ValueError("CPU actual Action Expert has missing/unexpected checkpoint keys")
    state = {}
    for shard in sorted(set(keys.values())):
        path = (base / shard).resolve()
        if not path.is_relative_to(base):
            raise ValueError("Head checkpoint shard escapes original base")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in keys:
                if keys[name] == shard:
                    value = handle.get_tensor(prefix + name)
                    if tuple(value.shape) != tuple(head.state_dict()[name].shape) or not bool(torch.isfinite(value).all()):
                        raise ValueError(f"Invalid actual Action Expert tensor: {name}")
                    state[name] = value
    head.load_state_dict(state, strict=True)
    head.beta_dist = torch.distributions.Beta(
        torch.tensor(float(config.noise_beta_alpha), dtype=torch.float32),
        torch.tensor(float(config.noise_beta_beta), dtype=torch.float32))
    return head.eval().requires_grad_(False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=9051)
    args = p.parse_args(argv)
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    source = Path(args.checkpoint).resolve()
    base = cache.manifest["model_path"]
    info = v7_checkpoint_info(base, source, expected_stage=1)
    if info["config"]["mode"] != "archive" or info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]:
        raise ValueError("Requires same-cache V7 Stage-1 archive")
    output = Path(args.output_dir).resolve()
    validate_output_scope(output, cache.path, cache.manifest.get("dataset_path"), base, source)
    if output.exists():
        raise FileExistsError("Use a new numerical verification output")
    episodes = MappedEpisodes(cache)
    # One predeclared deterministic TRAIN query, not chosen by its loss/result.
    choice = None
    for eid in sorted(cache.manifest["splits"]["train"]):
        ep = episodes.fetch(eid)
        eligible = [q for q in torch.where(ep["decision_mask"])[0].tolist()
                    if q >= 3 and bool(ep["action_mask"][q].any())]
        if eligible:
            choice = eid, eligible[0], ep
            break
    if choice is None:
        raise ValueError("No eligible TRAIN query with causal history")
    eid, query, ep = choice
    validate_decision(ep, query)
    inputs_before = {name: sha(source / name) for name in
                    ("checkpoint.json", "model.safetensors", "expert.safetensors", "cvom.safetensors", "training_state.pt")}
    output.mkdir(parents=True, exist_ok=False)
    sources = list((ROOT / "gr00t").rglob("*.py")) + [Path(__file__).resolve(),
        ROOT / "run_scripts/robomme/projector_adapter_v10.py",
        ROOT / "run_scripts/robomme/deployment_objective_v9.py",
        ROOT / "run_scripts/robomme/audit_archive_generation_v7.py"]
    _atomic_json(output / "plan.json", {"args": vars(args), "episode_id": eid, "query": query,
        "checkpoint_payload_sha256": inputs_before, "base_identity": info["metadata"]["base_model"],
        "cache_fingerprint": cache.manifest["fingerprint"],
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(sources))},
        "note": "One TRAIN query, no optimizer/update; not robot accuracy."})
    print(f"[verify-v10] loading actual HAMLET on {args.device}; train episode={eid}, query={query}", flush=True)
    head = actual_head(base, args.device)
    cfg = info["config"]
    with isolated_seed(args.seed, args.device):
        install_expert_lora(head, LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
        memory = RecurrentMemoryV7(MemoryV7Config(**cfg["memory"]))
        cvom = CVOMV7(memory.config)
    load_checkpoint_v7(source, memory, head, cvom)
    del cvom
    memory.to(args.device).eval().requires_grad_(True)
    set_expert_trainable(head, True)
    old_parameters = [*memory.parameters(), *expert_parameters(head)]
    fused = replay_queries(memory, ep, [query], mode="archive")[query][0]
    old = old_flow(head, ep, query, fused, seed=args.seed)
    old_prediction, old_loss = old["prediction"].detach().clone(), old["loss"].detach().clone()
    old_gradients = torch.autograd.grad(old["loss"], old_parameters, allow_unused=True)
    del old, fused
    # CUDA attention backward may use nondeterministic parallel reductions.
    # Measure the untouched implementation against ITSELF before interpreting
    # any zero-wrapper difference; CPU is still required to be bitwise exact.
    fused = replay_queries(memory, ep, [query], mode="archive")[query][0]
    repeated = old_flow(head, ep, query, fused, seed=args.seed)
    torch.testing.assert_close(repeated["prediction"], old_prediction, atol=0, rtol=0)
    repeated_gradients = torch.autograd.grad(repeated["loss"], old_parameters, allow_unused=True)
    del repeated, fused
    with torch.no_grad():
        fused = replay_queries(memory, ep, [query], mode="archive")[query][0]
        reference_actions = generated_action(head, ep, query, fused, seed=args.seed).detach().clone()
    print("[verify-v10] original flow/gradient/Euler4 computed", flush=True)
    install_projector(head, enabled=True)
    set_trainable(head, train_projector=True, train_lora=True)
    fused = replay_queries(memory, ep, [query], mode="archive")[query][0]
    result = expert_episode_flow_loss(head, ep, query, fused, seed=args.seed)
    torch.testing.assert_close(result["prediction"], old_prediction, atol=0, rtol=0)
    torch.testing.assert_close(result["loss"], old_loss, atol=0, rtol=0)
    new_gradients = torch.autograd.grad(result["loss"], old_parameters + list(projector_parameters(head)), allow_unused=True)
    gradient_rows = []
    # These are numerical-closeness checks, NOT bitwise-equivalence claims.
    # Earlier CUDA strict probes also failed on original-vs-original gradients
    # (1/2048 elements, max absolute error 1.86e-8). Keep all per-parameter
    # measurements even when a tolerance fails, rather than losing the evidence
    # at the first element. CPU still has zero tolerance.
    rtol, atol = (1e-3, 1e-7) if torch.device(args.device).type == "cuda" else (0., 0.)
    for index, (left, repeat, right) in enumerate(zip(old_gradients, repeated_gradients, new_gradients[:len(old_gradients)])):
        if (left is None) != (right is None) or (left is None) != (repeat is None):
            raise AssertionError("Existing gradient connectivity changed at zero delta")
        if left is not None:
            scale = float(left.float().norm())
            gradient_rows.append({"parameter_index": index,
                "finite": all(bool(torch.isfinite(g).all()) for g in (left, repeat, right)),
                "self_repeat_close": bool(torch.allclose(left, repeat, atol=atol, rtol=rtol)),
                "wrapper_close": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
                "reference_norm": scale,
                "self_repeat_max_abs": float((left - repeat).abs().max()),
                "wrapper_max_abs": float((left - right).abs().max()),
                "self_repeat_relative_l2": float((left - repeat).float().norm()) / max(scale, 1e-30),
                "wrapper_relative_l2": float((left - right).float().norm()) / max(scale, 1e-30)})
    delta_gradients = new_gradients[len(old_gradients):]
    if any(g is None or not bool(torch.isfinite(g).all()) or not bool(g.count_nonzero()) for g in delta_gradients):
        raise AssertionError("New projector did not receive finite nonzero gradients")
    with torch.no_grad():
        generated = generated_prefix_metrics(head, ep, query, fused.detach(), seed=args.seed)
        torch.testing.assert_close(generated["prediction"], reference_actions, atol=0, rtol=0)
    if inputs_before != {name: sha(source / name) for name in inputs_before}:
        raise RuntimeError("Source checkpoint changed during read-only probe")
    gradients_close = all(r["finite"] and r["self_repeat_close"] and r["wrapper_close"] for r in gradient_rows)
    report = {"passed": gradients_close, "episode_id": eid, "query": query, "device": args.device,
        "base_dtype": str(next(head.parameters()).dtype), "flow_loss": float(old_loss),
        "zero_delta_flow_prediction_exact": True,
        "existing_gradients_exact": all(r["wrapper_max_abs"] == 0 for r in gradient_rows),
        "existing_gradients_close": gradients_close, "gradient_tolerance": {"rtol": rtol, "atol": atol},
        "self_repeat_max_abs": max(r["self_repeat_max_abs"] for r in gradient_rows),
        "wrapper_max_abs": max(r["wrapper_max_abs"] for r in gradient_rows),
        "self_repeat_max_relative_l2": max(r["self_repeat_relative_l2"] for r in gradient_rows),
        "wrapper_max_relative_l2": max(r["wrapper_relative_l2"] for r in gradient_rows),
        "gradient_records": gradient_rows,
        "zero_delta_generated_actions_exact": True, "parent_payloads_unchanged": True,
        "existing_parameters_with_gradient": sum(g is not None for g in old_gradients),
        "projector_gradient_norms": [float(g.float().norm()) for g in delta_gradients],
        "optimizer_updates": 0, "cpu_head_only": torch.device(args.device).type == "cpu",
        "note": "Actual Action Expert on cached VLM features; not a full VLM/rollout or convergence/success test."}
    _atomic_json(output / "result.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "gradient_records"}, indent=2), flush=True)
    return 0 if gradients_close else 1


if __name__ == "__main__":
    raise SystemExit(main())
