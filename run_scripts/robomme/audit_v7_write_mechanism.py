#!/usr/bin/env python3
"""CPU-only V7 mechanistic audit; existing source/checkpoints are read-only.

This measures latent/conditioning sensitivity, not action loss or success rate.
One validation episode from each contiguous 100-episode dataset block is the
default sample. No action tensor is read, no model optimizer is constructed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load_file
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def scalar(x):
    return float(x.detach())


def rms(x):
    return x.square().mean().sqrt()


def spread(x, dim=1):
    return rms(x - x.mean(dim, keepdim=True))


def diversity(x):
    magnitude = scalar(rms(x))
    difference = scalar(spread(x))
    unit = F.normalize(x.double(), dim=-1)
    count = x.shape[1]
    cosine = scalar(((unit.sum(1).square().sum(-1) - unit.square().sum((1, 2)))
                     / max(1, count * (count - 1))).mean())
    return {"rms": magnitude, "slot_rms_spread": difference,
            "relative_spread": difference / max(magnitude, 1e-30),
            "off_diagonal_cosine_fp64": cosine}


def trace_write(memory, state, encoded, *, variant="real", details=True):
    """Reproduce WRITE exactly, also exposing otherwise hidden intermediates."""
    query = memory.write_query_norm(state + memory.slot_addresses[None])
    if variant in ("fixed_address_queries", "centered_keys_fixed_queries"):
        query = memory.write_query_norm(memory.slot_addresses[None]).expand_as(state)
    key_input = encoded
    if variant in ("centered_write_keys", "centered_keys_fixed_queries"):
        key_input = encoded - encoded.mean(1, keepdim=True)
    key = memory.write_key_norm(key_input)
    proposal, _ = memory.attention(query, key, encoded, need_weights=False)
    values = proposal + memory.write_ffn(memory.write_output_norm(proposal))
    gate = torch.sigmoid(memory.update_gate(torch.cat((state, values), dim=-1)))
    if variant == "fixed_multiscale_gate":
        # Diagnostic only: explicit temporal write diversity, not a trained V8.
        timescales = torch.logspace(math.log10(4), math.log10(256), state.shape[1])
        gate = (1 - torch.exp(-1 / timescales))[None, :, None].expand(state.shape[0], -1, -1)
    candidate = (1 - gate) * state + gate * values
    if not details:
        return candidate
    width, heads = memory.config.hidden_dim, memory.config.num_heads
    wq, wk, _ = memory.attention.in_proj_weight.split(width)
    q = F.linear(query, wq).reshape(state.shape[0], -1, heads, width // heads).transpose(1, 2)
    k = F.linear(key, wk).reshape(state.shape[0], -1, heads, width // heads).transpose(1, 2)
    weights = (q @ k.transpose(-2, -1) / math.sqrt(width // heads)).softmax(-1)
    info = {"query_slot_spread": scalar(spread(query)),
            "key_token_spread": scalar(spread(key)),
            "write_attention_slot_spread": scalar(spread(weights, dim=2)),
            "write_attention_entropy": scalar(-(weights * weights.clamp_min(1e-30).log()).sum(-1).mean()),
            "proposal_rms": scalar(rms(proposal)), "proposal_slot_spread": scalar(spread(proposal)),
            "value_rms": scalar(rms(values)), "value_slot_spread": scalar(spread(values)),
            "gate_mean": scalar(gate.mean()), "gate_std": scalar(gate.std(unbiased=False)),
            "gate_range": scalar(gate.max() - gate.min()),
            "state_to_address_rms_ratio": scalar(rms(state) / rms(memory.slot_addresses)),
            **diversity(candidate)}
    return candidate, info


def encode_parts(memory, episode):
    short = episode["short"].float()
    projected = memory.short_projection(memory.short_norm(short))
    state = memory.state_projection(episode["state"].float())
    z = episode["frames"].float() / memory.config.time_scale
    time_input = torch.stack([v for scale in (1., 10., 100.)
                              for v in (torch.sin(z / scale), torch.cos(z / scale))]
                             + [torch.log1p(z), episode["is_demo"].float()], dim=-1)
    time_part = memory.time_encoder(time_input)
    encoded = projected + memory.short_token_ids[None] + state[:, None] + time_part[:, None]
    native = memory.encode(short, episode["state"], episode["frames"], episode["is_demo"])
    torch.testing.assert_close(encoded, native, rtol=0, atol=0)
    result = {"raw_short_rms": scalar(rms(short)), "raw_short_token_spread": scalar(spread(short)),
              "projected_short_rms": scalar(rms(projected)),
              "projected_short_token_spread": scalar(spread(projected)),
              "token_id_rms": scalar(rms(memory.short_token_ids)),
              "state_component_rms": scalar(rms(state)), "time_component_rms": scalar(rms(time_part)),
              "state_plus_time_rms": scalar(rms(state + time_part)),
              "projected_short_mean_rms": scalar(rms(projected.mean(1))),
              "encoded_rms": scalar(rms(encoded)), "encoded_token_spread": scalar(spread(encoded)),
              "encoded_relative_token_spread": scalar(spread(encoded) / rms(encoded)),
              "normalized_key_token_spread": scalar(spread(memory.write_key_norm(encoded)))}
    return short, encoded, projected + memory.short_token_ids[None], result


def replay(memory, encoded, *, variant="real", traced=False):
    bank = memory.initial_state()
    rows = []
    for index in range(len(encoded)):
        if traced:
            old = bank
            bank, record = trace_write(memory, bank, encoded[index:index + 1], variant=variant)
            if variant == "real" and index in (0, len(encoded) - 1):
                native, _ = memory.write(old, encoded[index:index + 1])
                torch.testing.assert_close(bank, native, rtol=0, atol=0)
            rows.append(dict(write_index=index, **record))
        else:
            bank = trace_write(memory, bank, encoded[index:index + 1], variant=variant, details=False)
    return bank, rows


def probe_gradient(memory, encoded, short_query, encoded_query, seed):
    """VJP of a fixed random fused-token direction, not an action gradient."""
    memory.zero_grad(set_to_none=True)
    sequence = encoded.detach().clone().requires_grad_(True)
    bank, _ = replay(memory, sequence)
    fused, _ = memory.read(short_query, encoded_query.detach(), bank)
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(fused.shape, generator=generator)
    direction = direction / direction.norm()
    (fused * direction).sum().backward()
    per_observation = sequence.grad.flatten(1).norm(dim=-1)
    groups = defaultdict(float)
    for name, parameter in memory.named_parameters():
        norm = 0. if parameter.grad is None else scalar(parameter.grad.norm())
        groups[name] = norm
    width = memory.config.hidden_dim
    mha = memory.attention.in_proj_weight.grad
    if mha is not None:
        for name, tensor in zip(("q", "k", "v"), mha.split(width)):
            groups[f"attention_{name}_grad_norm"] = scalar(tensor.norm())
    result = {"probe": "unit random direction of fused-short output; no Action Expert loaded",
              "encoded_input_grad_norm_by_write": per_observation.tolist(),
              "first_to_last_input_grad_ratio": scalar(per_observation[0] / per_observation[-1].clamp_min(1e-30)),
              "parameter_grad_norm": dict(groups)}
    memory.zero_grad(set_to_none=True)
    return result


def direct_write_gradients(memory, encoded, seed):
    result = {}
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(memory.initial_state().shape, generator=generator)
    direction /= direction.norm()
    for label, x in (("real", encoded), ("identical_four_tokens", encoded.mean(1, keepdim=True).expand_as(encoded)),
                     ("distinct_synthetic", torch.randn(encoded.shape, generator=generator) * rms(encoded))):
        memory.zero_grad(set_to_none=True)
        candidate = trace_write(memory, memory.initial_state(), x.detach(), details=False)
        (candidate * direction).sum().backward()
        width = memory.config.hidden_dim
        attn = memory.attention.in_proj_weight.grad
        result[label] = {"slot_address_gradient": scalar(memory.slot_addresses.grad.norm()),
                         **{f"attention_{name}_gradient": scalar(grad.norm())
                            for name, grad in zip(("q", "k", "v"), attn.split(width))}}
    memory.zero_grad(set_to_none=True)
    return result


def perturbation_audit(memory, encoded, short_query, encoded_query, seed):
    """Only explicit old encoded observations change; future frozen X stays fixed."""
    count = len(encoded)
    bank, _ = replay(memory, encoded)
    fused, _ = memory.read(short_query, encoded_query, bank)
    averaged_bank = bank.mean(1, keepdim=True).expand_as(bank)
    averaged_fused, _ = memory.read(short_query, encoded_query, averaged_bank)
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(encoded[:1].shape, generator=generator)
    direction *= 0.1 * rms(encoded) / rms(direction)
    rows = []
    for index in sorted({0, count // 2, count - 1}):
        changed = encoded.clone()
        changed[index:index + 1] += direction
        changed_bank, _ = replay(memory, changed)
        changed_fused, _ = memory.read(short_query, encoded_query, changed_bank)
        rows.append({"perturbed_write_index": index, "later_writes": count - 1 - index,
                     "input_delta_rms": scalar(rms(direction)),
                     "final_memory_delta_rms": scalar(rms(changed_bank - bank)),
                     "final_memory_delta_relative": scalar(rms(changed_bank - bank) / rms(bank)),
                     "final_fused_short_delta_rms": scalar(rms(changed_fused - fused))})
    interventions = {}
    for label, start in (("last_4_writes_only", max(0, count - 4)),
                         ("last_16_writes_only", max(0, count - 16))):
        changed_bank, _ = replay(memory, encoded[start:])
        changed_fused, _ = memory.read(short_query, encoded_query, changed_bank)
        interventions[label] = {"final_memory_delta_relative": scalar(rms(changed_bank - bank) / rms(bank)),
                                "final_fused_short_delta_rms": scalar(rms(changed_fused - fused))}
    reversed_old = encoded.clone()
    old_end = max(0, count - 4)
    reversed_old[:old_end] = encoded[:old_end].flip(0)
    changed_bank, _ = replay(memory, reversed_old)
    changed_fused, _ = memory.read(short_query, encoded_query, changed_bank)
    interventions["reverse_old_prefix_keep_last_4"] = {
        "reversed_observations": old_end,
        "final_memory_delta_relative": scalar(rms(changed_bank - bank) / rms(bank)),
        "final_fused_short_delta_rms": scalar(rms(changed_fused - fused))}
    return {"single_observation_perturbations": rows, "prefix_interventions": interventions,
            "replace_all_slots_by_their_mean": {
                "fused_short_delta_rms": scalar(rms(averaged_fused - fused)),
                "fused_short_delta_max_abs": scalar((averaged_fused - fused).abs().max())},
            "limitation": "Frozen later HAMLET shorts can already contain old history; only explicit V7 writes change."}


def decay_probe(memory, real_encoded, seed):
    """128 chronological writes; inject old content once and measure survival."""
    generator = torch.Generator().manual_seed(seed)
    sequence = torch.cat([real_encoded] * math.ceil(128 / len(real_encoded)), dim=0)[:128]
    delta = torch.randn(sequence[:1].shape, generator=generator)
    delta *= 0.1 * rms(sequence) / rms(delta)
    result = {}
    for variant in ("real", "fixed_address_queries", "fixed_multiscale_gate"):
        bank = memory.initial_state()
        changed_bank = bank.clone()
        initial_delta = None
        curve = []
        for index in range(len(sequence)):
            x = sequence[index:index + 1]
            bank = trace_write(memory, bank, x, variant=variant, details=False)
            changed_bank = trace_write(memory, changed_bank, x + (delta if index == 0 else 0),
                                       variant=variant, details=False)
            difference = scalar(rms(changed_bank - bank))
            if initial_delta is None:
                initial_delta = difference
            if index in (0, 1, 3, 7, 15, 31, 63, 127):
                curve.append({"later_writes": index, "memory_delta_rms": difference,
                              "retention_relative_to_immediate_write": difference / max(initial_delta, 1e-30),
                              "per_slot_delta_rms": (changed_bank - bank).square().mean(-1).sqrt().flatten().tolist(),
                              "slot_relative_spread": diversity(bank)["relative_spread"]})
        result[variant] = curve
    return {"input_sequence": "real encoded episode cycled to 128 writes; chronology is synthetic beyond its end",
            "input_delta_rms": scalar(rms(delta)), "curves": result}


def aggregate(rows, key):
    values = torch.tensor([row[key] for row in rows], dtype=torch.float64)
    return {"mean": scalar(values.mean()), "median": scalar(values.median()),
            "min": scalar(values.min()), "max": scalar(values.max())}


def write_report(output, result):
    lines = ["# V7 write mechanism audit", "", result["scope"], "",
             "## Real cached observations", "",
             "| Checkpoint | Raw token spread/RMS | Encoded token spread/RMS | Final slot spread/RMS | Final gate std |",
             "|---|---:|---:|---:|---:|"]
    for checkpoint in result["checkpoints"]:
        episodes = checkpoint["episodes"]
        raw = sum(e["encoding"]["raw_short_token_spread"] / e["encoding"]["raw_short_rms"] for e in episodes) / len(episodes)
        x = sum(e["encoding"]["encoded_relative_token_spread"] for e in episodes) / len(episodes)
        slot = sum(e["variants"]["real"]["final"]["relative_spread"] for e in episodes) / len(episodes)
        gate = sum(e["variants"]["real"]["trajectory"][-1]["gate_std"] for e in episodes) / len(episodes)
        lines.append(f"| {checkpoint['step']} | {raw:.6g} | {x:.6g} | {slot:.6g} | {gate:.6g} |")
    final = result["checkpoints"][-1]
    episodes = final["episodes"]
    def mean_encoding(key):
        return sum(e["encoding"][key] for e in episodes) / len(episodes)
    def mean_trace(key, index=-1):
        return sum(e["variants"]["real"]["trajectory"][index][key] for e in episodes) / len(episodes)
    lines += ["", f"Trained checkpoint {final['step']} averages over {len(episodes)} episodes:", "",
              f"- Shared state-plus-time RMS: {mean_encoding('state_plus_time_rms'):.6g}; "
              f"projected short-token mean RMS: {mean_encoding('projected_short_mean_rms'):.6g}. "
              "The shared robot state/time terms do not dominate these inputs.",
              f"- Final bank/address RMS ratio: {mean_trace('state_to_address_rms_ratio'):.6g}. "
              f"Query slot spread contracts from {mean_trace('query_slot_spread', 0):.6g} "
              f"to {mean_trace('query_slot_spread'):.6g}.",
              f"- Final WRITE proposal slot spread: {mean_trace('proposal_slot_spread'):.6g}; "
              f"after the FFN: {mean_trace('value_slot_spread'):.6g}. "
              "The proposal is already collapsed before its FFN.",
              f"- Replacing all 64 final slots by their mean changes fused short tokens by only "
              f"{sum(e['perturbation']['replace_all_slots_by_their_mean']['fused_short_delta_rms'] for e in episodes) / len(episodes):.6g} RMS.",
              "", "## Same-checkpoint interventions", "",
              "| Input/write intervention | Mean final slot spread/RMS | Mean slot cosine (FP64) |",
              "|---|---:|---:|"]
    for label, stats in final["variant_summary"].items():
        lines.append(f"| {label} | {stats['relative_spread']['mean']:.6g} | "
                     f"{stats['off_diagonal_cosine_fp64']['mean']:.12g} |")
    lines += ["", "`token_centered` removes the token-common content from both keys and values; "
              "it can discard task content. `centered_write_keys` centers only keys before normalization, "
              "preserving all original value content. `fixed_multiscale_gate` uses fixed time scales "
              "4 through 256 writes. Its large spread is partly different slot magnitudes; the cosine "
              "column prevents equating spread alone with semantic event separation."]
    lines += ["", "## Mechanism", "",
              "Each slot attends over the same four current values. If all four values are equal, "
              "their weighted sum is independent of the query, so address and WRITE Q/K gradients vanish "
              "in exact arithmetic (the measured FP32 values can have roundoff). "
              "The identical-token intervention tests this limiting case. Small real token contrasts give "
              "a weak query-to-value route even when the slot queries differ. Query-dependent attention "
              "variation requires key contrast, and its effect on the weighted value sum also requires "
              "value contrast: with nearly equal K/V both factors suppress differentiation.", "",
              "As the shared bank content grows, it also dominates `state + slot_addresses` before query "
              "normalization. The fixed-address-query intervention isolates this additional contraction. "
              "The update gate does not receive slot addresses and its final weight starts at zero, so "
              "it begins exactly uniform and receives almost identical inputs in the collapsed state.", "",
              "READ returns a weighted sum of memory values. If those values coincide, making attention "
              "sharper does not change the recalled content. A 64-slot count then does not imply 64 "
              "independently retained observations.", "",
              "## Minimum architectural recommendation", "",
              "Give slots explicit, different temporal write behavior: feed slot identity to the update "
              "gate and initialize per-slot gate biases across several retention time scales. Keep slot "
              "address queries separate from the growing shared content (or normalize the two branches "
              "separately). Center only WRITE keys before normalization if preserving a current-token "
              "attention write; that gives token differences usable scale without deleting common values. "
              "The fixed multiscale-gate intervention demonstrates the first mechanism on "
              "the same real X without changing stored values by adding arbitrary address vectors. "
              "It is a diagnostic, not evidence of higher action success. A bounded chronological "
              "event store is the more direct alternative if the task needs individually retrievable events.", "",
              "## Gradient paths and chronological survival", ""]
    for checkpoint in result["checkpoints"]:
        es = checkpoint["episodes"]
        direct = es[0]["direct_write_gradients"]
        curve = checkpoint["128_write_retention"]["curves"]["real"][-1]
        first_last = sum(e["gradient"]["first_to_last_input_grad_ratio"] for e in es) / len(es)
        perturb_rms = sum(e["perturbation"]["single_observation_perturbations"][0]["final_fused_short_delta_rms"] for e in es) / len(es)
        lines += [f"Checkpoint {checkpoint['step']}:", "",
                  f"- Direct first-WRITE random-probe Q/V gradient norms: "
                  f"real {direct['real']['attention_q_gradient']:.6g}/{direct['real']['attention_v_gradient']:.6g}; "
                  f"distinct synthetic {direct['distinct_synthetic']['attention_q_gradient']:.6g}/"
                  f"{direct['distinct_synthetic']['attention_v_gradient']:.6g}.",
                  f"- Mean first/last encoded-input VJP norm ratio: {first_last:.6g}. "
                  f"Perturbing the first old write changes final fused tokens by {perturb_rms:.6g} RMS.",
                  f"- A separate 128-write repeated-real-input stress test retains "
                  f"{100 * curve['retention_relative_to_immediate_write']:.6g}% of the immediate "
                  "first-write perturbation after 127 later writes. This repeated sequence is synthetic "
                  "beyond the original episode's end.", ""]
    lines += ["The trained graph retains an old-observation path; slot collapse does not mean "
              "all history is detached or immediately erased. It means most retained content is a "
              "shared temporal aggregate, with very little independent information across slots.", "",
              "## Limits and reproduction", "",
              "This is a deterministic FP32 CPU mechanism audit, not an Action Expert loss or rollout "
              "evaluation. Parameter/input VJPs use a fixed random direction of fused conditioning. "
              "At checkpoint zero, the zero fusion projection intentionally blocks the upstream "
              "conditioning-loss gradient. Separate direct-WRITE gradients test the route behind it. "
              "Old-input perturbations leave future frozen HAMLET shorts unchanged, so they isolate "
              "explicit V7 memory; they cannot remove old information already in those shorts.", "",
              "Full per-write trajectories, intervention outputs, checkpoint/cache fingerprints, "
              "gradient norms, and chronological perturbation curves are in `audit.json`.", "",
              "```bash", result["command"], "```", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/long_memory/cache_full1600_v1")
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=[
        ROOT / "runs/long_memory/v7_recurrent_full_v1/checkpoint-000000",
        ROOT / "runs/long_memory/v7_recurrent_full_v1/checkpoint-009108"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-block", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes_per_block < 1:
        parser.error("--episodes-per-block must be positive")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    source_hash = sha256(ROOT / "gr00t/long_memory/recurrent_v7.py")
    manifest = json.loads((args.cache / "manifest.json").read_text())
    blocks = defaultdict(list)
    for record in manifest["episodes"]:
        if record["split"] == "val":
            blocks[int(record["episode_id"]) // 100].append(record)
    records = [r for block in sorted(blocks) for r in sorted(blocks[block], key=lambda r: r["episode_id"])[:args.episodes_per_block]]
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"scope": "FP32 CPU audit of validation-cache observations and immutable V7 checkpoints; no training or simulator.",
              "command": "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python " + " ".join(sys.argv),
              "seed": args.seed, "torch_version": torch.__version__, "threads": 2,
              "source_sha256": source_hash, "audit_script_sha256": sha256(__file__),
              "cache_fingerprint": manifest["fingerprint"],
              "cache_manifest_sha256": sha256(args.cache / "manifest.json"),
              "episode_selection": "First N validation episode IDs in each contiguous 100-ID block; deterministic, no performance selection.",
              "episode_ids": [r["episode_id"] for r in records], "checkpoints": []}
    selected_episodes = []
    for record in records:
        path = args.cache / record["path"]
        data = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        assert data["cache_fingerprint"] == manifest["fingerprint"]
        # Copy only observation fields and decision booleans; no action/target data.
        episode = {k: data[k].clone() for k in ("short", "state", "frames", "is_demo", "decision_mask")}
        queries = torch.nonzero(episode["decision_mask"], as_tuple=False).flatten()
        query = int(queries[-1])
        if query < 1:
            raise ValueError("Selected episode has no nonempty causal query")
        episode = {k: v[:query + 1] for k, v in episode.items() if k != "decision_mask"}
        selected_episodes.append((record, episode, query))
    for path in args.checkpoints:
        info = json.loads((path / "checkpoint.json").read_text())
        memory = RecurrentMemoryV7(MemoryV7Config(**info["config"]["memory"]))
        payload = sha256(path / "model.safetensors")
        assert payload == info["metadata"]["payload_sha256"]["model.safetensors"]
        assert info["metadata"]["cache_fingerprint"] == manifest["fingerprint"]
        memory.load_state_dict(load_file(str(path / "model.safetensors"), device="cpu"))
        memory.eval()
        checkpoint = {"path": str(path.resolve()), "step": info["step"], "config": asdict(memory.config),
                      "model_sha256": payload, "episodes": []}
        for record, episode, query in selected_episodes:
            eid = record["episode_id"]
            with torch.no_grad():
                short, all_encoded, no_state_time, encoding = encode_parts(memory, episode)
                encoded = all_encoded[:query]
                generator = torch.Generator().manual_seed(args.seed + eid)
                synthetic = torch.randn(encoded.shape, generator=generator)
                synthetic *= rms(encoded) / rms(synthetic)
                mean = encoded.mean(1, keepdim=True)
                variants = {"real": encoded, "no_state_time": no_state_time[:query],
                            "token_centered": encoded - mean, "contrast_x10": mean + 10 * (encoded - mean),
                            "distinct_synthetic": synthetic, "identical_four_tokens": mean.expand_as(encoded),
                            "fixed_address_queries": encoded, "fixed_multiscale_gate": encoded,
                            "centered_write_keys": encoded, "centered_keys_fixed_queries": encoded}
                row = {"episode_id": eid, "instruction": record["task"], "query_index": query,
                       "query_frame": int(episode["frames"][query]),
                       "demo_writes": int(episode["is_demo"][:query].sum()), "encoding": encoding, "variants": {}}
                for label, sequence in variants.items():
                    bank, trajectory = replay(memory, sequence, variant=label, traced=True)
                    _, read_metrics = memory.read(short[query:query + 1], all_encoded[query:query + 1], bank)
                    row["variants"][label] = {"final": diversity(bank), "trajectory": trajectory,
                                              "read": {k: scalar(v) for k, v in read_metrics.items()}}
                row["perturbation"] = perturbation_audit(memory, encoded, short[query:query + 1],
                                                         all_encoded[query:query + 1], args.seed + eid)
            # One broad-coverage gradient probe per episode is bounded and CPU-only.
            row["gradient"] = probe_gradient(memory, encoded, short[query:query + 1],
                                               all_encoded[query:query + 1], args.seed + eid)
            if eid == records[0]["episode_id"]:
                row["direct_write_gradients"] = direct_write_gradients(memory, encoded[:1], args.seed)
                with torch.no_grad():
                    checkpoint["128_write_retention"] = decay_probe(memory, encoded, args.seed)
            checkpoint["episodes"].append(row)
            print(json.dumps({"checkpoint": info["step"], "episode": eid, "writes": query,
                              "real_relative_spread": row["variants"]["real"]["final"]["relative_spread"],
                              "synthetic_relative_spread": row["variants"]["distinct_synthetic"]["final"]["relative_spread"]}), flush=True)
        checkpoint["variant_summary"] = {
            label: {key: aggregate([e["variants"][label]["final"] for e in checkpoint["episodes"]], key)
                    for key in ("rms", "slot_rms_spread", "relative_spread", "off_diagonal_cosine_fp64")}
            for label in variants}
        result["checkpoints"].append(checkpoint)
    assert sha256(ROOT / "gr00t/long_memory/recurrent_v7.py") == source_hash
    result["elapsed_seconds"] = time.monotonic() - started
    result["source_unchanged"] = True
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_report(args.output, result)
    print(json.dumps({"output": str(args.output.resolve()), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
