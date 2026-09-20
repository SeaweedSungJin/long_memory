#!/usr/bin/env python3
"""Read-only CPU V8 content/attention interventions, not robot performance.

Loads only the small memory model. No Action Expert, action target, generated
action, simulator, or success label enters the probe. See
docs/LONG_MEMORY_V8_RETRIEVAL_PROBE.md for precise intervention definitions.
"""

from contextlib import contextmanager
from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.checkpoint_v8 import v8_checkpoint_info
from gr00t.long_memory.event_v8 import EventMemoryV8, MemoryV8Config
from gr00t.long_memory.hamlet import validate_cache_checkpoint
from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.replay_v8 import encode_at, replay_state
from gr00t.long_memory.safety_v5 import validate_output_scope


def content_intervention(bank, num_tokens, mode="mean", generator=None):
    """Change only content; preserve every raw-frame/demo metadata entry."""
    if bank.ndim != 3 or bank.shape[-1] < 3 or bank.shape[1] < 1:
        raise ValueError("Require a nonempty [B,N,d+2] bank")
    if type(num_tokens) is not int or num_tokens < 1 or bank.shape[1] % num_tokens:
        raise ValueError("Bank must contain complete events")
    if mode not in ("mean", "shuffle"):
        raise ValueError("mode must be mean or shuffle")
    changed = bank.clone()
    if mode == "mean":
        changed[..., :-2] = bank[..., :-2].mean(1, keepdim=True)
    else:
        events = bank.shape[1] // num_tokens
        order = torch.randperm(events, generator=generator, device="cpu")
        if events > 1 and torch.equal(order, torch.arange(events)):
            order = order.roll(1)
        content = bank[..., :-2].reshape(bank.shape[0], events, num_tokens, -1)
        changed[..., :-2] = content[:, order.to(bank.device)].reshape(bank.shape[0], bank.shape[1], -1)
    return changed


@contextmanager
def _capture_attention(memory, captured, uniform):
    handles = []

    def hook(module, args, output):
        query, _, values = args[:3]
        if uniform:
            width = module.embed_dim
            if module.in_proj_weight is None or not module.batch_first or module.dropout:
                raise ValueError("Uniform intervention requires equal-width, batch-first zero-dropout MHA")
            value_bias = module.in_proj_bias[2 * width:] if module.in_proj_bias is not None else None
            projected = F.linear(values, module.in_proj_weight[2 * width:], value_bias)
            mean = projected.mean(1, keepdim=True).expand(-1, query.shape[1], -1)
            recalled = F.linear(mean, module.out_proj.weight, module.out_proj.bias)
            weights = query.new_full((query.shape[0], module.num_heads, query.shape[1], values.shape[1]),
                                     1.0 / values.shape[1])
            output = recalled, weights
        captured.append(output[1].detach().clone())
        return output

    try:
        for block in memory.read_blocks:
            handles.append(block.attention.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def captured_read(memory, short, encoded, bank, uniform=False):
    if memory.training or any(parameter.device.type != "cpu" for parameter in memory.parameters()):
        raise ValueError("This diagnostic requires an eval-mode CPU memory model")
    if type(uniform) is not bool:
        raise ValueError("uniform must be boolean")
    captured = []
    with _capture_attention(memory, captured, uniform):
        fused, metrics = memory.read(short, encoded, bank, mode="event")
    return fused, metrics, captured


def _event_weights(weights, tokens):
    return weights.reshape(*weights.shape[:-1], -1, tokens).sum(-1)


def _attention_summary(attentions, tokens):
    result = []
    for weights in attentions:
        event = _event_weights(weights, tokens)
        count = event.shape[-1]
        entropy = -(event * event.clamp_min(1e-12).log()).sum(-1)
        result.append(dict(event_entropy=float(entropy.mean()),
            normalized_event_entropy=float(entropy.mean() / math.log(count)) if count > 1 else 0.,
            effective_event_fraction=float(entropy.exp().mean() / count),
            total_variation_from_uniform=float((event - 1. / count).abs().sum(-1).mean() * .5),
            max_event_mass=float(event.amax(-1).mean()),
            mean_event_mass=event.mean(tuple(range(event.ndim - 1))).tolist()))
    return result


def _attention_tv(left, right, tokens):
    if len(left) != len(right) or not left:
        raise ValueError("Matched nonempty attention blocks are required")
    return float(torch.stack([(_event_weights(a, tokens) - _event_weights(b, tokens)).abs().sum(-1).mean() * .5
                              for a, b in zip(left, right)]).mean())


def _diversity(content):
    spread = (content - content.mean(1, keepdim=True)).square().mean().sqrt()
    rms = content.square().mean().sqrt()
    unit = F.normalize(content, dim=-1)
    count = content.shape[1]
    cosine = ((unit.sum(1).square().sum(-1) - unit.square().sum((1, 2)))
              / (count * (count - 1))).mean() if count > 1 else content.new_zeros(())
    return dict(relative_spread=float(spread / (rms + 1e-12)), mean_pair_cosine=float(cosine),
                rms=float(rms))


@torch.no_grad()
def probe_query(memory, episode, decision, seed=8481):
    """One causal query. Required episode fields are observations only."""
    if type(decision) is not int or decision < 2:
        raise ValueError("Choose a query with at least two past observations")
    if memory.training:
        raise ValueError("Memory must be eval()")
    bank = replay_state(memory, episode, decision, mode="event", checkpoint_segment=0)
    current = encode_at(memory, episode, decision)
    short = episode["short"][decision:decision + 1].float()
    q = memory.config.num_short_tokens
    events = bank.shape[1] // q
    if events < 2:
        raise ValueError("Probe requires at least two retained events")
    mean_bank = content_intervention(bank, q, "mean")
    shuffled_bank = content_intervention(bank, q, "shuffle", torch.Generator().manual_seed(seed))
    full, metrics, full_weights = captured_read(memory, short, current, bank)
    uniform, _, uniform_weights = captured_read(memory, short, current, bank, uniform=True)
    mean, _, mean_weights = captured_read(memory, short, current, mean_bank)
    shuffled, _, shuffled_weights = captured_read(memory, short, current, shuffled_bank)

    # Hold bank/time/demo constant; substitute actual, already-observed source
    # and short content from the oldest retained observation as a probe query.
    # This is a controlled query-content sensitivity test, not an alternate
    # physically valid execution state or a ground-truth relevant query.
    alternate = bank[:, :q].clone()
    alternate[..., -2:] = current[..., -2:]
    alternate_short = episode["short"][decision - events:decision - events + 1].float()
    _, _, alternate_weights = captured_read(memory, alternate_short, alternate, bank)
    content = bank[..., :-2]
    diversity = _diversity(content)
    event_diversity = _diversity(content.reshape(1, events, q, -1).mean(2))
    residual = full - short
    residual_norm = residual.norm()
    short_norm = short.norm()
    result = dict(decision=decision, raw_frame=int(episode["frames"][decision]), retained_events=events,
        oldest_retained_frame=int(bank[0, 0, -2]),
        retained_demo_events=int(bank[0, ::q, -1].sum()),
        content_relative_spread=diversity["relative_spread"], content_pair_cosine=diversity["mean_pair_cosine"],
        event_mean_relative_spread=event_diversity["relative_spread"],
        event_mean_pair_cosine=event_diversity["mean_pair_cosine"],
        residual_over_short_norm=float(residual_norm / short_norm.clamp_min(1e-12)),
        residual_rms=float(residual.square().mean().sqrt()),
        gate_mean=float(metrics["gate_mean"]),
        uniform_fused_delta_over_residual=float((full - uniform).norm() / residual_norm.clamp_min(1e-12)),
        mean_fused_delta_over_residual=float((full - mean).norm() / residual_norm.clamp_min(1e-12)),
        shuffled_fused_delta_over_residual=float((full - shuffled).norm() / residual_norm.clamp_min(1e-12)),
        query_attention_tv=_attention_tv(full_weights, alternate_weights, q),
        mean_attention_tv=_attention_tv(full_weights, mean_weights, q),
        shuffled_attention_tv=_attention_tv(full_weights, shuffled_weights, q),
        bf16_residual_changed_fraction=float((full.bfloat16() != short.bfloat16()).float().mean()),
        bf16_uniform_changed_fraction=float((full.bfloat16() != uniform.bfloat16()).float().mean()),
        bf16_mean_changed_fraction=float((full.bfloat16() != mean.bfloat16()).float().mean()),
        metadata_preserved=bool(torch.equal(bank[..., -2:], mean_bank[..., -2:])
                                and torch.equal(bank[..., -2:], shuffled_bank[..., -2:])),
        attention={name: _attention_summary(weights, q) for name, weights in (
            ("full", full_weights), ("uniform", uniform_weights), ("mean_content", mean_weights),
            ("shuffled_content", shuffled_weights), ("alternate_query_content", alternate_weights))})
    return result


def _load_observation_episode(cache, record):
    """Mmap payload; expose only the small observation fields to the probe."""
    path = (Path(cache.path) / record["path"]).resolve()
    if not path.is_relative_to(Path(cache.path).resolve()):
        raise ValueError("Episode path escapes cache")
    raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if raw.get("cache_fingerprint") != cache.manifest["fingerprint"] or int(raw["episode_id"]) != int(record["episode_id"]):
        raise ValueError("Episode cache identity mismatch")
    observations = {name: raw[name] for name in ("moment", "short", "state", "frames", "is_demo")}
    # This mask selects an observation at which policy control is active. No
    # target value, success flag or action is inspected or supplied to memory.
    decisions = torch.where(raw["decision_mask"])[0]
    decisions = decisions[decisions >= 2]
    return observations, decisions, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--seed", type=int, default=8481)
    args = parser.parse_args(argv)
    if not 1 <= args.queries <= 16 or args.seed < 0:
        raise ValueError("Require 1..16 queries and a nonnegative seed")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    base = cache.manifest["model_path"]
    checkpoint = Path(args.checkpoint).resolve()
    info = v8_checkpoint_info(base, checkpoint, expected_stage=1)
    if info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"] or info["config"]["mode"] != "event":
        raise ValueError("Require an event-mode V8 checkpoint from this cache")
    output = validate_output_scope(args.output_dir, cache.path, cache.manifest.get("dataset_path"), base, checkpoint)
    if output.exists():
        raise FileExistsError("Use a new probe output directory")
    with torch.random.fork_rng(devices=[]):
        memory = EventMemoryV8(MemoryV8Config(**info["config"]["memory"]))
    memory.load_state_dict(load_file(str(checkpoint / "model.safetensors"), device="cpu"), strict=True)
    memory.eval().requires_grad_(False)
    before = {name: value.detach().clone() for name, value in memory.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    records = {int(record["episode_id"]): record for record in cache.manifest["episodes"]}
    eligible = [records[int(eid)] for eid in cache.manifest["splits"]["val"]]
    random.Random(args.seed).shuffle(eligible)
    # Prefer different instruction groups, then fill any remaining budget.
    seen, first, remaining = set(), [], []
    for record in eligible:
        if record["task"] in seen:
            remaining.append(record)
        else:
            first.append(record)
            seen.add(record["task"])
    result, signatures = [], []
    for record in first + remaining:
        episode, decisions, path = _load_observation_episode(cache, record)
        if not len(decisions):
            continue
        # Fixed middle eligible decision provides nonempty history without
        # selecting on attention, action loss, or task outcomes.
        decision = int(decisions[len(decisions) // 2])
        stat = path.stat()
        row = probe_query(memory, episode, decision, seed=args.seed + int(record["episode_id"]))
        row.update(episode_id=int(record["episode_id"]), instruction=record["task"])
        result.append(row)
        signatures.append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
        print(f"[retrieval-probe] {len(result)}/{args.queries} episode={record['episode_id']} events={row['retained_events']} "
              f"query_tv={row['query_attention_tv']:.5f} mean_delta={row['mean_fused_delta_over_residual']:.5f}", flush=True)
        if len(result) == args.queries:
            break
    if not result:
        raise ValueError("No eligible validation queries")
    if any(not torch.equal(before[name], value) for name, value in memory.state_dict().items()) or not torch.equal(rng_before, torch.get_rng_state()):
        raise RuntimeError("Read-only diagnostic changed memory weights or global RNG")
    source_files = [Path(__file__).resolve(), ROOT / "gr00t/long_memory/event_v8.py", ROOT / "gr00t/long_memory/replay_v8.py"]
    numeric = [name for name, value in result[0].items() if isinstance(value, (float, int)) and not isinstance(value, bool)
               and name not in ("episode_id", "decision", "raw_frame", "oldest_retained_frame")]
    summary = {name: dict(mean=statistics.mean(row[name] for row in result),
                         median=statistics.median(row[name] for row in result),
                         minimum=min(row[name] for row in result), maximum=max(row[name] for row in result)) for name in numeric}
    summary["normalized_event_entropy"] = {
        f"block{block}": statistics.median(row["attention"]["full"][block]["normalized_event_entropy"] for row in result)
        for block in range(len(memory.read_blocks))}
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "manifest.json", dict(args=vars(args), checkpoint=str(checkpoint), step=info["step"],
        checkpoint_metadata=info["metadata"], cache_fingerprint=cache.manifest["fingerprint"], memory=asdict(memory.config),
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        cache_episode_signatures=signatures, num_queries=len(result), device="cpu", threads=2,
        note="One selected query per held-out episode; instruction coverage is not guaranteed benchmark-task coverage. "
             "Content/attention/conditioning diagnostics only, not cue correctness, action quality or robot accuracy."))
    _atomic_json(output / "records.json", result)
    _atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
