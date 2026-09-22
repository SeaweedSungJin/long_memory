"""Paired conditional memory-content utility on a frozen V19 actor.

The candidate always replaces the same FIFO victim. For each coalition S of
the other past events, compare [oldest]+S against S+[candidate]. Each pair has
the same token budget, preserves original chronological order, and uses the
same future observation, GT action, noise, and timestep. No intermediate/future
event is ever written to either branch. This measures CONDITIONAL information
utility at held-out future queries, NOT rollout return or FIFO continuation.

The single-context control repeats the full coalition with the same number of
noise draws/actor calls. The coalitional arm changes only the non-victim subset.
Full-coalition replicate zero is identical between arms. This estimator is not
claimed to be an exact Shapley value, causal rollout return, or success label.
"""
from __future__ import annotations

import hashlib
import math
import random

import torch

from run_scripts.robomme.cvom_admission_core import AdmissionConfig, admission_features


TEACHER_VERSION = "cvom_conditional_replacement_coalitions_v1"


def build_context_plan(cache, episodes, tasks, capacity, count, seed, *, split="train",
                       short_window=4, future_samples=2):
    """Only TRAIN/cache-VAL; select before observing teacher gains/outcomes.

    count=0 covers every eligible episode once (one seeded overflowing event
    per episode). Positive counts select at most count task-balanced episodes,
    never multiple nearby contexts from one episode. All
    future queries are beyond the short window of the storage decision, not
    just after it, and the planner never uses benchmark TEST or success labels.
    """
    if split not in ("train", "val"):
        raise ValueError("Only TRAIN/cache-VAL context planning is allowed")
    if type(capacity) is not int or capacity <= 0 or type(count) is not int or count < 0:
        raise ValueError("capacity must be positive and count nonnegative")
    if type(short_window) is not int or short_window < 1 or type(future_samples) is not int or future_samples < 1:
        raise ValueError("short_window and future_samples must be positive")
    rng, groups = random.Random(seed), {}
    for eid in sorted(int(value) for value in cache.manifest["splits"][split]):
        episode = episodes.fetch(eid)
        decisions = torch.where(torch.as_tensor(episode["decision_mask"], dtype=torch.bool))[0].tolist()
        eligible = [event for event in range(capacity, max(decisions, default=0) - short_window)
                    if sum(query > event + short_window for query in decisions) >= future_samples]
        if not eligible:
            continue
        events = [rng.choice(eligible)]
        for event in events:
            futures = [query for query in decisions if query > event + short_window]
            # Keep the first out-of-short-window query and sample the remainder
            # from the rest; deterministic and chosen without action losses.
            chosen = [futures[0]]
            if future_samples > 1:
                chosen += rng.sample(futures[1:], future_samples - 1)
            row = {"episode_id": eid, "event": event, "future": sorted(chosen), "task": tasks[eid]}
            groups.setdefault(tasks[eid], []).append(row)
    if not groups:
        raise ValueError(f"No {split} full FIFO contexts with a future beyond the short window")
    if count == 0:
        return sorted((row for rows in groups.values() for row in rows), key=lambda row: row["episode_id"])
    for rows in groups.values():
        rng.shuffle(rows)
    selected = []
    while len(selected) < count and any(groups.values()):
        for task in sorted(groups):
            if groups[task] and len(selected) < count:
                selected.append(groups[task].pop())
    return selected


def coalition_indices(capacity, count, seed):
    """Full coalition first, then seeded uniform-sized subsets in time order.

    Entries index the original full bank; index zero is always the victim and
    is never sampled into S. Uniform subset cardinality is an explicit design
    choice, not a proof of Shapley attribution. Capacity one is degenerate.
    """
    if type(capacity) is not int or capacity <= 0 or type(count) is not int or count <= 0:
        raise ValueError("capacity and coalition count must be positive integers")
    rng, others = random.Random(seed), list(range(1, capacity))
    result = [others]
    for _ in range(count - 1):
        # Exclude the already evaluated full subset whenever alternatives
        # exist. Empty S remains legal and has one event on each side.
        n = rng.randrange(len(others)) if others else 0
        result.append(sorted(rng.sample(others, n)))
    return result


def replacement_pair(bank, candidate, subset, num_tokens):
    """Equal-sized chronologically ordered KEEP and fixed-victim INSERT bank."""
    if (bank.ndim != 3 or candidate.ndim != 3 or bank.shape[0] != 1 or candidate.shape[0] != 1
            or candidate.shape[1] != num_tokens or bank.shape[1] % num_tokens
            or bank.shape[1] == 0 or bank.shape[2] != candidate.shape[2]):
        raise ValueError("replacement_pair requires a nonempty whole-event bank and one candidate")
    n = bank.shape[1] // num_tokens
    if (not isinstance(subset, list) or any(type(index) is not int for index in subset)
            or subset != sorted(set(subset)) or any(index < 1 or index >= n for index in subset)):
        raise ValueError("coalition must contain distinct increasing non-victim past indices")
    retained = torch.cat([bank[:, index*num_tokens:(index+1)*num_tokens] for index in subset], dim=1) if subset else bank[:, :0]
    return torch.cat((bank[:, :num_tokens], retained), dim=1), torch.cat((retained, candidate), dim=1)


def _seed(seed, episode_id, event, coalition, future, noise):
    value = f"{TEACHER_VERSION}:{seed}:{episode_id}:{event}:{coalition}:{future}:{noise}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little") % (2**63 - 1)


def _summarize(keep, replace, subsets):
    kept, inserted = torch.tensor(keep, dtype=torch.float64), torch.tensor(replace, dtype=torch.float64)
    gains = kept - inserted  # Improvement always has POSITIVE sign.
    # Noise variation is within each fixed coalition/future pair. Coalition
    # variation is among their averaged gains. Neither is an episode-level CI.
    n_noise = gains.shape[2]
    noise_variance = gains.var(dim=2, unbiased=True).mean() if n_noise > 1 else gains.new_zeros(())
    # Conditional SE from matched noise only, averaging F and C measurements;
    # correlation of queries/context is not represented. Do not call it a CI.
    conditional_noise_mean_std = (noise_variance / gains.numel()).sqrt()
    coalition_means = gains.mean(dim=(1, 2))
    return {"signed_mean": float(gains.mean()), "gains": gains.tolist(),
            "keep_losses": keep, "replace_losses": replace, "coalition_indices": subsets,
            "per_future_gain": gains.mean(dim=(0, 2)).tolist(),
            "noise_variance": float(noise_variance), "noise_std": float(noise_variance.sqrt()),
            "noise_mean_std": float(conditional_noise_mean_std),
            "coalition_std": float(coalition_means.std(unbiased=True)) if len(coalition_means) > 1 else 0.,
            "uncertainty_interpretation": "descriptive conditional Monte Carlo variation; NOT episode confidence interval"}


@torch.no_grad()
def label_contexts(core, head, episodes, contexts, *, seed, coalitions=4, noise_samples=2,
                   tail_weight=.25, flow_fn=None, progress=None):
    """JSON-serializable paired labels and causal inputs; never updates actor.

    Optional flow_fn is a synthetic-test hook with the episode_flow_v19 API.
    Real callers should omit it. The actor must already be frozen/eval; this
    function refuses live trainable teacher parameters rather than altering
    their settings behind the caller's back.
    """
    for name, model in (("core", core), ("head", head)):
        if isinstance(model, torch.nn.Module):
            if model.training or any(parameter.requires_grad for parameter in model.parameters()):
                raise ValueError(f"Teacher {name} must be explicitly frozen and eval()")
    if type(coalitions) is not int or coalitions <= 0 or type(noise_samples) is not int or noise_samples <= 0:
        raise ValueError("coalitions and noise_samples must be positive")
    if type(seed) is not int or seed < 0 or not math.isfinite(tail_weight) or not 0 < tail_weight <= 1:
        raise ValueError("Invalid seed or tail weight")
    real_actor = flow_fn is None
    if real_actor:
        from run_scripts.robomme.flow_objective_v19 import episode_flow_v19
        from gr00t.long_memory.cache_reader_v3 import validate_decision
        flow_fn = episode_flow_v19
    cfg = core.config
    admission = AdmissionConfig(dim=cfg.hidden_dim, num_tokens=cfg.num_short_tokens,
                                capacity_events=cfg.capacity_events)
    labels = []
    for number, row in enumerate(contexts):
        event, future = row["event"], row["future"]
        ep = episodes.fetch(row["episode_id"])
        if (type(event) is not int or event < cfg.capacity_events or not future
                or future != sorted(set(future)) or any(type(query) is not int for query in future)
                or any(query <= event + cfg.short_window for query in future)
                or max(future) >= min(len(ep["frames"]), len(ep["decision_mask"]))
                or not all(bool(ep["decision_mask"][query]) for query in future)):
            raise ValueError("Teacher needs a full causal FIFO bank and valid queries beyond the short window")
        if real_actor:
            # MappedEpisodes intentionally avoids scanning large VL/target
            # tensors. Validate only the actual teacher queries, including
            # conditioning-tail equality, before evaluating a causal pair.
            for query in future:
                validate_decision(ep, query)
        encoded = core.encode_prefix(ep, max(future) + 1)
        # Exact bank after FIFO writes [0,event); no current/future event here.
        bank = encoded["stored"][event-cfg.capacity_events:event].reshape(1, -1, cfg.hidden_dim)
        candidate = encoded["stored"][event:event+1]
        features = admission_features(bank, candidate, admission)[0].cpu().tolist()
        coalition_seed = _seed(seed, row["episode_id"], event, 0, 0, 0)
        sampled = coalition_indices(cfg.capacity_events, coalitions, coalition_seed)
        arms = {"single": [sampled[0] for _ in range(coalitions)], "coalitional": sampled}
        output, shared_first = {}, None
        actor_calls = 0
        for arm, subsets in arms.items():
            keep_losses, replace_losses, seeds = [], [], []
            for c_index, subset in enumerate(subsets):
                if arm == "coalitional" and c_index == 0:
                    # Both arms intentionally use the exact same first full
                    # coalition measurements, not just equal-looking settings.
                    keep, replace, used_seeds = shared_first
                else:
                    keep_bank, replace_bank = replacement_pair(bank, candidate, subset, cfg.num_short_tokens)
                    keep, replace, used_seeds = [], [], []
                    for query in future:
                        short, encoded_query = encoded["short"][query:query+1], encoded["query"][query:query+1]
                        fused_keep, _ = core.read_from_bank(short, encoded_query, keep_bank)
                        fused_replace, _ = core.read_from_bank(short, encoded_query, replace_bank)
                        keep_row, replace_row, seed_row = [], [], []
                        for repeat in range(noise_samples):
                            noise_seed = _seed(seed, row["episode_id"], event, c_index, query, repeat)
                            common = dict(seed=noise_seed, tail_weight=tail_weight, activation_checkpointing=False)
                            left = float(flow_fn(head, ep, query, fused_keep, **common)["loss"])
                            right = float(flow_fn(head, ep, query, fused_replace, **common)["loss"])
                            if not math.isfinite(left) or not math.isfinite(right):
                                raise FloatingPointError("Nonfinite CVoM paired action loss")
                            keep_row.append(left); replace_row.append(right); seed_row.append(noise_seed)
                            actor_calls += 2
                        keep.append(keep_row); replace.append(replace_row); used_seeds.append(seed_row)
                    if arm == "single" and c_index == 0:
                        shared_first = (keep, replace, used_seeds)
                keep_losses.append(keep); replace_losses.append(replace); seeds.append(used_seeds)
            output[arm] = _summarize(keep_losses, replace_losses, subsets)
            output[arm]["noise_seeds"] = seeds
        labels.append({**row, "bank_event_ids": list(range(event-cfg.capacity_events, event)),
            "victim_event_id": event-cfg.capacity_events, "candidate_event_id": event,
            "features": features, "labels": output, "actual_actor_calls": actor_calls,
            "logical_actor_calls_per_arm": 2*coalitions*len(future)*noise_samples})
        if progress is not None:
            progress(number+1, len(contexts), labels[-1])
    return {"version": TEACHER_VERSION, "contexts": labels, "settings": {
        "seed": seed, "coalitions": coalitions, "noise_samples": noise_samples, "tail_weight": tail_weight,
        "capacity_events": cfg.capacity_events, "short_window": cfg.short_window,
        "conditional_bank": "snapshot_at_candidate; NO future or intermediate writes",
        "comparison": "[oldest]+S vs S+[candidate], same event/token count and chronological order",
        "coalition_distribution": "first full; remaining uniform non-full cardinality and uniform subset",
        "control": "repeated full coalition; equal logical noise/actor-call budget; first full shared",
        "interpretation": "fixed-teacher conditional future action utility; NOT rollout return, success, or exact Shapley",
        "uncertainty": "descriptive within-context Monte Carlo spread; NOT episode-paired CI"}}
