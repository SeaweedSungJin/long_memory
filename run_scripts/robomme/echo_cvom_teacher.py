"""Frozen-teacher additive CVoM labels for Echo's all-slot utility predictor.

At storage context t, the known pool is the actual retained prefix bank plus
the newly observed candidate. For each sampled target e, compare J(S) with
J(S+e), where S excludes e and contains at most B-1 known slots. Thus the
intervention ADDS one slot; it neither replaces a fixed victim nor pretends
that the two branches have the same token count.

Both branches use the same later observed query, ground-truth action target,
noise and timestep. No intermediate/future memory writes occur. These labels
are conditional predictive utility under a frozen teacher, not rollout return
or exact Shapley values. Future observations/targets are teacher inputs only;
the predictor feature row is exactly manager.score at the current context.
"""
from __future__ import annotations

import hashlib
import math
import random

import torch


TEACHER_VERSION = "echo_cvom_additive_slot_marginals_v1"


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _seed(seed, *parts):
    message = ":".join(map(str, (TEACHER_VERSION, seed, *parts)))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little") % (2**63 - 1)


def _context_events(eligible, demo, capacity, count, rng):
    """Cover demo/execution phases, preferring pre-/post-capacity contexts.

    The explicitly requested one-context smoke mode is demo-biased when a
    demonstration is eligible; use the default two contexts for phase coverage.
    """
    count = min(count, len(eligible))
    demonstrations = [value for value in eligible if demo[value]]
    execution = [value for value in eligible if not demo[value]]
    if count == 1:
        return [rng.choice(demonstrations or eligible)]
    if demonstrations and execution:
        # A long demonstration may itself span both capacity regimes. Merely
        # drawing an early and late slot can otherwise omit execution entirely.
        early = [value for value in demonstrations if value < capacity]
        late = [value for value in execution if value >= capacity]
        selected = [rng.choice(early or demonstrations), rng.choice(late or execution)]
    else:
        early = [value for value in eligible if value < capacity]
        late = [value for value in eligible if value >= capacity]
        if not early or not late:
            middle = max(1, len(eligible) // 2)
            early, late = eligible[:middle], eligible[middle:]
        selected = [rng.choice(pool) for pool in (early, late) if pool]
    remaining = [value for value in eligible if value not in selected]
    extra = count - len(selected)
    for index in range(extra):
        pool = remaining[index * len(remaining) // extra:(index + 1) * len(remaining) // extra]
        selected.append(rng.choice(pool))
    return sorted(selected)


def build_plan(cache, episodes, tasks, config, *, split, seed, contexts_per_episode=2,
               limit=0, future_samples=2, targets_per_context=4, write_mode="fifo"):
    """Select causal contexts without reading action values or teacher outcomes.

    limit=0 visits every eligible episode in the requested TRAIN/cache-VAL
    split, with up to contexts_per_episode distinct early/late contexts each.
    Two or more contexts cover both demo and execution whenever both phases
    are eligible, preferring a pre-capacity demo and post-capacity execution.
    The optional one-context smoke mode instead prefers an eligible demo.
    A positive limit is a task-balanced upper bound on context rows. Each
    context samples a candidate and, later, uniformly sampled old retained
    slots. All future queries are beyond the current HAMLET short window.
    """
    if split not in ("train", "val"):
        raise ValueError("Only TRAIN/cache-VAL context planning is allowed")
    if write_mode not in ("fifo", "learned"):
        raise ValueError("write_mode must be explicitly fifo or learned")
    if type(seed) is not int or seed < 0 or type(limit) is not int or limit < 0:
        raise ValueError("seed and limit must be nonnegative integers")
    for value, name in ((contexts_per_episode, "contexts_per_episode"), (future_samples, "future_samples"),
                        (targets_per_context, "targets_per_context"), (config.capacity_events, "capacity_events"),
                        (config.short_window, "short_window")):
        _positive_integer(value, name)
    splits = cache.manifest["splits"]
    train, val = [list(map(int, splits[name])) for name in ("train", "val")]
    if len(train) != len(set(train)) or len(val) != len(set(val)) or set(train) & set(val):
        raise ValueError("TRAIN/cache-VAL episode identities must be unique and disjoint")
    grouped = {}
    for eid in sorted(train if split == "train" else val):
        episode = episodes.fetch(eid)
        frames = torch.as_tensor(episode["frames"])
        decisions = torch.as_tensor(episode["decision_mask"])
        if frames.ndim != 1 or decisions.ndim != 1 or decisions.dtype != torch.bool or len(decisions) > len(frames):
            raise ValueError("Require finite ordered observation frames and a boolean decision mask")
        if not bool(torch.isfinite(frames).all()) or not bool((frames[1:] > frames[:-1]).all()):
            raise ValueError("Require finite ordered observation frames")
        queries = torch.where(decisions)[0].tolist()
        eligible = [event for event in range(len(frames))
                    if sum(query > event + config.short_window for query in queries) >= future_samples]
        if not eligible:
            continue
        demo = torch.as_tensor(episode.get("is_demo", torch.zeros(len(frames), dtype=torch.bool)))
        if demo.shape != frames.shape or demo.dtype != torch.bool:
            raise ValueError("Demonstration mask must describe every observed endpoint")
        rng = random.Random(_seed(seed, "plan", split, eid))
        events = _context_events(eligible, demo, config.capacity_events, contexts_per_episode, rng)
        for event in events:
            future_pool = [query for query in queries if query > event + config.short_window]
            future = sorted(rng.sample(future_pool, future_samples))
            row = {"episode_id": eid, "event": event, "future": future, "task": tasks[eid],
                   "split": split, "write_mode": write_mode, "targets_per_context": targets_per_context}
            grouped.setdefault(tasks[eid], []).append(row)
    if not grouped:
        raise ValueError(f"No {split} contexts with enough future queries beyond the short window")
    if limit == 0:
        return sorted((row for rows in grouped.values() for row in rows), key=lambda row: (row["episode_id"], row["event"]))
    rng = random.Random(_seed(seed, "limit", split))
    for rows in grouped.values():
        rng.shuffle(rows)
    selected = []
    while len(selected) < limit and any(grouped.values()):
        for task in sorted(grouped):
            if grouped[task] and len(selected) < limit:
                selected.append(grouped[task].pop())
    return selected


def coalition_indices(pool_size, target, capacity, count, seed):
    """Largest legal subset first; then uniform cardinality, uniform subset.

    The target is absent from every S. Empty coalitions and repeated subsets
    are legal. The full-budget first draw is explicitly part of this sampling
    mixture; it is not an unbiased Shapley estimator.
    """
    for value, name in ((pool_size, "pool_size"), (capacity, "capacity"), (count, "count")):
        _positive_integer(value, name)
    if type(target) is not int or not 0 <= target < pool_size or pool_size > capacity + 1:
        raise ValueError("Target must index the actual bounded bank plus candidate")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    others = [index for index in range(pool_size) if index != target]
    largest = min(capacity - 1, len(others))
    rng = random.Random(seed)
    result = [sorted(rng.sample(others, largest))]
    for _ in range(count - 1):
        result.append(sorted(rng.sample(others, rng.randrange(largest + 1))))
    return result


def additive_pair(pool, target, subset, capacity):
    """Keep slot content/metadata unchanged while adding exactly the target."""
    if (type(target) is not int or not 0 <= target < pool.n_events
            or not isinstance(subset, list) or any(type(index) is not int for index in subset)
            or subset != sorted(set(subset)) or target in subset
            or any(index < 0 or index >= pool.n_events for index in subset)
            or len(subset) >= capacity):
        raise ValueError("Additive coalition must exclude target and contain at most B-1 known slots")
    return pool.select(subset), pool.select(sorted([*subset, target]))


def _summarize(without, with_target, subsets, seeds):
    left = torch.tensor(without, dtype=torch.float64)
    right = torch.tensor(with_target, dtype=torch.float64)
    gains = left - right
    signed = float(gains.mean())
    n_noise = gains.shape[2]
    noise_variance = gains.var(dim=2, unbiased=True).mean() if n_noise > 1 else gains.new_zeros(())
    noise_std = float(noise_variance.sqrt())
    noise_mean_std = float((noise_variance / gains.numel()).sqrt())
    draw_std = float(gains.std(unbiased=True)) if gains.numel() > 1 else 0.
    coalition_means = gains.mean(dim=(1, 2))
    return {"signed_mean": signed, "raw_signed_mean": signed, "positive_utility": max(0., signed),
        "gains": gains.tolist(), "without_losses": without, "with_losses": with_target,
        "coalition_indices": subsets, "noise_seeds": seeds,
        "per_future_gain": gains.mean(dim=(0, 2)).tolist(),
        "noise_variance": float(noise_variance), "noise_std": noise_std, "noise_mean_std": noise_mean_std,
        "draw_std": draw_std,
        "coalition_std": float(coalition_means.std(unbiased=True)) if len(coalition_means) > 1 else 0.,
        "num_draws": gains.numel(),
        "confidence_inputs": {"signed_mean": signed, "draw_std": draw_std,
                              "noise_mean_std": noise_mean_std, "num_draws": gains.numel()},
        "uncertainty_interpretation": "descriptive conditional draw variation and heuristic noise scale; NOT episode confidence interval"}


@torch.no_grad()
def label_contexts(core, head, episodes, rows, *, seed, coalitions=4, noise_samples=2,
                   tail_weight=.25, flow_fn=None, progress=None, teacher_snapshot_version=None):
    """Return every continuous target with exact deployment-context features.

    A learned collection mode replays only observations through t-1 with the
    frozen snapshot's writer. It is explicit in every context. The snapshot
    version should identify the immutable teacher checkpoint for later label
    refreshes; this routine never trains or updates that teacher.
    """
    for name, model in (("core", core), ("head", head)):
        if isinstance(model, torch.nn.Module) and (model.training or any(p.requires_grad for p in model.parameters())):
            raise ValueError(f"Teacher {name} must be explicitly frozen and eval()")
    for value, name in ((coalitions, "coalitions"), (noise_samples, "noise_samples")):
        _positive_integer(value, name)
    if (type(seed) is not int or seed < 0 or isinstance(tail_weight, bool)
            or not isinstance(tail_weight, (int, float)) or not math.isfinite(tail_weight) or not 0 < tail_weight <= 1):
        raise ValueError("Invalid teacher seed or tail weight")
    if teacher_snapshot_version is not None and (not isinstance(teacher_snapshot_version, str) or not teacher_snapshot_version):
        raise ValueError("teacher_snapshot_version must be a nonempty immutable version string")
    real_actor = flow_fn is None
    if real_actor and teacher_snapshot_version is None:
        raise ValueError("Production labels require an immutable teacher_snapshot_version")
    if real_actor:
        from run_scripts.robomme.flow_objective_v19 import episode_flow_v19
        from gr00t.long_memory.cache_reader_v3 import validate_decision
        flow_fn = episode_flow_v19
    cfg, contexts, total_calls = core.config, [], 0
    for number, row in enumerate(rows):
        event, future = row["event"], row["future"]
        mode, target_count = row.get("write_mode", "fifo"), row.get("targets_per_context", 4)
        _positive_integer(target_count, "targets_per_context")
        if mode not in ("fifo", "learned") or row.get("split", "train") not in ("train", "val"):
            raise ValueError("Only explicit FIFO/learned collections from TRAIN/cache-VAL are allowed")
        if mode == "learned" and teacher_snapshot_version is None:
            raise ValueError("Learned collection requires an immutable teacher_snapshot_version")
        episode = episodes.fetch(row["episode_id"])
        if (type(event) is not int or event < 0 or not isinstance(future, list) or not future
                or any(type(query) is not int for query in future) or future != sorted(set(future))
                or any(query <= event + cfg.short_window for query in future)
                or max(future) >= min(len(episode["frames"]), len(episode["decision_mask"]))
                or not all(bool(episode["decision_mask"][query]) for query in future)):
            raise ValueError("Require causal observed candidates and valid future queries beyond the short window")
        if real_actor:
            for query in future:
                validate_decision(episode, query)
        # Construct the predictor's entire input using only [0,t]. Future
        # query selection cannot change even the feature-encoding batch size.
        encoded = core.encode_prefix(episode, event + 1)
        bank = core.initial_state()
        demo = episode.get("is_demo", torch.zeros(len(episode["frames"]), dtype=torch.bool))
        for index in range(event):
            bank, _ = core.manager.update(bank, encoded["stored"][index:index+1], encoded["query"][index:index+1],
                int(episode["frames"][index]), bool(demo[index]), index, mode=mode)
        if not 0 <= bank.n_events <= cfg.capacity_events:
            raise ValueError("Collected bank exceeds the frozen teacher capacity")
        candidate, current_query = encoded["stored"][event:event+1], encoded["query"][event:event+1]
        frame, is_demo = int(episode["frames"][event]), bool(demo[event])
        scores = core.manager.score(bank, candidate, current_query, frame, is_demo)
        features = scores["features"]
        if features.ndim != 2 or features.shape[0] != bank.n_events + 1 or not bool(torch.isfinite(features).all()):
            raise ValueError("manager.score must expose finite per-slot plus candidate deployment features")
        pool = bank.append(candidate, frame, is_demo, event)
        if any(any(identifier > event for identifier in group) for group in pool.event_ids):
            raise ValueError("Future event entered the current teacher pool")
        rng = random.Random(_seed(seed, row["episode_id"], event, "targets"))
        selected = sorted(rng.sample(range(bank.n_events), min(target_count - 1, bank.n_events))) + [bank.n_events]
        future_encoded = core.encode_prefix(episode, max(future) + 1)
        targets, calls = [], 0
        for target in selected:
            subsets = coalition_indices(pool.n_events, target, cfg.capacity_events, coalitions,
                _seed(seed, row["episode_id"], event, target, "coalitions"))
            losses_without, losses_with, all_seeds = [], [], []
            for coalition, subset in enumerate(subsets):
                without_bank, with_bank = additive_pair(pool, target, subset, cfg.capacity_events)
                without_rows, with_rows, seed_rows = [], [], []
                for query in future:
                    short = future_encoded["short"][query:query+1]
                    encoded_query = future_encoded["query"][query:query+1]
                    fused_without, _ = core.read_from_bank(short, encoded_query, without_bank.tokens)
                    fused_with, _ = core.read_from_bank(short, encoded_query, with_bank.tokens)
                    without_noise, with_noise, noise_seeds = [], [], []
                    for noise in range(noise_samples):
                        draw_seed = _seed(seed, row["episode_id"], event, target, coalition, query, noise, "flow")
                        kwargs = dict(seed=draw_seed, tail_weight=tail_weight, activation_checkpointing=False)
                        left = float(flow_fn(head, episode, query, fused_without, **kwargs)["loss"])
                        right = float(flow_fn(head, episode, query, fused_with, **kwargs)["loss"])
                        if not math.isfinite(left) or not math.isfinite(right):
                            raise FloatingPointError("Nonfinite additive CVoM paired action loss")
                        without_noise.append(left)
                        with_noise.append(right)
                        noise_seeds.append(draw_seed)
                        calls += 2
                    without_rows.append(without_noise)
                    with_rows.append(with_noise)
                    seed_rows.append(noise_seeds)
                losses_without.append(without_rows)
                losses_with.append(with_rows)
                all_seeds.append(seed_rows)
            summary = _summarize(losses_without, losses_with, subsets, all_seeds)
            targets.append({"index": target, "event_id": pool.event_ids[target][-1],
                "event_ids": list(pool.event_ids[target]), "is_new": target == bank.n_events,
                "features": features[target].detach().cpu().tolist(), **summary})
        context = {**row, "write_mode": mode, "bank_event_ids": [list(group) for group in bank.event_ids],
            "candidate_event_id": event, "targets": targets, "actual_actor_calls": calls,
            "teacher_snapshot_version": teacher_snapshot_version}
        contexts.append(context)
        total_calls += calls
        if progress is not None:
            progress(number + 1, len(rows), context)
    return {"version": TEACHER_VERSION, "contexts": contexts, "actual_actor_calls": total_calls,
        "settings": {"seed": seed, "coalitions": coalitions, "noise_samples": noise_samples,
            "tail_weight": tail_weight, "capacity_events": cfg.capacity_events, "short_window": cfg.short_window,
            "teacher_snapshot_version": teacher_snapshot_version,
            "write_modes": sorted({row["write_mode"] for row in contexts}),
            "conditional_bank": "snapshot_at_current_context; NO future or intermediate writes",
            "comparison": "J(S)-J(S+target); additive one-slot intervention; no fixed victim",
            "coalition_distribution": "first maximum budget <=B-1; remaining uniform cardinality 0..min(B-1,others), uniform subset",
            "target_distribution": "new candidate plus uniform old retained slots without replacement",
            "feature_context": "exact manager.score(actual retained bank,current candidate,current query,frame,demo)",
            "utility_target": "all signed means retained; positive utility=max(0,mean(gains)) only AFTER averaging",
            "interpretation": "frozen-teacher conditional future action utility; NOT rollout return, success, or exact Shapley",
            "uncertainty": "descriptive draw spread and heuristic conditional noise scale; NOT episode confidence interval"}}
