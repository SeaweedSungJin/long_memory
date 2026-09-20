"""Signed future WRITE utility on recorded trajectories (NOT simulator reward).

Stage 2 compares KEEP and UPDATE from exactly the same causal prefix, then
replays every intervening observation under one frozen policy snapshot. Noise
is paired between branches. Independent audit draws test whether apparently
beneficial KEEP decisions survive resampling, avoiding a noisy-minimum claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch
from torch.nn import functional as F

from .cache_reader_v3 import validate_decision
from .expert_v4 import expert_episode_flow_loss
from .replay_v7 import apply_write, encode_at, replay_state


@dataclass(frozen=True)
class LabelV7Config:
    future_samples: int = 4
    noise_samples: int = 4
    label_scale: float = 0.001
    threshold: float = 0.05
    memory_window: int = 4

    def __post_init__(self):
        for name in ("future_samples", "noise_samples", "memory_window"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"V7 {name} must be a positive integer")
        if self.noise_samples < 2:
            raise ValueError("V7 utility labels require at least two paired noise draws")
        for name in ("label_scale", "threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"V7 {name} must be finite")
        if self.label_scale <= 0 or self.threshold < 0:
            raise ValueError("V7 scale must be positive and threshold nonnegative")


def _flow(head, ep, decision, fused, *, seed):
    validate_decision(ep, decision)
    return expert_episode_flow_loss(head, ep, decision, fused, seed=seed)


def make_storage_contexts(episodes, episode_ids, count, seed, future_samples=4, memory_window=4):
    """Round-robin shuffled EPISODES, then writes, rather than one long episode.

The caller supplies disjoint train/val episode sets. Demo writes are eligible;
the last write with no future GT action is not. Indices are sampling metadata,
never fed to the actor/CVOM as future observations or actions.
"""
    if count < 1 or future_samples < 1 or memory_window < 1:
        raise ValueError("Positive storage count, future_samples and memory_window required")
    rng = random.Random(seed)
    choices = []
    for eid in sorted(set(map(int, episode_ids))):
        ep = episodes.fetch(eid)
        valid = torch.where(ep["decision_mask"])[0].tolist()
        if not valid or valid[-1] <= 0:
            continue
        indices = list(range(valid[-1]))
        rng.shuffle(indices)
        choices.append([eid, valid, indices])
    if not choices:
        raise ValueError("No causal WRITE with a later valid action query in this episode split")
    rng.shuffle(choices)
    result = []
    while choices and len(result) < count:
        remaining = []
        for eid, valid, indices in choices:
            if len(result) >= count:
                break
            index = indices.pop()
            futures = [q for q in valid if q > index]
            # Stratification covers nearby and far future without inventing
            # distant queries for short episodes. Each stratum supplies one.
            k = min(len(futures), future_samples)
            selected = [futures[rng.randrange(j * len(futures) // k,
                        (j + 1) * len(futures) // k)] for j in range(k)]
            # A query at least K cached observations later has an entirely
            # newer original K-window. Reserve such a query when it exists.
            outside = [q for q in futures if q - index >= memory_window]
            if outside and not any(q - index >= memory_window for q in selected):
                selected[-1] = rng.choice(outside)
            selected.sort()
            result.append(dict(episode_id=eid, write_index=index, future_queries=selected,
                               seed=rng.randrange(2**31), memory_window=memory_window))
            if indices:
                remaining.append([eid, valid, indices])
        choices = remaining
    return result


def _seed(context, query, draw, audit):
    return (int(context["seed"]) + 104729 * int(query) + 1009 * draw
            + (1_000_000_007 if audit else 0)) % (2**63 - 1)


def _future_losses(memory, teacher, head, ep, context, initial, cfg, audit):
    state = initial.clone()  # branches must not alias mutable state
    queries = sorted(context["future_queries"])
    wanted = set(queries)
    losses = {}
    for index in range(int(context["write_index"]) + 1, queries[-1] + 1):
        x = encode_at(memory, ep, index)
        if index in wanted:
            short = ep["short"][index].to(device=x.device, dtype=torch.float32)[None]
            fused, _ = memory.read(short, x, state, mode="recurrent")
            row = []
            for draw in range(cfg.noise_samples):
                loss = _flow(head, ep, index, fused, seed=_seed(context, index, draw, audit))["loss"]
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise FloatingPointError("Nonfinite V7 counterfactual action loss")
                row.append(value)
            losses[index] = row
        if index < queries[-1]:
            state, _ = apply_write(memory, state, x, mode="recurrent", cvom=teacher,
                                    threshold=cfg.threshold)
    return torch.tensor([losses[q] for q in queries], dtype=torch.float64)


@torch.no_grad()
def build_storage_label(memory, teacher_cvom, head, episode, context, config, *, audit=False):
    """Generate detached state/target tensors from a FROZEN Stage-1 actor.

``audit=True`` adds a second independent set of paired noise draws. It doubles
this context's Expert forwards; training-label generation normally omits it.
The teacher controls BOTH prefix and future writes and stays fixed this round.
"""
    if any(p.requires_grad for p in memory.parameters()) or any(p.requires_grad for p in head.parameters()):
        raise ValueError("V7 Stage-2 labels require frozen memory AND adapted Action Expert")
    if teacher_cvom is not None and any(p.requires_grad for p in teacher_cvom.parameters()):
        raise ValueError("V7 label teacher must be a frozen CVOM snapshot")
    i = int(context["write_index"])
    queries = context["future_queries"]
    if context.get("memory_window", config.memory_window) != config.memory_window:
        raise ValueError("V7 context original short-window setting differs from label config")
    if (int(context["episode_id"]) != int(episode["episode_id"])
            or not 0 <= i < len(episode["frames"]) or not queries
            or len(queries) > config.future_samples or len(set(queries)) != len(queries)):
        raise ValueError("Invalid V7 storage context")
    for query in queries:
        if type(query) is not int or query <= i:
            raise ValueError("V7 WRITE utility requires strictly future action queries")
        validate_decision(episode, query)
    state = replay_state(memory, episode, stop_before=i, mode="recurrent", checkpoint_segment=0,
                         cvom=teacher_cvom, threshold=config.threshold)
    x = encode_at(memory, episode, i)
    candidate, _ = memory.write(state, x)
    keep = _future_losses(memory, teacher_cvom, head, episode, context, state, config, False)
    update = _future_losses(memory, teacher_cvom, head, episode, context, candidate, config, False)
    differences = keep - update
    gain = float(differences.mean())
    draw_means = differences.mean(0)
    standard_error = float(draw_means.std(unbiased=True) / math.sqrt(config.noise_samples))
    metrics = dict(raw_gain=gain, gain_noise_se=standard_error,
                   keep_action_loss=float(keep.mean()), update_action_loss=float(update.mean()),
                   keep_opportunity=float(gain < 0), selected_noise_keep_gain=max(-gain, 0.0),
                   future_queries=float(len(queries)),
                   outside_short_window_fraction=sum(q - i >= config.memory_window for q in queries) / len(queries),
                   expert_forwards=float(2 * len(queries) * config.noise_samples * (2 if audit else 1)))
    audit_gain = None
    if audit:
        keep_audit = _future_losses(memory, teacher_cvom, head, episode, context, state, config, True)
        update_audit = _future_losses(memory, teacher_cvom, head, episode, context, candidate, config, True)
        audit_gain = float((keep_audit - update_audit).mean())
        metrics.update(audit_raw_gain=audit_gain,
                       independent_noise_selected_gain=-audit_gain if gain < 0 else 0.0,
                       sign_stable=float((gain < 0) == (audit_gain < 0)))
    return dict(x=x.detach().cpu(), state=state.detach().cpu(), candidate=candidate.detach().cpu(),
                target=torch.tensor([gain / config.label_scale], dtype=torch.float32),
                raw_gain=gain, audit_raw_gain=audit_gain, metrics=metrics,
                context=dict(context))


def storage_loss(cvom, memory, label, config):
    """Regression remains meaningful for zero-valued targets; do not skip it."""
    device = next(cvom.parameters()).device
    args = [label[k].to(device=device, dtype=torch.float32).detach() for k in ("x", "state", "candidate")]
    prediction = cvom(*args, memory.slot_addresses.detach()).reshape(-1)
    target = label["target"].to(device=device, dtype=torch.float32).reshape(-1).detach()
    if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("Invalid CVOM prediction shape or nonfinite value")
    loss = F.smooth_l1_loss(prediction, target)
    raw_prediction = float(prediction.detach().mean()) * config.label_scale
    do_keep = float(prediction.detach().mean()) < -config.threshold
    # Evaluation on independent draws when supplied, not the same noisy draw
    # that selected the hindsight best branch.
    actual_gain = label["audit_raw_gain"] if label.get("audit_raw_gain") is not None else label["raw_gain"]
    selected_gain = -actual_gain if do_keep else 0.0
    metrics = dict(utility_loss=float(loss.detach()), utility_prediction=raw_prediction,
                   signed_gain=label["raw_gain"], keep_rate=float(do_keep),
                   selected_gain=selected_gain,
                   selection_regret=max(-actual_gain, 0.0) - selected_gain,
                   sign_accuracy=float((raw_prediction < 0) == (actual_gain < 0)))
    return dict(loss=loss, prediction=prediction, metrics=metrics)


def summarize_storage_labels(labels):
    if not labels:
        raise ValueError("No V7 labels to summarize")
    keys = set.intersection(*(set(row["metrics"]) for row in labels))
    metrics = {key: sum(row["metrics"][key] for row in labels) / len(labels) for key in sorted(keys)}
    metrics.update(context_count=float(len(labels)),
                   independent_episodes=float(len({row["context"]["episode_id"] for row in labels})),
                   total_expert_forwards=sum(row["metrics"]["expert_forwards"] for row in labels))
    return metrics
