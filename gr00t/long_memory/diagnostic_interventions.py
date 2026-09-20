"""Causal, inference-only memory interventions for existing v3/v4 weights.

These are diagnosis tools, not a new training recipe. The writer's bank is
never changed by a read intervention. In particular ``shuffled_old`` reassigns
*available same-episode* old event content to old timestamp slots. It is not a
task-matched wrong-episode oracle and cannot establish that a semantic cue was
remembered. A successful shuffle tests temporal/content association, not cue
availability: every original old content item still occurs in the bank.

An event is old only if its END frame is strictly before the oldest current
HAMLET short-window endpoint. Endpoint equality is deliberately not old.
"""

import numbers

import torch

from gr00t.long_memory.replay_v3 import (
    _validate_bank,
    encode_until,
    event_inputs,
    read_bank,
    storage_prediction,
)


READ_MODES = ("full", "no_memory", "no_old", "only_old", "shuffled_old")


def _endpoint_boundary(episode, decision, memory_window):
    if (isinstance(memory_window, bool) or not isinstance(memory_window, numbers.Integral)
            or memory_window < 1):
        raise ValueError("memory_window must be a positive integer")
    if (isinstance(decision, bool) or not isinstance(decision, numbers.Integral)
            or not 0 <= decision <= len(episode["actions"])):
        raise ValueError("Decision out of bounds")
    frames = torch.as_tensor(episode["frames"])
    if (frames.ndim != 1 or len(frames) <= decision
            or not bool(torch.isfinite(frames[:decision + 1]).all())
            or bool((frames[:decision + 1] < 0).any())
            or bool((frames[1:decision + 1] <= frames[:decision]).any())):
        raise ValueError("Endpoint frames must be finite, nonnegative, strictly increasing")
    return float(frames[max(0, decision - (memory_window - 1))])


def old_event_ids(episode, decision, bank_ids, memory_window=4):
    """Return strictly-old IDs; validate causal bank membership without a model."""
    boundary = _endpoint_boundary(episode, decision, memory_window)
    bank = list(bank_ids)
    if any(isinstance(i, bool) or not isinstance(i, numbers.Integral) for i in bank):
        raise ValueError("Bank IDs must be integers")
    if bank != sorted(set(bank)):
        raise ValueError("Bank IDs must be unique and chronologically sorted")
    if any(i < 0 or i >= decision or not bool(episode["transition_valid"][i]) for i in bank):
        raise ValueError("Bank contains future, unfinished, or invalid event")
    return [int(i) for i in bank if float(episode["frames"][i + 1]) < boundary]


def _short_only(memory, episode, decision):
    """Bypass ALL reader/fusion calculations, including learned affine biases."""
    device = next(memory.parameters()).device
    short = episode["short"][decision].to(device=device, dtype=torch.float32)[None]
    b, q, _ = short.shape
    zero = short.new_zeros(b)
    return {
        "fused_short": short,
        "read": short.new_zeros(b, q, memory.config.hidden_dim),
        "weights": short.new_ones(b, q, 1),
        "event_scores": short.new_empty(b, 0),
        "null_score": short.new_zeros(b, 1),
        "gate_mean": zero.clone(),  # Unmeasured sentinel, not a learned gate statistic.
        "read_norm": zero.clone(),
        "residual_norm": zero.clone(),
        "null_weight": short.new_ones(b),
    }


@torch.no_grad()
def read_intervention(memory, episode, decision, bank_ids, *, mode="full", memory_window=4,
                      encoded=None):
    """Return normal ``memory.read`` tensors plus JSON-safe ``diagnostic`` metadata.

    Query short tokens, state, language and visual features are not modified.
    ``no_memory`` therefore means the SAME adapted AE with its additional memory
    input bypassed, not an independently trained no-memory expert baseline.

    ``shuffled_old`` cyclically rotates old event content by one chronological
    slot and re-encodes it using each destination's original start/end frames.
    This preserves recent rows, bank size, timestamp distribution, and all old
    content as a multiset. It is an association corruption, not cue deletion.
    ``intervention_effective`` describes a changed reader input, not a changed
    action or scientifically significant success-rate effect.
    """
    if mode not in READ_MODES:
        raise ValueError(f"Unknown read intervention {mode!r}; expected {READ_MODES}")
    bank = list(bank_ids)
    old = old_event_ids(episode, decision, bank, memory_window)
    _validate_bank(memory, episode, decision, bank)
    boundary = _endpoint_boundary(episode, decision, memory_window)
    old_set = set(old)
    selected = (list(old) if mode == "only_old" else
                [i for i in bank if i not in old_set] if mode == "no_old" else
                [] if mode == "no_memory" else bank.copy())
    metadata = {
        "mode": mode,
        "decision": int(decision),
        "decision_frame": float(episode["frames"][decision]),
        "memory_window": int(memory_window),
        "oldest_short_frame": boundary,
        "old_definition": "event_end_frame < oldest_short_frame",
        "source_bank_ids": [int(i) for i in bank],
        "read_bank_ids": [int(i) for i in selected],
        "old_event_ids": old,
        "read_old_event_ids": [int(i) for i in selected if i in old_set],
        "source_old_count": len(old),
        "source_recent_count": len(bank) - len(old),
        "read_old_count": sum(i in old_set for i in selected),
        "read_recent_count": sum(i not in old_set for i in selected),
        "intervention_effective": selected != bank,
        "changed_content_slots": 0,
        "content_source_by_slot": [],
        "writer_bank_unchanged": True,
        "gate_measured": mode != "no_memory",
    }
    if mode == "no_memory":
        result = _short_only(memory, episode, decision)
    elif mode != "shuffled_old" or len(old) < 2:
        result = read_bank(memory, episode, decision, selected, encoded)
        if mode == "shuffled_old":
            metadata["ineffective_reason"] = "fewer_than_two_old_events"
    else:
        if encoded is None:
            encoded = encode_until(memory, episode, decision)
        if len(encoded["event"]) < decision:
            raise ValueError("Encoded events do not cover decision prefix")
        device = next(memory.parameters()).device
        # Clone no episode tensors: all selected rows are fresh index_select
        # results. Re-encoding, rather than swapping time-bearing keys/values,
        # ensures that the destination slot's temporal metadata stays intact.
        raw = event_inputs(episode, device, decision)
        destinations = torch.tensor(old, device=device, dtype=torch.long)
        sources = torch.tensor(old[1:] + old[:1], device=device, dtype=torch.long)
        shuffled_raw = {
            name: value.index_select(0, destinations if name in ("start_frames", "end_frames")
                                     else sources)
            for name, value in raw.items()
        }
        rotated = memory.encode_events(shuffled_raw)
        index = torch.tensor(bank, device=device, dtype=torch.long)
        keys = encoded["keys"].index_select(0, index)
        values = encoded["values"].index_select(0, index)
        positions = torch.tensor([bank.index(i) for i in old], device=device, dtype=torch.long)
        for row, slot in enumerate(positions.tolist()):
            metadata["changed_content_slots"] += int(
                not torch.equal(keys[slot], rotated["keys"][row])
                or not torch.equal(values[slot], rotated["values"][row]))
        keys.index_copy_(0, positions, rotated["keys"])
        values.index_copy_(0, positions, rotated["values"])
        short = episode["short"][decision].to(device=device, dtype=torch.float32)[None]
        state = episode["state"][decision].to(device=device, dtype=torch.float32)[None]
        frame = torch.as_tensor(episode["frames"][decision], device=device,
                                dtype=torch.float32).reshape(1)
        result = memory.read(short, state, keys[None], values[None],
                             torch.ones((1, len(bank)), device=device, dtype=torch.bool), frame)
        metadata["content_source_by_slot"] = [
            {"slot_event_id": int(dst), "content_event_id": int(src),
             "slot_start_frame": float(episode["frames"][dst]),
             "slot_end_frame": float(episode["frames"][dst + 1])}
            for dst, src in zip(old, old[1:] + old[:1])
        ]
        metadata["intervention_effective"] = metadata["changed_content_slots"] > 0
        if not metadata["intervention_effective"]:
            metadata["ineffective_reason"] = "identical_encoded_old_content"
    result["diagnostic"] = metadata
    return result


@torch.no_grad()
def continuation_bank(memory, episode, bank_ids, start_candidate, decision, encoded=None,
                      policy="hard"):
    """Continue causal writes from an existing bank; do not reset its history.

    ``start_candidate`` is the next not-yet-processed event, inclusive. To
    evaluate an option chosen for event c, pass ``start_candidate=c+1``.
    Writes continue through ``decision-1``; each event sees only its completion
    endpoint and prior selected bank. Future GT actions may be used later as a
    loss target, but are never used for these storage decisions.

    Statistics cover only this suffix. Eviction timestamps refer to observed
    event-completion frames, not imagined continuous-time memory changes.
    """
    if policy not in ("all", "hard"):
        raise ValueError("Continuation policy must be all or hard")
    if (isinstance(start_candidate, bool) or not isinstance(start_candidate, numbers.Integral)
            or isinstance(decision, bool) or not isinstance(decision, numbers.Integral)
            or not 0 <= start_candidate <= decision <= len(episode["actions"])):
        raise ValueError("Continuation bounds must satisfy 0 <= start_candidate <= decision")
    bank = list(bank_ids)
    _validate_bank(memory, episode, start_candidate, bank)
    if encoded is None:
        encoded = encode_until(memory, episode, decision)
    if len(encoded["event"]) < decision:
        raise ValueError("Encoded events do not cover continuation prefix")
    stats = dict(attempted=0, accepted=0, replaced=0, forced=0,
                 learned_attempted=0, learned_accepted=0, learned_rejected=0,
                 initial_event_first_eviction_frame={})
    initial = set(bank)
    for candidate in range(start_candidate, decision):
        if not bool(episode["transition_valid"][candidate]):
            continue
        stats["attempted"] += 1
        before = bank.copy()
        if policy == "all":
            bank = (bank + [candidate])[-memory.config.capacity:]
        elif len(bank) < memory.config.min_fill:
            bank.append(candidate)
            stats["forced"] += 1
        else:
            prediction = storage_prediction(memory, episode, candidate, bank, encoded)
            choice = int(prediction["logits"].argmax().item())
            bank = list(prediction["options"][choice])
            stats["learned_attempted"] += 1
            stats["learned_accepted"] += int(choice != 0)
            stats["learned_rejected"] += int(choice == 0)
        accepted = candidate in bank
        stats["accepted"] += int(accepted)
        stats["replaced"] += int(accepted and len(before) == memory.config.capacity)
        for removed in set(before) - set(bank):
            if removed in initial:
                stats["initial_event_first_eviction_frame"][str(removed)] = float(
                    episode["frames"][candidate + 1])
    stats.update(bank_fill=len(bank), initial_bank_ids=list(bank_ids),
                 start_candidate=int(start_candidate), decision=int(decision), policy=policy)
    return bank, stats
