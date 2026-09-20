"""Shared causal offline/online observation proposals for v6 visual memory.

No actions, targets, subgoals, future observations, or human cue labels enter
this API. Archive capping is deterministic streaming temporal thinning, not
learned eviction or FIFO. Offline prefixes replay the same thinning operation
that online inference applies after each observed frame; memory remains bounded.
"""

import torch
from torch.nn import functional as F

from .core_v6 import CANDIDATE_NAMES, _frame_number, observation_image_tokens


def uniform_positions(length, budget):
    if type(length) is not int or type(budget) is not int or length < 0 or budget <= 0:
        raise ValueError("length/budget must be nonnegative/positive integers")
    if length <= budget:
        return list(range(length))
    if budget == 1:
        return [0]
    return [round(i * (length - 1) / (budget - 1)) for i in range(budget)]


def bound_archive(observations, max_events):
    """Bound a stream while preserving first/latest temporal endpoints.

    Overflow drops the interior observation with the smallest neighboring time
    span (the most densely covered region). Ties drop the earlier interior row.
    This is a heuristic, NOT optimal semantic retention. Applying it to a full
    prefix or incrementally to ``retained + [new]`` gives identical results.
    Stable event_id values must be assigned before calling this function.
    """
    if type(max_events) is not int or max_events <= 0:
        raise ValueError("max_events must be a positive integer")
    retained, previous = [], None
    for observation in observations:
        frame = _frame_number(observation["frame"])
        if previous is not None and frame <= previous:
            raise ValueError("Archive stream must have unique increasing frames")
        previous = frame
        retained.append(observation)
        if len(retained) > max_events:
            if max_events == 1:
                retained.pop()  # Explicit earliest-evidence preference.
            else:
                remove = min(range(1, len(retained) - 1), key=lambda i:
                             (_frame_number(retained[i + 1]["frame"]) - _frame_number(retained[i - 1]["frame"]), i))
                retained.pop(remove)
    return retained


def observation_from_episode(episode, index):
    if type(index) is not int or not 0 <= index < len(episode["frames"]):
        raise ValueError("Observation index out of bounds")
    if "is_demo" not in episode:
        raise ValueError("V6 requires explicit cached is_demo tags; rebuild this stale cache")
    return {"features": episode["features"][index], "image_mask": episode["image_masks"][index],
            "attention_mask": episode["attention_masks"][index], "short": episode["short"][index],
            "state": episode["state"][index], "frame": _frame_number(episode["frames"][index]),
            "is_demo": bool(episode["is_demo"][index]), "event_id": index}


def _model_inputs(observation):
    return {key: observation[key] for key in
            ("features", "image_mask", "attention_mask", "short", "state", "frame", "is_demo")}


@torch.no_grad()
def proposal_vector(observation):
    """Frozen contextual image cosine proposal; deliberately non-differentiable."""
    return observation_image_tokens(observation["features"], observation["image_mask"],
                                    observation["attention_mask"], observation["short"]).mean(0).detach()


def build_candidates(memory, observations, current):
    """Build equal-budget chronological packs from strictly past observations.

    Returns query [1,H] and ordered candidates uniform/relevant/hybrid/null.
    Each non-null token sequence has at most read_budget * visual_tokens tokens.
    The semantic hard proposal has no action gradient; the learned query/key
    weighting inside every pack does. This distinction is part of the protocol.
    """
    now = _frame_number(current["frame"])
    frames = [_frame_number(row["frame"]) for row in observations]
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise ValueError("Observation frames must be unique and chronologically increasing")
    if any(frame >= now for frame in frames):
        raise ValueError("Archive contains current or future observation; require frame < query frame")
    ids = [row.get("event_id", i) for i, row in enumerate(observations)]
    if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Observation IDs must be unique nonnegative integers")
    query = memory.query_features(**_model_inputs(current)).reshape(1, -1)
    # IDs are stable before thinning. Online callers retain these IDs rather
    # than renumbering the bounded list at every action query.
    tagged = [dict(row, event_id=event_id) for row, event_id in zip(observations, ids)]
    archive = bound_archive(tagged, memory.config.max_archive_events)
    archive_ids = [row["event_id"] for row in archive]
    count = len(archive)
    budget = min(count, memory.config.read_budget)
    if not count:
        packs = {name: memory.pack(query, {}, [], []) for name in CANDIDATE_NAMES}
        return {"query": query, "candidates": packs,
                "details": {"query_frame": now, "observed_count": 0, "archive_count": 0,
                            "encoded_count": 0, "archive_ids": [], "proposal": "frozen_image_cosine",
                            "archive_policy": "streaming_temporal_thinning"}}

    current_vector = proposal_vector(current)
    vectors = torch.stack([proposal_vector(row).to(current_vector.device) for row in archive])
    similarities = (F.normalize(vectors, dim=-1) * F.normalize(current_vector, dim=-1)).sum(-1)
    # Stable ties prefer earlier observations, preserving reproducibility.
    relevance = sorted(range(count), key=lambda i: (-float(similarities[i]), i))
    uniform = uniform_positions(count, max(1, budget))
    relevant = sorted(relevance[:budget])
    coverage = uniform_positions(count, max(1, (budget + 1) // 2))
    hybrid = list(coverage)
    for index in relevance:
        if index not in hybrid:
            hybrid.append(index)
        if len(hybrid) == budget:
            break
    hybrid = sorted(hybrid[:budget])
    chosen = {"uniform": uniform, "relevant": relevant, "hybrid": hybrid, "null": []}
    union = sorted(set(uniform + relevant + hybrid))
    encoded = [memory.encode_observation(**_model_inputs(archive[i])) for i in union]
    positions = {i: j for j, i in enumerate(union)}
    packs = {}
    for name in CANDIDATE_NAMES:
        selected = chosen[name]
        # Contextualize each set separately. Information from unselected sets
        # must not contaminate the counterfactual utility comparison.
        context = memory.contextualize([encoded[positions[i]] for i in selected])
        packs[name] = memory.pack(query, context, list(range(len(selected))),
                                  [archive_ids[i] for i in selected])
    return {"query": query, "candidates": packs,
            "details": {"query_frame": now, "observed_count": len(observations),
                        "archive_count": count, "encoded_count": len(union), "archive_ids": archive_ids,
                        "encoded_ids": [archive_ids[i] for i in union],
                        "archive_frames": [row["frame"] for row in archive],
                        "archive_policy": "streaming_temporal_thinning",
                        "proposal": "frozen_image_cosine", "hard_proposal_has_grad": False}}


def prepare_candidates(memory, episode, decision):
    if type(decision) is not int or not 0 <= decision < len(episode["frames"]):
        raise ValueError("Decision out of bounds")
    # Inspect prefix only. Targets/actions and later observations are never read.
    current = observation_from_episode(episode, decision)
    past = [observation_from_episode(episode, index) for index in range(decision)]
    return build_candidates(memory, past, current)
