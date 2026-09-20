"""Offline cue -> sampled endpoint -> bank -> attention audit (no training).

Annotations are *evaluation targets*, never inputs to the writer or reader.
An evidence group is one required fact/event; its intervals are interchangeable
observations of that same fact. Different groups are all required. This avoids
calling a counting/order query solved because just one of its cues was found.

Coverage here means direct pre/post moment-token endpoint coverage, NOT proof
that the feature encoded the fact. HAMLET short tokens can also carry a cue
indirectly. Attention is not causal use, and these metrics are not robot success.
"""
from __future__ import annotations

import copy
import math

import torch

from .diagnostic_interventions import _endpoint_boundary, old_event_ids
from .replay_v3 import encode_until, read_bank, replay_bank


SCHEMA_VERSION = 1
STATUSES = ("unknown", "none", "verified")
LIMITATIONS = [
    "Offline demonstration replay, not closed-loop robot success or action loss.",
    "Unknown annotations are excluded from evidence metrics; missing denominators are null, not zero.",
    "Endpoint coverage is not semantic encoding. Short tokens may also carry evidence indirectly.",
    "Attention rank/mass is a retrieval proxy, not proof that the action expert used the evidence.",
    "A high rank is not a predicted discrete retrieval: the deployed reader uses all soft attention weights.",
    "Candidate frames at or after the query and unfinished events are never annotation positives.",
    "Human cue labels are privileged scoring targets, never policy or writer inputs.",
    "Multiple groups are required jointly; alternative intervals within a group describe the same fact.",
    "Descriptive rates on manually selected queries are not unbiased task-suite estimates.",
    "Old means event end strictly before oldest short-window endpoint; age alone does not prove memory need.",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def validate_annotation(row, episode, *, cache_fingerprint, split):
    """Validate against immutable cache IDs/frames and return an independent row.

    This deliberately does not infer evidence from subgoal labels, age, or
    attention. ``none`` is an explicit reviewer judgment; blank means unknown.
    """
    _require(isinstance(row, dict), "Annotation must be a JSON object")
    required = {"schema_version", "cache_fingerprint", "episode_id", "split", "decision",
                "query_frame", "status", "cues", "reviewer", "notes"}
    _require(required <= row.keys(), f"Missing annotation fields: {sorted(required - row.keys())}")
    _require(type(row["schema_version"]) is int and row["schema_version"] == SCHEMA_VERSION,
             "Unsupported evidence schema_version")
    _require(isinstance(cache_fingerprint, str) and len(cache_fingerprint) == 64
             and all(c in "0123456789abcdef" for c in cache_fingerprint), "Invalid cache fingerprint")
    _require(row["cache_fingerprint"] == cache_fingerprint, "Annotation/cache fingerprint mismatch")
    if "cache_fingerprint" in episode:
        _require(episode["cache_fingerprint"] == cache_fingerprint, "Episode/cache fingerprint mismatch")
    _require(split in ("train", "val") and row["split"] == split, "Annotation/cache split mismatch")
    eid = _integer(row["episode_id"], "episode_id")
    _require(eid == int(episode["episode_id"]), "Annotation episode_id mismatch")
    decision = _integer(row["decision"], "decision")
    _require(decision < len(episode["actions"]), "Query decision out of bounds")
    _require(bool(episode["decision_mask"][decision]), "Query must be an active decision")
    frames = torch.as_tensor(episode["frames"])
    _require(frames.ndim == 1 and len(frames) == len(episode["actions"]) + 1,
             "Episode frame count mismatch")
    _require(frames.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
             "Episode frames must be integers")
    _require(bool((frames >= 0).all()) and bool((frames[1:] > frames[:-1]).all()),
             "Episode frames must be nonnegative and strictly increasing")
    query = _integer(row["query_frame"], "query_frame")
    _require(query == int(frames[decision]), "Annotation query_frame differs from cached decision")
    status = row["status"]
    _require(status in STATUSES, f"status must be one of {STATUSES}")
    _require(isinstance(row["reviewer"], str) and isinstance(row["notes"], str),
             "reviewer and notes must be strings")
    _require(isinstance(row["cues"], list), "cues must be a list")
    if status in ("verified", "none"):
        _require(bool(row["reviewer"].strip()), "Verified/none annotations require a reviewer")
    if status == "verified":
        _require(bool(row["cues"]), "Verified annotation requires evidence cues")
    if status == "none":
        _require(not row["cues"], "status none must not contain cues")
    ids = set()
    for cue in row["cues"]:
        _require(isinstance(cue, dict), "Each cue must be an object")
        _require(isinstance(cue.get("cue_id"), str) and bool(cue["cue_id"].strip()),
                 "cue_id must be a nonempty string")
        _require(cue["cue_id"] not in ids, "Duplicate cue_id")
        ids.add(cue["cue_id"])
        _require(isinstance(cue.get("description"), str) and bool(cue["description"].strip()),
                 "Cue description is required")
        intervals = cue.get("intervals")
        _require(isinstance(intervals, list) and bool(intervals), "Cue intervals must be nonempty")
        previous = -1
        for interval in intervals:
            _require(isinstance(interval, list) and len(interval) == 2,
                     "Each interval must be [start_frame, end_frame] (inclusive)")
            start, end = interval
            _integer(start, "cue start_frame")
            _integer(end, "cue end_frame")
            _require(start <= end < query, "Cue intervals must be entirely before the query frame")
            _require(start > previous, "Cue intervals must be sorted and non-overlapping")
            previous = end
    return copy.deepcopy(row)


def rank_interval(scores, positive_indices, *, tie_tolerance=1e-8):
    """Best/worst rank of the FIRST relevant item; never break ties by event ID.

    The worst rank counts tied *negatives*, not tied alternative positives.
    Thus a bank of equivalent positive cues cannot be penalized for their order.
    """
    values = [float(s) for s in scores]
    _require(all(math.isfinite(v) for v in values), "Rank scores must be finite")
    _require(math.isfinite(tie_tolerance) and tie_tolerance >= 0, "Invalid tie tolerance")
    positives = list(positive_indices)
    _require(all(type(i) is int and 0 <= i < len(values) for i in positives),
             "Positive rank index out of bounds")
    _require(len(set(positives)) == len(positives), "Duplicate positive rank indices")
    if not positives:
        return {"best": None, "worst": None}
    score = max(values[i] for i in positives)
    positive_set = set(positives)
    negatives = [v for i, v in enumerate(values) if i not in positive_set]
    return {"best": 1 + sum(v > score + tie_tolerance for v in negatives),
            "worst": 1 + sum(v >= score - tie_tolerance for v in negatives)}


def _top_k(rank, n, p, top_k):
    results = {}
    for k in top_k:
        take = min(k, n)
        chance = (1 - math.comb(n - p, take) / math.comb(n, take)) if p else None
        results[str(k)] = {
            "guaranteed": rank["worst"] <= k if p else None,
            "possible": rank["best"] <= k if p else None,
            "chance": chance,
        }
    return results


def score_cues(row, episode, bank_ids, event_weights, null_weight, *, memory_window=4, top_k=(1, 4)):
    """Score direct carriers AFTER a label-blind replay/read has completed.

    This pure scoring entry point also permits controlled synthetic tests.
    Caller validates the annotation first. Bank validation remains mandatory.
    """
    top_k = tuple(top_k)
    _require(bool(top_k) and all(type(k) is int and k > 0 for k in top_k)
             and len(set(top_k)) == len(top_k), "top_k must contain unique positive integers")
    decision = row["decision"]
    old_bank = old_event_ids(episode, decision, bank_ids, memory_window)
    boundary = _endpoint_boundary(episode, decision, memory_window)
    weights = [float(x) for x in event_weights]
    _require(len(weights) == len(bank_ids), "Attention weights do not align with bank IDs")
    _require(all(math.isfinite(x) and x >= 0 for x in weights + [float(null_weight)]),
             "Attention weights must be finite and nonnegative")
    _require(math.isclose(sum(weights) + float(null_weight), 1.0, abs_tol=1e-5),
             "Event plus null attention weights must sum to one")
    if row["status"] != "verified":
        return []
    frames = [int(x) for x in episode["frames"][:decision + 1]]
    valid = [i for i in range(decision) if bool(episode["transition_valid"][i])]
    bank_positions = {i: j for j, i in enumerate(bank_ids)}
    results = []
    for cue in row["cues"]:
        contains = lambda frame: any(lo <= frame <= hi for lo, hi in cue["intervals"])
        sampled = [f for f in frames[:-1] if contains(f)]
        # An event spanning a cue is NOT enough: its actual observed endpoints
        # must include the cue. Unsampled frames cannot be assumed encoded.
        direct = [i for i in valid if contains(frames[i]) or contains(frames[i + 1])]
        retained = [i for i in bank_ids if i in direct]
        positives = [bank_positions[i] for i in retained]
        rank = rank_interval(weights, positives)
        with_null = rank_interval(weights + [float(null_weight)], positives)
        results.append({
            "cue_id": cue["cue_id"], "description": cue["description"],
            "intervals": copy.deepcopy(cue["intervals"]), "sampled_frames": sampled,
            "direct_event_ids": direct, "bank_event_ids": retained,
            "old_bank_event_ids": [i for i in retained if i in old_bank],
            "temporally_old": all(hi < boundary for _, hi in cue["intervals"]),
            "attention_mass": sum(weights[i] for i in positives),
            "rank": rank, "rank_with_null": with_null,
            "top_k": _top_k(rank, len(weights), len(positives), top_k),
            "top_k_with_null": _top_k(with_null, len(weights) + 1, len(positives), top_k),
        })
    return results


@torch.no_grad()
def audit_query(memory, episode, row, *, cache_fingerprint, split,
                writer_policy="all", memory_window=4, top_k=(1, 4)):
    """Read-only replay using a frozen/eval memory and completed past events.

    It does not alter module modes, parameters, or RNG state via sampling. The
    checkpoint loader is responsible for setting eval mode; train mode is an
    error so future dropout additions cannot silently make the audit stochastic.
    """
    row = validate_annotation(row, episode, cache_fingerprint=cache_fingerprint, split=split)
    _require(not memory.training, "Audit requires memory.eval(); it never changes module mode")
    decision = row["decision"]
    # Only decision selects the observed prefix. None of row['cues'], labels,
    # notes, or GT future action targets enters encoding, storage, or reading.
    encoded = encode_until(memory, episode, decision)
    bank, storage = replay_bank(memory, episode, decision, policy=writer_policy, encoded=encoded)
    read = read_bank(memory, episode, decision, bank, encoded=encoded)
    # This is the actual attention used to read values. Averaged logits from
    # event_scores have a different order-of-operations and are not used here.
    weights = read["weights"][0].float().mean(0).cpu().tolist()
    event_weights, null_weight = weights[:-1], weights[-1]
    old = old_event_ids(episode, decision, bank, memory_window)
    cues = score_cues(row, episode, bank, event_weights, null_weight,
                     memory_window=memory_window, top_k=top_k)
    all_covered = None if not cues else all(bool(c["sampled_frames"]) for c in cues)
    all_retained = None if not cues else all(bool(c["bank_event_ids"]) for c in cues)
    return {
        "schema_version": SCHEMA_VERSION, "episode_id": row["episode_id"],
        "split": split, "decision": decision, "query_frame": row["query_frame"],
        "status": row["status"], "reviewer": row["reviewer"], "notes": row["notes"],
        "bank_ids": bank, "old_bank_ids": old, "writer_policy": writer_policy,
        "oldest_short_frame": int(_endpoint_boundary(episode, decision, memory_window)),
        "storage": storage,
        "attention": {"event_weights": event_weights, "null_weight": null_weight,
                      "total_old_weight": sum(w for i, w in zip(bank, event_weights) if i in old),
                      "total_recent_weight": sum(w for i, w in zip(bank, event_weights) if i not in old)},
        "gate_mean": float(read["gate_mean"].item()),
        "residual_norm": float(read["residual_norm"].item()),
        "cues": cues, "all_groups_sampled": all_covered, "all_groups_retained": all_retained,
        "top_k_requested": list(top_k),
    }


def _fraction(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def _mean(values):
    return sum(values) / len(values) if values else None


def _summarize_groups(groups, keys):
    sampled = [c for c in groups if c["sampled_frames"]]
    carriers = [c for c in sampled if c["direct_event_ids"]]
    retained = [c for c in carriers if c["bank_event_ids"]]
    report = {
        "endpoint_coverage": _fraction(len(sampled), len(groups)),
        "carrier_coverage": _fraction(len(carriers), len(sampled)),
        "bank_retention": _fraction(len(retained), len(carriers)),
        "end_to_end_retention": _fraction(len(retained), len(groups)),
        "attention_mass_mean_given_retained": _mean([c["attention_mass"] for c in retained]),
    }
    for metric in ("top_k", "top_k_with_null"):
        report[metric] = {}
        for key in keys:
            eligible = [c[metric][key] for c in retained if key in c[metric]]
            report[metric][key] = {
                "guaranteed": _fraction(sum(c["guaranteed"] for c in eligible), len(eligible)),
                "possible": _fraction(sum(c["possible"] for c in eligible), len(eligible)),
                "chance_mean": _mean([c["chance"] for c in eligible]),
                "unconditional_guaranteed": _fraction(sum(c["guaranteed"] for c in eligible), len(groups)),
            }
    return report


def summarize_audit(results):
    """Explicit denominators, joint group coverage, and no NaN percentages."""
    verified = [r for r in results if r["status"] == "verified"]
    groups = [c for r in verified for c in r["cues"]]
    keys = sorted({str(k) for r in results for k in r["top_k_requested"]}, key=int)
    # Do not compare mixtures of different top-k plans silently.
    _require(all(set(map(str, r["top_k_requested"])) == set(keys) for r in results),
             "Audit results use different top-k plans")
    report = {
        "counts": {"queries": len(results), "unknown": sum(r["status"] == "unknown" for r in results),
                   "none": sum(r["status"] == "none" for r in results), "verified": len(verified),
                   "cue_groups": len(groups)},
        "structural": {
            "mean_bank_fill": _mean([len(r["bank_ids"]) for r in results]),
            "mean_old_bank_fill": _mean([len(r["old_bank_ids"]) for r in results]),
            "mean_old_attention_mass": _mean([r["attention"]["total_old_weight"] for r in results]),
            "mean_null_weight": _mean([r["attention"]["null_weight"] for r in results]),
            "mean_null_weight_on_none": _mean([r["attention"]["null_weight"] for r in results if r["status"] == "none"]),
        },
        "evidence": _summarize_groups(groups, keys),
        "old_evidence": _summarize_groups([c for c in groups if c["temporally_old"]], keys),
        "joint_queries": {
            "all_groups_sampled": _fraction(sum(r["all_groups_sampled"] for r in verified), len(verified)),
            "all_groups_retained": _fraction(sum(r["all_groups_retained"] for r in verified), len(verified)),
            "all_groups_top_k": {},
        },
        "limitations": list(LIMITATIONS),
    }
    # Joint top-k coverage is unconditional: a missing cue is a miss, not
    # excluded. It still does NOT mean the action/ordered sequence was correct.
    for key in keys:
        joint = sum(all(c["top_k"][key]["guaranteed"] is True for c in r["cues"]) for r in verified)
        report["joint_queries"]["all_groups_top_k"][key] = _fraction(joint, len(verified))
    return report
