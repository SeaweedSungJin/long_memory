"""Read-only V19 bank provenance and matched whole-event interventions.

These are offline normalized action-error diagnostics, not simulator success.
The loaded V18/V19 FIFO core and Action Expert must already be in eval mode.
Replay happens once per query. All interventions share the current short/query,
cached observations, GT masks, flow noise/time seed and pure-noise sampler seed.
Only visible whole bank events are removed. Each chronological partition has
both a same-size random event-set control and a separately seeded random
contiguous-block control. No K/V permutation is used.
"""
from __future__ import annotations

import math
import random

import torch

from gr00t.long_memory.cache_reader_v3 import validate_decision
from gr00t.long_memory.replay_v7 import _validate_prefix
from run_scripts.robomme.deployment_objective_v9 import generated_prefix_objective
from run_scripts.robomme.flow_objective_v19 import episode_flow_v19
from run_scripts.robomme.validation_v19 import generated_metrics_v19


DIAGNOSTIC_VERSION = "v19_visible_fifo_event_interventions_v2"


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def bank_provenance(ep, decision, cfg):
    """Describe the FIFO immediately before current-query READ, without actions.

    An event is an observation encoding, not a recorded-action transition.
    ``source_frames`` preserves left-padding repetitions in the original
    HAMLET K-window. Its Q output tokens each mix this window; individual
    output-token positions do not identify individual images. Moment-storage
    arms instead store the endpoint's Q moment tokens, with the current query
    still using the HAMLET window. Times below are canonical event indices;
    insertion/deletion occur AFTER the named event's READ.

    Only observed past deletions are recorded. ``deletion_at=None`` means not
    deleted as of this query, not guaranteed permanent retention.
    """
    _integer(decision, "decision")
    if decision >= len(ep["frames"]):
        raise ValueError("decision is outside the episode")
    _validate_prefix(ep, decision + 1)
    capacity = _integer(cfg.capacity_events, "capacity_events", 1)
    tokens = _integer(cfg.num_short_tokens, "num_short_tokens", 1)
    window = _integer(cfg.short_window, "short_window", 1)
    if cfg.representation not in ("short", "adapted_short", "moment"):
        raise ValueError("Unsupported V18/V19 representation")
    eid = int(ep["episode_id"])
    frames = [int(value) for value in ep["frames"][:decision + 1]]
    demos = [bool(value) for value in ep["is_demo"][:decision + 1]]
    first = max(0, decision - capacity)

    def source(index):
        indices = [max(0, j) for j in range(index - window + 1, index + 1)]
        return indices, [frames[j] for j in indices]

    events = []
    for index in range(decision):
        short_indices, short_frames = source(index)
        visible = index >= first
        position = index - first if visible else None
        eviction = index + capacity
        events.append({"episode_id": eid, "event_id": index,
            "endpoint_frame": frames[index], "is_demo": demos[index],
            "source_event_ids": [index] if cfg.representation == "moment" else short_indices,
            "source_frames": [frames[index]] if cfg.representation == "moment" else short_frames,
            "hamlet_source_event_ids": short_indices, "hamlet_source_frames": short_frames,
            "insertion_at": index, "insertion_frame": frames[index],
            "first_read_at": index + 1,
            "deletion_at": eviction if eviction < decision else None,
            "deletion_frame": frames[eviction] if eviction < decision else None,
            "visible": visible, "current_position": position,
            "token_positions": list(range(position * tokens, (position + 1) * tokens)) if visible else [],
            "tokens_per_event": tokens})
    query_indices, query_frames = source(decision)
    return {"version": DIAGNOSTIC_VERSION, "episode_id": eid, "decision": decision,
        "representation": cfg.representation, "capacity_events": capacity,
        "tokens_per_event": tokens, "hamlet_window": window,
        "token_semantics": "Q mixed feature tokens per event; tokens are not individual images",
        "write_timing": "after event READ; current query has not been inserted",
        "query": {"episode_id": eid, "event_id": decision,
            "endpoint_frame": frames[decision], "is_demo": demos[decision],
            "source_event_ids": query_indices, "source_frames": query_frames,
            "insertion_at": None, "current_position": None},
        "visible_event_ids": list(range(first, decision)),
        "visible_event_count": decision - first,
        "visible_token_count": (decision - first) * tokens, "events": events}


def drop_event_positions(bank, positions, tokens_per_event):
    """Return an independent tensor, retaining whole events in original order."""
    tokens = _integer(tokens_per_event, "tokens_per_event", 1)
    if not isinstance(bank, torch.Tensor) or bank.ndim != 3 or bank.shape[1] % tokens:
        raise ValueError("bank must be [B,events*tokens_per_event,D]")
    count = bank.shape[1] // tokens
    positions = list(positions)
    if any(type(index) is not int or not 0 <= index < count for index in positions):
        raise ValueError("Removal positions must refer to visible bank events")
    if len(set(positions)) != len(positions):
        raise ValueError("Removal positions must be unique")
    removed = set(positions)
    rows = [row for event in range(count) if event not in removed
            for row in range(event * tokens, (event + 1) * tokens)]
    indices = torch.tensor(rows, dtype=torch.long, device=bank.device)
    return bank.index_select(1, indices)


def intervention_specs(visible_event_ids, *, blocks=4, removal_seed=190019):
    """Prespecify partitions, random event sets and random contiguous blocks.

    Partition by event count, oldest first. Uneven final blocks differ by at
    most one event. For small banks, empty blocks are explicitly retained as
    no-op controls. Random sampling never consumes Python/Torch global RNG.
    Random blocks sample uniformly over ALL valid contiguous starts, including
    possible overlap/equality with the paired fixed partition. Their independent
    seed domain preserves every v1 event-set control and its original seed.
    """
    _integer(blocks, "blocks", 1)
    _integer(removal_seed, "removal_seed")
    ids = list(visible_event_ids)
    if any(type(index) is not int or index < 0 for index in ids) or ids != sorted(set(ids)):
        raise ValueError("Visible event IDs must be unique chronological integers")
    count = len(ids)
    result = [{"intervention": "normal-bank", "read_enabled": True,
               "removed_positions": [], "removed_event_ids": [], "block_index": None,
               "removal_kind": "none", "removal_seed": None},
              {"intervention": "memory-off", "read_enabled": False,
               "removed_positions": [], "removed_event_ids": [], "block_index": None,
               "removal_kind": "read-disabled", "removal_seed": None}]
    offset = 0
    for block in range(blocks):
        size = count // blocks + int(block < count % blocks)
        positions = list(range(offset, offset + size))
        offset += size
        random_seed = removal_seed + block
        sampled = sorted(random.Random(random_seed).sample(range(count), size))
        random_block_seed = removal_seed + 1000003 + block
        start = random.Random(random_block_seed).randrange(count - size + 1)
        sampled_block = list(range(start, start + size))
        for role, selected, kind, used_seed in (
            (f"block-{block}", positions, "contiguous-visible-event-block", None),
            (f"random-set-{block}", sampled, "same-size-random-event-set", random_seed),
            (f"random-block-{block}", sampled_block, "same-size-random-contiguous-event-block", random_block_seed)):
            result.append({"intervention": role, "read_enabled": True,
                "removed_positions": selected, "removed_event_ids": [ids[j] for j in selected],
                "block_index": block, "removal_kind": kind, "removal_seed": used_seed})
    return result


def _scalar_metrics(values):
    result = {}
    for name, value in values.items():
        if name in {"prediction", "loss", "action_loss"}:
            continue
        if isinstance(value, (int, float)) or isinstance(value, torch.Tensor) and value.numel() == 1:
            scalar = float(value)
            if not math.isfinite(scalar):
                raise FloatingPointError(f"Nonfinite diagnostic metric: {name}")
            result[name] = scalar
    return result


@torch.no_grad()
def interventions(head, core, ep, decision, *, seed, generation_seed, blocks=4,
                  removal_seed=190019, action_steps=16):
    """Evaluate normal/off/partition/random-set/random-block matched reads.

    Returns JSON-ready ``records`` and ``provenance``. A positive metric delta
    from normal means removing that input increased error. This is a paired
    local sensitivity measurement, not proof of the visual semantics stored.
    Model loading, held-out selection and artifact writing belong to the caller.
    """
    _integer(seed, "seed")
    _integer(generation_seed, "generation_seed")
    _integer(action_steps, "action_steps", 1)
    _integer(blocks, "blocks", 1)
    _integer(removal_seed, "removal_seed")
    if head.training or core.training:
        raise ValueError("Diagnostic head/core must already be in eval mode")
    validate_decision(ep, decision)
    provenance = bank_provenance(ep, decision, core.config)
    replay = core.replay(ep, decision, activation_checkpointing=False)
    bank, short, query = replay["bank"], replay["short"], replay["encoded_current"]
    if bank.shape[1] != provenance["visible_token_count"]:
        raise ValueError("Replayed bank does not match plain V18/V19 FIFO provenance")
    bank_before, short_before, query_before = bank.clone(), short.clone(), query.clone()
    visible = provenance["visible_event_ids"]
    records, normal_flow, normal_generated = [], None, None
    for spec in intervention_specs(visible, blocks=blocks, removal_seed=removal_seed):
        selected = drop_event_positions(bank, spec["removed_positions"], core.config.num_short_tokens)
        if spec["intervention"] == "normal-bank":
            conditioning, reader_metrics = replay["fused"], replay["metrics"]
        else:
            conditioning, reader_metrics = core.read_from_bank(short, query, selected,
                                                               read_enabled=spec["read_enabled"])
        if not spec["read_enabled"] and not torch.equal(conditioning, short):
            raise AssertionError("Memory-off changed the fixed current short")
        flow = episode_flow_v19(head, ep, decision, conditioning, seed=seed,
            tail_weight=1., action_steps=action_steps, activation_checkpointing=False)
        generated = generated_prefix_objective(head, ep, decision, conditioning,
            seed=generation_seed, action_steps=action_steps, activation_checkpointing=False)
        metrics = _scalar_metrics(flow)
        metrics["action_loss"] = float(flow["original_flow_loss"])
        metrics.update(generated_metrics_v19(generated, ep, decision, action_steps))
        if normal_flow is None:
            normal_flow = flow["prediction"].detach().clone()
            normal_generated = generated["prediction"].detach().clone()
        # Prediction-change norms are restricted to the same diagnostic masks.
        valid = ep["target_mask"][decision].to(flow["prediction"].device)[None]
        executed = valid.clone()
        executed[:, action_steps:] = False
        executed[:, :action_steps] &= ep["action_mask"][decision].to(valid.device)[None, :, None]
        removed = set(spec["removed_event_ids"])
        retained = [index for index in visible if index not in removed]
        record = {"version": DIAGNOSTIC_VERSION, "episode_id": int(ep["episode_id"]),
            "decision": decision, "flow_seed": seed, "generation_seed": generation_seed,
            **spec, "removed_event_count": len(removed), "remaining_event_ids": retained,
            "read_event_ids": retained if spec["read_enabled"] else [],
            "remaining_event_count": len(retained),
            "fused_delta_from_normal_norm": float((conditioning - replay["fused"]).float().norm()),
            "flow_prediction_delta_from_normal_norm": float((flow["prediction"].float() - normal_flow.float())[valid].norm()),
            "generated_prefix_delta_from_normal_norm": float((generated["prediction"].float() - normal_generated.float())[executed].norm()),
            **{"reader_" + key: value for key, value in _scalar_metrics(reader_metrics).items()},
            **metrics}
        reference = record if not records else records[0]
        for name in ("action_loss", "executed_prefix_flow_loss", "generated_prefix_mse",
                     "generated_executed_joint_mse", "generated_executed_gripper_mse"):
            record[name + "_delta_from_normal"] = record[name] - reference[name]
        records.append(record)
        if not (torch.equal(bank, bank_before) and torch.equal(short, short_before)
                and torch.equal(query, query_before)):
            raise AssertionError("Intervention mutated the original bank/short/query")
    return {"version": DIAGNOSTIC_VERSION, "records": records, "provenance": provenance}
