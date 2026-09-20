"""Observation-only inputs and content-location control for the V15 probe.

The ordinary bank is the frozen V13 singleton/framewise prefix.  The optional
control moves BOTH camera image contents of EVERY observed demo frame through
a deterministic single-cycle derangement, while keeping destination time,
demo and spatial identities fixed.  Execution history and the current query
are untouched.  Z/C are re-encoded, rather than moving time-bearing Z tokens.

No label enters ``build_probe_inputs``.  ``remap_positive_frames`` is a separate
target-only operation: targets follow their image CONTENT to its destination.
This is a retrieval equivariance diagnostic, not a physically valid rollout or
a claim that deliberately wrong memory should degrade an action prediction.
No Action Expert, model loading, parameter mutation, or global random draw.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import random

import torch

from run_scripts.robomme.framewise_demo_tail_v13 import build_framewise_prefix
from run_scripts.robomme.replay_visual_patch_v11 import (
    OBSERVATION_KEYS, _columns, _options, _row, _rows,
)
from run_scripts.robomme import visual_demo_tail_bank_v13 as core


CAMERA_ORDER = core.CAMERA_ORDER
TRAINABLE_NAMES = frozenset(("query_projection.weight", "key_projection.weight"))
PERMUTATION_KIND = "all_demo_single_cycle_content_derangement_v15"


def _scope(memory):
    if not isinstance(memory, core.VisualDemoTailMemoryV13):
        raise TypeError("V15 inputs require the actual VisualDemoTailMemoryV13")
    if memory.read_mode != "differential":
        raise ValueError("V15 probes the differential reader's original past attention")
    extra = [name for name, parameter in memory.named_parameters()
             if parameter.requires_grad and name not in TRAINABLE_NAMES]
    if extra:
        raise ValueError(f"V15 bank encoders and all non-Q/K weights must be frozen: {extra}")


def _mapping(frames, demos, *, seed, episode_id, query_frame):
    """A local Sattolo cycle: label/content-independent, no global RNG change.

This samples a single cycle, not the uniform distribution over all possible
derangements.  All demo frames move when at least two exist.  No-demo and
singleton-demo cases are explicitly recorded as an ineffective intervention.
"""
    source_indices = list(range(len(frames)))
    demo_indices = [index for index, demo in enumerate(demos) if demo]
    if seed is not None and len(demo_indices) >= 2:
        payload = json.dumps([PERMUTATION_KIND, seed, episode_id, query_frame],
                             separators=(",", ":")).encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(payload).digest(), "big"))
        shuffled = demo_indices.copy()
        for index in range(len(shuffled) - 1, 0, -1):
            other = rng.randrange(index)  # Strictly smaller: no fixed points.
            shuffled[index], shuffled[other] = shuffled[other], shuffled[index]
        for destination, source in zip(demo_indices, shuffled):
            source_indices[destination] = source
    return source_indices


def _original_images(memory, columns, query, sidecar, record, *, episode_id,
                     cache_fingerprint, sidecar_fingerprint, current_frame):
    """Read raw images from [0, query) and the validated, strictly past tail.

The original framewise builder has already validated the full canonical/tail
contract.  Re-read only those observation columns; no current/future feature,
sidecar label, action or state is inspected while deriving control images.
"""
    rows = _rows(columns, query, memory)
    images, frames, demos, _ = core._sidecar_prefix(sidecar, record,
        episode_id=episode_id, cache_fingerprint=cache_fingerprint,
        sidecar_fingerprint=sidecar_fingerprint, current_frame=current_frame,
        width=memory.config.feature_dim)
    entries = [(int(row[3]), core._canonical_images(row, memory), bool(row[4]))
               for row in rows]
    entries.extend((int(frames[index]), images[index].to(device=memory._device()), bool(demos[index]))
                   for index in range(len(frames)))
    entries.sort(key=lambda item: item[0])
    return entries


def build_probe_inputs(memory, observations, query, *, sidecar, record,
                       episode_id, cache_fingerprint, sidecar_fingerprint,
                       permutation_seed=None):
    """Return (unchanged current observation, strict-past bank, JSON-safe map).

    ``observations`` is accessed solely through OBSERVATION_KEYS.  The caller
    must supply independently verified sidecar identities.  Non-Q/K parameters
    must already be frozen; this function never changes requires_grad flags.
    Historical encoding runs under no_grad, but current Q remains attached.
    Passing a seed requests a content control; passing None is exact original
    framewise replay.  No targets are accepted or accessed by this function.
    """
    _scope(memory)
    _options(memory, observations, query, CAMERA_ORDER, False)
    if permutation_seed is not None and (type(permutation_seed) is not int or permutation_seed < 0):
        raise ValueError("permutation_seed must be None or a nonnegative Python integer")
    columns = _columns(observations)
    feature, image, attention, frame, demo = _row(columns, query, memory)
    if bool(demo):
        raise ValueError("V15 input construction is action-query only")

    with torch.no_grad():
        bank = build_framewise_prefix(memory, observations, query,
            camera_order=CAMERA_ORDER, include_tail=True,
            sidecar=sidecar, record=record, episode_id=episode_id,
            cache_fingerprint=cache_fingerprint, sidecar_fingerprint=sidecar_fingerprint)
        frames = bank.frames[0].tolist()
        demos = bank.is_demo[0].tolist()
        source_indices = _mapping(frames, demos, seed=permutation_seed,
                                  episode_id=episode_id, query_frame=int(frame))
        effective = source_indices != list(range(len(frames)))
        if effective:
            entries = _original_images(memory, columns, query, sidecar, record,
                episode_id=episode_id, cache_fingerprint=cache_fingerprint,
                sidecar_fingerprint=sidecar_fingerprint, current_frame=int(frame))
            if ([entry[0] for entry in entries] != frames
                    or [entry[2] for entry in entries] != demos):
                raise ValueError("Original images disagree with the validated framewise bank")
            address, content = [], []
            for destination, source in enumerate(source_indices):
                if destination == source:
                    # In particular, execution encodings are exactly untouched.
                    address.append(bank.tokens[0, destination])
                    content.append(bank.content[0, destination])
                    continue
                record_at_destination = memory.encode_bank_images(
                    entries[source][1][None], bank.frames[:, destination],
                    bank.is_demo[:, destination], camera_order=CAMERA_ORDER)
                address.append(record_at_destination.encoded[0])
                content.append(record_at_destination.content[0])
            bank = core.DifferentialBank(torch.stack(address)[None], bank.frames,
                bank.is_demo, bank.valid, torch.stack(content)[None])

    # Ordinary V13 encoding keeps the actual current image and HAMLET tokens.
    # Do not wrap this call in no_grad: it includes the trainable Q projection.
    current = memory.encode_observation(feature[None], image[None], attention[None],
        frame[None], demo[None], camera_order=CAMERA_ORDER)
    memory._bank(bank, current)  # Native chronology/layout/finite guard.
    info = {
        "episode_id": episode_id,
        "query_frame": int(frame),
        "candidate_frames": frames,
        "candidate_is_demo": demos,
        "destination_to_source_indices": source_indices,
        "destination_to_source_frames": [frames[index] for index in source_indices],
        "permutation_seed": permutation_seed,
        "permutation_kind": "identity" if permutation_seed is None else PERMUTATION_KIND,
        "permutation_effective": effective,
    }
    return current, bank, info


def remap_positive_frames(positive_frames, info):
    """Return sorted destination frames containing the positive source images.

    This helper sees targets, unlike the builder.  It never changes their count,
    invents new evidence or keeps a target at its old timestamp after moving its
    content.  Unknown/duplicate/non-demo positives and malformed maps fail shut.
    Empty targets are representable here; the objective must reject an empty
    supervised bag rather than assign it a successful or zero loss.
    """
    if not isinstance(info, Mapping):
        raise TypeError("info must be the builder's mapping")
    frames = info.get("candidate_frames")
    demos = info.get("candidate_is_demo")
    indices = info.get("destination_to_source_indices")
    source_frames = info.get("destination_to_source_frames")
    if (not isinstance(frames, list) or any(type(x) is not int or x < 0 for x in frames)
            or any(left >= right for left, right in zip(frames, frames[1:]))):
        raise ValueError("Candidate frames must be unique chronological nonnegative integers")
    count = len(frames)
    if (not isinstance(demos, list) or len(demos) != count or any(type(x) is not bool for x in demos)
            or any(not left and right for left, right in zip(demos, demos[1:]))):
        raise ValueError("Candidate demo flags must form a boolean prefix")
    if (not isinstance(indices, list) or len(indices) != count
            or any(type(x) is not int for x in indices) or sorted(indices) != list(range(count))):
        raise ValueError("Destination-to-source indices must be a complete bijection")
    if (not isinstance(source_frames, list) or len(source_frames) != count
            or any(type(x) is not int for x in source_frames)
            or source_frames != [frames[index] for index in indices]):
        raise ValueError("Source frame identities disagree with the bijection")
    if any(demos[destination] != demos[source]
           or (not demos[destination] and destination != source)
           for destination, source in enumerate(indices)):
        raise ValueError("Only demo contents may move; execution must retain identity")
    if (not isinstance(positive_frames, Sequence) or isinstance(positive_frames, (str, bytes))
            or any(type(x) is not int or x < 0 for x in positive_frames)
            or len(set(positive_frames)) != len(positive_frames)):
        raise ValueError("Positive frames must be a sequence of distinct nonnegative integers")
    positives = set(positive_frames)
    eligible = {frame for frame, demo in zip(frames, demos) if demo}
    if not positives <= eligible:
        raise ValueError("Every positive must identify an actually observed demo frame")
    return [frames[destination] for destination, source in enumerate(indices)
            if frames[source] in positives]
