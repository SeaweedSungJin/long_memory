"""V16 probe-only replay with a gradient-connected shared image projection.

The two controlled scopes are original Q/K only, or original Q/K plus the
existing image_projection.weight. Its bias and all other visual parameters stay
frozen. No parameter, model input, timestamp, label or memory admission rule is
added. This is NOT a deployable policy or evidence of robot task success.

Both current and historical original observations are re-encoded on EVERY call.
No encoded bank is cached across optimizer updates. Unlike V15's deliberately
frozen past encoder, neither ordinary history nor permuted demo re-encoding is
wrapped in no_grad/detach here. The caller's grad context is respected, including
explicit no_grad validation. Q/K probability/loss helpers remain the same V15
functions; V15's Q/K-only scope assertion must not be used for the expanded arm.
"""
from __future__ import annotations

import torch

from run_scripts.robomme.framewise_demo_tail_v13 import build_framewise_prefix
from run_scripts.robomme.replay_visual_patch_v11 import OBSERVATION_KEYS, _columns, _options, _row
from run_scripts.robomme.segment_probe_inputs_v15 import (
    CAMERA_ORDER, PERMUTATION_KIND, _mapping, _original_images, remap_positive_frames,
)
from run_scripts.robomme.segment_retrieval_probe_v15 import QK_PARAMETER_NAMES, _memory
from run_scripts.robomme.visual_demo_tail_bank_v13 import DifferentialBank

PROJECTION_PARAMETER_NAME = "image_projection.weight"
EXPANDED_PARAMETER_NAMES = (*QK_PARAMETER_NAMES, PROJECTION_PARAMETER_NAME)


def _scope(memory):
    """Read-only contract: either controlled scope, or fully frozen evaluation."""
    parameters = _memory(memory)
    if PROJECTION_PARAMETER_NAME not in parameters:
        raise ValueError("Missing original shared image projection weight")
    actual = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if actual not in (set(), set(QK_PARAMETER_NAMES), set(EXPANDED_PARAMETER_NAMES)):
        raise ValueError("V16 permits only Q/K, Q/K + image_projection.weight, or all-frozen evaluation")
    return parameters


def configure_scope(memory, train_projection: bool):
    """IN PLACE on an owned copy: select exactly 2 or 3 original parameters.

    Clears stale gradients but never changes any parameter value. Return order
    is Q, K, then (only for the expanded arm) image_projection.weight. There is
    no bias thaw, new projection, optimizer construction or extra RNG draw.
    The caller is responsible for keeping the immutable source model separate.
    """
    if type(train_projection) is not bool:
        raise ValueError("train_projection must be an explicit Boolean")
    parameters = _memory(memory)
    if PROJECTION_PARAMETER_NAME not in parameters:
        raise ValueError("Missing original shared image projection weight")
    selected = EXPANDED_PARAMETER_NAMES if train_projection else QK_PARAMETER_NAMES
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in selected)
        parameter.grad = None
    _scope(memory)
    return tuple(parameters[name] for name in selected)


def build_probe_inputs_v16(memory, observations, query, *, sidecar, record,
                           episode_id, cache_fingerprint, sidecar_fingerprint,
                           permutation_seed=None):
    """Return the same (current, bank, info) value contract as the V15 builder.

    Only original observation columns are accepted by the replay path; no
    targets, actions or simulator subgoals enter the encoding. Full V13 raw-time
    chronology, camera/layout, sidecar binding and strict-past guards are reused.
    The fixed V15 content permutation moves both demo cameras together and
    re-encodes at DESTINATION timestamps; current/execution content is untouched.
    Target remapping remains a separate caller operation after construction.

    The helper never changes requires_grad flags, gradients, parameters, inputs
    or global RNG. It does not force enable_grad during evaluation. A training
    caller must invoke it with gradients enabled after EACH optimizer update.
    """
    _scope(memory)
    _options(memory, observations, query, CAMERA_ORDER, False)
    if permutation_seed is not None and (type(permutation_seed) is not int or permutation_seed < 0):
        raise ValueError("permutation_seed must be None or a nonnegative Python integer")
    columns = _columns(observations)
    feature, image, attention, frame, demo = _row(columns, query, memory)
    if bool(demo):
        raise ValueError("V16 input construction is action-query only")

    # No no_grad here: the expanded arm must learn P from *past* keys as well
    # as from the current image/short query. The singleton call shape and native
    # encoding expression are identical to V15 and the frozen V13 online path.
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
        if ([entry[0] for entry in entries] != frames or [entry[2] for entry in entries] != demos):
            raise ValueError("Original images disagree with the validated framewise bank")
        addresses, contents = [], []
        for destination, source in enumerate(source_indices):
            if destination == source:
                addresses.append(bank.tokens[0, destination])
                contents.append(bank.content[0, destination])
                continue
            moved = memory.encode_bank_images(entries[source][1][None], bank.frames[:, destination],
                bank.is_demo[:, destination], camera_order=CAMERA_ORDER)
            addresses.append(moved.encoded[0])
            contents.append(moved.content[0])
        bank = DifferentialBank(torch.stack(addresses)[None], bank.frames,
            bank.is_demo, bank.valid, torch.stack(contents)[None])

    current = memory.encode_observation(feature[None], image[None], attention[None],
        frame[None], demo[None], camera_order=CAMERA_ORDER)
    memory._bank(bank, current)
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
