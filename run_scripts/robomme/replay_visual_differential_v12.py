"""Observation-only full-prefix replay for V12, preserving both address/content.

The five-column, raw chronological/demo, variable-sequence and original-short
contract reuses frozen V11 validators. Padding is inserted BEFORE the last four
short tokens. The current query retains its original sequence shape. Optional
non-reentrant checkpointing returns BOTH encoder streams without any detach.
Current-only still builds/writes the bank, but its READ ignores past content.
Cached batched versus online encoding is numerically close, not GPU bit-exact.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from run_scripts.robomme.replay_visual_patch_v11 import OBSERVATION_KEYS, _columns, _options as _v11_options, _rows
from run_scripts.robomme.visual_differential_memory_v12 import (
    CAMERA_ORDER, DifferentialBank, VisualDifferentialMemoryV12,
)


def _options(memory, observations, index, camera_order, checkpoint_encoding):
    if not isinstance(memory, VisualDifferentialMemoryV12):
        raise TypeError("memory must be VisualDifferentialMemoryV12")
    _v11_options(memory, observations, index, camera_order, checkpoint_encoding)


def _bank_from_rows(memory, rows, camera_order, checkpoint_encoding):
    if not rows:
        return memory.empty_bank(1)
    short = memory.config.num_short_tokens
    width = max(row[0].shape[0] for row in rows)
    features, images, attention = [], [], []
    for feature, image, mask, _, _ in rows:
        padding = width - feature.shape[0]
        features.append(torch.cat((feature[:-short], feature.new_zeros(padding, feature.shape[1]), feature[-short:])))
        images.append(torch.cat((image[:-short], image.new_zeros(padding), image[-short:])))
        attention.append(torch.cat((mask[:-short], mask.new_zeros(padding), mask[-short:])))
    features, images, attention = torch.stack(features), torch.stack(images), torch.stack(attention)
    frames = torch.stack([row[3] for row in rows])
    demo = torch.stack([row[4] for row in rows])

    def encode(bound_features):
        observation = memory.encode_observation(bound_features, images, attention, frames, demo,
                                                 camera_order=camera_order)
        return observation.encoded, observation.content

    address, content = (checkpoint(encode, features, use_reentrant=False, preserve_rng_state=False)
                        if checkpoint_encoding and torch.is_grad_enabled() else encode(features))
    return DifferentialBank(address[None], frames[None], demo[None],
        torch.ones((1, len(rows)), dtype=torch.bool, device=address.device), content[None])


def build_visual_differential_bank(memory, observations, stop_before, *, camera_order, checkpoint_encoding=False):
    """Encode all ORIGINAL [0,stop_before), never inspecting current/future rows."""
    _options(memory, observations, stop_before, camera_order, checkpoint_encoding)
    rows = _rows(_columns(observations), stop_before, memory) if stop_before else []
    return _bank_from_rows(memory, rows, camera_order, checkpoint_encoding)


def replay_visual_differential(memory, observations, query, *, camera_order,
                               visual_read_enabled=True, checkpoint_encoding=False):
    """Return (current_features, prior_bank); READ current q before any WRITE.

    GT/actions/state columns are never inspected. Prior and current rows use
    the current encoder weights and original short summaries; callers apply a
    separately frozen parent's short replacement only AFTER this helper returns.
    Empty and disabled READ keep current features exact in both modes.
    """
    _options(memory, observations, query, camera_order, checkpoint_encoding)
    if type(visual_read_enabled) is not bool:
        raise ValueError("visual_read_enabled must be boolean")
    rows = _rows(_columns(observations), query + 1, memory)
    bank = _bank_from_rows(memory, rows[:-1], camera_order, checkpoint_encoding)
    feature, image, attention, frame, demo = rows[-1]
    current = memory.encode_observation(feature[None], image[None], attention[None],
        frame[None], demo[None], camera_order=camera_order)
    return memory.read(current, bank, enabled=visual_read_enabled), bank
