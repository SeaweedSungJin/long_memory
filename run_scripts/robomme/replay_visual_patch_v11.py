"""Observation-only cached replay for the standalone, unadopted V11 prototype.

Only five columns are accessed: features, image_masks, attention_masks, frames,
and is_demo. Each row is an ORIGINAL pre-visual-READ observation, including its
original four-token HAMLET tail. No Action Expert, archive-short replacement,
state, action, target, or loss is involved. Extra mapping columns are ignored.

All cached observations [0, q) are retained, including demos. Past rows are
padded BEFORE their final short tail and encoded as one batch; the current row
is never padded or written. A direct PatchBank avoids repeated growing APPEND
copies/scans. This is a single-episode helper, not an episode-batch trainer.

Rebuild with current weights after each optimizer update. No tensor is detached;
optional non-reentrant activation checkpointing preserves full-prefix gradients.
Batched encoding is numerically close to per-row encoding, not promised GPU
bitwise equivalence. This module does not establish online or robot performance.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.utils.checkpoint import checkpoint

from run_scripts.robomme.visual_patch_memory_v11 import (
    CAMERA_ORDER, PatchBank, VisualPatchMemoryV11,
)


OBSERVATION_KEYS = ("features", "image_masks", "attention_masks", "frames", "is_demo")


def _options(memory, observations, index, camera_order, checkpoint_encoding):
    if not isinstance(memory, VisualPatchMemoryV11):
        raise TypeError("memory must be VisualPatchMemoryV11")
    if not isinstance(observations, Mapping):
        raise TypeError("observations must be a mapping of original observation columns")
    if type(index) is not int or index < 0:
        raise ValueError("Observation index must be a nonnegative Python integer")
    if tuple(camera_order) != CAMERA_ORDER:
        raise ValueError(f"Expected audited camera_order={CAMERA_ORDER}")
    if type(checkpoint_encoding) is not bool:
        raise ValueError("checkpoint_encoding must be boolean")


def _columns(observations):
    # Never enumerate or inspect extra columns, especially action/target/state.
    try:
        return {key: observations[key] for key in OBSERVATION_KEYS}
    except KeyError as error:
        raise ValueError(f"Missing observation column: {error.args[0]}") from error


def _row(columns, index, memory):
    # Integer indexing only: even column lengths and all future metadata remain
    # unread. This also supports lazy or guarded observation-only mappings.
    try:
        feature, image, attention, frame, demo = (columns[key][index] for key in OBSERVATION_KEYS)
    except (IndexError, KeyError) as error:
        raise ValueError(f"Incomplete observation prefix at index {index}") from error
    c = memory.config
    if (not isinstance(feature, torch.Tensor) or feature.ndim != 2
            or feature.shape[1] != c.feature_dim or not feature.is_floating_point()
            or feature.shape[0] <= c.num_short_tokens):
        raise ValueError("Each feature row must be floating [sequence, feature_dim]")
    for name, mask in (("image_masks", image), ("attention_masks", attention)):
        if (not isinstance(mask, torch.Tensor) or mask.shape != feature.shape[:1]
                or mask.dtype != torch.bool):
            raise ValueError(f"Each {name} row must be boolean [sequence]")
    device = memory.image_projection.weight.device
    frame = torch.as_tensor(frame, device=device)
    demo = torch.as_tensor(demo, device=device)
    if (frame.ndim != 0 or frame.dtype == torch.bool or frame.is_floating_point()
            or frame.is_complex() or bool(frame < 0)):
        raise ValueError("Each frame must be a nonnegative integer scalar")
    if demo.ndim != 0 or demo.dtype != torch.bool:
        raise ValueError("Each is_demo must be an explicit boolean scalar")
    # Device transfer preserves original dtype and autograd; never detach.
    return (feature.to(device=device), image.to(device=device), attention.to(device=device),
            frame.to(torch.long), demo)


def _rows(columns, count, memory):
    rows = [_row(columns, index, memory) for index in range(count)]
    if rows:
        if any(row[0].dtype != rows[0][0].dtype for row in rows):
            raise ValueError("Selected feature rows must share one original dtype")
        frames = torch.stack([row[3] for row in rows])
        demo = torch.stack([row[4] for row in rows])
        if bool((frames.diff() <= 0).any()):
            raise ValueError("Selected frames must be strictly increasing and unique")
        if bool((~demo[:-1] & demo[1:]).any()):
            raise ValueError("Demo observations must form a prefix before execution")
    return rows


def _bank_from_rows(memory, rows, camera_order, checkpoint_encoding):
    if not rows:
        return memory.empty_bank(1)
    short = memory.config.num_short_tokens
    width = max(row[0].shape[0] for row in rows)
    features, images, attention = [], [], []
    for feature, image, mask, _, _ in rows:
        padding = width - feature.shape[0]
        # The four short tokens must remain at the final positions in every row.
        features.append(torch.cat((feature[:-short], feature.new_zeros(padding, feature.shape[1]),
                                   feature[-short:]), dim=0))
        images.append(torch.cat((image[:-short], image.new_zeros(padding), image[-short:])))
        attention.append(torch.cat((mask[:-short], mask.new_zeros(padding), mask[-short:])))
    features = torch.stack(features)
    images, attention = torch.stack(images), torch.stack(attention)
    frames = torch.stack([row[3] for row in rows])
    demo = torch.stack([row[4] for row in rows])

    def encode(bound_features):
        return memory.encode_observation(bound_features, images, attention, frames, demo,
                                         camera_order=camera_order).encoded

    tokens = (checkpoint(encode, features, use_reentrant=False, preserve_rng_state=False)
              if checkpoint_encoding and torch.is_grad_enabled() else encode(features))
    return PatchBank(tokens[None], frames[None], demo[None],
                     torch.ones((1, len(rows)), dtype=torch.bool, device=tokens.device))


def build_visual_patch_bank(memory, observations, stop_before, *, camera_order,
                            checkpoint_encoding=False):
    """Encode every original row in [0, stop_before), without reading the next.

    ``observations`` provides indexable columns named by OBSERVATION_KEYS. Each
    feature/mask row may have a different sequence length. Frames are strictly
    increasing raw integer indices; is_demo is a boolean demo-prefix marker.
    Selected tensors move to the reader's device without casting or detaching.
    The returned bank is [1, stop_before, 162, hidden_dim], including all demos.
    An empty prefix needs no observation-column access.
    """
    _options(memory, observations, stop_before, camera_order, checkpoint_encoding)
    rows = _rows(_columns(observations), stop_before, memory) if stop_before else []
    return _bank_from_rows(memory, rows, camera_order, checkpoint_encoding)


def replay_visual_patch(memory, observations, query, *, camera_order,
                        visual_read_enabled=True, checkpoint_encoding=False):
    """Return (current_features, prior_bank), READ q against all original [0,q).

    The current observation retains its original sequence layout and original
    short-summary query; no current WRITE is returned. ``visual_read_enabled``
    disables only READ, never prior encoding/storage. Online callers still use
    the separate core APPEND after acting, including with READ disabled.
    Prefix and current-input gradients remain attached, including under optional
    checkpointing. Never reuse the returned bank across optimizer updates.
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
