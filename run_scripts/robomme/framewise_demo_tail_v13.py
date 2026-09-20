"""Singleton original-observation encoding for both V13 visual-history arms.

Every original paired-camera frame is encoded as [1,2,81,D], then APPENDed in
strict raw-time order. This matches the online call shape; it never batches
different observations into a projection/normalization call. Canonical images
use the exact image-only Z/C expression of ordinary singleton V12 encoding.
No new parameters, state, random draw, admission rule, target or query surrogate.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from run_scripts.robomme import visual_demo_tail_bank_v13 as core
from run_scripts.robomme.replay_visual_patch_v11 import _columns, _options, _rows

REPLAY_ENCODING = "framewise"


def build_framewise_prefix(memory, observations, query, *, camera_order, include_tail,
                           sidecar=None, record=None, episode_id=None, cache_fingerprint=None,
                           sidecar_fingerprint=None, checkpoint_encoding=False):
    """Build an actual DifferentialBank; never access current/future features.

    The current row contributes scalar frame/demo metadata ONLY to the causal
    guard. Additional sidecar images require independent loader-verified IDs.
    All temporal/layout checks finish before any encoding, then each original
    image pair gets its own call. Optional nonreentrant checkpointing is per
    frame, returning both Z and C and preserving gradients to every past image.
    """
    if not isinstance(memory, core.VisualDemoTailMemoryV13):
        raise TypeError("Framewise replay requires the actual V13 bank-capable memory")
    _options(memory, observations, query, camera_order, checkpoint_encoding)
    if type(include_tail) is not bool:
        raise ValueError("include_tail must be an explicit boolean")
    columns, device = _columns(observations), memory._device()
    try:
        raw_frame = torch.as_tensor(columns["frames"][query], device=device)
        raw_demo = torch.as_tensor(columns["is_demo"][query], device=device)
    except (KeyError, IndexError) as error:
        raise ValueError("Missing current action-query metadata") from error
    if raw_frame.ndim or raw_demo.ndim:
        raise ValueError("Current frame/demo metadata must be scalar")
    current = core._frame_vector(raw_frame[None], 1, device)
    demo = core._boolean_vector(raw_demo[None], 1, device, "is_demo")
    if bool(demo[0]):
        raise ValueError("Framewise replay is action-query only")
    rows = _rows(columns, query, memory)
    if any(int(row[3]) >= int(current[0]) for row in rows):
        raise ValueError("Current/future canonical evidence is forbidden")
    entries = [(int(row[3]), core._canonical_images(row, memory), bool(row[4])) for row in rows]
    if include_tail:
        images, frames, demos, n_demo = core._sidecar_prefix(sidecar, record,
            episode_id=episode_id, cache_fingerprint=cache_fingerprint,
            sidecar_fingerprint=sidecar_fingerprint, current_frame=int(current[0]), width=memory.config.feature_dim)
        if any(bool(row[4]) != (int(row[3]) < n_demo) for row in rows):
            raise ValueError("Canonical observations contradict the demo boundary")
        expected_demo = sorted({0, *range(n_demo - 16, -1, -16)}) if n_demo else []
        if [int(row[3]) for row in rows if bool(row[4])] != expected_demo:
            raise ValueError("Missing original canonical demo priming observations")
        if rows and images.numel() and images.dtype != rows[0][0].dtype:
            raise ValueError("Canonical/sidecar feature dtypes differ")
        entries.extend((int(frames[i]), images[i].to(device=device), bool(demos[i])) for i in range(len(frames)))
    indices = [item[0] for item in entries]
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate canonical/tail raw frames")
    entries.sort(key=lambda item: item[0])
    if any(not left[2] and right[2] for left, right in zip(entries, entries[1:])):
        raise ValueError("Merged demonstrations must form a prefix")

    def encode(images, frames, demos):
        observation = memory.encode_bank_images(images, frames, demos, camera_order=camera_order)
        return observation.encoded, observation.content

    bank = None
    for frame, image, is_demo in entries:
        frame_tensor = torch.tensor([frame], dtype=torch.long, device=device)
        demo_tensor = torch.tensor([is_demo], dtype=torch.bool, device=device)
        args = (image[None], frame_tensor, demo_tensor)
        address, content = (checkpoint(encode, *args, use_reentrant=False, preserve_rng_state=False)
                            if checkpoint_encoding and torch.is_grad_enabled() else encode(*args))
        observation = core.BankImageRecord(address, content, frame_tensor, demo_tensor)
        bank = memory.append_bank_images(bank, observation)
    return memory.empty_bank(1) if bank is None else bank
