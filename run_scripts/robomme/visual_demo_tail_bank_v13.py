"""V13 image-only evidence ingestion; unchanged V12 parameters and READ math.

This is NOT a policy, cache extractor or trainer. Additional demo images enter
only the visual bank. No invented short tokens or unused queries are computed.
The canonical query index, current image/short conditioning, parent archive,
targets and noise are outside this module's mutation and input contract.

The sidecar loader remains responsible for disk/source hashes. The merge helper
also verifies episode/fingerprint bindings and the exact uniform demo-tail rule.
It is an ACTION-query helper: no current/future tail may be passed during primes.
Rebuild all encodings after each optimizer update; no learned tensors are cached.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from run_scripts.robomme.replay_visual_patch_v11 import OBSERVATION_KEYS, _columns, _rows
from run_scripts.robomme.visual_patch_memory_v11 import (
    CAMERA_ORDER, PATCHES_PER_OBSERVATION, PATCHES_PER_VIEW, _boolean_vector, _frame_vector,
)
from run_scripts.robomme.visual_differential_memory_v12 import (
    DifferentialBank, VisualDifferentialConfig, VisualDifferentialMemoryV12,
)

SIDECAR_KEYS = frozenset(("episode_id", "cache_fingerprint", "sidecar_fingerprint", "images", "frames", "is_demo"))


@dataclass(frozen=True)
class BankImageRecord:
    """One original observation per batch row, without a short/query surrogate."""

    encoded: torch.Tensor             # Z: FP32 [B,162,H]
    content: torch.Tensor             # C: FP32 [B,162,H]
    frames: torch.Tensor              # int64 [B]
    is_demo: torch.Tensor             # bool [B]


class VisualDemoTailMemoryV13(VisualDifferentialMemoryV12):
    """Inherits initialization, state_dict, ordinary encode, and READ unchanged.

    No constructor override, new parameters/buffers, RNG draw, or admission
    model. ``append_bank_images`` is a transaction: return a new bank only after
    validating every input. A caller may commit that bank after other work.
    """

    def encode_bank_images(self, images, frames, is_demo, *, camera_order):
        if tuple(camera_order) != CAMERA_ORDER:
            raise ValueError(f"Expected audited camera_order={CAMERA_ORDER}")
        device, config = self._device(), self.config
        if (not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] <= 0
                or images.shape[1:] != (2, PATCHES_PER_VIEW, config.feature_dim)
                or not images.is_floating_point() or images.device != device
                or not bool(torch.isfinite(images).all())):
            raise ValueError("images must be finite floating [B,2,81,D] on the reader device")
        batch = images.shape[0]
        frames = _frame_vector(frames, batch, device)
        is_demo = _boolean_vector(is_demo, batch, device, "is_demo")
        images = images.reshape(batch, PATCHES_PER_OBSERVATION, config.feature_dim)
        with torch.autocast(device_type=device.type, enabled=False):
            # Preserve frozen V12's exact expression/order, including its shared
            # projection being evaluated separately for Z and C. No short/query.
            t = frames.float() / config.time_scale
            time_features = torch.stack(
                [v for scale in (1., 10., 100.) for v in (torch.sin(t / scale), torch.cos(t / scale))]
                + [torch.log1p(t), is_demo.float()], dim=-1)
            position = (self.camera_embedding(self.patch_cameras)
                        + self.row_embedding(self.patch_rows) + self.column_embedding(self.patch_columns))
            encoded = self.bank_norm(self.image_projection(images.float()) + position[None]
                                     + self.time_projection(time_features)[:, None])
            content = self.bank_norm(self.image_projection(images.float()))
        if not bool(torch.isfinite(encoded).all()) or not bool(torch.isfinite(content).all()):
            raise FloatingPointError("Nonfinite image-only address/content encoding")
        return BankImageRecord(encoded, content, frames, is_demo)

    def _image_record(self, record):
        device, config = self._device(), self.config
        if not isinstance(record, BankImageRecord):
            raise TypeError("Expected BankImageRecord, not a query observation")
        if (not isinstance(record.encoded, torch.Tensor) or record.encoded.ndim != 3
                or record.encoded.shape[0] <= 0
                or record.encoded.shape[1:] != (PATCHES_PER_OBSERVATION, config.hidden_dim)):
            raise ValueError("Record Z must have shape [B,162,H]")
        batch = record.encoded.shape[0]
        for value in (record.encoded, record.content):
            if (not isinstance(value, torch.Tensor) or value.shape != record.encoded.shape
                    or value.dtype != torch.float32 or value.device != device
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Record Z/C must be finite FP32 with equal shape/device")
        for value, dtype, name in ((record.frames, torch.long, "frames"), (record.is_demo, torch.bool, "is_demo")):
            if not isinstance(value, torch.Tensor) or value.shape != (batch,) or value.dtype != dtype or value.device != device:
                raise ValueError(f"Record {name} has incorrect shape/dtype/device")
        if bool((record.frames < 0).any()):
            raise ValueError("Record frames must be nonnegative")
        return batch, device

    def append_bank_images(self, bank, record, *, valid=None):
        batch, device = self._image_record(record)
        valid = (torch.ones(batch, dtype=torch.bool, device=device) if valid is None
                 else _boolean_vector(valid, batch, device, "valid"))
        if bank is None:
            bank = self.empty_bank(batch)
        if not isinstance(bank, DifferentialBank):
            raise TypeError("Expected an actual V12 DifferentialBank or None")
        if (not isinstance(bank.tokens, torch.Tensor) or bank.tokens.ndim != 4 or bank.tokens.shape[0] != batch
                or bank.tokens.shape[2:] != (PATCHES_PER_OBSERVATION, self.config.hidden_dim)):
            raise ValueError("Bank tokens must have shape [B,T,162,H]")
        for value in (bank.tokens, bank.content):
            if (not isinstance(value, torch.Tensor) or value.shape != bank.tokens.shape
                    or value.dtype != torch.float32 or value.device != device):
                raise ValueError("Bank Z/C must have equal FP32 shape/device")
        for name, dtype in (("frames", torch.long), ("is_demo", torch.bool), ("valid", torch.bool)):
            value = getattr(bank, name)
            if (not isinstance(value, torch.Tensor) or value.shape != bank.tokens.shape[:2]
                    or value.dtype != dtype or value.device != device):
                raise ValueError(f"Bank {name} has incorrect shape/dtype/device")
        if not all(bool(torch.isfinite(x[bank.valid]).all()) for x in (bank.tokens, bank.content)):
            raise ValueError("Valid bank Z/C must be finite")
        for row in range(batch):
            frames, demos = bank.frames[row, bank.valid[row]], bank.is_demo[row, bank.valid[row]]
            if (bool((frames < 0).any()) or bool((frames.diff() <= 0).any())
                    or bool((frames >= record.frames[row]).any())):
                raise ValueError("APPEND requires unique chronological strictly preceding bank frames")
            if bool((~demos[:-1] & demos[1:]).any()):
                raise ValueError("Bank demonstrations must form a prefix")
            if bool(valid[row] & record.is_demo[row]) and bool((~demos).any()):
                raise ValueError("Cannot APPEND demo evidence after execution")
        address = torch.where(valid[:, None, None], record.encoded, 0.)
        content = torch.where(valid[:, None, None], record.content, 0.)
        return DifferentialBank(
            torch.cat((bank.tokens, address[:, None]), dim=1),
            torch.cat((bank.frames, record.frames[:, None]), dim=1),
            torch.cat((bank.is_demo, record.is_demo[:, None]), dim=1),
            torch.cat((bank.valid, valid[:, None]), dim=1),
            torch.cat((bank.content, content[:, None]), dim=1))


def _fingerprint(value):
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _sidecar_prefix(sidecar, record, *, episode_id, cache_fingerprint, sidecar_fingerprint, current_frame, width):
    if not isinstance(sidecar, Mapping) or set(sidecar) != SIDECAR_KEYS or not isinstance(record, Mapping):
        raise ValueError("Expected exact observation-only sidecar schema and its manifest episode record")
    if type(episode_id) is not int or episode_id < 0:
        raise ValueError("Expected nonnegative episode_id")
    for key, expected in (("episode_id", episode_id), ("cache_fingerprint", cache_fingerprint),
                          ("sidecar_fingerprint", sidecar_fingerprint)):
        if type(sidecar[key]) is not type(expected) or sidecar[key] != expected:
            raise ValueError(f"Sidecar belongs to another episode/cache/source: {key}")
    if not _fingerprint(cache_fingerprint) or not _fingerprint(sidecar_fingerprint):
        raise ValueError("Expected explicit SHA256 cache/sidecar fingerprints")
    if type(record.get("episode_id")) is not int or record["episode_id"] != episode_id:
        raise ValueError("Sidecar manifest record belongs to another episode")
    n_demo = record.get("n_demo")
    if type(n_demo) is not int or n_demo < 0:
        raise ValueError("Sidecar manifest n_demo must be a nonnegative integer")
    last = max(0, n_demo - 16) if n_demo else -1
    if type(record.get("last_canonical_demo")) is not type(last) or record["last_canonical_demo"] != last:
        raise ValueError("Sidecar last canonical demo differs from original stride16 priming")
    expected_frames = list(range(max(last + 1, n_demo - 15, 0), n_demo))
    if (not isinstance(record.get("frames"), list)
            or any(type(frame) is not int for frame in record["frames"])
            or record["frames"] != expected_frames):
        raise ValueError("Sidecar manifest frames differ from the uniform omitted demo-tail rule")
    frames, demos = sidecar["frames"], sidecar["is_demo"]
    if (not isinstance(frames, torch.Tensor) or frames.dtype != torch.long
            or frames.shape != (len(expected_frames),) or frames.tolist() != expected_frames):
        raise ValueError("Sidecar frames must exactly match chronological, unique demo-tail indices")
    if (not isinstance(demos, torch.Tensor) or demos.shape != frames.shape or demos.dtype != torch.bool
            or not bool(demos.all())):
        raise ValueError("Sidecar evidence must be explicitly passive demo images")
    if current_frame < n_demo or bool((frames >= current_frame).any()):
        raise ValueError("Action query must follow every tail image strictly; no current/future evidence")
    # Validate time/identity before even accessing potentially future image data.
    images = sidecar["images"]
    if (not isinstance(images, torch.Tensor) or images.dtype != torch.bfloat16
            or images.shape != (len(expected_frames), 2, PATCHES_PER_VIEW, width)
            or not bool(torch.isfinite(images).all())):
        raise ValueError("Sidecar images must be finite original BF16 [N,2,81,D]")
    return images, frames, demos, n_demo


def _canonical_images(row, memory):
    feature, image, attention, _, _ = row
    short = memory.config.num_short_tokens
    if not bool(attention[-short:].all()):
        raise ValueError("Canonical short tail must have valid attention")
    mask = image & attention
    mask = mask.clone()
    mask[-short:] = False
    indices = mask.nonzero(as_tuple=False).flatten()
    if len(indices) != PATCHES_PER_OBSERVATION:
        raise ValueError("Canonical observation must have exactly 162 image patches")
    views = indices.reshape(2, PATCHES_PER_VIEW)
    if not bool((views.diff(dim=-1) == 1).all()) or not bool(views[1, 0] > views[0, -1] + 1):
        raise ValueError("Canonical camera layout must contain two separate 81-patch runs")
    return feature[indices].reshape(2, PATCHES_PER_VIEW, memory.config.feature_dim)


def merged_prefix_bank(memory, observations, query, sidecar, *, record, episode_id,
                       cache_fingerprint, sidecar_fingerprint, camera_order, checkpoint_encoding=False):
    """Build an actual V12 bank from canonical [0,q) plus its entire demo tail.

    ``query`` remains the ORIGINAL cache action-query index. Only current raw
    frame/demo metadata is accessed, never current features or future rows.
    Sidecar images must all be strictly past (no silently filtered future).
    The supplied record/sidecar must come from the hash-verifying sidecar reader;
    expected bindings are independent caller values, not inferred from payload.
    A future matched canonical-only arm should use the unchanged V12 canonical
    bank builder with these same weights, NOT a fabricated empty demo sidecar.
    """
    if not isinstance(memory, VisualDemoTailMemoryV13) or not isinstance(observations, Mapping):
        raise TypeError("Expected V13 memory and original canonical observation mapping")
    if type(query) is not int or query < 0 or type(checkpoint_encoding) is not bool:
        raise ValueError("Expected nonnegative original query index and boolean checkpoint flag")
    if tuple(camera_order) != CAMERA_ORDER:
        raise ValueError(f"Expected audited camera_order={CAMERA_ORDER}")
    columns = _columns(observations)
    device = memory._device()
    try:
        raw_frame = torch.as_tensor(columns["frames"][query], device=device)
        raw_demo = torch.as_tensor(columns["is_demo"][query], device=device)
        if raw_frame.ndim != 0 or raw_demo.ndim != 0:
            raise ValueError("Current frame/demo metadata must be scalar")
        frame = _frame_vector(raw_frame[None], 1, device)
        demo = _boolean_vector(raw_demo[None], 1, device, "is_demo")
    except (IndexError, KeyError) as error:
        raise ValueError("Missing canonical action-query frame/demo metadata") from error
    if bool(demo[0]):
        raise ValueError("merged_prefix_bank is action-query only; passive primes cannot read future demo tails")
    images, tail_frames, tail_demo, n_demo = _sidecar_prefix(sidecar, record, episode_id=episode_id,
        cache_fingerprint=cache_fingerprint, sidecar_fingerprint=sidecar_fingerprint,
        current_frame=int(frame[0]), width=memory.config.feature_dim)
    rows = _rows(columns, query, memory)
    for row in rows:
        if int(row[3]) >= int(frame[0]) or bool(row[4]) != (int(row[3]) < n_demo):
            raise ValueError("Canonical history contradicts strict past/demo boundary")
    expected_demo = sorted({0, *range(n_demo - 16, -1, -16)}) if n_demo else []
    if [int(row[3]) for row in rows if bool(row[4])] != expected_demo:
        raise ValueError("Canonical demo prefix must retain all original stride16 priming endpoints")
    if rows and images.numel() and rows[0][0].dtype != images.dtype:
        raise ValueError("Canonical/sidecar original feature dtypes differ")
    entries = [(int(row[3]), _canonical_images(row, memory), row[4]) for row in rows]
    entries.extend((int(tail_frames[i]), images[i].to(device=device), tail_demo[i].to(device=device))
                   for i in range(len(tail_frames)))
    indices = [entry[0] for entry in entries]
    if len(set(indices)) != len(indices):
        raise ValueError("Canonical/sidecar duplicate raw frames; never double-write an observation")
    entries.sort(key=lambda entry: entry[0])
    if not entries:
        return memory.empty_bank(1)
    merged_images = torch.stack([entry[1] for entry in entries])
    merged_frames = torch.tensor([entry[0] for entry in entries], dtype=torch.long, device=device)
    merged_demo = torch.stack([entry[2] for entry in entries])
    if bool((~merged_demo[:-1] & merged_demo[1:]).any()):
        raise ValueError("Merged demo evidence must precede execution observations")

    def encode(bound_images):
        result = memory.encode_bank_images(bound_images, merged_frames, merged_demo, camera_order=camera_order)
        return result.encoded, result.content

    address, content = (checkpoint(encode, merged_images, use_reentrant=False, preserve_rng_state=False)
                        if checkpoint_encoding and torch.is_grad_enabled() else encode(merged_images))
    return DifferentialBank(address[None], merged_frames[None], merged_demo[None],
        torch.ones((1, len(entries)), dtype=torch.bool, device=device), content[None])
