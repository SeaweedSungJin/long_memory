"""Standalone spatial visual-memory prototype; no trainer or policy integration.

The audited RoboMME layout is front/wrist, each a row-major 9 x 9 grid.
Callers must establish that camera order from their processor, not infer it from
162 tokens alone. Inputs are ORIGINAL post-LLM/post-VLLN/post-HAMLET features.
Each observation keeps every patch; APPEND is a rule, not learned admission.

Training must rebuild each causal prefix with the current encoder weights. An
online policy with fixed weights may retain the projected bank. Never WRITE
the image features returned by READ: ``forward`` encodes the original input,
reads the preceding bank, then appends that original observation's encoding.
There is no action, target, reward, or simulator-success input in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


CAMERA_ORDER = ("front_view", "wrist_view")
GRID_SIDE = 9
PATCHES_PER_VIEW = GRID_SIDE ** 2
PATCHES_PER_OBSERVATION = len(CAMERA_ORDER) * PATCHES_PER_VIEW


@dataclass(frozen=True)
class VisualPatchConfig:
    feature_dim: int = 2048
    hidden_dim: int = 256
    num_heads: int = 4
    num_short_tokens: int = 4
    time_scale: float = 16.0

    def __post_init__(self):
        for name in ("feature_dim", "hidden_dim", "num_heads", "num_short_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if (isinstance(self.time_scale, bool) or not isinstance(self.time_scale, (int, float))
                or not math.isfinite(self.time_scale) or self.time_scale <= 0):
            raise ValueError("time_scale must be finite and positive")


@dataclass(frozen=True)
class PatchObservation:
    """An encoding of one original observation per batch row; retains autograd."""

    features: torch.Tensor             # [B, sequence, D], never modified
    image_indices: torch.Tensor        # [B, 162], front then wrist
    encoded: torch.Tensor              # [B, 162, H]
    queries: torch.Tensor              # [B, 162, H]
    frames: torch.Tensor               # [B], raw observation frames
    is_demo: torch.Tensor              # [B]


@dataclass(frozen=True)
class PatchBank:
    """Chronological observations, with whole-observation padding per batch row.

    Tokens have shape [B, observations, 162, H]. ``valid`` may contain gaps
    when different batch rows have different observation schedules. No silent
    capacity limit, pooling, thinning, detach, or eviction is applied.
    """

    tokens: torch.Tensor
    frames: torch.Tensor               # [B, observations]
    is_demo: torch.Tensor              # [B, observations]
    valid: torch.Tensor                # [B, observations]


def _boolean_vector(value, batch, device, name):
    value = torch.as_tensor(value, device=device)
    if value.shape != (batch,) or value.dtype != torch.bool:
        raise ValueError(f"{name} must be boolean [B]")
    return value


def _frame_vector(value, batch, device):
    value = torch.as_tensor(value, device=device)
    if (value.shape != (batch,) or value.dtype == torch.bool
            or value.is_floating_point() or value.is_complex() or bool((value < 0).any())):
        raise ValueError("frames must be nonnegative integer [B]")
    return value.to(torch.long)


class VisualPatchMemoryV11(nn.Module):
    """One full-history soft READ and a zero-initialized image-position residual.

    Master weights, projected banks, and the reader stay FP32. Only the final
    residual is cast to the original feature dtype. Normalization operates in
    hidden_dim, never on the original feature_dim input. No dropout or random
    sampling occurs during encode/READ/APPEND. Reduced widths support CPU tests;
    the camera/grid layout is deliberately fixed to the audited cache contract.
    """

    def __init__(self, config: VisualPatchConfig | None = None):
        super().__init__()
        self.config = config or VisualPatchConfig()
        h, d = self.config.hidden_dim, self.config.feature_dim
        self.image_projection = nn.Linear(d, h)
        self.camera_embedding = nn.Embedding(len(CAMERA_ORDER), h)
        self.row_embedding = nn.Embedding(GRID_SIDE, h)
        self.column_embedding = nn.Embedding(GRID_SIDE, h)
        self.time_projection = nn.Linear(8, h, bias=False)
        self.bank_norm = nn.LayerNorm(h)
        self.query_norm = nn.LayerNorm(h)
        self.query_projection = nn.Linear(h, h, bias=False)
        self.key_projection = nn.Linear(h, h, bias=False)
        self.value_projection = nn.Linear(h, h, bias=False)
        self.output_projection = nn.Linear(h, d, bias=False)
        for table in (self.camera_embedding, self.row_embedding, self.column_embedding):
            nn.init.normal_(table.weight, std=.02)
        # Shared image projection plus identity Q/K starts from projected
        # content similarity, rather than two unrelated random rotations.
        # The matrices remain trainable, including V; output alone starts zero.
        for projection in (self.query_projection, self.key_projection, self.value_projection):
            nn.init.eye_(projection.weight)
        nn.init.zeros_(self.output_projection.weight)
        indices = torch.arange(PATCHES_PER_OBSERVATION)
        self.register_buffer("patch_cameras", indices // PATCHES_PER_VIEW, persistent=False)
        self.register_buffer("patch_rows", (indices % PATCHES_PER_VIEW) // GRID_SIDE, persistent=False)
        self.register_buffer("patch_columns", indices % GRID_SIDE, persistent=False)
        self.float()

    def _device(self):
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError("Visual patch reader master parameters must remain FP32")
        return self.image_projection.weight.device

    def empty_bank(self, batch_size):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        device = self._device()
        shape = (batch_size, 0)
        return PatchBank(
            torch.zeros((*shape, PATCHES_PER_OBSERVATION, self.config.hidden_dim),
                        dtype=torch.float32, device=device),
            torch.zeros(shape, dtype=torch.long, device=device),
            torch.zeros(shape, dtype=torch.bool, device=device),
            torch.zeros(shape, dtype=torch.bool, device=device))

    def encode_observation(self, features, image_mask, attention_mask, frames, is_demo,
                           *, camera_order):
        """Extract two contiguous valid 81-patch runs, excluding the short tail.

        Features may be left/right padded differently per row. Image offsets
        are recovered from masks, never absolute sequence slices. Explicit
        camera_order is a caller declaration and cannot validate image contents.
        """
        if tuple(camera_order) != CAMERA_ORDER:
            raise ValueError(f"Expected audited camera_order={CAMERA_ORDER}")
        device, c = self._device(), self.config
        if (not isinstance(features, torch.Tensor) or features.ndim != 3
                or features.shape[0] <= 0 or features.shape[2] != c.feature_dim
                or features.shape[1] <= PATCHES_PER_OBSERVATION + c.num_short_tokens
                or not features.is_floating_point() or features.device != device):
            raise ValueError("features must be floating [B, sequence, feature_dim] on the reader device")
        batch = features.shape[0]
        for name, mask in (("image_mask", image_mask), ("attention_mask", attention_mask)):
            if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                    or mask.shape != features.shape[:2] or mask.device != device):
                raise ValueError(f"{name} must be boolean [B, sequence] on the reader device")
        if not bool(attention_mask[:, -c.num_short_tokens:].all()):
            raise ValueError("The appended short tail must have valid attention")
        valid = image_mask & attention_mask
        valid = valid.clone()
        valid[:, -c.num_short_tokens:] = False
        if not bool((valid.sum(1) == PATCHES_PER_OBSERVATION).all()):
            raise ValueError("Expected exactly 162 valid image patches per observation")
        image_indices = valid.nonzero(as_tuple=False)[:, 1].reshape(batch, PATCHES_PER_OBSERVATION)
        views = image_indices.reshape(batch, len(CAMERA_ORDER), PATCHES_PER_VIEW)
        if (not bool((views.diff(dim=-1) == 1).all())
                or not bool((views[:, 1, 0] > views[:, 0, -1] + 1).all())):
            raise ValueError("Expected two separate contiguous 81-patch camera runs")
        images = features.gather(1, image_indices[..., None].expand(-1, -1, c.feature_dim))
        short = features[:, -c.num_short_tokens:]
        if not bool(torch.isfinite(images).all()) or not bool(torch.isfinite(short).all()):
            raise ValueError("Selected image patches and short summary must be finite")
        frames = _frame_vector(frames, batch, device)
        is_demo = _boolean_vector(is_demo, batch, device, "is_demo")
        with torch.autocast(device_type=device.type, enabled=False):
            t = frames.float() / c.time_scale
            time_features = torch.stack(
                [v for scale in (1., 10., 100.) for v in (torch.sin(t / scale), torch.cos(t / scale))]
                + [torch.log1p(t), is_demo.float()], dim=-1)
            position = (self.camera_embedding(self.patch_cameras)
                        + self.row_embedding(self.patch_rows)
                        + self.column_embedding(self.patch_columns))
            encoded = self.bank_norm(self.image_projection(images.float()) + position[None]
                                     + self.time_projection(time_features)[:, None])
            # Equal additive short-summary mixing at initialization; no extra
            # state encoder, gate, target supervision, or separate projection.
            summary = self.image_projection(short.float().mean(1))
            queries = self.query_projection(self.query_norm(encoded + summary[:, None]))
        if not bool(torch.isfinite(encoded).all()) or not bool(torch.isfinite(queries).all()):
            raise FloatingPointError("Nonfinite visual-memory encoding or query")
        return PatchObservation(features, image_indices, encoded, queries, frames, is_demo)

    def _bank(self, bank, observation):
        if bank is None:
            bank = self.empty_bank(observation.features.shape[0])
        if not isinstance(bank, PatchBank):
            raise ValueError("bank must be a PatchBank or None")
        batch, device = observation.features.shape[0], self._device()
        if (bank.tokens.ndim != 4 or bank.tokens.shape[0] != batch
                or bank.tokens.shape[2:] != (PATCHES_PER_OBSERVATION, self.config.hidden_dim)
                or bank.tokens.dtype != torch.float32 or bank.tokens.device != device):
            raise ValueError("Bank tokens must be FP32 [B, observations, 162, hidden_dim]")
        for name, dtype in (("frames", torch.long), ("is_demo", torch.bool), ("valid", torch.bool)):
            value = getattr(bank, name)
            if value.shape != bank.tokens.shape[:2] or value.dtype != dtype or value.device != device:
                raise ValueError(f"Bank {name} has incorrect shape/dtype/device")
        if not bool(torch.isfinite(bank.tokens[bank.valid]).all()):
            raise ValueError("Valid bank tokens must be finite")
        for row in range(batch):
            past = bank.frames[row, bank.valid[row]]
            if (bool((past < 0).any()) or bool((past.diff() <= 0).any())
                    or bool((past >= observation.frames[row]).any())):
                raise ValueError("Bank must be chronological and strictly past: past_frame < query_frame")
        return bank

    def read(self, observation: PatchObservation, bank: PatchBank | None, *, enabled=True):
        """Return a new feature tensor changing only current valid image positions.

        Empty/disabled rows return their original features exactly. Invalid
        whole observations never enter softmax. READ never changes either input
        or bank; all past encodings remain connected to the action gradient.
        """
        bank = self._bank(bank, observation)
        batch, device = observation.features.shape[0], self._device()
        if type(enabled) is bool:
            enabled = torch.full((batch,), enabled, dtype=torch.bool, device=device)
        else:
            enabled = _boolean_vector(enabled, batch, device, "enabled")
        rows = (bank.valid.any(1) & enabled).nonzero(as_tuple=False).flatten()
        if not rows.numel():
            return observation.features
        c = self.config
        with torch.autocast(device_type=device.type, enabled=False):
            tokens = bank.tokens[rows].flatten(1, 2)
            mask = bank.valid[rows].repeat_interleave(PATCHES_PER_OBSERVATION, dim=1)
            # Also neutralize padding values: zero attention weights multiplied
            # by a padded NaN value would otherwise still contaminate a row.
            tokens = torch.where(mask[..., None], tokens, 0.)
            query = observation.queries[rows]
            key, value = self.key_projection(tokens), self.value_projection(tokens)

            def heads(tensor):
                return tensor.reshape(len(rows), -1, c.num_heads, c.hidden_dim // c.num_heads).transpose(1, 2)

            recalled = F.scaled_dot_product_attention(
                heads(query), heads(key), heads(value),
                attn_mask=mask[:, None, None, :], dropout_p=0., is_causal=False)
            recalled = recalled.transpose(1, 2).reshape(len(rows), PATCHES_PER_OBSERVATION, c.hidden_dim)
            residual = self.output_projection(recalled)
        if not bool(torch.isfinite(residual).all()):
            raise FloatingPointError("Nonfinite visual-memory residual")
        positions = observation.image_indices[rows]
        original = observation.features[rows[:, None], positions]
        updated = original + residual.to(original.dtype)
        if not bool(torch.isfinite(updated).all()):
            raise FloatingPointError("Nonfinite image features after visual-memory residual cast/add")
        result = observation.features.clone()
        result[rows[:, None], positions] = updated
        return result

    def append(self, bank: PatchBank | None, observation: PatchObservation, *, valid=None):
        """Pure rule-based APPEND of ORIGINAL encodings; never detach in training."""
        bank = self._bank(bank, observation)
        batch, device = observation.features.shape[0], self._device()
        valid = (torch.ones(batch, dtype=torch.bool, device=device) if valid is None
                 else _boolean_vector(valid, batch, device, "valid"))
        tokens = torch.where(valid[:, None, None], observation.encoded, 0.)
        return PatchBank(torch.cat((bank.tokens, tokens[:, None]), dim=1),
                         torch.cat((bank.frames, observation.frames[:, None]), dim=1),
                         torch.cat((bank.is_demo, observation.is_demo[:, None]), dim=1),
                         torch.cat((bank.valid, valid[:, None]), dim=1))

    def forward(self, features, image_mask, attention_mask, frames, is_demo, bank=None,
                *, camera_order, read_enabled=True, append_valid=None):
        """READ-before-WRITE convenience step, returning (features, next_bank).

        The bank receives the encoding made before READ, even when the returned
        image positions changed. Call ``encode_observation``/``read``/``append``
        separately when an inference loop needs the action call between them.
        """
        observation = self.encode_observation(features, image_mask, attention_mask, frames,
                                              is_demo, camera_order=camera_order)
        fused = self.read(observation, bank, enabled=read_enabled)
        return fused, self.append(bank, observation, valid=append_valid)
