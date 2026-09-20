"""V12 content-only values and current-reference differential visual READ.

Address Z and content C share the V11 projection and bank LayerNorm, but only
Z receives explicit camera/grid/time/demo identities. Differential READ is
Wout(attention(Q,Kpast,Vpast) - attention(Q,Kcurrent,Vcurrent)); current_only
uses Wout(attention(Q,Kcurrent,Vcurrent)). Both bypass empty/disabled histories.
All original V11 parameter names/shapes/initialization remain exactly intact.

APPEND stores ORIGINAL address/content encodings, never READ-modified features.
Constant values cancel within floating-point tolerance; repeated nonuniform
scenes need not cancel because their address-dependent attention may differ.
No action, target, success label, learned admission or eviction is involved.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from run_scripts.robomme.visual_patch_memory_v11 import (
    CAMERA_ORDER, GRID_SIDE, PATCHES_PER_OBSERVATION, PATCHES_PER_VIEW,
    PatchBank, PatchObservation, VisualPatchConfig, VisualPatchMemoryV11, _boolean_vector,
)

READ_MODES = ("differential", "current_only")


@dataclass(frozen=True)
class VisualDifferentialConfig(VisualPatchConfig):
    """The same five validated fields/defaults as V11; no added parameters."""


@dataclass(frozen=True)
class DifferentialObservation(PatchObservation):
    """``encoded`` is address Z; ``content`` is C, each FP32 [B,162,H]."""

    content: torch.Tensor


@dataclass(frozen=True)
class DifferentialBank(PatchBank):
    """``tokens`` is address Z; ``content`` is C, each [B,T,162,H].

    The same whole-observation validity and strict chronological contract apply
    to both streams. Storage is twice V11's projected bank, without compression.
    """

    content: torch.Tensor


class VisualDifferentialMemoryV12(VisualPatchMemoryV11):
    """Shared FP32 encoder/reader, exact V11 initialization, zero image output.

    Reuses frozen V11 extraction/address/query validation and initialization.
    Content projection is recomputed from the selected ORIGINAL image patches
    to reuse that validation without changing V11; it shares the same weights
    and retains autograd. This extra projection is a compute cost, not a new
    parameter or independent content encoder. No mode-specific RNG is consumed.
    """

    def __init__(self, config: VisualDifferentialConfig | None = None, *, read_mode="differential"):
        if read_mode not in READ_MODES:
            raise ValueError(f"read_mode must be one of {READ_MODES}")
        if config is not None and not isinstance(config, VisualDifferentialConfig):
            raise TypeError("config must be VisualDifferentialConfig or None")
        super().__init__(config or VisualDifferentialConfig())
        self.read_mode = read_mode

    def empty_bank(self, batch_size):
        bank = super().empty_bank(batch_size)
        return DifferentialBank(bank.tokens, bank.frames, bank.is_demo, bank.valid, bank.tokens.clone())

    def encode_observation(self, features, image_mask, attention_mask, frames, is_demo, *, camera_order):
        original = super().encode_observation(features, image_mask, attention_mask, frames, is_demo,
                                               camera_order=camera_order)
        images = features.gather(1, original.image_indices[..., None].expand(-1, -1, self.config.feature_dim))
        with torch.autocast(device_type=features.device.type, enabled=False):
            content = self.bank_norm(self.image_projection(images.float()))
        if not bool(torch.isfinite(content).all()):
            raise FloatingPointError("Nonfinite differential content encoding")
        return DifferentialObservation(original.features, original.image_indices, original.encoded,
            original.queries, original.frames, original.is_demo, content)

    def _bank(self, bank, observation):
        if not isinstance(observation, DifferentialObservation):
            raise ValueError("Expected a DifferentialObservation with original address and content")
        content = observation.content
        if (content.shape != observation.encoded.shape or content.dtype != torch.float32
                or content.device != self._device() or not bool(torch.isfinite(content).all())):
            raise ValueError("Current content must match finite FP32 address shape/device")
        if bank is not None and not isinstance(bank, DifferentialBank):
            raise ValueError("Expected a DifferentialBank or None")
        bank = super()._bank(bank, observation)
        if (bank.content.shape != bank.tokens.shape or bank.content.dtype != torch.float32
                or bank.content.device != bank.tokens.device
                or not bool(torch.isfinite(bank.content[bank.valid]).all())):
            raise ValueError("Valid bank content must match finite FP32 address shape/device")
        return bank

    def read(self, observation, bank=None, *, enabled=True):
        """READ without mutation; both modes require at least one valid past row.

        Differential maintains both past and current gradient paths. The
        current_only arm validates/stores past rows but its output has no
        dependency on their content. Invalid padding never enters either READ.
        """
        if self.read_mode not in READ_MODES:
            raise ValueError("Invalid differential read_mode")
        bank = self._bank(bank, observation)
        batch, device = observation.features.shape[0], self._device()
        enabled = (torch.full((batch,), enabled, dtype=torch.bool, device=device) if type(enabled) is bool
                   else _boolean_vector(enabled, batch, device, "enabled"))
        rows = (bank.valid.any(1) & enabled).nonzero(as_tuple=False).flatten()
        if not rows.numel():
            return observation.features
        c = self.config
        with torch.autocast(device_type=device.type, enabled=False):
            def heads(tensor):
                return tensor.reshape(len(rows), -1, c.num_heads, c.hidden_dim // c.num_heads).transpose(1, 2)

            query = heads(observation.queries[rows])
            current = F.scaled_dot_product_attention(query,
                heads(self.key_projection(observation.encoded[rows])),
                heads(self.value_projection(observation.content[rows])), dropout_p=0., is_causal=False)
            if self.read_mode == "differential":
                mask = bank.valid[rows].repeat_interleave(PATCHES_PER_OBSERVATION, dim=1)
                addresses = torch.where(mask[..., None], bank.tokens[rows].flatten(1, 2), 0.)
                contents = torch.where(mask[..., None], bank.content[rows].flatten(1, 2), 0.)
                past = F.scaled_dot_product_attention(query, heads(self.key_projection(addresses)),
                    heads(self.value_projection(contents)), attn_mask=mask[:, None, None, :],
                    dropout_p=0., is_causal=False)
                recalled = past - current
            else:
                recalled = current
            recalled = recalled.transpose(1, 2).reshape(len(rows), PATCHES_PER_OBSERVATION, c.hidden_dim)
            residual = self.output_projection(recalled)
        if not bool(torch.isfinite(residual).all()):
            raise FloatingPointError("Nonfinite differential visual residual")
        positions = observation.image_indices[rows]
        original = observation.features[rows[:, None], positions]
        updated = original + residual.to(original.dtype)
        if not bool(torch.isfinite(updated).all()):
            raise FloatingPointError("Nonfinite image features after differential residual cast/add")
        result = observation.features.clone()
        result[rows[:, None], positions] = updated
        return result

    def append(self, bank, observation, *, valid=None):
        """Rule-based dual-stream APPEND of the ORIGINAL encoding, without detach."""
        bank = self._bank(bank, observation)
        addresses = super().append(bank, observation, valid=valid)
        content = torch.where(addresses.valid[:, -1, None, None], observation.content, 0.)
        return DifferentialBank(addresses.tokens, addresses.frames, addresses.is_demo, addresses.valid,
                                torch.cat((bank.content, content[:, None]), dim=1))
