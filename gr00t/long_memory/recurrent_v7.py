"""V7: causal short-token READ/WRITE with a persistent latent state.

The bank is 64 *latent slots*, not 64 independently preserved events.  Its
contents are explicit tensors returned to the caller: there is no hidden
session state, in-place update, action input, or truncation/detach here.
See docs/LONG_MEMORY_V7_RECURRENT_DESIGN.md for the research protocol.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MemoryV7Config:
    feature_dim: int = 2048
    state_dim: int = 128
    num_short_tokens: int = 4
    hidden_dim: int = 256
    capacity: int = 64
    num_heads: int = 4
    time_scale: float = 16.0
    update_init: float = 0.02

    def __post_init__(self):
        for name in ("feature_dim", "state_dim", "num_short_tokens", "hidden_dim", "capacity", "num_heads"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        for name in ("time_scale", "update_init"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.time_scale <= 0 or not 0 < self.update_init < 1:
            raise ValueError("Require time_scale > 0 and 0 < update_init < 1")


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Nonfinite {name}")


def _floating(value, shape, name, device):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    _finite(value, name)
    return value.to(device=device, dtype=torch.float32)


def _ffn(width):
    return nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))


class RecurrentMemoryV7(nn.Module):
    """One shared bias-free MHA, direction-specific norms/FFNs and two gates.

    Use ``read(H, X, M)`` before ``write(M, X)``.  ``archive`` is an explicitly
    unbounded information-preservation control over the *same projected short
    tokens*; it does not claim a storage/compute budget equal to this bank.
    """

    def __init__(self, config: MemoryV7Config):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.short_norm = nn.LayerNorm(config.feature_dim)
        self.short_projection = nn.Linear(config.feature_dim, d)
        self.short_token_ids = nn.Parameter(torch.empty(config.num_short_tokens, d))
        self.state_projection = nn.Linear(config.state_dim, d)
        self.time_encoder = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, d))
        self.slot_addresses = nn.Parameter(torch.empty(config.capacity, d))
        self.attention = nn.MultiheadAttention(d, config.num_heads, bias=False, dropout=0.0, batch_first=True)
        self.read_query_norm = nn.LayerNorm(d)
        self.read_key_norm = nn.LayerNorm(d)
        self.write_query_norm = nn.LayerNorm(d)
        self.write_key_norm = nn.LayerNorm(d)
        self.read_output_norm = nn.LayerNorm(d)
        self.write_output_norm = nn.LayerNorm(d)
        self.read_ffn = _ffn(d)
        self.write_ffn = _ffn(d)
        self.update_gate = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, max(1, d // 2)),
                                         nn.SiLU(), nn.Linear(max(1, d // 2), 1))
        self.fusion_gate = nn.Linear(2 * d, 1)
        self.fusion_projection = nn.Linear(d, config.feature_dim, bias=False)
        nn.init.normal_(self.short_token_ids, std=0.02)
        nn.init.normal_(self.slot_addresses, std=0.02)
        nn.init.zeros_(self.update_gate[-1].weight)
        nn.init.constant_(self.update_gate[-1].bias, math.log(config.update_init / (1 - config.update_init)))
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)
        nn.init.zeros_(self.fusion_projection.weight)

    def _device(self):
        if self.slot_addresses.dtype != torch.float32:
            raise TypeError("V7 memory parameters must stay FP32; do not cast this module to BF16/FP16")
        return self.slot_addresses.device

    def set_stage(self, stage):
        if type(stage) is not int or stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        self.requires_grad_(stage == 1)
        return self

    def actor_parameters(self):
        return self.parameters()

    def initial_state(self, batch_size=1, device=None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return torch.zeros(batch_size, self.config.capacity, self.config.hidden_dim,
                           device=self._device() if device is None else device, dtype=torch.float32)

    def encode(self, short, state, frames, is_demo):
        """Encode only current observed short/state/time; preserve all Q tokens."""
        device, c = self._device(), self.config
        if not isinstance(short, torch.Tensor) or short.ndim != 3:
            raise ValueError("short must be [B,Q,D]")
        batch = short.shape[0]
        if batch < 1:
            raise ValueError("Empty batches are not supported")
        short = _floating(short, (batch, c.num_short_tokens, c.feature_dim), "short", device)
        state = _floating(state, (batch, c.state_dim), "state", device)
        frames = torch.as_tensor(frames, device=device)
        demo = torch.as_tensor(is_demo, device=device)
        if frames.shape != (batch,) or frames.dtype == torch.bool or frames.is_complex():
            raise ValueError("frames must be real raw-frame indices [B]")
        _finite(frames, "frames")
        if bool((frames < 0).any()) or bool((frames != frames.floor()).any()):
            raise ValueError("frames must be nonnegative integer raw-frame indices")
        if demo.shape != (batch,) or demo.dtype != torch.bool:
            raise ValueError("is_demo must be explicit boolean [B]")
        with torch.autocast(device_type=device.type, enabled=False):
            z = frames.float() / c.time_scale
            time = torch.stack([component for scale in (1.0, 10.0, 100.0)
                                for component in (torch.sin(z / scale), torch.cos(z / scale))]
                               + [torch.log1p(z), demo.float()], dim=-1)
            encoded = (self.short_projection(self.short_norm(short)) + self.short_token_ids[None]
                       + self.state_projection(state)[:, None] + self.time_encoder(time)[:, None])
        _finite(encoded, "encoded short")
        return encoded

    def _inputs(self, state, encoded, *, archive=False):
        device, c = self._device(), self.config
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 3:
            raise ValueError("encoded short must be [B,Q,d]")
        batch = encoded.shape[0]
        encoded = _floating(encoded, (batch, c.num_short_tokens, c.hidden_dim), "encoded short", device)
        if not isinstance(state, torch.Tensor) or state.ndim != 3:
            raise ValueError("memory must be [B,N,d]")
        count = state.shape[1] if archive else c.capacity
        if archive and count % c.num_short_tokens:
            raise ValueError("archive must contain complete chronological Q-token observations")
        state = _floating(state, (batch, count, c.hidden_dim), "memory", device)
        return state, encoded

    def write(self, state, encoded):
        """Return the gated candidate without modifying ``state`` in place."""
        state, encoded = self._inputs(state, encoded)
        with torch.autocast(device_type=state.device.type, enabled=False):
            proposal, _ = self.attention(self.write_query_norm(state + self.slot_addresses[None]),
                                         self.write_key_norm(encoded), encoded, need_weights=False)
            values = proposal + self.write_ffn(self.write_output_norm(proposal))
            gate = torch.sigmoid(self.update_gate(torch.cat((state, values), dim=-1)))
            candidate = (1.0 - gate) * state + gate * values
        _finite(candidate, "WRITE candidate")
        return candidate, {"update_gate_mean": gate.mean(), "update_gate_min": gate.amin(),
                           "update_gate_max": gate.amax(), "update_norm": (candidate - state).norm(dim=-1).mean(),
                           "state_norm": candidate.norm(dim=-1).mean()}

    @staticmethod
    def _empty_metrics(reference):
        zero = reference.new_zeros(())
        return {name: zero for name in ("read_norm", "residual_norm", "fusion_gate_mean", "gate_mean",
                                        "read_entropy", "state_norm", "slot_cosine", "memory_tokens",
                                        "slot_rms_spread", "slot_relative_spread")}

    def read(self, short, encoded, state, *, mode="recurrent", memory_enabled=True):
        """Read all causal slots/tokens; no top-k, pooling of Q, or value addresses.

        An explicit disabled/empty session returns the original short tensor,
        even after learned LayerNorm/FFN biases change.  For a nonempty bank,
        the centered FFN additionally gives an exact R=0 -> residual=0 identity.
        """
        if mode not in ("recurrent", "archive", "none"):
            raise ValueError("mode must be recurrent, archive, or none")
        if not isinstance(memory_enabled, bool):
            raise ValueError("memory_enabled must be boolean")
        c, device = self.config, self._device()
        if not isinstance(short, torch.Tensor) or short.ndim != 3:
            raise ValueError("short must be [B,Q,D]")
        # Keep FP32 computation/return throughout; for standard cached FP32 H,
        # disabled read is also the exact same tensor (no affine shortcut).
        short = _floating(short, (short.shape[0], c.num_short_tokens, c.feature_dim), "short", device)
        if mode == "none" or not memory_enabled:
            return short, self._empty_metrics(short)
        state, encoded = self._inputs(state, encoded, archive=mode == "archive")
        if state.shape[0] != short.shape[0]:
            raise ValueError("short and memory batch sizes differ")
        if state.shape[1] == 0 or not bool(torch.count_nonzero(state)):
            return short, self._empty_metrics(short)
        with torch.autocast(device_type=device.type, enabled=False):
            keys = state + self.slot_addresses[None] if mode == "recurrent" else state
            recalled, weights = self.attention(self.read_query_norm(encoded), self.read_key_norm(keys), state,
                                               need_weights=True, average_attn_weights=False)
            # Center *the entire* read FFN, so its learned affine biases cannot
            # bypass the memory when recalled == 0.
            contextual = encoded + recalled
            delta = (contextual + self.read_ffn(self.read_output_norm(contextual))
                     - (encoded + self.read_ffn(self.read_output_norm(encoded))))
            gate = torch.sigmoid(self.fusion_gate(torch.cat((encoded, recalled), dim=-1)))
            residual = gate * self.fusion_projection(delta)
            fused = short + residual
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1).mean()
            # Mean off-diagonal cosine without constructing an N x N matrix.
            unit = F.normalize(state, dim=-1)
            count = state.shape[1]
            cosine = ((unit.sum(1).square().sum(-1) - unit.square().sum((1, 2)))
                      / (count * (count - 1))).mean() if count > 1 else state.new_zeros(())
            # Cosine can round to 1 in FP32 even when small slot differences
            # exist. Report their direct RMS too; neither metric proves that
            # semantic slot specialization has or has not been learned.
            diagnostic_state = state.detach()
            spread = (diagnostic_state - diagnostic_state.mean(1, keepdim=True)).square().mean().sqrt()
            relative_spread = spread / (diagnostic_state.square().mean().sqrt() + 1e-8)
        _finite(fused, "fused short")
        return fused, {"read_norm": recalled.norm(dim=-1).mean(), "residual_norm": residual.norm(dim=-1).mean(),
                       "fusion_gate_mean": gate.mean(), "gate_mean": gate.mean(), "read_entropy": entropy,
                       "state_norm": state.norm(dim=-1).mean(), "slot_cosine": cosine,
                       "slot_rms_spread": spread, "slot_relative_spread": relative_spread,
                       "memory_tokens": state.new_tensor(float(state.shape[1]))}
