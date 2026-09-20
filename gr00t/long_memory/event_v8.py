"""V8: independently encoded, timestamped events and bounded learned retrieval.

Admission and FIFO eviction are deterministic. Capacity counts whole events,
not tokens; on overflow even demonstration events may be evicted. The default
128-event bank covers the observed training-cache maximum of 90 endpoints.
Neither event encoding nor WRITE mixes previously stored content into values.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MemoryV8Config:
    feature_dim: int = 2048
    state_dim: int = 128
    num_short_tokens: int = 4
    hidden_dim: int = 256
    capacity: int = 128
    num_heads: int = 4
    time_scale: float = 16.0
    source: str = "moment"
    residual_scale: float = 0.1

    def __post_init__(self):
        for name in ("feature_dim", "state_dim", "num_short_tokens", "hidden_dim", "capacity", "num_heads"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.source not in ("moment", "short"):
            raise ValueError("source must be moment or short")
        for name in ("time_scale", "residual_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Nonfinite {name}")


def _floating(value, shape, name, device):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    _finite(value, name)
    converted = value.to(device=device, dtype=torch.float32)
    _finite(converted, name)
    return converted


def _ffn(width):
    return nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))


class _CrossReadBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.key_norm = nn.LayerNorm(width)
        self.output_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, bias=False, dropout=0.0, batch_first=True)
        self.ffn = _ffn(width)

    def forward(self, query, keys, values):
        recalled, weights = self.attention(self.query_norm(query), self.key_norm(keys), values,
                                           need_weights=True, average_attn_weights=False)
        # Center every affine FFN path. Zero memory values yield exactly zero
        # correction even after the LayerNorm and FFN biases have been learned.
        correction = recalled + (self.ffn(self.output_norm(query + recalled))
                                 - self.ffn(self.output_norm(query)))
        return query + correction, recalled, weights


class EventMemoryV8(nn.Module):
    def __init__(self, config: MemoryV8Config):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.source_norm = nn.LayerNorm(config.feature_dim)
        self.source_projection = nn.Linear(config.feature_dim, d)
        self.state_projection = nn.Linear(config.state_dim, d)
        self.encoder_norm = nn.LayerNorm(d)
        self.encoder_ffn = _ffn(d)
        self.short_norm = nn.LayerNorm(config.feature_dim)
        self.short_projection = nn.Linear(config.feature_dim, d)
        self.time_encoder = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, d))
        self.read_blocks = nn.ModuleList([_CrossReadBlock(d, config.num_heads) for _ in range(2)])
        self.fusion_gate = nn.Linear(2 * d, 1)
        self.fusion_projection = nn.Linear(d, config.feature_dim, bias=False)
        self.reconstruction_decoder = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.feature_dim))
        nn.init.zeros_(self.fusion_projection.weight)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)

    def _device(self):
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise TypeError("V8 memory parameters must stay FP32; do not cast this module to BF16/FP16")
        return self.source_projection.weight.device

    def set_stage(self, stage):
        if type(stage) is not int or stage != 1:
            raise ValueError("V8 only supports stage 1")
        self.requires_grad_(True)
        return self

    def actor_parameters(self):
        return self.parameters()

    def initial_state(self, batch_size=1, device=None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return torch.zeros(batch_size, 0, self.config.hidden_dim + 2,
                           device=self._device() if device is None else device, dtype=torch.float32)

    def encode(self, source, state, frames, is_demo):
        """Return content plus exact raw-frame/demo metadata, independently of history."""
        device, c = self._device(), self.config
        if not isinstance(source, torch.Tensor) or source.ndim != 3 or source.shape[0] < 1:
            raise ValueError("source must be nonempty [B,Q,D]")
        batch = source.shape[0]
        source = _floating(source, (batch, c.num_short_tokens, c.feature_dim), "source", device)
        state = _floating(state, (batch, c.state_dim), "state", device)
        frames, demo = torch.as_tensor(frames, device=device), torch.as_tensor(is_demo, device=device)
        if frames.shape != (batch,) or frames.dtype == torch.bool or frames.is_complex():
            raise ValueError("frames must be raw-frame indices [B]")
        _finite(frames, "frames")
        if bool(((frames < 0) | (frames > 2 ** 24) | (frames != frames.floor())).any()):
            raise ValueError("frames must be nonnegative integers <= 2**24 for exact FP32 metadata")
        if demo.shape != (batch,) or demo.dtype != torch.bool:
            raise ValueError("is_demo must be explicit boolean [B]")
        with torch.autocast(device_type=device.type, enabled=False):
            content = self.source_projection(self.source_norm(source)) + self.state_projection(state)[:, None]
            content = content + self.encoder_ffn(self.encoder_norm(content))
            metadata = torch.stack((frames.float(), demo.float()), dim=-1)[:, None].expand(-1, c.num_short_tokens, -1)
            encoded = torch.cat((content, metadata), dim=-1)
        _finite(encoded, "encoded event")
        return encoded

    def _validate_events(self, value, batch, name, *, single=False):
        c, device = self.config, self._device()
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError(f"{name} must be [B,N,d+2]")
        count = value.shape[1]
        if count % c.num_short_tokens or count > c.capacity * c.num_short_tokens:
            raise ValueError(f"{name} must contain complete events within capacity")
        if single and count != c.num_short_tokens:
            raise ValueError("encoded must contain exactly one complete event")
        value = _floating(value, (batch, count, c.hidden_dim + 2), name, device)
        if count:
            meta = value[..., -2:].reshape(batch, -1, c.num_short_tokens, 2)
            if not bool((meta == meta[:, :, :1]).all()):
                raise ValueError(f"{name} metadata must be constant across each event")
            frames, demo = meta[:, :, 0, 0], meta[:, :, 0, 1]
            if bool(((frames < 0) | (frames > 2 ** 24) | (frames != frames.floor())).any()):
                raise ValueError(f"{name} frames must be exact nonnegative raw-frame integers")
            if bool(((demo != 0) & (demo != 1)).any()):
                raise ValueError(f"{name} demo flags must be binary")
            if frames.shape[1] > 1 and (not bool((frames[:, 1:] > frames[:, :-1]).all())
                                         or bool((demo[:, 1:] > demo[:, :-1]).any())):
                raise ValueError(f"{name} events must be chronological, with demos before execution")
        return value

    def _inputs(self, bank, encoded):
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 3 or encoded.shape[0] < 1:
            raise ValueError("encoded must be nonempty [B,Q,d+2]")
        batch = encoded.shape[0]
        encoded = self._validate_events(encoded, batch, "encoded", single=True)
        bank = self._validate_events(bank, batch, "memory")
        if bank.shape[1]:
            if not bool((bank[:, -1, -2] < encoded[:, 0, -2]).all()):
                raise ValueError("Memory events must strictly precede the current observation")
            if bool((bank[:, -1, -1] < encoded[:, 0, -1]).any()):
                raise ValueError("A demonstration cannot follow an execution event")
        return bank, encoded

    def write(self, bank, encoded):
        """Append one whole event, evict oldest whole events, and never detach."""
        bank, encoded = self._inputs(bank, encoded)
        count = bank.shape[1] // self.config.num_short_tokens
        candidate = torch.cat((bank, encoded), dim=1)[:, -self.config.capacity * self.config.num_short_tokens:]
        return candidate, {"write_rate": candidate.new_ones(()), "keep_rate": candidate.new_zeros(()),
                           "evicted_events": candidate.new_tensor(float(count == self.config.capacity)),
                           "retained_events": candidate.new_tensor(float(candidate.shape[1] // self.config.num_short_tokens)),
                           "state_norm": candidate[..., :-2].norm(dim=-1).mean()}

    @staticmethod
    def _empty_metrics(reference):
        return {name: reference.new_zeros(()) for name in (
            "read_norm", "residual_norm", "fusion_gate_mean", "gate_mean", "read_entropy", "state_norm",
            "slot_cosine", "memory_tokens", "slot_rms_spread", "slot_relative_spread", "retained_events",
            "residual_relative_max")}

    def read(self, short, encoded, bank, *, mode="event", memory_enabled=True):
        if mode not in ("event", "none"):
            raise ValueError("mode must be event or none")
        if not isinstance(memory_enabled, bool):
            raise ValueError("memory_enabled must be boolean")
        c, device = self.config, self._device()
        if not isinstance(short, torch.Tensor) or short.ndim != 3 or short.shape[0] < 1:
            raise ValueError("short must be nonempty [B,Q,D]")
        short = _floating(short, (short.shape[0], c.num_short_tokens, c.feature_dim), "short", device)
        bank, encoded = self._inputs(bank, encoded)
        if bank.shape[0] != short.shape[0]:
            raise ValueError("short and memory batch sizes differ")
        if mode == "none" or not memory_enabled or bank.shape[1] == 0:
            return short, self._empty_metrics(short)
        with torch.autocast(device_type=device.type, enabled=False):
            content = bank[..., :-2]
            age = (encoded[:, :1, -2] - bank[..., -2]) / c.time_scale
            time = torch.stack([component for scale in (1.0, 10.0, 100.0)
                                for component in (torch.sin(age / scale), torch.cos(age / scale))]
                               + [torch.log1p(age), bank[..., -1]], dim=-1)
            keys = content + self.time_encoder(time)
            query = self.short_projection(self.short_norm(short)) + encoded[..., :-2]
            contextual = query
            for block in self.read_blocks:
                contextual, recalled, weights = block(contextual, keys, content)
            delta = contextual - query
            gate = torch.sigmoid(self.fusion_gate(torch.cat((query, delta), dim=-1)))
            rms = short.detach().square().mean(-1, keepdim=True).sqrt()
            residual = gate * c.residual_scale * rms * torch.tanh(self.fusion_projection(delta))
            fused = short + residual
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1).mean()
            diagnostic = content.detach()
            count = diagnostic.shape[1]
            unit = F.normalize(diagnostic, dim=-1)
            cosine = ((unit.sum(1).square().sum(-1) - unit.square().sum((1, 2)))
                      / (count * (count - 1))).mean() if count > 1 else diagnostic.new_zeros(())
            spread = (diagnostic - diagnostic.mean(1, keepdim=True)).square().mean().sqrt()
            relative_spread = spread / (diagnostic.square().mean().sqrt() + 1e-8)
        _finite(fused, "fused short")
        return fused, {"read_norm": recalled.norm(dim=-1).mean(), "residual_norm": residual.norm(dim=-1).mean(),
                       "fusion_gate_mean": gate.mean(), "gate_mean": gate.mean(), "read_entropy": entropy,
                       "state_norm": diagnostic.norm(dim=-1).mean(), "slot_cosine": cosine,
                       "slot_rms_spread": spread, "slot_relative_spread": relative_spread,
                       "memory_tokens": bank.new_tensor(float(bank.shape[1])),
                       "retained_events": bank.new_tensor(float(bank.shape[1] // c.num_short_tokens)),
                       "residual_relative_max": (residual.detach().abs() / rms.clamp_min(1e-12)).amax()}

    def reconstruction_loss(self, encoded, source, *, reduction="mean"):
        """Train-only normalized-feature MSE; targets never receive gradients.

        ``reduction='none'`` returns one scalar per independently encoded event
        so replay can average exactly the retained past events for each query.
        """
        if reduction not in ("mean", "none"):
            raise ValueError("reduction must be mean or none")
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 3 or encoded.shape[0] < 1:
            raise ValueError("encoded must be nonempty [B,Q,d+2]")
        c, device = self.config, self._device()
        encoded = self._validate_events(encoded, encoded.shape[0], "encoded", single=True)
        source = _floating(source, (encoded.shape[0], c.num_short_tokens, c.feature_dim), "source", device)
        with torch.autocast(device_type=device.type, enabled=False):
            target = F.layer_norm(source.detach(), (c.feature_dim,))
            prediction = self.reconstruction_decoder(encoded[..., :-2])
            loss = (prediction - target).square().mean((1, 2))
        _finite(loss, "storage reconstruction loss")
        return loss.mean() if reduction == "mean" else loss
