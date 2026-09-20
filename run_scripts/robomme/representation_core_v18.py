"""V18 controlled representation experiments, without modifying frozen HAMLET.

A (short): frozen HAMLET short tokens supply both query and stored event.
B (adapted_short): a *separate copy* of HAMLET's temporal Transformer gets
    attention LoRA; all past short deltas are recomputed with current weights.
    H_B = native_H + T_LoRA(quantized_m) - stopgrad(T_base(quantized_m)).
    This anchoring is necessary because the existing cache rounded normalized
    FP32 moments to BF16, whereas its stored native_H used full-precision
    normalized moments. It preserves exact A/B equality at zero LoRA without
    rerunning the VLM or pretending the rounded cache is lossless.
C (moment): query remains frozen short; only stored features become the cached
    VLLN-normalized, pre-HAMLET moment tokens.

All arms share the V7 encoder/reader and an event-count-bounded FIFO. This is
not V7's collapsing recurrent-slot writer. READ precedes the current WRITE;
neither actions nor future observations enter representation/replay methods.
Original AE safety assertions therefore remain intact. The caller supplies
``fused`` as a replacement for the cached VL feature tail to the AE bridge.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from gr00t.long_memory.replay_v7 import _validate_prefix


CORE_VERSION = "representation_v18_anchored_fifo_v1"


@dataclass(frozen=True)
class RepresentationConfigV18:
    version: str = CORE_VERSION
    feature_dim: int = 2048
    state_dim: int = 128
    num_short_tokens: int = 4
    hidden_dim: int = 256
    num_heads: int = 4
    capacity_events: int = 32
    representation: str = "short"
    gate: str = "linear"
    short_window: int = 4
    short_lora_rank: int = 8
    short_lora_alpha: float = 16.0
    time_scale: float = 16.0

    def __post_init__(self):
        if self.version != CORE_VERSION:
            raise ValueError("Unsupported V18 representation algorithm version")
        for name in ("feature_dim", "state_dim", "num_short_tokens", "hidden_dim",
                     "num_heads", "capacity_events", "short_window", "short_lora_rank"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        for name in ("short_lora_alpha", "time_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.representation not in ("short", "adapted_short", "moment"):
            raise ValueError("representation must be short, adapted_short, or moment")
        if self.gate not in ("linear", "mlp"):
            raise ValueError("gate must be linear or mlp")

    def to_dict(self):
        return asdict(self)


class ShortLoRALinearV18(nn.Module):
    """FP32 LoRA on a frozen HAMLET attention Linear, initialized to identity."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("Short attention LoRA requires an unwrapped Linear")
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.scale = float(alpha) / rank
        self.enabled = True
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        original = self.base(x)
        if not self.enabled:
            return original
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scale
        return original + delta.to(original.dtype)


def _adapter_items(transformer):
    if transformer is None:
        return []
    return [(name, module) for name, module in transformer.named_modules()
            if isinstance(module, ShortLoRALinearV18)]


class RepresentationMemoryV18(nn.Module):
    """Shared source-controlled reader with an optional copied short adapter.

    Keep this module's learned reader and LoRA parameters FP32. ``.to(device)``
    is supported; ``.to(dtype=...)`` is not. The copied frozen Transformer may
    retain BF16 weights from its original base model.
    """

    def __init__(self, config: RepresentationConfigV18, base_memory_transformer=None):
        super().__init__()
        self.config = config
        self.memory = RecurrentMemoryV7(MemoryV7Config(
            feature_dim=config.feature_dim, state_dim=config.state_dim,
            num_short_tokens=config.num_short_tokens, hidden_dim=config.hidden_dim,
            num_heads=config.num_heads, capacity=config.capacity_events,
            time_scale=config.time_scale))
        # These are inherited constants, not part of this FIFO experiment's
        # trainable model. Retaining them preserves the V7 READ implementation.
        for name in ("slot_addresses", "write_query_norm", "write_key_norm",
                     "write_output_norm", "write_ffn", "update_gate"):
            getattr(self.memory, name).requires_grad_(False)
        if config.gate == "mlp":
            # Do not change subsequent reader/AE random initializations merely
            # because this arm happens to construct an extra gate module.
            with torch.random.fork_rng(devices=[]):
                d = config.hidden_dim
                self.memory.fusion_gate = nn.Sequential(
                    nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, 1))
                nn.init.zeros_(self.memory.fusion_gate[-1].weight)
                nn.init.zeros_(self.memory.fusion_gate[-1].bias)
        self.short_transformer = None
        if config.representation == "adapted_short":
            if base_memory_transformer is None:
                raise ValueError("adapted_short requires the original loaded HAMLET memory Transformer")
            for attr, expected in (("dim", config.feature_dim), ("n_q", config.num_short_tokens),
                                   ("T", config.short_window)):
                if getattr(base_memory_transformer, attr, None) != expected:
                    raise ValueError(f"Base HAMLET Transformer {attr} does not match V18 config")
            self.short_transformer = deepcopy(base_memory_transformer).requires_grad_(False).eval()
            base_device = next(self.short_transformer.parameters()).device
            fork_devices = [base_device.index if base_device.index is not None else torch.cuda.current_device()] if base_device.type == "cuda" else []
            # The B-only adapter must not consume the AE's later initialization
            # randomness. This preserves A/B/C's common weights under one seed.
            with torch.random.fork_rng(devices=fork_devices):
                if not hasattr(self.short_transformer, "blocks") or not self.short_transformer.blocks:
                    raise ValueError("Unsupported HAMLET Transformer: missing blocks")
                for block in self.short_transformer.blocks:
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                        setattr(block.attn, name, ShortLoRALinearV18(
                            getattr(block.attn, name), config.short_lora_rank, config.short_lora_alpha))

    def train(self, mode=True):
        super().train(mode)
        # There is no Transformer dropout in original HAMLET, but enforce the
        # frozen module's inference behavior if upstream adds train-only state.
        if self.short_transformer is not None:
            self.short_transformer.eval()
        return self

    @property
    def device(self):
        return self.memory.short_projection.weight.device

    def reader_parameters(self):
        return (parameter for parameter in self.memory.parameters() if parameter.requires_grad)

    def short_parameters(self):
        for _, module in _adapter_items(self.short_transformer):
            yield module.lora_A
            yield module.lora_B

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def delta_state_dict(self):
        """Portable learned sidecar; never includes copied frozen HAMLET weights."""
        result = {f"memory.{name}": tensor.detach().clone()
                  for name, tensor in self.memory.state_dict().items()}
        result.update({f"short_transformer.{name}.{suffix}": getattr(module, suffix).detach().clone()
                       for name, module in _adapter_items(self.short_transformer)
                       for suffix in ("lora_A", "lora_B")})
        return result

    def load_delta_state_dict(self, state):
        expected = {f"memory.{name}": tensor for name, tensor in self.memory.state_dict().items()}
        expected.update({f"short_transformer.{name}.{suffix}": getattr(module, suffix)
                         for name, module in _adapter_items(self.short_transformer)
                         for suffix in ("lora_A", "lora_B")})
        if set(state) != set(expected):
            raise ValueError(f"V18 delta tensor keys differ: missing={sorted(set(expected)-set(state))}; extra={sorted(set(state)-set(expected))}")
        # Validate every entry before any mutation; a corrupt checkpoint must
        # not leave the model partially loaded.
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                raise ValueError(f"V18 delta shape/dtype mismatch: {name}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"V18 delta nonfinite tensor: {name}")
        with torch.no_grad():
            for name, value in state.items():
                expected[name].copy_(value)

    def _moments(self, moments):
        if not isinstance(moments, torch.Tensor) or moments.ndim != 3 or tuple(moments.shape[-2:]) != (self.config.num_short_tokens, self.config.feature_dim):
            raise ValueError("Normalized moments must be [T,Q,D]")
        if not moments.is_floating_point() or not bool(torch.isfinite(moments).all()):
            raise ValueError("Normalized moments must be finite floating point")
        return moments.to(device=self.device)

    @contextmanager
    def _short_adapter_mode(self, base_only):
        saved = [(module, module.enabled) for _, module in _adapter_items(self.short_transformer)]
        try:
            for module, _ in saved:
                module.enabled = not base_only
            yield
        finally:
            for module, enabled in saved:
                module.enabled = enabled

    def _transform_window(self, window, activation_checkpointing=False, *, base_only=False):
        if self.short_transformer is None:
            raise ValueError("Only adapted_short transforms moment windows")
        # RMSNorm weights may remain FP32 in some model-loading variants; the
        # attention projection, not that norm, defines Linear's input dtype.
        first = self.short_transformer.blocks[0].attn.q_proj.base.weight
        if any(p.dtype != torch.float32 for p in self.short_parameters()):
            raise TypeError("Short LoRA parameters must remain FP32")
        # The cache's VLLN ran under CUDA autocast, producing FP32 values that
        # were then saved as BF16. Both training and online delta computations
        # intentionally use the SAME BF16-quantized input, promoted back to
        # FP32 for that original CUDA autocast contract.
        use_amp = first.device.type == "cuda" and first.dtype == torch.bfloat16
        x = window.to(device=first.device, dtype=torch.bfloat16).float()
        if not use_amp:
            x = x.to(first.dtype)
        # Forward is deterministic. Gradient checkpointing saves the large
        # frozen Transformer intermediates while retaining gradients to LoRA.
        def forward(value):
            # Select base/adapted mode *inside* every invocation. Checkpoint
            # recomputation cannot accidentally inherit a later toggled flag.
            with self._short_adapter_mode(base_only), torch.autocast(
                    device_type=value.device.type, dtype=torch.bfloat16, enabled=use_amp):
                return self.short_transformer(value)[:, -self.config.num_short_tokens:]
        if activation_checkpointing and not base_only and torch.is_grad_enabled() and any(p.requires_grad for p in self.short_parameters()):
            return checkpoint(forward, x, use_reentrant=False, preserve_rng_state=False).float()
        return forward(x).float()

    def _adapt_short(self, native_short, window, activation_checkpointing=False):
        adapted = self._transform_window(window, activation_checkpointing)
        with torch.no_grad():
            reference = self._transform_window(window, base_only=True)
        return native_short.to(device=adapted.device, dtype=torch.float32) + (adapted-reference)

    def encode_prefix(self, episode, count, *, activation_checkpointing=False):
        """Encode observations [0,count); no targets/actions/future values read.

        B replays every causal K-window from the *normalized moment* cache,
        then anchors its learned delta onto the supplied native short tokens.
        Left padding repeats observation zero, exactly like HAMLET's rolling
        inference. Absolute frame numbers are metadata, not window indices.
        All historical event computations remain connected to current weights.
        """
        _validate_prefix(episode, count)
        if count == 0:
            raise ValueError("encode_prefix requires at least one observation")
        c = self.config
        if c.representation == "adapted_short":
            if "moment" not in episode or len(episode["moment"]) < count:
                raise ValueError("adapted_short requires cached normalized moments")
            moments = self._moments(episode["moment"][:count])
            indices = (torch.arange(count, device=moments.device)[:, None]
                       + torch.arange(1-c.short_window, 1, device=moments.device)[None]).clamp_min(0)
            # Process a single window at a time: bounded peak activations and
            # exact batch-size-one online/cache arithmetic. No temporal detach.
            shorts = torch.cat([self._adapt_short(episode["short"][index:index+1],
                moments[row].reshape(1, c.short_window*c.num_short_tokens, c.feature_dim),
                activation_checkpointing) for index, row in enumerate(indices)], dim=0)
        else:
            shorts = episode["short"][:count].to(device=self.device, dtype=torch.float32)
        state = episode["state"][:count]
        frames, demo = torch.as_tensor(episode["frames"][:count]), torch.as_tensor(episode["is_demo"][:count])
        queries = self.encode_event(shorts, state, frames, demo)
        if c.representation == "moment":
            if "moment" not in episode or len(episode["moment"]) < count:
                raise ValueError("moment storage requires cached normalized moments")
            stored = self.encode_event(self._moments(episode["moment"][:count]).to(torch.bfloat16), state, frames, demo)
        else:
            stored = queries
        return {"short": shorts, "query": queries, "stored": stored}

    def encode_event(self, source, state, frames, is_demo):
        return self.memory.encode(source, state, frames, is_demo)

    def initial_bank(self, batch_size=1):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return torch.zeros(batch_size, 0, self.config.hidden_dim, device=self.device, dtype=torch.float32)

    def write_fifo(self, bank, encoded):
        bank, encoded = self.memory._inputs(bank, encoded, archive=True)
        maximum = self.config.capacity_events * self.config.num_short_tokens
        if bank.shape[1] > maximum:
            raise ValueError("Input bank exceeds configured event capacity")
        return torch.cat((bank, encoded), dim=1)[:, -maximum:]

    def read_from_bank(self, short, encoded, bank, *, read_enabled=True):
        if bank.shape[1] > self.config.capacity_events*self.config.num_short_tokens:
            raise ValueError("Input bank exceeds configured event capacity")
        fused, metrics = self.memory.read(short, encoded, bank, mode="archive", memory_enabled=read_enabled)
        metrics = dict(metrics, bank_fill=fused.new_tensor(bank.shape[1] / (self.config.capacity_events*self.config.num_short_tokens)),
                       bank_events=fused.new_tensor(bank.shape[1] / self.config.num_short_tokens))
        return fused, metrics

    def _write(self, bank, encoded, *, write_policy=None, event_index=None, frame=None, is_demo=None):
        if write_policy is None:
            return self.write_fifo(bank, encoded), {"write_rate": encoded.new_ones(())}
        result, metrics = write_policy(bank, encoded, event_index=event_index, frame=frame, is_demo=is_demo)
        # Writer callbacks may select KEEP/INSERT/FIFO-REPLACE, but cannot break
        # whole-event capacity or inject nonfinite state into a later query.
        result, _ = self.memory._inputs(result, encoded, archive=True)
        if result.shape[1] > self.config.capacity_events*self.config.num_short_tokens:
            raise ValueError("Writer returned a bank exceeding event capacity")
        return result, metrics

    def replay(self, episode, decision, *, read_enabled=True, activation_checkpointing=False, write_policy: Callable | None = None):
        if type(decision) is not int or not 0 <= decision < len(episode["frames"]):
            raise ValueError("Decision observation is out of bounds")
        encoded = self.encode_prefix(episode, decision + 1, activation_checkpointing=activation_checkpointing)
        bank = self.initial_bank()
        writes = []
        for index in range(decision):
            bank, write_metrics = self._write(bank, encoded["stored"][index:index+1],
                write_policy=write_policy, event_index=index,
                frame=torch.as_tensor(episode["frames"][index]).reshape(1),
                is_demo=torch.as_tensor(episode["is_demo"][index]).reshape(1))
            if "write_rate" in write_metrics:
                writes.append(torch.as_tensor(write_metrics["write_rate"], device=self.device).float())
        short, query = encoded["short"][decision:decision+1], encoded["query"][decision:decision+1]
        fused, metrics = self.read_from_bank(short, query, bank, read_enabled=read_enabled)
        metrics.update(replayed_observations=fused.new_tensor(float(decision)),
                       write_rate=torch.stack(writes).mean() if writes else fused.new_zeros(()))
        return {"fused": fused, "short": short, "metrics": metrics, "bank": bank,
                "encoded_current": query, "stored_current": encoded["stored"][decision:decision+1]}

    def step(self, short, moment, state, frames, is_demo, *, bank=None, moment_history=None,
             read_enabled=True, write_enabled=True, write_policy: Callable | None = None,
             event_index=None):
        """One online observation; READ first, then return newly written state.

        ``moment`` is PRE-HAMLET and already VLLN-normalized. ``short`` is the
        frozen base model's POST-HAMLET token tail. The caller must reset both
        bank and moment_history at episode boundaries, including demo priming.
        """
        c = self.config
        if type(read_enabled) is not bool or type(write_enabled) is not bool:
            raise ValueError("read_enabled/write_enabled must be booleans")
        if not isinstance(short, torch.Tensor) or short.ndim != 3:
            raise ValueError("short must be [B,Q,D]")
        short = short.to(device=self.device, dtype=torch.float32)
        batch = short.shape[0]
        if bank is None:
            bank = self.initial_bank(batch)
        if c.representation == "adapted_short":
            moments = self._moments(moment)
            if moment_history is None:
                moment_history = moments.repeat(1, c.short_window, 1)
            else:
                if moment_history.shape != (batch, c.short_window*c.num_short_tokens, c.feature_dim):
                    raise ValueError("Online moment history has incompatible shape")
                moment_history = torch.cat((moment_history.to(self.device)[:, c.num_short_tokens:], moments), dim=1)
            short = self._adapt_short(short, moment_history)
        encoded = self.encode_event(short, state, frames, is_demo)
        # C's offline source was stored as BF16. Match that quantization when
        # online VLLN/autocast happens to expose an FP32 normalized moment.
        stored = self.encode_event(self._moments(moment).to(torch.bfloat16), state, frames, is_demo) if c.representation == "moment" else encoded
        fused, metrics = self.read_from_bank(short, encoded, bank, read_enabled=read_enabled)
        if write_enabled:
            bank, write_metrics = self._write(bank, stored, write_policy=write_policy,
                event_index=event_index, frame=frames, is_demo=is_demo)
            metrics.update(write_metrics)
        else:
            metrics["write_rate"] = fused.new_zeros(())
        return {"fused": fused, "short": short, "metrics": metrics, "bank": bank,
                "moment_history": moment_history, "encoded_current": encoded, "stored_current": stored}
