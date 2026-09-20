"""Action-value memory v3, independent of the preserved v1/v2 checkpoints.

Three role tokens describe a *completed* situation/action/outcome event. Learned
attention pooling preserves information from the individual HAMLET moment tokens
instead of averaging them before encoding. Current HAMLET tokens query the bank
individually. Absolute raw-frame time features enter both keys and values.

Only the writer submodule consumes detached features. The event/read path is
ordinary differentiable PyTorch; training must re-encode selected raw events.
No hard storage decision, future action, or utility target enters this module's
reader. The residual is algebraically zero for a zero read, even after training
the affine LayerNorm/FFN biases (not just for an entirely empty bank).
"""

from dataclasses import dataclass
import math
import numbers
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MemoryV3Config:
    feature_dim: int
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    num_heads: int = 4
    capacity: int = 32
    min_fill: int = 4
    max_victims: int = 3
    time_scale: float = 16.0
    residual_init: float = 0.0
    temperature: float = 0.2

    def __post_init__(self):
        for name in ("feature_dim", "state_dim", "action_dim", "hidden_dim", "num_heads",
                     "capacity", "max_victims"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if (isinstance(self.min_fill, bool) or not isinstance(self.min_fill, numbers.Integral)
                or not 0 <= self.min_fill <= self.capacity):
            raise ValueError("min_fill must be an integer between zero and capacity")
        for name in ("time_scale", "temperature"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
        if (isinstance(self.residual_init, bool) or not isinstance(self.residual_init, numbers.Real)
                or not math.isfinite(self.residual_init) or self.residual_init < 0):
            raise ValueError("residual_init must be finite and nonnegative")


def _shape(x, dimensions, name):
    if x.ndim != len(dimensions) or any(b is not None and a != b for a, b in zip(x.shape, dimensions)):
        raise ValueError(f"{name}: expected {dimensions}, got {tuple(x.shape)}")


class StorageChooser(nn.Module):
    """Score KEEP/APPEND/REPLACE from information available at event completion.

The three role tokens are summarized for this small writer only. Inputs are
detached here, so expected-loss supervision cannot alter retrieval features.
Each option describes the proposed bank and the victim being removed; the
writer therefore sees opportunity cost, not just candidate novelty.
"""

    metadata_dim = 8

    def __init__(self, config):
        super().__init__()
        h = config.hidden_dim
        self.short_projection = nn.Linear(config.feature_dim, h)
        self.state_projection = nn.Linear(config.state_dim, h)
        self.scorer = nn.Sequential(nn.Linear(6 * h + self.metadata_dim, h), nn.SiLU(),
                                    nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, short, state, candidate, bank_events, option_events, victims, metadata):
        short, state, candidate = short.detach(), state.detach(), candidate.detach()
        bank_events, victims, metadata = bank_events.detach(), victims.detach(), metadata.detach()
        if not option_events:
            raise ValueError("At least the KEEP option is required")
        count = len(option_events)
        h = candidate.shape[-1]
        _shape(candidate, (h,), "candidate")
        _shape(bank_events, (None, h), "bank_events")
        _shape(victims, (count, h), "victims")
        _shape(metadata, (count, self.metadata_dim), "option metadata")
        empty = candidate.new_zeros(h)
        bank_summary = bank_events.mean(0) if len(bank_events) else empty
        summaries = torch.stack([rows.detach().mean(0) if len(rows) else empty
                                 for rows in option_events])
        context = torch.cat((self.short_projection(short).mean(0),
                             self.state_projection(state), candidate, bank_summary))
        inputs = torch.cat((context[None].expand(count, -1), summaries, victims, metadata), -1)
        return self.scorer(inputs).squeeze(-1)


class ActionValueMemory(nn.Module):
    """Three-role event memory with shallow multi-query cross-attention."""

    num_roles = 3

    def __init__(self, config: MemoryV3Config):
        super().__init__()
        self.config = config
        d, h, s, a = config.feature_dim, config.hidden_dim, config.state_dim, config.action_dim
        self.short_projection = nn.Linear(d, h)
        self.moment_projection = nn.Linear(d, h)
        self.state_projection = nn.Linear(s, h)
        self.pool_queries = nn.Parameter(torch.randn(3, h) * 0.02)
        self.situation_pool = nn.MultiheadAttention(h, config.num_heads, batch_first=True, dropout=0)
        self.outcome_pool = nn.MultiheadAttention(h, config.num_heads, batch_first=True, dropout=0)
        self.summary_pool = nn.MultiheadAttention(h, config.num_heads, batch_first=True, dropout=0)
        self.action_encoder = nn.GRU(a, h, num_layers=1, batch_first=True)
        self.action_presence = nn.Linear(1, h)
        self.role_embeddings = nn.Parameter(torch.randn(3, h) * 0.02)
        self.role_norm = nn.LayerNorm(h)
        self.role_ffn = nn.Sequential(nn.Linear(h, 2 * h), nn.SiLU(), nn.Linear(2 * h, h))
        self.role_output_norm = nn.LayerNorm(h)
        self.register_buffer("time_frequencies", torch.tensor([1.0, 0.1, 0.01, 0.001]), persistent=False)
        self.time_encoder = nn.Sequential(nn.Linear(24, h), nn.SiLU(), nn.Linear(h, h))
        self.time_key = nn.Linear(h, h, bias=False)
        self.time_value = nn.Linear(h, h, bias=False)
        self.key = nn.Linear(h, h, bias=False)
        self.value = nn.Linear(h, h, bias=False)
        self.query = nn.Linear(h, h, bias=False)
        self.query_time = nn.Linear(h, h, bias=False)
        self.read_norm = nn.LayerNorm(h)
        self.read_ffn = nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(), nn.Linear(2 * h, h))
        self.read_output_norm = nn.LayerNorm(h)
        self.fusion = nn.Linear(h, d, bias=False)
        self.token_gate = nn.Linear(2 * h, d)
        self.writer = StorageChooser(config)
        if config.residual_init == 0:
            nn.init.zeros_(self.fusion.weight)
        else:
            nn.init.normal_(self.fusion.weight, std=config.residual_init)

    def reader_parameters(self):
        return (p for name, p in self.named_parameters() if not name.startswith("writer."))

    def writer_parameters(self):
        return self.writer.parameters()

    def time_features(self, starts, ends):
        """Raw frames, not event indices: partial intervals retain their duration."""
        scale = self.config.time_scale
        values = torch.stack((starts / scale, ends / scale, (ends - starts) / scale), -1)
        phase = values[..., None] * self.time_frequencies
        return self.time_encoder(torch.cat((phase.sin(), phase.cos()), -1).flatten(-2))

    def _actions(self, actions, mask):
        e, steps, _ = actions.shape
        _shape(actions, (e, steps, self.config.action_dim), "actions")
        _shape(mask, (e, steps), "action_mask")
        if mask.dtype != torch.bool:
            raise ValueError("action_mask must be boolean")
        if steps > 1 and bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError("action_mask must be a contiguous valid prefix")
        if not e or not steps:
            return actions.new_zeros((e, self.config.hidden_dim))
        safe = torch.where(mask[..., None], actions, torch.zeros_like(actions))
        outputs, _ = self.action_encoder(safe)
        lengths = mask.sum(-1)
        summary = outputs[torch.arange(e, device=actions.device), (lengths - 1).clamp(min=0)]
        return torch.where((lengths > 0)[:, None], summary, torch.zeros_like(summary))

    def encode_events(self, events: Mapping[str, Tensor]):
        """Encode completed events without detaching the action-loss graph.

        Inputs match replay.event_inputs plus start_frames/end_frames [E]. The
        action mask identifies real executed prefixes; passive events have none.
        """
        names = ("short", "pre_moment", "post_moment", "state", "next_state", "actions",
                 "action_mask", "valid", "start_frames", "end_frames")
        missing = set(names) - set(events)
        if missing:
            raise ValueError(f"Missing event tensors: {sorted(missing)}")
        short, valid = events["short"], events["valid"]
        _shape(short, (None, None, self.config.feature_dim), "short")
        e, q, _ = short.shape
        if q == 0:
            raise ValueError("Events require nonempty HAMLET tokens")
        _shape(valid, (e,), "valid")
        if valid.dtype != torch.bool:
            raise ValueError("valid must be boolean")
        for name in ("pre_moment", "post_moment"):
            _shape(events[name], (e, q, self.config.feature_dim), name)
        for name in ("state", "next_state"):
            _shape(events[name], (e, self.config.state_dim), name)
        for name in ("start_frames", "end_frames"):
            _shape(events[name], (e,), name)
        _shape(events["actions"], (e, None, self.config.action_dim), "actions")
        _shape(events["action_mask"], (e, events["actions"].shape[1]), "action_mask")
        if events["action_mask"].dtype != torch.bool:
            raise ValueError("action_mask must be boolean")

        def clean(name):
            x = events[name]
            return torch.where(valid.reshape((e,) + (1,) * (x.ndim - 1)), x, torch.zeros_like(x))

        starts, ends = clean("start_frames"), clean("end_frames")
        if bool((((starts < 0) | (ends <= starts)) & valid).any()):
            raise ValueError("Valid event frame endpoints must be nonnegative and increasing")
        if e == 0:
            roles = short.new_empty(0, self.num_roles, self.config.hidden_dim)
            return {"event": short.new_empty(0, self.config.hidden_dim), "tokens": roles,
                    "keys": roles, "values": roles, "valid": valid, "starts": starts, "ends": ends}
        pre, post = clean("pre_moment"), clean("post_moment")
        state, next_state = clean("state"), clean("next_state")
        situation = torch.cat((self.short_projection(clean("short")), self.moment_projection(pre),
                               self.state_projection(state)[:, None]), 1)
        outcome = torch.cat((self.moment_projection(post), self.moment_projection(post - pre),
                             self.state_projection(next_state)[:, None],
                             self.state_projection(next_state - state)[:, None]), 1)
        pool = self.pool_queries[None].expand(e, -1, -1)
        situation_role = self.situation_pool(pool[:, :1], situation, situation, need_weights=False)[0][:, 0]
        outcome_role = self.outcome_pool(pool[:, 1:2], outcome, outcome, need_weights=False)[0][:, 0]
        mask = events["action_mask"] & valid[:, None]
        action_role = self._actions(clean("actions"), mask)
        action_role = action_role + self.action_presence(mask.any(-1, keepdim=True).to(short.dtype))
        roles = self.role_norm(torch.stack((situation_role, action_role, outcome_role), 1)
                               + self.role_embeddings[None])
        roles = self.role_output_norm(roles + self.role_ffn(roles))
        summary = self.summary_pool(pool[:, 2:3], roles, roles, need_weights=False)[0][:, 0]
        # The summary is also part of the READ path. If it were used only by
        # the detached writer, its pooling parameters would never receive any
        # gradient and the writer would consume a permanently random summary.
        contextual_roles = roles + summary[:, None]
        times = self.time_features(starts, ends)
        keys = self.key(contextual_roles) + self.time_key(times)[:, None]
        values = self.value(contextual_roles) + self.time_value(times)[:, None]
        clean_roles = lambda x: torch.where(valid[:, None, None], x, torch.zeros_like(x))
        return {"event": torch.where(valid[:, None], summary, torch.zeros_like(summary)),
                "tokens": clean_roles(roles), "keys": clean_roles(keys), "values": clean_roles(values),
                "valid": valid, "starts": starts, "ends": ends}

    def read(self, short, state, keys, values, mask, frame):
        """Read [B,M,3,H]; event_scores [B,M] align with caller bank IDs.

        Each role receives a -log(3) prior so an event is not automatically three
        times as likely as the null option merely because it has three tokens.
        Scores and softmax are FP32, and invalid padding is zeroed before use.
        """
        cfg = self.config
        _shape(short, (None, None, cfg.feature_dim), "short")
        b, q, _ = short.shape
        if q == 0:
            raise ValueError("Read requires nonempty query tokens")
        _shape(state, (b, cfg.state_dim), "state")
        _shape(keys, (b, None, self.num_roles, cfg.hidden_dim), "keys")
        m = keys.shape[1]
        _shape(values, keys.shape, "values")
        _shape(mask, (b, m), "mask")
        _shape(frame, (b,), "frame")
        if mask.dtype != torch.bool:
            raise ValueError("mask must be boolean")
        safe_keys = torch.where(mask[:, :, None, None], keys, torch.zeros_like(keys))
        safe_values = torch.where(mask[:, :, None, None], values, torch.zeros_like(values))
        context = self.short_projection(short) + self.state_projection(state)[:, None]
        current_time = self.time_features(frame, frame)
        query = self.query(context) + self.query_time(current_time)[:, None]
        heads, width = cfg.num_heads, cfg.hidden_dim // cfg.num_heads
        query_heads = query.reshape(b, q, heads, width).transpose(1, 2)
        key_heads = safe_keys.reshape(b, m * self.num_roles, heads, width).transpose(1, 2)
        value_heads = safe_values.reshape(b, m * self.num_roles, heads, width).transpose(1, 2)
        # Cosine scores stabilize small-model FP32 training and bound logits.
        raw = torch.matmul(F.normalize(query_heads.float(), dim=-1),
                           F.normalize(key_heads.float(), dim=-1).transpose(-2, -1)) / cfg.temperature
        role_scores = raw.reshape(b, heads, q, m, self.num_roles) - math.log(self.num_roles)
        role_scores = role_scores.masked_fill(~mask[:, None, None, :, None], -torch.inf)
        event_scores = role_scores.logsumexp(-1).mean(dim=(1, 2))
        scores = role_scores.flatten(-2)
        scores = torch.cat((scores, scores.new_zeros(b, heads, q, 1)), -1)
        probabilities = scores.softmax(-1)
        read = torch.matmul(probabilities[..., :-1].to(value_heads.dtype), value_heads)
        read = read.transpose(1, 2).reshape(b, q, cfg.hidden_dim)

        def block(x):
            y = self.read_norm(x)
            return self.read_output_norm(y + self.read_ffn(y))

        # Subtract the SAME current-only function, including all learned biases.
        # This rules out the v1/v2 fusion(0) current-only-adapter shortcut.
        delta = block(context + read) - block(context)
        gate = torch.sigmoid(self.token_gate(torch.cat((context, read), -1)))
        residual = gate * self.fusion(delta)
        residual = torch.where(mask.any(-1)[:, None, None], residual, torch.zeros_like(residual))
        fused = short + residual
        event_weights = probabilities[..., :-1].reshape(b, heads, q, m, self.num_roles).sum(-1).mean(1)
        null = probabilities[..., -1].mean(1)
        weights = torch.cat((event_weights, null[..., None]), -1)
        return {"fused_short": fused, "read": read, "weights": weights,
                "event_scores": event_scores, "null_score": scores.new_zeros(b, 1),
                "gate_mean": gate.mean((1, 2)), "read_norm": read.flatten(1).norm(dim=-1),
                "residual_norm": residual.flatten(1).norm(dim=-1), "null_weight": null.mean(-1)}
