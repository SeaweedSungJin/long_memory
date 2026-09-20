"""Small differentiable modules for the two-stage episodic-memory experiment.

This file deliberately contains no dataset, checkpoint, simulator, or FIFO state.
The caller selects *completed past events*, rebuilds their embeddings with current
parameters, and calls :meth:`EpisodicMemory.read`. Recomputing selected events is
important: permanently caching detached learned keys/values would both make them
stale and cut the action-loss gradient to the event encoder.

Stage 1 trains the event/reader/fusion path with the frozen action expert's loss.
Stage 2 additionally trains utility and write heads with detached CVoM labels.
Those auxiliary heads detach their inputs here, rather than relying on every
training caller to remember the intended gradient boundary. Hard write decisions
are the trainer's responsibility and are not differentiable in this module.
"""

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MemoryConfig:
    """Dimensions come from the checkpoint/cache, not from RoboMME constants."""

    feature_dim: int
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    key_dim: int = 64
    value_dim: int = 128
    capacity: int = 32
    temperature: float = 0.2
    write_threshold: float = 0.5
    min_fill: int = 8
    residual_init: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "feature_dim", "state_dim", "action_dim", "hidden_dim",
            "key_dim", "value_dim", "capacity",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.temperature > 0:
            raise ValueError("temperature must be positive")
        if not 0 <= self.write_threshold <= 1:
            raise ValueError("write_threshold must be in [0, 1]")
        if not 0 <= self.min_fill <= self.capacity:
            raise ValueError("min_fill must be between zero and capacity")
        if not self.residual_init >= 0:
            raise ValueError("residual_init must be nonnegative")


def _expect_shape(tensor: Tensor, shape: tuple, name: str) -> None:
    if tensor.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(tensor.shape, shape)
    ):
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(tensor.shape)}")


def novelty(candidate_keys: Tensor, bank_keys: Tensor, mask: Tensor) -> Tensor:
    """Return 1 - best cosine similarity, or 1 for an empty bank.

    Inputs have shapes [B,K], [B,M,K], and [B,M]. The result is [B] in
    [0,2]. The function itself is differentiable; write_logits() explicitly
    detaches it so write supervision cannot distort the retrieval keys.
    """
    if candidate_keys.ndim != 2:
        raise ValueError("candidate_keys must have shape [B,K]")
    batch, width = candidate_keys.shape
    _expect_shape(bank_keys, (batch, None, width), "bank_keys")
    _expect_shape(mask, (batch, bank_keys.shape[1]), "mask")
    if mask.dtype != torch.bool:
        raise ValueError("bank mask must be boolean")
    if bank_keys.shape[1] == 0:
        return candidate_keys.new_ones(batch)
    # Zero padded slots before normalization, so even non-finite padding cannot
    # contaminate the similarity of valid slots.
    safe_bank = torch.where(mask.unsqueeze(-1), bank_keys, torch.zeros_like(bank_keys))
    scores = torch.einsum(
        "bk,bmk->bm", F.normalize(candidate_keys, dim=-1), F.normalize(safe_bank, dim=-1)
    )
    best = scores.masked_fill(~mask, -torch.inf).max(dim=1).values
    result = (1.0 - best).clamp(0.0, 2.0)
    return torch.where(mask.any(dim=1), result, torch.ones_like(result))


class EpisodicMemory(nn.Module):
    """Content-based event retrieval with baseline-preserving residual fusion.

    There is no position/age penalty. Consequently this is a content-memory
    prototype, not a claim to solve event order or counting. The null slot has
    fixed score zero and fixed value zero; it is always the *last* attention
    slot. Its weight can be logged to diagnose memory avoidance.
    """

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        d, s, a = config.feature_dim, config.state_dim, config.action_dim
        h, k, v = config.hidden_dim, config.key_dim, config.value_dim
        self.short_projection = nn.Linear(d, h)
        self.moment_projection = nn.Linear(d, h)
        self.state_projection = nn.Linear(s, h)
        self.action_encoder = nn.GRU(a, h, num_layers=1, batch_first=True)
        # 4 visual terms + state + state delta + action summary + action flag.
        self.event_encoder = nn.Sequential(
            nn.Linear(7 * h + 1, h), nn.SiLU(), nn.Linear(h, h), nn.LayerNorm(h)
        )
        self.query = nn.Sequential(nn.Linear(d + s, h), nn.SiLU(), nn.Linear(h, k))
        self.key = nn.Linear(h, k)
        self.value = nn.Linear(h, v)
        self.fusion = nn.Sequential(
            nn.Linear(v, h), nn.SiLU(), nn.LayerNorm(h), nn.Linear(h, d)
        )
        self.token_gate = nn.Linear(d + v, 1)
        self.utility_head = nn.Sequential(
            nn.Linear(h + d + v, h), nn.SiLU(), nn.Linear(h, 1)
        )
        self.write_head = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))
        self.reconstruction = nn.Sequential(
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d + s)
        )
        # Exactly zero residual preserves the baseline at initialization. The
        # first action backward trains this last layer; only after it changes
        # do earlier memory layers receive a nonzero action gradient. An
        # explicit small residual_init is provided only for gradient audits.
        if config.residual_init == 0:
            nn.init.zeros_(self.fusion[-1].weight)
        else:
            nn.init.normal_(self.fusion[-1].weight, std=config.residual_init)
        nn.init.zeros_(self.fusion[-1].bias)

    def _encode_actions(self, actions: Tensor, action_mask: Tensor) -> Tensor:
        """Encode the executed prefix only; passive/demo events return zeros."""
        if actions.ndim != 3:
            raise ValueError("actions must have shape [E,C,A]")
        count, steps, action_dim = actions.shape
        _expect_shape(actions, (count, steps, self.config.action_dim), "actions")
        _expect_shape(action_mask, (count, steps), "action_mask")
        if action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be boolean")
        if steps > 1 and bool((action_mask[:, 1:] & ~action_mask[:, :-1]).any()):
            raise ValueError("action_mask must contain a contiguous valid prefix")
        if count == 0 or steps == 0:
            return actions.new_zeros((count, self.config.hidden_dim))
        lengths = action_mask.long().sum(dim=1)
        # Ignore arbitrary padding values, including NaN. Only the last valid
        # output is gathered, so padded suffix controls cannot affect encoding.
        masked = torch.where(action_mask.unsqueeze(-1), actions, torch.zeros_like(actions))
        outputs, _ = self.action_encoder(masked)
        row = torch.arange(count, device=actions.device)
        summary = outputs[row, (lengths - 1).clamp(min=0)]
        return torch.where((lengths > 0).unsqueeze(-1), summary, torch.zeros_like(summary))

    def encode_events(self, events: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        """Encode completed event rows [E,...] without detaching their graph.

        Required keys: short/pre_moment/post_moment [E,Q,D], state/next_state
        [E,S], actions [E,C,A], action_mask [E,C], and valid [E]. This method
        cannot infer chronology; the cache/trainer must ensure the post-event
        observation exists no later than the current decision being trained.
        """
        names = (
            "short", "pre_moment", "post_moment", "state", "next_state",
            "actions", "action_mask", "valid",
        )
        missing = set(names) - set(events)
        if missing:
            raise ValueError(f"missing event tensors: {sorted(missing)}")
        short = events["short"]
        if short.ndim != 3 or short.shape[1] == 0:
            raise ValueError("short must have shape [E,Q,D] with Q > 0")
        count, tokens, _ = short.shape
        d, s = self.config.feature_dim, self.config.state_dim
        for name in ("short", "pre_moment", "post_moment"):
            _expect_shape(events[name], (count, tokens, d), name)
        for name in ("state", "next_state"):
            _expect_shape(events[name], (count, s), name)
        _expect_shape(events["valid"], (count,), "valid")
        if events["valid"].dtype != torch.bool:
            raise ValueError("valid must be boolean")
        if events["actions"].ndim != 3 or events["actions"].shape[0] != count:
            raise ValueError("actions must have shape [E,C,A] with matching event count")
        valid = events["valid"]

        def clean(name: str) -> Tensor:
            tensor = events[name]
            row_mask = valid.reshape((count,) + (1,) * (tensor.ndim - 1))
            return torch.where(row_mask, tensor, torch.zeros_like(tensor))

        pre = clean("pre_moment").mean(dim=1)
        post = clean("post_moment").mean(dim=1)
        action_mask = events["action_mask"]
        if action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be boolean")
        # Invalid transitions are not action-bearing events even if padded
        # metadata contains True. Still validate the raw mask's shape.
        _expect_shape(action_mask, (count, events["actions"].shape[1]), "action_mask")
        action_mask = action_mask & valid.unsqueeze(-1)
        action_summary = self._encode_actions(clean("actions"), action_mask)
        state = clean("state")
        parts = (
            self.short_projection(clean("short").mean(dim=1)),
            self.moment_projection(pre),
            self.moment_projection(post),
            self.moment_projection(post - pre),
            self.state_projection(state),
            self.state_projection(clean("next_state") - state),
            action_summary,
            action_mask.any(dim=1, keepdim=True).to(dtype=short.dtype),
        )
        event = self.event_encoder(torch.cat(parts, dim=-1))
        event = torch.where(valid.unsqueeze(-1), event, torch.zeros_like(event))
        keys = F.normalize(self.key(event), dim=-1)
        values = self.value(event)
        return {
            "event": event,
            "keys": torch.where(valid.unsqueeze(-1), keys, torch.zeros_like(keys)),
            "values": torch.where(valid.unsqueeze(-1), values, torch.zeros_like(values)),
            "valid": valid,
        }

    def read(
        self, short: Tensor, state: Tensor, keys: Tensor, values: Tensor, mask: Tensor
    ) -> Dict[str, Tensor]:
        """Read [B,M] event banks and return fused [B,Q,D] HAMLET tokens.

        weights is [B,M+1], with the null slot last. Diagnostic vectors are
        [B], so callers may average them without coupling the module to logging.
        Fully empty rows return exactly the original short tokens, even after
        the residual projection has been trained and acquired a nonzero bias.
        """
        if short.ndim != 3 or short.shape[1] == 0:
            raise ValueError("short must have shape [B,Q,D] with Q > 0")
        batch, token_count, _ = short.shape
        _expect_shape(short, (batch, token_count, self.config.feature_dim), "short")
        _expect_shape(state, (batch, self.config.state_dim), "state")
        _expect_shape(keys, (batch, None, self.config.key_dim), "keys")
        slots = keys.shape[1]
        _expect_shape(values, (batch, slots, self.config.value_dim), "values")
        _expect_shape(mask, (batch, slots), "mask")
        if mask.dtype != torch.bool:
            raise ValueError("bank mask must be boolean")
        q = F.normalize(self.query(torch.cat((short.mean(dim=1), state), dim=-1)), dim=-1)
        safe_keys = torch.where(mask.unsqueeze(-1), keys, torch.zeros_like(keys))
        safe_values = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values))
        scores = torch.einsum("bk,bmk->bm", q, F.normalize(safe_keys, dim=-1))
        scores = (scores / self.config.temperature).masked_fill(~mask, -torch.inf)
        scores = torch.cat((scores, scores.new_zeros(batch, 1)), dim=-1)
        # FP32 softmax is useful under bf16 autocast and when bank scores differ
        # substantially. Cast back before combining values.
        weights = F.softmax(scores.float(), dim=-1).to(dtype=safe_values.dtype)
        read = torch.einsum("bm,bmv->bv", weights[:, :-1], safe_values)
        expanded_read = read.unsqueeze(1).expand(-1, token_count, -1)
        gate = torch.sigmoid(self.token_gate(torch.cat((short, expanded_read), dim=-1)))
        residual = gate * self.fusion(read).unsqueeze(1)
        has_memory = mask.any(dim=1)
        residual = torch.where(
            has_memory[:, None, None], residual, torch.zeros_like(residual)
        )
        fused = short + residual
        return {
            "fused_short": fused,
            "read": read,
            "weights": weights,
            "gate_mean": gate.mean(dim=(1, 2)),
            "read_norm": read.norm(dim=-1),
            "residual_norm": residual.flatten(start_dim=1).norm(dim=-1),
            "null_weight": weights[:, -1],
        }

    def utility(self, event: Tensor, short: Tensor, read: Tensor) -> Tensor:
        """Predict nonnegative CVoM from detached event and past context."""
        if event.ndim != 2:
            raise ValueError("event must have shape [B,H]")
        batch = event.shape[0]
        _expect_shape(event, (batch, self.config.hidden_dim), "event")
        _expect_shape(short, (batch, None, self.config.feature_dim), "short")
        if short.shape[1] == 0:
            raise ValueError("utility short must contain at least one token")
        _expect_shape(read, (batch, self.config.value_dim), "read")
        inputs = torch.cat(
            (event.detach(), short.detach().mean(dim=1), read.detach()), dim=-1
        )
        return F.softplus(self.utility_head(inputs).squeeze(-1))

    def write_logits(self, utility: Tensor, novelty: Tensor) -> Tensor:
        """Two-input write head; hard thresholding occurs outside this module."""
        if utility.ndim != 1 or novelty.shape != utility.shape:
            raise ValueError("utility and novelty must have identical shape [B]")
        inputs = torch.stack((utility.detach(), novelty.detach()), dim=-1)
        return self.write_head(inputs).squeeze(-1)

    def reconstruct(self, event: Tensor) -> Dict[str, Tensor]:
        """Optional auxiliary decoder; this is reconstruction, NOT prediction.

        Post-event observation is already part of the encoded event. Targets
        should be detached frozen pooled-moment/state deltas, with appropriate
        scale normalization in the trainer. Invalid events must be excluded.
        """
        _expect_shape(event, (None, self.config.hidden_dim), "event")
        decoded = self.reconstruction(event)
        d = self.config.feature_dim
        return {"delta_moment": decoded[:, :d], "delta_state": decoded[:, d:]}

    novelty = staticmethod(novelty)
