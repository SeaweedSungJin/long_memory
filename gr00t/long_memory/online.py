"""Bounded, causal inference state for the EXISTING two-stage checkpoints.

This is an inference counterpart of replay.py, not a new storage algorithm.
Event i is admitted only after endpoint i+1 arrives. The write query uses the
PRE-event short/state and pre-write bank, exactly as training replay does.
Frozen inference weights permit storing detached K/V; training must continue
re-encoding raw events and must not use this class.
"""
from dataclasses import dataclass

import torch

from .core import EpisodicMemory


@dataclass
class Endpoint:
    short: torch.Tensor
    moment: torch.Tensor
    state: torch.Tensor
    frame: int
    passive: bool


class OnlineEpisodicBank:
    """One episode/session, FP32 new modules, original all/FIFO or hard policy."""

    def __init__(self, memory: EpisodicMemory, stride: int, policy: str):
        if policy not in ("all", "hard"):
            raise ValueError("Inference supports the trained all or hard write policy")
        if stride <= 0 or memory.training or any(p.requires_grad for p in memory.parameters()):
            raise ValueError("Online bank requires positive stride and a frozen eval memory")
        self.memory, self.stride, self.policy = memory, int(stride), policy
        self.reset()

    def reset(self):
        self.previous = None
        self.bank_ids, self.end_frames, self.keys, self.values = [], [], [], []
        self.attempted = self.accepted = self.forced = 0
        self.learned_attempted = self.learned_accepted = 0

    def _bank(self):
        ref = next(self.memory.parameters())
        cfg = self.memory.config
        keys = torch.stack(self.keys)[None] if self.keys else ref.new_empty(1, 0, cfg.key_dim)
        values = torch.stack(self.values)[None] if self.values else ref.new_empty(1, 0, cfg.value_dim)
        mask = torch.ones((1, len(self.keys)), dtype=torch.bool, device=ref.device)
        return keys, values, mask

    def _read(self, endpoint):
        return self.memory.read(endpoint.short[None], endpoint.state[None], *self._bank())

    @torch.inference_mode()
    def advance(self, short, moment, state, *, frame: int, passive: bool, actions=None):
        """Observe an endpoint, finish only the previous interval, then read.

        `actions` contains only controls actually executed since the previous
        endpoint, normalized with that endpoint's state/processor, shape [C,A].
        Passive intervals carry None (no fabricated demonstration actions).
        """
        ref = next(self.memory.parameters())
        cfg = self.memory.config
        current = Endpoint(*(torch.as_tensor(x, device=ref.device, dtype=torch.float32).detach().clone()
                             for x in (short, moment, state)), int(frame), bool(passive))
        if (current.short.ndim != 2 or current.short.shape[-1] != cfg.feature_dim
                or current.short.shape != current.moment.shape or current.short.shape[0] == 0
                or current.state.shape != (cfg.state_dim,) or current.frame < 0):
            raise ValueError("Invalid endpoint shapes/frame for the memory checkpoint")
        if not all(torch.isfinite(x).all() for x in (current.short, current.moment, current.state)):
            raise FloatingPointError("Nonfinite online endpoint")
        controls = ref.new_empty((0, cfg.action_dim)) if actions is None else torch.as_tensor(
            actions, device=ref.device, dtype=torch.float32)
        if controls.ndim != 2 or controls.shape[1] != cfg.action_dim or not torch.isfinite(controls).all():
            raise ValueError("Executed normalized actions must be finite [C, action_dim]")
        previous = self.previous
        if previous is None:
            if len(controls):
                raise ValueError("First endpoint cannot contain an unfinished previous action")
        else:
            gap = current.frame - previous.frame
            if not 0 < gap <= self.stride:
                raise ValueError("Endpoints must advance by 1..trained memory_stride frames")
            if previous.passive and len(controls):
                raise ValueError("Passive demonstration events must not have executed actions")
            if not previous.passive and (len(controls) != gap or current.passive):
                raise ValueError("Execution event needs one actual control per elapsed frame")

        written = forced = False
        probability = utility = event_novelty = None
        completed_id = None
        if previous is not None:
            padded = ref.new_zeros((1, self.stride, cfg.action_dim))
            action_mask = torch.zeros((1, self.stride), device=ref.device, dtype=torch.bool)
            padded[0, :len(controls)] = controls
            action_mask[0, :len(controls)] = True
            encoded = self.memory.encode_events({
                "short": previous.short[None], "pre_moment": previous.moment[None],
                "post_moment": current.moment[None], "state": previous.state[None],
                "next_state": current.state[None], "actions": padded, "action_mask": action_mask,
                "valid": torch.ones(1, dtype=torch.bool, device=ref.device),
            })
            completed_id = self.attempted
            self.attempted += 1
            forced = self.policy == "hard" and len(self.keys) < cfg.min_fill
            written = self.policy == "all" or forced
            if self.policy == "hard":
                # Reporting scores even during min-fill does not change admission.
                pre_read = self._read(previous)
                keys, _, mask = self._bank()
                u = self.memory.utility(encoded["event"], previous.short[None], pre_read["read"])
                n = self.memory.novelty(encoded["keys"], keys, mask)
                probability = float(self.memory.write_logits(u, n).sigmoid().item())
                utility, event_novelty = float(u.item()), float(n.item())
                if not forced:
                    self.learned_attempted += 1
                    written = probability >= cfg.write_threshold
                    self.learned_accepted += int(written)
            if written:
                self.accepted += 1
                self.forced += int(forced)
                self.bank_ids.append(completed_id)
                self.end_frames.append(current.frame)
                self.keys.append(encoded["keys"][0].clone())
                self.values.append(encoded["values"][0].clone())
                if len(self.keys) > cfg.capacity:
                    for bank in (self.bank_ids, self.end_frames, self.keys, self.values):
                        bank.pop(0)
        read = self._read(current)
        if not torch.isfinite(read["fused_short"]).all():
            raise FloatingPointError("Nonfinite long-memory fused tokens")
        self.previous = current  # Raw HAMLET short, NOT the already fused short.
        diagnostics = {
            "policy": self.policy, "frame_index": current.frame, "passive": current.passive,
            "completed_event_id": completed_id, "event_written": written, "forced_write": forced,
            "write_probability": probability, "utility": utility, "novelty": event_novelty,
            "attempted_writes": self.attempted, "accepted_writes": self.accepted,
            "forced_writes": self.forced, "learned_write_attempts": self.learned_attempted,
            "learned_write_accepts": self.learned_accepted,
            "learned_write_rate": self.learned_accepted / self.learned_attempted if self.learned_attempted else None,
            "write_rate": self.accepted / max(1, self.attempted), "bank_fill": len(self.keys),
            "bank_event_ids": list(self.bank_ids),
            "oldest_event_age": self.attempted - self.bank_ids[0] if self.bank_ids else 0,
            "oldest_event_age_frames": current.frame - self.end_frames[0] if self.end_frames else 0,
        }
        for name in ("gate_mean", "read_norm", "residual_norm", "null_weight"):
            diagnostics[name] = float(read[name].item())
        return read["fused_short"], diagnostics
