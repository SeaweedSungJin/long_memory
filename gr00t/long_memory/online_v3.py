"""Bounded online counterpart of replay_v3, for frozen action-value checkpoints.

Every event is formed from the previous endpoint, controls actually executed in
that interval, and the newly arrived endpoint. A writer chooses at that NEW
endpoint. No future controls, ground-truth actions, or teacher labels are used.
Only capacity event encodings and one raw previous endpoint are retained.
"""

from dataclasses import dataclass

import torch

from .core_v3 import ActionValueMemory
from .replay_v3 import storage_logits_from_encoded


@dataclass
class EndpointV3:
    short: torch.Tensor
    moment: torch.Tensor
    state: torch.Tensor
    frame: int
    passive: bool


class OnlineActionValueBank:
    def __init__(self, memory: ActionValueMemory, stride: int, policy: str):
        if policy not in ("all", "hard"):
            raise ValueError("V3 inference supports all or hard storage")
        if stride <= 0 or memory.training or any(p.requires_grad for p in memory.parameters()):
            raise ValueError("Online v3 requires a frozen eval memory and positive stride")
        self.memory, self.stride, self.policy = memory, int(stride), policy
        self.reset()

    def reset(self):
        self.previous = None
        self.bank_ids, self.encodings = [], {}
        self.attempted = self.accepted = self.forced = self.replaced = 0
        self.learned_attempted = self.learned_accepted = self.learned_replaced = 0

    def _read(self, endpoint):
        ref, cfg = next(self.memory.parameters()), self.memory.config
        shape = (1, 0, self.memory.num_roles, cfg.hidden_dim)
        keys = torch.stack([self.encodings[i]["keys"] for i in self.bank_ids])[None] if self.bank_ids else ref.new_empty(shape)
        values = torch.stack([self.encodings[i]["values"] for i in self.bank_ids])[None] if self.bank_ids else ref.new_empty(shape)
        mask = torch.ones((1, len(self.bank_ids)), device=ref.device, dtype=torch.bool)
        return self.memory.read(endpoint.short[None], endpoint.state[None], keys, values, mask,
                                ref.new_tensor([float(endpoint.frame)]))

    @torch.inference_mode()
    def advance(self, short, moment, state, *, frame: int, passive: bool, actions=None):
        ref, cfg = next(self.memory.parameters()), self.memory.config
        current = EndpointV3(*(torch.as_tensor(x, device=ref.device, dtype=torch.float32).detach().clone()
                               for x in (short, moment, state)), int(frame), bool(passive))
        if (current.short.ndim != 2 or current.short.shape[-1] != cfg.feature_dim
                or current.short.shape != current.moment.shape or current.short.shape[0] == 0
                or current.state.shape != (cfg.state_dim,) or current.frame < 0):
            raise ValueError("Invalid v3 endpoint shapes/frame")
        if not all(bool(torch.isfinite(x).all()) for x in (current.short, current.moment, current.state)):
            raise FloatingPointError("Nonfinite v3 online endpoint")
        controls = ref.new_empty((0, cfg.action_dim)) if actions is None else torch.as_tensor(
            actions, device=ref.device, dtype=torch.float32)
        if controls.ndim != 2 or controls.shape[1] != cfg.action_dim or not bool(torch.isfinite(controls).all()):
            raise ValueError("Executed normalized actions must be finite [C, action_dim]")
        previous = self.previous
        if previous is None:
            if len(controls):
                raise ValueError("First endpoint cannot contain a previous action")
        else:
            gap = current.frame - previous.frame
            if not 0 < gap <= self.stride:
                raise ValueError("Endpoints must advance by 1..trained memory_stride frames")
            if previous.passive and len(controls):
                raise ValueError("Passive demonstration events must not have executed actions")
            if not previous.passive and (len(controls) != gap or current.passive):
                raise ValueError("Execution event needs one actual control per elapsed frame")

        completed_id, choice, probability, victim = None, None, None, None
        written = forced = False
        option_banks = None
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
                "start_frames": ref.new_tensor([float(previous.frame)]),
                "end_frames": ref.new_tensor([float(current.frame)]),
            })
            completed_id = self.attempted
            self.attempted += 1
            row = {key: value[0].detach().clone() for key, value in encoded.items()}
            rows = {**self.encodings, completed_id: row}
            old_ids = self.bank_ids.copy()
            forced = self.policy == "hard" and len(old_ids) < cfg.min_fill
            if self.policy == "all" or forced:
                new_ids = (old_ids + [completed_id])[-cfg.capacity:]
                written = True
            else:
                result = storage_logits_from_encoded(self.memory, current.short, current.state,
                                                      completed_id, old_ids, rows)
                if not bool(torch.isfinite(result["logits"]).all()):
                    raise FloatingPointError("Nonfinite v3 storage logits")
                choice = int(result["logits"].argmax().item())
                probability = float(1 - result["logits"].softmax(-1)[0].item())
                option_banks = result["options"]
                new_ids = option_banks[choice]
                written = choice != 0
                self.learned_attempted += 1
                self.learned_accepted += int(written)
                self.learned_replaced += int(written and len(old_ids) == cfg.capacity)
            removed = [i for i in old_ids if i not in new_ids]
            victim = removed[0] if removed else None
            self.replaced += int(bool(removed))
            self.accepted += int(written)
            self.forced += int(forced)
            self.bank_ids = list(new_ids)
            # Bounded retention is essential for long simulator episodes.
            self.encodings = {i: rows[i] for i in new_ids}

        read = self._read(current)
        if not bool(torch.isfinite(read["fused_short"]).all()):
            raise FloatingPointError("Nonfinite v3 fused tokens")
        self.previous = current  # Raw HAMLET tokens; never recursively fused inputs.
        oldest_frames = current.frame - float(self.encodings[self.bank_ids[0]]["ends"].item()) if self.bank_ids else 0.0
        diagnostics = {
            "policy": self.policy, "frame_index": current.frame, "passive": current.passive,
            "completed_event_id": completed_id, "event_written": written, "forced_write": forced,
            "storage_choice": choice, "storage_options": option_banks, "replaced_event_id": victim,
            "write_probability": probability,
            "attempted_writes": self.attempted, "accepted_writes": self.accepted,
            "forced_writes": self.forced, "replaced": self.replaced,
            "replacement_writes": self.replaced, "learned_replacement_writes": self.learned_replaced,
            "learned_attempted": self.learned_attempted, "learned_accepted": self.learned_accepted,
            "learned_rejected": self.learned_attempted - self.learned_accepted,
            "learned_replaced": self.learned_replaced,
            # Existing rollout tooling expects these legacy-friendly aliases.
            "learned_write_attempts": self.learned_attempted,
            "learned_write_accepts": self.learned_accepted,
            "learned_write_rejects": self.learned_attempted - self.learned_accepted,
            "learned_write_rate": self.learned_accepted / self.learned_attempted if self.learned_attempted else None,
            "write_rate": self.accepted / max(self.attempted, 1), "bank_fill": len(self.bank_ids),
            "bank_event_ids": self.bank_ids.copy(),
            "oldest_event_age": self.attempted - self.bank_ids[0] if self.bank_ids else 0,
            "oldest_event_age_frames": oldest_frames,
        }
        for name in ("gate_mean", "read_norm", "residual_norm", "null_weight"):
            diagnostics[name] = float(read[name].item())
        return read["fused_short"], diagnostics
