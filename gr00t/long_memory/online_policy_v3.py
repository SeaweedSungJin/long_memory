"""Separate action-value v3 policy; legacy HAMLET/v1/v2 inference is unchanged.

Only the new memory construction and per-session bank differ. Observation
processing, frozen short-memory caching, executed-action normalization, action
denoising and resets reuse the existing tested causal RoboMME adapter.
"""
from __future__ import annotations

import torch

from .checkpoint_v3 import memory_v3_checkpoint_info
from .core_v3 import ActionValueMemory, MemoryV3Config
from .monitoring import load_checkpoint
from .online_policy import LongMemoryPolicy, _Session
from .online_v3 import OnlineActionValueBank


class LongMemoryV3Policy(LongMemoryPolicy):
    def __init__(self, base_model, memory_checkpoint=None, device="cuda:0", strict=True,
                 write_policy="checkpoint"):
        if write_policy not in ("checkpoint", "all"):
            raise ValueError("V3 write policy must be checkpoint or explicit all/FIFO ablation")
        if memory_checkpoint is None and write_policy != "checkpoint":
            raise ValueError("Baseline has no memory writer to override")
        # Validate the add-on BEFORE loading the expensive frozen base.
        info = memory_v3_checkpoint_info(base_model, memory_checkpoint) if memory_checkpoint else None
        super().__init__(base_model, memory_checkpoint=None, device=device, strict=strict)
        if info is not None:
            self.stage = info["config"]["stage"]
            self.memory = ActionValueMemory(MemoryV3Config(**info["config"]["memory"]))
            load_checkpoint(memory_checkpoint, self.memory)
            self.memory.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
            self.write_policy = "all" if self.stage == 1 or write_policy == "all" else "hard"
        self.write_policy_override = write_policy

    def _session(self, sid, seed, reset):
        if reset:
            self.sessions.pop(sid, None)
        if sid not in self.sessions:
            generator = torch.Generator(device=self.model.device).manual_seed(seed)
            bank = OnlineActionValueBank(self.memory, self.stride, self.write_policy) if self.memory else None
            self.sessions[sid] = _Session(generator, seed, bank)
        session = self.sessions[sid]
        if session.episode_seed != seed:
            raise ValueError("Episode seed changed without resetting its session")
        self.sessions.move_to_end(sid)
        while len(self.sessions) > self.session_cap:
            self.sessions.popitem(last=False)
        return session
