"""Opt-in adapted Action Expert inference; original HAMLET/v3 stay unchanged.

Both the LoRA adapter and the event reader are a single versioned checkpoint.
Expert-only is an inference ablation of the SAME adapted expert, not a model
trained without memory. A reader_mode=none training control disables memory
automatically and is separately identified in the evaluation manifest.
"""
from __future__ import annotations

import torch

from .checkpoint_v4 import load_checkpoint_v4, v4_checkpoint_info
from .core_v3 import ActionValueMemory, MemoryV3Config
from .expert_v4 import LoRAConfig, install_expert_lora, set_expert_trainable
from .online_policy import LongMemoryPolicy
from .online_policy_v3 import LongMemoryV3Policy


class LongMemoryV4Policy(LongMemoryPolicy):
    # Keep the tested v3 causal bank/session lifecycle, not a duplicate replay.
    _session = LongMemoryV3Policy._session

    def __init__(self, base_model, memory_checkpoint=None, device="cuda:0", strict=True,
                 write_policy="checkpoint", expert_only=False):
        if write_policy not in ("checkpoint", "all"):
            raise ValueError("V4 write policy must be checkpoint or explicit all/FIFO")
        if memory_checkpoint is None and (write_policy != "checkpoint" or expert_only):
            raise ValueError("Baseline has no adapted expert or memory to override")
        info = v4_checkpoint_info(base_model, memory_checkpoint) if memory_checkpoint else None
        super().__init__(base_model, memory_checkpoint=None, device=device, strict=strict)
        self.expert_only = bool(expert_only)
        self.reader_mode = "none"
        if info is not None:
            config = info["config"]
            self.stage = config["stage"]
            self.reader_mode = config.get("train", {}).get("reader_mode", config.get("reader_mode"))
            head = self.model.action_head
            install_expert_lora(head, LoRAConfig(**config["expert"]), targets=config["expert_targets"])
            memory = ActionValueMemory(MemoryV3Config(**config["memory"]))
            load_checkpoint_v4(memory_checkpoint, memory, head)
            set_expert_trainable(head, False)
            head.eval()
            memory.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
            if not expert_only and self.reader_mode != "none":
                self.memory = memory
                self.write_policy = "all" if self.stage == 1 or write_policy == "all" else "hard"
            else:
                self.write_policy = "disabled"
        self.write_policy_override = write_policy

    def _get_action(self, observation, options=None):
        actions, info = super()._get_action(observation, options)
        # Legacy baseline diagnostic is overwritten for an adapted no-memory
        # policy so downstream readers never mistake it for original HAMLET.
        if self.stage and self.memory is None:
            info["long_memory"]["policy"] = "expert-only" if self.expert_only else "trained-no-memory-control"
        info["expert_adapted"] = bool(self.stage)
        info["reader_mode"] = self.reader_mode
        return actions, info
