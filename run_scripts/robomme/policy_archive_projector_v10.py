"""Portable V10 Expert adaptation on the unchanged causal V7 archive rollout.

Only construction/checkpoint loading differs. The inherited READ-before-WRITE,
per-session RNG, passive demo APPEND, short cache, decoding and cleanup remain
the original V7 implementations. READ-off keeps the SAME learned LoRA and
projector, and continues APPEND writes. V10 bundles are never disguised as V7.
"""
from __future__ import annotations

from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora
from gr00t.long_memory.hamlet import isolated_seed
from gr00t.long_memory.online_policy import LongMemoryPolicy
from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from run_scripts.robomme.checkpoint_projector_v10 import VARIANT, checkpoint_info, load_checkpoint
from run_scripts.robomme.projector_adapter_v10 import (
    assert_expert_scope, install_projector, projector_spec, set_trainable,
)

import torch


class ArchiveProjectorV10Policy(LongMemoryV7Policy):
    """V10 checkpoint constructor with V7's unchanged observation/action path.

An explicit step-zero bundle is supported for initialization diagnostics; its
reported checkpoint_step remains zero. It is not relabeled as newly trained.
The original no-adapter HAMLET baseline is a separate clean V7 policy instance.
"""

    def __init__(self, base_model, memory_checkpoint, device="cuda:0", strict=True,
                 write_policy="checkpoint", archive_read_off=False):
        if type(archive_read_off) is not bool:
            raise TypeError("archive_read_off must be boolean")
        if type(strict) is not bool:
            raise TypeError("strict must be boolean")
        if write_policy != "checkpoint":
            raise ValueError("V10 archive requires checkpoint APPEND writes")
        if memory_checkpoint is None:
            raise ValueError("V10 policy requires an actual archive_projector_v10 checkpoint")
        # All payload checksums, shape/dtype/finite checks and base identity are
        # validated BEFORE allocating the large base model or opening a server.
        info = checkpoint_info(base_model, memory_checkpoint, expected_stage=1)
        config = info["config"]
        if (config.get("trainer_variant") != VARIANT or type(config.get("stage")) is not int
                or config["stage"] != 1 or config.get("mode") != "archive"):
            raise ValueError("V10 inference requires actual Stage-1 archive_projector_v10 metadata")
        if type(info.get("step")) is not int or info["step"] < 0:
            raise ValueError("V10 checkpoint step must be a nonnegative integer")
        spec = config["projector"]
        if type(spec.get("enabled")) is not bool:
            raise ValueError("V10 checkpoint projector policy must be Boolean")

        # Intentionally bypass ONLY LongMemoryV7Policy.__init__, whose strict
        # V7 loader must keep rejecting this separate V10 format. No on-disk
        # metadata, old guard or rollout implementation is patched.
        LongMemoryPolicy.__init__(self, base_model, memory_checkpoint=None, device=device, strict=strict)
        self.stage, self.mode = 1, "archive"
        self.memory_off, self.write_policy_override = archive_read_off, "checkpoint"
        self.write_policy, self.cvom_threshold = "append", .05  # CVOM is unused in archive mode.
        head = self.model.action_head
        with isolated_seed(0, device):
            install_expert_lora(head, LoRAConfig(**config["expert"]), targets=config["expert_targets"])
            actual_spec = install_projector(head, enabled=spec["enabled"])
            memory_config = MemoryV7Config(**config["memory"])
            self.memory, self.cvom = RecurrentMemoryV7(memory_config), CVOMV7(memory_config)
        if actual_spec != spec:
            raise ValueError("Installed projector differs from validated checkpoint specification")
        set_trainable(head, train_projector=False, train_lora=False)
        load_checkpoint(memory_checkpoint, self.memory, head, self.cvom)
        self.memory.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        self.cvom.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        head.eval().requires_grad_(False)
        assert_expert_scope(head)
        if any(p.requires_grad for module in (head, self.memory, self.cvom) for p in module.parameters()):
            raise RuntimeError("V10 inference must freeze every loaded parameter")
        self.projector_configuration = projector_spec(head)
        self.checkpoint_step = info["step"]
        self.checkpoint_variant = VARIANT

    def _get_action(self, observation, options=None):
        # The only extension to the inherited action result is honest metadata.
        actions, info = super()._get_action(observation, options)
        return actions, {**info, "checkpoint_variant": self.checkpoint_variant,
            "checkpoint_step": self.checkpoint_step,
            "projector_enabled": bool(self.model.action_head.model.proj_out_2.enabled),
            "projector_configured_enabled": self.projector_configuration["enabled"],
            "projector_kind": self.projector_configuration["kind"]}
