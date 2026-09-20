"""Genuine V14 visual + existing Action Expert LoRA checkpoint inference.

The original V7 archive initializes the frozen memory and the installed LoRA
structure. The NEW V14 serializer then loads BOTH saved visual and expert
payloads. Visual READ-off retains those same adapted expert weights. Only the
reviewed serial observation, demo-tail ingest and READ/APPEND methods are reused;
no V13 checkpoint constructor, loader, or metadata coercion is involved.
"""
from __future__ import annotations

import threading

import torch

from gr00t.long_memory.expert_v4 import ExpertLoRALinear
from gr00t.long_memory.hamlet import isolated_seed
from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
from run_scripts.robomme.checkpoint_visual_expert_v14 import (
    VARIANT, bind_visual_semantics, checkpoint_info, load_checkpoint,
)
from run_scripts.robomme.policy_demo_tail_ingest_v13 import DemoTailPolicyMixinV13
from run_scripts.robomme.policy_visual_differential_v12 import VisualDifferentialV12Policy
from run_scripts.robomme.visual_demo_tail_bank_v13 import (
    CAMERA_ORDER, VisualDifferentialConfig, VisualDemoTailMemoryV13,
)


class VisualExpertV14Policy(DemoTailPolicyMixinV13, VisualDifferentialV12Policy):
    """One new joint bundle, with the original archive READ/APPEND preserved.

    Step zero is accepted and remains honestly labeled zero. Both on and off
    require the same real demo-tail RPC before the first demo execution query.
    The inference server never enables training or disables expert adapters.
    """

    def __init__(self, base_model, memory_checkpoint, device="cuda:0", strict=True,
                 write_policy="checkpoint", visual_read_off=False, expected_include_tail=None):
        if type(strict) is not bool or type(visual_read_off) is not bool:
            raise TypeError("Strict/read-off controls must be booleans")
        if expected_include_tail is not None and type(expected_include_tail) is not bool:
            raise TypeError("Expected tail arm must be a boolean or None")
        if expected_include_tail is False:
            raise ValueError("V14 requires the genuine include_tail=True joint bundle")
        if write_policy != "checkpoint" or memory_checkpoint is None:
            raise ValueError("V14 requires an actual joint checkpoint and unchanged APPEND")
        info = checkpoint_info(base_model, memory_checkpoint, expected_stage=1)
        config, metadata = info["config"], info["metadata"]
        if (config.get("trainer_variant") != VARIANT or config.get("mode") != "visual_expert"
                or config.get("architecture") != "visual_demo_tail_v13"
                or type(config.get("stage")) is not int or config["stage"] != 1
                or type(info.get("step")) is not int or info["step"] < 0
                or config.get("camera_order") != list(CAMERA_ORDER)):
            raise ValueError("Expected genuine Stage1 V14 visual/expert checkpoint identity")
        for section in (config, config.get("train", {}), metadata):
            if (section.get("include_tail") is not True or section.get("read_mode") != "differential"
                    or section.get("replay_encoding") != "framewise"):
                raise ValueError("V14 joint tail/differential/framewise semantics differ")
        if "frozen_parent" in metadata or "initial_parent" not in metadata:
            raise ValueError("V14 requires truthful initial_parent, not a frozen complete expert")
        visual_config = VisualDifferentialConfig(**config["visual"])
        parent = metadata["initial_parent"]
        # This loads the genuine OLD parent only; memory_checkpoint is never
        # passed to a legacy loader. Its installed LoRA wrappers are reused.
        LongMemoryV7Policy.__init__(self, base_model, parent["path"], device=device,
            strict=strict, write_policy="checkpoint", memory_off=False)
        if self.stage != 1 or self.mode != "archive" or self.write_policy != "append" or self.memory_off:
            raise RuntimeError("V14 requires original archive READ+APPEND to remain active")
        cameras = tuple(self.modality_configs["video"].modality_keys)
        if (cameras != CAMERA_ORDER or self.n_q != visual_config.num_short_tokens
                or self.memory.config.feature_dim != visual_config.feature_dim):
            raise ValueError("Loaded parent/processor layout differs from V14 visual checkpoint")
        head = self.model.action_head
        self._expert_adapters = tuple((name, module) for name, module in head.named_modules()
                                      if isinstance(module, ExpertLoRALinear))
        if sorted(name for name, _ in self._expert_adapters) != config["expert_targets"]:
            raise ValueError("Installed expert targets differ from the V14 checkpoint")
        self._assert_expert_enabled()
        with isolated_seed(0, device):
            self.visual_memory = VisualDemoTailMemoryV13(visual_config, read_mode="differential")
            bind_visual_semantics(self.visual_memory, include_tail=True)
        loaded = load_checkpoint(memory_checkpoint, self.visual_memory, head)
        if any(loaded[key] != info[key] for key in ("step", "config", "metadata")):
            raise ValueError("V14 checkpoint identity changed after preflight")
        self.visual_memory.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        self.model.eval().requires_grad_(False)
        for module in (self.model, self.memory, self.cvom, self.visual_memory):
            if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise RuntimeError("Every V14 inference module must remain frozen/eval without gradients")
        self._assert_expert_enabled()
        self.visual_camera_order = cameras
        self.visual_read_off, self.visual_read_mode = visual_read_off, "differential"
        self.checkpoint_variant, self.checkpoint_step = VARIANT, info["step"]
        self.visual_weights_sha256 = metadata["payload_sha256"]["visual.safetensors"]
        self.expert_weights_sha256 = metadata["payload_sha256"]["expert.safetensors"]
        self.initial_parent_checkpoint_sha256 = parent["files_sha256"]["checkpoint.json"]
        # The unchanged inherited method needs this attribute internally. Its
        # returned legacy key is removed below: the complete parent AE is NOT
        # frozen at its initial weights in this new experiment.
        self.frozen_parent_checkpoint_sha256 = self.initial_parent_checkpoint_sha256
        self._visual_call_lock = threading.Lock()
        self.configure_demo_tail_ingest(enabled=True)

    def _assert_expert_enabled(self):
        if not self._expert_adapters or any(module.enabled is not True for _, module in self._expert_adapters):
            raise RuntimeError("V14 visual-on/off must retain every saved adapted expert projection")

    def _visual_get_action(self, observation, options):
        try:
            self._assert_expert_enabled()
            actions, info = super()._visual_get_action(observation, options)
            self._assert_expert_enabled()
            info.pop("frozen_parent_checkpoint_sha256", None)
            info.update(initial_parent_checkpoint_sha256=self.initial_parent_checkpoint_sha256,
                        expert_weights_sha256=self.expert_weights_sha256,
                        expert_adapters_enabled=True, expert_adapter_count=len(self._expert_adapters))
            return actions, info
        except BaseException:
            for sid in (options or {}).get("session_ids", []):
                self.sessions.pop(sid, None)
            raise
