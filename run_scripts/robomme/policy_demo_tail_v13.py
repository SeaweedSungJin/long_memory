"""V13 checkpoint construction + image-only demo-ingest + original serial V12.

No old checkpoint is relabeled. Canonical short/archive/AE behavior stays in
the frozen inherited policy; the mixin adds a separate passive visual bank.
The training sidecar is not an inference dependency: reset supplies live RGB.
"""
from __future__ import annotations

import threading

import torch

from gr00t.long_memory.hamlet import isolated_seed
from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
from run_scripts.robomme.checkpoint_demo_tail_v13 import (
    VARIANT, checkpoint_info, load_checkpoint, bind_visual_semantics,
)
from run_scripts.robomme.policy_demo_tail_ingest_v13 import DemoTailPolicyMixinV13
from run_scripts.robomme.policy_visual_differential_v12 import VisualDifferentialV12Policy
from run_scripts.robomme.visual_demo_tail_bank_v13 import (
    CAMERA_ORDER, VisualDifferentialConfig, VisualDemoTailMemoryV13,
)


class DemoTailV13Policy(DemoTailPolicyMixinV13, VisualDifferentialV12Policy):
    """Real V13 constructor; inherits only reviewed serial rollout methods.

    An initialization bundle remains honestly step zero. The evaluator must
    explicitly opt in before using it. READ-off still requires/ingests tails
    for demo episodes, allowing a real ingestion-transparency control.
    """

    def __init__(self, base_model, memory_checkpoint, device="cuda:0", strict=True,
                 write_policy="checkpoint", visual_read_off=False, expected_include_tail=None):
        if type(strict) is not bool or type(visual_read_off) is not bool:
            raise TypeError("Strict/read-off controls must be booleans")
        if expected_include_tail is not None and type(expected_include_tail) is not bool:
            raise TypeError("Expected tail arm must be a boolean or None")
        if write_policy != "checkpoint" or memory_checkpoint is None:
            raise ValueError("V13 requires an actual checkpoint and unchanged rule APPEND")
        info = checkpoint_info(base_model, memory_checkpoint, expected_stage=1)
        config, metadata = info["config"], info["metadata"]
        include_tail = config["include_tail"]
        if (config["trainer_variant"] != VARIANT or config["mode"] != "visual_demo_tail"
                or config["replay_encoding"] != "framewise" or config["read_mode"] != "differential"
                or type(include_tail) is not bool
                or expected_include_tail is not None and include_tail != expected_include_tail):
            raise ValueError("V13 checkpoint semantic/arm mismatch")
        if visual_read_off and not include_tail:
            raise ValueError("The V13 tail-off control requires the actual tail-on checkpoint")
        visual_config = VisualDifferentialConfig(**config["visual"])
        if config["camera_order"] != list(CAMERA_ORDER):
            raise ValueError("V13 requires original front/wrist camera order")
        # Skip V11/V12 constructors and serializers, not their rollout methods.
        LongMemoryV7Policy.__init__(self, base_model, metadata["frozen_parent"]["path"],
            device=device, strict=strict, write_policy="checkpoint", memory_off=False)
        if self.stage != 1 or self.mode != "archive" or self.write_policy != "append" or self.memory_off:
            raise RuntimeError("Original archive READ+APPEND must remain active")
        cameras = tuple(self.modality_configs["video"].modality_keys)
        if (cameras != CAMERA_ORDER or self.n_q != visual_config.num_short_tokens
                or self.memory.config.feature_dim != visual_config.feature_dim):
            raise ValueError("Loaded parent/processor layout differs from V13 visual checkpoint")
        with isolated_seed(0, device):
            self.visual_memory = VisualDemoTailMemoryV13(visual_config, read_mode="differential")
            bind_visual_semantics(self.visual_memory, include_tail=include_tail)
        loaded = load_checkpoint(memory_checkpoint, self.visual_memory)
        if any(loaded[key] != info[key] for key in ("step", "config", "metadata")):
            raise ValueError("V13 bundle changed after preflight")
        self.visual_memory.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        self.model.eval().requires_grad_(False)
        for module in (self.model, self.memory, self.cvom, self.visual_memory):
            if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise RuntimeError("Every V13 inference module must remain frozen/eval")
        self.visual_camera_order = cameras
        self.visual_read_off, self.visual_read_mode = visual_read_off, "differential"
        self.checkpoint_variant, self.checkpoint_step = VARIANT, info["step"]
        self.visual_weights_sha256 = metadata["payload_sha256"]["visual.safetensors"]
        self.frozen_parent_checkpoint_sha256 = metadata["frozen_parent"]["files_sha256"]["checkpoint.json"]
        self._visual_call_lock = threading.Lock()
        self.configure_demo_tail_ingest(enabled=include_tail)
