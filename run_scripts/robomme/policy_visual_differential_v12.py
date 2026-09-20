"""Strict V12 loading with the unchanged V11 serial observation/APPEND protocol.

Only model construction and truthful V12 diagnostics are new. The inherited
rollout captures original post-HAMLET features, preserves the frozen parent's
short READ, changes current image positions only, and appends after successful
action generation AND decoding. No legacy checkpoint metadata is rewritten.
"""
from __future__ import annotations

import threading

import torch

from gr00t.long_memory.hamlet import isolated_seed
from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
from run_scripts.robomme.policy_visual_patch_v11 import VisualPatchV11Policy
from run_scripts.robomme.checkpoint_visual_differential_v12 import VARIANT, checkpoint_info, load_checkpoint
from run_scripts.robomme.visual_differential_memory_v12 import (
    CAMERA_ORDER, VisualDifferentialConfig, VisualDifferentialMemoryV12,
)

READ_MODES = ("differential", "current_only")


def checkpoint_read_mode(info, expected=None):
    config, metadata = info["config"], info["metadata"]
    mode = config.get("read_mode")
    if (mode not in READ_MODES or config.get("train", {}).get("read_mode") != mode
            or metadata.get("read_mode") != mode or (expected is not None and mode != expected)):
        raise ValueError("V12 checkpoint read_mode differs from declared configuration/role")
    return mode


class VisualDifferentialV12Policy(VisualPatchV11Policy):
    """Both trained arms preserve the identical original parent and cadence.

    Initialization checkpoints remain honestly labeled step0 and are accepted
    by the policy; the paired evaluator needs an explicit opt-in for step0.
    visual_read_off is supported only for the differential bundle.
    """
    def __init__(self, base_model, memory_checkpoint, device="cuda:0", strict=True,
                 write_policy="checkpoint", visual_read_off=False, expected_read_mode=None):
        if type(strict) is not bool or type(visual_read_off) is not bool:
            raise TypeError("Strict/read-off flags must be boolean")
        if write_policy != "checkpoint" or memory_checkpoint is None:
            raise ValueError("V12 requires a real visual checkpoint and unchanged checkpoint APPEND")
        if expected_read_mode is not None and expected_read_mode not in READ_MODES:
            raise ValueError("Unknown expected V12 read mode")
        info = checkpoint_info(base_model, memory_checkpoint, expected_stage=1)
        config, metadata = info["config"], info["metadata"]
        mode = checkpoint_read_mode(info, expected_read_mode)
        if (config.get("trainer_variant") != VARIANT or config.get("mode") != "visual_differential"
                or type(config.get("stage")) is not int or config["stage"] != 1
                or type(info.get("step")) is not int or info["step"] < 0):
            raise ValueError("Requires an actual Stage1 V12 checkpoint identity")
        if visual_read_off and mode != "differential":
            raise ValueError("visual-off must use the differential checkpoint")
        visual_config = VisualDifferentialConfig(**config["visual"])
        if config.get("camera_order") != list(CAMERA_ORDER):
            raise ValueError("V12 requires the audited camera order")
        # Deliberately skip the V11 constructor/serializer, not its pure serial
        # rollout. Load the actual original V7 parent without relabeling files.
        LongMemoryV7Policy.__init__(self, base_model, metadata["frozen_parent"]["path"], device=device,
                                   strict=strict, write_policy="checkpoint", memory_off=False)
        if self.stage != 1 or self.mode != "archive" or self.write_policy != "append" or self.memory_off:
            raise RuntimeError("V12 parent archive READ and APPEND must remain enabled")
        cameras = tuple(self.modality_configs["video"].modality_keys)
        if cameras != CAMERA_ORDER or self.n_q != visual_config.num_short_tokens:
            raise ValueError("Loaded processor camera order/short layout differs")
        if self.memory.config.feature_dim != visual_config.feature_dim:
            raise ValueError("Parent/visual feature dimensions differ")
        with isolated_seed(0, device):
            self.visual_memory = VisualDifferentialMemoryV12(visual_config, read_mode=mode)
        loaded = load_checkpoint(memory_checkpoint, self.visual_memory)
        if any(loaded[key] != info[key] for key in ("step", "config", "metadata")):
            raise ValueError("V12 checkpoint identity changed between preflight and loading")
        if self.visual_memory.read_mode != mode:
            raise ValueError("Installed visual read mode differs from checkpoint")
        self.visual_memory.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        for module in (self.model.action_head, self.memory, self.cvom, self.visual_memory):
            if module.training or any(p.requires_grad for p in module.parameters()):
                raise RuntimeError("All V12 inference modules must remain frozen/eval")
        self.visual_camera_order = cameras
        self.visual_read_off, self.visual_read_mode = visual_read_off, mode
        self.checkpoint_variant, self.checkpoint_step = VARIANT, info["step"]
        self.visual_weights_sha256 = metadata["payload_sha256"]["visual.safetensors"]
        self.frozen_parent_checkpoint_sha256 = metadata["frozen_parent"]["files_sha256"]["checkpoint.json"]
        self._visual_call_lock = threading.Lock()

    def _visual_get_action(self, observation, options):
        # The inherited _get_action has already acquired the serial lock.
        if self.visual_memory.read_mode != self.visual_read_mode:
            raise RuntimeError("Runtime V12 read mode changed after checkpoint loading")
        captured = []
        def readout(module, args, output):
            captured.append((float(args[0].float().norm()), float(output.float().norm())))
        handle = self.visual_memory.output_projection.register_forward_hook(readout)
        completed = False
        try:
            actions, info = super()._visual_get_action(observation, options)
            completed = True
            visual = info["visual_memory"]
            enabled = visual["read_enabled"]
            if len(captured) != int(enabled):
                raise RuntimeError("V12 readout call count differs from active visual branch")
            visual.update(read_mode=self.visual_read_mode,
                past_read_enabled=enabled and self.visual_read_mode == "differential",
                current_reference_enabled=enabled,
                readout_norm=captured[0][0] if captured else 0.,
                projected_residual_norm=captured[0][1] if captured else 0.,
                readout_norm_semantics="H_past_minus_H_current" if self.visual_read_mode == "differential" else "H_current")
            return actions, {**info, "visual_read_mode": self.visual_read_mode}
        except BaseException:
            # Inherited logic handles parent/decode/APPEND failures; also clear
            # a completed parent session if these new metric checks fail.
            if completed:
                for sid in (options or {}).get("session_ids", []):
                    self.sessions.pop(sid, None)
            raise
        finally:
            handle.remove()
