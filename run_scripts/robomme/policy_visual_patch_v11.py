"""Serial V11 visual READ on the unchanged frozen V7 archive-1250 policy.

The external parent owns original preprocessing, HAMLET short cache, archive
READ/APPEND, action generation/decoding, cadence, and per-session noise. Scoped
head wrappers capture ORIGINAL post-VLLN/post-HAMLET features before archive
fusion and replace image positions only. They are restored exactly on every
exit. No target, action, or state is passed to the visual memory.

Visual APPEND follows successful parent generation AND decoding, including for
passive demo observations. Visual READ-off keeps both banks' APPEND and the
parent archive READ enabled. This wrapper is explicitly serial: concurrent or
reentrant action/reset calls are rejected, not allowed to share a head cache.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading

import torch

from gr00t.long_memory.hamlet import isolated_seed
from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
from run_scripts.robomme.checkpoint_visual_patch_v11 import VARIANT, checkpoint_info, load_checkpoint
from run_scripts.robomme.visual_patch_memory_v11 import (
    CAMERA_ORDER, PATCHES_PER_OBSERVATION, VisualPatchConfig, VisualPatchMemoryV11,
)


@contextmanager
def _scoped_methods(head, replacements):
    """Restore inherited descriptors and preexisting instance overrides exactly."""
    saved = {name: (name in head.__dict__, head.__dict__.get(name)) for name in replacements}
    installed = []
    try:
        for name, replacement in replacements.items():
            setattr(head, name, replacement)
            installed.append(name)
        yield
    finally:
        for name in reversed(installed):
            existed, value = saved[name]
            if existed:
                setattr(head, name, value)
            else:
                delattr(head, name)


class VisualPatchV11Policy(LongMemoryV7Policy):
    """Load a distinct visual-only bundle and its actual external V7 parent.

    Step zero is accepted for explicit initialization diagnostics and remains
    labeled zero. Evaluation that claims a trained V11 must separately require
    checkpoint_step > 0. Neither base nor parent files are rewritten or copied.
    """

    def __init__(self, base_model, memory_checkpoint, device="cuda:0", strict=True,
                 write_policy="checkpoint", visual_read_off=False):
        if type(visual_read_off) is not bool or type(strict) is not bool:
            raise TypeError("visual_read_off and strict must be boolean")
        if write_policy != "checkpoint":
            raise ValueError("V11 preserves checkpoint APPEND writes")
        if memory_checkpoint is None:
            raise ValueError("V11 policy requires an actual visual_patch_v11 checkpoint")
        # Validates visual payloads, exact base identity, and ALL external parent
        # files before the large model is allocated. This is never a fake V7.
        info = checkpoint_info(base_model, memory_checkpoint, expected_stage=1)
        config, metadata = info["config"], info["metadata"]
        if (config.get("trainer_variant") != VARIANT or config.get("mode") != "visual_patch"
                or type(config.get("stage")) is not int or config["stage"] != 1):
            raise ValueError("V11 requires actual Stage-1 visual_patch metadata")
        if type(info.get("step")) is not int or info["step"] < 0:
            raise ValueError("V11 checkpoint step must be a nonnegative integer")
        visual_config = VisualPatchConfig(**config["visual"])
        if config.get("camera_order") != list(CAMERA_ORDER):
            raise ValueError("V11 requires the audited camera order")
        parent = metadata["frozen_parent"]
        super().__init__(base_model, parent["path"], device=device, strict=strict,
                         write_policy="checkpoint", memory_off=False)
        if self.stage != 1 or self.mode != "archive" or self.write_policy != "append" or self.memory_off:
            raise RuntimeError("Loaded parent must retain its Stage-1 archive READ and APPEND")
        cameras = tuple(self.modality_configs["video"].modality_keys)
        if cameras != CAMERA_ORDER:
            raise ValueError("Loaded processor camera order differs from the audited V11 mapping")
        if self.n_q != visual_config.num_short_tokens or self.memory.config.feature_dim != visual_config.feature_dim:
            raise ValueError("Visual feature/short dimensions differ from the loaded parent")
        with isolated_seed(0, device):
            self.visual_memory = VisualPatchMemoryV11(visual_config)
        loaded = load_checkpoint(memory_checkpoint, self.visual_memory)
        if any(loaded[key] != info[key] for key in ("step", "config", "metadata")):
            raise ValueError("V11 checkpoint identity changed between preflight and actual loading")
        self.visual_memory.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        for module in (self.model.action_head, self.memory, self.cvom, self.visual_memory):
            if module.training or any(parameter.requires_grad for parameter in module.parameters()):
                raise RuntimeError("V11 inference requires the complete parent and visual modules frozen/eval")
        self.visual_camera_order = cameras
        self.visual_read_off = visual_read_off
        self.checkpoint_variant, self.checkpoint_step = VARIANT, info["step"]
        self.visual_weights_sha256 = metadata["payload_sha256"]["visual.safetensors"]
        self.frozen_parent_checkpoint_sha256 = parent["files_sha256"]["checkpoint.json"]
        self._visual_call_lock = threading.Lock()

    def _session(self, sid, seed, reset):
        session = super()._session(sid, seed, reset)
        if not hasattr(session, "visual_bank"):
            session.visual_bank = self.visual_memory.empty_bank(1)
            session.visual_demo_updates = 0
        return session

    def reset(self, options=None):
        if not self._visual_call_lock.acquire(blocking=False):
            raise RuntimeError("V11 policy is serial; action/reset calls cannot overlap")
        try:
            return super().reset(options)
        finally:
            self._visual_call_lock.release()

    @torch.inference_mode()
    def _get_action(self, observation, options=None):
        if not self._visual_call_lock.acquire(blocking=False):
            raise RuntimeError("V11 policy is serial; concurrent/reentrant calls are unsupported")
        try:
            return self._visual_get_action(observation, options)
        finally:
            self._visual_call_lock.release()

    def _visual_get_action(self, observation, options):
        options = options or {}
        head = self.model.action_head
        original_process, original_action = head.process_backbone_output, head.get_action_with_features
        captured = {"read_enabled": False, "image_delta_norm": 0., "image_changed_fraction": 0.}

        def process(*args, **kwargs):
            if "observation" in captured:
                raise RuntimeError("V11 expects exactly one original backbone processing call")
            backbone = original_process(*args, **kwargs)
            # The parent has validated the protocol and created this session by
            # the time it calls process_backbone_output. Never parse GT/actions.
            sid = options["session_ids"][0]
            session = self.sessions[sid]
            captured.update(sid=sid, session=session, bank=session.visual_bank)
            image = backbone["image_mask"].clone()
            attention = backbone["backbone_attention_mask"].clone()
            current = self.visual_memory.encode_observation(
                backbone["backbone_features"].clone(), image, attention,
                [int(options["frame_index"])], [bool(options.get("passive", False))],
                camera_order=self.visual_camera_order)
            captured.update(observation=current, image_mask=image, attention_mask=attention)
            return backbone

        def action(features, state_features, embodiment, backbone):
            if "observation" not in captured or captured.get("action_called", False):
                raise RuntimeError("V11 requires one capture before one Action Expert call")
            captured["action_called"] = True
            current, bank = captured["observation"], captured["bank"]
            if (features.shape != current.features.shape or features.dtype != current.features.dtype
                    or features.device != current.features.device
                    or not torch.equal(features[:, :-self.n_q], current.features[:, :-self.n_q])):
                raise ValueError("Parent conditioning changed outside the expected archive-fused short tail")
            if (not torch.equal(backbone["image_mask"], captured["image_mask"])
                    or not torch.equal(backbone["backbone_attention_mask"], captured["attention_mask"])):
                raise ValueError("Parent image/attention masks changed after original capture")
            recalled = self.visual_memory.read(current, bank, enabled=not self.visual_read_off)
            enabled = not self.visual_read_off and bool(bank.valid.any())
            captured["read_enabled"] = enabled
            if enabled:
                positions = current.image_indices
                rows = torch.arange(features.shape[0], device=features.device)[:, None]
                before, after = features[rows, positions], recalled[rows, positions]
                fused = features.clone()
                fused[rows, positions] = after
                difference = after.float() - before.float()
                captured["image_delta_norm"] = float(difference.norm())
                captured["image_changed_fraction"] = float((after != before).float().mean())
            else:
                # Preserve the original parent's exact input object for off and
                # empty-history controls; storage still occurs after success.
                fused = features
            return original_action(fused, state_features, embodiment, backbone)

        try:
            with _scoped_methods(head, {"process_backbone_output": process, "get_action_with_features": action}):
                actions, info = super()._get_action(observation, options)
            if "observation" not in captured:
                raise RuntimeError("Parent returned without capturing the original observation")
            current, session = captured["observation"], captured["session"]
            passive = bool(current.is_demo[0])
            if bool(captured.get("action_called", False)) == passive:
                raise RuntimeError("Parent passive/action cadence differs from the V11 contract")
            # Parent forward, archive WRITE, and physical action decode all
            # succeeded. Never APPEND a READ-modified image or fused short tail.
            following = self.visual_memory.append(captured["bank"], current)
            session.visual_bank = following
            session.visual_demo_updates += int(passive)
            count = following.tokens.shape[1]
            if session.observed != count:
                raise RuntimeError("Visual and parent observation counts diverged")
            visual_info = {"read_enabled": captured["read_enabled"], "observed": count,
                "append_updates": count, "demo_updates": session.visual_demo_updates,
                "bank_observations": count, "bank_tokens": count * PATCHES_PER_OBSERVATION,
                "frame_index": int(current.frames[0]), "passive": passive,
                "image_delta_norm": captured["image_delta_norm"],
                "image_changed_fraction": captured["image_changed_fraction"]}
            return actions, {**info, "checkpoint_variant": self.checkpoint_variant,
                "checkpoint_step": self.checkpoint_step, "visual_read_off": self.visual_read_off,
                "visual_weights_sha256": self.visual_weights_sha256,
                "frozen_parent_checkpoint_sha256": self.frozen_parent_checkpoint_sha256,
                "visual_memory": visual_info}
        except BaseException:
            # Parent already clears forward/decode failures. Also invalidate a
            # session if APPEND/contract checks fail or a call is interrupted.
            if "sid" in captured:
                self.sessions.pop(captured["sid"], None)
            raise
