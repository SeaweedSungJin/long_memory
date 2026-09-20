"""Opt-in real HAMLET inference adapter; original model/policy files stay intact.

The baseline uses the same strictly loaded frozen checkpoint and its unmodified
flow denoiser. The memory variant changes ONLY the post-short-memory token tail,
at exactly the feature location used by cache.py / episode_flow_loss().
"""
from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.policy import BasePolicy

from .core import EpisodicMemory, MemoryConfig
from .hamlet import checkpoint_identity, load_frozen_hamlet
from .monitoring import load_checkpoint
from .online import OnlineEpisodicBank


def memory_checkpoint_info(base_model, memory_checkpoint):
    """Preflight without constructing the multi-billion-parameter base model."""
    info = json.loads((Path(memory_checkpoint) / "checkpoint.json").read_text())
    if info.get("format_version") != 1 or info["config"]["stage"] not in (1, 2):
        raise ValueError("Expected a two-stage episodic-memory checkpoint, not a full HAMLET model")
    if info["metadata"]["base_model"] != checkpoint_identity(base_model):
        raise ValueError("Memory was trained on a different/changed base checkpoint; use its recorded base")
    MemoryConfig(**info["config"]["memory"])
    if not (Path(memory_checkpoint) / "model.safetensors").is_file():
        raise FileNotFoundError("Memory checkpoint lacks model.safetensors")
    return info


def normalize_executed_actions(processor, embodiment, actions, previous_states, padded_dim):
    """Use the ORIGINAL action normalizer on actual physical controls, not predictions.

    RoboMME transport fixes physical ordering to seven arm joints then gripper.
    The processor decides normalization/relative conversion and model key order.
    """
    raw = np.asarray(actions, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 8 or not np.isfinite(raw).all():
        raise ValueError("RoboMME executed_actions must be finite [C,8] joint+gripper controls")
    if len(raw) == 0:
        return torch.zeros((0, padded_dim), dtype=torch.float32)
    if previous_states is None:
        raise ValueError("Cannot normalize a previous action without its pre-event state")
    keys = processor.modality_configs[embodiment]["action"].modality_keys
    gripper_keys = [key for key in keys if key in ("gripper_close", "gripper_position")]
    if len(keys) != 2 or "joint_position" not in keys or len(gripper_keys) != 1:
        raise ValueError("This RoboMME adapter requires joint_position and one gripper action")
    action = {"joint_position": raw[:, :7], gripper_keys[0]: raw[:, 7:8]}
    normalized = processor.state_action_processor.apply_action(action, embodiment, state=previous_states)
    merged = np.concatenate([normalized[k] for k in keys], axis=-1)
    if merged.shape[1] > padded_dim or not np.isfinite(merged).all():
        raise ValueError("Invalid normalized executed controls")
    result = torch.zeros((len(raw), padded_dim), dtype=torch.float32)
    result[:, :merged.shape[1]] = torch.from_numpy(merged.copy())
    return result


@dataclass
class _Session:
    generator: torch.Generator
    episode_seed: int
    bank: OnlineEpisodicBank | None = None
    short_cache: torch.Tensor | None = None
    raw_states: dict | None = None
    frame: int | None = None
    passive: bool = True


class LongMemoryPolicy(Gr00tPolicy):
    """Single-environment calls; independent bounded state for multiple sessions.

    Batch size one is intentional for sequential RoboMME evaluation. Unsupported
    batching is rejected, not silently mixed across episodes.
    """

    def __init__(self, base_model, memory_checkpoint=None, device="cuda:0", strict=True):
        BasePolicy.__init__(self, strict=strict)
        info = memory_checkpoint_info(base_model, memory_checkpoint) if memory_checkpoint else None
        self.model, self.processor = load_frozen_hamlet(base_model, device)
        self.embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
        self.modality_configs = self.processor.get_modality_configs()[self.embodiment_tag.value]
        self.modality_configs["video"].delta_indices = [0]
        if self.modality_configs["state"].delta_indices != [0]:
            raise ValueError("Online memory requires a single current state")
        language = self.modality_configs["language"]
        if len(language.modality_keys) != 1 or language.delta_indices != [0]:
            raise ValueError("Online memory requires one current language instruction")
        self.language_key = language.modality_keys[0]
        self.collate_fn = self.processor.collator
        self.stride = int(self.model.config.memory_stride)
        self.n_q = int(self.model.config.n_moment_tokens)
        self.memory, self.write_policy, self.stage = None, "baseline", 0
        if info:
            self.stage = int(info["config"]["stage"])
            self.memory = EpisodicMemory(MemoryConfig(**info["config"]["memory"]))
            load_checkpoint(memory_checkpoint, self.memory)
            self.memory.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
            cfg = self.memory.config
            if (cfg.feature_dim != self.model.config.backbone_embedding_dim
                    or cfg.state_dim != self.processor.max_state_dim
                    or cfg.action_dim != self.processor.max_action_dim):
                raise ValueError("Memory feature/state/action dimensions differ from the base processor")
            self.write_policy = "all" if self.stage == 1 else "hard"
        self.sessions = OrderedDict()
        self.session_cap = 64
        # The original denoiser consults this switch, but takes the per-session
        # generator assigned below. No priming call advances that generator.
        os.environ.setdefault("GR00T_INFERENCE_SEED", "6")

    def reset(self, options=None):
        options = options or {}
        ids = options.get("session_ids")
        if ids is None:
            count = len(self.sessions)
            self.sessions.clear()
        else:
            if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
                raise ValueError("reset session_ids must be a list of strings")
            count = sum(self.sessions.pop(sid, None) is not None for sid in ids)
        self.model.action_head.reset_memory()
        self.model.action_head._inference_gen = None
        return {"cleared_sessions": count}

    def _session(self, sid, seed, reset):
        if reset:
            self.sessions.pop(sid, None)
        if sid not in self.sessions:
            generator = torch.Generator(device=self.model.device).manual_seed(seed)
            bank = OnlineEpisodicBank(self.memory, self.stride, self.write_policy) if self.memory else None
            self.sessions[sid] = _Session(generator, seed, bank)
        session = self.sessions[sid]
        if session.episode_seed != seed:
            raise ValueError("Episode seed changed without resetting its session")
        self.sessions.move_to_end(sid)
        while len(self.sessions) > self.session_cap:
            self.sessions.popitem(last=False)
        return session

    @torch.inference_mode()
    def _get_action(self, observation, options=None):
        options = options or {}
        observations = self._unbatch_observation(observation)
        if len(observations) != 1:
            raise ValueError("Long-memory RoboMME evaluation requires batch size 1")
        ids = options.get("session_ids")
        flags = options.get("reset_memory", [False])
        if not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], str) or not ids[0]:
            raise ValueError("Provide exactly one nonempty session_id for causal memory")
        if not isinstance(flags, list) or len(flags) != 1:
            raise ValueError("reset_memory must be a one-element list")
        if "frame_index" not in options or "episode_seed" not in options:
            raise ValueError("This adapter needs frame_index and episode_seed; use run_long_memory_rollout.py")
        seed, frame = int(options["episode_seed"]), int(options["frame_index"])
        if seed < 0 or frame < 0:
            raise ValueError("Negative episode seed/frame")
        passive = bool(options.get("passive", False))
        prime = bool(options.get("prime_only", False))
        if prime != passive:
            raise ValueError("Passive demo calls must be prime_only; execution calls must generate actions")
        session = self._session(ids[0], seed, bool(flags[0]))
        raw = options.get("executed_actions")
        raw = np.empty((0, 8), np.float32) if raw is None else np.asarray(raw, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 8 or not np.isfinite(raw).all():
            raise ValueError("executed_actions must be finite [C,8]")
        if session.frame is None:
            if len(raw):
                raise ValueError("First call cannot contain previous controls")
        else:
            gap = frame - session.frame
            if not 0 < gap <= self.stride:
                raise ValueError("Observation cadence differs from the trained memory stride")
            if session.passive and len(raw):
                raise ValueError("Demo transitions have no executed robot actions")
            if not session.passive and (len(raw) != gap or passive):
                raise ValueError("Missing/misaligned actually executed action prefix")

        obs = observations[0]
        step = self._to_vla_step_data(obs)
        step.is_demonstration = passive
        processed = self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
        state = torch.as_tensor(processed["state"]).reshape(-1).float()
        controls = normalize_executed_actions(self.processor, self.embodiment_tag.value,
                    raw, session.raw_states, self.processor.max_action_dim)
        batch = self.collate_fn([processed])["inputs"]
        head = self.model.action_head
        head._memory_cache = session.short_cache
        head._vision_cache = None
        head._inference_gen = session.generator
        try:
            backbone_inputs, action_inputs = self.model.prepare_input(batch)
            backbone = self.model.backbone(backbone_inputs)
            # Preserve normalized raw moment BEFORE short-memory replaces it.
            moment = head.vlln(backbone["backbone_features"])[:, -self.n_q:].clone()
            backbone = head.process_backbone_output(backbone, action_inputs_B=1)
            features = backbone["backbone_features"]
            if self.memory is not None:
                # Training uses FP32 new modules on BF16-cached frozen features.
                with torch.autocast(device_type=self.model.device.type, enabled=False):
                    fused, diagnostics = session.bank.advance(features[0, -self.n_q:], moment[0], state,
                            frame=frame, passive=passive, actions=controls)
                features = torch.cat((features[:, :-self.n_q], fused.to(features.dtype)), dim=1)
            else:
                diagnostics = {"policy": "baseline", "bank_fill": 0, "frame_index": frame, "passive": passive}
            if prime:
                prediction = features.new_zeros(1, head.action_horizon, head.action_dim)
            else:
                state_features = head.state_encoder(action_inputs.state, action_inputs.embodiment_id)
                prediction = head.get_action_with_features(features, state_features,
                            action_inputs.embodiment_id, backbone)["action_pred"]
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite model action; refusing to send controls")
            session.short_cache = head._memory_cache.detach().clone()
            session.raw_states = {k: np.array(v, copy=True) for k, v in step.states.items()}
            session.frame, session.passive = frame, passive
        except Exception:
            # A partially advanced session must never be retried as if unchanged.
            self.sessions.pop(ids[0], None)
            raise
        finally:
            head.reset_memory()
            head._inference_gen = None
        batched_states = {k: np.stack([step.states[k]]) for k in self.modality_configs["state"].modality_keys}
        decoded = self.processor.decode_action(prediction.float().cpu().numpy(), self.embodiment_tag, batched_states)
        actions = {k: np.asarray(v, dtype=np.float32) for k, v in decoded.items()}
        if any(not np.isfinite(v).all() for v in actions.values()):
            raise FloatingPointError("Nonfinite decoded physical action")
        return actions, {"long_memory": diagnostics, "stage": self.stage, "episode_seed": seed}
