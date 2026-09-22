"""ECHO online adapter: observed endpoints and completed physical controls only.

The action sequence received at endpoint t belongs to t-1 -> t. It is
normalized with the original processor and the previous raw state. Demonstration
and first endpoints contain no invented action. READ precedes endpoint WRITE,
matching the offline ECHO replay contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

from gr00t.data.types import MessageType
from gr00t.long_memory.online_policy import normalize_executed_actions
from gr00t.long_memory.online_policy_v7 import _scalar_metrics
from run_scripts.robomme.feature_precision_v19 import extract_hamlet_features
from run_scripts.robomme.policy_representation_v18 import RepresentationPolicyV18, RepresentationSession
from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18


VARIANT = "echo_cvom_v1"
RUNTIME_SOURCES = (
    "policy_echo_cvom.py", "serve_echo_cvom.py", "echo_cvom_core.py",
    "echo_cvom_checkpoint.py", "policy_representation_v18.py",
    "representation_core_v18.py", "checkpoint_representation_v18.py",
    "feature_precision_v19.py", "cvom_admission_checkpoint.py", "train_cvom_admission.py",
)


def runtime_source_identity():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in RUNTIME_SOURCES}


@dataclass
class EchoSession(RepresentationSession):
    echo_bank: object | None = None
    previous_moment: torch.Tensor | None = None
    previous_state: torch.Tensor | None = None
    previous_is_demo: torch.Tensor | None = None


def completed_controls(processor, embodiment, raw, previous_states, stride, *, device):
    """Return one padded, original-processor-normalized executed prefix."""
    controls = normalize_executed_actions(processor, embodiment, raw,
                                          previous_states, processor.max_action_dim)
    if len(controls) > stride:
        raise ValueError("Executed prefix exceeds the trained observation interval")
    padded = torch.zeros(1, stride, processor.max_action_dim, dtype=torch.float32, device=device)
    mask = torch.zeros(1, stride, dtype=torch.bool, device=device)
    padded[0, :len(controls)] = controls.to(device)
    mask[0, :len(controls)] = True
    return padded, mask


class EchoPolicyV1(RepresentationPolicyV18):
    def __init__(self, base_model, checkpoint, device="cuda:0", memory_off=False,
                 fifo=False, strict=True):
        if type(memory_off) is not bool or type(fifo) is not bool:
            raise TypeError("ECHO READ-off and FIFO flags must be Boolean")
        from run_scripts.robomme.echo_cvom_checkpoint import inspect_checkpoint, load_checkpoint
        from run_scripts.robomme.echo_cvom_core import EchoConfig, EchoMemoryV1
        from run_scripts.robomme.checkpoint_representation_v18 import checkpoint_info_v18

        before = runtime_source_identity()
        checkpoint = Path(checkpoint).resolve()
        info = inspect_checkpoint(base_model, checkpoint)
        parent_path = info["metadata"]["parent_identity"]["path"]
        parent_info = checkpoint_info_v18(base_model, parent_path)
        for key in ("representation", "expert", "expert_targets"):
            if info["config"][key] != parent_info["config"][key]:
                raise ValueError(f"ECHO and parent initialization configurations differ: {key}")
        super().__init__(base_model, parent_path, device=device, memory_off=memory_off,
                         strict=strict, feature_precision="native")
        with torch.random.fork_rng(devices=[]):
            echo = EchoMemoryV1(RepresentationConfigV18(**info["config"]["representation"]),
                                EchoConfig(**info["config"]["echo"]))
        echo.initialize_from_parent_delta(self.representation.delta_state_dict())
        loaded = load_checkpoint(checkpoint, echo, self.model.action_head)
        self.representation = echo.to(device=device).eval().requires_grad_(False)
        if echo.echo_config.action_dim != self.processor.max_action_dim:
            raise ValueError("ECHO action width differs from the original processor")
        self.model.action_head.eval().requires_grad_(False)
        self.stage = info["stage"]
        self.mode = VARIANT
        self.write_mode = "fifo" if fifo or self.stage == 1 else "learned"
        self.write_policy = "fifo" if self.write_mode == "fifo" else "echo-cvom"
        self.checkpoint_step = info["step"]
        self.payload_sha256 = info["metadata"]["payload_sha256"]
        self.echo_checkpoint_sha256 = info["files_sha256"]["checkpoint.json"]
        self.echo_parent_identity = info["metadata"]["parent_identity"]
        self.echo_source_sha256 = runtime_source_identity()
        if before != self.echo_source_sha256 or loaded["files_sha256"] != info["files_sha256"]:
            raise ValueError("ECHO runtime source or checkpoint changed while loading")

    def _session(self, sid, seed, reset):
        if reset:
            self.sessions.pop(sid, None)
        if sid not in self.sessions:
            generator = torch.Generator(device=self.model.device).manual_seed(seed)
            self.sessions[sid] = EchoSession(generator, seed)
        session = self.sessions[sid]
        if session.episode_seed != seed:
            raise ValueError("Episode seed changed without a session reset")
        self.sessions.move_to_end(sid)
        while len(self.sessions) > self.session_cap:
            self.sessions.popitem(last=False)
        return session

    @torch.inference_mode()
    def _get_action(self, observation, options=None):
        options = options or {}
        observations = self._unbatch_observation(observation)
        ids, resets = options.get("session_ids"), options.get("reset_memory", [False])
        if (len(observations) != 1 or not isinstance(ids, list) or len(ids) != 1
                or not isinstance(ids[0], str) or not ids[0]):
            raise ValueError("ECHO requires one observation and one nonempty session_id")
        if not isinstance(resets, list) or len(resets) != 1 or type(resets[0]) is not bool:
            raise ValueError("reset_memory must be a one-element Boolean list")
        seed, frame = options.get("episode_seed"), options.get("frame_index")
        if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0 for x in (seed, frame)):
            raise ValueError("Nonnegative integer episode seed/frame required")
        seed, frame = int(seed), int(frame)
        passive, prime = options.get("passive", False), options.get("prime_only", False)
        if type(passive) is not bool or type(prime) is not bool or prime != passive:
            raise ValueError("Passive demo calls must be prime_only")
        session = self._session(ids[0], seed, resets[0])
        raw = options.get("executed_actions")
        raw = np.empty((0, 8), np.float32) if raw is None else np.asarray(raw, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 8 or not np.isfinite(raw).all():
            raise ValueError("executed_actions must be finite [C,8]")
        if session.frame is None:
            if len(raw):
                raise ValueError("First endpoint cannot include previous controls")
        else:
            gap = frame - session.frame
            if (not 0 < gap <= self.stride or (session.passive and len(raw))
                    or (not session.passive and (len(raw) != gap or passive))):
                raise ValueError("Canonical observation/action cadence mismatch")

        head = self.model.action_head
        try:
            padded, action_mask = completed_controls(self.processor, self.embodiment_tag.value,
                raw, session.raw_states, self.stride, device=self.model.device)
            step = self._to_vla_step_data(observations[0])
            step.is_demonstration = passive
            processed = self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            state = torch.as_tensor(processed["state"]).reshape(1, -1).float().to(self.model.device)
            batch = self.collate_fn([processed])["inputs"]
            head._memory_cache, head._vision_cache = session.short_cache, None
            head._inference_gen = session.generator
            backbone_inputs, action_inputs = self.model.prepare_input(batch)
            backbone, moment = extract_hamlet_features(self.model, head, backbone_inputs, self.n_q, "native")
            features = backbone["backbone_features"]
            frozen_short = features[:, -self.n_q:]
            enabled = not self.memory_off and not prime and session.observations > 0
            demo = torch.tensor([passive], device=self.model.device, dtype=torch.bool)
            completed = {} if session.frame is None else {
                "previous_moment": session.previous_moment, "previous_state": session.previous_state,
                "previous_is_demo": session.previous_is_demo, "completed_actions": padded,
                "completed_action_mask": action_mask,
                "transition_valid": torch.ones(1, dtype=torch.bool, device=self.model.device)}
            with torch.autocast(device_type=self.model.device.type, enabled=False):
                result = self.representation.step(frozen_short.float(), moment, state,
                    torch.tensor([frame], device=self.model.device), demo,
                    bank_state=session.echo_bank, **completed,
                    event_index=session.observations, write_mode=self.write_mode,
                    read_enabled=enabled, write_enabled=True)
            cast = result["fused"].to(frozen_short.dtype)
            adapted = result["short"].to(frozen_short.dtype)
            features = torch.cat((features[:, :-self.n_q], cast), dim=1)
            metrics = _scalar_metrics(result["metrics"])
            metrics.update(ae_conditioning_delta_norm=float((cast.float() - adapted.float()).norm()),
                ae_conditioning_changed_fraction=float((cast != adapted).float().mean()),
                short_adaptation_delta_norm=float((adapted.float() - frozen_short.float()).norm()))
            if prime:
                prediction = features.new_zeros(1, head.action_horizon, head.action_dim)
            else:
                state_features = head.state_encoder(action_inputs.state, action_inputs.embodiment_id)
                prediction = head.get_action_with_features(features, state_features,
                    action_inputs.embodiment_id, backbone)["action_pred"]
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite ECHO action")
            session.echo_bank = result["bank_state"].detach()
            session.long_bank = result["bank"].detach()
            session.previous_moment = moment.detach().clone()
            session.previous_state = state.detach().clone()
            session.previous_is_demo = demo.detach().clone()
            session.observations += 1
            session.short_cache = head._memory_cache.detach().clone()
            session.raw_states = {key: np.array(value, copy=True) for key, value in step.states.items()}
            session.frame, session.passive = frame, passive
            states = {key: np.stack([step.states[key]]) for key in self.modality_configs["state"].modality_keys}
            decoded = self.processor.decode_action(prediction.float().cpu().numpy(), self.embodiment_tag, states)
            actions = {key: np.asarray(value, dtype=np.float32) for key, value in decoded.items()}
            if any(not np.isfinite(value).all() for value in actions.values()):
                raise FloatingPointError("Nonfinite decoded ECHO control")
        except Exception:
            self.sessions.pop(ids[0], None)
            raise
        finally:
            head.reset_memory()
            head._inference_gen = None
        return actions, {
            "long_memory": {"policy": self.write_policy, "mode": VARIANT,
                "memory_read_enabled": enabled, "frame_index": frame, "passive": passive,
                "observations_seen": session.observations, "memory_tokens": int(session.long_bank.shape[1]),
                "completed_action_count": len(raw), "read": metrics},
            "stage": self.stage, "episode_seed": seed, "expert_adapted": True,
            "checkpoint_variant": VARIANT, "checkpoint_step": self.checkpoint_step,
            "memory_off": self.memory_off, "representation": self.representation.config.representation,
            "feature_precision": "native", "feature_precision_rules": self.feature_precision_rules,
            "payload_sha256": self.payload_sha256, "echo_checkpoint_sha256": self.echo_checkpoint_sha256,
            "echo_parent_identity": self.echo_parent_identity, "echo_source_sha256": self.echo_source_sha256,
        }
