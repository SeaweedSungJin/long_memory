"""Online V18: canonical HAMLET observations, strictly-past READ, then WRITE.

The original backbone, VLLN and short-memory cache are never rewritten. B's
separate short adapter sees the SAME normalized moments as cached training.
READ-off retains that adapted short representation and the same adapted AE.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from gr00t.data.types import MessageType
from gr00t.long_memory.online_policy import LongMemoryPolicy, _Session
from gr00t.long_memory.online_policy_v7 import _scalar_metrics
from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora, set_expert_trainable
from run_scripts.robomme.checkpoint_representation_v18 import checkpoint_info_v18, load_checkpoint_v18
from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18, RepresentationMemoryV18


@dataclass
class RepresentationSession(_Session):
    long_bank: torch.Tensor | None = None
    moment_history: torch.Tensor | None = None
    observations: int = 0
    demo_updates: int = 0


class RepresentationPolicyV18(LongMemoryPolicy):
    def __init__(self, base_model, checkpoint, device="cuda:0", memory_off=False, strict=True):
        if type(memory_off) is not bool:
            raise TypeError("memory_off must be boolean")
        info = checkpoint_info_v18(base_model, checkpoint)
        super().__init__(base_model, memory_checkpoint=None, device=device, strict=strict)
        cfg = info["config"]
        head = self.model.action_head
        install_expert_lora(head, LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
        self.representation = RepresentationMemoryV18(
            RepresentationConfigV18(**cfg["representation"]),
            base_memory_transformer=head.memory_transformer,
        )
        load_checkpoint_v18(checkpoint, self.representation, head)
        set_expert_trainable(head, False)
        head.eval().requires_grad_(False)
        self.representation.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        self.memory_off = memory_off
        self.stage, self.mode, self.write_policy = 1, "representation_v18", "fifo"
        self.checkpoint_step = info["step"]
        self.payload_sha256 = info["metadata"]["payload_sha256"]
        rcfg = self.representation.config
        if (rcfg.num_short_tokens != self.n_q or rcfg.short_window != self.model.config.memory_window
                or rcfg.state_dim != self.processor.max_state_dim):
            raise ValueError("V18 dimensions/window differ from the original HAMLET processor")

    def _session(self, sid, seed, reset):
        if reset:
            self.sessions.pop(sid, None)
        if sid not in self.sessions:
            generator = torch.Generator(device=self.model.device).manual_seed(seed)
            self.sessions[sid] = RepresentationSession(generator, seed)
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
        ids, resets = options.get("session_ids"), options.get("reset_memory", [False])
        if len(observations) != 1 or not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], str) or not ids[0]:
            raise ValueError("V18 requires one observation and one nonempty session_id")
        if not isinstance(resets, list) or len(resets) != 1 or type(resets[0]) is not bool:
            raise ValueError("reset_memory must be a one-element Boolean list")
        seed, frame = options.get("episode_seed"), options.get("frame_index")
        if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0 for x in (seed, frame)):
            raise ValueError("Nonnegative integer episode_seed/frame_index required")
        seed, frame = int(seed), int(frame)
        passive, prime = options.get("passive", False), options.get("prime_only", False)
        if type(passive) is not bool or type(prime) is not bool or prime != passive:
            raise ValueError("Passive demonstration calls must be prime_only")
        session = self._session(ids[0], seed, resets[0])
        raw = options.get("executed_actions")
        raw = np.empty((0, 8), np.float32) if raw is None else np.asarray(raw, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 8 or not np.isfinite(raw).all():
            raise ValueError("executed_actions must be finite [C,8]")
        if session.frame is None:
            if len(raw):
                raise ValueError("First call cannot contain previous controls")
        else:
            gap = frame - session.frame
            if not 0 < gap <= self.stride or (session.passive and len(raw)) or (not session.passive and (len(raw) != gap or passive)):
                raise ValueError("Canonical observation/action cadence mismatch")

        step = self._to_vla_step_data(observations[0])
        step.is_demonstration = passive
        processed = self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
        state = torch.as_tensor(processed["state"]).reshape(1, -1).float().to(self.model.device)
        batch = self.collate_fn([processed])["inputs"]
        head = self.model.action_head
        head._memory_cache, head._vision_cache, head._inference_gen = session.short_cache, None, session.generator
        try:
            backbone_inputs, action_inputs = self.model.prepare_input(batch)
            raw_backbone = self.model.backbone(backbone_inputs)
            # Cache.py stores precisely this PRE-HAMLET, already normalized tail.
            # Calling VLLN here does not mutate raw_backbone; process_backbone_output
            # independently normalizes its original input exactly once.
            moment = head.vlln(raw_backbone["backbone_features"])[:, -self.n_q:].float()
            backbone = head.process_backbone_output(raw_backbone, action_inputs_B=1)
            features = backbone["backbone_features"]
            frozen_short = features[:, -self.n_q:]
            enabled = not self.memory_off and not prime and session.observations > 0
            with torch.autocast(device_type=self.model.device.type, enabled=False):
                result = self.representation.step(
                    frozen_short.float(), moment, state,
                    torch.tensor([frame], device=self.model.device),
                    torch.tensor([passive], dtype=torch.bool, device=self.model.device),
                    bank=session.long_bank, moment_history=session.moment_history,
                    read_enabled=enabled, write_enabled=True,
                )
            cast = result["fused"].to(frozen_short.dtype)
            adapted_short = result["short"].to(frozen_short.dtype)
            features = torch.cat((features[:, :-self.n_q], cast), dim=1)
            metrics = _scalar_metrics(result["metrics"])
            metrics.update(
                ae_conditioning_delta_norm=float((cast.float() - adapted_short.float()).norm()),
                ae_conditioning_changed_fraction=float((cast != adapted_short).float().mean()),
                short_adaptation_delta_norm=float((adapted_short.float() - frozen_short.float()).norm()),
            )
            if prime:
                prediction = features.new_zeros(1, head.action_horizon, head.action_dim)
            else:
                state_features = head.state_encoder(action_inputs.state, action_inputs.embodiment_id)
                prediction = head.get_action_with_features(features, state_features, action_inputs.embodiment_id, backbone)["action_pred"]
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite model action")
            session.long_bank = result["bank"].detach()
            session.moment_history = result["moment_history"].detach() if result["moment_history"] is not None else None
            session.observations += 1
            session.demo_updates += int(passive)
            diagnostics = {
                "policy": "fifo", "mode": self.mode, "memory_read_enabled": enabled,
                "frame_index": frame, "passive": passive, "observations_seen": session.observations,
                "memory_tokens": int(session.long_bank.shape[1]),
                "write_attempts": session.observations, "updates": session.observations, "keeps": 0,
                "demo_updates": session.demo_updates, "demo_keeps": 0, "read": metrics,
            }
            session.short_cache = head._memory_cache.detach().clone()
            session.raw_states = {key: np.array(value, copy=True) for key, value in step.states.items()}
            session.frame, session.passive = frame, passive
        except Exception:
            self.sessions.pop(ids[0], None)
            raise
        finally:
            head.reset_memory()
            head._inference_gen = None
        states = {key: np.stack([step.states[key]]) for key in self.modality_configs["state"].modality_keys}
        try:
            decoded = self.processor.decode_action(prediction.float().cpu().numpy(), self.embodiment_tag, states)
            actions = {key: np.asarray(value, dtype=np.float32) for key, value in decoded.items()}
            if any(not np.isfinite(value).all() for value in actions.values()):
                raise FloatingPointError("Nonfinite decoded physical action")
        except Exception:
            self.sessions.pop(ids[0], None)
            raise
        return actions, {"long_memory": diagnostics, "stage": self.stage, "episode_seed": seed,
            "expert_adapted": True, "checkpoint_variant": "representation_v18",
            "checkpoint_step": self.checkpoint_step, "memory_off": self.memory_off,
            "representation": self.representation.config.representation,
            "payload_sha256": self.payload_sha256}
