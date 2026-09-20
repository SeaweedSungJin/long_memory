"""V8 causal event-memory inference with the original HAMLET short pathway.

Moment-source features are normalized before HAMLET short-memory processing,
at the same location as cache.py. Events are appended only after action READ.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from gr00t.data.types import MessageType

from .checkpoint_v8 import load_checkpoint_v8, v8_checkpoint_info
from .event_v8 import EventMemoryV8, MemoryV8Config
from .expert_v4 import LoRAConfig, install_expert_lora, set_expert_trainable
from .online_policy import LongMemoryPolicy, _Session


@dataclass
class _EventSession(_Session):
    latent: torch.Tensor | None = None
    observed: int = 0
    write_attempts: int = 0
    appended_events: int = 0
    demo_appended_events: int = 0
    evicted_events: int = 0


def _scalar_metrics(values):
    result = {}
    for name, value in values.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.detach().item()
        if isinstance(value, (bool, int, float)):
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite event-memory diagnostic: {name}")
            result[name] = value
    return result


def process_memory_features(head, backbone_output, num_tokens, *, source):
    """Return processed backbone and cached-source tokens without double VLLN.

    The extra normalization call for moment extraction never replaces the raw
    backbone tensor that process_backbone_output will normalize itself.
    Clone the selected moment so an in-place head implementation cannot alias it.
    """
    if source not in ("moment", "short", None):
        raise ValueError("V8 source must be moment or short")
    moment = None
    if source == "moment":
        moment = head.vlln(backbone_output["backbone_features"])[:, -num_tokens:].clone()
    processed = head.process_backbone_output(backbone_output, action_inputs_B=1)
    selected = moment if source == "moment" else processed["backbone_features"][:, -num_tokens:]
    return processed, selected


class LongMemoryV8Policy(LongMemoryPolicy):
    def __init__(self, base_model, memory_checkpoint=None, device="cuda:0", strict=True,
                 memory_off=False):
        if not isinstance(memory_off, bool):
            raise ValueError("memory_off must be Boolean")
        if memory_checkpoint is None and memory_off:
            raise ValueError("Original HAMLET has no adapted reader to disable")
        info = v8_checkpoint_info(base_model, memory_checkpoint, expected_stage=1) if memory_checkpoint else None
        if info is not None and memory_off and info["config"]["mode"] != "event":
            raise ValueError("memory-off requires the same event reader checkpoint")
        super().__init__(base_model, memory_checkpoint=None, device=device, strict=strict)
        self.mode, self.memory_off, self.source = "none", memory_off, None
        if info is not None:
            cfg = info["config"]
            self.stage, self.mode = 1, cfg["mode"]
            head = self.model.action_head
            install_expert_lora(head, LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
            self.memory = EventMemoryV8(MemoryV8Config(**cfg["memory"]))
            load_checkpoint_v8(memory_checkpoint, self.memory, head)
            self.source = self.memory.config.source
            set_expert_trainable(head, False)
            head.eval().requires_grad_(False)
            self.memory.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        self.write_policy = "append_fifo" if self.mode == "event" else "none"

    def _session(self, sid, seed, reset):
        if reset:
            self.sessions.pop(sid, None)
        if sid not in self.sessions:
            generator = torch.Generator(device=self.model.device).manual_seed(seed)
            bank = self.memory.initial_state() if self.memory is not None else None
            self.sessions[sid] = _EventSession(generator, seed, latent=bank)
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
            raise ValueError("V8 RoboMME evaluation requires batch size 1")
        ids, flags = options.get("session_ids"), options.get("reset_memory", [False])
        if not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], str) or not ids[0]:
            raise ValueError("Provide exactly one nonempty session_id")
        if not isinstance(flags, list) or len(flags) != 1 or type(flags[0]) is not bool:
            raise ValueError("reset_memory must be a one-element Boolean list")
        if "frame_index" not in options or "episode_seed" not in options:
            raise ValueError("Provide frame_index and episode_seed via run_long_memory_rollout.py")
        seed, frame = options["episode_seed"], options["frame_index"]
        if isinstance(seed, bool) or isinstance(frame, bool) or not isinstance(seed, (int, np.integer)) or not isinstance(frame, (int, np.integer)):
            raise ValueError("Frame and seed must be integers")
        seed, frame = int(seed), int(frame)
        if seed < 0 or frame < 0:
            raise ValueError("Negative episode seed/frame")
        passive, prime = options.get("passive", False), options.get("prime_only", False)
        if type(passive) is not bool or type(prime) is not bool or prime != passive:
            raise ValueError("Passive demo calls must have Boolean prime_only=True")
        session = self._session(ids[0], seed, flags[0])
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
                raise ValueError("Observation cadence differs from trained memory stride")
            if session.passive and len(raw):
                raise ValueError("Demo transitions have no executed robot actions")
            if not session.passive and (len(raw) != gap or passive):
                raise ValueError("Missing/misaligned actually executed action prefix")

        step = self._to_vla_step_data(observations[0])
        step.is_demonstration = passive
        processed = self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
        state = torch.as_tensor(processed["state"]).reshape(1, -1).float()
        batch = self.collate_fn([processed])["inputs"]
        head = self.model.action_head
        head._memory_cache, head._vision_cache, head._inference_gen = session.short_cache, None, session.generator
        try:
            backbone_inputs, action_inputs = self.model.prepare_input(batch)
            backbone, source_tokens = process_memory_features(head, self.model.backbone(backbone_inputs), self.n_q,
                source=self.source if self.mode == "event" else None)
            features = backbone["backbone_features"]
            short = features[:, -self.n_q:]
            encoded, read_metrics, write_metrics = None, {}, {}
            read_enabled = False
            if self.memory is not None and self.mode == "event":
                with torch.autocast(device_type=self.model.device.type, enabled=False):
                    encoded = self.memory.encode(source_tokens.float(), state.to(self.model.device),
                        torch.tensor([frame], device=self.model.device),
                        torch.tensor([passive], dtype=torch.bool, device=self.model.device))
                    if not prime:
                        read_enabled = not self.memory_off and session.latent.shape[1] > 0
                        fused, read_metrics = self.memory.read(short.float(), encoded, session.latent,
                            mode="event", memory_enabled=read_enabled)
                        cast = fused.to(short.dtype)
                        read_metrics = dict(read_metrics,
                            ae_conditioning_delta_norm=(cast.float() - short.float()).norm(),
                            ae_conditioning_changed_fraction=(cast != short).float().mean())
                        features = torch.cat((features[:, :-self.n_q], cast), dim=1)
            if prime:
                prediction = features.new_zeros(1, head.action_horizon, head.action_dim)
            else:
                state_features = head.state_encoder(action_inputs.state, action_inputs.embodiment_id)
                prediction = head.get_action_with_features(features, state_features,
                            action_inputs.embodiment_id, backbone)["action_pred"]
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite model action; refusing to send controls")
            if encoded is not None:
                previous_events = session.latent.shape[1] // self.n_q
                following, write_metrics = self.memory.write(session.latent, encoded)
                session.latent = following.detach()
                retained = session.latent.shape[1] // self.n_q
                if session.latent.shape[1] % self.n_q or retained != min(previous_events + 1, self.memory.config.capacity):
                    raise ValueError("V8 WRITE did not append one complete bounded event")
                session.write_attempts += 1
                session.appended_events += 1
                session.demo_appended_events += int(passive)
                session.evicted_events += previous_events + 1 - retained
                if not torch.isfinite(session.latent).all():
                    raise FloatingPointError("Nonfinite event-memory state")
            session.observed += 1
            tokens = int(session.latent.shape[1]) if session.latent is not None else 0
            diagnostics = {"policy": self.write_policy if self.stage else "baseline", "mode": self.mode,
                "source": self.source, "memory_read_enabled": read_enabled, "frame_index": frame,
                "passive": passive, "observations_seen": session.observed,
                "retained_events": tokens // self.n_q, "memory_tokens": tokens,
                "capacity_events": self.memory.config.capacity if self.memory is not None else 0,
                "tokens_per_event": self.n_q, "write_attempts": session.write_attempts,
                "appended_events": session.appended_events, "demo_appended_events": session.demo_appended_events,
                "evicted_events": session.evicted_events,
                "read": _scalar_metrics(read_metrics), "write": _scalar_metrics(write_metrics)}
            session.short_cache = head._memory_cache.detach().clone()
            session.raw_states = {key: np.array(value, copy=True) for key, value in step.states.items()}
            session.frame, session.passive = frame, passive
        except Exception:
            self.sessions.pop(ids[0], None)
            raise
        finally:
            head.reset_memory()
            head._inference_gen = None
        batched_states = {key: np.stack([step.states[key]]) for key in self.modality_configs["state"].modality_keys}
        try:
            decoded = self.processor.decode_action(prediction.float().cpu().numpy(), self.embodiment_tag, batched_states)
            actions = {key: np.asarray(value, dtype=np.float32) for key, value in decoded.items()}
            if any(not np.isfinite(value).all() for value in actions.values()):
                raise FloatingPointError("Nonfinite decoded physical action")
        except Exception:
            self.sessions.pop(ids[0], None)
            raise
        return actions, {"long_memory": diagnostics, "stage": self.stage,
                         "episode_seed": seed, "expert_adapted": bool(self.stage)}
