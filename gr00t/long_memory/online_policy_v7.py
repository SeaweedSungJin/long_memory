"""V7 recurrent short-memory READ -> action -> WRITE, one state per session.

The original HAMLET observation processor, short-memory cache, action decoding,
episode noise and demo cadence are retained. No action, target or future state
enters WRITE. Archive/none are separately trained controls, not renamed v6.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from gr00t.data.types import MessageType

from .checkpoint_v7 import load_checkpoint_v7, v7_checkpoint_info
from .cvom_v7 import CVOMV7
from .expert_v4 import LoRAConfig, install_expert_lora, set_expert_trainable
from .online_policy import LongMemoryPolicy, _Session
from .recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from .replay_v7 import apply_write, initial_replay_state


@dataclass
class _RecurrentSession(_Session):
    latent: torch.Tensor | None = None
    observed: int = 0
    write_attempts: int = 0
    updates: int = 0
    keeps: int = 0
    demo_updates: int = 0
    demo_keeps: int = 0


def _scalar_metrics(values):
    """Only finite scalar diagnostics leave the policy over msgpack/JSON."""
    result = {}
    for name, value in values.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.detach().item()
        if isinstance(value, (bool, int, float)):
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite recurrent diagnostic: {name}")
            result[name] = value
    return result


class LongMemoryV7Policy(LongMemoryPolicy):
    def __init__(self, base_model, memory_checkpoint=None, device="cuda:0", strict=True,
                 write_policy="checkpoint", memory_off=False):
        if write_policy not in ("checkpoint", "update"):
            raise ValueError("V7 write_policy must be checkpoint or explicit always-update")
        if memory_checkpoint is None and (write_policy != "checkpoint" or memory_off):
            raise ValueError("Original HAMLET has no adapted memory policy to override")
        info = v7_checkpoint_info(base_model, memory_checkpoint) if memory_checkpoint else None
        if info is not None:
            cfg = info["config"]
            threshold = cfg.get("train", {}).get("cvom_threshold", 0.05)
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
                raise ValueError("Saved CVOM threshold must be finite and nonnegative")
            if memory_off and (cfg["stage"] != 1 or cfg["mode"] != "recurrent"):
                raise ValueError("memory-off is the same Stage-1 recurrent actor's READ-off ablation")
            if write_policy != "checkpoint" and cfg["mode"] != "recurrent":
                raise ValueError("Only recurrent memory supports an always-UPDATE override")
        super().__init__(base_model, memory_checkpoint=None, device=device, strict=strict)
        self.mode, self.cvom, self.cvom_threshold = "none", None, 0.05
        self.memory_off, self.write_policy_override = bool(memory_off), write_policy
        if info is not None:
            self.stage, self.mode = cfg["stage"], cfg["mode"]
            self.cvom_threshold = float(threshold)
            head = self.model.action_head
            install_expert_lora(head, LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
            memory_cfg = MemoryV7Config(**cfg["memory"])
            self.memory, self.cvom = RecurrentMemoryV7(memory_cfg), CVOMV7(memory_cfg)
            load_checkpoint_v7(memory_checkpoint, self.memory, head, self.cvom)
            set_expert_trainable(head, False)
            head.eval().requires_grad_(False)
            self.memory.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
            self.cvom.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        self.write_policy = "none" if self.mode == "none" else "append" if self.mode == "archive" else (
            "cvom" if self.stage == 2 and write_policy == "checkpoint" else "update")

    def _session(self, sid, seed, reset):
        if reset:
            self.sessions.pop(sid, None)
        if sid not in self.sessions:
            generator = torch.Generator(device=self.model.device).manual_seed(seed)
            state = None
            if self.memory is not None:
                state = initial_replay_state(self.memory, mode=self.mode, batch_size=1)
            self.sessions[sid] = _RecurrentSession(generator, seed, latent=state)
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
            raise ValueError("Recurrent RoboMME evaluation requires batch size 1")
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
        passive, prime = bool(options.get("passive", False)), bool(options.get("prime_only", False))
        if prime != passive:
            raise ValueError("Passive demo calls must be prime_only")
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
            backbone = head.process_backbone_output(self.model.backbone(backbone_inputs), action_inputs_B=1)
            features = backbone["backbone_features"]
            short = features[:, -self.n_q:]
            encoded, read_metrics, write_metrics = None, {}, {}
            read_enabled = False
            if self.memory is not None and self.mode != "none":
                with torch.autocast(device_type=self.model.device.type, enabled=False):
                    encoded = self.memory.encode(short.float(), state.to(self.model.device),
                        torch.tensor([frame], device=self.model.device),
                        torch.tensor([passive], dtype=torch.bool, device=self.model.device))
                    if not prime:
                        # READ uses M_t, never the candidate containing X_t.
                        read_enabled = not self.memory_off and session.updates > 0
                        fused, read_metrics = self.memory.read(short.float(), encoded, session.latent,
                            mode=self.mode, memory_enabled=read_enabled)
                        cast = fused.to(short.dtype)
                        read_metrics = dict(read_metrics, ae_conditioning_delta_norm=(cast.float() - short.float()).norm(),
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

            # Write every observed endpoint, INCLUDING every canonical demo
            # endpoint. Physically executed/GT actions are never model inputs.
            if encoded is not None:
                previous = session.latent
                critic = self.cvom if self.write_policy == "cvom" else None
                following, write_metrics = apply_write(self.memory, previous, encoded, mode=self.mode,
                                                       cvom=critic, threshold=self.cvom_threshold)
                session.latent = following.detach()
                session.write_attempts += 1
                # Shared apply_write reports a policy decision, not a test of
                # tensor inequality: an UPDATE may legitimately leave M equal.
                decision = float(write_metrics["write_rate"])
                if decision not in (0.0, 1.0):
                    raise ValueError("Single-session WRITE must choose exactly KEEP or UPDATE")
                update = decision == 1.0
                session.updates += int(update)
                session.keeps += int(not update)
                session.demo_updates += int(passive and update)
                session.demo_keeps += int(passive and not update)
                if not torch.isfinite(session.latent).all():
                    raise FloatingPointError("Nonfinite recurrent memory state")
            session.observed += 1
            diagnostics = {"policy": self.write_policy if self.stage else "baseline", "mode": self.mode,
                "memory_read_enabled": read_enabled, "frame_index": frame,
                "passive": passive, "observations_seen": session.observed,
                "memory_tokens": int(session.latent.shape[1]) if session.latent is not None else 0,
                "write_attempts": session.write_attempts, "updates": session.updates, "keeps": session.keeps,
                "demo_updates": session.demo_updates, "demo_keeps": session.demo_keeps,
                "cvom_threshold": self.cvom_threshold,
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
