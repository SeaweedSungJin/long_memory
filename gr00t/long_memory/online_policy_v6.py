"""Observation-only temporal archive + direct Action Expert memory inference.

Original HAMLET preprocessing, short-memory cadence and seeded denoising stay
unchanged. The archive holds only observations already seen in this session;
neither demonstration actions nor future states/labels enter the policy.
Retrieval runs before the current observation is inserted into the archive.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import math

import numpy as np
import torch

from gr00t.data.types import MessageType

from .checkpoint_v6 import load_checkpoint_v6, v6_checkpoint_info
from .core_v6 import CANDIDATE_NAMES, VisualMemoryV6, VisualMemoryV6Config
from .expert_v4 import LoRAConfig, install_expert_lora, set_expert_trainable
from .expert_v6 import install_memory_bridge, with_memory
from .objectives_v6 import select_candidate
from .online_policy import LongMemoryPolicy, _Session
from .replay_v6 import bound_archive, build_candidates


@dataclass
class _VisualSession(_Session):
    # CPU detached rows bound session memory and cannot retain training graphs.
    observations: list[dict] = field(default_factory=list)
    observation_count: int = 0
    reads: dict[str, int] = field(default_factory=lambda: {
        name: 0 for name in ("uniform", "relevant", "hybrid", "null")})


class LongMemoryV6Policy(LongMemoryPolicy):
    """Same RoboMME wire protocol, with read-time CVOM instead of a writer."""

    def __init__(self, base_model, memory_checkpoint=None, device="cuda:0", strict=True,
                 read_policy="checkpoint", expert_only=False):
        if read_policy not in ("checkpoint", "uniform"):
            raise ValueError("V6 read policy must be checkpoint or uniform/fixed ablation")
        if memory_checkpoint is None and (read_policy != "checkpoint" or expert_only):
            raise ValueError("Original HAMLET has no adapted expert or retrieval policy to override")
        info = v6_checkpoint_info(base_model, memory_checkpoint) if memory_checkpoint else None
        super().__init__(base_model, memory_checkpoint=None, device=device, strict=strict)
        self.expert_only = bool(expert_only)
        self.read_policy_override = read_policy
        self.read_policy = "none"
        self.reader_mode = "none"
        self.cvom_threshold = 0.0
        if info is not None:
            cfg = info["config"]
            self.stage = cfg["stage"]
            self.reader_mode = cfg.get("reader_mode", "memory")
            head = self.model.action_head
            install_expert_lora(head, LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
            bridge_cfg = cfg["bridge"]
            installed = install_memory_bridge(head, **bridge_cfg)
            if installed != bridge_cfg:
                raise ValueError("Installed Action Expert bridge differs from checkpoint architecture")
            memory = VisualMemoryV6(VisualMemoryV6Config(**cfg["memory"]))
            load_checkpoint_v6(memory_checkpoint, memory, head)
            set_expert_trainable(head, False)
            head.eval().requires_grad_(False)
            memory.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
            if not expert_only and self.reader_mode != "none":
                self.memory = memory
                self.read_policy = "uniform" if self.stage == 1 or read_policy == "uniform" else "cvom"
            self.cvom_threshold = float(cfg.get("train", {}).get("cvom_threshold", 0.05))
            if not math.isfinite(self.cvom_threshold) or self.cvom_threshold < 0:
                raise ValueError("Saved CVOM threshold must be finite and nonnegative")
            if cfg.get("train", {}).get("fixed_policy", "uniform") != "uniform":
                raise ValueError("V6 checkpoint uses an unsupported fixed retrieval policy")
        # Legacy callers inspect this attribute only for diagnostics.
        self.write_policy = "observation-archive" if self.memory is not None else "disabled"

    def _session(self, sid, seed, reset):
        if reset:
            self.sessions.pop(sid, None)
        if sid not in self.sessions:
            generator = torch.Generator(device=self.model.device).manual_seed(seed)
            self.sessions[sid] = _VisualSession(generator, seed)
        session = self.sessions[sid]
        if session.episode_seed != seed:
            raise ValueError("Episode seed changed without resetting its session")
        self.sessions.move_to_end(sid)
        while len(self.sessions) > self.session_cap:
            self.sessions.popitem(last=False)
        return session

    @staticmethod
    def _observation_row(features, backbone, state, n_q, frame, passive):
        """Mirror the cached post-HAMLET feature boundary and mask defaults."""
        feature = features[0].detach().to(device="cpu", dtype=torch.bfloat16).clone()
        attention = backbone.get("backbone_attention_mask")
        images = backbone.get("image_mask")
        return {"features": feature,
                "attention_mask": torch.ones(len(feature), dtype=torch.bool) if attention is None
                    else attention[0].detach().to(device="cpu", dtype=torch.bool).clone(),
                "image_mask": torch.zeros(len(feature), dtype=torch.bool) if images is None
                    else images[0].detach().to(device="cpu", dtype=torch.bool).clone(),
                "short": feature[-n_q:].clone(), "state": state.detach().float().cpu().clone(),
                "frame": frame, "is_demo": passive}

    @torch.inference_mode()
    def _get_action(self, observation, options=None):
        options = options or {}
        observations = self._unbatch_observation(observation)
        if len(observations) != 1:
            raise ValueError("Temporal-memory RoboMME evaluation requires batch size 1")
        ids, flags = options.get("session_ids"), options.get("reset_memory", [False])
        if not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], str) or not ids[0]:
            raise ValueError("Provide exactly one nonempty session_id for causal memory")
        if not isinstance(flags, list) or len(flags) != 1 or type(flags[0]) is not bool:
            raise ValueError("reset_memory must be a one-element Boolean list")
        if "frame_index" not in options or "episode_seed" not in options:
            raise ValueError("Provide frame_index and episode_seed through run_long_memory_rollout.py")
        seed, frame = int(options["episode_seed"]), int(options["frame_index"])
        if seed < 0 or frame < 0:
            raise ValueError("Negative episode seed/frame")
        passive, prime = bool(options.get("passive", False)), bool(options.get("prime_only", False))
        if prime != passive:
            raise ValueError("Passive demo calls must be prime_only; execution calls generate actions")
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
        state = torch.as_tensor(processed["state"]).reshape(-1).float()
        batch = self.collate_fn([processed])["inputs"]
        head = self.model.action_head
        head._memory_cache, head._vision_cache, head._inference_gen = session.short_cache, None, session.generator
        try:
            backbone_inputs, action_inputs = self.model.prepare_input(batch)
            backbone = head.process_backbone_output(self.model.backbone(backbone_inputs), action_inputs_B=1)
            features = backbone["backbone_features"]
            tokens, selected, scores, event_ids = None, "null", {}, []
            current = None
            if self.memory is not None:
                current = self._observation_row(features, backbone, state, self.n_q, frame, passive)
                current["event_id"] = session.observation_count
                # Prime all demo observations, but do not run the AE or advance
                # its noise generator until the first actual action request.
                if not prime:
                    with torch.autocast(device_type=self.model.device.type, enabled=False):
                        built = build_candidates(self.memory, session.observations, current)
                        if self.read_policy == "cvom":
                            predictions = self.memory.score_candidates(built["query"], built["candidates"])
                            selected = select_candidate(predictions, improvement_margin=self.cvom_threshold,
                                                        fallback="uniform")
                            scores = {name: float(value) for name, value in zip(CANDIDATE_NAMES, predictions)}
                        else:
                            selected = "uniform"
                        candidate = built["candidates"][selected]
                        tokens, event_ids = candidate["tokens"], list(candidate["event_ids"])
                        if tokens is None:
                            selected = "null"
                    session.reads[selected] += 1
            diagnostics = {"policy": self.read_policy if self.memory is not None else
                           "expert-only" if self.expert_only else "trained-no-memory-control" if self.stage else "baseline",
                           "bank_fill": len(session.observations), "archive_fill": len(session.observations),
                           "frame_index": frame, "passive": passive, "selected_candidate": selected,
                           "selected_event_ids": event_ids, "candidate_scores": scores,
                           "read_counts": dict(session.reads), "candidate_reads": sum(session.reads.values()),
                           "observations_seen": session.observation_count,
                           "cvom_threshold": self.cvom_threshold}
            if prime:
                prediction = features.new_zeros(1, head.action_horizon, head.action_dim)
            else:
                state_features = head.state_encoder(action_inputs.state, action_inputs.embodiment_id)
                context = with_memory(head, tokens) if self.stage else nullcontext()
                with context:
                    prediction = head.get_action_with_features(features, state_features,
                                action_inputs.embodiment_id, backbone)["action_pred"]
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite model action; refusing to send controls")
            session.short_cache = head._memory_cache.detach().clone()
            session.raw_states = {key: np.array(value, copy=True) for key, value in step.states.items()}
            session.frame, session.passive = frame, passive
            if current is not None:
                # Exactly the same incremental temporal-thinning heuristic as
                # offline replay; preserve stable IDs and first/latest evidence.
                session.observations = bound_archive(session.observations + [current],
                                                     self.memory.config.max_archive_events)
                session.observation_count += 1
        except Exception:
            # Never reuse a partially advanced archive or noise generator.
            self.sessions.pop(ids[0], None)
            raise
        finally:
            head.reset_memory()
            head._inference_gen = None
        batched_states = {key: np.stack([step.states[key]]) for key in self.modality_configs["state"].modality_keys}
        decoded = self.processor.decode_action(prediction.float().cpu().numpy(), self.embodiment_tag, batched_states)
        actions = {key: np.asarray(value, dtype=np.float32) for key, value in decoded.items()}
        if any(not np.isfinite(value).all() for value in actions.values()):
            self.sessions.pop(ids[0], None)
            raise FloatingPointError("Nonfinite decoded physical action")
        return actions, {"long_memory": diagnostics, "stage": self.stage,
                         "episode_seed": seed, "expert_adapted": bool(self.stage)}
