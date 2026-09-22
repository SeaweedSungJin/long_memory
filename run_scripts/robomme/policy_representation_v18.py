"""Online V18: canonical HAMLET observations, strictly-past READ, then WRITE.

The original backbone, VLLN and short-memory cache are never rewritten. B's
separate short adapter sees the SAME normalized moments as cached training.
READ-off retains that adapted short representation and the same adapted AE.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from gr00t.data.types import MessageType
from gr00t.long_memory.online_policy import LongMemoryPolicy, _Session
from gr00t.long_memory.online_policy_v7 import _scalar_metrics
from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora, set_expert_trainable
from run_scripts.robomme.checkpoint_representation_v18 import checkpoint_info_v18, load_checkpoint_v18
from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18, RepresentationMemoryV18
from run_scripts.robomme.feature_precision_v19 import (
    extract_hamlet_features, feature_precision_contract, validate_feature_precision,
)


CVOM_RUNTIME_SOURCES = (
    "cvom_admission_core.py", "cvom_admission_checkpoint.py", "policy_representation_v18.py",
    "serve_representation_v18.py", "representation_core_v18.py", "checkpoint_representation_v18.py",
    "feature_precision_v19.py",
)


@dataclass
class RepresentationSession(_Session):
    long_bank: torch.Tensor | None = None
    moment_history: torch.Tensor | None = None
    observations: int = 0
    demo_updates: int = 0
    updates: int = 0
    keeps: int = 0
    demo_keeps: int = 0
    appends: int = 0
    replacements: int = 0
    demo_appends: int = 0
    demo_replacements: int = 0


class RepresentationPolicyV18(LongMemoryPolicy):
    def __init__(self, base_model, checkpoint, device="cuda:0", memory_off=False, strict=True, writer_checkpoint=None,
                 feature_precision="native", semantic_memory=False, semantic_fifo=False,
                 cvom_admission=False, cvom_fifo=False):
        if type(memory_off) is not bool:
            raise TypeError("memory_off must be boolean")
        if type(semantic_memory) is not bool or type(semantic_fifo) is not bool:
            raise TypeError("semantic_memory/semantic_fifo must be boolean")
        if semantic_fifo and not semantic_memory:
            raise ValueError("semantic_fifo requires semantic_memory")
        if semantic_memory and (writer_checkpoint is not None or feature_precision != "native"):
            raise ValueError("Semantic memory requires native precision and no legacy writer")
        if type(cvom_admission) is not bool or type(cvom_fifo) is not bool:
            raise TypeError("cvom_admission/cvom_fifo must be boolean")
        if cvom_fifo and not cvom_admission:
            raise ValueError("cvom_fifo requires cvom_admission")
        if cvom_admission and (writer_checkpoint is None or semantic_memory or feature_precision != "native"):
            raise ValueError("CVoM admission requires a writer checkpoint, native precision, and no semantic manager")
        self.feature_precision = validate_feature_precision(feature_precision)
        self.feature_precision_rules = feature_precision_contract(self.feature_precision)
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
        # Keep the copied HAMLET transformer's original BF16 precision. The
        # portable reader/LoRA deltas are already FP32; casting the WHOLE module
        # would silently change B's frozen short computation versus training.
        self.representation.to(device=device).eval().requires_grad_(False)
        self.memory_off = memory_off
        self.stage, self.mode, self.write_policy = 1, "representation_v18", "fifo"
        self.checkpoint_step = info["step"]
        self.payload_sha256 = info["metadata"]["payload_sha256"]
        self.writer = self.writer_callback = None
        self.writer_sha256 = None
        self.cvom_admission = cvom_admission
        self.semantic_memory = semantic_memory
        if semantic_memory:
            from run_scripts.robomme.semantic_memory_checkpoint import load_manager
            from run_scripts.robomme.train_archive_deployment_v9 import file_hash
            # New sidecar construction must not consume subsequent action RNG.
            with torch.random.fork_rng(devices=[]):
                self.writer, semantic = load_manager(checkpoint, device=device)
            self.stage = semantic["stage"]
            self.semantic_manifest_sha256 = file_hash(Path(checkpoint) / "semantic.json")
            self.semantic_payload_sha256 = semantic["payload_sha256"]
            self.storage_manager_sha256 = semantic["payload_sha256"].get("storage.safetensors")
            if semantic_fifo and self.writer is None:
                raise ValueError("Semantic FIFO control requires a checkpoint with a manager")
            if self.writer is not None and not semantic_fifo:
                self.writer_callback = self.writer.make_policy()
                self.writer_sha256 = self.storage_manager_sha256
                self.write_policy = "semantic-cvom"
        if cvom_admission:
            from run_scripts.robomme.cvom_admission_checkpoint import load_controller
            from run_scripts.robomme.train_archive_deployment_v9 import file_hash
            # Sidecar initialization must not advance the actor's action RNG.
            with torch.random.fork_rng(devices=[]):
                self.writer, admission = load_controller(
                    writer_checkpoint, checkpoint, device=device, base_model=base_model)
            writer_cfg = admission["config"]
            rcfg = self.representation.config
            if (writer_cfg["capacity_events"] != rcfg.capacity_events or rcfg.capacity_events != 32
                    or writer_cfg["dim"] != rcfg.hidden_dim or writer_cfg["num_tokens"] != rcfg.num_short_tokens):
                raise ValueError("CVoM writer must match the parent's 32-event reader dimensions")
            self.cvom_manifest_sha256 = admission["manifest_sha256"]
            self.cvom_writer_sha256 = admission["writer_sha256"]
            self.cvom_parent_identity = admission["parent_identity"]
            source_root = Path(__file__).resolve().parent
            self.cvom_source_sha256 = {name: file_hash(source_root / name) for name in CVOM_RUNTIME_SOURCES}
            if not cvom_fifo:
                self.writer_callback = self.writer.make_policy()
                self.writer_sha256 = self.cvom_writer_sha256
                self.write_policy = "cvom-admission"
        elif writer_checkpoint is not None:
            from run_scripts.robomme.storage_cvom_v18 import load_storage_writer_v18, make_write_policy
            self.writer, writer_cfg, writer_manifest = load_storage_writer_v18(writer_checkpoint, checkpoint, device=device)
            if (writer_cfg.capacity_events != self.representation.config.capacity_events
                    or writer_cfg.memory_dim != self.representation.config.hidden_dim):
                raise ValueError("Writer reader capacity/dimension differs")
            self.writer_callback = make_write_policy(self.writer)
            self.writer_sha256 = writer_manifest["writer_sha256"]
            self.write_policy = "cvom"
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
            # The precision option is scoped to feature extraction and its
            # cache boundary only. AE state encoding and denoising below keep
            # their original execution precision and session RNG behavior.
            backbone, moment = extract_hamlet_features(
                self.model, head, backbone_inputs, self.n_q, self.feature_precision,
            )
            features = backbone["backbone_features"]
            frozen_short = features[:, -self.n_q:]
            enabled = not self.memory_off and not prime and session.observations > 0
            previous_events = 0 if session.long_bank is None else session.long_bank.shape[1] // self.n_q
            with torch.autocast(device_type=self.model.device.type, enabled=False):
                result = self.representation.step(
                    frozen_short.float(), moment, state,
                    torch.tensor([frame], device=self.model.device),
                    torch.tensor([passive], dtype=torch.bool, device=self.model.device),
                    bank=session.long_bank, moment_history=session.moment_history,
                    read_enabled=enabled, write_enabled=True,
                    write_policy=self.writer_callback, event_index=session.observations,
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
            inserted = metrics.get("writer_insert", metrics.get("write_rate", 1.0)) == 1.0
            session.updates += int(inserted)
            session.keeps += int(not inserted)
            session.demo_updates += int(passive and inserted)
            session.demo_keeps += int(passive and not inserted)
            if getattr(self, "cvom_admission", False):
                capacity = self.representation.config.capacity_events
                append = inserted and previous_events < capacity
                replace = inserted and previous_events == capacity
                expected = {"writer_append": float(append), "writer_replace": float(replace),
                            "writer_keep": float(not inserted), "writer_insert": float(inserted)}
                if not 0 <= previous_events <= capacity or (previous_events < capacity and not append):
                    raise ValueError("CVoM must append until its bank is full")
                if self.writer_callback is not None and any(metrics.get(key) != value for key, value in expected.items()):
                    raise ValueError("CVoM writer metrics disagree with its admission decision")
                if session.long_bank.shape[1] != (previous_events + int(append)) * self.n_q:
                    raise ValueError("CVoM writer changed whole-event capacity unexpectedly")
                metrics.update(expected, writer_capacity=float(capacity))
                session.appends += int(append)
                session.replacements += int(replace)
                session.demo_appends += int(passive and append)
                session.demo_replacements += int(passive and replace)
            diagnostics = {
                "policy": self.write_policy, "mode": self.mode, "memory_read_enabled": enabled,
                "frame_index": frame, "passive": passive, "observations_seen": session.observations,
                "memory_tokens": int(session.long_bank.shape[1]),
                "write_attempts": session.observations, "updates": session.updates, "keeps": session.keeps,
                "demo_updates": session.demo_updates, "demo_keeps": session.demo_keeps, "read": metrics,
            }
            if getattr(self, "cvom_admission", False):
                diagnostics.update(writer_decision="append" if append else "replace:0" if replace else "keep",
                    bank_events_before=previous_events, capacity_events=capacity,
                    appends=session.appends, replacements=session.replacements,
                    demo_appends=session.demo_appends, demo_replacements=session.demo_replacements)
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
        info = {"long_memory": diagnostics, "stage": self.stage, "episode_seed": seed,
            "expert_adapted": True, "checkpoint_variant": "representation_v18",
            "checkpoint_step": self.checkpoint_step, "memory_off": self.memory_off,
            "representation": self.representation.config.representation,
            "feature_precision": self.feature_precision,
            "feature_precision_rules": self.feature_precision_rules,
            "writer_sha256": self.writer_sha256,
            "payload_sha256": self.payload_sha256}
        if getattr(self, "semantic_memory", False):
            info.update(semantic_manifest_sha256=self.semantic_manifest_sha256,
                semantic_payload_sha256=self.semantic_payload_sha256,
                semantic_stage=self.stage, storage_manager_sha256=self.storage_manager_sha256)
        if getattr(self, "cvom_admission", False):
            info.update(cvom_manifest_sha256=self.cvom_manifest_sha256,
                cvom_writer_sha256=self.cvom_writer_sha256, cvom_parent_identity=self.cvom_parent_identity,
                cvom_source_sha256=self.cvom_source_sha256)
        return actions, info
