"""Candidate V13 policy mixin ONLY; no constructor, checkpoint or live server.

Future construction must install an actual frozen VisualDemoTailMemoryV13 and
explicitly configure this mixin. Canonical V12 observations/APPEND/counts stay
unchanged. One image-only RPC, after all ordinary demo primes, commits a separate
tail bank. Only READ receives an ephemeral chronological union of the banks.

This is not an adopted policy or an online parity result. A future client must
skip this RPC when the exact missing-demo tail is empty (including n_demo=0).
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping

import numpy as np
import torch

from run_scripts.robomme.demo_tail_rng_v13 import isolated_visual_ingest_rng
from run_scripts.robomme.demo_tail_sidecar_v13 import extract_image_features
from run_scripts.robomme.visual_demo_tail_bank_v13 import (
    CAMERA_ORDER, DifferentialBank, VisualDemoTailMemoryV13,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _demo_frames(n_demo):
    return sorted({0, *range(n_demo - 16, -1, -16)}) if n_demo else []


@dataclass(frozen=True)
class DemoTailMetadataV13:
    session_id: str
    episode_seed: int
    n_demo: int
    frames: tuple[int, ...]


def _validate_bank(bank, *, hidden_dim, device, current_frame, n_demo):
    _require(isinstance(bank, DifferentialBank), "Expected a DifferentialBank")
    _require(isinstance(bank.tokens, torch.Tensor) and bank.tokens.ndim == 4
             and bank.tokens.shape[0] == 1 and bank.tokens.shape[2:] == (162, hidden_dim),
             "Online bank must be singleton [1,T,162,H]")
    for value in (bank.tokens, bank.content):
        _require(isinstance(value, torch.Tensor) and value.shape == bank.tokens.shape
                 and value.dtype == torch.float32 and value.device == device
                 and bool(torch.isfinite(value).all()), "Bank Z/C must be finite FP32 with equal layout/device")
    for name, dtype in (("frames", torch.int64), ("is_demo", torch.bool), ("valid", torch.bool)):
        value = getattr(bank, name)
        _require(isinstance(value, torch.Tensor) and value.shape == bank.tokens.shape[:2]
                 and value.dtype == dtype and value.device == device, f"Invalid bank {name} shape/dtype/device")
    # Live singleton APPEND has no padding. Reject it rather than silently
    # giving invalid rows identities or dropping unvalidated feature contents.
    _require(bool(bank.valid.all()), "Live bank cannot contain padded/invalid observations")
    raw = bank.frames[0]
    _require(bool((raw >= 0).all()) and bool((raw.diff() > 0).all())
             and bool((raw < current_frame).all()), "Bank frames must be unique, chronological and strictly past")
    _require(torch.equal(bank.is_demo[0], raw < n_demo), "Bank demo flags disagree with raw demo boundary")
    return raw


def merge_demo_tail_banks(canonical, tail, *, current_frame, n_demo, hidden_dim, device):
    """Pure singleton READ view. Never mutate/persist/APPEND this merged bank."""
    _require(type(current_frame) is int and type(n_demo) is int
             and 0 <= n_demo <= current_frame, "READ requires an execution raw frame and valid demo boundary")
    device = torch.device(device)
    raw = _validate_bank(canonical, hidden_dim=hidden_dim, device=device,
                         current_frame=current_frame, n_demo=n_demo)
    _require(raw[raw < n_demo].tolist() == _demo_frames(n_demo), "Canonical demo prefix differs from original stride16 primes")
    tail_raw = _validate_bank(tail, hidden_dim=hidden_dim, device=device,
                              current_frame=current_frame, n_demo=n_demo)
    last = _demo_frames(n_demo)[-1] if n_demo else -1
    expected = list(range(max(last + 1, n_demo - 15, 0), n_demo))
    _require(tail_raw.tolist() == expected and bool(tail.is_demo.all()), "Tail bank differs from exact uniform missing demo frames")
    if not expected:
        return canonical
    times = torch.cat((canonical.frames, tail.frames), dim=1)
    order = times[0].argsort(stable=True)
    _require(bool((times[0, order].diff() > 0).all()), "Duplicate canonical/tail raw frame")
    values = {name: torch.cat((getattr(canonical, name), getattr(tail, name)), dim=1)[:, order]
              for name in ("tokens", "frames", "is_demo", "valid", "content")}
    merged = DifferentialBank(**values)
    _validate_bank(merged, hidden_dim=hidden_dim, device=device, current_frame=current_frame, n_demo=n_demo)
    return merged


@contextmanager
def _method(module, name, replacement):
    existed, original = name in module.__dict__, module.__dict__.get(name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        if existed:
            setattr(module, name, original)
        else:
            delattr(module, name)


def _snapshot(value):
    """Reference-sensitive snapshot of the parent's small mutable session state."""
    if isinstance(value, torch.Tensor):
        return ("tensor", value, value.detach().clone())
    if isinstance(value, np.ndarray):
        return ("numpy", value, value.copy())
    if isinstance(value, torch.Generator):
        return ("generator", value, value.get_state().clone())
    if is_dataclass(value) and not isinstance(value, type):
        return ("object", value, {field.name: _snapshot(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, dict):
        return ("dict", value, {key: _snapshot(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return ("sequence", value, [_snapshot(item) for item in value])
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        return ("scalar", value, value)
    raise TypeError(f"Unsupported mutable parent snapshot value: {type(value).__name__}")


def _unchanged(value, snapshot):
    kind, original, saved = snapshot
    if kind == "scalar":
        return type(value) is type(original) and value == saved
    if value is not original:
        return False
    if kind == "tensor":
        return value.dtype == saved.dtype and value.device == saved.device and torch.equal(value, saved)
    if kind == "numpy":
        return value.dtype == saved.dtype and np.array_equal(value, saved)
    if kind == "generator":
        return torch.equal(value.get_state(), saved)
    if kind == "object":
        return all(_unchanged(getattr(value, key), item) for key, item in saved.items())
    if kind == "dict":
        return list(value) == list(saved) and all(_unchanged(value[key], item) for key, item in saved.items())
    return len(value) == len(saved) and all(_unchanged(item, before) for item, before in zip(value, saved))


def _restore(snapshot):
    """Best-effort rollback on failure; unchanged tensors are not copied/bumped."""
    kind, original, saved = snapshot
    if kind == "tensor":
        if not torch.equal(original, saved):
            original.copy_(saved)
    elif kind == "numpy":
        if not np.array_equal(original, saved):
            np.copyto(original, saved)
    elif kind == "generator":
        if not torch.equal(original.get_state(), saved):
            original.set_state(saved)
    elif kind == "object":
        for key, item in saved.items():
            object.__setattr__(original, key, _restore(item))
    elif kind == "dict":
        original.clear()
        original.update((key, _restore(item)) for key, item in saved.items())
    elif kind == "sequence":
        restored = [_restore(item) for item in saved]
        if isinstance(original, list):
            original[:] = restored
    return original


class DemoTailPolicyMixinV13:
    """Place BEFORE VisualDifferentialV12Policy in a future policy's MRO.

    No construction or checkpoint relabeling occurs here. Only an explicitly
    configured VisualDemoTailMemoryV13 is accepted. The canonical policy lock
    encloses action/reset calls; this RPC acquires that same lock itself.
    """

    def configure_demo_tail_ingest(self, *, enabled):
        _require(type(enabled) is bool, "Demo-tail enable flag must be explicit bool")
        _require(not hasattr(self, "_demo_tail_ingest_enabled"), "Configure demo ingestion once, during construction")
        _require(not self.sessions, "Configure demo ingestion before any session exists")
        self._check_demo_tail_modules()
        self._demo_tail_ingest_enabled = enabled

    def _check_demo_tail_modules(self):
        _require(isinstance(self.visual_memory, VisualDemoTailMemoryV13), "Future constructor must install actual VisualDemoTailMemoryV13")
        _require(tuple(self.visual_camera_order) == CAMERA_ORDER, "Demo camera order mismatch")
        _require(hasattr(self, "_visual_call_lock"), "The existing visual action/reset lock is required")
        for module in (self.model, self.visual_memory):
            _require(not module.training and all(not p.requires_grad and p.grad is None for p in module.parameters()),
                     "Ingestion requires frozen eval modules without gradients")
        _require(self.visual_memory._device() == self.model.device, "Visual/model devices differ")

    def _tail_bound_session(self, sid, seed):
        # Match the unchanged parent's accepted action-protocol integer types.
        if (not isinstance(sid, str) or not sid or isinstance(seed, bool)
                or not isinstance(seed, (int, np.integer)) or seed < 0):
            return None
        session = self.sessions.get(sid)
        return session if session is not None and session.episode_seed == seed else None

    def _invalidate_tail_session(self, sid, session):
        if session is not None and self.sessions.get(sid) is session:
            self.sessions.pop(sid)

    @torch.inference_mode()
    def ingest_demo_tail(self, *, session_id=None, episode_seed=None, n_demo=None,
                         frames=None, images=None, texts=None, **unexpected):
        """One transactional RGB/text-only RPC after canonical demo priming.

        Failed lock acquisition or an unknown/wrong-seed request is unbound and
        cannot evict another live episode. Every failure after exact binding
        invalidates that session. No body work or state creation precedes lock.
        """
        if not self._visual_call_lock.acquire(blocking=False):
            raise RuntimeError("Demo ingestion is serial; action/reset/ingest cannot overlap")
        session, snapshots, head_snapshots, session_table = None, None, None, None
        source_snapshot = None
        try:
            session = self._tail_bound_session(session_id, episode_seed)
            _require(session is not None, "Unknown session or mismatched episode seed")
            episode_seed = int(episode_seed)
            _require(not unexpected, f"Unexpected/nonvisual ingest fields: {sorted(unexpected)}")
            _require(getattr(self, "_demo_tail_ingest_enabled", None) is True, "Demo-tail ingestion must be explicitly enabled")
            self._check_demo_tail_modules()
            _require(type(n_demo) is int and n_demo >= 0, "n_demo must be a nonnegative raw-frame integer")
            canonical = _demo_frames(n_demo)
            last = canonical[-1] if canonical else -1
            expected = list(range(max(last + 1, n_demo - 15, 0), n_demo))
            _require(bool(expected), "Client must skip RPC for an empty demo tail, including n_demo=0")
            _require(type(frames) is list and all(type(frame) is int for frame in frames)
                     and frames == expected, "Require exact unique chronological demo tail; exclude current/future frames")
            _require(session.passive is True and session.frame == last, "Ingest only immediately after the last canonical demo prime")
            _require(not hasattr(session, "visual_tail_bank") and not hasattr(session, "visual_tail_metadata"), "Demo tail already ingested")
            bank = session.visual_bank
            raw = _validate_bank(bank, hidden_dim=self.visual_memory.config.hidden_dim,
                                 device=self.model.device, current_frame=n_demo, n_demo=n_demo)
            _require(raw.tolist() == canonical and session.observed == len(canonical)
                     and session.visual_demo_updates == len(canonical), "Canonical observations/counts differ from original demo primes")
            _require(isinstance(images, Mapping) and set(images) == set(CAMERA_ORDER), "Require exactly front/wrist RGB images")
            for camera in CAMERA_ORDER:
                rgb = images[camera]
                _require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 4
                         and rgb.shape[0] == len(frames) and rgb.shape[-1] == 3
                         and rgb.shape[1] > 0 and rgb.shape[2] > 0, "Expected uint8 camera images [N,H,W,3]")
            _require(type(texts) is list and len(texts) == len(frames)
                     and all(isinstance(text, str) for text in texts), "Each image observation needs its own instruction string")
            snapshots = {key: _snapshot(value) for key, value in vars(session).items()}
            session_table = (self.sessions, list(self.sessions.items()))
            order = tuple((key, id(value)) for key, value in self.sessions.items())
            head = self.model.action_head
            head_snapshots = {key: (hasattr(head, key), _snapshot(getattr(head, key, None)))
                              for key in ("_memory_cache", "_vision_cache", "_inference_gen")}
            source_snapshot = _snapshot(dict(images))
            candidate = self.visual_memory.empty_bank(1)

            def forbidden(*args, **kwargs):
                raise RuntimeError("Image ingestion cannot call Action Expert or HAMLET short-memory processing")

            with isolated_visual_ingest_rng(self.model.device, session.generator), ExitStack() as guards:
                for name in ("process_backbone_output", "get_action_with_features", "get_action", "forward"):
                    guards.enter_context(_method(head, name, forbidden))
                if isinstance(getattr(head, "model", None), torch.nn.Module):
                    guards.enter_context(_method(head.model, "forward", forbidden))
                for index, frame in enumerate(frames):
                    feature, _ = extract_image_features(self.model.backbone, head.vlln, self.processor,
                        {camera: [images[camera][index]] for camera in CAMERA_ORDER}, texts[index],
                        device=self.model.device, embodiment=self.embodiment_tag.value)
                    _require(feature.dtype == torch.bfloat16
                             and feature.shape == (2, 81, self.visual_memory.config.feature_dim)
                             and bool(torch.isfinite(feature).all()), "Invalid original image-only extraction")
                    record = self.visual_memory.encode_bank_images(feature[None].to(self.model.device),
                        [frame], [True], camera_order=self.visual_camera_order)
                    candidate = self.visual_memory.append_bank_images(candidate, record)
            # RNG restoration and method restoration must both succeed BEFORE
            # checking/committing any new evidence or metadata to the session.
            _require(all(_unchanged(getattr(session, key), value) for key, value in snapshots.items())
                     and set(vars(session)) == set(snapshots), "Parent session changed during image ingestion")
            _require(self.sessions is session_table[0]
                     and tuple((key, id(value)) for key, value in self.sessions.items()) == order,
                     "Session table/order changed during image ingestion")
            _require(all(hasattr(head, key) == existed and _unchanged(getattr(head, key, None), snapshot)
                         for key, (existed, snapshot) in head_snapshots.items()), "Head cache/generator changed during image ingestion")
            _require(all(_unchanged(images[key], value) for key, value in source_snapshot[2].items()), "Raw RGB input changed during ingestion")
            merge_demo_tail_banks(bank, candidate, current_frame=n_demo, n_demo=n_demo,
                                 hidden_dim=self.visual_memory.config.hidden_dim, device=self.model.device)
            self._check_demo_tail_modules()
            session.visual_tail_bank = candidate
            session.visual_tail_metadata = DemoTailMetadataV13(session_id, episode_seed, n_demo, tuple(frames))
            return {"session_id": session_id, "episode_seed": episode_seed, "n_demo": n_demo,
                    "ingested_observations": len(frames), "first_tail_frame": frames[0], "last_tail_frame": frames[-1],
                    "canonical_observations": session.observed, "canonical_frame": session.frame,
                    "parent_unchanged": True, "rng_preserved": True, "read_performed": False}
        except BaseException:
            # Restore old references/values even when a guard caught a bad
            # collaborator, then invalidate exactly the bound episode.
            try:
                if snapshots is not None:
                    for key in tuple(vars(session)):
                        if key not in snapshots:
                            delattr(session, key)
                    for key, snapshot in snapshots.items():
                        setattr(session, key, _restore(snapshot))
            finally:
                try:
                    if head_snapshots is not None:
                        for key, (existed, snapshot) in head_snapshots.items():
                            if existed:
                                setattr(self.model.action_head, key, _restore(snapshot))
                            elif hasattr(self.model.action_head, key):
                                delattr(self.model.action_head, key)
                finally:
                    try:
                        if source_snapshot is not None:
                            _restore(source_snapshot)
                    finally:
                        try:
                            if session_table is not None:
                                table, items = session_table
                                # A guarded collaborator must not evict/reorder
                                # unrelated sessions, even on a failing call.
                                self.sessions = table
                                table.clear()
                                table.update(items)
                        finally:
                            self._invalidate_tail_session(session_id, session)
            raise
        finally:
            self._visual_call_lock.release()

    def _visual_get_action(self, observation, options):
        """Inherited V12 code owns canonical processing, READ, APPEND and decode."""
        _require(hasattr(self, "_demo_tail_ingest_enabled"), "Future constructor must explicitly configure demo ingestion")
        _require(self._visual_call_lock.locked(), "Inherited visual action lock must enclose this hook")
        options = options or {}
        ids = options.get("session_ids")
        sid = ids[0] if isinstance(ids, list) and len(ids) == 1 else None
        seed = options.get("episode_seed")
        bound = self._tail_bound_session(sid, seed)
        reset = options.get("reset_memory", [False]) == [True]
        original_read = self.visual_memory.read
        diagnostic = {"enabled": self._demo_tail_ingest_enabled, "read_enabled": False,
                      "tail_observations": 0, "canonical_prior_observations": 0,
                      "effective_prior_observations": 0}

        def read(current, canonical=None, *, enabled=True):
            nonlocal bound
            # Resolve AFTER the parent validates/creates/resets the session.
            bound = self._tail_bound_session(sid, seed)
            _require(bound is not None and canonical is bound.visual_bank, "READ must bind the parent's exact current session/canonical bank")
            diagnostic["canonical_prior_observations"] = canonical.tokens.shape[1]
            diagnostic["effective_prior_observations"] = canonical.tokens.shape[1]
            tail = getattr(bound, "visual_tail_bank", None)
            metadata = getattr(bound, "visual_tail_metadata", None)
            if self._demo_tail_ingest_enabled and tail is not None:
                _require(isinstance(metadata, DemoTailMetadataV13) and metadata.session_id == sid
                         and metadata.episode_seed == seed, "Tail bank belongs to another session/episode")
                diagnostic["tail_observations"] = tail.tokens.shape[1]
                diagnostic["n_demo"] = metadata.n_demo
                diagnostic["last_tail_frame"] = metadata.frames[-1]
                if enabled is True:
                    _require(current.frames.shape == (1,) and not bool(current.is_demo[0]), "Tail READ is action-only")
                    merged = merge_demo_tail_banks(canonical, tail, current_frame=int(current.frames[0]),
                        n_demo=metadata.n_demo, hidden_dim=self.visual_memory.config.hidden_dim, device=self.model.device)
                    _require(tail.frames[0].tolist() == list(metadata.frames), "Committed tail frame identity changed")
                    diagnostic["effective_prior_observations"] = merged.tokens.shape[1]
                    diagnostic["read_enabled"] = True
                    return original_read(current, merged, enabled=enabled)
            # Exact original objects, including when visual READ is off.
            return original_read(current, canonical, enabled=enabled)

        try:
            if (self._demo_tail_ingest_enabled and bound is not None and not reset
                    and bound.passive and not bool(options.get("passive", False))):
                raw_frame = options.get("frame_index")
                _require(not isinstance(raw_frame, bool) and isinstance(raw_frame, (int, np.integer))
                         and raw_frame >= 0, "First execution requires an integer n_demo frame")
                n_demo = int(raw_frame)
                canonical_frames = _demo_frames(n_demo)
                raw = _validate_bank(bound.visual_bank, hidden_dim=self.visual_memory.config.hidden_dim,
                    device=self.model.device, current_frame=n_demo, n_demo=n_demo)
                _require(raw.tolist() == canonical_frames and bound.observed == len(canonical_frames)
                         and bound.visual_demo_updates == len(canonical_frames),
                         "First execution requires the exact canonical demo prime prefix")
                last = canonical_frames[-1] if canonical_frames else -1
                if max(last + 1, n_demo - 15, 0) < n_demo:
                    # READ-off is an ablation of READ only: it must still prove
                    # that the same nonempty tail was actually ingested.
                    _require(hasattr(bound, "visual_tail_bank") and hasattr(bound, "visual_tail_metadata"),
                             "Missing required demo-tail ingest before first execution")
            if bound is not None and not reset and hasattr(bound, "visual_tail_metadata"):
                metadata = bound.visual_tail_metadata
                _require(not options.get("passive", False), "No further demo primes after tail ingestion")
                if bound.passive:
                    _require(options.get("frame_index") == metadata.n_demo, "First execution observation must equal n_demo")
            with _method(self.visual_memory, "read", read):
                actions, info = super()._visual_get_action(observation, options)
            return actions, {**info, "demo_tail": diagnostic}
        except BaseException:
            self._invalidate_tail_session(sid, self._tail_bound_session(sid, seed))
            raise
