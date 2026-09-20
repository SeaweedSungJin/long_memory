"""Immutable V7 actor/CVOM/Action-Expert bundles; never rewrite original HAMLET.

The latent state of an episode is NOT a model parameter and is not persisted
here. A checkpoint contains the learned update/read functions, separate critic,
AE LoRA, and optimizer-boundary training state needed for an exact resume.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .checkpoint_v4 import (file_sha256, _state_sha256, _expected_expert_shapes,
                            _validate_installed_head)
from .cvom_v7 import CVOMV7
from .expert_v4 import expert_state_dict, load_expert_state_dict
from .hamlet import checkpoint_identity
from .monitoring import _atomic_json, _restore_rng_state, _rng_state, _validate_step
from .recurrent_v7 import MemoryV7Config, RecurrentMemoryV7

VARIANT = "recurrent_memory_v7"
PAYLOADS = ("model.safetensors", "expert.safetensors", "cvom.safetensors")


def actor_state_sha256(module_or_state):
    return _state_sha256(module_or_state.state_dict() if hasattr(module_or_state, "state_dict") else module_or_state)


cvom_state_sha256 = actor_state_sha256


def _validate_rng(state):
    """Check local throw-away generators before touching model/global state."""
    random.Random().setstate(state["python"])
    np_state = state["numpy"]
    np.random.RandomState(0).set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.Generator(device="cpu").set_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Exact RNG resume requires the saved CUDA device count")
        for index, value in enumerate(state["cuda"]):
            torch.Generator(device=f"cuda:{index}").set_state(value)


def _validate_state(state, shapes, label):
    if set(state) != set(shapes):
        raise ValueError(f"V7 {label} tensor names differ from declared architecture")
    for name, tensor in state.items():
        if tuple(tensor.shape) != tuple(shapes[name]) or tensor.dtype != torch.float32:
            raise ValueError(f"V7 {label} shape/dtype mismatch: {name}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"V7 nonfinite {label} tensor: {name}")


def _config(config):
    if config.get("trainer_variant") != VARIANT:
        raise ValueError("Expected recurrent_memory_v7; older memory architectures cannot be loaded")
    if type(config.get("stage")) is not int or config["stage"] not in (1, 2):
        raise ValueError("V7 stage must be integer 1 or 2")
    mode = config.get("mode")
    if mode not in ("recurrent", "archive", "none"):
        raise ValueError("V7 mode must be recurrent, archive or none")
    if config["stage"] == 2 and mode != "recurrent":
        raise ValueError("V7 Stage 2 requires recurrent Stage-1 actor, not a control")
    if config.get("train", {}).get("mode", mode) != mode:
        raise ValueError("V7 conflicting mode declarations")
    return MemoryV7Config(**config["memory"])


def _expected_memory(cfg):
    # Meta construction reads only parameter shapes and leaves caller RNG alone.
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        actor = RecurrentMemoryV7(cfg)
        critic = CVOMV7(cfg)
        return ({k: tuple(v.shape) for k, v in actor.state_dict().items()},
                {k: tuple(v.shape) for k, v in critic.state_dict().items()})


def _frozen(info, actor, expert):
    if info["config"]["stage"] != 2:
        return
    meta = info["metadata"]
    parent = meta.get("stage1_parent", {})
    if not isinstance(parent.get("path"), str) or not parent["path"]:
        raise ValueError("V7 Stage 2 requires Stage-1 parent provenance")
    for key in ("checkpoint_sha256", "memory_sha256", "expert_sha256"):
        value = parent.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"V7 invalid parent fingerprint: {key}")
    if meta.get("frozen_actor_sha256") != actor_state_sha256(actor):
        raise ValueError("V7 frozen Stage-1 recurrent actor changed")
    if meta.get("frozen_expert_sha256") != _state_sha256(expert):
        raise ValueError("V7 frozen Stage-1 Action Expert changed")


def _installed(memory, head, cvom, config):
    cfg = _config(config)
    if asdict(memory.config) != asdict(cfg) or asdict(cvom.config) != asdict(cfg):
        raise ValueError("V7 installed memory/CVOM semantic config differs (including head count/time scale)")
    _validate_installed_head(head, config)


def v7_checkpoint_info(base_model, checkpoint, *, expected_stage=None):
    """Read-only validation, including shapes, finite payloads and frozen lineage.

Base identity is the established cache size/mtime+metadata identity, not a
cryptographic hash of all multi-GB base weights. Evaluation additionally hashes
the base payloads for immutable provenance. The parent directory is not needed
to deploy a self-contained Stage-2 bundle.
"""
    base, path = Path(base_model).resolve(), Path(checkpoint).resolve()
    info = json.loads((path / "checkpoint.json").read_text())
    if info.get("format_version") != 1:
        raise ValueError("Unsupported V7 checkpoint format")
    cfg = _config(info.get("config", {}))
    config, meta = info["config"], info.get("metadata", {})
    _validate_step(info.get("step"))
    if expected_stage is not None and config["stage"] != expected_stage:
        raise ValueError(f"Expected V7 Stage {expected_stage}, got {config['stage']}")
    if meta.get("base_model") != checkpoint_identity(base):
        raise ValueError("V7 base checkpoint identity changed")
    if not isinstance(meta.get("cache_fingerprint"), str) or not meta["cache_fingerprint"]:
        raise ValueError("V7 cache fingerprint missing")
    base_cfg = json.loads((base / "config.json").read_text())
    processor = json.loads((base / "processor_config.json").read_text())
    processor = processor.get("processor_kwargs", processor)
    if (base_cfg.get("hamlet_mode") != "finetune"
            or base_cfg.get("memory_type", "moment_token") != "moment_token"
            or base_cfg.get("mem_cond_type", "cross_attn") != "cross_attn"):
        raise ValueError("V7 requires original moment-token/cross-attention HAMLET")
    for name, expected in (("feature_dim", base_cfg["backbone_embedding_dim"]),
                           ("state_dim", processor["max_state_dim"]),
                           ("num_short_tokens", base_cfg["n_moment_tokens"]),
                           ("time_scale", base_cfg.get("memory_stride", 16))):
        if getattr(cfg, name) != expected:
            raise ValueError(f"V7 {name} differs from frozen checkpoint")
    for filename in PAYLOADS:
        if meta.get("payload_sha256", {}).get(filename) != file_sha256(path / filename):
            raise ValueError(f"V7 payload changed: {filename}")
    states = {filename: load_file(str(path / filename), device="cpu") for filename in PAYLOADS}
    actor_shapes, cvom_shapes = _expected_memory(cfg)
    _validate_state(states["model.safetensors"], actor_shapes, "actor")
    _validate_state(states["cvom.safetensors"], cvom_shapes, "CVOM")
    _validate_state(states["expert.safetensors"], _expected_expert_shapes(base, config), "expert")
    _frozen(info, states["model.safetensors"], states["expert.safetensors"])
    return info


def save_checkpoint_v7(output_dir, step, memory, head, cvom, optimizer, config, metadata,
                       best=False, *, training_state=None):
    """Publish a NEW directory; extra state includes coverage cursor/schedule.

Call only at optimizer boundaries. No checkpoint pruning or in-place overwrite.
"""
    step = _validate_step(step)
    _installed(memory, head, cvom, config)
    states = {"model.safetensors": memory.state_dict(), "expert.safetensors": expert_state_dict(head),
              "cvom.safetensors": cvom.state_dict()}
    states = {file: {k: v.detach().cpu().contiguous().clone() for k, v in values.items()}
              for file, values in states.items()}
    actor_shapes, cvom_shapes = _expected_memory(_config(config))
    _validate_state(states["model.safetensors"], actor_shapes, "actor")
    _validate_state(states["cvom.safetensors"], cvom_shapes, "CVOM")
    _validate_state(states["expert.safetensors"], {k: v.shape for k, v in expert_state_dict(head).items()}, "expert")
    info = {"format_version": 1, "step": step, "config": config, "metadata": dict(metadata)}
    _frozen(info, states["model.safetensors"], states["expert.safetensors"])
    json.dumps(info, allow_nan=False)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"checkpoint-{step:06d}"
    if destination.exists():
        raise FileExistsError(f"V7 checkpoint already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=root))
    try:
        for filename, state in states.items():
            save_file(state, str(temporary / filename))
        info["metadata"]["payload_sha256"] = {name: file_sha256(temporary / name) for name in PAYLOADS}
        # Familiar aliases make provenance easy to inspect alongside older runs.
        for name, filename in (("memory_sha256", "model.safetensors"), ("expert_sha256", "expert.safetensors"),
                               ("cvom_sha256", "cvom.safetensors")):
            info["metadata"][name] = info["metadata"]["payload_sha256"][filename]
        torch.save({"optimizer": optimizer.state_dict() if optimizer is not None else None,
                    "rng": _rng_state(), "extra": training_state or {}}, temporary / "training_state.pt")
        info["metadata"]["training_state_sha256"] = file_sha256(temporary / "training_state.pt")
        _atomic_json(temporary / "checkpoint.json", info)
        if destination.exists():
            raise FileExistsError(f"V7 checkpoint already exists: {destination}")
        temporary.rename(destination)
        if best:
            _atomic_json(root / "best_checkpoint.json", {"path": destination.name, "step": step})
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)  # only this save's newly allocated temporary directory


def load_checkpoint_v7(path, memory, head, cvom, optimizer=None):
    """Validate all parameters BEFORE mutation; optionally restore optimizer/RNG."""
    path = Path(path)
    preliminary = json.loads((path / "checkpoint.json").read_text())
    info = v7_checkpoint_info(preliminary["metadata"]["base_model"]["path"], path)
    _installed(memory, head, cvom, info["config"])
    states = {filename: load_file(str(path / filename), device="cpu") for filename in PAYLOADS}
    for filename, current in (("model.safetensors", memory.state_dict()), ("cvom.safetensors", cvom.state_dict()),
                               ("expert.safetensors", expert_state_dict(head))):
        _validate_state(states[filename], {k: v.shape for k, v in current.items()}, "installed " + filename)
    training = None
    if optimizer is not None:
        if info["metadata"].get("training_state_sha256") != file_sha256(path / "training_state.pt"):
            raise ValueError("V7 optimizer/RNG state changed")
        training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        if training.get("optimizer") is None:
            raise ValueError("V7 checkpoint has no optimizer state")
        saved_groups = training["optimizer"]["param_groups"]
        if len(saved_groups) != len(optimizer.param_groups) or any(
                len(a["params"]) != len(b["params"]) for a, b in zip(saved_groups, optimizer.param_groups)):
            raise ValueError("V7 optimizer parameter groups differ")
        _validate_rng(training["rng"])
    memory.load_state_dict(states["model.safetensors"], strict=True)
    load_expert_state_dict(head, states["expert.safetensors"])
    cvom.load_state_dict(states["cvom.safetensors"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(training["optimizer"])
        _restore_rng_state(training["rng"])
    return {**{key: info[key] for key in ("step", "config", "metadata")},
            "training_state": training.get("extra", {}) if training is not None else {}}
