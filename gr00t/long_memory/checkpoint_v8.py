"""Immutable event-memory/AE bundles, independent of all earlier experiments.

V8 Stage 1 learns the event encoder, retrieval/fusion and AE attention LoRA.
There is no learned admission policy in this bundle and no placeholder CVOM.
Episode contents are runtime state, never serialized as model parameters.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file

from .checkpoint_v4 import (file_sha256, _state_sha256, _expected_expert_shapes,
                            _validate_installed_head)
from .checkpoint_v7 import _validate_rng
from .event_v8 import MemoryV8Config, EventMemoryV8
from .expert_v4 import expert_state_dict, load_expert_state_dict
from .hamlet import checkpoint_identity
from .monitoring import _atomic_json, _restore_rng_state, _rng_state, _validate_step

VARIANT = "event_memory_v8"
PAYLOADS = ("model.safetensors", "expert.safetensors")


def actor_state_sha256(module_or_state):
    state = module_or_state.state_dict() if hasattr(module_or_state, "state_dict") else module_or_state
    return _state_sha256(state)


def _config(config):
    if config.get("trainer_variant") != VARIANT:
        raise ValueError("Expected event_memory_v8; V7 and other memory weights are not interchangeable")
    if type(config.get("stage")) is not int or config["stage"] != 1:
        raise ValueError("V8 currently supports Stage 1 event encoding/retrieval, not CVOM training")
    mode = config.get("mode")
    if mode not in ("event", "none"):
        raise ValueError("V8 mode must be event or none")
    if config.get("train", {}).get("mode", mode) != mode:
        raise ValueError("V8 conflicting mode declarations")
    return MemoryV8Config(**config["memory"])


def _expected_memory(cfg):
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        actor = EventMemoryV8(cfg)
        return {name: tuple(value.shape) for name, value in actor.state_dict().items()}


def _validate_state(state, shapes, label):
    if set(state) != set(shapes):
        raise ValueError(f"V8 {label} tensor names differ from declared architecture")
    for name, tensor in state.items():
        if tuple(tensor.shape) != tuple(shapes[name]) or tensor.dtype != torch.float32:
            raise ValueError(f"V8 {label} shape/dtype mismatch: {name}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"V8 nonfinite {label} tensor: {name}")


def _installed(memory, head, config):
    if asdict(memory.config) != asdict(_config(config)):
        raise ValueError("V8 installed memory semantic config differs (source/capacity/residual bound included)")
    _validate_installed_head(head, config)


def v8_checkpoint_info(base_model, checkpoint, *, expected_stage=None):
    """Read-only configuration, source-backbone, shape, dtype and payload check.

Base identity uses the existing cache's metadata/size/mtime contract; eval also
hashes the multi-GB original weight files. No full base model is instantiated.
"""
    base, path = Path(base_model).resolve(), Path(checkpoint).resolve()
    info = json.loads((path / "checkpoint.json").read_text())
    if info.get("format_version") != 1:
        raise ValueError("Unsupported V8 checkpoint format")
    config, meta = info.get("config", {}), info.get("metadata", {})
    cfg = _config(config)
    _validate_step(info.get("step"))
    if expected_stage is not None and config["stage"] != expected_stage:
        raise ValueError(f"Expected V8 Stage {expected_stage}, got {config['stage']}")
    if meta.get("base_model") != checkpoint_identity(base):
        raise ValueError("V8 base checkpoint identity changed")
    if not isinstance(meta.get("cache_fingerprint"), str) or not meta["cache_fingerprint"]:
        raise ValueError("V8 cache fingerprint missing")
    base_cfg = json.loads((base / "config.json").read_text())
    processor = json.loads((base / "processor_config.json").read_text())
    processor = processor.get("processor_kwargs", processor)
    if (base_cfg.get("hamlet_mode") != "finetune"
            or base_cfg.get("memory_type", "moment_token") != "moment_token"
            or base_cfg.get("mem_cond_type", "cross_attn") != "cross_attn"):
        raise ValueError("V8 requires original moment-token/cross-attention HAMLET")
    for name, expected in (("feature_dim", base_cfg["backbone_embedding_dim"]),
                           ("state_dim", processor["max_state_dim"]),
                           ("num_short_tokens", base_cfg["n_moment_tokens"]),
                           ("time_scale", base_cfg.get("memory_stride", 16))):
        if getattr(cfg, name) != expected:
            raise ValueError(f"V8 {name} differs from frozen checkpoint")
    states = {}
    for filename in PAYLOADS:
        if meta.get("payload_sha256", {}).get(filename) != file_sha256(path / filename):
            raise ValueError(f"V8 payload changed: {filename}")
        states[filename] = load_file(str(path / filename), device="cpu")
    _validate_state(states["model.safetensors"], _expected_memory(cfg), "actor")
    _validate_state(states["expert.safetensors"], _expected_expert_shapes(base, config), "expert")
    return info


def save_checkpoint_v8(output_dir, step, memory, head, optimizer, config, metadata,
                       best=False, *, training_state=None):
    """Publish a NEW optimizer-boundary checkpoint, without pruning old files."""
    step = _validate_step(step)
    _installed(memory, head, config)
    states = {"model.safetensors": memory.state_dict(), "expert.safetensors": expert_state_dict(head)}
    states = {file: {k: v.detach().cpu().contiguous().clone() for k, v in values.items()}
              for file, values in states.items()}
    _validate_state(states["model.safetensors"], _expected_memory(_config(config)), "actor")
    _validate_state(states["expert.safetensors"], {k: v.shape for k, v in expert_state_dict(head).items()}, "expert")
    info = {"format_version": 1, "step": step, "config": config, "metadata": dict(metadata)}
    json.dumps(info, allow_nan=False)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"checkpoint-{step:06d}"
    if destination.exists():
        raise FileExistsError(f"V8 checkpoint already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=root))
    try:
        for filename, state in states.items():
            save_file(state, str(temporary / filename))
        info["metadata"]["payload_sha256"] = {name: file_sha256(temporary / name) for name in PAYLOADS}
        for name, filename in (("memory_sha256", "model.safetensors"), ("expert_sha256", "expert.safetensors")):
            info["metadata"][name] = info["metadata"]["payload_sha256"][filename]
        torch.save({"optimizer": optimizer.state_dict() if optimizer is not None else None,
                    "rng": _rng_state(), "extra": training_state or {}}, temporary / "training_state.pt")
        info["metadata"]["training_state_sha256"] = file_sha256(temporary / "training_state.pt")
        _atomic_json(temporary / "checkpoint.json", info)
        if destination.exists():
            raise FileExistsError(f"V8 checkpoint already exists: {destination}")
        temporary.rename(destination)
        if best:
            _atomic_json(root / "best_checkpoint.json", {"path": destination.name, "step": step})
        return destination
    finally:
        if temporary.exists():
            # Only this failed save's newly allocated, fully resolved temp dir.
            shutil.rmtree(temporary)


def _validate_optimizer(optimizer, saved):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("V8 exact optimizer resume currently supports AdamW only")
    if not isinstance(saved, dict) or "param_groups" not in saved or "state" not in saved:
        raise ValueError("V8 checkpoint has no optimizer state")
    groups = saved["param_groups"]
    if len(groups) != len(optimizer.param_groups) or any(
            len(a["params"]) != len(b["params"]) for a, b in zip(groups, optimizer.param_groups)):
        raise ValueError("V8 optimizer parameter groups differ")
    for old, new in zip(groups, optimizer.param_groups):
        if old.get("kind") != new.get("kind"):
            raise ValueError("V8 optimizer group ownership differs")
        for identifier, parameter in zip(old["params"], new["params"]):
            values = saved["state"].get(identifier, {})
            required = {"step", "exp_avg", "exp_avg_sq"}
            if old.get("amsgrad", False):
                required.add("max_exp_avg_sq")
            if values and not required.issubset(values):
                raise ValueError("V8 AdamW state is missing required fields")
            if values:
                step = values["step"]
                if (not torch.is_tensor(step) or step.ndim != 0
                        or not bool(torch.isfinite(step)) or float(step) < 0):
                    raise ValueError("V8 AdamW step must be a finite nonnegative scalar tensor")
                if any(not torch.is_tensor(values[key]) for key in required - {"step"}):
                    raise ValueError("V8 AdamW moments must be tensors")
            for name, value in values.items():
                if torch.is_tensor(value):
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError("V8 nonfinite optimizer state")
                    if name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") and value.shape != parameter.shape:
                        raise ValueError("V8 optimizer tensor shape differs")


def load_checkpoint_v8(path, memory, head, optimizer=None):
    """Validate before mutation, then optionally restore optimizer/RNG/cursor."""
    path = Path(path)
    preliminary = json.loads((path / "checkpoint.json").read_text())
    info = v8_checkpoint_info(preliminary["metadata"]["base_model"]["path"], path)
    _installed(memory, head, info["config"])
    states = {filename: load_file(str(path / filename), device="cpu") for filename in PAYLOADS}
    for filename, current in (("model.safetensors", memory.state_dict()),
                               ("expert.safetensors", expert_state_dict(head))):
        _validate_state(states[filename], {k: v.shape for k, v in current.items()}, "installed " + filename)
    training = None
    if optimizer is not None:
        if info["metadata"].get("training_state_sha256") != file_sha256(path / "training_state.pt"):
            raise ValueError("V8 optimizer/RNG state changed")
        training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        _validate_optimizer(optimizer, training.get("optimizer"))
        _validate_rng(training["rng"])
    memory.load_state_dict(states["model.safetensors"], strict=True)
    load_expert_state_dict(head, states["expert.safetensors"])
    if optimizer is not None:
        optimizer.load_state_dict(training["optimizer"])
        _restore_rng_state(training["rng"])
    return {**{key: info[key] for key in ("step", "config", "metadata")},
            "training_state": training.get("extra", {}) if training is not None else {}}
