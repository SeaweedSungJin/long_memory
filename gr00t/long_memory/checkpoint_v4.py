"""Immutable, self-contained memory + Action Expert adapter checkpoint bundles.

The several-GB original HAMLET remains external and read-only. Both trainable
components are published atomically with optimizer/RNG state; no memory-only
checkpoint can accidentally masquerade as an adapted-Expert model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .core_v3 import ActionValueMemory, MemoryV3Config
from .expert_v4 import (ExpertLoRALinear, LoRAConfig, TARGET_PATTERN, expert_state_dict,
                       load_expert_state_dict)
from .hamlet import checkpoint_identity
from .monitoring import _atomic_json, _restore_rng_state, _rng_state, _validate_step


VARIANT = "action_expert_v4"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        # Parameters here are FP32; uint8 view also supports future BF16 states.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def reader_state_sha256(memory_or_state):
    state = memory_or_state.state_dict() if hasattr(memory_or_state, "state_dict") else memory_or_state
    return _state_sha256({name: value for name, value in state.items() if not name.startswith("writer.")})


def _expected_expert_shapes(base, config):
    """Safetensors metadata only: never instantiate or load the full base."""
    lora = LoRAConfig(**config["expert"])
    targets = config.get("expert_targets")
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    found = sorted(key[len("action_head."):-len(".weight")] for key in index
                   if key.startswith("action_head.") and key.endswith(".weight")
                   and TARGET_PATTERN.fullmatch(key[len("action_head."):-len(".weight")]))
    if not found or not isinstance(targets, list) or targets != found:
        raise ValueError("V4 expert_targets differ from exact base attention projections")
    result = {}
    # Header slices retrieve only shapes; no multi-GB tensor is materialized.
    for shard in sorted({index[f"action_head.{name}.weight"] for name in found}):
        shard_path = (base / shard).resolve()
        if not shard_path.is_relative_to(base):
            raise ValueError("Base weight shard path escapes base checkpoint")
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for name in found:
                key = f"action_head.{name}.weight"
                if index[key] != shard:
                    continue
                shape = handle.get_slice(key).get_shape()
                if len(shape) != 2 or min(shape) <= 0:
                    raise ValueError(f"Invalid base projection shape: {key}")
                result[name + ".lora_A"] = (lora.rank, shape[1])
                result[name + ".lora_B"] = (shape[0], lora.rank)
    return result


def _validate_state(state, shapes, *, label):
    if set(state) != set(shapes):
        raise ValueError(f"V4 {label} tensor names do not match architecture/config")
    for name, tensor in state.items():
        if tuple(tensor.shape) != tuple(shapes[name]):
            raise ValueError(f"V4 {label} tensor shape mismatch: {name}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"V4 {label} tensors must be FP32: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"V4 {label} contains nonfinite weights: {name}")


def _validate_frozen_stage2(info, memory, expert):
    if info["config"]["stage"] != 2:
        return
    metadata = info["metadata"]
    parent = metadata.get("stage1_parent", {})
    if not isinstance(parent, dict) or not isinstance(parent.get("path"), str) or not parent["path"]:
        raise ValueError("V4 Stage 2 must record stage1_parent provenance")
    for key in ("checkpoint_sha256", "memory_sha256", "expert_sha256"):
        value = parent.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid Stage 1 parent {key}")
    if metadata.get("frozen_reader_sha256") != reader_state_sha256(memory):
        raise ValueError("V4 Stage 2 frozen reader changed from Stage 1 initialization")
    if metadata.get("frozen_expert_sha256") != _state_sha256(expert):
        raise ValueError("V4 Stage 2 frozen Action Expert changed from Stage 1 initialization")


def _validate_installed_head(head, config):
    lora = LoRAConfig(**config["expert"])
    installed = sorted((name, module) for name, module in head.named_modules() if isinstance(module, ExpertLoRALinear))
    if [name for name, _ in installed] != config["expert_targets"]:
        raise ValueError("Installed Expert targets differ from checkpoint config")
    if any(module.scale != float(lora.alpha) / lora.rank or module.lora_A.shape[0] != lora.rank
           for _, module in installed):
        raise ValueError("Installed Expert rank/alpha differ from checkpoint config")


def v4_checkpoint_info(base_model, checkpoint, *, expected_stage=None):
    """Strict, read-only bundle validation without full HAMLET allocation.

    Stage 2 is independently deployable: parent fingerprints are recorded, but
    inference never needs the parent's directory to remain mounted.
    """
    base, checkpoint = Path(base_model).resolve(), Path(checkpoint).resolve()
    info = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    config, metadata = info.get("config", {}), info.get("metadata", {})
    if info.get("format_version") != 1 or config.get("trainer_variant") != VARIANT:
        raise ValueError("Expected an action_expert_v4 bundle; memory-only/legacy checkpoints are incompatible")
    stage = config.get("stage")
    if type(stage) is not int or stage not in (1, 2):
        raise ValueError("V4 checkpoint stage must be integer 1 or 2")
    if expected_stage is not None and stage != expected_stage:
        raise ValueError(f"Expected Stage {expected_stage}, got Stage {stage}: {checkpoint}")
    _validate_step(info.get("step"))
    if metadata.get("base_model") != checkpoint_identity(base):
        raise ValueError("V4 bundle was trained on a different/changed base checkpoint")
    if not isinstance(metadata.get("cache_fingerprint"), str) or not metadata["cache_fingerprint"]:
        raise ValueError("V4 requires a nonempty training cache_fingerprint")
    reader_mode = config.get("train", {}).get("reader_mode", config.get("reader_mode"))
    if "reader_mode" in config and config["reader_mode"] != reader_mode:
        raise ValueError("V4 checkpoint has conflicting reader_mode declarations")
    if reader_mode not in ("memory", "none") or (stage == 2 and reader_mode != "memory"):
        raise ValueError("V4 reader_mode must be memory/none, and Stage 2 requires memory")
    cfg = MemoryV3Config(**config["memory"])
    base_config = json.loads((base / "config.json").read_text())
    processor = json.loads((base / "processor_config.json").read_text())
    processor = processor.get("processor_kwargs", processor)
    if (base_config.get("hamlet_mode") != "finetune"
            or base_config.get("memory_type", "moment_token") != "moment_token"
            or base_config.get("mem_cond_type", "cross_attn") != "cross_attn"
            or int(base_config.get("n_moment_tokens", 0)) <= 0):
        raise ValueError("V4 requires trained HAMLET moment-token cross-attention base")
    for name, expected in (("feature_dim", base_config["backbone_embedding_dim"]),
                           ("state_dim", processor["max_state_dim"]),
                           ("action_dim", processor["max_action_dim"])):
        if getattr(cfg, name) != int(expected):
            raise ValueError(f"V4 {name} differs from base processor/model")
    if float(cfg.time_scale) != float(base_config.get("memory_stride", 16)):
        raise ValueError("V4 memory time_scale differs from base memory_stride")
    with torch.device("meta"):
        expected_memory = {name: tuple(value.shape) for name, value in ActionValueMemory(cfg).state_dict().items()}
    memory = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    expert = load_file(str(checkpoint / "expert.safetensors"), device="cpu")
    _validate_state(memory, expected_memory, label="memory")
    _validate_state(expert, _expected_expert_shapes(base, config), label="expert")
    for field, filename in (("memory_sha256", "model.safetensors"), ("expert_sha256", "expert.safetensors")):
        if metadata.get(field) != file_sha256(checkpoint / filename):
            raise ValueError(f"V4 {field} mismatch: bundle weights changed after save")
    _validate_frozen_stage2(info, memory, expert)
    return info


def save_checkpoint_v4(output_dir, step, memory, head, optimizer, config, metadata,
                       best=False, keep_last=None):
    """Atomically save both adapters, optimizer and RNG; never overwrite/prune."""
    step = _validate_step(step)
    if config.get("trainer_variant") != VARIANT or type(config.get("stage")) is not int or config["stage"] not in (1, 2):
        raise ValueError("save_checkpoint_v4 requires an action_expert_v4 Stage 1/2 config")
    lora = LoRAConfig(**config["expert"])
    _validate_installed_head(head, config)
    memory_state = {name: value.detach().cpu().contiguous().clone() for name, value in memory.state_dict().items()}
    expert_state = {name: value.cpu().contiguous().clone() for name, value in expert_state_dict(head).items()}
    if not expert_state:
        raise ValueError("Cannot publish a V4 bundle without Expert adapters")
    expected_names = {name + suffix for name in config["expert_targets"] for suffix in (".lora_A", ".lora_B")}
    if set(expert_state) != expected_names:
        raise ValueError("Expert targets differ from config during save")
    for state in (memory_state, expert_state):
        for name, value in state.items():
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"Refusing nonfinite/non-FP32 V4 checkpoint weights: {name}")
    if any(value.shape[0 if name.endswith("lora_A") else 1] != lora.rank for name, value in expert_state.items()):
        raise ValueError("Expert rank differs from config during save")
    manifest = {"format_version": 1, "step": step, "config": config, "metadata": dict(metadata)}
    _validate_frozen_stage2(manifest, memory_state, expert_state)
    json.dumps(manifest, allow_nan=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"checkpoint-{step:06d}"
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists; refusing to overwrite: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=output))
    try:
        save_file(memory_state, str(temporary / "model.safetensors"))
        save_file(expert_state, str(temporary / "expert.safetensors"))
        manifest["metadata"].update(memory_sha256=file_sha256(temporary / "model.safetensors"),
                                    expert_sha256=file_sha256(temporary / "expert.safetensors"))
        torch.save({"optimizer": optimizer.state_dict() if optimizer is not None else None, "rng": _rng_state()},
                   temporary / "training_state.pt")
        _atomic_json(temporary / "checkpoint.json", manifest)
        if destination.exists():
            raise FileExistsError(f"Checkpoint already exists; refusing to overwrite: {destination}")
        temporary.rename(destination)
        if best:
            _atomic_json(output / "best_checkpoint.json", {"path": destination.name, "step": step})
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_checkpoint_v4(path, memory, head, optimizer=None):
    """Strict weights-only initialization, or full optimizer/RNG resume."""
    path = Path(path)
    preliminary = json.loads((path / "checkpoint.json").read_text())
    info = v4_checkpoint_info(preliminary["metadata"]["base_model"]["path"], path)
    _validate_installed_head(head, info["config"])
    memory_state = load_file(str(path / "model.safetensors"), device="cpu")
    expert_state = load_file(str(path / "expert.safetensors"), device="cpu")
    _validate_state(memory_state, {name: value.shape for name, value in memory.state_dict().items()}, label="installed memory")
    _validate_state(expert_state, {name: value.shape for name, value in expert_state_dict(head).items()}, label="installed expert")
    state = None
    if optimizer is not None:
        state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        if state.get("optimizer") is None:
            raise ValueError("V4 checkpoint has no optimizer state and cannot exactly resume")
    memory.load_state_dict(memory_state, strict=True)
    load_expert_state_dict(head, expert_state)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        _restore_rng_state(state["rng"])
    return {key: info[key] for key in ("config", "metadata", "step")}
