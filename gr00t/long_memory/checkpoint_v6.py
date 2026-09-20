"""Immutable v6 visual-reader, read-time CVOM, AE LoRA and bridge bundles.

No original HAMLET parameter is saved here. Stage 2 may change only cvom.*;
reader, LoRA, and direct AE bridge must match the fixed Stage-1 teacher.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .checkpoint_v4 import (file_sha256, _state_sha256, _expected_expert_shapes,
                            _validate_installed_head, _validate_state)
from .expert_v4 import expert_state_dict, load_expert_state_dict
from .hamlet import checkpoint_identity
from .monitoring import _atomic_json, _rng_state, _restore_rng_state, _validate_step

VARIANT = "retrieval_cvom_v6"
PAYLOADS = ("memory.safetensors", "expert.safetensors", "bridge.safetensors")


def reader_state_sha256(memory_or_state):
    state = memory_or_state.state_dict() if hasattr(memory_or_state, "state_dict") else memory_or_state
    return _state_sha256({k: v for k, v in state.items() if not k.startswith("cvom.")})


def frozen_identity(memory, head):
    from .expert_v6 import bridge_state_dict
    return {"reader": reader_state_sha256(memory), "expert": _state_sha256(expert_state_dict(head)),
            "bridge": _state_sha256(bridge_state_dict(head))}


def _frozen_check(info, memory, expert, bridge):
    if info["config"]["stage"] != 2:
        return
    actual = {"reader": reader_state_sha256(memory), "expert": _state_sha256(expert),
              "bridge": _state_sha256(bridge)}
    parent = info["metadata"].get("stage1_parent", {})
    if info["metadata"].get("frozen_identity") != actual or parent.get("frozen_identity") != actual:
        raise ValueError("V6 Stage-2 reader/expert/bridge differs from its frozen Stage-1 teacher")
    if not isinstance(parent.get("checkpoint_sha256"), str) or len(parent["checkpoint_sha256"]) != 64:
        raise ValueError("V6 Stage 2 requires immutable Stage-1 parent provenance")


def _architecture_shapes(base, config):
    from .core_v6 import VisualMemoryV6, VisualMemoryV6Config
    from .expert_v6 import memory_bridge_shapes
    cfg = VisualMemoryV6Config(**config["memory"])
    base_cfg = json.loads((base / "config.json").read_text())
    proc = json.loads((base / "processor_config.json").read_text())
    proc = proc.get("processor_kwargs", proc)
    if (base_cfg.get("hamlet_mode") != "finetune" or base_cfg.get("memory_type", "moment_token") != "moment_token"
            or base_cfg.get("mem_cond_type", "cross_attn") != "cross_attn"):
        raise ValueError("V6 requires the original trained moment-token cross-attention HAMLET")
    if cfg.feature_dim != int(base_cfg["backbone_embedding_dim"]) or cfg.state_dim != int(proc["max_state_dim"]):
        raise ValueError("V6 memory dimensions differ from source HAMLET")
    if cfg.time_scale != float(base_cfg.get("memory_stride", 16)):
        raise ValueError("V6 raw-frame time scale differs from source observation cadence")
    bridge = config["bridge"]
    if bridge["hidden_dim"] != cfg.hidden_dim:
        raise ValueError("V6 bridge/memory dimensions differ")
    shapes = _expected_expert_shapes(base, config)
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    for block in bridge["block_indices"]:
        name = f"action_head.model.transformer_blocks.{block}.attn1.to_q.weight"
        if name not in index:
            raise ValueError(f"V6 bridge block absent from source Expert: {block}")
        shard = (base / index[name]).resolve()
        if not shard.is_relative_to(base):
            raise ValueError("Source shard escapes checkpoint")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            width = handle.get_slice(name).get_shape()[1]
        if width != bridge["expert_dim"]:
            raise ValueError("V6 bridge expert width differs from original Expert")
    with torch.device("meta"):
        memory = VisualMemoryV6(cfg)
    return ({k: tuple(v.shape) for k, v in memory.state_dict().items()}, shapes,
            memory_bridge_shapes(bridge))


def v6_checkpoint_info(base_model, checkpoint, *, expected_stage=None):
    base, root = Path(base_model).resolve(), Path(checkpoint).resolve()
    info = json.loads((root / "checkpoint.json").read_text())
    config, metadata = info.get("config", {}), info.get("metadata", {})
    if info.get("format_version") != 1 or config.get("trainer_variant") != VARIANT:
        raise ValueError("Expected a retrieval_cvom_v6 checkpoint; v3/v4/v5 are separate architectures")
    if type(config.get("stage")) is not int or config["stage"] not in (1, 2):
        raise ValueError("V6 stage must be 1 or 2")
    if expected_stage is not None and config["stage"] != expected_stage:
        raise ValueError(f"Expected V6 Stage {expected_stage}, got {config['stage']}")
    _validate_step(info.get("step"))
    if metadata.get("base_model") != checkpoint_identity(base):
        raise ValueError("V6 source HAMLET changed or differs")
    if not isinstance(metadata.get("cache_fingerprint"), str) or not metadata["cache_fingerprint"]:
        raise ValueError("V6 requires cache provenance")
    if config.get("reader_mode", "memory") not in ("memory", "none"):
        raise ValueError("Invalid V6 reader mode")
    if config["stage"] == 2 and config.get("reader_mode") != "memory":
        raise ValueError("V6 CVOM cannot be trained from an AE-only control")
    expected = _architecture_shapes(base, config)
    states = []
    for filename, shapes in zip(PAYLOADS, expected):
        state = load_file(str(root / filename), device="cpu")
        _validate_state(state, shapes, label="V6 " + filename)
        if metadata.get("payload_sha256", {}).get(filename) != file_sha256(root / filename):
            raise ValueError(f"V6 payload changed after publication: {filename}")
        states.append(state)
    _frozen_check(info, *states)
    return info


def _installed_config_check(memory, head, config):
    if memory.config.to_dict() != config["memory"]:
        raise ValueError("Installed V6 memory configuration differs")
    if getattr(head, "_long_memory_v6_bridge_config", None) != config["bridge"]:
        raise ValueError("Installed V6 bridge configuration differs")


def save_checkpoint_v6(output_dir, step, memory, head, optimizer, config, metadata, *, best=False):
    from .expert_v6 import bridge_state_dict
    step = _validate_step(step)
    if config.get("trainer_variant") != VARIANT or config.get("stage") not in (1, 2):
        raise ValueError("V6 checkpoint variant/stage required")
    _validate_installed_head(head, config)
    _installed_config_check(memory, head, config)
    expected = _architecture_shapes(Path(metadata["base_model"]["path"]).resolve(), config)
    states = [{k: v.detach().cpu().contiguous().clone() for k, v in state.items()} for state in
              (memory.state_dict(), expert_state_dict(head), bridge_state_dict(head))]
    for name, state, shapes in zip(PAYLOADS, states, expected):
        if not state:
            raise ValueError(f"Empty V6 state: {name}")
        _validate_state(state, shapes, label=name)
    info = {"format_version": 1, "step": step, "config": copy.deepcopy(config), "metadata": copy.deepcopy(metadata)}
    _frozen_check(info, *states)
    json.dumps(info, allow_nan=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"checkpoint-{step:06d}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=".v6-publish-", dir=output))
    try:
        for name, state in zip(PAYLOADS, states):
            save_file(state, str(temporary / name))
        info["metadata"]["payload_sha256"] = {name: file_sha256(temporary / name) for name in PAYLOADS}
        torch.save({"optimizer": optimizer.state_dict() if optimizer is not None else None,
                    "rng": _rng_state()}, temporary / "training_state.pt")
        info["metadata"]["training_state_sha256"] = file_sha256(temporary / "training_state.pt")
        _atomic_json(temporary / "checkpoint.json", info)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
        temporary.rename(destination)
        if best:
            _atomic_json(output / "best_checkpoint.json", {"path": destination.name, "step": step})
        return destination
    finally:
        # Only this call's exact newly created staging directory is removed.
        if temporary.exists():
            shutil.rmtree(temporary)


def load_checkpoint_v6(path, memory, head, optimizer=None):
    from .expert_v6 import bridge_state_dict, load_bridge_state_dict
    root = Path(path)
    preliminary = json.loads((root / "checkpoint.json").read_text())
    info = v6_checkpoint_info(preliminary["metadata"]["base_model"]["path"], root)
    _validate_installed_head(head, info["config"])
    _installed_config_check(memory, head, info["config"])
    states = [load_file(str(root / name), device="cpu") for name in PAYLOADS]
    for state, installed in zip(states, (memory.state_dict(), expert_state_dict(head), bridge_state_dict(head))):
        _validate_state(state, {k: v.shape for k, v in installed.items()}, label="installed v6")
    saved = None
    if optimizer is not None:
        if file_sha256(root / "training_state.pt") != info["metadata"].get("training_state_sha256"):
            raise ValueError("V6 optimizer/RNG state changed after publication")
        saved = torch.load(root / "training_state.pt", map_location="cpu", weights_only=True)
        if saved.get("optimizer") is None:
            raise ValueError("V6 checkpoint has no optimizer for exact resume")
    memory.load_state_dict(states[0], strict=True)
    load_expert_state_dict(head, states[1])
    load_bridge_state_dict(head, states[2])
    if optimizer is not None:
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng_state(saved["rng"])
    return info
