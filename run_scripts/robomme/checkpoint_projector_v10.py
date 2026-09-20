"""Portable, immutable archive + LoRA + full-rank projector-delta bundles.

V10 is deliberately NOT a V7 checkpoint.  The original HAMLET remains external
and read-only; the continuation's parent directory is never needed to load it.
Optimizer/RNG/cursor state is restored only on an explicit optimizer resume.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from gr00t.long_memory.checkpoint_v4 import (
    _expected_expert_shapes, _validate_installed_head, file_sha256,
)
from gr00t.long_memory.checkpoint_v7 import (
    VARIANT as V7_VARIANT, _config as _v7_config, _expected_memory, _validate_rng,
)
from gr00t.long_memory.expert_v4 import expert_state_dict, load_expert_state_dict
from gr00t.long_memory.hamlet import checkpoint_identity
from gr00t.long_memory.monitoring import _atomic_json, _restore_rng_state, _rng_state, _validate_step
from run_scripts.robomme.projector_adapter_v10 import (
    assert_expert_scope, load_projector_state_dict, projector_spec, projector_state_dict,
)


VARIANT = "archive_projector_v10"
PAYLOADS = ("model.safetensors", "expert.safetensors", "cvom.safetensors", "projector.safetensors")
_PROJECTOR_KEYS = {"target", "in_features", "out_features", "bias", "kind", "enabled"}


def _config(config):
    if not isinstance(config, dict) or config.get("trainer_variant") != VARIANT:
        raise ValueError("Expected archive_projector_v10; V7/other bundles are incompatible")
    if type(config.get("stage")) is not int or config["stage"] != 1:
        raise ValueError("V10 supports Stage 1 only")
    if config.get("mode") != "archive":
        raise ValueError("V10 supports archive mode only")
    spec = config.get("projector")
    if not isinstance(spec, dict) or set(spec) != _PROJECTOR_KEYS:
        raise ValueError("V10 exact projector specification is required")
    if (spec["target"] != "model.proj_out_2" or spec["kind"] != "full_rank_residual_fp32"
            or spec["bias"] is not True or type(spec["enabled"]) is not bool
            or any(type(spec[k]) is not int or spec[k] <= 0 for k in ("in_features", "out_features"))):
        raise ValueError("V10 unsupported projector semantics")
    train = config.get("train", {})
    if "train_projector" in train and (type(train["train_projector"]) is not bool or train["train_projector"] != spec["enabled"]):
        raise ValueError("V10 conflicting projector training-arm declarations")
    # This is only a local read-only validation view, never returned or written.
    return _v7_config({**config, "trainer_variant": V7_VARIANT})


def _projector_shapes(spec):
    return {"delta_weight": (spec["out_features"], spec["in_features"]),
            "delta_bias": (spec["out_features"],)}


def base_projector_spec(base_model, enabled=True):
    """Read ONLY the original safetensor headers, before allocating/output creation."""
    if type(enabled) is not bool:
        raise ValueError("V10 projector enabled must be boolean")
    base = Path(base_model).resolve()
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    shapes = {}
    for suffix in ("weight", "bias"):
        key = "action_head.model.proj_out_2." + suffix
        if key not in index:
            raise ValueError(f"V10 original projector missing: {key}")
        shard = (base / index[key]).resolve()
        if not shard.is_relative_to(base):
            raise ValueError("V10 projector shard escapes original base")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            shapes[suffix] = tuple(handle.get_slice(key).get_shape())
    weight, bias = shapes["weight"], shapes["bias"]
    if len(weight) != 2 or min(weight) <= 0 or bias != (weight[0],):
        raise ValueError("V10 original projector weight/bias shapes are invalid")
    return {"target": "model.proj_out_2", "in_features": weight[1], "out_features": weight[0],
            "bias": True, "kind": "full_rank_residual_fp32", "enabled": enabled}


def _validate_state(state, shapes, label):
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise ValueError(f"V10 {label} tensor names differ from declared architecture")
    for name, value in state.items():
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(shapes[name]) or value.dtype != torch.float32:
            raise ValueError(f"V10 {label} shape/dtype mismatch: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"V10 nonfinite {label}: {name}")


def _validate_delta(state, config, step):
    _validate_state(state, _projector_shapes(config["projector"]), "projector")
    if (step == 0 or not config["projector"]["enabled"]) and any(bool(v.count_nonzero()) for v in state.values()):
        raise ValueError("V10 step-zero and disabled-control projector deltas must be zero")


def _base_shapes(base, config, metadata):
    base = Path(base).resolve()
    cfg = _config(config)
    if metadata.get("base_model") != checkpoint_identity(base):
        raise ValueError("V10 original base checkpoint identity changed")
    if not isinstance(metadata.get("cache_fingerprint"), str) or not metadata["cache_fingerprint"]:
        raise ValueError("V10 cache fingerprint missing")
    base_cfg = json.loads((base / "config.json").read_text())
    processor = json.loads((base / "processor_config.json").read_text())
    processor = processor.get("processor_kwargs", processor)
    if (base_cfg.get("hamlet_mode") != "finetune"
            or base_cfg.get("memory_type", "moment_token") != "moment_token"
            or base_cfg.get("mem_cond_type", "cross_attn") != "cross_attn"):
        raise ValueError("V10 requires original moment-token/cross-attention HAMLET")
    for key, expected in (("feature_dim", base_cfg["backbone_embedding_dim"]),
                          ("state_dim", processor["max_state_dim"]),
                          ("num_short_tokens", base_cfg["n_moment_tokens"]),
                          ("time_scale", base_cfg.get("memory_stride", 16))):
        if getattr(cfg, key) != expected:
            raise ValueError(f"V10 {key} differs from original base")
    expert_shapes = _expected_expert_shapes(base, config)
    if base_projector_spec(base, config["projector"]["enabled"]) != config["projector"]:
        raise ValueError("V10 projector dimensions differ from original base header")
    actor, critic = _expected_memory(cfg)
    return {"model.safetensors": actor, "expert.safetensors": expert_shapes,
            "cvom.safetensors": critic, "projector.safetensors": _projector_shapes(config["projector"])}


def _installed(memory, head, cvom, config):
    cfg = _config(config)
    if asdict(memory.config) != asdict(cfg) or asdict(cvom.config) != asdict(cfg):
        raise ValueError("V10 installed memory/CVOM semantic config differs")
    _validate_installed_head(head, config)
    for label, state in (("memory", memory.state_dict()), ("CVOM", cvom.state_dict()),
                         ("expert", expert_state_dict(head))):
        if any(value.dtype != torch.float32 for value in state.values()):
            raise ValueError(f"V10 installed {label} master parameters must remain FP32")
    if projector_spec(head) != config["projector"]:
        raise ValueError("V10 installed projector semantic config differs")
    assert_expert_scope(head)


def _finite_tree(value, label):
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"V10 nonfinite {label}")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, (int, str)):
                raise ValueError(f"V10 unsupported {label} key")
            _finite_tree(item, label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tree(item, label)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"V10 nonfinite {label}")
    elif value is not None and not isinstance(value, (str, int, bool)):
        raise ValueError(f"V10 unsupported {label} value: {type(value).__name__}")


def _optimizer_names(optimizer, memory, head):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("V10 exact optimizer resume supports AdamW only")
    allowed = {id(p): "memory." + n for n, p in memory.named_parameters()}
    expert_names = set(expert_state_dict(head))
    allowed.update({id(p): "expert." + n for n, p in head.named_parameters() if n in expert_names})
    module = head.get_submodule("model.proj_out_2")
    allowed.update({id(getattr(module, n)): "projector." + n for n in ("delta_weight", "delta_bias")})
    result, seen = [], set()
    for group in optimizer.param_groups:
        names = []
        for p in group["params"]:
            if id(p) not in allowed or id(p) in seen:
                raise ValueError("V10 optimizer contains duplicate or non-actor/adapter parameters")
            seen.add(id(p))
            names.append(allowed[id(p)])
        result.append(names)
    return result


def _validate_optimizer(saved, names, shapes, step, config, optimizer=None):
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
        raise ValueError("V10 checkpoint has no valid AdamW optimizer state")
    groups, states = saved["param_groups"], saved["state"]
    if not isinstance(groups, list) or not isinstance(states, dict) or not isinstance(names, list) or len(groups) != len(names):
        raise ValueError("V10 optimizer parameter groups differ")
    if optimizer is not None and len(groups) != len(optimizer.param_groups):
        raise ValueError("V10 optimizer parameter groups differ")
    used, named = set(), set()
    for i, (group, group_names) in enumerate(zip(groups, names)):
        if not isinstance(group, dict) or not isinstance(group_names, list) or len(group.get("params", [])) != len(group_names):
            raise ValueError("V10 optimizer parameter groups differ")
        if optimizer is not None:
            current = optimizer.param_groups[i]
            if len(group["params"]) != len(current["params"]) or group.get("kind") != current.get("kind"):
                raise ValueError("V10 optimizer group ownership differs")
        # Check AdamW's actual numerical contract before load_state_dict can mutate it.
        for key in ("lr", "eps", "weight_decay"):
            value = group.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"V10 invalid AdamW {key}")
        betas = group.get("betas")
        if (not isinstance(betas, (list, tuple)) or len(betas) != 2
                or any(type(x) not in (int, float) or not 0 <= x < 1 for x in betas)):
            raise ValueError("V10 invalid AdamW betas")
        for key in ("amsgrad", "maximize", "capturable", "differentiable"):
            if type(group.get(key)) is not bool:
                raise ValueError(f"V10 invalid AdamW {key}")
        for key in ("foreach", "fused"):
            if group.get(key) is not None and type(group[key]) is not bool:
                raise ValueError(f"V10 invalid AdamW {key}")
        for identifier, name in zip(group["params"], group_names):
            if type(identifier) is not int or identifier in used or name not in shapes or name in named:
                raise ValueError("V10 optimizer parameter ownership/order invalid")
            used.add(identifier)
            named.add(name)
            values = states.get(identifier, {})
            if not isinstance(values, dict):
                raise ValueError("V10 invalid AdamW state")
            required = {"step", "exp_avg", "exp_avg_sq"}
            if group["amsgrad"]:
                required.add("max_exp_avg_sq")
            if values and set(values) != required:
                raise ValueError("V10 AdamW state has missing or unexpected fields")
            if values:
                count = values["step"]
                if (not torch.is_tensor(count) or count.ndim != 0 or count.dtype not in (torch.float32, torch.float64)
                        or not bool(torch.isfinite(count))
                        or float(count) < 0 or float(count) != int(float(count)) or float(count) > step):
                    raise ValueError("V10 invalid AdamW optimizer-boundary step")
                for key in required - {"step"}:
                    value = values[key]
                    if not torch.is_tensor(value) or tuple(value.shape) != shapes[name] or value.dtype != torch.float32:
                        raise ValueError("V10 optimizer tensor shape/dtype differs")
                    if not bool(torch.isfinite(value).all()) or (key != "exp_avg" and bool((value < 0).any())):
                        raise ValueError("V10 invalid/nonfinite AdamW moment")
                    if not config["projector"]["enabled"] and name.startswith("projector.") and bool(value.count_nonzero()):
                        raise ValueError("V10 disabled-control projector optimizer moments must be zero")
    if set(states) - used:
        raise ValueError("V10 optimizer has unowned parameter state")
    _finite_tree(saved, "optimizer state")


def _named_shapes(shapes):
    return {prefix + key: tuple(value) for filename, prefix in
            (("model.safetensors", "memory."), ("expert.safetensors", "expert."),
             ("projector.safetensors", "projector.")) for key, value in shapes[filename].items()}


def _validate_training(training, info, shapes, *, optimizer=None, names=None, restore=False):
    if (not isinstance(training, dict) or set(training) != {"optimizer", "optimizer_param_names", "rng", "extra", "optimizer_boundary"}
            or training["optimizer_boundary"] is not True or not isinstance(training["extra"], dict)):
        raise ValueError("V10 invalid optimizer-boundary training state")
    _finite_tree(training, "training state")
    rng = training["rng"]
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("V10 invalid RNG state")
    # Inference on CPU remains portable even if training used CUDA. CPU RNG is
    # still checked; exact CUDA device-count/state checks are only for resume.
    _validate_rng(rng if restore else {**rng, "cuda": None})
    if rng["cuda"] is not None and (not isinstance(rng["cuda"], (list, tuple)) or any(
            not torch.is_tensor(v) or v.dtype != torch.uint8 or v.ndim != 1 for v in rng["cuda"])):
        raise ValueError("V10 invalid saved CUDA RNG tensors")
    if training["optimizer"] is None:
        if restore or training["optimizer_param_names"] is not None:
            raise ValueError("V10 checkpoint has no optimizer state")
    else:
        _validate_optimizer(training["optimizer"], training["optimizer_param_names"], _named_shapes(shapes),
                            info["step"], info["config"], optimizer)
        if names is not None and training["optimizer_param_names"] != names:
            raise ValueError("V10 optimizer parameter ownership/order differs")


def _read(base_model, path, expected_stage=1):
    path = Path(path).resolve()
    info = json.loads((path / "checkpoint.json").read_text())
    if type(info.get("format_version")) is not int or info["format_version"] != 1:
        raise ValueError("Unsupported V10 checkpoint format")
    _config(info.get("config"))
    _validate_step(info.get("step"))
    if expected_stage is not None and (type(expected_stage) is not int or expected_stage != 1):
        raise ValueError("V10 expected stage must be 1")
    metadata = info.get("metadata", {})
    json.dumps(info, allow_nan=False)
    shapes = _base_shapes(base_model, info["config"], metadata)
    if set(metadata.get("payload_sha256", {})) != set(PAYLOADS):
        raise ValueError("V10 requires all four payload hashes")
    states = {}
    for filename in PAYLOADS:
        if metadata["payload_sha256"][filename] != file_sha256(path / filename):
            raise ValueError(f"V10 payload changed: {filename}")
        states[filename] = load_file(str(path / filename), device="cpu")
        _validate_state(states[filename], shapes[filename], filename)
    for prefix, filename in (("memory", "model.safetensors"), ("expert", "expert.safetensors"),
                             ("cvom", "cvom.safetensors"), ("projector", "projector.safetensors")):
        if metadata.get(prefix + "_sha256") != metadata["payload_sha256"][filename]:
            raise ValueError(f"V10 {prefix} payload hash alias differs")
    _validate_delta(states["projector.safetensors"], info["config"], info["step"])
    if metadata.get("training_state_sha256") != file_sha256(path / "training_state.pt"):
        raise ValueError("V10 optimizer/RNG state changed")
    training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    _validate_training(training, info, shapes)
    return info, states, training, shapes


def checkpoint_info(base_model, checkpoint, expected_stage=1):
    """Validate all payloads/header semantics without loading HAMLET or a parent."""
    return _read(base_model, checkpoint, expected_stage)[0]


def save_checkpoint(output_dir, step, memory, head, cvom, optimizer, config, metadata,
                    best=False, training_state=None):
    """Publish a new checkpoint only; call after an optimizer step, never mid-accumulation."""
    step = _validate_step(step)
    if type(best) is not bool:
        raise ValueError("V10 best must be boolean")
    _installed(memory, head, cvom, config)
    info = copy.deepcopy({"format_version": 1, "step": step, "config": config, "metadata": metadata})
    json.dumps(info, allow_nan=False)
    shapes = _base_shapes(info["metadata"]["base_model"]["path"], config, info["metadata"])
    states = {"model.safetensors": memory.state_dict(), "expert.safetensors": expert_state_dict(head),
              "cvom.safetensors": cvom.state_dict(), "projector.safetensors": projector_state_dict(head)}
    states = {file: {k: v.detach().cpu().contiguous().clone() for k, v in state.items()} for file, state in states.items()}
    for filename, state in states.items():
        _validate_state(state, shapes[filename], filename)
    _validate_delta(states["projector.safetensors"], config, step)
    training = {"optimizer": optimizer.state_dict() if optimizer is not None else None,
                "optimizer_param_names": _optimizer_names(optimizer, memory, head) if optimizer is not None else None,
                "rng": _rng_state(), "extra": {} if training_state is None else training_state,
                "optimizer_boundary": True}
    _validate_training(training, info, shapes)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"checkpoint-{step:06d}"
    if destination.exists():
        raise FileExistsError(f"V10 checkpoint already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=root))
    try:
        for filename, state in states.items():
            save_file(state, str(temporary / filename))
        info["metadata"]["payload_sha256"] = {name: file_sha256(temporary / name) for name in PAYLOADS}
        for prefix, filename in (("memory", "model.safetensors"), ("expert", "expert.safetensors"),
                                 ("cvom", "cvom.safetensors"), ("projector", "projector.safetensors")):
            info["metadata"][prefix + "_sha256"] = info["metadata"]["payload_sha256"][filename]
        torch.save(training, temporary / "training_state.pt")
        info["metadata"]["training_state_sha256"] = file_sha256(temporary / "training_state.pt")
        _atomic_json(temporary / "checkpoint.json", info)
        if destination.exists():
            raise FileExistsError(f"V10 checkpoint already exists: {destination}")
        temporary.rename(destination)
        if best:
            _atomic_json(root / "best_checkpoint.json", {"path": destination.name, "step": step})
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)  # Only this invocation's freshly allocated temporary directory.


def load_checkpoint(path, memory, head, cvom, optimizer=None):
    """Validate every bundle/optimizer/RNG field before mutating live objects."""
    path = Path(path).resolve()
    preliminary = json.loads((path / "checkpoint.json").read_text())
    info, states, training, shapes = _read(preliminary["metadata"]["base_model"]["path"], path)
    _installed(memory, head, cvom, info["config"])
    for filename, current in (("model.safetensors", memory.state_dict()), ("expert.safetensors", expert_state_dict(head)),
                               ("cvom.safetensors", cvom.state_dict()), ("projector.safetensors", projector_state_dict(head))):
        _validate_state(states[filename], {k: tuple(v.shape) for k, v in current.items()}, "installed " + filename)
    if optimizer is not None:
        names = _optimizer_names(optimizer, memory, head)
        _validate_training(training, info, shapes, optimizer=optimizer, names=names, restore=True)
    memory.load_state_dict(states["model.safetensors"], strict=True)
    load_expert_state_dict(head, states["expert.safetensors"])
    cvom.load_state_dict(states["cvom.safetensors"], strict=True)
    load_projector_state_dict(head, states["projector.safetensors"])
    if optimizer is not None:
        optimizer.load_state_dict(training["optimizer"])
        _restore_rng_state(training["rng"])
    return {**{key: info[key] for key in ("step", "config", "metadata")},
            "training_state": training["extra"] if optimizer is not None else {}}
