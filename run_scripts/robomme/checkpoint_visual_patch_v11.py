"""Immutable visual-only V11 bundles with an explicit EXTERNAL frozen parent.

This format is deliberately not V7/V10 and is not self-contained. Deployment
requires the unchanged original HAMLET and the exact Stage-1 V7 archive-1250
parent named by ``metadata.frozen_parent``. Neither is copied or modified.
Only visual-memory weights and their optimizer/RNG/cursor are published here.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, fields
import json
import math
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file

from gr00t.long_memory.checkpoint_v4 import file_sha256
from gr00t.long_memory.checkpoint_v7 import _validate_rng, v7_checkpoint_info
from gr00t.long_memory.hamlet import checkpoint_identity
from gr00t.long_memory.monitoring import _atomic_json, _restore_rng_state, _rng_state, _validate_step
from run_scripts.robomme.visual_patch_memory_v11 import CAMERA_ORDER, VisualPatchConfig, VisualPatchMemoryV11


VARIANT = "visual_patch_v11"
PAYLOADS = ("visual.safetensors",)
PARENT_FILES = ("checkpoint.json", "model.safetensors", "expert.safetensors", "cvom.safetensors", "training_state.pt")


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _finite_tree(value):
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("V11 nonfinite training state")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise ValueError("V11 unsupported training-state key")
            _finite_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite_tree(item)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("V11 nonfinite training state")
    elif value is not None and not isinstance(value, (str, int, bool)):
        raise ValueError("V11 unsupported training-state value")


def _config(config):
    if not isinstance(config, dict) or config.get("trainer_variant") != VARIANT:
        raise ValueError("Expected visual_patch_v11; other checkpoint architectures are incompatible")
    if config.get("driver_variant", VARIANT) != VARIANT:
        raise ValueError("V11 driver identity differs")
    if type(config.get("stage")) is not int or config["stage"] != 1 or config.get("mode") != "visual_patch":
        raise ValueError("V11 supports Stage 1 visual_patch mode only")
    if config.get("camera_order") != list(CAMERA_ORDER):
        raise ValueError("V11 exact audited camera order required")
    spec = config.get("visual")
    if not isinstance(spec, dict) or set(spec) != {f.name for f in fields(VisualPatchConfig)}:
        raise ValueError("V11 exact visual-memory specification required")
    return VisualPatchConfig(**spec)


def parent_reference(base_model, parent_path):
    """Validate the real frozen archive, and fingerprint ALL five parent files."""
    path = Path(parent_path).resolve()
    info = v7_checkpoint_info(base_model, path, expected_stage=1)
    if info["config"]["mode"] != "archive" or info["step"] != 1250:
        raise ValueError("V11 requires the frozen V7 Stage-1 archive checkpoint at step 1250")
    return {"path": str(path), "step": 1250,
            "files_sha256": {name: file_sha256(path / name) for name in PARENT_FILES}}


def _metadata(base_model, config, metadata):
    cfg = _config(config)
    if not isinstance(metadata, dict) or metadata.get("base_model") != checkpoint_identity(base_model):
        raise ValueError("V11 original base identity differs")
    parent = metadata.get("frozen_parent")
    if (not isinstance(parent, dict) or set(parent) != {"path", "step", "files_sha256"}
            or type(parent["step"]) is not int or parent["step"] != 1250
            or not isinstance(parent["path"], str) or not Path(parent["path"]).is_absolute()):
        raise ValueError("V11 explicit external frozen-parent reference required")
    if parent_reference(base_model, parent["path"]) != parent:
        raise ValueError("V11 frozen parent changed")
    parent_info = json.loads((Path(parent["path"]) / "checkpoint.json").read_text())
    for key in ("feature_dim", "num_short_tokens", "time_scale"):
        if getattr(cfg, key) != parent_info["config"]["memory"][key]:
            raise ValueError(f"V11 visual {key} differs from frozen parent")
    if not isinstance(metadata.get("cache_fingerprint"), str) or not metadata["cache_fingerprint"]:
        raise ValueError("V11 cache fingerprint required")
    sources = metadata.get("source_sha256")
    if not isinstance(sources, dict) or not sources or any(not isinstance(k, str) or not _sha(v) for k, v in sources.items()):
        raise ValueError("V11 explicit source hashes required")
    if not isinstance(metadata.get("runtime"), dict) or not metadata["runtime"]:
        raise ValueError("V11 runtime provenance required")
    # Meta construction cannot consume the caller's initialization/noise RNG.
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        visual = VisualPatchMemoryV11(cfg)
        return {name: tuple(value.shape) for name, value in visual.state_dict().items()}


def _state(state, shapes, step):
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise ValueError("V11 visual tensor names differ")
    for name, value in state.items():
        if not torch.is_tensor(value) or tuple(value.shape) != shapes[name] or value.dtype != torch.float32:
            raise ValueError(f"V11 visual shape/dtype differs: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"V11 nonfinite visual tensor: {name}")
    if step == 0 and bool(state["output_projection.weight"].count_nonzero()):
        raise ValueError("V11 step-zero visual output projection must be zero")


def _installed(visual, config):
    if not isinstance(visual, VisualPatchMemoryV11) or asdict(visual.config) != asdict(_config(config)):
        raise ValueError("V11 installed visual semantic config differs")
    if any(p.dtype != torch.float32 for p in visual.parameters()):
        raise ValueError("V11 visual master parameters must be FP32")


def _optimizer_names(optimizer, visual):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("V11 supports AdamW exact resume only")
    allowed = {id(p): name for name, p in visual.named_parameters()}
    seen, result = set(), []
    for group in optimizer.param_groups:
        if group.get("kind") != "visual":
            raise ValueError("V11 optimizer groups must be visual-only")
        names = []
        for parameter in group["params"]:
            if id(parameter) not in allowed or id(parameter) in seen or not parameter.requires_grad:
                raise ValueError("V11 optimizer contains duplicate, frozen or non-visual parameters")
            seen.add(id(parameter))
            names.append(allowed[id(parameter)])
        if not names:
            raise ValueError("V11 empty optimizer group")
        result.append(names)
    if seen != set(allowed):
        raise ValueError("V11 optimizer must own every visual parameter exactly once")
    return result


def _optimizer(saved, names, shapes, step, current=None):
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
        raise ValueError("V11 invalid AdamW state")
    groups, states = saved["param_groups"], saved["state"]
    if not isinstance(groups, list) or not isinstance(states, dict) or not isinstance(names, list) or len(groups) != len(names):
        raise ValueError("V11 optimizer groups differ")
    if current is not None and len(groups) != len(current.param_groups):
        raise ValueError("V11 optimizer groups differ")
    seen, named = set(), set()
    for index, (group, keys) in enumerate(zip(groups, names)):
        if (not isinstance(group, dict) or group.get("kind") != "visual" or not isinstance(keys, list) or not keys
                or not isinstance(group.get("params"), list) or len(group["params"]) != len(keys)):
            raise ValueError("V11 optimizer ownership differs")
        if current is not None and len(group["params"]) != len(current.param_groups[index]["params"]):
            raise ValueError("V11 optimizer group size differs")
        for key in ("lr", "eps", "weight_decay"):
            value = group.get(key)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"V11 invalid AdamW {key}")
        betas = group.get("betas")
        if (not isinstance(betas, (list, tuple)) or len(betas) != 2
                or any(type(x) not in (float, int) or not 0 <= x < 1 for x in betas)):
            raise ValueError("V11 invalid AdamW betas")
        for key in ("amsgrad", "maximize", "capturable", "differentiable"):
            if type(group.get(key)) is not bool:
                raise ValueError(f"V11 invalid AdamW {key}")
        for key in ("foreach", "fused"):
            if group.get(key) is not None and type(group[key]) is not bool:
                raise ValueError(f"V11 invalid AdamW {key}")
        for identifier, name in zip(group["params"], keys):
            if type(identifier) is not int or identifier in seen or name not in shapes or name in named:
                raise ValueError("V11 optimizer parameter ownership/order invalid")
            seen.add(identifier)
            named.add(name)
            values = states.get(identifier, {})
            required = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if group["amsgrad"] else set())
            if not isinstance(values, dict) or (values and set(values) != required):
                raise ValueError("V11 invalid AdamW moment fields")
            if not values:
                continue
            count = values["step"]
            if (not torch.is_tensor(count) or count.ndim != 0 or count.dtype not in (torch.float32, torch.float64)
                    or not bool(torch.isfinite(count)) or float(count) < 0
                    or float(count) != int(float(count)) or float(count) > step):
                raise ValueError("V11 invalid optimizer-boundary step")
            for key in required - {"step"}:
                value = values[key]
                if not torch.is_tensor(value) or tuple(value.shape) != shapes[name] or value.dtype != torch.float32:
                    raise ValueError("V11 optimizer moment shape/dtype differs")
                if not bool(torch.isfinite(value).all()) or (key != "exp_avg" and bool((value < 0).any())):
                    raise ValueError("V11 invalid AdamW moment")
    if named != set(shapes) or set(states) - seen:
        raise ValueError("V11 optimizer must own all and only visual parameters")


def _training(training, step, shapes, *, optimizer=None, names=None, restore=False):
    if (not isinstance(training, dict) or set(training) != {"optimizer", "optimizer_param_names", "rng", "extra", "optimizer_boundary"}
            or training["optimizer_boundary"] is not True):
        raise ValueError("V11 invalid optimizer-boundary training state")
    _finite_tree(training)
    extra = training["extra"]
    if (not isinstance(extra, dict) or extra.get("driver_variant") != VARIANT
            or type(extra.get("window_cursor")) is not int or extra["window_cursor"] != step
            or not _sha(extra.get("plan_sha256"))):
        raise ValueError("V11 driver/cursor/plan identity differs")
    rng = training["rng"]
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("V11 invalid RNG state")
    _validate_rng(rng if restore else {**rng, "cuda": None})
    if rng["cuda"] is not None and (not isinstance(rng["cuda"], (tuple, list)) or any(
            not torch.is_tensor(v) or v.dtype != torch.uint8 or v.ndim != 1 for v in rng["cuda"])):
        raise ValueError("V11 invalid CUDA RNG tensors")
    if training["optimizer"] is None:
        if restore or training["optimizer_param_names"] is not None:
            raise ValueError("V11 checkpoint has no optimizer state")
    else:
        _optimizer(training["optimizer"], training["optimizer_param_names"], shapes, step, optimizer)
        if names is not None and names != training["optimizer_param_names"]:
            raise ValueError("V11 optimizer parameter ownership/order differs")


def _read(base_model, path, expected_stage=1):
    path = Path(path).resolve()
    info = json.loads((path / "checkpoint.json").read_text())
    if type(info.get("format_version")) is not int or info["format_version"] != 1:
        raise ValueError("Unsupported V11 checkpoint format")
    _config(info.get("config"))
    _validate_step(info.get("step"))
    if expected_stage is not None and (type(expected_stage) is not int or expected_stage != 1):
        raise ValueError("V11 expected stage must be 1")
    if info.get("self_contained") is not False:
        raise ValueError("V11 bundle must declare its external frozen-parent dependency")
    json.dumps(info, allow_nan=False)
    shapes = _metadata(base_model, info["config"], info.get("metadata"))
    meta = info["metadata"]
    if meta.get("payload_sha256") != {"visual.safetensors": file_sha256(path / "visual.safetensors")}:
        raise ValueError("V11 visual payload changed")
    state = load_file(str(path / "visual.safetensors"), device="cpu")
    _state(state, shapes, info["step"])
    if meta.get("training_state_sha256") != file_sha256(path / "training_state.pt"):
        raise ValueError("V11 optimizer/RNG state changed")
    training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    _training(training, info["step"], shapes)
    return info, state, training, shapes


def checkpoint_info(base_model, path, expected_stage=1):
    """Strict read-only validation; no HAMLET allocation or RNG consumption."""
    info, _, training, _ = _read(base_model, path, expected_stage)
    return {**info, "training_state": training["extra"]}


def save_checkpoint(output_dir, step, visual, optimizer, config, metadata, best=False, training_state=None):
    """Atomically publish a NEW step directory after validating every field."""
    step = _validate_step(step)
    if type(best) is not bool:
        raise ValueError("V11 best must be boolean")
    _installed(visual, config)
    info = copy.deepcopy({"format_version": 1, "self_contained": False, "step": step, "config": config, "metadata": metadata})
    json.dumps(info, allow_nan=False)
    shapes = _metadata(info["metadata"]["base_model"]["path"], config, info["metadata"])
    state = {name: value.detach().cpu().contiguous().clone() for name, value in visual.state_dict().items()}
    _state(state, shapes, step)
    training = {"optimizer": optimizer.state_dict() if optimizer is not None else None,
                "optimizer_param_names": _optimizer_names(optimizer, visual) if optimizer is not None else None,
                "rng": _rng_state(), "extra": training_state, "optimizer_boundary": True}
    _training(training, step, shapes)
    root = Path(output_dir).resolve()
    for protected in (Path(info["metadata"]["base_model"]["path"]).resolve(), Path(info["metadata"]["frozen_parent"]["path"]).resolve()):
        if root == protected or root.is_relative_to(protected):
            raise ValueError("V11 output cannot modify its frozen base/parent directory")
    destination = root / f"checkpoint-{step:06d}"
    if destination.exists():
        raise FileExistsError(f"V11 checkpoint already exists: {destination}")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=root))
    try:
        save_file(state, str(temporary / "visual.safetensors"))
        torch.save(training, temporary / "training_state.pt")
        info["metadata"]["payload_sha256"] = {"visual.safetensors": file_sha256(temporary / "visual.safetensors")}
        info["metadata"]["training_state_sha256"] = file_sha256(temporary / "training_state.pt")
        _atomic_json(temporary / "checkpoint.json", info)
        if destination.exists():
            raise FileExistsError(f"V11 checkpoint already exists: {destination}")
        temporary.rename(destination)
        if best:
            _atomic_json(root / "best_checkpoint.json", {"path": destination.name, "step": step})
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)  # Only this call's fresh, exact temporary directory.


def load_checkpoint(path, visual, optimizer=None):
    """Validate external parent, all weights and resume state BEFORE any mutation."""
    path = Path(path).resolve()
    preliminary = json.loads((path / "checkpoint.json").read_text())
    info, state, training, shapes = _read(preliminary["metadata"]["base_model"]["path"], path)
    _installed(visual, info["config"])
    _state(state, {n: tuple(v.shape) for n, v in visual.state_dict().items()}, info["step"])
    if optimizer is not None:
        names = _optimizer_names(optimizer, visual)
        _training(training, info["step"], shapes, optimizer=optimizer, names=names, restore=True)
    visual.load_state_dict(state, strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(training["optimizer"])
        _restore_rng_state(training["rng"])
    return {**info, "training_state": training["extra"]}
