"""Distinct V14 bundles: unchanged V13 visual architecture + adapted AE LoRA.

The original base and archive1250 files remain immutable external dependencies.
The archive actor/CVOM stay frozen, but the effective AE adapters DO NOT: their
complete adapted state is saved here, separately from the initial parent hash.
Inference needs no training sidecar/cache. Exact resume additionally restores
all 270 trainable tensors' AdamW state, RNG, and the immutable query cursor.
No V13 metadata coercion, adapter stacking, or original-weight rewriting occurs.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, fields
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file

from gr00t.long_memory.checkpoint_v4 import file_sha256, _validate_installed_head
from gr00t.long_memory.checkpoint_v7 import _validate_rng
from gr00t.long_memory.expert_v4 import (
    ExpertLoRALinear, LoRAConfig, TARGET_PATTERN, expert_state_dict, load_expert_state_dict,
)
from gr00t.long_memory.hamlet import checkpoint_identity
from gr00t.long_memory.monitoring import _atomic_json, _restore_rng_state, _rng_state, _validate_step
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.checkpoint_demo_tail_v13 import (
    base_reference, sidecar_reference, parent_reference, bind_visual_semantics,
    CAMERA_ORDER, EXTRACTION_RULE, REPLAY_ENCODING, SIDECAR_KIND, _sidecar_record,
)
from run_scripts.robomme.checkpoint_visual_differential_v12 import _finite_tree, _sha, _state
from run_scripts.robomme.visual_demo_tail_bank_v13 import VisualDemoTailMemoryV13, VisualDifferentialConfig

VARIANT = "visual_expert_v14"
MODE = "visual_expert"
ARCHITECTURE = "visual_demo_tail_v13"
PAYLOADS = ("visual.safetensors", "expert.safetensors")
RUN_MARKER = ".v14_checkpoint_run.json"
EXPECTED_REAL_EXPERT_TARGETS = sorted(f"model.transformer_blocks.{i}.attn1.{suffix}"
                                     for i in range(32) for suffix in ("to_q", "to_k", "to_v", "to_out.0"))


def verify_frozen_references(metadata):
    """Rescan physical originals; do not claim the adapted in-memory AE is frozen."""
    base = metadata["base_model"]["path"]
    if base_reference(base, refresh=True) != metadata["frozen_base"]:
        raise ValueError("V14 original base content changed")
    if parent_reference(base, metadata["initial_parent"]["path"]) != metadata["initial_parent"]:
        raise ValueError("V14 original archive initialization files changed")
    return True


def _config(config):
    if (not isinstance(config, dict) or config.get("trainer_variant") != VARIANT
            or config.get("driver_variant") != VARIANT or config.get("architecture") != ARCHITECTURE):
        raise ValueError("Expected genuine visual_expert_v14 with unchanged V13 visual architecture")
    if type(config.get("stage")) is not int or config["stage"] != 1 or config.get("mode") != MODE:
        raise ValueError("V14 supports Stage 1 visual_expert only")
    train = config.get("train", {})
    if (not isinstance(train, dict) or config.get("read_mode") != "differential"
            or train.get("read_mode") != "differential" or config.get("include_tail") is not True
            or train.get("include_tail") is not True or config.get("replay_encoding") != REPLAY_ENCODING
            or train.get("replay_encoding") != REPLAY_ENCODING):
        raise ValueError("V14 requires differential, include_tail=True, framewise replay")
    if config.get("extraction_rule") != EXTRACTION_RULE or config.get("camera_order") != list(CAMERA_ORDER):
        raise ValueError("V14 image extraction/camera contract changed")
    expert, targets = config.get("expert"), config.get("expert_targets")
    if not isinstance(expert, dict) or set(expert) != {"rank", "alpha"}:
        raise ValueError("V14 requires exact original AE LoRA configuration")
    LoRAConfig(**expert)
    if (not isinstance(targets, list) or not targets or any(not isinstance(n, str) or not TARGET_PATTERN.fullmatch(n) for n in targets)
            or targets != sorted(set(targets))):
        raise ValueError("V14 requires the complete sorted initial-parent AE attention targets")
    objective = config.get("objective", {})
    if (not isinstance(objective, dict) or objective.get("trainable_scope") != "visual_and_expert_lora"
            or objective.get("flow_weight") != 1.0 or objective.get("generated_auxiliary_weight") != 0.0
            or objective.get("frozen_original_base") is not True or objective.get("frozen_archive") is not True
            or "parent_frozen" in objective or objective.get("rollout_selection") != "fixed_final_step"):
        raise ValueError("V14 requires truthful joint visual/LoRA scope and unchanged flow objective")
    if type(train.get("max_steps")) is not int or train["max_steps"] <= 0:
        raise ValueError("V14 requires a positive immutable training horizon")
    spec = config.get("visual")
    if not isinstance(spec, dict) or set(spec) != {f.name for f in fields(VisualDifferentialConfig)}:
        raise ValueError("V14 exact visual configuration required")
    return VisualDifferentialConfig(**spec)


def _metadata(base_model, config, metadata):
    cfg = _config(config)
    if (not isinstance(metadata, dict) or metadata.get("base_model") != checkpoint_identity(base_model)
            or metadata.get("frozen_base") != base_reference(base_model)):
        raise ValueError("V14 original base identity/content changed")
    if "frozen_parent" in metadata:
        raise ValueError("V14 adapted AE must not be described as a frozen_parent")
    if any(metadata.get(k) != config[k] for k in ("read_mode", "include_tail", "extraction_rule", "replay_encoding")):
        raise ValueError("V14 metadata reader/tail semantics differ")
    if metadata.get("include_tail") is not True:
        raise ValueError("V14 metadata tail flag must be Boolean true")
    parent = metadata.get("initial_parent")
    if (not isinstance(parent, dict) or set(parent) != {"path", "step", "files_sha256"}
            or type(parent["step"]) is not int or parent["step"] != 1250
            or not isinstance(parent["path"], str) or not Path(parent["path"]).is_absolute()):
        raise ValueError("V14 explicit immutable archive1250 initialization reference required")
    if parent_reference(base_model, parent["path"]) != parent:
        raise ValueError("V14 initial parent files changed")
    parent_info = json.loads((Path(parent["path"]) / "checkpoint.json").read_text())
    for k in ("feature_dim", "num_short_tokens", "time_scale"):
        if getattr(cfg, k) != parent_info["config"]["memory"][k]:
            raise ValueError(f"V14 visual {k} differs from initial archive")
    for k in ("expert", "expert_targets"):
        if config[k] != parent_info["config"][k]:
            raise ValueError("V14 AE LoRA topology differs from initial archive")
    if (not isinstance(metadata.get("cache_fingerprint"), str) or not metadata["cache_fingerprint"]
            or not _sha(metadata.get("plan_sha256"))):
        raise ValueError("V14 immutable cache/query-plan identity required")
    _sidecar_record(metadata.get("sidecar"), metadata["cache_fingerprint"])
    if (metadata["sidecar"]["scope"] == "proof_subset_only"
            and config["train"].get("allow_proof_subset") is not True):
        raise ValueError("V14 proof-sidecar requires explicit diagnostic opt-in")
    sources = metadata.get("source_sha256")
    if not isinstance(sources, dict) or not sources or any(not isinstance(k, str) or not _sha(v) for k, v in sources.items()):
        raise ValueError("V14 complete source hashes required")
    if not isinstance(metadata.get("runtime"), dict) or not metadata["runtime"]:
        raise ValueError("V14 runtime provenance required")
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        visual = VisualDemoTailMemoryV13(cfg, read_mode="differential")
        visual_shapes = {n: tuple(t.shape) for n, t in visual.state_dict().items()}
    initial_expert = load_file(str(Path(parent["path"]) / "expert.safetensors"), device="cpu")
    expert_shapes = {n: tuple(t.shape) for n, t in initial_expert.items()}
    expected = {f"{target}.{suffix}" for target in config["expert_targets"] for suffix in ("lora_A", "lora_B")}
    if len(visual_shapes) != 14 or set(expert_shapes) != expected:
        raise ValueError("V14 must retain fourteen visual and all initial-parent LoRA tensors")
    return visual_shapes, expert_shapes, initial_expert


def _expert_state(state, shapes, step, initial):
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise ValueError("V14 expert tensor names differ")
    for name, value in state.items():
        if (not torch.is_tensor(value) or tuple(value.shape) != shapes[name]
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise ValueError(f"V14 expert tensor shape/dtype/finite mismatch: {name}")
        if step == 0 and not torch.equal(value.cpu(), initial[name]):
            raise ValueError("V14 step-zero expert must exactly equal immutable archive1250 LoRA")


def _installed(visual, head, config):
    cfg = _config(config)
    if (not isinstance(visual, VisualDemoTailMemoryV13) or asdict(visual.config) != asdict(cfg)
            or visual.read_mode != "differential" or getattr(visual, "include_tail", None) is not True
            or getattr(visual, "extraction_rule", None) != EXTRACTION_RULE
            or getattr(visual, "replay_encoding", None) != REPLAY_ENCODING):
        raise ValueError("V14 installed visual/tail/framewise semantics differ; bind explicitly")
    _validate_installed_head(head, config)
    adapters = [(n, m) for n, m in head.named_modules() if isinstance(m, ExpertLoRALinear)]
    if sorted(n for n, _ in adapters) != config["expert_targets"] or any(m.enabled is not True for _, m in adapters):
        raise ValueError("V14 exact original LoRA adapters must remain enabled")
    allowed = {id(p) for _, m in adapters for p in (m.lora_A, m.lora_B)}
    if head.training or any((p.requires_grad or p.grad is not None) and id(p) not in allowed for p in head.parameters()):
        raise ValueError("V14 head must be eval and original AE weights must remain frozen")
    if any(p.dtype != torch.float32 for p in visual.parameters()) or any(
            p.dtype != torch.float32 for _, m in adapters for p in (m.lora_A, m.lora_B)):
        raise ValueError("V14 visual/LoRA master parameters must be FP32")


def _named_parameters(visual, head):
    named = {"visual." + n: p for n, p in visual.named_parameters()}
    if len(named) != 14:
        raise ValueError("V14 expected exactly fourteen visual parameters")
    for n, module in head.named_modules():
        if isinstance(module, ExpertLoRALinear):
            for suffix in ("lora_A", "lora_B"):
                named[f"expert.{n}.{suffix}"] = getattr(module, suffix)
    if len(named) == 14 or len({id(p) for p in named.values()}) != len(named):
        raise ValueError("V14 expected distinct visual and installed LoRA parameters")
    return named


def optimizer_parameter_names(optimizer, visual, head):
    """Names are visual.<local> / expert.<full LoRA name>, owned exactly once."""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("V14 exact resume supports AdamW only")
    allowed = {id(p): n for n, p in _named_parameters(visual, head).items()}
    seen, names = set(), []
    for group in optimizer.param_groups:
        kind = group.get("kind")
        row = []
        if kind not in ("visual", "expert"):
            raise ValueError("V14 optimizer groups must be visual or expert")
        for p in group["params"]:
            name = allowed.get(id(p))
            if (name is None or id(p) in seen or not p.requires_grad or p.dtype != torch.float32
                    or not name.startswith(kind + ".")):
                raise ValueError("V14 duplicate/frozen/outside/wrong-group optimizer parameter")
            seen.add(id(p))
            row.append(name)
        if not row or group.get("param_names") != row:
            raise ValueError("V14 optimizer explicit canonical param_names/order differs")
        names.append(row)
    if seen != set(allowed):
        raise ValueError("V14 optimizer must own all visual/initial-parent LoRA parameters")
    return names


def _optimizer(saved, names, shapes, step, current=None):
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
        raise ValueError("V14 invalid AdamW state")
    groups, states = saved["param_groups"], saved["state"]
    if (not isinstance(groups, list) or not isinstance(states, dict) or not isinstance(names, list)
            or len(groups) != len(names) or current is not None and len(groups) != len(current.param_groups)):
        raise ValueError("V14 optimizer group count differs")
    seen, named = set(), set()
    for index, (group, keys) in enumerate(zip(groups, names)):
        if (not isinstance(group, dict) or group.get("kind") not in ("visual", "expert")
                or not isinstance(keys, list) or not keys or group.get("param_names") != keys
                or not isinstance(group.get("params"), list) or len(group["params"]) != len(keys)):
            raise ValueError("V14 optimizer ownership differs")
        if current is not None:
            present = current.param_groups[index]
            # LR itself is restored, not inferred from the current optimizer.
            if (group["kind"] != present.get("kind") or keys != present.get("param_names")
                    or len(group["params"]) != len(present["params"])):
                raise ValueError("V14 current optimizer ownership/order differs")
        for key in ("lr", "eps", "weight_decay"):
            value = group.get(key)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"V14 invalid AdamW {key}")
        betas = group.get("betas")
        if (not isinstance(betas, (list, tuple)) or len(betas) != 2
                or any(type(v) not in (float, int) or not 0 <= v < 1 for v in betas)):
            raise ValueError("V14 invalid AdamW betas")
        for key in ("amsgrad", "maximize", "capturable", "differentiable"):
            if type(group.get(key)) is not bool:
                raise ValueError(f"V14 invalid AdamW {key}")
        for key in ("foreach", "fused"):
            if group.get(key) is not None and type(group[key]) is not bool:
                raise ValueError(f"V14 invalid AdamW {key}")
        for identifier, name in zip(group["params"], keys):
            if (type(identifier) is not int or identifier in seen or name not in shapes or name in named
                    or not name.startswith(group["kind"] + ".")):
                raise ValueError("V14 optimizer parameter identity differs")
            seen.add(identifier)
            named.add(name)
            values = states.get(identifier, {})
            required = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if group["amsgrad"] else set())
            if not isinstance(values, dict) or values and set(values) != required:
                raise ValueError("V14 invalid AdamW moment fields")
            if not values:
                continue  # Legal before a parameter's first connected update.
            count = values["step"]
            if (not torch.is_tensor(count) or count.ndim != 0 or count.dtype not in (torch.float32, torch.float64)
                    or not bool(torch.isfinite(count)) or float(count) < 0
                    or float(count) != int(float(count)) or float(count) > step):
                raise ValueError("V14 invalid optimizer-boundary step")
            for key in required - {"step"}:
                value = values[key]
                if (not torch.is_tensor(value) or tuple(value.shape) != shapes[name] or value.dtype != torch.float32
                        or not bool(torch.isfinite(value).all()) or key != "exp_avg" and bool((value < 0).any())):
                    raise ValueError("V14 invalid AdamW moment shape/dtype/finite")
    if named != set(shapes) or set(states) - seen:
        raise ValueError("V14 optimizer must own exactly all visual and LoRA parameters")


def _training(training, step, shapes, config, metadata, *, optimizer=None, names=None, restore=False):
    if (not isinstance(training, dict) or set(training) != {"optimizer", "optimizer_param_names", "rng", "extra", "optimizer_boundary"}
            or training["optimizer_boundary"] is not True):
        raise ValueError("V14 invalid optimizer-boundary training state")
    _finite_tree(training)
    extra = training["extra"]
    if (not isinstance(extra, dict) or extra.get("driver_variant") != VARIANT
            or type(extra.get("window_cursor")) is not int or extra["window_cursor"] != step
            or extra.get("plan_sha256") != metadata["plan_sha256"] or extra.get("replay_encoding") != REPLAY_ENCODING
            or extra.get("include_tail") is not True or extra.get("sidecar_fingerprint") != metadata["sidecar"]["fingerprint"]):
        raise ValueError("V14 exact resume driver/cursor/plan/tail/framewise identity differs")
    rng = training["rng"]
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("V14 invalid RNG state")
    _validate_rng(rng if restore else {**rng, "cuda": None})
    if rng["cuda"] is not None and (not isinstance(rng["cuda"], (tuple, list)) or any(
            not torch.is_tensor(v) or v.dtype != torch.uint8 or v.ndim != 1 for v in rng["cuda"])):
        raise ValueError("V14 invalid CUDA RNG state")
    if training["optimizer"] is None:
        if restore or training["optimizer_param_names"] is not None:
            raise ValueError("V14 checkpoint has no optimizer for exact resume")
    else:
        _optimizer(training["optimizer"], training["optimizer_param_names"], shapes, step, optimizer)
        if names is not None and names != training["optimizer_param_names"]:
            raise ValueError("V14 optimizer names/order differs")


def _joint_shapes(visual_shapes, expert_shapes):
    return {**{"visual." + k: v for k, v in visual_shapes.items()},
            **{"expert." + k: v for k, v in expert_shapes.items()}}


def _read(base_model, path, expected_stage=1):
    path = Path(path).resolve()
    info = json.loads((path / "checkpoint.json").read_text())
    if info.get("kind") != VARIANT or type(info.get("format_version")) is not int or info["format_version"] != 1:
        raise ValueError("Expected genuine visual_expert_v14 bundle, not V13")
    _config(info.get("config"))
    step = _validate_step(info.get("step"))
    if step > info["config"]["train"]["max_steps"]:
        raise ValueError("V14 checkpoint exceeds immutable horizon")
    if expected_stage is not None and (type(expected_stage) is not int or expected_stage != 1):
        raise ValueError("V14 expected stage must be 1")
    if info.get("self_contained") is not False:
        raise ValueError("V14 requires external original base and initial archive files")
    json.dumps(info, allow_nan=False)
    vs, es, initial = _metadata(base_model, info["config"], info.get("metadata"))
    meta = info["metadata"]
    if meta.get("payload_sha256") != {n: file_sha256(path / n) for n in PAYLOADS}:
        raise ValueError("V14 visual/expert payload hash changed")
    visual = load_file(str(path / "visual.safetensors"), device="cpu")
    expert = load_file(str(path / "expert.safetensors"), device="cpu")
    _state(visual, vs, step)
    _expert_state(expert, es, step, initial)
    if meta.get("training_state_sha256") != file_sha256(path / "training_state.pt"):
        raise ValueError("V14 optimizer/RNG payload hash changed")
    training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    _training(training, step, _joint_shapes(vs, es), info["config"], meta)
    return info, visual, expert, training, _joint_shapes(vs, es)


def checkpoint_info(base_model, path, expected_stage=1):
    """Read-only strict preflight; sidecar/raw/cache paths need not exist."""
    info, _, _, training, _ = _read(base_model, path, expected_stage)
    return {**info, "training_state": training["extra"]}


def _run_identity(config, metadata):
    encoded = json.dumps(config, sort_keys=True, allow_nan=False).encode()
    return {"variant": VARIANT, "config_sha256": hashlib.sha256(encoded).hexdigest(),
            **{k: metadata[k] for k in ("frozen_base", "initial_parent", "cache_fingerprint", "plan_sha256",
                                      "sidecar", "include_tail", "source_sha256", "runtime")}}


def save_checkpoint(output_dir, step, visual, head, optimizer, config, metadata, best=False, training_state=None):
    """Publish a NEW immutable four-file bundle, never take over another run."""
    step = _validate_step(step)
    if type(best) is not bool:
        raise ValueError("V14 best must be Boolean")
    _installed(visual, head, config)
    if step > config["train"]["max_steps"]:
        raise ValueError("V14 checkpoint exceeds immutable horizon")
    info = copy.deepcopy({"format_version": 1, "kind": VARIANT, "self_contained": False,
                          "step": step, "config": config, "metadata": metadata})
    json.dumps(info, allow_nan=False)
    vs, es, initial = _metadata(metadata["base_model"]["path"], config, metadata)
    state = {n: v.detach().cpu().contiguous().clone() for n, v in visual.state_dict().items()}
    expert = {n: v.cpu().contiguous() for n, v in expert_state_dict(head).items()}
    _state(state, vs, step)
    _expert_state(expert, es, step, initial)
    training = {"optimizer": optimizer.state_dict() if optimizer is not None else None,
                "optimizer_param_names": optimizer_parameter_names(optimizer, visual, head) if optimizer is not None else None,
                "rng": _rng_state(), "extra": training_state, "optimizer_boundary": True}
    _training(training, step, _joint_shapes(vs, es), config, metadata)
    if step == config["train"]["max_steps"]:
        verify_frozen_references(metadata)
    if Path(output_dir).is_symlink():
        raise ValueError("V14 output cannot be a symlink")
    root = validate_output_scope(output_dir, metadata["base_model"]["path"], metadata["initial_parent"]["path"],
                                 metadata["sidecar"]["path"], config["train"].get("cache_dir"))
    destination = root / f"checkpoint-{step:06d}"
    if destination.exists():
        raise FileExistsError(f"V14 checkpoint already exists: {destination}")
    marker, identity = root / RUN_MARKER, _run_identity(config, metadata)
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("V14 output belongs to another run/configuration")
    elif root.exists() and ((root / "best_checkpoint.json").exists() or any(root.glob("checkpoint-*"))):
        raise FileExistsError("V14 refuses a preexisting checkpoint run without matching ownership")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=root))
    try:
        save_file(state, str(temporary / "visual.safetensors"))
        save_file(expert, str(temporary / "expert.safetensors"))
        torch.save(training, temporary / "training_state.pt")
        info["metadata"]["payload_sha256"] = {n: file_sha256(temporary / n) for n in PAYLOADS}
        info["metadata"]["training_state_sha256"] = file_sha256(temporary / "training_state.pt")
        _atomic_json(temporary / "checkpoint.json", info)
        if destination.exists():
            raise FileExistsError(f"V14 checkpoint already exists: {destination}")
        if not marker.exists():
            _atomic_json(marker, identity)
        temporary.rename(destination)
        if best:
            _atomic_json(root / "best_checkpoint.json", {"path": destination.name, "step": step})
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)  # This call's exact fresh temporary folder only.


def load_checkpoint(path, visual, head, optimizer=None):
    """Validate both modules and complete resume state before copying either."""
    path = Path(path).resolve()
    preliminary = json.loads((path / "checkpoint.json").read_text())
    info, state, expert, training, shapes = _read(preliminary["metadata"]["base_model"]["path"], path)
    _installed(visual, head, info["config"])
    _state(state, {n: tuple(v.shape) for n, v in visual.state_dict().items()}, info["step"])
    _expert_state(expert, {n: tuple(v.shape) for n, v in expert_state_dict(head).items()}, info["step"], expert)
    if optimizer is not None:
        names = optimizer_parameter_names(optimizer, visual, head)
        _training(training, info["step"], shapes, info["config"], info["metadata"],
                  optimizer=optimizer, names=names, restore=True)
    visual.load_state_dict(state, strict=True)
    load_expert_state_dict(head, expert, strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(training["optimizer"])
        _restore_rng_state(training["rng"])
    return {**info, "training_state": training["extra"]}
