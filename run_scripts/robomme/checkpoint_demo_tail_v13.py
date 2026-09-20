"""Distinct V13 visual-only bundles; original base/archive1250 stay external.

Inference requires only this bundle, the exact original base, and archive1250.
The TRAIN/VAL sidecar is provenance, NOT an inference-time filesystem dependency.
Exact training resume must separately revalidate its sidecar/query plan, then
restore this bundle's visual-only AdamW state, cursor, and RNG. No V12 metadata
coercion, global overrides, old-file edits or large original-weight copies.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file

from gr00t.long_memory.checkpoint_v4 import file_sha256
from gr00t.long_memory.checkpoint_v7 import _validate_rng
from gr00t.long_memory.hamlet import checkpoint_identity
from gr00t.long_memory.monitoring import _atomic_json, _restore_rng_state, _rng_state, _validate_step
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.checkpoint_visual_differential_v12 import (
    _finite_tree, _optimizer, _optimizer_names, _sha, _state, parent_reference,
)
from run_scripts.robomme.demo_tail_sidecar_v13 import CAMERA_ORDER, DemoTailSidecar, KIND as SIDECAR_KIND
from run_scripts.robomme.visual_demo_tail_bank_v13 import VisualDemoTailMemoryV13, VisualDifferentialConfig

VARIANT = "visual_demo_tail_v13"
MODE = "visual_demo_tail"
EXTRACTION_RULE = "range(max(last_canonical_demo+1,n_demo-15,0),n_demo)"
RUN_MARKER = ".v13_checkpoint_run.json"
REPLAY_ENCODING = "framewise"
_BASE_CACHE = {}


def base_reference(base_model, *, refresh=False):
    """Full hashes once per unchanged process-visible base; force final rescan.

    The cached digest is invalidated by device/inode/size/mtime/ctime changes.
    Training must call verify_frozen_references before final completion; this
    explicitly rescans content instead of presenting a stat-cache hit as SHA.
    """
    if type(refresh) is not bool:
        raise ValueError("V13 refresh must be Boolean")
    root = Path(base_model).resolve(strict=True)
    identity = checkpoint_identity(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".safetensors")
                   and ".cache" not in p.parts)
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError("V13 original base has no safetensors")
    if any(not path.resolve().is_relative_to(root) for path in files):
        raise ValueError("V13 base payload escapes original directory")
    stats = tuple((str(path.relative_to(root)), (s := path.stat()).st_dev, s.st_ino,
                   s.st_size, s.st_mtime_ns, s.st_ctime_ns) for path in files)
    cached = _BASE_CACHE.get(str(root))
    if not refresh and cached is not None and cached[0] == stats:
        return copy.deepcopy(cached[1])
    reference = {"path": str(root), "identity": identity,
                 "files_sha256": {str(path.relative_to(root)): file_sha256(path) for path in files}}
    after = tuple((str(path.relative_to(root)), (s := path.stat()).st_dev, s.st_ino,
                   s.st_size, s.st_mtime_ns, s.st_ctime_ns) for path in files)
    if after != stats:
        raise ValueError("V13 original base changed during full-file hash scan")
    _BASE_CACHE[str(root)] = (stats, copy.deepcopy(reference))
    return reference


def verify_frozen_references(metadata):
    """Mandatory end-of-run fresh content scan; no sidecar dependency."""
    base = metadata["base_model"]["path"]
    if base_reference(base, refresh=True) != metadata["frozen_base"]:
        raise ValueError("V13 final original base content changed")
    if parent_reference(base, metadata["frozen_parent"]["path"]) != metadata["frozen_parent"]:
        raise ValueError("V13 final original archive parent changed")
    return True


def sidecar_reference(path, cache_fingerprint, *, allow_proof_subset=False):
    """Training-only completed inventory validation; never called by inference."""
    reader = DemoTailSidecar(path, expected_cache_fingerprint=cache_fingerprint)
    plan = reader.manifest["plan"]
    if type(allow_proof_subset) is not bool:
        raise ValueError("V13 proof-subset opt-in must be Boolean")
    scope = plan.get("scope")
    allowed = {"inventory_train_val", "proof_subset_only"} if allow_proof_subset else {"inventory_train_val"}
    if scope not in allowed or plan.get("rule") != EXTRACTION_RULE:
        raise ValueError("V13 training requires the explicit full TRAIN/VAL inventory, not a proof subset")
    if scope == "inventory_train_val":
        extra = reader.manifest.get("inventory_checks", {})
        if (reader.manifest.get("driver_variant") != "demo_tail_inventory_v13"
                or set(extra) != {"zero_action_calls", "original_short_memory_unchanged", "frozen_model_content_unchanged"}
                or not all(value is True for value in extra.values())):
            raise ValueError("V13 inventory-specific integrity checks did not pass")
    for eid in reader._records:
        reader.load(eid)
    return {"path": str(reader.path), "fingerprint": reader.manifest["fingerprint"],
            "manifest_sha256": file_sha256(reader.path / "manifest.json"), "cache_fingerprint": cache_fingerprint,
            "kind": SIDECAR_KIND, "scope": plan["scope"], "rule": EXTRACTION_RULE}


def bind_visual_semantics(visual, *, include_tail):
    """Bind instance-only experiment semantics, without adding/changing tensors."""
    if not isinstance(visual, VisualDemoTailMemoryV13) or visual.read_mode != "differential" or type(include_tail) is not bool:
        raise ValueError("V13 requires an actual differential reader and Boolean include_tail")
    semantics = (("include_tail", include_tail), ("extraction_rule", EXTRACTION_RULE), ("replay_encoding", REPLAY_ENCODING))
    for name, value in semantics:
        if hasattr(visual, name) and getattr(visual, name) != value:
            raise ValueError(f"V13 installed {name} cannot be silently changed")
    for name, value in semantics:
        setattr(visual, name, value)
    return visual


def _config(config):
    if (not isinstance(config, dict) or config.get("trainer_variant") != VARIANT
            or config.get("driver_variant", VARIANT) != VARIANT):
        raise ValueError("Expected visual_demo_tail_v13; V12 and other checkpoint formats are incompatible")
    if type(config.get("stage")) is not int or config["stage"] != 1 or config.get("mode") != MODE:
        raise ValueError("V13 supports Stage 1 visual_demo_tail mode only")
    train = config.get("train", {})
    if (config.get("read_mode") != "differential" or train.get("read_mode") != "differential"
            or type(config.get("include_tail")) is not bool or type(train.get("include_tail")) is not bool
            or train["include_tail"] != config["include_tail"]):
        raise ValueError("V13 explicit differential/include_tail training-arm identity required")
    if config.get("replay_encoding") != REPLAY_ENCODING or train.get("replay_encoding") != REPLAY_ENCODING:
        raise ValueError("V13 framewise replay encoding must match online one-observation APPEND")
    if config.get("extraction_rule") != EXTRACTION_RULE or config.get("camera_order") != list(CAMERA_ORDER):
        raise ValueError("V13 extraction rule/camera order differs from exact image proof")
    spec = config.get("visual")
    if not isinstance(spec, dict) or set(spec) != {field.name for field in fields(VisualDifferentialConfig)}:
        raise ValueError("V13 exact visual-memory configuration required")
    return VisualDifferentialConfig(**spec)


def _sidecar_record(value, cache_fingerprint):
    # Deliberately do NOT resolve/check existence/read files under this path.
    keys = {"path", "fingerprint", "manifest_sha256", "cache_fingerprint", "kind", "scope", "rule"}
    if (not isinstance(value, dict) or set(value) != keys or not isinstance(value["path"], str)
            or not Path(value["path"]).is_absolute() or not _sha(value["fingerprint"])
            or not _sha(value["manifest_sha256"]) or value["cache_fingerprint"] != cache_fingerprint
            or value["kind"] != SIDECAR_KIND or value["scope"] not in ("inventory_train_val", "proof_subset_only")
            or value["rule"] != EXTRACTION_RULE):
        raise ValueError("V13 explicit completed sidecar provenance is invalid")


def _metadata(base_model, config, metadata):
    cfg = _config(config)
    if not isinstance(metadata, dict) or metadata.get("base_model") != checkpoint_identity(base_model):
        raise ValueError("V13 original base identity differs")
    if metadata.get("frozen_base") != base_reference(base_model):
        raise ValueError("V13 original base full-file hashes changed")
    if any(metadata.get(key) != config[key] for key in ("read_mode", "include_tail", "extraction_rule", "replay_encoding")):
        raise ValueError("V13 metadata reader/tail semantics differ")
    if type(metadata.get("include_tail")) is not bool:
        raise ValueError("V13 metadata include_tail must be Boolean")
    parent = metadata.get("frozen_parent")
    if (not isinstance(parent, dict) or set(parent) != {"path", "step", "files_sha256"}
            or type(parent["step"]) is not int or parent["step"] != 1250
            or not isinstance(parent["path"], str) or not Path(parent["path"]).is_absolute()):
        raise ValueError("V13 explicit external archive1250 parent required")
    if parent_reference(base_model, parent["path"]) != parent:
        raise ValueError("V13 frozen archive parent changed")
    parent_info = json.loads((Path(parent["path"]) / "checkpoint.json").read_text())
    for key in ("feature_dim", "num_short_tokens", "time_scale"):
        if getattr(cfg, key) != parent_info["config"]["memory"][key]:
            raise ValueError(f"V13 visual {key} differs from frozen parent")
    if not isinstance(metadata.get("cache_fingerprint"), str) or not metadata["cache_fingerprint"] or not _sha(metadata.get("plan_sha256")):
        raise ValueError("V13 original cache and immutable query-plan fingerprint required")
    _sidecar_record(metadata.get("sidecar"), metadata["cache_fingerprint"])
    if (metadata["sidecar"]["scope"] == "proof_subset_only"
            and config["train"].get("allow_proof_subset") is not True):
        raise ValueError("V13 proof-subset checkpoint requires explicit diagnostic training opt-in")
    sources = metadata.get("source_sha256")
    if not isinstance(sources, dict) or not sources or any(not isinstance(k, str) or not _sha(v) for k, v in sources.items()):
        raise ValueError("V13 full training source hashes required")
    if not isinstance(metadata.get("runtime"), dict) or not metadata["runtime"]:
        raise ValueError("V13 runtime provenance required")
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        visual = VisualDemoTailMemoryV13(cfg, read_mode="differential")
        shapes = {name: tuple(value.shape) for name, value in visual.state_dict().items()}
    if len(shapes) != 14:
        raise ValueError("V13 must preserve the exact fourteen visual parameter tensors")
    return shapes


def _installed(visual, config):
    cfg = _config(config)
    if not isinstance(visual, VisualDemoTailMemoryV13) or asdict(visual.config) != asdict(cfg):
        raise ValueError("V13 installed module/visual configuration differs")
    if (visual.read_mode != "differential" or type(getattr(visual, "include_tail", None)) is not bool
            or visual.include_tail != config["include_tail"] or getattr(visual, "extraction_rule", None) != EXTRACTION_RULE
            or getattr(visual, "replay_encoding", None) != REPLAY_ENCODING):
        raise ValueError("V13 installed include_tail/read_mode/extraction_rule differs; bind explicitly, never coerce")
    if any(parameter.dtype != torch.float32 for parameter in visual.parameters()):
        raise ValueError("V13 visual master parameters must be FP32")


def _training(training, step, shapes, config, metadata, *, optimizer=None, names=None, restore=False):
    if (not isinstance(training, dict) or set(training) != {"optimizer", "optimizer_param_names", "rng", "extra", "optimizer_boundary"}
            or training["optimizer_boundary"] is not True):
        raise ValueError("V13 invalid optimizer-boundary training state")
    _finite_tree(training)
    extra = training["extra"]
    if (not isinstance(extra, dict) or extra.get("driver_variant") != VARIANT
            or type(extra.get("window_cursor")) is not int or extra["window_cursor"] != step
            or extra.get("plan_sha256") != metadata["plan_sha256"]
            or extra.get("replay_encoding") != REPLAY_ENCODING
            or type(extra.get("include_tail")) is not bool or extra["include_tail"] != config["include_tail"]
            or extra.get("sidecar_fingerprint") != metadata["sidecar"]["fingerprint"]):
        raise ValueError("V13 exact resume driver/cursor/query-plan/include_tail/sidecar identity differs")
    rng = training["rng"]
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("V13 invalid RNG state")
    _validate_rng(rng if restore else {**rng, "cuda": None})
    if rng["cuda"] is not None and (not isinstance(rng["cuda"], (tuple, list)) or any(
            not torch.is_tensor(v) or v.dtype != torch.uint8 or v.ndim != 1 for v in rng["cuda"])):
        raise ValueError("V13 invalid CUDA RNG state")
    if training["optimizer"] is None:
        if restore or training["optimizer_param_names"] is not None:
            raise ValueError("V13 checkpoint has no optimizer state for exact resume")
    else:
        _optimizer(training["optimizer"], training["optimizer_param_names"], shapes, step, optimizer)
        if names is not None and names != training["optimizer_param_names"]:
            raise ValueError("V13 visual optimizer parameter ownership/order differs")


def _read(base_model, path, expected_stage=1):
    path = Path(path).resolve()
    info = json.loads((path / "checkpoint.json").read_text())
    if info.get("kind") != VARIANT or type(info.get("format_version")) is not int or info["format_version"] != 1:
        raise ValueError("Expected visual_demo_tail_v13 bundle, not a V12 checkpoint")
    _config(info.get("config"))
    _validate_step(info.get("step"))
    if expected_stage is not None and (type(expected_stage) is not int or expected_stage != 1):
        raise ValueError("V13 expected stage must be 1")
    if info.get("self_contained") is not False:
        raise ValueError("V13 bundle must declare external frozen base and archive parent")
    json.dumps(info, allow_nan=False)
    shapes = _metadata(base_model, info["config"], info.get("metadata"))
    metadata = info["metadata"]
    if metadata.get("payload_sha256") != {"visual.safetensors": file_sha256(path / "visual.safetensors")}:
        raise ValueError("V13 visual payload hash changed")
    state = load_file(str(path / "visual.safetensors"), device="cpu")
    _state(state, shapes, info["step"])
    if metadata.get("training_state_sha256") != file_sha256(path / "training_state.pt"):
        raise ValueError("V13 optimizer/RNG state hash changed")
    training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    _training(training, info["step"], shapes, info["config"], metadata)
    return info, state, training, shapes


def checkpoint_info(base_model, path, expected_stage=1):
    """Strict read-only inference preflight; no sidecar/raw dataset dependency."""
    info, _, training, _ = _read(base_model, path, expected_stage)
    return {**info, "training_state": training["extra"]}


def _run_identity(config, metadata):
    encoded = json.dumps(config, sort_keys=True, allow_nan=False).encode()
    return {"variant": VARIANT, "config_sha256": hashlib.sha256(encoded).hexdigest(),
            **{key: metadata[key] for key in ("frozen_base", "frozen_parent", "cache_fingerprint", "plan_sha256",
                                             "sidecar", "include_tail", "source_sha256", "runtime")}}


def save_checkpoint(output_dir, step, visual, optimizer, config, metadata, best=False, training_state=None):
    """Publish one immutable NEW checkpoint; never take over an older run."""
    step = _validate_step(step)
    if type(best) is not bool:
        raise ValueError("V13 best must be Boolean")
    _installed(visual, config)
    info = copy.deepcopy({"format_version": 1, "kind": VARIANT, "self_contained": False,
                          "step": step, "config": config, "metadata": metadata})
    json.dumps(info, allow_nan=False)
    shapes = _metadata(metadata["base_model"]["path"], config, metadata)
    state = {name: value.detach().cpu().contiguous().clone() for name, value in visual.state_dict().items()}
    _state(state, shapes, step)
    training = {"optimizer": optimizer.state_dict() if optimizer is not None else None,
                "optimizer_param_names": _optimizer_names(optimizer, visual) if optimizer is not None else None,
                "rng": _rng_state(), "extra": training_state, "optimizer_boundary": True}
    _training(training, step, shapes, config, metadata)
    if step == config.get("train", {}).get("max_steps"):
        verify_frozen_references(metadata)
    if Path(output_dir).is_symlink():
        raise ValueError("V13 output cannot be a symlink")
    root = validate_output_scope(output_dir, metadata["base_model"]["path"], metadata["frozen_parent"]["path"],
                                 metadata["sidecar"]["path"], config.get("train", {}).get("cache_dir"))
    destination = root / f"checkpoint-{step:06d}"
    if destination.exists():
        raise FileExistsError(f"V13 checkpoint already exists: {destination}")
    marker, identity = root / RUN_MARKER, _run_identity(config, metadata)
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("V13 output belongs to another run/configuration")
    elif root.exists() and ((root / "best_checkpoint.json").exists() or any(root.glob("checkpoint-*"))):
        raise FileExistsError("V13 refuses a preexisting checkpoint run without matching ownership")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=root))
    try:
        save_file(state, str(temporary / "visual.safetensors"))
        torch.save(training, temporary / "training_state.pt")
        info["metadata"]["payload_sha256"] = {"visual.safetensors": file_sha256(temporary / "visual.safetensors")}
        info["metadata"]["training_state_sha256"] = file_sha256(temporary / "training_state.pt")
        _atomic_json(temporary / "checkpoint.json", info)
        if destination.exists():
            raise FileExistsError(f"V13 checkpoint already exists: {destination}")
        if not marker.exists():
            _atomic_json(marker, identity)
        temporary.rename(destination)
        if best:
            _atomic_json(root / "best_checkpoint.json", {"path": destination.name, "step": step})
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)  # Only this call's exact fresh temp directory.


def load_checkpoint(path, visual, optimizer=None):
    """Validate all semantics/weights/RNG/optimizer before the first mutation."""
    path = Path(path).resolve()
    preliminary = json.loads((path / "checkpoint.json").read_text())
    info, state, training, shapes = _read(preliminary["metadata"]["base_model"]["path"], path)
    _installed(visual, info["config"])
    _state(state, {name: tuple(value.shape) for name, value in visual.state_dict().items()}, info["step"])
    if optimizer is not None:
        names = _optimizer_names(optimizer, visual)
        _training(training, info["step"], shapes, info["config"], info["metadata"], optimizer=optimizer, names=names, restore=True)
    visual.load_state_dict(state, strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(training["optimizer"])
        _restore_rng_state(training["rng"])
    return {**info, "training_state": training["extra"]}
