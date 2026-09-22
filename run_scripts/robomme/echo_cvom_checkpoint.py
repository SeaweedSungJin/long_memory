"""Versioned ECHO sidecars: immutable HAMLET/parent, complete learned deltas.

No runtime bank, dataset annotations or future targets are checkpoint weights.
Every load validates all payloads before copying any parameter. A temporary
checkpoint is published only after weights/config/optimizer have been written.
"""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import torch
from safetensors.torch import load_file, save_file

from gr00t.long_memory.checkpoint_v4 import file_sha256, _validate_state, _expected_expert_shapes
from gr00t.long_memory.expert_v4 import expert_state_dict, load_expert_state_dict
from gr00t.long_memory.hamlet import checkpoint_identity
from gr00t.long_memory.monitoring import _atomic_json, _rng_state, _restore_rng_state, _validate_step
from gr00t.long_memory.checkpoint_v7 import _validate_rng
from run_scripts.robomme.cvom_admission_checkpoint import parent_identity

VARIANT = "echo_cvom_v1"
PAYLOADS = ("core.safetensors", "expert.safetensors")


def inspect_checkpoint(base_model, path):
    """Read-only verification. CUDA, VLM and simulator are never initialized."""
    from run_scripts.robomme.echo_cvom_core import EchoConfig, EchoMemoryV1
    from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18
    path = Path(path).resolve(strict=True)
    base_model = Path(base_model).resolve()
    info = json.loads((path / "checkpoint.json").read_text())
    if info.get("format_version") != 1 or info.get("variant") != VARIANT:
        raise ValueError("Expected echo_cvom_v1 checkpoint; legacy writers are not interchangeable")
    _validate_step(info.get("step"))
    if info.get("stage") not in (1, 2):
        raise ValueError("ECHO checkpoint must identify stage 1 or 2")
    cfg = RepresentationConfigV18(**info["config"]["representation"])
    ecfg = EchoConfig(**info["config"]["echo"])
    meta = info["metadata"]
    if meta.get("base_model") != checkpoint_identity(base_model):
        raise ValueError("Original HAMLET identity changed")
    if not meta.get("cache_fingerprint") or not meta.get("plan_sha256"):
        raise ValueError("Missing cache/plan provenance")
    recorded_parent = meta.get("parent_identity", {})
    if not recorded_parent or parent_identity(recorded_parent["path"]) != recorded_parent:
        raise ValueError("Frozen V19 initialization changed")
    if meta.get("future_inputs_at_inference") is not False:
        raise ValueError("Checkpoint must explicitly prohibit future inputs at inference")
    with torch.random.fork_rng(devices=[]):
        prototype = EchoMemoryV1(cfg, ecfg)
    expected = {"core.safetensors": {k: v.shape for k, v in prototype.delta_state_dict().items()},
                "expert.safetensors": _expected_expert_shapes(base_model, info["config"])}
    for name in PAYLOADS:
        if file_sha256(path / name) != meta.get("payload_sha256", {}).get(name):
            raise ValueError(f"ECHO payload modified: {name}")
        state = load_file(str(path / name), device="cpu")
        _validate_state(state, expected[name], label=name)
        if {k: list(t.shape) for k, t in state.items()} != meta.get("payload_shapes", {}).get(name):
            raise ValueError(f"Saved tensor schema differs: {name}")
    info["files_sha256"] = {name: file_sha256(path / name) for name in (*PAYLOADS, "checkpoint.json")}
    return info


def load_checkpoint(path, core, head, optimizer=None):
    path = Path(path).resolve(strict=True)
    preliminary = json.loads((path / "checkpoint.json").read_text())
    info = inspect_checkpoint(preliminary["metadata"]["base_model"]["path"], path)
    if (asdict(core.config) != info["config"]["representation"]
            or asdict(core.echo_config) != info["config"]["echo"]):
        raise ValueError("Installed ECHO config differs from saved architecture")
    state, expert = [load_file(str(path / name), device="cpu") for name in PAYLOADS]
    _validate_state(state, {k: v.shape for k, v in core.delta_state_dict().items()}, label="installed ECHO")
    _validate_state(expert, {k: v.shape for k, v in expert_state_dict(head).items()}, label="installed AE")
    training = None
    if optimizer is not None:
        if file_sha256(path / "training_state.pt") != info["metadata"].get("training_state_sha256"):
            raise ValueError("Optimizer/RNG state changed")
        training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        _validate_rng(training["rng"])
        saved, current = training["optimizer"]["param_groups"], optimizer.state_dict()["param_groups"]
        if len(saved) != len(current) or any(a["name"] != b["name"] or len(a["params"]) != len(b["params"])
                                            for a, b in zip(saved, current)):
            raise ValueError("Resume optimizer groups differ")
    core.load_delta_state_dict(state)
    load_expert_state_dict(head, expert)
    if training is not None:
        optimizer.load_state_dict(training["optimizer"])
        _restore_rng_state(training["rng"])
    return info


def save_checkpoint(output_dir, step, core, head, optimizer, config, metadata, *, stage):
    step = _validate_step(step)
    if stage not in (1, 2) or config["representation"] != asdict(core.config) or config["echo"] != asdict(core.echo_config):
        raise ValueError("Checkpoint stage/config mismatch")
    states = {"core.safetensors": core.delta_state_dict(), "expert.safetensors": expert_state_dict(head)}
    return _publish(output_dir, step, states, optimizer, config, metadata, stage)


def load_core(path):
    """Load only the small ECHO module for CPU-only utility learning."""
    from run_scripts.robomme.echo_cvom_core import EchoConfig, EchoMemoryV1
    from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18
    path = Path(path).resolve(strict=True)
    preliminary = json.loads((path / "checkpoint.json").read_text())
    info = inspect_checkpoint(preliminary["metadata"]["base_model"]["path"], path)
    with torch.random.fork_rng(devices=[]):
        core = EchoMemoryV1(RepresentationConfigV18(**info["config"]["representation"]),
                            EchoConfig(**info["config"]["echo"]))
    core.load_delta_state_dict(load_file(str(path / "core.safetensors"), device="cpu"))
    return core, info


def save_manager_checkpoint(output_dir, step, core, optimizer, source, config, metadata):
    """Update ONLY manager tensors; preserve exact reader/effect/AE payloads."""
    step = _validate_step(step)
    source = Path(source).resolve(strict=True)
    original_core, info = load_core(source)
    if config["representation"] != info["config"]["representation"] or config["echo"] != info["config"]["echo"]:
        raise ValueError("Writer-only checkpoint cannot change actor architecture")
    original, current = original_core.delta_state_dict(), core.delta_state_dict()
    if set(original) != set(current):
        raise ValueError("Writer update changed actor tensor inventory")
    for name, value in original.items():
        if not name.startswith("manager.") and not torch.equal(value.cpu(), current[name].detach().cpu()):
            raise ValueError(f"Writer training modified frozen actor tensor {name}")
    expert = load_file(str(source / "expert.safetensors"), device="cpu")
    return _publish(output_dir, step, {"core.safetensors": current, "expert.safetensors": expert},
                    optimizer, config, metadata, 2)


def _publish(output_dir, step, states, optimizer, config, metadata, stage):
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"checkpoint-{step:06d}"
    if destination.exists():
        raise FileExistsError(f"Never overwrite a checkpoint: {destination}")
    states = {name: {k: v.detach().float().cpu().contiguous().clone() for k, v in state.items()}
              for name, state in states.items()}
    for name, state in states.items():
        _validate_state(state, {k: v.shape for k, v in state.items()}, label=name)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=root))
    for name, state in states.items():
        save_file(state, str(temporary / name))
    info = {"format_version": 1, "variant": VARIANT, "stage": stage, "step": step,
            "config": config, "metadata": dict(metadata)}
    meta = info["metadata"]
    meta["payload_sha256"] = {name: file_sha256(temporary / name) for name in PAYLOADS}
    meta["payload_shapes"] = {name: {k: list(v.shape) for k, v in state.items()} for name, state in states.items()}
    torch.save({"optimizer": optimizer.state_dict() if optimizer is not None else None,
                "rng": _rng_state()}, temporary / "training_state.pt")
    meta["training_state_sha256"] = file_sha256(temporary / "training_state.pt")
    _atomic_json(temporary / "checkpoint.json", info)
    if destination.exists():
        raise FileExistsError(f"Checkpoint target appeared during save: {destination}")
    temporary.rename(destination)
    _atomic_json(root / "last_checkpoint.json", {"path": destination.name, "step": step, "stage": stage})
    return destination
