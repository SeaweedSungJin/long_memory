"""Immutable V18 sidecars: original HAMLET weights are never rewritten.

The copied short transformer is reconstructed from the recorded base. Only its
LoRA delta is persisted, alongside the long reader and Action Expert adapters.
Episode banks are runtime state, not checkpoint parameters.
"""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import torch
from safetensors.torch import load_file, save_file

from gr00t.long_memory.checkpoint_v4 import file_sha256, _expected_expert_shapes, _validate_state
from gr00t.long_memory.expert_v4 import expert_state_dict, load_expert_state_dict
from gr00t.long_memory.hamlet import checkpoint_identity
from gr00t.long_memory.monitoring import _atomic_json, _rng_state, _restore_rng_state, _validate_step
from gr00t.long_memory.checkpoint_v7 import _validate_rng
from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18

VARIANT = "representation_v18"
PAYLOADS = ("model.safetensors", "expert.safetensors")


def checkpoint_info_v18(base_model, checkpoint):
    """Read-only integrity/schema checks; never instantiate the VLM or simulator."""
    path, base = Path(checkpoint).resolve(), Path(base_model).resolve()
    info = json.loads((path / "checkpoint.json").read_text())
    if info.get("format_version") != 1 or info.get("variant") != VARIANT:
        raise ValueError("Expected an immutable representation_v18 checkpoint")
    _validate_step(info.get("step"))
    cfg = RepresentationConfigV18(**info["config"]["representation"])
    meta = info["metadata"]
    if meta.get("base_model") != checkpoint_identity(base):
        raise ValueError("V18 original HAMLET checkpoint identity changed")
    if not meta.get("cache_fingerprint") or not meta.get("plan_sha256"):
        raise ValueError("V18 cache/query-plan provenance is missing")
    bc = json.loads((base / "config.json").read_text())
    if (bc.get("hamlet_mode") != "finetune" or bc.get("mem_cond_type", "cross_attn") != "cross_attn"
            or bc.get("memory_type", "moment_token") != "moment_token"):
        raise ValueError("V18 requires original moment-token/cross-attention HAMLET")
    for name, expected in (("feature_dim", bc["backbone_embedding_dim"]),
                           ("num_short_tokens", bc["n_moment_tokens"]),
                           ("short_window", bc["memory_window"]), ("time_scale", bc.get("memory_stride", 16))):
        if getattr(cfg, name) != expected:
            raise ValueError(f"V18 base/config mismatch: {name}")
    processor = json.loads((base / "processor_config.json").read_text())
    if cfg.state_dim != processor.get("processor_kwargs", processor)["max_state_dim"]:
        raise ValueError("V18 state width differs from original processor")
    for name in PAYLOADS:
        if meta.get("payload_sha256", {}).get(name) != file_sha256(path / name):
            raise ValueError(f"V18 payload modified: {name}")
        state = load_file(str(path / name), device="cpu")
        if not state or any(t.dtype != torch.float32 or not bool(torch.isfinite(t).all()) for t in state.values()):
            raise ValueError(f"V18 invalid/nonfinite FP32 state: {name}")
        if {k: list(t.shape) for k, t in state.items()} != meta.get("payload_shapes", {}).get(name):
            raise ValueError(f"V18 saved tensor schema differs: {name}")
        if name == "expert.safetensors":
            _validate_state(state, _expected_expert_shapes(base, info["config"]), label="V18 expert")
    return info


def load_checkpoint_v18(path, core, head, optimizer=None):
    path = Path(path).resolve()
    preliminary = json.loads((path / "checkpoint.json").read_text())
    info = checkpoint_info_v18(preliminary["metadata"]["base_model"]["path"], path)
    if asdict(core.config) != info["config"]["representation"]:
        raise ValueError("Installed V18 architecture does not match the bundle")
    state = load_file(str(path / "model.safetensors"), device="cpu")
    expert = load_file(str(path / "expert.safetensors"), device="cpu")
    _validate_state(state, {k: v.shape for k, v in core.delta_state_dict().items()}, label="V18 core")
    _validate_state(expert, {k: v.shape for k, v in expert_state_dict(head).items()}, label="V18 installed expert")
    training = None
    if optimizer is not None:
        if info["metadata"].get("training_state_sha256") != file_sha256(path / "training_state.pt"):
            raise ValueError("V18 optimizer/RNG payload changed")
        training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        _validate_rng(training["rng"])
        saved = training["optimizer"]["param_groups"]
        current = optimizer.state_dict()["param_groups"]
        if len(saved) != len(current) or any(a["name"] != b["name"] or len(a["params"]) != len(b["params"])
                                            for a, b in zip(saved, current)):
            raise ValueError("V18 optimizer groups differ")
    core.load_delta_state_dict(state)
    load_expert_state_dict(head, expert)
    if training is not None:
        optimizer.load_state_dict(training["optimizer"])
        _restore_rng_state(training["rng"])
    return info


def save_checkpoint_v18(output_dir, step, core, head, optimizer, config, metadata):
    step = _validate_step(step)
    if config["representation"] != asdict(core.config):
        raise ValueError("Checkpoint config differs from installed V18 core")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"checkpoint-{step:06d}"
    if destination.exists():
        raise FileExistsError(f"No checkpoint overwrite: {destination}")
    states = {"model.safetensors": core.delta_state_dict(), "expert.safetensors": expert_state_dict(head)}
    states = {f: {k: v.detach().cpu().contiguous().clone() for k, v in state.items()} for f, state in states.items()}
    for filename, state in states.items():
        _validate_state(state, {k: v.shape for k, v in state.items()}, label=filename)
    # On failure leave a clearly named unpublished temporary directory for inspection.
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=root))
    for filename, state in states.items():
        save_file(state, str(temporary / filename))
    info = {"format_version": 1, "variant": VARIANT, "step": step, "config": config, "metadata": dict(metadata)}
    info["metadata"]["payload_sha256"] = {f: file_sha256(temporary / f) for f in PAYLOADS}
    info["metadata"]["payload_shapes"] = {f: {k: list(v.shape) for k, v in state.items()} for f, state in states.items()}
    torch.save({"optimizer": optimizer.state_dict() if optimizer is not None else None,
                "rng": _rng_state()}, temporary / "training_state.pt")
    info["metadata"]["training_state_sha256"] = file_sha256(temporary / "training_state.pt")
    _atomic_json(temporary / "checkpoint.json", info)
    if destination.exists():
        raise FileExistsError(f"No checkpoint overwrite: {destination}")
    temporary.rename(destination)
    _atomic_json(root / "last_checkpoint.json", {"path": destination.name, "step": step})
    return destination
