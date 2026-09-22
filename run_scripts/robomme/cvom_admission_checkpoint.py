"""Small immutable writer sidecars; the frozen V19 actor is referenced, never copied.

The parent identity covers its manifest AND actual reader/expert weights. An
admission checkpoint cannot silently attach to a different actor. Inspectors
do not initialize CUDA or load a simulator. All saved tensors must be FP32.
"""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import torch
from safetensors.torch import load_file, save_file

from gr00t.long_memory.checkpoint_v4 import file_sha256
from gr00t.long_memory.monitoring import _atomic_json
from run_scripts.robomme.checkpoint_representation_v18 import checkpoint_info_v18


VARIANT = "cvom_admission_v1"
MANIFEST = "cvom_admission.json"
WEIGHTS = "writer.safetensors"


def parent_identity(parent_checkpoint, *, base_model=None):
    path = Path(parent_checkpoint).resolve()
    preliminary = json.loads((path / "checkpoint.json").read_text())
    base = base_model or preliminary["metadata"]["base_model"]["path"]
    info = checkpoint_info_v18(base, path)
    return {"path": str(path), "checkpoint_sha256": file_sha256(path / "checkpoint.json"),
            "payload_sha256": {name: file_sha256(path / name)
                               for name in ("model.safetensors", "expert.safetensors")},
            "base_model": info["metadata"]["base_model"]}


def _config_matches_parent(config, parent):
    actor = json.loads((Path(parent["path"]) / "checkpoint.json").read_text())
    cfg = actor["config"]["representation"]
    for key, expected in (("dim", cfg["hidden_dim"]), ("num_tokens", cfg["num_short_tokens"]),
                          ("capacity_events", cfg["capacity_events"])):
        if config.get(key) != expected:
            raise ValueError(f"CVoM writer/parent mismatch: {key}")


def _new_controller(config):
    from run_scripts.robomme.cvom_admission_core import AdmissionConfig, CVoMAdmission
    # Inspect/load must not consume the policy's action RNG stream.
    with torch.random.fork_rng(devices=[]):
        return CVoMAdmission(AdmissionConfig(**config))


def _validate_tensors(state, controller):
    expected = controller.state_dict()
    if set(state) != set(expected):
        raise ValueError("CVoM writer tensor inventory differs")
    for name, value in state.items():
        if (value.shape != expected[name].shape or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Invalid CVoM writer tensor: {name}")


def inspect_checkpoint(checkpoint, parent_checkpoint=None, *, base_model=None):
    root = Path(checkpoint).resolve()
    info = json.loads((root / MANIFEST).read_text())
    if info.get("format_version") != 1 or info.get("variant") != VARIANT:
        raise ValueError("Not an immutable CVoM admission checkpoint")
    if info.get("arm") not in ("single", "coalitional"):
        raise ValueError("Unknown CVoM admission arm")
    if type(info.get("step")) is not int or info["step"] < 0:
        raise ValueError("Invalid CVoM writer checkpoint step")
    meta = info.get("metadata", {})
    if meta.get("actor_frozen") is not True or meta.get("future_inputs_at_inference") is not False:
        raise ValueError("CVoM must freeze actor and exclude future inputs at inference")
    parent = parent_identity(parent_checkpoint or info["parent_identity"]["path"], base_model=base_model)
    if parent != info.get("parent_identity"):
        raise ValueError("CVoM frozen parent identity changed or belongs to a different actor")
    _config_matches_parent(info["config"], parent)
    if set(info.get("payload_sha256", {})) != {WEIGHTS}:
        raise ValueError("CVoM payload inventory differs")
    digest = file_sha256(root / WEIGHTS)
    if info["payload_sha256"][WEIGHTS] != digest:
        raise ValueError("CVoM writer payload changed")
    controller = _new_controller(info["config"])
    _validate_tensors(load_file(str(root / WEIGHTS), device="cpu"), controller)
    files = {MANIFEST: file_sha256(root / MANIFEST), WEIGHTS: digest}
    training_hash = info.get("training_state_sha256")
    if training_hash is not None:
        if file_sha256(root / "training_state.pt") != training_hash:
            raise ValueError("CVoM training state changed")
        files["training_state.pt"] = training_hash
    return {**info, "writer_sha256": digest, "manifest_sha256": files[MANIFEST],
            "files_sha256": files}


def load_controller(checkpoint, parent_checkpoint=None, *, device="cpu", base_model=None):
    info = inspect_checkpoint(checkpoint, parent_checkpoint, base_model=base_model)
    controller = _new_controller(info["config"])
    controller.load_state_dict(load_file(str(Path(checkpoint) / WEIGHTS), device="cpu"), strict=True)
    return controller.to(device).eval().requires_grad_(False), info


def save_checkpoint(output_dir, controller, parent_checkpoint, *, arm, step, metadata, optimizer=None):
    """Publish at EXACT output_dir; never overwrite a parent or existing run."""
    destination = Path(output_dir).resolve()
    parent = parent_identity(parent_checkpoint)
    parent_path = Path(parent["path"])
    if (destination == parent_path or parent_path in destination.parents
            or destination in parent_path.parents):
        raise ValueError("Writer output must not overlap immutable actor checkpoint")
    if destination.exists():
        raise FileExistsError(f"Refusing checkpoint overwrite: {destination}")
    if arm not in ("single", "coalitional") or type(step) is not int or step < 0:
        raise ValueError("Invalid CVoM arm/step")
    if metadata.get("actor_frozen") is not True or metadata.get("future_inputs_at_inference") is not False:
        raise ValueError("Missing frozen actor/causal deployment declaration")
    config = asdict(controller.config)
    _config_matches_parent(config, parent)
    state = {k: v.detach().cpu().contiguous().clone() for k, v in controller.state_dict().items()}
    _validate_tensors(state, _new_controller(config))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    save_file(state, str(temporary / WEIGHTS))
    info = {"format_version": 1, "variant": VARIANT, "arm": arm, "step": step,
            "config": config, "parent_identity": parent, "metadata": dict(metadata),
            "payload_sha256": {WEIGHTS: file_sha256(temporary / WEIGHTS)}}
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state()},
                   temporary / "training_state.pt")
        info["training_state_sha256"] = file_sha256(temporary / "training_state.pt")
    _atomic_json(temporary / MANIFEST, info)
    inspect_checkpoint(temporary, parent_checkpoint)
    if destination.exists():
        raise FileExistsError(f"Refusing checkpoint overwrite: {destination}")
    temporary.rename(destination)
    return destination
