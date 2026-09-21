"""Hashed semantic-training extras bound to an ordinary V18 actor bundle.

The extra answer head is training-only. The optional storage manager is the
only additional online component. Existing checkpoints/caches are untouched.
"""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import torch
from safetensors.torch import load_file, save_file

from gr00t.long_memory.monitoring import _atomic_json
from run_scripts.robomme.train_archive_deployment_v9 import file_hash
from run_scripts.robomme.checkpoint_representation_v18 import checkpoint_info_v18, save_checkpoint_v18

VARIANT = "semantic_memory_v1"


def semantic_info(checkpoint, *, required=True):
    root = Path(checkpoint).resolve()
    path = root / "semantic.json"
    if not path.exists() and not required:
        return None
    info = json.loads(path.read_text())
    if (info.get("format_version") != 1 or info.get("variant") != VARIANT
            or type(info.get("stage")) is not int or info.get("stage") not in (1, 2)):
        raise ValueError("Not a semantic-memory checkpoint")
    actor = json.loads((root / "checkpoint.json").read_text())
    if actor["config"].get("driver_variant") != VARIANT or actor["config"]["train"]["stage"] != info["stage"]:
        raise ValueError("Semantic stage differs from actor training contract")
    if info.get("actor_payload_sha256") != actor["metadata"]["payload_sha256"]:
        raise ValueError("Semantic extras belong to another actor")
    for name, digest in info["actor_payload_sha256"].items():
        if name not in ("model.safetensors", "expert.safetensors") or file_hash(root / name) != digest:
            raise ValueError("Actor payload changed")
    expected = {"answers.safetensors"} | ({"storage.safetensors"} if info["storage_config"] is not None else set())
    if set(info.get("payload_sha256", {})) != expected:
        raise ValueError("Semantic payload inventory differs")
    for name, digest in info["payload_sha256"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"Semantic payload changed: {name}")
        state = load_file(str(root / name))
        if not state or any(t.dtype != torch.float32 or not bool(torch.isfinite(t).all()) for t in state.values()):
            raise ValueError("Invalid semantic FP32 tensors")
    if info["stage"] == 1 and info["storage_config"] is not None:
        raise ValueError("Stage 1 must retain FIFO")
    if info["metadata"].get("labels_at_inference") is not False:
        raise ValueError("Semantic labels must be training targets only")
    if info["storage_config"]:
        from run_scripts.robomme.semantic_memory_storage import StorageConfig
        StorageConfig(**info["storage_config"])
        cfg = actor["config"]["representation"]
        for key, value in (("capacity_events", cfg["capacity_events"]), ("num_tokens", cfg["num_short_tokens"]),
                           ("dim", cfg["hidden_dim"])):
            if info["storage_config"][key] != value:
                raise ValueError(f"Manager/actor mismatch: {key}")
    return info


def load_extras(checkpoint, answers=None, manager=None):
    info = semantic_info(checkpoint)
    if answers is not None:
        answers.load_state_dict(load_file(str(Path(checkpoint) / "answers.safetensors")), strict=True)
    if manager is not None:
        if asdict(manager.config) != info["storage_config"]:
            raise ValueError("Storage config differs from checkpoint")
        manager.load_state_dict(load_file(str(Path(checkpoint) / "storage.safetensors")), strict=True)
    return info


def load_manager(checkpoint, device="cpu"):
    from run_scripts.robomme.semantic_memory_storage import StorageConfig, StorageManager
    info = semantic_info(checkpoint)
    if info["storage_config"] is None:
        return None, info
    manager = StorageManager(StorageConfig(**info["storage_config"]))
    load_extras(checkpoint, manager=manager)
    return manager.to(device).eval().requires_grad_(False), info


def save_semantic(output, step, core, head, optimizer, config, metadata, answers,
                  answer_config, manager=None, extra_metadata=None):
    """Publish actor + extras atomically; never advertise a half-written bundle."""
    root = Path(output).resolve()
    destination = root / f"checkpoint-{step:06d}"
    if destination.exists():
        raise FileExistsError(destination)
    staging = Path(tempfile.mkdtemp(prefix=".semantic-publish-", dir=root))
    actor_dir = save_checkpoint_v18(staging, step, core, head, optimizer, config, metadata)
    states = {"answers.safetensors": answers.state_dict()}
    if manager is not None:
        states["storage.safetensors"] = manager.state_dict()
    for name, state in states.items():
        state = {key: value.detach().cpu().contiguous().clone() for key, value in state.items()}
        if any(t.dtype != torch.float32 or not bool(torch.isfinite(t).all()) for t in state.values()):
            raise FloatingPointError("Refusing nonfinite semantic weights")
        save_file(state, str(actor_dir / name))
    actor = json.loads((actor_dir / "checkpoint.json").read_text())
    info = {"format_version": 1, "variant": VARIANT, "stage": config["train"]["stage"],
        "actor_payload_sha256": actor["metadata"]["payload_sha256"],
        "answer_config": answer_config, "storage_config": asdict(manager.config) if manager is not None else None,
        "payload_sha256": {name: file_hash(actor_dir / name) for name in states},
        "metadata": {"labels_at_inference": False, **(extra_metadata or {})}}
    _atomic_json(actor_dir / "semantic.json", info)
    semantic_info(actor_dir)
    actor_dir.rename(destination)
    # Only remove staging files created by this function after publication.
    (staging / "last_checkpoint.json").unlink()
    staging.rmdir()
    _atomic_json(root / "last_checkpoint.json", {"path": destination.name, "step": step})
    return destination
