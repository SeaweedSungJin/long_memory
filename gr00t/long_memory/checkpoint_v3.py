"""Read-only validation for action-value v3 memory-only checkpoints.

V1/v2 checkpoints deliberately cannot enter this route. Checking small add-on
weights here catches incomplete or nonfinite saves before loading HAMLET or
starting a simulator. The frozen multi-GB base is identified using the same
contract as the feature cache; the evaluation driver additionally hashes files.
"""
from __future__ import annotations

import json
from pathlib import Path


VARIANT = "action_value_v3"


def memory_v3_checkpoint_info(base_model, memory_checkpoint, *, expected_stage=None):
    """Return validated metadata without allocating the full HAMLET model."""
    import torch
    from safetensors import safe_open

    from .core_v3 import ActionValueMemory, MemoryV3Config
    from .hamlet import checkpoint_identity

    base, checkpoint = Path(base_model).resolve(), Path(memory_checkpoint).resolve()
    info = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    config, metadata = info.get("config", {}), info.get("metadata", {})
    if info.get("format_version") != 1 or config.get("trainer_variant") != VARIANT:
        raise ValueError("Expected an action_value_v3 checkpoint; legacy v1/v2 weights are not compatible")
    stage = config.get("stage")
    if type(stage) is not int or stage not in (1, 2):
        raise ValueError("V3 checkpoint stage must be integer 1 or 2")
    if expected_stage is not None and stage != expected_stage:
        raise ValueError(f"Expected Stage {expected_stage}, got Stage {stage}: {checkpoint}")
    if type(info.get("step")) is not int or info["step"] < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    if metadata.get("base_model") != checkpoint_identity(base):
        raise ValueError("V3 memory was trained on a different/changed base checkpoint")
    if not isinstance(metadata.get("cache_fingerprint"), str) or not metadata["cache_fingerprint"]:
        raise ValueError("V3 checkpoint requires a nonempty training cache_fingerprint")
    cfg = MemoryV3Config(**config["memory"])
    base_config = json.loads((base / "config.json").read_text())
    processor = json.loads((base / "processor_config.json").read_text())
    processor = processor.get("processor_kwargs", processor)
    if (base_config.get("hamlet_mode") != "finetune"
            or base_config.get("memory_type", "moment_token") != "moment_token"
            or base_config.get("mem_cond_type", "cross_attn") != "cross_attn"
            or int(base_config.get("n_moment_tokens", 0)) <= 0):
        raise ValueError("V3 requires a trained HAMLET moment-token cross-attention base")
    for name, expected in (
        ("feature_dim", base_config["backbone_embedding_dim"]),
        ("state_dim", processor["max_state_dim"]),
        ("action_dim", processor["max_action_dim"]),
    ):
        if getattr(cfg, name) != int(expected):
            raise ValueError(f"V3 {name} differs from the base processor/model")

    # Meta construction checks exact names/shapes without random initialization,
    # advancing CPU RNG, or allocating real parameter storage.
    with torch.device("meta"):
        expected = ActionValueMemory(cfg).state_dict()
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError("V3 checkpoint tensor names do not match its architecture/config")
        for name, reference in expected.items():
            tensor = handle.get_tensor(name)
            if tensor.shape != reference.shape:
                raise ValueError(f"V3 checkpoint tensor shape mismatch: {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"V3 checkpoint contains nonfinite weights: {name}")
    return info
