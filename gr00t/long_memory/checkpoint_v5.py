"""Atomic recall-training bundles with the unchanged v4 inference payload.

V5 changes supervision, not the deployed network. ``model.safetensors`` and
``expert.safetensors`` therefore retain the v4 contract. A separate, hashed
``recall.safetensors`` is required for training/resume and Stage-2 labels, but
is never an input to the robot policy. All three are published together.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file

from .checkpoint_v4 import (file_sha256, _state_sha256, _validate_state,
                            load_checkpoint_v4, save_checkpoint_v4,
                            v4_checkpoint_info)
from .monitoring import _atomic_json, _validate_step

RECIPE = "recall_continuation_v5"


def recall_state_sha256(recall_or_state):
    state = recall_or_state.state_dict() if hasattr(recall_or_state, "state_dict") else recall_or_state
    return _state_sha256(state)


def _recall_shapes(config):
    from .recall_v5 import RecallHeads

    if config.get("training_recipe") != RECIPE:
        raise ValueError("Expected recall_continuation_v5 training recipe")
    spec = config.get("recall", {})
    for name in ("hidden_dim", "num_classes"):
        if type(spec.get(name)) is not int or spec[name] < 1:
            raise ValueError(f"Invalid recall architecture: {name}")
    if spec["hidden_dim"] != config["memory"]["hidden_dim"]:
        raise ValueError("Recall head must use the actual shared reader dimension")
    with torch.device("meta"):
        module = RecallHeads(spec["hidden_dim"], spec["num_classes"])
    return {name: tuple(value.shape) for name, value in module.state_dict().items()}


def _validate_recall(config, metadata, state):
    _validate_state(state, _recall_shapes(config), label="recall")
    fingerprint = metadata.get("recall_labels_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("V5 requires an immutable recall label fingerprint")
    if config["stage"] == 2:
        if metadata.get("frozen_recall_sha256") != recall_state_sha256(state):
            raise ValueError("Stage-2 frozen recall head changed")
        parent_hash = metadata.get("stage1_parent", {}).get("recall_sha256")
        if not isinstance(parent_hash, str) or len(parent_hash) != 64 or any(c not in "0123456789abcdef" for c in parent_hash):
            raise ValueError("Stage 2 must record its Stage-1 recall file hash")


def v5_checkpoint_info(base_model, checkpoint, *, expected_stage=None):
    """Validate every file, including training-only heads, without a full VLM."""
    root = Path(checkpoint)
    info = v4_checkpoint_info(base_model, root, expected_stage=expected_stage)
    recall = load_file(str(root / "recall.safetensors"), device="cpu")
    _validate_recall(info["config"], info["metadata"], recall)
    if info["metadata"].get("recall_sha256") != file_sha256(root / "recall.safetensors"):
        raise ValueError("Recall weights changed after checkpoint publication")
    return info


def save_checkpoint_v5(output_dir, step, memory, head, recall, optimizer, config,
                       metadata, best=False):
    """Never expose a partially written bundle or overwrite an old experiment."""
    step = _validate_step(step)
    state = {name: value.detach().cpu().contiguous().clone()
             for name, value in recall.state_dict().items()}
    _validate_recall(config, metadata, state)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"checkpoint-{step:06d}"
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=".v5-publish-", dir=output))
    try:
        aux_file = staging / "recall.safetensors"
        save_file(state, str(aux_file))
        saved_meta = dict(metadata, recall_sha256=file_sha256(aux_file))
        bundle = save_checkpoint_v4(staging, step, memory, head, optimizer,
                                    config, saved_meta, best=False)
        aux_file.rename(bundle / aux_file.name)
        if destination.exists():
            raise FileExistsError(f"Checkpoint already exists: {destination}")
        bundle.rename(destination)
        if best:
            _atomic_json(output / "best_checkpoint.json", {"path": destination.name, "step": step})
        return destination
    finally:
        # Only this function's freshly created staging directory is removed.
        if staging.exists():
            shutil.rmtree(staging)


def load_checkpoint_v5(path, memory, head, recall, optimizer=None):
    """Restore auxiliary parameters along with the v4 optimizer/RNG contract."""
    root = Path(path)
    preliminary = json.loads((root / "checkpoint.json").read_text())
    info = v5_checkpoint_info(preliminary["metadata"]["base_model"]["path"], root)
    state = load_file(str(root / "recall.safetensors"), device="cpu")
    _validate_state(state, {name: value.shape for name, value in recall.state_dict().items()},
                    label="installed recall")
    load_checkpoint_v4(root, memory, head, optimizer)
    recall.load_state_dict(state, strict=True)
    return {key: info[key] for key in ("config", "metadata", "step")}
