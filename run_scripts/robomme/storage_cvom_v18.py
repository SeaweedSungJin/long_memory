"""V18 causal storage critic: KEEP or INSERT with FIFO-oldest replacement.

The actor/reader are frozen while this small MLP learns a *paired future action
loss difference*. It is not a robot-success predictor or an evidence oracle.
Only currently observed candidate tokens and the already populated bank enter
the model; future observations/actions are used by the training target only.
No retrieval-output re-storage, learned merge, or variable replacement occurs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


VARIANT = "storage_cvom_v18"


@dataclass(frozen=True)
class WriterConfigV18:
    memory_dim: int = 256
    capacity_events: int = 32
    hidden_dim: int = 128
    threshold: float = 0.0

    def __post_init__(self):
        for name in ("memory_dim", "capacity_events", "hidden_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.threshold):
            raise ValueError("Writer threshold must be finite")


def _bank_and_candidate(bank, candidate, capacity_events):
    if candidate.ndim != 3 or candidate.shape[0] != 1 or not candidate.shape[1]:
        raise ValueError("Writer candidate must be one event [1,Q,D]")
    if bank is None:
        bank = candidate[:, :0]
    if bank.ndim != 3 or bank.shape[0] != 1 or bank.shape[2] != candidate.shape[2]:
        raise ValueError("Writer bank must have shape [1,N*Q,D]")
    if bank.shape[1] % candidate.shape[1] or bank.shape[1] > capacity_events * candidate.shape[1]:
        raise ValueError("Bank must contain complete events within the fixed capacity")
    if bank.device != candidate.device or bank.dtype != candidate.dtype:
        raise ValueError("Bank and candidate dtype/device must match")
    if not bool(torch.isfinite(bank).all()) or not bool(torch.isfinite(candidate).all()):
        raise FloatingPointError("Nonfinite causal writer input")
    return bank, candidate


def storage_features(bank, candidate, capacity_events):
    """Pool event tokens; retain oldest-event and redundancy information."""
    bank, candidate = _bank_and_candidate(bank, candidate, capacity_events)
    q, d = candidate.shape[-2:]
    current = candidate.float().mean(1)
    events = bank.float().reshape(1, -1, q, d).mean(2)
    if events.shape[1]:
        average, oldest = events.mean(1), events[:, 0]
        cosine = F.cosine_similarity(events, current[:, None], dim=-1)
        statistics = torch.stack((cosine.max(1).values, cosine.mean(1), cosine[:, 0],
            current.new_full((1,), events.shape[1] / capacity_events)), dim=1)
    else:
        average, oldest = torch.zeros_like(current), torch.zeros_like(current)
        statistics = current.new_zeros((1, 4))
    return torch.cat((current, average, oldest, current - average, statistics), dim=-1)


def fifo_insert(bank, candidate, capacity_events):
    bank, candidate = _bank_and_candidate(bank, candidate, capacity_events)
    return torch.cat((bank, candidate), dim=1)[:, -capacity_events * candidate.shape[1]:]


class StorageCVOMV18(nn.Module):
    """LayerNorm -> Linear -> SiLU -> Linear signed-utility predictor.

    Zero initialization uses INSERT at score == threshold, exactly reproducing
    FIFO before training. Underfull banks are filled independently of scores.
    """
    def __init__(self, config: WriterConfigV18):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(nn.LayerNorm(4 * config.memory_dim + 4),
            nn.Linear(4 * config.memory_dim + 4, config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward_features(self, features):
        return self.network(features.float()).squeeze(-1)

    def forward(self, bank, candidate, capacity_events=None):
        capacity = self.config.capacity_events if capacity_events is None else capacity_events
        if capacity != self.config.capacity_events or candidate.shape[-1] != self.config.memory_dim:
            raise ValueError("Writer dimension/capacity differs from its training contract")
        return self.forward_features(storage_features(bank, candidate, capacity))


def make_write_policy(writer: StorageCVOMV18, capacity_events=None):
    """Common replay and deployment callback; caller supplies causal inputs."""
    capacity = writer.config.capacity_events if capacity_events is None else capacity_events
    if capacity != writer.config.capacity_events:
        raise ValueError("FIFO and CVOM must use the identical trained event budget")

    @torch.no_grad()
    def policy(bank, encoded_current, *, event_index=None, frame=None, is_demo=None):
        # Metadata arguments are intentionally unused; no ground-truth future,
        # demo action, outcome, or retrieval result is an input to the critic.
        bank, candidate = _bank_and_candidate(bank, encoded_current, capacity)
        full = bank.shape[1] == capacity * candidate.shape[1]
        score = float(writer(bank, candidate).item())
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite storage utility score")
        insert = not full or score >= writer.config.threshold
        updated = fifo_insert(bank, candidate, capacity) if insert else bank
        return updated, {"writer_insert": float(insert), "writer_full": float(full),
            "writer_score": score, "writer_keep": float(not insert)}

    return policy


def label_statistics(utility, *, margin=1e-6):
    values = torch.as_tensor(utility, dtype=torch.float64).flatten()
    if not len(values) or not bool(torch.isfinite(values).all()):
        raise ValueError("Storage labels must be nonempty and finite")
    positive, negative = values > margin, values < -margin
    return {"count": len(values), "positive_label_rate": float(positive.float().mean()),
        "negative_label_rate": float(negative.float().mean()),
        "ambiguous_label_rate": float((~(positive | negative)).float().mean()),
        "mean_utility": float(values.mean()), "mean_abs_utility": float(values.abs().mean()),
        "max_abs_utility": float(values.abs().max())}


def require_informative_labels(utility, *, margin=1e-6):
    result = label_statistics(utility, margin=margin)
    if result["positive_label_rate"] == 0 or result["negative_label_rate"] == 0:
        raise ValueError("CVOM labels lack both positive INSERT and negative KEEP examples. "
            "Do not fit or claim a useful writer: inspect label_audit.json, increase contexts/future "
            "coverage, or first improve the reader's sensitivity to storage changes.")
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parent_identity(path):
    """Bind every inference tensor and checkpoint metadata, not path alone."""
    path = Path(path).resolve(strict=True)
    metadata = path / "checkpoint.json"
    tensors = sorted(path.glob("*.safetensors"))
    if not metadata.is_file() or not tensors:
        raise ValueError("Writer requires a complete Stage-1 checkpoint directory")
    return {"path": str(path), "sha256": {p.name: file_sha256(p) for p in [metadata, *tensors]}}


def save_storage_writer_v18(path, writer, parent_checkpoint, *, step, metadata):
    from safetensors.torch import save_file
    from gr00t.long_memory.monitoring import _atomic_json

    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    state = {name: value.detach().cpu().contiguous() for name, value in writer.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise FloatingPointError("Cannot save a nonfinite writer")
    save_file(state, str(path / "writer.safetensors"))
    manifest = {"variant": VARIANT, "config": asdict(writer.config), "step": int(step),
        "parent": parent_identity(parent_checkpoint), "metadata": metadata,
        "write_actions": ["KEEP", "INSERT_OR_REPLACE_OLDEST"], "fill_policy": "always_insert",
        "future_inputs_at_inference": False, "writer_sha256": file_sha256(path / "writer.safetensors")}
    _atomic_json(path / "writer.json", manifest)
    return manifest


def load_storage_writer_v18(path, parent_checkpoint, *, device="cpu"):
    from safetensors.torch import load_file

    path = Path(path).resolve(strict=True)
    manifest = json.loads((path / "writer.json").read_text())
    if (manifest.get("variant") != VARIANT or manifest.get("future_inputs_at_inference") is not False
            or manifest.get("write_actions") != ["KEEP", "INSERT_OR_REPLACE_OLDEST"]
            or manifest.get("fill_policy") != "always_insert"):
        raise ValueError("Not a compatible causal V18 storage writer")
    actual = parent_identity(parent_checkpoint)
    if actual["sha256"] != manifest["parent"]["sha256"]:
        raise ValueError("Writer belongs to different Stage-1 reader/short/AE weights")
    if file_sha256(path / "writer.safetensors") != manifest["writer_sha256"]:
        raise ValueError("Writer tensor file changed")
    config = WriterConfigV18(**manifest["config"])
    writer = StorageCVOMV18(config)
    state = load_file(str(path / "writer.safetensors"))
    if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Writer tensors must be finite FP32")
    writer.load_state_dict(state, strict=True)
    return writer.to(device).eval().requires_grad_(False), config, manifest
