"""Small, single-writer logging and memory-only checkpoint utilities.

Action loss / velocity MAE are continuous-action, offline metrics.  Write
accuracy, precision and recall describe the storage classifier, NOT simulator
task success.  Call these utilities only on the main training process.
"""

from __future__ import annotations

import csv
import json
import math
import numbers
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any
import warnings

import numpy as np
from safetensors.torch import load_file, save_file
import torch


_NULLABLE_VALIDATION_METRICS = {"write_precision", "write_recall", "utility_corr"}
_RESERVED_KEYS = {"step", "split"}
_METRIC_LABELS = {
    "action_loss": "Action flow-matching loss (offline)",
    "baseline_action_loss": "Baseline action loss (offline)",
    "wrong_memory_action_loss": "Wrong-memory action loss (offline)",
    "velocity_mae": "Flow velocity MAE (not task accuracy)",
    "utility_loss": "Utility regression loss",
    "write_loss": "Write classification loss",
    "write_accuracy": "Write classifier accuracy (not task success)",
    "write_precision": "Write classifier precision",
    "write_recall": "Write classifier recall",
    "utility_corr": "Utility / target correlation",
    "write_rate": "Memory write rate",
    "bank_fill": "Memory bank fill",
    "grad_norm": "Gradient norm",
    "elapsed_seconds": "Elapsed time (seconds)",
}
_METRIC_ORDER = (
    "action_loss", "baseline_action_loss", "wrong_memory_action_loss",
    "memory_gain", "velocity_mae", "loss", "utility_loss", "write_loss",
    "write_accuracy", "write_precision", "write_recall", "utility_corr",
    "positive_label_rate", "write_rate", "bank_fill", "signed_gain",
    "grad_norm", "gate_mean", "read_norm", "residual_norm", "null_weight",
    "oldest_event_age", "learning_rate", "elapsed_seconds",
)
_FRACTION_METRICS = {
    "write_accuracy", "write_precision", "write_recall", "positive_label_rate",
    "write_rate", "gate_mean", "null_weight",
}


def _validate_step(step: int) -> int:
    if isinstance(step, bool) or not isinstance(step, numbers.Integral) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    return int(step)


def _record(step: int, split: str, metrics: dict[str, Any]) -> dict[str, Any]:
    step = _validate_step(step)
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a nonempty string")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics must be a nonempty scalar dictionary")
    record: dict[str, Any] = {"step": step, "split": split}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name or name in _RESERVED_KEYS:
            raise ValueError(f"Invalid metric name: {name!r}")
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must be a scalar")
            value = value.detach().item()
        if value is None:
            if split not in {"val", "validation"} or name not in _NULLABLE_VALIDATION_METRICS:
                raise ValueError(f"None is only allowed for undefined validation precision/recall/correlation: {name}")
        elif isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite real scalar")
        else:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite metric {name}: {value}")
        record[name] = value
    return record


def _atomic_json(path: Path, value: Any) -> None:
    """Replace one small JSON file; leave the previous file on serialization failure."""
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class RunLogger:
    """Append metrics and render headless training / validation curves.

    ``metrics.jsonl`` is the canonical record. ``metrics.csv`` uses long format
    (step, split, metric, value), so newly introduced metrics need no header
    migration. On resume the CSV is rebuilt atomically from JSONL, recovering
    from an interruption between the two appends. Existing JSONL is never
    silently truncated; a partial/corrupt line raises an actionable error.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.csv_path = self.output_dir / "metrics.csv"
        self.records: list[dict[str, Any]] = []
        if self.jsonl_path.exists():
            with self.jsonl_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        row = json.loads(line)
                        step, split = row.pop("step"), row.pop("split")
                        self.records.append(_record(step, split, row))
                    except (ValueError, TypeError, KeyError, AttributeError) as exc:
                        raise ValueError(
                            f"Invalid log record at {self.jsonl_path}:{line_number}; "
                            "inspect/recover the file before resuming"
                        ) from exc
        elif self.csv_path.exists():
            raise ValueError("metrics.csv exists without canonical metrics.jsonl; refusing to overwrite it")
        # A run interrupted before its first log still has a valid empty journal.
        self.jsonl_path.touch(exist_ok=True)
        self._rebuild_csv()

    def _rebuild_csv(self) -> None:
        fd, temporary = tempfile.mkstemp(prefix=".metrics-", suffix=".csv", dir=self.output_dir)
        temporary_path = Path(temporary)
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["step", "split", "metric", "value"])
                for record in self.records:
                    writer.writerows(self._csv_rows(record))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.csv_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _csv_rows(record: dict[str, Any]):
        for key, value in record.items():
            if key not in _RESERVED_KEYS:
                yield [record["step"], record["split"], key, "" if value is None else value]

    def log(self, step: int, split: str, metrics: dict[str, Any]) -> None:
        record = _record(step, split, metrics)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(self._csv_rows(record))
            handle.flush()
            os.fsync(handle.fileno())

    def plot(self) -> Path:
        """Save latest and step-numbered PNGs atomically; no interactive backend."""
        if not self.records:
            raise ValueError("Cannot plot before at least one metric record exists")
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        available = set(
            key for record in self.records for key in record if key not in _RESERVED_KEYS
        )
        # Stable ordering across processes/resume; put learning metrics first,
        # regardless of which dictionary/validation record happened to arrive first.
        metric_names = [name for name in _METRIC_ORDER if name in available]
        metric_names += sorted(available - set(metric_names))
        columns = min(3, len(metric_names))
        rows = math.ceil(len(metric_names) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 3.5 * rows), squeeze=False)
        splits = list(dict.fromkeys(record["split"] for record in self.records))
        for axis, metric_name in zip(axes.flat, metric_names):
            has_values = False
            for split in splits:
                # Stable sorting preserves train/val and same-step repeated records.
                points = sorted((
                    (record["step"], record[metric_name])
                    for record in self.records
                    if record["split"] == split and record.get(metric_name) is not None
                ), key=lambda point: point[0])
                if points:
                    has_values = True
                    x, y = zip(*points)
                    axis.plot(x, y, marker="." if len(points) < 20 else None, label=split)
            axis.set_title(_METRIC_LABELS.get(metric_name, metric_name.replace("_", " ")))
            axis.set_xlabel("Optimizer step")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            if metric_name in _FRACTION_METRICS:
                axis.set_ylim(-0.02, 1.02)
            elif metric_name == "utility_corr":
                axis.set_ylim(-1.02, 1.02)
            axis.grid(alpha=0.25)
            if has_values:
                axis.legend()
            else:
                axis.text(0.5, 0.5, "Undefined for current validation samples", ha="center", transform=axis.transAxes)
        for axis in list(axes.flat)[len(metric_names):]:
            axis.set_visible(False)
        figure.suptitle(
            "Offline training / validation — action metrics are not simulator success rates\n"
            "Write accuracy / precision / recall refer only to the memory-storage classifier"
        )
        figure.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.5 * rows)))
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        step = max(record["step"] for record in self.records)
        latest_path = self.output_dir / "curves.png"
        snapshot_path = plots_dir / f"curves-step-{step:06d}.png"
        try:
            for destination in (snapshot_path, latest_path):
                fd, temporary = tempfile.mkstemp(prefix=".curves-", suffix=".png", dir=destination.parent)
                os.close(fd)
                temporary_path = Path(temporary)
                try:
                    figure.savefig(temporary_path, format="png", dpi=130)
                    os.replace(temporary_path, destination)
                finally:
                    temporary_path.unlink(missing_ok=True)
        finally:
            plt.close(figure)
        return latest_path


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        # Use only primitive values / tensors so torch.load(weights_only=True) works.
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    cuda_state = state["cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available() or len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError("Exact RNG resume requires the saved CUDA device count")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    metadata: dict[str, Any],
    best: bool = False,
    keep_last: int | None = 2,
) -> Path:
    """Save ONLY the passed memory module, never the base HAMLET model.

    Checkpoints are immutable and published only after all files are written.
    ``keep_last`` is advisory and currently retains all checkpoints for safety;
    no existing checkpoint is deleted. Pass None to suppress the reminder.
    Metadata should record the external base checkpoint / cache identity. RNG
    covers Python, NumPy and PyTorch global generators, not custom generators or
    data-loader state; the trainer must save those separately if it uses them.
    """
    step = _validate_step(step)
    if keep_last is not None:
        if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 1:
            raise ValueError("keep_last must be positive or None")
        warnings.warn("Checkpoint pruning is disabled for safety; keep_last retains all checkpoints", stacklevel=2)
    manifest = {"format_version": 1, "step": step, "config": config, "metadata": metadata}
    # Fail before creating a checkpoint if a manifest cannot be represented faithfully.
    json.dumps(manifest, allow_nan=False)
    weights: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name.startswith(("backbone.", "action_head.", "base_model.", "hamlet.", "action_expert.")):
            raise ValueError("Pass only the new memory module, not a wrapper containing the base HAMLET")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Non-tensor state_dict entry: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Refusing to save nonfinite weights: {name}")
        weights[name] = tensor.detach().to("cpu").contiguous().clone()
    if not weights:
        raise ValueError("Cannot save an empty memory state_dict")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"checkpoint-{step:06d}"
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists; refusing to overwrite: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=output_dir))
    try:
        save_file(weights, str(temporary / "model.safetensors"))
        torch.save(
            {"optimizer": optimizer.state_dict() if optimizer is not None else None, "rng": _rng_state()},
            temporary / "training_state.pt",
        )
        _atomic_json(temporary / "checkpoint.json", manifest)
        if destination.exists():
            raise FileExistsError(f"Checkpoint already exists; refusing to overwrite: {destination}")
        temporary.rename(destination)
        if best:
            _atomic_json(output_dir / "best_checkpoint.json", {"path": destination.name, "step": step})
        return destination
    finally:
        # Only the exact private directory created above can be removed on failure.
        if temporary.exists():
            shutil.rmtree(temporary)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Strictly load memory weights; restore optimizer/RNG only for a resume.

    Supplying no optimizer is an inference / weight-initialization load and
    deliberately leaves all global RNG streams unchanged. Base HAMLET weights
    must be loaded independently from the checkpoint referenced in metadata.
    """
    path = Path(path)
    manifest = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError(f"Unsupported memory checkpoint format: {manifest.get('format_version')}")
    _validate_step(manifest["step"])
    state = None
    if optimizer is not None:
        state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        if state.get("optimizer") is None:
            raise ValueError("This checkpoint has no optimizer state and cannot exactly resume")
    model.load_state_dict(load_file(str(path / "model.safetensors"), device="cpu"), strict=True)
    if optimizer is not None and state is not None:
        optimizer.load_state_dict(state["optimizer"])
        _restore_rng_state(state["rng"])
    return {key: manifest[key] for key in ("config", "metadata", "step")}
