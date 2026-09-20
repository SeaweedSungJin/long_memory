"""Read-only live TensorBoard mirror of the long-memory JSONL training logs.

The trainer keeps its existing append-only ``metrics.jsonl`` journal.  This
separate process imports its history, tails new complete records, and serves
TensorBoard on localhost.  Stopping/restarting the monitor never stops training.

Each monitor invocation creates a NEW event session, and TensorBoard serves only
that session.  Thus restarting does not append duplicate history to old event
files.  Source logs, checkpoints, and earlier event sessions are never modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
import math
import numbers
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Protocol


class ScalarWriter(Protocol):
    def add_scalar(self, tag: str, value: float, step: int) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class TensorBoardScalarWriter:
    """Use TensorBoard directly without a model or any GPU tensor allocation."""

    def __init__(self, log_dir: Path):
        from tensorboard.compat.proto import event_pb2, summary_pb2
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        self._event = event_pb2.Event
        self._summary = summary_pb2.Summary
        self._writer = EventFileWriter(str(log_dir))

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_event(self._event(
            wall_time=time.time(), step=step,
            summary=self._summary(value=[self._summary.Value(tag=tag, simple_value=value)]),
        ))

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


@dataclass
class _TailState:
    identity: tuple[int, int]
    offset: int = 0
    line_number: int = 0


def _validate_record(record: Any) -> tuple[int, str, dict[str, float]]:
    """Validate a complete line before emitting ANY scalar from that record."""
    if not isinstance(record, dict):
        raise ValueError("record must be a JSON object")
    step, split = record.get("step"), record.get("split")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if (
        not isinstance(split, str) or not split.strip()
        or split in {".", "..", "comparison"} or "/" in split or "\\" in split
    ):
        raise ValueError("split must be a safe nonempty directory component")
    metrics: dict[str, float] = {}
    for name, value in record.items():
        if name in {"step", "split"}:
            continue
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be nonempty strings")
        # Precision, recall, and correlation can legitimately be undefined on
        # small validation sets.  Do not turn undefined values into fake zeros.
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite scalar or null")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        metrics[name] = value
    return step, split, metrics


class JournalMirror:
    """Incrementally mirror all descendant ``metrics.jsonl`` journals.

    Layout in the TensorBoard session::

        <relative training directory>/train             action_loss, ...
        <relative training directory>/val               action_loss, ...
        <relative training directory>/comparison/baseline     action_loss
        <relative training directory>/comparison/wrong-memory action_loss

    SAME tags in DIFFERENT runs produce overlaid train/validation and baseline
    curves. Original metric tags remain in the ordinary validation run. V2
    adds first/FIFO/random/current-all and frozen Stage-1 controls only when
    those metrics exist; old journals keep exactly their previous layout.
    """

    def __init__(
        self, log_dir: str | Path, event_dir: str | Path,
        writer_factory: Callable[[Path], ScalarWriter] = TensorBoardScalarWriter,
    ):
        self.log_dir = Path(log_dir).resolve()
        self.event_dir = Path(event_dir).resolve()
        if self.log_dir == self.event_dir:
            raise ValueError("event_dir must differ from source log_dir")
        self.writer_factory = writer_factory
        self._tails: dict[Path, _TailState] = {}
        self._writers: dict[Path, ScalarWriter] = {}
        self.total_records = 0
        self.latest: dict[str, tuple[int, str]] = {}

    def _journals(self):
        # os.walk lets us skip all previous event sessions cheaply.  It does
        # not follow directory symlinks; a symlinked source file is rejected.
        for directory, subdirs, filenames in os.walk(self.log_dir, followlinks=False):
            parent = Path(directory)
            subdirs[:] = sorted(
                name for name in subdirs
                if name != ".tensorboard_live" and (parent / name).resolve() != self.event_dir
            )
            if "metrics.jsonl" in filenames:
                path = parent / "metrics.jsonl"
                if path.is_symlink():
                    raise ValueError(f"Refusing a symlinked source journal: {path}")
                yield path

    def _writer(self, run: Path) -> ScalarWriter:
        if run not in self._writers:
            self._writers[run] = self.writer_factory(self.event_dir / run)
        return self._writers[run]

    def _emit(self, path: Path, step: int, split: str, metrics: dict[str, float]):
        relative = path.parent.relative_to(self.log_dir)
        writer = self._writer(relative / split)
        for name, value in metrics.items():
            writer.add_scalar(name, value, step)
        if split in {"val", "validation"}:
            for source, curve in (
                ("baseline_action_loss", "baseline"),
                ("wrong_memory_action_loss", "wrong-memory"),
                ("first_action_loss", "first-min-fill"),
                ("fifo_action_loss", "matched-fifo"),
                ("random_action_loss", "matched-random"),
                ("all_action_loss", "all-fifo"),
                ("stage1_all_action_loss", "frozen-stage1"),
            ):
                if source in metrics:
                    self._writer(relative / "comparison" / curve).add_scalar(
                        "action_loss", metrics[source], step,
                    )
        self.latest[str(relative)] = (step, split)

    def poll(self) -> int:
        """Import new complete lines, discover new runs, and flush all writers.

        A partial final line stays unread until its newline arrives.  Completed
        malformed lines, truncation, and replacement of an already-seen log are
        explicit errors, never silent data loss or duplicated history.
        """
        added = 0
        for path in self._journals():
            with path.open("rb") as handle:
                stat = os.fstat(handle.fileno())
                identity = (stat.st_dev, stat.st_ino)
                state = self._tails.setdefault(path, _TailState(identity))
                if identity != state.identity or stat.st_size < state.offset:
                    raise ValueError(
                        f"Journal was replaced or truncated: {path}. "
                        "Check the source log, then restart the monitor to rebuild its mirror."
                    )
                handle.seek(state.offset)
                while True:
                    line = handle.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    try:
                        step, split, metrics = _validate_record(json.loads(line))
                    except (ValueError, TypeError, UnicodeError) as exc:
                        raise ValueError(
                            f"Invalid completed JSONL record at {path}:{state.line_number + 1}: {exc}. "
                            "The monitor has not modified this source log."
                        ) from exc
                    self._emit(path, step, split, metrics)
                    state.offset = handle.tell()
                    state.line_number += 1
                    self.total_records += 1
                    added += 1
        for writer in self._writers.values():
            writer.flush()
        return added

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()


def _positive_seconds(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logdir", type=Path, default=Path("runs/long_memory"),
                        help="Training run or parent directory; recursively discovers metrics.jsonl")
    parser.add_argument("--host", choices=("127.0.0.1", "localhost", "::1"), default="127.0.0.1",
                        help="Loopback only; use VS Code port forwarding for remote access")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--poll-seconds", type=_positive_seconds, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if importlib.util.find_spec("tensorboard") is None:
        print(
            "[monitor] TensorBoard is not installed in this Python environment.\n"
            f"Install: uv pip install --python {shlex.quote(sys.executable)} 'tensorboard==2.21.0'",
            file=sys.stderr,
        )
        return 2
    log_dir = args.logdir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    session_parent = log_dir / ".tensorboard_live"
    session_parent.mkdir(exist_ok=True)
    event_dir = Path(tempfile.mkdtemp(
        prefix=f"session-{datetime.now():%Y%m%d-%H%M%S}-", dir=session_parent,
    ))
    mirror = JournalMirror(log_dir, event_dir)
    process: subprocess.Popen | None = None
    try:
        imported = mirror.poll()
        print(f"[monitor] Source journals: {log_dir}", flush=True)
        print(f"[monitor] Fresh TensorBoard event session: {event_dir}", flush=True)
        print(f"[monitor] Imported {imported} history records from {len(mirror.latest)} run(s).", flush=True)
        process = subprocess.Popen([
            sys.executable, "-m", "tensorboard.main", "--logdir", str(event_dir),
            "--host", args.host, "--port", str(args.port), "--reload_interval", "5",
            # Zero is NOT unlimited for the generic scalar HTTP provider: it
            # returns an empty curve. Keep up to 100k displayed points per tag;
            # the complete, unmodified history remains in the source JSONL.
            "--load_fast", "false", "--samples_per_plugin", "scalars=100000",
        ])
        url_host = f"[{args.host}]" if ":" in args.host else args.host
        print(f"[monitor] Opening TensorBoard at http://{url_host}:{args.port} (wait for its ready message).", flush=True)
        print(
            "[monitor] Use Scalars / Time Series, horizontal axis Step.\n"
            "[monitor] Browser: Settings (gear) -> enable Reload data; set reload period (e.g. 30s).\n"
            "[monitor] Same action_loss tag overlays train, val, and comparison curves.\n"
            "[monitor] Write accuracy is a storage-classifier metric, NOT robot-task success.\n"
            "[monitor] Ctrl+C stops this monitor and TensorBoard only; training keeps running.",
            flush=True,
        )
        while True:
            if process.poll() is not None:
                print(f"[monitor] TensorBoard exited with status {process.returncode}; monitor stopped.", file=sys.stderr)
                return process.returncode or 0
            added = mirror.poll()
            if added:
                latest = ", ".join(f"{name}:{split}@{step}" for name, (step, split) in sorted(mirror.latest.items()))
                print(f"[monitor] {datetime.now():%H:%M:%S} +{added} record(s), total={mirror.total_records}; {latest}", flush=True)
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\n[monitor] Stopped. Training and source logs were left untouched.", flush=True)
        return 0
    except (OSError, ValueError) as exc:
        print(f"[monitor] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        mirror.close()


if __name__ == "__main__":
    raise SystemExit(main())
