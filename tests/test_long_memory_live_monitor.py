"""Live journal mirroring tests; no GPU or running training process required."""

import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from gr00t.long_memory.live_monitor import JournalMirror, TensorBoardScalarWriter, build_parser


class FakeWriter:
    def __init__(self, path):
        self.path = path
        self.scalars = []
        self.flushes = 0
        self.closed = False

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        self.flushes += 1

    def close(self):
        self.closed = True


class LiveMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.events = self.root / "events"
        self.writers = {}

        def factory(path):
            writer = FakeWriter(path)
            self.writers[path.relative_to(self.events).as_posix()] = writer
            return writer

        self.mirror = JournalMirror(self.logs, self.events, factory)

    def tearDown(self):
        self.mirror.close()
        self.temporary.cleanup()

    def append(self, run, record):
        path = self.logs / run / "metrics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return path

    def test_history_tail_new_runs_and_no_duplicates(self):
        self.append("stage1", {"step": 0, "split": "val", "action_loss": 1.2})
        self.assertEqual(self.mirror.poll(), 1)
        self.assertEqual(self.mirror.poll(), 0)
        self.append("stage1", {"step": 10, "split": "train", "action_loss": 0.8})
        self.append("stage2", {"step": 0, "split": "val", "action_loss": 0.7})
        self.assertEqual(self.mirror.poll(), 2)
        self.assertEqual(self.mirror.poll(), 0)
        self.assertEqual(self.mirror.total_records, 3)
        self.assertEqual(self.writers["stage1/val"].scalars, [("action_loss", 1.2, 0)])
        self.assertEqual(self.writers["stage1/train"].scalars, [("action_loss", 0.8, 10)])
        self.assertGreater(self.writers["stage1/val"].flushes, 1)

    def test_partial_final_line_waits_for_newline(self):
        path = self.logs / "metrics.jsonl"
        with path.open("wb") as handle:
            handle.write(b'{"step":1,"split":"train","action_loss":')
        self.assertEqual(self.mirror.poll(), 0)
        with path.open("ab") as handle:
            handle.write(b'0.25}')
        self.assertEqual(self.mirror.poll(), 0)
        with path.open("ab") as handle:
            handle.write(b'\n')
        self.assertEqual(self.mirror.poll(), 1)
        self.assertEqual(self.writers["train"].scalars, [("action_loss", 0.25, 1)])
        self.assertEqual(self.mirror.poll(), 0)

    def test_same_action_tag_for_baseline_and_wrong_memory(self):
        self.append("s1", {"step": 5, "split": "val", "action_loss": 0.2,
                           "baseline_action_loss": 0.3, "wrong_memory_action_loss": 0.4,
                           "write_precision": None, "utility_corr": None})
        self.mirror.poll()
        self.assertEqual(self.writers["s1/comparison/baseline"].scalars, [("action_loss", 0.3, 5)])
        self.assertEqual(self.writers["s1/comparison/wrong-memory"].scalars, [("action_loss", 0.4, 5)])
        original = self.writers["s1/val"].scalars
        self.assertIn(("baseline_action_loss", 0.3, 5), original)
        self.assertFalse(any(tag in {"write_precision", "utility_corr"} for tag, *_ in original))

    def test_invalid_completed_line_does_not_modify_source(self):
        path = self.logs / "metrics.jsonl"
        payload = b'{"step":0,"split":"train",broken}\n'
        path.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "Invalid completed JSONL record.*:1"):
            self.mirror.poll()
        self.assertEqual(path.read_bytes(), payload)
        self.assertEqual(self.mirror.total_records, 0)
        self.assertFalse(self.writers)

    def test_v2_policy_controls_overlay_without_renaming_original_metrics(self):
        controls = {"first_action_loss": "first-min-fill", "fifo_action_loss": "matched-fifo",
                    "random_action_loss": "matched-random", "all_action_loss": "all-fifo",
                    "stage1_all_action_loss": "frozen-stage1"}
        self.append("v2", {"step": 10, "split": "val", "action_loss": .25,
                           **{name: .3 for name in controls}})
        self.mirror.poll()
        for name, curve in controls.items():
            self.assertEqual(self.writers[f"v2/comparison/{curve}"].scalars, [("action_loss", .3, 10)])
            self.assertIn((name, .3, 10), self.writers["v2/val"].scalars)

    def test_nonfinite_and_invalid_split_rejected_before_any_event(self):
        for split, metric in (("train", float("nan")), ("../escape", 1.0), ("train", True)):
            with self.subTest(split=split, metric=metric):
                path = self.logs / "metrics.jsonl"
                path.write_text(json.dumps({"step": 1, "split": split, "loss": metric}) + "\n")
                with self.assertRaises(ValueError):
                    self.mirror.poll()
                self.assertFalse(self.writers)

    def test_truncation_and_replacement_require_explicit_restart(self):
        path = self.append("", {"step": 1, "split": "train", "action_loss": 1.0})
        self.mirror.poll()
        path.write_text("")
        with self.assertRaisesRegex(ValueError, "replaced or truncated"):
            self.mirror.poll()

    def test_replacement_detected_even_if_longer(self):
        path = self.append("", {"step": 1, "split": "train", "action_loss": 1.0})
        self.mirror.poll()
        replacement = self.logs / "replacement.jsonl"
        replacement.write_text(json.dumps({"step": 2, "split": "train", "action_loss": 1.0123456789}) + "\n")
        replacement.replace(path)
        with self.assertRaisesRegex(ValueError, "replaced or truncated"):
            self.mirror.poll()

    def test_old_tensorboard_sessions_not_scanned_and_restart_replays_once(self):
        self.append("stage1", {"step": 1, "split": "train", "loss": 1.0})
        self.append(".tensorboard_live/old", {"step": 99, "split": "train", "loss": 99.0})
        self.assertEqual(self.mirror.poll(), 1)
        second_writers = []

        def factory(path):
            writer = FakeWriter(path)
            second_writers.append(writer)
            return writer

        second = JournalMirror(self.logs, self.root / "fresh-events", factory)
        self.assertEqual(second.poll(), 1)
        self.assertEqual(second.poll(), 0)
        self.assertEqual(sum(len(writer.scalars) for writer in second_writers), 1)
        second.close()
        self.assertTrue(all(writer.closed for writer in second_writers))

    def test_localhost_only_and_positive_poll_interval(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args([]).host, "127.0.0.1")
        for argv in (["--host", "0.0.0.0"], ["--poll-seconds", "0"], ["--poll-seconds", "nan"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(argv)

    @unittest.skipUnless(importlib.util.find_spec("tensorboard"), "TensorBoard optional dependency not installed")
    def test_real_tensorboard_events_are_readable_after_flush(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        event_dir = self.root / "real-events"
        mirror = JournalMirror(self.logs, event_dir, TensorBoardScalarWriter)
        self.append("s1", {"step": 10, "split": "train", "action_loss": 0.5})
        mirror.poll()
        accumulator = EventAccumulator(str(event_dir / "s1/train"))
        accumulator.Reload()
        self.assertEqual([(event.step, event.value) for event in accumulator.Scalars("action_loss")], [(10, 0.5)])
        self.append("s1", {"step": 20, "split": "train", "action_loss": 0.25})
        mirror.poll()
        accumulator.Reload()
        self.assertEqual([(event.step, event.value) for event in accumulator.Scalars("action_loss")], [(10, 0.5), (20, 0.25)])
        mirror.close()

    @unittest.skipUnless(importlib.util.find_spec("tensorboard"), "TensorBoard optional dependency not installed")
    def test_cli_http_history_live_refresh_and_interrupt_cleanup(self):
        """Exercise the browser's HTTP path, not merely the event-file reader.

        CLI retention flags can make the scalars plugin return an empty list
        even when event files and the run/tag listing look entirely correct.
        This catches that integration regression and checks live tail refresh.
        """
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self.append("integration", {"step": 1, "split": "train", "action_loss": 1.0})
        script = Path(__file__).resolve().parents[1] / "run_scripts/robomme/monitor_long_memory.py"
        output_path = self.root / "monitor-output.txt"
        query = urlencode({"run": "integration/train", "tag": "action_loss"})
        url = f"http://127.0.0.1:{port}/data/plugin/scalars/scalars?{query}"

        def is_listening():
            with socket.socket() as connection:
                connection.settimeout(0.2)
                return connection.connect_ex(("127.0.0.1", port)) == 0

        with output_path.open("w") as output:
            process = subprocess.Popen([
                sys.executable, str(script), "--logdir", str(self.logs),
                "--port", str(port), "--poll-seconds", "0.1",
            ], stdout=output, stderr=subprocess.STDOUT, start_new_session=True)

            def wait_for_values(expected):
                deadline = time.monotonic() + 15
                observed = None
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail(f"Monitor exited early ({process.returncode}): {output_path.read_text()}")
                    try:
                        with urlopen(url, timeout=1) as response:
                            payload = json.load(response)
                        observed = [(point[1], point[2]) for point in payload]
                        if observed == expected:
                            return
                    except (URLError, OSError, ValueError):
                        pass  # Server/data loading is asynchronous at startup.
                    time.sleep(0.1)
                self.fail(f"HTTP scalar values did not update: {observed!r}, expected {expected!r}; {output_path.read_text()}")

            try:
                wait_for_values([(1, 1.0)])
                self.append("integration", {"step": 2, "split": "train", "action_loss": 0.5})
                wait_for_values([(1, 1.0), (2, 0.5)])
            finally:
                # Signal only the monitor first.  It must shut down its own
                # TensorBoard child, exactly as a separate-terminal Ctrl+C does.
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                deadline = time.monotonic() + 3
                while is_listening() and time.monotonic() < deadline:
                    time.sleep(0.1)
                if is_listening():
                    # Emergency cleanup is limited to the process group this
                    # test created, avoiding an orphan server after a failure.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.fail("TensorBoard child still listened after monitor Ctrl+C")
            self.assertEqual(process.returncode, 0, output_path.read_text())


if __name__ == "__main__":
    unittest.main()
