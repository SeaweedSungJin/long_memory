"""CPU watcher contracts; no observed trainer or evaluation process is launched."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import select
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "run_scripts/robomme/watch_archive_control_v7.py"
SPEC = importlib.util.spec_from_file_location("archive_control_v7_watcher_test_module", SOURCE)
watcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = watcher
SPEC.loader.exec_module(watcher)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def snapshot_tree(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


class ArchiveControlWatcherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        root_patch = mock.patch.object(watcher, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.run = self.root / "runs/long_memory/archive_fixture"
        self.cache = self.root / "runs/long_memory/cache_fixture"
        self.base = self.root / "checkpoints/base_fixture"
        self.workflow = self.root / "runs/long_memory/archive_fixture_watcher"
        self.evaluation = self.root / "runs/eval/robomme/archive_fixture"
        self.run.mkdir(parents=True)
        self.cache.mkdir(parents=True)
        self.base.mkdir(parents=True)
        self.pid, self.starttime = 43210, 987654321
        self.config = {
            "trainer_variant": "recurrent_memory_v7", "stage": 1, "mode": "archive",
            "train": {
                "stage": 1, "mode": "archive", "cache_dir": str(self.cache),
                "output_dir": str(self.run), "max_epochs": 3, "max_steps": None,
                "stop_after_steps": None, "queries_per_prefix": 2,
                "query_batch_size": 2, "seed": 6, "device": "cuda:0",
                "init_checkpoint": None, "resume": None,
            },
        }
        self.argv = [
            str(self.root / ".venv/bin/python"),
            "run_scripts/robomme/train_long_memory_v7.py",
            "--stage", "1", "--mode", "archive",
            "--cache-dir", str(self.cache), "--output-dir", str(self.run),
            "--max-epochs", "3", "--queries-per-prefix", "2",
            "--query-batch-size", "2", "--seed", "6", "--device", "cuda:0",
        ]
        self.identity = {
            "pid": self.pid, "starttime": self.starttime, "cwd": str(self.root),
            "argv": self.argv.copy(), "state": "S",
        }

    def valid_plan(self):
        plans = {
            "train": [[17, [0, 1]]], "val": [[18, [0]]], "train_query_count": 2,
            "validation": [[18, 0]], "epoch_validation": [], "files": {},
            "provenance_groups": {},
            "pairing": "chronologically_adjacent_queries_then_global_group_shuffle",
        }
        windows = [
            {"epoch": epoch, "groups": [[17, [0, 1]]], "query_count": 2, "epoch_end": True}
            for epoch in range(3)
        ]
        body = {"plans": plans, "windows": windows}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return {"sha256": digest, **body}

    def save_plan(self, plan=None, *, update_hash=False):
        plan = copy.deepcopy(plan if plan is not None else self.valid_plan())
        if update_hash:
            payload = {"plans": plan["plans"], "windows": plan["windows"]}
            plan["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        write_json(self.run / "query_plan.json", plan)
        return plan

    def completed_run(self, best_step=2):
        plan = self.save_plan()
        coverage = watcher.coverage_plan(self.run, self.config)
        source = self.root / "gr00t/frozen_source.py"
        source.parent.mkdir()
        source.write_text("# Fixture frozen training source.\n")
        provenance = {
            "base_model": {"path": str(self.base)}, "cache_fingerprint": "fixture-cache-id",
            "cache_dir": str(self.cache), "plan_sha256": plan["sha256"],
            "source_sha256": {str(source.relative_to(self.root)): hashlib.sha256(source.read_bytes()).hexdigest()},
        }
        status = {
            "status": "complete", "step": 3, "window_cursor": 3,
            "optimizer_updates": 3, "processed_queries": 6, "epoch": 3,
        }
        write_json(self.run / "run_config.json", self.config)
        write_json(self.run / "provenance.json", provenance)
        write_json(self.run / "status.json", status)
        for step in range(4):
            state = dict(status, step=step, window_cursor=step, optimizer_updates=step,
                         processed_queries=2 * step, epoch=step,
                         status="complete" if step == 3 else "training")
            write_json(self.run / f"checkpoint-{step:06d}" / "checkpoint.json", {
                "format_version": 1, "step": step, "config": self.config,
                "metadata": dict(provenance, train_state=state),
            })
        write_json(self.run / "last_checkpoint.json", {"path": "checkpoint-000003", "step": 3})
        write_json(self.run / "best_checkpoint.json", {"path": f"checkpoint-{best_step:06d}", "step": best_step})
        manifest = {"model_path": str(self.base), "fingerprint": "fixture-cache-id",
                    "dataset_path": str(self.root / "dataset")}
        write_json(self.cache / "manifest.json", manifest)
        for name in ("eval_long_memory_v7.py", "run_long_memory_v8_workflow.py"):
            path = self.root / "run_scripts/robomme" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Fixture frozen evaluation source.\n")
        return coverage, provenance, manifest

    def main_args(self):
        return [
            "--trainer-pid", str(self.pid), "--expected-starttime", str(self.starttime),
            "--run-dir", str(self.run), "--cache-dir", str(self.cache),
            "--workflow-dir", str(self.workflow), "--eval-dir", str(self.evaluation),
            "--include-epoch1", "--poll-seconds", "0.01", "--gpu", "1",
        ]

    def test_proc_identity_handles_parentheses_in_comm_and_exact_null_separated_argv(self):
        proc_root = self.root / "proc"
        process = proc_root / str(self.pid)
        process.mkdir(parents=True)
        fields = ["S", *(["0"] * 18), str(self.starttime), *(["0"] * 5)]
        (process / "stat").write_text(f"{self.pid} (python (archive worker)) {' '.join(fields)}\n")
        (process / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in self.argv) + b"\0")
        (process / "cwd").symlink_to(self.root, target_is_directory=True)
        actual = watcher.read_process_identity(self.pid, proc_root=proc_root)
        for key in ("pid", "starttime", "argv", "state"):
            self.assertEqual(actual[key], self.identity[key], key)
        self.assertEqual(Path(actual["cwd"]), self.root)

    def test_pidfd_wait_uses_exit_readiness_and_heartbeats_without_signalling(self):
        poller = mock.Mock()
        poller.poll.side_effect = [[], [(73, select.POLLIN)]]
        heartbeat = mock.Mock()
        with mock.patch.object(watcher.select, "poll", return_value=poller), \
                mock.patch.object(watcher.os, "kill", side_effect=AssertionError("must not signal trainer")):
            watcher.wait_for_exit(73, poll_seconds=0.01, heartbeat=heartbeat)
        self.assertEqual(poller.poll.call_count, 2)
        self.assertTrue(poller.register.called)
        self.assertEqual(poller.register.call_args.args[0], 73)
        self.assertGreaterEqual(heartbeat.call_count, 1)

    def test_identity_validation_rejects_reused_pid_wrong_command_and_wrong_run(self):
        watcher.validate_process_identity(
            self.identity, starttime=self.starttime, run=self.run, cache=self.cache,
            config=self.config,
        )
        mutations = [
            lambda identity: identity.update(starttime=self.starttime + 1),
            lambda identity: identity.update(cwd=str(self.root / "elsewhere")),
            lambda identity: identity["argv"].__setitem__(1, "run_scripts/robomme/train_long_memory_v8.py"),
            lambda identity: identity["argv"].__setitem__(identity["argv"].index("--mode") + 1, "recurrent"),
            lambda identity: identity["argv"].__setitem__(identity["argv"].index("--output-dir") + 1, str(self.run.with_name("other"))),
            lambda identity: identity["argv"].__setitem__(identity["argv"].index("--cache-dir") + 1, str(self.cache.with_name("other"))),
            lambda identity: identity["argv"].extend(["--resume", str(self.run / "checkpoint-000001")]),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(self.identity)
            mutate(changed)
            with self.subTest(identity=changed):
                with self.assertRaises((ValueError, RuntimeError)):
                    watcher.validate_process_identity(
                        changed, starttime=self.starttime, run=self.run, cache=self.cache,
                        config=self.config,
                    )

    def test_bind_rechecks_identity_after_opening_pidfd_and_closes_on_reuse(self):
        reused = copy.deepcopy(self.identity)
        reused["starttime"] += 1
        with mock.patch.object(watcher, "read_process_identity", side_effect=[self.identity, reused]), \
                mock.patch.object(watcher.os, "pidfd_open", return_value=73, create=True) as opened, \
                mock.patch.object(watcher.os, "close") as closed:
            with self.assertRaises((ValueError, RuntimeError)):
                watcher.bind_trainer(self.pid, self.starttime, self.run, self.cache, self.config)
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(opened.call_args.args[0], self.pid)
        closed.assert_called_once_with(73)

    def test_bind_tracks_same_process_across_scheduler_state_changes(self):
        running = copy.deepcopy(self.identity)
        running["state"] = "R"
        with mock.patch.object(watcher, "read_process_identity", side_effect=[self.identity, running]), \
                mock.patch.object(watcher.os, "pidfd_open", return_value=73, create=True), \
                mock.patch.object(watcher.os, "close") as closed:
            fd, identity = watcher.bind_trainer(self.pid, self.starttime, self.run, self.cache, self.config)
        self.assertEqual(fd, 73)
        self.assertEqual(identity["starttime"], self.starttime)
        closed.assert_not_called()

    def test_libc_pidfd_fallback_propagates_esrch_without_closing_invalid_fd(self):
        libc = SimpleNamespace(pidfd_open=mock.Mock(return_value=-1))
        os_without_pidfd = SimpleNamespace(strerror=watcher.os.strerror, close=mock.Mock())
        with mock.patch.object(watcher, "os", os_without_pidfd), \
                mock.patch.object(watcher.ctypes, "CDLL", return_value=libc), \
                mock.patch.object(watcher.ctypes, "get_errno", return_value=3):
            with self.assertRaises(ProcessLookupError) as caught:
                watcher.open_pidfd(self.pid)
        self.assertEqual(caught.exception.errno, 3)
        libc.pidfd_open.assert_called_once_with(self.pid, 0)
        os_without_pidfd.close.assert_not_called()

    def test_pidfd_errors_cannot_release_evaluation_wait(self):
        for flags in (select.POLLERR, select.POLLNVAL):
            poller = mock.Mock()
            poller.poll.return_value = [(73, flags)]
            with self.subTest(flags=flags), mock.patch.object(watcher.select, "poll", return_value=poller):
                with self.assertRaises(RuntimeError):
                    watcher.wait_for_exit(73, poll_seconds=0.01)

    def test_coverage_plan_counts_updates_queries_epochs_and_epoch_one(self):
        plan = self.save_plan()
        actual = watcher.coverage_plan(self.run, self.config)
        self.assertEqual(actual["step"], 3)
        self.assertEqual(actual["processed_queries"], 6)
        self.assertEqual(actual["epoch"], 3)
        self.assertEqual(actual["epoch1_step"], 1)
        self.assertEqual(actual["plan_sha256"], plan["sha256"])

    def test_coverage_plan_rejects_digest_changes_missing_queries_and_incomplete_epochs(self):
        changed_hash = self.valid_plan()
        changed_hash["sha256"] = "0" * 64
        self.save_plan(changed_hash)
        with self.assertRaises((ValueError, RuntimeError)):
            watcher.coverage_plan(self.run, self.config)
        mutations = [
            lambda plan: plan["windows"][0].update(query_count=1),
            lambda plan: plan["windows"][0]["groups"][0].__setitem__(1, [0, 0]),
            lambda plan: plan["windows"][0].update(epoch_end=False),
            lambda plan: plan["windows"].pop(),
        ]
        for mutate in mutations:
            plan = self.valid_plan()
            mutate(plan)
            self.save_plan(plan, update_hash=True)
            with self.subTest(plan=plan):
                with self.assertRaises((ValueError, RuntimeError)):
                    watcher.coverage_plan(self.run, self.config)

    def test_complete_run_selects_epoch_one_before_last_and_distinct_best(self):
        coverage, provenance, _ = self.completed_run()
        selected = watcher.verify_completion(
            self.run, self.config, coverage, provenance, include_epoch1=True,
        )
        self.assertEqual(list(selected), ["epoch1", "last", "best"])
        self.assertEqual(list(selected.values()), [
            self.run / "checkpoint-000001", self.run / "checkpoint-000003",
            self.run / "checkpoint-000002",
        ])

    def test_zero_best_and_same_checkpoint_are_not_evaluated_twice(self):
        coverage, provenance, _ = self.completed_run()
        for step in (0, 1, 3):
            write_json(self.run / "best_checkpoint.json", {"path": f"checkpoint-{step:06d}", "step": step})
            with self.subTest(best_step=step):
                selected = watcher.verify_completion(
                    self.run, self.config, coverage, provenance, include_epoch1=True,
                )
                self.assertEqual(list(selected), ["epoch1", "last"])

    def test_completion_requires_status_and_every_exact_coverage_counter(self):
        coverage, provenance, _ = self.completed_run()
        valid = json.loads((self.run / "status.json").read_text())
        for key, value in (("status", "training"), ("status", "paused"), ("step", 2),
                           ("window_cursor", 2), ("optimizer_updates", 2),
                           ("processed_queries", 5), ("epoch", 2)):
            write_json(self.run / "status.json", dict(valid, **{key: value}))
            with self.subTest(key=key, value=value):
                with self.assertRaises(RuntimeError):
                    watcher.verify_completion(self.run, self.config, coverage, provenance)

    def test_completion_rejects_mismatched_checkpoint_provenance_and_coverage(self):
        coverage, provenance, _ = self.completed_run()
        path = self.run / "checkpoint-000003/checkpoint.json"
        valid = json.loads(path.read_text())
        mutations = [
            lambda info: info["config"].update(mode="recurrent"),
            lambda info: info["metadata"].update(cache_fingerprint="other-cache"),
            lambda info: info["metadata"].update(plan_sha256="0" * 64),
            lambda info: info["metadata"]["train_state"].update(processed_queries=5),
            lambda info: info["metadata"]["train_state"].update(window_cursor=2),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(valid)
            mutate(changed)
            write_json(path, changed)
            with self.subTest(checkpoint=changed):
                with self.assertRaises(ValueError):
                    watcher.verify_completion(self.run, self.config, coverage, provenance)

    def test_epoch_one_and_checkpoint_paths_must_match_the_pinned_run(self):
        coverage, provenance, _ = self.completed_run()
        path = self.run / "checkpoint-000001/checkpoint.json"
        first = json.loads(path.read_text())
        first["metadata"]["train_state"]["epoch"] = 0
        write_json(path, first)
        with self.assertRaises(ValueError):
            watcher.verify_completion(self.run, self.config, coverage, provenance, include_epoch1=True)
        write_json(self.run / "last_checkpoint.json", {"path": "../other/checkpoint-000003", "step": 3})
        with self.assertRaises(ValueError):
            watcher.verify_completion(self.run, self.config, coverage, provenance)

    def test_main_waits_for_pidfd_then_preflights_every_candidate_before_rollouts(self):
        _, _, manifest = self.completed_run()
        original_training = snapshot_tree(self.run)
        original_cache = snapshot_tree(self.cache)
        calls = []

        def wait_for_exit(fd, poll_seconds, heartbeat):
            self.assertEqual(fd, 73)
            self.assertEqual(calls, [])
            self.assertFalse(self.evaluation.exists())
            heartbeat()

        def run_step(name, command, workflow):
            self.assertTrue((workflow / "trainer_exit.json").is_file())
            self.assertFalse(any("train_long_memory" in argument for argument in command))
            calls.append((name, command))

        with mock.patch.object(watcher, "EpisodeCache", return_value=SimpleNamespace(manifest=manifest)), \
                mock.patch.object(watcher, "validate_cache_checkpoint"), \
                mock.patch.object(watcher, "bind_trainer", return_value=(73, self.identity)), \
                mock.patch.object(watcher, "wait_for_exit", side_effect=wait_for_exit), \
                mock.patch.object(watcher, "run_step", side_effect=run_step), \
                mock.patch.object(watcher.os, "close") as closed, \
                mock.patch.object(watcher.os, "kill", side_effect=AssertionError("must not signal trainer")), \
                mock.patch("builtins.print"):
            self.assertEqual(watcher.main(self.main_args()), 0)
        self.assertEqual([name for name, _ in calls], [
            "epoch1_eval_preflight", "last_eval_preflight", "best_eval_preflight",
            "epoch1_eval", "last_eval", "best_eval",
        ])
        self.assertEqual(snapshot_tree(self.run), original_training)
        self.assertEqual(snapshot_tree(self.cache), original_cache)
        closed.assert_called_once_with(73)
        status = json.loads((self.workflow / "status.json").read_text())
        self.assertEqual(status["status"], "iteration_complete")
        self.assertFalse(status["goal_achieved"])

    def test_main_preflight_and_identity_failure_do_not_write_or_evaluate(self):
        _, _, manifest = self.completed_run()
        original = snapshot_tree(self.root)
        with mock.patch.object(watcher, "EpisodeCache", return_value=SimpleNamespace(manifest=manifest)), \
                mock.patch.object(watcher, "validate_cache_checkpoint"), \
                mock.patch.object(watcher, "bind_trainer", return_value=(73, self.identity)) as bind, \
                mock.patch.object(watcher, "wait_for_exit") as wait, \
                mock.patch.object(watcher, "run_step") as run_step, \
                mock.patch.object(watcher.os, "close") as close, mock.patch("builtins.print"):
            self.assertEqual(watcher.main(self.main_args() + ["--preflight-only"]), 0)
            close.assert_called_once_with(73)
            self.assertEqual(snapshot_tree(self.root), original)
            bind.side_effect = ValueError("reused PID")
            with self.assertRaisesRegex(ValueError, "reused PID"):
                watcher.main(self.main_args())
            wait.assert_not_called()
            run_step.assert_not_called()
        self.assertFalse(self.workflow.exists())
        self.assertFalse(self.evaluation.exists())
        self.assertEqual(snapshot_tree(self.root), original)

    def test_main_completed_status_never_bypasses_failed_pidfd_wait(self):
        _, _, manifest = self.completed_run()
        original_training = snapshot_tree(self.run)
        with mock.patch.object(watcher, "EpisodeCache", return_value=SimpleNamespace(manifest=manifest)), \
                mock.patch.object(watcher, "validate_cache_checkpoint"), \
                mock.patch.object(watcher, "bind_trainer", return_value=(73, self.identity)), \
                mock.patch.object(watcher, "wait_for_exit", side_effect=RuntimeError("pidfd wait failed")), \
                mock.patch.object(watcher, "run_step") as run_step, \
                mock.patch.object(watcher.os, "close") as close, mock.patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "pidfd wait failed"):
                watcher.main(self.main_args())
            run_step.assert_not_called()
            close.assert_called_once_with(73)
        self.assertEqual(snapshot_tree(self.run), original_training)
        self.assertFalse(self.evaluation.exists())
        status = json.loads((self.workflow / "status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertFalse(status["training_writes"])
        self.assertFalse(status["training_signals"])

    def test_main_incomplete_run_after_exit_cannot_start_evaluation(self):
        _, _, manifest = self.completed_run()
        status = json.loads((self.run / "status.json").read_text())
        write_json(self.run / "status.json", dict(status, status="paused"))
        original_training = snapshot_tree(self.run)
        with mock.patch.object(watcher, "EpisodeCache", return_value=SimpleNamespace(manifest=manifest)), \
                mock.patch.object(watcher, "validate_cache_checkpoint"), \
                mock.patch.object(watcher, "bind_trainer", return_value=(73, self.identity)), \
                mock.patch.object(watcher, "wait_for_exit") as wait, \
                mock.patch.object(watcher, "run_step") as run_step, \
                mock.patch.object(watcher.os, "close"), mock.patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "complete"):
                watcher.main(self.main_args())
            wait.assert_called_once()
            run_step.assert_not_called()
        self.assertFalse(self.evaluation.exists())
        self.assertEqual(snapshot_tree(self.run), original_training)

    def test_main_output_guard_runs_before_pid_binding_or_any_writes(self):
        _, _, manifest = self.completed_run()
        self.workflow.mkdir()
        (self.workflow / "existing-record.txt").write_text("Keep existing workflow.\n")
        original = snapshot_tree(self.root)
        with mock.patch.object(watcher, "EpisodeCache", return_value=SimpleNamespace(manifest=manifest)), \
                mock.patch.object(watcher, "validate_cache_checkpoint"), \
                mock.patch.object(watcher, "bind_trainer") as bind, \
                mock.patch.object(watcher, "run_step") as run_step:
            with self.assertRaises(FileExistsError):
                watcher.main(self.main_args())
            bind.assert_not_called()
            run_step.assert_not_called()
        self.assertEqual(snapshot_tree(self.root), original)

    def test_output_guard_accepts_new_scoped_paths_without_creating_them(self):
        before = snapshot_tree(self.run)
        watcher.validate_outputs(self.workflow, self.evaluation, self.run, self.cache, self.base)
        self.assertFalse(self.workflow.exists())
        self.assertFalse(self.evaluation.exists())
        self.assertEqual(snapshot_tree(self.run), before)

    def test_output_guard_rejects_existing_external_and_overlapping_paths(self):
        existing = self.workflow.with_name("existing_watcher")
        existing.mkdir()
        bad_pairs = (
            (existing, self.evaluation),
            (self.root / "outside_runs", self.evaluation),
            (self.workflow, self.root / "outside_eval"),
            (self.run / "watcher", self.evaluation),
            (self.workflow, self.cache / "evaluation"),
            (self.workflow, self.workflow / "evaluation"),
        )
        for workflow, evaluation in bad_pairs:
            with self.subTest(workflow=workflow, evaluation=evaluation):
                with self.assertRaises((ValueError, FileExistsError)):
                    watcher.validate_outputs(workflow, evaluation, self.run, self.cache, self.base)

    def test_eval_command_is_fixed_validation_archive_comparison_on_visible_gpu(self):
        checkpoint = self.run / "checkpoint-000006"
        command = watcher.eval_command(checkpoint, self.evaluation, self.base, gpu="1")
        self.assertIsInstance(command, list)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", command)
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")
        self.assertEqual(command[command.index("--dataset") + 1], "val")
        self.assertEqual(command[command.index("--n-episodes") + 1], "10")
        self.assertEqual(command[command.index("--seed") + 1], "6")
        self.assertEqual(command[command.index("--tasks") + 1], "all")
        self.assertEqual(command[command.index("--archive-checkpoint") + 1], str(checkpoint))
        self.assertEqual(command[command.index("--base-model") + 1], str(self.base))
        models_begin = command.index("--models") + 1
        models_end = next((i for i in range(models_begin, len(command)) if command[i].startswith("--")), len(command))
        self.assertEqual(command[models_begin:models_end], ["baseline", "archive"])
        self.assertFalse(any("train_long_memory" in argument for argument in command))


if __name__ == "__main__":
    unittest.main()
