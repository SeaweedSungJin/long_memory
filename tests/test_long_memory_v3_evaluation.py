"""CPU-only v3 checkpoint, provenance, launch and paired-report regressions."""
import contextlib
import csv
from dataclasses import asdict
import fcntl
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from safetensors.torch import save_file
import torch

from gr00t.eval.sim.robomme.compare_long_memory_v3_results import build_v3_report
from gr00t.long_memory.checkpoint_v3 import memory_v3_checkpoint_info
from gr00t.long_memory.core_v3 import ActionValueMemory, MemoryV3Config
from gr00t.long_memory.hamlet import checkpoint_identity
from run_scripts.robomme import eval_long_memory_v3 as driver

ROOT = Path(__file__).resolve().parents[1]


class V3EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.config = MemoryV3Config(feature_dim=8, state_dim=4, action_dim=3,
                                    hidden_dim=8, num_heads=2, capacity=3, min_fill=1)
        config = {"hamlet_mode": "finetune", "mem_cond_type": "cross_attn", "memory_type": "moment_token",
                  "memory_stride": 16, "n_moment_tokens": 4, "backbone_embedding_dim": 8}
        (self.base / "config.json").write_text(json.dumps(config))
        (self.base / "processor_config.json").write_text(json.dumps(
            {"processor_kwargs": {"max_state_dim": 4, "max_action_dim": 3}}))
        (self.base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"test": "model.safetensors"}}))
        (self.base / "model.safetensors").write_bytes(b"fake base weights: not loaded in CPU tests")
        for name in ("statistics.json", "embodiment_id.json"):
            (self.base / name).write_text("{}")
        self.reader = self.make_checkpoint("reader", stage=1)
        self.memory = self.make_checkpoint("memory", stage=2)

    def tearDown(self):
        self.temp.cleanup()

    def make_checkpoint(self, name, *, stage, fingerprint="cache-A"):
        path = self.root / name
        path.mkdir()
        model = ActionValueMemory(self.config)
        save_file(model.state_dict(), str(path / "model.safetensors"))
        (path / "checkpoint.json").write_text(json.dumps({"format_version": 1, "step": 20,
            "config": {"trainer_variant": "action_value_v3", "stage": stage, "memory": asdict(self.config)},
            "metadata": {"base_model": checkpoint_identity(self.base), "cache_fingerprint": fingerprint}}))
        return path

    def update_info(self, path, change):
        target = path / "checkpoint.json"
        info = json.loads(target.read_text())
        change(info)
        target.write_text(json.dumps(info))

    def args(self, *extra):
        return driver.build_parser().parse_args(["--base-model", str(self.base),
            "--reader-checkpoint", str(self.reader), "--memory-checkpoint", str(self.memory),
            "--output-dir", str(self.root / "output"), *extra])

    def identity(self, args=None):
        args = args or self.args()
        driver.validate_options(args)
        with patch.object(driver, "benchmark_identity", return_value={"simulator": "fake"}), contextlib.redirect_stdout(io.StringIO()):
            return driver.build_identity(args)

    def test_valid_checkpoint_check_preserves_rng_and_base_files(self):
        before = {p.name: p.read_bytes() for p in self.base.iterdir()}
        rng = torch.random.get_rng_state().clone()
        info = memory_v3_checkpoint_info(self.base, self.reader, expected_stage=1)
        self.assertEqual(info["config"]["trainer_variant"], "action_value_v3")
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.base.iterdir()})

    def test_legacy_variant_and_wrong_stage_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Stage 1"):
            memory_v3_checkpoint_info(self.base, self.memory, expected_stage=1)
        self.update_info(self.reader, lambda x: x["config"].update(trainer_variant="legacy"))
        with self.assertRaisesRegex(ValueError, "legacy"):
            memory_v3_checkpoint_info(self.base, self.reader)

    def test_missing_cache_changed_base_and_wrong_dimensions_rejected(self):
        for mutation, message in (
            (lambda x: x["metadata"].update(cache_fingerprint=""), "cache_fingerprint"),
            (lambda x: x["metadata"].update(base_model={}), "different/changed"),
            (lambda x: x["config"]["memory"].update(state_dim=6), "state_dim"),
        ):
            original = (self.reader / "checkpoint.json").read_text()
            self.update_info(self.reader, mutation)
            with self.assertRaisesRegex(ValueError, message):
                memory_v3_checkpoint_info(self.base, self.reader)
            (self.reader / "checkpoint.json").write_text(original)

    def test_nonfinite_or_incomplete_addon_weights_rejected(self):
        state = ActionValueMemory(self.config).state_dict()
        first = next(iter(state))
        broken = dict(state)
        broken.pop(first)
        save_file(broken, str(self.reader / "model.safetensors"))
        with self.assertRaisesRegex(ValueError, "tensor names"):
            memory_v3_checkpoint_info(self.base, self.reader)
        state[first].flatten()[0] = float("nan")
        save_file(state, str(self.reader / "model.safetensors"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            memory_v3_checkpoint_info(self.base, self.reader)

    def test_identity_hashes_shared_and_v3_dependencies_and_actual_weights(self):
        identity = self.identity(self.args("--models", "baseline", "reader", "memory", "fifo"))
        sources = identity["source_sha256"]
        for name in ("gr00t/long_memory/core_v3.py", "gr00t/long_memory/online_v3.py",
                     "gr00t/long_memory/online_policy.py", "gr00t/long_memory/monitoring.py",
                     "gr00t/eval/sim/robomme/compare_long_memory_results.py",
                     "run_scripts/robomme/eval_long_memory_comparison.py"):
            self.assertIn(name, sources)
        self.assertEqual(identity["base_file_sha256"]["model.safetensors"], driver.file_hash(self.base / "model.safetensors"))
        self.assertEqual(identity["models"]["memory"]["weights_sha256"], identity["models"]["fifo"]["weights_sha256"])
        self.assertEqual(identity["models"]["memory"]["write_policy"], "hard")
        self.assertEqual(identity["models"]["fifo"]["write_policy_override"], "all")
        self.assertFalse((self.root / "output").exists())

    def test_mixed_cache_or_architecture_cohorts_rejected(self):
        self.update_info(self.memory, lambda x: x["metadata"].update(cache_fingerprint="cache-B"))
        with self.assertRaisesRegex(ValueError, "share the training cache"):
            self.identity()

    def test_fifo_launch_is_explicit_and_uses_new_server(self):
        identity = self.identity(self.args("--models", "baseline", "memory", "fifo"))
        command = driver.server_command(self.args(), identity["models"]["fifo"], 1234)
        self.assertIn(str(ROOT / "run_scripts/robomme/serve_long_memory_v3.py"), command)
        self.assertEqual(command[command.index("--memory-checkpoint") + 1], str(self.memory))
        self.assertEqual(command[command.index("--write-policy") + 1], "all")

    def test_preflight_creates_no_output_and_does_not_launch(self):
        arguments = ["--models", "baseline", "--base-model", str(self.base), "--preflight-only",
                     "--output-dir", str(self.root / "preflight"), "--server-python", sys.executable]
        with patch.object(driver, "check_dependencies"), patch.object(driver, "benchmark_identity", return_value={}), \
                patch.object(driver, "run_evaluation") as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(arguments), 0)
        launch.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())

    def test_server_failure_is_incomplete_and_saved_not_fake_zero(self):
        args = self.args("--models", "baseline", "--tasks", "BinFill", "--n-episodes", "1")
        identity = self.identity(args)
        with patch.object(driver.subprocess, "Popen") as popen, \
                patch.object(driver, "server_ready", side_effect=RuntimeError("load failure")), \
                patch.object(driver, "stop_process") as stop, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(driver.run_evaluation(args, identity, {}), 1)
        stop.assert_called_once_with(popen.return_value)
        folder = self.root / "output"
        report = json.loads((folder / "comparison_summary.json").read_text())
        task = report["models"]["baseline"]["tasks"]["BinFill"]
        self.assertIsNone(task["success_rate"])
        self.assertEqual(task["completed"], 0)
        self.assertIn("load failure", (folder / "driver_status.json").read_text())
        self.assertTrue((folder / "baseline/server.log").is_file())

    def report_fixture(self):
        folder = self.root / "report"
        folder.mkdir()
        manifest = {"trainer_variant": "action_value_v3", "evaluation_id": "example", "models":
                    {name: {} for name in ("baseline", "reader", "memory", "fifo")},
                    "settings": {"tasks": ["BinFill", "PatternLock"], "n_episodes": 3,
                    "dataset": "test", "seed": 6, "n_action_steps": 16, "max_episode_steps": 1300}}
        (folder / "comparison_manifest.json").write_text(json.dumps(manifest))
        for name, outcomes in (("baseline", [0, 0, 1]), ("reader", [0, 1, 1]),
                               ("memory", [1, 0, 1]), ("fifo", [0, 0, 1])):
            task_dir = folder / name / "BinFill"
            task_dir.mkdir(parents=True)
            (task_dir / "policy_manifest.json").write_text(json.dumps({"evaluation_id": "example:" + name,
                "task_id": "BinFill", **{k: v for k, v in manifest["settings"].items()
                                         if k in ("dataset", "seed", "n_action_steps", "max_episode_steps")}}))
            with (task_dir / "simulation_results.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["episode_idx", "episode_seed", "success", "status"])
                writer.writeheader()
                for episode, success in enumerate(outcomes):
                    writer.writerow({"episode_idx": episode, "episode_seed": episode + 100,
                                     "success": success, "status": "success" if success else "fail"})
        return folder

    def test_report_has_same_checkpoint_fifo_comparison_and_missing_task(self):
        result, text = build_v3_report(self.report_fixture(), bootstrap_samples=20)
        contrast = result["additional_comparisons"]["fifo_to_memory"]
        self.assertEqual((contrast["paired_n"], contrast["wins"], contrast["losses"]), (3, 1, 0))
        self.assertAlmostEqual(contrast["paired_task_macro_delta"], 1 / 3)
        self.assertIsNone(result["models"]["memory"]["tasks"]["PatternLock"]["success_rate"])
        self.assertFalse(contrast["complete"])
        self.assertIn("isolates storage policy", text)
        json.dumps(result, allow_nan=False)

    def test_diagnostics_use_final_completed_session_not_failed_attempt_or_cumulative_sum(self):
        folder = self.report_fixture()
        journal = folder / "memory/BinFill/memory_diagnostics.jsonl"
        records = []
        for sid, attempts in (("failed", 99), ("complete", 1), ("complete", 4)):
            records.append({"kind": "policy_call", "session_id": sid, "episode_idx": 0,
                            "info": {"long_memory": {"learned_attempted": attempts,
                            "learned_accepted": attempts, "learned_replaced": attempts}}})
        records.append({"kind": "episode_error", "session_id": "failed", "episode_idx": 0})
        records.append({"kind": "episode_complete", "session_id": "complete", "episode_idx": 0,
                        "episode_seed": 100, "success": 1})
        journal.write_text("".join(json.dumps(record) + "\n" for record in records) + '{"torn":')
        result, _ = build_v3_report(folder, bootstrap_samples=20)
        counters = result["storage_diagnostics"]["memory"]
        self.assertEqual(counters["learned_attempted"], 4)
        self.assertEqual(counters["learned_replaced"], 4)
        self.assertEqual(counters["completed_sessions_with_diagnostics"], 1)
        self.assertEqual(counters["completed_sessions_missing_diagnostics"], 2)
        self.assertEqual(counters["ignored_torn_final_lines"], 1)

    def test_report_only_needs_no_model_dependencies_and_legacy_is_rejected(self):
        folder = self.report_fixture()
        with patch.object(driver, "check_dependencies") as dependencies, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(["--report-only", "--output-dir", str(folder)]), 0)
        dependencies.assert_not_called()
        manifest = json.loads((folder / "comparison_manifest.json").read_text())
        manifest.pop("trainer_variant")
        (folder / "comparison_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Not a v3"):
            driver.main(["--report-only", "--output-dir", str(folder)])

    def test_report_only_does_not_race_live_driver(self):
        folder = self.report_fixture()
        with (folder / ".driver.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Evaluation is running"):
                driver.main(["--report-only", "--output-dir", str(folder)])

    def test_completed_resume_never_starts_policy_server(self):
        args = self.args("--models", "baseline", "--tasks", "BinFill", "--n-episodes", "1")
        identity = self.identity(args)
        output = self.root / "output"
        driver.bind_manifest(output, identity)
        task = output / "baseline/BinFill"
        task.mkdir(parents=True)
        (task / "simulation_results.csv").write_text("episode_idx,episode_seed,success\n0,100,0\n")
        (task / "policy_manifest.json").write_text(json.dumps({"evaluation_id": identity["evaluation_id"] + ":baseline",
            "task_id": "BinFill", **{key: identity["settings"][key]
                                     for key in ("dataset", "seed", "n_action_steps", "max_episode_steps")}}))
        with patch.object(driver.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.run_evaluation(args, identity, {}), 0)
        popen.assert_not_called()

    def test_parser_rejects_missing_roles_and_invalid_options(self):
        for command in (["--models", "memory"], ["--models", "baseline", "baseline"],
                        ["--models", "baseline", "--tasks", "Typo"],
                        ["--models", "baseline", "--n-episodes", "0"],
                        ["--models", "baseline", "reader"]):
            with self.assertRaises(ValueError):
                driver.validate_options(driver.build_parser().parse_args(command))

    def test_entrypoint_help_is_lightweight(self):
        for script in ("eval_long_memory_v3.py", "serve_long_memory_v3.py"):
            result = subprocess.run([sys.executable, str(ROOT / "run_scripts/robomme" / script), "--help"],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)


class V3PolicyWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse only tiny fake processor/expert fixtures, not their test classes.
        spec = importlib.util.spec_from_file_location("legacy_policy_fixtures", ROOT / "tests/test_long_memory_online_policy.py")
        cls.fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fixtures)

    def policy(self, *, memory=True, policy="hard"):
        from gr00t.long_memory.online_policy_v3 import LongMemoryV3Policy
        original = self.fixtures._policy(memory=False)
        result = LongMemoryV3Policy.__new__(LongMemoryV3Policy)
        result.__dict__.update(original.__dict__)
        if memory:
            result.memory = ActionValueMemory(MemoryV3Config(feature_dim=6, state_dim=8, action_dim=8,
                hidden_dim=8, num_heads=2, capacity=3, min_fill=1, residual_init=.01)).eval().requires_grad_(False)
            result.stage, result.write_policy = (2 if policy == "hard" else 1), policy
        return result

    def test_v3_adapter_preserves_prefix_and_completed_event_protocol(self):
        import numpy as np
        from gr00t.long_memory.online_v3 import OnlineActionValueBank
        policy = self.policy()
        _, first = policy.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
        self.assertEqual(first["long_memory"]["bank_fill"], 0)
        _, second = policy.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        self.assertIsInstance(policy.sessions["A"].bank, OnlineActionValueBank)
        self.assertEqual(second["long_memory"]["bank_fill"], 1)
        head = policy.model.action_head
        self.assertTrue(torch.equal(head.last_features[:, :-2], head.last_processed[:, :-2]))
        self.assertTrue(torch.equal(policy.sessions["A"].bank.previous.short, head.last_processed[0, -2:].float()))
        controls = np.zeros((2, 8), dtype=np.float32)
        _, third = policy.get_action(self.fixtures._observation(), self.fixtures._options(frame=4, controls=controls))
        self.assertEqual(third["long_memory"]["learned_write_attempts"], 1)
        json.dumps(third, allow_nan=False)
        self.assertEqual(policy.reset({"session_ids": ["A"]}), {"cleared_sessions": 1})
        self.assertFalse(policy.sessions)

    def test_baseline_v3_route_has_identical_features_and_noise(self):
        import numpy as np
        original, v3 = self.fixtures._policy(), self.policy(memory=False)
        for candidate in (original, v3):
            candidate.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
        actions_a, _ = original.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        actions_b, _ = v3.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        np.testing.assert_array_equal(actions_a["joint_position"], actions_b["joint_position"])
        self.assertTrue(torch.equal(original.model.action_head.last_features, v3.model.action_head.last_features))

    def test_session_capacity_and_seed_guard(self):
        policy = self.policy()
        policy.session_cap = 1
        policy._session("A", 17, False)
        policy._session("B", 18, False)
        self.assertEqual(list(policy.sessions), ["B"])
        with self.assertRaisesRegex(ValueError, "seed changed"):
            policy._session("B", 19, False)
        self.assertEqual(policy._session("B", 19, True).episode_seed, 19)


if __name__ == "__main__":
    unittest.main()
