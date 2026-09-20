"""Small CPU-only tests for paired result accounting and launch safety."""
import csv
import contextlib
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gr00t.eval.sim.robomme.compare_long_memory_results import (
    TASKS, build_report, mcnemar_exact, paired_differences, paired_macro_bootstrap, read_results,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("comparison_driver", ROOT / "run_scripts/robomme/eval_long_memory_comparison.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def csv(self, model, task, outcomes, *, seed=123, status="success"):
        path = self.root / model / task / "simulation_results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "policy_manifest.json").write_text(json.dumps({"evaluation_id": "test:" + model,
            "task_id": task, "dataset": "test", "seed": 6, "n_action_steps": 16, "max_episode_steps": 1300}))
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["episode_idx", "episode_seed", "scenario_seed", "success", "status", "task_instruction"])
            writer.writeheader()
            for index, success in enumerate(outcomes):
                writer.writerow({"episode_idx": index, "episode_seed": seed + index, "scenario_seed": index,
                                 "success": success, "status": status, "task_instruction": "fill the bin"})
        return path

    def manifest(self, tasks=("BinFill",), n=3):
        result = {"evaluation_id": "test", "settings": {"tasks": list(tasks), "n_episodes": n,
                  "dataset": "test", "seed": 6, "n_action_steps": 16, "max_episode_steps": 1300},
                  "models": {"baseline": {}, "stage2": {}}}
        (self.root / "comparison_manifest.json").write_text(json.dumps(result))

    def test_task_failure_is_zero_but_missing_is_absent(self):
        self.manifest(tasks=("BinFill", "PatternLock"))
        self.csv("baseline", "BinFill", [0, 0, 0], status="step_limit")
        self.csv("stage2", "BinFill", [0, 0, 0], status="fail")
        result, text = build_report(self.root, bootstrap_samples=20)
        self.assertEqual(result["models"]["baseline"]["tasks"]["BinFill"]["success_rate"], 0)
        self.assertIsNone(result["models"]["baseline"]["tasks"]["PatternLock"]["success_rate"])
        self.assertEqual(result["comparisons"]["stage2"]["tasks"]["BinFill"]["paired_n"], 3)
        self.assertEqual(result["comparisons"]["stage2"]["tasks"]["BinFill"]["same"], 3)
        self.assertFalse(result["comparisons"]["stage2"]["complete"])
        self.assertIn("INCOMPLETE", text)
        json.dumps(result, allow_nan=False)

    def test_partial_csv_pairs_only_actual_matching_episodes(self):
        self.manifest()
        self.csv("baseline", "BinFill", [1, 0, 0])
        self.csv("stage2", "BinFill", [0, 1])
        result, _ = build_report(self.root, bootstrap_samples=20)
        comparison = result["comparisons"]["stage2"]
        self.assertEqual((comparison["paired_n"], comparison["wins"], comparison["losses"], comparison["same"]), (2, 1, 1, 0))
        self.assertEqual(comparison["paired_task_macro_delta"], 0)
        self.assertFalse(comparison["complete"])

    def test_mismatched_seed_or_instruction_rejected(self):
        left = {0: {"episode_seed": 6, "success": 0, "task_instruction": "a"}}
        right = {0: {"episode_seed": 7, "success": 1, "task_instruction": "a"}}
        with self.assertRaisesRegex(ValueError, "episode_seed"):
            paired_differences(left, right)
        right[0]["episode_seed"] = 6
        right[0]["task_instruction"] = "b"
        with self.assertRaisesRegex(ValueError, "task_instruction"):
            paired_differences(left, right)

    def test_copied_csv_identity_or_changed_scenario_rejected(self):
        self.manifest()
        self.csv("baseline", "BinFill", [0])
        self.csv("stage2", "BinFill", [1])
        path = self.root / "stage2/BinFill/policy_manifest.json"
        data = json.loads(path.read_text())
        data["evaluation_id"] = "another-evaluation"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "identity"):
            build_report(self.root, bootstrap_samples=20)
        data["evaluation_id"] = "test:stage2"
        data["scenario_metadata_sha256"] = "different-scenarios"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "scenario_metadata"):
            build_report(self.root, bootstrap_samples=20)

    def test_duplicate_invalid_and_error_rows_rejected(self):
        path = self.csv("baseline", "BinFill", [0], status="error")
        with self.assertRaisesRegex(ValueError, "unfinished/error"):
            read_results(path)
        path = self.csv("baseline", "BinFill", [2])
        with self.assertRaisesRegex(ValueError, "success must"):
            read_results(path)
        path = self.csv("baseline", "BinFill", [0])
        with path.open("a") as handle:
            handle.write("0,123,0,0,success,fill the bin\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_results(path)

    def test_out_of_requested_range_rejected(self):
        path = self.csv("baseline", "BinFill", [0, 1])
        with self.assertRaisesRegex(ValueError, "unexpected episode"):
            read_results(path, expected=1)

    def test_mcnemar_and_bootstrap(self):
        self.assertEqual(mcnemar_exact(0, 0), 1)
        self.assertEqual(mcnemar_exact(1, 1), 1)
        self.assertEqual(mcnemar_exact(10, 0), 2 / 1024)
        self.assertEqual(paired_macro_bootstrap([[0, 0], [0]], samples=30), [0, 0])
        self.assertEqual(paired_macro_bootstrap([[1, 1]], samples=30), [1, 1])
        self.assertIsNone(paired_macro_bootstrap([], samples=30))

    def test_macro_equal_weights_not_episode_pooled(self):
        self.manifest(tasks=("BinFill", "PatternLock"))
        self.csv("baseline", "BinFill", [0, 0, 0])
        self.csv("stage2", "BinFill", [1, 1, 1])
        self.csv("baseline", "PatternLock", [1])
        self.csv("stage2", "PatternLock", [0])
        result, _ = build_report(self.root, bootstrap_samples=20)
        self.assertEqual(result["comparisons"]["stage2"]["paired_task_macro_delta"], 0)
        self.assertIsNone(result["comparisons"]["stage2"]["paired_task_macro_bootstrap_ci95"])

    def test_manifest_refuses_changed_identity_and_unowned_data(self):
        output = self.root / "run"
        driver.bind_manifest(output, {"checkpoint": "a"})
        driver.bind_manifest(output, {"checkpoint": "a"})
        with self.assertRaisesRegex(ValueError, "NEW"):
            driver.bind_manifest(output, {"checkpoint": "b"})
        unrelated = self.root / "unowned"
        unrelated.mkdir()
        (unrelated / "results.csv").write_text("user data")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            driver.bind_manifest(unrelated, {})
        self.assertEqual((unrelated / "results.csv").read_text(), "user data")

    def test_python_path_does_not_dereference_venv_symlink(self):
        symlink = self.root / "python"
        symlink.symlink_to(sys.executable)
        self.assertEqual(driver.python_path(symlink), symlink)
        self.assertNotEqual(driver.python_path(symlink), symlink.resolve())

    def test_parser_task_and_model_validation(self):
        args = driver.build_parser().parse_args(["--tasks", "all"])
        driver.validate_options(args)
        self.assertEqual(args.tasks, TASKS)
        for command in (["--models", "stage2"], ["--tasks", "Typo"], ["--n-episodes", "0"],
                        ["--models", "baseline", "baseline"], ["--tasks", "BinFill", "BinFill"]):
            with self.assertRaises(ValueError):
                driver.validate_options(driver.build_parser().parse_args(command))

    def test_owns_only_spawned_process_group(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            driver.stop_process(process)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_help_is_lightweight(self):
        result = subprocess.run([sys.executable, str(ROOT / "run_scripts/robomme/eval_long_memory_comparison.py"), "--help"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--preflight-only", result.stdout)

    def test_complete_resume_does_not_start_server(self):
        self.manifest()
        self.csv("baseline", "BinFill", [0, 1, 0])
        self.csv("stage2", "BinFill", [0, 1, 1])
        identity = json.loads((self.root / "comparison_manifest.json").read_text())
        args = driver.build_parser().parse_args(["--output-dir", str(self.root), "--tasks", "BinFill"])
        with patch.object(driver.subprocess, "Popen", side_effect=AssertionError("Completed runs must not load models")):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(driver.run_evaluation(args, identity, {}), 0)
        result = json.loads((self.root / "comparison_summary.json").read_text())
        self.assertEqual(result["comparisons"]["stage2"]["wins"], 1)

    def test_concurrent_output_writer_rejected(self):
        self.manifest()
        identity = json.loads((self.root / "comparison_manifest.json").read_text())
        args = driver.build_parser().parse_args(["--output-dir", str(self.root), "--tasks", "BinFill"])
        with (self.root / ".driver.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                driver.run_evaluation(args, identity, {})

    def test_server_exited_is_not_readiness(self):
        class Exited:
            returncode = 1

            def poll(self):
                return 1

        with self.assertRaisesRegex(RuntimeError, "exited"):
            driver.server_ready(Exited(), driver.free_local_port(), .1)


if __name__ == "__main__":
    unittest.main()
