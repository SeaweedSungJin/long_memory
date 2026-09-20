"""CPU-only tests: fair training inputs, direct pairing, and no preflight jobs."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_scripts.robomme import eval_long_memory_v5_controls as driver


class ControlsV5Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def training_pair(self):
        pair = []
        for role, weights in (("reader", (0.001, 0.01)), ("control", (0., 0.))):
            output = self.root / role
            output.mkdir()
            (output / "sampling_plan.json").write_text(json.dumps({"validation": [[1, 2]]}))
            pair.append({"config": {"memory": {}, "expert": {}, "expert_targets": [], "recall": {},
                "train": {"reader_mode": "memory", "seed": 42, "max_steps": 2000,
                          "output_dir": str(output), "subgoal_weight": weights[0], "grounding_weight": weights[1]}},
                "metadata": {k: "same" for k in ("cache_fingerprint", "recall_labels_fingerprint",
                                                 "source_sha256", "initial_checkpoint_sha256")}})
        return pair

    def test_matched_pair_allowed_but_different_budget_rejected(self):
        reader, control = self.training_pair()
        driver.validate_training_pair(reader, control)
        control["config"]["train"]["max_steps"] = 1000
        with self.assertRaisesRegex(ValueError, "Training settings"):
            driver.validate_training_pair(reader, control)

    def test_control_auxiliary_or_different_plan_rejected(self):
        reader, control = self.training_pair()
        bad = copy.deepcopy(control)
        bad["config"]["train"]["grounding_weight"] = 0.01
        with self.assertRaisesRegex(ValueError, "both weights zero"):
            driver.validate_training_pair(reader, bad)
        path = Path(control["config"]["train"]["output_dir"]) / "sampling_plan.json"
        path.write_text(json.dumps({"validation": [[9, 9]]}))
        with self.assertRaisesRegex(ValueError, "sampling plans"):
            driver.validate_training_pair(reader, control)

    def manifest(self):
        data = {"evaluation_id": "test", "trainer_variant": "action_expert_v4", "experiment": driver.EXPERIMENT,
                "settings": {"tasks": ["BinFill"], "n_episodes": 3, "dataset": "val", "seed": 6,
                             "n_action_steps": 16, "max_episode_steps": 1300},
                "models": {role: {"reader_mode": "memory"} for role in ("baseline", "action_control", "reader")}}
        (self.root / "comparison_manifest.json").write_text(json.dumps(data))

    def results(self, role, outcomes, *, scenario="same", seed=6):
        output = self.root / role / "BinFill"
        output.mkdir(parents=True)
        (output / "policy_manifest.json").write_text(json.dumps({"evaluation_id": "test:" + role,
            "task_id": "BinFill", "dataset": "val", "seed": 6, "n_action_steps": 16,
            "max_episode_steps": 1300, "scenario_metadata_sha256": scenario}))
        with (output / "simulation_results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["episode_idx", "episode_seed", "success"])
            writer.writeheader()
            for i, success in enumerate(outcomes):
                writer.writerow({"episode_idx": i, "episode_seed": seed + i, "success": success})

    def test_direct_comparison_not_confused_with_original_baseline(self):
        self.manifest()
        self.results("baseline", [1, 1, 1])
        self.results("action_control", [0, 0, 1])
        self.results("reader", [1, 0, 1])
        result, report = driver.write_control_report(self.root, bootstrap_samples=20)
        contrast = result["additional_comparisons"]["action_control_to_reader"]
        self.assertEqual((contrast["wins"], contrast["losses"], contrast["same"]), (1, 0, 2))
        self.assertAlmostEqual(contrast["paired_task_macro_delta"], 1/3)
        self.assertEqual(result["comparisons"]["reader"]["losses"], 1)
        self.assertTrue(contrast["complete"])
        self.assertIn("PRIMARY COMPARISON", report)

    def test_missing_episodes_not_failures_and_mismatch_without_baseline_rejected(self):
        self.manifest()
        self.results("action_control", [0])
        self.results("reader", [1, 1])
        result, _ = driver.write_control_report(self.root, bootstrap_samples=20)
        c = result["additional_comparisons"]["action_control_to_reader"]
        self.assertEqual(c["paired_n"], 1)
        self.assertFalse(c["complete"])
        self.assertIsNone(c["paired_task_macro_bootstrap_ci95"])
        path = self.root / "reader/BinFill/policy_manifest.json"
        data = json.loads(path.read_text())
        data["scenario_metadata_sha256"] = "different"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "scenario_metadata"):
            driver.write_control_report(self.root, bootstrap_samples=20)

    def test_preflight_never_starts_or_creates_output(self):
        output = self.root / "uncreated"
        with patch.object(driver.engine, "check_dependencies"), \
             patch.object(driver.engine, "python_path", return_value=Path(sys.executable)), \
             patch.object(driver, "build_identity", return_value={}), \
             patch.object(driver.engine, "run_evaluation", side_effect=AssertionError("must not run")):
            code = driver.main(["--reader-checkpoint", "reader", "--control-checkpoint", "control",
                                "--output-dir", str(output), "--preflight-only"])
        self.assertEqual(code, 0)
        self.assertFalse(output.exists())

    def test_defaults_are_validation_not_test_and_no_writer(self):
        args = driver.build_parser().parse_args([])
        self.assertEqual(args.dataset, "val")
        self.assertEqual(args.tasks, ["all"])
        self.assertEqual(args.n_episodes, 50)
        self.assertIsNone(args.memory_checkpoint)


if __name__ == "__main__":
    unittest.main()
