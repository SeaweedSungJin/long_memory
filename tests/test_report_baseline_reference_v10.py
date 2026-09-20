"""CPU-only reference provenance, statistical parity and read-only reporting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from gr00t.eval.sim.robomme.compare_long_memory_results import build_report
from run_scripts.robomme.report_baseline_reference_v10 import build_reference_report


class BaselineReferenceReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.current, self.reference = self.root / "current", self.root / "reference"
        self.settings = {"tasks": ["BinFill", "PatternLock"], "n_episodes": 3,
                         "dataset": "val", "seed": 6, "n_action_steps": 16, "max_episode_steps": 1300}
        self.manifest = {"evaluation_id": "new-v10-id", "settings": self.settings,
                         "models": {"baseline": {}, "archive": {}, "archive-off": {}}}
        self.reference_manifest = {"evaluation_id": "original-v7-id", "settings": copy.deepcopy(self.settings),
                                   "models": {"baseline": {}}}
        for directory, manifest in ((self.current, self.manifest), (self.reference, self.reference_manifest)):
            directory.mkdir()
            (directory / "comparison_manifest.json").write_text(json.dumps(manifest))
        self.write_rows(self.reference, self.reference_manifest, "baseline", "BinFill", [0, 1, 1])
        self.write_rows(self.reference, self.reference_manifest, "baseline", "PatternLock", [0, 0, 1])
        self.write_rows(self.current, self.manifest, "archive", "BinFill", [1, 1, 0])
        self.write_rows(self.current, self.manifest, "archive-off", "BinFill", [0, 0, 1])

    def write_rows(self, root, manifest, model, task, successes):
        directory = root / model / task
        directory.mkdir(parents=True, exist_ok=True)
        policy = {"evaluation_id": manifest["evaluation_id"] + ":" + model, "task_id": task,
                  **{key: manifest["settings"][key] for key in
                     ("dataset", "seed", "n_action_steps", "max_episode_steps")},
                  "scenario_metadata_sha256": "same-scenarios", "model_config_sha256": "same-base-config",
                  "memory_window": 4, "demo_sampling": "backward_aligned_full_history"}
        (directory / "policy_manifest.json").write_text(json.dumps(policy))
        (directory / "simulation_results.csv").write_text(
            "episode_idx,episode_seed,success,status,scenario_seed,task_instruction\n" + "".join(
                f"{i},{100+i},{success},{'success' if success else 'fail'},{i},same instruction\n"
                for i, success in enumerate(successes)))

    def report(self):
        return build_reference_report(self.current, self.manifest, self.reference,
                                      self.reference_manifest, bootstrap_samples=40)

    def snapshot(self):
        return {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}

    def test_original_reference_identity_provenance_and_no_writes(self):
        # A stale local baseline must never be adopted as the reference.
        self.write_rows(self.current, self.manifest, "baseline", "BinFill", [1, 1, 1])
        before = self.snapshot()
        manifests = copy.deepcopy((self.manifest, self.reference_manifest))
        result, text = self.report()
        baseline = result["models"]["baseline"]
        self.assertEqual(baseline["tasks"]["BinFill"]["successes"], 2)
        self.assertEqual(baseline["origin"], "reused_reference")
        self.assertEqual(baseline["newly_rolled_out"], 0)
        self.assertEqual(baseline["source_run"], str(self.reference.resolve()))
        self.assertEqual(baseline["source_evaluation_id"], "original-v7-id")
        self.assertEqual(result["evaluation_id"], "new-v10-id")
        self.assertEqual(result["models"]["archive"]["origin"], "fresh")
        self.assertEqual(result["models"]["archive"]["newly_rolled_out"], 3)
        self.assertIn("baseline: REUSED", text)
        self.assertIn("newly rolled out 0", text)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(manifests, (self.manifest, self.reference_manifest))

    def test_exact_generic_statistical_schema_parity(self):
        # Independent synthetic generic fixtures use their own valid identities.
        generic = self.root / "generic"
        generic.mkdir()
        (generic / "comparison_manifest.json").write_text(json.dumps(self.manifest))
        for model, task, values in (("baseline", "BinFill", [0, 1, 1]),
                                    ("baseline", "PatternLock", [0, 0, 1]),
                                    ("archive", "BinFill", [1, 1, 0]),
                                    ("archive-off", "BinFill", [0, 0, 1])):
            self.write_rows(generic, self.manifest, model, task, values)
        expected, _ = build_report(generic, bootstrap_samples=40)
        actual, _ = self.report()
        for model in actual["models"].values():
            for key in ("origin", "newly_rolled_out", "source_run", "source_evaluation_id"):
                model.pop(key)
        self.assertEqual(expected, actual)

    def test_missing_current_results_remain_incomplete_not_failures(self):
        result, text = self.report()
        missing = result["models"]["archive"]["tasks"]["PatternLock"]
        self.assertIsNone(missing["success_rate"])
        self.assertEqual((missing["completed"], missing["successes"]), (0, 0))
        self.assertFalse(missing["complete"])
        self.assertEqual(result["comparisons"]["archive"]["paired_n"], 3)
        self.assertFalse(result["comparisons"]["archive"]["complete"])
        self.assertIn("INCOMPLETE", text)
        json.dumps(result, allow_nan=False)

    def test_reference_original_policy_id_is_required(self):
        path = self.reference / "baseline/BinFill/policy_manifest.json"
        identity = json.loads(path.read_text())
        identity["evaluation_id"] = self.manifest["evaluation_id"] + ":baseline"
        path.write_text(json.dumps(identity))
        with self.assertRaisesRegex(ValueError, "identity does not match"):
            self.report()

    def test_all_cross_policy_context_fields_must_match(self):
        path = self.current / "archive/BinFill/policy_manifest.json"
        original = json.loads(path.read_text())
        for key in ("scenario_metadata_sha256", "model_config_sha256", "memory_window", "demo_sampling"):
            path.write_text(json.dumps({**original, key: "different"}))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key + " differs"):
                self.report()
        path.write_text(json.dumps(original))

    def test_paired_csv_identity_and_invalid_results_rejected(self):
        path = self.current / "archive/BinFill/simulation_results.csv"
        original = path.read_text()
        for altered, message in ((original.replace("0,100,", "0,200,"), "episode_seed differs"),
                                 (original.replace(",0,same instruction", ",9,same instruction"), "scenario_seed differs"),
                                 (original.replace("same instruction", "different instruction"), "task_instruction differs"),
                                 (original.replace(",1,success,", ",1,error,"), "unfinished/error")):
            path.write_text(altered)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.report()
        path.write_text(original)


if __name__ == "__main__":
    unittest.main()
