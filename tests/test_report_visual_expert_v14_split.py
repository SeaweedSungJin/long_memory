"""CPU immutable V14 SAME-expert split reporting; no model or simulator."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from run_scripts.robomme import report_visual_expert_v14_split as report
from tests import test_visual_expert_eval_v14 as fixtures


class VisualExpertSplitTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.VisualExpertEvaluationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        _, full, self.reference, self.reference_manifest = self.fixture.report_fixture(reference=True)
        self.runs = [self.root / "visual_run", self.root / "off_run"]
        self.manifests = []
        for root, role, values in zip(self.runs, ("visual", "visual-off"), ([1, 1, 0], [0, 0, 1])):
            identity = copy.deepcopy(full)
            identity["models"] = {name: value for name, value in identity["models"].items()
                                  if name in ("baseline", role)}
            identity["evaluation_id"] = report.evaluation._identity_digest(identity)
            report.evaluation.bind_manifest(root, identity)
            self.fixture.write_rows(root, identity, role, values)
            self.manifests.append(identity)
            self.clean_status(root)

    @staticmethod
    def clean_status(root):
        (root / "driver_status.json").write_text(json.dumps({"interrupted": False, "failures": [],
            "fatal": None, "inference_files_unchanged": True}))

    def mocked_tiny_shape(self):
        # Pure report/identity/reference/CSV/diagnostic readers are REAL. The
        # evaluator fixture mocks production width/rank; only full model-header
        # file closure is bypassed here and exercised separately below.
        return patch.object(report, "immutable_files", return_value={})

    def snapshot(self):
        return {str(p): p.read_bytes() for root in (*self.runs, self.reference)
                for p in root.rglob("*") if p.is_file()}

    def test_complete_same_expert_contrast_baseline_once_original_ids_and_repeat(self):
        before = self.snapshot()
        with self.mocked_tiny_shape():
            result, text = report.build_split_report(*self.runs, bootstrap_samples=30)
            again, _ = report.build_split_report(*self.runs, bootstrap_samples=30)
        self.assertEqual(result, again)
        self.assertEqual(before, self.snapshot())
        self.assertTrue(result["complete"])
        self.assertTrue(result["source_drivers_complete"])
        self.assertTrue(result["visual_diagnostics_complete"])
        self.assertEqual(set(result["models"]), {"baseline", "visual", "visual-off"})
        self.assertEqual(result["models"]["baseline"]["newly_rolled_out"], 0)
        self.assertEqual(result["models"]["baseline"]["source_evaluation_id"], self.reference_manifest["evaluation_id"])
        self.assertEqual(len(result["baseline_comparisons"]), 2)
        self.assertEqual(len(result["additional_comparisons"]), 1)
        contrast = result["additional_comparisons"]["visual-off_to_visual"]
        self.assertEqual((contrast["paired_n"], contrast["wins"], contrast["losses"], contrast["same"]), (3, 2, 1, 0))
        self.assertNotEqual(result["source_runs"][0]["evaluation_id"], result["source_runs"][1]["evaluation_id"])
        self.assertIn("SAME NEW V14 adapted AE", text)

    def test_missing_rows_rpc_or_clean_terminal_driver_cannot_claim_complete(self):
        self.fixture.write_rows(self.runs[1], self.manifests[1], "visual-off", [0, 0])
        with self.mocked_tiny_shape():
            result, _ = report.build_split_report(*self.runs, bootstrap_samples=20)
        self.assertFalse(result["complete"])
        self.assertEqual(result["additional_comparisons"]["visual-off_to_visual"]["paired_n"], 2)
        self.fixture.write_rows(self.runs[1], self.manifests[1], "visual-off", [0, 0, 1])
        status = self.runs[1] / "driver_status.json"
        status.unlink()
        with self.mocked_tiny_shape():
            result, _ = report.build_split_report(*self.runs, bootstrap_samples=20)
        self.assertTrue(result["episode_results_complete"])
        self.assertFalse(result["source_drivers_complete"])
        self.assertFalse(result["complete"])
        self.clean_status(self.runs[1])
        journal = self.runs[1] / "visual-off/BinFill/memory_diagnostics.jsonl"
        journal.write_text("".join(line for line in journal.read_text().splitlines(keepends=True)
                                   if json.loads(line)["kind"] != "demo_tail_ingest"))
        with self.mocked_tiny_shape():
            result, _ = report.build_split_report(*self.runs, bootstrap_samples=20)
        self.assertTrue(result["episode_results_complete"])
        self.assertFalse(result["visual_diagnostics_complete"])
        self.assertFalse(result["complete"])

    def test_empty_role_is_excluded_not_counted_as_failures(self):
        (self.runs[1] / "visual-off/BinFill/simulation_results.csv").unlink()
        with self.mocked_tiny_shape():
            result, _ = report.build_split_report(*self.runs, bootstrap_samples=20)
        self.assertIsNone(result["models"]["visual-off"]["available_task_macro"])
        self.assertEqual(result["additional_comparisons"]["visual-off_to_visual"]["paired_n"], 0)
        self.assertFalse(result["complete"])

    def test_consistently_rehashed_different_bundle_or_expert_still_rejected(self):
        mutations = [lambda m: m["models"]["visual-off"].update(memory_checkpoint="/other-identical-looking-bundle"),
            lambda m: m["settings"].update(seed=7),
            lambda m: m["source_sha256"].update({"new.py": "a" * 64}),
            lambda m: m["baseline_reference"].update(source_evaluation_id="other-original"),
            lambda m: m["policy_package_versions"].update(torch="different"),
            lambda m: m["models"]["visual-off"].update(expert_weights_sha256="a" * 64)]
        for mutate in mutations:
            altered = copy.deepcopy(self.manifests[1]); mutate(altered)
            altered["evaluation_id"] = report.evaluation._identity_digest(altered)
            with self.assertRaises(ValueError):
                report.validate_pair(self.manifests[0], altered)
        altered = copy.deepcopy(self.manifests[1])
        actor = altered["models"]["visual-off"]
        actor["expert_weights_sha256"] = actor["checkpoint_files_sha256"]["expert.safetensors"] = "a" * 64
        actor["training_metadata"]["payload_sha256"]["expert.safetensors"] = "a" * 64
        altered["evaluation_id"] = report.evaluation._identity_digest(altered)
        with self.assertRaisesRegex(ValueError, "SAME V14 bundle"):
            report.validate_pair(self.manifests[0], altered)

    def test_actual_source_file_closure_expert_header_and_content_hashes(self):
        manifests = copy.deepcopy(self.manifests)
        actual = json.loads((self.fixture.bundle / "checkpoint.json").read_text())["config"]
        for manifest in manifests:
            for role, model in manifest["models"].items():
                if role != "baseline":
                    model["visual_config"] = actual["visual"]
                    model["expert_config"] = actual["expert"]
                    model["expert_targets"] = actual["expert_targets"]
        files = report.immutable_files(manifests)
        report.verify_files(files)
        self.assertIn(str(self.fixture.bundle / "expert.safetensors"), files)
        self.assertIn(str(self.fixture.fixture.parent / "model.safetensors"), files)
        self.assertIn(str(report.ROOT / report.evaluation.EVALUATOR), files)
        manifests[1]["models"]["visual-off"]["expert_targets"] = []
        with self.assertRaisesRegex(ValueError, "actual checkpoint header"):
            report.immutable_files(manifests)
        payload = self.fixture.bundle / "expert.safetensors"
        payload.write_bytes(payload.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "changed"):
            report.verify_files(files)

    def test_original_policy_id_scenario_and_duplicate_episode_rejected(self):
        path = self.runs[1] / "visual-off/BinFill/simulation_results.csv"
        before = path.read_text()
        path.write_text(before + before.splitlines()[-1] + "\n")
        with self.mocked_tiny_shape(), self.assertRaisesRegex(ValueError, "duplicate episode"):
            report.build_split_report(*self.runs, bootstrap_samples=20)
        path.write_text(before)
        policy = path.with_name("policy_manifest.json")
        context = json.loads(policy.read_text()); original = copy.deepcopy(context)
        context["evaluation_id"] = self.manifests[0]["evaluation_id"] + ":visual-off"
        policy.write_text(json.dumps(context))
        with self.mocked_tiny_shape(), self.assertRaisesRegex(ValueError, "identity"):
            report.build_split_report(*self.runs, bootstrap_samples=20)
        original["scenario_metadata_sha256"] = "wrong"
        policy.write_text(json.dumps(original))
        with self.mocked_tiny_shape(), self.assertRaisesRegex(ValueError, "scenario"):
            report.build_split_report(*self.runs, bootstrap_samples=20)

    def test_snapshot_race_or_runtime_change_fails_without_output(self):
        original, calls = report.result_snapshot, []
        def racing(*args):
            value = original(*args); calls.append(1)
            if len(calls) > 1:
                value["new-file"] = None
            return value
        with self.mocked_tiny_shape(), patch.object(report, "result_snapshot", side_effect=racing), \
                self.assertRaisesRegex(ValueError, "changed during read"):
            report.build_split_report(*self.runs, bootstrap_samples=20)
        with patch.object(report, "immutable_files", side_effect=[{}, {"changed": "a" * 64}]), \
                self.assertRaisesRegex(ValueError, "runtime/header"):
            report.build_split_report(*self.runs, bootstrap_samples=20)

    def test_reference_corruption_is_revalidated_and_never_copied(self):
        path = self.reference / "baseline/BinFill/simulation_results.csv"
        path.write_text(path.read_text().replace("same instruction", "changed"))
        with self.mocked_tiny_shape(), self.assertRaises(ValueError):
            report.build_split_report(*self.runs, bootstrap_samples=20)
        self.assertTrue(all(not (root / "baseline").exists() for root in self.runs))

    def test_new_output_only_preserves_all_sources(self):
        before, output = self.snapshot(), self.root / "combined"
        args = ["--visual-run", str(self.runs[0]), "--visual-off-run", str(self.runs[1]),
                "--output-dir", str(output), "--bootstrap-samples", "20"]
        with self.mocked_tiny_shape(), redirect_stdout(io.StringIO()):
            self.assertEqual(report.main(args), 0)
        self.assertEqual(before, self.snapshot())
        self.assertEqual({p.name for p in output.iterdir()}, {"comparison_summary.json", "comparison_summary.txt"})
        with self.assertRaises(FileExistsError):
            report.main(args)
        with self.assertRaises(ValueError):
            report.main(args[:-4] + ["--output-dir", str(self.runs[0] / "bad")])


if __name__ == "__main__":
    unittest.main()
