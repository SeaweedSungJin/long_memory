"""CPU-only immutable split-report protocol, pairing and incomplete evidence."""
from contextlib import ExitStack, redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from run_scripts.robomme import report_visual_differential_v12_split as report
from tests import test_visual_differential_eval_v12 as evaluator_fixtures


class SplitReportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = evaluator_fixtures.DifferentialEvaluationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.runs = [self.root / "split_diff", self.root / "split_current"]
        self.reference = self.root / "reference"
        self.reference.mkdir()
        full = self.fixture.identity()
        full["settings"]["n_episodes"] = 2
        for role, model in full["models"].items():
            if role == "baseline":
                continue
            model["training_config"]["max_steps"] = 512
            model["training_metadata"]["selection_metric"] = "val/generated_observed_prefix_mae"
            model["training_contract"] = report.evaluation.training_contract(
                {"train": model["training_config"], "objective": model.get("training_objective")}, model["training_metadata"])
        full["baseline_reference"] = {"kind": "completed_original_baseline_reference", "role": "baseline",
            "newly_rolled_out": 0, "source_run": str(self.reference), "source_evaluation_id": "original-reference-id",
            "files_sha256": {"comparison_manifest.json": "a"*64}, "completed_episodes": 2}
        self.manifests = []
        for root, keep in zip(self.runs, (("baseline", "differential", "visual-off"), ("baseline", "current-only"))):
            root.mkdir()
            value = copy.deepcopy(full)
            value["models"] = {role: value["models"][role] for role in keep}
            value["evaluation_id"] = report.evaluation._identity_digest(value)
            (root / "comparison_manifest.json").write_text(json.dumps(value))
            self.manifests.append(value)
        self.reference_manifest = copy.deepcopy(self.manifests[0])
        self.reference_manifest["models"] = {"baseline": self.reference_manifest["models"]["baseline"]}
        self.reference_manifest["evaluation_id"] = "original-reference-id"
        self.write_rows(self.reference, "baseline", self.reference_manifest, [1, 0])
        self.write_rows(self.runs[0], "differential", self.manifests[0], [1, 1])
        self.write_rows(self.runs[0], "visual-off", self.manifests[0], [0, 1])
        self.write_rows(self.runs[1], "current-only", self.manifests[1], [1, 0])

    def write_rows(self, root, role, manifest, successes):
        path = root / role / "BinFill"
        path.mkdir(parents=True, exist_ok=True)
        with (path / "simulation_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("episode_idx", "episode_seed", "scenario_seed", "task_instruction", "success"))
            writer.writeheader()
            for index, success in enumerate(successes):
                writer.writerow(dict(episode_idx=index, episode_seed=index+6, scenario_seed=index+100,
                                     task_instruction="fill the bin", success=success))
        identity = {"evaluation_id": manifest["evaluation_id"]+":"+role, "task_id": "BinFill",
                    **{key:manifest["settings"][key] for key in ("dataset","seed","n_action_steps","max_episode_steps")},
                    "scenario_metadata_sha256":"scenario", "model_config_sha256":"model", "memory_window":4,
                    "demo_sampling":"backward_aligned_full_history"}
        (path / "policy_manifest.json").write_text(json.dumps(identity))

    def rehash(self, value):
        value["evaluation_id"] = report.evaluation._identity_digest(value)
        return value

    def fake_reference(self, record, current):
        if record != self.manifests[0]["baseline_reference"]:
            raise ValueError("changed baseline reference")
        return self.reference, self.reference_manifest

    def mocked_external_identity(self):
        stack = ExitStack()
        # Real local manifest/result/scenario/role/statistics/diagnostic readers
        # remain active. Only unrelated real benchmark/base validation is mocked.
        stack.enter_context(patch.object(report, "immutable_files", return_value={}))
        stack.enter_context(patch.object(report.evaluation, "validate_reference", side_effect=self.fake_reference))
        return stack

    def snapshot(self):
        return {str(path):path.read_bytes() for root in (*self.runs,self.reference)
                for path in root.rglob("*") if path.is_file()}

    def test_pure_complete_union_three_contrasts_baseline_once_and_repeat(self):
        before = self.snapshot()
        with self.mocked_external_identity():
            result, text = report.build_split_report(*self.runs, bootstrap_samples=30)
            again, _ = report.build_split_report(*self.runs, bootstrap_samples=30)
        self.assertEqual(result, again)
        self.assertEqual(before,self.snapshot())
        self.assertTrue(result["complete"])
        self.assertFalse(result["visual_diagnostics_complete"])
        self.assertEqual(result["models"]["baseline"]["newly_rolled_out"],0)
        self.assertEqual(result["models"]["baseline"]["source_evaluation_id"],"original-reference-id")
        self.assertEqual(result["additional_comparisons"]["current-only_to_differential"]["paired_task_macro_delta"],.5)
        self.assertEqual(len(result["additional_comparisons"]),3)
        self.assertEqual(len(result["baseline_comparisons"]),3)
        self.assertNotEqual(result["source_runs"][0]["evaluation_id"],result["source_runs"][1]["evaluation_id"])
        self.assertIn("Baseline REUSED",text)

    def test_partial_and_empty_are_excluded_not_failures(self):
        self.write_rows(self.runs[1],"current-only",self.manifests[1],[1])
        with self.mocked_external_identity():
            result,text=report.build_split_report(*self.runs,bootstrap_samples=30)
        self.assertFalse(result["complete"])
        self.assertEqual(result["models"]["current-only"]["tasks"]["BinFill"]["success_rate"],1.)
        self.assertEqual(result["additional_comparisons"]["current-only_to_differential"]["paired_n"],1)
        self.assertIsNone(result["additional_comparisons"]["current-only_to_differential"]["paired_task_macro_bootstrap_ci95"])
        self.assertIn("INCOMPLETE",text)
        (self.runs[1]/"current-only/BinFill/simulation_results.csv").unlink()
        with self.mocked_external_identity():
            result,_=report.build_split_report(*self.runs,bootstrap_samples=30)
        self.assertIsNone(result["models"]["current-only"]["available_task_macro"])
        self.assertEqual(result["additional_comparisons"]["current-only_to_differential"]["paired_n"],0)

    def test_manifest_mode_parent_plan_source_rollout_reference_mismatch(self):
        mutations=[lambda m:m["models"]["current-only"].update(read_mode="differential"),
            lambda m:m["models"]["current-only"]["frozen_parent"].update(path="wrong"),
            lambda m:m["models"]["current-only"]["training_metadata"].update(plan_sha256="f"*64),
            lambda m:m["settings"].update(seed=7),
            lambda m:m["source_sha256"].update({"gr00t/changed.py":"a"*64}),
            lambda m:m["baseline_reference"].update(source_evaluation_id="other")]
        for mutation in mutations:
            right=copy.deepcopy(self.manifests[1]);mutation(right);self.rehash(right)
            with self.assertRaises(ValueError):report.validate_pair(self.manifests[0],right)

    def test_consistently_rehashed_cross_arm_training_or_parent_difference(self):
        right=copy.deepcopy(self.manifests[1]);model=right["models"]["current-only"]
        model["training_config"]["seed"]=999
        model["training_contract"]=report.evaluation.training_contract(
            {"train":model["training_config"],"objective":model.get("training_objective")},model["training_metadata"])
        self.rehash(right)
        with self.assertRaisesRegex(ValueError,"training_contract"):report.validate_pair(self.manifests[0],right)
        right=copy.deepcopy(self.manifests[1]);model=right["models"]["current-only"]
        model["frozen_parent"]["path"]="/another-parent"
        model["training_metadata"]["frozen_parent"]=copy.deepcopy(model["frozen_parent"])
        self.rehash(right)
        with self.assertRaisesRegex(ValueError,"frozen_parent"):report.validate_pair(self.manifests[0],right)

    def test_duplicate_episode_wrong_original_policy_id_and_scenario_fail(self):
        path=self.runs[1]/"current-only/BinFill/simulation_results.csv"
        original=path.read_text();path.write_text(original+original.splitlines()[-1]+"\n")
        with self.mocked_external_identity(),self.assertRaisesRegex(ValueError,"duplicate episode"):
            report.build_split_report(*self.runs,bootstrap_samples=10)
        path.write_text(original)
        policy=path.with_name("policy_manifest.json");value=json.loads(policy.read_text())
        value["evaluation_id"]="copied:id";policy.write_text(json.dumps(value))
        with self.mocked_external_identity(),self.assertRaisesRegex(ValueError,"identity"):
            report.build_split_report(*self.runs,bootstrap_samples=10)
        self.write_rows(self.runs[1],"current-only",self.manifests[1],[1,0])
        value=json.loads(policy.read_text());value["scenario_metadata_sha256"]="wrong";policy.write_text(json.dumps(value))
        with self.mocked_external_identity(),self.assertRaisesRegex(ValueError,"scenario"):
            report.build_split_report(*self.runs,bootstrap_samples=10)

    def test_step0_independent_selection_allowed_only_with_original_admission(self):
        right=copy.deepcopy(self.manifests[1]);right["allow_initialization_checkpoints"]=True
        model=right["models"]["current-only"];model.update(step=0,checkpoint_status="INITIALIZATION_SELECTED")
        self.rehash(right);report.validate_pair(self.manifests[0],right)
        right["allow_initialization_checkpoints"]=False;self.rehash(right)
        with self.assertRaises(ValueError):report.validate_pair(self.manifests[0],right)

    def test_snapshot_race_fails_without_output(self):
        actual=report.result_snapshot; calls=[]
        def changed(*args):
            result=actual(*args);calls.append(1)
            if len(calls)>1:result["new-file"]=None
            return result
        with self.mocked_external_identity(),patch.object(report,"result_snapshot",side_effect=changed),self.assertRaisesRegex(ValueError,"changed during read"):
            report.build_split_report(*self.runs,bootstrap_samples=10)

    def test_new_output_only_original_sources_preserved(self):
        before=self.snapshot(); output=self.root/"combined"
        args=["--differential-run",str(self.runs[0]),"--current-only-run",str(self.runs[1]),"--output-dir",str(output),"--bootstrap-samples","10"]
        with self.mocked_external_identity(),redirect_stdout(io.StringIO()):self.assertEqual(report.main(args),0)
        self.assertEqual(before,self.snapshot())
        self.assertEqual({p.name for p in output.iterdir()},{"comparison_summary.json","comparison_summary.txt"})
        with self.assertRaises(FileExistsError):report.main(args)
        with self.assertRaises(ValueError):report.main(args[:-4]+["--output-dir",str(self.runs[0]/"bad")])

    def test_file_hash_and_path_escape_fail(self):
        path=self.root/"immutable";path.write_text("original")
        expected={str(path):report.evaluation.file_hash(path)}
        report.verify_files(expected);path.write_text("changed")
        with self.assertRaisesRegex(ValueError,"changed"):report.verify_files(expected)
        with self.assertRaisesRegex(ValueError,"escapes"):report._contained(self.root,"../outside")

    def test_actual_cpu_immutable_file_closure_and_benchmark_binding(self):
        manifests=copy.deepcopy(self.manifests)
        benchmark=self.root/"benchmark_metadata.json";benchmark.write_text('{"fixture":true}')
        for manifest in manifests:
            manifest["benchmark"]={"source_and_scenario_sha256":{str(benchmark):report.evaluation.file_hash(benchmark)}}
        files=report.immutable_files(manifests)
        report.verify_files(files)
        self.assertIn(str(benchmark),files)
        self.assertIn(str(self.fixture.bundle/"visual.safetensors"),files)
        self.assertIn(str(self.fixture.fixture.parent/"expert.safetensors"),files)
        manifests[1]["source_sha256"]["unknown/source.py"]="a"*64
        with self.assertRaisesRegex(ValueError,"source closure"):report.immutable_files(manifests)
        del manifests[1]["source_sha256"]["unknown/source.py"]
        manifests[1]["benchmark"]["source_and_scenario_sha256"][str(benchmark)]="b"*64
        with self.assertRaisesRegex(ValueError,"Conflicting"):report.immutable_files(manifests)


if __name__=="__main__":unittest.main()
