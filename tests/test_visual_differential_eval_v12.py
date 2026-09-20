"""CPU provenance/dispatch and completed-session visual evidence regressions."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from run_scripts.robomme import eval_visual_differential_v12 as driver
from run_scripts.robomme import checkpoint_visual_differential_v12 as checkpoint
from tests import test_checkpoint_visual_differential_v12 as fixtures
from run_scripts.robomme.baseline_reference_v10 import SERVERS


class DifferentialEvaluationTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.VisualCheckpointTests("test_roundtrip_external_parent_immutable_and_readonly_rng")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.fixture = fixture
        self.root, self.base = fixture.root, fixture.base
        fixture.metadata.update(plan_sha256="c" * 64, initialization_reference={"path": "v11-reference", "initial_visual_sha256": "d" * 64})
        self.bundle = fixture.save(step=1)
        fixture.config["read_mode"] = fixture.config["train"]["read_mode"] = fixture.metadata["read_mode"] = "current_only"
        fixture.visual.read_mode = "current_only"
        self.current_bundle = fixture.save(name="current", step=2)
        # Only the evaluator's audited 2048-width header check is mocked: tiny
        # serializer/payload validation is separately REAL before this override.
        self.header = checkpoint.checkpoint_info(self.base, self.bundle)
        self.header["config"]["visual"]["feature_dim"] = 2048
        self.current_header = checkpoint.checkpoint_info(self.base, self.current_bundle)
        self.current_header["config"]["visual"]["feature_dim"] = 2048

    def args(self, *extra):
        return driver.build_parser().parse_args([
            "--base-model", str(self.base), "--differential-checkpoint", str(self.bundle),
            "--current-only-checkpoint", str(self.current_bundle),
            "--tasks", "BinFill", "--n-episodes", "1", "--output-dir", str(self.root / "eval"), *extra])

    def identity(self, args=None):
        args = args or self.args()
        driver.validate_options(args)
        with patch.object(checkpoint, "checkpoint_info", side_effect=lambda base, path, expected_stage: copy.deepcopy(
                    self.header if Path(path) == self.bundle else self.current_header)), \
                patch.object(driver, "benchmark_identity", return_value={"fixture": True}), \
                contextlib.redirect_stdout(io.StringIO()):
            return driver.build_identity(args)

    def test_identity_roles_source_closure_and_server_dispatch(self):
        identity = self.identity()
        driver.validate_manifest_contract(identity)
        self.assertFalse((self.root / "eval").exists())
        for role, model in identity["models"].items():
            command = driver.server_command(self.args(), model, 1234)
            server = driver.BASELINE_SERVER if role == "baseline" else driver.SERVER
            self.assertIn(str(driver.REPO_ROOT / server), command)
            self.assertEqual("--visual-read-off" in command, role == "visual-off")
            self.assertEqual("--memory-checkpoint" in command, role != "baseline")
            if role != "baseline":
                self.assertEqual(command[command.index("--expected-read-mode") + 1], model["read_mode"])
            self.assertNotIn("--memory-off", command)
            self.assertNotIn("--archive-read-off", command)
        for name in (driver.SERVER, driver.BASELINE_SERVER, driver.EVALUATOR, *driver.DEPENDENCIES):
            self.assertEqual(identity["source_sha256"][str(name)], driver.file_hash(driver.REPO_ROOT / name))
        self.assertEqual(identity["models"]["differential"]["frozen_parent"],
                         identity["models"]["visual-off"]["frozen_parent"])

    def test_role_parent_payload_and_digest_tampering_rejected(self):
        original = self.identity()
        changes = [lambda m: m["models"]["visual-off"].update(visual_read_off=False),
                   lambda m: m["models"]["differential"].update(archive_read_off=True),
                   lambda m: m["models"]["differential"].update(memory_off=0),
                   lambda m: m["models"]["differential"].update(weights_sha256="a" * 64),
                   lambda m: m["models"]["differential"]["frozen_parent"].update(step=9108),
                   lambda m: m["models"]["differential"]["visual_config"].update(hidden_dim=7),
                   lambda m: m["models"]["baseline"].update(server_script=str(driver.SERVER)),
                   lambda m: m["models"]["baseline"].update(visual_config={}),
                   lambda m: m["models"]["differential"].update(step=0),
                   lambda m: m["models"]["current-only"].update(read_mode="differential"),
                   lambda m: m["models"]["current-only"]["training_config"].update(read_mode="differential"),
                   lambda m: m["models"]["current-only"]["training_metadata"].update(plan_sha256="f" * 64),
                   lambda m: m["models"]["current-only"]["training_metadata"].update(initialization_reference={}),
                   lambda m: m["models"]["differential"].update(semantic_state_sha256="bad")]
        for i, change in enumerate(changes):
            changed = copy.deepcopy(original)
            change(changed)
            with self.subTest(case=i):
                with self.assertRaisesRegex(ValueError, "digest"):
                    driver.validate_manifest_contract(changed)
                changed["evaluation_id"] = driver._identity_digest(changed)
                with self.assertRaises(ValueError):
                    driver.validate_manifest_contract(changed)

    def test_readonly_preflight_and_immutable_output(self):
        identity = self.identity()
        output = self.root / "eval"
        driver.bind_manifest(output, identity)
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        self.assertEqual(self.identity(), identity)
        with self.assertRaisesRegex(ValueError, "NEW --output-dir"):
            self.identity(self.args("--seed", "7"))
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
        with patch.object(driver, "check_dependencies"), patch.object(driver, "build_identity", return_value=identity), \
                patch.object(driver, "run_evaluation") as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(["--preflight-only", "--server-python", sys.executable,
                "--differential-checkpoint", str(self.bundle), "--current-only-checkpoint", str(self.current_bundle),
                "--output-dir", str(self.root / "absent")]), 0)
        launch.assert_not_called()
        self.assertFalse((self.root / "absent").exists())

    def test_zero_step_stride_and_protected_output_rejected(self):
        with self.assertRaisesRegex(ValueError, "memory stride"):
            self.identity(self.args("--n-action-steps", "8"))
        with self.assertRaisesRegex(ValueError, "inside a source"):
            self.identity(self.args("--output-dir", str(self.fixture.parent / "unsafe")))
        self.header["step"] = 0
        with self.assertRaisesRegex(ValueError, "step>0"):
            self.identity()

    def test_initialization_selection_requires_explicit_honest_bound_flag(self):
        self.current_header["step"] = 0
        with self.assertRaisesRegex(ValueError, "allow-initialization"):
            self.identity()
        identity = self.identity(self.args("--allow-initialization-checkpoints"))
        self.assertTrue(identity["allow_initialization_checkpoints"])
        self.assertNotIn("allow_initialization_checkpoints", identity["settings"])
        model = identity["models"]["current-only"]
        self.assertEqual((model["step"], model["checkpoint_status"]), (0, "INITIALIZATION_SELECTED"))
        self.assertIn("zero learned visual updates", model["description"])
        for mutate in (lambda m: m.update(allow_initialization_checkpoints=False),
                       lambda m: m["models"]["current-only"].update(checkpoint_status="TRAINED")):
            changed = copy.deepcopy(identity)
            mutate(changed)
            changed["evaluation_id"] = driver._identity_digest(changed)
            with self.assertRaises(ValueError):
                driver.validate_manifest_contract(changed)

    def test_fatal_report_records_failure_and_preserves_original_exception(self):
        args = self.args("--models", "baseline")
        identity = self.identity(args)
        with patch.object(driver.subprocess, "Popen"), patch.object(driver, "server_ready"), \
                patch.object(driver, "stop_process"), patch.object(driver, "run_client"), \
                patch.object(driver, "write_report", side_effect=ValueError("contradictory parent evidence")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "contradictory parent evidence"):
                driver.run_evaluation(args, identity, {})
        status = json.loads((self.root / "eval/driver_status.json").read_text())
        self.assertEqual(status["fatal"]["error_type"], "ValueError")
        self.assertTrue(status["failures"])
        self.assertEqual(status["failures"][-1]["scope"], "driver_or_report")

    def records(self, role, model=None):
        model = model or self.identity()["models"][role]
        records = []
        for i, (frame, passive) in enumerate(((0, True), (16, False)), 1):
            enabled = role != "visual-off" and not passive and i > 1
            visual = {"read_enabled": enabled, "observed": i, "append_updates": i,
                      "demo_updates": 1, "bank_observations": i, "bank_tokens": i * 162,
                      "frame_index": frame, "passive": passive,
                      "image_delta_norm": .3 if enabled else 0.,
                      "image_changed_fraction": .5 if enabled else 0.,
                      "read_mode": model["read_mode"], "past_read_enabled": enabled and model["read_mode"] == "differential",
                      "current_reference_enabled": enabled, "readout_norm": .8 if enabled else 0.,
                      "projected_residual_norm": .3 if enabled else 0.,
                      "readout_norm_semantics": "H_current" if model["read_mode"] == "current_only" else "H_past_minus_H_current"}
            info = {"checkpoint_variant": driver.VARIANT, "checkpoint_step": model["step"],
                    "stage": 1, "expert_adapted": True, "visual_read_mode": model["read_mode"],
                    "visual_read_off": role == "visual-off", "visual_weights_sha256": model["weights_sha256"],
                    "frozen_parent_checkpoint_sha256": model["frozen_parent"]["files_sha256"]["checkpoint.json"],
                    "visual_memory": visual,
                    "long_memory": {"mode": "archive", "policy": "append",
                        "memory_read_enabled": not passive and i > 1,
                        "observations_seen": i, "write_attempts": i, "updates": i,
                        "keeps": 0, "demo_updates": 1, "demo_keeps": 0,
                        "frame_index": frame, "passive": passive}}
            records.append({"kind": "policy_call", "session_id": "kept", "episode_idx": 0,
                            "episode_seed": 100, "frame_index": frame, "passive": passive, "info": info})
        records.append({"kind": "episode_complete", "session_id": "kept", "episode_idx": 0,
                        "episode_seed": 100, "success": True})
        return model, records

    def diagnostics(self, role, model, records, suffix=""):
        folder = self.root / "journal" / role / "BinFill"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "simulation_results.csv").write_text(
            "episode_idx,episode_seed,success,status,scenario_seed,task_instruction\n0,100,1,success,0,fixture\n")
        (folder / "memory_diagnostics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records) + suffix)
        return driver.completed_visual_diagnostics(folder.parent.parent, role, ["BinFill"], 1, model)

    def test_actual_completed_evidence_and_visual_off(self):
        for role in ("differential", "current-only", "visual-off"):
            model, records = self.records(role)
            result = self.diagnostics(role, model, records)
            self.assertTrue(result["complete_evidence"])
            self.assertEqual(result["append_updates"], 2)
            self.assertEqual(result["demo_updates"], 1)
            self.assertEqual(result["read_enabled_calls"], int(role != "visual-off"))
            self.assertEqual(result["past_read_enabled_calls"], int(role == "differential"))
            self.assertEqual(result["current_reference_enabled_calls"], int(role != "visual-off"))
            self.assertEqual(result["completed_sessions"], 1)

    def test_abandoned_retry_and_torn_final_not_counted_as_complete(self):
        model, records = self.records("differential")
        abandoned = copy.deepcopy(records[0])
        abandoned.update(session_id="abandoned", frame_index=-99, info={})
        result = self.diagnostics("differential", model, [abandoned, *records], '{"torn"')
        self.assertTrue(result["complete_evidence"])
        self.assertEqual(result["calls"], 2)
        self.assertEqual(result["ignored_torn_final_lines"], 1)
        records[-1]["episode_seed"] = 999
        self.assertFalse(self.diagnostics("differential", model, records)["complete_evidence"])

    def test_missing_evidence_not_silently_interpreted_as_bypass(self):
        model, records = self.records("visual-off")
        records[1]["info"].pop("visual_memory")
        records[1]["info"].pop("visual_weights_sha256")
        result = self.diagnostics("visual-off", model, records)
        self.assertFalse(result["complete_evidence"])
        self.assertEqual(result["missing_identity"], 1)
        self.assertEqual(result["missing_calls"], 1)
        records[1]["info"].pop("long_memory")
        self.assertEqual(self.diagnostics("visual-off", model, records)["missing_parent_calls"], 1)

    def test_wrong_loaded_identity_cadence_counters_and_metric_rejected(self):
        model, original = self.records("differential")
        changes = [lambda r: r[1]["info"].update(checkpoint_step=99),
                   lambda r: r[1].update(frame_index=0),
                   lambda r: r[1]["info"]["visual_memory"].update(bank_tokens=162),
                   lambda r: r[1]["info"]["visual_memory"].update(read_enabled=False),
                   lambda r: r[1]["info"]["long_memory"].update(memory_read_enabled=False),
                   lambda r: r[1]["info"]["long_memory"].update(observations_seen=1),
                   lambda r: r[1]["info"].update(expert_adapted=False),
                   lambda r: r[1]["info"].update(visual_read_mode="current_only"),
                   lambda r: r[1]["info"]["visual_memory"].update(past_read_enabled=False),
                   lambda r: r[1]["info"]["visual_memory"].update(current_reference_enabled=False),
                   lambda r: r[1]["info"]["visual_memory"].update(readout_norm_semantics="H_current"),
                   lambda r: r[0]["info"]["visual_memory"].update(readout_norm=.1),
                   lambda r: r[0]["info"]["visual_memory"].update(image_delta_norm=.1),
                   lambda r: r[1]["info"]["visual_memory"].update(image_changed_fraction=float("nan"))]
        for i, change in enumerate(changes):
            records = copy.deepcopy(original)
            change(records)
            with self.subTest(case=i), self.assertRaises(ValueError):
                self.diagnostics("differential", model, records)
        with self.assertRaisesRegex(ValueError, "Malformed"):
            self.diagnostics("differential", model, original, '{"torn"\n')

    def test_arm_mismatch_rejected_during_readonly_preflight(self):
        self.current_header["config"]["read_mode"] = "differential"
        with self.assertRaisesRegex(ValueError, "read_mode"):
            self.identity()
        self.assertFalse((self.root / "eval").exists())

    def test_current_only_cannot_report_historical_read_or_missing_reference_metric(self):
        model, records = self.records("current-only")
        records[1]["info"]["visual_memory"]["past_read_enabled"] = True
        with self.assertRaisesRegex(ValueError, "past_read_enabled"):
            self.diagnostics("current-only", model, records)
        records[1]["info"]["visual_memory"]["past_read_enabled"] = False
        records[1]["info"]["visual_memory"].pop("readout_norm")
        self.assertFalse(self.diagnostics("current-only", model, records)["complete_evidence"])

    def write_rows(self, root, manifest, role, values, *, diagnostics=True):
        folder = root / role / "BinFill"
        folder.mkdir(parents=True, exist_ok=True)
        settings = manifest["settings"]
        context = {"evaluation_id": manifest["evaluation_id"] + ":" + role, "task_id": "BinFill",
            **{k: settings[k] for k in ("dataset", "seed", "n_action_steps", "max_episode_steps")},
            "model_config_sha256": manifest["base_file_sha256"]["config.json"], "memory_window": 4,
            "demo_sampling": "backward_aligned_full_history", "scenario_metadata_sha256": self.scenario_digest}
        (folder / "policy_manifest.json").write_text(json.dumps(context))
        (folder / "simulation_results.csv").write_text(
            "episode_idx,episode_seed,success,status,scenario_seed,task_instruction\n" + "".join(
                f"{i},{100+i},{v},{'success' if v else 'fail'},{i},same instruction\n" for i, v in enumerate(values)))
        if role != "baseline" and diagnostics:
            _, template = self.records(role, manifest["models"][role])
            records = []
            for episode, success in enumerate(values):
                for entry in copy.deepcopy(template):
                    entry.update(episode_idx=episode, episode_seed=100+episode, session_id=f"session-{episode}")
                    if entry["kind"] == "episode_complete":
                        entry["success"] = bool(success)
                    records.append(entry)
            (folder / "memory_diagnostics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    def report_fixture(self, *, reference=False):
        args = self.args("--n-episodes", "3")
        scenario = self.root / "benchmark/env_metadata/val/record_dataset_BinFill_metadata.json"
        scenario.parent.mkdir(parents=True, exist_ok=True)
        records = [{"task": "BinFill", "episode": i, "seed": i, "difficulty": "easy"} for i in range(5)]
        scenario.write_text(json.dumps({"env_id": "BinFill", "record_count": 5, "records": records}, indent=2) + "\n")
        self.scenario_digest = hashlib.sha256(json.dumps([
            {"episode_idx": i, "seed": i, "difficulty": "easy"} for i in range(5)], sort_keys=True).encode()).hexdigest()
        benchmark = {"versions": {"robomme": "fixture"}, "source_and_scenario_sha256": {str(scenario): driver.file_hash(scenario)}}
        with patch.object(driver, "benchmark_identity", return_value=benchmark):
            # identity() has its own generic benchmark mock; invoke the driver
            # directly here, retaining actual serializer and payload hashes.
            driver.validate_options(args)
            with patch.object(checkpoint, "checkpoint_info", side_effect=lambda base, path, expected_stage: copy.deepcopy(
                    self.header if Path(path) == self.bundle else self.current_header)), contextlib.redirect_stdout(io.StringIO()):
                identity = driver.build_identity(args)
        old_root, old = None, None
        if reference:
            old_root = self.root / "reference"
            old = copy.deepcopy(identity)
            old["trainer_variant"] = "archive_read_control_v7"
            old["models"] = {"baseline": old["models"]["baseline"]}
            script, sha = SERVERS[old["trainer_variant"]]
            old["models"]["baseline"]["server_script"] = script
            old["source_sha256"][script] = sha
            old["evaluation_id"] = driver._identity_digest(old)
            driver.bind_manifest(old_root, old)
            (old_root / "driver_status.json").write_text(json.dumps({"interrupted": False, "failures": []}))
            self.write_rows(old_root, old, "baseline", [0, 1, 1])
            args.baseline_reference = old_root
            with patch.object(driver, "benchmark_identity", return_value=benchmark), \
                    patch.object(checkpoint, "checkpoint_info", side_effect=lambda base, path, expected_stage: copy.deepcopy(
                        self.header if Path(path) == self.bundle else self.current_header)), contextlib.redirect_stdout(io.StringIO()):
                identity = driver.build_identity(args)
        self.assertFalse(args.output_dir.exists())
        driver.bind_manifest(args.output_dir, identity)
        if not reference:
            self.write_rows(args.output_dir, identity, "baseline", [0, 1, 1])
        for role, values in (("differential", [1, 1, 0]), ("current-only", [0, 1, 0]), ("visual-off", [0, 0, 1])):
            self.write_rows(args.output_dir, identity, role, values)
        return args, identity, old_root, old

    def test_complete_four_role_report_paired_counts_and_missing_excluded(self):
        args, identity, _, _ = self.report_fixture()
        result, text = driver.build_control_report(args.output_dir, bootstrap_samples=30)
        self.assertTrue(result["visual_diagnostics_complete"])
        self.assertTrue(all(model["complete"] for model in result["models"].values()))
        self.assertEqual(result["comparisons"]["differential"]["paired_n"], 3)
        extra = result["additional_comparisons"]
        self.assertEqual((extra["visual-off_to_differential"]["wins"], extra["visual-off_to_differential"]["losses"]), (2, 1))
        self.assertEqual((extra["current-only_to_differential"]["wins"], extra["current-only_to_differential"]["losses"]), (1, 0))
        self.assertIn("independently selected checkpoints", text)
        self.assertIn("not equal compute", text)
        self.write_rows(args.output_dir, identity, "current-only", [0, 1])
        result, _ = driver.build_control_report(args.output_dir, bootstrap_samples=10)
        self.assertFalse(result["models"]["current-only"]["complete"])
        self.assertEqual(result["additional_comparisons"]["current-only_to_differential"]["paired_n"], 2)
        self.assertFalse(result["visual_diagnostics_complete"])

    def test_reference_reuse_preserves_original_ids_bytes_and_launches_no_completed_roles(self):
        args, identity, old_root, old = self.report_fixture(reference=True)
        before = {str(p): p.read_bytes() for p in old_root.rglob("*") if p.is_file()}
        result, text = driver.build_control_report(args.output_dir, bootstrap_samples=10)
        original = result["models"]["baseline"]
        self.assertEqual((original["origin"], original["newly_rolled_out"], original["source_evaluation_id"]),
            ("reused_reference", 0, old["evaluation_id"]))
        self.assertIn("REUSED", text)
        self.assertTrue(result["visual_diagnostics_complete"])
        with patch.object(driver.subprocess, "Popen") as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.run_evaluation(args, identity, {}), 0)
        launch.assert_not_called()
        self.assertFalse((args.output_dir / "baseline").exists())
        self.assertEqual(before, {str(p): p.read_bytes() for p in old_root.rglob("*") if p.is_file()})
        policy_path = old_root / "baseline/BinFill/policy_manifest.json"
        context = json.loads(policy_path.read_text())
        self.assertEqual(context["evaluation_id"], old["evaluation_id"] + ":baseline")
        context["evaluation_id"] = identity["evaluation_id"] + ":baseline"
        policy_path.write_text(json.dumps(context))
        with self.assertRaises(ValueError):
            driver.build_control_report(args.output_dir, bootstrap_samples=5)
        with patch.object(driver.subprocess, "Popen") as launch, self.assertRaises(ValueError):
            driver.run_evaluation(args, identity, {})
        launch.assert_not_called()

    def test_cross_role_episode_context_identity_mismatch_rejected(self):
        args, _, _, _ = self.report_fixture()
        path = args.output_dir / "current-only/BinFill/simulation_results.csv"
        original = path.read_text()
        for altered in (original.replace("0,100,", "0,999,"),
                        original.replace(",0,same instruction", ",9,same instruction"),
                        original.replace("same instruction", "different instruction")):
            path.write_text(altered)
            with self.assertRaises(ValueError):
                driver.build_control_report(args.output_dir, bootstrap_samples=5)
        path.write_text(original)


if __name__ == "__main__":
    unittest.main()
