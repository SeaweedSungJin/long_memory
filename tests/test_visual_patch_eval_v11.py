"""CPU provenance/dispatch and completed-session visual evidence regressions."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from run_scripts.robomme import eval_visual_patch_v11 as driver
from run_scripts.robomme import checkpoint_visual_patch_v11 as checkpoint
from tests import test_checkpoint_visual_patch_v11 as fixtures


class VisualEvaluationTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.VisualCheckpointTests("test_roundtrip_external_parent_immutable_and_readonly_rng")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.fixture = fixture
        self.root, self.base = fixture.root, fixture.base
        self.bundle = fixture.save(step=1)
        # Only the evaluator's audited 2048-width header check is mocked: tiny
        # serializer/payload validation is separately REAL before this override.
        self.header = checkpoint.checkpoint_info(self.base, self.bundle)
        self.header["config"]["visual"]["feature_dim"] = 2048

    def args(self, *extra):
        return driver.build_parser().parse_args([
            "--base-model", str(self.base), "--visual-checkpoint", str(self.bundle),
            "--tasks", "BinFill", "--n-episodes", "1", "--output-dir", str(self.root / "eval"), *extra])

    def identity(self, args=None):
        args = args or self.args()
        driver.validate_options(args)
        with patch.object(checkpoint, "checkpoint_info", return_value=self.header), \
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
            self.assertNotIn("--memory-off", command)
            self.assertNotIn("--archive-read-off", command)
        for name in (driver.SERVER, driver.BASELINE_SERVER, driver.EVALUATOR, *driver.DEPENDENCIES):
            self.assertEqual(identity["source_sha256"][str(name)], driver.file_hash(driver.REPO_ROOT / name))
        self.assertEqual(identity["models"]["visual"]["frozen_parent"],
                         identity["models"]["visual-off"]["frozen_parent"])

    def test_role_parent_payload_and_digest_tampering_rejected(self):
        original = self.identity()
        changes = [lambda m: m["models"]["visual-off"].update(visual_read_off=False),
                   lambda m: m["models"]["visual"].update(archive_read_off=True),
                   lambda m: m["models"]["visual"].update(memory_off=0),
                   lambda m: m["models"]["visual"].update(weights_sha256="a" * 64),
                   lambda m: m["models"]["visual"]["frozen_parent"].update(step=9108),
                   lambda m: m["models"]["visual"]["visual_config"].update(hidden_dim=7),
                   lambda m: m["models"]["baseline"].update(server_script=str(driver.SERVER)),
                   lambda m: m["models"]["baseline"].update(visual_config={}),
                   lambda m: m["models"]["visual"].update(step=0),
                   lambda m: m["models"]["visual"].update(semantic_state_sha256="bad")]
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
                "--visual-checkpoint", str(self.bundle), "--output-dir", str(self.root / "absent")]), 0)
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

    def records(self, role):
        model = self.identity()["models"][role]
        records = []
        for i, (frame, passive) in enumerate(((0, True), (16, False)), 1):
            enabled = role == "visual" and not passive and i > 1
            visual = {"read_enabled": enabled, "observed": i, "append_updates": i,
                      "demo_updates": 1, "bank_observations": i, "bank_tokens": i * 162,
                      "frame_index": frame, "passive": passive,
                      "image_delta_norm": .3 if enabled else 0.,
                      "image_changed_fraction": .5 if enabled else 0.}
            info = {"checkpoint_variant": driver.VARIANT, "checkpoint_step": model["step"],
                    "stage": 1, "expert_adapted": True,
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
        for role in ("visual", "visual-off"):
            model, records = self.records(role)
            result = self.diagnostics(role, model, records)
            self.assertTrue(result["complete_evidence"])
            self.assertEqual(result["append_updates"], 2)
            self.assertEqual(result["demo_updates"], 1)
            self.assertEqual(result["read_enabled_calls"], int(role == "visual"))
            self.assertEqual(result["completed_sessions"], 1)

    def test_abandoned_retry_and_torn_final_not_counted_as_complete(self):
        model, records = self.records("visual")
        abandoned = copy.deepcopy(records[0])
        abandoned.update(session_id="abandoned", frame_index=-99, info={})
        result = self.diagnostics("visual", model, [abandoned, *records], '{"torn"')
        self.assertTrue(result["complete_evidence"])
        self.assertEqual(result["calls"], 2)
        self.assertEqual(result["ignored_torn_final_lines"], 1)
        records[-1]["episode_seed"] = 999
        self.assertFalse(self.diagnostics("visual", model, records)["complete_evidence"])

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
        model, original = self.records("visual")
        changes = [lambda r: r[1]["info"].update(checkpoint_step=99),
                   lambda r: r[1].update(frame_index=0),
                   lambda r: r[1]["info"]["visual_memory"].update(bank_tokens=162),
                   lambda r: r[1]["info"]["visual_memory"].update(read_enabled=False),
                   lambda r: r[1]["info"]["long_memory"].update(memory_read_enabled=False),
                   lambda r: r[1]["info"]["long_memory"].update(observations_seen=1),
                   lambda r: r[1]["info"].update(expert_adapted=False),
                   lambda r: r[0]["info"]["visual_memory"].update(image_delta_norm=.1),
                   lambda r: r[1]["info"]["visual_memory"].update(image_changed_fraction=float("nan"))]
        for i, change in enumerate(changes):
            records = copy.deepcopy(original)
            change(records)
            with self.subTest(case=i), self.assertRaises(ValueError):
                self.diagnostics("visual", model, records)
        with self.assertRaisesRegex(ValueError, "Malformed"):
            self.diagnostics("visual", model, original, '{"torn"\n')


if __name__ == "__main__":
    unittest.main()
