"""CPU V14 identity, original-baseline dispatch and actual-call evidence guards."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from run_scripts.robomme import eval_visual_expert_v14 as driver
from run_scripts.robomme import checkpoint_visual_expert_v14 as checkpoint
from tests import test_checkpoint_visual_expert_v14 as fixtures
from run_scripts.robomme.baseline_reference_v10 import SERVERS


class VisualExpertEvaluationTests(unittest.TestCase):
    def setUp(self):
        f = self.fixture = fixtures.JointCheckpointTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.root, self.base = f.root, f.base
        f.config["train"]["max_steps"] = 512
        f.config["objective"]["mae_best_is_diagnostic"] = True
        f.metadata["initialization_reference"] = {"path": "v11-reference", "initial_visual_sha256": "d" * 64}
        self.bundle = f.save(step=512)
        self.header = checkpoint.checkpoint_info(self.base, self.bundle)
        # Genuine tiny payload validation happens ABOVE. Only the production
        # 2048-width/32-block/rank8 audited header is mocked for evaluator tests.
        self.header["config"]["visual"]["feature_dim"] = 2048
        self.header["config"]["expert"] = {"rank": 8, "alpha": 16.}
        self.header["config"]["expert_targets"] = list(driver.EXPERT_TARGETS)

    def args(self, *extra):
        return driver.build_parser().parse_args(["--base-model", str(self.base),
            "--visual-checkpoint", str(self.bundle), "--tasks", "BinFill", "--n-episodes", "1",
            "--output-dir", str(self.root / "evaluation"), *extra])

    def identity(self, args=None):
        args = args or self.args()
        driver.validate_options(args)
        with patch.object(checkpoint, "checkpoint_info", return_value=copy.deepcopy(self.header)), \
                patch.object(driver, "benchmark_identity", return_value={"fixture": True}), \
                contextlib.redirect_stdout(io.StringIO()):
            return driver.build_identity(args)

    def test_roles_real_v14_source_closure_and_exact_original_baseline_dispatch(self):
        identity = self.identity()
        self.assertEqual(set(identity["models"]), {"baseline", "visual", "visual-off"})
        self.assertFalse((self.root / "evaluation").exists())
        for role, model in identity["models"].items():
            command = driver.server_command(self.args(), model, 1234)
            expected = driver.BASELINE_SERVER if role == "baseline" else driver.SERVER
            self.assertIn(str(driver.REPO_ROOT / expected), command)
            self.assertEqual("--memory-checkpoint" in command, role != "baseline")
            self.assertEqual("--visual-read-off" in command, role == "visual-off")
            client = driver.client_command(self.args(), model, role, "BinFill", self.root / role, 1234, "evaluation")
            self.assertIn(str(driver.REPO_ROOT / (driver.BASELINE_CLIENT if role == "baseline" else driver.CLIENT)), client)
            self.assertEqual("--include-tail" in client, role != "baseline")
            self.assertNotIn("--archive-read-off", command)
        on, off = (identity["models"][role] for role in ("visual", "visual-off"))
        self.assertEqual(on["expert_weights_sha256"], off["expert_weights_sha256"])
        self.assertEqual(on["memory_checkpoint"], off["memory_checkpoint"])
        self.assertNotIn("frozen_parent", on)
        for path in (driver.SERVER, driver.BASELINE_SERVER, driver.EVALUATOR, driver.CLIENT, *driver.DEPENDENCIES):
            self.assertEqual(identity["source_sha256"][str(path)], driver.file_hash(driver.REPO_ROOT / path))
        self.assertIn("checkpoint_visual_expert_v14.py", " ".join(identity["source_sha256"]))
        self.assertIn("checkpoint_demo_tail_v13.py", " ".join(identity["source_sha256"]))

    def test_expert_payload_role_or_frozen_claim_tampering_fails_even_rehashed_manifest(self):
        original = self.identity()
        mutations = [lambda m: m["models"]["visual-off"].update(visual_read_off=False),
            lambda m: m["models"]["visual"].update(expert_weights_sha256="a" * 64),
            lambda m: m["models"]["visual"].update(expert_targets=driver.EXPERT_TARGETS[:-1]),
            lambda m: m["models"]["visual"].update(expert_config={"rank": 4, "alpha": 16.}),
            lambda m: m["models"]["visual"].update(frozen_parent={}),
            lambda m: m["models"]["visual"]["training_metadata"].update(frozen_parent={}),
            lambda m: m["models"]["visual"].update(architecture="new_reader"),
            lambda m: m["models"]["visual"].update(trainer_variant="visual_demo_tail_v13"),
            lambda m: m["models"]["visual"].update(expert_semantic_state_sha256="bad"),
            lambda m: m["models"]["baseline"].update(server_script=str(driver.SERVER)),
            lambda m: m["models"]["baseline"].update(expert_config={}),
            lambda m: m["models"]["visual-off"].update(memory_checkpoint="another-expert")]
        for index, mutate in enumerate(mutations):
            bad = copy.deepcopy(original); mutate(bad)
            with self.subTest(case=index):
                with self.assertRaisesRegex(ValueError, "digest"):
                    driver.validate_manifest_contract(bad)
                bad["evaluation_id"] = driver._identity_digest(bad)
                with self.assertRaises(ValueError):
                    driver.validate_manifest_contract(bad)

    def test_fixed512_and_explicit_honest_initialization_only(self):
        self.header["step"] = 4
        self.header["training_state"]["window_cursor"] = 4
        with self.assertRaisesRegex(ValueError, "fixed final step512"):
            self.identity()
        self.header["step"] = 0
        self.header["training_state"]["window_cursor"] = 0
        with self.assertRaisesRegex(ValueError, "allow-initialization"):
            self.identity()
        identity = self.identity(self.args("--allow-initialization-checkpoints"))
        self.assertEqual(identity["models"]["visual"]["checkpoint_status"], "INITIALIZATION_SELECTED")
        self.assertIn("zero learned joint updates", identity["models"]["visual"]["description"])
        self.header["config"]["train"]["max_steps"] = 4
        with self.assertRaisesRegex(ValueError, "fixed final step512"):
            self.identity(self.args("--allow-initialization-checkpoints"))

    def test_readonly_preflight_and_resume_identity_do_not_touch_outputs(self):
        identity = self.identity()
        output = self.root / "evaluation"
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

    def records(self, role, model=None):
        model = model or self.identity()["models"][role]
        records = []
        for index, (frame, passive) in enumerate(((0, True), (16, False)), 1):
            enabled = role != "visual-off" and not passive
            info = {"checkpoint_variant": driver.VARIANT, "checkpoint_step": 512, "stage": 1,
                "expert_adapted": True, "visual_read_mode": "differential", "visual_read_off": role == "visual-off",
                "visual_weights_sha256": model["weights_sha256"], "expert_weights_sha256": model["expert_weights_sha256"],
                "initial_parent_checkpoint_sha256": model["initial_parent"]["files_sha256"]["checkpoint.json"],
                "expert_adapter_count": 128, "expert_adapters_enabled": True,
                "long_memory": {"mode": "archive", "policy": "append", "memory_read_enabled": not passive,
                    "observations_seen": index, "write_attempts": index, "updates": index, "keeps": 0,
                    "demo_updates": 1, "demo_keeps": 0, "frame_index": frame, "passive": passive},
                "visual_memory": {"read_enabled": enabled, "observed": index, "append_updates": index,
                    "demo_updates": 1, "bank_observations": index, "bank_tokens": 162 * index,
                    "frame_index": frame, "passive": passive, "image_delta_norm": .3 if enabled else 0.,
                    "image_changed_fraction": .5 if enabled else 0., "read_mode": "differential",
                    "past_read_enabled": enabled, "current_reference_enabled": enabled,
                    "readout_norm": .8 if enabled else 0., "projected_residual_norm": .3 if enabled else 0.,
                    "readout_norm_semantics": "H_past_minus_H_current"},
                "demo_tail": {"enabled": True, "read_enabled": enabled, "tail_observations": 0 if passive else 15,
                    "canonical_prior_observations": 0 if passive else 1,
                    "effective_prior_observations": 0 if passive else 16 if enabled else 1}}
            if not passive:
                info["demo_tail"].update(n_demo=16, last_tail_frame=15)
            records.append({"kind": "policy_call", "session_id": "session", "episode_idx": 0,
                            "episode_seed": 100, "frame_index": frame, "passive": passive, "info": info})
        records.insert(1, {"kind": "demo_tail_ingest", "session_id": "session", "episode_idx": 0,
            "episode_seed": 100, "frames": list(range(1, 16)), "info": {
                "session_id": "session", "episode_seed": 100, "n_demo": 16, "ingested_observations": 15,
                "first_tail_frame": 1, "last_tail_frame": 15, "canonical_observations": 1,
                "canonical_frame": 0, "parent_unchanged": True, "rng_preserved": True, "read_performed": False}})
        records.append({"kind": "episode_complete", "session_id": "session", "episode_idx": 0,
                        "episode_seed": 100, "success": True})
        return model, records

    def diagnostics(self, role, model, records):
        folder = self.root / "journal" / role / "BinFill"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "simulation_results.csv").write_text(
            "episode_idx,episode_seed,success,status,scenario_seed,task_instruction\n0,100,1,success,0,fixture\n")
        (folder / "memory_diagnostics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        return (driver.completed_visual_diagnostics(folder.parent.parent, role, ["BinFill"], 1, model),
                driver.completed_tail_diagnostics(folder.parent.parent, role, ["BinFill"], 1, model))

    def test_both_roles_require_saved_expert_and_real_tail_evidence(self):
        for role in ("visual", "visual-off"):
            model, records = self.records(role)
            visual, tail = self.diagnostics(role, model, records)
            self.assertTrue(visual["complete_evidence"])
            self.assertTrue(tail["complete_evidence"])
            self.assertEqual(visual["read_enabled_calls"], int(role == "visual"))
            self.assertEqual(tail["ingest_rpc"], 1)
            self.assertEqual(tail["tail_observations"], 15)
            self.assertEqual(tail["tail_read_calls"], int(role == "visual"))
            records[2]["info"].pop("expert_weights_sha256")
            self.assertFalse(self.diagnostics(role, model, records)[0]["complete_evidence"])

    def test_wrong_expert_identity_or_disabled_adapters_fail_for_on_and_off(self):
        changes = [lambda row: row["info"].update(expert_weights_sha256="a" * 64),
            lambda row: row["info"].update(expert_adapter_count=127),
            lambda row: row["info"].update(expert_adapters_enabled=False),
            lambda row: row["info"].update(initial_parent_checkpoint_sha256="c" * 64),
            lambda row: row["info"].update(frozen_parent_checkpoint_sha256="c" * 64),
            lambda row: row["info"]["long_memory"].update(memory_read_enabled=False),
            lambda row: row["info"]["visual_memory"].update(image_delta_norm=float("nan"))]
        for role in ("visual", "visual-off"):
            model, original = self.records(role)
            for index, change in enumerate(changes):
                rows = copy.deepcopy(original); change(rows[2])
                with self.subTest(role=role, change=index), self.assertRaises(ValueError):
                    self.diagnostics(role, model, rows)
            with self.assertRaisesRegex(ValueError, "bypass"):
                rows = copy.deepcopy(original)
                rows[0]["info"]["visual_memory"]["readout_norm"] = .1
                self.diagnostics(role, model, rows)

    def test_missing_rpc_and_unmatched_completed_sessions_are_incomplete(self):
        model, rows = self.records("visual-off")
        _, tail = self.diagnostics("visual-off", model, [rows[0], *rows[2:]])
        self.assertFalse(tail["complete_evidence"])
        self.assertEqual(tail["missing_rpc"], 1)
        rows[-1]["episode_seed"] = 999
        visual, tail = self.diagnostics("visual-off", model, rows)
        self.assertFalse(visual["complete_evidence"])
        self.assertFalse(tail["complete_evidence"])

    def test_fatal_report_preserves_error_and_records_driver_failure(self):
        args = self.args("--models", "baseline")
        identity = self.identity(args)
        with patch.object(driver.subprocess, "Popen"), patch.object(driver, "server_ready"), \
                patch.object(driver, "stop_process"), patch.object(driver, "run_client"), \
                patch.object(driver, "write_report", side_effect=ValueError("wrong expert evidence")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "wrong expert evidence"):
                driver.run_evaluation(args, identity, {})
        status = json.loads((self.root / "evaluation/driver_status.json").read_text())
        self.assertEqual(status["fatal"]["error_type"], "ValueError")
        self.assertTrue(status["failures"])

    def write_rows(self, root, manifest, role, values):
        folder = root / role / "BinFill"
        folder.mkdir(parents=True, exist_ok=True)
        settings = manifest["settings"]
        context = {"evaluation_id": manifest["evaluation_id"] + ":" + role, "task_id": "BinFill",
            **{k: settings[k] for k in ("dataset", "seed", "n_action_steps", "max_episode_steps")},
            "model_config_sha256": manifest["base_file_sha256"]["config.json"], "memory_window": 4,
            "demo_sampling": "backward_aligned_full_history", "scenario_metadata_sha256": self.scenario_digest}
        if role != "baseline":
            context.update(client_variant="demo_tail_v13", demo_tail_ingest=True)
        (folder / "policy_manifest.json").write_text(json.dumps(context))
        (folder / "simulation_results.csv").write_text(
            "episode_idx,episode_seed,success,status,scenario_seed,task_instruction\n" + "".join(
                f"{i},{100+i},{v},{'success' if v else 'fail'},{i},same instruction\n" for i, v in enumerate(values)))
        if role != "baseline":
            _, template = self.records(role, manifest["models"][role])
            records = []
            for episode, success in enumerate(values):
                for entry in copy.deepcopy(template):
                    entry.update(episode_idx=episode, episode_seed=100+episode, session_id=f"session-{episode}")
                    if entry["kind"] == "demo_tail_ingest":
                        entry["info"].update(episode_seed=100+episode, session_id=f"session-{episode}")
                    if entry["kind"] == "episode_complete":
                        entry["success"] = bool(success)
                    records.append(entry)
            (folder / "memory_diagnostics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))

    def report_fixture(self, *, reference=False):
        args = self.args("--n-episodes", "3")
        scenario = self.root / "benchmark/env_metadata/val/record_dataset_BinFill_metadata.json"
        scenario.parent.mkdir(parents=True, exist_ok=True)
        records = [{"task": "BinFill", "episode": i, "seed": i, "difficulty": "easy"} for i in range(5)]
        scenario.write_text(json.dumps({"env_id": "BinFill", "record_count": 5, "records": records}, indent=2) + "\n")
        self.scenario_digest = hashlib.sha256(json.dumps([
            {"episode_idx": i, "seed": i, "difficulty": "easy"} for i in range(5)], sort_keys=True).encode()).hexdigest()
        benchmark = {"versions": {"robomme": "fixture"}, "source_and_scenario_sha256": {str(scenario): driver.file_hash(scenario)}}
        driver.validate_options(args)
        def build():
            with patch.object(driver, "benchmark_identity", return_value=benchmark), \
                    patch.object(checkpoint, "checkpoint_info", return_value=copy.deepcopy(self.header)), \
                    contextlib.redirect_stdout(io.StringIO()):
                return driver.build_identity(args)
        identity = build()
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
            identity = build()
        self.assertFalse(args.output_dir.exists())
        driver.bind_manifest(args.output_dir, identity)
        if not reference:
            self.write_rows(args.output_dir, identity, "baseline", [0, 1, 1])
        self.write_rows(args.output_dir, identity, "visual", [1, 1, 0])
        self.write_rows(args.output_dir, identity, "visual-off", [0, 0, 1])
        return args, identity, old_root, old

    def test_complete_report_pairing_and_missing_are_not_task_failures(self):
        args, identity, _, _ = self.report_fixture()
        result, text = driver.build_control_report(args.output_dir, bootstrap_samples=30)
        self.assertTrue(result["visual_diagnostics_complete"])
        self.assertTrue(all(model["complete"] for model in result["models"].values()))
        paired = result["additional_comparisons"]["visual-off_to_visual"]
        self.assertEqual((paired["paired_n"], paired["wins"], paired["losses"]), (3, 2, 1))
        self.assertIn("SAME newly adapted V14 expert", text)
        self.write_rows(args.output_dir, identity, "visual", [1, 1])
        partial, _ = driver.build_control_report(args.output_dir, bootstrap_samples=10)
        self.assertFalse(partial["models"]["visual"]["complete"])
        self.assertFalse(partial["visual_diagnostics_complete"])
        self.assertEqual(partial["additional_comparisons"]["visual-off_to_visual"]["paired_n"], 2)

    def test_baseline_reference_preserves_ids_bytes_no_relaunch_and_rejects_corruption(self):
        args, identity, source, old = self.report_fixture(reference=True)
        before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
        report, text = driver.build_control_report(args.output_dir, bootstrap_samples=10)
        baseline = report["models"]["baseline"]
        self.assertEqual((baseline["origin"], baseline["newly_rolled_out"], baseline["source_evaluation_id"]),
                         ("reused_reference", 0, old["evaluation_id"]))
        self.assertIn("REUSED", text)
        with patch.object(driver.subprocess, "Popen") as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.run_evaluation(args, identity, {}), 0)
        launch.assert_not_called()
        self.assertFalse((args.output_dir / "baseline").exists())
        self.assertEqual(before, {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()})
        path = source / "baseline/BinFill/policy_manifest.json"
        context = json.loads(path.read_text())
        self.assertEqual(context["evaluation_id"], old["evaluation_id"] + ":baseline")
        context["evaluation_id"] = identity["evaluation_id"] + ":baseline"
        path.write_text(json.dumps(context))
        with self.assertRaises(ValueError):
            driver.build_control_report(args.output_dir, bootstrap_samples=5)
        with patch.object(driver.subprocess, "Popen") as launch, self.assertRaises(ValueError):
            driver.run_evaluation(args, identity, {})
        launch.assert_not_called()

    def test_cross_role_episode_seed_scenario_instruction_or_client_mismatch_rejected(self):
        args, _, _, _ = self.report_fixture()
        path = args.output_dir / "visual-off/BinFill/simulation_results.csv"
        original = path.read_text()
        for changed in (original.replace("0,100,", "0,999,"),
                        original.replace(",0,same instruction", ",9,same instruction"),
                        original.replace("same instruction", "different instruction")):
            path.write_text(changed)
            with self.assertRaises(ValueError):
                driver.build_control_report(args.output_dir, bootstrap_samples=5)
        path.write_text(original)
        policy = path.parent / "policy_manifest.json"
        context = json.loads(policy.read_text()); context["client_variant"] = "fake_v14_client"
        policy.write_text(json.dumps(context))
        with self.assertRaisesRegex(ValueError, "client"):
            driver.build_control_report(args.output_dir, bootstrap_samples=5)


if __name__ == "__main__":
    unittest.main()
