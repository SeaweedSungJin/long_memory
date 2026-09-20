"""CPU-only actual-V10 identities, projector provenance and paired driver guards."""
import contextlib
import copy
import fcntl
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from gr00t.long_memory.checkpoint_v4 import file_sha256
from run_scripts.robomme import eval_archive_projector_v10 as driver
from run_scripts.robomme.baseline_reference_v10 import SERVERS


class ArchiveProjectorV10EvaluationTests(unittest.TestCase):
    def setUp(self):
        from tests.test_checkpoint_projector_v10 import ProjectorCheckpointTests
        fixture = ProjectorCheckpointTests("test_header_preflight_exact_toy_shape_and_rng_neutral")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.fixture = fixture
        self.root, self.base = fixture.root, fixture.base
        with torch.no_grad():
            fixture.head.model.proj_out_2.delta_weight.fill_(.05)
            fixture.head.model.proj_out_2.delta_bias.fill_(.1)
        self.archive = fixture.save(step=20)
        self.info = json.loads((self.archive / "checkpoint.json").read_text())

    def save_info(self):
        (self.archive / "checkpoint.json").write_text(json.dumps(self.info))

    def args(self, *extra):
        return driver.build_parser().parse_args([
            "--base-model", str(self.base), "--archive-checkpoint", str(self.archive),
            "--output-dir", str(self.root / "output"), *extra])

    def identity(self, args=None):
        args = args or self.args()
        driver.validate_options(args)
        with patch.object(driver, "benchmark_identity", return_value={"simulator": "CPU-fixture"}), \
                contextlib.redirect_stdout(io.StringIO()):
            return driver.build_identity(args)

    def test_identity_same_weights_explicit_flags_and_both_new_script_hashes(self):
        identity = self.identity()
        self.assertEqual(identity["trainer_variant"], driver.VARIANT)
        self.assertEqual(identity["settings"]["tasks"], list(driver.TASKS))
        on, off = (identity["models"][role] for role in ("archive", "archive-off"))
        for key in ("memory_checkpoint", "weights_sha256", "expert_weights_sha256", "semantic_state_sha256",
                    "projector_weights_sha256", "projector_config", "checkpoint_files_sha256"):
            self.assertEqual(on[key], off[key])
        self.assertFalse(on["archive_read_off"])
        self.assertTrue(off["archive_read_off"])
        self.assertEqual(on["write_policy"], off["write_policy"])
        self.assertTrue(on["projector_config"]["enabled"])
        self.assertTrue(off["projector_config"]["enabled"])
        self.assertIn("projector", on["semantic_state_sha256"])
        self.assertEqual(on["trainer_variant"], "archive_projector_v10")
        self.assertNotIn("projector_config", identity["models"]["baseline"])
        for path in driver.DEPENDENCIES:
            self.assertEqual(identity["source_sha256"][str(path)], driver.file_hash(driver.REPO_ROOT / path))
        for filename in ("policy_archive_projector_v10.py", "checkpoint_projector_v10.py", "projector_adapter_v10.py",
                         "deployment_objective_v9.py", "audit_archive_generation_v7.py"):
            self.assertIn("run_scripts/robomme/" + filename, identity["source_sha256"])
        for path in (str(driver.SERVER), str(driver.EVALUATOR),
                     "run_scripts/robomme/eval_long_memory_comparison.py",
                     "gr00t/long_memory/online_policy_v7.py", "gr00t/long_memory/recurrent_v7.py"):
            self.assertIn(path, identity["source_sha256"])
        self.assertFalse((self.root / "output").exists())

    def test_disabled_training_arm_stays_disabled_in_both_archive_roles(self):
        from run_scripts.robomme.projector_adapter_v10 import set_trainable
        module = self.fixture.head.model.proj_out_2
        with torch.no_grad():
            module.delta_weight.zero_()
            module.delta_bias.zero_()
        module.enabled = module.configured_enabled = False
        set_trainable(self.fixture.head, False)
        self.fixture.config["projector"]["enabled"] = False
        self.archive = self.fixture.save("control", step=20)
        identity = self.identity()
        for name in ("archive", "archive-off"):
            self.assertFalse(identity["models"][name]["projector_config"]["enabled"])
            self.assertNotIn("--projector-off", driver.server_command(self.args(), identity["models"][name], 1234))

    def test_valid_zero_step_bundle_is_diagnostic_only_not_trained_evaluation(self):
        with torch.no_grad():
            self.fixture.head.model.proj_out_2.delta_weight.zero_()
            self.fixture.head.model.proj_out_2.delta_bias.zero_()
        self.archive = self.fixture.save("initial", step=0)
        with self.assertRaisesRegex(ValueError, "step>0"):
            self.identity()

    def test_untrained_wrong_stage_wrong_mode_and_nonfinite_payload_rejected(self):
        for mutation, error in (({"step": 0}, "step>0|step-zero"),
                                ({"config": {**self.info["config"], "stage": 2, "mode": "archive"}}, "Stage 1"),
                                ({"config": {**self.info["config"], "mode": "recurrent", "train": {"mode": "recurrent"}}}, "archive mode")):
            original = copy.deepcopy(self.info)
            self.info.update(mutation)
            self.save_info()
            with self.assertRaisesRegex(ValueError, error):
                self.identity()
            self.info = original
            self.save_info()
        path = self.archive / "model.safetensors"
        state = load_file(str(path))
        state[next(iter(state))].flatten()[0] = float("nan")
        save_file(state, str(path))
        self.info["metadata"]["payload_sha256"][path.name] = file_sha256(path)
        self.save_info()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.identity()

    def test_legacy_v7_checkpoint_and_projector_payload_corruption_rejected(self):
        self.info["config"]["trainer_variant"] = "recurrent_memory_v7"
        self.save_info()
        with self.assertRaisesRegex(ValueError, "archive_projector_v10"):
            self.identity()
        self.info["config"]["trainer_variant"] = "archive_projector_v10"
        self.save_info()
        path = self.archive / "projector.safetensors"
        state = load_file(str(path))
        state["delta_bias"].add_(1)
        save_file(state, str(path))
        with self.assertRaisesRegex(ValueError, "payload changed: projector"):
            self.identity()

    def test_projector_manifest_semantics_hashes_and_baseline_isolation(self):
        original = self.identity()
        changes = [lambda m: m["models"]["archive-off"]["semantic_state_sha256"].pop("projector"),
                   lambda m: m["models"]["archive-off"].update(trainer_variant="recurrent_memory_v7"),
                   lambda m: m["models"]["archive-off"]["projector_config"].update(enabled=False),
                   lambda m: m["models"]["archive-off"]["projector_config"].update(target="model.proj_out_1"),
                   lambda m: m["models"]["archive-off"].update(projector_weights_sha256="a" * 64),
                   lambda m: m["models"]["archive"].update(memory_off=0),
                   lambda m: m["models"]["baseline"].update(projector_config=m["models"]["archive"]["projector_config"])]
        for index, mutate in enumerate(changes):
            changed = copy.deepcopy(original)
            mutate(changed)
            changed["evaluation_id"] = driver._identity_digest(changed)
            with self.subTest(index=index), self.assertRaises(ValueError):
                driver.validate_manifest_contract(changed)

    def test_preflight_refuses_existing_provenance_and_unidentified_contents_readonly(self):
        identity = self.identity()
        output = self.root / "output"
        driver.bind_manifest(output, identity)
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        self.assertEqual(self.identity(), identity)
        with self.assertRaisesRegex(ValueError, "NEW --output-dir"):
            self.identity(self.args("--seed", "7"))
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
        unknown = self.root / "unknown"
        unknown.mkdir()
        (unknown / "unrelated.txt").write_text("keep me")
        with self.assertRaisesRegex(ValueError, "Unidentified"):
            self.identity(self.args("--output-dir", str(unknown)))
        self.assertEqual((unknown / "unrelated.txt").read_text(), "keep me")

    def test_stride_output_scope_and_payload_hash_guards(self):
        with self.assertRaisesRegex(ValueError, "memory_stride"):
            self.identity(self.args("--n-action-steps", "8"))
        with self.assertRaisesRegex(ValueError, "inside a source"):
            self.identity(self.args("--output-dir", str(self.archive / "unsafe")))
        path = self.archive / "expert.safetensors"
        state = load_file(str(path))
        state[next(iter(state))].add_(1)
        save_file(state, str(path))
        with self.assertRaisesRegex(ValueError, "payload changed"):
            self.identity()

    def test_manifest_pair_and_digest_tampering_rejected(self):
        original = self.identity()
        wrong = copy.deepcopy(original)
        wrong["models"]["archive-off"]["expert_weights_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "digest"):
            driver.validate_manifest_contract(wrong)
        wrong["evaluation_id"] = driver._identity_digest(wrong)
        with self.assertRaisesRegex(ValueError, "hash aliases|SAME checkpoint"):
            driver.validate_manifest_contract(wrong)
        wrong = copy.deepcopy(original)
        wrong["models"]["archive-off"]["archive_read_off"] = False
        wrong["evaluation_id"] = driver._identity_digest(wrong)
        with self.assertRaisesRegex(ValueError, "READ-off flag"):
            driver.validate_manifest_contract(wrong)

    def test_server_command_never_uses_legacy_memory_off_or_update_override(self):
        args = self.args()
        identity = self.identity(args)
        for name, model in identity["models"].items():
            command = driver.server_command(args, model, 1234)
            self.assertIn(str(driver.REPO_ROOT / driver.SERVER), command)
            self.assertEqual("--archive-read-off" in command, name == "archive-off")
            self.assertNotIn("--memory-off", command)
            self.assertNotIn("update", command)
            self.assertEqual("--memory-checkpoint" in command, name != "baseline")
            self.assertNotIn("--projector-off", command)
            if name != "baseline":
                self.assertEqual(command[command.index("--memory-checkpoint") + 1], str(self.archive))
                self.assertEqual(command[command.index("--write-policy") + 1], "checkpoint")

    def test_preflight_is_readonly_and_does_not_launch(self):
        output = self.root / "preflight"
        with patch.object(driver, "check_dependencies"), patch.object(driver, "benchmark_identity", return_value={}), \
                patch.object(driver, "run_evaluation") as launch, contextlib.redirect_stdout(io.StringIO()):
            result = driver.main(["--base-model", str(self.base), "--archive-checkpoint", str(self.archive),
                                  "--preflight-only", "--server-python", sys.executable, "--output-dir", str(output)])
        self.assertEqual(result, 0)
        launch.assert_not_called()
        self.assertFalse(output.exists())

    def test_manifest_change_refuses_resume_before_server(self):
        args = self.args("--tasks", "BinFill")
        identity = self.identity(args)
        driver.bind_manifest(self.root / "output", identity)
        changed = copy.deepcopy(identity)
        changed["settings"]["seed"] = 99
        changed["evaluation_id"] = driver._identity_digest(changed)
        with patch.object(driver.subprocess, "Popen") as popen, self.assertRaisesRegex(ValueError, "NEW --output-dir"):
            driver.run_evaluation(args, changed, {})
        popen.assert_not_called()

    def write_rows(self, root, identity, name, task, successes):
        directory = root / name / task
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "policy_manifest.json").write_text(json.dumps({
            "evaluation_id": identity["evaluation_id"] + ":" + name, "task_id": task,
            **{key: identity["settings"][key] for key in ("dataset", "seed", "n_action_steps", "max_episode_steps")}}))
        (directory / "simulation_results.csv").write_text(
            "episode_idx,episode_seed,success,status,scenario_seed,task_instruction\n" + "".join(
                f"{index},{100+index},{success},{'success' if success else 'fail'},{index},same task\n"
                for index, success in enumerate(successes)))

    def report_fixture(self, *, baseline=True):
        args = self.args("--tasks", "BinFill", "PatternLock", "--n-episodes", "3")
        identity = self.identity(args)
        root = self.root / "output"
        driver.bind_manifest(root, identity)
        for role, successes in (("baseline", [0, 1, 1]), ("archive", [1, 1, 1]), ("archive-off", [0, 0, 1])):
            if role != "baseline" or baseline:
                self.write_rows(root, identity, role, "BinFill", successes)
        return root, identity

    def test_report_direct_read_contrast_and_incomplete_is_not_failure(self):
        root, _ = self.report_fixture()
        result, text = driver.write_report(root, bootstrap_samples=30)
        contrast = result["additional_comparisons"]["archive-off_to_archive"]
        self.assertEqual((contrast["paired_n"], contrast["wins"], contrast["losses"]), (3, 2, 0))
        self.assertAlmostEqual(contrast["paired_task_macro_delta"], 2 / 3)
        self.assertFalse(contrast["complete"])
        self.assertIsNone(result["models"]["archive"]["tasks"]["PatternLock"]["success_rate"])
        self.assertEqual(result["storage_diagnostics"]["archive"]["completed_sessions_missing_diagnostics"], 3)
        self.assertIn("INCOMPLETE", text)
        self.assertIn("not a learned writer", text)
        json.dumps(result, allow_nan=False)
        self.assertTrue((root / "comparison_summary.txt").is_file())

    def test_on_off_scenario_mismatch_rejected_even_without_baseline_rows(self):
        root, _ = self.report_fixture(baseline=False)
        path = root / "archive-off/BinFill/policy_manifest.json"
        meta = json.loads(path.read_text())
        meta["scenario_metadata_sha256"] = "different"
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "scenario_metadata_sha256 differs"):
            driver.build_control_report(root, bootstrap_samples=5)

    def test_read_diagnostics_validate_completed_calls_and_preserve_missing_evidence(self):
        root, _ = self.report_fixture()
        path = root / "archive-off/BinFill/memory_diagnostics.jsonl"
        common = {"episode_idx": 0, "episode_seed": 100, "session_id": "complete"}
        diag = {"mode": "archive", "policy": "append", "memory_read_enabled": False,
                "read": {"ae_conditioning_delta_norm": 0., "ae_conditioning_changed_fraction": 0.}}
        records = [
            {**common, "kind": "policy_call", "passive": True,
             "info": {"long_memory": {**diag, "read": {}}}},
            {**common, "kind": "policy_call", "passive": False, "info": {"long_memory": diag}},
            {**common, "kind": "episode_complete", "success": 0},
            {**common, "session_id": "abandoned", "kind": "policy_call", "passive": False,
             "info": {"long_memory": {**diag, "memory_read_enabled": True}}},
        ]
        def save():
            path.write_text("".join(json.dumps(row) + "\n" for row in records) + '{"torn":')
        save()
        result = driver.completed_read_diagnostics(root, "archive-off", ["BinFill"], 3)
        self.assertEqual(result["completed_sessions_with_records"], 1)
        self.assertEqual(result["completed_sessions_missing_records"], 2)
        self.assertEqual(result["passive_calls"], 1)
        self.assertEqual(result["decision_calls_with_metrics"], 1)
        self.assertEqual(result["ignored_torn_final_lines"], 1)
        self.assertEqual(result["max_observed_read_metrics"]["ae_conditioning_delta_norm"], 0.)
        diag["memory_read_enabled"] = True
        save()
        with self.assertRaisesRegex(ValueError, "contradictory"):
            driver.completed_read_diagnostics(root, "archive-off", ["BinFill"], 3)
        diag["memory_read_enabled"] = False
        diag["read"]["ae_conditioning_delta_norm"] = .001
        save()
        with self.assertRaisesRegex(ValueError, "nonzero"):
            driver.completed_read_diagnostics(root, "archive-off", ["BinFill"], 3)
        diag["read"] = {}
        save()
        missing = driver.completed_read_diagnostics(root, "archive-off", ["BinFill"], 3)
        self.assertEqual(missing["decision_calls_missing_metrics"], 1)
        self.assertIsNone(missing["max_observed_read_metrics"]["ae_conditioning_delta_norm"])

    def test_episode_seed_and_policy_identity_mismatch_rejected(self):
        root, _ = self.report_fixture()
        path = root / "archive-off/BinFill/simulation_results.csv"
        path.write_text(path.read_text().replace("0,100,", "0,200,"))
        with self.assertRaisesRegex(ValueError, "episode_seed differs"):
            driver.build_control_report(root, bootstrap_samples=5)
        path.write_text(path.read_text().replace("0,200,", "0,100,"))
        meta_path = root / "archive/BinFill/policy_manifest.json"
        meta = json.loads(meta_path.read_text())
        meta["evaluation_id"] = "some-other-run:archive"
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "identity does not match"):
            driver.build_control_report(root, bootstrap_samples=5)

    def test_completed_read_off_keeps_v10_projector_identity_and_append_writes(self):
        root, identity = self.report_fixture()
        path = root / "archive-off/BinFill/memory_diagnostics.jsonl"
        common = {"episode_idx": 0, "episode_seed": 100, "session_id": "complete"}
        info = {"checkpoint_variant": "archive_projector_v10", "checkpoint_step": 20,
                "projector_enabled": True, "projector_kind": "full_rank_residual_fp32",
                "long_memory": {"mode": "archive", "policy": "append", "memory_read_enabled": False,
                    "observations_seen": 1, "write_attempts": 1, "updates": 1, "keeps": 0,
                    "demo_updates": 0, "demo_keeps": 0,
                    "read": {"ae_conditioning_delta_norm": 0., "ae_conditioning_changed_fraction": 0.}}}
        def save():
            path.write_text(json.dumps({**common, "kind": "policy_call", "passive": False, "info": info}) + "\n"
                            + json.dumps({**common, "kind": "episode_complete", "success": 0}) + "\n")
        save()
        result, _ = driver.build_control_report(root, bootstrap_samples=5)
        reads = result["read_diagnostics"]["archive-off"]
        self.assertEqual(reads["calls_with_projector_identity"], 1)
        self.assertEqual(reads["calls_missing_projector_identity"], 0)
        info["projector_enabled"] = False
        save()
        with self.assertRaisesRegex(ValueError, "projector identity differs"):
            driver.build_control_report(root, bootstrap_samples=5)
        info["projector_enabled"] = True
        info["long_memory"].update(updates=0, keeps=1)
        save()
        with self.assertRaisesRegex(ValueError, "APPEND"):
            driver.build_control_report(root, bootstrap_samples=5)
        info["long_memory"].update(updates=1, keeps=0)
        info.pop("projector_enabled")
        save()
        result, _ = driver.build_control_report(root, bootstrap_samples=5)
        self.assertEqual(result["read_diagnostics"]["archive-off"]["calls_missing_projector_identity"], 1)

    def test_server_failure_keeps_results_incomplete_and_resume_skips_completed(self):
        args = self.args("--models", "baseline", "--tasks", "BinFill", "--n-episodes", "1")
        identity = self.identity(args)
        with patch.object(driver.subprocess, "Popen") as popen, \
                patch.object(driver, "server_ready", side_effect=RuntimeError("load failed")), \
                patch.object(driver, "stop_process") as stop, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(driver.run_evaluation(args, identity, {}), 1)
        stop.assert_called_once_with(popen.return_value)
        root = self.root / "output"
        result = json.loads((root / "comparison_summary.json").read_text())
        self.assertIsNone(result["models"]["baseline"]["tasks"]["BinFill"]["success_rate"])
        self.write_rows(root, identity, "baseline", "BinFill", [0])
        with patch.object(driver.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.run_evaluation(args, identity, {}), 0)
        popen.assert_not_called()

    def test_report_only_no_dependencies_and_active_lock_rejected(self):
        root, _ = self.report_fixture()
        with patch.object(driver, "check_dependencies") as deps, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(["--report-only", "--output-dir", str(root)]), 0)
        deps.assert_not_called()
        with (root / ".driver.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Evaluation is running"):
                driver.main(["--report-only", "--output-dir", str(root)])

    def test_cli_guards_and_help(self):
        for argv in (["--models", "archive"], ["--models", "baseline", "archive"],
                     ["--models", "baseline", "baseline"], ["--models", "baseline", "--n-episodes", "51"],
                     ["--models", "baseline", "--tasks", "all", "BinFill"],
                     ["--models", "baseline", "--task-timeout", "nan"],
                     ["--models", "baseline", "--preflight-only", "--report-only"]):
            with self.assertRaises(ValueError):
                driver.validate_options(driver.build_parser().parse_args(argv))
        result = subprocess.run([sys.executable, str(driver.REPO_ROOT / driver.EVALUATOR), "--help"],
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def reference_fixture(self, variant="archive_read_control_v7"):
        args = self.args("--tasks", "BinFill", "PatternLock", "--n-episodes", "3")
        driver.validate_options(args)
        benchmark = {"versions": {"robomme": "fixture"}, "source_and_scenario_sha256": {}}
        self.reference_scenario_digests = {}
        directory = self.root / "benchmark/env_metadata/val"
        directory.mkdir(parents=True, exist_ok=True)
        for task in args.tasks:
            path = directory / f"record_dataset_{task}_metadata.json"
            # Real benchmark envelope, deliberately pretty printed and with more
            # available scenarios than the three requested rollout episodes.
            records = [{"task": task, "episode": index, "seed": index, "difficulty": "easy"}
                       for index in range(5)]
            path.write_text(json.dumps({"env_id": task, "record_count": 5, "records": records}, indent=2) + "\n")
            benchmark["source_and_scenario_sha256"][str(path)] = driver.file_hash(path)
            scenarios = [{"episode_idx": index, "seed": index, "difficulty": "easy"} for index in range(5)]
            self.reference_scenario_digests[task] = hashlib.sha256(json.dumps(scenarios, sort_keys=True).encode()).hexdigest()
        with patch.object(driver, "benchmark_identity", return_value=benchmark), contextlib.redirect_stdout(io.StringIO()):
            current = driver.build_identity(args)
        reference = copy.deepcopy(current)
        reference["trainer_variant"] = variant
        reference["models"] = {"baseline": reference["models"]["baseline"]}
        server, digest = SERVERS[variant]
        if variant != driver.VARIANT:
            reference["source_sha256"].pop(str(driver.SERVER))
        reference["source_sha256"][server] = digest
        reference["evaluation_id"] = driver._identity_digest(reference)
        root = self.root / "reference"
        driver.bind_manifest(root, reference)
        (root / "driver_status.json").write_text(json.dumps({"interrupted": False, "failures": []}))
        for task in args.tasks:
            self.write_rows(root, reference, "baseline", task, [0, 1, 1])
            self.add_context(root, reference, "baseline", task)
        args.baseline_reference = root
        return args, root, reference, benchmark

    def add_context(self, root, identity, role, task):
        path = root / role / task / "policy_manifest.json"
        policy = json.loads(path.read_text())
        policy.update(model_config_sha256=identity["base_file_sha256"]["config.json"], memory_window=4,
            demo_sampling="backward_aligned_full_history",
            scenario_metadata_sha256=self.reference_scenario_digests[task])
        path.write_text(json.dumps(policy))

    def reference_identity(self, args, benchmark):
        with patch.object(driver, "benchmark_identity", return_value=benchmark), contextlib.redirect_stdout(io.StringIO()):
            return driver.build_identity(args)

    def test_reference_preflight_is_readonly_and_preserves_original_ids(self):
        args, root, original, benchmark = self.reference_fixture()
        before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        with patch.object(driver, "benchmark_identity", return_value=benchmark), patch.object(driver, "check_dependencies"), \
                patch.object(driver, "run_evaluation") as launch, contextlib.redirect_stdout(io.StringIO()):
            result = driver.main(["--base-model", str(self.base), "--archive-checkpoint", str(self.archive),
                "--baseline-reference", str(root), "--tasks", "BinFill", "PatternLock", "--n-episodes", "3",
                "--server-python", sys.executable, "--output-dir", str(args.output_dir), "--preflight-only"])
        self.assertEqual(result, 0)
        launch.assert_not_called()
        self.assertFalse(args.output_dir.exists())
        identity = self.reference_identity(args, benchmark)
        ref = identity["baseline_reference"]
        self.assertEqual((ref["source_evaluation_id"], ref["completed_episodes"], ref["newly_rolled_out"]),
                         (original["evaluation_id"], 6, 0))
        self.assertEqual(len(ref["files_sha256"]), 6)
        self.assertNotEqual(identity["evaluation_id"], original["evaluation_id"])
        self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()})
        raw_hashes = set(benchmark["source_and_scenario_sha256"].values())
        for task in args.tasks:
            self.assertNotIn(self.reference_scenario_digests[task], raw_hashes)

    def test_reference_scenario_raw_bytes_and_resolved_digest_are_separate_guards(self):
        args, root, _, benchmark = self.reference_fixture()
        identity = self.reference_identity(args, benchmark)
        driver.bind_manifest(args.output_dir, identity)
        path = Path(next(iter(benchmark["source_and_scenario_sha256"])))
        original = path.read_text()
        # Reformatting alone leaves scenarios identical but violates pinned raw
        # benchmark bytes, including on report-only/resume after binding.
        path.write_text(json.dumps(json.loads(original), separators=(",", ":")))
        with self.assertRaisesRegex(ValueError, "raw file changed"):
            driver.build_control_report(args.output_dir, bootstrap_samples=5)
        path.write_text(original)
        policy_path = root / "baseline/BinFill/policy_manifest.json"
        policy = json.loads(policy_path.read_text())
        semantic = policy["scenario_metadata_sha256"]
        for wrong in (benchmark["source_and_scenario_sha256"][str(path)],
                      hashlib.sha256(json.dumps([{"episode_idx": i, "seed": i, "difficulty": "easy"}
                                                 for i in range(3)], sort_keys=True).encode()).hexdigest()):
            # Neither a raw-file digest nor a digest of only the evaluated
            # prefix is the original rollout's all-available-scenarios digest.
            policy["scenario_metadata_sha256"] = wrong
            policy_path.write_text(json.dumps(policy))
            with self.assertRaisesRegex(ValueError, "scenario metadata differs"):
                driver.build_control_report(args.output_dir, bootstrap_samples=5)
        policy["scenario_metadata_sha256"] = semantic
        policy_path.write_text(json.dumps(policy))
        csv = root / "baseline/BinFill/simulation_results.csv"
        csv.write_text(csv.read_text().replace(",0,same task", ",999,same task"))
        with self.assertRaisesRegex(ValueError, "result scenario seed differs"):
            driver.build_control_report(args.output_dir, bootstrap_samples=5)

    def test_reference_complete_run_skips_baseline_and_creates_no_baseline_artifacts(self):
        args, root, original, benchmark = self.reference_fixture()
        identity = self.reference_identity(args, benchmark)
        output = args.output_dir
        driver.bind_manifest(output, identity)
        before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        for role in ("archive", "archive-off"):
            for task in args.tasks:
                self.write_rows(output, identity, role, task, [1, 1, 1] if role == "archive" else [0, 0, 1])
                self.add_context(output, identity, role, task)
        with patch.object(driver.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.run_evaluation(args, identity, {}), 0)
        popen.assert_not_called()
        self.assertFalse((output / "baseline").exists())
        result = json.loads((output / "comparison_summary.json").read_text())
        base = result["models"]["baseline"]
        self.assertEqual((base["origin"], base["newly_rolled_out"], base["source_evaluation_id"]),
                         ("reused_reference", 0, original["evaluation_id"]))
        self.assertTrue(all(m["complete"] for m in result["models"].values()))
        self.assertEqual(result["models"]["archive"]["newly_rolled_out"], 6)
        self.assertEqual((result["comparisons"]["archive"]["paired_n"], result["comparisons"]["archive"]["wins"]), (6, 2))
        self.assertIn("baseline: REUSED", (output / "comparison_summary.txt").read_text())
        self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_reference_hash_corruption_rejected_on_report_and_resume_before_server(self):
        args, root, _, benchmark = self.reference_fixture()
        identity = self.reference_identity(args, benchmark)
        driver.bind_manifest(args.output_dir, identity)
        path = root / "baseline/BinFill/simulation_results.csv"
        path.write_text(path.read_text().replace("0,100,0,fail", "0,100,1,success"))
        with self.assertRaisesRegex(ValueError, "changed since evaluation binding"):
            driver.build_control_report(args.output_dir, bootstrap_samples=5)
        with patch.object(driver.subprocess, "Popen") as popen, self.assertRaises(ValueError):
            driver.run_evaluation(args, identity, {})
        popen.assert_not_called()
        with self.assertRaisesRegex(ValueError, "NEW --output-dir"):
            self.reference_identity(args, benchmark)

    def test_reference_incomplete_active_nested_or_adapted_sources_rejected(self):
        args, root, original, benchmark = self.reference_fixture()
        csv = root / "baseline/BinFill/simulation_results.csv"
        content = csv.read_text()
        csv.write_text("\n".join(content.splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.reference_identity(args, benchmark)
        csv.write_text(content)
        with (root / ".driver.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "still running"):
                self.reference_identity(args, benchmark)
        for mutate in (lambda m: m.update(baseline_reference={"kind": "reference"}),
                       lambda m: m["models"]["baseline"].update(memory_checkpoint="adapted-checkpoint")):
            bad = copy.deepcopy(original)
            mutate(bad)
            bad["evaluation_id"] = driver._identity_digest(bad)
            (root / "comparison_manifest.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                self.reference_identity(args, benchmark)
        self.assertFalse(args.output_dir.exists())

    def test_reference_strict_settings_base_runtime_benchmark_source_and_server_equivalence(self):
        args, root, original, benchmark = self.reference_fixture()
        changes = [lambda m: m["settings"].update(seed=7),
                   lambda m: m["settings"].update(n_episodes=4),
                   lambda m: m["settings"].update(tasks=["BinFill"]),
                   lambda m: m["base_file_sha256"].update({"config.json": "0" * 64}),
                   lambda m: m["policy_package_versions"].update(torch="different"),
                   lambda m: m.update(server_python="different-python"),
                   lambda m: m["benchmark"].update(versions={"robomme": "different"}),
                   lambda m: m["source_sha256"].update({"gr00t/long_memory/online_policy_v7.py": "0" * 64}),
                   lambda m: m["source_sha256"].update({SERVERS["archive_read_control_v7"][0]: "0" * 64})]
        for index, mutate in enumerate(changes):
            bad = copy.deepcopy(original)
            mutate(bad)
            bad["evaluation_id"] = driver._identity_digest(bad)
            (root / "comparison_manifest.json").write_text(json.dumps(bad))
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.reference_identity(args, benchmark)
        self.assertFalse(args.output_dir.exists())

    def test_reference_valid_v10_origin_and_missing_current_rows_are_not_failures(self):
        args, _, _, benchmark = self.reference_fixture(variant=driver.VARIANT)
        identity = self.reference_identity(args, benchmark)
        driver.bind_manifest(args.output_dir, identity)
        result, report = driver.build_control_report(args.output_dir, bootstrap_samples=5)
        self.assertTrue(result["models"]["baseline"]["complete"])
        self.assertFalse(result["models"]["archive"]["complete"])
        self.assertIsNone(result["models"]["archive"]["tasks"]["BinFill"]["success_rate"])
        self.assertIn("REUSED", report)
        (args.output_dir / "baseline").mkdir()
        with self.assertRaisesRegex(ValueError, "copied/local"):
            driver.build_control_report(args.output_dir, bootstrap_samples=5)


if __name__ == "__main__":
    unittest.main()
