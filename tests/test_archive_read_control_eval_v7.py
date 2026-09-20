"""CPU-only archive READ-control identities, reports and driver safety tests."""
import contextlib
import copy
from dataclasses import asdict
import fcntl
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
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.hamlet import checkpoint_identity
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from run_scripts.robomme import eval_archive_read_control_v7 as driver


class ArchiveReadControlEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base, self.archive = self.root / "base", self.root / "archive"
        self.base.mkdir()
        self.archive.mkdir()
        cfg = MemoryV7Config(feature_dim=8, state_dim=4, num_short_tokens=4,
                             hidden_dim=8, num_heads=2, capacity=3)
        targets = sorted("model.transformer_blocks.0.attn1." + name
                         for name in ("to_q", "to_k", "to_v", "to_out.0"))
        base_state = {"action_head." + name + ".weight": torch.zeros(8, 8) for name in targets}
        save_file(base_state, str(self.base / "model.safetensors"))
        (self.base / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {key: "model.safetensors" for key in base_state}}))
        (self.base / "config.json").write_text(json.dumps({
            "hamlet_mode": "finetune", "mem_cond_type": "cross_attn", "memory_type": "moment_token",
            "memory_stride": 16, "n_moment_tokens": 4, "backbone_embedding_dim": 8}))
        (self.base / "processor_config.json").write_text(json.dumps(
            {"processor_kwargs": {"max_state_dim": 4, "max_action_dim": 3}}))
        for name in ("statistics.json", "embodiment_id.json"):
            (self.base / name).write_text("{}")
        expert = {name + suffix: torch.zeros(shape) for name in targets
                  for suffix, shape in ((".lora_A", (2, 8)), (".lora_B", (8, 2)))}
        payloads = {"model.safetensors": RecurrentMemoryV7(cfg).state_dict(),
                    "expert.safetensors": expert, "cvom.safetensors": CVOMV7(cfg).state_dict()}
        for name, state in payloads.items():
            save_file(state, str(self.archive / name))
        self.info = {"format_version": 1, "step": 20,
            "config": {"trainer_variant": "recurrent_memory_v7", "stage": 1, "mode": "archive",
                       "memory": asdict(cfg), "expert": {"rank": 2, "alpha": 4}, "expert_targets": targets,
                       "train": {"mode": "archive", "cvom_threshold": 0.05}},
            "metadata": {"base_model": checkpoint_identity(self.base), "cache_fingerprint": "fixture-cache",
                         "payload_sha256": {name: file_sha256(self.archive / name) for name in payloads}}}
        self.save_info()

    def tearDown(self):
        self.temp.cleanup()

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
        for key in ("memory_checkpoint", "weights_sha256", "expert_weights_sha256", "semantic_state_sha256"):
            self.assertEqual(on[key], off[key])
        self.assertFalse(on["archive_read_off"])
        self.assertTrue(off["archive_read_off"])
        self.assertEqual(on["write_policy"], off["write_policy"])
        for path in (str(driver.SERVER), str(driver.EVALUATOR),
                     "run_scripts/robomme/eval_long_memory_comparison.py",
                     "gr00t/long_memory/online_policy_v7.py", "gr00t/long_memory/recurrent_v7.py"):
            self.assertIn(path, identity["source_sha256"])
        self.assertFalse((self.root / "output").exists())

    def test_untrained_wrong_stage_wrong_mode_and_nonfinite_payload_rejected(self):
        for mutation, error in (({"step": 0}, "step>0"),
                                ({"config": {**self.info["config"], "stage": 2, "mode": "archive"}}, "Stage 2"),
                                ({"config": {**self.info["config"], "mode": "recurrent", "train": {"mode": "recurrent"}}}, "mode=archive")):
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
        with self.assertRaisesRegex(ValueError, "SAME checkpoint"):
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


if __name__ == "__main__":
    unittest.main()
