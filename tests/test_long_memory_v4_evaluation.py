"""CPU-only v4 adapted-expert loading/provenance/paired-report regressions."""
import contextlib
import csv
from dataclasses import asdict
import fcntl
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from safetensors.torch import load_file, save_file
import torch
from torch import nn

from gr00t.eval.sim.robomme.compare_long_memory_v4_results import build_v4_report
from gr00t.long_memory.checkpoint_v4 import (
    _state_sha256, file_sha256, reader_state_sha256, v4_checkpoint_info,
)
from gr00t.long_memory.core_v3 import ActionValueMemory, MemoryV3Config
from gr00t.long_memory.expert_v4 import LoRAConfig
from gr00t.long_memory.hamlet import checkpoint_identity
from run_scripts.robomme import eval_long_memory_v4 as driver

ROOT = Path(__file__).resolve().parents[1]


class V4EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.config = MemoryV3Config(feature_dim=8, state_dim=4, action_dim=3,
                                    hidden_dim=8, num_heads=2, capacity=3, min_fill=1)
        config = {"hamlet_mode": "finetune", "mem_cond_type": "cross_attn", "memory_type": "moment_token",
                  "memory_stride": 16, "n_moment_tokens": 4, "backbone_embedding_dim": 8}
        (self.base / "config.json").write_text(json.dumps(config))
        (self.base / "processor_config.json").write_text(json.dumps(
            {"processor_kwargs": {"max_state_dim": 4, "max_action_dim": 3}}))
        self.targets = sorted("model.transformer_blocks.0.attn1." + name
                              for name in ("to_q", "to_k", "to_v", "to_out.0"))
        weights = {"action_head." + name + ".weight": torch.zeros(8, 8) for name in self.targets}
        save_file(weights, str(self.base / "model.safetensors"))
        (self.base / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {key: "model.safetensors" for key in weights}}))
        for name in ("statistics.json", "embodiment_id.json"):
            (self.base / name).write_text("{}")
        self.reader = self.make_checkpoint("reader", stage=1)
        self.memory = self.make_checkpoint("memory", stage=2)

    def tearDown(self):
        self.temp.cleanup()

    def make_checkpoint(self, name, *, stage, fingerprint="cache-A", reader_mode="memory"):
        path = self.root / name
        path.mkdir()
        memory = ActionValueMemory(self.config).state_dict()
        expert = {name + suffix: torch.zeros(shape) for name in self.targets
                  for suffix, shape in ((".lora_A", (2, 8)), (".lora_B", (8, 2)))}
        save_file(memory, str(path / "model.safetensors"))
        save_file(expert, str(path / "expert.safetensors"))
        metadata = {"base_model": checkpoint_identity(self.base), "cache_fingerprint": fingerprint,
                    "memory_sha256": file_sha256(path / "model.safetensors"),
                    "expert_sha256": file_sha256(path / "expert.safetensors")}
        if stage == 2:
            metadata.update(stage1_parent={"path": "/unmounted/parent",
                            **{key: "a" * 64 for key in ("checkpoint_sha256", "memory_sha256", "expert_sha256")}},
                            frozen_reader_sha256=reader_state_sha256(memory),
                            frozen_expert_sha256=_state_sha256(expert))
        (path / "checkpoint.json").write_text(json.dumps({"format_version": 1, "step": 20,
            "config": {"trainer_variant": "action_expert_v4", "stage": stage, "memory": asdict(self.config),
                       "expert": asdict(LoRAConfig(rank=2)), "expert_targets": self.targets,
                       "train": {"reader_mode": reader_mode}}, "metadata": metadata}))
        return path

    def update_info(self, path, mutation):
        target = path / "checkpoint.json"
        info = json.loads(target.read_text())
        mutation(info)
        target.write_text(json.dumps(info))

    def args(self, *extra):
        return driver.build_parser().parse_args(["--base-model", str(self.base),
            "--reader-checkpoint", str(self.reader), "--memory-checkpoint", str(self.memory),
            "--output-dir", str(self.root / "output"), *extra])

    def identity(self, args=None):
        args = args or self.args()
        driver.validate_options(args)
        with patch.object(driver, "benchmark_identity", return_value={"simulator": "fake"}), contextlib.redirect_stdout(io.StringIO()):
            return driver.build_identity(args)

    def test_identity_hashes_both_components_and_reused_dependencies(self):
        identity = self.identity(self.args("--models", "baseline", "reader", "memory", "fifo", "expert-only"))
        self.assertEqual(identity["trainer_variant"], "action_expert_v4")
        models = identity["models"]
        self.assertIsNone(models["baseline"]["memory_checkpoint"])
        self.assertEqual(models["memory"]["expert_weights_sha256"], models["fifo"]["expert_weights_sha256"])
        self.assertEqual(models["memory"]["weights_sha256"], models["fifo"]["weights_sha256"])
        self.assertEqual(models["reader"]["expert_weights_sha256"], models["expert-only"]["expert_weights_sha256"])
        self.assertEqual(models["expert-only"]["write_policy"], "none")
        self.assertIn("NOT a trained", models["expert-only"]["description"])
        self.assertIn("gr00t/long_memory/expert_v4.py", identity["source_sha256"])
        self.assertIn("gr00t/long_memory/online_policy.py", identity["source_sha256"])
        self.assertIn("run_scripts/robomme/eval_long_memory_comparison.py", identity["source_sha256"])
        self.assertFalse((self.root / "output").exists())

    def test_adapter_change_changes_identity_even_with_same_memory(self):
        before = self.identity()
        state = load_file(str(self.reader / "expert.safetensors"))
        state[next(iter(state))].add_(0.1)
        save_file(state, str(self.reader / "expert.safetensors"))
        self.update_info(self.reader, lambda info: info["metadata"].update(
            expert_sha256=file_sha256(self.reader / "expert.safetensors")))
        after = self.identity()
        self.assertNotEqual(before["evaluation_id"], after["evaluation_id"])
        self.assertEqual(before["models"]["reader"]["weights_sha256"], after["models"]["reader"]["weights_sha256"])

    def test_preflight_rejects_old_missing_nonfinite_or_wrong_expert(self):
        original = (self.reader / "checkpoint.json").read_text()
        self.update_info(self.reader, lambda info: info["config"].update(trainer_variant="action_value_v3"))
        with self.assertRaisesRegex(ValueError, "action_expert_v4"):
            self.identity()
        (self.reader / "checkpoint.json").write_text(original)
        state = load_file(str(self.reader / "expert.safetensors"))
        first = next(iter(state))
        state[first].flatten()[0] = float("nan")
        save_file(state, str(self.reader / "expert.safetensors"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.identity()
        state[first] = torch.zeros(99)
        save_file(state, str(self.reader / "expert.safetensors"))
        with self.assertRaisesRegex(ValueError, "shape"):
            self.identity()
        (self.reader / "expert.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            self.identity()

    def test_stage2_frozen_weights_validated_without_parent_directory(self):
        info = v4_checkpoint_info(self.base, self.memory, expected_stage=2)
        self.assertEqual(info["metadata"]["stage1_parent"]["path"], "/unmounted/parent")
        state = load_file(str(self.memory / "expert.safetensors"))
        state[next(iter(state))].add_(0.1)
        save_file(state, str(self.memory / "expert.safetensors"))
        self.update_info(self.memory, lambda info: info["metadata"].update(
            expert_sha256=file_sha256(self.memory / "expert.safetensors")))
        with self.assertRaisesRegex(ValueError, "frozen Action Expert"):
            self.identity(self.args("--models", "baseline", "memory"))

    def test_reader_mode_none_is_disabled_and_labeled_as_training_control(self):
        self.update_info(self.reader, lambda info: info["config"]["train"].update(reader_mode="none"))
        identity = self.identity()
        self.assertEqual(identity["models"]["reader"]["write_policy"], "none")
        self.assertIn("trained without", identity["models"]["reader"]["description"])

    def test_cohorts_cannot_mix_cache_or_architecture(self):
        self.update_info(self.memory, lambda info: info["metadata"].update(cache_fingerprint="different"))
        with self.assertRaisesRegex(ValueError, "share training cache"):
            self.identity(self.args("--models", "baseline", "reader", "memory"))

    def test_launches_new_server_with_exact_ablation_flags(self):
        args = self.args("--models", "baseline", "reader", "memory", "fifo", "expert-only")
        identity = self.identity(args)
        command = driver.server_command(args, identity["models"]["expert-only"], 1234)
        self.assertIn(str(ROOT / "run_scripts/robomme/serve_long_memory_v4.py"), command)
        self.assertIn("--expert-only", command)
        self.assertEqual(command[command.index("--memory-checkpoint") + 1], str(self.reader))
        command = driver.server_command(args, identity["models"]["fifo"], 1234)
        self.assertEqual(command[command.index("--write-policy") + 1], "all")
        self.assertNotIn("--expert-only", command)
        self.assertNotIn("--memory-checkpoint", driver.server_command(args, identity["models"]["baseline"], 1234))

    def test_readonly_preflight_baseline_needs_no_addon_or_output(self):
        arguments = ["--models", "baseline", "--base-model", str(self.base), "--preflight-only",
                     "--output-dir", str(self.root / "preflight"), "--server-python", sys.executable]
        with patch.object(driver, "check_dependencies"), patch.object(driver, "benchmark_identity", return_value={}), \
                patch.object(driver, "run_evaluation") as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(arguments), 0)
        launch.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())

    def test_server_failure_is_incomplete_not_fake_zero_and_logs_saved(self):
        args = self.args("--models", "baseline", "--tasks", "BinFill", "--n-episodes", "1")
        identity = self.identity(args)
        with patch.object(driver.subprocess, "Popen") as popen, \
                patch.object(driver, "server_ready", side_effect=RuntimeError("load failure")), \
                patch.object(driver, "stop_process") as stop, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(driver.run_evaluation(args, identity, {}), 1)
        stop.assert_called_once_with(popen.return_value)
        folder = self.root / "output"
        task = json.loads((folder / "comparison_summary.json").read_text())["models"]["baseline"]["tasks"]["BinFill"]
        self.assertIsNone(task["success_rate"])
        self.assertEqual(task["completed"], 0)
        self.assertTrue((folder / "baseline/server.log").is_file())
        self.assertIn("load failure", (folder / "driver_status.json").read_text())

    def test_completed_episode_resume_never_starts_policy_server(self):
        args = self.args("--models", "baseline", "--tasks", "BinFill", "--n-episodes", "1")
        identity = self.identity(args)
        output = self.root / "output"
        driver.bind_manifest(output, identity)
        task = output / "baseline/BinFill"
        task.mkdir(parents=True)
        (task / "simulation_results.csv").write_text("episode_idx,episode_seed,success\n0,100,0\n")
        (task / "policy_manifest.json").write_text(json.dumps({"evaluation_id": identity["evaluation_id"] + ":baseline",
            "task_id": "BinFill", **{key: identity["settings"][key]
                                     for key in ("dataset", "seed", "n_action_steps", "max_episode_steps")}}))
        with patch.object(driver.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.run_evaluation(args, identity, {}), 0)
        popen.assert_not_called()

    def report_fixture(self):
        folder = self.root / "report"
        folder.mkdir()
        manifest = {"trainer_variant": "action_expert_v4", "evaluation_id": "example", "models":
                    {name: {} for name in ("baseline", "reader", "memory", "fifo", "expert-only")},
                    "settings": {"tasks": ["BinFill", "PatternLock"], "n_episodes": 3,
                    "dataset": "test", "seed": 6, "n_action_steps": 16, "max_episode_steps": 1300}}
        (folder / "comparison_manifest.json").write_text(json.dumps(manifest))
        for name, outcomes in (("baseline", [0, 0, 1]), ("reader", [0, 1, 1]),
                               ("memory", [1, 0, 1]), ("fifo", [0, 0, 1]), ("expert-only", [0, 0, 0])):
            task_dir = folder / name / "BinFill"
            task_dir.mkdir(parents=True)
            (task_dir / "policy_manifest.json").write_text(json.dumps({"evaluation_id": "example:" + name,
                "task_id": "BinFill", **{key: value for key, value in manifest["settings"].items()
                                         if key in ("dataset", "seed", "n_action_steps", "max_episode_steps")}}))
            with (task_dir / "simulation_results.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["episode_idx", "episode_seed", "success", "status"])
                writer.writeheader()
                for episode, success in enumerate(outcomes):
                    writer.writerow({"episode_idx": episode, "episode_seed": episode + 100,
                                     "success": success, "status": "success" if success else "fail"})
        return folder

    def test_report_distinguishes_expert_ablation_and_storage_contrasts(self):
        result, text = build_v4_report(self.report_fixture(), bootstrap_samples=20)
        contrast = result["additional_comparisons"]["fifo_to_memory"]
        self.assertEqual((contrast["paired_n"], contrast["wins"], contrast["losses"]), (3, 1, 0))
        contrast = result["additional_comparisons"]["expert-only_to_reader"]
        self.assertEqual((contrast["paired_n"], contrast["wins"], contrast["losses"]), (3, 2, 0))
        self.assertFalse(contrast["complete"])
        self.assertNotIn("expert-only", result["storage_diagnostics"])
        self.assertIsNone(result["models"]["reader"]["tasks"]["PatternLock"]["success_rate"])
        self.assertIn("NOT a separately trained", text)
        json.dumps(result, allow_nan=False)

    def test_report_only_no_dependencies_and_driver_lock(self):
        folder = self.report_fixture()
        with patch.object(driver, "check_dependencies") as dependencies, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(["--report-only", "--output-dir", str(folder)]), 0)
        dependencies.assert_not_called()
        with (folder / ".driver.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Evaluation is running"):
                driver.main(["--report-only", "--output-dir", str(folder)])

    def test_entrypoints_help_and_parser_guards(self):
        for script in ("eval_long_memory_v4.py", "serve_long_memory_v4.py"):
            result = subprocess.run([sys.executable, str(ROOT / "run_scripts/robomme" / script), "--help"],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
        for command in (["--models", "reader"], ["--models", "baseline", "expert-only"],
                        ["--models", "baseline", "memory"], ["--models", "baseline", "baseline"],
                        ["--models", "baseline", "--n-episodes", "0"]):
            with self.assertRaises(ValueError):
                driver.validate_options(driver.build_parser().parse_args(command))


class V4PolicyWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("v4_legacy_policy_fixtures", ROOT / "tests/test_long_memory_online_policy.py")
        cls.fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fixtures)

    def policy(self, *, memory=False, adapted=False, expert_only=False):
        from gr00t.long_memory.online_policy_v4 import LongMemoryV4Policy
        original = self.fixtures._policy(memory=False)
        result = LongMemoryV4Policy.__new__(LongMemoryV4Policy)
        result.__dict__.update(original.__dict__)
        result.expert_only, result.reader_mode = expert_only, "memory" if memory else "none"
        if memory:
            result.memory = ActionValueMemory(MemoryV3Config(feature_dim=6, state_dim=8, action_dim=8,
                hidden_dim=8, num_heads=2, capacity=3, min_fill=1, residual_init=.01)).eval().requires_grad_(False)
            result.stage, result.write_policy = 1, "all"
        elif adapted:
            result.stage, result.write_policy = 1, "disabled"
        return result

    def test_baseline_preserves_original_features_and_noise(self):
        import numpy as np
        original, v4 = self.fixtures._policy(), self.policy()
        for candidate in (original, v4):
            candidate.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
        action_a, _ = original.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        action_b, info = v4.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        np.testing.assert_array_equal(action_a["joint_position"], action_b["joint_position"])
        self.assertFalse(info["expert_adapted"])

    def test_expert_only_and_trained_control_have_no_bank_and_distinct_diagnostics(self):
        for expert_only, label in ((True, "expert-only"), (False, "trained-no-memory-control")):
            policy = self.policy(adapted=True, expert_only=expert_only)
            _, info = policy.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
            self.assertIsNone(policy.sessions["A"].bank)
            self.assertEqual(info["long_memory"]["policy"], label)
            self.assertTrue(info["expert_adapted"])

    def test_v4_reader_uses_causal_v3_bank_and_preserves_visual_prefix(self):
        from gr00t.long_memory.online_v3 import OnlineActionValueBank
        policy = self.policy(memory=True)
        policy.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
        _, info = policy.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        self.assertIsInstance(policy.sessions["A"].bank, OnlineActionValueBank)
        self.assertEqual(info["long_memory"]["bank_fill"], 1)
        head = policy.model.action_head
        self.assertTrue(torch.equal(head.last_features[:, :-2], head.last_processed[:, :-2]))

    def test_constructor_installs_loads_and_freezes_saved_expert_including_no_memory(self):
        from gr00t.long_memory import online_policy_v4 as module
        cfg = MemoryV3Config(feature_dim=6, state_dim=8, action_dim=8, hidden_dim=8, num_heads=2)
        original = self.fixtures._policy()
        original.model.action_head = nn.Linear(6, 6)
        info = {"config": {"stage": 1, "train": {"reader_mode": "none"}, "memory": asdict(cfg),
                           "expert": asdict(LoRAConfig()), "expert_targets": ["example"]}}
        def fake_init(policy, *args, **kwargs):
            policy.__dict__.update(original.__dict__)
        with patch.object(module, "v4_checkpoint_info", return_value=info), \
                patch.object(module.LongMemoryPolicy, "__init__", fake_init), \
                patch.object(module, "install_expert_lora") as install, \
                patch.object(module, "load_checkpoint_v4") as load, \
                patch.object(module, "set_expert_trainable") as freeze:
            policy = module.LongMemoryV4Policy("base", "bundle", device="cpu")
        install.assert_called_once_with(original.model.action_head, LoRAConfig(), targets=["example"])
        load.assert_called_once()
        self.assertEqual(load.call_args.args[0], "bundle")
        self.assertIs(load.call_args.args[2], original.model.action_head)
        freeze.assert_called_once_with(original.model.action_head, False)
        self.assertIsNone(policy.memory)
        self.assertEqual(policy.write_policy, "disabled")


if __name__ == "__main__":
    unittest.main()
