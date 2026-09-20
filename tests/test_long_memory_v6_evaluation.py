"""Synthetic CPU tests for v6 bundle evaluation and causal online retrieval."""
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

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch import nn

from gr00t.eval.sim.robomme.compare_long_memory_v6_results import build_v6_report
from gr00t.long_memory.checkpoint_v4 import _state_sha256, file_sha256
from gr00t.long_memory.checkpoint_v6 import reader_state_sha256, v6_checkpoint_info
from gr00t.long_memory.core_v6 import VisualMemoryV6, VisualMemoryV6Config
from gr00t.long_memory.expert_v6 import memory_bridge_shapes, with_memory
from gr00t.long_memory.hamlet import checkpoint_identity
from gr00t.long_memory.replay_v6 import bound_archive, build_candidates
from run_scripts.robomme import eval_long_memory_v6 as driver

ROOT = Path(__file__).resolve().parents[1]


class V6EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.config = VisualMemoryV6Config(feature_dim=8, state_dim=4, hidden_dim=8,
                            num_heads=2, visual_tokens=2, max_archive_events=4, read_budget=2)
        self.bridge = dict(hidden_dim=8, num_heads=2, block_indices=[0], expert_dim=8)
        self.targets = sorted("model.transformer_blocks.0.attn1." + name
                              for name in ("to_q", "to_k", "to_v", "to_out.0"))
        weights = {"action_head." + name + ".weight": torch.zeros(8, 8) for name in self.targets}
        save_file(weights, str(self.base / "model.safetensors"))
        config = {"hamlet_mode": "finetune", "mem_cond_type": "cross_attn", "memory_type": "moment_token",
                  "memory_stride": 16, "n_moment_tokens": 4, "backbone_embedding_dim": 8}
        (self.base / "config.json").write_text(json.dumps(config))
        (self.base / "processor_config.json").write_text(json.dumps(
            {"processor_kwargs": {"max_state_dim": 4, "max_action_dim": 3}}))
        (self.base / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {key: "model.safetensors" for key in weights}}))
        for name in ("statistics.json", "embodiment_id.json"):
            (self.base / name).write_text("{}")
        self.reader = self.make_checkpoint("reader", 1)
        self.cvom = self.make_checkpoint("cvom", 2)

    def tearDown(self):
        self.temp.cleanup()

    def make_checkpoint(self, name, stage):
        path = self.root / name
        path.mkdir()
        memory = VisualMemoryV6(self.config).state_dict() if stage == 1 else load_file(str(self.reader / "memory.safetensors"))
        if stage == 2:
            memory["cvom.head.3.bias"].add_(0.1)
        expert = {name + suffix: torch.zeros(shape) for name in self.targets
                  for suffix, shape in ((".lora_A", (2, 8)), (".lora_B", (8, 2)))}
        bridge = {name: torch.zeros(shape) for name, shape in memory_bridge_shapes(self.bridge).items()}
        payloads = {"memory.safetensors": memory, "expert.safetensors": expert, "bridge.safetensors": bridge}
        for filename, state in payloads.items():
            save_file(state, str(path / filename))
        metadata = {"base_model": checkpoint_identity(self.base), "cache_fingerprint": "cache-A",
                    "payload_sha256": {name: file_sha256(path / name) for name in payloads}}
        if stage == 2:
            frozen = {"reader": reader_state_sha256(memory), "expert": _state_sha256(expert), "bridge": _state_sha256(bridge)}
            metadata.update(frozen_identity=frozen,
                stage1_parent={"path": "/unmounted/parent", "checkpoint_sha256": "a" * 64, "frozen_identity": frozen})
        info = {"format_version": 1, "step": 20, "config": {"trainer_variant": "retrieval_cvom_v6",
                "stage": stage, "memory": asdict(self.config), "expert": {"rank": 2, "alpha": 4},
                "expert_targets": self.targets, "bridge": self.bridge, "reader_mode": "memory",
                "train": {"cvom_threshold": 0.05, "fixed_policy": "uniform"}}, "metadata": metadata}
        (path / "checkpoint.json").write_text(json.dumps(info))
        return path

    def update_info(self, path, mutation):
        target = path / "checkpoint.json"
        info = json.loads(target.read_text())
        mutation(info)
        target.write_text(json.dumps(info))

    def args(self, *extra):
        return driver.build_parser().parse_args(["--base-model", str(self.base),
            "--reader-checkpoint", str(self.reader), "--memory-checkpoint", str(self.cvom),
            "--output-dir", str(self.root / "output"), *extra])

    def identity(self, args=None):
        args = args or self.args()
        driver.validate_options(args)
        with patch.object(driver, "benchmark_identity", return_value={"simulator": "fake"}), contextlib.redirect_stdout(io.StringIO()):
            return driver.build_identity(args)

    def test_identity_hashes_three_components_and_stage2_only_changes_cvom(self):
        identity = self.identity(self.args("--models", "baseline", "reader", "cvom", "fixed", "expert-only"))
        self.assertEqual(identity["trainer_variant"], "retrieval_cvom_v6")
        models = identity["models"]
        self.assertIsNone(models["baseline"]["memory_checkpoint"])
        self.assertEqual(models["reader"]["semantic_state_sha256"], models["cvom"]["semantic_state_sha256"])
        for key in ("expert_weights_sha256", "bridge_weights_sha256", "weights_sha256"):
            self.assertEqual(models["cvom"][key], models["fixed"][key])
        self.assertNotEqual(models["reader"]["weights_sha256"], models["cvom"]["weights_sha256"])
        self.assertEqual(models["expert-only"]["read_policy"], "none")
        self.assertEqual(models["fixed"]["read_policy_override"], "uniform")
        self.assertIn("NOT a trained", models["expert-only"]["description"])
        for path in ("gr00t/long_memory/expert_v6.py", "gr00t/long_memory/replay_v6.py",
                     "gr00t/long_memory/online_policy.py", "run_scripts/robomme/eval_long_memory_comparison.py"):
            self.assertIn(path, identity["source_sha256"])
        self.assertFalse((self.root / "output").exists())

    def test_stage1_parent_mismatch_rejected_and_stage2_selfcontained(self):
        info = v6_checkpoint_info(self.base, self.cvom, expected_stage=2)
        self.assertEqual(info["metadata"]["stage1_parent"]["path"], "/unmounted/parent")
        memory = load_file(str(self.reader / "memory.safetensors"))
        memory["token_position"].add_(0.1)
        save_file(memory, str(self.reader / "memory.safetensors"))
        self.update_info(self.reader, lambda info: info["metadata"]["payload_sha256"].update(
            {"memory.safetensors": file_sha256(self.reader / "memory.safetensors")}))
        with self.assertRaisesRegex(ValueError, "actual parent"):
            self.identity(self.args("--models", "baseline", "reader", "cvom"))

    def test_missing_nonfinite_or_tampered_bridge_rejected(self):
        path = self.reader / "bridge.safetensors"
        state = load_file(str(path))
        first = next(iter(state))
        state[first].flatten()[0] = float("nan")
        save_file(state, str(path))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.identity()
        state[first].zero_()
        state[first].flatten()[0] = 1
        save_file(state, str(path))
        with self.assertRaisesRegex(ValueError, "changed after publication"):
            self.identity()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.identity()

    def test_readonly_preflight_baseline_does_not_create_output_or_launch(self):
        command = ["--models", "baseline", "--base-model", str(self.base), "--preflight-only",
                   "--output-dir", str(self.root / "preflight"), "--server-python", sys.executable]
        with patch.object(driver, "check_dependencies"), patch.object(driver, "benchmark_identity", return_value={}), \
                patch.object(driver, "run_evaluation") as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(command), 0)
        launch.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())

    def test_source_nested_output_rejected(self):
        with self.assertRaisesRegex(ValueError, "inside a source"):
            self.identity(self.args("--output-dir", str(self.reader / "unsafe")))

    def test_invalid_checkpoint_selection_settings_rejected_before_launch(self):
        original = (self.reader / "checkpoint.json").read_text()
        for updates, message in (({"cvom_threshold": float("nan")}, "finite and nonnegative"),
                                  ({"cvom_threshold": -1}, "finite and nonnegative"),
                                  ({"fixed_policy": "unsupported"}, "unsupported fixed")):
            self.update_info(self.reader, lambda info: info["config"]["train"].update(updates))
            with self.assertRaisesRegex(ValueError, message):
                self.identity()
            (self.reader / "checkpoint.json").write_text(original)

    def test_server_commands_use_new_retrieval_flags(self):
        args = self.args("--models", "baseline", "reader", "cvom", "fixed", "expert-only")
        models = self.identity(args)["models"]
        command = driver.server_command(args, models["fixed"], 1234)
        self.assertIn(str(ROOT / "run_scripts/robomme/serve_long_memory_v6.py"), command)
        self.assertEqual(command[command.index("--read-policy") + 1], "uniform")
        self.assertNotIn("--write-policy", command)
        self.assertIn("--expert-only", driver.server_command(args, models["expert-only"], 1234))
        self.assertNotIn("--memory-checkpoint", driver.server_command(args, models["baseline"], 1234))

    def test_failure_is_incomplete_and_completed_resume_does_not_start_server(self):
        args = self.args("--models", "baseline", "--tasks", "BinFill", "--n-episodes", "1")
        identity = self.identity(args)
        with patch.object(driver.subprocess, "Popen") as popen, \
                patch.object(driver, "server_ready", side_effect=RuntimeError("load failure")), \
                patch.object(driver, "stop_process") as stop, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(driver.run_evaluation(args, identity, {}), 1)
        stop.assert_called_once_with(popen.return_value)
        folder = self.root / "output"
        report = json.loads((folder / "comparison_summary.json").read_text())
        self.assertIsNone(report["models"]["baseline"]["tasks"]["BinFill"]["success_rate"])
        self.assertIn("load failure", (folder / "driver_status.json").read_text())
        self.assertTrue((folder / "baseline/server.log").is_file())
        task = folder / "baseline/BinFill"
        task.mkdir()
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
        manifest = {"trainer_variant": "retrieval_cvom_v6", "evaluation_id": "example", "models":
                    {name: {} for name in ("baseline", "reader", "cvom", "fixed", "expert-only")},
                    "settings": {"tasks": ["BinFill", "PatternLock"], "n_episodes": 3,
                    "dataset": "test", "seed": 6, "n_action_steps": 16, "max_episode_steps": 1300}}
        (folder / "comparison_manifest.json").write_text(json.dumps(manifest))
        for name, outcomes in (("baseline", [0, 0, 1]), ("reader", [0, 1, 1]),
                               ("cvom", [1, 1, 1]), ("fixed", [0, 1, 1]), ("expert-only", [0, 0, 0])):
            task = folder / name / "BinFill"
            task.mkdir(parents=True)
            (task / "policy_manifest.json").write_text(json.dumps({"evaluation_id": "example:" + name,
                "task_id": "BinFill", **{key: value for key, value in manifest["settings"].items()
                                         if key in ("dataset", "seed", "n_action_steps", "max_episode_steps")}}))
            with (task / "simulation_results.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["episode_idx", "episode_seed", "success", "status"])
                writer.writeheader()
                for episode, success in enumerate(outcomes):
                    writer.writerow({"episode_idx": episode, "episode_seed": episode + 100,
                                     "success": success, "status": "success" if success else "fail"})
        return folder

    def test_reports_readtime_contrasts_not_write_accuracy(self):
        result, text = build_v6_report(self.report_fixture(), bootstrap_samples=20)
        contrast = result["additional_comparisons"]["fixed_to_cvom"]
        self.assertEqual((contrast["paired_n"], contrast["wins"], contrast["losses"]), (3, 1, 0))
        self.assertEqual(result["additional_comparisons"]["reader_to_fixed"]["paired_task_macro_delta"], 0)
        self.assertFalse(contrast["complete"])
        self.assertNotIn("expert-only", result["retrieval_diagnostics"])
        self.assertIsNone(result["models"]["reader"]["tasks"]["PatternLock"]["success_rate"])
        self.assertIn("NO learned write", text)
        self.assertIn("NOT a separately trained", text)
        json.dumps(result, allow_nan=False)

    def test_diagnostic_counts_only_final_completed_attempt(self):
        root = self.report_fixture()
        records = [{"kind": "policy_call", "session_id": sid, "info": {"long_memory": {"read_counts":
                   {"uniform": n, "relevant": 0, "hybrid": 0, "null": 0}}}}
                   for sid, n in (("aborted", 99), ("complete", 1), ("complete", 3))]
        records.append({"kind": "episode_complete", "session_id": "complete", "episode_idx": 0,
                        "episode_seed": 100, "success": 1})
        (root / "cvom/BinFill/memory_diagnostics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records) + '{"torn":')
        result, _ = build_v6_report(root, bootstrap_samples=20)
        counts = result["retrieval_diagnostics"]["cvom"]
        self.assertEqual(counts["read_counts"]["uniform"], 3)
        self.assertEqual(counts["completed_sessions_missing_diagnostics"], 2)
        self.assertEqual(counts["ignored_torn_final_lines"], 1)

    def test_reportonly_lock_and_parser_validation(self):
        folder = self.report_fixture()
        with patch.object(driver, "check_dependencies") as dependencies, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(["--report-only", "--output-dir", str(folder)]), 0)
        dependencies.assert_not_called()
        with (folder / ".driver.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Evaluation is running"):
                driver.main(["--report-only", "--output-dir", str(folder)])
        for command in (["--models", "reader"], ["--models", "baseline", "cvom"],
                        ["--models", "baseline", "--n-episodes", "51"],
                        ["--models", "baseline", "--task-timeout", "nan"]):
            with self.assertRaises(ValueError):
                driver.validate_options(driver.build_parser().parse_args(command))
        for name in ("eval_long_memory_v6.py", "serve_long_memory_v6.py"):
            result = subprocess.run([sys.executable, str(ROOT / "run_scripts/robomme" / name), "--help"],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)


class V6OnlinePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("v6_old_policy_fixture", ROOT / "tests/test_long_memory_online_policy.py")
        cls.fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fixtures)

    def policy(self, *, memory=False, stage=1, expert_only=False):
        from gr00t.long_memory.online_policy_v6 import LongMemoryV6Policy
        old = self.fixtures._policy(memory=False)
        result = LongMemoryV6Policy.__new__(LongMemoryV6Policy)
        result.__dict__.update(old.__dict__)
        result.expert_only, result.read_policy = expert_only, "none"
        result.cvom_threshold, result.stage = 0.05, stage if memory or expert_only else 0
        result.model.action_head.long_memory_v6_bridge = nn.ModuleDict()
        if memory:
            result.memory = VisualMemoryV6(VisualMemoryV6Config(feature_dim=6, state_dim=8, hidden_dim=8,
                visual_tokens=2, num_heads=2, max_archive_events=3, read_budget=2, time_scale=2)).eval().requires_grad_(False)
            result.read_policy = "cvom" if stage == 2 else "uniform"
        return result

    def test_baseline_identical_original_features_and_seeded_actions(self):
        original, v6 = self.fixtures._policy(), self.policy()
        for policy in (original, v6):
            policy.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
        action_a, _ = original.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        action_b, info = v6.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        np.testing.assert_array_equal(action_a["joint_position"], action_b["joint_position"])
        self.assertTrue(torch.equal(original.model.action_head.last_features, v6.model.action_head.last_features))
        self.assertFalse(info["expert_adapted"])

    def test_demo_archive_no_ae_or_rng_and_current_never_in_candidates(self):
        policy = self.policy(memory=True)
        session = policy._session("A", 123, False)
        rng = session.generator.get_state().clone()
        for frame in (0, 2):
            options = self.fixtures._options(frame=frame, passive=True)
            options["episode_seed"] = 123
            policy.get_action(self.fixtures._observation(), options)
        self.assertEqual(policy.model.action_head.denoiser_calls, 0)
        self.assertTrue(torch.equal(rng, session.generator.get_state()))
        self.assertEqual([row["frame"] for row in session.observations], [0, 2])
        options = self.fixtures._options(frame=4)
        options["episode_seed"] = 123
        _, info = policy.get_action(self.fixtures._observation(), options)
        self.assertEqual(info["long_memory"]["selected_event_ids"], [0, 1])
        self.assertNotIn(2, info["long_memory"]["selected_event_ids"])
        self.assertTrue(all(not value.requires_grad and value.device.type == "cpu"
                            for row in session.observations for value in row.values() if torch.is_tensor(value)))

    def test_online_bounded_archive_and_tokens_equal_full_prefix_replay(self):
        from gr00t.long_memory import online_policy_v6 as module
        policy = self.policy(memory=True)
        full, contexts = [], []
        @contextlib.contextmanager
        def record(head, tokens):
            contexts.append(None if tokens is None else tokens.clone())
            with with_memory(head, tokens):
                yield
        for i in range(8):
            controls = None if i < 2 else np.zeros((2, 8), dtype=np.float32)
            with patch.object(module, "with_memory", record):
                _, info = policy.get_action(self.fixtures._observation(),
                    self.fixtures._options(frame=2*i, passive=i == 0, controls=controls))
            session = policy.sessions["A"]
            current = session.observations[-1]
            if i:
                expected = build_candidates(policy.memory, full, current)["candidates"]["uniform"]
                self.assertEqual(info["long_memory"]["selected_event_ids"], expected["event_ids"])
                self.assertTrue(torch.equal(contexts[-1], expected["tokens"]))
            full.append(current)
            expected_archive = bound_archive(full, 3)
            self.assertEqual([row["event_id"] for row in session.observations],
                             [row["event_id"] for row in expected_archive])
            self.assertLessEqual(len(session.observations), 3)
            self.assertEqual(session.observations[0]["event_id"], 0)
            head = policy.model.action_head
            if i:
                self.assertTrue(torch.equal(head.last_features, head.last_processed))

    def test_cvom_selection_uses_saved_threshold_null_and_diagnostics(self):
        policy = self.policy(memory=True, stage=2)
        policy.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
        with patch.object(policy.memory, "score_candidates", return_value=torch.tensor([-0.5, -0.2, -0.3, 0.0])):
            _, info = policy.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        self.assertEqual(info["long_memory"]["selected_candidate"], "null")
        self.assertEqual(info["long_memory"]["read_counts"]["null"], 1)
        self.assertEqual(info["long_memory"]["candidate_scores"]["null"], 0)
        with patch.object(policy.memory, "score_candidates", return_value=torch.tensor([0.2, 0.22, 0.21, 0.0])):
            _, info = policy.get_action(self.fixtures._observation(),
                self.fixtures._options(frame=4, controls=np.zeros((2, 8), np.float32)))
        self.assertEqual(info["long_memory"]["selected_candidate"], "uniform")

    def test_expert_only_has_no_archive_and_resets_sessions(self):
        policy = self.policy(expert_only=True)
        _, info = policy.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
        self.assertEqual(policy.sessions["A"].observations, [])
        self.assertEqual(info["long_memory"]["policy"], "expert-only")
        self.assertEqual(policy.reset({"session_ids": ["A"]}), {"cleared_sessions": 1})
        self.assertFalse(policy.sessions)

    def test_partial_interval_and_failure_session_cleanup(self):
        policy = self.policy(memory=True)
        policy.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
        policy.get_action(self.fixtures._observation(), self.fixtures._options(frame=1))
        policy.model.action_head.fail_prediction = True
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite model action"):
            policy.get_action(self.fixtures._observation(),
                self.fixtures._options(frame=2, controls=np.zeros((1, 8), np.float32)))
        self.assertFalse(policy.sessions)
        self.assertIsNone(policy.model.action_head._memory_cache)
        self.assertIsNone(policy.model.action_head._inference_gen)

    def test_constructor_installs_three_components_then_freezes(self):
        from gr00t.long_memory import online_policy_v6 as module
        cfg = VisualMemoryV6Config(feature_dim=6, state_dim=8, hidden_dim=8, num_heads=2)
        bridge = dict(hidden_dim=8, block_indices=[0], num_heads=2, expert_dim=8)
        original = self.fixtures._policy()
        original.model.action_head = nn.Linear(6, 6)
        info = {"config": {"stage": 1, "memory": asdict(cfg), "expert": {"rank": 2, "alpha": 4},
                           "expert_targets": ["example"], "bridge": bridge, "train": {"cvom_threshold": .05}}}
        def fake_init(policy, *args, **kwargs):
            policy.__dict__.update(original.__dict__)
        with patch.object(module, "v6_checkpoint_info", return_value=info), \
                patch.object(module.LongMemoryPolicy, "__init__", fake_init), \
                patch.object(module, "install_expert_lora") as install, \
                patch.object(module, "install_memory_bridge", return_value=bridge) as bridge_install, \
                patch.object(module, "load_checkpoint_v6") as load, \
                patch.object(module, "set_expert_trainable") as freeze:
            policy = module.LongMemoryV6Policy("base", "bundle", device="cpu")
        install.assert_called_once()
        bridge_install.assert_called_once_with(original.model.action_head, **bridge)
        load.assert_called_once()
        self.assertIs(load.call_args.args[2], original.model.action_head)
        freeze.assert_called_once_with(original.model.action_head, False)
        self.assertFalse(any(parameter.requires_grad for parameter in policy.model.action_head.parameters()))
        self.assertEqual(policy.read_policy, "uniform")


if __name__ == "__main__":
    unittest.main()
