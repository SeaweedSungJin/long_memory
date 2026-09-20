"""CPU-only V7 evaluation, immutable provenance and streaming replay tests."""
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

from gr00t.eval.sim.robomme.compare_long_memory_v7_results import build_v7_report
from gr00t.long_memory.checkpoint_v4 import _state_sha256, file_sha256
from gr00t.long_memory.checkpoint_v7 import actor_state_sha256, v7_checkpoint_info
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.hamlet import checkpoint_identity
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from gr00t.long_memory.replay_v7 import apply_write, encode_at, initial_replay_state, replay_state
from run_scripts.robomme import eval_long_memory_v7 as driver

ROOT = Path(__file__).resolve().parents[1]


class V7EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.config = MemoryV7Config(feature_dim=8, state_dim=4, num_short_tokens=4,
                                    hidden_dim=8, num_heads=2, capacity=3)
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
        self.reader = self.make_checkpoint("reader", 1, "recurrent")
        self.memory = self.make_checkpoint("memory", 2, "recurrent")
        self.archive = self.make_checkpoint("archive", 1, "archive")
        self.ae = self.make_checkpoint("ae", 1, "none")

    def tearDown(self):
        self.temp.cleanup()

    def make_checkpoint(self, name, stage, mode):
        path = self.root / name
        path.mkdir()
        actor = RecurrentMemoryV7(self.config).state_dict() if stage == 1 else load_file(str(self.reader / "model.safetensors"))
        expert = {name + suffix: torch.zeros(shape) for name in self.targets
                  for suffix, shape in ((".lora_A", (2, 8)), (".lora_B", (8, 2)))}
        critic = CVOMV7(self.config).state_dict()
        if stage == 2:
            critic["head.2.weight"].add_(0.1)
        payloads = {"model.safetensors": actor, "expert.safetensors": expert, "cvom.safetensors": critic}
        for filename, state in payloads.items():
            save_file(state, str(path / filename))
        metadata = {"base_model": checkpoint_identity(self.base), "cache_fingerprint": "cache-A",
                    "payload_sha256": {name: file_sha256(path / name) for name in payloads}}
        if stage == 2:
            metadata.update(frozen_actor_sha256=actor_state_sha256(actor), frozen_expert_sha256=_state_sha256(expert),
                stage1_parent={"path": "/unmounted/parent", **{key: "a" * 64 for key in
                              ("checkpoint_sha256", "memory_sha256", "expert_sha256")}})
        info = {"format_version": 1, "step": 20, "config": {"trainer_variant": "recurrent_memory_v7",
                "stage": stage, "mode": mode, "memory": asdict(self.config), "expert": {"rank": 2, "alpha": 4},
                "expert_targets": self.targets, "train": {"cvom_threshold": 0.05, "mode": mode}}, "metadata": metadata}
        (path / "checkpoint.json").write_text(json.dumps(info))
        return path

    def update_info(self, path, mutation):
        target = path / "checkpoint.json"
        info = json.loads(target.read_text())
        mutation(info)
        target.write_text(json.dumps(info))

    def args(self, *extra):
        return driver.build_parser().parse_args(["--base-model", str(self.base),
            "--reader-checkpoint", str(self.reader), "--memory-checkpoint", str(self.memory),
            "--archive-checkpoint", str(self.archive), "--ae-checkpoint", str(self.ae),
            "--output-dir", str(self.root / "output"), *extra])

    def identity(self, args=None):
        args = args or self.args()
        driver.validate_options(args)
        with patch.object(driver, "benchmark_identity", return_value={"simulator": "fake"}), contextlib.redirect_stdout(io.StringIO()):
            return driver.build_identity(args)

    def test_all_roles_three_payloads_independent_controls_and_frozen_lineage(self):
        identity = self.identity(self.args("--models", "baseline", "reader", "memory", "archive", "ae-control", "memory-off", "fixed"))
        models = identity["models"]
        self.assertEqual(identity["trainer_variant"], "recurrent_memory_v7")
        self.assertEqual(identity["settings"]["dataset"], "val")
        self.assertIsNone(models["baseline"]["memory_checkpoint"])
        self.assertEqual(models["reader"]["semantic_state_sha256"], models["memory"]["semantic_state_sha256"])
        self.assertEqual(models["memory"]["cvom_weights_sha256"], models["fixed"]["cvom_weights_sha256"])
        self.assertEqual(models["reader"]["expert_weights_sha256"], models["memory-off"]["expert_weights_sha256"])
        self.assertEqual(models["archive"]["mode"], "archive")
        self.assertEqual(models["ae-control"]["mode"], "none")
        self.assertEqual(models["fixed"]["write_policy_override"], "update")
        for path in ("gr00t/long_memory/recurrent_v7.py", "gr00t/long_memory/cvom_v7.py",
                     "gr00t/long_memory/replay_v7.py", "gr00t/long_memory/online_policy.py"):
            self.assertIn(path, identity["source_sha256"])
        self.assertFalse((self.root / "output").exists())

    def test_wrong_control_mode_and_different_stage1_parent_rejected(self):
        with self.assertRaisesRegex(ValueError, "Role ae-control requires"):
            self.identity(self.args("--models", "baseline", "ae-control", "--ae-checkpoint", str(self.reader)))
        state = load_file(str(self.reader / "model.safetensors"))
        state["slot_addresses"].add_(.1)
        save_file(state, str(self.reader / "model.safetensors"))
        self.update_info(self.reader, lambda info: info["metadata"]["payload_sha256"].update(
            {"model.safetensors": file_sha256(self.reader / "model.safetensors")}))
        with self.assertRaisesRegex(ValueError, "actual parent"):
            self.identity(self.args("--models", "baseline", "reader", "memory"))
        v7_checkpoint_info(self.base, self.memory, expected_stage=2)  # no mounted parent needed

    def test_nonfinite_critic_missing_payload_and_threshold_are_rejected(self):
        path = self.reader / "cvom.safetensors"
        state = load_file(str(path))
        state[next(iter(state))].flatten()[0] = float("nan")
        save_file(state, str(path))
        self.update_info(self.reader, lambda info: info["metadata"]["payload_sha256"].update(
            {"cvom.safetensors": file_sha256(path)}))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.identity()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.identity()
        self.update_info(self.archive, lambda info: info["config"]["train"].update(cvom_threshold=float("nan")))
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            self.identity(self.args("--models", "baseline", "archive"))

    def test_preflight_baseline_is_readonly_and_output_cannot_be_under_source(self):
        args = ["--models", "baseline", "--base-model", str(self.base), "--preflight-only",
                "--output-dir", str(self.root / "preflight"), "--server-python", sys.executable]
        with patch.object(driver, "check_dependencies"), patch.object(driver, "benchmark_identity", return_value={}), \
                patch.object(driver, "run_evaluation") as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(args), 0)
        launch.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())
        with self.assertRaisesRegex(ValueError, "inside a source"):
            self.identity(self.args("--output-dir", str(self.reader / "unsafe")))

    def test_server_roles_use_explicit_read_off_and_write_override(self):
        args = self.args("--models", "baseline", "reader", "memory-off", "fixed")
        models = self.identity(args)["models"]
        cmd = driver.server_command(args, models["memory-off"], 1234)
        self.assertIn(str(ROOT / "run_scripts/robomme/serve_long_memory_v7.py"), cmd)
        self.assertIn("--memory-off", cmd)
        cmd = driver.server_command(args, models["fixed"], 1234)
        self.assertEqual(cmd[cmd.index("--write-policy") + 1], "update")
        self.assertNotIn("--memory-checkpoint", driver.server_command(args, models["baseline"], 1234))

    def test_server_failure_is_incomplete_and_completed_resume_never_launches(self):
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
        manifest = {"trainer_variant": "recurrent_memory_v7", "evaluation_id": "example", "models":
                    {name: {} for name in ("baseline", "reader", "memory", "archive", "ae-control", "memory-off", "fixed")},
                    "settings": {"tasks": ["BinFill", "PatternLock"], "n_episodes": 3,
                    "dataset": "val", "seed": 6, "n_action_steps": 16, "max_episode_steps": 1300}}
        (folder / "comparison_manifest.json").write_text(json.dumps(manifest))
        for name in manifest["models"]:
            outcomes = [1, 1, 1] if name == "memory" else [0, 1, 1]
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

    def test_reports_distinguish_controls_and_final_write_counters(self):
        folder = self.report_fixture()
        records = [{"kind": "policy_call", "session_id": sid, "info": {"long_memory": {
            "observations_seen": n, "write_attempts": n, "updates": n - 1, "keeps": 1,
            "demo_updates": 1, "demo_keeps": 0}}} for sid, n in (("aborted", 99), ("complete", 2), ("complete", 5))]
        records.append({"kind": "episode_complete", "session_id": "complete", "episode_idx": 0,
                        "episode_seed": 100, "success": 1})
        (folder / "memory/BinFill/memory_diagnostics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records) + '{"torn":')
        result, text = build_v7_report(folder, bootstrap_samples=20)
        contrast = result["additional_comparisons"]["fixed_to_memory"]
        self.assertEqual((contrast["paired_n"], contrast["wins"], contrast["losses"]), (3, 1, 0))
        self.assertFalse(contrast["complete"])
        self.assertIn("ae-control_to_reader", result["additional_comparisons"])
        self.assertNotIn("ae-control", result["storage_diagnostics"])
        self.assertEqual(result["storage_diagnostics"]["memory"]["updates"], 4)
        self.assertEqual(result["storage_diagnostics"]["memory"]["completed_sessions_missing_diagnostics"], 2)
        self.assertEqual(result["storage_diagnostics"]["memory"]["ignored_torn_final_lines"], 1)
        self.assertIn("not a capacity-matched", text)
        json.dumps(result, allow_nan=False)

    def test_report_only_lock_and_argument_guards(self):
        folder = self.report_fixture()
        with patch.object(driver, "check_dependencies") as deps, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(["--report-only", "--output-dir", str(folder)]), 0)
        deps.assert_not_called()
        with (folder / ".driver.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Evaluation is running"):
                driver.main(["--report-only", "--output-dir", str(folder)])
        for command in (["--models", "reader"], ["--models", "baseline", "archive"],
                        ["--models", "baseline", "ae-control"], ["--models", "baseline", "memory-off"],
                        ["--models", "baseline", "--task-timeout", "nan"],
                        ["--models", "baseline", "--n-episodes", "51"]):
            with self.assertRaises(ValueError):
                driver.validate_options(driver.build_parser().parse_args(command))
        for name in ("eval_long_memory_v7.py", "serve_long_memory_v7.py"):
            result = subprocess.run([sys.executable, str(ROOT / "run_scripts/robomme" / name), "--help"],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)


class V7OnlinePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("v7_old_policy_fixture", ROOT / "tests/test_long_memory_online_policy.py")
        cls.fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fixtures)

    def policy(self, *, mode=None, stage=1, memory_off=False):
        from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
        old = self.fixtures._policy(memory=False)
        result = LongMemoryV7Policy.__new__(LongMemoryV7Policy)
        result.__dict__.update(old.__dict__)
        result.mode, result.memory_off, result.cvom_threshold = mode or "none", memory_off, .05
        result.stage, result.cvom = stage if mode else 0, None
        if mode:
            cfg = MemoryV7Config(feature_dim=6, state_dim=8, num_short_tokens=2,
                                  hidden_dim=8, num_heads=2, capacity=3, time_scale=2)
            result.memory = RecurrentMemoryV7(cfg).eval().requires_grad_(False)
            result.cvom = CVOMV7(cfg).eval().requires_grad_(False)
            with torch.no_grad():
                result.memory.fusion_projection.weight.normal_(std=.1)
        result.write_policy = "none" if not mode or mode == "none" else "append" if mode == "archive" else (
            "cvom" if stage == 2 else "update")
        return result

    def test_baseline_matches_original_features_and_noise(self):
        original, policy = self.fixtures._policy(), self.policy()
        for candidate in (original, policy):
            candidate.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
        left, _ = original.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        right, info = policy.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        np.testing.assert_array_equal(left["joint_position"], right["joint_position"])
        self.assertTrue(torch.equal(original.model.action_head.last_features, policy.model.action_head.last_features))
        self.assertFalse(info["expert_adapted"])

    def test_full_demo_writes_preserve_rng_and_batched_session_isolation(self):
        policy = self.policy(mode="recurrent")
        session = policy._session("A", 123, False)
        untouched = policy._session("B", 456, False).latent.clone()
        rng = session.generator.get_state().clone()
        for frame in (0, 2, 3):
            options = self.fixtures._options(frame=frame, passive=True)
            options["episode_seed"] = 123
            _, info = policy.get_action(self.fixtures._observation(), options)
        self.assertEqual(session.updates, 3)
        self.assertEqual(info["long_memory"]["demo_updates"], 3)
        self.assertFalse(info["long_memory"]["memory_read_enabled"])
        self.assertEqual(policy.model.action_head.denoiser_calls, 0)
        self.assertTrue(torch.equal(rng, session.generator.get_state()))
        self.assertTrue(torch.equal(untouched, policy.sessions["B"].latent))
        self.assertEqual(policy.reset({"session_ids": ["A"]}), {"cleared_sessions": 1})
        self.assertIn("B", policy.sessions)
        self.assertFalse(policy._session("A", 123, False).latent.count_nonzero())

    def test_online_matches_offline_prefix_state_and_read_before_current_write(self):
        for mode in ("recurrent", "archive"):
            policy = self.policy(mode=mode)
            episode = {"short": [], "state": [], "frames": [], "is_demo": []}
            for index, frame in enumerate((0, 2, 3, 5, 7, 9)):
                # Two demo observations, followed by a partial first transition.
                passive = index < 2
                previous = episode["frames"][-1] if episode["frames"] else 0
                controls = None if index < 3 else np.zeros((frame - previous, 8), np.float32)
                _, info = policy.get_action(self.fixtures._observation(),
                    self.fixtures._options(frame=frame, passive=passive, controls=controls))
                session = policy.sessions["A"]
                head = policy.model.action_head
                episode["short"].append(head.last_processed[0, -2:].float())
                episode["state"].append(torch.from_numpy(np.concatenate(
                    [session.raw_states["joint_position"], session.raw_states["gripper_position"]], axis=-1)).reshape(-1))
                episode["frames"].append(frame)
                episode["is_demo"].append(passive)
                if not passive:
                    before = replay_state(policy.memory, episode, index, mode=mode, checkpoint_segment=0)
                    encoded = encode_at(policy.memory, episode, index)
                    expected, _ = policy.memory.read(episode["short"][-1][None], encoded, before, mode=mode)
                    self.assertTrue(torch.equal(head.last_features[:, -2:], expected.to(head.last_features.dtype)))
                    self.assertTrue(torch.equal(head.last_features[:, :-2], head.last_processed[:, :-2]))
                after = replay_state(policy.memory, episode, index + 1, mode=mode, checkpoint_segment=0)
                self.assertTrue(torch.equal(session.latent, after), mode)
                self.assertEqual(session.latent.dtype, torch.float32)
                self.assertFalse(session.latent.requires_grad)
            if mode == "archive":
                self.assertEqual(session.latent.shape[1], 12)
            else:
                self.assertEqual(session.latent.shape[1], 3)

    def test_cvom_keep_is_bitwise_and_threshold_tie_updates(self):
        policy = self.policy(mode="recurrent", stage=2)
        policy.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
        previous = policy.sessions["A"].latent.clone()
        with patch.object(policy.cvom, "forward", return_value=torch.tensor([-.1])):
            _, info = policy.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        self.assertTrue(torch.equal(previous, policy.sessions["A"].latent))
        self.assertEqual(info["long_memory"]["keeps"], 1)
        with patch.object(policy.cvom, "forward", return_value=torch.tensor([-.05])):
            _, info = policy.get_action(self.fixtures._observation(),
                self.fixtures._options(frame=4, controls=np.zeros((2, 8), np.float32)))
        self.assertEqual(info["long_memory"]["updates"], 2)
        self.assertEqual(info["long_memory"]["keeps"], 1)

    def test_memory_off_read_bypass_but_writes_continue_and_ae_control_no_write(self):
        for mode, off in (("recurrent", True), ("none", False)):
            policy = self.policy(mode=mode, memory_off=off)
            policy.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
            _, info = policy.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
            head = policy.model.action_head
            self.assertTrue(torch.equal(head.last_features, head.last_processed))
            self.assertEqual(info["long_memory"]["updates"], 2 if mode == "recurrent" else 0)

    def test_runtime_conditioning_change_is_measured_after_dtype_cast(self):
        policy = self.policy(mode="recurrent")
        policy.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
        with patch.object(policy.memory, "read", side_effect=lambda short, *args, **kwargs: (short + 1, {})):
            _, info = policy.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        self.assertEqual(info["long_memory"]["read"]["ae_conditioning_changed_fraction"], 1)
        self.assertGreater(info["long_memory"]["read"]["ae_conditioning_delta_norm"], 0)

    def test_failure_clears_session_and_model_cache(self):
        policy = self.policy(mode="recurrent")
        policy.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
        policy.model.action_head.fail_prediction = True
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite model action"):
            policy.get_action(self.fixtures._observation(), self.fixtures._options(frame=1))
        self.assertFalse(policy.sessions)
        self.assertIsNone(policy.model.action_head._memory_cache)
        self.assertIsNone(policy.model.action_head._inference_gen)

    def test_constructor_installs_lora_and_loads_actor_head_critic_in_order(self):
        from gr00t.long_memory import online_policy_v7 as module
        cfg = MemoryV7Config(feature_dim=6, state_dim=8, num_short_tokens=2, hidden_dim=8, num_heads=2)
        original = self.fixtures._policy()
        original.model.action_head = nn.Linear(6, 6)
        info = {"config": {"stage": 1, "mode": "recurrent", "memory": asdict(cfg),
                           "expert": {"rank": 2, "alpha": 4}, "expert_targets": ["example"], "train": {}}}
        def fake_init(policy, *args, **kwargs):
            policy.__dict__.update(original.__dict__)
        with patch.object(module, "v7_checkpoint_info", return_value=info), \
                patch.object(module.LongMemoryPolicy, "__init__", fake_init), \
                patch.object(module, "install_expert_lora") as install, \
                patch.object(module, "load_checkpoint_v7") as load, \
                patch.object(module, "set_expert_trainable") as freeze:
            policy = module.LongMemoryV7Policy("base", "bundle", device="cpu")
        install.assert_called_once()
        self.assertIs(load.call_args.args[1], policy.memory)
        self.assertIs(load.call_args.args[2], original.model.action_head)
        self.assertIs(load.call_args.args[3], policy.cvom)
        freeze.assert_called_once_with(original.model.action_head, False)
        self.assertFalse(any(parameter.requires_grad for parameter in policy.memory.parameters()))
        self.assertEqual(policy.write_policy, "update")


if __name__ == "__main__":
    unittest.main()
