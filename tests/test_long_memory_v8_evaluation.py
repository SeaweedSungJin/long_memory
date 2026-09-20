"""CPU contracts for V8 deployment, cache-source parity and paired reports."""
import contextlib
from dataclasses import asdict, replace
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

from gr00t.eval.sim.robomme.compare_long_memory_v8_results import build_v8_report
from gr00t.long_memory.checkpoint_v4 import file_sha256
from gr00t.long_memory.event_v8 import EventMemoryV8, MemoryV8Config
from gr00t.long_memory.hamlet import checkpoint_identity
from gr00t.long_memory.replay_v8 import encode_at, replay_state
from run_scripts.robomme import eval_long_memory_v8 as driver

ROOT = Path(__file__).resolve().parents[1]


class V8EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.config = MemoryV8Config(feature_dim=8, state_dim=4, num_short_tokens=4,
                                    hidden_dim=8, num_heads=2, capacity=3)
        self.targets = sorted("model.transformer_blocks.0.attn1." + name
                              for name in ("to_q", "to_k", "to_v", "to_out.0"))
        weights = {"action_head." + name + ".weight": torch.zeros(8, 8) for name in self.targets}
        save_file(weights, str(self.base / "model.safetensors"))
        (self.base / "config.json").write_text(json.dumps({"hamlet_mode": "finetune",
            "mem_cond_type": "cross_attn", "memory_type": "moment_token", "memory_stride": 16,
            "n_moment_tokens": 4, "backbone_embedding_dim": 8}))
        (self.base / "processor_config.json").write_text(json.dumps(
            {"processor_kwargs": {"max_state_dim": 4, "max_action_dim": 3}}))
        (self.base / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {key: "model.safetensors" for key in weights}}))
        for name in ("statistics.json", "embodiment_id.json"):
            (self.base / name).write_text("{}")
        self.reader = self.make_checkpoint("reader", "event")
        self.ae = self.make_checkpoint("ae", "none")
        self.short = self.make_checkpoint("short", "event", source="short")

    def tearDown(self):
        self.temp.cleanup()

    def make_checkpoint(self, name, mode, source="moment"):
        path = self.root / name
        path.mkdir()
        cfg = replace(self.config, source=source)
        expert = {name + suffix: torch.zeros(shape) for name in self.targets
                  for suffix, shape in ((".lora_A", (2, 8)), (".lora_B", (8, 2)))}
        payloads = {"model.safetensors": EventMemoryV8(cfg).state_dict(), "expert.safetensors": expert}
        for filename, state in payloads.items():
            save_file(state, str(path / filename))
        info = {"format_version": 1, "step": 20, "config": {"trainer_variant": "event_memory_v8",
                "stage": 1, "mode": mode, "memory": asdict(cfg), "expert": {"rank": 2, "alpha": 4},
                "expert_targets": self.targets, "train": {"mode": mode}}, "metadata": {
                    "base_model": checkpoint_identity(self.base), "cache_fingerprint": "cache-A",
                    "payload_sha256": {name: file_sha256(path / name) for name in payloads}}}
        (path / "checkpoint.json").write_text(json.dumps(info))
        return path

    def update_info(self, path, mutation):
        target = path / "checkpoint.json"
        info = json.loads(target.read_text())
        mutation(info)
        target.write_text(json.dumps(info))

    def args(self, *extra):
        return driver.build_parser().parse_args(["--base-model", str(self.base),
            "--reader-checkpoint", str(self.reader), "--ae-checkpoint", str(self.ae),
            "--short-checkpoint", str(self.short), "--output-dir", str(self.root / "output"), *extra])

    def identity(self, args=None):
        args = args or self.args()
        driver.validate_options(args)
        with patch.object(driver, "benchmark_identity", return_value={"simulator": "fake"}), contextlib.redirect_stdout(io.StringIO()):
            return driver.build_identity(args)

    def test_roles_two_payloads_source_capacity_and_same_checkpoint_ablation(self):
        identity = self.identity(self.args("--models", "baseline", "reader", "memory-off", "ae-control", "short-control"))
        models = identity["models"]
        self.assertEqual(identity["trainer_variant"], "event_memory_v8")
        self.assertEqual(identity["settings"]["dataset"], "val")
        self.assertIsNone(models["baseline"]["memory_checkpoint"])
        self.assertEqual(models["reader"]["semantic_state_sha256"], models["memory-off"]["semantic_state_sha256"])
        self.assertEqual(models["reader"]["checkpoint_sha256"], models["memory-off"]["checkpoint_sha256"])
        self.assertEqual(models["reader"]["capacity_events"], 3)
        self.assertEqual(models["reader"]["capacity_tokens"], 12)
        self.assertEqual(models["short-control"]["source"], "short")
        self.assertEqual(models["ae-control"]["mode"], "none")
        self.assertNotIn("cvom_weights_sha256", models["reader"])
        for path in ("gr00t/long_memory/event_v8.py", "gr00t/long_memory/replay_v8.py",
                     "gr00t/long_memory/online_policy_v8.py", "gr00t/long_memory/cache.py"):
            self.assertIn(path, identity["source_sha256"])
        self.assertFalse((self.root / "output").exists())

    def test_wrong_source_mode_stage_and_changed_capacity_are_rejected(self):
        for extra, message in ((["--models", "baseline", "ae-control", "--ae-checkpoint", str(self.reader)], "mode=none"),
                               (["--models", "baseline", "reader", "--reader-checkpoint", str(self.short)], "source=moment"),
                               (["--models", "baseline", "short-control", "--short-checkpoint", str(self.reader)], "source=short")):
            with self.assertRaisesRegex(ValueError, message):
                self.identity(self.args(*extra))
        self.update_info(self.ae, lambda x: x["config"]["memory"].update(capacity=4))
        with self.assertRaisesRegex(ValueError, "share training cache"):
            self.identity(self.args("--models", "baseline", "reader", "ae-control"))
        self.update_info(self.reader, lambda x: x["config"].update(stage=2))
        with self.assertRaisesRegex(ValueError, "Stage 1"):
            self.identity()

    def test_nonfinite_missing_or_hash_changed_payload_is_rejected(self):
        path = self.reader / "model.safetensors"
        state = load_file(str(path))
        state[next(iter(state))].flatten()[0] = float("nan")
        save_file(state, str(path))
        with self.assertRaisesRegex(ValueError, "payload changed"):
            self.identity()
        self.update_info(self.reader, lambda x: x["metadata"]["payload_sha256"].update({"model.safetensors": file_sha256(path)}))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.identity()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.identity()

    def test_preflight_is_readonly_and_output_scope_is_protected(self):
        args = ["--models", "baseline", "--base-model", str(self.base), "--preflight-only",
                "--output-dir", str(self.root / "preflight"), "--server-python", sys.executable]
        with patch.object(driver, "check_dependencies"), patch.object(driver, "benchmark_identity", return_value={}), \
                patch.object(driver, "run_evaluation") as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(args), 0)
        launch.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())
        with self.assertRaisesRegex(ValueError, "inside a source"):
            self.identity(self.args("--output-dir", str(self.reader / "unsafe")))

    def test_server_arguments_and_failed_then_completed_resume(self):
        args = self.args("--models", "baseline", "reader", "memory-off")
        models = self.identity(args)["models"]
        command = driver.server_command(args, models["memory-off"], 1234)
        self.assertIn("--memory-off", command)
        self.assertIn(str(ROOT / "run_scripts/robomme/serve_long_memory_v8.py"), command)
        self.assertNotIn("--write-policy", command)
        self.assertNotIn("--memory-checkpoint", driver.server_command(args, models["baseline"], 1234))
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
        task = folder / "baseline/BinFill"
        task.mkdir()
        (task / "simulation_results.csv").write_text("episode_idx,episode_seed,success\n0,100,0\n")
        (task / "policy_manifest.json").write_text(json.dumps({"evaluation_id": identity["evaluation_id"] + ":baseline",
            "task_id": "BinFill", **{key: identity["settings"][key]
                                     for key in ("dataset", "seed", "n_action_steps", "max_episode_steps")}}))
        with patch.object(driver.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.run_evaluation(args, identity, {}), 0)
        popen.assert_not_called()
        with self.assertRaisesRegex(ValueError, "different checkpoints/settings/code"):
            driver.bind_manifest(folder, {**identity, "evaluation_id": "changed-source"})

    def report_fixture(self):
        folder = self.root / "report"
        folder.mkdir()
        manifest = {"trainer_variant": "event_memory_v8", "evaluation_id": "example", "models": {
                    name: {} for name in ("baseline", "reader", "memory-off", "ae-control", "short-control")},
                    "settings": {"tasks": ["BinFill", "PatternLock"], "n_episodes": 3,
                    "dataset": "val", "seed": 6, "n_action_steps": 16, "max_episode_steps": 1300}}
        (folder / "comparison_manifest.json").write_text(json.dumps(manifest))
        for name in manifest["models"]:
            task = folder / name / "BinFill"
            task.mkdir(parents=True)
            (task / "policy_manifest.json").write_text(json.dumps({"evaluation_id": "example:" + name,
                "task_id": "BinFill", **{key: value for key, value in manifest["settings"].items()
                                         if key in ("dataset", "seed", "n_action_steps", "max_episode_steps")}}))
            outcomes = (1, 1, 1) if name == "reader" else (0, 1, 1)
            (task / "simulation_results.csv").write_text("episode_idx,episode_seed,success\n" +
                "".join(f"{i},{i+100},{value}\n" for i, value in enumerate(outcomes)))
        return folder

    def test_paired_reports_event_counters_exclude_aborted_sessions_and_torn_tail(self):
        folder = self.report_fixture()
        counters = {"observations_seen": 5, "write_attempts": 5, "appended_events": 5,
                    "demo_appended_events": 2, "evicted_events": 2, "retained_events": 3,
                    "memory_tokens": 12, "capacity_events": 3, "tokens_per_event": 4}
        records = [{"kind": "policy_call", "session_id": sid, "info": {"long_memory": data}}
                   for sid, data in (("aborted", {}), ("complete", counters))]
        records.append({"kind": "episode_complete", "session_id": "complete", "episode_idx": 0,
                        "episode_seed": 100, "success": 1})
        path = folder / "reader/BinFill/memory_diagnostics.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records) + '{"torn":')
        result, text = build_v8_report(folder, bootstrap_samples=20)
        contrast = result["additional_comparisons"]["memory-off_to_reader"]
        self.assertEqual((contrast["paired_n"], contrast["wins"], contrast["losses"]), (3, 1, 0))
        self.assertFalse(contrast["complete"])
        self.assertEqual(result["storage_diagnostics"]["reader"]["appended_events"], 5)
        self.assertEqual(result["storage_diagnostics"]["reader"]["evicted_events"], 2)
        self.assertEqual(result["storage_diagnostics"]["reader"]["completed_sessions_missing_diagnostics"], 2)
        self.assertEqual(result["storage_diagnostics"]["reader"]["ignored_torn_final_lines"], 1)
        self.assertNotIn("ae-control", result["storage_diagnostics"])
        self.assertIn("deterministic", text)
        json.dumps(result, allow_nan=False)
        counters["retained_events"] = 4
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        with self.assertRaisesRegex(ValueError, "invariants"):
            build_v8_report(folder, bootstrap_samples=20)

    def test_report_only_lock_argument_guards_and_cli_help(self):
        folder = self.report_fixture()
        with patch.object(driver, "check_dependencies") as deps, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(["--report-only", "--output-dir", str(folder)]), 0)
        deps.assert_not_called()
        with (folder / ".driver.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Evaluation is running"):
                driver.main(["--report-only", "--output-dir", str(folder)])
        for command in (["--models", "reader"], ["--models", "baseline", "ae-control"],
                        ["--models", "baseline", "short-control"], ["--models", "baseline", "memory-off"],
                        ["--models", "baseline", "--task-timeout", "nan"], ["--models", "baseline", "--n-episodes", "51"]):
            with self.assertRaises(ValueError):
                driver.validate_options(driver.build_parser().parse_args(command))
        for name in ("eval_long_memory_v8.py", "serve_long_memory_v8.py"):
            result = subprocess.run([sys.executable, str(ROOT / "run_scripts/robomme" / name), "--help"],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)


class V8OnlinePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("v8_policy_fixture", ROOT / "tests/test_long_memory_online_policy.py")
        cls.fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fixtures)

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def policy(self, *, mode=None, memory_off=False, source="moment"):
        from gr00t.long_memory.online_policy_v8 import LongMemoryV8Policy
        old = self.fixtures._policy(memory=False)
        result = LongMemoryV8Policy.__new__(LongMemoryV8Policy)
        result.__dict__.update(old.__dict__)
        result.mode, result.memory_off = mode or "none", memory_off
        result.stage, result.source = 1 if mode else 0, source if mode else None
        if mode:
            cfg = MemoryV8Config(feature_dim=6, state_dim=8, num_short_tokens=2,
                                  hidden_dim=8, num_heads=2, capacity=3, time_scale=2, source=source)
            result.memory = EventMemoryV8(cfg).eval().requires_grad_(False)
            with torch.no_grad():
                result.memory.fusion_projection.weight.normal_(std=.1)
        result.write_policy = "append_fifo" if mode == "event" else "none"
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

    def test_moment_extraction_is_pre_hamlet_once_normalized_and_preserves_raw_backbone(self):
        policy = self.policy(mode="event")
        for index, marker in enumerate((1, 13, 27)):
            raw = policy.model.backbone({"marker": marker})["backbone_features"].clone()
            expected = policy.model.action_head.vlln(raw)[:, -2:].float()
            with patch.object(policy.memory, "encode", wraps=policy.memory.encode) as encode:
                policy.get_action(self.fixtures._observation(marker=marker), self.fixtures._options(frame=index * 2, passive=True))
            torch.testing.assert_close(encode.call_args.args[0], expected, rtol=0, atol=0)
            head = policy.model.action_head
            torch.testing.assert_close(head.last_processed[:, :-2], head.vlln(raw)[:, :-2], rtol=0, atol=0)
            if index:
                self.assertFalse(torch.equal(encode.call_args.args[0], head.last_processed[:, -2:].float()))
        self.assertEqual(policy.model.action_head.process_calls, 3)

    def test_full_demo_writes_rng_isolation_reset_and_whole_event_eviction(self):
        policy = self.policy(mode="event")
        session = policy._session("A", 123, False)
        untouched = policy._session("B", 456, False).latent.clone()
        rng = session.generator.get_state().clone()
        for frame in (0, 2, 3, 5, 7):
            options = self.fixtures._options(frame=frame, passive=True, seed=123)
            _, info = policy.get_action(self.fixtures._observation(marker=frame + 1), options)
        diag = info["long_memory"]
        self.assertEqual((diag["appended_events"], diag["demo_appended_events"], diag["evicted_events"]), (5, 5, 2))
        self.assertEqual((diag["retained_events"], diag["memory_tokens"]), (3, 6))
        self.assertEqual(session.latent[0, ::2, -2].tolist(), [3, 5, 7])
        self.assertFalse(diag["memory_read_enabled"])
        self.assertEqual(policy.model.action_head.denoiser_calls, 0)
        self.assertTrue(torch.equal(rng, session.generator.get_state()))
        self.assertTrue(torch.equal(untouched, policy.sessions["B"].latent))
        self.assertEqual(policy.reset({"session_ids": ["A"]}), {"cleared_sessions": 1})
        self.assertIn("B", policy.sessions)
        self.assertEqual(policy._session("A", 123, False).latent.shape[1], 0)

    def test_online_offline_parity_both_sources_and_read_before_current_write(self):
        for source in ("moment", "short"):
            policy = self.policy(mode="event", source=source)
            episode = {"short": [], "moment": [], "state": [], "frames": [], "is_demo": []}
            for index, frame in enumerate((0, 2, 3, 5, 7, 9)):
                passive = index < 2
                previous = episode["frames"][-1] if episode["frames"] else 0
                controls = None if index < 3 else np.zeros((frame - previous, 8), np.float32)
                marker = index * 11 + 1
                moment = policy.model.action_head.vlln(policy.model.backbone({"marker": marker})["backbone_features"])[0, -2:].float()
                policy.get_action(self.fixtures._observation(marker=marker),
                    self.fixtures._options(frame=frame, passive=passive, controls=controls))
                session, head = policy.sessions["A"], policy.model.action_head
                episode["short"].append(head.last_processed[0, -2:].float())
                episode["moment"].append(moment)
                episode["state"].append(torch.from_numpy(np.concatenate(
                    [session.raw_states["joint_position"], session.raw_states["gripper_position"]], axis=-1)).reshape(-1))
                episode["frames"].append(frame)
                episode["is_demo"].append(passive)
                stacked = {key: torch.stack(value) if key in ("short", "moment", "state") else torch.tensor(value)
                           for key, value in episode.items()}
                if not passive:
                    before = replay_state(policy.memory, stacked, index, checkpoint_segment=0)
                    encoded = encode_at(policy.memory, stacked, index)
                    expected, _ = policy.memory.read(stacked["short"][-1][None], encoded, before)
                    torch.testing.assert_close(head.last_features[:, -2:], expected.to(head.last_features.dtype), rtol=0, atol=0)
                    self.assertTrue(torch.equal(head.last_features[:, :-2], head.last_processed[:, :-2]))
                after = replay_state(policy.memory, stacked, index + 1, checkpoint_segment=0)
                torch.testing.assert_close(session.latent, after, rtol=1e-6, atol=1e-6)
                self.assertEqual(session.latent.dtype, torch.float32)
                self.assertFalse(session.latent.requires_grad)

    def test_memory_off_preserves_writes_ae_control_has_none_and_measures_cast_delta(self):
        for mode, off in (("event", True), ("none", False)):
            policy = self.policy(mode=mode, memory_off=off)
            policy.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
            _, info = policy.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
            head = policy.model.action_head
            self.assertTrue(torch.equal(head.last_features, head.last_processed))
            self.assertEqual(info["long_memory"]["write_attempts"], 2 if mode == "event" else 0)
        policy = self.policy(mode="event")
        policy.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
        with patch.object(policy.memory, "read", side_effect=lambda short, *args, **kwargs: (short + 1, {})):
            _, info = policy.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        self.assertEqual(info["long_memory"]["read"]["ae_conditioning_changed_fraction"], 1)

    def test_bad_cadence_future_demo_and_nonfinite_action_fail_closed(self):
        policy = self.policy(mode="event")
        policy.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
        with self.assertRaisesRegex(ValueError, "cadence"):
            policy.get_action(self.fixtures._observation(), self.fixtures._options(frame=3))
        policy.model.action_head.fail_prediction = True
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite model action"):
            policy.get_action(self.fixtures._observation(), self.fixtures._options(frame=1))
        self.assertFalse(policy.sessions)
        self.assertIsNone(policy.model.action_head._memory_cache)
        self.assertIsNone(policy.model.action_head._inference_gen)

    def test_constructor_installs_then_loads_actor_expert_only_and_freezes(self):
        from gr00t.long_memory import online_policy_v8 as module
        cfg = MemoryV8Config(feature_dim=6, state_dim=8, num_short_tokens=2, hidden_dim=8, num_heads=2)
        original = self.fixtures._policy()
        original.model.action_head = nn.Linear(6, 6)
        info = {"config": {"stage": 1, "mode": "event", "memory": asdict(cfg),
                           "expert": {"rank": 2, "alpha": 4}, "expert_targets": ["example"], "train": {}}}
        def fake_init(policy, *args, **kwargs):
            policy.__dict__.update(original.__dict__)
        with patch.object(module, "v8_checkpoint_info", return_value=info), \
                patch.object(module.LongMemoryPolicy, "__init__", fake_init), \
                patch.object(module, "install_expert_lora") as install, \
                patch.object(module, "load_checkpoint_v8") as load, \
                patch.object(module, "set_expert_trainable") as freeze:
            policy = module.LongMemoryV8Policy("base", "bundle", device="cpu")
        install.assert_called_once()
        self.assertEqual(len(load.call_args.args), 3)
        self.assertIs(load.call_args.args[1], policy.memory)
        self.assertIs(load.call_args.args[2], original.model.action_head)
        freeze.assert_called_once_with(original.model.action_head, False)
        self.assertFalse(any(parameter.requires_grad for parameter in policy.memory.parameters()))
        self.assertEqual(policy.write_policy, "append_fifo")


if __name__ == "__main__":
    unittest.main()
