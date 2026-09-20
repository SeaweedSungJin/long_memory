"""CPU archive READ controls; no real model, policy server or simulator loads."""
import contextlib
import copy
from dataclasses import asdict
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn

from gr00t.long_memory import online_policy_v7 as inherited
from gr00t.long_memory.expert_v4 import ExpertLoRALinear, adapter_disabled, expert_state_dict
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from run_scripts.robomme import serve_archive_read_control_v7 as server

ROOT = Path(__file__).resolve().parents[1]


class ArchiveReadControlV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "archive_read_control_cpu_fixture", ROOT / "tests/test_long_memory_online_policy.py")
        cls.fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fixtures)

    def setUp(self):
        self.config = MemoryV7Config(feature_dim=6, state_dim=8, num_short_tokens=2,
                                    hidden_dim=8, num_heads=2, capacity=3, time_scale=2)
        self.targets = sorted("model.transformer_blocks.0.attn1." + name
                              for name in ("to_q", "to_k", "to_v", "to_out.0"))
        self.info = {"step": 12, "config": {
            "stage": 1, "mode": "archive", "memory": asdict(self.config),
            "expert": {"rank": 2, "alpha": 4}, "expert_targets": self.targets,
            "train": {"mode": "archive", "cvom_threshold": .05}}}

    def policy(self, off=False):
        """Mock only base-model construction and disk I/O, retaining V7 wiring."""
        fixture_head = self.fixtures._Head

        class CPUHead(nn.Module, fixture_head):
            def __init__(self):
                nn.Module.__init__(self)
                fixture_head.__init__(self)
                self.model = nn.Module()
                block, attention = nn.Module(), nn.Module()
                attention.to_q = nn.Linear(6, 6)
                attention.to_k = nn.Linear(6, 6)
                attention.to_v = nn.Linear(6, 6)
                attention.to_out = nn.ModuleList([nn.Linear(6, 6)])
                block.attn1 = attention
                self.model.transformer_blocks = nn.ModuleList([block])

            def get_action_with_features(self, features, state_features, embodiment, backbone):
                result = fixture_head.get_action_with_features(self, features, state_features, embodiment, backbone)
                # A real nonzero LoRA projection participates in the tiny expert.
                correction = self.model.transformer_blocks[0].attn1.to_q(features.float()).mean() * .01
                result["action_pred"] = result["action_pred"].float() + correction
                return result

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(731)
            base = self.fixtures._policy()
            base.model.action_head = CPUHead()
            trained = RecurrentMemoryV7(self.config)
            with torch.no_grad():
                trained.fusion_projection.weight.normal_(std=2)
                for name, parameter in trained.named_parameters():
                    if name.endswith("bias"):
                        parameter.normal_(std=.3)
            weights = copy.deepcopy(trained.state_dict())

            def fake_base_init(policy, *args, **kwargs):
                policy.__dict__.update(base.__dict__)

            def fake_load(checkpoint, memory, head, cvom):
                self.assertEqual(checkpoint, "archive-bundle")
                memory.load_state_dict(weights)
                with torch.no_grad():
                    for module in head.modules():
                        if isinstance(module, ExpertLoRALinear):
                            module.lora_A.fill_(.1)
                            module.lora_B.fill_(.25)

            with patch.object(server, "v7_checkpoint_info", return_value=self.info) as validate, \
                    patch.object(inherited, "v7_checkpoint_info", return_value=self.info), \
                    patch.object(inherited.LongMemoryPolicy, "__init__", fake_base_init), \
                    patch.object(inherited, "load_checkpoint_v7", side_effect=fake_load) as load:
                policy = server.ArchiveReadControlV7Policy(
                    "base", "archive-bundle", device="cpu", archive_read_off=off)
            validate.assert_called_once_with("base", "archive-bundle", expected_stage=1)
            load.assert_called_once()
        return policy

    def test_strict_stage_mode_step_and_boolean_guards_precede_base_load(self):
        for stage, mode, step in ((2, "archive", 12), (1, "recurrent", 12), (1, "none", 12),
                                  (True, "archive", 12), (1, "archive", 0), (1, "archive", True)):
            info = copy.deepcopy(self.info)
            info["config"].update(stage=stage, mode=mode)
            info["step"] = step
            with self.subTest(stage=stage, mode=mode, step=step), \
                    patch.object(server, "v7_checkpoint_info", return_value=info), \
                    patch.object(server.LongMemoryV7Policy, "__init__") as construct:
                with self.assertRaises(ValueError):
                    server.ArchiveReadControlV7Policy("base", "bundle", device="cpu")
                construct.assert_not_called()
        for bad in (None, 0, 1, "false"):
            with self.subTest(flag=bad), patch.object(server, "v7_checkpoint_info") as validate:
                with self.assertRaises(TypeError):
                    server.ArchiveReadControlV7Policy("base", "bundle", archive_read_off=bad)
                validate.assert_not_called()
        with patch.object(server, "v7_checkpoint_info") as validate:
            with self.assertRaisesRegex(ValueError, "require a trained"):
                server.ArchiveReadControlV7Policy("base", None)
            with self.assertRaisesRegex(ValueError, "APPEND"):
                server.ArchiveReadControlV7Policy("base", "bundle", write_policy="update")
            validate.assert_not_called()

    def test_payload_validation_failure_is_not_bypassed(self):
        with patch.object(server, "v7_checkpoint_info", side_effect=ValueError("V7 payload changed")), \
                patch.object(server.LongMemoryV7Policy, "__init__") as construct:
            with self.assertRaisesRegex(ValueError, "payload changed"):
                server.ArchiveReadControlV7Policy("base", "bundle", archive_read_off=True)
            construct.assert_not_called()

    def test_original_recurrent_only_memory_off_guard_remains_intact(self):
        with patch.object(inherited, "v7_checkpoint_info", return_value=self.info), \
                patch.object(inherited.LongMemoryPolicy, "__init__") as construct:
            with self.assertRaisesRegex(ValueError, "recurrent actor"):
                inherited.LongMemoryV7Policy("base", "bundle", memory_off=True)
            construct.assert_not_called()

    def test_same_trained_archive_and_adapted_expert_load_frozen(self):
        on, off = self.policy(), self.policy(True)
        for policy in (on, off):
            self.assertEqual((policy.stage, policy.mode, policy.write_policy), (1, "archive", "append"))
            for module in (policy.memory, policy.cvom, policy.model.action_head):
                self.assertFalse(module.training)
                self.assertFalse(any(p.requires_grad for p in module.parameters()))
            adapters = [m for m in policy.model.action_head.modules() if isinstance(m, ExpertLoRALinear)]
            self.assertEqual(len(adapters), 4)
            self.assertTrue(all(m.enabled and bool(m.lora_B.any()) for m in adapters))
        self.assertFalse(on.memory_off)
        self.assertTrue(off.memory_off)
        for name, value in on.memory.state_dict().items():
            self.assertTrue(torch.equal(value, off.memory.state_dict()[name]), name)
        for name, value in expert_state_dict(on.model.action_head).items():
            self.assertTrue(torch.equal(value, expert_state_dict(off.model.action_head)[name]), name)

    def test_read_off_is_exact_after_training_while_append_short_cache_and_rng_match(self):
        on, off = self.policy(), self.policy(True)
        first_rng = torch.Generator(device="cpu").manual_seed(17).get_state()
        global_rng = torch.random.get_rng_state().clone()
        observations, controls = self.fixtures._observation, np.zeros((2, 8), np.float32)
        sequence = [(0, True, None), (2, True, None), (4, False, None), (6, False, controls)]
        for index, (frame, passive, executed) in enumerate(sequence, start=1):
            info_by_role = []
            for policy in (on, off):
                _, info = policy.get_action(observations(marker=index), self.fixtures._options(
                    frame=frame, passive=passive, controls=executed))
                info_by_role.append(info["long_memory"])
                session = policy.sessions["A"]
                self.assertEqual(session.updates, index)
                self.assertEqual(session.write_attempts, index)
                self.assertEqual(session.keeps, 0)
                self.assertEqual(session.latent.shape[1], index * policy.n_q)
                self.assertEqual(info["long_memory"]["policy"], "append")
                if passive:
                    self.assertTrue(torch.equal(first_rng, session.generator.get_state()))
                    self.assertEqual(policy.model.action_head.denoiser_calls, 0)
                self.assertIsNone(policy.model.action_head._memory_cache)
                self.assertIsNone(policy.model.action_head._inference_gen)
            left, right = on.sessions["A"], off.sessions["A"]
            self.assertTrue(torch.equal(left.latent, right.latent))
            self.assertTrue(torch.equal(left.short_cache, right.short_cache))
            self.assertTrue(torch.equal(left.generator.get_state(), right.generator.get_state()))
            if not passive:
                head = off.model.action_head
                self.assertTrue(torch.equal(head.last_features, head.last_processed))
                self.assertFalse(info_by_role[1]["memory_read_enabled"])
                self.assertEqual(info_by_role[1]["read"]["residual_norm"], 0)
                self.assertEqual(info_by_role[1]["read"]["ae_conditioning_delta_norm"], 0)
                self.assertEqual(info_by_role[1]["read"]["ae_conditioning_changed_fraction"], 0)
                self.assertTrue(info_by_role[0]["memory_read_enabled"])
                self.assertGreater(info_by_role[0]["read"]["ae_conditioning_delta_norm"], 0)
        self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))
        self.assertFalse(torch.equal(first_rng, off.sessions["A"].generator.get_state()))
        self.assertEqual(off.sessions["A"].demo_updates, 2)

    def test_read_off_does_not_disable_nonzero_ae_adapters(self):
        adapted, baseline_expert = self.policy(True), self.policy(True)
        for policy in (adapted, baseline_expert):
            policy.get_action(self.fixtures._observation(), self.fixtures._options(passive=True))
        action, _ = adapted.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        with adapter_disabled(baseline_expert.model.action_head):
            original, _ = baseline_expert.get_action(self.fixtures._observation(), self.fixtures._options(frame=2))
        self.assertFalse(np.array_equal(action["joint_position"], original["joint_position"]))
        self.assertTrue(torch.equal(adapted.sessions["A"].generator.get_state(),
                                    baseline_expert.sessions["A"].generator.get_state()))
        self.assertTrue(all(m.enabled for m in adapted.model.action_head.modules()
                            if isinstance(m, ExpertLoRALinear)))

    def test_reset_and_failed_prediction_keep_inherited_session_contract(self):
        policy = self.policy(True)
        observation, options = self.fixtures._observation, self.fixtures._options
        first, _ = policy.get_action(observation(), options())
        policy.get_action(observation(marker=9), options(session="B", seed=99))
        other = policy.sessions["B"].latent.clone()
        self.assertEqual(policy.reset({"session_ids": ["A"]}), {"cleared_sessions": 1})
        again, info = policy.get_action(observation(), options())
        np.testing.assert_array_equal(first["joint_position"], again["joint_position"])
        self.assertEqual(info["long_memory"]["updates"], 1)
        self.assertTrue(torch.equal(other, policy.sessions["B"].latent))
        policy.model.action_head.fail_prediction = True
        with self.assertRaises(FloatingPointError):
            policy.get_action(observation(), options(frame=2, controls=np.zeros((2, 8), np.float32)))
        self.assertNotIn("A", policy.sessions)
        self.assertIn("B", policy.sessions)
        self.assertIsNone(policy.model.action_head._memory_cache)
        self.assertIsNone(policy.model.action_head._inference_gen)

    def test_help_and_role_dispatch_are_explicit(self):
        result = subprocess.run([sys.executable, "-B", "-S", str(Path(server.__file__)), "--help"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--archive-read-off", result.stdout)
        self.assertIn("--memory-checkpoint", result.stdout)
        parser = server.build_parser()
        with patch.object(server, "ArchiveReadControlV7Policy") as archive, \
                patch.object(server, "LongMemoryV7Policy") as baseline:
            args = parser.parse_args(["--base-model", "base", "--device", "cpu"])
            server.create_policy(args)
            baseline.assert_called_once_with("base", device="cpu", write_policy="checkpoint", memory_off=False)
            archive.assert_not_called()
            args = parser.parse_args(["--base-model", "base", "--memory-checkpoint", "bundle",
                                      "--archive-read-off", "--device", "cpu"])
            server.create_policy(args)
            archive.assert_called_once_with("base", "bundle", device="cpu", write_policy="checkpoint",
                                             archive_read_off=True)
        with patch.object(server, "create_policy") as create:
            with self.assertRaisesRegex(ValueError, "requires --memory-checkpoint"):
                server.main(["--base-model", "base", "--archive-read-off"])
            create.assert_not_called()
        for invalid in (["--write-policy", "update"], ["--memory-off"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["--base-model", "base", *invalid])

    def test_server_cleanup_on_normal_exit_and_failure_without_opening_sockets(self):
        for failure in (None, KeyboardInterrupt(), RuntimeError("serve failed")):
            fake_policy = MagicMock(stage=1, mode="archive", write_policy="append", memory_off=True)
            fake_server = MagicMock()
            fake_server.run.side_effect = failure
            with self.subTest(failure=type(failure).__name__), \
                    patch.object(server, "create_policy", return_value=fake_policy), \
                    patch("gr00t.policy.gr00t_policy.Gr00tSimPolicyWrapper", return_value="wrapped") as wrapper, \
                    patch("gr00t.policy.server_client.PolicyServer", return_value=fake_server) as constructor, \
                    contextlib.redirect_stdout(io.StringIO()):
                argv = ["--base-model", "base", "--memory-checkpoint", "bundle", "--archive-read-off",
                        "--device", "cpu", "--port", "5432"]
                if isinstance(failure, RuntimeError):
                    with self.assertRaisesRegex(RuntimeError, "serve failed"):
                        server.main(argv)
                else:
                    self.assertEqual(server.main(argv), 0)
            wrapper.assert_called_once_with(fake_policy)
            constructor.assert_called_once_with("wrapped", host="127.0.0.1", port=5432)
            fake_policy.reset.assert_called_once_with()
            fake_server.socket.close.assert_called_once_with(linger=0)
            fake_server.context.term.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
