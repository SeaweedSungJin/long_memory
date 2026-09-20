"""CPU V10 online wiring and unchanged archive causality; no simulator/GPU."""
import contextlib
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file

from gr00t.long_memory.checkpoint_v7 import v7_checkpoint_info
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import ExpertLoRALinear, LoRAConfig, expert_state_dict, install_expert_lora
from gr00t.long_memory.hamlet import checkpoint_identity
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from gr00t.long_memory.replay_v7 import replay_state, encode_at
from run_scripts.robomme import policy_archive_projector_v10 as policy_module
from run_scripts.robomme import serve_archive_projector_v10 as server
from run_scripts.robomme.checkpoint_projector_v10 import save_checkpoint
from run_scripts.robomme.projector_adapter_v10 import all_adapters_disabled, install_projector, projector_state_dict, set_trainable
from tests import test_long_memory_online_policy as fixture


class CPUHead(nn.Module, fixture._Head):
    def __init__(self):
        nn.Module.__init__(self)
        fixture._Head.__init__(self)
        self.model = nn.Module()
        block, attention = nn.Module(), nn.Module()
        for name in ("to_q", "to_k", "to_v"):
            setattr(attention, name, nn.Linear(6, 6))
        attention.to_out = nn.ModuleList([nn.Linear(6, 6)])
        block.attn1 = attention
        self.model.transformer_blocks = nn.ModuleList([block])
        self.model.proj_out_2 = nn.Linear(6, 6)

    def get_action_with_features(self, features, state_features, embodiment, backbone):
        result = fixture._Head.get_action_with_features(self, features, state_features, embodiment, backbone)
        adapted = self.model.transformer_blocks[0].attn1.to_q(features.float())
        correction = self.model.proj_out_2(adapted).mean() * .01
        result["action_pred"] = result["action_pred"].float() + correction
        return result


class ArchiveProjectorPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.config = MemoryV7Config(feature_dim=6, state_dim=8, num_short_tokens=2,
            hidden_dim=8, num_heads=2, capacity=3, time_scale=2)
        self.targets = sorted("model.transformer_blocks.0.attn1." + name
            for name in ("to_q", "to_k", "to_v", "to_out.0"))
        self.info = {"step": 32, "config": {"trainer_variant": "archive_projector_v10",
            "driver_variant": "archive_projector_v10", "stage": 1, "mode": "archive",
            "memory": asdict(self.config), "expert": {"rank": 2, "alpha": 4}, "expert_targets": self.targets,
            "projector": {"target": "model.proj_out_2", "in_features": 6, "out_features": 6,
                "bias": True, "kind": "full_rank_residual_fp32", "enabled": True}}}

    def policy(self, off=False, enabled=True, step=32):
        info = copy.deepcopy(self.info)
        info["config"]["projector"]["enabled"] = enabled
        info["step"] = step
        events = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(731)
            base = fixture._policy()
            base.model.action_head = CPUHead()
            trained = RecurrentMemoryV7(self.config)
            with torch.no_grad():
                trained.fusion_projection.weight.normal_(std=2)
                for name, parameter in trained.named_parameters():
                    if name.endswith("bias"):
                        parameter.normal_(std=.3)
            weights = copy.deepcopy(trained.state_dict())

            def fake_validate(base_model, checkpoint, expected_stage):
                self.assertEqual((base_model, checkpoint, expected_stage), ("base", "v10-bundle", 1))
                events.append("validated")
                return info

            def fake_init(policy, *args, **kwargs):
                self.assertEqual(events, ["validated"])
                self.assertIsNone(kwargs["memory_checkpoint"])
                events.append("base_loaded")
                policy.__dict__.update(base.__dict__)

            def fake_load(checkpoint, memory, head, cvom):
                self.assertEqual(checkpoint, "v10-bundle")
                self.assertEqual(events, ["validated", "base_loaded"])
                self.assertFalse(any(p.requires_grad for p in head.parameters()))
                events.append("v10_loaded")
                memory.load_state_dict(weights)
                with torch.no_grad():
                    for module in head.modules():
                        if isinstance(module, ExpertLoRALinear):
                            module.lora_A.fill_(.1)
                            module.lora_B.fill_(.25)
                    if enabled and step > 0:
                        head.model.proj_out_2.delta_weight.fill_(.02)
                        head.model.proj_out_2.delta_bias.fill_(.01)

            with patch.object(policy_module, "checkpoint_info", side_effect=fake_validate), \
                 patch.object(policy_module.LongMemoryPolicy, "__init__", fake_init), \
                 patch.object(policy_module, "load_checkpoint", side_effect=fake_load), \
                 patch("gr00t.long_memory.online_policy_v7.v7_checkpoint_info", side_effect=AssertionError("V10 must not use old loader")):
                policy = policy_module.ArchiveProjectorV10Policy("base", "v10-bundle", device="cpu", archive_read_off=off)
        self.assertEqual(events, ["validated", "base_loaded", "v10_loaded"])
        return policy

    def test_preflight_variant_flags_and_corruption_fail_before_base(self):
        for change in ({"stage": 2}, {"mode": "recurrent"}, {"trainer_variant": "recurrent_memory_v7"}):
            info = copy.deepcopy(self.info)
            info["config"].update(change)
            with patch.object(policy_module, "checkpoint_info", return_value=info), \
                 patch.object(policy_module.LongMemoryPolicy, "__init__") as load, self.assertRaises(ValueError):
                policy_module.ArchiveProjectorV10Policy("base", "bundle", device="cpu")
            load.assert_not_called()
        with patch.object(policy_module, "checkpoint_info", side_effect=ValueError("corrupt projector payload")), \
             patch.object(policy_module.LongMemoryPolicy, "__init__") as load, self.assertRaisesRegex(ValueError, "corrupt"):
            policy_module.ArchiveProjectorV10Policy("base", "bundle", device="cpu")
        load.assert_not_called()
        for invalid in (None, 0, "false"):
            with patch.object(policy_module, "checkpoint_info") as validate, self.assertRaises(TypeError):
                policy_module.ArchiveProjectorV10Policy("base", "bundle", archive_read_off=invalid)
            validate.assert_not_called()
        with self.assertRaises(ValueError):
            policy_module.ArchiveProjectorV10Policy("base", None)
        with self.assertRaises(ValueError):
            policy_module.ArchiveProjectorV10Policy("base", "bundle", write_policy="update")

    def test_both_policy_controls_load_same_adapters_frozen_and_preserve_enabled(self):
        on, off = self.policy(), self.policy(True)
        for policy in (on, off):
            self.assertEqual((policy.stage, policy.mode, policy.write_policy), (1, "archive", "append"))
            for module in (policy.model.action_head, policy.memory, policy.cvom):
                self.assertFalse(module.training)
                self.assertFalse(any(p.requires_grad for p in module.parameters()))
            self.assertTrue(policy.model.action_head.model.proj_out_2.enabled)
            self.assertTrue(bool(policy.model.action_head.model.proj_out_2.delta_weight.any()))
        for state_function in (expert_state_dict, projector_state_dict):
            a, b = state_function(on.model.action_head), state_function(off.model.action_head)
            self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))
        control = self.policy(enabled=False)
        self.assertFalse(control.model.action_head.model.proj_out_2.enabled)
        self.assertFalse(any(bool(v.any()) for v in projector_state_dict(control.model.action_head).values()))

    def test_read_off_retains_append_short_cache_rng_and_original_read_before_write(self):
        on, off = self.policy(), self.policy(True)
        global_rng = torch.get_rng_state().clone()
        episode = {"short": [], "state": [], "frames": [], "is_demo": []}
        initial_rng = torch.Generator().manual_seed(17).get_state()
        for index, (frame, passive) in enumerate(((0, True), (2, True), (3, False), (5, False))):
            controls = None if index < 3 else np.zeros((2, 8), np.float32)
            infos = []
            for policy in (on, off):
                _, info = policy.get_action(fixture._observation(marker=index + 1), fixture._options(
                    frame=frame, passive=passive, controls=controls))
                infos.append(info)
                session, head = policy.sessions["A"], policy.model.action_head
                self.assertEqual(session.updates, index + 1)
                self.assertEqual(session.write_attempts, index + 1)
                self.assertEqual(session.latent.shape[1], (index + 1) * 2)
                self.assertIsNone(head._memory_cache)
                self.assertIsNone(head._inference_gen)
                if passive:
                    self.assertEqual(head.denoiser_calls, 0)
                    self.assertTrue(torch.equal(initial_rng, session.generator.get_state()))
            left, right = on.sessions["A"], off.sessions["A"]
            self.assertTrue(torch.equal(left.latent, right.latent))
            self.assertTrue(torch.equal(left.short_cache, right.short_cache))
            self.assertTrue(torch.equal(left.generator.get_state(), right.generator.get_state()))
            head = on.model.action_head
            episode["short"].append(head.last_processed[0, -2:].float())
            episode["state"].append(torch.from_numpy(np.concatenate(
                [left.raw_states["joint_position"], left.raw_states["gripper_position"]], axis=-1)).reshape(-1))
            episode["frames"].append(frame)
            episode["is_demo"].append(passive)
            if not passive:
                before = replay_state(on.memory, episode, index, mode="archive", checkpoint_segment=0)
                expected, _ = on.memory.read(episode["short"][-1][None], encode_at(on.memory, episode, index), before, mode="archive")
                self.assertTrue(torch.equal(head.last_features[:, -2:], expected.to(head.last_features.dtype)))
                self.assertTrue(torch.equal(off.model.action_head.last_features, off.model.action_head.last_processed))
                self.assertFalse(infos[1]["long_memory"]["memory_read_enabled"])
                self.assertEqual(infos[1]["long_memory"]["read"]["ae_conditioning_delta_norm"], 0.)
            self.assertTrue(torch.equal(left.latent, replay_state(on.memory, episode, index + 1, mode="archive", checkpoint_segment=0)))
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))

    def test_read_off_keeps_nonzero_projector_and_lora_action_contribution(self):
        adapted, disabled = self.policy(True), self.policy(True)
        for policy in (adapted, disabled):
            policy.get_action(fixture._observation(), fixture._options(passive=True))
        actual, info = adapted.get_action(fixture._observation(), fixture._options(frame=2))
        with all_adapters_disabled(disabled.model.action_head):
            original, _ = disabled.get_action(fixture._observation(), fixture._options(frame=2))
        self.assertFalse(np.array_equal(actual["joint_position"], original["joint_position"]))
        self.assertTrue(torch.equal(adapted.sessions["A"].generator.get_state(), disabled.sessions["A"].generator.get_state()))
        self.assertEqual(info["checkpoint_variant"], "archive_projector_v10")
        self.assertEqual(info["checkpoint_step"], 32)
        self.assertTrue(info["projector_enabled"])

    def test_step_zero_is_honest_and_session_reset_failure_cleanup_unchanged(self):
        policy = self.policy(True, step=0)
        first, info = policy.get_action(fixture._observation(), fixture._options())
        self.assertEqual(info["checkpoint_step"], 0)
        policy.get_action(fixture._observation(marker=9), fixture._options(session="B", seed=44))
        other = policy.sessions["B"].latent.clone()
        self.assertEqual(policy.reset({"session_ids": ["A"]}), {"cleared_sessions": 1})
        again, _ = policy.get_action(fixture._observation(), fixture._options())
        np.testing.assert_array_equal(first["joint_position"], again["joint_position"])
        self.assertTrue(torch.equal(policy.sessions["B"].latent, other))
        policy.model.action_head.fail_prediction = True
        with self.assertRaises(FloatingPointError):
            policy.get_action(fixture._observation(), fixture._options(frame=2, controls=np.zeros((2, 8), np.float32)))
        self.assertNotIn("A", policy.sessions)
        self.assertIn("B", policy.sessions)
        self.assertIsNone(policy.model.action_head._memory_cache)
        self.assertIsNone(policy.model.action_head._inference_gen)

    def test_actual_v10_bundle_loads_frozen_and_old_v7_loader_rejects_it(self):
        with tempfile.TemporaryDirectory() as temporary, torch.random.fork_rng(devices=[]):
            root = Path(temporary)
            base_path = root / "base"
            base_path.mkdir()
            native = CPUHead().eval().requires_grad_(False)
            native_state = copy.deepcopy(native.state_dict())
            state = {"action_head." + key: value for key, value in native_state.items()}
            save_file(state, str(base_path / "model.safetensors"))
            (base_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                key: "model.safetensors" for key in state}}))
            (base_path / "config.json").write_text(json.dumps({"hamlet_mode": "finetune", "memory_type": "moment_token",
                "mem_cond_type": "cross_attn", "n_moment_tokens": 2, "memory_stride": 2, "backbone_embedding_dim": 6}))
            (base_path / "processor_config.json").write_text(json.dumps({"max_state_dim": 8}))
            for name in ("statistics.json", "embodiment_id.json"):
                (base_path / name).write_text("{}")
            original_bytes = {p.name: p.read_bytes() for p in base_path.iterdir()}
            targets = install_expert_lora(native, LoRAConfig(2, 4))
            spec = install_projector(native)
            set_trainable(native, False, False)
            with torch.no_grad():
                native.model.proj_out_2.delta_bias.fill_(.07)
                for module in native.modules():
                    if isinstance(module, ExpertLoRALinear):
                        module.lora_B.fill_(.03)
            memory, critic = RecurrentMemoryV7(self.config), CVOMV7(self.config)
            config = {**copy.deepcopy(self.info["config"]), "expert_targets": targets, "projector": spec}
            checkpoint = save_checkpoint(root / "saved", 32, memory, native, critic, None, config,
                {"base_model": checkpoint_identity(base_path), "cache_fingerprint": "toy-v10"})
            with self.assertRaisesRegex(ValueError, "Expected recurrent_memory_v7"):
                v7_checkpoint_info(base_path, checkpoint)

            def base_init(policy, *args, **kwargs):
                base = fixture._policy()
                base.model.action_head = CPUHead().eval().requires_grad_(False)
                base.model.action_head.load_state_dict(native_state)
                policy.__dict__.update(base.__dict__)

            with patch.object(policy_module.LongMemoryPolicy, "__init__", base_init):
                loaded = policy_module.ArchiveProjectorV10Policy(base_path, checkpoint, device="cpu", archive_read_off=True)
            self.assertTrue(all(torch.equal(v, projector_state_dict(loaded.model.action_head)[k])
                                for k, v in projector_state_dict(native).items()))
            self.assertTrue(all(torch.equal(v, expert_state_dict(loaded.model.action_head)[k])
                                for k, v in expert_state_dict(native).items()))
            self.assertFalse(any(p.requires_grad for p in loaded.model.action_head.parameters()))
            loaded.get_action(fixture._observation(), fixture._options(passive=True))
            actions, info = loaded.get_action(fixture._observation(), fixture._options(frame=2))
            self.assertTrue(np.isfinite(actions["joint_position"]).all())
            self.assertTrue(info["projector_enabled"])
            self.assertFalse(info["long_memory"]["memory_read_enabled"])
            self.assertEqual(original_bytes, {p.name: p.read_bytes() for p in base_path.iterdir()})

    def test_cli_help_clean_baseline_dispatch_and_invalid_arguments(self):
        result = subprocess.run([sys.executable, "-B", "-S", str(Path(server.__file__)), "--help"],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--archive-read-off", result.stdout)
        parser = server.build_parser()
        with patch.object(server, "LongMemoryV7Policy") as baseline, patch.object(server, "ArchiveProjectorV10Policy") as memory:
            server.create_policy(parser.parse_args(["--base-model", "base", "--device", "cpu"]))
            baseline.assert_called_once_with("base", device="cpu", write_policy="checkpoint", memory_off=False)
            memory.assert_not_called()
            server.create_policy(parser.parse_args(["--base-model", "base", "--memory-checkpoint", "bundle", "--archive-read-off"]))
            memory.assert_called_once_with("base", "bundle", device="cuda:0", write_policy="checkpoint", archive_read_off=True)
        for invalid in (["--archive-read-off"], ["--port", "0"], ["--port", "65536"], ["--host", ""]):
            with patch.object(server, "create_policy") as create, self.assertRaises(ValueError):
                server.main(["--base-model", "base", *invalid])
            create.assert_not_called()

    def test_server_cleanup_without_opening_sockets(self):
        for failure in (None, KeyboardInterrupt(), RuntimeError("server failure")):
            policy = MagicMock(stage=1, mode="archive", write_policy="append", memory_off=True, checkpoint_step=32)
            transport = MagicMock()
            transport.run.side_effect = failure
            with patch.object(server, "create_policy", return_value=policy), \
                 patch("gr00t.policy.gr00t_policy.Gr00tSimPolicyWrapper", return_value="wrapped"), \
                 patch("gr00t.policy.server_client.PolicyServer", return_value=transport), \
                 contextlib.redirect_stdout(io.StringIO()):
                if isinstance(failure, RuntimeError):
                    with self.assertRaisesRegex(RuntimeError, "server failure"):
                        server.main(["--base-model", "base"])
                else:
                    self.assertEqual(server.main(["--base-model", "base"]), 0)
            policy.reset.assert_called_once_with()
            transport.socket.close.assert_called_once_with(linger=0)
            transport.context.term.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
