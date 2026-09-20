"""CPU V14 constructor/serial action/ingest tests; no real model or simulator."""
import contextlib
import copy
from dataclasses import asdict
import io
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn

from gr00t.long_memory.expert_v4 import (
    ExpertLoRALinear, LoRAConfig, expert_state_dict, install_expert_lora,
    load_expert_state_dict,
)
from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
from run_scripts.robomme import policy_visual_expert_v14 as policy_module
from run_scripts.robomme import serve_visual_expert_v14 as server
from run_scripts.robomme.visual_demo_tail_bank_v13 import VisualDemoTailMemoryV13, VisualDifferentialConfig
from tests import test_policy_visual_patch_v11 as legacy
from tests.test_long_memory_v4_expert import _Attention
from tests.test_policy_demo_tail_ingest_v13 import ingest, request


class ExpertCPUHead(legacy.CPUHead):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.transformer_blocks = nn.ModuleList([nn.Module() for _ in range(32)])
        for block in self.model.transformer_blocks:
            block.attn1 = _Attention()

    def get_action_with_features(self, features, state_features, embodiment, backbone):
        result = super().get_action_with_features(features, state_features, embodiment, backbone)
        # Exercise an ACTUAL installed adapter in the tiny action output; the
        # remaining 127 wrappers still participate in identity/weight guards.
        projection = self.model.transformer_blocks[0].attn1.to_q
        result["action_pred"] += projection(features.float()).mean() * .3
        return result


def tiny_parent():
    actor = legacy.parent_policy()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(443)
        actor.model.action_head = ExpertCPUHead()
        install_expert_lora(actor.model.action_head, LoRAConfig())
        with torch.no_grad():
            for name, parameter in actor.model.action_head.named_parameters():
                if name.endswith("lora_B"):
                    parameter.fill_(.01)
    model = actor.model
    model.training = False
    model.parameters = model.action_head.parameters
    model.eval = lambda: (model.action_head.eval(), model)[1]
    model.requires_grad_ = lambda flag: (model.action_head.requires_grad_(flag), model)[1]
    model.eval().requires_grad_(False)
    actor.stride = 16
    return actor


def mock_info(parent, step=0):
    targets = sorted(name for name, module in parent.model.action_head.named_modules()
                     if isinstance(module, ExpertLoRALinear))
    semantics = dict(include_tail=True, read_mode="differential", replay_encoding="framewise")
    return {"step": step,
        "config": {"trainer_variant": "visual_expert_v14", "stage": 1,
            "mode": "visual_expert", "architecture": "visual_demo_tail_v13", **semantics,
            "visual": asdict(legacy.VISUAL_CONFIG), "camera_order": list(policy_module.CAMERA_ORDER),
            "expert": {"rank": 8, "alpha": 16.}, "expert_targets": targets, "train": dict(semantics)},
        "metadata": {**semantics, "initial_parent": {"path": "archive1250", "step": 1250,
            "files_sha256": {"checkpoint.json": "b" * 64}},
            "payload_sha256": {"visual.safetensors": "a" * 64, "expert.safetensors": "c" * 64}}}


class VisualExpertPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    call = legacy.VisualPatchPolicyTests.call
    assert_actions_equal = legacy.VisualPatchPolicyTests.assert_actions_equal

    def actor(self, *, adapted=False, awake=False, off=False, step=0):
        parent = tiny_parent()
        info = mock_info(parent, step=step)
        expert = expert_state_dict(parent.model.action_head)
        if adapted:
            for name, tensor in expert.items():
                if name.endswith("lora_B"):
                    tensor.add_(.12)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(911)
            visual = VisualDemoTailMemoryV13(VisualDifferentialConfig(**asdict(legacy.VISUAL_CONFIG)))
            if awake:
                with torch.no_grad():
                    visual.output_projection.weight.normal_(std=.05)
        state, events = copy.deepcopy(visual.state_dict()), []

        def preflight(base, checkpoint, expected_stage):
            self.assertEqual((base, checkpoint, expected_stage), ("base", "joint", 1))
            events.append("preflight")
            return copy.deepcopy(info)

        def initialize(actor, base, checkpoint, **kwargs):
            self.assertEqual(events, ["preflight"])
            self.assertEqual((base, checkpoint), ("base", "archive1250"))
            self.assertEqual(kwargs, dict(device="cpu", strict=True, write_policy="checkpoint", memory_off=False))
            actor.__dict__.update(parent.__dict__)
            events.append("initial_parent")

        def load(checkpoint, visual, head):
            self.assertEqual(events, ["preflight", "initial_parent"])
            self.assertEqual(checkpoint, "joint")
            self.assertIs(head, parent.model.action_head)
            visual.load_state_dict(state)
            load_expert_state_dict(head, expert)
            events.append("both_payloads")
            return copy.deepcopy(info)

        with patch.object(policy_module, "checkpoint_info", side_effect=preflight), \
                patch.object(LongMemoryV7Policy, "__init__", initialize), \
                patch.object(policy_module, "load_checkpoint", side_effect=load):
            actor = policy_module.VisualExpertV14Policy("base", "joint", device="cpu", visual_read_off=off)
        self.assertEqual(events, ["preflight", "initial_parent", "both_payloads"])
        return actor

    def test_initialization_exact_parent_actions_and_truthful_new_diagnostics(self):
        parent, on, off = tiny_parent(), self.actor(), self.actor(off=True)
        rng = torch.get_rng_state().clone()
        for index, frame in enumerate((0, 16)):
            controls = np.zeros((16, 8), np.float32) if index else None
            expected, parent_info = self.call(parent, marker=index + 1, frame=frame, controls=controls)
            for actor in (on, off):
                result, info = self.call(actor, marker=index + 1, frame=frame, controls=controls)
                self.assert_actions_equal(expected, result)
                self.assertEqual(info["long_memory"], parent_info["long_memory"])
                self.assertEqual(info["checkpoint_variant"], "visual_expert_v14")
                self.assertEqual(info["checkpoint_step"], 0)
                self.assertEqual(info["expert_adapter_count"], 128)
                self.assertTrue(info["expert_adapters_enabled"])
                self.assertEqual(info["expert_weights_sha256"], "c" * 64)
                self.assertEqual(info["initial_parent_checkpoint_sha256"], "b" * 64)
                self.assertNotIn("frozen_parent_checkpoint_sha256", info)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_off_retains_saved_adapted_expert_not_initial_parent(self):
        original, off = tiny_parent(), self.actor(adapted=True, awake=True, off=True, step=4)
        adapted_reference = tiny_parent()
        load_expert_state_dict(adapted_reference.model.action_head, expert_state_dict(off.model.action_head))
        for index, frame in enumerate((0, 16)):
            opts = dict(marker=index + 1, frame=frame,
                        controls=np.zeros((16, 8), np.float32) if index else None)
            base_actions, _ = self.call(original, **opts)
            expected, _ = self.call(adapted_reference, **opts)
            actual, info = self.call(off, **opts)
            self.assert_actions_equal(actual, expected)
            self.assertTrue(any(not np.array_equal(actual[k], base_actions[k]) for k in actual))
            self.assertFalse(info["visual_memory"]["read_enabled"])
            self.assertEqual(info["visual_memory"]["image_delta_norm"], 0.)
            self.assertTrue(all(module.enabled for _, module in off._expert_adapters))

    def test_required_rgb_ingest_both_roles_and_reset(self):
        for off in (False, True):
            actor = self.actor(adapted=True, off=off)
            for frame in (0, 16, 32):
                self.call(actor, marker=frame + 1, frame=frame, passive=True)
            with self.assertRaisesRegex(ValueError, "ingest|tail|Tail"):
                self.call(actor, frame=48)
            self.assertFalse(actor.sessions)
            for frame in (0, 16, 32):
                self.call(actor, marker=frame + 1, frame=frame, passive=True)
            weights = expert_state_dict(actor.model.action_head)
            ingest(actor, request("A"))
            _, info = self.call(actor, marker=4, frame=48)
            self.assertEqual(info["demo_tail"]["tail_observations"], 15)
            self.assertEqual(info["demo_tail"]["effective_prior_observations"], 3 if off else 18)
            for name, tensor in weights.items():
                self.assertTrue(torch.equal(tensor, expert_state_dict(actor.model.action_head)[name]))
            actor.reset()
            self.assertFalse(actor.sessions)
            self.assertFalse(actor._visual_call_lock.locked())

    def test_all_modules_frozen_rng_and_runtime_adapter_bypass_rejected(self):
        rng = torch.get_rng_state().clone()
        actor = self.actor(adapted=True, awake=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for module in (actor.model, actor.memory, actor.cvom, actor.visual_memory):
            self.assertFalse(module.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
        self.call(actor)
        actor._expert_adapters[0][1].enabled = False
        with self.assertRaisesRegex(RuntimeError, "saved adapted expert"):
            self.call(actor, frame=16, controls=np.zeros((16, 8), np.float32))
        self.assertFalse(actor.sessions)
        self.assertFalse(actor._visual_call_lock.locked())

    def test_invalid_new_identity_and_preflight_hash_error_before_model(self):
        info = mock_info(tiny_parent())
        changes = [lambda x: x["config"].update(trainer_variant="visual_demo_tail_v13"),
            lambda x: x["config"].update(mode="visual_demo_tail"),
            lambda x: x["config"].update(stage=2),
            lambda x: x["config"]["train"].update(include_tail=False),
            lambda x: x["metadata"].update(read_mode="current_only"),
            lambda x: x["metadata"].update(frozen_parent={}),
            lambda x: x.update(step=True)]
        for change in changes:
            bad = copy.deepcopy(info); change(bad)
            with patch.object(policy_module, "checkpoint_info", return_value=bad), \
                    patch.object(LongMemoryV7Policy, "__init__") as model, self.assertRaises(ValueError):
                policy_module.VisualExpertV14Policy("base", "joint", device="cpu")
            model.assert_not_called()
        with patch.object(policy_module, "checkpoint_info", side_effect=ValueError("expert payload hash changed")), \
                patch.object(LongMemoryV7Policy, "__init__") as model, self.assertRaisesRegex(ValueError, "hash"):
            policy_module.VisualExpertV14Policy("base", "joint", device="cpu")
        model.assert_not_called()

    def test_changed_identity_during_actual_load_and_decode_failure_cleanup(self):
        info = mock_info(tiny_parent())
        def initialize(actor, *args, **kwargs):
            actor.__dict__.update(tiny_parent().__dict__)
        changed = copy.deepcopy(info); changed["metadata"]["payload_sha256"]["expert.safetensors"] = "d" * 64
        with patch.object(policy_module, "checkpoint_info", return_value=info), \
                patch.object(LongMemoryV7Policy, "__init__", initialize), \
                patch.object(policy_module, "load_checkpoint", return_value=changed), \
                self.assertRaisesRegex(ValueError, "identity changed"):
            policy_module.VisualExpertV14Policy("base", "joint", device="cpu")
        actor = self.actor(adapted=True)
        with patch.object(actor.processor, "decode_action", side_effect=RuntimeError("decode failure")):
            with self.assertRaisesRegex(RuntimeError, "decode failure"):
                self.call(actor)
        self.assertFalse(actor.sessions)
        self.assertFalse(actor._visual_call_lock.locked())
        self.assertNotIn("get_action_with_features", actor.model.action_head.__dict__)
        self.assertNotIn("process_backbone_output", actor.model.action_head.__dict__)

    def test_genuine_saved_v14_both_payloads_load_without_any_v13_loader(self):
        from tests.test_checkpoint_visual_expert_v14 import JointCheckpointTests
        from run_scripts.robomme import checkpoint_demo_tail_v13 as old
        fixture = JointCheckpointTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)

        def original_parent():
            actor = tiny_parent()
            # Only heavy original-model construction is mocked. The genuine
            # parent-derived installed expert and both V14 payload loaders run.
            actor.model.action_head.model = fixture.make_head().model
            actor.model.eval().requires_grad_(False)
            return actor

        def initialize(actor, base, checkpoint, **kwargs):
            self.assertEqual((base, checkpoint), (fixture.base, str(fixture.parent)))
            actor.__dict__.update(original_parent().__dict__)

        for step in (0, 4):
            if step:
                with torch.no_grad():
                    for name, parameter in fixture.head.named_parameters():
                        if name.endswith("lora_B"):
                            parameter.add_(.15)
                    fixture.visual.output_projection.weight.fill_(.025)
            bundle = fixture.save(name=f"genuine_{step}", step=step)
            for off in (False, True):
                with patch.object(LongMemoryV7Policy, "__init__", initialize), \
                        patch.object(old, "checkpoint_info", side_effect=AssertionError("legacy info used")), \
                        patch.object(old, "load_checkpoint", side_effect=AssertionError("legacy load used")):
                    actor = policy_module.VisualExpertV14Policy(fixture.base, bundle, device="cpu", visual_read_off=off)
                fixture.assert_tree(expert_state_dict(fixture.head), expert_state_dict(actor.model.action_head))
                fixture.assert_tree(fixture.visual.state_dict(), actor.visual_memory.state_dict())
                self.assertEqual(actor.checkpoint_step, step)
                expected = original_parent()
                load_expert_state_dict(expected.model.action_head, expert_state_dict(fixture.head))
                reference, _ = self.call(expected)
                result, info = self.call(actor)  # Empty history: only saved AE can affect output.
                self.assert_actions_equal(result, reference)
                self.assertEqual(info["checkpoint_variant"], "visual_expert_v14")
                self.assertEqual(info["expert_adapter_count"], len(fixture.config["expert_targets"]))
                self.assertFalse(info["visual_memory"]["read_enabled"])

    def test_genuine_expert_payload_hash_corruption_fails_before_model_allocation(self):
        from tests.test_checkpoint_visual_expert_v14 import JointCheckpointTests
        fixture = JointCheckpointTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        bundle = fixture.save()
        payload = bundle / "expert.safetensors"
        payload.write_bytes(payload.read_bytes() + b"corrupted")
        with patch.object(LongMemoryV7Policy, "__init__") as model, self.assertRaises(ValueError):
            policy_module.VisualExpertV14Policy(fixture.base, bundle, device="cpu")
        model.assert_not_called()

    def test_server_help_original_baseline_route_and_strict_flags(self):
        result = subprocess.run([sys.executable, "-B", "-S", server.__file__, "--help"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--visual-read-off", result.stdout)
        args = server.build_parser().parse_args(["--base-model", "base"])
        with patch.object(LongMemoryV7Policy, "__init__", return_value=None) as constructor:
            policy = server.create_policy(args)
        self.assertIs(type(policy), LongMemoryV7Policy)
        constructor.assert_called_once_with("base", memory_checkpoint=None, device="cuda:0",
                                            write_policy="checkpoint", memory_off=False)
        with patch.object(policy_module, "VisualExpertV14Policy") as create:
            args = server.build_parser().parse_args(["--base-model", "base", "--memory-checkpoint", "v14",
                                                    "--visual-read-off", "--expected-include-tail"])
            server.create_policy(args)
            create.assert_called_once_with("base", "v14", device="cuda:0", write_policy="checkpoint",
                                          visual_read_off=True, expected_include_tail=True)
        for flags in (["--visual-read-off"], ["--expected-include-tail"],
                      ["--memory-checkpoint", "v14", "--no-expected-include-tail"], ["--port", "0"]):
            with self.assertRaises(ValueError):
                server.validate_options(server.build_parser().parse_args(["--base-model", "base", *flags]))

    def test_server_registers_only_joint_rpc_and_closes_every_exit(self):
        for joint in (False, True):
            for error in (None, KeyboardInterrupt(), RuntimeError("injected")):
                actor, transport = MagicMock(), MagicMock()
                transport.run.side_effect = error
                flags = ["--base-model", "base"] + (["--memory-checkpoint", "v14"] if joint else [])
                with patch.object(server, "create_policy", return_value=actor), \
                        patch("gr00t.policy.gr00t_policy.Gr00tSimPolicyWrapper", return_value="wrapped"), \
                        patch("gr00t.policy.server_client.PolicyServer", return_value=transport), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if isinstance(error, RuntimeError):
                        with self.assertRaisesRegex(RuntimeError, "injected"):
                            server.main(flags)
                    else:
                        self.assertEqual(server.main(flags), 0)
                if joint:
                    transport.register_endpoint.assert_called_once_with(server.ENDPOINT, actor.ingest_demo_tail)
                else:
                    transport.register_endpoint.assert_not_called()
                actor.reset.assert_called_once_with()
                transport.socket.close.assert_called_once_with(linger=0)
                transport.context.term.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
