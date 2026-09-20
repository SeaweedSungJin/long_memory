"""Real CPU visual/parent memories; tiny mocked AE, no simulator or CUDA."""
import contextlib
import copy
from dataclasses import asdict
import io
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import numpy as np

from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
from run_scripts.robomme import policy_visual_differential_v12 as policy_module
from run_scripts.robomme import serve_visual_differential_v12 as server
from run_scripts.robomme.visual_differential_memory_v12 import (
    CAMERA_ORDER, VisualDifferentialConfig, VisualDifferentialMemoryV12,
)
from tests import test_policy_visual_patch_v11 as legacy


CONFIG = VisualDifferentialConfig(feature_dim=8, hidden_dim=16, time_scale=2.)


def checkpoint_info(step=32, mode="differential"):
    info = legacy.checkpoint_info(step)
    info["config"].update(trainer_variant="visual_differential_v12", mode="visual_differential",
        visual=asdict(CONFIG), read_mode=mode, train={"read_mode": mode})
    info["metadata"]["read_mode"] = mode
    return info


class VisualDifferentialPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    call = legacy.VisualPatchPolicyTests.call
    assert_actions_equal = legacy.VisualPatchPolicyTests.assert_actions_equal

    def policy(self, *, awake=True, off=False, step=32, parent=None, mode="differential"):
        info, events = checkpoint_info(step, mode), []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(911)
            visual = VisualDifferentialMemoryV12(CONFIG, read_mode=mode)
            if awake:
                with torch.no_grad():
                    visual.output_projection.weight.normal_(std=.05)
        state = copy.deepcopy(visual.state_dict())

        def preflight(base, checkpoint, expected_stage):
            self.assertEqual((base, checkpoint, expected_stage), ("base", "visual-checkpoint", 1))
            events.append("validated")
            return copy.deepcopy(info)

        def initialize(policy, base, checkpoint, *, device, strict, write_policy, memory_off):
            self.assertEqual(events, ["validated"])
            self.assertEqual((base, checkpoint, device, strict, write_policy, memory_off),
                             ("base", "parent-checkpoint", "cpu", True, "checkpoint", False))
            policy.__dict__.update((parent or legacy.parent_policy()).__dict__)
            events.append("parent_loaded")

        def load(checkpoint, module):
            self.assertEqual(checkpoint, "visual-checkpoint")
            self.assertEqual(events, ["validated", "parent_loaded"])
            self.assertEqual(module.read_mode, mode)
            module.load_state_dict(state)
            events.append("visual_loaded")
            return copy.deepcopy(info)

        with patch.object(policy_module, "checkpoint_info", side_effect=preflight), \
                patch.object(LongMemoryV7Policy, "__init__", initialize), \
                patch.object(policy_module, "load_checkpoint", side_effect=load):
            policy = policy_module.VisualDifferentialV12Policy("base", "visual-checkpoint", device="cpu",
                visual_read_off=off, expected_read_mode=mode)
        self.assertEqual(events, ["validated", "parent_loaded", "visual_loaded"])
        return policy

    # Run the existing serial rollout contracts AGAINST THE NEW policy class.
    # Fixture methods have no legacy-module patches or legacy type assumptions.
    test_zero_off_parent_parity = legacy.VisualPatchPolicyTests.test_zero_and_visual_off_match_parent_actions_archive_short_and_session_rng
    test_images_only_original_write = legacy.VisualPatchPolicyTests.test_warmed_read_changes_images_only_preserves_archive_tail_and_stores_original
    test_passive_and_write_after_decode = legacy.VisualPatchPolicyTests.test_passive_appends_without_visual_read_action_or_noise_and_append_follows_decode
    test_session_reset_isolation = legacy.VisualPatchPolicyTests.test_interleaved_sessions_and_resets_isolate_banks_rng_and_short_cache
    test_failure_cleanup = legacy.VisualPatchPolicyTests.test_forward_decode_append_failures_clear_session_and_restore_exact_methods
    test_reentrant_methods = legacy.VisualPatchPolicyTests.test_success_restores_descriptors_and_reentrant_action_reset_are_rejected
    test_nonshort_invariant = legacy.VisualPatchPolicyTests.test_changed_nonshort_conditioning_is_rejected_and_no_visual_append_occurs
    test_mask_invariant = legacy.VisualPatchPolicyTests.test_changed_backbone_masks_are_rejected_before_expert_or_visual_write
    test_cadence_and_lru = legacy.VisualPatchPolicyTests.test_original_cadence_errors_and_lru_reset_leave_no_cross_session_visual_state

    def test_strict_mode_binding_and_legacy_rejection_before_parent_allocation(self):
        changes = [lambda i: i["config"].update(read_mode="current_only"),
            lambda i: i["metadata"].update(read_mode="current_only"),
            lambda i: i["config"]["train"].update(read_mode="current_only"),
            lambda i: i["config"].update(trainer_variant="visual_patch_v11"),
            lambda i: i["config"].update(mode="visual_patch"),
            lambda i: i["config"].update(stage=2)]
        for change in changes:
            info = checkpoint_info()
            change(info)
            with patch.object(policy_module, "checkpoint_info", return_value=info), \
                    patch.object(LongMemoryV7Policy, "__init__") as base, self.assertRaises(ValueError):
                policy_module.VisualDifferentialV12Policy("base", "bundle", device="cpu", expected_read_mode="differential")
            base.assert_not_called()
        for options in ({"expected_read_mode": "differential"}, {"visual_read_off": True}):
            with patch.object(policy_module, "checkpoint_info", return_value=checkpoint_info(mode="current_only")), \
                    patch.object(LongMemoryV7Policy, "__init__") as base, self.assertRaises(ValueError):
                policy_module.VisualDifferentialV12Policy("base", "bundle", device="cpu", **options)
            base.assert_not_called()

    def test_frozen_initialization_rng_and_actual_load_identity(self):
        rng = torch.get_rng_state().clone()
        for mode in ("differential", "current_only"):
            policy = self.policy(mode=mode)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for module in (policy.memory, policy.cvom, policy.model.action_head, policy.visual_memory):
                self.assertFalse(module.training)
                self.assertFalse(any(p.requires_grad for p in module.parameters()))
        def initialize(policy, *args, **kwargs):
            policy.__dict__.update(legacy.parent_policy().__dict__)
        with patch.object(policy_module, "checkpoint_info", return_value=checkpoint_info()), \
                patch.object(LongMemoryV7Policy, "__init__", initialize), \
                patch.object(policy_module, "load_checkpoint", return_value=checkpoint_info(64)), \
                self.assertRaisesRegex(ValueError, "identity changed"):
            policy_module.VisualDifferentialV12Policy("base", "bundle", device="cpu")

    def test_real_v12_bundle_loads_both_modes_through_new_serializer(self):
        from tests.test_checkpoint_visual_differential_v12 import VisualCheckpointTests
        fixture = VisualCheckpointTests("test_roundtrip_external_parent_immutable_and_readonly_rng")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        def initialize(policy, base, parent, **kwargs):
            self.assertEqual((base, parent), (fixture.base, str(fixture.parent)))
            # Only heavyweight original-base construction is replaced. Both
            # checkpoint validation calls and actual visual tensor loading run.
            policy.__dict__.update(legacy.parent_policy().__dict__)
        for mode in ("differential", "current_only"):
            fixture.config["read_mode"] = fixture.config["train"]["read_mode"] = fixture.metadata["read_mode"] = mode
            fixture.visual.read_mode = mode
            bundle = fixture.save(name=mode, step=0)
            with patch.object(LongMemoryV7Policy, "__init__", initialize):
                policy = policy_module.VisualDifferentialV12Policy(fixture.base, bundle, device="cpu", expected_read_mode=mode)
            self.assertEqual(policy.checkpoint_step, 0)
            for name, value in fixture.visual.state_dict().items():
                self.assertTrue(torch.equal(value, policy.visual_memory.state_dict()[name]))
            self.call(policy, passive=True)
            _, info = self.call(policy, marker=2, frame=2)
            self.assertEqual(info["visual_memory"]["image_delta_norm"], 0.)
            self.assertEqual(info["visual_read_mode"], mode)

    def test_both_modes_empty_bypass_dual_original_banks_and_truthful_hook_metrics(self):
        parent = legacy.parent_policy()
        modes = [self.policy(mode=mode) for mode in ("differential", "current_only")]
        baseline, _ = self.call(parent)
        for policy in modes:
            actions, info = self.call(policy)
            self.assert_actions_equal(actions, baseline)
            self.assertFalse(info["visual_memory"]["read_enabled"])
            self.assertEqual(info["visual_memory"]["readout_norm"], 0.)
            self.assertEqual(info["visual_memory"]["projected_residual_norm"], 0.)
            bank = policy.sessions["A"].visual_bank
            self.assertEqual(bank.content.shape, bank.tokens.shape)
            content, tokens = bank.content.clone(), bank.tokens.clone()
            _, info = self.call(policy, marker=2, frame=2, controls=np.zeros((2, 8), np.float32))
            v = info["visual_memory"]
            self.assertEqual(info["visual_read_mode"], policy.visual_read_mode)
            self.assertEqual(v["read_mode"], policy.visual_read_mode)
            self.assertEqual(v["past_read_enabled"], policy.visual_read_mode == "differential")
            self.assertTrue(v["current_reference_enabled"])
            self.assertGreater(v["readout_norm"], 0.)
            self.assertGreater(v["projected_residual_norm"], 0.)
            self.assertEqual(v["readout_norm_semantics"],
                "H_past_minus_H_current" if policy.visual_read_mode == "differential" else "H_current")
            self.assertTrue(torch.equal(policy.sessions["A"].visual_bank.content[:, :1], content))
            self.assertTrue(torch.equal(policy.sessions["A"].visual_bank.tokens[:, :1], tokens))
            self.assertFalse(policy.visual_memory.output_projection._forward_hooks)
            policy.visual_memory.read_mode = "corrupt"
            with self.assertRaisesRegex(RuntimeError, "mode changed"):
                self.call(policy, marker=3, frame=4)
            self.assertFalse(policy._visual_call_lock.locked())

    def test_current_only_output_ignores_past_content_but_keeps_appending(self):
        correct, wrong = self.policy(mode="current_only"), self.policy(mode="current_only")
        for policy in (correct, wrong):
            self.call(policy, passive=True)
        with torch.inference_mode():
            wrong.sessions["A"].visual_bank.content.add_(99)
            wrong.sessions["A"].visual_bank.tokens.mul_(-13)
        a, info = self.call(correct, marker=2, frame=2)
        b, _ = self.call(wrong, marker=2, frame=2)
        self.assert_actions_equal(a, b)
        self.assertFalse(info["visual_memory"]["past_read_enabled"])
        self.assertEqual(correct.sessions["A"].visual_bank.content.shape[1], 2)

    def test_server_help_original_parent_and_mode_dispatch(self):
        help_result = subprocess.run([sys.executable, "-B", "-S", server.__file__, "--help"],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--expected-read-mode", help_result.stdout)
        parser = server.build_parser()
        with patch.object(server, "LongMemoryV7Policy") as parent, \
                patch.object(server, "VisualDifferentialV12Policy") as visual, \
                patch.object(server, "parent_reference") as validate:
            server.create_policy(parser.parse_args(["--base-model", "base", "--device", "cpu"]))
            parent.assert_called_once_with("base", device="cpu", write_policy="checkpoint", memory_off=False)
            visual.assert_not_called(); validate.assert_not_called()
            server.create_policy(parser.parse_args(["--base-model", "base", "--parent-checkpoint", "parent"]))
            validate.assert_called_once_with("base", "parent")
            for mode in ("differential", "current_only"):
                server.create_policy(parser.parse_args(["--base-model", "base", "--memory-checkpoint", "visual",
                    "--expected-read-mode", mode]))
                visual.assert_called_with("base", "visual", device="cuda:0", write_policy="checkpoint",
                    visual_read_off=False, expected_read_mode=mode)
        for flags in (["--visual-read-off"], ["--expected-read-mode", "differential"],
            ["--memory-checkpoint", "visual", "--expected-read-mode", "current_only", "--visual-read-off"],
            ["--port", "0"], ["--host", ""]):
            with patch.object(server, "create_policy") as create, self.assertRaises(ValueError):
                server.main(["--base-model", "base", *flags])
            create.assert_not_called()

    def test_server_cleanup(self):
        for error in (None, KeyboardInterrupt(), RuntimeError("transport failed")):
            policy, transport = MagicMock(), MagicMock()
            transport.run.side_effect = error
            with patch.object(server, "create_policy", return_value=policy), \
                    patch("gr00t.policy.gr00t_policy.Gr00tSimPolicyWrapper", return_value="wrapped"), \
                    patch("gr00t.policy.server_client.PolicyServer", return_value=transport), \
                    contextlib.redirect_stdout(io.StringIO()):
                if isinstance(error, RuntimeError):
                    with self.assertRaisesRegex(RuntimeError, "transport failed"):
                        server.main(["--base-model", "base"])
                else:
                    self.assertEqual(server.main(["--base-model", "base"]), 0)
            policy.reset.assert_called_once_with()
            transport.socket.close.assert_called_once_with(linger=0)
            transport.context.term.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
