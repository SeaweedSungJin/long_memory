"""CPU checks of V19's objective, not RoboMME performance measurements."""
import unittest
from unittest.mock import patch

import torch

from gr00t.long_memory.expert_v4 import (LoRAConfig, expert_episode_flow_loss,
    expert_parameters, install_expert_lora)
from gr00t.long_memory.hamlet import sample_noise_time
from run_scripts.robomme.flow_objective_v19 import episode_flow_v19
from tests.test_long_memory_v4_expert import FakeHead, _Projection


def fixture():
    torch.manual_seed(1901)
    head = FakeHead()
    head.action_encoder = _Projection(10, 8)
    head.action_decoder = _Projection(8, 10)
    head.eval().requires_grad_(False)
    install_expert_lora(head, LoRAConfig(rank=2, alpha=4))
    mask = torch.zeros(1, 5, 10, dtype=torch.bool)
    mask[:, :, :8] = True
    ep = {"features": [torch.randn(6, 8)], "state": torch.randn(1, 4),
        "targets": torch.randn(1, 5, 10), "target_mask": mask,
        "attention_masks": [torch.ones(6, dtype=torch.bool)],
        "image_masks": [torch.zeros(6, dtype=torch.bool)], "embodiment_id": 0,
        "action_mask": torch.ones(1, 2, dtype=torch.bool)}
    return head, ep, torch.randn(1, 2, 8, requires_grad=True)


class FlowObjectiveV19Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_control_exact_original_loss_predictions_gradients_and_rng(self):
        head, ep, fused = fixture()
        before = torch.get_rng_state().clone()
        original = expert_episode_flow_loss(head, ep, 0, fused, seed=29)
        parameters = [fused, *expert_parameters(head)]
        grads = torch.autograd.grad(original["loss"], parameters)
        with patch("run_scripts.robomme.flow_objective_v19.expert_episode_flow_loss",
                   wraps=expert_episode_flow_loss) as bridge:
            control = episode_flow_v19(head, ep, 0, fused, seed=29, action_steps=2)
        self.assertEqual(bridge.call_count, 1)
        self.assertIs(control["loss"], control["original_flow_loss"])
        for key in ("loss", "prediction", "velocity_mae"):
            torch.testing.assert_close(control[key], original[key], rtol=0, atol=0)
        control_grads = torch.autograd.grad(control["loss"], parameters)
        for left, right in zip(grads, control_grads):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_treatment_changes_only_reduction_not_model_inputs_or_predictions(self):
        head, ep, fused = fixture()
        targets_before, mask_before = ep["targets"].clone(), ep["target_mask"].clone()
        trajectories = []
        hook = head.action_encoder.register_forward_pre_hook(
            lambda module, args: trajectories.append(args[0].detach().clone()))
        before = torch.get_rng_state().clone()
        try:
            control = episode_flow_v19(head, ep, 0, fused, seed=5, action_steps=2)
            candidate = episode_flow_v19(head, ep, 0, fused, seed=5, tail_weight=.25, action_steps=2)
        finally:
            hook.remove()
        torch.testing.assert_close(trajectories[0], trajectories[1], rtol=0, atol=0)
        torch.testing.assert_close(control["prediction"], candidate["prediction"], rtol=0, atol=0)
        noise, _ = sample_noise_time(head, ep["targets"], 5)
        target = torch.where(ep["target_mask"], ep["targets"], 0.)
        errors = (candidate["prediction"] - (target - noise)).square()
        expected = (errors[:, :2, :8].sum() + .25 * errors[:, 2:, :8].sum()) / (16 + .25*24 + 1e-6)
        torch.testing.assert_close(candidate["loss"], expected)
        self.assertNotEqual(float(candidate["loss"]), float(control["loss"]))
        candidate["loss"].backward()
        self.assertGreater(float(fused.grad.abs().sum()), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in expert_parameters(head)))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(ep["targets"], targets_before))
        self.assertTrue(torch.equal(ep["target_mask"], mask_before))

    def test_padding_nan_is_zero_and_joint_gripper_counts_are_correct(self):
        head, ep, fused = fixture()
        original = episode_flow_v19(head, ep, 0, fused, seed=17, tail_weight=.25, action_steps=2)
        ep["targets"][..., 8:] = float("nan")
        padded = episode_flow_v19(head, ep, 0, fused, seed=17, tail_weight=.25, action_steps=2)
        torch.testing.assert_close(original["loss"], padded["loss"], rtol=0, atol=0)
        self.assertEqual(int(padded["target_valid_values"]), 40)
        self.assertEqual(int(padded["executed_prefix_valid_values"]), 16)
        self.assertEqual(int(padded["tail_valid_values"]), 24)
        self.assertEqual(int(padded["joint_valid_values"]), 35)
        self.assertEqual(int(padded["gripper_valid_values"]), 5)
        self.assertEqual(int(padded["executed_prefix_joint_valid_values"]), 14)
        self.assertEqual(int(padded["executed_prefix_gripper_valid_values"]), 2)

    def test_final_partial_prefix_and_empty_tail_supported(self):
        head, ep, fused = fixture()
        ep["target_mask"][:, 1:] = False
        ep["action_mask"][:, 1:] = False
        result = episode_flow_v19(head, ep, 0, fused, seed=3, tail_weight=.25, action_steps=2)
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertEqual(int(result["executed_prefix_valid_values"]), 8)
        self.assertEqual(int(result["tail_valid_values"]), 0)
        self.assertEqual(float(result["tail_flow_loss"]), 0.)
        torch.testing.assert_close(result["loss"], result["original_flow_loss"], rtol=0, atol=0)

    def test_observed_mask_changes_diagnostics_not_weighted_training(self):
        head, ep, fused = fixture()
        full = episode_flow_v19(head, ep, 0, fused, seed=3, tail_weight=.25, action_steps=2)
        ep["action_mask"][:, 1] = False
        partial = episode_flow_v19(head, ep, 0, fused, seed=3, tail_weight=.25, action_steps=2)
        torch.testing.assert_close(full["loss"], partial["loss"], rtol=0, atol=0)
        self.assertEqual(int(partial["executed_prefix_valid_values"]), 8)
        self.assertEqual(int(partial["nominal_prefix_valid_values"]), 16)

    def test_bf16_velocity_arithmetic_and_checkpointed_gradients(self):
        head, ep, fused = fixture()
        # Frozen base is BF16; adapter masters remain FP32, as in production.
        for name, p in head.named_parameters():
            if "lora_" not in name:
                p.data = p.data.to(torch.bfloat16)
        reference = expert_episode_flow_loss(head, ep, 0, fused, seed=7)
        result = episode_flow_v19(head, ep, 0, fused, seed=7, tail_weight=.25,
                                 action_steps=2, activation_checkpointing=True)
        target = ep["targets"].bfloat16()
        noise, _ = sample_noise_time(head, target, 7)
        velocity = torch.where(ep["target_mask"], target, 0.) - noise
        err = torch.where(ep["target_mask"], reference["prediction"].float()-velocity.float(), 0.)
        weights = ep["target_mask"].float()
        weights[:, 2:] *= .25
        expected = (err.square()*weights).sum()/(weights.sum()+1e-6)
        torch.testing.assert_close(result["loss"], expected, rtol=0, atol=0)
        result["loss"].backward()
        self.assertGreater(float(fused.grad.abs().sum()), 0)

    def test_invalid_options_and_non_robomme_actions_rejected(self):
        head, ep, fused = fixture()
        for kwargs in ({"seed": None}, {"seed": True}, {"seed": -1},
                       {"tail_weight": 0}, {"tail_weight": 1.01},
                       {"tail_weight": float("nan")}, {"tail_weight": True},
                       {"action_steps": 0}, {"action_steps": 6}, {"action_steps": True}):
            options = dict(seed=3, action_steps=2)
            options.update(kwargs)
            with self.subTest(options=options), self.assertRaises(ValueError):
                episode_flow_v19(head, ep, 0, fused, **options)
        ep["target_mask"][..., 8] = True
        with self.assertRaisesRegex(ValueError, "padding"):
            episode_flow_v19(head, ep, 0, fused, seed=3, action_steps=2)


if __name__ == "__main__":
    unittest.main()
