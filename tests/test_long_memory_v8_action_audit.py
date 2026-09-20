"""The generated-action diagnostic must not consume GT or alter caller RNG."""
import unittest

import torch
from torch import nn

from gr00t.long_memory.action_audit_v8 import action_errors, cached_inputs, generated_action


class AuditHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self._inference_gen = object()

    def state_encoder(self, state, embodiment):
        return state

    def get_action_with_features(self, feature, state, embodiment, masks):
        # Deliberately has no target argument. Use global RNG branch of the
        # original API to test isolated seed and caller-state preservation.
        noise = torch.randn(1, 3, 2, device=feature.device)
        return {"action_pred": noise + feature.mean() + state.mean()}


class V8ActionAuditTests(unittest.TestCase):
    def setUp(self):
        self.head = AuditHead().eval()
        self.episode = dict(features=[torch.arange(24).float().reshape(6, 4)],
            state=torch.zeros(1, 2), embodiment_id=0,
            attention_masks=[torch.ones(6, dtype=torch.bool)],
            image_masks=[torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool)])

    def test_generation_needs_no_target_or_mask(self):
        result = generated_action(self.head, self.episode, 0, seed=8)
        self.assertEqual(tuple(result.shape), (1, 3, 2))
        self.episode["targets"] = float("nan")
        self.episode["target_mask"] = "deliberately invalid and unused"
        torch.testing.assert_close(result, generated_action(self.head, self.episode, 0, seed=8), rtol=0, atol=0)

    def test_seed_rng_and_previous_generator_preserved(self):
        torch.manual_seed(918)
        before = torch.get_rng_state().clone()
        generator = self.head._inference_gen
        left = generated_action(self.head, self.episode, 0, seed=33)
        right = generated_action(self.head, self.episode, 0, seed=33)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertIs(generator, self.head._inference_gen)

    def test_only_conditioning_tail_replaced(self):
        fused = torch.ones(1, 4, 4) * 12
        feature, _, _, _ = cached_inputs(self.head, self.episode, 0, fused)
        torch.testing.assert_close(feature[0, :2], self.episode["features"][0][:2])
        torch.testing.assert_close(feature[:, -4:], fused)

    def test_errors_ignore_invalid_padding_and_slice_execution_prefix(self):
        target = torch.zeros(1, 3, 2)
        pred = torch.tensor([[[1., 2.], [3., 4.], [float("nan"), float("nan")]]])
        mask = torch.tensor([[[1, 1], [1, 1], [0, 0]]], dtype=torch.bool)
        self.assertEqual(action_errors(pred, target, mask), {"mse": 7.5, "mae": 2.5})
        self.assertEqual(action_errors(pred, target, mask, steps=1), {"mse": 2.5, "mae": 1.5})
        with self.assertRaises(ValueError):
            action_errors(pred, target, torch.zeros_like(mask))

    def test_generated_error_preserves_fp32_target_precision(self):
        prediction = torch.ones(1, 1, 1, dtype=torch.bfloat16)
        target = torch.full((1, 1, 1), 1.001, dtype=torch.float32)
        result = action_errors(prediction, target, torch.ones_like(target, dtype=torch.bool))
        self.assertAlmostEqual(result["mae"], .001, places=6)


if __name__ == "__main__":
    unittest.main()
