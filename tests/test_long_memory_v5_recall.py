"""Auxiliary recall heads: masked loss, target isolation, and gradient checks."""
import unittest

import torch

from gr00t.long_memory.recall_v5 import RecallHeads, recall_loss


class RecallTests(unittest.TestCase):
    def test_valid_targets_update_head_and_retrieved_latent(self):
        heads = RecallHeads(8, 3)
        latent = torch.randn(1, 4, 8, requires_grad=True)
        result = recall_loss(heads, {"read": latent}, {
            "class_id": 2, "xy": [.1, .8], "class_valid": True, "xy_valid": True})
        result["loss"].backward()
        self.assertGreater(float(latent.grad.abs().sum()), 0)
        self.assertGreater(float(heads.subgoal.weight.grad.abs().sum()), 0)
        self.assertGreater(float(heads.grounding.weight.grad.abs().sum()), 0)

    def test_unknown_targets_are_masked_not_class_zero(self):
        heads = RecallHeads(8, 3)
        latent = torch.randn(1, 4, 8, requires_grad=True)
        result = recall_loss(heads, {"read": latent}, {
            "class_id": -1, "xy": [float("nan"), 0], "class_valid": False, "xy_valid": False})
        self.assertEqual(float(result["loss"]), 0)
        self.assertIsNone(result["subgoal_accuracy"])
        self.assertIsNone(result["grounding_mae"])
        result["loss"].backward()
        self.assertEqual(float(latent.grad.abs().sum()), 0)

    def test_partial_targets_and_validation(self):
        heads = RecallHeads(8, 3)
        read = {"read": torch.randn(1, 4, 8)}
        result = recall_loss(heads, read, {"class_id": 1, "class_valid": True, "xy_valid": False})
        self.assertGreater(float(result["subgoal_loss"]), 0)
        self.assertEqual(float(result["grounding_loss"]), 0)
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            recall_loss(heads, read, {"class_id": 3, "class_valid": True})
        with self.assertRaisesRegex(ValueError, "normalized"):
            recall_loss(heads, read, {"xy": [-1, 0], "xy_valid": True})

    def test_prediction_takes_no_gt_argument(self):
        heads = RecallHeads(8, 3)
        latent = torch.randn(1, 4, 8)
        before = heads(latent)
        recall_loss(heads, {"read": latent}, {"class_id": 2, "class_valid": True})
        after = heads(latent)
        torch.testing.assert_close(before["logits"], after["logits"])


if __name__ == "__main__":
    unittest.main()
