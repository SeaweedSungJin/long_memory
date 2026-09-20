"""Analytic CPU checks for common-mode energy and diagonal-free directions."""
import unittest

import torch

from run_scripts.robomme.audit_visual_common_mode_v11 import audit_query, direction_cosines, energy_decomposition
from run_scripts.robomme.visual_patch_memory_v11 import VisualPatchConfig, VisualPatchMemoryV11
from tests.test_replay_visual_patch_v11 import ObservationOnlyMapping, GuardedColumn, observations


class CommonModeTests(unittest.TestCase):
    def test_constant_centered_and_mixed_exact_decomposition(self):
        constant = torch.tensor([[2., 0.], [2., 0.]])
        self.assertEqual(energy_decomposition(constant)["common_energy_fraction"], 1.)
        centered = torch.tensor([[0., 3.], [0., -3.]])
        self.assertEqual(energy_decomposition(centered)["common_energy_fraction"], 0.)
        result = energy_decomposition(constant + centered)
        self.assertEqual(result["total_energy"], 26.)
        self.assertEqual(result["common_energy"], 8.)
        self.assertEqual(result["variation_energy"], 18.)
        self.assertAlmostEqual(result["common_energy_fraction"], 8/26)
        self.assertEqual(result["decomposition_abs_error"], 0.)
        self.assertIsNone(energy_decomposition(torch.zeros(2, 3))["common_energy_fraction"])

    def test_camera_means_are_different_from_global_patch_mean(self):
        value = torch.tensor([[[2., 0.], [2., 0.]], [[-2., 0.], [-2., 0.]]])
        self.assertEqual(energy_decomposition(value)["common_energy_fraction"], 1.)
        self.assertEqual(energy_decomposition(value.reshape(4, 2))["common_energy_fraction"], 0.)

    def test_pairwise_cosine_excludes_diagonal(self):
        matrix, result = direction_cosines(torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]))
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(result["n"], 6)
        self.assertEqual(result["max"], 0.)
        self.assertEqual(result["min"], -1.)
        self.assertAlmostEqual(result["mean"], -1/3)
        with self.assertRaises(ValueError): direction_cosines(torch.zeros(2, 3))

    def test_hooks_capture_original_replay_without_supervision_or_future(self):
        torch.set_num_threads(2)
        with torch.random.fork_rng():
            torch.manual_seed(2)
            memory = VisualPatchMemoryV11(VisualPatchConfig(feature_dim=8, hidden_dim=16)).eval().requires_grad_(False)
            with torch.no_grad(): memory.output_projection.weight.normal_(std=.04)
        source = ObservationOnlyMapping({k: GuardedColumn(v, 3) for k, v in observations().items()})
        source.update(targets=object(), actions=object(), state=object())
        hooks_before = [len(module._forward_hooks) for module in memory.modules()]
        parameters = {k: v.clone() for k, v in memory.state_dict().items()}
        row, mean = audit_query(memory, source, 3)
        self.assertEqual(mean.shape, (8,))
        self.assertEqual(row["prior_patch_tokens"], 486)
        self.assertEqual(row["prior_raw_frames"], [0, 16, 32])
        self.assertTrue(row["only_current_images_changed"])
        self.assertGreater(row["common_patch_decomposition"]["common_energy_fraction"], 0.)
        self.assertEqual([len(module._forward_hooks) for module in memory.modules()], hooks_before)
        for key, value in memory.state_dict().items(): self.assertTrue(torch.equal(value, parameters[key]))
        self.assertTrue(all(p.grad is None for p in memory.parameters()))


if __name__ == "__main__":
    unittest.main()
