"""CPU unit contracts for generated-action solver and target-mask diagnosis."""
from types import SimpleNamespace
import unittest

import torch

from run_scripts.robomme.audit_archive_generation_v7 import prefix_masks, query_plan, solver_steps


class ArchiveGenerationAuditTests(unittest.TestCase):
    def test_observed_prefix_excludes_terminal_target_without_mutating_inputs(self):
        target = torch.ones(1, 5, 2, dtype=torch.bool)
        transitions = torch.tensor([True, True, False])
        nominal, observed = prefix_masks(target, transitions, 3)
        self.assertEqual(int(nominal.sum()), 6)
        self.assertEqual(int(observed.sum()), 4)
        self.assertTrue(bool(target.all()))
        self.assertFalse(bool(observed[:, 2:].any()))
        target[0, 0, 1] = False
        _, intersected = prefix_masks(target, transitions, 3)
        self.assertEqual(int(intersected.sum()), 3)

    def test_empty_observed_prefix_is_not_fabricated(self):
        nominal, observed = prefix_masks(torch.ones(1, 4, 2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), 2)
        self.assertTrue(bool(nominal.any()))
        self.assertFalse(bool(observed.any()))

    def test_invalid_masks_and_counts_rejected(self):
        mask = torch.ones(1, 4, 2, dtype=torch.bool)
        for target, observed, count in ((mask.float(), torch.ones(2, dtype=torch.bool), 2),
                                        (mask, torch.ones(2), 2),
                                        (mask, torch.ones(3, dtype=torch.bool), 2),
                                        (mask, torch.ones(2, dtype=torch.bool), 0)):
            with self.assertRaises(ValueError):
                prefix_masks(target, observed, count)

    def test_solver_restores_original_setting_even_on_failure(self):
        head = SimpleNamespace(num_inference_timesteps=4)
        with solver_steps(head, 16):
            self.assertEqual(head.num_inference_timesteps, 16)
        self.assertEqual(head.num_inference_timesteps, 4)
        with self.assertRaises(RuntimeError):
            with solver_steps(head, 8):
                raise RuntimeError("generation failed")
        self.assertEqual(head.num_inference_timesteps, 4)
        for bad in (0, 65, True, 1.5):
            with self.assertRaises(ValueError):
                with solver_steps(head, bad):
                    self.fail("invalid solver entered")

    def test_plan_is_distinct_deterministic_val_only_and_has_exact_count(self):
        cache = SimpleNamespace(manifest={"splits": {"val": [2, 3, 4], "train": [9]}})
        calls = []
        def fetch(eid):
            calls.append(eid)
            return {"decision_mask": torch.tensor([False, True, True])}
        episodes = SimpleNamespace(fetch=fetch)
        first = query_plan(cache, episodes, 3, 42)
        self.assertEqual(first, query_plan(cache, episodes, 3, 42))
        self.assertEqual({eid for eid, _ in first}, {2, 3, 4})
        self.assertTrue(all(q in (1, 2) for _, q in first))
        self.assertNotIn(9, calls)
        with self.assertRaises(ValueError):
            query_plan(cache, episodes, 4, 42)


if __name__ == "__main__":
    unittest.main()
