"""Action-policy comparison checks with CPU memory and a mocked frozen head."""

import copy
import math
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gr00t.long_memory.core import EpisodicMemory, MemoryConfig
from gr00t.long_memory.replay import read_bank
from gr00t.long_memory.stage2_validation import action_guard, validate_actions


def fixture(accept=False):
    memory = EpisodicMemory(MemoryConfig(feature_dim=6, state_dim=3, action_dim=4,
        hidden_dim=8, key_dim=5, value_dim=7, capacity=4, min_fill=2, residual_init=0.01))
    with torch.no_grad():
        for parameter in memory.write_head.parameters():
            parameter.zero_()
        memory.write_head[-1].bias.fill_(20 if accept else -20)
    episode = {"episode_id": 10, "short": torch.randn(8, 2, 6),
               "moment": torch.randn(8, 2, 6), "state": torch.randn(8, 3),
               "actions": torch.randn(7, 2, 4),
               "action_mask": torch.ones(7, 2, dtype=torch.bool),
               "transition_valid": torch.ones(7, dtype=torch.bool),
               "decision_mask": torch.ones(7, dtype=torch.bool)}
    episode["transition_valid"][1] = False
    return memory, episode


def fake_loss(head, episode, decision, fused_short=None, *, seed):
    tokens = episode["short"][decision:decision + 1] if fused_short is None else fused_short
    return {"loss": tokens.square().mean() + seed * 1e-8,
            "velocity_mae": tokens.abs().mean()}


class TestActionValidation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(810)
        self.memory, self.episode = fixture()

    def test_rejecting_writer_has_zero_learned_rate_but_nonzero_forced_rate(self):
        with patch("gr00t.long_memory.stage2_validation.episode_flow_loss", side_effect=fake_loss):
            metrics = validate_actions(self.memory, None, lambda _: self.episode,
                                       [(10, 6, None), (10, 3, None)], seed=18)
        # Decision 6 sees valid events 0,2,3,4,5: two forced, three learned.
        # Decision 3 sees only 0,2: both forced, zero learned attempts.
        self.assertEqual(metrics["learned_write_attempts"], 3)
        self.assertEqual(metrics["learned_write_accepts"], 0)
        self.assertEqual(metrics["learned_write_rate"], 0)
        self.assertEqual(metrics["bank_fill"], 2)
        self.assertAlmostEqual(metrics["write_rate"], (2 / 5 + 1) / 2)
        self.assertEqual(metrics["action_loss"], metrics["first_action_loss"])
        self.assertEqual(metrics["action_validation_count"], 2)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_accepting_writer_counts_evicted_writes_not_only_retained_slots(self):
        memory, episode = fixture(accept=True)
        with patch("gr00t.long_memory.stage2_validation.episode_flow_loss", side_effect=fake_loss):
            metrics = validate_actions(memory, None, lambda _: episode, [(10, 6, None)], seed=18)
        self.assertEqual(metrics["bank_fill"], 4)
        self.assertEqual(metrics["write_rate"], 1)
        self.assertEqual(metrics["learned_write_attempts"], 3)
        self.assertEqual(metrics["learned_write_accepts"], 3)
        self.assertEqual(metrics["learned_write_rate"], 1)
        self.assertEqual(metrics["action_loss"], metrics["all_action_loss"])
        self.assertEqual(metrics["action_loss"], metrics["fifo_action_loss"])

    def test_all_controls_use_paired_seed_and_matched_occupancy(self):
        banks, calls = [], []

        def capture_read(memory, episode, decision, ids, encoded=None):
            banks.append((decision, tuple(ids)))
            return read_bank(memory, episode, decision, ids, encoded)

        def capture_loss(*args, **kwargs):
            calls.append((args[2], kwargs["seed"]))
            return fake_loss(*args, **kwargs)

        with patch("gr00t.long_memory.stage2_validation.read_bank", side_effect=capture_read), patch(
            "gr00t.long_memory.stage2_validation.episode_flow_loss", side_effect=capture_loss
        ):
            validate_actions(self.memory, None, lambda _: self.episode, [(10, 6, None), (10, 5, None)], seed=29)
        self.assertTrue(all(s == (29 if d == 6 else 1038) for d, s in calls))
        self.assertIn((6, (0, 2)), banks)  # Hard = first forced two.
        self.assertIn((6, (4, 5)), banks)  # Matched FIFO.
        self.assertIn((6, (2, 3, 4, 5)), banks)  # Capacity reference.
        self.assertTrue(all(len(ids) in (2, 4) for d, ids in banks if d == 6))
        self.assertTrue(all(all(i < d and i != 1 for i in ids) for d, ids in banks))

    def test_determinism_rng_modes_and_parameter_preservation(self):
        self.memory.train()
        self.memory.event_encoder.eval()  # Mixed mode must survive validation.
        head = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Linear(1, 1))
        head.train()
        head[0].eval()
        modes = {id(m): m.training for model in (self.memory, head) for m in model.modules()}
        parameters = copy.deepcopy(self.memory.state_dict())
        torch_state, python_state, numpy_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
        with patch("gr00t.long_memory.stage2_validation.episode_flow_loss", side_effect=fake_loss):
            a = validate_actions(self.memory, head, lambda _: self.episode, [(10, 6, None)], seed=1)
            b = validate_actions(self.memory, head, lambda _: self.episode, [(10, 6, None)], seed=1)
        self.assertEqual(a, b)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        np.testing.assert_equal(numpy_state, np.random.get_state())
        self.assertEqual(modes, {id(m): m.training for model in (self.memory, head) for m in model.modules()})
        self.assertTrue(all(torch.equal(value, self.memory.state_dict()[name]) for name, value in parameters.items()))
        self.assertTrue(all(parameter.grad is None for parameter in self.memory.parameters()))

    def test_exception_restores_modes_and_all_rngs(self):
        self.memory.train()
        self.memory.event_encoder.eval()
        modes = {id(m): m.training for m in self.memory.modules()}
        torch_state, python_state, numpy_state = torch.get_rng_state(), random.getstate(), np.random.get_state()

        def failing_head(*args, **kwargs):
            torch.rand(3)
            random.random()
            np.random.random()
            raise RuntimeError("intentional test failure")

        with patch("gr00t.long_memory.stage2_validation.episode_flow_loss", side_effect=failing_head):
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                validate_actions(self.memory, None, lambda _: self.episode, [(10, 6, None)], seed=2)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        np.testing.assert_equal(numpy_state, np.random.get_state())
        self.assertEqual(modes, {id(m): m.training for m in self.memory.modules()})

    def test_empty_history_and_invalid_plan_handling(self):
        with patch("gr00t.long_memory.stage2_validation.episode_flow_loss", side_effect=fake_loss):
            result = validate_actions(self.memory, None, lambda _: self.episode, [(10, 0, None)], seed=2)
            for plan in ([], [(10, 8, None)], [(10, 7, None)], [(10,)]):
                with self.assertRaises(ValueError):
                    validate_actions(self.memory, None, lambda _: self.episode, plan, seed=2)
        self.assertEqual(result["learned_write_attempts"], 0)
        self.assertEqual(result["learned_write_rate"], 0)
        self.assertEqual(result["bank_fill"], 0)
        for key in ("first_action_loss", "fifo_action_loss", "random_action_loss", "all_action_loss", "baseline_action_loss"):
            self.assertEqual(result["action_loss"], result[key])

    def test_nonfinite_loss_rejected_and_guard_is_conservative(self):
        with patch("gr00t.long_memory.stage2_validation.episode_flow_loss", return_value={"loss": torch.tensor(float("nan"))}):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_actions(self.memory, None, lambda _: self.episode, [(10, 3, None)], seed=2)
        self.assertEqual(action_guard({"action_loss": 1.04, "all_action_loss": 1.0}), (True, []))
        self.assertFalse(action_guard({"action_loss": 1.06, "all_action_loss": 1.0})[0])
        for metrics in ({}, {"action_loss": float("nan"), "all_action_loss": 1},
                        {"action_loss": 1, "all_action_loss": -1}):
            self.assertFalse(action_guard(metrics)[0])
        with self.assertRaises(ValueError):
            action_guard({}, tolerance=-1)


if __name__ == "__main__":
    unittest.main()
