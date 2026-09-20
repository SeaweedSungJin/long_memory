"""Small CPU audits: independent noise, forced writes, and causal replay."""

import copy
import json
import unittest
from unittest.mock import patch

import torch

from gr00t.long_memory.core_v3 import ActionValueMemory, MemoryV3Config
from gr00t.long_memory.diagnostic_writer import audit_storage_context


def model(capacity=3, min_fill=0, max_victims=3):
    return ActionValueMemory(MemoryV3Config(
        feature_dim=8, state_dim=3, action_dim=2, hidden_dim=8, num_heads=2,
        capacity=capacity, min_fill=min_fill, max_victims=max_victims,
        residual_init=0.01)).eval().requires_grad_(False)


def episode(steps=14):
    return {
        "episode_id": 23, "frames": torch.arange(steps + 1) * 16,
        "short": torch.randn(steps + 1, 4, 8), "moment": torch.randn(steps + 1, 4, 8),
        "state": torch.randn(steps + 1, 3), "actions": torch.randn(steps, 3, 2),
        "action_mask": torch.ones(steps, 3, dtype=torch.bool),
        "transition_valid": torch.ones(steps, dtype=torch.bool),
        "decision_mask": torch.ones(steps, dtype=torch.bool),
    }


def bank_read(memory, data, decision, bank_ids, encoded=None):
    return {"fused_short": torch.tensor(bank_ids, dtype=torch.float64)}


def weighted_loss(weights, calls=None):
    def loss(head, data, decision, bank, *, seed):
        ids = bank.tolist()
        if calls is not None:
            calls.append((decision, seed, tuple(ids)))
        return 100 + seed % 17 - sum(weights.get(int(i), 0) for i in ids)
    return loss


def audit(memory=None, data=None, **kwargs):
    defaults = dict(candidate=3, bank_ids=[0, 1, 2], future_samples=2,
                    noise_samples=2, seed=18, continuation_policy="all",
                    loss_fn=weighted_loss({0: 10, 1: 1, 2: 2, 3: 3}))
    defaults.update(kwargs)
    with patch("gr00t.long_memory.diagnostic_writer.read_bank", side_effect=bank_read):
        return audit_storage_context(memory or model(), None, data or episode(), **defaults)


class WriterDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def test_exact_paired_noise_and_independent_selection_split(self):
        calls = []
        result = audit(loss_fn=weighted_loss({0: 10, 1: 1, 2: 2, 3: 3}, calls))
        self.assertEqual(result["status"], "ok")
        split_a = {s for row in result["noise_seeds"]["A"] for s in row}
        split_b = {s for row in result["noise_seeds"]["B"] for s in row}
        self.assertFalse(split_a & split_b)
        self.assertEqual(len(split_a), 4)
        self.assertEqual(len(split_b), 4)
        for variant in ("fixed_bank", "continuation"):
            for split in ("A", "B"):
                self.assertEqual(len(result[variant]["loss_draws"][split]), 2)
        self.assertEqual(result["fixed_bank"]["keep_relative_gains"]["A"], [0, -7, 2, 1])
        self.assertEqual(result["fixed_bank"]["keep_relative_gains"]["B"], [0, -7, 2, 1])
        self.assertEqual(len(calls), result["unique_expert_forwards"])
        self.assertLess(result["unique_expert_forwards"], result["nominal_expert_forwards_without_deduplication"])
        # Every fixed option is evaluated with every declared paired seed.
        for decision, seeds in zip(result["future_decisions"], result["noise_seeds"]["A"]):
            for seed in seeds:
                for option in result["options"]:
                    self.assertIn((decision, seed, tuple(option)), calls)
        json.dumps(result, allow_nan=False)

    def test_continuation_replays_intervening_events_and_exposes_proxy_mismatch(self):
        result = audit()
        self.assertEqual(result["fixed_bank"]["best_option_A"], 2)
        self.assertEqual(result["continuation"]["keep_relative_gains"]["B"], [0, 0, 0, 0])
        for decision, banks in zip(result["future_decisions"], result["continuation"]["future_bank_ids"]):
            for bank in banks:
                self.assertEqual(bank, list(range(decision - 3, decision)))
                self.assertTrue(all(i < decision for i in bank))
        self.assertFalse(any(any(row) for row in result["continuation"]["candidate_retained_at_queries"]))
        self.assertEqual(result["fixed_vs_continuation"]["fixed_A_best_gain_evaluated_under_continuation_B"], 0)

    def test_all_victims_maps_deployed_indices_without_changing_choice_set(self):
        memory = model(max_victims=2)
        options = [[0, 1, 2], [1, 2, 3], [0, 1, 3]]
        prediction = {"options": options, "logits": torch.tensor([0., 0., 1.])}
        with patch("gr00t.long_memory.diagnostic_writer.storage_prediction", return_value=prediction):
            result = audit(memory, all_victims=True)
        self.assertEqual(result["deployed_to_audited_option_indices"], [0, 1, 3])
        self.assertEqual(result["writer_deployed_default_option"], 2)
        self.assertEqual(result["writer_selected_option"], 3)
        self.assertEqual(result["fixed_bank"]["best_option_A"], 2)
        self.assertEqual(result["fixed_bank"]["deployed_options_best_A"], 3)
        self.assertEqual(result["fixed_bank"]["expanded_vs_deployed_A_best_gain_on_B"], 1)
        self.assertEqual(result["fixed_bank"]["writer_minus_A_best_loss_on_B"], 1)

    def test_forced_min_fill_uses_append_not_raw_writer_argmax(self):
        memory = model(min_fill=2)
        prediction = {"options": [[0], [0, 2]], "logits": torch.tensor([100., -100.])}
        with patch("gr00t.long_memory.diagnostic_writer.storage_prediction", return_value=prediction):
            result = audit(memory, candidate=2, bank_ids=[0])
        self.assertTrue(result["writer_forced_min_fill"])
        self.assertEqual(result["writer_raw_argmax_default_option"], 0)
        self.assertEqual(result["writer_selected_option"], 1)
        self.assertEqual(result["bank_coverage"], "partial")

    def test_true_old_eligibility_uses_event_end_and_irregular_short_endpoints(self):
        data = episode(10)
        data["frames"] = torch.tensor([0, 6, 22, 25, 41, 57, 73, 89, 105, 121, 124])
        data["decision_mask"][7] = False
        result = audit(data=data, candidate=2, bank_ids=[0, 1], future_samples=30)
        self.assertEqual(result["future_decisions"], [8, 9])
        self.assertTrue(all(result["candidate_end_frame"] < f for f in result["oldest_short_endpoint_frames"]))
        skipped = audit(data=data, candidate=6, bank_ids=[0, 1, 2])
        self.assertEqual(skipped["status"], "skipped")
        self.assertEqual(skipped["reason"], "no_strict_old_future_action_query")

    def test_ties_are_not_misreported_as_writer_failure(self):
        result = audit(loss_fn=weighted_loss({}))
        fixed = result["fixed_bank"]
        self.assertTrue(fixed["writer_tie_aware_agreement_with_A"])
        self.assertTrue(fixed["writer_tie_aware_agreement_with_B"])
        self.assertEqual(fixed["writer_empirical_B_regret_beyond_tolerance"], 0)
        self.assertIsNone(fixed["non_tie_gain_sign_agreement_fraction"])

    def test_reports_noise_instability_and_separate_selection_and_measurement(self):
        initial = audit()
        split_a = {s for row in initial["noise_seeds"]["A"] for s in row}

        def unstable(head, data, decision, bank, *, seed):
            gain = 1 if seed in split_a else -1
            return 100 - gain * int(3 in bank.tolist())

        result = audit(loss_fn=unstable)
        fixed = result["fixed_bank"]
        self.assertEqual(fixed["gain_sign_agreement_fraction"], 0)
        self.assertEqual(fixed["non_tie_gain_sign_agreement_fraction"], 0)
        self.assertNotEqual(fixed["best_option_A"], fixed["best_option_B_empirical"])
        self.assertEqual(fixed["A_best_keep_relative_gain_on_B"], -1)

    def test_read_only_frozen_contract_and_global_rng_unchanged(self):
        memory, data = model(), episode()
        before = copy.deepcopy(memory.state_dict())
        rng = torch.get_rng_state().clone()
        audit(memory, data)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value, memory.state_dict()[key]) for key, value in before.items()))
        self.assertTrue(all(parameter.grad is None for parameter in memory.parameters()))
        with self.assertRaisesRegex(ValueError, "frozen"):
            audit(memory.train(), data)
        with self.assertRaisesRegex(ValueError, "frozen"):
            audit(memory.eval().requires_grad_(True), data)
        with self.assertRaisesRegex(ValueError, "action expert"):
            audit_storage_context(model(), torch.nn.Linear(1, 1), data, 3, [0, 1, 2], loss_fn=weighted_loss({}))

    def test_invalid_arguments_and_nonfinite_values_fail(self):
        for kwargs in ({"noise_samples": 1}, {"noise_samples": True}, {"future_samples": 0},
                       {"tie_tolerance": float("nan")}, {"seed": True}, {"continuation_policy": "learn"},
                       {"bank_ids": [3]}, {"bank_ids": [0, 0]}, {"bank_ids": [0.0]}, {"all_victims": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                audit(**kwargs)
        with self.assertRaises(FloatingPointError):
            audit(loss_fn=lambda *args, **kwargs: float("nan"))


if __name__ == "__main__":
    unittest.main()
