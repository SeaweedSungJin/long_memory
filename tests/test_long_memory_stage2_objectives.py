"""CPU unit checks for v2 objectives; these do not train the real checkpoint."""

import copy
import math
import unittest

import numpy as np
import torch

from gr00t.long_memory.core import EpisodicMemory, MemoryConfig
from gr00t.long_memory.replay import encode_until, read_bank
from gr00t.long_memory.stage2_objectives import (
    contextual_predictions, fixed_bank, parameter_groups, prediction_metrics,
    readiness, set_trainable_phase, weighted_auxiliary_loss,
)


def make_memory():
    return EpisodicMemory(MemoryConfig(feature_dim=6, state_dim=3, action_dim=4,
        hidden_dim=8, key_dim=5, value_dim=7, capacity=4, min_fill=1, residual_init=0.01))


def make_episode():
    return {"episode_id": 10, "short": torch.randn(8, 2, 6),
            "moment": torch.randn(8, 2, 6), "state": torch.randn(8, 3),
            "actions": torch.randn(7, 2, 4),
            "action_mask": torch.ones(7, 2, dtype=torch.bool),
            "transition_valid": torch.ones(7, dtype=torch.bool)}


def grads(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


class TestContextualPredictions(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(120)
        self.memory, self.episode = make_memory(), make_episode()

    def test_exact_same_context_and_no_future_leakage(self):
        result = contextual_predictions(self.memory, self.episode, 3, [0, 2])
        encoded = encode_until(self.memory, self.episode, 4)
        read = read_bank(self.memory, self.episode, 3, [0, 2], encoded)
        self.assertTrue(torch.equal(result["read"], read["read"]))
        expected = self.memory.utility(encoded["event"][3:4], self.episode["short"][3:4], read["read"])
        self.assertTrue(torch.equal(result["utility"], expected))
        self.assertTrue(torch.equal(result["logits"], self.memory.write_logits(expected, result["novelty"])))
        # Candidate event 3 ends at observation 4. Everything strictly after
        # that completion is unavailable to the write decision being trained.
        altered = copy.deepcopy(self.episode)
        for key in ("short", "moment", "state"):
            altered[key][5:] *= 1000
        altered["actions"][4:] *= 1000
        newer = contextual_predictions(self.memory, altered, 3, [0, 2])
        self.assertTrue(torch.equal(result["utility"], newer["utility"]))
        self.assertTrue(torch.equal(result["logits"], newer["logits"]))

    def test_bank_changes_context_and_empty_bank_is_valid(self):
        first = contextual_predictions(self.memory, self.episode, 3, [0])
        second = contextual_predictions(self.memory, self.episode, 3, [1, 2])
        self.assertFalse(torch.equal(first["read"], second["read"]))
        empty = contextual_predictions(self.memory, self.episode, 0, [])
        self.assertTrue(torch.equal(empty["novelty"], torch.ones(1)))
        self.assertEqual(empty["read"].abs().sum().item(), 0)

    def test_invalid_duplicate_future_full_banks_rejected(self):
        for candidate, ids in ((3, [0, 0]), (3, [3]), (3, [-1]), (3, [0.5]),
                               (5, [0, 1, 2, 3]), (8, []), (True, [])):
            with self.subTest(candidate=candidate, ids=ids), self.assertRaises(ValueError):
                contextual_predictions(self.memory, self.episode, candidate, ids)
        self.episode["transition_valid"][1] = False
        for candidate, ids in ((1, []), (3, [1])):
            with self.assertRaises(ValueError):
                contextual_predictions(self.memory, self.episode, candidate, ids)

    def test_write_loss_cannot_deform_utility_or_reader(self):
        result = contextual_predictions(self.memory, self.episode, 3, [0, 2])
        loss = weighted_auxiliary_loss(result["utility"], result["logits"],
            [{"utility_target": 0.2, "write_target": 1}], utility_weight=0)
        loss["loss"].backward()
        self.assertGreater(grads(self.memory.write_head), 0)
        self.assertEqual(grads(self.memory.utility_head), 0)
        for module in (self.memory.event_encoder, self.memory.query, self.memory.key, self.memory.value):
            self.assertEqual(grads(module), 0)

    def test_utility_gradient_only_reaches_utility_head(self):
        result = contextual_predictions(self.memory, self.episode, 3, [0, 2])
        result["utility"].square().sum().backward()
        self.assertGreater(grads(self.memory.utility_head), 0)
        for module in (self.memory.write_head, self.memory.event_encoder, self.memory.query, self.memory.value):
            self.assertEqual(grads(module), 0)


class TestPhasesAndLosses(unittest.TestCase):
    def test_optimizer_groups_stay_stable_across_phase_switch(self):
        memory, episode = make_memory(), make_episode()
        counts = set_trainable_phase(memory, joint=False)
        self.assertGreater(counts["heads"], 0)
        self.assertEqual(counts["reader"], 0)
        groups = parameter_groups(memory)
        ids = [[id(p) for p in group["params"]] for group in groups]
        self.assertFalse(set(ids[0]) & set(ids[1]))
        optimizer = torch.optim.AdamW(groups)
        original_reader = {k: v.detach().clone() for k, v in memory.named_parameters() if not k.startswith(("utility_head.", "write_head."))}
        output = contextual_predictions(memory, episode, 3, [0, 2])
        weighted_auxiliary_loss(output["utility"], output["logits"],
                                [{"utility_target": 0.1, "write_target": 1}])["loss"].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for name, expected in original_reader.items():
            self.assertTrue(torch.equal(dict(memory.named_parameters())[name], expected), name)
        self.assertGreater(set_trainable_phase(memory, joint=True)["reader"], 0)
        self.assertEqual(ids, [[id(p) for p in group["params"]] for group in parameter_groups(memory)])
        read = read_bank(memory, episode, 3, [0, 2])
        read["fused_short"].square().mean().backward()
        self.assertGreater(grads(memory.event_encoder), 0)
        self.assertTrue(all(not p.requires_grad for p in memory.reconstruction.parameters()))
        set_trainable_phase(memory, joint=False)
        self.assertTrue(all(p.grad is None for p in memory.event_encoder.parameters()))

    def test_confidence_normalization_ignores_ambiguous_not_negative(self):
        utility = torch.tensor([0.0, 0.0], requires_grad=True)
        logits = torch.tensor([2.0, -20.0], requires_grad=True)
        records = [{"utility_target": 0.0, "write_target": 1, "write_weight": 0.2},
                   {"utility_target": 0.0, "write_target": None, "write_weight": 0.0}]
        loss = weighted_auxiliary_loss(utility, logits, records)
        expected = torch.nn.functional.softplus(-logits[0])
        self.assertAlmostEqual(loss["write_loss"].item(), expected.item(), places=6)
        loss["loss"].backward()
        self.assertEqual(logits.grad[1].item(), 0)
        self.assertNotEqual(logits.grad[0].item(), 0)

    def test_all_ambiguous_batch_has_finite_differentiable_zero_write_loss(self):
        utility = torch.tensor([0.3, 0.2], requires_grad=True)
        logits = torch.tensor([100.0, -100.0], requires_grad=True)
        records = [{"utility_target": 0.4, "write_target": None, "write_weight": 0} for _ in range(2)]
        result = weighted_auxiliary_loss(utility, logits, records)
        self.assertTrue(all(torch.isfinite(value).all() for value in result.values()))
        self.assertEqual(result["write_loss"].item(), 0)
        result["loss"].backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros(2)))
        self.assertGreater(utility.grad.abs().sum().item(), 0)

    def test_no_double_log_transform_and_loss_metrics_match(self):
        target = math.log1p(10.0)
        records = [{"utility_target": target, "write_target": 1},
                   {"utility_target": 0.0, "write_target": 0, "utility_weight": 0.3}]
        utility, logits = torch.tensor([target, 0.0]), torch.tensor([1.0, -2.0])
        loss = weighted_auxiliary_loss(utility, logits, records)
        metrics = prediction_metrics(utility, logits, records)
        self.assertEqual(loss["utility_loss"].item(), 0)
        for name in ("utility_loss", "write_loss", "zero_utility_loss"):
            self.assertAlmostEqual(loss[name].item(), metrics[name], places=6)

    def test_invalid_weights_shapes_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            weighted_auxiliary_loss(torch.ones(1), torch.ones(1), [])
        for bad in (float("nan"), -1):
            with self.assertRaises(ValueError):
                weighted_auxiliary_loss(torch.ones(1), torch.ones(1),
                    [{"utility_target": 0, "write_target": 1, "write_weight": bad}])
        with self.assertRaises(ValueError):
            prediction_metrics([float("nan")], [0], [{"utility_target": 0, "write_target": 1}])


class TestMetricsAndPolicies(unittest.TestCase):
    def test_constant_score_ties_auc_half_ap_prevalence_and_collapse(self):
        records = [{"utility_target": float(i % 2), "write_target": i % 2} for i in range(20)]
        metrics = prediction_metrics(np.zeros(20), np.full(20, -0.1), records)
        self.assertEqual(metrics["write_auc"], 0.5)
        self.assertEqual(metrics["utility_auc"], 0.5)
        self.assertEqual(metrics["write_ap"], 0.5)
        self.assertEqual(metrics["write_accuracy"], 0.5)
        self.assertEqual(metrics["write_balanced_accuracy"], 0.5)
        self.assertEqual(metrics["write_all_reject"], 1)
        self.assertEqual(metrics["write_recall"], 0)
        self.assertNotIn("write_precision", metrics)
        self.assertNotIn("utility_corr", metrics)
        self.assertFalse(readiness(metrics)[0])

    def test_perfect_scores_pass_readiness_and_reversed_scores_fail(self):
        truth = np.tile([0, 1], 10)
        records = [{"utility_target": float(t), "write_target": int(t)} for t in truth]
        metrics = prediction_metrics(truth, truth * 10 - 5, records)
        self.assertEqual(metrics["utility_auc"], 1)
        self.assertEqual(metrics["write_ap"], 1)
        self.assertEqual(metrics["write_balanced_accuracy"], 1)
        self.assertEqual(readiness(metrics), (True, []))
        self.assertFalse(readiness({**metrics, "utility_auc": float("nan")})[0])
        reverse = prediction_metrics(1 - truth, 5 - truth * 10, records)
        self.assertEqual(reverse["write_auc"], 0)
        self.assertFalse(readiness(reverse)[0])

    def test_ambiguous_and_single_class_metrics_omit_undefined(self):
        records = [{"utility_target": 0.1, "write_target": 1, "write_weight": 0},
                   {"utility_target": 0.3, "write_target": 0}]
        metrics = prediction_metrics([0.1, 0.3], [1e4, -1e4], records)
        self.assertEqual(metrics["confident_count"], 1)
        self.assertEqual(metrics["confident_positive_count"], 0)
        self.assertEqual(metrics["confident_coverage"], 0.5)
        self.assertNotIn("write_auc", metrics)
        self.assertNotIn("write_ap", metrics)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        records[1]["write_weight"] = 0
        metrics = prediction_metrics([0.1, 0.3], [0, 0], records)
        self.assertEqual(metrics["confident_count"], 0)
        self.assertNotIn("write_loss", metrics)
        self.assertFalse(readiness(metrics)[0])

    def test_fixed_controls_obey_causality_validity_and_budget(self):
        episode = make_episode()
        episode["transition_valid"][1] = False
        self.assertEqual(fixed_bank(episode, 5, 2, "first"), [0, 2])
        self.assertEqual(fixed_bank(episode, 5, 2, "fifo"), [3, 4])
        first = fixed_bank(episode, 5, 2, "random", seed=100)
        self.assertEqual(first, fixed_bank(episode, 5, 2, "random", seed=100))
        self.assertEqual(len(first), 2)
        self.assertTrue(all(i < 5 and i != 1 for i in first))
        self.assertEqual(fixed_bank(episode, 0, 2), [])
        self.assertEqual(fixed_bank(episode, 5, 0), [])
        for decision, budget, policy in ((8, 2, "fifo"), (3, -1, "fifo"), (3, 2, "hard")):
            with self.assertRaises(ValueError):
                fixed_bank(episode, decision, budget, policy)


if __name__ == "__main__":
    unittest.main()
