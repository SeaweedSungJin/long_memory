"""CPU causal writer tests; no pretrained weights, simulator or real training."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from gr00t.long_memory.core_v3 import ActionValueMemory, MemoryV3Config
from gr00t.long_memory.objectives_v5 import ContinuationLabelsV5, LabelV5Config, storage_loss
from gr00t.long_memory.replay_v3 import storage_options


def model(capacity=3, min_fill=0, frozen=True):
    return ActionValueMemory(MemoryV3Config(
        feature_dim=8, state_dim=3, action_dim=2, hidden_dim=8, num_heads=2,
        capacity=capacity, min_fill=min_fill, max_victims=3,
        residual_init=0.01)).eval().requires_grad_(not frozen)


def episode(steps=14):
    return {"episode_id": 17, "cache_fingerprint": "test-cache",
            "frames": torch.arange(steps + 1) * 16,
            "short": torch.randn(steps + 1, 4, 8), "moment": torch.randn(steps + 1, 4, 8),
            "state": torch.randn(steps + 1, 3), "actions": torch.randn(steps, 3, 2),
            "action_mask": torch.ones(steps, 3, dtype=torch.bool),
            "transition_valid": torch.ones(steps, dtype=torch.bool),
            "decision_mask": torch.ones(steps, dtype=torch.bool),
            "targets": torch.randn(steps, 3, 2)}


def read_ids(memory, ep, decision, bank, encoded=None):
    return {"fused_short": torch.tensor(bank, dtype=torch.float64),
            "read": torch.tensor(bank, dtype=torch.float64)}


def weighted(weights, calls=None):
    def loss(head, ep, decision, bank, *, seed):
        ids = [int(i) for i in bank.tolist()]
        if calls is not None:
            calls.append((decision, seed, tuple(ids)))
        return 100.0 + seed % 17 - sum(weights.get(i, 0.0) for i in ids)
    return loss


def keep(memory, ep, candidate, bank, encoded=None):
    options = storage_options(bank, candidate, memory.config.capacity, memory.config.max_victims)
    return {"options": options, "logits": torch.tensor([1.] + [0.] * (len(options) - 1))}


def identity(**extra):
    return {"cache_fingerprint": "test-cache", "teacher_version": 0, **extra}


def make_label(*, teacher=None, ep=None, cfg=None, candidate=3, bank=None,
               loss=None, callback=None, recall=None, continuation=keep):
    teacher, ep = teacher or model(), ep or episode()
    cfg = cfg or LabelV5Config(scale=1., margin=0.)
    with tempfile.TemporaryDirectory() as path:
        labels = ContinuationLabelsV5(teacher, None, cfg, path,
                    identity(**({"recall_fingerprint": "recall-semantics-v1"} if callback else {})),
                    recall_cost=callback, recall_module=recall)
        with patch("gr00t.long_memory.objectives_v5.read_bank", side_effect=read_ids), \
             patch("gr00t.long_memory.diagnostic_interventions.storage_prediction", side_effect=continuation):
            return labels.get_storage(ep, candidate, [0, 1, 2] if bank is None else bank,
                                      19, loss_fn=loss or weighted({0: 10, 1: 1, 2: 2, 3: 3}))


class ContinuationObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(37)

    def test_invalid_configuration_and_budgets(self):
        for kwargs in ({"audit_noise_samples": 1}, {"audit_noise_samples": True},
                       {"continuation_policy": "invalid"}, {"recall_weight": float("nan")},
                       {"confidence_screen": 1}, {"max_teacher_forwards": 0},
                       {"max_continuation_events": False}, {"noise_samples": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LabelV5Config(**kwargs)
        for kwargs in ({"max_teacher_forwards": 2}, {"max_continuation_events": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "exceeds"):
                make_label(cfg=LabelV5Config(**kwargs))

    def test_intervening_fifo_eviction_erases_candidate_benefit(self):
        calls = []
        label = make_label(cfg=LabelV5Config(continuation_policy="all", scale=1., margin=0.),
                           loss=weighted({3: 5}, calls))
        self.assertEqual(label["costs"], [0.] * 4)
        self.assertEqual(label["effective_future_contexts"], 0)
        self.assertEqual(label["distinct_branch_sequences"], 1)
        self.assertEqual(label["collapsed_option_query_fraction"], 1.)
        self.assertEqual(label["unique_expert_forwards"], 4)
        self.assertEqual(len(calls), 4)
        self.assertEqual(label["nominal_expert_forwards"], 16)
        self.assertFalse(any(any(row) for row in label["candidate_retained_at_queries"]))
        for d, banks in zip(label["future_decisions"], label["future_bank_ids"]):
            self.assertTrue(all(bank == list(range(d - 3, d)) for bank in banks))

    def test_branch_dependent_continuation_uses_each_branch_bank(self):
        # Presence of candidate 3 changes a later write. Copying one global
        # suffix bank into every option would incorrectly erase this effect.
        def path_dependent(memory, ep, candidate, bank, encoded=None):
            result = keep(memory, ep, candidate, bank, encoded)
            if candidate == 4 and 3 in bank:
                result["logits"][1] = 2.
            return result
        label = make_label(continuation=path_dependent, loss=weighted({4: 7}))
        self.assertEqual(label["costs"], [0., -7., -7., -7.])
        for banks in label["future_bank_ids"]:
            self.assertNotIn(4, banks[0])
            self.assertTrue(all(4 in bank for bank in banks[1:]))

    def test_paired_train_draws_and_independent_audit(self):
        calls = []
        label = make_label(cfg=LabelV5Config(scale=1., margin=0., audit_noise_samples=4),
                           loss=weighted({3: 3}, calls))
        a = {s for row in label["noise_seeds"] for s in row}
        b = {s for row in label["audit"]["noise_seeds"] for s in row}
        self.assertFalse(a & b)
        self.assertEqual(len(a), 4)
        self.assertEqual(len(b), 8)
        self.assertFalse(label["audit"]["used_for_training"])
        self.assertEqual(label["audit"]["A_best_gain_on_B"], 3.)
        for d, seeds in zip(label["future_decisions"], label["noise_seeds"]):
            for seed in seeds:
                rows = [call for call in calls if call[:2] == (d, seed)]
                self.assertEqual(len(rows), 4)
        self.assertEqual(len(calls), label["unique_expert_forwards"])

    def test_independent_audit_cannot_change_training_cost(self):
        cfg = LabelV5Config(scale=1., margin=0., audit_noise_samples=2)
        prototype = make_label(cfg=cfg)
        a = {s for row in prototype["noise_seeds"] for s in row}
        def loss(head, ep, decision, bank, *, seed):
            return 100 - (4 if seed in a else -4) * int(3 in bank.tolist())
        label = make_label(cfg=cfg, loss=loss)
        self.assertEqual(label["costs"], [0., -4., -4., -4.])
        self.assertEqual(label["audit"]["A_best_gain_on_B"], -4.)
        self.assertEqual(label["audit"]["non_tie_sign_agreement"], 0.)

    def test_known_useful_write_has_correct_nonzero_writer_gradient(self):
        data, student = episode(), model(frozen=False)
        label = make_label(ep=data, bank=[0, 1], candidate=2, loss=weighted({2: 3}))
        logits = torch.nn.Parameter(torch.zeros(2))
        with patch("gr00t.long_memory.objectives_v5.storage_prediction",
                   return_value={"options": label["options"], "logits": logits}):
            result = storage_loss(student, data, label)
        result["loss"].backward()
        self.assertGreater(float(logits.grad[0]), 0)
        self.assertLess(float(logits.grad[1]), 0)
        self.assertEqual(result["metrics"]["storage_signal_fraction"], 1.)

    def test_tied_costs_give_connected_zero_gradient(self):
        data, student = episode(), model(frozen=False)
        label = make_label(ep=data, loss=weighted({}))
        result = storage_loss(student, data, label)
        result["loss"].backward()
        self.assertEqual(float(result["loss"]), 0.)
        self.assertTrue(any(p.grad is not None for p in student.writer_parameters()))
        self.assertTrue(all(p.grad is None or not bool(p.grad.abs().sum()) for p in student.writer_parameters()))
        self.assertTrue(all(p.grad is None for p in student.reader_parameters()))
        self.assertEqual(result["metrics"]["storage_tie_aware_agreement"], 1.)

    def test_noisy_positive_mean_is_screened_to_zero(self):
        cfg = LabelV5Config(scale=1., margin=0., future_samples=1)
        prototype = make_label(cfg=cfg, bank=[0, 1])
        first = prototype["noise_seeds"][0][0]
        def noisy(head, ep, decision, bank, *, seed):
            return 100 - (1.1 if seed == first else -0.9) * int(3 in bank.tolist())
        label = make_label(cfg=cfg, bank=[0, 1], loss=noisy)
        self.assertAlmostEqual(label["raw_costs"][1], -.1)
        self.assertEqual(label["costs"], [0., 0.])
        self.assertEqual(label["confident_options"], 0)

    def test_forced_fill_never_becomes_classifier_supervision(self):
        data, teacher = episode(), model(min_fill=3)
        label = make_label(teacher=teacher, ep=data, bank=[0], candidate=2,
                           loss=lambda *a, **k: self.fail("forced fill must skip expert"))
        self.assertEqual(label["status"], "skipped")
        self.assertEqual(label["skip_reason"], "forced_min_fill")
        self.assertEqual(label["options"], [])
        student = model(min_fill=3, frozen=False)
        with patch("gr00t.long_memory.objectives_v5.storage_prediction", side_effect=AssertionError("no classifier")):
            result = storage_loss(student, data, label)
        result["loss"].backward()
        self.assertEqual(float(result["loss"]), 0.)
        self.assertEqual(result["metrics"]["storage_options"], 0)

    def test_no_future_query_skips_and_invalid_candidate_fails(self):
        label = make_label(candidate=12, bank=[0, 1, 2])
        self.assertEqual(label["skip_reason"], "no_old_future_query")
        for candidate, bank in ((True, [0]), (3, [3]), (3, [1, 0]), (3, [0, 0]), (3, [0.0])):
            with self.subTest(candidate=candidate, bank=bank), self.assertRaises(ValueError):
                make_label(candidate=candidate, bank=bank)

    def test_old_future_boundary_is_strict_and_terminal_not_query(self):
        data = episode(10)
        data["frames"] = torch.tensor([0, 6, 22, 25, 41, 57, 73, 89, 105, 121, 124])
        data["decision_mask"][7] = False
        label = make_label(ep=data, candidate=2, bank=[0, 1], cfg=LabelV5Config(future_samples=20))
        self.assertEqual(label["future_decisions"], [8, 9])

    def test_recall_cost_is_shared_read_supervision_and_can_drive_writer(self):
        calls = []
        def recall(read, ep, d):
            calls.append((d, tuple(read["read"].tolist())))
            return 2. - float(3 in read["read"].tolist())
        label = make_label(cfg=LabelV5Config(scale=1., margin=0., recall_weight=.5),
                           callback=recall, loss=weighted({}))
        self.assertEqual(label["costs"], [0., -.5, -.5, -.5])
        self.assertEqual(len(calls), 8)  # not once per repeated flow-noise draw
        self.assertTrue(all(len(set(row)) == 1 for row in label["action_loss_draws"][0]))

    def test_student_options_nonfinite_loss_and_tampered_cost_fail(self):
        data = episode()
        label = make_label(ep=data)
        with self.assertRaisesRegex(ValueError, "options differ"):
            storage_loss(model(capacity=4, frozen=False), data, label)
        label["costs"][1] += 1.
        with self.assertRaisesRegex(ValueError, "screened costs"):
            storage_loss(model(frozen=False), data, label)
        with self.assertRaises(FloatingPointError):
            make_label(loss=lambda *a, **k: float("nan"))
        with self.assertRaises(FloatingPointError):
            make_label(cfg=LabelV5Config(recall_weight=1.), callback=lambda *a: float("inf"))

    def test_cache_refuses_changed_writer_or_recall_semantics(self):
        data, teacher = episode(), model()
        recall = torch.nn.Linear(8, 1).eval().requires_grad_(False)
        cfg = LabelV5Config(recall_weight=1.)
        with tempfile.TemporaryDirectory() as path:
            ident = identity(recall_fingerprint="targets-v1")
            labels = ContinuationLabelsV5(teacher, None, cfg, path, ident,
                        recall_cost=lambda *a: 0., recall_module=recall)
            changed = copy.deepcopy(teacher)
            with torch.no_grad():
                next(changed.writer.parameters()).add_(.1)
            with self.assertRaisesRegex(ValueError, "different teacher/writer/recall"):
                ContinuationLabelsV5(changed, None, cfg, path, ident,
                    recall_cost=lambda *a: 0., recall_module=recall)
            with self.assertRaisesRegex(ValueError, "different teacher/writer/recall"):
                ContinuationLabelsV5(teacher, None, cfg, path, identity(recall_fingerprint="targets-v2"),
                    recall_cost=lambda *a: 0., recall_module=recall)
            with torch.no_grad():
                recall.weight.add_(1.)
            with self.assertRaisesRegex(ValueError, "Frozen recall module changed"):
                labels.get_storage(data, 3, [0, 1, 2], 1)

    def test_mutation_inside_immutable_teacher_requires_refresh(self):
        with tempfile.TemporaryDirectory() as path:
            labels = ContinuationLabelsV5(model(), None, LabelV5Config(), path, identity())
            with torch.no_grad():
                next(labels.teacher.writer.parameters()).add_(.1)
            with self.assertRaisesRegex(ValueError, "Frozen teacher/writer changed"):
                labels.get_storage(episode(), 3, [0, 1, 2], 1)

    def test_cache_hit_has_no_extra_forwards_and_preserves_live_student(self):
        data, student = episode(), model(frozen=False)
        before = copy.deepcopy(student.state_dict())
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as path:
            labels = ContinuationLabelsV5(student, None, LabelV5Config(), path, identity())
            with patch("gr00t.long_memory.objectives_v5.read_bank", side_effect=read_ids), \
                 patch("gr00t.long_memory.diagnostic_interventions.storage_prediction", side_effect=keep):
                first = labels.get_storage(data, 3, [0, 1, 2], 1, weighted({3: 1}))
            second = labels.get_storage(data, 3, [0, 1, 2], 1,
                       lambda *a, **k: self.fail("cache hit must not recompute"))
            self.assertEqual(first, second)
            self.assertEqual(len(list(Path(path).glob("storage-*.json"))), 1)
            json.dumps(first, allow_nan=False)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, student.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(all(p.requires_grad for p in student.parameters()))

    def test_future_ground_truth_never_enters_storage_inputs(self):
        data = episode()
        first = make_label(ep=data)
        altered = copy.deepcopy(data)
        altered["targets"] += 10000.
        # Future targets affect only an explicit loss callback, not branching.
        second = make_label(ep=altered)
        self.assertEqual(first["future_bank_ids"], second["future_bank_ids"])
        self.assertEqual(first["costs"], second["costs"])
        self.assertTrue(all(event < d for d, banks in zip(first["future_decisions"], first["future_bank_ids"])
                            for bank in banks for event in bank))


if __name__ == "__main__":
    unittest.main()
