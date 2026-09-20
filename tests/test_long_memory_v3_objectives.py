"""CPU action-value objective tests; no pretrained weights or simulator needed."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from gr00t.long_memory.core_v3 import ActionValueMemory, MemoryV3Config
from gr00t.long_memory.objectives_v3 import (
    ActionValueLabelsV3, LabelV3Config, build_retrieval_label, build_storage_label,
    retrieval_loss, storage_loss,
)
from gr00t.long_memory.replay_v3 import storage_prediction


def model(capacity=3, frozen=True):
    value = ActionValueMemory(MemoryV3Config(
        feature_dim=8, state_dim=3, action_dim=2, hidden_dim=8, num_heads=2,
        capacity=capacity, min_fill=0, max_victims=3, residual_init=0.01))
    return value.eval().requires_grad_(not frozen)


def episode(steps=14):
    return {
        "episode_id": 17, "cache_fingerprint": "fake-data-and-base-expert",
        "frames": torch.arange(steps + 1) * 16,
        "short": torch.randn(steps + 1, 4, 8), "moment": torch.randn(steps + 1, 4, 8),
        "state": torch.randn(steps + 1, 3), "actions": torch.randn(steps, 3, 2),
        "action_mask": torch.ones(steps, 3, dtype=torch.bool),
        "transition_valid": torch.ones(steps, dtype=torch.bool),
        "decision_mask": torch.ones(steps, dtype=torch.bool),
        "targets": torch.randn(steps, 3, 2),
    }


def bank_read(teacher, data, decision, bank_ids, encoded=None):
    # An injectable action objective can inspect exact bank IDs in tiny tests.
    return {"fused_short": torch.tensor(bank_ids, dtype=torch.float64)}


def weighted_loss(weights, calls=None):
    def loss(head, data, decision, fused_short, *, seed):
        bank = fused_short.tolist()
        if calls is not None:
            calls.append((decision, seed, list(bank)))
        # The shared stochastic term cancels only if the branches are paired.
        return {"loss": torch.tensor(100 + seed % 7 - sum(weights.get(int(i), 0) for i in bank), dtype=torch.float64)}
    return loss


def make_storage(teacher=None, data=None, cfg=None, weights=None, candidate=3, bank=None, seed=11):
    teacher, data = teacher or model(), data or episode()
    with patch("gr00t.long_memory.objectives_v3.read_bank", side_effect=bank_read):
        return build_storage_label(teacher, None, data, candidate, [0, 1, 2] if bank is None else bank,
                                   cfg or LabelV3Config(scale=1), seed,
                                   loss_fn=weighted_loss(weights or {0: 10, 1: 1, 2: 2, 3: 3}))


class TestConfigurationAndCausality(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(314)

    def test_invalid_config(self):
        for kwargs in ({"noise_samples": 1}, {"memory_window": 0}, {"future_samples": 1.5},
                       {"scale": 0}, {"scale": float("nan")}, {"margin": -1},
                       {"uncertainty_z": -1}, {"max_retrieval_events": True},
                       {"ranking_temperature": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LabelV3Config(**kwargs)

    def test_teacher_and_expert_must_be_frozen_eval(self):
        teacher, data = model(), episode()
        for mutate in (lambda m: m.train(), lambda m: m.requires_grad_(True)):
            bad = mutate(copy.deepcopy(teacher))
            with self.assertRaisesRegex(ValueError, "frozen"):
                build_storage_label(bad, None, data, 3, [0, 1, 2], LabelV3Config(), 1)
        head = torch.nn.Linear(1, 1)
        with self.assertRaisesRegex(ValueError, "action expert"):
            build_storage_label(teacher, head, data, 3, [0, 1, 2], LabelV3Config(), 1)

    def test_bank_rejects_future_duplicates_invalid_and_overcapacity(self):
        teacher, data = model(), episode()
        for bank in ([0, 0], [1, 0], [3], [-1], [0.0], [True]):
            with self.subTest(bank=bank), self.assertRaises(ValueError):
                build_storage_label(teacher, None, data, 3, bank, LabelV3Config(), 1)
        data["transition_valid"][1] = False
        with self.assertRaisesRegex(ValueError, "invalid"):
            build_storage_label(teacher, None, data, 3, [1], LabelV3Config(), 1)
        with self.assertRaisesRegex(ValueError, "capacity"):
            build_retrieval_label(teacher, None, data, 7, [0, 2, 3, 4], LabelV3Config(), 1)

    def test_future_queries_strictly_exclude_candidate_end_from_short_window(self):
        data = episode(steps=10)
        data["frames"] = torch.tensor([0, 6, 22, 25, 41, 57, 73, 89, 105, 121, 124])
        data["decision_mask"][7] = False
        label = make_storage(data=data, candidate=2, bank=[0, 1], cfg=LabelV3Config(future_samples=20))
        # i=2 ends at endpoint 3. At d=6 that endpoint still occurs in K=4.
        self.assertEqual(label["future_decisions"], [8, 9])
        self.assertTrue(all(d < len(data["decision_mask"]) for d in label["future_decisions"]))
        with self.assertRaisesRegex(ValueError, "old-only"):
            make_storage(data=data, candidate=6, bank=[0, 1, 2])

    def test_labels_do_not_modify_student_or_rng(self):
        student, data = model(frozen=False), episode()
        before = {k: v.clone() for k, v in student.state_dict().items()}
        state = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            labels = ActionValueLabelsV3(student, None, LabelV3Config(), directory,
                {"cache_fingerprint": data["cache_fingerprint"], "teacher_version": 0})
            with patch("gr00t.long_memory.objectives_v3.read_bank", side_effect=bank_read):
                labels.get_storage(data, 3, [0, 1, 2], seed=4, loss_fn=weighted_loss({3: 1}))
        self.assertTrue(all(p.requires_grad for p in student.parameters()))
        self.assertTrue(all(torch.equal(value, student.state_dict()[key]) for key, value in before.items()))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))


class TestStorageLabels(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_full_bank_opportunity_cost_and_exact_paired_noise(self):
        teacher, data, calls = model(), episode(), []
        with patch("gr00t.long_memory.objectives_v3.read_bank", side_effect=bank_read):
            label = build_storage_label(teacher, None, data, 3, [0, 1, 2], LabelV3Config(scale=1), 10,
                loss_fn=weighted_loss({0: 10, 1: 1, 2: 2, 3: 3}, calls))
        self.assertEqual(label["options"], [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]])
        self.assertEqual(label["costs"], [0, 7, -2, -1])
        self.assertEqual(label["best_option"], 2)
        self.assertEqual(label["confident_options"], 3)
        self.assertEqual(len(calls), 2 * 2 * 4)
        for offset in range(0, len(calls), 4):
            self.assertEqual(len({tuple(call[:2]) for call in calls[offset:offset + 4]}), 1)
            self.assertTrue(all(len(call[2]) == 3 for call in calls[offset:offset + 4]))
            self.assertTrue(all(call[0] >= 8 for call in calls[offset:offset + 4]))

    def test_append_respects_budget_and_only_adds_candidate(self):
        label = make_storage(candidate=2, bank=[0, 1], weights={0: 1, 1: 2, 2: 5})
        self.assertEqual(label["options"], [[0, 1], [0, 1, 2]])
        self.assertEqual(label["costs"], [0, -5])
        self.assertEqual(label["option_gain_stats"][1]["mean"], 5)

    def test_mc_ambiguity_is_logged_without_thresholding_expected_loss(self):
        def noisy(head, data, decision, bank, *, seed):
            sign = 1 if seed % 2 else -1
            return {"loss": 100 - (0.1 + sign) * float(3 in bank.tolist())}
        with patch("gr00t.long_memory.objectives_v3.read_bank", side_effect=bank_read):
            label = build_storage_label(model(), None, episode(), 3, [0, 1], LabelV3Config(scale=1), 1, noisy)
        self.assertEqual(label["confident_options"], 0)
        self.assertAlmostEqual(label["costs"][1], -0.1)
        self.assertGreater(label["option_gain_stats"][1]["se"], 0)

    def test_expected_objective_gradient_has_correct_sign(self):
        teacher, student, data = model(), model(frozen=False), episode()
        label = make_storage(teacher=teacher, data=data)
        logits = torch.nn.Parameter(torch.zeros(4))
        with patch("gr00t.long_memory.objectives_v3.storage_prediction", return_value={"logits": logits, "options": label["options"]}):
            result = storage_loss(student, data, label)
        self.assertAlmostEqual(float(result["loss"]), 1.0)
        result["loss"].backward()
        # Gradient descent suppresses harmful replacement 1 and promotes 2,3.
        self.assertGreater(float(logits.grad[1]), 0)
        self.assertLess(float(logits.grad[2]), 0)
        self.assertLess(float(logits.grad[3]), 0)

    def test_storage_gradient_only_updates_writer_and_uses_completion_endpoint(self):
        teacher, student, data = model(), model(frozen=False), episode()
        label = make_storage(teacher=teacher, data=data)
        storage_loss(student, data, label)["loss"].backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in student.writer_parameters()))
        self.assertTrue(all(p.grad is None for p in student.reader_parameters()))
        original = storage_prediction(student, data, 3, [0, 1, 2])["logits"].detach()
        future = copy.deepcopy(data)
        future["short"][5:] += 100
        future["moment"][5:] += 100
        future["state"][5:] += 100
        future["actions"][4:] += 100
        future["targets"] += 10000
        actual = storage_prediction(student, future, 3, [0, 1, 2])["logits"].detach()
        self.assertTrue(torch.equal(original, actual), "future/GT targets must not enter writer inputs")
        current = copy.deepcopy(data)
        current["short"][4] += 10
        self.assertFalse(torch.equal(original, storage_prediction(student, current, 3, [0, 1, 2])["logits"].detach()))

    def test_nonfinite_loss_and_different_student_options_fail(self):
        teacher, student, data = model(), model(frozen=False), episode()
        with patch("gr00t.long_memory.objectives_v3.read_bank", side_effect=bank_read), self.assertRaises(FloatingPointError):
            build_storage_label(teacher, None, data, 3, [0, 1, 2], LabelV3Config(), 1,
                                lambda *args, **kwargs: {"loss": float("nan")})
        label = make_storage(teacher=teacher, data=data)
        with self.assertRaisesRegex(ValueError, "options differ"):
            storage_loss(model(capacity=4, frozen=False), data, label)


class TestRetrievalLabels(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(74)

    def build(self, weights, *, data=None, decision=8, bank=None, cfg=None, loss=None):
        data = data or episode()
        with patch("gr00t.long_memory.objectives_v3.read_bank", side_effect=bank_read):
            label = build_retrieval_label(model(), None, data, decision, [0, 1, 2] if bank is None else bank,
                                          cfg or LabelV3Config(), 1, loss or weighted_loss(weights))
        return data, label

    def test_query_specific_leave_one_out_and_null_rankings(self):
        calls = []
        _, label = self.build({0: 3, 1: 1, 2: -2}, loss=weighted_loss({0: 3, 1: 1, 2: -2}, calls))
        pairs = {(p["better_index"], p["worse_index"]) for p in label["pairs"]}
        self.assertEqual(pairs, {(0, 1), (0, 2), (0, -1), (1, 2), (1, -1), (-1, 2)})
        self.assertEqual(label["paired_event_gains"], [[3, 3], [1, 1], [-2, -2]])
        self.assertEqual(len(calls), 2 * 4)
        for offset in range(0, len(calls), 4):
            self.assertEqual(len({tuple(call[:2]) for call in calls[offset:offset + 4]}), 1)
            self.assertEqual(calls[offset][2], [0, 1, 2])
            self.assertTrue(all(len(call[2]) == 2 for call in calls[offset + 1:offset + 4]))

    def test_redundant_tied_or_uncertain_events_do_not_force_positive(self):
        for weights in ({}, {0: 0.000001, 1: 0.000001, 2: 0.000001}):
            _, label = self.build(weights)
            self.assertEqual(label["pairs"], [])
        def noise(head, data, decision, bank, *, seed):
            return 100 - (1 if seed % 2 else -1) * float(0 in bank.tolist())
        _, label = self.build({}, loss=noise)
        self.assertEqual(label["pairs"], [])
        self.assertGreater(label["event_gain_stats"][0]["se"], 0)

    def test_no_old_events_skips_teacher_forwards_and_returns_backwardable_zero(self):
        fake = weighted_loss({0: 3})
        with patch("gr00t.long_memory.objectives_v3.episode_flow_loss", side_effect=AssertionError("must skip")):
            data, label = self.build({}, decision=3, loss=fake)
        self.assertEqual(label["sampled_event_ids"], [])
        self.assertEqual(label["loss_draws"], [])
        student = model(frozen=False)
        result = retrieval_loss(student, data, label)
        self.assertEqual(result["metrics"]["retrieval_rank_accuracy"], None)
        self.assertEqual(float(result["loss"]), 0)
        result["loss"].backward()
        self.assertTrue(all(p.grad is None or not bool(p.grad.any()) for p in student.parameters()))
        _, empty = self.build({}, data=data, decision=8, bank=[])
        retrieval_loss(student, data, empty)["loss"].backward()

    def test_strict_boundary_and_sampling_do_not_use_future_targets(self):
        data = episode()
        _, label = self.build({0: 2}, data=data, decision=5)
        self.assertEqual(label["sampled_event_ids"], [0])
        # Event 1 ends at endpoint 2, still inside [2,3,4,5].
        data["targets"] *= 1000
        _, another = self.build({0: 2}, data=data, decision=5)
        self.assertEqual(label["sampled_event_ids"], another["sampled_event_ids"])
        self.assertEqual(label["noise_seeds"], another["noise_seeds"])

    def test_ranking_gradient_updates_event_encoder_query_but_not_writer(self):
        data, label = self.build({0: 3, 1: 1, 2: -2})
        student = model(frozen=False)
        result = retrieval_loss(student, data, label)
        result["loss"].backward()
        self.assertTrue(bool(student.query.weight.grad.abs().sum()))
        self.assertTrue(bool(student.key.weight.grad.abs().sum()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in student.situation_pool.parameters()))
        self.assertTrue(all(p.grad is None for p in student.writer.parameters()))

    def test_ranking_sign_and_future_input_invariance(self):
        data, label = self.build({0: 3, 1: 1, 2: -2})
        scores = torch.nn.Parameter(torch.zeros(1, 3))
        student = model(frozen=False)
        with patch("gr00t.long_memory.objectives_v3.read_bank", return_value={"event_scores": scores, "null_score": torch.zeros(1, 1)}):
            loss = retrieval_loss(student, data, label)["loss"]
        loss.backward()
        self.assertLess(float(scores.grad[0, 0]), 0)
        self.assertGreater(float(scores.grad[0, 2]), 0)
        baseline = retrieval_loss(student, data, label)["loss"].detach()
        changed = copy.deepcopy(data)
        changed["short"][9:] += 100
        changed["state"][9:] += 100
        changed["moment"][9:] += 100
        changed["actions"][8:] += 100
        changed["targets"] += 100
        self.assertTrue(torch.equal(baseline, retrieval_loss(student, changed, label)["loss"].detach()))


class TestLabelCache(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(84)

    def test_cache_reuse_exact_identity_and_changed_context(self):
        teacher, data, calls = model(), episode(), []
        with tempfile.TemporaryDirectory() as directory:
            labels = ActionValueLabelsV3(teacher, None, LabelV3Config(scale=1), directory,
                {"cache_fingerprint": data["cache_fingerprint"], "teacher_version": 0})
            with patch("gr00t.long_memory.objectives_v3.read_bank", side_effect=bank_read):
                first = labels.get_storage(data, 3, [0, 1, 2], seed=5, loss_fn=weighted_loss({3: 1}, calls))
                count = len(calls)
                self.assertEqual(labels.get_storage(data, 3, [0, 1, 2], seed=5, loss_fn=weighted_loss({}, calls)), first)
                self.assertEqual(len(calls), count)
                second = labels.get_storage(data, 3, [0, 1], seed=5, loss_fn=weighted_loss({3: 1}, calls))
                self.assertNotEqual(first["cache_key"], second["cache_key"])
                self.assertEqual(first["future_decisions"], second["future_decisions"])
                self.assertEqual(first["noise_seeds"], second["noise_seeds"])
                read = labels.get_retrieval(data, 8, [0, 1, 2], seed=6, loss_fn=weighted_loss({0: 2}, calls))
                self.assertEqual(read, labels.get_retrieval(data, 8, [0, 1, 2], seed=6))
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 4)
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(manifest["identity"]["teacher_version"], 0)
            with self.assertRaisesRegex(ValueError, "different teacher/data"):
                ActionValueLabelsV3(teacher, None, LabelV3Config(scale=1), directory,
                    {"cache_fingerprint": data["cache_fingerprint"], "teacher_version": 1})

    def test_mutated_teacher_and_wrong_data_are_rejected(self):
        teacher, data = model(), episode()
        with tempfile.TemporaryDirectory() as directory:
            labels = ActionValueLabelsV3(teacher, None, LabelV3Config(), directory,
                {"cache_fingerprint": data["cache_fingerprint"], "teacher_version": 0})
            altered = copy.deepcopy(data)
            altered["cache_fingerprint"] = "different"
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                labels.get_storage(altered, 3, [0, 1, 2], seed=5)
            with torch.no_grad():
                next(labels.teacher.parameters()).add_(1)
            with self.assertRaisesRegex(ValueError, "teacher changed"):
                labels.get_storage(data, 3, [0, 1, 2], seed=5)

    def test_student_context_mismatch_rejected(self):
        data = episode()
        label = make_storage(data=data)
        for field, mutate in (
            ("episode", lambda value: value.update(episode_id=18)),
            ("layout", lambda value: value["frames"].add_(1)),
            ("cache", lambda value: value.update(cache_fingerprint="changed")),
        ):
            bad = copy.deepcopy(data)
            mutate(bad)
            with self.subTest(field=field), self.assertRaises(ValueError):
                storage_loss(model(frozen=False), bad, label)
        broken = copy.deepcopy(label)
        broken["bank_ids"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "bank differs"):
            storage_loss(model(frozen=False), data, broken)


if __name__ == "__main__":
    unittest.main()
