"""CPU-only Stage-2 v2 context/causality/uncertainty regression tests."""

import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from gr00t.long_memory.contextual_cvom import (
    ContextualCVoMConfig, ContextualCVoMLabels, eligible_candidates,
    eligible_futures, sample_context, validate_context,
)
from gr00t.long_memory.core import EpisodicMemory, MemoryConfig


def memory(capacity=4):
    return EpisodicMemory(MemoryConfig(
        feature_dim=6, state_dim=3, action_dim=4, hidden_dim=8,
        key_dim=5, value_dim=7, capacity=capacity, min_fill=0,
        residual_init=0.01,
    ))


def episode(steps=12):
    return {
        "episode_id": 17, "frames": torch.arange(steps + 1) * 16,
        "short": torch.randn(steps + 1, 2, 6),
        "moment": torch.randn(steps + 1, 2, 6),
        "state": torch.randn(steps + 1, 3),
        "actions": torch.randn(steps, 2, 4),
        "action_mask": torch.ones(steps, 2, dtype=torch.bool),
        "transition_valid": torch.ones(steps, dtype=torch.bool),
        # Production EpisodeCache contract: T actionable decisions, T+1
        # observations. Never invent an action target at the final endpoint.
        "decision_mask": torch.ones(steps, dtype=torch.bool),
    }


def fake_read(model, data, decision, bank_ids, encoded):
    return {"fused_short": torch.tensor([float(len(bank_ids))], dtype=torch.float64)}


def fake_loss(head, data, decision, fused_short, *, seed):
    # Same MC draw cancels exactly between the two branches.
    return {"loss": 10.0 + seed % 7 - fused_short.sum()}


def labeler(directory, model=None, **settings):
    return ContextualCVoMLabels(
        memory() if model is None else model, None,
        ContextualCVoMConfig(**settings), directory,
        {"cache_fingerprint": "fake-data-and-expert", "teacher_version": 0},
    )


class TestContextualEligibility(unittest.TestCase):
    def test_true_event_end_must_leave_short_window(self):
        data = episode()
        self.assertNotIn(6, eligible_futures(data, candidate=2, memory_window=4))
        self.assertEqual(eligible_futures(data, candidate=2, memory_window=4), list(range(7, 12)))
        self.assertEqual(eligible_candidates(data, memory_window=4), list(range(7)))
        self.assertEqual(eligible_futures(data, candidate=2, memory_window=1), list(range(4, 12)))

    def test_irregular_frames_and_passive_queries(self):
        data = episode(steps=8)
        data["frames"] = torch.tensor([0, 6, 22, 38, 54, 70, 86, 102, 107])
        data["decision_mask"][6] = False
        self.assertEqual(eligible_futures(data, candidate=1, memory_window=4), [7])
        data["transition_valid"][1] = False
        self.assertNotIn(1, eligible_candidates(data, memory_window=4))
        with self.assertRaisesRegex(ValueError, "valid completed"):
            eligible_futures(data, 1)
        data["decision_mask"][:] = False
        self.assertEqual(eligible_candidates(data), [])

    def test_production_mask_contract_excludes_terminal_endpoint(self):
        data = episode(steps=5)
        self.assertEqual(len(data["frames"]), len(data["decision_mask"]) + 1)
        # Candidate 0 first leaves K=4 memory at endpoint 5, but endpoint 5
        # is final and has no GT action. This episode has NO eligible labels.
        self.assertEqual(eligible_candidates(data, memory_window=4), [])
        data["decision_mask"] = torch.ones(6, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "T action decisions"):
            eligible_candidates(data, memory_window=4)

    def test_irregular_gap_uses_actual_rolling_endpoints_not_nominal_stride(self):
        data = episode(steps=7)
        data["frames"] = torch.tensor([0, 6, 22, 25, 41, 57, 73, 80])
        # At d=5, the actual K=4 rolling cache is [22,25,41,57]. Event 0
        # ends at 6 and is old; event 1 ends at 22 and still overlaps. Using
        # 57 - 3*16=9 instead would describe a different short input history.
        self.assertIn(5, eligible_futures(data, 0, memory_window=4))
        self.assertNotIn(5, eligible_futures(data, 1, memory_window=4))
        self.assertIn(6, eligible_futures(data, 1, memory_window=4))

    def test_invalid_layout_and_configuration_are_rejected(self):
        for settings in ({"noise_samples": 1}, {"memory_window": 0}, {"future_samples": 1.5},
                         {"utility_scale": 0}, {"utility_scale": float("nan")},
                         {"uncertainty_z": -1}, {"write_delta": float("inf")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ContextualCVoMConfig(**settings)
        data = episode()
        data["frames"][3] = data["frames"][2]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            eligible_candidates(data)
        data = episode()
        data["decision_mask"] = data["decision_mask"].float()
        with self.assertRaisesRegex(ValueError, "boolean mask"):
            eligible_candidates(data)

    def test_context_rejects_duplicates_future_invalid_and_full_banks(self):
        data = episode()
        self.assertEqual(validate_context(data, 4, [0, 2], 3), [0, 2])
        for bank in ([2, 0], [0, 0], [4], [-1], [0.0], [True], [0, 1, 2]):
            with self.subTest(bank=bank), self.assertRaises(ValueError):
                validate_context(data, 4, bank, 3)
        data["transition_valid"][1] = False
        with self.assertRaisesRegex(ValueError, "valid events"):
            validate_context(data, 4, [1], 3)
        for seed in range(30):
            bank = sample_context(data, 4, 3, random.Random(seed))
            self.assertEqual(bank, validate_context(data, 4, bank, 3))
            self.assertNotIn(1, bank)
        self.assertEqual(sample_context(data, 0, 3, random.Random(0)), [])


class TestContextualLabels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(191)

    def test_pairing_cache_and_future_context_are_exact(self):
        data = episode()
        read_calls, loss_calls = [], []

        def tracked_read(model, data, decision, bank_ids, encoded):
            read_calls.append((decision, list(bank_ids)))
            return fake_read(model, data, decision, bank_ids, encoded)

        def tracked_loss(head, data, decision, fused_short, *, seed):
            loss_calls.append((decision, seed))
            return fake_loss(head, data, decision, fused_short, seed=seed)

        with tempfile.TemporaryDirectory() as directory:
            labels = labeler(directory)
            with patch("gr00t.long_memory.contextual_cvom.read_bank", side_effect=tracked_read), patch(
                "gr00t.long_memory.contextual_cvom.episode_flow_loss", side_effect=tracked_loss
            ):
                record = labels.get(data, 2, [0])
                self.assertEqual(len(loss_calls), 4 * 4 * 2)
                self.assertEqual(labels.get(data, 2, [0]), record)
                self.assertEqual(len(loss_calls), 4 * 4 * 2, "cache must skip expert forwards")
                other = labels.get(data, 2, [1])
            self.assertNotEqual(record["cache_key"], other["cache_key"])
            self.assertEqual(record["future_decisions"], other["future_decisions"])
            self.assertEqual(record["noise_seeds"], other["noise_seeds"])
            self.assertEqual(len(list(Path(directory).glob("episode-*.json"))), 2)
            self.assertEqual(record["signed_gain"], 1.0)
            self.assertEqual(record["utility_target"], math.log1p(1000))
            self.assertEqual(record["write_target"], 1)
            self.assertEqual(record["write_weight"], 1)
            self.assertTrue(record["label_confident"])
            self.assertEqual(record["gain_se"], 0)
            self.assertTrue(all(d >= 7 for d in record["future_decisions"]))
            for offset in range(0, len(loss_calls), 2):
                self.assertEqual(loss_calls[offset], loss_calls[offset + 1])
            for offset in range(0, len(read_calls), 2):
                decision, without = read_calls[offset]
                other_decision, with_candidate = read_calls[offset + 1]
                self.assertEqual(decision, other_decision)
                self.assertEqual(with_candidate, without + [2])
                self.assertTrue(all(i < 2 for i in without))

    def test_noise_uncertainty_masks_ambiguous_bce_without_dropping_regression(self):
        data = episode()
        counter = 0

        def noisy_loss(head, data, decision, fused_short, *, seed):
            nonlocal counter
            draw = counter // 2
            counter += 1
            gain = -1.0 if draw % 2 == 0 else 1.0
            return {"loss": 10.0 - fused_short.sum() * gain}

        with tempfile.TemporaryDirectory() as directory:
            labels = labeler(directory, future_samples=1, noise_samples=4)
            with patch("gr00t.long_memory.contextual_cvom.read_bank", side_effect=fake_read), patch(
                "gr00t.long_memory.contextual_cvom.episode_flow_loss", side_effect=noisy_loss
            ):
                record = labels.get(data, 2, [])
            self.assertEqual(record["signed_gain"], 0)
            self.assertGreater(record["within_noise_se"], 0)
            self.assertEqual(record["future_mean_std"], 0)
            self.assertFalse(record["label_confident"])
            self.assertEqual(record["write_weight"], 0)
            self.assertEqual(record["utility_weight"], 1)
            self.assertLess(record["gain_lower"], labels.config.write_delta)
            self.assertGreater(record["gain_upper"], labels.config.write_delta)

    def test_future_heterogeneity_not_misreported_as_noise(self):
        data = episode()

        def heterogeneous_loss(head, data, decision, fused_short, *, seed):
            gain = -1.0 if decision % 2 else 1.0
            return {"loss": 10.0 - fused_short.sum() * gain}

        with tempfile.TemporaryDirectory() as directory:
            labels = labeler(directory, future_samples=100)
            with patch("gr00t.long_memory.contextual_cvom.read_bank", side_effect=fake_read), patch(
                "gr00t.long_memory.contextual_cvom.episode_flow_loss", side_effect=heterogeneous_loss
            ):
                record = labels.get(data, 2, [])
            self.assertEqual(record["within_noise_se"], 0)
            self.assertGreater(record["future_mean_std"], 0)
            self.assertGreater(record["gain_se"], 0)
            self.assertEqual(record["write_weight"], 0)

    def test_teacher_identity_weight_data_and_config_changes_invalidate_cache(self):
        teacher = memory()
        identity = {"cache_fingerprint": "data-a", "teacher_version": 0}
        with tempfile.TemporaryDirectory() as directory:
            first = ContextualCVoMLabels(teacher, None, ContextualCVoMConfig(), directory, identity)
            same = ContextualCVoMLabels(copy.deepcopy(teacher), None, ContextualCVoMConfig(), directory, identity)
            self.assertEqual(first.identity_hash, same.identity_hash)
            for key, value in (("cache_fingerprint", "data-b"), ("teacher_version", 1)):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "different teacher/data"):
                    ContextualCVoMLabels(copy.deepcopy(teacher), None, ContextualCVoMConfig(), directory, dict(identity, **{key: value}))
            with self.assertRaisesRegex(ValueError, "different teacher/data"):
                ContextualCVoMLabels(copy.deepcopy(teacher), None, ContextualCVoMConfig(noise_samples=2), directory, identity)
            changed = copy.deepcopy(teacher)
            with torch.no_grad():
                next(changed.parameters()).add_(0.01)
            with self.assertRaisesRegex(ValueError, "different teacher/data"):
                ContextualCVoMLabels(changed, None, ContextualCVoMConfig(), directory, identity)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "teacher_version"):
            ContextualCVoMLabels(memory(), None, ContextualCVoMConfig(), directory, {"cache_fingerprint": "a"})

    def test_nonfinite_teacher_loss_and_cached_label_rejected(self):
        data = episode()
        with tempfile.TemporaryDirectory() as directory:
            labels = labeler(directory)
            with patch("gr00t.long_memory.contextual_cvom.episode_flow_loss", return_value={"loss": torch.tensor(float("nan"))}):
                with self.assertRaisesRegex(FloatingPointError, "teacher loss"):
                    labels.get(data, 2, [])
            self.assertEqual(list(Path(directory).glob("episode-*.json")), [])
            with patch("gr00t.long_memory.contextual_cvom.read_bank", side_effect=fake_read), patch(
                "gr00t.long_memory.contextual_cvom.episode_flow_loss", side_effect=fake_loss
            ):
                record = labels.get(data, 2, [])
            path = next(Path(directory).glob("episode-*.json"))
            record["signed_gain"] = float("nan")
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(FloatingPointError, "label cache"):
                labels.get(data, 2, [])

    def test_candidate_without_old_query_and_unfrozen_teacher_rejected(self):
        data = episode()
        with tempfile.TemporaryDirectory() as directory:
            labels = labeler(directory)
            with self.assertRaisesRegex(ValueError, "no old-only"):
                labels.get(data, 8, [])
            labels.teacher.train()
            with self.assertRaisesRegex(ValueError, "frozen and in eval"):
                labels.get(data, 2, [])


if __name__ == "__main__":
    unittest.main()
