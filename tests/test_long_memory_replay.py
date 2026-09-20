"""Causal replay/CVoM tests with small CPU tensors, not simulator success tests."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from gr00t.long_memory.core import EpisodicMemory, MemoryConfig
from gr00t.long_memory.cvom import CVoMConfig, CVoMLabels, candidate_indices
from gr00t.long_memory.replay import (
    candidate_predictions, encode_until, event_inputs, read_bank, replay_bank,
)


def memory(capacity=3, min_fill=0, residual_init=0.01):
    return EpisodicMemory(MemoryConfig(
        feature_dim=6, state_dim=3, action_dim=4, hidden_dim=8,
        key_dim=5, value_dim=7, capacity=capacity, min_fill=min_fill,
        residual_init=residual_init,
    ))


def episode(steps=7):
    return {
        "episode_id": 10,
        "short": torch.randn(steps + 1, 2, 6),
        "moment": torch.randn(steps + 1, 2, 6),
        "state": torch.randn(steps + 1, 3),
        "actions": torch.randn(steps, 2, 4),
        "action_mask": torch.ones(steps, 2, dtype=torch.bool),
        "transition_valid": torch.ones(steps, dtype=torch.bool),
        "decision_mask": torch.ones(steps + 1, dtype=torch.bool),
    }


def gradient_sum(module):
    return sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)


def force_write(model, accept):
    with torch.no_grad():
        for parameter in model.write_head.parameters():
            parameter.zero_()
        model.write_head[-1].bias.fill_(30 if accept else -30)


class TestCausalReplay(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(52)

    def test_fifo_capacity_ignores_invalid_events(self):
        model, data = memory(capacity=3), episode()
        data["transition_valid"][4] = False
        ids, diagnostics = replay_bank(model, data, 7, policy="all")
        self.assertEqual(ids, [3, 5, 6])
        self.assertEqual(diagnostics["bank_fill"], 3)
        self.assertEqual(diagnostics["write_rate"], 1)
        self.assertEqual(diagnostics["oldest_event_age"], 4)
        self.assertEqual(replay_bank(model, data, 0)[0], [])

    def test_hard_write_all_none_and_minimum_fill(self):
        data = episode()
        rejecting = memory()
        force_write(rejecting, False)
        self.assertEqual(replay_bank(rejecting, data, 7, "hard")[0], [])
        forced_fill = memory(min_fill=2)
        force_write(forced_fill, False)
        self.assertEqual(replay_bank(forced_fill, data, 7, "hard")[0], [0, 1])
        accepting = memory()
        force_write(accepting, True)
        self.assertEqual(replay_bank(accepting, data, 7, "hard")[0], [4, 5, 6])

    def test_future_perturbation_cannot_change_earlier_read_or_candidate_write(self):
        model, data = memory(min_fill=1), episode()
        altered = copy.deepcopy(data)
        decision = 3
        # Decision 3 may read events 0..2, whose final endpoint is observation
        # 3. Never perturb the current observation while claiming it is future.
        for key in ("short", "moment", "state"):
            altered[key][decision + 1:] = torch.randn_like(altered[key][decision + 1:]) * 1000
        altered["actions"][decision:] = torch.randn_like(altered["actions"][decision:]) * 1000
        altered["action_mask"][decision:] = False
        altered["transition_valid"][decision:] = False
        ids, stats = replay_bank(model, data, decision, "hard")
        other_ids, other_stats = replay_bank(model, altered, decision, "hard")
        self.assertEqual(ids, other_ids)
        self.assertEqual(stats, other_stats)
        original = read_bank(model, data, decision, ids)
        changed = read_bank(model, altered, decision, ids)
        self.assertTrue(torch.equal(original["fused_short"], changed["fused_short"]))
        self.assertTrue(torch.equal(original["read"], changed["read"]))
        u0, w0 = candidate_predictions(model, data, decision - 1)
        u1, w1 = candidate_predictions(model, altered, decision - 1)
        self.assertTrue(torch.equal(u0, u1))
        self.assertTrue(torch.equal(w0, w1))

    def test_unfinished_future_and_invalid_bank_events_are_rejected(self):
        model, data = memory(), episode()
        for ids in ([3], [4], [-1]):
            with self.assertRaisesRegex(ValueError, "future, unfinished, or invalid"):
                read_bank(model, data, decision=3, bank_ids=ids)
        data["transition_valid"][1] = False
        with self.assertRaisesRegex(ValueError, "invalid"):
            read_bank(model, data, decision=3, bank_ids=[1])
        with self.assertRaisesRegex(ValueError, "capacity"):
            read_bank(model, data, decision=7, bank_ids=[0, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            event_inputs(data, "cpu", stop=100)

    def test_selected_past_event_keeps_action_gradient(self):
        model, data = memory(residual_init=0), episode()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        target = torch.randn(1, 2, 6)
        first = read_bank(model, data, 4, [0, 2])
        self.assertTrue(torch.equal(first["fused_short"], data["short"][4:5]))
        (first["fused_short"] - target).square().mean().backward()
        self.assertGreater(gradient_sum(model.fusion[-1]), 0)
        self.assertEqual(gradient_sum(model.event_encoder), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        rebuilt = encode_until(model, data, 4)
        second = read_bank(model, data, 4, [0, 2], rebuilt)
        (second["fused_short"] - target).square().mean().backward()
        for module in (model.event_encoder, model.action_encoder, model.key, model.value):
            self.assertGreater(gradient_sum(module), 0)

    def test_rejected_candidate_still_receives_auxiliary_supervision(self):
        model, data = memory(min_fill=1), episode()
        force_write(model, False)
        self.assertNotIn(2, replay_bank(model, data, 3, "hard")[0])
        utility, logit = candidate_predictions(model, data, 2, "hard")
        loss = utility.square().sum() + torch.nn.functional.binary_cross_entropy_with_logits(
            logit, torch.ones_like(logit)
        )
        loss.backward()
        self.assertGreater(gradient_sum(model.utility_head), 0)
        self.assertGreater(gradient_sum(model.write_head), 0)
        for module in (model.event_encoder, model.query, model.key, model.value, model.fusion):
            self.assertEqual(gradient_sum(module), 0)


class TestCVoMPairing(unittest.TestCase):
    def test_candidate_set_excludes_invalid_or_unfinished_events(self):
        data = episode()
        data["transition_valid"][1] = False
        data["decision_mask"][:] = False
        data["decision_mask"][3] = True
        self.assertEqual(candidate_indices(data), [0, 2])
        data["decision_mask"][:] = False
        self.assertEqual(candidate_indices(data), [])

    def test_coalition_pairing_same_noise_seed_and_no_intervening_events(self):
        teacher, data = memory(), episode()
        config = CVoMConfig(future_samples=3, coalitions=2, far_delay=3, seed=7)
        read_calls, loss_calls = [], []

        def fake_read(memory_model, episode_data, decision, bank_ids, encoded):
            read_calls.append((decision, list(bank_ids)))
            # A deliberately simple scalar lets the fake objective count the
            # candidate's known gain independently of the neural reader.
            return {"fused_short": torch.tensor([float(len(bank_ids))])}

        def fake_loss(head, episode_data, decision, fused_short, *, seed):
            loss_calls.append((decision, seed))
            return {"loss": 10.0 - fused_short.sum()}

        with tempfile.TemporaryDirectory() as directory:
            labeler = CVoMLabels(teacher, None, config, directory, identity={"test": "paired"})
            with patch("gr00t.long_memory.cvom.read_bank", side_effect=fake_read), patch(
                "gr00t.long_memory.cvom.episode_flow_loss", side_effect=fake_loss
            ):
                label = labeler.get(data, candidate=2)
                count = len(loss_calls)
                self.assertEqual(labeler.get(data, candidate=2), label)
                self.assertEqual(len(loss_calls), count, "disk cache must not recompute labels")
            self.assertEqual(label["signed_gain"], 1.0)
            self.assertEqual(label["write_target"], 1)
            self.assertTrue(all(i > 2 for i in label["future_decisions"]))
            for offset in range(0, len(loss_calls), 2):
                self.assertEqual(loss_calls[offset], loss_calls[offset + 1])
                decision, without = read_calls[offset]
                other_decision, with_candidate = read_calls[offset + 1]
                self.assertEqual(decision, other_decision)
                self.assertEqual(with_candidate, without + [2])
                self.assertTrue(all(i < 2 for i in without))
                self.assertLessEqual(len(with_candidate), teacher.config.capacity)
            with self.assertRaisesRegex(ValueError, "no valid future"):
                labeler.get(data, candidate=7)
            self.assertTrue((Path(directory) / "manifest.json").is_file())
            with self.assertRaisesRegex(ValueError, "different teacher"):
                CVoMLabels(memory(), None, config, directory, identity={"test": "different"})


if __name__ == "__main__":
    unittest.main()
