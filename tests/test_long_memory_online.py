"""Online/replay parity tests: no simulator, real robot, or large model required."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from gr00t.long_memory.core import EpisodicMemory, MemoryConfig
from gr00t.long_memory.online import OnlineEpisodicBank
from gr00t.long_memory.replay import read_bank, replay_bank


def model(capacity=3, min_fill=2):
    return EpisodicMemory(MemoryConfig(feature_dim=6, state_dim=3, action_dim=4,
        hidden_dim=8, key_dim=5, value_dim=7, capacity=capacity,
        min_fill=min_fill, residual_init=0.03)).eval().requires_grad_(False)


def episode():
    steps, stride = 8, 2
    masks = torch.ones(steps, stride, dtype=torch.bool)
    masks[:2] = False
    actions = torch.randn(steps, stride, 4)
    actions[:2] = 0
    return {"short": torch.randn(steps + 1, 2, 6), "moment": torch.randn(steps + 1, 2, 6),
        "state": torch.randn(steps + 1, 3), "actions": actions, "action_mask": masks,
        "transition_valid": torch.ones(steps, dtype=torch.bool),
        "frames": torch.arange(steps + 1) * stride,
        "is_demo": torch.tensor([True, True] + [False] * (steps - 1))}


def advance(bank, data, index):
    return bank.advance(data["short"][index], data["moment"][index], data["state"][index],
        frame=int(data["frames"][index]), passive=bool(data["is_demo"][index]),
        actions=None if index == 0 else data["actions"][index - 1][data["action_mask"][index - 1]])


class TestOnlineBank(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_replay_parity_all_hard_accept_reject(self):
        for policy, bias in [("all", None), ("hard", 30), ("hard", -30), ("hard", None)]:
            with self.subTest(policy=policy, bias=bias):
                memory, data = model(), episode()
                if bias is not None:
                    for p in memory.write_head.parameters():
                        p.zero_()
                    memory.write_head[-1].bias.fill_(bias)
                online = OnlineEpisodicBank(memory, 2, policy)
                for d in range(len(data["frames"])):
                    fused, stats = advance(online, data, d)
                    ids, expected_stats = replay_bank(memory, data, d, policy)
                    expected = read_bank(memory, data, d, ids)
                    self.assertEqual(online.bank_ids, ids)
                    torch.testing.assert_close(fused, expected["fused_short"], atol=2e-6, rtol=2e-5)
                    for key in ("bank_fill", "oldest_event_age", "write_rate"):
                        self.assertEqual(stats[key], expected_stats[key])
                    self.assertLessEqual(len(online.keys), memory.config.capacity)
                    self.assertTrue(all(not key.requires_grad for key in online.keys))

    def test_rejected_writer_reports_forced_vs_learned(self):
        memory, data = model(), episode()
        for p in memory.write_head.parameters():
            p.zero_()
        memory.write_head[-1].bias.fill_(-30)
        bank = OnlineEpisodicBank(memory, 2, "hard")
        for d in range(len(data["frames"])):
            _, stats = advance(bank, data, d)
        self.assertEqual(stats["bank_event_ids"], [0, 1])
        self.assertEqual(stats["forced_writes"], 2)
        self.assertEqual(stats["learned_write_attempts"], 6)
        self.assertEqual(stats["learned_write_accepts"], 0)
        self.assertEqual(stats["learned_write_rate"], 0)

    def test_first_endpoint_identity_and_reset(self):
        memory, data = model(), episode()
        bank = OnlineEpisodicBank(memory, 2, "all")
        fused, stats = advance(bank, data, 0)
        self.assertTrue(torch.equal(fused[0], data["short"][0]))
        self.assertEqual(stats["attempted_writes"], 0)
        advance(bank, data, 1)
        bank.reset()
        fused, stats = advance(bank, data, 0)
        self.assertTrue(torch.equal(fused[0], data["short"][0]))
        self.assertEqual(stats["bank_fill"], 0)

    def test_future_changes_do_not_change_past_bank(self):
        memory, first = model(), episode()
        second = copy.deepcopy(first)
        second["moment"][4:] += 100
        a, b = OnlineEpisodicBank(memory, 2, "hard"), OnlineEpisodicBank(memory, 2, "hard")
        for d in range(4):
            x, _ = advance(a, first, d)
            y, _ = advance(b, second, d)
            self.assertTrue(torch.equal(x, y))
            self.assertEqual(a.bank_ids, b.bank_ids)

    def test_no_fabricated_passive_actions(self):
        memory, data = model(), episode()
        bank = OnlineEpisodicBank(memory, 2, "all")
        advance(bank, data, 0)
        with self.assertRaisesRegex(ValueError, "Passive"):
            bank.advance(data["short"][1], data["moment"][1], data["state"][1],
                frame=2, passive=True, actions=torch.zeros(2, 4))
        self.assertEqual(bank.attempted, 0)

    def test_wrong_cadence_and_missing_controls_rejected(self):
        memory, data = model(), episode()
        bank = OnlineEpisodicBank(memory, 2, "all")
        for d in range(3):
            advance(bank, data, d)
        for frame, controls in [(4, torch.zeros(0, 4)), (9, torch.zeros(5, 4)), (6, None)]:
            with self.assertRaises(ValueError):
                bank.advance(data["short"][3], data["moment"][3], data["state"][3],
                    frame=frame, passive=False, actions=controls)
        self.assertEqual(bank.attempted, 2)

    def test_partial_final_interval(self):
        memory, data = model(), episode()
        data["frames"][-1] -= 1
        data["action_mask"][-1, 1] = False
        bank = OnlineEpisodicBank(memory, 2, "all")
        for d in range(len(data["frames"])):
            fused, _ = advance(bank, data, d)
        ids, _ = replay_bank(memory, data, 8, "all")
        torch.testing.assert_close(fused, read_bank(memory, data, 8, ids)["fused_short"])

    def test_requires_frozen_eval_model(self):
        memory = model().train()
        with self.assertRaises(ValueError):
            OnlineEpisodicBank(memory, 2, "all")


class TestExecutedActionNormalization(unittest.TestCase):
    def test_original_normalizer_prestate_key_order_and_padding(self):
        from gr00t.long_memory.online_policy import normalize_executed_actions
        captured = {}
        state = {"joint_position": np.zeros((1, 7), np.float32)}

        def apply_action(action, embodiment, state):
            captured.update(action=action, embodiment=embodiment, state=state)
            return {key: np.clip(value * 2, -1, 1) for key, value in action.items()}

        processor = SimpleNamespace(
            modality_configs={"new_embodiment": {"action": SimpleNamespace(modality_keys=["gripper_close", "joint_position"])}},
            state_action_processor=SimpleNamespace(apply_action=apply_action))
        raw = np.arange(16, dtype=np.float32).reshape(2, 8) / 10
        result = normalize_executed_actions(processor, "new_embodiment", raw, state, 128)
        self.assertIs(captured["state"], state)
        np.testing.assert_array_equal(captured["action"]["gripper_close"], raw[:, 7:8])
        np.testing.assert_array_equal(result[:, 0].numpy(), np.clip(raw[:, 7] * 2, -1, 1))
        self.assertTrue(torch.equal(result[:, 8:], torch.zeros(2, 120)))
        self.assertEqual(result.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
