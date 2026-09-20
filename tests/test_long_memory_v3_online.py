"""Frozen online and recomputed training replay must make identical decisions."""

import contextlib
import json
import unittest
from unittest.mock import patch

import torch

from gr00t.long_memory.online_v3 import OnlineActionValueBank
from gr00t.long_memory.replay_v3 import read_bank, replay_bank
from tests.test_long_memory_v3_core import episode, model


def data_with_demo():
    data = episode()
    data["action_mask"][:2] = False
    data["actions"][:2] = 0
    data["is_demo"] = torch.tensor([True, True] + [False] * 7)
    return data


def advance(bank, data, decision):
    return bank.advance(data["short"][decision], data["moment"][decision], data["state"][decision],
                        frame=int(data["frames"][decision]), passive=bool(data["is_demo"][decision]),
                        actions=None if decision == 0 else data["actions"][decision - 1][data["action_mask"][decision - 1]])


class TestV3Online(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)

    def test_replay_parity_all_hard_learned_keep_replace_and_partial_interval(self):
        for policy, chooser in (("all", None), ("hard", None), ("hard", "keep"), ("hard", "last")):
            with self.subTest(policy=policy, chooser=chooser):
                memory, data = model().eval().requires_grad_(False), data_with_demo()
                data["frames"][-1] -= 1
                data["action_mask"][-1, 1] = False

                def choose(*args):
                    scores = torch.arange(len(args[4]), dtype=torch.float32)
                    return -scores if chooser == "keep" else scores

                manager = patch.object(memory.writer, "forward", side_effect=choose) if chooser else contextlib.nullcontext()
                with manager:
                    online = OnlineActionValueBank(memory, 2, policy)
                    for decision in range(9):
                        fused, stats = advance(online, data, decision)
                        ids, expected_stats = replay_bank(memory, data, decision, policy)
                        self.assertEqual(online.bank_ids, ids)
                        with torch.no_grad():
                            expected = read_bank(memory, data, decision, ids)["fused_short"]
                        torch.testing.assert_close(fused, expected, atol=3e-6, rtol=3e-5)
                        for key in ("bank_fill", "write_rate", "oldest_event_age", "oldest_event_age_frames",
                                    "learned_attempted", "learned_accepted", "learned_rejected", "learned_replaced"):
                            self.assertEqual(stats[key], expected_stats[key])
                        self.assertLessEqual(len(online.encodings), memory.config.capacity)
                        self.assertTrue(all(not row["keys"].requires_grad for row in online.encodings.values()))
                        json.dumps(stats, allow_nan=False)

    def test_reset_restores_empty_exact_baseline(self):
        memory, data = model().eval().requires_grad_(False), data_with_demo()
        bank = OnlineActionValueBank(memory, 2, "hard")
        for decision in range(4):
            advance(bank, data, decision)
        bank.reset()
        fused, stats = advance(bank, data, 0)
        self.assertTrue(torch.equal(fused[0], data["short"][0]))
        self.assertEqual(stats["attempted_writes"], 0)
        self.assertEqual(stats["bank_fill"], 0)

    def test_no_fabricated_passive_controls_and_no_future_action_chunk(self):
        memory, data = model().eval().requires_grad_(False), data_with_demo()
        bank = OnlineActionValueBank(memory, 2, "all")
        advance(bank, data, 0)
        with self.assertRaisesRegex(ValueError, "Passive"):
            bank.advance(data["short"][1], data["moment"][1], data["state"][1],
                         frame=2, passive=True, actions=torch.zeros(2, 4))
        advance(bank, data, 1)
        advance(bank, data, 2)
        with self.assertRaisesRegex(ValueError, "one actual control"):
            bank.advance(data["short"][3], data["moment"][3], data["state"][3],
                         frame=6, passive=False, actions=torch.zeros(5, 4))
        self.assertEqual(bank.attempted, 2)

    def test_requires_frozen_eval_memory_and_valid_cadence(self):
        with self.assertRaises(ValueError):
            OnlineActionValueBank(model(), 2, "all")
        memory, data = model().eval().requires_grad_(False), data_with_demo()
        bank = OnlineActionValueBank(memory, 2, "all")
        advance(bank, data, 0)
        with self.assertRaisesRegex(ValueError, "Endpoints"):
            bank.advance(data["short"][1], data["moment"][1], data["state"][1], frame=5, passive=True)


if __name__ == "__main__":
    unittest.main()
