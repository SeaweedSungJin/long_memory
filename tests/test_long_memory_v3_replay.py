"""Causal option-set and replay tests for action-value memory v3."""

import copy
import unittest
from unittest.mock import patch

import torch

from gr00t.long_memory.replay_v3 import (encode_until, event_inputs, read_bank,
                                        replay_bank, storage_options, storage_prediction)
from tests.test_long_memory_v3_core import episode, model


class TestV3Replay(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(22)
        torch.set_num_threads(1)

    def test_options_keep_append_or_fixed_capacity_replacements(self):
        self.assertEqual(storage_options([], 3, 3, 2), [[], [3]])
        self.assertEqual(storage_options([0, 2], 4, 3, 2), [[0, 2], [0, 2, 4]])
        self.assertEqual(storage_options([0, 2, 4], 6, 3, 2), [[0, 2, 4], [2, 4, 6], [0, 2, 6]])
        options = storage_options(list(range(64)), 65, 64, 3)
        self.assertEqual(len(options), 4)
        self.assertTrue(all(len(row) == 64 and row == sorted(set(row)) for row in options))
        self.assertEqual(storage_options([0], 1, 1, 3), [[0], [1]])
        for ids in ([1, 0], [0, 0], [-1], [5]):
            with self.assertRaises(ValueError):
                storage_options(ids, 5, 3, 2)

    def test_all_fifo_skips_invalid_and_hard_min_fill_counts(self):
        memory, data = model(), episode()
        data["transition_valid"][3] = False
        bank, stats = replay_bank(memory, data, 7, "all")
        self.assertEqual(bank, [4, 5, 6])
        self.assertEqual(stats["attempted_writes"], 6)
        with torch.no_grad():
            for p in memory.writer.parameters():
                p.zero_()  # Equal logits -> deterministic KEEP, no hidden threshold.
        bank, stats = replay_bank(memory, data, 7, "hard")
        self.assertEqual(bank, [0])
        self.assertEqual(stats["forced_writes"], 1)
        self.assertEqual(stats["learned_attempted"], 5)
        self.assertEqual(stats["learned_rejected"], 5)
        self.assertEqual(stats["learned_accepted"], 0)

    def test_hard_replacement_is_the_scored_option_not_unconditional_fifo(self):
        memory, data = model(), episode()

        def prefer_last(*args):
            return torch.arange(len(args[4]), dtype=torch.float32)

        with patch.object(memory.writer, "forward", side_effect=prefer_last):
            bank, stats = replay_bank(memory, data, 8, "hard")
        # Newest victim replaced each time: events 0 and 1 remain, unlike FIFO.
        self.assertEqual(bank, [0, 1, 7])
        self.assertEqual(stats["learned_replaced"], 5)

    def test_writer_reads_completion_endpoint_and_no_future_observations(self):
        memory, data = model(), episode()
        captured = {}

        def capture(*args):
            captured.update(short=args[0], state=args[1])
            return torch.zeros(len(args[4]))

        with patch.object(memory.writer, "forward", side_effect=capture):
            storage_prediction(memory, data, 2, [0, 1])
        torch.testing.assert_close(captured["short"], data["short"][3])
        torch.testing.assert_close(captured["state"], data["state"][3])
        changed = copy.deepcopy(data)
        for key in ("short", "moment", "state"):
            changed[key][4:] += 1000
        changed["actions"][3:] += 1000
        a = storage_prediction(memory, data, 2, [0, 1])
        b = storage_prediction(memory, changed, 2, [0, 1])
        self.assertTrue(torch.equal(a["logits"], b["logits"]))
        self.assertEqual(a["options"], b["options"])

    def test_read_and_replay_causal_under_future_perturbation(self):
        memory, data = model(), episode()
        changed = copy.deepcopy(data)
        for key in ("short", "moment", "state"):
            changed[key][5:] += 1000
        changed["actions"][4:] += 1000
        a, sa = replay_bank(memory, data, 4, "hard")
        b, sb = replay_bank(memory, changed, 4, "hard")
        self.assertEqual(a, b)
        self.assertEqual(sa, sb)
        self.assertTrue(torch.equal(read_bank(memory, data, 4, a)["fused_short"],
                                    read_bank(memory, changed, 4, b)["fused_short"]))

    def test_duplicate_future_invalid_and_oversized_banks_rejected(self):
        memory, data = model(), episode()
        for ids in ([0, 0], [2, 0], [4], [-1], [0, 1, 2, 3]):
            with self.assertRaises(ValueError):
                read_bank(memory, data, 4, ids)
        data["transition_valid"][1] = False
        with self.assertRaisesRegex(ValueError, "invalid"):
            read_bank(memory, data, 4, [1])
        with self.assertRaises(ValueError):
            storage_prediction(memory, data, 1, [0])

    def test_raw_frame_metadata_is_required_and_partial_intervals_preserved(self):
        memory, data = model(), episode()
        data["frames"][-1] -= 1
        encoded = encode_until(memory, data, 8)
        self.assertEqual(float(encoded["ends"][-1] - encoded["starts"][-1]), 1)
        del data["frames"]
        with self.assertRaisesRegex(ValueError, "raw endpoint frames"):
            event_inputs(data, "cpu")


if __name__ == "__main__":
    unittest.main()
