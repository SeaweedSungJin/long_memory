"""CPU checks of intervention contracts, not evidence of task success."""

import copy
import unittest
from unittest.mock import patch

import torch

from gr00t.long_memory.diagnostic_interventions import (
    continuation_bank, old_event_ids, read_intervention,
)
from gr00t.long_memory.replay_v3 import encode_until, event_inputs, read_bank, replay_bank
from tests.test_long_memory_v3_core import episode, model


class TestDiagnosticInterventions(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(817)
        torch.set_num_threads(1)
        self.memory = model(capacity=8)
        self.data = episode()

    def test_strict_boundary_uses_event_end_not_start_or_index(self):
        self.data["frames"] = torch.tensor([0, 3, 5, 6, 10, 11, 16, 18, 19])
        # d=6, K=4 => boundary endpoint 3, frame 6. Event 2 ends
        # exactly at frame 6, so it belongs to recent, NOT old.
        self.assertEqual(old_event_ids(self.data, 6, [0, 1, 2, 3, 5], 4), [0, 1])
        self.assertEqual(old_event_ids(self.data, 2, [0, 1], 4), [])
        self.assertEqual(old_event_ids(self.data, 6, [0, 1, 2, 3, 5], 1), [0, 1, 2, 3])

    def test_full_matches_legacy_exactly_and_filter_preserves_recent(self):
        bank = [0, 1, 2, 3, 5]
        encoded = encode_until(self.memory, self.data, 6)
        full = read_intervention(self.memory, self.data, 6, bank, encoded=encoded)
        normal = read_bank(self.memory, self.data, 6, bank, encoded)
        self.assertTrue(torch.equal(full["fused_short"], normal["fused_short"]))
        recent = read_intervention(self.memory, self.data, 6, bank, mode="no_old", encoded=encoded)
        old = read_intervention(self.memory, self.data, 6, bank, mode="only_old", encoded=encoded)
        self.assertEqual(recent["diagnostic"]["read_bank_ids"], [2, 3, 5])
        self.assertEqual(old["diagnostic"]["read_bank_ids"], [0, 1])
        self.assertTrue(torch.equal(recent["fused_short"],
                                    read_bank(self.memory, self.data, 6, [2, 3, 5], encoded)["fused_short"]))

    def test_no_memory_is_exact_and_never_calls_reader_or_encoder(self):
        with torch.no_grad():
            for name, value in self.memory.named_parameters():
                if "bias" in name:
                    value.add_(100)
        with patch.object(self.memory, "read", side_effect=AssertionError("Must bypass read")), \
                patch.object(self.memory, "encode_events", side_effect=AssertionError("No events needed")):
            result = read_intervention(self.memory, self.data, 6, [0, 3], mode="no_memory")
        self.assertTrue(torch.equal(result["fused_short"], self.data["short"][6:7]))
        self.assertEqual(float(result["residual_norm"]), 0)
        self.assertFalse(result["diagnostic"]["gate_measured"])
        self.assertEqual(result["weights"].shape, (1, 2, 1))

    def test_shuffle_preserves_each_slot_time_and_recent_keys_values(self):
        data, memory = self.data, self.memory
        bank = [0, 1, 2, 3, 5]
        encoded = encode_until(memory, data, 6)
        raw = event_inputs(data, "cpu", 6)
        expected = memory.encode_events({
            name: value[[0, 1] if name in ("start_frames", "end_frames") else [1, 0]]
            for name, value in raw.items()
        })
        original_read = memory.read
        captured = {}

        def observe(*args):
            captured.update(keys=args[2].clone(), values=args[3].clone())
            return original_read(*args)

        with patch.object(memory, "read", side_effect=observe):
            result = read_intervention(memory, data, 6, bank, mode="shuffled_old", encoded=encoded)
        torch.testing.assert_close(captured["keys"][0, :2], expected["keys"])
        torch.testing.assert_close(captured["values"][0, :2], expected["values"])
        for key in ("keys", "values"):
            self.assertTrue(torch.equal(captured[key][0, 2:], encoded[key][[2, 3, 5]]))
        self.assertEqual(result["diagnostic"]["content_source_by_slot"], [
            dict(slot_event_id=0, content_event_id=1, slot_start_frame=0., slot_end_frame=2.),
            dict(slot_event_id=1, content_event_id=0, slot_start_frame=2., slot_end_frame=4.),
        ])
        self.assertTrue(result["diagnostic"]["intervention_effective"])

    def test_interventions_never_mutate_episode_bank_encoded_or_weights(self):
        data, memory = self.data, self.memory
        bank = [0, 1, 2, 3, 5]
        encoded = encode_until(memory, data, 6)
        before_data = copy.deepcopy(data)
        before_encoded = {key: value.clone() for key, value in encoded.items()}
        before_state = copy.deepcopy(memory.state_dict())
        for mode in ("full", "no_memory", "no_old", "only_old", "shuffled_old"):
            read_intervention(memory, data, 6, bank, mode=mode, encoded=encoded)
        self.assertEqual(bank, [0, 1, 2, 3, 5])
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, before_data[key]), key)
        for key, value in encoded.items():
            self.assertTrue(torch.equal(value, before_encoded[key]), key)
        for key, value in memory.state_dict().items():
            self.assertTrue(torch.equal(value, before_state[key]), key)

    def test_shuffle_insufficient_old_is_explicit_noop(self):
        bank = [1, 2, 3, 5]
        result = read_intervention(self.memory, self.data, 6, bank, mode="shuffled_old")
        full = read_intervention(self.memory, self.data, 6, bank)
        self.assertFalse(result["diagnostic"]["intervention_effective"])
        self.assertEqual(result["diagnostic"]["ineffective_reason"], "fewer_than_two_old_events")
        self.assertTrue(torch.equal(result["fused_short"], full["fused_short"]))

    def test_future_changes_do_not_affect_intervention(self):
        changed = copy.deepcopy(self.data)
        for key in ("short", "moment", "state"):
            changed[key][7:] += 10000
        changed["actions"][6:] += 10000
        for mode in ("full", "no_memory", "no_old", "only_old", "shuffled_old"):
            a = read_intervention(self.memory, self.data, 6, [0, 1, 3, 5], mode=mode)
            b = read_intervention(self.memory, changed, 6, [0, 1, 3, 5], mode=mode)
            self.assertTrue(torch.equal(a["fused_short"], b["fused_short"]), mode)

    def test_empty_bank_and_decision_zero_supported(self):
        for mode in ("full", "no_memory", "no_old", "only_old", "shuffled_old"):
            result = read_intervention(self.memory, self.data, 0, [], mode=mode)
            self.assertEqual(result["diagnostic"]["source_old_count"], 0)
            self.assertTrue(torch.equal(result["fused_short"], self.data["short"][:1]))

    def test_reject_bad_window_prefix_and_noncausal_bank(self):
        for window in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                read_intervention(self.memory, self.data, 6, [0], memory_window=window)
        for bank in ([6], [-1], [1, 0], [0, 0], [1.0]):
            with self.assertRaises(ValueError):
                read_intervention(self.memory, self.data, 6, bank)
        with self.assertRaises(ValueError):
            read_intervention(self.memory, self.data, 6, [0], mode="wrong_episode")

    def test_continuation_from_empty_matches_existing_replay(self):
        memory = model(capacity=3)
        data = self.data
        data["transition_valid"][3] = False
        for policy in ("all", "hard"):
            bank, stats = continuation_bank(memory, data, [], 0, 8, policy=policy)
            expected, base = replay_bank(memory, data, 8, policy=policy)
            self.assertEqual(bank, expected)
            self.assertEqual(stats["attempted"], base["attempted_writes"])
            self.assertEqual(stats["accepted"], base["accepted_writes"])
            self.assertEqual(stats["replaced"], base["replaced"])

    def test_continuation_respects_post_choice_start_and_eviction_trace(self):
        memory = model(capacity=3)
        initial = [0, 1, 2]
        bank, stats = continuation_bank(memory, self.data, initial, 3, 5, policy="all")
        self.assertEqual(bank, [2, 3, 4])
        self.assertEqual(initial, [0, 1, 2])
        self.assertEqual(stats["attempted"], 2)
        self.assertEqual(stats["initial_event_first_eviction_frame"], {"0": 8., "1": 10.})
        with self.assertRaises(ValueError):
            continuation_bank(memory, self.data, [3], 3, 5)
        with self.assertRaises(ValueError):
            continuation_bank(memory, self.data, [], 5, 4)


if __name__ == "__main__":
    unittest.main()
