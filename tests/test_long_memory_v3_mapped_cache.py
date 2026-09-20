"""Mapped cache validation keeps causal checks without scanning all VL rows."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gr00t.long_memory.cache_reader_v3 import MappedEpisodes, validate_decision, validate_mapped_episode
from gr00t.long_memory.objectives_v3 import _scalar_loss


def episode():
    n, q, d, a = 4, 2, 6, 3
    short = torch.randn(n + 1, q, d)
    return {"episode_id": 7, "cache_fingerprint": "fixture", "embodiment_id": 1,
            "frames": torch.tensor([0, 2, 3, 5, 7]),
            "short": short, "moment": torch.randn_like(short), "state": torch.randn(n + 1, 2),
            "actions": torch.zeros(n, 2, a), "targets": torch.randn(n, 4, a),
            "action_mask": torch.tensor([[True, True], [True, False], [True, True], [True, True]]),
            "target_mask": torch.ones(n, 4, a, dtype=torch.bool),
            "decision_mask": torch.ones(n, dtype=torch.bool), "transition_valid": torch.ones(n, dtype=torch.bool),
            "is_demo": torch.zeros(n + 1, dtype=torch.bool),
            "features": [torch.cat((torch.randn(4, d), row)) for row in short],
            "attention_masks": [torch.ones(6, dtype=torch.bool) for _ in range(n + 1)],
            "image_masks": [torch.zeros(6, dtype=torch.bool) for _ in range(n + 1)]}


class TestMappedCache(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def cache(self, data):
        path = self.root / "episode.pt"
        torch.save(data, path)
        return SimpleNamespace(path=self.root, manifest={"fingerprint": "fixture", "feature_dim": 6,
            "state_dim": 2, "action_dim": 3, "episodes": [{"episode_id": 7, "path": "episode.pt"}]})

    def test_valid_mapping_and_lru(self):
        data = episode()
        mapped = MappedEpisodes(self.cache(data))
        first = mapped.fetch(7)
        self.assertIs(first, mapped.fetch(7))
        for d in range(4):
            validate_decision(first, d)
        self.assertEqual(mapped.fetch.cache_info().hits, 1)

    def test_passive_and_endpoint_prefix_leakage_rejected(self):
        mutations = [
            (lambda ep: ep["action_mask"].__setitem__((1, 1), True), "beyond next"),
            (lambda ep: ep["actions"].__setitem__((1, 1, 0), 1), "Padded/passive"),
            (lambda ep: ep["decision_mask"].__setitem__(0, False), "Passive demo"),
            (lambda ep: ep["is_demo"].__setitem__(0, True), "Demo endpoint"),
            (lambda ep: ep["frames"].__setitem__(4, 8), "cadence"),
        ]
        for mutate, error in mutations:
            data = episode()
            mutate(data)
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                MappedEpisodes(self.cache(data)).fetch(7)
        data = episode()
        data["decision_mask"][0] = False
        data["action_mask"][0] = False
        with self.assertRaisesRegex(ValueError, "action-loss target"):
            validate_mapped_episode(data)
        data["target_mask"][0] = False
        data["is_demo"][0] = True
        validate_mapped_episode(data)

    def test_large_unselected_vl_and_target_values_are_not_scanned(self):
        data = episode()
        data["features"][3][0, 0] = float("nan")
        data["targets"][2][0, 0] = float("nan")
        mapped = MappedEpisodes(self.cache(data))
        value = mapped.fetch(7)  # Shape/meta and small event tensors only.
        validate_decision(value, 0)
        with self.assertRaisesRegex(FloatingPointError, "features"):
            validate_decision(value, 3)
        with self.assertRaisesRegex(FloatingPointError, "targets"):
            validate_decision(value, 2)

    def test_small_event_nonfinite_rejected_before_reencoding(self):
        for name in ("short", "moment", "state", "actions"):
            data = episode()
            data[name].reshape(-1)[0] = float("nan")
            with self.subTest(name=name), self.assertRaisesRegex(FloatingPointError, name):
                MappedEpisodes(self.cache(data)).fetch(7)

    def test_selected_tail_and_mask_contract(self):
        data = episode()
        data["features"][2][-1, 0] += 1
        validate_mapped_episode(data)  # Tail values are selected-row only.
        with self.assertRaisesRegex(ValueError, "conditioning tail"):
            validate_decision(data, 2)
        for name in ("attention_masks", "image_masks"):
            data = episode()
            data[name][0] = torch.ones(5, dtype=torch.bool)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                validate_decision(data, 0)
            with self.assertRaisesRegex(ValueError, name):
                validate_mapped_episode(data)
        data = episode()
        data["target_mask"][0] = False
        with self.assertRaisesRegex(ValueError, "no valid target"):
            validate_decision(data, 0)

    def test_layout_shapes_identity_and_dimensions(self):
        cases = [
            ("frames", torch.arange(5).float()),
            ("action_mask", torch.ones(4, 2)),
            ("target_mask", torch.ones(4, 4, 3)),
            ("short", torch.ones(4, 2, 6)),
            ("state", torch.ones(4, 2)),
        ]
        for key, value in cases:
            data = episode(); data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_mapped_episode(data)
        with self.assertRaisesRegex(ValueError, "feature_dim"):
            validate_mapped_episode(episode(), {"feature_dim": 9})
        for key, value in (("episode_id", 8), ("cache_fingerprint", "wrong")):
            data = episode(); data[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "ID/cache"):
                MappedEpisodes(self.cache(data)).fetch(7)

    def test_teacher_label_selected_rows_are_validated(self):
        data = episode()
        data["features"][0][-1, 0] += 1
        loss = unittest.mock.Mock(return_value={"loss": 1.0})
        with self.assertRaisesRegex(ValueError, "conditioning tail"):
            _scalar_loss(loss, None, data, 0, torch.zeros(1), 1)
        loss.assert_not_called()

    def test_minimal_toy_cache_fallback_is_preserved(self):
        value = {"episode_id": 3, "fixture": True}
        loader = unittest.mock.Mock(return_value=value)
        mapped = MappedEpisodes(SimpleNamespace(manifest={}, load=loader))
        self.assertIs(mapped.fetch(3), value)
        loader.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()
