"""CPU-only cache tests: no pretrained downloads, GPUs, or simulator required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from gr00t.long_memory.cache import (
    CACHE_VERSION, CacheConfig, EpisodeCache, _extract_episode,
    decision_frames, split_episodes, validate_episode,
)


def synthetic_episode():
    """One passive event, then one action event, followed by an endpoint."""
    features = [torch.arange(24).reshape(6, 4).float() for _ in range(3)]
    return {
        "episode_id": 7, "task": "fake", "frames": torch.tensor([0, 2, 5]),
        "features": features, "attention_masks": [torch.ones(6, dtype=torch.bool) for _ in range(3)],
        "image_masks": [torch.zeros(6, dtype=torch.bool) for _ in range(3)],
        "moment": torch.ones(3, 2, 4), "short": torch.stack([f[-2:] for f in features]),
        "state": torch.zeros(3, 8), "targets": torch.zeros(2, 4, 8),
        "target_mask": torch.tensor([False, True])[:, None, None].expand(2, 4, 8).clone(),
        "actions": torch.zeros(2, 3, 8),
        "action_mask": torch.tensor([[False, False, False], [True, True, True]]),
        "transition_valid": torch.ones(2, dtype=torch.bool), "decision_mask": torch.tensor([False, True]),
        "is_demo": torch.tensor([True, False, False]), "embodiment_id": 10,
        "cache_fingerprint": "test",
    }


class CacheTimingTests(unittest.TestCase):
    def test_demo_priming_matches_execution_window(self):
        demo = np.arange(141) < 86
        frames = decision_frames(demo, 16).tolist()
        i = frames.index(86)
        self.assertEqual(frames[i - 3:i + 1], [38, 54, 70, 86])
        self.assertIn(6, frames)  # A much older demo cue is retained.
        self.assertEqual(frames[-1], 140)
        self.assertLessEqual(max(np.diff(frames)), 16)

    def test_short_demo_and_no_demo(self):
        self.assertEqual(decision_frames(np.arange(25) < 3, 16).tolist(), [0, 3, 19, 24])
        self.assertEqual(decision_frames(np.zeros(18, dtype=bool), 16).tolist(), [0, 16, 17])

    def test_bad_demo_rejected(self):
        with self.assertRaises(ValueError):
            decision_frames(np.array([True, False, True, False]), 16)
        with self.assertRaises(ValueError):
            decision_frames(np.ones(20, dtype=bool), 16)

    def test_split_is_disjoint_deterministic_and_task_stratified(self):
        episodes = [{"episode_id": i, "task": str(i // 4)} for i in range(12)]
        a = split_episodes(episodes, .25, 42)
        b = split_episodes(list(reversed(episodes)), .25, 42)
        self.assertEqual(a, b)
        self.assertFalse(set(a["train"]) & set(a["val"]))
        self.assertEqual(len(a["train"]) + len(a["val"]), 12)
        self.assertEqual({i // 4 for i in a["val"]}, {0, 1, 2})
        smoke = split_episodes(episodes, .2, 42, 2)
        self.assertEqual(len(smoke["train"]), 1)
        self.assertEqual(smoke["train"][0] // 4, smoke["val"][0] // 4)

    def test_good_contract(self):
        validate_episode(synthetic_episode())

    def test_causal_and_passive_guards(self):
        episode = synthetic_episode()
        episode["action_mask"][0, 0] = True
        with self.assertRaisesRegex(ValueError, "Passive"):
            validate_episode(episode)
        episode = synthetic_episode()
        episode["frames"][-1] = 4
        with self.assertRaisesRegex(ValueError, "beyond"):
            validate_episode(episode)
        episode = synthetic_episode()
        episode["target_mask"][0, 0, 0] = True
        with self.assertRaisesRegex(ValueError, "Passive"):
            validate_episode(episode)

    def test_finite_and_tail_guards(self):
        episode = synthetic_episode()
        episode["state"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            validate_episode(episode)
        episode = synthetic_episode()
        episode["short"] += 1
        with self.assertRaisesRegex(ValueError, "tail"):
            validate_episode(episode)

    def test_cache_file_fingerprint_and_episode_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            episode = synthetic_episode()
            torch.save(episode, path / "7.pt")
            other = copy.deepcopy(episode)
            other["episode_id"] = 8
            torch.save(other, path / "8.pt")
            manifest = {
                "format_version": CACHE_VERSION, "status": "complete", "fingerprint": "test",
                "splits": {"train": [7], "val": [8]},
                "episodes": [{"episode_id": 7, "path": "7.pt"}, {"episode_id": 8, "path": "8.pt"}],
            }
            (path / "manifest.json").write_text(json.dumps(manifest))
            cache = EpisodeCache(path)
            self.assertEqual(cache.load(7)["episode_id"], 7)
            episode["cache_fingerprint"] = "stale"
            torch.save(episode, path / "7.pt")
            with self.assertRaisesRegex(ValueError, "Stale"):
                cache.load(7)
            manifest["splits"]["val"] = [7, 8]
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "split"):
                EpisodeCache(path)

    def test_extractor_keeps_demo_and_masks_padded_future_targets(self):
        """Exercise the actual extraction orchestration with tiny fake frozen ops.

        The data/state/action path is real, but VLM/processor are fakes so this
        can test the temporal contract on CPU without a 3B model or CUDA.
        """
        import pandas as pd

        n = 23
        raw = pd.DataFrame({"is_demo": np.arange(n) < 6})
        table = pd.DataFrame({
            "state.joint": [np.array([i, i], np.float32) for i in range(n)],
            "action.joint": [np.array([i, i], np.float32) for i in range(n)],
            "language.annotation.task": ["remember"] * n,
        })
        modalities = {
            "video": SimpleNamespace(modality_keys=["front"]),
            "state": SimpleNamespace(modality_keys=["joint"], delta_indices=[0]),
            "action": SimpleNamespace(modality_keys=["joint"], delta_indices=list(range(6))),
            "language": SimpleNamespace(modality_keys=["annotation.task"]),
        }

        class Processor:
            modality_configs = {"new_embodiment": modalities}
            embodiment_id_mapping = {"new_embodiment": 10}

            def __call__(self, messages):
                step = messages[0]["content"]
                actions = torch.zeros(8, 4)
                actions[:6, :2] = torch.from_numpy(step.actions["joint"])
                mask = torch.zeros_like(actions, dtype=torch.bool)
                mask[:6, :2] = True
                return {"state": torch.from_numpy(step.states["joint"]),
                        "action": actions, "action_mask": mask,
                        "frame": int(step.images["front"][0][0, 0, 0])}

            def collator(self, samples):
                return {"inputs": samples[0]}

        class Head:
            vlln = torch.nn.Identity()

            def reset_memory(self):
                pass

            def process_backbone_output(self, output, action_inputs_B):
                output["backbone_features"] = output["backbone_features"].clone()
                output["backbone_features"][:, -2:] += 1
                return output

        head = Head()
        model = SimpleNamespace(
            config=SimpleNamespace(memory_stride=4, n_moment_tokens=2), action_head=head,
            prepare_input=lambda batch: (batch, batch),
            backbone=lambda batch: {"backbone_features": torch.full((1, 5, 4), float(batch["frame"]))},
        )
        loader = SimpleNamespace(
            dataset_path=Path("/fake"), data_path_pattern="{episode_index}.parquet", chunk_size=1000,
            _load_parquet_data=lambda episode_id: table,
            _load_video_data=lambda episode_id, frames: {"front": np.stack([np.full((2, 2, 3), i, dtype=np.uint8) for i in frames])},
        )
        config = CacheConfig("/fake", "/fake", "/fake", device="cpu")
        record = {"episode_id": 1, "task": "fake", "metadata": {"length": n}}
        with patch("pandas.read_parquet", return_value=raw):
            result = _extract_episode(model, Processor(), loader, record, config)
        self.assertEqual(result["frames"].tolist(), [0, 2, 6, 10, 14, 18, 22])
        self.assertEqual(result["decision_mask"].tolist(), [False, False, True, True, True, True])
        self.assertEqual(result["actions"][2, :, 0].tolist(), [6, 7, 8, 9])
        self.assertFalse(result["target_mask"][:2].any())
        self.assertEqual(result["target_mask"][-1, :, 0].tolist(), [True] * 5 + [False] * 3)
        self.assertTrue(torch.equal(result["short"], result["moment"] + 1))
        validate_episode(result)


if __name__ == "__main__":
    unittest.main()
