"""Tiny CPU verifier checks; these never load the actual Action Expert."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from run_scripts.robomme import verify_visual_patch_v11 as probe
from run_scripts.robomme.visual_patch_memory_v11 import VisualPatchConfig, VisualPatchMemoryV11


def episode_fixture():
    generator = torch.Generator().manual_seed(55)
    features = torch.randn(5, 236, 8, generator=generator).bfloat16()
    images = torch.zeros(5, 236, dtype=torch.bool)
    images[:, 10:91] = True
    images[:, 98:179] = True
    return {"features": features, "image_masks": images, "attention_masks": torch.ones_like(images),
            "frames": torch.arange(5) * 16, "is_demo": torch.tensor([True, True, False, False, False]),
            "targets": torch.full((5, 2, 8), .2), "target_mask": torch.ones(5, 2, 8, dtype=torch.bool),
            "state": torch.zeros(5, 8), "embodiment_id": 0}


class VisualPatchVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def memory(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41)
            return VisualPatchMemoryV11(VisualPatchConfig(feature_dim=8, hidden_dim=16))

    def test_full_past_only_and_no_target_or_future_access(self):
        episode = episode_fixture()
        class Prefix:
            def __init__(self, value): self.value = value
            def __getitem__(self, index):
                if not 0 <= index <= 3:
                    raise AssertionError("Future access")
                return self.value[index]
        class Observations(dict):
            def __getitem__(self, key):
                if key not in probe.OBSERVATION_KEYS:
                    raise AssertionError("Non-observation access")
                return super().__getitem__(key)
        observations = Observations({key: Prefix(episode[key]) for key in probe.OBSERVATION_KEYS})
        memory = self.memory()
        features, bank, earliest = probe.visual_features(memory, observations, 3, track_earliest=True)
        self.assertEqual(bank.tokens.shape, (1, 3, 162, 16))
        self.assertEqual(bank.frames.tolist(), [[0, 16, 32]])
        self.assertTrue(torch.equal(features, episode["features"][3:4]))
        features.float().square().mean().backward()
        self.assertEqual(probe.tensor_record(earliest.grad)["nonzero"], 0)
        self.assertGreater(memory.output_projection.weight.grad.count_nonzero(), 0)
        self.assertEqual(memory.image_projection.weight.grad.count_nonzero(), 0)

    def test_gradient_records_distinguish_absent_zero_tiny_and_nonfinite(self):
        self.assertFalse(probe.tensor_record(None)["present"])
        self.assertEqual(probe.tensor_record(torch.zeros(2))["norm"], 0.)
        tiny = probe.tensor_record(torch.tensor([1e-30]))
        self.assertGreater(tiny["norm"], 0.)
        self.assertEqual(tiny["nonzero"], 1)
        bad = probe.tensor_record(torch.tensor([float("nan")]))
        self.assertFalse(bad["finite"])
        self.assertIsNone(bad["norm"])

    def run_toy(self, *, break_generated_parity=False):
        episode = episode_fixture()
        visual = self.memory()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            head = nn.Linear(8, 8, bias=False).bfloat16().eval().requires_grad_(False)
            parent = nn.Linear(8, 8).eval().requires_grad_(False)
        head.num_inference_timesteps = 4
        fused = episode["features"][3:4, -4:].float() + .125
        def cached(head, ep, query, short=None):
            features = ep["features"][query][None]
            if short is not None:
                features = probe.replace_short(features, short)
            return features, None, None, None
        def flow(head, ep, query, features, *unused, **kwargs):
            prediction = head(features).float().mean(1)[:, None].expand(-1, 2, -1)
            return {"prediction": prediction, "loss": (prediction - ep["targets"][query][None]).square().mean()}
        def original(head, ep, query, short=None, **kwargs):
            return flow(head, ep, query, cached(head, ep, query, short)[0])
        def generated(head, ep, query, short=None, **kwargs):
            result = head(cached(head, ep, query, short)[0]).float().mean(1)[:, None]
            # An intentional numerical mismatch must be reported without
            # hiding the later gradient records or adding optimizer updates.
            if break_generated_parity and isinstance(ep["features"], dict):
                result = result + 1
            return result
        report = {"optimizer_updates": 0, "checks": {}, "gradient_passes": []}
        snapshots = []
        with patch.object(probe, "cached_inputs", side_effect=cached), \
             patch.object(probe, "replay_queries", return_value={3: (fused, {"replayed_observations": torch.tensor(3.)})}), \
             patch.object(probe, "expert_episode_flow_loss", side_effect=original), \
             patch.object(probe, "flow_at_features", side_effect=flow), \
             patch.object(probe, "sample_noise_time", return_value=(torch.zeros(1, 2, 8).bfloat16(), torch.ones(1, 1, 1).bfloat16() * .5)), \
             patch.object(probe, "generated_action", side_effect=generated):
            probe.run_diagnostic(head, parent, episode, 3, visual, 9111, report,
                                 lambda: snapshots.append(copy.deepcopy(report)))
        return report, snapshots

    def test_exactly_two_transient_updates_rebuild_bank_wake_encoder_and_preserve_parent(self):
        report, snapshots = self.run_toy()
        self.assertEqual(report["optimizer_updates"], 2)
        self.assertEqual([row["after_optimizer_updates"] for row in report["gradient_passes"]], [0, 1, 2])
        self.assertTrue(all(report["checks"].values()), report["checks"])
        self.assertGreaterEqual(len(snapshots), 6)
        self.assertEqual(report["frozen_before_sha256"], report["frozen_after_sha256"])
        self.assertEqual(set(report["optimizer_parameter_names"]), set(report["gradient_passes"][0]["parameter_gradients"]))
        self.assertNotIn("parent", " ".join(report["optimizer_parameter_names"]))

    def test_failed_parity_is_retained_with_all_gradient_records(self):
        report, snapshots = self.run_toy(break_generated_parity=True)
        self.assertFalse(report["checks"]["zero_generated_euler4_exact"])
        self.assertEqual(len(report["gradient_passes"]), 3)
        self.assertEqual(report["optimizer_updates"], 2)
        self.assertFalse(snapshots[-1]["checks"]["zero_generated_euler4_exact"])
        self.assertTrue(report["checks"]["frozen_parent_and_head_unchanged"])


if __name__ == "__main__":
    unittest.main()
