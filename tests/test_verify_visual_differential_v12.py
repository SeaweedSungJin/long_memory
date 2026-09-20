"""Tiny actual-head diagnostic harness tests; never load real model or CUDA."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from run_scripts.robomme import verify_visual_differential_v12 as probe
from run_scripts.robomme.visual_differential_memory_v12 import VisualDifferentialConfig, VisualDifferentialMemoryV12
from tests.test_visual_patch_verifier_v11 import episode_fixture


class DifferentialVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, mode):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41)
            return VisualDifferentialMemoryV12(VisualDifferentialConfig(feature_dim=8, hidden_dim=16), read_mode=mode)

    def test_tracked_prefix_accesses_only_past_and_current_observations(self):
        episode = episode_fixture()
        class Prefix:
            def __init__(self, value): self.value = value
            def __getitem__(self, index):
                if not 0 <= index <= 3:
                    raise AssertionError("Future row read")
                return self.value[index]
        observations = {name: Prefix(episode[name]) for name in probe.old.OBSERVATION_KEYS}
        memory = self.model("differential")
        features, bank, earliest, current = probe.tracked_features(memory, observations, 3)
        self.assertEqual(bank.tokens.shape, (1, 3, 162, 16))
        self.assertEqual(bank.content.shape, bank.tokens.shape)
        self.assertTrue(torch.equal(features, episode["features"][3:4]))
        features.float().square().mean().backward()
        self.assertEqual(probe.old.tensor_record(earliest.grad)["nonzero"], 0)
        self.assertGreater(probe.old.tensor_record(current.grad)["nonzero"], 0)
        with self.assertRaises(ValueError):
            probe.tracked_features(memory, observations, 0)

    def run_mock(self, mode, *, broken=False):
        episode = episode_fixture()
        visual = self.model(mode)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            head = nn.Linear(8, 8, bias=False).bfloat16().eval().requires_grad_(False)
            parent = nn.Linear(8, 8).eval().requires_grad_(False)
        head.num_inference_timesteps = 4
        fused = episode["features"][3:4, -4:].float() + .125
        def cached(head, ep, query, short=None):
            features = ep["features"][query][None]
            return (probe.old.replace_short(features, short) if short is not None else features), None, None, None
        def flow(head, ep, query, features, *unused, **kwargs):
            prediction = head(features).float().mean(1)[:, None].expand(-1, 2, -1)
            return {"prediction": prediction, "loss": (prediction - ep["targets"][query][None]).square().mean()}
        def original(head, ep, query, short=None, **kwargs):
            return flow(head, ep, query, cached(head, ep, query, short)[0])
        def generated(head, ep, query, short=None, **kwargs):
            value = head(cached(head, ep, query, short)[0]).float().mean(1)[:, None]
            return value + 1 if broken and isinstance(ep["features"], dict) else value
        report, snapshots = {"optimizer_updates": 0}, []
        with patch.object(probe.old, "cached_inputs", side_effect=cached), \
             patch.object(probe.old, "replay_queries", return_value={3: (fused, {})}), \
             patch.object(probe.old, "expert_episode_flow_loss", side_effect=original), \
             patch.object(probe.old, "flow_at_features", side_effect=flow), \
             patch.object(probe.old, "sample_noise_time", return_value=(torch.zeros(1, 2, 8).bfloat16(), torch.ones(1, 1, 1).bfloat16() * .5)), \
             patch.object(probe.old, "generated_action", side_effect=generated):
            if broken:
                with self.assertRaisesRegex(RuntimeError, "Diagnostic check failed"):
                    probe.diagnose_arm(head, parent, episode, 3, visual, 9111, report, lambda: snapshots.append(copy.deepcopy(report)))
            else:
                probe.diagnose_arm(head, parent, episode, 3, visual, 9111, report, lambda: snapshots.append(copy.deepcopy(report)))
        return report, snapshots

    def test_both_arms_two_updates_nonzero_wake_and_correct_past_gradient_contract(self):
        for mode in probe.MODES:
            with self.subTest(mode=mode):
                report, _ = self.run_mock(mode)
                self.assertEqual(report["optimizer_updates"], 2)
                self.assertTrue(all(report["checks"].values()), report["checks"])
                self.assertEqual([row["after_updates"] for row in report["passes"]], [0, 1, 2])
                past = report["passes"][2]["earliest_image_gradient"]["nonzero"]
                self.assertGreater(past, 0) if mode == "differential" else self.assertEqual(past, 0)

    def test_failed_zero_parity_preserved_without_any_optimizer_update(self):
        report, snapshots = self.run_mock("differential", broken=True)
        self.assertEqual(report["optimizer_updates"], 0)
        self.assertFalse(report["checks"]["zero_generated_euler4_exact"])
        self.assertEqual(len(report["passes"]), 1)
        self.assertTrue(snapshots)


if __name__ == "__main__":
    unittest.main()
