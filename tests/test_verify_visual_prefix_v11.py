"""Tiny CPU checks for the fixed-prefix diagnostic; no real AE or CUDA."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from run_scripts.robomme import verify_visual_prefix_v11 as probe
from run_scripts.robomme.visual_patch_memory_v11 import VisualPatchConfig, VisualPatchMemoryV11


class WorstPrefixDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_two_updates_use_trainer_objective_preserve_frozen_scope_and_wake_oldest_input(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(63)
            visual = VisualPatchMemoryV11(VisualPatchConfig(feature_dim=8, hidden_dim=16))
            modules = [nn.Linear(8, 8).bfloat16().eval().requires_grad_(False) for _ in range(3)]
            features = torch.randn(90, 209, 8).bfloat16()
        head, parent, cvom = modules
        images = torch.zeros(90, 209, dtype=torch.bool)
        images[:, 33:114] = True; images[:, 121:202] = True
        episode = {"features": features, "short": features[:, -4:], "image_masks": images,
            "attention_masks": torch.ones_like(images), "frames": torch.arange(90) * 16,
            "is_demo": torch.arange(90) < 71, "actions": torch.zeros(89, 2, 8),
            "decision_mask": torch.arange(89) >= 71, "state": torch.zeros(90, 8),
            "targets": torch.full((89, 2, 8), .2), "target_mask": torch.ones(89, 2, 8, dtype=torch.bool)}
        original_first = features[0].clone()
        args = SimpleNamespace(seed=9111, visual_learning_rate=1e-4, weight_decay=.01,
                               max_grad_norm=1., checkpoint_encoding=True, activation_checkpointing=True)
        report = {"optimizer_updates": 0, "checks": {}, "steps": []}
        flows, persisted = [], []

        def flow(model, ep, query, *, seed, activation_checkpointing):
            flows.append((query, seed, activation_checkpointing))
            prediction = model(ep["features"][query]).float().mean(0)[None]
            difference = prediction - ep["targets"][query]
            return {"loss": difference.square().mean(), "velocity_mae": difference.abs().mean()}

        with patch.object(probe.trainer, "parent_short", return_value=features[88:89, -4:].float()), \
                patch.object(probe.trainer, "expert_episode_flow_loss", side_effect=flow), \
                contextlib.redirect_stdout(io.StringIO()):
            probe.run_steps(args, visual, parent, head, cvom, episode, report, lambda: persisted.append(True))
        self.assertEqual(flows, [(88, 9111, True)] * 2)
        self.assertEqual(report["optimizer_updates"], 2)
        self.assertEqual([row["after_optimizer_updates"] for row in report["steps"]], [0, 1])
        self.assertTrue(all(report["checks"].values()), report["checks"])
        self.assertTrue(torch.equal(features[0], original_first))
        self.assertIsNone(features.grad)
        self.assertGreaterEqual(len(persisted), 5)
        self.assertEqual(report["frozen_modules_before_sha256"], report["frozen_modules_after_sha256"])
        for row in report["steps"]:
            self.assertEqual(row["metrics"]["visual_bank_tokens"], 14256.)
            self.assertTrue(all(value >= 0 for value in row["timing_seconds"].values()))

    def test_fixed_scope_and_cpu_measurements_do_not_initialize_cuda(self):
        self.assertEqual((probe.EPISODE, probe.QUERY, probe.UPDATES), (1303, 88, 2))
        was_initialized = torch.cuda.is_initialized()
        meter = probe.Measurement("cpu")
        meter.reset()
        self.assertEqual(set(meter.snapshot().values()), {0})
        self.assertEqual(torch.cuda.is_initialized(), was_initialized)


if __name__ == "__main__":
    unittest.main()
