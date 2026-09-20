"""Small CPU tests; no downloaded model, GPU, cache writes or simulator."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from gr00t.long_memory import diagnostic_offline as diagnostic
from gr00t.long_memory.core_v3 import ActionValueMemory, MemoryV3Config


def toy_episode(eid=2):
    g = torch.Generator().manual_seed(15 + eid)
    n = 12
    return {"episode_id": eid, "frames": torch.arange(n + 1) * 2,
            "short": torch.randn(n + 1, 2, 6, generator=g),
            "moment": torch.randn(n + 1, 2, 6, generator=g),
            "state": torch.randn(n + 1, 3, generator=g),
            "actions": torch.randn(n, 2, 4, generator=g),
            "action_mask": torch.ones(n, 2, dtype=torch.bool),
            "transition_valid": torch.ones(n, dtype=torch.bool),
            "decision_mask": torch.ones(n, dtype=torch.bool)}


class OfflineDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(143)
        self.memory = ActionValueMemory(MemoryV3Config(feature_dim=6, state_dim=3, action_dim=4,
            hidden_dim=8, num_heads=2, capacity=5, min_fill=1, max_victims=2, time_scale=2)).eval().requires_grad_(False)
        self.head = nn.Linear(6, 1).eval().requires_grad_(False)
        self.ep = toy_episode()

    def test_query_paired_seeds_and_no_mutation(self):
        calls = []

        def loss(head, ep, decision, fused=None, *, seed):
            calls.append(seed)
            short = ep["short"][decision:decision + 1] if fused is None else fused
            return {"loss": head(short).square().mean() + torch.rand((), generator=torch.Generator().manual_seed(seed))}

        state = {name: value.clone() for name, value in self.memory.state_dict().items()}
        out = diagnostic.query_experiment(self.memory, self.head, self.ep, 10, window=2, noise_samples=2, seed=31, loss_fn=loss)
        self.assertEqual(len(calls), 14)
        self.assertEqual(calls[:7], [31] * 7)
        self.assertEqual(calls[7:], [1040] * 7)
        self.assertEqual(set(out["conditions"]), set(diagnostic.CONDITIONS))
        self.assertEqual(out["conditions"]["full"]["loss_minus_full"], 0)
        self.assertGreater(out["available_old_count"], 0)
        for name, tensor in self.memory.state_dict().items():
            self.assertTrue(torch.equal(state[name], tensor))
        json.dumps(out, allow_nan=False)

    def test_native_stream_parity_and_invalid_prefix(self):
        out = diagnostic.parity_experiment(self.memory, self.ep, 10, stride=2, atol=1e-5)
        self.assertEqual(out["status"], "pass")
        self.assertEqual(len(out["endpoints"]), 11)
        self.ep["transition_valid"][1] = False
        self.assertEqual(diagnostic.parity_experiment(self.memory, self.ep, 10, stride=2, atol=1e-5)["status"], "unsupported")

    def test_incomplete_prefix_unsupported(self):
        self.ep["action_mask"][2, 1] = False
        self.assertEqual(diagnostic.parity_experiment(self.memory, self.ep, 8, stride=2, atol=1e-5)["status"], "unsupported")

    def test_episode_sampling_only_validation_reproducible(self):
        manifest = {"splits": {"train": [0, 1], "val": [2, 3, 4, 5]},
                    "episodes": [{"episode_id": i, "task": "a" if i % 2 == 0 else "b"} for i in range(6)]}
        a = diagnostic._balanced_episodes(manifest, 6, 20)
        self.assertEqual(a, diagnostic._balanced_episodes(manifest, 6, 20))
        self.assertEqual(set(a), {2, 3, 4, 5})
        self.assertNotEqual(a[0] % 2, a[1] % 2)

    def test_report_only_cannot_touch_unrelated_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "manifest.json").write_text('{"variant": "some-original-eval"}')
            (root / "summary.txt").write_text("user result")
            with self.assertRaisesRegex(ValueError, "unrelated"):
                diagnostic.main(["--output-dir", folder, "--report-only"])
            self.assertEqual((root / "summary.txt").read_text(), "user result")
            self.assertFalse((root / ".lock").exists())

    def test_partial_report_empty_not_nan(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            summary = diagnostic.write_report(Path(folder))
            self.assertEqual(summary["query_count"], 0)
            json.dumps(summary, allow_nan=False)
            self.assertIn("NOT robot success", (Path(folder) / "summary.txt").read_text())

    def test_writer_coverage_does_not_invent_full_banks(self):
        rows = [{"episode_id": 2, "candidate": c} for c in (0, 1, 8)]
        selected, coverage = diagnostic.select_writer_contexts(self.memory, lambda _: self.ep, rows, 3, "fifo-stress")
        self.assertEqual(coverage["candidate_pool_counts"], {"empty": 1, "partial": 1, "full": 1})
        self.assertFalse(coverage["natural_deployed_distribution"])
        self.assertEqual([r["fill_stratum"] for r in selected], ["full", "partial", "empty"])


if __name__ == "__main__":
    unittest.main()
