"""Probe causality and evaluation masks, using small CPU features only."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gr00t.long_memory.recall_probe_v5 import (REPRESENTATIONS, fit_probes, frozen_probe_vectors,
    main, masked_probe_loss, probe_metrics, select_probe_plan, train_standardization)
from tests.test_long_memory_v3_core import episode, model


def probe_episode():
    data = episode(8)
    data["decision_mask"] = torch.ones(8, dtype=torch.bool)
    data["features"] = [torch.cat((torch.randn(3, 6), short)) for short in data["short"]]
    data["image_masks"] = [torch.tensor([True, True, True, False, False]) for _ in data["short"]]
    data["attention_masks"] = [torch.ones(5, dtype=torch.bool) for _ in data["short"]]
    return data


class TestRecallProbe(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_past_is_strictly_older_and_future_changes_are_inert(self):
        memory, ep = model().eval().requires_grad_(False), probe_episode()
        a, meta = frozen_probe_vectors(memory, ep, 5, window=4)
        self.assertEqual(meta["old_ids"], [0])  # event0 ends2 < shortboundary4
        modified = copy.deepcopy(ep)
        for name in ("short", "moment", "state"):
            modified[name][6:] += 1000
        modified["actions"][5:] += 1000
        for i in range(6, 9):
            modified["features"][i] += 1000
        b, other = frozen_probe_vectors(memory, modified, 5, window=4)
        for name in REPRESENTATIONS:
            torch.testing.assert_close(a[name], b[name])
        self.assertEqual(meta, other)
        self.assertTrue(all(p.grad is None for p in memory.parameters()))

    def test_no_old_is_zero_not_current_content(self):
        memory, ep = model().eval(), probe_episode()
        values, info = frozen_probe_vectors(memory, ep, 2, window=4)
        self.assertFalse(info["available_old"])
        for name in ("past_moment", "past_image", "past_event"):
            self.assertEqual(float(values[name].abs().sum()), 0.0)
        self.assertGreater(float(values["current_short"].abs().sum()), 0.0)

    def test_masked_images_not_short_memory_tail(self):
        memory, ep = model().eval(), probe_episode()
        for mask in ep["image_masks"]:
            mask[:] = False
            mask[-2:] = True
        values, info = frozen_probe_vectors(memory, ep, 5, window=4)
        self.assertFalse(info["past_image_available"])
        self.assertEqual(float(values["past_image"].abs().sum()), 0.0)

    def test_standardizer_is_train_only(self):
        feature = torch.tensor([[1., 2.], [3., 4.], [1e6, -1e6]])
        x, mean, scale = train_standardization(feature, torch.tensor([True, True, False]))
        torch.testing.assert_close(mean, torch.tensor([2., 3.]))
        torch.testing.assert_close(scale, torch.ones(2))

    def test_unknown_labels_and_empty_grounding_do_not_train(self):
        logits = torch.randn(2, 3, requires_grad=True)
        xy = torch.randn(2, 2, requires_grad=True)
        loss = masked_probe_loss(logits, xy, torch.tensor([-1, -1]), torch.zeros(2, 2),
                                 torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_fit_and_heldout_metrics_do_not_claim_success(self):
        rows = [{"split": "train" if i < 8 else "val", "class_id": i % 2,
                 "class_valid": True, "xy_valid": i % 2 == 0, "xy": [.2, .8],
                 "available_old": i % 2 == 0} for i in range(12)]
        features = {name: torch.randn(12, 6 if name in ("current_short", "past_moment", "past_image") else 8)
                    for name in REPRESENTATIONS}
        report, states = fit_probes({"features": features, "rows": rows}, 2, epochs=2, hidden=4, batch_size=4)
        for name in REPRESENTATIONS:
            self.assertEqual(report[name]["metrics"]["val/all"]["class_targets"], 4)
            self.assertEqual(report[name]["metrics"]["val/old_available"]["class_targets"], 2)
            self.assertEqual(len(report[name]["history"]), 2)
            self.assertIn("feature_mean", states[name])

    def test_probe_output_cannot_contaminate_any_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = SimpleNamespace(path=root / "cache", manifest={"dataset_path": str(root / "raw"),
                                     "model_path": str(root / "base")})
            labels = SimpleNamespace(root=root / "labels")
            checkpoint = root / "checkpoint"
            for source in (cache.path, labels.root, checkpoint, root / "raw", root / "base"):
                destination = source / "new_probe"
                with patch("gr00t.long_memory.recall_probe_v5.EpisodeCache", return_value=cache), \
                     patch("gr00t.long_memory.recall_probe_v5.RecallLabels", return_value=labels), \
                     self.subTest(source=source), self.assertRaisesRegex(ValueError, "inside"):
                    main(["--cache-dir", str(cache.path), "--labels-dir", str(labels.root),
                          "--checkpoint", str(checkpoint), "--output-dir", str(destination)])
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
