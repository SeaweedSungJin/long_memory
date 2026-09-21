"""Causal target, frozen READ, masking and matched decoder checks on CPU."""
import copy
import unittest

import torch

from run_scripts.robomme import diagnose_v19_subgoal as probe


def fixture():
    config = probe.RepresentationConfigV18(feature_dim=8, state_dim=3, hidden_dim=8,
        num_heads=2, capacity_events=3, num_short_tokens=4)
    torch.manual_seed(22)
    core = probe.RepresentationMemoryV18(config).eval().requires_grad_(False)
    with torch.no_grad():
        core.memory.fusion_projection.weight.normal_(std=.1)
    episode = {"short": torch.randn(7, 4, 8), "state": torch.randn(7, 3),
        "frames": torch.tensor([0, 16, 32, 48, 64, 80, 95]),
        "is_demo": torch.tensor([True, True, True, False, False, False, False]),
        "decision_mask": torch.tensor([False, False, False, True, True, True]),
        "action_mask": torch.ones(6, 16, dtype=torch.bool),
        "target_mask": torch.ones(6, 50, 3, dtype=torch.bool)}
    raw = {"episode_index": [7] * 96, "frame_index": list(range(96)),
        "is_demo": [True] * 48 + [False] * 48,
        "simple_subgoal_online": ["move right"] * 96}
    raw["simple_subgoal_online"][49] = "move left"
    episode["action_mask"][5, 15:] = False
    return core, episode, raw


class SubgoalProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_current_target_not_future_or_full_chunk_label(self):
        _, ep, raw = fixture()
        rows = probe.align_targets(raw, ep, 7, "train")
        self.assertEqual([r["query_frame"] for r in rows], [48, 64, 80])
        self.assertEqual([r["label"] for r in rows], ["move right"] * 3)
        broken = copy.deepcopy(raw)
        broken["frame_index"][48] = 49
        with self.assertRaisesRegex(ValueError, "identity"):
            probe.align_targets(broken, ep, 7, "train")

    def test_future_poison_and_extra_labels_cannot_change_features(self):
        core, ep, _ = fixture()
        maps = probe.FrozenProjections(core.config)
        before = probe.tensor_hash(core.state_dict())
        expected, audit = probe.extract_stages(core, ep, 3, maps)
        poisoned = copy.deepcopy(ep)
        poisoned["short"][4:] = float("nan")
        poisoned["state"][4:] = float("nan")
        poisoned["frames"][4:] = -1
        poisoned["simple_subgoal_online"] = object()
        poisoned["targets"] = object()
        actual, _ = probe.extract_stages(core, poisoned, 3, maps)
        for stage in probe.STAGES:
            self.assertTrue(torch.equal(expected[stage], actual[stage]), stage)
        self.assertTrue(audit["p2_p3_equal"])
        self.assertEqual(before, probe.tensor_hash(core.state_dict()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in core.parameters()))
        _, evicted = probe.extract_stages(core, ep, 5, maps)
        self.assertEqual(evicted["fifo_evicted"], 2)
        self.assertFalse(evicted["p2_p3_equal"])

    def test_actual_replay_and_shared_natural_projection(self):
        core, ep, _ = fixture()
        maps = probe.FrozenProjections(core.config)
        values, _ = probe.extract_stages(core, ep, 4, maps)
        with torch.no_grad():
            fused = core.replay(ep, 4)["fused"][0]
        torch.testing.assert_close(values["P5"][:, :64], maps("feature", fused), rtol=0, atol=0)
        self.assertTrue(torch.equal(values["P0"][:, 64:], values["P5"][:, 64:]))
        self.assertEqual(int(torch.count_nonzero(values["Ptime"][:, :80])), 0)
        self.assertGreater(int(torch.count_nonzero(values["Ptime"][:, 80:91])), 0)
        for stage in ("P0", "P4", "P5"):
            self.assertEqual(int(torch.count_nonzero(values[stage][:, 80:91])), 0)
        for stage in ("P1", "P2", "P3"):
            self.assertEqual(int(torch.count_nonzero(values[stage][:4, 80:91])), 0)

    def test_train_only_normalization_and_masked_padding(self):
        rows = [{"split": "train"}, {"split": "val"}]
        train, val = torch.randn(4, 97), torch.randn(8, 97)
        x, mask, mean, scale = probe.standardize_and_pad([train, val], rows)
        _, _, other_mean, other_scale = probe.standardize_and_pad([train, val * 1000], rows)
        self.assertTrue(torch.equal(mean, other_mean))
        self.assertTrue(torch.equal(scale, other_scale))
        model = probe.Decoder(97, 8)
        original = model(x, mask)
        x[~mask] = float("nan")
        torch.testing.assert_close(original, model(x, mask), rtol=0, atol=0)

    def test_fixed_batches_capacity_and_frozen_feature_fit(self):
        rows = [{"episode_id": i // 2, "decision": i % 2, "query_frame": i,
                 "split": "train" if i < 6 else "val", "class_id": i % 2, "label": str(i % 2)} for i in range(8)]
        feature = [torch.randn(4 + (i % 2) * 4, 97) for i in range(8)]
        values = {stage: [x.clone() for x in feature] for stage in probe.STAGES}
        states, predictions, reports, _ = probe.fit_decoders(values, rows, 2, steps=2, seeds=(14,), batch_size=2)
        reference = states["P0/seed14"]["model"]
        for stage in probe.STAGES:
            for key, tensor in reference.items():
                self.assertTrue(torch.equal(tensor, states[f"{stage}/seed14"]["model"][key]))
        self.assertEqual(len({r["parameters"] for r in reports.values()}), 1)
        summary = probe.summarize(predictions, bootstrap_samples=20)
        self.assertEqual(summary["paired_val_contrasts"]["P5_minus_P0"]["accuracy_gain"], 0.)


if __name__ == "__main__":
    unittest.main()
