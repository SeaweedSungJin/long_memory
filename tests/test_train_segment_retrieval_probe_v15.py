"""CPU-only tests for the bounded probe sampler, statistics, and actual update."""
import copy
import unittest

import numpy as np
import torch

from run_scripts.robomme import train_segment_retrieval_probe_v15 as trainer
from run_scripts.robomme.segment_retrieval_probe_v15 import (
    configure_qk_only, assert_qk_scope, snapshot_frozen_parameters,
)
from tests.test_segment_retrieval_probe_v15 import memory, problem


def examples():
    return [{"episode_id": eid, "decision": q, "split": "train" if eid < 3 else "val"}
            for eid in range(5) for q in range(eid + 1)]


class TinyInputs:
    def get(self, model, row):
        current, bank = problem(model, batch=1)
        target = torch.tensor([[True, False, False]])
        return current, bank, target, {}


def validation_rows(episodes=12, loss=1.):
    return [{"episode_id": eid, "decision": q, "mode": mode, "loss": loss,
             "uniform_all/loss": 2., "uniform_demo/loss": 1.8, "time_only/loss": 1.6}
            for eid in range(episodes) for q in range(2) for mode in ("correct", "content_permuted")]


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_plan_is_label_independent_train_only_and_episode_balanced(self):
        rows = examples()
        first = trainer.make_plan(rows, 30, 4, 9151)
        altered = [dict(r, label="DO NOT READ", positive_frames=[999]) for r in rows]
        self.assertEqual(first, trainer.make_plan(altered, 30, 4, 9151))
        visits = [r for batch in first["schedule"] for r in batch]
        self.assertEqual({r["episode_id"] for r in visits}, {0, 1, 2})
        self.assertEqual([sum(r["episode_id"] == eid for r in visits) for eid in range(3)], [40, 40, 40])
        self.assertNotEqual(first, trainer.make_plan(rows, 30, 4, 9152))
        self.assertEqual(rows, examples())

    def test_plan_rejects_duplicates_missing_train_and_invalid_options(self):
        with self.assertRaises(ValueError):
            trainer.make_plan(examples() + [examples()[0]], 3, 2, 0)
        with self.assertRaises(ValueError):
            trainer.make_plan([r for r in examples() if r["split"] == "val"], 3, 2, 0)
        with self.assertRaises(ValueError):
            trainer.make_plan(examples(), 0, 2, 0)

    def test_causal_time_controls_use_no_future_duration_or_label(self):
        frames = torch.tensor([0, 16, 32, 47, 48, 64])
        demos = frames < 48
        controls = trainer.control_probabilities(frames, demos, 80, 48)
        self.assertEqual(set(controls), set(trainer.CONTROL_NAMES))
        for p in controls.values():
            torch.testing.assert_close(p.sum(), torch.tensor(1.))
        self.assertEqual(int(controls["time_only"].argmax()), 2)
        self.assertTrue(torch.equal(controls["uniform_demo"][4:], torch.zeros(2)))
        with self.assertRaises(ValueError):
            trainer.control_probabilities(frames, demos, 64, 48)
        with self.assertRaises(ValueError):
            trainer.control_probabilities(frames, ~demos, 80, 48)

    def test_distribution_and_episode_macro_not_query_weighted(self):
        metrics = trainer.distribution_metrics(torch.tensor([.1, .3, .6]), torch.tensor([True, True, False]))
        self.assertAlmostEqual(metrics["positive_mass"], .4, places=6)
        self.assertEqual(metrics["span_hit"], 0.)
        rows = [{"episode_id": 1, "loss": 0.}] * 20 + [{"episode_id": 2, "loss": 2.}]
        macro = trainer.episode_macro(rows, "loss")
        self.assertEqual(np.mean(list(macro.values())), 1.)
        with self.assertRaises(ValueError):
            trainer.distribution_metrics(torch.tensor([.1, .3]), torch.tensor([True, False]))
        with self.assertRaises(FloatingPointError):
            trainer.distribution_metrics(torch.tensor([0., 1.]), torch.tensor([True, False]))

    def test_assessment_is_fixed_paired_and_not_robot_success(self):
        original, final = validation_rows(loss=1.5), validation_rows(loss=1.)
        result = trainer.assess(original, final)
        self.assertEqual(result["decision"], "go_to_action_probe")
        self.assertFalse(result["policy_ready"])
        self.assertFalse(result["goal_30_percent_achieved"])
        self.assertEqual(result, trainer.assess(original, final))
        altered = copy.deepcopy(final)
        altered[0]["decision"] = 99
        with self.assertRaises(ValueError):
            trainer.assess(original, altered)
        with self.assertRaises(ValueError):
            trainer.assess(original, final + [final[0]])

    def test_insufficient_coverage_cannot_pass_even_with_good_loss(self):
        self.assertEqual(trainer.assess(validation_rows(9, 1.5), validation_rows(9, .01))["decision"],
                         "inconclusive_coverage")
        bad = validation_rows(loss=1.)
        for row in bad:
            if row["mode"] == "content_permuted":
                row["loss"] = 2.
        self.assertEqual(trainer.assess(validation_rows(loss=1.5), bad)["decision"], "no_go")

    def test_actual_two_updates_change_only_qk_without_action_expert(self):
        model = memory()
        configure_qk_only(model)
        frozen = snapshot_frozen_parameters(model)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.)
        for _ in range(2):
            metrics = trainer.update(model, optimizer, [{}, {}], TinyInputs(), 1.)
            self.assertTrue(np.isfinite(metrics["loss"]))
            assert_qk_scope(model, frozen, require_gradients=True)
        changed = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
        self.assertEqual(set(changed), {"query_projection.weight", "key_projection.weight"})
        self.assertEqual(len(optimizer.state), 2)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
