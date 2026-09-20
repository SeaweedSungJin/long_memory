"""Validation aggregation and RoboMME joint/gripper masking CPU tests."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from run_scripts.robomme import validation_v19 as validation


class ValidationV19Tests(unittest.TestCase):
    def records(self):
        rows = []
        # Three query records for a long task versus one for a short task.
        # Macro mean=(1+9)/2=5, while incorrectly pooling queries gives3.
        for role in validation.ROLES:
            for index, (task, loss) in enumerate((("long", 1.), ("long", 1.), ("long", 1.), ("short", 9.))):
                rows.append(dict(role=role, task=task, episode_id=index, decision=0, repeat=0,
                    flow_seed=index, generation_seed=index+10, action_loss=loss,
                    generated_prefix_mse=loss, loss=loss))
        return rows

    def test_equal_task_macro_not_query_mean_and_concentration(self):
        summary, by_task = validation.aggregate_v19(self.records())
        self.assertEqual(summary["reader"]["action_loss"], 5.)
        self.assertEqual(summary["reader"]["loss"], 5.)
        self.assertEqual(by_task["reader"]["long"]["action_loss"], 1.)
        self.assertEqual(by_task["reader"]["short"]["action_loss"], 9.)
        self.assertEqual(summary["reader"]["generated_prefix_mse_median"], 1.)
        self.assertAlmostEqual(summary["reader"]["generated_mse_top2_share"], 10/12)
        self.assertEqual(summary["reader"]["memory_gain"], 0.)
        self.assertEqual(summary["reader"]["validation_task_count"], 2)

    def test_reject_unmatched_roles_and_nonfinite_metrics(self):
        rows = self.records()
        with self.assertRaisesRegex(ValueError, "matched"):
            validation.aggregate_v19(rows[1:])
        rows[-1]["generated_prefix_mse"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validation.aggregate_v19(rows)

    def test_generated_joint_gripper_partial_observation_and_padding(self):
        target = torch.zeros(1, 4, 10)
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[..., :8] = True
        target[..., 8:] = float("nan")
        prediction = torch.full((1, 4, 10), 999.)
        prediction[:, 0, :7] = 2.
        prediction[:, 0, 7] = 4.
        # Second nominal-prefix action is not observed: excluded from observed
        # metrics but included in the explicitly labeled nominal diagnostic.
        ep = {"targets": target, "target_mask": mask, "action_mask": torch.tensor([[True, False]])}
        metrics = validation.generated_metrics_v19({"prediction": prediction}, ep, 0, 2)
        self.assertEqual(metrics["generated_executed_joint_mse"], 4.)
        self.assertEqual(metrics["generated_executed_joint_mae"], 2.)
        self.assertEqual(metrics["generated_executed_gripper_mse"], 16.)
        self.assertEqual(metrics["generated_executed_gripper_mae"], 4.)
        self.assertEqual(metrics["generated_prefix_mse"], (7*4+16)/8)
        self.assertEqual(metrics["generated_prefix_valid_values"], 8)
        self.assertEqual(metrics["generated_nominal_prefix_valid_values"], 16)
        self.assertGreater(metrics["generated_nominal_prefix_mse"], metrics["generated_prefix_mse"])

    def test_three_role_validation_baseline_cache_and_unweighted_flow(self):
        target = torch.zeros(1, 20, 8)
        ep = {"targets": target, "target_mask": torch.ones_like(target, dtype=torch.bool),
              "action_mask": torch.ones(1, 16, dtype=torch.bool)}
        episodes = SimpleNamespace(fetch=lambda eid: ep)
        fused, short = torch.ones(1), torch.zeros(1)
        core = SimpleNamespace(replay=lambda ep, d: {"fused": fused, "short": short})
        plan = {"validation_schedule": [dict(task="BinFill", episode_id=0, decision=0, flow_seed=31,
                                             generation_seed=77, repeat=0)]}
        baseline_cache = {}
        def flow(head, ep, d, conditioning, **kwargs):
            self.assertEqual(kwargs["tail_weight"], 1.)
            self.assertEqual(kwargs["action_steps"], 16)
            value = 2. if conditioning is fused else 3.
            return dict(loss=torch.tensor(100.), original_flow_loss=torch.tensor(value),
                        prediction=torch.zeros_like(target), velocity_mae=torch.tensor(.5))
        with patch.object(validation, "validate_decision"), \
             patch.object(validation, "adapter_disabled", side_effect=lambda head: nullcontext()) as disabled, \
             patch.object(validation, "episode_flow_v19", side_effect=flow) as flow_call, \
             patch.object(validation, "generated_prefix_objective", return_value={"prediction": torch.zeros_like(target)}) as gen:
            first = validation.validate_v19(SimpleNamespace(), core, object(), episodes, plan, baseline_cache)
            second = validation.validate_v19(SimpleNamespace(), core, object(), episodes, plan, baseline_cache)
        self.assertEqual(flow_call.call_count, 5)
        self.assertEqual(gen.call_count, 5)
        self.assertEqual(disabled.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["reader"]["action_loss"], 2.)
        self.assertEqual(first[0]["reader"]["memory_gain"], 1.)
        self.assertEqual(first[2]["reader"]["BinFill"]["action_loss"], 2.)
        self.assertEqual(len(first[1]), 3)


if __name__ == "__main__":
    unittest.main()
