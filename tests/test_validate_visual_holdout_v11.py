"""CPU-only plan-prefix, pairing/statistics and frozen holdout-driver checks."""
import argparse
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from run_scripts.robomme import validate_visual_holdout_v11 as audit


def plan_fixture():
    args = argparse.Namespace(seed=9111, val_samples=32, val_noise_samples=2,
                              max_steps=4, query_batch_size=2)
    cache = SimpleNamespace(path=Path("/unused"), manifest={"splits": {"train": [0, 1], "val": list(range(2, 36))},
                                                        "action_steps": 16, "episodes": []})
    def fetch(eid):
        decisions = torch.tensor([True, eid < 2])
        return {"decision_mask": decisions, "target_mask": decisions[:, None, None].expand(2, 16, 8),
                "action_mask": decisions[:, None].expand(2, 16)}
    episodes = SimpleNamespace(fetch=fetch)
    original, sha = audit.trainer.build_plan(args, cache, episodes)
    return args, cache, episodes, {"sha256": sha, **original}


def statistics_fixture():
    plan = {"validation": [[7, 0], [8, 1], [9, 3]], "validation_schedule": []}
    values = {7: ((0., 2.), (1., 3.)), 8: ((2., 4.), (1., 3.)), 9: ((3., 5.), (3., 5.))}
    records = []
    for eid, query in plan["validation"]:
        for repeat in range(2):
            item = {"episode_id": eid, "decision": query, "repeat": repeat,
                    "flow_seed": eid * 10 + repeat, "generation_seed": eid * 100 + repeat}
            plan["validation_schedule"].append(item)
            for index, role in enumerate(audit.ROLES):
                records.append({**item, "role": role, audit.METRICS[0]: values[eid][index][repeat]})
    return plan, records


class VisualHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_expansion_exact_original_prefix_training_schedule_and_q0_retained(self):
        args, cache, episodes, saved = plan_fixture()
        before = copy.deepcopy(saved)
        expanded, sha, holdout = audit.expanded_holdout(args, cache, episodes, saved)
        self.assertEqual(saved, before)
        self.assertEqual(args.val_samples, 32)
        self.assertEqual(expanded["validation"][:32], saved["validation"])
        self.assertEqual(expanded["validation_schedule"][:64], saved["validation_schedule"])
        self.assertEqual(expanded["schedule"], saved["schedule"])
        self.assertEqual(expanded["files"], saved["files"])
        self.assertEqual(sha, audit.trainer.digest(expanded))
        self.assertEqual(len(holdout["validation"]), 2)
        self.assertEqual(len(holdout["validation_schedule"]), 4)
        self.assertTrue(all(q == 0 for _, q in holdout["validation"]))
        self.assertFalse({eid for eid, _ in holdout["validation"]} & {eid for eid, _ in saved["validation"]})

    def test_reject_changed_saved_seed_plan_noise_or_training_schedule(self):
        args, cache, episodes, saved = plan_fixture()
        changed = copy.deepcopy(args); changed.seed = 9112
        with self.assertRaisesRegex(ValueError, "seed9111"):
            audit.expanded_holdout(changed, cache, episodes, saved)
        for field in ("validation_schedule", "schedule"):
            changed = copy.deepcopy(saved)
            if field == "validation_schedule":
                changed[field][0]["flow_seed"] += 1
            else:
                changed[field][0]["queries"][0]["flow_seed"] += 1
            changed["sha256"] = audit.trainer.digest({k: v for k, v in changed.items() if k != "sha256"})
            with self.assertRaisesRegex(ValueError, "rebuild exactly"):
                audit.expanded_holdout(args, cache, episodes, changed)

    def test_equal_query_means_medians_counts_leave_one_out_and_cluster_bootstrap(self):
        plan, records = statistics_fixture()
        before = torch.get_rng_state().clone()
        result = audit.paired_statistics(records, plan, bootstrap_draws=200)
        again = audit.paired_statistics(records, plan, bootstrap_draws=200)
        self.assertEqual(result, again)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        metric = result["metrics"][audit.METRICS[0]]
        self.assertEqual(metric["equal_query_mean"], {"reader": 8/3, "visual-off": 8/3})
        self.assertEqual(metric["query_median"], {"reader": 3., "visual-off": 2.})
        self.assertEqual((metric["improved"], metric["worse"], metric["tied"]), (1, 1, 1))
        self.assertEqual(metric["mean_visual_minus_off"], 0.)
        self.assertEqual(metric["median_visual_minus_off"], 0.)
        self.assertEqual(metric["leave_one_out_delta_range"], [-.5, .5])
        self.assertEqual(metric["paired_query_bootstrap"]["sampling_unit"], "query/episode with both roles and both noises retained")
        self.assertEqual(result["per_query"][0]["decision"], 0)

    def test_reject_mismatched_missing_duplicate_or_nonfinite_paired_records(self):
        plan, records = statistics_fixture()
        variants = [records[:-1], records + [records[0]]]
        changed = copy.deepcopy(records); changed[0]["generation_seed"] += 1; variants.append(changed)
        changed = copy.deepcopy(records); changed[0][audit.METRICS[0]] = float("nan"); variants.append(changed)
        for rows in variants:
            with self.assertRaises(ValueError):
                audit.paired_statistics(rows, plan, bootstrap_draws=20)

    def test_zero_off_denominator_reports_null_instead_of_nan(self):
        plan, records = statistics_fixture()
        for row in records:
            row[audit.METRICS[0]] = 0.
        metric = audit.paired_statistics(records, plan, bootstrap_draws=20)["metrics"][audit.METRICS[0]]
        self.assertIsNone(metric["relative_mean_gain_percent"])
        self.assertIsNone(metric["paired_query_bootstrap"]["relative_mean_gain_percent_95pct_percentile_interval"])
        self.assertTrue(all(row["relative_mean_gain_percent"] is None for row in metric["leave_one_out"]))

    def test_guard_requires_eval_nograd_all_frozen_and_tracks_versions(self):
        module = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, "no-grad"):
            audit.frozen_guard((module,))
        with torch.no_grad():
            guard = audit.frozen_guard((module,))
            module.weight.add_(1)
            self.assertTrue(any(value._version != version for value, version in guard))
            module.train()
            with self.assertRaises(RuntimeError):
                audit.frozen_guard((module,))
            module.eval().requires_grad_(True)
            with self.assertRaises(RuntimeError):
                audit.frozen_guard((module,))

    def test_preflight_main_never_loads_head_or_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not_created"
            payload = (None, None, None, None, None, None,
                       {"validation": [[1, 0]], "validation_schedule": [{}, {}]},
                       {"audit_id": "abc", "q0_retained": 1, "expanded_plan_sha256": "def"})
            with patch.object(audit, "preflight", return_value=payload), patch.object(audit, "run_validation") as run:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(audit.main(["--training-run", directory, "--output-dir", str(output), "--preflight-only"]), 0)
                run.assert_not_called()
            self.assertFalse(output.exists())

    def test_actual_run_refuses_nonisolated_gpu_without_preflight(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}), patch.object(audit, "preflight") as preflight:
            with self.assertRaisesRegex(ValueError, "GPU1"):
                audit.main(["--training-run", "/unused", "--output-dir", "/unused"])
            preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
