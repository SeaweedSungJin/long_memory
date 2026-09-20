"""CPU-only checks for exact TRAIN coverage and independently held-out VAL."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from run_scripts.robomme import training_plan_v19 as module


class TrainingPlanV19Tests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = SimpleNamespace(epochs=2, query_batch_size=4, seed=9191,
            val_per_task=4, val_noise_samples=2, task_weighting="macro", expected_tasks=("alpha", "beta"))
        self.data, records, splits = {}, [], {"train": [], "val": []}
        # Six alpha vs nine beta TRAIN queries: intentionally unbalanced and
        # not divisible by four.  IDs carry no information about the task.
        settings = [(103, "alpha", "train", 6), (9, "beta", "train", 9)]
        settings += [(200 + i, task, "val", 9) for i, task in enumerate(["alpha", "beta"] * 4)]
        for eid, task, split, n in settings:
            mask = torch.tensor([False] + [True] * n)
            action_mask = mask[:, None].expand(-1, 16).clone()
            target_mask = mask[:, None, None].expand(-1, 20, 3).clone()
            self.data[eid] = {"episode_id": eid, "task": "instruction for " + task,
                "decision_mask": mask, "action_mask": action_mask, "target_mask": target_mask,
                "frames": torch.arange(n + 2) * 16,
                "is_demo": torch.tensor([True] + [False] * (n + 1))}
            records.append({"episode_id": eid, "task": "instruction for " + task,
                            "task_group": task, "split": split})
            splits[split].append(eid)
        self.cache = SimpleNamespace(path=Path(self.temp.name), manifest={"episodes": records,
            "splits": splits, "action_steps": 16, "fingerprint": "synthetic"})
        self.episodes = SimpleNamespace(fetch=lambda eid: self.data[eid])

    def build(self):
        return module.build_plan_v19(self.args, self.cache, self.episodes)

    def test_exact_full_epochs_partial_batch_and_no_max_steps_truncation(self):
        self.args.max_steps = 1  # An obsolete argument must NEVER truncate epochs.
        plan, _ = self.build()
        self.assertEqual(plan["train_query_count"], 15)
        self.assertEqual(plan["total_query_presentations"], 30)
        self.assertEqual(plan["total_steps"], 8)
        self.assertEqual([w["query_count"] for w in plan["windows"]], [4, 4, 4, 3] * 2)
        self.assertEqual(plan["epoch_boundaries"], [
            {"epoch": 1, "step": 4, "processed_queries": 15},
            {"epoch": 2, "step": 8, "processed_queries": 30}])
        expected = Counter((eid, d) for eid, ds in plan["train"] for d in ds)
        for epoch in range(2):
            actual = Counter((q["episode_id"], q["decision"]) for s in plan["schedule"]
                             if s["epoch"] == epoch for q in s["queries"])
            self.assertEqual(actual, expected)
        self.assertEqual(plan["coverage"]["eligible_train_episodes"], 2)
        self.assertEqual(plan["task_statistics"]["alpha"]["train"]["demo_endpoints"], 1)

    def test_macro_weights_exactly_equal_task_mass_per_epoch(self):
        plan, _ = self.build()
        mass = Counter()
        for batch in plan["schedule"][:4]:
            for q in batch["queries"]:
                mass[q["task"]] += q["task_weight"]
        self.assertAlmostEqual(mass["alpha"], 7.5)
        self.assertAlmostEqual(mass["beta"], 7.5)
        self.assertAlmostEqual(sum(mass.values()), 15)
        self.args.task_weighting = "query"
        query_plan, _ = self.build()
        self.assertTrue(all(q["task_weight"] == 1 for s in query_plan["schedule"] for q in s["queries"]))

    def test_validation_balanced_disjoint_distinct_episodes_and_temporal_thirds(self):
        plan, _ = self.build()
        val = plan["validation_metadata"]
        self.assertEqual(len(val), 8)
        self.assertEqual(len(plan["validation_schedule"]), 16)
        self.assertEqual(Counter(x["task"] for x in val), {"alpha": 4, "beta": 4})
        self.assertEqual(len({x["episode_id"] for x in val}), 8)
        self.assertFalse({x["episode_id"] for x in val} & set(self.cache.manifest["splits"]["train"]))
        for task in ("alpha", "beta"):
            rows = [x for x in val if x["task"] == task]
            self.assertEqual({r["phase"] for r in rows}, {"early", "middle", "late"})
            for row in rows:
                low, high = {"early": (1, 3), "middle": (4, 6), "late": (7, 9)}[row["phase"]]
                self.assertTrue(low <= row["decision"] <= high)
        pairs = defaultdict_seeds(plan["validation_schedule"])
        self.assertTrue(all(len(values) == 2 for values in pairs.values()))

    def test_plan_identical_between_objective_arms_and_reproducible(self):
        self.args.tail_weight = 1.0
        left, left_sha = self.build()
        self.args.tail_weight = .25
        self.args.output_dir = "a different output"
        right, right_sha = self.build()
        self.assertEqual(left, right)
        self.assertEqual(left_sha, right_sha)
        self.assertEqual(module.digest(json.loads(json.dumps(left))), left_sha)
        self.args.seed += 1
        different, other_sha = self.build()
        self.assertNotEqual(left_sha, other_sha)
        self.assertNotEqual(left["schedule"], different["schedule"])

    def test_task_names_not_derived_from_episode_number(self):
        plan, _ = self.build()
        self.assertEqual(plan["episode_tasks"]["103"], "alpha")
        self.assertEqual(plan["episode_tasks"]["9"], "beta")
        self.assertEqual(plan["task_mapping"]["method"], "explicit_episode_metadata")
        self.cache.manifest["episodes"][0]["env_id"] = "beta"
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.build()

    def test_unknown_instruction_only_cache_rejected_instead_of_eid_guess(self):
        for record in self.cache.manifest["episodes"]:
            record.pop("task_group")
        with self.assertRaisesRegex(ValueError, "cannot infer task families"):
            self.build()

    def test_duplicate_overlap_record_mismatch_rejected(self):
        original = deepcopy(self.cache.manifest)
        self.cache.manifest["splits"]["train"].append(103)
        with self.assertRaisesRegex(ValueError, "unique"):
            self.build()
        self.cache.manifest = deepcopy(original)
        self.cache.manifest["splits"]["val"].append(103)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.build()
        self.cache.manifest = deepcopy(original)
        self.cache.manifest["episodes"][0]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "split disagrees"):
            self.build()

    def test_missing_task_or_insufficient_validation_fails(self):
        self.args.expected_tasks = ("alpha", "beta", "gamma")
        with self.assertRaisesRegex(ValueError, "required task gamma"):
            self.build()
        self.args.expected_tasks = ("alpha", "beta")
        self.args.val_per_task = 5
        with self.assertRaisesRegex(ValueError, "Insufficient distinct"):
            self.build()

    def test_passive_supervision_and_empty_executed_prefix_fail(self):
        self.data[103]["target_mask"][0, 0, 0] = True
        with self.assertRaisesRegex(ValueError, "Passive"):
            self.build()
        self.data[103]["target_mask"][0, 0, 0] = False
        self.data[103]["action_mask"][1] = False
        with self.assertRaisesRegex(ValueError, "observed-prefix"):
            self.build()

    def test_signature_changes_when_source_file_changes(self):
        path = self.cache.path / "episode.pt"
        path.write_bytes(b"first")
        self.cache.manifest["episodes"][0]["path"] = path.name
        left, sha_left = self.build()
        path.write_bytes(b"second and longer")
        right, sha_right = self.build()
        self.assertNotEqual(sha_left, sha_right)
        self.assertNotEqual(left["files"], right["files"])

    def test_short_val_episode_fallback_is_explicit(self):
        for eid in self.cache.manifest["splits"]["val"]:
            ep = self.data[eid]
            ep["decision_mask"][2:] = False
            ep["target_mask"][2:] = False
            ep["action_mask"][2:] = False
        plan, _ = self.build()
        self.assertTrue(any(x["phase_short_episode_fallback"] for x in plan["validation_metadata"]))
        self.assertTrue(all(x["decision"] == 1 for x in plan["validation_metadata"]))

    def test_public_audit_detects_missing_or_duplicate_query(self):
        plan, _ = self.build()
        plan["schedule"][0]["queries"][0]["decision"] = 1000
        with self.assertRaisesRegex(ValueError, "Schedule disagrees"):
            module.assert_complete_epochs_v19(plan)

    def test_bound_legacy_mapping_does_not_guess_ambiguous_swap_names(self):
        records = [{"episode_id": i, "task": "same ambiguous instruction"} for i in range(1600)]
        manifest = {"episodes": records, "fingerprint": module.LEGACY_CACHE_FINGERPRINT}
        sha = module.digest([[i, "same ambiguous instruction"] for i in range(1600)])
        with patch.object(module, "LEGACY_INSTRUCTIONS_SHA256", sha):
            mapping, info = module.resolve_tasks_v19(manifest)
        self.assertEqual(mapping[400], "VideoUnmaskFamily_block400")
        self.assertEqual(mapping[1500], "VideoUnmaskFamily_block1500")
        self.assertEqual(len(set(mapping.values())), 16)
        self.assertIn("unresolved_family_names", info)
        manifest["fingerprint"] = "other"
        with patch.object(module, "LEGACY_INSTRUCTIONS_SHA256", sha):
            with self.assertRaisesRegex(ValueError, "cannot infer task families"):
                module.resolve_tasks_v19(manifest)


def defaultdict_seeds(rows):
    result = {}
    for row in rows:
        result.setdefault((row["episode_id"], row["decision"]), set()).add(row["flow_seed"])
    return result


if __name__ == "__main__":
    unittest.main()
