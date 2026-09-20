"""Read-only V18 comparison contracts. All outcomes and hashes below are toy fixtures."""
import copy
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from run_scripts.robomme import compare_representation_v18 as compare


class ComparisonV18Tests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.infos = {}
        self.stack.enter_context(patch.object(compare, "file_hash", return_value="fixture-hash"))
        self.stack.enter_context(patch.object(compare, "checkpoint_info_v18",
            side_effect=lambda base, checkpoint: self.infos[str(checkpoint)]))

    def run_fixture(self, name, representation="short", gate="linear", successes=(0, 1)):
        checkpoint = f"/fixture/{name}"
        self.infos[checkpoint] = {"step": 1000, "config": {
            "representation": {"representation": representation, "gate": gate,
                "hidden_dim": 256, "capacity_events": 32},
            "train": {"representation": representation, "gate": gate, "max_steps": 1000,
                "seed": 9181, "query_batch_size": 4, "memory_learning_rate": 1e-4,
                "output_dir": checkpoint, "preflight_only": False},
            "expert": {"rank": 8, "alpha": 16}, "expert_targets": ["fixture"],
            "objective": "GT_action_flow", "selection": "fixed_final_step", "storage": "FIFO"},
            "metadata": {key: f"same-{key}" for key in ("base_model", "cache_fingerprint", "plan_sha256",
                "source_sha256", "runtime", "initialization", "initial_shared_reader_sha256", "initial_expert_sha256")}}
        identity = {"scenario_metadata_sha256": "same", "model_config_sha256": "same",
                    "memory_window": 4, "demo_sampling": "same"}
        return {"root": f"/run/{name}", "manifest": {
            "models": {"memory": {"base_model": {"path": "/base"}, "memory_checkpoint": checkpoint,
                "checkpoint_files_sha256": {"checkpoint.json": "fixture-hash", "model.safetensors": "fixture-hash"}}},
            "settings": {"tasks": ["BinFill"], "dataset": "val", "n_episodes": 2, "seed": 6,
                "n_action_steps": 16, "max_episode_steps": 1300, "device": "cuda:0"},
            "benchmark": {"id": "same"}, "base_file_sha256": {"model": "same"},
            "policy_package_versions": {"torch": "same"}, "source_sha256": {"policy.py": "same"}},
            "rows": {"baseline": {"BinFill": {0: {"success": 1}, 1: {"success": 0}}},
                "memory": {"BinFill": {i: {"success": int(value)} for i, value in enumerate(successes)}}},
            "identities": {"baseline": {"BinFill": copy.deepcopy(identity)},
                "memory": {"BinFill": copy.deepcopy(identity)}}}

    def test_only_a_to_b_or_a_to_c_are_source_comparisons(self):
        a = self.run_fixture("a")
        b = self.run_fixture("b", "adapted_short")
        c = self.run_fixture("c", "moment")
        compare.matched_training(a, b, "representation")
        compare.matched_training(a, c, "representation")
        with self.assertRaisesRegex(ValueError, "not B -> C"):
            compare.matched_training(b, c, "representation")
        with self.assertRaisesRegex(ValueError, "Use A"):
            compare.matched_training(b, a, "representation")

    def test_gate_comparison_requires_same_representation(self):
        a = self.run_fixture("a", "moment", "linear")
        b = self.run_fixture("b", "moment", "mlp")
        compare.matched_training(a, b, "gate")
        self.infos["/fixture/b"]["config"]["representation"]["representation"] = "short"
        with self.assertRaisesRegex(ValueError, "More than gate changed"):
            compare.matched_training(a, b, "gate")

    def test_plan_sources_runtime_and_initial_weights_must_match(self):
        a, b = self.run_fixture("a"), self.run_fixture("b", "adapted_short")
        for key in ("plan_sha256", "source_sha256", "initial_shared_reader_sha256", "initial_expert_sha256",
                    "runtime", "cache_fingerprint", "base_model"):
            original = self.infos["/fixture/b"]["metadata"][key]
            self.infos["/fixture/b"]["metadata"][key] = "changed"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"Unmatched training: {key}"):
                compare.matched_training(a, b, "representation")
            self.infos["/fixture/b"]["metadata"][key] = original

    def test_no_checkpoint_selection_budget_changes_or_extra_factors(self):
        a, b = self.run_fixture("a"), self.run_fixture("b", "moment")
        self.infos["/fixture/b"]["step"] = 500
        with self.assertRaisesRegex(ValueError, "fixed-final"):
            compare.matched_training(a, b, "representation")
        self.infos["/fixture/b"]["config"]["train"]["max_steps"] = 500
        with self.assertRaisesRegex(ValueError, "budgets differ"):
            compare.matched_training(a, b, "representation")
        self.infos["/fixture/b"]["step"] = 1000
        self.infos["/fixture/b"]["config"]["train"]["max_steps"] = 1000
        for key, value in (("seed", 99), ("query_batch_size", 8), ("memory_learning_rate", .001)):
            old = self.infos["/fixture/b"]["config"]["train"][key]
            self.infos["/fixture/b"]["config"]["train"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "More than representation changed"):
                compare.matched_training(a, b, "representation")
            self.infos["/fixture/b"]["config"]["train"][key] = old

    def test_writer_and_tampered_payload_are_not_source_ablation(self):
        a, b = self.run_fixture("a"), self.run_fixture("b", "moment")
        b["manifest"]["models"]["memory"]["writer_checkpoint"] = "/writer"
        with self.assertRaisesRegex(ValueError, "changed writer"):
            compare.matched_training(a, b, "representation")
        b["manifest"]["models"]["memory"].pop("writer_checkpoint")
        with patch.object(compare, "file_hash", return_value="tampered"):
            with self.assertRaisesRegex(ValueError, "modified"):
                compare.matched_training(a, b, "representation")

    def test_report_uses_paired_counts_and_does_not_claim_significant_gain(self):
        a, b = self.run_fixture("a"), self.run_fixture("b", "moment", successes=(1, 1))
        with patch.object(compare, "load_completed", side_effect=[a, b]):
            result, text = compare.report("/run/a", "/run/b", "representation")
        primary = result["comparisons"]["representation: left -> right"]
        self.assertEqual(primary["delta"], .5)
        self.assertEqual((primary["wins"], primary["losses"]), (1, 0))
        self.assertEqual(primary["mcnemar_p"], 1.)
        self.assertIn("not proof of TEST improvement", text)
        self.assertIn("repeated model selection adds optimism", text)

    def test_report_rejects_different_baseline_and_eval_closure(self):
        a, b = self.run_fixture("a"), self.run_fixture("b", "moment")
        b["rows"]["baseline"]["BinFill"][0]["success"] = 0
        with patch.object(compare, "load_completed", side_effect=[a, b]):
            with self.assertRaisesRegex(ValueError, "baseline outcomes changed"):
                compare.report("/run/a", "/run/b", "representation")
        b["rows"]["baseline"]["BinFill"][0]["success"] = 1
        b["manifest"]["source_sha256"]["new-production-file.py"] = "new"
        with patch.object(compare, "load_completed", side_effect=[a, b]):
            with self.assertRaisesRegex(ValueError, "full closure|source closure differs"):
                compare.report("/run/a", "/run/b", "representation")

    def test_bash_default_is_safe_help_even_without_python(self):
        script = compare.ROOT / "run_scripts/robomme/run_representation_v18.sh"
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
        env = {**os.environ, "R18_PYTHON": "/does/not/exist/python"}
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, check=True)
        self.assertIn("no training/eval launched by default", result.stdout)
        bad = subprocess.run(["bash", str(script), "help"], env={**env, "R18_STEPS": "0"}, capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)


if __name__ == "__main__":
    unittest.main()
