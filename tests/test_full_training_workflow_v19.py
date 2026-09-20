"""CPU-only V19 workflow/report contracts; all scores below are toy fixtures."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from run_scripts.robomme import compare_full_training_v19 as compare


class WorkflowV19Tests(unittest.TestCase):
    def info(self, weight=1.):
        return {"step": 7, "config": {
            "driver_variant": "full_memory_v19",
            "representation": {"representation": "short", "gate": "linear", "capacity_events": 32},
            "train": {"epochs": 1, "max_steps": 7, "tail_weight": weight, "query_batch_size": 4,
                      "seed": 9191, "output_dir": f"/fixture/{weight}", "task_weighting": "macro"},
            "objective": {"name": "fixture-flow", "tail_weight": weight, "prefix_steps": 16},
            "expert": {"rank": 8}, "storage": "FIFO", "selection": "fixed_final_epoch"},
            "metadata": {**{k: f"same-{k}" for k in ("base_model", "cache_fingerprint", "plan_sha256", "source_sha256",
                "runtime", "initialization", "initial_shared_reader_sha256", "initial_expert_sha256")},
                "full_coverage": {"train_query_count": 28, "epochs": 1, "total_query_presentations": 28, "total_steps": 7}}}

    def run_fixture(self, name, successes=(0, 1)):
        identity = {"scenario_metadata_sha256": "same", "model_config_sha256": "same",
                    "memory_window": 4, "demo_sampling": "same"}
        return {"root": f"/run/{name}", "manifest": {
            "models": {"memory": {"base_model": {"path": "/base"}, "memory_checkpoint": f"/fixture/{name}/checkpoint-000007",
                "checkpoint_files_sha256": {"checkpoint.json": "fixture-hash"}}},
            "settings": {"tasks": ["BinFill"], "dataset": "val", "n_episodes": 2, "seed": 6,
                "n_action_steps": 16, "max_episode_steps": 1300, "device": "cuda:0"},
            "benchmark": {"id": "same"}, "base_file_sha256": {"model": "same"},
            "policy_package_versions": {"torch": "same"}, "source_sha256": {"policy.py": "same"}},
            "rows": {"baseline": {"BinFill": {0: {"success": 1}, 1: {"success": 0}}},
                "memory": {"BinFill": {i: {"success": int(v)} for i, v in enumerate(successes)}}},
            "identities": {"baseline": {"BinFill": copy.deepcopy(identity)}, "memory": {"BinFill": copy.deepcopy(identity)}}}

    def pair(self, a, b):
        with patch.object(compare, "_info_for_run", side_effect=[a, b]):
            return compare.matched_training(self.run_fixture("control"), self.run_fixture("prefix"))

    def test_only_tail_changes_and_all_full_coverage_provenance_matches(self):
        a, b = self.info(), self.info(.25)
        result = self.pair(a, b)
        self.assertEqual(result["epochs"], 1)
        for key in ("plan_sha256", "cache_fingerprint", "source_sha256", "runtime", "base_model",
                    "initial_shared_reader_sha256", "initial_expert_sha256"):
            bad = copy.deepcopy(b)
            bad["metadata"][key] = "different"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Unmatched full-coverage"):
                self.pair(a, bad)
        for key, value in (("seed", 5), ("epochs", 2), ("task_weighting", "query"), ("query_batch_size", 8)):
            bad = copy.deepcopy(b)
            bad["config"]["train"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "More than tail_weight"):
                self.pair(a, bad)

    def test_reject_extra_objective_architecture_and_wrong_tail_changes(self):
        a, b = self.info(), self.info(.25)
        for section, key, value in (("objective", "prefix_steps", 8), ("representation", "gate", "mlp"),
                                     ("expert", "rank", 16)):
            bad = copy.deepcopy(b)
            bad["config"][section][key] = value
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, "More than tail_weight"):
                self.pair(a, bad)
        with self.assertRaisesRegex(ValueError, "Expected full-chunk control"):
            self.pair(b, a)
        b["config"]["objective"]["tail_weight"] = .5
        with self.assertRaisesRegex(ValueError, "Expected full-chunk control"):
            self.pair(a, b)

    def test_exact_resume_paths_allowed_but_plan_runtime_and_initialization_still_match(self):
        a, b = self.info(), self.info(.25)
        a["config"]["train"]["resume"] = None
        b["config"]["train"]["resume"] = "/fixture/paused-prefix/checkpoint-000003"
        b["metadata"]["resume_parent"] = b["config"]["train"]["resume"]
        self.pair(a, b)
        for key in ("plan_sha256", "runtime", "source_sha256", "initial_shared_reader_sha256", "initial_expert_sha256"):
            invalid = copy.deepcopy(b)
            invalid["metadata"][key] = "different-resume"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Unmatched full-coverage"):
                self.pair(a, invalid)

    def test_final_checkpoint_requires_actual_v19_final_epoch_and_architecture(self):
        compare.validate_final_info(self.info())
        for mutate, message in (
            (lambda x: x["config"].update(driver_variant="representation_v18"), "actual full_memory"),
            (lambda x: x.update(step=3), "fixed-final"),
            (lambda x: x["config"]["train"].update(epochs=0), "positive full-epoch"),
            (lambda x: x["config"]["representation"].update(capacity_events=64), "V18 A"),
            (lambda x: x["metadata"]["full_coverage"].update(total_query_presentations=4), "coverage provenance"),
        ):
            bad = self.info()
            mutate(bad)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                compare.validate_final_info(bad)

    def test_final_selector_uses_last_not_best_and_rejects_paused_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint-000007"
            checkpoint.mkdir()
            info = self.info()
            plan = info["metadata"]["full_coverage"].copy()
            plan.update(train=[[1, list(range(28))]], windows=[], schedule=[])
            for start in range(0, 28, 4):
                decisions = list(range(start, start + 4))
                plan["windows"].append({"epoch": 0, "groups": [[1, decisions]], "query_count": 4})
                plan["schedule"].append({"queries": [{"episode_id": 1, "decision": d} for d in decisions]})
            info["metadata"]["plan_sha256"] = compare.digest(plan)
            (root / "query_plan.json").write_text(json.dumps({"sha256": compare.digest(plan), **plan}))
            (checkpoint / "checkpoint.json").write_text(json.dumps(info))
            (root / "last_checkpoint.json").write_text(json.dumps({"path": checkpoint.name, "step": 7}))
            (root / "best_checkpoint.json").write_text(json.dumps({"path": "checkpoint-000002", "step": 2}))
            status = {"status": "complete", "step": 7, "max_steps": 7, "completed_epochs": 1, "processed_queries": 28}
            (root / "status.json").write_text(json.dumps(status))
            self.assertEqual(compare.final_checkpoint(root), checkpoint)
            for update, message in (({"status": "paused"}, "not complete"), ({"completed_epochs": 0}, "disagree"),
                                    ({"step": 6}, "disagree")):
                (root / "status.json").write_text(json.dumps({**status, **update}))
                with self.subTest(update=update), self.assertRaisesRegex(ValueError, message):
                    compare.final_checkpoint(root)
            (root / "status.json").write_text(json.dumps(status))
            (root / "query_plan.json").write_text(json.dumps({"sha256": compare.digest(plan), **plan, "modified": True}))
            with self.assertRaisesRegex(ValueError, "plan identity changed"):
                compare.final_checkpoint(root)
            (root / "query_plan.json").write_text(json.dumps({"sha256": compare.digest(plan), **plan}))
            (root / "failure.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "failure.json"):
                compare.final_checkpoint(root)

    def test_checkpoint_payload_integrity_and_no_writer(self):
        run = self.run_fixture("control")
        checkpoint = Path(run["manifest"]["models"]["memory"]["memory_checkpoint"])
        with patch.object(compare, "final_checkpoint", return_value=checkpoint), \
                patch.object(compare, "file_hash", return_value="tampered"):
            with self.assertRaisesRegex(ValueError, "modified"):
                compare._info_for_run(run)
        run["manifest"]["models"]["memory"]["writer_checkpoint"] = "/writer"
        with self.assertRaisesRegex(ValueError, "CVOM is a separate"):
            compare._info_for_run(run)

    def test_report_paired_toy_counts_and_scientific_caveats(self):
        a, b = self.run_fixture("control"), self.run_fixture("prefix", (1, 1))
        with patch.object(compare, "load_completed", side_effect=[a, b]), \
                patch.object(compare, "_info_for_run", side_effect=[self.info(), self.info(.25)]):
            result, text = compare.report("/run/control", "/run/prefix")
        row = result["comparisons"]["full-chunk control -> execution-prefix"]
        self.assertEqual((row["delta"], row["wins"], row["losses"], row["mcnemar_p"]), (.5, 1, 0, 1.))
        self.assertIn("not proof of TEST improvement", text)
        self.assertIn("not every raw video frame", text)
        self.assertIn("does not by itself prove better memory", text)

    def test_report_rejects_mismatched_baseline_or_eval_source(self):
        a, b = self.run_fixture("control"), self.run_fixture("prefix")
        b["rows"]["baseline"]["BinFill"][0]["success"] = 0
        with patch.object(compare, "load_completed", side_effect=[a, b]):
            with self.assertRaisesRegex(ValueError, "baseline outcomes changed"):
                compare.report("/run/control", "/run/prefix")
        b["rows"]["baseline"]["BinFill"][0]["success"] = 1
        b["manifest"]["source_sha256"]["new.py"] = "new"
        with patch.object(compare, "load_completed", side_effect=[a, b]):
            with self.assertRaisesRegex(ValueError, "source closure differs"):
                compare.report("/run/control", "/run/prefix")

    def test_shell_default_safe_train_phases_complete_and_test_opt_in(self):
        script = compare.ROOT / "run_scripts/robomme/run_full_training_v19.sh"
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
        env = {**os.environ, "R19_PYTHON": "/does/not/exist/python"}
        env.pop("R19_SELECTED_ARM", None)
        default = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=True)
        self.assertIn("no training/eval launched by default", default.stdout)
        bad = subprocess.run(["bash", str(script), "plan"], env={**env, "R19_EPOCHS": "0"}, capture_output=True)
        self.assertEqual(bad.returncode, 2)
        final = subprocess.run(["bash", str(script), "final-test"], env=env, text=True, capture_output=True)
        self.assertEqual(final.returncode, 2)
        self.assertIn("Explicitly set R19_SELECTED_ARM", final.stderr)
        # echo only: assert generated training CLI without invoking Python/GPU.
        planned = subprocess.run(["bash", str(script), "preflight"], env={**env, "R19_PYTHON": "/bin/echo"},
            text=True, capture_output=True, check=True).stdout
        self.assertEqual(planned.count("--preflight-only"), 2)
        self.assertIn("--tail-weight 1", planned)
        self.assertIn("--tail-weight .25", planned)
        self.assertIn("--epochs 1", planned)
        self.assertNotIn("--max-steps", planned)


if __name__ == "__main__":
    unittest.main()
