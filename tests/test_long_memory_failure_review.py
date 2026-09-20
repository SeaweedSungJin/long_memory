"""Standard-library-only tests for existing-result manual failure panels."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from run_scripts.robomme import prepare_long_memory_failure_review as review


class FailureReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "evaluation"
        self.source.mkdir()
        self.tasks = ["BinFill", "MoveCube"]
        self.manifest = {
            "trainer_variant": "action_expert_v4", "evaluation_id": "paired-fixture",
            "models": {"baseline": {}, "memory": {}},
            "settings": {"tasks": self.tasks, "n_episodes": 16, "dataset": "test", "seed": 6,
                         "n_action_steps": 16, "max_episode_steps": 1300},
        }
        (self.source / "comparison_manifest.json").write_text(json.dumps(self.manifest))
        for task in self.tasks:
            for role in ("baseline", "memory"):
                directory = self.source / role / task
                directory.mkdir(parents=True)
                identity = {"evaluation_id": "paired-fixture:" + role, "task_id": task,
                            "scenario_metadata_sha256": "same", "model_config_sha256": "same",
                            "memory_window": 4, "demo_sampling": "same",
                            **{key: self.manifest["settings"][key] for key in (
                                "dataset", "seed", "n_action_steps", "max_episode_steps")}}
                (directory / "policy_manifest.json").write_text(json.dumps(identity))
                (directory / "rollout.log").write_text("[rollout] ep=0 example completed log\n")
                rows = []
                for index in range(16):
                    left, right = ((0, 0), (0, 1), (1, 0), (1, 1))[index % 4]
                    success = left if role == "baseline" else right
                    rows.append({"episode_idx": index, "episode_seed": 500 + index,
                                 "scenario_seed": 1500 + index, "success": success,
                                 "status": "success" if success else "fail", "steps": 60 + index,
                                 "video_path": "present.mp4" if index == 0 else "missing.mp4",
                                 "task_instruction": "matched goal"})
                (directory / "present.mp4").write_bytes(b"fixture, not a real video")
                self.write_results(directory / "simulation_results.csv", rows)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write_results(path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def rows(path):
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def create(self, name="review", **kwargs):
        output = self.root / name
        options = dict(tasks=self.tasks, n_per_task=8, seed=6)
        options.update(kwargs)
        manifest = review.prepare_review(self.source, output, **options)
        return output, manifest

    def test_balanced_deterministic_selection_independent_of_task_order(self):
        output, manifest = self.create()
        other, _ = self.create("again", tasks=list(reversed(self.tasks)))
        chosen = lambda path: {(row["task"], row["episode_id"]) for row in self.rows(path / "failure_review.csv")}
        self.assertEqual(chosen(output), chosen(other))
        for task in self.tasks:
            self.assertEqual(manifest["coverage"][task]["selected_by_stratum"], {name: 2 for name in review.STRATA})
            self.assertEqual(manifest["coverage"][task]["available_by_stratum"], {name: 4 for name in review.STRATA})
        for row in self.rows(output / "failure_review.csv"):
            self.assertTrue(all(row[name] == "" for name in review.LABEL_COLUMNS))
        self.assertIn("NOT frequency-representative", manifest["selection"])

    def test_missing_videos_blank_and_existing_logs_exact(self):
        output, manifest = self.create(n_per_task=100)
        self.assertEqual(sum(c["selected"] for c in manifest["coverage"].values()), 32)
        for row in self.rows(output / "failure_review.csv"):
            for role in ("baseline", "memory"):
                if row["episode_id"] == "0":
                    self.assertTrue(Path(row[f"{role}_video_path"]).is_file())
                else:
                    self.assertEqual(row[f"{role}_video_path"], "")
                self.assertEqual(row[f"{role}_log_path"], str(self.source / role / row["task"] / "rollout.log"))
        self.assertEqual(manifest["missing_video_counts"], {"baseline": 30, "memory": 30})

    def test_input_hashes_preserved_and_existing_output_not_overwritten(self):
        originals = {str(path): review._sha256(path) for path in self.source.rglob("*") if path.is_file()}
        output, manifest = self.create()
        self.assertTrue(manifest["input_sha256"])
        self.assertTrue(all(review._sha256(path) == value for path, value in originals.items()))
        old = (output / "failure_review.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(old, (output / "failure_review.csv").read_bytes())

    def test_unpaired_seed_and_incomplete_results_rejected_without_output(self):
        target = self.source / "memory" / "BinFill" / "simulation_results.csv"
        rows = self.rows(target)
        rows[0]["scenario_seed"] = "99999"
        self.write_results(target, rows)
        with self.assertRaisesRegex(ValueError, "scenario_seed"):
            self.create()
        self.assertFalse((self.root / "review").exists())
        rows[0]["scenario_seed"] = "1500"
        self.write_results(target, rows[:-1])
        with self.assertRaisesRegex(ValueError, "completed paired"):
            self.create()

    def test_bad_policy_identity_status_and_escaping_video_rejected(self):
        target = self.source / "memory" / "BinFill" / "simulation_results.csv"
        rows = self.rows(target)
        rows[0]["status"] = "success"
        self.write_results(target, rows)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.create()
        rows[0]["status"] = "fail"
        rows[0]["video_path"] = str(self.root / "outside.mp4")
        self.write_results(target, rows)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.create(n_per_task=16)
        rows[0]["video_path"] = ""
        self.write_results(target, rows)
        identity_path = target.parent / "policy_manifest.json"
        identity = json.loads(identity_path.read_text())
        identity["evaluation_id"] = "another:memory"
        identity_path.write_text(json.dumps(identity))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.create()

    def test_human_annotation_summary_and_original_values_immutable(self):
        output, _ = self.create(n_per_task=16)
        path = output / "failure_review.csv"
        result = review.summarize_review(path)
        self.assertEqual(result["memory_categories"], {"unreviewed": 32})
        rows = self.rows(path)
        rows[0].update(failure_category="unknown", baseline_failure_category="unknown", cue_observed="unknown",
                       cue_start_frame="5", cue_end_frame="9", required_at_frame="99", review_notes="No visual evidence")
        self.write_results(path, rows)
        before = path.read_bytes()
        result = review.summarize_review(path)
        self.assertEqual(result["annotation_counts"], {"any_label": 1, "no_labels": 31})
        self.assertEqual(result["memory_categories"]["unknown"], 1)
        self.assertEqual(before, path.read_bytes())
        json.dumps(result, allow_nan=False)
        rows[0]["memory_success"] = "1"
        self.write_results(path, rows)
        with self.assertRaisesRegex(ValueError, "Immutable"):
            review.summarize_review(path)

    def test_invalid_manual_labels_and_deleted_rows_fail(self):
        output, _ = self.create(n_per_task=16)
        path = output / "failure_review.csv"
        original = self.rows(path)
        for update in ({"failure_category": "automatically memory broken"}, {"correct_goal": "maybe"},
                       {"cue_start_frame": "-1"}, {"cue_start_frame": "10", "cue_end_frame": "5"},
                       {"failure_category": "success"}):
            rows = [dict(row) for row in original]
            rows[0].update(update)
            self.write_results(path, rows)
            with self.subTest(update=update), self.assertRaises(ValueError):
                review.summarize_review(path)
        self.write_results(path, original[:-1])
        with self.assertRaisesRegex(ValueError, "dropped"):
            review.summarize_review(path)

    def test_cli_help_requires_only_standard_library(self):
        result = subprocess.run([sys.executable, "-S", str(Path(review.__file__)), "--help"],
                                capture_output=True, text=True, check=True)
        self.assertIn("--report-only", result.stdout)
        self.assertIn("--n-per-task", result.stdout)
        self.assertIn("--memory-role", result.stdout)

    def test_diagnostic_full_role_and_optional_comparator_keep_explicit_csv_mapping(self):
        self.manifest["trainer_variant"] = "action_expert_v4_diagnostic_v1"
        self.manifest["models"]["full"] = self.manifest["models"].pop("memory")
        (self.source / "comparison_manifest.json").write_text(json.dumps(self.manifest))
        (self.source / "memory").rename(self.source / "full")
        for task in self.tasks:
            path = self.source / "full" / task / "policy_manifest.json"
            identity = json.loads(path.read_text())
            identity["evaluation_id"] = "paired-fixture:full"
            path.write_text(json.dumps(identity))
        output, manifest = self.create(memory_role="full", n_per_task=16)
        self.assertEqual(manifest["roles"], {"baseline": "baseline", "memory": "full"})
        self.assertIn("`full`", (output / "README.md").read_text())
        for row in self.rows(output / "failure_review.csv"):
            self.assertEqual(row["memory_log_path"], str(self.source / "full" / row["task"] / "rollout.log"))
            if row["episode_id"] == "0":
                self.assertEqual(row["memory_video_path"], str(self.source / "full" / row["task"] / "present.mp4"))
        self.assertEqual(review.summarize_review(output / "failure_review.csv")["roles"], manifest["roles"])
        reverse, reverse_manifest = self.create("reverse", baseline_role="full", memory_role="baseline", n_per_task=16)
        row = next(r for r in self.rows(reverse / "failure_review.csv") if r["episode_id"] == "1")
        self.assertEqual((row["baseline_success"], row["memory_success"]), ("1", "0"))
        self.assertEqual(reverse_manifest["roles"], {"baseline": "full", "memory": "baseline"})
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.create("invalid", baseline_role="full", memory_role="full")


if __name__ == "__main__":
    unittest.main()
