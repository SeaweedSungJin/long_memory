"""Small CPU checks of evidence review I/O, provenance, and no-training scope."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from gr00t.long_memory import evidence_audit_cli as cli
from gr00t.long_memory.evidence_audit import validate_annotation
from tests.test_long_memory_v3_core import episode, model


class EvidenceAuditCliTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(80)
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache_root = self.root / "cache"
        self.cache_root.mkdir()
        self.data = episode()
        self.data["decision_mask"] = self.data["decision_mask"][:-1]
        self.data["cache_fingerprint"] = "f" * 64
        manifest = {"fingerprint": "f" * 64, "splits": {"train": [11], "val": [10]},
                    "episodes": [{"episode_id": 10, "split": "val", "task": "watch a pattern", "path": "ten.pt"},
                                 {"episode_id": 11, "split": "train", "task": "put the cube", "path": "eleven.pt"}],
                    "model_path": str(self.root / "base"), "dataset_path": str(self.root / "raw"),
                    "feature_dim": 6, "state_dim": 3, "action_dim": 4}
        self.cache = SimpleNamespace(path=self.cache_root, manifest=manifest)
        (self.cache_root / "manifest.json").write_text(json.dumps(manifest))
        torch.save(self.data, self.cache_root / "ten.pt")
        self.args = cli.parser().parse_args(["prepare", "--cache-dir", str(self.cache_root),
                            "--output-dir", str(self.root / "review")])

    def tearDown(self):
        self.temp.cleanup()

    def rows(self):
        return cli.make_prepare_records(self.args, self.cache, lambda _: self.data, [10])[0]

    def test_default_is_cpu_fifo_and_explicit_prepare_is_unknown(self):
        args = cli.parser().parse_args(["audit", "--cache-dir", "cache", "--annotations", "labels",
                                       "--checkpoint", "checkpoint", "--output-dir", "output"])
        self.assertEqual((args.device, args.writer_policy, args.top_k), ("cpu", "all", [1, 4]))
        for row in self.rows():
            self.assertEqual(row["status"], "unknown")
            self.assertEqual(row["cues"], [])
            self.assertEqual(row["reviewer"], "")
            validate_annotation(row, self.data, cache_fingerprint="f" * 64, split="val")

    def test_seeded_queries_are_chronological_strata_not_gt_selected(self):
        first, review = cli.make_prepare_records(self.args, self.cache, lambda _: self.data, [10])
        changed = copy.deepcopy(self.data)
        changed["targets"] = torch.full((8, 50, 4), float("nan"))
        changed["simple_subgoal"] = ["not evidence"] * 8
        second, _ = cli.make_prepare_records(self.args, self.cache, lambda _: changed, [10])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        decisions = [row["decision"] for row in first]
        self.assertEqual(decisions, sorted(set(decisions)))
        for row in review:
            self.assertLessEqual(max(row["sampled_frames_through_query"]), row["query_frame"])
            self.assertGreater(row["decision"], 0)

    def test_episode_filters_use_instruction_text_and_split(self):
        self.args.instruction_contains = ["PATTERN"]
        self.assertEqual(cli._choose_episode_ids(self.args, self.cache), [10])
        self.args.instruction_contains = ["VideoUnmask"]
        self.assertEqual(cli._choose_episode_ids(self.args, self.cache), [])
        self.args.instruction_contains = []
        self.args.episode_ids = [11]
        with self.assertRaisesRegex(ValueError, "split"):
            cli._choose_episode_ids(self.args, self.cache)

    def test_output_rejects_protected_descendants_and_existing_directories(self):
        for target in (self.cache_root / "new", self.root / "base/new", self.root / "raw/new"):
            self.args.output_dir = target
            with self.assertRaises(ValueError):
                cli._output_path(self.args, self.cache)
        self.args.output_dir = self.root
        with self.assertRaises(FileExistsError):
            cli._output_path(self.args, self.cache)
        self.args.output_dir = self.root / "new"
        self.assertEqual(cli._output_path(self.args, self.cache), self.root / "new")

    def test_duplicate_questions_rejected_instead_of_double_counted(self):
        path = self.root / "annotations.jsonl"
        row = self.rows()[0]
        path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            cli.load_annotations(path)

    def test_all_invalid_annotations_reported_before_metrics(self):
        rows = self.rows()
        rows[0]["split"] = "train"
        rows[1]["query_frame"] += 1
        calls = []
        def fetch(eid):
            calls.append(eid)
            return self.data
        with self.assertRaisesRegex(ValueError, "row 1.*\nrow 2"):
            cli.validate_annotations(rows, self.cache, fetch)
        self.assertEqual(len(calls), len(rows))
        self.assertFalse((self.root / "review").exists())

    def test_html_escapes_text_and_thumbnail_plan_never_adds_future_frames(self):
        _, review = cli.make_prepare_records(self.args, self.cache, lambda _: self.data, [10])
        review[0]["instruction"] = "<script>unsafe</script>"
        output = cli._review_html(review)
        self.assertIn("&lt;script&gt;", output)
        self.assertNotIn("<script>unsafe", output)
        self.assertEqual(cli._preview_frames(list(range(100)), 2), [0, 99])
        for row in review:
            frames = cli._preview_frames(row["sampled_frames_through_query"], 4)
            self.assertLessEqual(max(frames), row["query_frame"])

    def test_prepare_writes_unknown_panel_without_loading_checkpoint(self):
        with patch.object(cli, "EpisodeCache", return_value=self.cache), \
                patch.object(cli, "MappedEpisodes", return_value=SimpleNamespace(fetch=lambda _: self.data)), \
                patch.object(cli, "v4_checkpoint_info", side_effect=AssertionError("No checkpoint loading")), \
                patch.object(cli, "v5_checkpoint_info", side_effect=AssertionError("No checkpoint loading")):
            result = cli.prepare(self.args)
        output = Path(result["output_dir"])
        self.assertEqual(set(p.name for p in output.iterdir()),
                         {"annotations.jsonl", "review.json", "review_manifest.json", "README.md", "review.html"})
        self.assertTrue(all(r["status"] == "unknown" for r in cli.load_annotations(output / "annotations.jsonl")))
        self.assertIn("ten.pt", json.dumps(cli._json(output / "review_manifest.json")))
        with patch.object(cli, "EpisodeCache", return_value=self.cache):
            with self.assertRaises(FileExistsError):
                cli.prepare(self.args)

    def test_preflight_only_does_not_allocate_memory_or_publish_files(self):
        args = cli.parser().parse_args(["audit", "--cache-dir", str(self.cache_root), "--annotations", "labels",
                 "--checkpoint", "checkpoint", "--output-dir", str(self.root / "audit"), "--preflight-only"])
        data = (self.cache, lambda _: self.data, self.rows(), {10: "val"},
                {"config": {"stage": 1}}, None, 4, self.root / "audit", {"queries": 3})
        with patch.object(cli, "audit_preflight", return_value=data), \
                patch.object(cli, "ActionValueMemory", side_effect=AssertionError("No allocation")):
            result = cli.audit(args)
        self.assertEqual(result["queries"], 3)
        self.assertFalse((self.root / "audit").exists())

    def test_actual_small_memory_audit_unknown_denominator_is_not_accuracy(self):
        memory = model().eval()
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        save_file(memory.state_dict(), str(checkpoint / "model.safetensors"))
        args = cli.parser().parse_args(["audit", "--cache-dir", str(self.cache_root), "--annotations", "labels",
                 "--checkpoint", str(checkpoint), "--output-dir", str(self.root / "audit")])
        provenance = {"source_sha256": {}, "scope": "test"}
        data = (self.cache, lambda _: self.data, self.rows(), {10: "val"},
                {"config": {"stage": 1}}, memory.config, 4, self.root / "audit", provenance)
        with patch.object(cli, "audit_preflight", return_value=data):
            summary = cli.audit(args)
        output = self.root / "audit"
        self.assertTrue((output / "report.md").is_file())
        rows = cli.load_annotations(output / "results.jsonl")
        self.assertTrue(all(r["cues"] == [] and r["status"] == "unknown" for r in rows))
        self.assertFalse(any(p.grad is not None for p in memory.parameters()))
        self.assertEqual(summary, cli._json(output / "summary.json"))


    def preflight_fixture(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        base = self.root / "base"
        base.mkdir()
        (base / "config.json").write_text(json.dumps({"memory_window": 4}))
        labels = self.root / "labels"
        labels.mkdir()
        cli._write_jsonl(labels / "annotations.jsonl", self.rows())
        memory = model()
        info = {"step": 12, "config": {"stage": 1, "memory": asdict(memory.config),
                "training_recipe": cli.RECIPE, "train": {"reader_mode": "memory"}},
                "metadata": {"cache_fingerprint": "f" * 64, "base_model": {"path": str(base)}}}
        for name, field in (("model.safetensors", "memory_sha256"),
                            ("expert.safetensors", "expert_sha256"),
                            ("recall.safetensors", "recall_sha256")):
            (checkpoint / name).write_bytes(name.encode())
            info["metadata"][field] = cli.file_sha256(checkpoint / name)
        (checkpoint / "checkpoint.json").write_text(json.dumps(info))
        args = cli.parser().parse_args(["audit", "--cache-dir", str(self.cache_root),
                    "--annotations", str(labels / "annotations.jsonl"), "--checkpoint", str(checkpoint),
                    "--output-dir", str(self.root / "audit"), "--preflight-only"])
        return args, info

    def test_v5_recipe_requires_v5_validator_and_hashes_recall_payload(self):
        args, info = self.preflight_fixture()
        with patch.object(cli, "EpisodeCache", return_value=self.cache), \
                patch.object(cli, "MappedEpisodes", return_value=SimpleNamespace(fetch=lambda _: self.data)), \
                patch.object(cli, "validate_cache_checkpoint"), \
                patch.object(cli, "v5_checkpoint_info", return_value=info) as v5, \
                patch.object(cli, "v4_checkpoint_info", side_effect=AssertionError("Cannot skip recall validation")), \
                patch.object(cli, "ActionValueMemory", side_effect=AssertionError("No deployed memory allocation")):
            result = cli.audit_preflight(args)
        v5.assert_called_once()
        self.assertIn(str(args.checkpoint / "recall.safetensors"), result[-1]["source_sha256"])
        self.assertFalse(args.output_dir.exists())

    def test_untrained_stage1_writer_cannot_be_presented_as_learned_writer(self):
        args, info = self.preflight_fixture()
        args.writer_policy = "hard"
        with patch.object(cli, "EpisodeCache", return_value=self.cache), \
                patch.object(cli, "MappedEpisodes", return_value=SimpleNamespace(fetch=lambda _: self.data)), \
                patch.object(cli, "validate_cache_checkpoint"), \
                patch.object(cli, "v5_checkpoint_info", return_value=info):
            with self.assertRaisesRegex(ValueError, "trained Stage-2"):
                cli.audit_preflight(args)
        self.assertFalse(args.output_dir.exists())

    def test_terminal_endpoint_never_prepared_as_action_decision(self):
        self.data["decision_mask"] = torch.ones(len(self.data["actions"]) + 1, dtype=torch.bool)
        self.args.queries_per_episode = 100
        rows = self.rows()
        self.assertTrue(all(row["decision"] < len(self.data["actions"]) for row in rows))

    def test_unknown_episode_hash_is_informative_not_an_uncaught_keyerror(self):
        with self.assertRaisesRegex(ValueError, "Episode 999"):
            cli._hash_inputs(self.cache, [999])


if __name__ == "__main__":
    unittest.main()
