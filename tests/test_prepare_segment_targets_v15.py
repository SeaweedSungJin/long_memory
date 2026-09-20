"""Synthetic CPU tests only: maximal-run weak labels, not visual cue correctness."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from run_scripts.robomme import prepare_segment_targets_v15 as prep


def episode(labels=("move right", "move left")):
    phase = [label for label in labels for _ in range(48)]
    n_demo = len(phase)
    raw_demo = [True] * n_demo + [False] * n_demo
    length = len(raw_demo)
    canonical = sorted({0, length - 1, *range(n_demo, -1, -16), *range(n_demo, length, 16)})
    return {"episode_id": 0, "split": "train", "raw_demo": raw_demo,
            "planner": phase * 2, "online": phase * 2,
            "canonical_frames": canonical,
            "decision_mask": [frame >= n_demo for frame in canonical[:-1]],
            "tail_frames": list(range(n_demo - 15, n_demo))}


class PureTests(unittest.TestCase):
    def test_split_before_runs_and_single_positive_episode_excluded(self):
        args = episode(("move right",))
        result = prep.episode_targets(**args)
        self.assertEqual(result["demo_run_count"], 1)
        self.assertEqual(result["execution_run_count"], 1)
        self.assertEqual(result["episode_exclusion"], "fewer_than_two_maximal_runs")
        self.assertEqual(result["examples"], [])

    def test_positive_candidates_causality_margin_and_actual_short_horizon(self):
        args = episode()
        result = prep.episode_targets(**args)
        self.assertIsNone(result["episode_exclusion"])
        row = next(x for x in result["examples"] if x["query_frame"] == 112)
        self.assertEqual(row["decision"], args["canonical_frames"].index(112))
        self.assertEqual(row["n_demo"], 96)
        self.assertEqual(row["positive_interval"], [2, 46])
        self.assertEqual(row["positive_frames"], [16, 32])
        self.assertEqual(row["oldest_short_frame"], 64)
        self.assertEqual(row["old_positive_frames"], [16, 32])
        self.assertEqual(row["candidate_frames"], sorted([f for f in args["canonical_frames"]
            + args["tail_frames"] if f < 112]))
        self.assertNotIn(112, row["candidate_frames"])
        self.assertEqual(result["query_exclusions"]["query_within_planner_boundary_margin"], 2)
        later = next(x for x in result["examples"] if x["query_frame"] == 160)
        self.assertEqual(later["positive_interval"], [50, 94])
        self.assertNotIn(48, later["positive_frames"])
        self.assertNotIn(94, later["positive_frames"])
        self.assertIn(93, later["positive_frames"])

    def test_old_positive_is_strict_before_oldest_not_equal_or_fixed_age(self):
        result = prep.episode_targets(**episode(), memory_window=6)
        row = next(x for x in result["examples"] if x["query_frame"] == 112)
        self.assertEqual(row["positive_frames"], [16, 32])
        self.assertEqual(row["oldest_short_frame"], 32)
        self.assertEqual(row["old_positive_frames"], [16])

    def test_online_agreement_applies_to_current_and_each_positive(self):
        args = episode()
        args["online"] = args["online"].copy()
        args["online"][112] = "move left"
        args["online"][16] = "no record"
        result = prep.episode_targets(**args)
        self.assertNotIn(112, [r["query_frame"] for r in result["examples"]])
        row = next(r for r in result["examples"] if r["query_frame"] == 128)
        self.assertEqual(row["positive_frames"], [32])
        self.assertEqual(result["query_exclusions"]["query_planner_online_disagree"], 1)

    def test_no_observed_positive_is_not_invented_from_unobserved_raw_frames(self):
        args = episode()
        args["online"] = args["online"].copy()
        args["online"][16] = args["online"][32] = "no record"
        result = prep.episode_targets(**args)
        self.assertTrue(all(r["raw_run_ordinal"] == 1 for r in result["examples"]))
        self.assertEqual(result["query_exclusions"]["no_observed_positive_after_margin_and_agreement"], 2)

    def test_repeated_nonconsecutive_labels_and_mismatched_sequence_rejected(self):
        repeated = prep.episode_targets(**episode(("move right", "move left", "move right")))
        self.assertEqual(repeated["episode_exclusion"], "repeated_run_label_ambiguous_occurrence")
        args = episode()
        args["planner"] = args["planner"][:96] + args["planner"][144:] + args["planner"][96:144]
        self.assertEqual(prep.episode_targets(**args)["episode_exclusion"],
                         "demo_execution_run_sequence_mismatch")

    def test_contiguous_identical_atomic_events_are_only_one_maximal_run(self):
        self.assertEqual(prep.maximal_runs(["move right"] * 10, 0, 10),
                         [{"start": 0, "end": 10, "label": "move right"}])

    def test_invalid_demo_order_duplicate_future_tail_or_active_demo_fail_closed(self):
        for mutation in ("demo_order", "duplicate_tail", "future_tail", "active_demo", "bad_mask", "test_split"):
            args = episode()
            if mutation == "demo_order":
                args["raw_demo"][0], args["raw_demo"][-1] = False, True
            elif mutation == "duplicate_tail":
                args["tail_frames"] = [80] + args["tail_frames"]
            elif mutation == "future_tail":
                args["tail_frames"].append(100)
            elif mutation == "active_demo":
                args["decision_mask"][0] = True
            elif mutation == "bad_mask":
                args["decision_mask"].append(True)
            else:
                args["split"] = "test"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                prep.episode_targets(**args)

    def test_changing_future_online_labels_does_not_change_prior_query_target(self):
        args = episode()
        before = prep.episode_targets(**args)
        args["online"] = args["online"].copy()
        args["online"][113:] = ["FUTURE_POISON"] * (len(args["online"]) - 113)
        after = prep.episode_targets(**args)
        self.assertEqual(next(r for r in before["examples"] if r["query_frame"] == 112),
                         next(r for r in after["examples"] if r["query_frame"] == 112))


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache, self.sidecar = self.root / "cache", self.root / "sidecar"
        dataset, base = self.root / "dataset", self.root / "base"
        for path in (self.cache / "episodes", self.sidecar / "episodes", dataset / "meta", dataset / "data", base):
            path.mkdir(parents=True)
        goal = sorted(prep.GOALS)[0]
        def write(path, value):
            path.write_text(json.dumps(value))
        write(base / "config.json", {"memory_window": 4, "memory_stride": 16})
        write(dataset / "meta/info.json", {"chunks_size": 1000, "data_path": "data/{episode_index}.parquet"})
        write(dataset / "meta/modality.json", {})
        (dataset / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": goal}) + "\n")
        (dataset / "meta/episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": eid}) for eid in range(3)))
        cache_records, tail_records, payload_records, files = [], [], [], {}
        for eid, split, labels in ((0, "train", ("move right", "move left")),
                                  (1, "val", ("move forward", "move backward")),
                                  (2, "train", ("move right",))):
            args = episode(labels)
            length, n_demo = len(args["raw_demo"]), sum(args["raw_demo"])
            parquet = dataset / f"data/{eid}.parquet"
            pq.write_table(pa.table({"episode_index": [eid] * length, "frame_index": list(range(length)),
                "task_index": [0] * length, "is_demo": args["raw_demo"], "simple_subgoal": args["planner"],
                "simple_subgoal_online": args["online"], "state": ["DO_NOT_READ_STATE"] * length,
                "actions": ["DO_NOT_READ_ACTIONS"] * length, "image": ["DO_NOT_READ_IMAGE"] * length}), parquet)
            relative = f"episodes/{eid}.pt"
            cached = self.cache / relative
            torch.save({"episode_id": eid, "cache_fingerprint": "cache_fixture",
                "frames": torch.tensor(args["canonical_frames"], dtype=torch.int64),
                "is_demo": torch.tensor([args["raw_demo"][f] for f in args["canonical_frames"]]),
                "decision_mask": torch.tensor(args["decision_mask"]),
                "features": "DO_NOT_READ_FEATURES", "state": "DO_NOT_READ_STATE",
                "actions": "DO_NOT_READ_ACTIONS", "targets": "DO_NOT_READ_TARGETS"}, cached)
            tail = {"episode_id": eid, "split": split, "task": goal,
                "canonical_frames": args["canonical_frames"], "frames": args["tail_frames"],
                "n_demo": n_demo, "last_canonical_demo": n_demo - 16}
            payload = self.sidecar / relative
            payload.write_bytes(b"opaque fixture payload; must not decode images")
            cache_records.append({"episode_id": eid, "split": split, "task": goal, "path": relative})
            tail_records.append(tail)
            payload_records.append({**tail, "path": relative, "payload_sha256": prep.file_sha256(payload)})
            files.update({str(p): prep.file_sha256(p) for p in (parquet, cached)})
        cache = {"status": "complete", "fingerprint": "cache_fixture", "dataset_path": str(dataset),
                 "model_path": str(base), "episodes": cache_records, "splits": {"train": [0, 2], "val": [1]},
                 "identity": {"metadata": [], "code": [], "checkpoint": []}}
        write(self.cache / "manifest.json", cache)
        plan = {"scope": "inventory_train_val", "cache_fingerprint": "cache_fixture",
                "cache_dir": str(self.cache), "dataset_path": str(dataset), "base_model": str(base),
                "episodes": tail_records, "source_sha256": {}, "files_sha256": files}
        sidecar = {"kind": "demo_tail_sidecar_v13", "driver_variant": "demo_tail_inventory_v13",
            "status": "complete", "fingerprint": prep.digest(plan), "plan": plan,
            "episodes": payload_records, "completed_episodes": [0, 1, 2], "integrity_errors": [],
            "checks": dict.fromkeys(("input_files_unchanged", "sources_unchanged", "runtime_unchanged",
                "frozen_model_versions_unchanged", "model_frozen_no_grad", "all_planned_payloads_complete"), True),
            "inventory_checks": dict.fromkeys(("zero_action_calls", "original_short_memory_unchanged",
                "frozen_model_content_unchanged"), True)}
        write(self.sidecar / "manifest.json", sidecar)

    def build(self):
        return prep.build_manifest(self.cache, self.sidecar)

    def test_build_is_deterministic_target_only_and_train_vocab_not_val(self):
        with patch.object(pq, "read_table", wraps=pq.read_table) as reads, \
                patch.object(torch.cuda, "init", side_effect=AssertionError("No CUDA")):
            one, two = self.build(), self.build()
        self.assertEqual(one, two)
        self.assertEqual(one["vocabulary"], ["move left", "move right"])
        self.assertTrue(all(r["label_id"] == -1 for r in one["examples"] if r["split"] == "val"))
        self.assertEqual(one["exclusions"]["train"]["episodes"], {"fewer_than_two_maximal_runs": 1})
        self.assertTrue(all(c.kwargs["columns"] == list(prep.RAW_COLUMNS) for c in reads.call_args_list))
        self.assertFalse(torch.cuda.is_initialized())

    def test_preflight_no_output_and_explicit_publish_new_only_roundtrip(self):
        output = self.root / "targets"
        args = ["--cache-dir", str(self.cache), "--sidecar-dir", str(self.sidecar), "--output-dir", str(output)]
        with patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(prep.main(args + ["--preflight-only"]), 0)
        self.assertFalse(output.exists())
        manifest = self.build()
        path = prep.publish(manifest, output)
        self.assertEqual(prep.load_manifest(path), manifest)
        with self.assertRaisesRegex(ValueError, "NEW"):
            prep.publish(manifest, output)

    def test_stale_sidecar_payload_and_incomplete_manifest_fail(self):
        (self.sidecar / "episodes/0.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "content changed"):
            self.build()
        path = self.sidecar / "manifest.json"
        data = json.loads(path.read_text()); data["status"] = "running"; path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "complete"):
            self.build()

    def test_configuration_and_hashed_input_mutations_rejected(self):
        manifest = self.build()
        for changed in ("vocabulary", "future", "old", "config"):
            broken = copy.deepcopy(manifest)
            if changed == "vocabulary":
                broken["vocabulary"].append("move forward")
            elif changed == "future":
                broken["examples"][0]["candidate_frames"].append(broken["examples"][0]["query_frame"])
            elif changed == "old":
                broken["examples"][0]["old_positive_frames"] = []
            else:
                broken["config"]["memory_window"] = 5
            broken["fingerprint"] = prep.digest({k: v for k, v in broken.items() if k != "fingerprint"})
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                prep.validate_manifest(broken)
        (self.cache / "episodes/0.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Changed files_sha256"):
            prep.verify_identity(manifest)

    def test_exact_global_task_family_selection_not_substring_or_episode_id(self):
        path = self.cache / "manifest.json"
        cache = json.loads(path.read_text())
        for record in cache["episodes"]:
            record["task"] += " move right"
        path.write_text(json.dumps(cache))
        with self.assertRaisesRegex(ValueError, "No exact PatternLock"):
            self.build()

    def test_missing_completed_inventory_guard_fails_closed(self):
        path = self.sidecar / "manifest.json"
        data = json.loads(path.read_text())
        del data["checks"]["all_planned_payloads_complete"]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "sidecar guards"):
            self.build()

    def test_new_directory_inside_any_immutable_input_root_cannot_be_published(self):
        manifest = self.build()
        for name in ("cache_dir", "sidecar_dir", "base_model", "dataset_path"):
            output = Path(manifest["identity"][name]) / "forbidden_new_targets"
            with self.subTest(root=name), self.assertRaisesRegex(ValueError, "inside protected"):
                prep.publish(manifest, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
