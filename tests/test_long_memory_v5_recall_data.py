"""CPU tests for label identity, train-only fitting and explicit target masks."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import pandas as pd
import torch

from gr00t.long_memory.recall_data_v5 import (RecallLabels, endpoint_targets, json_fingerprint,
                                             parse_grounded_xy, prepare_recall_labels, sha256_file)


def table(eid=0, *, unknown=False):
    return pd.DataFrame({"episode_index": [eid] * 9, "frame_index": list(range(9)),
        "is_demo": [True, True] + [False] * 7,
        "simple_subgoal_online": ["static", "static"] + ["heldout target" if unknown else "pick cube"] * 7,
        "grounded_subgoal_online": ["static", "static"] + ["pick cube at <64, 128>"] * 7,
        "simple_subgoal": ["planner label"] * 9,
        "grounded_subgoal": ["planner target at <32, 96>"] * 9})


def endpoint(eid=0):
    return {"episode_id": eid, "cache_fingerprint": "cache-fixture", "frames": torch.arange(0, 9, 2),
            "decision_mask": torch.tensor([False, True, True, True]),
            "is_demo": torch.tensor([True, False, False, False, False]),
            "transition_valid": torch.ones(4, dtype=torch.bool)}


class TestRecallLabels(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def targets(self, data=None, ep=None, **kwargs):
        return endpoint_targets(table() if data is None else data, endpoint() if ep is None else ep,
            eid=0, task="A", split="train", view=kwargs.get("view", "online"), height=256, width=256,
            window=kwargs.get("window", 2), stride=2)

    def fixture(self):
        raw, cached, base = self.root / "raw", self.root / "cache", self.root / "base"
        for path in (raw / "meta", cached / "episodes", base):
            path.mkdir(parents=True)
        info_path = raw / "meta/info.json"
        info_path.write_text(json.dumps({"features": {"image": {"shape": [256, 256, 3],
                             "names": ["height", "width", "channel"]}}, "chunks_size": 1000,
                             "data_path": "episode_{episode_index:06d}.parquet"}))
        (base / "config.json").write_text(json.dumps({"memory_window": 2, "memory_stride": 2}))
        signatures = []
        for eid in (0, 1):
            path = raw / f"episode_{eid:06d}.parquet"
            table(eid, unknown=bool(eid)).to_parquet(path)
            stat = path.stat()
            signatures.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            torch.save(endpoint(eid), cached / f"episodes/{eid}.pt")
        stat = info_path.stat()
        manifest = {"fingerprint": "cache-fixture", "dataset_path": str(raw), "model_path": str(base),
                    "splits": {"train": [0], "val": [1]},
                    "episodes": [{"episode_id": i, "path": f"episodes/{i}.pt", "task": "A", "split": s}
                                  for i, s in ((0, "train"), (1, "val"))],
                    "identity": {"payloads": signatures, "metadata": [{"path": str(info_path),
                       "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(info_path)}]}}
        (cached / "manifest.json").write_text(json.dumps(manifest))
        return SimpleNamespace(path=cached, manifest=manifest), raw, self.root / "labels"

    def test_xy_yx_swap_normalization_and_masks(self):
        xy, valid = parse_grounded_xy("pick <64, 128>", height=256, width=256)
        self.assertTrue(valid)
        self.assertEqual(xy, [128 / 255, 64 / 255])
        for bad in ("unknown", "pick <nan,2>", "pick <256,2>", "pick <-1,2>", "from <2,3> to <4,5>",
                    "from <64,128> to <target>", "<nan,1> and <2,3>", "<2,3> then <unclosed",
                    "<2,3> then stray >", "nested <<2,3>>"):
            self.assertFalse(parse_grounded_xy(bad, height=256, width=256)[1])

    def test_online_and_planner_views_are_not_mixed(self):
        self.assertEqual(self.targets()[1]["text"], "pick cube")
        self.assertEqual(self.targets(view="planner")[1]["text"], "planner label")

    def test_passive_and_terminal_targets_masked(self):
        rows = self.targets()
        for index in (0, 4):
            self.assertFalse(rows[index]["active"])
            self.assertFalse(rows[index]["xy_valid"])
            self.assertEqual(rows[index]["text"], "")
        self.assertFalse(rows[2]["available_old"])  # oldest short end=2; event0 ends=2, strict <.
        self.assertTrue(rows[3]["available_old"])

    def test_source_episode_frame_demo_and_endpoint_misalignment_fail(self):
        for field, change in (("frame_index", lambda df: df.__setitem__("frame_index", list(reversed(range(9))))),
                              ("episode_index", lambda df: df.__setitem__("episode_index", [7] * 9)),
                              ("is_demo", lambda df: df.__setitem__("is_demo", [False] * 9))):
            data = table()
            change(data)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.targets(data)
        ep = endpoint()
        ep["frames"][2] = 5
        with self.assertRaisesRegex(ValueError, "endpoints"):
            self.targets(ep=ep)

    def test_future_labels_do_not_change_current_target(self):
        data = table()
        before = self.targets(data)[1]
        data.loc[3:, "simple_subgoal_online"] = "changed future"
        data.loc[3:, "grounded_subgoal_online"] = "changed <7,9>"
        self.assertEqual(self.targets(data)[1], before)

    def test_train_only_vocab_unknown_val_and_reload(self):
        cache, raw, output = self.fixture()
        manifest = prepare_recall_labels(cache, raw, output, verify_video=False)
        self.assertEqual(manifest["class_names"], ["pick cube"])
        labels = RecallLabels(output, cache)
        self.assertTrue(labels.get(0, 1)["class_valid"])
        self.assertFalse(labels.get(1, 1)["class_valid"])
        self.assertEqual(labels.get(1, 1)["class_id"], -1)
        self.assertFalse(labels.get(1, 1)["xy_valid"])
        self.assertEqual(manifest["coverage"]["val"]["unknown_class"], 3)
        value = labels.get(0, 1)
        value["xy"][0] = 10
        self.assertLessEqual(labels.get(0, 1)["xy"][0], 1)

    def test_refuses_overwrite_changed_source_and_manifest(self):
        cache, raw, output = self.fixture()
        prepare_recall_labels(cache, raw, output, verify_video=False)
        with self.assertRaises(FileExistsError):
            prepare_recall_labels(cache, raw, output, verify_video=False)
        path = output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["class_names"] = ["tampered"]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            RecallLabels(output, cache)

    def test_output_cannot_be_nested_inside_immutable_inputs(self):
        cache, raw, _ = self.fixture()
        for source in (raw, cache.path, Path(cache.manifest["model_path"])):
            destination = source / "new_recall_labels"
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "inside"):
                prepare_recall_labels(cache, raw, destination, verify_video=False)
            self.assertFalse(destination.exists())

    def test_changed_label_file_source_split_and_negative_index_rejected(self):
        cache, raw, output = self.fixture()
        prepare_recall_labels(cache, raw, output, verify_video=False)
        labels = RecallLabels(output, cache)
        with self.assertRaises(ValueError):
            labels.get(0, -1)
        changed = copy.deepcopy(cache)
        changed.manifest["splits"] = {"train": [1], "val": [0]}
        with self.assertRaisesRegex(ValueError, "split"):
            RecallLabels(output, changed)
        (output / "episodes/episode_000000.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            labels.get(0, 1)
        path = raw / "episode_000000.parquet"
        table(0, unknown=True).to_parquet(path)
        with self.assertRaisesRegex(ValueError, "hash"):
            RecallLabels(output, cache)


if __name__ == "__main__":
    unittest.main()
