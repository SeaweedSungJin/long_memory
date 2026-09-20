"""Synthetic CPU-only full inventory tests; no real model/video/cache is used."""
from contextlib import ExitStack
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from run_scripts.robomme import demo_tail_inventory_v13 as inv
from run_scripts.robomme import demo_tail_sidecar_v13 as sidecar
from run_scripts.robomme import verify_demo_tail_v13 as proof
from tests import test_demo_tail_sidecar_v13 as fixtures
from tests.test_demo_tail_sidecar_v13 import FakeProcessor, FakeBackbone


class FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.vlln = nn.Identity(), nn.Identity()
        self._memory_cache = None

    def process_backbone_output(self, value):
        raise AssertionError("Inventory must not call HAMLET short processing")


def model_fixture():
    model = nn.Module()
    model.backbone, model.action_head = FakeBackbone(), FakeHead()
    return model.eval().requires_grad_(False)


def write_numerical_reference(path, plan, model_hash):
    path.mkdir()
    plan = copy.deepcopy(plan)
    plan["args"] = {"preflight_only": False, "preparation_only": False, "device": "cuda:0"}
    plan["proof_frames"] = proof.proof_frames(plan)
    rows = []
    exact = {"exact": True, "finite": True, "same_shape": True, "same_dtype": True, "max_abs": 0.}
    for frame in plan["proof_frames"]:
        comparisons = {name: copy.deepcopy(exact) for name in
                       ("native_repeat", "direct_vs_native", "native_vs_saved_cache", "direct_vs_saved_cache")}
        comparisons.update({name: {key: {"exact": True} for key in sidecar.INPUT_KEYS} for name in
                            ("prepared_native_repeat", "device_inputs_native_repeat", "prepared_direct", "device_inputs_direct")})
        rows.append({**frame, "structural_passed": True, "all_feature_comparisons_exact": True,
                     "checks": {"fixture": True}, "comparisons": comparisons,
                     "action_calls": {"action_head_forward": 0, "action_dit_forward": 0}})
    result = {"passed": True, "structural_passed": True, "all_feature_comparisons_exact": True,
              "numerical_review_required": False, "rows": rows,
              "frozen_before_sha256": model_hash, "frozen_after_sha256": model_hash,
              "checks": dict.fromkeys(("frozen_no_grad_before", "frozen_model_content_unchanged", "frozen_no_grad_after",
                                        "source_unchanged", "protected_files_unchanged", "runtime_unchanged"), True)}
    (path / "plan.json").write_text(json.dumps(plan))
    (path / "result.json").write_text(json.dumps(result))


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root, self.base, self.cache = fixtures.PlanTests()._fixture(self.temp.name)
        self.dataset = self.root / "dataset"
        self.model = model_fixture()
        # Keep proof TRAIN demos unchanged. VAL has one two-frame tail and one
        # no-demo episode, exercising empty payloads without large test files.
        manifest = json.loads((self.cache / "manifest.json").read_text())
        for eid, demo_count in ((11, 3), (12, 0)):
            demo = np.arange(65) < demo_count
            pd.DataFrame({"is_demo": demo, "task_index": [4] * 65,
                          "action": ["POISON_GT"] * 65, "observation.state": ["POISON_STATE"] * 65}).to_parquet(self.dataset / f"data/{eid}.parquet")
            frames = sidecar.decision_frames(demo, 16)
            torch.save({"episode_id": eid, "cache_fingerprint": manifest["fingerprint"],
                        "frames": torch.from_numpy(frames), "is_demo": torch.from_numpy(demo[frames])}, self.cache / f"{eid}.pt")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("gr00t.utils.video_utils.resolve_backend", return_value="opencv"))
        self.stack.enter_context(patch.object(sidecar, "source_identity", return_value={"fixture": "source"}))
        self.stack.enter_context(patch.object(proof, "source_hashes", return_value={"fixture": "source"}))
        self.stack.enter_context(patch.object(inv, "source_identity", return_value={"fixture": "source", "inventory": "new"}))
        template = sidecar.build_plan(self.cache, self.base)
        self.reference = self.root / "numerical-proof"
        write_numerical_reference(self.reference, template, proof.module_sha(self.model))

    def plan(self, splits=("train", "val")):
        return inv.build_inventory_plan(self.cache, self.base, self.reference, splits=splits)

    def fake_model_context(self):
        stack = ExitStack()
        stack.enter_context(patch("gr00t.long_memory.hamlet.load_frozen_hamlet", return_value=(self.model, FakeProcessor())))
        loader = SimpleNamespace(_load_video_data=lambda eid, frames: {
            camera: np.stack([np.full((3, 4, 3), int(frame) % 255, dtype=np.uint8) for frame in frames])
            for camera in sidecar.CAMERA_ORDER})
        stack.enter_context(patch("gr00t.data.dataset.lerobot_episode_loader.LeRobotEpisodeLoader", return_value=loader))
        stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
        return stack

    def test_full_fixed_partitions_keep_empty_episodes_no_gt_reads(self):
        with patch("pandas.read_parquet", wraps=pd.read_parquet) as reads:
            plan = self.plan()
        self.assertTrue(all(call.kwargs["columns"] == ["is_demo", "task_index"] for call in reads.call_args_list))
        self.assertEqual(plan["scope"], inv.SCOPE)
        self.assertEqual(plan["selection"]["episode_ids"], list(range(13)))
        self.assertEqual(plan["summary"]["episodes"], 13)
        self.assertEqual(plan["summary"]["demo_episodes"], 11)
        self.assertEqual(plan["summary"]["empty_episodes"], 2)
        self.assertEqual(plan["summary"]["frames"], 152)
        self.assertEqual(plan["episodes"][1]["frames"], [])
        self.assertEqual(plan["episodes"][1]["last_canonical_demo"], -1)
        self.assertEqual(sidecar.digest(plan), sidecar.digest(self.plan()))
        inv.validate_plan(plan)
        self.assertFalse(torch.cuda.is_initialized())

    def test_partition_selection_no_test_duplicates_or_silent_subset(self):
        for splits in (["test"], [], ["train", "train"]):
            with self.assertRaisesRegex(ValueError, "TRAIN/VAL"):
                self.plan(splits)
        plan = self.plan(("val",))
        self.assertEqual(plan["selection"]["episode_ids"], [11, 12])
        broken = copy.deepcopy(plan)
        broken["episodes"].pop()
        broken["selection"]["episode_ids"].pop()
        broken["summary"] = inv.inventory_summary(broken["episodes"])
        with self.assertRaisesRegex(ValueError, "every episode"):
            inv.validate_plan(broken)

    def test_stale_raw_cached_boundary_rejected(self):
        cached = torch.load(self.cache / "12.pt", weights_only=True)
        cached["frames"][1] = 15
        torch.save(cached, self.cache / "12.pt")
        with self.assertRaisesRegex(ValueError, "query indices"):
            self.plan()

    def test_wrong_proof_failed_nonexact_processor_only_or_changed_source_rejected(self):
        result_path, plan_path = self.reference / "result.json", self.reference / "plan.json"
        saved_result, saved_plan = json.loads(result_path.read_text()), json.loads(plan_path.read_text())
        for changed in ("failed", "nonexact", "missing_rows", "action_calls"):
            result = copy.deepcopy(saved_result)
            if changed == "failed":
                result["passed"] = False
            elif changed == "nonexact":
                result["rows"][0]["comparisons"]["direct_vs_native"]["max_abs"] = .001
            elif changed == "missing_rows":
                result["rows"].pop()
            else:
                result["rows"][0]["action_calls"]["action_dit_forward"] = 1
            result_path.write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                self.plan()
        result_path.write_text(json.dumps(saved_result))
        changed = copy.deepcopy(saved_plan)
        changed["args"]["preparation_only"] = True
        plan_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "actual numerical"):
            self.plan()
        changed = copy.deepcopy(saved_plan)
        changed["source_sha256"]["fixture"] = "changed"
        plan_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "sources changed"):
            self.plan()

    def test_cost_and_free_space_preflight_new_output_only(self):
        plan = self.plan()
        self.assertEqual(plan["summary"]["raw_payload_bytes"], 152 * 663552)
        self.assertGreaterEqual(plan["summary"]["required_free_bytes"], plan["summary"]["raw_payload_bytes"] + 2 * inv.GIB)
        output = self.root / "output"
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                inv.check_disk_space(plan, output)
        self.assertFalse(output.exists())
        for protected in (self.cache / "new", self.base / "new", self.reference / "new"):
            with self.assertRaisesRegex(ValueError, "overlap"):
                inv.check_disk_space(plan, protected)
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "NEW"):
            inv.check_disk_space(plan, output)

    def test_preflight_summary_no_output_model_or_cuda(self):
        output = self.root / "no-output"
        with patch("gr00t.long_memory.hamlet.load_frozen_hamlet", side_effect=AssertionError("no model")) as model, \
                patch("torch.cuda.init", side_effect=AssertionError("no CUDA")) as cuda, \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = inv.main(["--cache-dir", str(self.cache), "--base-model", str(self.base),
                             "--numerical-proof", str(self.reference), "--output-dir", str(output), "--preflight-only"])
        self.assertEqual(code, 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["summary"]["episodes"], 13)
        self.assertNotIn("episodes", summary)
        self.assertFalse(output.exists())
        model.assert_not_called()
        cuda.assert_not_called()

    def test_atomic_complete_roundtrip_empty_payload_and_no_overwrite(self):
        plan, output = self.plan(("val",)), self.root / "complete"
        with self.fake_model_context():
            self.assertEqual(inv.extract_inventory(plan, output, device="cpu"), output)
        reader = sidecar.DemoTailSidecar(output, expected_cache_fingerprint=plan["cache_fingerprint"])
        self.assertEqual(reader.load(11)["images"].shape, (2, 2, 81, 2048))
        empty = reader.load(12)
        self.assertEqual(empty["images"].shape, (0, 2, 81, 2048))
        self.assertEqual(empty["images"].dtype, torch.bfloat16)
        self.assertEqual(empty["frames"].dtype, torch.int64)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(set(manifest["checks"]), sidecar.CHECK_KEYS)
        self.assertTrue(all(manifest["inventory_checks"].values()))
        self.assertEqual(json.loads((output / "progress.json").read_text())["status"], "complete")
        with self.assertRaisesRegex(ValueError, "NEW"):
            inv.extract_inventory(plan, output, device="cpu")

    def test_interruption_preserves_first_episode_and_unusable_failed_manifest(self):
        plan, output = self.plan(("val",)), self.root / "interrupted"
        original = sidecar._atomic_torch_save
        calls = 0
        def interrupt(path, value):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("synthetic interruption")
            return original(path, value)
        with self.fake_model_context(), patch.object(sidecar, "_atomic_torch_save", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                inv.extract_inventory(plan, output, device="cpu")
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["completed_episodes"], [11])
        self.assertTrue((output / "episodes/episode_000011.pt").is_file())
        self.assertTrue((output / "records/episode_000011.json").is_file())
        with self.assertRaisesRegex(ValueError, "incomplete"):
            sidecar.DemoTailSidecar(output)

    def test_nonfinite_extraction_stops_before_payload_write(self):
        plan, output = self.plan(("val",)), self.root / "nonfinite"
        with self.fake_model_context(), patch.object(sidecar, "extract_observations", return_value=(
                torch.full((2, 2, 81, 2048), float("nan"), dtype=torch.bfloat16), [])):
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                inv.extract_inventory(plan, output, device="cpu")
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse(list((output / "episodes").iterdir()))


if __name__ == "__main__":
    unittest.main()
