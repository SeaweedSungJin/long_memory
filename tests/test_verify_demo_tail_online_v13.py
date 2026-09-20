"""CPU orchestration + genuine tiny V13 constructor/ingest/parent chain."""
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from run_scripts.robomme import verify_demo_tail_online_v13 as probe
from run_scripts.robomme import policy_demo_tail_ingest_v13 as ingest_module
from tests import test_policy_visual_patch_v11 as legacy
from tests.test_checkpoint_demo_tail_v13 import DemoTailCheckpointTests


class TinyBackbone(nn.Module):
    def forward(self, batch):
        return legacy.CPUModel.backbone(batch)


class TinyModel(nn.Module):
    device = torch.device("cpu")
    def __init__(self, old):
        super().__init__()
        self.action_head = old.action_head
        self.action_head.num_inference_timesteps = 4
        self.backbone = TinyBackbone()
    prepare_input = staticmethod(legacy.CPUModel.prepare_input)


def initialize_tiny(actor, *args, **kwargs):
    actor.__dict__.update(legacy.parent_policy().__dict__)
    actor.model = TinyModel(actor.model).eval().requires_grad_(False)
    actor.stride = 16


def tail_image(frame):
    return (torch.arange(2 * 81 * 8).reshape(2, 81, 8).float() / 100 + frame / 17).bfloat16()


def fake_extract(backbone, vlln, processor, images, text, **kwargs):
    frame = int(images[probe.CAMERA_ORDER[0]][0][0, 0, 0])
    random.random(); np.random.rand(); torch.rand(1)  # RPC must restore all.
    backbone({"marker": frame})
    return tail_image(frame), {"synthetic": True}


def prepared(n_demo=48):
    primes = [0, 16, 32] if n_demo else []
    tail = list(range(33, 48)) if n_demo else []
    def observation(frame):
        image = {camera: [np.full((2, 2, 3), frame + 1, np.uint8)] for camera in probe.CAMERA_ORDER}
        state = {"joint_position": np.full((1, 7), .123, np.float32), "gripper_position": np.full((1, 1), .123, np.float32)}
        return probe.flat_observation({"images": image, "text": "remember"}, state, "task")
    return {"episode_id": 0 if n_demo else 100, "n_demo": n_demo, "primes": primes,
        "observations": {f: observation(f) for f in primes + [n_demo]},
        "tail": {"n_demo": n_demo, "frames": tail,
                 "images": {camera: np.stack([np.full((2, 2, 3), f, np.uint8) for f in tail])
                            for camera in probe.CAMERA_ORDER} if tail else {}, "texts": ["remember"] * len(tail)}}


class OnlineRGBProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.fixture = DemoTailCheckpointTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.bundle = self.fixture.save()

    def actors(self, awake=False):
        parent = probe.LongMemoryV7Policy.__new__(probe.LongMemoryV7Policy)
        initialize_tiny(parent)
        with patch.object(probe.LongMemoryV7Policy, "__init__", initialize_tiny):
            actor = probe.DemoTailV13Policy(self.fixture.base, self.bundle, device="cpu", visual_read_off=True)
        if awake:
            with torch.no_grad():
                actor.visual_memory.output_projection.weight.fill_(.01)
        return parent, actor

    def paired_case(self, *, n_demo=48, awake=False, corrupt=None):
        parent, actor = self.actors(awake)
        item = prepared(n_demo)
        payload = {"images": torch.stack([tail_image(f) for f in item["tail"]["frames"]])
                   if n_demo else torch.empty((0, 2, 81, 8), dtype=torch.bfloat16)}
        row, reference_row, snapshots = {}, {}, []
        with probe.same_case_seed(9111):
            reference = probe.run_case(parent, copy.deepcopy(item), payload, 9111, reference_row, lambda: None)
        with probe.same_case_seed(9111), patch.object(ingest_module, "extract_image_features", side_effect=fake_extract):
            if corrupt is None:
                probe.run_case(actor, item, payload, 9111, row, lambda: snapshots.append(copy.deepcopy(row)), candidate=True, reference=reference)
            else:
                original = actor.ingest_demo_tail
                def broken(**kwargs):
                    result = original(**kwargs)
                    corrupt(actor)
                    return result
                with patch.object(actor, "ingest_demo_tail", side_effect=broken), self.assertRaisesRegex(RuntimeError, "ingestion invariant"):
                    probe.run_case(actor, item, payload, 9111, row, lambda: snapshots.append(copy.deepcopy(row)), candidate=True, reference=reference)
        self.assertFalse(actor.sessions)
        self.assertFalse(parent.sessions)
        return row, snapshots

    def test_real_tiny_constructor_chain_demo_off_exact_at_zero_and_awake(self):
        for awake in (False, True):
            row, _ = self.paired_case(awake=awake)
            self.assertTrue(all(row["checks"].values()), row["checks"])
            self.assertTrue(all(row["ingest"]["checks"].values()), row["ingest"]["checks"])
            self.assertEqual(len(row["calls"]), 4)
            self.assertEqual(len(row["ingest"]["rgb_feature_comparisons"]), 15)
            self.assertTrue(all(v["exact"] for v in row["ingest"]["rgb_feature_comparisons"]))
            self.assertTrue(row["calls"][-1]["native_euler4_comparison"]["exact"])
            self.assertTrue(all(v["exact"] for call in row["calls"] for v in call["decoded_comparisons"].values()))

    def test_no_demo_first_action_skips_ingest_and_retains_exact_parent(self):
        row, _ = self.paired_case(n_demo=0, awake=True)
        self.assertNotIn("ingest", row)
        self.assertEqual(len(row["calls"]), 1)
        self.assertTrue(all(row["checks"].values()))
        self.assertTrue(row["calls"][0]["native_euler4_comparison"]["exact"])

    def test_independent_guard_detects_mutated_parent_after_rpc_and_records_then_resets(self):
        row, snapshots = self.paired_case(corrupt=lambda actor: setattr(next(iter(actor.sessions.values())), "observed", 99))
        self.assertFalse(row["ingest"]["checks"]["parent_fields_and_references_unchanged"])
        self.assertTrue(row["checks"]["reset_after_empty"])
        self.assertTrue(snapshots)

    def test_independent_rng_guard_catches_post_rpc_consumption(self):
        row, _ = self.paired_case(corrupt=lambda actor: torch.rand(1))
        self.assertFalse(row["ingest"]["checks"]["global_rng_unchanged"])

    def test_signature_detects_value_dtype_and_reference_changes(self):
        value = torch.ones(2)
        self.assertEqual(probe.signature(value), probe.signature(value.clone()))
        self.assertNotEqual(probe.signature(value, references=True), probe.signature(value.clone(), references=True))
        self.assertNotEqual(probe.signature(value), probe.signature(value.double()))
        original = probe.signature(value)
        value.add_(1)
        self.assertNotEqual(original, probe.signature(value))
        with self.assertRaises(TypeError):
            probe.signature(object())

    def test_flat_observation_uses_only_observed_telemetry_and_native_shapes(self):
        row = prepared()["observations"][0]
        self.assertEqual(row["video.front_view"].shape, (1, 1, 2, 2, 3))
        self.assertEqual(row["state.joint_position"].shape, (1, 1, 7))
        with self.assertRaises(ValueError):
            probe.flat_observation({"images": {c: [np.zeros((2, 2, 3), np.uint8)] for c in probe.CAMERA_ORDER}, "text": "x"},
                                   {"action": np.ones((1, 8), np.float32)}, "task")

    def test_constructors_are_separate_genuine_parent_and_v13_paths(self):
        plan = {"base_model": "base", "parent": {"path": "archive1250"}}
        with patch.object(probe, "LongMemoryV7Policy") as parent, patch.object(probe, "DemoTailV13Policy") as visual:
            probe.create_policy(plan, {"kind": "parent"}, "cpu")
            parent.assert_called_once_with("base", "archive1250", device="cpu", strict=True, write_policy="checkpoint", memory_off=False)
            visual.assert_not_called()
            probe.create_policy(plan, {"kind": "v13_off", "path": "real_v13"}, "cpu")
            visual.assert_called_once_with("base", "real_v13", device="cpu", strict=True, write_policy="checkpoint", visual_read_off=True, expected_include_tail=True)

    def test_exact_gate_never_waives_tiny_numerical_error(self):
        report = {"passed": True, "roles": [{"cases": [{"calls": [{"native_euler4_comparison": {"exact": True}}]}]}]}
        self.assertEqual(probe.completion_code(report), 0)
        report["roles"][0]["cases"][0]["calls"][0]["native_euler4_comparison"]["exact"] = False
        self.assertEqual(probe.completion_code(report), 2)
        report["passed"] = False
        self.assertEqual(probe.completion_code(report), 1)

    def arguments(self, output, *flags):
        return ["--cache-dir", "unused", "--sidecar-dir", "unused", "--checkpoint", "step0", "step4",
                "--output-dir", str(output), *flags]

    def test_readonly_preflight_never_constructs_policy_or_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            plan = {"roles": [], "cases": [{"episode_id": 0, "n_demo": 48, "frames": list(range(33, 48))}]}
            with patch.object(probe, "preflight", return_value=(plan, None, output)), \
                    patch.object(probe, "create_policy", side_effect=AssertionError("No model")), \
                    patch("torch.cuda.init", side_effect=AssertionError("No CUDA")), patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(probe.main(self.arguments(output, "--preflight-only")), 0)
            self.assertFalse(output.exists())

    def test_failed_constructor_and_final_guard_persist_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            plan = {"roles": [{"kind": "parent"}], "source_sha256": {}, "files_sha256": {}, "runtime": {}}
            with patch.object(probe, "preflight", return_value=(plan, None, output)), \
                    patch.object(probe, "create_policy", side_effect=RuntimeError("original constructor failure")), \
                    patch.object(probe, "source_hashes", side_effect=FileNotFoundError("missing source")), \
                    patch.object(probe, "runtime_identity", return_value={}), patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(probe.main(self.arguments(output)), 1)
            report = json.loads((output / "result.json").read_text())
            self.assertEqual(report["error"]["message"], "original constructor failure")
            self.assertFalse(report["checks"]["sources_unchanged"])
            self.assertEqual(sorted(p.name for p in output.iterdir()), ["plan.json", "result.json"])


if __name__ == "__main__":
    unittest.main()
