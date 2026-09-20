"""Synthetic CPU-only proof tests; no real cache, weights, videos or GPU."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from run_scripts.robomme import demo_tail_sidecar_v13 as sidecar


class FakeProcessor:
    training = False
    formalize_language = True
    eval_image_transform = object()
    modality_configs = {"new_embodiment": {"video": SimpleNamespace(modality_keys=list(sidecar.CAMERA_ORDER))}}

    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        raise AssertionError("Full processor/state/action path must not be called")

    def _get_vlm_inputs(self, order, images, masks, transform, language):
        self.calls.append((order, masks, transform, language))
        return {"vlm_content": {"images": images, "text": language}}

    def collator(self, values):
        images = values[0]["vlm_content"]["images"]
        pixels = torch.tensor([int(images[c][0][0, 0, 0]) for c in sidecar.CAMERA_ORDER], dtype=torch.float32)
        return {"inputs": {"input_ids": torch.zeros(1, 184, dtype=torch.long),
                           "attention_mask": torch.ones(1, 184, dtype=torch.long),
                           "pixel_values": pixels}}


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        mask = torch.zeros(1, 188, dtype=torch.bool)
        mask[:, 3:84] = True
        mask[:, 91:172] = True
        features = torch.zeros(1, 188, sidecar.WIDTH, dtype=torch.bfloat16)
        features[:, 3:84] = inputs["pixel_values"][0]
        features[:, 91:172] = inputs["pixel_values"][1]
        return {"backbone_features": features, "image_mask": mask,
                "backbone_attention_mask": torch.ones_like(mask)}


class FakeListProcessor(FakeProcessor):
    """Match actual Eagle: a Python list with one separate tensor per image."""
    def collator(self, values):
        batch = super().collator(values)
        pixels = batch["inputs"]["pixel_values"]
        batch["inputs"]["pixel_values"] = [pixels[0:1], pixels[1:2]]
        batch["inputs"]["image_sizes"] = torch.tensor([[252, 252], [252, 252]], dtype=torch.int64)
        return batch


def rgb_pair():
    return {camera: [np.full((3, 4, 3), value, dtype=np.uint8)]
            for camera, value in zip(sidecar.CAMERA_ORDER, (7, 13))}


def tiny_record():
    return {"episode_id": 0, "frames": [1, 2], "canonical_frames": [0, 3, 8],
            "n_demo": 3, "last_canonical_demo": 0}


def tiny_payload(record=None):
    record = tiny_record() if record is None else record
    return {"episode_id": record["episode_id"], "cache_fingerprint": "cache",
            "sidecar_fingerprint": "sidecar", "frames": torch.tensor(record["frames"], dtype=torch.int64),
            "is_demo": torch.ones(len(record["frames"]), dtype=torch.bool),
            "images": torch.zeros(len(record["frames"]), 2, 81, sidecar.WIDTH, dtype=torch.bfloat16)}


def validate(payload, record=None):
    sidecar.validate_sidecar_episode(payload, tiny_record() if record is None else record,
                                    cache_fingerprint="cache", sidecar_fingerprint="sidecar")


class TailRuleTests(unittest.TestCase):
    def test_exact_known_tail_and_boundary_exclusion(self):
        demo = np.arange(65) < 48
        result = sidecar.plan_demo_tail(demo, [0, 16, 32, 48, 64])
        self.assertEqual(result["frames"], list(range(33, 48)))
        self.assertEqual(result["last_canonical_demo"], 32)
        self.assertNotIn(48, result["frames"])

    def test_short_demo_no_demo_and_already_dense(self):
        self.assertEqual(sidecar.plan_demo_tail(np.arange(20) < 3, [0, 3, 19])["frames"], [1, 2])
        self.assertEqual(sidecar.plan_demo_tail(np.zeros(20, dtype=bool), [0, 16, 19])["frames"], [])
        self.assertEqual(sidecar.plan_demo_tail(np.arange(20) < 3, [0, 1, 2, 3, 19])["frames"], [])

    def test_invalid_flags_order_and_types(self):
        for demo, frames in ((np.array([True, False, True]), [0, 1, 2]),
                             (np.ones(3, dtype=bool), [0, 2]),
                             (np.array([1, 1, 0]), [0, 2]),
                             (np.arange(4) < 2, [0, 0, 3]),
                             (np.arange(4) < 2, [0.0, 3.0]),
                             (np.arange(4) < 2, [-1, 3])):
            with self.assertRaises(ValueError):
                sidecar.plan_demo_tail(demo, frames)


class ExtractionTests(unittest.TestCase):
    def test_actual_eagle_metadata_is_validated_then_excluded_from_model_inputs(self):
        prepared = sidecar.prepare_visual_inputs(FakeListProcessor(), rgb_pair(), "TASK!")
        self.assertEqual(set(prepared), sidecar.INPUT_KEYS)
        self.assertNotIn("image_sizes", prepared)
        self.assertIs(type(prepared["pixel_values"]), list)
        self.assertEqual([value.item() for value in prepared["pixel_values"]], [7., 13.])

    def test_metadata_shape_and_unknown_gt_fields_fail_informatively(self):
        processor = FakeListProcessor()
        native = processor.collator
        for field in ("state", "action", "targets", "image_grid_thw", "unrecognized_metadata"):
            processor.collator = lambda values, field=field: {
                "inputs": {**native(values)["inputs"], field: torch.zeros(1)}}
            with self.assertRaisesRegex(ValueError, field):
                sidecar.prepare_visual_inputs(processor, rgb_pair(), "TASK")
        for value in (torch.ones(2, 2), torch.ones(1, 2, dtype=torch.int64),
                      torch.zeros(2, 2, dtype=torch.int64)):
            processor.collator = lambda values, value=value: {
                "inputs": {**native(values)["inputs"], "image_sizes": value}}
            with self.assertRaisesRegex(ValueError, "image_sizes"):
                sidecar.prepare_visual_inputs(processor, rgb_pair(), "TASK")

    def test_nested_container_cast_matches_original_tree_map_structure(self):
        import tree

        original = {"input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
                    "attention_mask": torch.tensor([[True, False]]),
                    "pixel_values": [torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
                                     (torch.tensor([7.5]), [torch.tensor([13.25], dtype=torch.float64)])]}
        native = tree.map_structure(lambda x: x.to("cpu", dtype=torch.bfloat16)
                                    if torch.is_floating_point(x) else x.to("cpu"), original)
        moved = sidecar.move_visual_inputs(original, "cpu")
        self.assertIs(type(moved["pixel_values"]), list)
        self.assertIs(type(moved["pixel_values"][1]), tuple)
        self.assertIs(type(moved["pixel_values"][1][1]), list)
        self.assertEqual(moved["input_ids"].dtype, torch.int64)
        self.assertEqual(moved["attention_mask"].dtype, torch.bool)
        self.assertEqual(sidecar.input_tree_signature(moved), sidecar.input_tree_signature(native))
        self.assertEqual(original["pixel_values"][0].dtype, torch.float32)

    def test_signature_binds_list_tuple_order_shape_dtype_and_bytes(self):
        left, right = torch.tensor([7.]), torch.tensor([13.])
        signature = sidecar.input_tree_signature([left, right])
        sidecar.validate_input_signature(signature)
        for alternative in ((left, right), [right, left], [left.reshape(1, 1), right],
                            [left.to(torch.bfloat16), right], [left + 1, right]):
            self.assertNotEqual(signature, sidecar.input_tree_signature(alternative))
        with self.assertRaisesRegex(ValueError, "Non-tensor"):
            sidecar.input_tree_signature([left, 3.0])
        with self.assertRaisesRegex(ValueError, "tensor leaves"):
            sidecar.move_visual_inputs({"input_ids": left, "attention_mask": right, "pixel_values": [3.0]}, "cpu")
        with self.assertRaisesRegex(ValueError, "sequence"):
            sidecar.validate_input_signature({"container": "list", "items": {}})

    def test_actual_list_shape_extraction_preserves_camera_boundaries(self):
        backbone = FakeBackbone().eval()
        images, proof = sidecar.extract_image_features(backbone, torch.nn.Identity().eval(),
                                                        FakeListProcessor(), rgb_pair(), "TASK!", device="cpu")
        self.assertIs(type(backbone.inputs["pixel_values"]), list)
        self.assertEqual(len(backbone.inputs["pixel_values"]), 2)
        self.assertTrue(all(value.shape == (1,) and value.dtype == torch.bfloat16
                            for value in backbone.inputs["pixel_values"]))
        self.assertTrue(bool((images[0] == 7).all() and (images[1] == 13).all()))
        self.assertEqual(proof["prepared"]["pixel_values"]["container"], "list")
        self.assertEqual(proof["encoder"]["pixel_values"]["items"][0]["dtype"], "torch.bfloat16")
        for stage in proof.values():
            for signature in stage.values():
                sidecar.validate_input_signature(signature)

    def test_preparation_exact_language_order_eval_and_only_visual(self):
        processor = FakeProcessor()
        batch = sidecar.prepare_visual_inputs(processor, rgb_pair(), "Café: Move, LEFT!")
        self.assertEqual(set(batch), sidecar.INPUT_KEYS)
        order, masks, transform, text = processor.calls[0]
        self.assertEqual(order, list(sidecar.CAMERA_ORDER))
        self.assertIsNone(masks)
        self.assertIs(transform, processor.eval_image_transform)
        self.assertEqual(text, "café move left")
        self.assertEqual(batch["pixel_values"].tolist(), [7, 13])

    def test_preparation_rejects_wrong_camera_training_and_extra_input(self):
        processor = FakeProcessor()
        processor.training = True
        with self.assertRaisesRegex(ValueError, "eval"):
            sidecar.prepare_visual_inputs(processor, rgb_pair(), "task")
        processor.training = False
        wrong = rgb_pair()
        wrong["state"] = [np.zeros(3)]
        with self.assertRaisesRegex(ValueError, "Camera"):
            sidecar.prepare_visual_inputs(processor, wrong, "task")
        native = processor.collator
        processor.collator = lambda values: {"inputs": {**native(values)["inputs"], "action": torch.zeros(1)}}
        with self.assertRaisesRegex(ValueError, "Nonvisual"):
            sidecar.prepare_visual_inputs(processor, rgb_pair(), "task")

    def test_frozen_backbone_vlln_only_and_bf16_camera_mapping(self):
        backbone = FakeBackbone().eval()
        vlln = torch.nn.Identity().eval()
        before = torch.random.get_rng_state().clone()
        images, proof = sidecar.extract_image_features(backbone, vlln, FakeProcessor(), rgb_pair(), "TASK!", device="cpu")
        self.assertEqual(images.dtype, torch.bfloat16)
        self.assertEqual(images.shape, (2, 81, 2048))
        self.assertTrue(bool((images[0] == 7).all() and (images[1] == 13).all()))
        self.assertEqual(set(backbone.inputs), sidecar.INPUT_KEYS)
        self.assertEqual(backbone.inputs["pixel_values"].dtype, torch.bfloat16)
        self.assertEqual(set(proof), {"prepared", "encoder"})
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        backbone.train()
        with self.assertRaisesRegex(ValueError, "frozen"):
            sidecar.extract_image_features(backbone, vlln, FakeProcessor(), rgb_pair(), "TASK", device="cpu")

    def test_masks_variable_absolute_position_and_bad_layout(self):
        features = torch.arange(230 * 2048, dtype=torch.float32).reshape(1, 230, 2048)
        mask = torch.zeros(1, 230, dtype=torch.bool)
        mask[:, 17:98] = True
        mask[:, 110:191] = True
        attention = torch.ones_like(mask)
        result = sidecar.image_tokens(features, mask, attention)
        self.assertTrue(torch.equal(result[0], features[0, 17:98]))
        self.assertTrue(torch.equal(result[1], features[0, 110:191]))
        attention[:, 18] = False
        with self.assertRaisesRegex(ValueError, "outside"):
            sidecar.image_tokens(features, mask, attention)
        attention.fill_(True)
        mask[:, 18] = False
        mask[:, 99] = True
        with self.assertRaisesRegex(ValueError, "camera runs"):
            sidecar.image_tokens(features, mask, attention)

    def test_explicit_observations_reject_gt_and_empty(self):
        backbone, norm = FakeBackbone().eval(), torch.nn.Identity().eval()
        values, proof = sidecar.extract_observations(backbone, norm, FakeProcessor(), [], device="cpu")
        self.assertEqual(values.shape, (0, 2, 81, 2048))
        self.assertEqual(proof, [])
        with self.assertRaisesRegex(ValueError, "Only RGB"):
            sidecar.extract_observations(backbone, norm, FakeProcessor(),
                                         [{"frame": 1, "images": rgb_pair(), "text": "x", "actions": object()}], device="cpu")

    def test_comparison_reports_not_accepts_numeric_difference(self):
        reference = torch.ones(2, dtype=torch.bfloat16)
        exact = sidecar.compare_features(reference, reference.clone())
        self.assertTrue(exact["exact_equal"])
        changed = sidecar.compare_features(reference * 2, reference)
        self.assertEqual(changed["max_abs"], 1)
        self.assertEqual(changed["acceptance"], "not_decided")
        self.assertFalse(changed["exact_equal"])
        self.assertIsNone(sidecar.compare_features(torch.zeros(2), torch.zeros(2))["relative_l2"])


class PayloadTests(unittest.TestCase):
    def test_exact_payload_and_empty(self):
        validate(tiny_payload())
        record = {"episode_id": 0, "frames": [], "canonical_frames": [0, 3],
                  "n_demo": 0, "last_canonical_demo": -1}
        validate(tiny_payload(record), record)

    def test_no_extra_fields_bad_dtype_nonfinite_wrong_ids_or_current(self):
        for field, value in (("actions", torch.zeros(1)), ("episode_id", 1),
                             ("cache_fingerprint", "wrong"), ("sidecar_fingerprint", "wrong"),
                             ("frames", torch.tensor([1, 3])), ("is_demo", torch.tensor([True, False]))):
            payload = tiny_payload()
            payload[field] = value
            with self.assertRaises(ValueError):
                validate(payload)
        payload = tiny_payload()
        payload["images"] = payload["images"].float()
        with self.assertRaisesRegex(ValueError, "dtype"):
            validate(payload)
        payload = tiny_payload()
        payload["images"][0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            validate(payload)

    def _write_reader_fixture(self, directory):
        root = Path(directory)
        record = tiny_record()
        plan = {"episodes": [record], "cache_fingerprint": "cache", "camera_order": list(sidecar.CAMERA_ORDER),
                "grid": [9, 9], "width": 2048, "feature_point": sidecar.FEATURE_POINT}
        payload = tiny_payload()
        payload["sidecar_fingerprint"] = sidecar.digest(plan)
        torch.save(payload, root / "episode.pt")
        hashes = {key: sidecar.tensor_signature(torch.zeros(1)) for key in sidecar.INPUT_KEYS}
        manifest = {"format_version": 1, "kind": sidecar.KIND, "status": "complete", "plan": plan,
                    "fingerprint": sidecar.digest(plan), "checks": dict.fromkeys(sidecar.CHECK_KEYS, True), "completed_episodes": [0],
                    "episodes": [{**record, "path": "episode.pt", "payload_sha256": sidecar.file_sha256(root / "episode.pt"),
                                  "preparation": [{"frame": f, "prepared": hashes, "encoder": hashes} for f in (1, 2)]}]}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return root, manifest

    def test_reader_hash_complete_binding_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self._write_reader_fixture(directory)
            reader = sidecar.DemoTailSidecar(root, expected_cache_fingerprint="cache")
            self.assertEqual(reader.load(0)["frames"].tolist(), [1, 2])
            with self.assertRaisesRegex(ValueError, "binding"):
                sidecar.DemoTailSidecar(root, expected_cache_fingerprint="wrong")
            manifest["status"] = "building"
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                sidecar.DemoTailSidecar(root)
            manifest["status"] = "complete"
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "episode.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                reader.load(0)
            (root / "episode.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                sidecar.DemoTailSidecar(root)

    def test_reader_rejects_missing_checks_and_preparation_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self._write_reader_fixture(directory)
            manifest["checks"] = {"imaginary_check": True}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "checks"):
                sidecar.DemoTailSidecar(root)
            manifest["checks"] = dict.fromkeys(sidecar.CHECK_KEYS, True)
            manifest["episodes"][0]["preparation"][0]["encoder"]["state"] = {}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "image/text"):
                sidecar.DemoTailSidecar(root)

    def test_reader_accepts_nested_tensor_signature_without_coercion(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self._write_reader_fixture(directory)
            for item in manifest["episodes"][0]["preparation"]:
                for stage in ("prepared", "encoder"):
                    item[stage]["pixel_values"] = sidecar.input_tree_signature(
                        [torch.ones(3, 2, 2), (torch.zeros(3, 2, 2),)])
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(sidecar.DemoTailSidecar(root).load(0)["frames"].tolist(), [1, 2])


class PlanTests(unittest.TestCase):
    def _fixture(self, directory):
        import pandas as pd

        root = Path(directory)
        base, cache, dataset = root / "base", root / "cache", root / "dataset"
        for folder in (base, cache, dataset / "meta", dataset / "data", dataset / "videos"):
            folder.mkdir(parents=True)
        config = {"hamlet_mode": "finetune", "memory_stride": 16, "n_moment_tokens": 4}
        processor = {"processor_kwargs": {"modality_configs": {"new_embodiment": {
            "video": {"modality_keys": list(sidecar.CAMERA_ORDER), "delta_indices": [-48, -32, -16, 0]},
            "language": {"modality_keys": ["annotation.task"], "delta_indices": [0]}}}}}
        (base / "config.json").write_text(json.dumps(config))
        (base / "processor_config.json").write_text(json.dumps(processor))
        (base / "model.safetensors").write_bytes(b"synthetic-placeholder-never-loaded")
        info = {"chunks_size": 1000, "data_path": "data/{episode_index}.parquet",
                "video_path": "videos/{video_key}-{episode_index}.mp4"}
        modality = {"annotation": {"task": {"original_key": "task_index"}},
                    "video": {camera: {"original_key": camera} for camera in sidecar.CAMERA_ORDER}}
        (dataset / "meta/info.json").write_text(json.dumps(info))
        (dataset / "meta/modality.json").write_text(json.dumps(modality))
        (dataset / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 4, "task": "Observe, THEN act!"}) + "\n")
        rows = [{"episode_index": eid, "length": 65, "tasks": ["Observe, THEN act!"]} for eid in range(13)]
        (dataset / "meta/episodes.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        signatures = [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
                       "sha256": sidecar.file_sha256(p)} for p in base.iterdir()]
        identity = {"checkpoint": signatures, "embodiment": "new_embodiment", "video_backend": "opencv"}
        fingerprint = sidecar._json_digest(identity)
        for eid in range(13):
            demo = np.arange(65) < (0 if eid == 1 else 48)
            pd.DataFrame({"is_demo": demo, "task_index": [4] * 65,
                          "action": ["POISON_GT"] * 65, "observation.state": ["POISON_STATE"] * 65}).to_parquet(dataset / f"data/{eid}.parquet")
            frames = sidecar.decision_frames(demo, 16)
            torch.save({"episode_id": eid, "cache_fingerprint": fingerprint,
                        "frames": torch.from_numpy(frames), "is_demo": torch.from_numpy(demo[frames])}, cache / f"{eid}.pt")
            for camera in sidecar.CAMERA_ORDER:
                (dataset / f"videos/{camera}-{eid}.mp4").write_bytes(b"synthetic-video-never-decoded")
        manifest = {"format_version": 1, "status": "complete", "identity": identity, "fingerprint": fingerprint,
                    "dataset_path": str(dataset), "model_path": str(base),
                    "splits": {"train": list(range(11)), "val": [11, 12]},
                    "episodes": [{"episode_id": eid, "path": f"{eid}.pt"} for eid in range(13)]}
        (cache / "manifest.json").write_text(json.dumps(manifest))
        return root, base, cache

    def _plan(self, cache, base):
        with patch("gr00t.utils.video_utils.resolve_backend", return_value="opencv"), \
                patch.object(sidecar, "source_identity", return_value={"fixture": "source"}):
            return sidecar.build_plan(cache, base)

    def test_deterministic_proof_subset_only_metadata_columns_no_model(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            root, base, cache = self._fixture(directory)
            with patch("pandas.read_parquet", wraps=pd.read_parquet) as reads:
                plan = self._plan(cache, base)
            self.assertEqual(plan["selection"]["train_proof_ids"], list(range(2, 10)))
            self.assertEqual(plan["episodes"][0]["role"], "known_ep0_proof")
            self.assertEqual(plan["episodes"][0]["frames"], list(range(33, 48)))
            self.assertEqual(plan["processor_video_delta_indices"], [-48, -32, -16, 0])
            self.assertEqual(plan["extracted_frame_offsets_per_observation"], [0])
            self.assertEqual(plan["selection"]["attempted_train_ids_in_order"], list(range(10)))
            self.assertTrue(all(call.kwargs["columns"] == ["is_demo", "task_index"] for call in reads.call_args_list))
            self.assertEqual(sidecar.digest(plan), sidecar.digest(self._plan(cache, base)))
            self.assertFalse((root / "output").exists())
            self.assertFalse(torch.cuda.is_initialized())

    def test_planning_rejects_stale_canonical_and_wrong_base_or_count(self):
        with tempfile.TemporaryDirectory() as directory:
            _, base, cache = self._fixture(directory)
            for kwargs in ({"proof_train_count": 7}, {"proof_train_count": 17}, {"proof_episode": 1}):
                with self.assertRaises(ValueError):
                    sidecar.build_plan(cache, base, **kwargs)
            payload = torch.load(cache / "0.pt", weights_only=True)
            payload["frames"][1] = 15
            torch.save(payload, cache / "0.pt")
            with self.assertRaisesRegex(ValueError, "canonical frames"):
                self._plan(cache, base)

    def test_explicit_raw_loader_and_future_instruction_no_gt_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            _, base, cache = self._fixture(directory)
            record = self._plan(cache, base)["episodes"][0]
            processor = FakeProcessor()
            loader = SimpleNamespace(_load_video_data=lambda eid, frames: {
                camera: np.stack([np.full((3, 4, 3), frame, dtype=np.uint8) for frame in frames])
                for camera in sidecar.CAMERA_ORDER})
            observations = sidecar.load_visual_observations(processor, loader, record, [32, 47, 48])
            self.assertEqual([o["frame"] for o in observations], [32, 47, 48])
            self.assertEqual(observations[0]["text"], "Observe, THEN act!")
            self.assertEqual(int(observations[1]["images"]["front_view"][0][0, 0, 0]), 47)
            self.assertTrue(all(set(o) == {"frame", "images", "text"} for o in observations))
            for frames in ([65], [2, 1], [1, 1], [1.0]):
                with self.assertRaisesRegex(ValueError, "raw frames"):
                    sidecar.load_visual_observations(processor, loader, record, frames)

    def test_new_output_and_atomic_failure_manifest_no_model_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root, base, cache = self._fixture(directory)
            plan = self._plan(cache, base)
            for path in (cache, cache / "new", base / "new"):
                with self.assertRaises(ValueError):
                    sidecar.validate_output_path(plan, path)
            output = root / "output"
            with patch.object(sidecar, "_check_inputs", return_value={"fixture": True}), \
                    patch("gr00t.long_memory.hamlet.load_frozen_hamlet", side_effect=RuntimeError("synthetic model failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic model failure"):
                    sidecar.build_sidecar(plan, output, device="cpu")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(list((output / "episodes").iterdir()), [])
            self.assertEqual(list(output.glob("*.tmp-*")), [])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                sidecar.DemoTailSidecar(output)

    def test_preflight_writes_nothing_and_never_calls_extractor(self):
        with tempfile.TemporaryDirectory() as directory:
            root, base, cache = self._fixture(directory)
            plan = self._plan(cache, base)
            output = root / "output"
            with patch.object(sidecar, "build_plan", return_value=plan), \
                    patch.object(sidecar, "build_sidecar", side_effect=AssertionError("must not extract")), \
                    patch("builtins.print") as printed:
                sidecar.main(["--cache-dir", str(cache), "--base-model", str(base),
                              "--output-dir", str(output), "--preflight-only"])
            result = json.loads(printed.call_args.args[0])
            self.assertTrue(result["passed"])
            self.assertFalse(result["cuda_initialized"])
            self.assertFalse(output.exists())

    def test_atomic_complete_roundtrip_with_fake_frozen_backbone_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, cache, dataset = (root / name for name in ("base", "cache", "dataset"))
            for path in (base, cache, dataset):
                path.mkdir()
            records = [{**tiny_record(), "episode_id": eid} for eid in range(9)]
            plan = {"scope": "proof_subset_only", "episodes": records, "cache_dir": str(cache),
                    "base_model": str(base), "dataset_path": str(dataset), "cache_fingerprint": "cache",
                    "embodiment": "new_embodiment", "video_backend": "opencv",
                    "camera_order": list(sidecar.CAMERA_ORDER), "grid": [9, 9], "width": 2048,
                    "feature_point": sidecar.FEATURE_POINT}
            model = torch.nn.Module()
            model.backbone = FakeBackbone()
            model.action_head = torch.nn.Module()
            model.action_head.vlln = torch.nn.Identity()
            model.action_head.process_backbone_output = lambda *a, **kw: self.fail("Short memory must not run")
            model.action_head.get_action = lambda *a, **kw: self.fail("Action head must not run")
            model.eval().requires_grad_(False)
            observations = [{"frame": frame, "images": rgb_pair(), "text": "Observe!"} for frame in (1, 2)]
            output = root / "output"
            checks = {key: True for key in ("input_files_unchanged", "sources_unchanged", "runtime_unchanged")}
            with patch.object(sidecar, "_check_inputs", return_value=checks), \
                    patch("gr00t.long_memory.hamlet.load_frozen_hamlet", return_value=(model, FakeListProcessor())), \
                    patch("gr00t.data.dataset.lerobot_episode_loader.LeRobotEpisodeLoader"), \
                    patch.object(sidecar, "load_visual_observations", return_value=observations):
                self.assertEqual(sidecar.build_sidecar(plan, output, device="cpu"), output)
            reader = sidecar.DemoTailSidecar(output, expected_cache_fingerprint="cache")
            self.assertEqual(reader.manifest["status"], "complete")
            self.assertEqual(len(reader.manifest["episodes"]), 9)
            self.assertEqual(set(reader.manifest["checks"]), sidecar.CHECK_KEYS)
            self.assertTrue(bool((reader.load(8)["images"][:, 1] == 13).all()))
            self.assertEqual(sorted(p.name for p in output.iterdir()), ["episodes", "manifest.json"])
            self.assertEqual(list(output.rglob("*.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
