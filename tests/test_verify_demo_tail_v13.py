"""CPU-only extraction-proof harness; no real checkpoint/cache/video loads."""
from contextlib import ExitStack
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from run_scripts.robomme import verify_demo_tail_v13 as proof


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, inputs):
        pixels = inputs["pixel_values"]
        if isinstance(pixels, (list, tuple)):
            # The real Siglip accepts per-image tensors and combines patches
            # inside its own forward. The verifier must not combine the list.
            pixels = torch.cat(pixels, dim=1)
        return {"backbone_features": pixels * self.scale,
                "image_mask": inputs["input_ids"] == 1,
                "backbone_attention_mask": inputs["attention_mask"].bool()}


class ToyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlln = nn.Identity()
        self.model = nn.Linear(4, 4)
        self._memory_cache = torch.tensor([3., 4.])

    def process_backbone_output(self, output, action_inputs_B=1):
        result = dict(output)
        result["backbone_features"] = output["backbone_features"].clone()
        result["backbone_features"][:, -4:] += 7
        self._memory_cache = self._memory_cache + 1
        return result


def toy(*, pixel_list=False):
    model = nn.Module()
    model.backbone, model.action_head = ToyBackbone(), ToyHead()
    model.eval().requires_grad_(False)
    ids = torch.zeros((1, 170), dtype=torch.long)
    ids[:, 1:82], ids[:, 84:165] = 1, 1
    data = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.arange(680).reshape(1, 170, 4).float() / 100}
    if pixel_list:
        data["pixel_values"] = list(data["pixel_values"].split([82, 88], dim=1))

    def native():
        prepared = proof.clone_tree(data)
        output = model.action_head.process_backbone_output(model.backbone(BatchFeature(data=prepared)))
        return prepared, output

    def prepare():
        return proof.clone_tree(data)

    def extract():
        output = model.backbone(prepare())
        return proof.image_selection(output)[0], {"prepared": "toy"}

    return model, data, native, prepare, extract


def plan_fixture():
    def record(eid, role):
        return {"episode_id": eid, "split": "train", "role": role,
                "canonical_frames": [0, 16, 32, 48, 64], "last_canonical_demo": 32,
                "frames": list(range(33, 48))}
    return {"episodes": [record(0, "known_ep0_proof"), record(2, "train_proof"), record(5, "train_proof"),
                          record(6, "train_proof")], "runtime": {}}


class HarnessTests(unittest.TestCase):
    def test_cpu_preparation_exact_list_and_rng_restoration(self):
        _, data, _, _, _ = toy(pixel_list=True)
        state = proof.rng_state()
        def native():
            torch.rand(3)
            return {**proof.input_clone(data), "state": torch.ones(1, 8)}
        row = proof.processor_comparison(native, lambda: proof.input_clone(data))
        self.assertTrue(row["prepared_inputs_exact"])
        self.assertIn("state", row["native_collated_keys"])
        self.assertNotIn("state", row["direct_model_keys"])
        self.assertEqual(row["direct_signatures"]["pixel_values"]["container"], "list")
        self.assertTrue(proof.tree_equal(state, proof.rng_state()))

    def test_cpu_preparation_mismatch_and_exception_restore_rng(self):
        _, data, _, _, _ = toy(pixel_list=True)
        wrong = proof.input_clone(data)
        wrong["input_ids"][0, 0] += 1
        self.assertFalse(proof.processor_comparison(lambda: data, lambda: wrong)["prepared_inputs_exact"])
        state = proof.rng_state()
        def fail():
            torch.rand(5)
            raise ValueError("broken metadata")
        with self.assertRaisesRegex(ValueError, "broken"):
            proof.processor_comparison(fail, lambda: data)
        self.assertTrue(proof.tree_equal(state, proof.rng_state()))

    def test_preparation_only_cannot_request_cuda(self):
        with self.assertRaisesRegex(ValueError, "requires --device cpu"):
            proof.main(["--cache-dir", "unused", "--base-model", "unused", "--output-dir", "unused",
                        "--preparation-only", "--device", "cuda:0"])

    def test_native_eagle_style_unequal_image_list_preserved_and_exact(self):
        model, data, native, prepare, extract = toy(pixel_list=True)
        row = proof.verify_frame(model, native, prepare, extract)
        self.assertTrue(row["structural_passed"], row)
        self.assertTrue(row["all_feature_comparisons_exact"])
        signature = row["prepared_input_signatures"]["pixel_values"]
        self.assertEqual(signature["container"], "list")
        self.assertEqual([child["shape"] for child in signature["items"]], [[1, 82, 4], [1, 88, 4]])
        self.assertEqual(row["comparisons"]["device_inputs_direct"]["pixel_values"]["container"], "list")

    def test_list_to_tensor_coercion_fails_even_with_equal_model_features(self):
        model, data, native, prepare, extract = toy(pixel_list=True)
        def flattened():
            batch = prepare()
            batch["pixel_values"] = torch.cat(batch["pixel_values"], dim=1)
            return proof.image_selection(model.backbone(batch))[0], {}
        row = proof.verify_frame(model, native, prepare, flattened)
        self.assertFalse(row["structural_passed"])
        self.assertFalse(row["checks"]["device_inputs_direct_exact"])
        self.assertTrue(row["comparisons"]["direct_vs_native"]["exact"])

    def test_input_tree_container_type_length_and_leaf_values_bound(self):
        original = [torch.arange(3), (torch.ones(2), {"image": torch.zeros(1)})]
        cloned = proof.input_clone(original)
        self.assertTrue(proof.compare_input_tree(original, cloned)["exact"])
        self.assertIs(type(cloned), list)
        self.assertIs(type(cloned[1]), tuple)
        self.assertNotEqual(proof.input_signature(original), proof.input_signature(tuple(original)))
        self.assertFalse(proof.compare_input_tree(original, tuple(original))["exact"])
        self.assertFalse(proof.compare_input_tree(original, original[:-1])["exact"])
        cloned[1][1]["image"] += 1
        self.assertFalse(proof.compare_input_tree(original, cloned)["exact"])
        self.assertEqual(float(original[1][1]["image"][0]), 0.)

    def test_tree_clone_does_not_accept_non_tensor_payload_leaves(self):
        with self.assertRaises(TypeError):
            proof.input_clone([torch.ones(1), "not an image tensor"])

    def test_native_direct_cache_exact_and_state_restoration(self):
        model, data, native, prepare, extract = toy()
        with proof.preserve_call_state(model.action_head), torch.no_grad():
            _, cached = native()
        old_cache, rng = model.action_head._memory_cache, proof.rng_state()
        row = proof.verify_frame(model, native, prepare, extract, cached=cached)
        self.assertTrue(row["structural_passed"], row)
        self.assertTrue(row["all_feature_comparisons_exact"])
        self.assertEqual(row["action_calls"], {"action_head_forward": 0, "action_dit_forward": 0})
        self.assertIs(model.action_head._memory_cache, old_cache)
        self.assertTrue(proof.tree_equal(rng, proof.rng_state()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_global_rng_consumption_reported_not_hidden(self):
        model, data, native, prepare, extract = toy()
        old = proof.rng_state()
        def consumes():
            torch.rand(3)
            np.random.rand()
            proof.random.random()
            return extract()
        row = proof.verify_frame(model, native, prepare, consumes)
        self.assertTrue(row["structural_passed"])
        self.assertFalse(row["raw_extraction_rng_unchanged"])
        self.assertTrue(row["checks"]["caller_rng_restored"])
        self.assertTrue(proof.tree_equal(old, proof.rng_state()))

    def test_prepared_pixel_mismatch_is_structural_failure(self):
        model, data, native, prepare, extract = toy()
        def wrong_prepare():
            value = prepare()
            value["pixel_values"] += 0.1
            return value
        row = proof.verify_frame(model, native, wrong_prepare, extract)
        self.assertFalse(row["structural_passed"])
        self.assertFalse(row["checks"]["prepared_direct_exact"])

    def test_wrong_actual_cast_inputs_is_structural_failure(self):
        model, data, native, prepare, extract = toy()
        def wrong_extract():
            value = prepare()
            value["pixel_values"] += 0.1
            output = model.backbone(value)
            return proof.image_selection(output)[0], {}
        row = proof.verify_frame(model, native, prepare, wrong_extract)
        self.assertFalse(row["checks"]["device_inputs_direct_exact"])
        self.assertFalse(row["all_feature_comparisons_exact"])

    def test_nonexact_features_are_not_arbitrarily_accepted(self):
        model, data, native, prepare, extract = toy()
        def offset():
            images, hashes = extract()
            return images + 0.1, hashes
        row = proof.verify_frame(model, native, prepare, offset)
        self.assertTrue(row["structural_passed"])
        self.assertFalse(row["all_feature_comparisons_exact"])
        self.assertGreater(row["comparisons"]["direct_vs_native"]["max_abs"], 0.)

    def test_direct_cache_mutation_detected_and_restored(self):
        model, data, native, prepare, extract = toy()
        cache = model.action_head._memory_cache
        before = cache.clone()
        def mutate():
            cache.add_(1)
            return extract()
        row = proof.verify_frame(model, native, prepare, mutate)
        self.assertFalse(row["checks"]["direct_memory_cache_exact"])
        self.assertIs(model.action_head._memory_cache, cache)
        self.assertTrue(torch.equal(cache, before))

    def test_frozen_parameter_mutation_detected(self):
        model, data, native, prepare, extract = toy()
        def mutate():
            model.backbone.scale.add_(1)
            return extract()
        row = proof.verify_frame(model, native, prepare, mutate)
        self.assertFalse(row["checks"]["frozen_versions_and_grad_flags_unchanged"])

    def test_action_and_head_processing_calls_forbidden_and_hooks_restored(self):
        for illegal in ("action", "processing"):
            model, data, native, prepare, extract = toy()
            cache, rng = model.action_head._memory_cache, proof.rng_state()
            def bad():
                if illegal == "action":
                    model.action_head.model(torch.zeros(1, 4))
                else:
                    model.action_head.process_backbone_output({})
                return extract()
            with self.assertRaises(RuntimeError):
                proof.verify_frame(model, native, prepare, bad)
            self.assertIs(model.action_head._memory_cache, cache)
            self.assertTrue(proof.tree_equal(rng, proof.rng_state()))
            self.assertFalse(model.backbone._forward_pre_hooks)
            self.assertFalse(model.action_head.model._forward_pre_hooks)
            self.assertEqual(model.action_head.process_backbone_output.__func__, ToyHead.process_backbone_output)

    def test_requires_grad_or_training_rejected(self):
        model, data, native, prepare, extract = toy()
        model.backbone.scale.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            proof.verify_frame(model, native, prepare, extract)

    def test_nonfinite_and_bad_geometry_are_rejected(self):
        model, data, native, prepare, extract = toy()
        data["pixel_values"][0, 1, 0] = float("nan")
        row = proof.verify_frame(model, native, prepare, extract)
        self.assertFalse(row["checks"]["finite_features"])
        data["input_ids"][0, -1] = 1
        with self.assertRaisesRegex(ValueError, "overlap"):
            proof.verify_frame(model, native, prepare, extract)

    def test_zero_reference_relative_error_undefined(self):
        result = proof.numerical_comparison(torch.zeros(3), torch.ones(3))
        self.assertIsNone(result["relative_l2"])
        self.assertEqual(result["max_abs"], 1.)
        self.assertFalse(result["exact"])

    def test_nonexact_complete_proof_requires_review_exit_not_success(self):
        self.assertEqual(proof.completion_code({"structural_passed": True, "all_feature_comparisons_exact": True}), 0)
        self.assertEqual(proof.completion_code({"structural_passed": True, "all_feature_comparisons_exact": False}), 2)
        self.assertEqual(proof.completion_code({"structural_passed": False, "all_feature_comparisons_exact": True}), 1)

    def test_inference_tensor_cache_restores_original_identity(self):
        model, data, native, prepare, extract = toy()
        with torch.inference_mode():
            model.action_head._memory_cache = torch.tensor([3., 4.])
        original = model.action_head._memory_cache
        row = proof.verify_frame(model, native, prepare, extract)
        self.assertTrue(row["structural_passed"])
        self.assertIs(model.action_head._memory_cache, original)
        rng = proof.rng_state()
        def bad():
            torch.rand(3)
            raise RuntimeError("deliberate inference-cache exception")
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            proof.verify_frame(model, native, prepare, bad)
        self.assertIs(model.action_head._memory_cache, original)
        self.assertTrue(proof.tree_equal(rng, proof.rng_state()))

    def test_unreadable_integrity_file_does_not_mask_original_error(self):
        report = {"checks": {}, "error": {"type": "RuntimeError", "message": "original"}}
        def missing():
            raise FileNotFoundError("protected input removed")
        proof.guarded_check(report, "files", missing)
        self.assertFalse(report["checks"]["files"])
        self.assertEqual(report["error"]["message"], "original")
        self.assertEqual(report["integrity_errors"][0]["type"], "FileNotFoundError")

    def test_proof_selection_fixed_train_and_excludes_current(self):
        plan = plan_fixture()
        rows = proof.proof_frames(plan)
        self.assertEqual(len(rows), 12)
        self.assertEqual([r["frame"] for r in rows if r["kind"] == "tail_ingestion_example"], list(range(42, 48)))
        self.assertEqual({r["episode_id"] for r in rows}, {0, 2, 5})
        self.assertNotIn(48, [r["frame"] for r in rows])
        plan["episodes"][0]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            proof.proof_frames(plan)

    def test_short_demo_cannot_silently_reduce_predeclared_12_rows(self):
        plan = plan_fixture()
        plan["episodes"][0]["frames"] = [33, 34]
        with self.assertRaisesRegex(ValueError, "exactly 12"):
            proof.proof_frames(plan)

    def test_observed_states_reads_no_action_column(self):
        import pandas as pd
        processor = SimpleNamespace(modality_configs={"new_embodiment": {"state": SimpleNamespace(modality_keys=["arm"])}})
        loader = SimpleNamespace(modality_meta={"state": {"arm": {"original_key": "observation.state"}}},
                                 _extract_joint_groups=lambda df, keys, kind: pd.DataFrame({"arm": [np.array([1., 2.])] }))
        with patch("pandas.read_parquet", return_value=pd.DataFrame({"observation.state": [np.array([1., 2.])] })) as read:
            states = proof.observed_states(loader, processor, {"parquet_path": "not-read"}, 0, "new_embodiment")
        self.assertEqual(read.call_args.kwargs["columns"], ["observation.state"])
        self.assertEqual(states["arm"].shape, (1, 2))

    def test_metadata_preflight_no_model_output_or_cuda(self):
        from run_scripts.robomme import demo_tail_sidecar_v13 as sidecar
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("cache", "base", "dataset"):
                (root / name).mkdir()
            output = root / "new-output"
            plan = plan_fixture()
            plan.update(dataset_path=str(root / "dataset"), files_sha256={})
            cache = SimpleNamespace(path=root / "cache", manifest={"model_path": str(root / "base")})
            with ExitStack() as stack:
                stack.enter_context(patch("gr00t.long_memory.cache.EpisodeCache", return_value=cache))
                stack.enter_context(patch("gr00t.long_memory.hamlet.validate_cache_checkpoint"))
                model = stack.enter_context(patch("gr00t.long_memory.hamlet.load_frozen_hamlet", side_effect=AssertionError("no model")))
                stack.enter_context(patch.object(sidecar, "build_plan", return_value=plan))
                stack.enter_context(patch.object(sidecar, "runtime_identity", return_value={}))
                stack.enter_context(patch.object(proof, "source_hashes", return_value={}))
                stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
                cuda = stack.enter_context(patch("torch.cuda.init", side_effect=AssertionError("no CUDA")))
                code = proof.main(["--cache-dir", str(cache.path), "--base-model", str(root / "base"),
                                   "--output-dir", str(output), "--preflight-only"])
                self.assertEqual(code, 0)
                model.assert_not_called()
                cuda.assert_not_called()
                self.assertFalse(output.exists())
                output.mkdir()
                with self.assertRaises(FileExistsError):
                    proof.main(["--cache-dir", str(cache.path), "--base-model", str(root / "base"),
                                "--output-dir", str(output), "--preflight-only"])

    def test_model_failure_and_final_hash_failure_preserve_report(self):
        import json
        from run_scripts.robomme import demo_tail_sidecar_v13 as sidecar
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("cache", "base", "dataset"):
                (root / name).mkdir()
            output = root / "failed-proof"
            plan = plan_fixture()
            plan.update(dataset_path=str(root / "dataset"), files_sha256={})
            cache = SimpleNamespace(path=root / "cache", manifest={"model_path": str(root / "base")})
            with ExitStack() as stack:
                stack.enter_context(patch("gr00t.long_memory.cache.EpisodeCache", return_value=cache))
                stack.enter_context(patch("gr00t.long_memory.hamlet.validate_cache_checkpoint"))
                stack.enter_context(patch("gr00t.long_memory.hamlet.load_frozen_hamlet", side_effect=RuntimeError("original model load failure")))
                stack.enter_context(patch.object(sidecar, "build_plan", return_value=plan))
                stack.enter_context(patch.object(sidecar, "runtime_identity", return_value={}))
                stack.enter_context(patch.object(proof, "source_hashes", side_effect=[{}, FileNotFoundError("removed source")]))
                stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
                code = proof.main(["--cache-dir", str(cache.path), "--base-model", str(root / "base"), "--output-dir", str(output)])
            saved = json.loads((output / "result.json").read_text())
            self.assertEqual(code, 1)
            self.assertEqual(saved["error"]["message"], "original model load failure")
            self.assertFalse(saved["checks"]["source_unchanged"])
            self.assertEqual(saved["integrity_errors"][0]["message"], "removed source")


if __name__ == "__main__":
    unittest.main()
