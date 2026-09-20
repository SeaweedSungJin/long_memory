"""Synthetic CPU tests only; no real sidecar, checkpoint, Action Expert or GPU."""
from contextlib import ExitStack
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from run_scripts.robomme import verify_demo_tail_replay_v13 as probe
from tests import test_visual_demo_tail_bank_v13 as fixtures
from tests.test_replay_visual_patch_v11 import GuardedColumn, ObservationOnlyMapping


def episode(eid=1):
    value = fixtures.source()
    count = len(value["frames"]) - 1
    value.update(episode_id=eid, short=torch.stack([row[-4:] for row in value["features"]]),
        frames=torch.tensor(value["frames"]), is_demo=torch.tensor(value["is_demo"]),
        state=torch.zeros(count, 8), targets=torch.full((count, 2, 8), .2),
        target_mask=torch.ones(count, 2, 8, dtype=torch.bool), actions=torch.zeros(count, 16, 8),
        action_mask=torch.ones(count, 16, dtype=torch.bool), decision_mask=torch.tensor([False, False, False, True, True]),
        embodiment_id=0)
    return value


def case(eid=1):
    payload, record = fixtures.sidecar()
    payload["episode_id"] = record["episode_id"] = eid
    return {"episode_id": eid, "query": 3, "frame": 48, "episode": episode(eid), "payload": payload, "record": record}


def bindings(eid=1):
    return {"episode_id": eid, "cache_fingerprint": fixtures.CACHE, "sidecar_fingerprint": fixtures.SIDECAR}


class ReplayProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_bank_helpers_causal_raw_frames_and_sequential_repeat(self):
        item, model = case(), fixtures.memory(awake=True)
        original = item["episode"]
        protected = ObservationOnlyMapping({key: GuardedColumn(original[key], 3) for key in probe.native.OBSERVATION_KEYS})
        protected.update(targets=object(), actions=object(), state=object(), parent=object())
        obs, payload, _ = probe.observation_view(protected, 3, item["payload"])
        rng = torch.get_rng_state().clone()
        with patch.object(probe, "replay_demo_tail", wraps=probe.replay_demo_tail) as production:
            features, banks, comparisons = probe.replay_pair(model, obs, 3, payload, item["record"], bindings())
        self.assertEqual([call.kwargs["include_tail"] for call in production.call_args_list], [True, False, True, True])
        self.assertIs(production.call_args_list[2].kwargs["visual_read_enabled"], False)
        self.assertEqual(banks["canonical"].frames.tolist(), [[0, 16, 32]])
        self.assertEqual(banks["merged"].frames.tolist(), [[0, 16, 32, *range(33, 48)]])
        self.assertTrue(torch.equal(features["off"][0], original["features"][3]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(c["finite"] and c["same_shape"] for c in comparisons.values()))
        self.assertTrue(all(value["exact"] for value in comparisons.values()))
        self.assertTrue(torch.equal(features["merged"], features["sequential"]))
        self.assertNotEqual(banks["canonical"].tokens.shape, banks["merged"].tokens.shape)

    def test_comparator_uses_actual_canonical_online_path_and_only_tail_bank_images(self):
        item, model = case(), fixtures.memory(awake=True)
        obs, payload, _ = probe.observation_view(item["episode"], 3, item["payload"])
        frames = [0, 16, 32, *range(33, 48)]
        with patch.object(model, "encode_observation", wraps=model.encode_observation) as ordinary, \
                patch.object(model, "append", wraps=model.append) as append, \
                patch.object(model, "encode_bank_images", wraps=model.encode_bank_images) as tail, \
                patch.object(model, "append_bank_images", wraps=model.append_bank_images) as append_tail:
            bank = probe.sequential_bank(model, obs, 3, payload, frames)
        self.assertEqual(ordinary.call_count, 3)
        self.assertEqual(append.call_count, 3)
        self.assertEqual(tail.call_count, 15)
        self.assertEqual(append_tail.call_count, 15)
        self.assertEqual(bank.frames[0].tolist(), frames)
        for index, call in enumerate(ordinary.call_args_list):
            self.assertTrue(torch.equal(call.args[0][0], obs["features"][index]))
            self.assertTrue(torch.equal(call.args[1][0], obs["image_masks"][index]))
            self.assertTrue(torch.equal(call.args[2][0], obs["attention_masks"][index]))
        self.assertTrue(all(call.args[0].shape == (1, 2, 81, 8) for call in tail.call_args_list))

    def test_new_framewise_dependency_is_in_source_identity(self):
        path = "run_scripts/robomme/framewise_demo_tail_v13.py"
        self.assertEqual(probe.source_hashes()[path], probe.native.sha(probe.ROOT / path))

    def test_grad_input_leaves_are_distinct_and_originals_unchanged(self):
        item, model = case(), fixtures.memory(awake=True)
        saved = copy.deepcopy(item)
        obs, payload, leaves = probe.observation_view(item["episode"], 3, item["payload"], track=True)
        features, _, _ = probe.replay_pair(model, obs, 3, payload, item["record"], bindings())
        features["merged"].float().square().mean().backward()
        self.assertEqual(set(leaves), {"tail_images", "earliest_canonical_image", "current_image"})
        for leaf in leaves.values():
            self.assertIsNotNone(leaf.grad)
            self.assertTrue(torch.isfinite(leaf.grad).all())
            self.assertGreater(int(leaf.grad.count_nonzero()), 0)
        self.assertTrue(torch.equal(item["payload"]["images"], saved["payload"]["images"]))
        for left, right in zip(item["episode"]["features"], saved["episode"]["features"]):
            self.assertTrue(torch.equal(left, right))
            self.assertIsNone(left.grad)

    def test_duplicate_current_or_future_tail_rejected_before_read(self):
        item, model = case(), fixtures.memory()
        obs, payload, _ = probe.observation_view(item["episode"], 3, item["payload"])
        for frame in (32, 48, 64):
            bad = copy.deepcopy(payload); bad["frames"][-1] = frame
            with self.assertRaises(ValueError), patch.object(model, "read", side_effect=AssertionError("No READ on invalid input")):
                probe.replay_pair(model, obs, 3, bad, item["record"], bindings())

    def test_prepared_case_records_original_query_and_complete_strict_past(self):
        item = case()
        metadata = probe.case_metadata(item)
        self.assertEqual(metadata["query"], 3)
        self.assertEqual(metadata["frame"], 48)
        self.assertEqual(metadata["bank_observation_counts"], {"canonical": 3, "merged": 18})
        self.assertEqual(metadata["bank_token_counts"]["merged"], 18 * 162)
        self.assertNotIn(48, metadata["merged_frames"])
        self.assertIn("targets", metadata["current_query_sha256"])
        item["payload"]["frames"][-1] = 48
        with self.assertRaises(ValueError):
            probe.case_metadata(item)

    def test_bf16_nextafter_is_per_element_directional_measurement(self):
        reference = torch.tensor([1., -1., 0., 2., .5], dtype=torch.bfloat16)
        direction = torch.tensor([2., 0., -1., 3., 0.], dtype=torch.bfloat16)
        once = torch.nextafter(reference, direction)
        candidate = once.clone()
        candidate[3] = torch.nextafter(once[3], direction[3])
        candidate[4] = reference[4]
        row = probe.bf16_ulp_measurements(reference, candidate)
        self.assertEqual(row["differing_values"], 4)
        self.assertEqual(row["within_one_directional_ulp"], 3)
        self.assertEqual(row["beyond_one_directional_ulp"], 1)
        self.assertEqual(row["max_directional_ulp_ratio"], 2.)
        self.assertEqual(row["examples"][0]["one_ulp_toward_candidate"], .0078125)
        self.assertEqual(row["examples"][1]["one_ulp_toward_candidate"], .00390625)
        self.assertGreater(row["examples"][2]["one_ulp_toward_candidate"], 0.)
        self.assertEqual(probe.bf16_ulp_measurements(reference, reference)["differing_values"], 0)
        with self.assertRaises(ValueError):
            probe.bf16_ulp_measurements(reference.float(), candidate)
        with self.assertRaises(FloatingPointError):
            probe.bf16_ulp_measurements(reference, torch.full_like(reference, float("nan")))

    def test_first_execution_selection_train_only_no_skip_invalid_query(self):
        cases = {eid: case(eid) for eid in (0, 1, 3)}
        records = [{**cases[eid]["record"], "split": "train", "role": "known_ep0_proof" if eid == 0 else "train_proof"}
                   for eid in (0, 1, 3)]
        reader = SimpleNamespace(manifest={"plan": {"episodes": records}}, _records={r["episode_id"]: r for r in records},
                                 load=lambda eid: cases[eid]["payload"])
        cache = SimpleNamespace(manifest={"splits": {"train": [0, 1, 3], "val": [4]}})
        episodes = SimpleNamespace(fetch=lambda eid: cases[eid]["episode"])
        selected = probe.select_cases(cache, reader, episodes)
        self.assertEqual([(r["episode_id"], r["query"]) for r in selected], [(1, 3), (3, 3), (0, 3)])
        cases[1]["episode"]["decision_mask"][3] = False
        with self.assertRaises(ValueError):
            probe.select_cases(cache, reader, episodes)
        cases[1]["episode"]["decision_mask"][3] = True
        records[1]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "TRAIN-only"):
            probe.select_cases(cache, reader, episodes)

    def run_toy(self, *, updates=2, broken=False):
        item, visual = case(), fixtures.memory()
        ep, q = item["episode"], item["query"]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            head = nn.Linear(8, 8, bias=False).bfloat16().eval().requires_grad_(False)
            parent = nn.Linear(8, 8).eval().requires_grad_(False)
        head.num_inference_timesteps = 4
        before = {name: probe.native.module_digest(module) for name, module in (("head", head), ("parent", parent))}
        fused = ep["features"][q][None, -4:].float() + .125
        def cached(head, ep, query, short=None):
            feature = ep["features"][query][None]
            return (probe.native.replace_short(feature, short) if short is not None else feature), None, None, None
        def flow(head, ep, query, features, *args, **kwargs):
            prediction = head(features).float().mean(1)[:, None].expand(-1, 2, -1)
            return {"prediction": prediction, "loss": (prediction - ep["targets"][query][None]).square().mean()}
        def original(head, ep, query, short=None, **kwargs):
            return flow(head, ep, query, cached(head, ep, query, short)[0])
        def generated(head, ep, query, short=None, **kwargs):
            value = head(cached(head, ep, query, short)[0]).float().mean(1)[:, None]
            return value + 1 if broken and isinstance(ep["features"], dict) else value
        report, snapshots = {}, []
        with ExitStack() as stack:
            for name, replacement in (("cached_inputs", cached), ("flow_at_features", flow),
                                      ("expert_episode_flow_loss", original), ("generated_action", generated)):
                stack.enter_context(patch.object(probe.native, name, side_effect=replacement))
            stack.enter_context(patch.object(probe.native, "replay_queries", return_value={q: (fused, {})}))
            stack.enter_context(patch.object(probe.native, "sample_noise_time", return_value=(torch.zeros(1, 2, 8).bfloat16(), torch.full((1, 1, 1), .5).bfloat16())))
            if broken:
                with self.assertRaisesRegex(RuntimeError, "records preserved"):
                    probe.diagnose_case(head, parent, item, visual, bindings(), 9111, report,
                                        lambda: snapshots.append(copy.deepcopy(report)), updates=updates)
            else:
                probe.diagnose_case(head, parent, item, visual, bindings(), 9111, report,
                                    lambda: snapshots.append(copy.deepcopy(report)), updates=updates)
        after = {name: probe.native.module_digest(module) for name, module in (("head", head), ("parent", parent))}
        self.assertEqual(before, after)
        self.assertTrue(all(p.grad is None for module in (head, parent) for p in module.parameters()))
        return report, snapshots

    def test_exactly_two_updates_all14_gradients_and_parameter_deltas_reported(self):
        report, snapshots = self.run_toy()
        self.assertEqual(report["optimizer_updates"], 2)
        self.assertEqual([r["after_updates"] for r in report["passes"]], [0, 1, 2])
        self.assertTrue(all(report["checks"].values()), report["checks"])
        self.assertEqual(len(report["optimizer_parameter_names"]), 14)
        for row in report["passes"]:
            self.assertEqual(len(row["parameter_gradients"]), 14)
            self.assertEqual(len(row["parameter_changes_from_initial"]), 14)
            self.assertIn("sequential", row["generated_comparisons"])
            self.assertTrue(row["generated_production_vs_online"]["exact"])
            self.assertTrue(all(value["exact"] for value in row["comparisons"].values()))
            self.assertEqual(row["replay_encoding"], "framewise")
            self.assertEqual(set(row["generated_tensor_sha256"]), {"off", "merged", "canonical", "sequential"})
            self.assertEqual(row["read_bf16_ulp"]["dtype"], "torch.bfloat16")
        initial, awake = report["passes"][0], report["passes"][2]
        self.assertEqual(initial["parameter_gradients"]["image_projection.weight"]["nonzero"], 0)
        self.assertGreater(awake["parameter_gradients"]["image_projection.weight"]["nonzero"], 0)
        self.assertGreater(awake["input_gradients"]["tail_images"]["nonzero"], 0)
        self.assertGreater(awake["image_changed_fraction"], 0)
        self.assertEqual(len(snapshots), 3)

    def test_zero_only_case_has_no_optimizer_or_gradients(self):
        report, _ = self.run_toy(updates=0)
        self.assertEqual(report["optimizer_updates"], 0)
        self.assertEqual(report["initial_visual_sha256"], report["final_visual_sha256"])
        self.assertNotIn("parameter_gradients", report["passes"][0])

    def test_off_nonexact_failure_recorded_before_any_update(self):
        report, snapshots = self.run_toy(broken=True)
        self.assertEqual(report["optimizer_updates"], 0)
        self.assertFalse(report["checks"]["pass0_off_native_euler4_exact"])
        self.assertEqual(len(report["passes"]), 1)
        self.assertTrue(snapshots)

    def test_preflight_only_creates_nothing_loads_no_model(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new"
            plan = {"cases": [{"episode_id": 1, "query": 3, "frame": 48}]}
            argv = ["--cache-dir", "unused", "--sidecar-dir", "unused", "--checkpoint", "unused",
                    "--visual-init-reference", "unused", "--output-dir", str(output), "--preflight-only"]
            with patch.object(probe, "preflight", return_value=(plan, [], {}, None, output)), \
                    patch.object(probe.native, "actual_head", side_effect=AssertionError("No actual model")), \
                    patch("torch.cuda.init", side_effect=AssertionError("No CUDA")), patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(probe.main(argv), 0)
            self.assertFalse(output.exists())

    def test_model_load_failure_persists_error_and_integrity_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new"
            plan = {"cases": [], "base_model": "missing", "source_sha256": {}, "files_sha256": {}, "runtime": {}}
            argv = ["--cache-dir", "unused", "--sidecar-dir", "unused", "--checkpoint", "unused",
                    "--visual-init-reference", "unused", "--output-dir", str(output)]
            with patch.object(probe, "preflight", return_value=(plan, [], {}, None, output)), \
                    patch.object(probe.native, "actual_head", side_effect=RuntimeError("original load failure")), \
                    patch.object(probe, "source_hashes", side_effect=FileNotFoundError("removed source")), \
                    patch.object(probe, "runtime_identity", return_value={}), patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(probe.main(argv), 1)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["error"]["message"], "original load failure")
            self.assertEqual(result["integrity_errors"][0]["message"], "removed source")
            self.assertEqual(sorted(p.name for p in output.iterdir()), ["plan.json", "result.json"])

    def test_existing_output_not_overwritten_even_after_preflight_race(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "existing"
            output.mkdir()
            argv = ["--cache-dir", "unused", "--sidecar-dir", "unused", "--checkpoint", "unused",
                    "--visual-init-reference", "unused", "--output-dir", str(output)]
            with patch.object(probe, "preflight", return_value=({}, [], {}, None, output)), \
                    patch.object(probe.native, "actual_head", side_effect=AssertionError("No actual model")):
                with self.assertRaises(FileExistsError):
                    probe.main(argv)
            self.assertEqual(list(output.iterdir()), [])

    def test_rng_restore_error_cannot_mask_original_failure_or_report(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new"
            plan = {"base_model": "missing", "source_sha256": {}, "files_sha256": {}, "runtime": {}}
            argv = ["--cache-dir", "unused", "--sidecar-dir", "unused", "--checkpoint", "unused",
                    "--visual-init-reference", "unused", "--output-dir", str(output)]
            with patch.object(probe, "preflight", return_value=(plan, [], {}, None, output)), \
                    patch.object(probe.native, "actual_head", side_effect=RuntimeError("original failure")), \
                    patch.object(probe, "restore_rng", side_effect=RuntimeError("RNG restore failure")), \
                    patch.object(probe, "source_hashes", return_value={}), \
                    patch.object(probe, "runtime_identity", return_value={}), patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(probe.main(argv), 1)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["error"]["message"], "original failure")
            self.assertFalse(result["checks"]["caller_rng_restored"])
            self.assertEqual(result["integrity_errors"][0]["message"], "RNG restore failure")

    def test_cuda_initialization_failure_is_reported_not_lost(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new"
            plan = {"source_sha256": {}, "files_sha256": {}, "runtime": {}}
            argv = ["--cache-dir", "unused", "--sidecar-dir", "unused", "--checkpoint", "unused",
                    "--visual-init-reference", "unused", "--output-dir", str(output), "--device", "cuda:0"]
            with patch.object(probe, "preflight", return_value=(plan, [], {}, None, output)), \
                    patch("torch.cuda.init", side_effect=RuntimeError("unavailable CUDA")), \
                    patch.object(probe, "source_hashes", return_value={}), \
                    patch.object(probe, "runtime_identity", return_value={}), patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(probe.main(argv), 1)
            self.assertEqual(json.loads((output / "result.json").read_text())["error"]["message"], "unavailable CUDA")


if __name__ == "__main__":
    unittest.main()
