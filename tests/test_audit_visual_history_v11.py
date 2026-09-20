"""CPU-only counterfactual contracts; no original model, GPU, training or eval."""
from contextlib import redirect_stdout
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from run_scripts.robomme import audit_visual_history_v11 as audit
from run_scripts.robomme.visual_patch_memory_v11 import VisualPatchConfig, VisualPatchMemoryV11


def observations(seed, count=5):
    generator = torch.Generator().manual_seed(seed)
    result = {key: [] for key in audit.OBSERVATION_KEYS}
    for i in range(count):
        length = 210 + 3 * ((seed + i) % 3)
        feature = torch.randn(length, 8, generator=generator)
        image = torch.zeros(length, dtype=torch.bool)
        offset = length - 177
        image[offset:offset + 81] = True
        image[offset + 88:offset + 169] = True
        image[-4:] = True  # Core explicitly excludes HAMLET tail.
        attention = torch.ones_like(image)
        values = (feature, image, attention, 16 * i, i < 2)
        for key, value in zip(audit.OBSERVATION_KEYS, values):
            result[key].append(value)
    return result


class Column:
    def __init__(self, values, maximum):
        self.values, self.maximum, self.seen = values, maximum, []

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index <= self.maximum:
            raise AssertionError("Future row or broad slice accessed")
        self.seen.append(index)
        return self.values[index]

    def __len__(self):
        raise AssertionError("Receiver/donor whole history inspected")


class ObservationOnly(dict):
    def __getitem__(self, key):
        if key not in audit.OBSERVATION_KEYS:
            raise AssertionError("Supervision/action/state touched")
        return super().__getitem__(key)

    def __iter__(self):
        raise AssertionError("Enclosing episode enumerated")


class VisualHistoryAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def memory(self):
        with torch.random.fork_rng():
            torch.manual_seed(81)
            memory = VisualPatchMemoryV11(VisualPatchConfig(feature_dim=8, hidden_dim=16))
            with torch.no_grad():
                memory.output_projection.weight.normal_(std=.05)
        return memory.eval().requires_grad_(False)

    def test_only_past_patch_content_changes_with_offset_camera_mapping(self):
        receiver, donor = observations(1), observations(2)
        before = copy.deepcopy(receiver)
        changed, mapping = audit.transplant_history(receiver, donor, 3)
        self.assertEqual(len(mapping), 3)
        for i in range(3):
            dst = audit.patch_indices(receiver["features"][i], receiver["image_masks"][i], receiver["attention_masks"][i], 4)
            src = audit.patch_indices(donor["features"][i], donor["image_masks"][i], donor["attention_masks"][i], 4)
            self.assertFalse(torch.equal(dst, src))
            torch.testing.assert_close(changed["features"][i][dst], donor["features"][i][src], rtol=0, atol=0)
            keep = torch.ones(receiver["features"][i].shape[0], dtype=torch.bool)
            keep[dst] = False
            self.assertTrue(torch.equal(changed["features"][i][keep], receiver["features"][i][keep]))
            self.assertTrue(torch.equal(changed["features"][i][-4:], receiver["features"][i][-4:]))
            self.assertEqual(mapping[i]["receiver_row"], mapping[i]["donor_row"])
            for key in audit.OBSERVATION_KEYS[1:]:
                self.assertIs(changed[key][i], receiver[key][i])
        for key in audit.OBSERVATION_KEYS:
            self.assertIs(changed[key][3], receiver[key][3])
        for actual, expected in zip(receiver["features"], before["features"]):
            self.assertTrue(torch.equal(actual, expected))

    def test_no_receiver_future_or_donor_gt_action_state_access(self):
        receiver = ObservationOnly({key: Column(values, 3) for key, values in observations(1).items()})
        donor = ObservationOnly({key: Column(values, 2) for key, values in observations(2).items()})
        # Toxic extras are never enumerated or looked up.
        for source in (receiver, donor):
            source.update(targets=object(), actions=object(), state=object(), target_mask=object())
        result, mapping = audit.transplant_history(receiver, donor, 3)
        self.assertEqual(len(result["features"]), 4)
        self.assertEqual(len(mapping), 3)
        for key in audit.OBSERVATION_KEYS:
            self.assertTrue(all(i <= 3 for i in receiver[key].seen))
            self.assertTrue(all(i <= 2 for i in donor[key].seen))

    def test_empty_history_exact_without_any_donor_access(self):
        receiver = ObservationOnly({key: Column(values, 0) for key, values in observations(1).items()})
        class Toxic:
            def __getitem__(self, key): raise AssertionError("Empty history touched donor")
        views, mapping = audit.transplant_history(receiver, Toxic(), 0)
        self.assertEqual(mapping, [])
        self.assertIs(views["features"][0], receiver["features"][0])
        memory = self.memory()
        fused = torch.randn(1, 4, 8)
        with torch.no_grad():
            features, _ = audit.conditioned_views(memory, receiver, 0, fused, Toxic())
        expected = torch.cat((receiver["features"][0][None, :-4], fused), 1)
        for value in features.values():
            self.assertTrue(torch.equal(value, expected))

    def test_capacity_receiver_metadata_and_parent_short_are_exact_frozen(self):
        receiver, donor, memory = observations(1), observations(2), self.memory()
        donor["frames"] = [v + 900 for v in donor["frames"]]
        donor["is_demo"] = [False] * 5
        fused = torch.randn(1, 4, 8)
        original_state = {k: v.clone() for k, v in memory.state_dict().items()}
        with torch.no_grad():
            features, mapping = audit.conditioned_views(memory, receiver, 3, fused, donor)
        self.assertEqual([r["receiver_frame"] for r in mapping], [0, 16, 32])
        self.assertEqual([r["donor_frame"] for r in mapping], [900, 916, 932])
        image = receiver["image_masks"][3].clone()
        image[-4:] = False
        expected = torch.cat((receiver["features"][3][None, :-4], fused), 1)
        for value in features.values():
            self.assertTrue(torch.equal(value[:, -4:], fused))
            self.assertTrue(torch.equal(value[:, ~image], expected[:, ~image]))
            self.assertFalse(value.requires_grad)
        self.assertTrue(torch.equal(features[audit.ROLES[1]], expected))
        self.assertFalse(torch.equal(features[audit.ROLES[0]], features[audit.ROLES[2]]))
        for name, value in memory.state_dict().items():
            self.assertTrue(torch.equal(value, original_state[name]))
        self.assertTrue(all(p.grad is None for p in memory.parameters()))

    def test_donor_selection_literal_instruction_train_only_enough_no_cycling(self):
        records = {i: {"task": "same instruction"} for i in range(6)}
        records[0]["task"] = "different instruction"
        episodes = {1: ObservationOnly(observations(1, 2)), 2: ObservationOnly(observations(2, 4)),
                    3: ObservationOnly(observations(3, 5))}
        seen = []
        def load(eid):
            seen.append(eid)
            return episodes[eid]
        result = audit.select_donors(records, [3, 0, 2, 1], [[4, 3], [5, 0]], load)
        self.assertEqual(result[4, 3]["donor_episode_id"], 2)
        self.assertIsNone(result[4, 3]["canonical_benchmark_task"])
        self.assertIsNone(result[5, 0]["donor_episode_id"])
        self.assertEqual(seen, [1, 2])
        with self.assertRaisesRegex(ValueError, "No sufficiently long"):
            audit.select_donors(records, [1], [[4, 3]], load)

    def test_bad_patch_layout_dtype_and_nonfinite_fail_closed(self):
        for kind in ("layout", "dtype", "finite"):
            receiver, donor = observations(1), observations(2)
            if kind == "layout": donor["image_masks"][0][10] = True
            elif kind == "dtype": donor["features"][0] = donor["features"][0].double()
            else:
                index = audit.patch_indices(donor["features"][0], donor["image_masks"][0], donor["attention_masks"][0], 4)[0]
                donor["features"][0][index] = float("nan")
            with self.assertRaises(ValueError): audit.transplant_history(receiver, donor, 1)

    def test_saved_validation_schedule_not_resampled_and_tampering_rejected(self):
        from run_scripts.robomme.train_archive_deployment_v9 import digest as training_digest
        self.assertEqual(audit.digest({"b": [1, 2], "a": 3}), training_digest({"b": [1, 2], "a": 3}))
        validation = [[i, i % 3] for i in range(32)]
        plan = {"validation": validation, "validation_schedule": [
            {"episode_id": i, "decision": q, "repeat": r, "flow_seed": i * 10 + r,
             "generation_seed": i * 10 + r + 2} for i, q in validation for r in range(2)], "files": {}}
        info = {"metadata": {"plan_sha256": audit.digest(plan)},
                "config": {"train": {"val_samples": 32, "val_noise_samples": 2}}}
        manifest = {"splits": {"train": [32], "val": list(range(32))}}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "query_plan.json"
            path.write_text(json.dumps({"sha256": audit.digest(plan), **plan}))
            self.assertEqual(audit.validation_plan(path, info, manifest), plan)
            changed = copy.deepcopy(plan)
            changed["validation_schedule"][0]["generation_seed"] += 1
            path.write_text(json.dumps({"sha256": audit.digest(changed), **changed}))
            with self.assertRaisesRegex(ValueError, "digest differs"):
                audit.validation_plan(path, info, manifest)

    def test_frozen_runtime_native_controls_seeds_and_no_generation_targets(self):
        from gr00t.long_memory.recurrent_v7 import MemoryV7Config
        cfg = {"memory": asdict(MemoryV7Config(feature_dim=8, state_dim=3, num_short_tokens=4,
                    hidden_dim=16, num_heads=4, capacity=4)),
               "expert": {"rank": 2, "alpha": 4.}, "expert_targets": []}
        ep = observations(1)
        ep.update(short=torch.stack([f[-4:] for f in ep["features"]]), state=torch.randn(5, 3),
                  targets=torch.randn(4, 16, 8), target_mask=torch.ones(4, 16, 8, dtype=torch.bool),
                  action_mask=torch.ones(4, 16, dtype=torch.bool), actions=torch.zeros(4, 16, 8),
                  decision_mask=torch.ones(4, dtype=torch.bool), embodiment_id=0)
        donor = ObservationOnly(observations(2))
        head = torch.nn.Linear(8, 8).eval().requires_grad_(False)
        head.num_inference_timesteps = 4
        info = {"step": 384, "config": {"visual": asdict(VisualPatchConfig(feature_dim=8, hidden_dim=16))},
                "metadata": {"frozen_parent": {"path": "unused"}}}
        plan = {"validation_schedule": [{"episode_id": 0, "decision": 2, "repeat": r,
                   "flow_seed": 20+r, "generation_seed": 30+r} for r in range(2)]}
        donors = {(0, 2): {"donor_episode_id": 1, "cache_instruction_key": "literal instruction"}}
        def load_visual(path, visual):
            with torch.no_grad(): visual.output_projection.weight.normal_(std=.03)
            return info
        def generate(head, view, query, *, seed):
            self.assertEqual(set(view), {"features", "state", "embodiment_id", "image_masks", "attention_masks"})
            self.assertFalse(torch.is_grad_enabled())
            value = head(view["features"][query].mean(0))[None, None].expand(1, 16, 8)
            return value + torch.randn(1, 16, 8, generator=torch.Generator().manual_seed(seed)) * .1
        result = {"records": []}
        with patch.object(audit, "v7_checkpoint_info", return_value={"config": cfg}), \
             patch.object(audit, "actual_head", return_value=head), patch.object(audit, "install_expert_lora"), \
             patch.object(audit, "load_checkpoint_v7"), patch.object(audit, "set_expert_trainable"), \
             patch.object(audit, "load_checkpoint", side_effect=load_visual), \
             patch.object(audit, "generated_action", side_effect=generate) as generated, \
             patch.object(audit, "expert_episode_flow_loss", return_value={"loss": torch.tensor(.25)}), \
             redirect_stdout(io.StringIO()):
            audit.run(SimpleNamespace(device="cpu", checkpoint="unused"), None, info, plan,
                      SimpleNamespace(load=lambda eid: ep if eid == 0 else donor), donors, result, lambda: None)
        self.assertEqual(len(result["records"]), 6)
        self.assertEqual([c.kwargs["seed"] for c in generated.call_args_list], [30, 30, 30, 31, 31, 31])
        self.assertTrue(result["frozen_modules_unchanged"])
        self.assertTrue(result["no_parameter_gradients"])
        self.assertTrue(all(r["history_patch_tokens"] == 324 for r in result["records"]))
        self.assertIn("generated_vs_correct_observed_prefix_mse", result["records"][2])

    def test_training_parity_records_deltas_and_rejects_wrong_noise(self):
        rows = [{"role": role, "episode_id": 2, "decision": 3, "repeat": 0, "flow_seed": 10,
                 "generation_seed": 11, "flow_loss": .2, "prefix_mse": .3, "prefix_mae": .4,
                 "joint7_mse": .5, "gripper1_mse": .6} for role in audit.ROLES[:2]]
        saved = {"step": 384, "plan_sha256": "fixed", "records": [
            {"role": role, "episode_id": 2, "decision": 3, "repeat": 0, "flow_seed": 10,
             "generation_seed": 11, "flow_action_loss": .2, "generated_observed_prefix_mse": .3,
             "generated_observed_prefix_mae": .4, "generated_observed_joint7_mse": .5,
             "generated_observed_gripper1_mse": .6} for role in ("reader", "visual-off")]}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json"
            path.write_text(json.dumps(saved))
            result = audit.training_parity(rows, path, step=384, plan_sha256="fixed")
            self.assertTrue(result["all_metrics_bitexact"])
            rows[0]["flow_loss"] += .01
            result = audit.training_parity(rows, path, step=384, plan_sha256="fixed")
            self.assertFalse(result["all_metrics_bitexact"])
            self.assertAlmostEqual(result["summary"][audit.ROLES[0]]["flow_loss"]["max_abs_delta"], .01)
            rows[0]["generation_seed"] += 1
            with self.assertRaisesRegex(ValueError, "noise differs"):
                audit.training_parity(rows, path, step=384, plan_sha256="fixed")

    def test_prefix_metric_mask_and_query_cluster_summary(self):
        episode = {"targets": torch.zeros(1, 20, 10), "target_mask": torch.zeros(1, 20, 10, dtype=torch.bool),
                   "action_mask": torch.zeros(1, 16, dtype=torch.bool)}
        episode["target_mask"][:, :, :8] = True
        episode["action_mask"][:, :3] = True
        prediction = torch.ones(1, 20, 10)
        prediction[:, :, 7] = 2
        prediction[:, 3:] = float("nan")  # Invalid/padded values do not enter metrics.
        prediction[:, :, 8:] = float("nan")
        metrics = audit.action_metrics(prediction, episode, 0)
        self.assertEqual(metrics["prefix_count"], 24)
        self.assertEqual(metrics["joint7_mse"], 1.)
        self.assertEqual(metrics["gripper1_mse"], 4.)
        rows = []
        for eid in (0, 1):
            for repeat in (0, 1):
                for offset, role in enumerate(audit.ROLES):
                    m = {**metrics, **{g + "_mse": metrics[g + "_mse"] + offset for g in ("prefix", "joint7", "gripper1")}}
                    rows.append({"episode_id": eid, "decision": 2, "repeat": repeat, "generation_seed": 10 + repeat,
                                 "flow_seed": 20 + repeat, "role": role, **m})
        summary = audit.summarize(rows, bootstrap_samples=50)
        result = summary["paired_contrasts"][audit.ROLES[2] + "_minus_correct_history"]["prefix_mse"]
        self.assertEqual(result["other_minus_correct"], 2.)
        self.assertEqual(result["query_n"], 2)  # NOT four independent noise draws.
        self.assertEqual(result["paired_query_bootstrap_ci95"], [2., 2.])
        rows[-1]["generation_seed"] += 1
        with self.assertRaisesRegex(ValueError, "noise pairing"):
            audit.summarize(rows, 50)

    def test_cli_preflight_does_not_create_output_or_load_head(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            prepared = (None, None, {"step": 512}, {"validation": [], "validation_schedule": []}, None, None,
                        {"audit_id": "a", "donors": []})
            with patch.object(audit, "preflight", return_value=prepared), patch.object(audit, "run") as run, \
                    patch.object(audit, "actual_head") as head, redirect_stdout(io.StringIO()):
                result = audit.main(["--cache-dir", "unused", "--checkpoint", "unused", "--training-run", "unused",
                                     "--output-dir", str(output), "--preflight-only"])
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())
            head.assert_not_called()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
