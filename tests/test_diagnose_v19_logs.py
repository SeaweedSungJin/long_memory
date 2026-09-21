"""Synthetic CPU checks for paired episode resampling and saved-log integrity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_scripts.robomme import diagnose_v19_logs as audit


def fixture():
    schedule, records = [], []
    for task, eid, decision in (("A", 1, 0), ("A", 1, 1), ("A", 2, 0), ("B", 3, 0), ("B", 4, 0)):
        for repeat in range(2):
            item = dict(task=task, episode_id=eid, decision=decision, repeat=repeat,
                        flow_seed=eid * 10 + decision * 2 + repeat, generation_seed=eid * 100 + decision * 2 + repeat)
            schedule.append(item)
            for role, offset in (("baseline", 3), ("memory-off", 2), ("reader", 1)):
                records.append({**item, "role": role, **{metric: float(eid + repeat + offset) for metric in audit.METRICS}})
    doc = {"records": records, "summary": {}, "by_task": {}}
    for role in audit.ROLES:
        doc["by_task"][role] = {task: {metric: float(np.mean([row[metric] for row in records if row["task"] == task and row["role"] == role]))
                                      for metric in audit.METRICS} for task in ("A", "B")}
        doc["summary"][role] = {metric: float(np.mean([doc["by_task"][role][task][metric] for task in ("A", "B")])) for metric in audit.METRICS}
    return schedule, doc


class DiagnoseV19LogsTests(unittest.TestCase):
    def test_final_metadata_extension_and_payload_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "checkpoint-000001"
            checkpoint.mkdir()
            coverage = {"train_query_count": 1, "epochs": 1, "total_query_presentations": 1, "total_steps": 1}
            plan = {**coverage, "validation_schedule": []}
            plan = {"sha256": audit.digest(plan), **plan}
            config = {"driver_variant": "full_memory_v19", "selection": "fixed_final_epoch", "train": {"epochs": 1, "max_steps": 1}}
            provenance = {"full_coverage": coverage, "plan_sha256": plan["sha256"]}
            for name in ("model.safetensors", "expert.safetensors", "training_state.pt"):
                (checkpoint / name).write_bytes(b"synthetic payload")
            metadata = {**provenance, "payload_shapes": {"fixture": [1]},
                        "payload_sha256": {name: audit.file_hash(checkpoint / name) for name in ("model.safetensors", "expert.safetensors")},
                        "training_state_sha256": audit.file_hash(checkpoint / "training_state.pt")}
            files = {"run_config.json": config, "provenance.json": provenance, "query_plan.json": plan,
                     "last_checkpoint.json": {"step": 1, "path": "checkpoint-000001"},
                     "status.json": {"status": "complete", "step": 1, "max_steps": 1, "completed_epochs": 1, "processed_queries": 1},
                     "checkpoint-000001/checkpoint.json": {"step": 1, "config": config, "metadata": metadata},
                     "validation-000000.json": {"step": 0}, "validation-000001.json": {"step": 1}}
            for name, value in files.items():
                (root / name).write_text(json.dumps(value))
            loaded = audit.load_run(root)
            self.assertEqual(loaded["final_checkpoint_metadata"]["metadata"], metadata)
            self.assertEqual(loaded["checkpoint_payloads"]["model.safetensors"]["verified_sha256"], metadata["payload_sha256"]["model.safetensors"])
            (checkpoint / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "payload hash differs"):
                audit.load_run(root)

    def test_repeat_averaging_and_gain_signs(self):
        schedule, doc = fixture()
        groups, _ = audit.matched_records(doc, schedule)
        keys, values, repeats = audit.query_values(groups)
        self.assertEqual(repeats, 2)
        self.assertEqual(len(keys), 5)
        np.testing.assert_allclose(values[0, :, 0], [4.5, 3.5, 2.5])
        for name, (left, right) in audit.GAINS.items():
            np.testing.assert_allclose(values[:, left] - values[:, right], 2 if name == "total_gain" else 1)

    def test_episode_resampling_retains_adjacent_queries_and_task_macro(self):
        keys = [("A", 1, 0), ("A", 1, 1), ("A", 2, 0), ("B", 3, 0)]
        indices, weights = audit.bootstrap_weights(keys, 1000, 2)
        np.testing.assert_array_equal(weights["A"][:, 0], weights["A"][:, 1])
        np.testing.assert_allclose(weights["A"].sum(axis=1), 1)
        estimates, ci = audit.estimates_and_ci(np.array([0., 0., 6., 10.]), indices, weights)
        self.assertEqual(estimates["__macro__"], 6.)
        self.assertNotEqual(estimates["__macro__"], 4.)
        np.testing.assert_allclose(ci["B"], [10., 10.])

    def test_constant_paired_effect_has_exact_interval(self):
        keys = [("A", 1, 0), ("A", 2, 0), ("B", 3, 0)]
        indices, weights = audit.bootstrap_weights(keys, 100, 42)
        estimates, ci = audit.estimates_and_ci(np.full((3, 3, 7), 2.), indices, weights)
        np.testing.assert_allclose(estimates["__macro__"], 2.)
        np.testing.assert_allclose(ci["__macro__"], 2.)

    def test_leave_out_renormalizes_and_does_not_drop_tasks(self):
        keys = [("A", 1, 0), ("A", 1, 1), ("A", 2, 0), ("B", 3, 0)]
        indices, _ = audit.bootstrap_weights(keys, 100, 0)
        values = np.array([0., 0., 6., 10.])
        self.assertEqual(audit.leave_out_gain(keys, values, indices, {0, 1}), 8.)
        self.assertIsNone(audit.leave_out_gain(keys, values, indices, {3}))

    def test_query_concentration_uses_macro_weights(self):
        keys = [("A", 1, 0), ("A", 2, 0), ("B", 3, 0)]
        indices, _ = audit.bootstrap_weights(keys, 100, 0)
        result = audit.concentration(np.array([1., 1., 2.]), keys, indices)
        self.assertAlmostEqual(result["macro"], 1.5)
        self.assertAlmostEqual(result["top1_share"], 2 / 3)
        self.assertAlmostEqual(result["top2_share"], 5 / 6)

    def test_reject_duplicate_missing_noise_schema_nonfinite_and_summary(self):
        schedule, original = fixture()
        mutations = [lambda doc: doc["records"].append(copy.deepcopy(doc["records"][0])),
                     lambda doc: doc["records"].pop(),
                     lambda doc: doc["records"][0].update(flow_seed=999),
                     lambda doc: doc["records"][0].pop(audit.METRICS[0]),
                     lambda doc: doc["records"][0].update(original_flow_loss=float("nan")),
                     lambda doc: doc["summary"]["baseline"].update(original_flow_loss=999.)]
        for mutate in mutations:
            document = copy.deepcopy(original)
            mutate(document)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                audit.matched_records(document, schedule)

    def test_unequal_noise_repeats_fail_closed(self):
        schedule, doc = fixture()
        groups, _ = audit.matched_records(doc, schedule)
        for role in audit.ROLES:
            groups[role].pop(next(iter(groups[role])))
        with self.assertRaisesRegex(ValueError, "Unequal"):
            audit.query_values(groups)

    def test_new_output_guard_preserves_inputs_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input"
            source.mkdir()
            with self.assertRaises(FileExistsError):
                audit.new_output(source, [source])
            with self.assertRaises(ValueError):
                audit.new_output(source / "report", [source])
            self.assertEqual(audit.new_output(root / "new", [source]), root / "new")


if __name__ == "__main__":
    unittest.main()
