"""Stdlib-only synthetic artifact tests; no model, GPU or simulator."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from run_scripts.robomme import summarize_deployment_v9_pilots as summary


class PilotSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = {"train": [[1, [0, 1]]], "validation": [[eid, 1] for eid in range(2, 34)],
            "validation_schedule": [{"episode_id": eid, "decision": 1, "repeat": 0,
                "flow_seed": eid + 10, "generation_seed": eid + 20} for eid in range(2, 34)],
            "schedule": [{"step": step} for step in range(1, 129)]}
        self.plan_sha = summary.digest(self.plan)
        self.flow = self.make_run("flow", 0.)
        self.aux = self.make_run("aux", .1)

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def make_run(self, name, weight):
        run = self.root / name
        run.mkdir()
        config = {"trainer_variant": summary.ARCHITECTURE, "driver_variant": summary.DRIVER,
            "stage": 1, "mode": "archive", "memory": {"capacity": 64}, "expert": {"rank": 8},
            "train": {"driver_variant": summary.DRIVER, "max_steps": 128, "aux_weight": weight,
                "output_dir": str(run), "init_checkpoint": "original-parent", "resume": None,
                "stop_after_steps": None, "preflight_only": False, "memory_learning_rate": 1e-5},
            "objective": {"driver_variant": summary.DRIVER, "generated_observed_prefix_weight": weight,
                "selection_metric": summary.SELECTION, "flow_weight": 1.}}
        provenance = {"driver_variant": summary.DRIVER, "selection_metric": summary.SELECTION,
            "base_model": {"path": "original-base"}, "cache_fingerprint": "cache-A", "cache_dir": "cache",
            "cache_manifest_sha256": "cachehash", "source_sha256": {"train.py": "codehash"},
            "runtime": {"torch": "same"}, "plan_sha256": self.plan_sha,
            "original_continuation_parent": {"path": "parent", "step": 1250, "files": {"weights": "parenthash"}}}
        self.write(run / "run_config.json", config)
        self.write(run / "provenance.json", provenance)
        self.write(run / "query_plan.json", {"sha256": self.plan_sha, **self.plan})
        self.write(run / "status.json", {"status": "training", "optimizer_updates": 32, "processed_queries": 128})
        for step in (0, 32):
            self.add_boundary(run, step, config, provenance, weight)
        self.write(run / "best_checkpoint.json", {"path": "checkpoint-000000", "step": 0})
        return run

    def add_boundary(self, run, step, config=None, provenance=None, weight=None):
        config = config or summary.read(run / "run_config.json")
        provenance = provenance or summary.read(run / "provenance.json")
        weight = config["train"]["aux_weight"] if weight is None else weight
        checkpoint = run / f"checkpoint-{step:06d}"
        checkpoint.mkdir()
        hashes = {}
        for name in summary.PAYLOADS:
            path = checkpoint / name
            path.write_bytes(("identical-initial-" + name).encode())
            hashes[name] = summary.file_hash(path)
        self.write(checkpoint / "checkpoint.json", {"step": step, "config": config,
            "metadata": {**provenance, "payload_sha256": hashes}})
        rows = []
        for query in self.plan["validation_schedule"]:
            for role in summary.ROLES:
                value = .1 if role == "baseline" else .2 if role == "memory-off" else .25
                if step > 0 and role == "reader":
                    value -= weight * .01
                rows.append({**query, "role": role, **{metric: value for metric in summary.METRICS}})
        means = {role: {metric: sum(r[metric] for r in rows if r["role"] == role) / 32
                        for metric in summary.METRICS} for role in summary.ROLES}
        self.write(run / f"validation-{step:06d}.json", {"step": step, "plan_sha256": self.plan_sha,
            "selection_metric": summary.SELECTION, "records": rows, "summary": means})

    def test_paired_values_best_zero_and_incomplete_is_not_failure(self):
        result = summary.compare(self.flow, self.aux)
        self.assertTrue(result["pairing_verified"])
        self.assertFalse(result["all_fixed_steps_complete"])
        self.assertEqual(result["common_steps"], [0, 32])
        self.assertEqual(result["flow"]["best"]["meaning"], "unchanged_parent_initialization")
        delta = result["comparisons"][1]["roles"]["reader"]["generated_observed_prefix_mse"]["aux_minus_flow"]
        self.assertAlmostEqual(delta, -.001)
        self.assertIn("NOT task accuracy", summary.render(result))
        result = summary.compare(self.flow, self.root / "not_started")
        self.assertFalse(result["pairing_verified"])
        self.assertEqual(result["aux"]["recorded_status"], "not_started_or_missing")

    def test_complete_requires_final_checkpoint_validation_and_recorded_status(self):
        for run in (self.flow, self.aux):
            for step in (64, 96, 128):
                self.add_boundary(run, step)
            self.write(run / "status.json", {"status": "complete", "optimizer_updates": 128, "processed_queries": 512})
        self.assertTrue(summary.compare(self.flow, self.aux)["all_fixed_steps_complete"])

    def test_rejects_config_and_provenance_mismatch(self):
        path = self.aux / "run_config.json"
        original = summary.read(path)
        changed = copy.deepcopy(original)
        changed["train"]["memory_learning_rate"] *= 2
        self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            summary.compare(self.flow, self.aux)
        self.write(path, original)
        path = self.aux / "provenance.json"
        changed = summary.read(path)
        changed["source_sha256"]["train.py"] = "differentcode"
        self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            summary.compare(self.flow, self.aux)

    def test_rejects_plan_initial_payload_and_validation_corruption(self):
        path = self.aux / "query_plan.json"
        original = summary.read(path)
        changed = copy.deepcopy(original)
        changed["validation_schedule"][0]["flow_seed"] += 1
        self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "checksum"):
            summary.compare(self.flow, self.aux)
        self.write(path, original)
        payload = self.aux / "checkpoint-000000/model.safetensors"
        before = payload.read_bytes()
        payload.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            summary.compare(self.flow, self.aux)
        payload.write_bytes(before)
        path = self.aux / "validation-000032.json"
        changed = summary.read(path)
        changed["records"][0][summary.METRICS[0]] += 1
        self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "Stored mean"):
            summary.compare(self.flow, self.aux)

    def test_rejects_duplicate_pairs_and_nonfinite_json(self):
        path = self.aux / "validation-000032.json"
        original = summary.read(path)
        changed = copy.deepcopy(original)
        changed["records"][1] = changed["records"][0]
        self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summary.compare(self.flow, self.aux)
        changed = copy.deepcopy(original)
        changed["summary"]["reader"][summary.METRICS[0]] = float("nan")
        self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            summary.compare(self.flow, self.aux)

    def test_new_output_only_no_overwrite_and_default_readonly(self):
        before = {str(p): p.read_bytes() for run in (self.flow, self.aux) for p in run.rglob("*") if p.is_file()}
        args = ["--flow-run", str(self.flow), "--aux-run", str(self.aux)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(summary.main(args), 0)
            output = self.root / "summary.json"
            self.assertEqual(summary.main(args + ["--output-file", str(output)]), 0)
            with self.assertRaises(FileExistsError):
                summary.main(args + ["--output-file", str(output)])
        after = {str(p): p.read_bytes() for run in (self.flow, self.aux) for p in run.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_resume_uses_original_initialization_and_only_pre_pause_history(self):
        resumed = self.root / "aux_resumed"
        resumed.mkdir()
        config = summary.read(self.aux / "run_config.json")
        provenance = summary.read(self.aux / "provenance.json")
        parent = self.aux / "checkpoint-000032"
        config["train"].update(resume=str(parent), init_checkpoint=None, output_dir=str(resumed))
        provenance["parent_checkpoint"] = {"path": str(parent), "step": 32,
            "files": {"checkpoint.json": summary.file_hash(parent / "checkpoint.json")}}
        self.write(resumed / "run_config.json", config)
        self.write(resumed / "provenance.json", provenance)
        self.write(resumed / "query_plan.json", {"sha256": self.plan_sha, **self.plan})
        self.write(resumed / "status.json", {"status": "training", "optimizer_updates": 32, "processed_queries": 128})
        self.add_boundary(self.aux, 64)  # Original branch subsequently continued.
        self.add_boundary(self.flow, 64)
        report = summary.compare(self.flow, resumed)
        self.assertTrue(report["pairing_verified"])
        self.assertEqual(report["common_steps"], [0, 32])  # Do not import future original-branch step 64.
        self.assertEqual(report["aux"]["initial_checkpoint"]["path"], str(self.aux / "checkpoint-000000"))
        self.add_boundary(resumed, 64, config, provenance, .1)
        self.assertEqual(summary.compare(self.flow, resumed)["common_steps"], [0, 32, 64])


if __name__ == "__main__":
    unittest.main()
