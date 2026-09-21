"""CPU-only runner selection, input-preservation and GPU refusal contracts."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from run_scripts.robomme import diagnose_v19_runtime as runtime
from tests.test_diagnose_v19_memory import fixture


class RuntimeDiagnosticV19Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def validation(self):
        rows = []
        # Deliberately nonsorted IDs, losses and repeats. A rank-by-error or
        # numeric-ID selection would choose a different held-out panel.
        for eid, task, error in ((91, "VideoPlaceOrder", 1000.),
                                 (4, "OtherTask", 999999.),
                                 (8, "VideoPlaceOrder", .0001),
                                 (2, "VideoPlaceOrder", 10.)):
            for repeat in (0, 1):
                for role in ("baseline", "memory-off", "reader"):
                    rows.append({"task": task, "episode_id": eid, "decision": 7 + repeat,
                        "repeat": repeat, "role": role, "action_loss": error,
                        "generated_prefix_mse": error, "flow_seed": 100 + repeat,
                        "generation_seed": 500 + repeat})
        return {"records": rows}

    def test_selection_preserves_prespecified_order_and_all_repeats(self):
        validation = self.validation()
        rows = runtime.selected_rows(validation, "VideoPlaceOrder", 2)
        expected = [row for row in validation["records"]
                    if row["role"] == "reader" and row["episode_id"] in (91, 8)]
        self.assertEqual(rows, expected)
        self.assertEqual([row["episode_id"] for row in rows], [91, 91, 8, 8])
        self.assertEqual([row["repeat"] for row in rows], [0, 1, 0, 1])
        self.assertTrue(all(left is right for left, right in zip(rows, expected)))

    def test_selection_is_independent_of_outcome_values(self):
        validation = self.validation()
        expected = [(row["episode_id"], row["decision"], row["repeat"])
                    for row in runtime.selected_rows(validation, "VideoPlaceOrder", 2)]
        for row in validation["records"]:
            # Neither numeric ranking nor any attempt to inspect an outcome
            # should be necessary for selection.
            row["action_loss"] = object()
            row["generated_prefix_mse"] = object()
        actual = [(row["episode_id"], row["decision"], row["repeat"])
                  for row in runtime.selected_rows(validation, "VideoPlaceOrder", 2)]
        self.assertEqual(actual, expected)
        with self.assertRaisesRegex(ValueError, "absent"):
            runtime.selected_rows(validation, "MissingTask", 2)

    def test_branch_preserves_supervision_and_native_feature_tail(self):
        _, _, ep = fixture()
        original_short = ep["short"].clone()
        captures = []
        for index in range(len(ep["frames"])):
            features = (ep["features"][index] + 1).bfloat16()
            captures.append({"features": features,
                "attention_masks": ep["attention_masks"][index].clone(),
                "image_masks": ep["image_masks"][index].clone(),
                "short": torch.full((1, 4, 8), 1234., dtype=torch.float32),
                "moment": torch.full((1, 4, 8), float(index), dtype=torch.float32),
                "state": ep["state"][index:index + 1] + 2})
        branch = runtime.branch_episode(ep, captures)
        for key in ("targets", "target_mask", "action_mask", "actions", "decision_mask",
                    "frames", "is_demo", "embodiment_id"):
            self.assertIs(branch[key], ep[key])
        self.assertEqual(branch["short"].dtype, torch.bfloat16)
        self.assertEqual(branch["moment"].dtype, torch.float32)
        for index, capture in enumerate(captures):
            self.assertTrue(torch.equal(branch["short"][index], capture["features"][-4:]))
            self.assertTrue(torch.equal(branch["moment"][index], capture["moment"][0]))
            self.assertTrue(torch.equal(branch["state"][index], capture["state"][0]))
            self.assertIs(branch["features"][index], capture["features"])
        self.assertTrue(torch.equal(ep["short"], original_short))
        # Both tensors must satisfy the bridge's exact feature-tail contract,
        # even though the core separately captures a promoted FP32 short.
        from gr00t.long_memory.cache_reader_v3 import validate_decision
        validate_decision(branch, 10)

    def test_gpu_guard_refuses_busy_process_before_status_or_model_use(self):
        busy = "GPU-abc, 12345, another-job, 12000 MiB\n"
        with patch.object(runtime.subprocess, "run", return_value=SimpleNamespace(stdout=busy)) as run:
            with self.assertRaisesRegex(RuntimeError, "Existing GPU work"):
                runtime.gpu_snapshot("cuda:0")
        self.assertEqual(run.call_count, 1)
        args, kwargs = run.call_args
        self.assertIn("--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", args[0])
        self.assertTrue(kwargs["check"])
        self.assertNotIn("-i", args[0])  # Guard checks every GPU, including remapped ordinals.

    def test_gpu_guard_records_idle_host_and_fails_closed_on_query_error(self):
        with patch.object(runtime.subprocess, "run", side_effect=[
                SimpleNamespace(stdout="  \n"),
                SimpleNamespace(stdout="GPU-abc, Test device, 0 MiB, 100 MiB, 0 %\n")]) as run:
            report = runtime.gpu_snapshot("cuda:1")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(report["compute_processes_before"], "")
        self.assertIn("GPU-abc", report["devices_before"])
        with patch.object(runtime.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
            with self.assertRaises(subprocess.CalledProcessError):
                runtime.gpu_snapshot("cuda:0")
        with patch.object(runtime.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "CUDA"):
                runtime.gpu_snapshot("cpu")
        run.assert_not_called()

    def test_main_preserves_existing_output_before_opening_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing-diagnostic"
            output.mkdir()
            marker = output / "evidence.json"
            marker.write_text('{"preserve": true}\n')
            argv = ["diagnose_v19_runtime.py", "--cache-dir", "unused-cache",
                    "--checkpoint", "unused-checkpoint", "--validation", "unused-validation",
                    "--output-dir", str(output)]
            with patch.object(runtime.sys, "argv", argv), \
                 patch.object(runtime, "EpisodeCache") as cache, \
                 patch.object(runtime, "gpu_snapshot") as gpu:
                with self.assertRaisesRegex(FileExistsError, "preserving"):
                    runtime.main()
            cache.assert_not_called()
            gpu.assert_not_called()
            self.assertEqual(marker.read_text(), '{"preserve": true}\n')

    def test_main_rejects_new_output_inside_each_read_only_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path, run_path = root / "cache", root / "run"
            dataset, model = root / "dataset", root / "model"
            cache = SimpleNamespace(path=cache_path,
                manifest={"dataset_path": str(dataset), "model_path": str(model)})
            for source in (cache_path, run_path, dataset, model):
                output = source / "new-diagnostic"
                argv = ["diagnose_v19_runtime.py", "--cache-dir", str(cache_path),
                    "--checkpoint", str(run_path / "checkpoint-1"),
                    "--validation", str(run_path / "validation.json"), "--output-dir", str(output)]
                with self.subTest(source=source), patch.object(runtime.sys, "argv", argv), \
                     patch.object(runtime, "EpisodeCache", return_value=cache), \
                     patch.object(runtime, "MappedEpisodes") as mapped, \
                     patch.object(runtime, "gpu_snapshot") as gpu:
                    with self.assertRaisesRegex(ValueError, "inside a source"):
                        runtime.main()
                    mapped.assert_not_called()
                    gpu.assert_not_called()
                    self.assertFalse(output.exists())

    def test_main_rejects_checkpoint_cache_or_validation_mismatch_before_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            checkpoint = run / "checkpoint-1"
            checkpoint.mkdir(parents=True)
            wrong_run = root / "other-run"
            wrong_run.mkdir()
            validation = {"step": 1, "records": self.validation()["records"]}
            for parent in (run, wrong_run):
                runtime.write_json(parent / "validation.json", validation)
            cache = SimpleNamespace(path=root / "cache", manifest={"fingerprint": "cache-a",
                "dataset_path": str(root / "dataset"), "model_path": str(root / "model"),
                "splits": {"val": [91, 8, 2]}})
            for fingerprint, step, report_parent in (("wrong", 1, run), ("cache-a", 2, run),
                                                     ("cache-a", 1, wrong_run)):
                runtime.write_json(checkpoint / "checkpoint.json",
                    {"metadata": {"cache_fingerprint": fingerprint}, "step": step})
                output = root / "new-diagnostic"
                argv = ["diagnose_v19_runtime.py", "--cache-dir", str(cache.path),
                    "--checkpoint", str(checkpoint), "--validation", str(report_parent / "validation.json"),
                    "--output-dir", str(output)]
                with self.subTest(fingerprint=fingerprint, step=step, parent=report_parent), \
                     patch.object(runtime.sys, "argv", argv), \
                     patch.object(runtime, "EpisodeCache", return_value=cache), \
                     patch.object(runtime, "MappedEpisodes"), \
                     patch.object(runtime, "gpu_snapshot") as gpu:
                    with self.assertRaisesRegex(ValueError, "provenance must match"):
                        runtime.main()
                    gpu.assert_not_called()
                    self.assertFalse(output.exists())

    def test_json_rejects_nonfinite_before_overwriting_existing_journal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "journal.json"
            runtime.write_json(path, {"records": [1, 2]})
            before = path.read_text()
            with self.assertRaises(ValueError):
                runtime.write_json(path, {"invalid": float("nan")})
            self.assertEqual(path.read_text(), before)
            self.assertEqual(json.loads(before), {"records": [1, 2]})


if __name__ == "__main__":
    unittest.main()
