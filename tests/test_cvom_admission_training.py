"""CPU contracts for paired, frozen-actor CVoM admission training."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from gr00t.long_memory.monitoring import _atomic_json
from run_scripts.robomme.cvom_admission_core import AdmissionConfig, CVoMAdmission
from run_scripts.robomme import train_cvom_admission as driver


def args_for(output, stage="train"):
    return driver.parser().parse_args([stage, "--parent-checkpoint", "/synthetic-parent",
        "--cache-dir", "/synthetic-cache", "--output-dir", str(output), "--device", "cpu",
        "--updates", "3", "--batch-size", "4", "--hidden-dim", "8",
        "--eval-steps", "3", "--save-steps", "3"])


def example(eid, y, *, coalition=None, noise=0.):
    label = lambda target: {"signed_mean": target, "noise_mean_std": noise}
    return {"episode_id": eid, "event": 2, "future": [7, 9], "task": "synthetic",
        "features": [float(eid % 3) + x/10 for x in range(12)],
        "labels": {"single": label(y), "coalitional": label(y if coalition is None else coalition)}}


def write_fixture(output, examples):
    output.mkdir(parents=True)
    plan = {split: [{k: row[k] for k in ("episode_id", "event", "future", "task")}
                    for row in rows] for split, rows in examples.items()}
    protocol = {"plan": plan, "plan_sha256": driver.digest(plan), "parent": {"identity": "frozen"},
                "source_identity": {}, "fingerprint": "synthetic-protocol"}
    actor = {"core": "unchanged-core", "head": "unchanged-head"}
    manifest = {"format": driver.FORMAT, "protocol_fingerprint": protocol["fingerprint"],
        "actor_state_hash_before": actor, "actor_state_hash_after": actor,
        "actor_frozen": True, "future_inputs_at_inference": False, "context_files": {}}
    _atomic_json(output / "label_protocol.json", protocol)
    for split, rows in examples.items():
        (output / "contexts" / split).mkdir(parents=True)
        for index, row in enumerate(rows):
            relative = f"contexts/{split}/context-{index:06d}.json"
            packet = {"protocol_fingerprint": protocol["fingerprint"], "split": split,
                "context_sha256": driver.digest(plan[split][index]), "actor_state_hash": actor,
                "result": {"contexts": [row]}}
            packet["sha256"] = driver.digest(packet)
            _atomic_json(output / relative, packet)
            manifest["context_files"][relative] = driver.file_hash(output / relative)
    manifest["sha256"] = driver.digest(manifest)
    _atomic_json(output / "label_manifest.json", manifest)
    cfg = SimpleNamespace(hidden_dim=2, num_short_tokens=2, capacity_events=2)
    cache = SimpleNamespace(manifest={"fingerprint": "cache"})
    return protocol, {}, cfg, cache, None, output


class AdmissionTrainingTests(unittest.TestCase):
    def test_confidence_preserves_signed_values_and_excludes_ambiguous_both_losses(self):
        rows = [example(0, -.2), example(1, .3), example(2, 9., noise=20.)]
        data = driver.tensor_examples(rows, "single", margin=.01, confidence_z=1.96)
        self.assertEqual(data["confident"].tolist(), [True, True, False])
        utility = torch.tensor([0., 0., 100.], requires_grad=True)
        logit = torch.tensor([0., 0., -100.], requires_grad=True)
        total, _, _ = driver.writer_loss({"utility": utility, "logit": logit}, data,
                                         torch.arange(3), .1, .25)
        total.backward()
        self.assertEqual(float(utility.grad[2]), 0.)
        self.assertEqual(float(logit.grad[2]), 0.)
        self.assertGreater(float(utility.grad[0]), 0.)
        self.assertLess(float(utility.grad[1]), 0.)

    def test_common_train_scale_does_not_accept_validation(self):
        values = {arm: {"utility": torch.tensor([-.2, .4])} for arm in driver.ARMS}
        self.assertAlmostEqual(driver.training_scale(values, 1e-6), .2)
        values["unrelated_validation"] = {"utility": torch.tensor([1e9])}
        self.assertAlmostEqual(driver.training_scale(values, 1e-6), .2)

    def test_validation_uses_deployment_tie_and_common_full_bank_target(self):
        controller = CVoMAdmission(AdmissionConfig(dim=2, num_tokens=2, capacity_events=2,
                                                   hidden_dim=8)).eval()
        rows = [example(0, -.25, coalition=.75), example(1, .5, coalition=-.5)]
        data = driver.tensor_examples(rows, "coalitional", margin=0., confidence_z=0.)
        metrics = driver.validation_metrics(controller, data, 1.)
        # Zero-initialized controller is FIFO, including utility == margin.
        self.assertEqual(metrics["write_rate"], 1.)
        self.assertEqual(metrics["full_bank_conditional_gain_vs_fifo"], 0.)
        self.assertAlmostEqual(metrics["full_bank_conditional_oracle_regret"], .125)
        self.assertAlmostEqual(metrics["conditional_oracle_regret"], .25)

    def test_tampered_context_cannot_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "labels"
            context = write_fixture(output, {"train": [example(0, 1.)], "val": [example(1, -1.)]})
            path = output / "contexts/train/context-000000.json"
            payload = json.loads(path.read_text())
            payload["result"]["contexts"][0]["labels"]["single"]["signed_mean"] = -999.
            _atomic_json(path, payload)
            with self.assertRaisesRegex(ValueError, "published manifest"):
                driver.load_examples(output, context[0])

    def test_cpu_training_shared_initialization_same_labels_same_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "labels"
            context = write_fixture(output, {"train": [example(0, -.2), example(1, .3)],
                                             "val": [example(2, -.1), example(3, .4)]})
            saved = []
            def save(path, controller, parent, **kwargs):
                self.assertTrue(all(p.device.type == "cpu" for p in controller.parameters()))
                self.assertTrue(kwargs["metadata"]["actor_frozen"])
                self.assertFalse(kwargs["metadata"]["future_inputs_at_inference"])
                saved.append((kwargs["arm"], driver.state_hash(controller), kwargs["metadata"]))
                path.mkdir(parents=True)
                return path
            with patch("run_scripts.robomme.cvom_admission_checkpoint.save_checkpoint", save), \
                 patch("run_scripts.robomme.cvom_admission_checkpoint.parent_identity", return_value=context[0]["parent"]), \
                 patch.object(driver, "load_actor", side_effect=AssertionError("Training must not load actor")):
                driver.train(args_for(output), context)
            self.assertEqual([row[0] for row in saved], list(driver.ARMS))
            self.assertEqual(saved[0][1], saved[1][1])
            self.assertEqual(saved[0][2]["training_plan"], saved[1][2]["training_plan"])
            self.assertEqual(json.loads((output / "train_status.json").read_text())["status"], "complete")
            with self.assertRaises(FileExistsError):
                driver.train(args_for(output), context)

    def test_no_informative_labels_stops_before_controller_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "labels"
            context = write_fixture(output, {"train": [example(0, 0.)], "val": [example(1, 0.)]})
            with self.assertRaisesRegex(ValueError, "No confident TRAIN"):
                driver.train(args_for(output), context)
            self.assertFalse((output / "single").exists())
            self.assertEqual(json.loads((output / "train_status.json").read_text())["status"], "stopped_uninformative")

    def test_prepare_resumes_only_missing_contexts_without_changing_teacher(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "labels"
            rows = {"train": [{k: example(0, 1.)[k] for k in ("episode_id", "event", "future", "task")}],
                    "val": [{k: example(1, -1.)[k] for k in ("episode_id", "event", "future", "task")}]}
            protocol = {"plan": rows, "parent": {"identity": "frozen"}, "fingerprint": "same-plan"}
            core = torch.nn.Linear(2, 2).eval().requires_grad_(False)
            head = torch.nn.Linear(2, 2).eval().requires_grad_(False)
            context = (protocol, {}, None, None, None, output)
            calls = []
            def labels(c, h, episodes, contexts, **kwargs):
                row = contexts[0]
                calls.append(row["episode_id"])
                return {"contexts": [example(row["episode_id"], 1.)]}
            args = args_for(output, "prepare")
            args.stop_after_contexts = 1
            with patch.object(driver, "load_actor", return_value=(core, head)), \
                 patch("run_scripts.robomme.cvom_admission_checkpoint.parent_identity", return_value=protocol["parent"]), \
                 patch("run_scripts.robomme.cvom_admission_teacher.label_contexts", labels):
                driver.prepare(args, context)
                self.assertEqual(json.loads((output / "prepare_status.json").read_text())["status"], "paused")
                args.stop_after_contexts = 0
                driver.prepare(args, context)
                driver.prepare(args, context)
            self.assertEqual(calls, [0, 1])
            examples, manifest = driver.load_examples(output, protocol)
            self.assertEqual(len(examples["train"]), 1)
            self.assertEqual(manifest["actor_state_hash_before"], manifest["actor_state_hash_after"])

    def test_gpu_guard_refuses_occupied_device_without_loading(self):
        with patch.object(driver.subprocess, "run", return_value=SimpleNamespace(stdout="GPU-1, 234, training, 8000 MiB")):
            with self.assertRaisesRegex(RuntimeError, "Existing GPU"):
                driver.gpu_guard("cuda:0")


if __name__ == "__main__":
    unittest.main()
