"""Small CPU integration checks: full plans, immutable base and exact resume.

Real encoder/reader/LoRA/optimizer/checkpoint code; toy Expert objective. Real
flow gradients and joint/gripper validation have separate V19 unit tests.
"""
from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from run_scripts.robomme import train_full_memory_v19 as trainer
from run_scripts.robomme.training_plan_v19 import build_plan_v19
from run_scripts.robomme.compare_full_training_v19 import final_checkpoint
from tests import test_archive_deployment_trainer_v9 as fixture
from tests import test_long_memory_v4_trainer as toy


def objective(head, ep, decision, fused=None, *, tail_weight, **kwargs):
    result = toy.toy_loss(head, ep, decision, fused, **kwargs)
    return {**result, "original_flow_loss": result["loss"], "loss": result["loss"] * tail_weight}


def validation(args, core, head, episodes, plan, baseline_cache):
    # No RNG consumption or gradient changes. Actual production validator is
    # covered by test_validation_v19 and the real-cache smoke.
    metrics = {"action_loss": 1., "loss": 1., "generated_prefix_mse": 1.}
    return ({r: dict(metrics) for r in ("reader", "memory-off", "baseline")}, [],
            {r: {"A": dict(metrics), "B": dict(metrics)} for r in ("reader", "memory-off", "baseline")})


class FullTrainingV19Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        fixture.DeploymentTrainerTests.setUp(self)
        self.manifest["episodes"] = [{"episode_id": eid, "task_group": "A" if eid % 2 == 0 else "B",
            "split": "train" if eid < 2 else "val"} for eid in range(34)]
        self.stack.enter_context(patch.object(trainer, "EpisodeCache", return_value=self.cache))
        self.stack.enter_context(patch.object(trainer, "MappedEpisodes", return_value=SimpleNamespace(fetch=self.cache.load)))
        self.stack.enter_context(patch.object(trainer, "validate_cache_checkpoint"))
        self.loader = self.stack.enter_context(patch.object(trainer, "load_frozen_hamlet", side_effect=fixture.toy_base))
        self.stack.enter_context(patch.object(trainer, "episode_flow_v19", side_effect=objective))
        self.stack.enter_context(patch.object(trainer, "validate_v19", side_effect=validation))
        def build(args, cache, episodes):
            settings = SimpleNamespace(**vars(args), expected_tasks=("A", "B"))
            return build_plan_v19(settings, cache, episodes)
        self.stack.enter_context(patch.object(trainer, "build_plan_v19", side_effect=build))

    def args(self, name, tail=1.):
        return ["--cache-dir", str(self.cache.path), "--output-dir", str(self.root / name),
            "--hidden-dim", "8", "--num-heads", "2", "--capacity-events", "32", "--lora-rank", "2", "--lora-alpha", "4",
            "--device", "cpu", "--epochs", "1", "--query-batch-size", "5", "--tail-weight", str(tail),
            "--val-per-task", "1", "--val-noise-samples", "1", "--eval-steps", "3", "--save-steps", "3",
            "--log-steps", "1", "--plot-steps", "3", "--memory-learning-rate", ".001", "--expert-learning-rate", ".001"]

    def run_train(self, argv):
        with redirect_stdout(io.StringIO()):
            return trainer.main(argv)

    def ckpt(self, name, step):
        return self.root / name / f"checkpoint-{step:06d}"

    def test_preflight_is_read_only(self):
        self.run_train(self.args("preflight") + ["--preflight-only"])
        self.loader.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())
        with self.assertRaises(ValueError):
            self.run_train(self.args("bad") + ["--tail-weight", "nan"])
        with self.assertRaises(ValueError):
            self.run_train(self.args("badzero") + ["--stop-after-steps", "0"])

    def test_matched_initialization_full_epoch_and_scope(self):
        original = {p.name: trainer.file_hash(p) for p in self.base.iterdir()}
        for name, weight in (("control", 1.), ("prefix", .25)):
            self.run_train(self.args(name, weight))
            status = json.loads((self.root / name / "status.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["completed_epochs"], 1)
            self.assertEqual(status["processed_queries"], 24)
            self.assertEqual(status["max_steps"], 5)  # final partial batch kept
            self.assertEqual(final_checkpoint(self.root / name), self.ckpt(name, 5))
            initial = load_file(str(self.ckpt(name, 0) / "model.safetensors"))
            final = load_file(str(self.ckpt(name, 5) / "model.safetensors"))
            self.assertTrue(any(not torch.equal(initial[k], final[k]) for k in initial))
        for filename in ("model.safetensors", "expert.safetensors"):
            a = load_file(str(self.ckpt("control", 0) / filename))
            b = load_file(str(self.ckpt("prefix", 0) / filename))
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertEqual(json.loads((self.root / "control/query_plan.json").read_text()),
                         json.loads((self.root / "prefix/query_plan.json").read_text()))
        self.assertEqual(original, {p.name: trainer.file_hash(p) for p in self.base.iterdir()})
        with self.assertRaises(FileExistsError):
            self.run_train(self.args("control"))

    def test_exact_resume_and_paused_checkpoint_rejected_for_comparison(self):
        self.run_train(self.args("full", .25))
        self.run_train(self.args("paused", .25) + ["--stop-after-steps", "2"])
        with self.assertRaises(ValueError):
            final_checkpoint(self.root / "paused")
        self.run_train(["--resume", str(self.ckpt("paused", 2)), "--output-dir", str(self.root / "resumed")])
        for filename in ("model.safetensors", "expert.safetensors"):
            a = load_file(str(self.ckpt("full", 5) / filename))
            b = load_file(str(self.ckpt("resumed", 5) / filename))
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "Exact resume changed"):
            self.run_train(["--resume", str(self.ckpt("paused", 2)), "--output-dir", str(self.root / "badresume"), "--epochs", "2"])


if __name__ == "__main__":
    unittest.main()
