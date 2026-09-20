"""Small CPU integration checks, not RoboMME performance measurements."""
from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from run_scripts.robomme import train_representation_v18 as trainer
from run_scripts.robomme.checkpoint_representation_v18 import checkpoint_info_v18
from tests import test_archive_deployment_trainer_v9 as fixture
from tests import test_long_memory_v4_trainer as toy


class TrainingV18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        fixture.DeploymentTrainerTests.setUp(self)
        self.stack.enter_context(patch.object(trainer, "EpisodeCache", return_value=self.cache))
        self.stack.enter_context(patch.object(trainer, "validate_cache_checkpoint"))
        self.loader = self.stack.enter_context(patch.object(trainer, "load_frozen_hamlet", side_effect=fixture.toy_base))
        self.stack.enter_context(patch.object(trainer, "expert_episode_flow_loss", side_effect=toy.toy_loss))
        self.stack.enter_context(patch.object(trainer, "generated_prefix_objective", side_effect=fixture.toy_generated))

    def args(self, name, representation="short"):
        return ["--cache-dir", str(self.cache.path), "--output-dir", str(self.root / name),
            "--representation", representation, "--hidden-dim", "8", "--num-heads", "2", "--capacity-events", "3",
            "--lora-rank", "2", "--lora-alpha", "4", "--device", "cpu", "--max-steps", "4",
            "--query-batch-size", "2", "--val-samples", "2", "--eval-steps", "2", "--save-steps", "2",
            "--log-steps", "1", "--plot-steps", "2", "--memory-learning-rate", ".001", "--expert-learning-rate", ".001"]

    def run_train(self, argv):
        with redirect_stdout(io.StringIO()):
            return trainer.main(argv)

    def ckpt(self, name, step):
        return self.root / name / f"checkpoint-{step:06d}"

    def test_preflight_no_model_no_output(self):
        for name, rep in (("a", "short"), ("b", "adapted_short"), ("c", "moment")):
            self.assertEqual(self.run_train(self.args(name, rep) + ["--preflight-only"]), 0)
            self.assertFalse((self.root / name).exists())
        self.loader.assert_not_called()
        with self.assertRaises(ValueError):
            self.run_train(self.args("bad") + ["--output-dir", str(self.cache.path / "bad")])

    def test_matched_plan_initial_state_and_real_updates(self):
        original = {p.name: trainer.file_hash(p) for p in self.base.iterdir()}
        for name, rep in (("a", "short"), ("c", "moment")):
            self.assertEqual(self.run_train(self.args(name, rep)), 0)
            info = checkpoint_info_v18(self.base, self.ckpt(name, 4))
            self.assertEqual(info["config"]["representation"]["representation"], rep)
            before = load_file(str(self.ckpt(name, 0) / "model.safetensors"))
            after = load_file(str(self.ckpt(name, 4) / "model.safetensors"))
            self.assertFalse(torch.equal(before["memory.fusion_projection.weight"], after["memory.fusion_projection.weight"]))
        left = json.loads((self.root / "a/query_plan.json").read_text())
        right = json.loads((self.root / "c/query_plan.json").read_text())
        self.assertEqual(left, right)
        for filename in ("model.safetensors", "expert.safetensors"):
            a = load_file(str(self.ckpt("a", 0) / filename))
            c = load_file(str(self.ckpt("c", 0) / filename))
            self.assertEqual(a.keys(), c.keys())
            for name in a:
                torch.testing.assert_close(a[name], c[name], rtol=0, atol=0)
        self.assertEqual(original, {p.name: trainer.file_hash(p) for p in self.base.iterdir()})
        with self.assertRaises(FileExistsError):
            self.run_train(self.args("a"))

    def test_exact_resume_and_tamper_rejection(self):
        self.run_train(self.args("full"))
        self.run_train(self.args("pause") + ["--stop-after-steps", "2"])
        self.run_train(["--resume", str(self.ckpt("pause", 2)), "--output-dir", str(self.root / "resumed")])
        for filename in ("model.safetensors", "expert.safetensors"):
            a = load_file(str(self.ckpt("full", 4) / filename))
            b = load_file(str(self.ckpt("resumed", 4) / filename))
            for name in a:
                torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "Exact resume changed"):
            self.run_train(["--resume", str(self.ckpt("pause", 2)), "--output-dir", str(self.root / "badresume"), "--max-steps", "8"])
        path = self.ckpt("full", 4) / "expert.safetensors"
        with path.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "payload modified"):
            checkpoint_info_v18(self.base, self.ckpt("full", 4))


if __name__ == "__main__":
    unittest.main()
