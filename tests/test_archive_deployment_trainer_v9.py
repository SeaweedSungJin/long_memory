"""CPU pilot contracts: real archive/LoRA/optimizer/bundle, toy Expert losses.

Native Euler parity and gradients are separately tested in the objective suite.
These tests do not measure RoboMME task success or claim neural convergence.
"""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from safetensors.torch import load_file
import torch

from gr00t.long_memory import train_v7 as old_trainer
from gr00t.long_memory.checkpoint_v7 import save_checkpoint_v7, v7_checkpoint_info
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora
from gr00t.long_memory.hamlet import isolated_seed
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from run_scripts.robomme import train_archive_deployment_v9 as trainer
from tests import test_long_memory_v4_trainer as fixture


def toy_base(path, device):
    base, processor = fixture.toy_base(path, device)
    base.action_head.num_inference_timesteps = 4
    return base, processor


def toy_generated(head, ep, decision, fused_short=None, *, seed, action_steps,
                  activation_checkpointing=True):
    # Explicit local generation RNG; unlike a teacher interpolant, its input
    # trajectory does not depend on the GT subsequently used by the error.
    g = torch.Generator().manual_seed(seed)
    short = ep["short"][decision:decision + 1] if fused_short is None else fused_short
    prediction = head(short) + torch.randn((1, 16, 4), generator=g) * .1
    target = ep["targets"][decision][None]
    error = prediction - target
    loss = error.square().mean()
    return {"loss": loss, "generated_prefix_mae": error.abs().mean(),
            "prediction": prediction, "valid_values": 64}


class DeploymentTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        fixture.TestV4Trainer.setUp(self)
        self.cache.path = self.root / "cache"
        self.manifest["action_steps"] = 16
        self.manifest["splits"]["val"] = list(range(2, 34))
        for eid in range(34):
            ep = fixture.episode(eid)
            g = torch.Generator().manual_seed(500 + eid)
            ep.update(is_demo=torch.zeros(13, dtype=torch.bool), frames=torch.arange(13) * 16,
                actions=torch.randn(12, 16, 4, generator=g), action_mask=torch.ones(12, 16, dtype=torch.bool),
                targets=torch.randn(12, 16, 4, generator=g), target_mask=torch.ones(12, 16, 4, dtype=torch.bool))
            self.episodes[eid] = ep
        cfg = MemoryV7Config(feature_dim=6, state_dim=3, num_short_tokens=2, hidden_dim=8,
                             num_heads=2, capacity=4, time_scale=2)
        with isolated_seed(91, "cpu"):
            memory, cvom = RecurrentMemoryV7(cfg), CVOMV7(cfg)
        head = toy_base(None, "cpu")[0].action_head
        expert = LoRAConfig(2, 4)
        targets = install_expert_lora(head, expert)
        self.initial = save_checkpoint_v7(self.root / "archive_parent", 1250, memory, head, cvom, None,
            {"trainer_variant": trainer.ARCHITECTURE, "stage": 1, "mode": "archive", "memory": asdict(cfg),
             "expert": asdict(expert), "expert_targets": targets},
            {"base_model": self.identity, "cache_fingerprint": self.manifest["fingerprint"]})
        self.stack.enter_context(patch.object(trainer, "EpisodeCache", return_value=self.cache))
        self.stack.enter_context(patch.object(trainer, "validate_cache_checkpoint", return_value=None))
        self.base_loader = self.stack.enter_context(patch.object(trainer, "load_frozen_hamlet", side_effect=toy_base))
        self.stack.enter_context(patch.object(trainer, "expert_episode_flow_loss", side_effect=fixture.toy_loss))
        self.generated = self.stack.enter_context(patch.object(trainer, "generated_prefix_objective", side_effect=toy_generated))

    def args(self, name, aux=0.):
        return ["--cache-dir", str(self.cache.path), "--init-checkpoint", str(self.initial),
                "--output-dir", str(self.root / name), "--device", "cpu", "--max-steps", "4",
                "--query-batch-size", "3", "--aux-weight", str(aux), "--val-samples", "32",
                "--val-noise-samples", "1", "--eval-steps", "2", "--save-steps", "2",
                "--log-steps", "1", "--plot-steps", "2", "--memory-learning-rate", ".001",
                "--expert-learning-rate", ".001"]

    def run_train(self, argv):
        with redirect_stdout(io.StringIO()):
            return trainer.main(argv)

    def checkpoint(self, name, step):
        return self.root / name / f"checkpoint-{step:06d}"

    def test_preflight_no_model_or_output_and_protects_source(self):
        self.assertEqual(self.run_train(self.args("preflight", .1) + ["--preflight-only"]), 0)
        self.base_loader.assert_not_called()
        self.generated.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())
        with self.assertRaisesRegex(ValueError, "inside a source"):
            self.run_train(self.args("unsafe") + ["--output-dir", str(self.initial / "unsafe"), "--preflight-only"])
        (self.root / "existing").mkdir()
        with self.assertRaises(FileExistsError):
            self.run_train(self.args("existing") + ["--preflight-only"])

    def test_matched_arms_full_plan_and_old_checkpoint_compatibility(self):
        original = {p.name: trainer.file_hash(p) for p in self.initial.iterdir()}
        for name, aux in (("flow", 0.), ("aux", .1)):
            self.run_train(self.args(name, aux))
        a = json.loads((self.root / "flow/query_plan.json").read_text())
        b = json.loads((self.root / "aux/query_plan.json").read_text())
        self.assertEqual(a, b)
        self.assertEqual(a["train_query_count"], 24)
        self.assertEqual(len(a["validation"]), 32)
        self.assertEqual(len({eid for eid, _ in a["validation"]}), 32)
        self.assertFalse({eid for eid, _ in a["train"]} & {eid for eid, _ in a["validation"]})
        info = v7_checkpoint_info(self.base, self.checkpoint("aux", 4), expected_stage=1)
        self.assertEqual(info["config"]["trainer_variant"], trainer.ARCHITECTURE)
        self.assertEqual(info["config"]["driver_variant"], trainer.DRIVER)
        self.assertEqual(info["config"]["objective"]["generated_observed_prefix_weight"], .1)
        self.assertEqual(original, {p.name: trainer.file_hash(p) for p in self.initial.iterdir()})
        self.assertEqual((self.initial / "cvom.safetensors").read_bytes(),
                         (self.checkpoint("aux", 4) / "cvom.safetensors").read_bytes())
        for name in ("flow", "aux"):
            rows = [json.loads(line) for line in (self.root / name / "metrics.jsonl").read_text().splitlines()]
            train = [r for r in rows if r["split"] == "train"]
            self.assertTrue(any(r["memory_grad_norm"] > 0 and r["expert_grad_norm"] > 0 for r in train))
            self.assertEqual(all("generated_observed_prefix_mse" in r for r in train), name == "aux")
            baseline = [r for r in rows if r["split"] == "comparison/baseline"]
            self.assertEqual(len({r["generated_observed_prefix_mse"] for r in baseline}), 1)
            record = json.loads((self.root / name / "validation-000004.json").read_text())
            self.assertEqual(len(record["records"]), 32 * 3)
            self.assertEqual(record["selection_metric"], trainer.SELECTION)
        self.assertFalse(torch.equal(load_file(str(self.checkpoint("flow", 4) / "expert.safetensors"))["model.transformer_blocks.0.attn1.to_q.lora_B"],
                                     load_file(str(self.checkpoint("aux", 4) / "expert.safetensors"))["model.transformer_blocks.0.attn1.to_q.lora_B"]))

    def test_exact_resume_tensor_optimizer_rng_cursor_and_lr_horizon(self):
        self.run_train(self.args("full", .1))
        self.run_train(self.args("part", .1) + ["--stop-after-steps", "2"])
        resume = ["--resume", str(self.checkpoint("part", 2)), "--output-dir", str(self.root / "resumed")]
        self.run_train(resume)
        for filename in ("model.safetensors", "expert.safetensors", "cvom.safetensors"):
            a = load_file(str(self.checkpoint("full", 4) / filename))
            b = load_file(str(self.checkpoint("resumed", 4) / filename))
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
        fixture.TestV4Trainer.assert_exact_resume(self, self.checkpoint("full", 4), self.checkpoint("resumed", 4))
        for option, value in (("--max-steps", "5"), ("--aux-weight", ".2"), ("--seed", "9043")):
            with self.assertRaisesRegex(ValueError, "Exact resume option"):
                self.run_train(resume + ["--output-dir", str(self.root / "bad"), option, value, "--preflight-only"])
        saved = json.loads((self.checkpoint("resumed", 4) / "checkpoint.json").read_text())
        self.assertEqual(saved["metadata"]["train_state"]["processed_queries"], 12)
        self.assertEqual(saved["metadata"]["original_continuation_parent"]["step"], 1250)

    def test_old_trainer_cannot_exact_resume_new_objective(self):
        self.run_train(self.args("new"))
        with patch.object(old_trainer, "EpisodeCache", return_value=self.cache), \
             patch.object(old_trainer, "validate_cache_checkpoint"), self.assertRaisesRegex(ValueError, "Exact resume option"):
            old_trainer.main(["--resume", str(self.checkpoint("new", 2)),
                "--output-dir", str(self.root / "old_bad"), "--preflight-only"])
        self.assertFalse((self.root / "old_bad").exists())

    def test_zero_weight_returns_exact_flow_tensor_gradient_and_rng(self):
        p = torch.nn.Parameter(torch.tensor(2.))
        loss = p.square()
        rng = torch.get_rng_state().clone()
        with patch.object(trainer, "expert_episode_flow_loss", return_value={"loss": loss, "velocity_mae": p.abs()}), \
             patch.object(trainer, "generated_prefix_objective", side_effect=AssertionError("zero weight called aux")):
            returned, row = trainer.query_objective(None, self.episodes[0], 1, None,
                flow_seed=12, generation_seed=999, aux_weight=0., action_steps=16, activation_checkpointing=True)
        self.assertIs(returned, loss)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(float(torch.autograd.grad(returned, p)[0]), 4.)
        self.assertNotIn("generated_observed_prefix_mse", row)

    def test_generated_components_ignore_unobserved_tail_and_padding(self):
        ep = {"targets": torch.zeros(1, 4, 10), "target_mask": torch.ones(1, 4, 10, dtype=torch.bool),
              "action_mask": torch.tensor([[True, False]])}
        ep["target_mask"][:, :, 8:] = False
        pred = torch.full((1, 4, 10), 1000.)
        pred[:, 0, :7] = 2.
        pred[:, 0, 7] = 3.
        result = {"prediction": pred, "loss": torch.tensor(37 / 8), "generated_prefix_mae": torch.tensor(17 / 8)}
        row = trainer.generated_components(ep, 0, result, 2)
        self.assertEqual(row["generated_valid_values"], 8.)
        self.assertEqual(row["generated_observed_joint7_mse"], 4.)
        self.assertEqual(row["generated_observed_gripper1_mse"], 9.)

    def test_nonfinite_optimizer_rejected(self):
        p = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.AdamW([p])
        p.square().backward(); optimizer.step()
        trainer.assert_finite_optimizer(optimizer)
        optimizer.state[p]["exp_avg"].fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            trainer.assert_finite_optimizer(optimizer)

    def test_failure_publishes_no_partial_checkpoint(self):
        def failing(*args, **kwargs):
            if torch.is_grad_enabled():
                raise RuntimeError("simulated training failure")
            return fixture.toy_loss(*args, **kwargs)
        with patch.object(trainer, "expert_episode_flow_loss", side_effect=failing), self.assertRaisesRegex(RuntimeError, "simulated"):
            self.run_train(self.args("failed"))
        self.assertTrue(self.checkpoint("failed", 0).is_dir())
        self.assertFalse(self.checkpoint("failed", 1).exists())
        self.assertTrue((self.root / "failed/failure.json").is_file())
        self.assertFalse(list((self.root / "failed").glob(".checkpoint-*")))

    def test_strict_options_and_split_errors(self):
        for option, value in (("--val-samples", "31"), ("--aux-weight", "nan"),
                              ("--max-steps", "0"), ("--warmup-fraction", "nan")):
            with self.assertRaises(ValueError):
                trainer.validate_options(trainer.parse_args(self.args("unused") + [option, value]))
        self.manifest["splits"]["val"].append(0)
        with self.assertRaisesRegex(ValueError, "episode-disjoint"):
            self.run_train(self.args("overlap") + ["--preflight-only"])


if __name__ == "__main__":
    unittest.main()
