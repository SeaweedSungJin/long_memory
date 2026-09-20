"""CPU integration: matched adaptation scope, immutable bundles and exact resume.

These tests use real archive/LoRA/projector/AdamW/checkpoint code with a tiny
action head. They verify the driver, not robot success or neural convergence.
Native flow/Euler arithmetic has its separate projector-adapter test suite.
"""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from safetensors.torch import load_file, save_file
import torch
from torch import nn

from gr00t.long_memory.checkpoint_v7 import save_checkpoint_v7, v7_checkpoint_info
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora
from gr00t.long_memory.hamlet import checkpoint_identity, isolated_seed
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from run_scripts.robomme import train_archive_projector_v10 as trainer
from run_scripts.robomme import train_archive_deployment_v9 as old
from run_scripts.robomme.checkpoint_projector_v10 import checkpoint_info
from tests import test_long_memory_v4_trainer as fixture


class ToyProjectorHead(fixture.ToyHead):
    def __init__(self):
        super().__init__()
        self.model.proj_out_2 = nn.Linear(6, 4)
        self.num_inference_timesteps = 4

    def forward(self, short):
        a = self.model.transformer_blocks[0].attn1
        q, k, v = a.to_q(short), a.to_k(short), a.to_v(short)
        weights = (q @ k.transpose(-2, -1) / 6 ** .5).softmax(-1)
        return self.model.proj_out_2(a.to_out(weights @ v)).mean()


def toy_base(path, device):
    with isolated_seed(432, "cpu"):
        head = ToyProjectorHead().eval().requires_grad_(False)
    return SimpleNamespace(action_head=head.to(device)), None


def toy_generated(head, ep, decision, fused_short=None, *, seed, action_steps,
                  activation_checkpointing=False):
    generator = torch.Generator().manual_seed(seed)
    short = ep["short"][decision:decision + 1] if fused_short is None else fused_short
    prediction = head(short) + torch.randn((1, 16, 4), generator=generator) * .1
    error = prediction - ep["targets"][decision][None]
    return {"loss": error.square().mean(), "generated_prefix_mae": error.abs().mean(),
            "prediction": prediction, "valid_values": 64}


class ProjectorTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        fixture.TestV4Trainer.setUp(self)
        # Rewrite only this test's temporary base fixture, never real checkpoints.
        state = {"action_head." + k: v for k, v in toy_base(None, "cpu")[0].action_head.state_dict().items()}
        save_file(state, str(self.base / "model.safetensors"))
        (self.base / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {key: "model.safetensors" for key in state}}))
        self.identity = checkpoint_identity(self.base)
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
            {"trainer_variant": "recurrent_memory_v7", "stage": 1, "mode": "archive", "memory": asdict(cfg),
             "expert": asdict(expert), "expert_targets": targets},
            {"base_model": self.identity, "cache_fingerprint": self.manifest["fingerprint"]})
        self.stack.enter_context(patch.object(trainer, "EpisodeCache", return_value=self.cache))
        self.stack.enter_context(patch.object(trainer, "validate_cache_checkpoint", return_value=None))
        self.base_loader = self.stack.enter_context(patch.object(trainer, "load_frozen_hamlet", side_effect=toy_base))
        self.stack.enter_context(patch.object(trainer, "expert_episode_flow_loss", side_effect=fixture.toy_loss))
        self.stack.enter_context(patch.object(trainer, "generated_prefix_metrics", side_effect=toy_generated))

    def args(self, name, enabled=True):
        return ["--cache-dir", str(self.cache.path), "--init-checkpoint", str(self.initial),
                "--output-dir", str(self.root / name), "--device", "cpu", "--max-steps", "4",
                "--query-batch-size", "3", "--train-projector" if enabled else "--no-train-projector",
                "--val-samples", "32", "--val-noise-samples", "1", "--eval-steps", "2",
                "--save-steps", "2", "--log-steps", "1", "--plot-steps", "2",
                "--memory-learning-rate", ".001", "--expert-learning-rate", ".001",
                "--projector-learning-rate", ".001"]

    def run_train(self, argv):
        with redirect_stdout(io.StringIO()):
            return trainer.main(argv)

    def ckpt(self, name, step):
        return self.root / name / f"checkpoint-{step:06d}"

    def test_preflight_readonly_scope_and_initial_format(self):
        self.assertEqual(self.run_train(self.args("preflight") + ["--preflight-only"]), 0)
        self.base_loader.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())
        with self.assertRaisesRegex(ValueError, "inside a source"):
            self.run_train(self.args("bad") + ["--output-dir", str(self.initial / "bad"), "--preflight-only"])

    def test_matched_plan_initial_values_scope_and_frozen_parent(self):
        original = {p.name: trainer.file_hash(p) for p in self.initial.iterdir()}
        for name, enabled in (("control", False), ("projector", True)):
            self.assertEqual(self.run_train(self.args(name, enabled)), 0)
        a = json.loads((self.root / "control/query_plan.json").read_text())
        b = json.loads((self.root / "projector/query_plan.json").read_text())
        self.assertEqual(a, b)
        self.assertEqual(json.loads((self.root / "control/validation-000000.json").read_text()),
                         json.loads((self.root / "projector/validation-000000.json").read_text()))
        for filename in ("model.safetensors", "expert.safetensors", "cvom.safetensors", "projector.safetensors"):
            self.assertEqual((self.ckpt("control", 0) / filename).read_bytes(),
                             (self.ckpt("projector", 0) / filename).read_bytes())
        control = load_file(str(self.ckpt("control", 4) / "projector.safetensors"))
        adapted = load_file(str(self.ckpt("projector", 4) / "projector.safetensors"))
        self.assertTrue(all(not bool(t.any()) for t in control.values()))
        self.assertTrue(all(bool(t.any()) for t in adapted.values()))
        self.assertEqual(original, {p.name: trainer.file_hash(p) for p in self.initial.iterdir()})
        for name in ("control", "projector"):
            info = checkpoint_info(self.base, self.ckpt(name, 4))
            self.assertEqual(info["config"]["trainer_variant"], trainer.DRIVER)
            self.assertEqual(info["metadata"]["train_state"]["processed_queries"], 12)
            self.assertEqual((self.initial / "cvom.safetensors").read_bytes(),
                             (self.ckpt(name, 4) / "cvom.safetensors").read_bytes())
            records = [json.loads(line) for line in (self.root / name / "metrics.jsonl").read_text().splitlines()]
            baselines = [r["generated_observed_prefix_mse"] for r in records if r["split"] == "comparison/baseline"]
            self.assertEqual(len(set(baselines)), 1)
            train = [r for r in records if r["split"] == "train"]
            self.assertTrue(any(r["memory_grad_norm"] > 0 and r["expert_grad_norm"] > 0 for r in train))
            self.assertEqual(any(r["projector_grad_norm"] > 0 for r in train), name == "projector")

    def test_exact_resume_and_off_cadence_pause_selection(self):
        self.run_train(self.args("full"))
        # Pausing step1 is NOT a scheduled validation boundary (eval every2).
        self.run_train(self.args("part") + ["--stop-after-steps", "1"])
        self.assertFalse((self.root / "part/validation-000001.json").exists())
        self.run_train(["--resume", str(self.ckpt("part", 1)), "--output-dir", str(self.root / "resumed")])
        for filename in ("model.safetensors", "expert.safetensors", "cvom.safetensors", "projector.safetensors"):
            left, right = load_file(str(self.ckpt("full", 4) / filename)), load_file(str(self.ckpt("resumed", 4) / filename))
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        fixture.TestV4Trainer.assert_exact_resume(self, self.ckpt("full", 4), self.ckpt("resumed", 4))
        for step in (2, 4):
            self.assertEqual(json.loads((self.root / "full" / f"validation-{step:06d}.json").read_text()),
                             json.loads((self.root / "resumed" / f"validation-{step:06d}.json").read_text()))
        full = json.loads((self.ckpt("full", 4) / "checkpoint.json").read_text())["metadata"]["train_state"]
        resumed = json.loads((self.ckpt("resumed", 4) / "checkpoint.json").read_text())["metadata"]["train_state"]
        self.assertEqual(full["best_generated_prefix_mse"], resumed["best_generated_prefix_mse"])
        resume = ["--resume", str(self.ckpt("part", 1)), "--output-dir", str(self.root / "invalid"), "--preflight-only"]
        for override in (["--max-steps", "5"], ["--projector-learning-rate", ".002"], ["--no-train-projector"]):
            with self.assertRaisesRegex(ValueError, "Exact resume option"):
                self.run_train(resume + override)

    def test_legacy_loader_and_resume_refuse_new_payload(self):
        self.run_train(self.args("new"))
        with self.assertRaisesRegex(ValueError, "recurrent_memory_v7"):
            v7_checkpoint_info(self.base, self.ckpt("new", 4))
        with self.assertRaisesRegex(ValueError, "archive_deployment_v9"):
            old.parse_args(["--resume", str(self.ckpt("new", 4)), "--output-dir", str(self.root / "bad")])

    def test_failure_never_publishes_partial_optimizer_update(self):
        def broken(*args, **kwargs):
            if torch.is_grad_enabled():
                raise RuntimeError("simulated projector training failure")
            return fixture.toy_loss(*args, **kwargs)
        with patch.object(trainer, "expert_episode_flow_loss", side_effect=broken):
            with self.assertRaisesRegex(RuntimeError, "simulated projector"):
                self.run_train(self.args("failure"))
        self.assertTrue(self.ckpt("failure", 0).is_dir())
        self.assertFalse(self.ckpt("failure", 1).exists())
        self.assertTrue((self.root / "failure/failure.json").is_file())


if __name__ == "__main__":
    unittest.main()
