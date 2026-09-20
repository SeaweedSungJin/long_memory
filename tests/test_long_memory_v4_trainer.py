"""CPU integration tests: real memory/LoRA/labels/checkpoints, toy action loss.

These exercise optimizer ownership, original-baseline comparisons, immutable
lineage, and exact resume. They do not claim simulator or convergence results.
"""
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from safetensors.torch import load_file, save_file
import torch
from torch import nn

from gr00t.long_memory import train_v4 as trainer
from gr00t.long_memory.core_v3 import ActionValueMemory, MemoryV3Config
from gr00t.long_memory.expert_v4 import (LoRAConfig, adapter_disabled, expert_parameters,
                                       install_expert_lora, set_expert_trainable)
from gr00t.long_memory.hamlet import checkpoint_identity, isolated_seed
from gr00t.long_memory.monitoring import save_checkpoint


def episode(eid):
    g = torch.Generator().manual_seed(123 + eid)
    n = 12
    return {"episode_id": eid, "cache_fingerprint": "toy-v4", "frames": torch.arange(n + 1) * 2,
            "short": torch.randn(n + 1, 2, 6, generator=g),
            "moment": torch.randn(n + 1, 2, 6, generator=g),
            "state": torch.randn(n + 1, 3, generator=g),
            "actions": torch.randn(n, 2, 4, generator=g),
            "action_mask": torch.ones(n, 2, dtype=torch.bool),
            "transition_valid": torch.ones(n, dtype=torch.bool),
            "decision_mask": torch.ones(n, dtype=torch.bool),
            "targets": torch.randn(n, 3, 4, generator=g)}


class ToyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        block = nn.Module()
        block.attn1 = nn.Module()
        for name in ("to_q", "to_k", "to_v"):
            setattr(block.attn1, name, nn.Linear(6, 6))
        block.attn1.to_out = nn.Sequential(nn.Linear(6, 6))
        self.model.transformer_blocks = nn.ModuleList([block])

    def forward(self, short):
        a = self.model.transformer_blocks[0].attn1
        q, k, v = a.to_q(short), a.to_k(short), a.to_v(short)
        weights = (q @ k.transpose(-2, -1) / 6 ** 0.5).softmax(-1)
        return a.to_out(weights @ v).mean()


def toy_base(path, device):
    with isolated_seed(432, "cpu"):
        head = ToyHead().eval().requires_grad_(False)
    return SimpleNamespace(action_head=head.to(device)), None


def toy_loss(head, ep, decision, fused_short=None, *, seed=None, activation_checkpointing=False):
    if seed is None:
        noise = torch.rand(())
    else:
        with isolated_seed(seed, "cpu"):
            noise = torch.rand(())
    short = ep["short"][decision:decision + 1] if fused_short is None else fused_short
    difference = head(short) - ep["targets"][decision].mean() - noise
    return {"loss": difference.square(), "velocity_mae": difference.abs()}


class TestV4Trainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "config.json").write_text(json.dumps({"hamlet_mode": "finetune", "memory_type": "moment_token",
            "mem_cond_type": "cross_attn", "n_moment_tokens": 2, "backbone_embedding_dim": 6,
            "memory_window": 2, "memory_stride": 2}))
        (self.base / "processor_config.json").write_text(json.dumps({"max_state_dim": 3, "max_action_dim": 4}))
        (self.base / "statistics.json").write_text("{}")
        (self.base / "embodiment_id.json").write_text("{}")
        state = {"action_head." + k: v for k, v in toy_base(None, "cpu")[0].action_head.state_dict().items()}
        save_file(state, str(self.base / "model.safetensors"))
        (self.base / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {key: "model.safetensors" for key in state}}))
        self.identity = checkpoint_identity(self.base)
        self.manifest = {"model_path": str(self.base), "fingerprint": "toy-v4",
                         "feature_dim": 6, "state_dim": 3, "action_dim": 4,
                         "splits": {"train": [0, 1], "val": [2, 3]}}
        self.episodes = {i: episode(i) for i in range(4)}
        self.cache = SimpleNamespace(manifest=self.manifest, load=lambda eid: self.episodes[eid])
        self.cfg = MemoryV3Config(feature_dim=6, state_dim=3, action_dim=4, hidden_dim=8,
                                  num_heads=2, capacity=2, min_fill=1, max_victims=2, time_scale=2)
        with isolated_seed(142, "cpu"):
            memory = ActionValueMemory(self.cfg)
            # Simulate an already-learned v3 reader, not a zero residual adapter.
            with torch.no_grad():
                for p in memory.reader_parameters():
                    p.add_(torch.randn_like(p) * .005)
        self.initial = save_checkpoint(self.root / "v3", 5, memory, None,
            {"trainer_variant": "action_value_v3", "stage": 1, "memory": asdict(self.cfg)},
            {"base_model": self.identity, "cache_fingerprint": "toy-v4"}, keep_last=None)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(trainer, "EpisodeCache", return_value=self.cache))
        self.stack.enter_context(patch.object(trainer, "validate_cache_checkpoint", return_value=None))
        self.base_mock = self.stack.enter_context(patch.object(trainer, "load_frozen_hamlet", side_effect=toy_base))
        self.stack.enter_context(patch.object(trainer, "expert_episode_flow_loss", side_effect=toy_loss))
        self.stack.enter_context(patch.object(trainer.RunLogger, "plot", return_value=None))

    def args(self, name, steps=3, stage=1, initial=None):
        return ["--stage", str(stage), "--cache-dir", str(self.root / "cache"),
                "--init-checkpoint", str(initial or self.initial),
                "--output-dir", str(self.root / name), "--device", "cpu", "--max-steps", str(steps),
                "--grad-accum", "1", "--lora-rank", "2", "--lora-alpha", "4", "--warmup-steps", "1",
                "--writer-batch-size", "1", "--context-refresh-steps", "2",
                "--future-samples", "2", "--noise-samples", "2", "--storage-contexts", "2",
                "--val-samples", "1", "--val-storage-samples", "1", "--eval-steps", "1",
                "--log-steps", "1", "--plot-steps", "1", "--save-steps", "1",
                "--reader-learning-rate", "0.001", "--expert-learning-rate", "0.001",
                "--writer-learning-rate", "0.001"]

    def run_train(self, args):
        with redirect_stdout(io.StringIO()):
            return trainer.main(args)

    def ckpt(self, name, step):
        return self.root / name / f"checkpoint-{step:06d}"

    def rows(self, name):
        return [json.loads(line) for line in (self.root / name / "metrics.jsonl").read_text().splitlines()]

    def assert_exact_resume(self, a, b):
        for filename in ("model.safetensors", "expert.safetensors"):
            left, right = load_file(str(a / filename)), load_file(str(b / filename))
            self.assertEqual(left.keys(), right.keys())
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0, msg=name)
        sa = torch.load(a / "training_state.pt", map_location="cpu", weights_only=True)
        sb = torch.load(b / "training_state.pt", map_location="cpu", weights_only=True)
        self.assertTrue(torch.equal(sa["rng"]["torch"], sb["rng"]["torch"]))
        self.assertEqual(sa["rng"]["python"], sb["rng"]["python"])
        self.assertEqual(sa["optimizer"]["param_groups"], sb["optimizer"]["param_groups"])
        for param, values in sa["optimizer"]["state"].items():
            for key, value in values.items():
                torch.testing.assert_close(value, sb["optimizer"]["state"][param][key], rtol=0, atol=0)

    def test_preflight_is_read_only_and_requires_initial_reader(self):
        self.assertEqual(self.run_train(self.args("preflight") + ["--preflight-only"]), 0)
        self.base_mock.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())
        with self.assertRaisesRegex(ValueError, "init-checkpoint"):
            self.run_train(["--stage", "1", "--cache-dir", "cache", "--output-dir", str(self.root / "missing")])

    def test_stage1_updates_reader_expert_not_writer_and_preserves_v3(self):
        before = hashlib.sha256((self.initial / "model.safetensors").read_bytes()).hexdigest()
        self.assertEqual(self.run_train(self.args("joint", 2)), 0)
        start = load_file(str(self.ckpt("joint", 0) / "model.safetensors"))
        end = load_file(str(self.ckpt("joint", 2) / "model.safetensors"))
        self.assertTrue(any(not torch.equal(start[k], end[k]) for k in start if not k.startswith("writer.")))
        self.assertTrue(all(torch.equal(start[k], end[k]) for k in start if k.startswith("writer.")))
        expert = load_file(str(self.ckpt("joint", 2) / "expert.safetensors"))
        self.assertTrue(any(bool(v.any()) for k, v in expert.items() if k.endswith("lora_B")))
        self.assertEqual(before, hashlib.sha256((self.initial / "model.safetensors").read_bytes()).hexdigest())
        rows = self.rows("joint")
        self.assertTrue(any(r.get("expert_grad_norm", 0) > 0 and r.get("reader_grad_norm", 0) > 0 for r in rows))
        baseline = [r["baseline_action_loss"] for r in rows if r["split"] == "val"]
        self.assertEqual(len(set(baseline)), 1)
        from gr00t.long_memory.live_monitor import _validate_record
        for row in rows:
            _validate_record(row)

    def test_stage1_exact_resume_inherits_options(self):
        self.run_train(self.args("full", 3))
        self.run_train(self.args("part", 2))
        self.run_train(["--resume", str(self.ckpt("part", 2)), "--max-steps", "3",
                        "--output-dir", str(self.root / "resumed")])
        self.assert_exact_resume(self.ckpt("full", 3), self.ckpt("resumed", 3))

    def test_stage2_only_writer_frozen_teacher_and_exact_resume(self):
        self.run_train(self.args("reader", 2))
        source = self.ckpt("reader", 2)
        self.run_train(self.args("writer", 3, 2, source))
        self.run_train(self.args("writer_part", 2, 2, source))
        self.run_train(["--resume", str(self.ckpt("writer_part", 2)), "--max-steps", "3",
                        "--output-dir", str(self.root / "writer_resumed")])
        self.assert_exact_resume(self.ckpt("writer", 3), self.ckpt("writer_resumed", 3))
        begin = load_file(str(source / "model.safetensors"))
        end = load_file(str(self.ckpt("writer", 3) / "model.safetensors"))
        self.assertTrue(all(torch.equal(begin[k], end[k]) for k in begin if not k.startswith("writer.")))
        self.assertTrue(any(not torch.equal(begin[k], end[k]) for k in begin if k.startswith("writer.")))
        self.assertEqual((source / "expert.safetensors").read_bytes(),
                         (self.ckpt("writer", 3) / "expert.safetensors").read_bytes())
        records = self.rows("writer")
        self.assertTrue(any(r.get("writer_grad_norm", 0) > 0 for r in records))
        self.assertTrue(all(r.get("reader_grad_norm", 0) == r.get("expert_grad_norm", 0) == 0 for r in records))
        self.assertTrue(any("teacher_keep_strict_best_fraction" in r and "predicted_keep_fraction" in r for r in records))
        label_files = list((self.root / "writer/labels/fixed-adapted-expert").glob("storage-*.json"))
        self.assertTrue(label_files)
        manifests = json.loads((self.root / "writer/labels/fixed-adapted-expert/manifest.json").read_text())
        self.assertEqual(manifests["identity"]["trainer_variant"], "action_expert_v4")
        self.assertEqual(len(manifests["identity"]["expert_adapters_sha256"]), 64)

    def test_expert_only_control_never_changes_memory_and_cannot_start_writer(self):
        self.run_train(self.args("control", 2) + ["--reader-mode", "none"])
        before = load_file(str(self.initial / "model.safetensors"))
        after = load_file(str(self.ckpt("control", 2) / "model.safetensors"))
        for name in before:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
        val = [r for r in self.rows("control") if r["split"] == "val"]
        self.assertTrue(all(r["action_loss"] == r["expert_no_memory_action_loss"] for r in val))
        with self.assertRaisesRegex(ValueError, "Expert-only"):
            self.run_train(self.args("badwriter", 1, 2, self.ckpt("control", 2)))

    def test_stage2_rejects_old_frozen_expert_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "adapted Expert"):
            self.run_train(self.args("badwriter", 1, 2))

    def test_output_and_resume_option_safety(self):
        target = self.root / "existing"
        target.mkdir()
        (target / "user.txt").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "NEW"):
            self.run_train(self.args("existing"))
        self.assertEqual((target / "user.txt").read_text(), "preserve")
        self.run_train(self.args("original", 1))
        with self.assertRaisesRegex(ValueError, "Exact resume option"):
            self.run_train(["--resume", str(self.ckpt("original", 1)), "--max-steps", "2",
                            "--expert-learning-rate", ".5", "--output-dir", str(self.root / "badresume")])

    def test_no_memory_validation_is_not_original_baseline(self):
        memory = ActionValueMemory(self.cfg)
        head = toy_base(None, "cpu")[0].action_head
        install_expert_lora(head, LoRAConfig(2, 4))
        set_expert_trainable(head, True)
        with torch.no_grad():
            for name, p in head.named_parameters():
                if name.endswith("lora_B"):
                    p.fill_(.5)
        result = trainer.validate(memory, head, lambda i: self.episodes[i], [[2, 4]], "all", 700, 2, "none")
        self.assertEqual(result["action_loss"], result["expert_no_memory_action_loss"])
        self.assertNotEqual(result["action_loss"], result["baseline_action_loss"])
        with adapter_disabled(head):
            expected = toy_loss(head, self.episodes[2], 4, seed=700)["loss"]
        self.assertEqual(float(expected), result["baseline_action_loss"])

    def test_tie_aware_keep_diagnostics_do_not_force_rejection(self):
        aux = {"loss": torch.tensor(-.1), "metrics": {}}
        label = {"option_mean_losses": [.01, .01], "best_option": 0, "options": [[0], [0, 1]]}
        result = trainer.storage_diagnostics(aux, label, torch.tensor([0., 1.]), 1e-5)
        self.assertEqual(result["teacher_keep_strict_best_fraction"], 0)
        self.assertEqual(result["teacher_keep_tie_best_fraction"], 1)
        self.assertEqual(result["predicted_keep_fraction"], 0)
        label["option_mean_losses"] = [.01, .02]
        result = trainer.storage_diagnostics(aux, label, torch.tensor([0., 1.]), 1e-5)
        self.assertEqual(result["teacher_keep_strict_best_fraction"], 1)
        self.assertGreater(result["storage_selected_regret"], 0)


if __name__ == "__main__":
    unittest.main()
