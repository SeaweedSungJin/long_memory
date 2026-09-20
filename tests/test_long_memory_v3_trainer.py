"""CPU orchestration tests. Toy action expert, REAL v3 modules/labels/checkpoints.

These tests validate causality, resume, and integration, not robot performance.
"""
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from safetensors.torch import load_file
import torch

from gr00t.long_memory import train_v3 as trainer
from gr00t.long_memory.hamlet import isolated_seed


def episode(eid):
    g = torch.Generator().manual_seed(123 + eid)
    n = 12
    return {"episode_id": eid, "cache_fingerprint": "toy-v3", "frames": torch.arange(n + 1) * 2,
            "short": torch.randn(n + 1, 2, 6, generator=g),
            "moment": torch.randn(n + 1, 2, 6, generator=g),
            "state": torch.randn(n + 1, 3, generator=g),
            "actions": torch.randn(n, 2, 4, generator=g),
            "action_mask": torch.ones(n, 2, dtype=torch.bool),
            "transition_valid": torch.ones(n, dtype=torch.bool),
            "decision_mask": torch.ones(n, dtype=torch.bool),
            "targets": torch.randn(n, 3, 4, generator=g)}


def toy_base(path, device):
    with isolated_seed(432, "cpu"):
        head = torch.nn.Linear(6, 1).eval().requires_grad_(False)
    return SimpleNamespace(action_head=head.to(device)), None


def toy_loss(head, ep, decision, fused_short=None, *, seed=None, activation_checkpointing=False):
    if seed is None:
        noise = torch.rand(())
    else:
        with isolated_seed(seed, "cpu"):
            noise = torch.rand(())
    short = ep["short"][decision:decision + 1] if fused_short is None else fused_short
    difference = head(short.mean(1)).mean() - ep["targets"][decision].mean() - noise
    return {"loss": difference.square(), "velocity_mae": difference.abs()}


class TestV3Trainer(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "config.json").write_text(json.dumps({"memory_window": 2, "memory_stride": 2}))
        self.manifest = {"model_path": str(self.base), "fingerprint": "toy-v3",
                         "feature_dim": 6, "state_dim": 3, "action_dim": 4,
                         "splits": {"train": [0, 1], "val": [2, 3]}}
        self.episodes = {i: episode(i) for i in range(4)}
        self.cache = SimpleNamespace(manifest=self.manifest, load=lambda eid: self.episodes[eid])
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(trainer, "EpisodeCache", return_value=self.cache))
        self.stack.enter_context(patch.object(trainer, "checkpoint_identity", return_value={"fixture": "frozen-base"}))
        self.stack.enter_context(patch.object(trainer, "validate_cache_checkpoint", return_value=None))
        self.base_mock = self.stack.enter_context(patch.object(trainer, "load_frozen_hamlet", side_effect=toy_base))
        self.stack.enter_context(patch.object(trainer, "episode_flow_loss", side_effect=toy_loss))
        self.stack.enter_context(patch("gr00t.long_memory.objectives_v3.episode_flow_loss", side_effect=toy_loss))
        self.stack.enter_context(patch.object(trainer.RunLogger, "plot", return_value=None))

    def args(self, name, steps=3, stage=1, initial=None):
        return ["--stage", str(stage), "--cache-dir", str(self.root / "cache"),
                "--output-dir", str(self.root / name), "--device", "cpu", "--max-steps", str(steps),
                "--grad-accum", "1", "--hidden-dim", "8", "--num-heads", "2", "--capacity", "2",
                "--min-fill", "1", "--max-victims", "2", "--warmup-steps", "1",
                "--ranking-start", "1", "--ranking-every", "1", "--writer-bootstrap", "1",
                "--writer-every", "1", "--writer-batch-size", "1", "--teacher-refresh-steps", "2",
                "--policy-ramp-steps", "2", "--future-samples", "2", "--noise-samples", "2",
                "--storage-contexts", "2", "--val-samples", "1", "--val-storage-samples", "1",
                "--eval-steps", "1", "--log-steps", "1", "--plot-steps", "1", "--save-steps", "1",
                "--reader-learning-rate", "0.001", "--writer-learning-rate", "0.001",
                *( ["--init-checkpoint", str(initial)] if initial else [] )]

    def run_train(self, args):
        with redirect_stdout(io.StringIO()):
            return trainer.main(args)

    def ckpt(self, name, step):
        return self.root / name / f"checkpoint-{step:06d}"

    def assert_weights_equal(self, a, b):
        wa, wb = load_file(str(a / "model.safetensors")), load_file(str(b / "model.safetensors"))
        self.assertEqual(wa.keys(), wb.keys())
        for key in wa:
            torch.testing.assert_close(wa[key], wb[key], rtol=0, atol=0, msg=key)
        sa = torch.load(a / "training_state.pt", map_location="cpu", weights_only=True)
        sb = torch.load(b / "training_state.pt", map_location="cpu", weights_only=True)
        self.assertTrue(torch.equal(sa["rng"]["torch"], sb["rng"]["torch"]))
        self.assertEqual(sa["rng"]["python"], sb["rng"]["python"])
        self.assertEqual(sa["optimizer"]["param_groups"], sb["optimizer"]["param_groups"])
        for param, values in sa["optimizer"]["state"].items():
            for key, value in values.items():
                torch.testing.assert_close(value, sb["optimizer"]["state"][param][key], rtol=0, atol=0)

    def test_preflight_no_model_and_no_output(self):
        self.assertEqual(self.run_train(self.args("preflight") + ["--preflight-only"]), 0)
        self.base_mock.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())

    def test_stage1_exact_resume_inherits_options(self):
        self.assertEqual(self.run_train(self.args("full", 3)), 0)
        self.assertEqual(self.run_train(self.args("part", 2)), 0)
        self.assertEqual(self.run_train(["--resume", str(self.ckpt("part", 2)),
            "--max-steps", "3", "--output-dir", str(self.root / "resumed")]), 0)
        self.assert_weights_equal(self.ckpt("full", 3), self.ckpt("resumed", 3))
        rows = [json.loads(x) for x in (self.root / "full/metrics.jsonl").read_text().splitlines()]
        self.assertTrue(any(r["split"] == "val-old-only" for r in rows))
        self.assertTrue(any("action_loss" in r and r["split"] == "train" for r in rows))
        from gr00t.long_memory.live_monitor import _validate_record
        for record in rows:
            _validate_record(record)

    def test_stage2_real_labels_resume_and_no_source_overwrite(self):
        self.assertEqual(self.run_train(self.args("reader", 3)), 0)
        initial = self.ckpt("reader", 3)
        before = hashlib.sha256((initial / "model.safetensors").read_bytes()).hexdigest()
        self.assertEqual(self.run_train(self.args("joint", 3, 2, initial)), 0)
        self.assertEqual(self.run_train(self.args("joint_part", 2, 2, initial)), 0)
        self.assertEqual(self.run_train(["--resume", str(self.ckpt("joint_part", 2)),
            "--max-steps", "3", "--output-dir", str(self.root / "joint_resume")]), 0)
        self.assert_weights_equal(self.ckpt("joint", 3), self.ckpt("joint_resume", 3))
        self.assertEqual(before, hashlib.sha256((initial / "model.safetensors").read_bytes()).hexdigest())
        labels = list((self.root / "joint/labels").rglob("*.json"))
        self.assertTrue(labels)

    def test_refuse_legacy_checkpoint_and_overwriting(self):
        legacy = self.root / "legacy"
        legacy.mkdir()
        (legacy / "checkpoint.json").write_text(json.dumps({"format_version": 1, "config": {"stage": 1}}))
        with self.assertRaisesRegex(ValueError, "legacy"):
            self.run_train(self.args("bad", 2, 2, legacy))
        existing = self.root / "existing"
        existing.mkdir()
        (existing / "user-data.txt").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "NEW"):
            self.run_train(self.args("existing"))
        self.assertEqual((existing / "user-data.txt").read_text(), "preserve")

    def test_changed_resume_option_rejected(self):
        self.run_train(self.args("original", 2))
        with self.assertRaisesRegex(ValueError, "Exact resume"):
            self.run_train(["--resume", str(self.ckpt("original", 2)), "--max-steps", "3",
                            "--output-dir", str(self.root / "bad-resume"), "--reader-learning-rate", "0.03"])

    def test_invalid_selected_cache_fails(self):
        self.episodes[0]["targets"][0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            trainer._flow(toy_base(None, "cpu")[0].action_head, self.episodes[0], 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
