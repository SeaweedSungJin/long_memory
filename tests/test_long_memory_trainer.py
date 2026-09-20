"""Tiny CPU end-to-end orchestration checks, not real HAMLET performance tests.

The memory, replay, CVoM, optimizer, checkpoint and JSONL paths are real. Only
the external feature cache and multi-billion-parameter frozen expert are faked;
plot rendering has separate monitoring tests and is skipped here for speed.
"""

from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from safetensors.torch import load_file, save_file
import torch

from gr00t.long_memory import hamlet, train
from gr00t.long_memory.hamlet import isolated_seed


def fake_episode(eid):
    generator = torch.Generator().manual_seed(200 + eid)
    events, tokens, width = 4, 2, 6
    short = torch.randn(events + 1, tokens, width, generator=generator)
    return {
        "episode_id": eid,
        "short": short,
        "moment": torch.randn(events + 1, tokens, width, generator=generator),
        "state": torch.randn(events + 1, 3, generator=generator),
        "actions": torch.randn(events, 2, 4, generator=generator),
        "action_mask": torch.ones(events, 2, dtype=torch.bool),
        "transition_valid": torch.ones(events, dtype=torch.bool),
        "decision_mask": torch.ones(events, dtype=torch.bool),
        "targets": torch.randn(events, 3, 4, generator=generator),
    }


def fake_frozen_model(path, device):
    with isolated_seed(198, "cpu"):
        head = torch.nn.Linear(6, 1).eval().requires_grad_(False)
    return SimpleNamespace(action_head=head.to(device)), None


def fake_flow_loss(head, episode, decision, fused_short=None, *, seed=None,
                   activation_checkpointing=False):
    short = episode["short"][decision][None] if fused_short is None else fused_short
    if seed is None:
        noise = torch.rand(())
    else:
        with isolated_seed(seed, "cpu"):
            noise = torch.rand(())
    prediction = head(short.mean(1))
    difference = prediction - episode["targets"][decision].mean() - noise
    return {"loss": difference.square().mean(), "velocity_mae": difference.abs().mean()}


class TestTinyTrainer(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "config.json").write_text(json.dumps({"memory_window": 2}))
        save_file({"weight": torch.ones(1)}, str(self.base / "model.safetensors"))
        checkpoint_signatures = []
        for path in sorted(self.base.iterdir()):
            stat = path.stat()
            signature = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if path.suffix == ".json":
                signature["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            checkpoint_signatures.append(signature)
        self.cache_path = self.root / "cache"
        self.manifest = {
            "model_path": str(self.base), "fingerprint": "mock-cache-v1",
            "feature_dim": 6, "state_dim": 3, "action_dim": 4,
            "splits": {"train": [0, 1], "val": [2, 3]},
            "identity": {"checkpoint": checkpoint_signatures},
        }
        episodes = {eid: fake_episode(eid) for eid in range(4)}
        cache = SimpleNamespace(manifest=self.manifest, load=lambda eid: episodes[eid])
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(train, "EpisodeCache", return_value=cache))
        stack.enter_context(patch.object(train, "checkpoint_identity", return_value={"fixture": "immutable-base"}))
        stack.enter_context(patch.object(train, "load_frozen_hamlet", side_effect=fake_frozen_model))
        stack.enter_context(patch.object(train, "episode_flow_loss", side_effect=fake_flow_loss))
        stack.enter_context(patch("gr00t.long_memory.cvom.episode_flow_loss", side_effect=fake_flow_loss))
        stack.enter_context(patch.object(train.RunLogger, "plot", return_value=None))

    def run_stage(self, stage, name, steps, *extra):
        output = self.root / name
        args = [
            "--stage", str(stage), "--cache-dir", str(self.cache_path),
            "--output-dir", str(output), "--device", "cpu", "--max-steps", str(steps),
            "--hidden-dim", "8", "--key-dim", "4", "--value-dim", "6",
            "--capacity", "3", "--min-fill", "1", "--learning-rate", "0.003",
            "--warmup-steps", "1", "--eval-steps", "2", "--val-samples", "3",
            "--log-steps", "1", "--plot-steps", "2", "--save-steps", "1",
            "--utility-every", "1", "--future-samples", "2", "--seed", "17",
            *extra,
        ]
        with redirect_stdout(io.StringIO()):
            train.main(args)
        return output / f"checkpoint-{steps:06d}"

    def assert_weights_identical(self, first, second):
        a = load_file(str(first / "model.safetensors"))
        b = load_file(str(second / "model.safetensors"))
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0, msg=key)

    def test_stage_one_resume_is_exact_and_validation_is_paired(self):
        uninterrupted = self.run_stage(1, "continuous", 4)
        initial = self.run_stage(1, "resumed", 2)
        resumed = self.run_stage(1, "resumed", 4, "--resume", str(initial))
        self.assert_weights_identical(uninterrupted, resumed)
        rows = [json.loads(line) for line in (resumed.parent / "metrics.jsonl").read_text().splitlines()]
        val_zero = next(row for row in rows if row["split"] == "val" and row["step"] == 0)
        self.assertEqual(val_zero["action_loss"], val_zero["baseline_action_loss"])
        self.assertEqual(val_zero["action_loss"], val_zero["wrong_memory_action_loss"])
        self.assertEqual([row["step"] for row in rows if row["split"] == "train"], [1, 2, 3, 4])

    def test_stage_two_keeps_action_learning_and_resumes_exactly(self):
        teacher = self.run_stage(1, "reader", 2)
        continuous = self.run_stage(2, "joint-continuous", 4, "--init-checkpoint", str(teacher))
        initial = self.run_stage(2, "joint-resumed", 2, "--init-checkpoint", str(teacher))
        resumed = self.run_stage(2, "joint-resumed", 4, "--resume", str(initial))
        self.assert_weights_identical(continuous, resumed)
        old = load_file(str(teacher / "model.safetensors"))
        new = load_file(str(resumed / "model.safetensors"))
        for prefix in ("fusion.", "query.", "event_encoder.", "utility_head.", "write_head."):
            self.assertTrue(any(not torch.equal(new[key], old[key]) for key in old if key.startswith(prefix)), prefix)
        rows = [json.loads(line) for line in (resumed.parent / "metrics.jsonl").read_text().splitlines()]
        train_rows = [row for row in rows if row["split"] == "train"]
        self.assertTrue(all("action_loss" in row and "utility_loss" in row and "write_loss" in row for row in train_rows))
        manifest = json.loads((resumed / "checkpoint.json").read_text())
        self.assertEqual(manifest["metadata"]["teacher_checkpoint"], str(teacher))
        self.assertTrue((resumed.parent / "cvom_labels" / "manifest.json").is_file())

    def test_invalid_stage_change_and_exact_resume_option_are_rejected(self):
        checkpoint = self.run_stage(1, "reader", 2)
        with self.assertRaisesRegex(ValueError, "cannot change stages"):
            self.run_stage(2, "invalid-stage", 4, "--resume", str(checkpoint))
        with self.assertRaisesRegex(ValueError, "Exact resume option changed: grad_accum"):
            self.run_stage(1, "reader", 4, "--resume", str(checkpoint), "--grad-accum", "2")

    def test_resume_fork_preserves_newer_original_logs_and_exact_weights(self):
        uninterrupted = self.run_stage(1, "continuous", 4)
        initial = self.run_stage(1, "interrupted", 2)
        train.RunLogger(initial.parent).log(3, "train", {"action_loss": 123.0})
        original_logs = (initial.parent / "metrics.jsonl").read_bytes()
        original_config = (initial.parent / "run_config.json").read_bytes()
        original_provenance = (initial.parent / "provenance.json").read_bytes()
        with self.assertRaises(ValueError):
            self.run_stage(1, "interrupted", 4, "--resume", str(initial))
        # Failed preflight must not mutate the original experiment's history.
        self.assertEqual((initial.parent / "run_config.json").read_bytes(), original_config)
        self.assertEqual((initial.parent / "provenance.json").read_bytes(), original_provenance)
        resumed = self.run_stage(1, "fork", 4, "--resume", str(initial))
        self.assert_weights_identical(uninterrupted, resumed)
        self.assertEqual((initial.parent / "metrics.jsonl").read_bytes(), original_logs)
        self.assertEqual((initial.parent / "run_config.json").read_bytes(), original_config)
        self.assertTrue((initial.parent / "checkpoint-000000").is_dir())

    def test_stage_two_resume_accepts_matching_original_init_flag(self):
        teacher = self.run_stage(1, "reader", 2)
        uninterrupted = self.run_stage(2, "continuous", 4, "--init-checkpoint", str(teacher))
        initial = self.run_stage(2, "resumed", 2, "--init-checkpoint", str(teacher))
        resumed = self.run_stage(2, "resumed", 4, "--resume", str(initial), "--init-checkpoint", str(teacher))
        self.assert_weights_identical(uninterrupted, resumed)

    def test_changed_teacher_hash_is_rejected_even_without_existing_label_cache(self):
        teacher = self.run_stage(1, "reader", 2)
        initial = self.run_stage(2, "joint", 2, "--init-checkpoint", str(teacher))
        teacher_weights = teacher / "model.safetensors"
        changed = load_file(str(teacher_weights))
        first = next(iter(changed))
        changed[first] = changed[first] + 0.125
        save_file(changed, str(teacher_weights))
        # A fresh output has no CVoM manifest, so only checkpoint provenance can
        # detect this mutation. Do not rely on an existing label cache to fail.
        with patch.object(train, "CVoMLabels", side_effect=AssertionError("Teacher check was too late")):
            with self.assertRaisesRegex(ValueError, "[Tt]eacher"):
                self.run_stage(2, "fork-with-changed-teacher", 4, "--resume", str(initial))
        self.assertFalse((self.root / "fork-with-changed-teacher" / "run_config.json").exists())

    def test_cache_base_provenance_checks_hash_and_shard_stat(self):
        hamlet.validate_cache_checkpoint(self.manifest)
        config_path = self.base / "config.json"
        original = config_path.read_bytes()
        stat = config_path.stat()
        config_path.write_bytes(original.replace(b"2", b"3"))
        os.utime(config_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaises(ValueError):
            hamlet.validate_cache_checkpoint(self.manifest)
        config_path.write_bytes(original)
        os.utime(config_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        hamlet.validate_cache_checkpoint(self.manifest)
        shard = self.base / "model.safetensors"
        shard.write_bytes(shard.read_bytes() + b"changed")
        with self.assertRaises(ValueError):
            hamlet.validate_cache_checkpoint(self.manifest)

    def test_initial_memory_snapshot_is_saved_as_best_even_before_updates(self):
        checkpoint = self.run_stage(1, "initial-snapshot", 2)
        initial = checkpoint.parent / "checkpoint-000000"
        self.assertTrue((initial / "model.safetensors").is_file())
        best = json.loads((checkpoint.parent / "best_checkpoint.json").read_text())
        self.assertTrue((checkpoint.parent / best["path"] / "model.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
