"""Tiny CPU integration tests of v2 orchestration, not real-model training.

Memory modules, optimizer state, causal contexts, checkpoints, journals and
phase scheduling are real. A toy frozen expert and deterministic diagnostics
make phase-control/resume tests fast; a separate test uses real v2 labeling.
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

from gr00t.long_memory import train_stage2_v2 as trainer
from gr00t.long_memory.core import EpisodicMemory, MemoryConfig
from gr00t.long_memory.hamlet import isolated_seed
from gr00t.long_memory.monitoring import save_checkpoint


def tiny_episode(eid):
    generator = torch.Generator().manual_seed(901 + eid)
    count = 16
    return {"episode_id": eid, "frames": torch.arange(count + 1) * 16,
            "short": torch.randn(count + 1, 2, 6, generator=generator),
            "moment": torch.randn(count + 1, 2, 6, generator=generator),
            "state": torch.randn(count + 1, 3, generator=generator),
            "actions": torch.randn(count, 2, 4, generator=generator),
            "action_mask": torch.ones(count, 2, dtype=torch.bool),
            "transition_valid": torch.ones(count, dtype=torch.bool),
            "decision_mask": torch.ones(count, dtype=torch.bool),
            "targets": torch.randn(count, 3, 4, generator=generator)}


def tiny_base(path, device):
    with isolated_seed(755, "cpu"):
        head = torch.nn.Linear(6, 1).eval().requires_grad_(False)
    return SimpleNamespace(action_head=head.to(device)), None


def tiny_flow(head, episode, decision, fused_short=None, *, seed=None,
              activation_checkpointing=False):
    if seed is None:
        noise = torch.rand(())
    else:
        with isolated_seed(seed, "cpu"):
            noise = torch.rand(())
    short = episode["short"][decision:decision + 1] if fused_short is None else fused_short
    difference = head(short.mean(1)) - episode["targets"][decision].mean() - noise
    return {"loss": difference.square().mean(), "velocity_mae": difference.abs().mean()}


def healthy_heads(*args, **kwargs):
    # A deliberately controlled gate, not a claim that a tiny run learned these
    # statistics. Actual classifier metric math has separate unit tests.
    return {"utility_auc": 0.9, "write_auc": 0.9, "utility_loss": 0.1,
            "zero_utility_loss": 0.5, "confident_positive_count": 10.0,
            "confident_negative_count": 10.0, "confident_predicted_write_rate": 0.5}


def healthy_actions(*args, **kwargs):
    return {"action_loss": 0.2, "baseline_action_loss": 0.3, "all_action_loss": 0.2,
            "first_action_loss": 0.21, "fifo_action_loss": 0.22, "random_action_loss": 0.23,
            "learned_write_accepts": 2.0, "learned_write_attempts": 4.0,
            "learned_write_rate": 0.5}


class TestStage2V2Trainer(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "config.json").write_text(json.dumps({"memory_window": 4}))
        save_file({"weight": torch.ones(1)}, str(self.base / "model.safetensors"))
        self.cache_path = self.root / "cache"
        self.identity = {"fixture": "immutable-toy-base"}
        self.manifest = {"model_path": str(self.base), "fingerprint": "toy-v2-cache",
                         "feature_dim": 6, "state_dim": 3, "action_dim": 4,
                         "splits": {"train": [0, 1], "val": [2, 3]}}
        self.episodes = {eid: tiny_episode(eid) for eid in range(4)}
        cache = SimpleNamespace(manifest=self.manifest, load=lambda eid: self.episodes[eid])
        self.cfg = MemoryConfig(feature_dim=6, state_dim=3, action_dim=4, hidden_dim=8,
                                key_dim=4, value_dim=6, capacity=4, min_fill=2, residual_init=0.01)
        with isolated_seed(810, "cpu"):
            memory = EpisodicMemory(self.cfg)
        self.stage1 = save_checkpoint(self.root / "stage1", 3, memory, None,
            {"stage": 1, "memory": asdict(self.cfg)},
            {"base_model": self.identity, "cache_fingerprint": self.manifest["fingerprint"],
             "validation_plan": [[2, 12, None], [3, 13, None]]}, keep_last=None)
        self.original_hash = hashlib.sha256((self.stage1 / "model.safetensors").read_bytes()).hexdigest()
        self.prepared_versions = []
        self.real_prepare = trainer.prepare_round
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(trainer, "EpisodeCache", return_value=cache))
        self.stack.enter_context(patch.object(trainer, "checkpoint_identity", return_value=self.identity))
        self.stack.enter_context(patch.object(trainer, "validate_cache_checkpoint", return_value=None))
        self.load_base_mock = self.stack.enter_context(patch.object(trainer, "load_frozen_hamlet", side_effect=tiny_base))
        self.stack.enter_context(patch.object(trainer, "episode_flow_loss", side_effect=tiny_flow))
        self.stack.enter_context(patch("gr00t.long_memory.contextual_cvom.episode_flow_loss", side_effect=tiny_flow))
        self.stack.enter_context(patch.object(trainer, "validate_heads", side_effect=healthy_heads))
        self.stack.enter_context(patch.object(trainer, "validate_actions", side_effect=healthy_actions))
        self.prepare_mock = self.stack.enter_context(patch.object(trainer, "prepare_round", side_effect=self.quick_prepare))
        self.stack.enter_context(patch.object(trainer.RunLogger, "plot", return_value=None))

    def quick_prepare(self, memory, teacher, head, fetch, plans, config, output, version, fingerprint):
        self.prepared_versions.append((str(output), version))
        records, packs = {}, {}
        for split in ("train", "val"):
            records[split], packs[split] = [], []
            for index, context in enumerate(plans["contexts"][split]):
                truth = index % 2
                records[split].append({"utility_target": (0.2 + version * 0.01) if truth else 0.0,
                                       "write_target": truth, "write_weight": 1.0,
                                       "utility_weight": 1.0, "signed_gain": 0.001 if truth else -0.001})
                packs[split].append(trainer.feature_pack(memory, fetch(context["episode_id"]), context))
        return records, packs

    def arguments(self, name, steps, resume=None, *extra):
        return ["--cache-dir", str(self.cache_path), "--output-dir", str(self.root / name),
                *( ["--resume", str(resume)] if resume else ["--init-checkpoint", str(self.stage1)] ),
                "--device", "cpu", "--max-steps", str(steps),
                "--bootstrap-steps", "1", "--bootstrap-max-steps", "3",
                "--aux-batch-size", "4", "--grad-accum", "1",
                "--train-contexts", "8", "--val-contexts", "8", "--val-action-samples", "2",
                "--head-learning-rate", "0.001", "--reader-learning-rate", "0.0001",
                "--warmup-steps", "1", "--future-samples", "2", "--noise-samples", "2",
                "--teacher-refresh-steps", "2", "--policy-ramp-steps", "2",
                "--eval-steps", "1", "--log-steps", "1", "--plot-steps", "1", "--save-steps", "1",
                "--seed", "71", *extra]

    def run_training(self, name, steps, resume=None, *extra):
        with redirect_stdout(io.StringIO()):
            code = trainer.main(self.arguments(name, steps, resume, *extra))
        return code, self.root / name / f"checkpoint-{steps:06d}"

    def rows(self, output):
        return [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]

    def assert_same_weights_and_state(self, first, second):
        a, b = load_file(str(first / "model.safetensors")), load_file(str(second / "model.safetensors"))
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0, msg=key)
        astate = torch.load(first / "training_state.pt", map_location="cpu", weights_only=True)
        bstate = torch.load(second / "training_state.pt", map_location="cpu", weights_only=True)

        def equal(a, b):
            if isinstance(a, torch.Tensor):
                self.assertTrue(torch.equal(a, b))
            elif isinstance(a, dict):
                self.assertEqual(a.keys(), b.keys())
                for key in a:
                    equal(a[key], b[key])
            elif isinstance(a, (list, tuple)):
                self.assertEqual(len(a), len(b))
                for x, y in zip(a, b):
                    equal(x, y)
            else:
                self.assertEqual(a, b)
        equal(astate, bstate)

    def test_writer_only_bootstrap_leaves_entire_reader_bitwise_frozen(self):
        code, checkpoint = self.run_training("writer", 3, None, "--writer-only")
        self.assertEqual(code, 0)
        old = load_file(str(self.stage1 / "model.safetensors"))
        new = load_file(str(checkpoint / "model.safetensors"))
        for key in old:
            if not key.startswith(("utility_head.", "write_head.")):
                self.assertTrue(torch.equal(old[key], new[key]), key)
        for prefix in ("utility_head.", "write_head."):
            self.assertTrue(any(not torch.equal(old[k], new[k]) for k in old if k.startswith(prefix)))
        status = json.loads((checkpoint.parent / "status.json").read_text())
        self.assertEqual(status["phase"], "bootstrap")
        self.assertFalse((checkpoint.parent / "best_checkpoint.json").exists())
        self.assertEqual(self.original_hash, hashlib.sha256((self.stage1 / "model.safetensors").read_bytes()).hexdigest())

    def test_gate_transition_trains_reader_next_update_at_lower_lr(self):
        code, checkpoint = self.run_training("joint", 3)
        self.assertEqual(code, 0)
        rows = [r for r in self.rows(checkpoint.parent) if r["split"] == "train"]
        self.assertEqual([r["phase_joint"] for r in rows], [0, 1, 1])
        self.assertEqual(rows[0]["reader_learning_rate"], 0)
        self.assertAlmostEqual(rows[1]["reader_learning_rate"], 0.0001)
        self.assertLess(rows[1]["reader_learning_rate"], rows[1]["learning_rate"])
        self.assertEqual([r["hard_fraction"] for r in rows], [0, 0.5, 1])
        old = load_file(str(self.stage1 / "model.safetensors"))
        bootstrap = load_file(str(checkpoint.parent / "checkpoint-000001" / "model.safetensors"))
        new = load_file(str(checkpoint / "model.safetensors"))
        for prefix in ("event_encoder.", "query.", "fusion."):
            self.assertTrue(all(torch.equal(old[k], bootstrap[k]) for k in old if k.startswith(prefix)))
            self.assertTrue(any(not torch.equal(old[k], new[k]) for k in old if k.startswith(prefix)))
        self.assertTrue((checkpoint.parent / "best_checkpoint.json").is_file())

    def test_failed_bootstrap_and_joint_gates_stop_with_diagnostic_checkpoint(self):
        bad = {**healthy_heads(), "write_auc": 0.5}
        with patch.object(trainer, "validate_heads", return_value=bad):
            code, _ = self.run_training("failed-bootstrap", 5)
        self.assertEqual(code, 2)
        output = self.root / "failed-bootstrap"
        self.assertTrue((output / "checkpoint-000003").is_dir())
        self.assertFalse((output / "checkpoint-000004").exists())
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "bootstrap_not_ready")
        with patch.object(trainer, "validate_heads", side_effect=[healthy_heads(), healthy_heads(), bad]):
            code, _ = self.run_training("failed-joint", 5)
        self.assertEqual(code, 2)
        output = self.root / "failed-joint"
        self.assertTrue((output / "checkpoint-000002").is_dir())
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "joint_validation_not_ready")

    def test_continuous_resume_and_fork_are_bitwise_exact_through_teacher_refresh(self):
        code, continuous = self.run_training("continuous", 5)
        self.assertEqual(code, 0)
        _, initial = self.run_training("resumed", 3)
        _, resumed = self.run_training("resumed", 5, initial)
        self.assert_same_weights_and_state(continuous, resumed)
        _, initial_fork = self.run_training("source", 3)
        trainer.RunLogger(initial_fork.parent).log(4, "train", {"action_loss": 999.0})
        saved = {name: (initial_fork.parent / name).read_bytes() for name in
                 ("metrics.jsonl", "run_config.json", "provenance.json")}
        with self.assertRaisesRegex(ValueError, "Logs newer"):
            self.run_training("source", 5, initial_fork)
        _, fork = self.run_training("fork", 5, initial_fork)
        self.assert_same_weights_and_state(continuous, fork)
        for name, content in saved.items():
            self.assertEqual((initial_fork.parent / name).read_bytes(), content)

    def test_teacher_refresh_creates_distinct_immutable_snapshot_versions(self):
        _, checkpoint = self.run_training("refresh", 5)
        versions = [version for output, version in self.prepared_versions if output == str(checkpoint.parent)]
        self.assertEqual(versions, [0, 1])
        teacher0 = checkpoint.parent / "teachers" / "checkpoint-000000"
        teacher1 = checkpoint.parent / "teachers" / "checkpoint-000003"
        self.assertTrue((teacher0 / "model.safetensors").is_file())
        self.assertTrue((teacher1 / "model.safetensors").is_file())
        state = json.loads((checkpoint / "checkpoint.json").read_text())["metadata"]["v2_state"]
        self.assertEqual(state["teacher_version"], 1)
        self.assertEqual(state["teacher_step"], 3)
        self.assertEqual(state["teacher_weights_sha256"], hashlib.sha256((teacher1 / "model.safetensors").read_bytes()).hexdigest())

    def test_real_label_preparation_uses_old_only_contexts_and_version_directories(self):
        # Exercise actual CVoM flow comparisons/cache I/O on a tiny CPU expert;
        # readiness remains mocked to test orchestration, not learning efficacy.
        with patch.object(trainer, "prepare_round", side_effect=self.real_prepare):
            code, checkpoint = self.run_training("real-labels", 4, None,
                "--train-contexts", "2", "--val-contexts", "2")
        self.assertEqual(code, 0)
        directories = sorted((checkpoint.parent / "contextual_labels").glob("teacher-*"))
        self.assertEqual([p.name for p in directories], ["teacher-000000", "teacher-000001"])
        for directory in directories:
            self.assertTrue((directory / "manifest.json").is_file())
            labels = [json.loads(p.read_text()) for p in directory.rglob("*.json") if p.name != "manifest.json"]
            self.assertTrue(labels)
            for label in labels:
                self.assertTrue(all(i < label["candidate"] for i in label["bank_ids"]))
                self.assertTrue(all(d >= label["candidate"] + 5 for d in label["future_decisions"]))

    def test_provenance_preflight_only_and_changed_resume_arguments(self):
        args = self.arguments("preflight", 3, None, "--preflight-only")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(trainer.main(args), 0)
        self.assertFalse((self.root / "preflight").exists())
        self.load_base_mock.assert_not_called()
        _, checkpoint = self.run_training("arguments", 3)
        with self.assertRaisesRegex(ValueError, "Exact resume option changed: grad_accum"):
            self.run_training("changed", 5, checkpoint, "--grad-accum", "2")
        self.assertFalse((self.root / "changed").exists())
        bad = json.loads((checkpoint / "checkpoint.json").read_text())
        bad["config"]["trainer_variant"] = "old-v1"
        (checkpoint / "checkpoint.json").write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "V1 Stage 2 cannot resume"):
            self.run_training("bad-variant", 5, checkpoint)

    def test_rejects_invalid_arguments_and_reusing_finished_output(self):
        for extra in (("--noise-samples", "1"), ("--bootstrap-max-steps", "0"),
                      ("--reader-learning-rate", "nan"), ("--write-delta", "-1"),
                      ("--min-writer-auc", "0.4")):
            args = trainer.parser().parse_args(self.arguments("invalid", 3, None, *extra))
            with self.assertRaises(ValueError):
                trainer.validate_args(args)
        self.run_training("exists", 3)
        with self.assertRaisesRegex(ValueError, "existing runs are preserved"):
            self.run_training("exists", 3)

    def test_same_output_is_locked_and_exception_releases_lock(self):
        output = self.root / "locked"
        with trainer.output_lock(output):
            with self.assertRaisesRegex(RuntimeError, "Another trainer"):
                self.run_training("locked", 3)
        self.load_base_mock.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "test exception"):
            with trainer.output_lock(output):
                raise RuntimeError("test exception")
        with trainer.output_lock(output):
            pass

    def test_modified_teacher_is_detected_before_label_regeneration(self):
        _, initial = self.run_training("teacher-source", 3)
        state = json.loads((initial / "checkpoint.json").read_text())["metadata"]["v2_state"]
        teacher_file = Path(state["teacher_checkpoint"]) / "model.safetensors"
        changed = load_file(str(teacher_file))
        key = next(iter(changed))
        changed[key] = changed[key] + 0.125
        save_file(changed, str(teacher_file))
        with patch.object(trainer, "prepare_round", side_effect=AssertionError("teacher check must happen first")):
            with self.assertRaisesRegex(ValueError, "teacher snapshot changed"):
                self.run_training("teacher-fork", 5, initial)

    def test_synthetic_classifier_health_does_not_override_collapsed_actual_replay(self):
        for name, accepted in (("no-writes", 0.0), ("all-writes", 4.0)):
            with self.subTest(name=name), patch.object(trainer, "validate_actions", return_value={
                **healthy_actions(), "learned_write_accepts": accepted,
            }):
                code, checkpoint = self.run_training(name, 3)
                self.assertEqual(code, 2)
                state = json.loads((checkpoint.parent / "status.json").read_text())
                self.assertEqual(state["phase"], "bootstrap")
                self.assertTrue(any("actual hard replay" in reason for reason in state["gate_reasons"]))

    def test_fixed_stage1_guard_detects_joint_drift_even_when_current_all_also_degrades(self):
        worse = {**healthy_actions(), "action_loss": .4, "all_action_loss": .4}
        with patch.object(trainer, "validate_actions", side_effect=[healthy_actions(), healthy_actions(), worse]):
            code, checkpoint = self.run_training("fixed-stage1-drift", 2)
        self.assertEqual(code, 2)
        state = json.loads((checkpoint / "checkpoint.json").read_text())["metadata"]["v2_state"]
        self.assertEqual(state["stage1_all_action_loss"], .2)
        self.assertEqual(state["status"], "joint_validation_not_ready")
        self.assertTrue(any("fixed initial Stage-1" in reason for reason in state["gate_reasons"]))


if __name__ == "__main__":
    unittest.main()
