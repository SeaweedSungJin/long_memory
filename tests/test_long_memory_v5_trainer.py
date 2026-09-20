"""Tiny CPU v5 orchestration: real LoRA/memory/recall/continuation/checkpoints.

Only the expensive base/cache and flow objective are synthetic. Exact resume
includes teacher writer refresh, all three saved modules, optimizer and RNG.
"""
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from safetensors.torch import load_file
import torch

from gr00t.long_memory import train_v5 as trainer
from gr00t.long_memory.checkpoint_v4 import save_checkpoint_v4
from gr00t.long_memory.core_v3 import ActionValueMemory
from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora
from gr00t.long_memory.monitoring import load_checkpoint
from tests import test_long_memory_v4_trainer as v4_fixture

toy_base, toy_loss = v4_fixture.toy_base, v4_fixture.toy_loss


class FakeLabels:
    def __init__(self, manifest):
        self.manifest = {"class_names": ["left", "right", "wait"], "num_classes": 3,
                         "cache_fingerprint": manifest["fingerprint"], "fingerprint": "c" * 64}

    def get(self, eid, decision):
        return {"class_id": (decision + eid) % 3, "xy": [.2, .7], "class_valid": decision > 0,
                "xy_valid": True, "available_old": decision >= 5}


class TestV5Trainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        # Reuse only fixture construction, not the old trainer's test methods.
        v4_fixture.TestV4Trainer.setUp(self)
        self.stack.enter_context(patch.object(trainer, "EpisodeCache", return_value=self.cache))
        self.stack.enter_context(patch.object(trainer, "validate_cache_checkpoint", return_value=None))
        self.v5_base = self.stack.enter_context(patch.object(trainer, "load_frozen_hamlet", side_effect=toy_base))
        self.stack.enter_context(patch.object(trainer, "_flow", side_effect=toy_loss))
        self.fake_labels = FakeLabels(self.manifest)
        self.stack.enter_context(patch.object(trainer, "RecallLabels", return_value=self.fake_labels))
        memory = ActionValueMemory(self.cfg)
        load_checkpoint(self.initial, memory)
        head = toy_base(None, "cpu")[0].action_head
        lora = LoRAConfig(2, 4)
        targets = install_expert_lora(head, lora)
        conf = {"trainer_variant": "action_expert_v4", "stage": 1, "reader_mode": "memory",
                "train": {"reader_mode": "memory"}, "memory": asdict(self.cfg),
                "expert": asdict(lora), "expert_targets": targets}
        self.initial_v4 = save_checkpoint_v4(self.root / "v4", 2, memory, head, None, conf,
            {"base_model": self.identity, "cache_fingerprint": self.manifest["fingerprint"]})

    def args(self, name, steps=3, stage=1, initial=None):
        return ["--stage", str(stage), "--cache-dir", str(self.root / "cache"),
            "--recall-labels", str(self.root / "targets"), "--init-checkpoint", str(initial or self.initial_v4),
            "--output-dir", str(self.root / name), "--device", "cpu", "--max-steps", str(steps),
            "--grad-accum", "1", "--warmup-steps", "1", "--writer-batch-size", "1",
            "--context-refresh-steps", "2", "--storage-contexts", "2", "--val-samples", "1",
            "--val-storage-samples", "1", "--future-samples", "2", "--noise-samples", "2",
            "--audit-noise-samples", "2", "--eval-steps", "1", "--log-steps", "1", "--plot-steps", "1",
            "--save-steps", "1", "--reader-learning-rate", ".001", "--expert-learning-rate", ".001",
            "--recall-learning-rate", ".001", "--writer-learning-rate", ".001", "--label-margin", "0",
            "--no-confidence-screen", "--subgoal-weight", ".01", "--grounding-weight", ".1"]

    def run_train(self, args):
        with redirect_stdout(io.StringIO()):
            return trainer.main(args)

    def ckpt(self, name, step):
        return self.root / name / f"checkpoint-{step:06d}"

    def rows(self, name):
        return [json.loads(x) for x in (self.root / name / "metrics.jsonl").read_text().splitlines()]

    def assert_exact(self, a, b):
        for filename in ("model.safetensors", "expert.safetensors", "recall.safetensors"):
            left, right = load_file(str(a / filename)), load_file(str(b / filename))
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, msg=key)
        left = torch.load(a / "training_state.pt", weights_only=True, map_location="cpu")
        right = torch.load(b / "training_state.pt", weights_only=True, map_location="cpu")
        self.assertEqual(left["rng"]["python"], right["rng"]["python"])
        torch.testing.assert_close(left["rng"]["torch"], right["rng"]["torch"], rtol=0, atol=0)
        self.assertEqual(left["optimizer"]["param_groups"], right["optimizer"]["param_groups"])
        for key, values in left["optimizer"]["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(value, right["optimizer"]["state"][key][name], rtol=0, atol=0)

    def test_preflight_and_output_safety(self):
        self.assertEqual(self.run_train(self.args("preflight") + ["--preflight-only"]), 0)
        self.v5_base.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())
        existing = self.root / "existing"
        existing.mkdir(); (existing / "user.txt").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "NEW"):
            self.run_train(self.args("existing"))
        self.assertEqual((existing / "user.txt").read_text(), "preserve")

    def test_stage1_all_three_trainable_components_and_original_baseline(self):
        initial_hash = hashlib.sha256((self.initial_v4 / "model.safetensors").read_bytes()).hexdigest()
        self.run_train(self.args("joint", 2))
        rows = self.rows("joint")
        train = [r for r in rows if r["split"] == "train"]
        self.assertTrue(any(r["reader_grad_norm"] > 0 and r["expert_grad_norm"] > 0 and r["recall_grad_norm"] > 0 for r in train))
        self.assertTrue(all(r["writer_grad_norm"] == 0 for r in train))
        for filename in ("expert.safetensors", "recall.safetensors"):
            a, b = load_file(str(self.ckpt("joint", 0) / filename)), load_file(str(self.ckpt("joint", 2) / filename))
            self.assertTrue(any(not torch.equal(a[k], b[k]) for k in a))
        initial = load_file(str(self.initial_v4 / "model.safetensors"))
        end = load_file(str(self.ckpt("joint", 2) / "model.safetensors"))
        self.assertTrue(all(torch.equal(initial[k], end[k]) for k in initial if k.startswith("writer.")))
        baseline = [r["baseline_action_loss"] for r in rows if r["split"] == "val"]
        self.assertEqual(len(set(baseline)), 1)
        self.assertTrue(any(r["split"] == "val-available-old" for r in rows))
        self.assertTrue(any("no_old_subgoal_accuracy" in r and "recall_no_old_gap" in r for r in rows))
        self.assertEqual(initial_hash, hashlib.sha256((self.initial_v4 / "model.safetensors").read_bytes()).hexdigest())
        from gr00t.long_memory.live_monitor import _validate_record
        for row in rows:
            _validate_record(row)

    def test_stage1_exact_resume(self):
        self.run_train(self.args("full", 3))
        self.run_train(self.args("part", 2))
        self.run_train(["--resume", str(self.ckpt("part", 2)), "--max-steps", "3", "--output-dir", str(self.root / "resumed")])
        self.assert_exact(self.ckpt("full", 3), self.ckpt("resumed", 3))

    def test_stage2_writer_only_and_resume_across_teacher_refresh(self):
        self.run_train(self.args("reader", 2))
        source = self.ckpt("reader", 2)
        self.run_train(self.args("writer", 3, 2, source))
        self.run_train(self.args("writer_part", 2, 2, source))
        self.run_train(["--resume", str(self.ckpt("writer_part", 2)), "--max-steps", "3",
                        "--output-dir", str(self.root / "writer_resumed")])
        self.assert_exact(self.ckpt("writer", 3), self.ckpt("writer_resumed", 3))
        for filename in ("expert.safetensors", "recall.safetensors"):
            a, b = load_file(str(source / filename)), load_file(str(self.ckpt("writer", 3) / filename))
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        a, b = load_file(str(source / "model.safetensors")), load_file(str(self.ckpt("writer", 3) / "model.safetensors"))
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a if not k.startswith("writer.")))
        rows = [r for r in self.rows("writer") if r["split"] == "train"]
        self.assertTrue(all(r["reader_grad_norm"] == r["expert_grad_norm"] == r["recall_grad_norm"] == 0 for r in rows))
        self.assertTrue(any(r["writer_grad_norm"] > 0 for r in rows))
        self.assertTrue((self.root / "writer/labels/teacher-000000/manifest.json").is_file())
        self.assertTrue((self.root / "writer/labels/teacher-000001/manifest.json").is_file())
        labels = list((self.root / "writer/labels").rglob("storage-*.json"))
        self.assertTrue(any("future_bank_ids" in json.loads(p.read_text()) for p in labels))

    def test_action_only_control_freezes_heads_and_stage2_aux_guard(self):
        self.run_train(self.args("control", 2) + ["--subgoal-weight", "0", "--grounding-weight", "0"])
        a = load_file(str(self.ckpt("control", 0) / "recall.safetensors"))
        b = load_file(str(self.ckpt("control", 2) / "recall.safetensors"))
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        self.assertTrue(all(r.get("recall_grad_norm", 0) == 0 for r in self.rows("control")))
        with self.assertRaisesRegex(ValueError, "untrained"):
            self.run_train(self.args("badwriter", 1, 2, self.ckpt("control", 2)))
        with self.assertRaisesRegex(ValueError, "auxiliary weights zero"):
            self.run_train(self.args("badnone", 1) + ["--reader-mode", "none"])

    def test_changed_labels_and_options_cannot_resume(self):
        self.run_train(self.args("original", 1))
        resume = ["--resume", str(self.ckpt("original", 1)), "--max-steps", "2", "--output-dir", str(self.root / "bad")]
        with self.assertRaisesRegex(ValueError, "Exact resume option"):
            self.run_train(resume + ["--subgoal-weight", ".9"])
        self.fake_labels.manifest["fingerprint"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "recall labels changed"):
            self.run_train(resume)

    def test_sampling_proxy_retains_regular_queries(self):
        plans = {"train": [[0, [0]]], "available_old_train": [[0, [9]]]}
        self.assertEqual(trainer.sample_query(plans, 0), (0, 0, False))
        with self.assertRaisesRegex(ValueError, "ordinary"):
            self.run_train(self.args("badfraction") + ["--delayed-fraction", "1"])

    def test_nested_output_cannot_modify_source_directories(self):
        for source in (self.initial_v4, self.root / "cache", self.root / "targets", self.base):
            output = source / "accidental-training"
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "inside a source"):
                self.run_train(self.args("unused") + ["--output-dir", str(output), "--preflight-only"])
            self.assertFalse(output.exists())

    def test_untrained_step_zero_heads_block_writer_but_inherited_heads_do_not(self):
        self.run_train(self.args("reader", 1))
        with self.assertRaisesRegex(ValueError, "supervised_updates=0"):
            self.run_train(self.args("badzero", 1, 2, self.ckpt("reader", 0)) + ["--preflight-only"])
        self.run_train(self.args("continued", 1, 1, self.ckpt("reader", 1)))
        info = json.loads((self.ckpt("continued", 0) / "checkpoint.json").read_text())
        self.assertEqual(info["metadata"]["v5_state"]["subgoal_supervised_updates"], 1)
        self.assertEqual(info["metadata"]["v5_state"]["grounding_supervised_updates"], 1)
        self.assertEqual(self.run_train(self.args("goodzero", 1, 2, self.ckpt("continued", 0)) + ["--preflight-only"]), 0)

    def test_no_writer_signal_cannot_trigger_decay_or_momentum(self):
        self.run_train(self.args("reader", 2))
        initial = self.ckpt("reader", 2)
        training_calls = 0
        def zero_loss(memory, ep, label):
            nonlocal training_calls
            if torch.is_grad_enabled():
                training_calls += 1
                if training_calls == 1:
                    # Seed nonzero Adam momentum, then ensure later screened
                    # batches do not move it or the parameters through decay.
                    loss = sum(p.square().mean() for p in memory.writer_parameters())
                    return {"loss": loss, "metrics": {"storage_signal_fraction": 1., "storage_loss": float(loss.detach())}}
            loss = sum(p.sum() * 0 for p in memory.writer_parameters())
            return {"loss": loss, "metrics": {"storage_signal_fraction": 0., "storage_loss": 0.}}
        with patch.object(trainer, "storage_loss", side_effect=zero_loss):
            self.run_train(self.args("nosignal", 3, 2, initial))
        a, b = load_file(str(self.ckpt("nosignal", 1) / "model.safetensors")), load_file(str(self.ckpt("nosignal", 3) / "model.safetensors"))
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        rows = [r for r in self.rows("nosignal") if r["split"] == "train"]
        self.assertEqual([r["optimizer_step_applied"] for r in rows], [1., 0., 0.])
        first = torch.load(self.ckpt("nosignal", 1) / "training_state.pt", weights_only=True, map_location="cpu")
        last = torch.load(self.ckpt("nosignal", 3) / "training_state.pt", weights_only=True, map_location="cpu")
        for key, values in first["optimizer"]["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(value, last["optimizer"]["state"][key][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
