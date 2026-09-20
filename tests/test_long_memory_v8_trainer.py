"""CPU integration of real V8 memory, replay, LoRA, checkpoints and optimizer.

Only the large frozen Expert and materialized cache are replaced by tiny
fixtures. These tests verify training contracts, not RoboMME performance.
"""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import random
import subprocess
import unittest
from unittest.mock import patch

from safetensors.torch import load_file
import torch

from gr00t.long_memory import train_v8 as trainer
from tests import test_long_memory_v4_trainer as fixture


class V8TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        fixture.TestV4Trainer.setUp(self)
        for ep in self.episodes.values():
            ep["is_demo"] = torch.zeros(len(ep["frames"]), dtype=torch.bool)
        self.stack.enter_context(patch.object(trainer, "EpisodeCache", return_value=self.cache))
        self.stack.enter_context(patch.object(trainer, "validate_cache_checkpoint", return_value=None))
        self.base_mock_v8 = self.stack.enter_context(patch.object(trainer, "load_frozen_hamlet", side_effect=fixture.toy_base))
        self.stack.enter_context(patch.object(trainer, "expert_episode_flow_loss", side_effect=fixture.toy_loss))

    def args(self, name, mode="event"):
        return ["--cache-dir", str(self.root / "cache"), "--output-dir", str(self.root / name),
                "--stage", "1", "--mode", mode, "--device", "cpu", "--max-epochs", "2",
                "--capacity", "4", "--hidden-dim", "8", "--num-heads", "2", "--lora-rank", "2", "--lora-alpha", "4",
                "--query-batch-size", "5", "--queries-per-prefix", "2", "--checkpoint-segment", "3",
                "--memory-learning-rate", ".001", "--expert-learning-rate", ".001", "--warmup-fraction", "0",
                "--val-samples", "2", "--val-noise-samples", "1", "--epoch-val-samples", "0",
                "--eval-steps", "2", "--save-steps", "1", "--log-steps", "1", "--plot-steps", "2"]

    def run_train(self, args):
        with redirect_stdout(io.StringIO()):
            return trainer.main(args)

    def ckpt(self, name, step):
        return self.root / name / f"checkpoint-{step:06d}"

    def rows(self, name):
        return [json.loads(row) for row in (self.root / name / "metrics.jsonl").read_text().splitlines()]

    def assert_exact(self, left, right):
        for filename in ("model.safetensors", "expert.safetensors"):
            a, b = load_file(str(left / filename)), load_file(str(right / filename))
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
        a = torch.load(left / "training_state.pt", weights_only=True, map_location="cpu")
        b = torch.load(right / "training_state.pt", weights_only=True, map_location="cpu")
        for key in ("processed_queries", "window_cursor", "optimizer_updates", "epoch", "plan_sha256", "best_action_loss"):
            self.assertEqual(a["extra"][key], b["extra"][key])
        self.assertEqual(a["rng"]["python"], b["rng"]["python"])
        torch.testing.assert_close(a["rng"]["torch"], b["rng"]["torch"], rtol=0, atol=0)
        self.assertEqual(a["optimizer"]["param_groups"], b["optimizer"]["param_groups"])
        self.assertEqual(a["optimizer"]["state"].keys(), b["optimizer"]["state"].keys())
        for key, state in a["optimizer"]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, b["optimizer"]["state"][key][name], rtol=0, atol=0)

    def test_preflight_readonly_stage2_rejected_and_output_protected(self):
        self.assertEqual(self.run_train(self.args("preflight") + ["--preflight-only"]), 0)
        self.base_mock_v8.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())
        with self.assertRaisesRegex(ValueError, "Stage 1 only"):
            self.run_train(self.args("unsupported") + ["--stage", "2"])
        with self.assertRaisesRegex(ValueError, "inside a source"):
            self.run_train(self.args("bad") + ["--output-dir", str(self.root / "cache" / "bad"), "--preflight-only"])

    def test_past_encoder_reconstruction_reader_and_expert_receive_gradients(self):
        self.run_train(self.args("reader") + ["--max-steps", "3", "--memory-dropout", "0"])
        rows = self.rows("reader")
        training = [r for r in rows if r["split"] == "train"]
        for name in ("event_encoder_grad_norm", "reconstruction_grad_norm", "expert_grad_norm"):
            self.assertTrue(any(r[name] > 0 for r in training), name)
        self.assertTrue(any(r["retrieval_grad_norm"] > 0 for r in training[1:]))
        self.assertTrue(all(r["loss"] > r["action_loss"] for r in training))
        vals = [r for r in rows if r["split"] == "val"]
        self.assertEqual(vals[0]["action_loss"], vals[0]["baseline_action_loss"])
        self.assertEqual(len({r["baseline_action_loss"] for r in vals}), 1)
        for row in vals:
            self.assertEqual(row["loss"], row["action_loss"])
            self.assertIn(trainer.AUXILIARY, row)
            self.assertGreaterEqual(row[trainer.AUXILIARY], 0)
            self.assertNotIn("weighted_storage_reconstruction_loss", row)
            self.assertIn("expert_no_memory_action_loss", row)
        self.assertFalse((self.ckpt("reader", 3) / "cvom.safetensors").exists())
        from gr00t.long_memory.live_monitor import _validate_record
        for row in rows:
            _validate_record(row)

    def test_memory_dropout_bypasses_action_read_but_preserves_auxiliary_graph(self):
        self.run_train(self.args("dropout") + ["--max-steps", "2", "--memory-dropout", "1"])
        training = [r for r in self.rows("dropout") if r["split"] == "train"]
        self.assertTrue(all(r["memory_dropout_rate"] == 1 for r in training))
        self.assertTrue(all(r["conditioning_delta_norm"] == 0 for r in training))
        self.assertTrue(all(r["event_encoder_grad_norm"] > 0 and r["reconstruction_grad_norm"] > 0 for r in training))
        self.assertTrue(all(r["fusion_grad_norm"] == 0 and r["retrieval_grad_norm"] == 0 for r in training))

    def test_dropout_is_query_local_and_rng_neutral(self):
        args = trainer.parse_args(self.args("unused"))
        before_python, before_torch = random.getstate(), torch.get_rng_state().clone()
        forward = [trainer.memory_dropped(args, 1, 0, q) for q in range(64)]
        backward = [trainer.memory_dropped(args, 1, 0, q) for q in reversed(range(64))]
        self.assertEqual(forward, list(reversed(backward)))
        self.assertTrue(any(forward) and not all(forward))
        self.assertEqual(before_python, random.getstate())
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))

    def test_pause_resume_preserves_exact_weights_optimizer_rng_and_query_count(self):
        self.run_train(self.args("full") + ["--max-steps", "4"])
        self.run_train(self.args("part") + ["--max-steps", "4", "--stop-after-steps", "2"])
        self.run_train(["--resume", str(self.ckpt("part", 2)), "--output-dir", str(self.root / "resumed")])
        self.assert_exact(self.ckpt("full", 4), self.ckpt("resumed", 4))
        with self.assertRaisesRegex(ValueError, "Exact resume option"):
            self.run_train(["--resume", str(self.ckpt("part", 2)), "--max-steps", "5", "--output-dir", str(self.root / "badcap")])

    def test_epoch_tail_query_count_and_none_control(self):
        self.run_train(self.args("epoch") + ["--max-epochs", "1", "--epoch-val-samples", "3", "--epoch-val-noise-samples", "2"])
        state = json.loads((self.root / "epoch/status.json").read_text())
        self.assertEqual((state["processed_queries"], state["epoch"], state["step"]), (24, 1, 5))
        epoch = [r for r in self.rows("epoch") if r["split"] == "val-epoch"]
        self.assertEqual((len(epoch), epoch[0]["queries"], epoch[0]["noise_draws_per_query"]), (1, 3, 2))
        self.run_train(self.args("none", "none") + ["--max-steps", "2"])
        before = load_file(str(self.ckpt("none", 0) / "model.safetensors"))
        after = load_file(str(self.ckpt("none", 2) / "model.safetensors"))
        self.assertTrue(all(torch.equal(before[k], after[k]) for k in before))
        self.assertTrue(all(r["memory_grad_norm"] == 0 for r in self.rows("none") if r["split"] == "train"))

    def test_no_false_trained_best_checkpoint_when_only_aux_improves(self):
        # Action is constant and differentiable; reconstruction alone updates.
        def constant_action(head, ep, decision, fused_short=None, **kwargs):
            anchor = next(trainer.expert_parameters(head)).sum() * 0
            return {"loss": anchor + 1, "velocity_mae": anchor + 1}
        with patch.object(trainer, "expert_episode_flow_loss", side_effect=constant_action):
            self.run_train(self.args("constant") + ["--max-steps", "2"])
        best = json.loads((self.root / "constant/best_checkpoint.json").read_text())
        self.assertTrue(best["path"].endswith("checkpoint-000000"))

    def test_auxiliary_cannot_be_missing_or_nonfinite_and_selected_targets_are_checked(self):
        args = trainer.parse_args(self.args("unused"))
        ep = self.episodes[0]
        head = fixture.toy_base(None, "cpu")[0].action_head
        fused = ep["short"][1:2]
        with self.assertRaisesRegex(ValueError, "reconstruction"):
            trainer.query_objective(args, head, ep, 1, (fused, {}), 0)
        with self.assertRaisesRegex(FloatingPointError, "reconstruction"):
            trainer.query_objective(args, head, ep, 1, (fused, {trainer.AUXILIARY: torch.tensor(float("nan"))}), 0)
        broken = copy.deepcopy(ep)
        broken["targets"][0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            trainer.flow(head, broken, 0)

    def test_source_provenance_and_cli(self):
        identity = trainer.source_identity()
        for name in ("event_v8", "replay_v8", "checkpoint_v8", "train_v8", "expert_v4", "train_v7"):
            self.assertIn(f"gr00t/long_memory/{name}.py", identity)
        self.assertNotIn("gr00t/long_memory/train_v8.pyc", identity)
        root = Path(trainer.__file__).resolve().parents[2]
        result = subprocess.run([str(root / ".venv/bin/python"), str(root / "run_scripts/robomme/train_long_memory_v8.py"), "--help"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--storage-reconstruction-weight", result.stdout)


if __name__ == "__main__":
    unittest.main()
