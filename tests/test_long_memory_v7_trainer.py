"""CPU orchestration tests: real recurrent/critic/LoRA/checkpoints, toy Expert.

The fixture has complete short-token episodes. Expensive original Expert
forward is replaced, but full-prefix graphs, coverage, optimizer ownership,
paired randomness, teacher rounds, checkpoint safety and resume are real.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from safetensors.torch import load_file
import torch

from gr00t.long_memory import train_v7 as trainer
from gr00t.long_memory import objectives_v7
from tests import test_long_memory_v4_trainer as fixture


class V7TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        fixture.TestV4Trainer.setUp(self)
        for ep in self.episodes.values():
            ep["is_demo"] = torch.zeros(len(ep["frames"]), dtype=torch.bool)
        self.stack.enter_context(patch.object(trainer, "EpisodeCache", return_value=self.cache))
        self.stack.enter_context(patch.object(trainer, "validate_cache_checkpoint", return_value=None))
        self.base_mock_v7 = self.stack.enter_context(patch.object(trainer, "load_frozen_hamlet", side_effect=fixture.toy_base))
        self.stack.enter_context(patch.object(trainer, "expert_episode_flow_loss", side_effect=fixture.toy_loss))
        self.stack.enter_context(patch.object(objectives_v7, "_flow", side_effect=fixture.toy_loss))

    def args(self, name, mode="recurrent", stage=1, initial=None):
        return ["--cache-dir", str(self.root / "cache"), "--output-dir", str(self.root / name),
            "--stage", str(stage), "--mode", mode, "--device", "cpu", "--max-epochs", "2",
            "--capacity", "4", "--hidden-dim", "8", "--num-heads", "2", "--lora-rank", "2", "--lora-alpha", "4",
            "--query-batch-size", "5", "--queries-per-prefix", "2", "--checkpoint-segment", "3",
            "--memory-learning-rate", ".001", "--expert-learning-rate", ".001", "--warmup-fraction", "0",
            "--cvom-learning-rate", ".001", "--cvom-rounds", "2", "--cvom-steps-per-round", "2",
            "--cvom-batch-size", "1", "--cvom-contexts", "2", "--cvom-val-contexts", "2",
            "--future-samples", "2", "--noise-samples", "2", "--val-samples", "2", "--val-noise-samples", "1",
            "--epoch-val-samples", "0", "--eval-steps", "2", "--save-steps", "1", "--log-steps", "1", "--plot-steps", "2",
            *(["--init-checkpoint", str(initial)] if initial else [])]

    def run_train(self, args):
        with redirect_stdout(io.StringIO()):
            return trainer.main(args)

    def ckpt(self, name, step):
        return self.root / name / f"checkpoint-{step:06d}"

    def rows(self, name):
        return [json.loads(line) for line in (self.root / name / "metrics.jsonl").read_text().splitlines()]

    def assert_exact(self, left, right):
        for filename in ("model.safetensors", "expert.safetensors", "cvom.safetensors"):
            a, b = load_file(str(left / filename)), load_file(str(right / filename))
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
        a = torch.load(left / "training_state.pt", weights_only=True, map_location="cpu")
        b = torch.load(right / "training_state.pt", weights_only=True, map_location="cpu")
        self.assertEqual(a["extra"], b["extra"])
        self.assertEqual(a["rng"]["python"], b["rng"]["python"])
        torch.testing.assert_close(a["rng"]["torch"], b["rng"]["torch"], rtol=0, atol=0)
        self.assertEqual(a["optimizer"]["param_groups"], b["optimizer"]["param_groups"])
        for key, state in a["optimizer"]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, b["optimizer"]["state"][key][name], rtol=0, atol=0)

    def test_coverage_includes_first_and_tail_no_duplicates(self):
        source = [[0, [0, 2, 4]], [1, [1, 3, 8, 9]]]
        windows = trainer.coverage_windows(source, 2, 2, 5, 42)
        self.assertEqual([w["query_count"] for w in windows], [5, 2, 5, 2])
        expected = sorted((eid, q) for eid, qs in source for q in qs)
        for epoch in (0, 1):
            got = sorted((eid, q) for w in windows if w["epoch"] == epoch for eid, qs in w["groups"] for q in qs)
            self.assertEqual(got, expected)
            self.assertEqual(sum(w["epoch_end"] for w in windows if w["epoch"] == epoch), 1)
        self.assertEqual(windows, trainer.coverage_windows(source, 2, 2, 5, 42))

    def test_preflight_readonly_and_output_protection(self):
        self.assertEqual(self.run_train(self.args("preflight") + ["--preflight-only"]), 0)
        self.base_mock_v7.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())
        with self.assertRaisesRegex(ValueError, "inside a source"):
            self.run_train(self.args("bad") + ["--output-dir", str(self.root / "cache" / "bad"), "--preflight-only"])

    def test_stage1_joint_gradient_and_original_baseline(self):
        self.run_train(self.args("reader") + ["--max-steps", "3", "--val-memory-off"])
        train = [r for r in self.rows("reader") if r["split"] == "train"]
        self.assertTrue(any(r["expert_grad_norm"] > 0 for r in train))
        self.assertTrue(any(r["write_ffn_grad_norm"] > 0 and r["update_gate_grad_norm"] > 0 for r in train[1:]))
        self.assertTrue(all(r["cvom_grad_norm"] == 0 for r in train))
        vals = [r for r in self.rows("reader") if r["split"] == "val"]
        self.assertEqual(len(set(r["baseline_action_loss"] for r in vals)), 1)
        self.assertEqual(vals[0]["action_loss"], vals[0]["baseline_action_loss"])
        self.assertTrue(any(r["conditioning_changed_token_fraction"] > 0 for r in train[1:]))
        from gr00t.long_memory.live_monitor import _validate_record
        for row in self.rows("reader"):
            _validate_record(row)

    def test_pause_resume_preserves_planned_cosine_and_coverage(self):
        self.run_train(self.args("full") + ["--max-steps", "4"])
        self.run_train(self.args("part") + ["--max-steps", "4", "--stop-after-steps", "2"])
        self.run_train(["--resume", str(self.ckpt("part", 2)), "--output-dir", str(self.root / "resumed")])
        self.assert_exact(self.ckpt("full", 4), self.ckpt("resumed", 4))
        with self.assertRaisesRegex(ValueError, "Exact resume option"):
            self.run_train(["--resume", str(self.ckpt("part", 2)), "--max-steps", "5", "--output-dir", str(self.root / "badhorizon")])

    def test_controls_match_query_noise_plan_and_ae_initialization(self):
        for mode in ("recurrent", "archive", "none"):
            self.run_train(self.args(mode, mode) + ["--max-steps", "2"])
        plans = [json.loads((self.root / mode / "query_plan.json").read_text()) for mode in ("recurrent", "archive", "none")]
        self.assertTrue(all(p["windows"] == plans[0]["windows"] for p in plans))
        experts = [load_file(str(self.ckpt(mode, 0) / "expert.safetensors")) for mode in ("recurrent", "archive", "none")]
        for key in experts[0]:
            self.assertTrue(all(torch.equal(state[key], experts[0][key]) for state in experts))
        a, b = load_file(str(self.ckpt("none", 0) / "model.safetensors")), load_file(str(self.ckpt("none", 2) / "model.safetensors"))
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        archive = [r for r in self.rows("archive") if r["split"] == "train"]
        self.assertTrue(all(r["write_ffn_grad_norm"] == 0 and r["update_gate_grad_norm"] == 0 for r in archive))

    def test_full_epoch_coverage_and_exempt_weight_decay(self):
        self.run_train(self.args("epoch") + ["--max-epochs", "1"])
        state = json.loads((self.root / "epoch/status.json").read_text())
        self.assertEqual(state["processed_queries"], 24)
        self.assertEqual(state["epoch"], 1)
        self.assertEqual(state["step"], 5)
        self.assertTrue((self.root / "epoch/epoch-001_checkpoint.json").is_file())
        cfg = trainer.MemoryV7Config(feature_dim=6, state_dim=3, num_short_tokens=2, hidden_dim=8, num_heads=2)
        memory, cvom = trainer.RecurrentMemoryV7(cfg), trainer.CVOMV7(cfg)
        head = fixture.toy_base(None, "cpu")[0].action_head
        trainer.install_expert_lora(head, trainer.LoRAConfig(2, 4))
        args = trainer.parse_args(self.args("unused"))
        groups = trainer.optimizer_groups(args, memory, head, cvom)
        exempt = {id(p) for g in groups if g["weight_decay"] == 0 for p in g["params"]}
        self.assertIn(id(memory.slot_addresses), exempt)
        self.assertIn(id(memory.short_token_ids), exempt)

    def test_stage2_audit_and_two_round_writer_only_resume(self):
        self.run_train(self.args("reader") + ["--max-steps", "2"])
        source = self.ckpt("reader", 2)
        with self.assertRaisesRegex(ValueError, "allow-stage2"):
            self.run_train(self.args("nogate", stage=2, initial=source))
        self.run_train(self.args("audit", stage=2, initial=source) + ["--stage2-opportunity-only"])
        self.assertFalse(any((self.root / "audit").glob("checkpoint-*")))
        self.assertTrue((self.root / "audit/opportunity_audit.json").is_file())
        self.run_train(self.args("critic", stage=2, initial=source) + ["--allow-stage2"])
        self.run_train(self.args("critic_part", stage=2, initial=source) + ["--allow-stage2", "--stop-after-steps", "3"])
        self.run_train(["--resume", str(self.ckpt("critic_part", 3)), "--output-dir", str(self.root / "critic_resumed")])
        self.assert_exact(self.ckpt("critic", 4), self.ckpt("critic_resumed", 4))
        for filename in ("model.safetensors", "expert.safetensors"):
            a, b = load_file(str(source / filename)), load_file(str(self.ckpt("critic", 4) / filename))
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        vals = [r for r in self.rows("critic") if r["split"] == "val-cvom"]
        self.assertTrue(vals)
        for folder in ("round-00", "round-01"):
            train = json.loads((self.root / "critic" / folder / "train_contexts.json").read_text())
            val = json.loads((self.root / "critic" / folder / "val_contexts.json").read_text())
            self.assertFalse({x["episode_id"] for x in train} & {x["episode_id"] for x in val})

    def test_invalid_query_fails_and_help_works(self):
        self.episodes[0]["targets"][0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            trainer.flow(fixture.toy_base(None, "cpu")[0].action_head, self.episodes[0], 0)
        root = Path(trainer.__file__).resolve().parents[2]
        result = subprocess.run([str(root / ".venv/bin/python"), str(root / "run_scripts/robomme/train_long_memory_v7.py"), "--help"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--stop-after-steps", result.stdout)


if __name__ == "__main__":
    unittest.main()
