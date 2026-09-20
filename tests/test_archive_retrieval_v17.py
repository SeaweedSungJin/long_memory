"""CPU contracts; real archive/LoRA/checkpoints, tiny toy AE. NOT task accuracy."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from gr00t.long_memory.replay_v7 import encode_at, replay_state
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from gr00t.long_memory.checkpoint_v7 import v7_checkpoint_info
from run_scripts.robomme import archive_retrieval_objective_v17 as objective
from run_scripts.robomme import train_archive_retrieval_v17 as trainer
from run_scripts.robomme import compare_archive_retrieval_v17 as report
from tests import test_archive_deployment_trainer_v9 as deployment
from tests import test_long_memory_v4_trainer as fixture


def target(eid=0, split="train"):
    return {"episode_id": eid, "decision": 5, "query_frame": 80, "n_demo": 48,
            "positive_frames": [16], "split": split}


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        self.memory = RecurrentMemoryV7(MemoryV7Config(feature_dim=6, state_dim=3,
            num_short_tokens=2, hidden_dim=8, num_heads=2, capacity=4)).eval()
        self.ep = fixture.episode(0)
        self.ep["frames"] = torch.arange(13) * 16
        self.ep["is_demo"] = torch.arange(13) < 3

    def test_native_attention_and_past_encoder_gradient(self):
        p = objective.archive_probabilities(self.memory, self.ep, 5, checkpoint_segment=2)
        bank = replay_state(self.memory, self.ep, 5, mode="archive", checkpoint_segment=0)
        query = encode_at(self.memory, self.ep, 5)
        _, w = self.memory.attention(self.memory.read_query_norm(query), self.memory.read_key_norm(bank),
            bank, need_weights=True, average_attn_weights=False)
        torch.testing.assert_close(p, w.mean((1, 2)).reshape(1, 5, 2).sum(-1)[0])
        torch.testing.assert_close(p.sum(), torch.tensor(1.))
        self.ep["short"].requires_grad_(True)
        loss, _ = objective.retrieval_objective(self.memory, self.ep, 5, target(), checkpoint_segment=2)
        loss.backward()
        self.assertGreater(float(self.ep["short"].grad[1].abs().sum()), 0)
        self.assertEqual(float(self.ep["short"].grad[6:].abs().sum()), 0)
        self.assertGreater(float(self.memory.short_projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.memory.attention.in_proj_weight.grad.abs().sum()), 0)

    def test_future_actions_labels_cannot_change_forward(self):
        p = objective.archive_probabilities(self.memory, self.ep, 5)
        changed = copy.deepcopy(self.ep)
        changed["short"][6:] = float("nan")
        changed["state"][6:] = float("nan")
        changed["targets"][:] = float("nan")
        changed["actions"][:] = float("nan")
        torch.testing.assert_close(p, objective.archive_probabilities(self.memory, changed, 5), rtol=0, atol=0)
        for bad in ({**target(), "positive_frames": [80]}, {**target(), "episode_id": 1},
                    {**target(), "positive_frames": [1]}):
            with self.assertRaises(ValueError):
                objective.positive_mask(self.ep, 5, bad)

    def test_zero_weight_no_auxiliary_work(self):
        with patch.object(trainer, "retrieval_objective", side_effect=AssertionError("must not run")):
            self.assertEqual(trainer.backward_retrieval(SimpleNamespace(retrieval_weight=0), None, None, None, []), {})


class TrainerTests(unittest.TestCase):
    def setUp(self):
        deployment.DeploymentTrainerTests.setUp(self)
        for ep in self.episodes.values():
            ep["is_demo"] = torch.arange(13) < 3
            ep["decision_mask"][:3] = False
        self.target_path = self.root / "targets/manifest.json"
        self.target_path.parent.mkdir()
        self.target_path.write_text("{}")
        self.targets = {"identity": {"cache_fingerprint": self.manifest["fingerprint"],
            "original_splits": self.manifest["splits"]}, "fingerprint": "toy-targets",
            "examples": [target(0), target(1), target(2, "val"), target(3, "val")]}
        self.stack.enter_context(patch.object(trainer, "load_manifest", return_value=self.targets))
        self.stack.enter_context(patch.object(trainer, "EpisodeCache", return_value=self.cache))
        self.stack.enter_context(patch.object(trainer, "validate_cache_checkpoint"))
        self.base_loader = self.stack.enter_context(patch.object(trainer, "load_frozen_hamlet", side_effect=deployment.toy_base))
        self.stack.enter_context(patch.object(trainer, "expert_episode_flow_loss", side_effect=fixture.toy_loss))
        self.stack.enter_context(patch.object(trainer, "generated_prefix_objective", side_effect=deployment.toy_generated))

    def args(self, name, weight=.01):
        return ["--cache-dir", str(self.cache.path), "--targets", str(self.target_path),
                "--init-checkpoint", str(self.initial), "--output-dir", str(self.root / name),
                "--device", "cpu", "--max-steps", "4", "--query-batch-size", "3",
                "--retrieval-weight", str(weight), "--eval-steps", "2", "--save-steps", "2",
                "--memory-learning-rate", ".001", "--expert-learning-rate", ".001"]

    def run_train(self, args):
        with redirect_stdout(io.StringIO()):
            return trainer.main(args)

    def ckpt(self, name, step):
        return self.root / name / f"checkpoint-{step:06d}"

    def test_preflight_read_only_and_unsafe_output(self):
        self.assertEqual(self.run_train(self.args("preflight") + ["--preflight-only"]), 0)
        self.base_loader.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())
        with self.assertRaises(ValueError):
            self.run_train(self.args("x") + ["--output-dir", str(self.initial / "x"), "--preflight-only"])

    def test_matched_training_real_gradients_and_portable_checkpoint(self):
        before = {p.name: trainer.file_hash(p) for p in self.initial.iterdir()}
        for name, weight in (("control", 0), ("guided", .01)):
            self.run_train(self.args(name, weight))
            info = v7_checkpoint_info(self.base, self.ckpt(name, 4), expected_stage=1)
            self.assertEqual(info["config"]["driver_variant"], trainer.DRIVER)
            self.assertEqual(info["config"]["objective"]["weak_retrieval_weight"], weight)
            self.assertFalse((self.root / name / "best_checkpoint.json").exists())
            rows = [json.loads(s) for s in (self.root / name / "metrics.jsonl").read_text().splitlines()]
            self.assertTrue(any(r.get("memory_grad_norm", 0) > 0 and r.get("expert_grad_norm", 0) > 0 for r in rows))
            self.assertEqual((self.initial / "cvom.safetensors").read_bytes(), (self.ckpt(name, 4) / "cvom.safetensors").read_bytes())
        plans = [json.loads((self.root / n / "query_plan.json").read_text()) for n in ("control", "guided")]
        self.assertEqual(plans[0], plans[1])
        self.assertEqual(before, {p.name: trainer.file_hash(p) for p in self.initial.iterdir()})
        a, b = [load_file(str(self.ckpt(n, 4) / "model.safetensors")) for n in ("control", "guided")]
        self.assertFalse(torch.equal(a["short_projection.weight"], b["short_projection.weight"]))

    def test_exact_resume_and_changed_objective_rejected(self):
        self.run_train(self.args("full"))
        self.run_train(self.args("part") + ["--stop-after-steps", "2"])
        resume = ["--resume", str(self.ckpt("part", 2)), "--output-dir", str(self.root / "resume")]
        self.run_train(resume)
        fixture.TestV4Trainer.assert_exact_resume(self, self.ckpt("full", 4), self.ckpt("resume", 4))
        with self.assertRaisesRegex(ValueError, "Exact resume option"):
            self.run_train(resume + ["--output-dir", str(self.root / "bad"), "--retrieval-weight", ".1", "--preflight-only"])

    def test_completed_report_and_incomplete_history_rejected(self):
        for name, weight in (("control", 0), ("guided", .01)):
            self.run_train(self.args(name, weight))
            checkpoint = self.ckpt(name, 4)
            root = self.root / f"eval_{name}"
            root.mkdir()
            baseline = {"base_model": self.identity, "memory_checkpoint": None, "mode": "none",
                        "write_policy": "none", "write_policy_override": "checkpoint",
                        "memory_off": False, "archive_read_off": False}
            actor = {**baseline, "memory_checkpoint": str(checkpoint), "mode": "archive", "stage": 1,
                     "step": 4, "write_policy": "append",
                     "checkpoint_sha256": trainer.file_hash(checkpoint / "checkpoint.json")}
            manifest = {"trainer_variant": "archive_read_control_v7",
                "models": {"baseline": baseline, "archive": actor,
                           "archive-off": {**actor, "memory_off": True, "archive_read_off": True}},
                "settings": {"tasks": ["BinFill"], "dataset": "val", "n_episodes": 3, "seed": 6,
                             "n_action_steps": 16, "max_episode_steps": 1300, "device": "cpu"},
                "benchmark": {"versions": {"toy": "1"}}, "base_file_sha256": {"base": "same"},
                "source_sha256": {"toy.py": "same"}, "policy_package_versions": {"toy": "1"}}
            manifest["evaluation_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            (root / "comparison_manifest.json").write_text(json.dumps(manifest))
            for role in manifest["models"]:
                task_dir = root / role / "BinFill"
                task_dir.mkdir(parents=True)
                identity = {**manifest["settings"], "evaluation_id": manifest["evaluation_id"] + ":" + role,
                    "task_id": "BinFill", "scenario_metadata_sha256": "same", "model_config_sha256": "same",
                    "memory_window": 4, "demo_sampling": "same"}
                (task_dir / "policy_manifest.json").write_text(json.dumps(identity))
                success = [1, int(name == "guided" and role == "archive"), 0]
                (task_dir / "simulation_results.csv").write_text("episode_idx,episode_seed,success\n" +
                    "".join(f"{i},{6+i},{value}\n" for i, value in enumerate(success)))
        control, guided = self.root / "eval_control", self.root / "eval_guided"
        result, text = report.build_report(control, guided)
        self.assertAlmostEqual(result["comparisons"]["action-only -> retrieval-guided"]["delta"], 1/3)
        self.assertIn("not proof", text)
        with self.assertRaisesRegex(ValueError, "all16 TEST"):
            report.build_report(control, guided, self.root / "nonexistent_history")
        (guided / "archive/BinFill/simulation_results.csv").write_text("episode_idx,episode_seed,success\n0,6,1\n")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            report.build_report(control, guided)


class ReportTests(unittest.TestCase):
    def test_settings_fail_before_numbers(self):
        a = {"manifest": {"settings": {"dataset": "test"}}}
        b = {"manifest": {"settings": {"dataset": "val"}}}
        with self.assertRaisesRegex(ValueError, "dataset"):
            report.compatible(a, b)

    def test_paired_arithmetic(self):
        def run(success):
            return {"manifest": {"settings": {"tasks": ["BinFill"]}}, "rows": {"archive": {
                "BinFill": {i: {"success": s, "episode_seed": 6 + i} for i, s in enumerate(success)}}}}
        result = report.contrast(run([0, 0, 1, 1]), "archive", run([1, 0, 0, 1]), "archive", samples=40)
        self.assertEqual((result["wins"], result["losses"], result["delta"]), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
