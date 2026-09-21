"""Tiny CPU integration tests: no policy checkpoint, simulator, or GPU loads."""
from types import SimpleNamespace
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18, RepresentationMemoryV18
from run_scripts.robomme.semantic_memory_core import AnswerSuite, semantic_replay
from run_scripts.robomme.semantic_memory_storage import StorageConfig, StorageManager
from run_scripts.robomme.semantic_memory_teacher import context_plan, label_contexts, controller_loss, replay_bank
from run_scripts.robomme.train_semantic_memory import parse_args


def tiny_episode(count=9):
    generator = torch.Generator().manual_seed(113)
    return {"frames": torch.arange(count, dtype=torch.long) * 16,
            "is_demo": torch.tensor([True, True] + [False] * (count - 2)),
            "short": torch.randn(count, 2, 8, generator=generator),
            "state": torch.randn(count, 4, generator=generator),
            "decision_mask": torch.tensor([False, False] + [True] * (count - 3))}


def tiny_core():
    return RepresentationMemoryV18(RepresentationConfigV18(feature_dim=8, state_dim=4,
        num_short_tokens=2, hidden_dim=8, num_heads=2, capacity_events=3, short_window=2)).eval()


class HookAndControlTests(unittest.TestCase):
    def test_hook_parity_no_rng_or_causal_mutation(self):
        torch.manual_seed(1)
        core, ep = tiny_core(), tiny_episode()
        before = torch.random.get_rng_state().clone()
        direct = core.replay(ep, 6)
        after_direct = torch.random.get_rng_state().clone()
        observed = semantic_replay(core, ep, 6)
        self.assertTrue(torch.equal(before, after_direct))
        self.assertTrue(torch.equal(after_direct, torch.random.get_rng_state()))
        for key in ("fused", "short", "bank", "encoded_current"):
            self.assertTrue(torch.equal(direct[key], observed[key]), key)
        self.assertTrue(observed["has_read"])
        self.assertEqual(observed["retrieved"].shape, (1, 2, 8))
        self.assertEqual(len(core.memory.attention._forward_hooks), 0)
        # An unobserved future cannot affect either the real READ or hook result.
        future = {k: v.clone() for k, v in ep.items()}
        future["short"][7:] = float("nan")
        changed = semantic_replay(core, future, 6)
        self.assertTrue(torch.equal(observed["retrieved"], changed["retrieved"]))

    def test_read_off_leaves_writes_but_no_answer_content(self):
        core, ep = tiny_core(), tiny_episode()
        on = semantic_replay(core, ep, 6)
        off = semantic_replay(core, ep, 6, read_enabled=False)
        self.assertTrue(torch.equal(on["bank"], off["bank"]))
        self.assertTrue(torch.equal(off["fused"], off["short"]))
        self.assertFalse(off["has_read"])
        self.assertTrue(torch.equal(off["retrieved"], torch.zeros_like(off["retrieved"])))

    def test_current_clock_control_gradients_are_detached(self):
        suite = AnswerSuite({"input_dim": 8, "hidden_dim": 8,
            "classification_sizes": {"direction": 2}, "regression_sizes": {}})
        query = torch.randn(1, 2, 8, requires_grad=True)
        target = {"classification": {"direction": 1}}
        (current, _), (clock, _) = suite.controls(query, 32, False, target)
        (current + clock).backward()
        self.assertIsNone(query.grad)
        self.assertTrue(any(p.grad is not None for p in suite.current.parameters()))
        self.assertTrue(any(p.grad is not None for p in suite.clock.parameters()))
        self.assertTrue(all(p.grad is None for p in suite.memory.parameters()))

    def test_clock_control_remains_informative_after_layer_norm(self):
        suite = AnswerSuite({"input_dim": 8, "hidden_dim": 8,
            "classification_sizes": {"direction": 2}, "regression_sizes": {}})
        ref = torch.zeros(1, 2, 8)
        early = suite.clock_input(ref, 16, False)
        late = suite.clock_input(ref, 800, False)
        # One positive scalar plus zeros collapses to the same vector under LN.
        # A useful clock baseline needs a multi-component causal time basis.
        a = suite.clock.trunk[0](early.mean(1))
        b = suite.clock.trunk[0](late.mean(1))
        self.assertGreater(float((a - b).abs().max()), .01)


class FakeEpisodes:
    def __init__(self, episode):
        self.episode = episode
        self.fetch_ids = []

    def fetch(self, episode_id):
        self.fetch_ids.append(episode_id)
        return self.episode


class MarkedCore(torch.nn.Module):
    """Each event's content is its own index, making causality inspectable."""
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.encoded_counts = []

    def delta_state_dict(self):
        return {"scale": self.scale.detach().clone()}

    def encode_prefix(self, episode, count):
        self.encoded_counts.append(count)
        value = torch.arange(count).float()[:, None, None].expand(-1, 1, 4) * self.scale
        return {"short": value, "stored": value, "query": value}

    def initial_bank(self):
        return torch.zeros(1, 0, 4)

    def write_fifo(self, bank, candidate):
        return torch.cat((bank, candidate), dim=1)[:, -2:]


class NullAnswers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def loss(self, retrieved, target):
        return retrieved.sum() * 0 + self.weight * 0, {}


class TargetRecorder:
    def __init__(self):
        self.requests = []

    def get(self, eid, decision):
        self.requests.append((eid, decision))
        return {"classification": {"privileged_label": 1}}


class TeacherTests(unittest.TestCase):
    def make_pack(self):
        core = MarkedCore()
        manager = StorageManager(StorageConfig(capacity_events=2, num_tokens=1, dim=4, hidden_dim=8))
        episodes = FakeEpisodes({"frames": torch.arange(7), "decision_mask": torch.ones(6, dtype=torch.bool)})
        targets = TargetRecorder()
        reads, seeds = [], []

        def read(core, short, query, bank):
            decision = int(query[0, 0, 0])
            values = bank[0, :, 0].tolist()
            self.assertTrue(all(v < decision for v in values), (decision, values))
            reads.append((decision, values))
            retrieved = bank.mean(dim=1, keepdim=True)
            return retrieved, retrieved, {}

        def flow(head, ep, decision, fused, *, seed, **kwargs):
            seeds.append((decision, seed))
            jitter = torch.rand((), generator=torch.Generator().manual_seed(seed)) * .01
            return {"loss": (fused.mean() + jitter).square()}

        with patch("gr00t.long_memory.expert_v4.expert_state_sha256", return_value="fake-expert"), \
             patch("run_scripts.robomme.semantic_memory_teacher.semantic_read", side_effect=read), \
             patch("run_scripts.robomme.semantic_memory_teacher.episode_flow_v19", side_effect=flow):
            before = torch.random.get_rng_state().clone()
            pack = label_contexts(core, object(), NullAnswers(), targets, manager, episodes,
                [{"episode_id": 3, "event": 2, "future": [3, 5], "task": "Fixture"}],
                seed=23, noise_samples=2, candidate_count=3, ambiguity_margin=1e-7)
            # NullAnswers initializes zeros without global random draws.
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        return core, manager, episodes, targets, pack, reads, seeds

    def test_single_write_counterfactual_read_before_write_and_paired_seeds(self):
        core, manager, episodes, targets, pack, reads, seeds = self.make_pack()
        self.assertEqual(core.encoded_counts, [6])
        self.assertEqual(pack["contexts"][0]["prefix_operations"], ["append", "append"])
        self.assertEqual(pack["contexts"][0]["operations"], ["keep", "replace:0", "replace:1"])
        self.assertEqual(len(seeds), 12)
        self.assertEqual(seeds[:4], seeds[4:8])
        self.assertEqual(seeds[:4], seeds[8:12])
        self.assertEqual(reads, [(3, [0., 1.]), (5, [3., 4.]),
                                (3, [1., 2.]), (5, [3., 4.]),
                                (3, [0., 2.]), (5, [3., 4.])])
        self.assertEqual(targets.requests, [(3, 3), (3, 5)] * 3)
        self.assertEqual(pack["contexts"][0]["gains"][0], 0)
        self.assertGreater(pack["informative_operations"], 0)

    def test_writer_training_replays_labelled_trace_and_detaches_encoder(self):
        core, manager, episodes, _, pack, _, _ = self.make_pack()
        row = pack["contexts"][0]
        with patch.object(manager, "choose", side_effect=AssertionError("Must replay fixed membership")):
            loss, metrics = controller_loss(core, manager, episodes, pack, [row])
        loss.backward()
        self.assertIsNone(core.scale.grad)
        self.assertGreater(metrics["writer_labeled_operations"], 0)
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in manager.parameters()))
        self.assertEqual(core.encoded_counts[-1], row["event"] + 1)

    def test_context_plan_uses_requested_split_and_future_beyond_short_window(self):
        ep = {"decision_mask": torch.tensor([False, False] + [True] * 10)}
        episodes = FakeEpisodes(ep)
        cache = SimpleNamespace(manifest={"splits": {"train": [1, 2], "val": [3]}})
        contexts = context_plan(cache, episodes, {1: "A", 2: "B", 3: "A"}, 2, 4, 31, short_window=2)
        self.assertEqual(set(episodes.fetch_ids), {1, 2})
        self.assertTrue(all(c["episode_id"] in (1, 2) and c["event"] >= 2 and
                            min(c["future"]) > c["event"] and max(c["future"]) > c["event"] + 2 for c in contexts))
        self.assertEqual({c["task"] for c in contexts}, {"A", "B"})

    def test_invalid_teacher_trace_rejected(self):
        core = MarkedCore()
        manager = StorageManager(StorageConfig(capacity_events=2, num_tokens=1, dim=4, hidden_dim=8))
        encoded = core.encode_prefix({}, 5)
        with self.assertRaisesRegex(ValueError, "causal prefix"):
            replay_bank(core, manager, encoded, 3, trace=["append"])


class ArgumentTests(unittest.TestCase):
    def base(self):
        return ["--stage", "1", "--cache-dir", "cache", "--targets-dir", "targets", "--output-dir", "new"]

    def test_preflight_parses_without_loading_paths(self):
        args = parse_args(self.base() + ["--preflight-only", "--stop-after-steps", "1"])
        self.assertTrue(args.preflight_only)
        self.assertEqual(args.stop_after_steps, 1)
        self.assertEqual(args.stage, 1)

    def test_zero_pause_must_not_silently_launch_full_epoch(self):
        with self.assertRaisesRegex(ValueError, "stop_after_steps|stop-after-steps|positive"):
            parse_args(self.base() + ["--stop-after-steps", "0"])

    def test_nan_semantic_weight_and_invalid_teacher_budget_rejected(self):
        for options in (["--answer-weight", "nan"], ["--teacher-candidates", "2"], ["--teacher-noise-samples", "0"]):
            with self.assertRaises(ValueError):
                parse_args(self.base() + options)


class TrainingRoundtripTests(unittest.TestCase):
    """Real learned tensors/optimizer/bundles; toy AE is explicitly not RoboMME."""
    def setUp(self):
        from tests import test_archive_deployment_trainer_v9 as fixture
        from run_scripts.robomme import train_semantic_memory as trainer
        from run_scripts.robomme.checkpoint_representation_v18 import save_checkpoint_v18
        from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora
        from run_scripts.robomme.training_plan_v19 import build_plan_v19
        fixture.DeploymentTrainerTests.setUp(self)
        self.trainer = trainer
        cfg = RepresentationConfigV18(feature_dim=6, state_dim=3, num_short_tokens=2,
            hidden_dim=8, num_heads=2, capacity_events=32, short_window=2, time_scale=2)
        core = RepresentationMemoryV18(cfg)
        head = fixture.toy_base(None, "cpu")[0].action_head
        expert = LoRAConfig(2, 4)
        installed = install_expert_lora(head, expert)
        self.parent = save_checkpoint_v18(self.root / "parent-v18", 0, core, head, None,
            {"representation": asdict(cfg), "expert": asdict(expert), "expert_targets": installed},
            {"base_model": self.identity, "cache_fingerprint": self.manifest["fingerprint"],
             "plan_sha256": "fixture-parent"})
        self.manifest["episodes"] = [{"episode_id": eid, "task_group": "A" if eid % 2 == 0 else "B",
            "split": "train" if eid < 2 else "val"} for eid in range(34)]
        self.target_dir = self.root / "targets"
        self.target_dir.mkdir()
        (self.target_dir / "manifest.json").write_text("{}")
        fake_targets = SimpleNamespace(
            answer_config=lambda input_dim, hidden_dim: {"input_dim": input_dim, "hidden_dim": hidden_dim,
                "classification_sizes": {"direction": 2}, "regression_sizes": {}},
            get=lambda eid, decision: {"episode_id": eid, "decision": decision,
                "classification": {"direction": decision % 2}, "regression": {}})
        self.stack.enter_context(patch.object(trainer, "EpisodeCache", return_value=self.cache))
        self.stack.enter_context(patch.object(trainer, "MappedEpisodes", return_value=SimpleNamespace(fetch=self.cache.load)))
        self.stack.enter_context(patch.object(trainer, "SemanticTargets", return_value=fake_targets))
        self.stack.enter_context(patch.object(trainer, "validate_cache_checkpoint"))
        self.stack.enter_context(patch.object(trainer, "load_frozen_hamlet", side_effect=fixture.toy_base))
        self.stack.enter_context(patch.object(trainer, "sources", return_value={"fixture": "unchanged"}))
        self.stack.enter_context(patch.object(trainer, "resolve_tasks_v19", return_value=({eid: "A" if eid % 2 == 0 else "B" for eid in range(34)}, {})))

        def plan(args, cache, episodes):
            return build_plan_v19(SimpleNamespace(**vars(args), expected_tasks=("A", "B")), cache, episodes)

        def objective(head, ep, decision, fused=None, *, tail_weight, **kwargs):
            from tests.test_long_memory_v4_trainer import toy_loss
            result = toy_loss(head, ep, decision, fused, **kwargs)
            return {**result, "original_flow_loss": result["loss"], "loss": result["loss"] * tail_weight}

        def validation(*args, **kwargs):
            row = {"action_loss": 1., "loss": 1., "generated_prefix_mse": 1.}
            return {r: dict(row) for r in ("reader", "memory-off", "baseline")}, [], {}

        self.objective = objective
        self.stack.enter_context(patch.object(trainer, "build_plan_v19", side_effect=plan))
        self.stack.enter_context(patch.object(trainer, "episode_flow_v19", side_effect=objective))
        self.stack.enter_context(patch.object(trainer, "validate_v19", side_effect=validation))

    def args(self, name):
        return ["--stage", "1", "--cache-dir", str(self.cache.path), "--targets-dir", str(self.target_dir),
            "--init-checkpoint", str(self.parent), "--output-dir", str(self.root / name), "--device", "cpu",
            "--epochs", "1", "--query-batch-size", "5", "--hidden-dim", "8", "--num-heads", "2",
            "--capacity-events", "32", "--lora-rank", "2", "--lora-alpha", "4", "--answer-hidden-dim", "8",
            "--answer-val-samples", "4", "--eval-steps", "3", "--save-steps", "3", "--plot-steps", "3",
            "--log-steps", "1", "--val-per-task", "1", "--val-noise-samples", "1"]

    def run_train(self, args):
        with redirect_stdout(io.StringIO()):
            return self.trainer.main(args)

    def test_exact_resume_preserves_actor_expert_and_all_answer_heads(self):
        self.run_train(self.args("full"))
        self.run_train(self.args("pause") + ["--stop-after-steps", "2"])
        paused = self.root / "pause/checkpoint-000002"
        self.run_train(self.args("resumed") + ["--resume", str(paused)])
        for name in ("model.safetensors", "expert.safetensors", "answers.safetensors"):
            full = load_file(str(self.root / "full/checkpoint-000005" / name))
            resumed = load_file(str(self.root / "resumed/checkpoint-000005" / name))
            self.assertEqual(full.keys(), resumed.keys())
            for key in full:
                torch.testing.assert_close(full[key], resumed[key], rtol=0, atol=0, msg=f"{name}:{key}")
        from run_scripts.robomme.semantic_memory_checkpoint import semantic_info
        info = semantic_info(self.root / "full/checkpoint-000005")
        self.assertIsNone(info["storage_config"])
        self.assertEqual(info["metadata"]["labels_at_inference"], False)
        with self.assertRaisesRegex(ValueError, "Exact resume changed"):
            self.run_train(self.args("bad-resume") + ["--resume", str(paused), "--answer-weight", ".1"])

    def test_stage2_equal_budget_fifo_accepts_null_storage_and_keeps_extras(self):
        self.run_train(self.args("s1"))
        parent = self.root / "s1/checkpoint-000005"
        args = self.args("s2-fifo") + ["--stage", "2", "--init-checkpoint", str(parent), "--storage-policy", "fifo"]
        self.run_train(args)
        from run_scripts.robomme.semantic_memory_checkpoint import semantic_info, load_manager
        info = semantic_info(self.root / "s2-fifo/checkpoint-000005")
        self.assertEqual(info["stage"], 2)
        self.assertIsNone(info["storage_config"])
        self.assertIsNone(load_manager(self.root / "s2-fifo/checkpoint-000005")[0])

    def test_stage2_cvom_teacher_journal_and_writer_exact_resume(self):
        # More than32 observations are necessary to exercise storage decisions.
        for eid, old in self.episodes.items():
            generator = torch.Generator().manual_seed(700 + eid)
            n = 44
            self.episodes[eid] = {**old, "frames": torch.arange(n + 1) * 16,
                "is_demo": torch.zeros(n + 1, dtype=torch.bool),
                "short": torch.randn(n + 1, 2, 6, generator=generator),
                "state": torch.randn(n + 1, 3, generator=generator),
                "decision_mask": torch.ones(n, dtype=torch.bool),
                "actions": torch.randn(n, 16, 4, generator=generator),
                "action_mask": torch.ones(n, 16, dtype=torch.bool),
                "targets": torch.randn(n, 16, 4, generator=generator),
                "target_mask": torch.ones(n, 16, 4, dtype=torch.bool)}
        shared = ["--query-batch-size", "22", "--storage-contexts", "2", "--val-storage-contexts", "1",
                  "--writer-bootstrap-steps", "2", "--writer-batch-size", "1", "--teacher-candidates", "3",
                  "--utility-margin", "1e-12"]
        self.run_train(self.args("cvom-s1") + shared)
        parent = self.root / "cvom-s1/checkpoint-000004"
        with patch("run_scripts.robomme.semantic_memory_teacher.episode_flow_v19", side_effect=self.objective):
            options = ["--stage", "2", "--init-checkpoint", str(parent)] + shared
            self.run_train(self.args("cvom-full") + options)
            self.run_train(self.args("cvom-pause") + options + ["--stop-after-steps", "2"])
            paused = self.root / "cvom-pause/checkpoint-000002"
            self.run_train(self.args("cvom-resumed") + options + ["--resume", str(paused)])
        for filename in ("model.safetensors", "expert.safetensors", "answers.safetensors", "storage.safetensors"):
            left = load_file(str(self.root / "cvom-full/checkpoint-000004" / filename))
            right = load_file(str(self.root / "cvom-resumed/checkpoint-000004" / filename))
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, msg=f"{filename}:{key}")
        from run_scripts.robomme.semantic_memory_checkpoint import semantic_info
        info = semantic_info(paused)
        self.assertIsNotNone(info["storage_config"])
        self.assertEqual(info["metadata"]["teacher_labels"]["step"], 0)


if __name__ == "__main__":
    unittest.main()
