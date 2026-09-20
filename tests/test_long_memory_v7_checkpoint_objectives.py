"""CPU contracts for immutable V7 checkpoints and future storage labels."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from gr00t.long_memory.checkpoint_v7 import (VARIANT, PAYLOADS, actor_state_sha256,
    file_sha256, load_checkpoint_v7, save_checkpoint_v7, v7_checkpoint_info)
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import (LoRAConfig, expert_parameters, expert_state_sha256,
                                        install_expert_lora, set_expert_trainable)
from gr00t.long_memory.hamlet import checkpoint_identity
from gr00t.long_memory.objectives_v7 import (LabelV7Config, build_storage_label,
    make_storage_contexts, storage_loss, summarize_storage_labels)
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from test_long_memory_v4_expert import FakeHead


def episode(eid=1, length=10):
    short = torch.randn(length, 4, 8)
    demo = torch.arange(length) < 2
    decisions = ~demo[:-1]
    return dict(episode_id=eid, short=short, state=torch.randn(length, 4),
                frames=torch.arange(length) * 16, is_demo=demo, decision_mask=decisions,
                actions=torch.zeros(length - 1, 16, 3), targets=torch.randn(length - 1, 2, 3),
                target_mask=decisions[:, None, None].expand(-1, 2, 3),
                features=[torch.cat([torch.randn(2, 8), row]) for row in short],
                attention_masks=[torch.ones(6, dtype=torch.bool) for _ in short],
                image_masks=[torch.zeros(6, dtype=torch.bool) for _ in short], embodiment_id=0)


class V7CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.head = FakeHead().eval().requires_grad_(False)
        original = {"action_head." + key: value for key, value in self.head.state_dict().items()}
        save_file(original, str(self.base / "model.safetensors"))
        (self.base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            key: "model.safetensors" for key in original}}))
        (self.base / "config.json").write_text(json.dumps(dict(hamlet_mode="finetune", mem_cond_type="cross_attn",
            memory_type="moment_token", n_moment_tokens=4, memory_stride=16, backbone_embedding_dim=8)))
        (self.base / "processor_config.json").write_text(json.dumps({"processor_kwargs": dict(max_state_dim=4)}))
        for name in ("statistics.json", "embodiment_id.json"):
            (self.base / name).write_text("{}")
        self.cfg = MemoryV7Config(feature_dim=8, state_dim=4, hidden_dim=8, capacity=4, num_heads=2)
        self.memory, self.cvom = RecurrentMemoryV7(self.cfg), CVOMV7(self.cfg)
        lora = LoRAConfig(rank=2, alpha=4)
        targets = install_expert_lora(self.head, lora)
        self.config = dict(trainer_variant=VARIANT, stage=1, mode="recurrent", memory=asdict(self.cfg),
                           expert=asdict(lora), expert_targets=targets)
        self.meta = dict(base_model=checkpoint_identity(self.base), cache_fingerprint="v7-test-cache")

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name="run", optimizer=None, **kwargs):
        return save_checkpoint_v7(self.root / name, 1, self.memory, self.head, self.cvom,
                                  optimizer, self.config, self.meta, **kwargs)

    def test_roundtrip_is_immutable_rng_neutral_and_preserves_base(self):
        before = {p.name: p.read_bytes() for p in self.base.iterdir()}
        path = self.save(best=True)
        rng = torch.get_rng_state().clone()
        info = v7_checkpoint_info(self.base, path, expected_stage=1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(info["metadata"]["payload_sha256"], {p: file_sha256(path / p) for p in PAYLOADS})
        actor_before = actor_state_sha256(self.memory)
        with torch.no_grad():
            next(self.memory.parameters()).add_(1)
        load_checkpoint_v7(path, self.memory, self.head, self.cvom)
        self.assertEqual(actor_before, actor_state_sha256(self.memory))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.base.iterdir()})
        with self.assertRaises(FileExistsError):
            self.save()

    def test_semantic_config_mismatch_rejected_before_mutation(self):
        path = self.save()
        wrong = RecurrentMemoryV7(MemoryV7Config(**dict(asdict(self.cfg), num_heads=4)))
        old = actor_state_sha256(wrong)
        with self.assertRaisesRegex(ValueError, "semantic config"):
            load_checkpoint_v7(path, wrong, self.head, self.cvom)
        self.assertEqual(old, actor_state_sha256(wrong))
        self.config["memory"]["time_scale"] = 8
        with self.assertRaises(ValueError):
            self.save("bad-save")

    def test_all_payloads_checked_against_hashes(self):
        for file in PAYLOADS:
            path = self.save(file)
            values = load_file(str(path / file))
            values[next(iter(values))].flatten()[0] += .1
            save_file(values, str(path / file))
            with self.assertRaisesRegex(ValueError, "payload changed"):
                v7_checkpoint_info(self.base, path)

    def test_optimizer_rng_and_coverage_extra_restore(self):
        optimizer = torch.optim.AdamW(self.memory.parameters(), lr=.002)
        sum(p.square().sum() for p in self.memory.parameters()).backward()
        optimizer.step()
        path = self.save(optimizer=optimizer, training_state=dict(epoch=2, cursor=17, plan_hash="abc"))
        expected = torch.randn(5)
        torch.manual_seed(3)
        info = load_checkpoint_v7(path, self.memory, self.head, self.cvom, optimizer)
        self.assertTrue(torch.equal(expected, torch.randn(5)))
        self.assertEqual(info["training_state"], dict(epoch=2, cursor=17, plan_hash="abc"))
        state = torch.load(path / "training_state.pt", weights_only=True)
        state["extra"]["cursor"] = 99
        torch.save(state, path / "training_state.pt")
        with self.assertRaisesRegex(ValueError, "optimizer/RNG state changed"):
            load_checkpoint_v7(path, self.memory, self.head, self.cvom, optimizer)

    def test_stage2_freezes_actor_expert_not_cvom_and_parent_is_not_required(self):
        parent = self.save("parent")
        self.config["stage"] = 2
        self.meta.update(frozen_actor_sha256=actor_state_sha256(self.memory),
            frozen_expert_sha256=expert_state_sha256(self.head), stage1_parent=dict(path="/not-mounted/parent",
            checkpoint_sha256=file_sha256(parent / "checkpoint.json"),
            memory_sha256=file_sha256(parent / "model.safetensors"), expert_sha256=file_sha256(parent / "expert.safetensors")))
        with torch.no_grad():
            next(self.cvom.parameters()).add_(.2)
        path = self.save("stage2")
        self.assertEqual(v7_checkpoint_info(self.base, path)["config"]["stage"], 2)
        with torch.no_grad():
            next(self.memory.parameters()).add_(.2)
        with self.assertRaisesRegex(ValueError, "frozen Stage-1"):
            self.save("bad-stage2")

    def test_ae_control_and_archive_cannot_initialize_stage2(self):
        for mode in ("none", "archive"):
            self.config.update(mode=mode, stage=1)
            self.save(mode)
            self.config["stage"] = 2
            with self.assertRaisesRegex(ValueError, "Stage 2 requires"):
                self.save(mode + "-stage2")

    def test_malformed_rng_rejected_before_actor_mutation(self):
        optimizer = torch.optim.AdamW(self.memory.parameters())
        path = self.save(optimizer=optimizer)
        training = torch.load(path / "training_state.pt", weights_only=True)
        training["rng"]["torch"] = torch.zeros(1, dtype=torch.uint8)
        torch.save(training, path / "training_state.pt")
        info = json.loads((path / "checkpoint.json").read_text())
        info["metadata"]["training_state_sha256"] = file_sha256(path / "training_state.pt")
        (path / "checkpoint.json").write_text(json.dumps(info))
        with torch.no_grad():
            next(self.memory.parameters()).add_(1)
        before = actor_state_sha256(self.memory)
        with self.assertRaises(RuntimeError):
            load_checkpoint_v7(path, self.memory, self.head, self.cvom, optimizer)
        self.assertEqual(before, actor_state_sha256(self.memory))


class V7ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.cfg = MemoryV7Config(feature_dim=8, state_dim=4, hidden_dim=8, capacity=4, num_heads=2)
        self.memory = RecurrentMemoryV7(self.cfg).requires_grad_(False).eval()
        self.cvom = CVOMV7(self.cfg)
        self.head = FakeHead().eval().requires_grad_(False)
        install_expert_lora(self.head, LoRAConfig(rank=2, alpha=4))
        set_expert_trainable(self.head, False)
        self.ep = episode()
        self.context = dict(episode_id=1, write_index=1, future_queries=[3, 7], seed=17)
        self.label_cfg = LabelV7Config(future_samples=2, noise_samples=2)

    def label(self, **kwargs):
        return build_storage_label(self.memory, None, self.head, self.ep, self.context, self.label_cfg, **kwargs)

    def test_zero_fusion_true_zero_gain_and_paired_noise_audit(self):
        label = self.label(audit=True)
        self.assertEqual(label["raw_gain"], 0)
        self.assertEqual(label["audit_raw_gain"], 0)
        self.assertEqual(label["metrics"]["expert_forwards"], 16)
        self.assertTrue(all(not label[k].requires_grad for k in ("x", "state", "candidate", "target")))
        summary = summarize_storage_labels([label])
        self.assertEqual(summary["independent_episodes"], 1)

    def test_frozen_guard_and_strictly_future_queries(self):
        self.memory.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen memory"):
            self.label()
        self.memory.requires_grad_(False)
        self.context["future_queries"] = [1, 7]
        with self.assertRaisesRegex(ValueError, "strictly future"):
            self.label()

    def test_demo_write_is_eligible_and_future_beyond_horizon_not_read(self):
        first = self.label()
        self.ep["short"][9].fill_(float("nan"))
        second = self.label()
        self.assertTrue(torch.equal(first["target"], second["target"]))
        self.assertTrue(torch.equal(first["state"], second["state"]))

    def test_zero_target_still_trains_nonzero_prediction(self):
        label = self.label()
        with torch.no_grad():
            self.cvom.head[-1].weight.normal_(0, .2)
        result = storage_loss(self.cvom, self.memory, label, self.label_cfg)
        self.assertGreater(float(result["loss"]), 0)
        result["loss"].backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in self.cvom.parameters()))
        self.assertTrue(all(p.grad is None for p in self.memory.parameters()))

    def test_context_sampler_covers_distinct_episodes_and_never_terminal_writes(self):
        eps = {eid: episode(eid) for eid in (1, 2, 3, 4)}
        class Episodes:
            fetch = staticmethod(lambda eid: eps[eid])
        contexts = make_storage_contexts(Episodes(), [1, 2, 3], 3, 42, future_samples=4)
        self.assertEqual({c["episode_id"] for c in contexts}, {1, 2, 3})
        self.assertTrue(all(q > c["write_index"] for c in contexts for q in c["future_queries"]))
        self.assertEqual(contexts, make_storage_contexts(Episodes(), [1, 2, 3], 3, 42, future_samples=4))

    def test_future_sampler_reserves_outside_short_window_when_available(self):
        ep = episode(1, length=20)
        class Episodes:
            fetch = staticmethod(lambda eid: ep)
        contexts = make_storage_contexts(Episodes(), [1], 100, 42, future_samples=1, memory_window=4)
        for context in contexts:
            i = context["write_index"]
            if len(ep["decision_mask"]) - 1 - i >= 4:
                self.assertTrue(any(q - i >= 4 for q in context["future_queries"]))


if __name__ == "__main__":
    unittest.main()
