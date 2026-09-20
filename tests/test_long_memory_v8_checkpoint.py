"""Small CPU tests for V8 provenance, semantic config and resume safety."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

from gr00t.long_memory.checkpoint_v8 import (VARIANT, PAYLOADS, actor_state_sha256,
    file_sha256, load_checkpoint_v8, save_checkpoint_v8, v8_checkpoint_info)
from gr00t.long_memory.event_v8 import MemoryV8Config, EventMemoryV8
from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora
from gr00t.long_memory.hamlet import checkpoint_identity
from tests.test_long_memory_v4_expert import FakeHead


class V8CheckpointTests(unittest.TestCase):
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
        self.cfg = MemoryV8Config(feature_dim=8, state_dim=4, hidden_dim=8, capacity=4, num_heads=2)
        self.memory = EventMemoryV8(self.cfg)
        lora = LoRAConfig(rank=2, alpha=4)
        targets = install_expert_lora(self.head, lora)
        self.config = dict(trainer_variant=VARIANT, stage=1, mode="event", memory=asdict(self.cfg),
                           expert=asdict(lora), expert_targets=targets)
        self.meta = dict(base_model=checkpoint_identity(self.base), cache_fingerprint="v8-test-cache")

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name="run", optimizer=None, **kwargs):
        return save_checkpoint_v8(self.root / name, 1, self.memory, self.head,
                                  optimizer, self.config, self.meta, **kwargs)

    def test_roundtrip_preserves_base_rng_and_is_immutable(self):
        before = {p.name: p.read_bytes() for p in self.base.iterdir()}
        path = self.save(best=True)
        rng = torch.get_rng_state().clone()
        info = v8_checkpoint_info(self.base, path, expected_stage=1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(info["metadata"]["payload_sha256"]), set(PAYLOADS))
        self.assertFalse((path / "cvom.safetensors").exists())
        prior = actor_state_sha256(self.memory)
        with torch.no_grad():
            next(self.memory.parameters()).add_(1)
        load_checkpoint_v8(path, self.memory, self.head)
        self.assertEqual(prior, actor_state_sha256(self.memory))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.base.iterdir()})
        with self.assertRaises(FileExistsError):
            self.save()

    def test_source_and_residual_bound_semantics_checked_before_mutation(self):
        path = self.save()
        for override in ({"source": "short"}, {"residual_scale": .2}, {"num_heads": 4}):
            wrong = EventMemoryV8(MemoryV8Config(**{**asdict(self.cfg), **override}))
            before = actor_state_sha256(wrong)
            with self.assertRaisesRegex(ValueError, "semantic config"):
                load_checkpoint_v8(path, wrong, self.head)
            self.assertEqual(before, actor_state_sha256(wrong))

    def test_payload_changes_rejected(self):
        for file in PAYLOADS:
            path = self.save(file)
            values = load_file(str(path / file))
            values[next(iter(values))].flatten()[0] += .1
            save_file(values, str(path / file))
            with self.assertRaisesRegex(ValueError, "payload changed"):
                v8_checkpoint_info(self.base, path)

    def test_nonfinite_rejected_even_if_hash_updated(self):
        path = self.save()
        file = "model.safetensors"
        values = load_file(str(path / file))
        values[next(iter(values))].flatten()[0] = float("nan")
        save_file(values, str(path / file))
        info = json.loads((path / "checkpoint.json").read_text())
        info["metadata"]["payload_sha256"][file] = file_sha256(path / file)
        (path / "checkpoint.json").write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            v8_checkpoint_info(self.base, path)

    def test_no_stage2_or_previous_variant_masquerading(self):
        self.config["stage"] = 2
        with self.assertRaisesRegex(ValueError, "Stage 1"):
            self.save()
        self.config["stage"] = 1
        self.config["trainer_variant"] = "recurrent_memory_v7"
        with self.assertRaisesRegex(ValueError, "event_memory_v8"):
            self.save()

    def test_optimizer_rng_and_logical_cursor_resume(self):
        optimizer = torch.optim.AdamW(self.memory.parameters(), lr=.002)
        sum(p.square().sum() for p in self.memory.parameters()).backward()
        optimizer.step()
        path = self.save(optimizer=optimizer, training_state=dict(epoch=2, window_cursor=17, plan_sha256="abc"))
        expected = torch.randn(5)
        torch.manual_seed(3)
        info = load_checkpoint_v8(path, self.memory, self.head, optimizer)
        self.assertTrue(torch.equal(expected, torch.randn(5)))
        self.assertEqual(info["training_state"], dict(epoch=2, window_cursor=17, plan_sha256="abc"))

    def test_optimizer_shape_corruption_checked_before_model_mutation(self):
        optimizer = torch.optim.AdamW(self.memory.parameters(), lr=.002)
        sum(p.square().sum() for p in self.memory.parameters()).backward()
        optimizer.step()
        path = self.save(optimizer=optimizer)
        saved = torch.load(path / "training_state.pt", weights_only=True)
        first = next(iter(saved["optimizer"]["state"].values()))
        first["exp_avg"] = torch.ones(123)
        torch.save(saved, path / "training_state.pt")
        info = json.loads((path / "checkpoint.json").read_text())
        info["metadata"]["training_state_sha256"] = file_sha256(path / "training_state.pt")
        (path / "checkpoint.json").write_text(json.dumps(info))
        with torch.no_grad():
            next(self.memory.parameters()).add_(1)
        before = actor_state_sha256(self.memory)
        with self.assertRaisesRegex(ValueError, "optimizer tensor shape"):
            load_checkpoint_v8(path, self.memory, self.head, optimizer)
        self.assertEqual(before, actor_state_sha256(self.memory))

    def test_missing_optimizer_step_rejected_before_model_mutation(self):
        optimizer = torch.optim.AdamW(self.memory.parameters(), lr=.002)
        sum(p.square().sum() for p in self.memory.parameters()).backward()
        optimizer.step()
        path = self.save(optimizer=optimizer)
        saved = torch.load(path / "training_state.pt", weights_only=True)
        next(iter(saved["optimizer"]["state"].values())).pop("step")
        torch.save(saved, path / "training_state.pt")
        info = json.loads((path / "checkpoint.json").read_text())
        info["metadata"]["training_state_sha256"] = file_sha256(path / "training_state.pt")
        (path / "checkpoint.json").write_text(json.dumps(info))
        with torch.no_grad():
            next(self.memory.parameters()).add_(1)
        before = actor_state_sha256(self.memory)
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            load_checkpoint_v8(path, self.memory, self.head, optimizer)
        self.assertEqual(before, actor_state_sha256(self.memory))


if __name__ == "__main__":
    unittest.main()
