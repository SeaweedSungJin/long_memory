"""CPU-only external-parent, visual ownership and exact-resume regressions."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from gr00t.long_memory.checkpoint_v4 import file_sha256
from gr00t.long_memory.checkpoint_v7 import save_checkpoint_v7, v7_checkpoint_info
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora
from gr00t.long_memory.hamlet import checkpoint_identity
from gr00t.long_memory.monitoring import _rng_state
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from run_scripts.robomme import checkpoint_visual_patch_v11 as checkpoint
from run_scripts.robomme.visual_patch_memory_v11 import CAMERA_ORDER, VisualPatchConfig, VisualPatchMemoryV11
from tests.test_long_memory_v4_expert import FakeHead


class VisualCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(109)
        random.seed(31)
        np.random.seed(37)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        head = FakeHead().eval().requires_grad_(False)
        original = {"action_head." + n: p for n, p in head.state_dict().items()}
        save_file(original, str(self.base / "model.safetensors"))
        (self.base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: "model.safetensors" for n in original}}))
        (self.base / "config.json").write_text(json.dumps({"hamlet_mode": "finetune", "memory_type": "moment_token",
            "mem_cond_type": "cross_attn", "n_moment_tokens": 4, "memory_stride": 16, "backbone_embedding_dim": 8}))
        (self.base / "processor_config.json").write_text(json.dumps({"processor_kwargs": {"max_state_dim": 4}}))
        for name in ("statistics.json", "embodiment_id.json"):
            (self.base / name).write_text("{}")
        memory_config = MemoryV7Config(feature_dim=8, state_dim=4, hidden_dim=8, capacity=4, num_heads=2)
        memory, cvom = RecurrentMemoryV7(memory_config), CVOMV7(memory_config)
        lora = LoRAConfig(rank=2, alpha=4)
        targets = install_expert_lora(head, lora)
        parent_config = {"trainer_variant": "recurrent_memory_v7", "stage": 1, "mode": "archive",
                         "memory": asdict(memory_config), "expert": asdict(lora), "expert_targets": targets}
        parent_metadata = {"base_model": checkpoint_identity(self.base), "cache_fingerprint": "toy-cache"}
        self.parent = save_checkpoint_v7(self.root / "parent", 1250, memory, head, cvom, None, parent_config, parent_metadata)
        self.protected = {str(p): p.read_bytes() for root in (self.base, self.parent) for p in root.iterdir()}
        self.visual_config = VisualPatchConfig(feature_dim=8, hidden_dim=8, num_heads=2)
        self.visual = VisualPatchMemoryV11(self.visual_config)
        self.config = {"trainer_variant": checkpoint.VARIANT, "driver_variant": checkpoint.VARIANT,
                       "stage": 1, "mode": "visual_patch", "visual": asdict(self.visual_config),
                       "camera_order": list(CAMERA_ORDER), "train": {"objective": "flow"}}
        self.metadata = {"base_model": checkpoint_identity(self.base), "cache_fingerprint": "toy-cache",
                         "frozen_parent": checkpoint.parent_reference(self.base, self.parent),
                         "source_sha256": {"fixture.py": "a" * 64}, "runtime": {"torch": torch.__version__}}

    def tearDown(self):
        self.assertEqual(self.protected, {name: Path(name).read_bytes() for name in self.protected})
        self.temp.cleanup()

    def extra(self, step):
        return {"driver_variant": checkpoint.VARIANT, "window_cursor": step, "plan_sha256": "b" * 64}

    def optimizer(self, visual=None):
        visual = self.visual if visual is None else visual
        decay = [p for p in visual.parameters() if p.ndim > 1]
        other = [p for p in visual.parameters() if p.ndim <= 1]
        return torch.optim.AdamW([{"params": decay, "kind": "visual", "weight_decay": .01},
                                 {"params": other, "kind": "visual", "weight_decay": 0.}], lr=.001)

    def save(self, name="run", step=0, optimizer=None, **kwargs):
        return checkpoint.save_checkpoint(self.root / name, step, self.visual, optimizer, self.config,
                                          self.metadata, training_state=self.extra(step), **kwargs)

    def assert_tree(self, first, second):
        if torch.is_tensor(first):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_tree(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_tree(a, b)
        else:
            self.assertEqual(first, second)

    def update(self, visual, optimizer):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.rand(()) + random.random() + np.random.random()
        loss = sum((p.square().sum() + p.sum()) * scale for p in visual.parameters())
        loss.backward()
        optimizer.step()

    def snapshot(self, optimizer=None):
        return copy.deepcopy({"visual": self.visual.state_dict(), "rng": _rng_state(),
                              "optimizer": optimizer.state_dict() if optimizer else None})

    def rewrite_training(self, path, fn):
        value = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        fn(value)
        torch.save(value, path / "training_state.pt")
        info = json.loads((path / "checkpoint.json").read_text())
        info["metadata"]["training_state_sha256"] = file_sha256(path / "training_state.pt")
        (path / "checkpoint.json").write_text(json.dumps(info))

    def test_roundtrip_external_parent_immutable_and_readonly_rng(self):
        optimizer = self.optimizer()
        before = self.snapshot(optimizer)
        path = self.save(optimizer=optimizer, best=True)
        self.assert_tree(before, self.snapshot(optimizer))
        info = checkpoint.checkpoint_info(self.base, path)
        self.assertFalse(info["self_contained"])
        self.assertEqual(info["training_state"], self.extra(0))
        self.assertEqual(set(p.name for p in path.iterdir()), {"visual.safetensors", "training_state.pt", "checkpoint.json"})
        with torch.no_grad():
            next(self.visual.parameters()).add_(1)
        rng = _rng_state()
        checkpoint.load_checkpoint(path, self.visual)
        self.assert_tree(rng, _rng_state())
        self.assert_tree(before, self.snapshot(optimizer))
        with self.assertRaises(FileExistsError):
            self.save(optimizer=optimizer)
        self.assertEqual(json.loads((path.parent / "best_checkpoint.json").read_text())["step"], 0)
        with self.assertRaisesRegex(ValueError, "recurrent_memory_v7"):
            v7_checkpoint_info(self.base, path)

    def test_exact_adamw_rng_resume_and_group_names(self):
        optimizer = self.optimizer()
        self.update(self.visual, optimizer)
        path = self.save(step=1, optimizer=optimizer)
        for _ in range(2):
            self.update(self.visual, optimizer)
        expected = self.snapshot(optimizer)
        other = VisualPatchMemoryV11(self.visual_config)
        other_optimizer = self.optimizer(other)
        loaded = checkpoint.load_checkpoint(path, other, other_optimizer)
        self.assertEqual(loaded["training_state"], self.extra(1))
        for _ in range(2):
            self.update(other, other_optimizer)
        self.assert_tree(expected["visual"], other.state_dict())
        self.assert_tree(expected["optimizer"], other_optimizer.state_dict())
        self.assert_tree(expected["rng"], _rng_state())

    def test_parent_missing_or_changed_fails_before_mutation(self):
        path = self.save()
        parent_training = self.parent / "training_state.pt"
        original = parent_training.read_bytes()
        before = self.snapshot()
        try:
            parent_training.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "frozen parent changed"):
                checkpoint.load_checkpoint(path, self.visual)
            self.assert_tree(before, self.snapshot())
        finally:
            parent_training.write_bytes(original)
        moved = self.parent.with_name("temporarily_moved")
        self.parent.rename(moved)
        try:
            with self.assertRaises(FileNotFoundError):
                checkpoint.load_checkpoint(path, self.visual)
            self.assert_tree(before, self.snapshot())
        finally:
            moved.rename(self.parent)

    def test_metadata_config_and_zero_output_fail_before_publication(self):
        original = copy.deepcopy(self.config)
        for index, change in enumerate(({"stage": 2}, {"stage": True}, {"mode": "archive"},
                {"trainer_variant": "recurrent_memory_v7"}, {"camera_order": list(reversed(CAMERA_ORDER))},
                {"visual": {**original["visual"], "feature_dim": 16}},
                {"visual": {**original["visual"], "num_heads": 4}})):
            self.config = {**original, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.save("bad" + str(index))
            self.assertFalse((self.root / ("bad" + str(index))).exists())
        self.config = original
        with torch.no_grad():
            self.visual.output_projection.weight.fill_(.1)
        with self.assertRaisesRegex(ValueError, "step-zero"):
            self.save("zero")
        self.assertFalse((self.root / "zero").exists())
        with self.assertRaisesRegex(ValueError, "frozen base/parent"):
            checkpoint.save_checkpoint(self.parent, 1, self.visual, None, self.config, self.metadata, training_state=self.extra(1))

    def test_shape_dtype_nonfinite_tampering_is_atomic_even_rehashed(self):
        modifications = [lambda s: s["image_projection.weight"].fill_(float("nan")),
                         lambda s: s.update({"image_projection.weight": torch.zeros(7, 8)}),
                         lambda s: s.update({"image_projection.weight": s["image_projection.weight"].double()}),
                         lambda s: s.update({"surprise": torch.zeros(1)})]
        for index, fn in enumerate(modifications):
            path = self.save(str(index))
            values = load_file(str(path / "visual.safetensors"))
            fn(values)
            save_file(values, str(path / "visual.safetensors"))
            info = json.loads((path / "checkpoint.json").read_text())
            info["metadata"]["payload_sha256"]["visual.safetensors"] = file_sha256(path / "visual.safetensors")
            (path / "checkpoint.json").write_text(json.dumps(info))
            before = self.snapshot()
            with self.assertRaises(ValueError):
                checkpoint.load_checkpoint(path, self.visual)
            self.assert_tree(before, self.snapshot())

    def test_optimizer_must_own_all_and_only_visual_parameters(self):
        outside = torch.nn.Parameter(torch.zeros(1))
        for params in ([next(self.visual.parameters())], [*self.visual.parameters(), outside]):
            optimizer = torch.optim.AdamW([{"params": params, "kind": "visual"}], lr=.001)
            with self.assertRaisesRegex(ValueError, "optimizer"):
                self.save(optimizer=optimizer)
        self.assertFalse((self.root / "run").exists())

    def test_optimizer_rng_and_cursor_tampering_does_not_mutate_live_state(self):
        optimizer = self.optimizer()
        self.update(self.visual, optimizer)
        callbacks = [lambda t: t["extra"].update(window_cursor=0),
                     lambda t: t["extra"].update(plan_sha256="bad"),
                     lambda t: t["rng"].update(torch=torch.zeros(1, dtype=torch.uint8)),
                     lambda t: t["optimizer"]["param_groups"][0].update(lr=float("nan")),
                     lambda t: t["optimizer"]["state"][0].update(exp_avg=torch.zeros(1)),
                     lambda t: t["optimizer_param_names"][0].reverse()]
        for index, fn in enumerate(callbacks):
            path = self.save(str(index), step=1, optimizer=optimizer)
            self.rewrite_training(path, fn)
            before = self.snapshot(optimizer)
            with self.subTest(index=index), self.assertRaises((ValueError, RuntimeError)):
                checkpoint.load_checkpoint(path, self.visual, optimizer)
            self.assert_tree(before, self.snapshot(optimizer))

    def test_save_failure_removes_only_own_temp_preserves_prior_best(self):
        path = self.save(best=True)
        prior = {p.name: p.read_bytes() for p in path.iterdir()}
        pointer = (path.parent / "best_checkpoint.json").read_bytes()
        with patch.object(checkpoint, "save_file", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.save(step=1, best=True)
        self.assertEqual(prior, {p.name: p.read_bytes() for p in path.iterdir()})
        self.assertEqual(pointer, (path.parent / "best_checkpoint.json").read_bytes())
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["best_checkpoint.json", "checkpoint-000000"])


if __name__ == "__main__":
    unittest.main()
