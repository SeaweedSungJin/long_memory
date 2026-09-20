"""CPU-only V10 publication, portable inference and transactional-load contracts."""
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
from torch import nn
from safetensors.torch import load_file, save_file

from gr00t.long_memory.checkpoint_v4 import _state_sha256, file_sha256
from gr00t.long_memory.checkpoint_v7 import v7_checkpoint_info
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import LoRAConfig, expert_parameters, install_expert_lora
from gr00t.long_memory.hamlet import checkpoint_identity
from gr00t.long_memory.monitoring import _rng_state
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from run_scripts.robomme import checkpoint_projector_v10 as checkpoint
from run_scripts.robomme.projector_adapter_v10 import (
    install_projector, projector_parameters, set_trainable,
)
from tests.test_long_memory_v4_expert import FakeHead


class ToyHead(FakeHead):
    """Private fixture: no shared FakeHead/source is changed."""
    def __init__(self):
        super().__init__()
        self.model.proj_out_2 = nn.Linear(8, 5)


class ProjectorCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(109)
        random.seed(37)
        np.random.seed(41)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.head = ToyHead().eval().requires_grad_(False)
        original = {"action_head." + n: p for n, p in self.head.state_dict().items()}
        save_file(original, str(self.base / "model.safetensors"))
        (self.base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            n: "model.safetensors" for n in original}}))
        (self.base / "config.json").write_text(json.dumps({"hamlet_mode": "finetune", "memory_type": "moment_token",
            "mem_cond_type": "cross_attn", "n_moment_tokens": 4, "memory_stride": 16, "backbone_embedding_dim": 8}))
        (self.base / "processor_config.json").write_text(json.dumps({"processor_kwargs": {"max_state_dim": 4}}))
        for name in ("statistics.json", "embodiment_id.json"):
            (self.base / name).write_text("{}")
        self.base_bytes = {p.name: p.read_bytes() for p in self.base.iterdir()}
        cfg = MemoryV7Config(feature_dim=8, state_dim=4, hidden_dim=8, capacity=4, num_heads=2)
        self.memory, self.cvom = RecurrentMemoryV7(cfg), CVOMV7(cfg).eval().requires_grad_(False)
        lora = LoRAConfig(rank=2, alpha=4)
        targets = install_expert_lora(self.head, lora)
        spec = install_projector(self.head, enabled=True)
        set_trainable(self.head, True)
        self.config = {"trainer_variant": checkpoint.VARIANT, "driver_variant": checkpoint.VARIANT,
            "stage": 1, "mode": "archive", "memory": asdict(cfg), "expert": asdict(lora),
            "expert_targets": targets, "projector": spec, "train": {"mode": "archive"}}
        self.meta = {"base_model": checkpoint_identity(self.base), "cache_fingerprint": "toy-cache",
                     "initial_checkpoint": {"path": "/parent/not/mounted", "step": 1250}}

    def tearDown(self):
        self.assertEqual(self.base_bytes, {p.name: p.read_bytes() for p in self.base.iterdir()})
        self.temp.cleanup()

    def optimizer(self):
        return torch.optim.AdamW([
            {"params": list(self.memory.parameters()), "kind": "memory", "lr": .002},
            {"params": list(expert_parameters(self.head)), "kind": "expert", "lr": .001},
            {"params": list(projector_parameters(self.head)), "kind": "projector", "lr": .001}], lr=.001)

    def save(self, name="run", step=1, optimizer=None, **kwargs):
        return checkpoint.save_checkpoint(self.root / name, step, self.memory, self.head, self.cvom,
            optimizer, self.config, self.meta, **kwargs)

    def update(self, optimizer):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.rand(()) + random.random() + np.random.random()
        loss = sum((p.square().sum() + p.sum()) * scale for g in optimizer.param_groups for p in g["params"])
        loss.backward()
        optimizer.step()

    def snapshot(self, optimizer=None):
        result = {"memory": _state_sha256(self.memory.state_dict()), "head": _state_sha256(self.head.state_dict()),
                  "cvom": _state_sha256(self.cvom.state_dict()), "rng": _rng_state()}
        if optimizer is not None:
            result["optimizer"] = copy.deepcopy(optimizer.state_dict())
        return result

    def assert_tree_equal(self, a, b):
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for k in a:
                self.assert_tree_equal(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def rewrite_training(self, path, callback):
        training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        callback(training)
        torch.save(training, path / "training_state.pt")
        info = json.loads((path / "checkpoint.json").read_text())
        info["metadata"]["training_state_sha256"] = file_sha256(path / "training_state.pt")
        (path / "checkpoint.json").write_text(json.dumps(info))

    def rewrite_payload(self, path, filename, callback):
        state = load_file(str(path / filename))
        callback(state)
        save_file(state, str(path / filename))
        info = json.loads((path / "checkpoint.json").read_text())
        digest = file_sha256(path / filename)
        info["metadata"]["payload_sha256"][filename] = digest
        alias = {"model.safetensors": "memory", "expert.safetensors": "expert", "cvom.safetensors": "cvom",
                 "projector.safetensors": "projector"}[filename]
        info["metadata"][alias + "_sha256"] = digest
        (path / "checkpoint.json").write_text(json.dumps(info))

    def test_header_preflight_exact_toy_shape_and_rng_neutral(self):
        before = self.snapshot()
        with patch.object(checkpoint, "load_file", side_effect=AssertionError("tensor load not allowed")):
            spec = checkpoint.base_projector_spec(self.base, enabled=False)
        self.assertEqual(spec, {**self.config["projector"], "enabled": False})
        self.assert_tree_equal(before, self.snapshot())
        self.assertEqual(list(self.root.iterdir()), [self.base])

    def test_roundtrip_all_payloads_immutable_original_bytes_and_parent_not_needed(self):
        optimizer = self.optimizer()
        self.update(optimizer)
        expected = self.snapshot()
        metadata_before, config_before = copy.deepcopy(self.meta), copy.deepcopy(self.config)
        path = self.save(optimizer=optimizer, best=True, training_state={"cursor": 4})
        self.assert_tree_equal(expected, self.snapshot())
        info = checkpoint.checkpoint_info(self.base, path)
        self.assertEqual(info["config"]["trainer_variant"], checkpoint.VARIANT)
        self.assertEqual(info["metadata"]["payload_sha256"], {n: file_sha256(path / n) for n in checkpoint.PAYLOADS})
        self.assertEqual(self.meta, metadata_before)
        self.assertEqual(self.config, config_before)
        for module in (self.memory, self.cvom):
            with torch.no_grad():
                next(module.parameters()).add_(.5)
        with torch.no_grad():
            next(expert_parameters(self.head)).add_(.5)
            next(projector_parameters(self.head)).add_(.5)
        rng = _rng_state()
        loaded = checkpoint.load_checkpoint(path, self.memory, self.head, self.cvom)
        self.assertEqual(loaded["training_state"], {})
        self.assert_tree_equal(rng, _rng_state())
        self.assert_tree_equal(expected, self.snapshot())
        self.assertEqual(json.loads((path.parent / "best_checkpoint.json").read_text()), {"path": path.name, "step": 1})
        frozen_bytes = {p.name: p.read_bytes() for p in path.iterdir()}
        with self.assertRaises(FileExistsError):
            self.save(optimizer=optimizer)
        self.assertEqual(frozen_bytes, {p.name: p.read_bytes() for p in path.iterdir()})

    def test_old_v7_loader_rejects_even_zero_control(self):
        for enabled in (True, False):
            if not enabled:
                self.head.model.proj_out_2.enabled = self.head.model.proj_out_2.configured_enabled = False
                set_trainable(self.head, False)
                self.config["projector"]["enabled"] = False
            path = self.save(str(enabled), step=0)
            with self.assertRaisesRegex(ValueError, "recurrent_memory_v7"):
                v7_checkpoint_info(self.base, path)

    def test_stage_mode_variant_and_projector_specs_fail_before_publication(self):
        changes = [{"stage": 2}, {"stage": True}, {"mode": "recurrent"}, {"trainer_variant": "recurrent_memory_v7"},
                   {"projector": {**self.config["projector"], "target": "model.proj_out_1"}},
                   {"projector": {**self.config["projector"], "enabled": 1}},
                   {"projector": {**self.config["projector"], "in_features": 9}},
                   {"projector": {**self.config["projector"], "extra": 1}}]
        original = copy.deepcopy(self.config)
        for i, changeset in enumerate(changes):
            self.config = {**original, **changeset}
            with self.subTest(changes=changeset), self.assertRaises(ValueError):
                self.save("invalid" + str(i))
            self.assertFalse((self.root / ("invalid" + str(i))).exists())
        self.config = original

    def test_step_zero_and_disabled_control_cannot_hide_nonzero_delta(self):
        with torch.no_grad():
            self.head.model.proj_out_2.delta_bias.fill_(.1)
        with self.assertRaisesRegex(ValueError, "must be zero"):
            self.save(step=0)
        with torch.no_grad():
            self.head.model.proj_out_2.delta_bias.zero_()
        self.head.model.proj_out_2.enabled = self.head.model.proj_out_2.configured_enabled = False
        set_trainable(self.head, False)
        self.config["projector"]["enabled"] = False
        with torch.no_grad():
            self.head.model.proj_out_2.delta_bias.fill_(.1)
        with self.assertRaisesRegex(ValueError, "must.*zero"):
            self.save(step=1)

    def test_every_payload_and_training_hash_checked_even_for_inference(self):
        for filename in (*checkpoint.PAYLOADS, "training_state.pt"):
            path = self.save(filename)
            with (path / filename).open("ab") as handle:
                handle.write(b"changed")
            before = self.snapshot()
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "changed"):
                checkpoint.load_checkpoint(path, self.memory, self.head, self.cvom)
            self.assert_tree_equal(before, self.snapshot())

    def test_projector_nan_shape_dtype_names_rejected_even_with_updated_hash(self):
        modifications = [lambda s: s["delta_bias"].fill_(float("nan")),
                         lambda s: s.update(delta_weight=torch.zeros(5, 7)),
                         lambda s: s.update(delta_bias=s["delta_bias"].double()),
                         lambda s: s.update(unexpected=torch.zeros(1))]
        for i, modification in enumerate(modifications):
            path = self.save(str(i))
            self.rewrite_payload(path, "projector.safetensors", modification)
            before = self.snapshot()
            with self.subTest(index=i), self.assertRaises(ValueError):
                checkpoint.load_checkpoint(path, self.memory, self.head, self.cvom)
            self.assert_tree_equal(before, self.snapshot())

    def test_all_other_payloads_nan_rejected_with_updated_hash(self):
        for filename in checkpoint.PAYLOADS[:-1]:
            path = self.save(filename)
            self.rewrite_payload(path, filename, lambda s: next(iter(s.values())).flatten()[0].fill_(float("nan")))
            before = self.snapshot()
            with self.subTest(file=filename), self.assertRaisesRegex(ValueError, "nonfinite"):
                checkpoint.load_checkpoint(path, self.memory, self.head, self.cvom)
            self.assert_tree_equal(before, self.snapshot())

    def test_same_shape_semantic_head_memory_and_projector_mismatch_before_mutation(self):
        path = self.save()
        wrong = RecurrentMemoryV7(type(self.memory.config)(**{**asdict(self.memory.config), "num_heads": 4}))
        before, wrong_before = self.snapshot(), _state_sha256(wrong.state_dict())
        with self.assertRaisesRegex(ValueError, "semantic config"):
            checkpoint.load_checkpoint(path, wrong, self.head, self.cvom)
        self.assertEqual(wrong_before, _state_sha256(wrong.state_dict()))
        self.assert_tree_equal(before, self.snapshot())
        self.head.model.proj_out_2.enabled = False
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "projector semantic"):
            checkpoint.load_checkpoint(path, self.memory, self.head, self.cvom)
        self.assert_tree_equal(before, self.snapshot())

    def test_optimizer_rng_exact_resume_matches_next_update(self):
        optimizer = self.optimizer()
        self.update(optimizer)
        path = self.save(optimizer=optimizer, training_state={"cursor": 7, "planned_steps": 3})
        self.update(optimizer)
        expected = self.snapshot(optimizer)
        torch.manual_seed(7)
        random.seed(8)
        np.random.seed(9)
        result = checkpoint.load_checkpoint(path, self.memory, self.head, self.cvom, optimizer)
        self.assertEqual(result["training_state"], {"cursor": 7, "planned_steps": 3})
        self.update(optimizer)
        self.assert_tree_equal(expected, self.snapshot(optimizer))

    def test_invalid_rng_optimizer_moments_and_boundary_do_not_partially_load(self):
        optimizer = self.optimizer()
        self.update(optimizer)
        def first(t):
            return next(iter(t["optimizer"]["state"].values()))
        cases = [lambda t: t["rng"].update(torch=torch.zeros(1, dtype=torch.uint8)),
                 lambda t: first(t).update(exp_avg=torch.zeros(123)),
                 lambda t: first(t)["exp_avg"].fill_(float("inf")),
                 lambda t: first(t)["exp_avg_sq"].fill_(-1),
                 lambda t: first(t).pop("step"),
                 lambda t: first(t).update(step=torch.tensor(2.)),
                 lambda t: t["optimizer"]["param_groups"][0].update(lr=float("nan")),
                 lambda t: t.update(optimizer_boundary=False),
                 lambda t: t["optimizer_param_names"][0].reverse(),
                 lambda t: t["optimizer"]["state"].update({999999: {}})]
        for i, corrupt in enumerate(cases):
            path = self.save("corrupt" + str(i), optimizer=optimizer)
            self.rewrite_training(path, corrupt)
            with torch.no_grad():
                next(self.memory.parameters()).add_(.01)
                self.head.model.proj_out_2.delta_bias.add_(.01)
            before = self.snapshot(optimizer)
            with self.subTest(index=i), self.assertRaises((ValueError, RuntimeError)):
                checkpoint.load_checkpoint(path, self.memory, self.head, self.cvom, optimizer)
            self.assert_tree_equal(before, self.snapshot(optimizer))

    def test_optimizer_order_and_unknown_parameter_rejected(self):
        optimizer = self.optimizer()
        path = self.save(optimizer=optimizer)
        optimizer.param_groups[0]["params"].reverse()
        before = self.snapshot(optimizer)
        with self.assertRaisesRegex(ValueError, "ownership/order"):
            checkpoint.load_checkpoint(path, self.memory, self.head, self.cvom, optimizer)
        self.assert_tree_equal(before, self.snapshot(optimizer))
        optimizer = torch.optim.AdamW(self.cvom.parameters())
        with self.assertRaisesRegex(ValueError, "non-actor/adapter"):
            self.save("unknown", optimizer=optimizer)

    def test_installed_non_fp32_master_parameters_rejected_before_mutation(self):
        path = self.save()
        self.memory.bfloat16()
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "must remain FP32"):
            checkpoint.load_checkpoint(path, self.memory, self.head, self.cvom)
        self.assert_tree_equal(before, self.snapshot())

    def test_cpu_inference_accepts_cuda_trained_bundle_without_restoring_cuda_rng(self):
        path = self.save()
        self.rewrite_training(path, lambda t: t["rng"].update(cuda=[torch.zeros(1024, dtype=torch.uint8)]))
        before = self.snapshot()
        with patch("torch.cuda.is_available", return_value=False):
            info = checkpoint.load_checkpoint(path, self.memory, self.head, self.cvom)
        self.assertEqual(info["config"]["trainer_variant"], checkpoint.VARIANT)
        self.assert_tree_equal(before, self.snapshot())

    def test_no_optimizer_resume_refused_without_model_or_rng_mutation(self):
        path = self.save()
        optimizer = self.optimizer()
        before = self.snapshot(optimizer)
        with self.assertRaisesRegex(ValueError, "no optimizer"):
            checkpoint.load_checkpoint(path, self.memory, self.head, self.cvom, optimizer)
        self.assert_tree_equal(before, self.snapshot(optimizer))

    def test_failed_atomic_publication_leaves_no_checkpoint_or_best_pointer(self):
        with patch.object(checkpoint, "save_file", side_effect=RuntimeError("simulated save failure")):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                self.save(best=True)
        self.assertEqual(list((self.root / "run").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
