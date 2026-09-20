"""Synthetic immutable v6 bundles: architecture, resume and frozen CVOM teacher."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

from gr00t.long_memory.checkpoint_v4 import file_sha256
from gr00t.long_memory.checkpoint_v6 import (PAYLOADS, VARIANT, frozen_identity,
    load_checkpoint_v6, reader_state_sha256, save_checkpoint_v6, v6_checkpoint_info)
from gr00t.long_memory.core_v6 import VisualMemoryV6, VisualMemoryV6Config
from gr00t.long_memory.expert_v4 import (LoRAConfig, expert_parameters, expert_state_dict,
                                        install_expert_lora)
from gr00t.long_memory.expert_v6 import bridge_parameters, bridge_state_dict, install_memory_bridge
from gr00t.long_memory.hamlet import checkpoint_identity
from test_long_memory_v6_expert import Head


class V6CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(131)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.head = Head().eval().requires_grad_(False)
        original = {"action_head." + key: value for key, value in self.head.state_dict().items()}
        save_file(original, str(self.base / "model.safetensors"))
        (self.base / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {key: "model.safetensors" for key in original}}))
        (self.base / "config.json").write_text(json.dumps(dict(hamlet_mode="finetune",
            mem_cond_type="cross_attn", memory_type="moment_token", n_moment_tokens=4,
            memory_stride=16, backbone_embedding_dim=8)))
        (self.base / "processor_config.json").write_text(json.dumps({
            "processor_kwargs": dict(max_state_dim=4, max_action_dim=3)}))
        for name in ("statistics.json", "embodiment_id.json"):
            (self.base / name).write_text("{}")
        self.memory_config = VisualMemoryV6Config(feature_dim=8, state_dim=4, hidden_dim=8,
            visual_tokens=3, num_heads=2, max_archive_events=8, read_budget=3)
        self.memory = VisualMemoryV6(self.memory_config)
        self.lora = LoRAConfig(rank=2, alpha=4)
        targets = install_expert_lora(self.head, self.lora)
        bridge = install_memory_bridge(self.head, hidden_dim=8, block_indices=(0, 1), num_heads=2)
        self.config = dict(trainer_variant=VARIANT, stage=1, reader_mode="memory",
            memory=asdict(self.memory_config), expert=asdict(self.lora), expert_targets=targets,
            bridge=bridge)
        self.metadata = dict(base_model=checkpoint_identity(self.base), cache_fingerprint="cache-test-v6")

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name="run", step=1, optimizer=None, best=False):
        return save_checkpoint_v6(self.root / name, step, self.memory, self.head, optimizer,
                                  self.config, self.metadata, best=best)

    def mutate_manifest(self, path, change):
        info = json.loads((path / "checkpoint.json").read_text())
        change(info)
        (path / "checkpoint.json").write_text(json.dumps(info))

    def set_stage2(self):
        parent = self.save("parent")
        identity = frozen_identity(self.memory, self.head)
        self.config["stage"] = 2
        self.metadata["frozen_identity"] = identity
        self.metadata["stage1_parent"] = dict(path="/unmounted/previous-experiment",
            checkpoint_sha256=file_sha256(parent / "checkpoint.json"), frozen_identity=identity)

    def test_roundtrip_best_pointer_base_immutable_and_validation_rng_neutral(self):
        base = {p.name: p.read_bytes() for p in self.base.iterdir()}
        config, metadata = copy.deepcopy(self.config), copy.deepcopy(self.metadata)
        state = {name: tensor.detach().clone() for name, tensor in self.memory.state_dict().items()}
        expert, bridge = expert_state_dict(self.head), bridge_state_dict(self.head)
        path = self.save(best=True)
        self.assertEqual({p.name for p in path.iterdir()}, set(PAYLOADS) | {"checkpoint.json", "training_state.pt"})
        self.assertEqual(json.loads((path.parent / "best_checkpoint.json").read_text()),
                         {"path": "checkpoint-000001", "step": 1})
        before = torch.get_rng_state().clone()
        info = v6_checkpoint_info(self.base, path, expected_stage=1)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(info["metadata"]["payload_sha256"],
                         {name: file_sha256(path / name) for name in PAYLOADS})
        with torch.no_grad():
            for parameters in (self.memory.parameters(), expert_parameters(self.head), bridge_parameters(self.head)):
                next(iter(parameters)).add_(1)
        load_checkpoint_v6(path, self.memory, self.head)
        for expected, actual in ((state, self.memory.state_dict()), (expert, expert_state_dict(self.head)),
                                 (bridge, bridge_state_dict(self.head))):
            self.assertTrue(all(torch.equal(value, actual[key]) for key, value in expected.items()))
        self.assertEqual(config, self.config)
        self.assertEqual(metadata, self.metadata)
        self.assertEqual(base, {p.name: p.read_bytes() for p in self.base.iterdir()})
        with self.assertRaises(FileExistsError):
            self.save()

    def test_each_payload_rejects_names_shapes_dtype_nonfinite_and_finite_tamper(self):
        for file_index, filename in enumerate(PAYLOADS):
            for kind in ("name", "shape", "dtype", "nonfinite", "tamper"):
                with self.subTest(filename=filename, kind=kind):
                    path = self.save(f"damage-{file_index}-{kind}")
                    state = load_file(str(path / filename))
                    key = next(iter(state))
                    if kind == "name":
                        state.pop(key)
                    elif kind == "shape":
                        state[key] = state[key].reshape(-1)[:1]
                    elif kind == "dtype":
                        state[key] = state[key].bfloat16()
                    elif kind == "nonfinite":
                        state[key].flatten()[0] = float("nan")
                    else:
                        state[key].flatten()[0] += .1
                    save_file(state, str(path / filename))
                    with self.assertRaises(ValueError):
                        v6_checkpoint_info(self.base, path)

    def test_stage2_accepts_only_cvom_changes_and_does_not_need_parent_path(self):
        self.set_stage2()
        reader_before = reader_state_sha256(self.memory)
        with torch.no_grad():
            next(iter(self.memory.cvom_parameters())).add_(.1)
        path = self.save("stage2")
        self.assertEqual(v6_checkpoint_info(self.base, path)["config"]["stage"], 2)
        self.assertEqual(reader_before, reader_state_sha256(self.memory))
        for name, parameters in (("reader", self.memory.reader_parameters()),
                                 ("expert", list(expert_parameters(self.head))),
                                 ("bridge", list(bridge_parameters(self.head)))):
            p = parameters[0]
            old = p.detach().clone()
            with torch.no_grad():
                p.add_(.1)
            with self.assertRaisesRegex(ValueError, "frozen Stage-1"):
                self.save("invalid-" + name)
            self.assertFalse((self.root / ("invalid-" + name) / "checkpoint-000001").exists())
            with torch.no_grad():
                p.copy_(old)

    def test_stage2_frozen_identity_rechecks_payload_even_with_updated_file_hash(self):
        self.set_stage2()
        for index, filename in enumerate(PAYLOADS):
            path = self.save("frozen-tamper-" + str(index))
            state = load_file(str(path / filename))
            key = next(key for key in state if not key.startswith("cvom."))
            state[key].flatten()[0] += .1
            save_file(state, str(path / filename))
            self.mutate_manifest(path, lambda info: info["metadata"]["payload_sha256"].update(
                {filename: file_sha256(path / filename)}))
            with self.assertRaisesRegex(ValueError, "frozen Stage-1"):
                v6_checkpoint_info(self.base, path)

    def test_resume_optimizer_rng_and_training_state_hash(self):
        params = list(self.memory.parameters()) + list(expert_parameters(self.head)) + list(bridge_parameters(self.head))
        optimizer = torch.optim.AdamW(params, lr=.002)
        sum(p.square().sum() for p in params).backward()
        optimizer.step()
        path = self.save(optimizer=optimizer)
        next_random = torch.randn(4)
        optimizer.param_groups[0]["lr"] = .01
        torch.manual_seed(7)
        load_checkpoint_v6(path, self.memory, self.head, optimizer)
        self.assertTrue(torch.equal(next_random, torch.randn(4)))
        self.assertEqual(optimizer.param_groups[0]["lr"], .002)
        state = torch.load(path / "training_state.pt", weights_only=True)
        state["optimizer"]["param_groups"][0]["lr"] = .1
        torch.save(state, path / "training_state.pt")
        before = frozen_identity(self.memory, self.head)
        with self.assertRaisesRegex(ValueError, "optimizer/RNG state changed"):
            load_checkpoint_v6(path, self.memory, self.head, optimizer)
        self.assertEqual(before, frozen_identity(self.memory, self.head))

    def test_resume_requires_optimizer_state(self):
        path = self.save()
        optimizer = torch.optim.AdamW(self.memory.parameters())
        with self.assertRaisesRegex(ValueError, "no optimizer"):
            load_checkpoint_v6(path, self.memory, self.head, optimizer)

    def test_invalid_manifest_variant_stage_and_base_provenance(self):
        mutations = [lambda i: i["config"].update(trainer_variant="action_expert_v4"),
                     lambda i: i["config"].update(stage=True),
                     lambda i: i["config"].update(stage=0),
                     lambda i: i["metadata"].update(base_model={}),
                     lambda i: i["metadata"].update(cache_fingerprint=""),
                     lambda i: i["config"].update(reader_mode="unknown"),
                     lambda i: i["config"]["bridge"].update(expert_dim=9),
                     lambda i: i["config"]["bridge"].update(block_indices=[2]),
                     lambda i: i["config"]["memory"].update(time_scale=8)]
        for index, change in enumerate(mutations):
            path = self.save("manifest-" + str(index))
            self.mutate_manifest(path, change)
            with self.assertRaises(ValueError):
                v6_checkpoint_info(self.base, path)
        with self.assertRaisesRegex(ValueError, "Expected V6 Stage"):
            v6_checkpoint_info(self.base, self.save("wrong-stage"), expected_stage=2)

    def test_reader_none_is_valid_stage1_but_not_stage2(self):
        self.config["reader_mode"] = "none"
        path = self.save()
        self.assertEqual(v6_checkpoint_info(self.base, path)["config"]["reader_mode"], "none")
        self.mutate_manifest(path, lambda i: i["config"].update(stage=2))
        with self.assertRaisesRegex(ValueError, "AE-only control"):
            v6_checkpoint_info(self.base, path)

    def test_save_rejects_declared_memory_or_bridge_config_different_from_installed(self):
        original = copy.deepcopy(self.config)
        for index, mutation in enumerate((lambda c: c["memory"].update(temperature=.5),
                                          lambda c: c["bridge"].update(num_heads=4))):
            self.config = copy.deepcopy(original)
            mutation(self.config)
            with self.assertRaises(ValueError):
                self.save("wrong-installed-" + str(index))
            self.assertFalse((self.root / ("wrong-installed-" + str(index)) / "checkpoint-000001").exists())

    def test_load_rejects_same_shape_memory_semantic_mismatch_without_mutation(self):
        path = self.save()
        wrong = VisualMemoryV6(VisualMemoryV6Config(**dict(asdict(self.memory_config), temperature=.5)))
        before = {key: value.clone() for key, value in wrong.state_dict().items()}
        with self.assertRaises(ValueError):
            load_checkpoint_v6(path, wrong, self.head)
        self.assertTrue(all(torch.equal(value, wrong.state_dict()[key]) for key, value in before.items()))

    def test_load_rejects_same_shape_bridge_heads_mismatch(self):
        path = self.save()
        other = Head().eval().requires_grad_(False)
        install_expert_lora(other, self.lora)
        install_memory_bridge(other, hidden_dim=8, block_indices=(0, 1), num_heads=4)
        with self.assertRaises(ValueError):
            load_checkpoint_v6(path, self.memory, other)


if __name__ == "__main__":
    unittest.main()
