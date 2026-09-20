"""CPU joint visual/LoRA serialization, atomic validation and exact resume."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from gr00t.long_memory.expert_v4 import (
    LoRAConfig, expert_parameters, expert_state_dict, install_expert_lora,
    load_expert_state_dict, set_expert_trainable,
)
from gr00t.long_memory.monitoring import _rng_state
from run_scripts.robomme import checkpoint_visual_expert_v14 as checkpoint
from run_scripts.robomme.visual_demo_tail_bank_v13 import VisualDemoTailMemoryV13
from tests.test_long_memory_v4_expert import FakeHead
from tests import test_checkpoint_visual_differential_v12 as fixtures


class JointCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.VisualCheckpointTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root, self.base, self.parent = self.fixture.root, self.fixture.base, self.fixture.parent
        parent_cfg = json.loads((self.parent / "checkpoint.json").read_text())["config"]
        self.head = self.make_head()
        self.visual = self.make_visual()
        self.config = {
            "trainer_variant": checkpoint.VARIANT, "driver_variant": checkpoint.VARIANT,
            "architecture": checkpoint.ARCHITECTURE, "mode": checkpoint.MODE, "stage": 1,
            "read_mode": "differential", "include_tail": True, "replay_encoding": checkpoint.REPLAY_ENCODING,
            "extraction_rule": checkpoint.EXTRACTION_RULE, "camera_order": list(checkpoint.CAMERA_ORDER),
            "visual": asdict(self.visual.config), "expert": parent_cfg["expert"], "expert_targets": parent_cfg["expert_targets"],
            "objective": {"trainable_scope": "visual_and_expert_lora", "flow_weight": 1.0,
                          "generated_auxiliary_weight": 0.0, "frozen_original_base": True,
                          "frozen_archive": True, "rollout_selection": "fixed_final_step"},
            "train": {"read_mode": "differential", "include_tail": True, "replay_encoding": checkpoint.REPLAY_ENCODING,
                      "max_steps": 4, "visual_learning_rate": .001, "expert_learning_rate": .0001},
        }
        self.metadata = {k: copy.deepcopy(v) for k, v in self.fixture.metadata.items() if k != "frozen_parent"}
        self.metadata.update(
            initial_parent=checkpoint.parent_reference(self.base, self.parent), frozen_base=checkpoint.base_reference(self.base),
            include_tail=True, extraction_rule=checkpoint.EXTRACTION_RULE, replay_encoding=checkpoint.REPLAY_ENCODING,
            plan_sha256="b" * 64,
            sidecar={"path": str(self.root / "not-installed-sidecar"), "fingerprint": "c" * 64,
                     "manifest_sha256": "d" * 64, "cache_fingerprint": "toy-cache", "kind": checkpoint.SIDECAR_KIND,
                     "scope": "inventory_train_val", "rule": checkpoint.EXTRACTION_RULE})

    def make_head(self):
        head = FakeHead().eval().requires_grad_(False)
        head.load_state_dict({n.removeprefix("action_head."): v for n, v in
                              load_file(str(self.base / "model.safetensors")).items()}, strict=True)
        cfg = json.loads((self.parent / "checkpoint.json").read_text())["config"]
        install_expert_lora(head, LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
        load_expert_state_dict(head, load_file(str(self.parent / "expert.safetensors")))
        set_expert_trainable(head, True)
        return head

    def make_visual(self):
        return checkpoint.bind_visual_semantics(VisualDemoTailMemoryV13(self.fixture.visual_config), include_tail=True)

    def extra(self, step):
        return {"driver_variant": checkpoint.VARIANT, "window_cursor": step, "plan_sha256": self.metadata["plan_sha256"],
                "include_tail": True, "replay_encoding": checkpoint.REPLAY_ENCODING,
                "sidecar_fingerprint": self.metadata["sidecar"]["fingerprint"]}

    def optimizer(self, visual=None, head=None):
        visual, head = visual if visual is not None else self.visual, head if head is not None else self.head
        named = checkpoint._named_parameters(visual, head)
        groups = []
        for kind in ("visual", "expert"):
            names = sorted(n for n in named if n.startswith(kind + "."))
            groups.append({"kind": kind, "name": kind, "param_names": names,
                           "params": [named[n] for n in names], "lr": .001 if kind == "visual" else .0001})
        return torch.optim.AdamW(groups)

    def save(self, name="v14", step=0, optimizer=None, **kwargs):
        return checkpoint.save_checkpoint(self.root / name, step, self.visual, self.head, optimizer,
                                          self.config, self.metadata, training_state=self.extra(step), **kwargs)

    def snapshot(self, visual=None, head=None, optimizer=None):
        visual, head = visual if visual is not None else self.visual, head if head is not None else self.head
        return copy.deepcopy({"visual": visual.state_dict(), "head": head.state_dict(), "rng": _rng_state(),
                              "optimizer": optimizer.state_dict() if optimizer is not None else None})

    def update(self, optimizer):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.rand(()) + random.random() + np.random.random()
        sum((p.square().sum() + p.sum()) * scale for g in optimizer.param_groups for p in g["params"]).backward()
        optimizer.step()

    def rewrite_training(self, path, callback):
        value = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        callback(value)
        torch.save(value, path / "training_state.pt")
        info = json.loads((path / "checkpoint.json").read_text())
        info["metadata"]["training_state_sha256"] = checkpoint.file_sha256(path / "training_state.pt")
        (path / "checkpoint.json").write_text(json.dumps(info))

    def rewrite_payload(self, path, filename, callback):
        value = load_file(str(path / filename))
        callback(value)
        save_file(value, str(path / filename))
        info = json.loads((path / "checkpoint.json").read_text())
        info["metadata"]["payload_sha256"][filename] = checkpoint.file_sha256(path / filename)
        (path / "checkpoint.json").write_text(json.dumps(info))

    def assert_tree(self, first, second):
        self.fixture.assert_tree(first, second)

    def test_roundtrip_distinct_four_files_no_sidecar_dependency_and_rng_neutral(self):
        optimizer = self.optimizer()
        before = self.snapshot(optimizer=optimizer)
        path = self.save(optimizer=optimizer, best=True)
        self.assert_tree(before, self.snapshot(optimizer=optimizer))
        self.assertFalse(Path(self.metadata["sidecar"]["path"]).exists())
        info = checkpoint.checkpoint_info(self.base, path)
        self.assertEqual(info["kind"], checkpoint.VARIANT)
        self.assertNotIn("frozen_parent", info["metadata"])
        self.assertEqual(info["training_state"], self.extra(0))
        self.assertEqual(set(p.name for p in path.iterdir()),
                         {"visual.safetensors", "expert.safetensors", "training_state.pt", "checkpoint.json"})
        with torch.no_grad():
            next(self.visual.parameters()).add_(1)
            next(expert_parameters(self.head)).add_(2)
        checkpoint.load_checkpoint(path, self.visual, self.head)
        self.assert_tree(before, self.snapshot(optimizer=optimizer))

    def test_v13_and_v14_loaders_reject_each_other(self):
        from run_scripts.robomme import checkpoint_demo_tail_v13 as previous
        path = self.save()
        with self.assertRaisesRegex(ValueError, "visual_demo_tail_v13"):
            previous.checkpoint_info(self.base, path)
        old = self.fixture.save(name="old_v12")
        with self.assertRaisesRegex(ValueError, "visual_expert_v14"):
            checkpoint.checkpoint_info(self.base, old)

    def test_production_sized_topology_owns_exactly270_tensors(self):
        head = FakeHead()
        first = head.model.transformer_blocks[0]
        head.model.transformer_blocks = torch.nn.ModuleList([copy.deepcopy(first) for _ in range(32)])
        targets = install_expert_lora(head, LoRAConfig())
        self.assertEqual(targets, checkpoint.EXPECTED_REAL_EXPERT_TARGETS)
        optimizer = self.optimizer(head=head)
        names = checkpoint.optimizer_parameter_names(optimizer, self.visual, head)
        self.assertEqual([len(row) for row in names], [14, 256])
        self.assertEqual(len(set(n for row in names for n in row)), 270)

    def test_step_zero_expert_exact_initial_parent_and_visual_zero(self):
        with torch.no_grad():
            next(expert_parameters(self.head)).add_(.01)
        with self.assertRaisesRegex(ValueError, "step-zero expert"):
            self.save()
        self.assertFalse((self.root / "v14").exists())
        load_expert_state_dict(self.head, load_file(str(self.parent / "expert.safetensors")))
        with torch.no_grad():
            self.visual.output_projection.weight.fill_(.01)
        with self.assertRaisesRegex(ValueError, "step-zero"):
            self.save()

    def test_exact_two_to_four_resume_restores_both_modules_optimizer_rng(self):
        optimizer = self.optimizer()
        original_base = {n: p.clone() for n, p in self.head.named_parameters() if ".lora_" not in n}
        for _ in range(2):
            self.update(optimizer)
        path = self.save(step=2, optimizer=optimizer)
        for _ in range(2):
            self.update(optimizer)
        expected = self.snapshot(optimizer=optimizer)
        visual, head = self.make_visual(), self.make_head()
        resumed_optimizer = self.optimizer(visual, head)
        info = checkpoint.load_checkpoint(path, visual, head, resumed_optimizer)
        self.assertEqual(info["training_state"], self.extra(2))
        for _ in range(2):
            self.update(resumed_optimizer)
        self.assert_tree(expected, self.snapshot(visual, head, resumed_optimizer))
        self.assert_tree(original_base, {n: p for n, p in head.named_parameters() if ".lora_" not in n})
        newpath = checkpoint.save_checkpoint(self.root / "resume_new", 4, visual, head, resumed_optimizer,
                                             self.config, self.metadata, best=True, training_state=self.extra(4))
        self.assertEqual(checkpoint.checkpoint_info(self.base, newpath)["step"], 4)

    def test_config_scope_topology_flags_and_horizon_fail_before_output(self):
        original = copy.deepcopy(self.config)
        edits = [("include_tail", False), ("mode", "visual_demo_tail"), ("architecture", "new_block"),
                 ("expert_targets", self.config["expert_targets"][:-1]), ("expert", {"rank": 3, "alpha": 4})]
        for key, value in edits:
            self.config = copy.deepcopy(original)
            self.config[key] = value
            with self.assertRaises(ValueError):
                self.save(name="bad")
            self.assertFalse((self.root / "bad").exists())
        self.config = original
        self.config["objective"]["parent_frozen"] = True
        with self.assertRaisesRegex(ValueError, "truthful"):
            self.save(name="bad")
        del self.config["objective"]["parent_frozen"]
        with self.assertRaisesRegex(ValueError, "horizon"):
            self.save(step=5)

    def test_initial_parent_metadata_cannot_claim_effective_ae_frozen(self):
        self.metadata["frozen_parent"] = self.metadata["initial_parent"]
        with self.assertRaisesRegex(ValueError, "frozen_parent"):
            self.save()
        self.assertFalse((self.root / "v14").exists())

    def test_ownership_duplicate_missing_frozen_outside_wrong_names(self):
        for mode in ("duplicate", "missing", "frozen", "outside", "names", "kind"):
            optimizer = self.optimizer()
            group = optimizer.param_groups[1]
            param = group["params"][0]
            if mode == "duplicate": group["params"].append(param); group["param_names"].append(group["param_names"][0])
            elif mode == "missing": group["params"].pop(); group["param_names"].pop()
            elif mode == "frozen": param.requires_grad_(False)
            elif mode == "outside": group["params"][0] = torch.nn.Parameter(torch.ones_like(param))
            elif mode == "names": group["param_names"][0] = "expert.wrong"
            else: group["kind"] = "visual"
            try:
                with self.assertRaisesRegex(ValueError, "optimizer"):
                    self.save(name="bad", optimizer=optimizer)
                self.assertFalse((self.root / "bad").exists())
            finally:
                param.requires_grad_(True)

    def test_base_ae_or_disabled_adapter_rejected(self):
        outside = self.head.state_encoder.weight
        outside.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "original AE weights"):
            self.save()
        outside.requires_grad_(False)
        adapter = self.head.get_submodule(self.config["expert_targets"][0])
        adapter.enabled = False
        with self.assertRaisesRegex(ValueError, "enabled"):
            self.save()

    def test_expert_corruption_rehashed_is_rejected_before_visual_mutation(self):
        path = self.save(step=1)
        for mutation in ("nan", "dtype", "shape", "name"):
            original = {p.name: p.read_bytes() for p in path.iterdir()}
            def corrupt(state):
                name = sorted(state)[-1]
                if mutation == "nan": state[name].flatten()[0] = float("nan")
                elif mutation == "dtype": state[name] = state[name].half()
                elif mutation == "shape": state[name] = state[name].flatten()
                else: state["unexpected"] = state.pop(name)
            self.rewrite_payload(path, "expert.safetensors", corrupt)
            before = self.snapshot()
            with self.assertRaisesRegex(ValueError, "expert"):
                checkpoint.load_checkpoint(path, self.visual, self.head)
            self.assert_tree(before, self.snapshot())
            for name, content in original.items(): (path / name).write_bytes(content)

    def test_bad_optimizer_or_rng_rehashed_cannot_partially_load_modules(self):
        optimizer = self.optimizer()
        self.update(optimizer)
        path = self.save(step=1, optimizer=optimizer)
        original = {p.name: p.read_bytes() for p in path.iterdir()}
        def bad_moment(t):
            first = next(iter(t["optimizer"]["state"].values()))
            first["exp_avg"] = torch.zeros(1)
        edits = [bad_moment, lambda t: t["extra"].update(window_cursor=0),
                 lambda t: t["rng"].update(torch=torch.zeros(2, dtype=torch.uint8)),
                 lambda t: t["optimizer"]["param_groups"][1].update(kind="visual"),
                 lambda t: t["optimizer"]["param_groups"][1].update(lr=float("nan"))]
        for edit in edits:
            self.rewrite_training(path, edit)
            before = self.snapshot(optimizer=optimizer)
            with self.assertRaises((ValueError, RuntimeError)):
                checkpoint.load_checkpoint(path, self.visual, self.head, optimizer)
            self.assert_tree(before, self.snapshot(optimizer=optimizer))
            for name, content in original.items(): (path / name).write_bytes(content)

    def test_installed_mismatch_load_is_atomic_for_visual_and_expert(self):
        path = self.save()
        self.visual.include_tail = False
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "installed"):
            checkpoint.load_checkpoint(path, self.visual, self.head)
        self.assert_tree(before, self.snapshot())
        self.visual.include_tail = True
        head = FakeHead().eval().requires_grad_(False)
        install_expert_lora(head, LoRAConfig(rank=3, alpha=4))
        before = self.snapshot(head=head)
        with self.assertRaises(ValueError):
            checkpoint.load_checkpoint(path, self.visual, head)
        self.assert_tree(before, self.snapshot(head=head))

    def test_initial_files_modified_reject_load_and_final_forces_scan(self):
        path = self.save()
        target = self.parent / "training_state.pt"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"changed")
            before = self.snapshot()
            with self.assertRaisesRegex(ValueError, "parent files changed"):
                checkpoint.load_checkpoint(path, self.visual, self.head)
            self.assert_tree(before, self.snapshot())
        finally:
            target.write_bytes(original)
        with patch.object(checkpoint, "verify_frozen_references", wraps=checkpoint.verify_frozen_references) as verify:
            self.save(name="last", step=4)
            verify.assert_called_once_with(self.metadata)

    def test_immutable_collision_and_failed_second_payload_preserve_previous_bundle(self):
        path = self.save(best=True)
        original = {p.name: p.read_bytes() for p in path.iterdir()}
        best = (path.parent / "best_checkpoint.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.save()
        actual_save = checkpoint.save_file
        def fail_expert(state, filename):
            if Path(filename).name == "expert.safetensors":
                raise RuntimeError("synthetic second-payload disk failure")
            return actual_save(state, filename)
        with patch.object(checkpoint, "save_file", side_effect=fail_expert):
            with self.assertRaisesRegex(RuntimeError, "disk failure"):
                self.save(step=1, best=True)
        self.assertEqual(original, {p.name: p.read_bytes() for p in path.iterdir()})
        self.assertEqual(best, (path.parent / "best_checkpoint.json").read_bytes())
        self.assertFalse(list(path.parent.glob(".checkpoint-*")))
        foreign = self.root / "foreign"
        (foreign / "checkpoint-000000").mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError, "preexisting"):
            self.save(name="foreign", step=1)


if __name__ == "__main__":
    unittest.main()
