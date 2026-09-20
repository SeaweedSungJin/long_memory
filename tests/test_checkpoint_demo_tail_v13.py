"""CPU-only distinct V13 format, immutable originals, semantic and RNG tests."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from gr00t.long_memory.monitoring import _rng_state
from run_scripts.robomme import checkpoint_demo_tail_v13 as checkpoint
from run_scripts.robomme.visual_demo_tail_bank_v13 import VisualDemoTailMemoryV13
from tests import test_checkpoint_visual_differential_v12 as fixtures


class DemoTailCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.VisualCheckpointTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root, self.base, self.parent = self.fixture.root, self.fixture.base, self.fixture.parent
        self.visual = checkpoint.bind_visual_semantics(VisualDemoTailMemoryV13(self.fixture.visual_config), include_tail=True)
        self.config = {"trainer_variant": checkpoint.VARIANT, "driver_variant": checkpoint.VARIANT,
                       "stage": 1, "mode": checkpoint.MODE, "read_mode": "differential", "include_tail": True,
                       "visual": asdict(self.visual.config), "camera_order": list(checkpoint.CAMERA_ORDER),
                       "extraction_rule": checkpoint.EXTRACTION_RULE, "replay_encoding": checkpoint.REPLAY_ENCODING,
                       "train": {"objective": "flow", "read_mode": "differential", "include_tail": True,
                                 "replay_encoding": checkpoint.REPLAY_ENCODING, "max_steps": 4}}
        self.metadata = {**self.fixture.metadata, "frozen_base": checkpoint.base_reference(self.base),
                         "include_tail": True, "extraction_rule": checkpoint.EXTRACTION_RULE,
                         "replay_encoding": checkpoint.REPLAY_ENCODING, "plan_sha256": "b" * 64,
                         "sidecar": {"path": str(self.root / "not-installed-sidecar"), "fingerprint": "c" * 64,
                                     "manifest_sha256": "d" * 64, "cache_fingerprint": "toy-cache",
                                     "kind": checkpoint.SIDECAR_KIND, "scope": "inventory_train_val", "rule": checkpoint.EXTRACTION_RULE}}

    def extra(self, step):
        return {"driver_variant": checkpoint.VARIANT, "window_cursor": step,
                "plan_sha256": self.metadata["plan_sha256"], "include_tail": self.config["include_tail"],
                "replay_encoding": checkpoint.REPLAY_ENCODING, "sidecar_fingerprint": self.metadata["sidecar"]["fingerprint"]}

    def optimizer(self, visual=None):
        return self.fixture.optimizer(self.visual if visual is None else visual)

    def save(self, name="v13", step=0, optimizer=None, **kwargs):
        return checkpoint.save_checkpoint(self.root / name, step, self.visual, optimizer, self.config,
                                          self.metadata, training_state=self.extra(step), **kwargs)

    def snapshot(self, visual=None, optimizer=None):
        return copy.deepcopy({"visual": (self.visual if visual is None else visual).state_dict(), "rng": _rng_state(),
                              "optimizer": optimizer.state_dict() if optimizer is not None else None})

    def test_bind_adds_no_parameters_or_rng_and_never_flips_arm(self):
        visual = VisualDemoTailMemoryV13(self.fixture.visual_config)
        original, rng = copy.deepcopy(visual.state_dict()), _rng_state()
        checkpoint.bind_visual_semantics(visual, include_tail=False)
        self.fixture.assert_tree(original, visual.state_dict())
        self.fixture.assert_tree(rng, _rng_state())
        self.assertEqual(len(visual.state_dict()), 14)
        self.assertEqual(visual.replay_encoding, "framewise")
        with self.assertRaisesRegex(ValueError, "silently"):
            checkpoint.bind_visual_semantics(visual, include_tail=True)
        with self.assertRaisesRegex(ValueError, "Boolean"):
            checkpoint.bind_visual_semantics(VisualDemoTailMemoryV13(self.fixture.visual_config), include_tail=1)

    def test_distinct_roundtrip_no_sidecar_inference_dependency(self):
        optimizer = self.optimizer()
        before = self.snapshot(optimizer=optimizer)
        path = self.save(optimizer=optimizer, best=True)
        self.fixture.assert_tree(before, self.snapshot(optimizer=optimizer))
        self.assertFalse(Path(self.metadata["sidecar"]["path"]).exists())
        info = checkpoint.checkpoint_info(self.base, path)
        self.assertEqual(info["kind"], checkpoint.VARIANT)
        self.assertEqual(info["training_state"], self.extra(0))
        self.assertFalse(info["self_contained"])
        with torch.no_grad():
            next(self.visual.parameters()).add_(1)
        checkpoint.load_checkpoint(path, self.visual)
        self.fixture.assert_tree(before, self.snapshot(optimizer=optimizer))
        from run_scripts.robomme.checkpoint_visual_differential_v12 import checkpoint_info as old_info
        with self.assertRaisesRegex(ValueError, "visual_differential_v12"):
            old_info(self.base, path)
        old = self.fixture.save(name="old_v12")
        with self.assertRaisesRegex(ValueError, "visual_demo_tail_v13"):
            checkpoint.checkpoint_info(self.base, old)

    def test_bind_rejects_late_semantic_mismatch_without_partial_attributes(self):
        visual = VisualDemoTailMemoryV13(self.fixture.visual_config)
        visual.replay_encoding = "batch"
        before = self.snapshot(visual)
        with self.assertRaisesRegex(ValueError, "replay_encoding"):
            checkpoint.bind_visual_semantics(visual, include_tail=True)
        self.assertFalse(hasattr(visual, "include_tail"))
        self.assertFalse(hasattr(visual, "extraction_rule"))
        self.fixture.assert_tree(before, self.snapshot(visual))

    def test_control_and_framewise_identity_rejected_before_mutation(self):
        path = self.save()
        control = checkpoint.bind_visual_semantics(VisualDemoTailMemoryV13(self.fixture.visual_config), include_tail=False)
        before = self.snapshot(control)
        with self.assertRaisesRegex(ValueError, "installed"):
            checkpoint.load_checkpoint(path, control)
        self.fixture.assert_tree(before, self.snapshot(control))
        unbound = VisualDemoTailMemoryV13(self.fixture.visual_config)
        with self.assertRaisesRegex(ValueError, "bind explicitly"):
            checkpoint.load_checkpoint(path, unbound)
        for container in (self.config, self.config["train"], self.metadata):
            original = container["replay_encoding"]
            container["replay_encoding"] = "batch"
            with self.assertRaises(ValueError):
                self.save(name="bad_encoding")
            container["replay_encoding"] = original
        self.visual = control
        for container in (self.config, self.config["train"], self.metadata):
            container["include_tail"] = False
        saved = self.save(name="control")
        self.assertFalse(checkpoint.checkpoint_info(self.base, saved)["config"]["include_tail"])

    def test_exact_two_to_four_adamw_rng_resume(self):
        optimizer = self.optimizer()
        for _ in range(2):
            self.fixture.update(self.visual, optimizer)
        path = self.save(step=2, optimizer=optimizer)
        for _ in range(2):
            self.fixture.update(self.visual, optimizer)
        expected = self.snapshot(optimizer=optimizer)
        resumed = checkpoint.bind_visual_semantics(VisualDemoTailMemoryV13(self.fixture.visual_config), include_tail=True)
        resumed_optimizer = self.optimizer(resumed)
        info = checkpoint.load_checkpoint(path, resumed, resumed_optimizer)
        self.assertEqual(info["training_state"], self.extra(2))
        for _ in range(2):
            self.fixture.update(resumed, resumed_optimizer)
        self.fixture.assert_tree(expected, self.snapshot(resumed, resumed_optimizer))

    def test_extra_cursor_plan_tail_and_encoding_are_strict(self):
        optimizer = self.optimizer()
        for key, value in (("window_cursor", 1), ("plan_sha256", "e" * 64),
                           ("include_tail", False), ("sidecar_fingerprint", "f" * 64), ("replay_encoding", "batch")):
            extra = {**self.extra(0), key: value}
            with self.assertRaisesRegex(ValueError, "exact resume"):
                checkpoint.save_checkpoint(self.root / "bad", 0, self.visual, optimizer, self.config,
                                           self.metadata, training_state=extra)
            self.assertFalse((self.root / "bad").exists())

    def test_base_hash_cache_and_forced_final_scan(self):
        checkpoint._BASE_CACHE.clear()
        with patch.object(checkpoint, "file_sha256", wraps=checkpoint.file_sha256) as hashed:
            first = checkpoint.base_reference(self.base)
            count = hashed.call_count
            self.assertGreater(count, 0)
            self.assertEqual(checkpoint.base_reference(self.base), first)
            self.assertEqual(hashed.call_count, count)
            self.assertEqual(checkpoint.base_reference(self.base, refresh=True), first)
            self.assertEqual(hashed.call_count, count * 2)
        with patch.object(checkpoint, "verify_frozen_references", wraps=checkpoint.verify_frozen_references) as final:
            self.save(name="last", step=4)
            final.assert_called_once()

    def test_base_content_parent_and_sidecar_schema_mismatches_fail_readonly(self):
        path = self.save()
        before = self.snapshot()
        marker = self.base / "statistics.json"
        original = marker.read_bytes()
        try:
            marker.write_bytes(b'{"changed":1}')
            with self.assertRaises(ValueError):
                checkpoint.load_checkpoint(path, self.visual)
            self.fixture.assert_tree(before, self.snapshot())
        finally:
            marker.write_bytes(original)
        parent_file = self.parent / "training_state.pt"
        original = parent_file.read_bytes()
        try:
            parent_file.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "parent changed"):
                checkpoint.load_checkpoint(path, self.visual)
            self.fixture.assert_tree(before, self.snapshot())
        finally:
            parent_file.write_bytes(original)
        self.metadata["sidecar"]["rule"] = "all future frames"
        with self.assertRaisesRegex(ValueError, "sidecar provenance"):
            self.save(name="bad_sidecar")

    def test_base_scan_rejects_files_modified_during_hashing(self):
        checkpoint._BASE_CACHE.clear()
        target = self.base / "statistics.json"
        original, hashed = target.read_bytes(), checkpoint.file_sha256

        def changed(path):
            result = hashed(path)
            if Path(path) == target:
                target.write_bytes(original + b" ")
            return result

        try:
            with patch.object(checkpoint, "file_sha256", side_effect=changed):
                with self.assertRaisesRegex(ValueError, "during full-file"):
                    checkpoint.base_reference(self.base, refresh=True)
        finally:
            target.write_bytes(original)

    def test_training_sidecar_reference_loads_every_payload_and_rejects_unchecked_inventory(self):
        path = self.root / "sidecar"
        path.mkdir()
        (path / "manifest.json").write_text("{}")
        manifest = {"fingerprint": "a" * 64, "driver_variant": "demo_tail_inventory_v13",
                    "plan": {"scope": "inventory_train_val", "rule": checkpoint.EXTRACTION_RULE},
                    "inventory_checks": dict.fromkeys(("zero_action_calls", "original_short_memory_unchanged",
                                                        "frozen_model_content_unchanged"), True)}
        reader = SimpleNamespace(path=path, manifest=manifest, _records={0: {}, 2: {}}, load=Mock())
        with patch.object(checkpoint, "DemoTailSidecar", return_value=reader) as constructor:
            reference = checkpoint.sidecar_reference(path, "toy-cache")
            constructor.assert_called_once_with(path, expected_cache_fingerprint="toy-cache")
            self.assertEqual([call.args[0] for call in reader.load.call_args_list], [0, 2])
            self.assertEqual(reference["manifest_sha256"], checkpoint.file_sha256(path / "manifest.json"))
            manifest["inventory_checks"]["zero_action_calls"] = False
            with self.assertRaisesRegex(ValueError, "integrity checks"):
                checkpoint.sidecar_reference(path, "toy-cache")
            manifest["plan"]["scope"] = "proof_subset_only"
            with self.assertRaisesRegex(ValueError, "proof subset"):
                checkpoint.sidecar_reference(path, "toy-cache")
            self.assertEqual(checkpoint.sidecar_reference(path, "toy-cache", allow_proof_subset=True)["scope"],
                             "proof_subset_only")

    def test_finite_zero_output_and_optimizer_scope(self):
        with torch.no_grad():
            self.visual.output_projection.weight.fill_(1)
        with self.assertRaisesRegex(ValueError, "step-zero"):
            self.save()
        with torch.no_grad():
            self.visual.output_projection.weight.zero_()
        optimizer = self.optimizer()
        optimizer.param_groups[0]["params"].append(nn_parameter := torch.nn.Parameter(torch.ones(1)))
        with self.assertRaisesRegex(ValueError, "non-visual"):
            self.save(optimizer=optimizer)
        optimizer.param_groups[0]["params"].pop()
        with torch.no_grad():
            next(self.visual.parameters()).flatten()[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.save(step=1)

    def test_immutable_steps_foreign_run_and_failed_save_preserve_best(self):
        path = self.save(best=True)
        original = {str(p): p.read_bytes() for p in path.iterdir()}
        pointer = (path.parent / "best_checkpoint.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.save()
        with patch.object(checkpoint, "save_file", side_effect=RuntimeError("synthetic disk failure")):
            with self.assertRaisesRegex(RuntimeError, "disk failure"):
                self.save(step=1, best=True)
        self.assertEqual(pointer, (path.parent / "best_checkpoint.json").read_bytes())
        self.assertEqual(original, {str(p): p.read_bytes() for p in path.iterdir()})
        self.assertFalse(list(path.parent.glob(".checkpoint-*")))
        foreign = self.root / "foreign"
        (foreign / "checkpoint-000000").mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError, "preexisting"):
            self.save(name="foreign", step=1)

    def test_rehashed_bad_resume_payload_rejected_atomically(self):
        optimizer = self.optimizer()
        self.fixture.update(self.visual, optimizer)
        path = self.save(step=1, optimizer=optimizer)
        saved = torch.load(path / "training_state.pt", weights_only=True)
        saved["extra"]["include_tail"] = False
        torch.save(saved, path / "training_state.pt")
        info = json.loads((path / "checkpoint.json").read_text())
        info["metadata"]["training_state_sha256"] = checkpoint.file_sha256(path / "training_state.pt")
        (path / "checkpoint.json").write_text(json.dumps(info))
        before = self.snapshot(optimizer=optimizer)
        with self.assertRaisesRegex(ValueError, "exact resume"):
            checkpoint.load_checkpoint(path, self.visual, optimizer)
        self.fixture.assert_tree(before, self.snapshot(optimizer=optimizer))

    def test_proof_subset_reference_requires_explicit_training_opt_in(self):
        self.metadata["sidecar"]["scope"] = "proof_subset_only"
        with self.assertRaisesRegex(ValueError, "diagnostic training opt-in"):
            self.save()
        self.config["train"]["allow_proof_subset"] = True
        path = self.save()
        self.assertEqual(checkpoint.checkpoint_info(self.base, path)["metadata"]["sidecar"]["scope"], "proof_subset_only")


if __name__ == "__main__":
    unittest.main()
