"""V5 auxiliary weights must publish/restore atomically with the v4 payload."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
from test_long_memory_v4_expert import BundleV4Tests
from gr00t.long_memory.checkpoint_v4 import (file_sha256, reader_state_sha256,
                                            _state_sha256, v4_checkpoint_info)
from gr00t.long_memory.checkpoint_v5 import (RECIPE, load_checkpoint_v5,
    recall_state_sha256, save_checkpoint_v5, v5_checkpoint_info)
from gr00t.long_memory.expert_v4 import expert_parameters, expert_state_dict
from gr00t.long_memory.recall_v5 import RecallHeads


class BundleV5Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = BundleV4Tests()
        self.fixture.setUp()
        for name in ("root", "base", "head", "memory", "config", "metadata"):
            setattr(self, name, getattr(self.fixture, name))
        self.recall = RecallHeads(8, 3)
        self.config.update(training_recipe=RECIPE, recall={"hidden_dim": 8, "num_classes": 3})
        self.metadata["recall_labels_fingerprint"] = "a" * 64

    def tearDown(self):
        self.fixture.tearDown()

    def save(self, name="v5", step=1, optimizer=None):
        return save_checkpoint_v5(self.root / name, step, self.memory, self.head,
                                  self.recall, optimizer, self.config, self.metadata)

    def test_bundle_is_v4_deployable_but_auxiliary_validated_separately(self):
        path = self.save()
        self.assertEqual(v4_checkpoint_info(self.base, path)["step"], 1)
        self.assertEqual(v5_checkpoint_info(self.base, path)["metadata"]["recall_sha256"], file_sha256(path / "recall.safetensors"))
        expected = recall_state_sha256(self.recall)
        with torch.no_grad():
            next(self.recall.parameters()).add_(1)
        load_checkpoint_v5(path, self.memory, self.head, self.recall)
        self.assertEqual(recall_state_sha256(self.recall), expected)

    def test_recall_tampering_rejected_before_loading_any_weights(self):
        path = self.save()
        state = load_file(str(path / "recall.safetensors"))
        next(iter(state.values())).add_(1)
        save_file(state, str(path / "recall.safetensors"))
        before = recall_state_sha256(self.recall)
        with self.assertRaisesRegex(ValueError, "changed after"):
            load_checkpoint_v5(path, self.memory, self.head, self.recall)
        self.assertEqual(recall_state_sha256(self.recall), before)

    def test_refuses_old_recipe_missing_labels_or_overwrite(self):
        path = self.save()
        with self.assertRaises(FileExistsError):
            self.save()
        self.config["training_recipe"] = "wrong"
        with self.assertRaises(ValueError):
            self.save("bad")
        self.config["training_recipe"] = RECIPE
        self.metadata.pop("recall_labels_fingerprint")
        with self.assertRaises(ValueError):
            self.save("bad")
        self.assertTrue(path.is_dir())
        self.assertFalse((self.root / "bad").exists())

    def test_stage2_frozen_recall_enforced(self):
        parent = self.save("parent")
        self.config["stage"] = 2
        self.metadata.update(stage1_parent={"path": str(parent),
            "checkpoint_sha256": file_sha256(parent / "checkpoint.json"),
            "memory_sha256": file_sha256(parent / "model.safetensors"),
            "expert_sha256": file_sha256(parent / "expert.safetensors"),
            "recall_sha256": file_sha256(parent / "recall.safetensors")},
            frozen_reader_sha256=reader_state_sha256(self.memory),
            frozen_expert_sha256=_state_sha256(expert_state_dict(self.head)),
            frozen_recall_sha256=recall_state_sha256(self.recall))
        path = self.save("writer")
        self.assertEqual(v5_checkpoint_info(self.base, path)["config"]["stage"], 2)
        with torch.no_grad():
            next(self.recall.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, "frozen recall"):
            self.save("contaminated")

    def test_failed_publish_has_no_partial_checkpoint(self):
        with patch("gr00t.long_memory.checkpoint_v5.save_checkpoint_v4", side_effect=RuntimeError("intentional")):
            with self.assertRaises(RuntimeError):
                self.save()
        self.assertFalse((self.root / "v5" / "checkpoint-000001").exists())
        self.assertEqual(list((self.root / "v5").iterdir()), [])

    def test_optimizer_and_rng_restore_include_auxiliary_parameters(self):
        params = list(self.memory.parameters()) + list(expert_parameters(self.head)) + list(self.recall.parameters())
        optimizer = torch.optim.AdamW(params, lr=.001)
        self.recall(torch.randn(1, 4, 8))["logits"].sum().backward()
        optimizer.step()
        path = self.save(optimizer=optimizer)
        rng = torch.get_rng_state().clone()
        expected = copy.deepcopy(optimizer.state_dict())
        torch.rand(100)
        optimizer.param_groups[0]["lr"] = 5
        load_checkpoint_v5(path, self.memory, self.head, self.recall, optimizer)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(optimizer.param_groups[0]["lr"], expected["param_groups"][0]["lr"])
        self.assertEqual(len(optimizer.state), len(expected["state"]))


if __name__ == "__main__":
    unittest.main()
