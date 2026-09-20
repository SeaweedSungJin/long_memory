"""V5 validates training provenance without changing inference inputs."""
import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).parent))
import test_long_memory_v5_checkpoint as fixtures
from gr00t.long_memory.checkpoint_v4 import file_sha256, reader_state_sha256, _state_sha256
from gr00t.long_memory.checkpoint_v5 import recall_state_sha256
from gr00t.long_memory.expert_v4 import expert_state_dict
from run_scripts.robomme import eval_long_memory_v5 as driver


class EvaluationV5Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BundleV5Tests()
        self.fixture.setUp()
        f = self.fixture
        self.reader = f.save("reader")
        f.config["stage"] = 2
        f.metadata.update(stage1_parent={"path": str(self.reader),
            "checkpoint_sha256": file_sha256(self.reader / "checkpoint.json"),
            "memory_sha256": file_sha256(self.reader / "model.safetensors"),
            "expert_sha256": file_sha256(self.reader / "expert.safetensors"),
            "recall_sha256": file_sha256(self.reader / "recall.safetensors")},
            frozen_reader_sha256=reader_state_sha256(f.memory),
            frozen_expert_sha256=_state_sha256(expert_state_dict(f.head)),
            frozen_recall_sha256=recall_state_sha256(f.recall))
        self.memory = f.save("writer")
        self.args = driver.build_parser().parse_args(["--base-model", str(f.base),
            "--reader-checkpoint", str(self.reader), "--memory-checkpoint", str(self.memory),
            "--models", "baseline", "reader", "memory", "fifo"])

    def tearDown(self):
        self.fixture.tearDown()

    def test_compatible_parent_and_auxiliary_validated(self):
        self.assertEqual(set(driver.validate_bundles(self.args)), {1, 2})

    def test_different_reader_rejected_even_when_architecture_matches(self):
        f = self.fixture
        f.config["stage"] = 1
        with torch.no_grad():
            next(f.memory.reader_parameters()).add_(1)
        self.args.reader_checkpoint = f.save("different_reader")
        with self.assertRaisesRegex(ValueError, "actual Stage-1 parent"):
            driver.validate_bundles(self.args)

    def test_auxiliary_corruption_rejected_before_inference(self):
        (self.reader / "recall.safetensors").rename(self.reader / "missing_aux")
        with self.assertRaises(FileNotFoundError):
            driver.validate_bundles(self.args)

    def test_reader_alone_does_not_require_writer_training(self):
        self.args.models = ["baseline", "reader", "expert-only"]
        self.args.memory_checkpoint = None
        self.assertEqual(set(driver.validate_bundles(self.args)), {1})

    def test_policy_command_has_no_auxiliary_or_label_arguments(self):
        model = {"base_model": {"path": str(self.fixture.base)},
                 "memory_checkpoint": str(self.memory), "write_policy_override": "checkpoint", "expert_only": False}
        command = driver.engine.server_command(self.args, model, 42456)
        self.assertNotIn("--recall-labels", command)
        self.assertNotIn("--subgoal", command)
        self.assertTrue(any("serve_long_memory_v4.py" in x for x in command))


if __name__ == "__main__":
    unittest.main()
