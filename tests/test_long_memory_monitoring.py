"""CPU-only checks; can run with unittest when pytest is not installed."""

import csv
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from gr00t.long_memory.monitoring import RunLogger, load_checkpoint, save_checkpoint


class RunLoggerTests(unittest.TestCase):
    def test_append_resume_and_dynamic_csv_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            logger = RunLogger(output)
            logger.log(1, "train", {"action_loss": torch.tensor(0.25), "write_accuracy": np.float64(0.5)})
            logger.log(1, "val", {"action_loss": 0.3, "utility_corr": None, "write_precision": None})
            resumed = RunLogger(output)
            self.assertEqual(len(resumed.records), 2)
            resumed.log(2, "train", {"action_loss": 0.2, "new_metric": 7})
            with (output / "metrics.jsonl").open() as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual([row["step"] for row in records], [1, 1, 2])
            with (output / "metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 7)
            self.assertEqual(rows[-1]["metric"], "new_metric")
            self.assertEqual(rows[3]["value"], "")

    def test_reject_nonfinite_and_non_scalar_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger = RunLogger(temporary)
            for value in (float("nan"), float("inf"), -float("inf"), None, [1.0], torch.ones(2)):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    logger.log(0, "train", {"action_loss": value})
            with self.assertRaises(ValueError):
                logger.log(-1, "val", {"action_loss": 1})
            with self.assertRaises(ValueError):
                logger.log(1, "val", {"step": 1})
            self.assertEqual((Path(temporary) / "metrics.jsonl").read_text(), "")
            self.assertEqual(logger.records, [])

    def test_resume_before_first_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            RunLogger(temporary)
            resumed = RunLogger(temporary)
            resumed.log(0, "train", {"action_loss": 1.0})
            self.assertEqual(len(resumed.records), 1)

    def test_plots_latest_snapshot_and_undefined_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger = RunLogger(temporary)
            logger.log(2, "train", {"action_loss": 1, "write_accuracy": 0.5})
            logger.log(2, "val", {"action_loss": 2, "utility_corr": None})
            result = logger.plot()
            self.assertEqual(result, Path(temporary) / "curves.png")
            self.assertEqual(result.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            first = Path(temporary) / "plots" / "curves-step-000002.png"
            self.assertTrue(first.is_file())
            RunLogger(temporary).log(3, "val", {"action_loss": 0.9})
            RunLogger(temporary).plot()
            self.assertTrue(first.is_file())
            self.assertTrue((Path(temporary) / "plots" / "curves-step-000003.png").is_file())

    def test_csv_recovers_from_canonical_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger = RunLogger(temporary)
            logger.log(1, "train", {"action_loss": 0.2})
            (Path(temporary) / "metrics.csv").write_text("interrupted csv\n")
            resumed = RunLogger(temporary)
            with resumed.csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["metric"], "action_loss")

    def test_corrupt_log_is_not_silently_truncated(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            path.write_text('{"step": 1,')
            with self.assertRaisesRegex(ValueError, "inspect/recover"):
                RunLogger(temporary)
            self.assertEqual(path.read_text(), '{"step": 1,')


class CheckpointTests(unittest.TestCase):
    @staticmethod
    def update(model, optimizer):
        optimizer.zero_grad()
        prediction = model(torch.tensor([[1.0, 2.0], [0.5, -1.0]]))
        loss = (prediction - torch.tensor([[0.4], [0.2]])).square().mean()
        loss.backward()
        optimizer.step()

    def test_exact_tiny_model_resume_and_rng(self):
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        self.update(model, optimizer)
        with tempfile.TemporaryDirectory() as temporary:
            path = save_checkpoint(
                temporary, 1, model, optimizer,
                {"hidden_dim": 2}, {"base_checkpoint": "/external/base"},
                best=True, keep_last=None,
            )
            self.assertEqual(path.name, "checkpoint-000001")
            self.assertEqual(set(item.name for item in path.iterdir()), {
                "model.safetensors", "training_state.pt", "checkpoint.json",
            })
            best = json.loads((Path(temporary) / "best_checkpoint.json").read_text())
            self.assertEqual(best["path"], path.name)
            expected_random = (random.random(), float(np.random.rand()), torch.rand(3))
            self.update(model, optimizer)
            expected_weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
            resumed = torch.nn.Linear(2, 1)
            resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.4)
            info = load_checkpoint(path, resumed, resumed_optimizer)
            self.assertEqual(info["step"], 1)
            self.assertEqual(info["metadata"]["base_checkpoint"], "/external/base")
            self.assertEqual(random.random(), expected_random[0])
            self.assertEqual(float(np.random.rand()), expected_random[1])
            torch.testing.assert_close(torch.rand(3), expected_random[2], rtol=0, atol=0)
            self.assertEqual(resumed_optimizer.param_groups[0]["lr"], 0.01)
            self.update(resumed, resumed_optimizer)
            for name, tensor in resumed.state_dict().items():
                torch.testing.assert_close(tensor, expected_weights[name], rtol=0, atol=0)

    def test_weights_only_load_does_not_restore_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = torch.nn.Linear(2, 1)
            path = save_checkpoint(temporary, 3, model, None, {}, {}, keep_last=None)
            torch.manual_seed(21)
            random.seed(22)
            np.random.seed(23)
            before = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
            load_checkpoint(path, model)
            torch.testing.assert_close(torch.get_rng_state(), before[0])
            self.assertEqual(random.getstate(), before[1])
            np.testing.assert_array_equal(np.random.get_state()[1], before[2][1])
            optimizer = torch.optim.AdamW(model.parameters())
            with self.assertRaisesRegex(ValueError, "no optimizer"):
                load_checkpoint(path, model, optimizer)

    def test_no_overwrite_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = torch.nn.Linear(2, 1)
            path = save_checkpoint(temporary, 1, model, None, {}, {}, keep_last=None)
            original = (path / "model.safetensors").read_bytes()
            with self.assertRaises(FileExistsError):
                save_checkpoint(temporary, 1, model, None, {}, {}, keep_last=None)
            self.assertEqual((path / "model.safetensors").read_bytes(), original)
            with torch.no_grad():
                model.weight.fill_(float("nan"))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                save_checkpoint(temporary, 2, model, None, {}, {}, keep_last=None)
            self.assertFalse((Path(temporary) / "checkpoint-000002").exists())

    def test_base_model_wrapper_rejected_and_no_pruning(self):
        with tempfile.TemporaryDirectory() as temporary:
            wrapper = torch.nn.ModuleDict({"backbone": torch.nn.Linear(2, 1)})
            with self.assertRaisesRegex(ValueError, "only the new memory module"):
                save_checkpoint(temporary, 1, wrapper, None, {}, {}, keep_last=None)
            model = torch.nn.Linear(2, 1)
            for step in (1, 2, 3):
                with self.assertWarnsRegex(UserWarning, "pruning is disabled"):
                    save_checkpoint(temporary, step, model, None, {}, {}, keep_last=1)
            self.assertEqual(len(list(Path(temporary).glob("checkpoint-*"))), 3)


if __name__ == "__main__":
    unittest.main()
