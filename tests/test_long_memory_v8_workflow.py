"""Controller orchestration only: no model, training or simulator is launched."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "run_scripts/robomme/run_long_memory_v8_workflow.py"
SPEC = importlib.util.spec_from_file_location("v8_workflow_fixture", SOURCE)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class V8WorkflowTests(unittest.TestCase):
    def exercise(self, best_step):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        run = root / "runs/long_memory/v8_event_full_v1"
        calls = []
        def fake_step(name, command, control):
            calls.append((name, command))
            if name == "train":
                run.mkdir(parents=True)
                (run / "status.json").write_text(json.dumps({"status": "complete"}))
                (run / "last_checkpoint.json").write_text(json.dumps({"path": str(run / "checkpoint-009108"), "step": 9108}))
                (run / "best_checkpoint.json").write_text(json.dumps({"path": f"checkpoint-{best_step:06d}", "step": best_step}))
        cache = SimpleNamespace(manifest={"dataset_path": str(root / "data"), "model_path": str(root / "base")})
        with patch.object(workflow, "ROOT", root), patch.object(workflow, "run_step", fake_step), \
             patch.object(workflow, "EpisodeCache", return_value=cache):
            self.assertEqual(workflow.main([]), 0)
            with self.assertRaises(FileExistsError):
                workflow.main([])
        status = json.loads((run.with_name(run.name + "_workflow") / "status.json").read_text())
        return calls, status

    def test_best_and_last_evaluated_only_on_val(self):
        calls, status = self.exercise(5500)
        self.assertEqual([name for name, _ in calls], ["train_preflight", "train", "last_action_audit",
            "last_eval_preflight", "last_eval", "best_action_audit", "best_eval_preflight", "best_eval"])
        for name, command in calls:
            if "eval" in name:
                self.assertEqual(command[command.index("--dataset") + 1], "val")
        self.assertFalse(status["goal_achieved"])

    def test_step_zero_best_is_not_called_trained_memory(self):
        calls, _ = self.exercise(0)
        self.assertFalse(any(name.startswith("best") for name, _ in calls))

    def test_same_best_and_last_not_duplicated(self):
        calls, _ = self.exercise(9108)
        self.assertEqual(sum(name.endswith("_eval") for name, _ in calls), 1)


if __name__ == "__main__":
    unittest.main()
