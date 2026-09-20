"""Sequential commands stay bounded, explicit and independent of shell state."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_scripts.robomme.long_memory_v5_workflow import (STEPS, best_checkpoint,
                                                        build_command, main)


class WorkflowV5Tests(unittest.TestCase):
    def test_every_step_builds_in_read_only_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "not_created"
            env = {"V5_RUN_DIR": str(run)}
            for step in STEPS:
                if step != "plan":
                    command = build_command(step, env, dry_run=True)
                    self.assertTrue(command[0].endswith(".venv/bin/python"))
                    self.assertIn("run_scripts/robomme/", command[1])
            self.assertFalse(run.exists())

    def test_control_matches_init_sampling_and_budget(self):
        reference = build_command("stage1", {}, dry_run=True)
        control = build_command("action-control", {}, dry_run=True)
        for flag in ("--init-checkpoint", "--cache-dir", "--recall-labels", "--delayed-fraction", "--max-steps", "--grad-accum"):
            self.assertEqual(reference[reference.index(flag)+1], control[control.index(flag)+1])
        self.assertEqual(control[-4:], ["--subgoal-weight", "0", "--grounding-weight", "0"])
        expert = build_command("expert-control", {}, dry_run=True)
        self.assertEqual(expert[-2:], ["--reader-mode", "none"])

    def test_preflight_is_not_smoke_and_development_uses_val(self):
        for step in ("stage1-preflight", "stage2-preflight", "eval-preflight"):
            self.assertIn("--preflight-only", build_command(step, {}, dry_run=True))
        for step in ("stage1-smoke", "stage2-smoke", "eval-smoke", "eval-reader-smoke"):
            self.assertNotIn("--preflight-only", build_command(step, {}, dry_run=True))
        for step in ("eval-val", "eval-reader-val", "eval-smoke"):
            command = build_command(step, {}, dry_run=True)
            self.assertEqual(command[command.index("--dataset")+1], "val")
        command = build_command("eval-test", {}, dry_run=True)
        self.assertEqual(command[command.index("--dataset")+1], "test")
        self.assertEqual(command[command.index("--tasks")+1], "all")

    def test_missing_previous_stage_does_not_start_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "preceding stage"):
                build_command("stage2", {"V5_RUN_DIR": tmp})

    def test_best_pointer_is_validated_and_can_use_nonfinal_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "checkpoint-003600"
            ckpt.mkdir()
            (ckpt / "checkpoint.json").write_text(json.dumps({"step": 3600}))
            (root / "best_checkpoint.json").write_text(json.dumps({"path": ckpt.name, "step": 3600}))
            self.assertEqual(best_checkpoint(root), ckpt)
            (root / "best_checkpoint.json").write_text(json.dumps({"path": "../outside", "step": 3600}))
            with self.assertRaises(ValueError):
                best_checkpoint(root)

    def test_dry_run_does_not_spawn_any_job(self):
        with patch("subprocess.call", side_effect=AssertionError("must not launch")):
            self.assertEqual(main(["stage1", "--dry-run"]), 0)


if __name__ == "__main__":
    unittest.main()
