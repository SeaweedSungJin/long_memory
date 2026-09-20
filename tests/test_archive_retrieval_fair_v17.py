"""CPU-only report/protocol tests. Toy payload bytes are identity fixtures, not models."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_scripts.robomme import compare_archive_retrieval_fair_v17 as report
from run_scripts.robomme import eval_archive_retrieval_fair_v17 as workflow


class FairReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoints = {}
        for role, step in (("parent", 1250), ("control", 512), ("guided", 512)):
            p = self.root / "checkpoints" / role
            p.mkdir(parents=True)
            for name in ("model.safetensors", "expert.safetensors", "cvom.safetensors"):
                (p / name).write_bytes(f"toy:{role}:{name}".encode())
            config = {"stage": 1, "mode": "archive", "driver_variant": "archive_retrieval_v17",
                      "train": {"max_steps": 512, "retrieval_weight": .01 if role == "guided" else 0,
                                "output_dir": str(p), "seed": 9171}}
            metadata = {"plan_sha256": "same-plan", "source_sha256": {"trainer.py": "same"}, "runtime": "same"}
            if role != "parent":
                origin = self.checkpoints["parent"]
                metadata["original_continuation_parent"] = {"path": str(origin), "step": 1250,
                    "files": {name: report.file_hash(origin / name) for name in report.PAYLOAD_KEYS}}
            (p / "checkpoint.json").write_text(json.dumps({"step": step, "config": config, "metadata": metadata}))
            self.checkpoints[role] = p
        self.runs = {role: self.make_run(role) for role in ("parent", "control", "guided")}

    def make_run(self, role):
        root = self.root / f"eval_{role}"
        root.mkdir()
        base = {"base_model": {"path": "same-base"}, "memory_checkpoint": None, "mode": "none",
                "write_policy": "none", "write_policy_override": "checkpoint",
                "memory_off": False, "archive_read_off": False}
        actor = {**base, **report.checkpoint_model(self.checkpoints[role]), "mode": "archive", "stage": 1,
                 "write_policy": "append"}
        models = {"baseline": base, "archive": actor}
        if role == "guided":
            models["archive-off"] = {**actor, "memory_off": True, "archive_read_off": True}
        manifest = {"trainer_variant": "archive_read_control_v7", "models": models,
                    "settings": {"tasks": ["BinFill"], "dataset": "val", "n_episodes": 4, "seed": 6,
                        "n_action_steps": 16, "max_episode_steps": 1300, "device": "cpu"},
                    "benchmark": {"versions": "same"}, "base_file_sha256": {"base": "same"},
                    "source_sha256": {"eval.py": "same"}, "policy_package_versions": {"torch": "same"}}
        self.write_manifest(root, manifest)
        for model in models:
            successes = [1, 0, 0, 0]
            if model == "archive":
                successes = {"parent": [1, 1, 0, 0], "control": [1, 0, 0, 0], "guided": [1, 1, 1, 0]}[role]
            (root / model / "BinFill/simulation_results.csv").write_text("episode_idx,episode_seed,success\n" +
                "".join(f"{i},{6+i},{s}\n" for i, s in enumerate(successes)))
        return root

    def write_manifest(self, root, manifest):
        manifest = copy.deepcopy(manifest)
        manifest.pop("evaluation_id", None)
        manifest["evaluation_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        (root / "comparison_manifest.json").write_text(json.dumps(manifest))
        for role in manifest["models"]:
            task = root / role / "BinFill"
            task.mkdir(parents=True, exist_ok=True)
            identity = {**manifest["settings"], "evaluation_id": manifest["evaluation_id"] + ":" + role,
                        "task_id": "BinFill", "scenario_metadata_sha256": "same", "model_config_sha256": "same",
                        "memory_window": 4, "demo_sampling": "same"}
            (task / "policy_manifest.json").write_text(json.dumps(identity))

    def build(self):
        return report.build_report(self.runs["parent"], self.runs["control"], self.runs["guided"], samples=40)

    def test_primary_parent_onoff_arithmetic_and_no_copy(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result, text = self.build()
        rows = result["comparisons"]
        self.assertEqual(rows[report.PRIMARY]["delta"], .5)
        self.assertEqual(rows["unchanged parent -> retrieval-guided"]["delta"], .25)
        self.assertEqual(rows["same guided AE READ-off -> READ-on"]["delta"], .5)
        self.assertEqual((rows[report.PRIMARY]["wins"], rows[report.PRIMARY]["losses"]), (2, 0))
        self.assertIn("exploratory", text)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_matching_family_before_rollout(self):
        identities = report.validate_checkpoint_family(*[self.checkpoints[k] for k in ("parent", "control", "guided")])
        self.assertEqual([i["step"] for i in identities], [1250, 512, 512])

    def test_wrong_parent_rejected_even_if_rollout_hashes_match(self):
        path = self.checkpoints["guided"] / "checkpoint.json"
        info = json.loads(path.read_text())
        info["metadata"]["original_continuation_parent"]["files"]["model.safetensors"] = "not-parent"
        path.write_text(json.dumps(info))
        # Both arms claim the same but wrong initializer: matched_training alone
        # would accept it, while explicit evaluated-parent verification must fail.
        other = self.checkpoints["control"] / "checkpoint.json"
        control = json.loads(other.read_text())
        control["metadata"]["original_continuation_parent"] = info["metadata"]["original_continuation_parent"]
        other.write_text(json.dumps(control))
        with self.assertRaisesRegex(ValueError, "parent payload"):
            report.validate_checkpoint_family(*[self.checkpoints[k] for k in ("parent", "control", "guided")])

    def test_post_rollout_payload_swap_rejected(self):
        (self.checkpoints["guided"] / "expert.safetensors").write_bytes(b"swapped")
        with self.assertRaisesRegex(ValueError, "payload changed"):
            self.build()

    def test_incomplete_episodes_rejected(self):
        (self.runs["parent"] / "archive/BinFill/simulation_results.csv").write_text("episode_idx,episode_seed,success\n0,6,1\n")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.build()

    def test_val_test_mixing_rejected(self):
        root = self.runs["parent"]
        manifest = json.loads((root / "comparison_manifest.json").read_text())
        manifest["settings"]["dataset"] = "test"
        self.write_manifest(root, manifest)
        with self.assertRaisesRegex(ValueError, "dataset"):
            self.build()

    def test_read_off_must_use_exact_same_checkpoint(self):
        root = self.runs["guided"]
        manifest = json.loads((root / "comparison_manifest.json").read_text())
        manifest["models"]["archive-off"]["expert_weights_sha256"] = "wrong-expert"
        self.write_manifest(root, manifest)
        with self.assertRaisesRegex(ValueError, "SAME checkpoint"):
            self.build()

    def test_changed_baseline_scenario_source_and_test_selection_rejected(self):
        root = self.runs["control"]
        manifest = json.loads((root / "comparison_manifest.json").read_text())
        manifest["source_sha256"]["eval.py"] = "changed"
        self.write_manifest(root, manifest)
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.build()
        manifest["source_sha256"]["eval.py"] = "same"
        self.write_manifest(root, manifest)
        (root / "baseline/BinFill/simulation_results.csv").write_text(
            "episode_idx,episode_seed,success\n0,6,0\n1,7,0\n2,8,0\n3,9,0\n")
        with self.assertRaisesRegex(ValueError, "baseline outcomes changed"):
            self.build()

    def test_historical_requires_complete_test_protocol(self):
        with self.assertRaisesRegex(ValueError, "all16 TEST"):
            report.build_report(*[self.runs[k] for k in ("parent", "control", "guided")], historical_dir=self.root / "old")


class WorkflowTests(unittest.TestCase):
    def test_default_plan_no_subprocess_no_output(self):
        with patch.object(workflow.subprocess, "run", side_effect=AssertionError("Plan launched subprocess")), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(workflow.main([]), 0)
        self.assertIn("5600 rollouts", output.getvalue())
        self.assertIn(str(workflow.ROOT / ".venv/bin/python"), output.getvalue())

    def test_protocol_changes_and_unsafe_namespace_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = workflow.parser().parse_args(["--test-root", temp])
            with self.assertRaisesRegex(ValueError, "direct NEW"):
                workflow.validate_test_root(args)
            args.test_root = Path(temp) / "new"
            workflow.bind_protocol(args, {"source": "original"}, write=True)
            workflow.bind_protocol(args, {"source": "original"}, write=False)
            with self.assertRaisesRegex(ValueError, "changed"):
                workflow.bind_protocol(args, {"source": "changed"}, write=True)
            self.assertEqual(json.loads((args.test_root / "fair_protocol.json").read_text()), {"source": "original"})


if __name__ == "__main__":
    unittest.main()
