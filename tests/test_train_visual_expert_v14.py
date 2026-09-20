"""CPU V14 integration: real replay, joint LoRA optimizer and new bundles.
Synthetic width-six observations/AE do not establish robot-task performance.
Original source modules are never edited; the toy sidecar width is test-only.
"""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from safetensors.torch import load_file, save_file
import torch

from gr00t.long_memory.checkpoint_v7 import save_checkpoint_v7
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora
from gr00t.long_memory.hamlet import checkpoint_identity, isolated_seed
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from run_scripts.robomme import checkpoint_visual_expert_v14 as bundles
from run_scripts.robomme import checkpoint_visual_patch_v11 as reference_checkpoint
from run_scripts.robomme import checkpoint_demo_tail_v13 as sidecar_contract
from run_scripts.robomme import train_visual_expert_v14 as trainer
from run_scripts.robomme.visual_differential_memory_v12 import VisualDifferentialConfig
from run_scripts.robomme.visual_patch_memory_v11 import VisualPatchConfig, VisualPatchMemoryV11
from tests import test_visual_differential_trainer_v12 as legacy


class TinySidecarReader:
    """Test-only disk reader; checks toy payload identity, not real extraction."""
    def __init__(self, path, *, expected_cache_fingerprint=None):
        self.path = Path(path).resolve(strict=True)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("status") != "complete":
            raise ValueError("Synthetic sidecar incomplete")
        if self.manifest["plan"]["cache_fingerprint"] != expected_cache_fingerprint:
            raise ValueError("Synthetic sidecar/cache mismatch")
        self._records = {r["episode_id"]: r for r in self.manifest["episodes"]}
        if len(self._records) != len(self.manifest["episodes"]):
            raise ValueError("Synthetic duplicate episode")

    def load(self, eid):
        record = self._records[eid]
        path = self.path / record["path"]
        if trainer.file_hash(path) != record["payload_sha256"]:
            raise ValueError("Synthetic payload hash changed")
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if (payload["episode_id"] != eid or payload["cache_fingerprint"] != self.manifest["plan"]["cache_fingerprint"]
                or payload["sidecar_fingerprint"] != self.manifest["fingerprint"]
                or payload["images"].shape != (len(record["frames"]), 2, 81, 6)
                or payload["images"].dtype != torch.bfloat16 or not bool(torch.isfinite(payload["images"]).all())
                or payload["frames"].tolist() != record["frames"] or not bool(payload["is_demo"].all())):
            raise ValueError("Synthetic payload contract mismatch")
        return payload


def float_view(episode, query):
    # The toy CPU linear AE has FP32 parameters; real deployment uses its native
    # BF16/autocast path. This cast retains the BF16 visual residual gradient.
    return {**episode, "features": {query: episode["features"][query].float()}}


def toy_flow(head, episode, query, *args, **kwargs):
    return legacy.toy_flow(head, float_view(episode, query), query, *args, **kwargs)


def toy_generated(head, episode, query, *args, **kwargs):
    return legacy.toy_generated(head, float_view(episode, query), query, *args, **kwargs)


class VisualExpertTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.base = self.root / "base"
        self.base.mkdir()
        values = {"config.json": {"hamlet_mode": "finetune", "memory_type": "moment_token",
            "mem_cond_type": "cross_attn", "n_moment_tokens": 2, "backbone_embedding_dim": 6,
            "memory_window": 2, "memory_stride": 2}, "processor_config.json": {"max_state_dim": 3, "max_action_dim": 8},
            "statistics.json": {}, "embodiment_id.json": {}}
        for name, value in values.items():
            (self.base / name).write_text(json.dumps(value))
        state = {"action_head." + key: value for key, value in legacy.toy_head(None, "cpu").state_dict().items()}
        save_file(state, str(self.base / "model.safetensors"))
        (self.base / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {key: "model.safetensors" for key in state}}))
        self.manifest = {"model_path": str(self.base), "fingerprint": "a" * 64,
            "feature_dim": 6, "state_dim": 3, "action_dim": 8, "action_steps": 16,
            "splits": {"train": [0, 1], "val": list(range(2, 34))}}
        self.episodes = {}
        for eid in range(34):
            generator = torch.Generator().manual_seed(820 + eid)
            features = torch.randn(6, 236, 6, generator=generator).bfloat16()
            images = torch.zeros(6, 236, dtype=torch.bool)
            images[:, 10:91] = images[:, 98:179] = True
            no_demo = 2 <= eid < 18
            decisions = torch.zeros(5, dtype=torch.bool)
            decisions[0 if no_demo else 2] = True
            if eid < 2:
                decisions[2:] = True
            self.episodes[eid] = {"episode_id": eid, "cache_fingerprint": self.manifest["fingerprint"],
                "features": features, "image_masks": images, "attention_masks": torch.ones_like(images),
                "short": features[:, -2:].float().clone(), "moment": torch.randn(6, 2, 6, generator=generator),
                "state": torch.randn(6, 3, generator=generator),
                "frames": torch.arange(6) * 16,
                "is_demo": torch.tensor([not no_demo, not no_demo, False, False, False, False]),
                "embodiment_id": 0, "actions": torch.randn(5, 16, 8, generator=generator),
                "action_mask": decisions[:, None].expand(5, 16).clone(),
                "targets": torch.randn(5, 16, 8, generator=generator),
                "target_mask": decisions[:, None, None].expand(5, 16, 8).clone(),
                "transition_valid": torch.ones(5, dtype=torch.bool), "decision_mask": decisions}
        self.cache = SimpleNamespace(path=self.root / "cache", manifest=self.manifest, load=lambda eid: self.episodes[eid])
        cfg = MemoryV7Config(feature_dim=6, state_dim=3, num_short_tokens=2, hidden_dim=8,
                            num_heads=2, capacity=4, time_scale=2.)
        with isolated_seed(104, "cpu"):
            parent, cvom = RecurrentMemoryV7(cfg), CVOMV7(cfg)
        head = legacy.toy_head(None, "cpu")
        expert = LoRAConfig(2, 4.)
        targets = install_expert_lora(head, expert)
        with torch.no_grad():
            for name, parameter in head.named_parameters():
                if name.endswith("lora_B"):
                    parameter.fill_(.002)
        self.initial = save_checkpoint_v7(self.root / "archive", 1250, parent, head, cvom, None,
            {"trainer_variant": "recurrent_memory_v7", "stage": 1, "mode": "archive", "memory": asdict(cfg),
             "expert": asdict(expert), "expert_targets": targets},
            {"base_model": checkpoint_identity(self.base), "cache_fingerprint": self.manifest["fingerprint"]})
        self.visual_cfg = VisualDifferentialConfig(feature_dim=6, hidden_dim=16, num_heads=4,
                                                   num_short_tokens=2, time_scale=2.)
        self.stack.enter_context(patch.object(trainer, "EpisodeCache", return_value=self.cache))
        self.stack.enter_context(patch.object(trainer, "validate_cache_checkpoint"))
        self.loader = self.stack.enter_context(patch.object(trainer, "actual_head", side_effect=legacy.toy_head))
        self.stack.enter_context(patch.object(trainer, "expert_episode_flow_loss", side_effect=toy_flow))
        self.stack.enter_context(patch.object(trainer, "generated_action", side_effect=toy_generated))
        self.stack.enter_context(patch.object(trainer, "visual_config", return_value=self.visual_cfg))
        class SilentPlots(trainer.RunLogger):
            def plot(self):
                pass
        self.stack.enter_context(patch.object(trainer, "RunLogger", SilentPlots))
        # Other agents are preparing NEW sources concurrently. This test binds
        # a synthetic immutable source identity, not a real-run provenance proof.
        self.stack.enter_context(patch.object(trainer, "source_identity", return_value={"synthetic_test_only": "c" * 64}))
        self.stack.enter_context(patch.object(trainer, "DemoTailSidecar", TinySidecarReader))
        self.stack.enter_context(patch.object(sidecar_contract, "DemoTailSidecar", TinySidecarReader))
        self.sidecar = self.root / "sidecar"
        self.make_sidecar()
        self.reference = self.root / "reference_v11"
        self.make_reference()

    def make_sidecar(self):
        self.sidecar.mkdir()
        records = []
        for eid in range(34):
            n_demo = 0 if 2 <= eid < 18 else 32
            frames = [] if n_demo == 0 else list(range(17, 32))
            generator = torch.Generator().manual_seed(202 + eid)
            payload = {"episode_id": eid, "cache_fingerprint": self.manifest["fingerprint"],
                "sidecar_fingerprint": "b" * 64, "frames": torch.tensor(frames, dtype=torch.long),
                "is_demo": torch.ones(len(frames), dtype=torch.bool),
                "images": torch.randn(len(frames), 2, 81, 6, generator=generator).bfloat16()}
            path = self.sidecar / f"episode-{eid}.pt"
            torch.save(payload, path)
            records.append({"episode_id": eid, "n_demo": n_demo, "last_canonical_demo": -1 if not n_demo else 16,
                            "frames": frames, "path": path.name, "payload_sha256": trainer.file_hash(path)})
        manifest = {"driver_variant": "demo_tail_inventory_v13", "status": "complete", "fingerprint": "b" * 64,
            "episodes": records, "plan": {"scope": "inventory_train_val", "rule": bundles.EXTRACTION_RULE,
                "cache_fingerprint": self.manifest["fingerprint"], "selection": {"test_partition": False,
                    "splits": ["train", "val"]}},
            "inventory_checks": {"zero_action_calls": True, "original_short_memory_unchanged": True,
                                 "frozen_model_content_unchanged": True}}
        (self.sidecar / "manifest.json").write_text(json.dumps(manifest))

    def make_reference(self):
        args = trainer.parse_args(self.args("reference_args"))
        plan, sha = trainer.build_plan(args, self.cache, trainer.MappedEpisodes(self.cache))
        train = vars(args).copy()
        train.update(driver_variant="visual_patch_v11", mode="visual_patch")
        config = {"trainer_variant": "visual_patch_v11", "driver_variant": "visual_patch_v11", "stage": 1,
            "mode": "visual_patch", "visual": asdict(self.visual_cfg), "camera_order": list(trainer.CAMERA_ORDER), "train": train}
        metadata = {"base_model": checkpoint_identity(self.base), "cache_fingerprint": self.manifest["fingerprint"],
            "frozen_parent": trainer.parent_reference(self.base, self.initial), "plan_sha256": sha,
            "source_sha256": {"synthetic_reference": "d" * 64}, "runtime": {"torch": torch.__version__}}
        with isolated_seed(1984, "cpu"):
            visual = VisualPatchMemoryV11(VisualPatchConfig(**asdict(self.visual_cfg)))
        reference_checkpoint.save_checkpoint(self.reference, 0, visual, None, config, metadata,
            training_state={"driver_variant": "visual_patch_v11", "window_cursor": 0, "plan_sha256": sha})
        (self.reference / "run_config.json").write_text(json.dumps(config))
        (self.reference / "query_plan.json").write_text(json.dumps({"sha256": sha, **plan}))

    def args(self, name):
        return ["--cache-dir", str(self.cache.path), "--sidecar-dir", str(self.sidecar),
            "--init-checkpoint", str(self.initial), "--reference-run", str(self.reference),
            "--output-dir", str(self.root / name), "--device", "cpu", "--max-steps", "4",
            "--query-batch-size", "3", "--val-samples", "32", "--val-noise-samples", "2",
            "--eval-steps", "2", "--save-steps", "2", "--plot-steps", "2", "--log-steps", "1"]

    def run_train(self, argv):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return trainer.main(argv)

    def checkpoint(self, name, step):
        return self.root / name / f"checkpoint-{step:06d}"

    def test_readonly_preflight_scope_and_reject_other_arm(self):
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("No CUDA")), \
                patch.object(torch.cuda, "get_rng_state_all", side_effect=AssertionError("No CUDA RNG")):
            self.assertEqual(self.run_train(self.args("preflight") + ["--preflight-only"]), 0)
        self.loader.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())
        with self.assertRaisesRegex(ValueError, "include_tail=True"):
            self.run_train(self.args("bad_arm") + ["--no-include-tail", "--preflight-only"])
        for value in ("nan", "0", "-1"):
            with self.assertRaises(ValueError):
                self.run_train(self.args("bad_lr") + ["--expert-learning-rate", value, "--preflight-only"])

    def test_joint_gradients_both_payloads_and_originals_unchanged(self):
        snapshots = {str(p): trainer.file_hash(p) for directory in (self.initial, self.base, self.sidecar)
                     for p in directory.iterdir() if p.is_file()}
        original_scope, gradients = trainer.assert_scope, []
        def scope(visual, parent, head, cvom, guard):
            original_scope(visual, parent, head, cvom, guard)
            named = dict(visual.named_parameters())
            expert = list(trainer.expert_parameters(head))
            self.assertEqual(len(named), 14)
            self.assertTrue(all(p.requires_grad for p in expert))
            self.assertTrue(all(p.grad is None for module in (parent, cvom) for p in module.parameters()))
            if any(p.grad is not None for p in named.values()):
                gradients.append((all(p.grad is not None and float(p.grad.norm()) > 0 for p in named.values()),
                                  all(p.grad is not None and float(p.grad.norm()) > 0 for p in expert)))
        with patch.object(trainer, "assert_scope", side_effect=scope):
            self.assertEqual(self.run_train(self.args("joint")), 0)
        begin = json.loads((self.root / "joint/validation-000000.json").read_text())
        end = json.loads((self.root / "joint/validation-000004.json").read_text())
        self.assertEqual(len(begin["records"]), 128)
        for key in ("generated_observed_prefix_mae", "flow_action_loss"):
            self.assertEqual(begin["summary"]["reader"][key], begin["summary"]["visual-off"][key])
        self.assertNotEqual(begin["summary"]["visual-off"]["flow_action_loss"],
                            end["summary"]["visual-off"]["flow_action_loss"])
        zero, final = self.checkpoint("joint", 0), self.checkpoint("joint", 4)
        original = load_file(str(self.initial / "expert.safetensors"))
        initial = load_file(str(zero / "expert.safetensors"))
        adapted = load_file(str(final / "expert.safetensors"))
        self.assertTrue(all(torch.equal(initial[n], original[n]) for n in original))
        self.assertTrue(all(not torch.equal(initial[n], adapted[n]) for n in original))
        v0, v4 = [load_file(str(p / "visual.safetensors")) for p in (zero, final)]
        self.assertTrue(all(not torch.equal(v0[n], v4[n]) for n in v0))
        self.assertIn((True, True), gradients)
        info = bundles.checkpoint_info(self.base, final)
        self.assertNotIn("frozen_parent", info["metadata"])
        self.assertEqual(info["config"]["objective"]["trainable_scope"], "visual_and_expert_lora")
        saved = torch.load(final / "training_state.pt", weights_only=True)
        names = [n for group in saved["optimizer_param_names"] for n in group]
        self.assertEqual(len(names), 14 + len(original))
        self.assertEqual(len(saved["optimizer"]["state"]), len(names))
        self.assertEqual(snapshots, {p: trainer.file_hash(Path(p)) for p in snapshots})

    def test_exact_pause_resume_two_payloads_optimizer_rng(self):
        self.run_train(self.args("full"))
        self.run_train(self.args("part") + ["--stop-after-steps", "1"])
        partial = self.checkpoint("part", 1)
        self.assertFalse((partial.parent / "validation-000001.json").exists())
        self.assertFalse(json.loads((partial.parent / "rollout_checkpoint.json").read_text())["ready"])
        self.run_train(["--resume", str(partial), "--output-dir", str(self.root / "resume")])
        a, b = self.checkpoint("full", 4), self.checkpoint("resume", 4)
        for filename in ("visual.safetensors", "expert.safetensors"):
            sa, sb = [load_file(str(p / filename)) for p in (a, b)]
            for name in sa:
                torch.testing.assert_close(sa[name], sb[name], rtol=0, atol=0)
        sa, sb = [torch.load(p / "training_state.pt", weights_only=True) for p in (a, b)]
        self.assertEqual(sa["extra"], sb["extra"])
        self.assertEqual(sa["optimizer_param_names"], sb["optimizer_param_names"])
        self.assertEqual(sa["optimizer"]["param_groups"], sb["optimizer"]["param_groups"])
        for index, values in sa["optimizer"]["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(value, sb["optimizer"]["state"][index][name], rtol=0, atol=0)
        self.assertEqual(sa["rng"]["python"], sb["rng"]["python"])
        self.assertEqual(sa["rng"]["numpy"], sb["rng"]["numpy"])
        self.assertTrue(torch.equal(sa["rng"]["torch"], sb["rng"]["torch"]))
        for flags in (["--expert-learning-rate", "0.0002"], ["--seed", "1"], ["--max-steps", "5"]):
            with self.assertRaisesRegex(ValueError, "Exact resume option"):
                self.run_train(["--resume", str(partial), "--output-dir", str(self.root / "bad"),
                                *flags, "--preflight-only"])
        with self.assertRaisesRegex(FileExistsError, "NEW output"):
            self.run_train(["--resume", str(partial), "--output-dir", str(partial.parent), "--preflight-only"])

    def test_frozen_original_mutation_and_disabled_lora_rejected(self):
        original_scope, tested = trainer.assert_scope, []
        def scope(visual, parent, head, cvom, guard):
            original_scope(visual, parent, head, cvom, guard)
            if tested:
                return
            tested.append(True)
            outside = next(p for p in head.parameters() if not p.requires_grad)
            outside.requires_grad_(True)
            with self.assertRaisesRegex(RuntimeError, "Original Action Expert"):
                original_scope(visual, parent, head, cvom, guard)
            outside.requires_grad_(False)
            adapter = next(m for m in head.modules() if isinstance(m, trainer.ExpertLoRALinear))
            adapter.enabled = False
            with self.assertRaisesRegex(RuntimeError, "adapters must be enabled"):
                original_scope(visual, parent, head, cvom, guard)
            adapter.enabled = True
            with torch.no_grad():
                outside.add_(0)
            with self.assertRaisesRegex(RuntimeError, "tensor changed"):
                original_scope(visual, parent, head, cvom, guard)
            raise RuntimeError("Guard checks finished")
        with patch.object(trainer, "assert_scope", side_effect=scope), self.assertRaisesRegex(RuntimeError, "Guard checks finished"):
            self.run_train(self.args("guard"))
        self.assertTrue(tested)
        self.assertFalse((self.root / "guard/checkpoint-000000").exists())

    def test_final_rollout_selection_not_offline_best(self):
        calls = []
        def validation(*_args):
            index = len(calls); calls.append(index)
            rows = [{"role": role, "episode_id": 2, "decision": 0, "repeat": repeat,
                "flow_seed": 1 + repeat, "generation_seed": 2 + repeat,
                "generated_observed_prefix_mae": 1. + index if role == "reader" else .5 + index,
                "generated_observed_prefix_mse": 1., "flow_action_loss": 1.}
                for role in ("reader", "visual-off") for repeat in (0, 1)]
            return trainer.validation_summaries(rows), rows
        with patch.object(trainer, "validate", side_effect=validation):
            self.run_train(self.args("selection"))
        best = json.loads((self.root / "selection/best_checkpoint.json").read_text())
        rollout = json.loads((self.root / "selection/rollout_checkpoint.json").read_text())
        self.assertEqual(best["step"], 0)
        self.assertTrue(rollout["ready"] and rollout["mae_best_is_diagnostic"])
        self.assertEqual(Path(rollout["path"]), self.checkpoint("selection", 4))

    def test_failure_preserves_originals_and_complete_zero_bundle(self):
        def fail(*args, **kwargs):
            if torch.is_grad_enabled():
                raise RuntimeError("Synthetic training failure")
            return toy_flow(*args, **kwargs)
        with patch.object(trainer, "expert_episode_flow_loss", side_effect=fail), self.assertRaisesRegex(RuntimeError, "Synthetic"):
            self.run_train(self.args("failed"))
        self.assertTrue(self.checkpoint("failed", 0).is_dir())
        self.assertFalse(self.checkpoint("failed", 1).exists())
        self.assertFalse((self.root / "failed/rollout_checkpoint.json").exists())


if __name__ == "__main__":
    unittest.main()
