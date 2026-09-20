"""CPU integration: real V12 replay/optimizer/bundles, tiny frozen AE only."""
from contextlib import redirect_stdout
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from safetensors.torch import load_file
import torch

from gr00t.long_memory.checkpoint_v7 import save_checkpoint_v7, v7_checkpoint_info
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora
from gr00t.long_memory.hamlet import checkpoint_identity, isolated_seed
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from run_scripts.robomme import train_visual_differential_v12 as trainer
from run_scripts.robomme.visual_differential_memory_v12 import VisualDifferentialConfig
from run_scripts.robomme import checkpoint_visual_patch_v11 as reference_checkpoint
from run_scripts.robomme.visual_patch_memory_v11 import VisualPatchConfig, VisualPatchMemoryV11
from tests import test_long_memory_v4_trainer as fixture


def toy_head(path, device):
    head = fixture.toy_base(path, device)[0].action_head
    head.num_inference_timesteps = 4
    return head


def toy_flow(head, episode, query, fused_short=None, *, seed, activation_checkpointing=False):
    features = episode["features"][query][None]
    if fused_short is not None:
        features = torch.cat((features[:, :-2], fused_short.to(features)), 1)
    noise = torch.rand((), generator=torch.Generator().manual_seed(seed))
    error = head(features) - episode["targets"][query].mean() - noise
    return {"loss": error.square(), "velocity_mae": error.abs()}


@torch.no_grad()
def toy_generated(head, episode, query, fused_short=None, *, seed):
    features = episode["features"][query][None]
    if fused_short is not None:
        features = torch.cat((features[:, :-2], fused_short.to(features)), 1)
    return head(features).expand(1, 16, 8) + torch.randn((1, 16, 8), generator=torch.Generator().manual_seed(seed)) * .1


class VisualDifferentialTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        fixture.TestV4Trainer.setUp(self)
        self.cache.path = self.root / "cache"
        self.manifest.update(action_steps=16, action_dim=8)
        self.manifest["splits"]["val"] = list(range(2, 34))
        for eid in range(34):
            generator = torch.Generator().manual_seed(820 + eid)
            features = torch.randn(6, 236, 6, generator=generator)
            images = torch.zeros(6, 236, dtype=torch.bool)
            images[:, 10:91] = True
            images[:, 98:179] = True
            decisions = torch.ones(5, dtype=torch.bool)
            # Every validation query has genuinely empty history. It must NOT
            # be dropped like TRAIN q0, and its visual branch stays identical.
            if eid >= 2:
                decisions[1:] = False
            actions = torch.randn(5, 16, 8, generator=generator)
            actions[~decisions] = 0
            self.episodes[eid] = {
                "episode_id": eid, "cache_fingerprint": self.manifest["fingerprint"],
                "features": features, "image_masks": images, "attention_masks": torch.ones_like(images),
                "short": features[:, -2:].clone(), "moment": torch.randn(6, 2, 6, generator=generator),
                "state": torch.randn(6, 3, generator=generator), "frames": torch.arange(6) * 16,
                "is_demo": torch.zeros(6, dtype=torch.bool), "embodiment_id": 0,
                "actions": actions, "action_mask": decisions[:, None].expand(5, 16).clone(),
                "targets": torch.randn(5, 16, 8, generator=generator),
                "target_mask": decisions[:, None, None].expand(5, 16, 8).clone(),
                "transition_valid": torch.ones(5, dtype=torch.bool), "decision_mask": decisions}
        cfg = MemoryV7Config(feature_dim=6, state_dim=3, num_short_tokens=2, hidden_dim=8,
                             num_heads=2, capacity=4, time_scale=2.)
        with isolated_seed(104, "cpu"):
            parent, cvom = RecurrentMemoryV7(cfg), CVOMV7(cfg)
        head = toy_head(None, "cpu")
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
        self.stack.enter_context(patch.object(trainer, "EpisodeCache", return_value=self.cache))
        self.stack.enter_context(patch.object(trainer, "validate_cache_checkpoint"))
        self.loader = self.stack.enter_context(patch.object(trainer, "actual_head", side_effect=toy_head))
        self.flow = self.stack.enter_context(patch.object(trainer, "expert_episode_flow_loss", side_effect=toy_flow))
        self.generated = self.stack.enter_context(patch.object(trainer, "generated_action", side_effect=toy_generated))
        self.stack.enter_context(patch.object(trainer, "visual_config", return_value=VisualDifferentialConfig(
            feature_dim=6, hidden_dim=16, num_heads=4, num_short_tokens=2, time_scale=2.)))
        self.reference = self.root / "reference_v11"
        self.make_reference()

    def make_reference(self):
        # Genuine old-format reference, including immutable step-zero tensors.
        # Called once per fixture; tests changing synthetic data use a NEW ref.
        args = trainer.parse_args(self.args("reference_args"))
        plan, sha = trainer.build_plan(args, self.cache, trainer.MappedEpisodes(self.cache))
        cfg = asdict(trainer.visual_config(None))
        train = vars(args).copy()
        train.update(driver_variant="visual_patch_v11", mode="visual_patch")
        config = {"trainer_variant": "visual_patch_v11", "driver_variant": "visual_patch_v11",
                  "stage": 1, "mode": "visual_patch", "visual": cfg,
                  "camera_order": list(trainer.CAMERA_ORDER), "train": train}
        meta = {"base_model": checkpoint_identity(self.base), "cache_fingerprint": self.manifest["fingerprint"],
                "frozen_parent": trainer.parent_reference(self.base, self.initial), "plan_sha256": sha,
                "source_sha256": {"fixture.py": "a" * 64}, "runtime": {"torch": torch.__version__}}
        with isolated_seed(1984, "cpu"):
            visual = VisualPatchMemoryV11(VisualPatchConfig(**cfg))
        reference_checkpoint.save_checkpoint(self.reference, 0, visual, None, config, meta,
            training_state={"driver_variant": "visual_patch_v11", "window_cursor": 0, "plan_sha256": sha})
        (self.reference / "run_config.json").write_text(json.dumps(config))
        (self.reference / "query_plan.json").write_text(json.dumps({"sha256": sha, **plan}))

    def args(self, name):
        return ["--cache-dir", str(self.cache.path), "--init-checkpoint", str(self.initial),
                "--reference-run", str(self.reference),
                "--output-dir", str(self.root / name), "--device", "cpu", "--max-steps", "4",
                "--query-batch-size", "3", "--val-samples", "32", "--val-noise-samples", "1",
                "--eval-steps", "2", "--save-steps", "2", "--plot-steps", "2", "--log-steps", "1"]

    def run_train(self, argv):
        with redirect_stdout(io.StringIO()):
            return trainer.main(argv)

    def checkpoint(self, name, step):
        return self.root / name / f"checkpoint-{step:06d}"

    def test_preflight_readonly_structural_filter_and_validation_unchanged(self):
        self.assertEqual(self.run_train(self.args("preflight") + ["--preflight-only"]), 0)
        self.loader.assert_not_called()
        self.assertFalse((self.root / "preflight").exists())
        args = trainer.parse_args(self.args("unused"))
        plan, digest = trainer.build_plan(args, self.cache, trainer.MappedEpisodes(self.cache))
        self.assertEqual(plan["train_eligibility"], {
            "policy": trainer.FILTER, "excluded": [{"episode_id": 0, "decision": 0}, {"episode_id": 1, "decision": 0}],
            "excluded_count": 2, "before_count": 10, "after_count": 8, "validation_unchanged": True})
        self.assertEqual(plan["train_query_count"], 8)
        self.assertTrue(all(query == 0 for _, query in plan["validation"]))
        self.assertTrue(all(bool(self.episodes[eid]["decision_mask"][0]) for eid in range(34)))
        self.assertEqual(digest, trainer.digest(plan))
        with self.assertRaisesRegex(ValueError, "inside a source"):
            self.run_train(self.args("unsafe") + ["--output-dir", str(self.initial / "unsafe"), "--preflight-only"])

    def test_val128_preserves_entire_train_schedule_and_first32_two_draws(self):
        self.manifest["splits"]["val"] = list(range(2, 130))
        for eid in range(34, 130):
            self.episodes[eid] = {**self.episodes[2], "episode_id": eid}
        args = trainer.parse_args(self.args("plan") + ["--val-noise-samples", "2", "--max-steps", "512"])
        original, _ = trainer.build_plan(args, self.cache, trainer.MappedEpisodes(self.cache))
        args.val_samples = 128
        expanded, _ = trainer.build_plan(args, self.cache, trainer.MappedEpisodes(self.cache))
        for key in ("train", "files", "windows", "schedule", "train_query_count", "train_eligibility"):
            self.assertEqual(original[key], expanded[key], key)
        self.assertEqual(len(expanded["validation"]), 128)
        self.assertEqual(original["validation"], expanded["validation"][:32])
        self.assertEqual(original["validation_schedule"], expanded["validation_schedule"][:64])
        args.max_steps = 4
        smoke, _ = trainer.build_plan(args, self.cache, trainer.MappedEpisodes(self.cache))
        self.assertEqual(smoke["schedule"], expanded["schedule"][:4])
        self.assertEqual(smoke["windows"], expanded["windows"][:4])
        self.assertEqual(smoke["validation_schedule"], expanded["validation_schedule"])

    def test_visual_only_real_replay_and_immutable_parent_control(self):
        before = {p.name: trainer.file_hash(p) for p in self.initial.iterdir()}
        self.assertEqual(self.run_train(self.args("full")), 0)
        start = load_file(str(self.checkpoint("full", 0) / "visual.safetensors"))
        end = load_file(str(self.checkpoint("full", 4) / "visual.safetensors"))
        self.assertFalse(bool(start["output_projection.weight"].any()))
        self.assertFalse(torch.equal(start["output_projection.weight"], end["output_projection.weight"]))
        self.assertFalse(torch.equal(start["image_projection.weight"], end["image_projection.weight"]))
        self.assertEqual(before, {p.name: trainer.file_hash(p) for p in self.initial.iterdir()})
        self.assertEqual({p.name for p in self.checkpoint("full", 4).iterdir()},
                         {"checkpoint.json", "visual.safetensors", "training_state.pt"})
        info = trainer.checkpoint_info(self.base, self.checkpoint("full", 4))
        self.assertFalse(info["self_contained"])
        self.assertEqual(info["metadata"]["frozen_parent"]["path"], str(self.initial))
        with self.assertRaises(ValueError):
            v7_checkpoint_info(self.base, self.checkpoint("full", 4))
        moved = self.reference.with_name("reference_temporarily_unavailable")
        self.reference.rename(moved)
        try:
            # Runtime deployment needs base+archive only, not the V11 init run.
            self.assertEqual(trainer.checkpoint_info(self.base, self.checkpoint("full", 4))["step"], 4)
        finally:
            moved.rename(self.reference)
        rows = [json.loads(line) for line in (self.root / "full/metrics.jsonl").read_text().splitlines()]
        train = [row for row in rows if row["split"] == "train"]
        self.assertTrue(all(row["visual_grad_norm"] > 0 and row["output_grad_norm"] > 0 for row in train))
        self.assertTrue(any(row["query_grad_norm"] > 0 for row in train))
        control = [row for row in rows if row["split"] == "comparison/visual-off"]
        self.assertEqual(len({row["generated_observed_prefix_mse"] for row in control}), 1)
        validation = json.loads((self.root / "full/validation-000004.json").read_text())
        self.assertEqual(validation["summary"]["reader"], validation["summary"]["visual-off"])
        self.assertEqual(len(validation["records"]), 64)
        self.assertEqual(validation["summary"]["reader"]["empty_history"], 1.)
        self.assertIn("generated_observed_joint7_mse", validation["summary"]["reader"])
        self.assertIn("generated_observed_gripper1_mse", validation["summary"]["reader"])

    def test_exact_pause1_resume4_weights_optimizer_rng_queries_and_no_extra_validation(self):
        self.run_train(self.args("full"))
        self.run_train(self.args("part") + ["--stop-after-steps", "1"])
        self.assertFalse((self.root / "part/validation-000001.json").exists())
        self.run_train(["--resume", str(self.checkpoint("part", 1)), "--output-dir", str(self.root / "resumed")])
        a = load_file(str(self.checkpoint("full", 4) / "visual.safetensors"))
        b = load_file(str(self.checkpoint("resumed", 4) / "visual.safetensors"))
        for name in a:
            torch.testing.assert_close(a[name], b[name], atol=0, rtol=0, msg=name)
        left = torch.load(self.checkpoint("full", 4) / "training_state.pt", weights_only=True)
        right = torch.load(self.checkpoint("resumed", 4) / "training_state.pt", weights_only=True)
        self.assertEqual(left["extra"], right["extra"])
        self.assertEqual(left["rng"]["python"], right["rng"]["python"])
        self.assertTrue(torch.equal(left["rng"]["torch"], right["rng"]["torch"]))
        self.assertEqual(left["optimizer"]["param_groups"], right["optimizer"]["param_groups"])
        for index, values in left["optimizer"]["state"].items():
            for key, value in values.items():
                torch.testing.assert_close(value, right["optimizer"]["state"][index][key], rtol=0, atol=0)
        info = trainer.checkpoint_info(self.base, self.checkpoint("resumed", 4))
        plan = json.loads((self.root / "full/query_plan.json").read_text())
        self.assertEqual(info["metadata"]["train_state"]["processed_queries"], sum(w["query_count"] for w in plan["windows"]))
        for option, value in (("--max-steps", "5"), ("--seed", "3"), ("--visual-learning-rate", ".1")):
            with self.assertRaisesRegex(ValueError, "Exact resume option"):
                self.run_train(["--resume", str(self.checkpoint("part", 1)), "--output-dir", str(self.root / "bad"),
                                option, value, "--preflight-only"])

    def test_both_arms_same_initialization_plan_and_exact_pause2_resume4(self):
        initial_states, plans = [], []
        for mode in ("differential", "current_only"):
            full, part, resumed = mode + "_full", mode + "_part", mode + "_resumed"
            self.run_train(self.args(full) + ["--read-mode", mode])
            self.run_train(self.args(part) + ["--read-mode", mode, "--stop-after-steps", "2"])
            self.run_train(["--resume", str(self.checkpoint(part, 2)), "--output-dir", str(self.root / resumed)])
            initial_states.append(load_file(str(self.checkpoint(full, 0) / "visual.safetensors")))
            plans.append(json.loads((self.root / full / "query_plan.json").read_text()))
            left = load_file(str(self.checkpoint(full, 4) / "visual.safetensors"))
            right = load_file(str(self.checkpoint(resumed, 4) / "visual.safetensors"))
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0, msg=mode + "/" + name)
            a = torch.load(self.checkpoint(full, 4) / "training_state.pt", weights_only=True)
            b = torch.load(self.checkpoint(resumed, 4) / "training_state.pt", weights_only=True)
            self.assertEqual(a["optimizer"]["param_groups"], b["optimizer"]["param_groups"])
            for index, values in a["optimizer"]["state"].items():
                for key, value in values.items():
                    torch.testing.assert_close(value, b["optimizer"]["state"][index][key], rtol=0, atol=0)
            self.assertEqual(a["extra"], b["extra"])
            self.assertTrue(torch.equal(a["rng"]["torch"], b["rng"]["torch"]))
            self.assertEqual(a["rng"]["python"], b["rng"]["python"])
            info = trainer.checkpoint_info(self.base, self.checkpoint(resumed, 4))
            self.assertEqual(info["config"]["read_mode"], mode)
            self.assertEqual(info["metadata"]["read_mode"], mode)
            self.assertIsNone(info["config"]["train"]["init_checkpoint"])
            train = [json.loads(line) for line in (self.root / full / "metrics.jsonl").read_text().splitlines()
                     if json.loads(line)["split"] == "train"]
            self.assertTrue(any(row["query_grad_norm"] > 0 and row["key_grad_norm"] > 0
                                and row["value_grad_norm"] > 0 and row["image_encoder_grad_norm"] > 0 for row in train))
        self.assertEqual(plans[0], plans[1])
        reference = load_file(str(self.reference / "checkpoint-000000/visual.safetensors"))
        for name in reference:
            self.assertTrue(torch.equal(reference[name], initial_states[0][name]))
            self.assertTrue(torch.equal(reference[name], initial_states[1][name]))

    def test_mae_selection_is_predeclared_not_mse_and_query_weights_equal(self):
        # Synthetic validation deliberately makes MAE improve as MSE worsens.
        calls = []
        def observed(*_args):
            index = len(calls)
            calls.append(index)
            rows = []
            for role in ("reader", "visual-off"):
                for query in range(2):
                    # Unequal draw counts test query-first averaging itself.
                    for repeat in range(query + 1):
                        mae = (1. - .1 * index) if role == "reader" else 1.
                        mse = 1. + index if role == "reader" else 1.
                        rows.append({"role": role, "episode_id": query + 2, "decision": query,
                                     "repeat": repeat, "flow_seed": 1, "generation_seed": 2,
                                     "generated_observed_prefix_mae": mae + query,
                                     "generated_observed_prefix_mse": mse + query})
            return trainer.validation_summaries(rows), rows
        with patch.object(trainer, "validate", side_effect=observed):
            self.run_train(self.args("mae_selection"))
        self.assertEqual(json.loads((self.root / "mae_selection/best_checkpoint.json").read_text())["step"], 4)
        first = json.loads((self.root / "mae_selection/validation-000000.json").read_text())
        last = json.loads((self.root / "mae_selection/validation-000004.json").read_text())
        self.assertEqual(first["summary"]["reader"]["generated_observed_prefix_mae"], 1.5)
        self.assertEqual(last["selection_metric"], "val/generated_observed_prefix_mae")
        self.assertGreater(last["summary"]["reader"]["generated_observed_prefix_mse"],
                           first["summary"]["reader"]["generated_observed_prefix_mse"])
        self.assertEqual(last["summary"]["reader"]["query_prefix_mae_improved_fraction"], 1.)

    def test_reference_schedule_and_initialization_changed_fail_before_model_output(self):
        path = self.reference / "query_plan.json"
        saved = path.read_text()
        plan = json.loads(saved)
        plan["schedule"][0]["queries"][0]["flow_seed"] += 1
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "reference plan digest"):
            self.run_train(self.args("bad_reference") + ["--preflight-only"])
        path.write_text(saved)
        self.loader.assert_not_called()
        self.assertFalse((self.root / "bad_reference").exists())
        self.run_train(self.args("resume_ref") + ["--stop-after-steps", "2"])
        # Whitespace changes preserve JSON semantics but invalidate the exact
        # training provenance; resume must not silently accept another record.
        path.write_text(saved + "\n")
        with self.assertRaisesRegex(ValueError, "reference provenance"):
            self.run_train(["--resume", str(self.checkpoint("resume_ref", 2)),
                            "--output-dir", str(self.root / "bad_resume_ref"), "--preflight-only"])
        self.assertFalse((self.root / "bad_resume_ref").exists())
        path.write_text(saved)

    def test_scope_guard_catches_frozen_weight_or_trainability_mutation(self):
        visual = trainer.VisualDifferentialMemoryV12(trainer.visual_config(None))
        parent = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        head = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        cvom = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        guard = trainer.frozen_guard(parent, head, cvom)
        trainer.assert_scope(visual, parent, head, cvom, guard)
        head.weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "Only visual"):
            trainer.assert_scope(visual, parent, head, cvom, guard)
        head.weight.requires_grad_(False)
        with torch.no_grad(): head.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, "tensor changed"):
            trainer.assert_scope(visual, parent, head, cvom, guard)

    def test_nonempty_validation_keeps_frozen_parent_control_constant(self):
        for eid in self.manifest["splits"]["val"]:
            ep = self.episodes[eid]
            ep["decision_mask"][:] = True
            ep["action_mask"][:] = True
            ep["target_mask"][:] = True
        self.reference = self.root / "reference_nonempty"
        self.make_reference()
        self.run_train(self.args("nonempty"))
        before = json.loads((self.root / "nonempty/validation-000000.json").read_text())
        after = json.loads((self.root / "nonempty/validation-000004.json").read_text())
        self.assertEqual(before["summary"]["reader"], before["summary"]["visual-off"])
        self.assertEqual(before["summary"]["visual-off"], after["summary"]["visual-off"])
        self.assertGreater(after["summary"]["reader"]["image_changed_fraction"], 0)
        self.assertGreater(after["summary"]["reader"]["visual_bank_tokens"], 0)

    def test_corrupt_cursor_or_selection_rejected_in_readonly_resume_preflight(self):
        self.run_train(self.args("part") + ["--stop-after-steps", "1"])
        path = self.checkpoint("part", 1) / "checkpoint.json"
        saved = json.loads(path.read_text())
        mutations = [lambda value: value["metadata"]["train_state"].update(processed_queries=999),
                     lambda value: value["metadata"]["train_state"].update(window_cursor=0),
                     lambda value: value["config"]["objective"].update(generated_auxiliary_weight=.1),
                     lambda value: value["metadata"].update(selection_metric="task_success")]
        for index, mutation in enumerate(mutations):
            corrupted = copy.deepcopy(saved)
            mutation(corrupted)
            path.write_text(json.dumps(corrupted))
            output = self.root / f"corrupt{index}"
            with self.assertRaisesRegex(ValueError, "cursor|semantics"):
                self.run_train(["--resume", str(path.parent), "--output-dir", str(output), "--preflight-only"])
            self.assertFalse(output.exists())
        path.write_text(json.dumps(saved))

    def test_zero_history_training_fails_and_failure_preserves_checkpoint0(self):
        with self.assertRaisesRegex(ValueError, "No-history"):
            trainer.query_objective(None, None, None, None, {}, {"decision": 0})
        def fail(*args, **kwargs):
            if torch.is_grad_enabled():
                raise RuntimeError("simulated visual training failure")
            return toy_flow(*args, **kwargs)
        with patch.object(trainer, "expert_episode_flow_loss", side_effect=fail), self.assertRaisesRegex(RuntimeError, "simulated"):
            self.run_train(self.args("failed"))
        self.assertTrue(self.checkpoint("failed", 0).is_dir())
        self.assertFalse(self.checkpoint("failed", 1).exists())
        self.assertTrue((self.root / "failed/failure.json").is_file())
        self.assertFalse(list((self.root / "failed").glob(".checkpoint-*")))


if __name__ == "__main__":
    unittest.main()
