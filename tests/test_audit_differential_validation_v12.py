"""Synthetic CPU tests only; no actual running pilot/cache preflight or GPU."""
import argparse
from contextlib import redirect_stdout
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from run_scripts.robomme import audit_differential_validation_v12 as audit


def plan_fixture():
    args = argparse.Namespace(seed=9111, val_samples=128, val_noise_samples=2, max_steps=4, query_batch_size=2)
    cache = SimpleNamespace(path=Path("/unused"), manifest={"splits": {"train": [0, 1], "val": list(range(2, 132))},
                                                        "action_steps": 16, "episodes": []})
    def fetch(eid):
        mask = torch.tensor([True, eid < 2])
        return {"decision_mask": mask, "target_mask": mask[:, None, None].expand(2, 16, 8),
                "action_mask": mask[:, None].expand(2, 16)}
    episodes = SimpleNamespace(fetch=fetch)
    plan, sha = audit.trainer.build_plan(args, cache, episodes)
    return args, cache, episodes, {"sha256": sha, **plan}


def three_role_fixture():
    plan = {"validation": [[7, 0], [8, 1]], "validation_schedule": []}
    records = []
    for eid, query in plan["validation"]:
        for repeat in range(2):
            item = {"episode_id": eid, "decision": query, "repeat": repeat,
                    "flow_seed": eid * 10 + repeat, "generation_seed": eid * 100 + repeat}
            plan["validation_schedule"].append(item)
            for role in audit.ROLES:
                value = 0. if query == 0 else {"differential": 1., "current-only": 2., "visual-off": 3.}[role]
                row = {**item, "role": role, "generated_valid_values": 128., **{metric: value for metric in audit.METRICS}}
                # Deliberately opposite ranking to MAE, catching mislabeled
                # use of the V11 statistics helper's required primary field.
                row["generated_observed_prefix_mse"] = 0. if query == 0 else {
                    "differential": 4., "current-only": 1., "visual-off": 2.}[role]
                records.append(row)
    return plan, records


def write_run(root, mode="differential", best_step=0):
    def save(name, value):
        (root / name).write_text(json.dumps(value))
    args = audit.trainer.parse_args(["--cache-dir", "/unused/cache", "--output-dir", str(root),
        "--init-checkpoint", "/unused/parent", "--read-mode", mode])
    config = {"trainer_variant": audit.trainer.DRIVER, "read_mode": mode, "objective": audit.trainer.OBJECTIVE,
              "train": vars(args), "visual": asdict(audit.trainer.VisualDifferentialConfig())}
    plain = {"fixture_only": True}
    plan = {"sha256": audit.trainer.digest(plain), **plain}
    save("run_config.json", config); save("query_plan.json", plan)
    scores = {step: .2 + step / 10000 for step in range(0, 513, 128)}
    scores[best_step] = .1
    state = {"status": "complete", "optimizer_updates": 512, "window_cursor": 512, "processed_queries": 2048,
             "best_checkpoint": str(root / f"checkpoint-{best_step:06d}"), "best_generated_prefix_mae": .1}
    save("status.json", state)
    save("best_checkpoint.json", {"path": f"checkpoint-{best_step:06d}", "step": best_step})
    save("last_checkpoint.json", {"path": str(root / "checkpoint-000512"), "step": 512})
    for step, score in scores.items():
        save(f"validation-{step:06d}.json", {"step": step, "plan_sha256": plan["sha256"],
            "selection_metric": audit.trainer.SELECTION, "read_mode": mode,
            "summary": {"reader": {audit.PRIMARY: score}}})
    return save


class DifferentialValidationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_expand128_keeps256_draws_training_and_all_complement_q0(self):
        args, cache, episodes, saved = plan_fixture()
        before = copy.deepcopy(saved)
        expanded, sha, extra = audit.expanded_plan(args, cache, episodes, saved)
        self.assertEqual(saved, before)
        self.assertEqual(args.val_samples, 128)
        self.assertEqual(expanded["validation"][:128], saved["validation"])
        self.assertEqual(expanded["validation_schedule"][:256], saved["validation_schedule"])
        self.assertEqual(expanded["schedule"], saved["schedule"])
        self.assertEqual(sha, audit.trainer.digest(expanded))
        self.assertEqual(len(extra["validation"]), 2)
        self.assertEqual(len(extra["validation_schedule"]), 4)
        self.assertTrue(all(q == 0 for _, q in extra["validation"]))
        self.assertFalse({eid for eid, _ in extra["validation"]} & {eid for eid, _ in saved["validation"]})

    def test_reject_changed_seed_saved_noise_or_training_structure(self):
        args, cache, episodes, saved = plan_fixture()
        changed_args = copy.deepcopy(args); changed_args.seed += 1
        with self.assertRaisesRegex(ValueError, "seed9111"):
            audit.expanded_plan(changed_args, cache, episodes, saved)
        for field in ("validation_schedule", "schedule"):
            changed = copy.deepcopy(saved)
            if field == "validation_schedule": changed[field][0]["flow_seed"] += 1
            else: changed[field][0]["queries"][0]["flow_seed"] += 1
            changed["sha256"] = audit.trainer.digest({k: v for k, v in changed.items() if k != "sha256"})
            with self.assertRaisesRegex(ValueError, "rebuild exactly"):
                audit.expanded_plan(args, cache, episodes, changed)

    def test_completed_runs_accept_honest_step0_or_later_mae_best(self):
        for mode, step in (("differential", 0), ("current_only", 256)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_run(root, mode, step)
                record = audit.completed_run(root, mode)
                self.assertEqual(record["best_step"], step)
                self.assertEqual(record["args"].read_mode, mode)
                self.assertEqual(len(record["selection_scores"]), 5)

    def test_incomplete_mode_or_false_best_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save = write_run(root)
            with self.assertRaisesRegex(ValueError, "arm"):
                audit.completed_run(root, "current_only")
            state = json.loads((root / "status.json").read_text())
            state["status"] = "training"; save("status.json", state)
            with self.assertRaisesRegex(ValueError, "complete512"):
                audit.completed_run(root, "differential")
            state["status"] = "complete"
            state["best_checkpoint"] = str(root / "checkpoint-000128")
            state["best_generated_prefix_mae"] = .2128
            save("status.json", state)
            save("best_checkpoint.json", {"path": "checkpoint-000128", "step": 128})
            with self.assertRaisesRegex(ValueError, "MAE-selected"):
                audit.completed_run(root, "differential")

    def test_preflight_never_inspects_real_cache_when_either_run_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save = write_run(root)
            save("status.json", {"status": "training"})
            args = argparse.Namespace(differential_run=str(root), current_only_run=str(root))
            with patch.object(audit.trainer, "EpisodeCache") as cache:
                with self.assertRaisesRegex(ValueError, "Both runs"):
                    audit.preflight(args)
                cache.assert_not_called()

    def test_boundary_bundles_verify_actual_initial_tensors_and_final_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save = write_run(root)
            run = audit.completed_run(root, "differential")
            metadata = {key: "same" for key in ("frozen_parent", "base_model", "source_sha256", "runtime", "cache_fingerprint")}
            metadata.update(plan_sha256=run["plan"]["sha256"], initialization_reference={"path": "/fixture/reference"},
                            train_state=run["status"])
            run["info"] = {"metadata": metadata}
            def info(base, path):
                return {"step": int(path.name.split("-")[1]), "config": run["config"], "metadata": copy.deepcopy(metadata)}
            with patch.object(audit.trainer, "checkpoint_info", side_effect=info), \
                    patch.object(audit.trainer, "validate_resume_state") as resume, \
                    patch.object(audit.trainer, "load_file", return_value={"image_projection.weight": torch.ones(2, 2)}):
                files, checks = audit.verify_run_boundaries("/fixture/base", run)
                self.assertEqual(len(files), 7)
                self.assertTrue(all(checks.values()))
                resume.assert_called_once()
            def different(path, device):
                return {"image_projection.weight": torch.ones(2, 2) * (2 if path.startswith(str(root)) else 1)}
            with patch.object(audit.trainer, "checkpoint_info", side_effect=info), \
                    patch.object(audit.trainer, "validate_resume_state"), \
                    patch.object(audit.trainer, "load_file", side_effect=different):
                with self.assertRaisesRegex(ValueError, "Actual V12 step0"):
                    audit.verify_run_boundaries("/fixture/base", run)
            save("last_checkpoint.json", {"path": str(root / "checkpoint-000384"), "step": 384})
            with self.assertRaisesRegex(ValueError, "completed512"):
                audit.verify_run_boundaries("/fixture/base", run)

    def test_pair_requires_same_parent_initialization_plan_and_options(self):
        metadata = {key: "same" for key in ("frozen_parent", "base_model", "initialization_reference",
            "source_sha256", "runtime", "plan_sha256", "cache_fingerprint")}
        runs = {mode: {"args": argparse.Namespace(read_mode=mode, output_dir=mode, seed=9111),
                       "plan": {"sha256": "same"}, "info": {"metadata": copy.deepcopy(metadata)}} for mode in audit.MODES}
        audit.validate_pair_identity(runs)
        for field in metadata:
            changed = copy.deepcopy(runs); changed["current_only"]["info"]["metadata"][field] = "changed"
            with self.assertRaisesRegex(ValueError, field):
                audit.validate_pair_identity(changed)
        changed = copy.deepcopy(runs); changed["current_only"]["args"].seed += 1
        with self.assertRaisesRegex(ValueError, "beyond"):
            audit.validate_pair_identity(changed)

    def test_both_native_off_copies_exact_then_collapse_three_roles(self):
        plan, records = three_role_fixture()
        by_mode = {mode: [{**row, "role": "reader" if row["role"] == audit.ROLE[mode] else "visual-off"}
                          for row in records if row["role"] in (audit.ROLE[mode], "visual-off")] for mode in audit.MODES}
        merged, proofs = audit.merge_query_rows(by_mode, plan["validation_schedule"])
        self.assertEqual(merged, records)
        self.assertEqual(len(proofs), 4)
        self.assertTrue(all(row["off_copies_exact"] for row in proofs))
        broken = copy.deepcopy(by_mode)
        next(row for row in broken["current_only"] if row["role"] == "visual-off")[audit.PRIMARY] += 1e-12
        with self.assertRaisesRegex(ValueError, "not exactly equal"):
            audit.merge_query_rows(broken, plan["validation_schedule"])
        broken = copy.deepcopy(by_mode); broken["differential"][0][audit.PRIMARY] += .1
        with self.assertRaisesRegex(ValueError, "q=0"):
            audit.merge_query_rows(broken, plan["validation_schedule"])

    def test_metric_role_adapter_mae_primary_correct_medians_counts_loo_no_mutation(self):
        plan, records = three_role_fixture()
        before = copy.deepcopy(records)
        constants = (audit.stats_v11.METRICS, audit.stats_v11.ROLES)
        rng = torch.get_rng_state().clone()
        result = audit.paired_statistics(records, plan, bootstrap_draws=40)
        self.assertEqual(result, audit.paired_statistics(records, plan, bootstrap_draws=40))
        self.assertEqual(records, before)
        self.assertEqual(constants, (audit.stats_v11.METRICS, audit.stats_v11.ROLES))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(result["primary_metric"], audit.PRIMARY)
        pair = result["comparisons"]["differential_vs_current-only"]["metrics"]
        mae, mse = pair[audit.PRIMARY], pair["generated_observed_prefix_mse"]
        self.assertEqual(mae["equal_query_mean"], {"differential": .5, "current-only": 1.})
        self.assertEqual(mae["median_left_minus_right"], -.5)
        self.assertEqual((mae["improved"], mae["worse"], mae["tied"]), (1, 0, 1))
        self.assertEqual(mae["relative_mean_gain_percent"], 50.)
        self.assertEqual(mae["leave_one_out_delta_range"], [-1., 0.])
        self.assertEqual(mse["mean_left_minus_right"], 1.5)
        self.assertEqual(len(result["comparisons"]), 3)

    def test_zero_ties_missing_noise_missing_role_or_metric_fail_closed(self):
        plan, records = three_role_fixture()
        for row in records:
            for metric in audit.METRICS: row[metric] = 0.
        result = audit.paired_statistics(records, plan, bootstrap_draws=20)
        for comparison in result["comparisons"].values():
            metric = comparison["metrics"][audit.PRIMARY]
            self.assertEqual(metric["tied"], 2)
            self.assertIsNone(metric["relative_mean_gain_percent"])
            self.assertIsNone(metric["paired_query_bootstrap"]["relative_mean_gain_percent_95pct_percentile_interval"])
        for change in ("noise", "role", "metric", "missing"):
            broken = copy.deepcopy(records)
            if change == "noise": broken[0]["generation_seed"] += 1
            elif change == "role": broken[0]["role"] = "unknown"
            elif change == "metric": del broken[0][audit.PRIMARY]
            else: broken.pop()
            with self.assertRaises(ValueError):
                audit.paired_statistics(broken, plan, bootstrap_draws=20)

    def test_frozen_scope_versions_and_preflight_main_no_output_or_model(self):
        module = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        with self.assertRaises(RuntimeError): audit.stats_v11.frozen_guard((module,))
        with torch.no_grad():
            guard = audit.stats_v11.frozen_guard((module,))
            module.weight.add_(1)
            self.assertTrue(any(value._version != version for value, version in guard))
            module.requires_grad_(True)
            with self.assertRaises(RuntimeError): audit.stats_v11.frozen_guard((module,))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "never_created"
            payload = ({}, None, None, {"validation": [[1, 0]], "validation_schedule": [{}, {}]},
                       {"audit_id": "a", "selected_bests": {}, "q0_retained": 1})
            with patch.object(audit, "preflight", return_value=payload), patch.object(audit, "run_validation") as run:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(audit.main(["--differential-run", directory, "--current-only-run", directory,
                        "--output-dir", str(output), "--preflight-only"]), 0)
                run.assert_not_called()
            self.assertFalse(output.exists())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
