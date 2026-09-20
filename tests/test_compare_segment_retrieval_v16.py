"""CPU tests of fixed, episode-paired reporting; no policy/model loading."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch
from gr00t.long_memory.safety_v5 import validate_output_scope

from run_scripts.robomme.compare_segment_retrieval_v16 import (
    compare_rows, paired_rows, protected_roots, validate_complete_val, validate_fixed_study,
    validate_tensor_layout,
)


def rows(correct=1.5, permuted=1.5, episodes=12):
    return [{"episode_id": eid, "decision": q, "mode": mode,
             "query_frame": 80, "candidate_frames": [0, 16, 32, 64],
             "positive_frames": [16], "original_positive_frames": [16],
             "destination_to_source_frames": [16, 32, 0, 64] if mode == "content_permuted" else [0, 16, 32, 64],
             "old_positive_count": 1, "loss": correct if mode == "correct" else permuted,
             "positive_mass": .2, "span_hit": 0.,
             "uniform_all/loss": 2., "uniform_demo/loss": 1.8, "time_only/loss": 1.6}
            for eid in range(episodes) for q in range(2) for mode in ("correct", "content_permuted")]


class PairedProjectionTests(unittest.TestCase):
    def test_actual_initial_tensor_values_shapes_and_dtypes_are_bound(self):
        reference = {"weight": torch.ones(2, 3)}
        validate_tensor_layout(reference, {"weight": torch.ones(2, 3)}, exact=True)
        for bad in ({"weight": torch.zeros(2, 3)}, {"weight": torch.ones(3, 2)},
                    {"weight": torch.ones(2, 3, dtype=torch.float64)}, {"other": torch.ones(2, 3)}):
            with self.assertRaises(ValueError):
                validate_tensor_layout(reference, bad, exact=True)
        validate_tensor_layout(reference, {"weight": torch.zeros(2, 3)})

    def test_output_protects_entire_semantic_dataset_and_model_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {k: str(root / k) for k in ("dataset_path", "cache_dir", "sidecar_dir", "base_model")}
            roots = protected_roots({"identity": identity}, root / "targets/manifest.json",
                                    root / "v14/checkpoint-000512",
                                    {"metadata": {"initial_parent": {"path": str(root / "parent")}}})
            for protected in roots:
                with self.assertRaises(ValueError):
                    validate_output_scope(Path(protected) / "new-report", *roots)
            self.assertEqual(validate_output_scope(root / "reports/new", *roots), root / "reports/new")

    def test_smoke_or_changed_selection_is_not_the_fixed_study(self):
        protocol = {"train": {"max_steps": 256, "batch_size": 4, "learning_rate": 1e-4,
                              "eval_steps": 64, "seed": 9151},
                    "fixed_final_selection": True, "permutation_seed": 9152,
                    "bootstrap_seed": 9151, "bootstrap_samples": 10000, "time_only_scale": 16,
                    "minimum_val_episodes": 10, "minimum_val_queries": 20}
        validate_fixed_study(protocol)
        for key, value in (("max_steps", 4), ("batch_size", 8), ("seed", 9152)):
            altered = copy.deepcopy(protocol)
            altered["train"][key] = value
            with self.assertRaises(ValueError):
                validate_fixed_study(altered)
        with self.assertRaises(ValueError):
            validate_fixed_study({**protocol, "fixed_final_selection": False})

    def test_complete_bound_val_required_not_just_matching_subsets(self):
        validation = rows()
        examples = [{"episode_id": eid, "decision": q, "split": "val"}
                    for eid in range(12) for q in range(2)]
        examples.append({"episode_id": 999, "decision": 0, "split": "train"})
        validate_complete_val(validation, examples)
        for subset in (validation[:-2], validation + [validation[0]],
                       [r for r in validation if r["episode_id"] != 0]):
            with self.assertRaises(ValueError):
                validate_complete_val(subset, examples)

    def test_primary_gain_never_substitutes_for_original_gates(self):
        initial, control, candidate = rows(2., 2.), rows(1.4, 1.9), rows(1.7, 1.7)
        result = compare_rows(initial, control, copy.deepcopy(initial), candidate)
        self.assertTrue(result["projection_arm_superiority_supported"])
        self.assertEqual(result["decision"], "no_go")
        self.assertEqual(result["original_gate_candidate"]["decision"], "no_go")
        self.assertFalse(result["policy_ready"])
        self.assertFalse(result["goal_30_percent_achieved"])

    def test_both_primary_and_original_gates_required_and_reproducible(self):
        initial, control, candidate = rows(2., 2.), rows(1.5, 1.5), rows(1., 1.)
        result = compare_rows(initial, control, copy.deepcopy(initial), candidate)
        self.assertEqual(result["decision"], "go_to_action_probe")
        self.assertEqual(result, compare_rows(initial, control, copy.deepcopy(initial), candidate))
        self.assertFalse(result["policy_ready"])
        self.assertFalse(result["goal_30_percent_achieved"])

    def test_matched_ids_are_insufficient_if_targets_or_maps_differ(self):
        left = rows()
        for field, new in (("positive_frames", [32]), ("candidate_frames", [0, 16, 32, 80]),
                           ("destination_to_source_frames", [0, 32, 16, 64]),
                           ("time_only/loss", 7.)):
            right = copy.deepcopy(left)
            right[0][field] = new
            with self.assertRaises(ValueError):
                paired_rows(left, right)

    def test_reject_missing_duplicate_nonfinite_or_different_initial(self):
        a = rows()
        for right in (a[:-1], a + [a[0]], [], [{**a[0], "loss": float("nan")}]+a[1:]):
            with self.assertRaises(ValueError):
                paired_rows(a, right)
        with self.assertRaises(ValueError):
            compare_rows(rows(2., 2.), rows(), rows(2.1, 2.), rows(1., 1.))

    def test_insufficient_coverage_is_never_superiority_or_go(self):
        result = compare_rows(rows(2., 2., 9), rows(1.5, 1.5, 9),
                              rows(2., 2., 9), rows(1., 1., 9))
        self.assertFalse(result["projection_arm_superiority_supported"])
        self.assertEqual(result["decision"], "no_go")
        self.assertEqual(result["original_gate_candidate"]["decision"], "inconclusive_coverage")

    def test_episode_macro_not_adjacent_query_pseudoreplication(self):
        # Add many unique queries only to episode0. Each episode still counts once.
        initial, control, candidate = rows(3., 3.), rows(2., 2.), rows(1., 1.)
        for i in range(2, 102):
            for mode in ("correct", "content_permuted"):
                seed_row = next(r for r in initial if r["episode_id"] == 0 and r["mode"] == mode)
                initial.append({**seed_row, "decision": i})
                control.append({**seed_row, "decision": i, "loss": 2.})
                candidate.append({**seed_row, "decision": i, "loss": 1.})
        result = compare_rows(initial, control, copy.deepcopy(initial), candidate)
        primary = result["contrasts"]["content_permuted"]
        self.assertEqual(primary["episodes"], 12)
        self.assertAlmostEqual(primary["mean_loss_gain"], 1.)


if __name__ == "__main__":
    unittest.main()
