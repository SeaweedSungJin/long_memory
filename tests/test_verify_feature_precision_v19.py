"""CPU guardrails for fixed-panel precision regression; no model loading."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from run_scripts.robomme.verify_feature_precision_v19 import (
    POLICY, STAGES, action_seed, compare_captures, comparison_pass,
    legacy_source, measured, panel_from_metadata, parse_args,
)


def row(eid, demo, count):
    return {"episode_id": eid, "demo_events": demo, "events": count}


class PrecisionRegressionTests(unittest.TestCase):
    def test_panel_is_chronology_only_and_deterministic(self):
        rows = [row(1355, 67, 81), row(1363, 71, 85), row(1, 71, 84),
                row(626, 0, 9), row(2, 0, 10), row(3, 4, 7)]
        expected = [1355, 1363, 626]
        self.assertEqual([r["episode_id"] for r in panel_from_metadata(rows)], expected)
        self.assertEqual([r["episode_id"] for r in panel_from_metadata(list(reversed(rows)))], expected)
        for r in rows:
            r["success"] = r["episode_id"] != 1363
            r["loss"] = -r["episode_id"]
        self.assertEqual([r["episode_id"] for r in panel_from_metadata(rows)], expected)

    def test_panel_fails_closed_when_required_groups_missing(self):
        with self.assertRaisesRegex(ValueError, "1355"):
            panel_from_metadata([row(1, 90, 100)])
        with self.assertRaisesRegex(ValueError, "FIFO32"):
            panel_from_metadata([row(1355, 67, 81), row(2, 0, 3)])

    def test_value_equality_does_not_mistake_bf16_promotion_for_error(self):
        saved = torch.tensor([1.1, -2.3], dtype=torch.bfloat16)
        result = measured(saved, saved.float())
        self.assertTrue(result["values_equal"] and result["finite"])
        self.assertFalse(result["exact"] or result["same_dtype"])
        self.assertTrue(comparison_pass([result]))

    def test_no_tolerance_can_hide_small_same_path_difference(self):
        ref = torch.tensor([1.0])
        cur = torch.nextafter(ref, torch.tensor([2.0]))
        self.assertFalse(comparison_pass([measured(ref, cur)]))
        self.assertFalse(comparison_pass([]))
        self.assertFalse(comparison_pass([measured(torch.tensor([float("nan")]), torch.tensor([float("nan")]))]))

    def test_all_required_stages_and_masks_are_compared(self):
        reference = {key: torch.zeros(1, 2) for key in STAGES}
        current = {key: value.clone() for key, value in reference.items()}
        current["image_masks"][0, 0] = 1
        rows = compare_captures(3, "aligned", [reference], [current])
        self.assertEqual(set(r["stage"] for r in rows), set(STAGES))
        self.assertFalse(comparison_pass(rows))
        with self.assertRaises(AssertionError):
            compare_captures(3, "aligned", [reference], [])

    def test_historical_source_must_match_immutable_manifest(self):
        source = b"historical native policy"
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "manifest.json"
            manifest.write_text(json.dumps({"source_sha256": {POLICY: hashlib.sha256(source).hexdigest()}}))
            with patch("run_scripts.robomme.verify_feature_precision_v19.subprocess.check_output", return_value=source):
                self.assertEqual(legacy_source("original-ref", manifest), (source, hashlib.sha256(source).hexdigest()))
                manifest.write_text(json.dumps({"source_sha256": {POLICY: "wrong"}}))
                with self.assertRaisesRegex(ValueError, "Historical"):
                    legacy_source("original-ref", manifest)

    def test_action_seeds_are_fixed_and_branch_independent(self):
        self.assertEqual(action_seed(1355, 67), action_seed(1355, 67))
        self.assertNotEqual(action_seed(1355, 67), action_seed(1355, 68))
        self.assertNotEqual(action_seed(1355, 67), action_seed(1363, 67))

    def test_cli_has_no_training_or_panel_override(self):
        args = parse_args(["--output-dir", "/tmp/new-diagnostic", "--prepare-only"])
        self.assertTrue(args.prepare_only)
        self.assertFalse(args.execute_plan)
        self.assertFalse(hasattr(args, "episodes") or hasattr(args, "steps"))


if __name__ == "__main__":
    unittest.main()
