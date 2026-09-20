"""CPU-only contracts for evidence audits, not robot-success or cue-GT proof.

The synthetic annotations below are deliberately explicit.  Neither a subgoal
label nor an old event's presence is evidence that its content is task-relevant.
"""

import copy
import json
import unittest
from unittest.mock import patch

import torch

from gr00t.long_memory.evidence_audit import (
    audit_query, rank_interval, score_cues, summarize_audit, validate_annotation,
)
from tests.test_long_memory_v3_core import episode, model


FINGERPRINT = "a" * 64


def annotation(data, *, decision=6, status="verified", cues=None):
    if cues is None:
        cues = ([{"cue_id": "cue-1", "description": "Synthetic past cue",
                  "intervals": [[2, 2]]}] if status == "verified" else [])
    return {"schema_version": 1, "cache_fingerprint": FINGERPRINT,
            "episode_id": int(data["episode_id"]), "split": "val",
            "decision": decision, "query_frame": int(data["frames"][decision]),
            "status": status, "cues": cues,
            "reviewer": "unit-test" if status != "unknown" else "",
            "notes": "Synthetic fixture, not benchmark evidence."}


class TestEvidenceAnnotation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(604)
        torch.set_num_threads(1)
        self.data = episode()

    def validate(self, row):
        return validate_annotation(row, self.data, cache_fingerprint=FINGERPRINT, split="val")

    def test_verified_none_unknown_and_frame_zero_are_representable(self):
        for status in ("verified", "none", "unknown"):
            row = annotation(self.data, status=status)
            original = copy.deepcopy(row)
            result = self.validate(row)
            self.assertEqual(result["status"], status)
            self.assertEqual(row, original)
        row = annotation(self.data, cues=[{
            "cue_id": "initial-cue", "description": "Visible at the initial frame",
            "intervals": [[0, 0]],
        }])
        self.assertEqual(self.validate(row)["cues"][0]["intervals"], [[0, 0]])

    def test_schema_version_and_integer_identity_reject_bools_floats_strings(self):
        for field in ("schema_version", "episode_id", "decision", "query_frame"):
            for value in (True, False, 1.0, "1", None):
                with self.subTest(field=field, value=value):
                    row = annotation(self.data)
                    row[field] = value
                    with self.assertRaises(ValueError):
                        self.validate(row)
        row = annotation(self.data)
        row["schema_version"] = 2
        with self.assertRaises(ValueError):
            self.validate(row)

    def test_fingerprint_episode_split_and_query_frame_are_bound_to_cache(self):
        for field, value in (("cache_fingerprint", "b" * 64),
                             ("cache_fingerprint", "a" * 63),
                             ("episode_id", 11), ("split", "train"),
                             ("split", "test"), ("query_frame", 13)):
            with self.subTest(field=field, value=value):
                row = annotation(self.data)
                row[field] = value
                with self.assertRaises(ValueError):
                    self.validate(row)

    def test_decision_must_be_in_bounds_and_active(self):
        for decision in (-1, len(self.data["frames"])):
            row = annotation(self.data)
            row["decision"] = decision
            with self.assertRaises(ValueError):
                self.validate(row)
        row = annotation(self.data)
        self.data["decision_mask"][row["decision"]] = False
        with self.assertRaises(ValueError):
            self.validate(row)

    def test_verified_and_none_require_explicit_human_review(self):
        for status in ("verified", "none"):
            for reviewer in ("", "   ", None, False):
                with self.subTest(status=status, reviewer=reviewer):
                    row = annotation(self.data, status=status)
                    row["reviewer"] = reviewer
                    with self.assertRaises(ValueError):
                        self.validate(row)
        row = annotation(self.data)
        row["cues"] = []
        with self.assertRaises(ValueError):
            self.validate(row)
        row = annotation(self.data)
        row["status"] = "none"
        with self.assertRaises(ValueError):
            self.validate(row)

    def test_unknown_status_does_not_silently_mean_verified(self):
        for status in ("annotated", "", None, True):
            row = annotation(self.data)
            row["status"] = status
            with self.assertRaises(ValueError):
                self.validate(row)

    def test_intervals_must_be_raw_integer_past_frames(self):
        for interval in ([12, 12], [11, 12], [13, 14], [-1, 1], [4, 2],
                         [True, 2], [1, False], [1.0, 2], [1, "2"], [1],
                         [1, 2, 3], [], None):
            with self.subTest(interval=interval):
                row = annotation(self.data)
                row["cues"][0]["intervals"] = [interval]
                with self.assertRaises(ValueError):
                    self.validate(row)

    def test_empty_duplicate_and_overlapping_alternatives_are_rejected(self):
        for intervals in ([], [[1, 2], [2, 3]], [[2, 2], [2, 2]], [[1, 4], [2, 3]]):
            with self.subTest(intervals=intervals):
                row = annotation(self.data)
                row["cues"][0]["intervals"] = intervals
                with self.assertRaises(ValueError):
                    self.validate(row)
        row = annotation(self.data)
        row["cues"][0]["intervals"] = [[1, 1], [4, 5]]
        self.assertEqual(len(self.validate(row)["cues"][0]["intervals"]), 2)

    def test_cue_ids_are_unique_nonempty_strings(self):
        row = annotation(self.data)
        row["cues"].append(copy.deepcopy(row["cues"][0]))
        with self.assertRaises(ValueError):
            self.validate(row)
        for cue_id in ("", "  ", None, 4, True):
            row = annotation(self.data)
            row["cues"][0]["cue_id"] = cue_id
            with self.assertRaises(ValueError):
                self.validate(row)


class TestEvidenceAudit(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(604)
        torch.set_num_threads(1)
        self.data = episode()
        self.memory = model(capacity=8).eval()

    def audit(self, row=None, *, memory=None, data=None, **kwargs):
        data = self.data if data is None else data
        return audit_query(self.memory if memory is None else memory, data,
                           annotation(data) if row is None else row,
                           cache_fingerprint=FINGERPRINT, split="val", **kwargs)

    def test_cue_between_endpoints_is_not_assumed_encoded(self):
        row = annotation(self.data)
        row["cues"][0]["intervals"] = [[1, 1]]
        result = self.audit(row)
        cue = result["cues"][0]
        self.assertEqual(cue["sampled_frames"], [])
        self.assertEqual(cue["direct_event_ids"], [])
        self.assertEqual(cue["bank_event_ids"], [])
        self.assertEqual(cue["rank"], {"best": None, "worst": None})
        self.assertIsNone(cue["top_k"]["1"]["guaranteed"])
        summary = summarize_audit([result])
        self.assertEqual(summary["evidence"]["endpoint_coverage"]["rate"], 0.0)
        self.assertIsNone(summary["evidence"]["bank_retention"]["rate"])

    def test_endpoint_has_two_carriers_and_frame_zero_is_retained(self):
        cue = self.audit()["cues"][0]
        self.assertEqual(cue["sampled_frames"], [2])
        self.assertEqual(cue["direct_event_ids"], [0, 1])
        self.assertEqual(cue["bank_event_ids"], [0, 1])
        row = annotation(self.data)
        row["cues"][0]["intervals"] = [[0, 0]]
        cue = self.audit(row)["cues"][0]
        self.assertEqual(cue["sampled_frames"], [0])
        self.assertEqual(cue["direct_event_ids"], [0])

    def test_invalid_event_does_not_hide_valid_adjacent_endpoint_carrier(self):
        self.data["transition_valid"][0] = False
        cue = self.audit()["cues"][0]
        self.assertEqual(cue["sampled_frames"], [2])
        self.assertEqual(cue["direct_event_ids"], [1])
        self.assertEqual(cue["bank_event_ids"], [1])

    def test_sampled_but_evicted_is_separate_from_not_sampled(self):
        result = self.audit(memory=model(capacity=3).eval())
        self.assertEqual(result["bank_ids"], [3, 4, 5])
        cue = result["cues"][0]
        self.assertEqual(cue["sampled_frames"], [2])
        self.assertEqual(cue["direct_event_ids"], [0, 1])
        self.assertEqual(cue["bank_event_ids"], [])
        summary = summarize_audit([result])["evidence"]
        self.assertEqual(summary["endpoint_coverage"]["rate"], 1.0)
        self.assertEqual(summary["carrier_coverage"]["rate"], 1.0)
        self.assertEqual(summary["bank_retention"]["rate"], 0.0)
        self.assertIsNone(summary["top_k"]["1"]["guaranteed"]["rate"])

    def test_old_carriers_use_strict_event_end_before_short_boundary(self):
        result = self.audit()
        self.assertEqual(result["oldest_short_frame"], 6)
        self.assertEqual(result["old_bank_ids"], [0, 1])
        self.assertEqual(result["cues"][0]["old_bank_event_ids"], [0, 1])
        row = annotation(self.data)
        row["cues"][0]["intervals"] = [[6, 6]]
        cue = self.audit(row)["cues"][0]
        self.assertEqual(cue["direct_event_ids"], [2, 3])
        self.assertEqual(cue["old_bank_event_ids"], [])
        self.assertFalse(cue["temporally_old"])

    def test_separate_required_cues_are_not_collapsed_to_any_positive(self):
        row = annotation(self.data, cues=[
            {"cue_id": "first", "description": "First required count event",
             "intervals": [[2, 2], [4, 4]]},
            {"cue_id": "second", "description": "Second required count event",
             "intervals": [[10, 10]]},
        ])
        result = self.audit(row, memory=model(capacity=3).eval())
        self.assertEqual(len(result["cues"]), 2)
        self.assertEqual(result["cues"][0]["direct_event_ids"], [0, 1, 2])
        self.assertEqual(result["cues"][0]["bank_event_ids"], [])
        self.assertEqual(result["cues"][1]["bank_event_ids"], [4, 5])
        self.assertTrue(result["all_groups_sampled"])
        self.assertFalse(result["all_groups_retained"])
        summary = summarize_audit([result])
        self.assertEqual(summary["counts"]["cue_groups"], 2)
        self.assertEqual(summary["evidence"]["bank_retention"],
                         {"numerator": 1, "denominator": 2, "rate": 0.5})
        self.assertEqual(summary["joint_queries"]["all_groups_retained"]["rate"], 0.0)

    def test_rank_uses_actual_attention_not_auxiliary_event_score(self):
        original_read = self.memory.read

        def disagreeing_scores(*args, **kwargs):
            result = original_read(*args, **kwargs)
            result["weights"] = torch.tensor([[[.40, .30, .10, .05, .05, .05, .05],
                                                 [.40, .30, .10, .05, .05, .05, .05]]])
            result["event_scores"] = torch.tensor([[-100., 0., 1., 2., 3., 100.]])
            return result

        row = annotation(self.data)
        row["cues"][0]["intervals"] = [[0, 0]]
        with patch.object(self.memory, "read", side_effect=disagreeing_scores):
            result = self.audit(row)
        self.assertAlmostEqual(result["attention"]["event_weights"][0], .4)
        self.assertAlmostEqual(result["cues"][0]["attention_mass"], .4)
        self.assertEqual(result["cues"][0]["rank"], {"best": 1, "worst": 1})

    def test_uniform_attention_exposes_optimistic_and_guaranteed_tie_bounds(self):
        original_read = self.memory.read

        def tied_attention(*args, **kwargs):
            result = original_read(*args, **kwargs)
            result["weights"] = torch.full_like(result["weights"], 1 / 7)
            return result

        with patch.object(self.memory, "read", side_effect=tied_attention):
            cue = self.audit()["cues"][0]
        self.assertEqual(cue["rank"], {"best": 1, "worst": 5})
        self.assertEqual(cue["rank_with_null"], {"best": 1, "worst": 6})
        self.assertFalse(cue["top_k"]["4"]["guaranteed"])
        self.assertTrue(cue["top_k"]["1"]["possible"])
        self.assertAlmostEqual(cue["top_k"]["1"]["chance"], 2 / 6)
        self.assertAlmostEqual(cue["top_k"]["4"]["chance"], 14 / 15)

    def test_unknown_and_none_do_not_become_zero_retrieval_accuracy(self):
        results = [self.audit(annotation(self.data, status=status))
                   for status in ("unknown", "none")]
        for result in results:
            self.assertEqual(result["cues"], [])
            self.assertIsInstance(result["attention"]["null_weight"], float)
        summary = summarize_audit(results)
        self.assertEqual(summary["counts"]["queries"], 2)
        self.assertEqual(summary["counts"]["unknown"], 1)
        self.assertEqual(summary["counts"]["none"], 1)
        self.assertEqual(summary["counts"]["verified"], 0)
        self.assertEqual(summary["counts"]["cue_groups"], 0)
        self.assertIsNone(summary["evidence"]["endpoint_coverage"]["rate"])
        self.assertIsNone(summary["evidence"]["top_k"]["1"]["guaranteed"]["rate"])
        json.dumps(summary, allow_nan=False)

    def test_future_observations_actions_and_invalid_transitions_do_not_leak(self):
        changed = copy.deepcopy(self.data)
        for key in ("short", "moment", "state"):
            changed[key][7:] = float("nan")
        changed["actions"][6:] = float("nan")
        changed["transition_valid"][6:] = False
        first, second = self.audit(), self.audit(data=changed)
        self.assertEqual(first, second)
        self.assertTrue(all(i < first["decision"] for i in first["bank_ids"]))

    def test_audit_preserves_parameters_eval_mode_gradients_and_episode(self):
        for parameter in self.memory.parameters():
            parameter.grad = torch.full_like(parameter, .125)
        state = copy.deepcopy(self.memory.state_dict())
        gradients = {name: p.grad.clone() for name, p in self.memory.named_parameters()}
        data = copy.deepcopy(self.data)
        row = annotation(self.data)
        before_row = copy.deepcopy(row)
        result = self.audit(row)
        self.assertFalse(self.memory.training)
        self.assertEqual(row, before_row)
        for name, tensor in self.memory.state_dict().items():
            self.assertTrue(torch.equal(tensor, state[name]), name)
        for name, parameter in self.memory.named_parameters():
            self.assertTrue(torch.equal(parameter.grad, gradients[name]), name)
        for name, tensor in self.data.items():
            if isinstance(tensor, torch.Tensor):
                self.assertTrue(torch.equal(tensor, data[name]), name)
        json.dumps(result, allow_nan=False)

    def test_train_mode_is_rejected_without_changing_it(self):
        self.memory.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            self.audit()
        self.assertTrue(self.memory.training)

    def test_annotation_targets_never_change_memory_storage_or_read(self):
        first = annotation(self.data)
        second = annotation(self.data, status="unknown")
        second["notes"] = "Different labels are scoring-only, not policy features."
        for policy in ("all", "hard"):
            a = self.audit(first, writer_policy=policy)
            b = self.audit(second, writer_policy=policy)
            for key in ("bank_ids", "old_bank_ids", "storage", "attention",
                        "gate_mean", "residual_norm"):
                self.assertEqual(a[key], b[key], (policy, key))

    def test_cpu_rng_state_is_unchanged(self):
        before = torch.get_rng_state().clone()
        self.audit(writer_policy="hard")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_empty_bank_is_a_valid_unknown_query_not_nan(self):
        row = annotation(self.data, decision=0, status="unknown")
        result = self.audit(row)
        self.assertEqual(result["bank_ids"], [])
        self.assertEqual(result["attention"]["null_weight"], 1.0)
        self.assertEqual(result["attention"]["event_weights"], [])
        json.dumps(result, allow_nan=False)

    def test_scorer_rejects_future_unfinished_duplicate_or_unsorted_bank(self):
        row = annotation(self.data)
        for bank in ([6], [-1], [1, 0], [0, 0], [True], [1.0]):
            with self.subTest(bank=bank), self.assertRaises(ValueError):
                score_cues(row, self.data, bank, [.5 / len(bank)] * len(bank), .5)
        self.data["transition_valid"][0] = False
        with self.assertRaises(ValueError):
            score_cues(row, self.data, [0], [.5], .5)

    def test_scorer_rejects_invalid_attention_and_top_k(self):
        row = annotation(self.data)
        for weights, null in (([float("nan")], .5), ([.5], float("inf")),
                              ([-.1], 1.1), ([.7], .7), ([], .5)):
            with self.subTest(weights=weights, null=null), self.assertRaises(ValueError):
                score_cues(row, self.data, [0], weights, null)
        for top_k in ((), (0,), (-1,), (True,), (1.0,), (1, 1)):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                score_cues(row, self.data, [0], [.5], .5, top_k=top_k)

    def test_summary_refuses_mismatched_top_k_plans(self):
        first = self.audit(top_k=(1, 4))
        second = self.audit(top_k=(1, 3))
        with self.assertRaisesRegex(ValueError, "top-k"):
            summarize_audit([first, second])


class TestEvidenceRanks(unittest.TestCase):
    def test_first_correct_tie_bounds_do_not_count_other_positives_as_failures(self):
        self.assertEqual(rank_interval([.2] * 6, [0, 1]), {"best": 1, "worst": 5})
        self.assertEqual(rank_interval([.1, .9, .2, .3], [0, 2]),
                         {"best": 3, "worst": 3})
        self.assertEqual(rank_interval([.5, .5], [0, 1]), {"best": 1, "worst": 1})
        self.assertEqual(rank_interval([.5, .5], []), {"best": None, "worst": None})

    def test_numerical_near_ties_are_not_advertised_as_certain_wins(self):
        self.assertEqual(rank_interval([.5, .5 - 1e-9], [0]), {"best": 1, "worst": 2})

    def test_rank_rejects_duplicate_bad_indices_and_nonfinite_scores(self):
        for positives in ([0, 0], [-1], [2], [True], [1.0]):
            with self.subTest(positives=positives), self.assertRaises(ValueError):
                rank_interval([.1, .2], positives)
        for score in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(score=score), self.assertRaises(ValueError):
                rank_interval([score, .2], [0])


if __name__ == "__main__":
    unittest.main()
