"""CPU fixtures for frozen-runtime aggregation and intervention pairing."""
import copy
import unittest

from run_scripts.robomme import summarize_v19_diagnostics as summary


def fixture():
    selection, records = [], []
    for episode in (1, 2, 3, 4):
        for repeat in (0, 1):
            key = dict(task="Task", episode_id=episode, decision=5, repeat=repeat, flow_seed=episode * 10 + repeat,
                       generation_seed=episode * 20 + repeat)
            selection.append(key)
            for name in summary.CONDITIONS:
                offset = 0 if name == "normal-bank" else 1 if name.startswith("random") else 2
                records.append({**key, "intervention": name, "removed_event_count": 2,
                                **{metric: float(episode + repeat + offset) for metric in summary.PRIMARY}})
    return selection, records


class RuntimeSummaryTests(unittest.TestCase):
    def test_pairing_noise_aggregation_and_delta_direction(self):
        selection, records = fixture()
        rows, queries = summary.paired_interventions(records, selection, draws=100)
        for row in rows:
            expected = 1 if row["contrast"].startswith("block-minus-matched-random") or row["condition"].startswith("random") else 2
            self.assertEqual(row["error_delta"], expected)
            self.assertEqual(row["ci95_low"], expected)
            self.assertEqual(row["ci95_high"], expected)
            self.assertEqual(row["episodes"], 4)
        query = next(row for row in queries if row["episode_id"] == 1 and row["condition"] == "memory-off")
        self.assertEqual(query["reference_error"], 1.5)
        self.assertEqual(query["condition_error"], 3.5)

    def test_both_random_event_sets_and_random_contiguous_blocks(self):
        selection, records = fixture()
        extra = []
        for row in records:
            if row["intervention"].startswith("random-set"):
                item = copy.deepcopy(row)
                item["intervention"] = item["intervention"].replace("random-set", "random-block")
                extra.append(item)
        rows, _ = summary.paired_interventions(records + extra, selection, draws=100)
        kinds = {row["contrast"] for row in rows}
        self.assertIn("block-minus-matched-random-block", kinds)
        self.assertIn("block-minus-matched-random-set", kinds)

    def test_reject_unmatched_noise_missing_duplicate_or_unmatched_removal(self):
        selection, records = fixture()
        for mutate in (lambda data: data.pop(), lambda data: data.append(copy.deepcopy(data[0])),
                       lambda data: data[0].update(flow_seed=1000), lambda data: data[2].update(removed_event_count=3)):
            changed = copy.deepcopy(records)
            mutate(changed)
            with self.assertRaises(ValueError):
                summary.paired_interventions(changed, selection, draws=100)

    def test_parity_undefined_relative_l2_and_exact_dtypes(self):
        rows = [{"branch": "x", "stage": "bank", "episode_id": 1, "decision": 0,
                 "exact": "True", "same_dtype": "True", "same_shape": "True", "finite": "True",
                 "reference_dtype": "float", "candidate_dtype": "float", "relative_l2": "", "max_abs": "0"},
                {"branch": "x", "stage": "bank", "episode_id": 1, "decision": 1,
                 "exact": "False", "same_dtype": "False", "same_shape": "True", "finite": "True",
                 "reference_dtype": "float", "candidate_dtype": "bfloat", "relative_l2": "0.2", "max_abs": "0.1"}]
        result = summary.parity_summary(rows)[0]
        self.assertEqual(result["exact_count"], 1)
        self.assertEqual(result["relative_l2_undefined_count"], 1)
        self.assertEqual(result["relative_l2_min"], .2)
        self.assertEqual(result["max_abs_max"], .1)

    def test_action_branch_keys_and_noise_means(self):
        rows = [{"branch": branch, "episode_id": 1, "decision": 2, "repeat": repeat,
                 **{metric: repeat + offset for metric in summary.PRIMARY}}
                for branch, offset in (("saved-cache", 1), ("live", 2)) for repeat in (0, 1)]
        result = summary.action_summary(rows)
        self.assertEqual(result[0]["branch"], "live")
        self.assertEqual(result[0]["mean_error"], 2.5)
        self.assertEqual(result[0]["delta_from_saved_cache"], 1)
        with self.assertRaises(ValueError):
            summary.action_summary(rows[:-1])

    def test_removed_window_can_overlap_retained_and_query(self):
        events = [{"event_id": i, "visible": True, "is_demo": True, "endpoint_frame": i * 10,
                   "source_event_ids": [max(0, i - 1), i], "source_frames": [max(0, i - 1) * 10, i * 10]}
                  for i in range(3)]
        bank = {"episode_id": 1, "decision": 3, "representation": "short", "events": events, "visible_event_ids": [0, 1, 2],
                "visible_event_count": 3, "visible_token_count": 12, "query": {"source_event_ids": [2, 3], "source_frames": [20, 30]},
                "token_semantics": "mixed", "write_timing": "after read"}
        records = [{"episode_id": 1, "decision": 3, "repeat": 0, "intervention": name, "removed_event_ids": [2],
                    "remaining_event_ids": [0, 1], "read_event_ids": [0, 1], "read_enabled": True} for name in summary.CONDITIONS]
        _, overlaps = summary.summarize_provenance({"1:3": bank}, records)
        self.assertEqual(overlaps[0]["removed_sources_overlap_retained"], [1])
        self.assertEqual(overlaps[0]["removed_sources_overlap_query"], [2])
        self.assertEqual(overlaps[0]["removed_sources_absent_from_retained_and_query"], [])


if __name__ == "__main__":
    unittest.main()
