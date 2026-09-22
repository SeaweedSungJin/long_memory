"""CPU-only, synthetic tests for independent fixed-budget teacher analysis."""
from copy import deepcopy
import json
import unittest

import numpy as np

from run_scripts.robomme.cvom_budget_statistics import summarize_budget_contexts


def context(eid=0, event=32, task="task_a", *, a=(4., 8., 10.), b=(5., 8., 10.), held=(6., 8., 10.), generated=True):
    # KEEP is intentionally NOT index zero; output must honor the inventory.
    panels = {}
    for name, mean, queries in (("selection_a", a, [40, 44]), ("selection_b", b, [40, 44]),
                                ("heldout", held, [48, 52])):
        panels[name] = {"flow_prefix": [list(mean), list(mean)],
            "draw_ids": [f"{eid}:{event}:{name}:query-{query}:seed-{i}" for i, query in enumerate(queries)],
            "query_ids": queries}
    if generated:
        panels["heldout"]["generated_mse"] = [[.3, .6, 1.], [.3, .6, 1.]]
    return {"episode_id": eid, "task": task, "event": event,
        "operations": [{"id": "drop_middle", "kind": "drop", "drop_event_id": 1},
            {"id": "fifo", "kind": "fifo", "drop_event_id": 0},
            {"id": "keep", "kind": "keep", "drop_event_id": 32}],
        "panels": panels, "manager_operation_id": "fifo"}


def summarize(rows, **kwargs):
    return summarize_budget_contexts(rows, bootstrap_samples=100, min_episodes=2, **kwargs)


def gain(result, metric="flow_prefix", name="selected_a_vs_keep", macro="episode_macro"):
    return result["overall"][metric][macro]["gains"][name]


def test_positive_signed_gains_same_condition_random_expectation_and_generation_only_heldout():
    result = summarize([context(0), context(1)])
    row = result["per_context"][0]
    assert row["selection_a_operation_id"] == "drop_middle"
    assert row["selection_b_operation_id"] == "drop_middle"
    assert gain(result)["mean"] == 4.
    assert gain(result, name="selected_a_vs_fifo")["mean"] == 2.
    assert gain(result, name="selected_a_vs_uniform_random_expectation")["mean"] == 2.
    assert gain(result, name="manager_vs_keep")["mean"] == 2.
    assert abs(gain(result, metric="generated_mse")["mean"] - .7) < 1e-12
    assert row["selection_panel_generalization_gap"]["keep"] == 2.
    assert result["qualification_evidence"]["declared_panel_separation_verified"] is True
    assert "no automatic PASS" in result["qualification_evidence"]["decision"]
    # All floats are finite and undefined uncertainty is represented by null.
    json.dumps(result, allow_nan=False)


def test_selection_never_uses_heldout_and_oracle_is_marked_in_sample():
    value = context(a=(1., 4., 5.), b=(5., 4., 1.), held=(9., 7., 4.))
    result = summarize([value])
    row = result["per_context"][0]
    assert row["selection_a_operation_id"] == "drop_middle"
    assert row["selection_b_operation_id"] == "keep"
    assert gain(result)["mean"] == -5.
    assert gain(result, name="heldout_oracle_vs_selected_a_in_sample")["mean"] == 5.
    assert row["stability"]["pairwise_agreement"] == 0.
    assert row["selection_a_in_sample"]["gain_vs_keep"] == 4.
    assert row["selection_panel_generalization_gap"]["keep"] == 9.
    assert gain(result)["ci95"] is None  # one episode is not a bootstrap sample
    assert not result["qualification_evidence"]["enough_distinct_episodes_for_predeclared_minimum"]


def test_flat_teacher_prefers_keep_but_does_not_become_useful_evidence():
    rows = [context(i, a=(1., 1., 1.), b=(1., 1., 1.), held=(1., 1., 1.), generated=False) for i in range(3)]
    result = summarize(rows)
    assert all(row["selection_a_operation_id"] == "keep" for row in result["per_context"])
    assert result["stability"]["top1_repeat_episode_macro"]["value"] == 1.
    assert result["stability"]["pairwise_repeat_pooled_episode_bootstrap"]["value"] is None
    assert result["stability"]["pairwise_tied_both"] == 9
    assert result["stability"]["all_operations_tied_a"] == 3
    assert gain(result)["mean"] == 0.
    assert not result["qualification_evidence"]["positive_heldout_ci_evidence"]["flow_prefix"]["keep"]
    assert "generated_mse" not in result["overall"]


def test_pairwise_ties_are_excluded_and_reported_not_half_credit():
    result = summarize([context(a=(1., 1., 1.), b=(1., 2., 3.))])
    row = result["per_context"][0]["stability"]
    assert row["pairwise_comparable"] == 0
    assert row["pairwise_tied_one_panel"] == 3
    assert row["pairwise_tied_both"] == 0
    assert row["spearman_exact_average_ties"] is None


def test_average_contexts_within_episode_before_estimation_and_bootstrap():
    # Ten contexts from one episode must not outweigh one other episode.
    rows = [context(0, event=i, held=(8., 9., 10.), generated=False) for i in range(10)]
    rows.append(context(1, held=(12., 9., 10.), generated=False))
    result = summarize(rows)
    assert result["overall"]["flow_prefix"]["contexts"] == 11
    assert result["overall"]["flow_prefix"]["episodes"] == 2
    assert gain(result)["mean"] == 0.
    assert gain(result)["ci95"] == [-2., 2.]
    assert result["stability"]["top1_repeat_episode_macro"]["denominator"] == 2.


def test_task_macro_and_episode_macro_are_distinct_and_singleton_task_warns():
    rows = [context(0, task="a", held=(8., 9., 10.), generated=False),
            context(1, task="a", held=(8., 9., 10.), generated=False),
            context(2, task="b", held=(12., 9., 10.), generated=False)]
    result = summarize(rows)
    assert abs(gain(result)["mean"] - 2/3) < 1e-12
    assert gain(result, macro="task_macro")["mean"] == 0.
    assert result["overall"]["flow_prefix"]["task_macro"]["sparse_task_warning"]
    assert result["by_task"]["b"]["flow_prefix"]["episode_macro"]["gains"]["selected_a_vs_keep"]["ci95"] is None


def test_bootstrap_is_deterministic_paired_and_does_not_modify_input_or_global_rng():
    rows = [context(i, held=(float(i+2), 8., 10.)) for i in range(5)]
    saved = deepcopy(rows)
    np.random.seed(91)
    before = np.random.get_state()
    first, second = summarize(rows), summarize(list(reversed(rows)))
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert rows == saved
    assert first["overall"] == second["overall"]
    assert first["by_task"] == second["by_task"]
    # FIFO minus KEEP is constant across episodes, so paired contrasts differ
    # by exactly two, including every bootstrap endpoint.
    k, f = gain(first), gain(first, name="selected_a_vs_fifo")
    assert k["mean"] - f["mean"] == 2.
    assert np.allclose(np.asarray(k["ci95"]) - np.asarray(f["ci95"]), 2.)


def test_heldout_draw_and_query_overlap_are_rejected_but_shared_selection_queries_allowed():
    value = context()
    summarize([value])  # A/B share future queries, with different draw seeds.
    bad = deepcopy(value)
    bad["panels"]["heldout"]["draw_ids"][0] = bad["panels"]["selection_a"]["draw_ids"][0]
    with unittest.TestCase().assertRaisesRegex(ValueError, "draw identities overlap"):
        summarize([bad])
    bad = deepcopy(value)
    bad["panels"]["heldout"]["query_ids"][0] = 40
    with unittest.TestCase().assertRaisesRegex(ValueError, "future queries overlap"):
        summarize([bad])
    bad = deepcopy(value)
    bad["panels"]["selection_b"]["draw_ids"] = bad["panels"]["selection_a"]["draw_ids"]
    with unittest.TestCase().assertRaises(ValueError):
        summarize([bad])


def test_missing_panel_metadata_is_not_fabricated_as_independence():
    value = context()
    for panel in value["panels"].values():
        panel.pop("draw_ids"); panel.pop("query_ids")
    result = summarize([value])
    assert result["per_context"][0]["panel_audit"]["disjoint_draw_ids_verified"] is False
    assert result["qualification_evidence"]["declared_panel_separation_verified"] is False


def test_optional_generated_mse_cannot_silently_select_a_subset():
    with unittest.TestCase().assertRaisesRegex(ValueError, "all included contexts"):
        summarize([context(0), context(1, generated=False)])


def test_joint_gripper_and_weighted_panels_keep_original_flow_prefix_selection():
    rows = [context(0), context(1)]
    for row in rows:
        for name in ("generated_joint", "generated_gripper", "flow_joint", "flow_gripper", "flow_weighted"):
            row["panels"]["heldout"][name] = [[.8, .6, .4], [.8, .6, .4]]
    result = summarize(rows)
    assert all(row["selection_a_operation_id"] == "drop_middle" for row in result["per_context"])
    assert gain(result)["mean"] == 4.
    for metric in ("generated_joint", "generated_gripper", "flow_joint", "flow_gripper", "flow_weighted"):
        assert abs(gain(result, metric=metric)["mean"] + .4) < 1e-12
        assert metric in result["by_task"]["task_a"]
    partial = deepcopy(rows)
    partial[1]["panels"]["heldout"].pop("generated_joint")
    with unittest.TestCase().assertRaisesRegex(ValueError, "generated_joint must cover all"):
        summarize(partial)


def test_near_tie_rule_is_explicit_and_oracle_bound_uses_true_minimum():
    value = context(a=(1., 1.001, 1.002), b=(1., 1.001, 1.002), held=(1., 1.001, 1.002))
    result = summarize([value], tie_tolerance=.01)
    assert result["per_context"][0]["selection_a_operation_id"] == "keep"
    assert abs(gain(result, name="heldout_oracle_vs_selected_a_in_sample")["mean"]-.002) < 1e-12


def test_invalid_inventory_nonfinite_loss_shape_and_duplicate_evidence_fail_closed():
    value = context()
    bad = deepcopy(value); bad["manager_operation_id"] = "not-evaluated"
    cases = [bad]
    bad = deepcopy(value); bad["operations"][0]["drop_event_id"] = 0; cases.append(bad)
    bad = deepcopy(value); bad["operations"][0]["kind"] = "keep"; cases.append(bad)
    bad = deepcopy(value); bad["panels"]["heldout"]["flow_prefix"][0][0] = float("nan"); cases.append(bad)
    bad = deepcopy(value); bad["panels"]["heldout"]["flow_prefix"][0][0] = -1.; cases.append(bad)
    bad = deepcopy(value); bad["panels"]["heldout"]["flow_prefix"] = [[1., 2.]]; cases.append(bad)
    for case in cases:
        with unittest.TestCase().assertRaises(ValueError):
            summarize([case])
    with unittest.TestCase().assertRaisesRegex(ValueError, "Duplicate context"):
        summarize([value, deepcopy(value)])
    wrong_task = context(task="other", event=33)
    with unittest.TestCase().assertRaisesRegex(ValueError, "multiple tasks"):
        summarize([value, wrong_task])
    with unittest.TestCase().assertRaisesRegex(ValueError, "flow_prefix only"):
        summarize([value], metric="generated_mse")


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in sorted(globals().items())
                              if name.startswith("test_") and callable(fn))


if __name__ == "__main__":
    unittest.main()
