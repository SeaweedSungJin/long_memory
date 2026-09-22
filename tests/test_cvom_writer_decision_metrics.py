"""CPU synthetic tests for admission/ranking/heldout decision separation."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from run_scripts.robomme.cvom_writer_decision_metrics import (
    load_probe_contexts, main, summarize_writer_decisions,
)


def context(eid=0, *, a=(2., 4., 1., 3.), b=None, held=None, p=.8,
            utility=(2., 1., 3., -100.), retention=(2., 1., 3., -100.),
            manager="drop-1", task="task_a", event=32, generated=False):
    # Loss order = [KEEP, drop old2, drop old1, FIFO old0]. Score arrays
    # instead follow chronological [old0, old1, old2, candidate] order.
    operations = [
        {"id": "keep", "kind": "keep", "drop_event_id": event, "drop_pool_index": 3},
        {"id": "drop-2", "kind": "drop", "drop_event_id": event-1, "drop_pool_index": 2},
        {"id": "drop-1", "kind": "drop", "drop_event_id": event-2, "drop_pool_index": 1},
        {"id": "fifo", "kind": "fifo", "drop_event_id": event-3, "drop_pool_index": 0},
    ]
    panels = {}
    for name, losses, queries in (("selection_a", a, [40, 44]),
            ("selection_b", a if b is None else b, [40, 44]),
            ("heldout", a if held is None else held, [48, 52])):
        panels[name] = {"flow_prefix": [list(losses), list(losses)], "query_ids": queries,
            "draw_ids": [f"{eid}:{event}:{name}:{i}" for i in range(2)]}
    if generated:
        panels["heldout"]["generated_mse"] = [[.5, .1, .9, .3]]
        panels["heldout"]["generated_joint"] = [[.2, .4, .6, .8]]
        panels["heldout"]["generated_gripper"] = [[.8, .6, .4, .2]]
    return {"episode_id": eid, "task": task, "event": event,
        "context_id": f"{eid}:{event}", "operations": operations, "panels": panels,
        "manager_scores": {"utility": list(utility), "retention": list(retention),
                           "write_probability": [.01, .99, .5, p]},
        "manager_operation_id": manager}


def summarize(rows, **kwargs):
    return summarize_writer_decisions(rows, bootstrap_samples=100, **kwargs)


def gain(result, name="manager_vs_fifo", metric="flow_prefix"):
    return result["overall"]["heldout"][metric]["episode_macro"]["gains"][name]


def test_candidate_only_accuracy_and_pool_operation_mapping():
    rows = [context(0), context(1, a=(.5, 4., 1., 3.), p=.1, manager="keep")]
    result = summarize(rows)
    admission = result["overall"]["admission"]
    assert admission["candidate_decisions"] == 2
    assert admission["confusion"] == dict(true_keep=1, false_admit=0, false_keep=0, true_admit=1)
    assert admission["metrics"]["accuracy"]["value"] == 1.
    assert admission["metrics"]["balanced_accuracy"]["value"] == 1.
    assert admission["metrics"]["majority_accuracy"]["value"] == .5
    assert result["per_context"][0]["selected_operation_ids"]["retention_forced_replacement"] == "drop-1"
    changed = deepcopy(rows)
    for row in changed:
        row["manager_scores"]["write_probability"][:3] = [1., 0., 1.]
    assert summarize(changed)["overall"]["admission"] == admission
    json.dumps(result, allow_nan=False)


def test_uncertain_band_excludes_does_not_make_negative_and_missing_class_balanced_undefined():
    result = summarize([context(0, a=(1., 4., 1., 3.)), context(1)])
    stats = result["overall"]["admission"]
    assert stats["uncertain_contexts"] == 1
    assert stats["confident_contexts"] == 1
    assert stats["metrics"]["balanced_accuracy"]["value"] is None
    assert result["per_context"][0]["admission"]["target_admit"] is None
    result = summarize([context(0, a=(1., 4., 1., 3.))])
    assert all(v["value"] is None for v in result["overall"]["admission"]["metrics"].values())


def test_admission_margin_and_threshold_are_separate_from_actual_manager():
    result = summarize([context(p=.6, manager="keep")], admission_margin=2., write_threshold=.7)
    row = result["per_context"][0]["admission"]
    assert row["target_admit"] is False
    assert row["predicted_admit"] is False
    assert row["manager_actual_admitted"] is False
    changed = summarize([context(p=.5, manager="keep")])
    assert changed["per_context"][0]["admission"]["predicted_admit"] is True
    assert changed["per_context"][0]["selected_operation_ids"]["manager"] == "keep"


def test_selection_never_uses_heldout_min_and_generated_never_reselects():
    result = summarize([context(held=(2., 0., 5., 3.), generated=True)])
    row = result["per_context"][0]
    assert row["admission"]["selection_a_replacement_id"] == "drop-1"
    assert row["admission"]["selection_a_advantage"] == 1.
    assert row["admission"]["heldout_advantage_same_a_replacement"] == -3.
    assert gain(result, "a_selected_replacement_vs_keep")["mean"] == -3.
    assert gain(result, "a_selected_replacement_vs_keep", "generated_mse")["mean"] == pytest.approx(-.4)
    assert gain(result, "a_selected_replacement_vs_keep", "generated_joint")["mean"] == pytest.approx(-.4)
    assert gain(result, "a_selected_replacement_vs_keep", "generated_gripper")["mean"] == pytest.approx(.4)
    assert gain(result)["ci95"] is None


def test_old_only_eviction_direction_not_candidate_or_operation_array_order():
    result = summarize([context()])
    eviction = result["overall"]["eviction"]
    for name in ("utility", "retention"):
        rank = eviction[name]["heldout"]
        assert rank["pairs_total"] == 3  # Not all four pool entries' six pairs.
        assert rank["pairwise_accuracy"]["value"] == 1.
        assert rank["top1_unique_target_accuracy"]["value"] == 1.
    result = summarize([context(utility=(-2., -1., -3., 999.))])
    assert result["overall"]["eviction"]["utility"]["heldout"]["pairwise_accuracy"]["value"] == 0.
    assert result["overall"]["eviction"]["retention"]["heldout"]["pairwise_accuracy"]["value"] == 1.


def test_ties_have_explicit_denominators_and_flat_labels_no_ranking_evidence():
    result = summarize([context(retention=(1., 1., 1., 1.))])
    rank = result["overall"]["eviction"]["retention"]["selection_a"]
    assert rank["pairs_prediction_tied_among_comparable"] == 3
    assert rank["pairwise_accuracy"]["value"] == .5
    assert rank["prediction_top1_tied_contexts"] == 1
    assert rank["top1_unique_target_accuracy"]["value"] == 0.
    flat = summarize([context(a=(1., 1., 1., 1.))])
    rank = flat["overall"]["eviction"]["retention"]["selection_a"]
    assert rank["pairs_target_tied"] == 3
    assert rank["pairwise_accuracy"]["value"] is None
    assert rank["top1_unique_target_accuracy"]["value"] is None
    assert rank["all_target_slots_tied_contexts"] == 1


def test_score_tolerance_cannot_change_forced_deployment_victim():
    row = context(retention=(1., .99, 2., 3.))
    result = summarize([row], score_tie_tolerance=.1)
    assert result["per_context"][0]["selected_operation_ids"]["retention_forced_replacement"] == "drop-1"
    assert result["per_context"][0]["eviction"]["retention"]["selection_a"]["prediction_tie_count"] == 2


def test_a_b_repeat_vs_heldout_rank_and_fixed_a_replacement_are_distinct():
    result = summarize([context(b=(2., 1., 4., 3.), held=(2., 0., 5., 3.))])
    row = result["per_context"][0]
    assert row["admission"]["selection_b_replacement_id"] == "drop-2"
    assert row["admission"]["selection_b_advantage_same_a_replacement"] == -2.
    assert result["overall"]["eviction"]["teacher_repeat_a_b"]["pairwise_accuracy"]["value"] < 1.


def test_cluster_bootstrap_deterministic_and_heldout_episode_weighting_not_frame_weighting():
    rows = [context(0, event=i+32, held=(4., 6., 2., 3.)) for i in range(8)]
    rows.append(context(1, held=(4., 6., 4., 3.), task="task_b"))
    snapshot, before_rng = deepcopy(rows), np.random.get_state()
    result = summarize(rows)
    assert gain(result)["mean"] == 0.
    assert gain(result)["ci95"] == [-1., 1.]
    assert result == summarize(list(reversed(rows)))
    assert rows == snapshot
    after_rng = np.random.get_state()
    assert before_rng[0] == after_rng[0] and np.array_equal(before_rng[1], after_rng[1])
    assert before_rng[2:] == after_rng[2:]
    assert result["overall"]["eviction"]["retention"]["heldout"]["pairwise_accuracy"]["episodes"] == 2


@pytest.mark.parametrize("mutation", [
    lambda row: row["operations"][0].update(drop_pool_index=0),
    lambda row: row["manager_scores"].update(retention=[1., 2., 3.]),
    lambda row: row["manager_scores"].update(utility=[1., float("nan"), 2., 3.]),
    lambda row: row["manager_scores"].update(write_probability=[1., 0., .5, 1.1]),
    lambda row: row["panels"]["heldout"].update(query_ids=[40]),
])
def test_bad_schema_or_panel_overlap_is_rejected(mutation):
    row = context()
    mutation(row)
    with pytest.raises(ValueError):
        summarize([row])


@pytest.mark.parametrize("kwargs", [dict(uncertain_band=-1.), dict(admission_margin=float("nan")),
    dict(write_threshold=1.1), dict(loss_tie_tolerance=True), dict(seed=-1), dict(bootstrap_samples=-1)])
def test_bad_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        summarize_writer_decisions([context()], **kwargs)


def _packet(row):
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"result": row, "result_sha256": hashlib.sha256(encoded).hexdigest(),
            "protocol_fingerprint": "fixed", "actor_state": {"model": "frozen"}}


def test_cli_immutable_packet_hashes_new_output_and_overwrite_refusal(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    for eid in range(2):
        (source/f"context-{eid:04d}.json").write_text(json.dumps(_packet(context(eid))))
    before = {path: path.read_bytes() for path in source.iterdir()}
    destination = tmp_path/"analysis"
    result = main(["--run-dir", str(source), "--output-dir", str(destination), "--bootstrap-samples", "10"])
    assert (destination/"summary.json").exists() and (destination/"per_context.csv").exists()
    assert len(result["input_packets"]) == 2
    assert {path: path.read_bytes() for path in source.iterdir()} == before
    with pytest.raises(SystemExit):
        main(["--run-dir", str(source), "--output-dir", str(destination)])
    changed = json.loads(next(source.iterdir()).read_text())
    changed["result"]["manager_scores"]["utility"][0] = 999.
    next(source.iterdir()).write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Invalid or changed"):
        load_probe_contexts(source)


def test_packet_mixed_protocol_rejected(tmp_path):
    for eid in range(2):
        packet = _packet(context(eid))
        packet["protocol_fingerprint"] = str(eid)
        (tmp_path/f"context-{eid:04d}.json").write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="single protocol"):
        load_probe_contexts(tmp_path)
