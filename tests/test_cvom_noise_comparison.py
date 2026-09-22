"""CPU-only tests for strict nested-noise teacher qualification comparisons."""
from copy import deepcopy
import json

import pytest

from run_scripts.robomme import compare_cvom_budget_noise as comparison


FLOW_METRICS = ("flow_prefix", "flow_unweighted", "flow_weighted", "flow_joint", "flow_gripper")


def _result(episode, noise_samples, *, stable=False):
    operations = [
        {"id": "keep", "kind": "keep", "drop_event_id": 32, "drop_pool_index": 2},
        {"id": "fifo", "kind": "fifo", "drop_event_id": 0, "drop_pool_index": 0},
        {"id": "drop-1", "kind": "drop", "drop_event_id": 1, "drop_pool_index": 1},
    ]
    result = {"episode_id": episode, "event": 32, "task": "synthetic_task",
        "context_id": f"val-{episode}-32", "operations": operations,
        "manager_operation_id": "fifo", "manager_scores": {
            "utility": [.1, .2, .3], "retention": [.4, .5, .6], "write_probability": [.2, .7, .8]},
        "bank_event_ids": [[0], [1]], "pool_provenance": [
            {"event_ids": [0], "first_frame": 0, "last_frame": 0, "is_demo": False},
            {"event_ids": [1], "first_frame": 16, "last_frame": 16, "is_demo": False},
            {"event_ids": [32], "first_frame": 512, "last_frame": 512, "is_demo": False}],
        "panels": {}}
    for panel_name in ("selection_a", "selection_b", "heldout"):
        queries = [48, 52] if panel_name == "heldout" else [40, 44]
        panel = {"query_ids": queries, "draw_ids": [], **{metric: [] for metric in FLOW_METRICS}}
        for query in queries:
            for noise in range(noise_samples):
                panel["draw_ids"].append(f"{episode}:{panel_name}:{query}:noise-{noise}")
                if stable:
                    values = [4., 3., 1.]
                elif panel_name == "heldout":
                    # Heldout KEEP is best, yet selection A must not choose it.
                    values = [1., 9., 1.] if noise < 2 else [1., 4., 8.]
                else:
                    values = [10., 8., 4.] if noise < 2 else [10., 1., 9.]
                for metric_index, metric in enumerate(FLOW_METRICS):
                    panel[metric].append([value*(metric_index+1) for value in values])
        if panel_name == "heldout":
            panel["generation_draws"] = [{"query": query, "seed": 600+query} for query in queries]
            generated = [.4, .3, .1] if stable else [.1, .2, .8]
            for index, metric in enumerate(("generated_mse", "generated_joint", "generated_gripper")):
                panel[metric] = [[value*(index+1) for value in generated] for _ in queries]
        result["panels"][panel_name] = panel
    return result


def _pair(episodes=2, *, stable=False):
    common = {"settings": {"noise_samples": 2, "future_samples": 2, "generation_repeats": 1},
        "checkpoint": {"path": "frozen/checkpoint", "sha256": "weights-a"},
        "source_sha256": {"frozen_probe.py": "source-a"},
        "plan": [{"episode_id": episode, "event": 32, "selection_queries": [40, 44],
                  "heldout_queries": [48, 52]} for episode in range(episodes)],
        "seed": 6, "cache": "fixed-cache"}
    low, high = deepcopy(common), deepcopy(common)
    high["settings"]["noise_samples"] = 8
    low["fingerprint"], high["fingerprint"] = "low", "high"
    packets = []
    for count in (2, 8):
        current = []
        for episode in range(episodes):
            result = _result(episode, count, stable=stable)
            current.append({"row_sha256": f"row-{episode}", "actor_state": {"core": "core-sha", "expert": "ae-sha"},
                "result": result, "result_sha256": comparison.digest(result)})
        packets.append(current)
    return low, high, packets[0], packets[1]


def test_only_noise_count_changes_and_query_major_shared_draws_are_matched_by_id():
    args = _pair()
    before = deepcopy(args)
    audit = comparison.validate_noise_pair(*args)
    assert audit == {"noise_samples": [2, 8], "same_actor_context_and_future": True,
        "nested_flow_draws_bit_exact": True, "generation_metrics_bit_exact": True}
    left = args[2][0]["result"]["panels"]["selection_a"]["draw_ids"]
    right = args[3][0]["result"]["panels"]["selection_a"]["draw_ids"]
    assert left != right[:len(left)]
    assert set(left) <= set(right)
    assert args == before


@pytest.mark.parametrize("change", [
    lambda high: high["checkpoint"].update(sha256="different-weights"),
    lambda high: high["checkpoint"].update(path="different/checkpoint"),
    lambda high: high["source_sha256"].update({"frozen_probe.py": "different-source"}),
    lambda high: high["plan"][0].update(selection_queries=[41, 44]),
    lambda high: high["settings"].update(future_samples=3),
    lambda high: high["settings"].update(generation_repeats=2),
    lambda high: high.update(seed=7),
    lambda high: high.update(cache="different-cache"),
    lambda high: high["settings"].update(noise_samples=2),
])
def test_reject_protocol_changes_other_than_increased_noise(change):
    low, high, lp, hp = _pair()
    change(high)
    with pytest.raises(ValueError, match="Only noise_samples"):
        comparison.validate_noise_pair(low, high, lp, hp)


@pytest.mark.parametrize("change", [
    lambda packet: packet.update(row_sha256="different-row"),
    lambda packet: packet["actor_state"].update(expert="different-ae"),
    lambda packet: packet["result"].update(manager_operation_id="keep"),
    lambda packet: packet["result"]["manager_scores"]["utility"].__setitem__(0, 999.),
    lambda packet: packet["result"]["bank_event_ids"].__setitem__(0, [9]),
    lambda packet: packet["result"]["pool_provenance"][0].update(last_frame=1),
    lambda packet: packet["result"]["operations"][0].update(drop_event_id=99),
    lambda packet: packet["result"]["panels"]["selection_a"].update(query_ids=[41, 44]),
])
def test_reject_actor_storage_current_inputs_or_query_changes(change):
    low, high, lp, hp = _pair()
    change(hp[0])
    with pytest.raises(ValueError):
        comparison.validate_noise_pair(low, high, lp, hp)


@pytest.mark.parametrize("panel", ["selection_a", "selection_b", "heldout"])
@pytest.mark.parametrize("metric", FLOW_METRICS)
def test_every_shared_raw_flow_value_must_be_exact(panel, metric):
    low, high, lp, hp = _pair()
    hp[0]["result"]["panels"][panel][metric][0][1] += 1e-10
    with pytest.raises(ValueError, match="Shared flow draw did not reproduce exactly"):
        comparison.validate_noise_pair(low, high, lp, hp)


def test_missing_nested_draw_fails_even_if_all_loss_values_match():
    low, high, lp, hp = _pair()
    hp[0]["result"]["panels"]["selection_a"]["draw_ids"][0] = "different-noise-identity"
    with pytest.raises(ValueError, match="not nested"):
        comparison.validate_noise_pair(low, high, lp, hp)


@pytest.mark.parametrize("metric", ["generation_draws", "generated_mse", "generated_joint", "generated_gripper"])
def test_generation_evidence_must_be_identical_and_present(metric):
    low, high, lp, hp = _pair()
    hp[0]["result"]["panels"]["heldout"].pop(metric)
    with pytest.raises(ValueError, match="identical actual-generation"):
        comparison.validate_noise_pair(low, high, lp, hp)
    low, high, lp, hp = _pair()
    panel = hp[0]["result"]["panels"]["heldout"]
    if metric == "generation_draws":
        panel[metric][0]["seed"] += 1
    else:
        panel[metric][0][0] += 1e-10
    with pytest.raises(ValueError, match="identical actual-generation"):
        comparison.validate_noise_pair(low, high, lp, hp)


def test_both_preselected_choices_evaluated_on_same_larger_heldout_without_heldout_selection():
    args = _pair()
    snapshot = deepcopy(args)
    result = comparison.compare(*args)
    for row in result["per_context_comparison"]:
        assert row["noise2_choice"] == "drop-1"
        assert row["noise8_choice"] == "fifo"
        # Noise8 heldout: KEEP=1, FIFO=5.25, drop1=6.25. Both choices
        # selected on their respective A panels, not heldout's best KEEP.
        assert row["flow_prefix_selection_gain"] == 1.
        assert row["generated_mse_selection_gain"] == pytest.approx(.6)
    assert result["noise2"]["per_context"][0]["heldout"]["flow_prefix"]["losses"]["selected_a"] == 1.
    assert result["noise2_choices_on_noise8_heldout"]["per_context"][0]["heldout"]["flow_prefix"]["losses"]["selected_a"] == 6.25
    assert result["noise8"]["per_context"][0]["heldout"]["flow_prefix"]["losses"]["selected_a"] == 5.25
    assert result["selection_comparison_same_heldout"]["flow_prefix"]["mean"] == 1.
    assert args == snapshot
    json.dumps(result, allow_nan=False)


def test_gate_fails_closed_for_too_few_episodes_or_nonpositive_independent_gain():
    result = comparison.compare(*_pair())
    gate = result["teacher_gate"]
    assert gate["eligible_for_manual_training_review"] is False
    assert gate["automatic_training_allowed"] is False
    assert "at_least_8_episodes" in gate["failed_criteria"]
    assert "flow_prefix_heldout_gain_vs_fifo_ci_positive" in gate["failed_criteria"]
    assert result["selection_comparison_same_heldout"]["flow_prefix"]["mean"] > 0


def test_even_positive_exploratory_gate_never_authorizes_automatic_training():
    result = comparison.compare(*_pair(8, stable=True))
    gate = result["teacher_gate"]
    assert all(gate["criteria"].values())
    assert gate["eligible_for_manual_training_review"] is True
    assert gate["automatic_training_allowed"] is False
    assert "explicit human" in gate["interpretation"]


def test_gate_requires_generated_gain_not_only_positive_flow_gain():
    low, high, lp, hp = _pair(8, stable=True)
    for packets in (lp, hp):
        for packet in packets:
            # Flow still selects drop1, but actual generation is worse there.
            packet["result"]["panels"]["heldout"]["generated_mse"] = [[.1, .2, .9], [.1, .2, .9]]
    gate = comparison.compare(low, high, lp, hp)["teacher_gate"]
    assert gate["criteria"]["flow_prefix_heldout_gain_vs_fifo_ci_positive"] is True
    assert gate["criteria"]["generated_mse_heldout_gain_vs_fifo_ci_positive"] is False
    assert gate["eligible_for_manual_training_review"] is False
    assert gate["automatic_training_allowed"] is False


def test_flat_repeat_targets_have_no_rank_evidence_and_gate_fails():
    low, high, lp, hp = _pair(8, stable=True)
    for packets in (lp, hp):
        for packet in packets:
            for panel in packet["result"]["panels"].values():
                for metric in FLOW_METRICS:
                    panel[metric] = [[1., 1., 1.] for _ in panel[metric]]
    result = comparison.compare(low, high, lp, hp)
    assert result["noise8"]["stability"]["pairwise_repeat_pooled_episode_bootstrap"]["ci95"] is None
    assert result["teacher_gate"]["criteria"]["repeat_rank_ci_above_chance"] is False
    assert result["teacher_gate"]["eligible_for_manual_training_review"] is False


def _verified_loader_fixture(tmp_path, monkeypatch):
    protocol = {"plan": [{"episode_id": 1}], "source_sha256": {"frozen_probe.py": "expected"}}
    protocol["fingerprint"] = comparison.digest(protocol)
    (tmp_path/"protocol.json").write_text(json.dumps(protocol))
    packets = [{"synthetic": "packet"}]
    monkeypatch.setattr(comparison, "read_packets", lambda output, actual: packets)
    monkeypatch.setattr(comparison, "matching_verification", lambda *args: "verification.json")
    monkeypatch.setattr(comparison, "file_hash", lambda path: "expected")
    return protocol, packets


def test_loader_requires_complete_verification_and_unchanged_actual_source(tmp_path, monkeypatch):
    protocol, packets = _verified_loader_fixture(tmp_path, monkeypatch)
    assert comparison.load_verified_run(tmp_path) == (protocol, packets)
    monkeypatch.setattr(comparison, "file_hash", lambda path: "changed")
    with pytest.raises(ValueError, match="Probe source changed"):
        comparison.load_verified_run(tmp_path)
    monkeypatch.setattr(comparison, "file_hash", lambda path: "expected")
    monkeypatch.setattr(comparison, "matching_verification", lambda *args: None)
    with pytest.raises(ValueError, match="complete, invariance-verified"):
        comparison.load_verified_run(tmp_path)


def test_loader_rejects_protocol_tampering_and_partial_packets(tmp_path, monkeypatch):
    protocol, _ = _verified_loader_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(comparison, "read_packets", lambda *args: [])
    with pytest.raises(ValueError, match="complete, invariance-verified"):
        comparison.load_verified_run(tmp_path)
    protocol["plan"].append({"episode_id": 2})
    (tmp_path/"protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Modified probe protocol"):
        comparison.load_verified_run(tmp_path)
