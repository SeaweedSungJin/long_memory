"""Stored-telemetry contracts; no actor loading, Torch, GPU or robot replay."""
from copy import deepcopy
import csv
import json

import pytest

from run_scripts.robomme.audit_echo_storage import (
    IDENTITY_FIELDS, aggregate, compare_models, digest, read_model, summarize_session, write_audit,
)


def trace(*, success=1, operations=("append", "keep", "append", "append", "append", "keep", "replace")):
    identity = {key: "identity" for key in IDENTITY_FIELDS}
    identity.update(payload_sha256={"core.safetensors": "core", "expert.safetensors": "expert"},
                    echo_checkpoint_sha256="checkpoint", echo_source_sha256={"core.py": "source"},
                    feature_precision="native")
    complete = {"kind": "episode_complete", "session_id": "T:0:session", "env_idx": 0,
                "episode_idx": 0, "episode_seed": 5, "scenario_seed": 11,
                "success": success, "status": "success" if success else "fail", "steps": 100}
    records, n = [], 0
    for i, op in enumerate(operations):
        read = {f"writer_{name}": int(op == name) for name in ("append", "keep", "replace", "merge")}
        read.update(bank_fill=n/4, writer_full=int(n == 4),
                    writer_probability=.1 if op == "keep" else .9)
        n += op == "append"
        read["bank_events"] = n
        record = {"kind": "policy_call", "session_id": complete["session_id"], "call": i,
                  "episode_idx": 0, "episode_seed": 5, "frame_index": i*16,
                  "passive": i < 2, "executed_action_count": 0 if i < 2 else 16,
                  "info": deepcopy(identity)}
        record["info"]["long_memory"] = {"frame_index": i*16, "passive": i < 2,
            "completed_action_count": record["executed_action_count"],
            "memory_read_enabled": i >= 2, "read": read}
        records.append({"record": record, "source_line": i+1})
    return records, complete


def session(operations=None):
    calls, complete = trace(**({"operations": operations} if operations is not None else {}))
    return summarize_session(calls, complete, task="T", label="memory", capacity=4, min_fill=1, threshold=.5)


def fixture_run(tmp_path):
    root = tmp_path/"input"
    policies = ("fifo", "memory")
    metadata = {"evaluation_id": "evaluation", "settings": {
        "tasks": ["T"], "dataset": "val", "seed": 6, "n_action_steps": 16,
        "max_episode_steps": 1000, "n_episodes": 1}, "models": {}}
    for label in policies:
        directory = root/label/"T"
        directory.mkdir(parents=True)
        metadata["models"][label] = {
            "training_config": {"echo": {"capacity_events": 4, "min_fill": 1, "write_threshold": .5}},
            "write_policy": label, "checkpoint_files_sha256": {
                "core.safetensors": "core", "expert.safetensors": "expert", "checkpoint.json": "checkpoint"},
            "echo_source_sha256": {"core.py": "source"}}
        policy = {**metadata["settings"], "task_id": "T", "evaluation_id": f"evaluation:{label}",
                  "memory_window": 4, "demo_sampling": "backward", "scenario_metadata_sha256": "scenarios",
                  "model_config_sha256": "base"}
        (directory/"policy_manifest.json").write_text(json.dumps(policy))
        calls, complete = trace(success=int(label == "memory"))
        if label == "fifo":
            calls, complete = trace(success=0, operations=("append", "append", "append", "append", "replace"))
        (directory/"memory_diagnostics.jsonl").write_text("\n".join(
            json.dumps(x) for x in [*[item["record"] for item in calls], complete])+"\n")
        with (directory/"simulation_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[k for k in complete if k not in ("kind", "session_id")])
            writer.writeheader()
            writer.writerow({k: v for k, v in complete.items() if k not in ("kind", "session_id")})
    (root/"comparison_manifest.json").write_text(json.dumps(metadata))
    return root


def test_pre_full_denominator_and_occupancy_semantics():
    row = session()
    assert row["counts"]["pre_full_rejections"] == 1
    assert row["counts"]["pre_full_decisions"] == 5
    assert row["pre_full_rejection_rate"] == .2
    assert row["full_rejection_rate"] == .5
    assert row["counts"]["passive_pre_full_rejections"] == 1
    assert row["final_bank_events"] == 4
    assert row["policy_call_mean_pre_write_events"] == pytest.approx(15/7)
    assert row["policy_call_mean_post_write_events"] == pytest.approx(19/7)
    assert row["first_full"]["call"] == 4
    assert row["first_pre_full_rejection"]["bank_events_before"] == 1
    assert row["first_pre_full_rejection"]["source_line"] == 2
    assert row["counts"]["action_counter_mismatches"] == 0


def test_episode_final_and_call_weighted_means_do_not_collapse():
    rows = [session(), session(("append", "append"))]
    result = aggregate(rows)
    assert result["episode_mean_final_bank_events"] == 3
    assert result["policy_call_pooled_mean_pre_write_events"] == pytest.approx(16/9)
    assert result["episode_macro_policy_call_mean_pre_write_events"] == pytest.approx((15/7+.5)/2)
    assert result["final_bank_events_histogram"] == {2: 1, 4: 1}
    assert aggregate([rows[1]])["full_rejection_rate"] is None


@pytest.mark.parametrize("corruption", ("post_as_pre", "missing_call", "bad_full", "extra_operation", "nan_probability"))
def test_telemetry_inconsistency_fails_closed(corruption):
    calls, complete = trace()
    if corruption == "post_as_pre":
        calls[0]["record"]["info"]["long_memory"]["read"]["bank_fill"] = .25
    elif corruption == "missing_call":
        calls.pop(1)
    elif corruption == "bad_full":
        calls[0]["record"]["info"]["long_memory"]["read"]["writer_full"] = 1
    elif corruption == "extra_operation":
        calls[0]["record"]["info"]["long_memory"]["read"]["writer_keep"] = 1
    else:
        calls[0]["record"]["info"]["long_memory"]["read"]["writer_probability"] = float("nan")
    with pytest.raises(ValueError):
        summarize_session(calls, complete, task="T", label="memory", capacity=4, min_fill=1, threshold=.5)


def test_complete_csv_pairing_and_runtime_identity_are_required(tmp_path):
    root = fixture_run(tmp_path)
    inputs = {}
    model = read_model("memory", root/"memory", inputs)
    assert len(model["rows"]) == 1 and len(inputs) == 4
    path = root/"memory"/"T"/"memory_diagnostics.jsonl"
    records = [json.loads(x) for x in path.read_text().splitlines()]
    records[-1]["success"] = 0
    path.write_text("\n".join(json.dumps(x) for x in records))
    with pytest.raises(ValueError, match="outcome mismatch"):
        read_model("memory", root/"memory", {})
    records[-1]["success"] = 1
    records[1]["info"]["echo_checkpoint_sha256"] = "other"
    path.write_text("\n".join(json.dumps(x) for x in records))
    with pytest.raises(ValueError, match="identity changed"):
        read_model("memory", root/"memory", {})


def test_incomplete_sessions_are_not_in_occupancy_or_success_denominators(tmp_path):
    root = fixture_run(tmp_path)
    path = root/"memory"/"T"/"memory_diagnostics.jsonl"
    call = trace()[0][0]["record"]
    call["session_id"] = "interrupted"
    with path.open("a") as handle:
        handle.write(json.dumps(call)+"\n")
    model = read_model("memory", root/"memory", {})
    assert len(model["rows"]) == 1
    assert model["incomplete_sessions_excluded"] == [{"task": "T", "session_id": "interrupted", "calls": 1}]


def test_cross_run_pairing_is_exact_and_reports_unmatched_episodes(tmp_path):
    root = fixture_run(tmp_path)
    a, b = [read_model(label, root/label, {}) for label in ("fifo", "memory")]
    result = compare_models(a, b)
    assert result["paired_episodes"] == 1 and result["wins"] == 1 and result["losses"] == 0
    assert result["same_actor_payloads"]
    extra = deepcopy(b["rows"][0])
    extra["episode_idx"] = 1
    b["rows"].append(extra)
    assert compare_models(a, b)["unmatched_candidate_episodes"] == 1
    b["contracts"]["T"]["scenario_metadata_sha256"] = "other"
    with pytest.raises(ValueError, match="provenance"):
        compare_models(a, b)


def test_audit_hashes_inputs_and_never_overwrites_output(tmp_path):
    root = fixture_run(tmp_path)
    output = tmp_path/"audit"
    result = write_audit([("fifo", root/"fifo"), ("memory", root/"memory")], output)
    manifest = json.loads((output/"audit_manifest.json").read_text())
    assert len(manifest["input_files_sha256"]) == 7
    assert manifest["input_hashes_rechecked_after_analysis"]
    assert manifest["summary_sha256"] == digest(json.loads((output/"summary.json").read_text()))
    assert result["models"]["memory"]["overall"]["pre_full_rejection_rate"] == .2
    before = (output/"summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_audit([("memory", root/"memory")], output)
    assert (output/"summary.json").read_bytes() == before


def test_histogram_digest_survives_integer_json_key_coercion():
    histogram = {"occupancy": {2: 3, 10: 7, 32: 1}}
    assert digest(histogram) == digest(json.loads(json.dumps(histogram)))
