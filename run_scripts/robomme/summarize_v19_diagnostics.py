#!/usr/bin/env python3
"""CPU-only summary of completed V19 runtime parity and paired bank removals."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KEYS = ("task", "episode_id", "decision", "repeat", "flow_seed", "generation_seed")
CONDITIONS = ("normal-bank", "memory-off") + tuple(name for i in range(4) for name in (f"block-{i}", f"random-set-{i}"))
PRIMARY = ("action_loss", "executed_prefix_flow_loss", "generated_prefix_mse", "generated_prefix_mae",
           "generated_executed_joint_mse", "generated_executed_joint_mae", "generated_executed_gripper_mse", "generated_executed_gripper_mae")
LIMITS = [
    "These are frozen-model offline normalized errors on teacher observation trajectories, not closed-loop success or semantic recall evidence.",
    "Removal delta = error(condition) - error(normal-bank); positive means removal harmed performance. Block-minus-random is positive when removing the chronological block harmed more than its matched random removal. Random-set samples event IDs; random-block samples a contiguous block start.",
    "Memory-off disables reading in the same jointly trained checkpoint; it is not independently trained AE-only.",
    "Noise repeats are averaged within each query, then paired episodes are resampled within each task with equal task weight. Adjacent queries remain in their episode cluster.",
    "Only four episodes from one task and two fixed noise draws are available in this runtime run. Intervals are descriptive and coarse, with no multiplicity adjustment or training-seed uncertainty.",
    "A removed event contains mixed features from an overlapping source window. Its observations may remain represented in retained event windows or the current query; removing an event is not erasing all its source observations.",
    "Parity exact counts require identical dtype and values. Relative L2 is undefined for zero reference norm and is excluded from numeric ranges, with missing counts reported.",
    "Action parity CSV does not contain seed columns. Repeat identities are matched across branches; original seeds are recorded in runtime configuration, not independently proven by this CSV.",
]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def boolean(value):
    if value in (True, "True", "true"):
        return True
    if value in (False, "False", "false"):
        return False
    raise ValueError(f"Invalid boolean: {value}")


def numeric(value, optional=False):
    if optional and value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite diagnostic number")
    return result


def parity_summary(rows, recurrence=False):
    groups, seen = defaultdict(list), set()
    for row in rows:
        branch = "cached-batched-vs-sequential" if recurrence else row["branch"]
        key = (branch, row["stage"], row.get("episode_id"), int(row["decision"]))
        if key in seen:
            raise ValueError("Duplicate parity endpoint/stage")
        seen.add(key)
        groups[key[:2]].append(row)
    output = []
    for (branch, stage), group in sorted(groups.items()):
        row = {"branch": branch, "stage": stage, "endpoints": len(group),
               "decision_min": min(int(item["decision"]) for item in group), "decision_max": max(int(item["decision"]) for item in group),
               "exact_count": sum(boolean(item["exact"]) for item in group),
               "same_dtype_count": sum(boolean(item["same_dtype"]) for item in group),
               "same_shape_count": sum(boolean(item["same_shape"]) for item in group),
               "finite_count": sum(boolean(item["finite"]) for item in group),
               "reference_dtypes": sorted({item["reference_dtype"] for item in group}),
               "candidate_dtypes": sorted({item["candidate_dtype"] for item in group})}
        for metric in ("relative_l2", "max_abs"):
            values = [numeric(item[metric], optional=True) for item in group]
            available = [value for value in values if value is not None]
            row[metric + "_min"] = min(available) if available else None
            row[metric + "_max"] = max(available) if available else None
            row[metric + "_undefined_count"] = len(values) - len(available)
        output.append(row)
    return output


def action_summary(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(int(row[name]) for name in ("episode_id", "decision", "repeat"))
        if key in groups[row["branch"]]:
            raise ValueError("Duplicate branch action record")
        groups[row["branch"]][key] = row
    if "saved-cache" not in groups:
        raise ValueError("Missing saved-cache action reference")
    reference = groups["saved-cache"]
    if any(set(group) != set(reference) for group in groups.values()):
        raise ValueError("Action branches do not have matched query/repeat keys")
    output = []
    for branch, group in sorted(groups.items()):
        queries = sorted({key[:2] for key in reference})
        for episode, decision in queries:
            keys = [key for key in reference if key[:2] == (episode, decision)]
            for metric in PRIMARY:
                values = [numeric(group[key][metric]) for key in keys]
                normal = [numeric(reference[key][metric]) for key in keys]
                output.append({"branch": branch, "episode_id": episode, "decision": decision, "noise_repeats": len(keys),
                               "metric": metric, "mean_error": float(np.mean(values)),
                               "delta_from_saved_cache": float(np.mean(np.subtract(values, normal)))})
    return output


def condition_names(records):
    observed = {row["intervention"] for row in records}
    required = {"normal-bank", "memory-off"} | {f"block-{i}" for i in range(4)}
    if not required <= observed:
        raise ValueError("Missing required intervention condition")
    random_kinds = []
    for kind in ("random-set", "random-block"):
        names = {f"{kind}-{i}" for i in range(4)}
        if observed & names:
            if not names <= observed:
                raise ValueError("Incomplete matched random intervention family")
            random_kinds.append(kind)
    if not random_kinds:
        raise ValueError("Missing matched random interventions")
    names = ("normal-bank", "memory-off") + tuple(name for i in range(4)
        for name in (f"block-{i}", *(f"{kind}-{i}" for kind in random_kinds)))
    if set(names) != observed:
        raise ValueError("Unexpected intervention condition")
    return names, random_kinds


def paired_interventions(records, selection, draws=2000, seed=20260920):
    expected = {tuple(row[key] for key in KEYS) for row in selection}
    if len(expected) != len(selection):
        raise ValueError("Duplicate selected query/noise identity")
    conditions, random_kinds = condition_names(records)
    groups = {name: {} for name in conditions}
    required = set(PRIMARY)
    # Include every saved scalar action error, excluding reader diagnostics and precomputed deltas.
    metrics = sorted(key for key in records[0] if key.endswith(("_loss", "_mae", "_mse")) and not key.startswith("reader_"))
    if not required <= set(metrics):
        raise ValueError("Missing required action metrics")
    for row in records:
        name = row["intervention"]
        if name not in groups:
            raise ValueError("Unexpected intervention condition")
        key = tuple(row[field] for field in KEYS)
        if key in groups[name]:
            raise ValueError("Duplicate intervention query/noise identity")
        groups[name][key] = row
        for metric in metrics:
            numeric(row[metric])
    if any(set(group) != expected for group in groups.values()):
        raise ValueError("Intervention conditions are not matched to selection/noises")
    for key in expected:
        reference = groups["normal-bank"][key]
        for name, group in groups.items():
            row = group[key]
            for field in reference:
                if field.endswith("valid_values") and row.get(field) != reference[field]:
                    raise ValueError("Intervention masks differ")
            for metric in metrics:
                field = metric + "_delta_from_normal"
                if field in row and not math.isclose(row[field], row[metric] - reference[metric], abs_tol=1e-12, rel_tol=1e-10):
                    raise ValueError("Recorded intervention delta disagrees")
        for block in range(4):
            for kind in random_kinds:
                a, b = (groups[name][key] for name in (f"block-{block}", f"{kind}-{block}"))
                if a["removed_event_count"] != b["removed_event_count"]:
                    raise ValueError("Block/random event removal counts differ")
    for name, group in groups.items():
        fixed_specs = {}
        for key, row in group.items():
            fields = ("removed_positions", "removed_event_ids", "remaining_event_ids", "read_event_ids", "read_enabled", "removed_event_count")
            spec = {field: row[field] for field in fields if field in row}
            if key[:3] in fixed_specs and fixed_specs[key[:3]] != spec:
                raise ValueError("Intervention event selection changed across noise repeats")
            fixed_specs[key[:3]] = spec
    query_keys = sorted({key[:3] for key in expected})
    repeats = {key: sorted([identity for identity in expected if identity[:3] == key]) for key in query_keys}
    if {len(items) for items in repeats.values()} != {2}:
        raise ValueError("Expected two matched noise repeats per query")
    values = np.array([[[np.mean([groups[name][identity][metric] for identity in repeats[key]]) for metric in metrics]
                        for name in conditions] for key in query_keys])
    # One shared bootstrap draw matrix pairs all conditions and metrics.
    rng = np.random.default_rng(seed)
    task_indices = {task: [i for i, key in enumerate(query_keys) if key[0] == task] for task in sorted({key[0] for key in query_keys})}
    weights = {}
    for task, indices in task_indices.items():
        episodes = sorted({query_keys[i][1] for i in indices})
        members = np.array([[int(query_keys[i][1] == eid) for i in indices] for eid in episodes])
        counts = rng.multinomial(len(episodes), np.full(len(episodes), 1 / len(episodes)), size=draws)
        raw = counts @ members
        weights[task] = raw / raw.sum(axis=1, keepdims=True)
    comparisons = [(name, "normal-bank", "condition-minus-normal") for name in conditions if name != "normal-bank"]
    comparisons += [(f"block-{i}", f"{kind}-{i}", f"block-minus-matched-{kind}") for i in range(4) for kind in random_kinds]
    summaries, query_rows = [], []
    for condition, reference, contrast in comparisons:
        ci, ri = conditions.index(condition), conditions.index(reference)
        delta = values[:, ci] - values[:, ri]
        task_means = {task: delta[indices].mean(axis=0) for task, indices in task_indices.items()}
        task_samples = {task: weights[task] @ delta[indices] for task, indices in task_indices.items()}
        estimates = {**task_means, "__macro__": np.mean(list(task_means.values()), axis=0)}
        samples = {**task_samples, "__macro__": np.mean(list(task_samples.values()), axis=0)}
        for task, estimate in estimates.items():
            indices = list(range(len(query_keys))) if task == "__macro__" else task_indices[task]
            bounds = np.quantile(samples[task], [.025, .975], axis=0)
            for mi, metric in enumerate(metrics):
                summaries.append({"task": task, "condition": condition, "reference": reference, "contrast": contrast,
                    "metric": metric, "error_delta": float(estimate[mi]), "ci95_low": float(bounds[0, mi]), "ci95_high": float(bounds[1, mi]),
                    "queries": len(indices), "episodes": len({query_keys[i][:2] for i in indices}), "noise_repeats_per_query": 2,
                    "queries_harmed": int((delta[indices, mi] > 0).sum()), "queries_helped": int((delta[indices, mi] < 0).sum()),
                    "queries_tied": int((delta[indices, mi] == 0).sum())})
        for i, key in enumerate(query_keys):
            for mi, metric in enumerate(metrics):
                query_rows.append({"task": key[0], "episode_id": key[1], "decision": key[2], "noise_repeats": 2,
                    "condition": condition, "reference": reference, "metric": metric, "condition_error": float(values[i, ci, mi]),
                    "reference_error": float(values[i, ri, mi]), "error_delta": float(delta[i, mi])})
    return summaries, query_rows


def span(values):
    values = sorted(set(values))
    return [values[0], values[-1]] if values else None


def summarize_provenance(provenance, records):
    banks, overlaps = [], []
    conditions, _ = condition_names(records)
    for key, bank in sorted(provenance.items()):
        events = {row["event_id"]: row for row in bank["events"]}
        visible = set(bank["visible_event_ids"])
        if visible != {eid for eid, row in events.items() if row["visible"]} or len(visible) != bank["visible_event_count"]:
            raise ValueError("Visible bank provenance is inconsistent")
        source_ids = {eid for event in visible for eid in events[event]["source_event_ids"]}
        source_frames = {frame for event in visible for frame in events[event]["source_frames"]}
        query_ids = set(bank["query"]["source_event_ids"])
        banks.append({"episode_id": bank["episode_id"], "decision": bank["decision"], "representation": bank["representation"],
            "visible_event_count": len(visible), "visible_token_count": bank["visible_token_count"],
            "visible_event_range": span(visible), "visible_endpoint_frame_range": span(events[eid]["endpoint_frame"] for eid in visible),
            "visible_demo_event_count": sum(events[eid]["is_demo"] for eid in visible),
            "all_prior_demo_event_count": sum(row["is_demo"] for row in events.values()),
            "visible_source_event_range": span(source_ids), "visible_source_frame_range": span(source_frames),
            "query_source_event_ids": sorted(query_ids), "query_source_frames": bank["query"]["source_frames"],
            "visible_sources_overlap_query": sorted(source_ids & query_ids),
            "token_semantics": bank["token_semantics"], "write_timing": bank["write_timing"]})
        chosen = [row for row in records if row["episode_id"] == bank["episode_id"] and row["decision"] == bank["decision"] and row["repeat"] == 0]
        if {row["intervention"] for row in chosen} != set(conditions):
            raise ValueError("Provenance does not bind to all intervention conditions")
        for row in chosen:
            removed, retained = set(row["removed_event_ids"]), set(row["remaining_event_ids"])
            if removed & retained or removed | retained != visible:
                raise ValueError("Intervention removed/retained IDs do not partition visible bank")
            removed_sources = {eid for event in removed for eid in events[event]["source_event_ids"]}
            retained_sources = {eid for event in retained for eid in events[event]["source_event_ids"]}
            overlaps.append({"episode_id": bank["episode_id"], "decision": bank["decision"], "condition": row["intervention"],
                "read_enabled": row["read_enabled"], "removed_event_count": len(removed), "removed_event_ids": sorted(removed),
                "removed_demo_events": sum(events[eid]["is_demo"] for eid in removed),
                "removed_source_event_ids": sorted(removed_sources), "removed_source_frame_range": span(
                    frame for event in removed for frame in events[event]["source_frames"]),
                "removed_sources_overlap_retained": sorted(removed_sources & retained_sources),
                "removed_sources_overlap_query": sorted(removed_sources & query_ids),
                "removed_sources_absent_from_retained_and_query": sorted(removed_sources - retained_sources - query_ids),
                "remaining_event_count": len(retained), "read_event_count": len(row["read_event_ids"])})
    expected = {f"{row['episode_id']}:{row['decision']}" for row in records}
    if set(provenance) != expected:
        raise ValueError("Bank provenance keys do not match intervention queries")
    return banks, overlaps


def text_report(result):
    lines = ["# V19 runtime diagnostic summary", "", "Parity against saved cache (repeat branch compares against live-default):", "",
             "| Branch | Stage | Exact/endpoints | Relative L2 range | Max abs |", "|---|---|---:|---:|---:|"]
    for row in result["parity_summary"]:
        if row["stage"] in ("moment", "moment_both_quantized_bf16", "short", "encoded", "fused"):
            lines.append(f"| {row['branch']} | {row['stage']} | {row['exact_count']}/{row['endpoints']} | "
                         f"{row['relative_l2_min']:.6g}–{row['relative_l2_max']:.6g} | {row['max_abs_max']:.6g} |")
    lines += ["", "Cached batched/sequential recurrence:", ""]
    for row in result["recurrence_summary"]:
        lines.append(f"- {row['stage']}: exact {row['exact_count']}/{row['endpoints']}; maximum relative L2 {row['relative_l2_max']:.6g}; maximum absolute error {row['max_abs_max']:.6g}.")
    lines += ["", "Action errors by parity branch, averaging the two saved noise repeats:", "",
              "| Branch | Episode/query | Metric | Error | Delta from cache |", "|---|---|---|---:|---:|"]
    for row in result["branch_action_metrics"]:
        if row["metric"] in ("action_loss", "generated_prefix_mse", "generated_prefix_mae"):
            lines.append(f"| {row['branch']} | {row['episode_id']}/{row['decision']} | {row['metric']} | "
                         f"{row['mean_error']:.6g} | {row['delta_from_saved_cache']:+.6g} |")
    lines += ["", "Paired intervention error changes; positive means removal harmed (2,000 episode bootstrap draws by default):", "",
              "| Condition minus reference | Metric | Error delta [95% CI] | Queries harmed/helped/tied |",
              "|---|---|---:|---:|"]
    for row in result["intervention_summary"]:
        if row["task"] == "__macro__" and row["metric"] in ("action_loss", "generated_prefix_mse", "generated_prefix_mae"):
            lines.append(f"| {row['condition']} − {row['reference']} | {row['metric']} | {row['error_delta']:+.6g} "
                         f"[{row['ci95_low']:+.6g}, {row['ci95_high']:+.6g}] | {row['queries_harmed']}/{row['queries_helped']}/{row['queries_tied']} |")
    lines += ["", "Visible FIFO provenance:", ""]
    for row in result["bank_summary"]:
        lines.append(f"- Episode {row['episode_id']}, query {row['decision']}: visible events {row['visible_event_range']}, "
                     f"demo events {row['visible_demo_event_count']}/{row['visible_event_count']}; source frames {row['visible_source_frame_range']}; "
                     f"query source frames {row['query_source_frames']}.")
    lines += ["", "Interpretation limits:", ""] + ["- " + item for item in LIMITS]
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parity-dir", help="Completed parity run to reuse when --runtime-dir is a later skip-parity intervention run")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args(argv)
    if not 100 <= args.bootstrap <= 20000:
        parser.error("Bootstrap draws must be bounded to 100..20000")
    source, output = Path(args.runtime_dir).resolve(), Path(args.output_dir).resolve()
    parity_source = Path(args.parity_dir).resolve() if args.parity_dir else source
    if output.exists():
        raise FileExistsError("Use a NEW summary directory")
    protected = [source, parity_source, ROOT / "runs/long_memory", ROOT / "checkpoints", ROOT / "data"]
    if any(output == root or output in root.parents or root in output.parents for root in protected):
        raise ValueError("Output overlaps protected inputs")
    completed = read_json(source / "completed.json")
    if (completed.get("status") != "complete" or completed.get("input_sha256_unchanged") is not True
            or completed.get("trainable_policy_parameters") != 0):
        raise ValueError("Runtime diagnostic is not a completed frozen-model run")
    parity_completed = read_json(parity_source / "completed.json")
    if (parity_completed.get("status") != "complete" or parity_completed.get("parity_executed") is not True
            or parity_completed.get("input_sha256_unchanged") is not True or parity_completed.get("trainable_policy_parameters") != 0):
        raise ValueError("Supply a completed frozen-model parity run, using --parity-dir for skip-parity input")
    parity_names = ("parity.csv", "recurrence.csv", "parity_action_errors.csv", "parity_summary.json", "configuration.json", "completed.json")
    names = ("interventions.json", "bank_provenance.json", "completed.json", "configuration.json")
    hashes = {str(source / name): sha(source / name) for name in names}
    hashes.update({str(parity_source / name): sha(parity_source / name) for name in parity_names})
    configuration = read_json(source / "configuration.json")
    parity_configuration = read_json(parity_source / "configuration.json")
    for key in ("cache_fingerprint", "selection", "base_model"):
        if configuration.get(key) != parity_configuration.get(key):
            raise ValueError(f"Separate parity/intervention run differs in {key}")
    if configuration["args"]["checkpoint"] != parity_configuration["args"]["checkpoint"]:
        raise ValueError("Separate parity/intervention runs use different checkpoints")
    records = read_json(source / "interventions.json")
    if completed["intervention_records"] != len(records):
        raise ValueError("Completed record count differs from saved interventions")
    tables = {}
    for name in parity_names[:3]:
        with (parity_source / name).open(newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
    summary, episodes = paired_interventions(records, configuration["selection"], args.bootstrap, args.seed)
    banks, overlaps = summarize_provenance(read_json(source / "bank_provenance.json"), records)
    result = {"parity_summary": parity_summary(tables["parity.csv"]),
              "recurrence_summary": parity_summary(tables["recurrence.csv"], recurrence=True),
              "branch_action_metrics": action_summary(tables["parity_action_errors.csv"]),
              "intervention_summary": summary, "intervention_query_metrics": episodes,
              "bank_summary": banks, "removal_source_overlap": overlaps,
              "limitations": LIMITS, "metadata": {"created_utc": datetime.now(timezone.utc).isoformat(), "runtime_dir": str(source),
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__)},
                  "input_sha256": hashes, "runtime_configuration": configuration, "runtime_completion": completed,
                  "parity_dir": str(parity_source), "runtime_parity_configuration": parity_configuration,
                  "runtime_parity_metadata": read_json(parity_source / "parity_summary.json"),
                  "bootstrap": {"draws": args.bootstrap, "seed": args.seed, "confidence": .95, "unit": "paired episode within task", "task_weighting": "equal macro"}}}
    if any(sha(path) != value for path, value in hashes.items()):
        raise ValueError("Runtime files changed during summary")
    report = text_report(result)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in result.items():
        if name not in ("limitations", "metadata"):
            write_csv(output / f"{name}.csv", rows)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(report)
    print(json.dumps({"output_dir": str(output), "episodes": len(banks), "intervention_contrast_rows": len(summary)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
