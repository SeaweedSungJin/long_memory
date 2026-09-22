#!/usr/bin/env python3
"""Audit recorded ECHO storage decisions; no model, GPU, replay or rollout.

The runtime's bank_fill describes the pre-write bank; bank_events describes
the post-write bank. Means over policy calls are observation-sampled means,
not wall-clock means or final occupancy. Uncompleted sessions are excluded.

Additional --source LABEL=MODEL_DIRECTORY arguments can compare an eventual
min_fill=capacity rollout with the current learned/FIFO records. Existing logs
cannot reconstruct that alternative trajectory: retained token contents and
counterfactual scores are not recorded. A pre-full KEEP only identifies where
the alternative rule would change the operation on the observed bank.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

VERSION = "echo_storage_telemetry_audit_v1"
DEFAULT_RUN = "runs/eval/robomme/echo_cvom_full_v1_val160_seed6"
OPERATIONS = ("append", "keep", "replace", "merge")
CONTRACT_FIELDS = ("task_id", "dataset", "seed", "n_action_steps", "max_episode_steps",
                   "memory_window", "demo_sampling", "scenario_metadata_sha256", "model_config_sha256")
IDENTITY_FIELDS = ("checkpoint_variant", "checkpoint_step", "stage", "payload_sha256",
                   "echo_checkpoint_sha256", "echo_parent_identity", "echo_source_sha256",
                   "feature_precision", "representation", "memory_off")


def digest(value):
    # JSON coerces integer histogram keys to strings. Normalize that coercion
    # before sorting so the digest survives a written/read artifact round trip.
    value = json.loads(json.dumps(value, allow_nan=False))
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite integer")
    if value != int(value) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def count_template():
    names = ["calls", "passive_calls", "execution_calls", "read_enabled_calls", "action_counter_mismatches",
             "pre_full_rejections_below_min_fill", "pre_full_rejections_at_or_above_probability_threshold"]
    names += [f"writer_{op}" for op in OPERATIONS]
    names += [f"{phase}{regime}_{kind}" for phase in ("", "passive_", "execution_")
              for regime in ("pre_full", "full") for kind in ("decisions", "rejections")]
    names += [f"{regime}_probability_below_threshold" for regime in ("pre_full", "full")]
    return Counter(dict.fromkeys(names, 0))


def episode_key(task, row):
    return (task, *(int(row[k]) for k in ("env_idx", "episode_idx", "episode_seed", "scenario_seed")))


def summarize_session(calls, complete, *, task, label, capacity, min_fill, threshold):
    """Validate complete per-call telemetry and distinguish each denominator."""
    if not calls:
        raise ValueError("Completed episode has no per-call storage telemetry")
    counts, rejected_at, probabilities = count_template(), Counter(), []
    pre_values, post_values, first_rejection, first_full = [], [], None, None
    previous_post, previous_frame = 0, -1
    for expected_call, item in enumerate(calls):
        record, source_line = item["record"], item["source_line"]
        memory = record["info"]["long_memory"]
        read = memory["read"]
        if integer(record["call"], "call") != expected_call:
            raise ValueError("Policy calls must start at zero and be contiguous")
        frame = integer(record["frame_index"], "frame_index")
        if frame <= previous_frame or int(memory["frame_index"]) != frame:
            raise ValueError("Policy observation frames must strictly increase and agree")
        if type(record["passive"]) is not bool or memory["passive"] != record["passive"]:
            raise ValueError("Passive flags disagree")
        if (int(record["episode_idx"]) != int(complete["episode_idx"])
                or int(record["episode_seed"]) != int(complete["episode_seed"])):
            raise ValueError("Policy call differs from completed episode identity")
        flags = {op: integer(read[f"writer_{op}"], f"writer_{op}") for op in OPERATIONS}
        if sum(flags.values()) != 1 or any(v not in (0, 1) for v in flags.values()):
            raise ValueError("Require exactly one append/keep/replace/merge per call")
        operation = next(op for op in OPERATIONS if flags[op])
        post = integer(read["bank_events"], "post-write bank_events")
        pre = post - flags["append"]
        if not 0 <= pre <= capacity or not 0 <= post <= capacity or pre != previous_post:
            raise ValueError("Bank continuity/capacity mismatch")
        fill = read["bank_fill"]
        if not isinstance(fill, (int, float)) or not math.isfinite(fill) or abs(fill * capacity - pre) > 1e-5:
            raise ValueError("bank_fill must equal pre-write occupancy/capacity")
        full_flag = integer(read["writer_full"], "writer_full")
        if full_flag != int(pre == capacity):
            raise ValueError("writer_full disagrees with pre-write occupancy")
        if (pre < capacity and operation not in ("append", "keep")) or (pre == capacity and operation == "append"):
            raise ValueError("Writer operation is incompatible with occupancy")
        probability = read["writer_probability"]
        if not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid write probability")
        phase = "passive" if record["passive"] else "execution"
        regime = "full" if pre == capacity else "pre_full"
        counts["calls"] += 1
        counts[f"{phase}_calls"] += 1
        counts[f"writer_{operation}"] += 1
        counts[f"{regime}_decisions"] += 1
        counts[f"{phase}_{regime}_decisions"] += 1
        counts[f"{regime}_probability_below_threshold"] += probability < threshold
        counts["read_enabled_calls"] += bool(memory["memory_read_enabled"])
        counts["action_counter_mismatches"] += int(record["executed_action_count"]) != int(memory["completed_action_count"])
        if operation == "keep":
            counts[f"{regime}_rejections"] += 1
            counts[f"{phase}_{regime}_rejections"] += 1
            if pre < capacity:
                rejected_at[pre] += 1
                counts["pre_full_rejections_below_min_fill"] += pre < min_fill
                counts["pre_full_rejections_at_or_above_probability_threshold"] += probability >= threshold
                if first_rejection is None:
                    first_rejection = {"call": expected_call, "frame": frame, "bank_events_before": pre,
                                       "probability": probability, "source_line": source_line}
        if post == capacity and first_full is None:
            first_full = {"call": expected_call, "frame": frame, "source_line": source_line}
        probabilities.append(float(probability))
        pre_values.append(pre)
        post_values.append(post)
        previous_post, previous_frame = post, frame
    row = {"model": label, "task": task, "session_id": complete["session_id"],
           **{k: int(complete[k]) for k in ("env_idx", "episode_idx", "episode_seed", "scenario_seed", "success", "steps")},
           "capacity": capacity, "effective_min_fill": min_fill, "write_threshold": threshold,
           "calls": counts["calls"], "final_bank_events": post_values[-1],
           "ever_full": first_full is not None, "first_full": first_full,
           "first_pre_full_rejection": first_rejection,
           "pre_full_rejection_by_occupancy": dict(sorted(rejected_at.items())),
           "policy_call_mean_pre_write_events": statistics.mean(pre_values),
           "policy_call_mean_post_write_events": statistics.mean(post_values),
           "pre_write_event_sum": sum(pre_values), "post_write_event_sum": sum(post_values),
           "write_probability_mean": statistics.mean(probabilities),
           "write_probability_min": min(probabilities), "write_probability_max": max(probabilities),
           "counts": dict(counts)}
    for regime in ("pre_full", "full"):
        row[f"{regime}_rejection_rate"] = rate(counts[f"{regime}_rejections"], counts[f"{regime}_decisions"])
    return row


def aggregate(rows):
    if not rows:
        raise ValueError("No completed episode telemetry")
    counts, final, rejected_at = count_template(), Counter(), Counter()
    for row in rows:
        counts.update(row["counts"])
        final[row["final_bank_events"]] += 1
        rejected_at.update({int(k): v for k, v in row["pre_full_rejection_by_occupancy"].items()})
    result = {"episodes": len(rows), "successes": sum(r["success"] for r in rows),
              "success_rate": statistics.mean(r["success"] for r in rows),
              "counts": dict(counts), "episodes_ever_full": sum(r["ever_full"] for r in rows),
              "episodes_with_pre_full_rejection": sum(r["first_pre_full_rejection"] is not None for r in rows),
              "final_bank_events_histogram": dict(sorted(final.items())),
              "episode_mean_final_bank_events": statistics.mean(r["final_bank_events"] for r in rows),
              "policy_call_pooled_mean_pre_write_events": sum(r["pre_write_event_sum"] for r in rows) / counts["calls"],
              "policy_call_pooled_mean_post_write_events": sum(r["post_write_event_sum"] for r in rows) / counts["calls"],
              "episode_macro_policy_call_mean_pre_write_events": statistics.mean(r["policy_call_mean_pre_write_events"] for r in rows),
              "episode_macro_policy_call_mean_post_write_events": statistics.mean(r["policy_call_mean_post_write_events"] for r in rows),
              "pre_full_rejection_by_occupancy": dict(sorted(rejected_at.items()))}
    for phase in ("", "passive_", "execution_"):
        for regime in ("pre_full", "full"):
            key = phase + regime
            result[key + "_rejection_rate"] = rate(counts[key + "_rejections"], counts[key + "_decisions"])
    return result


def read_model(label, directory, inputs):
    directory = Path(directory).resolve(strict=True)
    manifest_path = directory.parent / "comparison_manifest.json"

    def tracked(path):
        path = Path(path).resolve(strict=True)
        inputs[str(path)] = file_hash(path)
        return path

    manifest = json.loads(tracked(manifest_path).read_text())
    spec = manifest["models"][directory.name]
    config = spec["training_config"]["echo"]
    capacity = integer(config["capacity_events"], "capacity", 1)
    min_fill = spec.get("effective_min_fill", spec.get("min_fill_override"))
    if min_fill is None:
        min_fill, min_fill_source = config["min_fill"], "checkpoint_training_config; no runtime override recorded"
    else:
        min_fill_source = "explicit runtime model metadata"
    min_fill = integer(min_fill, "min_fill", 1)
    threshold = float(config["write_threshold"])
    if min_fill > capacity or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid capacity/admission configuration")
    settings, contracts, rows, incomplete, runtime_identity = manifest["settings"], {}, [], [], None
    tasks = settings["tasks"]
    if len(tasks) != len(set(tasks)):
        raise ValueError("Duplicate tasks in comparison manifest")
    for task in tasks:
        task_dir = directory / task
        policy = json.loads(tracked(task_dir / "policy_manifest.json").read_text())
        if policy["task_id"] != task or policy["evaluation_id"] != f"{manifest['evaluation_id']}:{directory.name}":
            raise ValueError("Policy manifest evaluation/task identity mismatch")
        contracts[task] = {k: policy[k] for k in CONTRACT_FIELDS}
        for field in ("dataset", "seed", "n_action_steps", "max_episode_steps"):
            if policy[field] != settings[field]:
                raise ValueError(f"Policy manifest benchmark setting mismatch: {field}")
        with tracked(task_dir / "simulation_results.csv").open(newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        expected = {episode_key(task, r): r for r in csv_rows}
        if len(expected) != len(csv_rows):
            raise ValueError("Duplicate completed CSV episode identity")
        sessions, completions = defaultdict(list), {}
        diagnostic = tracked(task_dir / "memory_diagnostics.jsonl")
        with diagnostic.open() as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                if record.get("kind") not in ("policy_call", "episode_complete"):
                    continue
                sid = record.get("session_id")
                if not isinstance(sid, str) or not sid:
                    raise ValueError("Missing diagnostic session identity")
                if record["kind"] == "episode_complete":
                    if sid in completions:
                        raise ValueError("Duplicate session completion")
                    completions[sid] = record
                    continue
                if sid in completions:
                    raise ValueError("Policy call after session completion")
                info = record["info"]
                identity = {k: info[k] for k in IDENTITY_FIELDS}
                if runtime_identity is None:
                    runtime_identity = identity
                elif identity != runtime_identity:
                    raise ValueError("Runtime checkpoint/source identity changed within a model")
                if (info["echo_checkpoint_sha256"] != spec["checkpoint_files_sha256"]["checkpoint.json"]
                        or info["payload_sha256"] != {k: spec["checkpoint_files_sha256"][k]
                                                     for k in ("core.safetensors", "expert.safetensors")}
                        or info["echo_source_sha256"] != spec["echo_source_sha256"]):
                    raise ValueError("Runtime checkpoint/source disagrees with manifest")
                if info.get("effective_min_fill", min_fill) != min_fill:
                    raise ValueError("Runtime effective_min_fill disagrees with model metadata")
                sessions[sid].append({"record": record, "source_line": line_number})
        observed = set()
        for sid, complete in completions.items():
            key = episode_key(task, complete)
            if key not in expected or key in observed:
                raise ValueError("Diagnostic completion missing/duplicated in CSV")
            observed.add(key)
            csv_row = expected[key]
            for field in ("success", "steps"):
                if int(complete[field]) != int(csv_row[field]):
                    raise ValueError(f"Diagnostic/CSV outcome mismatch: {field}")
            if complete["status"] != csv_row["status"] or int(complete["success"]) not in (0, 1):
                raise ValueError("Diagnostic/CSV completion status mismatch")
            row = summarize_session(sessions[sid], complete, task=task, label=label,
                                    capacity=capacity, min_fill=min_fill, threshold=threshold)
            row["diagnostic_path"] = str(diagnostic)
            rows.append(row)
        if observed != set(expected):
            raise ValueError("Completed CSV episodes lack complete per-call telemetry")
        incomplete.extend({"task": task, "session_id": sid, "calls": len(calls)}
                          for sid, calls in sessions.items() if sid not in completions)
    if not rows:
        raise ValueError("No completed episodes")
    return {"label": label, "directory": str(directory), "evaluation_id": manifest["evaluation_id"],
            "write_policy": spec["write_policy"], "capacity": capacity,
            "effective_min_fill": min_fill, "min_fill_source": min_fill_source,
            "write_threshold": threshold, "runtime_identity": runtime_identity,
            "contracts": contracts, "expected_episodes_per_task": settings["n_episodes"],
            "incomplete_sessions_excluded": incomplete, "rows": rows}


def compare_models(left, right):
    common_tasks = sorted(set(left["contracts"]) & set(right["contracts"]))
    if not common_tasks:
        raise ValueError("Models have no common benchmark tasks")
    if any(left["contracts"][t] != right["contracts"][t] for t in common_tasks):
        raise ValueError("Models differ in paired benchmark/scenario provenance")
    a = {episode_key(r["task"], r): r for r in left["rows"]}
    b = {episode_key(r["task"], r): r for r in right["rows"]}
    common = sorted(a.keys() & b.keys())
    if not common:
        raise ValueError("Models have no matching completed episodes")
    wins = sum(b[k]["success"] > a[k]["success"] for k in common)
    losses = sum(b[k]["success"] < a[k]["success"] for k in common)
    discordant = wins + losses
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant) if discordant else 1.
    paired = [{"task": k[0], "env_idx": k[1], "episode_idx": k[2], "episode_seed": k[3], "scenario_seed": k[4],
               "candidate_minus_reference_success": b[k]["success"]-a[k]["success"],
               "candidate_minus_reference_final_bank_events": b[k]["final_bank_events"]-a[k]["final_bank_events"],
               "candidate_minus_reference_policy_call_mean_pre_write_events":
                   b[k]["policy_call_mean_pre_write_events"]-a[k]["policy_call_mean_pre_write_events"]}
              for k in common]
    return {"reference": left["label"], "candidate": right["label"], "paired_episodes": len(common),
            "unmatched_reference_episodes": len(a)-len(common), "unmatched_candidate_episodes": len(b)-len(common),
            "same_actor_payloads": left["runtime_identity"]["payload_sha256"] == right["runtime_identity"]["payload_sha256"],
            "same_feature_precision": left["runtime_identity"]["feature_precision"] == right["runtime_identity"]["feature_precision"],
            "wins": wins, "losses": losses, "same": len(common)-discordant,
            "paired_success_delta": (wins-losses)/len(common), "mcnemar_exact_p": p,
            "paired_mean_final_bank_event_delta": statistics.mean(r["candidate_minus_reference_final_bank_events"] for r in paired),
            "paired_episode_mean_policy_call_pre_write_event_delta": statistics.mean(
                r["candidate_minus_reference_policy_call_mean_pre_write_events"] for r in paired),
            "interpretation": "Matched observed rollouts; occupancy differences also reflect diverging episode trajectories/lengths.",
            "episodes": paired}


def write_audit(sources, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Audit output already exists; use a new directory")
    inputs, models = {}, []
    for label, directory in sources:
        directory = Path(directory).resolve(strict=True)
        if output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError("Audit output must not overlap input directories")
        models.append(read_model(label, directory, inputs))
    if len({m["label"] for m in models}) != len(models):
        raise ValueError("Duplicate model labels")
    comparisons = [compare_models(a, b) for a, b in itertools.combinations(models, 2)]
    summary = {"version": VERSION, "models": {}, "comparisons": comparisons,
               "definitions": {
                   "pre_full_rejection_rate": "KEEP / all pre-write occupancy < capacity decisions, including APPEND and KEEP",
                   "full_rejection_rate": "KEEP / all pre-write occupancy == capacity decisions, including REPLACE/MERGE/KEEP",
                   "policy_call_means": "Observed policy-call averages; include passive demonstration calls; not elapsed-time weighting",
                   "final_occupancy": "Post-write bank_events on the last policy call of a CSV-matched completed session"},
               "min_fill_capacity_interpretation": "Every observed pre-full KEEP would locally become APPEND under min_fill=capacity. Later banks, scores, actions and task success cannot be reconstructed from these logs.",
               "limitations": [
                   "No new rollout, frozen-actor replay or policy training was performed.",
                   "Retained token contents, all-slot scores and alternative-bank outcomes are absent from runtime telemetry.",
                   "Counter agreement does not independently validate the semantic correctness of completed-action effects.",
                   "Per-call sampling does not provide wall-clock or frame-duration-weighted occupancy.",
                   "Matched outcome tests use recorded development episodes and one inference seed; not final TEST generalization."]}
    episode_rows, task_rows = [], []
    for model in models:
        rows = model["rows"]
        by_task = {task: aggregate([r for r in rows if r["task"] == task]) for task in sorted(model["contracts"])
                   if any(r["task"] == task for r in rows)}
        summary["models"][model["label"]] = {**{k: v for k, v in model.items() if k != "rows"},
            "overall": aggregate(rows), "by_task": by_task,
            "all_requested_episodes_completed": all(len([r for r in rows if r["task"] == task]) == model["expected_episodes_per_task"]
                                                     for task in model["contracts"])}
        episode_rows.extend(rows)
        task_rows.extend({"model": model["label"], "task": task, **values} for task, values in by_task.items())
    for path, expected in inputs.items():
        if file_hash(path) != expected:
            raise ValueError(f"Input changed during audit: {path}")
    manifest = {"version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "sources": [{"label": m["label"], "directory": m["directory"]} for m in models],
                "input_files_sha256": inputs, "audit_source_sha256": file_hash(__file__),
                "input_hashes_rechecked_after_analysis": True, "summary_sha256": digest(summary),
                "execution": "Python standard library only; no model/GPU/rollout", "output_files_sha256": {}}
    lines = ["# ECHO storage telemetry audit", "", "All counts use completed sessions matched to CSV episode/scenario identities.", "",
             "| Model | Episodes | Successes | Pre-full KEEP / decisions | Full KEEP / decisions | Ever full | Mean final events | Call-mean pre-write events |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, model in summary["models"].items():
        a, c = model["overall"], Counter(model["overall"]["counts"])
        lines.append(f"| {name} | {a['episodes']} | {a['successes']} | {c['pre_full_rejections']} / {c['pre_full_decisions']} | "
                     f"{c['full_rejections']} / {c['full_decisions']} | {a['episodes_ever_full']} | "
                     f"{a['episode_mean_final_bank_events']:.4f} | {a['policy_call_pooled_mean_pre_write_events']:.4f} |")
    lines += ["", "Final occupancy and sampled occupancy use different denominators. Call means include passive demonstrations; they are not wall-clock means.",
              "", "## Conditional min_fill=capacity comparison", "", summary["min_fill_capacity_interpretation"],
              "A later rollout can be added with repeated `--source LABEL=/path/to/model-directory` arguments; scenario/seed and policy contracts must match.",
              "", "## Limits", "", *[f"- {s}" for s in summary["limitations"]], "",
              "See summary.json for per-task and paired contrasts; episodes.csv for session-level evidence; audit_manifest.json for input hashes."]
    output.mkdir(parents=True)

    def exclusive_json(name, value):
        with (output/name).open("x") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    exclusive_json("summary.json", summary)
    for name, rows in (("episodes.csv", episode_rows), ("tasks.csv", task_rows)):
        fields = sorted(set().union(*(r.keys() for r in rows)))
        with (output/name).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                              for k, v in row.items()} for row in rows)
    with (output/"report.md").open("x") as handle:
        handle.write("\n".join(lines)+"\n")
    manifest["output_files_sha256"] = {name: file_hash(output/name)
                                      for name in ("summary.json", "episodes.csv", "tasks.csv", "report.md")}
    exclusive_json("audit_manifest.json", manifest)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN)
    parser.add_argument("--models", nargs="+", default=("fifo", "memory"))
    parser.add_argument("--source", action="append", metavar="LABEL=MODEL_DIRECTORY",
                        help="Repeat for models in different runs; replaces --run-dir/--models")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    sources = []
    if args.source:
        for item in args.source:
            label, separator, path = item.partition("=")
            if not separator or not label or not path:
                parser.error("--source requires LABEL=MODEL_DIRECTORY")
            sources.append((label, Path(path)))
    else:
        sources = [(name, Path(args.run_dir)/name) for name in args.models]
    summary = write_audit(sources, args.output_dir)
    for label, model in summary["models"].items():
        value, counts = model["overall"], Counter(model["overall"]["counts"])
        print(f"{label}: {value['episodes']} episodes; pre-full rejection "
              f"{counts['pre_full_rejections']}/{counts['pre_full_decisions']}; "
              f"full rejection {counts['full_rejections']}/{counts['full_decisions']}; "
              f"final-full {value['episodes_ever_full']}; no new rollout")
    print(Path(args.output_dir).resolve()/"report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
