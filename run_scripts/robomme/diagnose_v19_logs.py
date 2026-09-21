#!/usr/bin/env python3
"""Bounded CPU diagnosis of existing V19 validation logs; never loads a model.

All losses and gains reuse recorded, matched query/noise evaluations. Episode
clusters are sampled within each task, retaining all queries and noise repeats;
task means retain equal weight. Output must be a new directory outside inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("baseline", "memory-off", "reader")
METRICS = ("original_flow_loss", "generated_prefix_mse", "generated_prefix_mae",
           "generated_executed_joint_mse", "generated_executed_joint_mae",
           "generated_executed_gripper_mse", "generated_executed_gripper_mae")
IDENTITY = ("task", "episode_id", "decision", "repeat", "flow_seed", "generation_seed")
GAINS = {"memory_gain": (1, 2), "AE_gain": (0, 1), "total_gain": (0, 2)}
LIMITATIONS = [
    "Offline normalized-action errors are not closed-loop robot success or proof of useful recall.",
    "Memory-off uses the SAME jointly trained checkpoint with memory reading disabled; it is NOT an independently trained AE-only model. AE_gain is only the arithmetic baseline-minus-memory-off contrast and may include learned short-context changes.",
    "Positive gains mean lower error: memory = memory-off - reader; AE = baseline - memory-off; total = baseline - reader.",
    "Original unweighted flow is comparable across arms; weighted training objectives must not be compared as if identical.",
    "Generated prefix means observed executed-prefix coordinates; joint means joint7 and gripper means gripper1. Errors use saved normalization.",
    "Percentile 95% intervals resample paired episodes within each fixed task, retaining adjacent queries and noise repeats as one cluster. They condition on these tasks and fixed noises; they do not measure training-seed uncertainty.",
    "Only four held-out episodes per task are recorded in these runs. Task intervals are coarse; stages and metrics are multiple descriptive comparisons with no multiplicity correction.",
    "Leave-one-query/episode-out renormalizes the affected task before taking the equal-task macro. If a task becomes empty the result is unavailable, never silently drops that task.",
    "Query concentration first averages all noise repeats; it does not mistake two draws from one endpoint for two independent outliers.",
    "This command reuses logs only: no GPU/model inference, optimizer, new training, checkpoint selection, or simulator execution.",
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def new_output(path, inputs):
    """Reject existing destinations, input trees, and ancestors of input trees."""
    output = Path(path).resolve()
    protected = [Path(p).resolve() for p in inputs] + [ROOT / "runs/long_memory", ROOT / "checkpoints", ROOT / "data"]
    if output.exists():
        raise FileExistsError("Use a NEW output directory; existing results are never overwritten")
    for source in protected:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"Output overlaps a protected input tree: {source}")
    return output


def load_run(path):
    root = Path(path).resolve()
    names = ("run_config.json", "provenance.json", "status.json", "last_checkpoint.json", "query_plan.json")
    config, provenance, status, last, plan = (load_json(root / name) for name in names)
    step = last.get("step")
    if type(step) is not int or step < 1 or last.get("path") != f"checkpoint-{step:06d}":
        raise ValueError("Invalid fixed-final checkpoint pointer")
    checkpoint = (root / last["path"]).resolve()
    if checkpoint.parent != root:
        raise ValueError("Final checkpoint escapes input run")
    info = load_json(checkpoint / "checkpoint.json")
    if (info.get("step") != step or info.get("config") != config
            or any(info.get("metadata", {}).get(key) != value for key, value in provenance.items())):
        raise ValueError("Actual final checkpoint metadata disagrees with recorded run")
    train = config["train"]
    coverage = provenance["full_coverage"]
    epochs, queries = train.get("epochs"), coverage.get("train_query_count")
    if (config.get("driver_variant") != "full_memory_v19" or config.get("selection") != "fixed_final_epoch"
            or type(epochs) is not int or epochs < 1 or type(queries) is not int or queries < 1
            or status.get("status") != "complete" or (root / "failure.json").exists()
            or step != train.get("max_steps") or step != status.get("step") or step != status.get("max_steps")
            or status.get("completed_epochs") != epochs or coverage.get("epochs") != epochs
            or coverage.get("total_steps") != step or coverage.get("total_query_presentations") != epochs * queries
            or status.get("processed_queries") != epochs * queries):
        raise ValueError("V19 run is not a consistent completed full-epoch final checkpoint")
    if digest({k: v for k, v in plan.items() if k != "sha256"}) != plan.get("sha256") or plan["sha256"] != provenance.get("plan_sha256"):
        raise ValueError("Saved query plan hash disagrees with actual plan/checkpoint")
    if any(plan.get(k) != value for k, value in coverage.items()):
        raise ValueError("Actual plan and checkpoint full-coverage metadata disagree")
    payload_hashes = {**info["metadata"].get("payload_sha256", {}),
                      "training_state.pt": info["metadata"].get("training_state_sha256")}
    for name in ("model.safetensors", "expert.safetensors", "training_state.pt"):
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise ValueError(f"Missing final checkpoint payload: {name}")
        if not payload_hashes.get(name) or file_hash(checkpoint / name) != payload_hashes[name]:
            raise ValueError(f"Actual final checkpoint payload hash differs: {name}")
    files = [root / name for name in names] + [checkpoint / "checkpoint.json"]
    stages = {}
    for file in sorted(root.glob("validation-*.json")):
        data = load_json(file)
        stage = data.get("step")
        if type(stage) is not int or stage < 0 or file.name != f"validation-{stage:06d}.json" or stage in stages:
            raise ValueError("Validation filename/stage identity mismatch")
        stages[stage] = data
        files.append(file)
    if not stages or min(stages) != 0 or max(stages) != step:
        raise ValueError("Validation must include actual initialization and fixed-final stage")
    return {"path": str(root), "config": config, "provenance": provenance, "status": status,
            "final_checkpoint": str(checkpoint), "final_checkpoint_metadata": info,
            "checkpoint_payloads": {p.name: {"size_bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
                                              "verified_sha256": payload_hashes.get(p.name)}
                                    for p in checkpoint.iterdir() if p.is_file()},
            "files_sha256": {str(p): file_hash(p) for p in files}, "plan": plan, "stages": stages}


def match_runs(control, prefix):
    for key in ("base_model", "cache_fingerprint", "plan_sha256", "source_sha256", "runtime", "initialization",
                "initial_shared_reader_sha256", "initial_expert_sha256", "full_coverage"):
        if not control["provenance"].get(key) or control["provenance"].get(key) != prefix["provenance"].get(key):
            raise ValueError(f"Unmatched recorded training provenance: {key}")
    if set(control["stages"]) != set(prefix["stages"]):
        raise ValueError("Arms have different validation stages")
    a, b = control["config"], prefix["config"]
    for key in a.keys() | b.keys():
        if key not in {"train", "objective"} and a.get(key) != b.get(key):
            raise ValueError(f"Training configs differ beyond tail weight: {key}")
    for section, allowed in (("train", {"output_dir", "resume", "preflight_only", "stop_after_steps", "tail_weight"}),
                             ("objective", {"tail_weight"})):
        for key in a[section].keys() | b[section].keys():
            if key not in allowed and a[section].get(key) != b[section].get(key):
                raise ValueError(f"Training configs differ beyond tail weight: {section}/{key}")
    for cfg, expected in ((a, 1.), (b, .25)):
        if cfg["train"].get("tail_weight") != expected or cfg["objective"].get("tail_weight") != expected:
            raise ValueError("Expected control tail=1 and prefix tail=0.25")


def matched_records(document, schedule):
    expected = {tuple(row[key] for key in IDENTITY) for row in schedule}
    if len(expected) != len(schedule):
        raise ValueError("Duplicate query/noise identities in saved plan")
    groups = {role: {} for role in ROLES}
    schema = None
    for row in document["records"]:
        if row.get("role") not in ROLES or not isinstance(row.get("task"), str) or not row["task"]:
            raise ValueError("Invalid role/task in records")
        identity = tuple(row[key] for key in IDENTITY)
        if any(type(row[key]) is not int or row[key] < 0 for key in IDENTITY[1:]):
            raise ValueError("Query/noise identifiers must be nonnegative integers")
        metrics = set(row) - set(IDENTITY) - {"role"}
        schema = metrics if schema is None else schema
        if metrics != schema or not set(METRICS) <= metrics:
            raise ValueError("Metric schema/keys differ between validation records")
        if any(type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0 for key in metrics):
            raise ValueError("Metrics must be finite nonnegative numbers")
        for alias in ("action_loss", "loss"):
            if alias in row and row[alias] != row["original_flow_loss"]:
                raise ValueError("Original unweighted flow disagrees with saved loss alias")
        if identity in groups[row["role"]]:
            raise ValueError("Duplicate query/noise record")
        groups[row["role"]][identity] = row
    if any(set(rows) != expected for rows in groups.values()):
        raise ValueError("Roles do not have matched saved-plan query/noise keys")
    for identity in expected:
        for key in schema:
            if key.endswith("valid_values") and len({groups[role][identity][key] for role in ROLES}) != 1:
                raise ValueError("Valid coordinate masks differ between roles")
    # Verify the serialized task/macro summaries against the original records.
    for role, rows in groups.items():
        tasks = defaultdict(list)
        for row in rows.values():
            tasks[row["task"]].append(row)
        for metric in METRICS:
            means = {task: math.fsum(row[metric] for row in items) / len(items) for task, items in tasks.items()}
            macro = math.fsum(means.values()) / len(means)
            if not math.isclose(document["summary"][role][metric], macro, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Saved task-macro summary does not match records")
            for task, value in means.items():
                if not math.isclose(document["by_task"][role][task][metric], value, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("Saved task summary does not match records")
    return groups, schema


def query_values(groups):
    """Average paired noise draws first; return [query, role, metric]."""
    queries = defaultdict(list)
    for identity in groups[ROLES[0]]:
        queries[identity[:3]].append(identity)
    counts = {len(rows) for rows in queries.values()}
    if len(counts) != 1:
        raise ValueError("Unequal query noise-repeat counts would change the saved estimand")
    keys = sorted(queries)
    values = np.array([[[np.mean([groups[role][identity][metric] for identity in queries[key]])
                         for metric in METRICS] for role in ROLES] for key in keys], dtype=np.float64)
    return keys, values, counts.pop()


def bootstrap_weights(keys, draws, seed):
    """Paired episode-cluster bootstrap; never resample noises or adjacent queries."""
    rng = np.random.default_rng(seed)
    task_indices = {task: np.array([i for i, key in enumerate(keys) if key[0] == task])
                    for task in sorted({key[0] for key in keys})}
    weights = {}
    for task, indices in task_indices.items():
        episodes = sorted({keys[i][1] for i in indices})
        membership = np.array([[float(keys[i][1] == episode) for i in indices] for episode in episodes])
        counts = rng.multinomial(len(episodes), np.full(len(episodes), 1 / len(episodes)), size=draws)
        raw = counts @ membership
        weights[task] = raw / raw.sum(axis=1, keepdims=True)
    return task_indices, weights


def estimates_and_ci(values, indices, weights):
    tasks, estimates, samples = list(indices), {}, {}
    for task in tasks:
        block = values[indices[task]]
        estimates[task] = block.mean(axis=0)
        samples[task] = np.einsum("bq,q...->b...", weights[task], block)
    estimates["__macro__"] = np.mean([estimates[task] for task in tasks], axis=0)
    samples["__macro__"] = np.mean([samples[task] for task in tasks], axis=0)
    return estimates, {task: np.quantile(sample, [.025, .975], axis=0) for task, sample in samples.items()}


def leave_out_gain(keys, gains, indices, omitted):
    remaining = {task: [i for i in entries if i not in omitted] for task, entries in indices.items()}
    if any(not entries for entries in remaining.values()):
        return None
    return np.mean([gains[entries].mean(axis=0) for entries in remaining.values()], axis=0)


def concentration(values, keys, indices):
    """Shares sum contributions to equal-task macro, after noise averaging."""
    contributions = np.empty(len(keys), dtype=np.float64)
    for entries in indices.values():
        contributions[entries] = values[entries] / (len(indices) * len(entries))
    order = np.argsort(-contributions, kind="stable")
    total = float(contributions.sum())
    return {"top1_share": float(contributions[order[:1]].sum() / total) if total else 0.,
            "top2_share": float(contributions[order[:2]].sum() / total) if total else 0.,
            "top5_share": float(contributions[order[:5]].sum() / total) if total else 0.,
            "top10_share": float(contributions[order[:10]].sum() / total) if total else 0.,
            "query_median": float(np.median(values)), "macro": total,
            "top_queries": [{"task": keys[i][0], "episode_id": keys[i][1], "decision": keys[i][2],
                             "query_error": float(values[i]), "macro_contribution": float(contributions[i])}
                            for i in order[:10]]}


def analyze(control, prefix, draws=2000, seed=20260920):
    match_runs(control, prefix)
    metric_rows, gain_rows, query_rows, leave_rows, concentration_rows, top_gain_rows = [], [], [], [], [], []
    reference_keys = reference_schema = reference_baseline = None
    shape = None
    for arm, run in (("control", control), ("prefix", prefix)):
        schedule = run["plan"]["validation_schedule"]
        for stage, document in sorted(run["stages"].items()):
            groups, schema = matched_records(document, schedule)
            keys, values, noise_count = query_values(groups)
            baseline = groups["baseline"]
            if reference_keys is None:
                reference_keys, reference_schema, reference_baseline = keys, schema, baseline
                indices, weights = bootstrap_weights(keys, draws, seed)
                shape = {"tasks": len(indices), "queries": len(keys), "episodes": len({key[:2] for key in keys}),
                         "noise_repeats_per_query": noise_count, "query_noise_records_per_role": len(baseline),
                         "per_task": {task: {"queries": len(entries), "episodes": len({keys[i][1] for i in entries})}
                                      for task, entries in indices.items()}}
            elif keys != reference_keys or schema != reference_schema or baseline != reference_baseline:
                raise ValueError("Arms/stages do not share identical query/noise baseline records and schema")
            estimates, cis = estimates_and_ci(values, indices, weights)
            gains = np.stack([values[:, left] - values[:, right] for left, right in GAINS.values()], axis=1)
            gain_estimates, gain_cis = estimates_and_ci(gains, indices, weights)
            leave_cache = {"query": [], "episode": []}
            for unit in leave_cache:
                units = [(key, {i}) for i, key in enumerate(keys)] if unit == "query" else [
                    ((task, episode, None), {i for i, key in enumerate(keys) if key[:2] == (task, episode)})
                    for task, episode in sorted({key[:2] for key in keys})]
                for key, omitted in units:
                    result = leave_out_gain(keys, gains, indices, omitted)
                    leave_cache[unit].append(result)
                    for gi, name in enumerate(GAINS):
                        for mi, metric in enumerate(METRICS):
                            leave_rows.append({"arm": arm, "stage": stage, "omitted_unit": unit, "task": key[0],
                                "episode_id": key[1], "decision": key[2], "gain": name, "metric": metric,
                                "macro_after_omission": None if result is None else float(result[gi, mi]),
                                "all_tasks_retained": result is not None})
            for task in estimates:
                selected = np.arange(len(keys)) if task == "__macro__" else indices[task]
                common = {"arm": arm, "stage": stage, "task": task, "queries": len(selected),
                          "episodes": len({keys[i][:2] for i in selected}), "noise_repeats_per_query": noise_count}
                for ri, role in enumerate(ROLES):
                    for mi, metric in enumerate(METRICS):
                        metric_rows.append({**common, "role": role, "metric": metric,
                            "estimate": float(estimates[task][ri, mi]), "ci95_low": float(cis[task][0, ri, mi]),
                            "ci95_high": float(cis[task][1, ri, mi])})
                for gi, name in enumerate(GAINS):
                    for mi, metric in enumerate(METRICS):
                        delta = gains[selected, gi, mi]
                        row = {**common, "gain": name, "metric": metric, "estimate": float(gain_estimates[task][gi, mi]),
                            "ci95_low": float(gain_cis[task][0, gi, mi]), "ci95_high": float(gain_cis[task][1, gi, mi]),
                            "queries_improved": int((delta > 0).sum()), "queries_worsened": int((delta < 0).sum()),
                            "queries_tied": int((delta == 0).sum())}
                        for unit, results in leave_cache.items():
                            available = [float(result[gi, mi]) for result in results if result is not None]
                            row[f"leave_one_{unit}_macro_min"] = min(available) if available and task == "__macro__" else None
                            row[f"leave_one_{unit}_macro_max"] = max(available) if available and task == "__macro__" else None
                        gain_rows.append(row)
            for ri, role in enumerate(ROLES):
                for mi, metric in enumerate(METRICS):
                    detail = concentration(values[:, ri, mi], keys, indices)
                    concentration_rows.append({"arm": arm, "stage": stage, "role": role, "metric": metric, **detail})
            for gi, name in enumerate(GAINS):
                for mi, metric in enumerate(METRICS):
                    contributions = np.zeros(len(keys))
                    for entries in indices.values():
                        contributions[entries] = gains[entries, gi, mi] / (len(indices) * len(entries))
                    positive = float(contributions[contributions > 0].sum())
                    net = float(contributions.sum())
                    for i, key in enumerate(keys):
                        query_rows.append({"arm": arm, "stage": stage, "task": key[0], "episode_id": key[1],
                            "decision": key[2], "gain": name, "metric": metric, "noise_repeats": noise_count,
                            "query_gain": float(gains[i, gi, mi]), "macro_contribution": float(contributions[i])})
                    for direction, order in (("gainer", np.argsort(-contributions, kind="stable")),
                                             ("loser", np.argsort(contributions, kind="stable"))):
                        filtered = [i for i in order if (contributions[i] > 0 if direction == "gainer" else contributions[i] < 0)]
                        for rank, i in enumerate(filtered[:10], 1):
                            key = keys[i]
                            top_gain_rows.append({"arm": arm, "stage": stage, "gain": name, "metric": metric,
                                "direction": direction, "rank": rank, "task": key[0], "episode_id": key[1], "decision": key[2],
                                "query_gain": float(gains[i, gi, mi]), "macro_contribution": float(contributions[i]),
                                "share_of_net_gain": float(contributions[i] / net) if net != 0 else None,
                                "share_of_positive_gain_mass": float(contributions[i] / positive) if positive > 0 else None})
    return {"shape": shape, "stage_metrics": metric_rows, "stage_gains": gain_rows, "query_gains": query_rows,
            "leave_out": leave_rows, "concentration": concentration_rows, "top_gain_queries": top_gain_rows}


def markdown(result):
    lines = ["# V19 saved-log diagnostic", "", f"Reused {result['shape']['queries']} queries across {result['shape']['episodes']} episodes, "
             f"{result['shape']['tasks']} tasks and {result['shape']['noise_repeats_per_query']} fixed noise draws per query.", "",
             "All stages and task-level values, paired intervals, and query sensitivity are in the CSV files. Final-stage task-macro results:", "",
             "| Arm | Metric | Memory gain [95% CI] | AE gain [95% CI] | Total gain [95% CI] |",
             "|---|---|---|---|---|"]
    finals = {arm: max(row["stage"] for row in result["stage_gains"] if row["arm"] == arm) for arm in ("control", "prefix")}
    final_rows = {(row["arm"], row["metric"], row["gain"]): row for row in result["stage_gains"]
                  if row["task"] == "__macro__" and row["stage"] == finals[row["arm"]]}
    for arm in finals:
        for metric in METRICS:
            cells = []
            for gain in GAINS:
                row = final_rows[arm, metric, gain]
                cells.append(f"{row['estimate']:+.6g} [{row['ci95_low']:+.6g}, {row['ci95_high']:+.6g}]")
            lines.append(f"| {arm} | {metric} | " + " | ".join(cells) + " |")
    lines += ["", "## Concentration and sensitivity", ""]
    for arm in finals:
        concentration_row = next(row for row in result["concentration"] if row["arm"] == arm and row["stage"] == finals[arm]
                                 and row["role"] == "reader" and row["metric"] == "generated_prefix_mse")
        row = final_rows[arm, "generated_prefix_mse", "memory_gain"]
        lines.append(f"- {arm}: reader generated-prefix MSE top two distinct queries contribute {concentration_row['top2_share']:.2%} of macro error; "
                     f"memory improves/worsens/ties {row['queries_improved']}/{row['queries_worsened']}/{row['queries_tied']} queries. "
                     f"Leave-one-query-out memory gain ranges from {row['leave_one_query_macro_min']:+.6g} to {row['leave_one_query_macro_max']:+.6g}.")
    lines += ["", "## Interpretation limits", ""] + [f"- {item}" for item in LIMITATIONS]
    lines += ["", "Metadata records the diagnostic Git HEAD, diagnostic source hash, input JSON hashes, recorded training config/provenance, "
              "and actual final checkpoint metadata and verified payload SHA256 hashes. Payload tensors are not loaded or evaluated by this log-only diagnostic.", ""]
    return "\n".join(lines)


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        fields = list(rows[0]) if rows else []
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", default=str(ROOT / "runs/long_memory/v19_fullcoverage_v1/control_full"))
    parser.add_argument("--prefix-run", default=str(ROOT / "runs/long_memory/v19_fullcoverage_v1/prefix_full"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args(argv)
    if not 100 <= args.bootstrap <= 20000:
        parser.error("--bootstrap must be bounded between 100 and 20000")
    output = new_output(args.output_dir, (args.control_run, args.prefix_run))
    control, prefix = load_run(args.control_run), load_run(args.prefix_run)
    result = analyze(control, prefix, args.bootstrap, args.seed)
    result["metadata"] = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv if argv is None else argv,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "diagnostic_source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)): file_hash(__file__)},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "device": "CPU; saved JSON only"},
        "bootstrap": {"draws": args.bootstrap, "seed": args.seed, "confidence": .95, "method": "paired within-task episode-cluster percentile",
                      "noise_repeat_policy": "average repeats within each query; resample full episodes retaining all queries", "task_weighting": "equal macro"},
        "runs": {arm: {key: value for key, value in run.items() if key not in {"plan", "stages"}}
                 for arm, run in (("control", control), ("prefix", prefix))}}
    result["limitations"] = LIMITATIONS
    # Recheck immutable inputs after computation, before publishing a new report.
    for run in (control, prefix):
        if any(file_hash(path) != expected for path, expected in run["files_sha256"].items()):
            raise ValueError("Input changed while diagnostic was running")
    output.mkdir(parents=True, exist_ok=False)
    for key in ("stage_metrics", "stage_gains", "query_gains", "leave_out", "concentration", "top_gain_queries"):
        write_csv(output / f"{key}.csv", result[key])
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(markdown(result))
    print(json.dumps({"output_dir": str(output), "shape": result["shape"], "bootstrap": args.bootstrap,
                      "stage_metric_rows": len(result["stage_metrics"]), "stage_gain_rows": len(result["stage_gains"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
