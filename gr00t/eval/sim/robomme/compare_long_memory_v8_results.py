"""V8 paired task success and bounded event storage diagnostics.

Uses saved reports only: no simulator, model, or tensor-library dependency.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .compare_long_memory_results import build_report, read_results
from .compare_long_memory_v3_results import _contrast


def completed_event_diagnostics(root, model, tasks, expected):
    """Use only final counters of sessions matching completed CSV episodes."""
    names = ("observations_seen", "write_attempts", "appended_events", "demo_appended_events",
             "evicted_events", "retained_events", "memory_tokens")
    totals = {name: 0 for name in names}
    recorded = missing = torn = maximum_events = maximum_tokens = 0
    for task in tasks:
        rows = read_results(root / model / task / "simulation_results.csv", expected=expected)
        path = root / model / task / "memory_diagnostics.jsonl"
        sessions, completed = {}, {}
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        if not line.endswith("\n"):
                            torn += 1
                            continue
                        raise ValueError(f"Malformed complete diagnostics record: {path}")
                    sid = record.get("session_id")
                    if record.get("kind") == "policy_call":
                        sessions[sid] = record.get("info", {}).get("long_memory")
                    elif record.get("kind") == "episode_complete":
                        episode = record.get("episode_idx")
                        row = rows.get(episode)
                        if row and record.get("episode_seed") == row["episode_seed"] and record.get("success") == row["success"]:
                            completed[episode] = sid
        for episode in rows:
            counters = sessions.get(completed.get(episode))
            if counters is None:
                missing += 1
                continue
            for name in (*names, "capacity_events", "tokens_per_event"):
                value = counters.get(name)
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid cumulative {name} for {model}/{task}/{episode}")
            if (counters["write_attempts"] != counters["appended_events"]
                    or counters["appended_events"] != counters["evicted_events"] + counters["retained_events"]
                    or counters["retained_events"] > counters["capacity_events"]
                    or counters["memory_tokens"] != counters["retained_events"] * counters["tokens_per_event"]
                    or counters["demo_appended_events"] > counters["appended_events"]
                    or counters["appended_events"] > counters["observations_seen"]):
                raise ValueError("Event storage counters violate append/eviction/capacity invariants")
            for name in names:
                totals[name] += counters[name]
            maximum_events = max(maximum_events, counters["retained_events"])
            maximum_tokens = max(maximum_tokens, counters["memory_tokens"])
            recorded += 1
    return {"completed_sessions_with_diagnostics": recorded, "completed_sessions_missing_diagnostics": missing,
            "ignored_torn_final_lines": torn, "max_retained_events": maximum_events,
            "max_memory_tokens": maximum_tokens, **totals}


def build_v8_report(run_dir, *, bootstrap_samples=5000):
    root = Path(run_dir)
    manifest = json.loads((root / "comparison_manifest.json").read_text())
    if manifest.get("trainer_variant") != "event_memory_v8":
        raise ValueError("Expected an event_memory_v8 comparison manifest")
    result, report = build_report(root, bootstrap_samples=bootstrap_samples)
    result.update(trainer_variant="event_memory_v8", additional_comparisons={}, storage_diagnostics={},
                  control_training_contract={})
    descriptions = {name: model.get("description", "") for name, model in manifest["models"].items()}
    result["model_descriptions"] = descriptions
    lines = [report.rstrip(), "", "V8 learned event encoding and retrieval"]
    lines.extend(f"  {name}: {description}" for name, description in descriptions.items())
    lines += ["Memory-off disables READ in the EXACT reader checkpoint; event writes continue.",
              "AE and short-source controls are independently trained; compare their recorded training budgets.",
              "FIFO admission/eviction is deterministic; this model does not claim learned storage admission."]
    roles = set(manifest["models"])
    if "reader" in roles:
        reference = manifest["models"]["reader"].get("training_contract", {})
        for control in ("ae-control", "short-control"):
            if control not in roles:
                continue
            candidate = manifest["models"][control].get("training_contract", {})
            keys = ("plan_sha256", "processed_queries", "seed", "expert_learning_rate")
            known = all(reference.get(key) is not None and candidate.get(key) is not None for key in keys)
            match = reference == candidate if known else None
            result["control_training_contract"][control] = {"metadata_matches": match,
                "reader": reference, "control": candidate}
            lines.append(f"Training-contract metadata reader/{control}: " + (
                "MATCH (recorded query plan/count/seed/AE LR/init; not equal compute)." if match else
                "DIFFERS; this is not a matched-budget memory comparison." if known else
                "UNKNOWN; required training metadata is missing."))
    for left, right, explanation in (
        ("memory-off", "reader", "Exact same reader/AE: memory READ off -> on"),
        ("ae-control", "reader", "Independent AE-only control -> moment-event reader"),
        ("short-control", "reader", "Independent short-source -> moment-source event reader"),
    ):
        if not {left, right} <= roles:
            continue
        contrast = _contrast(root, left, right, manifest["settings"]["tasks"],
                             manifest["settings"]["n_episodes"], bootstrap_samples)
        result["additional_comparisons"][f"{left}_to_{right}"] = contrast
        lines.append(explanation)
        delta = contrast["paired_task_macro_delta"]
        if delta is None:
            lines.append("  INCOMPLETE: no matched completed episodes.")
            continue
        lines.append(f"  Delta={100*delta:+.2f}pp; N={contrast['paired_n']}; "
                     f"wins/losses/same={contrast['wins']}/{contrast['losses']}/{contrast['same']}; "
                     f"McNemar p={contrast['mcnemar_exact_p']:.6g}")
        interval = contrast["paired_task_macro_bootstrap_ci95"]
        if interval:
            lines.append(f"  95% within-task paired bootstrap CI: [{100*interval[0]:+.2f}, {100*interval[1]:+.2f}]pp")
        if not contrast["complete"]:
            lines.append("  INCOMPLETE: only matched completed episodes are included.")
    lines += ["", "Bounded event storage (final cumulative counters from completed episodes)"]
    for name, model in manifest["models"].items():
        if name in ("baseline", "ae-control") or model.get("mode") == "none":
            continue
        counters = completed_event_diagnostics(root, name, manifest["settings"]["tasks"],
                                               manifest["settings"]["n_episodes"])
        result["storage_diagnostics"][name] = counters
        lines.append(f"  {name}: recorded sessions={counters['completed_sessions_with_diagnostics']}; "
                     f"missing diagnostics={counters['completed_sessions_missing_diagnostics']}; "
                     f"appended/evicted events={counters['appended_events']}/{counters['evicted_events']}; "
                     f"max retained events/tokens={counters['max_retained_events']}/{counters['max_memory_tokens']}")
    lines += ["Capacity counts events: 128 events x 4 tokens = 512 tokens; runtime retained counts can be smaller.",
              "FIFO overflow may evict demonstration events. Encoded event counts do not establish semantic recall accuracy.",
              "Missing diagnostics are not evidence of zero memory activity."]
    return result, "\n".join(lines) + "\n"


def write_v8_report(run_dir, *, bootstrap_samples=5000):
    result, report = build_v8_report(run_dir, bootstrap_samples=bootstrap_samples)
    root = Path(run_dir)
    for name, payload in {"comparison_summary.json": json.dumps(result, indent=2, allow_nan=False) + "\n",
                          "comparison_summary.txt": report}.items():
        descriptor, filename = tempfile.mkstemp(prefix=f".{name}-", dir=root)
        temporary = Path(filename)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(root / name)
        finally:
            temporary.unlink(missing_ok=True)
    return result, report
