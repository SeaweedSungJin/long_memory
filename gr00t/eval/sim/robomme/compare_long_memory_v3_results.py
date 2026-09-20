"""V3 reporting: legacy paired statistics plus reader/writer-isolation contrasts.

No simulator, torch or model import is required to read an existing run. Raw
diagnostics remain in per-task journals; this report never confuses missing
episodes or program errors with simulator task failure.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .compare_long_memory_results import (
    build_report, mcnemar_exact, paired_differences, paired_macro_bootstrap, read_results,
)


def completed_storage_diagnostics(root, model, tasks, expected):
    """Sum each completed session's FINAL counters, not every cumulative call.

    A crashed/retried episode may leave several sessions in the journal. Only a
    session with an episode_complete marker matching a completed CSV row counts.
    A torn final journal line is ignored; a malformed complete line fails closed.
    The final executed interval has no later policy call, so these counters cover
    decisions actually made, not an imagined terminal memory write.
    """
    names = ("attempted_writes", "accepted_writes", "forced_writes", "replaced",
             "learned_attempted", "learned_accepted", "learned_rejected", "learned_replaced")
    totals = {name: 0 for name in names}
    episodes, missing, torn = 0, 0, 0
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
                        raise ValueError(f"Malformed diagnostics record: {path}")
                    sid = record.get("session_id")
                    if record.get("kind") == "policy_call":
                        sessions[sid] = record.get("info", {}).get("long_memory", {})
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
            episodes += 1
            for name in names:
                value = counters.get(name, 0)
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid cumulative {name} for {model}/{task}/{episode}")
                totals[name] += value
    return {"completed_sessions_with_diagnostics": episodes, "completed_sessions_missing_diagnostics": missing,
            "ignored_torn_final_lines": torn, **totals,
            "learned_write_rate": totals["learned_accepted"] / totals["learned_attempted"]
                if totals["learned_attempted"] else None}


def _contrast(root, left, right, tasks, expected, bootstrap_samples):
    groups, per_task = [], {}
    for task in tasks:
        values = paired_differences(read_results(root / left / task / "simulation_results.csv", expected=expected),
                                    read_results(root / right / task / "simulation_results.csv", expected=expected))
        groups.append(values)
        per_task[task] = {"paired_n": len(values), "wins": values.count(1), "losses": values.count(-1),
                          "same": values.count(0), "paired_delta": sum(values) / len(values) if values else None}
    nonempty = [group for group in groups if group]
    all_values = [value for group in groups for value in group]
    wins, losses = all_values.count(1), all_values.count(-1)
    return {"reference": left, "candidate": right, "tasks": per_task,
            "paired_n": len(all_values), "wins": wins, "losses": losses, "same": all_values.count(0),
            "paired_task_macro_delta": sum(sum(g) / len(g) for g in nonempty) / len(nonempty) if nonempty else None,
            "paired_task_macro_bootstrap_ci95": None if any(len(g) == 1 for g in groups)
                else paired_macro_bootstrap(groups, samples=bootstrap_samples),
            "mcnemar_exact_p": mcnemar_exact(wins, losses),
            "complete": all(len(group) == expected for group in groups)}


def build_v3_report(run_dir, *, bootstrap_samples=5000):
    root = Path(run_dir)
    manifest = json.loads((root / "comparison_manifest.json").read_text())
    if manifest.get("trainer_variant") != "action_value_v3":
        raise ValueError("Expected an action_value_v3 comparison manifest")
    # The shared report first validates scenario identity, episode seeds and
    # ownership for EVERY model before any extra contrasts are calculated.
    result, report = build_report(root, bootstrap_samples=bootstrap_samples)
    result["trainer_variant"] = "action_value_v3"
    result["additional_comparisons"] = {}
    result["storage_diagnostics"] = {}
    lines = [report.rstrip(), "", "V3 storage/read isolation comparisons"]
    roles = set(manifest["models"])
    for left, right, explanation in (
        ("reader", "memory", "Stage 1 -> Stage 2: includes reader changes and learned storage"),
        ("fifo", "memory", "Same Stage 2 weights: FIFO -> learned storage; isolates storage policy"),
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
    if not result["additional_comparisons"]:
        lines.append("Select reader + memory and/or fifo + memory to obtain these contrasts.")
    lines += ["", "Storage decisions (final cumulative counters from completed-session journals)"]
    for model in manifest["models"]:
        if model == "baseline":
            continue
        counters = completed_storage_diagnostics(root, model, manifest["settings"]["tasks"],
                                                 manifest["settings"]["n_episodes"])
        result["storage_diagnostics"][model] = counters
        lines.append(f"  {model}: recorded sessions={counters['completed_sessions_with_diagnostics']}; "
                     f"missing diagnostics={counters['completed_sessions_missing_diagnostics']}; "
                     f"learned accept/reject/replace={counters['learned_accepted']}/"
                     f"{counters['learned_rejected']}/{counters['learned_replaced']}; "
                     f"forced writes={counters['forced_writes']}")
    lines.append("FIFO has no learned decisions; missing diagnostics are not evidence of zero writes.")
    lines.append("Raw executed-control/memory decisions are recorded in each task's memory_diagnostics.jsonl.")
    return result, "\n".join(lines) + "\n"


def write_v3_report(run_dir, *, bootstrap_samples=5000):
    result, report = build_v3_report(run_dir, bootstrap_samples=bootstrap_samples)
    root = Path(run_dir)
    payloads = {"comparison_summary.json": json.dumps(result, indent=2, allow_nan=False) + "\n",
                "comparison_summary.txt": report}
    for name, payload in payloads.items():
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
