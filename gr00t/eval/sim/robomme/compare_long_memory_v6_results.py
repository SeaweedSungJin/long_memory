"""Paired v6 reports: visual retrieval and read-time CVOM, not write accuracy."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .compare_long_memory_results import build_report, read_results
from .compare_long_memory_v3_results import _contrast


def completed_read_diagnostics(root, model, tasks, expected):
    """Sum FINAL read counts only from sessions with completed episode rows.

    Aborted/retried attempts may coexist in the journal; their counters are not
    summed. Missing observations are reported as missing, never inferred zero.
    """
    totals = {name: 0 for name in ("uniform", "relevant", "hybrid", "null")}
    recorded = missing = torn = 0
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
                        sessions[sid] = record.get("info", {}).get("long_memory", {}).get("read_counts")
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
            if not isinstance(counters, dict) or set(counters) != set(totals):
                raise ValueError(f"Invalid retrieval counter schema for {model}/{task}/{episode}")
            for name, value in counters.items():
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid cumulative retrieval count for {model}/{task}/{episode}/{name}")
                totals[name] += value
            recorded += 1
    count = sum(totals.values())
    return {"completed_sessions_with_diagnostics": recorded, "completed_sessions_missing_diagnostics": missing,
            "ignored_torn_final_lines": torn, "read_counts": totals, "total_action_reads": count,
            "selection_fractions": {name: value / count if count else None for name, value in totals.items()}}


def build_v6_report(run_dir, *, bootstrap_samples=5000):
    root = Path(run_dir)
    manifest = json.loads((root / "comparison_manifest.json").read_text())
    if manifest.get("trainer_variant") != "retrieval_cvom_v6":
        raise ValueError("Expected a retrieval_cvom_v6 comparison manifest")
    result, report = build_report(root, bootstrap_samples=bootstrap_samples)
    result["trainer_variant"] = "retrieval_cvom_v6"
    result["additional_comparisons"] = {}
    result["retrieval_diagnostics"] = {}
    result["model_descriptions"] = {name: model.get("description", "")
                                    for name, model in manifest["models"].items()}
    lines = [report.rstrip(), "", "V6 temporal visual retrieval / read-time CVOM comparisons"]
    for name, description in result["model_descriptions"].items():
        lines.append(f"  {name}: {description}")
    lines.append("CVOM selects a memory set when reading; this experiment has NO learned write/eviction policy.")
    if "expert-only" in manifest["models"]:
        lines.append("expert-only disables memory in the SAME Stage-1 adapted Expert; "
                     "it is NOT a separately trained no-memory control.")
    roles = set(manifest["models"])
    for left, right, explanation in (
        ("expert-only", "reader", "Same Stage-1 Expert: direct memory off -> uniform temporal visual memory on"),
        ("reader", "cvom", "Matched frozen Stage-1 reader/Expert/bridge: uniform -> read-time CVOM"),
        ("fixed", "cvom", "Identical Stage-2 weights: uniform -> CVOM retrieval; isolates learned read selection"),
        ("reader", "fixed", "Matched Stage-1/Stage-2 frozen weights with uniform retrieval (parity check)"),
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
    lines += ["", "Read selections (final cumulative counters of completed sessions)"]
    for name, model in manifest["models"].items():
        if name in ("baseline", "expert-only") or model.get("reader_mode") == "none":
            continue
        counters = completed_read_diagnostics(root, name, manifest["settings"]["tasks"],
                                              manifest["settings"]["n_episodes"])
        result["retrieval_diagnostics"][name] = counters
        counts = counters["read_counts"]
        lines.append(f"  {name}: recorded sessions={counters['completed_sessions_with_diagnostics']}; "
                     f"missing diagnostics={counters['completed_sessions_missing_diagnostics']}; "
                     f"uniform/relevant/hybrid/null={counts['uniform']}/{counts['relevant']}/"
                     f"{counts['hybrid']}/{counts['null']}")
    lines.append("Missing diagnostics are not evidence of zero reads. Null means memory bypass, not task failure.")
    return result, "\n".join(lines) + "\n"


def write_v6_report(run_dir, *, bootstrap_samples=5000):
    result, report = build_v6_report(run_dir, bootstrap_samples=bootstrap_samples)
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
