"""V7 paired task success and recurrent KEEP/UPDATE diagnostics.

The independently trained AE-only control is separate from READ-off inference.
No simulator, tensor library or model is needed to regenerate saved reports.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .compare_long_memory_results import build_report, read_results
from .compare_long_memory_v3_results import _contrast


def completed_write_diagnostics(root, model, tasks, expected):
    """Final completed-session counters only; failed/retried attempts excluded."""
    names = ("observations_seen", "write_attempts", "updates", "keeps", "demo_updates", "demo_keeps")
    totals = {name: 0 for name in names}
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
            for name in names:
                value = counters.get(name)
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid cumulative {name} for {model}/{task}/{episode}")
            if counters["updates"] + counters["keeps"] != counters["write_attempts"]:
                raise ValueError("Recurrent update/keep counters do not sum to attempts")
            for name in names:
                totals[name] += counters[name]
            recorded += 1
    return {"completed_sessions_with_diagnostics": recorded, "completed_sessions_missing_diagnostics": missing,
            "ignored_torn_final_lines": torn, **totals,
            "keep_rate": totals["keeps"] / totals["write_attempts"] if totals["write_attempts"] else None}


def build_v7_report(run_dir, *, bootstrap_samples=5000):
    root = Path(run_dir)
    manifest = json.loads((root / "comparison_manifest.json").read_text())
    if manifest.get("trainer_variant") != "recurrent_memory_v7":
        raise ValueError("Expected a recurrent_memory_v7 comparison manifest")
    result, report = build_report(root, bootstrap_samples=bootstrap_samples)
    result.update(trainer_variant="recurrent_memory_v7", additional_comparisons={}, storage_diagnostics={})
    result["model_descriptions"] = {name: model.get("description", "")
                                    for name, model in manifest["models"].items()}
    result["control_training_contract"] = {}
    lines = [report.rstrip(), "", "V7 recurrent WRITE/READ and storage-CVOM comparisons"]
    for name, description in result["model_descriptions"].items():
        lines.append(f"  {name}: {description}")
    lines += ["ae-control is separately trained; memory-off disables READ in the SAME reader checkpoint.",
              "Archive preserves all projected short-token history with growing compute/storage; not a capacity-matched control.",
              "Check matching training query/noise/epoch budgets before attributing trained-control differences to memory."]
    roles = set(manifest["models"])
    if "reader" in roles:
        reference = manifest["models"]["reader"].get("training_contract", {})
        for control in ("ae-control", "archive"):
            if control not in roles:
                continue
            candidate = manifest["models"][control].get("training_contract", {})
            keys = ("plan_sha256", "processed_queries", "seed", "expert_learning_rate")
            known = all(reference.get(key) is not None and candidate.get(key) is not None for key in keys)
            match = reference == candidate if known else None
            result["control_training_contract"][control] = {"metadata_matches": match,
                "reader": reference, "control": candidate}
            lines.append(f"Training-contract metadata reader/{control}: " + (
                "MATCH (recorded plan/query count/seed/AE LR/init; not equal memory compute)." if match else
                "DIFFERS; do not describe this as a matched-budget memory comparison." if known else
                "UNKNOWN; required training metadata is missing."))
    for left, right, explanation in (
        ("ae-control", "reader", "Independent AE-only control -> recurrent actor (check matched training budgets)"),
        ("archive", "reader", "Independent all-history archive -> recurrent compression (not equal memory capacity)"),
        ("memory-off", "reader", "Same Stage-1 actor/AE: READ off -> on, WRITE unchanged"),
        ("reader", "memory", "Matched frozen Stage-1 actor/AE: always UPDATE -> storage CVOM"),
        ("fixed", "memory", "Identical Stage-2 checkpoint: always UPDATE -> learned KEEP/UPDATE"),
        ("reader", "fixed", "Same frozen actor/AE and always-UPDATE policy (parity check)"),
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
    lines += ["", "WRITE decisions (final cumulative counters from completed episodes)"]
    for name, model in manifest["models"].items():
        if name in ("baseline", "ae-control") or model.get("mode") == "none":
            continue
        counters = completed_write_diagnostics(root, name, manifest["settings"]["tasks"],
                                               manifest["settings"]["n_episodes"])
        result["storage_diagnostics"][name] = counters
        lines.append(f"  {name}: recorded sessions={counters['completed_sessions_with_diagnostics']}; "
                     f"missing diagnostics={counters['completed_sessions_missing_diagnostics']}; "
                     f"UPDATE/KEEP={counters['updates']}/{counters['keeps']}; "
                     f"demo UPDATE/KEEP={counters['demo_updates']}/{counters['demo_keeps']}")
    lines += ["Fixed latent slot count is not bank fill or preserved event count.",
              "KEEP means retaining the previous memory contents, not task failure; more KEEP is not automatically better.",
              "Missing diagnostics are not evidence of zero memory activity."]
    return result, "\n".join(lines) + "\n"


def write_v7_report(run_dir, *, bootstrap_samples=5000):
    result, report = build_v7_report(run_dir, bootstrap_samples=bootstrap_samples)
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
