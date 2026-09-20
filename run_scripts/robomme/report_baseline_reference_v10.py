"""Pure paired report rendering with an explicit, unmodified baseline reference.

Reference immutability and compatibility are checked by the caller. This module
still validates each original policy identity and every paired CSV row. It never
copies results, rewrites evaluation IDs, or writes a report to either run.
"""
from __future__ import annotations

from pathlib import Path

from gr00t.eval.sim.robomme.compare_long_memory_results import (
    mcnemar_exact, paired_differences, paired_macro_bootstrap, read_results,
    validate_result_identity,
)


def build_reference_report(run_dir, manifest, reference_root, reference_manifest, *, bootstrap_samples=5000):
    """Use generic paired-report statistics, reading baseline from its own run."""
    run_dir, reference_root = Path(run_dir), Path(reference_root)
    settings = manifest["settings"]
    tasks, expected = settings["tasks"], settings["n_episodes"]
    models = list(manifest["models"])
    if "baseline" not in models or "baseline" not in reference_manifest["models"]:
        raise ValueError("A paired reference report requires a baseline model in both manifests")
    origins = {model: (reference_root, reference_manifest) if model == "baseline"
               else (run_dir, manifest) for model in models}
    rows = {model: {task: read_results(origins[model][0] / model / task / "simulation_results.csv", expected=expected)
                    for task in tasks} for model in models}
    identities = {}
    for model in models:
        for task in tasks:
            if rows[model][task]:
                root, source_manifest = origins[model]
                identities[model, task] = validate_result_identity(root, model, task, source_manifest)
    for model in models:
        for task in tasks:
            left, right = identities.get(("baseline", task)), identities.get((model, task))
            if left and right:
                for key in ("scenario_metadata_sha256", "model_config_sha256", "memory_window", "demo_sampling"):
                    if left.get(key) != right.get(key):
                        raise ValueError(f"{model}/{task}: {key} differs from baseline; refusing unmatched evaluation")
    result = {"evaluation_id": manifest["evaluation_id"], "metric": "closed_loop_task_success",
              "requested_tasks": tasks, "requested_episodes_per_task": expected, "models": {}, "comparisons": {}}
    lines = ["RoboMME paired task-success comparison", f"Run: {run_dir.resolve()}",
             "Metric: simulator task success, NOT action-loss accuracy.",
             "Missing episodes are excluded, never counted as failures.", ""]
    for model in models:
        per_task = {}
        for task in tasks:
            values = rows[model][task]
            n = len(values)
            wins = sum(row["success"] for row in values.values())
            per_task[task] = {"successes": wins, "completed": n, "requested": expected,
                              "success_rate": wins / n if n else None, "complete": n == expected}
        available = [entry["success_rate"] for entry in per_task.values() if entry["success_rate"] is not None]
        complete = all(entry["complete"] for entry in per_task.values())
        root, source_manifest = origins[model]
        reused = model == "baseline"
        completed = sum(entry["completed"] for entry in per_task.values())
        result["models"][model] = {
            "tasks": per_task, "complete": complete,
            "available_task_macro": sum(available) / len(available) if available else None,
            "origin": "reused_reference" if reused else "fresh",
            "newly_rolled_out": 0 if reused else completed,
            "source_run": str(root.resolve()), "source_evaluation_id": source_manifest["evaluation_id"],
        }
        status = "COMPLETE" if complete else "INCOMPLETE"
        if reused:
            lines.append(f"baseline: REUSED reference {root.resolve()}; source evaluation "
                         f"{source_manifest['evaluation_id']}; newly rolled out 0; {status}")
        else:
            lines.append(f"{model}: {status}; fresh; newly rolled out {completed}")
        for task, entry in per_task.items():
            rate = "--" if entry["success_rate"] is None else f"{100 * entry['success_rate']:.2f}%"
            lines.append(f"  {task:<20} {rate:>8}  {entry['successes']}/{entry['completed']} successes"
                         f"  ({entry['completed']}/{expected} episodes completed)")
        macro = result["models"][model]["available_task_macro"]
        lines.append(f"  Available-task macro ({len(available)}/{len(tasks)} tasks): "
                     + ("--" if macro is None else f"{100 * macro:.2f}%"))
        lines.append("")
    for model in models:
        if model == "baseline":
            continue
        per_task = {}
        groups = []
        lines += [f"baseline -> {model}", f"{'Task':<22} {'Paired N':>8} {'Delta':>10} {'Wins/Losses/Same':>19}"]
        for task in tasks:
            differences = paired_differences(rows["baseline"][task], rows[model][task])
            groups.append(differences)
            wins, losses = differences.count(1), differences.count(-1)
            n = len(differences)
            delta = sum(differences) / n if n else None
            per_task[task] = {"paired_n": n, "wins": wins, "losses": losses, "same": n - wins - losses,
                              "paired_delta": delta}
            delta_text = "--" if delta is None else f"{100 * delta:+.2f}pp"
            lines.append(f"{task:<22} {n:>8} {delta_text:>10} {wins:>5}/{losses}/{n - wins - losses}")
        all_differences = [value for group in groups for value in group]
        available = [sum(group) / len(group) for group in groups if group]
        delta = sum(available) / len(available) if available else None
        wins, losses = all_differences.count(1), all_differences.count(-1)
        # Match the generic report's smoke-run uncertainty convention exactly.
        too_small = any(len(group) == 1 for group in groups)
        ci = None if too_small else paired_macro_bootstrap(groups, samples=bootstrap_samples)
        comparison = {"tasks": per_task, "paired_n": len(all_differences), "wins": wins, "losses": losses,
                      "same": len(all_differences) - wins - losses, "paired_task_macro_delta": delta,
                      "paired_task_macro_bootstrap_ci95": ci, "mcnemar_exact_p": mcnemar_exact(wins, losses),
                      "complete": all(len(group) == expected for group in groups)}
        result["comparisons"][model] = comparison
        if delta is not None:
            lines.append(f"Paired task-macro delta: {100 * delta:+.2f}pp over {len(available)}/{len(tasks)} tasks"
                         f", {len(all_differences)} matched episodes")
            if ci:
                lines.append(f"95% within-task paired bootstrap CI: [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]pp")
            elif too_small:
                lines.append("95% CI not reported: a task has only one paired episode (smoke test).")
            lines.append(f"McNemar exact p={comparison['mcnemar_exact_p']:.6g}")
        else:
            lines.append("No matched completed episodes yet.")
        if not comparison["complete"]:
            lines.append("INCOMPLETE: partial results are not a full paired benchmark comparison.")
        lines.append("")
    lines += ["Wins/losses mean memory success + baseline failure / the reverse; 0/0 does NOT mean skipped.",
              "A 1-3 episode smoke test checks execution, not reliable model superiority.",
              "Bootstrap uncertainty is conditional on these tasks and this inference seed."]
    return result, "\n".join(lines) + "\n"
