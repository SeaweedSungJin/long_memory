#!/usr/bin/env python3
"""Read-only statistics for paired closed-loop RoboMME evaluations.

Unlike a training action loss, ``success`` is the simulator's task completion
indicator. Missing episodes are never converted into failures. Pairing requires
both the episode index and its explicitly recorded inference seed to agree.
This module deliberately depends only on the standard library.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


SUITES = {
    "Counting": ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"],
    "Permanence": ["VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap"],
    "Reference": ["PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"],
    "Imitation": ["MoveCube", "InsertPeg", "PatternLock", "RouteStick"],
}
TASKS = [task for tasks in SUITES.values() for task in tasks]


def read_results(path: Path, *, expected: int | None = None) -> dict[int, dict]:
    """Validate completed CSV rows; an absent/empty file is an incomplete run."""
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return {}
        required = {"episode_idx", "episode_seed", "success"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required CSV columns {sorted(required - set(reader.fieldnames))}")
        result = {}
        for raw in reader:
            try:
                episode = int(raw["episode_idx"])
                seed = int(raw["episode_seed"])
                success = int(raw["success"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid episode/seed/success row {raw}") from exc
            if episode < 0 or seed < 0 or success not in (0, 1):
                raise ValueError(f"{path}: episode/seed must be nonnegative; success must be 0 or 1")
            if expected is not None and episode >= expected:
                raise ValueError(f"{path}: unexpected episode {episode} outside requested 0..{expected - 1}")
            if raw.get("status", "complete") not in ("complete", "completed", "success", "failure", "fail", "timeout", "step_limit"):
                raise ValueError(f"{path}: unfinished/error rows must not be recorded as task failures")
            if episode in result:
                raise ValueError(f"{path}: duplicate episode {episode}")
            result[episode] = {**raw, "episode_idx": episode, "episode_seed": seed, "success": success}
    return result


def paired_differences(baseline: dict[int, dict], memory: dict[int, dict]) -> list[int]:
    differences = []
    for episode in sorted(set(baseline) & set(memory)):
        left, right = baseline[episode], memory[episode]
        for key in ("episode_seed", "scenario_seed", "task_instruction"):
            # Optional fields must also match when supplied by either CSV.
            if left.get(key, "") != right.get(key, ""):
                raise ValueError(f"Episode {episode}: {key} differs across policies; refusing an unpaired comparison")
        differences.append(right["success"] - left["success"])
    return differences


def validate_result_identity(run_dir: Path, model: str, task: str, manifest: dict):
    """Never silently adopt a CSV copied from another policy/scenario run."""
    path = Path(run_dir) / model / task / "policy_manifest.json"
    if not path.is_file():
        raise ValueError(f"{path}: results exist without their policy identity manifest")
    identity = json.loads(path.read_text())
    if identity.get("evaluation_id") != manifest["evaluation_id"] + ":" + model:
        raise ValueError(f"{path}: evaluation/model identity does not match comparison manifest")
    if identity.get("task_id") != task:
        raise ValueError(f"{path}: task identity does not match its result directory")
    for key in ("dataset", "seed", "n_action_steps", "max_episode_steps"):
        if identity.get(key) != manifest["settings"].get(key):
            raise ValueError(f"{path}: {key} does not match comparison settings")
    return identity


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact binomial test of discordant episode outcomes."""
    n = wins + losses
    if n == 0:
        return 1.0
    # Integer accumulation avoids factorial overflow even for large evaluations.
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n))


def paired_macro_bootstrap(task_differences: list[list[int]], *, samples: int = 5000, seed: int = 17):
    """Resample paired episodes within each task; keep task weights equal.

    This is conditional on the evaluated tasks and one policy-noise seed. It is
    not uncertainty across training seeds or unseen task categories.
    """
    groups = [values for values in task_differences if values]
    if not groups or samples <= 0:
        return None
    rng = random.Random(seed)
    draws = sorted(sum(sum(rng.choices(group, k=len(group))) / len(group) for group in groups) / len(groups)
                   for _ in range(samples))
    return [draws[int((samples - 1) * .025)], draws[int((samples - 1) * .975)]]


def build_report(run_dir: Path, *, bootstrap_samples: int = 5000) -> tuple[dict, str]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "comparison_manifest.json").read_text())
    settings = manifest["settings"]
    tasks, expected = settings["tasks"], settings["n_episodes"]
    models = list(manifest["models"])
    if "baseline" not in models:
        raise ValueError("A paired report requires a baseline model")
    rows = {model: {task: read_results(run_dir / model / task / "simulation_results.csv", expected=expected)
                    for task in tasks} for model in models}
    identities = {}
    for model in models:
        for task in tasks:
            if rows[model][task]:
                identities[model, task] = validate_result_identity(run_dir, model, task, manifest)
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
        result["models"][model] = {"tasks": per_task, "complete": complete,
                                    "available_task_macro": sum(available) / len(available) if available else None}
        lines.append(f"{model}: {'COMPLETE' if complete else 'INCOMPLETE'}")
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
        # A one-episode task always resamples the exact same observation and
        # produces a misleading zero-width interval. Keep smoke runs descriptive.
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


def write_report(run_dir: Path, *, bootstrap_samples: int = 5000) -> tuple[dict, str]:
    result, report = build_report(run_dir, bootstrap_samples=bootstrap_samples)
    run_dir = Path(run_dir)
    (run_dir / "comparison_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (run_dir / "comparison_summary.txt").write_text(report)
    return result, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args(argv)
    _, report = write_report(args.run_dir, bootstrap_samples=args.bootstrap_samples)
    print(report)


if __name__ == "__main__":
    main()
