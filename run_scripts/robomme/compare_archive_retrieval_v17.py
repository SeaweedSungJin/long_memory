#!/usr/bin/env python3
"""Strict completed-run contrasts: matched action-only -> guided; old V4 -> guided.

No policy runs or checkpoint selection. Incomplete/mismatched results fail closed.
The historical contrast is optional and only accepted on all16 TEST n50 seed6.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gr00t.eval.sim.robomme.compare_long_memory_results import (
    TASKS, read_results, validate_result_identity, paired_differences,
    paired_macro_bootstrap, mcnemar_exact,
)


def load_run(root, roles):
    root = Path(root).resolve()
    manifest = json.loads((root / "comparison_manifest.json").read_text())
    expected = hashlib.sha256(json.dumps({k: v for k, v in manifest.items()
        if k != "evaluation_id"}, sort_keys=True).encode()).hexdigest()
    if manifest.get("evaluation_id") != expected:
        raise ValueError("Evaluation manifest identity changed")
    if manifest.get("trainer_variant") == "archive_read_control_v7":
        from run_scripts.robomme.eval_archive_read_control_v7 import validate_manifest_contract
        validate_manifest_contract(manifest)
    settings = manifest["settings"]
    rows, identities = {}, {}
    for role in roles:
        if role not in manifest["models"]:
            raise ValueError(f"Missing role: {role}")
        rows[role], identities[role] = {}, {}
        for task in settings["tasks"]:
            values = read_results(root / role / task / "simulation_results.csv", expected=settings["n_episodes"])
            if set(values) != set(range(settings["n_episodes"])):
                raise ValueError(f"Incomplete evaluation: {root}/{role}/{task}")
            rows[role][task] = values
            identities[role][task] = validate_result_identity(root, role, task, manifest)
    return {"root": str(root), "manifest": manifest, "rows": rows, "identities": identities}


def compatible(left, right):
    a, b = left["manifest"], right["manifest"]
    for key in ("tasks", "dataset", "n_episodes", "seed", "n_action_steps", "max_episode_steps", "device"):
        if a["settings"].get(key) != b["settings"].get(key):
            raise ValueError(f"Evaluation settings differ: {key}; VAL and TEST are not interchangeable")
    for key in ("benchmark", "base_file_sha256", "policy_package_versions"):
        if not a.get(key) or a.get(key) != b.get(key):
            raise ValueError(f"Evaluation provenance differs: {key}")
    # Policy-specific server files may differ intentionally. Shared production
    # files must not have silently changed between the compared evaluations.
    for path in a["source_sha256"].keys() & b["source_sha256"].keys():
        if a["source_sha256"][path] != b["source_sha256"][path]:
            raise ValueError(f"Shared evaluation source changed: {path}")
    for task in a["settings"]["tasks"]:
        anchor = left["identities"]["baseline"][task]
        for run in (left, right):
            for role in run["rows"]:
                identity = run["identities"][role][task]
                for key in ("scenario_metadata_sha256", "model_config_sha256", "memory_window", "demo_sampling"):
                    if identity.get(key) != anchor.get(key):
                        raise ValueError(f"Unmatched scenario/policy preprocessing: {task}/{role}/{key}")
        differences = paired_differences(left["rows"]["baseline"][task], right["rows"]["baseline"][task])
        if any(differences):
            raise ValueError("Original HAMLET baseline outcomes changed; rerun old model under the current setup")


def matched_training(control, guided):
    infos = []
    for run in (control, guided):
        model = run["manifest"]["models"]["archive"]
        path = Path(model["memory_checkpoint"])
        info = json.loads((path / "checkpoint.json").read_text())
        # Evaluator embeds payload identities: verify the checkpoint file wasn't
        # swapped after rollout, then compare the actual recorded training plan.
        from run_scripts.robomme.eval_long_memory_comparison import file_hash
        if model.get("checkpoint_sha256") != file_hash(path / "checkpoint.json"):
            raise ValueError("Evaluated checkpoint identity changed")
        if info["config"].get("driver_variant") != "archive_retrieval_v17":
            raise ValueError("Control/guided must be actual V17 policy checkpoints")
        if info["step"] != info["config"]["train"]["max_steps"]:
            raise ValueError("Use the prespecified fixed-final checkpoint, not a TEST-selected intermediate")
        infos.append(info)
    a, b = infos
    for key in ("plan_sha256", "original_continuation_parent", "source_sha256", "runtime"):
        if a["metadata"][key] != b["metadata"][key]:
            raise ValueError(f"Unmatched training arms: {key}")
    allowed = {"output_dir", "retrieval_weight", "init_checkpoint", "resume", "stop_after_steps", "preflight_only"}
    for key in a["config"]["train"].keys() | b["config"]["train"].keys():
        if key not in allowed and a["config"]["train"].get(key) != b["config"]["train"].get(key):
            raise ValueError(f"More than retrieval loss changed: {key}")
    if a["config"]["train"]["retrieval_weight"] != 0 or b["config"]["train"]["retrieval_weight"] <= 0:
        raise ValueError("Expected lambda=0 control and lambda>0 guided arm")


def contrast(left, left_role, right, right_role, *, samples=5000):
    tasks = left["manifest"]["settings"]["tasks"]
    groups, table = [], {}
    for task in tasks:
        a, b = left["rows"][left_role][task], right["rows"][right_role][task]
        d = paired_differences(a, b)
        groups.append(d)
        table[task] = {"n": len(d), "left_successes": sum(r["success"] for r in a.values()),
                      "right_successes": sum(r["success"] for r in b.values()),
                      "wins": d.count(1), "losses": d.count(-1)}
    wins, losses = sum(d.count(1) for d in groups), sum(d.count(-1) for d in groups)
    left_rate = sum(v["left_successes"] / v["n"] for v in table.values()) / len(tasks)
    right_rate = sum(v["right_successes"] / v["n"] for v in table.values()) / len(tasks)
    return {"left_macro": left_rate, "right_macro": right_rate, "delta": right_rate - left_rate,
            "ci95": paired_macro_bootstrap(groups, samples=samples), "wins": wins, "losses": losses,
            "mcnemar_p": mcnemar_exact(wins, losses), "tasks": table}


def build_report(control_dir, guided_dir, historical_dir=None):
    control = load_run(control_dir, ("baseline", "archive", "archive-off"))
    guided = load_run(guided_dir, ("baseline", "archive", "archive-off"))
    compatible(control, guided)
    matched_training(control, guided)
    result = {"settings": guided["manifest"]["settings"], "comparisons": {
        "action-only -> retrieval-guided": contrast(control, "archive", guided, "archive"),
        "same adapted AE READ-off -> READ-on": contrast(guided, "archive-off", guided, "archive"),
        "original HAMLET -> retrieval-guided": contrast(guided, "baseline", guided, "archive"),
    }, "run_dirs": [str(Path(control_dir).resolve()), str(Path(guided_dir).resolve())]}
    if historical_dir:
        s = result["settings"]
        if s["dataset"] != "test" or s["tasks"] != TASKS or s["n_episodes"] != 50 or s["seed"] != 6:
            raise ValueError("Historical 19.375% comparison requires all16 TEST n50 seed6")
        old = load_run(historical_dir, ("baseline", "memory"))
        compatible(old, guided)
        result["comparisons"]["historical V4 -> retrieval-guided"] = contrast(old, "memory", guided, "archive")
        result["run_dirs"].append(str(Path(historical_dir).resolve()))
    lines = ["V17: closed-loop task success (not retrieval/loss accuracy)",
             "Intervals are conditional on these tasks and this inference seed."]
    for name, row in result["comparisons"].items():
        lo, hi = row["ci95"]
        lines.append(f"{name}: {100*row['left_macro']:.3f}% -> {100*row['right_macro']:.3f}% "
                     f"({100*row['delta']:+.3f}pp); CI [{100*lo:+.3f},{100*hi:+.3f}]pp; "
                     f"wins/losses={row['wins']}/{row['losses']}; p={row['mcnemar_p']:.6g}")
    lines.append("A positive point estimate alone is not proof of a reliable improvement.")
    return result, "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-run", required=True)
    p.add_argument("--guided-run", required=True)
    p.add_argument("--historical-run")
    p.add_argument("--output-dir", required=True, help="NEW report-only directory")
    args = p.parse_args(argv)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a NEW report directory")
    from gr00t.long_memory.safety_v5 import validate_output_scope
    validate_output_scope(output, args.control_run, args.guided_run, args.historical_run,
                          ROOT / "checkpoints", ROOT / "data", ROOT / "runs/long_memory")
    result, text = build_report(args.control_run, args.guided_run, args.historical_run)
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(result, indent=2))
    (output / "comparison.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
