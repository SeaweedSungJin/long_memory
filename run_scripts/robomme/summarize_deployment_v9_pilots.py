#!/usr/bin/env python3
"""Read-only integrity and offline-loss comparison for matched V9 pilots.

No PyTorch, simulator, model loading or training imports. A missing/in-progress
arm is a pending comparison, not a failure. Recorded status does not establish
process liveness. Loss differences are NOT task-accuracy measurements.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

DRIVER = "archive_deployment_v9"
ARCHITECTURE = "recurrent_memory_v7"
SELECTION = "val/generated_observed_prefix_mse"
STEPS = (0, 32, 64, 96, 128)
ROLES = ("reader", "memory-off", "baseline")
METRICS = ("generated_observed_prefix_mse", "generated_observed_joint7_mse",
           "generated_observed_gripper1_mse", "flow_action_loss")
PAYLOADS = ("model.safetensors", "expert.safetensors", "cvom.safetensors")
QUERY_KEYS = ("episode_id", "decision", "repeat", "flow_seed", "generation_seed")
NOTE = ("Offline normalized losses only; negative deltas mean lower error, not higher robot accuracy. "
        "No simulator superiority or 30% success claim follows. Step 0 is unchanged parent initialization.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def bad_constant(value):
        raise ValueError(f"Nonfinite JSON constant {value}: {path}")
    return json.loads(Path(path).read_text(), parse_constant=bad_constant)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def normalized_config(config):
    result = copy.deepcopy(config)
    require(result.get("driver_variant") == DRIVER and result.get("trainer_variant") == ARCHITECTURE,
            "Expected same-architecture V9-driver configuration")
    require(result.get("stage") == 1 and result.get("mode") == "archive", "Expected Stage-1 archive")
    train, objective = result["train"], result["objective"]
    require(train.get("driver_variant") == objective.get("driver_variant") == DRIVER, "Conflicting driver declarations")
    require(objective.get("selection_metric") == SELECTION, "Different checkpoint-selection objective")
    require(train["aux_weight"] == objective["generated_observed_prefix_weight"], "Conflicting auxiliary weights")
    require(finite(train["aux_weight"]) and train["aux_weight"] >= 0, "Invalid auxiliary weight")
    # Output/pause/source paths are orchestration, not the optimizer horizon.
    # Original parent and initial payloads are validated separately below.
    for key in ("output_dir", "aux_weight", "init_checkpoint", "resume", "stop_after_steps", "preflight_only"):
        train.pop(key, None)
    objective.pop("generated_observed_prefix_weight")
    return result


def lineage_initial(run, config, provenance):
    """Find the immutable step-0 diagnosis, following exact-resume parents only."""
    current, cfg, meta, seen = run, config, provenance, set()
    history, through_step = [], config["train"]["max_steps"]
    while True:
        require(str(current) not in seen, "Cycle in exact-resume ancestry")
        seen.add(str(current))
        history.append({"run": str(current), "through_step": through_step})
        candidate = current / "checkpoint-000000"
        if candidate.is_dir():
            info = read(candidate / "checkpoint.json")
            require(info["step"] == 0 and info["config"]["driver_variant"] == DRIVER, "Invalid initial diagnosis bundle")
            require(normalized_config(info["config"]) == normalized_config(config), "Initial configuration differs")
            require(info["config"]["train"]["aux_weight"] == config["train"]["aux_weight"], "Initial arm weight differs")
            for key in ("base_model", "source_sha256", "cache_fingerprint", "plan_sha256", "original_continuation_parent"):
                require(info["metadata"][key] == provenance[key], f"Initial {key} differs from run")
            hashes = {name: file_hash(candidate / name) for name in PAYLOADS}
            require(all(hashes[name] == info["metadata"]["payload_sha256"][name] for name in PAYLOADS), "Initial payload checksum differs")
            return {"path": str(candidate), "payload_sha256": hashes, "validation_history": history}
        if not cfg["train"].get("resume"):
            return None  # Step-0 validation/save may still be running.
        parent = meta["parent_checkpoint"]
        checkpoint = Path(parent["path"]).resolve()
        require(checkpoint.is_dir(), "Missing exact-resume ancestor needed to verify initialization")
        require(file_hash(checkpoint / "checkpoint.json") == parent["files"]["checkpoint.json"], "Resume-parent manifest changed")
        info = read(checkpoint / "checkpoint.json")
        require(info["config"].get("driver_variant") == DRIVER, "Resume ancestor is not the V9 driver")
        require(normalized_config(info["config"]) == normalized_config(config), "Resume ancestor optimizer/objective configuration differs")
        require(info["config"]["train"]["aux_weight"] == config["train"]["aux_weight"], "Resume ancestor auxiliary weight differs")
        current, cfg, meta, through_step = checkpoint.parent, info["config"], info["metadata"], info["step"]


def validation_record(path, plan, plan_sha, step):
    data = read(path)
    require(data["step"] == step and data["plan_sha256"] == plan_sha, "Validation step/plan hash differs")
    require(data["selection_metric"] == SELECTION, "Validation selection criterion changed")
    expected = {tuple(item[key] for key in QUERY_KEYS) + (role,)
                for item in plan["validation_schedule"] for role in ROLES}
    require(len(expected) == len(plan["validation_schedule"]) * len(ROLES), "Duplicate fixed validation schedule")
    rows = {}
    for row in data["records"]:
        key = tuple(row[k] for k in QUERY_KEYS) + (row["role"],)
        require(key in expected and key not in rows, "Unexpected/duplicate validation role/query/noise pair")
        for name, value in row.items():
            if name not in QUERY_KEYS and name != "role":
                require(finite(value), f"Invalid validation scalar {name}")
        require(all(name in row for name in METRICS), "Missing required prefix/component/flow metric")
        rows[key] = row
    require(set(rows) == expected, "Incomplete published validation role/query records")
    require(set(data["summary"]) == set(ROLES), "Validation role summary differs")
    summaries = {}
    for role in ROLES:
        selected = [r for r in rows.values() if r["role"] == role]
        summaries[role] = {}
        for metric in METRICS:
            mean = sum(r[metric] for r in selected) / len(selected)
            require(close(mean, data["summary"][role][metric]), f"Stored mean differs from validation records: {role}/{metric}")
            summaries[role][metric] = mean
    return summaries, rows


def inspect_run(path):
    path = Path(path).resolve()
    result = {"path": str(path), "available": path.is_dir(), "recorded_status": "not_started_or_missing",
              "completed": False, "published_validation_steps": [], "published_checkpoint_steps": [],
              "note": "Status is recorded file evidence, not process liveness."}
    required = ("run_config.json", "provenance.json", "query_plan.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        result["pending_files"] = missing
        return result, None
    config, provenance, plan = (read(path / name) for name in required)
    normalized_config(config)
    require(provenance["driver_variant"] == DRIVER and provenance["selection_metric"] == SELECTION, "Invalid V9 provenance")
    plan_sha = plan.pop("sha256")
    require(digest(plan) == plan_sha == provenance["plan_sha256"], "Query/noise plan checksum differs")
    require(len(plan["validation"]) >= 32, "Pilot validation has fewer than 32 queries")
    train_ids = {eid for eid, _ in plan["train"]}
    require(not train_ids & {eid for eid, _ in plan["validation"]}, "Training/validation episode leakage")
    require(len({eid for eid, _ in plan["validation"]}) == len(plan["validation"]), "Expected distinct validation episodes")
    schedule_pairs = {(r["episode_id"], r["decision"]) for r in plan["validation_schedule"]}
    require(schedule_pairs == {tuple(q) for q in plan["validation"]}, "Validation schedule does not cover fixed queries")
    require(len(plan["schedule"]) == config["train"]["max_steps"], "Plan length differs from fixed optimizer horizon")
    initial = lineage_initial(path, config, provenance)
    result.update(aux_weight=config["train"]["aux_weight"], planned_updates=config["train"]["max_steps"],
                  plan_sha256=plan_sha, initial_checkpoint=initial)
    status = read(path / "status.json") if (path / "status.json").is_file() else None
    if status:
        result["recorded_status"] = status["status"]
        result["recorded_updates"] = status["optimizer_updates"]
        result["processed_queries"] = status["processed_queries"]
    else:
        result["recorded_status"] = "initialization_or_step0_pending"
    if (path / "failure.json").is_file():
        result["recorded_failure"] = read(path / "failure.json")
    best_path = path / "best_checkpoint.json"
    if best_path.is_file():
        pointer = read(best_path)
        candidate = Path(pointer["path"])
        candidate = candidate if candidate.is_absolute() else path / candidate
        if candidate.is_dir():
            best_step = read(candidate / "checkpoint.json")["step"]
            result["best"] = {"path": str(candidate), "step": best_step, "selection_metric": SELECTION,
                              "meaning": "unchanged_parent_initialization" if best_step == 0 else "lowest_fixed_offline_generated_prefix_loss"}
        else:
            result["best"] = {"pointer": pointer, "pending_or_unavailable": True}
    validations, checkpoints = {}, {}
    for step in STEPS:
        history = initial["validation_history"] if initial else [{"run": str(path), "through_step": config["train"]["max_steps"]}]
        eligible = [Path(item["run"]) for item in history if step <= item["through_step"]]
        validation = next((folder / f"validation-{step:06d}.json" for folder in eligible
                           if (folder / f"validation-{step:06d}.json").is_file()), None)
        checkpoint = next((folder / f"checkpoint-{step:06d}" for folder in eligible
                           if (folder / f"checkpoint-{step:06d}").is_dir()), None)
        if validation is not None:
            validations[step] = validation_record(validation, plan, plan_sha, step)
            result["published_validation_steps"].append(step)
        if checkpoint is not None:
            info = read(checkpoint / "checkpoint.json")
            require(info["step"] == step and info["metadata"]["plan_sha256"] == plan_sha, "Checkpoint step/plan differs")
            require(normalized_config(info["config"]) == normalized_config(config), "Checkpoint configuration differs")
            require(info["config"]["train"]["aux_weight"] == config["train"]["aux_weight"], "Checkpoint arm differs")
            for name in PAYLOADS:
                require(file_hash(checkpoint / name) == info["metadata"]["payload_sha256"][name],
                        f"Published checkpoint payload checksum differs: {step}/{name}")
            checkpoints[step] = info
            result["published_checkpoint_steps"].append(step)
    last = config["train"]["max_steps"]
    if status and status["status"] == "complete" and status["optimizer_updates"] == last:
        result["completed"] = last in checkpoints and last in validations
    return result, {"config": config, "provenance": provenance, "plan": plan, "initial": initial,
                    "validation": validations, "checkpoints": checkpoints}


def compare(flow_run, aux_run):
    flow_status, flow = inspect_run(flow_run)
    aux_status, aux = inspect_run(aux_run)
    report = {"flow": flow_status, "aux": aux_status, "pairing_verified": False,
              "all_fixed_steps_complete": False, "common_steps": [], "comparisons": [], "note": NOTE}
    if flow is None or aux is None:
        report["pending"] = "One or both run manifests are not published yet."
        return report
    require(flow["config"]["train"]["aux_weight"] == 0 and aux["config"]["train"]["aux_weight"] > 0,
            "Expected flow-only weight 0 and positive auxiliary arm")
    require(normalized_config(flow["config"]) == normalized_config(aux["config"]), "Unmatched optimizer/architecture configuration")
    require(flow["plan"] == aux["plan"], "Unmatched query/validation/flow-noise/generation-noise schedule")
    for key in ("base_model", "cache_fingerprint", "cache_dir", "cache_manifest_sha256", "source_sha256", "runtime", "original_continuation_parent"):
        require(flow["provenance"][key] == aux["provenance"][key], f"Unmatched provenance: {key}")
    if flow["initial"] is None or aux["initial"] is None:
        report["pending"] = "Initial checkpoint publication is pending; initialization equality is not yet proven."
        return report
    require(flow["initial"]["payload_sha256"] == aux["initial"]["payload_sha256"], "Different initial memory/Expert/CVOM payloads")
    report["pairing_verified"] = True
    report["checks"] = ["configuration_except_weight_and_run_paths", "full_query_and_noise_plan",
        "disjoint_fixed_validation", "matching_recorded_original_parent_base_cache_source_runtime",
        "actual_initial_payload_hashes", "published_checkpoint_payload_checksums", "validation_record_means"]
    baseline_reference = None
    for step in STEPS:
        if step not in flow["validation"] or step not in aux["validation"]:
            continue
        a, a_rows = flow["validation"][step]
        b, b_rows = aux["validation"][step]
        require(set(a_rows) == set(b_rows), "Validation pair keys differ")
        baseline = {key: {metric: row[metric] for metric in METRICS}
                    for key, row in a_rows.items() if key[-1] == "baseline"}
        for key in baseline:
            for metric in METRICS:
                require(close(a_rows[key][metric], b_rows[key][metric]), "Original baseline differs across matched arms")
                if baseline_reference is not None:
                    require(close(baseline[key][metric], baseline_reference[key][metric]), "Original baseline changed across checkpoints")
        baseline_reference = baseline
        if step == 0:
            require(all(close(a_rows[key][metric], b_rows[key][metric]) for key in a_rows for metric in METRICS),
                    "Step-0 validation differs despite matched initial payloads")
        item = {"step": step, "checkpoint_published_both": step in flow["checkpoints"] and step in aux["checkpoints"],
                "roles": {}, "reader_minus_memory_off": {}}
        for role in ROLES:
            item["roles"][role] = {metric: {"flow": a[role][metric], "aux": b[role][metric],
                "aux_minus_flow": b[role][metric] - a[role][metric]} for metric in METRICS}
        for arm, summary in (("flow", a), ("aux", b)):
            item["reader_minus_memory_off"][arm] = {metric: summary["reader"][metric] - summary["memory-off"][metric] for metric in METRICS}
        report["common_steps"].append(step)
        report["comparisons"].append(item)
    report["all_fixed_steps_complete"] = (report["common_steps"] == list(STEPS) and
        all(c["checkpoint_published_both"] for c in report["comparisons"]) and flow_status["completed"] and aux_status["completed"])
    if not report["all_fixed_steps_complete"]:
        report["pending"] = "Not all fixed pilot boundaries are published/completed; available pairs are reported without extrapolation."
    return report


def render(report):
    lines = ["Matched V9 deployment pilots — offline loss, NOT task accuracy"]
    for arm in ("flow", "aux"):
        run = report[arm]
        lines.append(f"{arm}: recorded_status={run['recorded_status']} updates={run.get('recorded_updates', '?')}/{run.get('planned_updates', '?')} "
                     f"validation={run['published_validation_steps']}")
        if "best" in run:
            best = run["best"]
            lines.append(f"  best step={best.get('step', '?')}: {best.get('meaning', 'pointer unavailable')}")
    lines.append(f"Pairing verified: {report['pairing_verified']}; all fixed boundaries complete: {report['all_fixed_steps_complete']}")
    if report.get("pending"):
        lines.append(report["pending"])
    for item in report["comparisons"]:
        lines.append(f"\nStep {item['step']} (both checkpoints published: {item['checkpoint_published_both']})")
        lines.append("Role          Metric                                  Flow             Aux      Aux - Flow")
        for role, metrics in item["roles"].items():
            for metric, values in metrics.items():
                lines.append(f"{role:12}  {metric:36} {values['flow']:14.8g}  {values['aux']:14.8g}  {values['aux_minus_flow']:+14.8g}")
        for arm in ("flow", "aux"):
            delta = item["reader_minus_memory_off"][arm]["generated_observed_prefix_mse"]
            lines.append(f"  {arm} reader - same adapted AE memory-off, prefix MSE: {delta:+.8g}")
    lines.extend(("", NOTE))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow-run", required=True)
    p.add_argument("--aux-run", required=True)
    p.add_argument("--output-file", help="Optional NEW JSON file; parent must exist, no overwrite")
    args = p.parse_args(argv)
    report = compare(args.flow_run, args.aux_run)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output_file:
        with Path(args.output_file).open("x", encoding="utf-8") as handle:
            handle.write(payload)
    print(render(report))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, FileExistsError) as exc:
        print(f"[integrity] {exc}", file=sys.stderr)
        raise SystemExit(1)
