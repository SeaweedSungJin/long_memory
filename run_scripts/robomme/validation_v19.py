"""Task-macro offline validation for the full-coverage RoboMME experiment.

All training arms are assessed with the SAME unweighted flow objective, plus
pure-noise generated actions. Task means are averaged equally; longer tasks or
extra noise repeats cannot silently dominate the main validation metric.
Observed-prefix joint7/gripper1 diagnostics and query-error concentration make
the previous two-outlier-dominated monitor visible. None is robot accuracy.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import math
import statistics

import torch

from gr00t.long_memory.cache_reader_v3 import validate_decision
from gr00t.long_memory.expert_v4 import adapter_disabled
from run_scripts.robomme.audit_archive_generation_v7 import prefix_masks
from run_scripts.robomme.deployment_objective_v9 import generated_prefix_objective
from run_scripts.robomme.flow_objective_v19 import episode_flow_v19


VALIDATION_VERSION = "robomme_task_macro_validation_v19_v1"
ROLES = ("reader", "memory-off", "baseline")


def generated_metrics_v19(generated, episode, decision, action_steps=16):
    """Measure only masked GT coordinates; generation has already finished."""
    prediction = generated["prediction"].float()
    target = episode["targets"][decision].to(device=prediction.device, dtype=torch.float32)[None]
    mask = episode["target_mask"][decision].to(device=prediction.device)[None]
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError("Generated actions must match one [1,H,A] target")
    if prediction.shape[-1] < 8 or bool(mask[..., 8:].any()):
        raise ValueError("RoboMME requires joint7/gripper1 with all remaining coordinates masked")
    nominal, observed = prefix_masks(mask, episode["action_mask"][decision], action_steps)
    if not bool(observed.any()):
        raise ValueError("No observed action coordinates in validation query")
    difference = prediction - target

    def measure(valid):
        values = difference[valid]
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("Nonfinite valid generated-action errors")
        return {"mse": float(values.square().mean()) if values.numel() else 0.,
                "mae": float(values.abs().mean()) if values.numel() else 0.,
                "valid_values": int(values.numel())}

    result = {"generated_prefix_" + key: value for key, value in measure(observed).items()}
    result.update({"generated_nominal_prefix_" + key: value for key, value in measure(nominal).items()})
    for name, part in (("joint", slice(0, 7)), ("gripper", slice(7, 8))):
        selected = torch.zeros_like(observed)
        selected[..., part] = observed[..., part]
        result.update({f"generated_executed_{name}_" + key: value for key, value in measure(selected).items()})
    return result


def aggregate_v19(records):
    """Return equal-task macro summaries and role/task arithmetic means.

    The two concentration diagnostics intentionally inspect individual query /
    noise records (not task averages). A zero total error has top-two share 0.
    Task-macro metrics remain distinct from these descriptive distributions.
    """
    if not records:
        raise ValueError("Validation requires nonempty records")
    groups = defaultdict(lambda: defaultdict(list))
    excluded = {"role", "task", "episode_id", "decision", "repeat", "flow_seed", "generation_seed"}
    metric_keys = None
    for record in records:
        role, task = record.get("role"), record.get("task")
        if role not in ROLES or not isinstance(task, str) or not task:
            raise ValueError("Every validation record requires a known role and task")
        keys = set(record) - excluded
        if metric_keys is None:
            metric_keys = keys
        if keys != metric_keys or not {"action_loss", "generated_prefix_mse"} <= keys:
            raise ValueError("Validation metric schema differs between records")
        if any(isinstance(record[key], bool) or not isinstance(record[key], (int, float))
               or not math.isfinite(record[key]) for key in keys):
            raise ValueError("All validation metrics must be finite numbers")
        groups[role][task].append(record)
    if set(groups) != set(ROLES):
        raise ValueError("Validation requires reader, memory-off and baseline")
    expected_tasks = set(groups["reader"])
    if any(set(groups[role]) != expected_tasks for role in ROLES):
        raise ValueError("Validation roles must contain the same task set")
    # Roles must evaluate the same query/noise occurrences, not merely the same
    # task labels. Duplicates are compared as counts to catch missing repeats.
    from collections import Counter
    identities = {role: Counter((row["task"], row["episode_id"], row["decision"], row.get("repeat", 0),
        row["flow_seed"], row["generation_seed"]) for task in groups[role].values() for row in task)
        for role in ROLES}
    if any(identities[role] != identities["reader"] for role in ROLES):
        raise ValueError("Validation roles do not have matched query/noise records")
    by_task, summaries = {}, {}
    for role in ROLES:
        by_task[role] = {task: {key: statistics.fmean(row[key] for row in rows) for key in metric_keys}
                         for task, rows in sorted(groups[role].items())}
        summaries[role] = {key: statistics.fmean(values[key] for values in by_task[role].values())
                           for key in metric_keys}
        errors = [row["generated_prefix_mse"] for rows in groups[role].values() for row in rows]
        total = math.fsum(errors)
        summaries[role].update(generated_mse_top2_share=math.fsum(sorted(errors, reverse=True)[:2])/total if total else 0.,
            generated_prefix_mse_median=statistics.median(errors), validation_task_count=len(expected_tasks),
            validation_query_noise_records=len(errors))
        summaries[role]["loss"] = summaries[role]["action_loss"]
    summaries["reader"]["memory_gain"] = summaries["memory-off"]["action_loss"] - summaries["reader"]["action_loss"]
    summaries["reader"]["generated_memory_gain"] = summaries["memory-off"]["generated_prefix_mse"] - summaries["reader"]["generated_prefix_mse"]
    return summaries, by_task


@torch.no_grad()
def validate_v19(args, core, head, episodes, plan, baseline_cache):
    """Matched three-role validation; no training or simulator state updates."""
    records = []
    for item in plan["validation_schedule"]:
        eid, decision = item["episode_id"], item["decision"]
        if not isinstance(item.get("task"), str) or not item["task"]:
            raise ValueError("V19 validation schedule must carry canonical task names")
        ep = episodes.fetch(eid)
        validate_decision(ep, decision)
        replay = core.replay(ep, decision)
        for role, conditioning in (("reader", replay["fused"]), ("memory-off", replay["short"]), ("baseline", None)):
            key = (VALIDATION_VERSION, eid, decision, item["flow_seed"], item["generation_seed"])
            if role == "baseline" and key in baseline_cache:
                metrics = dict(baseline_cache[key])
            else:
                with adapter_disabled(head) if role == "baseline" else nullcontext():
                    flow = episode_flow_v19(head, ep, decision, conditioning, seed=item["flow_seed"],
                                            tail_weight=1., action_steps=16, activation_checkpointing=False)
                    generated = generated_prefix_objective(head, ep, decision, conditioning,
                        seed=item["generation_seed"], action_steps=16, activation_checkpointing=False)
                metrics = {name: float(value) for name, value in flow.items()
                           if name not in {"prediction", "loss", "action_loss"}
                           and (isinstance(value, (int, float)) or isinstance(value, torch.Tensor) and value.numel() == 1)}
                metrics.update(action_loss=float(flow["original_flow_loss"]), loss=float(flow["original_flow_loss"]))
                metrics.update(generated_metrics_v19(generated, ep, decision, 16))
                if role == "baseline":
                    baseline_cache[key] = dict(metrics)
            identity = {name: item[name] for name in ("task", "episode_id", "decision", "flow_seed", "generation_seed")}
            records.append({**identity, "repeat": item.get("repeat", 0), "role": role, **metrics})
    summaries, by_task = aggregate_v19(records)
    return summaries, records, by_task
