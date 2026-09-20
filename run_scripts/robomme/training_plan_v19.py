"""Immutable, full-TRAIN-epoch plans for the V19 objective comparison.

This module never creates files, loads a policy, or changes an episode.  A query
means one eligible *cached decision*, not one original video frame.  Both arms
receive the exact same plan: objective weights and output paths are intentionally
absent.  Task weights correct long-trajectory over-representation without dropping
or oversampling any TRAIN decision.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import torch

from gr00t.long_memory.train_v7 import coverage_windows, query_seed


TASKS = (
    "BinFill", "PickXtimes", "SwingXtimes", "StopCube", "VideoUnmask",
    "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap", "PickHighlight",
    "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder", "MoveCube", "InsertPeg",
    "PatternLock", "RouteStick",
)

# The legacy cache predates explicit benchmark-family fields.  Language alone
# cannot distinguish VideoUnmask from VideoUnmaskSwap.  Do NOT generalize this
# table to an arbitrary/reindexed cache: both identity and the COMPLETE ordered
# episode/instruction inventory must match.  Prefer explicit metadata whenever
# present.  The mapping is metadata for sampling/metrics, never model input.
LEGACY_CACHE_FINGERPRINT = "f4ee3fabe5d0e41840a72b8345d2688a94e94a4296689dea98adc284935ccb46"
LEGACY_INSTRUCTIONS_SHA256 = "245317115a304ac575b2f412b8ee276f88c1ddf156cb4bb7a69e9dc6f3f8a080"
LEGACY_TASK_RANGES = (
    (0, 100, "PatternLock"), (100, 200, "ButtonUnmaskSwap"),
    (200, 300, "ButtonUnmask"), (300, 400, "VideoPlaceButton"),
    (400, 500, "VideoUnmaskFamily_block400"), (500, 600, "PickXtimes"),
    (600, 700, "StopCube"), (700, 800, "SwingXtimes"),
    (800, 900, "PickHighlight"), (900, 1000, "MoveCube"),
    (1000, 1100, "InsertPeg"), (1100, 1200, "RouteStick"),
    (1200, 1300, "BinFill"), (1300, 1400, "VideoPlaceOrder"),
    (1400, 1500, "VideoRepick"), (1500, 1600, "VideoUnmaskFamily_block1500"),
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _explicit_task(record, expected_tasks):
    choices = [str(record[name]) for name in ("task_group", "task_name", "env_id")
               if record.get(name) is not None]
    # ``task`` is usually the instruction, but synthetic/new caches can use the
    # canonical family name.  Never treat an arbitrary instruction as a family.
    if record.get("task") in expected_tasks:
        choices.append(record["task"])
    if not choices:
        return None
    if any(name not in expected_tasks for name in choices) or len(set(choices)) != 1:
        raise ValueError(f"Invalid/conflicting explicit task metadata: {choices}")
    return choices[0]


def resolve_tasks_v19(manifest, *, expected_tasks=TASKS):
    """Return episode->family and auditable resolution provenance; fail closed."""
    records = manifest.get("episodes", [])
    ids = [int(record["episode_id"]) for record in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError("Task resolution requires unique episode manifest records")
    explicit = {int(record["episode_id"]): _explicit_task(record, expected_tasks)
                for record in records}
    if all(value is not None for value in explicit.values()):
        return explicit, {"method": "explicit_episode_metadata",
                          "mapping_sha256": digest({str(eid): task for eid, task in explicit.items()})}
    instruction_inventory = sorted([[int(r["episode_id"]), r.get("task")] for r in records])
    inventory_sha = digest(instruction_inventory)
    if (tuple(expected_tasks) != TASKS or manifest.get("fingerprint") != LEGACY_CACHE_FINGERPRINT
            or inventory_sha != LEGACY_INSTRUCTIONS_SHA256 or set(ids) != set(range(1600))):
        missing = [eid for eid, value in explicit.items() if value is None]
        raise ValueError("Cache needs explicit task_group/task_name/env_id metadata; "
                         f"cannot infer task families from episode numbers or ambiguous instructions ({missing[:5]})")
    mapping = {eid: task for start, end, task in LEGACY_TASK_RANGES for eid in range(start, end)}
    for eid, value in explicit.items():
        if value is not None and value != mapping[eid]:
            raise ValueError("Explicit task metadata conflicts with bound legacy mapping")
    return mapping, {"method": "fingerprint_and_full_instruction_inventory_bound_legacy_mapping",
        "cache_fingerprint": LEGACY_CACHE_FINGERPRINT, "instruction_inventory_sha256": inventory_sha,
        "mapping_sha256": digest({str(eid): task for eid, task in mapping.items()}),
        "ranges": [list(row) for row in LEGACY_TASK_RANGES],
        "resolved_task_groups": [task for _, _, task in LEGACY_TASK_RANGES],
        "unresolved_family_names": {"VideoUnmaskFamily_block400": ["VideoUnmask", "VideoUnmaskSwap"],
                                    "VideoUnmaskFamily_block1500": ["VideoUnmask", "VideoUnmaskSwap"]},
        "note": "Only this exact unreindexed cache is supported by the legacy mapping; "
                "VideoUnmask and VideoUnmaskSwap have identical instruction templates. "
                "Their two blocks are separately balanced but deliberately NOT assigned "
                "an unverified canonical task name; demo duration is not a task identity."}


def _validated_ids(splits, records):
    output = {}
    for split in ("train", "val"):
        ids = [int(eid) for eid in splits[split]]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Require nonempty unique {split} episode IDs")
        output[split] = sorted(ids)
    if set(output["train"]) & set(output["val"]):
        raise ValueError("TRAIN and VAL episodes must be disjoint")
    if set(output["train"]) | set(output["val"]) != set(records):
        raise ValueError("Episode records and TRAIN/VAL split inventory disagree")
    for split, ids in output.items():
        for eid in ids:
            if records[eid].get("split", split) != split:
                raise ValueError("Episode record split disagrees with split inventory")
    return output


def _episode_inventory(ep, eid, action_steps):
    decision_mask = ep["decision_mask"]
    if decision_mask.dtype != torch.bool or decision_mask.ndim != 1:
        raise ValueError("decision_mask must be a one-dimensional boolean mask")
    decisions = torch.where(decision_mask)[0].tolist()
    target_mask, action_mask = ep["target_mask"], ep["action_mask"]
    if (target_mask.dtype != torch.bool or target_mask.ndim != 3
            or action_mask.dtype != torch.bool or action_mask.ndim != 2
            or len(target_mask) != len(decision_mask) or len(action_mask) != len(decision_mask)):
        raise ValueError("Cached action/target masks disagree with decision inventory")
    if target_mask.shape[1] < action_steps or action_mask.shape[1] < action_steps:
        raise ValueError("Cache does not contain the deployed action prefix")
    if bool(target_mask[~decision_mask].any()) or bool(action_mask[~decision_mask].any()):
        raise ValueError("Passive observations/demo have action supervision")
    if "is_demo" in ep and bool((decision_mask & ep["is_demo"][:-1]).any()):
        raise ValueError("Demonstration endpoints cannot be action-loss queries")
    for decision in decisions:
        observed = target_mask[decision, :action_steps] & action_mask[decision, :action_steps, None]
        if not bool(observed.any()):
            raise ValueError(f"Decision lacks observed-prefix supervision: {eid}/{decision}")
    frames = ep.get("frames")
    if frames is not None:
        if len(frames) != len(decision_mask) + 1 or not bool((frames[1:] > frames[:-1]).all()):
            raise ValueError("Cached endpoint chronology is invalid")
        decision_frames = [int(frames[d]) for d in decisions]
    else:
        decision_frames = decisions
    return decisions, decision_frames


def assert_complete_epochs_v19(plan):
    """Public audit used by tests and trainers before model loading/resume."""
    expected = Counter((int(eid), int(d)) for eid, decisions in plan["train"] for d in decisions)
    if not expected or any(count != 1 for count in expected.values()):
        raise ValueError("TRAIN inventory must contain unique episode/decision pairs")
    actual = defaultdict(Counter)
    for window, batch in zip(plan["windows"], plan["schedule"], strict=True):
        pairs = [(int(row["episode_id"]), int(row["decision"])) for row in batch["queries"]]
        group_pairs = [(int(eid), int(d)) for eid, ds in window["groups"] for d in ds]
        if pairs != group_pairs or len(pairs) != window["query_count"]:
            raise ValueError("Schedule disagrees with coverage windows")
        actual[int(window["epoch"])].update(pairs)
    if set(actual) != set(range(plan["epochs"])) or any(actual[e] != expected for e in actual):
        raise ValueError("Every TRAIN query must appear exactly once in every epoch")
    if len(plan["schedule"]) != plan["total_steps"]:
        raise ValueError("Optimizer horizon disagrees with full-epoch schedule")


def build_plan_v19(args, cache, episodes):
    """Plan every eligible TRAIN query; no max-steps truncation or drop_last.

    Required caller settings: epochs, query_batch_size, seed, val_per_task,
    val_noise_samples, task_weighting.  ``expected_tasks`` is an optional fixture
    injection; production callers omit it and require all 16 RoboMME families.
    """
    for name in ("epochs", "query_batch_size", "val_per_task", "val_noise_samples"):
        value = getattr(args, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(args.seed) is not int or args.seed < 0 or args.task_weighting not in ("macro", "query"):
        raise ValueError("Invalid deterministic seed/task weighting")
    expected_tasks = tuple(getattr(args, "expected_tasks", TASKS))
    if not expected_tasks or len(set(expected_tasks)) != len(expected_tasks):
        raise ValueError("Expected task names must be nonempty and unique")
    manifest = cache.manifest
    records = {int(r["episode_id"]): r for r in manifest.get("episodes", [])}
    tasks, task_mapping = resolve_tasks_v19(manifest, expected_tasks=expected_tasks)
    production = expected_tasks == TASKS
    expected_tasks = tuple(task_mapping.get("resolved_task_groups", expected_tasks))
    splits = _validated_ids(manifest["splits"], records)
    action_steps = int(manifest["action_steps"])
    if action_steps <= 0:
        raise ValueError("Invalid cached action-prefix length")
    inventory, files, details = {}, {}, []
    task_statistics = {task: {split: {"episodes": 0, "eligible_episodes": 0, "queries": 0,
        "passive_endpoints_excluded": 0, "demo_endpoints": 0} for split in ("train", "val")}
        for task in expected_tasks}
    root = Path(cache.path).resolve()
    for split, ids in splits.items():
        for eid in ids:
            ep = episodes.fetch(eid)
            if int(ep["episode_id"]) != eid:
                raise ValueError("Loaded episode ID disagrees with manifest")
            explicit = _explicit_task(ep, expected_tasks)
            if explicit is not None and explicit != tasks[eid]:
                raise ValueError("Episode payload task disagrees with manifest")
            if (ep.get("task") is not None and records[eid].get("task") is not None
                    and ep["task"] != records[eid]["task"]):
                raise ValueError("Episode instruction disagrees with manifest")
            decisions, frames = _episode_inventory(ep, eid, action_steps)
            inventory[eid] = decisions
            stats = task_statistics[tasks[eid]][split]
            stats["episodes"] += 1
            stats["eligible_episodes"] += int(bool(decisions))
            stats["queries"] += len(decisions)
            stats["passive_endpoints_excluded"] += len(ep["decision_mask"]) - len(decisions)
            stats["demo_endpoints"] += int(ep["is_demo"].sum()) if "is_demo" in ep else 0
            details.append({"episode_id": eid, "split": split, "task": tasks[eid],
                            "decisions": decisions, "decision_frames": frames})
            if "path" in records[eid]:
                path = (root / records[eid]["path"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Episode path escapes cache")
                stat = path.stat()
                files[str(eid)] = [str(path), stat.st_size, stat.st_mtime_ns]
            elif production:
                raise ValueError("Production cache needs an episode file signature")
    for task in expected_tasks:
        if not task_statistics[task]["train"]["queries"]:
            raise ValueError(f"No eligible TRAIN queries for required task {task}")
        if task_statistics[task]["val"]["eligible_episodes"] < args.val_per_task:
            raise ValueError(f"Insufficient distinct held-out VAL episodes for {task}")
    train = [[eid, inventory[eid]] for eid in splits["train"] if inventory[eid]]
    count = sum(len(ds) for _, ds in train)
    weights = {task: count / (len(expected_tasks) * task_statistics[task]["train"]["queries"])
               if args.task_weighting == "macro" else 1.0 for task in expected_tasks}
    # queries_per_prefix=1 shuffles all queries while replay still reconstructs
    # each selected query's entire causal prefix.  Partial last batches stay.
    windows = coverage_windows(train, args.epochs, 1, args.query_batch_size, args.seed)
    schedule = []
    epoch_boundaries = []
    for index, window in enumerate(windows):
        schedule.append({"step": index + 1, "epoch": window["epoch"], "queries": [
            {"episode_id": eid, "decision": d, "task": tasks[eid], "task_weight": weights[tasks[eid]],
             "flow_seed": query_seed(args.seed, window["epoch"], eid, d, domain="train"),
             "generation_seed": query_seed(args.seed, window["epoch"], eid, d, domain="deployment-train-generated")}
            for eid, ds in window["groups"] for d in ds]})
        if window["epoch_end"]:
            epoch_boundaries.append({"epoch": window["epoch"] + 1, "step": index + 1,
                                     "processed_queries": (window["epoch"] + 1) * count})
    validation, validation_metadata = [], []
    for task_index, task in enumerate(expected_tasks):
        candidates = [eid for eid in splits["val"] if tasks[eid] == task and inventory[eid]]
        rng = random.Random(query_seed(args.seed, 0, task_index, 0, domain="v19-fixed-validation"))
        rng.shuffle(candidates)
        for index, eid in enumerate(candidates[:args.val_per_task]):
            choices = inventory[eid]
            phase_index = index % 3
            # Three disjoint temporal thirds when >=3 decisions are available.
            lower = len(choices) * phase_index // 3
            upper = len(choices) * (phase_index + 1) // 3
            phase_choices = choices[lower:upper]
            if not phase_choices:  # Explicitly recorded short-episode fallback.
                phase_choices = [choices[min(lower, len(choices) - 1)]]
            d = rng.choice(phase_choices)
            validation.append([eid, d])
            validation_metadata.append({"episode_id": eid, "decision": d, "task": task,
                "phase": ("early", "middle", "late")[phase_index],
                "phase_short_episode_fallback": upper == lower,
                "decision_fraction": choices.index(d) / max(1, len(choices) - 1),
                "task_weight": 1.0})
    validation_schedule = [{**item, "repeat": repeat,
        "flow_seed": query_seed(args.seed, 0, item["episode_id"], item["decision"], repeat, "deployment-val-flow"),
        "generation_seed": query_seed(args.seed, 0, item["episode_id"], item["decision"], repeat, "deployment-val-generated")}
        for item in validation_metadata for repeat in range(args.val_noise_samples)]
    manifest_path = root / "manifest.json"
    plan = {"format_version": 19, "train": train, "validation": validation, "files": files,
        "cache_fingerprint": manifest.get("fingerprint"),
        "cache_manifest_sha256": _file_hash(manifest_path) if manifest_path.is_file() else None,
        "windows": windows, "schedule": schedule, "validation_schedule": validation_schedule,
        "validation_metadata": validation_metadata, "train_query_count": count, "epochs": args.epochs,
        "total_steps": len(windows), "total_query_presentations": args.epochs * count,
        "epoch_boundaries": epoch_boundaries, "task_weighting": args.task_weighting,
        "task_weights": weights, "task_statistics": task_statistics,
        "episode_tasks": {str(eid): task for eid, task in tasks.items()},
        "task_mapping": task_mapping, "inventory_sha256": digest(details),
        "pairing": "single_query_full_causal_prefix_then_global_shuffle_exactly_once_per_epoch",
        "validation_sampling": "task_balanced_distinct_heldout_episodes_temporal_thirds_fixed_before_training",
        "coverage": {"unit": "eligible_cached_action_decision_not_every_raw_timestep",
            "action_steps": action_steps, "train_episodes": len(splits["train"]),
            "eligible_train_episodes": len(train), "heldout_episodes": len(splits["val"]),
            "queries_per_epoch": count, "all_tasks": list(expected_tasks),
            "partial_last_batch_kept": True, "max_steps_truncation": False,
            "cache_cadence_note": "Uses existing cached observations (RoboMME cache stride16); "
                                  "no claim of training on every original video frame."}}
    assert_complete_epochs_v19(plan)
    return plan, digest(plan)
