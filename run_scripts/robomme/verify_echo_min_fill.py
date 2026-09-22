#!/usr/bin/env python3
"""Real-AE regression for the explicit min-fill override, without a simulator.

One loaded policy replays fixed observed inputs through its inherited parent,
the wrapper at checkpoint settings, and the wrapper with min_fill=32. This
checks implementation equivalence and forced filling, not task performance.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time
from types import MethodType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gr00t.long_memory.cache import EpisodeCache, _episode_records
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes
from gr00t.long_memory.checkpoint_v4 import _state_sha256
from gr00t.long_memory.expert_v4 import expert_state_sha256
from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.echo_cvom_checkpoint import inspect_checkpoint
from run_scripts.robomme.policy_echo_cvom import EchoPolicyV1, runtime_source_identity
from run_scripts.robomme.policy_echo_min_fill import (
    EchoMinFillPolicy, apply_min_fill_override, min_fill_source_identity,
)
from run_scripts.robomme.train_cvom_admission import gpu_guard
from run_scripts.robomme.verify_echo_cvom import replay


KIND = "echo_min_fill_regression_v1"
PANEL = ((1355, 72),)
VERIFICATION_SOURCES = ("verify_echo_min_fill.py", "verify_echo_cvom.py")


def verification_source_identity():
    return {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in VERIFICATION_SOURCES}


def _sources():
    return {"echo_source_sha256": runtime_source_identity(),
            "min_fill_source_sha256": min_fill_source_identity(),
            "verification_source_sha256": verification_source_identity()}


def _parameters(policy):
    return {"core": _state_sha256(policy.representation.delta_state_dict()),
            "expert": expert_state_sha256(policy.model.action_head)}


def _decoded_hashes(actions):
    return {name: {"sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                   "shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
            for name, value in sorted(actions.items())}


def _branch(policy, loader, record, episode, count, name, *, parent=False):
    dispatch = MethodType(EchoPolicyV1._get_action, policy) if parent else policy._get_action
    decoded, runtime = [], []

    def capture(observation, options=None):
        actions, info = dispatch(observation, options)
        decoded.append(_decoded_hashes(actions))
        runtime.append(info)
        if len(decoded) % 16 == 0 or len(decoded) == count:
            print(f"[echo-min-fill-verify] {name}: {len(decoded)}/{count} endpoints", flush=True)
        return actions, info

    with patch.object(policy, "_get_action", capture):
        rows, errors = replay(policy, loader, record, episode, count, False)
    if len(rows) != count or len(decoded) != count:
        raise AssertionError("Incomplete fixed-input verification replay")
    for row, action in zip(rows, decoded, strict=True):
        row["decoded_actions"] = action
    return rows, errors, runtime


def _validate_forced_fill(rows, capacity=32):
    previous, prefull, rejections = 0, 0, 0
    for index, row in enumerate(rows):
        metrics = row["writer"]
        operations = {key: metrics["writer_" + key] for key in ("append", "keep", "replace", "merge")}
        if any(value not in (0., 1.) for value in operations.values()) or sum(operations.values()) != 1:
            raise AssertionError("Invalid writer operation evidence")
        if metrics["writer_full"] != float(previous == capacity):
            raise AssertionError("Writer full flag does not describe the pre-WRITE bank")
        n = len(row["event_ids"])
        if n != previous + int(operations["append"]) or n > capacity:
            raise AssertionError("Bank occupancy changed outside the recorded WRITE operation")
        if previous < capacity:
            prefull += 1
            rejections += int(operations["keep"])
            if operations["append"] != 1. or row["event_ids"] != tuple((i,) for i in range(index + 1)):
                raise AssertionError("min_fill=32 did not retain every initial observed endpoint")
        previous = n
    if prefull != capacity or previous != capacity or rejections:
        raise AssertionError("Fixed panel did not establish complete minimum filling")
    return {"prefull_calls": prefull, "prefull_rejections": rejections,
            "postfull_calls": len(rows) - prefull, "final_bank_events": previous}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/long_memory/cache_full1600_v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cache = EpisodeCache(args.cache_dir)
    if not {eid for eid, _ in PANEL} <= set(cache.manifest["splits"]["val"]):
        raise ValueError("Fixed integration panel must remain in cache-VAL")
    checkpoint = args.checkpoint.resolve(strict=True)
    info = inspect_checkpoint(cache.manifest["model_path"], checkpoint)
    cfg = info["config"]["echo"]
    if (info["stage"] != 2 or cfg["min_fill"] != 4 or cfg["capacity_events"] != 32
            or cfg["merge_threshold"] is not None):
        raise ValueError("Regression requires the stage-2 min_fill=4/capacity=32/no-merge checkpoint")
    output = args.output_dir.resolve()
    validate_output_scope(output, cache.path, checkpoint, Path(cache.manifest["model_path"]),
                          Path(cache.manifest["dataset_path"]))
    if output.exists():
        raise FileExistsError("Minimum-fill regression requires a NEW output directory")
    sources = _sources()
    binding = {"kind": KIND, "panel": PANEL, "checkpoint_path": str(checkpoint),
        "checkpoint_files_sha256": info["files_sha256"], "checkpoint_stage": info["stage"],
        "checkpoint_step": info["step"], "cache_fingerprint": cache.manifest["fingerprint"],
        "original_min_fill": 4, "effective_min_fill": 32, "device": args.device, **sources}
    if args.preflight_only:
        print("[echo-min-fill-verify] Preflight passed; no policy/GPU/simulator/output initialized.")
        return 0
    gpu = gpu_guard(args.device)
    output.mkdir(parents=True)
    _atomic_json(output / "plan.json", {**binding, "gpu_check": gpu})
    started = time.monotonic()
    policy = None
    try:
        policy = EchoMinFillPolicy(cache.manifest["model_path"], checkpoint,
                                   device=args.device, min_fill=None)
        before = _parameters(policy)
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        loader = LeRobotEpisodeLoader(Path(cache.manifest["dataset_path"]),
            policy.processor.modality_configs["new_embodiment"], video_backend="opencv")
        _, raw_records = _episode_records(Path(cache.manifest["dataset_path"]))
        records = {int(row["episode_id"]): row for row in raw_records}
        episodes = MappedEpisodes(cache)
        events, component_errors, fills = [], [], []
        for eid, count in PANEL:
            episode = episodes.fetch(eid)
            parent, parent_errors, parent_info = _branch(
                policy, loader, records[eid], episode, count, "parent", parent=True)
            control, control_errors, control_info = _branch(
                policy, loader, records[eid], episode, count, "wrapper-checkpoint")
            if parent != control:
                raise AssertionError("Wrapper default changed actions, inputs, bank, WRITE decisions or episode RNG")
            extra = {"min_fill_override", "original_min_fill", "effective_min_fill", "min_fill_source_sha256"}
            for left, right in zip(parent_info, control_info, strict=True):
                if left != {key: value for key, value in right.items() if key not in extra}:
                    raise AssertionError("Wrapper default changed original runtime information")
                if (right["min_fill_override"] is not None or right["original_min_fill"] != 4
                        or right["effective_min_fill"] != 4
                        or right["min_fill_source_sha256"] != sources["min_fill_source_sha256"]):
                    raise AssertionError("Wrapper default override provenance differs")
            cpu_rng = torch.random.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state(policy.model.device).clone()
            policy.min_fill_metadata = apply_min_fill_override(policy.representation, 32)
            if (not torch.equal(cpu_rng, torch.random.get_rng_state())
                    or not torch.equal(cuda_rng, torch.cuda.get_rng_state(policy.model.device))
                    or _parameters(policy) != before):
                raise AssertionError("Minimum-fill override changed model tensors or RNG")
            filled, filled_errors, filled_info = _branch(
                policy, loader, records[eid], episode, count, "wrapper-minfill32")
            for left, right, runtime in zip(control, filled, filled_info, strict=True):
                for key in ("query", "stored", "short", "hamlet_cache", "session_rng", "is_demo",
                            "rng_before_ae", "rng_after_ae"):
                    if left.get(key) != right.get(key):
                        raise AssertionError(f"Minimum-fill override changed fixed observed input or RNG: {key}")
                if (runtime["min_fill_override"] != 32 or runtime["original_min_fill"] != 4
                        or runtime["effective_min_fill"] != 32
                        or runtime["min_fill_source_sha256"] != sources["min_fill_source_sha256"]):
                    raise AssertionError("Runtime min_fill=32 provenance differs")
            fills.append({"episode_id": eid, **_validate_forced_fill(filled)})
            for role, rows, errors in (("parent", parent, parent_errors),
                    ("wrapper-checkpoint", control, control_errors), ("wrapper-minfill32", filled, filled_errors)):
                events.extend({**row, "role": role} for row in rows)
                component_errors.append({"episode_id": eid, "role": role, **errors})
        after = _parameters(policy)
        if before != after or inspect_checkpoint(cache.manifest["model_path"], checkpoint) != info:
            raise AssertionError("Regression changed model parameters or checkpoint files")
        if _sources() != sources:
            raise AssertionError("Bound verification/runtime source changed")
        _atomic_json(output / "events.json", events)
        result = {**binding, "passed": True, "control_same_as_parent": True,
            "minfill32_prefull_reject_zero": True, "minfill32_prefull_rejections": 0,
            "parameters_unchanged": True, "parameter_state_sha256_before": before,
            "parameter_state_sha256_after": after, "override_rng_unchanged": True,
            "minfill32_fixed_input_rng_unchanged": True, "fill_evidence": fills,
            "endpoints": len(events), "real_ae_calls": sum("action" in row for row in events),
            "calls_by_role": {role: sum(row["role"] == role for row in events)
                for role in ("parent", "wrapper-checkpoint", "wrapper-minfill32")},
            "component_parity": component_errors, "elapsed_seconds": time.monotonic() - started,
            "events_sha256": hashlib.sha256((output / "events.json").read_bytes()).hexdigest(),
            "interpretation": "Real-AE fixed-observation regression; not closed-loop task success"}
        _atomic_json(output / "completed.json", result)
        print(f"[echo-min-fill-verify] Passed: {result['real_ae_calls']} real AE calls; {output / 'completed.json'}", flush=True)
    finally:
        if policy is not None:
            policy.reset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
