#!/usr/bin/env python3
"""Native ECHO replay on two fixed cache-VAL episodes, without a simulator.

Runs real HAMLET and AE on teacher observations. Checks actual-control
normalization against the immutable cache, completed-event offline/online
parity on matched inputs, READ-off WRITE/RNG invariance, and frozen parameters.
This is an integration diagnostic, never a task-success measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
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
from run_scripts.robomme.train_cvom_admission import gpu_guard
from run_scripts.robomme.verify_read_ablation_v19 import tensor_hash

PANEL = ((1355, 72), (626, 6))


@torch.inference_mode()
def replay(policy, loader, record, episode, count, memory_off):
    eid, frames = int(record["episode_id"]), episode["frames"][:count].tolist()
    if len(frames) != count:
        raise ValueError("Fixed replay panel does not fit this episode")
    table = loader._load_parquet_data(eid)
    language = policy.language_key
    if language == "task":
        if len(record["metadata"]["tasks"]) != 1:
            raise ValueError("Ambiguous instruction")
        table["language.task"] = record["metadata"]["tasks"][0]
    videos = loader._load_video_data(eid, np.asarray(frames))
    core, head = policy.representation, policy.model.action_head
    old_step, old_read, old_write, old_generate = core.step, core.read_from_bank, core.manager.update, head.get_action_with_features
    rows, captures, current, order = [], [], {}, []
    policy.memory_off = memory_off

    def read(*args, **kwargs):
        order.append("read")
        return old_read(*args, **kwargs)

    def write(*args, **kwargs):
        order.append("write")
        return old_write(*args, **kwargs)

    def step(short, moment, state, frame, demo, **kwargs):
        d = current["decision"]
        if d:
            controls = kwargs["completed_actions"].detach().cpu()[0]
            mask = kwargs["completed_action_mask"].detach().cpu()[0]
            torch.testing.assert_close(controls, episode["actions"][d-1], atol=0, rtol=0)
            if not torch.equal(mask, episode["action_mask"][d-1]):
                raise AssertionError("Native completed-control mask differs from cache")
        elif any(key in kwargs for key in ("completed_actions", "previous_moment", "previous_state")):
            raise AssertionError("First observation invents a completed transition")
        generator_before = head._inference_gen.get_state().clone()
        result = old_step(short, moment, state, frame, demo, **kwargs)
        if not torch.equal(generator_before, head._inference_gen.get_state()):
            raise AssertionError("Memory consumed action-generation randomness")
        if kwargs["write_enabled"] is not True or kwargs["read_enabled"] != (not memory_off and not bool(demo[0]) and d > 0):
            raise AssertionError("READ-off changed the WRITE/READ schedule")
        if memory_off:
            torch.testing.assert_close(result["fused"], result["short"], atol=0, rtol=0)
        captures.append({name: tensor.detach().cpu().clone() for name, tensor in {
            "short": short, "moment": moment, "state": state,
            "stored": result["stored_current"], "query": result["encoded_current"]}.items()})
        current.update(bank=tensor_hash(result["bank"]), query=tensor_hash(result["encoded_current"]),
            stored=tensor_hash(result["stored_current"]), short=tensor_hash(short),
            event_ids=result["bank_state"].event_ids,
            writer={k: float(v) for k, v in result["metrics"].items() if k.startswith("writer_")})
        return result

    def generate(*args, **kwargs):
        current["rng_before_ae"] = tensor_hash(head._inference_gen.get_state())
        result = old_generate(*args, **kwargs)
        current["rng_after_ae"] = tensor_hash(head._inference_gen.get_state())
        current["action"] = tensor_hash(result["action_pred"])
        return result

    sid = f"echo-replay-{eid}"
    with patch.object(core, "step", step), patch.object(core, "read_from_bank", read), \
            patch.object(core.manager, "update", write), patch.object(head, "get_action_with_features", generate):
        for d, frame in enumerate(frames):
            current.clear()
            order.clear()
            current.update(episode_id=eid, decision=d, frame=frame)
            observation = {
                "video": {key: videos[key][d][None, None] for key in policy.modality_configs["video"].modality_keys},
                "state": {key: np.asarray(table[f"state.{key}"].iloc[frame], np.float32)[None, None]
                          for key in policy.modality_configs["state"].modality_keys},
                "language": {language: [[str(table[f"language.{language}"].iloc[frame])]]}}
            controls = np.empty((0, 8), np.float32)
            if d and not bool(episode["is_demo"][d-1]):
                controls = np.concatenate([np.stack(table[f"action.{key}"].iloc[frames[d-1]:frame].to_list())
                    for key in policy.modality_configs["action"].modality_keys], axis=-1).astype(np.float32)
            demo = bool(episode["is_demo"][d])
            _, info = policy._get_action(observation, {"session_ids": [sid], "reset_memory": [d == 0],
                "episode_seed": 190021 + eid, "frame_index": frame, "passive": demo, "prime_only": demo,
                "executed_actions": controls})
            if order != ["read", "write"] or info["checkpoint_variant"] != "echo_cvom_v1":
                raise AssertionError("Native ECHO dispatch/read-before-write contract changed")
            session = policy.sessions[sid]
            current.update(hamlet_cache=tensor_hash(session.short_cache),
                session_rng=tensor_hash(session.generator.get_state()), is_demo=demo)
            rows.append(dict(current))
    # Match actual native observations first. Raw cached VLM tokens can differ
    # due to the established native/cache precision boundary; do not hide that
    # by globally changing backbone precision or demand false bitwise parity.
    matched = {name: torch.cat([row[name] for row in captures]) for name in ("short", "moment", "state")}
    matched.update({name: episode[name][:count if name in ("frames", "is_demo") else count-1]
                    for name in ("frames", "is_demo", "actions", "action_mask", "transition_valid")})
    with torch.autocast(device_type=policy.model.device.type, enabled=False):
        encoded = core.encode_prefix(matched, count)
    errors = {}
    for name, output in (("stored", "stored"), ("query", "query")):
        reference = torch.cat([row[name] for row in captures])
        actual = encoded[output].detach().cpu()
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
        errors[name + "_max_abs_error"] = float((actual-reference).abs().max())
    policy.reset({"session_ids": [sid]})
    return rows, errors


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path("runs/long_memory/cache_full1600_v1"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cache = EpisodeCache(args.cache_dir)
    if not {eid for eid, _ in PANEL} <= set(cache.manifest["splits"]["val"]):
        raise ValueError("Fixed integration panel must stay in cache-VAL")
    info = inspect_checkpoint(cache.manifest["model_path"], args.checkpoint)
    out = args.output_dir.resolve()
    validate_output_scope(out, cache.path, args.checkpoint.resolve(), Path(cache.manifest["model_path"]))
    if out.exists():
        raise FileExistsError("Integration verification requires a new output directory")
    gpu = gpu_guard(args.device)
    out.mkdir(parents=True)
    sources = runtime_source_identity()
    sources["verify_echo_cvom.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _atomic_json(out / "plan.json", {"panel": PANEL, "checkpoint": info, "source_sha256": sources,
        "cache_fingerprint": cache.manifest["fingerprint"], "device": args.device, "gpu_check": gpu,
        "kind": "native_teacher_observation_real_AE_not_success"})
    policy = EchoPolicyV1(cache.manifest["model_path"], args.checkpoint, device=args.device)
    before = {"core": _state_sha256(policy.representation.delta_state_dict()),
              "expert": expert_state_sha256(policy.model.action_head)}
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    loader = LeRobotEpisodeLoader(Path(cache.manifest["dataset_path"]),
        policy.processor.modality_configs["new_embodiment"], video_backend="opencv")
    _, raw_records = _episode_records(Path(cache.manifest["dataset_path"]))
    records = {int(row["episode_id"]): row for row in raw_records}
    episodes, events, component_errors, started = MappedEpisodes(cache), [], [], time.monotonic()
    for eid, count in PANEL:
        pairs = [replay(policy, loader, records[eid], episodes.fetch(eid), count, flag) for flag in (False, True)]
        on, off = pairs[0][0], pairs[1][0]
        for left, right in zip(on, off, strict=True):
            if {k: v for k, v in left.items() if k != "action"} != {k: v for k, v in right.items() if k != "action"}:
                raise AssertionError("READ-off changed writer, observation state or episode RNG")
        events += [{**row, "memory_off": flag} for flag, rows in ((False, on), (True, off)) for row in rows]
        component_errors += [{"episode_id": eid, "memory_off": flag, **pair[1]}
                             for flag, pair in zip((False, True), pairs)]
        print(f"[echo-verify] {eid}: {count} endpoints; actual-control normalization and completed-event parity passed", flush=True)
    after = {"core": _state_sha256(policy.representation.delta_state_dict()),
             "expert": expert_state_sha256(policy.model.action_head)}
    if before != after or inspect_checkpoint(cache.manifest["model_path"], args.checkpoint) != info:
        raise AssertionError("Verification changed model parameters or checkpoint")
    if any(runtime_source_identity()[key] != value for key, value in sources.items() if key != "verify_echo_cvom.py"):
        raise AssertionError("Runtime source changed during verification")
    _atomic_json(out / "events.json", events)
    _atomic_json(out / "completed.json", {"passed": True, "real_ae_calls": sum("action" in row for row in events),
        "endpoints": len(events), "component_parity": component_errors, "parameters_unchanged": before,
        "elapsed_seconds": time.monotonic()-started,
        "interpretation": "Native teacher-observation integration check; not closed-loop success"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
