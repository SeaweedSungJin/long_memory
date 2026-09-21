#!/usr/bin/env python3
"""READ-only V19 intervention check; frozen policy, teacher observations, no training.

Unlike the earlier precision panel, this runs every non-demo AE call with its
ordinary persistent episode generator: no per-query reseeding or denoising stub.
This is a path/RNG check, NOT a simulator success-rate evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from gr00t.long_memory.cache import EpisodeCache, _episode_records
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes
from gr00t.long_memory.expert_v4 import expert_state_dict, expert_state_sha256
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.verify_feature_precision_v19 import SOURCES, CHECKPOINT_FILES, sha
from run_scripts.robomme.diagnose_v19_runtime import gpu_snapshot, write_json

CHECKPOINT = "runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072"


def tensor_hash(t):
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256(str((list(t.shape), str(t.dtype))).encode())
    h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def replay(policy, loader, record, ep, off):
    eid = int(record["episode_id"])
    frames = ep["frames"].tolist()
    table = loader._load_parquet_data(eid)
    language = policy.language_key
    if language == "task":
        assert len(record["metadata"]["tasks"]) == 1
        table["language.task"] = record["metadata"]["tasks"][0]
    video = loader._load_video_data(eid, np.asarray(frames))
    head, core = policy.model.action_head, policy.representation
    old_step, old_generate = core.step, head.get_action_with_features
    rows, current = [], {}
    policy.memory_off = off
    sid = f"v19-read-{eid}"
    # Same derived-seed-style fixed positive integer in both passes. Rollout
    # scenario/inference seed identity is checked separately in its manifest.
    seed = 190021 + eid

    def step(short, moment, state, frame, demo, **kw):
        before = torch.cuda.get_rng_state(policy.model.device)
        out = old_step(short, moment, state, frame, demo, **kw)
        assert torch.equal(before, torch.cuda.get_rng_state(policy.model.device))
        assert kw["write_enabled"] is True
        d = current["decision"]
        assert kw["read_enabled"] == (not off and not bool(demo[0]) and d > 0)
        assert out["bank"].shape[1] == min(d + 1, core.config.capacity_events) * policy.n_q
        if off:
            assert torch.equal(out["fused"], out["short"])
        current.update(bank=tensor_hash(out["bank"]), short=tensor_hash(short),
                       encoded=tensor_hash(out["encoded_current"]),
                       read_enabled=kw["read_enabled"], write_enabled=kw["write_enabled"])
        return out

    def generate(*a, **kw):
        current["rng_before_ae"] = tensor_hash(head._inference_gen.get_state())
        value = old_generate(*a, **kw)
        current["rng_after_ae"] = tensor_hash(head._inference_gen.get_state())
        current["action"] = tensor_hash(value["action_pred"])
        return value

    with patch.object(core, "step", step), patch.object(head, "get_action_with_features", generate):
        for d, frame in enumerate(frames):
            current.clear()
            current.update(episode_id=eid, decision=d, frame=frame, memory_off=off)
            observation = {
                "video": {k: video[k][d][None, None] for k in policy.modality_configs["video"].modality_keys},
                "state": {k: np.asarray(table[f"state.{k}"].iloc[frame], np.float32)[None, None]
                          for k in policy.modality_configs["state"].modality_keys},
                "language": {language: [[str(table[f"language.{language}"].iloc[frame])]]},
            }
            controls = np.empty((0, 8), np.float32)
            if d and not bool(ep["is_demo"][d - 1]):
                controls = np.concatenate([np.stack(table[f"action.{k}"].iloc[frames[d - 1]:frame].to_list())
                    for k in policy.modality_configs["action"].modality_keys], axis=-1).astype(np.float32)
            demo = bool(ep["is_demo"][d])
            _, info = policy._get_action(observation, {"session_ids": [sid], "reset_memory": [d == 0],
                "episode_seed": seed, "frame_index": frame, "passive": demo, "prime_only": demo,
                "executed_actions": controls})
            session = policy.sessions[sid]
            assert info["expert_adapted"] is True and info["feature_precision"] == "native"
            assert info["long_memory"]["updates"] == d + 1 and info["long_memory"]["keeps"] == 0
            if off:
                assert info["long_memory"]["read"]["ae_conditioning_delta_norm"] == 0
            current.update(session_rng=tensor_hash(session.generator.get_state()),
                           hamlet_cache=tensor_hash(session.short_cache), is_demo=demo)
            rows.append(dict(current))
    policy.sessions.clear()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir).resolve()
    cache = EpisodeCache("runs/long_memory/cache_full1600_v1")
    checkpoint = ROOT / CHECKPOINT
    validate_output_scope(out, cache.path, checkpoint, cache.manifest["dataset_path"], cache.manifest["model_path"])
    if out.exists():
        raise FileExistsError("Use a NEW diagnostic directory")
    if os.environ.get("GR00T_INFERENCE_SEED") != "6":
        raise ValueError("Set GR00T_INFERENCE_SEED=6 as in the rollout evaluator")
    ids = [1355, 626]  # Fixed before results: long demo/FIFO overflow and no demo.
    assert set(ids) <= set(cache.manifest["splits"]["val"])
    bindings = {"checkpoint_files_sha256": {n: sha(checkpoint / n) for n in CHECKPOINT_FILES},
                "source_sha256": {n: sha(ROOT / n) for n in SOURCES}}
    out.mkdir(parents=True)
    write_json(out / "plan.json", {"episodes": ids, "feature_precision": "native", **bindings,
        "scope": "All canonical cached endpoints; real AE at every non-demo endpoint; identical teacher observations"})
    write_json(out / "gpu_before.json", gpu_snapshot("cuda:0"))
    from run_scripts.robomme.policy_representation_v18 import RepresentationPolicyV18
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    policy = RepresentationPolicyV18(cache.manifest["model_path"], str(checkpoint), device="cuda:0",
                                    memory_off=True, feature_precision="native")
    head = policy.model.action_head
    disk = load_file(str(checkpoint / "expert.safetensors"))
    installed = expert_state_dict(head)
    assert set(installed) == set(disk) and all(torch.equal(installed[n].cpu(), disk[n]) for n in disk)
    assert sum(int(torch.count_nonzero(v)) for n, v in disk.items() if n.endswith("lora_B")) > 0
    expert_before = expert_state_sha256(head)
    assert not any(p.requires_grad for p in policy.model.parameters())
    loader = LeRobotEpisodeLoader(Path(cache.manifest["dataset_path"]),
                                  policy.processor.modality_configs["new_embodiment"], video_backend="opencv")
    _, records = _episode_records(Path(cache.manifest["dataset_path"]))
    records = {int(r["episode_id"]): r for r in records}
    episodes, evidence, compared = MappedEpisodes(cache), [], 0
    for eid in ids:
        ep = episodes.fetch(eid)
        on, off = (replay(policy, loader, records[eid], ep, value) for value in (False, True))
        for a, b in zip(on, off):
            for key in ("frame", "is_demo", "bank", "short", "encoded", "hamlet_cache", "session_rng",
                        "rng_before_ae", "rng_after_ae", "write_enabled"):
                assert a.get(key) == b.get(key), (eid, a["decision"], key)
            compared += 1
        evidence += on + off
        print(f"[read-check] episode={eid} endpoints={len(on)} ON/OFF RNG/cache/FIFO match", flush=True)
    assert expert_state_sha256(head) == expert_before
    assert bindings["checkpoint_files_sha256"] == {n: sha(checkpoint / n) for n in CHECKPOINT_FILES}
    assert bindings["source_sha256"] == {n: sha(ROOT / n) for n in SOURCES}
    write_json(out / "events.json", evidence)
    result = {"passed": True, "kind": "v19_native_read_ablation_check", **bindings,
        "endpoints_per_condition": compared, "real_ae_calls": sum("action" in r for r in evidence),
        "expert_tensor_count": len(installed), "expert_tensor_sha256": expert_before,
        "expert_matches_checkpoint": True, "expert_unchanged": True, "read_off_keeps_fifo": True,
        "persistent_episode_rng_matches": True, "hamlet_cache_matches": True,
        "evidence_sha256": {n: sha(out / n) for n in ("events.json", "plan.json", "gpu_before.json")},
        "limitation": "Teacher observations, not closed-loop successes; changed generated actions are allowed."}
    write_json(out / "completed.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
