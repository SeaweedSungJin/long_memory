#!/usr/bin/env python3
"""Real native VLM/AE causal replay; NOT a simulator success measurement.

Fixed cache-VAL panel includes a long demo overflowing FIFO32 and a no-demo
episode. READ ON/OFF must preserve WRITE decisions, HAMLET history and action
RNG consumption on identical teacher observations. No training or reseeding
between policy calls. The writer and parent checkpoints remain unchanged.
"""
import argparse
import json
import os
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
from run_scripts.robomme.cvom_admission_checkpoint import inspect_checkpoint
from run_scripts.robomme.diagnose_v19_runtime import gpu_snapshot
from run_scripts.robomme.verify_read_ablation_v19 import tensor_hash

PANEL = ((1355, 72), (626, 6))


@torch.inference_mode()
def replay(policy, loader, record, episode, count, memory_off):
    eid, frames = int(record["episode_id"]), episode["frames"][:count].tolist()
    if len(frames) != count:
        raise ValueError("Fixed panel does not fit this episode")
    table = loader._load_parquet_data(eid)
    language = policy.language_key
    if language == "task":
        if len(record["metadata"]["tasks"]) != 1:
            raise ValueError("Ambiguous task instruction")
        table["language.task"] = record["metadata"]["tasks"][0]
    videos = loader._load_video_data(eid, np.asarray(frames))
    core, head = policy.representation, policy.model.action_head
    original_read, original_write = core.read_from_bank, core._write
    original_generate = head.get_action_with_features
    rows, current, order = [], {}, []
    policy.memory_off = memory_off

    def read(short, encoded, bank, **kwargs):
        order.append("read")
        current.update(read_bank=tensor_hash(bank), short=tensor_hash(short), encoded=tensor_hash(encoded))
        return original_read(short, encoded, bank, **kwargs)

    def write(bank, encoded, **kwargs):
        order.append("write")
        assert tensor_hash(bank) == current["read_bank"]
        before = torch.cuda.get_rng_state(policy.model.device)
        result, metrics = original_write(bank, encoded, **kwargs)
        assert torch.equal(before, torch.cuda.get_rng_state(policy.model.device))
        current.update(bank=tensor_hash(result), bank_events=result.shape[1] // policy.n_q,
                       writer_metrics={k: float(v) for k, v in metrics.items()})
        return result, metrics

    def generate(*args, **kwargs):
        current["rng_before_ae"] = tensor_hash(head._inference_gen.get_state())
        result = original_generate(*args, **kwargs)
        current["rng_after_ae"] = tensor_hash(head._inference_gen.get_state())
        current["action"] = tensor_hash(result["action_pred"])
        return result

    sid = f"cvom-fixed-panel-{eid}"
    with patch.object(core, "read_from_bank", read), patch.object(core, "_write", write), \
            patch.object(head, "get_action_with_features", generate):
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
            assert order == ["read", "write"], order
            assert info["expert_adapted"] is True and info["feature_precision"] == "native"
            if memory_off:
                assert info["long_memory"]["read"]["ae_conditioning_delta_norm"] == 0
            assert current["bank_events"] == min(d + 1, core.config.capacity_events)
            if d == 0:
                assert current["bank_events"] == 1
            session = policy.sessions[sid]
            current.update(hamlet_cache=tensor_hash(session.short_cache),
                           session_rng=tensor_hash(session.generator.get_state()), is_demo=demo)
            assert ("action" in current) == (not demo)
            rows.append(dict(current))
    policy.sessions.clear()
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--writer-checkpoint", required=True, type=Path)
    p.add_argument("--cache-dir", type=Path, default=Path("runs/long_memory/cache_full1600_v1"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args(argv)
    info = inspect_checkpoint(a.writer_checkpoint)
    out = a.output_dir.resolve()
    if out.exists():
        raise FileExistsError("Use a new verification output")
    if os.environ.get("GR00T_INFERENCE_SEED") != "6":
        raise ValueError("Set GR00T_INFERENCE_SEED=6")
    cache = EpisodeCache(a.cache_dir)
    if not {eid for eid, _ in PANEL} <= set(cache.manifest["splits"]["val"]):
        raise ValueError("Panel must be cache-VAL")
    gpu = gpu_snapshot(a.device)
    out.mkdir(parents=True)
    _atomic_json(out / "plan.json", {"panel": PANEL, "checkpoint": info, "gpu": gpu,
        "cache_fingerprint": cache.manifest["fingerprint"], "kind": "teacher_observation_real_AE_not_success"})
    from run_scripts.robomme.policy_representation_v18 import RepresentationPolicyV18
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    policy = RepresentationPolicyV18(cache.manifest["model_path"], info["parent_identity"]["path"],
        device=a.device, feature_precision="native", writer_checkpoint=str(a.writer_checkpoint), cvom_admission=True)
    before = {"expert": expert_state_sha256(policy.model.action_head),
              "reader": _state_sha256(policy.representation.delta_state_dict()),
              "writer": _state_sha256(policy.writer.state_dict())}
    loader = LeRobotEpisodeLoader(Path(cache.manifest["dataset_path"]),
        policy.processor.modality_configs["new_embodiment"], video_backend="opencv")
    _, records = _episode_records(Path(cache.manifest["dataset_path"]))
    records = {int(row["episode_id"]): row for row in records}
    episodes, evidence, started = MappedEpisodes(cache), [], time.monotonic()
    for eid, count in PANEL:
        on, off = [replay(policy, loader, records[eid], episodes.fetch(eid), count, flag) for flag in (False, True)]
        for left, right in zip(on, off):
            assert {k: v for k, v in left.items() if k != "action"} == {k: v for k, v in right.items() if k != "action"}
        evidence += [{**row, "memory_off": flag} for flag, rows in ((False, on), (True, off)) for row in rows]
        print(f"[cvom-verify] episode={eid} {count} endpoints: causal WRITE/RNG/native short match", flush=True)
    after = {"expert": expert_state_sha256(policy.model.action_head),
             "reader": _state_sha256(policy.representation.delta_state_dict()),
             "writer": _state_sha256(policy.writer.state_dict())}
    assert before == after and inspect_checkpoint(a.writer_checkpoint) == info
    _atomic_json(out / "events.json", evidence)
    _atomic_json(out / "completed.json", {"passed": True, "real_ae_calls": sum("action" in r for r in evidence),
        "endpoints": len(evidence), "parameters_unchanged": before, "elapsed_seconds": time.monotonic()-started,
        "interpretation": "Native teacher-observation replay; not closed-loop success"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
