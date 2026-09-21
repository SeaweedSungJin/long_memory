#!/usr/bin/env python3
"""Actual semantic policy ON/OFF invariants on a fixed held-out teacher panel.

This is NOT closed-loop RoboMME success evaluation. Every non-demo endpoint
runs the genuine AE denoiser, without stubs or per-query reseeding. The panel
is fixed by chronology (long demo/overflow and no demo), not measured outcomes.
KEEP/REPLACE decisions are legal: the invariant is identical causal WRITE under
identical observations, not "every write must insert".
"""
from __future__ import annotations

import argparse
import importlib.metadata
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
from safetensors.torch import load_file

from gr00t.long_memory.cache import EpisodeCache, _episode_records
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes
from gr00t.long_memory.expert_v4 import expert_state_dict, expert_state_sha256
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.diagnose_v19_runtime import gpu_snapshot, write_json
from run_scripts.robomme.verify_read_ablation_v19 import tensor_hash
from run_scripts.robomme.verify_feature_precision_v19 import sha
from run_scripts.robomme.semantic_memory_checkpoint import semantic_info

PANEL = ((1355, 72), (626, 6))
RUNTIME_SOURCES = (
    "policy_representation_v18.py", "representation_core_v18.py", "checkpoint_representation_v18.py",
    "semantic_memory_storage.py", "semantic_memory_checkpoint.py", "feature_precision_v19.py",
    "verify_semantic_memory.py", "verify_read_ablation_v19.py", "diagnose_v19_runtime.py",
)


def optional_tensor_hash(value):
    return None if value is None else tensor_hash(torch.as_tensor(value))


@torch.inference_mode()
def replay(policy, loader, record, episode, *, endpoint_limit, memory_off):
    eid = int(record["episode_id"])
    frames = episode["frames"][:endpoint_limit].tolist()
    if len(frames) != endpoint_limit:
        raise ValueError("Fixed diagnostic panel does not fit this cache")
    table = loader._load_parquet_data(eid)
    language = policy.language_key
    if language == "task":
        if len(record["metadata"]["tasks"]) != 1:
            raise ValueError("Ambiguous task instruction")
        table["language.task"] = record["metadata"]["tasks"][0]
    videos = loader._load_video_data(eid, np.asarray(frames))
    head, core = policy.model.action_head, policy.representation
    old_step, old_read, old_write = core.step, core.read_from_bank, core._write
    old_generate, old_prepare, old_process = head.get_action_with_features, policy.model.prepare_input, head.process_backbone_output
    policy.memory_off = memory_off
    sid, seed = f"semantic-panel-{eid}", 190021 + eid
    rows, current, order = [], {}, []

    def prepare(*args, **kwargs):
        backbone, action = old_prepare(*args, **kwargs)
        current.update(action_input_mask=optional_tensor_hash(action.get("action_mask")),
                       embodiment_id=optional_tensor_hash(action.get("embodiment_id")),
                       prepared_state=optional_tensor_hash(action.get("state")))
        return backbone, action

    def process(*args, **kwargs):
        result = old_process(*args, **kwargs)
        current.update(attention_mask=optional_tensor_hash(result.get("backbone_attention_mask")),
                       image_mask=optional_tensor_hash(result.get("image_mask")))
        return result

    def read(short, encoded, bank, **kwargs):
        order.append("read")
        current["read_bank"] = tensor_hash(bank)
        return old_read(short, encoded, bank, **kwargs)

    def write(bank, encoded, **kwargs):
        order.append("write")
        assert current["read_bank"] == tensor_hash(bank), "WRITE happened before the recorded READ"
        result, metrics = old_write(bank, encoded, **kwargs)
        current.update(write_operation_code=float(metrics.get("writer_operation_code", 0)),
                       write_victim_index=float(metrics.get("writer_selected_index", -1)),
                       write_inserted=bool(metrics.get("writer_insert", metrics.get("write_rate", 1))),
                       writer_score=float(metrics.get("writer_score", 0)))
        return result, metrics

    def step(short, moment, state, frame, demo, **kwargs):
        global_before = torch.cuda.get_rng_state(policy.model.device)
        order.clear()
        result = old_step(short, moment, state, frame, demo, **kwargs)
        assert torch.equal(global_before, torch.cuda.get_rng_state(policy.model.device)), "Memory path consumed action RNG"
        assert order == ["read", "write"], order
        d = current["decision"]
        assert kwargs["write_enabled"] is True
        assert kwargs["read_enabled"] == (not memory_off and not bool(demo[0]) and d > 0)
        assert result["bank"].shape[1] == min(d + 1, core.config.capacity_events) * policy.n_q
        if d == 0:
            assert kwargs.get("bank") is None, "Session reset retained an old bank"
        if memory_off:
            assert torch.equal(result["fused"], result["short"])
        current.update(bank=tensor_hash(result["bank"]), short=tensor_hash(short),
            encoded=tensor_hash(result["encoded_current"]), stored=tensor_hash(result["stored_current"]),
            read_enabled=kwargs["read_enabled"], write_enabled=True,
            bank_events=result["bank"].shape[1] // policy.n_q)
        return result

    def generate(*args, **kwargs):
        current["rng_before_ae"] = tensor_hash(head._inference_gen.get_state())
        result = old_generate(*args, **kwargs)  # genuine configured denoising
        current["rng_after_ae"] = tensor_hash(head._inference_gen.get_state())
        current["generated_action"] = tensor_hash(result["action_pred"])
        return result

    with patch.object(core, "step", step), patch.object(core, "read_from_bank", read), \
            patch.object(core, "_write", write), patch.object(head, "get_action_with_features", generate), \
            patch.object(policy.model, "prepare_input", prepare), patch.object(head, "process_backbone_output", process):
        for decision, frame in enumerate(frames):
            current.clear()
            current.update(episode_id=eid, decision=decision, frame=frame, memory_off=memory_off,
                episode_seed=seed, cached_executed_prefix_mask=tensor_hash(episode["action_mask"][decision]))
            observation = {
                "video": {key: videos[key][decision][None, None] for key in policy.modality_configs["video"].modality_keys},
                "state": {key: np.asarray(table[f"state.{key}"].iloc[frame], np.float32)[None, None]
                          for key in policy.modality_configs["state"].modality_keys},
                "language": {language: [[str(table[f"language.{language}"].iloc[frame])]]},
            }
            controls = np.empty((0, 8), np.float32)
            if decision and not bool(episode["is_demo"][decision - 1]):
                controls = np.concatenate([
                    np.stack(table[f"action.{key}"].iloc[frames[decision-1]:frame].to_list())
                    for key in policy.modality_configs["action"].modality_keys], axis=-1).astype(np.float32)
            demo = bool(episode["is_demo"][decision])
            _, info = policy._get_action(observation, {"session_ids": [sid], "reset_memory": [decision == 0],
                "episode_seed": seed, "frame_index": frame, "passive": demo, "prime_only": demo,
                "executed_actions": controls})
            session, memory = policy.sessions[sid], info["long_memory"]
            assert info["expert_adapted"] is True and info["feature_precision"] == "native"
            assert info["semantic_manifest_sha256"] == policy.semantic_manifest_sha256
            assert info["storage_manager_sha256"] == policy.storage_manager_sha256
            assert memory["write_attempts"] == decision + 1
            assert memory["updates"] + memory["keeps"] == decision + 1
            assert memory["policy"] == "semantic-cvom"
            assert ("generated_action" in current) is (not demo)
            if memory_off:
                assert memory["read"]["ae_conditioning_delta_norm"] == 0
            current.update(session_rng=tensor_hash(session.generator.get_state()),
                hamlet_cache=tensor_hash(session.short_cache), is_demo=demo, write_attempts=memory["write_attempts"],
                updates=memory["updates"], keeps=memory["keeps"], executed_action_count=len(controls),
                read_delta_norm=memory["read"]["ae_conditioning_delta_norm"])
            rows.append(dict(current))
    policy.sessions.clear()
    return rows


def compare_passes(on, off):
    if len(on) != len(off):
        raise AssertionError("ON/OFF endpoint counts differ")
    same = ("episode_id", "decision", "frame", "is_demo", "episode_seed", "bank", "read_bank", "short",
        "encoded", "stored", "hamlet_cache", "session_rng", "rng_before_ae", "rng_after_ae",
        "write_enabled", "bank_events", "write_operation_code", "write_victim_index", "write_inserted",
        "writer_score", "action_input_mask", "cached_executed_prefix_mask", "attention_mask", "image_mask",
        "embodiment_id", "prepared_state", "write_attempts", "updates", "keeps", "executed_action_count")
    for a, b in zip(on, off):
        for key in same:
            if a.get(key) != b.get(key):
                raise AssertionError((a["episode_id"], a["decision"], key, a.get(key), b.get(key)))
    return {"equal_fields": list(same), "endpoints": len(on),
            "generated_actions_changed": sum(a.get("generated_action") != b.get("generated_action") for a, b in zip(on, off))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path("runs/long_memory/cache_full1600_v1"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    out, checkpoint = args.output_dir.resolve(), args.checkpoint.resolve()
    cache = EpisodeCache(args.cache_dir)
    validate_output_scope(out, cache.path, checkpoint, cache.manifest["dataset_path"], cache.manifest["model_path"])
    if out.exists():
        raise FileExistsError("Use a NEW diagnostic directory; nothing overwritten")
    if os.environ.get("GR00T_INFERENCE_SEED") != "6":
        raise ValueError("Set GR00T_INFERENCE_SEED=6 as in evaluation")
    semantic = semantic_info(checkpoint)
    if semantic["storage_config"] is None:
        raise ValueError("This diagnostic requires a learned semantic manager")
    if not {eid for eid, _ in PANEL} <= set(cache.manifest["splits"]["val"]):
        raise ValueError("Fixed panel must remain entirely cache-VAL")
    gpu = gpu_snapshot(args.device)  # refuses to disturb any current GPU work
    files = ("checkpoint.json", "model.safetensors", "expert.safetensors", "semantic.json", *semantic["payload_sha256"])
    source_paths = [path.relative_to(ROOT) for path in (ROOT / "gr00t").rglob("*.py")]
    source_paths += [Path("run_scripts/robomme") / name for name in RUNTIME_SOURCES]
    bindings = {"checkpoint_files_sha256": {name: sha(checkpoint / name) for name in files},
                "source_sha256": {str(path): sha(ROOT / path) for path in sorted(source_paths)}}
    out.mkdir(parents=True)
    write_json(out / "gpu_before.json", gpu)
    write_json(out / "plan.json", {"panel": [{"episode_id": eid, "first_endpoints": count} for eid, count in PANEL],
        "checkpoint": str(checkpoint), "cache_dir": str(cache.path), "dataset": cache.manifest["dataset_path"],
        "base_model": cache.manifest["model_path"], "cache_fingerprint": cache.manifest.get("fingerprint"),
        "feature_precision": "native", "inference_seed_environment": 6, "persistent_seed_formula": "190021 + episode_id",
        "semantic_manifest": semantic, **bindings,
        "scope": "Identical teacher observations; real AE for every non-demo endpoint; no policy training/simulator"})
    started = time.monotonic()
    from run_scripts.robomme.policy_representation_v18 import RepresentationPolicyV18
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    policy = RepresentationPolicyV18(cache.manifest["model_path"], str(checkpoint), device=args.device,
                                    memory_off=True, feature_precision="native", semantic_memory=True)
    head = policy.model.action_head
    disk, installed = load_file(str(checkpoint / "expert.safetensors")), expert_state_dict(head)
    assert set(installed) == set(disk) and all(torch.equal(installed[n].cpu(), disk[n]) for n in disk)
    assert sum(int(torch.count_nonzero(value)) for name, value in disk.items() if name.endswith("lora_B")) > 0
    manager_disk, manager_state = load_file(str(checkpoint / "storage.safetensors")), policy.writer.state_dict()
    assert set(manager_disk) == set(manager_state) and all(torch.equal(manager_state[n].cpu(), manager_disk[n]) for n in manager_disk)
    expert_before = expert_state_sha256(head)
    assert not any(parameter.requires_grad for parameter in policy.model.parameters())
    assert not any(parameter.requires_grad for parameter in policy.writer.parameters())
    loader = LeRobotEpisodeLoader(Path(cache.manifest["dataset_path"]),
                                  policy.processor.modality_configs["new_embodiment"], video_backend="opencv")
    _, records = _episode_records(Path(cache.manifest["dataset_path"]))
    records = {int(record["episode_id"]): record for record in records}
    episodes, evidence, comparisons = MappedEpisodes(cache), [], []
    for eid, count in PANEL:
        episode = episodes.fetch(eid)
        on, off = (replay(policy, loader, records[eid], episode, endpoint_limit=count, memory_off=value)
                   for value in (False, True))
        comparisons.append({"episode_id": eid, **compare_passes(on, off)})
        evidence.extend(on + off)
        print(f"[semantic-check] episode={eid} endpoints={count}; ON/OFF bank/cache/mask/writer/RNG match", flush=True)
    assert expert_state_sha256(head) == expert_before
    assert all(torch.equal(policy.writer.state_dict()[name].cpu(), manager_disk[name]) for name in manager_disk)
    assert bindings["checkpoint_files_sha256"] == {name: sha(checkpoint / name) for name in files}
    assert bindings["source_sha256"] == {str(path): sha(ROOT / path) for path in sorted(source_paths)}
    write_json(out / "events.json", evidence)
    result = {"passed": True, "kind": "semantic_native_read_policy_invariants", **bindings,
        "endpoints_per_condition": sum(count for _, count in PANEL),
        "real_ae_calls": sum("generated_action" in row for row in evidence),
        "write_attempts": len(evidence), "keep_calls": sum(not row["write_inserted"] for row in evidence),
        "replace_calls": sum(row["write_operation_code"] == 2 for row in evidence),
        "merge_calls": sum(row["write_operation_code"] == 3 for row in evidence),
        "comparisons": comparisons, "expert_tensor_count": len(installed), "expert_tensor_sha256": expert_before,
        "expert_matches_checkpoint": True, "expert_unchanged": True, "manager_matches_checkpoint": True,
        "manager_unchanged": True, "read_off_keeps_same_writer": True,
        "persistent_episode_rng_matches": True, "masks_and_cadence_match": True, "hamlet_cache_matches": True,
        "num_inference_timesteps": head.num_inference_timesteps, "capacity_events": policy.representation.config.capacity_events,
        "elapsed_seconds": time.monotonic() - started,
        "package_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors", "numpy")},
        "evidence_sha256": {name: sha(out / name) for name in ("events.json", "plan.json", "gpu_before.json")},
        "limitation": "Teacher observations, not closed-loop successes; generated actions may differ. Smoke checkpoint is not trained performance evidence."}
    write_json(out / "completed.json", result)
    print(json.dumps({key: result[key] for key in ("passed", "endpoints_per_condition", "real_ae_calls", "keep_calls", "replace_calls", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
