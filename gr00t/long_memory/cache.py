"""Frozen HAMLET features and causal episode caches for the long-memory experiment.

This is deliberately separate from the original HAMLET dataset/trainer. A cache
contains *whole sampled episodes*, not independent shuffled action windows. Event
``i`` spans ``frames[i] -> frames[i+1]`` and is only available to decisions
``j >= i+1``. Its recorded actions never extend beyond the result observation.

Features are stored AFTER VLLN and the original short-memory Transformer. The
training bridge must not apply either transformation a second time. The original
checkpoint processor supplies all state/action normalization and padding masks.
Only frozen inputs are cached; trainable event/key/value embeddings are rebuilt.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch


CACHE_VERSION = 1


def decision_frames(is_demo: np.ndarray, stride: int) -> np.ndarray:
    """Sample passive demos and policy calls, including the final observation.

    Align demos backwards from the first execution observation. Thus the last
    K-1 demo entries equal RoboMME's HAMLET priming indices (clamped repetitions
    are supplied by the original rolling cache), while earlier cues remain
    available to long memory. A final short event preserves episode tails.
    """
    demo = np.asarray(is_demo, dtype=bool)
    if demo.ndim != 1 or len(demo) < 2 or stride <= 0:
        raise ValueError("Need a >=2-frame episode and a positive stride")
    execution = np.flatnonzero(~demo)
    if not len(execution):
        raise ValueError("Episode contains no execution observations")
    start = int(execution[0])
    if demo[start:].any():
        raise ValueError("is_demo must be one contiguous prefix, then execution")
    frames = {0, len(demo) - 1}
    frames.update(range(start, -1, -stride))
    frames.update(range(start, len(demo), stride))
    return np.asarray(sorted(frames), dtype=np.int64)


def split_episodes(
    episodes: list[dict[str, Any]], val_fraction: float, seed: int,
    max_episodes: int | None = None,
) -> dict[str, list[int]]:
    """Deterministic episode-level, task-stratified split with no frame leakage.

    ``task`` is a benchmark task or provenance block when known, otherwise the
    recorded instruction. Small budgets first select paired episodes per task,
    so even a two-episode smoke run has one training and one validation episode.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    if len({int(e["episode_id"]) for e in episodes}) != len(episodes):
        raise ValueError("Duplicate episode IDs")
    rng = random.Random(seed)
    grouped: dict[str, list[int]] = defaultdict(list)
    for e in episodes:
        grouped[str(e["task"])].append(int(e["episode_id"]))
    groups = sorted(grouped)
    rng.shuffle(groups)
    for task in groups:
        grouped[task].sort()
        rng.shuffle(grouped[task])
    budget = len(episodes) if max_episodes is None else min(max_episodes, len(episodes))
    if budget < 2:
        raise ValueError("At least two episodes are required for held-out validation")
    selected: dict[str, list[int]] = defaultdict(list)
    # Paired round-robin prevents a small smoke subset from consisting solely of
    # one episode from each unrelated task, with no within-task validation.
    remaining = budget
    while remaining:
        progress = False
        for task in groups:
            take = min(2, len(grouped[task]), remaining)
            if take:
                selected[task].extend(grouped[task][:take])
                del grouped[task][:take]
                remaining -= take
                progress = True
            if not remaining:
                break
        if not progress:
            break
    train, val = [], []
    for task in groups:
        ids = selected[task]
        if len(ids) >= 2:
            n_val = min(len(ids) - 1, max(1, round(len(ids) * val_fraction)))
            val.extend(ids[:n_val])
            train.extend(ids[n_val:])
        else:
            train.extend(ids)
    if not val:
        rng.shuffle(train)
        n_val = min(len(train) - 1, max(1, round(len(train) * val_fraction)))
        val, train = train[:n_val], train[n_val:]
    return {"train": sorted(train), "val": sorted(val)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_episode(episode: dict[str, Any]) -> None:
    """Fail closed on stale shapes, nonfinite values, and future-action leakage."""
    required = {
        "episode_id", "task", "frames", "features", "attention_masks", "image_masks",
        "moment", "short", "state", "targets", "target_mask", "actions",
        "action_mask", "transition_valid", "decision_mask", "embodiment_id",
    }
    missing = required - episode.keys()
    _require(not missing, f"Missing cache fields: {sorted(missing)}")
    frames = torch.as_tensor(episode["frames"])
    _require(frames.ndim == 1 and len(frames) >= 2, "Need at least two endpoints")
    _require(bool((frames[1:] > frames[:-1]).all()), "Frames must strictly increase")
    n_events = len(frames) - 1
    targets, actions = episode["targets"], episode["actions"]
    _require(targets.ndim == 3 and targets.shape[0] == n_events, "targets shape must be [T,H,A]")
    _require(actions.ndim == 3 and actions.shape[0] == n_events, "actions shape must be [T,C,A]")
    _require(actions.shape[-1] == targets.shape[-1], "Action dimensions disagree")
    _require(episode["target_mask"].shape == targets.shape, "target_mask shape mismatch")
    _require(episode["action_mask"].shape == actions.shape[:2], "action_mask shape mismatch")
    for name in ("transition_valid", "decision_mask"):
        _require(episode[name].shape == (n_events,), f"{name} shape mismatch")
        _require(episode[name].dtype == torch.bool, f"{name} must be bool")
    _require(episode["action_mask"].dtype == torch.bool, "action_mask must be bool")
    decisions = episode["decision_mask"]
    _require(not bool(episode["target_mask"][~decisions].any()), "Passive demo has action-loss target")
    _require(not bool(episode["action_mask"][~decisions].any()), "Passive demo has recorded action prefix")
    gaps = frames[1:] - frames[:-1]
    positions = torch.arange(actions.shape[1]).unsqueeze(0)
    beyond = positions >= gaps.unsqueeze(1)
    _require(not bool(episode["action_mask"][beyond].any()), "Recorded action extends beyond next observation")
    _require(not bool(actions[~episode["action_mask"]].any()), "Padded/passive executed actions must be zero")
    _require(bool((gaps <= actions.shape[1]).all()), "Endpoint cadence exceeds action-prefix capacity")
    _require(episode["moment"].ndim == 3, "moment shape must be [T+1,Q,D]")
    _require(episode["short"].shape == episode["moment"].shape, "short/moment shape mismatch")
    _require(episode["moment"].shape[0] == n_events + 1, "Missing moment endpoint")
    _require(episode["state"].ndim == 2 and episode["state"].shape[0] == n_events + 1, "state shape mismatch")
    q, d = episode["short"].shape[-2:]
    for name in ("features", "attention_masks", "image_masks"):
        _require(len(episode[name]) == n_events + 1, f"{name} endpoint count mismatch")
    for idx, features in enumerate(episode["features"]):
        _require(features.ndim == 2 and features.shape[1] == d, "Feature width mismatch")
        _require(features.shape[0] >= q, "Features omit short-memory tokens")
        for name in ("attention_masks", "image_masks"):
            mask = episode[name][idx]
            _require(mask.shape == features.shape[:1] and mask.dtype == torch.bool, f"{name} shape/dtype mismatch")
        _require(torch.equal(features[-q:], episode["short"][idx]), "Conditioning tail is not short-memory output")
    if "is_demo" in episode:
        demo = episode["is_demo"]
        _require(demo.shape == frames.shape and demo.dtype == torch.bool, "Endpoint demo mask mismatch")
        _require(not bool((decisions & demo[:-1]).any()), "Demo endpoint used as a decision")
    for name in ("moment", "short", "state", "targets", "target_mask", "actions"):
        _require(bool(torch.isfinite(episode[name]).all()), f"Nonfinite {name}")
    for features in episode["features"]:
        _require(bool(torch.isfinite(features).all()), "Nonfinite conditioning features")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_signature(path: Path, *, hash_contents: bool = False) -> dict[str, Any]:
    path = path.resolve(strict=True)
    stat = path.stat()
    result: dict[str, Any] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hash_contents:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _atomic_json(path: Path, data: Any) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch_save(path: Path, data: Any) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp-")
    os.close(fd)
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class EpisodeCache:
    """Read a finished cache; arbitrary legacy pickle objects are not loaded."""

    def __init__(self, path: str | Path, *, allow_incomplete: bool = False):
        self.path = Path(path).resolve()
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("format_version") != CACHE_VERSION:
            raise ValueError("Unsupported cache format; rebuild with this version")
        if not allow_incomplete and self.manifest.get("status") != "complete":
            raise ValueError("Cache is incomplete. Resume cache_long_memory.py first")
        self._records = {int(e["episode_id"]): e for e in self.manifest["episodes"]}
        splits = self.manifest["splits"]
        train, val = set(splits["train"]), set(splits["val"])
        _require(bool(train) and bool(val) and not train & val, "Invalid episode train/val split")
        _require(train | val == self._records.keys(), "Split and episode files disagree")

    def load(self, episode_id: int) -> dict[str, Any]:
        record = self._records[int(episode_id)]
        path = (self.path / record["path"]).resolve()
        if not path.is_relative_to(self.path):
            raise ValueError("Episode path escapes cache directory")
        episode = torch.load(path, map_location="cpu", weights_only=True)
        validate_episode(episode)
        _require(int(episode["episode_id"]) == int(episode_id), "Episode file ID mismatch")
        _require(episode.get("cache_fingerprint") == self.manifest["fingerprint"], "Stale episode fingerprint")
        return episode


@dataclass
class CacheConfig:
    model_path: str
    dataset_path: str
    output_dir: str
    max_episodes: int = 16
    val_fraction: float = 0.2
    seed: int = 42
    device: str = "cuda:0"
    video_backend: str = "opencv"
    embodiment: str = "new_embodiment"


def _episode_records(dataset: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info = json.loads((dataset / "meta/info.json").read_text())
    metadata = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    block_for_id = {}
    subset_path = dataset / "subset_manifest.json"
    if subset_path.exists():
        # Our existing reindexed subset manifest documents contiguous task blocks.
        subset = json.loads(subset_path.read_text())
        offset = 0
        for block, originals in sorted(subset.get("episode_indices_by_block", {}).items(), key=lambda item: int(item[0])):
            for _ in originals:
                block_for_id[offset] = f"robomme_block_{int(block):02d}"
                offset += 1
        if offset != len(metadata):
            block_for_id = {}  # Do not guess a partially documented reindexing.
    records = []
    for row in metadata:
        episode_id = int(row["episode_index"])
        task = row.get("task_name") or row.get("task") or block_for_id.get(episode_id)
        if not task:
            task = " | ".join(str(t) for t in row.get("tasks", [])) or "unknown_task"
        records.append({"episode_id": episode_id, "task": str(task), "metadata": row})
    return info, records


def _cache_identity(config: CacheConfig, info: dict, records: list[dict], splits: dict) -> dict:
    from gr00t.utils.video_utils import resolve_backend

    model, dataset = Path(config.model_path).resolve(), Path(config.dataset_path).resolve()
    checkpoint = []
    for path in sorted(model.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".safetensors") and ".cache" not in path.parts:
            checkpoint.append(_file_signature(path, hash_contents=path.suffix == ".json"))
    if not any(s["path"].endswith(".safetensors") for s in checkpoint):
        raise FileNotFoundError("No local safetensors checkpoint: download the model first")
    metadata = [_file_signature(p, hash_contents=True) for p in sorted((dataset / "meta").glob("*.json*"))]
    if (dataset / "subset_manifest.json").exists():
        metadata.append(_file_signature(dataset / "subset_manifest.json", hash_contents=True))
    payloads = []
    modality_meta = json.loads((dataset / "meta/modality.json").read_text())
    # RoboMME's exported video features are labelled dtype='image' in info.json;
    # the modality mapping, not that dtype label, identifies actual MP4 paths.
    video_keys = sorted({value.get("original_key", f"observation.images.{key}")
                         for key, value in modality_meta.get("video", {}).items()})
    for row in records:
        episode_id = row["episode_id"]
        substitutions = {"episode_index": episode_id, "episode_chunk": episode_id // info["chunks_size"]}
        payloads.append(_file_signature(dataset / info["data_path"].format(**substitutions)))
        for key in video_keys:
            video_path = dataset / info["video_path"].format(video_key=key, **substitutions)
            signature = _file_signature(video_path)
            signature["resolved_backend"] = resolve_backend(str(video_path), config.video_backend)
            payloads.append(signature)
    # Size/mtime identify large payloads without rereading several GB on resume.
    # Metadata/processor JSONs are content-hashed. This is not a cryptographic
    # content guarantee for large weights; the convention is immutable raw data.
    repo = Path(__file__).resolve().parents[1]
    code_paths = [Path(__file__), repo / "model/gr00t_n1d6/gr00t_n1d6.py",
                  repo / "model/gr00t_n1d6/processing_gr00t_n1d6.py",
                  repo / "model/modules/eagle_backbone.py", repo / "model/modules/memory.py"]
    code_signatures = [_file_signature(p, hash_contents=True) for p in code_paths]
    return {
        "version": CACHE_VERSION, "checkpoint": checkpoint, "metadata": metadata, "code": code_signatures,
        "payloads": payloads, "splits": splits, "seed": config.seed,
        "val_fraction": config.val_fraction, "max_episodes": config.max_episodes,
        "video_backend": config.video_backend, "embodiment": config.embodiment,
        "feature_point": "post_vlln_post_short_memory", "demo_sampling": "backward_aligned_full_history",
        "storage_dtype": "bfloat16_features_float32_state_action",
    }


def _extract_episode(model, processor, loader, record: dict, config: CacheConfig) -> dict:
    """Run only frozen backbone and short memory, never the action DiT."""
    import pandas as pd
    from gr00t.data.types import EmbodimentTag, MessageType, VLAStepData

    episode_id = record["episode_id"]
    parquet = loader.dataset_path / loader.data_path_pattern.format(
        episode_chunk=episode_id // loader.chunk_size, episode_index=episode_id,
    )
    raw = pd.read_parquet(parquet)
    _require("is_demo" in raw.columns, "RoboMME cache requires explicit is_demo; do not guess passive actions")
    _require(len(raw) == int(record["metadata"]["length"]), "Episode length differs from metadata")
    demo = raw["is_demo"].to_numpy(dtype=bool)
    stride = int(model.config.memory_stride)
    frames = decision_frames(demo, stride)
    table = loader._load_parquet_data(episode_id)
    modalities = processor.modality_configs[config.embodiment]
    lang_key = modalities["language"].modality_keys[0]
    if lang_key in ("task", "sub_task"):
        # Avoid random language choices hidden inside LeRobotEpisodeLoader.
        if lang_key != "task" or len(record["metadata"].get("tasks", [])) != 1:
            raise ValueError("Use a deterministic frame-level language annotation for this dataset")
        table[f"language.{lang_key}"] = record["metadata"]["tasks"][0]
    video = loader._load_video_data(episode_id, frames)
    action_offsets = np.asarray(modalities["action"].delta_indices, dtype=np.int64)
    _require(np.array_equal(action_offsets, np.arange(len(action_offsets))), "Expected contiguous action deltas starting at zero")
    _require(modalities["state"].delta_indices == [0], "Only the current state is supported")
    _require(stride <= len(action_offsets), "Executed stride exceeds checkpoint action horizon")
    n_q = int(model.config.n_moment_tokens)
    features, attention, image_masks, moment, short, states = [], [], [], [], [], []
    targets, target_masks, actions, action_masks, decisions = [], [], [], [], []
    head = model.action_head
    head.reset_memory()
    device = torch.device(config.device)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    with torch.inference_mode(), amp:
        for idx, frame in enumerate(frames):
            frame = int(frame)
            actual_indices = frame + action_offsets
            clipped = np.minimum(actual_indices, len(table) - 1)
            step = VLAStepData(
                images={key: [video[key][idx]] for key in modalities["video"].modality_keys},
                states={key: np.stack([table[f"state.{key}"].iloc[frame]]).astype(np.float32) for key in modalities["state"].modality_keys},
                actions={key: np.stack(table[f"action.{key}"].iloc[clipped].to_list()).astype(np.float32) for key in modalities["action"].modality_keys},
                text=str(table[f"language.{lang_key}"].iloc[frame]),
                embodiment=EmbodimentTag(config.embodiment), is_demonstration=bool(demo[frame]),
            )
            processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            batch = processor.collator([processed])["inputs"]
            backbone_input, action_input = model.prepare_input(batch)
            backbone_output = model.backbone(backbone_input)
            normalized_moment = head.vlln(backbone_output["backbone_features"])[:, -n_q:]
            backbone_output = head.process_backbone_output(backbone_output, action_inputs_B=1)
            feat = backbone_output["backbone_features"][0].detach().cpu().to(torch.bfloat16).clone()
            am = backbone_output.get("backbone_attention_mask", torch.ones((1, len(feat)), device=device, dtype=torch.bool))
            im = backbone_output.get("image_mask", torch.zeros((1, len(feat)), device=device, dtype=torch.bool))
            features.append(feat)
            attention.append(am[0].bool().cpu().clone())
            image_masks.append(im[0].bool().cpu().clone())
            moment.append(normalized_moment[0].detach().cpu().to(torch.bfloat16).clone())
            short.append(feat[-n_q:].clone())
            # Take pre-bfloat16 processor tensors, preserving normalization precision.
            states.append(torch.as_tensor(processed["state"]).squeeze(0).float().cpu().clone())
            if idx == len(frames) - 1:
                continue  # Endpoint only: no unobserved result event is invented.
            target = torch.as_tensor(processed["action"]).float().cpu().clone()
            mask = torch.as_tensor(processed["action_mask"]).bool().cpu().clone()
            valid = torch.zeros(target.shape[0], dtype=torch.bool)
            valid[:len(action_offsets)] = torch.as_tensor((actual_indices < len(table)) & ~demo[clipped])
            decision = not bool(demo[frame])
            mask &= valid[:, None]
            if not decision:
                mask.zero_()
            target[~mask] = 0
            prefix = torch.zeros((stride, target.shape[-1]), dtype=torch.float32)
            prefix_mask = torch.zeros(stride, dtype=torch.bool)
            gap = int(frames[idx + 1] - frame)
            if decision:
                prefix[:gap] = target[:gap]
                prefix_mask[:gap] = mask[:gap].any(-1)
                prefix[~prefix_mask] = 0
            targets.append(target)
            target_masks.append(mask)
            actions.append(prefix)
            action_masks.append(prefix_mask)
            decisions.append(decision and bool(mask.any()))
    head.reset_memory()
    result = {
        "episode_id": episode_id, "task": record["task"], "frames": torch.from_numpy(frames),
        "features": features, "attention_masks": attention, "image_masks": image_masks,
        "moment": torch.stack(moment), "short": torch.stack(short), "state": torch.stack(states),
        "targets": torch.stack(targets), "target_mask": torch.stack(target_masks),
        "actions": torch.stack(actions), "action_mask": torch.stack(action_masks),
        "transition_valid": torch.ones(len(frames) - 1, dtype=torch.bool),
        "decision_mask": torch.tensor(decisions, dtype=torch.bool),
        "is_demo": torch.as_tensor(demo[frames]),
        "embodiment_id": int(processor.embodiment_id_mapping[config.embodiment]),
    }
    validate_episode(result)
    _require(bool(result["decision_mask"].any()), "No trainable decisions in this episode")
    return result


def build_cache(config: CacheConfig) -> Path:
    """Resume safely, rejecting a cache from different data/weights/settings."""
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.long_memory.hamlet import load_frozen_hamlet

    dataset = Path(config.dataset_path).resolve(strict=True)
    output = Path(config.output_dir).resolve()
    info, all_records = _episode_records(dataset)
    splits = split_episodes(all_records, config.val_fraction, config.seed, config.max_episodes)
    selected = set(splits["train"] + splits["val"])
    records = [row for row in all_records if row["episode_id"] in selected]
    identity = _cache_identity(config, info, records, splits)
    fingerprint = _json_digest(identity)
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes").mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        _require(manifest.get("fingerprint") == fingerprint, "Existing cache is stale or settings differ. Use a NEW output directory")
        if manifest.get("status") == "complete":
            for row in records:
                EpisodeCache(output).load(row["episode_id"])
            print(f"[cache] already complete and validated: {output}", flush=True)
            return output
    else:
        _require(not any((output / "episodes").iterdir()), "Unmanifested episode files exist; choose a new output directory")
        manifest = {
            "format_version": CACHE_VERSION, "status": "building", "fingerprint": fingerprint,
            "identity": identity, "dataset_path": str(dataset), "model_path": str(Path(config.model_path).resolve()),
            "splits": splits, "train": splits["train"], "val": splits["val"],
            "episodes": [{"episode_id": row["episode_id"], "task": row["task"],
                          "path": f"episodes/episode_{row['episode_id']:06d}.pt",
                          "split": "train" if row["episode_id"] in splits["train"] else "val"}
                         for row in records],
            "completed_episodes": [],
        }
        _atomic_json(manifest_path, manifest)
    print(f"[cache] train={len(splits['train'])} val={len(splits['val'])}; episodic split fixed BEFORE extraction", flush=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model, processor = load_frozen_hamlet(config.model_path, config.device)
    _require(getattr(model.config, "hamlet_mode", "off") == "finetune", "Need a trained HAMLET checkpoint")
    _require(getattr(model.config, "mem_cond_type", None) == "cross_attn", "Only cross-attention HAMLET is supported")
    _require(getattr(model.config, "memory_type", "moment_token") == "moment_token", "Need moment-token memory")
    _require(getattr(model.config, "memory_stride", None) is not None, "Checkpoint must record memory_stride")
    processor.eval()
    loader = LeRobotEpisodeLoader(dataset, processor.modality_configs[config.embodiment], video_backend=config.video_backend)
    for num, record in enumerate(records, 1):
        path = output / "episodes" / f"episode_{record['episode_id']:06d}.pt"
        if path.exists():
            episode = EpisodeCache(output, allow_incomplete=True).load(record["episode_id"])
            print(f"[cache] {num}/{len(records)} resume episode={record['episode_id']}", flush=True)
        else:
            episode = _extract_episode(model, processor, loader, record, config)
            episode["cache_fingerprint"] = fingerprint
            _atomic_torch_save(path, episode)
            print(f"[cache] {num}/{len(records)} episode={record['episode_id']} events={len(episode['actions'])} decisions={int(episode['decision_mask'].sum())} size={path.stat().st_size / 1e6:.1f} MB", flush=True)
        dimensions = {
            "feature_dim": episode["moment"].shape[-1], "n_moment_tokens": episode["moment"].shape[-2],
            "state_dim": episode["state"].shape[-1], "action_dim": episode["targets"].shape[-1],
            "action_steps": episode["actions"].shape[1], "action_horizon": episode["targets"].shape[1],
        }
        if "dimensions" in manifest:
            _require(manifest["dimensions"] == dimensions, "Episode dimensions disagree")
        manifest["dimensions"] = dimensions
        manifest.update(dimensions)
        manifest["completed_episodes"] = sorted(set(manifest["completed_episodes"]) | {record["episode_id"]})
        _atomic_json(manifest_path, manifest)
    manifest["status"] = "complete"
    _atomic_json(manifest_path, manifest)
    print(f"[cache] complete: {output} (frozen features only; not a learned model)", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local raw HAMLET checkpoint folder")
    parser.add_argument("--dataset-path", required=True, help="RoboMME LeRobot dataset or reindexed subset")
    parser.add_argument("--output-dir", required=True, help="New/resumable cache directory")
    parser.add_argument("--max-episodes", type=int, default=16, help="Small default; use 160 for the existing full subset")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-backend", default="opencv", choices=("decord", "torchcodec", "opencv"))
    parser.add_argument("--embodiment", default="new_embodiment")
    build_cache(CacheConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
