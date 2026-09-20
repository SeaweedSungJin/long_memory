#!/usr/bin/env python3
"""Proof-subset, image-only demo-tail extraction; never advances HAMLET memory.

This standalone candidate does not alter the original cache or any policy. Only
the first 8--16 TRAIN demos with a nonempty missing tail, plus separate episode
0, may be planned by the CLI. This is an input/numerical proof, not a training
set or a representative quality evaluation. No bulk extractor is provided.

The original eval image transform, language formalization and collator are
reused, but state/action processing is bypassed. Actual canonical-frame parity
against the genuine processor is REQUIRED before adopting this alternate input
path. Public preparation/extraction functions expose that comparison; they do
not silently choose an acceptance tolerance or restore caller RNG state.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gr00t.long_memory.cache import (  # noqa: E402
    EpisodeCache, _atomic_json, _atomic_torch_save, _episode_records,
    _json_digest, decision_frames,
)

VERSION = 1
KIND = "demo_tail_sidecar_v13"
CAMERA_ORDER = ("front_view", "wrist_view")
GRID = (9, 9)
WIDTH = 2048
INPUT_KEYS = frozenset(("input_ids", "attention_mask", "pixel_values"))
# The local Eagle processor also emits image_sizes (int64 [views, 2]). The
# frozen EagleBackbone.forward discards it when selecting its three inputs.
# Do not admit arbitrary metadata, state/action fields or other model families.
VISUAL_METADATA_KEYS = frozenset(("image_sizes",))
PAYLOAD_KEYS = frozenset(("episode_id", "cache_fingerprint", "sidecar_fingerprint",
                          "images", "frames", "is_demo"))
FEATURE_POINT = "post_backbone_post_vlln_image_only"
CHECK_KEYS = frozenset(("input_files_unchanged", "sources_unchanged", "runtime_unchanged",
                        "frozen_model_versions_unchanged", "model_frozen_no_grad",
                        "all_planned_payloads_complete"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def tensor_signature(tensor: torch.Tensor) -> dict:
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def input_tree_signature(value: Any) -> dict:
    """Bind tensor bytes AND exact container type/order; never merge images."""
    if isinstance(value, torch.Tensor):
        return tensor_signature(value)
    if type(value) in (list, tuple):
        return {"container": "list" if type(value) is list else "tuple",
                "items": [input_tree_signature(item) for item in value]}
    if type(value) is dict:
        require(all(isinstance(key, str) for key in value), "Input mapping keys must be strings")
        return {"container": "dict", "items": {key: input_tree_signature(item) for key, item in value.items()}}
    raise ValueError(f"Non-tensor encoder leaf/container: {type(value).__name__}")


def validate_input_signature(signature: dict) -> None:
    require(isinstance(signature, dict), "Invalid prepared tensor/container signature")
    if "container" not in signature:
        require(set(signature) == {"shape", "dtype", "sha256"}
                and isinstance(signature["shape"], list)
                and all(type(size) is int and size >= 0 for size in signature["shape"])
                and isinstance(signature["dtype"], str)
                and isinstance(signature["sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", signature["sha256"]) is not None,
                "Invalid prepared tensor signature")
        return
    require(set(signature) == {"container", "items"}, "Invalid prepared container signature")
    kind, items = signature["container"], signature["items"]
    if kind in ("list", "tuple"):
        require(isinstance(items, list), "Invalid sequence signature items")
        children = items
    else:
        require(kind == "dict" and isinstance(items, dict)
                and all(isinstance(key, str) for key in items), "Invalid mapping signature items")
        children = items.values()
    for child in children:
        validate_input_signature(child)


def move_visual_inputs(inputs: dict, device: str | torch.device,
                       dtype: torch.dtype = torch.bfloat16) -> dict:
    """Apply the original model.prepare_input leaf cast with the same tree helper.

    Eagle's pixel_values is a list of image tensors, not a pre-stacked tensor.
    Its order, boundaries and any tuple/list nesting reach Siglip unchanged.
    """
    import tree

    require(set(inputs) == INPUT_KEYS, "Expected only the three image/text input keys")
    target = torch.device(device)

    def cast(value):
        require(isinstance(value, torch.Tensor), "Encoder inputs must have tensor leaves")
        return value.to(target, dtype=dtype) if torch.is_floating_point(value) else value.to(target)

    return tree.map_structure(cast, inputs)


def source_identity() -> dict[str, str]:
    """Bind full local GR00T, Eagle/Siglip, tokenizer and transform sources."""
    paths = set((ROOT / "gr00t").rglob("*.py"))
    eagle = ROOT / "gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2"
    paths.update(p for p in eagle.rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts and p.suffix != ".pyc")
    paths.add(Path(__file__).resolve())
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(paths)}


def runtime_identity() -> dict:
    names = ("torch", "transformers", "numpy", "pandas", "pyarrow", "torchvision",
             "albumentations", "opencv-python", "av", "flash-attn", "dm-tree")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "packages": versions}


def plan_demo_tail(is_demo: np.ndarray, canonical_frames: Any) -> dict:
    """Uniform missing demo tail, strictly before the first execution frame."""
    demo = np.asarray(is_demo)
    require(demo.ndim == 1 and demo.dtype == np.bool_ and len(demo) >= 2,
            "is_demo must be an explicit bool vector with >=2 raw frames")
    execution = np.flatnonzero(~demo)
    require(bool(len(execution)), "Need an execution observation")
    n_demo = int(execution[0])
    require(not demo[n_demo:].any(), "Demo must be one contiguous prefix")
    frames = np.asarray(canonical_frames)
    require(frames.ndim == 1 and np.issubdtype(frames.dtype, np.integer),
            "Canonical raw frames must be integer vector")
    require(bool(len(frames)) and bool(np.all(np.diff(frames) > 0))
            and int(frames[0]) >= 0 and int(frames[-1]) < len(demo),
            "Canonical frames must be in-range and strictly increasing")
    past = frames[frames < n_demo]
    last = int(past[-1]) if len(past) else -1
    tail = list(range(max(last + 1, n_demo - 15, 0), n_demo))
    require(not set(tail) & set(frames.tolist()), "Tail overlaps canonical frames")
    return {"length": len(demo), "n_demo": n_demo, "last_canonical_demo": last,
            "canonical_frames": frames.tolist(), "frames": tail}


def _read_demo_texts(record: dict) -> tuple[np.ndarray, list[str]]:
    """Read ONLY the demo flag and deterministic language column, never GT."""
    import pandas as pd

    column = record["language_column"]
    columns = ["is_demo"] + ([column] if column else [])
    table = pd.read_parquet(record["parquet_path"], columns=columns)
    demo = table["is_demo"].to_numpy()
    require(demo.dtype == np.bool_ and len(demo) == record["length"],
            "Raw demo dtype/length mismatch")
    if column:
        rows = [json.loads(line) for line in Path(record["tasks_path"]).read_text().splitlines()
                if line.strip()]
        tasks = {row["task_index"]: row["task"] for row in rows}
        require(len(tasks) == len(rows), "Duplicate task indices")
        texts = [tasks[index] for index in table[column].tolist()]
    else:
        tasks = record["episode_tasks"]
        require(record["language_key"] == "task" and len(tasks) == 1,
                "No random or ambiguous episode-level language is allowed")
        texts = [tasks[0]] * len(demo)
    require(all(isinstance(text, str) for text in texts), "Task text must be strings")
    return demo, texts


def _safe_input(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    require(path.is_file() and path.is_relative_to(root), "Input path escapes its root")
    return path


def _validate_original_signatures(manifest: dict, selected: set[Path]) -> None:
    """Check the old cache's recorded signatures, then add stronger new hashes."""
    identity = manifest["identity"]
    signatures = list(identity.get("metadata", [])) + list(identity.get("code", []))
    signatures += [saved for saved in identity.get("payloads", [])
                   if Path(saved["path"]).resolve() in selected]
    for saved in signatures:
        path = Path(saved["path"]).resolve(strict=True)
        stat = path.stat()
        require(stat.st_size == saved["size"] and stat.st_mtime_ns == saved["mtime_ns"],
                f"Original cache source signature changed: {path}")
        if "sha256" in saved:
            require(file_sha256(path) == saved["sha256"], f"Original cache source hash changed: {path}")


def build_plan(cache_dir: str | Path, base_model: str | Path, *,
               proof_train_count: int = 8, proof_episode: int = 0) -> dict:
    """Read-only proof planning; no model, video decoding, CUDA or output write.

    Original cache endpoints are mmap-read for selected candidates only. Large
    input files are content-hashed, not loaded as tensor payloads. This function
    is intentionally not a bulk dataset planner.
    """
    from gr00t.long_memory.hamlet import validate_cache_checkpoint
    from gr00t.utils.video_utils import resolve_backend

    require(type(proof_train_count) is int and 8 <= proof_train_count <= 16,
            "Proof TRAIN count must be 8--16")
    require(proof_episode == 0, "Separate known proof is fixed episode 0")
    cache = EpisodeCache(cache_dir)
    manifest = cache.manifest
    require(_json_digest(manifest["identity"]) == manifest["fingerprint"],
            "Original cache fingerprint is inconsistent")
    base = Path(base_model).resolve(strict=True)
    require(base == Path(manifest["model_path"]).resolve(), "Must use original cache base model")
    validate_cache_checkpoint(manifest)
    dataset = Path(manifest["dataset_path"]).resolve(strict=True)
    info, rows = _episode_records(dataset)
    records = {row["episode_id"]: row for row in rows}
    require(len(records) == len(rows), "Duplicate dataset episode IDs")
    config = json.loads((base / "config.json").read_text())
    require(config.get("hamlet_mode") == "finetune"
            and config.get("memory_type", "moment_token") == "moment_token"
            and config.get("mem_cond_type", "cross_attn") == "cross_attn"
            and config.get("n_moment_tokens") == 4 and config.get("memory_stride") == 16,
            "Need unchanged K4/stride16 moment-token base")
    processor = json.loads((base / "processor_config.json").read_text())["processor_kwargs"]
    embodiment = manifest["identity"]["embodiment"]
    modalities = processor["modality_configs"][embodiment]
    # The saved training processor advertises K4 history. The original cache
    # producer nevertheless passes ONE supplied current frame per view in eval
    # (cache.py::_extract_episode); _get_vlm_inputs consumes those supplied
    # frames and never applies modality delta_indices. Preserve both identities.
    require(tuple(modalities["video"]["modality_keys"]) == CAMERA_ORDER
            and modalities["video"]["delta_indices"] == [-48, -32, -16, 0],
            "Camera order/original K4 cadence mismatch")
    require(len(modalities["language"]["modality_keys"]) == 1
            and modalities["language"]["delta_indices"] == [0], "Language must be current-only")
    language_key = modalities["language"]["modality_keys"][0]
    modality = json.loads((dataset / "meta/modality.json").read_text())
    if language_key in ("task", "sub_task"):
        language_column = None
    else:
        require(language_key.startswith("annotation."), "Unsupported language mapping")
        language_column = modality["annotation"][language_key.replace("annotation.", "")].get(
            "original_key", language_key)
    train = sorted(manifest["splits"]["train"])
    require(proof_episode in train, "Known episode 0 must be TRAIN")
    files = {cache.path / "manifest.json"}
    files.update(p.resolve() for p in base.rglob("*") if p.is_file() and ".cache" not in p.parts)
    files.update(p.resolve() for p in (dataset / "meta").glob("*.json*"))
    if (dataset / "subset_manifest.json").exists():
        files.add(dataset / "subset_manifest.json")
    selected = []
    attempted = []
    for episode_id in [proof_episode] + [eid for eid in train if eid != proof_episode]:
        row = records[episode_id]
        substitutions = {"episode_index": episode_id,
                         "episode_chunk": episode_id // info["chunks_size"]}
        parquet = _safe_input(dataset, info["data_path"].format(**substitutions))
        record = {"episode_id": episode_id, "split": "train", "task": row["task"],
                  "length": int(row["metadata"]["length"]), "parquet_path": str(parquet),
                  "tasks_path": str(dataset / "meta/tasks.jsonl"),
                  "episode_tasks": row["metadata"].get("tasks", []),
                  "language_key": language_key, "language_column": language_column}
        demo, texts = _read_demo_texts(record)
        canonical = decision_frames(demo, 16)
        tail = plan_demo_tail(demo, canonical)
        attempted.append(episode_id)
        # Include skipped flags in the immutable selection evidence.
        files.add(parquet)
        if not tail["frames"]:
            require(episode_id != proof_episode, "Known proof episode has no missing tail")
            continue
        cached = _safe_input(cache.path, cache._records[episode_id]["path"])
        payload = torch.load(cached, map_location="cpu", weights_only=True, mmap=True)
        require(payload["episode_id"] == episode_id
                and payload["cache_fingerprint"] == manifest["fingerprint"], "Stale canonical episode")
        require(torch.equal(payload["frames"], torch.from_numpy(canonical))
                and torch.equal(payload["is_demo"], torch.from_numpy(demo[canonical])),
                "Original cached canonical frames/demo do not match raw metadata")
        del payload
        files.add(cached)
        videos = {}
        backends = {}
        for camera in CAMERA_ORDER:
            key = modality["video"][camera].get("original_key", f"observation.images.{camera}")
            path = _safe_input(dataset, info["video_path"].format(video_key=key, **substitutions))
            videos[camera] = str(path)
            backends[camera] = resolve_backend(str(path), manifest["identity"]["video_backend"])
            files.add(path)
        selected.append({**record, **tail, "video_paths": videos, "resolved_backends": backends,
                         "role": "known_ep0_proof" if episode_id == proof_episode else "train_proof",
                         "texts": [texts[f] for f in tail["frames"]],
                         "task_text_sha256": digest(texts), "canonical_cache_path": str(cached)})
        if len(selected) == proof_train_count + 1:
            break
    require(len(selected) == proof_train_count + 1, "Not enough TRAIN demos for proof subset")
    _validate_original_signatures(manifest, files)
    return {"format_version": VERSION, "kind": KIND, "scope": "proof_subset_only",
            "cache_dir": str(cache.path), "cache_fingerprint": manifest["fingerprint"],
            "base_model": str(base), "dataset_path": str(dataset), "embodiment": embodiment,
            "video_backend": manifest["identity"]["video_backend"],
            "camera_order": list(CAMERA_ORDER), "grid": list(GRID), "width": WIDTH,
            "processor_video_delta_indices": modalities["video"]["delta_indices"],
            "extracted_frame_offsets_per_observation": [0],
            "feature_point": FEATURE_POINT, "storage_dtype": "bfloat16",
            "rule": "range(max(last_canonical_demo+1,n_demo-15,0),n_demo)",
            "selection": {"proof_episode": 0, "proof_train_count": proof_train_count,
                          "attempted_train_ids_in_order": attempted,
                          "train_proof_ids": [r["episode_id"] for r in selected[1:]],
                          "quality_selection": False},
            "episodes": selected, "files_sha256": {str(p): file_sha256(p) for p in sorted(files)},
            "source_sha256": source_identity(), "runtime": runtime_identity(),
            "canonical_numerical_parity": "not_evaluated_requires_separate_probe"}


def prepare_visual_inputs(processor, images: dict, text: str, *,
                          embodiment: str = "new_embodiment") -> dict[str, Any]:
    """Expose collated image/text inputs without flattening tensor containers."""
    require(not processor.training, "Processor must be in eval mode")
    order = tuple(processor.modality_configs[embodiment]["video"].modality_keys)
    require(order == CAMERA_ORDER and set(images) == set(CAMERA_ORDER), "Camera mapping mismatch")
    require(isinstance(text, str), "Instruction must be text")
    for camera in order:
        require(len(images[camera]) == 1, "Exactly one raw frame per camera")
        rgb = np.asarray(images[camera][0])
        require(rgb.ndim == 3 and rgb.shape[-1] == 3 and rgb.dtype == np.uint8,
                "Expected uint8 RGB frame")
    language = re.sub(r"[^\w\s]", "", text.lower()) if processor.formalize_language else text
    content = processor._get_vlm_inputs(list(order), images, None,
                                        processor.eval_image_transform, language)
    require(set(content) == {"vlm_content"}, "Unexpected processor stream")
    batch = processor.collator([content])["inputs"]
    missing = INPUT_KEYS - set(batch)
    unexpected = set(batch) - (INPUT_KEYS | VISUAL_METADATA_KEYS)
    require(not missing, f"Missing visual/text model inputs: {sorted(missing)}; actual keys={sorted(batch)}")
    require(not unexpected, f"Nonvisual data or unknown processor metadata: {sorted(unexpected)}; "
            f"actual keys={sorted(batch)}; allowed auxiliary keys={sorted(VISUAL_METADATA_KEYS)}")
    if "image_sizes" in batch:
        sizes = batch["image_sizes"]
        require(isinstance(sizes, torch.Tensor) and sizes.dtype == torch.int64
                and sizes.shape == (len(CAMERA_ORDER), 2) and bool((sizes > 0).all()),
                "image_sizes must be positive int64 [2,2] camera image metadata")
    # The frozen Eagle backbone itself consumes exactly these three keys.
    return {key: batch[key] for key in sorted(INPUT_KEYS)}


def image_tokens(features: torch.Tensor, image_mask: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    """Select two exact 81-token runs, never a hard-coded absolute slice."""
    require(features.ndim == 3 and features.shape[0] == 1 and features.shape[-1] == WIDTH,
            "Expected singleton [1,L,2048] normalized features")
    require(image_mask.dtype == attention_mask.dtype == torch.bool
            and image_mask.shape == attention_mask.shape == features.shape[:2], "Mask shape/dtype mismatch")
    require(not bool((image_mask & ~attention_mask).any()), "Image token outside attention mask")
    indices = torch.nonzero((image_mask & attention_mask)[0], as_tuple=False).flatten()
    require(len(indices) == 162, "Need exactly 162 image patches")
    groups = [indices[:81], indices[81:]]
    require(all(bool((part[1:] - part[:-1] == 1).all()) for part in groups)
            and int(indices[81] - indices[80]) > 1, "Expected separate camera runs of 81 tokens")
    require(not bool(image_mask[0, -4:].any()), "Image mask overlaps original moment tail")
    result = features[0, indices].reshape(2, 81, WIDTH)
    require(bool(torch.isfinite(result).all()), "Nonfinite image features")
    return result


@torch.inference_mode()
def extract_image_features(backbone, vlln, processor, images: dict, text: str, *,
                           device: str | torch.device,
                           embodiment: str = "new_embodiment") -> tuple[torch.Tensor, dict]:
    """Backbone -> VLLN only; no action head processing, queues or RNG restore.

    The caller supplies the original frozen backbone/VLLN, not an adapted
    archive. Returned features are CPU BF16 [2,81,2048]. Preparation hashes bind
    pre-cast and actually supplied encoder tensors for a later parity probe.
    """
    for module in (backbone, vlln):
        require(not module.training and all(not p.requires_grad for p in module.parameters()),
                "Extractor modules must be frozen eval")
    prepared = prepare_visual_inputs(processor, images, text, embodiment=embodiment)
    prepared_hashes = {key: input_tree_signature(value) for key, value in prepared.items()}
    target = torch.device(device)
    inputs = move_visual_inputs(prepared, target)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if target.type == "cuda" else contextlib.nullcontext()
    with amp:
        output = backbone(inputs)
        normalized = vlln(output["backbone_features"])
        result = image_tokens(normalized, output["image_mask"], output["backbone_attention_mask"])
    result = result.to(device="cpu", dtype=torch.bfloat16).clone()
    require(bool(torch.isfinite(result).all()), "Nonfinite BF16 image payload")
    return result, {"prepared": prepared_hashes,
                    "encoder": {key: input_tree_signature(value) for key, value in inputs.items()}}


def load_visual_observations(processor, loader, record: dict, frames: list[int], *,
                             embodiment: str = "new_embodiment") -> list[dict]:
    """Load explicit raw frames for tail extraction OR a canonical parity probe.

    Frames can include canonical execution observations for the separate probe;
    only the sidecar writer/reader authorize the strictly-demo tail payload.
    State/action/target columns are never loaded by this helper.
    """
    require(all(type(f) is int for f in frames) and frames == sorted(set(frames))
            and all(0 <= f < record["length"] for f in frames), "Invalid requested raw frames")
    require(tuple(processor.modality_configs[embodiment]["video"].modality_keys) == CAMERA_ORDER,
            "Camera order mismatch")
    demo, texts = _read_demo_texts(record)
    require(plan_demo_tail(demo, record["canonical_frames"])["frames"] == record["frames"],
            "Raw demo timing changed after planning")
    require(digest(texts) == record["task_text_sha256"], "Task text changed after planning")
    if not frames:
        return []
    video = loader._load_video_data(record["episode_id"], np.asarray(frames, dtype=np.int64))
    require(set(video) == set(CAMERA_ORDER) and all(len(video[c]) == len(frames) for c in CAMERA_ORDER),
            "Video frame/camera count mismatch")
    return [{"frame": frame, "images": {c: [video[c][i]] for c in CAMERA_ORDER}, "text": texts[frame]}
            for i, frame in enumerate(frames)]


def extract_observations(backbone, vlln, processor, observations: list[dict], *,
                         device: str | torch.device, embodiment: str = "new_embodiment") -> tuple[torch.Tensor, list]:
    """Frame-at-a-time extraction matches original cache batching; no short tokens."""
    images, preparation = [], []
    for obs in observations:
        require(set(obs) == {"frame", "images", "text"}, "Only RGB/text/frame observations allowed")
        feature, hashes = extract_image_features(backbone, vlln, processor, obs["images"], obs["text"],
                                                 device=device, embodiment=embodiment)
        images.append(feature)
        preparation.append({"frame": obs["frame"], **hashes})
    return (torch.stack(images) if images else torch.empty(0, 2, 81, WIDTH, dtype=torch.bfloat16)), preparation


def compare_features(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    """Report differences only. A caller must explicitly decide acceptance."""
    require(actual.shape == reference.shape, "Comparison shape mismatch")
    require(bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()), "Nonfinite comparison")
    left, right = actual.detach().float().cpu(), reference.detach().float().cpu()
    delta = left - right
    denominator = float(right.norm())
    return {"exact_equal": actual.dtype == reference.dtype and torch.equal(left, right),
            "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
            "relative_l2": float(delta.norm()) / denominator if denominator else None,
            "reference_zero_norm": denominator == 0, "acceptance": "not_decided"}


def validate_sidecar_episode(payload: dict, record: dict, *, cache_fingerprint: str,
                             sidecar_fingerprint: str) -> None:
    require(set(payload) == PAYLOAD_KEYS, "Sidecar must contain only image/frame/demo payload fields")
    require(type(payload["episode_id"]) is int and payload["episode_id"] == record["episode_id"], "Episode ID mismatch")
    require(payload["cache_fingerprint"] == cache_fingerprint
            and payload["sidecar_fingerprint"] == sidecar_fingerprint, "Sidecar/cache fingerprint mismatch")
    frames, demo, images = payload["frames"], payload["is_demo"], payload["images"]
    require(type(record["n_demo"]) is int and record["n_demo"] >= 0
            and type(record["last_canonical_demo"]) is int
            and -1 <= record["last_canonical_demo"] < record["n_demo"], "Invalid demo boundary")
    canonical = record["canonical_frames"]
    require(all(type(f) is int and f >= 0 for f in canonical)
            and canonical == sorted(set(canonical)), "Invalid canonical frame identity")
    canonical_demo = [f for f in canonical if f < record["n_demo"]]
    require(record["last_canonical_demo"] == (canonical_demo[-1] if canonical_demo else -1),
            "Last canonical demo identity mismatch")
    require(isinstance(frames, torch.Tensor) and frames.dtype == torch.int64
            and frames.ndim == 1 and frames.tolist() == record["frames"], "Sidecar raw frame list mismatch")
    expected = list(range(max(record["last_canonical_demo"] + 1, record["n_demo"] - 15, 0), record["n_demo"]))
    require(record["frames"] == expected and not set(expected) & set(record["canonical_frames"]),
            "Sidecar is not the exact missing demo tail")
    require(isinstance(demo, torch.Tensor) and demo.dtype == torch.bool
            and demo.shape == frames.shape and bool(demo.all()), "Sidecar observations must all be demo")
    require(isinstance(images, torch.Tensor) and images.dtype == torch.bfloat16
            and images.shape == (len(frames), 2, 81, WIDTH), "Sidecar image shape/dtype mismatch")
    require(bool(torch.isfinite(images).all()), "Nonfinite sidecar images")


class DemoTailSidecar:
    """Strict completed sidecar reader; hashes each payload before safe loading."""
    def __init__(self, path: str | Path, *, expected_cache_fingerprint: str | None = None):
        self.path = Path(path).resolve(strict=True)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        m = self.manifest
        require(m.get("format_version") == VERSION and m.get("kind") == KIND
                and m.get("status") == "complete", "Unsupported or incomplete sidecar")
        require(m["fingerprint"] == digest(m["plan"]), "Sidecar plan fingerprint mismatch")
        p = m["plan"]
        require(p["camera_order"] == list(CAMERA_ORDER) and p["grid"] == list(GRID)
                and p["width"] == WIDTH and p["feature_point"] == FEATURE_POINT, "Sidecar layout mismatch")
        if expected_cache_fingerprint is not None:
            require(p["cache_fingerprint"] == expected_cache_fingerprint, "Wrong original cache binding")
        require(set(m.get("checks", {})) == CHECK_KEYS
                and all(value is True for value in m["checks"].values()), "Sidecar integrity checks did not pass")
        self._records = {r["episode_id"]: r for r in m["episodes"]}
        planned = {r["episode_id"]: r for r in p["episodes"]}
        require(len(self._records) == len(m["episodes"]) == len(planned)
                and self._records.keys() == planned.keys()
                and sorted(m["completed_episodes"]) == sorted(planned), "Incomplete or duplicate episodes")
        for eid, record in self._records.items():
            require(all(record.get(k) == v for k, v in planned[eid].items()), "Episode metadata changed")
            target = _safe_input(self.path, record["path"])
            require(file_sha256(target) == record["payload_sha256"], "Sidecar payload hash mismatch")
            require([x["frame"] for x in record["preparation"]] == record["frames"], "Missing preparation proof")
            for item in record["preparation"]:
                require(set(item) == {"frame", "prepared", "encoder"}, "Invalid preparation proof schema")
                for stage in ("prepared", "encoder"):
                    require(set(item[stage]) == INPUT_KEYS, "Preparation proof must be image/text only")
                    for signature in item[stage].values():
                        validate_input_signature(signature)

    def load(self, episode_id: int) -> dict:
        record = self._records[episode_id]
        path = _safe_input(self.path, record["path"])
        require(file_sha256(path) == record["payload_sha256"], "Sidecar payload hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_sidecar_episode(payload, record, cache_fingerprint=self.manifest["plan"]["cache_fingerprint"],
                                 sidecar_fingerprint=self.manifest["fingerprint"])
        return payload


def _check_inputs(plan: dict) -> dict:
    return {"input_files_unchanged": all(file_sha256(path) == sha for path, sha in plan["files_sha256"].items()),
            "sources_unchanged": source_identity() == plan["source_sha256"],
            "runtime_unchanged": runtime_identity() == plan["runtime"]}


def validate_output_path(plan: dict, output_dir: str | Path) -> Path:
    raw = Path(output_dir)
    require(not raw.is_symlink(), "Output cannot be a symlink")
    output = raw.resolve()
    require(not output.exists(), "Output must be a NEW directory")
    for key in ("cache_dir", "base_model", "dataset_path"):
        protected = Path(plan[key]).resolve()
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output),
                "Output overlaps immutable inputs")
    return output


def build_sidecar(plan: dict, output_dir: str | Path, *, device: str = "cuda:0") -> Path:
    """Explicit proof-only execution. Fresh output only; never resumes/overwrites."""
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.long_memory.hamlet import load_frozen_hamlet

    require(plan["scope"] == "proof_subset_only" and 9 <= len(plan["episodes"]) <= 17,
            "Only an approved-size proof subset is supported")
    output = validate_output_path(plan, output_dir)
    require(all(_check_inputs(plan).values()), "Inputs changed after planning")
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    manifest = {"format_version": VERSION, "kind": KIND, "status": "building",
                "fingerprint": digest(plan), "plan": plan, "episodes": [], "completed_episodes": []}
    _atomic_json(output / "manifest.json", manifest)
    try:
        model, processor = load_frozen_hamlet(plan["base_model"], device)
        require(not model.training and all(not p.requires_grad for p in model.parameters()), "Original model not frozen")
        versions = {name: value._version for name, value in list(model.named_parameters()) + list(model.named_buffers())}
        loader = LeRobotEpisodeLoader(Path(plan["dataset_path"]), processor.modality_configs[plan["embodiment"]],
                                      video_backend=plan["video_backend"])
        for record in plan["episodes"]:
            observations = load_visual_observations(processor, loader, record, record["frames"], embodiment=plan["embodiment"])
            images, preparation = extract_observations(model.backbone, model.action_head.vlln, processor,
                                                       observations, device=device, embodiment=plan["embodiment"])
            payload = {"episode_id": record["episode_id"], "cache_fingerprint": plan["cache_fingerprint"],
                       "sidecar_fingerprint": manifest["fingerprint"], "images": images,
                       "frames": torch.tensor(record["frames"], dtype=torch.int64),
                       "is_demo": torch.ones(len(images), dtype=torch.bool)}
            validate_sidecar_episode(payload, record, cache_fingerprint=plan["cache_fingerprint"],
                                     sidecar_fingerprint=manifest["fingerprint"])
            relative = f"episodes/episode_{record['episode_id']:06d}.pt"
            _atomic_torch_save(output / relative, payload)
            manifest["episodes"].append({**record, "path": relative,
                                         "payload_sha256": file_sha256(output / relative), "preparation": preparation})
            manifest["completed_episodes"].append(record["episode_id"])
            _atomic_json(output / "manifest.json", manifest)
        manifest["checks"] = {**_check_inputs(plan),
                              "frozen_model_versions_unchanged": versions == {
                                  name: value._version for name, value in list(model.named_parameters()) + list(model.named_buffers())},
                              "model_frozen_no_grad": not model.training and all(
                                  not p.requires_grad and p.grad is None for p in model.parameters()),
                              "all_planned_payloads_complete": len(manifest["completed_episodes"]) == len(plan["episodes"])}
        require(all(manifest["checks"].values()), "Sidecar integrity guard failed")
        manifest["status"] = "complete"
        _atomic_json(output / "manifest.json", manifest)
        reader = DemoTailSidecar(output, expected_cache_fingerprint=plan["cache_fingerprint"])
        for record in plan["episodes"]:
            reader.load(record["episode_id"])
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        try:
            manifest["checks_after_failure"] = _check_inputs(plan)
        except Exception as check_error:
            manifest["checks_after_failure"] = {"error": str(check_error)}
        _atomic_json(output / "manifest.json", manifest)
        raise
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--proof-train-count", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--extract", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(args.cache_dir, args.base_model, proof_train_count=args.proof_train_count)
    validate_output_path(plan, args.output_dir)
    if args.preflight_only:
        print(json.dumps({"passed": True, "plan": plan, "sidecar_fingerprint": digest(plan),
                          "cuda_initialized": torch.cuda.is_initialized(), "output_created": False}, indent=2))
    else:
        print(build_sidecar(plan, args.output_dir, device=args.device))


if __name__ == "__main__":
    main()
