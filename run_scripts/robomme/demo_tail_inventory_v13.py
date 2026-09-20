#!/usr/bin/env python3
"""Plan/extract a separate full TRAIN/VAL demo-tail inventory, never modify cache.

This driver reuses the frozen, numerically verified V13 image-only extractor.
It changes coverage, not encoding, canonical query indices or HAMLET history.
Every selected original episode has a payload; episodes without demonstrations
have a well-typed empty payload. No task-based selection, GT action reads, TEST
partition, batching, resume, training or simulator execution is provided.

Run --preflight-only first. --extract needs explicit GPU availability and a NEW
directory. Interrupted inventories remain unusable by the strict frozen reader.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from run_scripts.robomme import demo_tail_sidecar_v13 as frozen
from run_scripts.robomme import verify_demo_tail_v13 as proof

DRIVER = "demo_tail_inventory_v13"
SCOPE = "inventory_train_val"
BYTES_PER_FRAME = 2 * 81 * 2048 * 2
GIB = 1024 ** 3


def source_identity():
    return {**proof.source_hashes(), str(Path(__file__).relative_to(ROOT)): frozen.file_sha256(__file__)}


def validate_numerical_proof(path, *, cache_fingerprint, base_model):
    """Require the actual 12-frame exact proof, not metadata or processor-only PASS."""
    path = Path(path).resolve(strict=True)
    plan = json.loads((path / "plan.json").read_text())
    result = json.loads((path / "result.json").read_text())
    args = plan["args"]
    frozen.require(args.get("preflight_only") is False and args.get("preparation_only") is False,
                   "Reference must be actual numerical proof, not preflight/preparation-only")
    frozen.require(plan["cache_fingerprint"] == cache_fingerprint
                   and Path(plan["base_model"]).resolve() == Path(base_model).resolve(),
                   "Numerical proof belongs to another original cache/base")
    frozen.require(all(result.get(key) is True for key in
        ("passed", "structural_passed", "all_feature_comparisons_exact"))
        and result.get("numerical_review_required") is False and "error" not in result,
        "Numerical proof did not pass exactly; no tolerance waiver")
    expected = proof.proof_frames(plan)
    frozen.require(plan["proof_frames"] == expected and len(result["rows"]) == 12,
                   "Reference does not contain all twelve predeclared numerical comparisons")
    required_checks = {"frozen_no_grad_before", "frozen_model_content_unchanged", "frozen_no_grad_after",
                       "source_unchanged", "protected_files_unchanged", "runtime_unchanged"}
    frozen.require(required_checks <= result["checks"].keys()
                   and all(v is True for v in result["checks"].values()), "Reference integrity did not pass")
    frozen.require(result["frozen_before_sha256"] == result["frozen_after_sha256"]
                   and len(result["frozen_before_sha256"]) == 64, "Reference frozen model changed")
    for actual, planned in zip(result["rows"], expected):
        frozen.require(all(actual.get(k) == v for k, v in planned.items()), "Reference frame identity mismatch")
        frozen.require(actual.get("structural_passed") is True and actual.get("all_feature_comparisons_exact") is True
                       and actual.get("checks") and all(v is True for v in actual["checks"].values()),
                       "Reference frame structural/numerical checks failed")
        names = ["native_repeat", "direct_vs_native"]
        if planned["kind"] == "canonical":
            names += ["native_vs_saved_cache", "direct_vs_saved_cache"]
        for name in names:
            comparison = actual["comparisons"][name]
            frozen.require(comparison.get("exact") is True and comparison.get("finite") is True
                           and comparison.get("same_shape") is True and comparison.get("same_dtype") is True
                           and comparison.get("max_abs") == 0., f"Nonexact reference feature comparison: {name}")
        for name in ("prepared_native_repeat", "device_inputs_native_repeat", "prepared_direct", "device_inputs_direct"):
            frozen.require(set(actual["comparisons"][name]) == frozen.INPUT_KEYS
                           and all(row.get("exact") is True for row in actual["comparisons"][name].values()),
                           "Nonexact native/sidecar input tree")
        frozen.require(actual["action_calls"] == {"action_head_forward": 0, "action_dit_forward": 0},
                       "Reference executed an Action Expert")
    current_sources = proof.source_hashes()
    frozen.require(current_sources == plan["source_sha256"],
                   "Numerical proof sources changed, including helper/verifier/legacy implementation")
    frozen.require(frozen.runtime_identity() == plan["runtime"], "Numerical proof runtime differs")
    frozen.require(all(frozen.file_sha256(name) == saved for name, saved in plan["files_sha256"].items()),
                   "Numerical proof original inputs changed")
    return {"path": str(path), "files_sha256": {str(path / name): frozen.file_sha256(path / name)
                                               for name in ("plan.json", "result.json")},
            "source_sha256": current_sources, "frozen_model_sha256": result["frozen_before_sha256"],
            "frames": expected, "passed": True}


def selected_episode_ids(manifest, splits):
    frozen.require(isinstance(splits, (list, tuple)) and bool(splits)
                   and len(set(splits)) == len(splits) and set(splits) <= {"train", "val"},
                   "Only distinct original TRAIN/VAL partitions are allowed; no TEST")
    normalized = [split for split in ("train", "val") if split in splits]
    train, val = set(manifest["splits"]["train"]), set(manifest["splits"]["val"])
    frozen.require(not train & val and train and val, "Original TRAIN/VAL partitions overlap or are empty")
    ids = sorted(eid for split in normalized for eid in manifest["splits"][split])
    frozen.require(all(type(eid) is int and eid >= 0 for eid in ids) and len(ids) == len(set(ids)),
                   "Duplicate or invalid selected canonical episode IDs")
    return normalized, ids


def inventory_summary(records):
    split_counts = {name: {"episodes": 0, "demo_episodes": 0, "empty_episodes": 0, "frames": 0}
                    for name in ("train", "val")}
    for record in records:
        count = split_counts[record["split"]]
        count["episodes"] += 1
        count["demo_episodes"] += int(record["n_demo"] > 0)
        count["empty_episodes"] += int(not record["frames"])
        count["frames"] += len(record["frames"])
    totals = {key: sum(s[key] for s in split_counts.values()) for key in next(iter(split_counts.values()))}
    payload = totals["frames"] * BYTES_PER_FRAME
    buffer = max(2 * GIB, math.ceil(payload * .2))
    return {**totals, "by_split": split_counts, "bytes_per_paired_frame": BYTES_PER_FRAME,
            "raw_payload_bytes": payload, "raw_payload_gib": payload / GIB,
            "safety_buffer_bytes": buffer, "required_free_bytes": payload + buffer,
            "storage_note": "Raw BF16 image payload estimate; safety buffer covers serialization, manifests and atomic temporary files",
            "runtime_estimate": None, "batching": "one paired-camera frame per backbone forward"}


def build_inventory_plan(cache_dir, base_model, numerical_proof, *, splits=("train", "val")):
    """Read-only full inventory; metadata and content hashes, no model/CUDA/output."""
    from gr00t.long_memory.cache import EpisodeCache, _episode_records, decision_frames
    from gr00t.utils.video_utils import resolve_backend
    cache = EpisodeCache(cache_dir)
    normalized, ids = selected_episode_ids(cache.manifest, splits)
    # Reuse the reviewed original-base/processor/geometry/signature preflight.
    # Its small proof selection is NOT the full inventory selection below.
    template = frozen.build_plan(cache.path, base_model, proof_train_count=8, proof_episode=0)
    reference = validate_numerical_proof(numerical_proof, cache_fingerprint=template["cache_fingerprint"],
                                         base_model=template["base_model"])
    dataset = Path(template["dataset_path"])
    info, rows = _episode_records(dataset)
    metadata = {row["episode_id"]: row for row in rows}
    frozen.require(len(metadata) == len(rows) and set(ids) <= metadata.keys(), "Missing/duplicate canonical dataset metadata")
    modality = json.loads((dataset / "meta/modality.json").read_text())
    language = template["episodes"][0]
    files = {Path(name) for name in template["files_sha256"]}
    files.update(Path(name) for name in reference["files_sha256"])
    records = []
    train_ids = set(cache.manifest["splits"]["train"])
    for eid in ids:
        row = metadata[eid]
        substitutions = {"episode_index": eid, "episode_chunk": eid // info["chunks_size"]}
        parquet = frozen._safe_input(dataset, info["data_path"].format(**substitutions))
        record = {"episode_id": eid, "split": "train" if eid in train_ids else "val",
                  "role": "inventory", "task": row["task"], "length": int(row["metadata"]["length"]),
                  "parquet_path": str(parquet), "tasks_path": str(dataset / "meta/tasks.jsonl"),
                  "episode_tasks": row["metadata"].get("tasks", []),
                  "language_key": language["language_key"], "language_column": language["language_column"]}
        demo, texts = frozen._read_demo_texts(record)  # No state/action/target columns.
        canonical = decision_frames(demo, 16)
        tail = frozen.plan_demo_tail(demo, canonical)
        canonical_path = frozen._safe_input(cache.path, cache._records[eid]["path"])
        cached = torch.load(canonical_path, map_location="cpu", weights_only=True, mmap=True)
        frozen.require(type(cached["episode_id"]) is int and cached["episode_id"] == eid
                       and cached["cache_fingerprint"] == template["cache_fingerprint"], "Wrong canonical cache episode binding")
        frozen.require(cached["frames"].dtype == torch.int64 and cached["is_demo"].dtype == torch.bool
                       and torch.equal(cached["frames"], torch.from_numpy(canonical))
                       and torch.equal(cached["is_demo"], torch.from_numpy(demo[canonical])),
                       "Raw demonstration/canonical query indices differ from immutable cache")
        del cached
        videos, backends = {}, {}
        for camera in frozen.CAMERA_ORDER:
            key = modality["video"][camera].get("original_key", f"observation.images.{camera}")
            path = frozen._safe_input(dataset, info["video_path"].format(video_key=key, **substitutions))
            videos[camera] = str(path)
            backends[camera] = resolve_backend(str(path), template["video_backend"])
            files.add(path)
        records.append({**record, **tail, "canonical_cache_path": str(canonical_path),
                        "video_paths": videos, "resolved_backends": backends,
                        "texts": [texts[frame] for frame in tail["frames"]], "task_text_sha256": frozen.digest(texts)})
        files.update((parquet, canonical_path))
    frozen._validate_original_signatures(cache.manifest, files)
    result = {key: value for key, value in template.items()
              if key not in {"episodes", "selection", "scope", "files_sha256", "source_sha256", "canonical_numerical_parity"}}
    result.update(scope=SCOPE, driver_variant=DRIVER, episodes=records,
                  selection={"splits": normalized, "episode_ids": ids, "rule": "all episodes in selected immutable cache partitions",
                             "quality_selection": False, "test_partition": False},
                  summary=inventory_summary(records), numerical_proof=reference,
                  canonical_numerical_parity="explicit_passed_twelve_frame_reference",
                  files_sha256={str(path): frozen.file_sha256(path) for path in sorted(files)},
                  source_sha256=source_identity(), runtime=frozen.runtime_identity())
    return result


def validate_plan(plan):
    from gr00t.long_memory.cache import EpisodeCache
    frozen.require(plan.get("scope") == SCOPE and plan.get("driver_variant") == DRIVER,
                   "Expected explicit full TRAIN/VAL inventory plan, not proof-subset plan")
    selected = plan["selection"]
    splits = selected["splits"]
    frozen.require(splits and splits == [x for x in ("train", "val") if x in splits]
                   and selected.get("quality_selection") is False and selected.get("test_partition") is False,
                   "Invalid inventory partition/selection policy")
    ids = [record["episode_id"] for record in plan["episodes"]]
    frozen.require(ids == sorted(set(ids)) == selected["episode_ids"]
                   and all(record["split"] in splits for record in plan["episodes"]),
                   "Missing/duplicate/out-of-partition inventory episodes")
    cache = EpisodeCache(plan["cache_dir"])
    actual_splits, actual_ids = selected_episode_ids(cache.manifest, splits)
    frozen.require(splits == actual_splits and ids == actual_ids
                   and plan["cache_fingerprint"] == cache.manifest["fingerprint"]
                   and Path(plan["base_model"]).resolve() == Path(cache.manifest["model_path"]).resolve()
                   and Path(plan["dataset_path"]).resolve() == Path(cache.manifest["dataset_path"]).resolve(),
                   "Inventory must contain every episode in the exact original selected partitions")
    frozen.require(plan["camera_order"] == list(frozen.CAMERA_ORDER) and plan["grid"] == list(frozen.GRID)
                   and plan["width"] == frozen.WIDTH and plan["feature_point"] == frozen.FEATURE_POINT,
                   "Inventory observation layout differs from verified extractor")
    train = set(cache.manifest["splits"]["train"])
    required_files = {str(cache.path / "manifest.json"), *plan["numerical_proof"]["files_sha256"].keys()}
    for record in plan["episodes"]:
        eid = record["episode_id"]
        expected_path = frozen._safe_input(cache.path, cache._records[eid]["path"])
        frozen.require(record["split"] == ("train" if eid in train else "val")
                       and Path(record["canonical_cache_path"]).resolve() == expected_path,
                       "Episode split/canonical payload binding changed")
        episode = torch.load(expected_path, map_location="cpu", weights_only=True, mmap=True)
        frames, demos = episode["frames"], episode["is_demo"]
        frozen.require(episode["episode_id"] == eid and episode["cache_fingerprint"] == plan["cache_fingerprint"]
                       and frames.dtype == torch.int64 and demos.dtype == torch.bool and frames.shape == demos.shape
                       and frames.tolist() == record["canonical_frames"] and bool((~demos).any()),
                       "Canonical cached episode identity changed")
        n_demo = int(frames[~demos][0])
        last_demo = int(frames[demos][-1]) if bool(demos.any()) else -1
        frozen.require(record["n_demo"] == n_demo and record["last_canonical_demo"] == last_demo
                       and record["frames"] == list(range(max(last_demo + 1, n_demo - 15, 0), n_demo)),
                       "Inventory tail differs from immutable canonical boundaries")
        required_files.update((str(expected_path), record["parquet_path"], *record["video_paths"].values()))
    frozen.require(required_files <= plan["files_sha256"].keys(), "Inventory omits required protected source hashes")
    frozen.require(plan["summary"] == inventory_summary(plan["episodes"]), "Inventory size estimate changed")
    frozen.require(plan["numerical_proof"].get("passed") is True, "Actual numerical proof is mandatory")


def input_checks(plan):
    return {"input_files_unchanged": all(frozen.file_sha256(path) == saved for path, saved in plan["files_sha256"].items()),
            "sources_unchanged": source_identity() == plan["source_sha256"],
            "runtime_unchanged": frozen.runtime_identity() == plan["runtime"]}


def check_disk_space(plan, output_dir):
    output = frozen.validate_output_path(plan, output_dir)
    reference = Path(plan["numerical_proof"]["path"]).resolve()
    frozen.require(not output.is_relative_to(reference) and not reference.is_relative_to(output),
                   "Output overlaps immutable numerical proof")
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    required = plan["summary"]["required_free_bytes"]
    frozen.require(free >= required, f"Insufficient free disk: need {required / GIB:.2f} GiB, have {free / GIB:.2f} GiB")
    return output, {"filesystem_probe": str(ancestor), "free_bytes": free, "required_bytes": required}


def extract_inventory(plan, output_dir, *, device="cuda:0"):
    """Fresh-only atomic episode writes. Failed/incomplete inventories cannot load."""
    from gr00t.long_memory.hamlet import load_frozen_hamlet
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    validate_plan(plan)
    output, disk = check_disk_space(plan, output_dir)
    frozen.require(all(input_checks(plan).values()), "Immutable inputs changed after inventory planning")
    # Recheck the actual passed numerical report, not just a caller-provided flag.
    frozen.require(validate_numerical_proof(plan["numerical_proof"]["path"], cache_fingerprint=plan["cache_fingerprint"],
                   base_model=plan["base_model"]) == plan["numerical_proof"], "Numerical proof reference changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    (output / "records").mkdir()
    manifest = {"format_version": frozen.VERSION, "kind": frozen.KIND, "driver_variant": DRIVER,
                "status": "building", "fingerprint": frozen.digest(plan), "plan": plan,
                "episodes": [], "completed_episodes": [], "inventory_checks": {}, "disk_preflight": disk}
    frozen._atomic_json(output / "manifest.json", manifest)
    model = None
    frozen_before = None
    start = time.monotonic()
    frames_done = 0
    versions = None
    error = None
    try:
        model, processor = load_frozen_hamlet(plan["base_model"], device)
        frozen.require(not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters()),
                       "Original model must be frozen eval with no gradients")
        versions, frozen_before = proof.version_state(model), proof.module_sha(model)
        manifest["frozen_before_sha256"] = frozen_before
        frozen.require(frozen_before == plan["numerical_proof"]["frozen_model_sha256"],
                       "Loaded model content differs from the exact numerical proof model")
        loader = LeRobotEpisodeLoader(Path(plan["dataset_path"]), processor.modality_configs[plan["embodiment"]],
                                     video_backend=plan["video_backend"])
        head = model.action_head
        original_cache = proof.clone_tree(head._memory_cache)
        with ExitStack() as stack:
            action_calls = stack.enter_context(proof.forbid_action_calls(head))
            stack.enter_context(patch.object(head, "process_backbone_output", side_effect=RuntimeError("Image-only inventory must not advance HAMLET memory")))
            for record in plan["episodes"]:
                observations = (frozen.load_visual_observations(processor, loader, record, record["frames"], embodiment=plan["embodiment"])
                                if record["frames"] else [])
                images, preparation = frozen.extract_observations(model.backbone, head.vlln, processor, observations,
                                                                   device=device, embodiment=plan["embodiment"])
                payload = {"episode_id": record["episode_id"], "cache_fingerprint": plan["cache_fingerprint"],
                           "sidecar_fingerprint": manifest["fingerprint"], "images": images,
                           "frames": torch.tensor(record["frames"], dtype=torch.int64),
                           "is_demo": torch.ones(len(images), dtype=torch.bool)}
                frozen.validate_sidecar_episode(payload, record, cache_fingerprint=plan["cache_fingerprint"],
                                                sidecar_fingerprint=manifest["fingerprint"])
                relative = f"episodes/episode_{record['episode_id']:06d}.pt"
                destination = output / relative
                frozen.require(not destination.exists(), "Episode payload already exists; resume/overwrite is forbidden")
                frozen._atomic_torch_save(destination, payload)
                loaded = torch.load(destination, map_location="cpu", weights_only=True)
                frozen.validate_sidecar_episode(loaded, record, cache_fingerprint=plan["cache_fingerprint"],
                                                sidecar_fingerprint=manifest["fingerprint"])
                frozen.require(torch.equal(payload["images"], loaded["images"]), "Episode roundtrip changed BF16 images")
                saved_record = {**record, "path": relative, "payload_sha256": frozen.file_sha256(destination), "preparation": preparation}
                frozen._atomic_json(output / "records" / f"episode_{record['episode_id']:06d}.json", saved_record)
                manifest["episodes"].append(saved_record)
                manifest["completed_episodes"].append(record["episode_id"])
                frames_done += len(images)
                frozen.require(proof.tree_equal(head._memory_cache, original_cache), "Image inventory changed original HAMLET short memory")
                frozen._atomic_json(output / "progress.json", {"status": "building", "episodes_done": len(manifest["episodes"]),
                    "episodes_planned": len(plan["episodes"]), "frames_done": frames_done,
                    "frames_planned": plan["summary"]["frames"], "last_episode_id": record["episode_id"],
                    "elapsed_seconds": time.monotonic() - start})
                print(f"[inventory] {len(manifest['episodes'])}/{len(plan['episodes'])} episodes; {frames_done}/{plan['summary']['frames']} frames", flush=True)
            manifest["inventory_checks"]["zero_action_calls"] = not any(action_calls.values())
            manifest["action_calls"] = action_calls
            manifest["inventory_checks"]["original_short_memory_unchanged"] = proof.tree_equal(head._memory_cache, original_cache)
    except BaseException as exc:
        error = exc
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        checks = {"checks": {}, "integrity_errors": []}
        for name in ("input_files_unchanged", "sources_unchanged", "runtime_unchanged"):
            def check_one(key=name):
                if key == "input_files_unchanged":
                    return all(frozen.file_sha256(path) == saved for path, saved in plan["files_sha256"].items())
                if key == "sources_unchanged":
                    return source_identity() == plan["source_sha256"]
                return frozen.runtime_identity() == plan["runtime"]
            proof.guarded_check(checks, name, check_one)
        proof.guarded_check(checks, "frozen_model_versions_unchanged", lambda: model is not None and versions == proof.version_state(model))
        proof.guarded_check(checks, "model_frozen_no_grad", lambda: model is not None and not model.training
                            and all(not p.requires_grad and p.grad is None for p in model.parameters()))
        checks["checks"]["all_planned_payloads_complete"] = manifest["completed_episodes"] == plan["selection"]["episode_ids"]
        manifest["checks"] = checks["checks"]
        manifest["integrity_errors"] = checks["integrity_errors"]
        if model is not None and frozen_before is not None:
            def model_content():
                manifest["frozen_after_sha256"] = proof.module_sha(model)
                return frozen_before == manifest["frozen_after_sha256"]
            extra = {"checks": manifest["inventory_checks"]}
            proof.guarded_check(extra, "frozen_model_content_unchanged", model_content)
            manifest["integrity_errors"] += extra.get("integrity_errors", [])
        manifest["elapsed_seconds"] = time.monotonic() - start
        complete = (error is None and set(manifest["checks"]) == frozen.CHECK_KEYS
                    and all(v is True for v in manifest["checks"].values())
                    and set(manifest["inventory_checks"]) == {"zero_action_calls", "original_short_memory_unchanged", "frozen_model_content_unchanged"}
                    and all(v is True for v in manifest["inventory_checks"].values()))
        manifest["status"] = "complete" if complete else "failed"
        frozen._atomic_json(output / "manifest.json", manifest)
        frozen._atomic_json(output / "progress.json", {"status": manifest["status"], "episodes_done": len(manifest["episodes"]),
            "episodes_planned": len(plan["episodes"]), "frames_done": frames_done, "frames_planned": plan["summary"]["frames"],
            "elapsed_seconds": manifest["elapsed_seconds"]})
    if error is not None:
        raise error
    frozen.require(manifest["status"] == "complete", "Inventory integrity failed; incomplete cache is not usable")
    try:
        reader = frozen.DemoTailSidecar(output, expected_cache_fingerprint=plan["cache_fingerprint"])
        for record in plan["episodes"]:
            reader.load(record["episode_id"])
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        frozen._atomic_json(output / "manifest.json", manifest)
        frozen._atomic_json(output / "progress.json", {"status": "failed", "episodes_done": len(manifest["episodes"]),
            "episodes_planned": len(plan["episodes"]), "frames_done": frames_done, "frames_planned": plan["summary"]["frames"],
            "elapsed_seconds": time.monotonic() - start, "failure": "strict final reader roundtrip"})
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--numerical-proof", required=True, help="Directory of passed actual 12-frame proof; content hashes stored in inventory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--extract", action="store_true")
    args = parser.parse_args(argv)
    started = time.monotonic()
    plan = build_inventory_plan(args.cache_dir, args.base_model, args.numerical_proof, splits=args.splits)
    validate_plan(plan)
    _, space = check_disk_space(plan, args.output_dir)
    summary = {"passed": True, "scope": SCOPE, "fingerprint": frozen.digest(plan), "splits": plan["selection"]["splits"],
               "summary": plan["summary"], "disk": space, "protected_input_files": len(plan["files_sha256"]),
               "source_files": len(plan["source_sha256"]), "numerical_proof": plan["numerical_proof"]["path"],
               "preflight_seconds": time.monotonic() - started, "cuda_initialized": torch.cuda.is_initialized(), "output_created": False}
    print(json.dumps(summary, indent=2), flush=True)
    if args.extract:
        print(f"[inventory] complete: {extract_inventory(plan, args.output_dir, device=args.device)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
