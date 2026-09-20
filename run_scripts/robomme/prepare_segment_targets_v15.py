#!/usr/bin/env python3
"""Prepare PatternLock maximal-label-run weak retrieval targets, not cue truth.

Only immutable cache TRAIN/cache-VAL partitions are supported. Labels are target
metadata, never model inputs. Runs describe maximal equal planner-label spans;
consecutive identical atomic events cannot be recovered from these annotations.
No simulator, image decoder, model, optimizer, or CUDA execution is used.

The default CLI is read-only. Publication requires --publish and a new directory.
Cache tensors are mmap-loaded solely for frames/is_demo/decision_mask; feature,
state, action and target tensors are never indexed. Payload files are hashed as
opaque bytes to bind provenance, including the existing visual tail sidecar.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
KIND = "patternlock_maximal_run_weak_targets_v15"
GOALS = frozenset((
    "watch the video carefully, then use the stick attached to the robot to retrace the same pattern",
    "watch the video carefully, then use the stick attached to the robot to retrace the same pattern shown in the video",
))
DIRECTIONS = frozenset("move " + value for value in (
    "forward", "backward", "left", "right", "forward-left", "forward-right",
    "backward-left", "backward-right"))
CONFIG = {"margin": 2, "memory_window": 4, "memory_stride": 16,
          "minimum_runs": 2, "require_unique_run_labels": True,
          "interval_convention": "half_open",
          "margin_boundaries": "planner_maximal_run",
          "online_agreement": "at_each_query_and_each_positive_frame",
          "candidate_rule": "all_canonical_and_v13_tail_frames_strictly_before_query",
          "old_rule": "positive_frame < canonical_frames[max(0,decision-memory_window+1)]",
          "target_kind": "maximal_direction_label_run_not_atomic_event_or_verified_cue"}
RAW_COLUMNS = ("episode_index", "frame_index", "is_demo", "task_index",
               "simple_subgoal", "simple_subgoal_online")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    # Same canonical JSON serialization as the frozen V13 sidecar fingerprint.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text())


def _inside(root, relative):
    root = Path(root).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    require(path.is_file() and path.is_relative_to(root), "Input path escapes declared root")
    return path


def _frames(values, length, name):
    require(isinstance(values, (list, tuple)) and all(type(x) is int for x in values),
            f"{name} must contain integer raw frame indices")
    require(list(values) == sorted(set(values)) and all(0 <= x < length for x in values),
            f"{name} must be sorted, unique, and within the episode")
    return list(values)


def maximal_runs(labels, start, end):
    """Return half-open equal-string spans, never claim atomic event boundaries."""
    require(type(start) is int and type(end) is int and 0 <= start <= end <= len(labels),
            "Invalid run interval")
    require(all(isinstance(x, str) for x in labels[start:end]), "Labels must be strings")
    runs = []
    for frame in range(start, end):
        if not runs or labels[frame] != runs[-1]["label"]:
            runs.append({"start": frame, "end": frame + 1, "label": labels[frame]})
        else:
            runs[-1]["end"] = frame + 1
    return runs


def episode_targets(episode_id, split, raw_demo, planner, online, canonical_frames,
                    decision_mask, tail_frames, *, margin=2, memory_window=4):
    """Pure weak-target construction; no state/action/image arguments exist.

    The cached decision index is the current canonical-observation index. Tail
    observations add memory candidates without renumbering canonical decisions.
    Whole-episode annotations determine eligibility, not any model input.
    """
    require(type(episode_id) is int and episode_id >= 0 and split in ("train", "val"),
            "Require original TRAIN/cache-VAL episode identity")
    require(type(margin) is int and margin >= 0 and type(memory_window) is int
            and memory_window > 0, "Invalid margin/window")
    length = len(raw_demo)
    require(length >= 2 and len(planner) == len(online) == length
            and all(type(x) is bool for x in raw_demo), "Invalid label/demo lengths or types")
    require(all(isinstance(x, str) for x in planner)
            and all(isinstance(x, str) for x in online), "Labels must be strings")
    n_demo = sum(raw_demo)
    require(raw_demo == [True] * n_demo + [False] * (length - n_demo)
            and n_demo < length, "Demo must be a contiguous prefix followed by execution")
    canonical = _frames(canonical_frames, length, "canonical_frames")
    tail = _frames(tail_frames, length, "tail_frames")
    require(canonical and canonical[0] == 0 and canonical[-1] == length - 1,
            "Canonical cache must retain first and final endpoints")
    require(len(decision_mask) == len(canonical) - 1
            and all(type(x) is bool for x in decision_mask), "Invalid cached decision_mask")
    require(not any(active and raw_demo[canonical[i]] for i, active in enumerate(decision_mask)),
            "Passive demo cannot be an active cached decision")
    require(not set(canonical) & set(tail) and all(frame < n_demo for frame in tail),
            "Tail must contain only distinct omitted demo frames")
    result = {"examples": [], "episode_exclusion": None, "query_exclusions": {},
              "active_decisions": sum(decision_mask), "n_demo": n_demo}
    demo_runs = maximal_runs(planner, 0, n_demo)
    execution_runs = maximal_runs(planner, n_demo, length)
    result.update(demo_run_count=len(demo_runs), execution_run_count=len(execution_runs))
    demo_labels = [run["label"] for run in demo_runs]
    execution_labels = [run["label"] for run in execution_runs]
    reason = None
    if not n_demo:
        reason = "no_demo"
    elif any(label not in DIRECTIONS for label in demo_labels + execution_labels):
        reason = "non_direction_planner_label"
    elif len(demo_labels) < 2:
        reason = "fewer_than_two_maximal_runs"
    elif demo_labels != execution_labels:
        reason = "demo_execution_run_sequence_mismatch"
    elif len(set(demo_labels)) != len(demo_labels):
        reason = "repeated_run_label_ambiguous_occurrence"
    if reason:
        result["episode_exclusion"] = reason
        return result
    exclusions = Counter()
    observed = sorted(canonical + tail)
    for decision, active in enumerate(decision_mask):
        if not active:
            continue
        frame = canonical[decision]
        ordinal = next(i for i, run in enumerate(execution_runs)
                       if run["start"] <= frame < run["end"])
        run, past = execution_runs[ordinal], demo_runs[ordinal]
        if not run["start"] + margin <= frame < run["end"] - margin:
            exclusions["query_within_planner_boundary_margin"] += 1
            continue
        if online[frame] != planner[frame]:
            exclusions["query_planner_online_disagree"] += 1
            continue
        interval = [past["start"] + margin, past["end"] - margin]
        candidates = [f for f in observed if f < frame]
        positives = [f for f in candidates if interval[0] <= f < interval[1]
                     and raw_demo[f] and planner[f] == online[f] == run["label"]]
        if not positives:
            exclusions["no_observed_positive_after_margin_and_agreement"] += 1
            continue
        oldest = canonical[max(0, decision - memory_window + 1)]
        result["examples"].append({"episode_id": episode_id, "split": split,
            "decision": decision, "query_frame": frame, "n_demo": n_demo, "label": run["label"],
            "raw_run_ordinal": ordinal, "positive_interval": interval,
            "positive_frames": positives, "candidate_frames": candidates,
            "oldest_short_frame": oldest, "old_positive_frames": [f for f in positives if f < oldest]})
    result["query_exclusions"] = dict(exclusions)
    return result


def _cache_layout(path, episode_id, fingerprint):
    import torch
    require(not torch.cuda.is_initialized(), "Preparer must not initialize CUDA")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    require(payload["episode_id"] == episode_id and payload["cache_fingerprint"] == fingerprint,
            "Cache payload identity differs")
    expected = {"frames": torch.int64, "is_demo": torch.bool, "decision_mask": torch.bool}
    for key, dtype in expected.items():
        require(isinstance(payload[key], torch.Tensor) and payload[key].ndim == 1
                and payload[key].dtype == dtype, f"Wrong cache layout tensor: {key}")
    return {key: payload[key].tolist() for key in expected}


def verify_identity(manifest):
    """Fail closed if any bound metadata, selected payload, or source changes."""
    for group in ("files_sha256", "source_sha256"):
        for name, expected in manifest[group].items():
            require(file_sha256(name) == expected, f"Changed {group}: {name}")


def validate_manifest(manifest):
    """Validate supervision-only schema/fingerprint without opening model data."""
    require(manifest.get("kind") == KIND and manifest.get("format_version") == 1
            and manifest.get("status") == "complete" and manifest.get("config") == CONFIG,
            "Wrong weak-target format, status, or fixed configuration")
    require(manifest.get("fingerprint") == digest({k: v for k, v in manifest.items() if k != "fingerprint"}),
            "Invalid target manifest fingerprint")
    vocab = manifest["vocabulary"]
    require(vocab == sorted({row["label"] for row in manifest["examples"] if row["split"] == "train"}),
            "Vocabulary must come exclusively from retained TRAIN targets")
    splits = manifest["identity"]["original_splits"]
    require(not set(splits["train"]) & set(splits["val"]), "Target splits overlap")
    seen = set()
    for row in manifest["examples"]:
        require(row["split"] in ("train", "val") and row["episode_id"] in splits[row["split"]],
                "Example is outside its immutable split")
        key = (row["episode_id"], row["decision"])
        require(key not in seen, "Duplicate target query")
        seen.add(key)
        require(type(row["decision"]) is int and row["decision"] >= 0
                and type(row["query_frame"]) is int and type(row["n_demo"]) is int
                and 0 < row["n_demo"] <= row["query_frame"],
                "Invalid query identity/demo boundary")
        candidates = _frames(row["candidate_frames"], row["query_frame"], "candidate_frames")
        positives = _frames(row["positive_frames"], row["n_demo"], "positive_frames")
        interval = row["positive_interval"]
        require(positives and set(positives) <= set(candidates)
                and len(interval) == 2 and all(type(x) is int for x in interval)
                and 0 <= interval[0] < interval[1] <= row["n_demo"]
                and all(interval[0] <= f < interval[1] for f in positives), "Invalid positive interval/membership")
        require(row["label"] in DIRECTIONS and type(row["raw_run_ordinal"]) is int
                and row["raw_run_ordinal"] >= 0, "Invalid supervision label/run ordinal")
        require(row["label_id"] == (vocab.index(row["label"]) if row["label"] in vocab else -1),
                "Label ID differs from TRAIN-only vocabulary")
        require(type(row["oldest_short_frame"]) is int and 0 <= row["oldest_short_frame"] <= row["query_frame"]
                and row["old_positive_frames"] == [f for f in positives if f < row["oldest_short_frame"]],
                "Old positives differ from strict oldest-short boundary")
    return manifest


def load_manifest(path, *, verify_files=True):
    """Load a published manifest; full selected-input/source hashes checked by default."""
    path = Path(path)
    manifest = validate_manifest(_read_json(path / "manifest.json" if path.is_dir() else path))
    if verify_files:
        verify_identity(manifest)
    return manifest


def build_manifest(cache_dir, sidecar_dir):
    """Read-only preparation of all eligible PatternLock cache TRAIN/VAL rows."""
    import pyarrow.parquet as pq
    cache_dir, sidecar_dir = Path(cache_dir).resolve(strict=True), Path(sidecar_dir).resolve(strict=True)
    cache_path, sidecar_path = cache_dir / "manifest.json", sidecar_dir / "manifest.json"
    cache, sidecar = _read_json(cache_path), _read_json(sidecar_path)
    require(cache.get("status") == "complete" and sidecar.get("status") == "complete",
            "Cache and sidecar must both be complete")
    plan = sidecar["plan"]
    require(sidecar.get("kind") == "demo_tail_sidecar_v13"
            and sidecar.get("driver_variant") == "demo_tail_inventory_v13"
            and plan.get("scope") == "inventory_train_val"
            and sidecar["fingerprint"] == digest(plan), "Wrong or stale full V13 sidecar")
    require(plan["cache_fingerprint"] == cache["fingerprint"]
            and Path(plan["cache_dir"]).resolve() == cache_dir, "Sidecar/cache identity differs")
    required_checks = {"input_files_unchanged", "sources_unchanged", "runtime_unchanged",
        "frozen_model_versions_unchanged", "model_frozen_no_grad", "all_planned_payloads_complete"}
    required_inventory = {"zero_action_calls", "original_short_memory_unchanged", "frozen_model_content_unchanged"}
    require(not sidecar.get("integrity_errors") and required_checks <= sidecar.get("checks", {}).keys()
            and all(x is True for x in sidecar["checks"].values())
            and required_inventory <= sidecar.get("inventory_checks", {}).keys()
            and all(x is True for x in sidecar["inventory_checks"].values()), "Incomplete sidecar guards")
    dataset, base = Path(cache["dataset_path"]).resolve(strict=True), Path(cache["model_path"]).resolve(strict=True)
    require(Path(plan["dataset_path"]).resolve() == dataset
            and Path(plan["base_model"]).resolve() == base, "Sidecar dataset/base identity differs")
    base_config = _read_json(base / "config.json")
    require(base_config.get("memory_window") == CONFIG["memory_window"]
            and base_config.get("memory_stride") == CONFIG["memory_stride"], "Original HAMLET window/stride changed")
    info = _read_json(dataset / "meta/info.json")
    tasks = {r["task_index"]: r["task"] for r in map(json.loads, (dataset / "meta/tasks.jsonl").read_text().splitlines())}
    records = {r["episode_id"]: r for r in cache["episodes"]}
    tails = {r["episode_id"]: r for r in plan["episodes"]}
    payloads = {r["episode_id"]: r for r in sidecar["episodes"]}
    require(len(records) == len(cache["episodes"]) and len(tails) == len(plan["episodes"])
            and len(payloads) == len(sidecar["episodes"]) and set(records) == set(tails) == set(payloads)
            and sidecar.get("completed_episodes") == sorted(records), "Missing/duplicate sidecar episodes")
    train, val = set(cache["splits"]["train"]), set(cache["splits"]["val"])
    require(not train & val and train | val == set(records), "Original TRAIN/VAL split differs")
    selected = sorted(eid for eid, record in records.items() if record["task"] in GOALS)
    require(bool(selected), "No exact PatternLock global instructions in cache")
    files = {str(p): file_sha256(p) for p in (cache_path, sidecar_path, base / "config.json",
        dataset / "meta/info.json", dataset / "meta/tasks.jsonl", dataset / "meta/episodes.jsonl",
        dataset / "meta/modality.json")}
    sources = {str(Path(__file__).resolve()): file_sha256(__file__)}
    for name, expected in plan["source_sha256"].items():
        path = (ROOT / name).resolve(strict=True)
        require(file_sha256(path) == expected, f"Sidecar source changed: {name}")
        sources[str(path)] = expected
    for signature in cache["identity"]["metadata"] + cache["identity"]["code"]:
        if "sha256" in signature:
            require(file_sha256(signature["path"]) == signature["sha256"], "Original cache source/metadata changed")
            (sources if signature in cache["identity"]["code"] else files)[signature["path"]] = signature["sha256"]
    for signature in cache["identity"]["checkpoint"]:
        path = Path(signature["path"])
        if "sha256" in signature:
            require(file_sha256(path) == signature["sha256"], "Original base metadata changed")
            files[str(path)] = signature["sha256"]
        else:
            current = path.stat()
            require(current.st_size == signature["size"] and current.st_mtime_ns == signature["mtime_ns"],
                    "Original base weight file signature changed")
    examples, audits = [], []
    coverage = {split: Counter() for split in ("train", "val")}
    exclusions = {split: {"episodes": Counter(), "queries": Counter()} for split in ("train", "val")}
    for eid in selected:
        record, tail, payload = records[eid], tails[eid], payloads[eid]
        split = "train" if eid in train else "val"
        require(record["split"] == tail["split"] == payload["split"] == split
                and tail["task"] == record["task"], "Episode split/task differs")
        parquet = _inside(dataset, info["data_path"].format(episode_chunk=eid // info["chunks_size"], episode_index=eid))
        cached = _inside(cache_dir, record["path"])
        sidecar_payload = _inside(sidecar_dir, payload["path"])
        for path, expected in ((parquet, plan["files_sha256"][str(parquet)]),
                               (cached, plan["files_sha256"][str(cached)]),
                               (sidecar_payload, payload["payload_sha256"])):
            require(file_sha256(path) == expected, f"Selected input content changed: {path}")
            files[str(path)] = expected
        table = pq.read_table(parquet, columns=list(RAW_COLUMNS)).to_pydict()
        length = len(table["is_demo"])
        require(table["episode_index"] == [eid] * length and table["frame_index"] == list(range(length)),
                "Raw episode/frame identity differs")
        require(all(tasks[index] == record["task"] for index in table["task_index"]),
                "Raw global instruction differs from task-family selection")
        layout = _cache_layout(cached, eid, cache["fingerprint"])
        n_demo = sum(table["is_demo"])
        canonical = sorted({0, length - 1, *range(n_demo, -1, CONFIG["memory_stride"] * -1),
                            *range(n_demo, length, CONFIG["memory_stride"])})
        last_demo = max((f for f in canonical if f < n_demo), default=-1)
        expected_tail = list(range(max(last_demo + 1, n_demo - 15, 0), n_demo))
        require(layout["frames"] == tail["canonical_frames"] == canonical
                and layout["is_demo"] == [table["is_demo"][f] for f in canonical], "Canonical layout differs")
        require(tail["frames"] == payload["frames"] == expected_tail
                and tail["n_demo"] == n_demo and tail["last_canonical_demo"] == last_demo,
                "Sidecar omitted-demo frame rule differs")
        result = episode_targets(eid, split, table["is_demo"], table["simple_subgoal"],
            table["simple_subgoal_online"], canonical, layout["decision_mask"], expected_tail,
            margin=CONFIG["margin"], memory_window=CONFIG["memory_window"])
        rows = result.pop("examples")
        audits.append({"episode_id": eid, "split": split, **result, "examples": len(rows)})
        c = coverage[split]
        c.update(selected_episodes=1, eligible_episodes=int(result["episode_exclusion"] is None),
                 episodes_with_examples=int(bool(rows)), active_decisions=result["active_decisions"],
                 eligible_active_decisions=result["active_decisions"] if result["episode_exclusion"] is None else 0,
                 excluded_episode_active_decisions=result["active_decisions"] if result["episode_exclusion"] else 0,
                 examples=len(rows), examples_with_old_positive=sum(bool(r["old_positive_frames"]) for r in rows))
        if result["episode_exclusion"]:
            exclusions[split]["episodes"][result["episode_exclusion"]] += 1
        exclusions[split]["queries"].update(result["query_exclusions"])
        examples.extend(rows)
    vocabulary = sorted({row["label"] for row in examples if row["split"] == "train"})
    for row in examples:
        row["label_id"] = vocabulary.index(row["label"]) if row["label"] in vocabulary else -1
    manifest = {"format_version": 1, "kind": KIND, "status": "complete", "config": CONFIG.copy(),
        "identity": {"cache_dir": str(cache_dir), "cache_fingerprint": cache["fingerprint"],
            "sidecar_dir": str(sidecar_dir), "sidecar_fingerprint": sidecar["fingerprint"],
            "base_model": str(base), "base_checkpoint_identity_recorded": cache["identity"]["checkpoint"],
            "base_weights_loaded_or_rehashed": False, "dataset_path": str(dataset),
            "selected_episode_ids": selected, "task_family_exact_instructions": sorted(GOALS),
            "original_splits": cache["splits"]},
        "vocabulary": vocabulary, "vocabulary_source": "retained_TRAIN_targets_only",
        "coverage": {k: dict(v) for k, v in coverage.items()},
        "exclusions": {k: {name: dict(v) for name, v in split.items()} for k, split in exclusions.items()},
        "episode_audits": audits, "examples": examples, "files_sha256": files, "source_sha256": sources,
        "claims": {"verified_visual_cue": False, "atomic_event_alignment": False,
                   "simulation_accuracy_evidence": False, "labels_are_model_inputs": False}}
    verify_identity(manifest)
    manifest["fingerprint"] = digest(manifest)
    return validate_manifest(manifest)


def validate_output_scope(manifest, output_dir):
    """Do not publish even a new child directory inside an immutable input root."""
    output = Path(output_dir).resolve()
    require(not output.exists(), "Output must be a NEW directory")
    for name in ("cache_dir", "sidecar_dir", "base_model", "dataset_path"):
        protected = Path(manifest["identity"][name]).resolve(strict=True)
        require(not output.is_relative_to(protected), f"Output is inside protected {name}")
    return output


def publish(manifest, output_dir):
    """Publish one manifest atomically, refusing any existing output directory."""
    validate_manifest(manifest)
    output = validate_output_scope(manifest, output_dir)
    verify_identity(manifest)
    output.mkdir(parents=True, exist_ok=False)
    fd, temporary = tempfile.mkstemp(prefix="manifest.", suffix=".tmp", dir=output)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output / "manifest.json")
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return output / "manifest.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--sidecar-dir", required=True)
    parser.add_argument("--output-dir")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    require(not args.publish or args.output_dir, "--publish requires --output-dir")
    if args.output_dir:
        require(not Path(args.output_dir).exists(), "Output must be a NEW directory")
    manifest = build_manifest(args.cache_dir, args.sidecar_dir)
    if args.output_dir:
        validate_output_scope(manifest, args.output_dir)
    destination = str(publish(manifest, args.output_dir)) if args.publish else None
    print(json.dumps({"preflight_only": not args.publish, "output": destination,
        "fingerprint": manifest["fingerprint"], "coverage": manifest["coverage"],
        "exclusions": manifest["exclusions"], "vocabulary": manifest["vocabulary"],
        "protected_files": len(manifest["files_sha256"]), "sources": len(manifest["source_sha256"])}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
