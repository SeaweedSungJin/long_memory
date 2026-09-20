"""Train-only supervision sidecars for immutable RoboMME endpoint caches.

These are *targets*, never policy observations. ``online`` refers to the
simulator's current required subgoal, not an online-accessible nonprivileged
input. It can advance earlier than the recording planner's subgoal. We keep
the two annotation views separate. Available-old is only an age/availability
proxy: neither this flag nor subgoal accuracy proves memory dependence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

import torch

from .cache import EpisodeCache, _atomic_json, decision_frames
from .safety_v5 import validate_output_scope


FORMAT = "robomme_recall_targets_v5"
INVALID_TEXT = {"", "unknown", "none", "null", "nan", "static", "no record", "done", "completed"}
PIXEL = re.compile(r"<\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*>")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def normalized_text(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    if not isinstance(value, str):
        return ""
    value = " ".join(value.strip().split())
    return "" if value.lower() in INVALID_TEXT else value


def parse_grounded_xy(text, *, height, width):
    """RoboMME stores <row, column> = <y,x>; expose normalized [x,y].

    A single output cannot represent several grounded objects. Multiple,
    missing, nonfinite or out-of-image coordinates are masked, never guessed.
    Values are divided by width-1 / height-1, matching corner pixel centers.
    """
    if height < 2 or width < 2:
        raise ValueError("Source image dimensions must exceed one pixel")
    text = normalized_text(text)
    # Counting numeric matches alone could silently choose the wrong object
    # from e.g. 'from <64,128> to <target>' or '<nan,1> and <2,3>'. Reject
    # every additional/unresolved/malformed angle-bracket target instead.
    matches = PIXEL.findall(text)
    if text.count("<") != 1 or text.count(">") != 1 or len(matches) != 1:
        return [0.0, 0.0], False
    y, x = map(float, matches[0])
    valid = math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height
    return ([x / (width - 1), y / (height - 1)], True) if valid else ([0.0, 0.0], False)


def _inside(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Path escapes its declared source directory")
    return path


def _cache_episode(cache, record):
    path = _inside(cache.path, record["path"])
    ep = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    if ep.get("cache_fingerprint") != cache.manifest["fingerprint"] or int(ep["episode_id"]) != int(record["episode_id"]):
        raise ValueError("Cache episode identity mismatch")
    return ep


def endpoint_targets(table, ep, *, eid, task, split, view, height, width, window, stride):
    """Pure row alignment; no action values or future subgoals become inputs."""
    import numpy as np

    suffix = "_online" if view == "online" else ""
    required = {"episode_index", "frame_index", "is_demo", "simple_subgoal" + suffix,
                "grounded_subgoal" + suffix}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing source labels: {sorted(required - set(table.columns))}")
    n = len(table)
    frames = ep["frames"].long().tolist()
    if not n or not np.issubdtype(table["frame_index"].dtype, np.integer) or table["frame_index"].tolist() != list(range(n)):
        raise ValueError("Raw frame_index must match positional cache rows exactly")
    if any(type(x) not in (int, np.int32, np.int64) or int(x) != eid for x in table["episode_index"]):
        raise ValueError("Raw episode_index differs from cache episode ID")
    raw_demo = table["is_demo"].to_numpy()
    if raw_demo.dtype != np.bool_:
        raise ValueError("Source is_demo must be explicitly boolean")
    if frames != decision_frames(raw_demo, stride).tolist():
        raise ValueError("Cache endpoints differ from raw demonstration/stride alignment")
    if "is_demo" not in ep or ep["is_demo"].tolist() != raw_demo[frames].tolist():
        raise ValueError("Cache/source passive demonstration masks differ")
    if len(ep["decision_mask"]) != len(frames) - 1 or len(ep["transition_valid"]) != len(frames) - 1:
        raise ValueError("Invalid cached decision/event lengths")
    if any(bool(ep["decision_mask"][d]) and raw_demo[f] for d, f in enumerate(frames[:-1])):
        raise ValueError("Passive endpoint marked trainable in cache")
    result = []
    for d, frame in enumerate(frames):
        active = d < len(frames) - 1 and bool(ep["decision_mask"][d]) and not bool(raw_demo[frame])
        text = normalized_text(table.iloc[frame]["simple_subgoal" + suffix]) if active else ""
        xy, xy_valid = parse_grounded_xy(table.iloc[frame]["grounded_subgoal" + suffix],
                                        height=height, width=width)
        oldest_short = frames[max(0, d - window + 1)]
        old = any(bool(ep["transition_valid"][i]) and frames[i + 1] < oldest_short for i in range(d))
        result.append({"decision": d, "frame": frame, "episode_id": eid, "task": task, "split": split,
                       "text": text, "xy": xy if active and text and xy_valid else [0.0, 0.0],
                       "xy_valid": bool(active and text and xy_valid), "active": active,
                       "available_old": bool(active and old)})
    return result


def _source_signatures(cache):
    return {str(Path(s["path"]).resolve()): s for kind in ("metadata", "payloads")
            for s in cache.manifest.get("identity", {}).get(kind, [])}


def _validate_original_signature(path, signatures):
    signature = signatures.get(str(Path(path).resolve()))
    if signature is None:
        raise ValueError(f"Source is absent from cache provenance: {path}")
    stat = Path(path).stat()
    if stat.st_size != signature["size"] or stat.st_mtime_ns != signature["mtime_ns"]:
        raise ValueError(f"Source changed since cache extraction: {path}")
    if signature.get("sha256") and sha256_file(path) != signature["sha256"]:
        raise ValueError(f"Source hash differs from cache provenance: {path}")


def prepare_recall_labels(cache, dataset_path, output_dir, *, view="online", verify_video=True):
    """Build a new sidecar, never replace an old directory or edit the cache."""
    import pandas as pd

    if view not in ("online", "planner"):
        raise ValueError("Annotation view must be online or planner")
    root, output = Path(dataset_path).resolve(), Path(output_dir).resolve()
    validate_output_scope(output, root, cache.path, cache.manifest["model_path"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite recall sidecar: {output}")
    if root != Path(cache.manifest["dataset_path"]).resolve():
        raise ValueError("Dataset root differs from cache provenance")
    signatures = _source_signatures(cache)
    info_path = root / "meta/info.json"
    _validate_original_signature(info_path, signatures)
    info = json.loads(info_path.read_text())
    image_schema = info["features"].get("image", {})
    if image_schema.get("names") != ["height", "width", "channel"] or len(image_schema.get("shape", [])) != 3:
        raise ValueError("Need explicit source front-image height/width/channel schema")
    height, width, channels = image_schema["shape"]
    if not all(type(x) is int for x in (height, width, channels)) or channels != 3 or min(height, width) < 2:
        raise ValueError("Invalid source image shape")
    base_config_path = Path(cache.manifest["model_path"]) / "config.json"
    base_config = json.loads(base_config_path.read_text())
    window, stride = int(base_config["memory_window"]), int(base_config["memory_stride"])
    if window <= 0 or stride <= 0:
        raise ValueError("Invalid HAMLET memory window/stride")
    splits = {key: list(map(int, cache.manifest["splits"][key])) for key in ("train", "val")}
    split_map = {eid: key for key, ids in splits.items() for eid in ids}
    rows_by_episode, sources, classes = {}, [], set()
    first_record = cache.manifest["episodes"][0]
    dimension_check = {"schema": [height, width], "video_header_checked": False}
    if verify_video:
        import cv2

        eid = int(first_record["episode_id"])
        video = _inside(root, info["video_path"].format(episode_index=eid,
                        episode_chunk=eid // info["chunks_size"], video_key="image"))
        _validate_original_signature(video, signatures)
        cap = cv2.VideoCapture(str(video))
        try:
            dimensions = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))]
            if not cap.isOpened() or dimensions != [height, width]:
                raise ValueError("Source front-video dimensions disagree with image schema")
        finally:
            cap.release()
        dimension_check.update(video_header_checked=True, path=str(video), height=height, width=width)
    for number, record in enumerate(cache.manifest["episodes"]):
        eid = int(record["episode_id"])
        path = _inside(root, info["data_path"].format(episode_index=eid, episode_chunk=eid // info["chunks_size"]))
        _validate_original_signature(path, signatures)
        columns = ["episode_index", "frame_index", "is_demo", "simple_subgoal" + ("_online" if view == "online" else ""),
                   "grounded_subgoal" + ("_online" if view == "online" else "")]
        table = pd.read_parquet(path, columns=columns)
        ep = _cache_episode(cache, record)
        rows = endpoint_targets(table, ep, eid=eid, task=str(record["task"]), split=split_map[eid],
                                view=view, height=height, width=width, window=window, stride=stride)
        rows_by_episode[eid] = rows
        if record.get("split", split_map[eid]) != split_map[eid]:
            raise ValueError("Cached episode split differs from manifest split list")
        if split_map[eid] == "train":
            classes.update(row["text"] for row in rows if row["text"])
        sources.append({"path": str(path), "sha256": sha256_file(path), "episode_id": eid})
        if number % 100 == 0:
            print(f"[recall-labels] validated {number + 1}/{len(cache.manifest['episodes'])} episodes", flush=True)
    class_names = sorted(classes)
    if not class_names:
        raise ValueError("Training split contains no valid recall classes")
    vocab = {name: index for index, name in enumerate(class_names)}
    coverage = defaultdict(Counter)
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    episodes = []
    for eid, rows in rows_by_episode.items():
        for row in rows:
            row["class_id"] = vocab.get(row["text"], -1)
            row["class_valid"] = row["class_id"] >= 0 and row["active"]
            # Unknown validation text is never converted into a newly learned class.
            row["xy_valid"] = bool(row["xy_valid"] and row["class_valid"])
            for group in (row["split"], row["split"] + "/" + row["task"]):
                coverage[group].update({"endpoints": 1, "active": int(row["active"]),
                                       "class_valid": int(row["class_valid"]), "xy_valid": int(row["xy_valid"]),
                                       "unknown_class": int(bool(row["text"]) and row["class_id"] < 0),
                                       "available_old": int(row["available_old"])})
        relative = f"episodes/episode_{eid:06d}.json"
        _atomic_json(output / relative, {"episode_id": eid, "rows": rows})
        episodes.append({"episode_id": eid, "path": relative, "sha256": sha256_file(output / relative),
                         "split": rows[0]["split"], "task": rows[0]["task"]})
    manifest = {"format": FORMAT, "status": "complete", "cache_fingerprint": cache.manifest["fingerprint"],
                "cache_manifest_sha256": sha256_file(Path(cache.path) / "manifest.json"),
                "dataset_path": str(root), "view": view, "memory_window": window, "memory_stride": stride,
                "image_dimensions": dimension_check, "class_names": class_names, "num_classes": len(class_names),
                "splits": splits, "episodes": episodes, "sources": sources,
                "metadata_sources": [{"path": str(info_path), "sha256": sha256_file(info_path)},
                                     {"path": str(base_config_path), "sha256": sha256_file(base_config_path)}],
                "coverage": dict(coverage), "coordinate_order": "source_yx_to_normalized_xy",
                "available_old_is_memory_dependence_label": False,
                "target_semantics": "Current simulator-required subgoal; privileged supervision, never policy input."}
    manifest["fingerprint"] = json_fingerprint(manifest)
    _atomic_json(output / "manifest.json", manifest)
    lines = ["RoboMME v5 recall-label readiness", f"Annotation view: {view}",
             f"Training-only classes: {len(class_names)}", f"Cache fingerprint: {manifest['cache_fingerprint']}",
             f"Label fingerprint: {manifest['fingerprint']}",
             "Subgoals are privileged targets, NEVER model observations.",
             "available_old means a valid completed event exists in the entire historical prefix",
             "before the oldest HAMLET short-memory endpoint. It does not mean that event",
             "survives the learned bank or is necessary for the query.",
             "Unknown validation classes, passive demo labels, terminal labels and ambiguous",
             "multi-point grounding targets are masked. Subgoal label is not task success.", "",
             "Split/task | active | class valid | XY valid | unknown class | old available"]
    for group, counts in sorted(coverage.items()):
        lines.append(f"{group} | {counts['active']} | {counts['class_valid']} | {counts['xy_valid']} | "
                     f"{counts['unknown_class']} | {counts['available_old']}")
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


class RecallLabels:
    """Hash-checked labels with the immutable cache's exact episode split."""

    def __init__(self, root, cache, *, verify_sources=True):
        self.root, self.cache = Path(root).resolve(), cache
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        info = self.manifest
        if info.get("format") != FORMAT or info.get("status") != "complete":
            raise ValueError("Expected a complete v5 recall target sidecar")
        if info.get("fingerprint") != json_fingerprint({k: v for k, v in info.items() if k != "fingerprint"}):
            raise ValueError("Recall manifest fingerprint mismatch")
        if info["cache_fingerprint"] != cache.manifest["fingerprint"] or info["splits"] != cache.manifest["splits"]:
            raise ValueError("Recall/cache identity or split mismatch")
        if info["cache_manifest_sha256"] != sha256_file(Path(cache.path) / "manifest.json"):
            raise ValueError("Cache manifest changed after label preparation")
        if info["class_names"] != sorted(set(info["class_names"])) or len(info["class_names"]) != info["num_classes"]:
            raise ValueError("Invalid train-only recall vocabulary")
        self.records = {int(r["episode_id"]): r for r in info["episodes"]}
        self.cache_records = {int(r["episode_id"]): r for r in cache.manifest["episodes"]}
        if len(self.records) != len(info["episodes"]) or set(self.records) != set(self.cache_records):
            raise ValueError("Recall/cache episodes differ")
        if verify_sources:
            for source in info["sources"] + info["metadata_sources"]:
                if sha256_file(source["path"]) != source["sha256"]:
                    raise ValueError(f"Recall source hash mismatch: {source['path']}")

    @lru_cache(maxsize=16)
    def _rows(self, eid):
        record = self.records[eid]
        path = _inside(self.root, record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError("Recall episode labels changed after preparation")
        payload = json.loads(path.read_text())
        ep = _cache_episode(self.cache, self.cache_records[eid])
        rows = payload["rows"]
        if payload["episode_id"] != eid or [r["frame"] for r in rows] != ep["frames"].tolist():
            raise ValueError("Recall endpoint identity mismatch")
        for d, row in enumerate(rows):
            cid = row["class_id"]
            if row["decision"] != d or row["episode_id"] != eid or row["split"] != record["split"]:
                raise ValueError("Recall decision/split mismatch")
            if type(cid) is not int or not -1 <= cid < self.manifest["num_classes"]:
                raise ValueError("Recall class out of vocabulary")
            if row["class_valid"] != (cid >= 0 and row["active"]):
                raise ValueError("Recall class mask inconsistent")
            if row["active"] != (d < len(rows) - 1 and bool(ep["decision_mask"][d]) and not bool(ep["is_demo"][d])):
                raise ValueError("Recall passive/terminal mask inconsistent")
            if len(row["xy"]) != 2 or any(not isinstance(x, (float, int)) or not math.isfinite(x) or not 0 <= x <= 1 for x in row["xy"]):
                raise ValueError("Recall coordinate outside normalized image")
        return rows

    def get(self, eid, decision):
        if type(decision) is not int or decision < 0:
            raise ValueError("Decision must be a nonnegative integer endpoint index")
        rows = self._rows(int(eid))
        if decision >= len(rows):
            raise IndexError("Recall decision outside episode")
        # Return fresh containers; callers cannot alter the validated cache.
        row = dict(rows[decision])
        row["xy"] = list(row["xy"])
        return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dataset-path", help="Defaults to immutable cache provenance")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--annotation-view", choices=("online", "planner"), default="online")
    args = parser.parse_args(argv)
    cache = EpisodeCache(args.cache_dir)
    manifest = prepare_recall_labels(cache, args.dataset_path or cache.manifest["dataset_path"],
                                     args.output_dir, view=args.annotation_view)
    print(json.dumps({"fingerprint": manifest["fingerprint"], "num_classes": manifest["num_classes"],
                      "coverage": {s: manifest["coverage"][s] for s in ("train", "val")}}, indent=2))
    print("[recall-labels] Supervision only; this does not measure robot success or prove long-memory use.")
    return 0
