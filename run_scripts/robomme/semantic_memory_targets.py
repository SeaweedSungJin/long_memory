#!/usr/bin/env python3
"""Labels-only memory-answer targets on immutable TRAIN/cache-VAL episodes.

This is not V15's matching-equal-subgoal retrieval supervision. In particular,
VideoPlaceOrder/Button answers identify the *selected past target anchor*, not
the word "place", a weak segment ID, or a current tracked object position.
Metadata, anchors, episode IDs and target masks MUST NOT enter policy inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KIND = "semantic_memory_answers_v1"
RAW_COLUMNS = ("episode_index", "frame_index", "is_demo", "simple_subgoal",
               "simple_subgoal_online", "grounded_subgoal", "grounded_subgoal_online")
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
DIRECTIONS = ("forward", "backward", "left", "right", "forward-left", "forward-right",
              "backward-left", "backward-right")
EXCLUDED = frozenset(("", "no record", "static", "remain static"))
ANCHOR_NAME = "past_target_anchor_yx"
CONFIG = {
    "anchor_coordinate_order": "y,x", "anchor_normalizer": 255.0,
    "anchor_consistency_pixels": 12.0, "boundary_margin_frames": 2,
    "target_semantics": "subgoal-onset past demo target anchor, NOT current tracking",
    "labels_only": True, "splits": ["train", "val"],
    "classification_fit": "TRAIN-only vocabulary; globally constant heads disabled",
    "answer_query_input": "actual reader attention output only, no labels/state shortcut",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def label_text(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    require(isinstance(value, str), "Metadata label must be string or UTF8 bytes")
    return " ".join(value.lower().split())


def compositional_labels(online):
    """Conservative exact attributes; no generic action-name classification."""
    text = label_text(online)
    if text in EXCLUDED:
        return {}
    labels = {}
    if text in {"move " + x for x in DIRECTIONS}:
        labels["direction"] = text.removeprefix("move ")
    ordinals = [x for x in ORDINALS if re.search(r"\b" + x + r"\b", text)]
    if len(ordinals) == 1:
        labels["ordinal"] = ordinals[0]
    grasp = re.findall(r"grasping the (near|far) end", text)
    if len(grasp) == 1:
        labels["grasp_end"] = grasp[0]
    side = re.findall(r"insert the peg from the (left|right) side", text)
    if len(side) == 1:
        labels["insert_side"] = side[0]
    return labels


def maximal_runs(labels, start, end):
    runs = []
    for frame in range(start, end):
        value = label_text(labels[frame])
        if not runs or runs[-1]["label"] != value:
            runs.append({"start": frame, "end": frame + 1, "label": value})
        else:
            runs[-1]["end"] = frame + 1
    return runs


def role_anchor(text, *, execution=False):
    """Parse the target role only, never the first coordinate in arbitrary text."""
    role = "place the cube onto the correct target" if execution else "drop the cube onto target"
    match = re.fullmatch(re.escape(role) + r" at <(\d+),\s*(\d+)>", label_text(text))
    if match is None:
        return None
    point = [int(match[1]), int(match[2])]
    return point if all(0 <= x <= 255 for x in point) else None


def selected_demo_anchor(goal, raw, frames):
    """Validated instruction-to-observed-past relation. Returns target or reason.

    Grounded annotations are subgoal-onset anchors. Whole-episode annotations
    may validate a training target; no such future/privileged field is exposed
    as an inference input. Current target coordinates are only a consistency
    check: the actual regression answer is taken from the past demo.
    """
    goal = label_text(goal)
    order = re.fullmatch(r"watch the video carefully, then place the \w+ cube on the (\w+) target it was previously placed on", goal)
    button = re.fullmatch(r"watch the video carefully, then place the \w+ cube on the target right (before|after) the button was pressed", goal)
    if order is None and button is None:
        return None, "not_supported_anchor_task"
    n_demo = sum(raw["is_demo"])
    if not n_demo:
        return None, "anchor_no_demo"
    runs = maximal_runs(raw["simple_subgoal"], 0, n_demo)
    targets = [r for r in runs if r["label"] == "drop the cube onto target"]
    if order:
        if order[1] not in ORDINALS:
            return None, "anchor_unknown_ordinal"
        ordinal = ORDINALS.index(order[1])
        if ordinal >= len(targets):
            return None, "anchor_missing_requested_occurrence"
        selected = targets[ordinal]
        relation = {"kind": "ordinal", "ordinal": ordinal + 1}
    else:
        presses = [r for r in runs if r["label"] == "press the button"]
        if len(presses) != 1:
            return None, "anchor_ambiguous_button_occurrence"
        before = [r for r in targets if r["end"] <= presses[0]["start"]]
        after = [r for r in targets if r["start"] >= presses[0]["end"]]
        options = before[-1:] if button[1] == "before" else after[:1]
        if not options:
            return None, "anchor_no_adjacent_placement"
        selected = options[0]
        relation = {"kind": "button_" + button[1], "button_start": presses[0]["start"]}
    start, end = selected["start"], selected["end"]
    anchors = {tuple(role_anchor(v) or ()) for v in raw["grounded_subgoal"][start:end]}
    if len(anchors) != 1 or () in anchors:
        return None, "anchor_past_grounding_missing_or_changes_within_run"
    anchor = list(next(iter(anchors)))
    margin = CONFIG["boundary_margin_frames"]
    observed = [f for f in frames if start + margin <= f < end - margin
                and label_text(raw["simple_subgoal_online"][f]) == selected["label"]]
    if not observed:
        return None, "anchor_past_run_not_observed_with_online_agreement"
    # Refuse spatially inconsistent annotations; missing online coordinates are
    # allowed only because the stable planner role anchor is explicitly known.
    tolerance = CONFIG["anchor_consistency_pixels"]
    if any((p := role_anchor(raw["grounded_subgoal_online"][f])) is not None
           and math.dist(p, anchor) > tolerance for f in observed):
        return None, "anchor_past_planner_online_spatial_disagreement"
    execution = []
    for f in range(n_demo, len(raw["is_demo"])):
        p = role_anchor(raw["grounded_subgoal"][f], execution=True)
        q = role_anchor(raw["grounded_subgoal_online"][f], execution=True)
        if p is not None and q is not None:
            if math.dist(p, q) > tolerance:
                return None, "anchor_execution_planner_online_spatial_disagreement"
            execution.append(p)
    if not execution:
        return None, "anchor_execution_target_not_validated"
    if any(math.dist(p, anchor) > tolerance for p in execution):
        return None, "anchor_selected_past_execution_target_disagree"
    return {"point_yx": anchor, "source_interval": [start, end],
            "observed_source_frames": observed, "relation": relation,
            "semantics": CONFIG["target_semantics"]}, None


def episode_rows(raw, episode, episode_id, split, goal, *, stride=16):
    """Align only labels to existing cached decision IDs. Never inspect actions."""
    import numpy as np
    from gr00t.long_memory.cache import decision_frames
    n = len(raw["frame_index"])
    require(n > 1 and all(len(raw[k]) == n for k in RAW_COLUMNS), "Metadata lengths differ")
    require(raw["frame_index"] == list(range(n)) and raw["episode_index"] == [episode_id] * n,
            "Raw episode/frame identity differs")
    require(all(type(x) is bool for x in raw["is_demo"]), "Demo flags must be booleans")
    n_demo = sum(raw["is_demo"])
    require(raw["is_demo"] == [True] * n_demo + [False] * (n - n_demo),
            "Demo must be a contiguous observed prefix")
    frames = list(episode["frames"])
    decision_mask = list(episode["decision_mask"])
    require(frames == decision_frames(np.asarray(raw["is_demo"], dtype=bool), stride).tolist(),
            "Cache observation cadence differs from metadata")
    require(list(episode["is_demo"]) == [raw["is_demo"][f] for f in frames]
            and len(decision_mask) == len(frames) - 1, "Cache demo/decision layout differs")
    anchor, reason = selected_demo_anchor(goal, raw, frames)
    exclusions = Counter({reason: 1}) if reason else Counter()
    rows = []
    online_runs = maximal_runs(raw["simple_subgoal_online"], n_demo, n)
    for decision, active in enumerate(decision_mask):
        if not active:
            continue
        f = frames[decision]
        require(not raw["is_demo"][f], "Demo cannot receive an action decision target")
        label = label_text(raw["simple_subgoal_online"][f])
        run = next(r for r in online_runs if r["start"] <= f < r["end"])
        raw_labels = compositional_labels(label)
        regression, provenance = {}, {}
        if label in EXCLUDED:
            exclusions["excluded_online_label"] += 1
            raw_labels = {}
        # Avoid online/planner transition ambiguities for compositional labels.
        if label_text(raw["simple_subgoal"][f]) != label:
            exclusions["planner_online_disagreement"] += 1
            raw_labels = {}
        if not run["start"] + 2 <= f < run["end"] - 2:
            exclusions["online_boundary_margin"] += 1
            raw_labels = {}
        if (anchor and label == "place the cube onto the correct target"
                and label_text(raw["simple_subgoal"][f]) == label
                and run["start"] + 2 <= f < run["end"] - 2
                and max(anchor["observed_source_frames"]) < f):
            regression[ANCHOR_NAME] = [x / CONFIG["anchor_normalizer"] for x in anchor["point_yx"]]
            provenance[ANCHOR_NAME] = anchor
        rows.append({"episode_id": episode_id, "decision": decision, "query_frame": f,
                     "split": split, "classification_labels": raw_labels,
                     "classification": {}, "regression": regression,
                     "target_provenance": provenance})
    return rows, dict(exclusions)


def fit_vocabulary(rows):
    """A VAL-only label can never allocate a head or determine vocabulary."""
    train = defaultdict(set)
    for row in rows:
        if row["split"] == "train":
            for key, value in row["classification_labels"].items():
                train[key].add(value)
    vocabulary = {key: sorted(values) for key, values in sorted(train.items()) if len(values) > 1}
    disabled = {key: "constant_or_absent_in_train" for key, values in train.items() if len(values) < 2}
    for row in rows:
        row["classification"] = {key: vocabulary[key].index(value)
            for key, value in row["classification_labels"].items()
            if key in vocabulary and value in vocabulary[key]}
    # Regression heads also require varied TRAIN answers, not a constant anchor.
    train_reg = defaultdict(set)
    for row in rows:
        if row["split"] == "train":
            for key, value in row["regression"].items():
                train_reg[key].add(tuple(value))
    regression_sizes = {key: len(next(iter(values))) for key, values in train_reg.items() if len(values) > 1}
    for row in rows:
        row["regression"] = {k: v for k, v in row["regression"].items() if k in regression_sizes}
    return vocabulary, regression_sizes, disabled


def prepare(cache_dir, dataset_path=None):
    import pyarrow.parquet as pq
    import torch
    require(not torch.cuda.is_initialized(), "Metadata preparation must not initialize CUDA")
    cache_dir = Path(cache_dir).resolve(strict=True)
    cache_path = cache_dir / "manifest.json"
    cache = json.loads(cache_path.read_text())
    require(cache.get("status") == "complete", "Require completed immutable cache")
    dataset = Path(dataset_path or cache["dataset_path"]).resolve(strict=True)
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    splits = cache["splits"]
    require(set(splits) == {"train", "val"} and not set(splits["train"]) & set(splits["val"]),
            "Require disjoint immutable TRAIN/cache-VAL splits, never TEST")
    rows, episodes, missing, exclusions = [], [], Counter(), Counter()
    record_ids = set()
    for record in sorted(cache["episodes"], key=lambda x: x["episode_id"]):
        eid, split = record["episode_id"], record["split"]
        require(eid not in record_ids and eid in splits[split], "Duplicate or misplaced cache episode")
        record_ids.add(eid)
        path = (cache_dir / record["path"]).resolve(strict=True)
        require(path.is_relative_to(cache_dir), "Cache path escapes declared root")
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        require(payload["episode_id"] == eid and payload["cache_fingerprint"] == cache["fingerprint"],
                "Cache payload identity differs")
        layout = {key: payload[key].tolist() for key in ("frames", "is_demo", "decision_mask")}
        parquet = (dataset / info["data_path"].format(episode_chunk=eid // info["chunks_size"], episode_index=eid)).resolve(strict=True)
        require(parquet.is_relative_to(dataset), "Metadata path escapes declared dataset")
        schema = pq.read_schema(parquet)
        absent = sorted(set(RAW_COLUMNS) - set(schema.names))
        identity = {"episode_id": eid, "split": split, "cache_path": str(path),
                    "cache_layout_sha256": digest(layout), "metadata_path": str(parquet),
                    "metadata_sha256": file_sha256(parquet), "schema_sha256": hashlib.sha256(schema.serialize()).hexdigest(),
                    "available_columns": schema.names, "missing_required_columns": absent}
        if absent:
            missing.update(absent)
            identity["target_queries"] = 0
        else:
            raw = pq.read_table(parquet, columns=list(RAW_COLUMNS)).to_pydict()
            local, reasons = episode_rows(raw, layout, eid, split, record["task"], stride=cache["action_steps"])
            rows.extend(local)
            exclusions.update(reasons)
            identity["target_queries"] = len(local)
        episodes.append(identity)
    require(record_ids == set(splits["train"]) | set(splits["val"]), "Cache episode coverage differs from splits")
    vocabulary, regression_sizes, disabled = fit_vocabulary(rows)
    for row in rows:
        row["has_target"] = bool(row["classification"] or row["regression"])
    coverage = {}
    for split in ("train", "val"):
        selected = [r for r in rows if r["split"] == split]
        coverage[split] = {"episodes": len(splits[split]), "queries": len(selected),
            "queries_with_targets": sum(r["has_target"] for r in selected),
            "head_counts": dict(Counter(k for r in selected for k in list(r["classification"]) + list(r["regression"]))),
            "unseen_class_labels": dict(Counter(k for r in selected for k in r["classification_labels"] if k in vocabulary and k not in r["classification"]))}
    manifest = {"kind": KIND, "format_version": 1, "status": "complete", "config": CONFIG,
        "cache_dir": str(cache_dir), "cache_fingerprint": cache["fingerprint"],
        "cache_manifest_sha256": file_sha256(cache_path), "dataset_path": str(dataset),
        "dataset_info_sha256": file_sha256(info_path), "splits": splits, "episodes": episodes,
        "source_sha256": {str(Path(__file__).resolve()): file_sha256(__file__)},
        "vocabulary": vocabulary, "regression_sizes": regression_sizes, "disabled_heads": disabled,
        "coverage": coverage, "missing_column_episode_counts": dict(missing),
        "exclusion_counts": dict(exclusions), "rows": rows,
        "identity_limits": "Feature tensor bytes are not duplicated/hashed here; existing immutable cache fingerprint plus exact layout and metadata hashes bind supervision."}
    manifest["fingerprint"] = digest(manifest)
    return manifest


class SemanticTargets:
    """Training-only lookup. This class is deliberately absent from policy code."""
    def __init__(self, path, cache_manifest):
        path = Path(path)
        if path.is_dir():
            path = path / "manifest.json"
        self.manifest = m = json.loads(path.read_text())
        require(m.get("kind") == KIND and m.get("status") == "complete" and m.get("config") == CONFIG,
                "Invalid semantic target schema")
        require(m["fingerprint"] == digest({k: v for k, v in m.items() if k != "fingerprint"}), "Target fingerprint differs")
        require(m["cache_fingerprint"] == cache_manifest["fingerprint"] and m["splits"] == cache_manifest["splits"],
                "Target cache identity/splits differ")
        require(not set(m["splits"]["train"]) & set(m["splits"]["val"]), "Targets split overlap")
        self.rows = {}
        for row in m["rows"]:
            key = (row["episode_id"], row["decision"])
            require(key not in self.rows and row["episode_id"] in m["splits"][row["split"]], "Duplicate/misplaced target")
            for name, value in row["classification"].items():
                require(type(value) is int and 0 <= value < len(m["vocabulary"][name]), "Invalid class target")
            for name, value in row["regression"].items():
                require(len(value) == m["regression_sizes"][name] and all(math.isfinite(x) and 0 <= x <= 1 for x in value), "Invalid normalized regression target")
            self.rows[key] = row

    def get(self, episode_id, decision):
        return self.rows.get((int(episode_id), int(decision)))

    def answer_config(self, input_dim=256, hidden_dim=128):
        return {"input_dim": input_dim, "hidden_dim": hidden_dim,
                "classification_sizes": {k: len(v) for k, v in self.manifest["vocabulary"].items()},
                "regression_sizes": self.manifest["regression_sizes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dataset-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    require(not output.exists(), f"Refusing existing output directory: {output}")
    manifest = prepare(args.cache_dir, args.dataset_path)
    print(json.dumps({k: manifest[k] for k in ("fingerprint", "coverage", "vocabulary", "regression_sizes", "exclusion_counts", "missing_column_episode_counts")}, indent=2))
    if args.preflight_only:
        print("[targets] preflight only: no files written, no CUDA/model loaded")
        return
    output.mkdir(parents=True, exist_ok=False)
    # Exclusive creation: old artifacts are never overwritten, including partial runs.
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"[targets] saved {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
