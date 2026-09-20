"""Read-only evidence review and memory-only audit for existing v4/v5 bundles.

This tool does not train, load the Action Expert/VLM, or run the simulator.
Prepared questions have UNKNOWN evidence labels; temporal age and instruction
text are navigation aids, never automatically promoted to causal supervision.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import html
import json
from pathlib import Path
import random
import sys

import torch

from .cache import EpisodeCache
from .cache_reader_v3 import MappedEpisodes
from .checkpoint_v4 import file_sha256, v4_checkpoint_info
from .checkpoint_v5 import RECIPE, v5_checkpoint_info
from .core_v3 import ActionValueMemory, MemoryV3Config
from .hamlet import validate_cache_checkpoint
from .safety_v5 import validate_output_scope


ROOT = Path(__file__).resolve().parents[2]
VARIANT = "past-evidence-audit-v1"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare UNKNOWN cue/use review questions, never infer GT")
    prepare.add_argument("--cache-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--split", choices=("train", "val"), default="val")
    prepare.add_argument("--episodes", type=int, default=4)
    prepare.add_argument("--queries-per-episode", type=int, default=3)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--episode-ids", type=int, nargs="+", help="Explicit IDs override --episodes; all must belong to --split")
    prepare.add_argument("--instruction-contains", action="append", default=[],
                         help="Case-insensitive instruction substring; repeat for OR. These are NOT benchmark task IDs.")
    prepare.add_argument("--review-images", action="store_true", help="Embed bounded raw-video contact sheets in review.html")
    prepare.add_argument("--max-review-frames", type=int, default=16, help="Maximum thumbnails per question, per camera (2..64)")
    audit = commands.add_parser("audit", help="Replay frozen memory and score verified evidence annotations only")
    audit.add_argument("--cache-dir", type=Path, required=True)
    audit.add_argument("--annotations", type=Path, required=True)
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--writer-policy", choices=("all", "hard"), default="all",
                       help="all = FIFO; hard = learned Stage-2 writer, not an untrained Stage-1 writer")
    audit.add_argument("--device", default="cpu")
    audit.add_argument("--top-k", nargs="+", type=int, default=[1, 4])
    audit.add_argument("--preflight-only", action="store_true", help="Validate all labels and bundle; no output or deployed model allocation")
    return p


def resolve(path):
    path = Path(path).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path, records):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _output_path(args, cache):
    inputs = [cache.path, cache.manifest.get("model_path"), cache.manifest.get("dataset_path")]
    if args.command == "audit":
        inputs += [resolve(args.checkpoint), resolve(args.annotations).parent]
    result = validate_output_scope(resolve(args.output_dir), *inputs)
    if result.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {result}")
    return result


def _records(cache):
    rows = cache.manifest["episodes"]
    result = {int(row["episode_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate episode IDs in cache manifest")
    return result


def _episode_path(cache, eid):
    records = _records(cache)
    if eid not in records:
        raise ValueError(f"Episode {eid} is not in the selected cache")
    result = (Path(cache.path) / records[eid]["path"]).resolve()
    if not result.is_relative_to(Path(cache.path).resolve()) or not result.is_file():
        raise ValueError(f"Missing episode or path outside cache: {result}")
    return result


def _hash_inputs(cache, episode_ids, extra=()):
    paths = [Path(cache.path) / "manifest.json"]
    paths += [_episode_path(cache, eid) for eid in sorted(set(episode_ids))]
    paths += [Path(path) for path in extra]
    return {str(path): file_sha256(path) for path in paths}


def _check_unchanged(hashes):
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"Input changed during diagnosis; no results published: {path}")


def _code_hashes():
    names = ["evidence_audit_cli.py", "evidence_audit.py", "cache.py", "cache_reader_v3.py",
             "core_v3.py", "replay_v3.py", "diagnostic_interventions.py", "checkpoint_v4.py",
             "checkpoint_v5.py", "hamlet.py", "safety_v5.py"]
    paths = [Path(__file__).parent / name for name in names]
    paths.append(ROOT / "run_scripts/robomme/audit_long_memory_evidence.py")
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def _choose_episode_ids(args, cache):
    records = _records(cache)
    available = set(cache.manifest["splits"][args.split])
    needles = [text.casefold() for text in args.instruction_contains]
    if any(not text.strip() for text in needles):
        raise ValueError("Instruction filters must be nonempty")
    selected = sorted(eid for eid in available
                      if not needles or any(text in records[eid].get("task", "").casefold() for text in needles))
    if args.episode_ids is not None:
        ids = args.episode_ids
        if len(ids) != len(set(ids)) or any(type(eid) is not int or eid < 0 for eid in ids):
            raise ValueError("--episode-ids must be unique nonnegative integers")
        if not set(ids) <= set(selected):
            raise ValueError("Explicit episode IDs must match the requested split and instruction filters")
        return sorted(ids)
    # Balance exact instruction groups, not inferred RoboMME task names.
    groups = defaultdict(list)
    for eid in selected:
        groups[records[eid].get("task", "")].append(eid)
    rng = random.Random(args.seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for ids in groups.values():
        rng.shuffle(ids)
    result = []
    while len(result) < args.episodes and any(groups.values()):
        for key in keys:
            if groups[key] and len(result) < args.episodes:
                result.append(groups[key].pop())
    return result


def make_prepare_records(args, cache, fetch, episode_ids):
    """Deterministic chronological strata; no subgoal, XY, outcome, or cue GT read."""
    annotations, review = [], []
    records = _records(cache)
    for eid in episode_ids:
        episode = fetch(eid)
        decisions = [int(d) for d in torch.where(episode["decision_mask"])[0].tolist()
                     if 0 < d < len(episode["actions"])]
        n = min(args.queries_per_episode, len(decisions))
        rng = random.Random(args.seed * 1000003 + eid)
        chosen = [rng.choice(decisions[i * len(decisions) // n:(i + 1) * len(decisions) // n])
                  for i in range(n)]
        for decision in chosen:
            frames = episode["frames"][:decision + 1].tolist()
            row = {"schema_version": 1, "cache_fingerprint": cache.manifest["fingerprint"],
                   "episode_id": eid, "split": args.split, "decision": decision,
                   "query_frame": int(frames[-1]), "status": "unknown", "cues": [],
                   "reviewer": "", "notes": "", "instruction": records[eid].get("task", "")}
            annotations.append(row)
            valid = episode["transition_valid"][:decision].tolist()
            review.append({"episode_id": eid, "decision": decision, "query_frame": int(frames[-1]),
                           "instruction": row["instruction"], "sampled_frames_through_query": frames,
                           "completed_event_count": sum(valid),
                           "events": [{"event_id": i, "start_frame": frames[i], "end_frame": frames[i + 1],
                                       "valid": bool(valid[i])} for i in range(decision)],
                           "warning": "Endpoints show sampling only; not proof the encoded representation contains the cue."})
    if not annotations:
        raise ValueError("No eligible active nonfirst decisions in selected episodes")
    return annotations, review


def _video_paths(cache, episode_ids):
    """Use the modality map, since this dataset labels MP4 features as image."""
    dataset = Path(cache.manifest["dataset_path"]).resolve()
    info_path, modality_path = dataset / "meta/info.json", dataset / "meta/modality.json"
    if not info_path.is_file() or not modality_path.is_file():
        return {}, {}, "Raw video metadata unavailable; cached endpoint navigation remains usable."
    info, modality = _json(info_path), _json(modality_path)
    hashes = {str(path): file_sha256(path) for path in (info_path, modality_path)}
    saved_metadata = {str(Path(row["path"]).resolve()): row
                      for row in cache.manifest.get("identity", {}).get("metadata", [])}
    for path, digest in hashes.items():
        saved = saved_metadata.get(path)
        if saved and saved.get("sha256") != digest:
            raise ValueError(f"Raw video metadata changed since cache extraction: {path}")
    saved_payloads = {str(Path(row["path"]).resolve()): row
                      for row in cache.manifest.get("identity", {}).get("payloads", [])}
    keys = sorted({value.get("original_key", f"observation.images.{key}")
                   for key, value in modality.get("video", {}).items()})
    paths = {}
    for eid in episode_ids:
        rows = []
        for key in keys:
            path = (dataset / info["video_path"].format(video_key=key, episode_index=eid,
                        episode_chunk=eid // int(info["chunks_size"]))).resolve()
            if not path.is_relative_to(dataset):
                raise ValueError(f"Video path escapes source dataset: {path}")
            saved = saved_payloads.get(str(path))
            if saved and path.is_file():
                stat = path.stat()
                if saved["size"] != stat.st_size or saved["mtime_ns"] != stat.st_mtime_ns:
                    raise ValueError(f"Source video changed since cache extraction: {path}")
            rows.append({"key": key, "path": str(path), "exists": path.is_file()})
        paths[eid] = rows
    return paths, hashes, None


def _preview_frames(frames, count):
    if len(frames) <= count:
        return list(frames)
    return [frames[round(i * (len(frames) - 1) / (count - 1))] for i in range(count)]


def _review_html(review, *, images=False, max_frames=16):
    """Optional thumbnails are sampled observations <= query, never future frames.

    This contact sheet is only a navigation aid: brief cues between endpoints
    can be absent. Review the raw video before marking evidence verified.
    """
    cv2 = None
    if images:
        import cv2
    parts = ["<!doctype html><meta charset='utf-8'><title>Evidence review</title>",
             "<style>body{font:16px sans-serif;margin:2em}section{border-top:1px solid #aaa}figure{display:inline-block;margin:5px}img{max-width:240px}pre{white-space:pre-wrap}</style>",
             "<h1>Past evidence review — all labels start unknown</h1>",
             "<p>Review raw videos, then edit annotations.jsonl. These thumbnails are sampled endpoints, not all raw frames; missing thumbnails do not prove a missing cue. No future-than-query frame is shown.</p>"]
    for row in review:
        parts += [f"<section><h2>Episode {row['episode_id']}, decision {row['decision']}, query frame {row['query_frame']}</h2>",
                  f"<p>{html.escape(row['instruction'])}</p>",
                  f"<pre>Sampled frames: {html.escape(str(row['sampled_frames_through_query']))}</pre>"]
        for video in row.get("source_videos", []):
            parts.append(f"<p>{html.escape(video['key'])}: <code>{html.escape(video['path'])}</code></p>")
            if not images or not video["exists"]:
                continue
            capture = cv2.VideoCapture(video["path"])
            try:
                for frame in _preview_frames(row["sampled_frames_through_query"], max_frames):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, rgb = capture.read()
                    if not ok:
                        parts.append(f"<p>Frame {frame}: decode failed; not evidence of cue absence.</p>")
                        continue
                    height, width = rgb.shape[:2]
                    scale = min(1.0, 240 / max(width, height))
                    rgb = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))))
                    ok, data = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if not ok:
                        raise ValueError("JPEG preview encoding failed")
                    encoded = base64.b64encode(data.tobytes()).decode("ascii")
                    parts.append(f"<figure><img src='data:image/jpeg;base64,{encoded}'><figcaption>frame {frame}</figcaption></figure>")
            finally:
                capture.release()
        parts.append("</section>")
    return "\n".join(parts)


ANNOTATION_HELP = """# Evidence audit review

`annotations.jsonl` is editable; one row = one active cached decision.
All rows initially have `status: "unknown"`. Instructions, event age, and
attention are NOT cue labels. Inspect the original raw video through the
query frame before annotating; review.html thumbnails skip intermediate frames.

Do not change schema_version/cache_fingerprint/episode_id/split/decision/query_frame.
Set `reviewer` to your name and `notes` to the reason for your decision.

- `unknown`: uncertain; not scored as correct or incorrect evidence.
- `none`: you verified no past cue is required; keep cues empty.
- `verified`: at least one manually verified cue group is required.

Example cues for a verified row (replace these example frames with real ones):
`[{"cue_id":"object_identity","description":"The demonstrated target object",
"intervals":[[8,12],[20,24]]}]`
Each interval is inclusive, nonnegative, and strictly BEFORE query_frame.
Intervals within one cue group are interchangeable evidence occurrences.
Multiple cue groups mean ALL groups are needed (e.g. target identity + order).
Do not use the whole episode as a positive interval just because the target
subgoal is mentioned. A cue inside an unsampled interval is not automatically
represented by that interval's event endpoint features.

Then run the audit subcommand with this annotations file and a NEW output
directory outside the review, cache, dataset and checkpoint directories.
Unknown rows can already be audited for bank/attention structure, but they
never create pseudo-GT retrieval accuracy. The audit does NOT measure robot
success or prove that a retrieved feature semantically preserves the cue.
"""


def prepare(args):
    if args.episodes < 1 or args.queries_per_episode < 1 or args.seed < 0:
        raise ValueError("Positive episode/query counts and a nonnegative seed are required")
    if not 2 <= args.max_review_frames <= 64:
        raise ValueError("--max-review-frames must be between 2 and 64")
    cache = EpisodeCache(resolve(args.cache_dir))
    output = _output_path(args, cache)
    ids = _choose_episode_ids(args, cache)
    if not ids:
        raise ValueError("No episodes match split/instruction selection")
    hashes = _hash_inputs(cache, ids)
    annotations, review = make_prepare_records(args, cache, MappedEpisodes(cache).fetch, ids)
    videos, metadata_hashes, video_warning = _video_paths(cache, ids)
    hashes.update(metadata_hashes)
    for row in review:
        row["source_videos"] = videos.get(row["episode_id"], [])
    if args.review_images:
        for rows in videos.values():
            for video in rows:
                if video["exists"]:
                    hashes[video["path"]] = file_sha256(video["path"])
    review_html = _review_html(review, images=args.review_images, max_frames=args.max_review_frames)
    _check_unchanged(hashes)
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "annotations.jsonl", annotations)
    _write_json(output / "review.json", review)
    _write_json(output / "review_manifest.json", {
        "variant": VARIANT, "cache_fingerprint": cache.manifest["fingerprint"],
        "selection": "Exact-instruction group-balanced episodes; seeded chronological active-decision strata; no GT selection",
        "split": args.split, "seed": args.seed, "episode_ids": ids, "queries": len(annotations),
        "review_images": args.review_images, "max_review_frames": args.max_review_frames,
        "source_sha256": hashes, "code_sha256": _code_hashes(), "video_warning": video_warning,
        "labels": "ALL UNKNOWN; no causal cue labels inferred"})
    (output / "README.md").write_text(ANNOTATION_HELP, encoding="utf-8")
    (output / "review.html").write_text(review_html, encoding="utf-8")
    print(f"[evidence] Prepared {len(annotations)} UNKNOWN questions: {output / 'annotations.jsonl'}", flush=True)
    print(f"[evidence] Navigation: {output / 'review.html'}; manually inspect raw video before verified labels.", flush=True)
    return {"output_dir": str(output), "questions": len(annotations)}


def load_annotations(path):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, text in enumerate(handle, 1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid annotation JSON at line {line_number}: {error}") from error
            if not isinstance(row, dict) or type(row.get("episode_id")) is not int or type(row.get("decision")) is not int:
                raise ValueError(f"Line {line_number}: object with integer episode_id/decision required")
            key = (row["episode_id"], row["decision"])
            if key in seen:
                raise ValueError(f"Duplicate annotation episode/decision at line {line_number}: {key}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("Annotations contain no questions")
    return rows


def validate_annotations(rows, cache, fetch):
    """Validate EVERY question before running any diagnostic or creating output."""
    from .evidence_audit import validate_annotation
    known = _records(cache)
    splits = {eid: split for split in ("train", "val") for eid in cache.manifest["splits"][split]}
    errors, validated = [], []
    for index, row in enumerate(rows, 1):
        eid = row["episode_id"]
        try:
            if eid not in known or eid not in splits:
                raise ValueError("Episode is not in the selected cache")
            validated.append(validate_annotation(row, fetch(eid),
                cache_fingerprint=cache.manifest["fingerprint"], split=splits[eid]))
        except (ValueError, KeyError, TypeError, IndexError) as error:
            errors.append(f"row {index} episode={eid} decision={row['decision']}: {error}")
    if errors:
        raise ValueError("Rejected annotations; no output or metrics created:\n" + "\n".join(errors))
    return validated, splits


def audit_preflight(args):
    if not args.top_k or len(set(args.top_k)) != len(args.top_k) or any(k < 1 for k in args.top_k):
        raise ValueError("--top-k must contain unique positive integers")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and explicitly selected CUDA devices are supported")
    cache = EpisodeCache(resolve(args.cache_dir))
    output = _output_path(args, cache)
    annotations_path, checkpoint = resolve(args.annotations), resolve(args.checkpoint)
    annotations_sha256 = file_sha256(annotations_path)
    rows = load_annotations(annotations_path)
    hashes = _hash_inputs(cache, [row["episode_id"] for row in rows], extra=[annotations_path])
    if hashes[str(annotations_path)] != annotations_sha256:
        raise RuntimeError("Annotations changed while being read; retry with a stable review file")
    fetch = MappedEpisodes(cache).fetch
    rows, splits = validate_annotations(rows, cache, fetch)
    validate_cache_checkpoint(cache.manifest)
    base = Path(cache.manifest["model_path"]).resolve()
    preliminary = _json(checkpoint / "checkpoint.json")
    is_v5 = preliminary.get("config", {}).get("training_recipe") == RECIPE
    check = v5_checkpoint_info if is_v5 else v4_checkpoint_info
    info = check(base, checkpoint)
    if info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]:
        raise ValueError("Checkpoint and cache fingerprints differ")
    if args.writer_policy == "hard" and info["config"]["stage"] != 2:
        raise ValueError("hard requires a trained Stage-2 writer; Stage-1 uses --writer-policy all (FIFO)")
    mode = info["config"].get("train", {}).get("reader_mode", info["config"].get("reader_mode"))
    if mode != "memory":
        raise ValueError("Cannot audit an intentionally disabled reader as an active memory model")
    cfg = MemoryV3Config(**info["config"]["memory"])
    for name in ("feature_dim", "state_dim", "action_dim"):
        if getattr(cfg, name) != cache.manifest[name]:
            raise ValueError(f"Checkpoint/cache {name} mismatch")
    window = int(_json(base / "config.json")["memory_window"])
    files = [checkpoint / name for name in ("checkpoint.json", "model.safetensors", "expert.safetensors")]
    if is_v5:
        files.append(checkpoint / "recall.safetensors")
    hashes.update({str(path): file_sha256(path) for path in files})
    for name, field in (("model.safetensors", "memory_sha256"), ("expert.safetensors", "expert_sha256")):
        if hashes[str(checkpoint / name)] != info["metadata"][field]:
            raise RuntimeError(f"Checkpoint changed during validation: {name}")
    if is_v5 and hashes[str(checkpoint / "recall.safetensors")] != info["metadata"]["recall_sha256"]:
        raise RuntimeError("Recall checkpoint changed during validation")
    _check_unchanged(hashes)
    provenance = {"variant": VARIANT, "cache_fingerprint": cache.manifest["fingerprint"],
                  "checkpoint": str(checkpoint), "checkpoint_step": info["step"], "checkpoint_stage": info["config"]["stage"],
                  "recipe": info["config"].get("training_recipe", "action_expert_v4"),
                  "base_identity": info["metadata"]["base_model"],
                  "base_identity_limit": "Base weights checked by recorded size/mtime; cache features and selected bundle files SHA256-hashed.",
                  "source_sha256": hashes, "code_sha256": _code_hashes(), "memory_window": window,
                  "writer_policy": args.writer_policy, "top_k": args.top_k, "device": str(device),
                  "torch": str(torch.__version__), "annotations": str(annotations_path), "queries": len(rows),
                  "scope": "Cached demonstration memory replay only. No VLM, Action Expert, training, simulator, or robot-success metric."}
    return cache, fetch, rows, splits, info, cfg, window, output, provenance


def audit(args):
    from safetensors.torch import load_file
    from .evidence_audit import audit_query, summarize_audit
    cache, fetch, rows, splits, info, cfg, window, output, provenance = audit_preflight(args)
    if args.preflight_only:
        print(f"[evidence][preflight] Validated {len(rows)} questions and Stage-{info['config']['stage']} bundle. No output, model, training or simulation started.", flush=True)
        return provenance
    # Only the small deployed memory module is instantiated; AE weights were
    # validated as bundle provenance above, never installed in a full expert.
    memory = ActionValueMemory(cfg)
    memory.load_state_dict(load_file(str(resolve(args.checkpoint) / "model.safetensors"), device="cpu"), strict=True)
    memory.to(args.device).requires_grad_(False).eval()
    results = []
    with torch.no_grad():
        for index, row in enumerate(sorted(rows, key=lambda r: (r["episode_id"], r["decision"])), 1):
            eid = row["episode_id"]
            results.append(audit_query(memory, fetch(eid), row,
                cache_fingerprint=cache.manifest["fingerprint"], split=splits[eid],
                writer_policy=args.writer_policy, memory_window=window, top_k=tuple(args.top_k)))
            print(f"[evidence] {index}/{len(rows)} episode={eid} decision={row['decision']} status={row['status']}", flush=True)
    summary = summarize_audit(results)
    _check_unchanged(provenance["source_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "results.jsonl", results)
    _write_json(output / "summary.json", summary)
    _write_json(output / "provenance.json", provenance)
    report = ("# Cached past-evidence diagnostic\n\n"
              "This is NOT RoboMME success rate. Unknown annotations have no evidence accuracy denominator.\n"
              "Endpoint coverage is a sampling proxy, not proof that features encode the cue; attention is not causal use.\n\n"
              "```json\n" + json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n```\n")
    (output / "report.md").write_text(report, encoding="utf-8")
    print(f"[evidence] Saved: {output / 'report.md'}", flush=True)
    return summary


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return prepare(args) if args.command == "prepare" else audit(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"[evidence] ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
