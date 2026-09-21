#!/usr/bin/env python3
"""Bounded CPU decoders of frozen V19 stages, not policy or retrieval training.

Current online subgoals are privileged targets at the current action decision.
They never enter extraction. Fixed random projections equalize decoder width;
their information bottleneck and limited decoder mean a negative result cannot
establish absence of information. Episode splits and final budgets are fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gr00t.long_memory.cache import decision_frames
from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.checkpoint_representation_v18 import checkpoint_info_v18
from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18, RepresentationMemoryV18
from run_scripts.robomme.prepare_segment_targets_v15 import GOALS, DIRECTIONS

STAGES = ("P0", "Ptime", "P1", "P2", "P3", "P4", "P5")
OBSERVATIONS = ("short", "state", "frames", "is_demo")
SEMANTICS = {
    "P0": "current short and current state",
    "Ptime": "causal current clocks/progress only",
    "P1": "current short/state plus all past short/state/time",
    "P2": "current short/state plus all past encoded events before FIFO",
    "P3": "current short/state plus actual FIFO before current write",
    "P4": "current short/state plus actual attention READ before fusion",
    "P5": "actual fused short and current state",
}


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def tensor_hash(values):
    value = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        tensor = tensor.detach().cpu().contiguous()
        value.update(name.encode())
        value.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        value.update(tensor.view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def align_targets(raw, episode, eid, split, *, stride=16):
    """Current labels only; preserve partial last chunks and annotation changes."""
    n = len(raw["frame_index"])
    if (raw["frame_index"] != list(range(n)) or raw["episode_index"] != [eid] * n
            or any(type(v) is not bool for v in raw["is_demo"])):
        raise ValueError("Raw episode/frame/demo identity differs")
    frames = episode["frames"].tolist()
    if frames != decision_frames(np.asarray(raw["is_demo"], dtype=bool), stride).tolist():
        raise ValueError("Canonical observation alignment differs")
    if (episode["is_demo"].tolist() != [raw["is_demo"][f] for f in frames]
            or len(episode["decision_mask"]) != len(frames) - 1):
        raise ValueError("Cache demo/decision alignment differs")
    result = []
    for d in torch.where(episode["decision_mask"])[0].tolist():
        f = frames[d]
        if raw["is_demo"][f] or not bool((episode["target_mask"][d, :stride]
                & episode["action_mask"][d, :stride, None]).any()):
            raise ValueError("Active decision lacks current action supervision")
        label = raw["simple_subgoal_online"][f]
        if label not in DIRECTIONS:
            raise ValueError(f"Unexpected current PatternLock label: {label!r}")
        result.append({"episode_id": eid, "decision": d, "query_frame": f,
                       "split": split, "label": label})
    return result


def prepare_plan(cache_dir, *, train_episodes=32, val_episodes=16, seed=9191):
    import pyarrow.parquet as pq
    cache_dir = Path(cache_dir).resolve()
    cache = json.loads((cache_dir / "manifest.json").read_text())
    dataset = Path(cache["dataset_path"]).resolve()
    info = json.loads((dataset / "meta/info.json").read_text())
    records = {r["episode_id"]: r for r in cache["episodes"] if r["task"] in GOALS}
    if set(cache["splits"]["train"]) & set(cache["splits"]["val"]):
        raise ValueError("TRAIN/VAL overlap")
    if not 1 <= train_episodes <= 32 or not 1 <= val_episodes <= 16:
        raise ValueError("Probe budget is TRAIN <=32 and cache-VAL <=16 episodes")
    episodes, rows, selected, hashes, schemas = {}, [], {}, {}, []
    for offset, (split, limit) in enumerate((("train", train_episodes), ("val", val_episodes))):
        candidates = sorted(set(records) & set(cache["splits"][split]))
        random.Random(seed + offset).shuffle(candidates)
        selected[split] = candidates[:limit]
        if len(selected[split]) != limit:
            raise ValueError("Insufficient episodes for fixed plan")
        for eid in selected[split]:
            record = records[eid]
            if record["split"] != split:
                raise ValueError("Manifest record split differs")
            path = (cache_dir / record["path"]).resolve()
            if not path.is_relative_to(cache_dir):
                raise ValueError("Cache episode path escapes root")
            episode = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if episode["cache_fingerprint"] != cache["fingerprint"] or episode["episode_id"] != eid:
                raise ValueError("Cache payload identity differs")
            parquet = (dataset / info["data_path"].format(episode_chunk=eid // info["chunks_size"], episode_index=eid)).resolve()
            if not parquet.is_relative_to(dataset):
                raise ValueError("Raw episode path escapes root")
            names = pq.read_schema(parquet).names
            requested = ("simple_subgoal", "simple_subgoal_online", "grounded_subgoal",
                "grounded_subgoal_online", "is_demo", "episode_index", "frame_index",
                "is_subgoal_boundary", "choice_action")
            schemas.append({"episode_id": eid, "path": str(parquet), "columns": names,
                            "requested_column_availability": {name: name in names for name in requested}})
            columns = ("episode_index", "frame_index", "is_demo", "simple_subgoal_online")
            raw = pq.read_table(parquet, columns=list(columns)).to_pydict()
            rows.extend(align_targets(raw, episode, eid, split, stride=cache["action_steps"]))
            episodes[eid] = {k: episode[k] for k in OBSERVATIONS}
            hashes.update({str(path): file_hash(path), str(parquet): file_hash(parquet)})
    vocabulary = sorted({r["label"] for r in rows if r["split"] == "train"})
    for row in rows:
        row["class_id"] = vocabulary.index(row["label"]) if row["label"] in vocabulary else -1
    for path in (cache_dir / "manifest.json", dataset / "meta/info.json", dataset / "meta/tasks.jsonl"):
        hashes[str(path)] = file_hash(path)
    plan = {"selected_episode_ids": selected, "selection_seeds": {"train": seed, "val": seed + 1},
            "rows": rows, "vocabulary": vocabulary, "cache_fingerprint": cache["fingerprint"],
            "dataset_path": str(dataset), "files_sha256": hashes,
            "raw_parquet_schema_audit": schemas,
            "coverage": {s: {"episodes": len(selected[s]), "queries": sum(r["split"] == s for r in rows),
                            "unknown_classes": sum(r["split"] == s and r["class_id"] < 0 for r in rows),
                            "class_counts": dict(Counter(r["label"] for r in rows if r["split"] == s))}
                         for s in ("train", "val")}}
    return plan, episodes, cache


class FrozenProjections:
    """Shared seeded Gaussian maps, never fitted on either split or labels."""
    def __init__(self, config, seed=192019):
        generator = torch.Generator().manual_seed(seed)
        self.matrices = {name: torch.randn(width, out, generator=generator) / width ** .5
                         for name, width, out in (("feature", config.feature_dim, 64),
                             ("encoded", config.hidden_dim, 64), ("state", config.state_dim, 16))}
        self.seed = seed

    def __call__(self, name, x):
        return x.float() @ self.matrices[name]


def clock_features(frames, demos, query_frame, first_execution):
    z = frames.float() / 16
    elapsed = (frames.float() - first_execution).clamp_min(0) / 16
    age = (query_frame - frames.float()) / 16
    if bool((age < 0).any()):
        raise ValueError("Future feature frame")
    return torch.stack((z, torch.log1p(z), z.sin(), z.cos(), elapsed,
        torch.log1p(elapsed), elapsed.sin(), elapsed.cos(), age,
        torch.log1p(age), demos.float()), dim=-1)


@torch.no_grad()
def extract_stages(core, observations, decision, projections):
    """No target arguments or action access. Prefix ends at the current query."""
    if core.device.type != "cpu" or any(p.requires_grad for p in core.parameters()):
        raise ValueError("Extraction requires a fully frozen CPU policy")
    # Slice each allowed column before replay; future values are not inspected.
    prefix = {k: observations[k][:decision + 1] for k in OBSERVATIONS}
    frames, demos = prefix["frames"], prefix["is_demo"]
    if bool(demos[-1]):
        raise ValueError("Probe requires a current action observation")
    qframe = int(frames[-1])
    first_execution = int(frames[torch.where(~demos)[0][0]])
    encoded = core.encode_prefix(prefix, decision + 1)
    bank = core.initial_bank()
    for index in range(decision):
        bank = core.write_fifo(bank, encoded["stored"][index:index + 1])
    short, query = encoded["short"][-1:], encoded["query"][-1:]
    captured = []
    handle = core.memory.attention.register_forward_hook(lambda module, inputs, output: captured.append(output[0].detach().clone()))
    try:
        fused, _ = core.read_from_bank(short, query, bank)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Expected one actual nonempty READ")
    recalled = captured[0]
    tokens = core.config.num_short_tokens

    def pack(content, source, event_ids, *, states=None, current=False):
        ids = torch.as_tensor(event_ids, dtype=torch.long)
        count = len(ids)
        content = projections(source, content.reshape(count, tokens, -1))
        state = torch.zeros(count, 16) if states is None else projections("state", states)
        clocks = clock_features(frames[ids], demos[ids], qframe, first_execution)
        qids = torch.eye(tokens)[None].expand(count, -1, -1)
        role = torch.tensor([float(current), float(not current)])[None, None].expand(count, tokens, -1)
        return torch.cat((content, state[:, None].expand(-1, tokens, -1),
            clocks[:, None].expand(-1, tokens, -1), qids, role), dim=-1).reshape(count * tokens, -1)

    current = pack(short, "feature", [decision], states=prefix["state"][-1:], current=True)
    historical = list(range(decision))
    fifo_start = max(0, decision - core.config.capacity_events)
    past_short = pack(prefix["short"][:-1], "feature", historical, states=prefix["state"][:-1])
    pre_fifo = pack(encoded["stored"][:-1], "encoded", historical)
    fifo = pack(bank.reshape(-1, tokens, core.config.hidden_dim), "encoded", list(range(fifo_start, decision)))
    read = pack(recalled, "encoded", [decision])
    time_only = current.clone()
    time_only[:, :80] = 0  # Neither short nor state is available to this baseline.
    current[:, 80:91] = 0  # P0 has short/state only; Ptime is the explicit clock control.
    read[:, 80:91] = 0  # The READ itself already reflects the policy's time-conditioned query.
    fused_current = pack(fused, "feature", [decision], states=prefix["state"][-1:], current=True)
    fused_current[:, 80:91] = 0
    result = {"P0": current, "Ptime": time_only,
              "P1": torch.cat((current, past_short)), "P2": torch.cat((current, pre_fifo)),
              "P3": torch.cat((current, fifo)), "P4": torch.cat((current, read)),
              "P5": fused_current}
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite frozen features")
    return result, {"past_events": decision, "fifo_events": decision - fifo_start,
                   "fifo_evicted": fifo_start, "p2_p3_equal": torch.equal(result["P2"], result["P3"]),
                   "query_frame": qframe, "history_latest_frame": int(frames[-2])}


class Decoder(nn.Module):
    """Same small current-conditioned attention decoder for every stage."""
    def __init__(self, width, classes, hidden=64, current_tokens=4):
        super().__init__()
        self.current_tokens = current_tokens
        self.query = nn.Linear(width, hidden)
        self.key = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.output = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, classes))

    def forward(self, features, valid):
        # Clean padding before projection so even nonfinite masked payloads cannot leak.
        features = features.masked_fill(~valid[..., None], 0)
        current = self.query(features[:, :self.current_tokens].mean(1))
        key, value = self.key(features), self.value(features)
        score = torch.einsum("bd,btd->bt", current, key) / key.shape[-1] ** .5
        score = score.masked_fill(~valid, float("-inf"))
        read = torch.einsum("bt,btd->bd", score.softmax(-1), value)
        return self.output(torch.cat((current, read), dim=-1))


def standardize_and_pad(features, rows):
    selected = torch.cat([x for x, row in zip(features, rows, strict=True) if row["split"] == "train"])
    mean, scale = selected.mean(0), selected.std(0, unbiased=False).clamp_min(1e-3)
    width, count = features[0].shape[-1], max(len(x) for x in features)
    result, mask = torch.zeros(len(features), count, width), torch.zeros(len(features), count, dtype=torch.bool)
    for i, x in enumerate(features):
        result[i, :len(x)] = (x - mean) / scale
        mask[i, :len(x)] = True
    return result, mask, mean, scale


def fit_decoders(features, rows, classes, *, steps=300, seeds=(91951, 91952), batch_size=32, lr=1e-3):
    train = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "train" and row["class_id"] >= 0])
    labels = torch.tensor([row["class_id"] for row in rows])
    trained, predictions, reports, schedules = {}, [], {}, {}
    prepared = {stage: standardize_and_pad(features[stage], rows) for stage in STAGES}
    for seed in seeds:
        rng = torch.Generator().manual_seed(seed + 1000)
        schedule = train[torch.randint(len(train), (steps, batch_size), generator=rng)]
        schedules[str(seed)] = digest(schedule.tolist())
        for stage in STAGES:
            x, mask, mean, scale = prepared[stage]
            torch.manual_seed(seed)
            model = Decoder(x.shape[-1], classes)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
            losses = []
            for step, indices in enumerate(schedule):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x[indices], mask[indices]), labels[indices])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite decoder loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
                    losses.append({"step": step + 1, "train_classification_loss": float(loss.detach())})
            model.eval()
            with torch.no_grad():
                logits = torch.cat([model(x[i:i + 64], mask[i:i + 64]) for i in range(0, len(x), 64)])
                probability, predicted = logits.softmax(-1), logits.argmax(-1)
            key = f"{stage}/seed{seed}"
            trained[key] = {"model": model.state_dict(), "mean": mean, "scale": scale}
            reports[key] = {"parameters": sum(p.numel() for p in model.parameters()), "loss_history": losses}
            for i, row in enumerate(rows):
                target = row["class_id"]
                predictions.append({**row, "stage": stage, "seed": seed, "prediction": int(predicted[i]),
                    "correct": bool(predicted[i] == target) if target >= 0 else None,
                    "confidence": float(probability[i].max()),
                    "target_probability": float(probability[i, target]) if target >= 0 else None})
            print(json.dumps({"completed": key, "last_train_loss": losses[-1]["train_classification_loss"]}), flush=True)
    if len({v["parameters"] for v in reports.values()}) != 1:
        raise ValueError("Decoder parameter counts differ")
    return trained, predictions, reports, schedules


def summarize(predictions, *, bootstrap_samples=5000, seed=192020):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in predictions:
        if row["correct"] is not None:
            grouped[(row["stage"], row["split"], row["seed"])][row["episode_id"]].append(float(row["correct"]))
    summaries, episodes = {}, {}
    for stage in STAGES:
        summaries[stage] = {}
        for split in ("train", "val"):
            per_seed = {str(s): {str(eid): float(np.mean(values)) for eid, values in groups.items()}
                        for (name, part, s), groups in grouped.items() if name == stage and part == split}
            ids = sorted({int(eid) for values in per_seed.values() for eid in values})
            macro = {eid: float(np.mean([values[str(eid)] for values in per_seed.values()])) for eid in ids}
            summaries[stage][split] = {"episode_macro_accuracy": float(np.mean(list(macro.values()))),
                "per_seed_macro_accuracy": {s: float(np.mean(list(values.values()))) for s, values in per_seed.items()},
                "episodes": len(ids), "episode_accuracy_seed_mean": macro}
            episodes[(stage, split)] = macro
    contrasts = {}
    for other, base in ((s, b) for s in STAGES for b in ("P0", "Ptime") if s != b):
        a, b = episodes[(other, "val")], episodes[(base, "val")]
        if set(a) != set(b):
            raise ValueError("Paired held-out episode sets differ")
        delta = np.asarray([a[eid] - b[eid] for eid in sorted(a)])
        rng = np.random.default_rng(seed)
        means = delta[rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))].mean(1)
        contrasts[f"{other}_minus_{base}"] = {"accuracy_gain": float(delta.mean()),
            "paired_episode_bootstrap_95_ci": np.quantile(means, [.025, .975]).tolist()}
    return {"stages": summaries, "paired_val_contrasts": contrasts,
            "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
            "estimand": "Mean of two fixed decoder seeds; bootstrap unit is held-out episode"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="runs/long_memory/cache_full1600_v1")
    parser.add_argument("--checkpoint", default="runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.threads < 1:
        raise ValueError("Positive fixed steps and thread count required")
    if torch.cuda.is_initialized():
        raise ValueError("CPU diagnostic must not initialize CUDA")
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a NEW diagnostic output directory")
    plan, episodes, cache = prepare_plan(args.cache_dir)
    checkpoint = Path(args.checkpoint).resolve()
    info = checkpoint_info_v18(cache["model_path"], checkpoint)
    cfg = RepresentationConfigV18(**info["config"]["representation"])
    if cfg.representation != "short" or cfg.num_short_tokens != 4 or info["metadata"]["cache_fingerprint"] != cache["fingerprint"]:
        raise ValueError("Require matching frozen short/FIFO checkpoint and cache")
    validate_output_scope(output, Path(args.cache_dir), checkpoint, cache["dataset_path"], cache["model_path"])
    from safetensors.torch import load_file
    core = RepresentationMemoryV18(cfg).eval().requires_grad_(False)
    core.load_delta_state_dict(load_file(str(checkpoint / "model.safetensors"), device="cpu"))
    model_before = tensor_hash(core.state_dict())
    projections = FrozenProjections(cfg)
    signatures = dict(plan["files_sha256"])
    prior_paths = [ROOT / name for name in (
        "docs/LONG_MEMORY_V15_SEGMENT_RETRIEVAL.md", "docs/LONG_MEMORY_V16_PROJECTION_PROBE.md",
        "runs/long_memory/v15_segment_probe256_20260917/assessment.json",
        "runs/long_memory/v16_projection_pair256_20260917/comparison.json")]
    prior_contents = {str(path): path.read_text() for path in prior_paths}
    for path in (Path(__file__), ROOT / "run_scripts/robomme/representation_core_v18.py",
            ROOT / "gr00t/long_memory/recurrent_v7.py", ROOT / "run_scripts/robomme/prepare_segment_targets_v15.py",
            ROOT / "run_scripts/robomme/checkpoint_representation_v18.py",
            checkpoint / "checkpoint.json", checkpoint / "model.safetensors", checkpoint / "expert.safetensors",
            *prior_paths):
        signatures[str(path.resolve())] = file_hash(path)
    protocol = {"kind": "frozen_v19_current_subgoal_diagnostic", "deployable_policy": False,
        "checkpoint": str(checkpoint), "checkpoint_step": info["step"], "config": asdict(cfg),
        "steps": args.steps, "seeds": [91951, 91952], "batch_size": 32, "learning_rate": 1e-3,
        "optimizer": "AdamW", "weight_decay": 1e-3, "clip_grad_norm": 1.,
        "projection_seed": projections.seed, "projection_sha256": tensor_hash(projections.matrices),
        "projection_shapes": {k: list(v.shape) for k, v in projections.matrices.items()},
        "normalization": "TRAIN tokens only, independently for each representation",
        "clock_inputs": "Ptime explicit current clocks; P1/P2/P3 explicit historical clocks; P0/P5 and common current context/P4 READ have no appended clocks",
        "decoder": "current-conditioned single attention read, 64 hidden, 64 hidden MLP",
        "stages": SEMANTICS, "selection": "fixed final; no VAL model or budget selection",
        "target": "simple_subgoal_online at frames[decision], current nonterminal action decision",
        "prior_experiments": {"files_read": list(prior_contents),
            "v15_decision": json.loads(prior_contents[str(prior_paths[2])])["decision"],
            "v16_decision": json.loads(prior_contents[str(prior_paths[3])])["decision"],
            "difference": "V15/V16 trained Q/K or Q/K/P on selected weak demo-run retrieval targets. This fits separate decoders of a frozen V19 policy to all selected current online-subgoal action decisions; no retrieval NLL, cue labels or policy updates."},
        "files_sha256": signatures, "policy_sha256_before": model_before,
        "limitations": ["Fixed projections may discard relevant information; a negative probe does not show its absence.",
            "Small decoder optimization/capacity limits remain; equal parameter counts are not equal information content.",
            "Current subgoal labels are privileged supervision, not verified cue truth or memory-dependence labels.",
            "These cache-VAL episodes were used in earlier exploratory studies; no new independent confirmation.",
            "Classification accuracy is not robot success or generated-action quality.",
            "Causal positional features preserve time/token identity, but the decoder has limited ability to reason about order."]}
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "protocol.json", protocol)
    _atomic_json(output / "target_plan.json", plan)
    features, audits = {stage: [] for stage in STAGES}, []
    for row in plan["rows"]:
        values, audit = extract_stages(core, episodes[row["episode_id"]], row["decision"], projections)
        audits.append({**row, **audit})
        for stage in STAGES:
            features[stage].append(values[stage])
    _atomic_json(output / "extraction_audit.json", audits)
    trained, predictions, training, schedules = fit_decoders(features, plan["rows"], len(plan["vocabulary"]), steps=args.steps)
    report = summarize(predictions)
    model_after = tensor_hash(core.state_dict())
    if model_after != model_before or any(p.grad is not None or p.requires_grad for p in core.parameters()):
        raise ValueError("Frozen policy changed or acquired gradients")
    if any(file_hash(path) != expected for path, expected in signatures.items()) or torch.cuda.is_initialized():
        raise ValueError("Protected input/source changed or CUDA initialized")
    checks = {"policy_sha256_before": model_before, "policy_sha256_after": model_after,
        "policy_unchanged_and_no_grad": True, "input_source_hashes_unchanged": True, "cuda_initialized": False,
        "decoder_parameters_each": next(iter(training.values()))["parameters"],
        "same_train_schedule_hashes_each_stage": schedules,
        "val_p2_p3_inputs_all_equal": all(r["p2_p3_equal"] for r in audits if r["split"] == "val"),
        "fifo_eviction_queries": {s: sum(r["fifo_evicted"] > 0 for r in audits if r["split"] == s) for s in ("train", "val")},
        "elapsed_seconds": time.monotonic() - started}
    report.update(coverage=plan["coverage"], checks=checks, limitations=protocol["limitations"])
    report["limitations"].append("P2 and P3 are identical at every selected VAL input: FIFO eviction is not tested on held-out queries.")
    torch.save({"decoders": trained, "projections": projections.matrices}, output / "diagnostic_decoders.pt")
    _atomic_json(output / "training.json", training)
    _atomic_json(output / "predictions.json", predictions)
    _atomic_json(output / "summary.json", report)
    _atomic_json(output / "status.json", {"status": "complete", "policy_training": False, **checks})
    print(json.dumps({"output": str(output), "checks": checks,
        "val_accuracy": {s: report["stages"][s]["val"]["episode_macro_accuracy"] for s in STAGES}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
