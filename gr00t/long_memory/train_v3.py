"""Two-stage action-value memory, separate from all legacy trainers.

Stage 1 trains typed-event retrieval/fusion with the original frozen HAMLET
action loss, then optionally confidence-screened query-specific relevance.
Stage 2 trains a bounded KEEP/APPEND/REPLACE chooser with the conditional
expected future action loss; after a short writer bootstrap the reader also
continues learning. This is not full-episode BPTT or environment reward RL.

Only the new memory weights/optimizer/RNG are saved. Source/cache provenance,
immutable teacher snapshots, paired validation plans and JSONL journals make
resume and subsequent RoboMME comparisons auditable.
"""
import argparse
import copy
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

from .cache import EpisodeCache
from .cache_reader_v3 import MappedEpisodes, validate_decision
from .contextual_cvom import eligible_candidates
from .core_v3 import ActionValueMemory, MemoryV3Config
from .hamlet import checkpoint_identity, episode_flow_loss, load_frozen_hamlet, validate_cache_checkpoint
from .monitoring import RunLogger, _atomic_json, load_checkpoint, save_checkpoint
from .objectives_v3 import (ActionValueLabelsV3, LabelV3Config, retrieval_loss, storage_loss)
from .replay_v3 import encode_until, read_bank, replay_bank

VARIANT = "action_value_v3"


class _Tee:
    """Keep terminal output and full tracebacks alongside scalar journals."""
    def __init__(self, terminal, journal):
        self.terminal, self.journal = terminal, journal

    def write(self, text):
        self.terminal.write(text)
        self.journal.write(text)
        self.journal.flush()
        return len(text)

    def flush(self):
        self.terminal.flush(); self.journal.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, choices=(1, 2))
    p.add_argument("--cache-dir")
    p.add_argument("--base-model")
    p.add_argument("--output-dir", required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--init-checkpoint", help="Stage 2 requires a v3 Stage-1 checkpoint, not legacy weights")
    source.add_argument("--resume", help="Inherit saved options/RNG; max-steps is TOTAL updates")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--reader-learning-rate", type=float, default=1e-4)
    p.add_argument("--writer-learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--capacity", type=int)
    p.add_argument("--hidden-dim", type=int)
    p.add_argument("--num-heads", type=int)
    p.add_argument("--min-fill", type=int)
    p.add_argument("--max-victims", type=int)
    p.add_argument("--ranking-weight", type=float, default=0.01)
    p.add_argument("--ranking-start", type=int, default=200, help="No self-teacher ranking before reader warm-up")
    p.add_argument("--ranking-every", type=int, default=8)
    p.add_argument("--writer-weight", type=float, default=0.1)
    p.add_argument("--writer-bootstrap", type=int, default=100, help="Stage-2 updates with reader frozen")
    p.add_argument("--writer-every", type=int, default=4)
    p.add_argument("--writer-batch-size", type=int, default=4)
    p.add_argument("--policy-ramp-steps", type=int, default=200)
    p.add_argument("--teacher-refresh-steps", type=int, default=250)
    p.add_argument("--future-samples", type=int, default=2)
    p.add_argument("--noise-samples", type=int, default=2)
    p.add_argument("--label-scale", type=float, default=0.001)
    p.add_argument("--label-margin", type=float, default=1e-5)
    p.add_argument("--uncertainty-z", type=float, default=2.0)
    p.add_argument("--retrieval-candidates", type=int, default=4)
    p.add_argument("--storage-contexts", type=int, default=128, help="Episode-balanced pool refreshed per teacher; causal bank recomputed on use")
    p.add_argument("--val-samples", type=int, default=16)
    p.add_argument("--val-storage-samples", type=int, default=4)
    p.add_argument("--max-train-episodes", type=int, default=0, help="0=entire existing train split; smoke-only restriction")
    p.add_argument("--max-val-episodes", type=int, default=0, help="0=entire existing validation split")
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--plot-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--preflight-only", action="store_true", help="Metadata checks only; no model, training or output writes")
    return p


def _info(path):
    info = json.loads((Path(path) / "checkpoint.json").read_text())
    if info.get("format_version") != 1 or info["config"].get("trainer_variant") != VARIANT:
        raise ValueError("This entrypoint requires action_value_v3 weights; legacy v1/v2 remain separate")
    return info


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    if args.resume:
        saved = _info(args.resume)["config"]["train"]
        explicit = {p._option_string_actions[token.split("=", 1)[0]].dest
                    for token in argv if token.split("=", 1)[0] in p._option_string_actions}
        for name, value in saved.items():
            if hasattr(args, name) and name not in explicit:
                setattr(args, name, value)
        args.init_checkpoint = None
        args.resume = p.parse_args(argv).resume
    return args


def validate_args(args):
    if args.stage not in (1, 2) or not args.cache_dir:
        raise ValueError("Provide --stage and --cache-dir, or inherit them with --resume")
    if args.stage == 1 and args.init_checkpoint:
        raise ValueError("v3 Stage 1 starts from the frozen base/cache; legacy memory cannot initialize it")
    if args.stage == 2 and not (args.init_checkpoint or args.resume):
        raise ValueError("Stage 2 requires --init-checkpoint from v3 Stage 1")
    for name in ("max_steps", "grad_accum", "ranking_every", "writer_every", "writer_batch_size",
                 "policy_ramp_steps", "teacher_refresh_steps", "future_samples", "noise_samples",
                 "retrieval_candidates", "storage_contexts", "val_samples", "val_storage_samples",
                 "eval_steps", "log_steps", "plot_steps", "save_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("seed", "warmup_steps", "ranking_start", "writer_bootstrap", "max_train_episodes", "max_val_episodes"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in ("reader_learning_rate", "writer_learning_rate", "max_grad_norm", "label_scale"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "ranking_weight", "writer_weight", "label_margin", "uncertainty_z"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.noise_samples < 2:
        raise ValueError("At least two paired noise draws are required for uncertainty diagnostics")
    if args.stage == 2 and args.writer_weight <= 0:
        raise ValueError("Stage 2 requires positive writer-weight; Stage 1 is the FIFO control")


def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "gr00t/long_memory").glob("*.py"))
    paths += [root / "run_scripts/robomme/train_long_memory_v3.py"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def runtime_identity():
    """Resume is exact only in a matching software/math environment."""
    return {"torch": str(torch.__version__), "numpy": str(np.__version__), "cuda_build": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}


def preflight(args):
    validate_args(args)
    cache = EpisodeCache(args.cache_dir)
    manifest = cache.manifest
    base = str(Path(args.base_model or manifest["model_path"]).resolve())
    if base != str(Path(manifest["model_path"]).resolve()):
        raise ValueError("Base differs from frozen-feature cache provenance")
    validate_cache_checkpoint(manifest)
    identity = checkpoint_identity(base)
    base_cfg = json.loads((Path(base) / "config.json").read_text())
    window, stride = int(base_cfg["memory_window"]), int(base_cfg.get("memory_stride", 16))
    info = _info(args.resume or args.init_checkpoint) if (args.resume or args.init_checkpoint) else None
    if info:
        if info["metadata"]["base_model"] != identity or info["metadata"]["cache_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Initial/resume checkpoint base/cache differs")
        if args.resume and info["config"]["stage"] != args.stage:
            raise ValueError("Resume cannot change stage")
        if args.init_checkpoint and info["config"]["stage"] != 1:
            raise ValueError("Stage 2 must initialize from a v3 Stage-1 reader")
        if not (Path(args.resume or args.init_checkpoint) / "model.safetensors").is_file():
            raise FileNotFoundError("Memory weights missing")
        cfg = MemoryV3Config(**info["config"]["memory"])
        for name in ("capacity", "hidden_dim", "num_heads", "min_fill", "max_victims"):
            if getattr(args, name) is not None and getattr(args, name) != getattr(cfg, name):
                raise ValueError(f"{name} must match the initial/resume architecture")
    else:
        options = {n: getattr(args, n) for n in ("capacity", "hidden_dim", "num_heads", "min_fill", "max_victims")
                   if getattr(args, n) is not None}
        cfg = MemoryV3Config(**{n: manifest[n] for n in ("feature_dim", "state_dim", "action_dim")},
                             time_scale=float(stride), **options)
    if not 1 <= cfg.min_fill < cfg.capacity:
        raise ValueError("Require 1 <= min-fill < capacity, so learned choices have opportunities")
    for name in ("feature_dim", "state_dim", "action_dim"):
        if getattr(cfg, name) != manifest[name]:
            raise ValueError(f"Cache/memory dimension mismatch: {name}")
    if float(cfg.time_scale) != float(stride):
        raise ValueError("Time scale must match the cached/inference memory stride")
    train, val = manifest["splits"]["train"], manifest["splits"]["val"]
    if not train or not val or set(train) & set(val):
        raise ValueError("Require nonempty disjoint episode train/validation splits")
    if args.resume:
        if args.max_steps <= info["step"] or not (Path(args.resume) / "training_state.pt").is_file():
            raise ValueError("Resume needs optimizer/RNG and max-steps greater than the saved total step")
        mutable = {"max_steps", "resume", "init_checkpoint", "output_dir", "base_model", "preflight_only"}
        for name, value in vars(args).items():
            if name not in mutable and info["config"]["train"].get(name) != value:
                raise ValueError(f"Exact resume option changed: {name}")
        if info["metadata"].get("source_sha256") != source_identity():
            raise ValueError("Training source changed after checkpoint; do not silently resume changed logic")
        if info["metadata"].get("runtime") != runtime_identity():
            raise ValueError("Runtime/math settings changed; exact resume requires the recorded environment")
    return cache, info, cfg, base, identity, window


def make_plans(fetch, manifest, cfg, window, args):
    """Episode-balanced train sampling; fixed normal + delayed validation plans."""
    rng = random.Random(args.seed + 1907)
    train = list(manifest["splits"]["train"])
    val = list(manifest["splits"]["val"])
    if args.max_train_episodes:
        train = rng.sample(train, min(args.max_train_episodes, len(train)))
    if args.max_val_episodes:
        val = rng.sample(val, min(args.max_val_episodes, len(val)))
    actions, storage = {"train": [], "val": []}, {"train": [], "val": []}
    for split, ids in (("train", train), ("val", val)):
        for index, eid in enumerate(ids):
            ep = fetch(eid)
            decisions = torch.where(ep["decision_mask"])[0].tolist()
            if split == "train":
                decisions = [d for d in decisions if bool(ep["transition_valid"][:d].any())]
            if decisions:
                actions[split].append([int(eid), decisions])
            if args.stage == 2:
                candidates = [c for c in eligible_candidates(ep, window)
                              if int(ep["transition_valid"][:c].sum()) >= cfg.min_fill]
                if candidates:
                    storage[split].append([int(eid), candidates])
            if (index + 1) % 100 == 0:
                print(f"[v3][index] {split} {index+1}/{len(ids)} (no full VL scan)", flush=True)
    if not actions["train"] or not actions["val"]:
        raise ValueError("No eligible train/validation action decisions")
    if args.stage == 2 and (not storage["train"] or not storage["val"]):
        raise ValueError("No post-min-fill old-only storage targets; use more/longer episodes")
    pool = [[eid, d] for eid, ds in actions["val"] for d in ds]
    validation = rng.sample(pool, min(args.val_samples, len(pool)))
    # Cover old-memory queries explicitly without replacing normal validation.
    delayed = []
    for eid, ds in actions["val"]:
        ep = fetch(eid)
        for d in ds:
            boundary = int(ep["frames"][max(0, d - (window - 1))])
            if any(bool(ep["transition_valid"][i]) and int(ep["frames"][i + 1]) < boundary for i in range(d)):
                delayed.append([eid, d])
    delayed = rng.sample(delayed, min(args.val_samples, len(delayed)))
    contexts = {}
    for split, count in (("train", args.storage_contexts), ("val", args.val_storage_samples)):
        rows, seen = [], set()
        if storage[split]:
            for _ in range(count * 50):
                eid, cs = rng.choice(storage[split])
                pair = (eid, rng.choice(cs))
                if pair not in seen:
                    rows.append(list(pair)); seen.add(pair)
                if len(rows) == count:
                    break
        contexts[split] = rows
    return {"train": actions["train"], "validation": validation, "delayed_validation": delayed,
            "storage": contexts, "train_episode_ids": train, "val_episode_ids": val}


def refresh_storage_plan(plans, fetch, cfg, window, args, version):
    """Rotate writer examples across the full eligible training episode split.

    Selection never uses teacher losses. The finite per-teacher pool controls
    costly expert comparisons without permanently restricting writer learning
    to the same initial 128 events. Validation candidates remain fixed.
    """
    rng = random.Random(args.seed + version * 100003 + 4409)
    ids = list(plans["train_episode_ids"])
    rng.shuffle(ids)
    rows = []
    for eid in ids:
        ep = fetch(eid)
        candidates = [c for c in eligible_candidates(ep, window)
                      if int(ep["transition_valid"][:c].sum()) >= cfg.min_fill]
        if candidates:
            rows.append([int(eid), rng.choice(candidates)])
        if len(rows) >= args.storage_contexts:
            break
    if not rows:
        raise ValueError("Refreshed writer plan has no causal old-only examples")
    plans["storage"]["train"] = rows


def _mean(rows):
    keys = set().union(*(r.keys() for r in rows)) if rows else set()
    return {k: float(np.mean([r[k] for r in rows if r.get(k) is not None]))
            for k in keys if any(r.get(k) is not None for r in rows)}


def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _phase(memory, reader_on, writer_on):
    for p in memory.reader_parameters():
        p.requires_grad_(reader_on)
    for p in memory.writer_parameters():
        p.requires_grad_(writer_on)


def _grad_norm(parameters):
    terms = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


def _storage_metrics(aux, label, margin):
    losses = label["option_mean_losses"]
    contrast = max(losses) - min(losses)
    return {**aux["metrics"], "writer_loss": float(aux["loss"].detach()),
            "storage_cost_spread": contrast, "storage_signal_fraction": float(contrast > margin)}


def _flow(head, ep, d, fused=None, **kwargs):
    validate_decision(ep, d)
    return episode_flow_loss(head, ep, d, fused, **kwargs)


def _old_ids(ep, d, bank, window):
    boundary = int(ep["frames"][max(0, d - (window - 1))])
    return [i for i in bank if int(ep["frames"][i + 1]) < boundary]


@torch.no_grad()
def validate(memory, head, fetch, plan, policy, seed, window, delayed=False):
    rows = []
    for index, (eid, d) in enumerate(plan):
        ep = fetch(eid)
        encoded = encode_until(memory, ep, d)
        bank, stats = replay_bank(memory, ep, d, policy, encoded)
        if delayed:
            bank = _old_ids(ep, d, bank, window)
        read = read_bank(memory, ep, d, bank, encoded)
        paired = seed + index * 1009
        out = _flow(head, ep, d, read["fused_short"], seed=paired)
        base = _flow(head, ep, d, seed=paired)
        fifo_ids, _ = replay_bank(memory, ep, d, "all", encoded)
        if delayed:
            fifo_ids = _old_ids(ep, d, fifo_ids, window)
        if fifo_ids == bank:
            fifo = out  # Identical conditioning AND noise: no duplicate expert pass.
        else:
            fifo_read = read_bank(memory, ep, d, fifo_ids, encoded)
            fifo = _flow(head, ep, d, fifo_read["fused_short"], seed=paired)
        row = {"action_loss": float(out["loss"]), "baseline_action_loss": float(base["loss"]),
               "fifo_action_loss": float(fifo["loss"]), "memory_gain": float(base["loss"] - out["loss"]),
               "velocity_mae": float(out["velocity_mae"]), "retrieved_events": float(len(bank))}
        row.update({k: float(v) for k, v in stats.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        row.update({k: float(read[k].mean()) for k in ("gate_mean", "read_norm", "residual_norm", "null_weight")})
        rows.append(row)
    result = _mean(rows)
    result["queries"] = float(len(rows))
    return result


def main(argv=None):
    args = parse_args(argv)
    cache, initial, cfg, base_path, identity, window = preflight(args)
    if args.preflight_only:
        print(json.dumps({"trainer_variant": VARIANT, "stage": args.stage, "base_model": base_path,
                          "memory": asdict(cfg), "cache_fingerprint": cache.manifest["fingerprint"],
                          "train_episodes": len(cache.manifest["splits"]["train"]),
                          "val_episodes": len(cache.manifest["splits"]["val"]),
                          "note": "Metadata verified only; no model loaded and no output written."}, indent=2))
        return 0
    output = Path(args.output_dir).resolve()
    start = initial["step"] if args.resume else 0
    if output.exists() and any(output.iterdir()):
        if not args.resume or Path(args.resume).resolve().parent != output:
            raise ValueError("Use a NEW empty output directory; legacy/existing runs are never overwritten")
        journal = output / "metrics.jsonl"
        if journal.exists() and any(json.loads(x)["step"] > start for x in journal.read_text().splitlines() if x.strip()):
            raise ValueError("Journal is newer than resume checkpoint; resume into a NEW output directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another training process owns this output directory") from exc
        with (output / "training.log").open("a", encoding="utf-8") as journal:
            with redirect_stdout(_Tee(sys.stdout, journal)), redirect_stderr(_Tee(sys.stderr, journal)):
                try:
                    return _run(args, cache, initial, cfg, base_path, identity, window, output, start)
                except Exception:
                    traceback.print_exc()
                    raise


def _run(args, cache, initial, cfg, base_path, identity, window, output, start):
    _seed(args.seed)
    episodes = MappedEpisodes(cache)
    fetch = episodes.fetch
    plans = initial["metadata"]["plans"] if args.resume else make_plans(fetch, cache.manifest, cfg, window, args)
    _atomic_json(output / "sampling_plan.json", plans)
    memory = ActionValueMemory(cfg).to(args.device)
    if initial:
        load_checkpoint(args.resume or args.init_checkpoint, memory)
    _phase(memory, True, args.stage == 2)
    optimizer = torch.optim.AdamW([
        {"params": list(memory.reader_parameters()), "lr": args.reader_learning_rate, "name": "reader"},
        {"params": list(memory.writer_parameters()), "lr": args.writer_learning_rate, "name": "writer"}],
        weight_decay=args.weight_decay)
    base, _ = load_frozen_hamlet(base_path, args.device)
    head = base.action_head
    del base  # Keep only the frozen expert; cached VLM/short features are reused.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    config = {"trainer_variant": VARIANT, "stage": args.stage, "memory": asdict(cfg), "train": vars(args).copy()}
    state = copy.deepcopy(initial["metadata"]["v3_state"]) if args.resume else {
        "teacher_step": 0, "teacher_version": 0, "best_action_loss": None, "last_eval": None,
        "status": "initialized", "elapsed_seconds": 0.0}
    metadata = {"base_model": identity, "cache_fingerprint": cache.manifest["fingerprint"],
                "cache_dir": str(Path(args.cache_dir).resolve()), "plans": plans,
                "validation_plan": plans["validation"], "source_sha256": source_identity(),
                "runtime": runtime_identity(), "v3_state": state,
                "initial_checkpoint": str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None,
                "resumed_from": str(Path(args.resume).resolve()) if args.resume else None,
                "note": "Offline action/choice metrics are not RoboMME success. KEEP/REPLACE credit is conditional, not full-episode BPTT."}
    label_config = LabelV3Config(future_samples=args.future_samples, noise_samples=args.noise_samples,
        scale=args.label_scale, margin=args.label_margin, uncertainty_z=args.uncertainty_z,
        memory_window=window, max_retrieval_events=args.retrieval_candidates)

    def teacher_labels(step, restore=False):
        teacher = copy.deepcopy(memory).eval().requires_grad_(False)
        if restore:
            path = Path(state["teacher_checkpoint"])
            if hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest() != state["teacher_sha256"]:
                raise ValueError("Resume teacher snapshot changed/missing")
            load_checkpoint(path, teacher)
        else:
            path = output / "teachers" / f"checkpoint-{step:06d}"
            if path.exists():
                # A crash may leave a future teacher but no main checkpoint.
                # Reuse only an exactly matching immutable orphan snapshot.
                saved = copy.deepcopy(teacher)
                load_checkpoint(path, saved)
                if any(not torch.equal(v, saved.state_dict()[k]) for k, v in teacher.state_dict().items()):
                    raise ValueError("Orphan teacher differs; resume into a NEW output directory")
            else:
                path = save_checkpoint(output / "teachers", step, teacher, None, config,
                                       {"base_model": identity, "cache_fingerprint": cache.manifest["fingerprint"]}, keep_last=None)
            state.update(teacher_checkpoint=str(path), teacher_sha256=hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest(),
                         teacher_step=step)
        # deepcopy/load can invalidate cuDNN's flattened recurrent weight cache.
        teacher.action_encoder.flatten_parameters()
        return ActionValueLabelsV3(teacher, head, label_config,
            output / "labels" / f"teacher-{state['teacher_version']:06d}",
            {"cache_fingerprint": cache.manifest["fingerprint"], "teacher_version": state["teacher_version"],
             "teacher_sha256": state["teacher_sha256"], "base_model": identity}, copy_teacher=False)

    labels = teacher_labels(state["teacher_step"], restore=bool(args.resume))
    if args.resume:
        load_checkpoint(args.resume, memory, optimizer)
    else:
        _seed(args.seed + 17)
    logger = RunLogger(output)
    _atomic_json(output / "run_config.json", config)
    _atomic_json(output / "provenance.json", metadata)
    elapsed_before = state["elapsed_seconds"]
    started = time.monotonic()
    interval = []
    last_saved = start

    def storage_bank(ep, candidate, policy):
        with torch.no_grad():
            bank, _ = replay_bank(memory, ep, candidate, policy)
        return bank

    def evaluate(step):
        memory.eval()
        policy = "all" if args.stage == 1 else "hard"
        metrics = validate(memory, head, fetch, plans["validation"], policy, args.seed + 50000, window)
        logger.log(step, "val", metrics)
        # The existing live monitor derives comparison/baseline and
        # comparison/matched-fifo from these val metric keys. Journals must
        # keep split names to a SINGLE safe directory component.
        if plans["delayed_validation"]:
            delayed = validate(memory, head, fetch, plans["delayed_validation"], policy, args.seed + 75000, window, True)
            logger.log(step, "val-old-only", delayed)
        if args.stage == 2:
            aux_rows = []
            with torch.no_grad():
                for index, (eid, candidate) in enumerate(plans["storage"]["val"]):
                    ep = fetch(eid)
                    bank = storage_bank(ep, candidate, "hard")
                    label = labels.get_storage(ep, candidate, bank, seed=args.seed + 90000 + index)
                    aux = storage_loss(memory, ep, label)
                    aux_rows.append(_storage_metrics(aux, label, args.label_margin))
            if aux_rows:
                logger.log(step, "val-storage", {**_mean(aux_rows), "teacher_version": float(state["teacher_version"])})
        state["last_eval"] = metrics
        print(f"[v3][val] step={step} action={metrics['action_loss']:.7f} "
              f"base={metrics['baseline_action_loss']:.7f} fifo={metrics['fifo_action_loss']:.7f}", flush=True)
        memory.train()
        return metrics

    if not args.resume:
        evaluate(0)
        save_checkpoint(output, 0, memory, optimizer, config, metadata, keep_last=None)
        logger.plot()
    memory.train()
    try:
        for step in range(start + 1, args.max_steps + 1):
            bootstrap = args.stage == 2 and step <= args.writer_bootstrap
            reader_on = not bootstrap
            _phase(memory, reader_on, args.stage == 2)
            # Snapshot the now-warm reader immediately when ranking starts,
            # and periodically thereafter. A label always records this version.
            ranking_ready = args.ranking_weight > 0 and (args.stage == 2 or step > args.ranking_start)
            refresh = ((step - 1 > state["teacher_step"] and (step - 1) % args.teacher_refresh_steps == 0)
                       or (args.stage == 1 and ranking_ready and state["teacher_step"] < args.ranking_start))
            if refresh and not bootstrap:
                state["teacher_version"] += 1
                labels = teacher_labels(step - 1)
                if args.stage == 2:
                    refresh_storage_plan(plans, fetch, cfg, window, args, state["teacher_version"])
                    _atomic_json(output / "sampling_plan.json", plans)
                    print(f"[v3][teacher] version={state['teacher_version']} rotated writer contexts="
                          f"{len(plans['storage']['train'])}", flush=True)
            hard_fraction = 0.0 if args.stage == 1 else (0.0 if bootstrap else min(1.0, (step - args.writer_bootstrap) / args.policy_ramp_steps))
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                age = max(1, step - args.writer_bootstrap) if group["name"] == "reader" and args.stage == 2 else step
                lr = args.reader_learning_rate if group["name"] == "reader" else args.writer_learning_rate
                group["lr"] = lr * min(1.0, age / max(1, args.warmup_steps))
            row, action_rows, ranks = {"loss": 0.0}, [], []
            if reader_on:
                for micro in range(args.grad_accum):
                    eid, ds = random.choice(plans["train"])
                    d = random.choice(ds)
                    ep = fetch(eid)
                    policy = "hard" if random.random() < hard_fraction else "all"
                    # Selection is discrete/no_grad; selected event encodings
                    # remain differentiable and are never permanently detached.
                    encoded = encode_until(memory, ep, d)
                    bank, stats = replay_bank(memory, ep, d, policy, encoded)
                    read = read_bank(memory, ep, d, bank, encoded)
                    action = _flow(head, ep, d, read["fused_short"], activation_checkpointing=args.activation_checkpointing)
                    if not torch.isfinite(action["loss"]) or not action["loss"].requires_grad:
                        raise FloatingPointError("Nonfinite or disconnected action loss")
                    (action["loss"] / args.grad_accum).backward()
                    record = {"action_loss": float(action["loss"].detach()), "velocity_mae": float(action["velocity_mae"].detach())}
                    record.update({k: float(v) for k, v in stats.items() if isinstance(v, (float, int)) and not isinstance(v, bool)})
                    record.update({k: float(read[k].detach().mean()) for k in ("gate_mean", "read_norm", "residual_norm", "null_weight")})
                    action_rows.append(record)
                    if micro == 0 and ranking_ready and step % args.ranking_every == 0:
                        label = labels.get_retrieval(ep, d, bank, seed=args.seed + eid * 1009 + d)
                        ranked = retrieval_loss(memory, ep, label)
                        if ranked["loss"].requires_grad:
                            (args.ranking_weight * ranked["loss"]).backward()
                        ranks.append({**ranked["metrics"], "ranking_loss": float(ranked["loss"].detach())})
                row.update(_mean(action_rows))
                row["loss"] += row["action_loss"]
                if ranks:
                    row.update(_mean(ranks)); row["loss"] += args.ranking_weight * row["ranking_loss"]
            if args.stage == 2 and (bootstrap or step % args.writer_every == 0):
                aux_rows = []
                for _ in range(args.writer_batch_size):
                    eid, candidate = random.choice(plans["storage"]["train"])
                    ep = fetch(eid)
                    policy = "hard" if random.random() < hard_fraction else "all"
                    bank = storage_bank(ep, candidate, policy)
                    label = labels.get_storage(ep, candidate, bank, seed=args.seed + 31000 + eid * 1009 + candidate)
                    aux = storage_loss(memory, ep, label)
                    factor = 1.0 if bootstrap else args.writer_weight
                    (aux["loss"] * factor / args.writer_batch_size).backward()
                    aux_rows.append(_storage_metrics(aux, label, args.label_margin))
                row.update(_mean(aux_rows)); row["loss"] += factor * row["writer_loss"]
                state["writer_signal_batches"] = state.get("writer_signal_batches", 0) + int(row["storage_signal_fraction"] > 0)
            row["reader_grad_norm"] = _grad_norm(memory.reader_parameters())
            row["writer_grad_norm"] = _grad_norm(memory.writer_parameters())
            norm = torch.nn.utils.clip_grad_norm_(memory.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            row.update(grad_norm=float(norm), phase_bootstrap=float(bootstrap), hard_fraction=hard_fraction,
                       learning_rate=optimizer.param_groups[0]["lr"], writer_learning_rate=optimizer.param_groups[1]["lr"],
                       teacher_version=float(state["teacher_version"]), elapsed_seconds=elapsed_before + time.monotonic() - started)
            interval.append(row)
            state.update(status="writer_bootstrap" if bootstrap else "training", elapsed_seconds=row["elapsed_seconds"])
            if args.stage == 2 and step == args.writer_bootstrap and not state.get("writer_signal_batches", 0):
                print("[v3][WARNING] No writer option contrast above label-margin during bootstrap. "
                      "Inspect reader/old-only gain and writer_grad_norm; completed updates do NOT establish learned storage.", flush=True)
            if step == 1 or step % args.log_steps == 0 or step == args.max_steps:
                logger.log(step, "train", _mean(interval)); interval = []
                print(f"[v3][train] stage={args.stage} step={step} phase={state['status']} "
                      f"action={row.get('action_loss', 'n/a')} writer={row.get('writer_loss', 'n/a')} "
                      f"grad={float(norm):.5f}", flush=True)
            best = False
            due_eval = step % args.eval_steps == 0 or step == args.max_steps
            if due_eval:
                metrics = evaluate(step)
                if not bootstrap and (state["best_action_loss"] is None or metrics["action_loss"] < state["best_action_loss"]):
                    state["best_action_loss"] = metrics["action_loss"]; best = True
                _atomic_json(output / "status.json", {"step": step, **state})
            if step % args.save_steps == 0 or step == args.max_steps or best:
                save_checkpoint(output, step, memory, optimizer, config, metadata, best=best, keep_last=None)
                last_saved = step
            if step % args.plot_steps == 0 or due_eval:
                logger.plot()
    except KeyboardInterrupt:
        print(f"[v3] Interrupted; last immutable checkpoint is update {last_saved}. No partial update saved.", flush=True)
        return 130
    state["status"] = "complete_bootstrap_only" if args.stage == 2 and args.max_steps <= args.writer_bootstrap else "complete"
    _atomic_json(output / "status.json", {"step": args.max_steps, **state})
    print(f"[v3] Completed; results in {output}; offline loss is NOT task success", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
