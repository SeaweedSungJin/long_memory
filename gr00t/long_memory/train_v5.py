"""Recall-supervised reader/AE adaptation followed by continuation-aware writing.

The deployed architecture remains v4-compatible: an unchanged frozen HAMLET,
small AE LoRA adapters and the typed-event memory module. New training-only
heads supervise retrieved information, not future observations. Stage 2 fixes
reader, AE and recall heads; only writer snapshots refresh for offline replay
through subsequent recorded events. This is not a simulator counterfactual.
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

import torch

from .cache import EpisodeCache
from .cache_reader_v3 import MappedEpisodes
from .checkpoint_v4 import load_checkpoint_v4, v4_checkpoint_info, reader_state_sha256
from .checkpoint_v5 import (load_checkpoint_v5, save_checkpoint_v5, v5_checkpoint_info,
                            recall_state_sha256)
from .core_v3 import ActionValueMemory, MemoryV3Config
from .expert_v4 import (LoRAConfig, adapter_disabled, expert_parameters, expert_state_sha256,
                        install_expert_lora, set_expert_trainable)
from .hamlet import checkpoint_identity, load_frozen_hamlet, validate_cache_checkpoint
from .monitoring import RunLogger, _atomic_json, load_checkpoint, save_checkpoint
from .objectives_v5 import ContinuationLabelsV5, LabelV5Config, storage_loss
from .recall_data_v5 import RecallLabels
from .recall_v5 import RecallHeads, recall_loss, recall_metrics
from .safety_v5 import validate_output_scope
from .replay_v3 import encode_until, read_bank, replay_bank, storage_prediction
from .train_v3 import (_Tee, _grad_norm, _mean, _old_ids, _phase, _seed,
                       make_plans, refresh_storage_plan, runtime_identity)
from .train_v4 import _flow, _lineage, parser as v4_parser, source_identity as v4_source_identity

VARIANT = "action_expert_v4"
RECIPE = "recall_continuation_v5"


def parser():
    p = v4_parser()
    p.description = __doc__
    p.add_argument("--recall-labels", help="Immutable target metadata aligned to this feature cache")
    p.add_argument("--subgoal-weight", type=float, default=None)
    p.add_argument("--grounding-weight", type=float, default=None)
    p.add_argument("--recall-learning-rate", type=float, default=1e-4)
    p.add_argument("--delayed-fraction", type=float, default=0.5,
                   help="Probability of sampling available_old proxy; remaining batches use ordinary decisions")
    p.add_argument("--audit-noise-samples", type=int, default=0)
    p.add_argument("--no-confidence-screen", action="store_true")
    p._option_string_actions["--context-refresh-steps"].help = "Rotate contexts AND snapshot continuation writer (reader/AE/recall remain fixed)"
    return p


def _info(path):
    info = json.loads((Path(path) / "checkpoint.json").read_text())
    if info.get("format_version") != 1 or info.get("config", {}).get("trainer_variant") != VARIANT:
        raise ValueError("v5 requires an adapted-Expert v4/v5 checkpoint, not v3")
    return info


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    if args.resume:
        info = _info(args.resume)
        if info["config"].get("training_recipe") != RECIPE:
            raise ValueError("Exact resume requires v5; use --init-checkpoint for v4")
        explicit = {p._option_string_actions[t.split("=", 1)[0]].dest for t in argv
                    if t.split("=", 1)[0] in p._option_string_actions}
        for name, value in info["config"]["train"].items():
            if hasattr(args, name) and name not in explicit:
                setattr(args, name, value)
        args.init_checkpoint = None
        args.resume = p.parse_args(argv).resume
    return args


def source_identity():
    identity = v4_source_identity()
    root = Path(__file__).resolve().parents[2]
    # Explicit new recipe dependencies; existing v4 helper closure is retained.
    for relative in ("gr00t/long_memory/train_v5.py", "gr00t/long_memory/recall_v5.py",
                     "gr00t/long_memory/recall_data_v5.py", "gr00t/long_memory/objectives_v5.py",
                     "gr00t/long_memory/diagnostic_interventions.py",
                     "gr00t/long_memory/safety_v5.py",
                     "gr00t/long_memory/checkpoint_v5.py", "run_scripts/robomme/train_long_memory_v5.py"):
        identity[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    return dict(sorted(identity.items()))


def preflight(args):
    if args.stage not in (1, 2) or not args.cache_dir or not (args.resume or args.init_checkpoint):
        raise ValueError("Provide --stage, --cache-dir and --init-checkpoint, or --resume")
    source = args.resume or args.init_checkpoint
    initial = _info(source)
    conf, meta = initial["config"], initial["metadata"]
    is_v5 = conf.get("training_recipe") == RECIPE
    if args.resume and (not is_v5 or conf["stage"] != args.stage):
        raise ValueError("Exact resume requires same v5 stage")
    if args.init_checkpoint and conf["stage"] != 1:
        raise ValueError("Initialize from a Stage-1 reader checkpoint")
    if args.stage == 2 and not is_v5:
        raise ValueError("Stage 2 requires v5 Stage 1 with saved recall heads")
    for name, default in (("subgoal_weight", .001), ("grounding_weight", .01)):
        if getattr(args, name) is None:
            setattr(args, name, conf.get("train", {}).get(name, default) if is_v5 else default)
    args.recall_labels = args.recall_labels or (conf.get("train", {}).get("recall_labels") if is_v5 else None)
    if not args.recall_labels:
        raise ValueError("--recall-labels is required (also for action-only matched sampling)")
    args.reader_mode = args.reader_mode or conf.get("reader_mode", conf.get("train", {}).get("reader_mode", "memory"))
    if args.reader_mode == "none" and (args.subgoal_weight > 0 or args.grounding_weight > 0):
        raise ValueError("reader-mode none requires both auxiliary weights zero")
    if args.stage == 2 and (args.reader_mode != "memory" or conf.get("reader_mode", "memory") != "memory"):
        raise ValueError("Stage 2 requires memory-enabled Stage 1")
    if args.stage == 2:
        for key, count_key in (("subgoal_weight", "subgoal_supervised_updates"),
                               ("grounding_weight", "grounding_supervised_updates")):
            if getattr(args, key) > 0 and conf["train"].get(key, 0) <= 0:
                raise ValueError(f"Cannot use untrained Stage-1 recall head for {key}")
            count = meta.get("v5_state", {}).get(count_key, 0)
            if getattr(args, key) > 0 and (type(count) is not int or count <= 0):
                raise ValueError(f"Untrained recall head: {count_key}=0; positive config weights alone "
                                 "do not establish supervision (check checkpoint-000000)")
    for name in ("max_steps", "grad_accum", "writer_batch_size", "storage_contexts", "context_refresh_steps",
                 "future_samples", "noise_samples", "val_samples", "val_storage_samples", "eval_steps",
                 "log_steps", "plot_steps", "save_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("seed", "warmup_steps", "max_train_episodes", "max_val_episodes", "audit_noise_samples"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in ("reader_learning_rate", "expert_learning_rate", "writer_learning_rate", "recall_learning_rate",
                 "max_grad_norm", "label_scale"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("subgoal_weight", "grounding_weight", "weight_decay", "label_margin", "uncertainty_z"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(args.delayed_fraction) or not 0 <= args.delayed_fraction < 1:
        raise ValueError("delayed-fraction must be in [0,1); retain ordinary training queries")
    if args.noise_samples < 2:
        raise ValueError("At least two paired noise draws are required")
    if args.audit_noise_samples == 1:
        raise ValueError("audit-noise-samples must be zero or at least two")
    cache = EpisodeCache(args.cache_dir)
    manifest = cache.manifest
    base = str(Path(args.base_model or manifest["model_path"]).resolve())
    if base != str(Path(manifest["model_path"]).resolve()):
        raise ValueError("Base differs from frozen-feature cache")
    validate_cache_checkpoint(manifest)
    identity = checkpoint_identity(base)
    initial = v5_checkpoint_info(base, source) if is_v5 else v4_checkpoint_info(base, source)
    conf, meta = initial["config"], initial["metadata"]
    if meta["base_model"] != identity or meta["cache_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Initial checkpoint base/cache differs")
    labels = RecallLabels(args.recall_labels, cache)
    validate_output_scope(args.output_dir, getattr(cache, "path", args.cache_dir),
                          manifest.get("dataset_path"), base, args.recall_labels, source)
    if labels.manifest["cache_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Recall targets belong to a different feature cache")
    if is_v5 and meta.get("recall_labels_fingerprint") != labels.manifest["fingerprint"]:
        raise ValueError("Initial/resume recall labels changed")
    expert_cfg = LoRAConfig(**conf["expert"])
    for arg, attr in (("lora_rank", "rank"), ("lora_alpha", "alpha")):
        if getattr(args, arg) is not None and getattr(args, arg) != getattr(expert_cfg, attr):
            raise ValueError(f"{arg} differs from inherited Expert architecture")
        setattr(args, arg, getattr(expert_cfg, attr))
    cfg = MemoryV3Config(**conf["memory"])
    for name in ("feature_dim", "state_dim", "action_dim"):
        if getattr(cfg, name) != manifest[name]:
            raise ValueError(f"Cache/memory dimension mismatch: {name}")
    base_cfg = json.loads((Path(base) / "config.json").read_text())
    window = int(base_cfg["memory_window"])
    if cfg.time_scale != base_cfg.get("memory_stride", 16) or not 1 <= cfg.min_fill < cfg.capacity:
        raise ValueError("Invalid memory stride/min-fill for writer experiment")
    if is_v5 and conf["recall"] != {"hidden_dim": cfg.hidden_dim, "num_classes": labels.manifest["num_classes"]}:
        raise ValueError("Recall vocabulary/architecture changed")
    train, val = manifest["splits"]["train"], manifest["splits"]["val"]
    if not train or not val or set(train) & set(val):
        raise ValueError("Require nonempty disjoint episode train/validation splits")
    if args.resume:
        if args.max_steps <= initial["step"] or not (Path(source) / "training_state.pt").is_file():
            raise ValueError("Resume requires optimizer/RNG and greater TOTAL max-steps")
        mutable = {"max_steps", "resume", "init_checkpoint", "output_dir", "base_model", "preflight_only"}
        for name, value in vars(args).items():
            if name not in mutable and conf["train"].get(name) != value:
                raise ValueError(f"Exact resume option changed: {name}")
        if meta.get("source_sha256") != source_identity() or meta.get("runtime") != runtime_identity():
            raise ValueError("Source/runtime changed; exact resume requires recorded implementation")
    return cache, labels, initial, cfg, expert_cfg, base, identity, window


def make_recall_plans(fetch, manifest, cfg, window, args, labels):
    plans = make_plans(fetch, manifest, cfg, window, args)
    plans["available_old_train"] = []
    for eid, decisions in plans["train"]:
        old = [d for d in decisions if labels.get(eid, d)["available_old"]]
        if old:
            plans["available_old_train"].append([eid, old])
    pool = []
    for eid in plans["val_episode_ids"]:
        ep = fetch(eid)
        pool.extend([eid, int(d)] for d in torch.where(ep["decision_mask"])[0].tolist()
                    if labels.get(eid, int(d))["available_old"])
    rng = random.Random(args.seed + 41003)
    plans["available_old_validation"] = rng.sample(pool, min(args.val_samples, len(pool)))
    if args.delayed_fraction > 0 and not plans["available_old_train"]:
        raise ValueError("No available_old training proxy queries; set --delayed-fraction 0 or check labels")
    plans["sampling_note"] = "available_old is annotation/history availability, NOT proven long-memory dependence"
    return plans


def sample_query(plans, delayed_fraction):
    delayed = random.random() < delayed_fraction
    pool = plans["available_old_train"] if delayed else plans["train"]
    eid, decisions = random.choice(pool)
    return eid, random.choice(decisions), delayed


def _aux(read, ep, decision, recall, targets, args):
    result = recall_loss(recall, read, targets.get(int(ep["episode_id"]), decision))
    weighted = args.subgoal_weight * result["subgoal_loss"] + args.grounding_weight * result["grounding_loss"]
    return weighted, result


@torch.no_grad()
def validate(memory, head, recall, targets, fetch, plan, policy, seed, args, window=None):
    rows = []
    for index, (eid, decision) in enumerate(plan):
        ep, paired = fetch(eid), seed + index * 1009
        with adapter_disabled(head):
            baseline = _flow(head, ep, decision, seed=paired)
        expert_only = _flow(head, ep, decision, seed=paired)
        stats, read = {}, {}
        aux_cost, aux_result = torch.tensor(0.), {}
        if args.reader_mode == "none":
            out = fifo = expert_only
        else:
            encoded = encode_until(memory, ep, decision)
            bank, stats = replay_bank(memory, ep, decision, policy, encoded)
            read = read_bank(memory, ep, decision, bank, encoded)
            out = _flow(head, ep, decision, read["fused_short"], seed=paired)
            fifo_ids, _ = replay_bank(memory, ep, decision, "all", encoded)
            fifo = out if fifo_ids == bank else _flow(head, ep, decision,
                read_bank(memory, ep, decision, fifo_ids, encoded)["fused_short"], seed=paired)
            aux_cost, aux_result = _aux(read, ep, decision, recall, targets, args)
        target = targets.get(eid, decision)
        row = {"action_loss": float(out["loss"]), "loss": float(out["loss"] + aux_cost),
               "baseline_action_loss": float(baseline["loss"]), "fifo_action_loss": float(fifo["loss"]),
               "expert_no_memory_action_loss": float(expert_only["loss"]),
               "memory_gain": float(baseline["loss"] - out["loss"]),
               "memory_gain_over_adapted_expert": float(expert_only["loss"] - out["loss"]),
               "velocity_mae": float(out["velocity_mae"]), "weighted_recall_loss": float(aux_cost),
               "available_old_fraction": float(target["available_old"])}
        row.update(recall_metrics(aux_result))
        if args.reader_mode == "memory" and window is not None:
            old = set(_old_ids(ep, decision, bank, window))
            no_old = read if not old else read_bank(memory, ep, decision,
                [event for event in bank if event not in old], encoded)
            no_old_cost, no_old_result = _aux(no_old, ep, decision, recall, targets, args)
            # Same trained head, inference-time retrieval ablation only. There
            # is no extra Action Expert forward and no robot-success claim.
            row.update(no_old_subgoal_accuracy=no_old_result["subgoal_accuracy"],
                       no_old_grounding_mae=no_old_result["grounding_mae"],
                       recall_no_old_gap=float(no_old_cost - aux_cost),
                       removed_old_events=float(len(old)))
        row.update({k: float(v) for k, v in stats.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        row.update({k: float(read[k].mean()) for k in ("gate_mean", "read_norm", "residual_norm", "null_weight") if k in read})
        rows.append(row)
    return {**_mean(rows), "queries": float(len(rows))}


def main(argv=None):
    args = parse_args(argv)
    prepared = preflight(args)
    cache, targets, initial, cfg, expert_cfg, base, identity, window = prepared
    if args.preflight_only:
        print(json.dumps({"training_recipe": RECIPE, "stage": args.stage, "memory": asdict(cfg),
            "expert": asdict(expert_cfg), "recall_classes": targets.manifest["num_classes"],
            "recall_labels_fingerprint": targets.manifest["fingerprint"],
            "note": "Read-only validation; no model, training, or output created."}, indent=2))
        return 0
    output = Path(args.output_dir).resolve()
    start = initial["step"] if args.resume else 0
    if output.exists() and any(output.iterdir()):
        if not args.resume or Path(args.resume).resolve().parent != output:
            raise ValueError("Use a NEW empty output directory; old experiments are never overwritten")
        journal = output / "metrics.jsonl"
        if journal.exists() and any(json.loads(x)["step"] > start for x in journal.read_text().splitlines() if x.strip()):
            raise ValueError("Journal newer than checkpoint; resume into a NEW output directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another training process owns this output directory") from exc
        with (output / "training.log").open("a", encoding="utf-8") as journal:
            with redirect_stdout(_Tee(sys.stdout, journal)), redirect_stderr(_Tee(sys.stderr, journal)):
                try:
                    return _run(args, *prepared, output, start)
                except Exception:
                    traceback.print_exc()
                    raise


def _run(args, cache, targets, initial, cfg, expert_cfg, base_path, identity, window, output, start):
    _seed(args.seed)
    fetch = MappedEpisodes(cache).fetch
    plans = copy.deepcopy(initial["metadata"]["plans"]) if args.resume else make_recall_plans(fetch, cache.manifest, cfg, window, args, targets)
    _atomic_json(output / "sampling_plan.json", plans)
    memory = ActionValueMemory(cfg).to(args.device)
    recall = RecallHeads(cfg.hidden_dim, targets.manifest["num_classes"]).to(args.device)
    base, _ = load_frozen_hamlet(base_path, args.device)
    head = base.action_head
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    source = args.resume or args.init_checkpoint
    expert_targets = install_expert_lora(head, expert_cfg, targets=initial["config"]["expert_targets"])
    if initial["config"].get("training_recipe") == RECIPE:
        load_checkpoint_v5(source, memory, head, recall)
    else:
        load_checkpoint_v4(source, memory, head)
    reader_on = args.stage == 1 and args.reader_mode == "memory"
    recall_on = reader_on and (args.subgoal_weight > 0 or args.grounding_weight > 0)
    _phase(memory, reader_on, args.stage == 2)
    set_expert_trainable(head, args.stage == 1)
    recall.requires_grad_(recall_on).train(recall_on)
    # A masked/inactive task head must not silently drift under AdamW decay.
    recall.subgoal.requires_grad_(recall_on and args.subgoal_weight > 0)
    recall.grounding.requires_grad_(recall_on and args.grounding_weight > 0)
    groups = []
    if reader_on:
        groups.append({"params": list(memory.reader_parameters()), "lr": args.reader_learning_rate, "name": "reader"})
    if args.stage == 1:
        groups.append({"params": list(expert_parameters(head)), "lr": args.expert_learning_rate, "name": "expert"})
    if recall_on:
        groups.append({"params": [p for p in recall.parameters() if p.requires_grad], "lr": args.recall_learning_rate, "name": "recall"})
    if args.stage == 2:
        groups.append({"params": list(memory.writer_parameters()), "lr": args.writer_learning_rate, "name": "writer"})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    parameters = [p for g in groups for p in g["params"]]
    config = {"trainer_variant": VARIANT, "training_recipe": RECIPE, "stage": args.stage,
              "reader_mode": args.reader_mode, "memory": asdict(cfg), "expert": asdict(expert_cfg),
              "expert_targets": expert_targets, "recall": {"hidden_dim": cfg.hidden_dim, "num_classes": targets.manifest["num_classes"]},
              "train": vars(args).copy()}
    state = copy.deepcopy(initial["metadata"]["v5_state"]) if args.resume else {
        "best_action_loss": None, "last_eval": None, "elapsed_seconds": 0., "status": "initialized",
        "context_version": 0, "teacher_version": 0, "writer_signal_batches": 0, "optimizer_updates": 0,
        "subgoal_supervised_updates": initial["metadata"].get("v5_state", {}).get("subgoal_supervised_updates", 0),
        "grounding_supervised_updates": initial["metadata"].get("v5_state", {}).get("grounding_supervised_updates", 0)}
    metadata = {"base_model": identity, "cache_fingerprint": cache.manifest["fingerprint"],
        "cache_dir": str(Path(args.cache_dir).resolve()), "recall_labels_fingerprint": targets.manifest["fingerprint"],
        "plans": plans, "validation_plan": plans["validation"], "source_sha256": source_identity(),
        "runtime": runtime_identity(), "v5_state": state,
        "initial_checkpoint": initial["metadata"].get("initial_checkpoint") if args.resume else str(Path(source).resolve()),
        "initial_checkpoint_sha256": initial["metadata"].get("initial_checkpoint_sha256") if args.resume else hashlib.sha256((Path(source) / "checkpoint.json").read_bytes()).hexdigest(),
        "resumed_from": str(Path(args.resume).resolve()) if args.resume else None,
        "note": "Auxiliary annotation recall is not success or proof of memory necessity. Continuation uses recorded trajectories, not simulator counterfactuals."}
    labeler = None
    if args.stage == 2:
        if args.resume:
            metadata["stage1_parent"] = copy.deepcopy(initial["metadata"]["stage1_parent"])
        else:
            metadata["stage1_parent"] = {**_lineage(source), "recall_sha256": hashlib.sha256((Path(source) / "recall.safetensors").read_bytes()).hexdigest()}
        metadata.update(frozen_reader_sha256=reader_state_sha256(memory), frozen_expert_sha256=expert_state_sha256(head),
                        frozen_recall_sha256=recall_state_sha256(recall))
        if args.resume:
            for key in ("frozen_reader_sha256", "frozen_expert_sha256", "frozen_recall_sha256"):
                if initial["metadata"].get(key) != metadata[key]:
                    raise ValueError(f"Frozen Stage-2 state changed: {key}")
    label_config = LabelV5Config(memory_window=window, future_samples=args.future_samples,
        noise_samples=args.noise_samples, scale=args.label_scale, margin=args.label_margin,
        uncertainty_z=args.uncertainty_z, continuation_policy="hard", audit_noise_samples=args.audit_noise_samples,
        recall_weight=1. if args.subgoal_weight > 0 or args.grounding_weight > 0 else 0.,
        confidence_screen=not args.no_confidence_screen)

    def make_labeler(step, restore=False):
        teacher = copy.deepcopy(memory).eval().requires_grad_(False)
        if restore:
            path = Path(state["teacher_checkpoint"])
            if hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest() != state["teacher_memory_sha256"]:
                raise ValueError("Continuation teacher snapshot changed/missing")
            load_checkpoint(path, teacher)
        else:
            path = output / "teachers" / f"checkpoint-{step:06d}"
            if path.exists():
                saved = copy.deepcopy(teacher)
                load_checkpoint(path, saved)
                if any(not torch.equal(value, saved.state_dict()[key]) for key, value in teacher.state_dict().items()):
                    raise ValueError("Orphan teacher differs; resume into a NEW output directory")
            else:
                save_checkpoint(output / "teachers", step, teacher, None, config,
                    {"base_model": identity, "cache_fingerprint": cache.manifest["fingerprint"]}, keep_last=None)
            state.update(teacher_checkpoint=str(path.resolve()),
                         teacher_memory_sha256=hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest())
        teacher.action_encoder.flatten_parameters()
        def recall_cost(read_output, episode, decision):
            return _aux(read_output, episode, decision, recall, targets, args)[0]
        return ContinuationLabelsV5(teacher, head, label_config,
            output / "labels" / f"teacher-{state['teacher_version']:06d}",
            {"cache_fingerprint": cache.manifest["fingerprint"], "teacher_version": state["teacher_version"],
             "action_expert_fingerprint": metadata["frozen_expert_sha256"], "base_model": identity,
             "teacher_memory_sha256": state["teacher_memory_sha256"],
             "recall_fingerprint": metadata["frozen_recall_sha256"],
             "recall_labels_fingerprint": targets.manifest["fingerprint"],
             "subgoal_weight": args.subgoal_weight, "grounding_weight": args.grounding_weight},
            copy_teacher=False, recall_cost=recall_cost if label_config.recall_weight else None,
            recall_module=recall if label_config.recall_weight else None)

    if args.resume:
        # Restore parameters before the labeler captures the frozen Expert's
        # tensor mutation-version guards. Constructing a copied teacher below
        # consumes no RNG; restore/checkpoint loading itself does not sample.
        load_checkpoint_v5(source, memory, head, recall, optimizer)
    else:
        _seed(args.seed + 17)
    if args.stage == 2:
        labeler = make_labeler(0, restore=bool(args.resume))
    logger = RunLogger(output)
    _atomic_json(output / "run_config.json", config)
    _atomic_json(output / "provenance.json", metadata)
    print(f"[v5] stage={args.stage} recipe={RECIPE}; trainable " + " ".join(
        f"{g['name']}={sum(p.numel() for p in g['params'])}" for g in groups), flush=True)
    print("[v5] available_old sampling is a proxy, not evidence of necessary long-term memory", flush=True)
    elapsed_before, started, interval, last_saved = state["elapsed_seconds"], time.monotonic(), [], start

    def storage_aux(ep, candidate, seed):
        with torch.no_grad():
            bank, _ = replay_bank(memory, ep, candidate, "hard")
        label = labeler.get_storage(ep, candidate, bank, seed=seed, loss_fn=_flow)
        aux = storage_loss(memory, ep, label)
        metrics = {**aux["metrics"], "writer_loss": float(aux["loss"].detach())}
        costs = label.get("option_mean_losses", [])
        metrics["storage_cost_spread"] = max(costs) - min(costs) if costs else 0.
        metrics["storage_raw_signal_fraction"] = float(metrics["storage_cost_spread"] > args.label_margin)
        if costs:
            with torch.no_grad():
                probabilities = storage_prediction(memory, ep, candidate, bank)["logits"].flatten().softmax(0)
            keep_advantage = min(costs[1:]) - costs[0]
            metrics.update(teacher_keep_argmin_fraction=float(label["best_option"] == 0),
                teacher_keep_strict_best_fraction=float(keep_advantage > args.label_margin),
                teacher_keep_tie_best_fraction=float(costs[0] <= min(costs) + args.label_margin),
                teacher_keep_margin=keep_advantage, predicted_keep_fraction=float(int(probabilities.argmax()) == 0),
                predicted_keep_probability=float(probabilities[0]))
        return aux, metrics

    def assert_frozen():
        if args.stage == 2 and (reader_state_sha256(memory) != metadata["frozen_reader_sha256"]
                or expert_state_sha256(head) != metadata["frozen_expert_sha256"]
                or recall_state_sha256(recall) != metadata["frozen_recall_sha256"]):
            raise RuntimeError("Frozen reader/Expert/recall changed during writer-only learning")

    def evaluate(step):
        assert_frozen()
        memory.eval(); recall.eval()
        policy = "all" if args.stage == 1 else "hard"
        metrics = validate(memory, head, recall, targets, fetch, plans["validation"], policy, args.seed + 50000, args, window)
        logger.log(step, "val", metrics)
        if plans["available_old_validation"]:
            logger.log(step, "val-available-old", validate(memory, head, recall, targets, fetch,
                       plans["available_old_validation"], policy, args.seed + 75000, args, window))
        if args.stage == 2:
            with torch.no_grad():
                rows = [storage_aux(fetch(eid), candidate, args.seed + 90000 + index)[1]
                        for index, (eid, candidate) in enumerate(plans["storage"]["val"])]
            logger.log(step, "val-storage", {**_mean(rows), "teacher_version": float(state["teacher_version"])})
        state["last_eval"] = metrics
        print(f"[v5][val] step={step} action={metrics['action_loss']:.7f} original={metrics['baseline_action_loss']:.7f} "
              f"recall={metrics['weighted_recall_loss']:.7f} (not robot success)", flush=True)
        memory.train(reader_on); memory.writer.train(args.stage == 2); recall.train(recall_on)
        return metrics

    if not args.resume:
        metrics = evaluate(0)
        state["best_action_loss"] = metrics["action_loss"]
        save_checkpoint_v5(output, 0, memory, head, recall, optimizer, config, metadata, best=True)
        logger.plot()
    memory.train(reader_on); memory.writer.train(args.stage == 2); recall.train(recall_on)
    try:
        for step in range(start + 1, args.max_steps + 1):
            update_started = time.monotonic()
            if args.stage == 2 and step > 1 and (step - 1) % args.context_refresh_steps == 0:
                state["context_version"] += 1
                state["teacher_version"] += 1
                labeler = make_labeler(step - 1)
                refresh_storage_plan(plans, fetch, cfg, window, args, state["context_version"])
                _atomic_json(output / "sampling_plan.json", plans)
                print(f"[v5] continuation writer snapshot version={state['teacher_version']}; reader/Expert/recall stay frozen", flush=True)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = getattr(args, group["name"] + "_learning_rate") * min(1., step / max(1, args.warmup_steps))
            rows = []
            if args.stage == 1:
                for _ in range(args.grad_accum):
                    eid, decision, delayed = sample_query(plans, args.delayed_fraction)
                    ep = fetch(eid)
                    stats, read, fused = {}, {}, None
                    if args.reader_mode == "memory":
                        encoded = encode_until(memory, ep, decision)
                        bank, stats = replay_bank(memory, ep, decision, "all", encoded)
                        read = read_bank(memory, ep, decision, bank, encoded)
                        fused = read["fused_short"]
                    action = _flow(head, ep, decision, fused, activation_checkpointing=args.activation_checkpointing)
                    aux_cost, result = (action["loss"] * 0, {}) if args.reader_mode == "none" else _aux(read, ep, decision, recall, targets, args)
                    total = action["loss"] + aux_cost
                    if not torch.isfinite(total) or not total.requires_grad:
                        raise FloatingPointError("Nonfinite or disconnected action/recall objective")
                    (total / args.grad_accum).backward()
                    row = {"action_loss": float(action["loss"].detach()), "loss": float(total.detach()),
                           "weighted_recall_loss": float(aux_cost.detach()), "sampled_delayed_fraction": float(delayed),
                           "available_old_fraction": float(targets.get(eid, decision)["available_old"]),
                           "velocity_mae": float(action["velocity_mae"].detach()), **recall_metrics(result)}
                    row.update({k: float(v) for k, v in stats.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
                    row.update({k: float(read[k].detach().mean()) for k in ("gate_mean", "read_norm", "residual_norm", "null_weight") if k in read})
                    rows.append(row)
            else:
                for _ in range(args.writer_batch_size):
                    eid, candidate = random.choice(plans["storage"]["train"])
                    aux, row = storage_aux(fetch(eid), candidate, args.seed + 31000 + eid * 1009 + candidate)
                    if not torch.isfinite(aux["loss"]) or not aux["loss"].requires_grad:
                        raise FloatingPointError("Nonfinite or disconnected writer objective")
                    (aux["loss"] / args.writer_batch_size).backward()
                    rows.append({**row, "loss": row["writer_loss"]})
            row = _mean(rows)
            row.update(reader_grad_norm=_grad_norm(memory.reader_parameters()), expert_grad_norm=_grad_norm(expert_parameters(head)),
                       recall_grad_norm=_grad_norm(recall.parameters()), writer_grad_norm=_grad_norm(memory.writer_parameters()))
            norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
            # A screened-out label supplies no objective signal. Do not let
            # AdamW decay or accumulated momentum silently change the writer.
            apply_update = args.stage == 1 or row["storage_signal_fraction"] > 0
            if apply_update:
                optimizer.step()
                state["optimizer_updates"] += 1
            if args.stage == 1:
                for weight, valid, count in ((args.subgoal_weight, "class_valid", "subgoal_supervised_updates"),
                                             (args.grounding_weight, "xy_valid", "grounding_supervised_updates")):
                    state[count] += int(weight > 0 and row.get(valid, 0) > 0)
            row.update(grad_norm=float(norm), learning_rate=optimizer.param_groups[0]["lr"],
                       optimizer_step_applied=float(apply_update),
                       optimizer_updates=float(state["optimizer_updates"]),
                       subgoal_supervised_updates=float(state["subgoal_supervised_updates"]),
                       grounding_supervised_updates=float(state["grounding_supervised_updates"]),
                       teacher_version=float(state["teacher_version"]), update_seconds=time.monotonic() - update_started,
                       elapsed_seconds=elapsed_before + time.monotonic() - started)
            for group in optimizer.param_groups:
                row[group["name"] + "_learning_rate"] = group["lr"]
            interval.append(row)
            state.update(status="joint_recall_training" if args.stage == 1 else "continuation_writer_training", elapsed_seconds=row["elapsed_seconds"])
            if args.stage == 2:
                state["writer_signal_batches"] += int(row["storage_signal_fraction"] > 0)
            if step == 1 or step % args.log_steps == 0 or step == args.max_steps:
                logger.log(step, "train", _mean(interval)); interval = []
                print(f"[v5][train] stage={args.stage} step={step} action={row.get('action_loss', 'n/a')} "
                      f"recall={row.get('weighted_recall_loss', 'n/a')} writer={row.get('writer_loss', 'n/a')} "
                      f"grads(reader/expert/recall/writer)={row['reader_grad_norm']:.5g}/{row['expert_grad_norm']:.5g}/"
                      f"{row['recall_grad_norm']:.5g}/{row['writer_grad_norm']:.5g}", flush=True)
            best, due_eval = False, step % args.eval_steps == 0 or step == args.max_steps
            if due_eval:
                metrics = evaluate(step)
                if metrics["action_loss"] < state["best_action_loss"]:
                    state["best_action_loss"] = metrics["action_loss"]; best = True
                _atomic_json(output / "status.json", {"step": step, **state})
            if step % args.save_steps == 0 or step == args.max_steps or best:
                assert_frozen()
                save_checkpoint_v5(output, step, memory, head, recall, optimizer, config, metadata, best=best)
                last_saved = step
            if step % args.plot_steps == 0 or due_eval:
                logger.plot()
    except KeyboardInterrupt:
        print(f"[v5] Interrupted; last complete immutable checkpoint is update {last_saved}", flush=True)
        return 130
    state["status"] = "complete"
    _atomic_json(output / "status.json", {"step": args.max_steps, **state})
    print(f"[v5] Complete: {output}; evaluate simulator success separately", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
