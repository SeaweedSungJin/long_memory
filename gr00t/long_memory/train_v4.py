"""Action-Expert adaptation, isolated from the frozen-Expert v3 experiments.

Stage 1 starts from an existing reader and jointly tunes that reader plus small
attention LoRA adapters in the Action Expert. Original HAMLET weights never
change. Stage 2 fixes the adapted Expert AND reader, then learns only the
causal KEEP/APPEND/REPLACE writer. Labels measure conditional future action
loss, not environment success. There is no moving teacher in Stage 2.

The optional Stage-1 ``reader-mode none`` run is the matched Expert-only
control. It deliberately cannot initialize Stage 2. All checkpoints include
the memory module and the exact Expert adapters needed at inference.
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
from .cache_reader_v3 import MappedEpisodes, validate_decision
from .checkpoint_v3 import memory_v3_checkpoint_info
from .checkpoint_v4 import (load_checkpoint_v4, save_checkpoint_v4,
                            reader_state_sha256, v4_checkpoint_info)
from .core_v3 import ActionValueMemory, MemoryV3Config
from .expert_v4 import (LoRAConfig, adapter_disabled, expert_parameters,
                        expert_state_sha256, expert_episode_flow_loss,
                        install_expert_lora, set_expert_trainable)
from .hamlet import checkpoint_identity, load_frozen_hamlet, validate_cache_checkpoint
from .monitoring import RunLogger, _atomic_json, load_checkpoint
from .objectives_v3 import ActionValueLabelsV3, LabelV3Config, storage_loss
from .replay_v3 import encode_until, read_bank, replay_bank, storage_prediction
from .train_v3 import (_Tee, _grad_norm, _mean, _old_ids, _phase, _seed,
                       make_plans, refresh_storage_plan, runtime_identity)

VARIANT = "action_expert_v4"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, choices=(1, 2))
    p.add_argument("--cache-dir")
    p.add_argument("--base-model")
    p.add_argument("--output-dir", required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--init-checkpoint", help="Stage 1: v3/v4 reader; Stage 2: v4 memory-enabled Stage 1")
    source.add_argument("--resume", help="Exact optimizer/RNG resume; max-steps is TOTAL updates")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reader-mode", choices=("memory", "none"), help="none = matched Expert-only Stage-1 control")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--reader-learning-rate", type=float, default=1e-5)
    p.add_argument("--expert-learning-rate", type=float, default=1e-5)
    p.add_argument("--writer-learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int)
    p.add_argument("--lora-alpha", type=float)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--writer-batch-size", type=int, default=4)
    p.add_argument("--storage-contexts", type=int, default=128)
    p.add_argument("--context-refresh-steps", type=int, default=250,
                   help="Rotate writer examples only; frozen teacher weights never refresh")
    p.add_argument("--future-samples", type=int, default=2)
    p.add_argument("--noise-samples", type=int, default=2)
    p.add_argument("--label-scale", type=float, default=0.001)
    p.add_argument("--label-margin", type=float, default=1e-5)
    p.add_argument("--uncertainty-z", type=float, default=2.0)
    p.add_argument("--val-samples", type=int, default=16)
    p.add_argument("--val-storage-samples", type=int, default=4)
    p.add_argument("--max-train-episodes", type=int, default=0)
    p.add_argument("--max-val-episodes", type=int, default=0)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--plot-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--preflight-only", action="store_true", help="Read-only metadata check; no model/output creation")
    return p


def _info(path):
    root = Path(path)
    info = json.loads((root / "checkpoint.json").read_text())
    variant = info.get("config", {}).get("trainer_variant")
    if info.get("format_version") != 1 or variant not in ("action_value_v3", VARIANT):
        raise ValueError("Expected v3 or v4 checkpoint; legacy v1/v2 are separate experiments")
    if not (root / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing memory weights: {root}")
    if variant == VARIANT and not (root / "expert.safetensors").is_file():
        raise FileNotFoundError(f"Missing adapted Action Expert: {root}")
    return info


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    if args.resume:
        info = _info(args.resume)
        if info["config"]["trainer_variant"] != VARIANT:
            raise ValueError("Resume requires v4; use --init-checkpoint to adapt a v3 reader")
        explicit = {p._option_string_actions[token.split("=", 1)[0]].dest
                    for token in argv if token.split("=", 1)[0] in p._option_string_actions}
        for name, value in info["config"]["train"].items():
            if hasattr(args, name) and name not in explicit:
                setattr(args, name, value)
        args.init_checkpoint = None
        args.resume = p.parse_args(argv).resume
    return args


def validate_args(args):
    if args.stage not in (1, 2) or not args.cache_dir:
        raise ValueError("Provide --stage and --cache-dir, or inherit them with --resume")
    if not (args.init_checkpoint or args.resume):
        raise ValueError("Provide --init-checkpoint; v4 preserves and adapts an existing reader")
    for name in ("max_steps", "grad_accum", "writer_batch_size", "storage_contexts",
                 "context_refresh_steps", "future_samples", "noise_samples", "val_samples",
                 "val_storage_samples", "eval_steps", "log_steps", "plot_steps", "save_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("seed", "warmup_steps", "max_train_episodes", "max_val_episodes"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in ("reader_learning_rate", "expert_learning_rate", "writer_learning_rate",
                 "max_grad_norm", "label_scale"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "label_margin", "uncertainty_z"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.noise_samples < 2:
        raise ValueError("At least two paired noise draws are required")


def source_identity():
    """Hash the Python dependency closure, not future unrelated experiments."""
    # The long-memory package is small. Resolve relative imports transitively
    # so importing a legacy helper also records that helper's dependencies.
    import ast
    root = Path(__file__).resolve().parents[2]
    pending = [Path(__file__).resolve(), root / "run_scripts/robomme/train_long_memory_v4.py"]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    parent = path.parent
                    for _ in range(node.level - 1):
                        parent = parent.parent
                    target = parent / ((node.module or "").replace(".", "/") + ".py")
                elif node.module and node.module.startswith("gr00t."):
                    target = root / (node.module.replace(".", "/") + ".py")
                else:
                    continue
                if target.is_file():
                    pending.append(target.resolve())
    # HAMLET loading registers dynamic AutoModel classes via ``import
    # gr00t.model``. These dispatch targets are invisible to ImportFrom alone.
    # Hash model/config Python files, but not unrelated future memory trainers.
    for directory in ("gr00t/model", "gr00t/configs"):
        seen.update(p.resolve() for p in (root / directory).rglob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(seen)}


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
    source = args.resume or args.init_checkpoint
    initial = _info(source)
    if initial["config"]["trainer_variant"] == VARIANT:
        initial = v4_checkpoint_info(base, source)
    else:
        initial = memory_v3_checkpoint_info(base, source)
    conf, meta = initial["config"], initial["metadata"]
    if meta["base_model"] != identity or meta["cache_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Initial/resume checkpoint base/cache differs")
    if args.resume and conf["stage"] != args.stage:
        raise ValueError("Exact resume cannot change stage")
    if args.init_checkpoint and conf["stage"] != 1:
        raise ValueError("Initialization requires a Stage-1 reader; Stage 2 is not a new reader")
    if args.stage == 2 and conf["trainer_variant"] != VARIANT:
        raise ValueError("Stage 2 requires a v4 Stage-1 checkpoint with adapted Expert")
    saved_mode = conf.get("reader_mode", conf.get("train", {}).get("reader_mode", "memory"))
    args.reader_mode = args.reader_mode or saved_mode
    if args.stage == 2 and (saved_mode != "memory" or args.reader_mode != "memory"):
        raise ValueError("Stage 2 cannot initialize from the Expert-only no-memory control")
    if conf["trainer_variant"] == VARIANT:
        expert_cfg = LoRAConfig(**conf["expert"])
        for arg, attr in (("lora_rank", "rank"), ("lora_alpha", "alpha")):
            if getattr(args, arg) is not None and getattr(args, arg) != getattr(expert_cfg, attr):
                raise ValueError(f"{arg} must match the inherited Expert architecture")
    else:
        expert_cfg = LoRAConfig(rank=args.lora_rank if args.lora_rank is not None else 8,
                               alpha=args.lora_alpha if args.lora_alpha is not None else 16.0)
    args.lora_rank, args.lora_alpha = expert_cfg.rank, expert_cfg.alpha
    cfg = MemoryV3Config(**conf["memory"])
    if not 1 <= cfg.min_fill < cfg.capacity:
        raise ValueError("Require 1 <= min-fill < capacity")
    for name in ("feature_dim", "state_dim", "action_dim"):
        if getattr(cfg, name) != manifest[name]:
            raise ValueError(f"Cache/memory dimension mismatch: {name}")
    if float(cfg.time_scale) != float(stride):
        raise ValueError("Memory time scale must match cached/inference stride")
    train, val = manifest["splits"]["train"], manifest["splits"]["val"]
    if not train or not val or set(train) & set(val):
        raise ValueError("Require nonempty disjoint episode train/validation splits")
    if args.resume:
        if args.max_steps <= initial["step"] or not (Path(args.resume) / "training_state.pt").is_file():
            raise ValueError("Resume requires optimizer/RNG and a greater TOTAL max-steps")
        mutable = {"max_steps", "resume", "init_checkpoint", "output_dir", "base_model", "preflight_only"}
        for name, value in vars(args).items():
            if name not in mutable and conf["train"].get(name) != value:
                raise ValueError(f"Exact resume option changed: {name}")
        if meta.get("source_sha256") != source_identity():
            raise ValueError("Training source changed; exact resume requires unchanged source")
        if meta.get("runtime") != runtime_identity():
            raise ValueError("Runtime/math settings changed; exact resume requires recorded environment")
    return cache, initial, cfg, expert_cfg, base, identity, window


def reader_fingerprint(memory):
    """Canonical fingerprint excluding writer, which legitimately learns in S2."""
    return reader_state_sha256(memory)


def _lineage(path):
    path = Path(path).resolve()
    return {"path": str(path), **{key: hashlib.sha256((path / filename).read_bytes()).hexdigest()
            for key, filename in (("checkpoint_sha256", "checkpoint.json"),
                                  ("memory_sha256", "model.safetensors"),
                                  ("expert_sha256", "expert.safetensors"))}}


def _flow(head, ep, decision, fused=None, **kwargs):
    validate_decision(ep, decision)
    return expert_episode_flow_loss(head, ep, decision, fused, **kwargs)


def storage_diagnostics(aux, label, logits, margin):
    """Distinguish absent KEEP benefit from failure to learn an available one.

    'tie-best' includes ties within the configured raw-loss tolerance;
    'strict-best' requires an absolute loss advantage above that margin. Neither
    is a reward or a forced rejection target. Margins are raw action-loss units.
    """
    losses = label["option_mean_losses"]
    alternative = min(losses[1:])
    keep_advantage = alternative - losses[0]
    probabilities = logits.detach().flatten().softmax(0)
    selected = int(probabilities.argmax())
    contrast = max(losses) - min(losses)
    return {**aux["metrics"], "writer_loss": float(aux["loss"].detach()),
            "storage_cost_spread": contrast, "storage_signal_fraction": float(contrast > margin),
            "teacher_keep_argmin_fraction": float(label["best_option"] == 0),
            "teacher_keep_strict_best_fraction": float(keep_advantage > margin),
            "teacher_keep_tie_best_fraction": float(losses[0] <= min(losses) + margin),
            "teacher_keep_margin": keep_advantage,
            "teacher_append_replace_strict_best_fraction": float(keep_advantage < -margin),
            "predicted_keep_fraction": float(selected == 0),
            "predicted_keep_probability": float(probabilities[0]),
            "storage_selected_regret": losses[selected] - min(losses),
            "storage_full_bank_fraction": float(len(label["options"][0]) == len(label["options"][1]))}


@torch.no_grad()
def validate(memory, head, fetch, plan, policy, seed, window, reader_mode="memory", delayed=False):
    rows = []
    for index, (eid, decision) in enumerate(plan):
        ep, paired = fetch(eid), seed + index * 1009
        # Adapter-disabled AND unmodified cached features is original HAMLET.
        # Merely removing long memory while adapters remain enabled is NOT it.
        with adapter_disabled(head):
            baseline = _flow(head, ep, decision, seed=paired)
        expert_only = _flow(head, ep, decision, seed=paired)
        stats, read, bank = {}, {}, []
        if reader_mode == "none":
            out = fifo = expert_only
        else:
            encoded = encode_until(memory, ep, decision)
            bank, stats = replay_bank(memory, ep, decision, policy, encoded)
            if delayed:
                bank = _old_ids(ep, decision, bank, window)
            read = read_bank(memory, ep, decision, bank, encoded)
            out = _flow(head, ep, decision, read["fused_short"], seed=paired)
            fifo_ids, _ = replay_bank(memory, ep, decision, "all", encoded)
            if delayed:
                fifo_ids = _old_ids(ep, decision, fifo_ids, window)
            if fifo_ids == bank:
                fifo = out
            else:
                fifo_read = read_bank(memory, ep, decision, fifo_ids, encoded)
                fifo = _flow(head, ep, decision, fifo_read["fused_short"], seed=paired)
        row = {"action_loss": float(out["loss"]), "baseline_action_loss": float(baseline["loss"]),
               "expert_no_memory_action_loss": float(expert_only["loss"]),
               "fifo_action_loss": float(fifo["loss"]),
               "memory_gain": float(baseline["loss"] - out["loss"]),
               "memory_gain_over_adapted_expert": float(expert_only["loss"] - out["loss"]),
               "velocity_mae": float(out["velocity_mae"]), "retrieved_events": float(len(bank))}
        row.update({k: float(v) for k, v in stats.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        row.update({k: float(read[k].mean()) for k in ("gate_mean", "read_norm", "residual_norm", "null_weight") if k in read})
        rows.append(row)
    return {**_mean(rows), "queries": float(len(rows))}


def main(argv=None):
    args = parse_args(argv)
    prepared = preflight(args)
    cache, initial, cfg, expert_cfg, base_path, identity, window = prepared
    if args.preflight_only:
        print(json.dumps({"trainer_variant": VARIANT, "stage": args.stage, "reader_mode": args.reader_mode,
                          "base_model": base_path, "memory": asdict(cfg), "expert": asdict(expert_cfg),
                          "cache_fingerprint": cache.manifest["fingerprint"],
                          "train_episodes": len(cache.manifest["splits"]["train"]),
                          "val_episodes": len(cache.manifest["splits"]["val"]),
                          "note": "Metadata only; no model loaded/output written; old runs untouched."}, indent=2))
        return 0
    output = Path(args.output_dir).resolve()
    start = initial["step"] if args.resume else 0
    if output.exists() and any(output.iterdir()):
        if not args.resume or Path(args.resume).resolve().parent != output:
            raise ValueError("Use a NEW empty output directory; existing runs are never overwritten")
        journal = output / "metrics.jsonl"
        if journal.exists() and any(json.loads(x)["step"] > start for x in journal.read_text().splitlines() if x.strip()):
            raise ValueError("Journal is newer than resume checkpoint; use a NEW output directory")
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


def _run(args, cache, initial, cfg, expert_cfg, base_path, identity, window, output, start):
    _seed(args.seed)
    episodes = MappedEpisodes(cache)
    fetch = episodes.fetch
    plans = copy.deepcopy(initial["metadata"]["plans"]) if args.resume else make_plans(fetch, cache.manifest, cfg, window, args)
    _atomic_json(output / "sampling_plan.json", plans)
    memory = ActionValueMemory(cfg).to(args.device)
    base, _ = load_frozen_hamlet(base_path, args.device)
    head = base.action_head
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    source = args.resume or args.init_checkpoint
    initial_v4 = initial["config"]["trainer_variant"] == VARIANT
    targets = install_expert_lora(head, expert_cfg, targets=initial["config"].get("expert_targets") if initial_v4 else None)
    if initial_v4:
        load_checkpoint_v4(source, memory, head)
    else:
        load_checkpoint(source, memory)
    reader_on = args.stage == 1 and args.reader_mode == "memory"
    _phase(memory, reader_on, args.stage == 2)
    set_expert_trainable(head, args.stage == 1)
    head.eval()  # Frozen original dropout/batch-stat behaviour; LoRA still has autograd.
    groups = []
    if reader_on:
        groups.append({"params": list(memory.reader_parameters()), "lr": args.reader_learning_rate, "name": "reader"})
    if args.stage == 1:
        groups.append({"params": list(expert_parameters(head)), "lr": args.expert_learning_rate, "name": "expert"})
    else:
        groups.append({"params": list(memory.writer_parameters()), "lr": args.writer_learning_rate, "name": "writer"})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    parameters = [p for group in groups for p in group["params"]]
    config = {"trainer_variant": VARIANT, "stage": args.stage, "reader_mode": args.reader_mode,
              "memory": asdict(cfg), "expert": asdict(expert_cfg), "expert_targets": targets, "train": vars(args).copy()}
    state = copy.deepcopy(initial["metadata"]["v4_state"]) if args.resume else {
        "best_action_loss": None, "last_eval": None, "status": "initialized", "elapsed_seconds": 0.0,
        "context_version": 0, "writer_signal_batches": 0}
    metadata = {"base_model": identity, "cache_fingerprint": cache.manifest["fingerprint"],
                "cache_dir": str(Path(args.cache_dir).resolve()), "plans": plans,
                "validation_plan": plans["validation"], "source_sha256": source_identity(),
                "runtime": runtime_identity(), "v4_state": state,
                "initial_checkpoint": str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else initial["metadata"].get("initial_checkpoint"),
                "initial_checkpoint_sha256": initial["metadata"].get("initial_checkpoint_sha256") if args.resume else hashlib.sha256((Path(source) / "checkpoint.json").read_bytes()).hexdigest(),
                "initial_memory_sha256": initial["metadata"].get("initial_memory_sha256") if args.resume else hashlib.sha256((Path(source) / "model.safetensors").read_bytes()).hexdigest(),
                "resumed_from": str(Path(args.resume).resolve()) if args.resume else None,
                "note": "Attention LoRA, not full Expert fine-tuning. S1 action-loss-only; S2 writer-only conditional future loss. Offline metrics are not success rates."}
    labels = None
    if args.stage == 2:
        metadata["stage1_parent"] = copy.deepcopy(initial["metadata"]["stage1_parent"]) if args.resume else _lineage(source)
        metadata["frozen_reader_sha256"] = reader_fingerprint(memory)
        metadata["frozen_expert_sha256"] = expert_state_sha256(head)
        if args.resume:
            for key in ("frozen_reader_sha256", "frozen_expert_sha256"):
                if initial["metadata"].get(key) != metadata[key]:
                    raise ValueError(f"Stage-2 frozen parameter invariant violated: {key}")
        # Copy the complete initial memory ONCE. Its writer is unused for label
        # branch evaluation, but must stay identical across exact resumes too.
        teacher = copy.deepcopy(memory).eval().requires_grad_(False)
        teacher_source = Path(state.get("teacher_checkpoint", source))
        if args.resume:
            if hashlib.sha256((teacher_source / "model.safetensors").read_bytes()).hexdigest() != state["teacher_memory_sha256"]:
                raise ValueError("Frozen Stage-2 teacher source changed/missing")
            load_checkpoint(teacher_source, teacher)
        else:
            state["teacher_checkpoint"] = str(teacher_source.resolve())
            state["teacher_memory_sha256"] = hashlib.sha256((teacher_source / "model.safetensors").read_bytes()).hexdigest()
        teacher.action_encoder.flatten_parameters()
        label_config = LabelV3Config(memory_window=window, future_samples=args.future_samples,
                                    noise_samples=args.noise_samples, scale=args.label_scale,
                                    margin=args.label_margin, uncertainty_z=args.uncertainty_z)
        labels = ActionValueLabelsV3(teacher, head, label_config, output / "labels" / "fixed-adapted-expert",
                    {"trainer_variant": VARIANT, "cache_fingerprint": cache.manifest["fingerprint"],
                     "teacher_version": "fixed-stage1", "base_model": identity,
                     "teacher_memory_sha256": state["teacher_memory_sha256"],
                     "frozen_reader_sha256": metadata["frozen_reader_sha256"],
                     "expert_adapters_sha256": metadata["frozen_expert_sha256"]}, copy_teacher=False)
    if args.resume:
        # Restoring RNG last prevents initialization/teacher construction from
        # perturbing the next microbatch or stochastic flow-matching draw.
        load_checkpoint_v4(args.resume, memory, head, optimizer)
    else:
        _seed(args.seed + 17)
    logger = RunLogger(output)
    _atomic_json(output / "run_config.json", config)
    _atomic_json(output / "provenance.json", metadata)
    print(f"[v4] stage={args.stage} reader_mode={args.reader_mode} "
          f"trainable_reader={sum(p.numel() for p in memory.reader_parameters() if p.requires_grad)} "
          f"trainable_expert={sum(p.numel() for p in expert_parameters(head) if p.requires_grad)} "
          f"trainable_writer={sum(p.numel() for p in memory.writer_parameters() if p.requires_grad)}", flush=True)
    elapsed_before, started = state["elapsed_seconds"], time.monotonic()
    interval, last_saved = [], start

    def storage_aux(ep, candidate, seed):
        with torch.no_grad():
            bank, _ = replay_bank(memory, ep, candidate, "hard")
        label = labels.get_storage(ep, candidate, bank, seed=seed, loss_fn=_flow)
        aux = storage_loss(memory, ep, label)
        with torch.no_grad():
            logits = storage_prediction(memory, ep, candidate, bank)["logits"]
        return aux, storage_diagnostics(aux, label, logits, args.label_margin)

    def assert_frozen():
        if args.stage == 2:
            if reader_fingerprint(memory) != metadata["frozen_reader_sha256"]:
                raise RuntimeError("Stage-2 reader changed: refusing contaminated writer experiment")
            if expert_state_sha256(head) != metadata["frozen_expert_sha256"]:
                raise RuntimeError("Stage-2 Expert changed: refusing contaminated labels")

    def evaluate(step):
        assert_frozen()
        memory.eval()
        policy = "all" if args.stage == 1 else "hard"
        metrics = validate(memory, head, fetch, plans["validation"], policy, args.seed + 50000, window, args.reader_mode)
        logger.log(step, "val", metrics)
        if plans["delayed_validation"]:
            logger.log(step, "val-old-only", validate(memory, head, fetch, plans["delayed_validation"],
                       policy, args.seed + 75000, window, args.reader_mode, delayed=True))
        if args.stage == 2:
            with torch.no_grad():
                rows = [storage_aux(fetch(eid), candidate, args.seed + 90000 + index)[1]
                        for index, (eid, candidate) in enumerate(plans["storage"]["val"])]
            if rows:
                logger.log(step, "val-storage", _mean(rows))
        state["last_eval"] = metrics
        print(f"[v4][val] step={step} action={metrics['action_loss']:.7f} "
              f"original={metrics['baseline_action_loss']:.7f} fifo={metrics['fifo_action_loss']:.7f} "
              f"expert_without_memory={metrics['expert_no_memory_action_loss']:.7f}", flush=True)
        memory.train(args.stage == 1 and args.reader_mode == "memory")
        memory.writer.train(args.stage == 2)
        return metrics

    if not args.resume:
        metrics = evaluate(0)
        state["best_action_loss"] = metrics["action_loss"]
        save_checkpoint_v4(output, 0, memory, head, optimizer, config, metadata, best=True)
        logger.plot()
    memory.train(args.stage == 1 and args.reader_mode == "memory")
    memory.writer.train(args.stage == 2)
    try:
        for step in range(start + 1, args.max_steps + 1):
            update_started = time.monotonic()
            if args.stage == 2 and step > 1 and (step - 1) % args.context_refresh_steps == 0:
                state["context_version"] += 1
                refresh_storage_plan(plans, fetch, cfg, window, args, state["context_version"])
                _atomic_json(output / "sampling_plan.json", plans)
                print(f"[v4] writer contexts rotated; teacher remains fixed; version={state['context_version']}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = getattr(args, group["name"] + "_learning_rate") * min(1.0, step / max(1, args.warmup_steps))
            rows = []
            if args.stage == 1:
                for _ in range(args.grad_accum):
                    eid, decisions = random.choice(plans["train"])
                    decision = random.choice(decisions)
                    ep = fetch(eid)
                    stats, read, fused = {}, {}, None
                    if args.reader_mode == "memory":
                        encoded = encode_until(memory, ep, decision)
                        bank, stats = replay_bank(memory, ep, decision, "all", encoded)
                        read = read_bank(memory, ep, decision, bank, encoded)
                        fused = read["fused_short"]
                    action = _flow(head, ep, decision, fused, activation_checkpointing=args.activation_checkpointing)
                    if not torch.isfinite(action["loss"]) or not action["loss"].requires_grad:
                        raise FloatingPointError("Nonfinite or disconnected action loss")
                    (action["loss"] / args.grad_accum).backward()
                    row = {"action_loss": float(action["loss"].detach()), "loss": float(action["loss"].detach()),
                           "velocity_mae": float(action["velocity_mae"].detach())}
                    row.update({k: float(v) for k, v in stats.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
                    row.update({k: float(read[k].detach().mean()) for k in ("gate_mean", "read_norm", "residual_norm", "null_weight") if k in read})
                    rows.append(row)
            else:
                for _ in range(args.writer_batch_size):
                    eid, candidate = random.choice(plans["storage"]["train"])
                    aux, row = storage_aux(fetch(eid), candidate, args.seed + 31000 + eid * 1009 + candidate)
                    if not torch.isfinite(aux["loss"]) or not aux["loss"].requires_grad:
                        raise FloatingPointError("Nonfinite or disconnected writer loss")
                    (aux["loss"] / args.writer_batch_size).backward()
                    rows.append({**row, "loss": row["writer_loss"]})
            row = _mean(rows)
            row.update(reader_grad_norm=_grad_norm(memory.reader_parameters()),
                       expert_grad_norm=_grad_norm(expert_parameters(head)),
                       writer_grad_norm=_grad_norm(memory.writer_parameters()))
            norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            row.update(grad_norm=float(norm), learning_rate=optimizer.param_groups[0]["lr"],
                       update_seconds=time.monotonic() - update_started,
                       elapsed_seconds=elapsed_before + time.monotonic() - started)
            for group in optimizer.param_groups:
                row[group["name"] + "_learning_rate"] = group["lr"]
            interval.append(row)
            state.update(status="reader_expert_training" if args.stage == 1 else "writer_only_training",
                         elapsed_seconds=row["elapsed_seconds"])
            if args.stage == 2:
                state["writer_signal_batches"] += int(row["storage_signal_fraction"] > 0)
            if step == 1 or step % args.log_steps == 0 or step == args.max_steps:
                logger.log(step, "train", _mean(interval))
                interval = []
                print(f"[v4][train] stage={args.stage} step={step} action={row.get('action_loss', 'n/a')} "
                      f"writer={row.get('writer_loss', 'n/a')} reader_grad={row['reader_grad_norm']:.6g} "
                      f"expert_grad={row['expert_grad_norm']:.6g} writer_grad={row['writer_grad_norm']:.6g}", flush=True)
            best = False
            due_eval = step % args.eval_steps == 0 or step == args.max_steps
            if due_eval:
                metrics = evaluate(step)
                if metrics["action_loss"] < state["best_action_loss"]:
                    state["best_action_loss"] = metrics["action_loss"]
                    best = True
                _atomic_json(output / "status.json", {"step": step, **state})
            if step % args.save_steps == 0 or step == args.max_steps or best:
                assert_frozen()
                save_checkpoint_v4(output, step, memory, head, optimizer, config, metadata, best=best)
                last_saved = step
            if step % args.plot_steps == 0 or due_eval:
                logger.plot()
    except KeyboardInterrupt:
        print(f"[v4] Interrupted; last complete immutable checkpoint is update {last_saved}.", flush=True)
        return 130
    state["status"] = "complete"
    _atomic_json(output / "status.json", {"step": args.max_steps, **state})
    if args.stage == 2 and not state["writer_signal_batches"]:
        print("[v4][WARNING] No teacher option contrast above label-margin. More updates do not establish selective storage.", flush=True)
    print(f"[v4] Complete: {output}; offline action/choice metrics are NOT RoboMME success", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
