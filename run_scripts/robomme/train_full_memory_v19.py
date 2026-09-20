#!/usr/bin/env python3
"""Full-TRAIN matched experiment: uniform vs deployment-prefix-weighted flow.

Same V18 short/FIFO/reader/AE-LoRA architecture in both arms. The only
between-arm change is the reduction weight of future velocity errors; we never
truncate the GT/noise trajectory. Epochs cover every eligible cached TRAIN
query. VAL is held out and never used by the optimizer.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import fcntl
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch

from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes, validate_decision
from gr00t.long_memory.checkpoint_v4 import _state_sha256
from gr00t.long_memory.expert_v4 import LoRAConfig, expert_state_sha256, install_expert_lora, set_expert_trainable
from gr00t.long_memory.hamlet import checkpoint_identity, isolated_seed, load_frozen_hamlet, validate_cache_checkpoint
from gr00t.long_memory.monitoring import RunLogger, _atomic_json
from gr00t.long_memory.safety_v5 import validate_output_scope
from gr00t.long_memory.train_v3 import _Tee, _mean, _seed, runtime_identity
from gr00t.long_memory.train_v7 import lr_factor
from run_scripts.robomme.train_archive_deployment_v9 import file_hash
from run_scripts.robomme.train_representation_v18 import configuration, optimizer_groups, assert_scope
from run_scripts.robomme.representation_core_v18 import RepresentationMemoryV18
from run_scripts.robomme.checkpoint_representation_v18 import checkpoint_info_v18, save_checkpoint_v18, load_checkpoint_v18
from run_scripts.robomme.flow_objective_v19 import episode_flow_v19
from run_scripts.robomme.training_plan_v19 import build_plan_v19
from run_scripts.robomme.validation_v19 import validate_v19

DRIVER = "full_memory_v19"


def source_identity():
    # Include everything participating in training, but not unrelated eval/docs.
    paths = list((ROOT / "gr00t").rglob("*.py"))
    paths += [ROOT / "run_scripts/robomme" / name for name in (
        "train_full_memory_v19.py", "flow_objective_v19.py", "training_plan_v19.py", "validation_v19.py",
        "train_representation_v18.py", "representation_core_v18.py", "checkpoint_representation_v18.py",
        "train_archive_deployment_v9.py", "deployment_objective_v9.py", "audit_archive_generation_v7.py")]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default="runs/long_memory/cache_full1600_v1")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=1, help="Full TRAIN passes, not early-stop/best-VAL selection")
    p.add_argument("--query-batch-size", type=int, default=4)
    p.add_argument("--tail-weight", type=float, default=1., help="1=control; .25=prefix-priority treatment")
    p.add_argument("--task-weighting", choices=("macro", "query"), default="macro")
    p.add_argument("--val-per-task", type=int, default=4)
    p.add_argument("--val-noise-samples", type=int, default=2)
    p.add_argument("--memory-learning-rate", type=float, default=1e-4)
    p.add_argument("--expert-learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--warmup-fraction", type=float, default=.05)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--capacity-events", type=int, default=32)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.)
    p.add_argument("--eval-steps", type=int, default=1000)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--plot-steps", type=int, default=1000)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=9191)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--stop-after-steps", type=int, help="Explicit smoke/pause, marked INCOMPLETE (not a full run)")
    p.add_argument("--resume", help="Exact resume into a NEW directory, unchanged epoch horizon/objective")
    return p


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    explicit = {p._option_string_actions[t.split("=", 1)[0]].dest for t in argv
                if t.split("=", 1)[0] in p._option_string_actions}
    if args.resume:
        info = json.loads((Path(args.resume) / "checkpoint.json").read_text())
        if info["config"].get("driver_variant") != DRIVER:
            raise ValueError("Exact resume requires a V19 training checkpoint")
        for name, value in info["config"]["train"].items():
            if name not in explicit and name not in {"resume", "stop_after_steps", "preflight_only", "max_steps"}:
                setattr(args, name, value)
    for name in ("epochs", "query_batch_size", "val_per_task", "val_noise_samples", "eval_steps", "save_steps",
                 "plot_steps", "log_steps", "hidden_dim", "num_heads", "capacity_events", "lora_rank"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("memory_learning_rate", "expert_learning_rate", "max_grad_norm", "lora_alpha"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 < args.tail_weight <= 1 or not math.isfinite(args.tail_weight):
        raise ValueError("tail_weight must be in (0,1]")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0 or not 0 <= args.warmup_fraction < 1 or args.seed < 0:
        raise ValueError("Invalid decay/warmup/seed")
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        raise ValueError("stop_after_steps must be positive (omitting it runs all epochs)")
    # Architecture is deliberately fixed. These values also make the old,
    # tested configuration/optimizer builders reusable without mutating them.
    args.representation, args.gate = "short", "linear"
    args.short_lora_rank, args.short_lora_alpha, args.short_learning_rate = 8, 16., 1e-5
    return args


def main(argv=None):
    args = parse_args(argv)
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    if cache.manifest["action_steps"] != 16:
        raise ValueError("RoboMME execution prefix must be 16")
    base_path = str(Path(cache.manifest["model_path"]).resolve())
    cfg, expert_cfg = configuration(args, cache.manifest), LoRAConfig(args.lora_rank, args.lora_alpha)
    output = validate_output_scope(args.output_dir, cache.path, cache.manifest.get("dataset_path"), base_path, args.resume)
    if output.exists():
        raise FileExistsError("Use a NEW output directory; no old run/checkpoint is overwritten")
    episodes = MappedEpisodes(cache)
    plan, plan_sha = build_plan_v19(args, cache, episodes)
    args.max_steps = len(plan["schedule"])
    sources = source_identity()
    initial = checkpoint_info_v18(base_path, args.resume) if args.resume else None
    start = initial["step"] if initial else 0
    end = args.stop_after_steps or args.max_steps
    if not 0 <= start < end <= args.max_steps:
        raise ValueError("Pause/resume must be inside the fixed full-epoch horizon")
    if initial:
        mutable = {"resume", "output_dir", "stop_after_steps", "preflight_only"}
        for key, value in vars(args).items():
            if key not in mutable and initial["config"]["train"].get(key) != value:
                raise ValueError(f"Exact resume changed {key}; keep the original horizon/settings")
        if (initial["metadata"]["plan_sha256"] != plan_sha or initial["metadata"]["source_sha256"] != sources
                or initial["metadata"]["runtime"] != runtime_identity()):
            raise ValueError("Exact resume requires unchanged data plan, source and runtime")
    if args.preflight_only:
        print(json.dumps({"driver_variant": DRIVER, "architecture": asdict(cfg), "epochs": args.epochs,
            "planned_updates": args.max_steps, "train_queries_per_epoch": plan["train_query_count"],
            "planned_query_presentations": sum(w["query_count"] for w in plan["windows"]),
            "task_statistics": plan["task_statistics"], "task_mapping": plan["task_mapping"],
            "val_queries": len(plan["validation"]), "val_forward_seeds": len(plan["validation_schedule"]),
            "tail_weight": args.tail_weight, "plan_sha256": plan_sha,
            "note": "READ ONLY: no model, output, training or simulator started. Both arms must share the plan hash."}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    with (output / ".training.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / "training.log").open("a", encoding="utf-8") as journal:
            with redirect_stdout(_Tee(sys.stdout, journal)), redirect_stderr(_Tee(sys.stderr, journal)):
                try:
                    return run(args, cfg, expert_cfg, cache, episodes, plan, plan_sha, sources, initial, output)
                except BaseException as exc:
                    traceback.print_exc()
                    _atomic_json(output / "failure.json", {"error": type(exc).__name__, "message": str(exc)})
                    raise


def progress(plan, step, epochs):
    windows = plan["windows"][:step]
    return {"step": step, "max_steps": len(plan["schedule"]), "epochs": epochs,
        "completed_epochs": sum(bool(w["epoch_end"]) for w in windows),
        "processed_queries": sum(w["query_count"] for w in windows),
        "train_queries_per_epoch": plan["train_query_count"],
        "planned_query_presentations": sum(w["query_count"] for w in plan["windows"])}


def scalar_metrics(values):
    return {k: float(v.detach()) if isinstance(v, torch.Tensor) else float(v) for k, v in values.items()
            if (isinstance(v, torch.Tensor) and v.numel() == 1) or isinstance(v, (int, float))}


def run(args, cfg, expert_cfg, cache, episodes, plan, plan_sha, sources, initial, output):
    _seed(args.seed)
    base_path = str(Path(cache.manifest["model_path"]).resolve())
    base, processor = load_frozen_hamlet(base_path, args.device)
    head = base.action_head
    with isolated_seed(args.seed + 100, args.device):
        core = RepresentationMemoryV18(cfg).to(args.device).eval()
    with isolated_seed(args.seed + 101, args.device):
        targets = install_expert_lora(head, expert_cfg)
    del base, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    set_expert_trainable(head, True)
    assert_scope(core, head)
    optimizer = torch.optim.AdamW(optimizer_groups(args, core, head))
    core_init, expert_init = _state_sha256(core.delta_state_dict()), expert_state_sha256(head)
    if initial:
        load_checkpoint_v18(args.resume, core, head, optimizer)
    config = {"driver_variant": DRIVER, "representation": asdict(cfg), "expert": asdict(expert_cfg),
        "expert_targets": targets, "train": vars(args).copy(), "selection": "fixed_final_epoch",
        "objective": {"name": "same_trajectory_weighted_velocity", "action_steps": 16,
                      "tail_weight": args.tail_weight, "task_weighting": args.task_weighting},
        "storage": "FIFO; causal read-before-write; no learned writer in this controlled experiment"}
    metadata = {"driver_variant": DRIVER, "base_model": checkpoint_identity(base_path),
        "cache_fingerprint": cache.manifest["fingerprint"], "cache_dir": str(Path(cache.path).resolve()),
        "plan_sha256": plan_sha, "source_sha256": sources, "runtime": runtime_identity(),
        "initialization": "original_author_HAMLET_plus_fresh_reader_zero_AE_LoRA",
        "initial_core_sha256": initial["metadata"]["initial_core_sha256"] if initial else core_init,
        "initial_shared_reader_sha256": initial["metadata"]["initial_shared_reader_sha256"] if initial else core_init,
        "initial_expert_sha256": initial["metadata"]["initial_expert_sha256"] if initial else expert_init,
        "full_coverage": {"train_query_count": plan["train_query_count"], "epochs": args.epochs,
                          "total_query_presentations": sum(w["query_count"] for w in plan["windows"]), "total_steps": args.max_steps},
        "resume_parent": str(Path(args.resume).resolve()) if args.resume else None}
    _atomic_json(output / "run_config.json", config)
    _atomic_json(output / "query_plan.json", {"sha256": plan_sha, **plan})
    _atomic_json(output / "provenance.json", metadata)
    logger, baseline_cache = RunLogger(output), {}
    start = initial["step"] if initial else 0
    best, saved = (float("inf"), -1), set()

    def save(step):
        if step not in saved:
            assert_scope(core, head)
            save_checkpoint_v18(output, step, core, head, optimizer, config, metadata)
            saved.add(step)

    def status(step, state):
        row = {**progress(plan, step, args.epochs), "status": state}
        _atomic_json(output / "status.json", row)
        _atomic_json(output / "coverage.json", {**row, "task_statistics": plan["task_statistics"],
            "note": "A completed epoch contains every eligible TRAIN cache query exactly once; VAL is held out."})

    def evaluate(step):
        nonlocal best
        summaries, records, by_task = validate_v19(args, core, head, episodes, plan, baseline_cache)
        for role, split in (("reader", "val"), ("memory-off", "comparison/memory-off"), ("baseline", "comparison/baseline")):
            logger.log(step, split, summaries[role])
            for task, values in by_task[role].items():
                logger.log(step, f"{split}/task/{task}", values)
        _atomic_json(output / f"validation-{step:06d}.json", {"step": step, "summary": summaries,
            "by_task": by_task, "records": records, "aggregation": "equal task macro; same unweighted flow in both arms"})
        score = summaries["reader"]["generated_prefix_mse"]
        if score < best[0]:
            best = (score, step)
            save(step)
            _atomic_json(output / "best_checkpoint.json", {"path": f"checkpoint-{step:06d}", "step": step,
                "metric": "val/generated_prefix_mse", "value": score,
                "note": "Diagnostic only. Controlled rollout uses the fixed FINAL epoch, not per-arm best."})
        print(f"[v19][val] step={step} task_macro_flow={summaries['reader']['action_loss']:.6f} "
              f"generated_MSE={score:.6f} OFF={summaries['memory-off']['generated_prefix_mse']:.6f}; NOT success rate", flush=True)

    print(f"[v19] full TRAIN: {plan['train_query_count']} queries x {args.epochs} epochs; {args.max_steps} updates; "
          f"tail_weight={args.tail_weight}; task_weighting={args.task_weighting}", flush=True)
    print("[v19] trainable parameters:", {g["name"]: sum(p.numel() for p in g["params"]) for g in optimizer.param_groups}, flush=True)
    status(start, "running")
    if not initial:
        evaluate(0)
        save(0)
        logger.plot()
    begin, interval = time.monotonic(), []
    end = args.stop_after_steps or args.max_steps
    for step in range(start + 1, end + 1):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = getattr(args, group["kind"] + "_learning_rate") * lr_factor(step - 1, args.max_steps, args.warmup_fraction)
        items = plan["schedule"][step - 1]["queries"]
        for item in items:
            ep, decision = episodes.fetch(item["episode_id"]), item["decision"]
            validate_decision(ep, decision)
            out = core.replay(ep, decision, activation_checkpointing=args.activation_checkpointing)
            flow = episode_flow_v19(head, ep, decision, out["fused"], seed=item["flow_seed"],
                tail_weight=args.tail_weight, activation_checkpointing=args.activation_checkpointing)
            loss = flow["loss"]
            if not bool(torch.isfinite(loss)) or not loss.requires_grad:
                raise FloatingPointError("Nonfinite/disconnected action loss")
            # Global inverse task frequency. Do NOT normalize by within-batch
            # weights: that would cancel balancing in single-task minibatches.
            (loss * item["task_weight"] / len(items)).backward()
            interval.append({**scalar_metrics(out["metrics"]), **scalar_metrics(flow),
                "action_loss": float(flow["original_flow_loss"].detach()),
                "loss": float(loss.detach()) * item["task_weight"], "task_weight": item["task_weight"]})
        params = [p for g in optimizer.param_groups for p in g["params"]]
        grad = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in params):
            raise FloatingPointError("Nonfinite updated parameters; checkpoint not published")
        epoch_end = bool(plan["windows"][step - 1]["epoch_end"])
        if step % args.log_steps == 0 or step == end or epoch_end:
            row = {**_mean(interval), "grad_norm": float(grad), "learning_rate": optimizer.param_groups[0]["lr"],
                   "elapsed_seconds": time.monotonic() - begin, "completed_epochs": progress(plan, step, args.epochs)["completed_epochs"]}
            logger.log(step, "train", row)
            print(f"[v19] step={step}/{args.max_steps} objective={row['loss']:.6f} original_flow={row['action_loss']:.6f} "
                  f"grad={float(grad):.5f}", flush=True)
            interval.clear()
        if step % args.eval_steps == 0 or step == end or epoch_end:
            evaluate(step)
        if step % args.save_steps == 0 or step == end or epoch_end:
            save(step)
        if step % args.plot_steps == 0 or step == end or epoch_end:
            logger.plot()
        status(step, "complete" if step == args.max_steps else "paused" if step == end else "running")
    print(f"[v19] {'complete' if end == args.max_steps else 'paused (NOT full-coverage result)'}: "
          f"{output / f'checkpoint-{end:06d}'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
