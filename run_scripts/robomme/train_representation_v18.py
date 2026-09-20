#!/usr/bin/env python3
"""Matched representation / gate experiments, with causal episode-prefix replay.

A: frozen HAMLET short -> bank; B: LoRA-adapted short -> bank;
C: frozen short query, normalized pre-HAMLET moment -> bank.
All start from the SAME original HAMLET, fresh reader and zero AE LoRA.
Only action flow loss trains the actor. This is not a simulator-success metric.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import fcntl
import hashlib
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
from gr00t.long_memory.expert_v4 import (LoRAConfig, adapter_disabled, expert_episode_flow_loss,
    expert_parameters, expert_state_sha256, install_expert_lora, set_expert_trainable)
from gr00t.long_memory.checkpoint_v4 import _state_sha256
from gr00t.long_memory.hamlet import checkpoint_identity, isolated_seed, load_frozen_hamlet, validate_cache_checkpoint
from gr00t.long_memory.monitoring import RunLogger, _atomic_json
from gr00t.long_memory.safety_v5 import validate_output_scope
from gr00t.long_memory.train_v3 import _Tee, _mean, _seed, runtime_identity
from gr00t.long_memory.train_v7 import lr_factor
from run_scripts.robomme.train_archive_deployment_v9 import build_plan, digest, file_hash
from run_scripts.robomme.deployment_objective_v9 import generated_prefix_objective
from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18, RepresentationMemoryV18
from run_scripts.robomme.checkpoint_representation_v18 import (VARIANT, checkpoint_info_v18,
    save_checkpoint_v18, load_checkpoint_v18)


def source_identity():
    paths = list((ROOT / "gr00t").rglob("*.py")) + list((ROOT / "run_scripts/robomme").glob("*v18.py"))
    paths += [ROOT / "run_scripts/robomme" / name for name in (
        "train_archive_deployment_v9.py", "deployment_objective_v9.py", "audit_archive_generation_v7.py")]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default="runs/long_memory/cache_full1600_v1")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--representation", choices=("short", "adapted_short", "moment"), default="short")
    p.add_argument("--gate", choices=("linear", "mlp"), default="linear")
    p.add_argument("--capacity-events", type=int, default=32, help="Same observed-event budget in ALL compared arms")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--short-lora-rank", type=int, default=8)
    p.add_argument("--short-lora-alpha", type=float, default=16.)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.)
    p.add_argument("--max-steps", type=int, default=1000, help="Fixed optimizer horizon, not number of episodes")
    p.add_argument("--stop-after-steps", type=int, help="Pause without changing the horizon/LR schedule")
    p.add_argument("--resume", help="EXACT V18 resume into a NEW directory; no weights-only cross-arm initialization")
    p.add_argument("--query-batch-size", type=int, default=4)
    p.add_argument("--memory-learning-rate", type=float, default=1e-4)
    p.add_argument("--short-learning-rate", type=float, default=1e-5)
    p.add_argument("--expert-learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--warmup-fraction", type=float, default=.05)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--val-samples", type=int, default=32)
    p.add_argument("--val-noise-samples", type=int, default=1)
    p.add_argument("--eval-steps", type=int, default=250)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--plot-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=9181)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    return p


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    explicit = {p._option_string_actions[t.split("=", 1)[0]].dest for t in argv
                if t.split("=", 1)[0] in p._option_string_actions}
    if args.resume:
        info = json.loads((Path(args.resume) / "checkpoint.json").read_text())
        if info.get("variant") != VARIANT:
            raise ValueError("Exact resume requires V18, not an older memory architecture")
        for name, value in info["config"]["train"].items():
            if name not in explicit and name not in {"resume", "stop_after_steps", "preflight_only"}:
                setattr(args, name, value)
    for name in ("max_steps", "query_batch_size", "val_samples", "val_noise_samples", "eval_steps", "save_steps", "log_steps", "plot_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("memory_learning_rate", "short_learning_rate", "expert_learning_rate", "max_grad_norm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0 or not 0 <= args.warmup_fraction < 1 or args.seed < 0:
        raise ValueError("Invalid decay/warmup/seed")
    if args.stop_after_steps is not None and not 0 < args.stop_after_steps <= args.max_steps:
        raise ValueError("Pause must be inside the fixed total horizon")
    return args


def configuration(args, manifest):
    base = json.loads((Path(manifest["model_path"]) / "config.json").read_text())
    if (base.get("hamlet_mode") != "finetune" or base.get("mem_cond_type", "cross_attn") != "cross_attn"
            or base.get("memory_type", "moment_token") != "moment_token"):
        raise ValueError("Require original moment-token/cross-attention HAMLET")
    if manifest["feature_dim"] != base["backbone_embedding_dim"]:
        raise ValueError("Cache feature dimension differs from base")
    return RepresentationConfigV18(feature_dim=manifest["feature_dim"], state_dim=manifest["state_dim"],
        num_short_tokens=base["n_moment_tokens"], short_window=base["memory_window"],
        time_scale=float(base.get("memory_stride", 16)), hidden_dim=args.hidden_dim, num_heads=args.num_heads,
        capacity_events=args.capacity_events, representation=args.representation, gate=args.gate,
        short_lora_rank=args.short_lora_rank, short_lora_alpha=args.short_lora_alpha)


def optimizer_groups(args, core, head):
    groups = []
    for name, params in (("memory", core.reader_parameters()), ("short", core.short_parameters()),
                         ("expert", expert_parameters(head))):
        params = [p for p in params if p.requires_grad]
        for decay in (False, True):
            selected = [p for p in params if (p.ndim > 1) == decay]
            if selected:
                groups.append({"name": f"{name}_{'decay' if decay else 'no_decay'}", "kind": name,
                    "params": selected, "lr": getattr(args, name + "_learning_rate"),
                    "weight_decay": args.weight_decay if decay else 0.})
    ids = [id(p) for g in groups for p in g["params"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Optimizer has duplicate parameters")
    return groups


def assert_scope(core, head):
    allowed = {id(p) for p in expert_parameters(head)}
    if core.training or head.training or any(p.requires_grad and id(p) not in allowed for p in head.parameters()):
        raise RuntimeError("Original HAMLET must remain eval/frozen; only external core and AE LoRA may train")
    if not all(p.requires_grad for p in expert_parameters(head)):
        raise RuntimeError("AE LoRA is unexpectedly frozen")


@torch.no_grad()
def validate(args, core, head, episodes, plan, baseline_cache):
    records = []
    for item in plan["validation_schedule"]:
        eid, d = item["episode_id"], item["decision"]
        ep = episodes.fetch(eid)
        validate_decision(ep, d)
        out = core.replay(ep, d)
        for role, conditioning in (("reader", out["fused"]), ("memory-off", out["short"]), ("baseline", None)):
            # OFF retains THIS arm's adapted short summary and AE; original is separate.
            key = (eid, d, item["flow_seed"], item["generation_seed"])
            if role == "baseline" and key in baseline_cache:
                metrics = baseline_cache[key]
            else:
                from contextlib import nullcontext
                with adapter_disabled(head) if role == "baseline" else nullcontext():
                    flow = expert_episode_flow_loss(head, ep, d, conditioning, seed=item["flow_seed"])
                    generated = generated_prefix_objective(head, ep, d, conditioning,
                        seed=item["generation_seed"], action_steps=16, activation_checkpointing=False)
                metrics = {"action_loss": float(flow["loss"]), "loss": float(flow["loss"]),
                    "velocity_mae": float(flow["velocity_mae"]),
                    "generated_prefix_mse": float(generated["loss"]),
                    "generated_prefix_mae": float(generated["generated_prefix_mae"])}
                if role == "baseline":
                    baseline_cache[key] = metrics
            records.append({**item, "role": role, **metrics})
    summaries = {role: _mean([{k: row[k] for k in ("action_loss", "loss", "velocity_mae", "generated_prefix_mse", "generated_prefix_mae")}
                            for row in records if row["role"] == role]) for role in ("reader", "memory-off", "baseline")}
    summaries["reader"]["memory_gain"] = summaries["memory-off"]["action_loss"] - summaries["reader"]["action_loss"]
    return summaries, records


def main(argv=None):
    args = parse_args(argv)
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    if cache.manifest["action_steps"] != 16:
        raise ValueError("V18 RoboMME protocol fixes the execution prefix to 16")
    base_path = str(Path(cache.manifest["model_path"]).resolve())
    cfg = configuration(args, cache.manifest)
    expert_cfg = LoRAConfig(args.lora_rank, args.lora_alpha)
    output = validate_output_scope(args.output_dir, cache.path, cache.manifest.get("dataset_path"), base_path, args.resume)
    if output.exists():
        raise FileExistsError("Use a NEW output directory; old runs/checkpoints are never overwritten")
    episodes = MappedEpisodes(cache)
    plan, plan_sha = build_plan(args, cache, episodes)
    sources = source_identity()
    initial = checkpoint_info_v18(base_path, args.resume) if args.resume else None
    start = initial["step"] if initial else 0
    if start >= args.max_steps or (args.stop_after_steps is not None and args.stop_after_steps <= start):
        raise ValueError("Requested horizon/pause has already been reached")
    if initial:
        mutable = {"resume", "output_dir", "stop_after_steps", "preflight_only"}
        for key, value in vars(args).items():
            if key not in mutable and initial["config"]["train"].get(key) != value:
                raise ValueError(f"Exact resume changed {key}; use original settings")
        if (initial["metadata"]["plan_sha256"] != plan_sha or initial["metadata"]["source_sha256"] != sources
                or initial["metadata"]["runtime"] != runtime_identity()):
            raise ValueError("Exact resume requires unchanged data plan, source and runtime")
    if args.preflight_only:
        print(json.dumps({"variant": VARIANT, "representation": asdict(cfg), "planned_updates": args.max_steps,
            "train_queries_per_epoch": plan["train_query_count"], "planned_query_presentations": sum(w["query_count"] for w in plan["windows"]),
            "val_queries": len(plan["validation"]), "plan_sha256": plan_sha,
            "note": "READ ONLY: no model, simulator, output or training started. Shared plan hash must match all arms."}, indent=2))
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


def run(args, cfg, expert_cfg, cache, episodes, plan, plan_sha, sources, initial, output):
    _seed(args.seed)
    base_path = str(Path(cache.manifest["model_path"]).resolve())
    base, processor = load_frozen_hamlet(base_path, args.device)
    head = base.action_head
    # Independent streams keep common reader/AE initialization identical in A/B/C.
    with isolated_seed(args.seed + 100, args.device):
        core = RepresentationMemoryV18(cfg, base_memory_transformer=head.memory_transformer
            if cfg.representation == "adapted_short" else None).to(args.device).eval()
    with isolated_seed(args.seed + 101, args.device):
        targets = install_expert_lora(head, expert_cfg)
    del base, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    set_expert_trainable(head, True)
    assert_scope(core, head)
    optimizer = torch.optim.AdamW(optimizer_groups(args, core, head))
    shared_init = _state_sha256({k: v for k, v in core.delta_state_dict().items()
                                if k.startswith("memory.") and not k.startswith("memory.fusion_gate.")})
    expert_init = expert_state_sha256(head)
    if initial:
        load_checkpoint_v18(args.resume, core, head, optimizer)
    config = {"representation": asdict(cfg), "expert": asdict(expert_cfg), "expert_targets": targets,
        "train": vars(args).copy(), "objective": "original_GT_action_flow_only", "selection": "fixed_final_step",
        "storage": "FIFO; read before write; no learned writer at this phase"}
    metadata = {"base_model": checkpoint_identity(base_path), "cache_fingerprint": cache.manifest["fingerprint"],
        "cache_dir": str(Path(cache.path).resolve()), "plan_sha256": plan_sha, "source_sha256": sources,
        "runtime": runtime_identity(), "initialization": "original_author_HAMLET_plus_fresh_reader_zero_AE_LoRA",
        "initial_shared_reader_sha256": initial["metadata"]["initial_shared_reader_sha256"] if initial else shared_init,
        "initial_expert_sha256": initial["metadata"]["initial_expert_sha256"] if initial else expert_init,
        "initial_core_sha256": digest({k: hashlib.sha256(v.cpu().contiguous().numpy().tobytes()).hexdigest()
                                        for k, v in core.delta_state_dict().items()}),
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
    def evaluate(step):
        nonlocal best
        summaries, records = validate(args, core, head, episodes, plan, baseline_cache)
        for role, split in (("reader", "val"), ("memory-off", "comparison/memory-off"), ("baseline", "comparison/baseline")):
            logger.log(step, split, summaries[role])
        _atomic_json(output / f"validation-{step:06d}.json", {"step": step, "summary": summaries, "records": records})
        score = summaries["reader"]["generated_prefix_mse"]
        if score < best[0]:
            best = (score, step)
            save(step)
            _atomic_json(output / "best_checkpoint.json", {"path": f"checkpoint-{step:06d}", "step": step,
                "metric": "val/generated_prefix_mse", "value": score, "note": "Diagnostic; paired pilot uses fixed final, not per-arm TEST selection."})
        print(f"[v18][val] step={step} flow={summaries['reader']['action_loss']:.6f} generated_MSE={score:.6f} "
              f"OFF={summaries['memory-off']['generated_prefix_mse']:.6f}; not task accuracy", flush=True)
    print(f"[v18] {cfg.representation}/{cfg.gate}; capacity={cfg.capacity_events} events; full causal prefix; {args.max_steps} updates", flush=True)
    print("[v18] trainable parameters:", {g["name"]: sum(p.numel() for p in g["params"]) for g in optimizer.param_groups}, flush=True)
    if not initial:
        evaluate(0)
        save(0)
        logger.plot()
    begin = time.monotonic()
    end = args.stop_after_steps or args.max_steps
    interval = []
    for step in range(start + 1, end + 1):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = getattr(args, group["kind"] + "_learning_rate") * lr_factor(step - 1, args.max_steps, args.warmup_fraction)
        items = plan["schedule"][step - 1]["queries"]
        for item in items:
            ep, d = episodes.fetch(item["episode_id"]), item["decision"]
            validate_decision(ep, d)
            out = core.replay(ep, d, activation_checkpointing=args.activation_checkpointing)
            flow = expert_episode_flow_loss(head, ep, d, out["fused"], seed=item["flow_seed"],
                activation_checkpointing=args.activation_checkpointing)
            loss = flow["loss"]
            if not bool(torch.isfinite(loss)) or not loss.requires_grad:
                raise FloatingPointError("Nonfinite/disconnected action loss")
            (loss / len(items)).backward()
            metrics = {k: float(v.detach()) if isinstance(v, torch.Tensor) else float(v) for k, v in out["metrics"].items()
                       if (isinstance(v, torch.Tensor) and v.numel() == 1) or isinstance(v, (int, float))}
            interval.append({**metrics, "action_loss": float(loss.detach()), "loss": float(loss.detach()),
                             "velocity_mae": float(flow["velocity_mae"].detach())})
        params = [p for g in optimizer.param_groups for p in g["params"]]
        grad = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in params):
            raise FloatingPointError("Nonfinite updated parameters; checkpoint not published")
        if step % args.log_steps == 0 or step == end:
            row = {**_mean(interval), "grad_norm": float(grad), "learning_rate": optimizer.param_groups[0]["lr"],
                   "elapsed_seconds": time.monotonic() - begin}
            logger.log(step, "train", row)
            print(f"[v18] step={step}/{args.max_steps} loss={row['loss']:.6f} grad={float(grad):.5f}", flush=True)
            interval.clear()
        if step % args.eval_steps == 0 or step == end:
            evaluate(step)
        if step % args.save_steps == 0 or step == end:
            save(step)
        if step % args.plot_steps == 0 or step == end:
            logger.plot()
        _atomic_json(output / "status.json", {"step": step, "max_steps": args.max_steps,
            "status": "complete" if step == args.max_steps else "paused" if step == end else "running",
            "processed_queries": sum(w["query_count"] for w in plan["windows"][:step])})
    print(f"[v18] {'complete' if end == args.max_steps else 'paused'}: {output / f'checkpoint-{end:06d}'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
