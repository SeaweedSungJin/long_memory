#!/usr/bin/env python3
"""Two-phase metadata-answer + operation-CVoM learning on the V19 actor.

Stage 1: FIFO, joint encoder/reader/fusion/AE-LoRA and answer head.
Stage 2: counterfactual controller bootstrap, then joint actor updates under
actual hard-controlled banks; refresh fixed teacher labels periodically.
No ground-truth metadata is ever passed to the policy or storage manager.
Full TRAIN epoch coverage reuses V19 plans. Every output must be NEW.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import fcntl
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch

from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes
from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora, set_expert_trainable
from gr00t.long_memory.hamlet import load_frozen_hamlet, validate_cache_checkpoint, checkpoint_identity, isolated_seed
from gr00t.long_memory.monitoring import RunLogger, _atomic_json
from gr00t.long_memory.safety_v5 import validate_output_scope
from gr00t.long_memory.train_v3 import _Tee, _mean, _seed, runtime_identity
from gr00t.long_memory.train_v7 import lr_factor
from run_scripts.robomme.train_full_memory_v19 import parser as actor_parser, source_identity as actor_sources, progress
from run_scripts.robomme.train_representation_v18 import optimizer_groups, assert_scope
from run_scripts.robomme.checkpoint_representation_v18 import checkpoint_info_v18, load_checkpoint_v18
from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18, RepresentationMemoryV18
from run_scripts.robomme.training_plan_v19 import build_plan_v19, resolve_tasks_v19
from run_scripts.robomme.flow_objective_v19 import episode_flow_v19
from run_scripts.robomme.validation_v19 import validate_v19
from run_scripts.robomme.train_archive_deployment_v9 import file_hash
from run_scripts.robomme.semantic_memory_targets import SemanticTargets
from run_scripts.robomme.semantic_memory_core import AnswerSuite, semantic_replay, ReplayWithWriter
from run_scripts.robomme.semantic_memory_checkpoint import semantic_info, load_extras, save_semantic
from run_scripts.robomme.semantic_memory_storage import StorageConfig, StorageManager
from run_scripts.robomme.semantic_memory_teacher import context_plan, label_contexts, controller_loss

DRIVER = "semantic_memory_v1"


def parser():
    p = actor_parser()
    p.description = __doc__
    p.set_defaults(tail_weight=.25, memory_learning_rate=3e-5, expert_learning_rate=3e-6,
                   val_per_task=2, val_noise_samples=1, seed=9211)
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument("--targets-dir", required=True)
    p.add_argument("--init-checkpoint", default="runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072")
    p.add_argument("--answer-weight", type=float, default=.01)
    p.add_argument("--answer-learning-rate", type=float, default=1e-4)
    p.add_argument("--answer-hidden-dim", type=int, default=128)
    p.add_argument("--answer-val-samples", type=int, default=128)
    p.add_argument("--storage-policy", choices=("cvom", "fifo"), default="cvom",
                   help="Stage2 equal-budget FIFO control; stage1 always FIFO")
    p.add_argument("--writer-learning-rate", type=float, default=1e-4)
    p.add_argument("--writer-batch-size", type=int, default=4)
    p.add_argument("--writer-bootstrap-steps", type=int, default=100)
    p.add_argument("--storage-contexts", type=int, default=128)
    p.add_argument("--val-storage-contexts", type=int, default=16)
    p.add_argument("--context-refresh-steps", type=int, default=1500)
    p.add_argument("--teacher-noise-samples", type=int, default=2)
    p.add_argument("--teacher-candidates", type=int, default=5)
    p.add_argument("--utility-margin", type=float, default=1e-6)
    p.add_argument("--smoke-only", action="store_true", help="Two updates/very small validation; INCOMPLETE pipeline test, not a trained model")
    return p


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    explicit = {p._option_string_actions[token.split("=", 1)[0]].dest for token in argv
                if token.split("=", 1)[0] in p._option_string_actions}
    if args.resume:
        previous = json.loads((Path(args.resume) / "checkpoint.json").read_text())
        if previous["config"].get("driver_variant") != DRIVER:
            raise ValueError("Exact resume requires semantic-memory checkpoint")
        for key, value in previous["config"]["train"].items():
            if key not in explicit and key not in {"resume", "output_dir", "stop_after_steps", "preflight_only", "max_steps"}:
                setattr(args, key, value)
    for name in ("epochs", "query_batch_size", "val_per_task", "val_noise_samples", "eval_steps", "save_steps",
                 "plot_steps", "log_steps", "answer_hidden_dim", "answer_val_samples", "writer_batch_size",
                 "storage_contexts", "val_storage_contexts", "context_refresh_steps", "teacher_noise_samples", "teacher_candidates"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("memory_learning_rate", "expert_learning_rate", "answer_learning_rate", "writer_learning_rate",
                 "max_grad_norm", "utility_margin"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (not math.isfinite(args.answer_weight) or args.answer_weight < 0 or args.writer_bootstrap_steps < 0
            or args.teacher_candidates < 3 or not 0 < args.tail_weight <= 1 or not 0 <= args.warmup_fraction < 1
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0 or args.seed < 0):
        raise ValueError("Invalid semantic learning options")
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        raise ValueError("stop_after_steps must be positive")
    args.short_learning_rate = 1e-5
    if args.smoke_only:
        if args.resume:
            raise ValueError("Smoke runs are diagnostics, not resumable full training")
        args.stop_after_steps, args.val_per_task, args.val_noise_samples = 2, 1, 1
        args.answer_val_samples = 8
        # Two alphabetically first contexts were too narrow even for a wiring
        # check. Cover overflowing task groups; never weaken label uncertainty.
        args.storage_contexts, args.val_storage_contexts = 16, 4
        args.writer_bootstrap_steps, args.writer_batch_size, args.teacher_candidates = 2, 1, 3
        args.log_steps = 1
    return args


def sources():
    result = actor_sources()
    for path in sorted((ROOT / "run_scripts/robomme").glob("*semantic_memory*.py")):
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def gpu_guard(device):
    """Read current usage before starting CUDA; never stop someone else's job."""
    if not device.startswith("cuda"):
        return
    import os
    import subprocess
    value = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                     "--format=csv,noheader"], text=True)
    if value.strip():
        raise RuntimeError("GPU compute process already present; inspect before launching (no process was stopped):\n" + value)
    print("[gpu] No active GPU compute processes; CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"), flush=True)


def main(argv=None):
    args = parse_args(argv)
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    if cache.manifest["action_steps"] != 16:
        raise ValueError("Execution interval must remain 16")
    targets = SemanticTargets(args.targets_dir, cache.manifest)
    parent = args.resume or args.init_checkpoint
    initial = checkpoint_info_v18(cache.manifest["model_path"], parent)
    if initial["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]:
        raise ValueError("Initialization/cache mismatch")
    cfg = RepresentationConfigV18(**initial["config"]["representation"])
    if cfg.representation != "short" or cfg.gate != "linear" or cfg.capacity_events != 32:
        raise ValueError("This experiment fixes V19 short/linear/FIFO32 architecture")
    for key, expected in (("hidden_dim", cfg.hidden_dim), ("num_heads", cfg.num_heads),
                          ("capacity_events", cfg.capacity_events),
                          ("lora_rank", initial["config"]["expert"]["rank"]),
                          ("lora_alpha", initial["config"]["expert"]["alpha"])):
        if getattr(args, key) != expected:
            raise ValueError(f"{key} must match the initialization checkpoint ({expected})")
    extra = semantic_info(parent, required=False)
    if args.stage == 2 and (extra is None or (not args.resume and extra["stage"] != 1)):
        raise ValueError("Stage2 initializes from a completed semantic Stage1 checkpoint")
    if (args.stage == 2 and not args.resume and not args.smoke_only
            and initial["step"] != initial["config"]["train"]["max_steps"]):
        raise ValueError("Stage1 is incomplete; finish/resume it before Stage2")
    output = validate_output_scope(args.output_dir, cache.path, cache.manifest.get("dataset_path"),
                                   cache.manifest["model_path"], parent, args.targets_dir)
    if output.exists():
        raise FileExistsError("Use a NEW output directory; no previous results are overwritten")
    episodes = MappedEpisodes(cache)
    plan, plan_sha = build_plan_v19(args, cache, episodes)
    args.max_steps = len(plan["schedule"])
    start, end = initial["step"] if args.resume else 0, args.stop_after_steps or args.max_steps
    if not 0 <= start < end <= args.max_steps:
        raise ValueError("Pause/resume is outside the fixed epoch horizon")
    source = sources()
    target_hash = file_hash(Path(args.targets_dir) / "manifest.json")
    if extra and extra["metadata"].get("targets_manifest_sha256") != target_hash:
        raise ValueError("Semantic target contract differs from parent")
    if args.resume:
        allowed = {"resume", "output_dir", "stop_after_steps", "preflight_only"}
        for key, value in vars(args).items():
            if key not in allowed and initial["config"]["train"].get(key) != value:
                raise ValueError(f"Exact resume changed {key}")
        if (source != initial["metadata"]["source_sha256"] or plan_sha != initial["metadata"]["plan_sha256"]
                or runtime_identity() != initial["metadata"]["runtime"]):
            raise ValueError("Resume requires identical code, plan and runtime")
    if args.preflight_only:
        print(json.dumps({"driver": DRIVER, "stage": args.stage, "architecture": asdict(cfg),
            "initial_checkpoint": str(Path(parent).resolve()), "train_queries_per_epoch": plan["train_query_count"],
            "planned_updates": args.max_steps, "end_step": end, "plan_sha256": plan_sha,
            "targets_manifest_sha256": target_hash, "answer_weight": args.answer_weight,
            "storage": "FIFO" if args.stage == 1 or args.storage_policy == "fifo" else "append until32; operation_CVoM KEEP/REPLACE",
            "merge": "disabled pending semantic preservation evidence", "metadata_at_inference": False,
            "note": "READ ONLY; no model, GPU, training or output created"}, indent=2))
        return 0
    gpu_guard(args.device)
    output.mkdir(parents=True, exist_ok=False)
    with (output / ".training.lock").open("a+") as lock, (output / "training.log").open("a") as stream:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with redirect_stdout(_Tee(sys.stdout, stream)), redirect_stderr(_Tee(sys.stderr, stream)):
            try:
                return run(args, cache, episodes, targets, target_hash, cfg, initial, extra,
                           plan, plan_sha, source, output, start, end)
            except BaseException as exc:
                traceback.print_exc()
                _atomic_json(output / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
                raise


def metrics(values):
    return {key: float(value.detach()) if torch.is_tensor(value) else float(value)
            for key, value in values.items() if isinstance(value, (float, int)) or torch.is_tensor(value) and value.numel() == 1}


def run(args, cache, episodes, targets, target_hash, cfg, initial, extra,
        plan, plan_sha, source, output, start, end):
    _seed(args.seed)
    base, processor = load_frozen_hamlet(cache.manifest["model_path"], args.device)
    head = base.action_head
    core = RepresentationMemoryV18(cfg).to(args.device).eval()
    expert_cfg = LoRAConfig(**initial["config"]["expert"])
    installed = install_expert_lora(head, expert_cfg, targets=initial["config"]["expert_targets"])
    load_checkpoint_v18(args.resume or args.init_checkpoint, core, head)
    del base, processor
    torch.cuda.empty_cache() if args.device.startswith("cuda") else None
    set_expert_trainable(head, True)
    answer_config = targets.answer_config(input_dim=cfg.hidden_dim, hidden_dim=args.answer_hidden_dim)
    with isolated_seed(args.seed + 17, args.device):
        answers = AnswerSuite(answer_config).to(args.device).eval()
        manager = StorageManager(StorageConfig(capacity_events=cfg.capacity_events,
            num_tokens=cfg.num_short_tokens, dim=cfg.hidden_dim)).to(args.device).eval() \
            if args.stage == 2 and args.storage_policy == "cvom" else None
    if extra:
        if extra["answer_config"] != answer_config:
            raise ValueError("Answer-head architecture changed")
        load_extras(args.resume or args.init_checkpoint, answers, manager if args.resume else None)
    groups = optimizer_groups(args, core, head)
    groups.append({"name": "answers", "kind": "answer", "params": list(answers.parameters()),
                   "lr": args.answer_learning_rate, "weight_decay": args.weight_decay})
    if manager is not None:
        groups.append({"name": "writer", "kind": "writer", "params": list(manager.parameters()),
                       "lr": args.writer_learning_rate, "weight_decay": args.weight_decay})
    optimizer = torch.optim.AdamW(groups)
    if args.resume:
        load_checkpoint_v18(args.resume, core, head, optimizer)
    assert_scope(core, head)
    config = {"driver_variant": DRIVER, "representation": asdict(cfg), "expert": asdict(expert_cfg),
        "expert_targets": installed, "train": vars(args).copy(), "selection": "fixed_final_epoch",
        "objective": "V19 prefix-flow + masked memory-answer; detached current/clock controls",
        "storage": "operation_CVoM" if manager else "FIFO", "merge_enabled": False}
    metadata = {"base_model": checkpoint_identity(cache.manifest["model_path"]),
        "cache_fingerprint": cache.manifest["fingerprint"], "plan_sha256": plan_sha,
        "cache_dir": str(Path(cache.path).resolve()), "source_sha256": source, "runtime": runtime_identity(),
        "initialization": {"path": str(Path(args.init_checkpoint).resolve()),
                           "checkpoint_sha256": file_hash(Path(args.init_checkpoint) / "checkpoint.json")},
        "labels_at_inference": False, "targets_manifest_sha256": target_hash,
        "native_rollout_default": True, "cache_precision": "unchanged legacy cache; no precision intervention"}
    for name, data in (("run_config.json", config), ("provenance.json", metadata),
                       ("query_plan.json", {"sha256": plan_sha, **plan})):
        _atomic_json(output / name, data)
    logger, baseline_cache, saved = RunLogger(output), {}, set()
    teacher_pack, teacher_path, teacher_step = None, None, None
    teacher_val_pack, teacher_val_path = None, None
    if args.resume and manager is not None:
        record = extra["metadata"]["teacher_labels"]
        teacher_path = Path(record["path"])
        if file_hash(teacher_path) != record["sha256"]:
            raise ValueError("Resume teacher label journal changed")
        teacher_pack = json.loads(teacher_path.read_text())
        teacher_step = record["step"]
        val_record = extra["metadata"].get("teacher_val_labels")
        if val_record:
            teacher_val_path = Path(val_record["path"])
            if file_hash(teacher_val_path) != val_record["sha256"]:
                raise ValueError("Resume validation teacher journal changed")
            teacher_val_pack = json.loads(teacher_val_path.read_text())
    tasks, _ = resolve_tasks_v19(cache.manifest)
    contexts = context_plan(cache, episodes, tasks, cfg.capacity_events, args.storage_contexts,
        args.seed+71, short_window=cfg.short_window) if manager is not None else []
    val_contexts = context_plan(cache, episodes, tasks, cfg.capacity_events, args.val_storage_contexts,
        args.seed+72, split="val", short_window=cfg.short_window) if manager is not None else []
    _atomic_json(output / "storage_context_plan.json", {"train": contexts, "val": val_contexts})

    def save(step):
        if step in saved:
            return
        details = {"targets_manifest_sha256": target_hash, "heads": ["memory", "current", "clock"],
            "smoke_only": args.smoke_only,
            "merge_status": "not enabled; requires separate content-preservation validation"}
        if teacher_path is not None:
            details["teacher_labels"] = {"path": str(teacher_path.resolve()), "sha256": file_hash(teacher_path), "step": teacher_step}
        if teacher_val_path is not None:
            details["teacher_val_labels"] = {"path": str(teacher_val_path.resolve()), "sha256": file_hash(teacher_val_path)}
        save_semantic(output, step, core, head, optimizer, config, metadata, answers, answer_config, manager, details)
        saved.add(step)

    @torch.no_grad()
    def evaluate(step):
        summary, records, by_task = validate_v19(args, ReplayWithWriter(core, manager), head, episodes, plan, baseline_cache)
        for role, split in (("reader", "val"), ("memory-off", "comparison/memory-off"), ("baseline", "comparison/baseline")):
            logger.log(step, split, summary[role])
        _atomic_json(output / f"validation-{step:06d}.json", {"summary": summary, "records": records, "by_task": by_task})
        answer_rows = []
        # Fixed eligible VAL inventory, independent of outcome and checkpoint.
        candidates = []
        for eid in sorted(cache.manifest["splits"]["val"]):
            ep = episodes.fetch(int(eid))
            for d in torch.where(ep["decision_mask"])[0].tolist():
                target = targets.get(int(eid), d)
                if target and (target.get("classification") or target.get("regression")):
                    candidates.append((int(eid), d))
        random.Random(args.seed+301).shuffle(candidates)
        for eid, d in candidates[:args.answer_val_samples]:
            ep, target = episodes.fetch(eid), targets.get(eid, d)
            out = semantic_replay(core, ep, d, manager=manager)
            if not out["has_read"]:
                continue
            _, mm = answers.loss(out["retrieved"], target)
            (_, cm), (_, tm) = answers.controls(out["encoded_current"], ep["frames"][d], ep["is_demo"][d], target)
            _, nm = answers.loss(torch.zeros_like(out["retrieved"]), target)
            for role, values in (("memory", mm), ("current-only", cm), ("clock-only", tm), ("null-memory", nm)):
                answer_rows.append({"episode_id": eid, "decision": d, "role": role, "metrics": metrics(values)})
        for role in ("memory", "current-only", "clock-only", "null-memory"):
            rows = [r["metrics"] for r in answer_rows if r["role"] == role]
            if rows:
                logger.log(step, "answers/"+role, _mean(rows))
        _atomic_json(output / f"answers-validation-{step:06d}.json", {"records": answer_rows,
            "note": "Answer metrics are not robot success. Current/clock are independently trained matched-capacity controls; null is an input intervention."})
        if manager is not None and teacher_val_pack is not None:
            _, writer_metrics = controller_loss(core, manager, episodes, teacher_val_pack, teacher_val_pack["contexts"])
            writer_metrics["teacher_step"] = float(teacher_step)
            logger.log(step, "writer/val", writer_metrics)
        print(f"[semantic][val] step={step} flow={summary['reader']['action_loss']:.6f} "
              f"generated_MSE={summary['reader']['generated_prefix_mse']:.6f}; NOT rollout success", flush=True)

    def refresh(step):
        nonlocal teacher_pack, teacher_path, teacher_step, teacher_val_pack, teacher_val_path
        print(f"[semantic] fixed-teacher CVoM refresh at step={step}; contexts={len(contexts)}", flush=True)
        kwargs = dict(seed=args.seed+500+step, noise_samples=args.teacher_noise_samples,
            candidate_count=args.teacher_candidates, answer_weight=args.answer_weight,
            tail_weight=args.tail_weight, ambiguity_margin=args.utility_margin)
        teacher_pack = label_contexts(core, head, answers, targets, manager, episodes, contexts, **kwargs)
        teacher_path, teacher_step = output / f"teacher-{step:06d}.json", step
        _atomic_json(teacher_path, teacher_pack)
        if teacher_pack["informative_operations"] == 0:
            raise RuntimeError("No informative signed CVoM operation labels. Saved audit; do not fit an arbitrary writer.")
        teacher_val_pack = label_contexts(core, head, answers, targets, manager, episodes, val_contexts, **kwargs)
        teacher_val_path = output / f"teacher-val-{step:06d}.json"
        _atomic_json(teacher_val_path, teacher_val_pack)

    if manager is not None and teacher_pack is None:
        refresh(start)
        for update in range(args.writer_bootstrap_steps):
            optimizer.zero_grad(set_to_none=True)
            available = [row for row in teacher_pack["contexts"] if any(row["informative"])]
            rows = random.Random(args.seed+update).sample(available, min(args.writer_batch_size, len(available)))
            loss, values = controller_loss(core, manager, episodes, teacher_pack, rows)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(manager.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()  # actor/answer grads are None; only manager updates.
            if (update+1) % args.log_steps == 0:
                logger.log(update+1, "writer/bootstrap", values)
    if not args.resume:
        evaluate(0)
        save(0)
        logger.plot()
    begin, interval = time.monotonic(), []
    for step in range(start+1, end+1):
        if manager is not None and step > 1 and (step-1) % args.context_refresh_steps == 0:
            refresh(step-1)
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = getattr(args, group["kind"]+"_learning_rate") * lr_factor(step-1, args.max_steps, args.warmup_fraction)
        items = plan["schedule"][step-1]["queries"]
        for item in items:
            ep, d = episodes.fetch(item["episode_id"]), item["decision"]
            out = semantic_replay(core, ep, d, manager=manager, activation_checkpointing=args.activation_checkpointing)
            flow = episode_flow_v19(head, ep, d, out["fused"], seed=item["flow_seed"],
                tail_weight=args.tail_weight, activation_checkpointing=args.activation_checkpointing)
            target = targets.get(item["episode_id"], d) if out["has_read"] else None
            aux, am = answers.loss(out["retrieved"], target)
            (current, _), (clock, _) = answers.controls(out["encoded_current"], ep["frames"][d], ep["is_demo"][d], target)
            loss = flow["loss"] + args.answer_weight*(aux+current+clock)
            if not bool(torch.isfinite(loss)) or not loss.requires_grad:
                raise FloatingPointError("Nonfinite/disconnected training loss")
            (loss * item["task_weight"] / len(items)).backward()
            interval.append({**metrics(out["metrics"]), **metrics(am), "action_loss": float(flow["original_flow_loss"].detach()),
                "answer_loss": float(aux.detach()), "loss": float(loss.detach()), "task_weight": item["task_weight"]})
        if manager is not None:
            available = [row for row in teacher_pack["contexts"] if any(row["informative"])]
            rows = random.Random(args.seed+step*41).sample(available, min(args.writer_batch_size, len(available)))
            writer_loss, values = controller_loss(core, manager, episodes, teacher_pack, rows)
            writer_loss.backward()
            interval.append(values)
        params = [p for group in optimizer.param_groups for p in group["params"]]
        grad = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        assert_scope(core, head)
        if any(not bool(torch.isfinite(p).all()) for p in params):
            raise FloatingPointError("Nonfinite updated parameters")
        if step % args.log_steps == 0 or step == end:
            row = {**_mean(interval), "grad_norm": float(grad), "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.monotonic()-begin}
            logger.log(step, "train", row)
            print(f"[semantic] stage={args.stage} step={step}/{args.max_steps} action={row['action_loss']:.6f} "
                  f"answer={row['answer_loss']:.6f} grad={float(grad):.4f}", flush=True)
            interval.clear()
        if step % args.eval_steps == 0 or step == end:
            evaluate(step)
        if step % args.save_steps == 0 or step == end:
            save(step)
        if step % args.plot_steps == 0 or step == end:
            logger.plot()
        _atomic_json(output / "status.json", {**progress(plan, step, args.epochs),
            "status": "complete" if step == args.max_steps else "paused" if step == end else "running"})
    print(f"[semantic] {'complete' if end == args.max_steps else 'paused (not full training)'}: "
          f"{output / f'checkpoint-{end:06d}'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
