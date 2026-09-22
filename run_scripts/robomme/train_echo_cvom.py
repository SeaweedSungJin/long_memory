#!/usr/bin/env python3
"""RoboMME ECHO: causal actor warm-up, additive CVoM labels, CPU slot learning.

Two learning stages, with an explicit frozen-teacher label preparation between
them. Re-running labels from a stage-2 snapshot collects learned-policy bank
states, not only FIFO states. No test scenarios, oracle labels or future action
targets enter the deployed encoder/query/utility inputs.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
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
from torch.nn import functional as F

from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes, validate_decision
from gr00t.long_memory.checkpoint_v4 import _state_sha256
from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora, set_expert_trainable, expert_parameters
from gr00t.long_memory.hamlet import load_frozen_hamlet, validate_cache_checkpoint, isolated_seed
from gr00t.long_memory.monitoring import RunLogger, _atomic_json
from gr00t.long_memory.safety_v5 import validate_output_scope
from gr00t.long_memory.train_v3 import _Tee, _seed, runtime_identity
from run_scripts.robomme.cvom_admission_checkpoint import parent_identity
from run_scripts.robomme.train_cvom_admission import digest, file_hash, gpu_guard, state_hash
from run_scripts.robomme.checkpoint_representation_v18 import checkpoint_info_v18, load_checkpoint_v18
from run_scripts.robomme.representation_core_v18 import RepresentationMemoryV18, RepresentationConfigV18
from run_scripts.robomme.training_plan_v19 import build_plan_v19, resolve_tasks_v19
from run_scripts.robomme.flow_objective_v19 import episode_flow_v19
from run_scripts.robomme.validation_v19 import validate_v19
from run_scripts.robomme.echo_cvom_checkpoint import (
    inspect_checkpoint, load_checkpoint, save_checkpoint, load_core, save_manager_checkpoint)

VARIANT = "echo_cvom_v1"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("warmup", "labels", "writer"))
    p.add_argument("--cache-dir", default="runs/long_memory/cache_full1600_v1")
    p.add_argument("--parent-checkpoint", default="runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072")
    p.add_argument("--checkpoint", help="Frozen ECHO snapshot for labels/writer")
    p.add_argument("--labels-dir")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=192201)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--query-batch-size", type=int, default=4)
    p.add_argument("--task-weighting", choices=("macro", "query"), default="macro")
    p.add_argument("--val-per-task", type=int, default=2)
    p.add_argument("--val-noise-samples", type=int, default=2)
    p.add_argument("--memory-learning-rate", type=float, default=3e-5)
    p.add_argument("--expert-learning-rate", type=float, default=3e-6)
    p.add_argument("--effect-weight", type=float, default=.01)
    p.add_argument("--min-fill", type=int, default=4, help="Warmup architecture: fill before learned admission")
    p.add_argument("--merge-threshold", type=float, default=None,
                   help="Optional conservative cosine merge; omitted = OFF (recommended first comparison)")
    p.add_argument("--recency-weight", type=float, default=.05)
    p.add_argument("--diversity-weight", type=float, default=.05)
    p.add_argument("--tail-weight", type=float, default=.25)
    p.add_argument("--eval-steps", type=int, default=1000)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--plot-steps", type=int, default=500)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--stop-after-steps", type=int, help="Smoke/pause, NOT complete-epoch training")
    p.add_argument("--resume", help="Warmup only, unchanged plan/source, NEW output directory")
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--contexts-per-episode", type=int, default=2)
    p.add_argument("--train-context-limit", type=int, default=0, help="0 includes every eligible TRAIN episode")
    p.add_argument("--val-context-limit", type=int, default=64)
    p.add_argument("--targets-per-context", type=int, default=4)
    p.add_argument("--coalitions", type=int, default=4)
    p.add_argument("--future-samples", type=int, default=2)
    p.add_argument("--noise-samples", type=int, default=2)
    p.add_argument("--stop-after-contexts", type=int, default=0, help="Resumable teacher preparation pause")
    p.add_argument("--writer-updates", type=int, default=1000)
    p.add_argument("--writer-batch-size", type=int, default=64)
    p.add_argument("--writer-learning-rate", type=float, default=3e-4)
    p.add_argument("--write-margin", type=float, default=1e-6)
    p.add_argument("--confidence-z", type=float, default=1.96)
    p.add_argument("--bce-weight", type=float, default=.25)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--preflight-only", action="store_true")
    return p


def validate_args(a):
    positive = ("epochs", "query_batch_size", "val_per_task", "val_noise_samples", "eval_steps", "save_steps",
                "plot_steps", "log_steps", "contexts_per_episode", "val_context_limit", "targets_per_context",
                "coalitions", "future_samples", "noise_samples", "writer_updates", "writer_batch_size", "cpu_threads")
    if any(getattr(a, n) <= 0 for n in positive) or a.seed < 0 or a.train_context_limit < 0 or a.stop_after_contexts < 0:
        raise ValueError("Invalid positive budget or nonnegative seed/limit")
    for name in ("memory_learning_rate", "expert_learning_rate", "writer_learning_rate", "tail_weight"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("effect_weight", "write_margin", "confidence_z", "bce_weight", "recency_weight", "diversity_weight"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    if a.tail_weight > 1 or a.noise_samples < 2 or a.future_samples < 1 or a.targets_per_context < 2:
        raise ValueError("Need valid flow weight, >=2 noise samples and >=2 all-slot targets")
    if a.stop_after_steps is not None and a.stop_after_steps <= 0:
        raise ValueError("stop-after-steps must be positive")
    if a.command != "warmup" and (not a.checkpoint or a.resume):
        raise ValueError("labels/writer require --checkpoint and cannot use warmup --resume")
    if a.command == "writer" and not a.labels_dir:
        raise ValueError("writer requires --labels-dir")
    if not 1 <= a.min_fill <= 32 or (a.merge_threshold is not None and
            (not math.isfinite(a.merge_threshold) or not 0 <= a.merge_threshold <= 1)):
        raise ValueError("Invalid min-fill or optional merge threshold")
    if a.command != "warmup" and (a.min_fill != 4 or a.merge_threshold is not None
                                 or a.recency_weight != .05 or a.diversity_weight != .05):
        raise ValueError("Storage architecture options belong to warmup; later stages inherit the checkpoint")


def sources():
    names = ("train_echo_cvom.py", "echo_cvom_core.py", "echo_cvom_teacher.py", "echo_cvom_checkpoint.py",
             "representation_core_v18.py", "checkpoint_representation_v18.py", "flow_objective_v19.py",
             "training_plan_v19.py", "validation_v19.py", "deployment_objective_v9.py",
             "cvom_admission_checkpoint.py", "train_cvom_admission.py")
    paths = list((ROOT / "gr00t").rglob("*.py")) + [ROOT / "run_scripts/robomme" / n for n in names]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}


def load_actor(parent_path, cache, device, *, snapshot=None, seed=0, echo_config=None):
    from run_scripts.robomme.echo_cvom_core import EchoMemoryV1, EchoConfig
    base_path = cache.manifest["model_path"]
    parent = checkpoint_info_v18(base_path, parent_path)
    info = inspect_checkpoint(base_path, snapshot) if snapshot else None
    config = info["config"] if info else parent["config"]
    base, processor = load_frozen_hamlet(base_path, device)
    head = base.action_head
    install_expert_lora(head, LoRAConfig(**config["expert"]), config["expert_targets"])
    rcfg = RepresentationConfigV18(**config["representation"])
    if rcfg.representation != "short":
        raise ValueError("Initial ECHO implementation requires the verified native short parent")
    with isolated_seed(seed, device):
        core = EchoMemoryV1(rcfg, EchoConfig(**config["echo"]) if info else (echo_config or EchoConfig())).to(device)
    if info:
        load_checkpoint(snapshot, core, head)
    else:
        parent_core = RepresentationMemoryV18(rcfg).to(device)
        load_checkpoint_v18(parent_path, parent_core, head)
        core.initialize_from_parent_delta(parent_core.delta_state_dict())
        del parent_core
    base.action_head = None
    del base, processor
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return core, head, parent


def preparation(a):
    from run_scripts.robomme.echo_cvom_core import EchoConfig
    cache = EpisodeCache(a.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    episodes = MappedEpisodes(cache, max_cached=2)
    parent = parent_identity(a.parent_checkpoint)
    info = checkpoint_info_v18(cache.manifest["model_path"], a.parent_checkpoint)
    if info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]:
        raise ValueError("Parent/cache mismatch")
    if cache.manifest["action_steps"] != 16 or info["config"]["representation"]["representation"] != "short":
        raise ValueError("Require verified RoboMME short/native parent and execution prefix 16")
    output = validate_output_scope(a.output_dir, cache.path, cache.manifest["model_path"],
        cache.manifest.get("dataset_path"), a.parent_checkpoint, a.checkpoint, a.resume, a.labels_dir)
    source = sources()
    snapshot = inspect_checkpoint(cache.manifest["model_path"], a.checkpoint) if a.checkpoint else None
    if snapshot and (snapshot["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]
                     or snapshot["metadata"]["parent_identity"] != parent):
        raise ValueError("ECHO snapshot/cache/parent mismatch")
    return cache, episodes, parent, info, snapshot, output, source


def scalar_values(values):
    return {k: float(v.detach()) if isinstance(v, torch.Tensor) else float(v) for k, v in values.items()
            if isinstance(v, (int, float)) or isinstance(v, torch.Tensor) and v.numel() == 1}


def warmup(a, context):
    from run_scripts.robomme.echo_cvom_core import EchoConfig
    cache, episodes, parent_id, parent_info, _, output, source = context
    plan, plan_sha = build_plan_v19(a, cache, episodes)
    end = a.stop_after_steps or plan["total_steps"]
    start_info = inspect_checkpoint(cache.manifest["model_path"], a.resume) if a.resume else None
    start = start_info["step"] if start_info else 0
    if not 0 <= start < end <= plan["total_steps"]:
        raise ValueError("Stop/resume must lie within the fixed full-epoch schedule")
    if start_info:
        if (start_info["stage"] != 1 or start_info["metadata"]["plan_sha256"] != plan_sha
                or start_info["metadata"]["source_sha256"] != source):
            raise ValueError("Exact warmup resume requires unchanged stage, plan and sources")
        ignored = {"resume", "output_dir", "stop_after_steps", "preflight_only"}
        if any(start_info["config"]["train"].get(k) != v for k, v in vars(a).items() if k not in ignored):
            raise ValueError("Exact warmup resume changed training settings")
    report = {"stage": 1, "planned_updates": plan["total_steps"], "query_count": plan["train_query_count"],
        "task_statistics": plan["task_statistics"], "source_files": len(source), "plan_sha256": plan_sha,
        "trainable": "event effect adapter + original external reader/fusion + AE LoRA, NOT HAMLET/VLM",
        "storage": "FIFO warm-up; no utility supervision yet", "end_step": end}
    if a.preflight_only:
        print(json.dumps(report, indent=2)); return
    if output.exists():
        raise FileExistsError("Warmup requires a NEW output; resume uses a NEW output too")
    gpu = gpu_guard(a.device)
    output.mkdir(parents=True)
    _atomic_json(output / "preflight.json", {**report, "gpu": gpu})
    _seed(a.seed)
    core, head, parent_info = load_actor(a.parent_checkpoint, cache, a.device, snapshot=a.resume, seed=a.seed,
        echo_config=EchoConfig(min_fill=a.min_fill, merge_threshold=a.merge_threshold,
                               recency_weight=a.recency_weight, diversity_weight=a.diversity_weight))
    core.eval().requires_grad_(True)
    core.manager.requires_grad_(False)
    head.eval(); set_expert_trainable(head, True)
    groups = [{"name": "memory", "params": [p for p in core.parameters() if p.requires_grad], "lr": a.memory_learning_rate},
              {"name": "expert", "params": list(expert_parameters(head)), "lr": a.expert_learning_rate}]
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    if start_info:
        load_checkpoint(a.resume, core, head, optimizer)
    config = {"representation": asdict(core.config), "echo": asdict(core.echo_config),
        "expert": parent_info["config"]["expert"], "expert_targets": parent_info["config"]["expert_targets"],
        "train": vars(a).copy(), "selection": "fixed_final_epoch", "feature_precision": "native"}
    metadata = {"base_model": parent_info["metadata"]["base_model"], "parent_identity": parent_id,
        "cache_fingerprint": cache.manifest["fingerprint"], "cache_dir": str(cache.path),
        "plan_sha256": plan_sha, "source_sha256": source, "runtime": runtime_identity(),
        "future_inputs_at_inference": False, "supervision": "action flow + optional detached transition reconstruction"}
    _atomic_json(output / "run_config.json", config)
    _atomic_json(output / "query_plan.json", {"sha256": plan_sha, **plan})
    logger, baseline_cache, saved = RunLogger(output), {}, set()
    initial_manager = state_hash(core.manager)

    def save(step):
        if step not in saved:
            save_checkpoint(output, step, core, head, optimizer, config,
                {**metadata, "training_complete": step == plan["total_steps"],
                 "smoke_only": step != plan["total_steps"]}, stage=1)
            saved.add(step)

    def evaluate(step):
        summary, records, tasks = validate_v19(a, core, head, episodes, plan, baseline_cache)
        for role, split in (("reader", "val"), ("memory-off", "comparison/memory-off"), ("baseline", "comparison/baseline")):
            logger.log(step, split, summary[role])
        _atomic_json(output / f"validation-{step:06d}.json", {"step": step, "summary": summary,
            "by_task": tasks, "records": records, "interpretation": "offline action errors, NOT rollout success"})
        print(f"[echo][val] step={step} action={summary['reader']['action_loss']:.6f} "
              f"generated_prefix_MSE={summary['reader']['generated_prefix_mse']:.6f}; NOT rollout success", flush=True)

    evaluate(start); save(start)
    started = time.monotonic()
    for step in range(start + 1, end + 1):
        optimizer.zero_grad(set_to_none=True)
        rows, metrics = plan["schedule"][step-1]["queries"], []
        for row in rows:
            ep, d = episodes.fetch(row["episode_id"]), row["decision"]
            validate_decision(ep, d)
            out = core.replay(ep, d, write_mode="fifo", activation_checkpointing=a.activation_checkpointing)
            flow = episode_flow_v19(head, ep, d, out["fused"], seed=row["flow_seed"],
                tail_weight=a.tail_weight, activation_checkpointing=a.activation_checkpointing)
            reconstruction = core.effect_loss(ep, d+1)
            loss = flow["loss"] + a.effect_weight*reconstruction
            if not bool(torch.isfinite(loss)) or not loss.requires_grad:
                raise FloatingPointError("Nonfinite/disconnected warmup loss")
            (loss*row["task_weight"]/len(rows)).backward()
            metrics.append({"loss": float(loss.detach()), "action_loss": float(flow["original_flow_loss"].detach()),
                "effect_reconstruction": float(reconstruction.detach()), **scalar_values(out["metrics"])})
        params = [p for g in optimizer.param_groups for p in g["params"]]
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in params):
            raise FloatingPointError("Nonfinite updated ECHO/AE parameter")
        if step % a.log_steps == 0 or step == end:
            values = {key: sum(m[key] for m in metrics)/len(metrics) for key in metrics[0]}
            values.update(grad_norm=float(norm), learning_rate=groups[0]["lr"], elapsed_seconds=time.monotonic()-started)
            logger.log(step, "train", values)
            print(f"[echo] warmup {step}/{plan['total_steps']} action={values['action_loss']:.6f} "
                  f"effect={values['effect_reconstruction']:.6f}", flush=True)
        if step % a.eval_steps == 0 or step == end:
            evaluate(step)
        if step % a.save_steps == 0 or step == end:
            save(step)
        if step % a.plot_steps == 0 or step == end:
            logger.plot()
        _atomic_json(output / "status.json", {"stage": 1, "step": step, "total_steps": plan["total_steps"],
            "status": "complete" if step == plan["total_steps"] else "paused" if step == end else "running"})
    if state_hash(core.manager) != initial_manager or parent_identity(a.parent_checkpoint) != parent_id:
        raise ValueError("Warmup unexpectedly modified utility manager or original parent")


def episode_signatures(cache, ids):
    """Detect ordinary payload edits on resume without hashing the entire large cache.

    These filesystem signatures are NOT cryptographic content hashes. The
    separately hashed cache manifest/checkpoint/source still identify the run.
    """
    root = Path(cache.path).resolve()
    records = {int(r["episode_id"]): r for r in cache.manifest["episodes"]}
    result = {}
    for eid in sorted(set(ids)):
        path = (root/records[eid]["path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Episode escapes cache")
        stat = path.stat()
        result[str(eid)] = {"path": str(path), "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
    return result


def label_protocol(a, context):
    from run_scripts.robomme.echo_cvom_teacher import build_plan
    cache, episodes, parent, _, snapshot, output, source = context
    cfg = RepresentationConfigV18(**snapshot["config"]["representation"])
    tasks, resolution = resolve_tasks_v19(cache.manifest)
    mode = "fifo" if snapshot["stage"] == 1 else "learned"
    plan = {split: build_plan(cache, episodes, tasks, cfg, split=split, seed=a.seed,
        contexts_per_episode=a.contexts_per_episode, limit=limit, future_samples=a.future_samples,
        targets_per_context=a.targets_per_context, write_mode=mode)
        for split, limit in (("train", a.train_context_limit), ("val", a.val_context_limit))}
    if not plan["train"] or not plan["val"] or ({r["episode_id"] for r in plan["train"]} & {r["episode_id"] for r in plan["val"]}):
        raise ValueError("Require nonempty disjoint TRAIN/cache-VAL plans")
    result = {"variant": VARIANT, "definition": "additive_coalitional_future_flow_utility",
        "snapshot": {"path": str(Path(a.checkpoint).resolve()), "files_sha256": snapshot["files_sha256"]},
        "cache_fingerprint": cache.manifest["fingerprint"], "cache_manifest_sha256": file_hash(Path(cache.path)/"manifest.json"),
        "episode_file_stat_signatures": episode_signatures(cache, [r["episode_id"] for rows in plan.values() for r in rows]),
        "plan": plan, "task_resolution": resolution, "write_mode": mode, "source_sha256": source,
        "teacher": {"seed": a.seed, "coalitions": a.coalitions, "noise_samples": a.noise_samples,
            "future_samples": a.future_samples, "tail_weight": a.tail_weight, "targets_per_context": a.targets_per_context},
        "limitation": "Frozen conditional memory-content proxy, not rollout return; labels never online inputs"}
    result["fingerprint"] = digest(result)
    return result


def labels(a, context):
    from run_scripts.robomme.echo_cvom_teacher import label_contexts
    cache, episodes, _, _, snapshot, output, _ = context
    protocol = label_protocol(a, context)
    rows = [(split, i, row) for split in ("train", "val") for i, row in enumerate(protocol["plan"][split])]
    report = {"contexts": {s: len(protocol["plan"][s]) for s in ("train", "val")},
        "actor_call_upper_bound": len(rows)*a.targets_per_context*a.coalitions*a.future_samples*a.noise_samples*2,
        "teacher_write_mode": protocol["write_mode"], "fingerprint": protocol["fingerprint"],
        "note": "Includes pre-capacity observations and existing slots; no success-based selection"}
    if a.preflight_only:
        print(json.dumps(report, indent=2)); return
    if output.exists():
        if not (output/"protocol.json").is_file() or json.loads((output/"protocol.json").read_text()) != protocol:
            raise ValueError("Label resume requires identical immutable snapshot/source/plan")
    else:
        output.mkdir(parents=True); _atomic_json(output/"protocol.json", protocol)
    pending = []
    for split, i, row in rows:
        path = output/split/f"context-{i:06d}.json"
        if path.exists():
            verify_label(json.loads(path.read_text()), protocol, split, row)
        else:
            pending.append((split, row, path))
    if not pending and (output/"manifest.json").is_file():
        print("[echo] labels already complete and verified"); return
    gpu = gpu_guard(a.device)
    _atomic_json(output/"preflight.json", {**report, "gpu": gpu})
    core, head, _ = load_actor(a.parent_checkpoint, cache, a.device, snapshot=a.checkpoint)
    core.eval().requires_grad_(False); head.eval(); set_expert_trainable(head, False)
    before = {"core": state_hash(core), "head": state_hash(head)}
    count, started = 0, time.monotonic()
    for split, row, path in pending:
        result = label_contexts(core, head, episodes, [row], seed=a.seed, coalitions=a.coalitions,
            noise_samples=a.noise_samples, tail_weight=a.tail_weight, teacher_snapshot_version=protocol["fingerprint"])
        packet = {"protocol_fingerprint": protocol["fingerprint"], "split": split, "row": row,
                  "actor_state_hash": before, "result": result}
        packet["sha256"] = digest(packet)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, packet)
        count += 1
        _atomic_json(output/"status.json", {"status": "running", "new_contexts": count, "remaining": len(pending)-count})
        print(f"[echo-label] {split} new={count}/{len(pending)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        if a.stop_after_contexts and count >= a.stop_after_contexts:
            break
    after = {"core": state_hash(core), "head": state_hash(head)}
    current = inspect_checkpoint(cache.manifest["model_path"], a.checkpoint)
    if before != after or current["files_sha256"] != snapshot["files_sha256"]:
        raise ValueError("Frozen CVoM teacher changed")
    if episode_signatures(cache, [r["episode_id"] for _, _, r in rows]) != protocol["episode_file_stat_signatures"]:
        raise ValueError("Cache payload changed during label generation")
    if count == len(pending):
        all_paths = [output/s/f"context-{i:06d}.json" for s, i, _ in rows]
        hashes = {str(p.relative_to(output)): file_hash(p) for p in all_paths}
        _atomic_json(output/"manifest.json", {"protocol_fingerprint": protocol["fingerprint"], "files": hashes,
            "actor_before": before, "actor_after": after, "definition": protocol["definition"]})
    _atomic_json(output/"status.json", {"status": "complete" if count == len(pending) else "paused",
                                       "new_contexts": count, "remaining": len(pending)-count})


def verify_label(packet, protocol, split, row):
    if (packet.get("protocol_fingerprint") != protocol["fingerprint"] or packet.get("split") != split
            or packet.get("row") != row or packet.get("sha256") != digest({k:v for k,v in packet.items() if k != "sha256"})):
        raise ValueError("Label identity/hash differs")
    if len(packet["result"]["contexts"]) != 1:
        raise ValueError("Expected one context per immutable label file")
    context = packet["result"]["contexts"][0]
    if (any(context.get(k) != v for k, v in row.items())
            or context.get("teacher_snapshot_version") != protocol["fingerprint"]
            or context.get("candidate_event_id") != row["event"]
            or not context.get("targets")):
        raise ValueError("Inner label context differs or has no targets")
    pool = context["bank_event_ids"] + [[row["event"]]]
    indices = [t["index"] for t in context["targets"]]
    if len(indices) != len(set(indices)) or len(pool)-1 not in indices:
        raise ValueError("Require unique targets including the candidate")
    for target in context["targets"]:
        i = target["index"]
        if (type(i) is not int or not 0 <= i < len(pool) or target["event_ids"] != pool[i]
                or target["event_id"] != pool[i][-1] or target["is_new"] != (i == len(pool)-1)
                or not target["features"]):
            raise ValueError("Label slot identity/features differ")
        vals = [*target["features"], target["signed_mean"], target["noise_mean_std"]]
        if not all(math.isfinite(v) for v in vals) or target["noise_mean_std"] < 0:
            raise ValueError("Nonfinite label values")
    return context


def read_labels(path, snapshot, source):
    path = Path(path).resolve(strict=True)
    protocol, manifest = [json.loads((path/name).read_text()) for name in ("protocol.json", "manifest.json")]
    if (protocol["snapshot"]["files_sha256"] != snapshot["files_sha256"] or protocol["source_sha256"] != source
            or manifest["protocol_fingerprint"] != protocol["fingerprint"]
            or protocol["fingerprint"] != digest({k:v for k,v in protocol.items() if k != "fingerprint"})
            or manifest["actor_before"] != manifest["actor_after"]):
        raise ValueError("Labels require identical frozen teacher and source")
    result = {}
    for split in ("train", "val"):
        result[split] = []
        for i, row in enumerate(protocol["plan"][split]):
            relative = f"{split}/context-{i:06d}.json"
            if manifest["files"].get(relative) != file_hash(path/relative):
                raise ValueError("Label file changed after publication")
            packet = json.loads((path/relative).read_text())
            if packet["actor_state_hash"] != manifest["actor_before"]:
                raise ValueError("Labels mix frozen teachers")
            context = verify_label(packet, protocol, split, row)
            result[split] += [{**target, "episode_id": row["episode_id"], "event": row["event"], "context_index": i}
                              for target in context["targets"]]
    if {r["episode_id"] for r in result["train"]} & {r["episode_id"] for r in result["val"]}:
        raise ValueError("Writer TRAIN/cache-VAL overlap")
    return result, protocol


def tensors(rows, scale, margin, z):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Utility scale must be positive and finite")
    x = torch.tensor([r["features"] for r in rows], dtype=torch.float32)
    raw = torch.tensor([r["signed_mean"] for r in rows], dtype=torch.float32)
    se = torch.tensor([r["noise_mean_std"] for r in rows], dtype=torch.float32)
    if not rows or not all(bool(torch.isfinite(v).all()) for v in (x, raw, se)) or bool((se < 0).any()):
        raise ValueError("Invalid/empty utility observations")
    # Continuous regression retains EVERY label, including near-zero/negative
    # utility. Noise affects weight but cannot erase >80% of examples again.
    weight = (1/(1+(se.double()/scale).square())).clamp_min(.1).float()
    utility = torch.asinh(raw.double()/scale).float()
    if not bool(torch.isfinite(utility).all()):
        raise ValueError("Utility normalization overflow")
    return {"features": x, "raw": raw, "utility": utility, "weight": weight,
            "confident": raw.abs() > torch.maximum(se*z, se.new_full(se.shape, margin)),
            "write": (raw > margin).float()}


def utility_loss(prediction, data, indices, bce_weight):
    weight = data["weight"][indices]
    reg = (F.smooth_l1_loss(prediction["utility"], data["utility"][indices], reduction="none")*weight).sum()/weight.sum()
    mask = data["confident"][indices]
    bce = F.binary_cross_entropy_with_logits(prediction["logit"][mask], data["write"][indices][mask]) if bool(mask.any()) else prediction["logit"].sum()*0
    return reg+bce_weight*bce, reg, bce


@torch.no_grad()
def writer_metrics(manager, data):
    pred = manager.forward_features(data["features"])
    _, reg, bce = utility_loss(pred, data, torch.arange(len(data["raw"])), .25)
    mask = data["confident"]
    x, y = pred["utility"], data["utility"]
    denominator = ((x-x.mean()).square().sum()*(y-y.mean()).square().sum()).sqrt()
    corr = float(((x-x.mean())*(y-y.mean())).sum()/denominator) if denominator > 0 else 0.
    return {"utility_loss": float(reg), "write_loss": float(bce), "utility_corr": corr,
        "utility_corr_defined": float(denominator > 0), "regression_examples": len(y), "confident_examples": int(mask.sum()),
        "write_accuracy": float(((pred["write_probability"][mask] >= .5) == data["write"][mask].bool()).float().mean()) if bool(mask.any()) else 0.,
        "write_accuracy_defined": float(bool(mask.any())), "predicted_admission_rate": float((pred["write_probability"] >= .5).float().mean()),
        "positive_label_rate": float(data["write"].mean())}


def writer(a, context):
    cache, _, parent_id, _, snapshot, output, source = context
    rows, protocol = read_labels(a.labels_dir, snapshot, source)
    ids = [r["episode_id"] for rows_split in protocol["plan"].values() for r in rows_split]
    if episode_signatures(cache, ids) != protocol["episode_file_stat_signatures"]:
        raise ValueError("Cache payload changed after teacher preparation")
    scale = max(float(torch.tensor([abs(r["signed_mean"]) for r in rows["train"]]).median()), a.write_margin, 1e-9)
    data = {s: tensors(r, scale, a.write_margin, a.confidence_z) for s,r in rows.items()}
    audit = {s: {"regression_examples": len(v["raw"]), "confident_bce_examples": int(v["confident"].sum()),
        "positive_examples": int(v["write"].sum()), "distinct_episodes": len({r["episode_id"] for r in rows[s]})} for s,v in data.items()}
    if a.preflight_only:
        print(json.dumps({"stage": 2, "labels": audit, "scale_train_only": scale,
                          "write_mode_collection": protocol["write_mode"], "planned_updates": a.writer_updates}, indent=2)); return
    if output.exists():
        raise FileExistsError("Writer training requires a NEW output; interrupted runs are not silently restarted")
    output.mkdir(parents=True)
    core, _ = load_core(a.checkpoint)
    core.eval().requires_grad_(False); core.manager.requires_grad_(True)
    frozen_before = _state_sha256({k:v for k,v in core.delta_state_dict().items() if not k.startswith("manager.")})
    optimizer = torch.optim.AdamW([{"name": "utility_writer", "params": list(core.manager.parameters()),
                                   "lr": a.writer_learning_rate}], weight_decay=.01)
    config = {**snapshot["config"], "train": vars(a).copy(), "utility_units": "asinh(signed_future_flow_gain/TRAIN_median_abs_gain)",
              "selection": "fixed_final_update"}
    metadata = {**snapshot["metadata"], "source_sha256": source, "plan_sha256": protocol["fingerprint"],
        "teacher_snapshot": protocol["snapshot"], "teacher_collection_mode": protocol["write_mode"],
        "utility_scale": scale, "label_audit": audit, "training_complete": True,
        "stage2_actor_frozen": True, "inherited_actor_training_complete": snapshot["metadata"].get(
            "inherited_actor_training_complete", snapshot["metadata"].get("training_complete", False))}
    for key in ("payload_sha256", "payload_shapes", "training_state_sha256"):
        metadata.pop(key, None)
    _atomic_json(output/"run_config.json", config); _atomic_json(output/"label_audit.json", audit)
    logger, gen, started = RunLogger(output), torch.Generator().manual_seed(a.seed), time.monotonic()
    for step in range(1, a.writer_updates+1):
        ids = torch.randint(len(rows["train"]), (a.writer_batch_size,), generator=gen)
        prediction = core.manager.forward_features(data["train"]["features"][ids])
        loss, reg, bce = utility_loss(prediction, data["train"], ids, a.bce_weight)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(core.manager.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % a.log_steps == 0 or step == a.writer_updates:
            logger.log(step, "train", {"loss": float(loss.detach()), "utility_loss": float(reg.detach()),
                "write_loss": float(bce.detach()), "grad_norm": float(norm), "elapsed_seconds": time.monotonic()-started})
        if step % min(a.eval_steps, 100) == 0 or step == a.writer_updates:
            metrics = writer_metrics(core.manager, data["val"])
            logger.log(step, "val", metrics)
            print(f"[echo-writer] step={step}/{a.writer_updates} val={metrics}; NOT rollout success", flush=True)
        if step % a.save_steps == 0 or step == a.writer_updates:
            complete = step == a.writer_updates
            save_manager_checkpoint(output, step, core, optimizer, a.checkpoint, config,
                {**metadata, "training_complete": complete,
                 "smoke_only": not complete or not metadata["inherited_actor_training_complete"]})
        if step % a.plot_steps == 0 or step == a.writer_updates:
            logger.plot()
        _atomic_json(output/"status.json", {"stage": 2, "step": step, "total_steps": a.writer_updates,
            "status": "complete" if step == a.writer_updates else "running"})
    frozen_after = _state_sha256({k:v for k,v in core.delta_state_dict().items() if not k.startswith("manager.")})
    if frozen_before != frozen_after or parent_identity(a.parent_checkpoint) != parent_id:
        raise ValueError("Writer optimizer modified frozen actor or parent")
    _atomic_json(output/"training_summary.json", {"stage": 2, "status": "complete", "audit": audit,
        "metrics": writer_metrics(core.manager, data["val"]), "actor_unchanged": True,
        "source_checkpoint": str(Path(a.checkpoint).resolve()), "utility_scale": scale,
        "note": "Learned storage labels, NOT robot success. Online full-bank competition still requires rollout."})


def main(argv=None):
    a = parser().parse_args(argv); validate_args(a)
    torch.set_num_threads(a.cpu_threads)
    context = preparation(a)
    fn = {"warmup": warmup, "labels": labels, "writer": writer}[a.command]
    # Journal is outside the NEW output directory so its existence is never
    # confused with a valid run. Append preserves errors from interrupted work.
    if a.preflight_only:
        return fn(a, context)
    journal = Path(a.output_dir).resolve().with_name(Path(a.output_dir).name + ".launch.log")
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as stream, redirect_stdout(_Tee(sys.stdout, stream)), redirect_stderr(_Tee(sys.stderr, stream)):
        try:
            return fn(a, context)
        except BaseException:
            traceback.print_exc(); raise


if __name__ == "__main__":
    main()
