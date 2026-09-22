#!/usr/bin/env python3
"""Paired writer-only admission study; no actor optimization or robot rollout.

``preflight`` is read-only. ``prepare`` freezes the selected actor and journals
each paired counterfactual context, so interruption loses at most one context.
``train`` fits the two small controllers on CPU with identical initialization,
minibatch indices and update count. Future observations/actions are teacher-only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

ARMS = ("single", "coalitional")
FORMAT = "cvom_admission_labels_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def state_hash(module):
    """Include frozen parameters and buffers, one tensor at a time."""
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("preflight", "prepare", "train"))
    p.add_argument("--parent-checkpoint", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-contexts", type=int, default=0,
                   help="0: one context from every eligible TRAIN episode")
    p.add_argument("--val-contexts", type=int, default=64)
    p.add_argument("--seed", type=int, default=192109)
    p.add_argument("--coalitions", type=int, default=4)
    p.add_argument("--future-samples", type=int, default=2)
    p.add_argument("--noise-samples", type=int, default=2)
    p.add_argument("--tail-weight", type=float, default=.25)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--utility-margin", type=float, default=1e-6,
                   help="Raw teacher-gain ambiguity/scale floor; deployment normalized-utility threshold stays 0")
    p.add_argument("--confidence-z", type=float, default=1.96,
                   help="Heuristic conditional Monte Carlo noise multiplier, not an episode-level confidence interval")
    p.add_argument("--bce-weight", type=float, default=.25)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--device", default="cuda:0", help="Frozen teacher preparation only")
    p.add_argument("--stop-after-contexts", type=int, default=0,
                   help="Pause prepare after this many new contexts; 0 completes the plan")
    return p


def validate_options(args):
    if args.seed < 0 or args.train_contexts < 0 or args.stop_after_contexts < 0:
        raise ValueError("seed/train-contexts/stop-after-contexts must be nonnegative")
    for name in ("val_contexts", "coalitions", "future_samples", "noise_samples", "hidden_dim",
                 "updates", "batch_size", "eval_steps", "save_steps", "cpu_threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.coalitions != 4 or args.future_samples != 2 or args.noise_samples != 2:
        raise ValueError("This paired protocol requires 4 coalitions, 2 far-future queries and 2 noises")
    for name in ("learning_rate", "utility_margin", "confidence_z", "bce_weight", "tail_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.learning_rate == 0 or not 0 < args.tail_weight <= 1:
        raise ValueError("Positive learning rate and tail weight in (0,1] required")


def source_identity():
    names = ("train_cvom_admission.py", "cvom_admission_core.py", "cvom_admission_teacher.py",
             "cvom_admission_checkpoint.py", "representation_core_v18.py", "flow_objective_v19.py")
    paths = list((ROOT / "gr00t").rglob("*.py"))
    paths += [ROOT / "run_scripts/robomme" / name for name in names + (
        "checkpoint_representation_v18.py", "training_plan_v19.py")]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(set(paths))}


def gpu_guard(device):
    """Observe occupancy before loading; never stop another process."""
    if torch.device(device).type != "cuda":
        return {"device": device, "cuda_requested": False}
    active = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
    if active:
        raise RuntimeError("Existing GPU compute work detected; no actor loaded:\n" + active)
    devices = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
    return {"device": device, "compute_processes_before": active, "devices_before": devices,
            "limitation": "Occupancy check, not an exclusive device reservation"}


def preflight(args):
    from gr00t.long_memory.cache import EpisodeCache
    from gr00t.long_memory.cache_reader_v3 import MappedEpisodes
    from gr00t.long_memory.hamlet import validate_cache_checkpoint
    from gr00t.long_memory.safety_v5 import validate_output_scope
    from run_scripts.robomme.checkpoint_representation_v18 import checkpoint_info_v18
    from run_scripts.robomme.cvom_admission_checkpoint import parent_identity
    from run_scripts.robomme.cvom_admission_teacher import build_context_plan
    from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18
    from run_scripts.robomme.training_plan_v19 import resolve_tasks_v19
    parent = Path(args.parent_checkpoint).resolve(strict=True)
    info0 = json.loads((parent / "checkpoint.json").read_text())
    base = info0["metadata"]["base_model"]["path"]
    info = checkpoint_info_v18(base, parent)
    if info["step"] <= 0:
        raise ValueError("A trained frozen parent is required")
    cfg = RepresentationConfigV18(**info["config"]["representation"])
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    if cache.manifest["fingerprint"] != info["metadata"]["cache_fingerprint"]:
        raise ValueError("Parent and cache fingerprints differ")
    output = Path(args.output_dir).resolve()
    validate_output_scope(output, parent, cache.path, cache.manifest.get("dataset_path"), base)
    episodes = MappedEpisodes(cache, max_cached=2)
    tasks, task_provenance = resolve_tasks_v19(cache.manifest)
    plan = {split: build_context_plan(cache, episodes, tasks, cfg.capacity_events,
              count, args.seed, split=split, short_window=cfg.short_window,
              future_samples=args.future_samples)
            for split, count in (("train", args.train_contexts), ("val", args.val_contexts))}
    if not plan["train"] or not plan["val"]:
        raise ValueError("Both TRAIN and VAL need full-bank contexts with far-future decisions")
    split_ids = {s: {r["episode_id"] for r in plan[s]} for s in plan}
    if split_ids["train"] & split_ids["val"]:
        raise ValueError("Context episode splits overlap")
    protocol = {"format": FORMAT, "parent": parent_identity(parent),
        "cache_fingerprint": cache.manifest["fingerprint"],
        "cache_manifest_sha256": file_hash(Path(cache.path) / "manifest.json"),
        "splits": cache.manifest["splits"], "plan": plan, "plan_sha256": digest(plan),
        "task_resolution": task_provenance, "source_identity": source_identity(),
        "teacher": {"seed": args.seed, "coalitions": args.coalitions,
            "noise_samples": args.noise_samples, "future_samples": args.future_samples,
            "tail_weight": args.tail_weight, "future_bank": "fixed_conditional_no_continuation",
            "actor_frozen": True, "loss": "existing_action_flow_only"},
        "representation": asdict(cfg)}
    protocol["fingerprint"] = digest(protocol)
    print(f"[admission-preflight] TRAIN={len(plan['train'])} VAL={len(plan['val'])} "
          f"capacity={cfg.capacity_events}; actor frozen", flush=True)
    return protocol, info, cfg, cache, episodes, output


def verify_packet(packet, protocol, split, row):
    expected = {"protocol_fingerprint": protocol["fingerprint"], "split": split,
                "context_sha256": digest(row)}
    if any(packet.get(k) != v for k, v in expected.items()):
        raise ValueError("Prepared context belongs to a different protocol/context")
    if packet.get("sha256") != digest({k: v for k, v in packet.items() if k != "sha256"}):
        raise ValueError("Prepared context content hash differs")
    examples = packet["result"]["contexts"]
    if len(examples) != 1 or any(examples[0].get(k) != row[k] for k in ("episode_id", "event", "future")):
        raise ValueError("Teacher returned a different context identity")
    return examples[0]


def load_actor(parent, info, cfg, device):
    from gr00t.long_memory.hamlet import load_frozen_hamlet
    from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora, set_expert_trainable
    from run_scripts.robomme.checkpoint_representation_v18 import load_checkpoint_v18
    from run_scripts.robomme.representation_core_v18 import RepresentationMemoryV18
    model, _ = load_frozen_hamlet(info["metadata"]["base_model"]["path"], device)
    head = model.action_head
    install_expert_lora(head, LoRAConfig(**info["config"]["expert"]), info["config"]["expert_targets"])
    core = RepresentationMemoryV18(cfg, head.memory_transformer
        if cfg.representation == "adapted_short" else None).to(device)
    load_checkpoint_v18(parent, core, head)
    core.eval().requires_grad_(False)
    head.eval()
    set_expert_trainable(head, False)
    if any(p.requires_grad for m in (core, head) for p in m.parameters()):
        raise ValueError("All actor parameters must be frozen")
    # Cached inputs do not need the VLM. Releasing it saves teacher GPU memory.
    model.action_head = None
    del model
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return core, head


def prepare(args, context):
    from gr00t.long_memory.monitoring import _atomic_json
    from run_scripts.robomme.cvom_admission_checkpoint import parent_identity
    from run_scripts.robomme.cvom_admission_teacher import label_contexts
    protocol, info, cfg, cache, episodes, output = context
    if output.exists():
        path = output / "label_protocol.json"
        if not path.is_file() or json.loads(path.read_text()) != protocol:
            raise ValueError("Resume requires the identical saved label protocol")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _atomic_json(output / "label_protocol.json", protocol)
    rows = [(split, index, row) for split in ("train", "val")
            for index, row in enumerate(protocol["plan"][split])]
    pending = []
    for split, index, row in rows:
        path = output / "contexts" / split / f"context-{index:06d}.json"
        if path.exists():
            verify_packet(json.loads(path.read_text()), protocol, split, row)
        else:
            pending.append((split, index, row, path))
    if not pending and (output / "label_manifest.json").is_file():
        load_examples(output, protocol)
        print("[admission-prepare] All contexts already complete and verified", flush=True)
        return
    occupancy = gpu_guard(args.device)
    _atomic_json(output / "gpu_before.json", occupancy)
    core, head = load_actor(args.parent_checkpoint, info, cfg, args.device)
    before = {"core": state_hash(core), "head": state_hash(head)}
    for split, index, row in rows:
        path = output / "contexts" / split / f"context-{index:06d}.json"
        if path.exists() and json.loads(path.read_text()).get("actor_state_hash") != before:
            raise ValueError("Resumed contexts were generated by different in-memory actor weights")
    start, completed = time.monotonic(), 0
    _atomic_json(output / "prepare_status.json", {"status": "running", "pending": len(pending),
                 "actor_state_hash_before": before})
    try:
        for split, index, row, path in pending:
            result = label_contexts(core, head, episodes, [row], seed=args.seed,
                coalitions=args.coalitions, noise_samples=args.noise_samples,
                tail_weight=args.tail_weight)
            packet = {"protocol_fingerprint": protocol["fingerprint"], "split": split,
                      "context_sha256": digest(row), "actor_state_hash": before, "result": result}
            packet["sha256"] = digest(packet)
            verify_packet(packet, protocol, split, row)
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(path, packet)
            completed += 1
            _atomic_json(output / "prepare_status.json", {"status": "running", "new_contexts": completed,
                "pending": len(pending) - completed, "elapsed_seconds": time.monotonic() - start,
                "actor_state_hash_before": before})
            print(f"[admission-label] {split} {index + 1}/{len(protocol['plan'][split])}; "
                  f"new={completed} elapsed={time.monotonic()-start:.1f}s", flush=True)
            if args.stop_after_contexts and completed >= args.stop_after_contexts:
                break
    except BaseException as error:
        _atomic_json(output / "prepare_status.json", {"status": "failed", "new_contexts": completed,
            "pending": len(pending)-completed, "error_type": type(error).__name__, "error": str(error),
            "actor_state_hash_before": before})
        raise
    finally:
        after = {"core": state_hash(core), "head": state_hash(head)}
        if before != after or parent_identity(args.parent_checkpoint) != protocol["parent"]:
            _atomic_json(output / "prepare_status.json", {"status": "failed_actor_integrity",
                "actor_state_hash_before": before, "actor_state_hash_after": after})
            raise ValueError("Frozen actor changed during label preparation")
    all_complete = completed == len(pending)
    integrity = {"actor_state_hash_before": before, "actor_state_hash_after": after,
                 "actor_frozen": True, "future_inputs_at_inference": False}
    _atomic_json(output / "prepare_status.json", {"status": "complete" if all_complete else "paused",
                 "new_contexts": completed, "pending": len(pending)-completed, **integrity})
    if all_complete:
        manifest = {"format": FORMAT, "protocol_fingerprint": protocol["fingerprint"],
            "context_files": {str((Path("contexts") / split / f"context-{i:06d}.json")):
                file_hash(output / "contexts" / split / f"context-{i:06d}.json")
                for split, i, _ in rows}, **integrity}
        manifest["sha256"] = digest(manifest)
        _atomic_json(output / "label_manifest.json", manifest)


def load_examples(output, protocol):
    output = Path(output)
    manifest = json.loads((output / "label_manifest.json").read_text())
    if manifest.get("protocol_fingerprint") != protocol["fingerprint"] or manifest.get("sha256") != digest(
            {k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("Label manifest provenance/hash differs")
    if manifest["actor_state_hash_before"] != manifest["actor_state_hash_after"]:
        raise ValueError("Labels were not generated by an unchanged actor")
    examples = {}
    for split, rows in protocol["plan"].items():
        examples[split] = []
        for index, row in enumerate(rows):
            relative = f"contexts/{split}/context-{index:06d}.json"
            path = output / relative
            if manifest["context_files"].get(relative) != file_hash(path):
                raise ValueError("Prepared context differs from its published manifest")
            packet = json.loads(path.read_text())
            if packet.get("actor_state_hash") != manifest["actor_state_hash_before"]:
                raise ValueError("Context was generated by a different frozen actor")
            examples[split].append(verify_packet(packet, protocol, split, row))
    return examples, manifest


def tensor_examples(examples, arm, *, margin, confidence_z):
    x = torch.tensor([row["features"] for row in examples], dtype=torch.float32)
    labels = [row["labels"][arm] for row in examples]
    y = torch.tensor([row["signed_mean"] for row in labels], dtype=torch.float32)
    full = torch.tensor([row["labels"]["single"]["signed_mean"] for row in examples], dtype=torch.float32)
    noise = torch.tensor([row["noise_mean_std"] for row in labels], dtype=torch.float32)
    if x.ndim != 2 or y.ndim != 1 or not all(bool(torch.isfinite(t).all()) for t in (x, y, noise, full)):
        raise ValueError("Nonfinite or malformed controller examples")
    if bool((noise < 0).any()):
        raise ValueError("Teacher uncertainty must be nonnegative")
    threshold = torch.maximum(noise * confidence_z, noise.new_full(noise.shape, margin))
    return {"features": x, "utility": y, "confident": y.abs() > threshold,
            "noise": noise, "threshold": threshold, "full_bank_utility": full}


def training_scale(tensors, margin):
    """One robust common scale fitted exclusively on both arms' TRAIN labels."""
    values = torch.cat([tensors[arm]["utility"].abs() for arm in ARMS])
    return max(float(values.median()), margin, 1e-12)


def label_statistics(data):
    y, mask = data["utility"], data["confident"]
    return {"count": len(y), "confident_count": int(mask.sum()),
        "positive_count": int((mask & (y > 0)).sum()),
        "negative_count": int((mask & (y < 0)).sum()),
        "ambiguous_count": int((~mask).sum()), "mean_signed_gain": float(y.mean()),
        "mean_abs_gain": float(y.abs().mean())}


def writer_loss(prediction, data, indices, scale, bce_weight):
    mask = data["confident"][indices]
    if not bool(mask.any()):
        zero = (prediction["utility"].sum() + prediction["logit"].sum()) * 0
        return zero, zero, zero
    target = data["utility"][indices][mask]
    utility = F.smooth_l1_loss(prediction["utility"][mask], (target / scale).clamp(-10, 10))
    write = F.binary_cross_entropy_with_logits(prediction["logit"][mask], (target > 0).float())
    return utility + bce_weight * write, utility, write


@torch.no_grad()
def validation_metrics(controller, data, scale):
    output = controller.forward_features(data["features"])
    y, mask = data["utility"], data["confident"]
    prob = output["write_probability"]
    # Deployment requires both heads to agree; use exactly the same gate here.
    insert = (prob >= controller.config.write_threshold) & (output["utility"] >= controller.config.utility_margin)
    gain = torch.where(insert, y, torch.zeros_like(y))
    result = {"write_rate": float(insert.float().mean()), "confident_count": int(mask.sum()),
        "mean_signed_gain": float(y.mean()), "conditional_gain_vs_keep": float(gain.mean()),
        "conditional_gain_vs_fifo": float((gain-y).mean()),
        "conditional_oracle_regret": float((y.clamp_min(0)-gain).mean()),
        "constant_keep_regret": float(y.clamp_min(0).mean()),
        "constant_fifo_regret": float((-y).clamp_min(0).mean())}
    full = data["full_bank_utility"]
    full_gain = torch.where(insert, full, torch.zeros_like(full))
    result.update(full_bank_conditional_gain_vs_keep=float(full_gain.mean()),
        full_bank_conditional_gain_vs_fifo=float((full_gain-full).mean()),
        full_bank_conditional_oracle_regret=float((full.clamp_min(0)-full_gain).mean()),
        full_bank_constant_keep_regret=float(full.clamp_min(0).mean()),
        full_bank_constant_fifo_regret=float((-full).clamp_min(0).mean()))
    if bool(mask.any()):
        result.update(utility_loss=float(F.smooth_l1_loss(output["utility"][mask], (y[mask]/scale).clamp(-10, 10))),
            write_accuracy=float((insert[mask] == (y[mask] > 0)).float().mean()),
            write_brier=float((prob[mask]-(y[mask] > 0).float()).square().mean()))
    return result


def train(args, context):
    from gr00t.long_memory.monitoring import RunLogger, _atomic_json
    from run_scripts.robomme.cvom_admission_core import AdmissionConfig, CVoMAdmission
    from run_scripts.robomme.cvom_admission_checkpoint import save_checkpoint, parent_identity
    protocol, info, cfg, cache, episodes, output = context
    if json.loads((output / "label_protocol.json").read_text()) != protocol:
        raise ValueError("Training protocol differs from prepared labels")
    if any((output / arm).exists() for arm in ARMS):
        raise FileExistsError("Writer training never overwrites or partially resumes arm directories")
    examples, manifest = load_examples(output, protocol)
    torch.set_num_threads(args.cpu_threads)
    tensors = {split: {arm: tensor_examples(rows, arm, margin=args.utility_margin,
                confidence_z=args.confidence_z) for arm in ARMS} for split, rows in examples.items()}
    audit = {split: {arm: label_statistics(data) for arm, data in group.items()}
             for split, group in tensors.items()}
    _atomic_json(output / "label_audit.json", audit)
    for arm in ARMS:
        if audit["train"][arm]["confident_count"] == 0:
            _atomic_json(output / "train_status.json", {"status": "stopped_uninformative", "arm": arm})
            raise ValueError(f"No confident TRAIN labels for {arm}; do not train an arbitrary writer")
    scale = training_scale(tensors["train"], args.utility_margin)
    config = AdmissionConfig(dim=cfg.hidden_dim, num_tokens=cfg.num_short_tokens,
        capacity_events=cfg.capacity_events, hidden_dim=args.hidden_dim)
    torch.manual_seed(args.seed)
    initial = CVoMAdmission(config).cpu()
    initial_hash = state_hash(initial)
    controllers, optimizers, loggers = {}, {}, {}
    for arm in ARMS:
        controllers[arm] = CVoMAdmission(config).cpu()
        controllers[arm].load_state_dict(initial.state_dict())
        optimizers[arm] = torch.optim.AdamW(controllers[arm].parameters(), lr=args.learning_rate,
                                           weight_decay=.01)
        loggers[arm] = RunLogger(output / arm)
    metadata = {"cache_fingerprint": cache.manifest["fingerprint"],
        "label_manifest_sha256": file_hash(output / "label_manifest.json"),
        "source_identity": protocol["source_identity"], "seed": args.seed,
        "training_plan": {"context_plan_sha256": protocol["plan_sha256"],
            "updates": args.updates, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "selection": "fixed_final_step", "shared_initial_state_sha256": initial_hash,
            "shared_minibatch_seed": args.seed + 1, "scale": scale,
            "confidence_z": args.confidence_z, "utility_margin": args.utility_margin,
            "confidence_interpretation": "heuristic conditional noise guard; not an episode confidence interval",
            "bce_weight": args.bce_weight, "ambiguous_labels": "exclude_both_losses"},
        "actor_frozen": True, "future_inputs_at_inference": False,
        "actor_state_hash_before": manifest["actor_state_hash_before"],
        "actor_state_hash_after": manifest["actor_state_hash_after"],
        "metric_limitation": "conditional fixed-bank teacher flow utility, not rollout return"}
    _atomic_json(output / "training_config.json", metadata)
    _atomic_json(output / "train_status.json", {"status": "running", "step": 0})
    generator = torch.Generator().manual_seed(args.seed + 1)
    start = time.monotonic()
    for step in range(1, args.updates + 1):
        indices = torch.randint(len(examples["train"]), (args.batch_size,), generator=generator)
        for arm in ARMS:
            controller, optimizer = controllers[arm], optimizers[arm]
            controller.train()
            prediction = controller.forward_features(tensors["train"][arm]["features"][indices])
            loss, utility, write = writer_loss(prediction, tensors["train"][arm], indices, scale, args.bce_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1., error_if_nonfinite=True)
            # No weight-decay updates on wholly ambiguous batches.
            if bool(tensors["train"][arm]["confident"][indices].any()):
                optimizer.step()
            if step == 1 or step % 10 == 0:
                loggers[arm].log(step, "train", {"loss": float(loss.detach()),
                    "utility_loss": float(utility.detach()), "write_loss": float(write.detach()),
                    "grad_norm": float(norm), "elapsed_seconds": time.monotonic()-start})
            if step % args.eval_steps == 0 or step == args.updates:
                controller.eval()
                metrics = validation_metrics(controller, tensors["val"][arm], scale)
                loggers[arm].log(step, "val", metrics)
                print(f"[admission-train] {arm} step={step} val={metrics}", flush=True)
            if step % args.save_steps == 0 or step == args.updates:
                save_checkpoint(output / arm / f"checkpoint-{step:06d}", controller,
                    args.parent_checkpoint, arm=arm, step=step, metadata=metadata, optimizer=optimizer)
                loggers[arm].plot()
        if step % 10 == 0 or step == args.updates:
            _atomic_json(output / "train_status.json", {"status": "running", "step": step})
    if parent_identity(args.parent_checkpoint) != protocol["parent"]:
        raise ValueError("Parent files changed during CPU writer training")
    final = {arm: {"checkpoint": str(output / arm / f"checkpoint-{args.updates:06d}"),
        "metrics": validation_metrics(controllers[arm], tensors["val"][arm], scale),
        "final_state_sha256": state_hash(controllers[arm])} for arm in ARMS}
    _atomic_json(output / "training_summary.json", {"arms": final, "label_audit": audit,
        "scale": scale, "initial_state_sha256": initial_hash,
        "parent_unchanged": True, "selection": "fixed_final_step"})
    _atomic_json(output / "train_status.json", {"status": "complete", "step": args.updates})


def main(argv=None):
    args = parser().parse_args(argv)
    validate_options(args)
    torch.set_num_threads(args.cpu_threads)
    context = preflight(args)
    if args.stage == "prepare":
        prepare(args, context)
    elif args.stage == "train":
        train(args, context)


if __name__ == "__main__":
    main()
