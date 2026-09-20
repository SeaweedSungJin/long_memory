#!/usr/bin/env python3
"""Two transient visual-only updates at fixed TRAIN episode 1303/query 88.

GPU0 worst-cached-prefix wiring/resource diagnostic ONLY. Reuses the frozen V11
trainer's actual head, visual config, objective, optimizer groups and scope
guards. Saves JSON measurements, never model/optimizer tensors. No simulator,
validation, retrieval-quality, convergence, accuracy or GPU exact-gradient claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes, validate_decision
from gr00t.long_memory.checkpoint_v7 import load_checkpoint_v7, v7_checkpoint_info
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import LoRAConfig, install_expert_lora, set_expert_trainable
from gr00t.long_memory.hamlet import checkpoint_identity, isolated_seed, validate_cache_checkpoint
from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme import train_visual_patch_v11 as trainer
from run_scripts.robomme.checkpoint_visual_patch_v11 import parent_reference
from run_scripts.robomme.verify_projector_v10 import sha
from run_scripts.robomme.verify_visual_patch_v11 import module_digest, tensor_record
from run_scripts.robomme.visual_patch_memory_v11 import CAMERA_ORDER, VisualPatchMemoryV11

EPISODE, QUERY, UPDATES = 1303, 88, 2
LIMITATIONS = [
    "One predeclared TRAIN query and fixed flow seed; two transient updates, not a training run.",
    "No saved model/optimizer, simulator, validation, task accuracy, convergence or retrieval-quality measurement.",
    "Frozen real Action Expert on original cached VLM features; not a full live policy test.",
    "No GPU bitwise gradient or exact-resume claim; finite/nonzero gradients establish wiring only.",
    "GT enters only the unchanged flow objective AFTER observation-only visual conditioning.",
    "Fixed diagnostic LR 1e-4, not a claim to reproduce a scheduled training update.",
]


def source_hashes():
    values = trainer.source_identity()
    for name in ("verify_visual_prefix_v11.py", "verify_visual_patch_v11.py"):
        path = ROOT / "run_scripts/robomme" / name
        values[str(path.relative_to(ROOT))] = sha(path)
    return dict(sorted(values.items()))


class Measurement:
    """CUDA synchronized resource measurements; CPU mode is for tiny unit tests."""
    def __init__(self, device):
        self.device = torch.device(device)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset(self):
        self.synchronize()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self):
        if self.device.type != "cuda":
            return {name: 0 for name in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes")}
        return {"allocated_bytes": torch.cuda.memory_allocated(self.device),
                "reserved_bytes": torch.cuda.memory_reserved(self.device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)}


def run_steps(args, visual, parent, head, cvom, episode, report, persist):
    """Exactly two updates; use the existing trainer objective without wrappers."""
    guard = trainer.frozen_guard(parent, head, cvom)
    trainer.assert_scope(visual, parent, head, cvom, guard)
    modules = {"head": head, "archive": parent, "cvom": cvom}
    before = {name: module_digest(module) for name, module in modules.items()}
    report["frozen_modules_before_sha256"] = before
    optimizer = torch.optim.AdamW(trainer.optimizer_groups(args, visual))
    report["optimizer_parameter_names"] = [[*group["param_names"]] for group in optimizer.param_groups]
    # Trace the oldest ORIGINAL observation without modifying the immutable cache.
    view = {**episode, "features": list(episode["features"])}
    earliest = view["features"][0].detach().clone().requires_grad_(True)
    view["features"][0] = earliest
    old_image = episode["image_masks"][0].clone() & episode["attention_masks"][0]
    old_image[-visual.config.num_short_tokens:] = False
    meter = Measurement(next(visual.parameters()).device)
    meter.reset()  # Allocation/hash/setup cost is excluded from update peaks.
    report["post_model_allocation_memory"] = meter.snapshot()
    checks = report.setdefault("checks", {})
    try:
        for index in range(UPDATES):
            optimizer.zero_grad(set_to_none=True)
            earliest.grad = None
            meter.reset()
            stage = {"after_optimizer_updates": index, "memory_before": meter.snapshot()}
            report.setdefault("steps", []).append(stage)
            persist()
            started = time.monotonic()
            loss, metrics = trainer.query_objective(args, visual, parent, head, view,
                {"episode_id": EPISODE, "decision": QUERY, "flow_seed": args.seed})
            meter.synchronize()
            forward_done = time.monotonic()
            stage["metrics"] = metrics
            loss.backward()
            meter.synchronize()
            backward_done = time.monotonic()
            gradients = {name: tensor_record(parameter.grad) for name, parameter in visual.named_parameters()}
            stage["parameter_gradients"] = gradients
            stage["earliest_image_gradient"] = tensor_record(None if earliest.grad is None else earliest.grad[old_image])
            finite = all(value["present"] and value["finite"] for value in gradients.values())
            finite &= stage["earliest_image_gradient"]["present"] and stage["earliest_image_gradient"]["finite"]
            checks[f"step{index}_all_gradients_present_finite"] = bool(finite)
            checks[f"step{index}_full_prefix"] = metrics["visual_bank_observations"] == QUERY and metrics["visual_bank_tokens"] == QUERY * 162
            checks[f"step{index}_output_gradient_nonzero"] = gradients["output_projection.weight"]["nonzero"] > 0
            if index == 0:
                checks["zero_image_residual"] = metrics["image_residual_norm"] == metrics["image_changed_fraction"] == 0.
                checks["zero_upstream_gradients"] = all(value["nonzero"] == 0 for name, value in gradients.items() if name != "output_projection.weight")
            else:
                for name in ("image_projection.weight", "query_projection.weight", "key_projection.weight", "value_projection.weight"):
                    checks[f"awake_{name}_gradient_nonzero"] = gradients[name]["nonzero"] > 0
                checks["awake_earliest_image_gradient_nonzero"] = stage["earliest_image_gradient"]["nonzero"] > 0
                checks["awake_bf16_images_changed"] = metrics["image_changed_fraction"] > 0
            if not finite:
                raise FloatingPointError("Nonfinite/disconnected gradient; refusing diagnostic update")
            trainer.assert_scope(visual, parent, head, cvom, guard)
            update_started = time.monotonic()
            norm = torch.nn.utils.clip_grad_norm_(visual.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            stage["preclip_visual_grad_norm"] = float(norm)
            optimizer.step()
            report["optimizer_updates"] += 1
            trainer.assert_finite_optimizer(optimizer)
            checks[f"step{index}_visual_parameters_finite"] = all(bool(torch.isfinite(p).all()) for p in visual.parameters())
            if not checks[f"step{index}_visual_parameters_finite"]:
                raise FloatingPointError("Nonfinite visual parameters after transient update")
            trainer.assert_scope(visual, parent, head, cvom, guard)
            meter.synchronize()
            finished = time.monotonic()
            stage["timing_seconds"] = {"forward": forward_done - started, "backward": backward_done - forward_done,
                "update_and_finite_checks": finished - update_started, "total_including_gradient_records": finished - started}
            stage["memory_after"] = meter.snapshot()
            persist()
            print(f"[worst-prefix-v11] update={index+1}/2 flow={metrics['loss']:.8f} "
                  f"changed={metrics['image_changed_fraction']:.6f} "
                  f"peak={stage['memory_after']['peak_allocated_bytes']/2**30:.3f}GiB "
                  f"total={finished-started:.3f}s", flush=True)
            del loss
    finally:
        report["frozen_modules_after_sha256"] = {name: module_digest(module) for name, module in modules.items()}
        checks["frozen_modules_unchanged"] = before == report["frozen_modules_after_sha256"]
        checks["frozen_modules_have_no_gradients"] = all(p.grad is None for module in modules.values() for p in module.parameters())
        persist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Exact original archive1250 parent")
    parser.add_argument("--output-dir", required=True, help="NEW directory, JSON diagnostics only")
    parser.add_argument("--seed", type=int, default=9111)
    args = parser.parse_args(argv)
    if args.seed < 0 or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Require nonnegative seed and explicit CUDA_VISIBLE_DEVICES=0")
    torch.set_num_threads(2)
    args.checkpoint_encoding = args.activation_checkpointing = True
    args.visual_learning_rate, args.weight_decay, args.max_grad_norm = 1e-4, .01, 1.
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    base, checkpoint = cache.manifest["model_path"], Path(args.checkpoint).resolve()
    parent_record = parent_reference(base, checkpoint)
    info = v7_checkpoint_info(base, checkpoint, expected_stage=1)
    if info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"] or EPISODE not in cache.manifest["splits"]["train"]:
        raise ValueError("Require same-cache original parent and predeclared TRAIN episode")
    episode = MappedEpisodes(cache).fetch(EPISODE)
    validate_decision(episode, QUERY)
    if not all(episode["features"][index].dtype == torch.bfloat16 for index in range(QUERY + 1)):
        raise ValueError("Worst-prefix diagnostic requires original BF16 cached observations")
    output = Path(args.output_dir).resolve()
    validate_output_scope(output, cache.path, cache.manifest.get("dataset_path"), base, checkpoint)
    if output.exists():
        raise FileExistsError("Use the authorized NEW diagnostic output directory")
    sources, base_before = source_hashes(), checkpoint_identity(base)
    config = trainer.visual_config(info)
    plan = {"args": vars(args), "episode_id": EPISODE, "query": QUERY, "optimizer_updates": UPDATES,
        "device": "cuda:0", "CUDA_VISIBLE_DEVICES": "0", "visual_config": asdict(config),
        "camera_order": list(CAMERA_ORDER), "prefix_frames": episode["frames"][:QUERY + 1].tolist(),
        "prefix_is_demo": episode["is_demo"][:QUERY + 1].tolist(), "parent_before": parent_record,
        "source_before_sha256": sources, "base_before": base_before, "limitations": LIMITATIONS}
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "plan.json", plan)
    report = {"passed": False, "episode_id": EPISODE, "query": QUERY, "optimizer_updates": 0,
        "checks": {}, "steps": [], "limitations": LIMITATIONS}
    def persist():
        _atomic_json(output / "result.json", report)
    persist()
    try:
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory, "device_count_visible": torch.cuda.device_count()}
        print(f"[worst-prefix-v11] loading frozen actual head on {props.name}; TRAIN1303/q88", flush=True)
        head = trainer.actual_head(base, "cuda:0")
        cfg = info["config"]
        with isolated_seed(args.seed + 100, "cuda:0"):
            parent = RecurrentMemoryV7(MemoryV7Config(**cfg["memory"])).to("cuda:0")
            cvom = CVOMV7(parent.config).to("cuda:0")
            install_expert_lora(head, LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
            visual = VisualPatchMemoryV11(config).to("cuda:0")
        load_checkpoint_v7(checkpoint, parent, head, cvom)
        parent.eval().requires_grad_(False)
        cvom.eval().requires_grad_(False)
        set_expert_trainable(head, False)
        visual.train()
        report["checks"]["actual_head_bf16"] = next(head.parameters()).dtype == torch.bfloat16
        report["checks"]["native_euler4_unchanged"] = head.num_inference_timesteps == 4
        report["visual_parameter_count"] = sum(p.numel() for p in visual.parameters())
        run_steps(args, visual, parent, head, cvom, episode, report, persist)
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(f"[worst-prefix-v11] diagnostic failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
        report["parent_after"] = parent_reference(base, checkpoint)
        report["source_after_sha256"] = source_hashes()
        report["base_after"] = checkpoint_identity(base)
        report["checks"]["parent_files_unchanged"] = report["parent_after"] == parent_record
        report["checks"]["source_files_unchanged"] = report["source_after_sha256"] == sources
        report["checks"]["base_identity_unchanged"] = report["base_after"] == base_before
        report["passed"] = "error" not in report and report["optimizer_updates"] == UPDATES and all(report["checks"].values())
        persist()
    print(json.dumps({key: report[key] for key in ("passed", "optimizer_updates", "checks")}, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
