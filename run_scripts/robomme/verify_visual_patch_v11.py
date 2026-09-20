#!/usr/bin/env python3
"""One TRAIN-query, two-update CPU visual-path diagnostic, NOT model training.

Uses the actual frozen archive1250 reader and adapted Action Expert. Only a new
transient visual module receives two AdamW updates; no model is saved. These
fixed-query/fixed-noise measurements establish wiring, not convergence,
retrieval selectivity, held-out quality, or RoboMME success.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from gr00t.long_memory.action_audit_v8 import cached_inputs, generated_action
from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes, validate_decision
from gr00t.long_memory.checkpoint_v7 import load_checkpoint_v7, v7_checkpoint_info
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import (
    LoRAConfig, expert_episode_flow_loss, expert_flow_loss, install_expert_lora,
    set_expert_trainable,
)
from gr00t.long_memory.hamlet import isolated_seed, sample_noise_time, validate_cache_checkpoint
from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from gr00t.long_memory.replay_v7 import replay_queries
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.verify_projector_v10 import actual_head, sha
from run_scripts.robomme.visual_patch_memory_v11 import CAMERA_ORDER, VisualPatchConfig, VisualPatchMemoryV11


OBSERVATION_KEYS = ("features", "image_masks", "attention_masks", "frames", "is_demo")
LIMITATIONS = [
    "One predeclared TRAIN query and fixed noise; two transient diagnostic updates, not a training run.",
    "No saved model, validation, simulator, robot success, accuracy, or convergence measurement.",
    "Full prior CACHED observations only; absent video frames cannot be recovered.",
    "Frozen real Action Expert on cached VLM features, not a full live VLM/online policy test.",
    "Nonzero gradients and BF16 changes do not establish useful or selective retrieval.",
    "Original flow supervision uses GT in the noisy action/loss AFTER observation-only feature construction.",
]


def tensor_record(tensor):
    """Preserve disconnected, exact-zero, and near-zero gradients distinctly."""
    if tensor is None:
        return {"present": False, "finite": True, "norm": None, "max_abs": None, "nonzero": 0}
    value = tensor.detach().float()
    finite = bool(torch.isfinite(value).all())
    return {"present": True, "finite": finite, "dtype": str(tensor.dtype),
            "norm": float(value.double().norm()) if finite else None,
            "max_abs": float(value.abs().max()) if finite and value.numel() else None,
            "nonzero": int(value.count_nonzero()), "numel": value.numel()}


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)]).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def module_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_digest(value).encode())
    return digest.hexdigest()


def visual_features(memory, observations, query, *, track_earliest=False):
    """Rebuild [0,q) with current visual weights, then READ q before any WRITE.

    This helper has no head, state, action, target or loss input. Its caller
    passes an observation-only mapping. The bank never receives visual READ
    output or parent-archive-fused short tokens.
    """
    if type(query) is not int or query < 1:
        raise ValueError("A query with strictly past observations is required")
    bank, earliest, current = None, None, None
    for index in range(query + 1):
        features = observations["features"][index].detach()[None]
        if index == 0 and track_earliest:
            features = features.clone().requires_grad_(True)
            earliest = features
        current = memory.encode_observation(
            features, observations["image_masks"][index].bool()[None],
            observations["attention_masks"][index].bool()[None],
            torch.as_tensor(observations["frames"][index]).reshape(1),
            torch.as_tensor(observations["is_demo"][index]).reshape(1), camera_order=CAMERA_ORDER)
        if index < query:
            bank = memory.append(bank, current)
    return memory.read(current, bank), bank, earliest


def replace_short(features, fused_short):
    count = fused_short.shape[-2]
    return torch.cat((features[:, :-count], fused_short.to(features)), dim=1)


def flow_at_features(head, episode, query, features, noise, time):
    # Only now, AFTER both memory paths have constructed their conditioning,
    # introduce the original flow objective's supervised action/noise inputs.
    _, state, embodiment, masks = cached_inputs(head, episode, query)
    target = episode["targets"][query].to(features)[None]
    target_mask = episode["target_mask"][query].to(device=features.device)[None]
    return expert_flow_loss(head, features, state, target, target_mask,
        masks.backbone_attention_mask, masks.image_mask, embodiment,
        noise=noise, time=time, activation_checkpointing=True)


def generated_at_features(head, episode, query, features, seed):
    view = dict(episode)
    view["features"] = {query: features[0]}
    return generated_action(head, view, query, seed=seed)


def select_query(cache, episodes):
    for episode_id in sorted(cache.manifest["splits"]["train"]):
        episode = episodes.fetch(episode_id)
        for query in torch.where(episode["decision_mask"])[0].tolist():
            if query >= 3 and bool(episode["action_mask"][query].any()):
                validate_decision(episode, query)
                return episode_id, query, episode
    raise ValueError("No first eligible TRAIN query >=3 with observed actions")


def source_hashes():
    paths = list((ROOT / "gr00t").rglob("*.py")) + [ROOT / "run_scripts/robomme" / name for name in (
        "verify_visual_patch_v11.py", "visual_patch_memory_v11.py", "verify_projector_v10.py",
        "projector_adapter_v10.py", "deployment_objective_v9.py", "audit_archive_generation_v7.py")]
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(paths))}


def run_diagnostic(head, parent, episode, query, visual, seed, report, persist):
    """In-memory only; persist reports, never parameter/optimizer payloads."""
    if head.training or parent.training or any(p.requires_grad for m in (head, parent) for p in m.parameters()):
        raise ValueError("Parent archive and complete adapted Action Expert must be frozen/eval")
    if head.num_inference_timesteps != 4:
        raise ValueError("Diagnostic requires the original Euler4 sampler")
    before = {"head": module_digest(head), "archive": module_digest(parent)}
    observations = {key: episode[key] for key in OBSERVATION_KEYS}
    report["frozen_before_sha256"] = before
    with torch.no_grad():
        fused, metrics = replay_queries(parent, episode, [query], mode="archive")[query]
        original, _, _, _ = cached_inputs(head, episode, query, fused)
        reference = expert_episode_flow_loss(head, episode, query, fused, seed=seed)
        reference_prediction = reference["prediction"].detach()
        reference_loss = reference["loss"].detach()
        original_generated = generated_action(head, episode, query, fused, seed=seed + 1)
    report["parent_read_metrics"] = {key: float(value) for key, value in metrics.items()}
    report["original_flow_loss"] = float(reference_loss)
    report["original_generated_sha256"] = tensor_digest(original_generated)
    target = episode["targets"][query].to(original)[None]
    noise, time = sample_noise_time(head, target, seed)
    report["fixed_flow"] = {"seed": seed, "time": float(time.flatten()[0]),
                            "noise_sha256": tensor_digest(noise), "generation_seed": seed + 1}
    optimizer = torch.optim.AdamW(visual.parameters(), lr=1e-4, weight_decay=0.)
    report["optimizer_parameter_names"] = list(dict(visual.named_parameters()))
    checks = report.setdefault("checks", {})
    records = report.setdefault("gradient_passes", [])
    try:
        for updates in range(3):
            optimizer.zero_grad(set_to_none=True)
            features, bank, earliest = visual_features(visual, observations, query, track_earliest=True)
            features = replace_short(features, fused)
            image_mask = episode["image_masks"][query].bool()[None].clone()
            image_mask[:, -visual.config.num_short_tokens:] = False
            image_mask &= episode["attention_masks"][query].bool()[None]
            difference = features.detach().float() - original.float()
            stage = {"after_optimizer_updates": updates, "bank_observations": bank.tokens.shape[1],
                     "bank_patch_tokens": bank.tokens.shape[1] * bank.tokens.shape[2],
                     "image_delta": tensor_record(difference[image_mask]),
                     "bf16_image_changed_fraction": float((features.detach()[image_mask] != original[image_mask]).float().mean()),
                     "non_image_exact": torch.equal(features.detach()[~image_mask], original[~image_mask]),
                     "short_exact": torch.equal(features.detach()[:, -4:], original[:, -4:])}
            records.append(stage)
            result = flow_at_features(head, episode, query, features, noise, time)
            stage["flow_loss"] = float(result["loss"].detach())
            if updates == 0:
                checks["zero_visual_features_exact"] = torch.equal(features.detach(), original)
                checks["zero_flow_prediction_exact"] = torch.equal(result["prediction"].detach(), reference_prediction)
                checks["zero_flow_loss_exact"] = torch.equal(result["loss"].detach(), reference_loss)
                generated = generated_at_features(head, episode, query, features.detach(), seed + 1)
                checks["zero_generated_euler4_exact"] = torch.equal(generated, original_generated)
            result["loss"].backward()
            stage["parameter_gradients"] = {name: tensor_record(p.grad) for name, p in visual.named_parameters()}
            old_mask = observations["image_masks"][0].bool()[None].clone()
            old_mask[:, -visual.config.num_short_tokens:] = False
            old_mask &= observations["attention_masks"][0].bool()[None]
            stage["earliest_past_image_gradient"] = tensor_record(None if earliest.grad is None else earliest.grad[old_mask])
            grads = stage["parameter_gradients"]
            finite = all(g["finite"] for g in grads.values()) and stage["earliest_past_image_gradient"]["finite"]
            checks[f"pass{updates}_all_gradients_finite"] = finite
            checks[f"pass{updates}_full_causal_bank"] = bank.tokens.shape[1] == query
            checks[f"pass{updates}_only_images_changed"] = stage["non_image_exact"] and stage["short_exact"]
            checks[f"pass{updates}_output_gradient_nonzero"] = grads["output_projection.weight"]["nonzero"] > 0
            if updates == 0:
                checks["zero_upstream_gradients_expected_zero"] = all(
                    g["nonzero"] == 0 for name, g in grads.items() if name != "output_projection.weight")
                checks["zero_earliest_image_gradient_expected_zero"] = stage["earliest_past_image_gradient"]["nonzero"] == 0
            else:
                for name in ("image_projection.weight", "query_projection.weight", "key_projection.weight", "value_projection.weight"):
                    checks[f"pass{updates}_{name}_gradient_nonzero"] = grads[name]["nonzero"] > 0
                checks[f"pass{updates}_earliest_image_gradient_nonzero"] = stage["earliest_past_image_gradient"]["nonzero"] > 0
                checks[f"pass{updates}_bf16_images_changed"] = stage["bf16_image_changed_fraction"] > 0
            persist()
            if not finite:
                raise FloatingPointError("Nonfinite gradients recorded; refusing an optimizer update")
            if updates < 2:
                optimizer.step()
                report["optimizer_updates"] += 1
                stage["parameters_after_update"] = {name: tensor_record(p) for name, p in visual.named_parameters()}
                persist()
                if not all(g["finite"] for g in stage["parameters_after_update"].values()):
                    raise FloatingPointError("Nonfinite visual parameters after diagnostic update")
            del result, features, bank, earliest
    finally:
        after = {"head": module_digest(head), "archive": module_digest(parent)}
        report["frozen_after_sha256"] = after
        checks["frozen_parent_and_head_unchanged"] = before == after
        checks["frozen_parent_and_head_no_gradients"] = all(p.grad is None for m in (head, parent) for p in m.parameters())
        persist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Original trained archive1250 parent")
    parser.add_argument("--output-dir", required=True, help="NEW verification directory; no saved model")
    parser.add_argument("--seed", type=int, default=9111)
    parser.add_argument("--preflight-only", action="store_true", help="Read cache/metadata only; no head/output")
    args = parser.parse_args(argv)
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    torch.set_num_threads(2)
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    checkpoint = Path(args.checkpoint).resolve()
    base = cache.manifest["model_path"]
    info = v7_checkpoint_info(base, checkpoint, expected_stage=1)
    if (info["config"]["mode"] != "archive" or info["step"] != 1250
            or info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]):
        raise ValueError("Requires same-cache trained archive1250 parent")
    output = Path(args.output_dir).resolve()
    validate_output_scope(output, cache.path, cache.manifest.get("dataset_path"), base, checkpoint)
    if output.exists():
        raise FileExistsError("Use a NEW verification output directory")
    episode_id, query, episode = select_query(cache, MappedEpisodes(cache))
    if args.preflight_only:
        print(f"[verify-v11] TRAIN episode={episode_id}, query={query}; no model/output created")
        return 0
    payloads = {name: sha(checkpoint / name) for name in
                ("checkpoint.json", "model.safetensors", "expert.safetensors", "cvom.safetensors", "training_state.pt")}
    source = source_hashes()
    config = VisualPatchConfig()
    plan = {"args": vars(args), "device": "cpu", "torch_threads": 2,
            "episode_id": episode_id, "query": query, "query_rule": "first sorted TRAIN episode, first decision>=3 with observed action",
            "prefix_frames": torch.as_tensor(episode["frames"][:query + 1]).tolist(),
            "prefix_is_demo": torch.as_tensor(episode["is_demo"][:query + 1]).tolist(),
            "visual_config": asdict(config), "camera_order": list(CAMERA_ORDER),
            "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 0., "updates": 2, "scope": "visual module ONLY"},
            "objective": "original seeded flow loss; fixed query/noise/time; no clipping or auxiliary",
            "parent_payload_sha256": payloads, "base_identity": info["metadata"]["base_model"],
            "cache_fingerprint": cache.manifest["fingerprint"], "cache_manifest_sha256": sha(cache.path / "manifest.json"),
            "source_sha256": source, "limitations": LIMITATIONS}
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "plan.json", plan)
    report = {"passed": False, "episode_id": episode_id, "query": query, "device": "cpu",
              "optimizer_updates": 0, "checks": {}, "gradient_passes": [], "limitations": LIMITATIONS}
    def persist():
        _atomic_json(output / "result.json", report)
    persist()
    try:
        print(f"[verify-v11] actual CPU head, TRAIN episode={episode_id}, query={query}; diagnostic only", flush=True)
        head = actual_head(base, "cpu")
        cfg = info["config"]
        with isolated_seed(args.seed, "cpu"):
            install_expert_lora(head, LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
            parent = RecurrentMemoryV7(MemoryV7Config(**cfg["memory"]))
            cvom = CVOMV7(parent.config)
            visual = VisualPatchMemoryV11(config)
        load_checkpoint_v7(checkpoint, parent, head, cvom)
        del cvom
        set_expert_trainable(head, False)
        head.eval().requires_grad_(False)
        parent.float().eval().requires_grad_(False)
        visual.train()
        report["base_dtype"] = str(next(head.parameters()).dtype)
        report["checks"]["actual_head_bf16"] = next(head.parameters()).dtype == torch.bfloat16
        report["checks"]["cached_prefix_bf16"] = all(
            episode["features"][index].dtype == torch.bfloat16 for index in range(query + 1))
        report["visual_parameter_count"] = sum(p.numel() for p in visual.parameters())
        run_diagnostic(head, parent, episode, query, visual, args.seed, report, persist)
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        report["checks"]["parent_checkpoint_files_unchanged"] = payloads == {name: sha(checkpoint / name) for name in payloads}
        report["checks"]["source_files_unchanged"] = source == source_hashes()
        report["passed"] = ("error" not in report and report["optimizer_updates"] == 2
                            and bool(report["checks"]) and all(report["checks"].values()))
        persist()
    print(json.dumps({key: report[key] for key in ("passed", "optimizer_updates", "checks")}, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
