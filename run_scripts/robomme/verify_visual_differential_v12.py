#!/usr/bin/env python3
"""Actual frozen-AE wiring check for both V12 arms; no saved model.

One predeclared TRAIN query, fixed flow noise/time, two transient updates per
arm. This is not convergence, useful retrieval, validation or robot accuracy.
Original V11 step0 weights establish identical initial visual parameters.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import load_file

from run_scripts.robomme import verify_visual_patch_v11 as old
from run_scripts.robomme.checkpoint_visual_patch_v11 import checkpoint_info as initial_info
from run_scripts.robomme.visual_differential_memory_v12 import (
    CAMERA_ORDER, VisualDifferentialConfig, VisualDifferentialMemoryV12,
)

MODES = ("differential", "current_only")
LIMITATIONS = old.LIMITATIONS + [
    "Two diagnostic updates per independently initialized arm; no saved weights/optimizer.",
    "Current-only still appends past observations but its new visual output must have zero past-content gradient.",
    "A GPU check asserts explicit zero-init forward parity and finite gradients, not arbitrary GPU bitwise backward reproducibility.",
]


def source_hashes():
    result = old.source_hashes()
    for name in ("verify_visual_differential_v12.py", "visual_differential_memory_v12.py",
                 "checkpoint_visual_patch_v11.py"):
        path = ROOT / "run_scripts/robomme" / name
        result[str(path.relative_to(ROOT))] = old.sha(path)
    return result


def tracked_features(memory, observations, query):
    """Only observation columns; current WRITE is never performed before READ."""
    if type(query) is not int or query < 1:
        raise ValueError("Require a nonempty causal prefix")
    device = next(memory.parameters()).device
    bank, earliest, current_leaf, current = None, None, None, None
    for index in range(query + 1):
        feature = observations["features"][index].detach().to(device)[None]
        if index in (0, query):
            feature = feature.clone().requires_grad_(True)
            if index == 0:
                earliest = feature
            else:
                current_leaf = feature
        current = memory.encode_observation(feature,
            observations["image_masks"][index].to(device).bool()[None],
            observations["attention_masks"][index].to(device).bool()[None],
            torch.as_tensor(observations["frames"][index], device=device).reshape(1),
            torch.as_tensor(observations["is_demo"][index], device=device).reshape(1),
            camera_order=CAMERA_ORDER)
        if index < query:
            bank = memory.append(bank, current)
    return memory.read(current, bank), bank, earliest, current_leaf


def image_mask(episode, query, device):
    mask = (episode["image_masks"][query].to(device).bool()
            & episode["attention_masks"][query].to(device).bool())[None].clone()
    mask[:, -4:] = False
    return mask


def diagnose_arm(head, parent, episode, query, visual, seed, report, persist):
    observations = {key: episode[key] for key in old.OBSERVATION_KEYS}
    device = next(visual.parameters()).device
    checks = report.setdefault("checks", {})
    with torch.no_grad():
        fused, _ = old.replay_queries(parent, episode, [query], mode="archive")[query]
        original, _, _, _ = old.cached_inputs(head, episode, query, fused)
        reference = old.expert_episode_flow_loss(head, episode, query, fused, seed=seed)
        reference_prediction = reference["prediction"].detach()
        reference_loss = reference["loss"].detach()
        original_generated = old.generated_action(head, episode, query, fused, seed=seed + 1)
    noise, time = old.sample_noise_time(head, episode["targets"][query].to(original)[None], seed)
    report["fixed_noise_sha256"] = old.tensor_digest(noise)
    report["initial_visual_sha256"] = old.module_digest(visual)
    report["original_flow_loss"] = float(reference_loss)
    report["passes"] = []
    optimizer = torch.optim.AdamW(visual.parameters(), lr=1e-4, weight_decay=0.)
    old_mask, now_mask = image_mask(episode, 0, device), image_mask(episode, query, device)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        features, bank, earliest, current = tracked_features(visual, observations, query)
        features = old.replace_short(features, fused)
        result = old.flow_at_features(head, episode, query, features, noise, time)
        row = {"after_updates": step, "flow_loss": float(result["loss"].detach()),
               "past_observations": bank.tokens.shape[1],
               "image_delta": old.tensor_record((features.detach().float() - original.float())[now_mask]),
               "image_changed_fraction": float((features.detach()[now_mask] != original[now_mask]).float().mean())}
        report["passes"].append(row)
        checks[f"pass{step}_only_images_modified"] = torch.equal(features.detach()[~now_mask], original[~now_mask])
        checks[f"pass{step}_full_bank"] = bank.tokens.shape[1] == query and bank.content.shape == bank.tokens.shape
        checks[f"pass{step}_finite_loss"] = bool(torch.isfinite(result["loss"]))
        if step == 0:
            checks["zero_features_exact"] = torch.equal(features.detach(), original)
            checks["zero_flow_prediction_exact"] = torch.equal(result["prediction"].detach(), reference_prediction)
            checks["zero_flow_loss_exact"] = torch.equal(result["loss"].detach(), reference_loss)
            generated = old.generated_at_features(head, episode, query, features.detach(), seed + 1)
            checks["zero_generated_euler4_exact"] = torch.equal(generated, original_generated)
        result["loss"].backward()
        gradients = {name: old.tensor_record(p.grad) for name, p in visual.named_parameters()}
        row["parameter_gradients"] = gradients
        row["earliest_image_gradient"] = old.tensor_record(None if earliest.grad is None else earliest.grad[old_mask])
        row["current_image_gradient"] = old.tensor_record(None if current.grad is None else current.grad[now_mask])
        checks[f"pass{step}_finite_gradients"] = all(v["finite"] for v in gradients.values()) and all(
            row[name]["finite"] for name in ("earliest_image_gradient", "current_image_gradient"))
        checks[f"pass{step}_output_gradient_nonzero"] = gradients["output_projection.weight"]["nonzero"] > 0
        if step == 0:
            checks["zero_upstream_parameter_gradients"] = all(
                g["nonzero"] == 0 for name, g in gradients.items() if name != "output_projection.weight")
        else:
            for name in ("image_projection.weight", "query_projection.weight", "key_projection.weight", "value_projection.weight"):
                checks[f"pass{step}_{name}_gradient_nonzero"] = gradients[name]["nonzero"] > 0
            checks[f"pass{step}_current_image_gradient_nonzero"] = row["current_image_gradient"]["nonzero"] > 0
            checks[f"pass{step}_bf16_image_changed"] = row["image_changed_fraction"] > 0
        old_nonzero = row["earliest_image_gradient"]["nonzero"]
        checks[f"pass{step}_past_gradient_contract"] = (
            old_nonzero > 0 if visual.read_mode == "differential" and step > 0 else old_nonzero == 0)
        persist()
        if not all(checks.values()):
            raise RuntimeError("Diagnostic check failed; preserving records without further optimizer updates")
        if step < 2:
            optimizer.step()
            report["optimizer_updates"] += 1
            if not all(bool(torch.isfinite(p).all()) for p in visual.parameters()):
                raise FloatingPointError("Nonfinite transient visual weights")
        del features, bank, earliest, current, result
    report["final_visual_sha256"] = old.module_digest(visual)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--checkpoint", required=True, help="Frozen archive1250 parent")
    p.add_argument("--initial-visual", default="runs/long_memory/v11_visual_pilot512_20260916/checkpoint-000000")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=9111)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)
    if args.seed != 9111 or args.device not in ("cpu", "cuda:0"):
        raise ValueError("Predeclared seed9111 and CPU or explicitly isolated cuda:0 required")
    torch.set_num_threads(2)
    cache = old.EpisodeCache(args.cache_dir)
    old.validate_cache_checkpoint(cache.manifest)
    base, parent_path = Path(cache.manifest["model_path"]), Path(args.checkpoint).resolve()
    info = old.v7_checkpoint_info(base, parent_path, expected_stage=1)
    init_path = Path(args.initial_visual).resolve()
    init = initial_info(base, init_path)
    if (info["step"] != 1250 or info["config"]["mode"] != "archive" or init["step"] != 0
            or init["metadata"]["frozen_parent"]["path"] != str(parent_path)
            or info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]):
        raise ValueError("Expected same-cache archive1250 and its original V11 step0")
    output = Path(args.output_dir).resolve()
    old.validate_output_scope(output, cache.path, cache.manifest.get("dataset_path"), base, parent_path, init_path)
    if output.exists():
        raise FileExistsError("Use a NEW diagnostic output directory")
    eid, query, episode = old.select_query(cache, old.MappedEpisodes(cache))
    sources = source_hashes()
    files = [parent_path / name for name in ("checkpoint.json", "model.safetensors", "expert.safetensors", "cvom.safetensors", "training_state.pt")]
    files += [init_path / name for name in ("checkpoint.json", "visual.safetensors", "training_state.pt")]
    files += [cache.path / "manifest.json"]
    episode_path = (cache.path / cache._records[eid]["path"]).resolve()
    if not episode_path.is_relative_to(cache.path):
        raise ValueError("Selected episode payload escapes cache")
    files.append(episode_path)
    files += list(base.glob("*.safetensors")) + list(base.glob("*.json"))
    hashes = {str(path): old.sha(path) for path in files}
    if args.preflight_only:
        print(json.dumps({"preflight_passed": True, "episode_id": eid, "query": query,
            "modes": MODES, "cuda_initialized": torch.cuda.is_initialized(), "output_created": False}))
        return 0
    plan = {"args": vars(args), "episode_id": eid, "query": query, "modes": MODES,
            "source_sha256": sources, "files_sha256": hashes, "cache_fingerprint": cache.manifest["fingerprint"],
            "query_rule": "first sorted TRAIN episode, first valid decision>=3", "visual_config": asdict(VisualDifferentialConfig()),
            "updates_per_arm": 2, "optimizer": "AdamW visual-only lr1e-4 weight_decay0", "limitations": LIMITATIONS}
    output.mkdir(parents=True, exist_ok=False)
    old._atomic_json(output / "plan.json", plan)
    report = {"passed": False, "arms": {}, "checks": {}, "limitations": LIMITATIONS}
    persist = lambda: old._atomic_json(output / "result.json", report)
    try:
        head = old.actual_head(base, args.device)
        cfg = info["config"]
        with old.isolated_seed(args.seed, args.device):
            old.install_expert_lora(head, old.LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
            parent = old.RecurrentMemoryV7(old.MemoryV7Config(**cfg["memory"])).to(args.device)
            cvom = old.CVOMV7(parent.config).to(args.device)
        old.load_checkpoint_v7(parent_path, parent, head, cvom)
        old.set_expert_trainable(head, False)
        head.eval().requires_grad_(False)
        parent.float().eval().requires_grad_(False)
        cvom.eval().requires_grad_(False)
        frozen = {name: old.module_digest(m) for name, m in (("head", head), ("parent", parent), ("cvom", cvom))}
        report["frozen_before_sha256"] = frozen
        report["checks"]["actual_bf16_native_euler4"] = next(head.parameters()).dtype == torch.bfloat16 and head.num_inference_timesteps == 4
        initial_state = load_file(str(init_path / "visual.safetensors"), device="cpu")
        for mode in MODES:
            with old.isolated_seed(args.seed, args.device):
                visual = VisualDifferentialMemoryV12(VisualDifferentialConfig(), read_mode=mode).to(args.device)
            visual.load_state_dict(initial_state, strict=True)
            arm = {"read_mode": mode, "optimizer_updates": 0, "parameter_count": sum(p.numel() for p in visual.parameters())}
            report["arms"][mode] = arm
            diagnose_arm(head, parent, episode, query, visual, args.seed, arm, persist)
            del visual
        report["checks"]["identical_initial_visual_weights"] = len({r["initial_visual_sha256"] for r in report["arms"].values()}) == 1
        after = {name: old.module_digest(m) for name, m in (("head", head), ("parent", parent), ("cvom", cvom))}
        report["frozen_after_sha256"] = after
        report["checks"]["frozen_modules_unchanged"] = after == frozen
        report["checks"]["frozen_no_gradients"] = all(p.grad is None for m in (head, parent, cvom) for p in m.parameters())
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(report["error"]["traceback"], flush=True)
    finally:
        report["checks"].update(source_unchanged=sources == source_hashes(),
            input_files_unchanged=hashes == {path: old.sha(path) for path in hashes})
        report["passed"] = ("error" not in report and set(report["arms"]) == set(MODES)
            and all(report["checks"].values()) and all(
                r["optimizer_updates"] == 2 and all(r["checks"].values()) for r in report["arms"].values()))
        persist()
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
