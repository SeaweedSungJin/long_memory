#!/usr/bin/env python3
"""CPU observation-only V11 attention statistics on immutable fixed validation.

Compare checkpoint 0 with predeclared checkpoint 384. Reconstruct the core's
FP32 QK softmax using head_dim**-0.5, its exact valid-key mask and no causal
triangle (all keys already strictly precede the current observation). Compare
probability-weighted V with the same-input SDPA output numerically, not bitwise.

Mass alone is not selective retrieval: every group is compared with its valid
key availability. Raw age >64 and >48 are explicit diagnostic thresholds, not
assertions about the original model's exact receptive field. Neither weights
nor attention demonstrate correct cue use, causal value or task success. At
step zero, attention exists internally but the zero output projection prevents
it from affecting the Action Expert. No GT/action/state column is inspected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.checkpoint_visual_patch_v11 import checkpoint_info, load_checkpoint
from run_scripts.robomme.replay_visual_patch_v11 import OBSERVATION_KEYS, build_visual_patch_bank
from run_scripts.robomme.visual_patch_memory_v11 import (
    CAMERA_ORDER, PATCHES_PER_OBSERVATION, PATCHES_PER_VIEW, VisualPatchConfig, VisualPatchMemoryV11,
)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_hashes():
    paths = list((ROOT / "gr00t").rglob("*.py")) + [ROOT / "run_scripts/robomme" / name for name in (
        "audit_visual_attention_v11.py", "checkpoint_visual_patch_v11.py", "visual_patch_memory_v11.py",
        "replay_visual_patch_v11.py")]
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}


def module_hash(memory):
    result = hashlib.sha256()
    for name, tensor in sorted(memory.state_dict().items()):
        result.update(name.encode())
        result.update(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


@torch.no_grad()
def attention_probabilities(memory, current, bank):
    """Return [heads,162,T*162] probabilities and FP32 SDPA parity diagnostics."""
    if current.features.shape[0] != 1 or current.features.device.type != "cpu":
        raise ValueError("Attention audit requires one CPU observation")
    bank = memory._bank(bank, current)  # Same shape/finite/strict-past guard as READ.
    c = memory.config
    mask = bank.valid.repeat_interleave(PATCHES_PER_OBSERVATION, dim=1)
    if not bool(mask.any()):
        return torch.zeros(c.num_heads, PATCHES_PER_OBSERVATION, mask.shape[1]), {
            "has_history": False, "sdpa_close": True, "sdpa_max_abs": 0., "sdpa_relative_l2": 0.,
            "softmax_sum_max_error": 0., "scale": (c.hidden_dim // c.num_heads) ** -.5}
    with torch.autocast(device_type="cpu", enabled=False):
        tokens = torch.where(mask[..., None], bank.tokens.flatten(1, 2), 0.)
        query = current.queries
        key, value = memory.key_projection(tokens), memory.value_projection(tokens)

        def heads(value):
            return value.reshape(1, -1, c.num_heads, c.hidden_dim // c.num_heads).transpose(1, 2)

        query, key, value = heads(query), heads(key), heads(value)
        scale = (c.hidden_dim // c.num_heads) ** -.5
        logits = torch.matmul(query, key.transpose(-1, -2)) * scale
        probabilities = logits.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1)
        reconstructed = torch.matmul(probabilities, value)
        reference = F.scaled_dot_product_attention(query, key, value,
            attn_mask=mask[:, None, None, :], dropout_p=0., is_causal=False)
    difference = reconstructed - reference
    checks = {"has_history": True, "scale": scale,
        "sdpa_close": bool(torch.allclose(reconstructed, reference, rtol=5e-5, atol=5e-6)),
        "sdpa_max_abs": float(difference.abs().max()),
        "sdpa_relative_l2": float(difference.double().norm() / reference.double().norm().clamp_min(1e-30)),
        "softmax_sum_max_error": float((probabilities.sum(-1) - 1).abs().max())}
    if not bool(torch.isfinite(probabilities).all()) or not checks["sdpa_close"]:
        raise FloatingPointError(f"Core softmax/SDPA parity check failed: {checks}")
    return probabilities[0], checks


def group_statistics(probabilities, membership, valid):
    """Uniform-key availability versus mean per-head/current-patch group mass."""
    selected = membership.bool() & valid
    available = int(selected.sum()) / max(1, int(valid.sum()))
    mass = float(probabilities[..., selected].sum(-1).mean())
    return {"mass": mass, "availability": available, "mass_minus_availability": mass - available,
            "enrichment": mass / available if available > 0 else None}


def summarize_attention(memory, current, bank, probabilities):
    count = int(bank.valid.sum())
    valid = bank.valid[0].repeat_interleave(PATCHES_PER_OBSERVATION)
    ages = current.frames[0] - bank.frames[0]
    demo = bank.is_demo[0].repeat_interleave(PATCHES_PER_OBSERVATION)
    cameras = memory.patch_cameras.repeat(bank.tokens.shape[1])
    groups = {f"age_gt_{threshold}": group_statistics(probabilities,
              (ages > threshold).repeat_interleave(PATCHES_PER_OBSERVATION), valid) for threshold in (48, 64)}
    groups["demo"] = group_statistics(probabilities, demo, valid)
    for camera, name in enumerate(CAMERA_ORDER):
        groups["camera_" + name] = group_statistics(probabilities, cameras == camera, valid)
    observation_mass = probabilities.reshape(memory.config.num_heads, PATCHES_PER_OBSERVATION,
                                             bank.tokens.shape[1], PATCHES_PER_OBSERVATION).sum(-1)
    average = observation_mass.mean((0, 1))
    entropy = -(observation_mass * observation_mass.clamp_min(1e-30).log()).sum(-1)
    token_entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
    camera_matrix = {name: {key_name: group_statistics(
        probabilities[:, camera * PATCHES_PER_VIEW:(camera + 1) * PATCHES_PER_VIEW], cameras == key_camera, valid)
        for key_camera, key_name in enumerate(CAMERA_ORDER)} for camera, name in enumerate(CAMERA_ORDER)}
    return {"past_observations": count, "past_patch_tokens": int(valid.sum()),
        "past_frames": bank.frames[0].tolist(), "past_is_demo": bank.is_demo[0].tolist(),
        "past_valid": bank.valid[0].tolist(), "past_age_raw_frames": ages.tolist(),
        "observation_mass_mean_over_heads_queries": average.tolist(), "groups": groups,
        "camera_mass_by_current_query_camera": camera_matrix,
        "attended_age_raw_frames_mean": float((average * ages).sum()),
        "observation_entropy_mean": float(entropy.mean()) if count else None,
        "normalized_observation_entropy_mean": float(entropy.mean() / math.log(count)) if count > 1 else None,
        "effective_observations_mean": float(entropy.exp().mean()) if count else 0.,
        "effective_observations_fraction": float(entropy.exp().mean() / count) if count else None,
        "token_entropy_mean": float(token_entropy.mean()) if count else None,
        "normalized_token_entropy_mean": float(token_entropy.mean() / math.log(int(valid.sum()))) if count else None}


@torch.no_grad()
def audit_observation(memory, observations, query):
    """Uses five original-observation columns only; no head or supervised inputs."""
    bank = build_visual_patch_bank(memory, observations, query, camera_order=CAMERA_ORDER)
    current = memory.encode_observation(observations["features"][query][None],
        observations["image_masks"][query][None], observations["attention_masks"][query][None],
        torch.as_tensor(observations["frames"][query]).reshape(1),
        torch.as_tensor(observations["is_demo"][query]).reshape(1), camera_order=CAMERA_ORDER)
    probabilities, parity = attention_probabilities(memory, current, bank)
    return {"query_frame": int(current.frames[0]), "parity": parity,
            **summarize_attention(memory, current, bank, probabilities)}


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def aggregate(records):
    result = {"queries": len(records), "groups": {}}
    for name in records[0]["groups"]:
        values = [record["groups"][name] for record in records]
        mixed = [value for value in values if 0 < value["availability"] < 1]
        result["groups"][name] = {"mean_" + key: mean([value[key] for value in values])
            for key in ("mass", "availability", "mass_minus_availability", "enrichment")}
        result["groups"][name].update(available_queries=sum(v["availability"] > 0 for v in values),
            mixed_queries=len(mixed), mixed_mean_mass=mean([v["mass"] for v in mixed]),
            mixed_mean_availability=mean([v["availability"] for v in mixed]),
            mixed_mean_lift=mean([v["mass_minus_availability"] for v in mixed]))
    for key in ("attended_age_raw_frames_mean", "observation_entropy_mean", "normalized_observation_entropy_mean",
                "effective_observations_mean", "effective_observations_fraction", "normalized_token_entropy_mean"):
        result[key] = mean([row[key] for row in records])
    result["maximum_sdpa_abs_error"] = max(row["parity"]["sdpa_max_abs"] for row in records)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-count", type=int, choices=(8, 32), default=32,
                        help="All32 fixed validation queries, or deterministic first8 if resource limited")
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only audit requires explicit CUDA_VISIBLE_DEVICES='' and no initialized CUDA")
    torch.set_num_threads(2)
    run = Path(args.run_dir).resolve()
    config = json.loads((run / "run_config.json").read_text())
    plan = json.loads((run / "query_plan.json").read_text())
    selected = plan["validation"][:args.query_count]
    if len(plan["validation"]) != 32 or len(selected) != args.query_count or len({row[0] for row in selected}) != len(selected):
        raise ValueError("Expected immutable32 distinct-episode validation queries")
    cache = Path(config["train"]["cache_dir"]).resolve()
    manifest = json.loads((cache / "manifest.json").read_text())
    if any(eid not in manifest["splits"]["val"] for eid, _ in selected):
        raise ValueError("Attention audit selection must use the fixed held-out validation split")
    base = manifest["model_path"]
    checkpoints = {"step0": run / "checkpoint-000000", "step384": run / "checkpoint-000384"}
    output = Path(args.output_dir).resolve()
    validate_output_scope(output, run, cache, base, config["train"]["init_checkpoint"])
    if output.exists():
        raise FileExistsError("Use a NEW attention-audit output directory")
    protected = [run / "query_plan.json", run / "best_checkpoint.json", run / "run_config.json"]
    protected += [folder / name for folder in checkpoints.values()
                  for name in ("checkpoint.json", "visual.safetensors", "training_state.pt")]
    before = {str(path): sha(path) for path in protected}
    source_before = source_hashes()
    models, modules_before = {}, {}
    for label, path in checkpoints.items():
        info = checkpoint_info(base, path)
        if info["step"] != (0 if label == "step0" else 384) or info["metadata"]["plan_sha256"] != plan["sha256"]:
            raise ValueError("Checkpoint step or immutable validation-plan identity differs")
        with torch.random.fork_rng(devices=[]):
            models[label] = VisualPatchMemoryV11(VisualPatchConfig(**info["config"]["visual"]))
        load_checkpoint(path, models[label])
        models[label].eval().requires_grad_(False)
        modules_before[label] = module_hash(models[label])
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "plan.json", {"args": vars(args), "selection": selected,
        "selection_rule": "all32 fixed validation queries" if args.query_count == 32 else "deterministic first8 fixed validation queries",
        "query_plan_sha256": sha(run / "query_plan.json"), "checkpoint_files_before_sha256": before,
        "sources_before_sha256": source_before, "module_weights_before_sha256": modules_before,
        "attention_scale": "1/sqrt(hidden_dim/num_heads), default SDPA scale", "thresholds_raw_frames": [48, 64],
        "aggregation": "Each item averages group mass over4 heads and162 current patch queries; summaries macro-average fixed items. Effective observations average exp(per-head/query observation entropy).",
        "interpretation": "Attention mass/entropy is not correct-cue retrieval, causal use, success or accuracy evidence; step0 output remains zero."})
    report = {"passed": False, "records": {key: [] for key in models}, "checks": {}}
    records = {int(row["episode_id"]): row for row in manifest["episodes"]}
    started = time.monotonic()
    try:
        for index, (eid, query) in enumerate(selected):
            episode = torch.load(cache / records[eid]["path"], mmap=True, weights_only=True, map_location="cpu")
            if episode["episode_id"] != eid or episode["cache_fingerprint"] != manifest["fingerprint"]:
                raise ValueError("Cached episode provenance differs")
            observations = {key: episode[key] for key in OBSERVATION_KEYS}
            for label, memory in models.items():
                record = {"episode_id": eid, "query": query, **audit_observation(memory, observations, query)}
                report["records"][label].append(record)
            print(f"[visual-attention] {index+1}/{len(selected)} episode={eid} query={query}", flush=True)
            _atomic_json(output / "result.json", report)
        report["summary"] = {key: aggregate(values) for key, values in report["records"].items()}
        report["paired_mean_mass_change_step384_minus_step0"] = {name: mean([
            b["groups"][name]["mass"] - a["groups"][name]["mass"]
            for a, b in zip(report["records"]["step0"], report["records"]["step384"])])
            for name in report["records"]["step0"][0]["groups"]}
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["checkpoint_files_after_sha256"] = {str(path): sha(path) for path in protected}
        report["sources_after_sha256"] = source_hashes()
        report["module_weights_after_sha256"] = {key: module_hash(value) for key, value in models.items()}
        report["checks"] = {"checkpoint_files_unchanged": before == report["checkpoint_files_after_sha256"],
            "source_files_unchanged": source_before == report["sources_after_sha256"],
            "module_weights_unchanged": modules_before == report["module_weights_after_sha256"],
            "no_gradients_or_trainable_parameters": all(not p.requires_grad and p.grad is None for m in models.values() for p in m.parameters()),
            "cuda_not_initialized": not torch.cuda.is_initialized(),
            "complete_predeclared_selection": all(len(rows) == len(selected) for rows in report["records"].values()),
            "all_sdpa_parity_checks_pass": all(row["parity"]["sdpa_close"] for rows in report["records"].values() for row in rows)}
        report["passed"] = "error" not in report and all(report["checks"].values())
        _atomic_json(output / "result.json", report)
    print(json.dumps({"passed": report["passed"], "elapsed_seconds": report["elapsed_seconds"], "checks": report["checks"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
