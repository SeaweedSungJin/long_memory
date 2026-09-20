#!/usr/bin/env python3
"""Head-free CPU V11 residual common-mode diagnostic on the fixed32 queries.

No training, Action Expert, simulator, GPU, or parameter intervention. Temporary
forward hooks observe the original replay's pre-cast image residual and prior
image/time projections. Pre-LayerNorm component RMS is NOT additive causal
attribution: LN mixes components, and image_projection includes its bias.
"""
from __future__ import annotations

import argparse
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

from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.audit_visual_attention_v11 import module_hash, sha, source_hashes as attention_sources
from run_scripts.robomme.checkpoint_visual_patch_v11 import checkpoint_info, load_checkpoint
from run_scripts.robomme.replay_visual_patch_v11 import OBSERVATION_KEYS, replay_visual_patch
from run_scripts.robomme.visual_patch_memory_v11 import CAMERA_ORDER, VisualPatchConfig, VisualPatchMemoryV11

FORMULAS = {
    "residual": "R[p,d] is original output_projection(recalled), BEFORE BF16 cast/add; p=162 patches, d=2048",
    "common_energy": "N * sum_d(mean_p(R[p,d])**2)",
    "variation_energy": "sum_p,d((R[p,d]-mean_p(R[p,d]))**2)",
    "common_energy_fraction": "common_energy / sum_p,d(R[p,d]**2); null for zero-energy residual",
    "camera_common": "Sum camera-wise common energies with N=81, divided by total R energy",
    "direction_cosine": "cosine(mean_patch R_i, mean_patch R_j), ALL ordered i!=j pairs; diagonal excluded",
    "preLN_RMS": "sqrt(mean(component**2)) over all prior q*162*256 entries; broadcast time over patches, spatial over q",
}
LIMITATIONS = [
    "Pre-LN image/time/spatial RMS are not additive causal contribution fractions because LayerNorm mixes them.",
    "Image projection includes learned bias; time projection includes absolute raw frame features AND demo flag.",
    "Common across current patches does not mean constant across scenes/queries, or identify which input caused the common vector.",
    "Original cache frames/masks and original short tail are used; no action, GT, state, future receiver row or AE is read by replay.",
    "Camera order is declared front_view,wrist_view, two ordered contiguous81-patch9x9 grids excluding last4 short tokens; masks cannot certify pixel camera semantics.",
    "Raw frame labels are cached row indices (time_scale16), not elapsed seconds; retained cached observations are not every video frame.",
    "Fixed model-selected validation queries; no robot accuracy, task success, or new-model quality claim.",
]


def sources():
    result = attention_sources()
    result[str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
    return result


def statistics(values):
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    value = torch.tensor(values, dtype=torch.float64)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Nonfinite diagnostic statistic")
    return {"n": len(values), "min": float(value.min()), "median": float(value.quantile(.5)),
            "mean": float(value.mean()), "max": float(value.max())}


def energy_decomposition(value):
    """Orthogonal patch mean/centered decomposition; supports camera batches."""
    if value.ndim < 2 or value.shape[-2] < 1 or not bool(torch.isfinite(value).all()):
        raise ValueError("Expected finite [...,patch,channel] residual")
    value = value.double()
    mean = value.mean(-2, keepdim=True)
    total = value.square().sum()
    common = mean.square().sum() * value.shape[-2]
    centered = (value - mean).square().sum()
    return {"total_energy": float(total), "common_energy": float(common), "variation_energy": float(centered),
            "common_energy_fraction": float(common / total) if total else None,
            "decomposition_abs_error": float((total - common - centered).abs())}


def direction_cosines(vectors):
    if vectors.ndim != 2 or not bool(torch.isfinite(vectors).all()):
        raise ValueError("Expected finite query mean vectors")
    norms = vectors.norm(dim=-1)
    if bool((norms == 0).any()):
        raise ValueError("Direction is undefined for a zero query-mean residual")
    value = torch.nn.functional.normalize(vectors, dim=-1)
    matrix = value @ value.T
    mask = ~torch.eye(len(value), dtype=torch.bool, device=value.device)
    return matrix, statistics(matrix[mask].tolist())


@torch.no_grad()
def audit_query(memory, observations, query):
    if type(query) is not int or query < 1:
        raise ValueError("Common-mode audit requires nonempty past history")
    if memory.training or any(p.requires_grad or p.device.type != "cpu" for p in memory.parameters()):
        raise ValueError("Audit requires frozen eval CPU visual memory")
    captured = {}
    def hook(name):
        def save(module, args, output):
            captured.setdefault(name, []).append(output.detach())
        return save
    handles = [getattr(memory, name).register_forward_hook(hook(name)) for name in
               ("image_projection", "time_projection", "output_projection")]
    try:
        fused, bank = replay_visual_patch(memory, observations, query, camera_order=CAMERA_ORDER,
                                         checkpoint_encoding=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured["output_projection"]) != 1:
        raise RuntimeError("Expected exactly one original visual READ output")
    residual = captured["output_projection"][0][0]
    content, time = captured["image_projection"][0], captured["time_projection"][0]
    if content.shape != (query, 162, memory.config.hidden_dim) or time.shape != (query, memory.config.hidden_dim):
        raise RuntimeError("Hooks did not capture the complete original prior batch")
    spatial = (memory.camera_embedding(memory.patch_cameras) + memory.row_embedding(memory.patch_rows)
               + memory.column_embedding(memory.patch_columns))
    decomposition = energy_decomposition(residual)
    camera = energy_decomposition(residual.reshape(2, 81, -1))
    current = observations["features"][query]
    images = observations["image_masks"][query].clone() & observations["attention_masks"][query]
    images[-memory.config.num_short_tokens:] = False
    original = current[images].float()
    actual = fused[0, images].float() - original
    row = {"query": query, "query_raw_frame": int(observations["frames"][query]),
           "prior_raw_frames": bank.frames[0].tolist(), "prior_is_demo": bank.is_demo[0].tolist(),
           "prior_observations": query, "prior_patch_tokens": query * 162,
           "residual_shape": list(residual.shape), "residual_norm_precast": float(residual.norm()),
           "residual_norm_actual_cast_add": float(actual.norm()), "original_image_norm": float(original.norm()),
           "common_patch_decomposition": decomposition, "camera_common_decomposition": camera,
           "content_preLN_RMS": float(content.square().mean().sqrt()),
           "spatial_preLN_RMS": float(spatial.square().mean().sqrt()),
           "time_demo_preLN_RMS": float(time.square().mean().sqrt()),
           "only_current_images_changed": torch.equal(fused[0, ~images], current[~images])}
    if not row["only_current_images_changed"]:
        raise RuntimeError("Frozen replay changed non-image/current short positions")
    return row, residual.mean(0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=384)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Explicit CUDA_VISIBLE_DEVICES='' and uninitialized CUDA required")
    torch.set_num_threads(2)
    run = Path(args.run_dir).resolve()
    config = json.loads((run / "run_config.json").read_text())
    saved_plan = json.loads((run / "query_plan.json").read_text())
    query_plan = {k: v for k, v in saved_plan.items() if k != "sha256"}
    digest = hashlib.sha256(json.dumps(query_plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    cache = Path(config["train"]["cache_dir"]).resolve()
    manifest = json.loads((cache / "manifest.json").read_text())
    checkpoint = run / f"checkpoint-{args.checkpoint_step:06d}"
    info = checkpoint_info(manifest["model_path"], checkpoint)
    if (args.checkpoint_step <= 0 or info["step"] != args.checkpoint_step
            or digest != saved_plan["sha256"] or info["metadata"]["plan_sha256"] != digest
            or info["metadata"]["cache_fingerprint"] != manifest["fingerprint"]):
        raise ValueError("Immutable checkpoint/query-plan/cache identity differs")
    selected = query_plan["validation"]
    if (len(selected) != 32 or len({eid for eid, _ in selected}) != 32
            or any(eid not in manifest["splits"]["val"] or type(q) is not int or q < 1 for eid, q in selected)):
        raise ValueError("Requires all32 original nonempty held-out query identities")
    output = Path(args.output_dir).resolve()
    validate_output_scope(output, run, cache, manifest["model_path"], info["metadata"]["frozen_parent"]["path"])
    if output.exists():
        raise FileExistsError("Use a NEW diagnostic directory")
    records = {int(r["episode_id"]): r for r in manifest["episodes"]}
    paths = [run / "query_plan.json", run / "run_config.json", cache / "manifest.json"]
    paths += [checkpoint / name for name in ("checkpoint.json", "visual.safetensors", "training_state.pt")]
    paths += [Path(info["metadata"]["frozen_parent"]["path"]) / name
              for name in info["metadata"]["frozen_parent"]["files_sha256"]]
    episodes = {}
    for eid, _ in selected:
        path = (cache / records[eid]["path"]).resolve()
        stat = path.stat()
        if not path.is_relative_to(cache) or query_plan["files"][str(eid)] != [str(path), stat.st_size, stat.st_mtime_ns]:
            raise ValueError("Cached receiver file changed since the training plan")
        episodes[eid] = path
        paths.append(path)
    before = {str(path): sha(path) for path in paths}
    source_before = sources()
    with torch.random.fork_rng(devices=[]):
        memory = VisualPatchMemoryV11(VisualPatchConfig(**info["config"]["visual"]))
    loaded = load_checkpoint(checkpoint, memory)
    if any(loaded[key] != info[key] for key in ("step", "config", "metadata")):
        raise ValueError("Checkpoint changed during loading")
    memory.eval().requires_grad_(False)
    weights_before = module_hash(memory)
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "plan.json", {"args": vars(args), "queries": selected, "query_plan_sha256": digest,
        "camera_order": list(CAMERA_ORDER), "formulas": FORMULAS, "limitations": LIMITATIONS,
        "source_sha256": source_before, "files_sha256": before, "module_weights_sha256": weights_before,
        "runtime": {"torch": torch.__version__, "threads": torch.get_num_threads(), "device": "cpu"},
        "optimizer_updates": 0, "action_expert_loaded": False})
    result = {"passed": False, "records": [], "formulas": FORMULAS, "limitations": LIMITATIONS}
    vectors = []
    try:
        for eid, query in selected:
            ep = torch.load(episodes[eid], map_location="cpu", weights_only=True, mmap=True)
            if ep["episode_id"] != eid or ep["cache_fingerprint"] != manifest["fingerprint"]:
                raise ValueError("Cached receiver identity changed")
            row, vector = audit_query(memory, {key: ep[key] for key in OBSERVATION_KEYS}, query)
            result["records"].append({"episode_id": eid, **row})
            vectors.append(vector)
            _atomic_json(output / "result.json", result)
        matrix, cosines = direction_cosines(torch.stack(vectors))
        result["pairwise_direction_cosines"] = [{"left_episode": selected[i][0], "right_episode": selected[j][0],
                                                "cosine": float(matrix[i, j])}
                                               for i in range(32) for j in range(i + 1, 32)]
        result["summary"] = {key: statistics([row[key] for row in result["records"]]) for key in
            ("residual_norm_precast", "residual_norm_actual_cast_add", "original_image_norm",
             "content_preLN_RMS", "spatial_preLN_RMS", "time_demo_preLN_RMS")}
        for name in ("common_patch_decomposition", "camera_common_decomposition"):
            result["summary"][name] = statistics([row[name]["common_energy_fraction"] for row in result["records"]])
        result["summary"].update(across_query_direction_cosines=cosines,
            time_demo_RMS_exceeds_content_count=sum(row["time_demo_preLN_RMS"] > row["content_preLN_RMS"] for row in result["records"]))
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        result["checks"] = {"source_unchanged": sources() == source_before,
            "files_unchanged": all(sha(path) == value for path, value in before.items()),
            "weights_unchanged": module_hash(memory) == weights_before,
            "no_gradients_or_trainable_parameters": all(p.grad is None and not p.requires_grad for p in memory.parameters()),
            "cuda_not_initialized": not torch.cuda.is_initialized(), "complete32": len(result["records"]) == 32}
        result["passed"] = "error" not in result and all(result["checks"].values())
        _atomic_json(output / "result.json", result)
    print(json.dumps({"passed": result["passed"], "checks": result["checks"], "summary": result.get("summary")}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
