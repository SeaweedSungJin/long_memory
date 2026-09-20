#!/usr/bin/env python3
"""Bounded V13 image-extraction proof, not a policy, trainer or success test.

Run metadata preflight first. Actual execution needs an explicitly available
device and loads the original frozen HAMLET. Native processing uses observation
state (never GT actions); the proposed sidecar path accepts RGB/instruction only.
Numerical comparisons have NO invented tolerance: exact and finite are reported
alongside max error, relative L2 and repeated-native numerical noise. Nonexact
feature agreement requires review even when all structural checks pass.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

INPUT_KEYS = ("input_ids", "attention_mask", "pixel_values")
LIMITATIONS = [
    "A bounded TRAIN-only extraction proof; no bulk cache, training, validation or RoboMME accuracy.",
    "No online session/parent APPEND/Action Expert action parity has been demonstrated by this tool.",
    "RGB plus the currently supplied instruction are the sidecar's only model inputs; no GT actions are read.",
    "Native reference additionally preprocesses observed robot state, which Eagle does not consume.",
    "Known ep0 final-demo images demonstrate ingestion coverage only, not label-based selection or useful retrieval.",
    "Raw global RNG consumption and externally restored caller RNG are separate observations; processor-private RNG is not claimed restored.",
    "No arbitrary numerical tolerance is asserted; nonexact cache/features require interpretation against repeated-native noise.",
]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)]).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def module_sha(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_sha(value).encode())
    return digest.hexdigest()


def numerical_comparison(reference, candidate):
    """Measurements only. Relative L2 is undefined for a zero reference norm."""
    same_shape = reference.shape == candidate.shape
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    result = {"same_shape": same_shape, "same_dtype": reference.dtype == candidate.dtype,
              "reference_dtype": str(reference.dtype), "candidate_dtype": str(candidate.dtype),
              "finite": finite, "exact": same_shape and reference.dtype == candidate.dtype
              and torch.equal(reference.detach().cpu(), candidate.detach().cpu()),
              "max_abs": None, "relative_l2": None}
    if same_shape and finite:
        ref, cur = reference.detach().cpu().double(), candidate.detach().cpu().double()
        delta, norm = cur - ref, float(ref.norm())
        result.update(max_abs=float(delta.abs().max()) if delta.numel() else 0.,
                      relative_l2=float(delta.norm()) / norm if norm else None)
    return result


def input_clone(value):
    """Copy tensor leaves for evidence without flattening Eagle's image list."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if type(value) in (list, tuple):
        return type(value)(input_clone(child) for child in value)
    if type(value) is dict:
        return {key: input_clone(child) for key, child in value.items()}
    raise TypeError(f"Unsupported prepared-input leaf/container: {type(value).__name__}")


def input_signature(value):
    """Container kinds/order plus dtype/shape/content hashes of individual leaves."""
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha(value)}
    if type(value) in (list, tuple):
        return {"container": type(value).__name__, "items": [input_signature(child) for child in value]}
    if type(value) is dict:
        return {"container": "dict", "items": {key: input_signature(child) for key, child in value.items()}}
    raise TypeError(f"Unsupported prepared-input leaf/container: {type(value).__name__}")


def compare_input_tree(reference, candidate):
    """Exact means identical container types AND every matching tensor leaf."""
    if isinstance(reference, torch.Tensor) and isinstance(candidate, torch.Tensor):
        return numerical_comparison(reference, candidate)
    if type(reference) is not type(candidate):
        return {"exact": False, "structure_matches": False,
                "reference_type": type(reference).__name__, "candidate_type": type(candidate).__name__}
    if type(reference) in (list, tuple):
        children = [compare_input_tree(a, b) for a, b in zip(reference, candidate)]
        same = len(reference) == len(candidate)
        return {"container": type(reference).__name__, "reference_length": len(reference),
                "candidate_length": len(candidate), "structure_matches": same,
                "exact": same and all(row["exact"] for row in children), "items": children}
    if type(reference) is dict:
        same = reference.keys() == candidate.keys()
        children = {key: compare_input_tree(reference[key], candidate[key]) for key in reference.keys() & candidate.keys()}
        return {"container": "dict", "reference_keys": sorted(reference), "candidate_keys": sorted(candidate),
                "structure_matches": same, "exact": same and all(row["exact"] for row in children.values()), "items": children}
    raise TypeError(f"Unsupported prepared-input leaf/container: {type(reference).__name__}")


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_tree(v) for v in value)
    return copy.deepcopy(value)


def tree_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.device == right.device and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)


def rng_state():
    # Do not initialize CUDA in metadata preflight or CPU-only tests.
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [v.clone() for v in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def version_state(module):
    return {name: (id(value), value._version, value.requires_grad, value.grad is None)
            for name, value in list(module.named_parameters()) + list(module.named_buffers())}


@contextmanager
def preserve_call_state(head):
    """Restore the original cache object and global RNG even after a failed proof."""
    rng, original = rng_state(), getattr(head, "_memory_cache", None)
    saved = clone_tree(original)
    try:
        yield rng
    finally:
        try:
            if isinstance(original, torch.Tensor):
                with torch.inference_mode():
                    original.copy_(saved)
                head._memory_cache = original
            else:
                head._memory_cache = clone_tree(saved)
        finally:
            restore_rng(rng)


@contextmanager
def backbone_capture(backbone):
    calls = []

    def before(module, args):
        if len(args) != 1 or not isinstance(args[0], Mapping):
            raise ValueError("Expected one native backbone input mapping")
        actual = args[0]
        if any(k not in actual for k in INPUT_KEYS):
            raise ValueError("Missing actual backbone input tensor")
        calls.append({"inputs": {k: input_clone(actual[k]) for k in INPUT_KEYS}})

    def after(module, args, output):
        calls[-1]["outputs"] = {k: output[k].detach().clone() for k in
            ("backbone_features", "backbone_attention_mask", "image_mask")}

    hooks = [backbone.register_forward_pre_hook(before), backbone.register_forward_hook(after)]
    try:
        yield calls
    finally:
        for hook in hooks:
            hook.remove()


@contextmanager
def forbid_action_calls(head):
    counts = {"action_head_forward": 0, "action_dit_forward": 0}
    hooks = []
    for name, module in (("action_head_forward", head), ("action_dit_forward", head.model)):
        def reject(mod, args, key=name):
            counts[key] += 1
            raise RuntimeError(f"Extraction must not execute {key}")
        hooks.append(module.register_forward_pre_hook(reject))
    try:
        yield counts
    finally:
        for hook in hooks:
            hook.remove()


def image_selection(output, *, short_tokens=4):
    feature, mask, attention = (output[k] for k in
        ("backbone_features", "image_mask", "backbone_attention_mask"))
    if (feature.ndim != 3 or feature.shape[0] != 1 or mask.shape != feature.shape[:2]
            or attention.shape != mask.shape or mask.dtype != torch.bool or attention.dtype != torch.bool):
        raise ValueError("Expected a single canonical sequence and matching masks")
    selected = mask.bool() & attention.bool()
    if bool(selected[:, -short_tokens:].any()):
        raise ValueError("Image mask overlaps final HAMLET short tokens")
    indices = selected[0].nonzero().flatten()
    if indices.numel() != 162:
        raise ValueError("Audited two-camera geometry requires exactly 162 valid image tokens")
    if not all(bool((part.diff() == 1).all()) for part in indices.split(81)):
        raise ValueError("Each camera must have 81 contiguous image tokens")
    if int(indices[81] - indices[80]) <= 1 or bool((mask.bool() & ~attention.bool()).any()):
        raise ValueError("Camera runs must be separate and every image token attended")
    return feature[0, indices].reshape(2, 81, -1).detach().cpu().to(torch.bfloat16), indices.cpu()


def inputs_exact(reference, candidate):
    if set(reference) != set(INPUT_KEYS) or set(candidate) != set(INPUT_KEYS):
        raise ValueError("Prepared input mapping must contain exactly Eagle's three input keys")
    return {key: compare_input_tree(reference[key], candidate[key]) for key in INPUT_KEYS}


def verify_frame(model, native_call, prepare_direct, extract_direct, *, cached=None):
    """Testable proof harness. Native returns (three pre-cast tensors, processed output).

    The direct helper is called without a nested RNG context: its genuine global
    RNG consumption is measured, and only then is outer caller state restored.
    """
    head, backbone = model.action_head, model.backbone
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Proof requires a frozen eval model without parameter gradients")
    versions, caller_rng = version_state(model), rng_state()
    cache_object, cache_value = getattr(head, "_memory_cache", None), clone_tree(getattr(head, "_memory_cache", None))
    row = {"checks": {}, "comparisons": {}}
    with preserve_call_state(head), torch.no_grad(), forbid_action_calls(head) as action_counts:
        native_outputs, native_inputs, captures = [], [], []
        for _ in range(2):
            restore_rng(caller_rng)
            head._memory_cache = clone_tree(cache_value)
            with backbone_capture(backbone) as captured:
                inputs, output = native_call()
            if len(captured) != 1:
                raise ValueError("Native reference must execute exactly one backbone call")
            native_inputs.append(inputs)
            native_outputs.append(output)
            captures.append(captured[0])
        reference, indices = image_selection(native_outputs[0])
        repeated, repeated_indices = image_selection(native_outputs[1])
        row["comparisons"]["native_repeat"] = numerical_comparison(reference, repeated)
        row["comparisons"]["prepared_native_repeat"] = inputs_exact(native_inputs[0], native_inputs[1])
        row["comparisons"]["device_inputs_native_repeat"] = inputs_exact(captures[0]["inputs"], captures[1]["inputs"])
        row["checks"]["native_repeat_indices_exact"] = torch.equal(indices, repeated_indices)
        for key in ("image_mask", "backbone_attention_mask"):
            row["checks"][f"native_repeat_{key}_exact"] = numerical_comparison(
                captures[0]["outputs"][key], captures[1]["outputs"][key])["exact"]

        restore_rng(caller_rng)
        head._memory_cache = cache_object
        direct_prepared = prepare_direct()
        row["raw_prepare_rng_unchanged"] = tree_equal(caller_rng, rng_state())
        row["comparisons"]["prepared_direct"] = inputs_exact(native_inputs[0], direct_prepared)
        restore_rng(caller_rng)
        before_direct_rng = rng_state()
        def reject_process(*args, **kwargs):
            raise RuntimeError("Image-only extraction called HAMLET process_backbone_output")
        with patch.object(head, "process_backbone_output", side_effect=reject_process), backbone_capture(backbone) as captured:
            images, preparation_hashes = extract_direct()
        row["raw_extraction_rng_unchanged"] = tree_equal(before_direct_rng, rng_state())
        if len(captured) != 1:
            raise ValueError("Direct extraction must execute exactly one backbone call")
        if images.requires_grad or images.dtype != torch.bfloat16 or images.shape != reference.shape:
            raise ValueError("Direct images must be detached BF16 [2,81,D]")
        direct_capture = captured[0]
        row["comparisons"]["device_inputs_direct"] = inputs_exact(captures[0]["inputs"], direct_capture["inputs"])
        for key in ("image_mask", "backbone_attention_mask"):
            row["checks"][f"direct_{key}_exact"] = numerical_comparison(captures[0]["outputs"][key], direct_capture["outputs"][key])["exact"]
        _, direct_indices = image_selection(direct_capture["outputs"])
        row["checks"]["direct_indices_exact"] = torch.equal(indices, direct_indices)
        row["checks"]["direct_memory_cache_exact"] = head._memory_cache is cache_object and tree_equal(head._memory_cache, cache_value)
        row["prepared_input_signatures"] = {k: input_signature(v) for k, v in direct_prepared.items()}
        row["actual_encoder_input_signatures"] = {k: input_signature(v) for k, v in direct_capture["inputs"].items()}
        row["extract_preparation_hashes"] = preparation_hashes
        row["comparisons"]["direct_vs_native"] = numerical_comparison(reference, images)
        if cached is not None:
            cached_images, cached_indices = image_selection(cached)
            row["checks"]["cached_indices_exact"] = torch.equal(indices, cached_indices)
            for key in ("image_mask", "backbone_attention_mask"):
                row["checks"][f"cached_{key}_exact"] = numerical_comparison(native_outputs[0][key], cached[key])["exact"]
            row["comparisons"]["native_vs_saved_cache"] = numerical_comparison(cached_images, reference)
            row["comparisons"]["direct_vs_saved_cache"] = numerical_comparison(cached_images, images)
        row["action_calls"] = action_counts
        row["checks"]["zero_action_calls"] = not any(action_counts.values())
    row["checks"]["caller_rng_restored"] = tree_equal(caller_rng, rng_state())
    row["checks"]["caller_cache_restored"] = head._memory_cache is cache_object and tree_equal(head._memory_cache, cache_value)
    row["checks"]["frozen_versions_and_grad_flags_unchanged"] = version_state(model) == versions
    for category in ("prepared_native_repeat", "device_inputs_native_repeat", "prepared_direct", "device_inputs_direct"):
        row["checks"][f"{category}_exact"] = all(v["exact"] for v in row["comparisons"][category].values())
    numerical = [v for v in row["comparisons"].values() if "finite" in v]
    row["checks"]["finite_features"] = all(v["finite"] and v["same_shape"] for v in numerical)
    row["structural_passed"] = all(row["checks"].values())
    row["all_feature_comparisons_exact"] = all(v["exact"] for v in numerical)
    return row


def native_batch(processor, observation, states, *, embodiment):
    """Genuine original observation preprocessing; no model or action target."""
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import VLAStepData, MessageType
    step = VLAStepData(images=observation["images"], states=states, actions=None,
                       text=observation["text"], embodiment=EmbodimentTag(embodiment), is_demonstration=True)
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
    return processor.collator([processed])["inputs"]


def native_preparation(model, processor, observation, states, *, embodiment):
    # Match _extract_episode's inference_mode AND CUDA BF16 autocast, not only
    # its weights/device. The native cache producer wraps this whole sequence.
    with torch.inference_mode(), torch.autocast(device_type=model.device.type,
            dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
        batch = native_batch(processor, observation, states, embodiment=embodiment)
        prepared = {k: input_clone(batch[k]) for k in INPUT_KEYS}
        backbone_inputs, _ = model.prepare_input(batch)
        output = model.action_head.process_backbone_output(model.backbone(backbone_inputs), action_inputs_B=1)
    return prepared, output


def observed_states(loader, processor, record, frame, embodiment):
    """Column-projected observed telemetry ONLY; never the action parquet column."""
    import pandas as pd
    from gr00t.data.dataset.lerobot_episode_loader import DEFAULT_COLUMN_NAMES
    keys = processor.modality_configs[embodiment]["state"].modality_keys
    columns = sorted({loader.modality_meta["state"][key].get("original_key", DEFAULT_COLUMN_NAMES["state"]) for key in keys})
    if any("action" in key.lower() for key in columns):
        raise ValueError("Native observed-state metadata unexpectedly points at an action column")
    table = pd.read_parquet(record["parquet_path"], columns=columns)
    grouped = loader._extract_joint_groups(table, keys, "state")
    return {key: np.stack([grouped[key].iloc[frame]]).astype(np.float32) for key in keys}


def source_hashes():
    from run_scripts.robomme.demo_tail_sidecar_v13 import source_identity
    inherited = source_identity()  # Includes Eagle tokenizer/config assets, not just Python.
    paths = []
    paths += [ROOT / "run_scripts/robomme" / name for name in (
        "verify_demo_tail_v13.py", "demo_tail_sidecar_v13.py", "visual_demo_tail_bank_v13.py",
        "visual_differential_memory_v12.py", "visual_patch_memory_v11.py")]
    return {**inherited, **{str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}}


def proof_frames(plan):
    """At most two canonical frames per proof episode and ep0's final six tail frames.

    Ep0's six frames are an explicitly disclosed ingestion example, not a
    different admission rule: the sidecar plan still includes every final15.
    """
    rows = []
    records = [r for r in plan["episodes"] if r["role"] == "train_proof"][:2]
    records += [r for r in plan["episodes"] if r["episode_id"] == 0]
    if len(records) != 3 or any(r["split"] != "train" for r in records):
        raise ValueError("Proof requires two TRAIN demos plus separately identified TRAIN ep0")
    for record in records:
        canonical = record["canonical_frames"]
        selected = sorted({canonical[0], record["last_canonical_demo"]})
        for frame in selected:
            rows.append({"episode_id": record["episode_id"], "frame": frame, "kind": "canonical"})
        if record["episode_id"] == 0:
            for frame in record["frames"][-6:]:
                rows.append({"episode_id": 0, "frame": frame, "kind": "tail_ingestion_example"})
    if len({(r["episode_id"], r["frame"]) for r in rows}) != len(rows):
        raise ValueError("Duplicated proof frames")
    if (len(rows) != 12 or sum(r["kind"] == "canonical" for r in rows) != 6
            or sum(r["kind"] == "tail_ingestion_example" for r in rows) != 6):
        raise ValueError("This predeclared proof requires exactly 12 rows: six canonical and six ep0 tail images")
    return rows


def completion_code(report):
    """0 = complete exact proof, 1 = failed/incomplete, 2 = numerical review."""
    if not report["structural_passed"]:
        return 1
    return 0 if report["all_feature_comparisons_exact"] else 2


def guarded_check(report, name, check):
    """Unreadable/deleted protected inputs are failed evidence, not lost reports."""
    try:
        report["checks"][name] = bool(check())
    except Exception as exc:
        report["checks"][name] = False
        report.setdefault("integrity_errors", []).append(
            {"check": name, "type": type(exc).__name__, "message": str(exc)})


def processor_comparison(native_call, direct_call):
    """Exact CPU input comparison, explicitly NOT an image-feature comparison."""
    caller = rng_state()
    try:
        native = native_call()
        restore_rng(caller)
        direct = direct_call()
        comparison = inputs_exact({k: input_clone(native[k]) for k in INPUT_KEYS}, direct)
        return {"prepared_inputs_exact": all(v["exact"] for v in comparison.values()),
                "comparisons": comparison, "native_collated_keys": sorted(native),
                "direct_model_keys": sorted(direct),
                "direct_signatures": {k: input_signature(v) for k, v in direct.items()}}
    finally:
        restore_rng(caller)


def processor_proof(plan, output, *, base, cache, selected, hashes, sources):
    """Decode the same 12 TRAIN frames with AutoProcessor only; never load weights."""
    import gr00t.model  # noqa: F401 -- register original processor
    from transformers import AutoProcessor
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.long_memory.monitoring import _atomic_json
    from run_scripts.robomme import demo_tail_sidecar_v13 as sidecar
    report = {"mode": "processor_only", "passed": False, "rows": [], "checks": {},
              "limitation": "CPU RGB/text preprocessing only: no model, image features, action parity, training or accuracy."}
    try:
        if torch.cuda.is_initialized():
            raise RuntimeError("Processor-only proof must not initialize or inherit CUDA")
        processor = AutoProcessor.from_pretrained(base)
        processor.eval()
        identity = cache.manifest["identity"]
        embodiment = identity.get("embodiment", "new_embodiment")
        loader = LeRobotEpisodeLoader(Path(plan["dataset_path"]), processor.modality_configs[embodiment],
                                     video_backend=identity.get("video_backend", "opencv"))
        records = {r["episode_id"]: r for r in plan["episodes"]}
        for item in selected:
            record, frame = records[item["episode_id"]], item["frame"]
            obs = sidecar.load_visual_observations(processor, loader, record, [frame], embodiment=embodiment)[0]
            states = observed_states(loader, processor, record, frame, embodiment)
            row = processor_comparison(
                lambda: native_batch(processor, obs, states, embodiment=embodiment),
                lambda: sidecar.prepare_visual_inputs(processor, obs["images"], obs["text"], embodiment=embodiment))
            row.update(item)
            report["rows"].append(row)
            _atomic_json(output / "result.json", report)
            if not row["prepared_inputs_exact"]:
                raise RuntimeError("CPU native/direct prepared inputs differ")
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        guarded_check(report, "source_unchanged", lambda: sources == source_hashes())
        guarded_check(report, "protected_files_unchanged", lambda: hashes == {p: sha(p) for p in hashes})
        guarded_check(report, "runtime_unchanged", lambda: plan["runtime"] == sidecar.runtime_identity())
        report["checks"]["cuda_not_initialized"] = not torch.cuda.is_initialized()
        report["passed"] = ("error" not in report and len(report["rows"]) == len(selected) == 12
                            and all(report["checks"].values())
                            and all(r["prepared_inputs_exact"] for r in report["rows"]))
        _atomic_json(output / "result.json", report)
    print(json.dumps({"mode": report["mode"], "passed": report["passed"], "rows": len(report["rows"])}), flush=True)
    return 0 if report["passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--proof-train-count", type=int, default=8,
                        help="Metadata sidecar subset (8..16); encode only first two plus fixed ep0")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true", help="Metadata/hashes only; no RGB decode or output")
    mode.add_argument("--preparation-only", action="store_true", help="12-frame CPU processor parity; no model weights or CUDA")
    args = parser.parse_args(argv)
    if args.preparation_only and args.device != "cpu":
        raise ValueError("--preparation-only requires --device cpu")
    if not 8 <= args.proof_train_count <= 16:
        raise ValueError("Sidecar metadata plan requires 8..16 deterministic TRAIN demos")
    from run_scripts.robomme import demo_tail_sidecar_v13 as sidecar
    from gr00t.long_memory.cache import EpisodeCache
    from gr00t.long_memory.hamlet import load_frozen_hamlet, validate_cache_checkpoint
    from gr00t.long_memory.safety_v5 import validate_output_scope
    from gr00t.long_memory.monitoring import _atomic_json
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    base = Path(args.base_model).resolve()
    if base != Path(cache.manifest["model_path"]).resolve():
        raise ValueError("Base model differs from the original cache model")
    plan = sidecar.build_plan(cache.path, base, proof_train_count=args.proof_train_count, proof_episode=0)
    selected = proof_frames(plan)
    output = validate_output_scope(args.output_dir, cache.path, base, plan["dataset_path"])
    if output.exists():
        raise FileExistsError("Use a NEW proof output directory; existing results are never overwritten")
    hashes, sources = dict(plan["files_sha256"]), source_hashes()
    plan.update(proof_frames=selected, args=vars(args), files_sha256=hashes, source_sha256=sources, limitations=LIMITATIONS,
                proof_frame_rule="first two train_proof episodes plus known TRAIN ep0: first/last canonical demo; ep0 final six tail ingestion example")
    if args.preflight_only:
        print(json.dumps({"preflight_passed": True, "frames": selected, "cuda_initialized": torch.cuda.is_initialized(), "output_created": False}))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "plan.json", plan)
    if args.preparation_only:
        return processor_proof(plan, output, base=base, cache=cache, selected=selected, hashes=hashes, sources=sources)
    report = {"structural_passed": False, "rows": [], "checks": {}, "limitations": LIMITATIONS}
    persist = lambda: _atomic_json(output / "result.json", report)
    model = None
    try:
        model, processor = load_frozen_hamlet(base, device=args.device)
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        identity = cache.manifest["identity"]
        embodiment = identity.get("embodiment", "new_embodiment")
        loader = LeRobotEpisodeLoader(Path(plan["dataset_path"]), processor.modality_configs[embodiment],
                                     video_backend=identity.get("video_backend", "opencv"))
        model.requires_grad_(False).eval()
        processor.eval()
        report["frozen_before_sha256"] = module_sha(model)
        report["checks"]["frozen_no_grad_before"] = all(not p.requires_grad and p.grad is None for p in model.parameters())
        records = {r["episode_id"]: r for r in plan["episodes"]}
        for item in selected:
            record, frame = records[item["episode_id"]], item["frame"]
            observation = sidecar.load_visual_observations(processor, loader, record, [frame], embodiment=embodiment)[0]
            states = observed_states(loader, processor, record, frame, embodiment)
            cached = None
            if item["kind"] == "canonical":
                episode = cache.load(item["episode_id"])
                indices = (episode["frames"] == frame).nonzero().flatten()
                if indices.numel() != 1:
                    raise ValueError("Canonical reference frame not uniquely present in original cache")
                q = int(indices[0])
                cached = {name: episode[key][q][None] for name, key in
                    (("backbone_features", "features"), ("image_mask", "image_masks"), ("backbone_attention_mask", "attention_masks"))}
            row = verify_frame(model,
                lambda: native_preparation(model, processor, observation, states, embodiment=embodiment),
                lambda: sidecar.prepare_visual_inputs(processor, observation["images"], observation["text"], embodiment=embodiment),
                lambda: sidecar.extract_image_features(model.backbone, model.action_head.vlln, processor,
                    observation["images"], observation["text"], device=args.device, embodiment=embodiment), cached=cached)
            row.update(item)
            report["rows"].append(row)
            persist()
            print(json.dumps({"episode_id": item["episode_id"], "frame": frame, "structural_passed": row["structural_passed"],
                              "all_feature_comparisons_exact": row["all_feature_comparisons_exact"]}), flush=True)
            if not row["structural_passed"]:
                raise RuntimeError("Structural extraction proof failed; no further frames run")
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if model is not None and "frozen_before_sha256" in report:
            def check_model():
                report["frozen_after_sha256"] = module_sha(model)
                return report["frozen_before_sha256"] == report["frozen_after_sha256"]
            guarded_check(report, "frozen_model_content_unchanged", check_model)
            guarded_check(report, "frozen_no_grad_after", lambda: all(not p.requires_grad and p.grad is None for p in model.parameters()))
        guarded_check(report, "source_unchanged", lambda: sources == source_hashes())
        guarded_check(report, "protected_files_unchanged", lambda: hashes == {p: sha(p) for p in hashes})
        guarded_check(report, "runtime_unchanged", lambda: plan["runtime"] == sidecar.runtime_identity())
        report["structural_passed"] = ("error" not in report and len(report["rows"]) == len(selected)
            and all(report["checks"].values()) and all(r["structural_passed"] for r in report["rows"]))
        report["all_feature_comparisons_exact"] = bool(report["rows"]) and all(r["all_feature_comparisons_exact"] for r in report["rows"])
        report["numerical_review_required"] = not report["all_feature_comparisons_exact"]
        report["passed"] = report["structural_passed"] and report["all_feature_comparisons_exact"]
        persist()
    print(json.dumps({k: report[k] for k in ("passed", "structural_passed", "all_feature_comparisons_exact", "numerical_review_required")}), flush=True)
    return completion_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
