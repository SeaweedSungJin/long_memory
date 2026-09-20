#!/usr/bin/env python3
"""Frozen V11 historical-content sensitivity, NOT robot accuracy or training.

Use the saved validation query/noise schedule, native Euler4, and unchanged
archive1250/AE. Replace ONLY receiver past image-patch content with the first q
observations of another TRAIN episode with the exact same cache instruction.
The cache task field is an instruction fallback, NOT an authoritative benchmark
task ID: cross-variant/domain-shift confounds are reported explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from gr00t.long_memory.action_audit_v8 import generated_action
from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.cache_reader_v3 import validate_decision
from gr00t.long_memory.checkpoint_v7 import load_checkpoint_v7, v7_checkpoint_info
from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.expert_v4 import LoRAConfig, expert_episode_flow_loss, install_expert_lora, set_expert_trainable
from gr00t.long_memory.hamlet import isolated_seed, validate_cache_checkpoint
from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from gr00t.long_memory.replay_v7 import replay_queries
from gr00t.long_memory.safety_v5 import validate_output_scope
from gr00t.long_memory.train_v3 import runtime_identity
from run_scripts.robomme.audit_archive_generation_v7 import prefix_masks
from run_scripts.robomme.checkpoint_visual_patch_v11 import checkpoint_info, load_checkpoint
from run_scripts.robomme.replay_visual_patch_v11 import OBSERVATION_KEYS, replay_visual_patch
from run_scripts.robomme.verify_projector_v10 import actual_head
from run_scripts.robomme.visual_patch_memory_v11 import (
    CAMERA_ORDER, PATCHES_PER_OBSERVATION, PATCHES_PER_VIEW, VisualPatchConfig, VisualPatchMemoryV11,
)

ROLES = ("correct_history", "visual_off", "instruction_matched_wrong_history")
LIMITATIONS = [
    "Offline normalized-action errors and sensitivity, NOT RoboMME success/accuracy or correct-recall proof.",
    "Cache task keys are recorded instructions, not authoritative benchmark task IDs; exact text may cross task variants.",
    "The donor counterfactual may introduce domain/scene and demo/execution phase mismatch; this is not a natural intervention.",
    "Receiver camera/grid/time/demo labels stay fixed while donor patch contents are transplanted; donor timing is NOT used by memory.",
    "The receiver current observation, original HAMLET short history and frozen archive READ remain intact in ALL roles.",
    "Only full prior CACHED observations are tested; missing video frames are not recovered.",
    "One deterministic TRAIN donor per query; no donor search against action errors or success labels.",
    "Validation was already used for model selection; these are diagnostic validation results, not independent test results.",
    "On/off sensitivity alone does not establish historical-content selectivity; wrong-history degradation can reflect distribution shift.",
]


def digest(value):
    # Exact canonicalization used by the immutable V11/V9 training plans.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    result = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)]).encode())
    result.update(value.reshape(-1).view(torch.uint8).numpy())
    return result.hexdigest()


def observation_columns(episode):
    # No enumeration of the enclosing mapping and no supervision/state access.
    return {key: episode[key] for key in OBSERVATION_KEYS}


def patch_indices(feature, image, attention, short_tokens):
    """Mirror the frozen core's ordered two-camera 81+81 patch layout."""
    if (not isinstance(feature, torch.Tensor) or feature.ndim != 2 or not feature.is_floating_point()
            or feature.shape[0] <= PATCHES_PER_OBSERVATION + short_tokens):
        raise ValueError("Invalid observation features")
    for mask in (image, attention):
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != feature.shape[:1]:
            raise ValueError("Invalid observation mask")
    if not bool(attention[-short_tokens:].all()):
        raise ValueError("Short tail must have valid attention")
    valid = (image & attention).clone()
    valid[-short_tokens:] = False
    indices = valid.nonzero().flatten()
    if indices.numel() != PATCHES_PER_OBSERVATION:
        raise ValueError("Expected exactly 162 valid image patches")
    cameras = indices.reshape(2, PATCHES_PER_VIEW)
    if not bool((cameras.diff() == 1).all()) or not bool(cameras[1, 0] > cameras[0, -1] + 1):
        raise ValueError("Expected separate contiguous front/wrist 81-patch runs")
    if not bool(torch.isfinite(feature[indices]).all()) or not bool(torch.isfinite(feature[-short_tokens:]).all()):
        raise ValueError("Nonfinite selected observation features")
    return indices


def transplant_history(receiver, donor, query, *, short_tokens=4):
    """Observation-only [0,q] view; q remains EXACT, no row >q is accessed.

    Donor row i supplies only ordered patch contents for receiver past row i.
    Receiver non-images, masks, short tail, frame/demo metadata are retained.
    q=0 needs no donor and accesses no donor keys. Inputs are never mutated.
    """
    if type(query) is not int or query < 0 or type(short_tokens) is not int or short_tokens <= 0:
        raise ValueError("Invalid query/short-token count")
    source = observation_columns(receiver)
    result = {key: [source[key][i] for i in range(query + 1)] for key in OBSERVATION_KEYS}
    mapping = []
    for i in range(query):
        if donor is None:
            raise ValueError("Nonempty history requires a donor")
        old, new = result["features"][i], donor["features"][i]
        dst = patch_indices(old, result["image_masks"][i], result["attention_masks"][i], short_tokens)
        src = patch_indices(new, donor["image_masks"][i], donor["attention_masks"][i], short_tokens)
        if old.shape[1] != new.shape[1] or old.dtype != new.dtype or old.device != new.device:
            raise ValueError("Donor feature width/dtype/device differs; no implicit conversion")
        changed = old.clone()
        changed[dst] = new[src]
        result["features"][i] = changed
        mapping.append({"receiver_row": i, "donor_row": i,
            "receiver_frame": int(result["frames"][i]), "receiver_is_demo": bool(result["is_demo"][i]),
            "donor_frame": int(donor["frames"][i]), "donor_is_demo": bool(donor["is_demo"][i]),
            "receiver_patch_indices": dst.tolist(), "donor_patch_indices": src.tolist(),
            "receiver_patch_sha256": tensor_hash(old[dst]), "donor_patch_sha256": tensor_hash(new[src]),
            "changed_fraction": float((old[dst] != changed[dst]).float().mean())})
    # Exact current object identity is deliberate, including all masks/metadata.
    return result, mapping


class ObservationCache:
    """mmap without donor GT/action validation; only explicit observation fields.

    torch.load decodes tensor headers but mmap does not page donor action/target
    storages into the construction. Whole-file hashes separately bind bytes for
    provenance; they are never interpreted as control-selection/model inputs.
    """
    def __init__(self, cache, training_files):
        self.cache, self.training_files = cache, training_files
        self.records = {int(r["episode_id"]): r for r in cache.manifest["episodes"]}
        self.inspected = {}

    def load(self, eid):
        record = self.records[eid]
        root = Path(self.cache.path).resolve()
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Cache episode path escapes cache")
        stat = path.stat()
        identity = [str(path), stat.st_size, stat.st_mtime_ns]
        if self.training_files.get(str(eid)) != identity:
            raise ValueError("Episode file identity changed since immutable training plan")
        ep = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if (ep["episode_id"] != eid or ep.get("task") != record["task"]
                or ep.get("cache_fingerprint") != self.cache.manifest["fingerprint"]):
            raise ValueError("Episode identity/task/cache fingerprint differs")
        self.inspected[eid] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        return ep


def select_donors(records, train_ids, validation, load):
    """Smallest eligible TRAIN ID, exact cache instruction, enough first rows."""
    donors, inspected = {}, {}
    for eid, query in validation:
        key = records[eid]["task"]
        if not isinstance(key, str) or not key.strip() or key == "unknown_task":
            raise ValueError("Receiver lacks a literal cache instruction key")
        chosen, candidates = None, []
        if query:
            for candidate in sorted(train_ids):
                if candidate == eid or records[candidate]["task"] != key:
                    continue
                if candidate not in inspected:
                    ep = load(candidate)
                    # Only observation cardinality; no donor decision/supervision.
                    counts = {name: len(ep[name]) for name in OBSERVATION_KEYS}
                    if len(set(counts.values())) != 1:
                        raise ValueError("Donor observation columns disagree in length")
                    inspected[candidate] = counts["features"]
                count = inspected[candidate]
                candidates.append({"episode_id": candidate, "observations": count})
                if count >= query:
                    chosen = candidate
                    break
            if chosen is None:
                raise ValueError(f"No sufficiently long instruction-matched TRAIN donor for {eid}/{query}; no cycling/fallback")
        donors[eid, query] = {"receiver_episode_id": eid, "decision": query,
            "cache_instruction_key": key, "canonical_benchmark_task": None,
            "donor_episode_id": chosen, "donor_split": "train" if chosen is not None else None,
            "row_mapping": "donor i -> receiver i, i in [0,q)", "candidate_checks": candidates,
            "control": ROLES[2] if query else "empty_history_identity"}
    return donors


def validation_plan(path, info, manifest):
    payload = json.loads(Path(path).read_text())
    plan = {k: v for k, v in payload.items() if k != "sha256"}
    expected = info["metadata"]["plan_sha256"]
    if payload.get("sha256") != digest(plan) or payload["sha256"] != expected:
        raise ValueError("Training query/noise plan digest differs from V11 checkpoint")
    validation, schedule = plan["validation"], plan["validation_schedule"]
    config = info["config"]["train"]
    if config["val_samples"] != 32 or config["val_noise_samples"] != 2:
        raise ValueError("This diagnostic requires the declared 32-distinct-query x 2-noise V11 validation plan")
    if len(validation) != 32 or len({pair[0] for pair in validation}) != 32:
        raise ValueError("Validation must contain 32 distinct receiver episodes")
    train, val = set(manifest["splits"]["train"]), set(manifest["splits"]["val"])
    if train & val or any(type(eid) is not int or eid not in val or type(q) is not int or q < 0 for eid, q in validation):
        raise ValueError("Validation receivers must be strictly held-out cache episodes")
    identities = [(eid, q, repeat) for eid, q in validation for repeat in range(2)]
    if [(r["episode_id"], r["decision"], r["repeat"]) for r in schedule] != identities:
        raise ValueError("Validation schedule order/repeats differs")
    for row in schedule:
        if set(row) != {"episode_id", "decision", "repeat", "flow_seed", "generation_seed"}:
            raise ValueError("Unexpected validation schedule fields")
        if any(type(row[k]) is not int or row[k] < 0 for k in row):
            raise ValueError("Validation seeds/indices must be nonnegative integers")
    return plan


def source_hashes():
    paths = list((ROOT / "gr00t").rglob("*.py")) + [ROOT / "run_scripts/robomme" / name for name in (
        "audit_visual_history_v11.py", "checkpoint_visual_patch_v11.py", "visual_patch_memory_v11.py",
        "replay_visual_patch_v11.py", "verify_projector_v10.py", "projector_adapter_v10.py",
        "deployment_objective_v9.py", "audit_archive_generation_v7.py")]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}


def preflight(args):
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    base = Path(cache.manifest["model_path"]).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    info = checkpoint_info(base, checkpoint)
    if info["step"] <= 0:
        raise ValueError("Historical-content audit requires an honestly trained V11 step > 0")
    if info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"] or cache.manifest["action_steps"] != 16:
        raise ValueError("Checkpoint/cache/prefix protocol differs")
    if info["metadata"]["cache_manifest_sha256"] != digest(cache.manifest):
        raise ValueError("Immutable training cache manifest differs")
    training_run = Path(args.training_run).resolve()
    plan_path = training_run / "query_plan.json"
    plan = validation_plan(plan_path, info, cache.manifest)
    output = Path(args.output_dir).resolve()
    validate_output_scope(output, cache.path, cache.manifest.get("dataset_path"), base, checkpoint,
                          info["metadata"]["frozen_parent"]["path"], training_run)
    if output.exists():
        raise FileExistsError("Use a NEW diagnostic output directory")
    observations = ObservationCache(cache, plan["files"])
    donors = select_donors(observations.records, cache.manifest["splits"]["train"], plan["validation"], observations.load)
    mappings = []
    for eid, query in plan["validation"]:
        ep = observations.load(eid)
        donor = donors[eid, query]
        dep = observations.load(donor["donor_episode_id"]) if query else None
        _, rows = transplant_history(observation_columns(ep), observation_columns(dep) if dep is not None else None,
                                     query, short_tokens=info["config"]["visual"]["num_short_tokens"])
        mappings.append({**donor, "rows": rows})
    files = {str(cache.path / "manifest.json"): file_hash(cache.path / "manifest.json"),
             str(plan_path): file_hash(plan_path)}
    validation_reference = training_run / f"validation-{info['step']:06d}.json"
    if validation_reference.is_file():
        saved = json.loads(validation_reference.read_text())
        if saved.get("step") != info["step"] or saved.get("plan_sha256") != info["metadata"]["plan_sha256"]:
            raise ValueError("Saved training validation identity differs")
        files[str(validation_reference)] = file_hash(validation_reference)
    for name in ("checkpoint.json", "visual.safetensors", "training_state.pt"):
        files[str(checkpoint / name)] = file_hash(checkpoint / name)
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".safetensors", ".model", ".txt"):
            files[str(path)] = file_hash(path)
    parent = info["metadata"]["frozen_parent"]
    files.update({str(Path(parent["path"]) / name): h for name, h in parent["files_sha256"].items()})
    for record in observations.inspected.values():
        files[record["path"]] = file_hash(record["path"])
    record = {"format_version": 1, "diagnostic": "visual_history_v11", "checkpoint": str(checkpoint),
        "checkpoint_step": info["step"], "training_run": str(training_run), "plan_sha256": plan["sha256"] if "sha256" in plan else digest(plan),
        "base_model": info["metadata"]["base_model"],
        "cache_fingerprint": cache.manifest["fingerprint"], "visual_config": info["config"]["visual"],
        "camera_order": list(CAMERA_ORDER), "roles": list(ROLES), "validation_schedule": plan["validation_schedule"],
        "donor_policy": "smallest distinct TRAIN ID matching literal cache instruction, first q rows, no cycling",
        "canonical_task_mapping_available": False, "donors": mappings,
        "source_sha256": source_hashes(), "files_sha256": files, "runtime": runtime_identity(),
        "training_validation_reference": str(validation_reference) if validation_reference.is_file() else None,
        "device": args.device, "sampler": "native_original_Euler4", "action_steps": 16,
        "optimizer_updates": 0, "limits": LIMITATIONS}
    record["audit_id"] = digest(record)
    return cache, base, info, plan, observations, donors, record


def conditioned_views(visual, receiver, query, fused_short, donor=None):
    """Build all features before any action targets are consulted."""
    original = observation_columns(receiver)
    wrong, mapping = transplant_history(original, donor, query, short_tokens=visual.config.num_short_tokens)
    features, banks = {}, {}
    for role, source, enabled in ((ROLES[0], original, True), (ROLES[1], original, False), (ROLES[2], wrong, True)):
        value, bank = replay_visual_patch(visual, source, query, camera_order=CAMERA_ORDER,
                                         visual_read_enabled=enabled, checkpoint_encoding=False)
        n = visual.config.num_short_tokens
        features[role] = torch.cat((value[:, :-n], fused_short.to(value)), dim=1)
        banks[role] = bank
    reference = banks[ROLES[0]]
    for bank in banks.values():
        if (bank.tokens.shape != reference.tokens.shape or bank.tokens.shape[1] != query
                or not torch.equal(bank.frames, reference.frames) or not torch.equal(bank.is_demo, reference.is_demo)
                or not torch.equal(bank.valid, reference.valid)):
            raise RuntimeError("Counterfactual history capacity/temporal metadata changed")
    if not torch.equal(reference.tokens, banks[ROLES[1]].tokens):
        raise RuntimeError("Visual-off changed original stored history")
    if not query and any(not torch.equal(value, features[ROLES[1]]) for value in features.values()):
        raise RuntimeError("Empty-history controls must be exact frozen-parent identity")
    return features, mapping


def action_metrics(prediction, episode, query):
    target = episode["targets"][query].to(device=prediction.device, dtype=torch.float32)[None]
    _, mask = prefix_masks(episode["target_mask"][query].to(prediction.device)[None], episode["action_mask"][query], 16)
    if prediction.shape != target.shape or not bool(mask.any()):
        raise ValueError("Invalid observed-prefix action comparison")
    result = {}
    for group, lo, hi in (("prefix", 0, prediction.shape[-1]), ("joint7", 0, 7), ("gripper1", 7, 8)):
        selected = mask.clone()
        selected[:, :, :lo] = False
        selected[:, :, hi:] = False
        difference = (prediction.float() - target)[selected]
        if not difference.numel() or not bool(torch.isfinite(difference).all()):
            raise FloatingPointError("Missing/nonfinite observed generated-action values")
        result.update({f"{group}_mse": float(difference.square().mean()), f"{group}_mae": float(difference.abs().mean()),
                       f"{group}_squared_error_sum": float(difference.double().square().sum()), f"{group}_count": difference.numel()})
    return result


def summarize(rows, bootstrap_samples=5000):
    """Equal-query means; paired bootstrap clusters the two noises by query."""
    summary, by_role = {}, {role: {} for role in ROLES}
    for row in rows:
        key = row["episode_id"], row["decision"]
        by_role[row["role"]].setdefault(key, []).append(row)
    for role in ROLES:
        records = [r for r in rows if r["role"] == role]
        summary[role] = {"draws": len(records), "queries": len(by_role[role])}
        if records and all("flow_loss" in r for r in records):
            flow = [sum(r["flow_loss"] for r in items) / len(items) for items in by_role[role].values()]
            summary[role]["flow_loss_equal_query_mean"] = sum(flow) / len(flow)
        for group in ("prefix", "joint7", "gripper1"):
            key = group + "_mse"
            means = [sum(r[key] for r in items) / len(items) for items in by_role[role].values()]
            summary[role][key + "_equal_query_mean"] = sum(means) / len(means) if means else None
            count = sum(r[group + "_count"] for r in records)
            summary[role][key + "_pooled_coordinates"] = sum(r[group + "_squared_error_sum"] for r in records) / count if count else None
    contrasts = {}
    for other in ROLES[1:]:
        left, right = by_role[ROLES[0]], by_role[other]
        if left.keys() != right.keys():
            raise ValueError("Incomplete control/query pairing")
        result = {}
        for group in ("prefix", "joint7", "gripper1"):
            key = group + "_mse"
            differences = []
            for query in left:
                a, b = left[query], right[query]
                if [(r["repeat"], r["generation_seed"], r["flow_seed"]) for r in a] != [(r["repeat"], r["generation_seed"], r["flow_seed"]) for r in b]:
                    raise ValueError("Control noise pairing differs")
                differences.append(sum(y[key] - x[key] for x, y in zip(a, b)) / len(a))
            value = sum(differences) / len(differences) if differences else None
            rng = random.Random(17)
            samples = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                             for _ in range(bootstrap_samples)) if len(differences) > 1 else []
            result[key] = {"other_minus_correct": value, "query_n": len(differences),
                "positive_queries": sum(v > 0 for v in differences), "negative_queries": sum(v < 0 for v in differences),
                "equal_queries": sum(v == 0 for v in differences),
                "paired_query_bootstrap_ci95": [samples[int(.025 * len(samples))], samples[min(len(samples)-1, int(.975 * len(samples)))]] if samples else None}
        contrasts[other + "_minus_correct_history"] = result
    return {"roles": summary, "paired_contrasts": contrasts,
            "uncertainty": "query-cluster bootstrap averages both fixed noise draws first; conditional diagnostic validation only"}


def frozen_guard(modules):
    if any(m.training or any(p.requires_grad or p.grad is not None for p in m.parameters()) for m in modules):
        raise RuntimeError("All audit modules must be frozen/eval with no gradients")
    return [(value, value._version) for m in modules for value in list(m.parameters()) + list(m.buffers())]


def training_parity(rows, reference_path, *, step, plan_sha256):
    """Record actual deltas, not an assumed equivalence or fresh training score."""
    if reference_path is None:
        return {"available": False, "note": "This checkpoint has no saved training-validation boundary"}
    saved = json.loads(Path(reference_path).read_text())
    if saved["step"] != step or saved["plan_sha256"] != plan_sha256:
        raise ValueError("Training validation identity changed")
    fields = {"flow_loss": "flow_action_loss", "prefix_mse": "generated_observed_prefix_mse",
              "prefix_mae": "generated_observed_prefix_mae", "joint7_mse": "generated_observed_joint7_mse",
              "gripper1_mse": "generated_observed_gripper1_mse"}
    role_map = {ROLES[0]: "reader", ROLES[1]: "visual-off"}
    def index(records):
        result = {}
        for row in records:
            key = row["role"], row["episode_id"], row["decision"], row["repeat"]
            if key in result:
                raise ValueError("Duplicate validation parity row")
            result[key] = row
        return result
    previous = index(saved["records"])
    current = index([r for r in rows if r["role"] in role_map])
    expected = {(role_map[role], eid, q, repeat) for role, eid, q, repeat in current}
    if set(previous) != expected:
        raise ValueError("Training/current validation query-role sets differ")
    differences = []
    for (role, eid, q, repeat), row in current.items():
        old = previous[role_map[role], eid, q, repeat]
        if any(row[key] != old[key] for key in ("generation_seed", "flow_seed")):
            raise ValueError("Training/current validation noise differs")
        delta = {key: row[key] - old[old_key] for key, old_key in fields.items()}
        differences.append({"role": role, "episode_id": eid, "decision": q, "repeat": repeat,
                            "audit_minus_saved": delta})
    summary = {role: {key: {"mean_signed_delta": sum(r["audit_minus_saved"][key] for r in differences if r["role"] == role)
                           / sum(r["role"] == role for r in differences),
                           "max_abs_delta": max(abs(r["audit_minus_saved"][key]) for r in differences if r["role"] == role)}
                     for key in fields} for role in role_map}
    return {"available": True, "reference": str(reference_path), "summary": summary, "records": differences,
            "all_metrics_bitexact": all(v == 0 for r in differences for v in r["audit_minus_saved"].values()),
            "note": "Deltas are measured; nonzero values require interpretation before attributing control differences."}


@torch.no_grad()
def run(args, base, info, plan, observations, donors, result, persist):
    cfg = v7_checkpoint_info(base, info["metadata"]["frozen_parent"]["path"], expected_stage=1)["config"]
    head = actual_head(base, args.device)
    with isolated_seed(0, args.device):
        parent = RecurrentMemoryV7(MemoryV7Config(**cfg["memory"])).to(args.device)
        cvom = CVOMV7(parent.config).to(args.device)
        install_expert_lora(head, LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
        visual = VisualPatchMemoryV11(VisualPatchConfig(**info["config"]["visual"])).to(args.device)
    load_checkpoint_v7(info["metadata"]["frozen_parent"]["path"], parent, head, cvom)
    loaded = load_checkpoint(args.checkpoint, visual)
    if any(loaded[key] != info[key] for key in ("step", "config", "metadata")):
        raise ValueError("Visual checkpoint changed between preflight and actual loading")
    parent.eval().requires_grad_(False)
    cvom.eval().requires_grad_(False)
    visual.eval().requires_grad_(False)
    set_expert_trainable(head, False)
    if head.num_inference_timesteps != 4:
        raise ValueError("Native original Euler4 is required")
    modules = (head, parent, cvom, visual)
    guard = frozen_guard(modules)
    previous, features, episode = None, None, None
    for item in plan["validation_schedule"]:
        eid, query = item["episode_id"], item["decision"]
        if previous != (eid, query):
            episode = observations.load(eid)
            donor_id = donors[eid, query]["donor_episode_id"]
            donor = observation_columns(observations.load(donor_id)) if donor_id is not None else None
            fused = replay_queries(parent, episode, [query], mode="archive", checkpoint_segment=0)[query][0]
            features, _ = conditioned_views(visual, episode, query, fused, donor)
            # Supervision validation only AFTER construction of all control features.
            validate_decision(episode, query)
            previous = eid, query
        predictions = {}
        for role in ROLES:
            # Native generation receives only observation conditioning, never
            # action targets/masks (including via a broad episode dictionary).
            view = {key: episode[key] for key in ("state", "embodiment_id", "attention_masks", "image_masks")}
            view["features"] = {query: features[role][0]}
            prediction = generated_action(head, view, query, seed=item["generation_seed"])
            predictions[role] = prediction
            supervised = {**view, "targets": episode["targets"], "target_mask": episode["target_mask"]}
            flow = expert_episode_flow_loss(head, supervised, query, seed=item["flow_seed"])
            row = {**item, "role": role, "donor_episode_id": donors[eid, query]["donor_episode_id"],
                   "cache_instruction_key": donors[eid, query]["cache_instruction_key"],
                   "history_observations": query, "history_patch_tokens": query * PATCHES_PER_OBSERVATION,
                   "flow_loss": float(flow["loss"]), **action_metrics(prediction, episode, query),
                   "generated_sha256": tensor_hash(prediction), "conditioning_sha256": tensor_hash(features[role])}
            if any(not math.isfinite(v) for v in row.values() if isinstance(v, float)):
                raise FloatingPointError("Nonfinite diagnostic result")
            result["records"].append(row)
        original = predictions[ROLES[0]]
        _, observed = prefix_masks(episode["target_mask"][query].to(original.device)[None], episode["action_mask"][query], 16)
        for role in ROLES[1:]:
            row = result["records"][-3 + ROLES.index(role)]
            row["generated_exact_correct"] = torch.equal(predictions[role], original)
            difference = predictions[role].float() - original.float()
            row["generated_vs_correct_all_coordinates_mse"] = float(difference.square().mean())
            row["all_coordinate_sensitivity_shape"] = list(original.shape)
            row["all_coordinate_sensitivity_includes_padding"] = True
            row["generated_vs_correct_observed_prefix_mse"] = float(difference[observed].square().mean())
            row["conditioning_changed_fraction_vs_correct"] = float((features[role] != features[ROLES[0]]).float().mean())
            if not query and not row["generated_exact_correct"]:
                raise RuntimeError("q=0 generated controls differ despite identical conditioning/noise")
        frozen_guard(modules)
        if any(value._version != version for value, version in guard):
            raise RuntimeError("A frozen audit module parameter/buffer changed")
        persist()
        print(f"[visual-history] {len(result['records'])//3}/{len(plan['validation_schedule'])} paired draws; episode={eid}, q={query}", flush=True)
    result["summary"] = summarize(result["records"])
    result["frozen_modules_unchanged"] = True
    result["no_parameter_gradients"] = all(p.grad is None for m in modules for p in m.parameters())


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--training-run", required=True, help="Original immutable query_plan.json owner; no query resampling")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--preflight-only", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    cache, base, info, plan, observations, donors, record = preflight(args)
    if args.preflight_only:
        print(json.dumps({"audit_id": record["audit_id"], "checkpoint_step": info["step"],
            "queries": len(plan["validation"]), "draws": len(plan["validation_schedule"]),
            "donors": [{k: v for k, v in d.items() if k != "rows"} for d in record["donors"]],
            "note": "Read-only: no model/output/simulator/training; instruction-matched, not guaranteed same benchmark task"}, indent=2))
        return 0
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "plan.json", record)
    result = {"audit_id": record["audit_id"], "status": "running", "records": [],
              "optimizer_updates": 0, "limits": LIMITATIONS}
    def persist():
        _atomic_json(output / "result.json", result)
    persist()
    try:
        run(args, base, info, plan, observations, donors, result, persist)
        if len(result["records"]) != 3 * len(plan["validation_schedule"]):
            raise RuntimeError("Missing paired diagnostic rows")
        result["training_validation_parity"] = training_parity(result["records"], record["training_validation_reference"],
            step=info["step"], plan_sha256=info["metadata"]["plan_sha256"])
        result["source_unchanged"] = source_hashes() == record["source_sha256"]
        result["files_unchanged"] = all(file_hash(path) == h for path, h in record["files_sha256"].items())
        if not result["source_unchanged"] or not result["files_unchanged"]:
            raise RuntimeError("Source/checkpoint/cache/validation-plan changed during audit")
        result["status"] = "complete"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        persist()
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
