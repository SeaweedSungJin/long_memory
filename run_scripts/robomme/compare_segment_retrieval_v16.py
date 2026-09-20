#!/usr/bin/env python3
"""Paired Q/K versus Q/K+P diagnostic, never a robot-success assessment.

The predeclared primary is the episode-macro loss gain in the content-permuted
condition, at the SAME fixed final step. A favorable arm contrast does not
replace the original V15 GO gates and does not authorize a policy claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import load_file

from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.prepare_segment_targets_v15 import load_manifest
from run_scripts.robomme.train_segment_retrieval_probe_v15 import (
    assess, digest, file_hash, make_plan, paired_gain, verify_files,
)

KIND = "segment_retrieval_probe_v16"
QK = ("query_projection.weight", "key_projection.weight")
PROJECTION = "image_projection.weight"
MODEL_METRICS = {"loss", "positive_mass", "span_hit"}


def paired_rows(control, candidate):
    """Bind each query AND its full target/permutation/control metadata."""
    def index(rows):
        result = {}
        for row in rows:
            key = (row["episode_id"], row["decision"], row["mode"])
            if key in result or row["mode"] not in ("correct", "content_permuted"):
                raise ValueError("Duplicate or unknown validation identity")
            if not all(math.isfinite(float(row[m])) for m in MODEL_METRICS):
                raise ValueError("Nonfinite validation metric")
            result[key] = row
        if not result:
            raise ValueError("Empty validation")
        return result
    left, right = index(control), index(candidate)
    if left.keys() != right.keys():
        raise ValueError("Paired query identities differ")
    result = []
    for key in sorted(left):
        a, b = left[key], right[key]
        # This includes weak positives, strict-past frames, content mapping,
        # source identities, and ALL metrics of the deterministic controls.
        if ({k: v for k, v in a.items() if k not in MODEL_METRICS}
                != {k: v for k, v in b.items() if k not in MODEL_METRICS}):
            raise ValueError("Paired observations/targets/content maps/controls differ")
        result.append({**b, "qk_control/loss": a["loss"]})
    return result


def compare_rows(control_initial, control_final, candidate_initial, candidate_final):
    paired_rows(control_initial, control_final)
    paired_rows(candidate_initial, candidate_final)
    paired_rows(control_initial, candidate_initial)
    if control_initial != candidate_initial:
        raise ValueError("Arms must have identical initial validation distributions")
    rows = paired_rows(control_final, candidate_final)
    contrasts = {mode: paired_gain([r for r in rows if r["mode"] == mode], "qk_control")
                 for mode in ("correct", "content_permuted")}
    candidate_gate = assess(candidate_initial, candidate_final)
    primary = contrasts["content_permuted"]
    eligible = candidate_gate["decision"] != "inconclusive_coverage"
    favorable = eligible and primary["ci95"][0] > 0
    permitted = favorable and candidate_gate["decision"] == "go_to_action_probe"
    return {
        "kind": "segment_projection_paired_diagnostic_v16",
        "primary": "content_permuted_episode_macro_qk_minus_qkp_nll",
        "contrasts": contrasts,
        "projection_arm_superiority_supported": favorable,
        "original_gate_control": assess(control_initial, control_final),
        "original_gate_candidate": candidate_gate,
        "decision": "go_to_action_probe" if permitted else "no_go",
        "policy_ready": False,
        "goal_30_percent_achieved": False,
        "limitations": [
            "Selected maximal label runs are weak targets, not verified causal cues.",
            "This cache-VAL was already used for V15 diagnosis; it is not a new independent set.",
            "P affects current image/short queries and past addresses, with 524288 extra trainable weights.",
            "Content permutation is a nonphysical retrieval diagnostic, not task evaluation.",
            "No learned admission, action utility, or RoboMME success is established.",
        ],
    }


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_fixed_study(protocol):
    """This report is final256, not a smoke/best-step selection report."""
    expected = {"max_steps": 256, "batch_size": 4, "learning_rate": 1e-4,
                "eval_steps": 64, "seed": 9151}
    if any(protocol["train"].get(k) != value for k, value in expected.items()):
        raise ValueError("Comparison requires the predeclared final256 study budget/options")
    if (protocol.get("fixed_final_selection") is not True
            or protocol.get("permutation_seed") != 9152
            or protocol.get("bootstrap_seed") != 9151
            or protocol.get("bootstrap_samples") != 10000
            or protocol.get("time_only_scale") != 16
            or protocol.get("minimum_val_episodes") != 10
            or protocol.get("minimum_val_queries") != 20):
        raise ValueError("Predeclared selection/control/statistical protocol differs")


def validate_complete_val(rows, examples):
    expected = {(r["episode_id"], r["decision"], mode)
                for r in examples if r["split"] == "val"
                for mode in ("correct", "content_permuted")}
    actual = [(r["episode_id"], r["decision"], r["mode"]) for r in rows]
    if not expected or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Evaluation does not cover EVERY bound cache-VAL target in both modes")


def validate_tensor_layout(reference, other, *, exact=False):
    if (reference.keys() != other.keys()
            or any(v.shape != other[k].shape or v.dtype != other[k].dtype
                   for k, v in reference.items())):
        raise ValueError("Tensor keys, shapes or dtypes differ from the bound initialization")
    if exact and any(not torch.equal(v, other[k]) for k, v in reference.items()):
        raise ValueError("Initial tensors differ from the actual bound V14 weights")


def protected_roots(targets, target_path, init, init_header):
    identity = targets["identity"]
    return [identity[k] for k in ("dataset_path", "cache_dir", "sidecar_dir", "base_model")] + [
        str(Path(target_path).parent), str(init), init_header["metadata"]["initial_parent"]["path"]]


def load_run(path, arm):
    """Audit immutable files and actual optimizer scope before reporting."""
    path = Path(path).resolve(strict=True)
    protocol, status = read_json(path / "protocol.json"), read_json(path / "status.json")
    validate_fixed_study(protocol)
    if (protocol["kind"] != KIND or protocol["arm"] != arm
            or protocol["train"]["arm"] != arm or status["status"] != "complete"
            or status.get("policy_ready") is not False):
        raise ValueError("Expected a completed non-policy V16 arm")
    step, batch = protocol["train"]["max_steps"], protocol["train"]["batch_size"]
    if status["step"] != step or status["processed_queries"] != step * batch:
        raise ValueError("Run did not complete its exact predeclared budget")
    plan = read_json(path / "query_plan.json")
    if digest(plan) != protocol["plan_sha256"]:
        raise ValueError("Fixed query plan changed")
    verify_files(protocol["files_sha256"])
    target_path = Path(protocol["train"]["targets"]).resolve(strict=True)
    if protocol["files_sha256"].get(str(target_path)) != file_hash(target_path):
        raise ValueError("The target manifest is not bound in the protected protocol")
    targets = load_manifest(target_path, verify_files=False)
    for group in ("files_sha256", "source_sha256"):
        if any(protocol["files_sha256"].get(p) != signature for p, signature in targets[group].items()):
            raise ValueError("A target input/source is missing from the verified protocol")
    if plan != make_plan(targets["examples"], step, batch, protocol["train"]["seed"]):
        raise ValueError("Query plan is not the fixed TRAIN-only episode-balanced schedule")
    init = Path(protocol["train"]["init_checkpoint"]).resolve(strict=True)
    init_header = read_json(init / "checkpoint.json")
    if (protocol["files_sha256"].get(str(init / "visual.safetensors")) != file_hash(init / "visual.safetensors")
            or protocol.get("initial_visual_sha256") != file_hash(init / "visual.safetensors")
            or protocol["files_sha256"].get(str(init / "checkpoint.json")) != file_hash(init / "checkpoint.json")
            or init_header["step"] != 512):
        raise ValueError("Actual V14 initialization is not bound to the protocol")
    original = load_file(str(init / "visual.safetensors"))
    ordered = list(QK) + ([PROJECTION] if arm == "qkp" else [])
    selected = set(ordered)
    if protocol["selected_parameter_names"] != ordered:
        raise ValueError("Declared trainable scope differs")
    weights, buffers = {}, {}
    for boundary in (0, step):
        folder = path / f"checkpoint-{boundary:06d}"
        header = read_json(folder / "checkpoint.json")
        if (header["kind"] != KIND or header["arm"] != arm or header["step"] != boundary
                or header.get("deployable_policy") is not False
                or header["protocol_sha256"] != digest(protocol)
                or header["selected_parameter_names"] != ordered
                or header["payload_sha256"] != file_hash(folder / "probe.safetensors")
                or header["optimizer_sha256"] != file_hash(folder / "optimizer.pt")):
            raise ValueError("Checkpoint scope/provenance/payload mismatch")
        buffers[boundary] = header["frozen_buffers"]
        weights[boundary] = load_file(str(folder / "probe.safetensors"))
        if not all(bool(torch.isfinite(t).all()) for t in weights[boundary].values()):
            raise ValueError("Nonfinite checkpoint tensor")
        state = torch.load(folder / "optimizer.pt", map_location="cpu", weights_only=True)
        optimizer = state["optimizer"]
        entries = optimizer["state"]
        if (state["step"] != boundary or state["arm"] != arm or state["selected_parameter_names"] != ordered
                or len(entries) != (len(selected) if boundary else 0)
                or sum(len(g["params"]) for g in optimizer["param_groups"]) != len(selected)):
            raise ValueError("Optimizer scope or completed-step count differs")
        for entry in entries.values():
            if float(entry["step"]) != boundary or not all(
                    bool(torch.isfinite(v).all()) for v in entry.values() if isinstance(v, torch.Tensor)):
                raise ValueError("Invalid optimizer state")
    if weights[0].keys() != weights[step].keys():
        raise ValueError("Checkpoint layout changed")
    validate_tensor_layout(original, weights[0], exact=True)
    validate_tensor_layout(original, weights[step])
    if not buffers[0] or buffers[0] != buffers[step]:
        raise ValueError("Recorded frozen buffers changed")
    changed = {key for key in weights[0] if not torch.equal(weights[0][key], weights[step][key])}
    if changed != selected:
        raise ValueError("Actual updated tensors differ from selected scope")
    initial = read_json(path / "validation-000000.json")
    final = read_json(path / f"validation-{step:06d}.json")
    validate_complete_val(initial, targets["examples"])
    validate_complete_val(final, targets["examples"])
    if assess(initial, final) != read_json(path / "assessment.json"):
        raise ValueError("Stored original assessment is not reproducible")
    return {"path": path, "protocol": protocol, "status": status, "plan": plan,
            "weights0": weights[0], "initial": initial, "final": final,
            "buffers0": buffers[0],
            "protected_roots": protected_roots(targets, target_path, init, init_header)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    control, candidate = load_run(args.control, "qk"), load_run(args.candidate, "qkp")
    for key in ("files_sha256", "plan_sha256", "fixed_final_selection", "permutation_seed"):
        if control["protocol"][key] != candidate["protocol"][key]:
            raise ValueError(f"Matched protocol differs: {key}")
    train = lambda p: {k: v for k, v in p["train"].items() if k != "arm"}
    if train(control["protocol"]) != train(candidate["protocol"]) or control["plan"] != candidate["plan"]:
        raise ValueError("Arms differ in more than the trainable projection")
    validate_tensor_layout(control["weights0"], candidate["weights0"], exact=True)
    if control["buffers0"] != candidate["buffers0"]:
        raise ValueError("Initial frozen buffer identities differ")
    report = compare_rows(control["initial"], control["final"], candidate["initial"], candidate["final"])
    report.update({"control": str(control["path"]), "candidate": str(candidate["path"]),
                   "step": control["status"]["step"], "source_and_inputs_unchanged": True,
                   "original_weights_and_schedule_equal": True})
    output = validate_output_scope(args.output_dir, control["path"], candidate["path"],
                                   *control["protected_roots"], *candidate["protected_roots"],
                                   *(Path(p).parent for p in control["protocol"]["files_sha256"]))
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new comparison directory")
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "comparison.json", report)
    primary = report["contrasts"]["content_permuted"]
    summary = ("V16 weak-segment retrieval: Q/K versus Q/K+P\n"
               f"Fixed step: {report['step']}\n"
               f"Primary episode-macro NLL gain: {primary['mean_loss_gain']:+.6f}\n"
               f"95% episode bootstrap CI: {primary['ci95']}\n"
               f"Candidate original gates: {report['original_gate_candidate']['decision']}\n"
               f"Decision: {report['decision']}\n"
               "NOT a RoboMME success rate or deployable policy.\n")
    (output / "comparison.txt").write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
