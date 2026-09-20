#!/usr/bin/env python3
"""Recorded-RGB V13 session/ingestion parity; no simulator, learning or GT action.

Construct a real original archive1250 policy, release it, then construct genuine
V13 tail-on bundles with visual READ off. Compare identical canonical primes and
one first execution observation for TRAIN ep0 and the first no-demo TRAIN record.
This is an in-process policy/API proof, NOT transport or simulator/task accuracy.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import fields, is_dataclass
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.hamlet import validate_cache_checkpoint
from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
from gr00t.long_memory.safety_v5 import validate_output_scope
from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
from run_scripts.robomme import checkpoint_demo_tail_v13 as checkpoint
from run_scripts.robomme import verify_demo_tail_v13 as proof
from run_scripts.robomme.demo_tail_sidecar_v13 import (
    CAMERA_ORDER, DemoTailSidecar, load_visual_observations, runtime_identity,
)
from run_scripts.robomme.policy_demo_tail_v13 import DemoTailV13Policy
from run_scripts.robomme.policy_visual_patch_v11 import _scoped_methods

LIMITATIONS = [
    "Two predeclared TRAIN recorded-observation cases, not validation, simulator success or learned quality.",
    "Real policy constructors/native processor/backbone/AE and in-process ingest endpoint, not a socket/client/server or simulator proof.",
    "Recorded observed robot telemetry accompanies canonical observations; no GT action column is loaded and no executed feedback is invented.",
    "Only first execution is requested; no claim about later trajectories, interleaved sessions or live camera timing.",
    "Visual READ is off but the real RGB tail must still be ingested; this checks ingestion transparency, not usefulness of tail READ.",
    "Step-zero bundles are initialization diagnostics, never trained checkpoints.",
    "The raw RGB extractor consumes global RNG; the policy RPC, not the raw helper, must restore it.",
    "Every numerical comparison requires exact equality; nonexact evidence is preserved for review, never accepted by epsilon.",
]


def signature(value, *, references=False):
    """Independent JSON-safe byte/type fingerprint of small runtime state."""
    ref = {"object_id": id(value)} if references else {}
    if torch.is_tensor(value):
        return {**ref, "type": "tensor", "dtype": str(value.dtype), "shape": list(value.shape),
                "device": str(value.device), "sha256": proof.tensor_sha(value)}
    if isinstance(value, np.ndarray):
        return {**ref, "type": "ndarray", "dtype": str(value.dtype), "shape": list(value.shape),
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
    if isinstance(value, torch.Generator):
        return {**ref, "type": "generator", "device": str(value.device), "state": signature(value.get_state())}
    if is_dataclass(value) and not isinstance(value, type):
        return {**ref, "type": type(value).__name__, "fields": {f.name: signature(getattr(value, f.name), references=references)
                                                               for f in fields(value)}}
    if isinstance(value, dict):
        return {**ref, "type": "dict", "items": [[str(k), signature(v, references=references)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {**ref, "type": type(value).__name__, "items": [signature(v, references=references) for v in value]}
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        scalar = value.item() if isinstance(value, np.generic) else value
        if isinstance(scalar, float) and not np.isfinite(scalar):
            raise ValueError("Nonfinite runtime scalar")
        return {"type": type(value).__name__, "value": scalar}
    raise TypeError(f"Unsupported runtime snapshot: {type(value).__name__}")


def parent_snapshot(session):
    # Dataclass fields are the ORIGINAL V7 state; dynamic V13 banks excluded.
    return {f.name: signature(getattr(session, f.name)) for f in fields(session)}


def head_snapshot(head, *, references=False):
    return {name: {"exists": hasattr(head, name), "value": signature(getattr(head, name, None), references=references)}
            for name in ("_memory_cache", "_vision_cache", "_inference_gen")}


def modules(policy):
    result = {"model": policy.model, "parent": policy.memory, "cvom": policy.cvom}
    if hasattr(policy, "visual_memory"):
        result["visual"] = policy.visual_memory
    return result


def frozen(policy):
    return all(not m.training and all(not p.requires_grad and p.grad is None for p in m.parameters())
               for m in modules(policy).values())


@contextmanager
def same_case_seed(seed):
    original = proof.rng_state()
    try:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        proof.restore_rng(original)


@contextmanager
def capture_head(head):
    """Instance-only transparent wrappers; no class/global or file mutation."""
    calls = {"process": 0, "action": 0, "predictions": []}
    process, action = head.process_backbone_output, head.get_action_with_features
    def processed(*args, **kwargs):
        calls["process"] += 1
        return process(*args, **kwargs)
    def generated(*args, **kwargs):
        calls["action"] += 1
        result = action(*args, **kwargs)
        prediction = result["action_pred"]
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("Nonfinite native action prediction")
        calls["predictions"].append(prediction.detach().cpu().clone())
        return result
    with _scoped_methods(head, {"process_backbone_output": processed, "get_action_with_features": generated}):
        yield calls


def flat_observation(observation, states, language_key):
    """Original simulator-wrapper shape, with recorded observed telemetry only."""
    result = {f"video.{camera}": np.asarray(observation["images"][camera][0])[None, None]
              for camera in CAMERA_ORDER}
    for key, value in states.items():
        if "action" in key.lower() or not isinstance(value, np.ndarray) or value.dtype != np.float32:
            raise ValueError("Only original observed FP32 state groups are permitted")
        result[f"state.{key}"] = value[None]
    result[language_key] = [observation["text"]]
    return result


def prepare_case(policy, record, dataset, backend):
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    embodiment = policy.embodiment_tag.value
    loader = LeRobotEpisodeLoader(Path(dataset), policy.processor.modality_configs[embodiment], video_backend=backend)
    primes = [frame for frame in record["canonical_frames"] if frame < record["n_demo"]]
    frames = sorted(set(primes + record["frames"] + [record["n_demo"]]))
    observations = load_visual_observations(policy.processor, loader, record, frames, embodiment=embodiment)
    by_frame = {obs["frame"]: obs for obs in observations}
    native = {frame: flat_observation(by_frame[frame],
        proof.observed_states(loader, policy.processor, record, frame, embodiment), policy.language_key)
        for frame in primes + [record["n_demo"]]}
    tail = {"n_demo": record["n_demo"], "frames": list(record["frames"]),
            "images": {camera: np.stack([by_frame[f]["images"][camera][0] for f in record["frames"]])
                       for camera in CAMERA_ORDER} if record["frames"] else {},
            "texts": [by_frame[f]["text"] for f in record["frames"]]}
    return {"episode_id": record["episode_id"], "n_demo": record["n_demo"], "primes": primes,
            "observations": native, "tail": tail}


def ingest_and_check(policy, sid, seed, prepared, payload, capture, row):
    """Capture independent invariants immediately around the ACTUAL RGB RPC."""
    session, head = policy.sessions[sid], policy.model.action_head
    before = {key: signature(value, references=True) for key, value in vars(session).items()}
    table = [(key, id(value)) for key, value in policy.sessions.items()]
    caches, rng = head_snapshot(head, references=True), proof.rng_state()
    versions = {name: proof.version_state(module) for name, module in modules(policy).items()}
    input_before, counts = signature(prepared["tail"]), {name: capture[name] for name in ("process", "action")}
    extracted, backbone_calls = [], []
    encode = policy.visual_memory.encode_bank_images
    def encoded(images, *args, **kwargs):
        extracted.append(images.detach().cpu().clone())
        return encode(images, *args, **kwargs)
    with ExitStack() as stack:
        handle = policy.model.backbone.register_forward_hook(lambda *args: backbone_calls.append(1))
        stack.callback(handle.remove)
        stack.enter_context(_scoped_methods(policy.visual_memory, {"encode_bank_images": encoded}))
        response = policy.ingest_demo_tail(session_id=sid, episode_seed=seed, **prepared["tail"])
    checks = row.setdefault("checks", {})
    checks.update(parent_fields_and_references_unchanged=all(
        signature(getattr(session, key), references=True) == saved for key, saved in before.items()),
        only_tail_state_added=set(vars(session)) == set(before) | {"visual_tail_bank", "visual_tail_metadata"},
        session_table_unchanged=table == [(key, id(value)) for key, value in policy.sessions.items()],
        canonical_head_cache_and_generator_unchanged=caches == head_snapshot(head, references=True),
        global_rng_unchanged=proof.tree_equal(rng, proof.rng_state()),
        frozen_versions_unchanged=versions == {name: proof.version_state(module) for name, module in modules(policy).items()},
        zero_head_process_or_action_calls=counts == {name: capture[name] for name in counts},
        exactly_one_backbone_per_tail_frame=len(backbone_calls) == len(prepared["tail"]["frames"]),
        rgb_input_unchanged=input_before == signature(prepared["tail"]), all_modules_frozen=frozen(policy),
        tail_frames_exact=session.visual_tail_bank.frames[0].tolist() == prepared["tail"]["frames"],
        tail_all_demo=bool(session.visual_tail_bank.is_demo.all()))
    if len(extracted) != len(payload["images"]):
        raise ValueError("Missing/excess actual image encodings during ingestion")
    row["rgb_feature_comparisons"] = [proof.numerical_comparison(payload["images"][i], feature[0])
                                       for i, feature in enumerate(extracted)]
    checks["finite_rgb_features"] = all(v["finite"] and v["same_shape"] and v["same_dtype"]
                                         for v in row["rgb_feature_comparisons"])
    row.update(response=response, canonical_bank=signature(session.visual_bank),
               tail_bank=signature(session.visual_tail_bank),
               parent_before_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest())


def run_case(policy, prepared, payload, seed, row, persist, *, candidate=False, reference=None):
    sid = f"v13-online-proof:train:{prepared['episode_id']}"
    wrapper = Gr00tSimPolicyWrapper(policy)
    checks, calls, raw_calls = row.setdefault("checks", {}), row.setdefault("calls", []), []
    row.update(episode_id=prepared["episode_id"], n_demo=prepared["n_demo"], seed=seed, session_id=sid,
               input_sha256=hashlib.sha256(json.dumps(signature(prepared), sort_keys=True).encode()).hexdigest())
    if reference is not None and row["input_sha256"] != reference["input_sha256"]:
        raise ValueError("Parent/V13 observations differ")
    original_input = signature(prepared)
    policy.reset(options={"session_ids": [sid]})
    checks["reset_before_empty"] = not policy.sessions
    try:
        with capture_head(policy.model.action_head) as capture:
            for index, frame in enumerate(prepared["primes"] + [prepared["n_demo"]]):
                passive = frame < prepared["n_demo"]
                if candidate and not passive and prepared["tail"]["frames"]:
                    row["ingest"] = {"checks": {}}
                    ingest_and_check(policy, sid, seed, prepared, payload, capture, row["ingest"])
                    persist()
                    if not all(row["ingest"]["checks"].values()):
                        raise RuntimeError("RGB ingestion invariant failed")
                options = {"session_ids": [sid], "episode_seed": seed, "reset_memory": [index == 0],
                           "frame_index": frame, "passive": passive, "prime_only": passive, "executed_actions": None}
                previous = len(capture["predictions"])
                actions, info = wrapper.get_action(prepared["observations"][frame], options=options)
                prediction = None if passive else capture["predictions"][-1]
                checks[f"call{index}_native_action_count"] = len(capture["predictions"]) - previous == int(not passive)
                checks[f"call{index}_head_caches_cleared"] = all(
                    getattr(policy.model.action_head, name, None) is None for name in ("_memory_cache", "_vision_cache", "_inference_gen"))
                session = policy.sessions[sid]
                raw = {"actions": {key: np.array(value, copy=True) for key, value in actions.items()}, "prediction": prediction}
                raw_calls.append(raw)
                call = {"frame": frame, "passive": passive, "parent": parent_snapshot(session),
                        "head": head_snapshot(policy.model.action_head), "info": info,
                        "actions": signature(raw["actions"]), "native_prediction": signature(prediction),
                        "global_rng": signature(proof.rng_state())}
                if candidate:
                    target = reference["calls"][index]
                    expected = reference["raw_calls"][index]
                    checks[f"call{index}_parent_state_exact"] = call["parent"] == target["parent"]
                    checks[f"call{index}_parent_diagnostics_exact"] = info["long_memory"] == target["info"]["long_memory"]
                    checks[f"call{index}_global_rng_exact"] = call["global_rng"] == target["global_rng"]
                    checks[f"call{index}_visual_read_off"] = not info["visual_memory"]["read_enabled"] and not info["demo_tail"]["read_enabled"]
                    if set(raw["actions"]) != set(expected["actions"]):
                        raise ValueError("Decoded action keys differ")
                    call["decoded_comparisons"] = {key: proof.numerical_comparison(torch.from_numpy(expected["actions"][key]),
                        torch.from_numpy(value)) for key, value in raw["actions"].items()}
                    if not passive:
                        call["native_euler4_comparison"] = proof.numerical_comparison(expected["prediction"], prediction)
                calls.append(call)
                persist()
                if not all(checks.values()):
                    raise RuntimeError("Online runtime parity invariant failed")
        checks["original_inputs_unchanged"] = signature(prepared) == original_input
        checks["one_native_euler4_call"] = capture["action"] == 1
        checks["canonical_process_count"] = capture["process"] == len(prepared["primes"]) + 1
        if not prepared["tail"]["frames"]:
            checks["no_demo_ingest_skipped"] = "ingest" not in row and not hasattr(policy.sessions[sid], "visual_tail_bank")
    finally:
        proof.guarded_check(row, "reset_after_empty", lambda: (policy.reset(options={"session_ids": [sid]}), not policy.sessions)[1])
        persist()
    return {**row, "raw_calls": raw_calls}


def create_policy(plan, role, device):
    if role["kind"] == "parent":
        return LongMemoryV7Policy(plan["base_model"], plan["parent"]["path"], device=device,
                                  strict=True, write_policy="checkpoint", memory_off=False)
    return DemoTailV13Policy(plan["base_model"], role["path"], device=device, strict=True,
                            write_policy="checkpoint", visual_read_off=True, expected_include_tail=True)


def source_hashes():
    hashes = proof.source_hashes()
    for name in ("verify_demo_tail_online_v13.py", "policy_demo_tail_v13.py", "policy_demo_tail_ingest_v13.py",
                 "demo_tail_rng_v13.py", "checkpoint_demo_tail_v13.py", "policy_visual_differential_v12.py",
                 "checkpoint_visual_differential_v12.py", "policy_visual_patch_v11.py", "checkpoint_visual_patch_v11.py",
                 "serve_demo_tail_v13.py", "rollout_demo_tail_v13.py"):
        path = ROOT / "run_scripts/robomme" / name
        hashes[str(path.relative_to(ROOT))] = proof.sha(path)
    return hashes


def preflight(args):
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    reader = DemoTailSidecar(args.sidecar_dir, expected_cache_fingerprint=cache.manifest["fingerprint"])
    inventory = reader.manifest["plan"]
    inventory_checks = reader.manifest.get("inventory_checks", {})
    if (inventory["scope"] != "inventory_train_val" or reader.manifest.get("status") != "complete"
            or set(inventory_checks) != {"zero_action_calls", "original_short_memory_unchanged", "frozen_model_content_unchanged"}
            or not all(value is True for value in inventory_checks.values())):
        raise ValueError("Use the completed full inventory with its checked no-demo records")
    records = sorted((r for r in inventory["episodes"] if r["split"] == "train"), key=lambda r: r["episode_id"])
    selected = [next(r for r in records if r["episode_id"] == 0), next(r for r in records if r["n_demo"] == 0)]
    if selected[0]["n_demo"] != 48 or selected[0]["frames"] != list(range(33, 48)):
        raise ValueError("Declared ep0 raw demo timing differs")
    base = Path(cache.manifest["model_path"]).resolve()
    if Path(inventory["base_model"]).resolve() != base:
        raise ValueError("Original base/sidecar differ")
    if not 1 <= len(args.checkpoint) <= 2 or len(set(args.checkpoint)) != len(args.checkpoint):
        raise ValueError("Provide one or two distinct genuine V13 bundles")
    roles, parent, protected = [{"kind": "parent"}], None, {}
    for given in args.checkpoint:
        path = Path(given).resolve()
        info = checkpoint.checkpoint_info(base, path)
        meta = info["metadata"]
        if info["config"]["include_tail"] is not True or meta["cache_fingerprint"] != cache.manifest["fingerprint"]:
            raise ValueError("Require same-cache genuine tail-on V13 checkpoints")
        if meta["sidecar"]["fingerprint"] != reader.manifest["fingerprint"]:
            raise ValueError("Declared original training sidecar differs")
        if parent is not None and parent != meta["frozen_parent"]:
            raise ValueError("Checkpoint parent identity differs")
        parent = meta["frozen_parent"]
        roles.append({"kind": "v13_off", "path": str(path), "step": info["step"],
                      "label": "INITIALIZATION" if info["step"] == 0 else "POSITIVE_STEP",
                      "visual_sha256": meta["payload_sha256"]["visual.safetensors"]})
        protected.update({str(base / key): value for key, value in meta["frozen_base"]["files_sha256"].items()})
        protected.update({str(Path(parent["path"]) / key): value for key, value in parent["files_sha256"].items()})
        protected.update({str(path / name): proof.sha(path / name) for name in ("checkpoint.json", "visual.safetensors", "training_state.pt")})
    if len(roles) == 3 and not (roles[1]["step"] == 0 and roles[2]["step"] > 0):
        raise ValueError("Two-bundle proof must explicitly order initialization then positive step")
    for record in selected:
        if record["episode_id"] not in cache.manifest["splits"]["train"]:
            raise ValueError("Selected episode is not in original TRAIN split")
        reader.load(record["episode_id"])
        files = [record[key] for key in ("parquet_path", "tasks_path", "canonical_cache_path")]
        files += list(record["video_paths"].values())
        for filename in files:
            protected[filename] = inventory["files_sha256"][filename]
        sidecar_file = reader.path / reader._records[record["episode_id"]]["path"]
        protected[str(sidecar_file)] = proof.sha(sidecar_file)
    for path in (reader.path / "manifest.json", cache.path / "manifest.json", *Path(inventory["dataset_path"]).glob("meta/*.json*")):
        filename = str(path.resolve())
        protected[filename] = inventory["files_sha256"].get(filename, proof.sha(path))
    for name, expected in inventory["source_sha256"].items():
        if proof.sha(ROOT / name) != expected:
            raise ValueError("Frozen extraction dependency changed")
    for filename, expected in protected.items():
        if proof.sha(filename) != expected:
            raise ValueError(f"Protected input changed: {filename}")
    if inventory["runtime"] != runtime_identity():
        raise ValueError("Original RGB extraction runtime differs")
    output = validate_output_scope(args.output_dir, cache.path, reader.path, base, parent["path"],
                                   inventory["dataset_path"], *args.checkpoint)
    if output.exists():
        raise FileExistsError("Use a NEW online diagnostic output")
    plan = {"args": vars(args), "base_model": str(base), "parent": parent, "roles": roles,
            "cases": selected, "dataset_path": inventory["dataset_path"], "video_backend": inventory["video_backend"],
            "cache_fingerprint": cache.manifest["fingerprint"], "sidecar_fingerprint": reader.manifest["fingerprint"],
            "files_sha256": protected, "source_sha256": source_hashes(), "runtime": runtime_identity(),
            "limitations": LIMITATIONS}
    return plan, reader, output


def completion_code(report):
    if not report.get("passed"):
        return 1
    comparisons = [value for role in report["roles"] for case in role.get("cases", [])
                   for call in case.get("calls", []) for value in call.get("decoded_comparisons", {}).values()]
    comparisons += [call["native_euler4_comparison"] for role in report["roles"] for case in role.get("cases", [])
                    for call in case.get("calls", []) if "native_euler4_comparison" in call]
    comparisons += [value for role in report["roles"] for case in role.get("cases", [])
                    for value in case.get("ingest", {}).get("rgb_feature_comparisons", [])]
    return 0 if comparisons and all(value["exact"] for value in comparisons) else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-dir", "sidecar-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--checkpoint", required=True, nargs="+")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--seed", type=int, default=9111)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.seed != 9111:
        raise ValueError("Predeclared diagnostic seed is 9111")
    torch.set_num_threads(2)
    plan, reader, output = preflight(args)
    if args.preflight_only:
        print(json.dumps({"preflight_passed": True, "roles": plan["roles"],
                          "cases": [{k: r[k] for k in ("episode_id", "n_demo", "frames")} for r in plan["cases"]],
                          "cuda_initialized": torch.cuda.is_initialized(), "output_created": False}))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "plan.json", plan)
    report = {"passed": False, "roles": [], "checks": {}, "limitations": LIMITATIONS}
    persist = lambda: _atomic_json(output / "result.json", report)
    caller, references = proof.rng_state(), {}
    try:
        if args.device.startswith("cuda"):
            torch.cuda.init(); caller = proof.rng_state()
        report["rng_boundary"] = "after requested-device initialization, before real policy construction"
        for role in plan["roles"]:
            record = {**role, "cases": [], "checks": {}}
            report["roles"].append(record)
            policy = None
            before = None
            try:
                policy = create_policy(plan, role, args.device)
                if not frozen(policy) or policy.model.action_head.num_inference_timesteps != 4:
                    raise ValueError("Need fully frozen original native Euler4 policy")
                before = {name: proof.module_sha(module) for name, module in modules(policy).items()}
                record["frozen_before_sha256"] = before
                if role["kind"] != "parent":
                    expected = report["roles"][0]["frozen_before_sha256"]
                    record["checks"]["same_original_parent_and_model"] = all(before[key] == value for key, value in expected.items())
                    if not record["checks"]["same_original_parent_and_model"]:
                        raise ValueError("V13 loaded different original parent/model weights")
                for raw in plan["cases"]:
                    case_row = {"checks": {}}
                    record["cases"].append(case_row)
                    with same_case_seed(args.seed):
                        prepared = prepare_case(policy, raw, plan["dataset_path"], plan["video_backend"])
                        value = run_case(policy, prepared, reader.load(raw["episode_id"]), args.seed, case_row, persist,
                                         candidate=role["kind"] != "parent", reference=references.get(raw["episode_id"]))
                    if role["kind"] == "parent":
                        references[raw["episode_id"]] = value
            finally:
                if policy is not None:
                    def content_guard():
                        record["frozen_after_sha256"] = {name: proof.module_sha(module) for name, module in modules(policy).items()}
                        return before is not None and record["frozen_after_sha256"] == before
                    proof.guarded_check(record, "frozen_content_unchanged", content_guard)
                    proof.guarded_check(record, "frozen_no_gradients", lambda: frozen(policy))
                    proof.guarded_check(record, "sessions_reset", lambda: (policy.reset(), not policy.sessions)[1])
                    del policy
                    gc.collect()
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                persist()
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        def rng_guard():
            proof.restore_rng(caller)
            return proof.tree_equal(caller, proof.rng_state())
        proof.guarded_check(report, "caller_rng_restored", rng_guard)
        proof.guarded_check(report, "sources_unchanged", lambda: source_hashes() == plan["source_sha256"])
        proof.guarded_check(report, "protected_inputs_unchanged", lambda: all(proof.sha(p) == h for p, h in plan["files_sha256"].items()))
        proof.guarded_check(report, "runtime_unchanged", lambda: runtime_identity() == plan["runtime"])
        report["passed"] = ("error" not in report and len(report["roles"]) == len(plan["roles"])
            and all(report["checks"].values()) and all(len(role["cases"]) == 2 and all(role["checks"].values())
            and all(len(case.get("calls", [])) == (4 if case.get("n_demo") == 48 else 1)
                    and all(case["checks"].values()) and all(case.get("ingest", {}).get("checks", {}).values())
                    for case in role["cases"]) for role in report["roles"]))
        report["exit_code"] = completion_code(report)
        persist()
    print(json.dumps({"passed": report["passed"], "exit_code": report["exit_code"]}), flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
