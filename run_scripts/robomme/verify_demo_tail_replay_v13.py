#!/usr/bin/env python3
"""TRAIN-only V13 replay/gradient wiring diagnostic; never saves model weights.

Requires an already completed, hash-verified sidecar. Three predeclared first
execution queries receive zero-init/off checks; exactly TWO transient visual
updates use only the first query. This is not online ingestion or task quality.
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

from run_scripts.robomme import verify_visual_patch_v11 as native
from run_scripts.robomme import visual_demo_tail_bank_v13 as bank_core
from run_scripts.robomme.demo_tail_sidecar_v13 import DemoTailSidecar, runtime_identity
from run_scripts.robomme.checkpoint_visual_patch_v11 import checkpoint_info as initial_info
from run_scripts.robomme.replay_demo_tail_v13 import replay_demo_tail
from run_scripts.robomme.framewise_demo_tail_v13 import REPLAY_ENCODING
from run_scripts.robomme.replay_visual_patch_v11 import _columns, _rows
from run_scripts.robomme.verify_demo_tail_v13 import (
    numerical_comparison, rng_state, restore_rng, tree_equal, guarded_check,
)

CAMERAS = bank_core.CAMERA_ORDER
LIMITATIONS = [
    "Three predeclared TRAIN first-execution queries, not validation or task success.",
    "Exactly two transient AdamW updates on the first query, fixed noise; no saved weights or optimizer.",
    "Frozen archive1250 and complete adapted AE; only 14 visual parameter tensors are optimized.",
    "Canonical-only uses the real original canonical observations and SAME weights with framewise encoding, never a fake empty demo sidecar.",
    "Input-image gradients, parameter gradients, and changed BF16 outputs are different wiring checks, not useful retrieval.",
    "Production framewise replay is compared with ordinary canonical encode_observation+append and singleton tail encode_bank_images+append_bank_images.",
    "Metadata, Z/C, READ and native Euler4 must be exact against the independent online encoding comparator; repeated-run noise is recorded.",
    "Directional per-value BF16 nextafter bounds are measurements, not numerical acceptance rules; any nonexact comparison requires review (exit 2).",
    "Exact READ-off native Euler4 parity is required, but this does not establish online ingestion/session parity.",
    "GT enters only the original flow supervisor after observation-only bank/READ construction.",
    "This consumes cached tensors, not the RGB extractor; raw extraction is NOT RNG-transparent and future online callers must isolate it.",
]


def source_hashes():
    result = native.source_hashes()
    for name in ("verify_demo_tail_replay_v13.py", "demo_tail_sidecar_v13.py", "verify_demo_tail_v13.py",
                 "visual_demo_tail_bank_v13.py", "visual_differential_memory_v12.py",
                 "replay_visual_differential_v12.py", "replay_visual_patch_v11.py", "replay_demo_tail_v13.py",
                 "framewise_demo_tail_v13.py", "checkpoint_visual_patch_v11.py"):
        path = ROOT / "run_scripts/robomme" / name
        result[str(path.relative_to(ROOT))] = native.sha(path)
    return result


def select_cases(cache, reader, episodes):
    """No quality selection: first two planned proof IDs, then declared ep0."""
    records = reader.manifest["plan"]["episodes"]
    selected = [r for r in records if r["role"] == "train_proof"][:2]
    selected += [r for r in records if r["episode_id"] == 0]
    if len(selected) != 3 or len({r["episode_id"] for r in selected}) != 3:
        raise ValueError("Need two distinct proof episodes plus separate ep0")
    result = []
    for planned in selected:
        eid = planned["episode_id"]
        if planned["split"] != "train" or eid not in cache.manifest["splits"]["train"]:
            raise ValueError("Replay proof is TRAIN-only")
        ep = episodes.fetch(eid)
        execution = (~ep["is_demo"]).nonzero().flatten()
        if not len(execution):
            raise ValueError("Missing first execution query")
        q = int(execution[0])
        if q < 1 or int(ep["frames"][q]) != planned["n_demo"]:
            raise ValueError("Require true first execution query after a nonempty canonical demo")
        native.validate_decision(ep, q)
        if not bool(ep["action_mask"][q].any()):
            raise ValueError("First execution query lacks observed actions; never skip forward")
        payload = reader.load(eid)
        record = reader._records[eid]
        if not len(payload["frames"]):
            raise ValueError("Proof requires a real nonempty demo tail")
        result.append({"episode_id": eid, "query": q, "frame": int(ep["frames"][q]),
                       "episode": ep, "payload": payload, "record": record})
    return result


def observation_view(episode, query, payload, *, track=False, device="cpu"):
    """Copy only [0,q] original observations; GT/state/parent are inaccessible."""
    observations = {key: [episode[key][i] for i in range(query + 1)] for key in native.OBSERVATION_KEYS}
    leaves = {}
    if track:
        for i, name in ((0, "earliest_canonical_image"), (query, "current_image")):
            leaf = observations["features"][i].detach().to(device).clone().requires_grad_(True)
            observations["features"][i] = leaf
            leaves[name] = leaf
        payload = {**payload, "images": payload["images"].detach().to(device).clone().requires_grad_(True)}
        leaves["tail_images"] = payload["images"]
    return observations, payload, leaves


def sequential_bank(memory, observations, query, payload, validated_frames):
    """Actual online paths, NOT the production bank-image helper for canonical.

    Canonical observations retain original sequence/masks/short and take the
    ordinary encode_observation+append path. Only the new tail has bank-only
    images. Both paths encode singleton observations in validated raw order.
    """
    rows = _rows(_columns(observations), query, memory)
    entries = {int(row[3]): ("canonical", row) for row in rows}
    for i, frame in enumerate(payload["frames"].tolist()):
        if frame in entries:
            raise ValueError("Duplicate sequential evidence")
        entries[frame] = ("tail", i)
    if sorted(entries) != validated_frames:
        raise ValueError("Sequential replay differs from validated merged history")
    bank = None
    for frame in validated_frames:
        kind, value = entries[frame]
        if kind == "canonical":
            feature, image_mask, attention_mask, raw_frame, is_demo = value
            ordinary = memory.encode_observation(feature[None], image_mask[None], attention_mask[None],
                [int(raw_frame)], [bool(is_demo)], camera_order=CAMERAS)
            bank = memory.append(bank, ordinary)
        else:
            image = payload["images"][value].to(memory._device())
            record = memory.encode_bank_images(image[None], [frame], [bool(payload["is_demo"][value])], camera_order=CAMERAS)
            bank = memory.append_bank_images(bank, record)
    return memory.empty_bank(1) if bank is None else bank


def replay_pair(memory, observations, query, payload, record, bindings):
    """No GT/action/state input. Current observation is READ, never appended."""
    # Exercise the exact shared future-training entry point in every arm.
    merged_features, merged = replay_demo_tail(memory, observations, query, include_tail=True,
        sidecar=payload, record=record, camera_order=CAMERAS, **bindings)
    canonical_features, canonical = replay_demo_tail(memory, observations, query, include_tail=False,
        camera_order=CAMERAS)
    off_features, off_bank = replay_demo_tail(memory, observations, query, include_tail=True,
        sidecar=payload, record=record, camera_order=CAMERAS, visual_read_enabled=False, **bindings)
    sequential = sequential_bank(memory, observations, query, payload, merged.frames[0].tolist())
    repeated_features, repeated = replay_demo_tail(memory, observations, query, include_tail=True,
        sidecar=payload, record=record, camera_order=CAMERAS, **bindings)
    device = memory._device()
    current = memory.encode_observation(observations["features"][query].to(device)[None],
        observations["image_masks"][query].to(device)[None], observations["attention_masks"][query].to(device)[None],
        [int(observations["frames"][query])], [bool(observations["is_demo"][query])], camera_order=CAMERAS)
    features = {"merged": merged_features, "canonical": canonical_features,
                "sequential": memory.read(current, sequential), "repeat": repeated_features, "off": off_features}
    comparisons = {f"{name}_production_vs_online": numerical_comparison(getattr(merged, name), getattr(sequential, name))
                   for name in ("tokens", "content", "frames", "is_demo", "valid")}
    comparisons.update({f"{name}_production_repeat": numerical_comparison(getattr(merged, name), getattr(repeated, name))
                        for name in ("tokens", "content")})
    comparisons.update({f"{name}_production_off": numerical_comparison(getattr(merged, name), getattr(off_bank, name))
                        for name in ("tokens", "content")})
    comparisons.update(read_production_vs_online=numerical_comparison(features["merged"], features["sequential"]),
                       read_production_repeat=numerical_comparison(features["merged"], features["repeat"]))
    if not all(v["finite"] and v["same_shape"] and v["same_dtype"] for v in comparisons.values()):
        raise FloatingPointError("Invalid replay comparison")
    for bank in (merged, canonical, sequential, repeated, off_bank):
        if not bool((bank.frames < current.frames[:, None]).all()):
            raise ValueError("Current/future observation entered a bank")
    return features, {"canonical": canonical, "merged": merged}, comparisons


def immutable_query(episode, query):
    return {key: native.tensor_digest(episode[key][query]) for key in
        ("features", "short", "state", "targets", "target_mask", "actions", "action_mask", "frames", "is_demo")}


def case_metadata(case):
    q, ep, payload = case["query"], case["episode"], case["payload"]
    canonical = ep["frames"][:q].tolist()
    tail = payload["frames"].tolist()
    merged = sorted(canonical + tail)
    if len(merged) != len(set(merged)) or any(frame >= case["frame"] for frame in merged):
        raise ValueError("Prepared replay contains duplicate/current/future raw frames")
    return {**{key: case[key] for key in ("episode_id", "query", "frame")},
        "canonical_frames": canonical, "tail_frames": tail, "merged_frames": merged,
        "bank_observation_counts": {"canonical": len(canonical), "merged": len(merged)},
        "bank_token_counts": {"canonical": 162 * len(canonical), "merged": 162 * len(merged)},
        "current_query_sha256": immutable_query(ep, q)}


def image_mask(episode, query, device):
    result = (episode["image_masks"][query] & episode["attention_masks"][query]).to(device).clone()
    result[-4:] = False
    return result


def bf16_ulp_measurements(reference, candidate):
    """Measure each differing BF16 value against ONE nextafter step toward it.

    No maximum/global epsilon is substituted for the per-element spacing;
    binade boundaries, negative numbers and zero therefore keep their true ULP.
    This is evidence only and never changes a numerical PASS criterion.
    """
    if reference.dtype != torch.bfloat16 or candidate.dtype != torch.bfloat16 or reference.shape != candidate.shape:
        raise ValueError("ULP measurement requires equal-shape BF16 tensors")
    a, b = reference.detach().cpu().flatten(), candidate.detach().cpu().flatten()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise FloatingPointError("Nonfinite BF16 ULP input")
    indices = (a != b).nonzero().flatten()
    left, right = a[indices], b[indices]
    neighbor = torch.nextafter(left, right)
    delta = (right.double() - left.double()).abs()
    spacing = (neighbor.double() - left.double()).abs()
    if not bool((spacing > 0).all()):
        raise FloatingPointError("Invalid directional BF16 nextafter spacing")
    examples = [{"flat_index": int(index), "reference": float(x), "candidate": float(y),
                 "one_ulp_toward_candidate": float(s), "absolute_difference": float(d)}
                for index, x, y, s, d in zip(indices[:8], left[:8], right[:8], spacing[:8], delta[:8])]
    return {"dtype": "torch.bfloat16", "total_values": a.numel(), "differing_values": indices.numel(),
            "within_one_directional_ulp": int((delta <= spacing).sum()),
            "beyond_one_directional_ulp": int((delta > spacing).sum()),
            "max_directional_ulp_ratio": float((delta / spacing).max()) if len(indices) else 0.,
            "examples": examples, "interpretation": "measurement_only_no_acceptance_threshold"}


def diagnose_case(head, parent, case, visual, bindings, seed, report, persist, *, updates=0):
    if updates not in (0, 2) or visual.read_mode != "differential":
        raise ValueError("Only zero-check or exactly two differential diagnostic updates")
    if head.training or parent.training or any(p.requires_grad or p.grad is not None
            for module in (head, parent) for p in module.parameters()) or head.num_inference_timesteps != 4:
        raise ValueError("Require frozen/eval archive and actual native Euler4 expert")
    ep, q, payload, record = (case[k] for k in ("episode", "query", "payload", "record"))
    before = immutable_query(ep, q)
    payload_before = native.tensor_digest(payload["images"])
    checks = report.setdefault("checks", {})
    checks["actual_bf16_head_fp32_visual"] = next(head.parameters()).dtype == torch.bfloat16 and all(
        parameter.dtype == torch.float32 for parameter in visual.parameters())
    report.update(optimizer_updates=0, passes=[], initial_visual_sha256=native.module_digest(visual),
        parameter_count=sum(p.numel() for p in visual.parameters()),
        optimizer_parameter_names=[name for name, _ in visual.named_parameters()] if updates else [])
    initial = {name: p.detach().clone() for name, p in visual.named_parameters()}
    if len(initial) != 14 or bool(initial["output_projection.weight"].count_nonzero()):
        raise ValueError("Require original 14-parameter zero-output initialization")
    with torch.no_grad():
        fused, metrics = native.replay_queries(parent, ep, [q], mode="archive")[q]
        original, _, _, _ = native.cached_inputs(head, ep, q, fused)
        reference_flow = native.expert_episode_flow_loss(head, ep, q, fused, seed=seed)
        reference_generated = native.generated_action(head, ep, q, fused, seed=seed + 1)
    fused_before = native.tensor_digest(fused)
    report["parent_read_metrics"] = {k: float(v) for k, v in metrics.items()}
    report["original_generated_sha256"] = native.tensor_digest(reference_generated)
    noise, time = native.sample_noise_time(head, ep["targets"][q].to(original)[None], seed)
    fixed = {"noise": native.tensor_digest(noise), "time": native.tensor_digest(time)}
    report["fixed_flow"] = {**fixed, "seed": seed, "generation_seed": seed + 1, "realized_time": time.flatten().tolist()}
    optimizer = torch.optim.AdamW(visual.parameters(), lr=1e-4, weight_decay=0.) if updates else None
    for step in range(updates + 1):
        visual.zero_grad(set_to_none=True)
        observations, bound_payload, leaves = observation_view(ep, q, payload, track=bool(updates), device=visual._device())
        features, banks, comparisons = replay_pair(visual, observations, q, bound_payload, record, bindings)
        # Freeze the original parent short READ in every branch AFTER visual READ.
        conditioned = {name: native.replace_short(value, fused) for name, value in features.items()}
        mask = image_mask(ep, q, original.device)
        row = {"after_updates": step, "replay_encoding": REPLAY_ENCODING, "comparisons": comparisons,
            "read_bf16_ulp": bf16_ulp_measurements(conditioned["merged"], conditioned["sequential"]),
            "canonical_frames": banks["canonical"].frames[0].tolist(), "tail_frames": payload["frames"].tolist(),
            "merged_frames": banks["merged"].frames[0].tolist(),
            "bank_counts": {name: value.tokens.shape[1] for name, value in banks.items()},
            "parameter_changes_from_initial": {name: native.tensor_record(p.detach() - initial[name]) for name, p in visual.named_parameters()},
            "image_changed_fraction": float((conditioned["merged"].detach()[0, mask] != original[0, mask]).float().mean())}
        report["passes"].append(row)
        checks[f"pass{step}_only_images_changed"] = all(torch.equal(value.detach()[0, ~mask], original[0, ~mask]) for value in conditioned.values())
        checks[f"pass{step}_off_features_exact"] = torch.equal(conditioned["off"].detach(), original)
        checks[f"pass{step}_full_tail_history"] = banks["merged"].tokens.shape[1] == q + len(payload["frames"])
        generated = {name: native.generated_at_features(head, ep, q, conditioned[name].detach(), seed + 1)
                     for name in ("off", "merged", "canonical", "sequential")}
        checks[f"pass{step}_off_native_euler4_exact"] = torch.equal(generated["off"], reference_generated)
        row["generated_comparisons"] = {name: numerical_comparison(reference_generated, value) for name, value in generated.items()}
        row["generated_production_vs_online"] = numerical_comparison(generated["merged"], generated["sequential"])
        row["generated_tensor_sha256"] = {name: native.tensor_digest(value) for name, value in generated.items()}
        checks[f"pass{step}_finite_generated_actions"] = all(value["finite"] for value in row["generated_comparisons"].values())
        flow = native.flow_at_features(head, ep, q, conditioned["merged"], noise, time)
        row["flow_loss"] = float(flow["loss"].detach())
        checks[f"pass{step}_finite_loss"] = bool(torch.isfinite(flow["loss"]))
        if step == 0:
            checks["zero_all_conditioning_exact"] = all(torch.equal(value.detach(), original) for value in conditioned.values())
            checks["zero_native_flow_exact"] = torch.equal(flow["prediction"].detach(), reference_flow["prediction"]) and torch.equal(flow["loss"].detach(), reference_flow["loss"])
            checks["zero_all_generated_exact"] = all(torch.equal(value, reference_generated) for value in generated.values())
        if updates:
            flow["loss"].backward()
            grads = {name: native.tensor_record(p.grad) for name, p in visual.named_parameters()}
            row["parameter_gradients"] = grads
            row["input_gradients"] = {name: native.tensor_record(leaf.grad if name == "tail_images" or leaf.grad is None
                else leaf.grad[image_mask(ep, 0 if name.startswith("earliest") else q, leaf.device)]) for name, leaf in leaves.items()}
            checks[f"pass{step}_finite_gradients"] = all(g["finite"] for g in (*grads.values(), *row["input_gradients"].values()))
            checks[f"pass{step}_output_gradient_nonzero"] = grads["output_projection.weight"]["nonzero"] > 0
            if step == 0:
                checks["zero_upstream_gradients"] = all(g["nonzero"] == 0 for name, g in grads.items() if name != "output_projection.weight")
            else:
                for name in ("image_projection.weight", "query_projection.weight", "key_projection.weight", "value_projection.weight"):
                    checks[f"pass{step}_{name}_gradient_nonzero"] = grads[name]["nonzero"] > 0
                for name in leaves:
                    checks[f"pass{step}_{name}_gradient_nonzero"] = row["input_gradients"][name]["nonzero"] > 0
                checks[f"pass{step}_bf16_images_changed"] = row["image_changed_fraction"] > 0
        checks[f"pass{step}_immutable_query"] = before == immutable_query(ep, q)
        checks[f"pass{step}_immutable_parent_short"] = fused_before == native.tensor_digest(fused)
        checks[f"pass{step}_immutable_sidecar"] = payload_before == native.tensor_digest(payload["images"])
        checks[f"pass{step}_fixed_noise_time"] = fixed == {"noise": native.tensor_digest(noise), "time": native.tensor_digest(time)}
        persist()
        if not all(checks.values()):
            raise RuntimeError("Replay diagnostic check failed; records preserved, no further updates")
        if step < updates:
            optimizer.step()
            report["optimizer_updates"] += 1
            if not all(bool(torch.isfinite(p).all()) for p in visual.parameters()):
                raise FloatingPointError("Nonfinite transient visual parameter")
    report["final_visual_sha256"] = native.module_digest(visual)


def preflight(args):
    cache = native.EpisodeCache(args.cache_dir)
    native.validate_cache_checkpoint(cache.manifest)
    reader = DemoTailSidecar(args.sidecar_dir, expected_cache_fingerprint=cache.manifest["fingerprint"])
    base, parent_path, init_path = (Path(x).resolve() for x in (cache.manifest["model_path"], args.checkpoint, args.visual_init_reference))
    if Path(reader.manifest["plan"]["base_model"]).resolve() != base:
        raise ValueError("Sidecar uses another original base")
    parent = native.v7_checkpoint_info(base, parent_path, expected_stage=1)
    init = initial_info(base, init_path)
    if (parent["step"] != 1250 or parent["config"]["mode"] != "archive" or init["step"] != 0
            or init["metadata"]["frozen_parent"]["path"] != str(parent_path)
            or any(info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"] for info in (parent, init))):
        raise ValueError("Require same-cache original archive1250 and V11 zero initialization")
    config = bank_core.VisualDifferentialConfig(**init["config"]["visual"])
    if config.feature_dim != parent["config"]["memory"]["feature_dim"] or config.num_short_tokens != 4:
        raise ValueError("Visual/parent dimensions differ")
    output = native.validate_output_scope(args.output_dir, cache.path, base, parent_path, init_path, reader.path,
                                         cache.manifest.get("dataset_path"))
    if output.exists():
        raise FileExistsError("Use a NEW diagnostic output directory")
    cases = select_cases(cache, reader, native.MappedEpisodes(cache))
    hashes = dict(reader.manifest["plan"]["files_sha256"])
    for name, expected in reader.manifest["plan"]["source_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or native.sha(path) != expected:
            raise ValueError("Completed sidecar extraction source changed")
        hashes[str(path)] = expected
    if reader.manifest["plan"]["runtime"] != runtime_identity():
        raise ValueError("Completed sidecar extraction runtime differs")
    files = [reader.path / "manifest.json", cache.path / "manifest.json"]
    files += [reader.path / r["path"] for r in reader._records.values()]
    files += [parent_path / name for name in ("checkpoint.json", "model.safetensors", "expert.safetensors", "cvom.safetensors", "training_state.pt")]
    files += [init_path / name for name in ("checkpoint.json", "visual.safetensors", "training_state.pt")]
    for path in files:
        actual = native.sha(path)
        if str(path) in hashes and hashes[str(path)] != actual:
            raise ValueError("Conflicting protected file identity")
        hashes[str(path)] = actual
    for path, expected in hashes.items():
        if native.sha(path) != expected:
            raise ValueError(f"Sidecar source input changed: {path}")
    plan = {"args": vars(args), "base_model": str(base), "cache_fingerprint": cache.manifest["fingerprint"],
        "sidecar_fingerprint": reader.manifest["fingerprint"], "visual_config": asdict(config),
        "camera_order": list(CAMERAS), "cases": [case_metadata(case) for case in cases],
        "query_rule": "first two TRAIN train_proof records then ep0; first execution only; updates on first case only",
        "replay_encoding": REPLAY_ENCODING,
        "comparison_protocol": {"comparator": "ordinary canonical encode_observation+append; singleton tail encode_bank_images+append_bank_images",
            "initial_cases": 3, "euler4_branches": ["off", "merged", "canonical", "sequential"],
            "initial_branch_comparisons": 12, "awakened_branch_comparisons": 8,
            "production_online_pairs": 5, "exact_required": ["frames", "is_demo", "valid", "tokens", "content", "READ", "Euler4"]},
        "optimizer": {"kind": "AdamW", "learning_rate": 1e-4, "weight_decay": 0., "total_updates": 2},
        "files_sha256": hashes, "source_sha256": source_hashes(), "runtime": runtime_identity(), "limitations": LIMITATIONS}
    return plan, cases, parent, config, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-dir", "sidecar-dir", "checkpoint", "visual-init-reference", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--seed", type=int, default=9111)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.seed != 9111:
        raise ValueError("Predeclared diagnostic seed must be 9111")
    torch.set_num_threads(2)
    plan, cases, parent_info, config, output = preflight(args)
    if args.preflight_only:
        print(json.dumps({"preflight_passed": True, "cases": plan["cases"], "cuda_initialized": torch.cuda.is_initialized(), "output_created": False}))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    native._atomic_json(output / "plan.json", plan)
    report = {"passed": False, "cases": [], "checks": {}, "limitations": LIMITATIONS}
    persist = lambda: native._atomic_json(output / "result.json", report)
    caller_rng, frozen = rng_state(), {}
    try:
        # CUDA initialization cannot be undone. The actual-run RNG boundary
        # starts after requested-device initialization, never in preflight.
        if torch.device(args.device).type == "cuda":
            torch.cuda.init()
            caller_rng = rng_state()
        report["rng_boundary"] = "after explicit requested-device initialization, before model loading"
        head = native.actual_head(plan["base_model"], args.device)
        cfg = parent_info["config"]
        with native.isolated_seed(args.seed, args.device):
            native.install_expert_lora(head, native.LoRAConfig(**cfg["expert"]), targets=cfg["expert_targets"])
            parent = native.RecurrentMemoryV7(native.MemoryV7Config(**cfg["memory"])).to(args.device)
            cvom = native.CVOMV7(parent.config).to(args.device)
        native.load_checkpoint_v7(args.checkpoint, parent, head, cvom)
        for module in (head, parent, cvom):
            module.eval().requires_grad_(False)
        parent.float(); cvom.float()
        frozen = {"head": head, "parent": parent, "cvom": cvom}
        report["frozen_before_sha256"] = {name: native.module_digest(module) for name, module in frozen.items()}
        state = load_file(str(Path(args.visual_init_reference) / "visual.safetensors"), device="cpu")
        for index, case in enumerate(cases):
            with native.isolated_seed(args.seed, args.device):
                visual = bank_core.VisualDemoTailMemoryV13(config, read_mode="differential").to(args.device)
            visual.load_state_dict(state, strict=True)
            row = {k: case[k] for k in ("episode_id", "query", "frame")}
            report["cases"].append(row)
            bindings = {"episode_id": case["episode_id"], "cache_fingerprint": plan["cache_fingerprint"],
                        "sidecar_fingerprint": plan["sidecar_fingerprint"]}
            diagnose_case(head, parent, case, visual, bindings, args.seed, row, persist, updates=2 if index == 0 else 0)
            del visual
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        def restore_caller_rng():
            restore_rng(caller_rng)
            return tree_equal(caller_rng, rng_state())
        guarded_check(report, "caller_rng_restored", restore_caller_rng)
        if frozen and "frozen_before_sha256" in report:
            def frozen_check():
                report["frozen_after_sha256"] = {name: native.module_digest(module) for name, module in frozen.items()}
                return report["frozen_after_sha256"] == report["frozen_before_sha256"]
            guarded_check(report, "frozen_modules_unchanged", frozen_check)
            guarded_check(report, "frozen_no_gradients", lambda: all(not p.requires_grad and p.grad is None for m in frozen.values() for p in m.parameters()))
        guarded_check(report, "source_unchanged", lambda: source_hashes() == plan["source_sha256"])
        guarded_check(report, "protected_files_unchanged", lambda: all(native.sha(p) == h for p, h in plan["files_sha256"].items()))
        guarded_check(report, "runtime_unchanged", lambda: runtime_identity() == plan["runtime"])
        report["passed"] = ("error" not in report and len(report["cases"]) == 3 and all(report["checks"].values())
            and [r["optimizer_updates"] for r in report["cases"]] == [2, 0, 0]
            and len({r["initial_visual_sha256"] for r in report["cases"]}) == 1
            and all(all(r["checks"].values()) for r in report["cases"]))
        report["numerical_review_required"] = any(not value["exact"] for row in report["cases"]
            for stage in row.get("passes", []) for value in stage["comparisons"].values())
        report["numerical_review_required"] |= any(not stage.get("generated_production_vs_online", {}).get("exact", False)
            for row in report["cases"] for stage in row.get("passes", []))
        persist()
    print(json.dumps({k: report[k] for k in ("passed", "numerical_review_required")}), flush=True)
    return 1 if not report["passed"] else 2 if report["numerical_review_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
