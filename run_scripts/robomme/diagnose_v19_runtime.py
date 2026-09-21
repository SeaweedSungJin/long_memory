#!/usr/bin/env python3
"""Bounded, read-only V19 cache/online parity and bank intervention runner.

No policy training, cache writes or simulator rollouts. Outputs must be NEW.
The parity pass replays the same raw RGB/state/canonical endpoints through the
actual cache producer and policy. Only repeated action generation is stubbed
during extraction; unchanged AE objectives are evaluated separately afterward.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gr00t.long_memory.cache import CacheConfig, EpisodeCache, _episode_records, _extract_episode
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.diagnose_v19_memory import interventions, bank_provenance
from run_scripts.robomme.verify_demo_tail_v13 import numerical_comparison, sha


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with Path(path).open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def gpu_snapshot(device):
    """Fail closed if another CUDA compute process is already using the host.

    Conservative: checks both GPUs, not just an ambiguously remapped ordinal.
    No process is stopped, and this is not a resource reservation.
    """
    if not device.startswith("cuda"):
        raise ValueError("Actual BF16 deployment parity requires an explicit CUDA device")
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                            "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
    if query:
        raise RuntimeError("Existing GPU work detected; no model loaded:\n" + query)
    status = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu",
                             "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
    return {"compute_processes_before": query, "devices_before": status}


def selected_rows(validation, task, count):
    rows = [r for r in validation["records"] if r["task"] == task and r["role"] == "reader"]
    # Preserve the original prespecified validation schedule; never select by error.
    ids = list(dict.fromkeys(int(r["episode_id"]) for r in rows))[:count]
    if not ids:
        raise ValueError(f"Task {task!r} absent from validation")
    return [r for r in rows if int(r["episode_id"]) in ids]


def cpu(tensor):
    return tensor.detach().cpu().clone()


@torch.no_grad()
def cached_stages(core, ep):
    """Batched offline encoding vs sequential online recurrence on identical cache tensors."""
    encoded = core.encode_prefix(ep, len(ep["frames"]))
    bank, history, stages, checks = None, None, [], []
    for d in range(len(ep["frames"])):
        before = core.initial_bank() if bank is None else bank
        enabled = not bool(ep["is_demo"][d]) and d > 0
        result = core.step(ep["short"][d:d+1], ep["moment"][d:d+1], ep["state"][d:d+1],
                           ep["frames"][d:d+1], ep["is_demo"][d:d+1], bank=bank,
                           moment_history=history, read_enabled=enabled, event_index=d)
        off_fused, _ = core.read_from_bank(encoded["short"][d:d+1], encoded["query"][d:d+1],
                                           before, read_enabled=enabled)
        for name, ref, cur in (("encoded", encoded["query"][d:d+1], result["encoded_current"]),
                               ("fused", off_fused, result["fused"])):
            checks.append({"decision": d, "stage": name, **numerical_comparison(ref, cur)})
        stages.append({"moment": cpu(ep["moment"][d:d+1]), "short": cpu(ep["short"][d:d+1]),
                       "state": cpu(ep["state"][d:d+1]), "encoded": cpu(result["encoded_current"]),
                       "bank_before": cpu(before), "bank_after": cpu(result["bank"]),
                       "fused": cpu(result["fused"])})
        bank, history = result["bank"], result["moment_history"]
    return stages, checks


@torch.inference_mode()
def live_extract(policy, loader, record, ep, *, amp, repeat_name, cache_quantization=False):
    """Run the actual policy including session/reset/cadence; bypass only denoising.

    GT controls supplied as executed_actions are past physical controls and are
    not read by the V19 observation encoder. They satisfy the real session API.
    This is teacher-trajectory observation replay, NOT a closed-loop rollout.
    """
    import pandas as pd
    eid = int(record["episode_id"])
    frames = ep["frames"].tolist()
    raw = pd.read_parquet(loader.dataset_path / loader.data_path_pattern.format(
        episode_chunk=eid // loader.chunk_size, episode_index=eid))
    table = loader._load_parquet_data(eid)
    language = policy.language_key
    if language == "task":
        if len(record["metadata"]["tasks"]) != 1:
            raise ValueError("Ambiguous instruction")
        table["language.task"] = record["metadata"]["tasks"][0]
    video = loader._load_video_data(eid, np.array(frames))
    head, core = policy.model.action_head, policy.representation
    original_process, original_step = head.process_backbone_output, core.step
    captures, native = [], {}

    def process(*args, **kwargs):
        output = original_process(*args, **kwargs)
        if cache_quantization:
            # Explicit DIAGNOSTIC condition, not a change to the installed policy.
            # Cache.py stores these features in BF16 before both READ and AE.
            output["backbone_features"] = output["backbone_features"].to(torch.bfloat16)
        native["features"] = cpu(output["backbone_features"][0])
        native["attention_masks"] = cpu(output["backbone_attention_mask"][0])
        native["image_masks"] = cpu(output["image_mask"][0])
        return output

    def step(short, moment, state, frames_, demo, **kwargs):
        if cache_quantization:
            moment = moment.to(torch.bfloat16).float()
        before = kwargs.get("bank")
        before = core.initial_bank() if before is None else before
        result = original_step(short, moment, state, frames_, demo, **kwargs)
        captures.append({"moment": cpu(moment), "short": cpu(short), "state": cpu(state),
                         "encoded": cpu(result["encoded_current"]), "bank_before": cpu(before),
                         "bank_after": cpu(result["bank"]), "fused": cpu(result["fused"]), **native})
        return result

    def no_denoising(features, *args, **kwargs):
        return {"action_pred": features.new_zeros(1, head.action_horizon, head.action_dim)}

    modalities = policy.modality_configs
    sid = f"diagnostic-{eid}"
    # Same session ID intentionally reset for each pass; stale bank would fail comparisons.
    with patch.object(head, "process_backbone_output", process), patch.object(core, "step", step), \
            patch.object(head, "get_action_with_features", no_denoising):
        for d, frame in enumerate(frames):
            observation = {
                "video": {key: video[key][d][None, None] for key in modalities["video"].modality_keys},
                "state": {key: np.asarray(table[f"state.{key}"].iloc[frame], np.float32)[None, None]
                          for key in modalities["state"].modality_keys},
                "language": {language: [[str(table[f"language.{language}"].iloc[frame])]]},
            }
            passive = bool(raw["is_demo"].iloc[frame])
            controls = np.empty((0, 8), np.float32)
            if d and not bool(ep["is_demo"][d-1]):
                controls = np.concatenate([
                    np.stack(table[f"action.{key}"].iloc[frames[d-1]:frame].to_list())
                    for key in modalities["action"].modality_keys], axis=-1).astype(np.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
                policy._get_action(observation, {"session_ids": [sid], "reset_memory": [d == 0],
                    "episode_seed": 190020, "frame_index": frame, "passive": passive,
                    "prime_only": passive, "executed_actions": controls})
            if d % 16 == 0:
                print(f"[parity] {repeat_name} episode={eid} endpoint={d}/{len(frames)}", flush=True)
    if len(captures) != len(frames):
        raise AssertionError("Lost policy observations")
    return captures


def branch_episode(ep, captures):
    result = dict(ep)
    for key in ("features", "attention_masks", "image_masks"):
        result[key] = [c[key] for c in captures]
    for key in ("short", "moment", "state"):
        result[key] = torch.cat([c[key] for c in captures])
    # Preserve native feature-tail dtype, independently of the FP32 encoder input.
    result["short"] = torch.stack([c["features"][-ep["short"].shape[1]:] for c in captures])
    return result


@torch.no_grad()
def action_metrics(head, ep, d, fused, row):
    from run_scripts.robomme.flow_objective_v19 import episode_flow_v19
    from run_scripts.robomme.deployment_objective_v9 import generated_prefix_objective
    from run_scripts.robomme.validation_v19 import generated_metrics_v19
    flow = episode_flow_v19(head, ep, d, fused, seed=row["flow_seed"], tail_weight=1.,
                            action_steps=16, activation_checkpointing=False)
    generated = generated_prefix_objective(head, ep, d, fused, seed=row["generation_seed"],
                                           action_steps=16, activation_checkpointing=False)
    return {"action_loss": float(flow["original_flow_loss"]),
            "executed_prefix_flow_loss": float(flow["executed_prefix_flow_loss"]),
            **generated_metrics_v19(generated, ep, d)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default="VideoPlaceOrder")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 4:
        parser.error("Bounded diagnostic requires 1..4 episodes")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new diagnostic output; preserving {output}")
    cache = EpisodeCache(args.cache_dir)
    validate_output_scope(output, cache.path, Path(args.checkpoint).resolve().parent,
                          cache.manifest["dataset_path"], cache.manifest["model_path"])
    episodes = MappedEpisodes(cache)
    validation = json.loads(Path(args.validation).read_text())
    selection = selected_rows(validation, args.task, args.episodes)
    if any(r["episode_id"] not in cache.manifest["splits"]["val"] for r in selection):
        raise ValueError("Diagnostic must use existing cache-VAL, never TRAIN/TEST")
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_meta = json.loads((checkpoint / "checkpoint.json").read_text())
    if (checkpoint_meta["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]
            or checkpoint_meta["step"] != validation["step"]
            or Path(args.validation).resolve().parent != checkpoint.parent):
        raise ValueError("Validation report, checkpoint run/step and cache provenance must match")
    files = [checkpoint / f for f in ("checkpoint.json", "model.safetensors", "expert.safetensors")]
    files += [Path(args.validation).resolve(), cache.path / "manifest.json"]
    before = {str(p): sha(p) for p in files}
    meta = {"args": vars(args), "source_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            ROOT / "run_scripts/robomme/diagnose_v19_memory.py", ROOT / "gr00t/long_memory/cache.py",
            ROOT / "run_scripts/robomme/policy_representation_v18.py"]},
        "cache_fingerprint": cache.manifest["fingerprint"], "input_sha256": before,
        "checkpoint_metadata": checkpoint_meta,
        "base_model": cache.manifest["model_path"], "dataset": cache.manifest["dataset_path"],
        "selection": [{k: r[k] for k in ("task", "episode_id", "decision", "repeat", "flow_seed", "generation_seed")}
                      for r in selection], "gpu": gpu_snapshot(args.device),
        "limits": ["Frozen existing V19 model; no retraining or simulator rollout.",
            "Same teacher observation trajectory, not closed-loop policy states.",
            "Denoising stubbed only during parity extraction; real unchanged expert used for error metrics.",
            "Small cache-VAL selection and noise replicates are not independent episodes.",
            "FIFO source provenance does not establish semantic information preservation."]}
    output.mkdir(parents=True)
    write_json(output / "configuration.json", meta)
    start = time.monotonic()
    from run_scripts.robomme.policy_representation_v18 import RepresentationPolicyV18
    print("[diagnostic] loading existing checkpoint; all policy parameters frozen", flush=True)
    policy = RepresentationPolicyV18(cache.manifest["model_path"], str(checkpoint), device=args.device)
    head, core = policy.model.action_head, policy.representation
    if any(p.requires_grad for p in policy.model.parameters()) or any(p.requires_grad for p in core.parameters()):
        raise AssertionError("Policy must be frozen")
    if not args.skip_parity:
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        dataset = Path(cache.manifest["dataset_path"])
        _, records = _episode_records(dataset)
        eid = int(selection[0]["episode_id"])
        record = next(r for r in records if r["episode_id"] == eid)
        ep = episodes.fetch(eid)
        loader = LeRobotEpisodeLoader(dataset, policy.processor.modality_configs["new_embodiment"], video_backend="opencv")
        config = CacheConfig(model_path=cache.manifest["model_path"], dataset_path=str(dataset),
                             output_dir="UNUSED_NO_CACHE_WRITE", device=args.device)
        print(f"[parity] recomputing actual cache producer episode={eid}", flush=True)
        recache = _extract_episode(policy.model, policy.processor, loader, record, config)
        if not torch.equal(recache["frames"], ep["frames"]):
            raise AssertionError("Canonical source frame mismatch")
        stages, recurrence = cached_stages(core, ep)
        restages, _ = cached_stages(core, recache)
        write_csv(output / "recurrence.csv", recurrence)
        comparisons, branches, live = [], {"saved-cache": (ep, stages), "recache-amp": (recache, restages)}, {}
        for name, amp, quantize in (("live-default", False, False), ("live-amp", True, False),
                ("live-amp-cache-quantized", True, True), ("live-default-repeat", False, False)):
            captures = live_extract(policy, loader, record, ep, amp=amp, repeat_name=name, cache_quantization=quantize)
            branches[name] = (branch_episode(ep, captures), captures)
            live[name] = captures
        for name, (branch, current) in branches.items():
            if name == "saved-cache":
                continue
            refstages = live["live-default"] if name == "live-default-repeat" else stages
            refep = branches["live-default"][0] if name == "live-default-repeat" else ep
            for d, frame in enumerate(ep["frames"]):
                for key in ("moment", "short", "state", "encoded", "bank_before", "bank_after", "fused"):
                    ref, cur = refstages[d][key], current[d][key]
                    comparisons.append({"branch": name, "episode_id": eid, "decision": d, "frame": int(frame),
                                        "stage": key, **numerical_comparison(ref, cur)})
                    if key == "moment":
                        comparisons.append({"branch": name, "episode_id": eid, "decision": d, "frame": int(frame),
                            "stage": "moment_both_quantized_bf16", **numerical_comparison(ref.to(torch.bfloat16), cur.to(torch.bfloat16))})
                comparisons.append({"branch": name, "episode_id": eid, "decision": d, "frame": int(frame),
                    "stage": "all_backbone_features", **numerical_comparison(refep["features"][d], branch["features"][d])})
        write_csv(output / "parity.csv", comparisons)
        metrics = []
        # Prespecified validation query plus first active query; same seeds and masks across branches.
        first = int(ep["decision_mask"].nonzero()[0])
        for row in [r for r in selection if r["episode_id"] == eid]:
            for d in sorted({first, int(row["decision"])}):
                for name, (branch, current) in branches.items():
                    fused = current[d]["fused"].to(args.device)
                    metrics.append({"branch": name, "episode_id": eid, "decision": d, "repeat": row["repeat"],
                        **action_metrics(head, branch, d, fused, row)})
        write_csv(output / "parity_action_errors.csv", metrics)
        import pandas as pd
        from gr00t.eval.sim.robomme.run_long_memory_rollout import demo_endpoints
        raw_path = loader.dataset_path / loader.data_path_pattern.format(
            episode_chunk=eid // loader.chunk_size, episode_index=eid)
        raw = pd.read_parquet(raw_path, columns=["is_demo"])
        n_demo = int(raw["is_demo"].sum())
        expected_demo = demo_endpoints(n_demo, policy.stride)
        actual_demo = ep["frames"][ep["is_demo"]].tolist()
        if expected_demo != actual_demo:
            raise AssertionError("Actual rollout client's demo cadence differs from cache")
        write_json(output / "parity_summary.json", {"episode_id": eid, "endpoints": len(ep["frames"]),
            "frames": ep["frames"].tolist(), "demo_endpoints": ep["frames"][ep["is_demo"]].tolist(),
            "raw_parquet": str(raw_path), "raw_parquet_sha256": sha(raw_path),
            "online_client_demo_endpoints": expected_demo, "online_cache_demo_cadence_match": True,
            "demo_raw_frame_count": n_demo,
            "unobserved_final_demo_frames": list(range(max(actual_demo)+1, n_demo)) if actual_demo else [],
            "execution_cadence_match_except_terminal_endpoint": all(
                (int(f)-n_demo) % policy.stride == 0 for f in ep["frames"][:-1][~ep["is_demo"][:-1]]),
            "terminal_endpoint_note": "Cache retains a final endpoint even if closed-loop rollout would already terminate; no action loss evaluated there.",
            "actual_policy_reset": "same session ID; reset_memory=True at first endpoint of every pass",
            "recurrence_exact_count": sum(r["exact"] for r in recurrence), "recurrence_rows": len(recurrence)})
        del branches, live, stages, restages, recache
        policy.sessions.clear()
    all_rows, provenance = [], {}
    for i, row in enumerate(selection):
        eid, d = int(row["episode_id"]), int(row["decision"])
        ep = episodes.fetch(eid)
        print(f"[interventions] {i+1}/{len(selection)} episode={eid} decision={d} repeat={row['repeat']}", flush=True)
        result = interventions(head, core, ep, d, seed=row["flow_seed"], generation_seed=row["generation_seed"])
        for item in result["records"]:
            item.update(task=args.task, repeat=row["repeat"])
        all_rows.extend(result["records"])
        provenance[f"{eid}:{d}"] = result["provenance"]
        # Incremental fresh-directory journals survive an interrupted diagnostic.
        write_json(output / "interventions.json", all_rows)
        write_json(output / "bank_provenance.json", provenance)
    write_csv(output / "interventions.csv", all_rows)
    after = {str(p): sha(p) for p in files}
    if before != after:
        raise AssertionError("An input artifact changed during diagnosis")
    write_json(output / "completed.json", {"status": "complete", "elapsed_seconds": time.monotonic()-start,
        "input_sha256_unchanged": before == after, "intervention_records": len(all_rows),
        "trainable_policy_parameters": 0, "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "parity_executed": not args.skip_parity})
    print(f"[diagnostic] COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
