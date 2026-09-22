#!/usr/bin/env python3
"""Qualify a fixed-budget CVoM teacher, without training or robot rollouts.

A full FIFO bank and its newly observed event form B+1 alternatives. Every
operation drops exactly one whole event. KEEP drops the candidate; FIFO drops
the oldest. Competing operations share query, GT, masks, noise and timestep.
Selection A/B differ only in noise. Held-out queries are later, and their GT
chunks do not overlap selection chunks. Selecting a winner on held-out data
would be an optimistic oracle, NOT the primary result.

This deliberately reuses the existing ECHO actor, V19 flow bridge and deployed
Euler sampler. It does not install another writer or change any policy path.
The bank is held fixed after the intervention: conditional predictive utility,
not a counterfactual environmental return or future FIFO continuation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

VERSION = "echo_fixed_budget_teacher_probe_v1"
DEFAULT_CHECKPOINT = "runs/long_memory/echo_cvom_full_v1/refresh/checkpoint-001000"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def seed_for(seed, *parts):
    # Operation/victim identity must NEVER occur in this seed derivation.
    return int(digest([VERSION, seed, *parts])[:15], 16)


def future_partition(ep, event, short_window, future_samples, *, seed):
    """Use chronology/masks only, never losses, action values or annotations.

    Also separate *whole 50-step GT chunks*, not just 16-step prefixes, across
    selection and held-out panels. Queries within a panel may be correlated;
    confidence intervals therefore resample episodes, not individual draws.
    """
    queries = [int(q) for q in torch.where(ep["decision_mask"])[0].tolist()
               if q > event + short_window]
    n, frames, horizon = future_samples, ep["frames"], ep["targets"].shape[1]
    if len(queries) < 2*n:
        return None
    cuts = [i for i in range(n-1, len(queries)-n)
            if sum(int(frames[q]) >= int(frames[queries[i]]) + horizon
                   for q in queries[i+1:]) >= n]
    if not cuts:
        return None
    # Median feasible temporal split is fixed before observing actor outcomes.
    cut = cuts[len(cuts)//2]
    rng = random.Random(seed)
    selected = sorted(rng.sample(queries[:cut+1], n))
    later = [q for q in queries[cut+1:]
             if int(frames[q]) >= int(frames[queries[cut]]) + horizon]
    heldout = sorted(rng.sample(later, n))
    return {"selection_queries": selected, "heldout_queries": heldout,
            "target_horizon": int(horizon),
            "selection_to_heldout_frame_gap": int(frames[heldout[0]]-frames[selected[-1]])}


def build_plan(cache, episodes, tasks, cfg, *, split, count, future_samples, seed, task_filter=None):
    """One predetermined full-bank context per episode; TRAIN/cache-VAL only."""
    if split not in ("train", "val") or count < 1 or future_samples < 1:
        raise ValueError("Need TRAIN/cache-VAL and positive episode/future budgets")
    splits = {s: list(map(int, cache.manifest["splits"][s])) for s in ("train", "val")}
    if any(len(v) != len(set(v)) for v in splits.values()) or set(splits["train"]) & set(splits["val"]):
        raise ValueError("Cache episode splits overlap or contain duplicates")
    grouped, available = {}, Counter()
    for eid in sorted(splits[split]):
        task = tasks[eid]
        if task_filter and task not in task_filter:
            continue
        ep = episodes.fetch(eid)
        # Feasibility is monotone with event index. Inspect only the earliest
        # future endpoints needed, avoiding repeated construction of all plans.
        eligible = []
        for event in range(cfg.capacity_events, len(ep["decision_mask"])):
            panel = future_partition(ep, event, cfg.short_window, future_samples,
                                     seed=seed_for(seed, split, eid, event, "future"))
            if panel is None:
                break
            eligible.append((event, panel))
        if not eligible:
            continue
        event, panel = random.Random(seed_for(seed, split, eid, "event")).choice(eligible)
        row = {"episode_id": eid, "event": event, "task": task, "split": split,
               "endpoint_frame": int(ep["frames"][event]),
               "is_demo": bool(ep["is_demo"][event]), **panel}
        grouped.setdefault(task, []).append(row)
        available[task] += 1
    if not grouped:
        raise ValueError("No full FIFO contexts with disjoint later GT panels")
    for task, rows in grouped.items():
        random.Random(seed_for(seed, split, task, "episodes")).shuffle(rows)
    selected = []
    while len(selected) < count and any(grouped.values()):
        for task in sorted(grouped):
            if grouped[task] and len(selected) < count:
                selected.append(grouped[task].pop())
    return selected, dict(sorted(available.items()))


def drop_operations(pool, capacity):
    """Equal budget, chronological content and time kept together, no merging."""
    if pool.n_events != capacity + 1:
        raise ValueError("Exactly one candidate must overflow a full bank")
    operations, banks = [], []
    for index in [capacity, *range(capacity)]:
        operation = "keep" if index == capacity else "fifo" if index == 0 else f"drop-{index:02d}"
        operations.append({"id": operation, "kind": operation if index in (0, capacity) else "drop",
                           "drop_event_id": pool.event_ids[index][-1], "drop_pool_index": index})
        banks.append(pool.without([index]))
    if any(bank.n_events != capacity for bank in banks):
        raise AssertionError("Unequal operation budgets")
    return operations, banks


def rng_state():
    return [torch.get_rng_state().clone(), *[s.clone() for s in
            (torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])]]


@torch.no_grad()
def measure_context(core, head, ep, row, *, seed, noise_samples=2, generation_repeats=1,
                    progress=None, flow_fn=None, generation_fn=None):
    """Measure all operations with matched inputs; hooks exist for CPU tests."""
    for name, module in (("core", core), ("head", head)):
        if module.training or any(p.requires_grad for p in module.parameters()):
            raise ValueError(f"{name} must be frozen and eval()")
    if noise_samples < 2 or generation_repeats < 0:
        raise ValueError("Need >=2 noise repeats, >=0 generation repeats")
    if core.echo_config.merge_threshold is not None:
        raise ValueError("This drop-only diagnosis cannot silently disable an enabled merge policy")
    real_actor = flow_fn is None
    if real_actor:
        from gr00t.long_memory.cache_reader_v3 import validate_decision
        from run_scripts.robomme.flow_objective_v19 import episode_flow_v19
        from run_scripts.robomme.deployment_objective_v9 import generated_prefix_objective
        from run_scripts.robomme.validation_v19 import generated_metrics_v19
        flow_fn = episode_flow_v19

        def generation_fn(head, episode, query, fused, **kwargs):
            return generated_metrics_v19(generated_prefix_objective(head, episode, query, fused, **kwargs),
                                         episode, query, 16)
    event, cfg = row["event"], core.config
    selection, heldout = row["selection_queries"], row["heldout_queries"]
    if (row["split"] not in ("train", "val") or event < cfg.capacity_events or not selection or not heldout
            or selection != sorted(set(selection)) or heldout != sorted(set(heldout))
            or min(selection) <= event + cfg.short_window or min(heldout) <= max(selection)
            or int(ep["frames"][heldout[0]]-ep["frames"][selection[-1]]) < ep["targets"].shape[1]):
        raise ValueError("Require full-bank, causal, disjoint selection/held-out GT chunks")
    for query in selection + heldout:
        if not bool(ep["decision_mask"][query]):
            raise ValueError("Selected passive/non-decision query")
        if real_actor:
            validate_decision(ep, query)
    before_rng = rng_state()
    # Storage prefix encoding cannot depend on how many future queries we use.
    encoded = core.encode_prefix(ep, event+1)
    bank = core.initial_state()
    for index in range(event):
        bank, _ = core.manager.update(bank, encoded["stored"][index:index+1],
            encoded["query"][index:index+1], int(ep["frames"][index]), bool(ep["is_demo"][index]),
            index, mode="fifo")
    if bank.n_events != cfg.capacity_events:
        raise AssertionError("FIFO prefix did not fill")
    candidate, query = encoded["stored"][event:event+1], encoded["query"][event:event+1]
    frame, demo = int(ep["frames"][event]), bool(ep["is_demo"][event])
    pool = bank.append(candidate, frame, demo, event)
    operations, banks = drop_operations(pool, cfg.capacity_events)
    scores = core.manager.score(bank, candidate, query, frame, demo)
    learned, writer_metrics = core.manager.update(bank, candidate, query, frame, demo, event, mode="learned")
    matches = [o["id"] for o, b in zip(operations, banks) if b.event_ids == learned.event_ids]
    if len(matches) != 1:
        raise AssertionError("Learned manager result is not one of the tested operations")
    for b in banks:
        if any(eid > event for ids in b.event_ids for eid in ids):
            raise AssertionError("Future event entered a tested bank")
    future_encoded = core.encode_prefix(ep, max(heldout)+1)
    metrics = {"flow_prefix": "executed_prefix_flow_loss", "flow_unweighted": "original_flow_loss",
               "flow_weighted": "loss", "flow_joint": "executed_prefix_joint_flow_loss",
               "flow_gripper": "executed_prefix_gripper_flow_loss"}
    gen_metrics = {"generated_mse": "generated_prefix_mse", "generated_joint": "generated_executed_joint_mse",
                   "generated_gripper": "generated_executed_gripper_mse"}
    result = {**row, "context_id": f"{row['split']}-{row['episode_id']}-{event}",
              "operations": operations, "manager_operation_id": matches[0],
              "manager_scores": {k: scores[k].detach().cpu().tolist() for k in
                                 ("utility", "retention", "write_probability")},
              "manager_metrics": {k: float(v) for k, v in writer_metrics.items()},
              "bank_event_ids": [list(ids) for ids in bank.event_ids],
              "pool_provenance": [{"event_ids": list(ids), "first_frame": first, "last_frame": last,
                                   "is_demo": demo} for ids, first, last, demo in
                                  zip(pool.event_ids, pool.first_frames, pool.last_frames, pool.is_demo)],
              "panels": {}, "regression": {}, "calls": {"flow": 0, "generation": 0}}
    cached_fused = {}
    for panel_name, queries in (("selection_a", selection), ("selection_b", selection), ("heldout", heldout)):
        panel = {key: [] for key in metrics}
        panel.update(draw_ids=[], query_ids=list(queries), draws=[])
        if panel_name == "heldout" and generation_repeats:
            panel.update({key: [] for key in gen_metrics}, generation_draws=[])
        for q in queries:
            # The current short/query is identical across every intervention.
            if q not in cached_fused:
                short, eq = future_encoded["short"][q:q+1], future_encoded["query"][q:q+1]
                cached_fused[q] = [core.read_from_bank(short, eq, b.tokens)[0] for b in banks]
            fused = cached_fused[q]
            for noise in range(noise_samples):
                s = seed_for(seed, row["episode_id"], event, panel_name, q, noise, "flow")
                values = [flow_fn(head, ep, q, f, seed=s, tail_weight=.25,
                                 activation_checkpointing=False) for f in fused]
                result["calls"]["flow"] += len(values)
                for key, name in metrics.items():
                    panel[key].append([float(v[name]) for v in values])
                panel["draw_ids"].append(f"{q}:{s}")
                panel["draws"].append({"query": q, "endpoint_frame": int(ep["frames"][q]), "seed": s})
                if panel_name == "selection_a" and q == selection[0] and noise == 0:
                    again = flow_fn(head, ep, q, fused[0], seed=s, tail_weight=.25,
                                    activation_checkpointing=False)
                    error = max(abs(float(values[0][name])-float(again[name])) for name in metrics.values())
                    result["regression"]["same_path_flow_max_abs"] = error
                    result["calls"]["flow"] += 1
                    if error != 0:
                        raise AssertionError(f"Same-input flow not exactly reproducible: {error}")
            if panel_name == "heldout":
                for repeat in range(generation_repeats):
                    s = seed_for(seed, row["episode_id"], event, q, repeat, "generation")
                    values = [generation_fn(head, ep, q, f, seed=s, action_steps=16,
                                           activation_checkpointing=False) for f in fused]
                    result["calls"]["generation"] += len(values)
                    for key, name in gen_metrics.items():
                        panel[key].append([float(v[name]) for v in values])
                    panel["generation_draws"].append({"query": q, "seed": s})
                    if q == heldout[0] and repeat == 0:
                        again = generation_fn(head, ep, q, fused[0], seed=s, action_steps=16,
                                              activation_checkpointing=False)
                        error = max(abs(float(values[0][name])-float(again[name])) for name in gen_metrics.values())
                        result["regression"]["same_path_generated_max_abs"] = error
                        result["calls"]["generation"] += 1
                        if error != 0:
                            raise AssertionError(f"Same-input generated metrics not reproducible: {error}")
            if progress:
                progress(panel_name, q, result["calls"])
        for key in (*metrics, *gen_metrics):
            if key in panel and any(not math.isfinite(v) or v < 0 for draw in panel[key] for v in draw):
                raise FloatingPointError(f"Invalid {key} diagnostic loss")
        result["panels"][panel_name] = panel
    if any(not torch.equal(a, b) for a, b in zip(before_rng, rng_state(), strict=True)):
        raise AssertionError("Diagnostic altered global Torch RNG")
    result["regression"]["global_rng_unchanged"] = True
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preflight", "run", "report"))
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--cache-dir", default="runs/long_memory/cache_full1600_v1")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--contexts", type=int, default=16)
    p.add_argument("--tasks", nargs="+", help="Optional predetermined task names; omit for task-balanced coverage")
    p.add_argument("--future-samples", type=int, default=2, help="Separate query counts in selection and held-out panels")
    p.add_argument("--noise-samples", type=int, default=2)
    p.add_argument("--generation-repeats", type=int, default=1, help="0 is cheaper flow-only screening, not action generation")
    p.add_argument("--seed", type=int, default=260922)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--stop-after-contexts", type=int, default=0, help="Pause, do not change the predeclared plan")
    p.add_argument("--resume", action="store_true", help="Strict same-plan/source/environment continuation only")
    return p


def environment(device):
    import importlib.metadata
    result = {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
              "cuda_build": torch.version.cuda, "device": device,
              "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
              "packages": {}}
    for name in ("numpy", "transformers", "diffusers", "safetensors"):
        result["packages"][name] = importlib.metadata.version(name)
    if torch.device(device).type == "cuda":
        result["gpu_hardware"] = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
    return result


def prepare(args):
    from gr00t.long_memory.cache import EpisodeCache
    from gr00t.long_memory.cache_reader_v3 import MappedEpisodes
    from gr00t.long_memory.hamlet import validate_cache_checkpoint
    from gr00t.long_memory.safety_v5 import validate_output_scope
    from run_scripts.robomme.echo_cvom_checkpoint import inspect_checkpoint
    from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18
    from run_scripts.robomme.training_plan_v19 import resolve_tasks_v19
    from run_scripts.robomme.train_echo_cvom import episode_signatures, sources
    from run_scripts.robomme.train_cvom_admission import file_hash
    if min(args.contexts, args.future_samples, args.cpu_threads) < 1 or args.noise_samples < 2:
        raise ValueError("Positive budgets, at least two noise draws required")
    if min(args.seed, args.generation_repeats, args.stop_after_contexts) < 0:
        raise ValueError("Seed and optional budgets must be nonnegative")
    cache = EpisodeCache(args.cache_dir)
    validate_cache_checkpoint(cache.manifest)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    info = inspect_checkpoint(Path(cache.manifest["model_path"]), checkpoint)
    if info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]:
        raise ValueError("Checkpoint/cache mismatch")
    if info["config"]["echo"]["merge_threshold"] is not None:
        raise ValueError("Probe only supports the existing merge-OFF model")
    cfg = RepresentationConfigV18(**info["config"]["representation"])
    tasks, task_source = resolve_tasks_v19(cache.manifest)
    if args.tasks and (set(args.tasks)-set(tasks.values()) or len(args.tasks) != len(set(args.tasks))):
        raise ValueError("Unknown/duplicate task selection")
    output = Path(args.output_dir).resolve()
    validate_output_scope(output, cache.path, checkpoint, cache.manifest["model_path"], cache.manifest["dataset_path"])
    episodes = MappedEpisodes(cache, max_cached=2)
    plan, availability = build_plan(cache, episodes, tasks, cfg, split=args.split, count=args.contexts,
        future_samples=args.future_samples, seed=args.seed, task_filter=args.tasks)
    source = sources()
    for name in ("cvom_budget_probe.py", "cvom_budget_statistics.py", "audit_archive_generation_v7.py"):
        source[f"run_scripts/robomme/{name}"] = file_hash(ROOT/"run_scripts/robomme"/name)
    settings = {k: getattr(args, k) for k in ("split", "contexts", "tasks", "future_samples", "noise_samples",
        "generation_repeats", "seed", "device", "cpu_threads")}
    protocol = {"version": VERSION, "checkpoint": {"path": str(checkpoint), "files_sha256": info["files_sha256"]},
        "cache_fingerprint": cache.manifest["fingerprint"],
        "cache_manifest_sha256": file_hash(Path(cache.path)/"manifest.json"),
        "episode_file_stat_signatures": episode_signatures(cache, [r["episode_id"] for r in plan]),
        "source_sha256": source, "environment": environment(args.device), "settings": settings,
        "plan": plan, "available_episodes_by_task": availability, "task_resolution": task_source,
        "capacity": cfg.capacity_events, "actor_frozen": True, "collection": "FIFO prefix; fixed full-bank snapshot",
        "selection_metric": "executed-prefix flow MSE", "selection_a_b": "same queries; independent common noise panels",
        "heldout": "later disjoint GT chunks; not independent episodes",
        "feature_precision": "existing cache tensors and unchanged frozen flow/Euler bridges; NOT online native replay",
        "future_writes": False, "merge": False, "rollout_success": False,
        "scope": "conditional one-operation teacher reliability; no policy or writer training"}
    protocol["fingerprint"] = digest(protocol)
    ops, n = cfg.capacity_events+1, len(plan)
    print(f"[budget-preflight] {args.split}: {n} episodes, {len(set(r['task'] for r in plan))} tasks; "
          f"{ops} operations/context; flow calls={n*(ops*3*args.future_samples*args.noise_samples+1)}, "
          f"generation calls={n*(ops*args.future_samples*args.generation_repeats+bool(args.generation_repeats))}; "
          "no training/rollout", flush=True)
    print(f"[budget-preflight] available full-bank episodes by task: {availability}", flush=True)
    return cache, episodes, info, protocol, output


def read_packets(output, protocol):
    expected = {f"context-{i:04d}.json": row for i, row in enumerate(protocol["plan"])}
    packets = []
    for path in sorted(output.glob("context-*.json")):
        if path.name not in expected:
            raise ValueError(f"Unexpected diagnostic context file {path}")
        packet = json.loads(path.read_text())
        if (packet.get("protocol_fingerprint") != protocol["fingerprint"]
                or packet.get("row_sha256") != digest(expected[path.name])
                or packet.get("result_sha256") != digest(packet["result"])):
            raise ValueError(f"Context identity/content changed: {path}")
        if any(packet["result"][k] != v for k, v in expected[path.name].items()):
            raise ValueError("Result context differs from predeclared plan")
        packets.append(packet)
    return packets


def matching_verification(output, protocol, packets):
    """An unfinished/failed final audit cannot become qualified on resume."""
    expected = {p["row_sha256"]: p["result_sha256"] for p in packets}
    for path in sorted(output.glob("verification-*.json"), reverse=True):
        audit = json.loads(path.read_text())
        if (audit.get("protocol_fingerprint") == protocol["fingerprint"]
                and audit.get("verified_packets") == expected
                and audit.get("checkpoint_unchanged") is True
                and audit.get("cache_stat_signatures_unchanged") is True
                and audit.get("source_unchanged") is True
                and all(p.get("actor_state") == audit.get("actor_unchanged") for p in packets)):
            return str(path)
    return None


def report(output):
    from gr00t.long_memory.monitoring import _atomic_json
    from run_scripts.robomme.cvom_budget_statistics import summarize_budget_contexts
    protocol = json.loads((output/"protocol.json").read_text())
    if digest({k: v for k, v in protocol.items() if k != "fingerprint"}) != protocol["fingerprint"]:
        raise ValueError("Modified diagnostic protocol")
    from run_scripts.robomme.train_cvom_admission import file_hash
    for name in ("cvom_budget_probe.py", "cvom_budget_statistics.py"):
        key = f"run_scripts/robomme/{name}"
        if file_hash(ROOT/key) != protocol["source_sha256"].get(key):
            raise ValueError("Report-analysis source changed; cannot silently reinterpret the saved protocol")
    packets = read_packets(output, protocol)
    if not packets:
        raise ValueError("No completed contexts to summarize")
    verification = matching_verification(output, protocol, packets)
    if not verification:
        raise ValueError("No completed invariance audit covers these results. Resume the original run before reporting.")
    summary = summarize_budget_contexts([p["result"] for p in packets], seed=protocol["settings"]["seed"])
    summary.update(protocol_fingerprint=protocol["fingerprint"],
                   completed_contexts=len(packets), planned_contexts=len(protocol["plan"]),
                   complete=len(packets) == len(protocol["plan"]), rollout_success=False,
                   verification=verification)
    _atomic_json(output/"summary.json", summary)
    rows = summary["per_context"]
    # JSON fields inside CSV preserve structured details without silently
    # flattening away which metric/panel the value came from.
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (output/"contexts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()}
                         for r in rows)
    message = ("# Fixed-budget CVoM teacher diagnosis\n\n"
        f"Completed contexts: {len(packets)}/{len(protocol['plan'])}. NOT rollout success.\n\n"
        "Positive gain means reference error minus selected-operation error.\n"
        "Selection uses panel A only; held-out generation never selects a winner.\n"
        "Memory is fixed after t; future observations/actions are teacher-only.\n"
        "This is an offline diagnostic, not proof of success gain or a new learned writer.\n\n"
        "## Reliability evidence (manual interpretation required)\n\n```json\n" +
        json.dumps({k: summary[k] for k in ("stability", "qualification_evidence", "overall")}, indent=2) +
        "\n```\n\nSee summary.json, contexts.csv and immutable context-*.json for paired draws and provenance.\n")
    (output/"report.md").write_text(message)
    print(f"[budget] report: {output/'report.md'}; {len(packets)}/{len(protocol['plan'])}; NOT rollout success", flush=True)
    return summary


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "report":
        report(Path(args.output_dir).resolve(strict=True))
        return 0
    torch.set_num_threads(args.cpu_threads)
    cache, episodes, info, protocol, output = prepare(args)
    if args.command == "preflight":
        print("[budget-preflight] read-only: no model/GPU allocation or output created", flush=True)
        return 0
    from gr00t.long_memory.monitoring import _atomic_json
    from run_scripts.robomme.train_cvom_admission import gpu_guard, state_hash
    from run_scripts.robomme.train_echo_cvom import load_actor, episode_signatures
    from run_scripts.robomme.echo_cvom_checkpoint import inspect_checkpoint
    if output.exists():
        if not args.resume:
            raise FileExistsError("Use a NEW output or --resume with an identical frozen protocol")
        if json.loads((output/"protocol.json").read_text()) != protocol:
            raise ValueError("Resume requires identical checkpoint, cache signatures, sources, settings and environment")
    else:
        if args.resume:
            raise FileNotFoundError("Cannot resume a nonexistent diagnostic")
        output.mkdir(parents=True)
        _atomic_json(output/"protocol.json", protocol)
    existing = read_packets(output, protocol)
    if len(existing) == len(protocol["plan"]) and matching_verification(output, protocol, existing):
        report(output)
        return 0
    gpu = gpu_guard(args.device)
    _atomic_json(output/f"launch-{time.time_ns()}.json", {"argv": sys.argv, "gpu_check": gpu})
    parent = info["metadata"]["parent_identity"]["path"]
    core, head, _ = load_actor(parent, cache, args.device, snapshot=args.checkpoint, seed=args.seed)
    core.eval().requires_grad_(False)
    head.eval().requires_grad_(False)
    before = {"core": state_hash(core), "expert": state_hash(head)}
    if existing and any(p.get("actor_state") != before for p in existing):
        raise ValueError("Resumed in-memory actor differs from stored evidence")
    count, started = 0, time.monotonic()
    for i, row in enumerate(protocol["plan"]):
        path = output/f"context-{i:04d}.json"
        if path.exists():
            continue
        def progress(panel, q, calls):
            print(f"[budget] {i+1}/{len(protocol['plan'])} {row['task']} ep={row['episode_id']} "
                  f"{panel} query={q} flow={calls['flow']} generation={calls['generation']} "
                  f"elapsed={time.monotonic()-started:.1f}s", flush=True)
        with torch.no_grad(), torch.autocast(device_type=torch.device(args.device).type, enabled=False):
            result = measure_context(core, head, episodes.fetch(row["episode_id"]), row, seed=args.seed,
                noise_samples=args.noise_samples, generation_repeats=args.generation_repeats, progress=progress)
        _atomic_json(path, {"protocol_fingerprint": protocol["fingerprint"], "row_sha256": digest(row),
                           "actor_state": before, "result_sha256": digest(result), "result": result})
        count += 1
        if args.stop_after_contexts and count >= args.stop_after_contexts:
            break
    after = {"core": state_hash(core), "expert": state_hash(head)}
    if before != after or inspect_checkpoint(Path(cache.manifest["model_path"]), args.checkpoint) != info:
        raise AssertionError("Diagnostic changed actor/checkpoint")
    if episode_signatures(cache, [r["episode_id"] for r in protocol["plan"]]) != protocol["episode_file_stat_signatures"]:
        raise AssertionError("Selected cache payloads changed during diagnosis")
    from run_scripts.robomme.train_cvom_admission import file_hash
    if any(file_hash(ROOT/name) != value for name, value in protocol["source_sha256"].items()):
        raise AssertionError("Diagnostic source changed during execution")
    packets = read_packets(output, protocol)
    _atomic_json(output/f"verification-{time.time_ns()}.json", {"actor_unchanged": before,
        "checkpoint_unchanged": True, "cache_stat_signatures_unchanged": True, "new_contexts": count,
        "source_unchanged": True, "protocol_fingerprint": protocol["fingerprint"],
        "verified_packets": {p["row_sha256"]: p["result_sha256"] for p in packets},
        "elapsed_seconds": time.monotonic()-started, "num_inference_timesteps": head.num_inference_timesteps})
    report(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
