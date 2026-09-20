"""Frozen v4 diagnosis on existing episodic validation caches; never training.

Three separate questions are measured: dependence on old memory, stability of
writer action-loss labels, and cached-endpoint online/offline bank parity.
These teacher-forced action losses are NOT RoboMME task success rates. The
validation split was used for checkpoint selection, so this is development
diagnosis, not a new held-out benchmark claim.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

import torch

from .cache import EpisodeCache
from .cache_reader_v3 import MappedEpisodes, validate_decision
from .checkpoint_v4 import (file_sha256, load_checkpoint_v4, reader_state_sha256,
                            v4_checkpoint_info)
from .contextual_cvom import eligible_candidates
from .core_v3 import ActionValueMemory, MemoryV3Config
from .expert_v4 import (LoRAConfig, adapter_disabled, expert_episode_flow_loss,
                        expert_state_sha256, install_expert_lora)
from .hamlet import load_frozen_hamlet, validate_cache_checkpoint
from .monitoring import _atomic_json
from .online_v3 import OnlineActionValueBank
from .replay_v3 import encode_until, read_bank, replay_bank
from .train_v3 import _Tee, runtime_identity


ROOT = Path(__file__).resolve().parents[2]
VARIANT = "frozen-v4-diagnostic-v1"
CONDITIONS = ("baseline", "expert_only", "full", "no_old", "only_old", "shuffled_old", "fifo")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("runs/long_memory/cache_full1600_v1"))
    p.add_argument("--checkpoint", type=Path,
                   default=Path("runs/long_memory/v4_writer_stage2_full/checkpoint-000900"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--experiments", nargs="+", choices=("interventions", "writer", "parity"),
                   default=["interventions", "writer", "parity"])
    p.add_argument("--samples", type=int, default=64, help="At most one query per selected validation episode")
    p.add_argument("--episode-pool", type=int, default=128, help="Maximum validation episodes inspected for query/context plans")
    p.add_argument("--min-old-events", type=int, default=2,
                   help="Minimum AVAILABLE old history, not necessarily retained; targeted delayed panel (0=unrestricted)")
    p.add_argument("--noise-samples", type=int, default=4, help="Paired draws; writer uses this many PER independent A/B split")
    p.add_argument("--writer-contexts", type=int, default=12)
    p.add_argument("--writer-candidate-pool", type=int, default=128)
    p.add_argument("--writer-bank-source", choices=("deployed", "fifo-stress"), default="deployed",
                   help="fifo-stress is an explicitly counterfactual bank distribution, NOT natural writer coverage")
    p.add_argument("--all-victims", action="store_true", help="Expensive optional full replacement candidate-set audit")
    p.add_argument("--future-samples", type=int, default=2)
    p.add_argument("--tie-tolerance", type=float, default=1e-5, help="Raw action-loss units; report chosen tolerance")
    p.add_argument("--parity-samples", type=int, default=4)
    p.add_argument("--parity-atol", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true", help="Validate metadata/weights; no model/output/experiments")
    mode.add_argument("--report-only", action="store_true", help="Summarize saved journals, including partial runs")
    mode.add_argument("--resume", action="store_true", help="Resume identical provenance/arguments only")
    return p


def resolve(path):
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def validate_args(args):
    for name in ("samples", "episode_pool", "noise_samples", "writer_contexts",
                 "writer_candidate_pool", "future_samples", "parity_samples"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0 or args.min_old_events < 0 or args.noise_samples < 2:
        raise ValueError("Nonnegative seed/min-old-events and at least 2 paired noise samples required")
    for name in ("tie_tolerance", "parity_atol"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if len(set(args.experiments)) != len(args.experiments):
        raise ValueError("Duplicate experiments")


def preflight(args):
    validate_args(args)
    cache = EpisodeCache(resolve(args.cache_dir))
    validate_cache_checkpoint(cache.manifest)
    base = Path(cache.manifest["model_path"]).resolve()
    info = v4_checkpoint_info(base, resolve(args.checkpoint), expected_stage=2)
    if info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]:
        raise ValueError("Checkpoint and cache fingerprints differ")
    config = json.loads((base / "config.json").read_text())
    cfg = MemoryV3Config(**info["config"]["memory"])
    for name in ("feature_dim", "state_dim", "action_dim"):
        if getattr(cfg, name) != cache.manifest[name]:
            raise ValueError(f"Cache dimension mismatch: {name}")
    if cfg.time_scale != int(config["memory_stride"]):
        raise ValueError("Memory time scale differs from base stride")
    return cache, info, base, int(config["memory_window"]), int(config["memory_stride"])


def make_identity(args, cache, info, base):
    """Hash immutable inputs; selected episode payloads are hashed in the plan."""
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key not in ("output_dir", "preflight_only", "report_only", "resume")}
    checkpoint = resolve(args.checkpoint)
    sources = sorted((ROOT / "gr00t/long_memory").glob("*.py"))
    sources += sorted((ROOT / "gr00t/model").rglob("*.py"))
    sources += [ROOT / "run_scripts/robomme/diagnose_long_memory_v4.py"]
    return {"variant": VARIANT, "settings": settings, "torch": torch.__version__,
            "runtime": runtime_identity(), "cpu_threads": torch.get_num_threads(),
            "package_versions": {name: importlib.metadata.version(name) for name in ("transformers", "safetensors")},
            "cuda": torch.version.cuda, "base_model": str(base), "checkpoint": str(checkpoint),
            "checkpoint_step": info["step"], "cache": str(cache.path),
            "cache_fingerprint": cache.manifest["fingerprint"],
            "cache_manifest_sha256": file_sha256(cache.path / "manifest.json"),
            "checkpoint_sha256": {name: file_sha256(checkpoint / name) for name in
                                  ("checkpoint.json", "model.safetensors", "expert.safetensors")},
            "base_sha256": {str(p.relative_to(base)): file_sha256(p) for p in sorted(base.rglob("*"))
                            if p.is_file() and p.suffix in (".json", ".safetensors", ".model", ".txt")},
            "sources_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in sources}}


def _balanced_episodes(manifest, seed, limit):
    """Round-robin exact instruction groups, WITHOUT assuming task-name labels.

    RoboMME cache task strings are natural language and can encode instruction
    variants. This is an instruction-group-balanced development panel, not
    necessarily an equal-size 16-task sample or the benchmark macro statistic.
    """
    val = set(manifest["splits"]["val"])
    groups = defaultdict(list)
    for row in manifest["episodes"]:
        if row["episode_id"] in val:
            groups[row.get("task", "unknown")].append(int(row["episode_id"]))
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for ids in groups.values():
        rng.shuffle(ids)
    chosen = []
    while len(chosen) < limit and any(groups.values()):
        for key in keys:
            if groups[key]:
                chosen.append(groups[key].pop())
                if len(chosen) == limit:
                    break
    return chosen


def make_plan(args, cache, fetch, window):
    from .diagnostic_interventions import old_event_ids
    ids = _balanced_episodes(cache.manifest, args.seed, args.episode_pool)
    records = {int(row["episode_id"]): row for row in cache.manifest["episodes"]}
    queries, candidate_groups = [], []
    for eid in ids:
        ep = fetch(eid)
        rng = random.Random(args.seed * 1000003 + eid)
        valid = torch.where(ep["transition_valid"])[0].tolist()
        decisions = [d for d in torch.where(ep["decision_mask"])[0].tolist()
                     if len(old_event_ids(ep, d, [i for i in valid if i < d], window)) >= args.min_old_events]
        if decisions and len(queries) < args.samples:
            d = rng.choice(decisions)
            queries.append({"episode_id": eid, "decision": d, "frame": int(ep["frames"][d]),
                            "instruction_group": records[eid].get("task", "unknown")})
        if "writer" in args.experiments:
            candidates = eligible_candidates(ep, window)
            rng.shuffle(candidates)
            candidate_groups.append((eid, candidates))
        if len(queries) >= args.samples and "writer" not in args.experiments:
            break
    candidates = []
    while len(candidates) < args.writer_candidate_pool and any(rows for _, rows in candidate_groups):
        for eid, rows in candidate_groups:
            if rows:
                candidates.append({"episode_id": eid, "candidate": rows.pop()})
                if len(candidates) == args.writer_candidate_pool:
                    break
    if not queries and any(e in args.experiments for e in ("interventions", "parity")):
        raise ValueError("No eligible validation queries; lower --min-old-events or increase --episode-pool")
    used = sorted({row["episode_id"] for row in queries + candidates})
    return {"queries": queries, "writer_candidates": candidates,
            "sampling": "Fixed seed; at most one query/episode; round-robin exact instruction groups; eligible delayed queries uniform within episode",
            "episodes_inspected": len(ids), "available_val_episodes": len(cache.manifest["splits"]["val"]),
            "payload_sha256": {str(eid): file_sha256(cache.path / records[eid]["path"]) for eid in used}}


def _float(value):
    result = float(value.detach().item() if isinstance(value, torch.Tensor) else value)
    if not math.isfinite(result):
        raise FloatingPointError("Nonfinite diagnostic result")
    return result


def _flow(head, ep, decision, fused=None, *, seed):
    validate_decision(ep, decision)
    return expert_episode_flow_loss(head, ep, decision, fused, seed=seed)


@torch.no_grad()
def query_experiment(memory, head, ep, decision, *, window, noise_samples, seed, loss_fn=_flow):
    from .diagnostic_interventions import old_event_ids, read_intervention
    encoded = encode_until(memory, ep, decision)
    bank, stats = replay_bank(memory, ep, decision, "hard", encoded)
    fifo, fifo_stats = replay_bank(memory, ep, decision, "all", encoded)
    outputs, conditions = {}, {}
    for condition in CONDITIONS:
        if condition == "baseline":
            continue
        mode = "no_memory" if condition == "expert_only" else "full" if condition == "fifo" else condition
        result = read_intervention(memory, ep, decision, fifo if condition == "fifo" else bank,
                                   mode=mode, memory_window=window, encoded=encoded)
        outputs[condition] = result["fused_short"]
        conditions[condition] = {"diagnostic": result["diagnostic"], "draws": [],
                                 **{key: _float(result[key]) for key in
                                    ("gate_mean", "read_norm", "residual_norm", "null_weight") if key in result}}
    conditions["baseline"] = {"draws": [], "diagnostic": {"original_expert": True, "added_memory": False}}
    seeds = [seed + repeat * 1009 for repeat in range(noise_samples)]
    for paired in seeds:
        with adapter_disabled(head):
            value = loss_fn(head, ep, decision, seed=paired)
        conditions["baseline"]["draws"].append(_float(value["loss"]))
        for condition, fused in outputs.items():
            value = loss_fn(head, ep, decision, fused, seed=paired)
            conditions[condition]["draws"].append(_float(value["loss"]))
    for condition, result in conditions.items():
        result["action_loss"] = statistics.fmean(result["draws"])
        result["loss_minus_full"] = result["action_loss"] - statistics.fmean(conditions["full"]["draws"])
    past = [i for i in range(decision) if bool(ep["transition_valid"][i])]
    available_old = old_event_ids(ep, decision, past, window)
    retained_old = old_event_ids(ep, decision, bank, window)
    return {"decision": decision, "frame": int(ep["frames"][decision]), "paired_seeds": seeds,
            "available_old_count": len(available_old), "retained_old_count": len(retained_old),
            "retained_old_fraction": len(retained_old) / len(available_old) if available_old else None,
            "bank_ids": bank, "fifo_bank_ids": fifo, "writer_stats": stats, "fifo_stats": fifo_stats,
            "conditions": conditions}


@torch.no_grad()
def parity_experiment(memory, ep, decision, *, stride, atol):
    """Real cached endpoints through native online bank; NOT RGB/processor parity.

    GT executed prefixes are used only for demonstration replay here, never
    injected into closed-loop evaluation. Invalid/incomplete transitions have
    no matching native online update, and are explicitly unsupported/skipped.
    """
    if not bool(ep["transition_valid"][:decision].all()):
        return {"status": "unsupported", "reason": "invalid transition in selected prefix"}
    stream = OnlineActionValueBank(memory, stride, "hard")
    differences, history = [], []
    for endpoint in range(decision + 1):
        passive = bool(ep["is_demo"][endpoint]) if "is_demo" in ep else (
            not bool(ep["decision_mask"][endpoint]) if endpoint < len(ep["actions"]) else False)
        controls = None if endpoint == 0 else ep["actions"][endpoint - 1][ep["action_mask"][endpoint - 1]]
        if endpoint > 0 and not stream.previous.passive and len(controls) != int(ep["frames"][endpoint] - ep["frames"][endpoint - 1]):
            return {"status": "unsupported", "reason": "cached executed prefix incomplete", "endpoint": endpoint}
        if endpoint > 0 and (int(ep["frames"][endpoint] - ep["frames"][endpoint - 1]) > stride
                             or (not stream.previous.passive and passive)):
            return {"status": "unsupported", "reason": "non-deployable cadence or return to passive phase", "endpoint": endpoint}
        actual, info = stream.advance(ep["short"][endpoint], ep["moment"][endpoint], ep["state"][endpoint],
                                      frame=int(ep["frames"][endpoint]), passive=passive, actions=controls)
        encoded = encode_until(memory, ep, endpoint)
        bank, _ = replay_bank(memory, ep, endpoint, "hard", encoded)
        expected = read_bank(memory, ep, endpoint, bank, encoded)["fused_short"]
        difference = _float((actual - expected).abs().max())
        differences.append(difference)
        history.append({"endpoint": endpoint, "frame": int(ep["frames"][endpoint]),
                        "offline_bank_ids": bank, "online_bank_ids": info["bank_event_ids"],
                        "fused_max_abs_error": difference})
    passed = all(row["offline_bank_ids"] == row["online_bank_ids"] for row in history) and max(differences, default=0) <= atol
    return {"status": "pass" if passed else "mismatch", "max_abs_error": max(differences, default=0),
            "atol": atol, "endpoints": history,
            "scope": "Cached normalized endpoints/actions, bank/fusion only; not raw RGB extraction or actual rollout parity"}


@torch.no_grad()
def select_writer_contexts(memory, fetch, candidates, count, source):
    """Report natural coverage, then balance observed fill strata; no invented full banks."""
    groups = defaultdict(list)
    for row in candidates:
        ep = fetch(row["episode_id"])
        bank, _ = replay_bank(memory, ep, row["candidate"], "hard" if source == "deployed" else "all")
        category = "empty" if not bank else "full" if len(bank) == memory.config.capacity else "partial"
        groups[category].append({**row, "bank_ids": bank, "fill_stratum": category, "bank_source": source})
    coverage = {name: len(groups[name]) for name in ("empty", "partial", "full")}
    selected = []
    while len(selected) < count and any(groups.values()):
        for name in ("full", "partial", "empty"):
            if groups[name]:
                selected.append(groups[name].pop(0))
                if len(selected) == count:
                    break
    return selected, {"candidate_pool_counts": coverage, "selected_counts": dict(Counter(r["fill_stratum"] for r in selected)),
                      "bank_source": source, "natural_deployed_distribution": source == "deployed",
                      "note": "Missing full banks means replacement NOT adequately assessed; use separate fifo-stress run if desired"}


def _read_journal(path):
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))  # Fail closed on a truncated journal; never duplicate silently.
    return rows


def write_report(output):
    rows = _read_journal(output / "results.jsonl")
    queries = [r for r in rows if r["experiment"] == "interventions"]
    writers = [r for r in rows if r["experiment"] == "writer"]
    parities = [r for r in rows if r["experiment"] == "parity"]
    summary = {"variant": VARIANT, "query_count": len(queries), "writer_context_count": len(writers),
               "parity_status_counts": dict(Counter(r["status"] for r in parities)), "conditions": {}}
    lines = ["Frozen v4 diagnostic report (NOT robot success/accuracy)",
             f"Queries={len(queries)}; writer contexts={len(writers)}; parity={summary['parity_status_counts']}",
             "Positive loss_minus_full means normal memory has LOWER action loss.",
             "These are selected, teacher-forced validation episodes previously used for model selection.", "",
             "Condition         N       action_loss      loss_minus_full"]
    for name in CONDITIONS:
        values = [r["conditions"][name] for r in queries]
        if values:
            mean_loss = statistics.fmean(r["action_loss"] for r in values)
            delta = statistics.fmean(r["loss_minus_full"] for r in values)
            summary["conditions"][name] = {"count": len(values), "action_loss": mean_loss, "loss_minus_full": delta}
            lines.append(f"{name:17s} {len(values):4d} {mean_loss:17.9f} {delta:+18.9f}")
    summary["intervention_coverage"] = {}
    for name in ("no_old", "shuffled_old"):
        effective = [r["conditions"][name] for r in queries
                     if r["conditions"][name]["diagnostic"].get("intervention_effective")]
        entry = {"effective_queries": len(effective), "total_queries": len(queries),
                 "effective_loss_minus_full": statistics.fmean(r["loss_minus_full"] for r in effective) if effective else None}
        summary["intervention_coverage"][name] = entry
        lines.append(f"{name} effective: {len(effective)}/{len(queries)}; effective-only delta={entry['effective_loss_minus_full']}")
    audited = [r for r in writers if r.get("status") == "ok"]
    learned = [r for r in audited if not r.get("writer_forced_min_fill")]
    summary["writer"] = {"audited": len(audited), "learned_decisions": len(learned),
                         "forced_min_fill": len(audited) - len(learned),
                         "full_bank_contexts": sum(r["full_bank"] for r in audited)}
    lines += ["", f"Writer: {summary['writer']}"]
    for branch in ("fixed_bank", "continuation"):
        fields = ("gain_sign_agreement_fraction", "non_tie_option_count_both_splits",
                  "writer_minus_A_best_loss_on_B", "writer_keep_relative_gain_on_B",
                  "A_best_keep_relative_gain_on_B", "writer_tie_aware_agreement_with_A",
                  "expanded_vs_deployed_A_best_gain_on_B")
        means = {key: statistics.fmean(float(r[branch][key]) for r in learned) for key in fields} if learned else {}
        summary["writer"][branch] = means
        lines.append(f"{branch} (learned decisions only): {json.dumps(means, sort_keys=True)}")
    lines += ["", "Per-query effective intervention details, exact banks/seeds/draws: results.jsonl",
              "Writer A/B and fixed-bank/continuation details: writer_contexts.csv + results.jsonl",
              "Attention/bank fill alone does NOT demonstrate useful memory. No-old may be ineffective with no retained old events.",
              "Shuffled-old permutes temporal assignment within the SAME past; contents as a set stay present. A null result cannot rule out content use.",
              "Writer A-best evaluated on B avoids selection on the same noise. B-best regret alone is descriptive, not independent evidence.",
              "Continuation is teacher-forced causal write replay, NOT counterfactual simulator rollouts."]
    _atomic_json(output / "summary.json", summary)
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "query_losses.csv").open("w", newline="") as handle:
        fields = ["episode_id", "decision", "frame", "condition", "action_loss", "loss_minus_full", "diagnostic"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in queries:
            for name, result in row["conditions"].items():
                writer.writerow({"episode_id": row["episode_id"], "decision": row["decision"], "frame": row["frame"],
                                 "condition": name, "action_loss": result["action_loss"], "loss_minus_full": result["loss_minus_full"],
                                 "diagnostic": json.dumps(result.get("diagnostic", {}), sort_keys=True)})
    with (output / "writer_contexts.csv").open("w", newline="") as handle:
        fields = ["episode_id", "candidate", "fill_stratum", "bank_source", "forced_min_fill"]
        metrics = ("gain_sign_agreement_fraction", "non_tie_option_count_both_splits",
                   "writer_minus_A_best_loss_on_B", "A_best_keep_relative_gain_on_B",
                   "writer_tie_aware_agreement_with_A", "expanded_vs_deployed_A_best_gain_on_B")
        fields += [f"{branch}_{metric}" for branch in ("fixed_bank", "continuation") for metric in metrics]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in writers:
            flat = {k: row.get(k) for k in ("episode_id", "candidate", "fill_stratum", "bank_source")}
            flat["forced_min_fill"] = row.get("writer_forced_min_fill")
            flat.update({f"{branch}_{metric}": row.get(branch, {}).get(metric)
                         for branch in ("fixed_bank", "continuation") for metric in metrics})
            writer.writerow(flat)
    print("\n".join(lines), flush=True)
    return summary


def _verify_plan_payloads(cache, plan):
    records = {str(row["episode_id"]): row for row in cache.manifest["episodes"]}
    for eid, digest in plan["payload_sha256"].items():
        if file_sha256(cache.path / records[eid]["path"]) != digest:
            raise ValueError(f"Selected cached episode changed after diagnostic planning: {eid}")


def _execute(args, cache, info, base, window, stride, output, resume):
    from .diagnostic_writer import audit_storage_context
    fetch = MappedEpisodes(cache).fetch
    if resume:
        integrity = json.loads((output / "sampling_plan_integrity.json").read_text())
        if integrity["sha256"] != file_sha256(output / "sampling_plan.json"):
            raise ValueError("Sampling plan changed after diagnostic initialization")
        plan = json.loads((output / "sampling_plan.json").read_text())
        _verify_plan_payloads(cache, plan)
    else:
        plan = make_plan(args, cache, fetch, window)
        _atomic_json(output / "sampling_plan.json", plan)
        _atomic_json(output / "sampling_plan_integrity.json", {"sha256": file_sha256(output / "sampling_plan.json")})
    print(f"[diagnostic] planned {len(plan['queries'])} queries, {len(plan['writer_candidates'])} writer candidates; ALL weights frozen", flush=True)
    memory = ActionValueMemory(MemoryV3Config(**info["config"]["memory"])).to(args.device)
    model, processor = load_frozen_hamlet(base, args.device)
    head = model.action_head
    del model, processor  # Cached features need the Action Expert, not resident VLM.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    install_expert_lora(head, LoRAConfig(**info["config"]["expert"]), targets=info["config"]["expert_targets"])
    load_checkpoint_v4(resolve(args.checkpoint), memory, head)
    memory.eval().requires_grad_(False)
    head.eval().requires_grad_(False)
    initial_reader, initial_expert = reader_state_sha256(memory), expert_state_sha256(head)
    initial_writer = {name: value.detach().clone() for name, value in memory.writer.state_dict().items()}
    if "writer" in args.experiments:
        context_file = output / "writer_plan.json"
        if resume and context_file.exists():
            if json.loads((output / "writer_plan_integrity.json").read_text())["sha256"] != file_sha256(context_file):
                raise ValueError("Writer plan changed after diagnostic initialization")
            writer_plan = json.loads(context_file.read_text())
        else:
            selected, coverage = select_writer_contexts(memory, fetch, plan["writer_candidates"], args.writer_contexts, args.writer_bank_source)
            writer_plan = {"contexts": selected, "coverage": coverage}
            _atomic_json(context_file, writer_plan)
            _atomic_json(output / "writer_plan_integrity.json", {"sha256": file_sha256(context_file)})
        print("[diagnostic] writer coverage: " + json.dumps(writer_plan["coverage"]), flush=True)
    else:
        writer_plan = {"contexts": []}
    saved_rows = _read_journal(output / "results.jsonl")
    completed = {row["key"] for row in saved_rows}
    jobs = [("interventions", row) for row in plan["queries"] if "interventions" in args.experiments]
    jobs += [("parity", row) for row in plan["queries"][:args.parity_samples] if "parity" in args.experiments]
    jobs += [("writer", row) for row in writer_plan["contexts"]]
    planned_keys = {f"{kind}:{row['episode_id']}:{row.get('candidate', row.get('decision'))}" for kind, row in jobs}
    if len(completed) != len(saved_rows) or not completed <= planned_keys:
        raise ValueError("Duplicate or unplanned diagnostic journal rows; refusing ambiguous resume")
    start = time.monotonic()
    with (output / "results.jsonl").open("a", encoding="utf-8") as journal:
        for index, (experiment, row) in enumerate(jobs):
            key = f"{experiment}:{row['episode_id']}:{row.get('candidate', row.get('decision'))}"
            if key in completed:
                print(f"[diagnostic] resume skip {key}", flush=True)
                continue
            ep = fetch(row["episode_id"])
            paired_seed = args.seed + row["episode_id"] * 1000003 + row.get("candidate", row.get("decision")) * 10007
            print(f"[diagnostic] {index + 1}/{len(jobs)} {key}; elapsed={time.monotonic() - start:.1f}s", flush=True)
            if experiment == "interventions":
                result = query_experiment(memory, head, ep, row["decision"], window=window,
                                          noise_samples=args.noise_samples, seed=paired_seed)
            elif experiment == "parity":
                result = parity_experiment(memory, ep, row["decision"], stride=stride, atol=args.parity_atol)
            else:
                result = audit_storage_context(memory, head, ep, row["candidate"], row["bank_ids"], memory_window=window,
                    future_samples=args.future_samples, noise_samples=args.noise_samples, seed=paired_seed,
                    tie_tolerance=args.tie_tolerance, all_victims=args.all_victims, loss_fn=_flow)
            record = {**row, **result, "experiment": experiment, "key": key}
            journal.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            journal.flush()
    if initial_reader != reader_state_sha256(memory) or initial_expert != expert_state_sha256(head):
        raise RuntimeError("Frozen reader/Expert changed during diagnostic")
    if any(not torch.equal(value, memory.writer.state_dict()[name]) for name, value in initial_writer.items()):
        raise RuntimeError("Frozen writer changed during diagnostic")
    summary = write_report(output)
    failed = summary["parity_status_counts"].get("mismatch", 0) > 0
    _atomic_json(output / "completion.json", {"complete": True, "weights_unchanged": True,
                 "parity_mismatch": failed, "elapsed_seconds": time.monotonic() - start})
    return 2 if failed else 0


def main(argv=None):
    args = parser().parse_args(argv)
    output = resolve(args.output_dir)
    if args.report_only:
        if not (output / "manifest.json").is_file():
            raise FileNotFoundError("No diagnostic manifest at output directory")
        if json.loads((output / "manifest.json").read_text()).get("variant") != VARIANT:
            raise ValueError("Not a frozen v4 diagnostic output; refusing to modify unrelated summaries")
        with (output / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write_report(output)
        return 0
    cache, info, base, window, stride = preflight(args)
    if args.preflight_only:
        print(f"[preflight] compatible v4 Stage 2 step={info['step']}; cache train={len(cache.manifest['splits']['train'])}, val={len(cache.manifest['splits']['val'])}; K={window}, stride={stride}")
        print("[preflight] No model/GPU/simulator loaded; no output written; no experiment started.")
        return 0
    for protected in (cache.path, base, resolve(args.checkpoint)):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Diagnostic output cannot overlap cache/checkpoint storage")
    if args.resume and not (output / "manifest.json").is_file():
        raise FileNotFoundError("--resume requires an existing diagnostic manifest")
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("Output not empty. Use a NEW directory or identical --resume; never overwrite old experiments")
    identity = make_identity(args, cache, info, base)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if json.loads((output / "manifest.json").read_text()) != identity:
                raise ValueError("Diagnostic provenance/settings changed; use a NEW output directory")
        else:
            if any(p.name != ".lock" for p in output.iterdir()):
                raise FileExistsError("Output initialized by another process; refusing overwrite")
            _atomic_json(output / "manifest.json", identity)
        with (output / "diagnostic.log").open("a", encoding="utf-8") as log:
            with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
                try:
                    return _execute(args, cache, info, base, window, stride, output, args.resume)
                except BaseException:
                    traceback.print_exc()
                    # Existing flushed rows remain available even if a later job fails.
                    if (output / "results.jsonl").exists():
                        write_report(output)
                    raise
