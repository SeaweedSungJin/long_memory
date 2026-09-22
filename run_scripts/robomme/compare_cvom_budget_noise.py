#!/usr/bin/env python3
"""Strict nested-noise comparison, and a fail-closed teacher review gate.

The two panels must differ ONLY in noise_samples. Their shared flow draws and
all generated actions must reproduce exactly. Both selected decisions are
then evaluated on the SAME larger held-out panel, separating selection quality
from a change in how the evaluation mean itself was estimated.
No training command is launched, even if the exploratory criteria are met.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from run_scripts.robomme.cvom_budget_probe import digest, read_packets, matching_verification
from run_scripts.robomme.cvom_budget_statistics import summarize_budget_contexts
from run_scripts.robomme.train_cvom_admission import file_hash
from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.safety_v5 import validate_output_scope


def load_verified_run(path):
    path = Path(path).resolve(strict=True)
    protocol = json.loads((path/"protocol.json").read_text())
    if digest({k: v for k, v in protocol.items() if k != "fingerprint"}) != protocol["fingerprint"]:
        raise ValueError("Modified probe protocol")
    packets = read_packets(path, protocol)
    if len(packets) != len(protocol["plan"]) or not matching_verification(path, protocol, packets):
        raise ValueError("Need a complete, invariance-verified probe, not partial evidence")
    for name, sha in protocol["source_sha256"].items():
        if file_hash(ROOT/name) != sha:
            raise ValueError(f"Probe source changed: {name}; do not silently reinterpret old evidence")
    return protocol, packets


def validate_noise_pair(low, high, low_packets, high_packets):
    a, b = [copy.deepcopy(p) for p in (low, high)]
    for p in (a, b):
        p.pop("fingerprint")
    n0, n1 = a["settings"].pop("noise_samples"), b["settings"].pop("noise_samples")
    if (n0, n1) != (2, 8) or a != b:
        raise ValueError("Only noise_samples 2 -> 8 is allowed; every other protocol field must match")
    if len(low_packets) != len(high_packets):
        raise ValueError("Different context count")
    for p, q in zip(low_packets, high_packets, strict=True):
        if p["row_sha256"] != q["row_sha256"] or p["actor_state"] != q["actor_state"]:
            raise ValueError("Different fixed context or frozen actor")
        x, y = p["result"], q["result"]
        for key in ("operations", "manager_operation_id", "manager_scores", "bank_event_ids", "pool_provenance"):
            if x[key] != y[key]:
                raise ValueError(f"Noise changed current storage inputs/manager: {key}")
        for name in ("selection_a", "selection_b", "heldout"):
            left, right = x["panels"][name], y["panels"][name]
            if left["query_ids"] != right["query_ids"]:
                raise ValueError("Changed future queries")
            index = {value: i for i, value in enumerate(right["draw_ids"])}
            for i, identity in enumerate(left["draw_ids"]):
                if identity not in index:
                    raise ValueError("Noise panels are not nested")
                for metric in ("flow_prefix", "flow_unweighted", "flow_weighted", "flow_joint", "flow_gripper"):
                    if left[metric][i] != right[metric][index[identity]]:
                        raise ValueError(f"Shared flow draw did not reproduce exactly: {name}/{metric}")
            if name == "heldout":
                for key in ("generation_draws", "generated_mse", "generated_joint", "generated_gripper"):
                    if key not in left or left[key] != right.get(key):
                        raise ValueError("Need identical actual-generation heldout evidence")
    return {"noise_samples": [n0, n1], "same_actor_context_and_future": True,
            "nested_flow_draws_bit_exact": True, "generation_metrics_bit_exact": True}


def paired_gain(rows, key, seed=260922):
    # One context/episode in this study, but group before bootstrapping anyway.
    episodes = sorted({r["episode_id"] for r in rows})
    values = np.array([np.mean([r[key] for r in rows if r["episode_id"] == e]) for e in episodes])
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(5000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.percentile(draws, [2.5, 97.5]).tolist()
            if len(episodes) > 1 else None, "episodes": len(episodes),
            "gain_sign": "noise2-selected error minus noise8-selected error on SAME heldout panel"}


def compare(low, high, low_packets, high_packets):
    audit = validate_noise_pair(low, high, low_packets, high_packets)
    small = summarize_budget_contexts([p["result"] for p in low_packets])
    large = summarize_budget_contexts([p["result"] for p in high_packets])
    retested = []
    for p, q in zip(low_packets, high_packets, strict=True):
        result = copy.deepcopy(p["result"])
        result["panels"]["heldout"] = q["result"]["panels"]["heldout"]
        retested.append(result)
    small_on_large = summarize_budget_contexts(retested)
    metrics = list(large["overall"])
    rows = []
    for x, y in zip(small_on_large["per_context"], large["per_context"], strict=True):
        row = {"episode_id": x["episode_id"], "task": x["task"],
               "noise2_choice": x["selection_a_operation_id"], "noise8_choice": y["selection_a_operation_id"]}
        for metric in metrics:
            row[metric + "_selection_gain"] = (x["heldout"][metric]["losses"]["selected_a"] -
                                                 y["heldout"][metric]["losses"]["selected_a"])
        rows.append(row)
    # Exploratory minimum evidence gate, not a guarantee of rollout utility.
    criteria = {"at_least_8_episodes": len({r["episode_id"] for r in rows}) >= 8}
    rank_ci = large["stability"]["pairwise_repeat_pooled_episode_bootstrap"]["ci95"]
    criteria["repeat_rank_ci_above_chance"] = rank_ci is not None and rank_ci[0] > .5
    for metric in ("flow_prefix", "generated_mse"):
        for reference in ("fifo", "keep"):
            interval = large["overall"][metric]["episode_macro"]["gains"]["selected_a_vs_"+reference]["ci95"]
            criteria[f"{metric}_heldout_gain_vs_{reference}_ci_positive"] = interval is not None and interval[0] > 0
    return {"version": "cvom_noise_qualification_v1", "pair_audit": audit,
        "noise2": small, "noise8": large, "noise2_choices_on_noise8_heldout": small_on_large,
        "selection_comparison_same_heldout": {m: paired_gain(rows, m+"_selection_gain") for m in metrics},
        "per_context_comparison": rows,
        "teacher_gate": {"criteria": criteria, "eligible_for_manual_training_review": all(criteria.values()),
            "automatic_training_allowed": False, "failed_criteria": [k for k,v in criteria.items() if not v],
            "interpretation": "Exploratory offline minimum; positive review still requires explicit human decision. NOT rollout success."},
        "limitations": ["Repeated cache-VAL development data, not simulator VAL160/TEST.",
            "Intervals cluster episodes; only 16 episodes in 6 eligible tasks.",
            "Noise8 vs noise2 panels are nested, not independent experiments.",
            "The same larger heldout panel evaluates both preselected decisions; heldout is never used for selection.",
            "A failed gate means insufficient evidence, not proof that useful memory cannot be learned."]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--noise2-dir", type=Path, default=Path("runs/long_memory/cvom_budget_val16_v1"))
    p.add_argument("--noise8-dir", type=Path, default=Path("runs/long_memory/cvom_budget_val16_noise8_v1"))
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    output = validate_output_scope(a.output_dir, a.noise2_dir, a.noise8_dir)
    if output.exists():
        raise FileExistsError("Comparison requires a NEW output, never overwrite evidence")
    low, lp = load_verified_run(a.noise2_dir)
    high, hp = load_verified_run(a.noise8_dir)
    result = compare(low, high, lp, hp)
    result["provenance"] = {"noise2_dir": str(a.noise2_dir.resolve()), "noise8_dir": str(a.noise8_dir.resolve()),
        "protocol_fingerprints": [low["fingerprint"], high["fingerprint"]],
        "source_sha256": {name: file_hash(ROOT/"run_scripts/robomme"/name) for name in
                          ("compare_cvom_budget_noise.py", "cvom_budget_statistics.py", "cvom_budget_probe.py")},
        "result_payload_hashes": [[p["result_sha256"] for p in packets] for packets in (lp, hp)]}
    output.mkdir(parents=True)
    _atomic_json(output/"comparison.json", result)
    rows = result["per_context_comparison"]
    with (output/"paired_selection.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ["# Noise 2 → 8: fixed-budget CVoM teacher", "", "Offline diagnostic, NOT rollout success.", "",
             "| Metric | Noise 2 | Noise 8 |", "|---|---:|---:|"]
    for key, title in (("top1_repeat_episode_macro", "Same best choice across noise panels"),
                       ("pairwise_repeat_pooled_episode_bootstrap", "Pair rank repeatability")):
        lines.append(f"| {title} | {small_value(result['noise2'], key)} | {small_value(result['noise8'], key)} |")
    lines += ["", "## Same-heldout selection gains", "", "Positive: noise8 selected a better decision.", "",
              "```json", json.dumps(result["selection_comparison_same_heldout"], indent=2), "```", "",
              "## Teacher gate (never launches training)", "", "```json", json.dumps(result["teacher_gate"], indent=2), "```"]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result["teacher_gate"], indent=2))
    print(f"[noise-compare] {output/'report.md'}")
    return 0


def small_value(summary, key):
    block = summary["stability"][key]
    return f"{block['value']:.4f}; CI {block['ci95']}" if block["value"] is not None else "undefined"


if __name__ == "__main__":
    raise SystemExit(main())
