"""CPU-only decision diagnostics for immutable CVoM fixed-budget probe packets.

Admission evaluates exactly ONE candidate per context, never mixes old-slot
write logits into classifier accuracy. Eviction evaluates only OLD slots.
Choices use selection A or frozen manager scores; heldout never selects them.
This file deliberately leaves the original probe/statistics source untouched.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_scripts.robomme import cvom_budget_statistics as budget_stats


VERSION = "cvom_writer_decision_metrics_v1"


def _finite_parameter(value, name, *, nonnegative=True):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (nonnegative and value < 0)):
        raise ValueError(f"Invalid {name}")


def _pool_inventory(context):
    operations = context["operations"]
    count = len(operations)
    pool = [op.get("drop_pool_index") for op in operations]
    if any(type(index) is not int for index in pool) or sorted(pool) != list(range(count)):
        raise ValueError("drop_pool_index must uniquely cover the chronological bank plus candidate")
    keep = next(i for i, op in enumerate(operations) if op["kind"] == "keep")
    fifo = next(i for i, op in enumerate(operations) if op["kind"] == "fifo")
    if pool[keep] != count-1 or pool[fifo] != 0:
        raise ValueError("KEEP must drop candidate B; FIFO must drop oldest pool slot 0")
    old_operations = sorted((i for i in range(count) if i != keep), key=lambda i: pool[i])
    scores = {}
    for name in ("utility", "retention", "write_probability"):
        value = np.asarray(context.get("manager_scores", {}).get(name))
        if value.shape != (count,) or value.dtype.kind not in "fiu":
            raise ValueError(f"manager_scores.{name} must have chronological [B+1] shape")
        value = value.astype(np.float64)
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite manager {name}")
        if name == "write_probability" and ((value < 0).any() or (value > 1).any()):
            raise ValueError("write_probability must be in [0,1]")
        scores[name] = value
    return keep, fifo, old_operations, scores


def _old_choice(values, tolerance):
    tied = np.flatnonzero(values <= values.min()+tolerance).tolist()
    # Values are in chronological old-pool order: ties prefer oldest.
    return tied[0], tied


def _rank_record(scores, losses, loss_tolerance, score_tolerance):
    chosen, predicted_ties = _old_choice(scores, score_tolerance)
    target, target_ties = _old_choice(losses, loss_tolerance)
    comparable = concordant = discordant = predicted_pair_ties = target_pair_ties = 0
    for i in range(len(scores)):
        for j in range(i+1, len(scores)):
            loss_delta, score_delta = float(losses[i]-losses[j]), float(scores[i]-scores[j])
            if abs(loss_delta) <= loss_tolerance:
                target_pair_ties += 1
                continue
            comparable += 1
            if abs(score_delta) <= score_tolerance:
                predicted_pair_ties += 1
            elif (loss_delta > 0) == (score_delta > 0):
                concordant += 1
            else:
                discordant += 1
    credit = concordant+.5*predicted_pair_ties
    return {"old_slots": len(scores), "pairs_total": len(scores)*(len(scores)-1)//2,
        "pairs_comparable": comparable, "pairs_target_tied": target_pair_ties,
        "pairs_prediction_tied_among_comparable": predicted_pair_ties,
        "pairs_concordant": concordant, "pairs_discordant": discordant,
        "pairwise_credit": credit, "pairwise_accuracy": credit/comparable if comparable else None,
        "prediction_old_pool_index": chosen, "target_old_pool_index": target,
        "prediction_tie_count": len(predicted_ties), "target_tie_count": len(target_ties),
        "top1_exact_oldest_tie_break": chosen == target,
        "top1_target_set_hit": chosen in target_ties,
        "unique_target": len(target_ties) == 1,
        "all_target_slots_tied": len(target_ties) == len(losses)}


def _gain_block(losses, selected, keep, fifo, teacher):
    selected_losses = {name: float(losses[index]) for name, index in selected.items()}
    selected_losses.update(keep=float(losses[keep]), fifo=float(losses[fifo]))
    gains = {}
    for name in selected:
        for reference, index in (("keep", keep), ("fifo", fifo), ("a_selected_replacement", teacher)):
            if name != reference:
                gains[f"{name}_vs_{reference}"] = float(losses[index]-losses[selected[name]])
    return {"losses": selected_losses, "gains": gains}


def _context_metrics(context, validated, settings):
    keep, fifo, old, scores = _pool_inventory(context)
    operations = context["operations"]
    means = {name: np.asarray(context["panels"][name]["flow_prefix"], dtype=np.float64).mean(0)
             for name in budget_stats.PANEL_NAMES}
    # Exact argmin defines the admission teacher; near ties are explicitly
    # reported, not resolved using heldout. Chronological order breaks ties.
    chosen_a = old[int(np.argmin(means["selection_a"][old]))]
    chosen_b = old[int(np.argmin(means["selection_b"][old]))]
    advantage = float(means["selection_a"][keep]-means["selection_a"][chosen_a])
    centered = advantage-settings["admission_margin"]
    confident = abs(centered) > settings["uncertain_band"]
    target = centered > 0
    probability = float(scores["write_probability"][-1])
    predicted = probability >= settings["write_threshold"]
    manager = next(i for i, op in enumerate(operations) if op["id"] == context["manager_operation_id"])
    # Actual deployment uses exact argmin. score_tie_tolerance is descriptive
    # ranking tolerance only and must not invent a different deployed victim.
    retention_choice = old[int(np.argmin(scores["retention"][:-1]))]
    utility_choice = old[int(np.argmin(scores["utility"][:-1]))]
    selected = {"manager": manager, "a_selected_replacement": chosen_a,
        "a_selected_admission": chosen_a if target else keep,
        "retention_forced_replacement": retention_choice,
        "utility_forced_replacement": utility_choice,
        "candidate_gate_with_a_replacement": chosen_a if predicted else keep}
    output = {"context_id": validated["context_id"], "episode_id": context["episode_id"],
        "task": context["task"], "old_slots": len(old), "panel_audit": validated["panel_audit"],
        "manager_operation_id": context["manager_operation_id"],
        "selected_operation_ids": {name: operations[index]["id"] for name, index in selected.items()},
        "admission": {"candidate_pool_index": len(old), "write_probability": probability,
            "predicted_admit": predicted, "target_admit": target if confident else None,
            "target_admit_without_uncertainty_filter": target,
            "confident": confident, "selection_a_advantage": advantage,
            "selection_a_replacement_id": operations[chosen_a]["id"],
            "selection_b_replacement_id": operations[chosen_b]["id"],
            "selection_b_advantage_same_a_replacement": float(means["selection_b"][keep]-means["selection_b"][chosen_a]),
            "heldout_advantage_same_a_replacement": float(means["heldout"][keep]-means["heldout"][chosen_a]),
            "manager_actual_admitted": manager != keep},
        "eviction": {}, "heldout": {}}
    if "event" in context:
        output["event"] = context["event"]
    for score_name in ("utility", "retention"):
        output["eviction"][score_name] = {panel: _rank_record(scores[score_name][:-1], means[panel][old],
                settings["loss_tie_tolerance"], settings["score_tie_tolerance"])
            for panel in budget_stats.PANEL_NAMES}
    output["eviction"]["teacher_repeat_a_b"] = _rank_record(means["selection_a"][old],
        means["selection_b"][old], settings["loss_tie_tolerance"], settings["loss_tie_tolerance"])
    heldout = context["panels"]["heldout"]
    for metric in ("flow_prefix", *budget_stats.OPTIONAL_HELDOUT_METRICS):
        if metric in heldout:
            losses = budget_stats._matrix(heldout, metric, len(operations)).mean(0)
            output["heldout"][metric] = _gain_block(losses, selected, keep, fifo, chosen_a)
    return output


def _classification_values(counts):
    tn, fp, fn, tp = counts
    total, positive, negative = counts.sum(), fn+tp, tn+fp
    if total == 0:
        return {name: None for name in ("accuracy", "balanced_accuracy", "majority_accuracy",
            "positive_prevalence", "always_admit_accuracy", "always_keep_accuracy")}
    return {"accuracy": float((tn+tp)/total),
        "balanced_accuracy": float((tp/positive+tn/negative)/2) if positive and negative else None,
        "majority_accuracy": float(max(positive, negative)/total),
        "positive_prevalence": float(positive/total),
        "always_admit_accuracy": float(positive/total), "always_keep_accuracy": float(negative/total)}


def _admission_summary(rows, samples, seed):
    grouped = defaultdict(lambda: np.zeros(4, dtype=np.float64))
    uncertain = 0
    for row in rows:
        admission = row["admission"]
        counts = grouped[row["episode_id"]]  # Include episodes with only ambiguous contexts.
        if not admission["confident"]:
            uncertain += 1
            continue
        target, predicted = admission["target_admit"], admission["predicted_admit"]
        counts[2*int(target)+int(predicted)] += 1
    values = np.asarray([grouped[eid] for eid in sorted(grouped)])
    counts = values.sum(0)
    point = _classification_values(counts)
    draws = defaultdict(list)
    if len(values) >= 2 and samples:
        rng = np.random.default_rng(seed)
        for _ in range(samples):
            summary = _classification_values(values[rng.integers(len(values), size=len(values))].sum(0))
            for name, value in summary.items():
                if value is not None:
                    draws[name].append(value)
    return {"contexts": len(rows), "candidate_decisions": len(rows), "episodes": len(values),
        "confident_contexts": int(counts.sum()), "uncertain_contexts": uncertain,
        "confusion": dict(zip(("true_keep", "false_admit", "false_keep", "true_admit"), counts.astype(int).tolist())),
        "metrics": {name: {"value": value,
            "ci95": np.percentile(draws[name], [2.5, 97.5]).tolist() if draws[name] else None,
            "bootstrap_valid_draws": len(draws[name])} for name, value in point.items()},
        "single_target_class_warning": bool(counts.sum() and (counts[:2].sum() == 0 or counts[2:].sum() == 0)),
        "interpretation": "Candidate-only selection-A surrogate labels; pooled context accuracy with episode-cluster bootstrap. Ambiguous contexts are excluded, not relabeled KEEP. Majority accuracy uses this evaluation label prevalence, not a trained classifier. Balanced accuracy is undefined if either class is absent."}


def _ranking_summary(rows, getter, samples, seed):
    pairs, top_unique, top_tied = (defaultdict(lambda: [0., 0.]) for _ in range(3))
    totals = defaultdict(int)
    for row in rows:
        rank = getter(row)
        episode = row["episode_id"]
        pairs[episode][0] += rank["pairwise_credit"]
        pairs[episode][1] += rank["pairs_comparable"]
        if rank["unique_target"]:
            top_unique[episode][0] += int(rank["top1_target_set_hit"])
            top_unique[episode][1] += 1
        top_tied[episode][0] += int(rank["top1_target_set_hit"])
        top_tied[episode][1] += 1
        for key in ("pairs_total", "pairs_comparable", "pairs_target_tied",
                    "pairs_prediction_tied_among_comparable", "pairs_concordant", "pairs_discordant"):
            totals[key] += rank[key]
        totals["target_top1_tied_contexts"] += int(not rank["unique_target"])
        totals["prediction_top1_tied_contexts"] += int(rank["prediction_tie_count"] > 1)
        totals["all_target_slots_tied_contexts"] += int(rank["all_target_slots_tied"])
    return {"contexts": len(rows), **totals,
        "pairwise_accuracy": budget_stats._ratio_bootstrap(pairs, samples, budget_stats._seed(seed, "pairs")),
        "top1_unique_target_accuracy": budget_stats._ratio_bootstrap(top_unique, samples, budget_stats._seed(seed, "unique")),
        "top1_target_set_hit_including_ties": budget_stats._ratio_bootstrap(top_tied, samples, budget_stats._seed(seed, "tied")),
        "bootstrap_boundary_warning": "Zero/all observed hits can give a degenerate percentile CI [0,0]/[1,1]; this is NOT certainty about the population success probability. Inspect hit counts and the small number of episodes.",
        "tie_rule": "Exclude target-loss tied pairs; prediction ties receive 0.5 credit. Top1 uses oldest predicted tie and separately reports unique-target accuracy; all-flat targets provide no ranking evidence.",
        "weighting": "Pooled comparable old-slot pairs / context top1; confidence intervals resample episodes, never slot pairs."}


def _group_summary(rows, settings, seed):
    samples = settings["bootstrap_samples"]
    eviction = {score: {panel: _ranking_summary(rows, lambda row, s=score, p=panel: row["eviction"][s][p],
                samples, budget_stats._seed(seed, f"{score}:{panel}")) for panel in budget_stats.PANEL_NAMES}
        for score in ("utility", "retention")}
    eviction["teacher_repeat_a_b"] = _ranking_summary(rows, lambda row: row["eviction"]["teacher_repeat_a_b"],
        samples, budget_stats._seed(seed, "teacher_repeat"))
    metrics = rows[0]["heldout"].keys()
    return {"admission": _admission_summary(rows, samples, budget_stats._seed(seed, "admission")),
        "eviction": eviction,
        "heldout": {metric: budget_stats._metric_summary(rows, metric, bootstrap_samples=samples, seed=seed)
                    for metric in metrics}}


def summarize_writer_decisions(contexts, *, admission_margin=0., uncertain_band=1e-6,
        write_threshold=.5, loss_tie_tolerance=1e-6, score_tie_tolerance=0.,
        bootstrap_samples=5000, seed=260922):
    """Separate candidate admission, within-bank ranking, and heldout choice gain.

    Existing packet ``result`` dictionaries are accepted, not packet wrappers.
    Score arrays follow chronological pool order; loss matrices follow the
    operation inventory, which can be differently ordered. Selection A's best
    OLD-drop operation defines the candidate target relative to KEEP. Its SAME
    selected replacement is evaluated on heldout; there is no heldout argmin.
    """
    settings = dict(admission_margin=admission_margin, uncertain_band=uncertain_band,
        write_threshold=write_threshold, loss_tie_tolerance=loss_tie_tolerance,
        score_tie_tolerance=score_tie_tolerance, bootstrap_samples=bootstrap_samples, seed=seed)
    for name in ("admission_margin", "uncertain_band", "loss_tie_tolerance", "score_tie_tolerance"):
        _finite_parameter(settings[name], name)
    _finite_parameter(write_threshold, "write_threshold")
    if write_threshold > 1:
        raise ValueError("write_threshold must be in [0,1]")
    # Reuse strict original schema/panel validation, without modifying its
    # source or rerunning actor inference. Zero bootstrap avoids duplicate work.
    checked = budget_stats.summarize_budget_contexts(contexts, bootstrap_samples=0, seed=seed, min_episodes=2)
    if type(bootstrap_samples) is not int or bootstrap_samples < 0:
        raise ValueError("Invalid bootstrap_samples")
    rows = [_context_metrics(context, valid, settings) for context, valid in zip(contexts, checked["per_context"])]
    rows.sort(key=lambda row: (row["episode_id"], str(row["context_id"])))
    overall = _group_summary(rows, settings, seed)
    by_task = {task: _group_summary([row for row in rows if row["task"] == task], settings,
                budget_stats._seed(seed, task)) for task in sorted({row["task"] for row in rows})}
    return {"schema_version": VERSION, "settings": {**settings,
        "selection_metric": "flow_prefix", "gain_sign": "reference error minus chosen error; positive improves",
        "admission_target": "selection A KEEP loss minus minimum OLD-drop loss > admission_margin",
        "uncertain_rule": "abs(selection A advantage - admission_margin) <= uncertain_band",
        "old_rank_direction": "LOW utility/retention predicts LOW loss when dropping that old slot",
        "choice_ties": "Exact minima prefer oldest; ranking tolerances are descriptive only"},
        "per_context": rows, "overall": overall, "by_task": by_task,
        "qualification": "Diagnostic evidence only; no automatic PASS, writer retraining, or policy change.",
        "limitations": [
            "All reported errors/gains are offline conditional fixed-bank predictions, not rollout success or full continuation return.",
            "Admission accuracy is against selection-A surrogate labels, not independent truth. Its optimistic best-replacement label is checked using the SAME replacement on heldout.",
            "Selection A/B share future queries with distinct noise; heldout future queries are separate when metadata verifies this.",
            "The absolute uncertain band and ranking tolerance are heuristic guards, not confidence intervals or calibrated label confidence.",
            "Old-slot write probabilities never contribute to candidate admission accuracy. Candidate gate and full manager retention decision are different quantities.",
            "Forced utility/retention replacements and candidate-gate-plus-teacher-victim choices are diagnostic hybrids, not measured deployed policies.",
            "Every heldout metric uses the SAME flow-A/manager-selected operations. No optional metric selects its own best operation.",
            "Pair counts are descriptive; bootstrap clusters are episodes. Small task/episode panels and tied targets limit power.",
            "Intervals are exploratory, unadjusted for multiple comparisons, conditional on these tasks, actor, future panels and noise draws. They do not include new-noise or task-population uncertainty.",
            "A confidence interval crossing zero is inconclusive, not proof of no useful memory effect. Repeated development VAL is not final generalization evidence."]}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_probe_contexts(run_dir):
    """Read completed packets and verify result hashes; never mutate provenance."""
    paths = sorted(Path(run_dir).resolve().glob("context-*.json"))
    if not paths:
        raise ValueError(f"No context packets in {run_dir}")
    contexts, identities = [], []
    for path in paths:
        packet = json.loads(path.read_text())
        if (not isinstance(packet, dict) or "result" not in packet
                or not isinstance(packet.get("protocol_fingerprint"), str)
                or not packet["protocol_fingerprint"] or not packet.get("actor_state")
                or packet.get("result_sha256") != _digest(packet["result"])):
            raise ValueError(f"Invalid or changed probe packet: {path}")
        contexts.append(packet["result"])
        identities.append({"path": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "result_sha256": packet["result_sha256"], "protocol_fingerprint": packet.get("protocol_fingerprint"),
            "actor_state": packet.get("actor_state")})
    if len({json.dumps([p["protocol_fingerprint"], p["actor_state"]], sort_keys=True) for p in identities}) != 1:
        raise ValueError("Packets do not share a single protocol and actor identity")
    return contexts, identities


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", help="NEW directory only; omit to print JSON")
    parser.add_argument("--admission-margin", type=float, default=0.)
    parser.add_argument("--uncertain-band", type=float, default=1e-6)
    parser.add_argument("--write-threshold", type=float, default=.5)
    parser.add_argument("--loss-tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--score-tie-tolerance", type=float, default=0.)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=260922)
    args = parser.parse_args(argv)
    if args.output_dir and Path(args.output_dir).exists():
        parser.error("output-dir already exists; refusing to overwrite")
    contexts, provenance = load_probe_contexts(args.run_dir)
    result = summarize_writer_decisions(contexts, **{k: v for k, v in vars(args).items()
        if k not in ("run_dir", "output_dir")})
    result["input_packets"] = provenance
    result["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["statistics_source_sha256"] = hashlib.sha256(Path(budget_stats.__file__).read_bytes()).hexdigest()
    if args.output_dir:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=False)
        (output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        with (output/"per_context.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(
                key for row in result["per_context"] for key in row)))
            writer.writeheader()
            writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                              for key, value in row.items()} for row in result["per_context"])
        print(f"[writer-diagnostics] {len(contexts)} contexts; saved {output.resolve()}/summary.json")
        print("[writer-diagnostics] Offline candidate admission / old-slot ranking / heldout gains; NOT rollout success.")
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
