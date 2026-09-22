"""Read-only statistics for a fixed-budget CVoM teacher qualification panel.

Input contexts contain an immutable operation inventory (one KEEP, one FIFO,
other DROP operations), selection_a / selection_b / heldout loss panels, and
the learned manager's chosen operation ID. Each panel's ``flow_prefix`` is a
finite nonnegative [draw, operation] matrix. Only heldout needs the optional
``generated_mse`` matrix. Selection NEVER uses a heldout loss.

All gains are reference loss minus chosen loss: positive is improvement.
Contexts are averaged within episode before episode-macro inference. Bootstrap
resamples EPISODES, not nearby frames, noise draws, operations, or slot pairs.
No output is robot-task success, an automatic training recommendation, or an
unbiased estimate of final benchmark generalization.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math

import numpy as np


STATISTICS_VERSION = "cvom_budget_qualification_statistics_v1"
PANEL_NAMES = ("selection_a", "selection_b", "heldout")
OPTIONAL_HELDOUT_METRICS = ("flow_weighted", "flow_joint", "flow_gripper",
                          "generated_mse", "generated_joint", "generated_gripper")


def _seed(seed, label):
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "little")


def _matrix(panel, key, n_operations):
    if key not in panel:
        raise ValueError(f"Missing panel loss matrix: {key}")
    value = np.asarray(panel[key])
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != n_operations or value.dtype.kind not in "fiu":
        raise ValueError(f"{key} must be numeric [draw, operation], with at least one draw")
    value = value.astype(np.float64)
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"{key} must contain finite nonnegative errors")
    return value


def _operation_inventory(context):
    operations = context.get("operations")
    if not isinstance(operations, list) or len(operations) < 2:
        raise ValueError("Need a fixed operation list containing KEEP and FIFO")
    ids, kinds, dropped = [], [], []
    for operation in operations:
        identity, kind, event = operation.get("id"), operation.get("kind"), operation.get("drop_event_id")
        if not isinstance(identity, str) or not identity or kind not in ("keep", "fifo", "drop"):
            raise ValueError("Operations need unique nonempty IDs and kind keep/fifo/drop")
        if type(event) is not int or event < 0:
            raise ValueError("Every fixed-budget operation must identify its dropped event")
        ids.append(identity); kinds.append(kind); dropped.append(event)
    if len(ids) != len(set(ids)) or len(dropped) != len(set(dropped)):
        raise ValueError("Duplicate operation ID or dropped-event identity")
    if kinds.count("keep") != 1 or kinds.count("fifo") != 1:
        raise ValueError("Exactly one KEEP and one FIFO operation are required")
    manager = context.get("manager_operation_id")
    if manager not in ids:
        raise ValueError("Learned manager operation is not in the evaluated inventory")
    return ids, kinds.index("keep"), kinds.index("fifo"), ids.index(manager)


def _metadata_check(panels, matrices):
    """Validate declared separation; metadata alone cannot audit RNG execution."""
    present = ["draw_ids" in panels[name] for name in PANEL_NAMES]
    draw_sets = {}
    for name, available in zip(PANEL_NAMES, present):
        if not available:
            continue
        identities = panels[name]["draw_ids"]
        if (not isinstance(identities, list) or len(identities) != matrices[name].shape[0]
                or any(not isinstance(identity, str) or not identity for identity in identities)
                or len(set(identities)) != len(identities)):
            raise ValueError("draw_ids must uniquely identify every matrix row")
        draw_sets[name] = set(identities)
    for i, first in enumerate(PANEL_NAMES):
        for second in PANEL_NAMES[i+1:]:
            if first in draw_sets and second in draw_sets and draw_sets[first] & draw_sets[second]:
                raise ValueError("Selection/repeat/heldout draw identities overlap")
    queries = {}
    for name in PANEL_NAMES:
        if "query_ids" not in panels[name]:
            continue
        values = panels[name]["query_ids"]
        if (not isinstance(values, list) or not values
                or any(type(value) is not int or value < 0 for value in values)):
            raise ValueError("query_ids must list nonnegative future-query indices")
        queries[name] = set(values)
    if "heldout" in queries:
        for name in ("selection_a", "selection_b"):
            if name in queries and queries[name] & queries["heldout"]:
                raise ValueError("Heldout future queries overlap a selection panel")
    # A/B may intentionally share queries but must have independent noise.
    return {"disjoint_draw_ids_verified": all(present),
            "disjoint_heldout_query_ids_verified": len(queries) == 3,
            "limitation": "declaration validation only; does not prove actual RNG/conditioning equivalence"}


def _argmin(values, keep, fifo, tolerance):
    smallest = float(np.min(values))
    tied = [index for index, value in enumerate(values) if value <= smallest+tolerance]
    preferred = [keep, fifo] + [index for index in range(len(values)) if index not in (keep, fifo)]
    return next(index for index in preferred if index in tied), tied


def _average_ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    first = 0
    while first < len(order):
        last = first+1
        while last < len(order) and values[order[last]] == values[order[first]]:
            last += 1
        ranks[order[first:last]] = (first+last-1)/2
        first = last
    return ranks


def _stability(a, b, choice_a, choice_b, ties_a, ties_b, tolerance):
    comparable = concordant = tied_both = tied_one = 0
    for i in range(len(a)):
        for j in range(i+1, len(a)):
            da, db = float(a[i]-a[j]), float(b[i]-b[j])
            ta, tb = abs(da) <= tolerance, abs(db) <= tolerance
            if ta or tb:
                tied_both += int(ta and tb)
                tied_one += int(ta != tb)
                continue
            comparable += 1
            concordant += int((da > 0) == (db > 0))
    ar, br = _average_ranks(a), _average_ranks(b)
    if ar.std() == 0 or br.std() == 0:
        spearman = None
    else:
        spearman = float(np.corrcoef(ar, br)[0, 1])
    return {"top1_repeated": choice_a == choice_b,
            "top1_tie_count_a": len(ties_a), "top1_tie_count_b": len(ties_b),
            "all_operations_tied_a": len(ties_a) == len(a),
            "all_operations_tied_b": len(ties_b) == len(b),
            "best_second_margin_a": float(np.partition(a, 1)[1]-a.min()),
            "best_second_margin_b": float(np.partition(b, 1)[1]-b.min()),
            "pairwise_comparable": comparable, "pairwise_concordant": concordant,
            "pairwise_tied_both": tied_both, "pairwise_tied_one_panel": tied_one,
            "pairwise_agreement": concordant/comparable if comparable else None,
            "spearman_exact_average_ties": spearman}


def _heldout_metrics(means, chosen, keep, fifo, manager, tolerance):
    # This diagnostic bound uses the true minimum, even when selection has an
    # explicitly declared nonzero near-tie tolerance.
    oracle, tied = _argmin(means, keep, fifo, 0.)
    losses = {"selected_a": float(means[chosen[0]]), "selected_b": float(means[chosen[1]]),
              "manager": float(means[manager]), "keep": float(means[keep]), "fifo": float(means[fifo]),
              "uniform_random_expectation": float(means.mean()), "heldout_oracle_in_sample": float(means[oracle])}
    gains = {}
    for method in ("selected_a", "selected_b", "manager"):
        for reference in ("keep", "fifo", "uniform_random_expectation"):
            gains[f"{method}_vs_{reference}"] = losses[reference]-losses[method]
    gains["selected_a_vs_manager"] = losses["manager"]-losses["selected_a"]
    gains["heldout_oracle_vs_selected_a_in_sample"] = losses["selected_a"]-losses["heldout_oracle_in_sample"]
    gains["heldout_oracle_vs_fifo_in_sample"] = losses["fifo"]-losses["heldout_oracle_in_sample"]
    return {"losses": losses, "gains": gains, "heldout_oracle_index_in_sample": oracle,
            "heldout_oracle_tie_count": len(tied)}


def _context_result(context, index, metric, tolerance):
    episode, task = context.get("episode_id"), context.get("task")
    if type(episode) is not int or episode < 0 or not isinstance(task, str) or not task:
        raise ValueError("Every context needs a nonnegative episode_id and nonempty task")
    ids, keep, fifo, manager = _operation_inventory(context)
    panels = context.get("panels")
    if not isinstance(panels, dict) or any(not isinstance(panels.get(name), dict) for name in PANEL_NAMES):
        raise ValueError("Require selection_a, selection_b and heldout panel dictionaries")
    matrices = {name: _matrix(panels[name], metric, len(ids)) for name in PANEL_NAMES}
    audit = _metadata_check(panels, matrices)
    mean_a, mean_b = [matrices[name].mean(0) for name in PANEL_NAMES[:2]]
    a, ta = _argmin(mean_a, keep, fifo, tolerance)
    b, tb = _argmin(mean_b, keep, fifo, tolerance)
    result = {"context_id": context.get("context_id", f"{episode}:{context.get('event', index)}"),
        "episode_id": episode, "task": task, "operation_count": len(ids),
        "operation_ids": ids, "manager_operation_id": ids[manager],
        "selection_a_operation_id": ids[a], "selection_b_operation_id": ids[b],
        "selection_a_drop_event_id": context["operations"][a]["drop_event_id"],
        "selection_b_drop_event_id": context["operations"][b]["drop_event_id"],
        "selection_a_tied_operation_ids": [ids[i] for i in ta],
        "selection_b_tied_operation_ids": [ids[i] for i in tb],
        "panel_audit": audit, "stability": _stability(mean_a, mean_b, a, b, ta, tb, tolerance),
        "selection_a_in_sample": {"loss": float(mean_a[a]), "gain_vs_keep": float(mean_a[keep]-mean_a[a]),
            "gain_vs_fifo": float(mean_a[fifo]-mean_a[a]),
            "interpretation": "chosen on these same losses; optimistic and not independent evidence"},
        "heldout": {metric: _heldout_metrics(matrices["heldout"].mean(0), (a, b), keep, fifo, manager, tolerance)}}
    if "event" in context:
        result["event"] = context["event"]
    for optional in OPTIONAL_HELDOUT_METRICS:
        if optional in panels["heldout"]:
            matrix = _matrix(panels["heldout"], optional, len(ids))
            result["heldout"][optional] = _heldout_metrics(matrix.mean(0), (a, b), keep, fifo, manager, tolerance)
    # Panel differences include different future queries, not just selection
    # overfitting. Do not present this gap as an unbiased winner's-curse estimate.
    result["selection_panel_generalization_gap"] = {
        reference: result["selection_a_in_sample"][f"gain_vs_{reference}"]-
            result["heldout"][metric]["gains"][f"selected_a_vs_{reference}"]
        for reference in ("keep", "fifo")}
    return result


def _bootstrap_means(values, episode_tasks, *, samples, seed, task_macro=False):
    """Jointly bootstrap all contrast columns so each comparison remains paired."""
    n = len(values)
    if n < 2 or samples == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = []
    if task_macro:
        groups = [[i for i, name in enumerate(episode_tasks) if name == task] for task in sorted(set(episode_tasks))]
    for start in range(0, samples, 128):
        size = min(128, samples-start)
        if task_macro:
            estimates = []
            for group in groups:
                chosen = rng.integers(len(group), size=(size, len(group)))
                estimates.append(values[np.asarray(group)[chosen]].mean(1))
            draws.append(np.stack(estimates).mean(0))
        else:
            chosen = rng.integers(n, size=(size, n))
            draws.append(values[chosen].mean(1))
    return np.percentile(np.concatenate(draws), [2.5, 97.5], axis=0)


def _metric_summary(contexts, metric, *, bootstrap_samples, seed):
    # Include descriptive losses as well as gains; never use operation rows or
    # flow/generation draws as independent samples in these intervals.
    names = [(section, name) for section in ("losses", "gains")
             for name in contexts[0]["heldout"][metric][section]]
    grouped = defaultdict(list)
    task_by_episode = {}
    for row in contexts:
        grouped[row["episode_id"]].append([row["heldout"][metric][section][name] for section, name in names])
        task_by_episode[row["episode_id"]] = row["task"]
    episodes = sorted(grouped)
    values = np.asarray([np.mean(grouped[eid], axis=0) for eid in episodes], dtype=np.float64)
    episode_tasks = [task_by_episode[eid] for eid in episodes]
    result = {"contexts": len(contexts), "episodes": len(episodes), "tasks": len(set(episode_tasks)),
              "aggregation": "average draws per context, contexts per episode; then episode- or task-macro"}
    for method, task_macro in (("episode_macro", False), ("task_macro", True)):
        point = (np.mean([values[np.asarray(episode_tasks) == task].mean(0)
                          for task in sorted(set(episode_tasks))], axis=0) if task_macro else values.mean(0))
        interval = _bootstrap_means(values, episode_tasks, samples=bootstrap_samples,
            seed=_seed(seed, method), task_macro=task_macro)
        block = {"losses": {}, "gains": {}, "episode_counts_by_task": {
                    task: episode_tasks.count(task) for task in sorted(set(episode_tasks))},
                 "ci_interpretation": "unadjusted exploratory episode-cluster percentile interval; tasks fixed for task_macro",
                 "sparse_task_warning": any(episode_tasks.count(task) < 2 for task in set(episode_tasks))}
        for column, (section, name) in enumerate(names):
            block[section][name] = {"mean": float(point[column]),
                "ci95": interval[:, column].tolist() if interval is not None else None}
        result[method] = block
    return result


def _ratio_bootstrap(groups, samples, seed):
    if not groups:
        return {"numerator": 0., "denominator": 0., "value": None, "ci95": None, "episodes": 0}
    values = np.asarray([groups[eid] for eid in sorted(groups)], dtype=np.float64)
    total = values.sum(0)
    if total[1] == 0:
        return {"numerator": float(total[0]), "denominator": 0., "value": None, "ci95": None, "episodes": len(values)}
    draws = []
    if len(values) >= 2 and samples:
        rng = np.random.default_rng(seed)
        for start in range(0, samples, 128):
            indices = rng.integers(len(values), size=(min(128, samples-start), len(values)))
            counts = values[indices].sum(1)
            counts = counts[counts[:, 1] > 0]
            draws.extend((counts[:, 0]/counts[:, 1]).tolist())
    return {"numerator": float(total[0]), "denominator": float(total[1]), "value": float(total[0]/total[1]),
            "ci95": np.percentile(draws, [2.5, 97.5]).tolist() if draws else None, "episodes": len(values)}


def _stability_summary(contexts, samples, seed):
    top, pairs, episode_contexts = defaultdict(lambda: [0., 0.]), defaultdict(lambda: [0., 0.]), defaultdict(list)
    totals = {"pairwise_tied_both": 0, "pairwise_tied_one_panel": 0,
              "all_operations_tied_a": 0, "all_operations_tied_b": 0}
    for row in contexts:
        eid, s = row["episode_id"], row["stability"]
        episode_contexts[eid].append(float(s["top1_repeated"]))
        pairs[eid][0] += s["pairwise_concordant"]
        pairs[eid][1] += s["pairwise_comparable"]
        for key in totals:
            totals[key] += int(s[key])
    for eid, values in episode_contexts.items():
        top[eid] = [float(np.mean(values)), 1.]
    return {"top1_repeat_episode_macro": _ratio_bootstrap(top, samples, _seed(seed, "top1")),
            "pairwise_repeat_pooled_episode_bootstrap": _ratio_bootstrap(pairs, samples, _seed(seed, "pairwise")),
            **totals, "interpretation": "top1 ties favor KEEP; flat/tied teachers can appear reproducible without useful storage signal",
            "pairwise_tie_rule": "exclude a pair if either panel is tied; report excluded counts explicitly"}


def summarize_budget_contexts(contexts, *, bootstrap_samples=5000, seed=260922, min_episodes=8,
                              metric="flow_prefix", tie_tolerance=0.):
    """Analyze immutable paired panels, never modify files/models or select runs.

    ``manager_operation_id`` must have been chosen without heldout outcomes.
    Optional ``draw_ids`` and ``query_ids`` permit declared panel-separation
    checks. Without them separation is explicitly marked unverified. A/B may
    share future queries with different noise; heldout queries must be distinct.

    Returns per-context records, episode/task-macro paired error gains with
    clustered intervals, rank repeatability, and manually reviewable evidence.
    There is deliberately NO automatic pass flag or training recommendation.
    """
    if not isinstance(contexts, (list, tuple)) or not contexts:
        raise ValueError("Need at least one completed context")
    if (type(bootstrap_samples) is not int or bootstrap_samples < 0 or type(seed) is not int or seed < 0
            or type(min_episodes) is not int or min_episodes < 2):
        raise ValueError("Invalid bootstrap seed/count or minimum episode count")
    if metric != "flow_prefix":
        raise ValueError("This predeclared protocol selects operations using flow_prefix only")
    if isinstance(tie_tolerance, bool) or not isinstance(tie_tolerance, (int, float)) or not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("Tie tolerance must be finite and nonnegative")
    rows = [_context_result(context, index, metric, tie_tolerance) for index, context in enumerate(contexts)]
    if len({str(row["context_id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate context identity would duplicate evidence")
    task_by_episode = {}
    for row in rows:
        if row["episode_id"] in task_by_episode and task_by_episode[row["episode_id"]] != row["task"]:
            raise ValueError("One episode cannot belong to multiple tasks")
        task_by_episode[row["episode_id"]] = row["task"]
    metrics = [metric]
    for optional_metric in OPTIONAL_HELDOUT_METRICS:
        optional = [optional_metric in row["heldout"] for row in rows]
        if any(optional) and not all(optional):
            raise ValueError(f"Optional {optional_metric} must cover all included contexts; do not silently summarize a subset")
        if all(optional):
            metrics.append(optional_metric)
    overall = {name: _metric_summary(rows, name, bootstrap_samples=bootstrap_samples, seed=seed) for name in metrics}
    by_task = {task: {name: _metric_summary([row for row in rows if row["task"] == task], name,
                    bootstrap_samples=bootstrap_samples, seed=_seed(seed, task)) for name in metrics}
               for task in sorted(set(task_by_episode.values()))}
    independent = all(row["panel_audit"]["disjoint_draw_ids_verified"]
                      and row["panel_audit"]["disjoint_heldout_query_ids_verified"] for row in rows)
    evidence = {"decision": "manual_review_required; no automatic PASS or training launch",
        "enough_distinct_episodes_for_predeclared_minimum": len(task_by_episode) >= min_episodes,
        "minimum_distinct_episodes": min_episodes,
        "declared_panel_separation_verified": independent,
        "positive_heldout_ci_evidence": {}}
    for name in metrics:
        gains = overall[name]["episode_macro"]["gains"]
        evidence["positive_heldout_ci_evidence"][name] = {}
        for reference in ("keep", "fifo", "uniform_random_expectation"):
            interval = gains[f"selected_a_vs_{reference}"]["ci95"]
            evidence["positive_heldout_ci_evidence"][name][reference] = bool(interval is not None and interval[0] > 0)
    return {"schema_version": STATISTICS_VERSION,
        "settings": {"selection_metric": metric, "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
            "tie_tolerance": tie_tolerance, "tie_rule": "KEEP, then FIFO, then declared operation order",
            "gain_sign": "reference_error - chosen_error; positive is improvement",
            "random_reference": "uniform expectation over all declared operations, including KEEP and FIFO"},
        "per_context": rows, "overall": overall, "by_task": by_task,
        "stability": _stability_summary(rows, bootstrap_samples, seed), "qualification_evidence": evidence,
        "limitations": [
            "Offline prediction errors are not robot rollout success.",
            "Selection A/B choose only on flow_prefix; all optional generated/joint/gripper/weighted metrics evaluate those same choices on heldout data.",
            "Heldout-oracle results select and evaluate on the SAME heldout matrix and are optimistic diagnostic bounds, not deployable performance.",
            "Selection-panel generalization gaps also contain future-query distribution changes, not only winner's-curse bias.",
            "Intervals resample episodes; they do not quantify additional model/noise-seed or task-population uncertainty.",
            "Sparse task strata can produce degenerate within-task intervals; a single episode has no estimated interval.",
            "Intervals are exploratory and unadjusted for multiple contrasts; repeated development validation is not final generalization evidence.",
            "An interval including zero is inconclusive, not proof that memory/teacher has no useful effect.",
            "Panel metadata validation does not independently establish actual runtime common random numbers or fixed conditioning.",
            "No policy training, checkpoint selection, or automatic qualification decision is performed."]}
