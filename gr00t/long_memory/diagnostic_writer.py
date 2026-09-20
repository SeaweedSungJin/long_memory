"""Read-only audit of v4 writer supervision, not a new training objective.

The historical label freezes each KEEP/APPEND/REPLACE bank until an old-only
future query. This audit also replays intervening *completed* events with the
same frozen writer. Both variants use demonstrated observations/actions; the
continuation is teacher-forced replay, NOT a counterfactual simulator rollout.

Flow noise and time are paired across options and replay variants. Split A
selects reference options; independent split B measures their effect. A B-only
empirical minimum is reported for inspection but is not an unbiased oracle or
a confidence interval. No future target is supplied to the reader or writer.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import random
import statistics

import torch

from .cache_reader_v3 import validate_decision
from .contextual_cvom import eligible_futures
from .diagnostic_interventions import continuation_bank
from .replay_v3 import encode_until, read_bank, storage_options, storage_prediction


DEFINITION = "frozen-writer-split-noise-continuation-audit-v1"


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _check_frozen(memory, head):
    for name, module in (("memory", memory), ("action expert", head)):
        if module is not None and (module.training or any(p.requires_grad for p in module.parameters())):
            raise ValueError(f"{name} must be frozen and in eval mode for a read-only audit")


def _noise_seed(seed, episode_id, candidate, decision, split, repeat):
    # Separate named domains make A and B reproducible and independent of
    # selection order, Python hash randomization, and the global RNG state.
    value = f"{DEFINITION}/{seed}/{episode_id}/{candidate}/{decision}/{split}/{repeat}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**63 - 1)


def _sign(value, tolerance):
    return int(value > tolerance) - int(value < -tolerance)


def _means(draws, count):
    return [statistics.mean(row[index] for future in draws for row in future) for index in range(count)]


def _summarize(draws, selected, deployable, tolerance):
    count = len(draws["A"][0][0])
    means = {split: _means(values, count) for split, values in draws.items()}
    gains = {split: [values[0] - value for value in values] for split, values in means.items()}
    best_a = min(range(count), key=lambda i: means["A"][i])
    best_b = min(range(count), key=lambda i: means["B"][i])
    default_best_a = min(deployable, key=lambda i: means["A"][i])
    sign_a = [_sign(value, tolerance) for value in gains["A"]]
    sign_b = [_sign(value, tolerance) for value in gains["B"]]
    non_ties = [i for i in range(1, count) if sign_a[i] and sign_b[i]]
    return {
        "option_mean_losses": means,
        "keep_relative_gains": gains,
        "keep_relative_gain_signs": {"A": sign_a, "B": sign_b},
        "gain_sign_agreement_fraction": statistics.mean(sign_a[i] == sign_b[i] for i in range(1, count)),
        "non_tie_gain_sign_agreement_fraction": (
            statistics.mean(sign_a[i] == sign_b[i] for i in non_ties) if non_ties else None),
        "non_tie_option_count_both_splits": len(non_ties),
        "best_option_A": best_a,
        "best_option_B_empirical": best_b,
        "deployed_options_best_A": default_best_a,
        "best_option_split_agreement": best_a == best_b,
        "writer_exact_agreement_with_A": selected == best_a,
        "writer_tie_aware_agreement_with_A": means["A"][selected] <= means["A"][best_a] + tolerance,
        "writer_tie_aware_agreement_with_B": means["B"][selected] <= means["B"][best_b] + tolerance,
        # Signed differences can legitimately be negative on independent B.
        "writer_minus_A_best_loss_on_B": means["B"][selected] - means["B"][best_a],
        "writer_keep_relative_gain_on_B": gains["B"][selected],
        "A_best_keep_relative_gain_on_B": gains["B"][best_a],
        "expanded_vs_deployed_A_best_gain_on_B": means["B"][default_best_a] - means["B"][best_a],
        "writer_empirical_B_regret": means["B"][selected] - means["B"][best_b],
        "writer_empirical_B_regret_beyond_tolerance": max(0.0, means["B"][selected] - means["B"][best_b] - tolerance),
        "loss_draws": draws,
    }


@torch.no_grad()
def audit_storage_context(memory, head, episode, candidate, bank_ids, *,
                          memory_window=4, future_samples=2, noise_samples=4,
                          seed=6, loss_fn=None, continuation_policy="hard",
                          tie_tolerance=1e-5, all_victims=False):
    """Audit one legal pre-candidate bank; return only JSON-safe diagnostics.

    ``candidate`` is a valid event completed at endpoint ``candidate + 1``.
    ``bank_ids`` is the bank immediately BEFORE that candidate's write. The
    caller records whether this is a naturally visited or synthetic context.
    ``noise_samples`` is the number of flow draws PER independent A/B split.
    Nominal cost is 2 variants * 2 splits * futures * draws * options, although
    identical (query, bank, seed) forwards are reused. ``all_victims`` audits all
    capacity slots without changing the deployed writer's candidate set.
    """
    _check_frozen(memory, head)
    for name, value in (("memory_window", memory_window), ("future_samples", future_samples),
                        ("noise_samples", noise_samples)):
        _positive_integer(value, name)
    if noise_samples < 2:
        raise ValueError("noise_samples must be >= 2 per independent split")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer")
    if continuation_policy not in ("all", "hard"):
        raise ValueError("continuation_policy must be all or hard")
    if (isinstance(tie_tolerance, bool) or not isinstance(tie_tolerance, numbers.Real)
            or not math.isfinite(tie_tolerance) or tie_tolerance < 0):
        raise ValueError("tie_tolerance must be finite and nonnegative")
    if not isinstance(all_victims, bool):
        raise ValueError("all_victims must be boolean")
    futures = eligible_futures(episode, candidate, memory_window)
    candidate = int(candidate)
    bank_ids = list(bank_ids)
    if any(isinstance(i, bool) or not isinstance(i, numbers.Integral) for i in bank_ids):
        raise ValueError("bank_ids must contain integer event IDs")
    bank_ids = [int(i) for i in bank_ids]
    # storage_options validates chronology/capacity; reject invalid events too.
    default_options = storage_options(bank_ids, candidate, memory.config.capacity, memory.config.max_victims)
    if any(not bool(episode["transition_valid"][i]) for i in bank_ids):
        raise ValueError("bank_ids contains invalid events")
    options = storage_options(bank_ids, candidate, memory.config.capacity,
                              memory.config.capacity if all_victims else memory.config.max_victims)
    deployed = [options.index(option) for option in default_options]
    report = {
        "definition": DEFINITION, "episode_id": int(episode["episode_id"]),
        "candidate": candidate, "candidate_end_frame": int(episode["frames"][candidate + 1]),
        "bank_ids": bank_ids, "capacity": int(memory.config.capacity),
        "bank_fill": len(bank_ids), "full_bank": len(bank_ids) == memory.config.capacity,
        "bank_coverage": "empty" if not bank_ids else ("full" if len(bank_ids) == memory.config.capacity else "partial"),
        "options": options, "deployed_options": default_options,
        "deployed_to_audited_option_indices": deployed, "all_victims": all_victims,
        "continuation_policy": continuation_policy, "memory_window": int(memory_window),
        "noise_samples_per_split": int(noise_samples), "seed": int(seed),
        "tie_tolerance": float(tie_tolerance),
        "interpretation": {
            "gain": "positive = lower paired action loss than KEEP",
            "independent_regret": "writer_minus_A_best_loss_on_B is signed; A chooses, independent B measures",
            "empirical_B_regret": "descriptive B-minimum comparison, optimistically selected on B; not an unbiased oracle",
            "continuation": "causal frozen-writer replay on fixed demonstrated observations/actions, NOT simulator success",
            "noise": "A/B share future queries, but use independent flow noise/time; not independent episodes or calibrated CIs",
            "all_victims": "expanded diagnostic choices never change deployed writer choices",
        },
    }
    if not futures:
        report.update(status="skipped", reason="no_strict_old_future_action_query", future_decisions=[])
        return report
    rng = random.Random(int(seed) + int(episode["episode_id"]) * 100003 + candidate * 1009)
    count = min(int(future_samples), len(futures))
    decisions = [rng.choice(futures[j * len(futures) // count:(j + 1) * len(futures) // count])
                 for j in range(count)]
    report.update(status="ok", future_decisions=decisions,
                  future_frames=[int(episode["frames"][d]) for d in decisions],
                  candidate_age_at_queries_frames=[int(episode["frames"][d]) - report["candidate_end_frame"] for d in decisions],
                  oldest_short_endpoint_frames=[int(episode["frames"][max(0, d - memory_window + 1)]) for d in decisions])
    if "features" in episode:
        for decision in decisions:
            validate_decision(episode, decision)
    encoded = encode_until(memory, episode, max(decisions))
    prediction = storage_prediction(memory, episode, candidate, bank_ids, encoded)
    if prediction["options"] != default_options:
        raise ValueError("Deployed writer options differ from the audited default options")
    if not bool(torch.isfinite(prediction["logits"]).all()):
        raise FloatingPointError("Nonfinite deployed writer logits")
    raw_choice = int(prediction["logits"].argmax().item())
    forced = len(bank_ids) < memory.config.min_fill
    default_choice = 1 if forced else raw_choice
    selected = deployed[default_choice]
    report.update(writer_raw_argmax_default_option=raw_choice,
                  writer_deployed_default_option=default_choice,
                  writer_selected_option=selected, writer_forced_min_fill=forced,
                  writer_deployed_probabilities=prediction["logits"].float().softmax(-1).tolist())
    noise_plan = {split: [[_noise_seed(int(seed), int(episode["episode_id"]), candidate, d, split, r)
                          for r in range(noise_samples)] for d in decisions] for split in ("A", "B")}
    # A collision would defeat the selection/evaluation separation contract.
    all_seeds = [value for rows in noise_plan.values() for row in rows for value in row]
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError("Unexpected audit seed collision")
    report["noise_seeds"] = noise_plan

    branch_banks = {"fixed_bank": [], "continuation": []}
    continuation_stats = []
    for decision in decisions:
        branch_banks["fixed_bank"].append([list(option) for option in options])
        continued, stats = [], []
        for option in options:
            bank, info = continuation_bank(memory, episode, option, candidate + 1, decision,
                                            encoded=encoded, policy=continuation_policy)
            continued.append(list(bank))
            stats.append(info)
        branch_banks["continuation"].append(continued)
        continuation_stats.append(stats)

    if loss_fn is None:
        from .expert_v4 import expert_episode_flow_loss
        loss_fn = expert_episode_flow_loss
    fused_cache, scalar_cache = {}, {}

    def scalar(decision, bank, flow_seed):
        read_key = (decision, tuple(bank))
        key = (*read_key, flow_seed)
        if key not in scalar_cache:
            if read_key not in fused_cache:
                fused_cache[read_key] = read_bank(memory, episode, decision, bank, encoded)["fused_short"]
            result = loss_fn(head, episode, decision, fused_cache[read_key], seed=flow_seed)
            loss = result["loss"] if isinstance(result, dict) else result
            value = float(loss.detach().item() if isinstance(loss, torch.Tensor) else loss)
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite writer audit action loss")
            scalar_cache[key] = value
        return scalar_cache[key]

    for variant, banks_by_query in branch_banks.items():
        draws = {split: [[[scalar(d, bank, flow_seed) for bank in banks]
                           for flow_seed in seeds]
                          for d, banks, seeds in zip(decisions, banks_by_query, noise_plan[split])]
                 for split in ("A", "B")}
        summary = _summarize(draws, selected, deployed, float(tie_tolerance))
        summary.update(future_bank_ids=banks_by_query,
                       candidate_retained_at_queries=[[candidate in bank for bank in banks] for banks in banks_by_query])
        if variant == "continuation":
            summary["replay_stats"] = continuation_stats
        report[variant] = summary
    fixed, continued = report["fixed_bank"], report["continuation"]
    report["fixed_vs_continuation"] = {
        "best_option_A_agreement": fixed["best_option_A"] == continued["best_option_A"],
        "gain_sign_agreement_on_B": statistics.mean(
            a == b for a, b in zip(fixed["keep_relative_gain_signs"]["B"][1:],
                                  continued["keep_relative_gain_signs"]["B"][1:])),
        "fixed_A_best_gain_evaluated_under_continuation_B": continued["keep_relative_gains"]["B"][fixed["best_option_A"]],
        "continuation_A_best_gain_on_B": continued["A_best_keep_relative_gain_on_B"],
    }
    report["unique_expert_forwards"] = len(scalar_cache)
    report["nominal_expert_forwards_without_deduplication"] = 4 * len(decisions) * noise_samples * len(options)
    json.dumps(report, allow_nan=False)
    return report
