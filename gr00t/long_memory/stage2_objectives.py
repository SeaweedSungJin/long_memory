"""Small, independently testable objectives for contextual Stage-2 v2 training.

This module does not change the memory architecture, the original trainer, or
online hard-write semantics. A utility target is already transformed by the
v2 labeler (``log1p(max(gain, 0) / scale)``); do not transform it a second time.
Writer inputs remain exactly detached [predicted utility, novelty].

Ambiguous write labels receive zero weight, NOT an implicit negative label.
Undefined statistics are omitted so scalar journals never contain NaN/Inf.
"""

import math
from numbers import Integral

import numpy as np
import torch
from torch.nn import functional as F

from .replay import encode_until, read_bank


def contextual_predictions(memory, episode, candidate, bank_ids):
    """Predict from the *same pre-write bank* used to construct the target.

    Event i includes its observed endpoint i+1, but the context is the short
    feature/state at i, as in the existing runtime writer. No event j >= i is
    permitted in the pre-write bank. One slot is reserved for the candidate;
    replacement-value targets for a full bank are deliberately not implied.
    """
    if not isinstance(candidate, Integral) or isinstance(candidate, bool):
        raise ValueError("candidate must be an integer event index")
    candidate = int(candidate)
    if not 0 <= candidate < len(episode["actions"]):
        raise ValueError("candidate out of bounds")
    if not bool(episode["transition_valid"][candidate]):
        raise ValueError("candidate is not a valid completed event")
    bank_ids = list(bank_ids)
    if any(not isinstance(i, Integral) or isinstance(i, bool) for i in bank_ids):
        raise ValueError("bank IDs must be integer event indices")
    if len(set(bank_ids)) != len(bank_ids):
        raise ValueError("bank IDs must be unique")
    if len(bank_ids) >= memory.config.capacity:
        raise ValueError("context bank must reserve capacity for the candidate")
    if any(i < 0 or i >= candidate or not bool(episode["transition_valid"][i]) for i in bank_ids):
        raise ValueError("context bank must contain valid strictly past events")
    encoded = encode_until(memory, episode, candidate + 1)
    result = read_bank(memory, episode, candidate, bank_ids, encoded)
    device = next(memory.parameters()).device
    index = torch.tensor(bank_ids, device=device, dtype=torch.long)
    short = episode["short"][candidate:candidate + 1].to(device=device, dtype=torch.float32)
    event = encoded["event"][candidate:candidate + 1]
    novelty = memory.novelty(
        encoded["keys"][candidate:candidate + 1], encoded["keys"][index][None],
        torch.ones((1, len(bank_ids)), device=device, dtype=torch.bool),
    )
    utility = memory.utility(event, short, result["read"])
    return {"utility": utility, "logits": memory.write_logits(utility, novelty),
            "novelty": novelty, "event": event, "short": short, "read": result["read"]}


def set_trainable_phase(memory, joint=False):
    """Bootstrap only the heads; joint phase also enables the reader path.

    Clearing frozen gradients prevents a phase change from accidentally applying
    a stale gradient. Optional reconstruction is excluded in both phases.
    """
    counts = {"heads": 0, "reader": 0, "frozen": 0}
    for name, parameter in memory.named_parameters():
        is_head = name.startswith(("utility_head.", "write_head."))
        enabled = is_head or (joint and not name.startswith("reconstruction."))
        parameter.requires_grad_(enabled)
        if not enabled:
            parameter.grad = None
        counts["heads" if enabled and is_head else "reader" if enabled else "frozen"] += parameter.numel()
    return counts


def parameter_groups(memory, head_lr=1e-4, reader_lr=1e-5):
    """Stable optimizer groups across bootstrap/joint and checkpoint resume.

    Include currently frozen reader parameters: Adam skips their None gradients
    during bootstrap, and they can subsequently train without rebuilding Adam.
    """
    if not all(math.isfinite(lr) and lr > 0 for lr in (head_lr, reader_lr)):
        raise ValueError("learning rates must be finite and positive")
    heads, reader = [], []
    for name, parameter in memory.named_parameters():
        if name.startswith("reconstruction."):
            continue
        (heads if name.startswith(("utility_head.", "write_head.")) else reader).append(parameter)
    return [{"params": heads, "lr": head_lr, "name": "heads"},
            {"params": reader, "lr": reader_lr, "name": "reader"}]


def _record_arrays(records):
    if not records:
        raise ValueError("a nonempty record batch is required")
    targets, truths, utility_weights, write_weights = [], [], [], []
    for row in records:
        target = float(row["utility_target"])
        uw = float(row.get("utility_weight", 1.0))
        ww = float(row.get("write_weight", 1.0))
        # A confidence-masked target may be None; it never becomes a negative
        # observation because its BCE contribution and diagnostic count are zero.
        raw_truth = row.get("write_target")
        if raw_truth is None and ww != 0:
            raise ValueError("a positive-weight write label cannot be missing")
        truth = 0.0 if raw_truth is None and ww == 0 else float(raw_truth)
        if not all(math.isfinite(x) and x >= 0 for x in (target, uw, ww)):
            raise ValueError("targets/weights must be finite and nonnegative")
        if truth not in (0.0, 1.0):
            raise ValueError("write_target must be binary (or None with zero weight)")
        targets.append(target)
        truths.append(truth)
        utility_weights.append(uw)
        write_weights.append(ww)
    return tuple(np.asarray(x, dtype=np.float64) for x in
                 (targets, truths, utility_weights, write_weights))


def _weighted_tensor_mean(values, weights):
    # clamp only the denominator: an all-ambiguous batch returns graph-connected
    # exact zero with zero writer gradient, never NaN or a manufactured negative.
    return (values * weights).sum() / weights.sum().clamp_min(torch.finfo(values.dtype).tiny)


def weighted_auxiliary_loss(utility, logits, records, utility_weight=1.0, write_weight=1.0):
    """SmoothL1(log-utility) plus confidence-weighted, normalized write BCE."""
    if utility.ndim != 1 or logits.shape != utility.shape or utility.numel() != len(records):
        raise ValueError("utility/logits must have shape [number of records]")
    if utility.device != logits.device:
        raise ValueError("utility and logits must be on the same device")
    if not bool(torch.isfinite(utility).all() & torch.isfinite(logits).all()):
        raise ValueError("predictions must be finite")
    if not all(math.isfinite(w) and w >= 0 for w in (utility_weight, write_weight)):
        raise ValueError("loss coefficients must be finite and nonnegative")
    # FP32 loss reduction even when the caller evaluates heads under autocast.
    utility, logits = utility.float(), logits.float()
    target, truth, uw, ww = [utility.new_tensor(x) for x in _record_arrays(records)]
    utility_loss = _weighted_tensor_mean(F.smooth_l1_loss(utility, target, reduction="none"), uw)
    write_loss = _weighted_tensor_mean(F.binary_cross_entropy_with_logits(logits, truth, reduction="none"), ww)
    zero_loss = _weighted_tensor_mean(F.smooth_l1_loss(torch.zeros_like(target), target, reduction="none"), uw)
    return {"loss": utility_weight * utility_loss + write_weight * write_loss,
            "utility_loss": utility_loss, "write_loss": write_loss,
            "zero_utility_loss": zero_loss, "write_weight_sum": ww.sum(),
            "utility_weight_sum": uw.sum()}


def _auc(truth, scores):
    """Mann-Whitney AUROC with average ranks for ties; None for one class."""
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    ranks = np.empty(len(order), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    return float((ranks[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _average_precision(truth, scores):
    """Threshold-grouped AP: identical scores give prevalence, not row order."""
    total = int(truth.sum())
    if not total:
        return None
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    cumulative = np.cumsum(truth[order])
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(order) - 1]
    tp = cumulative[ends]
    recall = tp / total
    return float(np.sum(np.diff(np.r_[0.0, recall]) * tp / (ends + 1)))


def _array(values, name):
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return result


def prediction_metrics(utilities, logits, records, threshold=0.5):
    """Global batch diagnostics, with no dependency on sklearn or SciPy.

    Classification/ranking uses only records with positive confidence weight;
    AUROC and AP count each such record once. Loss/accuracy use its actual
    weight. Report the count/coverage so a tiny confident subset cannot hide.
    ``write_prior_bce`` is an optimistic same-set constant-prior reference,
    NOT a classifier calibrated on the held-out set for later deployment.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    utility, logits = _array(utilities, "utilities"), _array(logits, "logits")
    target, truth, uw, ww = _record_arrays(records)
    if utility.shape != target.shape or logits.shape != target.shape:
        raise ValueError("prediction and record counts differ")
    errors, zero_errors = np.abs(utility - target), np.abs(target)
    huber = lambda x: np.where(x < 1, 0.5 * x * x, x - 0.5)
    result = {"label_count": float(len(records)), "confident_coverage": float(np.mean(ww > 0)),
              "utility_prediction_mean": float(utility.mean()), "utility_prediction_std": float(utility.std())}
    if uw.sum() > 0:
        result["utility_loss"] = float(np.sum(huber(errors) * uw) / uw.sum())
        result["zero_utility_loss"] = float(np.sum(huber(zero_errors) * uw) / uw.sum())
        result["utility_mae"] = float(np.sum(errors * uw) / uw.sum())
        result["utility_zero_gain"] = result["zero_utility_loss"] - result["utility_loss"]
    active = uw > 0
    if active.sum() > 1 and utility[active].std() > 1e-12 and target[active].std() > 1e-12:
        result["utility_corr"] = float(np.corrcoef(utility[active], target[active])[0, 1])
    # Numerically stable sigmoid/entropy for even extremely saturated logits.
    probability = np.exp(-np.logaddexp(0.0, -logits))
    prediction = probability >= threshold
    log_p = -np.logaddexp(0.0, -logits)
    log_not_p = -np.logaddexp(0.0, logits)
    result.update(write_probability_mean=float(probability.mean()),
                  write_probability_std=float(probability.std()),
                  write_probability_min=float(probability.min()),
                  write_probability_max=float(probability.max()),
                  write_entropy=float(np.mean(-probability * log_p - (1 - probability) * log_not_p)),
                  predicted_write_rate=float(prediction.mean()),
                  write_all_reject=float(not prediction.any()), write_all_accept=float(prediction.all()))
    mask = ww > 0
    truth, pred, weights = truth[mask].astype(bool), prediction[mask], ww[mask]
    result.update(confident_count=float(mask.sum()), confident_positive_count=float(truth.sum()),
                  confident_negative_count=float((~truth).sum()))
    if not mask.any():
        return result
    confidence_logits = logits[mask]
    result["write_loss"] = float(np.sum((np.logaddexp(0, confidence_logits) - confidence_logits * truth) * weights) / weights.sum())
    prevalence = float(np.sum(weights * truth) / weights.sum())
    result.update(write_accuracy=float(np.sum(weights * (pred == truth)) / weights.sum()),
                  positive_label_rate=prevalence, confident_predicted_write_rate=float(pred.mean()))
    prior_bce = 0.0
    if 0 < prevalence < 1:
        prior_bce = -prevalence * math.log(prevalence) - (1 - prevalence) * math.log1p(-prevalence)
    result["write_prior_bce"] = prior_bce
    if pred.any():
        result["write_precision"] = float(np.sum(weights * (pred & truth)) / np.sum(weights * pred))
    if truth.any():
        result["write_recall"] = float(np.sum(weights * (pred & truth)) / np.sum(weights * truth))
    if (~truth).any():
        result["write_specificity"] = float(np.sum(weights * (~pred & ~truth)) / np.sum(weights * ~truth))
    if truth.any() and (~truth).any():
        result["write_balanced_accuracy"] = (result["write_recall"] + result["write_specificity"]) / 2
    for name, score in (("utility", utility[mask]), ("write", confidence_logits)):
        auc, ap = _auc(truth, score), _average_precision(truth, score)
        if auc is not None:
            result[name + "_auc"] = auc
        if ap is not None:
            result[name + "_ap"] = ap
    return result


def readiness(metrics, min_auc=0.55, min_confident_per_class=8):
    """Conservative head-readiness gate, not a claim of simulator improvement.

    The trainer must separately check action performance against fixed-bank
    controls before adopting learned storage. More elapsed steps alone never
    satisfy this predicate. No write quota or forced positive loss is applied.
    """
    if not 0.5 <= min_auc <= 1 or min_confident_per_class < 1:
        raise ValueError("invalid readiness thresholds")
    reasons = []
    # External callers may load hand-edited/older journals; never let a NaN
    # comparison accidentally pass the readiness guard.
    required = ("confident_positive_count", "confident_negative_count", "utility_auc",
                "write_auc", "utility_loss", "zero_utility_loss", "confident_predicted_write_rate")
    for name in required:
        value = metrics.get(name)
        if value is not None and not math.isfinite(float(value)):
            return False, [f"non-finite readiness metric: {name}"]
    for label in ("positive", "negative"):
        if metrics.get(f"confident_{label}_count", 0) < min_confident_per_class:
            reasons.append(f"insufficient confident {label} labels")
    for name in ("utility_auc", "write_auc"):
        if metrics.get(name, -math.inf) < min_auc:
            reasons.append(f"{name} below {min_auc:g}")
    if not metrics.get("utility_loss", math.inf) < metrics.get("zero_utility_loss", -math.inf):
        reasons.append("utility does not beat the zero predictor")
    fraction = metrics.get("confident_predicted_write_rate", 0.0)
    if not 0 < fraction < 1:
        reasons.append("writer predicts only one class on confident labels")
    return not reasons, reasons


def fixed_bank(episode, decision, budget, policy="fifo", seed=0):
    """Causal size-matched first/FIFO/random controls, returned chronologically.

    The random control is reservoir-equivalent at one queried decision. It is
    deterministic for a fixed seed; it is not a separately learned policy.
    """
    if not isinstance(decision, Integral) or isinstance(decision, bool) or not 0 <= decision <= len(episode["actions"]):
        raise ValueError("decision out of bounds")
    if not isinstance(budget, Integral) or isinstance(budget, bool) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if policy not in ("first", "fifo", "random"):
        raise ValueError("fixed policy must be first, fifo, or random")
    eligible = [i for i in range(decision) if bool(episode["transition_valid"][i])]
    if budget == 0:
        return []
    if policy == "first":
        return eligible[:budget]
    if policy == "fifo":
        return eligible[-budget:]
    if len(eligible) <= budget:
        return eligible
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(eligible, size=budget, replace=False))
