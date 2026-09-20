"""Noise-paired read-time utility labels and uncertainty-aware CVOM objectives.

Utility is SAME adapted Expert loss(null) minus loss(candidate), not robot
success probability. Draws must use identical flow noise/time across choices;
the trainer supplies that pairing. Tiny differences are never forced into an
argmax class label. Only confident pair differences enter ranking supervision.
"""

import math

import torch
from torch.nn import functional as F

from .core_v6 import CANDIDATE_NAMES


def paired_utility_targets(loss_draws, null_index=3, scale=0.001, margin=1e-5, uncertainty_z=2.0):
    draws = torch.as_tensor(loss_draws).detach().float()
    if draws.ndim != 2 or draws.shape[0] < 1 or draws.shape[1] < 2:
        raise ValueError("loss_draws must be [noise_draws, candidates]")
    if not bool(torch.isfinite(draws).all()) or bool((draws < 0).any()):
        raise ValueError("Teacher losses must be finite and nonnegative")
    if type(null_index) is not int or not 0 <= null_index < draws.shape[1]:
        raise ValueError("Invalid null candidate index")
    for name, value in (("scale", scale), ("margin", margin), ("uncertainty_z", uncertainty_z)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (name == "scale" and value == 0):
            raise ValueError(f"Invalid {name}")
    utility = draws[:, null_index:null_index + 1] - draws
    raw_gain = utility.mean(0)
    count = len(draws)
    # One draw cannot estimate uncertainty: retain calibrated regression but
    # reject every ranking claim rather than pretending SE=0 means certainty.
    stderr = utility.std(0, unbiased=True) / math.sqrt(count) if count > 1 else torch.full_like(raw_gain, float("inf"))
    if count == 1:
        stderr[null_index] = 0
    pair_draws = utility[:, :, None] - utility[:, None, :]
    pair_mean = pair_draws.mean(0)
    pair_se = pair_draws.std(0, unbiased=True) / math.sqrt(count) if count > 1 else torch.full_like(pair_mean, float("inf"))
    pair_mask = pair_mean.abs() > margin + uncertainty_z * pair_se if count > 1 else torch.zeros_like(pair_mean, dtype=torch.bool)
    pair_mask = pair_mask & torch.triu(torch.ones_like(pair_mask), diagonal=1)
    nonnull = torch.arange(draws.shape[1], device=draws.device) != null_index
    # Reliable nonzero-vs-null gain is tracked separately from pair ranks.
    gain_mask = ((raw_gain.abs() > margin + uncertainty_z * stderr) & nonnull) if count > 1 else torch.zeros_like(nonnull)
    return {"gain": raw_gain / scale, "raw_gain": raw_gain, "stderr": stderr,
            "pair_sign": pair_mean.sign(), "pair_mask": pair_mask,
            "gain_mask": gain_mask, "null_index": null_index, "scale": float(scale),
            "mean_losses": draws.mean(0), "noise_samples": count,
            "signal_fraction": float(pair_mask.sum()) / max(draws.shape[1] * (draws.shape[1] - 1) // 2, 1)}


def cvom_loss(predicted_gain, targets, ranking_weight=0.1):
    """Signed gain regression + confident-pair ranking (no fake best class).

    Regression includes uncertain near-zero means so the scorer learns neutral
    output rather than only positive-selected labels. Signal fraction refers to
    ranking confidence, not a request to skip all near-zero regression batches.
    """
    pred = predicted_gain.float()
    expected = torch.as_tensor(targets["gain"], device=pred.device, dtype=torch.float32)
    if pred.ndim != 1 or pred.shape != expected.shape or not bool(torch.isfinite(pred).all()):
        raise ValueError("Predicted utility must match finite candidate target vector")
    if not math.isfinite(ranking_weight) or ranking_weight < 0:
        raise ValueError("ranking_weight must be finite and nonnegative")
    mask = torch.arange(len(pred), device=pred.device) != targets["null_index"]
    regression = F.smooth_l1_loss(pred[mask], expected[mask])
    pair_mask = targets["pair_mask"].to(pred.device)
    pair_sign = targets["pair_sign"].to(pred.device)
    differences = pred[:, None] - pred[None, :]
    ranking = F.softplus(-differences[pair_mask] * pair_sign[pair_mask]).mean() if bool(pair_mask.any()) else pred.sum() * 0
    return {"loss": regression + ranking_weight * ranking, "regression": regression,
            "ranking": ranking, "signal_fraction": targets["signal_fraction"],
            "confident_pairs": int(pair_mask.sum())}


def select_candidate(predicted_gain, improvement_margin=0.0, fallback="uniform"):
    """Use CVOM only when predicted gain exceeds the fixed fallback by margin.

    Scores are normalized signed gains. Margin is a configured hyperparameter,
    NOT a calibrated confidence interval. Null may win when every pack harms
    the teacher according to CVOM; ties deterministically keep the fallback.
    """
    scores = torch.as_tensor(predicted_gain).detach().float()
    if scores.shape != (len(CANDIDATE_NAMES),) or not bool(torch.isfinite(scores).all()):
        raise ValueError("Expected four finite CVOM candidate scores")
    if fallback not in CANDIDATE_NAMES or not math.isfinite(improvement_margin) or improvement_margin < 0:
        raise ValueError("Invalid fallback/improvement_margin")
    if float(scores[-1]) != 0.0:
        raise ValueError("Null utility must be anchored to exact zero")
    default = CANDIDATE_NAMES.index(fallback)
    best = int(scores.argmax())
    return CANDIDATE_NAMES[best] if float(scores[best] - scores[default]) > improvement_margin else fallback
