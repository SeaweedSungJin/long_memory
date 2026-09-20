"""Execution-prefix weighting of the unchanged HAMLET flow trajectory.

V19 tests a *loss reduction*, not a different diffusion input or sampler.
The original bridge still receives its complete valid GT action chunk. We
change only the weight on velocity errors after the deployed action prefix.
Do not pass a prefix-only target_mask to the original bridge: that would also
zero future GT actions before constructing its noisy trajectory.

The explicit per-query seed allows reconstruction of the bridge's exact noise
without another Expert forward or advancing global RNG. This is a RoboMME
objective: action coordinates 0..6 are joints, 7 is gripper, and >=8 are pads.
Metrics are normalized-coordinate velocity errors, never task success rates.
"""
from __future__ import annotations

import math

import torch

from gr00t.long_memory.expert_v4 import expert_episode_flow_loss, expert_parameters
from gr00t.long_memory.hamlet import sample_noise_time


OBJECTIVE_VERSION = "robomme_flow_prefix_weight_v19_v1"


def _masked_mean(value, mask):
    """Original denominator convention; empty diagnostic groups return zero."""
    selected = torch.where(mask, value, 0.0)
    return selected.sum() / (mask.float().sum() + 1e-6)


def episode_flow_v19(head, ep, decision, fused=None, *, seed, tail_weight=1.,
                     action_steps=16, activation_checkpointing=False):
    """Return original predictions and a separately prefix-weighted flow loss.

    ``tail_weight=1`` is an exact original-loss control, including its autograd
    object. At 0.25, valid coordinates in the first ``action_steps`` retain
    weight 1 and valid later coordinates get weight 0.25. Both arms retain all
    original valid targets in the trajectory, with the original noise/time.

    ``action_mask`` limits only *observed executed-prefix diagnostics*. It does
    not censor valid future targets or turn unobserved future GT into runtime
    memory inputs. Final partial prefixes and fully masked tail coordinates
    are supported. Every zero-denominator diagnostic also reports its count.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("V19 requires an explicit nonnegative integer flow seed")
    if (isinstance(tail_weight, bool) or not isinstance(tail_weight, (int, float))
            or not math.isfinite(tail_weight) or not 0 < tail_weight <= 1):
        raise ValueError("tail_weight must be finite in (0, 1]")
    if type(action_steps) is not int or action_steps <= 0:
        raise ValueError("action_steps must be a positive integer")
    if type(decision) is not int or not 0 <= decision < len(ep["targets"]):
        raise ValueError("Decision is outside the episode's action targets")
    target_row, mask_row = ep["targets"][decision], ep["target_mask"][decision]
    if (target_row.ndim != 2 or target_row.shape != mask_row.shape
            or mask_row.dtype != torch.bool or target_row.shape[1] < 8
            or action_steps > target_row.shape[0]):
        raise ValueError("Require matching [H,A>=8] RoboMME targets/boolean mask and prefix <= H")
    if bool(mask_row[:, 8:].any()):
        raise ValueError("RoboMME coordinates beyond joint7/gripper1 must be masked padding")
    observed_row = ep["action_mask"][decision]
    if observed_row.shape != (action_steps,) or observed_row.dtype != torch.bool:
        raise ValueError("Observed action_mask must be boolean [action_steps]")
    if not bool(mask_row.any()):
        raise ValueError("Action query has no valid targets")

    # Preserve every safety assertion, target/noise/time input, activation-
    # checkpoint path, and AE/memory gradient in the existing production bridge.
    original = expert_episode_flow_loss(head, ep, decision, fused, seed=seed,
        activation_checkpointing=activation_checkpointing)
    adapters = {id(p) for p in expert_parameters(head)}
    reference = next(p for p in head.parameters() if id(p) not in adapters)
    target = target_row.to(device=reference.device, dtype=reference.dtype)[None]
    valid = mask_row.to(device=reference.device)[None]
    noise, _ = sample_noise_time(head, target, seed)
    # Subtract in the reference dtype just as expert_flow_loss does, then cast
    # to FP32 for the error. Computing FP32 target-noise first changes BF16 math.
    clean_target = torch.where(valid, target, torch.zeros_like(target))
    velocity = clean_target - noise
    difference = torch.where(valid, original["prediction"].float() - velocity.float(), 0.0)
    squared, absolute = difference.square(), difference.abs()

    nominal = valid.clone()
    nominal[:, action_steps:] = False
    executed = nominal.clone()
    executed[:, :action_steps] &= observed_row.to(device=valid.device)[None, :, None]
    tail = valid & ~nominal
    weights = torch.where(nominal, 1.0, float(tail_weight)) * valid.float()
    # Avoid even floating-point reassociation of the original control objective.
    loss = original["loss"] if tail_weight == 1. else (
        (squared * weights).sum() / (weights.sum() + 1e-6))
    result = dict(original, loss=loss, action_loss=loss,
        original_flow_loss=original["loss"],
        nominal_prefix_flow_loss=_masked_mean(squared, nominal),
        executed_prefix_flow_loss=_masked_mean(squared, executed),
        tail_flow_loss=_masked_mean(squared, tail),
        weighted_velocity_mae=(absolute * weights).sum() / (weights.sum() + 1e-6),
        executed_prefix_velocity_mae=_masked_mean(absolute, executed),
        target_valid_values=valid.sum(), nominal_prefix_valid_values=nominal.sum(),
        executed_prefix_valid_values=executed.sum(), tail_valid_values=tail.sum(),
        weighted_valid_values=weights.sum())
    for name, part in (("joint", slice(0, 7)), ("gripper", slice(7, 8))):
        for prefix, mask in (("", valid), ("executed_prefix_", executed)):
            selected = mask[..., part]
            result[prefix + name + "_flow_loss"] = _masked_mean(squared[..., part], selected)
            result[prefix + name + "_velocity_mae"] = _masked_mean(absolute[..., part], selected)
            result[prefix + name + "_valid_values"] = selected.sum()
    return result
