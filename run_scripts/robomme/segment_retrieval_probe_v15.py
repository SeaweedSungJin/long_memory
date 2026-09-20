"""Isolated Q/K segment-retrieval probe; NOT a writer, action model or policy.

This module exposes the *past* attention distribution of the unchanged V13
visual architecture (also used by V14). It never calls the value projection,
current-reference attention, visual READ/output or Action Expert. Supervised
segment labels enter only the loss/metric helpers, after probabilities exist.

Call ``clone_qk_probe`` before training so the original model stays untouched.
Rebuild the current observation after every optimizer update: its ``queries``
already contain the current query_projection. Past addresses can be reused
only while their frozen encoder and original observed inputs remain unchanged.
No checkpoint loading, optimizer, data selection or training loop lives here.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

import torch

from run_scripts.robomme.visual_demo_tail_bank_v13 import VisualDemoTailMemoryV13
from run_scripts.robomme.visual_differential_memory_v12 import DifferentialObservation
from run_scripts.robomme.visual_patch_memory_v11 import PATCHES_PER_OBSERVATION

QK_PARAMETER_NAMES = ("query_projection.weight", "key_projection.weight")


def _memory(memory):
    if not isinstance(memory, VisualDemoTailMemoryV13):
        raise TypeError("Expected the actual VisualDemoTailMemoryV13 architecture")
    if memory.read_mode != "differential":
        raise ValueError("The past-retrieval probe requires differential read_mode")
    memory._device()  # The existing architecture requires FP32 master weights.
    parameters = dict(memory.named_parameters())
    if not set(QK_PARAMETER_NAMES) <= parameters.keys():
        raise ValueError("Missing original query/key projection weights")
    return parameters


def configure_qk_only(memory):
    """IN PLACE: freeze/clear all visual parameters, enable only original Q/K.

    Use this only on a separately owned probe copy. Returns the two parameters
    in canonical Q/K order for an optimizer; no optimizer is created here.
    Frozen encoder, time/spatial identities, normalizations, V and output are
    not trained. Clearing stale gradients prevents accidental accumulation.
    """
    parameters = _memory(memory)
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in QK_PARAMETER_NAMES)
        parameter.grad = None
    assert_qk_scope(memory)
    return tuple(parameters[name] for name in QK_PARAMETER_NAMES)


def clone_qk_probe(memory):
    """Independent copy with identical weights; consumes no initialization RNG."""
    _memory(memory)
    probe = copy.deepcopy(memory)
    configure_qk_only(probe)
    return probe


def snapshot_frozen_parameters(memory):
    """Capture the other visual parameters as independent CPU tensors."""
    return {name: parameter.detach().cpu().clone()
            for name, parameter in _memory(memory).items()
            if name not in QK_PARAMETER_NAMES}


def assert_qk_scope(memory, frozen_snapshot=None, *, require_gradients=False):
    """Fail if scope/gradients differ or any snapshotted frozen weight changed.

    ``require_gradients`` requires finite connected Q/K gradients, not nonzero
    gradients: symmetric/degenerate inputs can legitimately have zero values.
    The caller must give an optimizer only ``configure_qk_only``'s parameters.
    """
    parameters = _memory(memory)
    frozen_names = set(parameters) - set(QK_PARAMETER_NAMES)
    if frozen_snapshot is not None:
        if not isinstance(frozen_snapshot, Mapping) or set(frozen_snapshot) != frozen_names:
            raise ValueError("Frozen snapshot must contain exactly the non-Q/K parameter names")
    for name, parameter in parameters.items():
        trainable = name in QK_PARAMETER_NAMES
        if parameter.requires_grad is not trainable:
            raise ValueError(f"Q/K-only requires_grad scope differs: {name}")
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(f"Nonfinite visual parameter: {name}")
        if not trainable and parameter.grad is not None:
            raise ValueError(f"Frozen parameter acquired a gradient: {name}")
        if trainable and parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"Nonfinite Q/K gradient: {name}")
        if trainable and require_gradients and parameter.grad is None:
            raise ValueError(f"Disconnected Q/K gradient: {name}; rebuild the current observation")
        if not trainable and frozen_snapshot is not None:
            expected = frozen_snapshot[name]
            if (not isinstance(expected, torch.Tensor) or expected.shape != parameter.shape
                    or expected.dtype != parameter.dtype
                    or not torch.equal(parameter.detach().cpu(), expected.detach().cpu())):
                raise ValueError(f"Frozen visual weight changed: {name}")
    return True


def past_frame_probabilities(memory, observation, bank):
    """Differentiable FP32 [B,N] past-frame probabilities, without labels.

    For each head and each of the 162 current patch queries, use exactly the
    existing Q, K and scale 1/sqrt(head_dim), with all 162 keys per valid past
    observation. Sum keys within each frame; average heads and query patches.
    This is the same softmax algebra as the native SDPA past READ, not a claim
    of bitwise equality across fused attention kernels. No extra renormalizing,
    temperature, top-k, current-self subtraction or learned pooling is added.

    Invalid whole-frame padding has probability exactly zero (padded NaNs are
    neutralized before K). Every row must have at least one strictly past frame;
    empty rows have no retrieval task and are rejected, not assigned fake mass.
    Existing bank validation enforces unique chronological strict-past frames.
    Inputs/banks are not mutated; no encoding or APPEND occurs here.
    """
    _memory(memory)
    if not isinstance(observation, DifferentialObservation):
        raise TypeError("Expected an original DifferentialObservation")
    device, config = memory._device(), memory.config
    features = observation.features
    if (not isinstance(features, torch.Tensor) or features.ndim != 3 or features.shape[0] < 1
            or features.shape[2] != config.feature_dim or features.device != device):
        raise ValueError("Current features must have original [B,S,D] shape/device")
    batch = features.shape[0]
    for name, dtype in (("frames", torch.long), ("is_demo", torch.bool)):
        value = getattr(observation, name)
        if (not isinstance(value, torch.Tensor) or value.shape != (batch,)
                or value.dtype != dtype or value.device != device):
            raise ValueError(f"Current {name} has incorrect shape/dtype/device")
    if bool((observation.frames < 0).any()):
        raise ValueError("Current frames must be nonnegative")
    expected_shape = (batch, PATCHES_PER_OBSERVATION, config.hidden_dim)
    for name in ("encoded", "queries"):
        value = getattr(observation, name)
        if (not isinstance(value, torch.Tensor) or value.shape != expected_shape
                or value.dtype != torch.float32 or value.device != device
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Current {name} must be finite FP32 [B,162,H]")
    bank = memory._bank(bank, observation)
    if not bool(bank.valid.any(1).all()):
        raise ValueError("Retrieval requires at least one valid past frame in every row")
    count = bank.tokens.shape[1]
    with torch.autocast(device_type=device.type, enabled=False):
        key_mask = bank.valid.repeat_interleave(PATCHES_PER_OBSERVATION, dim=1)
        addresses = torch.where(key_mask[..., None], bank.tokens.flatten(1, 2), 0.)
        head_dim = config.hidden_dim // config.num_heads

        def heads(tensor):
            return tensor.reshape(batch, -1, config.num_heads, head_dim).transpose(1, 2)

        query = heads(observation.queries)
        key = heads(memory.key_projection(addresses))
        scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim ** -.5)
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("Nonfinite Q/K retrieval scores")
        weights = scores.masked_fill(~key_mask[:, None, None, :], -torch.inf).softmax(dim=-1)
        probabilities = weights.reshape(batch, config.num_heads, PATCHES_PER_OBSERVATION,
                                        count, PATCHES_PER_OBSERVATION).sum(-1).mean((1, 2))
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError("Nonfinite frame retrieval probabilities")
    return probabilities


def _segment_inputs(probabilities, positive_mask, valid_mask):
    if (not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 2
            or min(probabilities.shape) < 1 or probabilities.dtype != torch.float32
            or not bool(torch.isfinite(probabilities).all())
            or bool((probabilities < 0).any()) or bool((probabilities > 1).any())):
        raise ValueError("Probabilities must be finite FP32 [B,N] values in [0,1]")
    for name, value in (("positive_mask", positive_mask), ("valid_mask", valid_mask)):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.bool
                or value.shape != probabilities.shape or value.device != probabilities.device):
            raise ValueError(f"{name} must be bool with the probabilities' shape/device")
    if bool((positive_mask & ~valid_mask).any()):
        raise ValueError("A positive frame cannot be invalid padding")
    counts, positives = valid_mask.sum(1), positive_mask.sum(1)
    if bool((counts == 0).any()):
        raise ValueError("Empty retrieval rows are not a supervised task")
    if bool((positives == 0).any()):
        raise ValueError("Every row requires a positive frame")
    if bool((positives == counts).any()):
        raise ValueError("All-positive rows are a trivial retrieval task and are rejected")
    if bool((probabilities[~valid_mask] != 0).any()):
        raise ValueError("Invalid frames must have exactly zero probability")
    # Only account for FP32 reduction roundoff; never renormalize a bad input.
    tolerance = 8 * torch.finfo(torch.float32).eps
    if not torch.allclose(probabilities.sum(1), torch.ones_like(probabilities[:, 0]),
                          rtol=tolerance, atol=tolerance):
        raise ValueError("Frame probabilities must sum to one per row")
    mass = probabilities.masked_fill(~positive_mask, 0.).sum(1)
    if bool((mass <= 0).any()):
        raise FloatingPointError("Positive mass underflowed to zero; no silent log-floor is applied")
    return mass, positives.to(torch.float32) / counts, counts, positives


def segment_mass_loss(probabilities, positive_mask, valid_mask, *, reduction="mean"):
    """-log(sum probability on positive frames), in nats; labels only here.

    No positive-span-size weighting is applied. Uniform expected mass depends
    on each row's valid-frame count and is reported separately in metrics.
    Zero positive mass fails explicitly instead of hiding it behind an epsilon.
    """
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be none, mean or sum")
    mass, _, _, _ = _segment_inputs(probabilities, positive_mask, valid_mask)
    losses = -mass.log()
    return losses if reduction == "none" else getattr(losses, reduction)()


def retrieval_metrics(probabilities, positive_mask, valid_mask):
    """Detached per-row metrics; these are retrieval labels, not action success.

    log_score_gain = log(positive_mass / uniform_positive_mass), in nats.
    Top-1 uses the first bank index when probabilities tie, as torch.argmax does.
    A top-1 hit therefore must not be mistaken for a uniform random-hit estimate.
    """
    mass, uniform, counts, positives = _segment_inputs(probabilities, positive_mask, valid_mask)
    top = probabilities.masked_fill(~valid_mask, -torch.inf).argmax(1, keepdim=True)
    values = {"positive_mass": mass, "uniform_positive_mass": uniform,
              "nll": -mass.log(), "uniform_nll": -uniform.log(),
              "log_score_gain": mass.log() - uniform.log(),
              "top1_span_hit": positive_mask.gather(1, top).squeeze(1).float(),
              "valid_frame_count": counts, "positive_frame_count": positives}
    return {name: value.detach() for name, value in values.items()}


def content_following_positive_mask(positive_mask, source_indices, valid_mask):
    """Labels for a content permutation; does NOT modify content or timestamps.

    ``source_indices[b,j]`` names the original frame whose image content is put
    at destination j by the caller. Only valid frames may be permuted; padding
    must stay at its own index. The runner separately keeps raw times sorted and
    recomputes original image/address encodings. This helper cannot certify that
    an external content intervention actually followed the supplied mapping.
    """
    if (not isinstance(valid_mask, torch.Tensor) or valid_mask.ndim != 2
            or valid_mask.dtype != torch.bool or min(valid_mask.shape) < 1):
        raise ValueError("valid_mask must be nonempty bool [B,N]")
    if (not isinstance(positive_mask, torch.Tensor) or positive_mask.shape != valid_mask.shape
            or positive_mask.dtype != torch.bool or positive_mask.device != valid_mask.device
            or bool((positive_mask & ~valid_mask).any())):
        raise ValueError("positive_mask must match valid frames")
    if (not isinstance(source_indices, torch.Tensor) or source_indices.shape != valid_mask.shape
            or source_indices.dtype != torch.long or source_indices.device != valid_mask.device):
        raise ValueError("source_indices must be int64 [B,N] on the mask device")
    index = torch.arange(valid_mask.shape[1], device=valid_mask.device)
    if (not torch.equal(source_indices.sort(1).values, index[None].expand_as(source_indices))
            or not torch.equal(source_indices[~valid_mask], index[None].expand_as(source_indices)[~valid_mask])):
        raise ValueError("Content mapping must be a permutation of valid frames with fixed padding")
    return positive_mask.gather(1, source_indices)
