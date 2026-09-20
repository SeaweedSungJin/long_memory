"""Optional full-rank residual adaptation of the existing AE output projector.

The frozen ``model.proj_out_2`` Linear is never overwritten. FP32 delta weight
and bias augment its native (normally BF16) output. This is NOT literal full-
weight finetuning: weight decay applies to the delta, and the original matmul
keeps its arithmetic. Enabled and control models have identical parameter
shapes; the disabled control bypasses the zero delta branch entirely.

Only the original flow-matching objective is used for V10 training. Generated
prefix metrics are a no-grad diagnostic, not a differentiable training bridge.
No original production module or its trainability guard is monkeypatched.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from gr00t.long_memory.expert_v4 import adapter_disabled, expert_parameters
from gr00t.long_memory.hamlet import sample_noise_time
from run_scripts.robomme.deployment_objective_v9 import generated_prefix_objective

TARGET = "model.proj_out_2"
KIND = "full_rank_residual_fp32"


class FullRankResidualLinear(nn.Module):
    """Frozen native Linear plus a zero-initialized FP32 full-rank residual."""

    def __init__(self, base, *, enabled=True):
        super().__init__()
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        if not isinstance(base, nn.Linear) or base.bias is None:
            raise TypeError("Projector adaptation requires an original Linear with bias")
        if not base.weight.is_floating_point() or not base.bias.is_floating_point():
            raise TypeError("Original projector parameters must be floating point")
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.enabled = enabled
        self.configured_enabled = enabled  # Baseline contexts only change .enabled.
        # zeros(), unlike a fresh Linear or LoRA A initialization, consumes no RNG.
        self.delta_weight = nn.Parameter(torch.zeros((self.out_features, self.in_features),
            dtype=torch.float32, device=base.weight.device), requires_grad=enabled)
        self.delta_bias = nn.Parameter(torch.zeros(self.out_features,
            dtype=torch.float32, device=base.weight.device), requires_grad=enabled)
        self.eval()

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        output = self.base(x)
        if not self.enabled:
            return output
        with torch.autocast(device_type=x.device.type, enabled=False):
            residual = F.linear(x.float(), self.delta_weight, self.delta_bias)
        return output + residual.to(output.dtype)


def _projector(head):
    try:
        module = head.get_submodule(TARGET)
    except AttributeError as exc:
        raise ValueError(f"Missing Action Expert projector {TARGET}") from exc
    if not isinstance(module, FullRankResidualLinear):
        raise ValueError("Install the V10 projector wrapper first (including disabled controls)")
    wrappers = [name for name, child in head.named_modules() if isinstance(child, FullRankResidualLinear)]
    if wrappers != [TARGET]:
        raise ValueError("Only the exact model.proj_out_2 may have a V10 projector adapter")
    return module


def projector_spec(head):
    module = _projector(head)
    return {"target": TARGET, "in_features": module.in_features, "out_features": module.out_features,
            "bias": True, "kind": KIND, "enabled": module.enabled}


def install_projector(head, enabled=True):
    """Install both candidate/control shapes without random draws or LoRA edits."""
    if type(enabled) is not bool:
        raise TypeError("enabled must be bool")
    if any(isinstance(module, FullRankResidualLinear) for module in head.modules()):
        raise ValueError("A V10 projector adapter is already installed")
    try:
        original = head.get_submodule(TARGET)
    except AttributeError as exc:
        raise ValueError(f"Missing Action Expert projector {TARGET}") from exc
    wrapper = FullRankResidualLinear(original, enabled=enabled)
    head.model.proj_out_2 = wrapper
    head.eval()
    return projector_spec(head)


def projector_parameters(head):
    module = _projector(head)
    yield module.delta_weight
    yield module.delta_bias


def projector_state_dict(head):
    module = _projector(head)
    return {name: getattr(module, name).detach().clone() for name in ("delta_weight", "delta_bias")}


def load_projector_state_dict(head, state):
    """Validate every key/tensor before mutation; disabled controls stay zero."""
    module = _projector(head)
    names = {"delta_weight", "delta_bias"}
    if not isinstance(state, Mapping) or set(state) != names:
        raise ValueError("Projector state must contain exactly delta_weight and delta_bias")
    for name in names:
        value, expected = state[name], getattr(module, name)
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError(f"Projector state requires dense tensor: {name}")
        if value.shape != expected.shape or value.dtype != torch.float32:
            raise ValueError(f"Projector shape/dtype mismatch: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite projector delta: {name}")
        if not module.enabled and bool(value.any()):
            raise ValueError("Disabled control projector delta must remain zero")
    with torch.no_grad():
        for name in names:
            getattr(module, name).copy_(state[name])


def set_trainable(head, train_projector: bool, train_lora: bool = True):
    """Select optimizer scope only; NEVER change the installed enabled policy."""
    if type(train_projector) is not bool or type(train_lora) is not bool:
        raise TypeError("Trainability flags must be bool")
    module = _projector(head)
    lora = list(expert_parameters(head))
    if train_lora and not lora:
        raise ValueError("Install the original attention LoRA before training it")
    if train_projector and not module.enabled:
        raise ValueError("A disabled projector control cannot train its bypassed delta")
    head.requires_grad_(False)
    for parameter in lora:
        parameter.requires_grad_(train_lora)
    for parameter in projector_parameters(head):
        parameter.requires_grad_(train_projector)
    head.eval()
    assert_expert_scope(head)


def trainable_expert_parameters(head):
    for parameter in (*expert_parameters(head), *projector_parameters(head)):
        if parameter.requires_grad:
            yield parameter


def assert_expert_scope(head):
    module = _projector(head)
    allowed = {id(p) for p in (*expert_parameters(head), *projector_parameters(head))}
    if any(child.training for child in head.modules()):
        raise ValueError("Action Expert and all dropout/stateful children must remain eval()")
    if any(p.requires_grad and id(p) not in allowed for p in head.parameters()):
        raise ValueError("Only attention LoRA and the exact output-projector delta may train")
    for parameter in projector_parameters(head):
        if parameter.dtype != torch.float32:
            raise ValueError("Projector delta master parameters must remain FP32")
    if not module.configured_enabled and any(p.requires_grad for p in projector_parameters(head)):
        raise ValueError("Disabled projector control has trainable bypassed parameters")
    if not module.configured_enabled and any(bool(p.detach().any()) for p in projector_parameters(head)):
        raise ValueError("Disabled control projector delta must remain zero")


@contextmanager
def all_adapters_disabled(head):
    """Recover the original Expert; nested contexts and exceptions restore flags."""
    module = _projector(head)
    enabled = module.enabled
    try:
        module.enabled = False
        with adapter_disabled(head):
            yield
    finally:
        module.enabled = enabled


def expert_flow_loss(head, features, state, target, target_mask, attention_mask,
                     image_mask, embodiment_id, *, noise=None, time=None,
                     activation_checkpointing=False):
    """Original v4 flow arithmetic, with a local LoRA+projector scope assertion."""
    assert_expert_scope(head)
    if (noise is None) != (time is None):
        raise ValueError("Pass both noise and time, or neither")
    if target.shape != target_mask.shape or target.ndim != 3:
        raise ValueError("Target and mask must have matching [B,H,A] shapes")
    valid = target_mask.bool()
    target = torch.where(valid, target, torch.zeros_like(target))
    if not torch.isfinite(target).all():
        raise FloatingPointError("Nonfinite valid action target")
    if noise is None:
        noise, time = sample_noise_time(head, target)
    if noise.shape != target.shape or time.shape != (target.shape[0], 1, 1):
        raise ValueError("Noise must be [B,H,A] and time [B,1,1]")
    if not torch.isfinite(noise).all() or not torch.isfinite(time).all():
        raise FloatingPointError("Nonfinite flow noise/time")
    trajectory = (1 - time) * noise + time * target
    velocity = target - noise
    timestep = (time[:, 0, 0] * head.num_timestep_buckets).long()
    state_features = head.state_encoder(state, embodiment_id)
    action_features = head.action_encoder(trajectory, timestep, embodiment_id)
    if head.config.add_pos_embed:
        positions = torch.arange(action_features.shape[1], device=features.device)
        action_features = action_features + head.position_embedding(positions)[None]
    sa = torch.cat([state_features, action_features], dim=1)

    def expert(vl):
        options = dict(hidden_states=sa, encoder_hidden_states=vl,
                       encoder_attention_mask=attention_mask, timestep=timestep,
                       return_all_hidden_states=True, temb_add=None)
        if head.config.use_alternate_vl_dit:
            options.update(image_mask=image_mask, backbone_attention_mask=attention_mask)
        out, _ = head.model(**options)
        return head.action_decoder(out, embodiment_id)[:, -target.shape[1]:]

    trainable_adapter = any(True for _ in trainable_expert_parameters(head))
    if activation_checkpointing and torch.is_grad_enabled() and (features.requires_grad or trainable_adapter):
        from torch.utils.checkpoint import checkpoint
        pred = checkpoint(expert, features, use_reentrant=False)
    else:
        pred = expert(features)
    mask = target_mask.float()
    if mask.sum().item() <= 0:
        raise ValueError("Cannot train/evaluate an action decision with no valid targets")
    difference = torch.where(valid, pred.float() - velocity.float(), 0.0)
    return {"loss": (difference.square() * mask).sum() / (mask.sum() + 1e-6),
            "velocity_mae": (difference.abs() * mask).sum() / (mask.sum() + 1e-6), "prediction": pred}


def expert_episode_flow_loss(head, episode, decision, fused_short=None, *, seed=None,
                             activation_checkpointing=False):
    excluded = {id(p) for p in (*expert_parameters(head), *projector_parameters(head))}
    reference = next(p for p in head.parameters() if id(p) not in excluded)
    device, dtype = reference.device, reference.dtype
    feature = episode["features"][decision].to(device=device, dtype=dtype)[None]
    if fused_short is not None:
        q = fused_short.shape[-2]
        feature = torch.cat([feature[:, :-q], fused_short.to(device=device, dtype=dtype)], dim=1)
    state = episode["state"][decision].to(device=device, dtype=dtype).reshape(1, 1, -1)
    target = episode["targets"][decision].to(device=device, dtype=dtype)[None]
    mask = episode["target_mask"][decision].to(device=device)[None]
    attention = episode["attention_masks"][decision].to(device=device, dtype=torch.bool)[None]
    images = episode["image_masks"][decision].to(device=device, dtype=torch.bool)[None]
    embodiment = torch.tensor([int(episode["embodiment_id"])], device=device, dtype=torch.long)
    noise, time = sample_noise_time(head, target, seed)
    return expert_flow_loss(head, feature, state, target, mask, attention, images, embodiment,
                            noise=noise, time=time, activation_checkpointing=activation_checkpointing)


@torch.no_grad()
def generated_prefix_metrics(head, episode, decision, fused_short=None, *, seed,
                             action_steps=16, activation_checkpointing=False):
    """Validation-only native V9 generation; never backpropagate this bridge.

The old generator's guard allows only trainable attention LoRA. Temporarily
freeze the projector parameters (NOT its enabled flag) within this no-grad
diagnostic, then restore every original requires_grad flag, including on error.
"""
    assert_expert_scope(head)
    parameters = list(projector_parameters(head))
    flags = [p.requires_grad for p in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        return generated_prefix_objective(head, episode, decision, fused_short, seed=seed,
            action_steps=action_steps, activation_checkpointing=activation_checkpointing)
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)
