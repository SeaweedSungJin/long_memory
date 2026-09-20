"""Action Expert attention adaptation, isolated from the frozen-expert v3 route.

Only attention projection LoRA matrices are trainable: the original model is
never rewritten or thawed. FP32 adapters augment the BF16 projection result.
The flow bridge intentionally duplicates the small frozen bridge instead of
weakening its safety assertion for older experiments.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch import nn
from torch.nn import functional as F

from .hamlet import sample_noise_time


TARGET_PATTERN = re.compile(r"^model\.transformer_blocks\.\d+\.attn1\.(?:to_q|to_k|to_v|to_out\.0)$")


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0

    def __post_init__(self):
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive and finite")


class ExpertLoRALinear(nn.Module):
    """Zero-initialized low-rank update; base output is initially unchanged."""
    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("Expert LoRA requires a Linear projection")
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.scale = float(config.alpha) / config.rank
        self.enabled = True
        self.lora_A = nn.Parameter(torch.empty(config.rank, self.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, config.rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

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
        # Explicitly disable autocast: optimizer/master adapter weights stay FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            update = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scale
        return output + update.to(output.dtype)


def _adapters(head):
    return [(name, module) for name, module in head.named_modules() if isinstance(module, ExpertLoRALinear)]


def install_expert_lora(head, config: LoRAConfig, targets=None):
    """Wrap exactly all Q/K/V/output attention projections under ``head.model``.

    Explicit targets are a compatibility assertion, not a partial-selection API.
    The deterministic ordered list is serialized and checked again at inference.
    """
    if _adapters(head):
        raise ValueError("Action Expert LoRA is already installed")
    found = sorted(name for name, module in head.named_modules()
                   if TARGET_PATTERN.fullmatch(name) and isinstance(module, nn.Linear))
    if not found:
        raise ValueError("No supported Action Expert attention projections found")
    if targets is not None and (not isinstance(targets, (list, tuple)) or list(targets) != found):
        raise ValueError("Expert target list differs from the exact base attention projections")
    head.requires_grad_(False)
    for name in found:
        parent_name, child_name = name.rsplit(".", 1)
        parent = head.get_submodule(parent_name)
        setattr(parent, child_name, ExpertLoRALinear(getattr(parent, child_name), config))
    head.eval()
    return found


def expert_parameters(head):
    for _, module in _adapters(head):
        yield module.lora_A
        yield module.lora_B


def set_expert_trainable(head, trainable: bool):
    if type(trainable) is not bool:
        raise TypeError("trainable must be bool")
    if not _adapters(head):
        raise ValueError("Install Action Expert LoRA before selecting its trainability")
    head.requires_grad_(False)
    for parameter in expert_parameters(head):
        parameter.requires_grad_(trainable)
    # No dropout/state noise: stochasticity comes only from explicit flow noise.
    head.eval()


def expert_state_dict(head):
    return {f"{name}.{suffix}": getattr(module, suffix).detach().clone()
            for name, module in _adapters(head) for suffix in ("lora_A", "lora_B")}


def load_expert_state_dict(head, state, strict=True):
    expected = {f"{name}.{suffix}": getattr(module, suffix)
                for name, module in _adapters(head) for suffix in ("lora_A", "lora_B")}
    if not expected:
        raise ValueError("Install Action Expert LoRA before loading adapters")
    if strict and set(state) != set(expected):
        raise ValueError("Expert adapter tensor names do not match installed targets")
    # Validate the entire state before making any in-place change.
    for name, tensor in state.items():
        if name not in expected:
            if not strict:
                continue
            raise ValueError(f"Unexpected adapter: {name}")
        if tensor.shape != expected[name].shape or tensor.dtype != torch.float32:
            raise ValueError(f"Expert adapter shape/dtype mismatch: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite Expert adapter: {name}")
    with torch.no_grad():
        for name in set(state) & set(expected):
            expected[name].copy_(state[name])


def expert_state_sha256(head):
    digest = hashlib.sha256()
    for name, tensor in sorted(expert_state_dict(head).items()):
        value = tensor.cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def adapter_disabled(head):
    """Temporarily recover the original Expert; restore state even on failure."""
    saved = [(module, module.enabled) for _, module in _adapters(head)]
    try:
        for module, _ in saved:
            module.enabled = False
        yield
    finally:
        for module, enabled in saved:
            module.enabled = enabled


disable_expert_adapters = adapter_disabled


def expert_flow_loss(head, features, state, target, target_mask, attention_mask,
                     image_mask, embodiment_id, *, noise=None, time=None,
                     activation_checkpointing=False):
    """Original velocity objective with autograd for LoRA AND memory inputs."""
    allowed = {id(parameter) for parameter in expert_parameters(head)}
    if head.training or any(parameter.requires_grad and id(parameter) not in allowed
                            for parameter in head.parameters()):
        raise ValueError("Expert must be eval() with only LoRA adapter parameters trainable")
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

    trainable_adapter = any(parameter.requires_grad for parameter in expert_parameters(head))
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
    adapters = {id(parameter) for parameter in expert_parameters(head)}
    reference = next(parameter for parameter in head.parameters() if id(parameter) not in adapters)
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
