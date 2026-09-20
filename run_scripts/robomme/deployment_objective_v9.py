"""Experimental differentiable version of HAMLET's deployed Euler sampler.

This changes a training objective, not model architecture or inference. Initial
noise is explicit and independent of GT; all denoising steps remain in the graph.
Optional per-step recomputation bounds activation memory. Historical training
and evaluation entrypoints are deliberately left untouched.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from gr00t.long_memory.action_audit_v8 import cached_inputs
from gr00t.long_memory.expert_v4 import expert_parameters
from run_scripts.robomme.audit_archive_generation_v7 import prefix_masks


def validate_expert(head):
    allowed = {id(p) for p in expert_parameters(head)}
    if head.training or any(p.requires_grad and id(p) not in allowed for p in head.parameters()):
        raise ValueError("Only Expert LoRA may train; the Expert must remain eval()")
    steps = head.num_inference_timesteps
    if type(steps) is not int or not 1 <= steps <= 64:
        raise ValueError("Invalid deployed Euler step count")


def differentiable_euler(head, features, state_features, embodiment, backbone, noise,
                         *, activation_checkpointing=True):
    """Match get_action_with_features after its initial random draw.

Do not accept targets/masks here. The tensor ``noise`` is the initial trajectory,
not a teacher-conditioned interpolation. Each checkpoint receives its timestep
explicitly to prevent backward recomputation using the last loop iteration.
"""
    validate_expert(head)
    expected = (features.shape[0], head.config.action_horizon, head.action_dim)
    if (noise.shape != expected or noise.device != features.device or noise.dtype != features.dtype
            or not noise.is_floating_point() or not bool(torch.isfinite(noise).all())):
        raise ValueError("Initial noise must match finite deployed [B,H,A] shape/device/dtype")
    if type(activation_checkpointing) is not bool:
        raise ValueError("activation_checkpointing must be boolean")
    if backbone.get("mem_temb_add") is not None:
        raise ValueError("This experimental objective supports the current cross-attention HAMLET only")

    def velocity(actions, vl, state, timestep):
        action_features = head.action_encoder(actions, timestep, embodiment)
        if head.config.add_pos_embed:
            positions = torch.arange(action_features.shape[1], dtype=torch.long, device=vl.device)
            action_features = action_features + head.position_embedding(positions).unsqueeze(0)
        options = dict(hidden_states=torch.cat((state, action_features), dim=1),
                       encoder_hidden_states=vl, timestep=timestep, temb_add=None)
        if head.config.use_alternate_vl_dit:
            options.update(image_mask=backbone.image_mask,
                           backbone_attention_mask=backbone.backbone_attention_mask)
        output = head.model(**options)
        return head.action_decoder(output, embodiment)[:, -head.action_horizon:]

    actions = noise
    steps = head.num_inference_timesteps
    dt = 1. / steps
    needs_gradient = features.requires_grad or state_features.requires_grad or any(p.requires_grad for p in expert_parameters(head))
    for index in range(steps):
        timestep = torch.full((features.shape[0],), int(index / float(steps) * head.num_timestep_buckets),
                              device=features.device)
        if activation_checkpointing and torch.is_grad_enabled() and needs_gradient:
            predicted = checkpoint(velocity, actions, features, state_features, timestep,
                                   use_reentrant=False, preserve_rng_state=True)
        else:
            predicted = velocity(actions, features, state_features, timestep)
        actions = actions + dt * predicted
    if not bool(torch.isfinite(actions).all()):
        raise FloatingPointError("Nonfinite differentiable generated action")
    return actions


def generated_prefix_objective(head, episode, decision, fused_short=None, *, seed,
                               action_steps=16, activation_checkpointing=True):
    """Pure-noise generation followed by supervised observed-prefix error.

Future GT exists only on the right-hand side of the final loss. Neither READ,
past encoding, nor the denoising query receives it. No seed/state is persisted
on the inference head, so optimizer recomputation has no RNG side effects.
"""
    if type(seed) is not int or seed < 0:
        raise ValueError("Use an explicit nonnegative integer generation seed")
    features, state, embodiment, backbone = cached_inputs(head, episode, decision, fused_short)
    generator = torch.Generator(device=features.device).manual_seed(seed)
    noise = torch.randn((features.shape[0], head.config.action_horizon, head.action_dim),
                        device=features.device, dtype=features.dtype, generator=generator)
    state_features = head.state_encoder(state, embodiment)
    prediction = differentiable_euler(head, features, state_features, embodiment, backbone, noise,
                                      activation_checkpointing=activation_checkpointing)
    target = episode["targets"][decision].to(device=prediction.device, dtype=torch.float32)[None]
    target_mask = episode["target_mask"][decision].to(device=prediction.device)[None]
    if target.shape != prediction.shape:
        raise ValueError("Generated prediction and supervised target shapes differ")
    _, observed = prefix_masks(target_mask, episode["action_mask"][decision], action_steps)
    if not bool(observed.any()):
        raise ValueError("No observed-transition action prefix to supervise")
    difference = (prediction.float() - target)[observed]
    if not bool(torch.isfinite(difference).all()):
        raise FloatingPointError("Nonfinite observed-prefix error")
    loss = difference.square().mean()
    return {"loss": loss, "generated_prefix_mse": loss,
            "generated_prefix_mae": difference.abs().mean(), "prediction": prediction,
            "valid_values": int(observed.sum())}
