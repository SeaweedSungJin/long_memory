"""Opt-in visual-memory cross-attention inside the otherwise unchanged Expert.

Unlike v4/v5, this bridge does NOT replace HAMLET's short conditioning tokens.
Independent residual adapters run after selected DiT blocks and receive the
multi-token memory selected by v6. Their output projection starts at zero, so
installation is initially an exact identity (reader gradients begin after the
first adapter-output update). No-memory is an unconditional exact bypass.

Hooks are inference-only by default: memory exists only within ``with_memory``.
The flow helper supplies memory as an explicit activation-checkpoint input and
re-enters that context INSIDE its closure, including backward recomputation.
Do not rely on an external context surviving until ``loss.backward()``.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn
from torch.nn import functional as F

from .expert_v4 import expert_parameters
from .hamlet import sample_noise_time


# Context-local immutable stack: independent heads, nesting, exceptions and
# concurrent inference tasks cannot overwrite each other's selected memory.
_MEMORY_CONTEXT = ContextVar("long_memory_v6_context", default=())


class ExpertMemoryCrossAttention(nn.Module):
    """FP32 compact attention; return the frozen Expert's original dtype."""

    def __init__(self, expert_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        if any(type(value) is not int or value <= 0
               for value in (expert_dim, hidden_dim, num_heads)):
            raise ValueError("Expert/memory dimensions and heads must be positive integers")
        if hidden_dim % num_heads:
            raise ValueError("Memory hidden dimension must be divisible by num_heads")
        self.expert_dim = expert_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.query_norm = nn.LayerNorm(expert_dim)
        self.query = nn.Linear(expert_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, expert_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, hidden, memory):
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("Expert block must return a Tensor [B,L,D]")
        if hidden.shape[-1] != self.expert_dim:
            raise ValueError("Expert block hidden width changed after bridge installation")
        if memory is None:
            return hidden
        if not isinstance(memory, torch.Tensor) or memory.ndim != 3:
            raise ValueError("Memory must have shape [B,T,H]")
        if memory.shape[-1] != self.hidden_dim or memory.shape[0] not in (1, hidden.shape[0]):
            raise ValueError("Memory batch/hidden dimension does not match the Expert bridge")
        if not memory.is_floating_point():
            raise ValueError("Memory tokens must be floating-point")
        if memory.shape[1] == 0:
            return hidden
        if memory.device != hidden.device:
            raise ValueError("Memory and Expert hidden states must be on the same device")
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            query = self.query(self.query_norm(hidden.float()))
            # Core normalizes BEFORE soft retrieval weighting. Per-token
            # LayerNorm here would cancel those positive weights and starve
            # the reader query/key scorer of a meaningful action gradient.
            key, value = self.key(memory.float()), self.value(memory.float())
            if memory.shape[0] == 1 and hidden.shape[0] != 1:
                key = key.expand(hidden.shape[0], -1, -1)
                value = value.expand(hidden.shape[0], -1, -1)
            def split(tensor):
                return tensor.reshape(tensor.shape[0], tensor.shape[1], self.num_heads,
                                      self.hidden_dim // self.num_heads).transpose(1, 2)
            read = F.scaled_dot_product_attention(split(query), split(key), split(value),
                                                  dropout_p=0.0)
            read = read.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], self.hidden_dim)
            residual = self.output(read)
        return hidden + residual.to(hidden.dtype)


def _bridge(head):
    result = getattr(head, "long_memory_v6_bridge", None)
    if not isinstance(result, nn.ModuleDict):
        raise ValueError("Install the v6 Expert memory bridge first")
    return result


def _block_width(model, block):
    width = getattr(block, "dim", None)
    if width is None:
        shape = getattr(getattr(block, "norm3", None), "normalized_shape", None)
        if shape:
            width = shape[-1]
    if width is None:
        width = getattr(model, "inner_dim", None)
    if type(width) is not int or width <= 0:
        raise ValueError("Cannot infer the selected Expert block's hidden width")
    model_width = getattr(model, "inner_dim", width)
    if model_width != width:
        raise ValueError("Expert model and selected block widths disagree")
    return width


def install_memory_bridge(head, hidden_dim=256, block_indices=(7, 23), num_heads=4,
                          expert_dim=None):
    """Attach isolated post-block modules, keeping old weights/LoRA untouched.

    Install v4 LoRA first if desired. Later v4 ``set_expert_trainable`` freezes
    EVERY non-LoRA parameter, so callers must re-enable bridge parameters after
    using that helper. Checkpoints save the returned exact architecture record.
    """
    if hasattr(head, "long_memory_v6_bridge"):
        raise ValueError("v6 Expert memory bridge is already installed")
    model = getattr(head, "model", None)
    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise ValueError("Expected head.model.transformer_blocks ModuleList")
    if not isinstance(block_indices, (list, tuple)) or not block_indices:
        raise ValueError("At least one Expert block index is required")
    if any(type(index) is not int or not 0 <= index < len(blocks) for index in block_indices):
        raise ValueError("Expert block indices must be in-range integers")
    if len(set(block_indices)) != len(block_indices):
        raise ValueError("Expert block indices must be unique")
    widths = [_block_width(model, blocks[index]) for index in block_indices]
    if len(set(widths)) != 1:
        raise ValueError("Selected Expert blocks must have the same hidden width")
    if expert_dim is not None and (type(expert_dim) is not int or expert_dim != widths[0]):
        raise ValueError("Saved Expert dimension does not match the actual base model")
    reference = next(model.parameters())
    # Construct/validate fully before attaching hooks or changing trainability.
    adapters = nn.ModuleDict({str(index): ExpertMemoryCrossAttention(widths[0], hidden_dim,
                                                                   num_heads)
                             for index in block_indices}).to(device=reference.device, dtype=torch.float32)
    allowed = {id(parameter) for parameter in expert_parameters(head)}
    for parameter in head.parameters():
        if id(parameter) not in allowed:
            parameter.requires_grad_(False)
    head.add_module("long_memory_v6_bridge", adapters)
    handles = []
    identity = id(head)
    def hook_for(adapter):
        def hook(_module, _args, output):
            memory = next((value for key, value in reversed(_MEMORY_CONTEXT.get())
                           if key == identity), None)
            return adapter(output, memory)
        return hook
    try:
        for index in block_indices:
            handles.append(blocks[index].register_forward_hook(hook_for(adapters[str(index)])))
    except Exception:
        for handle in handles:
            handle.remove()
        delattr(head, "long_memory_v6_bridge")
        raise
    head._long_memory_v6_hook_handles = handles
    head.eval()
    config = dict(hidden_dim=hidden_dim, block_indices=list(block_indices), num_heads=num_heads,
                  expert_dim=widths[0])
    head._long_memory_v6_bridge_config = config
    return dict(config)


def bridge_parameters(head):
    yield from _bridge(head).parameters()


def memory_bridge_shapes(config):
    """Validate serialized architecture and obtain tensor shapes without a model."""
    if not isinstance(config, dict) or set(config) != {
            "hidden_dim", "block_indices", "num_heads", "expert_dim"}:
        raise ValueError("Invalid memory bridge config fields")
    indices = config["block_indices"]
    if not isinstance(indices, (list, tuple)) or not indices or any(
            type(index) is not int or index < 0 for index in indices):
        raise ValueError("Invalid memory bridge block indices")
    if len(set(indices)) != len(indices):
        raise ValueError("Memory bridge block indices must be unique")
    with torch.device("meta"):
        modules = nn.ModuleDict({str(index): ExpertMemoryCrossAttention(
            config["expert_dim"], config["hidden_dim"], config["num_heads"])
            for index in indices})
    return {name: tuple(value.shape) for name, value in modules.state_dict().items()}


def bridge_state_dict(head):
    return {key: value.detach().clone() for key, value in _bridge(head).state_dict().items()}


def load_bridge_state_dict(head, state):
    """Validate all names/shapes/finite FP32 tensors before changing anything."""
    expected = _bridge(head).state_dict()
    if set(state) != set(expected):
        raise ValueError("Expert memory bridge tensor names do not match installed targets")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape or value.dtype != torch.float32:
            raise ValueError(f"Expert memory bridge shape/dtype mismatch: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite Expert memory bridge tensor: {key}")
    _bridge(head).load_state_dict(state, strict=True)


@contextmanager
def with_memory(head, tokens):
    """Temporarily provide memory for all diffusion calls; nesting is safe."""
    _bridge(head)
    token = _MEMORY_CONTEXT.set(_MEMORY_CONTEXT.get() + ((id(head), tokens),))
    try:
        yield
    finally:
        _MEMORY_CONTEXT.reset(token)


def v6_flow_loss(head, features, state, target, target_mask, attention_mask,
                 image_mask, embodiment_id, memory_tokens=None, *, noise=None,
                 time=None, activation_checkpointing=False):
    """Original flow objective with direct visual memory and optional AE LoRA.

    Only bridge and LoRA parameters may be trainable. Cached feature conditioning
    is not rewritten. An empty memory tensor has the same exact bypass as None.
    """
    bridge = _bridge(head)
    allowed = {id(parameter) for parameter in expert_parameters(head)}
    allowed.update(id(parameter) for parameter in bridge.parameters())
    if head.training or any(parameter.requires_grad and id(parameter) not in allowed
                            for parameter in head.parameters()):
        raise ValueError("Expert must be eval() with only LoRA/bridge parameters trainable")
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
    mask = target_mask.float()
    if mask.sum().item() <= 0:
        raise ValueError("Cannot train/evaluate an action decision with no valid targets")
    trajectory = (1 - time) * noise + time * target
    velocity = target - noise
    timestep = (time[:, 0, 0] * head.num_timestep_buckets).long()
    state_features = head.state_encoder(state, embodiment_id)
    action_features = head.action_encoder(trajectory, timestep, embodiment_id)
    if head.config.add_pos_embed:
        positions = torch.arange(action_features.shape[1], device=features.device)
        action_features = action_features + head.position_embedding(positions)[None]
    sa = torch.cat([state_features, action_features], dim=1)
    # None has an explicit empty tensor substitute so checkpoint recomputation
    # never consults an outer, expired memory context or a newer training sample.
    memory = (features.new_empty((features.shape[0], 0, next(iter(bridge.values())).hidden_dim),
                                 dtype=torch.float32)
              if memory_tokens is None else memory_tokens)

    def expert(vl, selected_memory):
        options = dict(hidden_states=sa, encoder_hidden_states=vl,
                       encoder_attention_mask=attention_mask, timestep=timestep,
                       return_all_hidden_states=True, temb_add=None)
        if head.config.use_alternate_vl_dit:
            options.update(image_mask=image_mask, backbone_attention_mask=attention_mask)
        with with_memory(head, selected_memory):
            out, _ = head.model(**options)
            return head.action_decoder(out, embodiment_id)[:, -target.shape[1]:]

    trainable = any(parameter.requires_grad for parameter in head.parameters())
    if activation_checkpointing and torch.is_grad_enabled() and (
            features.requires_grad or memory.requires_grad or sa.requires_grad or trainable):
        from torch.utils.checkpoint import checkpoint
        pred = checkpoint(expert, features, memory, use_reentrant=False)
    else:
        pred = expert(features, memory)
    difference = torch.where(valid, pred.float() - velocity.float(), 0.0)
    return {"loss": (difference.square() * mask).sum() / (mask.sum() + 1e-6),
            "velocity_mae": (difference.abs() * mask).sum() / (mask.sum() + 1e-6),
            "prediction": pred}


def v6_episode_flow_loss(head, episode, decision, memory_tokens=None, *, seed=None,
                         activation_checkpointing=False):
    """Cached episode bridge; preserve EVERY original feature/short token."""
    added = {id(parameter) for parameter in expert_parameters(head)}
    added.update(id(parameter) for parameter in bridge_parameters(head))
    reference = next(parameter for parameter in head.parameters() if id(parameter) not in added)
    device, dtype = reference.device, reference.dtype
    feature = episode["features"][decision].to(device=device, dtype=dtype)[None]
    state = episode["state"][decision].to(device=device, dtype=dtype).reshape(1, 1, -1)
    target = episode["targets"][decision].to(device=device, dtype=dtype)[None]
    mask = episode["target_mask"][decision].to(device=device)[None]
    attention = episode["attention_masks"][decision].to(device=device, dtype=torch.bool)[None]
    images = episode["image_masks"][decision].to(device=device, dtype=torch.bool)[None]
    embodiment = torch.tensor([int(episode["embodiment_id"])], device=device, dtype=torch.long)
    noise, time = sample_noise_time(head, target, seed)
    memory = memory_tokens.to(device=device) if memory_tokens is not None else None
    return v6_flow_loss(head, feature, state, target, mask, attention, images, embodiment,
                        memory, noise=noise, time=time,
                        activation_checkpointing=activation_checkpointing)
