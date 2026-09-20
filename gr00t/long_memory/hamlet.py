"""Frozen HAMLET loading and the differentiable cached flow-matching bridge.

The cache is AFTER vlln and short-memory conditioning. Do not call
``process_backbone_output`` again on cached features. The expert's parameters
are frozen, but its forward must retain autograd for the new memory inputs.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def validate_cache_checkpoint(manifest):
    """Reject stale frozen features even if a checkpoint was replaced in-place.

    JSON metadata is content-hashed. Multi-GB weight files use size/mtime as in
    the cache builder: this detects ordinary changes, not adversarial tampering.
    No training parquet/video scan is necessary after features are materialized.
    """
    signatures = manifest.get("identity", {}).get("checkpoint")
    if not signatures:
        raise ValueError("Cache has no source checkpoint signatures; rebuild the cache")
    root = Path(manifest["model_path"]).resolve()
    for saved in signatures:
        path = Path(saved["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Cache checkpoint file is missing or outside base directory: {path}")
        stat = path.stat()
        if stat.st_size != saved["size"] or stat.st_mtime_ns != saved["mtime_ns"]:
            raise ValueError(f"Base checkpoint changed after cache extraction: {path}; rebuild cache")
        if "sha256" in saved and hashlib.sha256(path.read_bytes()).hexdigest() != saved["sha256"]:
            raise ValueError(f"Base checkpoint metadata changed after cache extraction: {path}")


def checkpoint_identity(path):
    """Cheap provenance, not a cryptographic verification of multi-GB weights."""
    path = Path(path).resolve()
    names = ["config.json", "model.safetensors.index.json", "processor_config.json",
             "statistics.json", "embodiment_id.json"]
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((path / name).read_bytes())
    index = json.loads((path / "model.safetensors.index.json").read_text())
    shards = {}
    for name in sorted(set(index["weight_map"].values())):
        stat = (path / name).stat()
        shards[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"path": str(path), "metadata_sha256": digest.hexdigest(), "shards": shards}


def load_frozen_hamlet(model_path, device="cuda:0"):
    """Load complete moment-token/cross-attention N1.6 weights, fail on omissions.

The author's legacy name is mapped in memory; files are never rewritten. This
helper deliberately rejects vanilla, AdaLN, and vision-feature checkpoints:
they need different cache/fusion contracts, not silent fallback behavior.
"""
    import gr00t.model  # noqa: F401 -- register AutoModel/AutoProcessor
    from transformers import AutoModel, AutoProcessor

    path = Path(model_path).resolve()
    config = json.loads((path / "config.json").read_text())
    if (config.get("hamlet_mode") != "finetune"
            or config.get("mem_cond_type", "cross_attn") != "cross_attn"
            or config.get("memory_type", "moment_token") != "moment_token"
            or int(config.get("n_moment_tokens", 0)) <= 0):
        raise ValueError("This trainer requires HAMLET N1.6 moment_token + cross_attn")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    keys = index["weight_map"]
    kwargs = {}
    if "backbone.meta_emb" in keys:
        if "backbone.moment_tokens" in keys:
            raise ValueError("Checkpoint contains conflicting moment-token names")
        kwargs["key_mapping"] = {r"^backbone\.meta_emb$": "backbone.moment_tokens"}
        print("[long-memory] loading legacy meta_emb as moment_tokens", flush=True)
    model, info = AutoModel.from_pretrained(
        path, output_loading_info=True, torch_dtype=torch.bfloat16, **kwargs
    )
    errors = {k: info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
              if info.get(k)}
    if errors:
        raise RuntimeError(f"Incomplete frozen HAMLET load: {errors}")
    model.requires_grad_(False)
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    # HF constructs non-parameter distribution constants under torch_dtype too.
    # CPU Dirichlet/Beta sampling has no BF16 kernel; the reference trainer's
    # sampling distribution is FP32, then sample_time casts the sampled value.
    model.action_head.beta_dist = torch.distributions.Beta(
        torch.tensor(float(model.config.noise_beta_alpha), dtype=torch.float32),
        torch.tensor(float(model.config.noise_beta_beta), dtype=torch.float32),
    )
    processor = AutoProcessor.from_pretrained(path)
    processor.eval()
    print("[long-memory] frozen HAMLET loaded without missing/unexpected weights", flush=True)
    return model, processor


@contextmanager
def isolated_seed(seed, device):
    """Pair flow noise/time without advancing the training or unrelated GPU RNG."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed))
        for index in devices:
            torch.cuda.default_generators[index].manual_seed(int(seed))
        yield


def sample_noise_time(head, target, seed=None):
    def sample():
        return (torch.randn_like(target),
                head.sample_time(target.shape[0], target.device, target.dtype)[:, None, None])
    if seed is None:
        return sample()
    with isolated_seed(seed, target.device):
        return sample()


def flow_loss(head, features, state, target, target_mask, attention_mask,
              image_mask, embodiment_id, *, noise=None, time=None,
              activation_checkpointing=False):
    """Original velocity objective with explicit random tensors and FP32 reduction.

Inputs already have a batch axis. Unlike inference ``get_action``, this function
is intentionally NOT decorated with no_grad. State noise/dropout and model
dropout are disabled by keeping the frozen expert in eval mode. Return velocity
MAE, not physical joint MAE or robot success/accuracy.
"""
    if head.training or any(p.requires_grad for p in head.parameters()):
        raise ValueError("Action expert must be eval() and requires_grad_(False)")
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

    if activation_checkpointing and torch.is_grad_enabled() and features.requires_grad:
        from torch.utils.checkpoint import checkpoint
        pred = checkpoint(expert, features, use_reentrant=False)
    else:
        pred = expert(features)
    mask = target_mask.float()
    if mask.sum().item() <= 0:
        raise ValueError("Cannot train/evaluate an action decision with no valid targets")
    difference = torch.where(valid, pred.float() - velocity.float(), 0.0)
    loss = (difference.square() * mask).sum() / (mask.sum() + 1e-6)
    mae = (difference.abs() * mask).sum() / (mask.sum() + 1e-6)
    return {"loss": loss, "velocity_mae": mae, "prediction": pred}


def episode_flow_loss(head, episode, decision, fused_short=None, *, seed=None,
                      activation_checkpointing=False):
    """Single cached decision, preserving all non-memory conditioning tokens."""
    reference = next(head.parameters())
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
    return flow_loss(head, feature, state, target, mask, attention, images, embodiment,
                     noise=noise, time=time, activation_checkpointing=activation_checkpointing)
