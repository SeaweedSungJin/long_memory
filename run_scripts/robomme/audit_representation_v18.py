#!/usr/bin/env python3
"""Read-only lightweight V18 short-memory parity audit; no VLM or simulator.

Only the author's 19 HAMLET Transformer tensors are loaded, not the multi-GB
Action Expert/VLM. The cache rounded normalized moments from FP32 to BF16, so
direct recomputation of native short tokens is not bitwise equivalent even on
the original GPU. V18 B therefore anchors its LoRA delta onto stored/native H.
The report requires exact zero-delta and replay parity and separately reports
the *unanchored* cache information/rounding gap as a diagnostic, not a failure.
No files are created unless --output is explicitly supplied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from safetensors import safe_open

from gr00t.long_memory.hamlet import validate_cache_checkpoint
from gr00t.model.modules.memory import MemoryTransformer
from run_scripts.robomme.representation_core_v18 import (
    CORE_VERSION, RepresentationConfigV18, RepresentationMemoryV18,
)


def load_memory_only(base_model, device="cpu"):
    """Load exactly the native moment-token Transformer, with strict keys."""
    root = Path(base_model).resolve()
    cfg = json.loads((root / "config.json").read_text())
    if (cfg.get("hamlet_mode") != "finetune" or cfg.get("memory_type", "moment_token") != "moment_token"
            or cfg.get("mem_cond_type", "cross_attn") != "cross_attn"):
        raise ValueError("Audit requires moment-token/cross-attention HAMLET")
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "action_head.memory_transformer."
    names = sorted(name for name in index if name.startswith(prefix))
    if not names:
        raise ValueError("No HAMLET memory Transformer tensors in the base checkpoint")
    state = {}
    for shard in sorted({index[name] for name in names}):
        path = (root / shard).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Checkpoint shard escapes base model directory")
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in names:
                if index[name] == shard:
                    value = source.get_tensor(name)
                    if not torch.isfinite(value).all():
                        raise ValueError(f"Nonfinite native Transformer tensor: {name}")
                    state[name.removeprefix(prefix)] = value
    # The original action-head constructor uses MemoryTransformer's default
    # heads/FFN/RMS settings; there is no model-config head-count override.
    model = MemoryTransformer(dim=int(cfg["backbone_embedding_dim"]),
        n_q=int(cfg["n_moment_tokens"]), T=int(cfg["memory_window"]),
        num_layers=int(cfg["memory_num_layers"])).bfloat16()
    model.load_state_dict(state, strict=True)
    return model.requires_grad_(False).eval().to(device), cfg, len(state)


def audit(args):
    cache = Path(args.cache_dir).resolve()
    manifest = json.loads((cache / "manifest.json").read_text())
    validate_cache_checkpoint(manifest)
    base = Path(args.base_model or manifest["model_path"]).resolve()
    if base != Path(manifest["model_path"]).resolve():
        raise ValueError("The audit base must be the exact cache source checkpoint")
    records = {int(row["episode_id"]): row for row in manifest["episodes"]}
    episode_id = args.episode_id if args.episode_id is not None else min(records)
    record = records[episode_id]
    path = (cache / record["path"]).resolve()
    if not path.is_relative_to(cache):
        raise ValueError("Episode path escapes cache")
    ep = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if ep.get("cache_fingerprint") != manifest["fingerprint"] or int(ep["episode_id"]) != episode_id:
        raise ValueError("Cache episode provenance mismatch")
    count = min(args.count, len(ep["frames"]))
    if count < 1:
        raise ValueError("Need at least one observation")
    native, base_config, tensor_count = load_memory_only(base, args.device)
    config = RepresentationConfigV18(feature_dim=native.dim, state_dim=ep["state"].shape[-1],
        num_short_tokens=native.n_q, short_window=native.T, representation="adapted_short")
    core = RepresentationMemoryV18(config, native).to(args.device).eval()
    rows = []
    bank, history = None, None
    with torch.inference_mode():
        prefix = core.encode_prefix(ep, count)
        for index in range(count):
            ids = [max(0, j) for j in range(index-native.T+1, index+1)]
            window = ep["moment"][ids].to(args.device).reshape(1, native.T*native.n_q, native.dim)
            use_amp = torch.device(args.device).type == "cuda"
            native_input = window.bfloat16().float() if use_amp else window.bfloat16()
            with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16, enabled=use_amp):
                original = native(native_input)[:, -native.n_q:].float()
            raw_adapted = core._transform_window(window)
            adapted = prefix["short"][index:index+1]
            cached = ep["short"][index:index+1].to(args.device).float()
            result = core.step(ep["short"][index:index+1], ep["moment"][index:index+1],
                ep["state"][index:index+1], ep["frames"][index:index+1], ep["is_demo"][index:index+1],
                bank=bank, moment_history=history, read_enabled=False)
            bank, history = result["bank"], result["moment_history"]
            difference = adapted-cached
            raw_difference = raw_adapted-cached
            rows.append({"observation": index, "frame": int(ep["frames"][index]),
                "native_zero_lora_exact": torch.equal(original, raw_adapted),
                "online_offline_short_exact": torch.equal(result["short"], adapted),
                "read_off_same_short_exact": torch.equal(result["fused"], adapted),
                "cache_exact": torch.equal(cached, adapted),
                "cache_mse": float(difference.square().mean()),
                "cache_max_abs": float(difference.abs().max()),
                "cache_relative_rms": float(difference.square().mean().sqrt() / cached.square().mean().sqrt().clamp_min(1e-8)),
                "unanchored_cache_relative_rms": float(raw_difference.square().mean().sqrt() / cached.square().mean().sqrt().clamp_min(1e-8))})
    max_error = max(row["cache_relative_rms"] for row in rows)
    mechanical = all(row["native_zero_lora_exact"] and row["online_offline_short_exact"]
                     and row["read_off_same_short_exact"] for row in rows)
    return {"audit_version": CORE_VERSION, "device": args.device, "base_model": str(base),
        "cache_fingerprint": manifest["fingerprint"], "episode_id": episode_id,
        "native_tensor_count": tensor_count, "native_parameter_count": sum(p.numel() for p in native.parameters()),
        "short_lora_parameter_count": sum(p.numel() for p in core.short_parameters()),
        "mechanical_parity_pass": mechanical, "cache_max_relative_rms": max_error,
        "cache_relative_rms_tolerance": args.max_cache_relative_rms,
        "cache_tolerance_pass": max_error <= args.max_cache_relative_rms,
        "passed": mechanical and max_error <= args.max_cache_relative_rms,
        "note": "B uses native_H + T_LoRA(quantized_m) - T_base(quantized_m). Zero LoRA must exactly match cached/native H. Direct unanchored recomputation differs because the cache rounded FP32 VLLN moments to BF16; this is reported separately. This audit is not robot performance evidence.",
        "observations": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="runs/long_memory/cache_full1600_v1")
    parser.add_argument("--base-model")
    parser.add_argument("--episode-id", type=int)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-cache-relative-rms", type=float, default=0.0,
                        help="Tolerance for ANCHORED zero-LoRA versus cached H; default requires exact equality")
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing files are never replaced")
    args = parser.parse_args(argv)
    if args.count <= 0 or not 0 <= args.max_cache_relative_rms < 1:
        parser.error("--count must be positive and cache tolerance must lie in [0,1)")
    if args.output and args.output.exists():
        parser.error("--output exists; use a new path")
    result = audit(args)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(rendered + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
