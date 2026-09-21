"""Explicit inference-only feature precision contract for the V19 A/B test.

This is deliberately not an autocast wrapper around a policy. Only the frozen
VLM -> VLLN -> HAMLET feature producer changes in ``cache-aligned`` mode. Its
outputs then cross the same BF16 serialization boundary as ``cache.py``. The
external memory still runs in FP32, and state encoding / denoising are untouched.
"""
from __future__ import annotations

from contextlib import nullcontext


FEATURE_PRECISIONS = ("native", "cache-aligned")


def validate_feature_precision(value: str) -> str:
    if not isinstance(value, str) or value not in FEATURE_PRECISIONS:
        raise ValueError(f"feature_precision must be one of {FEATURE_PRECISIONS}, got {value!r}")
    return value


def feature_precision_contract(value: str) -> dict:
    """JSON-safe semantics shared by server diagnostics and eval provenance.

    Native means the original unwrapped producer, including its existing input
    and weight dtypes; it does not force a different global autocast state.
    """
    value = validate_feature_precision(value)
    aligned = value == "cache-aligned"
    return {
        "contract_version": 1,
        "feature_precision": value,
        "feature_extraction_autocast": "bfloat16" if aligned else "unchanged_native",
        "autocast_scope": ["model.backbone", "head.vlln", "head.process_backbone_output"] if aligned else [],
        "normalized_moment_boundary": "bfloat16_then_float32" if aligned else "float32_without_bfloat16_rounding",
        "entire_backbone_features_boundary": "bfloat16" if aligned else "unchanged_native_dtype",
        "short_boundary": "tail_of_entire_backbone_features",
        "external_memory_compute": "float32_autocast_disabled_unchanged",
        "fused_short_ae_boundary": "cast_to_backbone_features_dtype_unchanged",
        "backbone_attention_and_image_masks": "unchanged",
        "ae_state_encoder": "unchanged_outside_feature_autocast",
        "ae_denoising": "unchanged_outside_feature_autocast",
        "hamlet_recurrent_cache": "native_internal_cache_not_extra_quantized",
        "sampling_seed_steps_cadence_reset_and_action_execution": "unchanged",
    }


def extract_hamlet_features(model, head, backbone_inputs, n_q, feature_precision="native"):
    """Return the original backbone dict and normalized moments for memory.

    Quantization is on the ENTIRE postprocessed feature tensor, not merely the
    moment / short tail: cached AE inputs also contain the non-short features.
    HAMLET's private recurrent state is *not* rounded a second time: cache.py
    stores detached output features but carries its ordinary internal state.
    Masks and every other backbone field are preserved by reference.
    """
    import torch

    mode = validate_feature_precision(feature_precision)
    context = (torch.autocast(device_type=model.device.type, dtype=torch.bfloat16)
               if mode == "cache-aligned" else nullcontext())
    with context:
        raw_backbone = model.backbone(backbone_inputs)
        # Independently normalize the original tail, exactly as cache.py and
        # the previous native policy do. VLLN is not applied twice to one input.
        moment = head.vlln(raw_backbone["backbone_features"])[:, -n_q:].float()
        backbone = head.process_backbone_output(raw_backbone, action_inputs_B=1)
    if mode == "cache-aligned":
        # Match cache.py's BF16 storage, followed by the reader's FP32 load.
        moment = moment.to(torch.bfloat16).float()
        backbone["backbone_features"] = backbone["backbone_features"].to(torch.bfloat16)
    return backbone, moment
