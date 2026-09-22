"""Explicit, weight-preserving minimum-fill intervention on a loaded ECHO policy.

The original checkpoint is loaded and checked before the runtime configuration
is changed. The inherited observation, action, READ and learned full-bank
competition paths remain in use. This is an evaluation override, not training.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

from run_scripts.robomme.policy_echo_cvom import EchoPolicyV1


MIN_FILL_SOURCES = ("policy_echo_min_fill.py", "serve_echo_min_fill.py")


def min_fill_source_identity():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in MIN_FILL_SOURCES}


def _tensor_identity(core):
    """Check that assigning a config cannot replace or update model tensors."""
    return {name: (id(value), value.data_ptr(), tuple(value.shape), value.dtype,
                   value.device, value._version)
            for name, value in core.state_dict(keep_vars=True).items()}


def apply_min_fill_override(core, min_fill=None):
    """Apply only minimum fill after loading, without creating tensors or RNG draws."""
    original = core.echo_config
    if core.manager.config != original:
        raise ValueError("ECHO core and manager configurations already disagree")
    if min_fill is not None and (type(min_fill) is not int
                                or not 1 <= min_fill <= original.capacity_events):
        raise ValueError("min_fill override must be an integer within bank capacity")
    effective = original if min_fill is None else replace(original, min_fill=min_fill)
    before = _tensor_identity(core)
    core.echo_config = effective
    core.manager.config = effective
    if _tensor_identity(core) != before:
        raise RuntimeError("Minimum-fill override changed model tensors")
    return {"min_fill_override": min_fill, "original_min_fill": original.min_fill,
            "effective_min_fill": effective.min_fill}


class EchoMinFillPolicy(EchoPolicyV1):
    def __init__(self, base_model, checkpoint, device="cuda:0", memory_off=False,
                 fifo=False, strict=True, min_fill=None):
        before = min_fill_source_identity()
        # In particular, do not edit the saved EchoConfig before load_checkpoint.
        super().__init__(base_model, checkpoint, device=device, memory_off=memory_off,
                         fifo=fifo, strict=strict)
        self.min_fill_metadata = apply_min_fill_override(self.representation, min_fill)
        self.min_fill_source_sha256 = min_fill_source_identity()
        if before != self.min_fill_source_sha256:
            raise ValueError("Minimum-fill wrapper source changed while loading")

    def _get_action(self, observation, options=None):
        if (self.representation.echo_config != self.representation.manager.config
                or self.representation.echo_config.min_fill
                != self.min_fill_metadata["effective_min_fill"]):
            raise ValueError("Effective minimum-fill configuration changed during evaluation")
        actions, info = super()._get_action(observation, options)
        return actions, {**info, **self.min_fill_metadata,
                         "min_fill_source_sha256": dict(self.min_fill_source_sha256)}
