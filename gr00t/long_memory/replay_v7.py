"""Explicit causal memory replay shared by V7 training and inference helpers.

Only frozen short/state/frame/demo observations are read; action tensors and
future observations are never accessed. Non-reentrant checkpointing recomputes
small pure WRITE segments without detaching their boundary states. It is not
TBPTT: an early demo WRITE can receive a much later action-query gradient.
"""

import math

import torch
from torch.utils.checkpoint import checkpoint


def _mode(mode):
    if mode not in ("recurrent", "archive", "none"):
        raise ValueError("mode must be recurrent, archive, or none")


def initial_replay_state(memory, mode="recurrent", batch_size=1):
    _mode(mode)
    state = memory.initial_state(batch_size)
    return state[:, :0] if mode == "archive" else state


def _validate_prefix(episode, count):
    """Validate only the used prefix; do not inspect future feature contents."""
    if type(count) is not int or not 0 <= count <= len(episode["frames"]):
        raise ValueError("Replay prefix is out of bounds")
    for key in ("short", "state", "is_demo"):
        if key not in episode or len(episode[key]) < count:
            raise ValueError(f"Missing/incomplete cached {key}; explicit is_demo is required")
    values = torch.as_tensor(episode["frames"][:count])
    if values.ndim != 1 or values.dtype == torch.bool or values.is_complex():
        raise ValueError("Cached frames must be 1-D raw-frame indices")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()) or bool((values != values.floor()).any()):
        raise ValueError("Cached frames must be finite nonnegative integers")
    if count > 1 and not bool((values[1:] > values[:-1]).all()):
        raise ValueError("Cached frames must be unique and increasing")
    demo = torch.as_tensor(episode["is_demo"][:count])
    if demo.shape != (count,) or demo.dtype != torch.bool:
        raise ValueError("Cached is_demo must be explicit boolean [T]")
    if count > 1 and bool((~demo[:-1] & demo[1:]).any()):
        raise ValueError("Demo observations must form a prefix before execution")


def _row(episode, index):
    if type(index) is not int or not 0 <= index < len(episode["frames"]):
        raise ValueError("Observation index out of bounds")
    if "is_demo" not in episode:
        raise ValueError("Explicit cached is_demo is required")
    return (episode["short"][index].unsqueeze(0), episode["state"][index].unsqueeze(0),
            torch.as_tensor(episode["frames"][index]).reshape(1),
            torch.as_tensor(episode["is_demo"][index]).reshape(1))


def encode_at(memory, episode, index):
    return memory.encode(*_row(episode, index))


def apply_write(memory, state, encoded, *, mode="recurrent", cvom=None, threshold=0.05):
    """One pure online/offline WRITE; a sufficiently negative gain means KEEP.

    CVOM policy decisions are intentionally discrete and detached. Stage 2
    trains the critic directly by regression, not by backpropagating through
    this comparison. KEEP returns identical contents (no normalization).
    """
    _mode(mode)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if cvom is not None and mode != "recurrent":
        raise ValueError("Storage CVOM is only defined for recurrent latent memory")
    zero = encoded.new_zeros(())
    if mode == "none":
        return state, {"write_rate": zero, "keep_rate": zero, "cvom_gain": zero}
    if mode == "archive":
        state, encoded = memory._inputs(state, encoded, archive=True)
        return torch.cat((state, encoded), dim=1), {"write_rate": zero + 1, "keep_rate": zero, "cvom_gain": zero}
    candidate, metrics = memory.write(state, encoded)
    gain = encoded.new_zeros(encoded.shape[0])
    if cvom is not None:
        with torch.no_grad():
            gain = cvom(encoded, state, candidate, memory.slot_addresses)
        if gain.shape != (state.shape[0],) or not bool(torch.isfinite(gain).all()):
            raise ValueError("CVOM policy must return finite signed scores [B]")
    keep = gain < -threshold
    next_state = torch.where(keep[:, None, None], state, candidate)
    return next_state, dict(metrics, write_rate=(~keep).float().mean(), keep_rate=keep.float().mean(), cvom_gain=gain.mean())


def _validate_scan(checkpoint_segment, reset_before, stop):
    if type(checkpoint_segment) is not int or checkpoint_segment < 0:
        raise ValueError("checkpoint_segment must be a nonnegative integer (0 disables recomputation)")
    if reset_before is not None and (type(reset_before) is not int or not 0 <= reset_before <= stop):
        raise ValueError("reset_before must be an observation index no later than the first requested query")


def _advance(memory, episode, state, begin, end, *, mode, checkpoint_segment, cvom, threshold):
    if mode == "none" or begin == end:
        return state
    block = checkpoint_segment or max(1, end - begin)
    for start in range(begin, end, block):
        # Bind row tensors into this specific closure. A closure capturing the
        # changing outer start/end variables silently replays the wrong prefix
        # during backward; this factory deliberately avoids that failure.
        rows = tuple(_row(episode, index) for index in range(start, min(end, start + block)))

        def make_segment(bound_rows):
            def segment(bank):
                for row in bound_rows:
                    encoded = memory.encode(*row)
                    bank, _ = apply_write(memory, bank, encoded, mode=mode, cvom=cvom, threshold=threshold)
                return bank
            return segment

        segment = make_segment(rows)
        if checkpoint_segment and torch.is_grad_enabled() and any(p.requires_grad for p in memory.parameters()):
            state = checkpoint(segment, state, use_reentrant=False, preserve_rng_state=False)
        else:
            state = segment(state)
    return state


def replay_state(memory, episode, stop_before, *, mode="recurrent", checkpoint_segment=8,
                 reset_before=None, cvom=None, threshold=0.05):
    """Return M_k after writes [0,k), never after the current observation k.

    ``reset_before=r`` is an explicit last-L-writes diagnostic: start from zero
    at observation index r. It does not remove HAMLET history already present
    inside each frozen short token, and is not the normal training protocol.
    """
    _mode(mode)
    _validate_prefix(episode, stop_before)
    _validate_scan(checkpoint_segment, reset_before, stop_before)
    if cvom is not None and mode != "recurrent":
        raise ValueError("CVOM is only supported in recurrent mode")
    state = initial_replay_state(memory, mode)
    return _advance(memory, episode, state, reset_before or 0, stop_before, mode=mode,
                    checkpoint_segment=checkpoint_segment, cvom=cvom, threshold=threshold)


def replay_queries(memory, episode, queries, *, mode="recurrent", checkpoint_segment=8,
                   reset_before=None, cvom=None, threshold=0.05):
    """Return ``{query_index: (fused_short[B,Q,D], scalar_metrics)}``.

    Every observation before each selected query is written, including demos
    and unselected action observations. Reads remain in the graph; the final
    selected query's unused WRITE is omitted because no later read uses it.
    No Action Expert or action/noise/target tensor is used by this function.
    """
    _mode(mode)
    queries = list(queries)
    if any(type(index) is not int for index in queries) or len(set(queries)) != len(queries):
        raise ValueError("queries must be unique Python integer observation indices")
    if not queries:
        return {}
    queries.sort()
    if queries[0] < 0 or queries[-1] >= len(episode["frames"]):
        raise ValueError("Query out of bounds")
    _validate_prefix(episode, queries[-1] + 1)
    _validate_scan(checkpoint_segment, reset_before, queries[0])
    if cvom is not None and mode != "recurrent":
        raise ValueError("CVOM is only supported in recurrent mode")
    state = initial_replay_state(memory, mode)
    cursor = reset_before or 0
    result = {}
    for index in queries:
        state = _advance(memory, episode, state, cursor, index, mode=mode,
                         checkpoint_segment=checkpoint_segment, cvom=cvom, threshold=threshold)
        short, _, _, _ = _row(episode, index)
        encoded = encode_at(memory, episode, index)
        fused, metrics = memory.read(short, encoded, state, mode=mode)
        # This is a replay count, not semantic bank occupancy or event age.
        metrics = dict(metrics, replayed_observations=fused.new_tensor(
            0.0 if mode == "none" else float(index - (reset_before or 0))))
        result[index] = (fused, metrics)
        cursor = index  # The current WRITE occurs before the *next* query.
    return result
