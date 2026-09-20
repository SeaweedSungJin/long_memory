"""Causal V8 replay: re-encode retained raw events under current parameters.

Each event has an independent encoder graph, so a later query directly trains
historical encodings without a recurrent BPTT chain. ``checkpoint_segment`` is
accepted and validated for caller compatibility; this implementation batches
the causal prefix and does not perform activation checkpointing. Actions,
targets, and future observation contents are never inputs to memory.
"""

import torch


def _mode(mode):
    if mode not in ("event", "none"):
        raise ValueError("mode must be event or none")


def initial_replay_state(memory, mode="event", batch_size=1):
    _mode(mode)
    return memory.initial_state(batch_size)


def _validate_prefix(memory, episode, count):
    if "frames" not in episode or type(count) is not int or not 0 <= count <= len(episode["frames"]):
        raise ValueError("Replay prefix is out of bounds")
    for key in {memory.config.source, "short", "state", "is_demo"}:
        if key not in episode or len(episode[key]) < count:
            raise ValueError(f"Missing/incomplete cached {key}; explicit is_demo is required")
    frames = torch.as_tensor(episode["frames"][:count])
    if frames.ndim != 1 or frames.dtype == torch.bool or frames.is_complex():
        raise ValueError("Cached frames must be 1-D raw-frame indices")
    if (not bool(torch.isfinite(frames).all())
            or bool(((frames < 0) | (frames > 2 ** 24) | (frames != frames.floor())).any())):
        raise ValueError("Cached frames must be exact finite nonnegative integer indices <= 2**24")
    if count > 1 and not bool((frames[1:] > frames[:-1]).all()):
        raise ValueError("Cached frames must be unique and increasing")
    demo = torch.as_tensor(episode["is_demo"][:count])
    if demo.shape != (count,) or demo.dtype != torch.bool:
        raise ValueError("Cached is_demo must be explicit boolean [T]")
    if count > 1 and bool((~demo[:-1] & demo[1:]).any()):
        raise ValueError("Demo observations must form a prefix before execution")


def _validate_scan(checkpoint_segment, reset_before, stop):
    if type(checkpoint_segment) is not int or checkpoint_segment < 0:
        raise ValueError("checkpoint_segment must be a nonnegative integer")
    if reset_before is not None and (type(reset_before) is not int or not 0 <= reset_before <= stop):
        raise ValueError("reset_before must be an observation index no later than the first requested query")


def _encode_range(memory, episode, begin, end):
    return memory.encode(episode[memory.config.source][begin:end], episode["state"][begin:end],
                         torch.as_tensor(episode["frames"][begin:end]),
                         torch.as_tensor(episode["is_demo"][begin:end]))


def encode_at(memory, episode, index):
    if type(index) is not int or "frames" not in episode or not 0 <= index < len(episode["frames"]):
        raise ValueError("Observation index out of bounds")
    _validate_prefix(memory, episode, index + 1)
    return _encode_range(memory, episode, index, index + 1)


def apply_write(memory, state, encoded, *, mode="event"):
    _mode(mode)
    if mode == "none":
        state, _ = memory._inputs(state, encoded)
        return state, {"write_rate": encoded.new_zeros(()), "keep_rate": encoded.new_zeros(()),
                       "evicted_events": encoded.new_zeros(()), "retained_events": encoded.new_zeros(())}
    return memory.write(state, encoded)


def replay_state(memory, episode, stop_before, *, mode="event", checkpoint_segment=8, reset_before=None):
    """Return the retained writes [reset_before, stop_before), excluding current."""
    _mode(mode)
    _validate_prefix(memory, episode, stop_before)
    _validate_scan(checkpoint_segment, reset_before, stop_before)
    begin = max(reset_before or 0, stop_before - memory.config.capacity)
    if mode == "none" or begin == stop_before:
        return initial_replay_state(memory, mode)
    encoded = _encode_range(memory, episode, begin, stop_before)
    return encoded.reshape(1, -1, memory.config.hidden_dim + 2)


def replay_queries(memory, episode, queries, *, mode="event", checkpoint_segment=8, reset_before=None):
    """Return {query: (fused_short[B,Q,D], scalar_metrics)} with READ before WRITE.

    ``storage_reconstruction_loss`` averages only retained *past* events,
    excluding the current event. It remains differentiable and is a separate
    auxiliary objective, not an action or retrieval-success metric.
    ``reset_before`` is a suffix-only diagnostic, not the normal protocol.
    """
    _mode(mode)
    queries = list(queries)
    if any(type(index) is not int for index in queries) or len(set(queries)) != len(queries):
        raise ValueError("queries must be unique Python integer observation indices")
    if not queries:
        return {}
    queries.sort()
    if "frames" not in episode or queries[0] < 0 or queries[-1] >= len(episode["frames"]):
        raise ValueError("Query out of bounds")
    _validate_prefix(memory, episode, queries[-1] + 1)
    _validate_scan(checkpoint_segment, reset_before, queries[0])
    reset = reset_before or 0
    begin = max(reset, queries[0] - memory.config.capacity)
    if mode == "event":
        # No recurrent dependency: one batched encoding serves all causal reads.
        encoded = _encode_range(memory, episode, begin, queries[-1] + 1)
        past_count = queries[-1] - begin
        reconstruction = (memory.reconstruction_loss(encoded[:past_count],
                          episode[memory.config.source][begin:queries[-1]], reduction="none")
                          if past_count else None)
    result = {}
    for index in queries:
        if mode == "event":
            offset, low = index - begin, max(reset, index - memory.config.capacity) - begin
            current = encoded[offset:offset + 1]
            bank = encoded[low:offset].reshape(1, -1, memory.config.hidden_dim + 2)
        else:
            current = _encode_range(memory, episode, index, index + 1)
            bank = initial_replay_state(memory, mode)
        fused, metrics = memory.read(episode["short"][index:index + 1], current, bank, mode=mode)
        auxiliary = (reconstruction[low:offset].mean()
                     if mode == "event" and offset > low else fused.new_zeros(()))
        result[index] = (fused, dict(metrics, storage_reconstruction_loss=auxiliary,
                         replayed_observations=fused.new_tensor(float(index - reset) if mode == "event" else 0.0)))
    return result
