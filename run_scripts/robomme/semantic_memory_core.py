"""Training-only observations of the unchanged V18 reader.

The answer head receives the actual attention VALUE output, before query/FFN
residuals. A temporary hook observes rather than modifies that output: no
second attention call, additional RNG consumption, or deployment code change.
Ground-truth metadata must never be an argument to replay/read/write.
"""
from contextlib import contextmanager
from copy import deepcopy

import torch


class AnswerSuite(torch.nn.Module):
    """Memory head plus matched-capacity current-only and clock-only controls.

    Control inputs are detached: they cannot teach the actor to bypass memory.
    All three heads see the same TRAIN labels and optimizer budget. Clock uses
    observed absolute time/demo phase, never episode length/future progress.
    """
    def __init__(self, config):
        super().__init__()
        from run_scripts.robomme.semantic_memory_answers import SemanticAnswerHead
        self.memory = SemanticAnswerHead(config)
        self.current = deepcopy(self.memory)
        self.clock = deepcopy(self.memory)

    def loss(self, retrieved, target):
        return self.memory.loss(retrieved, target)

    @staticmethod
    def clock_input(reference, frame, is_demo):
        value = torch.zeros_like(reference)
        if value.shape[-1] < 5:
            raise ValueError("Clock control requires >=5 channels")
        time = torch.as_tensor(frame, device=value.device).float()/16
        # A lone scalar would lose its magnitude under the head's LayerNorm.
        value[..., 0], value[..., 1], value[..., 2] = time.log1p(), time.sin(), time.cos()
        value[..., 3], value[..., 4] = 1., float(bool(is_demo))
        return value

    def controls(self, query, frame, is_demo, target):
        current = self.current.loss(query.detach(), target)
        clock = self.clock.loss(self.clock_input(query.detach(), frame, is_demo), target)
        return current, clock


@contextmanager
def capture_read(core):
    values = []
    handle = core.memory.attention.register_forward_hook(
        lambda module, inputs, output: values.append(output[0]))
    try:
        yield values
    finally:
        handle.remove()


def semantic_replay(core, episode, decision, *, manager=None,
                    activation_checkpointing=False, read_enabled=True):
    writes = []
    policy = manager.make_policy() if manager is not None else None
    def tracked_policy(*args, **kwargs):
        bank, metrics = policy(*args, **kwargs)
        writes.append(metrics)
        return bank, metrics
    with capture_read(core) as values:
        out = core.replay(episode, decision, read_enabled=read_enabled,
            activation_checkpointing=activation_checkpointing,
            write_policy=tracked_policy if policy is not None else None)
    if len(values) > 1:
        raise RuntimeError("Expected exactly one final READ, not recurrent write attention")
    out["retrieved"] = values[0] if values else torch.zeros_like(out["encoded_current"])
    out["has_read"] = bool(values)
    if manager is not None:
        full = [row for row in writes if row["writer_full"] == 1.]
        out["metrics"]["storage_full_decisions"] = float(len(full))
        for key in ("writer_keep", "writer_replace", "writer_merge"):
            # Rates AFTER filling; forced initial insertions must not hide a
            # controller that rejects every subsequent observation.
            out["metrics"][key+"_when_full"] = sum(row[key] for row in full)/max(1, len(full))
    return out


def semantic_read(core, short, query, bank):
    with capture_read(core) as values:
        fused, metrics = core.read_from_bank(short, query, bank)
    return fused, values[0] if values else torch.zeros_like(query), metrics


class ReplayWithWriter:
    """Reuse original three-arm action validation with the actual hard writer."""
    def __init__(self, core, manager):
        self.core, self.manager = core, manager

    def replay(self, *args, **kwargs):
        kwargs["write_policy"] = self.manager.make_policy() if self.manager is not None else None
        return self.core.replay(*args, **kwargs)
