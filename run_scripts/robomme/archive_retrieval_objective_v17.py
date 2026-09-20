"""V17 auxiliary supervision for the EXISTING V7 archive's native reader.

Targets are offline weak labels, never query/key/value inputs. This changes no
inference architecture. APPEND is still rule based, not a learned CVOM writer.
"""
from __future__ import annotations

import torch

from gr00t.long_memory.replay_v7 import encode_at, replay_state, _validate_prefix


def positive_mask(episode, decision, target):
    """Intersect weak raw-frame positives with the strict-past canonical bank."""
    _validate_prefix(episode, decision + 1)
    if (int(episode["episode_id"]) != target["episode_id"]
            or decision != target["decision"]
            or int(episode["frames"][decision]) != target["query_frame"]):
        raise ValueError("Target/query identity mismatch")
    if decision <= 0 or bool(episode["is_demo"][decision]) or not bool(episode["decision_mask"][decision]):
        raise ValueError("Retrieval target must supervise an active execution query")
    positives = target["positive_frames"]
    if not positives or any(type(f) is not int or not 0 <= f < target["n_demo"]
                            or f >= target["query_frame"] for f in positives):
        raise ValueError("Positive labels must refer only to past demo observations")
    frames = torch.as_tensor(episode["frames"][:decision])
    mask = torch.tensor([int(f) in positives for f in frames], dtype=torch.bool)
    if bool((mask & ~torch.as_tensor(episode["is_demo"][:decision]).cpu()).any()):
        raise ValueError("Positive frame is not a demo in the actual cache")
    if not bool(mask.any()) or bool(mask.all()):
        raise ValueError("Canonical archive needs both positive and negative observations")
    return mask


def archive_probabilities(memory, episode, decision, *, checkpoint_segment=8):
    """The actual reader's softmax, averaged across heads/queries, summed over Q.

    All past observations compete, including execution observations. No target
    labels, actions, GT state, future frames, or selected segment enter READ.
    Full-prefix replay leaves gradients to earlier encodings intact.
    """
    _validate_prefix(episode, decision + 1)
    if decision <= 0 or memory.training or memory.attention.dropout != 0:
        raise ValueError("Nonempty archive and deterministic eval-mode reader required")
    bank = replay_state(memory, episode, decision, mode="archive", checkpoint_segment=checkpoint_segment)
    query = encode_at(memory, episode, decision)
    with torch.autocast(device_type=bank.device.type, enabled=False):
        _, weights = memory.attention(memory.read_query_norm(query), memory.read_key_norm(bank), bank,
                                      need_weights=True, average_attn_weights=False)
    return weights.mean(dim=(1, 2)).reshape(1, decision, memory.config.num_short_tokens).sum(-1)[0]


def retrieval_objective(memory, episode, decision, target, *, checkpoint_segment=8):
    # Forward is intentionally target-independent; labels are used ONLY below.
    probs = archive_probabilities(memory, episode, decision, checkpoint_segment=checkpoint_segment)
    mask = positive_mask(episode, decision, target).to(probs.device)
    mass = probs[mask].sum()
    loss = -mass.clamp_min(torch.finfo(probs.dtype).tiny).log()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite retrieval NLL")
    return loss, {"retrieval_nll": float(loss.detach()), "positive_attention_mass": float(mass.detach()),
                  "weak_span_top1": float(mask[probs.argmax()]),
                  "uniform_positive_mass": float(mask.float().mean())}
