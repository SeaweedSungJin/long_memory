"""Causal bank replay, separate from differentiable event/read modules.

An event numbered i finishes at observation i+1. Decision d may only read i<d.
Hard acceptance/FIFO is intentionally not differentiated; selected event features
are recomputed with autograd for the student's action loss on every update.
"""
import torch


def event_inputs(episode, device, stop=None):
    n = len(episode["actions"]) if stop is None else int(stop)
    if not 0 <= n <= len(episode["actions"]):
        raise ValueError("Event stop out of bounds")
    values = {
        "short": episode["short"][:n],
        "pre_moment": episode["moment"][:n],
        "post_moment": episode["moment"][1:n+1],
        "state": episode["state"][:n],
        "next_state": episode["state"][1:n+1],
        "actions": episode["actions"][:n],
        "action_mask": episode["action_mask"][:n],
        "valid": episode["transition_valid"][:n],
    }
    return {k: v.to(device=device, dtype=torch.bool if k in ("action_mask", "valid") else torch.float32)
            for k, v in values.items()}


def encode_until(memory, episode, decision):
    return memory.encode_events(event_inputs(episode, next(memory.parameters()).device, decision))


def read_bank(memory, episode, decision, bank_ids, encoded=None):
    if any(i < 0 or i >= decision or not bool(episode["transition_valid"][i]) for i in bank_ids):
        raise ValueError("Bank contains future, unfinished, or invalid event")
    if len(bank_ids) > memory.config.capacity:
        raise ValueError("Bank exceeds configured capacity")
    device = next(memory.parameters()).device
    if encoded is None:
        encoded = encode_until(memory, episode, decision)
    idx = torch.tensor(bank_ids, dtype=torch.long, device=device)
    keys = encoded["keys"].index_select(0, idx)[None]
    values = encoded["values"].index_select(0, idx)[None]
    short = episode["short"][decision].to(device=device, dtype=torch.float32)[None]
    state = episode["state"][decision].to(device=device, dtype=torch.float32)[None]
    mask = torch.ones((1, len(bank_ids)), device=device, dtype=torch.bool)
    return memory.read(short, state, keys, values, mask)


@torch.no_grad()
def replay_bank(memory, episode, decision, policy="all", encoded=None):
    """Replay exactly the selected policy; no training-only oracle retention.

Stage-1 all/FIFO can forget distant cues. Choose capacity appropriately and use
bank age diagnostics; do not claim it covers the entire episode by default.
"""
    if policy not in ("all", "hard", "novelty"):
        raise ValueError(f"Unknown write policy: {policy}")
    if encoded is None:
        encoded = encode_until(memory, episode, decision)
    bank, attempted, accepted = [], 0, 0
    device = next(memory.parameters()).device
    for i in range(decision):
        if not bool(episode["transition_valid"][i]):
            continue
        attempted += 1
        should_write = policy == "all" or len(bank) < memory.config.min_fill
        if not should_write:
            result = read_bank(memory, episode, i, bank, encoded)
            short = episode["short"][i].to(device=device, dtype=torch.float32)[None]
            index = torch.tensor(bank, device=device, dtype=torch.long)
            bank_keys = encoded["keys"][index][None]
            novelty = memory.novelty(encoded["keys"][i:i+1], bank_keys,
                                     torch.ones((1, len(bank)), device=device, dtype=torch.bool))
            if policy == "hard":
                utility = memory.utility(encoded["event"][i:i+1], short, result["read"])
                probability = memory.write_logits(utility, novelty).sigmoid().item()
                should_write = probability >= memory.config.write_threshold
            else:
                # Simple fixed heuristic comparator, not learned utility.
                should_write = novelty.item() >= 0.1
        if should_write:
            accepted += 1
            bank.append(i)
            if len(bank) > memory.config.capacity:
                bank.pop(0)
    return bank, {"write_rate": accepted / max(attempted, 1), "bank_fill": float(len(bank)),
                  "oldest_event_age": float(decision - bank[0]) if bank else 0.0}


def candidate_predictions(memory, episode, candidate, policy="hard"):
    """Supervise candidates even when rejected by the bank's hard decision."""
    encoded = encode_until(memory, episode, candidate + 1)
    bank, _ = replay_bank(memory, episode, candidate, policy, encoded)
    result = read_bank(memory, episode, candidate, bank, encoded)
    device = next(memory.parameters()).device
    short = episode["short"][candidate].to(device=device, dtype=torch.float32)[None]
    index = torch.tensor(bank, dtype=torch.long, device=device)
    novelty = memory.novelty(encoded["keys"][candidate:candidate+1], encoded["keys"][index][None],
                             torch.ones((1, len(bank)), device=device, dtype=torch.bool))
    utility = memory.utility(encoded["event"][candidate:candidate+1], short, result["read"])
    logit = memory.write_logits(utility, novelty)
    return utility, logit
