"""Causal training replay and deterministic storage options for action-value v3.

Event i finishes at endpoint i+1. Unlike legacy utility v1/v2, the writer's
current query is endpoint i+1, when the situation/action/outcome is available.
Teacher future losses supervise choices but are never inputs to the chooser.
"""

import torch


def event_inputs(episode, device, stop=None):
    n = len(episode["actions"]) if stop is None else int(stop)
    if not 0 <= n <= len(episode["actions"]):
        raise ValueError("Event stop out of bounds")
    if "frames" not in episode:
        raise ValueError("V3 requires raw endpoint frames, not inferred event indices")
    frames = torch.as_tensor(episode["frames"])
    if len(frames) < n + 1 or bool((frames[1:n + 1] <= frames[:n]).any()):
        raise ValueError("Endpoint frames must strictly increase")
    values = {"short": episode["short"][:n], "pre_moment": episode["moment"][:n],
              "post_moment": episode["moment"][1:n + 1], "state": episode["state"][:n],
              "next_state": episode["state"][1:n + 1], "actions": episode["actions"][:n],
              "action_mask": episode["action_mask"][:n], "valid": episode["transition_valid"][:n],
              "start_frames": frames[:n], "end_frames": frames[1:n + 1]}
    return {key: value.to(device=device, dtype=torch.bool if key in ("action_mask", "valid")
                         else torch.float32) for key, value in values.items()}


def encode_until(memory, episode, decision):
    return memory.encode_events(event_inputs(episode, next(memory.parameters()).device, decision))


def _validate_bank(memory, episode, decision, bank_ids):
    if not 0 <= decision <= len(episode["actions"]):
        raise ValueError("Decision out of bounds")
    if list(bank_ids) != sorted(set(bank_ids)):
        raise ValueError("Bank IDs must be unique and chronologically sorted")
    if any(i < 0 or i >= decision or not bool(episode["transition_valid"][i]) for i in bank_ids):
        raise ValueError("Bank contains future, unfinished, or invalid event")
    if len(bank_ids) > memory.config.capacity:
        raise ValueError("Bank exceeds configured capacity")


def read_bank(memory, episode, decision, bank_ids, encoded=None):
    _validate_bank(memory, episode, decision, bank_ids)
    device = next(memory.parameters()).device
    if encoded is None:
        encoded = encode_until(memory, episode, decision)
    if len(encoded["event"]) < decision:
        raise ValueError("Encoded events do not cover decision prefix")
    idx = torch.tensor(bank_ids, dtype=torch.long, device=device)
    short = episode["short"][decision].to(device=device, dtype=torch.float32)[None]
    state = episode["state"][decision].to(device=device, dtype=torch.float32)[None]
    frame = torch.as_tensor(episode["frames"][decision], device=device, dtype=torch.float32).reshape(1)
    return memory.read(short, state, encoded["keys"].index_select(0, idx)[None],
                       encoded["values"].index_select(0, idx)[None],
                       torch.ones((1, len(bank_ids)), device=device, dtype=torch.bool), frame)


def storage_options(bank_ids, candidate, capacity, max_victims):
    """KEEP first; APPEND if room, otherwise evenly spaced causal victims.

    Victims are chosen deterministically without future losses: oldest, evenly
    spaced intermediate slots, newest. Every replacement has the same capacity
    as KEEP. We do not silently evaluate all victims or increase bank capacity.
    """
    bank = list(bank_ids)
    if capacity <= 0 or max_victims <= 0 or len(bank) > capacity:
        raise ValueError("Invalid storage capacity/max_victims")
    if bank != sorted(set(bank)) or candidate < 0 or any(i < 0 or i >= candidate for i in bank):
        raise ValueError("Storage bank must contain unique sorted events before candidate")
    options = [bank.copy()]
    if len(bank) < capacity:
        return options + [bank + [int(candidate)]]
    count = min(max_victims, len(bank))
    positions = [0] if count == 1 else [round(j * (len(bank) - 1) / (count - 1)) for j in range(count)]
    return options + [[i for i in bank if i != bank[position]] + [int(candidate)] for position in positions]


def storage_logits_from_encoded(memory, short, state, candidate, bank_ids, encoded, options=None):
    """Shared inference/training chooser, reading only supplied completed rows.

    ``encoded`` is an ID->encoded-row mapping, allowing the online bank to keep
    at most capacity+1 rows instead of accumulating the entire episode.
    """
    cfg = memory.config
    options = storage_options(bank_ids, candidate, cfg.capacity, cfg.max_victims) if options is None else options
    ref = encoded[candidate]["event"]
    empty = ref.new_empty((0, cfg.hidden_dim))

    def rows(ids):
        return torch.stack([encoded[i]["event"] for i in ids]) if ids else empty

    now = encoded[candidate]["ends"]
    duration = (now - encoded[candidate]["starts"]) / cfg.time_scale
    oldest_age = (now - encoded[bank_ids[0]]["ends"]) / cfg.time_scale if bank_ids else now.new_zeros(())
    victims, metadata = [], []
    for index, option in enumerate(options):
        removed = [i for i in bank_ids if i not in option]
        victims.append(encoded[removed[0]]["event"] if removed else ref.new_zeros(cfg.hidden_dim))
        victim_age = (now - encoded[removed[0]]["ends"]) / cfg.time_scale if removed else now.new_zeros(())
        flags = ref.new_tensor([float(index == 0), float(index != 0 and not removed), float(bool(removed)),
                                len(bank_ids) / cfg.capacity, len(option) / cfg.capacity])
        metadata.append(torch.cat((flags, torch.stack((victim_age, duration, oldest_age)).to(ref.dtype))))
    logits = memory.writer(short, state, ref, rows(bank_ids), [rows(ids) for ids in options],
                           torch.stack(victims), torch.stack(metadata))
    return {"logits": logits, "options": options}


def storage_prediction(memory, episode, candidate, bank_ids, encoded=None):
    _validate_bank(memory, episode, candidate, bank_ids)
    if candidate >= len(episode["actions"]) or not bool(episode["transition_valid"][candidate]):
        raise ValueError("Storage candidate must be a valid completed event")
    device = next(memory.parameters()).device
    if encoded is None:
        encoded = encode_until(memory, episode, candidate + 1)
    if len(encoded["event"]) <= candidate:
        raise ValueError("Encoded events do not include completed candidate")
    needed = list(bank_ids) + [candidate]
    row_map = {i: {key: value[i] for key, value in encoded.items()} for i in needed}
    return storage_logits_from_encoded(memory,
        episode["short"][candidate + 1].to(device=device, dtype=torch.float32),
        episode["state"][candidate + 1].to(device=device, dtype=torch.float32),
        candidate, bank_ids, row_map)


@torch.no_grad()
def replay_bank(memory, episode, decision, policy="all", encoded=None):
    if policy not in ("all", "hard"):
        raise ValueError("V3 storage policy must be all or hard")
    _validate_bank(memory, episode, decision, [])
    if encoded is None:
        encoded = encode_until(memory, episode, decision)
    bank = []
    attempted = accepted = forced = learned_attempted = learned_accepted = replaced = learned_replaced = 0
    for candidate in range(decision):
        if not bool(episode["transition_valid"][candidate]):
            continue
        attempted += 1
        if policy == "all":
            replaced += int(len(bank) == memory.config.capacity)
            bank = (bank + [candidate])[-memory.config.capacity:]
            accepted += 1
        elif len(bank) < memory.config.min_fill:
            bank.append(candidate)
            accepted += 1
            forced += 1
        else:
            prediction = storage_prediction(memory, episode, candidate, bank, encoded)
            choice = int(prediction["logits"].argmax().item())
            learned_attempted += 1
            if choice:
                learned_accepted += 1
                accepted += 1
                replaced += int(len(bank) == memory.config.capacity)
                learned_replaced += int(len(bank) == memory.config.capacity)
            bank = prediction["options"][choice]
    return bank, {"write_rate": accepted / max(attempted, 1), "bank_fill": float(len(bank)),
                  "oldest_event_age": float(decision - bank[0]) if bank else 0.0,
                  "oldest_event_age_frames": float(episode["frames"][decision] - episode["frames"][bank[0] + 1]) if bank else 0.0,
                  "attempted_writes": attempted, "accepted_writes": accepted, "forced_writes": forced,
                  "learned_attempted": learned_attempted, "learned_accepted": learned_accepted,
                  "learned_rejected": learned_attempted - learned_accepted, "replaced": replaced,
                  "learned_replaced": learned_replaced,
                  "learned_write_rate": learned_accepted / max(learned_attempted, 1)}
