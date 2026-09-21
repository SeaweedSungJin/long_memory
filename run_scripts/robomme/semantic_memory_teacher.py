"""Operation-level CVoM labels, NOT robot-success or optimal-storage labels.

At a real causal overflowing bank, change exactly ONE write operation, then
use common FIFO continuation for all branches. Match future queries/noise.
The teacher is the fixed current actor snapshot during a refresh (no optimizer
steps). Only TRAIN contexts train the controller; VAL is diagnostic only.
"""
import math
import random
import statistics

import torch
from torch.nn import functional as F

from run_scripts.robomme.flow_objective_v19 import episode_flow_v19
from run_scripts.robomme.semantic_memory_core import semantic_read


def context_plan(cache, episodes, tasks, capacity, count, seed, *, split="train", short_window=4):
    """Task-balanced causal decision points, including passive demo writes.

    Require at least one future action query beyond the original short window.
    No successful/outcome-based selection and no access to benchmark TEST.
    """
    rng, groups = random.Random(seed), {}
    for eid in sorted(cache.manifest["splits"][split]):
        eid = int(eid)
        ep = episodes.fetch(eid)
        decisions = torch.where(ep["decision_mask"])[0].tolist()
        if not decisions:
            continue
        for event in range(capacity, decisions[-1] - short_window):
            future = [d for d in decisions if d > event]
            far = [d for d in future if d > event + short_window]
            if far:
                groups.setdefault(tasks[eid], []).append({"episode_id": eid, "event": event,
                    "future": sorted(set((future[0], rng.choice(far))))})
    if not groups:
        raise ValueError(f"No {split} overflowing causal contexts with a long-range future")
    for rows in groups.values():
        rng.shuffle(rows)
    selected = []
    while len(selected) < count and any(groups.values()):
        for task in sorted(groups):
            if groups[task] and len(selected) < count:
                selected.append({**groups[task].pop(), "task": task})
    return selected


def replay_bank(core, manager, encoded, event, *, trace=None, return_trace=False):
    bank = core.initial_bank()
    selected = []
    if trace is not None and len(trace) != event:
        raise ValueError("Teacher write trace does not cover the causal prefix")
    for index in range(event):
        value = encoded["stored"][index:index+1]
        if trace is None:
            operation = manager.choose(bank, value)
        else:
            lookup = {op.id: op for op in manager.operations(bank, value)}
            operation = lookup[trace[index]]
        selected.append(operation.id)
        bank = manager.apply(bank, value, operation)
    result = (bank, encoded["stored"][event:event+1])
    return (*result, selected) if return_trace else result


def shortlist(manager, bank, candidate, limit, rng):
    operations = manager.operations(bank, candidate)
    lookup = {op.id: op for op in operations}
    mandatory = [lookup["keep"], manager.fifo_operation(bank, candidate)]
    rest = [op for op in operations if op.id not in {p.id for p in mandatory}]
    # Include the current preferred alternative and rotate other victims. Do
    # not label only FIFO and then pretend all victim choices were supervised.
    scores = manager.scores(bank, candidate).detach().cpu().tolist()
    ranked = sorted(zip(scores, operations), key=lambda pair: (-pair[0], pair[1].id))
    preferred = next((op for _, op in ranked if op in rest), None)
    if preferred is not None and len(mandatory) < limit:
        mandatory.append(preferred)
        rest.remove(preferred)
    rng.shuffle(rest)
    return mandatory + rest[:max(0, limit-len(mandatory))]


@torch.no_grad()
def label_contexts(core, head, answers, targets, manager, episodes, contexts, *,
                   seed, noise_samples=2, candidate_count=5, answer_weight=.01,
                   tail_weight=.25, ambiguity_margin=1e-6):
    from gr00t.long_memory.checkpoint_v4 import _state_sha256
    from gr00t.long_memory.expert_v4 import expert_state_sha256
    teacher_id = {"core": _state_sha256(core.delta_state_dict()),
                  "expert": expert_state_sha256(head), "answers": _state_sha256(answers.state_dict()),
                  "storage": _state_sha256(manager.state_dict())}
    labels, all_gains = [], []
    rng = random.Random(seed)
    for number, context in enumerate(contexts):
        ep, event = episodes.fetch(context["episode_id"]), context["event"]
        encoded = core.encode_prefix(ep, max(context["future"]) + 1)
        bank, candidate, trace = replay_bank(core, manager, encoded, event, return_trace=True)
        operations = shortlist(manager, bank, candidate, candidate_count, rng)
        losses, flow_losses, answer_losses = [], [], []
        for operation in operations:
            branch = manager.apply(bank, candidate, operation)
            samples, flows, semantics = [[] for _ in range(noise_samples)], [], []
            for d in range(event+1, max(context["future"])+1):
                if d in context["future"]:
                    fused, retrieved, _ = semantic_read(core, encoded["short"][d:d+1],
                        encoded["query"][d:d+1], branch)
                    target = targets.get(context["episode_id"], d)
                    aux, _ = answers.loss(retrieved, target)
                    semantics.append(float(aux))
                    for repeat in range(noise_samples):
                        flow = episode_flow_v19(head, ep, d, fused,
                            seed=seed + number*100003 + d*101 + repeat,
                            tail_weight=tail_weight, activation_checkpointing=False)
                        value = float(flow["loss"])
                        flows.append(value)
                        samples[repeat].append(value + answer_weight*float(aux))
                # EXACT common FIFO continuation; current future not visible
                # to the READ immediately above. No action or label is written.
                branch = core.write_fifo(branch, encoded["stored"][d:d+1])
            losses.append([statistics.fmean(row) for row in samples])
            flow_losses.append(statistics.fmean(flows))
            answer_losses.append(statistics.fmean(semantics))
        values = torch.tensor(losses, dtype=torch.float64)
        paired = values[0:1] - values  # operation 0 is KEEP; signed gains.
        gains = paired.mean(1)
        stderr = paired.std(1, unbiased=True) / math.sqrt(noise_samples) if noise_samples > 1 else torch.zeros_like(gains)
        informative = (gains.abs() > torch.maximum(stderr, torch.full_like(stderr, ambiguity_margin)))
        informative[0] = False
        labels.append({**context, "prefix_operations": trace, "operations": [op.id for op in operations],
            "gains": gains.tolist(), "gain_stderr": stderr.tolist(), "informative": informative.tolist(),
            "flow_losses": flow_losses, "answer_losses": answer_losses})
        all_gains.extend(abs(float(g)) for g, use in zip(gains, informative) if bool(use))
    scale = max(statistics.median(all_gains), ambiguity_margin) if all_gains else ambiguity_margin
    return {"contexts": labels, "scale": scale, "teacher": teacher_id,
        "continuation": "common_FIFO_after_single_write_intervention",
        "noise_samples": noise_samples, "answer_weight": answer_weight, "tail_weight": tail_weight,
        "confidence_rule": "abs(paired mean) > max(1 standard error, absolute margin); heuristic, not a hypothesis test",
        "informative_operations": len(all_gains), "total_contexts": len(labels),
        "interpretation": "fixed-teacher future action/answer utility; NOT rollout success or optimal long-term return"}


def controller_loss(core, manager, episodes, label_pack, rows):
    """Train operation scorer only; encoder is trained by action/answer losses."""
    losses, accurate, count = [], 0, 0
    for row in rows:
        if not any(row["informative"]):
            continue
        with torch.no_grad():
            encoded = core.encode_prefix(episodes.fetch(row["episode_id"]), row["event"]+1)
            # Recreate the labelled bank membership, not a different bank from
            # a controller updated since refresh. Features are re-encoded with
            # current weights; teacher identity/refresh makes that drift explicit.
            bank, candidate = replay_bank(core, manager, encoded, row["event"], trace=row["prefix_operations"])
        operations = manager.operations(bank, candidate)
        indices = {op.id: i for i, op in enumerate(operations)}
        scores = manager.scores(bank.detach(), candidate.detach())
        selected = scores[[indices[name] for name in row["operations"]]]
        gains = selected.new_tensor(row["gains"]) / label_pack["scale"]
        active = torch.tensor(row["informative"], device=selected.device, dtype=torch.bool)
        # Calibrate signed score relative to KEEP; include both negative and
        # positive operations. Ambiguous alternatives get no artificial label.
        relative = selected - selected[0]
        losses.append(F.smooth_l1_loss(relative[active], gains[active]))
        accurate += int(((relative[active] > 0) == (gains[active] > 0)).sum())
        count += int(active.sum())
    loss = torch.stack(losses).mean() if losses else sum(p.sum()*0 for p in manager.parameters())
    return loss, {"writer_loss": float(loss.detach()), "writer_sign_accuracy": accurate/max(count, 1),
                  "writer_labeled_operations": count, "writer_contexts": len(losses)}
