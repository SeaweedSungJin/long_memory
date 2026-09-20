"""Fixed-query, paired-noise action controls for Stage-2 v2.

These are offline flow-matching losses, not simulator success probabilities.
FIFO/random controls match the learned bank's occupancy at each query; first-N
uses the runtime minimum fill, and all/FIFO uses the configured maximum capacity.
All models are the *same current reader*: only bank selection changes.
"""

import math
from numbers import Integral
import random

import numpy as np
import torch

from .hamlet import episode_flow_loss
from .replay import encode_until, read_bank, replay_bank
from .stage2_objectives import fixed_bank


def _module_modes(memory, head):
    modules = list(memory.modules())
    if isinstance(head, torch.nn.Module):
        modules.extend(head.modules())
    # Avoid duplicate objects in case a future integration shares a submodule.
    return {module: module.training for module in modules}


@torch.no_grad()
def validate_actions(memory, head, fetch, plan, seed):
    """Return finite scalar means on a deterministic [(episode, decision, ...)].

    The third plan field (wrong episode) is currently unused: occupancy-matched
    bank-policy controls are the readiness test here. No labels or parameters
    are updated. Train/eval modes and torch RNG states are restored even when
    a fetch/head call fails; random control sampling uses a private NumPy RNG.

    learned_write_accepts/attempts count *replayed query histories*, potentially
    including the same historical event under multiple validation queries.
    They exclude the first forced min_fill valid writes and must not be confused
    with unique events or new simulator rollouts.
    """
    plan = list(plan)
    if not plan:
        raise ValueError("action validation requires a nonempty fixed plan")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    modes = _module_modes(memory, head)
    python_rng_state, numpy_rng_state = random.getstate(), np.random.get_state()
    device_indices = set()
    for module in (memory, head):
        if isinstance(module, torch.nn.Module):
            for parameter in module.parameters():
                if parameter.device.type == "cuda":
                    device_indices.add(parameter.device.index)
    rows, learned_accepts, learned_attempts = [], 0, 0
    # Usually episode_flow_loss already isolates its noise RNG; the outer guard
    # also protects callers if a future head contains additional stochastic ops.
    with torch.random.fork_rng(devices=sorted(device_indices)):
        try:
            memory.eval()
            if isinstance(head, torch.nn.Module):
                head.eval()
            for query_index, item in enumerate(plan):
                if len(item) < 2:
                    raise ValueError("plan entries require episode ID and decision")
                episode = fetch(item[0])
                decision = item[1]
                if not isinstance(decision, Integral) or isinstance(decision, bool):
                    raise ValueError("validation decision must be an integer")
                decision = int(decision)
                if not 0 <= decision <= len(episode["actions"]):
                    raise ValueError("validation decision out of bounds")
                if "decision_mask" in episode and (decision >= len(episode["decision_mask"])
                                                    or not bool(episode["decision_mask"][decision])):
                    raise ValueError("validation decision has no action target")
                encoded = encode_until(memory, episode, decision)
                bank, stats = replay_bank(memory, episode, decision, "hard", encoded)
                paired_seed = int(seed) + query_index * 1009
                read = read_bank(memory, episode, decision, bank, encoded)

                # Deduplicate identical bank controls. Identical inputs with the
                # same flow seed are the same comparison, not independent draws.
                bank_losses = {}

                def loss_for_ids(ids, supplied_read=None):
                    key = tuple(ids)
                    if key not in bank_losses:
                        bank_read = supplied_read or read_bank(memory, episode, decision, ids, encoded)
                        output = episode_flow_loss(head, episode, decision, bank_read["fused_short"], seed=paired_seed)
                        bank_losses[key] = output
                    return bank_losses[key]

                actual = loss_for_ids(bank, read)
                baseline = episode_flow_loss(head, episode, decision, seed=paired_seed)
                controls = {
                    "first": fixed_bank(episode, decision, memory.config.min_fill, "first"),
                    "fifo": fixed_bank(episode, decision, len(bank), "fifo"),
                    "random": fixed_bank(episode, decision, len(bank), "random", seed=paired_seed + 31337),
                    "all": fixed_bank(episode, decision, memory.config.capacity, "fifo"),
                }
                row = {"action_loss": float(actual["loss"]),
                       "baseline_action_loss": float(baseline["loss"]), **stats}
                row["memory_gain"] = row["baseline_action_loss"] - row["action_loss"]
                if "velocity_mae" in actual:
                    row["velocity_mae"] = float(actual["velocity_mae"])
                for name, ids in controls.items():
                    row[name + "_action_loss"] = float(loss_for_ids(ids)["loss"])
                for name in ("gate_mean", "read_norm", "residual_norm", "null_weight"):
                    row[name] = float(read[name].mean())
                valid_count = int(episode["transition_valid"][:decision].sum())
                forced_count = min(memory.config.min_fill, valid_count)
                accepted_count = round(stats["write_rate"] * valid_count)
                learned_accepts += accepted_count - forced_count
                learned_attempts += valid_count - forced_count
                if not forced_count <= accepted_count <= valid_count:
                    raise ValueError("replay write counters violate min-fill invariant")
                if not all(math.isfinite(float(value)) for value in row.values()):
                    raise ValueError("non-finite action validation metric")
                if any(value < 0 for name, value in row.items() if name.endswith("action_loss")):
                    raise ValueError("action loss must be nonnegative")
                rows.append(row)
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            # Set individual flags instead of .train(): preserve mixed submodule
            # modes exactly, including explicitly frozen/eval child modules.
            for module, training in modes.items():
                module.training = training
    keys = set().union(*(row.keys() for row in rows))
    metrics = {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}
    metrics.update(action_validation_count=float(len(rows)),
                   learned_write_accepts=float(learned_accepts),
                   learned_write_attempts=float(learned_attempts),
                   learned_write_rate=learned_accepts / learned_attempts if learned_attempts else 0.0)
    return metrics


def action_guard(metrics, tolerance=0.05):
    """Require hard-bank action loss not to degrade beyond all/FIFO tolerance.

    This local guard complements utility/write discrimination checks. Passing
    it is neither statistical significance nor a closed-loop success claim.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("action tolerance must be finite and nonnegative")
    reasons = []
    values = {}
    for name in ("action_loss", "all_action_loss"):
        raw = metrics.get(name)
        if raw is None or not math.isfinite(float(raw)) or float(raw) < 0:
            reasons.append(f"missing or invalid {name}")
        else:
            values[name] = float(raw)
    if reasons:
        return False, reasons
    if values["action_loss"] > values["all_action_loss"] * (1 + tolerance):
        reasons.append(f"hard-bank action loss exceeds all/FIFO reference by more than {100 * tolerance:g}%")
    return not reasons, reasons
