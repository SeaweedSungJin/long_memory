"""Stage-2 v2 labels for one *specific* candidate/bank context.

Unlike ``cvom.py`` v1, a label is not shared across different bank coalitions.
The caller must pass the same sorted ``bank_ids`` to its utility/write inputs.
Only queries after the entire candidate has left HAMLET's short window are
eligible. Both frozen-teacher branches omit all intervening events and differ
only by candidate inclusion: this is a conditional addition surrogate, NOT a
rollout reward, causal effect, Shapley value, or eviction/replacement value.

Repeated paired flow draws retain the original action objective and its Beta
timestep distribution. The uncertainty band is a filtering heuristic, NOT a
calibrated confidence interval with guaranteed coverage. Future-query variation
and within-query Monte Carlo uncertainty are recorded separately. All labels
are versioned by frozen teacher weights, data fingerprint, and configuration.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numbers
from pathlib import Path
import random
import statistics
import tempfile

import torch

from .hamlet import episode_flow_loss
from .replay import encode_until, read_bank


DEFINITION = "fixed-context-old-only-log-utility-v2"


@dataclass(frozen=True)
class ContextualCVoMConfig:
    memory_window: int = 4
    future_samples: int = 4
    noise_samples: int = 4
    utility_scale: float = 0.001
    write_delta: float = 0.00001
    uncertainty_z: float = 2.0
    seed: int = 71

    def __post_init__(self):
        for name in ("memory_window", "future_samples", "noise_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.noise_samples < 2:
            raise ValueError("noise_samples must be >= 2 to estimate paired-noise variability")
        for name in ("utility_scale", "write_delta", "uncertainty_z"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.utility_scale == 0:
            raise ValueError("utility_scale must be strictly positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _episode_layout(episode):
    frames = torch.as_tensor(episode["frames"])
    valid = torch.as_tensor(episode["transition_valid"])
    decisions = torch.as_tensor(episode["decision_mask"])
    if frames.ndim != 1 or len(frames) < 2 or frames.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    ):
        raise ValueError("frames must be a one-dimensional integer endpoint sequence")
    if not bool((frames[1:] > frames[:-1]).all()) or bool((frames < 0).any()):
        raise ValueError("frames must be nonnegative and strictly increasing")
    if valid.shape != (len(frames) - 1,) or valid.dtype != torch.bool:
        raise ValueError("transition_valid must be a boolean mask over completed events")
    # The production cache has T events/action targets but T+1 observations.
    # Its final observation is an endpoint only, never an actionable query.
    if decisions.shape != valid.shape or decisions.dtype != torch.bool:
        raise ValueError("decision_mask must be a boolean mask over T action decisions, not T+1 endpoints")
    return frames, valid, decisions


def eligible_futures(episode, candidate, memory_window=4):
    """Queries whose oldest cached short endpoint is AFTER candidate end.

    ``build_episode`` feeds one cached endpoint per frozen-HAMLET call; the
    inference path shifts its rolling FIFO once per call. Thus the actual short
    window is the last K cached endpoints, including repeated endpoint zero at
    startup. It is NOT an independently recomputed nominal frame-stride window
    when endpoints are irregular. The terminal endpoint has no action target.

    Event i ends at frames[i+1], not frames[i]. The strict inequality excludes
boundary overlap: for an ordinary K=4 grid, d=i+4 is still too early and
d=i+5 is the first potentially eligible query. ``decision_mask`` also excludes
passive demonstration endpoints without action supervision.
"""
    if not isinstance(memory_window, int) or isinstance(memory_window, bool) or memory_window < 1:
        raise ValueError("memory_window must be a positive integer")
    frames, valid, decisions = _episode_layout(episode)
    if not isinstance(candidate, numbers.Integral) or isinstance(candidate, bool):
        raise ValueError("candidate must be an integer event index")
    if candidate < 0 or candidate >= len(valid) or not bool(valid[candidate]):
        raise ValueError("candidate must identify a valid completed event")
    end = int(frames[candidate + 1])
    return [
        decision for decision in torch.where(decisions)[0].tolist()
        if decision > candidate
        and end < int(frames[max(0, decision - (memory_window - 1))])
    ]


def eligible_candidates(episode, memory_window=4):
    """Only valid events with at least one genuinely old-only action query."""
    _, valid, _ = _episode_layout(episode)
    return [i for i in torch.where(valid)[0].tolist() if eligible_futures(episode, i, memory_window)]


def validate_context(episode, candidate, bank_ids, capacity):
    """Return canonical bank IDs; reject future/duplicate/full-bank contexts.

Addition labels reserve one slot for the candidate. When evaluating a full
FIFO bank, the caller must explicitly select its post-eviction coalition first;
this function never silently removes or rearranges an event.
"""
    _, valid, _ = _episode_layout(episode)
    if not isinstance(candidate, numbers.Integral) or isinstance(candidate, bool):
        raise ValueError("candidate must be an integer event index")
    if not 0 <= candidate < len(valid) or not bool(valid[candidate]):
        raise ValueError("candidate must identify a valid completed event")
    ids = list(bank_ids)
    if any(not isinstance(i, numbers.Integral) or isinstance(i, bool) for i in ids):
        raise ValueError("bank_ids must contain integer event indices")
    ids = [int(i) for i in ids]
    if ids != sorted(set(ids)):
        raise ValueError("bank_ids must be sorted and unique")
    if any(i < 0 or i >= candidate or not bool(valid[i]) for i in ids):
        raise ValueError("bank_ids must contain only valid events strictly before the candidate")
    if not isinstance(capacity, int) or capacity < 1 or len(ids) > capacity - 1:
        raise ValueError("bank must reserve one capacity slot for candidate addition")
    return ids


def sample_context(episode, candidate, capacity, rng):
    """Uniform-size past coalition, sampled independently of future GT actions."""
    _, valid, _ = _episode_layout(episode)
    validate_context(episode, candidate, [], capacity)
    past = [i for i in range(candidate) if bool(valid[i])]
    count = rng.randint(0, min(len(past), capacity - 1))
    return sorted(rng.sample(past, count))


def _teacher_fingerprint(teacher):
    """Hash the small memory module, never the multi-billion-parameter expert."""
    digest = hashlib.sha256()
    for name, tensor in sorted(teacher.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Nonfinite teacher parameter: {name}")
        digest.update(_json([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    config = asdict(teacher.config)
    digest.update(_json(config).encode())
    return digest.hexdigest()


def _atomic_json(path, payload):
    # Unique temporary files allow independent contexts to be produced safely.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class ContextualCVoMLabels:
    """Immutable frozen-teacher snapshot and its bank-conditioned label cache.

``identity`` must identify ``cache_fingerprint`` and ``teacher_version``. The
action expert identity should be covered by the cache fingerprint, as in the
existing HAMLET episode cache. To refresh a teacher, construct a NEW instance
in a NEW version directory. Never mutate this instance's teacher weights.
"""

    def __init__(self, teacher, action_head, config, output_dir, identity):
        if not isinstance(config, ContextualCVoMConfig):
            raise TypeError("config must be ContextualCVoMConfig")
        if not isinstance(identity, dict) or any(
            key not in identity or identity[key] is None or identity[key] == ""
            for key in ("cache_fingerprint", "teacher_version")
        ):
            raise ValueError("identity must include cache_fingerprint and teacher_version")
        self.teacher, self.action_head, self.config = teacher, action_head, config
        teacher.eval().requires_grad_(False)
        if action_head is not None:
            action_head.eval().requires_grad_(False)
        self.path = Path(output_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "definition": DEFINITION, "config": asdict(config), "identity": identity,
            "teacher_fingerprint": _teacher_fingerprint(teacher),
        }
        self.identity_hash = _digest(self.manifest)
        manifest_path = self.path / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != self.manifest:
                raise ValueError("Contextual CVoM cache belongs to a different teacher/data/configuration")
        else:
            _atomic_json(manifest_path, self.manifest)

    def _context(self, episode, candidate, bank_ids):
        ids = validate_context(episode, candidate, bank_ids, self.teacher.config.capacity)
        futures = eligible_futures(episode, candidate, self.config.memory_window)
        if not futures:
            raise ValueError("Candidate has no old-only future action decision")
        eid = int(episode["episode_id"])
        # Stratify by future index for delay coverage, rather than choosing four
        # adjacent decisions. Selection is reproducible and independent of S.
        rng = random.Random(self.config.seed + eid * 100003 + int(candidate) * 1009)
        count = min(len(futures), self.config.future_samples)
        selected = [rng.choice(futures[j * len(futures) // count:(j + 1) * len(futures) // count])
                    for j in range(count)]
        seeds = [[(self.config.seed + eid * 100003 + int(candidate) * 1009
                   + decision * 9176 + repeat * 123457) % (2**63 - 1)
                  for repeat in range(self.config.noise_samples)] for decision in selected]
        frames = torch.as_tensor(episode["frames"]).tolist()
        return {
            "identity_hash": self.identity_hash,
            "episode_id": eid, "candidate": int(candidate), "bank_ids": ids,
            "future_decisions": selected, "noise_seeds": seeds,
            "layout_hash": _digest({"frames": frames,
                                    "transition_valid": episode["transition_valid"].tolist(),
                                    "decision_mask": episode["decision_mask"].tolist()}),
        }

    @torch.no_grad()
    def get(self, episode, candidate, bank_ids):
        if self.teacher.training or any(p.requires_grad for p in self.teacher.parameters()):
            raise ValueError("Contextual teacher must remain frozen and in eval mode")
        if self.action_head is not None and (
            self.action_head.training or any(p.requires_grad for p in self.action_head.parameters())
        ):
            raise ValueError("Action expert must remain frozen and in eval mode")
        context = self._context(episode, candidate, bank_ids)
        key = _digest(context)
        path = self.path / f"episode-{context['episode_id']:06d}-event-{int(candidate):06d}-{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("context") != context or record.get("cache_key") != key:
                raise ValueError("Contextual label cache key/context mismatch")
            self._validate_record(record)
            return record
        selected = context["future_decisions"]
        encoded = encode_until(self.teacher, episode, max(selected))
        gains, losses_without, losses_with = [], [], []
        for decision, seeds in zip(selected, context["noise_seeds"]):
            minus = read_bank(self.teacher, episode, decision, context["bank_ids"], encoded)
            plus = read_bank(self.teacher, episode, decision, context["bank_ids"] + [int(candidate)], encoded)
            row, without_row, with_row = [], [], []
            for noise_seed in seeds:
                loss0 = episode_flow_loss(self.action_head, episode, decision, minus["fused_short"], seed=noise_seed)["loss"]
                loss1 = episode_flow_loss(self.action_head, episode, decision, plus["fused_short"], seed=noise_seed)["loss"]
                # Promote before subtraction: avoid low-precision cancellation
                # in the subtraction itself (expert outputs still use its dtype).
                value0, value1 = float(loss0.item()), float(loss1.item())
                if not math.isfinite(value0) or not math.isfinite(value1):
                    raise FloatingPointError("Nonfinite contextual teacher loss")
                without_row.append(value0)
                with_row.append(value1)
                row.append(value0 - value1)
            gains.append(row)
            losses_without.append(without_row)
            losses_with.append(with_row)
        means = [statistics.mean(row) for row in gains]
        within_vars = [statistics.variance(row) for row in gains]
        gain = statistics.mean(means)
        n_future, n_noise = len(gains), self.config.noise_samples
        within_se = math.sqrt(sum(v / n_noise for v in within_vars)) / n_future
        future_std = statistics.stdev(means) if n_future > 1 else 0.0
        # Future mean dispersion includes real utility heterogeneity plus some
        # MC error. Do not add it again to the within-query variance estimate.
        # Taking the larger SE is a deliberate screening heuristic, not a proof
        # of conservative frequentist coverage with only a handful of queries.
        gain_se = max(within_se, future_std / math.sqrt(n_future))
        lower, upper = gain - self.config.uncertainty_z * gain_se, gain + self.config.uncertainty_z * gain_se
        confident = lower > self.config.write_delta or upper < self.config.write_delta
        record = {
            "definition": DEFINITION, "cache_key": key, "context": context,
            "episode_id": context["episode_id"], "candidate": int(candidate),
            "bank_ids": context["bank_ids"], "future_decisions": selected,
            "noise_seeds": context["noise_seeds"], "paired_gains": gains,
            "losses_without": losses_without, "losses_with": losses_with,
            "future_mean_gains": means, "within_noise_variances": within_vars,
            "within_noise_se": within_se, "future_mean_std": future_std,
            "gain_se": gain_se, "gain_lower": lower, "gain_upper": upper,
            "signed_gain": gain,
            "utility_target": math.log1p(max(gain, 0.0) / self.config.utility_scale),
            "utility_weight": 1.0, "write_target": int(gain > self.config.write_delta),
            "write_weight": float(confident), "label_confident": bool(confident),
        }
        self._validate_record(record)
        _atomic_json(path, record)
        return record

    @staticmethod
    def _validate_record(record):
        def check(value):
            if isinstance(value, float) and not math.isfinite(value):
                raise FloatingPointError("Nonfinite contextual label cache value")
            if isinstance(value, dict):
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
        check(record)
        for field in ("signed_gain", "utility_target", "write_target", "write_weight", "label_confident"):
            if field not in record:
                raise ValueError(f"Contextual label is missing {field}")
        if record["utility_target"] < 0 or record["write_target"] not in (0, 1):
            raise ValueError("Invalid contextual utility/write target")
        if record["write_weight"] not in (0.0, 1.0) or record["write_weight"] != float(record["label_confident"]):
            raise ValueError("Invalid contextual label confidence weight")
