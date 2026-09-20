"""Action-value v3 targets: query-specific retrieval and bounded replacement.

Storage compares KEEP with the *same deployable* append/replacement options,
including full banks. Each option is frozen until sampled old-only future
queries; intervening writes are deliberately omitted. This is conditional
expected action loss, NOT a whole-episode policy gradient or success reward.

Retrieval compares leave-one-out losses in one identical bank at one query.
Only confidently ordered OLD events (or the zero marginal-value/null anchor)
are ranked. Ties/redundancy never create a forced positive. The uncertainty
screen is a small-sample heuristic, not a calibrated confidence interval.

Teacher losses are paired by exact seed (flow noise AND time), detached, and
versioned. Student event features are re-encoded with autograd on retrieval;
the storage API detaches writer inputs so its loss updates the writer only.
No future target is ever passed to a student reader or writer.

Builders return JSON-serializable labels; loss functions return
``{'loss': Tensor, 'metrics': dict}``. ``loss_fn`` is an optional replacement
for ``episode_flow_loss(head, episode, decision, fused_short, seed=seed)`` and
supports CPU tests without downloading a base model.
"""

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import numbers
from pathlib import Path
import random
import statistics

import torch
from torch.nn import functional as F

from .contextual_cvom import _atomic_json, _episode_layout, _teacher_fingerprint, eligible_futures
from .cache_reader_v3 import validate_decision
from .hamlet import episode_flow_loss
from .replay_v3 import encode_until, read_bank, storage_options, storage_prediction


DEFINITION = "action-value-old-only-retrieval-replacement-v3"


@dataclass(frozen=True)
class LabelV3Config:
    memory_window: int = 4
    future_samples: int = 2
    noise_samples: int = 2
    scale: float = 0.001
    margin: float = 0.00001
    uncertainty_z: float = 2.0
    max_retrieval_events: int = 4
    ranking_temperature: float = 1.0

    def __post_init__(self):
        for name in ("memory_window", "future_samples", "noise_samples", "max_retrieval_events"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.noise_samples < 2:
            raise ValueError("noise_samples must be >= 2 for paired-noise uncertainty")
        for name in ("scale", "margin", "uncertainty_z", "ranking_temperature"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.scale == 0 or self.ranking_temperature == 0:
            raise ValueError("scale and ranking_temperature must be strictly positive")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _check_frozen(teacher, head):
    for name, module in (("teacher", teacher), ("action expert", head)):
        if module is not None and (module.training or any(p.requires_grad for p in module.parameters())):
            raise ValueError(f"{name} must be frozen and in eval mode")


def _bank_context(episode, cutoff, bank_ids, capacity):
    _, valid, _ = _episode_layout(episode)
    if isinstance(cutoff, bool) or not isinstance(cutoff, numbers.Integral):
        raise ValueError("cutoff must be an integer")
    if cutoff < 0 or cutoff > len(valid):
        raise ValueError("cutoff is outside episode endpoints")
    ids = list(bank_ids)
    if any(isinstance(i, bool) or not isinstance(i, numbers.Integral) for i in ids):
        raise ValueError("bank_ids must contain integer event IDs")
    ids = [int(i) for i in ids]
    if ids != sorted(set(ids)):
        raise ValueError("bank_ids must be sorted and unique")
    if len(ids) > capacity:
        raise ValueError("bank exceeds capacity")
    if any(i < 0 or i >= cutoff or not bool(valid[i]) for i in ids):
        raise ValueError("bank contains future, unfinished, or invalid events")
    return ids


def _context(teacher, episode, cfg, seed, kind, identity=None):
    if not isinstance(cfg, LabelV3Config):
        raise TypeError("config must be LabelV3Config")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer")
    frames, valid, decisions = _episode_layout(episode)
    cache_fingerprint = episode.get("cache_fingerprint")
    if identity is not None:
        if identity["cache_fingerprint"] != cache_fingerprint:
            raise ValueError("Episode belongs to a different cache fingerprint")
    # Direct synthetic builders may omit provenance; production cached labels
    # require an explicit cache+expert identity in ActionValueLabelsV3.
    return {
        "definition": DEFINITION, "kind": kind, "config": asdict(cfg),
        "seed": int(seed), "episode_id": int(episode["episode_id"]),
        "teacher_fingerprint": _teacher_fingerprint(teacher),
        "cache_fingerprint": cache_fingerprint,
        "identity": identity,
        "layout_hash": _digest({"frames": frames.tolist(), "valid": valid.tolist(),
                                 "decisions": decisions.tolist()}),
    }


def _noise_seeds(seed, episode_id, index, decisions, repeats):
    return [[(int(seed) + episode_id * 100003 + index * 1009 + d * 9176 + r * 123457)
             % (2**63 - 1) for r in range(repeats)] for d in decisions]


def _storage_context(teacher, episode, candidate, bank_ids, cfg, seed, identity=None):
    # Eligibility validates the candidate and uses its END frame, not start.
    futures = eligible_futures(episode, candidate, cfg.memory_window)
    if not futures:
        raise ValueError("Candidate has no old-only future action decision")
    ids = _bank_context(episode, candidate, bank_ids, teacher.config.capacity)
    context = _context(teacher, episode, cfg, seed, "storage", identity)
    rng = random.Random(int(seed) + context["episode_id"] * 100003 + int(candidate) * 1009)
    count = min(cfg.future_samples, len(futures))
    selected = [rng.choice(futures[j * len(futures) // count:(j + 1) * len(futures) // count])
                for j in range(count)]
    options = storage_options(ids, int(candidate), teacher.config.capacity, teacher.config.max_victims)
    options = [list(option) for option in options]
    if not options or options[0] != ids or any(len(option) > teacher.config.capacity for option in options):
        raise ValueError("Storage options must start with KEEP and respect capacity")
    context.update(candidate=int(candidate), bank_ids=ids, options=options,
                   future_decisions=selected,
                   noise_seeds=_noise_seeds(seed, context["episode_id"], int(candidate), selected, cfg.noise_samples))
    return context


def _retrieval_context(teacher, episode, decision, bank_ids, cfg, seed, identity=None):
    frames, _, decisions = _episode_layout(episode)
    if isinstance(decision, bool) or not isinstance(decision, numbers.Integral):
        raise ValueError("decision must be an integer action query")
    if not 0 <= decision < len(decisions) or not bool(decisions[decision]):
        raise ValueError("decision must have a valid action target")
    ids = _bank_context(episode, decision, bank_ids, teacher.config.capacity)
    oldest_short = int(frames[max(0, decision - (cfg.memory_window - 1))])
    old_ids = [i for i in ids if int(frames[i + 1]) < oldest_short]
    context = _context(teacher, episode, cfg, seed, "retrieval", identity)
    rng = random.Random(int(seed) + context["episode_id"] * 100003 + int(decision) * 1009)
    selected = sorted(rng.sample(old_ids, min(len(old_ids), cfg.max_retrieval_events)))
    context.update(decision=int(decision), bank_ids=ids, sampled_event_ids=selected,
                   branch_banks=[ids] + [[i for i in ids if i != event] for event in selected],
                   noise_seeds=_noise_seeds(seed, context["episode_id"], int(decision), [int(decision)], cfg.noise_samples)[0])
    return context


def _scalar_loss(loss_fn, head, episode, decision, fused, seed):
    if "features" in episode:
        validate_decision(episode, decision)
    result = loss_fn(head, episode, decision, fused, seed=seed)
    loss = result["loss"] if isinstance(result, dict) else result
    value = float(loss.detach().item() if isinstance(loss, torch.Tensor) else loss)
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite paired teacher action loss")
    return value


def _gain_stats(rows, cfg):
    """Separate within-query Monte Carlo error and across-query heterogeneity."""
    means = [statistics.mean(row) for row in rows]
    within_variances = [statistics.variance(row) for row in rows]
    count = len(rows)
    within_se = math.sqrt(sum(var / len(row) for var, row in zip(within_variances, rows))) / count
    across_std = statistics.stdev(means) if count > 1 else 0.0
    se = max(within_se, across_std / math.sqrt(count))
    mean = statistics.mean(means)
    lower, upper = mean - cfg.uncertainty_z * se, mean + cfg.uncertainty_z * se
    return {"mean": mean, "se": se, "lower": lower, "upper": upper,
            "within_noise_se": within_se, "future_std": across_std,
            "confident_positive": lower > cfg.margin,
            "confident_negative": upper < -cfg.margin}


@torch.no_grad()
def _build_storage(teacher, head, episode, cfg, context, loss_fn):
    _check_frozen(teacher, head)
    options, decisions = context["options"], context["future_decisions"]
    encoded = encode_until(teacher, episode, max(decisions))
    draws = []  # [future query][paired flow draw][KEEP/append/replacement]
    for decision, seeds in zip(decisions, context["noise_seeds"]):
        branches = [read_bank(teacher, episode, decision, bank, encoded)["fused_short"] for bank in options]
        draws.append([[_scalar_loss(loss_fn, head, episode, decision, fused, seed) for fused in branches]
                      for seed in seeds])
    option_means = [statistics.mean([row[o] for future in draws for row in future]) for o in range(len(options))]
    stats = [_gain_stats([[row[0] - row[o] for row in future] for future in draws], cfg)
             for o in range(len(options))]
    record = {"definition": DEFINITION, "kind": "storage", "cache_key": _digest(context),
              "context": context, "episode_id": context["episode_id"],
              "candidate": context["candidate"], "bank_ids": context["bank_ids"], "options": options,
              "future_decisions": decisions, "noise_seeds": context["noise_seeds"],
              "loss_draws": draws, "option_mean_losses": option_means,
              "option_gain_stats": stats,
              "costs": [(value - option_means[0]) / cfg.scale for value in option_means],
              "best_option": min(range(len(options)), key=lambda i: option_means[i]),
              "confident_options": sum(s["confident_positive"] or s["confident_negative"] for s in stats[1:]),
              "scale": cfg.scale}
    _json(record)  # Fail before saving a non-serializable/nonfinite target.
    return record


@torch.no_grad()
def build_storage_label(teacher, head, episode, candidate, bank_ids, cfg, seed, loss_fn=None):
    """Detached KEEP-vs-append/replacement action losses in a causal context."""
    _check_frozen(teacher, head)
    context = _storage_context(teacher, episode, candidate, bank_ids, cfg, seed)
    return _build_storage(teacher, head, episode, cfg, context, loss_fn or episode_flow_loss)


@torch.no_grad()
def _build_retrieval(teacher, head, episode, cfg, context, loss_fn):
    _check_frozen(teacher, head)
    decision, ids, selected = context["decision"], context["bank_ids"], context["sampled_event_ids"]
    # No OLD candidate means no useful label; do not waste expert forwards.
    draws, gains, pairs, gain_stats = [], [], [], []
    if selected:
        encoded = encode_until(teacher, episode, decision)
        branches = [read_bank(teacher, episode, decision, bank, encoded)["fused_short"]
                    for bank in context["branch_banks"]]
        draws = [[_scalar_loss(loss_fn, head, episode, decision, fused, seed) for fused in branches]
                 for seed in context["noise_seeds"]]
        gains = [[row[index + 1] - row[0] for row in draws] for index in range(len(selected))]
        gain_stats = [_gain_stats([row], cfg) for row in gains]
        # The null anchor has zero *marginal* utility, not a fabricated GT event.
        choices = [ids.index(event) for event in selected] + [-1]
        evidence = gains + [[0.0] * cfg.noise_samples]
        for i in range(len(choices)):
            for j in range(i + 1, len(choices)):
                stats = _gain_stats([[a - b for a, b in zip(evidence[i], evidence[j])]], cfg)
                if stats["confident_positive"]:
                    better, worse = choices[i], choices[j]
                elif stats["confident_negative"]:
                    better, worse = choices[j], choices[i]
                else:
                    continue
                pairs.append({"better_index": better, "worse_index": worse,
                              "gain_difference": abs(stats["mean"]), "se": stats["se"]})
    record = {"definition": DEFINITION, "kind": "retrieval", "cache_key": _digest(context),
              "context": context, "episode_id": context["episode_id"], "decision": decision,
              "bank_ids": ids, "sampled_event_ids": selected,
              "noise_seeds": context["noise_seeds"], "loss_draws": draws,
              "paired_event_gains": gains, "event_gain_stats": gain_stats,
              "pairs": pairs, "ranking_temperature": cfg.ranking_temperature}
    _json(record)
    return record


@torch.no_grad()
def build_retrieval_label(teacher, head, episode, decision, bank_ids, cfg, seed, loss_fn=None):
    """Query-specific, uncertainty-screened OLD-event leave-one-out rankings."""
    _check_frozen(teacher, head)
    context = _retrieval_context(teacher, episode, decision, bank_ids, cfg, seed)
    return _build_retrieval(teacher, head, episode, cfg, context, loss_fn or episode_flow_loss)


def _validate_student_context(episode, label, kind):
    if label.get("definition") != DEFINITION or label.get("kind") != kind:
        raise ValueError("Wrong objective label definition/kind")
    if int(episode["episode_id"]) != label["episode_id"]:
        raise ValueError("Label belongs to a different episode")
    context = label["context"]
    if _digest(context) != label["cache_key"]:
        raise ValueError("Label context/cache key mismatch")
    frames, valid, decisions = _episode_layout(episode)
    layout = _digest({"frames": frames.tolist(), "valid": valid.tolist(), "decisions": decisions.tolist()})
    if context["layout_hash"] != layout:
        raise ValueError("Label belongs to a different episode layout")
    if context["cache_fingerprint"] != episode.get("cache_fingerprint"):
        raise ValueError("Label belongs to a different cache fingerprint")
    if context["bank_ids"] != label["bank_ids"]:
        raise ValueError("Label bank differs from context")
    _json(label)


def storage_loss(memory, episode, label):
    """Writer-only expected action loss over exactly deployable discrete options.

    KEEP anchoring removes an action-independent constant. No confidence mask,
    clipping, tiny-positive BCE, or per-context normalization changes the
    expectation. Negative objective values are valid (better than KEEP).
    """
    _validate_student_context(episode, label, "storage")
    if label["candidate"] != label["context"]["candidate"] or label["options"] != label["context"]["options"]:
        raise ValueError("Storage label candidate/options differ from context")
    prediction = storage_prediction(memory, episode, label["candidate"], label["bank_ids"])
    if [list(bank) for bank in prediction["options"]] != label["options"]:
        raise ValueError("Student and teacher storage options differ")
    logits = prediction["logits"].reshape(-1)
    costs = logits.new_tensor(label["costs"]).detach()
    if logits.shape != costs.shape or not bool(torch.isfinite(logits).all()):
        raise ValueError("Storage logits must be finite and match label options")
    probabilities = logits.softmax(dim=-1)
    loss = (probabilities * costs).sum()
    selected = int(probabilities.detach().argmax().item())
    return {"loss": loss, "metrics": {
        "storage_loss": float(loss.detach()), "storage_expected_gain": float(-loss.detach() * label["scale"]),
        "storage_write_probability": float(1 - probabilities[0].detach()),
        "storage_choice_agreement": float(selected == label["best_option"]),
        "storage_options": len(label["options"]),
        "storage_confident_options": label["confident_options"],
        "storage_best_gain": max(s["mean"] for s in label["option_gain_stats"]),
    }}


def retrieval_loss(memory, episode, label):
    """Re-encode student events and differentiate scores; targets stay detached."""
    _validate_student_context(episode, label, "retrieval")
    if label["decision"] != label["context"]["decision"]:
        raise ValueError("Retrieval decision differs from context")
    output = read_bank(memory, episode, label["decision"], label["bank_ids"])
    scores = output["event_scores"].reshape(-1)
    if len(scores) != len(label["bank_ids"]) or not bool(torch.isfinite(scores).all()):
        raise ValueError("Reader scores must be finite and aligned with bank IDs")
    null = output.get("null_score", scores.new_zeros(1)).reshape(-1)
    if len(null) != 1 or not bool(torch.isfinite(null).all()):
        raise ValueError("Reader null score must be one finite scalar")
    all_scores = torch.cat([scores, null])
    terms, correct = [], []
    for pair in label["pairs"]:
        better, worse = pair["better_index"], pair["worse_index"]
        if any(not isinstance(i, int) or i < -1 or i >= len(scores) for i in (better, worse)):
            raise ValueError("Invalid retrieval pair bank index")
        gap = all_scores[better] - all_scores[worse]
        terms.append(F.softplus(-gap / label["ranking_temperature"]))
        correct.append(float(gap.detach() > 0))
    loss = torch.stack(terms).mean() if terms else all_scores.sum() * 0.0
    if not loss.requires_grad:
        # Empty-bank warmup has no attention scores; callers may still backward.
        loss = loss + next(memory.parameters()).sum() * 0.0
    return {"loss": loss, "metrics": {
        "retrieval_loss": float(loss.detach()), "retrieval_pairs": len(terms),
        "retrieval_rank_accuracy": statistics.mean(correct) if correct else None,
        "retrieval_old_candidates": len(label["sampled_event_ids"]),
        "retrieval_null_pairs": sum(p["better_index"] == -1 or p["worse_index"] == -1 for p in label["pairs"]),
    }}


class ActionValueLabelsV3:
    """Immutable teacher snapshot and exact-context JSON cache.

    ``identity`` must identify cache+base-expert provenance and teacher_version.
    Refresh by constructing a NEW labeler in a NEW directory. Copying the
    small memory teacher by default never freezes/mutates the live student.
    The base expert is shared read-only and must already be frozen/eval.
    """

    def __init__(self, teacher, action_head, config, output_dir, identity, *, copy_teacher=True):
        if not isinstance(config, LabelV3Config):
            raise TypeError("config must be LabelV3Config")
        if not isinstance(identity, dict) or any(identity.get(k) in (None, "")
                                                for k in ("cache_fingerprint", "teacher_version")):
            raise ValueError("identity must include cache_fingerprint and teacher_version")
        self.teacher = copy.deepcopy(teacher) if copy_teacher else teacher
        self.teacher.eval().requires_grad_(False)
        self.action_head, self.config = action_head, config
        _check_frozen(self.teacher, action_head)
        self.identity = copy.deepcopy(identity)
        self.path = Path(output_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest = {"definition": DEFINITION, "config": asdict(config),
                         "identity": self.identity, "teacher_fingerprint": _teacher_fingerprint(self.teacher)}
        path = self.path / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != self.manifest:
                raise ValueError("Label cache belongs to a different teacher/data/configuration")
        else:
            _atomic_json(path, self.manifest)

    def _get(self, kind, episode, index, bank_ids, seed, loss_fn):
        _check_frozen(self.teacher, self.action_head)
        context_fn = _storage_context if kind == "storage" else _retrieval_context
        builder = _build_storage if kind == "storage" else _build_retrieval
        context = context_fn(self.teacher, episode, index, bank_ids, self.config, seed, self.identity)
        if context["teacher_fingerprint"] != self.manifest["teacher_fingerprint"]:
            raise ValueError("Frozen teacher changed; refresh with a new version/directory")
        key = _digest(context)
        path = self.path / f"{kind}-episode-{context['episode_id']:06d}-{index:06d}-{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("cache_key") != key or record.get("context") != context:
                raise ValueError("Label cache context/key mismatch")
            _validate_student_context(episode, record, kind)
            return record
        record = builder(self.teacher, self.action_head, episode, self.config, context, loss_fn or episode_flow_loss)
        _atomic_json(path, record)
        return record

    def get_storage(self, episode, candidate, bank_ids, seed, loss_fn=None):
        return self._get("storage", episode, candidate, bank_ids, seed, loss_fn)

    def get_retrieval(self, episode, decision, bank_ids, seed, loss_fn=None):
        return self._get("retrieval", episode, decision, bank_ids, seed, loss_fn)
