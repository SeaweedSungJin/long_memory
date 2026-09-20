"""Continuation-aware, confidence-screened writer targets for v5.

Unlike v3's fixed-bank target, every deployable KEEP/APPEND/REPLACE branch
processes all intervening completed events using one immutable writer snapshot.
The observations and executed actions remain the recorded demonstration. This
is *teacher-forced memory replay*, not an environment counterfactual or reward.

Action flow noise and time are paired between options. Optional split B is
independent of the split A used to construct training costs. Small-sample
uncertainty screens are heuristics, never calibrated confidence intervals.
Collapsed banks share forwards and are not counted as independent contexts.

``recall_cost(read_output, episode, decision)`` optionally supplies a detached
auxiliary recall cost through the SAME reader output. Any target access inside
that callback is supervision only; neither reader nor writer receives targets.
The caller must freeze/version its recall head and all target metadata.
"""

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import numbers
from pathlib import Path
import statistics

import torch

from .contextual_cvom import _atomic_json, _episode_layout, _teacher_fingerprint, eligible_futures
from .diagnostic_interventions import continuation_bank
from .hamlet import episode_flow_loss
from .objectives_v3 import (
    LabelV3Config, _bank_context, _check_frozen, _context, _digest, _gain_stats,
    _json, _scalar_loss, _storage_context,
)
from .replay_v3 import encode_until, read_bank, storage_options, storage_prediction


DEFINITION = "teacher-forced-continuation-screened-cost-v5"


@dataclass(frozen=True)
class LabelV5Config(LabelV3Config):
    continuation_policy: str = "hard"
    audit_noise_samples: int = 0
    recall_weight: float = 0.0
    confidence_screen: bool = True
    max_continuation_events: int = 4096
    max_teacher_forwards: int = 1024

    def __post_init__(self):
        super().__post_init__()
        if self.continuation_policy not in ("hard", "all"):
            raise ValueError("continuation_policy must be hard or all")
        if (isinstance(self.audit_noise_samples, bool)
                or not isinstance(self.audit_noise_samples, int)
                or self.audit_noise_samples < 0 or self.audit_noise_samples == 1):
            raise ValueError("audit_noise_samples must be zero or an integer >= 2")
        if (isinstance(self.recall_weight, bool) or not isinstance(self.recall_weight, numbers.Real)
                or not math.isfinite(self.recall_weight) or self.recall_weight < 0):
            raise ValueError("recall_weight must be finite and nonnegative")
        if not isinstance(self.confidence_screen, bool):
            raise ValueError("confidence_screen must be boolean")
        for name in ("max_continuation_events", "max_teacher_forwards"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def _module_fingerprint(module):
    """Hash a small optional recall module, never the billion-parameter AE."""
    if module is None:
        return None
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Nonfinite recall parameter: {name}")
        digest.update(_json([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _versions(module):
    # The large head is identified by caller-provided checkpoint/adaptor hashes.
    # Cheap version guards catch ordinary in-place changes while labels are used.
    if module is None:
        return None
    return tuple((name, id(tensor), tensor._version)
                 for name, tensor in list(module.named_parameters()) + list(module.named_buffers()))


def _check_recall(module, callback, cfg):
    if module is not None and (module.training or any(p.requires_grad for p in module.parameters())):
        raise ValueError("recall module must be frozen and in eval mode")
    if cfg.recall_weight and not callable(callback):
        raise ValueError("Positive recall_weight requires recall_cost callback")
    if callback is not None and not callable(callback):
        raise TypeError("recall_cost must be callable")


def _seed(seed, eid, candidate, decision, split, repeat):
    # Explicit domain separation makes audit B independent of training A.
    value = _json([DEFINITION, int(seed), int(eid), int(candidate), int(decision), split, repeat])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**63 - 1)


def _make_context(teacher, ep, candidate, bank_ids, cfg, seed, identity):
    if not isinstance(cfg, LabelV5Config):
        raise TypeError("config must be LabelV5Config")
    futures = eligible_futures(ep, candidate, cfg.memory_window)  # validates candidate
    ids = _bank_context(ep, candidate, bank_ids, teacher.config.capacity)
    forced = len(ids) < teacher.config.min_fill
    if forced or not futures:
        result = _context(teacher, ep, cfg, seed, "storage", identity)
        result.update(candidate=int(candidate), bank_ids=ids, options=[],
                      future_decisions=[], noise_seeds=[])
        result["skip_reason"] = "forced_min_fill" if forced else "no_old_future_query"
    else:
        result = _storage_context(teacher, ep, candidate, ids, cfg, seed, identity)
    result["definition"] = DEFINITION
    result["replay_semantics"] = "teacher_forced_demonstration_continuation"
    result["forced_min_fill"] = forced
    result["bank_coverage"] = ("empty" if not ids else "full" if len(ids) == teacher.config.capacity else "partial")
    result["noise_seeds"] = [[_seed(seed, ep["episode_id"], candidate, d, "A", r)
                              for r in range(cfg.noise_samples)] for d in result["future_decisions"]]
    result["audit_noise_seeds"] = [[_seed(seed, ep["episode_id"], candidate, d, "B", r)
                                    for r in range(cfg.audit_noise_samples)] for d in result["future_decisions"]]
    a = {s for row in result["noise_seeds"] for s in row}
    b = {s for row in result["audit_noise_seeds"] for s in row}
    if a & b:
        raise RuntimeError("Noise-domain collision; choose a different seed")
    if result["future_decisions"]:
        if max(result["future_decisions"]) - candidate - 1 > cfg.max_continuation_events:
            raise ValueError("Continuation exceeds max_continuation_events; adjust budget explicitly")
        requested = len(result["options"]) * len(result["future_decisions"]) * (cfg.noise_samples + cfg.audit_noise_samples)
        if requested > cfg.max_teacher_forwards:
            raise ValueError("Label exceeds max_teacher_forwards; adjust sampling or budget explicitly")
    return result


def _scalar_recall(callback, read, ep, decision):
    result = callback(read, ep, decision)
    value = result["loss"] if isinstance(result, dict) else result
    value = float(value.detach().item() if isinstance(value, torch.Tensor) else value)
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite teacher recall cost")
    return value


def _means(draws, count):
    return [statistics.mean(row[o] for future in draws for row in future) for o in range(count)]


def _screened_gain(stats, cfg):
    # Shrink noisy/tiny differences toward zero instead of producing a forced
    # argmin label. This IS a modified objective, not raw unbiased expected loss.
    if not cfg.confidence_screen:
        return stats["mean"] if abs(stats["mean"]) > cfg.margin else 0.0
    if stats["confident_positive"]:
        return max(0.0, stats["lower"] - cfg.margin)
    if stats["confident_negative"]:
        return min(0.0, stats["upper"] + cfg.margin)
    return 0.0


@torch.no_grad()
def _build(teacher, head, ep, cfg, context, loss_fn, recall_cost):
    common = {"definition": DEFINITION, "kind": "storage", "cache_key": _digest(context),
              "context": context, "episode_id": context["episode_id"],
              "candidate": context["candidate"], "bank_ids": context["bank_ids"],
              "options": context["options"], "future_decisions": context["future_decisions"],
              "noise_seeds": context["noise_seeds"], "scale": cfg.scale,
              "status": "skipped" if "skip_reason" in context else "ok",
              "forced_min_fill": context["forced_min_fill"], "bank_coverage": context["bank_coverage"]}
    if common["status"] == "skipped":
        common.update(skip_reason=context["skip_reason"], costs=[], raw_costs=[],
                      option_mean_losses=[], option_gain_stats=[], best_option=None,
                      confident_options=0, loss_draws=[], unique_expert_forwards=0,
                      effective_future_contexts=0, distinct_branch_sequences=0,
                      future_bank_ids=[], candidate_retained_at_queries=[])
        return common
    options, decisions = common["options"], common["future_decisions"]
    encoded = encode_until(teacher, ep, max(decisions))
    branch_banks, suffix_stats, retention = [], [], []
    action_draws, total_draws, audit_draws, recall_rows = [], [], [], []
    unique_forwards = 0
    for decision, seeds, audit_seeds in zip(decisions, context["noise_seeds"], context["audit_noise_seeds"]):
        banks, stats = [], []
        for option in options:
            bank, suffix = continuation_bank(teacher, ep, option, common["candidate"] + 1,
                                             decision, encoded, policy=cfg.continuation_policy)
            # Revalidate output even when a custom/replaced replay helper is used.
            bank = _bank_context(ep, decision, bank, teacher.config.capacity)
            banks.append(bank)
            stats.append(suffix)
        outputs = {tuple(bank): read_bank(teacher, ep, decision, bank, encoded) for bank in
                   {tuple(bank) for bank in banks}}
        recall = {bank: (_scalar_recall(recall_cost, read, ep, decision) if cfg.recall_weight else 0.0)
                  for bank, read in outputs.items()}
        values = {}
        for seed in seeds + audit_seeds:
            for bank, read in outputs.items():
                values[(seed, bank)] = _scalar_loss(loss_fn, head, ep, decision, read["fused_short"], seed)
                unique_forwards += 1
        actions = [[values[(seed, tuple(bank))] for bank in banks] for seed in seeds]
        totals = [[value + cfg.recall_weight * recall[tuple(bank)]
                   for value, bank in zip(row, banks)] for row in actions]
        audits = [[values[(seed, tuple(bank))] + cfg.recall_weight * recall[tuple(bank)]
                   for bank in banks] for seed in audit_seeds]
        action_draws.append(actions)
        total_draws.append(totals)
        audit_draws.append(audits)
        recall_rows.append([recall[tuple(bank)] for bank in banks])
        branch_banks.append(banks)
        suffix_stats.append(stats)
        retention.append([common["candidate"] in bank for bank in banks])
    count = len(options)
    means = _means(total_draws, count)
    gains = [_gain_stats([[row[0] - row[o] for row in future] for future in total_draws], cfg)
             for o in range(count)]
    costs = [-_screened_gain(gain, cfg) / cfg.scale for gain in gains]
    best = min(range(count), key=lambda i: means[i])
    common.update(loss_draws=total_draws, action_loss_draws=action_draws,
                  recall_costs=recall_rows, option_mean_losses=means,
                  option_mean_action_losses=_means(action_draws, count),
                  option_gain_stats=gains, costs=costs,
                  raw_costs=[(value - means[0]) / cfg.scale for value in means],
                  best_option=best, confident_options=sum(bool(g["confident_positive"] or g["confident_negative"]) for g in gains[1:]),
                  future_bank_ids=branch_banks, continuation_stats=suffix_stats,
                  candidate_retained_at_queries=retention,
                  effective_future_contexts=sum(len({tuple(bank) for bank in banks}) > 1 for banks in branch_banks),
                  distinct_branch_sequences=len({tuple(tuple(banks[o]) for banks in branch_banks) for o in range(count)}),
                  collapsed_option_query_fraction=statistics.mean(
                      (count - len({tuple(bank) for bank in banks})) / max(1, count - 1) for banks in branch_banks),
                  unique_expert_forwards=unique_forwards,
                  nominal_expert_forwards=count * len(decisions) * (cfg.noise_samples + cfg.audit_noise_samples),
                  objective="confidence_screened_keep_relative_expected_cost",
                  uncertainty_note="heuristic paired-noise screen; futures share one episode, not independent trajectories")
    if cfg.audit_noise_samples:
        audit_means = _means(audit_draws, count)
        common["audit"] = {"noise_seeds": context["audit_noise_seeds"], "loss_draws": audit_draws,
                           "option_mean_losses": audit_means, "A_best_option": best,
                           "A_best_gain_on_B": audit_means[0] - audit_means[best],
                           "used_for_training": False,
                           "non_tie_sign_agreement": statistics.mean(
                               (means[0] - means[o] > 0) == (audit_means[0] - audit_means[o] > 0)
                               for o in range(1, count)
                               if abs(means[0] - means[o]) > cfg.margin and abs(audit_means[0] - audit_means[o]) > cfg.margin)
                           if any(abs(means[0] - means[o]) > cfg.margin and abs(audit_means[0] - audit_means[o]) > cfg.margin
                                  for o in range(1, count)) else None}
    _json(common)
    return common


def _validate_label(ep, label):
    if label.get("definition") != DEFINITION or label.get("kind") != "storage":
        raise ValueError("Wrong continuation label definition/kind")
    context = label["context"]
    if label["cache_key"] != _digest(context) or context.get("definition") != DEFINITION:
        raise ValueError("Label context/cache key mismatch")
    frames, valid, decisions = _episode_layout(ep)
    layout = _digest({"frames": frames.tolist(), "valid": valid.tolist(), "decisions": decisions.tolist()})
    if (context["layout_hash"] != layout or context["cache_fingerprint"] != ep.get("cache_fingerprint")
            or label["episode_id"] != int(ep["episode_id"])):
        raise ValueError("Label belongs to a different episode/cache/layout")
    for key in ("candidate", "bank_ids", "options", "future_decisions", "noise_seeds", "forced_min_fill", "bank_coverage"):
        if label[key] != context[key]:
            raise ValueError(f"Storage label {key} differs from context")
    cfg = LabelV5Config(**context["config"])
    if label["scale"] != cfg.scale:
        raise ValueError("Label scale differs from configuration")
    if label["status"] not in ("ok", "skipped"):
        raise ValueError("Invalid continuation label status")
    if label["status"] == "skipped":
        if label["options"] or label["costs"] or "skip_reason" not in context:
            raise ValueError("Skipped label must have no learned contrast")
    else:
        if "skip_reason" in context or label["forced_min_fill"]:
            raise ValueError("Forced/invalid context cannot train a writer contrast")
        options = label["options"]
        if len(options) < 2 or options[0] != label["bank_ids"]:
            raise ValueError("Storage options must begin with KEEP")
        for key in ("costs", "raw_costs", "option_mean_losses", "option_gain_stats"):
            if len(label[key]) != len(options):
                raise ValueError(f"Storage {key} does not match option count")
        if label["best_option"] not in range(len(options)):
            raise ValueError("Invalid best_option")
        futures = eligible_futures(ep, label["candidate"], cfg.memory_window)
        if not label["future_decisions"] or any(d not in futures for d in label["future_decisions"]):
            raise ValueError("Label query is not a strictly-old future decision")
        expected = [-_screened_gain(g, cfg) / cfg.scale for g in label["option_gain_stats"]]
        if label["costs"] != expected:
            raise ValueError("Label screened costs do not match gain statistics")
    _json(label)


def storage_loss(memory, episode, label):
    """V4-compatible writer-only loss; forced/tied labels have connected zero.

    Confidence screening changes raw expected cost by shrinking uncertainty to
    zero. Independent B draws are audit data and NEVER determine this loss.
    """
    _validate_label(episode, label)
    bank = _bank_context(episode, label["candidate"], label["bank_ids"], memory.config.capacity)
    forced = len(bank) < memory.config.min_fill
    if forced != label["forced_min_fill"]:
        raise ValueError("Student min_fill differs from teacher")
    if label["status"] == "skipped":
        zero = sum((p.sum() * 0.0 for p in memory.writer_parameters()), start=next(memory.parameters()).new_zeros(()))
        return {"loss": zero, "metrics": {"storage_loss": 0.0, "storage_skipped": 1.0,
                "storage_forced_min_fill": float(forced), "storage_signal_fraction": 0.0,
                "storage_options": 0, "storage_confident_options": 0,
                "storage_effective_future_contexts": 0}}
    prediction = storage_prediction(memory, episode, label["candidate"], bank)
    if [list(ids) for ids in prediction["options"]] != label["options"]:
        raise ValueError("Student and teacher storage options differ")
    logits = prediction["logits"].reshape(-1)
    costs = logits.new_tensor(label["costs"]).detach()
    if logits.shape != costs.shape or not bool(torch.isfinite(logits).all()):
        raise ValueError("Storage logits must be finite and match label options")
    probs = logits.softmax(-1)
    loss = (probs * costs).sum()
    selected = int(probs.detach().argmax())
    means, margin = label["option_mean_losses"], label["context"]["config"]["margin"]
    raw = logits.new_tensor(label["raw_costs"])
    candidate_kept = [row[o] for row in label["candidate_retained_at_queries"] for o in range(1, len(row))]
    metrics = {"storage_loss": float(loss.detach()), "storage_skipped": 0.0,
               "storage_forced_min_fill": 0.0,
               "storage_expected_gain": float(-(probs.detach() * raw).sum() * label["scale"]),
               "storage_screened_expected_gain": float(-loss.detach() * label["scale"]),
               "storage_write_probability": float(1 - probs[0].detach()),
               "storage_choice_agreement": float(selected == label["best_option"]),
               "storage_tie_aware_agreement": float(means[selected] <= min(means) + margin),
               "storage_selected_regret": means[selected] - min(means),
               "storage_tie_aware_regret": max(0.0, means[selected] - min(means) - margin),
               "storage_options": len(means), "storage_confident_options": label["confident_options"],
               "storage_best_gain": means[0] - min(means),
               "storage_signal_fraction": float(any(cost != 0 for cost in label["costs"])),
               "storage_raw_cost_spread": max(means) - min(means),
               "storage_full_bank_fraction": float(label["bank_coverage"] == "full"),
               "storage_effective_future_contexts": label["effective_future_contexts"],
               "storage_collapsed_option_query_fraction": label["collapsed_option_query_fraction"],
               "storage_candidate_retention": statistics.mean(candidate_kept) if candidate_kept else 0.0,
               "storage_unique_expert_forwards": label["unique_expert_forwards"]}
    if "audit" in label:
        b = label["audit"]["option_mean_losses"]
        metrics.update(storage_A_best_gain_on_B=label["audit"]["A_best_gain_on_B"],
                       storage_selected_gain_on_B=b[0] - b[selected],
                       storage_sign_agreement_A_B=label["audit"]["non_tie_sign_agreement"])
    return {"loss": loss, "metrics": metrics}


class ContinuationLabelsV5:
    """Exact-context cache bound to immutable reader, writer, AE and recall.

    Refresh by making a NEW instance in a NEW version directory. Copying the
    small teacher by default does not freeze the live student. The action head
    and optional recall module are shared read-only. The callback must close
    over that same frozen recall module, not a live student module.
    """

    def __init__(self, teacher, action_head, config, output_dir, identity, *,
                 copy_teacher=True, recall_cost=None, recall_module=None):
        if not isinstance(config, LabelV5Config):
            raise TypeError("config must be LabelV5Config")
        if not isinstance(identity, dict) or any(identity.get(k) in (None, "") for k in ("cache_fingerprint", "teacher_version")):
            raise ValueError("identity requires cache_fingerprint and teacher_version")
        if action_head is not None and not any(identity.get(k) for k in (
                "action_expert_fingerprint", "expert_adapters_sha256", "frozen_expert_sha256")):
            raise ValueError("identity requires action expert checkpoint/adaptor fingerprint")
        if recall_cost is not None and not identity.get("recall_fingerprint"):
            raise ValueError("recall_cost requires recall_fingerprint including target/weight semantics")
        self.teacher = copy.deepcopy(teacher) if copy_teacher else teacher
        self.teacher.eval().requires_grad_(False)
        self.action_head, self.config = action_head, config
        self.recall_cost, self.recall_module = recall_cost, recall_module
        _check_frozen(self.teacher, action_head)
        _check_recall(recall_module, recall_cost, config)
        self.identity = copy.deepcopy(identity)
        self.identity["recall_module_sha256"] = _module_fingerprint(recall_module)
        self.path = Path(output_dir)
        self.manifest = {"definition": DEFINITION, "config": asdict(config), "identity": self.identity,
                         "teacher_fingerprint": _teacher_fingerprint(self.teacher)}
        self.head_versions = _versions(action_head)
        self.path.mkdir(parents=True, exist_ok=True)
        path = self.path / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != self.manifest:
                raise ValueError("Label cache belongs to a different teacher/writer/recall/data/configuration")
        else:
            _atomic_json(path, self.manifest)

    def get_storage(self, episode, candidate, bank_ids, seed, loss_fn=None):
        _check_frozen(self.teacher, self.action_head)
        _check_recall(self.recall_module, self.recall_cost, self.config)
        if _versions(self.action_head) != self.head_versions:
            raise ValueError("Frozen action expert changed; refresh teacher/version/cache")
        if _module_fingerprint(self.recall_module) != self.identity["recall_module_sha256"]:
            raise ValueError("Frozen recall module changed; refresh teacher/version/cache")
        context = _make_context(self.teacher, episode, candidate, bank_ids, self.config, seed, self.identity)
        if context["teacher_fingerprint"] != self.manifest["teacher_fingerprint"]:
            raise ValueError("Frozen teacher/writer changed; refresh teacher/version/cache")
        key = _digest(context)
        path = self.path / f"storage-episode-{context['episode_id']:06d}-{candidate:06d}-{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("cache_key") != key or record.get("context") != context:
                raise ValueError("Label cache context/key mismatch")
        else:
            record = _build(self.teacher, self.action_head, episode, self.config, context,
                            loss_fn or episode_flow_loss, self.recall_cost)
            _atomic_json(path, record)
        _validate_label(episode, record)
        return record
