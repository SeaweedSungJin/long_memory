"""Read existing immutable caches without scanning every large VL tensor.

Indexing uses memory-mapped PyTorch storage. It validates chronology, masks and
small event tensors, but never scans the large per-endpoint VL features or all
GT action targets. Selected action/VL rows are validated when actually used,
including frozen HAMLET conditioning-tail equality. No cache format changes.
"""
from functools import lru_cache
from pathlib import Path

import torch


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_mapped_episode(ep, dimensions=None):
    """Original cache causality contract without a full VL/target data scan."""
    required = {"episode_id", "frames", "features", "attention_masks", "image_masks",
                "moment", "short", "state", "targets", "target_mask", "actions",
                "action_mask", "transition_valid", "decision_mask", "embodiment_id"}
    _require(not required - ep.keys(), f"Missing cache fields: {sorted(required - ep.keys())}")
    frames = ep["frames"]
    _require(isinstance(frames, torch.Tensor) and frames.ndim == 1 and len(frames) >= 2,
             "Need at least two integer endpoints")
    _require(frames.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
             "Endpoint frames must be integers")
    _require(bool((frames >= 0).all()) and bool((frames[1:] > frames[:-1]).all()),
             "Endpoint frames must be nonnegative and strictly increasing")
    n = len(frames) - 1
    actions, targets = ep["actions"], ep["targets"]
    _require(actions.ndim == 3 and actions.shape[0] == n and actions.shape[1] > 0,
             "actions must be [T,C,A]")
    _require(targets.ndim == 3 and targets.shape[0] == n and targets.shape[1] > 0,
             "targets must be [T,H,A]")
    _require(actions.shape[-1] == targets.shape[-1] and actions.shape[-1] > 0,
             "Action dimensions disagree")
    _require(ep["action_mask"].shape == actions.shape[:2] and ep["action_mask"].dtype == torch.bool,
             "action_mask shape/dtype mismatch")
    _require(ep["target_mask"].shape == targets.shape and ep["target_mask"].dtype == torch.bool,
             "target_mask shape/dtype mismatch")
    for name in ("transition_valid", "decision_mask"):
        _require(ep[name].shape == (n,) and ep[name].dtype == torch.bool,
                 f"{name} shape/dtype mismatch")
    decisions = ep["decision_mask"]
    _require(not bool(ep["action_mask"][~decisions].any()), "Passive demo has recorded action prefix")
    _require(not bool(ep["target_mask"][~decisions].any()), "Passive demo has action-loss target")
    gap = frames[1:] - frames[:-1]
    _require(bool((gap <= actions.shape[1]).all()), "Endpoint cadence exceeds action-prefix capacity")
    beyond = torch.arange(actions.shape[1])[None] >= gap[:, None]
    _require(not bool(ep["action_mask"][beyond].any()), "Recorded action extends beyond next observation")
    _require(not bool(actions[~ep["action_mask"]].any()), "Padded/passive executed actions must be zero")
    if "is_demo" in ep:
        demo = ep["is_demo"]
        _require(demo.shape == frames.shape and demo.dtype == torch.bool, "Endpoint demo mask mismatch")
        _require(not bool((decisions & demo[:-1]).any()), "Demo endpoint used as a decision")
    moment, short, state = ep["moment"], ep["short"], ep["state"]
    _require(moment.ndim == 3 and moment.shape[0] == n + 1 and moment.shape[1] > 0,
             "moment must be [T+1,Q,D]")
    _require(short.shape == moment.shape, "short/moment shape mismatch")
    _require(state.ndim == 2 and state.shape[0] == n + 1, "state must be [T+1,S]")
    q, d = short.shape[-2:]
    for name in ("features", "attention_masks", "image_masks"):
        _require(len(ep[name]) == n + 1, f"{name} endpoint count mismatch")
    for index, features in enumerate(ep["features"]):
        # Tensor metadata only: do not page in all VL feature values here.
        _require(features.ndim == 2 and features.shape[1] == d and features.shape[0] >= q,
                 "Feature shape omits/mismatches short-memory tokens")
        for name in ("attention_masks", "image_masks"):
            mask = ep[name][index]
            _require(mask.shape == features.shape[:1] and mask.dtype == torch.bool,
                     f"{name} shape/dtype mismatch")
    if dimensions is not None:
        for name, actual in (("feature_dim", d), ("state_dim", state.shape[-1]),
                             ("action_dim", actions.shape[-1])):
            if name in dimensions:
                _require(actual == dimensions[name], f"Cache manifest {name} mismatch")
    # These are the small event inputs. Validate before batched re-encoding so
    # even an unselected corrupt event cannot inject NaNs through backward.
    for name in ("moment", "short", "state", "actions"):
        if not bool(torch.isfinite(ep[name]).all()):
            raise FloatingPointError(f"Nonfinite cached event {name}")


class MappedEpisodes:
    def __init__(self, cache, max_cached=2):
        self.cache = cache
        self.records = {int(r["episode_id"]): r for r in cache.manifest.get("episodes", [])}
        self.fetch = lru_cache(maxsize=max_cached)(self._load)

    def _load(self, eid):
        # Small synthetic tests can supply the same EpisodeCache interface.
        if not self.records:
            return self.cache.load(int(eid))
        root = Path(self.cache.path).resolve()
        path = (root / self.records[int(eid)]["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Episode path escapes cache directory")
        ep = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if int(ep["episode_id"]) != int(eid) or ep.get("cache_fingerprint") != self.cache.manifest["fingerprint"]:
            raise ValueError("Episode ID/cache fingerprint mismatch")
        validate_mapped_episode(ep, self.cache.manifest)
        return ep


def validate_decision(ep, decision):
    """Fail closed on a corrupted selected action target or conditioning row."""
    if not 0 <= decision < len(ep["actions"]) or not bool(ep["decision_mask"][decision]):
        raise ValueError("An action loss requires a valid cached decision")
    for name in ("features", "state", "targets"):
        if name in ep and not bool(torch.isfinite(ep[name][decision]).all()):
            raise FloatingPointError(f"Nonfinite selected cache {name}")
    if "target_mask" in ep and not bool(ep["target_mask"][decision].any()):
        raise ValueError("Selected action has no valid target dimensions")
    if "features" not in ep:
        return  # Synthetic CPU experts intentionally have no frozen VL cache.
    feature, short = ep["features"][decision], ep["short"][decision]
    _require(short.ndim == 2 and short.shape[0] > 0, "Selected short must be [Q,D]")
    q, d = short.shape
    _require(feature.ndim == 2 and feature.shape[1] == d and feature.shape[0] >= q,
             "Selected feature shape omits/mismatches short-memory tokens")
    _require(torch.equal(feature[-q:], short), "Selected conditioning tail is not short-memory output")
    for name in ("attention_masks", "image_masks"):
        _require(name in ep, f"Missing selected conditioning {name}")
        mask = ep[name][decision]
        _require(mask.shape == feature.shape[:1] and mask.dtype == torch.bool,
                 f"Selected {name} shape/dtype mismatch")
    _require("target_mask" in ep and ep["target_mask"][decision].shape == ep["targets"][decision].shape
             and ep["target_mask"].dtype == torch.bool, "Selected target_mask shape/dtype mismatch")
