"""Causal admission-only controller on top of a frozen V19 memory/actor.

This module never changes event content, temporal encodings, the reader, or the
FIFO victim rule. It appends until capacity and then chooses KEEP versus
REPLACE-OLDEST. An event is Q encoded tokens, not one frame or one object.

Only the two-headed MLP is learned. Its input is detached causal bank/candidate
content; future actions and teacher labels are deliberately absent from this
interface. Utility is in the training split's externally normalized units.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


ADMISSION_VERSION = "cvom_admission_fixed_fifo_victim_v1"


@dataclass(frozen=True)
class AdmissionConfig:
    dim: int = 256
    num_tokens: int = 4
    capacity_events: int = 32
    hidden_dim: int = 128
    write_threshold: float = 0.5
    utility_margin: float = 0.0
    version: str = ADMISSION_VERSION

    def __post_init__(self):
        if self.version != ADMISSION_VERSION:
            raise ValueError("Unsupported CVoM admission version")
        for name in ("dim", "num_tokens", "capacity_events", "hidden_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("write_threshold", "utility_margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.write_threshold <= 1:
            raise ValueError("write_threshold must be in [0,1]")
        if self.utility_margin < 0:
            raise ValueError("utility_margin must be nonnegative")

    def to_dict(self):
        return asdict(self)

    @property
    def feature_dim(self):
        return 4 * self.dim + 4


def _inputs(bank, candidate, config):
    if (not isinstance(candidate, torch.Tensor) or candidate.shape != (1, config.num_tokens, config.dim)
            or not candidate.is_floating_point()):
        raise ValueError("candidate must be one floating encoded event [1,Q,D]")
    if bank is None:
        bank = candidate[:, :0]
    if (not isinstance(bank, torch.Tensor) or bank.ndim != 3 or bank.shape[0] != 1
            or bank.shape[2] != config.dim or not bank.is_floating_point()):
        raise ValueError("bank must be floating [1,N*Q,D]")
    if bank.shape[1] % config.num_tokens or bank.shape[1] > config.capacity_events * config.num_tokens:
        raise ValueError("bank must contain complete events within the configured capacity")
    if bank.dtype != candidate.dtype or bank.device != candidate.device:
        raise ValueError("bank/candidate dtype and device must match")
    if not bool(torch.isfinite(bank).all()) or not bool(torch.isfinite(candidate).all()):
        raise FloatingPointError("Nonfinite admission bank/candidate")
    return bank, candidate


def admission_features(bank, candidate, config: AdmissionConfig):
    """Detached [1,4D+4] current/mean/oldest/difference + causal statistics.

    Temporal and demo information already embedded in the encoded tokens is
    retained; no annotation, absolute future duration, outcome, or target can
    enter through this API. Pooling here is only for the admission MLP: the
    actual stored events keep all Q tokens unchanged.
    """
    bank, candidate = _inputs(bank, candidate, config)
    with torch.no_grad(), torch.autocast(device_type=candidate.device.type, enabled=False):
        current = candidate.detach().float().mean(1)
        events = bank.detach().float().reshape(1, -1, config.num_tokens, config.dim).mean(2)
        n_events = events.shape[1]
        mean = events.mean(1) if n_events else torch.zeros_like(current)
        oldest = events[:, 0] if n_events else torch.zeros_like(current)
        if n_events:
            similarity = F.cosine_similarity(events, current[:, None], dim=-1)
            scalars = torch.stack((similarity.max(1).values, similarity.mean(1), similarity[:, 0],
                                   current.new_full((1,), n_events / config.capacity_events)), dim=1)
        else:
            scalars = current.new_zeros(1, 4)
        result = torch.cat((current, mean, oldest, current - mean, scalars), dim=1)
    return result.detach()


class CVoMAdmission(nn.Module):
    """Shared two-layer MLP with signed utility and binary admission heads.

    Both final heads start at zero. The >= tie rules then reproduce FIFO
    exactly with the default configuration, including after the bank fills.
    Decision-making is deterministic and consumes no sampling RNG.
    """
    def __init__(self, config: AdmissionConfig):
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU())
        self.utility_head = nn.Linear(config.hidden_dim, 1)
        self.write_head = nn.Linear(config.hidden_dim, 1)
        for head in (self.utility_head, self.write_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def features(self, bank, candidate):
        return admission_features(bank, candidate, self.config)

    def forward_features(self, features):
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or features.shape[1] != self.config.feature_dim or not features.is_floating_point()):
            raise ValueError("admission features must be floating [B,4D+4]")
        if not bool(torch.isfinite(features).all()):
            raise FloatingPointError("Nonfinite admission features")
        if features.device != self.utility_head.weight.device:
            raise ValueError("Admission controller and features must share a device")
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise TypeError("Admission controller parameters must remain FP32")
        with torch.autocast(device_type=features.device.type, enabled=False):
            hidden = self.trunk(features.detach().float())
            utility = self.utility_head(hidden).squeeze(-1)
            logit = self.write_head(hidden).squeeze(-1)
            if not bool(torch.isfinite(utility).all() & torch.isfinite(logit).all()):
                raise FloatingPointError("Nonfinite admission prediction")
            return {"utility": utility, "logit": logit, "write_probability": logit.sigmoid()}

    def forward(self, bank, candidate):
        return self.forward_features(self.features(bank, candidate))

    def _choice(self, n_events, prediction):
        if n_events < self.config.capacity_events:
            return "append"
        admit = ((prediction["write_probability"][0] >= self.config.write_threshold)
                 & (prediction["utility"][0] >= self.config.utility_margin))
        return "replace:0" if bool(admit) else "keep"

    @torch.no_grad()
    def choose(self, bank, candidate):
        bank, candidate = _inputs(bank, candidate, self.config)
        return self._choice(bank.shape[1] // self.config.num_tokens, self(bank, candidate))

    def apply(self, bank, candidate, choice):
        """Never alter event tokens or let a new event inherit an old position."""
        bank, candidate = _inputs(bank, candidate, self.config)
        full = bank.shape[1] == self.config.capacity_events * self.config.num_tokens
        if choice == "append" and not full:
            return torch.cat((bank, candidate), dim=1)
        if choice == "keep" and full:
            return bank
        if choice == "replace:0" and full:
            return torch.cat((bank[:, self.config.num_tokens:], candidate), dim=1)
        raise ValueError("Illegal admission operation for current capacity; only fixed FIFO victim is supported")

    def make_policy(self):
        """Stateless V18 WRITE callback; caller retains normal READ-before-WRITE."""
        @torch.no_grad()
        def policy(bank, candidate, *, event_index=None, frame=None, is_demo=None):
            bank, candidate = _inputs(bank, candidate, self.config)
            prediction = self(bank, candidate)
            choice = self._choice(bank.shape[1] // self.config.num_tokens, prediction)
            output = self.apply(bank, candidate, choice)
            number = lambda value: candidate.new_tensor(float(value))
            return output, {
                "write_rate": number(choice != "keep"),
                "writer_insert": number(choice != "keep"),
                "writer_append": number(choice == "append"),
                "writer_replace": number(choice == "replace:0"),
                "writer_keep": number(choice == "keep"),
                "writer_utility": prediction["utility"][0],
                "writer_probability": prediction["write_probability"][0],
                "writer_capacity": number(self.config.capacity_events),
            }
        return policy

    write_policy = make_policy
