"""Causal, fixed-budget storage operations for the metadata-memory experiment.

The controller chooses KEEP, REPLACE(j), or an explicitly enabled adjacent
MERGE+APPEND once the bank is full. Before capacity every observation event is
appended, regardless of controller scores. A slot is Q encoded tokens, *not* a
single RGB frame, object, or completed robot action.

Discrete decisions are detached; applying the selected operation is NOT. This
distinction lets action/answer gradients reach retained historical encodings and
the learned merger. It does not claim that argmax deletion receives an action
gradient: the operation scorer needs detached counterfactual utility targets.

Only causal encoded bank/candidate tensors enter this module. Labels, GT
actions, future observations, and retrieved/fused outputs must never be passed
as storage content. The callback deliberately ignores the core's provenance
arguments and holds no session state. Episode resets belong to the caller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


STORAGE_VERSION = "semantic_memory_storage_v1"
KEEP_OPERATION_ID = "keep"
FIFO_OPERATION_ID = "replace:0"
_KINDS = ("append", "keep", "replace", "merge")


@dataclass(frozen=True)
class StorageConfig:
    capacity_events: int = 32
    num_tokens: int = 4
    dim: int = 256
    hidden_dim: int = 128
    num_heads: int = 4
    enable_merge: bool = False
    # Enabling an untrained merger at deployment is not a valid ablation. The
    # surrounding checkpoint manifest must preserve the verification evidence.
    merge_verified: bool = False
    decision_margin: float = 0.0
    version: str = STORAGE_VERSION

    def __post_init__(self):
        if self.version != STORAGE_VERSION:
            raise ValueError("Unsupported semantic storage version")
        for name in ("capacity_events", "num_tokens", "dim", "hidden_dim", "num_heads"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.num_heads:
            raise ValueError("dim must be divisible by num_heads")
        for name in ("enable_merge", "merge_verified"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.enable_merge and not self.merge_verified:
            raise ValueError("enable_merge requires explicit merge_verified=True")
        if self.enable_merge and self.capacity_events < 2:
            raise ValueError("Adjacent merge requires at least two event slots")
        if (isinstance(self.decision_margin, bool)
                or not isinstance(self.decision_margin, (int, float))
                or not math.isfinite(self.decision_margin) or self.decision_margin < 0):
            raise ValueError("decision_margin must be finite and nonnegative")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class StorageOperation:
    """Index identifies a victim, or the first of two adjacent merged events.

    REPLACE removes that victim and appends the observed candidate at the end;
    it does not put a new observation into an old temporal position. MERGE
    preserves pair order, keeps its compressed event at the first position, and
    appends the current candidate. A provenance sidecar should follow this same
    operation separately from the model tensors.
    """
    kind: str
    index: int = -1

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"Unsupported storage operation: {self.kind!r}")
        if type(self.index) is not int:
            raise ValueError("Operation index must be an integer")
        if self.kind in ("append", "keep") and self.index != -1:
            raise ValueError("APPEND/KEEP must not specify a victim index")
        if self.kind in ("replace", "merge") and self.index < 0:
            raise ValueError("REPLACE/MERGE require a nonnegative event index")

    @property
    def id(self) -> str:
        return self.kind if self.index < 0 else f"{self.kind}:{self.index}"


class OrderedEventMerger(nn.Module):
    """Q learned queries cross-attend to an ordered pair of Q-token events.

    This is a compression model, not lossless memory. Token and earlier/later
    positions are explicit and dropout is zero. Visual similarity alone never
    triggers it. Merger usefulness must be checked by future-answer/action
    preservation before its configuration is authorized for deployment.
    """
    def __init__(self, config: StorageConfig):
        super().__init__()
        q, d = config.num_tokens, config.dim
        self.queries = nn.Parameter(torch.empty(1, q, d))
        self.token_positions = nn.Parameter(torch.empty(1, q, d))
        self.event_positions = nn.Parameter(torch.empty(2, 1, d))
        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.token_positions, std=0.02)
        nn.init.normal_(self.event_positions, std=0.02)
        self.input_norm = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, config.num_heads, dropout=0.0, batch_first=True)
        self.output_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, earlier: torch.Tensor, later: torch.Tensor) -> torch.Tensor:
        # FP32 learned modules, independently of the surrounding VLM autocast.
        with torch.autocast(device_type=earlier.device.type, enabled=False):
            first = earlier.float() + self.token_positions + self.event_positions[0]
            second = later.float() + self.token_positions + self.event_positions[1]
            context = self.input_norm(torch.cat((first, second), dim=1))
            query = self.queries.expand(earlier.shape[0], -1, -1)
            read, _ = self.attention(query, context, context, need_weights=False)
            merged = query + read
            merged = merged + self.ffn(self.output_norm(merged))
        return merged.to(earlier.dtype)


class StorageManager(nn.Module):
    """Shared operation MLP with deterministic, exact FIFO initialization.

    Scores are context-dependent signed utilities. Each row contains candidate,
    average bank, victim(s), candidate-minus-bank, operation kind, and causal
    order/redundancy statistics. This pooling is the controller's input only;
    stored and retrieved values remain their complete Q-token events.
    """
    def __init__(self, config: StorageConfig):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.LayerNorm(5 * config.dim + 9),
            nn.Linear(5 * config.dim + 9, config.hidden_dim),
            nn.SiLU(), nn.Linear(config.hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        # No dormant merger parameters or random draws when merging is disabled.
        self.merger = OrderedEventMerger(config) if config.enable_merge else None

    def _inputs(self, bank, candidate):
        c = self.config
        if (not isinstance(candidate, torch.Tensor)
                or candidate.shape != (1, c.num_tokens, c.dim)
                or not torch.is_floating_point(candidate)):
            raise ValueError(f"candidate must be one floating encoded event [1,{c.num_tokens},{c.dim}]")
        if bank is None:
            bank = candidate[:, :0]
        if (not isinstance(bank, torch.Tensor) or bank.ndim != 3
                or bank.shape[0] != 1 or bank.shape[2] != c.dim
                or not torch.is_floating_point(bank)):
            raise ValueError("bank must be floating [1,N*Q,D] with the configured dimension")
        if bank.shape[1] % c.num_tokens or bank.shape[1] > c.capacity_events * c.num_tokens:
            raise ValueError("bank must contain complete events within configured capacity")
        if bank.device != candidate.device or bank.dtype != candidate.dtype:
            raise ValueError("bank and candidate dtype/device must match")
        if not bool(torch.isfinite(bank).all()) or not bool(torch.isfinite(candidate).all()):
            raise FloatingPointError("Nonfinite storage bank/candidate")
        return bank, candidate

    def _legal_operations(self, n_events: int) -> tuple[StorageOperation, ...]:
        if n_events < self.config.capacity_events:
            return (StorageOperation("append"),)
        result = [StorageOperation("keep")]
        result.extend(StorageOperation("replace", i) for i in range(n_events))
        if self.config.enable_merge:
            result.extend(StorageOperation("merge", i) for i in range(n_events - 1))
        return tuple(result)

    def operations(self, bank, candidate) -> tuple[StorageOperation, ...]:
        bank, _ = self._inputs(bank, candidate)
        return self._legal_operations(bank.shape[1] // self.config.num_tokens)

    def fifo_operation(self, bank, candidate) -> StorageOperation:
        bank, _ = self._inputs(bank, candidate)
        full = bank.shape[1] == self.config.capacity_events * self.config.num_tokens
        return StorageOperation("replace", 0) if full else StorageOperation("append")

    def operation_features(self, bank, candidate) -> torch.Tensor:
        """Return causal rows in exactly ``operations`` order; no labels accepted."""
        bank, candidate = self._inputs(bank, candidate)
        c = self.config
        current = candidate.float().mean(1)[0]
        events = bank.float().reshape(-1, c.num_tokens, c.dim).mean(1)
        n = events.shape[0]
        average = events.mean(0) if n else torch.zeros_like(current)
        ops = self._legal_operations(n)
        # Vectorize over candidate operations. Prefix replay can invoke this
        # many times: a Python/GPU kernel loop for every victim is avoidable.
        padded = torch.cat((events, current.new_zeros(1, c.dim)), dim=0)
        first_ids = torch.tensor([op.index if op.index >= 0 else n for op in ops],
                                 device=current.device, dtype=torch.long)
        second_ids = torch.tensor([op.index + 1 if op.kind == "merge" else n for op in ops],
                                  device=current.device, dtype=torch.long)
        victim, other = padded[first_ids], padded[second_ids]
        kind_ids = torch.tensor([_KINDS.index(op.kind) for op in ops],
                                device=current.device, dtype=torch.long)
        kind = F.one_hot(kind_ids, num_classes=4).to(current.dtype)
        scalars = current.new_tensor([
            ((op.index + 1) / max(n, 1) if op.index >= 0 else 0.0,
             (op.index + 2) / n if op.kind == "merge" else 0.0,
             n / c.capacity_events) for op in ops])
        current_rows = current.expand(len(ops), -1)
        average_rows = average.expand(len(ops), -1)
        pair_cos = F.cosine_similarity(victim, other, dim=-1).unsqueeze(1)
        candidate_cos = F.cosine_similarity(current_rows, victim, dim=-1).unsqueeze(1)
        return torch.cat((current_rows, average_rows, victim, other, current_rows - average_rows,
                          kind, scalars, pair_cos, candidate_cos), dim=-1)

    def scores(self, bank, candidate) -> torch.Tensor:
        """Differentiable critic scores; detach inputs in critic-only training.

        A hard decision uses ``choose`` instead, which prevents action loss from
        accidentally updating a controller through any continuous score path.
        """
        features = self.operation_features(bank, candidate)
        if features.device != self.network[0].weight.device:
            raise ValueError("Storage manager and encoded inputs must share a device")
        if self.network[0].weight.dtype != torch.float32:
            raise ValueError("Storage manager learned parameters must remain FP32")
        with torch.autocast(device_type=features.device.type, enabled=False):
            result = self.network(features.float()).squeeze(-1)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Nonfinite storage operation scores")
        return result

    def _choice_index(self, operations, scores) -> int:
        # With all-zero logits, INSERT/REPLACE-oldest is the exact baseline.
        fifo = next(i for i, op in enumerate(operations)
                    if op.kind == "append" or op.id == FIFO_OPERATION_ID)
        winner = int(scores.argmax().item())
        if float(scores[winner] - scores[fifo]) <= self.config.decision_margin:
            return fifo
        return winner

    def choose(self, bank, candidate) -> StorageOperation:
        operations = self.operations(bank, candidate)
        if len(operations) == 1:
            return operations[0]
        with torch.no_grad():
            scores = self.scores(bank, candidate)
            return operations[self._choice_index(operations, scores)]

    def shortlist(self, bank, candidate, *, max_operations: int) -> tuple[StorageOperation, ...]:
        """Bound teacher branching while always retaining KEEP and FIFO controls.

        Other candidates use descending detached score and stable operation
        order to break ties. No sampling means this cannot consume action RNG.
        ``operations`` remains available for exhaustive teacher comparisons.
        """
        if type(max_operations) is not int or max_operations < 2:
            raise ValueError("max_operations must be an integer >= 2")
        ops = self.operations(bank, candidate)
        if len(ops) <= max_operations:
            return ops
        with torch.no_grad():
            values = self.scores(bank, candidate).cpu().tolist()
        selected = [i for i, op in enumerate(ops) if op.id in (KEEP_OPERATION_ID, FIFO_OPERATION_ID)]
        others = sorted((i for i in range(len(ops)) if i not in selected), key=lambda i: (-values[i], i))
        selected.extend(others[:max_operations-len(selected)])
        # Return canonical order so score/target alignment does not depend on
        # ranking, and require callers to select scores by operation ID.
        return tuple(ops[i] for i in sorted(selected))

    def apply(self, bank, candidate, operation: StorageOperation) -> torch.Tensor:
        """Apply a legal discrete operation WITHOUT detaching encoded contents."""
        bank, candidate = self._inputs(bank, candidate)
        if not isinstance(operation, StorageOperation):
            raise TypeError("operation must be a StorageOperation descriptor")
        if operation not in self._legal_operations(bank.shape[1] // self.config.num_tokens):
            raise ValueError(f"Illegal operation {operation.id} for current bank/configuration")
        q = self.config.num_tokens
        if operation.kind == "append":
            updated = torch.cat((bank, candidate), dim=1)
        elif operation.kind == "keep":
            updated = bank
        elif operation.kind == "replace":
            start = operation.index * q
            updated = torch.cat((bank[:, :start], bank[:, start + q:], candidate), dim=1)
        else:
            start = operation.index * q
            if self.merger is None:
                raise RuntimeError("Merge operation has no authorized merger")
            merged = self.merger(bank[:, start:start + q], bank[:, start + q:start + 2*q])
            updated = torch.cat((bank[:, :start], merged, bank[:, start + 2*q:], candidate), dim=1)
        if updated.shape[1] != min(bank.shape[1] + q, self.config.capacity_events * q):
            raise RuntimeError("Storage operation violated exact event capacity")
        if not bool(torch.isfinite(updated).all()):
            raise FloatingPointError("Nonfinite storage operation result")
        return updated

    def make_policy(self) -> Callable:
        """Core ``_write`` callback; no whole-callback ``no_grad`` decorator."""
        def policy(bank, encoded_current, *, event_index=None, frame=None, is_demo=None):
            # IDs/times/demo are provenance, not ground-truth controller input.
            # Storage sees the observed encoded event, NEVER the READ output.
            bank, candidate = self._inputs(bank, encoded_current)
            operations = self._legal_operations(bank.shape[1] // self.config.num_tokens)
            with torch.no_grad():
                if len(operations) == 1:
                    selected, score = 0, 0.0
                else:
                    scores = self.scores(bank, candidate)
                    selected = self._choice_index(operations, scores)
                    score = float(scores[selected])
            op = operations[selected]
            # Intentionally outside no_grad: the past/candidate/merger graph
            # remains differentiable when called by episode-prefix replay.
            updated = self.apply(bank, candidate, op)
            full = bank.shape[1] == self.config.capacity_events * self.config.num_tokens
            return updated, {
                "write_rate": float(op.kind != "keep"),
                "writer_insert": float(op.kind != "keep"),
                "writer_keep": float(op.kind == "keep"),
                "writer_replace": float(op.kind == "replace"),
                "writer_merge": float(op.kind == "merge"),
                "writer_full": float(full),
                "writer_score": score,
                "writer_operation_code": float(_KINDS.index(op.kind)),
                "writer_selected_index": float(op.index),
            }
        return policy
