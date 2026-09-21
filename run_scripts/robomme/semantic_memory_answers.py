"""Small labels-only auxiliary decoder of actual retrieved memory values.

No label, current state, query, instruction, or time is an argument to forward.
Success here is not robot-task success and does not establish unique retrieval.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SemanticAnswerConfig:
    input_dim: int = 256
    hidden_dim: int = 128
    classification_sizes: dict[str, int] = field(default_factory=dict)
    regression_sizes: dict[str, int] = field(default_factory=dict)


class SemanticAnswerHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config if isinstance(config, SemanticAnswerConfig) else SemanticAnswerConfig(**config)
        c = self.config
        if c.input_dim < 1 or c.hidden_dim < 1 or any(v < 2 for v in c.classification_sizes.values()) or any(v < 1 for v in c.regression_sizes.values()):
            raise ValueError("Invalid semantic head dimensions")
        self.trunk = nn.Sequential(nn.LayerNorm(c.input_dim), nn.Linear(c.input_dim, c.hidden_dim), nn.SiLU())
        self.classifiers = nn.ModuleDict({name: nn.Linear(c.hidden_dim, size) for name, size in c.classification_sizes.items()})
        self.regressors = nn.ModuleDict({name: nn.Linear(c.hidden_dim, size) for name, size in c.regression_sizes.items()})

    def config_dict(self):
        return asdict(self.config)

    def forward(self, retrieved):
        if retrieved.ndim != 3 or retrieved.shape[-1] != self.config.input_dim or retrieved.shape[1] < 1:
            raise ValueError("Expected actual attention output [batch, query_tokens, hidden_dim]")
        hidden = self.trunk(retrieved.float().mean(dim=1))
        return {"classification": {k: layer(hidden) for k, layer in self.classifiers.items()},
                "regression": {k: layer(hidden).sigmoid() for k, layer in self.regressors.items()}}

    def loss(self, retrieved, target):
        """Mean across active heads, then samples; missing targets contribute zero.

        Metrics carry counts for correct weighted aggregation. The normalized
        anchor MAE is not tracking error; pixel MAE uses the fixed 255 scale.
        """
        rows = ([None] * retrieved.shape[0] if target is None else
                [target] if isinstance(target, dict) else list(target))
        if len(rows) != retrieved.shape[0]:
            raise ValueError("Target batch differs from retrieved batch")
        output = self(retrieved)
        terms, metrics = [], {"answer_samples": float(sum(bool(r and (r.get("classification") or r.get("regression"))) for r in rows))}
        for name, logits in output["classification"].items():
            ids = [i for i, row in enumerate(rows) if row and name in row.get("classification", {})]
            if not ids:
                continue
            values = [rows[i]["classification"][name] for i in ids]
            if any(type(v) is not int or not 0 <= v < logits.shape[-1] for v in values):
                raise ValueError("Invalid class target")
            labels = torch.tensor(values, device=logits.device, dtype=torch.long)
            term = F.cross_entropy(logits[ids], labels)
            terms.append(term)
            metrics.update({f"answer_{name}_loss": float(term.detach()),
                            f"answer_{name}_accuracy": float((logits[ids].argmax(-1) == labels).float().mean().detach()),
                            f"answer_{name}_count": float(len(ids))})
        for name, prediction in output["regression"].items():
            ids = [i for i, row in enumerate(rows) if row and name in row.get("regression", {})]
            if not ids:
                continue
            values = [rows[i]["regression"][name] for i in ids]
            if any(len(v) != prediction.shape[-1] or any(not math.isfinite(x) or not 0 <= x <= 1 for x in v) for v in values):
                raise ValueError("Invalid normalized regression target")
            labels = prediction.new_tensor(values)
            term = F.mse_loss(prediction[ids], labels)
            terms.append(term)
            mae = (prediction[ids] - labels).abs().mean().detach()
            metrics.update({f"answer_{name}_loss": float(term.detach()),
                            f"answer_{name}_mae": float(mae),
                            f"answer_{name}_pixel_mae": float(mae * 255),
                            f"answer_{name}_count": float(len(ids))})
        loss = torch.stack(terms).mean() if terms else retrieved.float().sum() * 0
        metrics["answer_loss"] = float(loss.detach())
        metrics["answer_active_heads"] = float(len(terms))
        return loss, metrics
