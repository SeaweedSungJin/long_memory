"""Auxiliary recall targets supervise a retrieved latent, never model inputs.

These small heads predict an annotated subgoal class and normalized grounding
point from the mean retrieved memory vector. Ground-truth metadata is used only
on the loss side. High offline recall accuracy is not robot-task success and
does not, by itself, establish that a task requires long-term memory.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RecallHeads(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        for name, value in (("hidden_dim", hidden_dim), ("num_classes", num_classes)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.hidden_dim, self.num_classes = hidden_dim, num_classes
        self.trunk = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.subgoal = nn.Linear(hidden_dim, num_classes)
        self.grounding = nn.Linear(hidden_dim, 2)

    def forward(self, retrieved):
        if retrieved.ndim != 3 or retrieved.shape[-1] != self.hidden_dim or retrieved.shape[1] < 1:
            raise ValueError("Recall input must be retrieved [B,Q,hidden_dim] tokens")
        if not torch.isfinite(retrieved).all():
            raise FloatingPointError("Nonfinite retrieved memory latent")
        hidden = self.trunk(retrieved.mean(dim=1).float())
        return {"logits": self.subgoal(hidden), "xy": self.grounding(hidden).sigmoid()}


def recall_loss(heads, read, target):
    """Single-query masked CE + SmoothL1; trainer applies separate small weights.

    Returns tensor losses (including a differentiable zero for unavailable
    targets) and scalar/None metrics. Unknown class IDs and unobserved
    grounding targets are not silently changed into class zero or origin.
    """
    latent = read["read"]
    if latent.shape[0] != 1:
        raise ValueError("Recall labels are per-decision; expected batch size one")
    out = heads(latent)
    zero = out["logits"].sum() * 0 + out["xy"].sum() * 0
    ce, grounding = zero, zero
    class_valid, xy_valid = target.get("class_valid", False), target.get("xy_valid", False)
    if type(class_valid) is not bool or type(xy_valid) is not bool:
        raise ValueError("Recall validity fields must be booleans")
    accuracy, mae = None, None
    if class_valid:
        class_id = target["class_id"]
        if type(class_id) is not int or not 0 <= class_id < heads.num_classes:
            raise ValueError("Valid recall class is outside train vocabulary")
        label = torch.tensor([class_id], device=latent.device, dtype=torch.long)
        ce = F.cross_entropy(out["logits"], label)
        accuracy = float(out["logits"].detach().argmax(-1).item() == class_id)
    if xy_valid:
        xy = target["xy"]
        if len(xy) != 2 or any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in xy):
            raise ValueError("Valid grounding point must contain two finite normalized coordinates")
        label = out["xy"].new_tensor(xy).reshape(1, 2)
        grounding = F.smooth_l1_loss(out["xy"], label)
        mae = float((out["xy"].detach() - label).abs().mean())
    return {"loss": ce + grounding, "subgoal_loss": ce, "grounding_loss": grounding,
            "subgoal_accuracy": accuracy, "grounding_mae": mae,
            "class_valid": float(class_valid), "xy_valid": float(xy_valid)}


def recall_metrics(result):
    return {name: (float(value.detach()) if isinstance(value, torch.Tensor) else value)
            for name, value in result.items() if name != "loss"}
