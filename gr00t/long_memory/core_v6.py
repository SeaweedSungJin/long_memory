"""V6: ordered visual memory and read-time counterfactual value prediction.

This version deliberately does not import or mutate the v3-v5 writer. The
archive retains observations; a small detached proposal picks candidate sets,
while learned *soft* weights within each set receive action gradients. Values
keep multiple contextual image tokens instead of reducing an observation to a
single vector. Token order is retained, but camera/grid geometry is NOT known
from the old cache and no spatial-grid alignment is claimed.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


CANDIDATE_NAMES = ("uniform", "relevant", "hybrid", "null")


@dataclass(frozen=True)
class VisualMemoryV6Config:
    feature_dim: int
    state_dim: int
    hidden_dim: int = 256
    visual_tokens: int = 16
    num_heads: int = 4
    temporal_layers: int = 1
    max_archive_events: int = 256
    read_budget: int = 16
    time_scale: float = 16.0
    temperature: float = 0.2

    def __post_init__(self):
        for key in ("feature_dim", "state_dim", "hidden_dim", "visual_tokens", "num_heads",
                    "temporal_layers", "max_archive_events", "read_budget"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.read_budget > self.max_archive_events:
            raise ValueError("read_budget cannot exceed max_archive_events")
        for key in ("time_scale", "temperature"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")

    def to_dict(self):
        return asdict(self)


def _frame_number(frame):
    if torch.is_tensor(frame):
        if frame.numel() != 1:
            raise ValueError("frame must be one nonnegative integer")
        frame = frame.item()
    if isinstance(frame, bool) or not isinstance(frame, (int, float)) or not math.isfinite(frame) or frame < 0 or int(frame) != frame:
        raise ValueError("frame must be one nonnegative integer")
    return int(frame)


def observation_image_tokens(features, image_mask, attention_mask, short):
    """Select valid image tokens and always exclude the appended short tail.

    This helper also powers detached proposal scores, so offline and online
    histories share the exact masking rule. Missing images are errors, not a
    silently substituted all-zero representation.
    """
    features, short = torch.as_tensor(features), torch.as_tensor(short)
    image_mask = torch.as_tensor(image_mask, device=features.device)
    attention_mask = torch.as_tensor(attention_mask, device=features.device)
    if features.ndim != 2 or short.ndim != 2 or short.shape[0] < 1 or features.shape[1] != short.shape[1]:
        raise ValueError("features and short must be [tokens, matching feature_dim]")
    if features.shape[0] <= short.shape[0]:
        raise ValueError("features must include image tokens before the short tail")
    if image_mask.shape != features.shape[:1] or attention_mask.shape != image_mask.shape:
        raise ValueError("image/attention mask shape mismatch")
    if image_mask.dtype != torch.bool or attention_mask.dtype != torch.bool:
        raise ValueError("image/attention masks must be boolean")
    valid = (image_mask & attention_mask).clone()
    valid[-short.shape[0]:] = False
    if not bool(valid.any()):
        raise ValueError("Observation contains no valid image tokens after short-tail exclusion")
    result = features[valid].float()
    if not bool(torch.isfinite(result).all()) or not bool(torch.isfinite(short).all()):
        raise ValueError("Nonfinite selected image or short features")
    return result


class ReadTimeCVOM(nn.Module):
    """Predict signed action-loss gain for an ordered set, NOT task success.

    Query and reader representations are detached at this boundary. Null is
    anchored to exact zero by the caller, rather than learned as a free bias.
    """
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, 2 * hidden_dim,
                                            dropout=0.0, activation="gelu", batch_first=True,
                                            norm_first=True)
        self.set_encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.query_token = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim),
                                  nn.SiLU(), nn.Linear(hidden_dim, 1))
        # Initially all sets tie. Stage 1 never trains this scorer.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, query, ordered_events):
        query, ordered_events = query.detach().float(), ordered_events.detach().float()
        token = self.query_token(query).unsqueeze(1)
        encoded = self.set_encoder(torch.cat((token, ordered_events.unsqueeze(0)), dim=1))
        return self.head(torch.cat((encoded[:, 0], query), dim=-1)).reshape(())


class VisualMemoryV6(nn.Module):
    def __init__(self, config: VisualMemoryV6Config):
        super().__init__()
        self.config = config
        h, d = config.hidden_dim, config.feature_dim
        self.visual_projection = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h))
        self.token_position = nn.Parameter(torch.randn(config.visual_tokens, h) * 0.02)
        self.short_projection = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h))
        self.state_projection = nn.Linear(config.state_dim, h)
        self.time_projection = nn.Sequential(nn.Linear(8, h), nn.SiLU(), nn.Linear(h, h))
        self.summary_norm = nn.LayerNorm(h)
        layer = nn.TransformerEncoderLayer(h, config.num_heads, 2 * h, dropout=0.0,
                                            activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, config.temporal_layers, enable_nested_tensor=False)
        self.query = nn.Sequential(nn.LayerNorm(4 * h), nn.Linear(4 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.key = nn.Linear(h, h, bias=False)
        self.value_norm = nn.LayerNorm(h)
        self.cvom = ReadTimeCVOM(h, config.num_heads)
        self.float()

    def reader_parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith("cvom.")]

    def cvom_parameters(self):
        return list(self.cvom.parameters())

    def _device(self):
        return self.token_position.device

    def _time(self, frame, is_demo):
        frame = _frame_number(frame)
        if torch.is_tensor(is_demo):
            if is_demo.numel() != 1 or is_demo.dtype != torch.bool:
                raise ValueError("is_demo must be boolean")
            is_demo = bool(is_demo.item())
        if type(is_demo) is not bool:
            raise ValueError("is_demo must be boolean")
        t = frame / self.config.time_scale
        values = [math.sin(t / s) for s in (1, 10, 100)] + [math.cos(t / s) for s in (1, 10, 100)]
        return self.time_projection(self.token_position.new_tensor(values + [math.log1p(t), float(is_demo)]))

    def _inputs(self, features, image_mask, attention_mask, short, state):
        images = observation_image_tokens(features, image_mask, attention_mask, short).to(self._device())
        short = torch.as_tensor(short, device=self._device(), dtype=torch.float32)
        state = torch.as_tensor(state, device=self._device(), dtype=torch.float32)
        if images.shape[-1] != self.config.feature_dim or state.shape != (self.config.state_dim,):
            raise ValueError("Observation feature/state dimensions do not match config")
        if not bool(torch.isfinite(state).all()):
            raise ValueError("Nonfinite observation state")
        return images, short, state

    def encode_observation(self, features, image_mask, attention_mask, short, state, frame, is_demo=False):
        with torch.autocast(device_type=self._device().type, enabled=False):
            images, short, state = self._inputs(features, image_mask, attention_mask, short, state)
            # 1D adaptive bins retain token ordering; bins are NOT camera/grid cells.
            pooled = F.adaptive_avg_pool1d(images.T.unsqueeze(0), self.config.visual_tokens)[0].T
            values = self.visual_projection(pooled) + self.token_position
            time = self._time(frame, is_demo)
            summary = self.summary_norm(values.mean(0) + self.short_projection(short.mean(0))
                                        + self.state_projection(state) + time)
            return {"values": values, "summary": summary, "key": self.key(summary),
                    "frame": _frame_number(frame), "is_demo": bool(is_demo)}

    def query_features(self, features, image_mask, attention_mask, short, state, frame, is_demo=False):
        with torch.autocast(device_type=self._device().type, enabled=False):
            images, short, state = self._inputs(features, image_mask, attention_mask, short, state)
            return self.query(torch.cat((self.visual_projection(images.mean(0)),
                                          self.short_projection(short.mean(0)), self.state_projection(state),
                                          self._time(frame, is_demo)), dim=-1))

    def contextualize(self, observations):
        """Bidirectional mixing is safe here: every item is already in the past."""
        if not observations:
            return {}
        with torch.autocast(device_type=self._device().type, enabled=False):
            summaries = torch.stack([o["summary"] for o in observations])
            contexts = self.temporal(summaries[None])[0]
            return {"contexts": contexts, "keys": self.key(contexts),
                    "values": self.value_norm(torch.stack([o["values"] for o in observations]) + contexts[:, None])}

    def pack(self, query, contextual, positions, event_ids):
        if not positions:
            return {"tokens": None, "summary": query.new_zeros((1, self.config.hidden_dim)),
                    "event_ids": [], "event_weights": query.new_empty(0),
                    "event_summaries": query.new_empty((0, self.config.hidden_dim))}
        with torch.autocast(device_type=self._device().type, enabled=False):
            idx = torch.as_tensor(positions, dtype=torch.long, device=self._device())
            keys = contextual["keys"].index_select(0, idx)
            summaries = contextual["contexts"].index_select(0, idx)
            scores = (F.normalize(keys, dim=-1) * F.normalize(query.reshape(-1), dim=-1)).sum(-1) / self.config.temperature
            weights = scores.softmax(0)
            values = contextual["values"].index_select(0, idx)
            # Keep every selected event's token sequence; this is not a weighted
            # mean of all events. Scaling gives the soft scorer action gradients.
            weighted = values * (weights * len(positions))[:, None, None]
            return {"tokens": weighted.reshape(1, -1, self.config.hidden_dim),
                    "summary": (summaries * weights[:, None]).sum(0, keepdim=True),
                    "event_ids": list(event_ids), "event_weights": weights,
                    "event_summaries": summaries}

    def score_candidates(self, query, candidates):
        if tuple(candidates) != CANDIDATE_NAMES:
            raise ValueError("CVOM candidate names/order do not match the fixed v6 protocol")
        with torch.autocast(device_type=self._device().type, enabled=False):
            scores = []
            for name in CANDIDATE_NAMES:
                candidate = candidates[name]
                if name == "null" or candidate["tokens"] is None:
                    scores.append(query.new_zeros(()))
                else:
                    scores.append(self.cvom(query.reshape(1, -1), candidate["event_summaries"]))
            return torch.stack(scores)
