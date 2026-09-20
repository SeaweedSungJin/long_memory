"""Read-only storage-value critic: signed future loss gain of UPDATE vs KEEP.

The value function is shared between the two states, anchoring identical-state
utility at exactly zero. Its actor inputs are detached at this boundary.  The
score is a scaled signed regression output, never a success probability.
"""

import torch
from torch import nn

from .recurrent_v7 import MemoryV7Config, _finite, _floating


class CVOMV7(nn.Module):
    def __init__(self, config: MemoryV7Config):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.query = nn.Linear(config.num_short_tokens * d, d)
        self.key_norm = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, config.num_heads, dropout=0.0, bias=False, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * d, max(1, d // 2)), nn.SiLU(),
                                  nn.Linear(max(1, d // 2), 1, bias=False))
        nn.init.zeros_(self.head[-1].weight)

    def forward(self, encoded, state, candidate, slot_addresses):
        c = self.config
        device = self.query.weight.device
        if self.query.weight.dtype != torch.float32:
            raise TypeError("V7 CVOM parameters must stay FP32")
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 3:
            raise ValueError("encoded short must be [B,Q,d]")
        batch = encoded.shape[0]
        encoded = _floating(encoded, (batch, c.num_short_tokens, c.hidden_dim), "CVOM short", device).detach()
        state = _floating(state, (batch, c.capacity, c.hidden_dim), "CVOM memory", device).detach()
        candidate = _floating(candidate, state.shape, "CVOM candidate", device).detach()
        addresses = _floating(slot_addresses, (c.capacity, c.hidden_dim), "CVOM addresses", device).detach()
        with torch.autocast(device_type=device.type, enabled=False):
            query = self.query(encoded.flatten(1)).unsqueeze(1)

            def value(bank):
                pooled, _ = self.attention(query, self.key_norm(bank + addresses[None]), bank, need_weights=False)
                return self.head(torch.cat((query[:, 0], pooled[:, 0]), dim=-1)).squeeze(-1)

            gain = value(candidate) - value(state)
        _finite(gain, "CVOM prediction")
        return gain
