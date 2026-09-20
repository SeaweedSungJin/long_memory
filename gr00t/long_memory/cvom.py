"""Fixed Stage-1 teacher's sampled conditional marginal action-loss gain.

Not exact Shapley, rollout reward, or optimal eviction value. Both coalitions
exclude events occurring after the candidate, even at later query decisions.
Teacher and student share only the immutable action expert, not learned memory.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import torch

from .hamlet import episode_flow_loss
from .replay import encode_until, read_bank


@dataclass
class CVoMConfig:
    future_samples: int = 2
    coalitions: int = 1
    far_delay: int = 4
    utility_scale: float = 0.01
    write_delta: float = 0.0
    seed: int = 71

    def __post_init__(self):
        if self.future_samples < 1 or self.coalitions < 1 or self.far_delay < 1:
            raise ValueError("CVoM sample counts/delay must be positive")
        if self.utility_scale <= 0:
            raise ValueError("utility_scale must be fixed and positive")
        if self.write_delta < 0:
            raise ValueError("write_delta must be nonnegative")


def candidate_indices(episode):
    decisions = torch.where(episode["decision_mask"])[0].tolist()
    if not decisions:
        return []
    return [i for i in range(decisions[-1]) if bool(episode["transition_valid"][i])]


class CVoMLabels:
    def __init__(self, teacher, action_head, config, output_dir, identity):
        self.teacher, self.action_head, self.config = teacher, action_head, config
        teacher.eval().requires_grad_(False)
        self.path = Path(output_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        payload = {"config": asdict(config), "identity": identity, "definition": "past-coalition-addition-v1"}
        manifest = self.path / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != payload:
            raise ValueError("CVoM cache belongs to a different teacher/data/label configuration")
        if not manifest.exists():
            manifest.write_text(json.dumps(payload, indent=2))

    @torch.no_grad()
    def get(self, episode, candidate):
        if candidate not in candidate_indices(episode):
            raise ValueError("Candidate has no valid future action decision")
        eid = int(episode["episode_id"])
        path = self.path / f"episode-{eid:06d}-event-{candidate:06d}.json"
        if path.exists():
            return json.loads(path.read_text())
        cfg = self.config
        seed = cfg.seed + eid * 100003 + candidate * 1009
        rng = random.Random(seed)
        future = [i for i in torch.where(episode["decision_mask"])[0].tolist() if i > candidate]
        near = [i for i in future if i - candidate < cfg.far_delay]
        far = [i for i in future if i - candidate >= cfg.far_delay]
        selected = []
        if near:
            selected.append(rng.choice(near))
        if far and len(selected) < cfg.future_samples:
            selected.append(rng.choice(far))
        remaining = [i for i in future if i not in selected]
        selected += rng.sample(remaining, min(len(remaining), cfg.future_samples - len(selected)))
        if not selected:
            selected = [rng.choice(future)]
        encoded = encode_until(self.teacher, episode, max(selected))
        past = [i for i in range(candidate) if bool(episode["transition_valid"][i])]
        gains = []
        for coalition in range(cfg.coalitions):
            count = rng.randint(0, min(len(past), self.teacher.config.capacity - 1))
            bank = sorted(rng.sample(past, count))
            for decision in selected:
                noise_seed = seed + coalition * 131 + decision * 7
                minus = read_bank(self.teacher, episode, decision, bank, encoded)
                plus = read_bank(self.teacher, episode, decision, bank + [candidate], encoded)
                l0 = episode_flow_loss(self.action_head, episode, decision, minus["fused_short"], seed=noise_seed)["loss"]
                l1 = episode_flow_loss(self.action_head, episode, decision, plus["fused_short"], seed=noise_seed)["loss"]
                gains.append(float((l0 - l1).item()))
        gain = sum(gains) / len(gains)
        if not torch.isfinite(torch.tensor(gain)):
            raise FloatingPointError("Nonfinite CVoM teacher gain")
        record = {"episode_id": eid, "candidate": candidate, "signed_gain": gain,
                  "utility_target": max(gain, 0.0) / cfg.utility_scale,
                  "write_target": int(gain > cfg.write_delta),
                  "future_decisions": selected, "paired_gains": gains}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False))
        temporary.replace(path)
        return record
