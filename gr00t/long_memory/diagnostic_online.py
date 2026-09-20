"""Read-path-only causal interventions for a frozen v4 RoboMME checkpoint.

The native writer always updates its native bank first. Interventions alter only
the tokens passed to the Action Expert, never stored encodings or short-memory
inputs. Closed-loop trajectories may of course diverge after changed actions.
"""
from __future__ import annotations

from collections import deque

import torch

from .online_v3 import OnlineActionValueBank


MODES = ("baseline", "full", "expert-only", "no-old", "shuffled-old", "fifo")


def temporal_content_permutation(memory, rows, ids, old_ids):
    """Return independent key/value tensors; preserve every destination timestamp.

    This is a within-episode cyclic reassignment, NOT another-episode matched
    distractor control. Two old entries are required. The stored contextual roles
    are time-free: encode_events adds time only after key/value projection.
    Rebuilding from tokens + summary avoids subtraction/cancellation error.
    """
    donors = dict(zip(old_ids, old_ids[1:] + old_ids[:1])) if len(old_ids) >= 2 else {}
    keys, values = [], []
    for event_id in ids:
        destination = rows[event_id]
        if event_id not in donors:
            keys.append(destination["keys"])
            values.append(destination["values"])
            continue
        source = rows[donors[event_id]]
        roles = source["tokens"] + source["event"][None]
        times = memory.time_features(destination["starts"][None], destination["ends"][None])
        keys.append(memory.key(roles) + memory.time_key(times))
        values.append(memory.value(roles) + memory.time_value(times))
    return torch.stack(keys)[None], torch.stack(values)[None], donors


class DiagnosticOnlineBank(OnlineActionValueBank):
    def __init__(self, memory, stride, policy, *, memory_window, mode):
        if mode not in MODES[1:]:
            raise ValueError("A diagnostic bank needs an adapted-model mode")
        if type(memory_window) is not int or memory_window < 1:
            raise ValueError("memory_window must be a positive integer")
        self.memory_window, self.mode = memory_window, mode
        super().__init__(memory, stride, policy)

    def reset(self):
        super().reset()
        self.endpoint_frames = deque(maxlen=self.memory_window)
        self.last_intervention = {}

    def _read(self, endpoint):
        # The native advance calls _read once, AFTER native storage selection.
        self.endpoint_frames.append(endpoint.frame)
        boundary = self.endpoint_frames[0]
        old = [i for i in self.bank_ids if float(self.encodings[i]["ends"].item()) < boundary]
        original = super()._read(endpoint)
        result = original
        selected, donors = self.bank_ids.copy(), {}
        attempted = False
        if self.mode == "expert-only":
            # Keep the writer in shadow mode for diagnostics, but bypass fusion.
            result = {**original, "fused_short": endpoint.short[None]}
            attempted = bool(selected)
        elif self.mode == "no-old" and old:
            selected = [i for i in self.bank_ids if i not in old]
            attempted = True
        elif self.mode == "shuffled-old" and len(old) >= 2:
            attempted = True
        if self.mode in ("no-old", "shuffled-old") and attempted:
            ref = next(self.memory.parameters())
            if self.mode == "shuffled-old":
                keys, values, donors = temporal_content_permutation(
                    self.memory, self.encodings, selected, old)
            elif selected:
                keys = torch.stack([self.encodings[i]["keys"] for i in selected])[None]
                values = torch.stack([self.encodings[i]["values"] for i in selected])[None]
            else:
                shape = (1, 0, self.memory.num_roles, self.memory.config.hidden_dim)
                keys, values = ref.new_empty(shape), ref.new_empty(shape)
            mask = torch.ones((1, len(selected)), dtype=torch.bool, device=ref.device)
            result = self.memory.read(endpoint.short[None], endpoint.state[None], keys, values,
                                      mask, ref.new_tensor([float(endpoint.frame)]))
        difference = float((result["fused_short"] - original["fused_short"]).float().norm().item())
        self.last_intervention = {
            "diagnostic_mode": self.mode, "short_endpoint_frames": list(self.endpoint_frames),
            "old_boundary_frame": boundary, "old_event_ids": old, "old_event_count": len(old),
            "read_event_ids": [] if self.mode == "expert-only" else selected,
            "shuffled_donor_ids": {str(k): v for k, v in donors.items()},
            "intervention_applied": attempted, "intervention_changed_fused": difference > 0,
            "intervention_fused_delta_norm": difference,
            "shadow_bank_unchanged_by_read": True,
        }
        if self.mode == "expert-only":
            self.last_intervention["shadow_read_residual_norm"] = float(original["residual_norm"].item())
            self.last_intervention["residual_norm"] = 0.0
        return result

    @torch.inference_mode()
    def advance(self, *args, **kwargs):
        fused, diagnostics = super().advance(*args, **kwargs)
        diagnostics.update(self.last_intervention)
        return fused, diagnostics


def make_diagnostic_policy(base_model, checkpoint, device, mode):
    """Lazy heavy imports keep the bank tests and preflight model-free."""
    from .online_policy import _Session
    from .online_policy_v4 import LongMemoryV4Policy

    if mode not in MODES:
        raise ValueError(f"Unknown diagnostic mode: {mode}")
    if mode == "baseline":
        if checkpoint is not None:
            raise ValueError("Original HAMLET baseline must not load an adapter checkpoint")
        return LongMemoryV4Policy(base_model, device=device)
    if checkpoint is None:
        raise ValueError("Adapted diagnostic modes require the SAME Stage 2 checkpoint")

    class DiagnosticPolicy(LongMemoryV4Policy):
        def _session(self, sid, seed, reset):
            if reset:
                self.sessions.pop(sid, None)
            if sid not in self.sessions:
                bank = DiagnosticOnlineBank(self.memory, self.stride, self.write_policy,
                    memory_window=int(self.model.config.memory_window), mode=mode)
                generator = torch.Generator(device=self.model.device).manual_seed(seed)
                self.sessions[sid] = _Session(generator, seed, bank)
            session = self.sessions[sid]
            if session.episode_seed != seed:
                raise ValueError("Episode seed changed without resetting its session")
            self.sessions.move_to_end(sid)
            while len(self.sessions) > self.session_cap:
                self.sessions.popitem(last=False)
            return session

    policy = DiagnosticPolicy(base_model, checkpoint, device=device,
                              write_policy="all" if mode == "fifo" else "checkpoint")
    if policy.stage != 2 or policy.memory is None:
        raise ValueError("Diagnostic adapted modes require a Stage 2 memory-enabled v4 bundle")
    return policy
