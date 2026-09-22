"""ECHO: causal completed-action events and slot-wise CVoM storage.

Existing HAMLET observations/readout remain the control path. Only stored
events receive a zero-initialized effect residual derived from an ALREADY
executed transition. The current query never receives that residual. Demo
observations remain ordinary visual events (they have no invented actions).

All bank updates are functional. Discrete storage choices detach their scores,
not the retained event tensors: later action losses can still train earlier
event encodings. Metadata is a provenance sidecar, never a ground-truth label.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from run_scripts.robomme.representation_core_v18 import RepresentationMemoryV18


ECHO_VERSION = "echo_cvom_v1"


@dataclass(frozen=True)
class EchoConfig:
    version: str = ECHO_VERSION
    capacity_events: int = 32
    min_fill: int = 4
    utility_hidden: int = 128
    effect_hidden: int = 128
    action_dim: int = 128
    write_threshold: float = .5
    recency_weight: float = .05
    diversity_weight: float = .05
    merge_threshold: float | None = None
    merge_max_gap: int = 32
    merge_max_count: int = 2

    def __post_init__(self):
        if self.version != ECHO_VERSION:
            raise ValueError("Unsupported ECHO version")
        for name in ("capacity_events", "min_fill", "utility_hidden", "effect_hidden", "action_dim", "merge_max_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_fill > self.capacity_events:
            raise ValueError("min_fill cannot exceed capacity")
        if type(self.merge_max_gap) is not int or self.merge_max_gap < 0:
            raise ValueError("merge_max_gap must be a nonnegative integer")
        for name in ("write_threshold", "recency_weight", "diversity_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.write_threshold > 1:
            raise ValueError("write_threshold must be in [0,1]")
        if self.merge_threshold is not None:
            if (isinstance(self.merge_threshold, bool) or not isinstance(self.merge_threshold, (int, float))
                    or not math.isfinite(self.merge_threshold) or not 0 <= self.merge_threshold <= 1):
                raise ValueError("merge_threshold must be None or a cosine threshold in [0,1]")
            if self.merge_max_count < 2:
                raise ValueError("Enabled merging requires merge_max_count >= 2")

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def utility_feature_dim(hidden_dim):
        return 3 * hidden_dim + 8


@dataclass(frozen=True)
class EchoBank:
    """Complete Q-token events plus provenance, never labels or future state.

    Merged events carry ordered source IDs and first/last observation frames.
    Their averaged token time components are NOT claimed to preserve temporal
    semantics losslessly; merging is disabled by default.
    """
    tokens: torch.Tensor
    first_frames: tuple[int, ...] = ()
    last_frames: tuple[int, ...] = ()
    is_demo: tuple[bool, ...] = ()
    counts: tuple[int, ...] = ()
    event_ids: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        if (not isinstance(self.tokens, torch.Tensor) or self.tokens.ndim != 3
                or self.tokens.shape[0] != 1 or not self.tokens.is_floating_point()):
            raise ValueError("EchoBank tokens must be floating [1,N*Q,D]")
        n = len(self.first_frames)
        if any(not isinstance(value, tuple) or len(value) != n for value in
               (self.first_frames, self.last_frames, self.is_demo, self.counts, self.event_ids)):
            raise ValueError("EchoBank metadata must be equally sized immutable tuples")
        if (n == 0 and self.tokens.shape[1] != 0) or (n and (not self.tokens.shape[1] or self.tokens.shape[1] % n)):
            raise ValueError("EchoBank must have equal complete-token event groups")
        previous, seen = -1, set()
        for first, last, demo, count, ids in zip(self.first_frames, self.last_frames, self.is_demo, self.counts, self.event_ids):
            if (type(first) is not int or type(last) is not int or first < 0 or first > last or first <= previous
                    or type(demo) is not bool or type(count) is not int or count <= 0
                    or not isinstance(ids, tuple) or len(ids) != count
                    or any(type(eid) is not int or eid < 0 or eid in seen for eid in ids)
                    or list(ids) != sorted(set(ids))):
                raise ValueError("Invalid chronological ECHO event provenance")
            seen.update(ids); previous = last
        if any(not a and b for a, b in zip(self.is_demo, self.is_demo[1:])):
            raise ValueError("Demo provenance must precede execution events")

    @property
    def n_events(self):
        return len(self.first_frames)

    @property
    def num_tokens(self):
        return self.tokens.shape[1] // self.n_events if self.n_events else 0

    def select(self, indices):
        ids = list(indices)
        if (ids != sorted(set(ids)) or any(type(i) is not int or not 0 <= i < self.n_events for i in ids)):
            raise ValueError("Select distinct increasing event indices; never reorder time")
        q = self.num_tokens
        value = torch.cat([self.tokens[:, i*q:(i+1)*q] for i in ids], dim=1) if ids else self.tokens[:, :0]
        return EchoBank(value, *(tuple(getattr(self, name)[i] for i in ids) for name in
            ("first_frames", "last_frames", "is_demo", "counts", "event_ids")))

    def without(self, indices):
        omit = set(indices)
        if any(type(i) is not int or not 0 <= i < self.n_events for i in omit):
            raise ValueError("Invalid removed event index")
        return self.select([i for i in range(self.n_events) if i not in omit])

    def append(self, candidate, frame, is_demo, event_id):
        if (not isinstance(candidate, torch.Tensor) or candidate.ndim != 3 or candidate.shape[0] != 1
                or not candidate.is_floating_point() or candidate.shape[1] < 1
                or candidate.shape[2] != self.tokens.shape[2] or candidate.dtype != self.tokens.dtype
                or candidate.device != self.tokens.device or (self.n_events and candidate.shape[1] != self.num_tokens)):
            raise ValueError("Append requires one same-device/dtype complete encoded event")
        if type(frame) is not int or type(is_demo) is not bool or type(event_id) is not int:
            raise ValueError("Append provenance requires integer frame/event_id and boolean is_demo")
        return EchoBank(torch.cat((self.tokens, candidate), dim=1), self.first_frames+(frame,),
            self.last_frames+(frame,), self.is_demo+(is_demo,), self.counts+(1,), self.event_ids+((event_id,),))

    def detach(self):
        return EchoBank(self.tokens.detach(), self.first_frames, self.last_frames, self.is_demo, self.counts, self.event_ids)


class SlotUtilityMLP(nn.Module):
    """Signed utility and write logit for EVERY retained slot plus candidate.

    Raw utility remains signed for regression. Only the deployment retention
    score clamps utility to zero, then adds explicit recency/diversity bonuses.
    Features are detached so critic supervision cannot rewrite the actor.
    """
    def __init__(self, dim, num_tokens, config: EchoConfig, time_scale=16.):
        super().__init__()
        self.dim, self.num_tokens, self.config, self.time_scale = dim, num_tokens, config, float(time_scale)
        if type(dim) is not int or dim <= 0 or type(num_tokens) is not int or num_tokens <= 0 or self.time_scale <= 0:
            raise ValueError("Invalid ECHO memory dimensions/time scale")
        self.feature_dim = config.utility_feature_dim(dim)
        self.trunk = nn.Sequential(nn.LayerNorm(self.feature_dim), nn.Linear(self.feature_dim, config.utility_hidden),
                                   nn.SiLU(), nn.Linear(config.utility_hidden, config.utility_hidden), nn.SiLU())
        self.utility_head = nn.Linear(config.utility_hidden, 1)
        self.write_head = nn.Linear(config.utility_hidden, 1)
        for layer in (self.utility_head, self.write_head):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def _inputs(self, bank, candidate, query, frame, is_demo):
        if not isinstance(bank, EchoBank) or bank.n_events > self.config.capacity_events:
            raise ValueError("ECHO requires a bounded EchoBank")
        for name, value in (("candidate", candidate), ("query", query)):
            if (not isinstance(value, torch.Tensor) or value.shape != (1, self.num_tokens, self.dim)
                    or not value.is_floating_point() or value.device != bank.tokens.device or value.dtype != bank.tokens.dtype):
                raise ValueError(f"{name} must be same-device/dtype [1,Q,D]")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Nonfinite {name}")
        if bank.tokens.shape != (1, bank.n_events*self.num_tokens, self.dim) or not bool(torch.isfinite(bank.tokens).all()):
            raise ValueError("Invalid ECHO bank shape/values")
        if type(frame) is not int or frame < 0 or type(is_demo) is not bool:
            raise ValueError("Current frame/demo must be causal integer/bool metadata")
        if bank.n_events and (frame <= bank.last_frames[-1] or (not bank.is_demo[-1] and is_demo)):
            raise ValueError("Candidate must follow the retained history chronologically")

    def forward_features(self, features):
        if (features.ndim != 2 or features.shape[1] != self.feature_dim
                or not features.is_floating_point() or not bool(torch.isfinite(features).all())):
            raise ValueError("Invalid slot-utility feature matrix")
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise TypeError("ECHO critic parameters must remain FP32")
        with torch.autocast(device_type=features.device.type, enabled=False):
            z = self.trunk(features.detach().float())
            utility, logit = self.utility_head(z).squeeze(-1), self.write_head(z).squeeze(-1)
        if not bool(torch.isfinite(utility).all() & torch.isfinite(logit).all()):
            raise FloatingPointError("Nonfinite ECHO utility/logit")
        return {"utility": utility, "logit": logit, "write_probability": logit.sigmoid()}

    def score(self, bank, candidate, query, frame, is_demo):
        self._inputs(bank, candidate, query, frame, is_demo)
        with torch.no_grad(), torch.autocast(device_type=candidate.device.type, enabled=False):
            own = torch.cat((bank.tokens, candidate), dim=1).detach().float().reshape(-1, self.num_tokens, self.dim).mean(1)
            current = query.detach().float().mean(1)
            mean = own[:-1].mean(0, keepdim=True) if bank.n_events else torch.zeros_like(current)
            n = own.shape[0]
            unit = F.normalize(own, dim=-1)
            cosine = unit @ unit.T
            if n > 1:
                cosine.fill_diagonal_(-1.)
                similarity = cosine.max(1).values
            else:
                similarity = own.new_zeros(1)
            novelty = ((1 - similarity)/2).clamp(0, 1)
            age = (frame - own.new_tensor(bank.last_frames+(frame,))) / self.time_scale
            first_age = (frame - own.new_tensor(bank.first_frames+(frame,))) / self.time_scale
            recency = 1 / (1 + age)
            stats = torch.stack((age.log1p(), first_age.log1p(), recency, novelty,
                F.cosine_similarity(own, current.expand_as(own), dim=-1),
                own.new_tensor(bank.is_demo+(is_demo,)),
                own.new_tensor(bank.counts+(1,)) / self.config.merge_max_count,
                own.new_tensor([0.] * bank.n_events + [1.])), dim=1)
            features = torch.cat((own, current.expand_as(own), mean.expand_as(own), stats), dim=1).detach()
        result = self.forward_features(features)
        return dict(result, features=features, novelty=novelty, recency=recency,
            retention=result["utility"].clamp_min(0) + self.config.recency_weight*recency + self.config.diversity_weight*novelty)

    def _merge_pair(self, bank, first):
        """Count-weighted adjacent compression; provenance survives, semantics unproven."""
        q, a, b = self.num_tokens, bank.counts[first], bank.counts[first+1]
        value = (a * bank.tokens[:, first*q:(first+1)*q] + b * bank.tokens[:, (first+1)*q:(first+2)*q])/(a+b)
        tokens = torch.cat((bank.tokens[:, :first*q], value, bank.tokens[:, (first+2)*q:]), dim=1)
        return EchoBank(tokens,
            bank.first_frames[:first]+(bank.first_frames[first],)+bank.first_frames[first+2:],
            bank.last_frames[:first]+(bank.last_frames[first+1],)+bank.last_frames[first+2:],
            bank.is_demo[:first]+(bank.is_demo[first],)+bank.is_demo[first+2:],
            bank.counts[:first]+(a+b,)+bank.counts[first+2:],
            bank.event_ids[:first]+(bank.event_ids[first]+bank.event_ids[first+1],)+bank.event_ids[first+2:])

    def _merge_index(self, augmented):
        if self.config.merge_threshold is None:
            return None
        pooled = augmented.tokens.detach().float().reshape(-1, self.num_tokens, self.dim).mean(1)
        options = []
        for i in range(augmented.n_events - 1):
            if (augmented.is_demo[i] != augmented.is_demo[i+1]
                    or augmented.first_frames[i+1]-augmented.last_frames[i] > self.config.merge_max_gap
                    or augmented.counts[i]+augmented.counts[i+1] > self.config.merge_max_count):
                continue
            similarity = float(F.cosine_similarity(pooled[i:i+1], pooled[i+1:i+2], dim=-1)[0])
            if similarity >= self.config.merge_threshold:
                options.append((similarity, i))
        return max(options, key=lambda item: (item[0], -item[1]))[1] if options else None

    def update(self, bank, candidate, query, frame, is_demo, event_id, mode="learned"):
        self._inputs(bank, candidate, query, frame, is_demo)
        if mode not in ("fifo", "learned"):
            raise ValueError("ECHO write mode must be fifo or learned")
        if type(event_id) is not int or event_id < 0:
            raise ValueError("event_id must be nonnegative")
        n, c = bank.n_events, self.config
        # Constructing this validates chronology/provenance even if the hard
        # decision will reject the candidate. Never mutate input tensors.
        augmented = bank.append(candidate, frame, is_demo, event_id)
        with torch.no_grad():
            prediction = self.score(bank, candidate, query, frame, is_demo)
            admit = bool(prediction["write_probability"][-1] >= c.write_threshold)
            victim, merge, operation = -1, None, "keep"
            if mode == "fifo":
                operation, victim = ("append", -1) if n < c.capacity_events else ("replace", 0)
            elif n < c.min_fill:
                operation = "append"
            elif admit:
                if n < c.capacity_events:
                    operation = "append"
                else:
                    retention = prediction["retention"]
                    # The candidate loses ties: uncertain equal-valued content
                    # cannot churn the bank merely due to tensor row order.
                    if bool(retention[-1] > retention[:-1].min()):
                        victim = int(retention[:-1].argmin())
                        merge = self._merge_index(augmented)
                        operation = "merge" if merge is not None else "replace"
        if operation == "append":
            result = augmented
        elif operation == "replace":
            result = augmented.without([victim])
        elif operation == "merge":
            result = self._merge_pair(augmented, merge)
            victim = -1  # A merge does not masquerade as an eviction.
        else:
            result = bank
        if result.n_events > c.capacity_events:
            raise RuntimeError("ECHO capacity violation")
        scalar = lambda value: candidate.new_tensor(float(value))
        return result, {"write_rate": scalar(operation != "keep"), "writer_insert": scalar(operation != "keep"),
            "writer_keep": scalar(operation == "keep"), "writer_append": scalar(operation == "append"),
            "writer_replace": scalar(operation == "replace"), "writer_merge": scalar(operation == "merge"),
            "writer_victim": scalar(victim), "writer_full": scalar(n == c.capacity_events),
            "writer_utility": prediction["utility"][-1].detach(),
            "writer_probability": prediction["write_probability"][-1].detach(),
            "bank_events": scalar(result.n_events)}


EchoStorageManager = SlotUtilityMLP


class CompletedEffectEncoder(nn.Module):
    """Observed pre/post moment + executed controls + state-change residual."""
    def __init__(self, feature_dim, state_dim, action_dim, num_tokens, hidden_dim, effect_hidden):
        super().__init__()
        self.feature_dim, self.state_dim, self.action_dim = feature_dim, state_dim, action_dim
        self.num_tokens, self.hidden_dim = num_tokens, hidden_dim
        self.input_dim, self.target_dim = 2*feature_dim + state_dim + action_dim, feature_dim + state_dim + action_dim
        self.encoder = nn.Sequential(nn.LayerNorm(self.input_dim), nn.Linear(self.input_dim, effect_hidden),
                                     nn.SiLU(), nn.Linear(effect_hidden, effect_hidden), nn.SiLU())
        self.output = nn.Linear(effect_hidden, num_tokens*hidden_dim)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.decoder = nn.Linear(effect_hidden, self.target_dim)

    def _inputs(self, pre_moment, post_moment, pre_state, post_state, actions, action_mask, valid):
        b = pre_moment.shape[0]
        device = self.output.weight.device
        if (pre_moment.shape != (b, self.num_tokens, self.feature_dim) or post_moment.shape != pre_moment.shape
                or pre_state.shape != (b, self.state_dim) or post_state.shape != pre_state.shape
                or actions.ndim != 3 or actions.shape[0] != b or actions.shape[2] != self.action_dim
                or action_mask.shape != actions.shape[:2] or action_mask.dtype != torch.bool
                or valid.shape != (b,) or valid.dtype != torch.bool):
            raise ValueError("Invalid completed-effect observation/control/mask shapes")
        active = valid.to(device) & action_mask.to(device).any(1)
        # The existing cache rounded normalized moments to BF16. Apply this
        # boundary to BOTH online/cached effect input, not the original reader.
        pre = pre_moment.to(device=device, dtype=torch.bfloat16).float().mean(1)
        post = post_moment.to(device=device, dtype=torch.bfloat16).float().mean(1)
        before, after = pre_state.to(device=device, dtype=torch.float32), post_state.to(device=device, dtype=torch.float32)
        mask = action_mask.to(device) & active[:, None]
        executed = torch.where(mask[:, :, None], actions.to(device=device, dtype=torch.float32), 0.)
        action_mean = executed.sum(1)/mask.sum(1).clamp_min(1)[:, None]
        delta = after-before
        inputs = torch.cat((pre, post, delta, action_mean), dim=1)
        target = torch.cat((post-pre, delta, action_mean), dim=1).detach()
        inputs = torch.where(active[:, None], inputs, 0.)
        target = torch.where(active[:, None], target, 0.)
        if not bool(torch.isfinite(inputs).all() & torch.isfinite(target).all()):
            raise FloatingPointError("Nonfinite valid completed-effect input")
        return inputs, target, active

    def _hidden(self, *args):
        inputs, target, active = self._inputs(*args)
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise TypeError("ECHO effect encoder parameters must remain FP32")
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            hidden = self.encoder(inputs)
        return hidden, target, active

    def forward(self, *args):
        hidden, _, active = self._hidden(*args)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            residual = self.output(hidden).reshape(-1, self.num_tokens, self.hidden_dim)
        return torch.where(active[:, None, None], residual, 0.)

    def reconstruction_loss(self, *args):
        hidden, target, active = self._hidden(*args)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            predicted = self.decoder(hidden)
            # Block-normalized, so the wide visual target does not silently
            # outweigh state/controls just by having more coordinates.
            f, s = self.feature_dim, self.state_dim
            errors = torch.stack([(predicted[:, part]-target[:, part]).square().mean(1)
                for part in (slice(0, f), slice(f, f+s), slice(f+s, None))], dim=1).mean(1)
            return torch.where(active, errors, 0.).sum()/active.sum().clamp_min(1)


class EchoMemoryV1(RepresentationMemoryV18):
    """Original V19 short reader + causal effect-enriched event storage."""
    def __init__(self, representation_config, echo_config: EchoConfig):
        if representation_config.representation != "short":
            raise ValueError("ECHO v1 supports only the original short representation")
        if representation_config.capacity_events != echo_config.capacity_events:
            raise ValueError("ECHO capacity must match its frozen-parent representation config")
        super().__init__(representation_config)
        self.echo_config = echo_config
        c = representation_config
        self.effect_adapter = CompletedEffectEncoder(c.feature_dim, c.state_dim, echo_config.action_dim,
            c.num_short_tokens, c.hidden_dim, echo_config.effect_hidden)
        self.manager = SlotUtilityMLP(c.hidden_dim, c.num_short_tokens, echo_config, c.time_scale)

    def initialize_from_parent_delta(self, state):
        """Strict original reader inventory; new effect/controller stay intact."""
        super().load_delta_state_dict(state)
        return self

    def delta_state_dict(self):
        result = super().delta_state_dict()
        for prefix, module in (("effect_adapter", self.effect_adapter), ("manager", self.manager)):
            result.update({f"{prefix}.{name}": value.detach().clone() for name, value in module.state_dict().items()})
        return result

    def load_delta_state_dict(self, state):
        expected = self.state_dict()
        if set(state) != set(expected):
            raise ValueError("ECHO checkpoint tensor inventory differs")
        for name, value in state.items():
            if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                    or value.dtype != expected[name].dtype or not bool(torch.isfinite(value).all())):
                raise ValueError(f"ECHO checkpoint shape/dtype/finiteness mismatch: {name}")
        with torch.no_grad():
            for name, value in state.items():
                expected[name].copy_(value)

    def encode_completed_effect(self, pre_moment, post_moment, pre_state, post_state, actions, action_mask, valid):
        return self.effect_adapter(pre_moment, post_moment, pre_state, post_state, actions, action_mask, valid)

    def _effect_prefix_inputs(self, episode, count):
        if count <= 1:
            return None
        needed = ("moment", "actions", "action_mask", "transition_valid")
        if any(name not in episode for name in needed):
            raise ValueError("ECHO needs completed executed actions/masks and observed moments")
        if (len(episode["moment"]) < count or any(len(episode[name]) < count-1 for name in needed[1:])):
            raise ValueError("Incomplete causal completed-transition prefix")
        valid = torch.as_tensor(episode["transition_valid"][:count-1])
        demo = torch.as_tensor(episode["is_demo"][:count])
        if valid.dtype != torch.bool or valid.shape != (count-1,):
            raise ValueError("transition_valid must be boolean [T]")
        valid = valid & ~demo[:-1] & ~demo[1:]
        return (episode["moment"][:count-1], episode["moment"][1:count],
            episode["state"][:count-1], episode["state"][1:count],
            episode["actions"][:count-1], episode["action_mask"][:count-1], valid)

    def encode_prefix(self, episode, count, *, activation_checkpointing=False):
        result = super().encode_prefix(episode, count, activation_checkpointing=activation_checkpointing)
        arguments = self._effect_prefix_inputs(episode, count)
        if arguments is None:
            residual = torch.zeros_like(result["stored"])
        else:
            tail = self.encode_completed_effect(*arguments)
            residual = torch.cat((torch.zeros_like(result["stored"][:1]), tail), dim=0)
        return dict(result, stored=result["stored"]+residual, effect=residual)

    def effect_loss(self, episode, count):
        # Parent prefix validation is reused without reading action targets.
        from gr00t.long_memory.replay_v7 import _validate_prefix
        _validate_prefix(episode, count)
        if count < 1:
            raise ValueError("effect_loss needs a nonempty observation prefix")
        arguments = self._effect_prefix_inputs(episode, count)
        if arguments is None:
            return sum(parameter.sum()*0 for parameter in self.effect_adapter.parameters())
        return self.effect_adapter.reconstruction_loss(*arguments)

    def initial_state(self):
        return EchoBank(super().initial_bank())

    def replay(self, episode, decision, *, write_mode="fifo", read_enabled=True, activation_checkpointing=False):
        if type(decision) is not int or not 0 <= decision < len(episode["frames"]):
            raise ValueError("Invalid ECHO action-query observation")
        encoded = self.encode_prefix(episode, decision+1, activation_checkpointing=activation_checkpointing)
        bank, writes = self.initial_state(), []
        for i in range(decision):
            bank, metrics = self.manager.update(bank, encoded["stored"][i:i+1], encoded["query"][i:i+1],
                int(episode["frames"][i]), bool(episode["is_demo"][i]), i, mode=write_mode)
            writes.append(metrics)
        short, query = encoded["short"][decision:decision+1], encoded["query"][decision:decision+1]
        fused, metrics = self.read_from_bank(short, query, bank.tokens, read_enabled=read_enabled)
        metrics.update(replayed_observations=fused.new_tensor(float(decision)),
            write_rate=torch.stack([row["write_rate"] for row in writes]).mean() if writes else fused.new_zeros(()))
        return {"fused": fused, "short": short, "bank": bank.tokens, "bank_state": bank,
                "encoded_current": query, "stored_current": encoded["stored"][decision:decision+1], "metrics": metrics}

    def online_step(self, short, query, stored, *, bank_state=None, frame, is_demo, event_id,
                    write_mode="learned", read_enabled=True, write_enabled=True):
        if type(read_enabled) is not bool or type(write_enabled) is not bool:
            raise ValueError("read_enabled/write_enabled must be boolean")
        bank = self.initial_state() if bank_state is None else bank_state
        fused, metrics = self.read_from_bank(short, query, bank.tokens, read_enabled=read_enabled)
        if write_enabled:
            bank, write_metrics = self.manager.update(bank, stored, query, frame, is_demo, event_id, mode=write_mode)
            metrics.update(write_metrics)
        else:
            metrics["write_rate"] = short.new_zeros(())
        return {"fused": fused, "short": short, "bank": bank.tokens, "bank_state": bank,
                "moment_history": None, "encoded_current": query, "stored_current": stored, "metrics": metrics}

    def step(self, short, moment, state, frames, is_demo, *, bank_state=None, previous_moment=None,
             previous_state=None, previous_is_demo=None, completed_actions=None, completed_action_mask=None,
             transition_valid=None, event_index=None, write_mode="learned", read_enabled=True, write_enabled=True):
        if type(event_index) is not int or event_index < 0:
            raise ValueError("Online ECHO requires explicit nonnegative event_index")
        if short.shape[0] != 1:
            raise ValueError("Online ECHO supports one causal episode per session")
        short = short.to(device=self.device, dtype=torch.float32)
        query = self.encode_event(short, state, frames, is_demo)
        completed = (previous_moment, previous_state, previous_is_demo,
                     completed_actions, completed_action_mask, transition_valid)
        stored = query
        if any(value is not None for value in completed):
            if any(value is None for value in completed):
                raise ValueError("Pass all completed-transition inputs together")
            valid = torch.as_tensor(transition_valid, device=self.device)
            prev_demo = torch.as_tensor(previous_is_demo, device=self.device)
            now_demo = torch.as_tensor(is_demo, device=self.device)
            if valid.shape != (1,) or prev_demo.shape != (1,) or valid.dtype != torch.bool or prev_demo.dtype != torch.bool:
                raise ValueError("Transition validity and previous demo must be explicit boolean [1]")
            valid = valid & ~prev_demo & ~now_demo
            stored = query + self.encode_completed_effect(previous_moment, moment, previous_state, state,
                completed_actions, completed_action_mask, valid)
        return self.online_step(short, query, stored, bank_state=bank_state,
            frame=int(torch.as_tensor(frames).reshape(-1)[0]), is_demo=bool(torch.as_tensor(is_demo).reshape(-1)[0]),
            event_id=event_index, write_mode=write_mode, read_enabled=read_enabled, write_enabled=write_enabled)
