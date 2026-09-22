"""CPU causal/context tests for additive all-slot Echo teacher labels."""
from dataclasses import dataclass
import copy
import json
import random
from types import SimpleNamespace
import unittest

import torch

from run_scripts.robomme.echo_cvom_teacher import (
    _context_events, _summarize, additive_pair, build_plan, coalition_indices, label_contexts,
)


@dataclass(frozen=True)
class _Bank:
    tokens: torch.Tensor
    event_ids: tuple = ()
    first_frames: tuple = ()
    last_frames: tuple = ()
    is_demo: tuple = ()
    counts: tuple = ()

    @property
    def n_events(self):
        return len(self.event_ids)

    def append(self, candidate, frame, is_demo, event_id):
        assert not self.last_frames or frame > self.last_frames[-1]
        return _Bank(torch.cat((self.tokens, candidate), dim=1), (*self.event_ids, (event_id,)),
            (*self.first_frames, frame), (*self.last_frames, frame), (*self.is_demo, is_demo), (*self.counts, 1))

    def select(self, indices):
        assert indices == sorted(set(indices))
        return _Bank(self.tokens[:, indices], *(tuple(getattr(self, key)[index] for index in indices)
                     for key in ("event_ids", "first_frames", "last_frames", "is_demo", "counts")))


class _Manager(torch.nn.Module):
    def __init__(self, capacity=3):
        super().__init__()
        self.capacity, self.updates, self.scored = capacity, [], []
        self.eval()

    def update(self, bank, candidate, query, frame, is_demo, event_id, mode):
        assert not torch.is_grad_enabled()
        self.updates.append((event_id, mode))
        if mode == "learned" and bank.n_events == self.capacity and event_id % 2:
            return bank, {"keep": True}
        pool = bank.append(candidate, frame, is_demo, event_id)
        return pool.select(list(range(max(0, pool.n_events - self.capacity), pool.n_events))), {}

    def score(self, bank, candidate, query, frame, is_demo):
        assert not torch.is_grad_enabled()
        events = torch.cat((bank.tokens, candidate), dim=1)[0]
        features = torch.cat((events, query[0].expand(events.shape[0], -1),
            torch.full((events.shape[0], 1), float(bank.tokens.sum())),
            torch.full((events.shape[0], 1), float(frame)),
            torch.full((events.shape[0], 1), float(is_demo))), dim=-1)
        self.scored.append((bank, candidate.clone(), query.clone(), features.clone()))
        return {"features": features}


class _Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reference = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.manager = _Manager()
        self.config = SimpleNamespace(capacity_events=3, short_window=2, hidden_dim=2, num_short_tokens=1)
        self.reads = []
        self.eval()

    def encode_prefix(self, ep, count):
        assert not torch.is_grad_enabled()
        values = torch.arange(1, count + 1, dtype=torch.float32).reshape(count, 1, 1).expand(-1, 1, 2)
        return {"stored": values, "short": values + 100, "query": values + 200}

    def initial_state(self):
        return _Bank(torch.zeros(1, 0, 2))

    def read_from_bank(self, short, query, bank):
        assert not torch.is_grad_enabled()
        self.reads.append((short.clone(), query.clone(), bank.clone()))
        return torch.tensor([[[float(short.mean()), float(query.mean()), bank.shape[1], float(bank.sum())]]]), {}


def _episodes():
    records = {eid: {"episode_id": eid, "frames": 7 * torch.arange(20),
        "decision_mask": torch.arange(20) >= 3, "is_demo": torch.arange(20) < 3}
        for eid in range(4)}
    return SimpleNamespace(fetch=lambda eid: records[eid], records=records)


def _row(event=3, **extra):
    return {"episode_id": 0, "event": event, "future": [event + 3, event + 6], "task": "a",
            "split": "train", "write_mode": "fifo", "targets_per_context": 4, **extra}


def _flow(head, ep, query, fused, **kwargs):
    return {"loss": -fused[0, 0, -1]}


def test_plan_covers_each_split_episode_early_late_and_demonstrations():
    cache = SimpleNamespace(manifest={"splits": {"train": [0, 1], "val": [2, 3]}})
    tasks, config = {0: "a", 1: "b", 2: "a", 3: "b"}, _Core().config
    for split in ("train", "val"):
        args = (cache, _episodes(), tasks, config)
        rows = build_plan(*args, split=split, seed=42)
        assert rows == build_plan(*args, split=split, seed=42)
        assert len(rows) == 4
        assert {row["episode_id"] for row in rows} == set(cache.manifest["splits"][split])
        for eid in cache.manifest["splits"][split]:
            selected = [row for row in rows if row["episode_id"] == eid]
            assert len({row["event"] for row in selected}) == 2
            assert any(row["event"] < 3 for row in selected)  # demo/pre-capacity
            assert any(row["event"] >= 3 for row in selected)  # overflow
            for row in selected:
                assert len(row["future"]) == 2
                assert all(query > row["event"] + config.short_window for query in row["future"])
                assert row["split"] == split and row["write_mode"] == "fifo"
        bounded = build_plan(*args, split=split, seed=42, limit=2, write_mode="learned")
        assert len(bounded) == 2 and {row["task"] for row in bounded} == {"a", "b"}
        assert all(row["write_mode"] == "learned" for row in bounded)


def test_plan_uses_masks_not_gt_values_and_rejects_split_overlap_or_test():
    class NoActionAccess(dict):
        def __getitem__(self, name):
            assert name not in ("targets", "actions", "target_mask", "success")
            return super().__getitem__(name)
    episodes = _episodes()
    episodes.records[0] = NoActionAccess(episodes.records[0])
    cache = SimpleNamespace(manifest={"splits": {"train": [0], "val": [1]}})
    rows = build_plan(cache, episodes, {0: "a", 1: "b"}, _Core().config, split="train", seed=7)
    assert len(rows) == 2
    for split in ("test", "benchmark_val"):
        with unittest.TestCase().assertRaisesRegex(ValueError, "Only TRAIN"):
            build_plan(cache, episodes, {}, _Core().config, split=split, seed=7)
    cache.manifest["splits"]["val"] = [0]
    with unittest.TestCase().assertRaisesRegex(ValueError, "disjoint"):
        build_plan(cache, episodes, {0: "a"}, _Core().config, split="train", seed=7)


def test_plan_handles_short_episode_and_one_context_without_duplicate_events():
    episodes = _episodes()
    for episode in episodes.records.values():
        episode["frames"] = episode["frames"][:8]
        episode["decision_mask"] = episode["decision_mask"][:8]
        episode["is_demo"] = episode["is_demo"][:8]
    cache = SimpleNamespace(manifest={"splits": {"train": [0], "val": [1]}})
    config = SimpleNamespace(capacity_events=32, short_window=2)
    rows = build_plan(cache, episodes, {0: "a"}, config, split="train", seed=5, contexts_per_episode=10)
    assert len(rows) == len({row["event"] for row in rows}) == 4
    assert all(row["event"] < 32 for row in rows)
    one = build_plan(cache, episodes, {0: "a"}, config, split="train", seed=5, contexts_per_episode=1)
    assert len(one) == 1 and one[0]["event"] < 3


def test_context_sampling_covers_both_phases_even_when_demo_spans_capacity():
    eligible = list(range(18))
    demo = torch.arange(18) < 14
    for seed in range(40):
        selected = _context_events(eligible, demo, 3, 2, random.Random(seed))
        assert len(selected) == 2
        assert selected[0] < 3  # pre-capacity demonstration
        assert selected[1] >= 14  # execution, not another late demonstration
        all_selected = _context_events(eligible, demo, 3, 18, random.Random(seed))
        assert all_selected == eligible


def test_context_sampling_phase_and_capacity_fallbacks():
    eligible = list(range(8))
    demo = torch.arange(8) < 5
    for seed in range(20):
        # Both phases remain represented if neither has a post-capacity slot.
        phase = _context_events(eligible, demo, 32, 2, random.Random(seed))
        assert phase[0] < 5 <= phase[1]
        for phase_mask in (torch.zeros(8, dtype=torch.bool), torch.ones(8, dtype=torch.bool)):
            capacity = _context_events(eligible, phase_mask, 3, 2, random.Random(seed))
            assert capacity[0] < 3 <= capacity[1]
            temporal = _context_events(eligible, phase_mask, 32, 2, random.Random(seed))
            assert temporal[0] < 4 <= temporal[1]
        one = _context_events(eligible, demo, 3, 1, random.Random(seed))
        assert len(one) == 1 and one[0] < 5  # documented smoke-only demo bias


def test_coalitions_are_additive_target_excluding_and_bounded_with_uniform_sizes():
    sampled = coalition_indices(4, 1, 3, 40, 42)
    assert sampled == coalition_indices(4, 1, 3, 40, 42)
    assert len(sampled[0]) == 2 and {len(row) for row in sampled[1:]} == {0, 1, 2}
    assert all(1 not in row and row == sorted(set(row)) for row in sampled)
    assert any(3 in row for row in sampled)  # old target may condition on current known candidate
    bank = _Bank(torch.zeros(1, 0, 2))
    for index in range(4):
        bank = bank.append(torch.full((1, 1, 2), float(index + 1)), index, index < 2, index)
    for subset in sampled:
        left, right = additive_pair(bank, 1, subset, 3)
        assert right.n_events == left.n_events + 1 <= 3
        assert 1 not in [group[0] for group in left.event_ids]
        assert [group[0] for group in right.event_ids] == sorted([*subset, 1])
        assert right.first_frames == tuple(sorted([*subset, 1]))
    assert coalition_indices(1, 0, 1, 4, 9) == [[], [], [], []]
    for bad in ([1], [0, 2, 3], [2, 0], [0, 0], [4]):
        with unittest.TestCase().assertRaises(ValueError):
            additive_pair(bank, 1, bad, 3)


def test_teacher_scores_all_slots_with_current_context_and_matched_future_noise():
    core, calls = _Core(), []
    def flow(head, ep, query, fused, **kwargs):
        assert kwargs["tail_weight"] == .25 and kwargs["activation_checkpointing"] is False
        calls.append((query, kwargs["seed"], float(fused[0, 0, 2])))
        return _flow(head, ep, query, fused, **kwargs)
    before = torch.get_rng_state().clone()
    packet = label_contexts(core, None, _episodes(), [_row()], seed=19, flow_fn=flow,
                            teacher_snapshot_version="sha256:teacher-a")
    assert torch.equal(before, torch.get_rng_state())
    context = packet["contexts"][0]
    assert packet["actual_actor_calls"] == context["actual_actor_calls"] == len(calls) == 128
    assert context["bank_event_ids"] == [[0], [1], [2]] and context["candidate_event_id"] == 3
    assert [target["index"] for target in context["targets"]] == [0, 1, 2, 3]
    for target in context["targets"]:
        assert target["features"] == core.manager.scored[0][3][target["index"]].tolist()
        assert target["features"][2:4] == [204., 204.]  # current t query, never future query
        assert target["features"][4] == 12.  # whole ACTUAL bank, never coalition summary
        assert target["signed_mean"] == target["raw_signed_mean"] == target["positive_utility"] == 2 * (target["event_id"] + 1)
        assert target["num_draws"] == 16
        assert "NOT episode confidence interval" in target["uncertainty_interpretation"]
    for without, with_target in zip(calls[::2], calls[1::2]):
        assert without[:2] == with_target[:2]
        assert with_target[2] == without[2] + 1
    assert core.manager.updates == [(0, "fifo"), (1, "fifo"), (2, "fifo")]
    for short, query, bank in core.reads:
        assert float(short.mean()) in (107., 110.)
        assert float(query.mean()) in (207., 210.)
        assert not bank.numel() or float(bank.max()) <= 4.  # no future stored events
    assert packet["settings"]["teacher_snapshot_version"] == "sha256:teacher-a"
    assert "NO future or intermediate writes" in packet["settings"]["conditional_bank"]
    json.dumps(packet, allow_nan=False)


def test_positive_target_clips_after_averaging_and_negative_targets_are_not_discarded():
    summary = _summarize([[[0., 0.]]], [[[-2., 4.]]], [[]], [[[1, 2]]])
    assert summary["gains"] == [[[2., -4.]]]
    assert summary["raw_signed_mean"] == -1. and summary["positive_utility"] == 0.
    core = _Core()
    packet = label_contexts(core, None, _episodes(), [_row()], seed=8,
        flow_fn=lambda head, ep, query, fused, **kwargs: {"loss": fused[0, 0, -1]})
    assert len(packet["contexts"][0]["targets"]) == 4
    assert all(target["raw_signed_mean"] < 0 and target["positive_utility"] == 0
               for target in packet["contexts"][0]["targets"])


def test_pre_capacity_and_empty_bank_contexts_keep_all_available_targets():
    for event, expected in ((0, [0]), (1, [0, 1]), (2, [0, 1, 2])):
        result = label_contexts(_Core(), None, _episodes(), [_row(event)], seed=5, flow_fn=_flow)
        targets = result["contexts"][0]["targets"]
        assert [target["event_id"] for target in targets] == expected
        assert targets[-1]["is_new"] and sum(target["is_new"] for target in targets) == 1
    only_new = label_contexts(_Core(), None, _episodes(), [_row(targets_per_context=1)], seed=5, flow_fn=_flow)
    assert [target["event_id"] for target in only_new["contexts"][0]["targets"]] == [3]


def test_old_target_sampling_is_not_fixed_to_oldest_and_always_includes_new():
    selected = set()
    for seed in range(12):
        packet = label_contexts(_Core(), None, _episodes(), [_row(targets_per_context=2)],
            seed=seed, coalitions=1, noise_samples=1, flow_fn=_flow)
        targets = packet["contexts"][0]["targets"]
        assert len(targets) == 2 and targets[-1]["event_id"] == 3
        selected.add(targets[0]["event_id"])
    assert selected == {0, 1, 2}


def test_learned_collection_replays_actual_writer_state_and_requires_snapshot():
    row = _row(5, write_mode="learned")
    with unittest.TestCase().assertRaisesRegex(ValueError, "snapshot"):
        label_contexts(_Core(), None, _episodes(), [row], seed=4, flow_fn=_flow)
    core = _Core()
    learned = label_contexts(core, None, _episodes(), [row], seed=4, flow_fn=_flow,
        teacher_snapshot_version="immutable:refresh-2")["contexts"][0]
    fifo = label_contexts(_Core(), None, _episodes(), [_row(5)], seed=4, flow_fn=_flow)["contexts"][0]
    assert learned["bank_event_ids"] == [[1], [2], [4]]
    assert fifo["bank_event_ids"] == [[2], [3], [4]]
    assert learned["targets"][-1]["features"] != fifo["targets"][-1]["features"]
    assert core.manager.updates == [(index, "learned") for index in range(5)]
    assert learned["targets"][-1]["features"] == core.manager.scored[0][3][-1].tolist()


def test_teacher_determinism_progress_and_row_order_independence():
    a, b = _row(), _row(5, episode_id=1, task="b")
    progress = []
    left = label_contexts(_Core(), None, _episodes(), [a, b], seed=11, flow_fn=_flow,
                          progress=lambda *values: progress.append(values))
    right = label_contexts(_Core(), None, _episodes(), [b, a], seed=11, flow_fn=_flow)
    assert left["contexts"] == list(reversed(right["contexts"]))
    assert [(current, total) for current, total, _ in progress] == [(1, 2), (2, 2)]
    assert left["actual_actor_calls"] == right["actual_actor_calls"]


def test_invalid_causal_queries_trainable_teacher_and_nonfinite_loss_fail_closed():
    core = _Core()
    core.requires_grad_(True)
    with unittest.TestCase().assertRaisesRegex(ValueError, "explicitly frozen"):
        label_contexts(core, None, _episodes(), [_row()], seed=1, flow_fn=_flow)
    core.requires_grad_(False).train()
    with unittest.TestCase().assertRaisesRegex(ValueError, "explicitly frozen"):
        label_contexts(core, None, _episodes(), [_row()], seed=1, flow_fn=_flow)
    core.eval()
    for row in (_row(event=-1), _row(future=[5]), _row(future=[9, 6]), _row(future=[6, 6]),
                _row(future=[999]), _row(split="test"), _row(write_mode="oracle")):
        with unittest.TestCase().assertRaises(ValueError):
            label_contexts(core, None, _episodes(), [row], seed=1, flow_fn=_flow)
    with unittest.TestCase().assertRaises(FloatingPointError):
        label_contexts(core, None, _episodes(), [_row()], seed=1,
                       flow_fn=lambda *args, **kwargs: {"loss": float("nan")})
    with unittest.TestCase().assertRaisesRegex(ValueError, "Production labels require"):
        label_contexts(core, None, _episodes(), [_row()], seed=1)


def test_real_echo_core_features_match_exact_prefix_replay_and_ignore_future_contents():
    from run_scripts.robomme.echo_cvom_core import EchoConfig, EchoMemoryV1
    from tests.test_representation_core_v18 import config, episode
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        ep = episode(14)
        ep.update(actions=torch.randn(13, 2, 8), action_mask=torch.ones(13, 2, dtype=torch.bool),
                  transition_valid=torch.ones(13, dtype=torch.bool), decision_mask=torch.arange(13) >= 2)
        core = EchoMemoryV1(config(representation="short"),
            EchoConfig(capacity_events=3, min_fill=2, action_dim=8, utility_hidden=8, effect_hidden=8))
        with torch.no_grad():
            core.effect_adapter.output.weight.normal_(std=.05)
            core.memory.fusion_projection.weight.normal_(std=.05)
            core.manager.write_head.bias.fill_(-20.)
        core.eval().requires_grad_(False)
    state_before = {name: value.clone() for name, value in core.state_dict().items()}
    episodes = SimpleNamespace(fetch=lambda _: ep)
    row = _row(4, future=[8, 11], write_mode="learned")
    packet = label_contexts(core, None, episodes, [row], seed=5, coalitions=2, noise_samples=1,
        flow_fn=lambda head, episode, query, fused, **kw: {"loss": fused.square().mean()},
        teacher_snapshot_version="checkpoint:stage2-teacher-000007")
    context = packet["contexts"][0]
    with torch.no_grad():
        replay = core.replay(ep, 4, write_mode="learned")
        score = core.manager.score(replay["bank_state"], replay["stored_current"], replay["encoded_current"],
                                   int(ep["frames"][4]), bool(ep["is_demo"][4]))
    assert context["bank_event_ids"] == [[0], [1]]  # actual learned KEEP state, not FIFO suffix
    for target in context["targets"]:
        assert target["features"] == score["features"][target["index"]].tolist()
    changed = copy.deepcopy(ep)
    for name in ("short", "moment", "state"):
        changed[name][5:] += 123.
    changed["actions"][4:] += 321.
    changed_packet = label_contexts(core, None, SimpleNamespace(fetch=lambda _: changed), [row],
        seed=5, coalitions=2, noise_samples=1,
        flow_fn=lambda head, episode, query, fused, **kw: {"loss": fused.square().mean()},
        teacher_snapshot_version="checkpoint:stage2-teacher-000007")
    assert [target["features"] for target in context["targets"]] == [
        target["features"] for target in changed_packet["contexts"][0]["targets"]]
    assert all(torch.equal(value, core.state_dict()[name]) for name, value in state_before.items())


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in sorted(globals().items())
                              if name.startswith("test_") and callable(fn))
