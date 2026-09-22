"""Synthetic CPU teacher tests; no checkpoint, simulator or GPU is loaded."""
from types import SimpleNamespace
import unittest

import torch

from run_scripts.robomme.cvom_admission_teacher import (build_context_plan, coalition_indices,
    label_contexts, replacement_pair)


def _episodes():
    data = {eid: {"episode_id": eid, "frames": torch.arange(14),
                  "decision_mask": torch.arange(14) >= 3} for eid in range(4)}
    return SimpleNamespace(fetch=lambda eid: data[eid])


class _Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reference = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.config = SimpleNamespace(hidden_dim=2, num_short_tokens=1, capacity_events=3, short_window=2)
        self.reads = []
        self.eval()

    def encode_prefix(self, ep, count):
        assert not torch.is_grad_enabled()
        values = torch.arange(count, dtype=torch.float32).reshape(count, 1, 1).expand(-1, 1, 2)
        return {"stored": values, "short": values + 100, "query": values + 200}

    def read_from_bank(self, short, encoded, bank):
        assert not torch.is_grad_enabled()
        self.reads.append((short.clone(), encoded.clone(), bank.clone()))
        return bank.mean(1, keepdim=True) + short, {}


def test_context_plan_is_split_fixed_long_range_and_all_episode_coverage():
    cache = SimpleNamespace(manifest={"splits": {"train": [0, 1], "val": [2, 3]}})
    args = (cache, _episodes(), {0: "a", 1: "b", 2: "a", 3: "b"}, 3, 0, 42)
    first = build_context_plan(*args, short_window=2)
    assert first == build_context_plan(*args, short_window=2)
    assert [row["episode_id"] for row in first] == [0, 1]
    for split in ("train", "val"):
        rows = build_context_plan(*args[:4], 7, 42, split=split, short_window=2)
        assert len(rows) == 2  # Requested upper bound cannot duplicate episodes.
        assert len({row["episode_id"] for row in rows}) == 2
        assert abs(sum(row["task"] == "a" for row in rows) - sum(row["task"] == "b" for row in rows)) <= 1
        assert all(row["episode_id"] in cache.manifest["splits"][split] for row in rows)
        assert all(query > row["event"] + 2 for row in rows for query in row["future"])
        assert all(len(row["future"]) == 2 for row in rows)
    with unittest.TestCase().assertRaisesRegex(ValueError, "Only TRAIN"):
        build_context_plan(*args, split="test")


def test_coalition_sampling_preserves_time_fixed_victim_and_token_budget():
    plan = coalition_indices(4, 7, 12)
    assert plan == coalition_indices(4, 7, 12)
    assert plan[0] == [1, 2, 3]
    bank = torch.arange(4.).reshape(1, 4, 1).repeat_interleave(2, dim=1)
    candidate = torch.full((1, 2, 1), 4.)
    for subset in plan:
        assert subset == sorted(set(subset)) and 0 not in subset
        left, right = replacement_pair(bank, candidate, subset, 2)
        assert left.shape == right.shape == (1, 2 * (1 + len(subset)), 1)
        assert left[0, ::2, 0].tolist() == [0.] + subset
        assert right[0, ::2, 0].tolist() == subset + [4.]
    for illegal in ([0], [2, 1], [1, 1], [4]):
        with unittest.TestCase().assertRaisesRegex(ValueError, "distinct increasing"):
            replacement_pair(bank, candidate, illegal, 2)


def test_teacher_paired_noise_equal_budget_and_no_future_bank_leak():
    core, calls = _Core(), []
    def flow(head, ep, query, fused, **kwargs):
        generator = torch.Generator().manual_seed(kwargs["seed"])
        noise = float(torch.rand((), generator=generator))
        calls.append((query, kwargs["seed"], float(fused.mean())))
        return {"loss": fused.mean() * (1 + noise)}
    context = {"episode_id": 0, "event": 3, "future": [6, 9], "task": "a"}
    rng_before = torch.random.get_rng_state().clone()
    packet = label_contexts(core, None, _episodes(), [context], seed=19, flow_fn=flow)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    row = packet["contexts"][0]
    assert len(calls) == row["actual_actor_calls"] == 56
    assert row["logical_actor_calls_per_arm"] == 32
    assert row["bank_event_ids"] == [0, 1, 2]
    assert row["victim_event_id"] == 0 and row["candidate_event_id"] == 3
    for left, right in zip(calls[::2], calls[1::2]):
        assert left[:2] == right[:2]
    # Shared query content but only events <= candidate in every branch bank.
    for short, query, bank in core.reads:
        assert int(short[0, 0, 0]) in (106, 109)
        assert int(query[0, 0, 0]) in (206, 209)
        assert float(bank.max()) <= 3
        assert bank[0, :, 0].tolist() == sorted(bank[0, :, 0].tolist())
    single, coalition = row["labels"]["single"], row["labels"]["coalitional"]
    assert single["gains"][0] == coalition["gains"][0]
    assert single["keep_losses"][0] == coalition["keep_losses"][0]
    assert single["noise_seeds"] == coalition["noise_seeds"]
    assert len({seed for chunk in single["noise_seeds"] for future in chunk for seed in future}) == 16
    assert all(subset == [1, 2] for subset in single["coalition_indices"])
    assert coalition["coalition_indices"][0] == [1, 2]
    assert all(subset != [1, 2] for subset in coalition["coalition_indices"][1:])
    for label in (single, coalition):
        calculated = torch.tensor(label["keep_losses"], dtype=torch.float64) - torch.tensor(label["replace_losses"], dtype=torch.float64)
        assert torch.allclose(calculated, torch.tensor(label["gains"], dtype=torch.float64))
        assert abs(float(calculated.mean()) - label["signed_mean"]) < 1e-12
        assert label["signed_mean"] < 0 and label["noise_std"] > 0
    # Only causal full-bank features are shared by the two target variants.
    assert len(row["features"]) == 12 and row["features"][:2] == [3., 3.]
    assert "NO future or intermediate writes" in packet["settings"]["conditional_bank"]


def test_teacher_is_deterministic_and_order_independent():
    def flow(head, ep, query, fused, **kwargs):
        return {"loss": float(fused.mean()) + kwargs["seed"] % 13}
    a = {"episode_id": 0, "event": 3, "future": [6, 9], "task": "a"}
    b = {"episode_id": 1, "event": 5, "future": [8, 10], "task": "b"}
    first = label_contexts(_Core(), None, _episodes(), [a, b], seed=13, flow_fn=flow)
    second = label_contexts(_Core(), None, _episodes(), [b, a], seed=13, flow_fn=flow)
    assert first["contexts"] == list(reversed(second["contexts"]))


def test_teacher_refuses_trainable_actor_and_recent_or_nondeterministic_queries():
    core = _Core()
    row = {"episode_id": 0, "event": 3, "future": [6], "task": "a"}
    unused = lambda *args, **kwargs: {"loss": 1.}
    core.requires_grad_(True)
    with unittest.TestCase().assertRaisesRegex(ValueError, "explicitly frozen"):
        label_contexts(core, None, _episodes(), [row], seed=1, flow_fn=unused)
    core.requires_grad_(False).train()
    with unittest.TestCase().assertRaisesRegex(ValueError, "explicitly frozen"):
        label_contexts(core, None, _episodes(), [row], seed=1, flow_fn=unused)
    core.eval()
    for bad in ({**row, "future": [5]}, {**row, "future": [9, 6]},
                {**row, "event": 2}, {**row, "future": [6, 6]}):
        with unittest.TestCase().assertRaises(ValueError):
            label_contexts(core, None, _episodes(), [bad], seed=1, flow_fn=unused)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in sorted(globals().items())
                              if name.startswith("test_") and callable(fn))


if __name__ == "__main__":
    unittest.main()
