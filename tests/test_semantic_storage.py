"""CPU tests: causal storage, exact budget, deterministic routing, gradients."""
from dataclasses import asdict

import unittest
from unittest.mock import patch
import torch

from run_scripts.robomme.semantic_memory_storage import (
    FIFO_OPERATION_ID, KEEP_OPERATION_ID, StorageConfig, StorageManager, StorageOperation,
)


def manager(*, capacity=3, merge=False, **kwargs):
    return StorageManager(StorageConfig(capacity_events=capacity, num_tokens=2,
        dim=8, hidden_dim=12, num_heads=2, enable_merge=merge, merge_verified=merge, **kwargs))


def event(value, *, grad=False):
    return torch.full((1, 2, 8), float(value), requires_grad=grad)


def test_fill_then_exact_fifo_at_initialization_no_random_consumption():
    storage = manager()
    policy = storage.make_policy()
    before = torch.random.get_rng_state().clone()
    bank, expected = None, None
    for t in range(7):
        candidate = event(t)
        bank, metrics = policy(bank, candidate, event_index=t, frame=t*16, is_demo=t < 2)
        expected = candidate if expected is None else torch.cat((expected, candidate), 1)[:, -6:]
        assert torch.equal(bank, expected)
        assert metrics["write_rate"] == 1
        assert metrics["writer_merge"] == 0
    assert torch.equal(before, torch.random.get_rng_state())
    assert storage.choose(bank, event(8)).id == FIFO_OPERATION_ID


def test_legal_operations_and_chronological_arbitrary_replacement():
    storage = manager()
    candidate = event(3)
    assert [op.id for op in storage.operations(None, candidate)] == ["append"]
    bank = torch.cat([event(i) for i in range(3)], 1)
    assert [op.id for op in storage.operations(bank, candidate)] == ["keep", "replace:0", "replace:1", "replace:2"]
    actual = storage.apply(bank, candidate, StorageOperation("replace", 1))
    assert torch.equal(actual, torch.cat((event(0), event(2), candidate), 1))
    assert storage.apply(bank, candidate, StorageOperation("keep")) is bank
    assert storage.fifo_operation(bank, candidate).id == FIFO_OPERATION_ID
    assert storage.fifo_operation(None, candidate).id == "append"


def test_underfull_always_appends_even_if_network_broken():
    storage = manager()
    with torch.no_grad():
        storage.network[-1].weight.fill_(float("nan"))
    actual, _ = storage.make_policy()(None, event(1))
    assert torch.equal(actual, event(1))
    with unittest.TestCase().assertRaisesRegex(FloatingPointError, "scores"):
        storage.choose(torch.cat([event(i) for i in range(3)], 1), event(4))


def test_policy_gradients_survive_insert_and_replace_but_not_argmax():
    storage = manager(capacity=2)
    first, second, third = [event(i, grad=True) for i in range(3)]
    policy = storage.make_policy()
    bank, _ = policy(None, first)
    bank, _ = policy(bank, second)
    bank, _ = policy(bank, third)  # FIFO drops first, not second or third.
    bank.square().sum().backward()
    assert first.grad is not None and torch.equal(first.grad, torch.zeros_like(first))
    assert second.grad is not None and torch.count_nonzero(second.grad) == second.numel()
    assert third.grad is not None and torch.count_nonzero(third.grad) == third.numel()
    assert all(p.grad is None for p in storage.network.parameters())


def test_controller_can_train_separately_with_detached_encoded_inputs():
    torch.manual_seed(3)
    storage = manager()
    bank, candidate = torch.randn(1, 6, 8, requires_grad=True), torch.randn(1, 2, 8, requires_grad=True)
    scores = storage.scores(bank.detach(), candidate.detach())
    loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), torch.tensor([2]))
    loss.backward()
    assert storage.network[-1].weight.grad.abs().sum() > 0
    assert bank.grad is None and candidate.grad is None


def test_merge_guard_defaults_and_repeated_content_not_automatically_merged():
    with unittest.TestCase().assertRaisesRegex(ValueError, "merge_verified"):
        StorageConfig(enable_merge=True)
    with unittest.TestCase().assertRaisesRegex(ValueError, "at least two"):
        StorageConfig(capacity_events=1, enable_merge=True, merge_verified=True)
    regular = manager()
    assert regular.merger is None
    bank = event(1).repeat(1, 3, 1)
    with unittest.TestCase().assertRaisesRegex(ValueError, "Illegal operation"):
        regular.apply(bank, event(1), StorageOperation("merge", 0))
    enabled = manager(merge=True)
    assert len(enabled.operations(bank, event(1))) == 6
    assert enabled.choose(bank, event(1)).id == FIFO_OPERATION_ID
    result, metrics = enabled.make_policy()(bank, event(1))
    assert torch.equal(result, bank)
    assert metrics["writer_merge"] == 0


def test_merge_preserves_budget_order_and_gradients():
    torch.manual_seed(9)
    storage = manager(merge=True)
    events = [torch.randn(1, 2, 8, requires_grad=True) for _ in range(4)]
    bank = torch.cat(events[:3], 1)
    actual = storage.apply(bank, events[3], StorageOperation("merge", 0))
    assert actual.shape == bank.shape
    assert torch.equal(actual[:, 2:4], events[2])
    assert torch.equal(actual[:, 4:], events[3])
    actual.square().sum().backward()
    assert all(e.grad is not None and e.grad.abs().sum() > 0 for e in events)
    assert storage.merger.attention.in_proj_weight.grad.abs().sum() > 0
    assert storage.merger.event_positions.grad.abs().sum() > 0
    a, b = events[:2]
    assert not torch.allclose(storage.merger(a, b), storage.merger(b, a))


def test_policy_merge_application_is_not_under_no_grad():
    storage = manager(merge=True)
    # Force routing without relying on random scores; the scorer's selected
    # operation is discrete and the merger still receives the future gradient.
    ops = storage.operations(torch.zeros(1, 6, 8), event(1))
    merge_index = next(i for i, op in enumerate(ops) if op.id == "merge:1")
    bank = torch.randn(1, 6, 8, requires_grad=True)
    candidate = torch.randn(1, 2, 8, requires_grad=True)
    with patch.object(storage, "_choice_index", lambda operations, scores: merge_index):
        actual, metrics = storage.make_policy()(bank, candidate)
    actual.square().sum().backward()
    assert metrics["writer_merge"] == 1
    assert bank.grad.abs().sum() > 0 and candidate.grad.abs().sum() > 0
    assert storage.merger.attention.in_proj_weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in storage.network.parameters())


def test_callback_stateless_and_metadata_cannot_change_storage():
    storage = manager()
    policy = storage.make_policy()
    original = {k: v.clone() for k, v in storage.state_dict().items()}
    bank, candidate = torch.randn(1, 6, 8), torch.randn(1, 2, 8)
    first = policy(bank, candidate, event_index=3, frame=48, is_demo=False)[0]
    second = policy(bank, candidate, event_index=999, frame=999999, is_demo=True)[0]
    assert torch.equal(first, second)
    policy(None, candidate)  # new session has only the new event, not prior bank
    assert torch.equal(policy(None, candidate)[0], candidate)
    for key, value in storage.state_dict().items():
        assert torch.equal(value, original[key])


def test_shortlist_always_preserves_controls_and_stable_order():
    storage = manager(capacity=5, merge=True)
    bank, candidate = torch.randn(1, 10, 8), torch.randn(1, 2, 8)
    shortlist = storage.shortlist(bank, candidate, max_operations=3)
    assert [op.id for op in shortlist] == [KEEP_OPERATION_ID, FIFO_OPERATION_ID, "replace:1"]
    assert storage.shortlist(None, candidate, max_operations=2) == (StorageOperation("append"),)
    with unittest.TestCase().assertRaisesRegex(ValueError, ">= 2"):
        storage.shortlist(bank, candidate, max_operations=1)


def test_reject_invalid_candidate():
    for bad in (torch.zeros(1, 1, 8), torch.zeros(2, 2, 8), torch.zeros(1, 2, 7),
                torch.zeros(1, 2, 8, dtype=torch.long)):
        with unittest.TestCase().assertRaisesRegex(ValueError, "candidate"):
            manager().operations(None, bad)


def test_reject_partial_overfull_mixed_nonfinite_and_illegal_operations():
    storage = manager()
    for bank in (torch.zeros(1, 3, 8), torch.zeros(1, 8, 8)):
        with unittest.TestCase().assertRaisesRegex(ValueError, "complete events"):
            storage.operations(bank, event(1))
    with unittest.TestCase().assertRaisesRegex(ValueError, "dtype/device"):
        storage.operations(torch.zeros(1, 6, 8, dtype=torch.bfloat16), event(1))
    with unittest.TestCase().assertRaisesRegex(FloatingPointError, "Nonfinite"):
        storage.operations(None, event(float("nan")))
    with unittest.TestCase().assertRaisesRegex(ValueError, "Illegal operation"):
        storage.apply(None, event(1), StorageOperation("keep"))
    with unittest.TestCase().assertRaisesRegex(ValueError, "Illegal operation"):
        storage.apply(torch.zeros(1, 6, 8), event(1), StorageOperation("replace", 4))
    with unittest.TestCase().assertRaisesRegex(ValueError, "Unsupported"):
        StorageOperation("delete")


def test_config_roundtrip_and_state_dict_reproducible():
    storage = manager(merge=True)
    with torch.no_grad():
        storage.network[-1].weight.normal_(0, 0.1)
    copy = StorageManager(StorageConfig(**asdict(storage.config)))
    copy.load_state_dict(storage.state_dict(), strict=True)
    bank, candidate = torch.randn(1, 6, 8), torch.randn(1, 2, 8)
    assert torch.equal(storage.scores(bank, candidate), copy.scores(bank, candidate))
    assert torch.equal(storage.make_policy()(bank, candidate)[0], copy.make_policy()(bank, candidate)[0])


def test_real_core_callback_replay_online_parity_and_historical_gradients():
    from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18, RepresentationMemoryV18
    torch.manual_seed(14)
    core = RepresentationMemoryV18(RepresentationConfigV18(feature_dim=16, state_dim=3,
        num_short_tokens=2, hidden_dim=8, num_heads=2, short_window=3, capacity_events=3))
    with torch.no_grad():
        core.memory.fusion_projection.weight.normal_(std=0.1)
    storage = manager()
    policy = storage.make_policy()
    ep = {"short": torch.randn(6, 2, 16, requires_grad=True),
          "moment": torch.randn(6, 2, 16), "state": torch.randn(6, 3),
          "frames": torch.arange(6) * 16, "is_demo": torch.arange(6) < 2}
    replay = core.replay(ep, 4, write_policy=policy, activation_checkpointing=True)
    no_callback = core.replay(ep, 4, activation_checkpointing=True)
    torch.testing.assert_close(replay["bank"], no_callback["bank"], rtol=0, atol=0)
    torch.testing.assert_close(replay["fused"], no_callback["fused"], rtol=0, atol=0)
    bank = None
    for i in range(5):
        online = core.step(ep["short"][i:i+1], ep["moment"][i:i+1], ep["state"][i:i+1],
                           ep["frames"][i:i+1], ep["is_demo"][i:i+1], bank=bank,
                           write_policy=policy, event_index=i)
        bank = online["bank"]
    torch.testing.assert_close(replay["fused"], online["fused"], rtol=2e-6, atol=1e-6)
    replay["fused"].square().mean().backward()
    assert ep["short"].grad[1:4].abs().sum() > 0
    assert ep["short"].grad[5:].abs().sum() == 0
    assert all(p.grad is None for p in storage.parameters())


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in sorted(globals().items())
                              if name.startswith("test_") and callable(fn))


if __name__ == "__main__":
    unittest.main()
