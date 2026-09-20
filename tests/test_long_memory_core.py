"""CPU-only gradient/causality contracts; no model downloads or simulator needed."""

import unittest
import torch

from gr00t.long_memory.core import EpisodicMemory, MemoryConfig, novelty


def make_memory(**overrides):
    options = dict(feature_dim=12, state_dim=3, action_dim=4,
                   hidden_dim=10, key_dim=5, value_dim=7, capacity=4, min_fill=1)
    options.update(overrides)
    return EpisodicMemory(MemoryConfig(**options))


def make_events(count=3, steps=4):
    return {
        "short": torch.randn(count, 2, 12),
        "pre_moment": torch.randn(count, 2, 12),
        "post_moment": torch.randn(count, 2, 12),
        "state": torch.randn(count, 3),
        "next_state": torch.randn(count, 3),
        "actions": torch.randn(count, steps, 4),
        "action_mask": torch.ones(count, steps, dtype=torch.bool),
        "valid": torch.ones(count, dtype=torch.bool),
    }


def grad_sum(module):
    return sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)


def test_shapes_invalid_events_and_reconstruction():
    torch.manual_seed(4)
    model = make_memory()
    events = make_events()
    events["valid"][-1] = False
    # Invalid padding must not turn a batch's loss into NaN.
    for key, tensor in events.items():
        if tensor.is_floating_point():
            tensor[-1] = float("nan")
    encoded = model.encode_events(events)
    assert encoded["event"].shape == (3, 10)
    assert encoded["keys"].shape == (3, 5)
    assert encoded["values"].shape == (3, 7)
    for key in ("event", "keys", "values"):
        assert torch.isfinite(encoded[key]).all()
        assert torch.equal(encoded[key][-1], torch.zeros_like(encoded[key][-1]))
    rec = model.reconstruct(encoded["event"])
    assert rec["delta_moment"].shape == (3, 12)
    assert rec["delta_state"].shape == (3, 3)
    rec["delta_state"].square().sum().backward()
    assert grad_sum(model.event_encoder) > 0


def check_empty_bank_identity(slots):
    model = make_memory(residual_init=0.1)
    with torch.no_grad():
        model.fusion[-1].bias.fill_(2.0)
    short, state = torch.randn(2, 2, 12), torch.randn(2, 3)
    out = model.read(short, state, torch.full((2, slots, 5), float("nan")),
                     torch.full((2, slots, 7), float("nan")),
                     torch.zeros(2, slots, dtype=torch.bool))
    assert torch.equal(out["fused_short"], short)
    assert torch.equal(out["read"], torch.zeros(2, 7))
    assert torch.equal(out["weights"][:, -1], torch.ones(2))
    assert torch.isfinite(out["weights"]).all()


def test_empty_bank_without_slots():
    check_empty_bank_identity(0)


def test_empty_bank_with_all_slots_masked():
    check_empty_bank_identity(3)


def test_zero_init_then_action_gradient_reaches_past_event_encoder():
    torch.manual_seed(8)
    model = make_memory()
    events = make_events(count=3)
    short, state = torch.randn(1, 2, 12), torch.randn(1, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    target = torch.randn_like(short)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    def forward():
        encoded = model.encode_events(events)
        return model.read(short, state, encoded["keys"][None], encoded["values"][None], mask)

    output = forward()
    assert torch.equal(output["fused_short"], short)
    assert output["weights"].shape == (1, 4)
    assert torch.allclose(output["weights"].sum(dim=1), torch.ones(1))
    (output["fused_short"] - target).square().mean().backward()
    assert grad_sum(model.fusion[-1]) > 0
    assert grad_sum(model.event_encoder) == 0
    assert grad_sum(model.query) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    (forward()["fused_short"] - target).square().mean().backward()
    for module in (model.event_encoder, model.action_encoder, model.query, model.key, model.value):
        assert grad_sum(module) > 0


def test_utility_and_write_supervision_detach_representation_inputs():
    model = make_memory()
    event = torch.randn(2, 10, requires_grad=True)
    short = torch.randn(2, 2, 12, requires_grad=True)
    read = torch.randn(2, 7, requires_grad=True)
    prediction = model.utility(event, short, read)
    assert prediction.shape == (2,)
    assert (prediction > 0).all()
    prediction.sum().backward()
    assert grad_sum(model.utility_head) > 0
    assert event.grad is None and short.grad is None and read.grad is None
    model.zero_grad(set_to_none=True)
    prediction = model.utility(event, short, read)
    newness = torch.randn(2, requires_grad=True)
    logits = model.write_logits(prediction, newness)
    assert logits.shape == (2,)
    logits.sum().backward()
    assert grad_sum(model.write_head) > 0
    assert grad_sum(model.utility_head) == 0
    assert newness.grad is None


def test_action_encoder_uses_only_valid_prefix_and_handles_passive_events():
    model = make_memory()
    actions = torch.randn(3, 4, 4)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    first = model._encode_actions(actions, mask)
    changed = actions.clone()
    changed[~mask] = float("nan")
    second = model._encode_actions(changed, mask)
    assert torch.equal(first, second)
    assert torch.equal(first[1], torch.zeros(10))
    assert torch.isfinite(first).all()
    with unittest.TestCase().assertRaisesRegex(ValueError, "contiguous valid prefix"):
        model._encode_actions(actions, torch.tensor([[1, 0, 1, 0]] * 3, dtype=torch.bool))


def test_novelty_empty_identical_and_opposite_keys():
    candidate = torch.tensor([[1.0, 0.0]] * 3)
    banks = torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]], [[0.0, 1.0]]])
    mask = torch.tensor([[True], [True], [False]])
    assert torch.equal(novelty(candidate, banks, mask), torch.tensor([0.0, 2.0, 1.0]))
    assert torch.equal(novelty(candidate, banks[:, :0], mask[:, :0]), torch.ones(3))


def test_invalid_shapes_are_rejected_and_empty_event_batch_supported():
    model = make_memory()
    encoded = model.encode_events(make_events(count=0))
    assert encoded["event"].shape == (0, 10)
    with unittest.TestCase().assertRaisesRegex(ValueError, "feature_dim must be positive"):
        MemoryConfig(feature_dim=0, state_dim=3, action_dim=4)
    with unittest.TestCase().assertRaisesRegex(ValueError, "min_fill"):
        make_memory(min_fill=5)
    malformed = make_events()
    malformed["actions"] = malformed["actions"][:1]
    with unittest.TestCase().assertRaisesRegex(ValueError, "matching event count"):
        model.encode_events(malformed)


def test_bank_permutation_preserves_content_read_and_padded_slots_are_ignored():
    torch.manual_seed(9)
    model = make_memory(residual_init=0.01)
    short, state = torch.randn(2, 2, 12), torch.randn(2, 3)
    keys, values = torch.randn(2, 3, 5), torch.randn(2, 3, 7)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    result = model.read(short, state, keys, values, mask)
    order = torch.tensor([2, 0, 1])
    permuted = model.read(short, state, keys[:, order], values[:, order], mask[:, order])
    assert torch.allclose(result["read"], permuted["read"], atol=1e-6)
    assert torch.equal(result["fused_short"][1], short[1])
    assert result["read_norm"].shape == (2,)
    assert result["gate_mean"].shape == (2,)
    assert result["residual_norm"].shape == (2,)


def load_tests(loader, tests, pattern):
    """Expose plain-function tests to stdlib unittest; pytest is not required."""
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for name, test in sorted(globals().items())
        if name.startswith("test_") and callable(test)
    )


if __name__ == "__main__":
    unittest.main()
