"""Small CPU contracts for the frozen-actor admission-only experiment."""
from dataclasses import replace
import unittest

import torch
from torch.nn import functional as F

from run_scripts.robomme.cvom_admission_core import AdmissionConfig, CVoMAdmission, admission_features


def _controller():
    return CVoMAdmission(AdmissionConfig(dim=3, num_tokens=2, capacity_events=2, hidden_dim=8))


def test_config_roundtrip_and_invalid_settings():
    config = _controller().config
    assert AdmissionConfig(**config.to_dict()) == config
    for change in ({"capacity_events": 0}, {"num_tokens": True}, {"write_threshold": 1.1},
                   {"utility_margin": -1.}, {"dim": float("nan")}, {"version": "unknown"}):
        with unittest.TestCase().assertRaises(ValueError):
            replace(config, **change)


def test_zero_heads_are_exact_fifo_without_rng_or_input_mutation():
    controller, bank = _controller(), None
    policy = controller.make_policy()
    state = torch.random.get_rng_state().clone()
    for index in range(6):
        candidate = torch.full((1, 2, 3), float(index))
        previous = bank.clone() if bank is not None else candidate[:, :0]
        result, metrics = policy(bank, candidate, event_index=index, frame=index * 16, is_demo=index < 3)
        expected = torch.cat((previous, candidate), dim=1)[:, -4:]
        assert torch.equal(result, expected)
        if bank is not None:
            assert torch.equal(bank, previous)
        assert float(metrics["writer_insert"]) == 1
        assert float(metrics["writer_append"]) == (index < 2)
        assert float(metrics["writer_replace"]) == (index >= 2)
        assert float(metrics["writer_keep"]) == 0
        assert float(metrics["writer_probability"]) == .5
        bank = result
    assert torch.equal(state, torch.random.get_rng_state())


def test_negative_gate_or_utility_fills_then_keeps_whole_events():
    for head in ("utility_head", "write_head"):
        controller = _controller()
        with torch.no_grad():
            getattr(controller, head).bias.fill_(-1)
        policy, bank = controller.make_policy(), None
        for index in range(5):
            bank, metrics = policy(bank, torch.full((1, 2, 3), float(index)))
        assert torch.equal(bank, torch.cat((torch.zeros(1, 2, 3), torch.ones(1, 2, 3)), dim=1))
        assert float(metrics["writer_keep"]) == 1
        assert float(metrics["writer_insert"]) == 0


def test_causal_features_detached_and_metadata_independent():
    controller = _controller()
    bank, candidate = torch.randn(1, 4, 3, requires_grad=True), torch.randn(1, 2, 3, requires_grad=True)
    features = controller.features(bank, candidate)
    assert features.shape == (1, 16) and not features.requires_grad
    assert torch.equal(features[:, :3], candidate.detach().mean(1))
    assert torch.equal(features[:, 3:6], bank.detach().reshape(1, 2, 2, 3).mean(2).mean(1))
    assert torch.equal(features[:, 6:9], bank[:, :2].detach().mean(1))
    assert float(features[:, -1]) == 1
    policy = controller.make_policy()
    first = policy(bank, candidate, event_index=1, frame=16, is_demo=True)[0]
    second = policy(bank, candidate, event_index=999, frame=999999, is_demo=False)[0]
    assert torch.equal(first, second)


def test_no_gradient_to_actor_content_and_both_heads_learn():
    torch.manual_seed(17)
    controller = _controller()
    x = torch.randn(4, controller.config.feature_dim, requires_grad=True)
    target = torch.tensor([-1., -1., 1., 1.])
    optimizer = torch.optim.Adam(controller.parameters(), lr=.04)
    for _ in range(35):
        prediction = controller.forward_features(x)
        loss = F.smooth_l1_loss(prediction["utility"], target) + F.binary_cross_entropy_with_logits(
            prediction["logit"], (target > 0).float())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert x.grad is None
    prediction = controller.forward_features(x)
    assert bool(((prediction["utility"] > 0) == (target > 0)).all())
    assert bool(((prediction["write_probability"] > .5) == (target > 0)).all())
    restored = CVoMAdmission(controller.config)
    restored.load_state_dict(controller.state_dict())
    assert torch.equal(restored.forward_features(x)["utility"], prediction["utility"])


def test_invalid_bank_operations_and_nonfinite_fail_closed():
    controller = _controller()
    candidate = torch.ones(1, 2, 3)
    full = torch.zeros(1, 4, 3)
    for bank in (torch.zeros(1, 3, 3), torch.zeros(1, 6, 3), torch.zeros(2, 4, 3)):
        with unittest.TestCase().assertRaises(ValueError):
            controller.choose(bank, candidate)
    for bank, operation in ((full, "append"), (full[:, :0], "keep"), (full, "replace:1"), (full, "merge")):
        with unittest.TestCase().assertRaises(ValueError):
            controller.apply(bank, candidate, operation)
    with unittest.TestCase().assertRaises(FloatingPointError):
        admission_features(full, candidate * float("nan"), controller.config)
    with unittest.TestCase().assertRaises(TypeError):
        controller.to(torch.bfloat16).forward_features(torch.ones(1, 16))


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in sorted(globals().items())
                              if name.startswith("test_") and callable(fn))


if __name__ == "__main__":
    unittest.main()
