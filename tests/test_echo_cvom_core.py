"""CPU contracts for causal ECHO effects, provenance, and all-slot storage."""
from dataclasses import replace
from types import MethodType
import unittest

import torch

from run_scripts.robomme.echo_cvom_core import (EchoBank, EchoConfig, EchoMemoryV1,
    CompletedEffectEncoder, SlotUtilityMLP)
from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18, RepresentationMemoryV18


def _configs(**changes):
    representation = RepresentationConfigV18(feature_dim=8, state_dim=3, num_short_tokens=2,
        hidden_dim=4, num_heads=2, capacity_events=3, short_window=2)
    echo = EchoConfig(capacity_events=3, min_fill=2, utility_hidden=8, effect_hidden=8, action_dim=4,
                      **changes)
    return representation, echo


def _core(**changes):
    return EchoMemoryV1(*_configs(**changes))


def _episode(n=9):
    torch.manual_seed(91)
    demo = torch.arange(n) < 2
    mask = torch.ones(n-1, 2, dtype=torch.bool)
    mask[:2] = False
    return {"short": torch.randn(n, 2, 8), "moment": torch.randn(n, 2, 8),
        "state": torch.randn(n, 3), "frames": torch.arange(n)*16, "is_demo": demo,
        "actions": torch.randn(n-1, 2, 4), "action_mask": mask,
        "transition_valid": torch.ones(n-1, dtype=torch.bool)}


def _event(value):
    return torch.full((1, 2, 4), float(value))


def _bank():
    bank = EchoBank(torch.empty(1, 0, 4))
    for i in range(3):
        bank = bank.append(_event(i+1), i*16, i == 0, i)
    return bank


def _constant_scores(manager, utilities, probability=1.):
    def forward(self, features):
        values = features.new_tensor(utilities)
        assert len(values) == len(features)
        return {"utility": values, "logit": torch.zeros_like(values),
                "write_probability": torch.full_like(values, probability)}
    manager.forward_features = MethodType(forward, manager)


def test_config_rejects_unsupported_or_unsafe_modes():
    rep, config = _configs()
    assert EchoConfig(**config.to_dict()) == config
    for values in ({"min_fill": 4}, {"merge_threshold": 1.1}, {"merge_max_gap": -1},
                   {"action_dim": 0}, {"recency_weight": float("nan")}, {"write_threshold": 2}):
        with unittest.TestCase().assertRaises(ValueError):
            replace(config, **values)
    with unittest.TestCase().assertRaisesRegex(ValueError, "only the original short"):
        EchoMemoryV1(replace(rep, representation="moment"), config)
    with unittest.TestCase().assertRaisesRegex(ValueError, "capacity"):
        EchoMemoryV1(rep, replace(config, capacity_events=4))


def test_zero_effect_exactly_reproduces_parent_fifo_and_parent_loading():
    torch.manual_seed(3)
    rep, config = _configs()
    parent = RepresentationMemoryV18(rep)
    with torch.no_grad():
        parent.memory.fusion_projection.weight.normal_(0, .01)
    echo = EchoMemoryV1(rep, config)
    effect_before = {k: v.clone() for k, v in echo.effect_adapter.state_dict().items()}
    echo.initialize_from_parent_delta(parent.delta_state_dict())
    assert all(torch.equal(v, effect_before[k]) for k, v in echo.effect_adapter.state_dict().items())
    episode = _episode()
    original, new = parent.encode_prefix(episode, 8), echo.encode_prefix(episode, 8)
    assert torch.equal(original["query"], new["query"])
    assert torch.equal(original["stored"], new["stored"])
    assert not bool(new["effect"].any())
    for decision in (0, 1, 4, 7):
        before, after = parent.replay(episode, decision), echo.replay(episode, decision, write_mode="fifo")
        assert torch.equal(before["bank"], after["bank"])
        assert torch.equal(before["fused"], after["fused"])
    with unittest.TestCase().assertRaises(ValueError):
        echo.initialize_from_parent_delta(echo.delta_state_dict())


def test_echo_delta_roundtrip_strict_inventory_and_no_partial_bad_load():
    core, restored = _core(), _core()
    state = core.delta_state_dict()
    assert any(k.startswith("effect_adapter.") for k in state)
    assert any(k.startswith("manager.") for k in state)
    restored.load_delta_state_dict(state)
    assert all(torch.equal(v, restored.delta_state_dict()[k]) for k, v in state.items())
    bad = dict(state)
    bad["effect_adapter.output.bias"] = torch.full_like(state["effect_adapter.output.bias"], float("nan"))
    with unittest.TestCase().assertRaises(ValueError):
        restored.load_delta_state_dict(bad)
    assert all(torch.equal(v, restored.delta_state_dict()[k]) for k, v in state.items())


def test_completed_effect_is_past_only_and_query_is_unchanged():
    core, episode = _core(), _episode()
    with torch.no_grad():
        core.effect_adapter.output.weight.normal_(0, .1)
        core.effect_adapter.output.bias.fill_(.3)
    original = core.encode_prefix(episode, 6)
    assert not bool(original["effect"][:3].any())  # demo and demo->execution boundary
    assert bool(original["effect"][3:].abs().sum())
    changed = {k: v.clone() for k, v in episode.items()}
    changed["actions"][5:] = float("nan")
    changed["moment"][6:] = float("nan")
    changed["state"][6:] = float("nan")
    changed["short"][6:] = float("nan")
    # The encoder has no reason even to fetch future action targets.
    class NoTargets(dict):
        def __getitem__(self, key):
            if key in ("targets", "target_mask"):
                raise AssertionError("GT future target leak")
            return super().__getitem__(key)
    second = core.encode_prefix(NoTargets(changed), 6)
    assert torch.equal(original["stored"], second["stored"])
    changed = {k: v.clone() for k, v in episode.items()}
    changed["actions"][3] += 20
    with_action_change = core.encode_prefix(changed, 6)
    assert torch.equal(original["query"], with_action_change["query"])
    assert torch.equal(original["stored"][:4], with_action_change["stored"][:4])
    assert not torch.equal(original["stored"][4], with_action_change["stored"][4])
    assert torch.equal(original["stored"][5], with_action_change["stored"][5])


def test_effect_masks_invalid_transitions_and_ignores_padded_controls():
    core, episode = _core(), _episode()
    with torch.no_grad():
        core.effect_adapter.output.bias.fill_(1.)
    args = list(core._effect_prefix_inputs(episode, 6))
    args[-1] = torch.tensor([False, False, True, False, True])
    args[-2] = torch.zeros(5, 2, dtype=torch.bool)
    args[-2][2:, 0] = True
    args[-3][:, 1] = float("nan")
    effect = core.encode_completed_effect(*args)
    assert not bool(effect[[0, 1, 3]].any())
    assert bool((effect[[2, 4]] == 1).all())
    assert bool(torch.isfinite(core.effect_adapter.reconstruction_loss(*args)))


def test_effect_bf16_boundary_same_online_or_cached_and_aux_learns():
    core, episode = _core(), _episode()
    with torch.no_grad():
        core.effect_adapter.output.weight.normal_(0, .1)
    args = list(core._effect_prefix_inputs(episode, 6))
    quantized = list(args)
    quantized[0], quantized[1] = args[0].bfloat16(), args[1].bfloat16()
    assert torch.equal(core.encode_completed_effect(*args), core.encode_completed_effect(*quantized))
    loss = core.effect_loss(episode, 6)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert core.effect_adapter.encoder[1].weight.grad.abs().sum() > 0
    assert core.effect_adapter.decoder.weight.grad.abs().sum() > 0


def test_fifo_keeps_whole_events_provenance_and_demo_visuals():
    core, bank = _core(), None
    for i in range(5):
        bank = core.initial_state() if bank is None else bank
        bank, metrics = core.manager.update(bank, _event(i), _event(i), i*16, i < 2, i, mode="fifo")
        assert bank.n_events == min(i+1, 3)
        assert bank.event_ids == tuple((j,) for j in range(max(0, i-2), i+1))
        assert not float(metrics["writer_merge"])
    assert torch.equal(bank.tokens, torch.cat([_event(i) for i in (2, 3, 4)], dim=1))
    demo = core.replay(_episode(), 2, write_mode="fifo")["bank_state"]
    assert demo.is_demo == (True, True) and demo.n_events == 2


def test_learned_min_fill_then_gate_before_full_and_non_oldest_eviction():
    core = _core(recency_weight=0., diversity_weight=0.)
    with torch.no_grad():
        core.manager.write_head.bias.fill_(-20)
    bank = core.initial_state()
    for i in range(3):
        bank, metrics = core.manager.update(bank, _event(i), _event(i), i*16, False, i)
    assert bank.n_events == 2 and bank.event_ids == ((0,), (1,))
    assert float(metrics["writer_keep"]) == 1  # gate is active before capacity
    bank = _bank()
    _constant_scores(core.manager, [3., .1, 2., 4.])
    result, metrics = core.manager.update(bank, _event(4), _event(4), 48, False, 3)
    assert result.event_ids == ((0,), (2,), (3,))
    assert float(metrics["writer_victim"]) == 1  # not FIFO's oldest
    assert float(metrics["writer_replace"]) == 1
    assert bank.event_ids == ((0,), (1,), (2,))  # input immutable
    _constant_scores(core.manager, [2., 2., 2., 2.])
    result, metrics = core.manager.update(bank, _event(4), _event(4), 48, False, 3)
    assert result is bank and float(metrics["writer_keep"]) == 1  # conservative tie


def test_scoring_is_all_slots_causal_detached_and_signed_not_clamped():
    core, bank = _core(), _bank()
    candidate, query = _event(4).requires_grad_(), _event(5).requires_grad_()
    result = core.manager.score(bank, candidate, query, 64, False)
    assert result["features"].shape == (4, 20)
    assert not result["features"].requires_grad
    # Critic receives all three slots and the candidate, never future target.
    assert torch.equal(result["features"][:, :4], torch.stack([torch.full((4,), float(i)) for i in (1, 2, 3, 4)]))
    assert torch.equal(result["features"][:, 4:8], torch.full((4, 4), 5.))
    with torch.no_grad():
        core.manager.utility_head.bias.fill_(-3)
    result = core.manager.score(bank, candidate, query, 64, False)
    assert bool((result["utility"] == -3).all())
    assert bool((result["retention"] >= 0).all())
    (result["utility"].sum()+result["logit"].sum()).backward()
    assert candidate.grad is None and query.grad is None
    assert core.manager.utility_head.bias.grad is not None


def test_selected_storage_tensors_keep_gradients_but_hard_scores_do_not():
    core = _core(recency_weight=0., diversity_weight=0.)
    candidate = _event(4).requires_grad_()
    old = _bank()
    old_tokens = old.tokens.clone().requires_grad_()
    bank = EchoBank(old_tokens, old.first_frames, old.last_frames, old.is_demo, old.counts, old.event_ids)
    _constant_scores(core.manager, [3., .1, 2., 4.])
    result, _ = core.manager.update(bank, candidate, _event(8), 48, False, 3)
    result.tokens.sum().backward()
    assert bool((old_tokens.grad[:, :2] == 1).all())
    assert not bool(old_tokens.grad[:, 2:4].any())
    assert bool((old_tokens.grad[:, 4:] == 1).all())
    assert bool((candidate.grad == 1).all())
    assert all(p.grad is None for p in core.manager.parameters())


def test_optional_merge_is_bounded_same_phase_adjacent_and_preserves_sources():
    core = _core(recency_weight=0., diversity_weight=0., merge_threshold=.99)
    bank = _bank()
    _constant_scores(core.manager, [3., .1, 2., 4.])
    result, metrics = core.manager.update(bank, _event(4), _event(4), 48, False, 3)
    # Pair (0,1) cannot merge across demo/execution; (1,2) wins deterministic tie.
    assert result.event_ids == ((0,), (1, 2), (3,))
    assert result.counts == (1, 2, 1)
    assert result.first_frames == (0, 16, 48) and result.last_frames == (0, 32, 48)
    assert torch.equal(result.tokens[:, 2:4], _event(2.5))
    assert float(metrics["writer_merge"]) == 1 and float(metrics["writer_victim"]) == -1
    # Existing count-two memory cannot be merged again under the bounded rule.
    assert core.manager._merge_index(result) is None
    no_merge = _core(recency_weight=0., diversity_weight=0.)
    _constant_scores(no_merge.manager, [3., .1, 2., 4.])
    plain, metrics = no_merge.manager.update(bank, _event(4), _event(4), 48, False, 3)
    assert plain.counts == (1, 1, 1) and float(metrics["writer_merge"]) == 0


def test_replay_action_gradient_reaches_historical_effect_residual():
    core, episode = _core(), _episode()
    with torch.no_grad():
        core.memory.fusion_projection.weight.normal_(0, .05)
    out = core.replay(episode, 7)
    out["fused"].square().mean().backward()
    assert core.effect_adapter.output.weight.grad is not None
    assert core.effect_adapter.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in core.manager.parameters())


def test_online_read_before_write_matches_cached_replay_and_reset():
    core, episode = _core(), _episode()
    with torch.no_grad():
        core.memory.fusion_projection.weight.normal_(0, .05)
        core.effect_adapter.output.weight.normal_(0, .02)
    state = None
    for i in range(7):
        kwargs = {}
        if i:
            kwargs = {"previous_moment": episode["moment"][i-1:i], "previous_state": episode["state"][i-1:i],
                "previous_is_demo": episode["is_demo"][i-1:i], "completed_actions": episode["actions"][i-1:i],
                "completed_action_mask": episode["action_mask"][i-1:i], "transition_valid": episode["transition_valid"][i-1:i]}
        out = core.step(episode["short"][i:i+1], episode["moment"][i:i+1], episode["state"][i:i+1],
            episode["frames"][i:i+1], episode["is_demo"][i:i+1], bank_state=state,
            event_index=i, write_mode="fifo", **kwargs)
        reference = core.replay(episode, i, write_mode="fifo")
        assert torch.allclose(out["fused"], reference["fused"], atol=2e-6, rtol=1e-5)
        assert out["bank_state"].event_ids[-1] == (i,)
        assert reference["bank_state"].event_ids == (() if i == 0 else tuple((j,) for j in range(max(0, i-3), i)))
        state = out["bank_state"].detach()
    assert core.initial_state().n_events == 0 and state.n_events == 3
    with unittest.TestCase().assertRaises(ValueError):
        core.step(episode["short"][:1], episode["moment"][:1], episode["state"][:1],
            episode["frames"][:1], episode["is_demo"][:1], previous_state=episode["state"][:1], event_index=0)


def test_bank_validation_select_without_and_no_reorder():
    bank = _bank()
    assert bank.select([0, 2]).event_ids == ((0,), (2,))
    assert bank.without([1]).event_ids == ((0,), (2,))
    assert bank.select([]).tokens.shape == (1, 0, 4)
    for ids in ([2, 0], [1, 1], [-1], [3]):
        with unittest.TestCase().assertRaises(ValueError):
            bank.select(ids)
    with unittest.TestCase().assertRaises(ValueError):
        bank.append(_event(4), 16, False, 4)
    with unittest.TestCase().assertRaises(ValueError):
        bank.append(_event(4), 48, True, 4)
    with unittest.TestCase().assertRaises(ValueError):
        bank.append(_event(4), 48, False, 1)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in sorted(globals().items())
                              if name.startswith("test_") and callable(fn))


if __name__ == "__main__":
    unittest.main()
