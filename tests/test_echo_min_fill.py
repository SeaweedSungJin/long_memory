"""CPU evidence that the minimum-fill intervention is isolated and explicit."""
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest
import torch

from run_scripts.robomme.echo_cvom_core import EchoBank, EchoConfig, EchoMemoryV1
from run_scripts.robomme.policy_echo_cvom import EchoPolicyV1
from run_scripts.robomme.policy_echo_min_fill import (
    EchoMinFillPolicy, MIN_FILL_SOURCES, apply_min_fill_override, min_fill_source_identity,
)
from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18
from run_scripts.robomme.serve_echo_min_fill import build_parser
from tests.test_echo_cvom_runtime import actor, calls


def tiny_core():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(271)
        representation = RepresentationConfigV18(feature_dim=8, state_dim=3,
            num_short_tokens=2, hidden_dim=4, num_heads=2, capacity_events=32, short_window=2)
        return EchoMemoryV1(representation, EchoConfig(capacity_events=32, min_fill=4,
            action_dim=4, utility_hidden=8, effect_hidden=8))


def event(index):
    return torch.full((1, 2, 4), float(index + 1))


def test_override_changes_only_config_preserving_parameters_buffers_and_rng():
    core = tiny_core()
    before = {name: value.clone() for name, value in core.state_dict().items()}
    original = asdict(core.echo_config)
    rng = torch.random.get_rng_state().clone()
    report = apply_min_fill_override(core, 32)
    assert report == {"min_fill_override": 32, "original_min_fill": 4, "effective_min_fill": 32}
    assert asdict(core.echo_config) == {**original, "min_fill": 32}
    assert core.echo_config is core.manager.config
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(value, before[name]) for name, value in core.state_dict().items())


@pytest.mark.parametrize("value", [0, -1, 33, True, 4.0, "32"])
def test_invalid_override_does_not_change_config(value):
    core = tiny_core()
    config = core.echo_config
    with pytest.raises(ValueError, match="override"):
        apply_min_fill_override(core, value)
    assert core.echo_config is config and core.manager.config is config


def test_min_fill_32_appends_every_endpoint_before_full_even_with_rejected_admission():
    control, candidate = tiny_core(), tiny_core()
    with torch.no_grad():
        control.manager.write_head.bias.fill_(-20.)
        candidate.manager.write_head.bias.fill_(-20.)
    apply_min_fill_override(candidate, 32)
    left, right = control.initial_state(), candidate.initial_state()
    for index in range(32):
        for core, bank, name in ((control, left, "left"), (candidate, right, "right")):
            result, metrics = core.manager.update(bank, event(index), event(index),
                index * 16, index < 10, index)
            if name == "left":
                left = result
            else:
                right = result
                assert metrics["writer_append"] == 1 and result.n_events == index + 1
    assert left.n_events == 4 and right.n_events == 32
    assert right.event_ids == tuple((i,) for i in range(32))


@pytest.mark.parametrize("admit", [False, True])
def test_same_full_bank_has_exactly_same_learned_decision_after_override(admit):
    control, candidate = tiny_core(), tiny_core()
    with torch.no_grad():
        for core in (control, candidate):
            core.manager.write_head.bias.fill_(20. if admit else -20.)
    apply_min_fill_override(candidate, 32)
    bank = EchoBank(torch.empty(1, 0, 4))
    for index in range(32):
        bank = bank.append(event(index), index * 16, False, index)
    left, lm = control.manager.update(bank, event(32), event(32), 512, False, 32)
    right, rm = candidate.manager.update(bank, event(32), event(32), 512, False, 32)
    assert left.event_ids == right.event_ids and torch.equal(left.tokens, right.tokens)
    assert lm.keys() == rm.keys() and all(torch.equal(lm[k], rm[k]) for k in lm)
    assert lm["writer_replace" if admit else "writer_keep"] == 1


def test_wrapper_default_preserves_same_input_actions_bank_rng_and_legacy_info():
    source = actor()
    with patch.object(EchoPolicyV1, "__init__", lambda self, *a, **k: self.__dict__.update(source.__dict__)):
        wrapped = EchoMinFillPolicy("/base", "/echo", device="cpu")
    reference, changed = calls(actor()), calls(wrapped)
    extra = {"min_fill_override", "original_min_fill", "effective_min_fill", "min_fill_source_sha256"}
    for left, right in zip(reference, changed, strict=True):
        assert left[0].keys() == right[0].keys()
        assert all(np.array_equal(left[0][key], right[0][key]) for key in left[0])
        assert left[1] == {k: v for k, v in right[1].items() if k not in extra}
        assert left[2].event_ids == right[2].event_ids and torch.equal(left[2].tokens, right[2].tokens)
        assert torch.equal(left[3], right[3])
        assert right[1]["min_fill_override"] is None
        assert right[1]["effective_min_fill"] == right[1]["original_min_fill"] == 1
        assert right[1]["min_fill_source_sha256"] == min_fill_source_identity()


def test_wrapper_rejects_post_initialization_config_drift():
    source = actor()
    with patch.object(EchoPolicyV1, "__init__", lambda self, *a, **k: self.__dict__.update(source.__dict__)):
        wrapped = EchoMinFillPolicy("/base", "/echo", device="cpu")
    apply_min_fill_override(wrapped.representation, 2)
    with pytest.raises(ValueError, match="during evaluation"):
        wrapped._get_action({})


def test_server_override_is_explicit_and_sources_are_separate():
    base = ["--base-model", "/base", "--checkpoint", "/echo"]
    assert build_parser().parse_args(base).min_fill is None
    assert build_parser().parse_args(base + ["--min-fill", "32"]).min_fill == 32
    assert build_parser().parse_args(base + ["--min-fill", "4"]).min_fill == 4
    with pytest.raises(SystemExit):
        build_parser().parse_args(base + ["--min-fill", "8"])
    assert set(min_fill_source_identity()) == set(MIN_FILL_SOURCES)
    assert all(len(value) == 64 for value in min_fill_source_identity().values())
