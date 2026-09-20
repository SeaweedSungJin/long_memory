"""CPU-only V18 contracts: causality, source isolation, LoRA and replay parity."""
from dataclasses import replace

import pytest
import torch

from gr00t.model.modules.memory import MemoryTransformer
from run_scripts.robomme.representation_core_v18 import (
    RepresentationConfigV18, RepresentationMemoryV18,
)


def config(**kwargs):
    return RepresentationConfigV18(feature_dim=16, state_dim=3, num_short_tokens=2,
                                  hidden_dim=8, num_heads=2, short_window=3,
                                  short_lora_rank=2, capacity_events=3, **kwargs)


def transformer():
    return MemoryTransformer(dim=16, n_q=2, T=3, num_layers=2, num_heads=2).eval()


def episode(n=7):
    torch.manual_seed(47)
    return {"short": torch.randn(n, 2, 16), "moment": torch.randn(n, 2, 16),
            "state": torch.randn(n, 3), "frames": torch.arange(n)*16,
            "is_demo": torch.arange(n) < 2}


def live_fusion(core):
    with torch.no_grad():
        core.memory.fusion_projection.weight.normal_(std=0.1)


def run_online(core, ep, decision, **kwargs):
    bank, history = None, None
    result = None
    for index in range(decision+1):
        result = core.step(ep["short"][index:index+1], ep["moment"][index:index+1],
                           ep["state"][index:index+1], ep["frames"][index:index+1],
                           ep["is_demo"][index:index+1], bank=bank, moment_history=history,
                           event_index=index, **kwargs)
        bank, history = result["bank"], result["moment_history"]
    return result


@pytest.mark.parametrize("representation", ["short", "adapted_short", "moment"])
def test_online_replay_parity_read_before_write_and_fifo(representation):
    core = RepresentationMemoryV18(config(representation=representation), transformer())
    live_fusion(core)
    ep = episode()
    with torch.no_grad():
        replay = core.replay(ep, 5)
        online = run_online(core, ep, 5)
    torch.testing.assert_close(online["fused"], replay["fused"], rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(online["short"], replay["short"], rtol=0, atol=0)
    assert replay["bank"].shape == (1, 6, 8)
    encoded = core.encode_prefix(ep, 6)
    torch.testing.assert_close(replay["bank"], encoded["stored"][2:5].reshape(1, 6, 8))
    torch.testing.assert_close(online["bank"], encoded["stored"][3:6].reshape(1, 6, 8))
    assert replay["metrics"]["bank_fill"].item() == 1
    assert core.replay(ep, 0)["bank"].shape[1] == 0


@pytest.mark.parametrize("representation", ["short", "adapted_short", "moment"])
def test_no_future_or_action_access(representation):
    core = RepresentationMemoryV18(config(representation=representation), transformer())
    live_fusion(core)
    ep = episode()
    first = core.replay(ep, 3)
    # Future inputs are deliberately corrupt. Prefix functions must not even
    # validate future contents, nor consult supervision when creating memories.
    for name in ("short", "moment", "state"):
        ep[name][4:] = torch.nan
    ep["targets"] = object()
    ep["actions"] = object()
    second = core.replay(ep, 3)
    torch.testing.assert_close(first["fused"], second["fused"], rtol=0, atol=0)
    torch.testing.assert_close(first["bank"], second["bank"], rtol=0, atol=0)


def test_c_changes_only_stored_source_not_query():
    torch.manual_seed(4)
    a = RepresentationMemoryV18(config())
    torch.manual_seed(4)
    c = RepresentationMemoryV18(config(representation="moment"))
    ep = episode()
    a_encoded, c_encoded = a.encode_prefix(ep, 5), c.encode_prefix(ep, 5)
    torch.testing.assert_close(a_encoded["short"], c_encoded["short"], rtol=0, atol=0)
    torch.testing.assert_close(a_encoded["query"], c_encoded["query"], rtol=0, atol=0)
    assert not torch.equal(a_encoded["stored"], c_encoded["stored"])
    expected = c.encode_event(ep["moment"][:5], ep["state"][:5], ep["frames"][:5], ep["is_demo"][:5])
    torch.testing.assert_close(c_encoded["stored"], expected, rtol=0, atol=0)


def test_zero_lora_parity_left_padding_and_base_unchanged():
    original = transformer()
    before = {name: value.clone() for name, value in original.state_dict().items()}
    core = RepresentationMemoryV18(config(representation="adapted_short"), original)
    ep = episode()
    actual = core.encode_prefix(ep, 5)["short"]
    expected = []
    for index in range(5):
        ids = [max(0, j) for j in range(index-2, index+1)]
        expected.append(original(ep["moment"][ids].reshape(1, 6, 16))[:, -2:])
    torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)
    assert not set(map(id, core.parameters())) & set(map(id, original.parameters()))
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    learned = [name for name, p in core.short_transformer.named_parameters() if p.requires_grad]
    assert len(learned) == 16 and all("lora_" in name for name in learned)


def test_read_off_uses_same_adapted_short_and_has_short_gradient():
    core = RepresentationMemoryV18(config(representation="adapted_short"), transformer())
    ep = episode()
    with torch.no_grad():
        for parameter in core.short_parameters():
            parameter.add_(0.07)
    result = core.replay(ep, 4, read_enabled=False)
    torch.testing.assert_close(result["fused"], result["short"], rtol=0, atol=0)
    assert not torch.equal(result["short"], ep["short"][4:5])
    result["fused"].square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in core.short_parameters())
    assert result["metrics"]["read_norm"].item() == 0


@pytest.mark.parametrize("representation", ["short", "adapted_short", "moment"])
def test_historical_write_receives_gradient_without_future_detach(representation):
    core = RepresentationMemoryV18(config(representation=representation), transformer())
    live_fusion(core)
    ep = episode()
    for name in ("short", "moment"):
        ep[name] = ep[name].requires_grad_(True)
    result = core.replay(ep, 2, activation_checkpointing=True)
    result["fused"].square().mean().backward()
    source = "moment" if representation in ("moment", "adapted_short") else "short"
    assert ep[source].grad[0].abs().sum() > 0
    assert ep[source].grad[3:].abs().sum() == 0
    if representation == "adapted_short":
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in core.short_parameters())


def test_delta_roundtrip_is_compact_strict_and_atomic():
    original = transformer()
    core = RepresentationMemoryV18(config(representation="adapted_short"), original)
    delta = core.delta_state_dict()
    assert any(name.startswith("short_transformer.") for name in delta)
    assert not any(".base." in name or "ffn.gate_proj" in name for name in delta)
    restored = RepresentationMemoryV18(core.config, original)
    restored.load_delta_state_dict(delta)
    ep = episode()
    torch.testing.assert_close(core.replay(ep, 4)["fused"], restored.replay(ep, 4)["fused"], rtol=0, atol=0)
    before = restored.delta_state_dict()
    bad = {name: value + 1 for name, value in delta.items()}
    bad[list(bad)[-1]] = torch.tensor([float("nan")])
    with pytest.raises(ValueError):
        restored.load_delta_state_dict(bad)
    for name, value in before.items():
        torch.testing.assert_close(value, restored.delta_state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="keys differ"):
        restored.load_delta_state_dict({})


def test_variant_construction_preserves_common_initialization_and_later_rng():
    base = transformer()
    states = []
    for representation, gate in (("short", "linear"), ("adapted_short", "linear"),
                                 ("moment", "linear"), ("short", "mlp")):
        torch.manual_seed(61)
        core = RepresentationMemoryV18(config(representation=representation, gate=gate), base)
        states.append((core.memory.state_dict(), torch.rand(17)))
    for memory, later in states[1:]:
        torch.testing.assert_close(later, states[0][1], rtol=0, atol=0)
        for name, value in states[0][0].items():
            if not name.startswith("fusion_gate."):
                torch.testing.assert_close(value, memory[name], rtol=0, atol=0)


def test_callback_sees_only_past_and_respects_current_observation():
    core = RepresentationMemoryV18(config())
    ep = episode()
    calls = []

    def writer(bank, encoded, *, event_index, frame, is_demo):
        calls.append((event_index, frame.item(), is_demo.item()))
        if event_index % 2:
            return bank, {"write_rate": encoded.new_zeros(())}
        return core.write_fifo(bank, encoded), {"write_rate": encoded.new_ones(())}

    result = core.replay(ep, 4, write_policy=writer)
    assert calls == [(0, 0, True), (1, 16, True), (2, 32, False), (3, 48, False)]
    assert result["bank"].shape[1] == 4
    assert result["metrics"]["write_rate"].item() == 0.5


def test_validation_and_frozen_unused_parameters():
    with pytest.raises(ValueError):
        RepresentationConfigV18(capacity_events=True)
    with pytest.raises(ValueError):
        RepresentationMemoryV18(config(representation="adapted_short"))
    core = RepresentationMemoryV18(config())
    trainable = {name for name, p in core.named_parameters() if p.requires_grad}
    assert not any("write_" in name or "slot_addresses" in name or "update_gate" in name for name in trainable)
    with pytest.raises(ValueError):
        core.write_fifo(torch.zeros(1, 8, 8), torch.zeros(1, 2, 8))
    invalid = episode()
    invalid["frames"][2] = 0
    with pytest.raises(ValueError, match="increasing"):
        core.replay(invalid, 3)


def test_mlp_gate_keeps_empty_and_off_exact_identity():
    core = RepresentationMemoryV18(config(gate="mlp"))
    live_fusion(core)
    ep = episode()
    with torch.no_grad():
        core.memory.fusion_gate[-1].bias.fill_(5)
    empty = core.replay(ep, 0)
    disabled = core.replay(ep, 4, read_enabled=False)
    torch.testing.assert_close(empty["fused"], ep["short"][:1], rtol=0, atol=0)
    torch.testing.assert_close(disabled["fused"], ep["short"][4:5], rtol=0, atol=0)
