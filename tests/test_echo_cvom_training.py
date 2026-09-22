"""Small CPU tests for sidecar immutability and non-discarded utility labels."""
from dataclasses import asdict
import json

import pytest
import torch

from run_scripts.robomme.echo_cvom_core import EchoConfig, EchoMemoryV1
from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18
from run_scripts.robomme import echo_cvom_checkpoint as cp
from run_scripts.robomme.train_echo_cvom import tensors, utility_loss, digest, verify_label, validate_args, parser


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    rc = RepresentationConfigV18(feature_dim=8, state_dim=3, num_short_tokens=2,
        hidden_dim=4, num_heads=2, capacity_events=3, short_window=2)
    ec = EchoConfig(capacity_events=3, min_fill=2, action_dim=4, utility_hidden=8, effect_hidden=8)
    core = EchoMemoryV1(rc, ec)
    head = torch.nn.Linear(2, 1, bias=False)
    base, parent = {"path": str(tmp_path/"base"), "signature": "base"}, {"path": str(tmp_path/"parent"), "signature": "parent"}
    monkeypatch.setattr(cp, "checkpoint_identity", lambda path: base)
    monkeypatch.setattr(cp, "parent_identity", lambda path: parent)
    monkeypatch.setattr(cp, "_expected_expert_shapes", lambda base, cfg: {"weight": head.weight.shape})
    monkeypatch.setattr(cp, "expert_state_dict", lambda module: {"weight": module.weight})
    def load(module, state):
        with torch.no_grad():
            module.weight.copy_(state["weight"])
    monkeypatch.setattr(cp, "load_expert_state_dict", load)
    config = {"representation": asdict(rc), "echo": asdict(ec), "expert": {}, "expert_targets": []}
    meta = {"base_model": base, "parent_identity": parent, "cache_fingerprint": "cache",
            "plan_sha256": "plan", "future_inputs_at_inference": False}
    return tmp_path, core, head, config, meta


def test_sidecar_roundtrip_no_overwrite_and_no_rng_consumption(fixture):
    root, core, head, config, meta = fixture
    path = cp.save_checkpoint(root/"stage1", 2, core, head, None, config, meta, stage=1)
    rng = torch.get_rng_state().clone()
    info = cp.inspect_checkpoint(meta["base_model"]["path"], path)
    assert torch.equal(rng, torch.get_rng_state())
    assert info["stage"] == 1 and info["step"] == 2
    restored, _ = cp.load_core(path)
    assert all(torch.equal(value, restored.delta_state_dict()[key]) for key,value in core.delta_state_dict().items())
    with pytest.raises(FileExistsError):
        cp.save_checkpoint(root/"stage1", 2, core, head, None, config, meta, stage=1)


def test_writer_sidecar_changes_only_manager(fixture):
    root, core, head, config, meta = fixture
    source = cp.save_checkpoint(root/"stage1", 2, core, head, None, config, meta, stage=1)
    source_hashes = cp.inspect_checkpoint(meta["base_model"]["path"], source)["files_sha256"]
    with torch.no_grad():
        next(core.manager.parameters()).add_(.1)
    dest = cp.save_manager_checkpoint(root/"stage2", 3, core, None, source, config, meta)
    info = cp.inspect_checkpoint(meta["base_model"]["path"], dest)
    assert info["files_sha256"]["expert.safetensors"] == source_hashes["expert.safetensors"]
    assert cp.inspect_checkpoint(meta["base_model"]["path"], source)["files_sha256"] == source_hashes
    with torch.no_grad():
        next(core.effect_adapter.parameters()).add_(.1)
    with pytest.raises(ValueError, match="frozen actor"):
        cp.save_manager_checkpoint(root/"bad", 4, core, None, source, config, meta)


def test_tampering_refused_before_load(fixture):
    root, core, head, config, meta = fixture
    path = cp.save_checkpoint(root/"stage1", 2, core, head, None, config, meta, stage=1)
    before = core.delta_state_dict()
    with (path/"core.safetensors").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="modified"):
        cp.load_checkpoint(path, core, head)
    assert all(torch.equal(value, core.delta_state_dict()[key]) for key,value in before.items())


def test_uncertain_labels_still_train_continuous_value():
    rows = [{"features": [0., 1.], "signed_mean": 1e-7, "noise_mean_std": .01},
            {"features": [1., 0.], "signed_mean": -2e-7, "noise_mean_std": .01}]
    data = tensors(rows, 1e-6, 1e-6, 1.96)
    assert not data["confident"].any()
    assert (data["weight"] >= .1).all()
    utility, logit = torch.zeros(2, requires_grad=True), torch.zeros(2, requires_grad=True)
    loss, reg, bce = utility_loss({"utility": utility, "logit": logit}, data, torch.arange(2), .25)
    loss.backward()
    assert reg > 0 and bce == 0 and utility.grad.abs().sum() > 0
    assert logit.grad.abs().sum() == 0


def test_label_hash_and_split_identity():
    protocol, row = {"fingerprint": "fixed"}, {"episode_id": 1, "event": 2}
    context = {**row, "teacher_snapshot_version": "fixed", "candidate_event_id": 2, "bank_event_ids": [[1]],
        "targets": [{"index": 1, "event_id": 2, "event_ids": [2], "is_new": True,
                     "features": [0., 1.], "signed_mean": .1, "noise_mean_std": .01}]}
    packet = {"protocol_fingerprint": "fixed", "split": "train", "row": row,
              "result": {"contexts": [context]}}
    packet["sha256"] = digest(packet)
    assert verify_label(packet, protocol, "train", row) == context
    with pytest.raises(ValueError):
        verify_label(packet, protocol, "val", row)
    packet["result"]["contexts"][0]["targets"].append({"signed_mean": 99})
    with pytest.raises(ValueError):
        verify_label(packet, protocol, "train", row)


def test_inner_context_cannot_disagree_with_outer_row():
    protocol, row = {"fingerprint": "fixed"}, {"episode_id": 1, "event": 2}
    packet = {"protocol_fingerprint": "fixed", "split": "train", "row": row,
              "result": {"contexts": [{"episode_id": 999, "event": 100, "targets": []}]}}
    packet["sha256"] = digest(packet)
    with pytest.raises(ValueError, match="Inner label context"):
        verify_label(packet, protocol, "train", row)


def test_extreme_finite_utility_does_not_overflow():
    data = tensors([{"features": [0., 1.], "signed_mean": 3e38, "noise_mean_std": 1.}], 1e-9, 1e-6, 1.96)
    assert torch.isfinite(data["utility"]).all()
    with pytest.raises(ValueError):
        tensors([], 0., 0., 1.)


def test_episode_signatures_detect_payload_changes(tmp_path):
    from types import SimpleNamespace
    from run_scripts.robomme.train_echo_cvom import episode_signatures
    path = tmp_path/"ep.pt"
    path.write_bytes(b"data")
    cache = SimpleNamespace(path=tmp_path, manifest={"episodes": [{"episode_id": 1, "path": "ep.pt"}]})
    before = episode_signatures(cache, [1])
    path.write_bytes(b"different")
    assert episode_signatures(cache, [1]) != before


def test_default_two_contexts_and_invalid_cli():
    args = parser().parse_args(["warmup", "--output-dir", "new"])
    validate_args(args)
    assert args.contexts_per_episode == 2
    args.noise_samples = 1
    with pytest.raises(ValueError):
        validate_args(args)
