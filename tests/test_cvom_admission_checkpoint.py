"""Writer publication must never weaken the immutable actor contract."""
import json

import pytest
import torch

from run_scripts.robomme import cvom_admission_checkpoint as ckpt


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    from run_scripts.robomme.cvom_admission_core import AdmissionConfig, CVoMAdmission
    parent = tmp_path / "actor"
    parent.mkdir()
    (parent / "checkpoint.json").write_text(json.dumps({"config": {"representation": {
        "hidden_dim": 8, "num_short_tokens": 2, "capacity_events": 3}}}))
    identity = {"path": str(parent), "checkpoint_sha256": "fixed", "payload_sha256": {"expert": "fixed"}}
    def parent_info(path, **kwargs):
        return {**identity, "path": str(path)}
    monkeypatch.setattr(ckpt, "parent_identity", parent_info)
    controller = CVoMAdmission(AdmissionConfig(dim=8, num_tokens=2, capacity_events=3, hidden_dim=8))
    out = tmp_path / "writer"
    meta = {"actor_frozen": True, "future_inputs_at_inference": False}
    return parent, out, controller, meta


def test_roundtrip_and_rng_preserved(bundle):
    parent, out, model, meta = bundle
    ckpt.save_checkpoint(out, model, parent, arm="coalitional", step=4, metadata=meta)
    before = torch.get_rng_state().clone()
    restored, info = ckpt.load_controller(out, parent)
    assert torch.equal(before, torch.get_rng_state())
    assert info["arm"] == "coalitional"
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    assert not any(p.requires_grad for p in restored.parameters())
    with pytest.raises(FileExistsError):
        ckpt.save_checkpoint(out, model, parent, arm="single", step=4, metadata=meta)


def test_changed_parent_and_payload_rejected(bundle, tmp_path):
    parent, out, model, meta = bundle
    ckpt.save_checkpoint(out, model, parent, arm="single", step=2, metadata=meta)
    with pytest.raises(ValueError, match="parent identity"):
        ckpt.inspect_checkpoint(out, tmp_path / "other")
    with (out / ckpt.WEIGHTS).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="payload changed"):
        ckpt.inspect_checkpoint(out, parent)


def test_future_input_and_parent_overlap_rejected(bundle):
    parent, out, model, meta = bundle
    with pytest.raises(ValueError, match="overlap"):
        ckpt.save_checkpoint(parent / "writer", model, parent, arm="single", step=1, metadata=meta)
    with pytest.raises(ValueError, match="causal"):
        ckpt.save_checkpoint(out, model, parent, arm="single", step=1,
                             metadata={**meta, "future_inputs_at_inference": True})


def test_shape_contract_and_nonfinite_rejected(bundle):
    parent, out, model, meta = bundle
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="tensor"):
        ckpt.save_checkpoint(out, model, parent, arm="single", step=1, metadata=meta)
