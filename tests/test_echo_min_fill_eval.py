"""Narrow new-server contracts do not weaken old evaluation validators."""
import copy
import pytest
from run_scripts.robomme import eval_echo_min_fill as evaluation


@pytest.mark.parametrize("key", ["scenario_metadata_sha256", "model_config_sha256", "memory_window", "demo_sampling"])
def test_cross_arm_policy_contract_rejects_missing_or_changed_fields(key):
    before = dict(scenario_metadata_sha256="scenarios", model_config_sha256="config", memory_window=4,
                  demo_sampling="backward_aligned_full_history")
    evaluation.paired_policy_contract(before, dict(before))
    after = dict(before); after[key] = "changed"
    with pytest.raises(ValueError, match=key):
        evaluation.paired_policy_contract(before, after)
    after.pop(key)
    with pytest.raises(ValueError, match=key):
        evaluation.paired_policy_contract(before, after)


def minimal_identity():
    sources = evaluation.min_fill_source_identity()
    config = {"min_fill": 4, "capacity_events": 32}
    return {"min_fill_study": {"version": evaluation.VERSION}, "models": {"baseline": {}, "memory": {
        "server_script": evaluation.SERVER, "original_min_fill": 4, "effective_min_fill": 32,
        "min_fill_override": 32, "training_config": {"echo": config}, "stage": 2,
        "write_policy": "echo-cvom", "memory_off": False, "effective_echo_config": dict(config,min_fill=32),
        "min_fill_source_sha256": sources}},
        "source_sha256": {"run_scripts/robomme/"+k:v for k,v in sources.items()}}


def test_legacy_projection_changes_only_in_memory_server_and_leaves_evidence_intact(monkeypatch):
    identity = minimal_identity(); identity["evaluation_id"] = evaluation.old.identity_digest(identity)
    saved = copy.deepcopy(identity)
    captures = []
    monkeypatch.setattr(evaluation.old, "validate_manifest_contract", lambda value: captures.append(value))
    evaluation.validate_identity(identity)
    assert identity == saved
    assert captures[0]["models"]["memory"]["server_script"] == evaluation.old.SERVER
    assert identity["models"]["memory"]["server_script"] == evaluation.SERVER


@pytest.mark.parametrize("key,value", [("effective_min_fill",16),("min_fill_override",None),
    ("stage",1),("write_policy","fifo"),("memory_off",True),("server_script","wrong.py")])
def test_only_declared_runtime_minfill32_override_allowed(monkeypatch,key,value):
    identity = minimal_identity(); identity["models"]["memory"][key] = value
    identity["evaluation_id"] = evaluation.old.identity_digest(identity)
    monkeypatch.setattr(evaluation.old, "validate_manifest_contract", lambda value: None)
    with pytest.raises(ValueError): evaluation.validate_identity(identity)
