"""CPU contracts for real ECHO policy wiring and immutable evaluation roles."""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from run_scripts.robomme import eval_echo_cvom as evaluation
from run_scripts.robomme import policy_echo_cvom as policy_module
from run_scripts.robomme import serve_echo_cvom as server
from run_scripts.robomme.echo_cvom_core import EchoConfig, EchoMemoryV1
from run_scripts.robomme.feature_precision_v19 import feature_precision_contract
from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18
from tests.test_feature_precision_v19_policy import actor as original_actor
from tests.test_long_memory_online_policy import _observation, _options


def actor(*, off=False, fifo=False):
    parent = original_actor()
    policy = policy_module.EchoPolicyV1.__new__(policy_module.EchoPolicyV1)
    policy.__dict__.update(parent.__dict__)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9512)
        core = EchoMemoryV1(parent.representation.config, EchoConfig(capacity_events=3,
            min_fill=1, action_dim=8, effect_hidden=16, utility_hidden=16))
        core.initialize_from_parent_delta(parent.representation.delta_state_dict())
        with torch.no_grad():
            core.effect_adapter.output.weight.normal_(std=.2)
    policy.representation = core.eval().requires_grad_(False)
    policy.mode = policy_module.VARIANT
    policy.stage = 2
    policy.memory_off = off
    policy.write_mode = "fifo" if fifo else "learned"
    policy.write_policy = "fifo" if fifo else "echo-cvom"
    policy.echo_checkpoint_sha256 = "a" * 64
    policy.echo_parent_identity = {"path": "/parent"}
    policy.echo_source_sha256 = {name: "b" * 64 for name in policy_module.RUNTIME_SOURCES}
    return policy


def calls(policy):
    records = []
    for index, (frame, passive) in enumerate(((0, True), (2, True), (4, False), (6, False), (8, False))):
        controls = np.full((2, 8), .25 * index, np.float32) if index > 2 else None
        actions, info = policy.get_action(_observation(marker=index + 1),
            _options(frame=frame, passive=passive, controls=controls))
        session = policy.sessions["A"]
        records.append((actions, info, session.echo_bank.detach(), session.generator.get_state().clone()))
    return records


def test_completed_controls_use_previous_state_only_and_mask_padding():
    policy = actor()
    previous = {"joint_position": np.full((1, 7), .25), "gripper_position": np.ones((1, 1))}
    raw = np.full((1, 8), 3., np.float32)
    controls, mask = policy_module.completed_controls(policy.processor, "new_embodiment", raw,
        previous, 2, device="cpu")
    assert policy.processor.normalization_states[-1] is previous
    assert controls.shape == (1, 2, 8) and mask.tolist() == [[True, False]]
    assert torch.equal(controls[0, 0], torch.ones(8))
    assert torch.equal(controls[0, 1], torch.zeros(8))


def test_policy_receives_only_previous_execution_and_primes_without_noise():
    policy = actor(fifo=True)
    inputs = []
    original = policy.representation.step

    def observe(*args, **kwargs):
        inputs.append((args, dict(kwargs)))
        return original(*args, **kwargs)

    with patch.object(policy.representation, "step", observe):
        records = calls(policy)
    assert "completed_actions" not in inputs[0][1]
    assert not inputs[1][1]["completed_action_mask"].any()
    assert not inputs[2][1]["completed_action_mask"].any()  # demo -> first execution
    assert inputs[3][1]["completed_action_mask"].all()
    assert torch.equal(inputs[3][1]["completed_actions"], torch.full((1, 2, 8), .75))
    expected = torch.Generator().manual_seed(17).get_state()
    assert torch.equal(records[0][3], expected) and torch.equal(records[1][3], expected)
    assert records[-1][1]["checkpoint_variant"] == "echo_cvom_v1"
    assert records[-1][1]["long_memory"]["completed_action_count"] == 2
    assert records[-1][2].n_events == 3


def test_read_off_preserves_same_writes_and_episode_rng_on_fixed_observations():
    active, disabled = calls(actor()), calls(actor(off=True))
    assert any(row[1]["long_memory"]["read"]["ae_conditioning_delta_norm"] > 0 for row in active)
    for left, right in zip(active, disabled, strict=True):
        assert torch.equal(left[2].tokens, right[2].tokens)
        assert left[2].event_ids == right[2].event_ids
        assert torch.equal(left[3], right[3])
        assert right[1]["long_memory"]["memory_read_enabled"] is False
        assert right[1]["long_memory"]["read"]["ae_conditioning_delta_norm"] == 0
        for name in ("writer_append", "writer_keep", "writer_replace", "writer_merge"):
            assert left[1]["long_memory"]["read"][name] == right[1]["long_memory"]["read"][name]


def test_reset_drops_completed_transition_state_and_future_prefix_is_rejected():
    policy = actor()
    calls(policy)
    policy.reset({"session_ids": ["A"]})
    with pytest.raises(ValueError, match="First endpoint"):
        policy.get_action(_observation(), _options(frame=0, controls=np.zeros((1, 8), np.float32)))
    policy.reset({"session_ids": ["A"]})
    policy.get_action(_observation(), _options(frame=0, passive=False))
    with pytest.raises(ValueError, match="cadence"):
        policy.get_action(_observation(), _options(frame=2, controls=np.zeros((3, 8), np.float32)))


def manifest():
    base = {"path": "/base", "id": "base"}
    sources = {name: "d" * 64 for name in policy_module.RUNTIME_SOURCES}
    payload = {"core.safetensors": "a" * 64, "expert.safetensors": "b" * 64}
    config = {"representation": asdict(RepresentationConfigV18()), "echo": asdict(EchoConfig()),
              "expert": {}, "expert_targets": []}
    common = {"base_model": base, "memory_checkpoint": "/echo", "mode": evaluation.VARIANT,
        "server_script": evaluation.SERVER, "step": 5, "stage": 2,
        "representation_config": config["representation"], "training_config": config,
        "training_metadata": {"base_model": base, "future_inputs_at_inference": False,
            "parent_identity": {"path": "/parent"}, "payload_sha256": payload},
        "checkpoint_files_sha256": {**payload, "checkpoint.json": "c" * 64},
        "feature_precision": "native", "feature_precision_rules": feature_precision_contract("native"),
        "echo_source_sha256": sources}
    m = {"format_version": 1, "trainer_variant": evaluation.VARIANT,
        "settings": {**copy.deepcopy(evaluation.PANEL), "save_videos": False, "device": "cuda:0"},
        "baseline_reference": {"kind": "completed_original_baseline_reference"},
        "source_sha256": {"run_scripts/robomme/" + k: v for k, v in sources.items()},
        "models": {"baseline": {"base_model": base, "mode": "none", "memory_checkpoint": None,
            "write_policy": "none", "write_policy_override": "checkpoint", "memory_off": False,
            "archive_read_off": False, "server_script": evaluation.BASELINE_SERVER}}}
    for role in evaluation.ROLES:
        m["models"][role] = dict(copy.deepcopy(common), memory_off=role == "memory-off",
            write_policy="fifo" if role == "fifo" else "echo-cvom")
    m["evaluation_id"] = evaluation.identity_digest(m)
    return m


def test_fixed_eval_contract_and_server_dispatch():
    args = evaluation.build_parser().parse_args(["--checkpoint", "/echo", "--output-dir", "/new"])
    evaluation.validate_options(args)
    assert args.tasks == list(evaluation.TASKS) and args.n_episodes == 10 and args.seed == 6
    m = manifest()
    evaluation.validate_manifest_contract(m)
    for role in evaluation.ROLES:
        command = evaluation.server_command(args, m["models"][role], 40001)
        parsed = server.build_parser().parse_args(command[3:])
        assert parsed.checkpoint == "/echo" and parsed.fifo == (role == "fifo")
        assert parsed.memory_off == (role == "memory-off")


@pytest.mark.parametrize("field", ["variant", "payload", "source", "actor", "future", "precision", "seed"])
def test_eval_rejects_altered_actor_causality_source_and_panel(field):
    m = manifest()
    if field == "variant":
        m["trainer_variant"] = "representation_v18"
    elif field == "payload":
        m["models"]["memory"]["checkpoint_files_sha256"]["core.safetensors"] = "e" * 64
    elif field == "source":
        m["models"]["memory"]["echo_source_sha256"]["echo_cvom_core.py"] = "e" * 64
    elif field == "actor":
        m["models"]["memory-off"]["memory_checkpoint"] = "/other"
    elif field == "future":
        m["models"]["memory"]["training_metadata"]["future_inputs_at_inference"] = True
    elif field == "precision":
        m["models"]["memory"]["feature_precision"] = "cache-aligned"
    else:
        m["settings"]["seed"] = 7
    m["evaluation_id"] = evaluation.identity_digest(m)
    with pytest.raises(ValueError):
        evaluation.validate_manifest_contract(m)


@pytest.mark.parametrize("flag", ["training_complete", "inherited_actor_training_complete"])
def test_nonzero_incomplete_checkpoint_requires_explicit_diagnostic_opt_in(flag):
    m = manifest()
    for role in evaluation.ROLES:
        m["models"][role]["step"] = 2
        m["models"][role]["training_metadata"][flag] = False
    m["evaluation_id"] = evaluation.identity_digest(m)
    with pytest.raises(ValueError, match="incomplete"):
        evaluation.validate_manifest_contract(m)
    m["allow_initialization_checkpoints"] = True
    m["evaluation_id"] = evaluation.identity_digest(m)
    evaluation.validate_manifest_contract(m)


def test_preflight_has_no_rollout_or_output():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "absent"
        with patch.object(evaluation, "check_dependencies"), patch.object(evaluation, "build_identity", return_value=manifest()), \
                patch.object(evaluation, "run_evaluation", side_effect=AssertionError("unexpected rollout")):
            assert evaluation.main(["--checkpoint", "/echo", "--output-dir", str(out), "--preflight-only"]) == 0
        assert not out.exists()


def test_runtime_evidence_checks_completed_controls_source_and_write_when_read_off():
    policy = actor(off=True)
    outputs = calls(policy)
    m = manifest()
    model = m["models"]["memory-off"]
    model.update(representation_config=asdict(policy.representation.config),
        echo_source_sha256=policy.echo_source_sha256, step=policy.checkpoint_step,
        checkpoint_files_sha256={"checkpoint.json": policy.echo_checkpoint_sha256})
    model["training_metadata"].update(payload_sha256=policy.payload_sha256,
        parent_identity=policy.echo_parent_identity)
    m["settings"]["tasks"] = ["BinFill"]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "memory-off" / "BinFill"
        folder.mkdir(parents=True)
        with (folder / "simulation_results.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["episode_idx", "episode_seed", "success", "status"])
            writer.writeheader()
            writer.writerow({"episode_idx": 0, "episode_seed": 17, "success": 0, "status": "fail"})
        rows = []
        for i, (_, info, _, _) in enumerate(outputs):
            rows.append({"kind": "policy_call", "episode_idx": 0, "episode_seed": 17,
                "session_id": "run", "call": i, "frame_index": i * 2,
                "passive": i < 2, "executed_action_count": 2 if i > 2 else 0, "info": info})
        rows.append({"kind": "episode_complete", "episode_idx": 0, "episode_seed": 17,
                     "session_id": "run", "success": 0})
        path = folder / "memory_diagnostics.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = evaluation.completed_read_diagnostics(Path(tmp), "memory-off", m)
        assert result["completed_sessions"] == 1 and result["enabled_calls"] == 0
        assert sum(result["writer_decisions"].values()) == 5
        assert len(result["files_sha256"]) == 2 and not result["complete_evidence"]
        rows[3]["executed_action_count"] = 3
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(ValueError, match="prefix"):
            evaluation.completed_read_diagnostics(Path(tmp), "memory-off", m)


def test_shell_preflight_is_read_only_and_pins_full_training_arguments():
    script = Path(__file__).resolve().parents[1] / "run_scripts/robomme/run_echo_cvom.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "nested" / "absent"
        env = {**os.environ, "ECHO_PYTHON": "/bin/echo", "ECHO_RUN_DIR": str(run),
               "ECHO_EPOCHS": "1", "ECHO_QUERY_BATCH": "4", "ECHO_FINAL_PHASE": "refresh"}
        result = subprocess.run(["bash", str(script), "preflight"], env=env,
                                text=True, capture_output=True, check=True)
        assert "train_echo_cvom.py warmup" in result.stdout
        assert "--epochs 1 --query-batch-size 4 --activation-checkpointing --preflight-only" in result.stdout
        assert not run.parent.exists()
        assert not Path(str(run) + ".lock").exists()


def test_gpu_occupancy_rejects_native_verification_before_model_or_output():
    from run_scripts.robomme import verify_echo_cvom as verifier
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "verification"
        cache = SimpleNamespace(path=Path(tmp) / "cache", manifest={
            "model_path": str(Path(tmp) / "base"), "splits": {"val": [1355, 626]}})
        with patch.object(verifier, "EpisodeCache", return_value=cache), \
                patch.object(verifier, "inspect_checkpoint", return_value={}), \
                patch.object(verifier, "gpu_guard", side_effect=RuntimeError("Existing GPU compute")), \
                patch.object(verifier, "EchoPolicyV1") as constructor:
            with pytest.raises(RuntimeError, match="Existing GPU"):
                verifier.main(["--checkpoint", str(Path(tmp) / "checkpoint"), "--output-dir", str(out)])
            constructor.assert_not_called()
        assert not out.exists()


def test_gpu_occupancy_rejects_eval_before_starting_policy_server():
    with tempfile.TemporaryDirectory() as tmp:
        args = evaluation.build_parser().parse_args(["--checkpoint", "/echo", "--output-dir", str(Path(tmp) / "eval")])
        evaluation.validate_options(args)
        m = manifest()
        report = {"models": {"memory": {"complete": False}},
                  "runtime_evidence": {"memory": {"complete_evidence": False}}}
        with patch.object(evaluation, "validate_reference"), patch.object(evaluation, "verify_runtime_inputs"), \
                patch.object(evaluation, "free_local_port", return_value=40000), \
                patch.object(evaluation, "gpu_guard", side_effect=RuntimeError("Existing GPU compute")) as guard, \
                patch.object(evaluation, "write_report", return_value=(report, "blocked")), \
                patch.object(evaluation.subprocess, "Popen") as launch:
            assert evaluation.run_evaluation(args, m, {}) == 1
            assert guard.call_count == 3
            launch.assert_not_called()
