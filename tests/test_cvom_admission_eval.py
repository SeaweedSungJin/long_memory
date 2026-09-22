"""CPU contracts for writer-only CVoM evaluation; no model/simulator/GPU launch."""
from __future__ import annotations

import copy
from contextlib import redirect_stderr
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from run_scripts.robomme import eval_cvom_admission as fixed
from run_scripts.robomme import eval_representation_v18 as ev
from run_scripts.robomme import policy_representation_v18 as policy_module
from run_scripts.robomme import serve_representation_v18 as server
from run_scripts.robomme.cvom_admission_core import AdmissionConfig, CVoMAdmission
from run_scripts.robomme.feature_precision_v19 import feature_precision_contract
from run_scripts.robomme.representation_core_v18 import RepresentationMemoryV18
from tests.test_representation_eval_v18 import manifest as original_manifest, rehash


def admission_manifest():
    m = original_manifest()
    common = copy.deepcopy(m["models"]["memory"])
    common["representation_config"].update(capacity_events=32, num_short_tokens=4, hidden_dim=16)
    common["feature_precision"] = "native"
    common["feature_precision_rules"] = feature_precision_contract("native")
    common["writer_checkpoint"] = "/writer"
    common["writer_files_sha256"] = {"cvom_admission.json": "b" * 64, "writer.safetensors": "c" * 64}
    sources = {name: "d" * 64 for name in policy_module.CVOM_RUNTIME_SOURCES}
    info = {"format_version": 1, "variant": "cvom_admission_v1", "arm": "single", "step": 10,
        "config": asdict(AdmissionConfig(dim=16)),
        "parent_identity": {"path": common["memory_checkpoint"], "base_model": common["base_model"],
            "checkpoint_sha256": common["checkpoint_files_sha256"]["checkpoint.json"],
            "payload_sha256": common["training_metadata"]["payload_sha256"]},
        "metadata": {"actor_frozen": True, "future_inputs_at_inference": False},
        "payload_sha256": {"writer.safetensors": "c" * 64}, "writer_sha256": "c" * 64,
        "manifest_sha256": "b" * 64, "files_sha256": copy.deepcopy(common["writer_files_sha256"])}
    common["cvom_admission"] = {"manifest": info, "source_sha256": sources}
    for role in ("memory", "fifo", "memory-off"):
        m["models"][role] = copy.deepcopy(common)
        m["models"][role].update(memory_off=role == "memory-off", write_policy="fifo" if role == "fifo" else "cvom-admission")
    m["source_sha256"] = {"run_scripts/robomme/" + name: digest for name, digest in sources.items()}
    return rehash(m)


def test_fixed_panel_pins_parent_and_references_original_without_extra_read_off():
    parsed = fixed.build_parser().parse_args(["--writer-checkpoint", "/writer", "--output-dir", "/output"])
    args = ev.build_parser().parse_args(fixed.evaluator_arguments(parsed))
    ev.validate_options(args)
    assert args.checkpoint == fixed.PARENT_CHECKPOINT
    assert args.models == ["baseline", "memory", "fifo"]
    assert args.tasks == list(ev.TASKS) and len(args.tasks) == 16
    assert (args.dataset, args.n_episodes, args.seed, args.n_action_steps, args.max_episode_steps) == ("val", 10, 6, 16, 1300)
    assert args.cvom_admission and args.feature_precision == "native"
    assert args.baseline_reference.name == "archive_read_best1250_val_n10_seed6"
    for flag, value in (("--seed", "7"), ("--tasks", "BinFill"), ("--dataset", "test"),
                        ("--feature-precision", "cache-aligned"), ("--checkpoint", "/different")):
        with redirect_stderr(io.StringIO()), unittest.TestCase().assertRaises(SystemExit):
            fixed.build_parser().parse_args(["--output-dir", "/output", flag, value])


def test_modes_fail_closed_before_model_loading_and_preflight_has_no_output():
    base = ["--checkpoint", "/parent", "--writer-checkpoint", "/writer", "--cvom-admission"]
    for extra in (["--semantic-memory"], ["--feature-precision", "cache-aligned"], ["--seed", "7"]):
        args = ev.build_parser().parse_args(base + extra)
        with unittest.TestCase().assertRaises(ValueError):
            ev.validate_options(args)
    with patch.object(policy_module, "checkpoint_info_v18") as load:
        for kwargs in ({"cvom_fifo": True}, {"cvom_admission": True},
                       {"cvom_admission": True, "writer_checkpoint": "/writer", "feature_precision": "cache-aligned"}):
            with unittest.TestCase().assertRaises(ValueError):
                policy_module.RepresentationPolicyV18("/base", "/parent", **kwargs)
    load.assert_not_called()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "absent"
        with patch.object(ev, "check_dependencies"), patch.object(ev, "build_identity", return_value=admission_manifest()), \
                patch.object(ev, "run_evaluation", side_effect=AssertionError("started rollout")):
            assert fixed.main(["--writer-checkpoint", "/writer", "--output-dir", str(out), "--preflight-only"]) == 0
        assert not out.exists()
        args = fixed.build_parser().parse_args(["--output-dir", str(out), "--report-only"])
        assert "--report-only" in fixed.evaluator_arguments(args)


def test_manifest_rejects_parent_writer_capacity_source_and_actor_changes():
    ev.validate_manifest_contract(original_manifest())
    ev.validate_manifest_contract(admission_manifest())
    for change in ("parent", "payload", "base", "writer", "source", "capacity", "future", "frozen", "off", "fifo"):
        m = admission_manifest()
        model = m["models"]["memory"]
        info = model["cvom_admission"]["manifest"]
        if change == "parent":
            info["parent_identity"]["checkpoint_sha256"] = "f" * 64
        elif change == "payload":
            info["parent_identity"]["payload_sha256"] = {}
        elif change == "base":
            info["parent_identity"]["base_model"] = {}
        elif change == "writer":
            model["writer_files_sha256"]["writer.safetensors"] = "f" * 64
        elif change == "source":
            model["cvom_admission"]["source_sha256"]["cvom_admission_core.py"] = "f" * 64
        elif change == "capacity":
            info["config"]["capacity_events"] = 31
        elif change == "future":
            info["metadata"]["future_inputs_at_inference"] = True
        elif change == "frozen":
            info["metadata"]["actor_frozen"] = False
        elif change == "off":
            m["models"]["memory-off"]["writer_checkpoint"] = "/another"
        else:
            m["models"]["fifo"]["write_policy"] = "cvom-admission"
        with unittest.TestCase().assertRaises(ValueError):
            ev.validate_manifest_contract(rehash(m))


def test_writer_initialization_is_only_allowed_as_explicit_diagnostic():
    m = admission_manifest()
    for role in ("memory", "fifo", "memory-off"):
        m["models"][role]["cvom_admission"]["manifest"]["step"] = 0
    with unittest.TestCase().assertRaisesRegex(ValueError, "not a trained"):
        ev.validate_manifest_contract(rehash(m))
    m["allow_initialization_checkpoints"] = True
    ev.validate_manifest_contract(rehash(m))


def test_dispatch_binds_same_parent_and_sidecar_even_when_fifo_bypasses_controller():
    m = admission_manifest()
    args = ev.build_parser().parse_args([])
    for role in ("memory", "fifo", "memory-off"):
        command = ev.server_command(args, m["models"][role], 40001)
        parsed = server.build_parser().parse_args(command[3:])
        assert parsed.cvom_admission and parsed.checkpoint == "/candidate" and parsed.writer_checkpoint == "/writer"
        assert parsed.cvom_fifo == (role == "fifo") and parsed.memory_off == (role == "memory-off")
        assert not parsed.semantic_memory and parsed.feature_precision == "native"
    baseline = ev.server_command(args, m["models"]["baseline"], 40001)
    assert "--cvom-admission" not in baseline and "--writer-checkpoint" not in baseline


def test_runtime_hash_verification_covers_parent_writer_and_sources():
    m = admission_manifest()
    m["base_file_sha256"] = {}
    digests = {name: "a" * 64 for name in ("checkpoint.json", "model.safetensors", "expert.safetensors")}
    digests.update({"cvom_admission.json": "b" * 64, "writer.safetensors": "c" * 64})
    digests.update({name: "d" * 64 for name in policy_module.CVOM_RUNTIME_SOURCES})
    with patch.object(ev, "file_hash", side_effect=lambda path: digests[path.name]):
        ev.verify_runtime_inputs(m)
        for name in ("expert.safetensors", "writer.safetensors", "cvom_admission_core.py"):
            saved, digests[name] = digests[name], "f" * 64
            with unittest.TestCase().assertRaisesRegex(ValueError, name):
                ev.verify_runtime_inputs(m)
            digests[name] = saved


def _actor(*, off=False, fifo=False, keep=False):
    from tests.test_feature_precision_v19_policy import actor
    with torch.random.fork_rng(devices=[]):
        result = actor()
        config = replace(result.representation.config, capacity_events=32)
        representation = RepresentationMemoryV18(config).eval().requires_grad_(False)
        state = result.representation.state_dict()
        # The archive reader does not use recurrent-slot addresses, but its
        # state schema still contains this capacity-sized parameter.
        state["memory.slot_addresses"] = state["memory.slot_addresses"].repeat(11, 1)[:32]
        representation.load_state_dict(state)
        result.representation = representation
        writer = CVoMAdmission(AdmissionConfig(dim=config.hidden_dim)).eval().requires_grad_(False)
        if keep:
            with torch.no_grad():
                writer.utility_head.bias.fill_(-1)
        result.cvom_admission = True
        result.cvom_writer_sha256 = "c" * 64
        result.cvom_manifest_sha256 = "b" * 64
        result.cvom_parent_identity = admission_manifest()["models"]["memory"]["cvom_admission"]["manifest"]["parent_identity"]
        result.cvom_source_sha256 = {name: "d" * 64 for name in policy_module.CVOM_RUNTIME_SOURCES}
        result.writer, result.writer_callback = writer, None if fifo else writer.make_policy()
        result.writer_sha256 = None if fifo else result.cvom_writer_sha256
        result.write_policy = "fifo" if fifo else "cvom-admission"
        result.memory_off = off
    return result


def _calls(policy, count=36):
    from tests import test_long_memory_online_policy as fixture
    torch.set_num_threads(1)
    results = []
    for index in range(count):
        controls = np.zeros((2, 8), np.float32) if index > 2 else None
        action, info = policy.get_action(fixture._observation(marker=index + 1),
            fixture._options(frame=2 * index, passive=index < 2, controls=controls))
        session = policy.sessions["A"]
        results.append((action, info, session.long_bank.clone(), session.short_cache.clone(), session.generator.get_state().clone()))
    return results


def test_zero_writer_matches_fifo_actions_bank_and_rng_after_capacity():
    learned, fifo = _calls(_actor()), _calls(_actor(fifo=True))
    for left, right in zip(learned, fifo):
        for key in left[0]:
            np.testing.assert_array_equal(left[0][key], right[0][key])
        for i in (2, 3, 4):
            assert torch.equal(left[i], right[i])
        assert left[1]["cvom_parent_identity"] == right[1]["cvom_parent_identity"]
    for output in (learned, fifo):
        diag = output[-1][1]["long_memory"]
        assert (diag["appends"], diag["replacements"], diag["keeps"], diag["updates"]) == (32, 4, 0, 36)
        assert diag["writer_decision"] == "replace:0" and diag["memory_tokens"] == 128


def test_read_off_preserves_writes_short_cache_and_rng_with_keep_after_full():
    on, off = _calls(_actor(keep=True)), _calls(_actor(off=True, keep=True))
    progress = {}
    for left, right in zip(on, off):
        for i in (2, 3, 4):
            assert torch.equal(left[i], right[i])
        memory = right[1]["long_memory"]
        assert memory["memory_read_enabled"] is False
        assert memory["read"]["ae_conditioning_delta_norm"] == 0
        progress = ev._validate_cvom_decision(memory, progress, passive=memory["passive"], fifo=False, capacity=32, num_tokens=4)
    assert (progress["appends"], progress["keeps"], progress["replacements"]) == (32, 4, 0)


def test_completed_runtime_evidence_rejects_wrong_writer_and_fake_decision_counters():
    m = admission_manifest()
    outputs = _calls(_actor(off=True, keep=True))
    model = m["models"]["memory-off"]
    # The tiny actor uses step6072 and a distinct payload inventory; bind it.
    model["step"] = outputs[0][1]["checkpoint_step"]
    model["representation_config"]["representation"] = outputs[0][1]["representation"]
    model["training_metadata"]["payload_sha256"] = outputs[0][1]["payload_sha256"]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "memory-off" / "BinFill"
        folder.mkdir(parents=True)
        (folder / "simulation_results.csv").write_text("episode_idx,episode_seed,success\n0,6,1\n")
        records = [{"kind": "policy_call", "session_id": "A", "episode_idx": 0, "episode_seed": 6,
            "passive": value[1]["long_memory"]["passive"], "info": value[1]} for value in outputs]
        records.append({"kind": "episode_complete", "session_id": "A", "episode_idx": 0, "episode_seed": 6, "success": 1})
        def write():
            (folder / "memory_diagnostics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        write()
        report = ev.completed_read_diagnostics(tmp, "memory-off", m)
        assert report["complete_evidence"] and report["writer_decisions"]["keeps"] == 4
        records[0]["info"]["cvom_writer_sha256"] = "f" * 64
        write()
        with unittest.TestCase().assertRaisesRegex(ValueError, "identity differs"):
            ev.completed_read_diagnostics(tmp, "memory-off", m)
        records[0]["info"]["cvom_writer_sha256"] = "c" * 64
        records[-2]["info"]["long_memory"]["keeps"] += 1
        write()
        with unittest.TestCase().assertRaisesRegex(ValueError, "counter differs"):
            ev.completed_read_diagnostics(tmp, "memory-off", m)


def test_constructor_loads_same_frozen_parent_and_fifo_never_installs_callback():
    from run_scripts.robomme import cvom_admission_checkpoint as checkpoint
    from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18
    config = RepresentationConfigV18(feature_dim=8, state_dim=8, num_short_tokens=4,
        hidden_dim=16, num_heads=4, capacity_events=32, short_window=3)
    info = {"step": 6072, "metadata": {"payload_sha256": {}}, "config": {
        "expert": {"rank": 8, "alpha": 16.}, "expert_targets": [], "representation": asdict(config)}}
    sidecar = admission_manifest()["models"]["memory"]["cvom_admission"]["manifest"]
    writer = CVoMAdmission(AdmissionConfig(dim=16)).eval().requires_grad_(False)
    snapshots = []
    def base_init(self, *args, **kwargs):
        head = torch.nn.Module()
        head.memory_transformer = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
        self.model = SimpleNamespace(action_head=head, config=SimpleNamespace(memory_window=3))
        self.processor, self.n_q = SimpleNamespace(max_state_dim=8), 4
    def load_parent(*args):
        snapshots.append(torch.get_rng_state().clone())
    def load_writer(*args, **kwargs):
        torch.rand(19)  # deliberately exercise sidecar RNG isolation
        return writer, sidecar
    with patch.object(policy_module, "checkpoint_info_v18", return_value=info), \
            patch.object(policy_module.LongMemoryPolicy, "__init__", base_init), \
            patch.object(policy_module, "install_expert_lora"), patch.object(policy_module, "set_expert_trainable"), \
            patch.object(policy_module, "load_checkpoint_v18", side_effect=load_parent) as parent_load, \
            patch.object(checkpoint, "load_controller", side_effect=load_writer) as writer_load, \
            patch("run_scripts.robomme.train_archive_deployment_v9.file_hash", return_value="d" * 64):
        for fifo in (False, True):
            policy = policy_module.RepresentationPolicyV18("/base", "/parent", device="cpu",
                writer_checkpoint="/writer", cvom_admission=True, cvom_fifo=fifo)
            assert parent_load.call_args.args[0] == "/parent"
            assert writer_load.call_args.args == ("/writer", "/parent")
            assert writer_load.call_args.kwargs == {"device": "cpu", "base_model": "/base"}
            assert (policy.writer_callback is None) == fifo
            assert not any(parameter.requires_grad for parameter in policy.representation.parameters())
            assert not any(parameter.requires_grad for parameter in policy.model.action_head.parameters())
            assert torch.equal(torch.get_rng_state(), snapshots[-1])


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in sorted(globals().items())
                              if name.startswith("test_") and callable(fn))
