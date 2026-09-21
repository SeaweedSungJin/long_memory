"""No GPU/model/simulator: fixed VAL160 and semantic-sidecar deployment guards."""
from __future__ import annotations

import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
from contextlib import redirect_stderr
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from run_scripts.robomme import eval_representation_v18 as ev
from run_scripts.robomme import eval_semantic_memory as fixed
from run_scripts.robomme import serve_representation_v18 as server
from run_scripts.robomme.feature_precision_v19 import feature_precision_contract
from run_scripts.robomme.semantic_memory_storage import StorageConfig, StorageManager
from tests.test_representation_eval_v18 import manifest as original_manifest, rehash


def semantic_manifest(*, stage=2, storage=True):
    result = original_manifest()
    manifest = {"format_version": 1, "variant": "semantic_memory_v1", "stage": stage,
        "actor_payload_sha256": result["models"]["memory"]["training_metadata"]["payload_sha256"],
        "storage_config": asdict(StorageConfig(capacity_events=3, num_tokens=4, dim=16)) if storage else None,
        "payload_sha256": {"answers.safetensors": "b" * 64},
        "metadata": {"labels_at_inference": False}}
    if storage:
        manifest["payload_sha256"]["storage.safetensors"] = "c" * 64
    block = {"manifest": manifest, "files_sha256": {"semantic.json": "d" * 64, **manifest["payload_sha256"]}}
    for role in ("memory", "memory-off"):
        model = result["models"][role]
        model["semantic_memory"] = copy.deepcopy(block)
        model["representation_config"].update(capacity_events=3, num_short_tokens=4, hidden_dim=16)
        model["write_policy"] = "semantic-cvom" if storage else "fifo"
        model["feature_precision"] = "native"
        model["feature_precision_rules"] = feature_precision_contract("native")
    if storage:
        result["models"]["fifo"] = copy.deepcopy(result["models"]["memory"])
        result["models"]["fifo"]["write_policy"] = "fifo"
    return rehash(result)


def test_wrapper_pins_exact_development_panel_and_reuses_reference():
    args = fixed.build_parser().parse_args(["--checkpoint", "/candidate", "--output-dir", "/output"])
    parsed = ev.build_parser().parse_args(fixed.evaluator_arguments(args))
    ev.validate_options(parsed)
    assert parsed.tasks == list(ev.TASKS) and len(parsed.tasks) == 16
    assert parsed.n_episodes == 10 and parsed.dataset == "val" and parsed.seed == 6
    assert parsed.n_action_steps == 16 and parsed.max_episode_steps == 1300
    assert parsed.feature_precision == "native" and parsed.semantic_memory
    assert parsed.models == ["baseline", "memory"]
    assert parsed.baseline_reference.name == "archive_read_best1250_val_n10_seed6"
    for flag, value in (("--seed", "8"), ("--tasks", "PatternLock"), ("--dataset", "test"),
                        ("--n-episodes", "3"), ("--feature-precision", "cache-aligned")):
        with redirect_stderr(io.StringIO()), unittest.TestCase().assertRaises(SystemExit):
            fixed.build_parser().parse_args(["--output-dir", "/output", flag, value])


def test_wrapper_report_only_needs_no_checkpoint_and_preflight_creates_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "absent"
        args = fixed.build_parser().parse_args(["--output-dir", str(out), "--report-only"])
        assert "--report-only" in fixed.evaluator_arguments(args)
        with patch.object(ev, "check_dependencies"), patch.object(ev, "build_identity", return_value=semantic_manifest()), \
                patch.object(ev, "run_evaluation", side_effect=AssertionError("preflight started rollout")):
            assert fixed.main(["--checkpoint", "/candidate", "--output-dir", str(out), "--preflight-only"]) == 0
        assert not out.exists()


def test_semantic_model_manifest_is_strict_and_legacy_contract_still_works():
    ev.validate_manifest_contract(original_manifest())
    ev.validate_manifest_contract(semantic_manifest())
    ev.validate_manifest_contract(semantic_manifest(stage=1, storage=False))
    # A continued-actor-training FIFO control can legitimately be Stage 2.
    ev.validate_manifest_contract(semantic_manifest(stage=2, storage=False))
    for change in ("actor", "capacity", "hash", "label", "stage", "legacy", "precision", "off"):
        m = semantic_manifest()
        model = m["models"]["memory"]
        sem = model["semantic_memory"]["manifest"]
        if change == "actor":
            sem["actor_payload_sha256"] = {}
        elif change == "capacity":
            sem["storage_config"]["capacity_events"] += 1
        elif change == "hash":
            model["semantic_memory"]["files_sha256"]["storage.safetensors"] = "f" * 64
        elif change == "label":
            sem["metadata"]["labels_at_inference"] = True
        elif change == "stage":
            sem["stage"] = True
        elif change == "legacy":
            model["writer_checkpoint"] = "/different/writer"
        elif change == "precision":
            model["feature_precision"] = "cache-aligned"
        else:
            m["models"]["memory-off"]["semantic_memory"]["manifest"]["metadata"]["extra"] = 1
        with unittest.TestCase().assertRaises(ValueError):
            ev.validate_manifest_contract(rehash(m))


def test_smoke_bundle_requires_explicit_diagnostic_opt_in():
    m = semantic_manifest()
    for role in ("memory", "memory-off", "fifo"):
        m["models"][role]["semantic_memory"]["manifest"]["metadata"]["smoke_only"] = True
    with unittest.TestCase().assertRaisesRegex(ValueError, "Smoke-only"):
        ev.validate_manifest_contract(rehash(m))
    m["allow_initialization_checkpoints"] = True
    ev.validate_manifest_contract(rehash(m))


def test_fifo_and_read_off_dispatch_preserve_semantic_identity():
    m = semantic_manifest()
    args = ev.build_parser().parse_args(["--models", "baseline"])
    for role in ("memory", "memory-off", "fifo"):
        command = ev.server_command(args, m["models"][role], 40001)
        assert "--semantic-memory" in command
        assert ("--semantic-fifo" in command) == (role == "fifo")
        assert ("--memory-off" in command) == (role == "memory-off")
        assert "--writer-checkpoint" not in command
        parsed = server.build_parser().parse_args(command[3:])
        assert parsed.checkpoint == "/candidate" and parsed.feature_precision == "native"
    command = ev.server_command(args, m["models"]["baseline"], 40001)
    assert "--semantic-memory" not in command and "--checkpoint" not in command


def test_runtime_file_verification_covers_semantic_payloads():
    m = semantic_manifest()
    m.update(source_sha256={}, base_file_sha256={})
    digests = {"checkpoint.json": "a"*64, "model.safetensors": "a"*64,
               "expert.safetensors": "a"*64, "answers.safetensors": "b"*64,
               "storage.safetensors": "c"*64, "semantic.json": "d"*64}
    with patch.object(ev, "file_hash", side_effect=lambda path: digests[path.name]):
        ev.verify_runtime_inputs(m)
        digests["storage.safetensors"] = "f"*64
        with unittest.TestCase().assertRaisesRegex(ValueError, "storage.safetensors"):
            ev.verify_runtime_inputs(m)


def test_runtime_read_off_evidence_requires_same_storage_hash():
    m = semantic_manifest()
    model = m["models"]["memory-off"]
    sem = model["semantic_memory"]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "memory-off" / "BinFill"
        folder.mkdir(parents=True)
        (folder / "simulation_results.csv").write_text("episode_idx,episode_seed,success\n0,6,1\n")
        info = {"checkpoint_variant": ev.VARIANT, "checkpoint_step": model["step"],
            "representation": model["representation_config"]["representation"], "memory_off": True,
            "payload_sha256": model["training_metadata"]["payload_sha256"],
            "writer_sha256": "c"*64, "semantic_manifest_sha256": "d"*64,
            "semantic_payload_sha256": sem["manifest"]["payload_sha256"], "semantic_stage": 2,
            "storage_manager_sha256": "c"*64, "feature_precision": "native",
            "feature_precision_rules": feature_precision_contract("native"),
            "long_memory": {"policy": "semantic-cvom", "memory_read_enabled": False,
                            "read": {"ae_conditioning_delta_norm": 0.0}}}
        call = {"kind": "policy_call", "session_id": "A", "episode_idx": 0, "episode_seed": 6,
                "passive": False, "info": info}
        end = {"kind": "episode_complete", "session_id": "A", "episode_idx": 0, "episode_seed": 6, "success": 1}
        def write():
            (folder / "memory_diagnostics.jsonl").write_text(json.dumps(call)+"\n"+json.dumps(end)+"\n")
        write()
        assert ev.completed_read_diagnostics(Path(tmp), "memory-off", m)["complete_evidence"]
        info["storage_manager_sha256"] = "f"*64
        write()
        with unittest.TestCase().assertRaisesRegex(ValueError, "identity differs"):
            ev.completed_read_diagnostics(Path(tmp), "memory-off", m)


def test_zero_semantic_manager_online_actions_bank_rng_equal_native_fifo():
    from tests.test_feature_precision_v19_policy import actor, calls
    native, semantic = actor(), actor()
    storage = StorageManager(StorageConfig(capacity_events=3, num_tokens=4, dim=16)).eval().requires_grad_(False)
    semantic.semantic_memory = True
    semantic.stage = 2
    semantic.semantic_manifest_sha256 = "d"*64
    semantic.semantic_payload_sha256 = {"answers.safetensors": "b"*64, "storage.safetensors": "c"*64}
    semantic.storage_manager_sha256 = semantic.writer_sha256 = "c"*64
    semantic.writer_callback = storage.make_policy()
    semantic.write_policy = "semantic-cvom"
    first, second = calls(native), calls(semantic)
    for lhs, rhs in zip(first, second):
        for key in lhs[0]:
            assert (lhs[0][key] == rhs[0][key]).all()
        for i in (2, 3, 4):
            assert torch.equal(lhs[i], rhs[i])  # short cache / bank / AE RNG
        assert rhs[1]["semantic_manifest_sha256"] == "d"*64
        assert rhs[1]["storage_manager_sha256"] == "c"*64
        assert rhs[1]["payload_sha256"] == lhs[1]["payload_sha256"]


def test_semantic_online_read_off_keeps_same_writes_and_seed_progression():
    from tests.test_feature_precision_v19_policy import actor, calls
    first, second = actor(), actor()
    second.memory_off = True
    manager = StorageManager(StorageConfig(capacity_events=3, num_tokens=4, dim=16)).eval().requires_grad_(False)
    for policy in (first, second):
        policy.writer_callback = manager.make_policy()
        policy.write_policy = "semantic-cvom"
    on, off = calls(first), calls(second)
    for a, b in zip(on, off):
        for i in (2, 3, 4):
            assert torch.equal(a[i], b[i])
        assert b[1]["long_memory"]["memory_read_enabled"] is False
        assert b[1]["long_memory"]["read"]["ae_conditioning_delta_norm"] == 0


def test_constructor_rejects_semantic_legacy_or_precision_mix_before_loading():
    from run_scripts.robomme.policy_representation_v18 import RepresentationPolicyV18
    with unittest.TestCase().assertRaisesRegex(ValueError, "native precision"):
        RepresentationPolicyV18("missing", "missing", semantic_memory=True, writer_checkpoint="legacy")
    with unittest.TestCase().assertRaisesRegex(ValueError, "native precision"):
        RepresentationPolicyV18("missing", "missing", semantic_memory=True, feature_precision="cache-aligned")
    with unittest.TestCase().assertRaisesRegex(ValueError, "requires semantic_memory"):
        RepresentationPolicyV18("missing", "missing", semantic_fifo=True)


def test_real_policy_diagnostic_compares_writer_rng_not_action_equality():
    from run_scripts.robomme.verify_semantic_memory import PANEL, compare_passes
    assert PANEL == ((1355, 72), (626, 6))
    on = [{"episode_id": 1355, "decision": 0, "frame": 0, "bank": "same",
           "session_rng": "rng", "write_inserted": False, "write_operation_code": 1,
           "keeps": 1, "generated_action": "on-action"}]
    off = copy.deepcopy(on)
    off[0]["generated_action"] = "off-action"
    assert compare_passes(on, off)["generated_actions_changed"] == 1
    # KEEP is permitted, but switching KEEP to INSERT only in READ-off is not.
    off[0]["write_inserted"] = True
    with unittest.TestCase().assertRaises(AssertionError):
        compare_passes(on, off)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in sorted(globals().items())
                              if name.startswith("test_") and callable(fn))


if __name__ == "__main__":
    unittest.main()
