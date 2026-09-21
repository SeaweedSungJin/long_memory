"""Small synthetic fixtures; no model or simulator is loaded by these tests."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_scripts.robomme import compare_feature_precision_v19 as cmp
from run_scripts.robomme import eval_representation_v18 as ev
from run_scripts.robomme.feature_precision_v19 import feature_precision_contract


def rehash(m):
    m["evaluation_id"] = ev.identity_digest(m)
    return m


def fixtures():
    baseline = dict(base_model={"path": "/base", "identity": "base"}, memory_checkpoint=None,
        mode="none", write_policy="none", write_policy_override="checkpoint", memory_off=False,
        archive_read_off=False, server_script=ev.BASELINE_SERVER)
    candidate = dict(base_model=baseline["base_model"], memory_checkpoint=str(cmp.REPO_ROOT / cmp.CHECKPOINT),
        mode=ev.VARIANT, step=6072, memory_off=False, archive_read_off=False, server_script=ev.SERVER,
        write_policy="fifo", checkpoint_files_sha256={name: "a"*64 for name in
            ("checkpoint.json", "model.safetensors", "expert.safetensors")})
    source = {name: "old" for name in cmp.CHANGED_SOURCES}
    source["gr00t/example.py"] = "untouched"
    native = dict(format_version=1, trainer_variant=ev.VARIANT, models=dict(baseline=baseline, memory=candidate),
        settings=dict(tasks=cmp.TASKS, n_episodes=10, dataset="val", seed=6, n_action_steps=16,
            max_episode_steps=1300, save_videos=False, device="cuda:0"), source_sha256=source)
    for key in ("base_file_sha256", "policy_package_versions", "benchmark", "server_python", "robomme_python",
                "selection_protocol", "control_contract", "allow_initialization_checkpoints"):
        native[key] = {"same": key}
    native["allow_initialization_checkpoints"] = False
    aligned = copy.deepcopy(native)
    aligned["source_sha256"].update({name: "new" for name in cmp.CHANGED_SOURCES | cmp.ADDED_SOURCES})
    aligned["models"]["memory"].update(feature_precision="cache-aligned",
        feature_precision_rules=feature_precision_contract("cache-aligned"))
    return rehash(native), rehash(aligned)


def test_explicit_candidate_option_does_not_reach_baseline():
    args = ev.build_parser().parse_args(["--models", "baseline", "--feature-precision", "cache-aligned"])
    assert ev.build_parser().parse_args([]).feature_precision == "native"
    _, aligned = fixtures()
    command = ev.server_command(args, aligned["models"]["memory"], 12345)
    assert command[command.index("--feature-precision") + 1] == "cache-aligned"
    assert "--feature-precision" not in ev.server_command(args, aligned["models"]["baseline"], 12345)


def test_only_precision_and_reviewed_source_delta_allowed():
    native, aligned = fixtures()
    changes = cmp.validate_pair(native, aligned)
    assert set(changes["changed"]) == cmp.CHANGED_SOURCES
    for mutate, message in (
        (lambda x: x["source_sha256"].update({"gr00t/example.py": "altered"}), "Unreviewed"),
        (lambda x: x["source_sha256"].update({"run_scripts/robomme/new_policy.py": "new"}), "Unreviewed"),
        (lambda x: x["models"]["memory"].update(training_config={"loss": "new"}), "beyond feature"),
        (lambda x: x["settings"].update(seed=7), "fixed to"),
        (lambda x: x.update(benchmark={"different": "environment"}), "environment/protocol"),
    ):
        bad = copy.deepcopy(aligned)
        mutate(bad)
        with unittest.TestCase().assertRaisesRegex(ValueError, message):
            cmp.validate_pair(native, rehash(bad))


def test_manifest_rules_are_checked_and_legacy_native_is_read_only():
    native, aligned = fixtures()
    ev.validate_manifest_contract(native)
    bad = copy.deepcopy(aligned)
    bad["models"]["memory"]["feature_precision_rules"]["ae_denoising"] = "global autocast"
    with unittest.TestCase().assertRaisesRegex(ValueError, "precision rules"):
        ev.validate_manifest_contract(rehash(bad))
    bad = copy.deepcopy(aligned)
    bad["models"]["baseline"]["feature_precision"] = "cache-aligned"
    with unittest.TestCase().assertRaisesRegex(ValueError, "baseline precision"):
        ev.validate_manifest_contract(rehash(bad))


def test_source_removal_never_allowed():
    native, aligned = fixtures()
    aligned["source_sha256"].pop("gr00t/example.py")
    with unittest.TestCase().assertRaisesRegex(ValueError, "Unreviewed"):
        cmp.validate_pair(native, rehash(aligned))


def _regression_checks(tmp_path):
    native, aligned = fixtures()
    path = tmp_path / "native.json"
    path.write_text(json.dumps(native))
    report = tmp_path / "completed.json"
    record = dict(format_version=1, kind="v19_feature_precision_regression", passed=True,
        checks={name: True for name in cmp.REGRESSION_CHECKS},
        checkpoint_files_sha256=aligned["models"]["memory"]["checkpoint_files_sha256"],
        historical_native_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_sha256={name: "new" for name in cmp.CHANGED_SOURCES | cmp.ADDED_SOURCES})
    for name in cmp.REGRESSION_EVIDENCE:
        (tmp_path / name).write_text("diagnostic evidence")
    record["evidence_files_sha256"] = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in cmp.REGRESSION_EVIDENCE}
    original_hash = cmp.file_hash
    def hashes(value):
        return "new" if str(value).endswith(tuple(cmp.CHANGED_SOURCES | cmp.ADDED_SOURCES)) else original_hash(value)
    with patch.object(cmp, "file_hash", side_effect=hashes):
        report.write_text(json.dumps(record))
        assert cmp.validate_regression(report, path, aligned)["checks"] == record["checks"]
        bad = copy.deepcopy(record)
        bad.pop("evidence_files_sha256")
        report.write_text(json.dumps(bad))
        with unittest.TestCase().assertRaisesRegex(ValueError, "nonempty"):
            cmp.validate_regression(report, path, aligned)
        for key in cmp.REGRESSION_CHECKS:
            bad = copy.deepcopy(record)
            bad["checks"][key] = False
            report.write_text(json.dumps(bad))
            with unittest.TestCase().assertRaisesRegex(ValueError, "Regression must pass"):
                cmp.validate_regression(report, path, aligned)
        bad = copy.deepcopy(record)
        bad["historical_native_manifest_sha256"] = "wrong"
        report.write_text(json.dumps(bad))
        with unittest.TestCase().assertRaisesRegex(ValueError, "historical native manifest"):
            cmp.validate_regression(report, path, aligned)


def test_regression_requires_each_check_and_exact_source_binding():
    with tempfile.TemporaryDirectory() as directory:
        _regression_checks(Path(directory))


def test_paired_counts_and_flip_identity_not_totals():
    native, aligned = {}, {}
    for target in (native, aligned):
        target.update(contexts={task: {"scenario_metadata_sha256": "same"} for task in cmp.TASKS},
            rows={task: {i: dict(success=0, episode_seed=i+10, scenario_seed=str(i+20),
                task_instruction="same", status="fail", steps="100") for i in range(10)} for task in cmp.TASKS})
    native["rows"]["BinFill"][0]["success"] = 1
    aligned["rows"]["BinFill"][1]["success"] = 1
    aligned["rows"]["MoveCube"][3]["success"] = 1
    native["successes"], aligned["successes"] = 1, 2
    result = cmp.paired_summary(native, aligned, 1000)
    assert (result["wins"], result["losses"], result["same"], result["n"]) == (2, 1, 157, 160)
    assert result["delta"] == 1/160 and len(result["flips"]) == 3
    assert result["ci95"][0] <= 0 <= result["ci95"][1]
    aligned["rows"]["MoveCube"][3]["episode_seed"] = 99
    with unittest.TestCase().assertRaisesRegex(ValueError, "episode_seed differs"):
        cmp.paired_summary(native, aligned, 1000)


class FeaturePrecisionComparisonTests(unittest.TestCase):
    pass


for _name, _function in list(globals().items()):
    if _name.startswith("test_") and callable(_function):
        setattr(FeaturePrecisionComparisonTests, _name, lambda self, check=_function: check())
del _name, _function


if __name__ == "__main__":
    unittest.main()
