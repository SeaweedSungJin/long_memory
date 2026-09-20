"""Read-only, fail-closed reuse of a completed original-HAMLET baseline.

Only the two reviewed baseline server implementations are eligible. Reference
CSVs and their ORIGINAL policy/evaluation IDs are never copied or rewritten.
This is evidence reuse, not a new rollout or an independent replication.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

from gr00t.eval.sim.robomme.compare_long_memory_results import read_results, validate_result_identity
from run_scripts.robomme.eval_long_memory_comparison import TASKS, file_hash


PROFILE = "original_long_memory_v7_no_adapters_v1"
# These reviewed entrypoints dispatch baseline directly to LongMemoryV7Policy
# with no checkpoint, write_policy=checkpoint and memory_off=False. This is an
# explicit compatibility allowlist, NOT arbitrary matching of common filenames.
SERVERS = {
    "archive_read_control_v7": (
        "run_scripts/robomme/serve_archive_read_control_v7.py",
        "8f64313f1212b07e8494a32b677fc8f3f75a03f07b50190de2442daf671d2503"),
    "archive_projector_v10": (
        "run_scripts/robomme/serve_archive_projector_v10.py",
        "5bc7dc163f6f34c50005a1aadfdcd7d5fa58ea2153ecc0921b71ce6158f3f821"),
}
SETTINGS = {"tasks", "n_episodes", "dataset", "seed", "n_action_steps", "max_episode_steps", "save_videos", "device"}
CONTEXT = ("scenario_metadata_sha256", "model_config_sha256", "memory_window", "demo_sampling")


def _digest(manifest):
    return hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != "evaluation_id"},
                                    sort_keys=True).encode()).hexdigest()


def _original_baseline(manifest):
    model = manifest.get("models", {}).get("baseline", {})
    if (not isinstance(model, dict) or not model.get("base_model")
            or model.get("memory_checkpoint") is not None or model.get("mode") != "none"
            or model.get("stage", 0) != 0 or model.get("write_policy") != "none"
            or model.get("write_policy_override") != "checkpoint"
            or model.get("memory_off") is not False or model.get("archive_read_off") is not False):
        raise ValueError("Baseline reference must be original HAMLET with no memory/adapters")
    forbidden = {"checkpoint_sha256", "weights_sha256", "expert_weights_sha256", "projector_weights_sha256",
                 "cvom_weights_sha256", "semantic_state_sha256", "projector_config", "training_metadata",
                 "checkpoint_files_sha256", "training_state_sha256"}
    if set(model) & forbidden:
        raise ValueError("Baseline reference contains adapted-model provenance")
    return model


def _equivalent(reference, current):
    variant = reference.get("trainer_variant")
    if reference.get("format_version") != 1 or variant not in SERVERS:
        raise ValueError("Unsupported baseline-reference evaluation/server profile")
    if reference.get("evaluation_id") != _digest(reference):
        raise ValueError("Baseline-reference manifest identity digest changed")
    if reference.get("baseline_reference") is not None:
        raise ValueError("Nested baseline references are not supported; use the original completed rollout")
    old_model, new_model = _original_baseline(reference), _original_baseline(current)
    if old_model["base_model"] != new_model["base_model"]:
        raise ValueError("Baseline-reference original base identity differs")
    old, new = reference.get("settings", {}), current.get("settings", {})
    if set(old) != SETTINGS or set(new) != SETTINGS or old != new:
        raise ValueError("Baseline-reference settings/tasks/episode counts differ")
    if (not isinstance(new["tasks"], list) or not new["tasks"] or len(set(new["tasks"])) != len(new["tasks"])
            or any(task not in TASKS for task in new["tasks"])
            or type(new["n_episodes"]) is not int or new["n_episodes"] <= 0):
        raise ValueError("Invalid baseline-reference task/episode selection")
    for key in ("base_file_sha256", "policy_package_versions", "benchmark", "server_python", "robomme_python"):
        if not reference.get(key) or reference[key] != current.get(key):
            raise ValueError(f"Baseline-reference {key} differs or is missing")
    old_sources, new_sources = reference.get("source_sha256", {}), current.get("source_sha256", {})
    old_core = {k: v for k, v in old_sources.items() if k.startswith("gr00t/")}
    new_core = {k: v for k, v in new_sources.items() if k.startswith("gr00t/")}
    required = {"gr00t/long_memory/online_policy_v7.py", "gr00t/long_memory/online_policy.py",
                "gr00t/eval/sim/robomme/run_long_memory_rollout.py", "gr00t/model/gr00t_n1d6/gr00t_n1d6.py"}
    if not required <= set(new_core) or old_core != new_core:
        raise ValueError("Baseline-reference original policy/rollout source closure differs")
    utility = "run_scripts/robomme/eval_long_memory_comparison.py"
    if not old_sources.get(utility) or old_sources[utility] != new_sources.get(utility):
        raise ValueError("Baseline-reference evaluation protocol helper differs")
    old_server, old_hash = SERVERS[variant]
    new_server, new_hash = SERVERS["archive_projector_v10"]
    if old_sources.get(old_server) != old_hash or new_sources.get(new_server) != new_hash:
        raise ValueError("Baseline-reference server is not the reviewed no-adapter compatibility profile")
    return old_server


def _scenario_context(benchmark_files, dataset, task):
    """Verify raw bytes, then reproduce the rollout's *resolved-list* digest.

    BenchmarkEnvBuilder indexes records by (task, int(episode)), with later
    duplicates replacing earlier records. The rollout hashes ALL available
    episodes (not just n_episodes), after resolving integer seeds/difficulty.
    Keep this small parser simulator-free; the resolver and rollout sources
    remain covered by the required benchmark/original-policy provenance.
    """
    suffix = f"/env_metadata/{dataset}/record_dataset_{task}_metadata.json"
    matches = [(Path(path), digest) for path, digest in benchmark_files.items() if path.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Baseline-reference scenario metadata path is missing or ambiguous: {task}")
    path, raw_digest = matches[0]
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != raw_digest:
        raise ValueError(f"Baseline-reference scenario metadata raw file changed: {task}")
    metadata = json.loads(payload)
    default_task = str(metadata.get("env_id") or "").strip()
    index = {}
    for record in metadata.get("records", []):
        task_name = str(record.get("task") or default_task or "").strip()
        episode = record.get("episode")
        if not task_name or episode is None:
            continue
        try:
            episode = int(episode)
        except (TypeError, ValueError):
            continue
        index[(task_name, episode)] = record
    available = sum(key[0] == task for key in index)
    scenarios = []
    for episode in range(available):
        record = index.get((task, episode))
        if record is None or record.get("seed") is None:
            raise ValueError(f"Baseline-reference scenario metadata lacks episode/seed: {task}/{episode}")
        try:
            seed = int(record["seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Baseline-reference scenario seed is invalid: {task}/{episode}") from exc
        scenarios.append({"episode_idx": episode, "seed": seed, "difficulty": record.get("difficulty")})
    digest = hashlib.sha256(json.dumps(scenarios, sort_keys=True).encode()).hexdigest()
    return scenarios, digest


def _inspect(reference_root, current):
    root = Path(reference_root).resolve()
    manifest_path = root / "comparison_manifest.json"
    reference = json.loads(manifest_path.read_text())
    server = _equivalent(reference, current)
    status_path = root / "driver_status.json"
    status = json.loads(status_path.read_text())
    if status.get("interrupted") is not False or status.get("failures") != []:
        raise ValueError("Baseline reference must come from a clean completed driver")
    # Opening an existing lock read-only does not create/modify any artifact.
    # Do not reference a driver that is still modifying its results.
    lock_path = root / ".driver.lock"
    if lock_path.exists():
        with lock_path.open("rb") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Baseline-reference evaluation is still running") from exc
    files = {"comparison_manifest.json": file_hash(manifest_path), "driver_status.json": file_hash(status_path)}
    settings = current["settings"]
    expected = settings["n_episodes"]
    config = json.loads((Path(current["models"]["baseline"]["base_model"]["path"]) / "config.json").read_text())
    if file_hash(Path(current["models"]["baseline"]["base_model"]["path"]) / "config.json") != current["base_file_sha256"].get("config.json"):
        raise ValueError("Current original model configuration changed")
    benchmark_files = current["benchmark"].get("source_and_scenario_sha256", {})
    for task in settings["tasks"]:
        csv_path = root / "baseline" / task / "simulation_results.csv"
        rows = read_results(csv_path, expected=expected)
        if len(rows) != expected:
            raise ValueError(f"Baseline reference is incomplete: {task} {len(rows)}/{expected}")
        if any(not r.get("scenario_seed") or not r.get("task_instruction") for r in rows.values()):
            raise ValueError("Baseline reference requires explicit scenario seeds and task instructions")
        policy = validate_result_identity(root, "baseline", task, reference)
        if any(k not in policy for k in CONTEXT):
            raise ValueError("Baseline-reference policy context is incomplete")
        scenarios, scenario_digest = _scenario_context(benchmark_files, settings["dataset"], task)
        if len(scenarios) < expected or policy["scenario_metadata_sha256"] != scenario_digest:
            raise ValueError(f"Baseline-reference scenario metadata differs: {task}")
        if any(int(row["scenario_seed"]) != scenarios[episode]["seed"] for episode, row in rows.items()):
            raise ValueError(f"Baseline-reference result scenario seed differs: {task}")
        if (policy["model_config_sha256"] != current["base_file_sha256"]["config.json"]
                or policy["memory_window"] != config.get("memory_window", 4)
                or policy["demo_sampling"] != "backward_aligned_full_history"):
            raise ValueError(f"Baseline-reference original short/demo context differs: {task}")
        for path in (csv_path, csv_path.parent / "policy_manifest.json"):
            if not path.resolve().is_relative_to(root):
                raise ValueError("Baseline reference artifact escapes its source run")
            files[str(path.relative_to(root))] = file_hash(path)
    record = {"format_version": 1, "kind": "completed_original_baseline_reference", "source_run": str(root),
              "source_evaluation_id": reference["evaluation_id"], "role": "baseline", "policy_profile": PROFILE,
              "source_server": server, "files_sha256": files, "completed_episodes": expected * len(settings["tasks"]),
              "newly_rolled_out": 0}
    return record, reference


def build_reference(reference_root, current):
    """Read-only preflight: validate provenance and bind immutable result hashes."""
    return _inspect(reference_root, current)[0]


def validate_reference(record, current):
    """Revalidate external hashes and ORIGINAL policy IDs on reports/resume."""
    if not isinstance(record, dict) or record.get("kind") != "completed_original_baseline_reference":
        raise ValueError("Invalid explicit baseline-reference record")
    fresh, reference = _inspect(record["source_run"], current)
    if fresh != record:
        raise ValueError("Baseline-reference files/provenance changed since evaluation binding")
    return Path(record["source_run"]).resolve(), reference
