"""Explicit baseline reuse for V18, with an audited unchanged server profile.

Old references keep the existing V10 validator. A freshly evaluated V18
baseline is also accepted, but only when it ran that EXACT reviewed V10 server.
No CSV is copied and no old evaluation ID is rewritten.
"""
from __future__ import annotations
import fcntl
import json
from pathlib import Path

from run_scripts.robomme import baseline_reference_v10 as old
from run_scripts.robomme.eval_long_memory_comparison import file_hash, TASKS
from gr00t.eval.sim.robomme.compare_long_memory_results import read_results, validate_result_identity


def equivalent_v18(reference, current):
    if reference.get("trainer_variant") != "representation_v18" or reference.get("format_version") != 1:
        raise ValueError("Unsupported V18 baseline reference")
    if reference.get("evaluation_id") != old._digest(reference) or reference.get("baseline_reference") is not None:
        raise ValueError("Changed or nested baseline reference; use the original completed rollout")
    previous, new = old._original_baseline(reference), old._original_baseline(current)
    server, digest = old.SERVERS["archive_projector_v10"]
    if previous.get("server_script") != server or new.get("server_script") != server:
        raise ValueError("V18 baseline must run the unchanged reviewed V10 baseline entrypoint")
    if previous["base_model"] != new["base_model"]:
        raise ValueError("Original HAMLET base differs")
    settings = current.get("settings", {})
    if set(settings) != old.SETTINGS or reference.get("settings") != settings:
        raise ValueError("Baseline settings/tasks/counts differ")
    if (not settings["tasks"] or len(set(settings["tasks"])) != len(settings["tasks"])
            or any(task not in TASKS for task in settings["tasks"])
            or type(settings["n_episodes"]) is not int or settings["n_episodes"] <= 0):
        raise ValueError("Invalid task/episode selection")
    for key in ("base_file_sha256", "policy_package_versions", "benchmark", "server_python", "robomme_python"):
        if not reference.get(key) or reference[key] != current.get(key):
            raise ValueError(f"Baseline {key} changed")
    before, after = reference.get("source_sha256", {}), current.get("source_sha256", {})
    required = {"gr00t/long_memory/online_policy_v7.py", "gr00t/long_memory/online_policy.py",
                "gr00t/eval/sim/robomme/run_long_memory_rollout.py", "gr00t/model/gr00t_n1d6/gr00t_n1d6.py"}
    old_core = {k: v for k, v in before.items() if k.startswith("gr00t/")}
    new_core = {k: v for k, v in after.items() if k.startswith("gr00t/")}
    if not required <= set(new_core) or old_core != new_core:
        raise ValueError("Original HAMLET/rollout source closure changed")
    helper = "run_scripts/robomme/eval_long_memory_comparison.py"
    if not before.get(helper) or before[helper] != after.get(helper):
        raise ValueError("Original evaluation protocol helper changed")
    if before.get(server) != digest or after.get(server) != digest:
        raise ValueError("Baseline server is not the reviewed no-adapter source")
    return server


def _inspect_v18(root, current):
    root = Path(root).resolve()
    reference = json.loads((root / "comparison_manifest.json").read_text())
    if reference.get("trainer_variant") != "representation_v18":
        return old._inspect(root, current)
    server = equivalent_v18(reference, current)
    status_path = root / "driver_status.json"
    status = json.loads(status_path.read_text())
    if status.get("interrupted") is not False or status.get("failures") != []:
        raise ValueError("Baseline driver did not complete cleanly")
    lock_path = root / ".driver.lock"
    if lock_path.exists():
        with lock_path.open("rb") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Baseline evaluation is still running") from exc
    files = {"comparison_manifest.json": file_hash(root / "comparison_manifest.json"),
             "driver_status.json": file_hash(status_path)}
    settings = current["settings"]
    model_config = Path(current["models"]["baseline"]["base_model"]["path"]) / "config.json"
    if file_hash(model_config) != current["base_file_sha256"].get("config.json"):
        raise ValueError("Original model configuration changed")
    config = json.loads(model_config.read_text())
    for task in settings["tasks"]:
        csv = root / "baseline" / task / "simulation_results.csv"
        rows = read_results(csv, expected=settings["n_episodes"])
        if len(rows) != settings["n_episodes"] or any(not r.get("scenario_seed") or not r.get("task_instruction") for r in rows.values()):
            raise ValueError(f"Baseline incomplete or missing scenario identity: {task}")
        policy = validate_result_identity(root, "baseline", task, reference)
        scenarios, scenario_digest = old._scenario_context(current["benchmark"]["source_and_scenario_sha256"], settings["dataset"], task)
        if (len(scenarios) < settings["n_episodes"] or policy.get("scenario_metadata_sha256") != scenario_digest
                or any(int(row["scenario_seed"]) != scenarios[index]["seed"] for index, row in rows.items())
                or policy.get("model_config_sha256") != current["base_file_sha256"]["config.json"]
                or policy.get("memory_window") != config.get("memory_window", 4)
                or policy.get("demo_sampling") != "backward_aligned_full_history"):
            raise ValueError(f"Baseline scenario/preprocessing identity differs: {task}")
        for path in (csv, csv.parent / "policy_manifest.json"):
            if not path.resolve().is_relative_to(root):
                raise ValueError("Baseline artifact escapes original run")
            files[str(path.relative_to(root))] = file_hash(path)
    record = {"format_version": 1, "kind": "completed_original_baseline_reference", "source_run": str(root),
        "source_evaluation_id": reference["evaluation_id"], "role": "baseline", "policy_profile": old.PROFILE,
        "source_server": server, "files_sha256": files,
        "completed_episodes": settings["n_episodes"] * len(settings["tasks"]), "newly_rolled_out": 0}
    return record, reference


def build_reference(root, current):
    return _inspect_v18(root, current)[0]


def validate_reference(record, current):
    if not isinstance(record, dict) or record.get("kind") != "completed_original_baseline_reference":
        raise ValueError("Invalid baseline reference")
    fresh, manifest = _inspect_v18(record["source_run"], current)
    if fresh != record:
        raise ValueError("Baseline reference changed since binding")
    return Path(record["source_run"]).resolve(), manifest
