#!/usr/bin/env python3
"""Fixed development VAL160 for ECHO, with strict original-baseline reuse.

All fresh roles load one ECHO actor. FIFO changes storage only; READ-off keeps
the learned writer and every observation/action update. An old V19 FIFO result
is external context, never substituted for this same-actor FIFO control.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.safety_v5 import validate_output_scope
from gr00t.eval.sim.robomme.compare_long_memory_v3_results import _contrast
from run_scripts.robomme.eval_long_memory_comparison import (
    TASKS, benchmark_identity, bind_manifest, check_dependencies, file_hash,
    free_local_port, python_path, read_results, resolve_repo_path, run_client,
    server_ready, stop_process, validate_result_identity,
)
from run_scripts.robomme.baseline_reference_v18 import build_reference, validate_reference
from run_scripts.robomme.baseline_reference_v10 import _original_baseline
from run_scripts.robomme.report_baseline_reference_v10 import build_reference_report
from run_scripts.robomme.eval_representation_v18 import DEPENDENCIES, identity_digest
from run_scripts.robomme.feature_precision_v19 import feature_precision_contract
from run_scripts.robomme.policy_echo_cvom import RUNTIME_SOURCES, VARIANT
from run_scripts.robomme.train_cvom_admission import gpu_guard

SERVER = "run_scripts/robomme/serve_echo_cvom.py"
BASELINE_SERVER = "run_scripts/robomme/serve_archive_projector_v10.py"
CLIENT = "gr00t/eval/sim/robomme/run_long_memory_rollout.py"
ROLES = ("memory", "fifo", "memory-off")
PANEL = {"tasks": list(TASKS), "dataset": "val", "n_episodes": 10, "seed": 6,
         "n_action_steps": 16, "max_episode_steps": 1300}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--models", nargs="+", choices=ROLES, default=list(ROLES))
    p.add_argument("--base-model", type=Path, default=Path("checkpoints/author_hamlet_robomme/checkpoint-60000"))
    p.add_argument("--baseline-reference", type=Path,
                   default=Path("runs/eval/robomme/archive_read_best1250_val_n10_seed6"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--server-python", type=Path, default=REPO_ROOT / ".venv/bin/python")
    p.add_argument("--robomme-python", type=Path, default=Path("/home/sjkim/robomme_benchmark/.venv/bin/python"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--server-timeout", type=float, default=300)
    p.add_argument("--task-timeout", type=float, default=0)
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--allow-initialization-checkpoints", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    return p


def validate_options(args):
    if args.checkpoint is None and not args.report_only:
        raise ValueError("ECHO evaluation requires --checkpoint")
    if not args.models or len(set(args.models)) != len(args.models):
        raise ValueError("Fresh ECHO roles must be nonempty and unique")
    if (not all(math.isfinite(x) for x in (args.server_timeout, args.task_timeout))
            or args.server_timeout <= 0 or args.task_timeout < 0):
        raise ValueError("Invalid evaluation timeout")
    for key, value in PANEL.items():
        setattr(args, key, copy.deepcopy(value))


def validate_manifest_contract(identity):
    if (identity.get("format_version") != 1 or identity.get("trainer_variant") != VARIANT
            or identity.get("evaluation_id") != identity_digest(identity)):
        raise ValueError("Changed or non-ECHO evaluation manifest")
    if any(identity.get("settings", {}).get(k) != v for k, v in PANEL.items()):
        raise ValueError("ECHO uses the fixed VAL160/seed6/action16/max1300 panel")
    baseline = _original_baseline(identity)
    if baseline.get("server_script") != BASELINE_SERVER or not identity.get("baseline_reference"):
        raise ValueError("Original HAMLET must be explicitly reused from its completed source run")
    models = identity["models"]
    if not set(models) <= {"baseline", *ROLES} or len(models) < 2:
        raise ValueError("Missing ECHO candidate or unknown role")
    from run_scripts.robomme.echo_cvom_core import EchoConfig
    from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18
    for role, model in models.items():
        if role == "baseline":
            continue
        if (model.get("mode") != VARIANT or model.get("server_script") != SERVER
                or not model.get("memory_checkpoint") or model.get("base_model") != baseline["base_model"]
                or type(model.get("step")) is not int or model["step"] < 0
                or model.get("stage") not in (1, 2)):
            raise ValueError("Invalid ECHO actor identity")
        training_metadata = model.get("training_metadata", {})
        if ((model["step"] == 0 or training_metadata.get("smoke_only") is True
                or training_metadata.get("training_complete") is False
                or training_metadata.get("inherited_actor_training_complete") is False)
                and not identity.get("allow_initialization_checkpoints")):
            raise ValueError("Initialization/incomplete/smoke checkpoint is not a trained performance candidate")
        hashes = model.get("checkpoint_files_sha256", {})
        if (set(hashes) != {"checkpoint.json", "core.safetensors", "expert.safetensors"}
                or any(not isinstance(x, str) or len(x) != 64 or any(c not in "0123456789abcdef" for c in x)
                       for x in hashes.values())):
            raise ValueError("Incomplete ECHO checkpoint provenance")
        metadata = model.get("training_metadata", {})
        if metadata.get("payload_sha256") != {k: v for k, v in hashes.items() if k != "checkpoint.json"}:
            raise ValueError("ECHO payload hashes differ from actor metadata")
        if (metadata.get("future_inputs_at_inference") is not False or not metadata.get("parent_identity")
                or metadata.get("base_model") != baseline["base_model"]):
            raise ValueError("ECHO causal/parent provenance is missing")
        config = model.get("training_config", {})
        representation = RepresentationConfigV18(**config["representation"])
        echo = EchoConfig(**config["echo"])
        if representation.representation != "short" or representation.capacity_events != echo.capacity_events:
            raise ValueError("ECHO representation/storage capacities disagree")
        if model.get("representation_config") != config["representation"]:
            raise ValueError("ECHO representation identity differs")
        expected_policy = "fifo" if role == "fifo" or model["stage"] == 1 else "echo-cvom"
        if (model.get("write_policy") != expected_policy or model.get("memory_off") is not (role == "memory-off")
                or model.get("feature_precision") != "native"
                or model.get("feature_precision_rules") != feature_precision_contract("native")):
            raise ValueError("ECHO storage/READ/precision role differs")
        sources = model.get("echo_source_sha256", {})
        if set(sources) != set(RUNTIME_SOURCES) or any(
                identity.get("source_sha256", {}).get("run_scripts/robomme/" + k) != v for k, v in sources.items()):
            raise ValueError("ECHO runtime source closure differs")
    candidates = [m for role, m in models.items() if role != "baseline"]
    ignored = {"description", "memory_off", "write_policy"}
    canonical = lambda model: {k: v for k, v in model.items() if k not in ignored}
    if any(canonical(m) != canonical(candidates[0]) for m in candidates[1:]):
        raise ValueError("ECHO controls must share exactly one actor, manager and capacity")


def build_identity(args):
    from gr00t.long_memory.hamlet import checkpoint_identity
    from run_scripts.robomme.echo_cvom_checkpoint import inspect_checkpoint
    base, checkpoint = resolve_repo_path(args.base_model), resolve_repo_path(args.checkpoint)
    base_config = json.loads((base / "config.json").read_text())
    if (base_config.get("hamlet_mode") != "finetune" or base_config.get("n_moment_tokens") != 4
            or base_config.get("memory_window") != 4 or base_config.get("memory_stride", 16) != 16
            or base_config.get("mem_cond_type", "cross_attn") != "cross_attn"):
        raise ValueError("ECHO requires the original K4/Q4/stride16 HAMLET")
    info = inspect_checkpoint(base, checkpoint)
    base_id = checkpoint_identity(base)
    sources = {p.relative_to(REPO_ROOT) for p in (REPO_ROOT / "gr00t").rglob("*.py")}
    sources.update(Path("run_scripts/robomme") / name for name in (*DEPENDENCIES, *RUNTIME_SOURCES, "eval_echo_cvom.py"))
    sources.add(Path(BASELINE_SERVER))
    hashes = {str(p): file_hash(REPO_ROOT / p) for p in sorted(sources)}
    models = {"baseline": {"base_model": base_id, "memory_checkpoint": None, "mode": "none",
        "write_policy": "none", "write_policy_override": "checkpoint", "memory_off": False,
        "archive_read_off": False, "server_script": BASELINE_SERVER,
        "description": "Original HAMLET, reused without adapters"}}
    common = {"base_model": base_id, "memory_checkpoint": str(checkpoint), "mode": VARIANT,
        "step": info["step"], "stage": info["stage"], "server_script": SERVER,
        "write_policy_override": "checkpoint", "archive_read_off": False,
        "representation_config": info["config"]["representation"], "training_config": info["config"],
        "training_metadata": info["metadata"], "checkpoint_files_sha256": info["files_sha256"],
        "feature_precision": "native", "feature_precision_rules": feature_precision_contract("native"),
        "echo_source_sha256": {name: hashes["run_scripts/robomme/" + name] for name in RUNTIME_SOURCES}}
    for role in args.models:
        models[role] = dict(copy.deepcopy(common), memory_off=role == "memory-off",
            write_policy="fifo" if role == "fifo" or info["stage"] == 1 else "echo-cvom")
    validate_output_scope(resolve_repo_path(args.output_dir), base, checkpoint,
        resolve_repo_path(args.baseline_reference), Path(info["metadata"]["parent_identity"]["path"]))
    identity = {"format_version": 1, "trainer_variant": VARIANT, "models": models,
        "allow_initialization_checkpoints": args.allow_initialization_checkpoints,
        "settings": {**copy.deepcopy(PANEL), "save_videos": args.save_videos, "device": args.device},
        "source_sha256": hashes,
        "base_file_sha256": {str(p.relative_to(base)): file_hash(p) for p in sorted(base.rglob("*"))
            if p.is_file() and p.suffix in (".json", ".safetensors", ".model", ".txt")},
        "policy_package_versions": {name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "safetensors", "numpy")},
        "benchmark": benchmark_identity(args), "server_python": str(python_path(args.server_python)),
        "robomme_python": str(python_path(args.robomme_python)),
        "selection_protocol": "fixed development VAL160; not final TEST",
        "control_contract": "same ECHO actor; FIFO changes storage only; READ-off continues all writes"}
    identity["baseline_reference"] = build_reference(resolve_repo_path(args.baseline_reference), identity)
    identity["evaluation_id"] = identity_digest(identity)
    validate_manifest_contract(identity)
    old = resolve_repo_path(args.output_dir) / "comparison_manifest.json"
    if old.exists() and json.loads(old.read_text()) != identity:
        raise ValueError("Output belongs to a different experiment; use a new directory")
    return identity


def server_command(args, model, port):
    command = [str(python_path(args.server_python)), "-u", str(REPO_ROOT / SERVER),
        "--base-model", model["base_model"]["path"], "--checkpoint", model["memory_checkpoint"],
        "--device", args.device, "--host", "127.0.0.1", "--port", str(port)]
    if model["memory_off"]:
        command.append("--memory-off")
    if model["write_policy"] == "fifo":
        command.append("--fifo")
    return command


def verify_runtime_inputs(identity):
    files = {REPO_ROOT / path: digest for path, digest in identity["source_sha256"].items()}
    base = Path(identity["models"]["baseline"]["base_model"]["path"])
    files.update({base / path: digest for path, digest in identity["base_file_sha256"].items()})
    for role, model in identity["models"].items():
        if role != "baseline":
            files.update({Path(model["memory_checkpoint"]) / path: digest
                          for path, digest in model["checkpoint_files_sha256"].items()})
            parent = model["training_metadata"]["parent_identity"]
            files[Path(parent["path"]) / "checkpoint.json"] = parent["checkpoint_sha256"]
            files.update({Path(parent["path"]) / path: digest for path, digest in parent["payload_sha256"].items()})
    for path, digest in files.items():
        if file_hash(path) != digest:
            raise ValueError(f"Bound inference file changed: {path}")
    validate_reference(identity["baseline_reference"], identity)


def completed_read_diagnostics(root, role, manifest):
    model = manifest["models"][role]
    totals = dict(completed_sessions=0, missing_sessions=0, missing_identity=0, calls=0, enabled_calls=0)
    decisions = {name: 0 for name in ("append", "keep", "replace", "merge")}
    evidence, maximum = {}, 0.0
    for task in manifest["settings"]["tasks"]:
        folder = Path(root) / role / task
        rows = read_results(folder / "simulation_results.csv", expected=10)
        path = folder / "memory_diagnostics.jsonl"
        sessions, completed = {}, {}
        if path.exists():
            for line in path.read_text().splitlines(keepends=True):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith("\n"):
                        continue
                    raise ValueError(f"Malformed runtime evidence: {path}")
                if row.get("kind") == "policy_call":
                    sessions.setdefault(row.get("session_id"), []).append(row)
                elif row.get("kind") == "episode_complete":
                    result = rows.get(row.get("episode_idx"))
                    if result and all(result[k] == row.get(k) for k in ("episode_seed", "success")):
                        completed[row["episode_idx"]] = row.get("session_id")
            for name in ("memory_diagnostics.jsonl", "simulation_results.csv", "policy_manifest.json"):
                item = folder / name
                if item.exists():
                    evidence[str(item.relative_to(root))] = file_hash(item)
        for episode, row in rows.items():
            calls = sessions.get(completed.get(episode), [])
            if not calls:
                totals["missing_sessions"] += 1
                continue
            totals["completed_sessions"] += 1
            previous_frame, previous_passive, previous_events = None, True, 0
            for index, record in enumerate(calls):
                info, passive, frame = record.get("info", {}), record.get("passive"), record.get("frame_index")
                expected = {"checkpoint_variant": VARIANT, "checkpoint_step": model["step"],
                    "stage": model["stage"], "representation": model["representation_config"]["representation"],
                    "memory_off": model["memory_off"], "feature_precision": "native",
                    "feature_precision_rules": feature_precision_contract("native"),
                    "payload_sha256": model["training_metadata"]["payload_sha256"],
                    "echo_checkpoint_sha256": model["checkpoint_files_sha256"]["checkpoint.json"],
                    "echo_parent_identity": model["training_metadata"]["parent_identity"],
                    "echo_source_sha256": model["echo_source_sha256"]}
                if not set(expected) <= set(info):
                    totals["missing_identity"] += 1
                for key, value in expected.items():
                    if key in info and info[key] != value:
                        raise ValueError(f"Runtime ECHO identity differs: {role}/{task}/{key}")
                if (record.get("episode_idx") != episode or record.get("episode_seed") != row["episode_seed"]
                        or record.get("call") != index or type(passive) is not bool
                        or type(frame) is not int or frame < 0):
                    raise ValueError("Misidentified/unordered completed-session calls")
                if previous_frame is not None and (not 0 < frame - previous_frame <= 16 or (not previous_passive and passive)):
                    raise ValueError("Runtime observation chronology differs")
                count = record.get("executed_action_count")
                expected_count = 0 if previous_frame is None or previous_passive else frame - previous_frame
                if count != expected_count:
                    raise ValueError("Runtime controls include an unexecuted or misaligned prefix")
                memory = info.get("long_memory", {})
                enabled = not model["memory_off"] and not passive and index > 0
                if (memory.get("policy") != model["write_policy"] or memory.get("memory_read_enabled") is not enabled
                        or memory.get("completed_action_count") != count
                        or memory.get("observations_seen") != index + 1):
                    raise ValueError("Runtime ECHO READ/storage/action identity differs")
                metrics = memory.get("read", {})
                delta = metrics.get("ae_conditioning_delta_norm")
                if type(delta) not in (int, float) or not math.isfinite(delta) or delta < 0:
                    totals["missing_identity"] += 1
                else:
                    maximum = max(maximum, delta)
                    if not enabled and delta != 0:
                        raise ValueError("READ bypass changed same-actor conditioning")
                operations = [metrics.get("writer_" + name) for name in decisions]
                if any(x not in (0, 1) for x in operations) or sum(operations) != 1:
                    raise ValueError("Missing or ambiguous ECHO WRITE decision")
                operation = next(name for name in decisions if metrics["writer_" + name] == 1)
                capacity = model["representation_config"]["capacity_events"]
                events = previous_events + int(operation == "append")
                if (events > capacity or (operation == "replace" and previous_events == 0)
                        or (operation == "append" and previous_events == capacity)
                        or memory.get("memory_tokens") != events * model["representation_config"]["num_short_tokens"]):
                    raise ValueError("ECHO WRITE broke complete-event capacity")
                if model["write_policy"] == "fifo" and operation != ("append" if previous_events < capacity else "replace"):
                    raise ValueError("FIFO control used a learned operation")
                decisions[operation] += 1
                previous_frame, previous_passive, previous_events = frame, passive, events
                totals["calls"] += 1
                totals["enabled_calls"] += int(enabled)
    return {**totals, "max_read_delta_norm": maximum, "writer_decisions": decisions,
        "files_sha256": evidence, "complete_evidence": totals["completed_sessions"] == 160
        and totals["missing_sessions"] == totals["missing_identity"] == 0}


def write_report(root, bootstrap_samples=5000):
    root = Path(root)
    manifest = json.loads((root / "comparison_manifest.json").read_text())
    validate_manifest_contract(manifest)
    reference_root, reference = validate_reference(manifest["baseline_reference"], manifest)
    result, rendered = build_reference_report(root, manifest, reference_root, reference,
                                              bootstrap_samples=bootstrap_samples)
    result["additional_comparisons"] = {}
    for left, right in (("fifo", "memory"), ("memory-off", "memory")):
        if {left, right} <= set(manifest["models"]):
            result["additional_comparisons"][left + "_to_" + right] = _contrast(
                root, left, right, manifest["settings"]["tasks"], 10, bootstrap_samples)
    result["runtime_evidence"] = {role: completed_read_diagnostics(root, role, manifest)
        for role in manifest["models"] if role != "baseline"}
    result["limitations"] = "Repeated development VAL160, one inference seed; not final TEST generalization."
    result["evidence_sha256"] = identity_digest(result["runtime_evidence"])
    _atomic_json(root / "comparison_summary.json", result)
    return result, rendered


def run_evaluation(args, identity, env):
    output = resolve_repo_path(args.output_dir)
    bind_manifest(output, identity)
    with (output / ".driver.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another evaluation owns this output") from exc
        failures, interrupted = [], False
        try:
            for role, model in identity["models"].items():
                if role == "baseline":
                    validate_reference(identity["baseline_reference"], identity)
                    continue
                pending = []
                for task in args.tasks:
                    rows = read_results(output / role / task / "simulation_results.csv", expected=10)
                    if rows:
                        validate_result_identity(output, role, task, identity)
                    if len(rows) != 10:
                        pending.append(task)
                if not pending:
                    continue
                verify_runtime_inputs(identity)
                folder = output / role
                folder.mkdir(exist_ok=True)
                port, server = free_local_port(), None
                command = server_command(args, model, port)
                with (folder / "server.log").open("a") as log:
                    try:
                        gpu = gpu_guard(args.device)
                        # Occupancy is dynamic launch evidence, not part of
                        # immutable experiment identity used for safe resume.
                        plans = folder / "launch-plans"
                        plans.mkdir(exist_ok=True)
                        _atomic_json(plans / ("launch-" + uuid.uuid4().hex + ".json"), {
                            "evaluation_id": identity["evaluation_id"], "role": role,
                            "command": command, "gpu_check": gpu})
                        log.write("\n[driver] " + json.dumps(command) + "\n")
                        log.flush()
                        server = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
                        server_ready(server, port, args.server_timeout)
                        for task in pending:
                            task_dir = folder / task
                            task_dir.mkdir(exist_ok=True)
                            client = [str(python_path(args.robomme_python)), "-u", str(REPO_ROOT / CLIENT),
                                "--task-id", task, "--policy-client-host", "127.0.0.1", "--policy-client-port", str(port),
                                "--dataset", "val", "--n-episodes", "10", "--max-episode-steps", "1300",
                                "--n-action-steps", "16", "--model-config", model["base_model"]["path"],
                                "--output-dir", str(task_dir), "--seed", "6",
                                "--evaluation-id", identity["evaluation_id"] + ":" + role]
                            if args.save_videos:
                                client.append("--save-videos")
                            print(f"[echo] {role}/{task}: {task_dir / 'rollout.log'}", flush=True)
                            try:
                                run_client(client, env=env, log_path=task_dir / "rollout.log",
                                           server=server, timeout=args.task_timeout)
                                if len(read_results(task_dir / "simulation_results.csv", expected=10)) != 10:
                                    raise RuntimeError("Task returned incomplete episodes")
                                validate_result_identity(output, role, task, identity)
                            except (RuntimeError, TimeoutError, ValueError) as exc:
                                failures.append({"model": role, "task": task, "error": str(exc)})
                            write_report(output, 500)
                    except (RuntimeError, TimeoutError) as exc:
                        failures.append({"model": role, "error": str(exc)})
                    finally:
                        stop_process(server)
        except KeyboardInterrupt:
            interrupted = True
            raise
        except Exception as exc:
            failures.append({"model": "driver", "error": str(exc)})
            raise
        finally:
            _atomic_json(output / "driver_status.json", {"interrupted": interrupted, "failures": failures})
            result, report = write_report(output)
            print(report, flush=True)
    return int(bool(failures) or not all(x["complete"] for x in result["models"].values())
        or not all(x["complete_evidence"] for x in result["runtime_evidence"].values()))


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_options(args)
    if args.report_only:
        output = resolve_repo_path(args.output_dir)
        with (output / ".driver.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result, report = write_report(output)
        print(report)
        return int(not all(x["complete_evidence"] for x in result["runtime_evidence"].values()))
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONUNBUFFERED="1", GR00T_INFERENCE_SEED="6")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    if python_path(Path(sys.executable)) != python_path(args.server_python):
        return subprocess.call([str(python_path(args.server_python)), str(Path(__file__).resolve()),
            *(sys.argv[1:] if argv is None else argv)], cwd=REPO_ROOT, env=env)
    check_dependencies(python_path(args.server_python), "import torch, cv2, transformers, safetensors, msgpack, zmq", env, "policy")
    check_dependencies(python_path(args.robomme_python), "import robomme, msgpack, zmq, pandas, imageio", env, "simulator")
    identity = build_identity(args)
    if args.preflight_only:
        print("[echo] Preflight passed; no model/simulator constructed and no output created.")
        return 0
    output = resolve_repo_path(args.output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ("." + output.name + ".echo-launch.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_evaluation(args, identity, env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[echo-eval] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
