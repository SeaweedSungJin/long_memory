#!/usr/bin/env python3
"""One-variable min_fill=32 VAL160; reuse completed original min_fill=4 ECHO.

Uses an explicitly identified NEW server wrapper. Legacy manifests/checks are
not edited or relaxed. The narrow projection below is checked by the original
validator AFTER the new override fields are independently validated. Original
checkpoint training config remains intact; effective runtime config is separate.
"""
from __future__ import annotations
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_scripts.robomme import eval_echo_cvom as old
from run_scripts.robomme.policy_echo_min_fill import min_fill_source_identity
from gr00t.long_memory.monitoring import _atomic_json

SERVER = "run_scripts/robomme/serve_echo_min_fill.py"
VERSION = "echo_min_fill_only_val160_v1"


def parser():
    p = old.build_parser()
    p.description = __doc__
    p.set_defaults(models=["memory"], checkpoint=Path("runs/long_memory/echo_cvom_full_v1/refresh/checkpoint-001000"))
    p.add_argument("--control-reference", type=Path, default=Path("runs/eval/robomme/echo_cvom_full_v1_val160_seed6"))
    p.add_argument("--regression-report", type=Path,
                   default=Path("runs/long_memory/echo_min_fill_regression_20260922/completed.json"))
    return p


def validate_identity(identity):
    if identity.get("evaluation_id") != old.identity_digest(identity) or identity.get("min_fill_study", {}).get("version") != VERSION:
        raise ValueError("Changed/non-min-fill study manifest")
    if set(identity["models"]) != {"baseline", "memory"}:
        raise ValueError("Only one new memory arm plus reused original baseline allowed")
    model = identity["models"]["memory"]
    if (model.get("server_script") != SERVER or model.get("original_min_fill") != 4
            or model.get("effective_min_fill") != 32 or model.get("min_fill_override") != 32
            or model["training_config"]["echo"]["min_fill"] != 4 or model["stage"] != 2
            or model["write_policy"] != "echo-cvom" or model["memory_off"]):
        raise ValueError("Only stage2 learned storage min_fill4 ->32 is permitted")
    effective = dict(model["training_config"]["echo"], min_fill=32)
    if model.get("effective_echo_config") != effective or effective["capacity_events"] != 32:
        raise ValueError("Effective runtime config changed more than min_fill")
    if model.get("min_fill_source_sha256") != min_fill_source_identity():
        raise ValueError("Wrapper source differs from declared runtime")
    for name, sha in model["min_fill_source_sha256"].items():
        if identity["source_sha256"].get("run_scripts/robomme/"+name) != sha:
            raise ValueError("Incomplete wrapper source closure")
    # This is a LOCAL, one-field server projection for validating the unchanged
    # underlying ECHO checkpoint/control rules, never a mutation of evidence.
    projection = copy.deepcopy(identity)
    projection["models"]["memory"]["server_script"] = old.SERVER
    projection["evaluation_id"] = old.identity_digest(projection)
    old.validate_manifest_contract(projection)


def inspect_control(root, current):
    root = root.resolve(strict=True)
    old.validate_output_scope(Path(current["min_fill_study"]["output_dir"]), root)
    source = json.loads((root/"comparison_manifest.json").read_text())
    old.validate_manifest_contract(source)
    status = json.loads((root/"driver_status.json").read_text())
    if status.get("interrupted") is not False or status.get("failures") != []:
        raise ValueError("Control did not complete cleanly")
    with (root/".driver.lock").open("rb") as f:
        fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
    for key in ("settings", "base_file_sha256", "policy_package_versions", "benchmark", "server_python", "robomme_python"):
        if source[key] != current[key]:
            raise ValueError(f"Old min_fill4 control changed {key}; rerun control in a separate supported study")
    expected = copy.deepcopy(current["models"]["memory"])
    for key in ("original_min_fill", "effective_min_fill", "min_fill_override", "effective_echo_config", "min_fill_source_sha256"):
        expected.pop(key)
    expected["server_script"] = old.SERVER
    if source["models"]["memory"] != expected:
        raise ValueError("Control actor/config differs beyond the declared runtime min_fill override")
    # Every original inference/helper source must match; extra diagnostic
    # wrappers are allowed only because their native behavior is audited.
    for name, sha in source["source_sha256"].items():
        if current["source_sha256"].get(name) != sha:
            raise ValueError(f"Original control runtime source changed: {name}")
    evidence = old.completed_read_diagnostics(root, "memory", source)
    if not evidence["complete_evidence"]:
        raise ValueError("Control lacks complete per-episode runtime evidence")
    hashes = {"comparison_manifest.json": old.file_hash(root/"comparison_manifest.json"),
              "driver_status.json": old.file_hash(root/"driver_status.json"), **evidence["files_sha256"]}
    for task in old.PANEL["tasks"]:
        rows = old.read_results(root/"memory"/task/"simulation_results.csv", expected=10)
        if len(rows) != 10:
            raise ValueError("Control not complete on fixed VAL160")
        old.validate_result_identity(root, "memory", task, source)
    return {"source_run": str(root), "source_evaluation_id": source["evaluation_id"], "role": "memory",
            "files_sha256": hashes, "completed_episodes": 160, "newly_rolled_out": 0}


def verify_regression(path, model):
    from run_scripts.robomme.verify_echo_min_fill import verification_source_identity
    record = json.loads(path.read_text())
    if (record.get("kind") != "echo_min_fill_regression_v1" or record.get("panel") != [[1355, 72]]
            or record.get("real_ae_calls", 0) <= 0
            or record.get("calls_by_role") != {"parent": 72, "wrapper-checkpoint": 72, "wrapper-minfill32": 72}
            or record.get("passed") is not True or record.get("control_same_as_parent") is not True
            or record.get("minfill32_prefull_reject_zero") is not True
            or record.get("parameters_unchanged") is not True
            or record.get("checkpoint_files_sha256") != model["checkpoint_files_sha256"]
            or record.get("echo_source_sha256") != model["echo_source_sha256"]
            or record.get("verification_source_sha256") != verification_source_identity()
            or record.get("min_fill_source_sha256") != model["min_fill_source_sha256"]):
        raise ValueError("Missing/incompatible actual native min_fill regression evidence")
    events = path.parent/"events.json"
    if old.file_hash(events) != record.get("events_sha256"):
        raise ValueError("Native regression event evidence changed")
    return {"path": str(path.resolve()), "sha256": old.file_hash(path), "events_sha256": record["events_sha256"]}


def build_identity(args):
    if args.models != ["memory"]:
        raise ValueError("This experiment launches only min_fill32 memory; no FIFO/READ-off/new training")
    # Old builder correctly rejects a new-wrapper manifest on resume. Give
    # its read-only output collision check a never-created child path; the
    # ACTUAL manifest is still checked strictly by bind_manifest below.
    build_args = copy.copy(args)
    build_args.output_dir = args.output_dir/".identity-preflight-no-output"
    identity = old.build_identity(build_args)
    model = identity["models"]["memory"]
    model.update(server_script=SERVER, original_min_fill=4, effective_min_fill=32, min_fill_override=32,
        effective_echo_config=dict(model["training_config"]["echo"], min_fill=32),
        min_fill_source_sha256=min_fill_source_identity())
    for name in (*min_fill_source_identity(), "eval_echo_min_fill.py", "verify_echo_min_fill.py", "verify_echo_cvom.py"):
        identity["source_sha256"]["run_scripts/robomme/"+name] = old.file_hash(ROOT/"run_scripts/robomme"/name)
    identity["min_fill_study"] = {"version": VERSION, "output_dir": str(args.output_dir.resolve()),
        "only_behavior_change": "force append while bank occupancy <32; full-bank learned policy unchanged",
        "regression": verify_regression(args.regression_report.resolve(strict=True), model)}
    identity["evaluation_id"] = old.identity_digest(identity)
    identity["min_fill_study"]["control_reference"] = inspect_control(args.control_reference, identity)
    identity["evaluation_id"] = old.identity_digest(identity)
    validate_identity(identity)
    saved = args.output_dir.resolve()/"comparison_manifest.json"
    if saved.exists() and json.loads(saved.read_text()) != identity:
        raise ValueError("Output belongs to a different min-fill experiment")
    return identity


def verify_inputs(identity):
    validate_identity(identity)
    old.verify_runtime_inputs(identity)
    study = identity["min_fill_study"]
    audit = study["regression"]
    if old.file_hash(Path(audit["path"])) != audit["sha256"]:
        raise ValueError("Native regression evidence changed")
    if verify_regression(Path(audit["path"]), identity["models"]["memory"]) != audit:
        raise ValueError("Native regression evidence no longer validates")
    fresh = inspect_control(Path(study["control_reference"]["source_run"]), identity)
    if fresh != study["control_reference"]:
        raise ValueError("Reused min_fill4 results changed")


def runtime_override_evidence(root, manifest):
    model, count = manifest["models"]["memory"], 0
    required = {k: model[k] for k in ("min_fill_override", "original_min_fill", "effective_min_fill", "min_fill_source_sha256")}
    for task in old.PANEL["tasks"]:
        path = root/"memory"/task/"memory_diagnostics.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines(keepends=True):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    continue
                raise
            if row.get("kind") != "policy_call":
                continue
            info = row["info"]
            if any(info.get(k) != v for k,v in required.items()):
                raise ValueError("Actual server omitted/changed min_fill override")
            metrics = info["long_memory"]["read"]
            if metrics["writer_full"] == 0 and metrics["writer_keep"] != 0:
                raise ValueError("min_fill32 still rejected before capacity")
            count += 1
    return {"verified_policy_calls": count, "effective_min_fill": 32, "prefull_rejects": 0}


def paired_policy_contract(before, after):
    for key in ("scenario_metadata_sha256", "model_config_sha256", "memory_window", "demo_sampling"):
        if key not in before or key not in after or before[key] != after[key]:
            raise ValueError(f"Cross-arm observation/scenario policy differs: {key}")


def write_report(root):
    root = Path(root)
    manifest = json.loads((root/"comparison_manifest.json").read_text())
    verify_inputs(manifest)
    reference = Path(manifest["min_fill_study"]["control_reference"]["source_run"])
    reference_manifest = json.loads((reference/"comparison_manifest.json").read_text())
    result = {"version": VERSION, "tasks": {}, "min_fill4_successes": 0, "min_fill32_successes": 0}
    for task in old.PANEL["tasks"]:
        before = old.read_results(reference/"memory"/task/"simulation_results.csv", expected=10)
        after = old.read_results(root/"memory"/task/"simulation_results.csv", expected=10)
        if after:
            previous = old.validate_result_identity(reference, "memory", task, reference_manifest)
            current = old.validate_result_identity(root, "memory", task, manifest)
            paired_policy_contract(previous, current)
        b, a = sum(x["success"] for x in before.values()), sum(x["success"] for x in after.values())
        result["tasks"][task] = {"min_fill4_successes": b, "min_fill32_successes": a, "completed_new": len(after)}
        result["min_fill4_successes"] += b; result["min_fill32_successes"] += a
    result["comparison"] = old._contrast(root, str(reference/"memory"), "memory", old.PANEL["tasks"], 10, 5000)
    result["legacy_runtime_checks"] = old.completed_read_diagnostics(root, "memory", manifest)
    result["override_runtime_checks"] = runtime_override_evidence(root, manifest)
    result["complete"] = result["comparison"]["complete"] and result["legacy_runtime_checks"]["complete_evidence"]
    result["interpretation"] = "Paired fixed development VAL160: min_fill32 minus4. Not final TEST; same checkpoint, native precision."
    _atomic_json(root/"comparison_summary.json", result)
    text = f"min_fill4: {result['min_fill4_successes']}/160 (reused)\nmin_fill32: {result['min_fill32_successes']}/160; complete={result['complete']}\n"
    text += json.dumps(result["comparison"], indent=2)+"\n"
    (root/"comparison_summary.txt").write_text(text)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    old.validate_options(args)
    if args.report_only:
        root = args.output_dir.resolve(strict=True)
        with (root/".driver.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(write_report(root), indent=2))
        return 0
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", GR00T_INFERENCE_SEED="6")
    env.setdefault("MUJOCO_GL", "egl"); env.setdefault("PYOPENGL_PLATFORM", "egl")
    old.check_dependencies(old.python_path(args.server_python), "import torch, cv2, transformers, safetensors, msgpack, zmq", env, "policy")
    old.check_dependencies(old.python_path(args.robomme_python), "import robomme, msgpack, zmq, pandas, imageio", env, "simulator")
    identity = build_identity(args)
    verify_inputs(identity)
    if args.preflight_only:
        print("[min-fill] Preflight passed; same checkpoint/VAL160; reuse completed4; launch only32. No output/model created.")
        return 0
    root = args.output_dir.resolve()
    # New manifest binds a different actual server. Never rewrite old manifests.
    old.bind_manifest(root, identity)
    with (root/".driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        for task in args.tasks:
            rows = old.read_results(root/"memory"/task/"simulation_results.csv", expected=10)
            if rows: old.validate_result_identity(root, "memory", task, identity)
            if len(rows) < 10: pending.append(task)
        if not pending:
            return int(not write_report(root)["complete"])
        gpu = old.gpu_guard(args.device)
        folder = root/"memory"; folder.mkdir(exist_ok=True)
        port, server, failures, interrupted = old.free_local_port(), None, [], False
        command = [str(old.python_path(args.server_python)), "-u", str(ROOT/SERVER),
            "--base-model", str(args.base_model.resolve()), "--checkpoint", str(args.checkpoint.resolve()),
            "--device", args.device, "--port", str(port), "--min-fill", "32"]
        _atomic_json(folder/f"launch-{time.time_ns()}.json", {"command": command, "gpu_check": gpu, "evaluation_id": identity["evaluation_id"]})
        with (folder/"server.log").open("a") as log:
            try:
                server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                old.server_ready(server, port, args.server_timeout)
                for task in pending:
                    out = folder/task; out.mkdir(exist_ok=True)
                    client = [str(old.python_path(args.robomme_python)), "-u", str(ROOT/old.CLIENT),
                        "--task-id", task, "--policy-client-port", str(port), "--policy-client-host", "127.0.0.1",
                        "--dataset", "val", "--n-episodes", "10", "--max-episode-steps", "1300", "--n-action-steps", "16",
                        "--model-config", str(args.base_model.resolve()), "--output-dir", str(out), "--seed", "6",
                        "--evaluation-id", identity["evaluation_id"]+":memory"]
                    if args.save_videos: client.append("--save-videos")
                    print(f"[min-fill32] {task}: {out/'rollout.log'}", flush=True)
                    old.run_client(client, env=env, log_path=out/"rollout.log", server=server, timeout=args.task_timeout)
                    if len(old.read_results(out/"simulation_results.csv", expected=10)) != 10:
                        raise RuntimeError("Incomplete task")
                    old.validate_result_identity(root, "memory", task, identity)
                    write_report(root)
            except KeyboardInterrupt:
                interrupted = True
                raise
            except Exception as exc:
                failures.append(str(exc)); raise
            finally:
                old.stop_process(server)
                _atomic_json(root/"driver_status.json", {"failures": failures, "interrupted": interrupted})
                result = write_report(root)
        print(f"[min-fill] {root/'comparison_summary.txt'}")
        return int(not result["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
