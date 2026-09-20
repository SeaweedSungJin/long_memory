#!/usr/bin/env python3
"""V18 fixed VAL fast160 evaluation with explicitly reused original HAMLET.

Default: all 16 tasks x first 10 VAL scenarios, seed 6. These 160 scenarios
are a development set, not evidence of full TEST success. Baseline references
remain at their original path/identity. memory-off always loads the SAME
candidate's short adapter and Action Expert, bypassing only long READ/fusion.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from run_scripts.robomme.eval_long_memory_comparison import (
    TASKS, benchmark_identity, bind_manifest, check_dependencies, file_hash,
    free_local_port, python_path, read_results, resolve_repo_path, run_client,
    server_ready, stop_process, validate_result_identity,
)
from gr00t.eval.sim.robomme.compare_long_memory_results import build_report
from gr00t.eval.sim.robomme.compare_long_memory_v3_results import _contrast
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.baseline_reference_v18 import build_reference, validate_reference
from run_scripts.robomme.report_baseline_reference_v10 import build_reference_report

VARIANT = "representation_v18"
BASELINE_SERVER = "run_scripts/robomme/serve_archive_projector_v10.py"
SERVER = "run_scripts/robomme/serve_representation_v18.py"
CLIENT = "gr00t/eval/sim/robomme/run_long_memory_rollout.py"
ROLES = ("baseline", "memory", "memory-off", "fifo")
DEPENDENCIES = (
    "eval_representation_v18.py", "serve_representation_v18.py", "policy_representation_v18.py",
    "representation_core_v18.py", "checkpoint_representation_v18.py", "baseline_reference_v18.py",
    "baseline_reference_v10.py", "report_baseline_reference_v10.py", "eval_long_memory_comparison.py",
    "policy_archive_projector_v10.py", "checkpoint_projector_v10.py", "projector_adapter_v10.py",
    "storage_cvom_v18.py",
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", type=Path, default=Path("checkpoints/author_hamlet_robomme/checkpoint-60000"))
    p.add_argument("--checkpoint", type=Path, help="One genuine V18 bundle, shared by every candidate role")
    p.add_argument("--writer-checkpoint", type=Path, help="Optional CVOM sidecar bound to this exact reader checkpoint")
    p.add_argument("--baseline-reference", type=Path, help="Completed ORIGINAL run containing comparison_manifest.json; never a copied CSV")
    p.add_argument("--models", nargs="+", choices=ROLES, default=["baseline", "memory"])
    p.add_argument("--tasks", nargs="+", default=["all"])
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--dataset", choices=("train", "val", "test"), default="val")
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--n-action-steps", type=int, default=16)
    p.add_argument("--max-episode-steps", type=int, default=1300)
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("runs/eval/robomme/v18_val160"))
    p.add_argument("--server-python", type=Path, default=REPO_ROOT / ".venv/bin/python")
    p.add_argument("--robomme-python", type=Path, default=Path("/home/sjkim/robomme_benchmark/.venv/bin/python"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--server-timeout", type=float, default=300)
    p.add_argument("--task-timeout", type=float, default=0)
    p.add_argument("--allow-initialization-checkpoints", action="store_true")
    p.add_argument("--preflight-only", action="store_true", help="No model/simulator constructed; no output created")
    p.add_argument("--report-only", action="store_true")
    return p


def validate_options(args):
    if args.tasks == ["all"]:
        args.tasks = list(TASKS)
    if not args.tasks or len(set(args.tasks)) != len(args.tasks) or any(t not in TASKS for t in args.tasks):
        raise ValueError("Tasks must be unique RoboMME tasks or exactly all")
    if args.baseline_reference and "baseline" not in args.models:
        args.models.insert(0, "baseline")
    if "baseline" not in args.models or len(set(args.models)) != len(args.models):
        raise ValueError("Models must be unique and include baseline (reused or fresh)")
    if set(args.models) - {"baseline"} and args.checkpoint is None:
        raise ValueError("Candidate roles require --checkpoint")
    if args.writer_checkpoint is not None and args.checkpoint is None:
        raise ValueError("A writer requires its --checkpoint reader")
    if "fifo" in args.models and args.writer_checkpoint is None:
        raise ValueError("fifo is a same-capacity CVOM control; supply --writer-checkpoint")
    if not 1 <= args.n_episodes <= (100 if args.dataset == "train" else 50) or args.seed < 0:
        raise ValueError("Invalid episode count or seed")
    if args.n_action_steps != 16 or args.max_episode_steps < 1:
        raise ValueError("V18 preserves action interval 16 and a positive episode limit")
    if not all(math.isfinite(x) for x in (args.server_timeout, args.task_timeout)) or args.server_timeout <= 0 or args.task_timeout < 0:
        raise ValueError("Invalid timeout")
    if args.preflight_only and args.report_only:
        raise ValueError("Choose preflight-only or report-only")


def identity_digest(identity):
    return hashlib.sha256(json.dumps({k: v for k, v in identity.items() if k != "evaluation_id"}, sort_keys=True).encode()).hexdigest()


def validate_manifest_contract(identity):
    if identity.get("format_version") != 1 or identity.get("trainer_variant") != VARIANT or identity.get("evaluation_id") != identity_digest(identity):
        raise ValueError("Changed or non-V18 evaluation manifest")
    models = identity.get("models", {})
    if "baseline" not in models or not set(models) <= set(ROLES):
        raise ValueError("Missing baseline or unknown role")
    for role, model in models.items():
        if model.get("base_model") != models["baseline"].get("base_model"):
            raise ValueError("Candidate uses a different original HAMLET")
        if model.get("memory_off") is not (role == "memory-off") or model.get("archive_read_off") is not False:
            raise ValueError("Runtime READ flag disagrees with role")
        if role == "baseline":
            if (model.get("mode") != "none" or model.get("stage", 0) != 0 or model.get("memory_checkpoint") is not None
                    or model.get("write_policy") != "none" or model.get("write_policy_override") != "checkpoint"
                    or model.get("server_script") != BASELINE_SERVER
                    or any(key in model for key in ("checkpoint_files_sha256", "representation_config", "writer_checkpoint"))):
                raise ValueError("Baseline must be ORIGINAL HAMLET without adapters")
        else:
            if (model.get("mode") != VARIANT or model.get("server_script") != SERVER
                    or not model.get("memory_checkpoint") or type(model.get("step")) is not int
                    or model["step"] < 0 or (model["step"] == 0 and not identity.get("allow_initialization_checkpoints"))):
                raise ValueError("Candidate requires genuine trained V18 checkpoint")
            hashes = model.get("checkpoint_files_sha256", {})
            if set(hashes) != {"checkpoint.json", "model.safetensors", "expert.safetensors"}:
                raise ValueError("Incomplete candidate payload provenance")
            if model.get("write_policy") != ("cvom" if model.get("writer_checkpoint") and role != "fifo" else "fifo"):
                raise ValueError("Writer role disagrees with checkpoint")
    if {"memory", "memory-off"} <= set(models):
        ignored = {"description", "memory_off"}
        if {k: v for k, v in models["memory"].items() if k not in ignored} != {k: v for k, v in models["memory-off"].items() if k not in ignored}:
            raise ValueError("READ-off must retain SAME short adapter, AE, writer and checkpoint")
    if {"memory", "fifo"} <= set(models):
        ignored = {"description", "write_policy"}
        if {k: v for k, v in models["memory"].items() if k not in ignored} != {k: v for k, v in models["fifo"].items() if k not in ignored}:
            raise ValueError("FIFO/CVOM must retain SAME reader, short adapter, AE and capacity")


def build_identity(args):
    from gr00t.long_memory.hamlet import checkpoint_identity
    base = resolve_repo_path(args.base_model)
    original = json.loads((base / "config.json").read_text())
    if (original.get("hamlet_mode") != "finetune" or original.get("n_moment_tokens") != 4
            or original.get("memory_window") != 4 or original.get("memory_stride", 16) != args.n_action_steps
            or original.get("mem_cond_type", "cross_attn") != "cross_attn"):
        raise ValueError("V18 requires original K4/Q4/stride16 cross-attention HAMLET")
    base_id = checkpoint_identity(base)
    models = {"baseline": {"base_model": base_id, "memory_checkpoint": None, "mode": "none",
        "write_policy": "none", "write_policy_override": "checkpoint", "memory_off": False,
        "archive_read_off": False, "server_script": BASELINE_SERVER,
        "description": "Original HAMLET, no added adapter"}}
    inputs = [base]
    if set(args.models) - {"baseline"}:
        from run_scripts.robomme.checkpoint_representation_v18 import checkpoint_info_v18
        checkpoint = resolve_repo_path(args.checkpoint)
        info = checkpoint_info_v18(base, checkpoint)
        if info["step"] == 0 and not args.allow_initialization_checkpoints:
            raise ValueError("Step0 is initialization, not trained: explicitly opt in or select a trained checkpoint")
        common = {"base_model": base_id, "memory_checkpoint": str(checkpoint), "mode": VARIANT,
            "step": info["step"], "server_script": SERVER, "write_policy_override": "checkpoint",
            "archive_read_off": False, "representation_config": info["config"]["representation"],
            "training_config": info["config"], "training_metadata": info["metadata"],
            "checkpoint_files_sha256": {name: file_hash(checkpoint / name) for name in
                ("checkpoint.json", "model.safetensors", "expert.safetensors")}}
        inputs.append(checkpoint)
        cache = info["config"].get("train", {}).get("cache_dir")
        if cache:
            inputs.append(resolve_repo_path(Path(cache)))
        if args.writer_checkpoint:
            from run_scripts.robomme.storage_cvom_v18 import storage_writer_info_v18
            writer_path = resolve_repo_path(args.writer_checkpoint)
            writer_cfg, writer_manifest = storage_writer_info_v18(writer_path, checkpoint)
            if writer_cfg.capacity_events != common["representation_config"]["capacity_events"]:
                raise ValueError("Writer and reader capacity differ; train a matched bounded reader first")
            common.update(writer_checkpoint=str(writer_path), writer_manifest=writer_manifest,
                writer_files_sha256={str(path.relative_to(writer_path)): file_hash(path)
                    for path in sorted(writer_path.rglob("*")) if path.is_file()})
            inputs.append(writer_path)
        for role in args.models:
            if role == "baseline":
                continue
            model = copy.deepcopy(common)
            model.update(memory_off=role == "memory-off",
                write_policy="cvom" if args.writer_checkpoint and role != "fifo" else "fifo",
                description="SAME adapted short/AE, long READ bypassed" if role == "memory-off" else "V18 learned READ/fusion")
            models[role] = model
    if args.baseline_reference:
        inputs.append(resolve_repo_path(args.baseline_reference))
    validate_output_scope(resolve_repo_path(args.output_dir), *inputs)
    sources = {p.relative_to(REPO_ROOT) for p in (REPO_ROOT / "gr00t").rglob("*.py")}
    sources.update(Path("run_scripts/robomme") / name for name in DEPENDENCIES)
    sources.add(Path(BASELINE_SERVER))
    print("[preflight] hashing immutable original HAMLET, candidate and source provenance ...", flush=True)
    identity = {"format_version": 1, "trainer_variant": VARIANT, "models": models,
        "allow_initialization_checkpoints": args.allow_initialization_checkpoints,
        "settings": {k: getattr(args, k) for k in ("tasks", "n_episodes", "dataset", "seed", "n_action_steps", "max_episode_steps", "save_videos", "device")},
        "source_sha256": {str(p): file_hash(REPO_ROOT / p) for p in sorted(sources)},
        "base_file_sha256": {str(p.relative_to(base)): file_hash(p) for p in sorted(base.rglob("*"))
            if p.is_file() and p.suffix in (".json", ".safetensors", ".model", ".txt")},
        "policy_package_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors", "numpy")},
        "benchmark": benchmark_identity(args), "server_python": str(python_path(args.server_python)),
        "robomme_python": str(python_path(args.robomme_python)),
        "selection_protocol": "first_n_metadata_scenarios; VAL for development; TEST only after design fixed",
        "control_contract": "same canonical observations/noise; READ-off preserves adapted short and AE; no manual labels"}
    if args.baseline_reference:
        identity["baseline_reference"] = build_reference(resolve_repo_path(args.baseline_reference), identity)
    identity["evaluation_id"] = identity_digest(identity)
    validate_manifest_contract(identity)
    old_manifest = resolve_repo_path(args.output_dir) / "comparison_manifest.json"
    if old_manifest.exists() and json.loads(old_manifest.read_text()) != identity:
        raise ValueError("Output already belongs to a different experiment; use a NEW --output-dir")
    return identity


def server_command(args, model, port):
    command = [str(python_path(args.server_python)), "-u", str(REPO_ROOT / model["server_script"]),
        "--base-model", model["base_model"]["path"], "--device", args.device, "--host", "127.0.0.1", "--port", str(port)]
    if model["memory_checkpoint"]:
        command += ["--checkpoint", model["memory_checkpoint"]]
        if model["memory_off"]:
            command.append("--memory-off")
        if model.get("writer_checkpoint") and model["write_policy"] == "cvom":
            command += ["--writer-checkpoint", model["writer_checkpoint"]]
    return command


def verify_runtime_inputs(identity):
    files = {REPO_ROOT / path: digest for path, digest in identity["source_sha256"].items()}
    base = Path(identity["models"]["baseline"]["base_model"]["path"])
    files.update({base / path: digest for path, digest in identity["base_file_sha256"].items()})
    for model in identity["models"].values():
        if model["memory_checkpoint"]:
            files.update({Path(model["memory_checkpoint"]) / path: digest for path, digest in model["checkpoint_files_sha256"].items()})
        if model.get("writer_checkpoint"):
            files.update({Path(model["writer_checkpoint"]) / path: digest for path, digest in model["writer_files_sha256"].items()})
    for path, digest in files.items():
        if file_hash(path) != digest:
            raise ValueError(f"Bound inference file changed: {path}")
    if identity.get("baseline_reference"):
        validate_reference(identity["baseline_reference"], identity)


def completed_read_diagnostics(root, role, manifest):
    """Only completed-session calls count; missing evidence is never a clean OFF."""
    model = manifest["models"][role]
    totals = {key: 0 for key in ("completed_sessions", "missing_sessions", "calls", "enabled_calls", "missing_identity")}
    max_delta = 0.0
    for task in manifest["settings"]["tasks"]:
        rows = read_results(Path(root) / role / task / "simulation_results.csv", expected=manifest["settings"]["n_episodes"])
        path = Path(root) / role / task / "memory_diagnostics.jsonl"
        sessions, completed = {}, {}
        if path.exists():
            for line in path.read_text().splitlines(keepends=True):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith("\n"):
                        continue
                    raise ValueError(f"Malformed diagnostic: {path}")
                sid = record.get("session_id")
                if record.get("kind") == "policy_call":
                    sessions.setdefault(sid, []).append(record)
                elif record.get("kind") == "episode_complete":
                    row = rows.get(record.get("episode_idx"))
                    if row and row["episode_seed"] == record.get("episode_seed") and row["success"] == record.get("success"):
                        completed[record["episode_idx"]] = sid
        for episode, row in rows.items():
            calls = sessions.get(completed.get(episode), [])
            if not calls:
                totals["missing_sessions"] += 1
                continue
            totals["completed_sessions"] += 1
            for record in calls:
                info = record.get("info", {})
                expected = {"checkpoint_variant": VARIANT, "checkpoint_step": model["step"],
                    "representation": model["representation_config"]["representation"], "memory_off": role == "memory-off",
                    "payload_sha256": model["training_metadata"]["payload_sha256"],
                    "writer_sha256": model.get("writer_manifest", {}).get("writer_sha256") if model["write_policy"] == "cvom" else None}
                if not set(expected) <= set(info):
                    totals["missing_identity"] += 1
                for key, value in expected.items():
                    if key in info and info[key] != value:
                        raise ValueError(f"Runtime candidate identity differs: {role}/{task}/{key}")
                if record.get("episode_idx") != episode or record.get("episode_seed") != row["episode_seed"]:
                    raise ValueError("Completed session contains another episode")
                memory = info.get("long_memory", {})
                if memory.get("policy") != model["write_policy"]:
                    if "policy" not in memory:
                        totals["missing_identity"] += 1
                    else:
                        raise ValueError(f"Runtime storage policy differs: {role}/{task}")
                enabled = memory.get("memory_read_enabled")
                if type(enabled) is not bool:
                    totals["missing_identity"] += 1
                    continue
                if role == "memory-off" and enabled:
                    raise ValueError("READ-off actually enabled retrieval")
                if record.get("passive") is True and enabled:
                    raise ValueError("Passive demo unexpectedly requested READ")
                metrics = memory.get("read", {})
                delta = metrics.get("ae_conditioning_delta_norm")
                if type(delta) not in (int, float) or not math.isfinite(delta) or delta < 0:
                    totals["missing_identity"] += 1
                else:
                    max_delta = max(max_delta, delta)
                    if not enabled and delta != 0:
                        raise ValueError("READ bypass changed AE input relative to SAME adapted short")
                totals["calls"] += 1
                totals["enabled_calls"] += int(enabled)
    return {**totals, "max_read_delta_norm": max_delta,
        "complete_evidence": totals["completed_sessions"] == len(manifest["settings"]["tasks"]) * manifest["settings"]["n_episodes"]
            and totals["missing_sessions"] == 0 and totals["missing_identity"] == 0}


def write_report(root, bootstrap_samples=5000):
    root = Path(root)
    manifest = json.loads((root / "comparison_manifest.json").read_text())
    validate_manifest_contract(manifest)
    if manifest.get("baseline_reference"):
        ref_root, ref_manifest = validate_reference(manifest["baseline_reference"], manifest)
        result, rendered = build_reference_report(root, manifest, ref_root, ref_manifest, bootstrap_samples=bootstrap_samples)
    else:
        result, rendered = build_report(root, bootstrap_samples=bootstrap_samples)
    result["additional_comparisons"], result["read_diagnostics"] = {}, {}
    lines = [rendered.rstrip()]
    for left, right in (("memory-off", "memory"), ("fifo", "memory")):
        if {left, right} <= set(manifest["models"]):
            contrast = _contrast(root, left, right, manifest["settings"]["tasks"], manifest["settings"]["n_episodes"], bootstrap_samples)
            result["additional_comparisons"][left + "_to_" + right] = contrast
            delta = contrast["paired_task_macro_delta"]
            lines.append(f"{left} -> {right}: " + ("INCOMPLETE" if delta is None else
                f"{delta*100:+.2f}pp; N={contrast['paired_n']}; wins/losses={contrast['wins']}/{contrast['losses']}; p={contrast['mcnemar_exact_p']:.6g}"))
            lines.append(f"95% paired CI: {contrast['paired_task_macro_bootstrap_ci95']}")
    for role in manifest["models"]:
        if role != "baseline":
            diagnostics = completed_read_diagnostics(root, role, manifest)
            result["read_diagnostics"][role] = diagnostics
            lines.append(f"{role} runtime READ evidence: {json.dumps(diagnostics)}")
    lines += ["First 10 scenarios per task are fixed development scenarios, NOT a random or full TEST sample.",
        "A/B/C comparisons require matching training query plan, steps, loss and adaptation budget.",
        "READ-off is a same-actor inference ablation; it is not separately trained memory-free HAMLET."]
    text = "\n".join(lines) + "\n"
    for name, value in (("comparison_summary.json", json.dumps(result, indent=2, allow_nan=False) + "\n"), ("comparison_summary.txt", text)):
        fd, tmp = tempfile.mkstemp(prefix="." + name, dir=root)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(value)
            Path(tmp).replace(root / name)
        finally:
            Path(tmp).unlink(missing_ok=True)
    return result, text


def run_evaluation(args, identity, env):
    output = resolve_repo_path(args.output_dir)
    bind_manifest(output, identity)
    with (output / ".driver.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another driver owns this output") from exc
        failures, interrupted = [], False
        try:
            for role, model in identity["models"].items():
                if role == "baseline" and identity.get("baseline_reference"):
                    validate_reference(identity["baseline_reference"], identity)
                    print(f"[v18] baseline REUSED; newly rolled out 0; {identity['baseline_reference']['source_run']}", flush=True)
                    continue
                pending = []
                for task in args.tasks:
                    rows = read_results(output / role / task / "simulation_results.csv", expected=args.n_episodes)
                    if rows:
                        validate_result_identity(output, role, task, identity)
                    if len(rows) != args.n_episodes:
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
                        log.write("\n[driver] " + json.dumps(command) + "\n")
                        log.flush()
                        server = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        server_ready(server, port, args.server_timeout)
                        for number, task in enumerate(pending, 1):
                            task_dir = folder / task
                            task_dir.mkdir(exist_ok=True)
                            client = [str(python_path(args.robomme_python)), "-u", str(REPO_ROOT / CLIENT),
                                "--task-id", task, "--policy-client-host", "127.0.0.1", "--policy-client-port", str(port),
                                "--dataset", args.dataset, "--n-episodes", str(args.n_episodes),
                                "--max-episode-steps", str(args.max_episode_steps), "--n-action-steps", str(args.n_action_steps),
                                "--model-config", model["base_model"]["path"], "--output-dir", str(task_dir),
                                "--seed", str(args.seed), "--evaluation-id", identity["evaluation_id"] + ":" + role]
                            if args.save_videos:
                                client.append("--save-videos")
                            print(f"[v18] {role}: pending task {number}/{len(pending)} {task}; {task_dir / 'rollout.log'}", flush=True)
                            try:
                                run_client(client, env=env, log_path=task_dir / "rollout.log", server=server, timeout=args.task_timeout)
                                if len(read_results(task_dir / "simulation_results.csv", expected=args.n_episodes)) != args.n_episodes:
                                    raise RuntimeError("Task exited without all completed episodes")
                                validate_result_identity(output, role, task, identity)
                            except (RuntimeError, TimeoutError, ValueError) as exc:
                                failures.append({"model": role, "task": task, "error": str(exc)})
                                print(f"[v18] ERROR: {exc}", file=sys.stderr, flush=True)
                            write_report(output, 500)
                    except (RuntimeError, TimeoutError) as exc:
                        failures.append({"model": role, "error": str(exc)})
                    finally:
                        stop_process(server)
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            (output / "driver_status.json").write_text(json.dumps({"interrupted": interrupted, "failures": failures}, indent=2) + "\n")
            result, report = write_report(output)
            print(report, flush=True)
            print(f"[v18] Saved: {output / 'comparison_summary.txt'}", flush=True)
    return int(bool(failures) or not all(model["complete"] for model in result["models"].values()))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.report_only:
        if args.preflight_only:
            raise ValueError("Choose report-only or preflight-only")
        output = resolve_repo_path(args.output_dir)
        with (output / ".driver.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Evaluation running; read comparison_summary.txt") from exc
            _, report = write_report(output)
        print(report)
        return 0
    validate_options(args)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONUNBUFFERED="1", GR00T_INFERENCE_SEED=str(args.seed))
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    if python_path(Path(sys.executable)) != python_path(args.server_python):
        return subprocess.call([str(python_path(args.server_python)), str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)], cwd=REPO_ROOT, env=env)
    check_dependencies(python_path(args.server_python), "import torch, cv2, transformers, safetensors, msgpack, zmq", env, "policy")
    check_dependencies(python_path(args.robomme_python), "import robomme, msgpack, zmq, pandas, imageio", env, "simulator")
    identity = build_identity(args)
    new_roles = len(args.models) - int(bool(args.baseline_reference))
    print(f"[preflight] roles={args.models}; {len(args.tasks)} tasks x {args.n_episodes}; maximum NEW episodes={new_roles*len(args.tasks)*args.n_episodes}", flush=True)
    if args.preflight_only:
        print("[preflight] No model/simulator loaded; no training/evaluation/output started.")
        return 0
    output = resolve_repo_path(args.output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ("." + output.name + ".v18-launch.lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another launch owns this output") from exc
        return run_evaluation(args, identity, env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[v18] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
