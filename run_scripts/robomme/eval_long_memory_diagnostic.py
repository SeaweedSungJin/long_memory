#!/usr/bin/env python3
"""Frozen v4 RoboMME diagnostics: paired success AND effective intervention coverage.

All adapted conditions use ONE Stage 2 bundle. No-old and shuffled-old intervene
only in reads after the native writer has acted. Their subsequent trajectories
can diverge; they are not same-observation offline counterfactual losses.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from run_scripts.robomme import eval_long_memory_v4 as v4
from run_scripts.robomme.eval_long_memory_comparison import (
    TASKS, bind_manifest, check_dependencies, file_hash, free_local_port, python_path,
    read_results, resolve_repo_path, run_client, server_ready, stop_process, validate_result_identity,
)
from gr00t.eval.sim.robomme.compare_long_memory_results import build_report
from gr00t.eval.sim.robomme.compare_long_memory_v3_results import _contrast, completed_storage_diagnostics

VARIANT = "action_expert_v4_diagnostic_v1"
MODES = ("baseline", "full", "expert-only", "no-old", "shuffled-old", "fifo")
DESCRIPTIONS = {
    "baseline": "Original HAMLET, no adapted expert or long memory",
    "full": "Same Stage 2 adapted expert + reader + learned writer, native reads",
    "expert-only": "Same Stage 2 expert, memory bypassed at inference; NOT a trained no-memory control; writer runs in shadow",
    "no-old": "Same Stage 2 model, remove only events ending strictly before the oldest current short endpoint FROM READS",
    "shuffled-old": "Same Stage 2 model, cyclic same-episode old-content reassignment, destination time preserved; needs >=2 old events",
    "fifo": "Same Stage 2 expert + reader, FIFO instead of learned storage",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=Path("checkpoints/author_hamlet_robomme/checkpoint-60000"))
    parser.add_argument("--memory-checkpoint", type=Path, help="One frozen v4 Stage 2 bundle for ALL adapted modes")
    parser.add_argument("--models", nargs="+", choices=MODES,
                        default=["baseline", "full", "expert-only", "no-old", "shuffled-old", "fifo"])
    parser.add_argument("--tasks", nargs="+", default=["BinFill", "PatternLock"])
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--dataset", choices=("train", "val", "test"), default="test")
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=1300)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/eval/robomme/v4_diagnostic_smoke"))
    parser.add_argument("--server-python", type=Path, default=REPO_ROOT / ".venv/bin/python")
    parser.add_argument("--robomme-python", type=Path, default=Path("/home/sjkim/robomme_benchmark/.venv/bin/python"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--server-timeout", type=float, default=300)
    parser.add_argument("--task-timeout", type=float, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser


def validate_options(args):
    if args.tasks == ["all"]:
        args.tasks = list(TASKS)
    if not args.tasks or any(t not in TASKS for t in args.tasks) or len(set(args.tasks)) != len(args.tasks):
        raise ValueError("--tasks must be unique RoboMME task names or exactly all")
    if "baseline" not in args.models or len(set(args.models)) != len(args.models):
        raise ValueError("--models must be unique and include original baseline")
    if any(m != "baseline" for m in args.models) and args.memory_checkpoint is None:
        raise ValueError("All adapted modes require one --memory-checkpoint from v4 Stage 2")
    if min(args.n_episodes, args.n_action_steps, args.max_episode_steps) < 1 or args.seed < 0:
        raise ValueError("Counts must be positive and seed nonnegative")
    if args.server_timeout <= 0 or args.task_timeout < 0:
        raise ValueError("Invalid server/task timeout")
    if args.preflight_only and args.report_only:
        raise ValueError("Choose preflight-only or report-only, not both")


def build_identity(args):
    # Reuse strict v4 shape/finite/hash/parent-reader/AE validation, not its role
    # mapping (legacy expert-only points to Stage 1, which is wrong for this test).
    base_config = json.loads((resolve_repo_path(args.base_model) / "config.json").read_text())
    window = base_config.get("memory_window")
    if type(window) is not int or window < 1:
        raise ValueError("Diagnostic old-history boundary requires a positive integer base memory_window")
    adapted = any(m != "baseline" for m in args.models)
    native_args = argparse.Namespace(**vars(args))
    native_args.models = ["baseline", "memory"] if adapted else ["baseline"]
    native_args.reader_checkpoint = None
    identity = v4.build_identity(native_args)
    native = identity["models"]
    if adapted and native["memory"]["reader_mode"] != "memory":
        raise ValueError("Diagnostic interventions require a memory-enabled Stage 2 checkpoint")
    models = {}
    for name in args.models:
        entry = dict(native["baseline" if name == "baseline" else "memory"])
        entry.update(diagnostic_mode=name, description=DESCRIPTIONS[name])
        if name == "fifo":
            entry.update(write_policy="all", write_policy_override="all")
        # Expert-only retains a shadow native bank, not disabled storage. This is
        # deliberately distinct from the legacy v4 expert_only server flag.
        entry["read_intervention"] = name if name in ("expert-only", "no-old", "shuffled-old") else "none"
        models[name] = entry
    identity.update(trainer_variant=VARIANT, models=models,
        diagnostic_definition={
            "memory_window": window,
            "old": "event END frame < oldest of actual K most recent endpoint frames, including passive demo",
            "intervention_location": "read only after native write; no stored bank mutation",
            "shuffled_old": "cyclic same-episode old content, preserve destination time; >=2 old entries; not another-episode distractor",
            "expert_only": "same Stage 2 AE inference ablation, not a separately trained no-memory control",
            "closed_loop": "subsequent observations/writes may differ after changed actions",
        })
    for name in ("run_scripts/robomme/eval_long_memory_diagnostic.py",
                 "run_scripts/robomme/serve_long_memory_diagnostic.py",
                 "gr00t/long_memory/diagnostic_online.py"):
        identity["source_sha256"][name] = file_hash(REPO_ROOT / name)
    identity.pop("evaluation_id", None)
    identity["evaluation_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def completed_intervention_coverage(root, model, tasks, expected):
    """Only completed, seed-matched sessions count; exclude passive priming calls.

    A session may occur again after a crash. The last matching completion selects
    that session; failed/incomplete attempts never inflate coverage.
    """
    totals = dict(completed_sessions_with_diagnostics=0, completed_sessions_missing_diagnostics=0,
                  executed_policy_calls=0, calls_with_old=0, calls_with_two_old=0,
                  calls_intervention_applied=0, calls_fused_changed=0, ignored_torn_final_lines=0)
    for task in tasks:
        rows = read_results(root / model / task / "simulation_results.csv", expected=expected)
        path = root / model / task / "memory_diagnostics.jsonl"
        sessions, completed = {}, {}
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        if not line.endswith("\n"):
                            totals["ignored_torn_final_lines"] += 1
                            continue
                        raise ValueError(f"Malformed diagnostic journal: {path}")
                    sid = record.get("session_id")
                    if record.get("kind") == "policy_call" and not record.get("passive", False):
                        diag = record.get("info", {}).get("long_memory", {})
                        if diag.get("diagnostic_mode") == model:
                            calls = sessions.setdefault(sid, {})
                            if record.get("call") in calls:
                                raise ValueError(f"Duplicate policy call in diagnostic session: {path}")
                            calls[record.get("call")] = diag
                    elif record.get("kind") == "episode_complete":
                        row = rows.get(record.get("episode_idx"))
                        if row and record.get("episode_seed") == row["episode_seed"] and record.get("success") == row["success"]:
                            completed[record["episode_idx"]] = sid
        for episode in rows:
            calls = sessions.get(completed.get(episode))
            if not calls:
                totals["completed_sessions_missing_diagnostics"] += 1
                continue
            totals["completed_sessions_with_diagnostics"] += 1
            for diag in calls.values():
                old = diag.get("old_event_count")
                if type(old) is not int or old < 0:
                    raise ValueError("Invalid old-event coverage count")
                totals["executed_policy_calls"] += 1
                totals["calls_with_old"] += int(old > 0)
                totals["calls_with_two_old"] += int(old >= 2)
                for field, metric in (("intervention_applied", "calls_intervention_applied"),
                                      ("intervention_changed_fused", "calls_fused_changed")):
                    if type(diag.get(field)) is not bool:
                        raise ValueError(f"Invalid {field} in diagnostic journal")
                    totals[metric] += int(diag[field])
    denominator = totals["executed_policy_calls"]
    totals["old_coverage_fraction"] = totals["calls_with_old"] / denominator if denominator else None
    totals["effective_intervention_fraction"] = totals["calls_fused_changed"] / denominator if denominator else None
    return totals


def validate_diagnostic_manifest(manifest):
    """Reject mislabeled roles or edited provenance even in report-only mode."""
    if manifest.get("trainer_variant") != VARIANT:
        raise ValueError("Not a diagnostic evaluation manifest")
    models = manifest.get("models", {})
    if not isinstance(models, dict) or "baseline" not in models or set(models) - set(MODES):
        raise ValueError("Diagnostic manifest has unsupported/missing model roles")
    signature = None
    for name, entry in models.items():
        if entry.get("diagnostic_mode") != name:
            raise ValueError("Diagnostic model role does not match its saved intervention")
        if name == "baseline":
            if entry.get("memory_checkpoint") is not None or entry.get("expert_weights_sha256") is not None:
                raise ValueError("Original HAMLET baseline cannot contain adapted expert weights")
            continue
        if entry.get("stage") != 2 or entry.get("reader_mode") != "memory":
            raise ValueError("Every adapted diagnostic role must use a memory-enabled Stage 2 bundle")
        current = tuple(entry.get(key) for key in
            ("memory_checkpoint", "checkpoint_sha256", "weights_sha256", "expert_weights_sha256"))
        if any(not value for value in current) or (signature is not None and current != signature):
            raise ValueError("Diagnostic roles must use the SAME Stage 2 checkpoint/reader/expert hashes")
        signature = current
        expected = "all" if name == "fifo" else "hard"
        if entry.get("write_policy") != expected:
            raise ValueError(f"Diagnostic {name} has the wrong storage policy")
    content = {key: value for key, value in manifest.items() if key != "evaluation_id"}
    checksum = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    if manifest.get("evaluation_id") != checksum:
        raise ValueError("Diagnostic manifest checksum changed; refusing contaminated provenance")


def write_report(output, *, bootstrap_samples=5000):
    root = Path(output)
    manifest = json.loads((root / "comparison_manifest.json").read_text())
    validate_diagnostic_manifest(manifest)
    result, report = build_report(root, bootstrap_samples=bootstrap_samples)
    result.update(trainer_variant=VARIANT, diagnostic_comparisons={}, intervention_coverage={}, storage_diagnostics={})
    lines = [report.rstrip(), "", "Frozen SAME-checkpoint diagnostic contrasts"]
    models, settings = manifest["models"], manifest["settings"]
    for left in ("expert-only", "no-old", "shuffled-old", "fifo"):
        if left not in models or "full" not in models:
            continue
        contrast = _contrast(root, left, "full", settings["tasks"], settings["n_episodes"], bootstrap_samples)
        result["diagnostic_comparisons"][f"{left}_to_full"] = contrast
        delta, ci = contrast["paired_task_macro_delta"], contrast["paired_task_macro_bootstrap_ci95"]
        if delta is None:
            lines.append(f"  {left} -> full: INCOMPLETE, no paired episodes")
        else:
            interval = "--" if ci is None else f"[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]pp"
            lines.append(f"  {left} -> full: {100*delta:+.2f}pp, N={contrast['paired_n']}, "
                f"wins/losses/same={contrast['wins']}/{contrast['losses']}/{contrast['same']}, "
                f"95% CI={interval}, McNemar p={contrast['mcnemar_exact_p']:.6g}")
    lines += ["", "Effective intervention coverage (completed sessions, execution calls only)"]
    for model in models:
        if model == "baseline":
            continue
        coverage = completed_intervention_coverage(root, model, settings["tasks"], settings["n_episodes"])
        result["intervention_coverage"][model] = coverage
        result["storage_diagnostics"][model] = completed_storage_diagnostics(
            root, model, settings["tasks"], settings["n_episodes"])
        lines.append(f"  {model}: calls={coverage['executed_policy_calls']}, old>=1={coverage['calls_with_old']}, "
            f"old>=2={coverage['calls_with_two_old']}, applied={coverage['calls_intervention_applied']}, "
            f"fused changed={coverage['calls_fused_changed']}, missing completed-session logs="
            f"{coverage['completed_sessions_missing_diagnostics']}")
    lines += ["", "Interpretation limits:",
        "  full > expert-only tests using added memory with the SAME adapted AE, not a separately trained control.",
        "  full > no-old with nonzero intervention coverage supports old-history utility; absent effect is inconclusive.",
        "  shuffled-old changes content/time association within the SAME old set, not total information availability.",
        "  Writer state is not changed by a read intervention at that decision; later trajectories may diverge.",
        "  FIFO -> full isolates the deployed storage policy with identical reader/AE weights.",
        "  CI/p values are exploratory for this task panel and inference seed; no multiple-comparison adjustment."]
    report = "\n".join(lines) + "\n"
    for name, payload in (("comparison_summary.json", json.dumps(result, indent=2, allow_nan=False) + "\n"),
                          ("comparison_summary.txt", report)):
        descriptor, filename = tempfile.mkstemp(prefix=f".{name}-", dir=root)
        path = Path(filename)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            path.replace(root / name)
        finally:
            path.unlink(missing_ok=True)
    return result, report


@contextmanager
def driver_lock(output):
    # Lock before bind_manifest, including initial directory creation. The lock
    # is in the parent so a NEW run need not violate bind_manifest's empty-dir rule.
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = hashlib.sha256(str(output).encode()).hexdigest()[:20]
    with (output.parent / f".diagnostic-driver-{suffix}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another diagnostic evaluation driver owns {output}")
        yield


def server_command(args, model, port):
    command = [str(python_path(args.server_python)), "-u",
        str(REPO_ROOT / "run_scripts/robomme/serve_long_memory_diagnostic.py"),
        "--base-model", model["base_model"]["path"], "--mode", model["diagnostic_mode"],
        "--device", args.device, "--host", "127.0.0.1", "--port", str(port)]
    if model["memory_checkpoint"]:
        command += ["--memory-checkpoint", model["memory_checkpoint"]]
    return command


def run_evaluation(args, identity, env):
    output = resolve_repo_path(args.output_dir)
    failures, interrupted = [], False
    with driver_lock(output):
        validate_diagnostic_manifest(identity)
        bind_manifest(output, identity)
        try:
            for name, model in identity["models"].items():
                pending = []
                for task in args.tasks:
                    rows = read_results(output / name / task / "simulation_results.csv", expected=args.n_episodes)
                    if rows:
                        validate_result_identity(output, name, task, identity)
                    if len(rows) != args.n_episodes:
                        pending.append(task)
                    else:
                        print(f"[diagnostic-eval] resume: {name}/{task} complete", flush=True)
                if not pending:
                    continue
                folder = output / name
                folder.mkdir(exist_ok=True)
                port, server = free_local_port(), None
                command = server_command(args, model, port)
                with (folder / "server.log").open("a", encoding="utf-8") as log:
                    try:
                        print(f"[diagnostic-eval] loading {name}; {folder / 'server.log'}", flush=True)
                        log.write("\n[driver] command: " + json.dumps(command) + "\n")
                        log.flush()
                        server = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log,
                                                  stderr=subprocess.STDOUT, start_new_session=True)
                        server_ready(server, port, args.server_timeout)
                        for task in pending:
                            task_dir = folder / task
                            task_dir.mkdir(exist_ok=True)
                            client = [str(python_path(args.robomme_python)), "-u",
                                str(REPO_ROOT / "gr00t/eval/sim/robomme/run_long_memory_rollout.py"),
                                "--task-id", task, "--policy-client-host", "127.0.0.1", "--policy-client-port", str(port),
                                "--dataset", args.dataset, "--n-episodes", str(args.n_episodes),
                                "--max-episode-steps", str(args.max_episode_steps), "--n-action-steps", str(args.n_action_steps),
                                "--model-config", model["base_model"]["path"], "--output-dir", str(task_dir),
                                "--seed", str(args.seed), "--evaluation-id", identity["evaluation_id"] + ":" + name]
                            if args.save_videos:
                                client.append("--save-videos")
                            print(f"[diagnostic-eval] {name}/{task}; {task_dir / 'rollout.log'}", flush=True)
                            try:
                                run_client(client, env=env, log_path=task_dir / "rollout.log", server=server, timeout=args.task_timeout)
                                rows = read_results(task_dir / "simulation_results.csv", expected=args.n_episodes)
                                if len(rows) != args.n_episodes:
                                    raise RuntimeError("Rollout exited without all requested completed rows")
                                validate_result_identity(output, name, task, identity)
                            except (RuntimeError, TimeoutError, ValueError) as exc:
                                failures.append({"model": name, "task": task, "error": str(exc)})
                                print(f"[diagnostic-eval] ERROR {name}/{task}: {exc}", file=sys.stderr, flush=True)
                            write_report(output, bootstrap_samples=500)
                    except (RuntimeError, TimeoutError) as exc:
                        failures.append({"model": name, "error": str(exc)})
                        print(f"[diagnostic-eval] ERROR {name}: {exc}; see server.log", file=sys.stderr, flush=True)
                    finally:
                        stop_process(server)
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            (output / "driver_status.json").write_text(json.dumps({"interrupted": interrupted, "failures": failures}, indent=2) + "\n")
            result, report = write_report(output)
            print(report, flush=True)
            print(f"[diagnostic-eval] Summary: {output / 'comparison_summary.txt'}", flush=True)
    return int(bool(failures) or not all(model["complete"] for model in result["models"].values()))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.report_only:
        if args.preflight_only:
            raise ValueError("Choose report-only or preflight-only, not both")
        output = resolve_repo_path(args.output_dir)
        if not (output / "comparison_manifest.json").is_file():
            raise FileNotFoundError("Existing diagnostic comparison_manifest.json is required for report-only")
        with driver_lock(output):
            _, report = write_report(output)
        print(report)
        return 0
    validate_options(args)
    env = dict(os.environ)
    env.update(PYTHONPATH=str(REPO_ROOT), PYTHONUNBUFFERED="1", GR00T_INFERENCE_SEED=str(args.seed))
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    server_python = python_path(args.server_python)
    if python_path(Path(sys.executable)) != server_python:
        return subprocess.call([str(server_python), str(Path(__file__).resolve()),
                                *(sys.argv[1:] if argv is None else argv)], cwd=REPO_ROOT, env=env)
    check_dependencies(server_python, "import torch, cv2, transformers, safetensors, msgpack, zmq", env, "policy")
    check_dependencies(python_path(args.robomme_python), "import robomme, msgpack, zmq, pandas, imageio", env, "simulator")
    identity = build_identity(args)
    print(f"[preflight] compatible diagnostic modes={args.models}; tasks={args.tasks}; n={args.n_episodes}", flush=True)
    if args.preflight_only:
        print("[preflight] No model/simulator loaded; no output, training or evaluation started.")
        return 0
    return run_evaluation(args, identity, env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[diagnostic-eval] Interrupted; owned processes stopped, completed episodes resumable.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[diagnostic-eval] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
