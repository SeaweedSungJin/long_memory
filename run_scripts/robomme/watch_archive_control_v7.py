#!/usr/bin/env python3
"""Observe one already-live V7 archive trainer, then run bounded VAL controls.

This never starts, restarts, signals, or writes into training. A Linux pidfd
binds the observed process; completion metadata alone never releases the wait.
Only NEW workflow/evaluation directories are accepted. Offline loss and these
queued rollouts are not a declaration of the >=30% research target.
"""
import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.hamlet import validate_cache_checkpoint
from gr00t.long_memory.monitoring import _atomic_json
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.run_long_memory_v8_workflow import run_step


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_process_identity(pid, proc_root=Path("/proc")):
    directory = Path(proc_root) / str(pid)
    stat = (directory / "stat").read_text()
    # comm can itself contain spaces and ')'; fields after its last ')' begin
    # at field 3 (state), making field 22 (starttime) index 19 here.
    fields = stat[stat.rindex(")") + 2:].split()
    return {"pid": pid, "starttime": int(fields[19]), "state": fields[0],
            "cwd": str((directory / "cwd").resolve(strict=True)),
            "argv": [os.fsdecode(x) for x in (directory / "cmdline").read_bytes().split(b"\0") if x]}


def _option(argv, name):
    if argv.count(name) != 1:
        raise ValueError(f"Trainer argv must contain exactly one {name}")
    index = argv.index(name)
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise ValueError(f"Trainer argv has no value for {name}")
    return argv[index + 1]


def validate_process_identity(identity, *, starttime, run, cache, config):
    argv = identity["argv"]
    if identity["starttime"] != starttime or identity["cwd"] != str(ROOT):
        raise ValueError("Trainer PID was reused or has the wrong starttime/cwd")
    if identity["state"] in ("Z", "X", "x") or len(argv) < 2:
        raise ValueError("Watcher must bind an already-live trainer, not an exited process")
    if (ROOT / argv[0]).absolute() != ROOT / ".venv/bin/python" or (ROOT / argv[1]).resolve() != ROOT / "run_scripts/robomme/train_long_memory_v7.py":
        raise ValueError("PID is not the expected V7 trainer executable/entrypoint")
    for key, expected in (("--stage", "1"), ("--mode", "archive"),
                          ("--max-epochs", str(config["train"]["max_epochs"]))):
        if _option(argv, key) != expected:
            raise ValueError(f"Unexpected trainer option {key}")
    for key, expected in (("--output-dir", run), ("--cache-dir", cache)):
        if (ROOT / _option(argv, key)).resolve() != Path(expected).resolve():
            raise ValueError(f"Unexpected trainer path {key}")
    if any(flag in argv for flag in ("--resume", "--max-steps", "--stop-after-steps", "--preflight-only")):
        raise ValueError("Only the original uninterrupted full-coverage trainer is supported")


def open_pidfd(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid, 0)
    # The packaged Python omits os.pidfd_open; the host glibc still provides
    # the same Linux syscall. No numeric syscall/architecture guessing needed.
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "pidfd_open"):
        raise RuntimeError("Linux pidfd_open is required; no status-only fallback")
    libc.pidfd_open.argtypes = (ctypes.c_int, ctypes.c_uint)
    libc.pidfd_open.restype = ctypes.c_int
    fd = libc.pidfd_open(pid, 0)
    if fd < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return fd


def bind_trainer(pid, starttime, run, cache, config):
    before = read_process_identity(pid)
    validate_process_identity(before, starttime=starttime, run=run, cache=cache, config=config)
    fd = open_pidfd(pid)
    try:
        after = read_process_identity(pid)
        validate_process_identity(after, starttime=starttime, run=run, cache=cache, config=config)
        if any(before[key] != after[key] for key in ("pid", "starttime", "cwd", "argv")):
            raise RuntimeError("Trainer identity changed while binding pidfd")
    except BaseException:
        os.close(fd)
        raise
    return fd, after


def wait_for_exit(fd, poll_seconds=30, heartbeat=None):
    if not 0 < poll_seconds <= 30:
        raise ValueError("Polling interval must be in (0, 30] seconds")
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    while True:
        events = poller.poll(max(1, int(poll_seconds * 1000)))
        for ready_fd, flags in events:
            if ready_fd != fd or flags & (select.POLLERR | select.POLLNVAL):
                raise RuntimeError("Invalid trainer pidfd poll event")
            if flags & (select.POLLIN | select.POLLHUP):
                return  # This exact process exited, even if its numeric PID was reused.
            raise RuntimeError("Unexpected trainer pidfd poll event")
        if heartbeat is not None:
            heartbeat()


def coverage_plan(run, config):
    train = config["train"]
    if (config.get("trainer_variant") != "recurrent_memory_v7" or config.get("stage") != 1
            or config.get("mode") != "archive" or train.get("stage") != 1 or train.get("mode") != "archive"):
        raise ValueError("Expected exact V7 Stage-1 archive training configuration")
    if any(train.get(key) is not None for key in ("resume", "max_steps", "stop_after_steps")):
        raise ValueError("Only original full-epoch archive runs are supported")
    epochs = train["max_epochs"]
    if type(epochs) is not int or epochs < 1:
        raise ValueError("Invalid planned epoch count")
    plan = read_json(Path(run) / "query_plan.json")
    windows, plans = plan["windows"], plan["plans"]
    digest = hashlib.sha256(json.dumps({"plans": plans, "windows": windows},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if plan["sha256"] != digest or not windows:
        raise ValueError("Coverage plan hash mismatch or empty plan")
    expected = Counter((eid, q) for eid, queries in plans["train"] for q in queries)
    if not expected or any(count != 1 for count in expected.values()) or sum(expected.values()) != plans["train_query_count"]:
        raise ValueError("Invalid per-epoch training query coverage")
    observed = [Counter() for _ in range(epochs)]
    epoch1_step = None
    previous_epoch = 0
    for index, window in enumerate(windows):
        epoch = window["epoch"]
        if type(epoch) is not int or not 0 <= epoch < epochs or epoch < previous_epoch or epoch > previous_epoch + 1:
            raise ValueError("Invalid epoch ordering in coverage plan")
        queries = [(eid, q) for eid, qs in window["groups"] for q in qs]
        if not queries or len(queries) != window["query_count"]:
            raise ValueError("Coverage window query_count does not match its groups")
        observed[epoch].update(queries)
        is_end = index + 1 == len(windows) or windows[index + 1]["epoch"] != epoch
        if type(window["epoch_end"]) is not bool or window["epoch_end"] != is_end:
            raise ValueError("Invalid epoch-end marker")
        if epoch == 0 and is_end:
            epoch1_step = index + 1
        previous_epoch = epoch
    if any(actual != expected for actual in observed):
        raise ValueError("Plan does not cover every training query exactly once per epoch")
    return {"step": len(windows), "processed_queries": sum(w["query_count"] for w in windows),
            "epoch": epochs, "plan_sha256": digest, "epoch1_step": epoch1_step}


def validate_outputs(workflow, evaluation, *protected):
    if Path(workflow).is_symlink() or Path(evaluation).is_symlink():
        raise FileExistsError("Refusing symlink output paths")
    workflow, evaluation = Path(workflow).resolve(), Path(evaluation).resolve()
    for output, parent in ((workflow, ROOT / "runs/long_memory"), (evaluation, ROOT / "runs/eval/robomme")):
        if output == parent or not output.is_relative_to(parent):
            raise ValueError("Use a dedicated workflow/eval directory under the established runs roots")
        validate_output_scope(output, *protected)
        for source in protected:
            if source is not None and Path(source).resolve().is_relative_to(output):
                raise ValueError("Output cannot contain a protected input")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing old output: {output}")
    if workflow.is_relative_to(evaluation) or evaluation.is_relative_to(workflow):
        raise ValueError("Workflow and evaluation outputs must be disjoint")


def _checkpoint(run, path, step, config, coverage, provenance):
    path = Path(path)
    path = (path if path.is_absolute() else Path(run) / path).resolve()
    if path.parent != Path(run).resolve() or path.name != f"checkpoint-{step:06d}":
        raise ValueError("Checkpoint pointer escapes its run or disagrees with its step")
    info = read_json(path / "checkpoint.json")
    meta = info["metadata"]
    if info.get("format_version") != 1 or info.get("step") != step or info.get("config") != config:
        raise ValueError("Checkpoint config/step is not the exact archive run")
    for key in ("base_model", "cache_fingerprint", "cache_dir", "source_sha256"):
        if meta.get(key) != provenance.get(key):
            raise ValueError(f"Checkpoint provenance mismatch: {key}")
    if meta.get("plan_sha256") != coverage["plan_sha256"]:
        raise ValueError("Checkpoint plan differs from the verified coverage plan")
    state = meta["train_state"]
    if state.get("window_cursor") != step or state.get("optimizer_updates") != step:
        raise ValueError("Checkpoint is not at its claimed optimizer boundary")
    return path, info


def verify_completion(run, config, coverage, provenance, *, include_epoch1=False):
    run = Path(run)
    status = read_json(run / "status.json")
    required = {"status": "complete", "step": coverage["step"], "window_cursor": coverage["step"],
                "optimizer_updates": coverage["step"], "processed_queries": coverage["processed_queries"],
                "epoch": coverage["epoch"]}
    if any(status.get(key) != value for key, value in required.items()):
        raise RuntimeError("Trainer exited without complete, exact planned coverage")
    last = read_json(run / "last_checkpoint.json")
    if last.get("step") != coverage["step"]:
        raise ValueError("Last checkpoint does not reach the planned horizon")
    last_path, last_info = _checkpoint(run, last["path"], last["step"], config, coverage, provenance)
    if any(last_info["metadata"]["train_state"].get(key) != value for key, value in required.items() if key != "step"):
        raise ValueError("Last checkpoint metadata does not prove completed coverage")
    result = {}
    if include_epoch1:
        step = coverage["epoch1_step"]
        path, info = _checkpoint(run, f"checkpoint-{step:06d}", step, config, coverage, provenance)
        if (info["metadata"]["train_state"].get("epoch") != 1
                or info["metadata"]["train_state"].get("processed_queries") != coverage["processed_queries"] // coverage["epoch"]):
            raise ValueError("Pinned epoch-1 checkpoint does not finish epoch 1")
        result["epoch1"] = path
    if last_path not in result.values():
        result["last"] = last_path
    best = read_json(run / "best_checkpoint.json")
    step = best.get("step")
    if type(step) is not int or not 0 <= step <= coverage["step"]:
        raise ValueError("Invalid best-checkpoint step")
    if step:
        path, _ = _checkpoint(run, best["path"], step, config, coverage, provenance)
        if path not in result.values():
            result["best"] = path
    return result


def eval_command(checkpoint, output, base, gpu="1"):
    if not str(gpu).isdigit():
        raise ValueError("GPU must be a single nonnegative physical CUDA index")
    return ["env", f"CUDA_VISIBLE_DEVICES={gpu}", str(ROOT / ".venv/bin/python"),
        "run_scripts/robomme/eval_long_memory_v7.py", "--base-model", str(base),
        "--models", "baseline", "archive", "--archive-checkpoint", str(checkpoint),
        "--tasks", "all", "--dataset", "val", "--n-episodes", "10", "--seed", "6",
        "--device", "cuda:0", "--output-dir", str(output)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-pid", type=int, default=3662553)
    parser.add_argument("--expected-starttime", type=int, default=109671044)
    parser.add_argument("--run-dir", default="runs/long_memory/v7_archive_full_v1")
    parser.add_argument("--cache-dir", default="runs/long_memory/cache_full1600_v1")
    parser.add_argument("--workflow-dir", default="runs/long_memory/v7_archive_full_v1_eval_workflow")
    parser.add_argument("--eval-dir", default="runs/eval/robomme/v7_archive_full_v1_val_n10_seed6")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--include-epoch1", action="store_true", help="Evaluate the pinned first full epoch before last/best")
    parser.add_argument("--preflight-only", action="store_true", help="Validate inputs/live PID and print plan; no writes/wait/evaluation")
    args = parser.parse_args(argv)
    if min(args.trainer_pid, args.expected_starttime) < 1 or not 0 < args.poll_seconds <= 30 or not args.gpu.isdigit():
        raise ValueError("Invalid PID/starttime, polling interval, or physical GPU index")
    if any((ROOT / p).is_symlink() for p in (args.workflow_dir, args.eval_dir)):
        raise FileExistsError("Refusing symlink output paths")
    run, cache, workflow, evaluation = [(ROOT / p).resolve() for p in
        (args.run_dir, args.cache_dir, args.workflow_dir, args.eval_dir)]
    config = read_json(run / "run_config.json")
    provenance = read_json(run / "provenance.json")
    coverage = coverage_plan(run, config)
    manifest = EpisodeCache(cache).manifest
    validate_cache_checkpoint(manifest)
    base = Path(manifest["model_path"]).resolve()
    if ((ROOT / config["train"]["output_dir"]).resolve() != run
            or (ROOT / config["train"]["cache_dir"]).resolve() != cache
            or Path(provenance["cache_dir"]).resolve() != cache
            or provenance["base_model"]["path"] != str(base)
            or provenance["cache_fingerprint"] != manifest["fingerprint"]
            or provenance["plan_sha256"] != coverage["plan_sha256"]):
        raise ValueError("Run/cache/base/coverage provenance differs")
    protected = [run, cache, base, manifest.get("dataset_path"),
                 *(ROOT / name for name in ("gr00t", "run_scripts", "tests", "docs"))]
    validate_outputs(workflow, evaluation, *protected)
    sources = provenance["source_sha256"]
    if not sources:
        raise ValueError("Missing trained source identity")
    for relative, expected in sources.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or file_hash(path) != expected:
            raise ValueError(f"Trained source changed: {relative}")
    frozen_files = [run / name for name in ("run_config.json", "query_plan.json", "provenance.json")]
    frozen_files += [cache / "manifest.json", *(ROOT / name for name in sources), Path(__file__),
                     ROOT / "run_scripts/robomme/eval_long_memory_v7.py",
                     ROOT / "run_scripts/robomme/run_long_memory_v8_workflow.py"]
    hashes = {str(path): file_hash(path) for path in frozen_files}
    fd, identity = bind_trainer(args.trainer_pid, args.expected_starttime, run, cache, config)
    plan = {"args": vars(args), "trainer_identity": identity, "coverage": coverage,
            "frozen_inputs_sha256": hashes, "base_model": str(base),
            "candidate_order": (["epoch1"] if args.include_epoch1 else []) + ["last", "nonzero_best"],
            "evaluation": "baseline+archive; val all 16 tasks; 10 episodes/task; seed 6; deduplicate paths",
            "training_writes": False, "training_signals": False, "goal_achieved": False}
    if args.preflight_only:
        os.close(fd)
        print(json.dumps(plan, indent=2))
        return 0
    created = False
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Watcher interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        workflow.mkdir(parents=True, exist_ok=False)
        created = True
        _atomic_json(workflow / "plan.json", plan)
        def heartbeat():
            record = {"status": "waiting_for_exact_trainer_exit", "trainer_pid": args.trainer_pid,
                      "trainer_starttime": args.expected_starttime, "updated_at": datetime.now(timezone.utc).isoformat()}
            _atomic_json(workflow / "status.json", record)
            print(f"[v7-archive-watch] Waiting for exact PID {args.trainer_pid} to exit", flush=True)
        heartbeat()
        wait_for_exit(fd, args.poll_seconds, heartbeat)
        _atomic_json(workflow / "trainer_exit.json", {"pidfd_confirmed_exit": True,
            "trainer_identity": identity, "observed_at": datetime.now(timezone.utc).isoformat(),
            "exit_code": None, "note": "Not our child; exit code unavailable. Completion checks are mandatory."})
        if any(file_hash(path) != expected for path, expected in hashes.items()):
            raise ValueError("An immutable input/source changed while waiting")
        candidates = verify_completion(run, config, coverage, provenance, include_epoch1=args.include_epoch1)
        # Recheck the evaluation target after the wait; workflow is now owned.
        validate_output_scope(evaluation, *protected, workflow)
        if evaluation.exists() or evaluation.is_symlink():
            raise FileExistsError(f"Evaluation output appeared while waiting: {evaluation}")
        commands = {name: eval_command(checkpoint, evaluation / name, base, args.gpu)
                    for name, checkpoint in candidates.items()}
        _atomic_json(workflow / "resolved_evaluations.json", {"checkpoints": {k: str(v) for k, v in candidates.items()},
            "commands": commands, "coverage": coverage})
        # Validate every Stage-1 archive payload/base before starting any rollout.
        for name, command in commands.items():
            run_step(f"{name}_eval_preflight", command + ["--preflight-only"], workflow)
        evaluation.mkdir(parents=True, exist_ok=False)
        for name, command in commands.items():
            run_step(f"{name}_eval", command, workflow)
        _atomic_json(workflow / "status.json", {"status": "iteration_complete", "goal_achieved": False,
            "evaluations": {name: str(evaluation / name / "comparison_summary.txt") for name in candidates},
            "note": "Archive control VAL only; no RoboMME success claim inferred from offline losses."})
        return 0
    except BaseException as error:
        if created:
            _atomic_json(workflow / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error": f"{type(error).__name__}: {error}", "goal_achieved": False,
                "training_writes": False, "training_signals": False})
        raise
    finally:
        os.close(fd)
        signal.signal(signal.SIGTERM, previous_term_handler)


if __name__ == "__main__":
    raise SystemExit(main())
