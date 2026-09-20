#!/usr/bin/env python3
"""Compare original HAMLET, action-only Stage 1, and recall-supervised Stage 1.

This is an evaluation-only coordinator. Both adapted models use the unchanged
v4 policy server and FIFO storage; auxiliary heads never enter inference.
The v4 engine owns subprocesses, per-episode journals and resumable manifests.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_scripts.robomme import eval_long_memory_v5 as v5
from run_scripts.robomme.long_memory_v5_workflow import best_checkpoint
from gr00t.eval.sim.robomme.compare_long_memory_v4_results import build_v4_report
from gr00t.eval.sim.robomme.compare_long_memory_v3_results import _contrast

engine = v5.engine
EXPERIMENT = "v5_stage1_recall_vs_action_control"
TRAIN_RUN = Path("runs/long_memory/v5_recall_full_v1")


def build_parser():
    parser = v5.build_parser()
    parser.description = __doc__
    parser.set_defaults(tasks=["all"], n_episodes=50, dataset="val",
        output_dir=Path("runs/eval/robomme/v5_recall_vs_action_val_n50_seed6"))
    parser.add_argument("--reader-run", type=Path, default=TRAIN_RUN / "stage1",
                        help="Resolve best_checkpoint.json unless --reader-checkpoint is explicit")
    parser.add_argument("--control-run", type=Path, default=TRAIN_RUN / "action_control")
    parser.add_argument("--control-checkpoint", type=Path,
                        help="Action-only Stage 1 bundle; defaults to --control-run best")
    return parser


def validate_training_pair(reader, control):
    """Refuse an apparently controlled experiment with different training inputs."""
    r, c = reader["config"], control["config"]
    for key in ("memory", "expert", "expert_targets", "recall"):
        if r[key] != c[key]:
            raise ValueError(f"Stage 1 architecture differs: {key}")
    for key in ("cache_fingerprint", "recall_labels_fingerprint", "source_sha256",
                "initial_checkpoint_sha256"):
        if not reader["metadata"].get(key) or reader["metadata"][key] != control["metadata"].get(key):
            raise ValueError(f"Stage 1 training provenance differs or is missing: {key}")
    rt, ct = r["train"], c["train"]
    allowed = {"output_dir", "subgoal_weight", "grounding_weight"}
    differing = [key for key in rt.keys() | ct.keys()
                 if key not in allowed and rt.get(key) != ct.get(key)]
    if differing:
        raise ValueError(f"Training settings differ beyond auxiliary losses: {sorted(differing)}")
    if rt.get("reader_mode") != "memory" or ct.get("reader_mode") != "memory":
        raise ValueError("Both controls must train/use the long-memory reader")
    rw = [rt.get(k) for k in ("subgoal_weight", "grounding_weight")]
    cw = [ct.get(k) for k in ("subgoal_weight", "grounding_weight")]
    if (any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in rw + cw)
            or not any(rw) or any(cw)):
        raise ValueError("reader needs positive auxiliary supervision; action control needs both weights zero")
    plans = [Path(t["output_dir"]) / "sampling_plan.json" for t in (rt, ct)]
    if json.loads(plans[0].read_text()) != json.loads(plans[1].read_text()):
        raise ValueError("Stage 1 sampling plans differ")


def build_identity(args):
    # Validate both bundles and output isolation before any evaluation writes.
    left = copy.deepcopy(args)
    left.reader_checkpoint = args.control_checkpoint
    reader_info = v5.validate_bundles(args)[1][1]
    control_info = v5.validate_bundles(left)[1][1]
    validate_training_pair(reader_info, control_info)
    identity, control_identity = v5.build_identity(args), v5.build_identity(left)
    for key in ("settings", "source_sha256", "base_file_sha256", "benchmark",
                "policy_package_versions", "server_python", "robomme_python"):
        if identity[key] != control_identity[key]:
            raise ValueError(f"Evaluation inputs changed between model preflights: {key}")
    identity["experiment"] = EXPERIMENT
    identity["models"] = {
        "baseline": identity["models"]["baseline"],
        "action_control": control_identity["models"]["reader"],
        "reader": identity["models"]["reader"],
    }
    identity["models"]["action_control"]["description"] = (
        "Action-only Stage 1: adapted Action Expert + long memory, FIFO; no auxiliary loss")
    identity["models"]["reader"]["description"] = (
        "Recall-supervised Stage 1: adapted Action Expert + long memory, FIFO; auxiliary heads are training-only")
    for name in (Path(__file__).name, "eval_long_memory_v5_controls.sh"):
        path = Path("run_scripts/robomme") / name
        identity["source_sha256"][str(path)] = engine.file_hash(ROOT / path)
    identity.pop("evaluation_id", None)
    identity["evaluation_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def write_control_report(output, *, bootstrap_samples=5000):
    """Add the primary auxiliary-loss contrast, not just two baseline contrasts."""
    output = Path(output)
    manifest = json.loads((output / "comparison_manifest.json").read_text())
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError("Not a v5 action-only vs recall comparison directory")
    result, report = build_v4_report(output, bootstrap_samples=bootstrap_samples)
    settings = manifest["settings"]
    # Validate this pair directly even when the original baseline failed to run.
    for task in settings["tasks"]:
        pair = []
        for role in ("action_control", "reader"):
            if engine.read_results(output / role / task / "simulation_results.csv", expected=settings["n_episodes"]):
                pair.append(engine.validate_result_identity(output, role, task, manifest))
        if len(pair) == 2:
            for key in ("scenario_metadata_sha256", "model_config_sha256", "memory_window", "demo_sampling"):
                if pair[0].get(key) != pair[1].get(key):
                    raise ValueError(f"action_control/reader {task}: mismatched {key}")
    contrast = _contrast(output, "action_control", "reader", settings["tasks"],
                         settings["n_episodes"], bootstrap_samples)
    result["experiment"] = EXPERIMENT
    result["additional_comparisons"]["action_control_to_reader"] = contrast
    lines = [report.rstrip(), "", "PRIMARY COMPARISON: action_control -> reader (recall auxiliary losses)",
             "Positive delta / wins favor recall supervision; both models use FIFO.",
             f"{'Task':<22} {'Paired N':>8} {'Delta':>10} {'Wins/Losses/Same':>19}"]
    for task, row in contrast["tasks"].items():
        delta = "--" if row["paired_delta"] is None else f"{100*row['paired_delta']:+.2f}pp"
        lines.append(f"{task:<22} {row['paired_n']:>8} {delta:>10} "
                     f"{row['wins']:>5}/{row['losses']}/{row['same']}")
    delta = contrast["paired_task_macro_delta"]
    if delta is not None:
        lines.append(f"Paired task-macro delta: {100*delta:+.2f}pp; N={contrast['paired_n']}; "
                     f"wins/losses/same={contrast['wins']}/{contrast['losses']}/{contrast['same']}")
        ci = contrast["paired_task_macro_bootstrap_ci95"]
        if ci:
            lines.append(f"95% within-task paired bootstrap CI: [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]pp")
        lines.append(f"McNemar exact p={contrast['mcnemar_exact_p']:.6g}")
    else:
        lines.append("No matched completed episodes yet.")
    if not contrast["complete"]:
        lines.append("INCOMPLETE: missing episodes are not failures; resume the same command.")
    report = "\n".join(lines) + "\n"
    for name, payload in {"comparison_summary.json": json.dumps(result, indent=2, allow_nan=False) + "\n",
                          "comparison_summary.txt": report}.items():
        fd, temporary = tempfile.mkstemp(prefix="." + name + "-", dir=output)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
            Path(temporary).replace(output / name)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return result, report


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = engine.resolve_repo_path(args.output_dir)
    if args.report_only:
        if args.preflight_only:
            raise ValueError("Choose report-only or preflight-only")
        with (output / ".driver.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _, report = write_control_report(output)
        print(report)
        return 0
    if args.models != ["baseline", "reader"] or args.memory_checkpoint is not None:
        raise ValueError("This driver always compares baseline, action_control and reader; omit --models/--memory-checkpoint")
    args.reader_checkpoint = args.reader_checkpoint or best_checkpoint(engine.resolve_repo_path(args.reader_run))
    args.control_checkpoint = args.control_checkpoint or best_checkpoint(engine.resolve_repo_path(args.control_run))
    engine.validate_options(args)
    if not all(math.isfinite(x) for x in (args.server_timeout, args.task_timeout)):
        raise ValueError("Timeouts must be finite")
    if args.n_episodes > (100 if args.dataset == "train" else 50):
        raise ValueError("RoboMME supports at most 100 train or 50 val/test episodes per task")
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", GR00T_INFERENCE_SEED=str(args.seed))
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    python = engine.python_path(args.server_python)
    if engine.python_path(Path(sys.executable)) != python:
        return subprocess.call([str(python), str(Path(__file__).resolve()),
                                *(sys.argv[1:] if argv is None else argv)], cwd=ROOT, env=env)
    engine.check_dependencies(python, "import torch, cv2, transformers, safetensors, msgpack, zmq", env, "policy")
    engine.check_dependencies(engine.python_path(args.robomme_python),
                              "import robomme, msgpack, zmq, pandas, imageio", env, "simulator")
    identity = build_identity(args)
    print(f"[controls] reader={args.reader_checkpoint}\n[controls] action_control={args.control_checkpoint}", flush=True)
    print(f"[controls] models=3; tasks={len(args.tasks)}; episodes/model/task={args.n_episodes}; "
          f"total={3*len(args.tasks)*args.n_episodes}; dataset={args.dataset}; seed={args.seed}", flush=True)
    if args.preflight_only:
        print("[preflight] PASS. No model/simulator loaded; no evaluation/output started.")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ("." + output.name + ".v5-launch.lock")).open("a") as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Preserve the old engine's live reports/resume behavior. Add the direct
        # control contrast on completion/interruption while the launch lock is held.
        try:
            return engine.run_evaluation(args, identity, env)
        finally:
            path = output / "comparison_manifest.json"
            if path.is_file() and json.loads(path.read_text()).get("evaluation_id") == identity["evaluation_id"]:
                with (output / ".driver.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    _, report = write_control_report(output)
                print(report, flush=True)
                print(f"[controls] Final report: {output / 'comparison_summary.txt'}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[controls] Interrupted. Completed episodes remain resumable.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"[controls] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
