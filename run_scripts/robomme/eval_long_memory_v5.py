#!/usr/bin/env python3
"""Validate recall-trained bundles, then reuse unchanged v4 RoboMME inference.

Auxiliary subgoal/grounding heads are training-only. No labels, subgoals or
other simulator ground truth are fed to the policy. Same Stage-1 reader/AE
weights are required when comparing Stage 1 to its Stage-2 writer.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from run_scripts.robomme import eval_long_memory_v4 as engine
from gr00t.long_memory.checkpoint_v5 import RECIPE, v5_checkpoint_info
from gr00t.long_memory.checkpoint_v4 import reader_state_sha256
from gr00t.long_memory.safety_v5 import validate_output_scope
from safetensors.torch import load_file


def build_parser():
    parser = engine.build_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=Path("runs/eval/robomme/recall_v5_smoke"))
    return parser


def validate_bundles(args):
    base = engine.resolve_repo_path(args.base_model)
    selected = {}
    for stage, names, path in ((1, ("reader", "expert-only"), args.reader_checkpoint),
                               (2, ("memory", "fifo"), args.memory_checkpoint)):
        if any(name in args.models for name in names):
            root = engine.resolve_repo_path(path)
            selected[stage] = (root, v5_checkpoint_info(base, root, expected_stage=stage))
    if 1 in selected and 2 in selected:
        reader, s1 = selected[1]
        _, s2 = selected[2]
        reader_hash = reader_state_sha256(load_file(str(reader / "model.safetensors"), device="cpu"))
        from gr00t.long_memory.checkpoint_v4 import _state_sha256
        expert_hash = _state_sha256(load_file(str(reader / "expert.safetensors"), device="cpu"))
        if reader_hash != s2["metadata"]["frozen_reader_sha256"] or expert_hash != s2["metadata"]["frozen_expert_sha256"]:
            raise ValueError("Reader is not the Stage-2 model's fixed reader/Expert; use its actual Stage-1 parent")
        if (s1["metadata"]["recall_labels_fingerprint"] != s2["metadata"]["recall_labels_fingerprint"]
                or s1["metadata"]["recall_sha256"] != s2["metadata"]["recall_sha256"]):
            raise ValueError("Stage-1/Stage-2 recall target or frozen head differs")
    return selected


def build_identity(args):
    checked = validate_bundles(args)
    inputs = [engine.resolve_repo_path(args.base_model)]
    for path, info in checked.values():
        inputs += [path, info['metadata'].get('cache_dir'), info['config']['train'].get('recall_labels')]
        cache_dir = info['metadata'].get('cache_dir')
        if cache_dir and (Path(cache_dir) / 'manifest.json').is_file():
            inputs.append(json.loads((Path(cache_dir) / 'manifest.json').read_text()).get('dataset_path'))
    validate_output_scope(engine.resolve_repo_path(args.output_dir), *inputs)
    identity = engine.build_identity(args)
    identity["training_recipe"] = RECIPE
    identity["recall_is_policy_input"] = False
    identity["source_sha256"][str(Path(__file__).resolve().relative_to(REPO_ROOT))] = engine.file_hash(Path(__file__))
    for name, entry in identity["models"].items():
        if name == "baseline":
            continue
        _, info = checked[entry["stage"]]
        entry["training_recipe"] = RECIPE
        entry["training_only_recall_sha256"] = info["metadata"]["recall_sha256"]
        entry["recall_labels_fingerprint"] = info["metadata"]["recall_labels_fingerprint"]
        entry["description"] += "; v5 recall auxiliary heads used only during training"
    identity.pop("evaluation_id", None)
    identity["evaluation_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.report_only:
        if args.preflight_only:
            raise ValueError("Choose report-only or preflight-only, not both")
        output = engine.resolve_repo_path(args.output_dir)
        manifest = json.loads((output / "comparison_manifest.json").read_text())
        if manifest.get("training_recipe") != RECIPE:
            raise ValueError("Not a v5 evaluation directory")
        with (output / ".driver.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Evaluation is running; read its comparison_summary.txt") from exc
            _, report = engine.write_report(output)
        print(report)
        return 0
    engine.validate_options(args)
    if not all(math.isfinite(value) for value in (args.server_timeout, args.task_timeout)):
        raise ValueError('Timeouts must be finite')
    if args.n_episodes > (100 if args.dataset == 'train' else 50):
        raise ValueError('RoboMME supports at most 100 train or 50 val/test episodes per task')
    env = dict(os.environ)
    env.update(PYTHONPATH=str(REPO_ROOT), PYTHONUNBUFFERED="1", GR00T_INFERENCE_SEED=str(args.seed))
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    server_python = engine.python_path(args.server_python)
    if engine.python_path(Path(sys.executable)) != server_python:
        return subprocess.call([str(server_python), str(Path(__file__).resolve()),
                                *(sys.argv[1:] if argv is None else argv)], cwd=REPO_ROOT, env=env)
    engine.check_dependencies(server_python, "import torch, cv2, transformers, safetensors, msgpack, zmq", env, "policy")
    engine.check_dependencies(engine.python_path(args.robomme_python), "import robomme, msgpack, zmq, pandas, imageio", env, "simulator")
    identity = build_identity(args)
    print(f"[preflight] compatible v5 models={args.models}; tasks={args.tasks}; n={args.n_episodes}", flush=True)
    if args.preflight_only:
        print("[preflight] No model/simulator loaded; no training/evaluation/output started.")
        return 0
    # Parent-directory lock precedes v4 bind_manifest, whose first write happens
    # before its per-output driver lock. The lock is outside the still-empty
    # output, so v4's nonempty-directory guard is preserved.
    output = engine.resolve_repo_path(args.output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ('.' + output.name + '.v5-launch.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another v5 evaluation owns this output directory') from exc
        return engine.run_evaluation(args, identity, env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[v5-eval] Interrupted; completed episodes remain resumable.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[v5-eval] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
