#!/usr/bin/env python3
"""Evaluate original HAMLET and one V10 archive/projector checkpoint with READ on/off.

This is a separate immutable evaluation namespace, not an extension or rewrite
of earlier V7/V9 runs. Both archive roles load exactly the same trained actor,
attention LoRA and full-rank projector delta. READ-off still appends; it only
bypasses retrieval/fusion. Closed-loop trajectories can consequently diverge.
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
from gr00t.eval.sim.robomme.compare_long_memory_v7_results import completed_write_diagnostics
from gr00t.long_memory.safety_v5 import validate_output_scope
from run_scripts.robomme.baseline_reference_v10 import build_reference, validate_reference
from run_scripts.robomme.report_baseline_reference_v10 import build_reference_report

VARIANT = "archive_projector_v10"
SERVER = Path("run_scripts/robomme/serve_archive_projector_v10.py")
EVALUATOR = Path("run_scripts/robomme/eval_archive_projector_v10.py")
ROLES = ("baseline", "archive", "archive-off")
DEPENDENCIES = tuple(Path("run_scripts/robomme") / name for name in (
    "policy_archive_projector_v10.py", "checkpoint_projector_v10.py", "projector_adapter_v10.py",
    "deployment_objective_v9.py", "audit_archive_generation_v7.py", "eval_long_memory_comparison.py",
    "baseline_reference_v10.py", "report_baseline_reference_v10.py"))


def _projector_spec(spec):
    expected = {"target", "in_features", "out_features", "bias", "kind", "enabled"}
    if (not isinstance(spec, dict) or set(spec) != expected or spec.get("target") != "model.proj_out_2"
            or spec.get("kind") != "full_rank_residual_fp32" or spec.get("bias") is not True
            or type(spec.get("enabled")) is not bool
            or any(type(spec.get(k)) is not int or spec[k] <= 0 for k in ("in_features", "out_features"))):
        raise ValueError("Invalid V10 full-rank projector specification")


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=Path("checkpoints/author_hamlet_robomme/checkpoint-60000"))
    parser.add_argument("--archive-checkpoint", type=Path, help="One trained V10 Stage-1 mode=archive checkpoint for BOTH archive roles")
    parser.add_argument("--baseline-reference", type=Path,
                        help="Explicit completed original-baseline run; preserve its IDs/results and skip ONLY baseline rollout")
    parser.add_argument("--models", nargs="+", choices=ROLES, default=list(ROLES))
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--dataset", choices=("train", "val", "test"), default="val")
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=1300)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/eval/robomme/archive_projector_v10_val_n10_seed6"))
    parser.add_argument("--server-python", type=Path, default=REPO_ROOT / ".venv/bin/python")
    parser.add_argument("--robomme-python", type=Path, default=Path("/home/sjkim/robomme_benchmark/.venv/bin/python"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--server-timeout", type=float, default=300)
    parser.add_argument("--task-timeout", type=float, default=0)
    parser.add_argument("--preflight-only", action="store_true", help="Read-only dependency/provenance checks; no model or simulator loaded")
    parser.add_argument("--report-only", action="store_true", help="Regenerate the existing report without loading models")
    return parser


def validate_options(args):
    if args.tasks == ["all"]:
        args.tasks = list(TASKS)
    if not args.tasks or any(task not in TASKS for task in args.tasks) or len(set(args.tasks)) != len(args.tasks):
        raise ValueError("--tasks must be unique RoboMME tasks or exactly all")
    if "baseline" not in args.models or len(set(args.models)) != len(args.models):
        raise ValueError("--models must be unique and include baseline")
    if any(role != "baseline" for role in args.models) and args.archive_checkpoint is None:
        raise ValueError("archive/archive-off requires --archive-checkpoint")
    if args.n_episodes < 1 or args.seed < 0 or args.n_action_steps < 1 or args.max_episode_steps < 1:
        raise ValueError("Episode/action/step counts must be positive and seed nonnegative")
    if args.n_episodes > (100 if args.dataset == "train" else 50):
        raise ValueError("RoboMME supports at most 100 train or 50 val/test episodes per task")
    if not all(math.isfinite(value) for value in (args.server_timeout, args.task_timeout)):
        raise ValueError("Timeouts must be finite")
    if args.server_timeout <= 0 or args.task_timeout < 0:
        raise ValueError("Server timeout must be positive and task timeout nonnegative")
    if args.preflight_only and args.report_only:
        raise ValueError("Choose preflight-only or report-only, not both")


def _identity_digest(identity):
    return hashlib.sha256(json.dumps({key: value for key, value in identity.items()
                                     if key != "evaluation_id"}, sort_keys=True).encode()).hexdigest()


def validate_manifest_contract(identity):
    """Fail closed on mislabeled READ-off or mismatched actors, including reports."""
    if identity.get("trainer_variant") != VARIANT:
        raise ValueError("Not an archive_projector_v10 evaluation; use its matching report tool")
    if identity.get("evaluation_id") != _identity_digest(identity):
        raise ValueError("Evaluation manifest identity digest does not match its contents")
    models = identity["models"]
    if "baseline" not in models or not set(models) <= set(ROLES):
        raise ValueError("Unknown archive control roles or missing baseline")
    if "baseline_reference" in identity:
        ref = identity["baseline_reference"]
        if (not isinstance(ref, dict) or ref.get("kind") != "completed_original_baseline_reference"
                or ref.get("role") != "baseline" or ref.get("newly_rolled_out") != 0):
            raise ValueError("Invalid explicit original-baseline reference provenance")
    for name, model in models.items():
        if (type(model.get("archive_read_off")) is not bool or type(model.get("memory_off")) is not bool
                or model["archive_read_off"] != (name == "archive-off") or model["memory_off"] != (name == "archive-off")):
            raise ValueError(f"{name}: READ-off flag does not match role")
        if name == "baseline":
            if (model.get("memory_checkpoint") is not None or model.get("mode") != "none"
                    or model.get("stage", 0) != 0 or model.get("write_policy") != "none"
                    or model.get("write_policy_override") != "checkpoint"
                    or any(key in model for key in ("expert_weights_sha256", "semantic_state_sha256", "weights_sha256",
                               "projector_weights_sha256", "projector_config", "checkpoint_files_sha256", "training_metadata",
                               "checkpoint_sha256", "cvom_weights_sha256", "training_state_sha256"))):
                raise ValueError("baseline must be original HAMLET without LoRA or projector adapters")
        elif (model.get("mode") != "archive" or type(model.get("stage")) is not int or model["stage"] != 1
              or type(model.get("step")) is not int or model["step"] <= 0 or not model.get("memory_checkpoint")
              or model.get("write_policy") != "append" or model.get("write_policy_override") != "checkpoint"):
            raise ValueError(f"{name}: requires trained Stage-1 archive with unchanged append WRITE")
        if name != "baseline":
            if model.get("trainer_variant") != VARIANT:
                raise ValueError("Archive projector evaluation requires actual V10 checkpoints, not V7 manifests")
            _projector_spec(model.get("projector_config"))
            aliases = {"checkpoint_sha256": "checkpoint.json", "weights_sha256": "model.safetensors",
                       "cvom_weights_sha256": "cvom.safetensors", "expert_weights_sha256": "expert.safetensors",
                       "projector_weights_sha256": "projector.safetensors", "training_state_sha256": "training_state.pt"}
            files = model.get("checkpoint_files_sha256", {})
            if set(files) != set(aliases.values()) or any(not _hash(files[k]) for k in files):
                raise ValueError("V10 requires exact checkpoint/projector/training payload hashes")
            if any(model.get(key) != files[filename] for key, filename in aliases.items()):
                raise ValueError("V10 checkpoint hash aliases differ")
            semantic = model.get("semantic_state_sha256", {})
            if set(semantic) != {"actor", "expert", "cvom", "projector"} or any(not _hash(v) for v in semantic.values()):
                raise ValueError("V10 semantic state hashes must include projector/actor/expert/CVOM")
        if model.get("base_model") != models["baseline"].get("base_model"):
            raise ValueError("All roles must use the SAME original HAMLET base")
    if {"archive", "archive-off"} <= set(models):
        # Everything except the named READ bypass and description must match,
        # including checkpoint paths, file hashes, semantic hashes and AE config.
        ignored = {"description", "archive_read_off", "memory_off"}
        left, right = ({key: value for key, value in models[name].items() if key not in ignored}
                       for name in ("archive", "archive-off"))
        if left != right:
            raise ValueError("archive/archive-off must use the SAME checkpoint, adapted AE/projector and WRITE policy")


def build_identity(args):
    from safetensors.torch import load_file
    from gr00t.long_memory.checkpoint_v7 import actor_state_sha256
    from run_scripts.robomme.checkpoint_projector_v10 import checkpoint_info
    from gr00t.long_memory.checkpoint_v4 import _state_sha256
    from gr00t.long_memory.hamlet import checkpoint_identity

    base = resolve_repo_path(args.base_model)
    base_identity = checkpoint_identity(base)
    config = json.loads((base / "config.json").read_text())
    if (config.get("hamlet_mode") != "finetune" or config.get("mem_cond_type", "cross_attn") != "cross_attn"
            or config.get("memory_type", "moment_token") != "moment_token" or int(config.get("n_moment_tokens", 0)) <= 0):
        raise ValueError("Evaluation requires trained HAMLET moment-token/cross-attention weights")
    if args.n_action_steps != int(config.get("memory_stride", 16)):
        raise ValueError("--n-action-steps must equal original HAMLET memory_stride")
    common, inputs = None, [base]
    if any(name != "baseline" for name in args.models):
        checkpoint = resolve_repo_path(args.archive_checkpoint)
        info = checkpoint_info(base, checkpoint, expected_stage=1)
        cfg = info["config"]
        if cfg["mode"] != "archive" or info["step"] <= 0:
            raise ValueError("V10 archive READ control requires a trained step>0 Stage-1 mode=archive checkpoint")
        hashes = {filename: file_hash(checkpoint / filename) for filename in
                  ("checkpoint.json", "model.safetensors", "cvom.safetensors", "expert.safetensors",
                   "projector.safetensors", "training_state.pt")}
        common = {
            "base_model": base_identity, "memory_checkpoint": str(checkpoint), "stage": 1, "step": info["step"],
            "mode": "archive", "write_policy": "append", "write_policy_override": "checkpoint",
            "checkpoint_sha256": hashes["checkpoint.json"], "weights_sha256": hashes["model.safetensors"],
            "cvom_weights_sha256": hashes["cvom.safetensors"], "expert_weights_sha256": hashes["expert.safetensors"],
            "projector_weights_sha256": hashes["projector.safetensors"],
            "training_state_sha256": hashes["training_state.pt"], "checkpoint_files_sha256": hashes,
            "semantic_state_sha256": {
                "actor": actor_state_sha256(load_file(str(checkpoint / "model.safetensors"), device="cpu")),
                "expert": _state_sha256(load_file(str(checkpoint / "expert.safetensors"), device="cpu")),
                "cvom": _state_sha256(load_file(str(checkpoint / "cvom.safetensors"), device="cpu")),
                "projector": _state_sha256(load_file(str(checkpoint / "projector.safetensors"), device="cpu")),
            },
            "memory_config": cfg["memory"], "expert_config": cfg["expert"], "expert_targets": cfg["expert_targets"],
            "projector_config": cfg["projector"],
            "training_metadata": info["metadata"], "trainer_variant": VARIANT,
        }
        inputs.append(checkpoint)
        cache_name = info["metadata"].get("cache_dir") or cfg.get("train", {}).get("cache_dir")
        if cache_name:
            cache = resolve_repo_path(Path(cache_name))
            inputs.append(cache)
            if (cache / "manifest.json").is_file():
                inputs.append(json.loads((cache / "manifest.json").read_text()).get("dataset_path"))
    if args.baseline_reference is not None:
        inputs.append(resolve_repo_path(args.baseline_reference))
    validate_output_scope(resolve_repo_path(args.output_dir), *inputs)
    models = {}
    for name in args.models:
        entry = copy.deepcopy(common) if name != "baseline" else {
            "base_model": base_identity, "memory_checkpoint": None, "mode": "none",
            "write_policy": "none", "write_policy_override": "checkpoint",
        }
        entry.update(memory_off=name == "archive-off", archive_read_off=name == "archive-off", description={
            "baseline": "Original HAMLET; no added memory or adapted AE",
            "archive": "V10 archive with attention LoRA + projector delta; append WRITE and learned READ/fusion",
            "archive-off": "SAME V10 actor/LoRA/projector; append WRITE continues, ONLY READ/fusion bypassed",
        }[name])
        models[name] = entry
    sources = set(path.relative_to(REPO_ROOT) for path in (REPO_ROOT / "gr00t").rglob("*.py"))
    sources.update((SERVER, EVALUATOR, *DEPENDENCIES))
    # Missing new entrypoints are errors, not silently omitted hashes.
    code = {str(path): file_hash(REPO_ROOT / path) for path in sorted(sources)}
    print("[preflight] hashing original HAMLET and one trained archive/AE checkpoint ...", flush=True)
    identity = {
        "format_version": 1, "trainer_variant": VARIANT, "models": models,
        "settings": {key: getattr(args, key) for key in ("tasks", "n_episodes", "dataset", "seed",
                     "n_action_steps", "max_episode_steps", "save_videos", "device")},
        "source_sha256": code,
        "base_file_sha256": {str(path.relative_to(base)): file_hash(path) for path in sorted(base.rglob("*"))
                             if path.is_file() and path.suffix in (".json", ".safetensors", ".model", ".txt")},
        "policy_package_versions": {name: importlib.metadata.version(name)
                                    for name in ("torch", "transformers", "safetensors", "numpy")},
        "benchmark": benchmark_identity(args), "server_python": str(python_path(args.server_python)),
        "robomme_python": str(python_path(args.robomme_python)),
        "memory_input": "observed post-HAMLET short tokens + current state/frame/demo only",
        "control_contract": "same V10 archive actor/LoRA/projector; READ bypass ONLY; unchanged append WRITE, demo cadence and action noise schedule",
        "learned_storage_selection": False,
    }
    if args.baseline_reference is not None:
        identity["baseline_reference"] = build_reference(resolve_repo_path(args.baseline_reference), identity)
    identity["evaluation_id"] = _identity_digest(identity)
    validate_manifest_contract(identity)
    validate_output_reuse(resolve_repo_path(args.output_dir), identity)
    return identity


def validate_output_reuse(output, identity):
    """Read-only preflight counterpart to bind_manifest; never adopt old results."""
    output = Path(output)
    if "baseline_reference" in identity and (output / "baseline").exists():
        raise ValueError("Referenced baseline must not have copied/local baseline artifacts; use a NEW --output-dir")
    if not output.exists():
        return
    manifest = output / "comparison_manifest.json"
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        validate_manifest_contract(saved)
        if saved != identity:
            raise ValueError("Evaluation provenance changed; use a NEW --output-dir")
    elif not output.is_dir() or any(output.iterdir()):
        raise ValueError("Unidentified existing output contents; use a NEW --output-dir")


def server_command(args, model, port):
    command = [str(python_path(args.server_python)), "-u", str(REPO_ROOT / SERVER),
               "--base-model", model["base_model"]["path"], "--device", args.device,
               "--host", "127.0.0.1", "--port", str(port)]
    if model["memory_checkpoint"]:
        command += ["--memory-checkpoint", model["memory_checkpoint"], "--write-policy", "checkpoint"]
    if model["archive_read_off"]:
        command.append("--archive-read-off")
    return command


def completed_read_diagnostics(root, name, tasks, expected, *, expected_model=None):
    """Check ALL calls of completed sessions, ignoring abandoned retry sessions.

    Prime-only demo calls legitimately lack READ metrics. Missing diagnostics
    are counted as missing evidence, never interpreted as a successful bypass.
    """
    totals = {key: 0 for key in ("completed_sessions_with_records", "completed_sessions_missing_records",
              "passive_calls", "decision_calls", "calls_missing_read_flag", "decision_calls_missing_metrics",
              "decision_calls_with_metrics", "read_enabled_calls", "ignored_torn_final_lines",
              "calls_with_projector_identity", "calls_missing_projector_identity")}
    maxima = {key: None for key in ("ae_conditioning_delta_norm", "ae_conditioning_changed_fraction")}
    for task in tasks:
        rows = read_results(root / name / task / "simulation_results.csv", expected=expected)
        path = root / name / task / "memory_diagnostics.jsonl"
        sessions, completed = {}, {}
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        if not line.endswith("\n"):
                            totals["ignored_torn_final_lines"] += 1
                            continue
                        raise ValueError(f"Malformed complete diagnostics record: {path}")
                    sid = record.get("session_id")
                    if record.get("kind") == "policy_call":
                        sessions.setdefault(sid, []).append(record)
                    elif record.get("kind") == "episode_complete":
                        episode = record.get("episode_idx")
                        row = rows.get(episode)
                        if row and record.get("episode_seed") == row["episode_seed"] and record.get("success") == row["success"]:
                            completed[episode] = sid
        for episode, row in rows.items():
            calls = sessions.get(completed.get(episode), [])
            if not calls:
                totals["completed_sessions_missing_records"] += 1
                continue
            totals["completed_sessions_with_records"] += 1
            for record in calls:
                if record.get("episode_idx") != episode or record.get("episode_seed") != row["episode_seed"]:
                    raise ValueError(f"{name}/{task}/{episode}: completed-session call identity differs")
                call_info = record.get("info", {})
                if expected_model is not None:
                    fields = {"checkpoint_variant": VARIANT, "checkpoint_step": expected_model["step"],
                              "projector_enabled": expected_model["projector_config"]["enabled"],
                              "projector_kind": expected_model["projector_config"]["kind"]}
                    present = all(key in call_info for key in fields)
                    totals["calls_with_projector_identity" if present else "calls_missing_projector_identity"] += 1
                    for key, value in fields.items():
                        if key in call_info and (type(call_info[key]) is not type(value) or call_info[key] != value):
                            raise ValueError(f"{name}/{task}/{episode}: projector identity differs: {key}")
                diagnostics = call_info.get("long_memory", {})
                passive = record.get("passive")
                if type(passive) is not bool:
                    raise ValueError(f"{name}/{task}/{episode}: missing/invalid passive-call flag")
                totals["passive_calls" if passive else "decision_calls"] += 1
                if "memory_read_enabled" not in diagnostics:
                    totals["calls_missing_read_flag"] += 1
                else:
                    flag = diagnostics["memory_read_enabled"]
                    if type(flag) is not bool or (flag and (name == "archive-off" or passive)):
                        raise ValueError(f"{name}/{task}/{episode}: contradictory memory_read_enabled flag")
                    totals["read_enabled_calls"] += int(flag)
                for key, value in (("mode", "archive"), ("policy", "append")):
                    if key in diagnostics and diagnostics[key] != value:
                        raise ValueError(f"{name}/{task}/{episode}: unexpected {key} in archive diagnostics")
                read = diagnostics.get("read", {})
                if not passive:
                    totals["decision_calls_with_metrics" if all(key in read for key in maxima)
                           else "decision_calls_missing_metrics"] += 1
                for key in maxima:
                    if key not in read:
                        continue
                    value = read[key]
                    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                            or value < 0 or (key.endswith("fraction") and value > 1)):
                        raise ValueError(f"{name}/{task}/{episode}: invalid {key}")
                    if name == "archive-off" and value != 0:
                        raise ValueError(f"archive-off/{task}/{episode}: READ bypass has nonzero {key}")
                    maxima[key] = max(maxima[key], value) if maxima[key] is not None else value
    return {**totals, "max_observed_read_metrics": maxima}


def build_control_report(run_dir, *, bootstrap_samples=5000):
    root = Path(run_dir)
    manifest = json.loads((root / "comparison_manifest.json").read_text())
    validate_manifest_contract(manifest)
    if "baseline_reference" in manifest:
        validate_output_reuse(root, manifest)
        reference_root, reference_manifest = validate_reference(manifest["baseline_reference"], manifest)
        result, text = build_reference_report(root, manifest, reference_root, reference_manifest,
                                             bootstrap_samples=bootstrap_samples)
    else:
        result, text = build_report(root, bootstrap_samples=bootstrap_samples)
    result.update(trainer_variant=VARIANT, additional_comparisons={}, storage_diagnostics={}, read_diagnostics={})
    settings, models = manifest["settings"], manifest["models"]
    lines = [text.rstrip(), "", "V10 archive READ isolation: same checkpoint, attention LoRA AND projector delta"]
    if {"archive", "archive-off"} <= set(models):
        # The generic report compares metadata with baseline if baseline rows
        # exist. Validate on/off directly as well, even if baseline is incomplete.
        for task in settings["tasks"]:
            identities = []
            for name in ("archive-off", "archive"):
                if read_results(root / name / task / "simulation_results.csv", expected=settings["n_episodes"]):
                    identities.append(validate_result_identity(root, name, task, manifest))
            if len(identities) == 2:
                for key in ("scenario_metadata_sha256", "model_config_sha256", "memory_window", "demo_sampling"):
                    if identities[0].get(key) != identities[1].get(key):
                        raise ValueError(f"archive-off/archive {task}: {key} differs; refusing an unmatched contrast")
        contrast = _contrast(root, "archive-off", "archive", settings["tasks"], settings["n_episodes"], bootstrap_samples)
        result["additional_comparisons"]["archive-off_to_archive"] = contrast
        lines.append("archive-off -> archive: READ/fusion contribution with the SAME trained AE/LoRA/projector and append WRITE")
        lines.append(f"{'Task':<22} {'Paired N':>8} {'Delta':>10} {'Wins/Losses/Same':>19}")
        for task, entry in contrast["tasks"].items():
            delta_text = "--" if entry["paired_delta"] is None else f"{100*entry['paired_delta']:+.2f}pp"
            lines.append(f"{task:<22} {entry['paired_n']:>8} {delta_text:>10} "
                         f"{entry['wins']:>5}/{entry['losses']}/{entry['same']}")
        delta = contrast["paired_task_macro_delta"]
        if delta is None:
            lines.append("  INCOMPLETE: no matched completed episodes.")
        else:
            lines.append(f"  Delta={100*delta:+.2f}pp; N={contrast['paired_n']}; "
                         f"wins/losses/same={contrast['wins']}/{contrast['losses']}/{contrast['same']}; "
                         f"McNemar p={contrast['mcnemar_exact_p']:.6g}")
            ci = contrast["paired_task_macro_bootstrap_ci95"]
            if ci:
                lines.append(f"  95% within-task paired bootstrap CI: [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]pp")
            if not contrast["complete"]:
                lines.append("  INCOMPLETE: only matched completed episodes are included.")
    else:
        lines.append("Select BOTH archive and archive-off for the same-AE READ contrast.")
    for name in models:
        if name == "baseline":
            continue
        counters = completed_write_diagnostics(root, name, settings["tasks"], settings["n_episodes"])
        if (counters["keeps"] or counters["demo_keeps"] or counters["observations_seen"] != counters["updates"]
                or counters["write_attempts"] != counters["updates"] or counters["demo_updates"] > counters["updates"]):
            raise ValueError(f"{name}: completed WRITE diagnostics contradict unchanged archive APPEND")
        result["storage_diagnostics"][name] = counters
        lines.append(f"  {name}: completed sessions with WRITE diagnostics={counters['completed_sessions_with_diagnostics']}; "
                     f"missing={counters['completed_sessions_missing_diagnostics']}; append updates={counters['updates']}")
        reads = completed_read_diagnostics(root, name, settings["tasks"], settings["n_episodes"], expected_model=models[name])
        result["read_diagnostics"][name] = reads
        lines.append(f"  {name} READ evidence: completed sessions with calls={reads['completed_sessions_with_records']}; "
                     f"missing={reads['completed_sessions_missing_records']}; decisions with metrics={reads['decision_calls_with_metrics']}; "
                     f"decisions missing metrics={reads['decision_calls_missing_metrics']}; "
                     f"calls missing READ flag={reads['calls_missing_read_flag']}")
        lines.append(f"    max observed READ metrics: {reads['max_observed_read_metrics']}")
        lines.append(f"    calls with exact V10 checkpoint/projector identity={reads['calls_with_projector_identity']}; "
                     f"missing={reads['calls_missing_projector_identity']}")
    lines += ["WRITE rules are unchanged, but differing actions can produce different trajectories and write counts.",
              "Archive admission is rule-based append; this experiment tests learned representation/READ, not a learned writer.",
              "Archive-off is an inference ablation, not a separately trained memory-free AE control.",
              "READ-off retains the checkpoint's exact attention LoRA AND projector delta; it is NOT projector-off.",
              "Projector enabled/disabled follows the saved training arm, never an inference override.",
              "Missing READ diagnostics are missing evidence, not proof of zero contribution; passive demo calls omit READ metrics.",
              "An all-history archive is not a bounded-capacity or compute-matched memory control.",
              "Use validation for checkpoint selection; a subset/validation score does not establish test success >=30%."]
    return result, "\n".join(lines) + "\n"


def write_report(run_dir, *, bootstrap_samples=5000):
    result, text = build_control_report(run_dir, bootstrap_samples=bootstrap_samples)
    root = Path(run_dir)
    for name, payload in {"comparison_summary.json": json.dumps(result, indent=2, allow_nan=False) + "\n",
                          "comparison_summary.txt": text}.items():
        fd, filename = tempfile.mkstemp(prefix=f".{name}-", dir=root)
        temporary = Path(filename)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(root / name)
        finally:
            temporary.unlink(missing_ok=True)
    return result, text


def run_evaluation(args, identity, env):
    """Use existing protocol/client helpers, owning only this driver's children."""
    validate_manifest_contract(identity)
    output = resolve_repo_path(args.output_dir)
    validate_output_reuse(output, identity)
    if "baseline_reference" in identity:
        validate_reference(identity["baseline_reference"], identity)
    bind_manifest(output, identity)
    lock = (output / ".driver.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f"Another evaluation driver owns {output}")
    failures, interrupted = [], False
    try:
        for name, model in identity["models"].items():
            if name == "baseline" and "baseline_reference" in identity:
                reference = identity["baseline_reference"]
                print(f"[projector-v10-eval] baseline REUSED: {reference['completed_episodes']} completed; "
                      f"newly rolled out 0; source={reference['source_run']}; original ID={reference['source_evaluation_id']}", flush=True)
                continue
            pending = []
            for task in args.tasks:
                rows = read_results(output / name / task / "simulation_results.csv", expected=args.n_episodes)
                if rows:
                    validate_result_identity(output, name, task, identity)
                if len(rows) != args.n_episodes:
                    pending.append(task)
                else:
                    print(f"[projector-v10-eval] resume: {name}/{task} complete", flush=True)
            if not pending:
                continue
            folder = output / name
            folder.mkdir(exist_ok=True)
            port, server = free_local_port(), None
            command = server_command(args, model, port)
            with (folder / "server.log").open("a", encoding="utf-8") as log:
                try:
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
                        print(f"[projector-v10-eval] {name}/{task}; {task_dir / 'rollout.log'}", flush=True)
                        try:
                            run_client(client, env=env, log_path=task_dir / "rollout.log", server=server, timeout=args.task_timeout)
                            if len(read_results(task_dir / "simulation_results.csv", expected=args.n_episodes)) != args.n_episodes:
                                raise RuntimeError("Rollout exited without all requested completed episodes")
                            validate_result_identity(output, name, task, identity)
                        except (RuntimeError, TimeoutError, ValueError) as exc:
                            failures.append({"model": name, "task": task, "error": str(exc)})
                            print(f"[projector-v10-eval] ERROR {name}/{task}: {exc}", file=sys.stderr, flush=True)
                        write_report(output, bootstrap_samples=500)
                except (RuntimeError, TimeoutError) as exc:
                    failures.append({"model": name, "error": str(exc)})
                    print(f"[projector-v10-eval] ERROR {name}: {exc}; see server.log", file=sys.stderr, flush=True)
                finally:
                    stop_process(server)
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        try:
            (output / "driver_status.json").write_text(json.dumps({"interrupted": interrupted, "failures": failures}, indent=2) + "\n")
            result, report = write_report(output)
            print(report, flush=True)
            print(f"[projector-v10-eval] Summary: {output / 'comparison_summary.txt'}", flush=True)
        finally:
            lock.close()
    return int(bool(failures) or not all(model["complete"] for model in result["models"].values()))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.report_only:
        if args.preflight_only:
            raise ValueError("Choose preflight-only or report-only, not both")
        output = resolve_repo_path(args.output_dir)
        validate_manifest_contract(json.loads((output / "comparison_manifest.json").read_text()))
        with (output / ".driver.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Evaluation is running; read comparison_summary.txt instead")
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
    print(f"[preflight] compatible archive READ-control models={list(identity['models'])}; tasks={args.tasks}; n={args.n_episodes}", flush=True)
    if args.preflight_only:
        print("[preflight] No model/simulator loaded; no training/evaluation/output started.")
        return 0
    output = resolve_repo_path(args.output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ("." + output.name + ".archive-projector-v10-launch.lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another archive READ-control evaluation owns this output directory") from exc
        return run_evaluation(args, identity, env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[projector-v10-eval] Interrupted; owned processes stopped, completed episodes resumable.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[projector-v10-eval] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
