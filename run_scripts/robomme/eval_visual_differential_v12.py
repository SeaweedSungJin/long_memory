#!/usr/bin/env python3
"""Paired V12 visual READ evaluation with a frozen archive/Action Expert parent.

Differential/current-only are independently trained, matched-budget arms.
Differential/visual-off share one checkpoint. Original HAMLET is a separate baseline.
External baseline results keep their original
IDs and are explicitly REUSED; no previous adapted model results are relabeled.
The orchestration follows the existing immutable V10 evaluation protocol, while
V12 model identities and actual per-call visual diagnostics are checked here.
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

VARIANT = "visual_differential_v12"
SERVER = Path("run_scripts/robomme/serve_visual_differential_v12.py")
# Deliberately run the ALREADY REVIEWED no-adapter server for the baseline.
# This makes baseline_reference_v10's unchanged compatibility profile truthful.
BASELINE_SERVER = Path("run_scripts/robomme/serve_archive_projector_v10.py")
EVALUATOR = Path("run_scripts/robomme/eval_visual_differential_v12.py")
ROLES = ("baseline", "differential", "current-only", "visual-off")
DEPENDENCIES = tuple(Path("run_scripts/robomme") / name for name in (
    "policy_visual_differential_v12.py", "checkpoint_visual_differential_v12.py", "visual_differential_memory_v12.py",
    "policy_visual_patch_v11.py", "checkpoint_visual_patch_v11.py", "visual_patch_memory_v11.py",
    "policy_archive_projector_v10.py", "checkpoint_projector_v10.py", "projector_adapter_v10.py",
    "deployment_objective_v9.py", "audit_archive_generation_v7.py",
    "eval_long_memory_comparison.py", "baseline_reference_v10.py", "report_baseline_reference_v10.py",
))
CAMERAS = ["front_view", "wrist_view"]

def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=Path("checkpoints/author_hamlet_robomme/checkpoint-60000"))
    parser.add_argument("--differential-checkpoint", type=Path, help="Actual differential bundle; also used unchanged by visual-off")
    parser.add_argument("--current-only-checkpoint", type=Path, help="Independently trained current_only bundle")
    parser.add_argument("--allow-initialization-checkpoints", action="store_true",
                        help="Explicitly evaluate honest step0 selections (zero learned visual updates), not trained models")
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
    parser.add_argument("--output-dir", type=Path, default=Path("runs/eval/robomme/visual_differential_v12_val_n10_seed6"))
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
    if set(args.models) & {"differential", "visual-off"} and args.differential_checkpoint is None:
        raise ValueError("differential/visual-off requires --differential-checkpoint")
    if "current-only" in args.models and args.current_only_checkpoint is None:
        raise ValueError("current-only requires --current-only-checkpoint")
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


def training_contract(config, metadata):
    ignored = {"read_mode", "output_dir", "init_checkpoint", "resume", "stop_after_steps", "preflight_only", "device"}
    return {"plan_sha256": metadata["plan_sha256"], "cache_fingerprint": metadata["cache_fingerprint"],
            "source_sha256": metadata["source_sha256"],
            "initialization_reference": metadata.get("initialization_reference"),
            "runtime": metadata.get("runtime"), "selection_metric": metadata.get("selection_metric"),
            "objective": config.get("objective"),
            "train_options": {k: v for k, v in config["train"].items() if k not in ignored}}

def validate_manifest_contract(identity):
    """Model labels cannot silently disable the parent or load another visual actor."""
    if identity.get("format_version") != 1 or identity.get("trainer_variant") != VARIANT:
        raise ValueError("Expected visual_differential_v12 evaluation")
    if identity.get("evaluation_id") != _identity_digest(identity):
        raise ValueError("Evaluation identity digest changed")
    allow_initialization = identity.get("allow_initialization_checkpoints")
    if type(allow_initialization) is not bool:
        raise ValueError("Explicit initialization-checkpoint evaluation policy is missing")
    models = identity.get("models", {})
    if "baseline" not in models or not set(models) <= set(ROLES):
        raise ValueError("Unknown roles or missing original baseline")
    if "baseline_reference" in identity:
        ref = identity["baseline_reference"]
        if (ref.get("kind") != "completed_original_baseline_reference"
                or ref.get("role") != "baseline" or ref.get("newly_rolled_out") != 0):
            raise ValueError("Invalid explicit baseline reference")
    for name, model in models.items():
        if model.get("memory_off") is not False or model.get("archive_read_off") is not False:
            raise ValueError("V12 never disables the frozen parent's short archive")
        if model.get("visual_read_off") is not (name == "visual-off"):
            raise ValueError("Visual READ-off flag disagrees with role")
        if model.get("base_model") != models["baseline"].get("base_model"):
            raise ValueError("All roles must use the same original HAMLET")
        if name == "baseline":
            forbidden = {"checkpoint_files_sha256", "visual_config", "weights_sha256", "frozen_parent",
                         "training_metadata", "semantic_state_sha256", "checkpoint_sha256",
                         "training_state_sha256", "read_mode", "training_contract", "training_config", "training_objective", "checkpoint_status",
                         "expert_weights_sha256", "projector_weights_sha256", "cvom_weights_sha256"}
            if (model.get("memory_checkpoint") is not None or model.get("mode") != "none"
                    or model.get("stage", 0) != 0 or model.get("write_policy") != "none"
                    or model.get("write_policy_override") != "checkpoint"
                    or model.get("server_script") != str(BASELINE_SERVER) or forbidden & set(model)):
                raise ValueError("baseline must be original HAMLET without adapters")
            continue
        expected_mode = "current_only" if name == "current-only" else "differential"
        if (model.get("trainer_variant") != VARIANT or model.get("mode") != "visual_differential"
                or model.get("read_mode") != expected_mode
                or type(model.get("stage")) is not int or model["stage"] != 1
                or type(model.get("step")) is not int or model["step"] < 0
                or (model["step"] == 0 and not allow_initialization)
                or not model.get("memory_checkpoint") or model.get("write_policy") != "append"
                or model.get("write_policy_override") != "checkpoint"
                or model.get("server_script") != str(SERVER)):
            raise ValueError("Requires V12 checkpoint with unchanged APPEND; step0 needs explicit --allow-initialization-checkpoints")
        expected_status = "INITIALIZATION_SELECTED" if model["step"] == 0 else "TRAINED"
        if model.get("checkpoint_status") != expected_status:
            raise ValueError("Checkpoint status must honestly distinguish initialization from learned updates")
        cfg = model.get("visual_config", {})
        if (not isinstance(cfg, dict) or set(cfg) != {"feature_dim", "hidden_dim", "num_heads", "num_short_tokens", "time_scale"}
                or cfg["feature_dim"] != 2048 or cfg["num_short_tokens"] != 4
                or any(type(cfg[k]) is not int or cfg[k] <= 0 for k in
                       ("feature_dim", "hidden_dim", "num_heads", "num_short_tokens"))
                or cfg["hidden_dim"] % cfg["num_heads"] or type(cfg["time_scale"]) not in (int, float)
                or not math.isfinite(cfg["time_scale"]) or cfg["time_scale"] <= 0
                or model.get("camera_order") != CAMERAS):
            raise ValueError("Invalid audited visual layout/configuration")
        files = model.get("checkpoint_files_sha256", {})
        aliases = {"checkpoint_sha256": "checkpoint.json", "weights_sha256": "visual.safetensors",
                   "training_state_sha256": "training_state.pt"}
        if (not isinstance(files, dict) or set(files) != set(aliases.values()) or any(not _hash(v) for v in files.values())
                or any(model.get(k) != files[v] for k, v in aliases.items())
                or not _hash(model.get("semantic_state_sha256"))):
            raise ValueError("V12 visual checkpoint hashes missing or inconsistent")
        parent = model.get("frozen_parent", {})
        names = {"checkpoint.json", "model.safetensors", "expert.safetensors",
                 "cvom.safetensors", "training_state.pt"}
        if (not isinstance(parent, dict) or set(parent) != {"path", "step", "files_sha256"} or not parent.get("path")
                or type(parent.get("step")) is not int or parent["step"] != 1250
                or set(parent.get("files_sha256", {})) != names
                or any(not _hash(v) for v in parent["files_sha256"].values())):
            raise ValueError("Missing immutable archive1250 parent")
        if (not isinstance(model.get("training_metadata"), dict)
                or model["training_metadata"].get("frozen_parent") != parent):
            raise ValueError("Model and training parent identities disagree")
        if (not isinstance(model.get("training_config"), dict)
                or model["training_config"].get("read_mode") != expected_mode
                or model["training_metadata"].get("read_mode") != expected_mode):
            raise ValueError("Runtime role and checkpoint training read_mode disagree")
        expected_contract = training_contract({"train": model["training_config"], "objective": model.get("training_objective")}, model["training_metadata"])
        if model.get("training_contract") != expected_contract or not _hash(expected_contract["plan_sha256"]):
            raise ValueError("Training query/hyperparameter provenance differs")
    if {"differential", "visual-off"} <= set(models):
        ignored = {"description", "visual_read_off"}
        a, b = ({k: v for k, v in models[n].items() if k not in ignored} for n in ("differential", "visual-off"))
        if a != b:
            raise ValueError("Differential/visual-off must use the SAME frozen parent and visual checkpoint")
    candidates = [model for name, model in models.items() if name != "baseline"]
    if candidates:
        for model in candidates[1:]:
            for key in ("frozen_parent", "visual_config", "camera_order", "training_contract"):
                if model[key] != candidates[0][key]:
                    raise ValueError(f"V12 trained-arm comparison is unmatched: {key}")


def build_identity(args):
    from safetensors.torch import load_file
    from gr00t.long_memory.checkpoint_v4 import _state_sha256
    from gr00t.long_memory.hamlet import checkpoint_identity
    from run_scripts.robomme.checkpoint_visual_differential_v12 import checkpoint_info
    base = resolve_repo_path(args.base_model)
    base_identity = checkpoint_identity(base)
    original = json.loads((base / "config.json").read_text())
    if (original.get("hamlet_mode") != "finetune"
            or original.get("mem_cond_type", "cross_attn") != "cross_attn"
            or original.get("memory_type", "moment_token") != "moment_token"
            or original.get("n_moment_tokens") != 4):
        raise ValueError("Requires the audited original K4/moment-token HAMLET")
    if args.n_action_steps != int(original.get("memory_stride", 16)):
        raise ValueError("Action execution interval must equal original memory stride")
    bundles, inputs = {}, [base]
    required = {}
    if set(args.models) & {"differential", "visual-off"}:
        required["differential"] = args.differential_checkpoint
    if "current-only" in args.models:
        required["current_only"] = args.current_only_checkpoint
    for read_mode, path in required.items():
        checkpoint = resolve_repo_path(path)
        info = checkpoint_info(base, checkpoint, expected_stage=1)
        cfg, meta = info["config"], info["metadata"]
        if cfg.get("read_mode") != read_mode or cfg.get("train", {}).get("read_mode") != read_mode or meta.get("read_mode") != read_mode:
            raise ValueError("Checkpoint read_mode contradicts the requested trained arm")
        if info["step"] == 0 and not args.allow_initialization_checkpoints:
            raise ValueError("Evaluation defaults to trained step>0; a valid step0 selection needs explicit --allow-initialization-checkpoints")
        files = {name: file_hash(checkpoint / name) for name in
                 ("checkpoint.json", "visual.safetensors", "training_state.pt")}
        bundles[read_mode] = {
            "base_model": base_identity, "memory_checkpoint": str(checkpoint),
            "stage": 1, "step": info["step"], "mode": "visual_differential", "read_mode": read_mode,
            "checkpoint_status": "INITIALIZATION_SELECTED" if info["step"] == 0 else "TRAINED",
            "write_policy": "append", "write_policy_override": "checkpoint",
            "trainer_variant": VARIANT, "server_script": str(SERVER),
            "checkpoint_files_sha256": files, "checkpoint_sha256": files["checkpoint.json"],
            "weights_sha256": files["visual.safetensors"], "training_state_sha256": files["training_state.pt"],
            "semantic_state_sha256": _state_sha256(load_file(str(checkpoint / "visual.safetensors"), device="cpu")),
            "visual_config": cfg["visual"], "camera_order": cfg["camera_order"],
            "frozen_parent": copy.deepcopy(meta["frozen_parent"]), "training_metadata": meta,
            "training_config": copy.deepcopy(cfg["train"]), "training_objective": copy.deepcopy(cfg.get("objective")),
            "training_contract": training_contract(cfg, meta),
        }
        inputs += [checkpoint, meta["frozen_parent"]["path"]]
        # Inference is independent of the initialization run's continued
        # existence, but evaluation outputs must not be placed inside it.
        initialization = meta.get("initialization_reference")
        if isinstance(initialization, dict) and initialization.get("path"):
            inputs.append(initialization["path"])
        cache_path = meta.get("cache_dir") or cfg.get("train", {}).get("cache_dir")
        if cache_path:
            cache = resolve_repo_path(Path(cache_path))
            inputs.append(cache)
            if (cache / "manifest.json").is_file():
                inputs.append(json.loads((cache / "manifest.json").read_text()).get("dataset_path"))
    if args.baseline_reference is not None:
        inputs.append(resolve_repo_path(args.baseline_reference))
    validate_output_scope(resolve_repo_path(args.output_dir), *inputs)
    descriptions = {
        "baseline": "Original HAMLET, no added memory/adapted AE",
        "differential": "Frozen archive1250/AE plus trained content-only past-minus-current visual READ",
        "current-only": "Separately trained current-only visual adapter; SAME frozen parent and training query/hyperparameter budget",
        "visual-off": "SAME differential bundle; both visual READ branches bypassed, both APPEND streams and parent READ continue",
    }
    models = {}
    for name in args.models:
        model = copy.deepcopy(bundles["current_only" if name == "current-only" else "differential"]) if name != "baseline" else {
            "base_model": base_identity, "memory_checkpoint": None, "mode": "none",
            "write_policy": "none", "write_policy_override": "checkpoint", "server_script": str(BASELINE_SERVER),
        }
        model.update(memory_off=False, archive_read_off=False, visual_read_off=name == "visual-off",
                     description=descriptions[name])
        if name != "baseline" and model["step"] == 0:
            model["description"] = "INITIALIZATION_SELECTED: zero learned visual updates; " + model["description"].replace("trained ", "initialized ")
        models[name] = model
    sources = {p.relative_to(REPO_ROOT) for p in (REPO_ROOT / "gr00t").rglob("*.py")}
    sources.update((SERVER, BASELINE_SERVER, EVALUATOR, *DEPENDENCIES))
    print("[preflight] hashing original HAMLET, frozen parent, and V12 visual weights ...", flush=True)
    identity = {
        "format_version": 1, "trainer_variant": VARIANT, "models": models,
        # This is evaluation admission, NOT a rollout setting. Keep the exact
        # original settings schema required by the unchanged baseline profile.
        "allow_initialization_checkpoints": args.allow_initialization_checkpoints,
        "settings": {k: getattr(args, k) for k in ("tasks", "n_episodes", "dataset", "seed",
                     "n_action_steps", "max_episode_steps", "save_videos", "device")},
        "source_sha256": {str(p): file_hash(REPO_ROOT / p) for p in sorted(sources)},
        "base_file_sha256": {str(p.relative_to(base)): file_hash(p) for p in sorted(base.rglob("*"))
                            if p.is_file() and p.suffix in (".json", ".safetensors", ".model", ".txt")},
        "policy_package_versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "safetensors", "numpy")},
        "benchmark": benchmark_identity(args), "server_python": str(python_path(args.server_python)),
        "robomme_python": str(python_path(args.robomme_python)),
        "memory_input": "original observed visual patches with camera/2D/frame/demo and current short-query summary; no actions",
        "control_contract": "same frozen archive1250/AE; independently trained differential/current_only arms; visual-off is SAME differential bundle; APPEND/noise unchanged",
        "learned_storage_selection": False,
    }
    if args.baseline_reference is not None:
        identity["baseline_reference"] = build_reference(resolve_repo_path(args.baseline_reference), identity)
    identity["evaluation_id"] = _identity_digest(identity)
    validate_manifest_contract(identity)
    validate_output_reuse(resolve_repo_path(args.output_dir), identity)
    return identity


def server_command(args, model, port):
    command = [str(python_path(args.server_python)), "-u", str(REPO_ROOT / model["server_script"]),
               "--base-model", model["base_model"]["path"], "--device", args.device,
               "--host", "127.0.0.1", "--port", str(port)]
    if model["memory_checkpoint"]:
        command += ["--memory-checkpoint", model["memory_checkpoint"], "--write-policy", "checkpoint",
                    "--expected-read-mode", model["read_mode"]]
    if model["visual_read_off"]:
        command.append("--visual-read-off")
    return command


def completed_visual_diagnostics(root, role, tasks, expected, model):
    """Validate completed-session visual identity and causal APPEND counters.

    Retry/abandoned sessions are excluded. Missing evidence remains missing,
    never interpreted as a successful READ bypass.
    """
    totals = {k: 0 for k in ("completed_sessions", "missing_sessions", "calls", "missing_calls",
               "missing_identity", "missing_parent_calls", "passive_calls", "decision_calls", "read_enabled_calls",
               "append_updates", "demo_updates", "past_read_enabled_calls",
               "current_reference_enabled_calls", "ignored_torn_final_lines")}
    maxima = {"image_delta_norm": None, "image_changed_fraction": None,
              "readout_norm": None, "projected_residual_norm": None}
    identity = {"checkpoint_variant": VARIANT, "checkpoint_step": model["step"],
                "stage": 1, "expert_adapted": True, "visual_read_mode": model["read_mode"],
                "visual_read_off": role == "visual-off", "visual_weights_sha256": model["weights_sha256"],
                "frozen_parent_checkpoint_sha256": model["frozen_parent"]["files_sha256"]["checkpoint.json"]}
    for task in tasks:
        rows = read_results(root / role / task / "simulation_results.csv", expected=expected)
        path = root / role / task / "memory_diagnostics.jsonl"
        calls, completed = {}, {}
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        if not line.endswith("\n"):
                            totals["ignored_torn_final_lines"] += 1
                            continue
                        raise ValueError(f"Malformed complete visual diagnostic: {path}")
                    sid = record.get("session_id")
                    if record.get("kind") == "policy_call":
                        calls.setdefault(sid, []).append(record)
                    elif record.get("kind") == "episode_complete":
                        episode = record.get("episode_idx")
                        row = rows.get(episode)
                        if row and record.get("episode_seed") == row["episode_seed"] and record.get("success") == row["success"]:
                            completed[episode] = sid
        for episode, row in rows.items():
            records = calls.get(completed.get(episode), [])
            if not records:
                totals["missing_sessions"] += 1
                continue
            totals["completed_sessions"] += 1
            demos, previous_frame = 0, -1
            for index, record in enumerate(records, 1):
                if record.get("episode_idx") != episode or record.get("episode_seed") != row["episode_seed"]:
                    raise ValueError("Completed visual session call identity differs")
                info = record.get("info", {})
                totals["calls"] += 1
                if any(k not in info for k in identity):
                    totals["missing_identity"] += 1
                for key, value in identity.items():
                    if key in info and (type(info[key]) is not type(value) or info[key] != value):
                        raise ValueError(f"{role}/{task}: loaded visual identity differs: {key}")
                passive, frame = record.get("passive"), record.get("frame_index")
                if type(passive) is not bool or type(frame) is not int or frame <= previous_frame:
                    raise ValueError("Invalid visual observation cadence/passive flag")
                previous_frame = frame
                demos += int(passive)
                totals["passive_calls" if passive else "decision_calls"] += 1
                # Visual-off is NOT parent-archive-off. Verify the actual
                # inherited branch on every completed call, not just labels or
                # final APPEND totals, which cannot establish retained READ.
                parent_expected = {"mode": "archive", "policy": "append",
                    "memory_read_enabled": not passive and index > 1,
                    "observations_seen": index, "write_attempts": index, "updates": index,
                    "keeps": 0, "demo_updates": demos, "demo_keeps": 0,
                    "frame_index": frame, "passive": passive}
                parent = info.get("long_memory")
                if not isinstance(parent, dict) or not set(parent_expected) <= set(parent):
                    totals["missing_parent_calls"] += 1
                if isinstance(parent, dict):
                    for key, value in parent_expected.items():
                        if key in parent and (type(parent[key]) is not type(value) or parent[key] != value):
                            raise ValueError(f"{role}/{task}: frozen parent READ/APPEND differs: {key}")
                visual = info.get("visual_memory", {})
                required = {"read_enabled", "observed", "append_updates", "demo_updates", "bank_observations",
                            "bank_tokens", "frame_index", "passive", "image_delta_norm", "image_changed_fraction",
                            "read_mode", "past_read_enabled", "current_reference_enabled", "readout_norm",
                            "projected_residual_norm", "readout_norm_semantics"}
                if not isinstance(visual, dict) or not required <= set(visual):
                    totals["missing_calls"] += 1
                    continue
                for key, value in {"observed": index, "append_updates": index, "demo_updates": demos,
                                   "bank_observations": index, "bank_tokens": index * 162,
                                   "frame_index": frame, "passive": passive}.items():
                    if type(visual[key]) is not type(value) or visual[key] != value:
                        raise ValueError(f"{role}/{task}: visual APPEND counter/observation mismatch: {key}")
                enabled = role != "visual-off" and not passive and index > 1
                if type(visual["read_enabled"]) is not bool or visual["read_enabled"] != enabled:
                    raise ValueError("Visual READ flag contradicts role or strictly past bank")
                for key, value in {
                    "read_mode": model["read_mode"],
                    "past_read_enabled": enabled and model["read_mode"] == "differential",
                    "current_reference_enabled": enabled,
                    "readout_norm_semantics": "H_past_minus_H_current" if model["read_mode"] == "differential" else "H_current",
                }.items():
                    if type(visual[key]) is not type(value) or visual[key] != value:
                        raise ValueError(f"V12 runtime read-mode/branch differs: {key}")
                totals["read_enabled_calls"] += int(enabled)
                totals["past_read_enabled_calls"] += int(visual["past_read_enabled"])
                totals["current_reference_enabled_calls"] += int(visual["current_reference_enabled"])
                for key in maxima:
                    value = visual[key]
                    if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                            or (key.endswith("fraction") and value > 1) or (not enabled and value != 0)):
                        raise ValueError(f"Invalid visual READ metric/bypass: {key}")
                    maxima[key] = value if maxima[key] is None else max(maxima[key], value)
            totals["append_updates"] += len(records)
            totals["demo_updates"] += demos
    return {**totals, "max_observed_read_metrics": maxima,
            "complete_evidence": totals["completed_sessions"] == expected * len(tasks)
                and not any(totals[k] for k in ("missing_sessions", "missing_calls", "missing_identity", "missing_parent_calls"))}


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
    result.update(trainer_variant=VARIANT, additional_comparisons={}, storage_diagnostics={},
                  visual_diagnostics={}, visual_diagnostics_complete=True)
    settings, models = manifest["settings"], manifest["models"]
    lines = [text.rstrip(), "", "V12: differential/current-only arms; archive1250 and complete AE/LoRA stay frozen."]
    contrasts = (("visual-off", "differential", "same differential checkpoint, visual READ contribution"),
                 ("current-only", "differential", "same planned training budget, independently selected checkpoints; historical differential vs current-only"),
                 ("visual-off", "current-only", "trained current-only vs frozen parent"))
    for control, candidate, description in contrasts:
        if not {control, candidate} <= set(models):
            continue
        for task in settings["tasks"]:
            contexts = [validate_result_identity(root, role, task, manifest) for role in (control, candidate)
                        if read_results(root / role / task / "simulation_results.csv", expected=settings["n_episodes"])]
            if len(contexts) == 2 and any(contexts[0].get(k) != contexts[1].get(k) for k in
                    ("scenario_metadata_sha256", "model_config_sha256", "memory_window", "demo_sampling")):
                raise ValueError("V12 paired scenario or short-memory context differs")
        comparison = _contrast(root, control, candidate, settings["tasks"], settings["n_episodes"], bootstrap_samples)
        result["additional_comparisons"][f"{control}_to_{candidate}"] = comparison
        lines += [f"{control} -> {candidate}: {description}; original short archive remains ON",
                  f"{'Task':<22} {'Paired N':>8} {'Delta':>10} {'Wins/Losses/Same':>19}"]
        for task, row in comparison["tasks"].items():
            delta = "--" if row["paired_delta"] is None else f"{100*row['paired_delta']:+.2f}pp"
            lines.append(f"{task:<22} {row['paired_n']:>8} {delta:>10} {row['wins']:>5}/{row['losses']}/{row['same']}")
        delta, ci = comparison["paired_task_macro_delta"], comparison["paired_task_macro_bootstrap_ci95"]
        if delta is not None:
            lines.append(f"Delta={100*delta:+.2f}pp; N={comparison['paired_n']}; "
                         f"wins/losses={comparison['wins']}/{comparison['losses']}; McNemar p={comparison['mcnemar_exact_p']:.6g}")
            if ci:
                lines.append(f"95% within-task paired bootstrap CI: [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]pp")
        if not comparison["complete"]:
            lines.append("INCOMPLETE: not a full paired comparison.")
    for role, model in models.items():
        if role == "baseline":
            continue
        if model["step"] == 0:
            lines.append(f"{role}: INITIALIZATION_SELECTED (step0), zero learned visual updates; not a trained visual model.")
        else:
            lines.append(f"{role}: {model['checkpoint_status']}, checkpoint step={model['step']}, read_mode={model['read_mode']}.")
        short = completed_write_diagnostics(root, role, settings["tasks"], settings["n_episodes"])
        if (short["keeps"] or short["demo_keeps"] or short["updates"] != short["observations_seen"]
                or short["write_attempts"] != short["updates"]):
            raise ValueError("Frozen parent archive APPEND contract changed")
        visual = completed_visual_diagnostics(root, role, settings["tasks"], settings["n_episodes"], model)
        result["storage_diagnostics"][role] = short
        result["visual_diagnostics"][role] = visual
        result["visual_diagnostics_complete"] &= visual["complete_evidence"] and not short["completed_sessions_missing_diagnostics"]
        lines.append(f"{role}: visual completed sessions={visual['completed_sessions']}, "
                     f"missing={visual['missing_sessions']}; APPEND={visual['append_updates']}; "
                     f"READ={visual['read_enabled_calls']}; missing calls/identity/parent={visual['missing_calls']}/{visual['missing_identity']}/{visual['missing_parent_calls']}")
        lines.append(f"  image READ metrics: {visual['max_observed_read_metrics']}")
    lines += ["Visual-off retains frozen short-archive READ and exactly the same frozen AE; it is NOT all-memory-off.",
              "Both arms and visual-off APPEND original address/content banks and parent archive. Different actions may change write counts.",
              "current-only never reads past visual content; both modes bypass visual output on empty history.",
              "Visual storage representation and retrieval are learned; admission is rule-based, not CVOM/learned merge.",
              "All-history storage is unbounded; differential uses two attention reads vs one in current-only, not equal compute.",
              "On/off gains alone do not establish correct historical-content selectivity; wrong-history controls remain relevant.",
              "A validation/subset score does not establish final test success >=30%."]
    return result, "\n".join(lines) + "\n"

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
    failures, interrupted, fatal = [], False, None
    try:
        for name, model in identity["models"].items():
            if name == "baseline" and "baseline_reference" in identity:
                reference = identity["baseline_reference"]
                print(f"[visual-v12-eval] baseline REUSED: {reference['completed_episodes']} completed; "
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
                    print(f"[visual-v12-eval] resume: {name}/{task} complete", flush=True)
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
                        print(f"[visual-v12-eval] {name}/{task}; {task_dir / 'rollout.log'}", flush=True)
                        try:
                            run_client(client, env=env, log_path=task_dir / "rollout.log", server=server, timeout=args.task_timeout)
                            if len(read_results(task_dir / "simulation_results.csv", expected=args.n_episodes)) != args.n_episodes:
                                raise RuntimeError("Rollout exited without all requested completed episodes")
                            validate_result_identity(output, name, task, identity)
                        except (RuntimeError, TimeoutError, ValueError) as exc:
                            failures.append({"model": name, "task": task, "error": str(exc)})
                            print(f"[visual-v12-eval] ERROR {name}/{task}: {exc}", file=sys.stderr, flush=True)
                        write_report(output, bootstrap_samples=500)
                except (RuntimeError, TimeoutError) as exc:
                    failures.append({"model": name, "error": str(exc)})
                    print(f"[visual-v12-eval] ERROR {name}: {exc}; see server.log", file=sys.stderr, flush=True)
                finally:
                    stop_process(server)
    except KeyboardInterrupt:
        interrupted = True
        fatal = {"error_type": "KeyboardInterrupt", "error": "Evaluation interrupted"}
        raise
    except BaseException as exc:
        fatal = {"error_type": type(exc).__name__, "error": str(exc)}
        failures.append({"scope": "driver_or_report", **fatal})
        raise
    finally:
        try:
            if fatal is None:
                try:
                    result, report = write_report(output)
                    print(report, flush=True)
                    print(f"[visual-v12-eval] Summary: {output / 'comparison_summary.txt'}", flush=True)
                except BaseException as exc:
                    fatal = {"error_type": type(exc).__name__, "error": str(exc)}
                    failures.append({"scope": "final_report", **fatal})
                    raise
        finally:
            try:
                (output / "driver_status.json").write_text(json.dumps(
                    {"interrupted": interrupted, "failures": failures, "fatal": fatal}, indent=2) + "\n")
            finally:
                lock.close()
    return int(bool(failures) or not result["visual_diagnostics_complete"] or not all(model["complete"] for model in result["models"].values()))

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
    print(f"[preflight] compatible visual READ-control models={list(identity['models'])}; tasks={args.tasks}; n={args.n_episodes}", flush=True)
    if args.preflight_only:
        print("[preflight] No model/simulator loaded; no training/evaluation/output started.")
        return 0
    output = resolve_repo_path(args.output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ("." + output.name + ".visual-differential-v12-launch.lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another visual READ-control evaluation owns this output directory") from exc
        return run_evaluation(args, identity, env)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[visual-v12-eval] Interrupted; owned processes stopped, completed episodes resumable.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[visual-v12-eval] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
