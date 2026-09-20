#!/usr/bin/env python3
"""Matched diagnostic Q/K versus Q/K+P.weight learning; never a robot policy.

V15 targets, sampling, past attention, controls, fixed validation and the
predeclared assessment are reused unchanged.  The only trainable-scope change
in arm qkp is image_projection.weight; its bias and every other visual weight,
all named buffers, original HAMLET/AE and external inputs stay unchanged.
This is a new-only bounded run, not an exact-resume or policy checkpoint format.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from gr00t.long_memory.cache import EpisodeCache
from gr00t.long_memory.monitoring import RunLogger, _atomic_json
from gr00t.long_memory.safety_v5 import validate_output_scope
from gr00t.long_memory.train_v3 import _Tee
from run_scripts.robomme.checkpoint_visual_expert_v14 import checkpoint_info
from run_scripts.robomme.demo_tail_sidecar_v13 import DemoTailSidecar
from run_scripts.robomme.prepare_segment_targets_v15 import load_manifest
from run_scripts.robomme.replay_visual_patch_v11 import OBSERVATION_KEYS
from run_scripts.robomme.segment_probe_inputs_v16 import (
    configure_scope, build_probe_inputs_v16, remap_positive_frames,
)
from run_scripts.robomme.train_segment_retrieval_probe_v15 import (
    Inputs as V15Inputs, make_plan, evaluate, update, assess, episode_macro,
    file_hash, source_files, verify_files, digest,
)
from run_scripts.robomme.visual_demo_tail_bank_v13 import VisualDemoTailMemoryV13
from run_scripts.robomme.visual_differential_memory_v12 import VisualDifferentialConfig

KIND = "segment_retrieval_probe_v16"
ARMS = ("qk", "qkp")
QK_NAMES = ("query_projection.weight", "key_projection.weight")
PROJECTION_NAME = "image_projection.weight"


def selected_names(arm):
    if arm not in ARMS:
        raise ValueError("arm must be qk or qkp")
    return QK_NAMES + ((PROJECTION_NAME,) if arm == "qkp" else ())


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--sidecar-dir", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=9151)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def validate_bindings(targets, cache, sidecar, initial):
    """Metadata checks only: no model construction or observation encoding."""
    binding = targets["identity"]
    if (Path(binding["cache_dir"]).resolve() != cache.path
            or Path(binding["sidecar_dir"]).resolve() != sidecar.path
            or binding["cache_fingerprint"] != cache.manifest["fingerprint"]
            or binding["sidecar_fingerprint"] != sidecar.manifest["fingerprint"]
            or binding["original_splits"] != cache.manifest["splits"]):
        raise ValueError("Target/cache/sidecar paths, fingerprints, or splits differ")
    selected = defaultdict(set)
    for row in targets["examples"]:
        if row["split"] not in ("train", "val") or row["episode_id"] not in cache.manifest["splits"][row["split"]]:
            raise ValueError("Target split escaped immutable TRAIN/cache-VAL partition")
        selected[row["split"]].add(row["episode_id"])
    if not selected["train"] or not selected["val"]:
        raise ValueError("Need nonempty TRAIN and held-out episodes")
    if (Path(binding["base_model"]).resolve() != Path(cache.manifest["model_path"]).resolve()
            or Path(binding["base_model"]).resolve() != Path(initial["metadata"]["base_model"]["path"]).resolve()):
        raise ValueError("Targets, cache and initialization require the same original base model")
    if (initial["step"] != 512 or initial["config"]["include_tail"] is not True
            or initial["config"]["read_mode"] != "differential"):
        raise ValueError("Probe requires the fixed V14 final512 differential/tail model")
    if (initial["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]
            or initial["metadata"]["sidecar"]["fingerprint"] != sidecar.manifest["fingerprint"]):
        raise ValueError("Initialization cache/tail identity mismatch")
    return dict(selected)


def preflight(args):
    names = selected_names(args.arm)
    if (args.max_steps <= 0 or args.batch_size <= 0 or args.eval_steps <= 0 or args.seed < 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("Invalid fixed training options")
    targets = load_manifest(args.targets, verify_files=True)
    cache = EpisodeCache(args.cache_dir)
    sidecar = DemoTailSidecar(args.sidecar_dir, expected_cache_fingerprint=cache.manifest["fingerprint"])
    init = Path(args.init_checkpoint).resolve(strict=True)
    header = json.loads((init / "checkpoint.json").read_text())
    initial = checkpoint_info(header["metadata"]["base_model"]["path"], init)
    selected = validate_bindings(targets, cache, sidecar, initial)
    plan = make_plan(targets["examples"], args.max_steps, args.batch_size, args.seed)
    signatures = source_files()
    for manifest_key in ("files_sha256", "source_sha256"):
        for path, signature in targets[manifest_key].items():
            if path in signatures and signatures[path] != signature:
                raise ValueError("Target producer/source identity conflicts with current source")
            signatures[path] = signature
    signatures.update({str(init / name): file_hash(init / name)
                       for name in ("checkpoint.json", "visual.safetensors", "expert.safetensors", "training_state.pt")})
    signatures[str(Path(args.targets).resolve())] = file_hash(args.targets)
    signatures[str(cache.path / "manifest.json")] = file_hash(cache.path / "manifest.json")
    signatures[str(sidecar.path / "manifest.json")] = file_hash(sidecar.path / "manifest.json")
    output = validate_output_scope(args.output_dir, init, cache.path, sidecar.path, Path(args.targets).parent,
        initial["metadata"]["initial_parent"]["path"], initial["metadata"]["base_model"]["path"],
        cache.manifest["dataset_path"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a NEW output directory; V16 does not auto-resume")
    protocol = {
        "kind": KIND, "arm": args.arm,
        "train": {k: v for k, v in vars(args).items() if k not in ("output_dir", "preflight_only")},
        "plan_sha256": digest(plan), "files_sha256": signatures,
        "checkpoint_kind": "diagnostic_only_not_policy",
        "scope": "query_key_and_image_projection_weight" if args.arm == "qkp" else "query_projection_and_key_projection_only",
        "selected_parameter_names": list(names),
        "frozen_buffer_contract": "all_named_buffers_exact_to_initial_including_nonpersistent",
        "initial_visual_sha256": signatures[str(init / "visual.safetensors")],
        "fixed_final_selection": True, "time_only_scale": 16, "permutation_seed": args.seed + 1,
        "minimum_val_episodes": 10, "minimum_val_queries": 20,
        "bootstrap_seed": 9151, "bootstrap_samples": 10000,
        "control_semantics": "all_demo_content_bijection_targets_follow_content",
        "torch_version": torch.__version__, "numpy_version": np.__version__,
    }
    verify_files(signatures)  # Fresh full scan BEFORE any model/GPU/output.
    return {"targets": targets, "cache": cache, "sidecar": sidecar, "initial": initial,
            "init": init, "output": output, "selected": selected, "plan": plan, "protocol": protocol}


class Inputs(V15Inputs):
    """Same target binding and columns as V15; only the differentiable builder differs."""

    def get(self, memory, row, permutation_seed=None):
        episode = self.episodes.fetch(row["episode_id"])
        query = row["decision"]
        if not bool(episode["decision_mask"][query]) or int(episode["frames"][query]) != row["query_frame"]:
            raise ValueError("Target query differs from immutable active cache decision")
        observations = {name: episode[name] for name in OBSERVATION_KEYS}
        current, bank, info = build_probe_inputs_v16(memory, observations, query,
            sidecar=self.sidecar.load(row["episode_id"]), record=self.records[row["episode_id"]],
            episode_id=row["episode_id"], cache_fingerprint=self.cache.manifest["fingerprint"],
            sidecar_fingerprint=self.sidecar.manifest["fingerprint"], permutation_seed=permutation_seed)
        if info["candidate_frames"] != row["candidate_frames"]:
            raise ValueError("Label candidate frame set differs from actual observation bank")
        positives = set(remap_positive_frames(row["positive_frames"], info))
        mask = torch.tensor([[frame in positives for frame in info["candidate_frames"]]],
                            dtype=torch.bool, device=bank.tokens.device)
        return current, bank, mask, info


def frozen_snapshot(memory, arm):
    names = set(selected_names(arm))
    return {
        "parameters": {name: value.detach().cpu().clone() for name, value in memory.named_parameters() if name not in names},
        "buffers": {name: value.detach().cpu().clone() for name, value in memory.named_buffers()},
    }


def assert_scope(memory, optimizer, arm, snapshot, step, *, require_gradients=False):
    """Strict ownership plus finite Adam state; includes nonpersistent buffers."""
    names = selected_names(arm)
    parameters = dict(memory.named_parameters())
    if set(snapshot["parameters"]) != set(parameters) - set(names):
        raise ValueError("Frozen snapshot must contain exactly all nonselected parameters")
    if set(snapshot["buffers"]) != set(dict(memory.named_buffers())):
        raise ValueError("Frozen snapshot must contain exactly all named buffers")
    for name, parameter in parameters.items():
        selected = name in names
        if parameter.requires_grad is not selected:
            raise ValueError(f"V16 trainable scope differs: {name}")
        if parameter.dtype != torch.float32 or not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(f"Nonfinite/non-FP32 parameter: {name}")
        if not selected:
            if parameter.grad is not None or not torch.equal(parameter.detach().cpu(), snapshot["parameters"][name]):
                raise ValueError(f"Frozen parameter changed or acquired a gradient: {name}")
        elif ((require_gradients and parameter.grad is None)
              or (parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()))):
            raise FloatingPointError(f"Disconnected/nonfinite selected gradient: {name}")
    for name, buffer in memory.named_buffers():
        expected = snapshot["buffers"][name]
        if buffer.dtype != expected.dtype or buffer.shape != expected.shape or not torch.equal(buffer.detach().cpu(), expected):
            raise ValueError(f"Frozen named buffer changed: {name}")
    if not isinstance(optimizer, torch.optim.AdamW) or len(optimizer.param_groups) != 1:
        raise ValueError("Expected one explicit zero-decay AdamW group")
    group = optimizer.param_groups[0]
    if len(group["params"]) != len(names) or any(p is not parameters[name] for p, name in zip(group["params"], names)):
        raise ValueError("Optimizer parameter identity/order differs from declared scope")
    if group["weight_decay"] != 0:
        raise ValueError("V16 matches the zero-weight-decay V15 optimizer")
    expected_state = {parameters[name] for name in names} if step else set()
    if set(optimizer.state) != expected_state:
        raise ValueError("Optimizer state ownership differs from selected parameters/step")
    for parameter, state in optimizer.state.items():
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Unexpected AdamW state schema")
        tick = state["step"]
        if not isinstance(tick, torch.Tensor) or tick.numel() != 1 or not bool(torch.isfinite(tick).all()) or float(tick) != step:
            raise ValueError("Optimizer state step differs from checkpoint step")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if value.dtype != torch.float32 or value.shape != parameter.shape or not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Nonfinite or mismatched AdamW tensor: {key}")
    return True


def buffer_signatures(snapshot):
    import hashlib
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "sha256": hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            for name, value in snapshot["buffers"].items()}


def save_checkpoint(output, step, memory, optimizer, protocol, snapshot):
    """New immutable diagnostic bundle; publish only after all payloads validate."""
    if (protocol.get("kind") != KIND or protocol.get("arm") not in ARMS
            or protocol.get("selected_parameter_names") != list(selected_names(protocol["arm"]))
            or protocol.get("train", {}).get("arm") != protocol["arm"]):
        raise ValueError("Checkpoint protocol kind/arm/scope is inconsistent")
    if type(step) is not int or not 0 <= step <= protocol["train"]["max_steps"]:
        raise ValueError("Checkpoint step is outside the fixed horizon")
    output = Path(output)
    destination = output / f"checkpoint-{step:06d}"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Immutable diagnostic checkpoint already exists")
    verify_files(protocol["files_sha256"])
    assert_scope(memory, optimizer, protocol["arm"], snapshot, step, require_gradients=step > 0)
    folder = Path(tempfile.mkdtemp(prefix=f".checkpoint-{step:06d}-", dir=output))
    # Preserve incomplete private temp directories if saving fails. They never
    # have a published checkpoint name, and existing checkpoints remain intact.
    save_file({name: value.detach().cpu().contiguous() for name, value in memory.state_dict().items()},
              str(folder / "probe.safetensors"))
    torch.save({"optimizer": optimizer.state_dict(), "step": step,
                "selected_parameter_names": protocol["selected_parameter_names"], "arm": protocol["arm"]},
               folder / "optimizer.pt")
    _atomic_json(folder / "checkpoint.json", {
        "kind": KIND, "arm": protocol["arm"], "step": step, "deployable_policy": False,
        "selected_parameter_names": protocol["selected_parameter_names"],
        "protocol_sha256": digest(protocol), "frozen_buffers": buffer_signatures(snapshot),
        "payload_sha256": file_hash(folder / "probe.safetensors"),
        "optimizer_sha256": file_hash(folder / "optimizer.pt"),
    })
    folder.rename(destination)
    return destination


def run(args, context):
    output, protocol, plan = context["output"], context["protocol"], context["plan"]
    verify_files(protocol["files_sha256"])  # Before model/device allocation.
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "protocol.json", protocol)
    _atomic_json(output / "query_plan.json", plan)
    with (output / "train.log").open("x") as log, redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
        start, step = time.monotonic(), 0
        try:
            memory = VisualDemoTailMemoryV13(VisualDifferentialConfig(**context["initial"]["config"]["visual"]), read_mode="differential")
            memory.load_state_dict(load_file(str(context["init"] / "visual.safetensors")), strict=True)
            memory.to(args.device).eval()
            selected = configure_scope(memory, train_projection=args.arm == "qkp")
            optimizer = torch.optim.AdamW(selected, lr=args.learning_rate, weight_decay=0.)
            snapshot = frozen_snapshot(memory, args.arm)
            assert_scope(memory, optimizer, args.arm, snapshot, 0)
            provider = Inputs(context["cache"], context["sidecar"], context["targets"]["examples"])
            logger = RunLogger(output)
            heldout = [row for row in context["targets"]["examples"] if row["split"] == "val"]
            initial = evaluate(memory, provider, heldout, args.seed + 1)
            _atomic_json(output / "validation-000000.json", initial)

            def log_validation(boundary, rows):
                for mode in ("correct", "content_permuted"):
                    chosen = [row for row in rows if row["mode"] == mode]
                    values = {metric: float(np.mean(list(episode_macro(chosen, metric).values())))
                              for metric in ("loss", "positive_mass", "span_hit")}
                    logger.log(boundary, "val/" + mode, values)

            save_checkpoint(output, 0, memory, optimizer, protocol, snapshot)
            log_validation(0, initial)
            for step, batch in enumerate(plan["schedule"], 1):
                rows = [provider.examples[(row["episode_id"], row["decision"])] for row in batch]
                values = update(memory, optimizer, rows, provider, 1.)
                assert_scope(memory, optimizer, args.arm, snapshot, step, require_gradients=True)
                logger.log(step, "train", {**values, "learning_rate": args.learning_rate,
                                           "elapsed_seconds": time.monotonic() - start})
                if step % 16 == 0:
                    print(f"[probe-v16:{args.arm}] {step}/{args.max_steps} loss={values['loss']:.6f} grad={values['grad_norm']:.5f}", flush=True)
                if step % args.eval_steps == 0 or step == args.max_steps:
                    validation = evaluate(memory, provider, heldout, args.seed + 1)
                    _atomic_json(output / f"validation-{step:06d}.json", validation)
                    log_validation(step, validation)
                    save_checkpoint(output, step, memory, optimizer, protocol, snapshot)
                    logger.plot()
            result = assess(initial, validation)
            _atomic_json(output / "assessment.json", result)
            verify_files(protocol["files_sha256"])
            assert_scope(memory, optimizer, args.arm, snapshot, step, require_gradients=True)
            _atomic_json(output / "status.json", {"status": "complete", "kind": KIND, "arm": args.arm,
                "step": step, "processed_queries": step * args.batch_size, "assessment": result["decision"],
                "policy_ready": False, "elapsed_seconds": time.monotonic() - start})
            print("[probe-v16]", args.arm, result["decision"], "NOT a RoboMME task-success result", flush=True)
        except BaseException as error:
            integrity_error = None
            try:
                verify_files(protocol["files_sha256"])
            except BaseException as integrity:
                integrity_error = repr(integrity)
            _atomic_json(output / "status.json", {"status": "failed", "kind": KIND, "arm": args.arm,
                "step": step, "policy_ready": False, "error": repr(error), "integrity_error": integrity_error})
            traceback.print_exc()
            raise
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    context = preflight(args)
    print("[preflight]", json.dumps({"arm": args.arm,
        "train_episodes": len(context["selected"]["train"]), "val_episodes": len(context["selected"]["val"]),
        "queries": len(context["targets"]["examples"]), "plan_sha256": digest(context["plan"]),
        "selected_parameters": context["protocol"]["selected_parameter_names"], "diagnostic_only": True}), flush=True)
    if args.preflight_only:
        print("[preflight] no model/GPU/output/training started", flush=True)
        return 0
    return run(args, context)


if __name__ == "__main__":
    raise SystemExit(main())
