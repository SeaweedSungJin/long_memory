#!/usr/bin/env python3
"""Bounded weak-segment retrieval learning, NOT a deployable robot policy.

Only a separate copy of the V14 visual Q/K is optimized. Annotations enter the
loss/metrics after observation-only attention, never the query or bank. This
probe asks whether the representation can learn a historical correspondence;
it does not measure action quality, learned admission, or RoboMME success.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import math
from pathlib import Path
import random
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
from gr00t.long_memory.cache_reader_v3 import MappedEpisodes
from gr00t.long_memory.monitoring import RunLogger, _atomic_json
from gr00t.long_memory.safety_v5 import validate_output_scope
from gr00t.long_memory.train_v3 import _Tee
from run_scripts.robomme.checkpoint_visual_expert_v14 import checkpoint_info
from run_scripts.robomme.demo_tail_sidecar_v13 import DemoTailSidecar
from run_scripts.robomme.replay_visual_patch_v11 import OBSERVATION_KEYS
from run_scripts.robomme.segment_probe_inputs_v15 import build_probe_inputs, remap_positive_frames
from run_scripts.robomme.segment_retrieval_probe_v15 import (
    configure_qk_only, past_frame_probabilities, segment_mass_loss,
)
from run_scripts.robomme.visual_demo_tail_bank_v13 import VisualDemoTailMemoryV13
from run_scripts.robomme.visual_differential_memory_v12 import VisualDifferentialConfig

KIND = "segment_retrieval_probe_v15"
CONTROL_NAMES = ("uniform_all", "uniform_demo", "time_only")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_files():
    paths = list((ROOT / "gr00t").rglob("*.py"))
    # Include the checkpoint/replay helpers' transitive local imports, not just
    # this runner's direct imports. All current files remain frozen during a run.
    paths += list((ROOT / "run_scripts/robomme").glob("*.py"))
    return {str(p.resolve()): file_hash(p) for p in sorted(set(paths))}


def verify_files(signatures):
    for path, signature in signatures.items():
        if file_hash(path) != signature:
            raise ValueError(f"Immutable input/source changed: {path}")


def make_plan(examples, steps, batch_size, seed):
    """Episode-balanced, label-independent TRAIN sampling, fixed before training."""
    if steps <= 0 or batch_size <= 0 or seed < 0:
        raise ValueError("Invalid sampling options")
    groups = defaultdict(list)
    seen = set()
    for row in examples:
        key = (row["episode_id"], row["decision"])
        if key in seen:
            raise ValueError("Duplicate target query")
        seen.add(key)
        if row["split"] == "train":
            groups[row["episode_id"]].append(row["decision"])
    if not groups:
        raise ValueError("No eligible TRAIN examples")
    rng = random.Random(seed)
    ids, queue, schedule = sorted(groups), [], []
    for _ in range(steps):
        batch = []
        for _ in range(batch_size):
            if not queue:
                queue = ids.copy()
                rng.shuffle(queue)
            eid = queue.pop()
            batch.append({"episode_id": eid, "decision": rng.choice(sorted(groups[eid]))})
        schedule.append(batch)
    return {"sampling": "episode_balanced_train_only", "seed": seed,
            "steps": steps, "batch_size": batch_size, "schedule": schedule}


def control_probabilities(frames, is_demo, query_frame, n_demo):
    """Causal controls: no execution duration, true occurrence, or label input."""
    if (frames.ndim != 1 or is_demo.shape != frames.shape or is_demo.dtype != torch.bool
            or not len(frames) or not bool(is_demo.any())
            or bool((frames >= query_frame).any()) or n_demo <= 0):
        raise ValueError("Invalid strict-past control metadata")
    if not torch.equal(is_demo, frames < n_demo):
        raise ValueError("Demo boundary disagrees with observed frame metadata")
    uniform = torch.ones(len(frames), device=frames.device, dtype=torch.float32) / len(frames)
    demo = is_demo.float() / is_demo.sum()
    # Fixed one-stride bandwidth, chosen before looking at held-out outcomes.
    expected = min(max(int(query_frame) - int(n_demo), 0), int(n_demo) - 1)
    logits = -(frames.float() - expected).abs() / 16.0
    logits = logits.masked_fill(~is_demo, -torch.inf)
    return {"uniform_all": uniform, "uniform_demo": demo, "time_only": logits.softmax(0)}


def distribution_metrics(probabilities, positive):
    if probabilities.ndim != 1 or positive.shape != probabilities.shape or positive.dtype != torch.bool:
        raise ValueError("Expected one probability and positive-mask vector")
    if (not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any())
            or not torch.isclose(probabilities.sum(), probabilities.new_tensor(1.), atol=1e-5)
            or not bool(positive.any()) or bool(positive.all())):
        raise ValueError("Invalid informative retrieval distribution")
    mass = probabilities[positive].sum()
    if not bool(mass > 0):
        raise FloatingPointError("Positive probability mass underflowed; refusing a synthetic finite score")
    return {"positive_mass": float(mass), "loss": float(-mass.log()),
            "span_hit": float(positive[probabilities.argmax()])}


def episode_macro(rows, metric):
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["episode_id"])].append(float(row[metric]))
    if not groups:
        raise ValueError("No eligible evaluation records")
    return {eid: float(np.mean(values)) for eid, values in sorted(groups.items())}


def paired_gain(rows, control, *, bootstrap_samples=10000, seed=9151):
    """Positive means lower probe loss; resample episodes, never adjacent frames."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode_id"]].append(row[control + "/loss"] - row["loss"])
    values = np.asarray([np.mean(v) for _, v in sorted(grouped.items())], dtype=np.float64)
    if not len(values):
        raise ValueError("No episodes for paired comparison")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(bootstrap_samples, len(values)))].mean(1)
    lo, hi = np.quantile(draws, [.025, .975])
    return {"episodes": len(values), "mean_loss_gain": float(values.mean()),
            "ci95": [float(lo), float(hi)], "bootstrap_samples": bootstrap_samples, "seed": seed}


def assess(initial, final):
    """Predeclared gate for a subsequent action experiment, not robot success."""
    def keyed(rows):
        result = {(r["episode_id"], r["decision"], r["mode"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate validation rows")
        return result
    before, after = keyed(initial), keyed(final)
    if before.keys() != after.keys():
        raise ValueError("Fixed held-out validation identities changed")
    prepared = []
    for key, row in after.items():
        prepared.append({**row, "initial/loss": before[key]["loss"]})
    conditions, contrasts = [], {}
    for mode in ("correct", "content_permuted"):
        rows = [r for r in prepared if r["mode"] == mode]
        enough = len({r["episode_id"] for r in rows}) >= 10 and len(rows) >= 20
        conditions.append(enough)
        for control in (("initial", "uniform_all", "uniform_demo", "time_only")
                        if mode == "correct" else ("uniform_demo", "time_only")):
            contrast = paired_gain(rows, control)
            contrasts[mode + "/" + control] = contrast
            conditions.append(contrast["ci95"][0] > 0)
    sufficient = all(len({r["episode_id"] for r in prepared if r["mode"] == m}) >= 10
                     and sum(r["mode"] == m for r in prepared) >= 20
                     for m in ("correct", "content_permuted"))
    return {"decision": "inconclusive_coverage" if not sufficient else ("go_to_action_probe" if all(conditions) else "no_go"),
            "contrasts": contrasts, "metric": "weak_maximal_segment_retrieval_not_robot_success",
            "policy_ready": False, "goal_30_percent_achieved": False,
            "limitations": ["Weak maximal-run targets are not verified causal cues or atomic events.",
                "Content permutation is a diagnostic, not a physically valid rollout.",
                "Q/K training does not train admission, values, or the Action Expert.",
                "Passing permits a subsequent action experiment, never claims RoboMME improvement."]}


class Inputs:
    def __init__(self, cache, sidecar, examples):
        self.cache, self.sidecar = cache, sidecar
        self.episodes = MappedEpisodes(cache, max_cached=2)
        self.records = {r["episode_id"]: r for r in sidecar.manifest["episodes"]}
        self.examples = {(r["episode_id"], r["decision"]): r for r in examples}

    def get(self, memory, row, permutation_seed=None):
        ep = self.episodes.fetch(row["episode_id"])
        q = row["decision"]
        if not bool(ep["decision_mask"][q]) or int(ep["frames"][q]) != row["query_frame"]:
            raise ValueError("Target query differs from immutable active cache decision")
        # No label, raw telemetry, action, or future target enters the builder.
        observations = {name: ep[name] for name in OBSERVATION_KEYS}
        current, bank, info = build_probe_inputs(memory, observations, q,
            sidecar=self.sidecar.load(row["episode_id"]), record=self.records[row["episode_id"]],
            episode_id=row["episode_id"], cache_fingerprint=self.cache.manifest["fingerprint"],
            sidecar_fingerprint=self.sidecar.manifest["fingerprint"], permutation_seed=permutation_seed)
        if info["candidate_frames"] != row["candidate_frames"]:
            raise ValueError("Label candidate frame set differs from actual observation bank")
        positives = remap_positive_frames(row["positive_frames"], info)
        mask = torch.tensor([[f in set(positives) for f in info["candidate_frames"]]],
                            dtype=torch.bool, device=bank.tokens.device)
        return current, bank, mask, info


@torch.no_grad()
def evaluate(memory, provider, examples, permutation_seed):
    rows = []
    for example in examples:
        for mode, seed in (("correct", None), ("content_permuted", permutation_seed)):
            current, bank, positive, info = provider.get(memory, example, seed)
            p = past_frame_probabilities(memory, current, bank)[0]
            row = {"episode_id": example["episode_id"], "decision": example["decision"], "mode": mode,
                   "old_positive_count": len(example["old_positive_frames"]),
                   "query_frame": example["query_frame"],
                   "candidate_frames": info["candidate_frames"],
                   "original_positive_frames": example["positive_frames"],
                   "positive_frames": remap_positive_frames(example["positive_frames"], info),
                   "destination_to_source_frames": info["destination_to_source_frames"],
                   **distribution_metrics(p, positive[0])}
            controls = control_probabilities(bank.frames[0], bank.is_demo[0],
                                             example["query_frame"], example["n_demo"])
            for name, distribution in controls.items():
                row.update({name + "/" + k: v for k, v in distribution_metrics(distribution, positive[0]).items()})
            rows.append(row)
    return rows


def update(memory, optimizer, batch, provider, clip):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for row in batch:
        current, bank, positive, _ = provider.get(memory, row)
        probabilities = past_frame_probabilities(memory, current, bank)
        loss = segment_mass_loss(probabilities, positive, bank.valid)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite retrieval loss")
        (loss / len(batch)).backward()
        losses.append(float(loss.detach()))
    selected = [p for p in memory.parameters() if p.requires_grad]
    if not selected or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in selected):
        raise FloatingPointError("Missing/nonfinite Q/K gradient")
    norm = torch.nn.utils.clip_grad_norm_(selected, clip, error_if_nonfinite=True)
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in memory.parameters()):
        raise FloatingPointError("Nonfinite updated parameter")
    return {"loss": float(np.mean(losses)), "grad_norm": float(norm)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True, help="Published weak-target manifest.json")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--sidecar-dir", required=True)
    p.add_argument("--init-checkpoint", required=True, help="Immutable V14 final512 (never overwritten)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--eval-steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=9151)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)
    if (args.max_steps <= 0 or args.batch_size <= 0 or args.eval_steps <= 0 or args.seed < 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("Invalid fixed training options")
    # Preparation code validates raw-label/cached-frame alignment and provenance.
    from run_scripts.robomme.prepare_segment_targets_v15 import load_manifest
    targets = load_manifest(args.targets, verify_files=True)
    cache = EpisodeCache(args.cache_dir)
    sidecar = DemoTailSidecar(args.sidecar_dir, expected_cache_fingerprint=cache.manifest["fingerprint"])
    binding = targets["identity"]
    if (Path(binding["cache_dir"]).resolve() != cache.path
            or Path(binding["sidecar_dir"]).resolve() != sidecar.path
            or binding["cache_fingerprint"] != cache.manifest["fingerprint"]
            or binding["sidecar_fingerprint"] != sidecar.manifest["fingerprint"]
            or binding["original_splits"] != cache.manifest["splits"]):
        raise ValueError("Target/cache/sidecar paths, fingerprints, or splits differ")
    examples = targets["examples"]
    selected = defaultdict(set)
    for row in examples:
        if row["episode_id"] not in cache.manifest["splits"][row["split"]]:
            raise ValueError("Target split escaped immutable episode partition")
        selected[row["split"]].add(row["episode_id"])
    if not selected["train"] or not selected["val"]:
        raise ValueError("Need nonempty TRAIN and held-out episodes")
    init = Path(args.init_checkpoint).resolve(strict=True)
    header = json.loads((init / "checkpoint.json").read_text())
    info = checkpoint_info(header["metadata"]["base_model"]["path"], init)
    if (Path(binding["base_model"]).resolve() != Path(cache.manifest["model_path"]).resolve()
            or Path(binding["base_model"]).resolve() != Path(info["metadata"]["base_model"]["path"]).resolve()):
        raise ValueError("Weak targets, frozen cache and visual initialization require the same base model")
    if info["step"] != 512 or not info["config"]["include_tail"] or info["config"]["read_mode"] != "differential":
        raise ValueError("Probe requires the fixed V14 final512 differential/tail model")
    if (info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]
            or info["metadata"]["sidecar"]["fingerprint"] != sidecar.manifest["fingerprint"]):
        raise ValueError("Initialization cache/tail identity mismatch")
    plan = make_plan(examples, args.max_steps, args.batch_size, args.seed)
    signatures = source_files()
    for manifest_key in ("files_sha256", "source_sha256"):
        for path, signature in targets[manifest_key].items():
            if path in signatures and signatures[path] != signature:
                raise ValueError("Target producer/source fingerprint conflicts with current source")
            signatures[path] = signature
    signatures.update({str(init / name): file_hash(init / name)
                       for name in ("checkpoint.json", "visual.safetensors", "expert.safetensors", "training_state.pt")})
    signatures[str(Path(args.targets).resolve())] = file_hash(args.targets)
    signatures[str(cache.path / "manifest.json")] = file_hash(cache.path / "manifest.json")
    signatures[str(sidecar.path / "manifest.json")] = file_hash(sidecar.path / "manifest.json")
    output = validate_output_scope(args.output_dir, init, cache.path, sidecar.path, Path(args.targets).parent,
        info["metadata"]["initial_parent"]["path"], info["metadata"]["base_model"]["path"],
        cache.manifest["dataset_path"])
    if output.exists():
        raise FileExistsError("Use a NEW output directory; this bounded probe does not auto-resume")
    train_config = {k: v for k, v in vars(args).items() if k not in ("output_dir", "preflight_only")}
    protocol = {"kind": KIND, "train": train_config, "plan_sha256": digest(plan),
                "files_sha256": signatures, "checkpoint_kind": "diagnostic_only_not_policy",
                "scope": "query_projection_and_key_projection_only", "fixed_final_selection": True,
                "time_only_scale": 16, "permutation_seed": args.seed + 1,
                "minimum_val_episodes": 10, "minimum_val_queries": 20,
                "bootstrap_seed": 9151, "bootstrap_samples": 10000,
                "control_semantics": "all_demo_content_bijection_targets_follow_content",
                "torch_version": torch.__version__, "numpy_version": np.__version__}
    print("[preflight]", json.dumps({"train_episodes": len(selected["train"]), "val_episodes": len(selected["val"]),
        "queries": len(examples), "plan_sha256": digest(plan), "diagnostic_only": True}), flush=True)
    if args.preflight_only:
        print("[preflight] no model/GPU/output/training started", flush=True)
        return 0
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "protocol.json", protocol)
    _atomic_json(output / "query_plan.json", plan)
    with (output / "train.log").open("x") as log, redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
        start, step = time.monotonic(), 0
        try:
            memory = VisualDemoTailMemoryV13(VisualDifferentialConfig(**info["config"]["visual"]), read_mode="differential")
            memory.load_state_dict(load_file(str(init / "visual.safetensors")), strict=True)
            memory.to(args.device).eval()
            configure_qk_only(memory)
            frozen = {n: v.detach().cpu().clone() for n, v in memory.state_dict().items()
                      if not (n.startswith("query_projection.") or n.startswith("key_projection."))}
            optimizer = torch.optim.AdamW([p for p in memory.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.)
            provider, logger = Inputs(cache, sidecar, examples), RunLogger(output)
            heldout = [r for r in examples if r["split"] == "val"]
            lookup = provider.examples
            initial = evaluate(memory, provider, heldout, args.seed + 1)
            _atomic_json(output / "validation-000000.json", initial)

            def save(boundary):
                verify_files(signatures)
                for name, value in memory.state_dict().items():
                    if name in frozen and not torch.equal(value.cpu(), frozen[name]):
                        raise RuntimeError("Frozen encoder/value/output changed")
                destination = output / f"checkpoint-{boundary:06d}"
                if destination.exists():
                    raise FileExistsError("Immutable probe checkpoint already exists")
                folder = Path(tempfile.mkdtemp(prefix=f".checkpoint-{boundary:06d}-", dir=output))
                save_file({n: v.detach().cpu().contiguous() for n, v in memory.state_dict().items()}, str(folder / "probe.safetensors"))
                torch.save({"optimizer": optimizer.state_dict(), "step": boundary}, folder / "optimizer.pt")
                _atomic_json(folder / "checkpoint.json", {"kind": KIND, "step": boundary,
                    "deployable_policy": False, "protocol_sha256": digest(protocol),
                    "payload_sha256": file_hash(folder / "probe.safetensors"),
                    "optimizer_sha256": file_hash(folder / "optimizer.pt")})
                # Incomplete fresh temporary folders are preserved on failure;
                # no valid-looking checkpoint is published before all payloads.
                folder.rename(destination)

            def log_validation(boundary, rows):
                for mode in ("correct", "content_permuted"):
                    chosen = [r for r in rows if r["mode"] == mode]
                    values = {metric: float(np.mean(list(episode_macro(chosen, metric).values())))
                              for metric in ("loss", "positive_mass", "span_hit")}
                    logger.log(boundary, "val/" + mode, values)

            save(0)
            log_validation(0, initial)
            for step, batch in enumerate(plan["schedule"], 1):
                rows = [lookup[(r["episode_id"], r["decision"])] for r in batch]
                values = update(memory, optimizer, rows, provider, 1.)
                if any(p.grad is not None for p in memory.parameters() if not p.requires_grad):
                    raise RuntimeError("Gradient escaped Q/K-only scope")
                logger.log(step, "train", {**values, "learning_rate": args.learning_rate,
                                           "elapsed_seconds": time.monotonic() - start})
                if step % 16 == 0:
                    print(f"[probe] {step}/{args.max_steps} loss={values['loss']:.6f} grad={values['grad_norm']:.5f}", flush=True)
                if step % args.eval_steps == 0 or step == args.max_steps:
                    validation = evaluate(memory, provider, heldout, args.seed + 1)
                    _atomic_json(output / f"validation-{step:06d}.json", validation)
                    log_validation(step, validation)
                    save(step)
                    logger.plot()
            result = assess(initial, validation)
            _atomic_json(output / "assessment.json", result)
            verify_files(signatures)
            _atomic_json(output / "status.json", {"status": "complete", "step": step,
                "processed_queries": step * args.batch_size, "assessment": result["decision"],
                "policy_ready": False, "elapsed_seconds": time.monotonic() - start})
            print("[probe]", result["decision"], "NOT a RoboMME task-success result", flush=True)
        except BaseException as error:
            _atomic_json(output / "status.json", {"status": "failed", "step": step,
                "policy_ready": False, "error": repr(error)})
            traceback.print_exc()
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
