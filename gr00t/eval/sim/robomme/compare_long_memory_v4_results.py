"""Paired v4 reports distinguish original HAMLET, AE adaptation and storage.

Reports use the unchanged shared scenario/episode validation and paired
statistics. Inference removal of memory is not called a trained control.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .compare_long_memory_results import build_report
from .compare_long_memory_v3_results import _contrast, completed_storage_diagnostics


def build_v4_report(run_dir, *, bootstrap_samples=5000):
    root = Path(run_dir)
    manifest = json.loads((root / "comparison_manifest.json").read_text())
    if manifest.get("trainer_variant") != "action_expert_v4":
        raise ValueError("Expected an action_expert_v4 comparison manifest")
    result, report = build_report(root, bootstrap_samples=bootstrap_samples)
    result["trainer_variant"] = "action_expert_v4"
    result["additional_comparisons"] = {}
    result["storage_diagnostics"] = {}
    result["model_descriptions"] = {name: model.get("description", "")
                                    for name, model in manifest["models"].items()}
    lines = [report.rstrip(), "", "V4 adapted-expert comparisons"]
    for name, description in result["model_descriptions"].items():
        lines.append(f"  {name}: {description}")
    if "reader" in manifest["models"]:
        if manifest["models"]["reader"].get("reader_mode") == "none":
            lines.append("Original HAMLET -> reader compares AE-only training (reader_mode=none), without long memory.")
        else:
            lines.append("Original HAMLET -> reader includes BOTH AE adaptation and memory changes.")
    lines.append("expert-only disables memory in the SAME adapted Stage 1 model at inference; "
                 "it is NOT a separately trained no-memory control.")
    roles = set(manifest["models"])
    for left, right, explanation in (
        ("expert-only", "reader", "Same Stage 1 adapted expert: memory off -> on (inference ablation)"),
        ("reader", "memory", "Selected Stage 1 -> Stage 2; check training_metadata.stage1_parent for matched frozen weights"),
        ("fifo", "memory", "Same Stage 2 expert + reader: FIFO -> learned storage; isolates storage policy"),
    ):
        if not {left, right} <= roles:
            continue
        contrast = _contrast(root, left, right, manifest["settings"]["tasks"],
                             manifest["settings"]["n_episodes"], bootstrap_samples)
        result["additional_comparisons"][f"{left}_to_{right}"] = contrast
        lines.append(explanation)
        delta = contrast["paired_task_macro_delta"]
        if delta is None:
            lines.append("  INCOMPLETE: no matched completed episodes.")
            continue
        lines.append(f"  Delta={100*delta:+.2f}pp; N={contrast['paired_n']}; "
                     f"wins/losses/same={contrast['wins']}/{contrast['losses']}/{contrast['same']}; "
                     f"McNemar p={contrast['mcnemar_exact_p']:.6g}")
        interval = contrast["paired_task_macro_bootstrap_ci95"]
        if interval:
            lines.append(f"  95% within-task paired bootstrap CI: [{100*interval[0]:+.2f}, {100*interval[1]:+.2f}]pp")
        if not contrast["complete"]:
            lines.append("  INCOMPLETE: only matched completed episodes are included.")
    lines += ["", "Storage decisions (final cumulative counters of completed sessions)"]
    for name, model in manifest["models"].items():
        if name in ("baseline", "expert-only") or model.get("reader_mode") == "none":
            continue
        counters = completed_storage_diagnostics(root, name, manifest["settings"]["tasks"],
                                                 manifest["settings"]["n_episodes"])
        result["storage_diagnostics"][name] = counters
        lines.append(f"  {name}: recorded sessions={counters['completed_sessions_with_diagnostics']}; "
                     f"missing diagnostics={counters['completed_sessions_missing_diagnostics']}; "
                     f"learned accept/reject/replace={counters['learned_accepted']}/"
                     f"{counters['learned_rejected']}/{counters['learned_replaced']}; "
                     f"forced writes={counters['forced_writes']}")
    lines.append("Missing diagnostics are not evidence of zero writes; FIFO has no learned write decisions.")
    return result, "\n".join(lines) + "\n"


def write_v4_report(run_dir, *, bootstrap_samples=5000):
    result, report = build_v4_report(run_dir, bootstrap_samples=bootstrap_samples)
    root = Path(run_dir)
    payloads = {"comparison_summary.json": json.dumps(result, indent=2, allow_nan=False) + "\n",
                "comparison_summary.txt": report}
    for name, payload in payloads.items():
        descriptor, filename = tempfile.mkstemp(prefix=f".{name}-", dir=root)
        temporary = Path(filename)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(root / name)
        finally:
            temporary.unlink(missing_ok=True)
    return result, report
