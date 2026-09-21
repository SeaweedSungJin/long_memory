#!/usr/bin/env python3
"""Scoped native V19 READ OFF -> ON audit/report. Never rewrites old provenance.

Uses the completed precision-native regression to bridge ONLY the reviewed
precision wiring source changes. This does not relax the general comparator.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gr00t.eval.sim.robomme.compare_long_memory_results import TASKS, paired_differences, paired_macro_bootstrap, mcnemar_exact
from run_scripts.robomme.compare_archive_retrieval_v17 import load_run
from run_scripts.robomme.compare_feature_precision_v19 import (
    CHECKPOINT, DEFAULT_NATIVE, DEFAULT_BASELINE, source_changes, validate_regression,
)
from run_scripts.robomme.eval_representation_v18 import (
    build_parser, validate_options, build_identity, validate_manifest_contract,
    verify_runtime_inputs, completed_read_diagnostics,
)
from run_scripts.robomme.baseline_reference_v18 import validate_reference
from run_scripts.robomme.baseline_reference_v10 import _scenario_context
from run_scripts.robomme.eval_long_memory_comparison import file_hash, resolve_repo_path
from run_scripts.robomme.feature_precision_v19 import feature_precision_contract
from gr00t.long_memory.safety_v5 import validate_output_scope

DEFAULT_OFF = "runs/eval/robomme/v19_native_read_off_val160_seed6"
PRIOR = ROOT / "runs/diagnostics/v19_precision_ab_20260920"


def read(path):
    return json.loads(Path(path).read_text())


def validate_pair(on, off):
    settings = dict(tasks=TASKS, n_episodes=10, dataset="val", seed=6,
        n_action_steps=16, max_episode_steps=1300, save_videos=False, device="cuda:0")
    for m, role in ((on, "memory"), (off, "memory-off")):
        validate_manifest_contract(m)
        if m["settings"] != settings or set(m["models"]) != {"baseline", role}:
            raise ValueError("Only fixed native VAL160 seed6 and one candidate role are allowed")
        model = m["models"][role]
        if (Path(model["memory_checkpoint"]).resolve() != ROOT / CHECKPOINT or model["step"] != 6072
                or model.get("writer_checkpoint") or model["write_policy"] != "fifo"):
            raise ValueError("Checkpoint/writer differs from the prescribed V19 prefix")
        if model.get("feature_precision", "native") != "native":
            raise ValueError("Both ON and OFF must be native, not cache-aligned")
        if "feature_precision" in model and model.get("feature_precision_rules") != feature_precision_contract("native"):
            raise ValueError("Native precision contract changed")
    if off["models"]["memory-off"].get("feature_precision") != "native":
        raise ValueError("OFF must explicitly record native precision")
    ignored = {"memory_off", "description", "feature_precision", "feature_precision_rules"}
    a = {k: v for k, v in on["models"]["memory"].items() if k not in ignored}
    b = {k: v for k, v in off["models"]["memory-off"].items() if k not in ignored}
    if a != b:
        raise ValueError("AE/checkpoint/short/storage differs beyond external READ")
    for key in ("settings", "base_file_sha256", "policy_package_versions", "benchmark", "server_python",
                "robomme_python", "selection_protocol", "control_contract", "allow_initialization_checkpoints"):
        if key not in on or on[key] != off.get(key):
            raise ValueError(f"Environment or execution protocol changed: {key}")
    if on["models"]["baseline"] != off["models"]["baseline"]:
        raise ValueError("Original native HAMLET changed")
    return source_changes(on["source_sha256"], off["source_sha256"])


def validate_policy_check(path, manifest):
    result = read(path)
    checks = ("passed", "expert_matches_checkpoint", "expert_unchanged", "read_off_keeps_fifo",
              "persistent_episode_rng_matches", "hamlet_cache_matches")
    if result.get("kind") != "v19_native_read_ablation_check" or any(result.get(k) is not True for k in checks):
        raise ValueError("Actual-policy READ/RNG/LoRA check missing or failed")
    if result["checkpoint_files_sha256"] != manifest["models"]["memory-off"]["checkpoint_files_sha256"]:
        raise ValueError("Policy check used different checkpoint weights")
    for name, digest in result["source_sha256"].items():
        if manifest["source_sha256"].get(name) != digest or file_hash(ROOT / name) != digest:
            raise ValueError("Policy source changed after path check")
    for name, digest in result["evidence_sha256"].items():
        evidence = (path.parent / name).resolve()
        if not evidence.is_relative_to(path.parent.resolve()) or file_hash(evidence) != digest:
            raise ValueError("Policy-check evidence changed")
    return {"path": str(path), "sha256": file_hash(path), **{k: result[k] for k in checks},
            "expert_tensor_count": result["expert_tensor_count"], "expert_tensor_sha256": result["expert_tensor_sha256"]}


def extra_runtime_evidence(root, role, run):
    """Check actual completed calls, not flags declared only in a manifest."""
    calls, max_tokens = 0, 0
    model = run["manifest"]["models"][role]
    cap = model["representation_config"]["capacity_events"] * model["representation_config"]["num_short_tokens"]
    nq = model["representation_config"]["num_short_tokens"]
    for task in TASKS:
        sessions, ends = {}, {}
        for line in (root / role / task / "memory_diagnostics.jsonl").read_text().splitlines():
            x = json.loads(line)
            if x.get("kind") == "policy_call":
                sessions.setdefault(x["session_id"], []).append(x)
            elif x.get("kind") == "episode_complete":
                row = run["rows"][role][task].get(x["episode_idx"])
                if row and row["episode_seed"] == x["episode_seed"] and row["success"] == x["success"]:
                    ends[x["episode_idx"]] = x["session_id"]
        if set(ends) != set(range(10)):
            raise ValueError("Missing completed runtime session")
        for sid in ends.values():
            for index, x in enumerate(sessions[sid], 1):
                info, mem = x["info"], x["info"]["long_memory"]
                if info.get("expert_adapted") is not True:
                    raise ValueError("Runtime did not retain adapted AE")
                if any(mem.get(k) != v for k, v in dict(observations_seen=index, write_attempts=index,
                        updates=index, keeps=0, memory_tokens=min(index*nq, cap)).items()):
                    raise ValueError("FIFO WRITE/session progression changed")
                calls += 1
                max_tokens = max(max_tokens, mem["memory_tokens"])
    return {"checked_calls": calls, "expert_adapted_every_call": True, "fifo_write_every_call": True,
            "max_bank_tokens": max_tokens, "capacity_tokens": cap}


def completed(root, role, manifest):
    status = read(root / "driver_status.json")
    if status.get("interrupted") is not False or status.get("failures") != []:
        raise ValueError("Run did not complete cleanly")
    lock = root / ".driver.lock"
    if lock.exists():
        with lock.open("rb") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
    run = load_run(root, (role,))
    artifacts = {n: file_hash(root / n) for n in ("comparison_manifest.json", "driver_status.json")}
    for task in TASKS:
        context = run["identities"][role][task]
        scenarios, digest = _scenario_context(manifest["benchmark"]["source_and_scenario_sha256"], "val", task)
        if (context.get("scenario_metadata_sha256") != digest or context.get("memory_window") != 4
                or context.get("demo_sampling") != "backward_aligned_full_history"
                or context.get("model_config_sha256") != manifest["base_file_sha256"]["config.json"]):
            raise ValueError("Scenario/observation identity differs")
        for eid, row in run["rows"][role][task].items():
            payload = json.dumps([6, task, eid], separators=(",", ":")).encode()
            seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") % (2**31)
            if row["episode_seed"] != seed or int(row["scenario_seed"]) != scenarios[eid]["seed"]:
                raise ValueError("Scenario or inference seed differs")
        for name in ("simulation_results.csv", "policy_manifest.json", "memory_diagnostics.jsonl"):
            rel = f"{role}/{task}/{name}"
            artifacts[rel] = file_hash(root / rel)
    diagnostics = completed_read_diagnostics(root, role, manifest)
    if not diagnostics["complete_evidence"]:
        raise ValueError("Incomplete runtime identity/READ evidence")
    extra = extra_runtime_evidence(root, role, run)
    return run, {"root": str(root), "files_sha256": artifacts}, {**diagnostics, **extra}


def summarize(baseline, off, on):
    groups, table, flips = [], {}, []
    for task in TASKS:
        a, b, base = off[task], on[task], baseline[task]
        d = paired_differences(a, b)
        paired_differences(base, a)
        paired_differences(base, b)
        groups.append(d)
        both_success = sum(a[e]["success"] and b[e]["success"] for e in a)
        table[task] = dict(baseline_successes=sum(v["success"] for v in base.values()),
            off_successes=sum(v["success"] for v in a.values()), on_successes=sum(v["success"] for v in b.values()),
            n=10, wins=d.count(1), losses=d.count(-1), both_success=both_success,
            both_failure=10-d.count(1)-d.count(-1)-both_success, delta=sum(d)/10)
        for eid in sorted(a):
            if a[eid]["success"] != b[eid]["success"]:
                flips.append(dict(task=task, episode=eid, episode_seed=a[eid]["episode_seed"],
                    scenario_seed=a[eid]["scenario_seed"], task_instruction=a[eid]["task_instruction"],
                    baseline_success=base[eid]["success"], off_success=a[eid]["success"], on_success=b[eid]["success"],
                    off_status=a[eid].get("status"), on_status=b[eid].get("status"),
                    off_steps=a[eid].get("steps"), on_steps=b[eid].get("steps")))
    values = [v for group in groups for v in group]
    totals = {k: sum(t[k] for t in table.values()) for k in
              ("baseline_successes", "off_successes", "on_successes", "wins", "losses", "both_success", "both_failure")}
    return {**totals, "n": 160, "delta": sum(values)/160,
        "ci95": paired_macro_bootstrap(groups, samples=10000, seed=190021),
        "bootstrap_samples": 10000, "bootstrap_seed": 190021,
        "mcnemar_exact_p": mcnemar_exact(totals["wins"], totals["losses"]), "tasks": table, "flips": flips}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("audit", "report"))
    p.add_argument("--on-run", default=DEFAULT_NATIVE)
    p.add_argument("--off-run", default=DEFAULT_OFF)
    p.add_argument("--regression-report", type=Path, default=PRIOR / "regression/completed.json")
    p.add_argument("--prior-reuse-audit", type=Path, default=PRIOR / "reuse_audit/audit.json")
    p.add_argument("--policy-check", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)
    out, onroot, offroot = map(resolve_repo_path, (Path(args.output_dir), Path(args.on_run), Path(args.off_run)))
    validate_output_scope(out, onroot, offroot, ROOT/DEFAULT_BASELINE, ROOT/"runs/long_memory", ROOT/"checkpoints", ROOT/"data")
    if out.exists():
        raise FileExistsError("Use a NEW output directory")
    onpath = onroot / "comparison_manifest.json"
    on = read(onpath)
    if args.mode == "audit":
        opts = build_parser().parse_args(["--checkpoint", CHECKPOINT, "--feature-precision", "native",
            "--models", "memory-off", "--baseline-reference", DEFAULT_BASELINE,
            "--tasks", "all", "--dataset", "val", "--n-episodes", "10", "--seed", "6", "--output-dir", str(offroot)])
        validate_options(opts)
        off = build_identity(opts)
    else:
        off = read(offroot / "comparison_manifest.json")
    changes = validate_pair(on, off)
    # Only adapt this helper's model-role lookup; never invent or write a changed
    # manifest/evaluation ID. Both real manifests were validated above.
    regression = validate_regression(args.regression_report, onpath,
        {"models": {"memory": off["models"]["memory-off"]}, "source_sha256": off["source_sha256"]})
    policy_check = validate_policy_check(Path(args.policy_check).resolve(), off)
    verify_runtime_inputs(off)
    baseline_root, _ = validate_reference(off["baseline_reference"], off)
    validate_reference(on["baseline_reference"], on)
    onrun, onfiles, ondiag = completed(onroot, "memory", on)
    prior = read(args.prior_reuse_audit)
    if not prior.get("native_reuse_eligible") or prior["native_evaluation_id"] != on["evaluation_id"]:
        raise ValueError("Prior precision reuse audit is not bound to this ON run")
    for name, digest in prior["native_artifacts"]["files_sha256"].items():
        if onfiles["files_sha256"].get(name) != digest:
            raise ValueError("Original native result changed since precision audit")
    record = dict(kind="v19_native_read_ablation", on_reuse_eligible=True,
        on_evaluation_id=on["evaluation_id"], off_evaluation_id=off["evaluation_id"],
        allowed_source_changes=changes, native_regression=regression, policy_check=policy_check,
        comparison_script_sha256=file_hash(Path(__file__)), on_artifacts=onfiles, on_runtime=ondiag,
        settings=off["settings"], prior_reuse_audit_sha256=file_hash(args.prior_reuse_audit),
        limitation="Historical metadata binds Python/scenarios/packages, not all 3D asset bytes or old GPU drivers.")
    lines = ["# V19 native external-memory READ ablation", "", "Existing native ON reuse audit: PASS."]
    if args.mode == "report":
        offrun, offfiles, offdiag = completed(offroot, "memory-off", off)
        baseline = load_run(baseline_root, ("baseline",))
        for task in TASKS:
            anchor = baseline["identities"]["baseline"][task]
            for run, role in ((onrun, "memory"), (offrun, "memory-off")):
                for key in ("scenario_metadata_sha256", "model_config_sha256", "memory_window", "demo_sampling"):
                    if run["identities"][role][task].get(key) != anchor.get(key):
                        raise ValueError("Paired baseline/ON/OFF observation context differs")
        result = summarize(baseline["rows"]["baseline"], offrun["rows"]["memory-off"], onrun["rows"]["memory"])
        record.update(result=result, off_artifacts=offfiles, off_runtime=offdiag,
                      baseline_root=str(baseline_root), baseline_evaluation_id=baseline["manifest"]["evaluation_id"])
        lo, hi = result["ci95"]
        lines += [f"HAMLET {result['baseline_successes']}/160; OFF {result['off_successes']}/160; ON {result['on_successes']}/160.",
            f"ON - OFF: {100*result['delta']:+.3f} pp; 95% paired bootstrap CI [{100*lo:+.3f}, {100*hi:+.3f}] pp.",
            f"OFF failure -> ON success: {result['wins']}; reverse: {result['losses']}; both success: {result['both_success']}; both failure: {result['both_failure']}.",
            f"McNemar exact p={result['mcnemar_exact_p']:.7g}.", "",
            "|Task|HAMLET|OFF|ON|Win/loss|Both success/failure|", "|---|---:|---:|---:|---:|---:|"]
        lines += [f"|{task}|{t['baseline_successes']}/10|{t['off_successes']}/10|{t['on_successes']}/10|{t['wins']}/{t['losses']}|{t['both_success']}/{t['both_failure']}|" for task,t in result["tasks"].items()]
        lines += ["", "OFF retains jointly trained AE LoRA and short path: NOT a separately trained AE-only model.",
            "ON > OFF does not imply improvement over original HAMLET; OFF > ON does not isolate the reader as the cause.",
            "Repeated-development VAL160, fixed tasks/seed: not confirmatory generalization evidence."]
    out.mkdir(parents=True)
    (out / "audit.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    (out / "off_manifest.json").write_text(json.dumps(off, indent=2)+"\n")
    if "result" in record:
        for name, rows in (("tasks.csv", [dict(task=k, **v) for k,v in result["tasks"].items()]),
                           ("flipped_scenarios.csv", result["flips"])):
            if rows:
                with (out / name).open("w", newline="") as stream:
                    writer=csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    text = "\n".join(lines)+"\n"
    (out / "report.md").write_text(text)
    print(text)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
