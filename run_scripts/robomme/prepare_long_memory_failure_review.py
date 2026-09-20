#!/usr/bin/env python3
"""Prepare a balanced manual failure-review panel from EXISTING RoboMME results.

Standard library only: no checkpoint, simulator, GPU, or training is loaded.
This is outcome-stratified evidence collection, NOT a representative benchmark
sample. Missing videos stay blank; rollout logs cannot establish visual causes.
All explanatory labels are initially blank and must be supplied by a human.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gr00t.eval.sim.robomme.compare_long_memory_results import (
    TASKS, paired_differences, read_results, validate_result_identity,
)


DEFINITION = "paired-outcome-stratified-manual-failure-review-v1"
DEFAULT_TASKS = ["BinFill", "VideoUnmask", "VideoPlaceButton", "MoveCube", "PatternLock"]
STRATA = ("both_fail", "memory_win", "memory_loss", "both_success")
ROLES = ("baseline", "reader", "memory", "fifo", "expert-only", "full", "no-old", "shuffled-old")
CATEGORIES = (
    "remembered_goal_or_order_error", "manipulation_or_control_error", "cue_not_observed",
    "trajectory_divergence", "mixed", "success", "unknown",
)
LABEL_COLUMNS = [
    "failure_category", "baseline_failure_category", "cue_start_frame", "cue_end_frame",
    "required_at_frame", "cue_observed", "correct_goal", "manipulation_failure", "review_notes",
]
FIXED_COLUMNS = [
    "task", "episode_id", "episode_seed", "scenario_seed", "task_instruction",
    "baseline_success", "memory_success", "paired_stratum", "baseline_status", "memory_status",
    "baseline_steps", "memory_steps", "baseline_video_path", "baseline_log_path",
    "memory_video_path", "memory_log_path",
]
COLUMNS = FIXED_COLUMNS + LABEL_COLUMNS


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stratum(left, right):
    return {(0, 0): "both_fail", (0, 1): "memory_win",
            (1, 0): "memory_loss", (1, 1): "both_success"}[left, right]


def _existing_path(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Recorded artifact path escapes evaluation root: {path}")
    return str(path) if path.is_file() else ""


def _video_path(row, directory, root):
    raw = row.get("video_path", "").strip()
    if not raw:
        return ""
    return _existing_path(directory / raw, root)


def _select(groups, n_per_task, seed, task):
    # Task-specific local RNG makes task reordering irrelevant. Round-robin
    # selection balances available outcome strata, redistributing empty quotas.
    domain = f"{DEFINITION}/{seed}/{task}"
    rng = random.Random(int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big"))
    pools = {name: sorted(groups[name]) for name in STRATA}
    for pool in pools.values():
        rng.shuffle(pool)
    order = list(STRATA)
    rng.shuffle(order)
    selected = []
    while len(selected) < n_per_task:
        added = False
        for name in order:
            if pools[name] and len(selected) < n_per_task:
                selected.append(pools[name].pop())
                added = True
        if not added:
            break
    return sorted(selected)


def prepare_review(eval_root, output_dir, *, tasks=None, n_per_task=8, seed=6,
                   baseline_role="baseline", memory_role="memory"):
    """Validate the saved paired run, then exclusively create a new review folder."""
    root, output = Path(eval_root).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if isinstance(n_per_task, bool) or not isinstance(n_per_task, int) or n_per_task < 1:
        raise ValueError("n_per_task must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if baseline_role not in ROLES or memory_role not in ROLES or baseline_role == memory_role:
        raise ValueError("baseline_role and memory_role must be distinct supported model roles")
    roles = {"baseline": baseline_role, "memory": memory_role}
    tasks = list(DEFAULT_TASKS if tasks is None else tasks)
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in TASKS for t in tasks):
        raise ValueError("tasks must contain unique supported RoboMME names")
    manifest_path = root / "comparison_manifest.json"
    comparison = json.loads(manifest_path.read_text(encoding="utf-8"))
    if comparison.get("trainer_variant") not in ("action_expert_v4", "action_expert_v4_diagnostic_v1"):
        raise ValueError("Expected an existing v4 or v4 diagnostic comparison")
    if not set(roles.values()) <= comparison["models"].keys():
        raise ValueError(f"Existing comparison must include selected roles: {roles}")
    if any(task not in comparison["settings"]["tasks"] for task in tasks):
        raise ValueError("Requested task is not in the saved comparison manifest")
    expected = comparison["settings"]["n_episodes"]
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise ValueError("Invalid saved expected episode count")
    hashes = {str(manifest_path): _sha256(manifest_path)}
    records, coverage = [], {}
    for task in tasks:
        rows, identities = {}, {}
        for role in ("baseline", "memory"):
            directory = root / roles[role] / task
            result_path = directory / "simulation_results.csv"
            rows[role] = read_results(result_path, expected=expected)
            if len(rows[role]) != expected:
                raise ValueError(f"{role}/{task}: need completed paired results ({len(rows[role])}/{expected})")
            # The native writer's statuses encode the simulator success flag.
            for row in rows[role].values():
                if row.get("status") in ("success", "fail", "timeout", "step_limit"):
                    if (row["status"] == "success") != bool(row["success"]):
                        raise ValueError(f"{role}/{task}: success flag disagrees with native status")
            identities[role] = validate_result_identity(root, roles[role], task, comparison)
            for path in (result_path, directory / "policy_manifest.json"):
                hashes[str(path)] = _sha256(path)
        for key in ("scenario_metadata_sha256", "model_config_sha256", "memory_window", "demo_sampling"):
            if identities["baseline"].get(key) != identities["memory"].get(key):
                raise ValueError(f"{task}: unmatched saved {key} across policies")
        paired_differences(rows["baseline"], rows["memory"])
        groups = {name: [] for name in STRATA}
        for episode in sorted(rows["baseline"]):
            groups[_stratum(rows["baseline"][episode]["success"], rows["memory"][episode]["success"])].append(episode)
        selected = _select(groups, n_per_task, seed, task)
        selection_counts = Counter()
        for episode in selected:
            left, right = rows["baseline"][episode], rows["memory"][episode]
            stratum = _stratum(left["success"], right["success"])
            selection_counts[stratum] += 1
            record = {
                "task": task, "episode_id": str(episode), "episode_seed": str(left["episode_seed"]),
                "scenario_seed": str(left.get("scenario_seed", "")),
                "task_instruction": left.get("task_instruction", ""),
                "baseline_success": str(left["success"]), "memory_success": str(right["success"]),
                "paired_stratum": stratum, "baseline_status": left.get("status", ""),
                "memory_status": right.get("status", ""), "baseline_steps": left.get("steps", ""),
                "memory_steps": right.get("steps", ""),
            }
            for role, row in (("baseline", left), ("memory", right)):
                directory = root / roles[role] / task
                record[f"{role}_video_path"] = _video_path(row, directory, root)
                record[f"{role}_log_path"] = _existing_path(directory / "rollout.log", root)
            record.update({name: "" for name in LABEL_COLUMNS})
            records.append(record)
        coverage[task] = {
            "paired_available": expected, "selected": len(selected),
            "available_by_stratum": {name: len(groups[name]) for name in STRATA},
            "selected_by_stratum": {name: selection_counts[name] for name in STRATA},
        }
    review_manifest = {
        "definition": DEFINITION, "evaluation_id": comparison["evaluation_id"],
        "source_trainer_variant": comparison["trainer_variant"], "roles": roles,
        "eval_root": str(root), "seed": seed, "n_per_task": n_per_task, "tasks": tasks,
        "selection": "deterministic outcome-stratified round-robin; NOT frequency-representative",
        "input_sha256": hashes, "coverage": coverage,
        "immutable_columns": FIXED_COLUMNS, "label_columns": LABEL_COLUMNS,
        "selected_original_values": [{name: row[name] for name in FIXED_COLUMNS} for row in records],
        "missing_video_counts": {role: sum(not row[f"{role}_video_path"] for row in records)
                                 for role in ("baseline", "memory")},
        "failure_categories": list(CATEGORIES),
        "label_scope": (f"CSV memory_* and unprefixed manual fields describe role={memory_role}; "
                        f"CSV baseline_* and baseline_failure_category describe role={baseline_role}."),
    }
    # Verify read-only sources did not change during selection before writing.
    if any(_sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("Source evaluation changed while preparing review; retry after it finishes")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "failure_review.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    with (output / "review_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(review_manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "README.md").open("x", encoding="utf-8") as handle:
        handle.write(_review_instructions(baseline_role, memory_role))
    return review_manifest


def _review_instructions(baseline_role="baseline", memory_role="memory"):
    return ("# RoboMME 수동 실패 원인 검토\n\n"
            f"CSV의 `baseline_*`는 실제 `{baseline_role}` 역할을, `memory_*`는 실제 `{memory_role}` 역할을 가리킵니다.\n"
            "실제 역할 매핑은 `review_manifest.json`의 `roles`에도 기록했습니다.\n" + """

`failure_review.csv`의 label 열만 편집하세요. task/episode/outcome/경로는 그대로 둡니다.
전체 평가에서 결과별로 균형 있게 뽑았으므로, 이 표의 성공률이나 원인 비율을 전체 성능으로 해석하면 안 됩니다.
`review_manifest.json`에 전체/선택 strata 개수, seed, 입력 SHA256을 보존했습니다.

- `both_fail`: 둘 다 실패. `memory_win`: memory만 성공.
- `memory_loss`: baseline만 성공. `both_success`: 둘 다 성공.
- `failure_category`와 접두사가 없는 나머지 label은 memory rollout을 설명합니다.
- `baseline_failure_category`는 baseline rollout을 별도로 분류합니다.
- 영상 경로가 비어 있으면 저장 영상이 없습니다. 로그만 보고 기억/조작 원인을 추정하지 말고 `unknown` 또는 공란으로 남기세요.
- 로그는 episode별 파일이 아니라 task 전체 파일입니다. `episode_id`에 해당하는 `ep=N`/session을 찾으세요.

## 분류 값 (자동 판정 아님)

- `remembered_goal_or_order_error`: 목표/색/횟수/순서를 잘못 선택한 행동이 관찰됨. 내부 기억 실패로 단정하지 마세요.
- `manipulation_or_control_error`: 올바른 목표에 접근했으나 집기/접촉/정렬 등 제어 실패가 관찰됨.
- `cue_not_observed`: 필요한 단서가 실제 관측 영상에 제시되지 않았다는 증거가 있음.
- `trajectory_divergence`: 이른 행동 오류 뒤 상태가 달라져 실패함. 처음 달라진 시점을 메모하세요.
- `mixed`: 여러 요인이 섞여 우선 원인을 분리하기 어려움.
- `success`: 이 rollout의 성공을 검토했음. 성공 flag만으로는 자동 기입하지 않습니다.
- `unknown`: 근거 부족으로 분류할 수 없음.

`cue_observed`, `correct_goal`, `manipulation_failure`는 `yes` / `no` / `unknown` 또는 공란입니다.
`cue_start_frame`, `cue_end_frame`, `required_at_frame`은 원본 simulator frame 번호이며,
정확히 대응할 수 있을 때만 음이 아닌 정수를 씁니다. action chunk/저장 영상 프레임 번호를 그대로 쓰지 마세요.
`review_notes`에는 관찰 근거와 모호한 점을 남기세요. 단순 attention 크기를 정답 근거로 쓰지 않습니다.

검토 후 (원본 파일 쓰기 없이 stdout 요약):

```bash
python run_scripts/robomme/prepare_long_memory_failure_review.py \\
  --report-only --review-csv /absolute/path/to/failure_review.csv
```

이 표는 병목 가설을 고르는 도구입니다. 장기기억 사용의 인과적 증명은 동일 AE의 no-old 등의 개입 실험으로 따로 검사해야 합니다.
""")


def summarize_review(review_csv):
    """Validate human annotations against their immutable panel; do not write."""
    path = Path(review_csv).resolve()
    manifest = json.loads((path.parent / "review_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("definition") != DEFINITION:
        raise ValueError("Unexpected failure review manifest definition")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Review CSV columns differ from the generated panel")
        rows = list(reader)
    originals = {(row["task"], row["episode_id"]): row for row in manifest["selected_original_values"]}
    seen, category_counts, baseline_counts, annotation_counts = set(), Counter(), Counter(), Counter()
    per_task = {}
    for row in rows:
        key = (row["task"], row["episode_id"])
        if key not in originals or key in seen:
            raise ValueError(f"Unexpected or duplicate review episode: {key}")
        seen.add(key)
        if any(row[name] != originals[key][name] for name in FIXED_COLUMNS):
            raise ValueError(f"Immutable review values changed for {key}; edit label columns only")
        for name in LABEL_COLUMNS:
            if row[name] is None:
                raise ValueError(f"Missing CSV label cell for {key}: {name}")
            row[name] = row[name].strip()
        for name, counts in (("failure_category", category_counts), ("baseline_failure_category", baseline_counts)):
            if row[name] and row[name] not in CATEGORIES:
                raise ValueError(f"Invalid {name} for {key}: {row[name]!r}")
            counts[row[name] or "unreviewed"] += 1
            success_column = "memory_success" if name == "failure_category" else "baseline_success"
            if row[name] == "success" and row[success_column] != "1":
                raise ValueError(f"Success annotation conflicts with saved outcome for {key}: {name}")
        for name in ("cue_observed", "correct_goal", "manipulation_failure"):
            if row[name] not in ("", "yes", "no", "unknown"):
                raise ValueError(f"{key}: {name} must be yes/no/unknown/blank")
        frames = {}
        for name in ("cue_start_frame", "cue_end_frame", "required_at_frame"):
            if row[name]:
                if not row[name].isdigit():
                    raise ValueError(f"{key}: {name} must be a nonnegative integer or blank")
                frames[name] = int(row[name])
        if "cue_start_frame" in frames and "cue_end_frame" in frames and frames["cue_start_frame"] > frames["cue_end_frame"]:
            raise ValueError(f"{key}: cue_start_frame is after cue_end_frame")
        annotated = any(row[name] for name in LABEL_COLUMNS)
        annotation_counts["any_label" if annotated else "no_labels"] += 1
        task = per_task.setdefault(row["task"], {"selected": 0, "any_label": 0, "memory_categories": Counter()})
        task["selected"] += 1
        task["any_label"] += int(annotated)
        task["memory_categories"][row["failure_category"] or "unreviewed"] += 1
    if seen != set(originals):
        raise ValueError("Review CSV dropped selected episodes")
    return {
        "definition": DEFINITION, "review_csv": str(path), "review_csv_sha256": _sha256(path),
        "roles": manifest.get("roles", {"baseline": "baseline", "memory": "memory"}),
        "selected": len(rows), "annotation_counts": dict(annotation_counts),
        "memory_categories": dict(category_counts), "baseline_categories": dict(baseline_counts),
        "per_task": per_task,
        "warning": "Human labels on an outcome-balanced panel; counts are NOT population failure frequencies or causal proof.",
        "input_fingerprints": "Source SHA256 retained in review_manifest.json; report-only does not require original run to remain mounted.",
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("runs/eval/robomme/v4_best_all_n50_seed6"))
    parser.add_argument("--output-dir", type=Path, help="New folder only; never overwrites a prior panel")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=DEFAULT_TASKS)
    parser.add_argument("--n-per-task", type=int, default=8)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--baseline-role", choices=ROLES, default="baseline", help="Comparator stored under this role; CSV keeps baseline_* names")
    parser.add_argument("--memory-role", choices=ROLES, default="memory", help="Target stored under this role; use full for new diagnostic evaluations")
    parser.add_argument("--report-only", action="store_true", help="Validate annotations and print JSON; no files are changed")
    parser.add_argument("--review-csv", type=Path, help="Annotated failure_review.csv for --report-only")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.report_only:
        path = args.review_csv or (args.output_dir / "failure_review.csv" if args.output_dir else None)
        if path is None:
            parser.error("--report-only requires --review-csv or --output-dir")
        print(json.dumps(summarize_review(path), indent=2, ensure_ascii=False, allow_nan=False))
        return
    if args.review_csv:
        parser.error("--review-csv requires --report-only")
    if args.output_dir is None:
        parser.error("--output-dir is required when creating a new panel")
    manifest = prepare_review(args.eval_root, args.output_dir, tasks=args.tasks,
                              n_per_task=args.n_per_task, seed=args.seed,
                              baseline_role=args.baseline_role, memory_role=args.memory_role)
    print(f"[review] Saved {sum(c['selected'] for c in manifest['coverage'].values())} paired episodes: "
          f"{args.output_dir.resolve() / 'failure_review.csv'}")
    print("[review] Outcome-balanced panel, NOT population frequencies. All cause labels are blank.")
    print(f"[review] Missing saved videos: {manifest['missing_video_counts']}; blank video cells are intentional.")
    print(f"[review] Instructions: {args.output_dir.resolve() / 'README.md'}")


if __name__ == "__main__":
    main()
