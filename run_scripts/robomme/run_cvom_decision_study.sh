#!/usr/bin/env bash
# Explicit independent phases: no "all" and no automatic training/rollout.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
phase="${1:-status}"
if (( $# )); then shift; fi
case "$phase" in
  noise8-preflight|noise8)
    mode=preflight
    extra=()
    if [[ "$phase" == noise8 ]]; then
      mode=run
      if [[ -f runs/long_memory/cvom_budget_val16_noise8_v1/protocol.json ]]; then extra+=(--resume); fi
    fi
    exec .venv/bin/python -u run_scripts/robomme/cvom_budget_probe.py "$mode" \
      --contexts 16 --noise-samples 8 --output-dir runs/long_memory/cvom_budget_val16_noise8_v1 "${extra[@]}" "$@" ;;
  noise-compare)
    exec .venv/bin/python run_scripts/robomme/compare_cvom_budget_noise.py \
      --output-dir "${CVOM_COMPARISON_OUTPUT:-runs/long_memory/cvom_noise2_vs8_20260922_v2}" "$@" ;;
  storage-audit)
    exec .venv/bin/python run_scripts/robomme/audit_echo_storage.py \
      --output-dir "${CVOM_STORAGE_OUTPUT:-runs/long_memory/echo_storage_audit_20260922}" "$@" ;;
  writer2|writer8)
    input=runs/long_memory/cvom_budget_val16_v1
    output=runs/long_memory/cvom_writer_noise2_v1
    if [[ "$phase" == writer8 ]]; then
      input=runs/long_memory/cvom_budget_val16_noise8_v1
      output=runs/long_memory/cvom_writer_noise8_v1
    fi
    exec .venv/bin/python run_scripts/robomme/cvom_writer_decision_metrics.py --run-dir "$input" \
      --output-dir "${CVOM_WRITER_OUTPUT:-$output}" "$@" ;;
  minfill-verify)
    exec .venv/bin/python -u run_scripts/robomme/verify_echo_min_fill.py \
      --checkpoint runs/long_memory/echo_cvom_full_v1/refresh/checkpoint-001000 \
      --output-dir "${CVOM_REGRESSION_OUTPUT:-runs/long_memory/echo_min_fill_regression_20260922}" "$@" ;;
  minfill-preflight|minfill-eval|minfill-report)
    extra=()
    if [[ "$phase" == minfill-preflight ]]; then extra+=(--preflight-only); fi
    if [[ "$phase" == minfill-report ]]; then extra+=(--report-only); fi
    exec .venv/bin/python -u run_scripts/robomme/eval_echo_min_fill.py \
      --output-dir "${CVOM_MINFILL_OUTPUT:-runs/eval/robomme/echo_min_fill32_val160_seed6}" "${extra[@]}" "$@" ;;
  storage-compare)
    exec .venv/bin/python run_scripts/robomme/audit_echo_storage.py \
      --source learned4=runs/eval/robomme/echo_cvom_full_v1_val160_seed6/memory \
      --source learned32="${CVOM_MINFILL_OUTPUT:-runs/eval/robomme/echo_min_fill32_val160_seed6}/memory" \
      --output-dir "${CVOM_STORAGE_OUTPUT:-runs/long_memory/echo_storage_minfill4_vs32_v1}" "$@" ;;
  status|teacher-review)
    exec .venv/bin/python run_scripts/robomme/review_cvom_decision_study.py \
      --comparison "${CVOM_COMPARISON_OUTPUT:-runs/long_memory/cvom_noise2_vs8_20260922_v2}/comparison.json" "$@" ;;
  all|train|writer-train|adapt)
    echo "STOP: 전체 재학습/자동 후속 학습은 제공하지 않습니다. teacher-review 근거를 먼저 확인하세요." >&2
    exit 2 ;;
  *) echo "Use: status | noise8-preflight | noise8 | noise-compare | storage-audit | writer2 | writer8 | minfill-verify | minfill-preflight | minfill-eval | minfill-report | storage-compare | teacher-review" >&2; exit 2 ;;
esac
