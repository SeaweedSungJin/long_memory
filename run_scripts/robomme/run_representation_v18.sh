#!/usr/bin/env bash
# Explicit ordered phases. Opening/running without a phase does NOT launch jobs.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
R18_PYTHON="${R18_PYTHON:-.venv/bin/python}"
R18_RUN_DIR="${R18_RUN_DIR:-runs/long_memory/v18_representation_v1}"
R18_EVAL_DIR="${R18_EVAL_DIR:-runs/eval/robomme/v18_representation_v1}"
R18_CACHE="${R18_CACHE:-runs/long_memory/cache_full1600_v1}"
R18_STEPS="${R18_STEPS:-1000}"
R18_WRITER_STEPS="${R18_WRITER_STEPS:-1000}"
R18_BASELINE_REFERENCE="${R18_BASELINE_REFERENCE:-runs/eval/robomme/archive_read_best1250_val_n10_seed6}"
for value in "$R18_STEPS" "$R18_WRITER_STEPS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then echo "Step counts must be positive integers" >&2; exit 2; fi
done
printf -v R18_CHECKPOINT 'checkpoint-%06d' "$R18_STEPS"
printf -v R18_WRITER_CHECKPOINT 'checkpoint-%06d' "$R18_WRITER_STEPS"

train_arm() {
  local arm="$1" representation="$2" gate="$3"
  shift 3
  "$R18_PYTHON" run_scripts/robomme/train_representation_v18.py \
    --cache-dir "$R18_CACHE" --output-dir "$R18_RUN_DIR/$arm" \
    --representation "$representation" --gate "$gate" --capacity-events 32 \
    --max-steps "$R18_STEPS" --query-batch-size 4 --seed 9181 \
    --memory-learning-rate 1e-4 --short-learning-rate 1e-5 --expert-learning-rate 1e-5 \
    --val-samples 32 --val-noise-samples 1 --eval-steps 250 --save-steps 250 \
    --plot-steps 250 --activation-checkpointing "$@"
}

eval_arm() {
  local arm="$1"
  shift
  "$R18_PYTHON" run_scripts/robomme/eval_representation_v18.py \
    --checkpoint "$R18_RUN_DIR/$arm/$R18_CHECKPOINT" --models memory \
    --baseline-reference "$R18_BASELINE_REFERENCE" --tasks all --dataset val \
    --n-episodes 10 --seed 6 --output-dir "$R18_EVAL_DIR/$arm" "$@"
}

select_arm() {
  # Selection must come from the user, never from repeatedly inspecting TEST.
  if [[ -z "${R18_SELECTED_ARM:-}" ]]; then
    echo "Set R18_SELECTED_ARM=A_short, B_adapted, C_moment, or gate_mlp after examining VAL." >&2
    exit 2
  fi
  case "$R18_SELECTED_ARM" in
    A_short) selected_rep=short ;;
    B_adapted) selected_rep=adapted_short ;;
    C_moment) selected_rep=moment ;;
    gate_mlp) selected_rep="${R18_GATE_REPRESENTATION:?Set R18_GATE_REPRESENTATION to the representation used for gate_mlp}" ;;
    *) echo "Unknown R18_SELECTED_ARM" >&2; exit 2 ;;
  esac
}

case "${1:-help}" in
  preflight)
    train_arm A_short short linear --preflight-only
    train_arm B_adapted adapted_short linear --preflight-only
    train_arm C_moment moment linear --preflight-only
    ;;
  audit)
    "$R18_PYTHON" run_scripts/robomme/audit_representation_v18.py \
      --cache-dir "$R18_CACHE" --device "${R18_AUDIT_DEVICE:-cpu}"
    ;;
  smoke)
    # Isolated names: two updates are only an implementation smoke test.
    train_arm smoke_A short linear --max-steps 2 --val-samples 2 --eval-steps 2 --save-steps 2 --plot-steps 2
    train_arm smoke_B adapted_short linear --max-steps 2 --val-samples 2 --eval-steps 2 --save-steps 2 --plot-steps 2
    ;;
  baseline)
    "$R18_PYTHON" run_scripts/robomme/eval_representation_v18.py --models baseline \
      --tasks all --dataset val --n-episodes 10 --seed 6 --output-dir "$R18_EVAL_DIR/baseline"
    ;;
  train-a) train_arm A_short short linear ;;
  train-b) train_arm B_adapted adapted_short linear ;;
  train-ab)
    train_arm A_short short linear
    train_arm B_adapted adapted_short linear
    ;;
  eval-preflight)
    eval_arm A_short --preflight-only
    eval_arm B_adapted --preflight-only
    ;;
  eval-a) eval_arm A_short ;;
  eval-b) eval_arm B_adapted ;;
  eval-ab)
    eval_arm A_short
    eval_arm B_adapted
    ;;
  compare-ab)
    "$R18_PYTHON" run_scripts/robomme/compare_representation_v18.py \
      --left-run "$R18_EVAL_DIR/A_short" --right-run "$R18_EVAL_DIR/B_adapted" --factor representation
    ;;
  train-c) train_arm C_moment moment linear ;;
  eval-c) eval_arm C_moment ;;
  compare-ac)
    "$R18_PYTHON" run_scripts/robomme/compare_representation_v18.py \
      --left-run "$R18_EVAL_DIR/A_short" --right-run "$R18_EVAL_DIR/C_moment" --factor representation
    ;;
  train-mlp)
    select_arm
    if [[ "$R18_SELECTED_ARM" == gate_mlp ]]; then echo "Select the LINEAR reference arm first" >&2; exit 2; fi
    train_arm gate_mlp "$selected_rep" mlp
    ;;
  eval-mlp) eval_arm gate_mlp ;;
  compare-gate)
    select_arm
    if [[ "$R18_SELECTED_ARM" == gate_mlp ]]; then echo "Select the LINEAR reference arm" >&2; exit 2; fi
    "$R18_PYTHON" run_scripts/robomme/compare_representation_v18.py \
      --left-run "$R18_EVAL_DIR/$R18_SELECTED_ARM" --right-run "$R18_EVAL_DIR/gate_mlp" --factor gate
    ;;
  read-off)
    select_arm
    # Does not rerun the winner's READ-on; only its own matched OFF plus reused baseline.
    eval_arm "$R18_SELECTED_ARM" --models memory-off --output-dir "$R18_EVAL_DIR/${R18_SELECTED_ARM}_off"
    ;;
  compare-off)
    select_arm
    "$R18_PYTHON" run_scripts/robomme/compare_representation_v18.py \
      --left-run "$R18_EVAL_DIR/${R18_SELECTED_ARM}_off" --right-run "$R18_EVAL_DIR/$R18_SELECTED_ARM" --factor read
    ;;
  writer-preflight|train-writer)
    select_arm
    extra=()
    if [[ "$1" == writer-preflight ]]; then extra=(--preflight-only); fi
    "$R18_PYTHON" run_scripts/robomme/train_storage_cvom_v18.py \
      --reader-checkpoint "$R18_RUN_DIR/$R18_SELECTED_ARM/$R18_CHECKPOINT" \
      --cache-dir "$R18_CACHE" --output-dir "$R18_RUN_DIR/writer_$R18_SELECTED_ARM" \
      --max-steps "$R18_WRITER_STEPS" --storage-contexts 128 --val-storage-contexts 32 \
      --future-samples 2 --noise-samples 2 "${extra[@]}"
    ;;
  eval-writer)
    select_arm
    "$R18_PYTHON" run_scripts/robomme/eval_representation_v18.py \
      --checkpoint "$R18_RUN_DIR/$R18_SELECTED_ARM/$R18_CHECKPOINT" \
      --writer-checkpoint "$R18_RUN_DIR/writer_$R18_SELECTED_ARM/$R18_WRITER_CHECKPOINT" \
      --models memory fifo --baseline-reference "$R18_BASELINE_REFERENCE" \
      --tasks all --dataset val --n-episodes 10 --seed 6 --output-dir "$R18_EVAL_DIR/writer_$R18_SELECTED_ARM"
    ;;
  final-test)
    select_arm
    # Only an explicitly selected frozen candidate is tested. No automatic TEST search.
    extra=()
    if [[ -n "${R18_FINAL_WRITER:-}" ]]; then extra+=(--writer-checkpoint "$R18_FINAL_WRITER"); fi
    if [[ -n "${R18_TEST_BASELINE_REFERENCE:-}" ]]; then extra+=(--baseline-reference "$R18_TEST_BASELINE_REFERENCE"); fi
    "$R18_PYTHON" run_scripts/robomme/eval_representation_v18.py \
      --checkpoint "$R18_RUN_DIR/$R18_SELECTED_ARM/$R18_CHECKPOINT" --models baseline memory \
      --tasks all --dataset test --n-episodes 50 --seed 6 \
      --output-dir "$R18_EVAL_DIR/final_test_$R18_SELECTED_ARM" "${extra[@]}"
    ;;
  monitor)
    "$R18_PYTHON" run_scripts/robomme/monitor_long_memory.py --logdir "$R18_RUN_DIR" --port "${R18_TB_PORT:-6007}"
    ;;
  help|plan)
    echo "V18: explicit phases; no training/eval launched by default."
    echo "1 preflight -> audit -> smoke"
    echo "2 train-ab -> eval-preflight -> eval-ab -> compare-ab"
    echo "3 train-c -> eval-c -> compare-ac (A reused; do not compare B vs C as source-only)"
    echo "4 Set R18_SELECTED_ARM; train-mlp -> eval-mlp -> compare-gate"
    echo "5 Selected same-model read-off -> compare-off; writer-preflight -> train-writer -> eval-writer"
    echo "6 Freeze the selected design -> final-test (long; no automatic selection)"
    echo "monitor: live TensorBoard; default port6007"
    echo "Default:1000updates/4000queries per actor; same originalbase, plan, seed, capacity32."
    echo "VAL:16tasks x10=160 episodes PER candidate. A+B=320 new; C=160 more."
    echo "Existing baseline reused only after exact protocol/source/environment validation."
    echo "If reference is incompatible: baseline phase, then set R18_BASELINE_REFERENCE=$R18_EVAL_DIR/baseline."
    ;;
  *) echo "Unknown phase; use help" >&2; exit 2 ;;
esac
