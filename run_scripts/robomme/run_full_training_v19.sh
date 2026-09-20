#!/usr/bin/env bash
# Explicit, full-TRAIN-coverage experiments; no jobs start without a phase.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
R19_PYTHON="${R19_PYTHON:-.venv/bin/python}"
R19_RUN_DIR="${R19_RUN_DIR:-runs/long_memory/v19_fullcoverage_v1}"
R19_EVAL_DIR="${R19_EVAL_DIR:-runs/eval/robomme/v19_fullcoverage_v1}"
R19_CACHE="${R19_CACHE:-runs/long_memory/cache_full1600_v1}"
R19_EPOCHS="${R19_EPOCHS:-1}"
R19_BASELINE_REFERENCE="${R19_BASELINE_REFERENCE:-runs/eval/robomme/archive_read_best1250_val_n10_seed6}"
if [[ ! "$R19_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "R19_EPOCHS must be a positive integer" >&2; exit 2
fi

train_arm() {
  local arm="$1" tail_weight="$2"
  shift 2
  "$R19_PYTHON" run_scripts/robomme/train_full_memory_v19.py \
    --cache-dir "$R19_CACHE" --output-dir "$R19_RUN_DIR/$arm" \
    --epochs "$R19_EPOCHS" --query-batch-size 4 --seed 9191 \
    --tail-weight "$tail_weight" --task-weighting macro \
    --memory-learning-rate 1e-4 --expert-learning-rate 1e-5 \
    --val-per-task 4 --val-noise-samples 2 \
    --eval-steps 1000 --save-steps 1000 --plot-steps 1000 \
    --activation-checkpointing "$@"
}

final_checkpoint() {
  # Refuses paused smoke runs, best-metric selection, and incomplete epochs.
  "$R19_PYTHON" run_scripts/robomme/compare_full_training_v19.py \
    --checkpoint-run "$R19_RUN_DIR/$1"
}

eval_arm() {
  local arm="$1" checkpoint
  shift
  checkpoint="$(final_checkpoint "$arm")"
  "$R19_PYTHON" run_scripts/robomme/eval_representation_v18.py \
    --checkpoint "$checkpoint" --models memory \
    --baseline-reference "$R19_BASELINE_REFERENCE" --tasks all --dataset val \
    --n-episodes 10 --seed 6 --output-dir "$R19_EVAL_DIR/$arm" "$@"
}

selected_arm() {
  # The default is a prespecified candidate, NOT an automatically chosen winner.
  R19_SELECTED_ARM="${R19_SELECTED_ARM:-prefix_full}"
  case "$R19_SELECTED_ARM" in
    control_full|prefix_full) ;;
    *) echo "Set R19_SELECTED_ARM=control_full or prefix_full" >&2; exit 2 ;;
  esac
}

case "${1:-help}" in
  preflight)
    train_arm control_full 1 --preflight-only
    train_arm prefix_full .25 --preflight-only
    ;;
  smoke)
    # Separate output names; these paused 2-step runs are NOT full-epoch results.
    train_arm smoke_control 1 --stop-after-steps 2 --val-per-task 1 --val-noise-samples 1 --eval-steps 2 --save-steps 2 --plot-steps 2
    train_arm smoke_prefix .25 --stop-after-steps 2 --val-per-task 1 --val-noise-samples 1 --eval-steps 2 --save-steps 2 --plot-steps 2
    ;;
  train-control) train_arm control_full 1 ;;
  train-prefix) train_arm prefix_full .25 ;;
  train)
    train_arm control_full 1
    train_arm prefix_full .25
    ;;
  eval-preflight)
    eval_arm control_full --preflight-only
    eval_arm prefix_full --preflight-only
    ;;
  eval-control) eval_arm control_full ;;
  eval-prefix) eval_arm prefix_full ;;
  eval)
    eval_arm control_full
    eval_arm prefix_full
    ;;
  compare)
    "$R19_PYTHON" run_scripts/robomme/compare_full_training_v19.py \
      --control-run "$R19_EVAL_DIR/control_full" --prefix-run "$R19_EVAL_DIR/prefix_full"
    ;;
  read-off)
    selected_arm
    echo "[v19] READ-off for $R19_SELECTED_ARM; this is not automatic winner selection."
    eval_arm "$R19_SELECTED_ARM" --models memory-off --output-dir "$R19_EVAL_DIR/${R19_SELECTED_ARM}_off"
    ;;
  compare-off)
    selected_arm
    # Existing strict comparison retains the identical reader/AE/checkpoint.
    "$R19_PYTHON" run_scripts/robomme/compare_representation_v18.py \
      --left-run "$R19_EVAL_DIR/${R19_SELECTED_ARM}_off" \
      --right-run "$R19_EVAL_DIR/$R19_SELECTED_ARM" --factor read
    ;;
  monitor)
    "$R19_PYTHON" run_scripts/robomme/monitor_long_memory.py \
      --logdir "$R19_RUN_DIR" --port "${R19_TB_PORT:-6007}"
    ;;
  final-test)
    if [[ -z "${R19_SELECTED_ARM:-}" ]]; then
      echo "Explicitly set R19_SELECTED_ARM after VAL and SAME-model READ-off; no automatic TEST selection." >&2
      exit 2
    fi
    selected_arm
    checkpoint="$(final_checkpoint "$R19_SELECTED_ARM")"
    extra=()
    if [[ -n "${R19_TEST_BASELINE_REFERENCE:-}" ]]; then
      extra+=(--baseline-reference "$R19_TEST_BASELINE_REFERENCE")
    fi
    "$R19_PYTHON" run_scripts/robomme/eval_representation_v18.py \
      --checkpoint "$checkpoint" --models baseline memory --tasks all --dataset test \
      --n-episodes 50 --seed 6 --output-dir "$R19_EVAL_DIR/final_test_$R19_SELECTED_ARM" "${extra[@]}"
    ;;
  help|plan)
    echo "V19: no training/eval launched by default; all phases are explicit."
    echo "preflight -> smoke -> train -> eval-preflight -> eval -> compare"
    echo "Or train-control/train-prefix and eval-control/eval-prefix separately."
    echo "Every TRAIN query exactly once per epoch; default R19_EPOCHS=1, batch=4."
    echo "Same V18 A architecture, initialization, full-query plan and task-macro weights."
    echo "Only loss tail weight differs: control_full=1, prefix_full=0.25; first16=1."
    echo "VAL all16 x10=160 per candidate, 320 new rollouts; unchanged baseline reused if compatible."
    echo "Checkpoints are fixed final epochs from last_checkpoint.json + complete status, not best loss."
    echo "read-off -> compare-off defaults to prefix_full; choose R19_SELECTED_ARM explicitly for another arm."
    echo "monitor: TensorBoard port6007. Final-test needs explicit R19_SELECTED_ARM; all16 TEST x50."
    echo "Existing outputs/checkpoints are never overwritten; use new R19_RUN_DIR/R19_EVAL_DIR for a new experiment."
    echo "All cached TRAIN endpoints are used; cache resolution/caps and held-out VAL remain unchanged."
    ;;
  *) echo "Unknown phase; use help" >&2; exit 2 ;;
esac
