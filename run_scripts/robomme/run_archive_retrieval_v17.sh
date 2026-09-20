#!/usr/bin/env bash
# Explicit phases only. No automatic training/evaluation from merely opening it.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
R17_PYTHON="${R17_PYTHON:-.venv/bin/python}"
R17_RUN_DIR="${R17_RUN_DIR:-runs/long_memory/v17_retrieval_pilot}"
R17_EVAL_DIR="${R17_EVAL_DIR:-runs/eval/robomme/v17_retrieval_pilot}"
R17_STEPS="${R17_STEPS:-512}"
R17_WEIGHT="${R17_WEIGHT:-0.01}"
if [[ ! "$R17_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "R17_STEPS must be a positive integer" >&2
    exit 2
fi
printf -v R17_CHECKPOINT 'checkpoint-%06d' "$R17_STEPS"
common=(--cache-dir runs/long_memory/cache_full1600_v1
  --targets runs/long_memory/segment_targets_v15_20260917/manifest.json
  --init-checkpoint runs/long_memory/v7_archive_full_v1/checkpoint-001250
  --max-steps "$R17_STEPS" --query-batch-size 4 --seed 9171
  --memory-learning-rate 1e-5 --expert-learning-rate 1e-6
  --val-samples 32 --val-noise-samples 1 --eval-steps 128 --save-steps 128
  --activation-checkpointing)

train_arm() {
    local arm="$1" weight="$2"
    shift 2
    "$R17_PYTHON" run_scripts/robomme/train_archive_retrieval_v17.py "${common[@]}" \
      --output-dir "$R17_RUN_DIR/$arm" --retrieval-weight "$weight" "$@"
}

case "${1:-help}" in
  preflight)
    train_arm control 0 --preflight-only
    train_arm guided "$R17_WEIGHT" --preflight-only
    ;;
  smoke)
    # Separate smoke namespace; never mistaken for the final fixed-horizon arm.
    train_arm smoke "$R17_WEIGHT" --max-steps 2 --eval-steps 2 --save-steps 2
    ;;
  train-control) train_arm control 0 ;;
  train-guided) train_arm guided "$R17_WEIGHT" ;;
  train)
    train_arm control 0
    train_arm guided "$R17_WEIGHT"
    ;;
  val|test|eval-preflight)
    phase="$1"
    dataset=val
    episodes=10
    extra=()
    if [[ "$phase" == test ]]; then dataset=test; episodes=50; fi
    if [[ "$phase" == eval-preflight ]]; then extra=(--preflight-only); fi
    for arm in control guided; do
      "$R17_PYTHON" run_scripts/robomme/eval_archive_read_control_v7.py \
        --archive-checkpoint "$R17_RUN_DIR/$arm/$R17_CHECKPOINT" \
        --models baseline archive archive-off --tasks all --dataset "$dataset" \
        --n-episodes "$episodes" --seed 6 --n-action-steps 16 --max-episode-steps 1300 \
        --output-dir "$R17_EVAL_DIR/${dataset}_${arm}" "${extra[@]}"
    done
    ;;
  compare-val|compare-test)
    dataset="${1#compare-}"
    extra=()
    if [[ "$dataset" == test ]]; then
      extra=(--historical-run runs/eval/robomme/v4_best_all_n50_seed6)
    fi
    "$R17_PYTHON" run_scripts/robomme/compare_archive_retrieval_v17.py \
      --control-run "$R17_EVAL_DIR/${dataset}_control" \
      --guided-run "$R17_EVAL_DIR/${dataset}_guided" \
      --output-dir "$R17_EVAL_DIR/${dataset}_report" "${extra[@]}"
    ;;
  *)
    echo "Usage: bash $0 {preflight|smoke|train|train-control|train-guided|eval-preflight|val|compare-val|test|compare-test}"
    echo "Default: 512 updates per arm. No command launches the next phase automatically."
    echo "TEST: 2 arms x 3 roles x 16 tasks x 50 episodes = 4800 rollouts (long)."
    ;;
esac
