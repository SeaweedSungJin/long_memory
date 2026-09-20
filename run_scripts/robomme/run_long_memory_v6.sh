#!/usr/bin/env bash
# Isolated v6 workflow. Existing model/checkpoint files are never overwritten.
# Run with bash; see docs/LONG_MEMORY_V6_RETRIEVAL_CVOM.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PHASE="${1:-help}"
PYTHON="${V6_PYTHON:-.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NO_ALBUMENTATIONS_UPDATE=1
V6_CACHE="${V6_CACHE:-runs/long_memory/cache_full1600_v1}"
if [[ "$PHASE" == smoke ]]; then
  V6_RUN_ROOT="${V6_RUN_ROOT:-runs/long_memory/v6_visual_smoke_v1}"
else
  V6_RUN_ROOT="${V6_RUN_ROOT:-runs/long_memory/v6_visual_full_v1}"
fi

checkpoint_pointer() {
  "$PYTHON" -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]).resolve(); p=json.loads((root/sys.argv[2]).read_text()); target=(root/p["path"]).resolve(); assert (target/"checkpoint.json").is_file(), target; print(target)' "$1" "$2"
}

stage1() {
  "$PYTHON" run_scripts/robomme/train_long_memory_v6.py \
    --stage 1 --cache-dir "$V6_CACHE" --output-dir "$V6_RUN_ROOT/stage1" \
    --max-steps "${V6_STAGE1_STEPS:-2000}" --grad-accum "${V6_GRAD_ACCUM:-4}" \
    --val-samples "${V6_VAL_SAMPLES:-64}" --eval-steps 100 --save-steps 250 \
    --plot-steps 100 --activation-checkpointing
}

stage2() {
  V6_READER="$(checkpoint_pointer "$V6_RUN_ROOT/stage1" best_checkpoint.json)"
  "$PYTHON" -c 'import json,sys; from pathlib import Path; step=json.loads((Path(sys.argv[1])/"checkpoint.json").read_text())["step"]; assert step>0, "Stage-1 best is still step 0: inspect validation before training CVOM."' "$V6_READER"
  "$PYTHON" run_scripts/robomme/train_long_memory_v6.py \
    --stage 2 --cache-dir "$V6_CACHE" --init-checkpoint "$V6_READER" \
    --output-dir "$V6_RUN_ROOT/stage2" --max-steps "${V6_STAGE2_STEPS:-1000}" \
    --cvom-contexts 128 --cvom-batch-size 4 --context-refresh-steps 250 \
    --noise-samples 2 --val-samples "${V6_VAL_SAMPLES:-64}" \
    --eval-steps 100 --save-steps 250 --plot-steps 100
}

evaluate() {
  V6_MEMORY="$(checkpoint_pointer "$V6_RUN_ROOT/stage2" best_checkpoint.json)"
  # Use the actual fixed teacher, not a later unrelated Stage-1 checkpoint.
  V6_READER="$("$PYTHON" -c 'import json,sys; from pathlib import Path; p=json.loads((Path(sys.argv[1])/"checkpoint.json").read_text()); print(p["metadata"]["stage1_parent"]["path"])' "$V6_MEMORY")"
  local -a options
  if [[ "$PHASE" == eval-smoke ]]; then
    options=(--tasks BinFill PatternLock --n-episodes 3 --dataset val)
  else
    options=(--tasks all --n-episodes 50 --dataset "${PHASE#eval-}")
  fi
  local result_dir="${V6_EVAL_DIR:-runs/eval/robomme/$(basename "$V6_RUN_ROOT")_${PHASE}_seed6}"
  local -a command=("$PYTHON" run_scripts/robomme/eval_long_memory_v6.py
    --models baseline reader cvom fixed expert-only
    --reader-checkpoint "$V6_READER" --memory-checkpoint "$V6_MEMORY"
    "${options[@]}" --seed 6 --output-dir "$result_dir")
  "${command[@]}" --preflight-only
  "${command[@]}"
}

case "$PHASE" in
  stage1) stage1 ;;
  stage2) stage2 ;;
  smoke)
    "$PYTHON" run_scripts/robomme/train_long_memory_v6.py \
      --stage 1 --cache-dir "$V6_CACHE" --output-dir "$V6_RUN_ROOT/stage1" \
      --max-steps 4 --grad-accum 1 --warmup-steps 0 \
      --max-train-episodes 2 --max-val-episodes 2 --val-samples 2 \
      --eval-steps 2 --save-steps 2 --log-steps 1 --plot-steps 2 --activation-checkpointing
    V6_READER="$(checkpoint_pointer "$V6_RUN_ROOT/stage1" last_checkpoint.json)"
    "$PYTHON" run_scripts/robomme/train_long_memory_v6.py \
      --stage 2 --cache-dir "$V6_CACHE" --init-checkpoint "$V6_READER" \
      --output-dir "$V6_RUN_ROOT/stage2" --max-steps 2 --warmup-steps 0 \
      --max-train-episodes 2 --max-val-episodes 2 --cvom-contexts 2 \
      --cvom-batch-size 1 --val-samples 2 --eval-steps 2 --save-steps 2 --log-steps 1 --plot-steps 2
    printf '%s\n' "Training smoke complete. These few-step weights are NOT accuracy experiments."
    ;;
  eval-smoke|eval-val|eval-test) evaluate ;;
  monitor)
    "$PYTHON" run_scripts/robomme/monitor_long_memory.py --logdir "$V6_RUN_ROOT" --port "${V6_TB_PORT:-6011}"
    ;;
  *) printf '%s\n' 'Usage: bash run_scripts/robomme/run_long_memory_v6.sh {smoke|stage1|stage2|eval-smoke|eval-val|eval-test|monitor}'
     printf '%s\n' 'Optional: V6_RUN_ROOT, V6_STAGE1_STEPS, V6_STAGE2_STEPS, V6_VAL_SAMPLES, V6_GRAD_ACCUM, V6_EVAL_DIR.' ;;
esac
