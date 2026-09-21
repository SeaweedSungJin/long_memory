#!/usr/bin/env bash
# No conda switch, downloads, deletion, or implicit overwrite. Use the repo venv.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON="${SEMANTIC_PYTHON:-.venv/bin/python}"
RUN_DIR="${SEMANTIC_RUN_DIR:-runs/long_memory/semantic_full_v1}"
TARGETS_DIR="${SEMANTIC_TARGETS_DIR:-runs/long_memory/semantic_targets_full1600_v1}"
CACHE_DIR="${SEMANTIC_CACHE_DIR:-runs/long_memory/cache_full1600_v1}"
INIT="${SEMANTIC_INIT:-runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072}"
EVAL_DIR="${SEMANTIC_EVAL_DIR:-runs/eval/robomme/semantic_full_v1_val160_seed6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

latest() {
  "$PYTHON" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(p/json.loads((p/"last_checkpoint.json").read_text())["path"])' "$1"
}

prepare() {
  if [[ -e "$TARGETS_DIR" ]]; then
    "$PYTHON" -c 'import json,sys; from pathlib import Path; from run_scripts.robomme.semantic_memory_targets import SemanticTargets; m=json.loads((Path(sys.argv[2])/"manifest.json").read_text()); t=SemanticTargets(sys.argv[1],m); print("[targets] reuse immutable", t.manifest["fingerprint"])' "$TARGETS_DIR" "$CACHE_DIR"
  else
    "$PYTHON" run_scripts/robomme/semantic_memory_targets.py \
      --cache-dir "$CACHE_DIR" --output-dir "$TARGETS_DIR"
  fi
}

stage1() {
  "$PYTHON" run_scripts/robomme/train_semantic_memory.py \
    --stage 1 --cache-dir "$CACHE_DIR" --targets-dir "$TARGETS_DIR" \
    --init-checkpoint "$INIT" --output-dir "$RUN_DIR/stage1" \
    --epochs "${SEMANTIC_EPOCHS:-1}" --activation-checkpointing "$@"
}

stage2() {
  local checkpoint
  checkpoint="$(latest "$RUN_DIR/stage1")"
  "$PYTHON" run_scripts/robomme/train_semantic_memory.py \
    --stage 2 --cache-dir "$CACHE_DIR" --targets-dir "$TARGETS_DIR" \
    --init-checkpoint "$checkpoint" --output-dir "$RUN_DIR/stage2" \
    --epochs "${SEMANTIC_EPOCHS:-1}" --activation-checkpointing "$@"
}

evaluate() {
  local phase="$1"
  shift
  local checkpoint
  checkpoint="$(latest "$RUN_DIR/$phase")"
  "$PYTHON" run_scripts/robomme/eval_semantic_memory.py \
    --checkpoint "$checkpoint" --output-dir "$EVAL_DIR/$phase" "$@"
}

case "${1:-help}" in
  prepare) prepare ;;
  preflight) prepare; stage1 --preflight-only ;;
  stage1) prepare; stage1 ;;
  stage2-preflight) stage2 --preflight-only ;;
  stage2) stage2 ;;
  eval-stage1) evaluate stage1 ;;
  eval-preflight) evaluate stage2 --preflight-only ;;
  eval) evaluate stage2 ;;
  eval-ablation) evaluate stage2 --models memory fifo memory-off ;;
  all) prepare; stage1 --preflight-only; stage1; stage2 --preflight-only; stage2; evaluate stage2 --preflight-only; evaluate stage2 ;;
  monitor) "$PYTHON" run_scripts/robomme/monitor_long_memory.py --logdir "$RUN_DIR" --port "${SEMANTIC_TB_PORT:-6007}" ;;
  *) printf '%s\n' 'Usage: bash run_scripts/robomme/run_semantic_memory.sh {prepare|preflight|stage1|stage2-preflight|stage2|eval-stage1|eval-preflight|eval|eval-ablation|all|monitor}' ;;
esac
