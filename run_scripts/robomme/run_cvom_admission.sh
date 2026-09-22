#!/usr/bin/env bash
# Writer-only study. No checkpoint/cache overwrite, downloads or GPU preemption.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${CVOM_PYTHON:-.venv/bin/python}"
RUN_DIR="${CVOM_RUN_DIR:-runs/long_memory/cvom_admission_full_v1}"
EVAL_DIR="${CVOM_EVAL_DIR:-runs/eval/robomme/cvom_admission_full_v1_val160_seed6}"
PARENT="runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072"
CACHE="runs/long_memory/cache_full1600_v1"
UPDATES="${CVOM_UPDATES:-2000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export GR00T_INFERENCE_SEED=6

common=(--parent-checkpoint "$PARENT" --cache-dir "$CACHE" --output-dir "$RUN_DIR"
        --train-contexts "${CVOM_TRAIN_CONTEXTS:-0}" --val-contexts "${CVOM_VAL_CONTEXTS:-64}"
        --updates "$UPDATES")

checkpoint() {
    printf '%s/%s/checkpoint-%06d' "$RUN_DIR" "$1" "$UPDATES"
}

train_writer() {
    if [[ -f "$RUN_DIR/training_summary.json" ]]; then
        "$PYTHON" -c 'import json,sys; from pathlib import Path; from run_scripts.robomme.cvom_admission_checkpoint import inspect_checkpoint; p=Path(sys.argv[1]); n=int(sys.argv[2]); s=json.loads((p/"train_status.json").read_text()); assert s=={"status":"complete","step":n}; [inspect_checkpoint(p/a/f"checkpoint-{n:06d}",sys.argv[3]) for a in ("single","coalitional")]; print("[cvom] Reusing complete immutable writer training")' "$RUN_DIR" "$UPDATES" "$PARENT"
    else
        "$PYTHON" run_scripts/robomme/train_cvom_admission.py train "${common[@]}"
    fi
}

evaluate() {
    local arm="$1"
    shift
    "$PYTHON" run_scripts/robomme/eval_cvom_admission.py \
        --writer-checkpoint "$(checkpoint "$arm")" --output-dir "$EVAL_DIR/$arm" "$@"
}

compare() {
    "$PYTHON" run_scripts/robomme/summarize_cvom_admission.py \
        --single-run "$EVAL_DIR/single" --coalitional-run "$EVAL_DIR/coalitional" \
        --output-dir "${CVOM_SUMMARY_DIR:-$EVAL_DIR/summary}"
}

action="${1:-help}"
if [[ "$action" != "help" && "$action" != "monitor" ]]; then
    mkdir -p "$(dirname "$RUN_DIR")"
    exec 9>"$RUN_DIR.lock"
    flock -n 9 || { printf '%s\n' '[cvom] Another pipeline owns this run; nothing started.' >&2; exit 1; }
fi
case "$action" in
    preflight) "$PYTHON" run_scripts/robomme/train_cvom_admission.py preflight "${common[@]}" ;;
    prepare) "$PYTHON" run_scripts/robomme/train_cvom_admission.py prepare "${common[@]}" ;;
    train) train_writer ;;
    verify) "$PYTHON" run_scripts/robomme/verify_cvom_admission.py \
        --writer-checkpoint "$(checkpoint coalitional)" --output-dir "$RUN_DIR/runtime_verification" ;;
    eval-preflight)
        evaluate single --models fifo memory --preflight-only
        evaluate coalitional --models memory --preflight-only ;;
    eval)
        evaluate single --models fifo memory
        evaluate coalitional --models memory ;;
    compare) compare ;;
    all)
        "$PYTHON" run_scripts/robomme/train_cvom_admission.py preflight "${common[@]}"
        "$PYTHON" run_scripts/robomme/train_cvom_admission.py prepare "${common[@]}"
        train_writer
        if [[ ! -f "$RUN_DIR/runtime_verification/completed.json" ]]; then
            "$PYTHON" run_scripts/robomme/verify_cvom_admission.py \
                --writer-checkpoint "$(checkpoint coalitional)" --output-dir "$RUN_DIR/runtime_verification"
        else
            "$PYTHON" -c 'import json,sys; from pathlib import Path; from run_scripts.robomme.cvom_admission_checkpoint import inspect_checkpoint; p=Path(sys.argv[1]); assert json.loads((p/"completed.json").read_text())["passed"] is True; assert json.loads((p/"plan.json").read_text())["checkpoint"]==inspect_checkpoint(sys.argv[2]); print("[cvom] Reusing matching real-AE verification")' "$RUN_DIR/runtime_verification" "$(checkpoint coalitional)"
        fi
        evaluate single --models fifo memory --preflight-only
        evaluate coalitional --models memory --preflight-only
        evaluate single --models fifo memory
        evaluate coalitional --models memory
        if [[ ! -e "${CVOM_SUMMARY_DIR:-$EVAL_DIR/summary}" ]]; then compare; fi ;;
    monitor) "$PYTHON" run_scripts/robomme/monitor_long_memory.py --logdir "$RUN_DIR" --port "${CVOM_TB_PORT:-6007}" ;;
    *) printf '%s\n' 'Usage: bash run_scripts/robomme/run_cvom_admission.sh {preflight|prepare|train|verify|eval-preflight|eval|compare|all|monitor}' ;;
esac
