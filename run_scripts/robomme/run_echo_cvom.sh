#!/usr/bin/env bash
# Explicit stages, fixed final checkpoints, no deletion or checkpoint overwrite.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ECHO_PYTHON="${ECHO_PYTHON:-.venv/bin/python}"
ECHO_RUN_DIR="${ECHO_RUN_DIR:-runs/long_memory/echo_cvom_full_v1}"
ECHO_EVAL_DIR="${ECHO_EVAL_DIR:-runs/eval/robomme/echo_cvom_full_v1_val160_seed6}"
ECHO_CACHE_DIR="${ECHO_CACHE_DIR:-runs/long_memory/cache_full1600_v1}"
ECHO_PARENT="${ECHO_PARENT:-runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072}"
ECHO_BASELINE_REFERENCE="${ECHO_BASELINE_REFERENCE:-runs/eval/robomme/archive_read_best1250_val_n10_seed6}"
ECHO_DEVICE="${ECHO_DEVICE:-cuda:0}"
ECHO_SEED="${ECHO_SEED:-192201}"
ECHO_EPOCHS="${ECHO_EPOCHS:-1}"
ECHO_QUERY_BATCH="${ECHO_QUERY_BATCH:-4}"
ECHO_WRITER_UPDATES="${ECHO_WRITER_UPDATES:-1000}"
ECHO_FINAL_PHASE="${ECHO_FINAL_PHASE:-refresh}"
ECHO_ACTION="${1:-help}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export GR00T_INFERENCE_SEED=6

if (( $# > 1 )); then
    printf '%s\n' '[echo] Supply one command; configure paths/budgets through ECHO_* environment variables.' >&2
    exit 2
fi
if [[ "$ECHO_FINAL_PHASE" != stage2 && "$ECHO_FINAL_PHASE" != refresh ]]; then
    printf '%s\n' '[echo] ECHO_FINAL_PHASE must be stage2 or refresh.' >&2
    exit 2
fi

echo_common=(--cache-dir "$ECHO_CACHE_DIR" --parent-checkpoint "$ECHO_PARENT" --seed "$ECHO_SEED")

final_checkpoint() {
    # Resolve only the published fixed final step, never best-on-VAL or a glob.
    "$ECHO_PYTHON" - "$ECHO_RUN_DIR/$1" "$ECHO_CACHE_DIR" "$ECHO_PARENT" \
        "$ECHO_SEED" "$ECHO_EPOCHS" "$ECHO_QUERY_BATCH" "$ECHO_WRITER_UPDATES" "$1" <<'PY'
import json, sys
from pathlib import Path
from run_scripts.robomme.echo_cvom_checkpoint import inspect_checkpoint
from run_scripts.robomme.train_echo_cvom import sources
run, cache, parent = (Path(value).resolve() for value in sys.argv[1:4])
seed, epochs, batch, updates = map(int, sys.argv[4:8])
phase = sys.argv[8]
status = json.loads((run / "status.json").read_text())
last = json.loads((run / "last_checkpoint.json").read_text())
stage = 1 if phase == "stage1" else 2
if (status.get("status") != "complete" or status.get("stage") != stage
        or status.get("step") != status.get("total_steps")
        or last.get("step") != status.get("step") or last.get("stage") != stage):
    raise ValueError("Require the complete, published fixed final checkpoint: " + str(run))
checkpoint = (run / last["path"]).resolve()
if checkpoint.parent != run:
    raise ValueError("Published checkpoint escaped its stage directory")
manifest = json.loads((cache / "manifest.json").read_text())
info = inspect_checkpoint(manifest["model_path"], checkpoint)
meta, train = info["metadata"], info["config"]["train"]
if (info["step"] != last["step"] or info["stage"] != stage
        or meta.get("training_complete") is not True
        or meta.get("inherited_actor_training_complete") is False
        or meta.get("smoke_only") is True):
    raise ValueError("Incomplete/smoke checkpoints cannot drive the full ECHO workflow")
if (meta.get("cache_fingerprint") != manifest["fingerprint"]
        or Path(meta["parent_identity"]["path"]).resolve() != parent
        or meta.get("source_sha256") != sources() or train.get("seed") != seed):
    raise ValueError("Completed checkpoint no longer matches cache/parent/source/seed")
if stage == 1 and (train.get("epochs") != epochs or train.get("query_batch_size") != batch
                   or train.get("activation_checkpointing") is not True):
    raise ValueError("Completed warmup does not match the requested full-epoch configuration")
if stage == 2 and (status["step"] != updates or train.get("writer_updates") != updates):
    raise ValueError("Completed writer does not match requested fixed final updates")
print(checkpoint)
PY
}

stage1() {
    if [[ -e "$ECHO_RUN_DIR/stage1" ]]; then
        final_checkpoint stage1 >/dev/null
        printf '%s\n' '[echo] Reusing complete stage1 with matching immutable provenance.'
        return
    fi
    "$ECHO_PYTHON" run_scripts/robomme/train_echo_cvom.py warmup "${echo_common[@]}" \
        --output-dir "$ECHO_RUN_DIR/stage1" --device "$ECHO_DEVICE" \
        --epochs "$ECHO_EPOCHS" --query-batch-size "$ECHO_QUERY_BATCH" --activation-checkpointing
    final_checkpoint stage1 >/dev/null
}

preflight() {
    # No mkdir, lock file, model loading, or journal is performed by this path.
    "$ECHO_PYTHON" run_scripts/robomme/train_echo_cvom.py warmup "${echo_common[@]}" \
        --output-dir "$ECHO_RUN_DIR/stage1" --device "$ECHO_DEVICE" \
        --epochs "$ECHO_EPOCHS" --query-batch-size "$ECHO_QUERY_BATCH" \
        --activation-checkpointing --preflight-only
}

labels() {
    local source_phase="$1" destination="$2" echo_checkpoint
    echo_checkpoint="$(final_checkpoint "$source_phase")"
    "$ECHO_PYTHON" run_scripts/robomme/train_echo_cvom.py labels "${echo_common[@]}" \
        --checkpoint "$echo_checkpoint" --output-dir "$ECHO_RUN_DIR/$destination" --device "$ECHO_DEVICE" \
        --contexts-per-episode 2 --train-context-limit 0 --val-context-limit 64 \
        --targets-per-context 4 --coalitions 4 --future-samples 2 --noise-samples 2
    "$ECHO_PYTHON" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); s=json.loads((p/"status.json").read_text()); assert s.get("status")=="complete" and s.get("remaining")==0 and (p/"manifest.json").is_file(), "Label preparation is incomplete"' "$ECHO_RUN_DIR/$destination"
}

writer() {
    local source_phase="$1" label_phase="$2" destination="$3" echo_checkpoint
    echo_checkpoint="$(final_checkpoint "$source_phase")"
    if [[ -e "$ECHO_RUN_DIR/$destination" ]]; then
        final_checkpoint "$destination" >/dev/null
        "$ECHO_PYTHON" - "$ECHO_RUN_DIR/$destination" "$echo_checkpoint" "$ECHO_RUN_DIR/$label_phase" <<'PY'
import json, sys
from pathlib import Path
run, source, labels = map(Path, sys.argv[1:])
last = json.loads((run / "last_checkpoint.json").read_text())
info = json.loads((run / last["path"] / "checkpoint.json").read_text())
train, teacher = info["config"]["train"], info["metadata"]["teacher_snapshot"]
if Path(teacher["path"]).resolve() != source.resolve() or Path(train["labels_dir"]).resolve() != labels.resolve():
    raise ValueError("Completed writer belongs to a different teacher round")
print("[echo] Reusing complete writer with matching fixed teacher round.")
PY
        return
    fi
    "$ECHO_PYTHON" run_scripts/robomme/train_echo_cvom.py writer "${echo_common[@]}" \
        --checkpoint "$echo_checkpoint" --labels-dir "$ECHO_RUN_DIR/$label_phase" \
        --output-dir "$ECHO_RUN_DIR/$destination" --device cpu \
        --writer-updates "$ECHO_WRITER_UPDATES" --writer-batch-size 64
    final_checkpoint "$destination" >/dev/null
}

verify() {
    local echo_checkpoint echo_verification
    echo_checkpoint="$(final_checkpoint "$ECHO_FINAL_PHASE")"
    echo_verification="$ECHO_RUN_DIR/runtime_verification_$ECHO_FINAL_PHASE"
    if [[ -e "$echo_verification" ]]; then
        "$ECHO_PYTHON" - "$echo_verification" "$echo_checkpoint" "$ECHO_CACHE_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
from run_scripts.robomme.echo_cvom_checkpoint import inspect_checkpoint
from run_scripts.robomme.policy_echo_cvom import runtime_source_identity
from run_scripts.robomme.verify_echo_cvom import PANEL
out, checkpoint, cache = map(Path, sys.argv[1:])
plan = json.loads((out / "plan.json").read_text())
done = json.loads((out / "completed.json").read_text())
events = json.loads((out / "events.json").read_text())
manifest = json.loads((cache / "manifest.json").read_text())
sources = runtime_source_identity()
sources["verify_echo_cvom.py"] = hashlib.sha256(Path("run_scripts/robomme/verify_echo_cvom.py").read_bytes()).hexdigest()
if (done.get("passed") is not True or done.get("endpoints") != 2*sum(n for _, n in PANEL)
        or len(events) != done["endpoints"] or plan.get("panel") != [list(row) for row in PANEL]
        or plan.get("checkpoint") != inspect_checkpoint(manifest["model_path"], checkpoint)
        or plan.get("source_sha256") != sources or plan.get("cache_fingerprint") != manifest["fingerprint"]):
    raise ValueError("Existing verification is incomplete or belongs to different runtime inputs")
print("[echo] Reusing complete native runtime verification with matching inputs.")
PY
        return
    fi
    "$ECHO_PYTHON" run_scripts/robomme/verify_echo_cvom.py --checkpoint "$echo_checkpoint" \
        --cache-dir "$ECHO_CACHE_DIR" --output-dir "$echo_verification" --device "$ECHO_DEVICE"
}

evaluate() {
    local echo_checkpoint
    local -a echo_roles
    echo_checkpoint="$(final_checkpoint "$ECHO_FINAL_PHASE")"
    read -r -a echo_roles <<< "${ECHO_MODELS:-fifo memory}"
    "$ECHO_PYTHON" run_scripts/robomme/eval_echo_cvom.py --checkpoint "$echo_checkpoint" \
        --models "${echo_roles[@]}" --baseline-reference "$ECHO_BASELINE_REFERENCE" \
        --output-dir "$ECHO_EVAL_DIR" --device "$ECHO_DEVICE" --server-python "$ECHO_PYTHON" \
        --robomme-python "${ECHO_ROBOMME_PYTHON:-/home/sjkim/robomme_benchmark/.venv/bin/python}" "$@"
}

case "$ECHO_ACTION" in
    stage1|labels0|stage2|labels1|refresh|verify|eval|all)
        mkdir -p "$(dirname "$ECHO_RUN_DIR")"
        exec 9>"$ECHO_RUN_DIR.lock"
        flock -n 9 || { printf '%s\n' '[echo] Another pipeline owns this run; nothing started.' >&2; exit 1; }
        ;;
    preflight|eval-preflight|monitor|help|-h|--help) ;;
    *) printf '[echo] Unknown command: %s\n' "$ECHO_ACTION" >&2; exit 2 ;;
esac

case "$ECHO_ACTION" in
    preflight) preflight ;;
    stage1) stage1 ;;
    labels0) labels stage1 labels0 ;;
    stage2) writer stage1 labels0 stage2 ;;
    labels1) labels stage2 labels1 ;;
    refresh) writer stage2 labels1 refresh ;;
    verify) verify ;;
    eval-preflight) evaluate --preflight-only ;;
    eval) evaluate ;;
    all)
        preflight
        stage1
        labels stage1 labels0
        writer stage1 labels0 stage2
        labels stage2 labels1
        writer stage2 labels1 refresh
        verify
        evaluate --preflight-only
        evaluate
        ;;
    monitor)
        "$ECHO_PYTHON" run_scripts/robomme/monitor_long_memory.py \
            --logdir "$ECHO_RUN_DIR" --port "${ECHO_TB_PORT:-6008}"
        ;;
    *)
        printf '%s\n' \
            'Usage: bash run_scripts/robomme/run_echo_cvom.sh {preflight|stage1|labels0|stage2|labels1|refresh|verify|eval-preflight|eval|all|monitor}' \
            'Defaults: full stage1 epoch, TRAIN-wide two contexts/episode, 1000 CPU writer updates per round.' \
            'all includes labels1 + refresh. ECHO_FINAL_PHASE=stage2 selects first-round verify/eval explicitly.' \
            'ECHO_MODELS="fifo memory" is default: 320 new episodes plus strictly reused original baseline.' \
            'Add memory-off through ECHO_MODELS to run the same-actor READ ablation.' \
            'preflight and eval-preflight are read-only; partial training outputs must be resumed into a new directory.'
        ;;
esac
