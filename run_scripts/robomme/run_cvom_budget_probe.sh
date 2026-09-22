#!/usr/bin/env bash
# Frozen-teacher qualification only. Never starts training or a simulator.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

command="${1:-preflight}"
if (( $# )); then shift; fi
common=(
  --checkpoint "${CVOM_PROBE_CHECKPOINT:-runs/long_memory/echo_cvom_full_v1/refresh/checkpoint-001000}"
  --cache-dir "${CVOM_PROBE_CACHE:-runs/long_memory/cache_full1600_v1}"
  --output-dir "${CVOM_PROBE_OUTPUT:-runs/long_memory/cvom_budget_val16_v1}"
  --split val --contexts 16 --future-samples 2 --noise-samples 2
  --generation-repeats 1 --seed 260922 --device cuda:0 --cpu-threads 4
)
case "$command" in
  preflight|run|report)
    exec .venv/bin/python -u run_scripts/robomme/cvom_budget_probe.py "$command" "${common[@]}" "$@" ;;
  smoke)
    # Same fixed 16-context plan; pause after two, then use 'resume'.
    exec .venv/bin/python -u run_scripts/robomme/cvom_budget_probe.py run "${common[@]}" --stop-after-contexts 2 "$@" ;;
  resume)
    exec .venv/bin/python -u run_scripts/robomme/cvom_budget_probe.py run "${common[@]}" --resume "$@" ;;
  *)
    echo "Usage: bash run_scripts/robomme/run_cvom_budget_probe.sh preflight|smoke|run|resume|report" >&2
    exit 2 ;;
esac
