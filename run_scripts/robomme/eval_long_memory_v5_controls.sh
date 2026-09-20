#!/usr/bin/env bash
# Evaluation only: original HAMLET + both Stage-1 best checkpoints, sequentially.
# Explicit Python paths avoid depending on an activated conda/venv shell.
set -euo pipefail
V5_CONTROLS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$V5_CONTROLS_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec .venv/bin/python -u run_scripts/robomme/eval_long_memory_v5_controls.py "$@"
