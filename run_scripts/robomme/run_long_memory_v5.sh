#!/usr/bin/env bash
# One explicit step per invocation; no background training or implicit next stage.
set -euo pipefail
V5_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$V5_REPO_ROOT"
exec .venv/bin/python run_scripts/robomme/long_memory_v5_workflow.py "$@"
