#!/usr/bin/env python3
"""Prepare additive v5 targets; never modify source parquet/cache/checkpoints."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gr00t.long_memory.recall_data_v5 import main

if __name__ == "__main__":
    raise SystemExit(main())
