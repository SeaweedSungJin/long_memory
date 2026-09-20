#!/usr/bin/env python3
"""Opt-in action-value v3 CLI. The legacy trainers are unchanged."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gr00t.long_memory.train_v3 import main

if __name__ == "__main__":
    raise SystemExit(main())
