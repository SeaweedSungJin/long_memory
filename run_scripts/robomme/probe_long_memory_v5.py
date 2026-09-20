#!/usr/bin/env python3
"""Fit disposable diagnostic heads; never train the production memory/AE."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gr00t.long_memory.recall_probe_v5 import main

if __name__ == "__main__":
    raise SystemExit(main())
