#!/usr/bin/env python3
"""Stage-2 v2 CLI. Keep v1 train_long_memory.py available for reproductions."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gr00t.long_memory.train_stage2_v2 import main

if __name__ == "__main__":
    raise SystemExit(main())
