#!/usr/bin/env python3
"""Isolated recall/continuation recipe; leaves all existing training runs intact."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gr00t.long_memory.train_v5 import main

if __name__ == "__main__":
    raise SystemExit(main())
