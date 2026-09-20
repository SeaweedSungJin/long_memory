#!/usr/bin/env python3
"""Stable CLI; implementation lives in gr00t.long_memory.train."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gr00t.long_memory.train import main

if __name__ == "__main__":
    main()
