#!/usr/bin/env python3
"""Standalone v4 reader/Action-Expert adaptation and frozen-reader writer training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gr00t.long_memory.train_v4 import main

if __name__ == "__main__":
    raise SystemExit(main())
