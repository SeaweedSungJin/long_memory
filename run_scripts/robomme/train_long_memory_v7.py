#!/usr/bin/env python3
"""V7 full-prefix training entrypoint; prior experiment files remain untouched."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gr00t.long_memory.train_v7 import main

if __name__ == "__main__":
    raise SystemExit(main())
