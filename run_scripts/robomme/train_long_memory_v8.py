#!/usr/bin/env python3
"""Independent V8 event-memory Stage-1 training entrypoint."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gr00t.long_memory.train_v8 import main

if __name__ == "__main__":
    raise SystemExit(main())
