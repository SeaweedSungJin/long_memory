#!/usr/bin/env python3
"""Frozen v4 memory/writer diagnosis; no optimization or checkpoint writes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gr00t.long_memory.diagnostic_offline import main

if __name__ == "__main__":
    raise SystemExit(main())
