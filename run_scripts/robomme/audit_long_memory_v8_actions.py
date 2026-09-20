#!/usr/bin/env python3
"""Inspect generated actions and fixed-time flow error, never robot success."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gr00t.long_memory.action_audit_v8 import main

if __name__ == "__main__":
    raise SystemExit(main())
