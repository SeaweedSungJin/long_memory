#!/usr/bin/env python3
"""Memory-only cue sampling/storage/retrieval diagnosis; never starts training."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.long_memory.evidence_audit_cli import main


if __name__ == "__main__":
    main()
