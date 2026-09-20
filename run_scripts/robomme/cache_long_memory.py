#!/usr/bin/env python3
"""CLI entry point; cache format and extraction live in gr00t.long_memory.cache."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gr00t.long_memory.cache import main


if __name__ == "__main__":
    main()
