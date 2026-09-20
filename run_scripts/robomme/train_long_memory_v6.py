#!/usr/bin/env python
"""Train the isolated v6 visual reader / read-time CVOM. See the v6 protocol doc."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gr00t.long_memory.train_v6 import main

if __name__ == '__main__':
    main()
