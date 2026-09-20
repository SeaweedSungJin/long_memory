#!/usr/bin/env python
"""Launch the independent live TensorBoard dashboard; see module CLI help."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gr00t.long_memory.live_monitor import main


if __name__ == "__main__":
    raise SystemExit(main())
