"""Output isolation for new v5 tools; old data/checkpoints stay read-only."""
from pathlib import Path


def validate_output_scope(output, *inputs):
    result = Path(output).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    broad = {Path("/"), Path("/home"), Path.home().resolve(), repo}
    if result in broad:
        raise ValueError(f"Choose a dedicated output directory, not {result}")
    for source in inputs:
        if source is None:
            continue
        path = Path(source).expanduser().resolve()
        if result == path or result.is_relative_to(path):
            raise ValueError(f"Output must not be inside a source/cache/checkpoint/label directory: {path}")
    return result
