"""Production-scoped RNG isolation for image-only demo ingestion.

The actual V13 input proof showed identical features but consumed global RNG.
An extra preprocessing/backbone call must therefore preserve Python, NumPy,
CPU Torch, the model's one CUDA device, and any episode-local generator. This
utility does NOT initialize/snapshot other visible GPUs. It is not a policy or
proof of parent-cache/action parity; the caller must validate those separately.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import random

import numpy as np
import torch


@contextmanager
def isolated_visual_ingest_rng(device, episode_generator=None):
    """Restore caller RNG on success or failure, without changing any seed.

    Serial invocation is required: restoring global state would be unsafe if
    another thread concurrently generated random values. The policy's existing
    action/reset lock must also enclose this context. Processor-private RNG is
    outside this contract (eval transforms must separately prove determinism).
    """
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Demo ingestion supports CPU or one explicit CUDA device")
    if device.type == "cuda" and device.index is None:
        raise ValueError("CUDA ingestion requires an explicit model device index")
    if episode_generator is not None and not isinstance(episode_generator, torch.Generator):
        raise TypeError("episode_generator must be a Torch Generator or None")
    if episode_generator is not None and episode_generator.device != device:
        raise ValueError("Episode generator must belong to the model device")
    python_state = random.getstate()
    numpy_state = copy.deepcopy(np.random.get_state())
    cpu_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    episode_state = episode_generator.get_state().clone() if episode_generator is not None else None
    try:
        yield
    finally:
        # Restore independent streams even if an earlier restoration raises.
        try:
            random.setstate(python_state)
        finally:
            try:
                np.random.set_state(numpy_state)
            finally:
                try:
                    torch.set_rng_state(cpu_state)
                finally:
                    try:
                        if cuda_state is not None:
                            torch.cuda.set_rng_state(cuda_state, device)
                    finally:
                        if episode_state is not None:
                            episode_generator.set_state(episode_state)
