"""Trainable action-query replay with optional extra passive visual evidence.

Both arms keep the ORIGINAL canonical query/short tokens. Only the prior visual
bank changes. The frozen parent's short fusion happens later, in the caller.
No state, action, label or target column is accessed here. No encoder output is
cached across optimizer steps. This module adds no parameters or sampling.
"""
from __future__ import annotations

import torch

from run_scripts.robomme.replay_visual_patch_v11 import _columns, _options, _row
from run_scripts.robomme.visual_demo_tail_bank_v13 import VisualDemoTailMemoryV13
from run_scripts.robomme.framewise_demo_tail_v13 import REPLAY_ENCODING, build_framewise_prefix


def replay_demo_tail(memory, observations, query, *, camera_order, include_tail,
                     sidecar=None, record=None, episode_id=None, cache_fingerprint=None,
                     sidecar_fingerprint=None, visual_read_enabled=True, checkpoint_encoding=False,
                     replay_encoding=REPLAY_ENCODING):
    """Return (conditioned original current features, strict prior visual bank).

    Both arms encode each original past observation separately, matching the
    online singleton call shape. Canonical-only does NOT substitute an empty
    sidecar for a demo episode. For the
    tail arm, the supplied sidecar must come from the hash-verifying loader;
    independent episode/cache/source bindings are checked before encoding.
    Disabling READ never disables the requested bank construction.
    """
    if not isinstance(memory, VisualDemoTailMemoryV13):
        raise TypeError("V13 replay requires its actual bank-capable memory class")
    if type(replay_encoding) is not str or replay_encoding != REPLAY_ENCODING:
        raise ValueError("V13 replay_encoding must be 'framewise'; batched encoding is unsupported")
    _options(memory, observations, query, camera_order, checkpoint_encoding)
    if type(include_tail) is not bool or type(visual_read_enabled) is not bool:
        raise ValueError("Tail and READ controls must be explicit booleans")
    columns = _columns(observations)
    feature, image, attention, frame, demo = _row(columns, query, memory)
    if bool(demo):
        raise ValueError("V13 training replay is action-query only, not passive priming")
    bank = build_framewise_prefix(memory, observations, query, include_tail=include_tail,
        sidecar=sidecar, record=record, episode_id=episode_id, cache_fingerprint=cache_fingerprint,
        sidecar_fingerprint=sidecar_fingerprint, camera_order=camera_order,
        checkpoint_encoding=checkpoint_encoding)
    if bool((bank.frames[bank.valid] >= frame).any()):
        raise ValueError("Current/future canonical or tail observation entered prior bank")
    current = memory.encode_observation(feature[None], image[None], attention[None],
        frame[None], demo[None], camera_order=camera_order)
    return memory.read(current, bank, enabled=visual_read_enabled), bank
