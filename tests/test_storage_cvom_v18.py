"""CPU contracts for causal V18 storage; no simulator/model download required."""
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from run_scripts.robomme.storage_cvom_v18 import (StorageCVOMV18, WriterConfigV18,
    fifo_insert, load_storage_writer_v18, make_write_policy, require_informative_labels,
    save_storage_writer_v18, storage_features)
from run_scripts.robomme.train_storage_cvom_v18 import build_context_plan, label_contexts


def _writer():
    return StorageCVOMV18(WriterConfigV18(memory_dim=3, capacity_events=2, hidden_dim=8))


def test_zero_writer_is_exact_fifo_and_capacity_is_events():
    writer, bank, expected = _writer(), None, None
    policy = make_write_policy(writer)
    for i in range(5):
        event = torch.full((1, 4, 3), float(i))
        bank, metrics = policy(bank, event, event_index=i, frame=i * 16, is_demo=i < 2)
        expected = fifo_insert(expected, event, 2)
        assert torch.equal(bank, expected)
        assert bank.shape[1] <= 8
        assert metrics["writer_insert"] == 1
    assert torch.equal(bank, torch.cat((torch.full((1, 4, 3), 3.), torch.full((1, 4, 3), 4.)), 1))


def test_negative_writer_fills_then_keeps_and_does_not_store_retrieval():
    writer = _writer()
    with torch.no_grad():
        writer.network[-1].bias.fill_(-2)
    policy, bank = make_write_policy(writer), None
    for i in range(4):
        bank, metrics = policy(bank, torch.full((1, 2, 3), float(i)), event_index=i)
    assert bank.shape == (1, 4, 3)
    assert torch.equal(bank[:, :2], torch.zeros(1, 2, 3))
    assert torch.equal(bank[:, 2:], torch.ones(1, 2, 3))
    assert metrics["writer_keep"] == 1


def test_features_no_future_metadata_and_inputs_immutable():
    bank, candidate = torch.randn(1, 4, 3), torch.randn(1, 2, 3)
    old = bank.clone()
    feature = storage_features(bank, candidate, 2)
    assert feature.shape == (1, 16)
    assert feature[0, -1] == 1
    writer = _writer()
    policy = make_write_policy(writer)
    a = policy(bank, candidate, frame=1, is_demo=False)[0]
    b = policy(bank, candidate, frame=999999, is_demo=True)[0]
    assert torch.equal(a, b)
    assert torch.equal(bank, old)
    with unittest.TestCase().assertRaisesRegex(ValueError, "complete events"):
        storage_features(bank[:, :3], candidate, 2)
    with unittest.TestCase().assertRaisesRegex(ValueError, "identical trained event budget"):
        make_write_policy(writer, capacity_events=3)


def test_constant_or_nonfinite_labels_fail_closed():
    for values in ([0., 0.], [1., 2.], [-1., -2.]):
        with unittest.TestCase().assertRaisesRegex(ValueError, "lack both positive"):
            require_informative_labels(values)
    with unittest.TestCase().assertRaisesRegex(ValueError, "nonempty and finite"):
        require_informative_labels([float("nan")])
    result = require_informative_labels([-.1, 0, .1])
    assert result["count"] == 3


def test_only_writer_mlp_optimizes_from_detached_causal_features():
    torch.manual_seed(1)
    writer = _writer()
    frozen_actor = torch.nn.Linear(3, 3).requires_grad_(False)
    original_actor = {k: v.clone() for k, v in frozen_actor.state_dict().items()}
    features = []
    for sign in (-1., 1.):
        bank = frozen_actor(torch.ones(1, 4, 3))
        event = frozen_actor(torch.full((1, 2, 3), sign))
        features.append(storage_features(bank, event, 2).detach())
    x = torch.cat(features)
    targets = torch.tensor([-1., 1.])
    optimizer = torch.optim.Adam(writer.parameters(), lr=.02)
    for _ in range(40):
        loss = torch.nn.functional.mse_loss(writer.forward_features(x), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    score = writer.forward_features(x).detach()
    assert score[0] < 0 < score[1]
    for name, value in frozen_actor.state_dict().items():
        assert torch.equal(value, original_actor[name])
    assert all(p.grad is None for p in frozen_actor.parameters())


def test_writer_save_load_and_parent_hash_binding():
    with tempfile.TemporaryDirectory() as directory:
        _writer_save_load_and_parent_hash_binding(Path(directory))


def _writer_save_load_and_parent_hash_binding(tmp_path):
    from safetensors.torch import save_file
    parent = tmp_path / "reader"
    parent.mkdir()
    (parent / "checkpoint.json").write_text(json.dumps({"test": True}))
    save_file({"a": torch.ones(1)}, str(parent / "model.safetensors"))
    save_file({"b": torch.ones(1)}, str(parent / "expert.safetensors"))
    writer = _writer()
    path = tmp_path / "writer"
    save_storage_writer_v18(path, writer, parent, step=7, metadata={"actor_frozen": True})
    restored, cfg, manifest = load_storage_writer_v18(path, parent)
    assert cfg == writer.config and manifest["step"] == 7
    assert all(not p.requires_grad for p in restored.parameters())
    for key, value in writer.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    with unittest.TestCase().assertRaises(FileExistsError):
        save_storage_writer_v18(path, writer, parent, step=7, metadata={})
    save_file({"b": torch.zeros(1)}, str(parent / "expert.safetensors"))
    with unittest.TestCase().assertRaisesRegex(ValueError, "different Stage-1"):
        load_storage_writer_v18(path, parent)


def _episodes():
    data = {}
    for eid in range(4):
        data[eid] = {"episode_id": eid, "frames": torch.arange(8),
                     "decision_mask": torch.tensor([False, False, True, True, True, True, True])}
    return SimpleNamespace(fetch=lambda eid: data[eid])


def test_context_plan_is_full_causal_split_and_deterministic():
    cache = SimpleNamespace(manifest={"splits": {"train": [0, 1], "val": [2, 3]}})
    args = SimpleNamespace(storage_contexts=4, val_storage_contexts=4, future_samples=2,
                           future_horizon=3, seed=42)
    plan = build_context_plan(cache, _episodes(), 2, args)
    assert plan == build_context_plan(cache, _episodes(), 2, args)
    for split, rows in plan.items():
        assert len(rows) == 4
        for row in rows:
            assert row["episode_id"] in cache.manifest["splits"][split]
            assert row["candidate"] >= 2
            assert all(row["candidate"] < q <= row["candidate"] + 3 for q in row["future"])
    with unittest.TestCase().assertRaisesRegex(ValueError, "Changing capacity requires"):
        build_context_plan(cache, _episodes(), 9, args)


def test_counterfactual_labels_use_common_noise_and_true_future_only():
    import run_scripts.robomme.train_storage_cvom_v18 as trainer
    events = torch.arange(7.).reshape(7, 1, 1).expand(-1, 1, 3).contiguous()
    class Core:
        config = SimpleNamespace(capacity_events=2, num_short_tokens=1)
        def initial_bank(self):
            return events[0:1, :0]
        def encode_prefix(self, ep, count):
            assert count == 6
            return {"stored": events[:count], "short": events[:count], "query": events[:count]}
        def write_fifo(self, bank, candidate):
            return fifo_insert(bank, candidate, 2)
        def read_from_bank(self, short, query, bank):
            return bank.mean(1, keepdim=True), {}
    calls = []
    def flow(head, ep, q, fused, seed):
        calls.append((q, float(fused.mean()), seed))
        return float(fused.mean())
    args = SimpleNamespace(noise_samples=2, seed=42)
    rows = [{"episode_id": 0, "candidate": 2, "future": [3, 5]}]
    with patch.object(trainer, "_flow_loss", flow):
        examples = label_contexts(Core(), None, SimpleNamespace(fetch=lambda eid: {}), rows, _writer(), args,
                                 split="train", refresh=0)
    assert len(calls) == 8
    assert all(calls[i][0] == calls[i + 1][0] and calls[i][2] == calls[i + 1][2] for i in range(0, 8, 2))
    # q=3: KEEP retains [0,1], INSERT stores [1,2]. q=5: both FIFO
    # continuations retain [3,4], so the original write effect has expired.
    assert calls[0][1] == .5 and calls[1][1] == 1.5
    assert calls[4][1] == calls[5][1] == 3.5
    assert examples[0]["utility"] == -.5


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in sorted(globals().items())
                              if name.startswith("test_") and callable(fn))


if __name__ == "__main__":
    unittest.main()
