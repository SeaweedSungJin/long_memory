"""CPU contracts: teacher causality, capacity, pairing and held-out selection."""
from types import SimpleNamespace
import json

import pytest
import torch

from run_scripts.robomme.cvom_budget_probe import (
    build_plan, drop_operations, future_partition, measure_context, seed_for,
    matching_verification, read_packets, digest,
)
from run_scripts.robomme.echo_cvom_core import EchoBank, EchoConfig, EchoMemoryV1
from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18


def episode(eid=0):
    generator = torch.Generator().manual_seed(22+eid)
    n = 24
    mask = torch.ones(n-1, 2, dtype=torch.bool)
    mask[:2] = False
    actions = torch.randn(n-1, 2, 4, generator=generator)
    actions[~mask] = 0
    return {"episode_id": eid, "short": torch.randn(n, 2, 8, generator=generator),
        "moment": torch.randn(n, 2, 8, generator=generator),
        "state": torch.randn(n, 3, generator=generator), "frames": torch.arange(n)*16,
        "is_demo": torch.arange(n) < 2, "actions": actions, "action_mask": mask,
        "targets": torch.randn(n-1, 50, 4, generator=generator),
        "transition_valid": torch.ones(n-1, dtype=torch.bool),
        "decision_mask": torch.arange(n-1) >= 2}


def core():
    config = RepresentationConfigV18(feature_dim=8, state_dim=3, num_short_tokens=2,
        hidden_dim=4, num_heads=2, capacity_events=3, short_window=2)
    ecfg = EchoConfig(capacity_events=3, min_fill=2, utility_hidden=8, effect_hidden=8, action_dim=4)
    return EchoMemoryV1(config, ecfg).eval().requires_grad_(False)


def test_full_budget_drop_is_whole_event_and_never_time_shuffle():
    pool = EchoBank(torch.empty(1, 0, 4))
    for i in range(4):
        pool = pool.append(torch.full((1, 2, 4), float(i)), 16*i, i < 2, i)
    operations, banks = drop_operations(pool, 3)
    assert [o["id"] for o in operations] == ["keep", "fifo", "drop-01", "drop-02"]
    assert banks[0].event_ids == ((0,), (1,), (2,))
    assert banks[1].event_ids == ((1,), (2,), (3,))
    assert all(b.tokens.shape == (1, 6, 4) for b in banks)
    for b in banks:
        for index, ids in enumerate(b.event_ids):
            assert bool((b.tokens[:, index*2:(index+1)*2] == ids[0]).all())
    with pytest.raises(ValueError, match="overflow"):
        drop_operations(pool, 4)


def test_plan_is_split_disjoint_deterministic_and_gt_values_do_not_select():
    records = {i: episode(i) for i in range(6)}
    episodes = SimpleNamespace(fetch=records.__getitem__)
    cache = SimpleNamespace(manifest={"splits": {"train": [0, 1, 2], "val": [3, 4, 5]}})
    tasks = {i: "a" if i%2 else "b" for i in records}
    kwargs = dict(split="val", count=3, future_samples=2, seed=99)
    plan, inventory = build_plan(cache, episodes, tasks, core().config, **kwargs)
    for ep in records.values():
        ep["targets"].fill_(float("nan"))
    same, _ = build_plan(cache, episodes, tasks, core().config, **kwargs)
    assert plan == same and sum(inventory.values()) == 3
    assert {r["episode_id"] for r in plan} == {3, 4, 5}
    for r in plan:
        assert r["event"] >= 3
        assert min(r["selection_queries"]) > r["event"] + 2
        assert not set(r["selection_queries"]) & set(r["heldout_queries"])
        assert r["selection_to_heldout_frame_gap"] >= 50
    with pytest.raises(ValueError):
        build_plan(cache, episodes, tasks, core().config, **dict(kwargs, split="test"))


def test_insufficient_future_refused_and_feasibility_is_monotone():
    ep = episode()
    possible = [future_partition(ep, t, 2, 2, seed=3) is not None for t in range(24)]
    assert possible == sorted(possible, reverse=True)
    assert future_partition(ep, 20, 2, 2, seed=3) is None


def test_common_noise_frozen_actor_same_short_capacity_and_no_future_write():
    c, head, ep = core(), torch.nn.Identity().eval(), episode()
    row = {"episode_id": 0, "task": "synthetic", "split": "val", "event": 3,
           "selection_queries": [6, 7], "heldout_queries": [13, 14]}
    calls, writes, encoding_counts, read_queries, read_banks = [], [], [], [], []
    update, encode, read = c.manager.update, c.encode_prefix, c.read_from_bank

    def updated(*args, **kwargs):
        writes.append((args[5], kwargs["mode"]))
        return update(*args, **kwargs)

    def encoded(ep, count, **kwargs):
        encoding_counts.append(count)
        return encode(ep, count, **kwargs)

    def reading(short, query, bank):
        read_queries.append((short.clone(), query.clone()))
        read_banks.append(bank.clone())
        return read(short, query, bank)

    c.manager.update, c.encode_prefix, c.read_from_bank = updated, encoded, reading

    def flow(head, ep, q, fused, *, seed, **kwargs):
        assert not torch.is_grad_enabled()
        calls.append((q, seed, fused.clone()))
        value = float(fused.square().mean()) + (seed % 100)/100
        return {k: value for k in ("loss", "original_flow_loss", "executed_prefix_flow_loss",
                                  "executed_prefix_joint_flow_loss", "executed_prefix_gripper_flow_loss")}

    def generate(head, ep, q, fused, **kwargs):
        return {k: float(fused.square().mean()) for k in
                ("generated_prefix_mse", "generated_executed_joint_mse", "generated_executed_gripper_mse")}

    state_before = {k: v.clone() for k, v in c.state_dict().items()}
    result = measure_context(c, head, ep, row, seed=99, flow_fn=flow, generation_fn=generate)
    assert encoding_counts == [4, 15]  # Storage is encoded independently of future panel.
    assert writes == [(0, "fifo"), (1, "fifo"), (2, "fifo"), (3, "learned")]
    assert result["calls"] == {"flow": 49, "generation": 9}
    assert result["regression"] == {"same_path_flow_max_abs": 0., "same_path_generated_max_abs": 0.,
                                      "global_rng_unchanged": True}
    for start in range(0, len(read_queries), 4):
        assert all(torch.equal(read_queries[start][0], x[0]) and torch.equal(read_queries[start][1], x[1])
                   for x in read_queries[start:start+4])
    assert all(b.shape == (1, 6, 4) for b in read_banks)
    for panel in result["panels"].values():
        for draw in panel["draws"]:
            paired = [call for call in calls if call[:2] == (draw["query"], draw["seed"])]
            assert len(paired) in (4, 5)  # First KEEP is repeated for exact regression.
    assert all(torch.equal(v, c.state_dict()[k]) for k, v in state_before.items())
    c.train()
    with pytest.raises(ValueError, match="frozen"):
        measure_context(c, head, ep, row, seed=99, flow_fn=flow)


def test_seed_is_stable_and_panel_specific():
    assert seed_for(3, 20, "a", 99) == seed_for(3, 20, "a", 99)
    assert seed_for(3, 20, "a", 99) != seed_for(3, 20, "b", 99)


def test_final_audit_must_cover_exact_protocol_packets_and_actor(tmp_path):
    protocol = {"fingerprint": "p"}
    packet = {"row_sha256": "r", "result_sha256": "v", "actor_state": {"core": "c", "expert": "e"}}
    assert matching_verification(tmp_path, protocol, [packet]) is None
    audit = {"protocol_fingerprint": "p", "verified_packets": {"r": "v"},
             "actor_unchanged": packet["actor_state"], "checkpoint_unchanged": True,
             "cache_stat_signatures_unchanged": True, "source_unchanged": True}
    path = tmp_path/"verification-1.json"
    path.write_text(json.dumps(audit))
    assert matching_verification(tmp_path, protocol, [packet]) == str(path)
    assert matching_verification(tmp_path, protocol, [dict(packet, result_sha256="modified")]) is None
    audit["source_unchanged"] = False
    path.write_text(json.dumps(audit))
    assert matching_verification(tmp_path, protocol, [packet]) is None


def test_packet_identity_and_result_integrity_fail_closed(tmp_path):
    row = {"episode_id": 0, "event": 32}
    protocol = {"fingerprint": "p", "plan": [row]}
    packet = {"protocol_fingerprint": "p", "row_sha256": digest(row),
              "result_sha256": digest(row), "result": dict(row)}
    path = tmp_path/"context-0000.json"
    path.write_text(json.dumps(packet))
    assert read_packets(tmp_path, protocol) == [packet]
    packet["result"]["event"] = 33
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="changed"):
        read_packets(tmp_path, protocol)
