"""CPU contracts for learned event storage/retrieval; not robot-task results."""

import copy
import unittest

import torch

from gr00t.long_memory.event_v8 import EventMemoryV8, MemoryV8Config
from gr00t.long_memory.replay_v8 import apply_write, encode_at, initial_replay_state, replay_queries, replay_state


def make_memory(**kwargs):
    options = dict(feature_dim=12, state_dim=3, num_short_tokens=4, hidden_dim=8, capacity=8, num_heads=2)
    options.update(kwargs)
    return EventMemoryV8(MemoryV8Config(**options))


def make_episode(count=13, demo=3):
    return {"moment": torch.randn(count, 4, 12), "short": torch.randn(count, 4, 12),
            "state": torch.randn(count, 3), "frames": torch.arange(count) * 16,
            "is_demo": torch.arange(count) < demo, "actions": torch.full((count, 3), float("nan")),
            "targets": torch.full((count, 4, 3), float("nan"))}


def open_fusion(memory):
    with torch.no_grad():
        memory.fusion_projection.weight.normal_(std=0.1)


def grad_size(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


class V8CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2026)

    def test_config_and_stage_validation(self):
        for field in ("feature_dim", "state_dim", "num_short_tokens", "hidden_dim", "capacity", "num_heads"):
            for value in (True, 0, -1, 2.0, "2"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    make_memory(**{field: value})
        for options in ({"hidden_dim": 7}, {"time_scale": float("nan")}, {"time_scale": 0},
                        {"residual_scale": float("inf")}, {"residual_scale": 0},
                        {"residual_scale": True}, {"source": "future"}):
            with self.assertRaises(ValueError):
                make_memory(**options)
        memory = make_memory()
        self.assertIs(memory.set_stage(1), memory)
        self.assertTrue(all(p.requires_grad for p in memory.parameters()))
        self.assertEqual({id(p) for p in memory.actor_parameters()}, {id(p) for p in memory.parameters()})
        for stage in (0, 2, True, 1.0):
            with self.assertRaises(ValueError):
                memory.set_stage(stage)

    def test_shapes_metadata_fp32_and_meta_construction(self):
        memory, ep = make_memory(), make_episode()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            encoded = encode_at(memory, ep, 0)
            bank, _ = memory.write(memory.initial_state(), encoded)
            fused, _ = memory.read(ep["short"][1:2], encode_at(memory, ep, 1), bank)
            loss = memory.reconstruction_loss(encoded, ep["moment"][:1])
        self.assertEqual(encoded.shape, (1, 4, 10))
        self.assertEqual(bank.shape, (1, 4, 10))
        self.assertEqual(fused.shape, (1, 4, 12))
        self.assertTrue(torch.equal(encoded[0, :, -2], ep["frames"][0].float().expand(4)))
        self.assertTrue(torch.equal(encoded[0, :, -1], torch.ones(4)))
        for value in (encoded, bank, fused, loss):
            self.assertEqual(value.dtype, torch.float32)
        with torch.device("meta"):
            meta = make_memory()
        self.assertTrue(all(value.dtype == torch.float32 for value in meta.state_dict().values()))

    def test_encoder_preserves_token_specific_observation_and_metadata(self):
        memory, ep = make_memory(), make_episode()
        first = encode_at(memory, ep, 0)
        ep["moment"][0, 0, :2] += torch.tensor([4.0, -4.0])
        changed = encode_at(memory, ep, 0)
        self.assertFalse(torch.equal(first[:, 0, :-2], changed[:, 0, :-2]))
        self.assertTrue(torch.equal(first[:, 1:], changed[:, 1:]))
        self.assertTrue(torch.equal(first[..., -2:], changed[..., -2:]))

    def test_bounded_append_retains_distinct_values_without_mixing(self):
        memory, ep = make_memory(capacity=5), make_episode()
        bank = memory.initial_state()
        stored = []
        for index in range(10):
            encoded = encode_at(memory, ep, index)
            old = bank.clone()
            stored.append(encoded)
            bank, metrics = memory.write(bank, encoded)
            expected = torch.cat(stored[-5:], dim=1)
            self.assertTrue(torch.equal(bank, expected))
            self.assertEqual(float(metrics["retained_events"]), min(index + 1, 5))
            self.assertEqual(float(metrics["evicted_events"]), float(index >= 5))
            if index:
                self.assertTrue(torch.equal(old, torch.cat(stored[max(0, index - 5):index], dim=1)))
        self.assertGreater(float(bank[..., :-2].std(dim=1).mean()), 0.1)
        self.assertEqual(bank.shape[1], 20)
        self.assertEqual(float(bank[0, 0, -2]), 80)
        self.assertTrue(torch.equal(memory.initial_state(), torch.zeros(1, 0, 10)))

    def test_zero_init_has_projection_gradient_then_past_storage_gradient(self):
        memory, ep = make_memory(capacity=40), make_episode(count=31)
        ep["moment"].requires_grad_(True)
        fused = replay_queries(memory, ep, [30])[30][0]
        self.assertTrue(torch.equal(fused, ep["short"][30:31]))
        (fused * torch.randn_like(fused)).sum().backward()
        self.assertGreater(grad_size(memory.fusion_projection), 0)
        self.assertEqual(grad_size(memory.source_projection), 0)
        memory.zero_grad(set_to_none=True)
        ep["moment"].grad = None
        open_fusion(memory)
        fused = replay_queries(memory, ep, [30])[30][0]
        (fused * torch.randn_like(fused)).sum().backward()
        self.assertGreater(float(ep["moment"].grad[0].abs().sum()), 0)
        for module in (memory.source_projection, memory.encoder_ffn, memory.read_blocks, memory.short_projection):
            self.assertGreater(grad_size(module), 0)
        self.assertEqual(grad_size(memory.reconstruction_decoder), 0)

    def test_empty_and_disabled_identity_survives_learned_affine_weights(self):
        memory, ep = make_memory(), make_episode()
        open_fusion(memory)
        with torch.no_grad():
            for name, parameter in memory.named_parameters():
                if name.endswith("bias"):
                    parameter.normal_(std=3)
        current = encode_at(memory, ep, 3)
        full = replay_state(memory, ep, 3)
        for mode, enabled, bank in (("event", True, memory.initial_state()),
                                    ("event", False, full), ("none", True, full)):
            fused, metrics = memory.read(ep["short"][3:4], current, bank, mode=mode, memory_enabled=enabled)
            self.assertTrue(torch.equal(fused, ep["short"][3:4]))
            self.assertEqual(float(metrics["residual_norm"]), 0)

    def test_zero_content_has_no_affine_read_bypass(self):
        memory, ep = make_memory(), make_episode()
        open_fusion(memory)
        with torch.no_grad():
            for name, parameter in memory.named_parameters():
                if name.endswith("bias"):
                    parameter.normal_(std=3)
        bank = replay_state(memory, ep, 4).detach().clone()
        bank[..., :-2] = 0
        fused, metrics = memory.read(ep["short"][4:5], encode_at(memory, ep, 4), bank)
        self.assertTrue(torch.equal(fused, ep["short"][4:5]))
        self.assertEqual(float(metrics["residual_norm"]), 0)

    def test_residual_is_bounded_per_token_and_zero_short_stays_zero(self):
        memory, ep = make_memory(residual_scale=0.07), make_episode()
        with torch.no_grad():
            memory.fusion_projection.weight.normal_(std=100)
            memory.fusion_gate.bias.fill_(100)
        short = ep["short"][6:7]
        bank, current = replay_state(memory, ep, 6), encode_at(memory, ep, 6)
        fused, metrics = memory.read(short, current, bank)
        bound = 0.07 * short.square().mean(-1, keepdim=True).sqrt()
        self.assertTrue(bool(((fused - short).abs() <= bound + 2e-7).all()))
        self.assertLessEqual(float(metrics["residual_relative_max"]), 0.070001)
        zero = memory.read(torch.zeros_like(short), current, bank)[0]
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))

    def test_auxiliary_trains_encoder_and_decoder_but_detaches_target(self):
        memory, ep = make_memory(), make_episode()
        encoded = encode_at(memory, ep, 0)
        target = ep["moment"][:1].clone().requires_grad_(True)
        loss = memory.reconstruction_loss(encoded, target)
        loss.backward()
        self.assertGreater(grad_size(memory.source_projection), 0)
        self.assertGreater(grad_size(memory.reconstruction_decoder), 0)
        self.assertIsNone(target.grad)
        self.assertEqual(grad_size(memory.read_blocks), 0)

    def test_finite_shape_and_metadata_validation(self):
        memory, ep = make_memory(), make_episode()
        for mutate in (lambda x: x["moment"][0].fill_(float("nan")),
                       lambda x: x["state"][0].fill_(float("inf")),
                       lambda x: x.update(is_demo=x["is_demo"].float()),
                       lambda x: x.update(frames=x["frames"].float() + 0.5),
                       lambda x: x.update(moment=x["moment"][:, :2])):
            changed = copy.deepcopy(ep)
            mutate(changed)
            with self.assertRaises(ValueError):
                encode_at(memory, changed, 0)
        encoded = encode_at(memory, ep, 0)
        bad = encoded.clone()
        bad[0, 1, -2] = 1
        with self.assertRaises(ValueError):
            memory.write(memory.initial_state(), bad)
        with self.assertRaises(ValueError):
            memory.write(memory.initial_state()[:, :, :-1], encoded)
        with self.assertRaises(ValueError):
            memory.write(encoded, encoded)
        with self.assertRaises(TypeError):
            encode_at(memory.bfloat16(), ep, 0)


class V8ReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(21)

    def test_future_observations_actions_and_targets_cannot_change_earlier_read(self):
        memory, ep = make_memory(), make_episode()
        open_fusion(memory)
        first = replay_queries(memory, ep, [7])[7]
        changed = copy.deepcopy(ep)
        for key in ("short", "moment", "state"):
            changed[key][8:] = float("nan")
        changed["frames"][8:] = -999
        changed["is_demo"][8:] = True
        changed["actions"].fill_(1e20)
        changed["targets"].fill_(-1e20)
        second = replay_queries(memory, changed, [7])[7]
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1]["storage_reconstruction_loss"], second[1]["storage_reconstruction_loss"]))

    def test_offline_online_agree_and_current_write_is_excluded(self):
        memory, ep = make_memory(capacity=5), make_episode()
        open_fusion(memory)
        offline = replay_queries(memory, ep, [0, 2, 8, 12], checkpoint_segment=3)
        bank = initial_replay_state(memory)
        with torch.no_grad():
            for index in range(13):
                encoded = encode_at(memory, ep, index)
                if index in offline:
                    fused, _ = memory.read(ep["short"][index:index + 1], encoded, bank)
                    # Batched prefix GEMMs and single-event online GEMMs can
                    # round differently; their causal operations are identical.
                    torch.testing.assert_close(fused, offline[index][0], rtol=1e-6, atol=1e-6)
                    torch.testing.assert_close(bank, replay_state(memory, ep, index), rtol=1e-6, atol=1e-6)
                bank, _ = apply_write(memory, bank, encoded)
        before = replay_state(memory, ep, 8)
        ep["moment"][8].fill_(123)
        self.assertTrue(torch.equal(before, replay_state(memory, ep, 8)))
        self.assertEqual(float(offline[0][1]["storage_reconstruction_loss"]), 0)

    def test_auxiliary_exactly_covers_retained_past_and_has_no_current_target_gradient(self):
        memory, ep = make_memory(capacity=3), make_episode()
        ep["moment"].requires_grad_(True)
        result = replay_queries(memory, ep, [8])[8]
        encoded = memory.encode(ep["moment"][5:8], ep["state"][5:8], ep["frames"][5:8], ep["is_demo"][5:8])
        expected = memory.reconstruction_loss(encoded, ep["moment"][5:8])
        torch.testing.assert_close(result[1]["storage_reconstruction_loss"], expected)
        result[1]["storage_reconstruction_loss"].backward()
        self.assertGreater(float(ep["moment"].grad[5:8].abs().sum()), 0)
        self.assertEqual(float(ep["moment"].grad[:5].abs().sum()), 0)
        self.assertEqual(float(ep["moment"].grad[8:].abs().sum()), 0)

    def test_source_selection_none_and_suffix_reset(self):
        memory, ep = make_memory(source="short", capacity=20), make_episode()
        ep.pop("moment")
        open_fusion(memory)
        outputs = replay_queries(memory, ep, [3, 8], reset_before=3)
        self.assertTrue(torch.equal(outputs[3][0], ep["short"][3:4]))
        self.assertEqual(float(outputs[8][1]["retained_events"]), 5)
        self.assertEqual(float(outputs[8][1]["replayed_observations"]), 5)
        disabled = replay_queries(memory, ep, [8], mode="none")[8]
        self.assertTrue(torch.equal(disabled[0], ep["short"][8:9]))
        self.assertEqual(float(disabled[1]["storage_reconstruction_loss"]), 0)
        self.assertEqual(replay_state(memory, ep, 8, mode="none").shape[1], 0)

    def test_reencoding_uses_current_encoder_and_compatibility_argument_is_noop(self):
        memory, ep = make_memory(), make_episode()
        before = replay_state(memory, ep, 7)
        with torch.no_grad():
            memory.source_projection.weight.add_(torch.randn_like(memory.source_projection.weight))
        after = replay_state(memory, ep, 7)
        self.assertFalse(torch.equal(before[..., :-2], after[..., :-2]))
        self.assertTrue(torch.equal(before[..., -2:], after[..., -2:]))
        zero = replay_queries(memory, ep, [7], checkpoint_segment=0)[7]
        compat = replay_queries(memory, ep, [7], checkpoint_segment=3)[7]
        self.assertTrue(torch.equal(zero[0], compat[0]))
        self.assertTrue(torch.equal(zero[1]["storage_reconstruction_loss"], compat[1]["storage_reconstruction_loss"]))

    def test_replay_rejects_invalid_queries_chronology_and_demo_suffix(self):
        memory, ep = make_memory(), make_episode()
        for queries in ([1, 1], [True], [-1], [99]):
            with self.assertRaises(ValueError):
                replay_queries(memory, ep, queries)
        for kwargs in ({"reset_before": 5}, {"checkpoint_segment": -1}, {"mode": "recurrent"}):
            with self.assertRaises(ValueError):
                replay_queries(memory, ep, [4], **kwargs)
        for key, value in (("frames", 0), ("is_demo", True)):
            changed = copy.deepcopy(ep)
            changed[key][5] = value
            with self.assertRaises(ValueError):
                replay_queries(memory, changed, [6])
        self.assertEqual(replay_queries(memory, ep, []), {})


if __name__ == "__main__":
    unittest.main()
