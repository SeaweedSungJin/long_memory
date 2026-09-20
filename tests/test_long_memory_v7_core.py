"""CPU mechanical contracts, not evidence of robot-task improvement."""

import copy
import unittest

import torch
from torch import nn

from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from gr00t.long_memory.replay_v7 import apply_write, encode_at, initial_replay_state, replay_queries, replay_state


def make_memory(**kwargs):
    options = dict(feature_dim=12, state_dim=3, num_short_tokens=4, hidden_dim=8, capacity=6, num_heads=2)
    options.update(kwargs)
    return RecurrentMemoryV7(MemoryV7Config(**options))


def make_episode(count=13, demo=3):
    return {"short": torch.randn(count, 4, 12), "state": torch.randn(count, 3),
            "frames": torch.arange(count) * 16, "is_demo": torch.arange(count) < demo,
            "actions": torch.full((count, 3), float("nan")),
            "targets": torch.full((count, 4, 3), float("nan"))}


def open_fusion(memory):
    # The actual initialization intentionally blocks early reader gradients
    # behind a zero output projection. Test the learned regime separately.
    with torch.no_grad():
        memory.fusion_projection.weight.normal_(std=0.1)


def grad_size(parameters):
    return sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)


class V7CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2026)

    def test_config_validation(self):
        for field in ("feature_dim", "state_dim", "num_short_tokens", "hidden_dim", "capacity", "num_heads"):
            for value in (True, 0, -1, 2.0, "2"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    make_memory(**{field: value})
        for options in ({"hidden_dim": 7}, {"time_scale": float("nan")}, {"time_scale": 0},
                        {"update_init": 1}, {"update_init": -0.1}, {"update_init": True}):
            with self.assertRaises(ValueError):
                make_memory(**options)

    def test_shapes_and_fp32_inside_autocast(self):
        memory, data = make_memory(), make_episode()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            encoded = encode_at(memory, data, 0)
            state, _ = memory.write(memory.initial_state(), encoded)
            fused, _ = memory.read(data["short"][1:2], encode_at(memory, data, 1), state)
        self.assertEqual(encoded.shape, (1, 4, 8))
        self.assertEqual(state.shape, (1, 6, 8))
        self.assertEqual(fused.shape, (1, 4, 12))
        for value in (encoded, state, fused):
            self.assertEqual(value.dtype, torch.float32)

    def test_four_short_tokens_not_mean_pooled(self):
        memory, data = make_memory(), make_episode()
        first = encode_at(memory, data, 0)
        data["short"][0, 0, :2] += torch.tensor([4.0, -4.0])
        second = encode_at(memory, data, 0)
        self.assertFalse(torch.equal(first[:, 0], second[:, 0]))
        self.assertTrue(torch.equal(first[:, 1:], second[:, 1:]))

    def test_input_shape_frame_demo_nonfinite_validation(self):
        memory, data = make_memory(), make_episode()
        for mutation in (lambda d: d["short"][0].fill_(float("nan")),
                         lambda d: d["state"][0].fill_(float("inf")),
                         lambda d: d.update(is_demo=d["is_demo"].float()),
                         lambda d: d.update(frames=d["frames"].float() + 0.5),
                         lambda d: d.update(short=d["short"][:, :2])):
            changed = copy.deepcopy(data)
            mutation(changed)
            with self.assertRaises(ValueError):
                encode_at(memory, changed, 0)
        with self.assertRaises(TypeError):
            encode_at(memory.bfloat16(), data, 0)

    def test_zero_reset_slot_addresses_and_gate_initialization(self):
        memory, data = make_memory(), make_episode()
        before = memory.initial_state()
        addresses = memory.slot_addresses.detach().clone()
        candidate, metrics = memory.write(before, encode_at(memory, data, 0))
        self.assertTrue(torch.equal(before, torch.zeros_like(before)))
        self.assertTrue(torch.equal(addresses, memory.slot_addresses))
        self.assertGreater(float((candidate[:, 1:] - candidate[:, :1]).abs().max()), 1e-7)
        self.assertAlmostEqual(float(metrics["update_gate_mean"]), 0.02, places=7)
        self.assertTrue(torch.equal(memory.initial_state(), before))
        self.assertEqual(memory.attention.dropout, 0.0)
        self.assertIsNone(memory.attention.in_proj_bias)
        self.assertIsNone(memory.attention.out_proj.bias)

    def test_zero_init_conditioning_then_writer_gradient_after_opening(self):
        memory, data = make_memory(), make_episode()
        fused = replay_queries(memory, data, [8], checkpoint_segment=0)[8][0]
        self.assertTrue(torch.equal(fused, data["short"][8:9]))
        (fused * torch.randn_like(fused)).sum().backward()
        self.assertGreater(grad_size(memory.fusion_projection.parameters()), 0)
        self.assertEqual(grad_size(memory.write_ffn.parameters()), 0)
        memory.zero_grad(set_to_none=True)
        open_fusion(memory)
        fused = replay_queries(memory, data, [8], checkpoint_segment=0)[8][0]
        (fused * torch.randn_like(fused)).sum().backward()
        for module in (memory.write_ffn, memory.update_gate, memory.attention, memory.short_projection):
            self.assertGreater(grad_size(module.parameters()), 0)

    def test_empty_and_disabled_bypass_after_affine_parameters_change(self):
        memory, data = make_memory(), make_episode()
        open_fusion(memory)
        with torch.no_grad():
            for name, parameter in memory.named_parameters():
                if name.endswith("bias"):
                    parameter.normal_(std=1.0)
        short, encoded = data["short"][0:1], encode_at(memory, data, 0)
        state = memory.initial_state()
        for mode, enabled, bank in (("recurrent", True, state), ("recurrent", False, torch.ones_like(state)),
                                    ("archive", True, state[:, :0]), ("none", True, torch.ones_like(state))):
            fused, _ = memory.read(short, encoded, bank, mode=mode, memory_enabled=enabled)
            self.assertTrue(torch.equal(short, fused))

    def test_centered_fusion_has_no_affine_shortcut_when_read_is_zero(self):
        memory, data = make_memory(), make_episode()
        open_fusion(memory)
        with torch.no_grad():
            memory.attention.out_proj.weight.zero_()
            memory.read_ffn[-1].bias.fill_(11)
            memory.read_output_norm.bias.fill_(5)
        short = data["short"][0:1]
        fused, metrics = memory.read(short, encode_at(memory, data, 0), torch.randn_like(memory.initial_state()))
        self.assertTrue(torch.equal(short, fused))
        self.assertEqual(float(metrics["residual_norm"]), 0)

    def test_archive_uses_all_projected_short_tokens_without_slot_addresses(self):
        memory, data = make_memory(), make_episode()
        open_fusion(memory)
        state = replay_state(memory, data, 10, mode="archive", checkpoint_segment=0)
        expected = torch.cat([encode_at(memory, data, i) for i in range(10)], dim=1)
        self.assertTrue(torch.equal(state, expected))
        self.assertEqual(state.shape[1], 40)  # Deliberately exceeds 6 latent slots.
        first = memory.read(data["short"][10:11], encode_at(memory, data, 10), state, mode="archive")[0]
        with torch.no_grad():
            memory.slot_addresses.normal_(std=100)
        second = memory.read(data["short"][10:11], encode_at(memory, data, 10), state, mode="archive")[0]
        self.assertTrue(torch.equal(first, second))

    def test_stage_freezing_and_parameter_partition(self):
        memory = make_memory()
        self.assertEqual({id(p) for p in memory.actor_parameters()}, {id(p) for p in memory.parameters()})
        memory.set_stage(2)
        self.assertTrue(all(not p.requires_grad for p in memory.parameters()))
        memory.set_stage(1)
        self.assertTrue(all(p.requires_grad for p in memory.parameters()))
        with self.assertRaises(ValueError):
            memory.set_stage(3)

    def test_slot_spread_diagnostics_detect_small_differences(self):
        memory, data = make_memory(), make_episode()
        encoded, short = encode_at(memory, data, 0), data["short"][0:1]
        state = torch.ones_like(memory.initial_state())
        state[:, 0, 0] += 1e-4
        _, metrics = memory.read(short, encoded, state)
        for name in ("slot_rms_spread", "slot_relative_spread"):
            self.assertTrue(bool(torch.isfinite(metrics[name])))
            self.assertGreater(float(metrics[name]), 0)
            self.assertFalse(metrics[name].requires_grad)
            empty = memory.read(short, encoded, memory.initial_state())[1][name]
            self.assertEqual(float(empty), 0)

    def test_cvom_zero_anchor_and_gradient_isolation(self):
        memory, data = make_memory(), make_episode()
        critic = CVOMV7(memory.config)
        encoded = encode_at(memory, data, 2)
        state = replay_state(memory, data, 2, checkpoint_segment=0)
        candidate = memory.write(state, encoded)[0]
        self.assertTrue(torch.equal(critic(encoded, state, candidate, memory.slot_addresses), torch.zeros(1)))
        with torch.no_grad():
            critic.head[-1].weight.normal_()
        self.assertTrue(torch.equal(critic(encoded, state, state, memory.slot_addresses), torch.zeros(1)))
        prediction = critic(encoded, state, candidate, memory.slot_addresses)
        (prediction - 1.0).square().sum().backward()
        self.assertGreater(grad_size(critic.parameters()), 0)
        self.assertEqual(grad_size(memory.parameters()), 0)
        self.assertTrue(all(p.grad is None for p in memory.parameters()))

    def test_cvom_keep_is_bitwise_and_ties_update(self):
        memory, data = make_memory(), make_episode()
        state, encoded = memory.initial_state(), encode_at(memory, data, 0)

        class FixedCritic(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, x, *args):
                return x.new_full((x.shape[0],), self.value)

        kept, metrics = apply_write(memory, state, encoded, cvom=FixedCritic(-0.051), threshold=0.05)
        self.assertTrue(torch.equal(kept, state))
        self.assertEqual(float(metrics["keep_rate"]), 1)
        for gain in (-0.05, 0.0, 1.0):
            updated, _ = apply_write(memory, state, encoded, cvom=FixedCritic(gain), threshold=0.05)
            self.assertTrue(torch.equal(updated, memory.write(state, encoded)[0]))
        with self.assertRaises(ValueError):
            apply_write(memory, state, encoded, cvom=FixedCritic(float("nan")))
        with self.assertRaises(ValueError):
            apply_write(memory, state[:, :0], encoded, cvom=FixedCritic(0), mode="archive")


class V7ReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(21)

    def test_future_content_and_actions_are_not_memory_inputs(self):
        memory, data = make_memory(), make_episode()
        open_fusion(memory)
        first = replay_queries(memory, data, [7], checkpoint_segment=0)[7][0]
        changed = copy.deepcopy(data)
        changed["short"][8:] = float("nan")
        changed["state"][8:] = float("nan")
        changed["frames"][8:] = -999
        changed["is_demo"][8:] = True
        changed["actions"].fill_(1e20)
        changed["targets"].fill_(-1e20)
        second = replay_queries(memory, changed, [7], checkpoint_segment=0)[7][0]
        self.assertTrue(torch.equal(first, second))

    def test_read_before_current_write_and_online_parity(self):
        memory, data = make_memory(), make_episode()
        open_fusion(memory)
        offline = replay_queries(memory, data, [0, 2, 8, 12], checkpoint_segment=3)
        state = initial_replay_state(memory)
        with torch.no_grad():
            for index in range(13):
                encoded = encode_at(memory, data, index)
                if index in offline:
                    fused, _ = memory.read(data["short"][index:index + 1], encoded, state)
                    torch.testing.assert_close(fused, offline[index][0], rtol=0, atol=0)
                    torch.testing.assert_close(state, replay_state(memory, data, index), rtol=0, atol=0)
                state, _ = apply_write(memory, state, encoded)
        self.assertTrue(torch.equal(offline[0][0], data["short"][0:1]))
        original = replay_state(memory, data, 8, checkpoint_segment=0)
        data["short"][8].fill_(5)
        self.assertTrue(torch.equal(original, replay_state(memory, data, 8, checkpoint_segment=0)))

    def test_checkpoint_outputs_and_all_parameter_gradients_match(self):
        for mode in ("recurrent", "archive"):
            with self.subTest(mode=mode):
                plain, data = make_memory(), make_episode()
                open_fusion(plain)
                recomputed = copy.deepcopy(plain)
                targets = [torch.randn(1, 4, 12), torch.randn(1, 4, 12)]
                reference = replay_queries(plain, data, [4, 12], mode=mode, checkpoint_segment=0)
                checked = replay_queries(recomputed, data, [4, 12], mode=mode, checkpoint_segment=3)
                for index in (4, 12):
                    torch.testing.assert_close(reference[index][0], checked[index][0], rtol=0, atol=0)
                sum((reference[i][0] * targets[j]).sum() for j, i in enumerate((4, 12))).backward()
                sum((checked[i][0] * targets[j]).sum() for j, i in enumerate((4, 12))).backward()
                for (name, p), (other_name, q) in zip(plain.named_parameters(), recomputed.named_parameters()):
                    self.assertEqual(name, other_name)
                    self.assertEqual(p.grad is None, q.grad is None, name)
                    if p.grad is not None:
                        torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7, msg=name)

    def test_specific_early_demo_write_receives_far_future_gradient(self):
        memory, data = make_memory(), make_episode(count=31)
        open_fusion(memory)
        data["short"].requires_grad_(True)
        captured_logits, captured_values = [], []
        hooks = [memory.update_gate.register_forward_hook(lambda module, args, output: captured_logits.append(output)),
                 memory.write_ffn.register_forward_hook(lambda module, args, output: captured_values.append(output))]
        fused = replay_queries(memory, data, [30], checkpoint_segment=0)[30][0]
        for hook in hooks:
            hook.remove()
        loss = (fused * torch.randn_like(fused)).sum()
        # Aggregate parameter gradients are insufficient: the exact *first*
        # demo timestep's gate and candidate nodes must receive future credit.
        gradients = torch.autograd.grad(loss, (captured_logits[0], captured_values[0], data["short"]), retain_graph=True)
        for gradient in gradients[:2]:
            self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertGreater(float(gradients[2][0].abs().sum()), 0)
        self.assertGreater(float(gradients[2][7].abs().sum()), 0)  # Unselected execution observation.

    def test_early_state_gradient_survives_checkpoint_boundaries(self):
        memory, data = make_memory(), make_episode(count=31)
        open_fusion(memory)
        data["short"].requires_grad_(True)
        fused = replay_queries(memory, data, [30], checkpoint_segment=4)[30][0]
        (fused * torch.randn_like(fused)).sum().backward()
        self.assertGreater(float(data["short"].grad[0].abs().sum()), 0)
        self.assertGreater(grad_size(memory.update_gate.parameters()), 0)
        self.assertGreater(grad_size(memory.write_ffn.parameters()), 0)

    def test_reset_suffix_matches_explicit_state_reset(self):
        memory, data = make_memory(), make_episode()
        state = memory.initial_state()
        for index in range(6, 11):
            state, _ = memory.write(state, encode_at(memory, data, index))
        checked = replay_state(memory, data, 11, reset_before=6, checkpoint_segment=2)
        self.assertTrue(torch.equal(state, checked))
        empty = replay_queries(memory, data, [8], reset_before=8)[8][0]
        self.assertTrue(torch.equal(empty, data["short"][8:9]))

    def test_duplicate_unsorted_frames_bad_demo_and_queries_rejected(self):
        for mutate in (lambda d: d["frames"].__setitem__(2, d["frames"][1]),
                       lambda d: d["is_demo"].__setitem__(5, True),
                       lambda d: d.pop("is_demo")):
            memory, data = make_memory(), make_episode()
            mutate(data)
            with self.assertRaises(ValueError):
                replay_queries(memory, data, [8])
        memory, data = make_memory(), make_episode()
        for queries in ([2, 2], [-1], [99], [1.0]):
            with self.assertRaises(ValueError):
                replay_queries(memory, data, queries)
        with self.assertRaises(ValueError):
            replay_queries(memory, data, [2], reset_before=3)
        with self.assertRaises(ValueError):
            replay_state(memory, data, 3, checkpoint_segment=-1)

    def test_empty_queries_and_none_control(self):
        memory, data = make_memory(), make_episode()
        self.assertEqual(replay_queries(memory, data, []), {})
        open_fusion(memory)
        result = replay_queries(memory, data, [0, 9], mode="none")
        for index in (0, 9):
            self.assertTrue(torch.equal(result[index][0], data["short"][index:index + 1]))


if __name__ == "__main__":
    unittest.main()
