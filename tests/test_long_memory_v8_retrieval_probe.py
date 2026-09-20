"""CPU contracts for the standalone V8 retrieval diagnostic, not task success."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from gr00t.long_memory.event_v8 import EventMemoryV8, MemoryV8Config
from gr00t.long_memory.replay_v8 import encode_at, replay_state


_SCRIPT = Path(__file__).resolve().parents[1] / "run_scripts/robomme/diagnose_v8_event_retrieval.py"
_SPEC = importlib.util.spec_from_file_location("v8_retrieval_probe_test_module", _SCRIPT)
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


def make_memory():
    memory = EventMemoryV8(MemoryV8Config(
        feature_dim=12, state_dim=3, num_short_tokens=4,
        hidden_dim=8, capacity=5, num_heads=2, source="moment",
    ))
    # Zero-initialized fusion would hide a broken intervention comparison.
    with torch.no_grad():
        memory.fusion_projection.weight.normal_(std=0.15)
        memory.fusion_gate.weight.normal_(std=0.1)
    return memory


def make_episode(count=10):
    return {
        "moment": torch.randn(count, 4, 12),
        "short": torch.randn(count, 4, 12),
        "state": torch.randn(count, 3),
        "frames": torch.arange(count) * 16,
        "is_demo": torch.arange(count) < 2,
    }


class ConditioningOnlyEpisode(dict):
    """Even inspecting an action/target field is a diagnostic contract failure."""

    forbidden = {"actions", "targets", "target_mask", "action_mask"}

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"Diagnostic accessed forbidden field {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"Diagnostic accessed forbidden field {key}")
        return super().get(key, default)

    def __contains__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"Diagnostic inspected forbidden field {key}")
        return super().__contains__(key)


def hook_state(memory):
    return [
        (dict(module._forward_pre_hooks), dict(module._forward_hooks))
        for module in memory.modules()
    ]


class V8RetrievalProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)

    def setUp(self):
        previous_rng = torch.get_rng_state().clone()
        self.addCleanup(torch.set_rng_state, previous_rng)
        torch.manual_seed(2026)
        self.memory = make_memory().eval()
        self.episode = make_episode()
        self.decision = 7
        with torch.no_grad():
            self.bank = replay_state(self.memory, self.episode, self.decision)
            self.encoded = encode_at(self.memory, self.episode, self.decision)
        self.short = self.episode["short"][self.decision:self.decision + 1]

    def test_content_interventions_preserve_metadata_and_input(self):
        original = self.bank.clone()
        for mode in ("mean", "shuffle"):
            with self.subTest(mode=mode):
                changed = probe.content_intervention(
                    self.bank, self.memory.config.num_short_tokens, mode=mode,
                    generator=torch.Generator(device="cpu").manual_seed(31),
                )
                self.assertEqual(changed.shape, self.bank.shape)
                self.assertEqual(changed.dtype, self.bank.dtype)
                self.assertEqual(changed.device, self.bank.device)
                self.assertTrue(torch.equal(changed[..., -2:], original[..., -2:]))
                self.assertTrue(torch.equal(self.bank, original))
                self.assertNotEqual(changed.data_ptr(), self.bank.data_ptr())
                torch.testing.assert_close(
                    changed[..., :-2].mean(1), original[..., :-2].mean(1),
                    rtol=1e-6, atol=1e-6,
                )
                if mode == "mean":
                    torch.testing.assert_close(
                        changed[..., :-2],
                        original[..., :-2].mean(1, keepdim=True).expand_as(original[..., :-2]),
                        rtol=0, atol=0,
                    )

    def test_shuffle_moves_whole_event_content_without_changing_token_order(self):
        changed = probe.content_intervention(
            self.bank, self.memory.config.num_short_tokens, mode="shuffle",
            generator=torch.Generator(device="cpu").manual_seed(31),
        )
        old_events = self.bank[..., :-2].reshape(5, 4, 8)
        new_events = changed[..., :-2].reshape(5, 4, 8)
        matched = []
        for event in new_events:
            indices = [i for i, original in enumerate(old_events) if torch.equal(event, original)]
            self.assertEqual(len(indices), 1)
            matched.extend(indices)
        self.assertEqual(sorted(matched), list(range(5)))
        self.assertNotEqual(matched, list(range(5)))

    def test_captured_read_matches_production_read_and_returns_normalized_attention(self):
        before_hooks = hook_state(self.memory)
        with torch.no_grad():
            expected, expected_metrics = self.memory.read(self.short, self.encoded, self.bank)
            actual, actual_metrics, attentions = probe.captured_read(
                self.memory, self.short, self.encoded, self.bank,
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual_metrics.keys(), expected_metrics.keys())
        for key in expected_metrics:
            torch.testing.assert_close(actual_metrics[key], expected_metrics[key], rtol=0, atol=0)
        self.assertEqual(len(attentions), len(self.memory.read_blocks))
        for weights in attentions:
            self.assertEqual(weights.shape, (1, 2, 4, self.bank.shape[1]))
            self.assertTrue(bool((weights >= 0).all()))
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
        self.assertEqual(hook_state(self.memory), before_hooks)

    def test_uniform_attention_is_projected_value_mean_and_ignores_queries_and_keys(self):
        queries = torch.randn(1, 4, 8)
        keys = torch.randn(1, 20, 8)
        values = torch.randn(1, 20, 8)
        observed = []

        def direct_attention_read(*args, **kwargs):
            observed.clear()
            for block in self.memory.read_blocks:
                output, _ = block.attention(
                    queries, keys, values, need_weights=True, average_attn_weights=False,
                )
                observed.append(output.detach().clone())
            return observed[-1], {}

        with mock.patch.object(self.memory, "read", side_effect=direct_attention_read):
            with torch.no_grad():
                _, _, attentions = probe.captured_read(
                    self.memory, self.short, self.encoded, self.bank, uniform=True,
                )
                first_outputs = [output.clone() for output in observed]
                queries = queries * -31 + 14
                keys = keys.flip(1) * 79 - 50
                probe.captured_read(
                    self.memory, self.short, self.encoded, self.bank, uniform=True,
                )
        for index, block in enumerate(self.memory.read_blocks):
            attention = block.attention
            width = attention.embed_dim
            value_bias = (None if attention.in_proj_bias is None
                          else attention.in_proj_bias[2 * width:])
            projected_values = F.linear(values, attention.in_proj_weight[2 * width:], value_bias)
            expected = F.linear(
                projected_values.mean(1, keepdim=True),
                attention.out_proj.weight, attention.out_proj.bias,
            ).expand(-1, queries.shape[1], -1)
            torch.testing.assert_close(first_outputs[index], expected, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(observed[index], first_outputs[index], rtol=0, atol=0)
            torch.testing.assert_close(
                attentions[index], torch.full_like(attentions[index], 1 / keys.shape[1]),
                rtol=0, atol=0,
            )

    def test_capture_preserves_existing_hooks_and_cleans_up_after_failure(self):
        existing = self.memory.read_blocks[0].attention.register_forward_hook(
            lambda module, args, result: None,
        )
        self.addCleanup(existing.remove)
        before_hooks = hook_state(self.memory)
        for uniform in (False, True):
            with self.subTest(uniform=uniform):
                with torch.no_grad():
                    probe.captured_read(
                        self.memory, self.short, self.encoded, self.bank, uniform=uniform,
                    )
                self.assertEqual(hook_state(self.memory), before_hooks)
                with mock.patch.object(
                    self.memory.read_blocks[1].attention, "forward",
                    side_effect=RuntimeError("deliberate second attention failure"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "deliberate second attention failure"):
                        probe.captured_read(
                            self.memory, self.short, self.encoded, self.bank, uniform=uniform,
                        )
                self.assertEqual(hook_state(self.memory), before_hooks)

    def test_reads_leave_weights_rng_and_all_inputs_unchanged(self):
        original_weights = {name: value.clone() for name, value in self.memory.state_dict().items()}
        original_inputs = [value.clone() for value in (self.bank, self.encoded, self.short)]
        original_rng = torch.get_rng_state().clone()
        with torch.no_grad():
            expected = self.memory.read(self.short, self.encoded, self.bank)[0]
            for uniform in (False, True):
                probe.captured_read(
                    self.memory, self.short, self.encoded, self.bank, uniform=uniform,
                )
            actual = self.memory.read(self.short, self.encoded, self.bank)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, value in self.memory.state_dict().items():
            self.assertTrue(torch.equal(value, original_weights[name]), name)
        for value, original in zip((self.bank, self.encoded, self.short), original_inputs):
            self.assertTrue(torch.equal(value, original))
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))

    def test_probe_accepts_only_conditioning_fields_and_returns_finite_json(self):
        result = probe.probe_query(self.memory, self.episode, self.decision, seed=31)
        self.assertIsInstance(result, dict)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        for name in (
            "content_relative_spread", "uniform_fused_delta_over_residual",
            "mean_fused_delta_over_residual", "query_attention_tv",
        ):
            self.assertIsInstance(result[name], (float, int), name)
            self.assertGreaterEqual(result[name], 0, name)
        self.assertTrue(result["metadata_preserved"])
        self.assertGreater(result["content_relative_spread"], 0)

    def test_probe_never_reads_actions_targets_or_invalid_future(self):
        expected = probe.probe_query(self.memory, self.episode, self.decision, seed=31)
        changed = ConditioningOnlyEpisode(copy.deepcopy(self.episode))
        for key in ("moment", "short", "state"):
            changed[key][self.decision + 1:] = float("nan")
            # Additional invalid endpoints cannot affect an earlier query.
            changed[key] = torch.cat((changed[key], changed[key][-1:].clone()), dim=0)
        changed["frames"][self.decision + 1:] = -999
        changed["frames"] = torch.cat((changed["frames"], torch.tensor([-1000])))
        changed["is_demo"][self.decision + 1:] = True
        changed["is_demo"] = torch.cat((changed["is_demo"], torch.tensor([True])))
        for key in changed.forbidden:
            changed[key] = object()
        actual = probe.probe_query(self.memory, changed, self.decision, seed=31)
        self.assertEqual(actual, expected)

    def test_probe_preserves_reader_encoder_rng_inputs_gradients_and_modes(self):
        first_parameter = next(self.memory.parameters())
        first_parameter.grad = torch.ones_like(first_parameter)
        original_grad = first_parameter.grad.clone()
        original_weights = {name: value.clone() for name, value in self.memory.state_dict().items()}
        original_episode = {key: value.clone() for key, value in self.episode.items()}
        original_modes = [module.training for module in self.memory.modules()]
        original_requires_grad = [parameter.requires_grad for parameter in self.memory.parameters()]
        original_hooks = hook_state(self.memory)
        original_rng = torch.get_rng_state().clone()
        original_bank = self.bank.clone()
        with mock.patch.object(probe, "replay_state", return_value=self.bank):
            first = probe.probe_query(self.memory, self.episode, self.decision, seed=31)
            second = probe.probe_query(self.memory, self.episode, self.decision, seed=31)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(self.bank, original_bank))
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        for name, value in self.memory.state_dict().items():
            self.assertTrue(torch.equal(value, original_weights[name]), name)
        for key, value in self.episode.items():
            self.assertTrue(torch.equal(value, original_episode[key]), key)
        self.assertTrue(torch.equal(first_parameter.grad, original_grad))
        self.assertEqual([module.training for module in self.memory.modules()], original_modes)
        self.assertEqual(
            [parameter.requires_grad for parameter in self.memory.parameters()],
            original_requires_grad,
        )
        self.assertEqual(hook_state(self.memory), original_hooks)

    def test_training_mode_is_rejected_without_changing_model_or_hooks(self):
        self.memory.train()
        original_hooks = hook_state(self.memory)
        with self.assertRaisesRegex(ValueError, "eval"):
            probe.probe_query(self.memory, self.episode, self.decision, seed=31)
        with self.assertRaisesRegex(ValueError, "eval"):
            probe.captured_read(self.memory, self.short, self.encoded, self.bank)
        self.assertTrue(self.memory.training)
        self.assertTrue(all(module.training for module in self.memory.modules()))
        self.assertEqual(hook_state(self.memory), original_hooks)


if __name__ == "__main__":
    unittest.main()
