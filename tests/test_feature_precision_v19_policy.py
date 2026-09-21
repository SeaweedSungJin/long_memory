"""CPU precision scope / native regression contracts; no real GPU or robot."""
from __future__ import annotations

import copy
import inspect
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from run_scripts.robomme import feature_precision_v19 as precision
from run_scripts.robomme import policy_representation_v18 as policy_module
from run_scripts.robomme import serve_representation_v18 as server
from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18, RepresentationMemoryV18
from tests import test_policy_visual_patch_v11 as tiny
from tests import test_long_memory_online_policy as fixture


def previous_native_extract(model, head, inputs, n_q, *unused):
    """Literal pre-change producer, independent of the new helper."""
    raw = model.backbone(inputs)
    moment = head.vlln(raw["backbone_features"])[:, -n_q:].float()
    return head.process_backbone_output(raw, action_inputs_B=1), moment


class _FloatHead(tiny.CPUHead):
    def __init__(self):
        super().__init__()
        self.autocast_events = []

    def vlln(self, value):
        self.autocast_events.append(("vlln", torch.is_autocast_enabled("cpu")))
        # Deliberately non-BF16-representable output, including the non-short
        # prefix. Checking the tail only would miss an incomplete boundary.
        return value.float() * 1.234567 + .1234567

    def process_backbone_output(self, *args, **kwargs):
        self.autocast_events.append(("process", torch.is_autocast_enabled("cpu")))
        return super().process_backbone_output(*args, **kwargs)

    def state_encoder(self, *args):
        self.autocast_events.append(("state_encoder", torch.is_autocast_enabled("cpu")))
        return super().state_encoder(*args)

    def get_action_with_features(self, *args):
        self.autocast_events.append(("denoising", torch.is_autocast_enabled("cpu")))
        return super().get_action_with_features(*args)


class _FloatModel(tiny.CPUModel):
    def __init__(self):
        self.action_head = _FloatHead().eval().requires_grad_(False)
        self.autocast_events = []

    def backbone(self, inputs):
        self.autocast_events.append(torch.is_autocast_enabled("cpu"))
        return super().backbone(inputs)


def actor(mode="native"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91919)
        result = policy_module.RepresentationPolicyV18.__new__(policy_module.RepresentationPolicyV18)
        result.__dict__.update(tiny.parent_policy().__dict__)
        result.model = _FloatModel()
        result.representation = RepresentationMemoryV18(RepresentationConfigV18(
            feature_dim=8, state_dim=8, num_short_tokens=4, short_window=3,
            hidden_dim=16, num_heads=4, capacity_events=3,
        )).eval().requires_grad_(False)
        with torch.no_grad():
            result.representation.memory.fusion_projection.weight.normal_(std=.03)
        result.mode, result.write_policy = "representation_v18", "fifo"
        result.feature_precision = mode
        result.feature_precision_rules = precision.feature_precision_contract(mode)
        result.writer = result.writer_callback = result.writer_sha256 = None
        result.checkpoint_step = 6072
        result.payload_sha256 = {"model.safetensors": "a" * 64, "expert.safetensors": "b" * 64}
    return result


def calls(policy):
    outputs = []
    for index, (frame, passive) in enumerate(((0, True), (2, True), (4, False), (6, False), (8, False))):
        controls = np.zeros((2, 8), np.float32) if index > 2 else None
        output, info = policy.get_action(fixture._observation(marker=index + 1),
            fixture._options(frame=frame, passive=passive, controls=controls))
        outputs.append((copy.deepcopy(output), copy.deepcopy(info),
                        policy.sessions["A"].short_cache.clone(), policy.sessions["A"].long_bank.clone(),
                        policy.sessions["A"].generator.get_state().clone()))
    return outputs


class FeaturePrecisionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_contract_is_strict_json_and_does_not_change_ae_rules(self):
        for value in (None, True, 0, "bf16", "Native", []):
            with self.assertRaises(ValueError):
                precision.feature_precision_contract(value)
        native = precision.feature_precision_contract("native")
        aligned = precision.feature_precision_contract("cache-aligned")
        self.assertEqual(json.loads(json.dumps(aligned)), aligned)
        self.assertEqual(native["autocast_scope"], [])
        self.assertEqual(aligned["entire_backbone_features_boundary"], "bfloat16")
        for key in ("ae_state_encoder", "ae_denoising", "backbone_attention_and_image_masks",
                    "sampling_seed_steps_cadence_reset_and_action_execution", "external_memory_compute",
                    "hamlet_recurrent_cache"):
            self.assertEqual(native[key], aligned[key])

    def test_default_is_explicit_native_in_policy_and_server(self):
        sig = inspect.signature(policy_module.RepresentationPolicyV18)
        self.assertEqual(sig.parameters["feature_precision"].default, "native")
        args = server.build_parser().parse_args(["--base-model", "base", "--checkpoint", "ckpt"])
        self.assertEqual(args.feature_precision, "native")
        args = server.build_parser().parse_args(["--base-model", "base", "--checkpoint", "ckpt",
                                               "--feature-precision", "cache-aligned"])
        self.assertEqual(args.feature_precision, "cache-aligned")
        with patch.object(policy_module, "checkpoint_info_v18") as preflight:
            with self.assertRaises(ValueError):
                policy_module.RepresentationPolicyV18("base", "ckpt", feature_precision="all-autocast")
        preflight.assert_not_called()

    def test_helper_default_and_explicit_native_equal_original_including_private_cache(self):
        for option in (None, "native"):
            original, current = _FloatModel(), _FloatModel()
            before = torch.get_rng_state().clone()
            for index in range(5):
                expected, moment = previous_native_extract(original, original.action_head, {"marker": index}, 4)
                args = () if option is None else (option,)
                actual, actual_moment = precision.extract_hamlet_features(current, current.action_head,
                    {"marker": index}, 4, *args)
                for key in expected:
                    self.assertEqual(actual[key].dtype, expected[key].dtype)
                    self.assertTrue(torch.equal(actual[key], expected[key]))
                self.assertTrue(torch.equal(actual_moment, moment))
                self.assertTrue(torch.equal(current.action_head._memory_cache, original.action_head._memory_cache))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertFalse(any(current.autocast_events))

    def test_aligned_matches_cache_rules_for_full_features_not_only_short(self):
        expected_model, actual_model = _FloatModel(), _FloatModel()
        for index in range(5):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                expected, expected_moment = previous_native_extract(expected_model, expected_model.action_head,
                                                                    {"marker": index}, 4)
                stored_features = expected["backbone_features"].to(torch.bfloat16)
                stored_moment = expected_moment.to(torch.bfloat16)
            actual, actual_moment = precision.extract_hamlet_features(actual_model, actual_model.action_head,
                {"marker": index}, 4, "cache-aligned")
            self.assertEqual(actual["backbone_features"].dtype, torch.bfloat16)
            self.assertTrue(torch.equal(actual["backbone_features"], stored_features))
            self.assertTrue(torch.equal(actual_moment, stored_moment.float()))
            self.assertFalse(torch.equal(expected["backbone_features"][:, :-4], stored_features[:, :-4].float()))
            for key in ("image_mask", "backbone_attention_mask"):
                self.assertTrue(torch.equal(actual[key], expected[key]))
                self.assertEqual(actual[key].dtype, expected[key].dtype)
            self.assertTrue(torch.equal(actual_model.action_head._memory_cache, expected_model.action_head._memory_cache))
            # Internal recurrent state remains unrounded, just as cache.py.
            self.assertEqual(actual_model.action_head._memory_cache.dtype, torch.float32)
        self.assertTrue(all(actual_model.autocast_events))
        self.assertFalse(torch.is_autocast_enabled("cpu"))

    def test_precision_context_does_not_leak_on_feature_failure(self):
        model = SimpleNamespace(device=torch.device("cpu"))
        def fail(_):
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            raise RuntimeError("injected")
        model.backbone = fail
        with self.assertRaisesRegex(RuntimeError, "injected"):
            precision.extract_hamlet_features(model, None, None, 4, "cache-aligned")
        self.assertFalse(torch.is_autocast_enabled("cpu"))

    def test_full_native_policy_matches_old_producer_actions_sessions_masks_and_rng(self):
        reference, current = actor(), actor()
        with patch.object(policy_module, "extract_hamlet_features", previous_native_extract):
            old = calls(reference)
        new = calls(current)
        for expected, actual in zip(old, new):
            for key in expected[0]:
                np.testing.assert_array_equal(expected[0][key], actual[0][key])
            self.assertEqual(expected[1], actual[1])
            for left, right in zip(expected[2:], actual[2:]):
                self.assertTrue(torch.equal(left, right))
        self.assertEqual(current.model.action_head.denoiser_calls, 3)
        self.assertEqual(current.sessions["A"].long_bank.shape[1], 3 * 4)

    def test_aligned_policy_keeps_memory_state_encoder_and_denoising_outside_feature_autocast(self):
        policy = actor("cache-aligned")
        original = policy.representation.step
        records = []
        def step(short, moment, state, *args, **kwargs):
            records.append((short.dtype, moment.dtype, state.dtype, torch.is_autocast_enabled("cpu"),
                            kwargs["event_index"], kwargs["read_enabled"]))
            return original(short, moment, state, *args, **kwargs)
        with patch.object(policy.representation, "step", step):
            outputs = calls(policy)
        self.assertEqual(records, [(torch.float32, torch.float32, torch.float32, False, i, i >= 2)
                                   for i in range(5)])
        for name, active in policy.model.action_head.autocast_events:
            self.assertEqual(active, name in ("vlln", "process"))
        self.assertTrue(all(policy.model.autocast_events))
        self.assertEqual(policy.model.action_head.last_features.dtype, torch.bfloat16)
        self.assertEqual(policy.model.action_head.last_state.dtype, torch.bfloat16)
        for _, info, *_ in outputs:
            self.assertEqual(info["feature_precision"], "cache-aligned")
            self.assertEqual(info["feature_precision_rules"], precision.feature_precision_contract("cache-aligned"))
        policy.reset()
        self.assertFalse(policy.sessions)
        self.assertIsNone(policy.model.action_head._memory_cache)
        repeated = calls(policy)
        for before, after in zip(outputs, repeated):
            for key in before[0]:
                np.testing.assert_array_equal(before[0][key], after[0][key])
            self.assertEqual(before[1], after[1])


if __name__ == "__main__":
    unittest.main()
