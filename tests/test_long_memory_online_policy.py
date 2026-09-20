"""Session lifecycle and adapter wiring using tiny deterministic CPU doubles.

These tests exercise LongMemoryPolicy itself, without loading the multi-GB base
checkpoint, contacting a server, or importing the RoboMME simulator. The fake
expert records its conditioning and consumes the assigned generator exactly once
per action call; it intentionally does not pretend to test robot performance.
"""
from collections import OrderedDict
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.long_memory.core import EpisodicMemory, MemoryConfig
from gr00t.long_memory.online_policy import LongMemoryPolicy


class _Processor:
    max_action_dim = max_state_dim = 8

    def __init__(self):
        self.modality_configs = {"new_embodiment": {
            "video": ModalityConfig([0], ["front_view", "wrist_view"]),
            "state": ModalityConfig([0], ["joint_position", "gripper_position"]),
            "action": ModalityConfig(list(range(4)), ["joint_position", "gripper_close"]),
            "language": ModalityConfig([0], ["task"]),
        }}
        self.state_action_processor = SimpleNamespace(apply_action=self._normalize)
        self.normalization_states = []

    def _normalize(self, action, embodiment, state):
        self.normalization_states.append(state)
        return {name: np.clip(value, -1, 1) for name, value in action.items()}

    def __call__(self, messages):
        step = messages[0]["content"]
        state = np.concatenate([step.states["joint_position"], step.states["gripper_position"]], axis=-1)
        return {"state": torch.from_numpy(state.copy()),
                "marker": float(step.images["front_view"].reshape(-1)[0])}

    @staticmethod
    def collator(processed):
        return {"inputs": processed[0]}

    @staticmethod
    def decode_action(prediction, embodiment, states):
        return {"joint_position": prediction[..., :7], "gripper_close": prediction[..., 7:8]}


class _Head:
    action_horizon = 4
    action_dim = 8

    def __init__(self):
        self._memory_cache = self._vision_cache = self._inference_gen = None
        self.process_calls = self.denoiser_calls = 0
        self.last_processed = self.last_features = self.last_state = None
        self.fail_prediction = False

    @staticmethod
    def vlln(features):
        return features + 100

    def process_backbone_output(self, backbone, action_inputs_B):
        self.process_calls += 1
        assert action_inputs_B == 1
        features = self.vlln(backbone["backbone_features"])
        moment = features[:, -2:]
        self._memory_cache = (moment.repeat(1, 3, 1) if self._memory_cache is None else
                              torch.cat([self._memory_cache[:, 2:], moment], dim=1))
        short = self._memory_cache.reshape(1, 3, 2, 6).mean(dim=1)
        backbone["backbone_features"] = torch.cat([features[:, :-2], short], dim=1)
        self.last_processed = backbone["backbone_features"].clone()
        return backbone

    def state_encoder(self, state, embodiment):
        self.last_state = state.clone()
        return state

    def get_action_with_features(self, features, state_features, embodiment, backbone):
        self.denoiser_calls += 1
        self.last_features = features.clone()
        prediction = torch.randn((1, self.action_horizon, self.action_dim),
                                 dtype=features.dtype, generator=self._inference_gen)
        if self.fail_prediction:
            prediction.fill_(torch.nan)
        return {"action_pred": prediction}

    def reset_memory(self):
        self._memory_cache = self._vision_cache = None


class _Model:
    device = torch.device("cpu")

    def __init__(self):
        self.action_head = _Head()

    @staticmethod
    def prepare_input(batch):
        return batch, SimpleNamespace(state=batch["state"][None].to(torch.bfloat16),
                                      embodiment_id=torch.zeros(1, dtype=torch.long))

    @staticmethod
    def backbone(batch):
        features = (torch.arange(30).reshape(1, 5, 6) + batch["marker"]).to(torch.bfloat16)
        return {"backbone_features": features,
                "backbone_attention_mask": torch.ones(1, 5, dtype=torch.bool),
                "image_mask": torch.tensor([[True, True, False, False, False]])}


def _policy(memory=False):
    """Bypass only large-model construction; all real adapter methods stay live."""
    policy = LongMemoryPolicy.__new__(LongMemoryPolicy)
    policy.strict = True
    policy.model, policy.processor = _Model(), _Processor()
    policy.embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    policy.modality_configs = policy.processor.modality_configs["new_embodiment"]
    policy.language_key = "task"
    policy.collate_fn = policy.processor.collator
    policy.stride, policy.n_q = 2, 2
    policy.memory, policy.stage, policy.write_policy = None, 0, "baseline"
    if memory:
        policy.memory = EpisodicMemory(MemoryConfig(feature_dim=6, state_dim=8, action_dim=8,
            hidden_dim=8, key_dim=4, value_dim=5, capacity=4, min_fill=1))
        # A controlled residual reveals whether the adapter changes only the tail.
        with torch.no_grad():
            policy.memory.fusion[-1].bias.fill_(2.0)
            policy.memory.token_gate.weight.zero_()
            policy.memory.token_gate.bias.zero_()
        policy.memory.eval().requires_grad_(False)
        policy.stage, policy.write_policy = 1, "all"
    policy.sessions, policy.session_cap = OrderedDict(), 64
    return policy


def _observation(marker=1, state=0.1234567, batch=1):
    return {"video": {name: np.full((batch, 1, 2, 2, 3), marker, np.uint8)
                      for name in ("front_view", "wrist_view")},
            "state": {"joint_position": np.full((batch, 1, 7), state, np.float32),
                      "gripper_position": np.full((batch, 1, 1), state, np.float32)},
            "language": {"task": [["move the cube"] for _ in range(batch)]}}


def _options(session="A", frame=0, seed=17, passive=False, controls=None, reset=False):
    return {"session_ids": [session], "reset_memory": [reset], "frame_index": frame,
            "episode_seed": seed, "prime_only": passive, "passive": passive,
            "executed_actions": controls}


def _actions(policy, **options):
    return policy.get_action(_observation(), _options(**options))[0]["joint_position"]


class TestOnlinePolicy(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_baseline_features_processed_once_and_not_replaced(self):
        policy = _policy()
        policy.get_action(_observation(), _options())
        head = policy.model.action_head
        self.assertEqual(head.process_calls, 1)
        self.assertEqual(head.denoiser_calls, 1)
        self.assertTrue(torch.equal(head.last_features, head.last_processed))
        expected_prefix = _Model.backbone({"marker": 1})["backbone_features"][:, :-2] + 100
        self.assertTrue(torch.equal(head.last_features[:, :-2], expected_prefix))
        self.assertIsNone(head._memory_cache)
        self.assertIsNone(head._inference_gen)
        self.assertIsNotNone(policy.sessions["A"].short_cache)

    def test_prime_only_does_not_consume_action_noise(self):
        policy = _policy()
        expected_rng = torch.Generator().manual_seed(17).get_state()
        for frame in (0, 2):
            actions = _actions(policy, frame=frame, passive=True)
            np.testing.assert_array_equal(actions, np.zeros_like(actions))
            self.assertTrue(torch.equal(policy.sessions["A"].generator.get_state(), expected_rng))
        self.assertEqual(policy.model.action_head.denoiser_calls, 0)
        first = _actions(policy, frame=4)
        fresh = _actions(_policy())
        np.testing.assert_array_equal(first, fresh)

    def test_interleaved_sessions_do_not_share_rng_or_short_cache(self):
        isolated, interleaved = _policy(), _policy()
        controls = np.zeros((2, 8), np.float32)
        expected_first = _actions(isolated)
        expected_second = _actions(isolated, frame=2, controls=controls)
        first = _actions(interleaved)
        original_cache = interleaved.sessions["A"].short_cache.clone()
        interleaved.get_action(_observation(marker=99), _options(session="B", seed=83))
        self.assertTrue(torch.equal(interleaved.sessions["A"].short_cache, original_cache))
        second = _actions(interleaved, frame=2, controls=controls)
        np.testing.assert_array_equal(first, expected_first)
        np.testing.assert_array_equal(second, expected_second)

    def test_session_reset_restarts_rng_and_clears_only_target(self):
        policy = _policy(memory=True)
        expected = _actions(policy)
        _actions(policy, session="B", seed=83)
        _actions(policy, frame=2, controls=np.zeros((2, 8), np.float32))
        self.assertEqual(policy.sessions["A"].bank.attempted, 1)
        self.assertEqual(policy.reset({"session_ids": ["A"]}), {"cleared_sessions": 1})
        self.assertEqual(list(policy.sessions), ["B"])
        np.testing.assert_array_equal(_actions(policy), expected)
        self.assertEqual(policy.sessions["A"].bank.attempted, 0)
        self.assertEqual(policy.reset(), {"cleared_sessions": 2})
        self.assertFalse(policy.sessions)

    def test_memory_changes_only_tail_after_completed_event(self):
        policy = _policy(memory=True)
        _, first = policy.get_action(_observation(), _options(passive=True))
        self.assertEqual(first["long_memory"]["bank_fill"], 0)
        policy.get_action(_observation(), _options(frame=2))
        head = policy.model.action_head
        self.assertEqual(head.process_calls, 2)
        self.assertTrue(torch.equal(head.last_features[:, :-2], head.last_processed[:, :-2]))
        expected_tail = (head.last_processed[:, -2:].float() + 1.0).to(torch.bfloat16)
        self.assertTrue(torch.equal(head.last_features[:, -2:], expected_tail))
        self.assertFalse(torch.equal(head.last_features[:, -2:], head.last_processed[:, -2:]))
        previous = policy.sessions["A"].bank.previous
        self.assertTrue(torch.equal(previous.short, head.last_processed[0, -2:].float()))
        # Preserve FP32 normalized state for memory while expert sees BF16 state.
        self.assertEqual(previous.state.dtype, torch.float32)
        torch.testing.assert_close(previous.state, torch.full((8,), 0.1234567), rtol=0, atol=0)
        self.assertEqual(head.last_state.dtype, torch.bfloat16)

    def test_seed_change_requires_reset(self):
        policy = _policy()
        _actions(policy)
        with self.assertRaisesRegex(ValueError, "seed changed"):
            _actions(policy, frame=2, seed=18, controls=np.zeros((2, 8), np.float32))
        np.testing.assert_array_equal(_actions(policy, seed=18, reset=True), _actions(_policy(), seed=18))

    def test_missing_or_noncausal_protocol_rejected(self):
        malformed = [
            {}, {**_options(), "session_ids": []}, {**_options(), "reset_memory": []},
            {**_options(), "passive": True}, {**_options(), "frame_index": -1},
            {**_options(), "episode_seed": -1},
            {**_options(), "executed_actions": np.ones((1, 8), np.float32)},
            {**_options(), "executed_actions": np.full((1, 8), np.nan, np.float32)},
        ]
        for options in malformed:
            with self.subTest(options=options), self.assertRaises(ValueError):
                _policy().get_action(_observation(), options)
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            _policy().get_action(_observation(batch=2), _options())
        policy = _policy()
        _actions(policy)
        for frame, controls in ((0, None), (3, np.zeros((3, 8), np.float32)), (2, None)):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                _actions(policy, frame=frame, controls=controls)

    def test_prediction_failure_invalidates_partial_session(self):
        policy = _policy(memory=True)
        policy.model.action_head.fail_prediction = True
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite model action"):
            _actions(policy)
        self.assertNotIn("A", policy.sessions)
        self.assertIsNone(policy.model.action_head._memory_cache)
        self.assertIsNone(policy.model.action_head._inference_gen)

    def test_lru_session_eviction_is_bounded(self):
        policy = _policy()
        policy.session_cap = 2
        for session in ("A", "B", "C"):
            _actions(policy, session=session)
        self.assertEqual(list(policy.sessions), ["B", "C"])
        np.testing.assert_array_equal(_actions(policy, session="A"), _actions(_policy()))
        self.assertEqual(len(policy.sessions), 2)


if __name__ == "__main__":
    unittest.main()
