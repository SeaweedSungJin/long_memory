"""CPU evidence against the ACTUAL ordinary canonical online encoding path."""
import copy
from dataclasses import fields
import unittest
from unittest.mock import patch

import torch

from run_scripts.robomme import framewise_demo_tail_v13 as framewise
from run_scripts.robomme.replay_demo_tail_v13 import replay_demo_tail
from run_scripts.robomme.replay_visual_differential_v12 import build_visual_differential_bank
from tests import test_visual_demo_tail_bank_v13 as fixtures
from tests.test_replay_visual_patch_v11 import GuardedColumn, ObservationOnlyMapping


def build(memory, observations, query=3, *, tail=True, payload=None, record=None, checkpoint=False):
    if payload is None:
        payload, record = fixtures.sidecar()
    return framewise.build_framewise_prefix(memory, observations, query, include_tail=tail,
        sidecar=payload, record=record, episode_id=7, cache_fingerprint=fixtures.CACHE,
        sidecar_fingerprint=fixtures.SIDECAR, camera_order=fixtures.CAMERAS, checkpoint_encoding=checkpoint)


def online_bank(memory, observations, query, payload=None):
    """Canonical uses encode_observation+append, NOT the new helper itself."""
    chronological = {int(observations["frames"][i]): ("canonical", i) for i in range(query)}
    if payload is not None:
        chronological.update({int(frame): ("tail", i) for i, frame in enumerate(payload["frames"])})
    bank = None
    for frame, (kind, index) in sorted(chronological.items()):
        if kind == "canonical":
            row = memory.encode_observation(observations["features"][index][None],
                observations["image_masks"][index][None], observations["attention_masks"][index][None],
                [frame], [observations["is_demo"][index]], camera_order=fixtures.CAMERAS)
            bank = memory.append(bank, row)
        else:
            row = memory.encode_bank_images(payload["images"][index][None], [frame], [True], camera_order=fixtures.CAMERAS)
            bank = memory.append_bank_images(bank, row)
    return memory.empty_bank(1) if bank is None else bank


class FramewiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_bank_equal(self, left, right):
        for field in fields(left):
            self.assertTrue(torch.equal(getattr(left, field.name), getattr(right, field.name)), field.name)

    def test_exact_actual_online_canonical_and_tail_z_c_read_both_dtypes(self):
        for dtype in (torch.float32, torch.bfloat16):
            for tail in (False, True):
                if tail and dtype != torch.bfloat16:
                    continue  # Real sidecars are explicitly BF16 only.
                memory = fixtures.memory(awake=True)
                observations = fixtures.source(dtype=dtype)
                payload, record = fixtures.sidecar()
                actual = build(memory, observations, tail=tail, payload=payload, record=record)
                expected = online_bank(memory, observations, 3, payload if tail else None)
                self.assert_bank_equal(actual, expected)
                query = fixtures.current(memory, observations, 3)
                self.assertTrue(torch.equal(memory.read(query, actual), memory.read(query, expected)))
                # Old batched V12 encoding is only a numerical/semantic control.
                if not tail:
                    batched = build_visual_differential_bank(memory, observations, 3, camera_order=fixtures.CAMERAS)
                    torch.testing.assert_close(actual.tokens, batched.tokens, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(actual.content, batched.content, rtol=1e-5, atol=1e-6)

    def test_real_online_gradient_parity_to_all_parameters_canonical_and_tail_inputs(self):
        first, second = fixtures.memory(awake=True), fixtures.memory(awake=True)
        observations = fixtures.source(grad=True)
        other = copy.deepcopy(observations)
        payload, record = fixtures.sidecar(grad=True)
        other_payload = copy.deepcopy(payload)
        actual = build(first, observations, payload=payload, record=record)
        expected = online_bank(second, other, 3, other_payload)
        self.assert_bank_equal(actual, expected)
        a = first.read(fixtures.current(first, observations, 3), actual)
        b = second.read(fixtures.current(second, other, 3), expected)
        self.assertTrue(torch.equal(a, b))
        a.float().square().mean().backward()
        b.float().square().mean().backward()
        for (name, left), (_, right) in zip(first.named_parameters(), second.named_parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7, msg=name)
        for left, right in zip(observations["features"], other["features"]):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(payload["images"].grad, other_payload["images"].grad)
        self.assertGreater(int(payload["images"].grad.count_nonzero()), 0)
        self.assertGreater(int(observations["features"][0].grad.count_nonzero()), 0)
        self.assertIsNone(observations["features"][4].grad)

    def test_only_singleton_calls_no_historical_query_no_rng_or_weight_changes(self):
        memory, observations = fixtures.memory(awake=True), fixtures.source()
        weights = {name: value.clone() for name, value in memory.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with patch.object(memory, "encode_bank_images", wraps=memory.encode_bank_images) as encoder, \
                patch.object(memory.query_projection, "forward", side_effect=AssertionError("No historical query")):
            bank = build(memory, observations)
        self.assertEqual(len(encoder.call_args_list), 18)
        self.assertTrue(all(call.args[0].shape == (1, 2, 81, 8) for call in encoder.call_args_list))
        self.assertEqual([int(call.args[1][0]) for call in encoder.call_args_list], bank.frames[0].tolist())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(len(list(memory.parameters())), 14)
        for name, value in memory.state_dict().items():
            self.assertTrue(torch.equal(value, weights[name]))

    def test_per_frame_checkpoint_forward_and_gradient_parity(self):
        plain, checked = fixtures.memory(awake=True), fixtures.memory(awake=True)
        observations = fixtures.source(grad=True); other = copy.deepcopy(observations)
        payload, record = fixtures.sidecar(grad=True); other_payload = copy.deepcopy(payload)
        first = build(plain, observations, payload=payload, record=record)
        second = build(checked, other, payload=other_payload, record=record, checkpoint=True)
        self.assert_bank_equal(first, second)
        a = plain.read(fixtures.current(plain, observations, 3), first)
        b = checked.read(fixtures.current(checked, other, 3), second)
        self.assertTrue(torch.equal(a, b))
        a.float().square().mean().backward(); b.float().square().mean().backward()
        for (name, left), (_, right) in zip(plain.named_parameters(), checked.named_parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7, msg=name)
        self.assertTrue(torch.equal(payload["images"].grad, other_payload["images"].grad))

    def test_builder_never_reads_current_features_future_or_nonobservation_columns(self):
        observations = fixtures.source()
        protected = ObservationOnlyMapping({key: GuardedColumn(value, 3 if key in ("frames", "is_demo") else 2)
                                           for key, value in observations.items()})
        protected.update(actions=object(), targets=object(), state=object(), parent=object())
        bank = build(fixtures.memory(), protected)
        self.assertEqual(bank.frames[0].tolist(), [0, 16, 32, *range(33, 48)])
        self.assertEqual(protected["features"].accesses, [0, 1, 2])

    def test_invalid_future_duplicate_reordered_or_binding_fails_before_encoding(self):
        memory, original = fixtures.memory(), fixtures.source()
        for mutation in (lambda p: p["frames"].__setitem__(-1, 48),
                         lambda p: p["frames"].__setitem__(-1, 64),
                         lambda p: p["frames"].__setitem__(1, 33),
                         lambda p: p.update(frames=p["frames"].flip(0)),
                         lambda p: p.update(episode_id=8)):
            payload, record = fixtures.sidecar(); mutation(payload)
            with patch.object(memory, "encode_bank_images", side_effect=AssertionError("Must reject first")):
                with self.assertRaises(ValueError):
                    build(memory, original, payload=payload, record=record)
        for frames in ((16, 0, 32, 48), (0, 16, 32, 32)):
            with self.assertRaises(ValueError):
                build(memory, fixtures.source(frames=frames), tail=False)
        with self.assertRaisesRegex(ValueError, "action-query"):
            build(memory, original, 2)

    def test_production_replay_enforces_framewise_and_off_keeps_full_bank(self):
        memory, obs = fixtures.memory(awake=True), fixtures.source()
        payload, record = fixtures.sidecar()
        options = dict(camera_order=fixtures.CAMERAS, include_tail=True, sidecar=payload, record=record,
                       episode_id=7, cache_fingerprint=fixtures.CACHE, sidecar_fingerprint=fixtures.SIDECAR)
        actual, bank = replay_demo_tail(memory, obs, 3, visual_read_enabled=False, replay_encoding="framewise", **options)
        self.assertTrue(torch.equal(actual[0], obs["features"][3]))
        self.assertEqual(bank.tokens.shape[1], 18)
        for invalid in ("batched", "", None, True):
            with self.assertRaisesRegex(ValueError, "framewise"):
                replay_demo_tail(memory, obs, 3, replay_encoding=invalid, **options)
        empty_obs = fixtures.source(frames=(0, 16), n_demo=0)
        empty, empty_record = fixtures.sidecar(0)
        bank = build(memory, empty_obs, 0, payload=empty, record=empty_record)
        self.assertEqual(bank.tokens.shape, (1, 0, 162, 16))


if __name__ == "__main__":
    unittest.main()
