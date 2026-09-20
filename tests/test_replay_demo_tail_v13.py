"""CPU-only action replay tests; actual frozen-expert wiring is separate."""
import unittest

import torch

from run_scripts.robomme.replay_demo_tail_v13 import replay_demo_tail
from run_scripts.robomme.replay_visual_differential_v12 import replay_visual_differential
from tests.test_visual_demo_tail_bank_v13 import memory, source, sidecar, CACHE, SIDECAR, CAMERAS
from tests.test_replay_visual_patch_v11 import GuardedColumn, ObservationOnlyMapping


def replay(model, obs, q=3, *, tail=True, **kwargs):
    payload, record = sidecar(n_demo=48)
    return replay_demo_tail(model, obs, q, camera_order=CAMERAS, include_tail=tail,
        sidecar=payload, record=record, episode_id=7, cache_fingerprint=CACHE,
        sidecar_fingerprint=SIDECAR, **kwargs)


class DemoTailReplayTests(unittest.TestCase):
    def test_no_future_or_action_target_access_in_either_arm(self):
        for tail in (False, True):
            original = source()
            guarded = ObservationOnlyMapping({k: GuardedColumn(v, 3) for k, v in original.items()})
            features, bank = replay(memory(awake=True), guarded, tail=tail)
            self.assertEqual(features.shape[1:], original["features"][3].shape)
            self.assertTrue((bank.frames < 48).all())

    def test_canonical_control_numerically_matches_old_batched_replay(self):
        model, obs = memory(awake=True), source()
        actual, bank = replay(model, obs, tail=False)
        expected, reference = replay_visual_differential(model, obs, 3, camera_order=CAMERAS)
        # The old V12 reference batches history. V13 intentionally uses the
        # singleton online shape; exact online parity is tested separately.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(bank.tokens, reference.tokens, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(bank.content, reference.content, rtol=1e-5, atol=1e-6)

    def test_off_identity_retains_full_tail_bank_and_original_query(self):
        model, obs = memory(awake=True), source()
        original = obs["features"][3].clone()
        actual, bank = replay(model, obs, visual_read_enabled=False)
        self.assertTrue(torch.equal(actual[0], original))
        self.assertEqual(bank.frames[0].tolist(), [0, 16, 32] + list(range(33, 48)))
        self.assertTrue(torch.equal(obs["features"][3], original))

    def test_storage_retrieval_current_and_tail_gradients(self):
        model, obs = memory(awake=True), source(grad=True)
        payload, record = sidecar(grad=True)
        features, bank = replay_demo_tail(model, obs, 3, camera_order=CAMERAS, include_tail=True,
            sidecar=payload, record=record, episode_id=7, cache_fingerprint=CACHE,
            sidecar_fingerprint=SIDECAR, checkpoint_encoding=True)
        features.float().square().mean().backward()
        for value in (payload["images"].grad, obs["features"][0].grad, obs["features"][3].grad,
                      model.image_projection.weight.grad, model.query_projection.weight.grad,
                      model.key_projection.weight.grad, model.value_projection.weight.grad):
            self.assertIsNotNone(value)
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreater(int(value.count_nonzero()), 0)
        self.assertIsNone(obs["features"][4].grad)

    def test_passive_query_bad_flags_and_current_in_bank_rejected(self):
        model, obs = memory(), source()
        with self.assertRaisesRegex(ValueError, "action-query"):
            replay(model, obs, 2)
        with self.assertRaisesRegex(ValueError, "boolean"):
            replay_demo_tail(model, obs, 3, camera_order=CAMERAS, include_tail=1)
        obs["frames"][3] = 32
        with self.assertRaises(ValueError):
            replay(model, obs, tail=False)

    def test_no_demo_empty_query_is_identity_without_fake_tail(self):
        model, obs = memory(awake=True), source(frames=(0, 16), n_demo=0)
        payload, record = sidecar(n_demo=0)
        features, bank = replay_demo_tail(model, obs, 0, camera_order=CAMERAS, include_tail=True,
            sidecar=payload, record=record, episode_id=7, cache_fingerprint=CACHE, sidecar_fingerprint=SIDECAR)
        self.assertEqual(bank.tokens.shape[1], 0)
        self.assertTrue(torch.equal(features[0], obs["features"][0]))


if __name__ == "__main__":
    unittest.main()
