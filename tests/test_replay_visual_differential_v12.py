"""V12 dual-stream CPU replay plus inherited V11 prefix/mask regressions."""
import copy
from unittest.mock import patch

import torch

from run_scripts.robomme import replay_visual_differential_v12 as replay
from run_scripts.robomme.visual_differential_memory_v12 import (
    CAMERA_ORDER, READ_MODES, VisualDifferentialConfig, VisualDifferentialMemoryV12,
)
from tests import test_replay_visual_patch_v11 as legacy


class VisualDifferentialReplayTests(legacy.VisualPatchReplayTests):
    """The inherited replay contract runs against V12 without frozen-source edits."""

    def memory(self, awake=True, mode="differential"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            memory = VisualDifferentialMemoryV12(VisualDifferentialConfig(feature_dim=8, hidden_dim=16), read_mode=mode)
            if awake:
                with torch.no_grad():
                    memory.output_projection.weight.normal_(std=.04)
            return memory

    def replay(self, memory, source, query, **options):
        return replay.replay_visual_differential(memory, source, query, camera_order=CAMERA_ORDER, **options)

    def test_batched_dual_stream_matches_incremental_both_modes_and_bf16_zero(self):
        for mode in READ_MODES:
            for dtype, awake in ((torch.float32, True), (torch.bfloat16, False)):
                memory = self.memory(awake, mode)
                source = legacy.observations(5, dtype=dtype)
                expected, reference = legacy.sequential(memory, source, 4)
                with patch.object(memory, "append", side_effect=AssertionError("No growing APPEND")):
                    actual, bank = self.replay(memory, source, 4)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                for name in ("tokens", "content"):
                    torch.testing.assert_close(getattr(bank, name), getattr(reference, name), rtol=1e-5, atol=1e-6)
                    self.assertEqual(getattr(bank, name).dtype, torch.float32)
                if not awake:
                    self.assertTrue(torch.equal(actual, source["features"][4][None]))
                off, off_bank = self.replay(memory, source, 4, visual_read_enabled=False)
                self.assertTrue(torch.equal(off, source["features"][4][None]))
                self.assertTrue(torch.equal(off_bank.content, bank.content))
                self.assertTrue(torch.equal(off_bank.tokens, bank.tokens))

    def test_dual_stream_future_columns_and_empty_prefix_never_read(self):
        source = legacy.observations(5)
        for mode in READ_MODES:
            memory = self.memory(mode=mode)
            guarded = legacy.ObservationOnlyMapping({key: legacy.GuardedColumn(source[key], 2)
                                                     for key in replay.OBSERVATION_KEYS})
            guarded.update(targets=object(), actions=object(), state=object(), success=object())
            bank = replay.build_visual_differential_bank(memory, guarded, 3, camera_order=CAMERA_ORDER)
            self.assertEqual(bank.content.shape, (1, 3, 162, 16))
            for key in replay.OBSERVATION_KEYS:
                self.assertEqual(guarded[key].accesses, [0, 1, 2])
            empty = replay.build_visual_differential_bank(memory, {}, 0, camera_order=CAMERA_ORDER)
            self.assertEqual(empty.content.shape, (1, 0, 162, 16))
            current, prior = self.replay(memory, source, 0)
            self.assertTrue(torch.equal(current, source["features"][0][None]))
            self.assertEqual(prior.content.numel(), 0)

    def test_twenty_old_images_both_streams_and_current_reference_receive_gradients(self):
        memory, source = self.memory(), legacy.observations(21, requires_grad=True)
        result, bank = self.replay(memory, source, 20)
        bank.tokens.retain_grad(); bank.content.retain_grad()
        result.square().mean().backward()
        for stream in (bank.tokens, bank.content):
            self.assertTrue(torch.isfinite(stream.grad).all())
            self.assertTrue((stream.grad.abs().sum(dim=(-1, -2)) > 0).all())
        for index, feature in enumerate(source["features"]):
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())
            self.assertGreater(float(feature.grad[source["image_masks"][index]].norm()), 0)
        for name in ("image_projection", "query_projection", "key_projection", "value_projection"):
            self.assertGreater(float(getattr(memory, name).weight.grad.norm()), 0)

    def test_current_only_bank_encoded_but_every_old_input_gradient_absent(self):
        memory, source = self.memory(mode="current_only"), legacy.observations(21, requires_grad=True)
        result, bank = self.replay(memory, source, 20, checkpoint_encoding=True)
        bank.tokens.retain_grad(); bank.content.retain_grad()
        result.square().mean().backward()
        self.assertIsNone(bank.tokens.grad)
        self.assertIsNone(bank.content.grad)
        self.assertTrue(all(feature.grad is None for feature in source["features"][:-1]))
        self.assertGreater(float(source["features"][-1].grad.norm()), 0)
        self.assertEqual(bank.content.shape[1], 20)

    def test_checkpoint_preserves_both_streams_and_full_parameter_input_gradients(self):
        for mode in READ_MODES:
            plain_memory, checked_memory = self.memory(mode=mode), self.memory(mode=mode)
            source = legacy.observations(6, requires_grad=True)
            other = copy.deepcopy(source)
            plain, bank = self.replay(plain_memory, source, 5)
            checked, checked_bank = self.replay(checked_memory, other, 5, checkpoint_encoding=True)
            self.assertTrue(torch.equal(plain, checked))
            self.assertTrue(torch.equal(bank.content, checked_bank.content))
            self.assertTrue(torch.equal(bank.tokens, checked_bank.tokens))
            plain.square().mean().backward(); checked.square().mean().backward()
            for (name, left), (_, right) in zip(plain_memory.named_parameters(), checked_memory.named_parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7, msg=mode + "/" + name)
            for left, right in zip(source["features"], other["features"]):
                if left.grad is None:
                    self.assertIsNone(right.grad)
                else:
                    torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    import unittest
    unittest.main()
