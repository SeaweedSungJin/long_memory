"""CPU-only contracts for the standalone V11 prototype, not robot performance."""
from dataclasses import replace
import inspect
import unittest

import torch

from run_scripts.robomme.visual_patch_memory_v11 import (
    CAMERA_ORDER, PATCHES_PER_OBSERVATION, PatchBank, VisualPatchConfig,
    VisualPatchMemoryV11,
)


def inputs(seed=1, batch=2, dtype=torch.float32, feature_dim=8):
    """Two 81-token camera runs; differing text offsets and attention padding."""
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(batch, 236, feature_dim, generator=generator).to(dtype)
    images = torch.zeros(batch, 236, dtype=torch.bool)
    attention = torch.ones_like(images)
    for row in range(batch):
        start = 10 + row * 11
        images[row, start:start + 81] = True
        images[row, start + 88:start + 169] = True
        attention[row, :row * 3] = False
        attention[row, start + 172:-4] = False
    return features, images, attention


def observation(memory, seed=1, frames=(0, 0), demo=(False, False), dtype=torch.float32):
    args = inputs(seed, len(frames), dtype, memory.config.feature_dim)
    return memory.encode_observation(*args, frames, demo, camera_order=CAMERA_ORDER)


def one_row(value, row):
    return replace(value, **{name: getattr(value, name)[row:row + 1]
                             for name in value.__dataclass_fields__})


class VisualPatchMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def memory(self):
        # Keep construction isolated from unrelated training/session RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(811)
            return VisualPatchMemoryV11(VisualPatchConfig(feature_dim=8, hidden_dim=16))

    def wake(self, memory):
        generator = torch.Generator().manual_seed(99)
        with torch.no_grad():
            memory.output_projection.weight.copy_(
                torch.randn(memory.output_projection.weight.shape, generator=generator) * .04)

    def test_config_defaults_and_invalid_contracts(self):
        config = VisualPatchConfig()
        self.assertEqual((config.feature_dim, config.hidden_dim, config.num_heads,
                          config.num_short_tokens, PATCHES_PER_OBSERVATION), (2048, 256, 4, 4, 162))
        for options in ({"hidden_dim": 7}, {"feature_dim": 0}, {"num_heads": True},
                        {"time_scale": float("nan")}, {"time_scale": 0}, {"time_scale": True}):
            with self.assertRaises(ValueError):
                VisualPatchConfig(**options)
        with self.assertRaises(ValueError):
            self.memory().empty_bank(0)

    def test_query_key_value_start_identity_and_output_starts_zero(self):
        memory = self.memory()
        for name in ("query_projection", "key_projection", "value_projection"):
            weight = getattr(memory, name).weight
            self.assertTrue(weight.requires_grad)
            self.assertTrue(torch.equal(weight, torch.eye(memory.config.hidden_dim)))
        self.assertFalse(bool(memory.output_projection.weight.any()))
        self.assertIsNone(memory.output_projection.bias)

    def test_fp32_bank_ignores_global_default_dtype(self):
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            memory = self.memory()
            self.assertTrue(all(p.dtype == torch.float32 for p in memory.parameters()))
            self.assertEqual(memory.empty_bank(1).tokens.dtype, torch.float32)
        finally:
            torch.set_default_dtype(original_dtype)

    def test_nonfinite_encoded_features_and_residual_cast_are_rejected(self):
        memory = self.memory()
        with torch.no_grad():
            memory.image_projection.weight.fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "encoding"):
            observation(memory)

        memory = self.memory()
        bank = memory.append(None, observation(memory))
        current = observation(memory, frames=(16, 16), dtype=torch.float16)
        # Finite FP32 recall can overflow when cast to the original dtype.
        with torch.no_grad():
            memory.output_projection.weight.zero_()
            memory.output_projection.weight[:, 0] = 1e20
        with self.assertRaisesRegex(FloatingPointError, "cast/add"):
            memory.read(current, bank)

    def test_empty_and_zero_initialized_read_are_exact_fp32_bf16(self):
        for dtype in (torch.float32, torch.bfloat16):
            memory = self.memory()
            old = observation(memory, frames=(0, 0), dtype=dtype)
            bank = memory.append(None, old, valid=[True, False])
            current = observation(memory, seed=2, frames=(16, 16), dtype=dtype)
            before = current.features.clone()
            self.assertTrue(torch.equal(memory.read(current, None), before))
            self.assertTrue(torch.equal(memory.read(current, bank), before))
            self.assertTrue(torch.equal(current.features, before))
            self.assertEqual(bank.tokens.shape, (2, 1, 162, 16))
            self.assertEqual(bank.tokens.dtype, torch.float32)
            self.assertTrue(torch.equal(bank.valid, torch.tensor([[True], [False]])))

    def test_camera_grid_mapping_and_mask_offsets_are_explicit(self):
        memory = self.memory()
        args = inputs()
        current = memory.encode_observation(*args, [0, 0], [False, True], camera_order=CAMERA_ORDER)
        for row, start in enumerate((10, 21)):
            self.assertEqual(current.image_indices[row, :81].tolist(), list(range(start, start + 81)))
            self.assertEqual(current.image_indices[row, 81:].tolist(), list(range(start + 88, start + 169)))
        for index in range(162):
            self.assertEqual(int(memory.patch_cameras[index]), index // 81)
            self.assertEqual(int(memory.patch_rows[index]), index % 81 // 9)
            self.assertEqual(int(memory.patch_columns[index]), index % 9)
        for cameras in (CAMERA_ORDER[::-1], ("front_view",), ("left", "right")):
            with self.assertRaisesRegex(ValueError, "camera_order"):
                memory.encode_observation(*args, [0, 0], [False, True], camera_order=cameras)
        with self.assertRaises(TypeError):
            memory.encode_observation(*args, [0, 0], [False, True])

    def test_short_tail_and_attention_mask_excluded_without_mutating_masks(self):
        memory = self.memory()
        features, images, attention = inputs()
        images[:, -4:] = True  # Defensive exclusion even for a stale image mask.
        images[1, :3] = True   # Attention-masked prefix is not an image patch.
        saved_images, saved_attention = images.clone(), attention.clone()
        current = memory.encode_observation(features, images, attention, [0, 0], [False, False],
                                            camera_order=CAMERA_ORDER)
        self.assertEqual(current.image_indices.shape, (2, 162))
        self.assertTrue(bool((current.image_indices < features.shape[1] - 4).all()))
        self.assertTrue(torch.equal(images, saved_images))
        self.assertTrue(torch.equal(attention, saved_attention))

    def test_bad_patch_masks_and_metadata_are_rejected(self):
        memory = self.memory()
        features, images, attention = inputs()

        def encode(im=images, am=attention, frames=(0, 0), demo=(False, False), x=features):
            return memory.encode_observation(x, im, am, frames, demo, camera_order=CAMERA_ORDER)

        broken = images.clone(); broken[0, 10] = False
        with self.assertRaisesRegex(ValueError, "162"):
            encode(im=broken)
        broken = images.clone(); broken[0, 11] = False; broken[0, 9] = True
        with self.assertRaisesRegex(ValueError, "contiguous"):
            encode(im=broken)
        broken = images.clone(); broken[0] = False; broken[0, 10:172] = True
        with self.assertRaisesRegex(ValueError, "separate"):
            encode(im=broken)
        broken = attention.clone(); broken[0, -1] = False
        with self.assertRaisesRegex(ValueError, "short tail"):
            encode(am=broken)
        with self.assertRaisesRegex(ValueError, "boolean"):
            encode(im=images.float())
        for frames in ((-1, 0), (0., 1.), (True, False)):
            with self.assertRaisesRegex(ValueError, "frames"):
                encode(frames=frames)
        with self.assertRaisesRegex(ValueError, "is_demo"):
            encode(demo=(0, 1))
        broken = features.clone(); broken[0, 10, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            encode(x=broken)
        with self.assertRaisesRegex(ValueError, "FP32"):
            memory.bfloat16().encode_observation(features, images, attention, [0, 0], [False, False],
                                                 camera_order=CAMERA_ORDER)

    def test_camera_row_column_frame_and_demo_change_the_encoding(self):
        memory = self.memory()
        features, images, attention = inputs(batch=1)
        features.zero_()

        def encode(frame, demo):
            return memory.encode_observation(features, images, attention, [frame], [demo],
                                              camera_order=CAMERA_ORDER).encoded

        base = encode(0, False)
        for index in (1, 9, 81):
            self.assertFalse(torch.equal(base[:, 0], base[:, index]))
        self.assertFalse(torch.equal(base, encode(16, False)))
        self.assertFalse(torch.equal(base, encode(0, True)))

    def test_current_short_changes_queries_not_stored_visual_encodings(self):
        memory = self.memory()
        features, images, attention = inputs()
        base = memory.encode_observation(features, images, attention, [0, 0], [False, False],
                                          camera_order=CAMERA_ORDER)
        changed = features.clone(); changed[:, -4:, 0] += 4
        after = memory.encode_observation(changed, images, attention, [0, 0], [False, False],
                                           camera_order=CAMERA_ORDER)
        self.assertTrue(torch.equal(base.encoded, after.encoded))
        self.assertFalse(torch.equal(base.queries, after.queries))
        changed = features.clone(); changed[:, 5, :] += 100
        text_only = memory.encode_observation(changed, images, attention, [0, 0], [False, False],
                                               camera_order=CAMERA_ORDER)
        self.assertTrue(torch.equal(base.encoded, text_only.encoded))
        self.assertTrue(torch.equal(base.queries, text_only.queries))

    def test_read_before_write_and_strict_future_chronology_guard(self):
        memory = self.memory()
        self.wake(memory)
        args = inputs()
        first, bank = memory(*args, [0, 0], [True, True], camera_order=CAMERA_ORDER)
        self.assertTrue(torch.equal(first, args[0]))
        current = observation(memory, seed=2, frames=(16, 16))
        self.assertFalse(torch.equal(memory.read(current, bank), current.features))
        for frames in ((0, 16), (-16, 16)):
            modified = replace(bank, frames=torch.tensor([frames]).T)
            with self.assertRaisesRegex(ValueError, "strictly past"):
                memory.read(observation(memory, frames=(0, 16)), modified)
        with self.assertRaisesRegex(ValueError, "strictly past"):
            memory.append(bank, observation(memory, frames=(0, 0)))
        future = replace(bank, frames=bank.frames + 32)
        with self.assertRaisesRegex(ValueError, "strictly past"):
            memory.read(current, future, enabled=False)
        bank2 = memory.append(bank, current)
        unordered = replace(bank2, frames=bank2.frames.flip(1))
        with self.assertRaisesRegex(ValueError, "chronological"):
            memory.read(observation(memory, frames=(32, 32)), unordered)

    def test_woken_step_updates_only_images_and_appends_original_encoding(self):
        memory = self.memory()
        self.wake(memory)
        bank = memory.append(None, observation(memory, frames=(0, 0)))
        args = inputs(seed=2, dtype=torch.bfloat16)
        saved = [x.clone() for x in args]
        old_tokens = bank.tokens.clone()
        current = memory.encode_observation(*args, [16, 16], [False, False], camera_order=CAMERA_ORDER)
        fused, following = memory(*args, [16, 16], [False, False], bank, camera_order=CAMERA_ORDER)
        self.assertFalse(torch.equal(fused, args[0]))
        self.assertTrue(torch.equal(fused[~args[1]], args[0][~args[1]]))
        self.assertTrue(torch.equal(fused[:, -4:], args[0][:, -4:]))
        for actual, expected in zip(args, saved):
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(bank.tokens, old_tokens))
        self.assertTrue(torch.equal(following.tokens[:, -1], current.encoded))
        encoded_modified = memory.encode_observation(fused, args[1], args[2], [16, 16], [False, False],
                                                     camera_order=CAMERA_ORDER)
        self.assertFalse(torch.equal(following.tokens[:, -1], encoded_modified.encoded))

    def test_batched_variable_history_padding_and_disabled_rows(self):
        memory = self.memory()
        self.wake(memory)
        bank = memory.append(None, observation(memory, frames=(0, 0)), valid=[True, False])
        bank = memory.append(bank, observation(memory, seed=2, frames=(16, 16)), valid=[True, False])
        current = observation(memory, seed=3, frames=(32, 32))
        fused = memory.read(current, bank)
        self.assertFalse(torch.equal(fused[0], current.features[0]))
        self.assertTrue(torch.equal(fused[1], current.features[1]))
        individual = memory.read(one_row(current, 0), one_row(bank, 0))
        torch.testing.assert_close(fused[:1], individual, rtol=0, atol=0)
        padded = bank.tokens.clone(); padded[~bank.valid] = float("nan")
        self.assertTrue(torch.equal(fused, memory.read(current, replace(bank, tokens=padded))))
        self.assertTrue(torch.equal(memory.read(current, bank, enabled=[False, True]), current.features))
        # Padding among valid observations is permitted; it is not a token.
        bank = memory.append(bank, current, valid=[False, True])
        final = observation(memory, seed=4, frames=(48, 48))
        batched = memory.read(final, bank)
        for row in range(2):
            torch.testing.assert_close(batched[row:row + 1], memory.read(one_row(final, row), one_row(bank, row)),
                                       rtol=1e-6, atol=1e-6)

    def test_zero_output_wakes_then_read_encoder_and_early_history_receive_gradients(self):
        memory = self.memory()

        def graph(count):
            bank, observations = None, []
            for index in range(count):
                item = observation(memory, seed=index + 1, frames=(index * 16,), demo=(index < 2,))
                item.encoded.retain_grad()
                observations.append(item)
                bank = memory.append(bank, item)
            current = observation(memory, seed=90, frames=(count * 16,), demo=(False,))
            fused = memory.read(current, bank)
            return fused.square().mean(), observations

        loss, past = graph(2)
        loss.backward()
        self.assertGreater(float(memory.output_projection.weight.grad.norm()), 0)
        self.assertEqual(float(memory.image_projection.weight.grad.norm()), 0)
        self.assertEqual(float(memory.query_projection.weight.grad.norm()), 0)
        self.assertEqual(float(past[0].encoded.grad.norm()), 0)
        # A genuine first output-weight update, without creating a trainer.
        with torch.no_grad():
            grad = memory.output_projection.weight.grad
            memory.output_projection.weight.add_(-.02 * grad / grad.norm())
        memory.zero_grad(set_to_none=True)
        loss, past = graph(20)  # The earliest cue is older than V6's 16-event budget.
        loss.backward()
        for name in ("image_projection", "query_projection", "key_projection", "value_projection",
                     "output_projection", "time_projection", "camera_embedding", "row_embedding", "column_embedding"):
            grad = getattr(memory, name).weight.grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(bool(torch.isfinite(grad).all()), name)
            self.assertGreater(float(grad.norm()), 0, name)
        self.assertEqual(len(past), 20)
        self.assertTrue(all(item.encoded.grad is not None and bool(item.encoded.grad.abs().sum() > 0) for item in past))

    def test_full_prefix_and_incremental_forward_agree_and_do_not_consume_rng(self):
        memory = self.memory()
        self.wake(memory)
        before = torch.get_rng_state().clone()
        bank, replay = None, None
        for index in range(4):
            args = inputs(index + 1)
            item = memory.encode_observation(*args, [16 * index] * 2, [index < 2] * 2,
                                              camera_order=CAMERA_ORDER)
            expected = memory.read(item, replay)
            replay = memory.append(replay, item)
            actual, bank = memory(*args, [16 * index] * 2, [index < 2] * 2, bank, camera_order=CAMERA_ORDER)
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(replay.tokens, bank.tokens))
            self.assertTrue(torch.equal(replay.frames, bank.frames))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(bank.tokens.shape[1:3], (4, 162))
        self.assertTrue(torch.equal(bank.is_demo[0], torch.tensor([True, True, False, False])))

    def test_no_gt_action_api_or_implicit_randomness(self):
        memory = self.memory()
        for method in (memory.encode_observation, memory.read, memory.append, memory.forward):
            parameters = inspect.signature(method).parameters
            self.assertFalse({"actions", "targets", "reward", "success"} & parameters.keys())
            self.assertFalse(any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()))
        with self.assertRaises(TypeError):
            memory(*inputs(), [0, 0], [False, False], camera_order=CAMERA_ORDER, targets=torch.zeros(1))
        first = observation(memory)
        second = observation(memory)
        self.assertTrue(torch.equal(first.encoded, second.encoded))
        self.assertTrue(torch.equal(first.queries, second.queries))

    def test_bank_shape_dtype_validity_and_nonfinite_active_tokens(self):
        memory = self.memory()
        bank = memory.append(None, observation(memory))
        current = observation(memory, frames=(16, 16))
        invalid = (replace(bank, tokens=bank.tokens.bfloat16()),
                   replace(bank, tokens=bank.tokens[:, :, :-1]),
                   replace(bank, valid=bank.valid.float()),
                   replace(bank, frames=bank.frames.float()),
                   replace(bank, tokens=bank.tokens * float("nan")))
        for wrong in invalid:
            with self.assertRaises(ValueError):
                memory.read(current, wrong)


if __name__ == "__main__":
    unittest.main()
