"""Small CPU contracts for observation-only V11 cached replay, not a trainer."""
import copy
import unittest
from unittest.mock import patch

import torch

from run_scripts.robomme.replay_visual_patch_v11 import (
    OBSERVATION_KEYS, build_visual_patch_bank, replay_visual_patch,
)
from run_scripts.robomme.visual_patch_memory_v11 import (
    CAMERA_ORDER, VisualPatchConfig, VisualPatchMemoryV11,
)


def observations(count=5, dtype=torch.float32, requires_grad=False):
    generator = torch.Generator().manual_seed(111)
    result = {key: [] for key in OBSERVATION_KEYS}
    for index in range(count):
        length = (209, 214, 236)[index % 3]
        feature = torch.randn(length, 8, generator=generator).to(dtype).requires_grad_(requires_grad)
        image = torch.zeros(length, dtype=torch.bool)
        start = length - 176
        image[start:start + 81] = True
        image[start + 88:start + 169] = True
        attention = torch.ones(length, dtype=torch.bool)
        attention[:index % 3] = False
        for key, value in zip(OBSERVATION_KEYS, (feature, image, attention, index * 16, index < 2)):
            result[key].append(value)
    return result


def sequential(memory, source, query):
    bank = None
    for index in range(query + 1):
        current = memory.encode_observation(
            source["features"][index][None], source["image_masks"][index][None],
            source["attention_masks"][index][None], [source["frames"][index]],
            [source["is_demo"][index]], camera_order=CAMERA_ORDER)
        if index < query:
            bank = memory.append(bank, current)
    return memory.read(current, bank), bank


class GuardedColumn:
    def __init__(self, values, maximum):
        self.values, self.maximum, self.accesses = values, maximum, []

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index <= self.maximum:
            raise AssertionError("Future observation or broad slice was accessed")
        self.accesses.append(index)
        return self.values[index]

    def __len__(self):
        raise AssertionError("Full column length was inspected")


class ObservationOnlyMapping(dict):
    def __getitem__(self, key):
        if key not in OBSERVATION_KEYS:
            raise AssertionError("GT/action/state column was accessed")
        return super().__getitem__(key)

    def __iter__(self):
        raise AssertionError("Extra columns were enumerated")


class VisualPatchReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def memory(self, awake=True):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            memory = VisualPatchMemoryV11(VisualPatchConfig(feature_dim=8, hidden_dim=16))
            if awake:
                with torch.no_grad():
                    memory.output_projection.weight.normal_(std=.04)
        return memory

    def replay(self, memory, source, query, **options):
        return replay_visual_patch(memory, source, query, camera_order=CAMERA_ORDER, **options)

    def test_batched_prefix_matches_per_observation_append(self):
        memory, source = self.memory(), observations()
        expected, reference = sequential(memory, source, 4)
        # No growing bank APPEND or extra per-observation encoder calls.
        with patch.object(memory, "append", side_effect=AssertionError("No repeated APPEND")), \
                patch.object(memory, "encode_observation", wraps=memory.encode_observation) as encoder:
            actual, bank = self.replay(memory, source, 4)
        self.assertEqual(encoder.call_count, 2)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(bank.tokens, reference.tokens, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(bank.frames, reference.frames))
        self.assertTrue(torch.equal(bank.is_demo, reference.is_demo))
        self.assertTrue(bool(bank.valid.all()))
        self.assertEqual(bank.tokens.shape, (1, 4, 162, 16))
        self.assertEqual(actual.shape, (1, source["features"][4].shape[0], 8))

    def test_padding_is_before_short_tail_and_masks_are_false(self):
        memory, source = self.memory(), observations(4)
        saved = {key: [x.clone() if isinstance(x, torch.Tensor) else x for x in values]
                 for key, values in source.items()}
        with patch.object(memory, "encode_observation", wraps=memory.encode_observation) as encoder:
            self.replay(memory, source, 3)
        features, images, attention = encoder.call_args_list[0].args[:3]
        self.assertEqual(features.shape, (3, 236, 8))
        for row in range(3):
            length = source["features"][row].shape[0]
            self.assertTrue(torch.equal(features[row, :length - 4], source["features"][row][:-4]))
            self.assertTrue(torch.equal(features[row, -4:], source["features"][row][-4:]))
            self.assertFalse(bool(features[row, length - 4:-4].any()))
            self.assertFalse(bool(images[row, length - 4:-4].any()))
            self.assertFalse(bool(attention[row, length - 4:-4].any()))
            self.assertEqual(int(images[row].sum()), 162)
        self.assertEqual(encoder.call_args_list[1].args[0].shape, (1, 209, 8))
        for key in ("features", "image_masks", "attention_masks"):
            for before, after in zip(saved[key], source[key]):
                self.assertTrue(torch.equal(before, after))

    def test_bf16_zero_identity_empty_and_nonempty_prefix(self):
        memory, source = self.memory(awake=False), observations(dtype=torch.bfloat16)
        for query in (0, 1, 4):
            actual, bank = self.replay(memory, source, query)
            expected, _ = sequential(memory, source, query)
            self.assertTrue(torch.equal(actual, source["features"][query][None]))
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(bank.tokens.shape, (1, query, 162, 16))
            self.assertEqual(bank.tokens.dtype, torch.float32)

    def test_read_disabled_keeps_complete_original_prefix_and_never_writes_query(self):
        memory, source = self.memory(), observations()
        on, on_bank = self.replay(memory, source, 4)
        off, off_bank = self.replay(memory, source, 4, visual_read_enabled=False)
        self.assertFalse(torch.equal(on, off))
        self.assertTrue(torch.equal(off, source["features"][4][None]))
        self.assertTrue(torch.equal(on_bank.tokens, off_bank.tokens))
        self.assertEqual(off_bank.frames.tolist(), [[0, 16, 32, 48]])
        self.assertEqual(off_bank.is_demo.tolist(), [[True, True, False, False]])

    def test_future_and_target_action_state_columns_are_never_touched(self):
        memory, source = self.memory(), observations()
        guarded = ObservationOnlyMapping({key: GuardedColumn(source[key], 3) for key in OBSERVATION_KEYS})
        guarded.update({key: object() for key in ("targets", "actions", "state", "target_mask", "success")})
        result, bank = self.replay(memory, guarded, 3)
        self.assertEqual(bank.tokens.shape[1], 3)
        self.assertEqual(result.shape[1], 209)
        for key in OBSERVATION_KEYS:
            self.assertEqual(guarded[key].accesses, [0, 1, 2, 3])

        guarded = ObservationOnlyMapping({key: GuardedColumn(source[key], 2) for key in OBSERVATION_KEYS})
        bank = build_visual_patch_bank(memory, guarded, 3, camera_order=CAMERA_ORDER)
        self.assertEqual(bank.tokens.shape[1], 3)
        for key in OBSERVATION_KEYS:
            self.assertEqual(guarded[key].accesses, [0, 1, 2])
        empty = build_visual_patch_bank(memory, {}, 0, camera_order=CAMERA_ORDER)
        self.assertEqual(empty.tokens.shape, (1, 0, 162, 16))

    def test_all_twenty_old_observations_receive_image_gradients(self):
        memory, source = self.memory(), observations(21, requires_grad=True)
        result, bank = self.replay(memory, source, 20)
        bank.tokens.retain_grad()
        result.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(bank.tokens.grad).all()))
        self.assertTrue(bool((bank.tokens.grad.abs().sum(dim=(-1, -2)) > 0).all()))
        for index in range(20):
            grad = source["features"][index].grad
            self.assertIsNotNone(grad)
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(float(grad[source["image_masks"][index]].norm()), 0)
            self.assertEqual(float(grad[~source["image_masks"][index]].norm()), 0)
        for name in ("image_projection", "query_projection", "key_projection", "value_projection"):
            self.assertGreater(float(getattr(memory, name).weight.grad.norm()), 0, name)

    def test_checkpoint_preserves_outputs_parameter_and_old_input_gradients(self):
        first, second = self.memory(), self.memory()
        source = observations(7, requires_grad=True)
        other = copy.deepcopy(source)
        plain, bank = self.replay(first, source, 6)
        checked, checkpoint_bank = self.replay(second, other, 6, checkpoint_encoding=True)
        torch.testing.assert_close(checked, plain, rtol=0, atol=0)
        torch.testing.assert_close(checkpoint_bank.tokens, bank.tokens, rtol=0, atol=0)
        plain.square().mean().backward()
        checked.square().mean().backward()
        for (name, left), (_, right) in zip(first.named_parameters(), second.named_parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7, msg=name)
        for left, right in zip(source["features"], other["features"]):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)

    def test_checkpoint_supports_frozen_inputs_parameter_gradients_and_no_grad(self):
        memory, source = self.memory(), observations()
        result, _ = self.replay(memory, source, 4, checkpoint_encoding=True)
        result.square().mean().backward()
        self.assertGreater(float(memory.image_projection.weight.grad.norm()), 0)
        with torch.no_grad():
            plain, _ = self.replay(memory, source, 4)
            checked, _ = self.replay(memory, source, 4, checkpoint_encoding=True)
        self.assertTrue(torch.equal(plain, checked))

    def test_invalid_times_demo_metadata_and_incomplete_prefix_fail(self):
        memory = self.memory()
        for bad_frames in ([0, 16, 16], [16, 0, 32], [0, -1, 32], [0, 16., 32], [0, True, 32]):
            source = observations(3); source["frames"] = bad_frames
            with self.assertRaisesRegex(ValueError, "frame"):
                self.replay(memory, source, 2)
        source = observations(3); source["is_demo"] = [True, False, True]
        with self.assertRaisesRegex(ValueError, "prefix"):
            self.replay(memory, source, 2)
        source = observations(3); source["is_demo"][1] = 1
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.replay(memory, source, 2)
        source = observations(3); source["frames"][1] = torch.tensor([16])
        with self.assertRaisesRegex(ValueError, "scalar"):
            self.replay(memory, source, 2)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.replay(memory, observations(2), 2)

    def test_invalid_api_masks_dtype_and_camera_contracts_fail(self):
        memory, source = self.memory(), observations()
        for query in (-1, 1., True):
            with self.assertRaisesRegex(ValueError, "index"):
                self.replay(memory, source, query)
        for options in ({"checkpoint_encoding": 1}, {"visual_read_enabled": 1}):
            with self.assertRaisesRegex(ValueError, "boolean"):
                self.replay(memory, source, 3, **options)
        with self.assertRaisesRegex(ValueError, "camera_order"):
            replay_visual_patch(memory, source, 3, camera_order=CAMERA_ORDER[::-1])
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.replay(memory, {}, 0)
        broken = observations(); broken["features"][1] = broken["features"][1].bfloat16()
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.replay(memory, broken, 3)
        broken = observations(); broken["image_masks"][1] = broken["image_masks"][1].float()
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.replay(memory, broken, 3)

    def test_replay_is_deterministic_and_consumes_no_forward_rng(self):
        memory, source = self.memory(), observations()
        before = torch.get_rng_state().clone()
        first, bank = self.replay(memory, source, 4, checkpoint_encoding=True)
        second, other = self.replay(memory, source, 4, checkpoint_encoding=True)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(bank.tokens, other.tokens))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
