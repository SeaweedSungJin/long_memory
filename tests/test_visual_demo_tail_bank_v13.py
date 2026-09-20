"""Synthetic CPU evidence-ingestion tests; no real cache, model, policy or GPU."""
import copy
from dataclasses import fields, replace
import unittest
from unittest.mock import patch

import torch

from run_scripts.robomme import visual_demo_tail_bank_v13 as core
from run_scripts.robomme.replay_visual_differential_v12 import replay_visual_differential
from run_scripts.robomme.visual_differential_memory_v12 import VisualDifferentialMemoryV12
from tests import test_visual_patch_memory_v11 as fixture
from tests.test_replay_visual_patch_v11 import GuardedColumn, ObservationOnlyMapping


CAMERAS = core.CAMERA_ORDER
CACHE, SIDECAR = "a" * 64, "b" * 64


def memory(mode="differential", awake=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(731)
        model = core.VisualDemoTailMemoryV13(core.VisualDifferentialConfig(feature_dim=8, hidden_dim=16), read_mode=mode)
        if awake:
            with torch.no_grad():
                model.output_projection.weight.normal_(std=.04)
        return model


def source(frames=(0, 16, 32, 48, 64, 80), n_demo=48, dtype=torch.bfloat16, grad=False):
    result = {key: [] for key in core.OBSERVATION_KEYS}
    for index, frame in enumerate(frames):
        feature, image, attention = fixture.inputs(seed=20+index, batch=1, dtype=dtype)
        # Different prefix padding/absolute image offsets remain mask-driven.
        if index % 2:
            feature = torch.cat((feature[:, :3], feature), dim=1)
            image = torch.cat((image.new_zeros((1, 3)), image), dim=1)
            attention = torch.cat((attention.new_zeros((1, 3)), attention), dim=1)
        row = (feature[0].clone().requires_grad_(grad), image[0], attention[0], frame, frame < n_demo)
        for key, value in zip(core.OBSERVATION_KEYS, row):
            result[key].append(value)
    return result


def sidecar(n_demo=48, grad=False):
    last = max(0, n_demo-16) if n_demo else -1
    frames = list(range(max(last+1, n_demo-15, 0), n_demo))
    generator = torch.Generator().manual_seed(719)
    payload = {"episode_id": 7, "cache_fingerprint": CACHE, "sidecar_fingerprint": SIDECAR,
        "images": torch.randn(len(frames), 2, 81, 8, generator=generator).bfloat16().requires_grad_(grad),
        "frames": torch.tensor(frames, dtype=torch.long), "is_demo": torch.ones(len(frames), dtype=torch.bool)}
    record = {"episode_id": 7, "n_demo": n_demo, "last_canonical_demo": last, "frames": frames}
    return payload, record


def merge(model, observations, query, payload=None, record=None, **kwargs):
    if payload is None:
        payload, record = sidecar()
    return core.merged_prefix_bank(model, observations, query, payload, record=record,
        episode_id=7, cache_fingerprint=CACHE, sidecar_fingerprint=SIDECAR,
        camera_order=CAMERAS, **kwargs)


def current(model, observations, query):
    return model.encode_observation(observations["features"][query][None], observations["image_masks"][query][None],
        observations["attention_masks"][query][None], [observations["frames"][query]],
        [observations["is_demo"][query]], camera_order=CAMERAS)


class VisualDemoTailBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_bank_equal(self, first, second):
        self.assertIsInstance(first, core.DifferentialBank)
        for field in fields(first):
            self.assertTrue(torch.equal(getattr(first, field.name), getattr(second, field.name)), field.name)

    def test_exact_inherited_fullwidth_parameters_initialization_and_rng(self):
        models, rngs = [], []
        for factory in (VisualDifferentialMemoryV12, core.VisualDemoTailMemoryV13,
                        lambda: core.VisualDemoTailMemoryV13(read_mode="current_only")):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(9111)
                models.append(factory())
                rngs.append(torch.get_rng_state().clone())
        self.assertEqual(len(list(models[0].named_parameters())), 14)
        self.assertEqual(sum(p.numel() for p in models[0].parameters()), 1253632)
        for model, rng in zip(models[1:], rngs[1:]):
            self.assertEqual(list(model.state_dict()), list(models[0].state_dict()))
            self.assertTrue(torch.equal(rng, rngs[0]))
            for name, parameter in model.state_dict().items():
                self.assertTrue(torch.equal(parameter, models[0].state_dict()[name]), name)
        self.assertFalse(torch.cuda.is_initialized())

    def test_image_record_is_exact_canonical_z_c_without_short_or_query_calls(self):
        model = memory()
        for dtype in (torch.float32, torch.bfloat16):
            original = fixture.observation(model, frames=(0, 48), demo=(True, False), dtype=dtype)
            image = original.features.gather(1, original.image_indices[..., None].expand(-1, -1, 8)).reshape(2, 2, 81, 8)
            before = image.clone()
            rng = torch.get_rng_state().clone()
            with patch.object(model, "encode_observation", side_effect=AssertionError("No fake query observation")), \
                    patch.object(model.query_norm, "forward", side_effect=AssertionError("No query norm")), \
                    patch.object(model.query_projection, "forward", side_effect=AssertionError("No discarded query")):
                record = model.encode_bank_images(image, original.frames, original.is_demo, camera_order=CAMERAS)
            self.assertEqual({f.name for f in fields(record)}, {"encoded", "content", "frames", "is_demo"})
            self.assertTrue(torch.equal(original.encoded, record.encoded))
            self.assertTrue(torch.equal(original.content, record.content))
            self.assertTrue(torch.equal(image, before))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_append_exact_v12_bank_and_transactional_invalid_inputs(self):
        model = memory()
        ordinary = fixture.observation(model, frames=(0, 0), demo=(True, True))
        first = core.BankImageRecord(ordinary.encoded, ordinary.content, ordinary.frames, ordinary.is_demo)
        expected = model.append(None, ordinary, valid=[True, False])
        actual = model.append_bank_images(None, first, valid=[True, False])
        self.assert_bank_equal(expected, actual)
        saved = replace(actual, **{field.name: getattr(actual, field.name).detach().clone() for field in fields(actual)})
        following = model.encode_bank_images(torch.ones(2, 2, 81, 8), [16, 16], [False, False], camera_order=CAMERAS)
        for invalid in (replace(following, frames=torch.tensor([0, 16])),
                        replace(following, content=torch.full_like(following.content, float("nan"))),
                        replace(following, encoded=following.encoded.double()),
                        replace(following, is_demo=torch.ones(2)),
                        replace(following, frames=torch.tensor([-1, 16]))):
            with self.assertRaises((ValueError, TypeError)):
                model.append_bank_images(actual, invalid)
            self.assert_bank_equal(actual, saved)
        with self.assertRaises(ValueError):
            model.append_bank_images(actual, following, valid=[1, 0])
        committed = model.append_bank_images(actual, following)
        self.assertEqual(committed.tokens.shape[1], 2)
        self.assert_bank_equal(actual, saved)
        demo_again = model.encode_bank_images(torch.ones(2, 2, 81, 8), [32, 32], [True, True], camera_order=CAMERAS)
        with self.assertRaisesRegex(ValueError, "demo evidence after execution"):
            model.append_bank_images(committed, demo_again)

    def test_input_layout_dtype_nonfinite_and_per_row_frame_validity(self):
        model = memory()
        images = torch.ones(2, 2, 81, 8)
        for value in (images[:, :1], images[:, :, :80], images.long(), torch.full_like(images, float("inf"))):
            with self.assertRaises(ValueError):
                model.encode_bank_images(value, [0, 16], [True, False], camera_order=CAMERAS)
        for frames in ([0., 16.], [-1, 16], [True, False], [0]):
            with self.assertRaises(ValueError):
                model.encode_bank_images(images, frames, [True, False], camera_order=CAMERAS)
        with self.assertRaises(ValueError):
            model.encode_bank_images(images, [0, 16], [1, 0], camera_order=CAMERAS)
        with self.assertRaises(ValueError):
            model.encode_bank_images(images, [0, 16], [True, False], camera_order=CAMERAS[::-1])

    def test_canonical_only_forward_and_all_parameter_input_gradients_match_v12(self):
        for mode in ("differential", "current_only"):
            new = memory(mode, awake=True)
            old = VisualDifferentialMemoryV12(new.config, read_mode=mode)
            old.load_state_dict(new.state_dict())
            original = source(frames=(0, 16, 32, 48), n_demo=0, dtype=torch.float32, grad=True)
            other = copy.deepcopy(original)
            payload, record = sidecar(0)
            bank = merge(new, original, 3, payload, record)
            actual = new.read(current(new, original, 3), bank)
            expected, reference = replay_visual_differential(old, other, 3, camera_order=CAMERAS)
            self.assert_bank_equal(bank, reference)
            self.assertTrue(torch.equal(actual, expected))
            actual.square().mean().backward(); expected.square().mean().backward()
            for (name, left), (_, right) in zip(new.named_parameters(), old.named_parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7, msg=name)
            for left, right in zip(original["features"], other["features"]):
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)

    def test_real_canonical_demo_bank_only_forward_and_gradients_match_v12(self):
        for mode in ("differential", "current_only"):
            new = memory(mode, awake=True)
            old = VisualDifferentialMemoryV12(new.config, read_mode=mode)
            old.load_state_dict(new.state_dict())
            original = source(frames=(0, 16, 32, 48), n_demo=32, dtype=torch.float32, grad=True)
            other = copy.deepcopy(original)
            # The canonical-only arm uses real canonical demo observations,
            # never asks the tail helper to accept a fake empty demo sidecar.
            images = torch.stack([original["features"][i][original["image_masks"][i]].reshape(2, 81, 8) for i in range(3)])
            encoded = new.encode_bank_images(images, [0, 16, 32], [True, True, False], camera_order=CAMERAS)
            bank = core.DifferentialBank(encoded.encoded[None], encoded.frames[None], encoded.is_demo[None],
                torch.ones(1, 3, dtype=torch.bool), encoded.content[None])
            actual = new.read(current(new, original, 3), bank)
            expected, reference = replay_visual_differential(old, other, 3, camera_order=CAMERAS)
            self.assert_bank_equal(bank, reference)
            self.assertTrue(torch.equal(actual, expected))
            actual.square().mean().backward(); expected.square().mean().backward()
            for (name, left), (_, right) in zip(new.named_parameters(), old.named_parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7, msg=name)
            for left, right in zip(original["features"], other["features"]):
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)
            fake, record = sidecar(0)
            with self.assertRaisesRegex(ValueError, "demo boundary"):
                merge(new, original, 3, fake, record)

    def test_merge_raw_chronology_preserves_original_query_parent_target_and_noise(self):
        model, observations = memory(), source()
        payload, record = sidecar()
        before = copy.deepcopy((observations, payload, record))
        bank = merge(model, observations, 4, payload, record)
        self.assertEqual(bank.frames.tolist(), [[0, 16, 32, *range(33, 48), 48]])
        self.assertEqual(bank.tokens.shape, (1, 19, 162, 16))
        self.assertEqual(bank.is_demo.tolist(), [[True]*18+[False]])
        self.assertEqual(observations["frames"][4], 64)  # ORIGINAL query index unchanged.
        self.assertEqual(record, before[2])
        for key in core.OBSERVATION_KEYS:
            for left, right in zip(observations[key], before[0][key]):
                if torch.is_tensor(left): self.assertTrue(torch.equal(left, right))
                else: self.assertEqual(left, right)
        self.assertTrue(torch.equal(payload["images"], before[1]["images"]))
        # q=4 reads only past feature/mask rows0..3 and metadata frame4.
        protected = ObservationOnlyMapping({key: GuardedColumn(observations[key], 4 if key in ("frames", "is_demo") else 3)
                                            for key in core.OBSERVATION_KEYS})
        protected.update(parent=object(), state=object(), actions=object(), targets=object(), noise=object())
        rng = torch.get_rng_state().clone()
        with patch.object(model.query_projection, "forward", side_effect=AssertionError("No historical query")):
            guarded = merge(model, protected, 4, payload, record)
        self.assert_bank_equal(bank, guarded)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key in ("features", "image_masks", "attention_masks"):
            self.assertEqual(protected[key].accesses, [0, 1, 2, 3])

    def test_bindings_tail_rule_current_future_duplicate_and_reordered_fail(self):
        model, original = memory(), source()
        payload, record = sidecar()
        mutations = [lambda p, r: p.update(episode_id=8), lambda p, r: p.update(cache_fingerprint="c"*64),
            lambda p, r: p.update(sidecar_fingerprint="d"*64), lambda p, r: p.update(targets=object()),
            lambda p, r: r.update(n_demo=64), lambda p, r: r.update(last_canonical_demo=31),
            lambda p, r: r.update(frames=list(range(32, 47))),
            lambda p, r: p.update(frames=p["frames"].flip(0)),
            lambda p, r: p["frames"].__setitem__(1, p["frames"][0]),
            lambda p, r: p.update(is_demo=torch.zeros_like(p["is_demo"])),
            lambda p, r: p.update(images=p["images"].float()),
            lambda p, r: p["images"].__setitem__((0, 0, 0, 0), float("nan"))]
        for mutate in mutations:
            p, r = copy.deepcopy((payload, record)); mutate(p, r)
            with self.assertRaises(ValueError):
                merge(model, original, 3, p, r)
        # No prospective current/tail image values are consulted on rejection.
        class FutureImages(dict):
            def __getitem__(self, key):
                if key == "images": raise AssertionError("Future image data accessed")
                return super().__getitem__(key)
        for frame in (47, 40):
            observations = copy.deepcopy(original)
            observations["frames"][3] = frame
            observations["is_demo"][3] = False
            with self.assertRaisesRegex(ValueError, "current/future"):
                merge(model, observations, 3, FutureImages(payload), record)
        for frames in ((0, 16, 32, 33, 64), (16, 0, 32, 48), (0, 16, 32, 32, 64)):
            with self.assertRaises(ValueError):
                merge(model, source(frames=frames), len(frames)-1, payload, record)
        for value in (48., True, torch.tensor([48]), torch.tensor(48, dtype=torch.float32)):
            observations = copy.deepcopy(original); observations["frames"][3] = value
            with self.assertRaises(ValueError):
                merge(model, observations, 3, payload, record)
        with self.assertRaisesRegex(ValueError, "action-query only"):
            merge(model, original, 2, payload, record)

    def test_awake_tail_and_earliest_canonical_gradients_checkpoint_parity(self):
        plain, checked = memory(awake=True), memory(awake=True)
        observations = source(grad=True)
        other = copy.deepcopy(observations)
        payload, record = sidecar(grad=True)
        other_payload = copy.deepcopy(payload)
        bank = merge(plain, observations, 4, payload, record)
        checked_bank = merge(checked, other, 4, other_payload, record, checkpoint_encoding=True)
        self.assert_bank_equal(bank, checked_bank)
        result = plain.read(current(plain, observations, 4), bank)
        checked_result = checked.read(current(checked, other, 4), checked_bank)
        self.assertTrue(torch.equal(result, checked_result))
        result.float().square().mean().backward(); checked_result.float().square().mean().backward()
        for grad in (payload["images"].grad, observations["features"][0].grad):
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.float().norm()), 0)
        self.assertTrue((payload["images"].grad.float().abs().sum((1, 2, 3)) > 0).all())
        for name in ("image_projection", "key_projection", "value_projection", "query_projection", "output_projection"):
            self.assertGreater(float(getattr(plain, name).weight.grad.norm()), 0)
        for (_, left), (_, right) in zip(plain.named_parameters(), checked.named_parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)
        self.assertTrue(torch.equal(payload["images"].grad, other_payload["images"].grad))
        self.assertIsNone(observations["features"][5].grad)

    def test_empty_zero_off_and_current_only_ignore_tail_content(self):
        for mode in ("differential", "current_only"):
            model, observations = memory(mode), source()
            payload, record = sidecar()
            bank = merge(model, observations, 3, payload, record)
            observation = current(model, observations, 3)
            self.assertTrue(torch.equal(model.read(observation, bank), observation.features))
            with torch.no_grad(): model.output_projection.weight.normal_(std=.04)
            self.assertIs(model.read(observation, bank, enabled=False), observation.features)
            empty_payload, empty_record = sidecar(0)
            empty_source = source(frames=(0,), n_demo=0)
            empty = merge(model, empty_source, 0, empty_payload, empty_record)
            self.assertEqual(empty.tokens.shape, (1, 0, 162, 16))
            empty_observation = current(model, empty_source, 0)
            self.assertIs(model.read(empty_observation, empty), empty_observation.features)
        model, observations = memory("current_only", awake=True), source(grad=True)
        payload, record = sidecar(grad=True)
        bank = merge(model, observations, 3, payload, record)
        model.read(current(model, observations, 3), bank).float().square().mean().backward()
        self.assertIsNone(payload["images"].grad)
        self.assertTrue(all(x.grad is None for x in observations["features"][:3]))

    def test_actual_sidecar_planner_empty_record_without_current_features(self):
        import numpy as np
        from run_scripts.robomme.demo_tail_sidecar_v13 import plan_demo_tail

        payload, _ = sidecar(0)
        record = {"episode_id": 7, **plan_demo_tail(np.zeros(33, dtype=np.bool_), [0, 16, 32])}
        self.assertEqual(record["last_canonical_demo"], -1)
        self.assertEqual(record["frames"], [])
        observations = source(frames=(0, 16, 32), n_demo=0)
        protected = ObservationOnlyMapping({key: GuardedColumn(observations[key],
            0 if key in ("frames", "is_demo") else -1) for key in core.OBSERVATION_KEYS})
        model = memory(awake=True)
        with patch.object(model, "encode_bank_images", side_effect=AssertionError("No empty encoding")):
            bank = merge(model, protected, 0, payload, record)
        self.assert_bank_equal(bank, model.empty_bank(1))
        for key in ("features", "image_masks", "attention_masks"):
            self.assertEqual(protected[key].accesses, [])
        with self.assertRaises(ValueError):
            merge(model, protected, 0, payload, {**record, "last_canonical_demo": None})


if __name__ == "__main__":
    unittest.main()
