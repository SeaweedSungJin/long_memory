"""CPU-only observation/permutation contracts; no actual cache/model/training."""
import copy
from dataclasses import fields
import inspect
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from run_scripts.robomme import segment_probe_inputs_v15 as inputs
from run_scripts.robomme.framewise_demo_tail_v13 import build_framewise_prefix
from run_scripts.robomme.replay_visual_patch_v11 import _columns, _rows
from tests import test_visual_demo_tail_bank_v13 as fixture
from tests.test_replay_visual_patch_v11 import GuardedColumn, ObservationOnlyMapping


def memory(*, train=True):
    model = fixture.memory(awake=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(train and name in inputs.TRAINABLE_NAMES)
    return model


def build(model, observations, query=5, *, seed=None, payload=None, record=None):
    if payload is None:
        payload, record = fixture.sidecar()
    return inputs.build_probe_inputs(model, observations, query,
        sidecar=payload, record=record, episode_id=7,
        cache_fingerprint=fixture.CACHE, sidecar_fingerprint=fixture.SIDECAR,
        permutation_seed=seed)


def snapshot(source):
    return {key: [value.clone() if isinstance(value, torch.Tensor) else value for value in values]
            for key, values in source.items()}


class SegmentProbeInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_bank_equal(self, left, right):
        self.assertEqual(type(left), type(right))
        for field in fields(left):
            self.assertTrue(torch.equal(getattr(left, field.name), getattr(right, field.name)), field.name)

    def assert_observation_equal(self, left, right):
        self.assertEqual(type(left), type(right))
        for field in fields(left):
            self.assertTrue(torch.equal(getattr(left, field.name), getattr(right, field.name)), field.name)

    def test_identity_exact_original_framewise_prefix_and_current_observation(self):
        model, observations = memory(), fixture.source()
        payload, record = fixture.sidecar()
        actual, bank, info = build(model, observations, payload=payload, record=record)
        with torch.no_grad():
            expected = build_framewise_prefix(model, observations, 5,
                camera_order=fixture.CAMERAS, include_tail=True, sidecar=payload, record=record,
                episode_id=7, cache_fingerprint=fixture.CACHE, sidecar_fingerprint=fixture.SIDECAR)
        self.assert_bank_equal(bank, expected)
        self.assert_observation_equal(actual, fixture.current(model, observations, 5))
        self.assertEqual(info['candidate_frames'], [0, 16, 32, *range(33, 48), 48, 64])
        self.assertEqual(info['destination_to_source_indices'], list(range(20)))
        self.assertEqual(info['destination_to_source_frames'], info['candidate_frames'])
        self.assertEqual(info['permutation_kind'], 'identity')
        self.assertFalse(info['permutation_effective'])

    def test_all_demo_contents_move_both_cameras_and_destination_metadata_stays(self):
        model, observations = memory(), fixture.source()
        payload, record = fixture.sidecar()
        ordinary, original_bank, _ = build(model, observations, payload=payload, record=record)
        calls = []
        actual_encoder = model.encode_bank_images

        def encode(images, frames, demos, **kwargs):
            calls.append((images.clone(), frames.clone(), demos.clone(), torch.is_grad_enabled()))
            return actual_encoder(images, frames, demos, **kwargs)

        with patch.object(model, 'encode_bank_images', side_effect=encode):
            current, bank, info = build(model, observations, seed=913, payload=payload, record=record)
        self.assert_observation_equal(current, ordinary)
        self.assertTrue(info['permutation_effective'])
        self.assertEqual(info['candidate_is_demo'], [True] * 18 + [False] * 2)
        self.assertEqual(sorted(info['destination_to_source_indices']), list(range(20)))
        self.assertTrue(all(index != source for index, source in
                            enumerate(info['destination_to_source_indices'][:18])))
        self.assertEqual(info['destination_to_source_indices'][18:], [18, 19])
        self.assertEqual(len(calls), 20 + 18)
        self.assertTrue(all(call[0].shape == (1, 2, 81, 8) and not call[3] for call in calls))

        raw = {int(row[3]): inputs.core._canonical_images(row, model)
               for row in _rows(_columns(observations), 5, model)}
        raw.update({int(frame): payload['images'][index] for index, frame in enumerate(payload['frames'])})
        for destination, source in enumerate(info['destination_to_source_indices'][:18]):
            destination_frame = info['candidate_frames'][destination]
            source_frame = info['candidate_frames'][source]
            images, frames, demos, _ = calls[20 + destination]
            self.assertTrue(torch.equal(images[0], raw[source_frame]))
            self.assertEqual(frames.tolist(), [destination_frame])
            self.assertEqual(demos.tolist(), [True])
            with torch.no_grad():
                expected = actual_encoder(raw[source_frame][None], [destination_frame], [True],
                                          camera_order=fixture.CAMERAS)
            self.assertTrue(torch.equal(bank.tokens[0, destination], expected.encoded[0]))
            self.assertTrue(torch.equal(bank.content[0, destination], expected.content[0]))
        self.assertTrue(torch.equal(bank.frames, original_bank.frames))
        self.assertTrue(torch.equal(bank.is_demo, original_bank.is_demo))
        self.assertTrue(torch.equal(bank.tokens[:, 18:], original_bank.tokens[:, 18:]))
        self.assertTrue(torch.equal(bank.content[:, 18:], original_bank.content[:, 18:]))
        # Moving pre-encoded Z instead would incorrectly move the source time.
        permuted_old_z = original_bank.tokens[:, info['destination_to_source_indices'][:18]]
        self.assertFalse(torch.equal(bank.tokens[:, :18], permuted_old_z))

    def test_target_remap_follows_image_bijection_and_never_selects_permutation(self):
        model, observations = memory(), fixture.source()
        _, _, info = build(model, observations, seed=40)
        positives = [0, 33, 47]
        remapped = inputs.remap_positive_frames(positives, info)
        self.assertEqual(len(remapped), len(positives))
        self.assertEqual(remapped, sorted(remapped))
        self.assertEqual({info['destination_to_source_frames'][info['candidate_frames'].index(f)]
                          for f in remapped}, set(positives))
        before = copy.deepcopy(info)
        for target in ([16], [32, 34], [], info['candidate_frames'][:18]):
            inputs.remap_positive_frames(target, info)
        self.assertEqual(info, before)
        _, _, again = build(model, observations, seed=40)
        self.assertEqual(again, info)
        _, _, identity = build(model, observations)
        self.assertEqual(inputs.remap_positive_frames(positives, identity), positives)
        params = inspect.signature(inputs.build_probe_inputs).parameters
        self.assertFalse({'positive_frames', 'labels', 'targets', 'subgoal'} & set(params))

    def test_permutation_deterministic_label_content_independent_and_seed_bound(self):
        model, observations = memory(), fixture.source()
        payload, record = fixture.sidecar()
        _, _, first = build(model, observations, seed=1, payload=payload, record=record)
        altered = copy.deepcopy(observations)
        altered['features'] = [torch.zeros_like(value) for value in altered['features']]
        changed_payload = copy.deepcopy(payload)
        changed_payload['images'].zero_()
        _, _, second = build(model, altered, seed=1, payload=changed_payload, record=record)
        self.assertEqual(first, second)
        _, _, other = build(model, observations, seed=2, payload=payload, record=record)
        self.assertNotEqual(first['destination_to_source_indices'], other['destination_to_source_indices'])
        frames, demos = first['candidate_frames'], first['candidate_is_demo']
        self.assertNotEqual(inputs._mapping(frames, demos, seed=1, episode_id=7, query_frame=80),
                            inputs._mapping(frames, demos, seed=1, episode_id=8, query_frame=80))
        self.assertNotEqual(inputs._mapping(frames, demos, seed=1, episode_id=7, query_frame=80),
                            inputs._mapping(frames, demos, seed=1, episode_id=7, query_frame=96))

    def test_no_global_rng_input_weight_flag_or_grad_mutation(self):
        model, observations = memory(), fixture.source()
        payload, record = fixture.sidecar()
        old_observations, old_payload = snapshot(observations), copy.deepcopy(payload)
        weights = {name: value.clone() for name, value in model.state_dict().items()}
        flags = {name: value.requires_grad for name, value in model.named_parameters()}
        torch_rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        for seed in (None, 0, 982):
            build(model, observations, seed=seed, payload=payload, record=record)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        actual_numpy = np.random.get_state()
        self.assertEqual(numpy_rng[0], actual_numpy[0])
        self.assertTrue(np.array_equal(numpy_rng[1], actual_numpy[1]))
        self.assertEqual(numpy_rng[2:], actual_numpy[2:])
        for key, values in observations.items():
            for actual, expected in zip(values, old_observations[key]):
                if isinstance(actual, torch.Tensor):
                    self.assertTrue(torch.equal(actual, expected))
                else:
                    self.assertEqual(actual, expected)
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, old_payload[key]))
            else:
                self.assertEqual(value, old_payload[key])
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, weights[name]))
        self.assertEqual(flags, {name: value.requires_grad for name, value in model.named_parameters()})
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_q_key_gradients_live_but_past_encoder_sidecar_and_future_are_detached(self):
        for seed in (None, 92):
            model, observations = memory(), fixture.source(grad=True)
            payload, record = fixture.sidecar(grad=True)
            current, bank, _ = build(model, observations, 4, seed=seed, payload=payload, record=record)
            self.assertTrue(current.queries.requires_grad)
            self.assertFalse(bank.tokens.requires_grad)
            self.assertFalse(bank.content.requires_grad)
            logits = current.queries @ model.key_projection(bank.tokens.flatten(1, 2)).transpose(-1, -2)
            logits.float().square().mean().backward()
            for name, parameter in model.named_parameters():
                if name in inputs.TRAINABLE_NAMES:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
                else:
                    self.assertIsNone(parameter.grad, name)
            for index, feature in enumerate(observations['features']):
                if index == 4:
                    self.assertIsNotNone(feature.grad)
                    self.assertGreater(float(feature.grad.abs().sum()), 0)
                else:
                    self.assertIsNone(feature.grad)
            self.assertIsNone(payload['images'].grad)

    def test_actual_probe_probability_loss_and_target_mapping_integration(self):
        from run_scripts.robomme import segment_retrieval_probe_v15 as probe
        original, observations = memory(train=False), fixture.source()
        model = probe.clone_qk_probe(original)
        frozen = probe.snapshot_frozen_parameters(model)
        for seed in (None, 981):
            model.zero_grad(set_to_none=True)
            current, bank, info = build(model, observations, seed=seed)
            frames = info['candidate_frames']
            original_positive = torch.tensor([[frame in {0, 33, 47} for frame in frames]])
            positive_frames = inputs.remap_positive_frames([0, 33, 47], info)
            positive = torch.tensor([[frame in positive_frames for frame in frames]])
            indices = torch.tensor([info['destination_to_source_indices']], dtype=torch.long)
            self.assertTrue(torch.equal(positive,
                probe.content_following_positive_mask(original_positive, indices, bank.valid)))
            probability = probe.past_frame_probabilities(model, current, bank)
            loss = probe.segment_mass_loss(probability, positive, bank.valid)
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertGreater(float(loss.detach()), 0.)
            loss.backward()
            self.assertTrue(probe.assert_qk_scope(model, frozen, require_gradients=True))
            self.assertGreater(float(model.query_projection.weight.grad.norm()), 0.)
            self.assertGreater(float(model.key_projection.weight.grad.norm()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in original.parameters()))

    def test_current_and_history_guards_ignore_extra_columns_and_never_access_future(self):
        for seed in (None, 31):
            observations = fixture.source()
            guarded = ObservationOnlyMapping({key: GuardedColumn(value, 4)
                                              for key, value in observations.items()})
            guarded.update(actions=object(), targets=object(), state=object(), simple_subgoal=object())
            current, bank, info = build(memory(), guarded, 4, seed=seed)
            self.assertEqual(current.frames.tolist(), [64])
            self.assertEqual(info['candidate_frames'][-1], 48)
            self.assertTrue(bool((bank.frames < 64).all()))
            self.assertEqual(guarded['features'].accesses.count(4), 1)
            self.assertTrue(all(index <= 4 for index in guarded['features'].accesses))

    def test_future_mutation_does_not_change_any_result(self):
        model, observations = memory(), fixture.source()
        first = build(model, observations, 4, seed=209)
        changed = copy.deepcopy(observations)
        for key in inputs.OBSERVATION_KEYS:
            changed[key][5] = object()
        other = build(model, changed, 4, seed=209)
        self.assert_observation_equal(first[0], other[0])
        self.assert_bank_equal(first[1], other[1])
        self.assertEqual(first[2], other[2])

    def test_empty_and_single_demo_controls_are_explicit_identity(self):
        model = memory()
        for n_demo, frames, query in ((0, (0, 16), 0), (0, (0, 16, 32), 2), (1, (0, 1, 17), 2)):
            observations = fixture.source(frames=frames, n_demo=n_demo)
            payload, record = fixture.sidecar(n_demo)
            normal = build(model, observations, query, payload=payload, record=record)
            control = build(model, observations, query, seed=5, payload=payload, record=record)
            self.assert_bank_equal(normal[1], control[1])
            self.assertFalse(control[2]['permutation_effective'])
            self.assertEqual(control[2]['permutation_kind'], inputs.PERMUTATION_KIND)
            self.assertEqual(inputs.remap_positive_frames([], control[2]), [])

    def test_all_frozen_eval_allowed_but_unfrozen_encoder_or_other_variant_rejected(self):
        model = memory(train=False)
        current, bank, _ = build(model, fixture.source(), seed=9)
        self.assertFalse(current.queries.requires_grad)
        self.assertFalse(bank.tokens.requires_grad)
        model.image_projection.weight.requires_grad_(True)
        with patch.object(model, 'encode_bank_images', side_effect=AssertionError('Fail before encoding')):
            with self.assertRaisesRegex(ValueError, 'non-Q/K'):
                build(model, fixture.source())
        model = memory()
        model.read_mode = 'current_only'
        with self.assertRaisesRegex(ValueError, 'differential'):
            build(model, fixture.source())
        with self.assertRaises(TypeError):
            build(object(), fixture.source())

    def test_invalid_seed_prefix_tail_or_query_rejected_before_encoding(self):
        model = memory()
        for seed in (True, -1, 1.5, '3'):
            with patch.object(model, 'encode_bank_images', side_effect=AssertionError('Must fail first')):
                with self.assertRaises(ValueError):
                    build(model, fixture.source(), seed=seed)
        for mutate in (lambda p: p.update(episode_id=9),
                       lambda p: p['frames'].__setitem__(-1, 80),
                       lambda p: p['frames'].__setitem__(1, 33),
                       lambda p: p.update(images=torch.full_like(p['images'], float('nan')))):
            payload, record = fixture.sidecar(); mutate(payload)
            with patch.object(model, 'encode_bank_images', side_effect=AssertionError('Must reject first')):
                with self.assertRaises(ValueError):
                    build(model, fixture.source(), seed=5, payload=payload, record=record)
        with self.assertRaisesRegex(ValueError, 'action-query'):
            build(model, fixture.source(), 2)
        with self.assertRaises(ValueError):
            build(model, fixture.source(frames=(0, 16, 32, 48, 48, 80)), seed=5)
        with self.assertRaises(ValueError):
            build(model, fixture.source(frames=(0, 16, 31, 48, 64, 80)), seed=5)
        for query in (-1, True, 3.0):
            with self.assertRaises(ValueError):
                build(model, fixture.source(), query)

    def test_target_remap_rejects_unknown_execution_duplicate_and_malformed_mapping(self):
        _, _, info = build(memory(), fixture.source(), seed=98)
        for positives in ([999], [48], [0, 0], [False], [0.0], '0', {0}):
            with self.assertRaises(ValueError):
                inputs.remap_positive_frames(positives, info)
        for mutate in (
            lambda i: i['candidate_frames'].__setitem__(1, 0),
            lambda i: i['candidate_is_demo'].__setitem__(0, False),
            lambda i: i['candidate_is_demo'].__setitem__(0, 1),
            lambda i: i['destination_to_source_indices'].__setitem__(0, i['destination_to_source_indices'][1]),
            lambda i: i['destination_to_source_indices'].__setitem__(0, True),
            lambda i: i['destination_to_source_frames'].__setitem__(0, 999),
        ):
            broken = copy.deepcopy(info); mutate(broken)
            with self.assertRaises(ValueError):
                inputs.remap_positive_frames([0], broken)
        broken = copy.deepcopy(info)
        broken['destination_to_source_indices'][-1], broken['destination_to_source_indices'][-2] = (
            broken['destination_to_source_indices'][-2], broken['destination_to_source_indices'][-1])
        broken['destination_to_source_frames'] = [broken['candidate_frames'][i]
                                                 for i in broken['destination_to_source_indices']]
        with self.assertRaisesRegex(ValueError, 'execution must retain identity'):
            inputs.remap_positive_frames([0], broken)


if __name__ == '__main__':
    unittest.main()
