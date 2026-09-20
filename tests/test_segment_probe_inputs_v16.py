"""Synthetic CPU V16 replay/scope tests; no saved model/cache/GPU is loaded."""
import copy
from dataclasses import fields, replace
import inspect
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from run_scripts.robomme import segment_probe_inputs_v15 as old
from run_scripts.robomme import segment_probe_inputs_v16 as core
from run_scripts.robomme.segment_retrieval_probe_v15 import past_frame_probabilities, segment_mass_loss
from run_scripts.robomme.visual_demo_tail_bank_v13 import VisualDemoTailMemoryV13, VisualDifferentialConfig
from tests import test_visual_demo_tail_bank_v13 as fixture
from tests.test_replay_visual_patch_v11 import GuardedColumn, ObservationOnlyMapping


def memory(projection=True):
    model = fixture.memory(awake=True)
    core.configure_scope(model, train_projection=projection)
    return model


def build(model, observations, query=5, *, seed=None, payload=None, record=None, legacy=False):
    if payload is None:
        payload, record = fixture.sidecar()
    function = old.build_probe_inputs if legacy else core.build_probe_inputs_v16
    return function(model, observations, query, sidecar=payload, record=record,
        episode_id=7, cache_fingerprint=fixture.CACHE, sidecar_fingerprint=fixture.SIDECAR,
        permutation_seed=seed)


def positive_mask(info):
    positives = core.remap_positive_frames([0, 33, 47], info)
    return torch.tensor([[frame in positives for frame in info['candidate_frames']]], dtype=torch.bool)


def loss(model, current, bank, info):
    return segment_mass_loss(past_frame_probabilities(model, current, bank), positive_mask(info), bank.valid)


def norm_or_zero(gradient):
    return 0. if gradient is None else float(gradient.float().norm())


class SegmentProbeInputsV16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def assert_record_equal(self, left, right):
        self.assertEqual(type(left), type(right))
        for field in fields(left):
            self.assertTrue(torch.equal(getattr(left, field.name), getattr(right, field.name)), field.name)

    def test_configure_exact_two_or_three_original_weights_bias_frozen_no_rng_or_value_change(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9161)
            model = VisualDemoTailMemoryV13(VisualDifferentialConfig())
        original = {name: value.clone() for name, value in model.state_dict().items()}
        self.assertEqual(len(original), 14)
        for train_projection, count in ((False, 131072), (True, 655360)):
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            rng = torch.get_rng_state().clone()
            selected = core.configure_scope(model, train_projection)
            names = core.EXPANDED_PARAMETER_NAMES if train_projection else core.QK_PARAMETER_NAMES
            parameters = dict(model.named_parameters())
            self.assertEqual([id(p) for p in selected], [id(parameters[name]) for name in names])
            self.assertEqual(sum(p.numel() for p in selected), count)
            self.assertEqual({n for n, p in parameters.items() if p.requires_grad}, set(names))
            self.assertFalse(model.image_projection.bias.requires_grad)
            self.assertTrue(all(p.grad is None for p in parameters.values()))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for name, value in original.items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        flags = {name: p.requires_grad for name, p in model.named_parameters()}
        for bad in (0, 1, None, 'qkp'):
            with self.assertRaises(ValueError):
                core.configure_scope(model, bad)
            self.assertEqual(flags, {name: p.requires_grad for name, p in model.named_parameters()})
        self.assertFalse(torch.cuda.is_initialized())

    def test_frozen_projection_exact_v15_current_bank_info_probabilities_and_qk_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            for seed in (None, 819):
                model = memory(False); legacy = copy.deepcopy(model)
                # Actual nonempty sidecars are strictly BF16. FP32 is tested
                # only with a legitimate no-demo/empty-sidecar trajectory.
                n_demo = 48 if dtype == torch.bfloat16 else 0
                observations = fixture.source(dtype=dtype, n_demo=n_demo)
                payload, record = fixture.sidecar(n_demo)
                current, bank, info = build(model, observations, seed=seed, payload=payload, record=record)
                old_current, old_bank, old_info = build(legacy, observations, seed=seed,
                                                       payload=payload, record=record, legacy=True)
                self.assert_record_equal(current, old_current)
                self.assert_record_equal(bank, old_bank)
                self.assertEqual(info, old_info)
                p = past_frame_probabilities(model, current, bank)
                old_p = past_frame_probabilities(legacy, old_current, old_bank)
                self.assertTrue(torch.equal(p, old_p))
                positives = torch.zeros_like(bank.valid); positives[:, :3] = True
                segment_mass_loss(p, positives, bank.valid).backward()
                segment_mass_loss(old_p, positives, old_bank.valid).backward()
                for name in core.QK_PARAMETER_NAMES:
                    self.assertTrue(torch.equal(dict(model.named_parameters())[name].grad,
                                                dict(legacy.named_parameters())[name].grad), name)
                self.assertIsNone(model.image_projection.weight.grad)

    def test_projection_has_independent_current_and_historical_retrieval_gradient(self):
        for seed in (None, 916):
            for branch in ('past_only', 'current_only'):
                model = memory(True)
                observations = fixture.source(grad=True)
                payload, record = fixture.sidecar(grad=True)
                current, bank, info = build(model, observations, 4, seed=seed, payload=payload, record=record)
                self.assertTrue(bank.tokens.requires_grad)
                self.assertTrue(bank.content.requires_grad)
                self.assertTrue(current.queries.requires_grad)
                if branch == 'past_only':
                    current = replace(current, queries=current.queries.detach())
                else:
                    bank = replace(bank, tokens=bank.tokens.detach(), content=bank.content.detach())
                loss(model, current, bank, info).backward()
                gradient = model.image_projection.weight.grad
                self.assertIsNotNone(gradient, (seed, branch))
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(norm_or_zero(gradient), 0., (seed, branch))
                if branch == 'past_only':
                    self.assertIsNone(model.query_projection.weight.grad)
                    self.assertGreater(norm_or_zero(observations['features'][0].grad), 0.)
                    self.assertGreater(norm_or_zero(payload['images'].grad), 0.)
                    self.assertIsNone(observations['features'][4].grad)
                else:
                    self.assertGreater(norm_or_zero(model.query_projection.weight.grad), 0.)
                    self.assertGreater(norm_or_zero(observations['features'][4].grad), 0.)
                    self.assertTrue(all(value.grad is None for value in observations['features'][:4]))
                    self.assertIsNone(payload['images'].grad)
                self.assertIsNone(observations['features'][5].grad)  # Future never enters.
                for name, parameter in model.named_parameters():
                    if name not in core.EXPANDED_PARAMETER_NAMES:
                        self.assertFalse(parameter.requires_grad, name)
                        self.assertIsNone(parameter.grad, name)

    def test_canonical_and_tail_past_paths_each_reach_projection_without_current(self):
        for branch in ('canonical', 'tail'):
            model, observations = memory(True), fixture.source(grad=True)
            payload, record = fixture.sidecar(grad=True)
            current, bank, info = build(model, observations, 4, payload=payload, record=record)
            current = replace(current, queries=current.queries.detach())
            tail = (bank.frames >= 33) & (bank.frames < 48)
            selected = tail if branch == 'tail' else ~tail
            bank = replace(bank, tokens=torch.where(selected[..., None, None], bank.tokens, bank.tokens.detach()))
            loss(model, current, bank, info).backward()
            self.assertGreater(norm_or_zero(model.image_projection.weight.grad), 0., branch)
            self.assertIsNone(observations['features'][4].grad)
            if branch == 'tail':
                self.assertGreater(norm_or_zero(payload['images'].grad), 0.)
                self.assertTrue(all(norm_or_zero(value.grad) == 0. for value in observations['features']))
            else:
                self.assertGreater(norm_or_zero(observations['features'][0].grad), 0.)
                self.assertEqual(norm_or_zero(payload['images'].grad), 0.)

    def test_two_updates_only_selected_change_and_every_call_rebuilds_past_encodings(self):
        for projection in (False, True):
            model, observations = memory(projection), fixture.source()
            names = set(core.EXPANDED_PARAMETER_NAMES if projection else core.QK_PARAMETER_NAMES)
            initial = {name: p.detach().clone() for name, p in model.named_parameters()}
            optimizer = torch.optim.AdamW(core.configure_scope(model, projection), lr=1e-3, weight_decay=0.)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                current, bank, info = build(model, observations)
                previous = bank.tokens.detach().clone()
                loss(model, current, bank, info).backward()
                for name, parameter in model.named_parameters():
                    if name in names:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                        self.assertGreater(norm_or_zero(parameter.grad), 0., name)
                    else:
                        self.assertIsNone(parameter.grad, name)
                optimizer.step()
                new_current, new_bank, new_info = build(model, observations)
                self.assertEqual(info, new_info)
                self.assertEqual(torch.equal(previous, new_bank.tokens), not projection)
                self.assertFalse(torch.equal(current.queries.detach(), new_current.queries.detach()))
                self.assert_record_equal(new_current, fixture.current(model, observations, 5))
            changed = {n for n, p in model.named_parameters() if not torch.equal(initial[n], p)}
            self.assertEqual(changed, names)
            self.assertEqual(len(optimizer.state), len(names))

    def test_permutation_moves_raw_content_with_live_projection_at_fixed_time_current_execution_unchanged(self):
        model, observations = memory(True), fixture.source()
        payload, record = fixture.sidecar()
        current, bank, _ = build(model, observations, payload=payload, record=record)
        calls, original = [], model.encode_bank_images
        def encode(images, frames, demos, **kwargs):
            value = original(images, frames, demos, **kwargs)
            calls.append((images.clone(), frames.clone(), demos.clone(), torch.is_grad_enabled(), value.encoded.requires_grad))
            return value
        with patch.object(model, 'encode_bank_images', side_effect=encode):
            moved_current, moved, info = build(model, observations, seed=27, payload=payload, record=record)
        self.assert_record_equal(current, moved_current)
        self.assertEqual(len(calls), 20 + 18)
        self.assertTrue(all(call[3] and call[4] for call in calls))
        self.assertTrue(torch.equal(bank.frames, moved.frames))
        self.assertTrue(torch.equal(bank.is_demo, moved.is_demo))
        self.assertTrue(torch.equal(bank.tokens[:, 18:], moved.tokens[:, 18:]))
        self.assertTrue(torch.equal(bank.content[:, 18:], moved.content[:, 18:]))
        raw = old._original_images(model, old._columns(observations), 5, payload, record,
            episode_id=7, cache_fingerprint=fixture.CACHE, sidecar_fingerprint=fixture.SIDECAR, current_frame=80)
        for destination, source in enumerate(info['destination_to_source_indices'][:18]):
            self.assertNotEqual(destination, source)
            images, frames, demos, _, _ = calls[20 + destination]
            self.assertTrue(torch.equal(images[0], raw[source][1]))
            self.assertEqual(frames.tolist(), [info['candidate_frames'][destination]])
            self.assertEqual(demos.tolist(), [True])
            expected = original(images, frames, demos, camera_order=fixture.CAMERAS)
            self.assertTrue(torch.equal(expected.encoded[0], moved.tokens[0, destination]))
        self.assertEqual(core.remap_positive_frames([0, 33, 47], info), old.remap_positive_frames([0, 33, 47], info))

    def test_builder_preserves_flags_existing_gradients_weights_inputs_buffers_and_global_rng(self):
        model, observations = memory(True), fixture.source()
        payload, record = fixture.sidecar()
        model.image_projection.weight.grad = torch.ones_like(model.image_projection.weight)
        original_grad = model.image_projection.weight.grad.clone()
        weights = {name: value.clone() for name, value in model.state_dict().items()}
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        flags = {name: p.requires_grad for name, p in model.named_parameters()}
        source, tail = copy.deepcopy(observations), copy.deepcopy(payload)
        torch_rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        for seed in (None, 919):
            build(model, observations, seed=seed, payload=payload, record=record)
        self.assertTrue(torch.equal(model.image_projection.weight.grad, original_grad))
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        self.assertEqual(numpy_rng[0], np.random.get_state()[0])
        self.assertTrue(np.array_equal(numpy_rng[1], np.random.get_state()[1]))
        self.assertEqual(numpy_rng[2:], np.random.get_state()[2:])
        self.assertEqual(flags, {name: p.requires_grad for name, p in model.named_parameters()})
        for name, value in weights.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        for name, value in buffers.items():
            self.assertTrue(torch.equal(value, dict(model.named_buffers())[name]), name)
        for key, values in observations.items():
            for actual, expected in zip(values, source[key]):
                if isinstance(actual, torch.Tensor):
                    self.assertTrue(torch.equal(actual, expected))
                else:
                    self.assertEqual(actual, expected)
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, tail[key]))
            else:
                self.assertEqual(value, tail[key])
        self.assertFalse(torch.cuda.is_initialized())

    def test_eval_context_is_respected_without_temporary_flag_mutation(self):
        model, observations = memory(True), fixture.source()
        flags = {name: p.requires_grad for name, p in model.named_parameters()}
        with torch.no_grad():
            current, bank, _ = build(model, observations, seed=17)
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(current.queries.requires_grad)
            self.assertFalse(bank.tokens.requires_grad)
        self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(flags, {name: p.requires_grad for name, p in model.named_parameters()})
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        current, bank, _ = build(model, observations)
        self.assertFalse(current.queries.requires_grad)
        self.assertFalse(bank.tokens.requires_grad)

    def test_no_label_action_state_or_future_read_and_current_accessed_once(self):
        self.assertEqual(list(inspect.signature(core.build_probe_inputs_v16).parameters),
                         list(inspect.signature(old.build_probe_inputs).parameters))
        for seed in (None, 721):
            original = fixture.source()
            guarded = ObservationOnlyMapping({key: GuardedColumn(value, 4) for key, value in original.items()})
            guarded.update(labels=object(), actions=object(), targets=object(), state=object(), simple_subgoal=object())
            model = memory(True)
            current, bank, info = build(model, guarded, 4, seed=seed)
            self.assertEqual(current.frames.tolist(), [64])
            self.assertTrue(bool((bank.frames < 64).all()))
            self.assertEqual(guarded['features'].accesses.count(4), 1)
            changed = copy.deepcopy(original)
            for key in core.OBSERVATION_KEYS:
                changed[key][5] = object()
            later, later_bank, later_info = build(model, changed, 4, seed=seed)
            self.assert_record_equal(current, later)
            self.assert_record_equal(bank, later_bank)
            self.assertEqual(info, later_info)

    def test_invalid_scope_seed_causality_binding_and_passive_query_rejected(self):
        model = memory(True)
        for name in ('image_projection.bias', 'bank_norm.weight', 'value_projection.weight'):
            parameter = dict(model.named_parameters())[name]; parameter.requires_grad_(True)
            with patch.object(model, 'encode_bank_images', side_effect=AssertionError('Must reject scope first')):
                with self.assertRaisesRegex(ValueError, 'permits only'):
                    build(model, fixture.source())
            parameter.requires_grad_(False)
        model.key_projection.weight.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, 'permits only'):
            build(model, fixture.source())
        core.configure_scope(model, True)
        for seed in (True, -1, 1.2):
            with self.assertRaises(ValueError):
                build(model, fixture.source(), seed=seed)
        for mutate in (lambda p: p.update(episode_id=8),
                       lambda p: p.update(cache_fingerprint='wrong'),
                       lambda p: p['frames'].__setitem__(-1, 80),
                       lambda p: p['frames'].__setitem__(1, 33)):
            payload, record = fixture.sidecar(); mutate(payload)
            with patch.object(model, 'encode_bank_images', side_effect=AssertionError('Must reject before encoding')):
                with self.assertRaises(ValueError):
                    build(model, fixture.source(), payload=payload, record=record)
        with self.assertRaisesRegex(ValueError, 'action-query'):
            build(model, fixture.source(), 2)
        with self.assertRaises(ValueError):
            build(model, fixture.source(frames=(0, 16, 32, 48, 48, 80)))
        with self.assertRaises(ValueError):
            build(model, fixture.source(frames=(0, 16, 31, 48, 64, 80)))
        model.read_mode = 'current_only'
        with self.assertRaises(ValueError):
            build(model, fixture.source())

    def test_empty_and_single_demo_identity_contract_unchanged(self):
        model = memory(True)
        for n_demo, frames, query in ((0, (0, 16), 0), (0, (0, 16, 32), 2), (1, (0, 1, 17), 2)):
            observations = fixture.source(frames=frames, n_demo=n_demo)
            payload, record = fixture.sidecar(n_demo)
            ordinary = build(model, observations, query, payload=payload, record=record)
            moved = build(model, observations, query, seed=3, payload=payload, record=record)
            self.assert_record_equal(ordinary[0], moved[0])
            self.assert_record_equal(ordinary[1], moved[1])
            self.assertFalse(moved[2]['permutation_effective'])


if __name__ == '__main__':
    unittest.main()
