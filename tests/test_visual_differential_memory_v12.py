"""CPU V12 contracts, including the frozen V11 extraction/causality regressions."""
from dataclasses import asdict, fields, replace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from run_scripts.robomme import visual_differential_memory_v12 as core
from run_scripts.robomme.visual_patch_memory_v11 import VisualPatchConfig, VisualPatchMemoryV11
from tests import test_visual_patch_memory_v11 as legacy


class VisualDifferentialMemoryTests(legacy.VisualPatchMemoryTests):
    """Inherited V11 contracts run AGAIN against the new differential module."""

    def memory(self, mode="differential"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(811)
            return core.VisualDifferentialMemoryV12(core.VisualDifferentialConfig(feature_dim=8, hidden_dim=16),
                                                     read_mode=mode)

    def test_exact_fullwidth_parameter_names_shapes_values_count_and_rng_vs_v11(self):
        models, states = [], []
        for factory in (VisualPatchMemoryV11, core.VisualDifferentialMemoryV12,
                        lambda: core.VisualDifferentialMemoryV12(read_mode="current_only")):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(9111)
                models.append(factory())
                states.append(torch.get_rng_state().clone())
        self.assertEqual(sum(p.numel() for p in models[0].parameters()), 1253632)
        expected = models[0].state_dict()
        for model, rng in zip(models[1:], states[1:]):
            self.assertEqual(sum(p.numel() for p in model.parameters()), 1253632)
            self.assertEqual(list(model.state_dict()), list(expected))
            self.assertTrue(torch.equal(rng, states[0]))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]), name)
        self.assertEqual(asdict(core.VisualDifferentialConfig()), asdict(VisualPatchConfig()))
        self.assertEqual(len(fields(core.VisualDifferentialConfig)), 5)
        for mode in (None, True, "past_only", ""):
            with self.assertRaises(ValueError):
                self.memory(mode)
        with self.assertRaises(TypeError):
            core.VisualDifferentialMemoryV12(VisualPatchConfig())
        self.assertFalse(torch.cuda.is_initialized())

    def test_content_excludes_explicit_identifiers_but_address_and_query_match_v11(self):
        memory = self.memory()
        old = VisualPatchMemoryV11(VisualPatchConfig(feature_dim=8, hidden_dim=16))
        old.load_state_dict(memory.state_dict())
        features, images, attention = legacy.inputs(batch=1)
        args = (features, images, attention, [0], [False])
        first = memory.encode_observation(*args, camera_order=core.CAMERA_ORDER)
        reference = old.encode_observation(*args, camera_order=core.CAMERA_ORDER)
        self.assertTrue(torch.equal(first.encoded, reference.encoded))
        self.assertTrue(torch.equal(first.queries, reference.queries))
        selected = features[:, first.image_indices[0]]
        self.assertTrue(torch.equal(first.content, memory.bank_norm(memory.image_projection(selected))))
        later = memory.encode_observation(features, images, attention, [96], [True], camera_order=core.CAMERA_ORDER)
        self.assertTrue(torch.equal(first.content, later.content))
        self.assertFalse(torch.equal(first.encoded, later.encoded))
        with torch.no_grad():
            memory.camera_embedding.weight.add_(torch.arange(16))
            memory.row_embedding.weight.mul_(3)
            memory.column_embedding.weight.mul_(4)
            memory.time_projection.weight.mul_(2)
        changed = memory.encode_observation(*args, camera_order=core.CAMERA_ORDER)
        self.assertTrue(torch.equal(first.content, changed.content))
        self.assertFalse(torch.equal(first.encoded, changed.encoded))

    def test_both_modes_empty_off_zero_and_woken_bf16_inplace_mask_contract(self):
        for mode in core.READ_MODES:
            memory = self.memory(mode)
            bank = memory.append(None, legacy.observation(memory, dtype=torch.bfloat16), valid=[True, False])
            args = legacy.inputs(seed=9, dtype=torch.bfloat16)
            current = memory.encode_observation(*args, [16, 16], [False, False], camera_order=core.CAMERA_ORDER)
            self.assertTrue(torch.equal(memory.read(current, bank), current.features))
            self.wake(memory)
            self.assertIs(memory.read(current, None), current.features)
            self.assertIs(memory.read(current, bank, enabled=False), current.features)
            result = memory.read(current, bank)
            self.assertFalse(torch.equal(result[0], current.features[0]))
            self.assertTrue(torch.equal(result[1], current.features[1]))
            self.assertTrue(torch.equal(result[~args[1]], current.features[~args[1]]))
            self.assertTrue(torch.equal(result[:, -4:], current.features[:, -4:]))
            self.assertTrue(torch.isfinite(result).all())

    def test_constant_values_cancel_with_tolerance_not_nonuniform_scene_assumption(self):
        memory = self.memory(); self.wake(memory)
        vector = torch.arange(8).float() / 8
        bank = None
        # Every patch has exactly the same content, while addresses/time/demo vary.
        for index in range(4):
            features, images, attention = legacy.inputs(seed=index, batch=1)
            features[images] = vector
            current = memory.encode_observation(features, images, attention, [index * 16], [index < 2],
                                                  camera_order=core.CAMERA_ORDER)
            if index < 3:
                bank = memory.append(bank, current)
        self.assertTrue(torch.equal(bank.content, current.content[:, None].expand_as(bank.content)))
        captured = []
        with memory.output_projection.register_forward_pre_hook(lambda module, args: captured.append(args[0])):
            result = memory.read(current, bank)
        # FP32 SDPA reductions over 162 versus 486 keys need not sum bitwise
        # identically, even with exactly constant V (observed error ~2.6e-6).
        self.assertLess(float(captured[0].abs().max()), 5e-6)
        torch.testing.assert_close(result, current.features, rtol=0, atol=2e-6)

    def test_exact_differential_formula_and_current_only_one_read(self):
        memory = self.memory(); self.wake(memory)
        bank = memory.append(None, legacy.observation(memory))
        current = legacy.observation(memory, seed=2, frames=(16, 16))
        outputs = []
        sdpa = F.scaled_dot_product_attention
        def track(*args, **kwargs):
            value = sdpa(*args, **kwargs)
            outputs.append(value)
            self.assertEqual(kwargs["dropout_p"], 0.)
            self.assertFalse(kwargs["is_causal"])
            return value
        captured = []
        with patch.object(core.F, "scaled_dot_product_attention", side_effect=track), \
                memory.output_projection.register_forward_pre_hook(lambda module, args: captured.append(args[0])):
            memory.read(current, bank)
        self.assertEqual(len(outputs), 2)
        expected = (outputs[1] - outputs[0]).transpose(1, 2).reshape(2, 162, 16)
        self.assertTrue(torch.equal(captured[0], expected))
        memory.read_mode = "current_only"
        outputs.clear(); captured.clear()
        with patch.object(core.F, "scaled_dot_product_attention", side_effect=track), \
                memory.output_projection.register_forward_pre_hook(lambda module, args: captured.append(args[0])):
            memory.read(current, bank)
        self.assertEqual(len(outputs), 1)
        self.assertTrue(torch.equal(captured[0], outputs[0].transpose(1, 2).reshape(2, 162, 16)))

    def test_differential_past_and_current_reference_gradients_current_only_no_past_gradient(self):
        for mode in core.READ_MODES:
            memory = self.memory(mode); self.wake(memory)
            past = legacy.observation(memory, frames=(0,), demo=(True,))
            current = legacy.observation(memory, seed=5, frames=(16,), demo=(False,))
            for item in (past, current):
                item.encoded.retain_grad(); item.content.retain_grad(); item.queries.retain_grad()
            bank = memory.append(None, past)
            memory.read(current, bank).square().mean().backward()
            for name in ("encoded", "content", "queries"):
                gradient = getattr(current, name).grad
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.norm()), 0, mode + "/current/" + name)
                self.assertTrue(torch.isfinite(gradient).all())
            for name in ("encoded", "content"):
                gradient = getattr(past, name).grad
                if mode == "current_only":
                    self.assertIsNone(gradient)
                else:
                    self.assertIsNotNone(gradient)
                    self.assertGreater(float(gradient.norm()), 0, name)
            for name in ("image_projection", "query_projection", "key_projection", "value_projection"):
                self.assertGreater(float(getattr(memory, name).weight.grad.norm()), 0, mode + "/" + name)

    def test_independent_content_leaves_prove_current_reference_and_past_value_gradients(self):
        # Neither assertion can be satisfied by the direct features->features
        # identity gradient: both tested leaves replace only value-source C.
        for mode in core.READ_MODES:
            memory = self.memory(mode); self.wake(memory)
            self.assertTrue(bool(memory.output_projection.weight.any()))
            bank = memory.append(None, legacy.observation(memory, frames=(0,), demo=(True,)))
            current = legacy.observation(memory, seed=7, frames=(16,), demo=(False,))
            past_content = bank.content.detach().clone().requires_grad_(True)
            current_content = current.content.detach().clone().requires_grad_(True)
            bank = replace(bank, content=past_content)
            current = replace(current, content=current_content)
            memory.read(current, bank).square().mean().backward()
            self.assertIsNotNone(current_content.grad)
            self.assertTrue(torch.isfinite(current_content.grad).all())
            self.assertGreater(float(current_content.grad.norm()), 0, mode)
            if mode == "differential":
                self.assertIsNotNone(past_content.grad)
                self.assertTrue(torch.isfinite(past_content.grad).all())
                self.assertGreater(float(past_content.grad.norm()), 0)
            else:
                self.assertIsNone(past_content.grad)

    def test_dual_stream_append_original_and_invalid_padding_suppressed(self):
        memory = self.memory(); self.wake(memory)
        old = legacy.observation(memory)
        bank = memory.append(None, old, valid=[True, False])
        before = bank.content.clone()
        args = legacy.inputs(seed=8)
        current = memory.encode_observation(*args, [16, 16], [False, False], camera_order=core.CAMERA_ORDER)
        result, next_bank = memory(*args, [16, 16], [False, False], bank, camera_order=core.CAMERA_ORDER)
        self.assertTrue(torch.equal(next_bank.content[:, -1], current.content))
        self.assertTrue(torch.equal(next_bank.tokens[:, -1], current.encoded))
        self.assertTrue(torch.equal(bank.content, before))
        changed = memory.encode_observation(result, args[1], args[2], [16, 16], [False, False], camera_order=core.CAMERA_ORDER)
        self.assertFalse(torch.equal(next_bank.content[:, -1], changed.content))
        addresses, contents = bank.tokens.clone(), bank.content.clone()
        addresses[~bank.valid] = float("nan"); contents[~bank.valid] = float("nan")
        self.assertTrue(torch.equal(result, memory.read(current, replace(bank, tokens=addresses, content=contents))))
        contents[bank.valid] = float("nan")
        with self.assertRaisesRegex(ValueError, "bank content"):
            memory.read(current, replace(bank, content=contents))
        with self.assertRaisesRegex(ValueError, "bank content"):
            memory.read(current, replace(bank, content=bank.content.bfloat16()))


if __name__ == "__main__":
    import unittest
    unittest.main()
