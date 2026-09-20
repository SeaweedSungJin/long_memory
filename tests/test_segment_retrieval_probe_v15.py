"""Synthetic CPU-only retrieval algebra/scope tests; no weights or datasets loaded."""
from dataclasses import replace
import inspect
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from run_scripts.robomme import segment_retrieval_probe_v15 as probe
from run_scripts.robomme import visual_differential_memory_v12 as native
from run_scripts.robomme.visual_demo_tail_bank_v13 import VisualDemoTailMemoryV13
from run_scripts.robomme.visual_patch_memory_v11 import CAMERA_ORDER


def memory(mode="differential"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1511)
        return VisualDemoTailMemoryV13(
            native.VisualDifferentialConfig(feature_dim=8, hidden_dim=16, num_heads=4), read_mode=mode)


def observation(model, frame=48, *, batch=2, seed=91, demo=False, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(batch, 179, 8, generator=generator).to(dtype)
    image = torch.zeros(batch, 179, dtype=torch.bool)
    image[:, 3:84] = True
    image[:, 90:171] = True
    attention = torch.ones_like(image)
    return model.encode_observation(features, image, attention, [frame] * batch,
                                    [demo] * batch, camera_order=CAMERA_ORDER)


def problem(model, *, batch=2, dtype=torch.float32):
    bank = None
    for index, frame in enumerate((0, 16, 32)):
        old = observation(model, frame, batch=batch, seed=21 + index, demo=index < 2, dtype=dtype)
        valid = torch.ones(batch, dtype=torch.bool)
        if batch == 2 and index == 1:
            valid[1] = False
        bank = model.append(bank, old, valid=valid)
    return observation(model, batch=batch, dtype=dtype), bank


def manual_attention(model, current, bank):
    b, h, d = current.features.shape[0], model.config.num_heads, model.config.hidden_dim
    def heads(x):
        return x.reshape(b, -1, h, d // h).transpose(1, 2)
    mask = bank.valid.repeat_interleave(162, dim=1)[:, None, None, :]
    addresses = torch.where(mask[:, 0, 0, :, None], bank.tokens.flatten(1, 2), 0.)
    q, k = heads(current.queries), heads(model.key_projection(addresses))
    weights = (q @ k.transpose(-2, -1) / math.sqrt(d // h)).masked_fill(~mask, -torch.inf).softmax(-1)
    return weights, q, k, mask


class SegmentRetrievalProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_manual_probabilities_and_native_sdpa_frame_mass_and_values(self):
        model = memory()
        current, bank = problem(model)
        actual = probe.past_frame_probabilities(model, current, bank)
        weights, q, k, mask = manual_attention(model, current, bank)
        expected = weights.reshape(2, 4, 162, 3, 162).sum(-1).mean((1, 2))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.shape, (2, 3))
        torch.testing.assert_close(actual.sum(1), torch.ones(2), rtol=0, atol=2e-7)
        self.assertEqual(actual[1, 1].item(), 0.)
        # Diagnostic one-hot values make SDPA output exactly each frame's mass.
        one_hot = torch.eye(3).repeat_interleave(162, dim=0)[None, None].expand(2, 4, -1, -1)
        sdpa_frames = F.scaled_dot_product_attention(q, k, one_hot, attn_mask=mask,
                                                    dropout_p=0., is_causal=False).mean((1, 2))
        torch.testing.assert_close(actual, sdpa_frames, rtol=1e-6, atol=1e-7)
        values = torch.randn(2, 4, 486, 4, generator=torch.Generator().manual_seed(12))
        sdpa = F.scaled_dot_product_attention(q, k, values, attn_mask=mask, dropout_p=0., is_causal=False)
        torch.testing.assert_close(weights @ values, sdpa, rtol=2e-5, atol=2e-7)

    def test_actual_native_past_read_has_same_q_k_mask_and_head_scale(self):
        model = memory()
        current, bank = problem(model)
        weights, q, k, mask = manual_attention(model, current, bank)
        calls, sdpa = [], F.scaled_dot_product_attention
        def capture(query, key, value, **kwargs):
            result = sdpa(query, key, value, **kwargs)
            calls.append((query, key, value, kwargs, result))
            return result
        with patch.object(native.F, "scaled_dot_product_attention", side_effect=capture):
            model.read(current, bank)
        self.assertEqual(len(calls), 2)  # Native current reference, then past READ.
        nq, nk, nv, options, result = calls[1]
        self.assertTrue(torch.equal(nq, q))
        self.assertTrue(torch.equal(nk, k))
        self.assertTrue(torch.equal(options["attn_mask"], mask))
        self.assertEqual(options["dropout_p"], 0.)
        self.assertFalse(options["is_causal"])
        self.assertNotIn("scale", options)  # Native default 1/sqrt(head_dim).
        torch.testing.assert_close(weights @ nv, result, rtol=2e-5, atol=3e-7)

    def test_no_labels_no_value_no_reference_no_read_no_output_and_no_mutation(self):
        model = memory()
        current, bank = problem(model)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        bank_before = {name: getattr(bank, name).clone() for name in ("tokens", "content", "frames", "is_demo", "valid")}
        features = current.features.clone()
        rng = torch.get_rng_state().clone()
        self.assertEqual(list(inspect.signature(probe.past_frame_probabilities).parameters),
                         ["memory", "observation", "bank"])
        with patch.object(model, "read", side_effect=AssertionError("No visual READ")), \
                patch.object(model, "encode_observation", side_effect=AssertionError("No re-encoding")), \
                patch.object(model, "append", side_effect=AssertionError("No WRITE")), \
                patch.object(model.value_projection, "forward", side_effect=AssertionError("No V")), \
                patch.object(model.output_projection, "forward", side_effect=AssertionError("No output")), \
                patch.object(native.F, "scaled_dot_product_attention", side_effect=AssertionError("No reference read")):
            probabilities = probe.past_frame_probabilities(model, current, bank)
            changed_values = probe.past_frame_probabilities(model, replace(current, content=current.content + 17.),
                                                            replace(bank, content=bank.content - 81.))
        self.assertTrue(torch.equal(probabilities, changed_values))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(features, current.features))
        for name, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        for name, value in bank_before.items():
            self.assertTrue(torch.equal(value, getattr(bank, name)), name)
        self.assertFalse(torch.cuda.is_initialized())

    def test_padding_nan_is_neutralized_before_key_projection(self):
        model = memory()
        current, bank = problem(model)
        expected = probe.past_frame_probabilities(model, current, bank)
        addresses, contents = bank.tokens.clone(), bank.content.clone()
        addresses[~bank.valid] = float("nan")
        contents[~bank.valid] = float("nan")
        actual = probe.past_frame_probabilities(model, current, replace(bank, tokens=addresses, content=contents))
        self.assertTrue(torch.equal(expected, actual))
        addresses[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            probe.past_frame_probabilities(model, current, replace(bank, tokens=addresses))

    def test_empty_row_current_future_duplicate_and_reordered_frames_rejected(self):
        model = memory()
        current, bank = problem(model)
        with self.assertRaisesRegex(ValueError, "past frame"):
            probe.past_frame_probabilities(model, current, None)
        valid = bank.valid.clone(); valid[1] = False
        with self.assertRaisesRegex(ValueError, "past frame"):
            probe.past_frame_probabilities(model, current, replace(bank, valid=valid))
        for frames in ((0, 16, 48), (0, 16, 49), (0, 16, 16), (16, 0, 32), (-1, 16, 32)):
            with self.subTest(frames=frames), self.assertRaisesRegex(ValueError, "chronological"):
                probe.past_frame_probabilities(model, current, replace(bank, frames=torch.tensor([frames, frames])))
        self.assertTrue(torch.equal(bank.frames, torch.tensor([[0, 16, 32], [0, 16, 32]])))

    def test_malformed_current_or_nonpast_architecture_rejected(self):
        model = memory(); current, bank = problem(model)
        bad = (replace(current, queries=current.queries.bfloat16()),
               replace(current, queries=current.queries[:, :-1]),
               replace(current, queries=current.queries * float("nan")),
               replace(current, frames=current.frames.float()),
               replace(current, is_demo=current.is_demo.long()))
        for item in bad:
            with self.subTest(item=item.frames.dtype), self.assertRaises(ValueError):
                probe.past_frame_probabilities(model, item, bank)
        with self.assertRaises(TypeError):
            probe.past_frame_probabilities(torch.nn.Linear(8, 8), current, bank)
        with self.assertRaisesRegex(ValueError, "differential"):
            probe.clone_qk_probe(memory("current_only"))

    def test_fp32_math_under_cpu_autocast_and_bf16_original_features(self):
        model = memory(); current, bank = problem(model, dtype=torch.bfloat16)
        expected = probe.past_frame_probabilities(model, current, bank)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = probe.past_frame_probabilities(model, current, bank)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

    def test_clone_only_qk_gradients_and_updates_original_and_other_tensors_unchanged(self):
        original = memory()
        original.query_projection.weight.grad = torch.ones_like(original.query_projection.weight)
        original_state = {name: value.clone() for name, value in original.state_dict().items()}
        flags = {name: p.requires_grad for name, p in original.named_parameters()}
        rng = torch.get_rng_state().clone()
        model = probe.clone_qk_probe(original)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIsNone(model.query_projection.weight.grad)
        self.assertIsNotNone(original.query_projection.weight.grad)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertEqual(trainable, set(probe.QK_PARAMETER_NAMES))
        self.assertEqual(len(list(model.parameters())), 14)
        frozen = probe.snapshot_frozen_parameters(model)
        self.assertEqual(len(frozen), 12)
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.AdamW(probe.configure_qk_only(model), lr=1e-3, weight_decay=0.)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            current, bank = problem(model)
            positives = torch.tensor([[True, False, False], [False, False, True]])
            probabilities = probe.past_frame_probabilities(model, current, bank)
            probe.segment_mass_loss(probabilities, positives, bank.valid).backward()
            self.assertTrue(probe.assert_qk_scope(model, frozen, require_gradients=True))
            for name in probe.QK_PARAMETER_NAMES:
                self.assertGreater(float(dict(model.named_parameters())[name].grad.norm()), 0.)
            optimizer.step()
            self.assertTrue(probe.assert_qk_scope(model, frozen, require_gradients=True))
        for name, p in model.named_parameters():
            if name in probe.QK_PARAMETER_NAMES:
                self.assertFalse(torch.equal(initial[name], p), name)
            else:
                self.assertTrue(torch.equal(initial[name], p), name)
                self.assertIsNone(p.grad, name)
            self.assertNotEqual(p.data_ptr(), dict(original.named_parameters())[name].data_ptr())
        for name, value in original_state.items():
            self.assertTrue(torch.equal(value, original.state_dict()[name]), name)
        self.assertEqual(flags, {name: p.requires_grad for name, p in original.named_parameters()})

    def test_scope_rejects_stale_queries_changed_frozen_weights_flags_and_gradients(self):
        model = probe.clone_qk_probe(memory()); frozen = probe.snapshot_frozen_parameters(model)
        with torch.no_grad():
            current, bank = problem(model)
        probabilities = probe.past_frame_probabilities(model, current, bank)
        positives = torch.tensor([[True, False, False], [True, False, False]])
        probe.segment_mass_loss(probabilities, positives, bank.valid).backward()
        with self.assertRaisesRegex(ValueError, "Disconnected Q/K"):
            probe.assert_qk_scope(model, frozen, require_gradients=True)
        probe.configure_qk_only(model)
        model.value_projection.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "scope"):
            probe.assert_qk_scope(model, frozen)
        probe.configure_qk_only(model)
        model.value_projection.weight.grad = torch.zeros_like(model.value_projection.weight)
        with self.assertRaisesRegex(ValueError, "acquired a gradient"):
            probe.assert_qk_scope(model, frozen)
        probe.configure_qk_only(model)
        with torch.no_grad():
            model.value_projection.weight[0, 0] += 1.
        with self.assertRaisesRegex(ValueError, "changed"):
            probe.assert_qk_scope(model, frozen)
        with self.assertRaisesRegex(ValueError, "exactly"):
            probe.assert_qk_scope(model, {})

    def test_loss_metrics_uniform_prior_and_manual_gradient(self):
        probabilities = torch.tensor([[.2, .3, 0., .5], [.1, 0., .6, .3]], requires_grad=True)
        valid = torch.tensor([[True, True, False, True], [True, False, True, True]])
        positive = torch.tensor([[True, True, False, False], [False, False, True, False]])
        loss = probe.segment_mass_loss(probabilities, positive, valid)
        expected = -torch.tensor([.5, .6]).log()
        torch.testing.assert_close(loss, expected.mean())
        torch.testing.assert_close(probe.segment_mass_loss(probabilities, positive, valid, reduction="none"), expected)
        torch.testing.assert_close(probe.segment_mass_loss(probabilities, positive, valid, reduction="sum"), expected.sum())
        loss.backward()
        torch.testing.assert_close(probabilities.grad, torch.tensor([[-1., -1., 0., 0.], [0., 0., -1 / 1.2, 0.]]))
        metrics = probe.retrieval_metrics(probabilities, positive, valid)
        torch.testing.assert_close(metrics["positive_mass"], torch.tensor([.5, .6]))
        torch.testing.assert_close(metrics["uniform_positive_mass"], torch.tensor([2 / 3, 1 / 3]))
        torch.testing.assert_close(metrics["log_score_gain"], torch.tensor([.75, 1.8]).log())
        self.assertTrue(torch.equal(metrics["top1_span_hit"], torch.tensor([0., 1.])))
        self.assertTrue(torch.equal(metrics["valid_frame_count"], torch.tensor([3, 3])))
        self.assertTrue(torch.equal(metrics["positive_frame_count"], torch.tensor([2, 1])))
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))
        uniform = valid.float() / valid.sum(1, keepdim=True)
        torch.testing.assert_close(probe.retrieval_metrics(uniform, positive, valid)["log_score_gain"], torch.zeros(2))

    def test_supervision_rejects_empty_no_all_or_invalid_positives_and_bad_probabilities(self):
        probabilities = torch.tensor([[.5, 0., .5]])
        valid = torch.tensor([[True, False, True]])
        positive = torch.tensor([[True, False, False]])
        for target in (torch.zeros_like(valid), valid.clone(), torch.tensor([[False, True, False]])):
            with self.assertRaises(ValueError):
                probe.segment_mass_loss(probabilities, target, valid)
        for bad in (torch.tensor([[.4, .1, .5]]), probabilities * .5, probabilities.double(),
                    torch.tensor([[float("nan"), 0., .5]]), torch.tensor([[-.1, 0., 1.1]])):
            with self.assertRaises(ValueError):
                probe.segment_mass_loss(bad, positive, valid)
        with self.assertRaisesRegex(ValueError, "Empty"):
            probe.segment_mass_loss(torch.zeros_like(probabilities), torch.zeros_like(valid), torch.zeros_like(valid))
        with self.assertRaises(ValueError):
            probe.segment_mass_loss(probabilities, positive.float(), valid)
        with self.assertRaises(ValueError):
            probe.segment_mass_loss(probabilities, positive, valid, reduction="median")
        with self.assertRaisesRegex(FloatingPointError, "underflowed"):
            probe.segment_mass_loss(torch.tensor([[0., 0., 1.]]), positive, valid)

    def test_content_following_targets_exact_permutation_without_changing_metadata(self):
        valid = torch.tensor([[True, False, True, True], [True, True, True, True]])
        positive = torch.tensor([[True, False, False, False], [False, True, True, False]])
        source = torch.tensor([[2, 1, 3, 0], [3, 2, 1, 0]])
        before = [x.clone() for x in (positive, source, valid)]
        moved = probe.content_following_positive_mask(positive, source, valid)
        self.assertTrue(torch.equal(moved, torch.tensor([[False, False, False, True], [False, True, True, False]])))
        self.assertTrue(torch.equal(moved.sum(1), positive.sum(1)))
        for x, old in zip((positive, source, valid), before):
            self.assertTrue(torch.equal(x, old))
        for bad in (torch.tensor([[2, 0, 3, 1], [3, 2, 1, 0]]),
                    torch.tensor([[0, 1, 0, 3], [3, 2, 1, 0]]),
                    torch.tensor([[-1, 1, 2, 3], [3, 2, 1, 0]]), source.float()):
            with self.assertRaises(ValueError):
                probe.content_following_positive_mask(positive, bad, valid)


if __name__ == "__main__":
    unittest.main()
