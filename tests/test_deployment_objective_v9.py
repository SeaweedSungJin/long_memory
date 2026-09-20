"""Sampler parity and autograd tests on a tiny CPU Expert, not task accuracy."""
import copy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from gr00t.long_memory.expert_v4 import ExpertLoRALinear, LoRAConfig, expert_parameters
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from gr00t.long_memory.replay_v7 import replay_queries
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead
from run_scripts.robomme import deployment_objective_v9 as objective


class CategoryLinear(nn.Linear):
    def forward(self, x, embodiment):
        return super().forward(x)


class TinyActionEncoder(nn.Linear):
    def forward(self, x, timestep, embodiment):
        return super().forward(x) + timestep[:, None, None].float() / 1000


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = ExpertLoRALinear(nn.Linear(8, 8), LoRAConfig(rank=2, alpha=4))
        self.context = nn.Linear(8, 8)
        with torch.no_grad():
            self.attention.lora_B.fill_(.03)

    def forward(self, hidden_states, encoder_hidden_states, timestep, **kwargs):
        return self.attention(hidden_states) + self.context(encoder_hidden_states.mean(1))[:, None]


class TinyExpert(nn.Module):
    get_action_with_features = Gr00tN1d6ActionHead.get_action_with_features

    def __init__(self, alternate=True, position=True):
        super().__init__()
        self.config = SimpleNamespace(action_horizon=5, add_pos_embed=position, use_alternate_vl_dit=alternate)
        self.action_horizon, self.action_dim = 5, 3
        self.num_inference_timesteps, self.num_timestep_buckets = 4, 1000
        self.state_encoder = CategoryLinear(4, 8)
        self.action_encoder = TinyActionEncoder(3, 8)
        self.position_embedding = nn.Embedding(5, 8)
        self.model = TinyDiT()
        self.action_decoder = CategoryLinear(8, 3)
        self.requires_grad_(False).eval()
        for parameter in expert_parameters(self):
            parameter.requires_grad_(True)


class DeploymentObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(93)
        self.features = torch.randn(1, 7, 8)
        self.state = torch.randn(1, 1, 4)
        self.embodiment = torch.tensor([0])
        self.backbone = BatchFeature({"image_mask": torch.tensor([[True]*3+[False]*4]),
                                     "backbone_attention_mask": torch.ones(1, 7, dtype=torch.bool)})

    def test_exact_original_sampler_parity_with_and_without_recomputation(self):
        for alternate in (True, False):
            for position in (True, False):
                head = TinyExpert(alternate, position)
                head._inference_gen = torch.Generator().manual_seed(73)
                noise = torch.randn(1, 5, 3, generator=torch.Generator().manual_seed(73))
                state_features = head.state_encoder(self.state, self.embodiment)
                with patch.dict(os.environ, GR00T_INFERENCE_SEED="73"):
                    reference = head.get_action_with_features(self.features, state_features, self.embodiment, self.backbone)["action_pred"]
                rng = head._inference_gen.get_state().clone()
                for recompute in (False, True):
                    prediction = objective.differentiable_euler(head, self.features, state_features,
                        self.embodiment, self.backbone, noise, activation_checkpointing=recompute)
                    self.assertTrue(torch.equal(reference, prediction), (alternate, position, recompute))
                    self.assertTrue(torch.equal(head._inference_gen.get_state(), rng))

    def test_recomputed_gradients_match_plain_and_leave_base_frozen(self):
        head = TinyExpert()
        noise = torch.randn(1, 5, 3)
        results = []
        for recompute in (False, True):
            actor = copy.deepcopy(head)
            features = self.features.clone().requires_grad_()
            state_features = actor.state_encoder(self.state, self.embodiment)
            output = objective.differentiable_euler(actor, features, state_features,
                self.embodiment, self.backbone, noise, activation_checkpointing=recompute)
            output.square().mean().backward()
            allowed = {id(p) for p in expert_parameters(actor)}
            self.assertTrue(all(p.grad is None for p in actor.parameters() if id(p) not in allowed))
            grads = [features.grad] + [p.grad for p in expert_parameters(actor)]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in grads))
            results.append(grads)
        for left, right in zip(*results):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_gradients_match_original_unwrapped_sampler(self):
        # Compare with an independent implementation, not just two executions
        # of our solver: both executions could otherwise share a detach bug.
        native = Gr00tN1d6ActionHead.get_action_with_features.__wrapped__
        for alternate in (True, False):
            for position in (True, False):
                template = TinyExpert(alternate, position)
                reference = None
                for implementation in ("native", "plain", "checkpoint"):
                    with self.subTest(alternate=alternate, position=position,
                                      implementation=implementation):
                        head = copy.deepcopy(template)
                        features = self.features.clone().requires_grad_()
                        state_features = head.state_encoder(self.state, self.embodiment).detach().requires_grad_()
                        head._inference_gen = torch.Generator().manual_seed(73)
                        noise = torch.randn(1, 5, 3, generator=torch.Generator().manual_seed(73))
                        with patch.dict(os.environ, GR00T_INFERENCE_SEED="73"):
                            if implementation == "native":
                                prediction = native(head, features, state_features,
                                    self.embodiment, self.backbone)["action_pred"]
                            else:
                                prediction = objective.differentiable_euler(head, features,
                                    state_features, self.embodiment, self.backbone, noise,
                                    activation_checkpointing=implementation == "checkpoint")
                        parameters = list(expert_parameters(head))
                        gradients = torch.autograd.grad(prediction.square().mean(),
                            [features, state_features, *parameters])
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        # TinyDiT is token-local, so its state-token gradient
                        # is correctly zero; feature and adapter paths are not.
                        self.assertTrue(all(g.norm() > 0 for g in (gradients[0], *gradients[2:])))
                        actual = (prediction.detach(), *(g.detach() for g in gradients))
                        if reference is None:
                            reference = actual
                        else:
                            for expected, observed in zip(reference, actual):
                                torch.testing.assert_close(expected, observed, rtol=0, atol=0)

    def test_exact_native_mask_arguments_and_checkpoint_timesteps(self):
        # Nontrivial, distinct masks make omission or accidental swapping
        # observable even though the tiny DiT does not implement masking.
        backbone = BatchFeature({
            "image_mask": torch.tensor([[True, False, True, False, False, False, False]]),
            "backbone_attention_mask": torch.tensor([[True, True, False, True, False, True, True]]),
        })
        times = [0, 250, 500, 750]
        for alternate in (True, False):
            for implementation in ("native", "plain", "checkpoint"):
                with self.subTest(alternate=alternate, implementation=implementation):
                    head = TinyExpert(alternate=alternate)
                    features = self.features.clone().requires_grad_()
                    state = head.state_encoder(self.state, self.embodiment)
                    noise = torch.randn(1, 5, 3, generator=torch.Generator().manual_seed(73))
                    with patch.object(head.model, "forward", wraps=head.model.forward) as calls:
                        if implementation == "native":
                            head._inference_gen = torch.Generator().manual_seed(73)
                            with patch.dict(os.environ, GR00T_INFERENCE_SEED="73"):
                                head.get_action_with_features(features, state, self.embodiment, backbone)
                        else:
                            prediction = objective.differentiable_euler(head, features, state,
                                self.embodiment, backbone, noise,
                                activation_checkpointing=implementation == "checkpoint")
                            self.assertEqual(calls.call_count, 4)
                            prediction.square().mean().backward()
                        self.assertEqual(calls.call_count, 8 if implementation == "checkpoint" else 4)
                        observed_times = []
                        for call in calls.call_args_list:
                            self.assertEqual(call.args, ())
                            kwargs = call.kwargs
                            expected_keys = {"hidden_states", "encoder_hidden_states", "timestep", "temb_add"}
                            if alternate:
                                expected_keys |= {"image_mask", "backbone_attention_mask"}
                                self.assertIs(kwargs["image_mask"], backbone.image_mask)
                                self.assertIs(kwargs["backbone_attention_mask"], backbone.backbone_attention_mask)
                            self.assertEqual(set(kwargs), expected_keys)
                            self.assertIsNone(kwargs["temb_add"])
                            self.assertEqual(kwargs["timestep"].dtype, torch.int64)
                            self.assertEqual(tuple(kwargs["timestep"].shape), (1,))
                            observed_times.append(int(kwargs["timestep"][0]))
                        self.assertEqual(observed_times[:4], times)
                        if implementation == "checkpoint":
                            self.assertEqual(observed_times[4:], times[::-1])

    def episode(self):
        short = torch.randn(3, 2, 8, requires_grad=True)
        features = [torch.cat((torch.randn(5, 8), short[i].detach()), dim=0) for i in range(3)]
        return {"short": short, "state": torch.randn(3, 4), "frames": torch.tensor([0, 2, 4]),
                "is_demo": torch.tensor([True, False, False]), "features": features,
                "attention_masks": [self.backbone.backbone_attention_mask[0]]*3,
                "image_masks": [self.backbone.image_mask[0]]*3, "embodiment_id": 0,
                "targets": torch.randn(2, 5, 3), "target_mask": torch.ones(2, 5, 3, dtype=torch.bool),
                "action_mask": torch.tensor([[False, False], [True, False]])}

    def test_unrolled_feature_gradient_matches_finite_difference(self):
        head = TinyExpert()
        noise = torch.randn(1, 5, 3)
        features = self.features.clone().requires_grad_()
        state = head.state_encoder(self.state, self.embodiment)
        def loss(value):
            return objective.differentiable_euler(head, value, state, self.embodiment,
                self.backbone, noise, activation_checkpointing=False).square().mean()
        gradient, = torch.autograd.grad(loss(features), features)
        index = int(gradient.abs().reshape(-1).argmax())
        plus, minus = features.detach().clone(), features.detach().clone()
        epsilon = 1e-3
        plus.reshape(-1)[index] += epsilon
        minus.reshape(-1)[index] -= epsilon
        numeric = (loss(plus) - loss(minus)) / (2 * epsilon)
        torch.testing.assert_close(gradient.reshape(-1)[index], numeric, rtol=.01, atol=1e-5)

    def test_invalid_and_unobserved_targets_cannot_change_loss_or_gradient(self):
        head, original = TinyExpert(), self.episode()
        original["target_mask"][1, 0, 2] = False
        altered = {**original, "targets": original["targets"].clone()}
        altered["targets"][1, 1:] = float("nan")
        altered["targets"][1, 0, 2] = float("nan")
        results = []
        for ep in (original, altered):
            result = objective.generated_prefix_objective(head, ep, 1, seed=13, action_steps=2)
            grads = torch.autograd.grad(result["loss"], list(expert_parameters(head)))
            results.append((result["loss"].detach(), grads))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        for left, right in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_prefix_forward_backward_preserves_rng_and_head_state(self):
        for recompute in (False, True):
            for seeded_environment in (False, True):
                for generator_present in (False, True):
                    with self.subTest(recompute=recompute, seeded_environment=seeded_environment,
                                      generator_present=generator_present):
                        head, ep = TinyExpert(), self.episode()
                        if generator_present:
                            head._inference_gen = torch.Generator().manual_seed(4321)
                            inference_generator = head._inference_gen
                            inference_rng = inference_generator.get_state().clone()
                        before = {name: value.detach().clone() for name, value in head.state_dict().items()}
                        global_rng = torch.get_rng_state().clone()
                        with patch.dict(os.environ, GR00T_INFERENCE_SEED="123"):
                            if not seeded_environment:
                                os.environ.pop("GR00T_INFERENCE_SEED")
                            result = objective.generated_prefix_objective(head, ep, 1,
                                seed=19, action_steps=2, activation_checkpointing=recompute)
                            torch.autograd.grad(result["loss"], list(expert_parameters(head)))
                        self.assertTrue(torch.equal(torch.get_rng_state(), global_rng))
                        if generator_present:
                            self.assertIs(head._inference_gen, inference_generator)
                            self.assertTrue(torch.equal(inference_generator.get_state(), inference_rng))
                        else:
                            self.assertFalse(hasattr(head, "_inference_gen"))
                        for name, value in head.state_dict().items():
                            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_past_encoder_read_fusion_and_ae_receive_gradient_without_future_leakage(self):
        memory = RecurrentMemoryV7(MemoryV7Config(feature_dim=8, state_dim=4, num_short_tokens=2,
                                                hidden_dim=8, num_heads=2, capacity=3, time_scale=2))
        with torch.no_grad():
            memory.fusion_projection.weight.normal_(std=.1)
        head, ep = TinyExpert(), self.episode()
        fused, _ = replay_queries(memory, ep, [1], mode="archive")[1]
        result = objective.generated_prefix_objective(head, ep, 1, fused, seed=71, action_steps=2)
        self.assertEqual(result["valid_values"], 3)
        result["loss"].backward()
        self.assertGreater(float(ep["short"].grad[0].norm()), 0)
        self.assertEqual(float(ep["short"].grad[2].norm()), 0)
        parameters = dict(memory.named_parameters())
        for name in ("short_projection.weight", "state_projection.weight", "time_encoder.0.weight",
                     "attention.in_proj_weight", "attention.out_proj.weight", "read_query_norm.weight",
                     "read_key_norm.weight", "read_ffn.0.weight", "fusion_projection.weight",
                     "fusion_gate.weight"):
            with self.subTest(gradient_parameter=name):
                gradient = parameters[name].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.norm()), 0)
        self.assertTrue(all(p.grad is not None and p.grad.norm() > 0 for p in expert_parameters(head)))

    def test_targets_are_only_read_after_generation_and_cannot_change_prediction(self):
        head, source = TinyExpert(), self.episode()
        generated = [False]
        class GuardEpisode(dict):
            def __getitem__(self, key):
                if key in ("targets", "target_mask", "action_mask") and not generated[0]:
                    raise AssertionError("GT accessed before generation completed")
                return super().__getitem__(key)
        original = objective.differentiable_euler
        def traced(*args, **kwargs):
            value = original(*args, **kwargs)
            generated[0] = True
            return value
        predictions = []
        for offset in (0, 100):
            ep = GuardEpisode({**source, "targets": source["targets"] + offset})
            generated[0] = False
            with patch.object(objective, "differentiable_euler", side_effect=traced):
                predictions.append(objective.generated_prefix_objective(head, ep, 1, seed=9, action_steps=2)["prediction"])
        self.assertTrue(torch.equal(*predictions))

    def test_trainability_noise_and_empty_supervision_guards(self):
        head = TinyExpert()
        head.train()
        with self.assertRaises(ValueError):
            objective.validate_expert(head)
        head.eval()
        head.action_decoder.weight.requires_grad_(True)
        with self.assertRaises(ValueError):
            objective.validate_expert(head)
        head.action_decoder.weight.requires_grad_(False)
        with self.assertRaises(ValueError):
            objective.differentiable_euler(head, self.features, torch.randn(1, 1, 8), self.embodiment,
                                           self.backbone, torch.zeros(1, 4, 3))
        ep = self.episode()
        ep["action_mask"].zero_()
        with self.assertRaisesRegex(ValueError, "No observed"):
            objective.generated_prefix_objective(head, ep, 1, seed=3, action_steps=2)


if __name__ == "__main__":
    unittest.main()
