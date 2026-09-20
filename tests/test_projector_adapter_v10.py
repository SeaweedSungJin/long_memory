"""CPU native-arithmetic, gradient and safety tests for the V10 projector.

Toy dimensions exercise the same bridge calls; no training-convergence or
simulator-success claim is made. Real 1536→1024 compatibility is serializer-
validated separately before model loading.
"""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from gr00t.long_memory import expert_v4 as original
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead
from run_scripts.robomme import projector_adapter_v10 as adapter


class CategoryLinear(nn.Linear):
    def forward(self, x, embodiment):
        return super().forward(x)


class ActionLinear(nn.Linear):
    def forward(self, x, timestep, embodiment):
        return super().forward(x)


class ToyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.attn1 = nn.Module()
        for name in ("to_q", "to_k", "to_v"):
            setattr(block.attn1, name, nn.Linear(8, 8))
        block.attn1.to_out = nn.Sequential(nn.Linear(8, 8))
        self.transformer_blocks = nn.ModuleList([block])
        self.proj_out_2 = nn.Linear(8, 8)

    def forward(self, hidden_states, encoder_hidden_states, timestep,
                encoder_attention_mask=None, backbone_attention_mask=None,
                return_all_hidden_states=False, **kwargs):
        mask = backbone_attention_mask if backbone_attention_mask is not None else encoder_attention_mask
        context = encoder_hidden_states
        if mask is not None:
            context = (context * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
        else:
            context = context.mean(1)
        a = self.transformer_blocks[0].attn1
        hidden = a.to_out(a.to_q(hidden_states) + a.to_k(context)[:, None] + a.to_v(context)[:, None])
        result = self.proj_out_2(hidden)
        return (result, []) if return_all_hidden_states else result


class ToyHead(nn.Module):
    get_action_with_features = Gr00tN1d6ActionHead.get_action_with_features

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(action_horizon=5, add_pos_embed=True, use_alternate_vl_dit=True)
        self.action_horizon, self.action_dim = 5, 3
        self.num_inference_timesteps, self.num_timestep_buckets = 4, 1000
        self.state_encoder = CategoryLinear(4, 8)
        self.action_encoder = ActionLinear(3, 8)
        self.position_embedding = nn.Embedding(5, 8)
        self.model = ToyDiT()
        self.action_decoder = CategoryLinear(8, 3)

    def sample_time(self, batch, device, dtype):
        return torch.rand(batch, device=device, dtype=dtype)


def episode(dtype=torch.float32):
    g = torch.Generator().manual_seed(14)
    features = [torch.randn(7, 8, generator=g).to(dtype) for _ in range(3)]
    return {"features": features, "state": torch.randn(3, 4, generator=g).to(dtype),
        "targets": torch.randn(2, 5, 3, generator=g).to(dtype),
        "target_mask": torch.ones(2, 5, 3, dtype=torch.bool),
        "action_mask": torch.tensor([[False, False], [True, False]]),
        "attention_masks": [torch.tensor([1, 1, 1, 1, 1, 1, 0], dtype=torch.bool)] * 3,
        "image_masks": [torch.ones(7, dtype=torch.bool)] * 3, "embodiment_id": 0}


class ProjectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def head(self, enabled=True, dtype=torch.float32):
        torch.manual_seed(42)
        head = ToyHead().to(dtype).eval().requires_grad_(False)
        targets = original.install_expert_lora(head, original.LoRAConfig(2, 4))
        adapter.install_projector(head, enabled=enabled)
        adapter.set_trainable(head, enabled, True)
        return head, targets

    def test_install_zero_and_disabled_exact_native_output_rng_and_targets(self):
        for dtype in (torch.float32, torch.bfloat16):
            for enabled in (True, False):
                torch.manual_seed(71)
                head = ToyHead().to(dtype).eval().requires_grad_(False)
                targets = original.install_expert_lora(head, original.LoRAConfig(2, 4))
                original_lora = original.expert_state_dict(head)
                native = head.model.proj_out_2
                x = torch.randn(2, 5, 8).to(dtype)
                expected = native(x)
                rng = torch.get_rng_state().clone()
                spec = adapter.install_projector(head, enabled)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(torch.equal(expected, head.model.proj_out_2(x)))
                self.assertEqual(spec, {"target": "model.proj_out_2", "in_features": 8,
                    "out_features": 8, "bias": True, "kind": "full_rank_residual_fp32", "enabled": enabled})
                self.assertIs(native, head.model.proj_out_2.base)
                self.assertEqual(set(adapter.projector_state_dict(head)), {"delta_weight", "delta_bias"})
                self.assertTrue(all(p.dtype == torch.float32 and not bool(p.any()) for p in adapter.projector_parameters(head)))
                self.assertEqual(sorted({name.rsplit(".", 1)[0] for name in original.expert_state_dict(head)}), targets)
                self.assertTrue(all(torch.equal(value, original.expert_state_dict(head)[name]) for name, value in original_lora.items()))
                if not enabled:
                    with patch.object(adapter.F, "linear", wraps=adapter.F.linear) as operation:
                        self.assertTrue(torch.equal(expected, head.model.proj_out_2(x)))
                        self.assertEqual(operation.call_count, 1)  # Native Linear only.

    def test_zero_flow_prediction_existing_gradients_and_rng_match_original(self):
        for dtype in (torch.float32, torch.bfloat16):
            for recompute in (False, True):
                torch.manual_seed(5)
                native = ToyHead().to(dtype).eval().requires_grad_(False)
                original.install_expert_lora(native, original.LoRAConfig(2, 4))
                changed = copy.deepcopy(native)
                adapter.install_projector(changed)
                adapter.set_trainable(changed, True, True)
                ep = episode(dtype)
                fused_a = ep["features"][1][-2:].float()[None].clone().requires_grad_()
                fused_b = fused_a.detach().clone().requires_grad_()
                rng = torch.get_rng_state().clone()
                a = original.expert_episode_flow_loss(native, ep, 1, fused_a, seed=321,
                                                     activation_checkpointing=recompute)
                b = adapter.expert_episode_flow_loss(changed, ep, 1, fused_b, seed=321,
                                                    activation_checkpointing=recompute)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for key in ("loss", "prediction", "velocity_mae"):
                    self.assertTrue(torch.equal(a[key], b[key]), key)
                a["loss"].backward(); b["loss"].backward()
                self.assertTrue(torch.equal(fused_a.grad, fused_b.grad))
                for left, right in zip(original.expert_parameters(native), original.expert_parameters(changed)):
                    self.assertTrue(torch.equal(left.grad, right.grad))
                self.assertTrue(any(p.grad is not None and bool(p.grad.any()) for p in adapter.projector_parameters(changed)))

    def test_projector_lora_features_train_without_changing_frozen_base(self):
        head, _ = self.head()
        allowed = {id(p) for p in adapter.trainable_expert_parameters(head)}
        originals = {name: p.detach().clone() for name, p in head.named_parameters() if id(p) not in allowed}
        ep = episode()
        features = ep["features"][1][-2:][None].clone().requires_grad_()
        optimizer = torch.optim.AdamW(list(adapter.trainable_expert_parameters(head)), lr=.01)
        loss = adapter.expert_episode_flow_loss(head, ep, 1, features, seed=44)["loss"]
        loss.backward()
        self.assertTrue(bool(features.grad.any()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.any()) for p in original.expert_parameters(head)))
        self.assertTrue(all(p.grad is not None and bool(p.grad.any()) for p in adapter.projector_parameters(head)))
        optimizer.step()
        self.assertTrue(all(torch.equal(dict(head.named_parameters())[name], value) for name, value in originals.items()))
        self.assertTrue(any(bool(p.any()) for p in adapter.projector_parameters(head)))

    def test_nonzero_projector_checkpointed_flow_outputs_and_gradients_match(self):
        a, _ = self.head()
        with torch.no_grad():
            for p in adapter.projector_parameters(a):
                p.normal_(std=.01)
        b = copy.deepcopy(a)
        ep = episode()
        fused_a = ep["features"][1][-2:][None].clone().requires_grad_()
        fused_b = fused_a.detach().clone().requires_grad_()
        left = adapter.expert_episode_flow_loss(a, ep, 1, fused_a, seed=22, activation_checkpointing=False)
        right = adapter.expert_episode_flow_loss(b, ep, 1, fused_b, seed=22, activation_checkpointing=True)
        self.assertTrue(torch.equal(left["prediction"], right["prediction"]))
        left["loss"].backward(); right["loss"].backward()
        self.assertTrue(torch.equal(fused_a.grad, fused_b.grad))
        for p, q in zip(adapter.trainable_expert_parameters(a), adapter.trainable_expert_parameters(b)):
            self.assertTrue(torch.equal(p.grad, q.grad))

    def test_masked_target_padding_and_future_observations_do_not_change_flow(self):
        head, _ = self.head()
        ep = episode()
        ep["target_mask"][1, -1] = False
        before = adapter.expert_episode_flow_loss(head, ep, 1, seed=43)
        changed = copy.deepcopy(ep)
        changed["targets"][1, -1] = float("nan")
        changed["features"][2].fill_(float("nan"))
        changed["state"][2].fill_(float("nan"))
        after = adapter.expert_episode_flow_loss(head, changed, 1, seed=43)
        self.assertTrue(torch.equal(before["loss"], after["loss"]))
        self.assertTrue(torch.equal(before["prediction"], after["prediction"]))

    def test_all_adapter_baseline_nested_exception_restores_enabled_flags(self):
        head, _ = self.head()
        native = copy.deepcopy(head)
        for name, module in list(native.named_modules()):
            if isinstance(module, original.ExpertLoRALinear):
                parent, child = name.rsplit(".", 1)
                setattr(native.get_submodule(parent), child, module.base)
        native.model.proj_out_2 = native.model.proj_out_2.base
        native.eval().requires_grad_(False)
        ep = episode()
        with torch.no_grad():
            for p in (*original.expert_parameters(head), *adapter.projector_parameters(head)):
                p.add_(.01)
        modules = [m for m in head.modules() if isinstance(m, original.ExpertLoRALinear)]
        modules[0].enabled = False
        flags = [m.enabled for m in modules]
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with adapter.all_adapters_disabled(head):
                self.assertFalse(head.model.proj_out_2.enabled)
                self.assertFalse(any(m.enabled for m in modules))
                actual = adapter.expert_episode_flow_loss(head, ep, 1, seed=1)
                expected = original.expert_episode_flow_loss(native, ep, 1, seed=1)
                self.assertTrue(torch.equal(actual["loss"], expected["loss"]))
                self.assertTrue(torch.equal(actual["prediction"], expected["prediction"]))
                with adapter.all_adapters_disabled(head):
                    self.assertFalse(head.model.proj_out_2.enabled)
                self.assertFalse(head.model.proj_out_2.enabled)
                raise RuntimeError("sentinel")
        self.assertTrue(head.model.proj_out_2.enabled)
        self.assertEqual([m.enabled for m in modules], flags)

    def test_strict_atomic_load_finite_dtype_shape_keys_and_control_zeros(self):
        head, _ = self.head()
        initial = adapter.projector_state_dict(head)
        good = {name: value + .2 for name, value in initial.items()}
        for kind in ("key", "shape", "dtype", "nan"):
            state = copy.deepcopy(good)
            if kind == "key":
                state["other"] = torch.zeros(1)
            elif kind == "shape":
                state["delta_bias"] = state["delta_bias"][:-1]
            elif kind == "dtype":
                state["delta_bias"] = state["delta_bias"].bfloat16()
            else:
                state["delta_bias"][0] = float("nan")
            with self.assertRaises(ValueError):
                adapter.load_projector_state_dict(head, state)
            self.assertTrue(all(torch.equal(value, adapter.projector_state_dict(head)[name]) for name, value in initial.items()))
        adapter.load_projector_state_dict(head, good)
        self.assertTrue(all(torch.equal(value, adapter.projector_state_dict(head)[name]) for name, value in good.items()))
        disabled, _ = self.head(False)
        with self.assertRaisesRegex(ValueError, "remain zero"):
            adapter.load_projector_state_dict(disabled, good)
        adapter.load_projector_state_dict(disabled, initial)

    def test_trainability_flags_policy_and_outside_scope_guard(self):
        head, _ = self.head()
        for projector, lora in ((False, False), (True, False), (False, True), (True, True)):
            adapter.set_trainable(head, projector, lora)
            self.assertTrue(head.model.proj_out_2.enabled)
            self.assertEqual([p.requires_grad for p in adapter.projector_parameters(head)], [projector] * 2)
            self.assertTrue(all(p.requires_grad == lora for p in original.expert_parameters(head)))
        head.action_decoder.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "Only attention"):
            adapter.assert_expert_scope(head)
        adapter.set_trainable(head, False, False)
        head.model.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            adapter.assert_expert_scope(head)
        disabled, _ = self.head(False)
        adapter.set_trainable(disabled, False, False)
        self.assertFalse(disabled.model.proj_out_2.enabled)
        with self.assertRaises(ValueError):
            adapter.set_trainable(disabled, True, True)
        with torch.no_grad():
            disabled.model.proj_out_2.delta_bias[0] = 1.
        with self.assertRaisesRegex(ValueError, "remain zero"):
            adapter.assert_expert_scope(disabled)

    def test_autocast_keeps_fp32_delta_computation_and_state_copies(self):
        head, _ = self.head()
        module = head.model.proj_out_2
        with torch.no_grad():
            module.delta_weight.fill_(.019)
            module.delta_bias.fill_(.007)
        x = torch.randn(2, 5, 8)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            base = module.base(x)
            result = module(x)
        expected = base + torch.nn.functional.linear(x.float(), module.delta_weight, module.delta_bias).to(base.dtype)
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(module.delta_weight.dtype, torch.float32)
        copied = adapter.projector_state_dict(head)
        copied["delta_weight"].zero_()
        self.assertTrue(bool(module.delta_weight.any()))

    def test_generation_validation_no_grad_restores_requires_grad_and_policy(self):
        head, _ = self.head()
        ep = episode()
        with torch.no_grad():
            for p in adapter.projector_parameters(head):
                p.add_(.01)
        rng = torch.get_rng_state().clone()
        flags = [p.requires_grad for p in adapter.projector_parameters(head)]
        result = adapter.generated_prefix_metrics(head, ep, 1, seed=88, action_steps=2)
        self.assertFalse(result["loss"].requires_grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(head.model.proj_out_2.enabled)
        self.assertEqual([p.requires_grad for p in adapter.projector_parameters(head)], flags)
        changed = copy.deepcopy(ep)
        changed["targets"][1].add_(99.)
        after = adapter.generated_prefix_metrics(head, changed, 1, seed=88, action_steps=2)
        self.assertTrue(torch.equal(result["prediction"], after["prediction"]))
        with patch.object(adapter, "generated_prefix_objective", side_effect=RuntimeError("failed metric")), self.assertRaises(RuntimeError):
            adapter.generated_prefix_metrics(head, ep, 1, seed=88, action_steps=2)
        self.assertEqual([p.requires_grad for p in adapter.projector_parameters(head)], flags)
        with adapter.all_adapters_disabled(head):
            baseline = adapter.generated_prefix_metrics(head, ep, 1, seed=88, action_steps=2)
            self.assertFalse(head.model.proj_out_2.enabled)
        self.assertFalse(torch.equal(result["prediction"], baseline["prediction"]))

    def test_duplicate_wrong_target_no_bias_and_bad_flags_rejected(self):
        head, _ = self.head()
        with self.assertRaises(ValueError):
            adapter.install_projector(head)
        bare = ToyHead().eval()
        bare.model.proj_out_2 = nn.Linear(8, 8, bias=False)
        with self.assertRaises(TypeError):
            adapter.install_projector(bare)
        for bad in (1, None, "true"):
            with self.assertRaises(TypeError):
                adapter.install_projector(ToyHead(), bad)


if __name__ == "__main__":
    unittest.main()
