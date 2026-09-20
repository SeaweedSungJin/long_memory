"""CPU isolation/gradient/checkpoint tests for direct-memory Expert adapters."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from gr00t.long_memory.expert_v4 import LoRAConfig, expert_parameters, install_expert_lora
from gr00t.long_memory.expert_v6 import (ExpertMemoryCrossAttention, bridge_parameters,
    bridge_state_dict, install_memory_bridge, load_bridge_state_dict,
    memory_bridge_shapes, v6_episode_flow_loss, v6_flow_loss, with_memory)
from gr00t.long_memory.hamlet import flow_loss


class Projection(nn.Linear):
    def forward(self, x, *unused):
        return super().forward(x)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.to_k = nn.Linear(8, 8)
        self.to_v = nn.Linear(8, 8)
        self.to_out = nn.ModuleList([nn.Linear(8, 8)])

    def forward(self, x, features):
        read = F.scaled_dot_product_attention(self.to_q(x), self.to_k(features), self.to_v(features))
        return self.to_out[0](read)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 8
        self.attn1 = Attention()
        self.norm3 = nn.LayerNorm(8)
        self.ff = nn.Linear(8, 8)

    def forward(self, hidden_states, encoder_hidden_states):
        hidden_states = hidden_states + self.attn1(hidden_states, encoder_hidden_states)
        return hidden_states + self.ff(self.norm3(hidden_states))


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_dim = 8
        self.transformer_blocks = nn.ModuleList([Block(), Block()])

    def forward(self, hidden_states, encoder_hidden_states, **unused):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
        return hidden_states, []


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Expert()
        self.state_encoder = Projection(4, 8)
        self.action_encoder = Projection(3, 8)
        self.action_decoder = Projection(8, 3)
        self.num_timestep_buckets = 10
        self.config = SimpleNamespace(add_pos_embed=False, use_alternate_vl_dit=False)

    def sample_time(self, n, device, dtype):
        return torch.rand(n, device=device, dtype=dtype)


def inputs(dtype=torch.float32):
    return dict(features=torch.randn(1, 6, 8, dtype=dtype), state=torch.randn(1, 1, 4, dtype=dtype),
                target=torch.randn(1, 2, 3, dtype=dtype), target_mask=torch.ones(1, 2, 3),
                attention_mask=torch.ones(1, 6, dtype=torch.bool),
                image_mask=torch.tensor([[True, True, True, False, False, False]]),
                embodiment_id=torch.zeros(1, dtype=torch.long), noise=torch.randn(1, 2, 3, dtype=dtype),
                time=torch.tensor([[[.3]]], dtype=dtype))


def installed(*, lora=True, nonzero=True, dtype=torch.float32):
    torch.manual_seed(91)
    head = Head().to(dtype=dtype).eval().requires_grad_(False)
    if lora:
        install_expert_lora(head, LoRAConfig(rank=2, alpha=4))
    install_memory_bridge(head, hidden_dim=8, block_indices=(0, 1), num_heads=2)
    if nonzero:
        with torch.no_grad():
            for bridge in head.long_memory_v6_bridge.values():
                bridge.output.weight.normal_(0, .04)
    return head


class ExpertV6Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def test_zero_initialization_matches_original_and_no_memory_is_exact(self):
        head = Head().eval().requires_grad_(False)
        data = inputs()
        base = flow_loss(head, **data)
        before = {name: value.clone() for name, value in head.state_dict().items()}
        install_memory_bridge(head, hidden_dim=8, block_indices=(0, 1), num_heads=2)
        for memory in (None, torch.empty(1, 0, 8), torch.randn(1, 5, 8)):
            result = v6_flow_loss(head, **data, memory_tokens=memory)
            for key in ("prediction", "loss", "velocity_mae"):
                self.assertTrue(torch.equal(base[key], result[key]), key)
        for name, value in before.items():
            self.assertTrue(torch.equal(value, head.state_dict()[name]))
        with torch.no_grad():
            for parameter in bridge_parameters(head):
                parameter.fill_(.5)
        for memory in (None, torch.empty(1, 0, 8)):
            self.assertTrue(torch.equal(base["prediction"], v6_flow_loss(head, **data,
                                        memory_tokens=memory)["prediction"]))

    def test_nonzero_adapter_memory_and_lora_gradients_base_immutable(self):
        head = installed()
        memory = torch.randn(1, 7, 8, requires_grad=True)
        before = {name: value.clone() for name, value in head.state_dict().items()}
        v6_flow_loss(head, **inputs(), memory_tokens=memory)["loss"].backward()
        self.assertGreater(memory.grad.abs().sum().item(), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in bridge_parameters(head)))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in expert_parameters(head)))
        allowed = {id(p) for p in bridge_parameters(head)} | {id(p) for p in expert_parameters(head)}
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in head.parameters() if id(p) not in allowed))
        for name, value in before.items():
            self.assertTrue(torch.equal(value, head.state_dict()[name]), name)

    def test_checkpoint_exact_gradient_parity_after_context_has_expired(self):
        normal, checkpointed = installed(), installed()
        data = inputs()
        memory_a = torch.randn(1, 5, 8, requires_grad=True)
        memory_b = memory_a.detach().clone().requires_grad_()
        left = v6_flow_loss(normal, **data, memory_tokens=memory_a)
        right = v6_flow_loss(checkpointed, **data, memory_tokens=memory_b,
                             activation_checkpointing=True)
        self.assertTrue(torch.equal(left["prediction"], right["prediction"]))
        left["loss"].backward()
        # Recompute must use explicit memory_b, not the unrelated active context.
        with with_memory(checkpointed, torch.randn(1, 2, 8) * 100):
            right["loss"].backward()
        self.assertTrue(torch.equal(memory_a.grad, memory_b.grad))
        for (name_a, p_a), (name_b, p_b) in zip(normal.named_parameters(), checkpointed.named_parameters()):
            self.assertEqual(name_a, name_b)
            if p_a.grad is None:
                self.assertIsNone(p_b.grad, name_a)
            else:
                self.assertTrue(torch.equal(p_a.grad, p_b.grad), name_a)

    def test_bridge_only_checkpoint_with_all_detached_inputs_trains(self):
        head = installed(lora=False)
        from torch.utils.checkpoint import checkpoint
        with patch("torch.utils.checkpoint.checkpoint", wraps=checkpoint) as run:
            loss = v6_flow_loss(head, **inputs(), memory_tokens=torch.randn(1, 4, 8),
                                activation_checkpointing=True)["loss"]
            loss.backward()
        run.assert_called_once()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in bridge_parameters(head)))

    def test_frozen_bridge_still_propagates_input_gradients(self):
        head = installed(lora=False).requires_grad_(False)
        memory = torch.randn(1, 4, 8, requires_grad=True)
        v6_flow_loss(head, **inputs(), memory_tokens=memory, activation_checkpointing=True)["loss"].backward()
        self.assertGreater(memory.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_no_memory_has_no_bridge_gradients(self):
        head = installed()
        v6_flow_loss(head, **inputs(), memory_tokens=None, activation_checkpointing=True)["loss"].backward()
        self.assertTrue(all(p.grad is None for p in bridge_parameters(head)))
        self.assertTrue(any(p.grad is not None for p in expert_parameters(head)))

    def test_context_nested_none_exception_and_independent_heads(self):
        head, other = installed(), installed()
        x, features, memory = torch.randn(1, 3, 8), torch.randn(1, 6, 8), torch.randn(1, 4, 8)
        base = head.model(x, features)[0]
        with self.assertRaisesRegex(RuntimeError, "expected"):
            with with_memory(head, memory):
                changed = head.model(x, features)[0]
                self.assertFalse(torch.equal(base, changed))
                self.assertTrue(torch.equal(base, other.model(x, features)[0]))
                with with_memory(head, None):
                    self.assertTrue(torch.equal(base, head.model(x, features)[0]))
                self.assertTrue(torch.equal(changed, head.model(x, features)[0]))
                raise RuntimeError("expected")
        self.assertTrue(torch.equal(base, head.model(x, features)[0]))

    def test_bf16_base_fp32_bridge_and_episode_preserves_full_conditioning(self):
        head = installed(dtype=torch.bfloat16)
        self.assertTrue(all(p.dtype == torch.float32 for p in bridge_parameters(head)))
        features = torch.randn(6, 8)
        episode = dict(features=[features], state=torch.randn(1, 4), targets=torch.randn(1, 2, 3),
                       target_mask=torch.ones(1, 2, 3), attention_masks=[torch.ones(6)],
                       image_masks=[torch.tensor([1, 1, 1, 0, 0, 0])], embodiment_id=0)
        seen = []
        handle = head.model.register_forward_pre_hook(lambda _m, _args, kwargs:
            seen.append(kwargs["encoder_hidden_states"].detach().clone()), with_kwargs=True)
        memory = torch.randn(1, 7, 8, requires_grad=True)
        try:
            result = v6_episode_flow_loss(head, episode, 0, memory, seed=19,
                                          activation_checkpointing=True)
            result["loss"].backward()
        finally:
            handle.remove()
        self.assertEqual(result["prediction"].dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertGreater(memory.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.equal(value, features.to(torch.bfloat16)[None]) for value in seen))

    def test_configuration_rejections_are_before_install(self):
        invalid = [dict(block_indices=[]), dict(block_indices=[True]), dict(block_indices=[2]),
                   dict(block_indices=[0, 0]), dict(hidden_dim=7), dict(num_heads=0), dict(expert_dim=9)]
        for options in invalid:
            head = Head().eval()
            kwargs = dict(hidden_dim=8, block_indices=(0, 1), num_heads=2)
            kwargs.update(options)
            with self.assertRaises(ValueError):
                install_memory_bridge(head, **kwargs)
            self.assertFalse(hasattr(head, "long_memory_v6_bridge"))
            self.assertTrue(all(not block._forward_hooks for block in head.model.transformer_blocks))
        head = installed()
        with self.assertRaisesRegex(ValueError, "already installed"):
            install_memory_bridge(head, hidden_dim=8, block_indices=[0], num_heads=2)

    def test_state_roundtrip_shapes_strict_atomic_and_rng_neutral_shape_check(self):
        head = installed()
        state = bridge_state_dict(head)
        before_rng = torch.get_rng_state().clone()
        self.assertEqual(memory_bridge_shapes(head._long_memory_v6_bridge_config),
                         {key: tuple(value.shape) for key, value in state.items()})
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        for corruption in ("missing", "dtype", "shape", "nan"):
            broken = {key: value.clone() for key, value in state.items()}
            key = next(iter(broken))
            if corruption == "missing":
                broken.pop(key)
            elif corruption == "dtype":
                broken[key] = broken[key].bfloat16()
            elif corruption == "shape":
                broken[key] = broken[key][:1]
            else:
                broken[key].flatten()[0] = float("nan")
            with self.assertRaises(ValueError):
                load_bridge_state_dict(head, broken)
            self.assertTrue(all(torch.equal(value, bridge_state_dict(head)[key]) for key, value in state.items()))
        other = installed(nonzero=False)
        load_bridge_state_dict(other, state)
        self.assertTrue(all(torch.equal(value, bridge_state_dict(other)[key]) for key, value in state.items()))

    def test_thawed_original_or_train_mode_rejected(self):
        head = installed().train()
        with self.assertRaisesRegex(ValueError, "only LoRA/bridge"):
            v6_flow_loss(head, **inputs())
        head.eval().action_decoder.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "only LoRA/bridge"):
            v6_flow_loss(head, **inputs())

    def test_invalid_token_shapes_and_broadcast_batch(self):
        module = ExpertMemoryCrossAttention(8, 8, 2)
        hidden = torch.randn(2, 3, 8)
        for memory in (torch.randn(4, 8), torch.randn(3, 4, 8), torch.randn(1, 4, 7),
                       torch.ones(1, 4, 8, dtype=torch.long)):
            with self.assertRaises(ValueError):
                module(hidden, memory)
        self.assertTrue(torch.equal(hidden, module(hidden, torch.randn(1, 4, 8))))

    def test_core_soft_retrieval_receives_real_action_gradient_through_bridge(self):
        from gr00t.long_memory.core_v6 import VisualMemoryV6, VisualMemoryV6Config
        from gr00t.long_memory.replay_v6 import prepare_candidates
        head = installed(nonzero=True)
        memory = VisualMemoryV6(VisualMemoryV6Config(feature_dim=8, state_dim=4,
            hidden_dim=8, num_heads=2, visual_tokens=3, max_archive_events=8, read_budget=3))
        short = torch.randn(6, 2, 8)
        episode = dict(frames=torch.arange(6) * 16,
            features=[torch.cat([torch.randn(4, 8), short[i]], 0) for i in range(6)],
            short=short, state=torch.randn(6, 4), is_demo=torch.arange(6) < 2,
            image_masks=[torch.ones(6, dtype=torch.bool) for _ in range(6)],
            attention_masks=[torch.ones(6, dtype=torch.bool) for _ in range(6)],
            targets=torch.randn(6, 2, 3), target_mask=torch.ones(6, 2, 3), embodiment_id=0)
        selected = prepare_candidates(memory, episode, 5)["candidates"]["uniform"]
        selected["event_weights"].retain_grad()
        v6_episode_flow_loss(head, episode, 5, selected["tokens"], seed=17,
                              activation_checkpointing=True)["loss"].backward()
        for module in (memory.query, memory.key, memory.visual_projection, memory.temporal):
            total = sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
            self.assertGreater(total, 1e-6)
        self.assertGreater(float(selected["event_weights"].grad.abs().sum()), 1e-5)
        self.assertTrue(all(p.grad is None for p in memory.cvom_parameters()))

    def test_actual_alternate_vl_dit_blocks_cpu_flow_and_checkpoint(self):
        from gr00t.model.modules.dit import AlternateVLDiT
        head = Head().eval().requires_grad_(False)
        head.model = AlternateVLDiT(num_attention_heads=2, attention_head_dim=4, output_dim=8,
            num_layers=2, dropout=0, final_dropout=False, positional_embeddings=None,
            interleave_self_attention=True, cross_attention_dim=8, norm_type="ada_norm")
        head.config.use_alternate_vl_dit = True
        install_expert_lora(head, LoRAConfig(rank=2, alpha=4))
        install_memory_bridge(head, hidden_dim=8, block_indices=(0, 1), num_heads=2)
        with torch.no_grad():
            for adapter in head.long_memory_v6_bridge.values():
                adapter.output.weight.normal_(0, .02)
        memory = torch.randn(1, 5, 8, requires_grad=True)
        result = v6_flow_loss(head, **inputs(), memory_tokens=memory, activation_checkpointing=True)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertGreater(memory.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
