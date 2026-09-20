"""Frozen-expert gradient and paired flow-loss tests using a tiny CPU fake head.

These test tensor wiring, not numeric parity with the full HAMLET checkpoint.
The fake expert shares the call signature used by the real diffusion expert.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from gr00t.long_memory.hamlet import episode_flow_loss, flow_loss, sample_noise_time


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 8)

    def forward(self, state, embodiment_id):
        return self.linear(state)


class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)

    def forward(self, trajectory, timestep, embodiment_id):
        return self.linear(trajectory) + timestep[:, None, None].to(trajectory.dtype) / 100


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, hidden, embodiment_id):
        return self.linear(hidden)


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = nn.Linear(6, 8)
        self.action = nn.Linear(8, 8)

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask,
                timestep, return_all_hidden_states, temb_add, **kwargs):
        mask = encoder_attention_mask.unsqueeze(-1).to(encoder_hidden_states.dtype)
        pooled = (encoder_hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        output = torch.tanh(self.action(hidden_states) + self.context(pooled)[:, None])
        return output, [output]


class TinyFrozenHead(nn.Module):
    def __init__(self, alternate=False, positions=False):
        super().__init__()
        self.config = SimpleNamespace(add_pos_embed=positions, use_alternate_vl_dit=alternate)
        self.num_timestep_buckets = 100
        self.state_encoder, self.action_encoder = StateEncoder(), ActionEncoder()
        self.model, self.action_decoder = Expert(), Decoder()
        self.position_embedding = nn.Embedding(10, 8)
        self.eval().requires_grad_(False)

    def sample_time(self, batch, device, dtype):
        return torch.rand(batch, device=device, dtype=dtype) * 0.8 + 0.1


def inputs(dtype=torch.float32):
    return {
        "features": torch.randn(2, 5, 6, dtype=dtype, requires_grad=True),
        "state": torch.randn(2, 1, 3, dtype=dtype),
        "target": torch.randn(2, 3, 4, dtype=dtype),
        "target_mask": torch.ones(2, 3, 4, dtype=torch.bool),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool),
        "image_mask": torch.tensor([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool),
        "embodiment_id": torch.zeros(2, dtype=torch.long),
        "noise": torch.randn(2, 3, 4, dtype=dtype),
        "time": torch.tensor([0.2, 0.7], dtype=dtype)[:, None, None],
    }


class TestFrozenBridge(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(63)

    def test_explicit_noise_time_determinism_and_input_not_expert_gradient(self):
        for alternate in (False, True):
            head, values = TinyFrozenHead(alternate=alternate, positions=True), inputs()
            first, second = flow_loss(head, **values), flow_loss(head, **values)
            self.assertTrue(torch.equal(first["loss"], second["loss"]))
            self.assertEqual(first["loss"].shape, torch.Size([]))
            self.assertEqual(first["prediction"].shape, values["target"].shape)
            self.assertEqual(first["loss"].dtype, torch.float32)
            first["loss"].backward()
            self.assertGreater(values["features"].grad.abs().sum().item(), 0)
            self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))

    def test_activation_checkpointing_keeps_same_loss_and_feature_gradient(self):
        head, values = TinyFrozenHead(), inputs()
        ordinary = flow_loss(head, **values)
        ordinary["loss"].backward()
        gradient = values["features"].grad.clone()
        values["features"].grad = None
        checked = flow_loss(head, **values, activation_checkpointing=True)
        checked["loss"].backward()
        self.assertTrue(torch.equal(ordinary["loss"], checked["loss"]))
        self.assertTrue(torch.allclose(gradient, values["features"].grad, atol=1e-7))

    def test_invalid_target_padding_is_ignored_before_encoder_and_reduction(self):
        head, values = TinyFrozenHead(), inputs()
        values["target_mask"][..., -1] = False
        values["target"][..., -1] = 0
        expected = flow_loss(head, **values)
        values["target"][..., -1] = float("nan")
        actual = flow_loss(head, **values)
        self.assertTrue(torch.isfinite(actual["loss"]))
        self.assertTrue(torch.equal(expected["loss"], actual["loss"]))
        actual["loss"].backward()
        self.assertTrue(torch.isfinite(values["features"].grad).all())

    def test_bfloat16_prediction_has_fp32_loss_reduction(self):
        head, values = TinyFrozenHead().to(dtype=torch.bfloat16), inputs(torch.bfloat16)
        result = flow_loss(head, **values)
        self.assertEqual(result["prediction"].dtype, torch.bfloat16)
        self.assertEqual(result["loss"].dtype, torch.float32)
        self.assertEqual(result["velocity_mae"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(result["loss"]))

    def test_no_valid_target_partial_random_args_and_unfrozen_head_are_rejected(self):
        head, values = TinyFrozenHead(), inputs()
        values["target_mask"].zero_()
        with self.assertRaisesRegex(ValueError, "no valid targets"):
            flow_loss(head, **values)
        values = inputs()
        values["time"] = None
        with self.assertRaisesRegex(ValueError, "both noise and time"):
            flow_loss(head, **values)
        values = inputs()
        head.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            flow_loss(head, **values)
        head.eval().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "requires_grad"):
            flow_loss(head, **values)

    def test_seeded_noise_pairing_does_not_advance_global_rng(self):
        head, target = TinyFrozenHead(), torch.zeros(2, 3, 4)
        before = torch.random.get_rng_state().clone()
        first = sample_noise_time(head, target, seed=123)
        after = torch.random.get_rng_state()
        second = sample_noise_time(head, target, seed=123)
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))

    def test_episode_wrapper_replaces_only_short_suffix_and_keeps_gradient(self):
        head = TinyFrozenHead()
        data = {
            "features": [torch.randn(5, 6)],
            "state": torch.randn(1, 3),
            "targets": torch.randn(1, 3, 4),
            "target_mask": torch.ones(1, 3, 4, dtype=torch.bool),
            "attention_masks": [torch.ones(5, dtype=torch.bool)],
            "image_masks": [torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)],
            "embodiment_id": 0,
        }
        fused = data["features"][0][-2:][None].clone().requires_grad_(True)
        baseline = episode_flow_loss(head, data, 0, seed=17)
        unchanged = episode_flow_loss(head, data, 0, fused, seed=17)
        self.assertTrue(torch.equal(baseline["loss"], unchanged["loss"]))
        unchanged["loss"].backward()
        self.assertGreater(fused.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))

    def test_actual_original_forward_matches_cached_bridge_on_tiny_expert(self):
        """Execute the repository's real forward body without loading its VLM.

        Backbone processing is an identity stub because the cache's contract is
        post-VLLN/post-short-memory. This is exact flow-wiring parity on a tiny
        head, not an assertion that real checkpoint inference was validated.
        Original state dropout is disabled explicitly, matching the author
        checkpoint's probability zero and our frozen-objective design.
        """
        source = Path(__file__).resolve().parents[1] / "gr00t/model/gr00t_n1d6/gr00t_n1d6.py"
        parsed = ast.parse(source.read_text())
        head_class = next(
            node for node in parsed.body
            if isinstance(node, ast.ClassDef) and node.name == "Gr00tN1d6ActionHead"
        )
        forward = copy.deepcopy(next(
            node for node in head_class.body if isinstance(node, ast.FunctionDef) and node.name == "forward"
        ))
        forward.decorator_list = []

        class Batch(dict):
            def __getattr__(self, name):
                return self[name]

        namespace = {"torch": torch, "F": torch.nn.functional, "BatchFeature": Batch}
        code = ast.fix_missing_locations(ast.Module(body=[forward], type_ignores=[]))
        exec(compile(code, str(source), "exec"), namespace)
        original_forward = namespace["forward"]

        for alternate in (False, True):
            head, values = TinyFrozenHead(alternate=alternate, positions=True), inputs()
            head.state_dropout_prob = 0
            head.state_additive_noise_scale = 0
            head.set_frozen_modules_to_eval_mode = lambda: None
            head.process_backbone_output = lambda output, action_inputs_B: output
            # Include both temporal and channel padding, as the real processor
            # pads actions and marks only physical dimensions/horizon valid.
            values["target_mask"][..., -1] = False
            values["target_mask"][:, -1] = False
            values["target"][~values["target_mask"]] = 0
            backbone = Batch(
                backbone_features=values["features"],
                backbone_attention_mask=values["attention_mask"],
                image_mask=values["image_mask"],
            )
            action = Batch(state=values["state"], action=values["target"],
                           action_mask=values["target_mask"], embodiment_id=values["embodiment_id"])
            decoder_outputs = []
            hook = head.action_decoder.register_forward_hook(
                lambda module, args, output: decoder_outputs.append(output.detach().clone())
            )
            torch.manual_seed(901)
            original = original_forward(head, backbone, action)
            original["loss"].backward()
            gradient = values["features"].grad.clone()
            values["features"].grad = None
            torch.manual_seed(901)
            noise, time = sample_noise_time(head, values["target"])
            values.update(noise=noise, time=time)
            bridged = flow_loss(head, **values)
            bridged["loss"].backward()
            hook.remove()
            self.assertTrue(torch.equal(decoder_outputs[0], decoder_outputs[1]))
            self.assertTrue(torch.equal(original["loss"], bridged["loss"]))
            self.assertTrue(torch.equal(gradient, values["features"].grad))


if __name__ == "__main__":
    unittest.main()
