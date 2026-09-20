"""Small CPU tests: gradients/causality are not robot-task success evidence."""

import copy
import unittest

import torch

from gr00t.long_memory.core_v3 import ActionValueMemory, MemoryV3Config
from gr00t.long_memory.replay_v3 import encode_until, read_bank, storage_prediction


def model(**kwargs):
    config = dict(feature_dim=6, state_dim=3, action_dim=4, hidden_dim=8, num_heads=2,
                  capacity=3, min_fill=1, residual_init=0.03, time_scale=2)
    config.update(kwargs)
    return ActionValueMemory(MemoryV3Config(**config))


def episode(steps=8):
    return {"episode_id": 10, "short": torch.randn(steps + 1, 2, 6),
            "moment": torch.randn(steps + 1, 2, 6), "state": torch.randn(steps + 1, 3),
            "actions": torch.randn(steps, 2, 4), "action_mask": torch.ones(steps, 2, dtype=torch.bool),
            "transition_valid": torch.ones(steps, dtype=torch.bool),
            "decision_mask": torch.ones(steps + 1, dtype=torch.bool),
            "frames": torch.arange(steps + 1) * 2}


def grad_sum(parameters):
    return sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)


class TestV3Core(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(72)
        torch.set_num_threads(1)

    def test_role_shapes_and_multi_query_scores(self):
        memory, data = model(), episode()
        encoded = encode_until(memory, data, 5)
        self.assertEqual(encoded["tokens"].shape, (5, 3, 8))
        self.assertEqual(encoded["event"].shape, (5, 8))
        result = read_bank(memory, data, 5, [0, 2, 4], encoded)
        self.assertEqual(result["event_scores"].shape, (1, 3))
        self.assertEqual(result["null_score"].shape, (1, 1))
        self.assertEqual(result["read"].shape, (1, 2, 8))
        self.assertEqual(result["weights"].shape, (1, 2, 4))
        torch.testing.assert_close(result["weights"].sum(-1), torch.ones(1, 2))
        self.assertFalse(torch.equal(result["read"][:, 0], result["read"][:, 1]))

    def test_initial_zero_fusion_is_exact_baseline(self):
        memory, data = model(residual_init=0), episode()
        for bank in ([], [0, 2]):
            self.assertTrue(torch.equal(read_bank(memory, data, 4, bank)["fused_short"], data["short"][4:5]))

    def test_empty_and_zero_value_bank_identity_after_affine_biases_train(self):
        memory, data = model(), episode()
        with torch.no_grad():
            for name, parameter in memory.named_parameters():
                if "bias" in name:
                    parameter.normal_(mean=2, std=0.5)
        encoded = encode_until(memory, data, 4)
        encoded["values"] = torch.zeros_like(encoded["values"])
        for bank in ([], [0, 1, 2]):
            result = read_bank(memory, data, 4, bank, encoded)
            self.assertTrue(torch.equal(result["fused_short"], data["short"][4:5]))
            self.assertEqual(float(result["residual_norm"]), 0)

    def test_action_gradient_reaches_recomputed_events_after_zero_init_warmup(self):
        memory, data = model(residual_init=0), episode()
        optimizer = torch.optim.SGD(memory.reader_parameters(), lr=0.05)
        target = torch.randn(1, 2, 6)
        result = read_bank(memory, data, 4, [0, 2])
        (result["fused_short"] - target).square().mean().backward()
        self.assertGreater(grad_sum(memory.fusion.parameters()), 0)
        self.assertEqual(grad_sum(memory.action_encoder.parameters()), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        result = read_bank(memory, data, 4, [0, 2])
        (result["fused_short"] - target).square().mean().backward()
        for module in (memory.action_encoder, memory.situation_pool, memory.outcome_pool, memory.summary_pool,
                       memory.query, memory.key, memory.value, memory.time_encoder):
            self.assertGreater(grad_sum(module.parameters()), 0, type(module).__name__)
        self.assertEqual(grad_sum(memory.writer_parameters()), 0)

    def test_writer_gradient_does_not_enter_reader_and_groups_are_disjoint(self):
        memory, data = model(), episode()
        prediction = storage_prediction(memory, data, 4, [0, 1, 2])
        losses = prediction["logits"].new_tensor([1, 2, 0, 3])
        (prediction["logits"].softmax(-1) * losses).sum().backward()
        self.assertGreater(grad_sum(memory.writer_parameters()), 0)
        self.assertEqual(grad_sum(memory.reader_parameters()), 0)
        readers, writers = {id(p) for p in memory.reader_parameters()}, {id(p) for p in memory.writer_parameters()}
        self.assertFalse(readers & writers)
        self.assertEqual(readers | writers, {id(p) for p in memory.parameters()})

    def test_ranking_scores_differentiate_query_event_and_time(self):
        memory, data = model(residual_init=0), episode()
        scores = read_bank(memory, data, 5, [0, 3])["event_scores"]
        (scores[0, 0] - scores[0, 1]).backward()
        for module in (memory.query, memory.key, memory.situation_pool, memory.time_encoder):
            self.assertGreater(grad_sum(module.parameters()), 0)

    def test_timestamps_enter_both_keys_and_values_not_just_reader_logits(self):
        memory, data = model(), episode()
        other = copy.deepcopy(data)
        other["frames"] += 200
        first, second = encode_until(memory, data, 4), encode_until(memory, other, 4)
        torch.testing.assert_close(first["tokens"], second["tokens"])
        self.assertFalse(torch.equal(first["keys"], second["keys"]))
        self.assertFalse(torch.equal(first["values"], second["values"]))

    def test_learned_pooling_distinguishes_equal_mean_token_sets(self):
        memory, data = model(), episode()
        # Equal per-event means, different token distributions: no premature averaging.
        data["short"].zero_()
        data["moment"].zero_()
        changed = copy.deepcopy(data)
        changed["moment"][:, 0] = torch.randn_like(changed["moment"][:, 0]) * 3
        changed["moment"][:, 1] = -changed["moment"][:, 0]
        torch.testing.assert_close(data["moment"].mean(1), changed["moment"].mean(1))
        self.assertFalse(torch.allclose(encode_until(memory, data, 4)["tokens"],
                                        encode_until(memory, changed, 4)["tokens"]))

    def test_masked_action_suffix_and_passive_padding_cannot_leak(self):
        memory, data = model(), episode()
        data["action_mask"][:, 1] = False
        changed = copy.deepcopy(data)
        changed["actions"][:, 1] = float("nan")
        torch.testing.assert_close(encode_until(memory, data, 4)["tokens"],
                                   encode_until(memory, changed, 4)["tokens"])
        data["action_mask"][:] = False
        changed = copy.deepcopy(data)
        changed["actions"][:] = float("nan")
        torch.testing.assert_close(encode_until(memory, data, 4)["tokens"],
                                   encode_until(memory, changed, 4)["tokens"])

    def test_bad_prefix_and_bad_dimensions_rejected(self):
        memory, data = model(), episode()
        data["action_mask"][0] = torch.tensor([False, True])
        with self.assertRaisesRegex(ValueError, "prefix"):
            encode_until(memory, data, 4)
        for kwargs in ({"num_heads": 3}, {"time_scale": 0}, {"capacity": 1, "min_fill": 2}):
            with self.assertRaises(ValueError):
                model(**kwargs)

    def test_malformed_checkpoint_config_rejects_bool_float_dimensions_and_nonfinite(self):
        for name in ("feature_dim", "state_dim", "action_dim", "hidden_dim", "num_heads",
                     "capacity", "min_fill", "max_victims"):
            for value in (True, 2.0, "2"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    model(**{name: value})
        for name in ("time_scale", "temperature", "residual_init"):
            for value in (True, "1.0", float("nan"), float("inf"), float("-inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    model(**{name: value})


if __name__ == "__main__":
    unittest.main()
