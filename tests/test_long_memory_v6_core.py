"""CPU causal/gradient/protocol tests; passing is not robot success evidence."""

import copy
import unittest

import torch

from gr00t.long_memory.core_v6 import CANDIDATE_NAMES, VisualMemoryV6, VisualMemoryV6Config, observation_image_tokens
from gr00t.long_memory.replay_v6 import bound_archive, build_candidates, observation_from_episode, prepare_candidates, proposal_vector, uniform_positions
from gr00t.long_memory.objectives_v6 import cvom_loss, paired_utility_targets, select_candidate


def model(**kwargs):
    options = dict(feature_dim=6, state_dim=3, hidden_dim=8, visual_tokens=3,
                   num_heads=2, temporal_layers=1, max_archive_events=7, read_budget=3)
    options.update(kwargs)
    return VisualMemoryV6(VisualMemoryV6Config(**options))


def episode(n=12):
    short = torch.randn(n, 2, 6)
    features = [torch.cat((torch.randn(9, 6), short[i]), 0) for i in range(n)]
    return {"frames": torch.arange(n) * 16, "features": features, "short": short,
            "state": torch.randn(n, 3), "image_masks": [torch.ones(11, dtype=torch.bool) for _ in range(n)],
            "attention_masks": [torch.ones(11, dtype=torch.bool) for _ in range(n)],
            "is_demo": torch.arange(n) < 3,
            "targets": torch.full((n, 4, 3), float("nan")),
            "actions": torch.full((n, 4, 3), float("nan"))}


def grads(parameters):
    return sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)


class V6CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(54)

    def test_config_rejects_malformed_dimensions(self):
        for field in ("feature_dim", "state_dim", "hidden_dim", "visual_tokens", "num_heads",
                      "temporal_layers", "max_archive_events", "read_budget"):
            for value in (True, 0, -1, 2.0, "2"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    model(**{field: value})
        for field in ("time_scale", "temperature"):
            for value in (True, 0, float("nan"), float("inf"), "2"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    model(**{field: value})
        with self.assertRaises(ValueError):
            model(num_heads=3)
        with self.assertRaises(ValueError):
            model(read_budget=8)

    def test_value_tokens_preserve_more_than_mean_and_fp32(self):
        memory, data = model(), episode()
        observation = observation_from_episode(data, 0)
        inputs = {k: v for k, v in observation.items() if k != "event_id"}
        first = memory.encode_observation(**inputs)
        changed = copy.deepcopy(inputs)
        changed["features"][:3, 0] += 3
        changed["features"][3:6, 0] -= 3
        second = memory.encode_observation(**changed)
        self.assertEqual(first["values"].shape, (3, 8))
        self.assertEqual(first["values"].dtype, torch.float32)
        self.assertFalse(torch.allclose(first["values"], second["values"]))

    def test_image_mask_always_excludes_short_tail(self):
        data = episode()
        row = observation_from_episode(data, 0)
        first = proposal_vector(row)
        row["features"][-2:] = 1e12
        torch.testing.assert_close(first, proposal_vector(row))
        row["attention_mask"][:3] = False
        self.assertEqual(len(observation_image_tokens(row["features"], row["image_mask"], row["attention_mask"], row["short"])), 6)

    def test_missing_image_rejected_not_zero_fallback(self):
        data = episode()
        data["image_masks"][0][:9] = False
        with self.assertRaisesRegex(ValueError, "no valid image"):
            prepare_candidates(model(), data, 1)

    def test_malformed_masks_and_states_rejected(self):
        for mutation in (lambda d: d["image_masks"].__setitem__(0, torch.ones(11)),
                         lambda d: d["state"].__setitem__(0, float("nan"))):
            data = episode()
            mutation(data)
            with self.assertRaises(ValueError):
                prepare_candidates(model(), data, 1)

    def test_only_past_is_read_and_labels_are_not_inputs(self):
        memory, data = model(), episode()
        first = prepare_candidates(memory, data, 6)
        changed = copy.deepcopy(data)
        for index in range(7, len(data["frames"])):
            changed["features"][index].fill_(float("nan"))
        changed["targets"].fill_(1e30)
        changed["actions"].fill_(-1e30)
        second = prepare_candidates(memory, changed, 6)
        for name in CANDIDATE_NAMES[:-1]:
            torch.testing.assert_close(first["candidates"][name]["tokens"], second["candidates"][name]["tokens"])
            self.assertTrue(all(i < 6 for i in first["candidates"][name]["event_ids"]))

    def test_current_or_future_duplicate_unsorted_observations_rejected(self):
        memory, data = model(), episode()
        rows = [observation_from_episode(data, i) for i in range(4)]
        current = observation_from_episode(data, 3)
        for bad in (rows, rows[:2] + [rows[1]], [rows[1], rows[0]]):
            with self.assertRaises(ValueError):
                build_candidates(memory, bad, current)

    def test_empty_archive_exact_null_and_finite(self):
        memory, data = model(), episode()
        result = prepare_candidates(memory, data, 0)
        for candidate in result["candidates"].values():
            self.assertIsNone(candidate["tokens"])
        self.assertTrue(torch.equal(memory.score_candidates(result["query"], result["candidates"]), torch.zeros(4)))

    def test_candidates_equal_budget_order_and_image_token_count(self):
        result = prepare_candidates(model(), episode(), 10)
        self.assertEqual(tuple(result["candidates"]), CANDIDATE_NAMES)
        for name in CANDIDATE_NAMES[:-1]:
            candidate = result["candidates"][name]
            self.assertEqual(candidate["tokens"].shape, (1, 9, 8))
            self.assertEqual(candidate["event_ids"], sorted(set(candidate["event_ids"])))
            self.assertAlmostEqual(float(candidate["event_weights"].sum()), 1, places=6)
        self.assertLessEqual(result["details"]["archive_count"], 7)
        self.assertIn(0, result["details"]["archive_ids"])
        self.assertIn(9, result["details"]["archive_ids"])

    def test_action_gradient_reaches_soft_query_keys_values_temporal_only(self):
        memory = model()
        result = prepare_candidates(memory, episode(), 10)
        tokens = result["candidates"]["uniform"]["tokens"]
        (tokens * torch.randn_like(tokens)).mean().backward()
        for module in (memory.query, memory.key, memory.visual_projection, memory.temporal):
            self.assertGreater(grads(module.parameters()), 0)
        self.assertEqual(grads(memory.cvom_parameters()), 0)

    def test_cvom_gradient_isolation_and_null_anchor(self):
        memory = model()
        result = prepare_candidates(memory, episode(), 10)
        with torch.no_grad():
            memory.cvom.head[-1].weight.normal_()
        prediction = memory.score_candidates(result["query"], result["candidates"])
        self.assertEqual(float(prediction[-1]), 0)
        prediction[:3].sum().backward()
        self.assertEqual(grads(memory.reader_parameters()), 0)
        self.assertGreater(grads(memory.cvom_parameters()), 0)
        reader_ids = {id(p) for p in memory.reader_parameters()}
        cvom_ids = {id(p) for p in memory.cvom_parameters()}
        self.assertFalse(reader_ids & cvom_ids)
        self.assertEqual(reader_ids | cvom_ids, {id(p) for p in memory.parameters()})

    def test_demo_and_timestamp_change_encoding(self):
        memory, data = model(), episode()
        row = observation_from_episode(data, 0)
        inputs = {k: v for k, v in row.items() if k != "event_id"}
        first = memory.encode_observation(**inputs)["summary"]
        self.assertFalse(torch.equal(first, memory.encode_observation(**dict(inputs, is_demo=False))["summary"]))
        self.assertFalse(torch.equal(first, memory.encode_observation(**dict(inputs, frame=100))["summary"]))

    def test_train_eval_deterministic_no_dropout(self):
        memory, data = model(), episode()
        memory.train()
        first = prepare_candidates(memory, data, 10)["candidates"]["hybrid"]["tokens"]
        memory.eval()
        second = prepare_candidates(memory, data, 10)["candidates"]["hybrid"]["tokens"]
        torch.testing.assert_close(first, second)

    def test_bounded_archive_stream_equals_offline_and_preserves_endpoints(self):
        rows = [{"frame": i * i, "event_id": i} for i in range(50)]
        for cap in (1, 2, 3, 7, 32):
            streaming = []
            for end, row in enumerate(rows):
                streaming = bound_archive(streaming + [row], cap)
                self.assertEqual(streaming, bound_archive(rows[:end + 1], cap))
                self.assertLessEqual(len(streaming), cap)
                self.assertEqual(streaming[0]["event_id"], 0)
                if cap > 1:
                    self.assertEqual(streaming[-1]["event_id"], end)

    def test_online_bounded_prefix_has_same_pack_tensors_as_offline(self):
        memory, data = model(), episode(20)
        archive = []
        for decision in range(20):
            current = observation_from_episode(data, decision)
            offline = prepare_candidates(memory, data, decision)
            online = build_candidates(memory, archive, current)
            for name in CANDIDATE_NAMES:
                one, two = offline["candidates"][name], online["candidates"][name]
                self.assertEqual(one["event_ids"], two["event_ids"])
                if one["tokens"] is not None:
                    torch.testing.assert_close(one["tokens"], two["tokens"])
            archive = bound_archive(archive + [current], memory.config.max_archive_events)

    def test_uniform_positions_integer_unique(self):
        for n in range(30):
            for budget in range(1, 10):
                positions = uniform_positions(n, budget)
                self.assertEqual(positions, sorted(set(positions)))
                self.assertEqual(len(positions), min(n, budget))


class V6ObjectiveTests(unittest.TestCase):
    def test_signed_utility_uses_same_null_and_scale(self):
        labels = paired_utility_targets([[1, 2, 3, 2], [2, 3, 4, 3]], scale=1)
        torch.testing.assert_close(labels["gain"], torch.tensor([1., 0., -1., 0.]))
        self.assertEqual(float(labels["gain"][-1]), 0)
        self.assertGreater(labels["signal_fraction"], 0)

    def test_tiny_or_uncertain_differences_do_not_make_rank_labels(self):
        tiny = paired_utility_targets([[1, 1.000001, 1, 1]] * 4, margin=1e-4)
        self.assertEqual(int(tiny["pair_mask"].sum()), 0)
        uncertain = paired_utility_targets([[1, 3, 2, 2], [3, 1, 2, 2]], margin=0)
        self.assertEqual(int(uncertain["pair_mask"].sum()), 0)
        single = paired_utility_targets([[1, 2, 3, 2]], uncertainty_z=0)
        self.assertEqual(int(single["pair_mask"].sum()), 0)

    def test_pairing_common_noise_does_not_inflate_uncertainty(self):
        labels = paired_utility_targets([[100, 102, 103, 101], [1, 3, 4, 2]], scale=1)
        torch.testing.assert_close(labels["stderr"], torch.zeros(4))
        self.assertTrue(bool(labels["pair_mask"][0, 1]))

    def test_ties_train_neutral_regression_without_ranking(self):
        prediction = torch.tensor([1., -1., .5, 0.], requires_grad=True)
        result = cvom_loss(prediction, paired_utility_targets([[1, 1, 1, 1]] * 2))
        self.assertEqual(float(result["ranking"]), 0)
        self.assertGreater(float(result["regression"]), 0)
        result["loss"].backward()
        self.assertEqual(float(prediction.grad[-1]), 0)

    def test_cvom_loss_learns_correct_sign_and_rank(self):
        labels = paired_utility_targets([[1, 2, 3, 2]] * 2, scale=1)
        prediction = torch.zeros(4, requires_grad=True)
        result = cvom_loss(prediction, labels)
        result["loss"].backward()
        self.assertLess(float(prediction.grad[0]), 0)
        self.assertGreater(float(prediction.grad[2]), 0)

    def test_selection_ties_fallback_margin_and_null(self):
        self.assertEqual(select_candidate([0, 0, 0, 0]), "uniform")
        self.assertEqual(select_candidate([.2, .3, .1, 0], improvement_margin=.2), "uniform")
        self.assertEqual(select_candidate([.2, .8, .1, 0]), "relevant")
        self.assertEqual(select_candidate([-.2, -.3, -.1, 0]), "null")
        with self.assertRaises(ValueError):
            select_candidate([0, 1, 2, 3])

    def test_bad_teacher_targets_rejected(self):
        for bad in ([1, 2], [[1, float("nan"), 2, 3]], [[1, -2, 2, 3]], []):
            with self.assertRaises(ValueError):
                paired_utility_targets(bad)


if __name__ == "__main__":
    unittest.main()
