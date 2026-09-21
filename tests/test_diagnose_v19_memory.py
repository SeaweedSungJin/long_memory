"""CPU contracts for visible-event V19 ablations and exact source provenance."""
from dataclasses import replace
import random
import unittest
from unittest.mock import patch

import torch

from run_scripts.robomme import diagnose_v19_memory as diagnostic
from run_scripts.robomme.representation_core_v18 import RepresentationConfigV18, RepresentationMemoryV18
from tests.test_deployment_objective_v9 import CategoryLinear, TinyActionEncoder, TinyDiT, TinyExpert


class DiagnosticDiT(TinyDiT):
    def forward(self, *args, return_all_hidden_states=False, **kwargs):
        output = super().forward(*args, **kwargs)
        return (output, []) if return_all_hidden_states else output


class DiagnosticExpert(TinyExpert):
    def __init__(self):
        super().__init__()
        self.action_dim = 10
        self.action_encoder = TinyActionEncoder(10, 8)
        self.action_decoder = CategoryLinear(8, 10)
        self.model = DiagnosticDiT()
        self.requires_grad_(False).eval()

    def sample_time(self, n, device, dtype):
        return torch.rand(n, device=device, dtype=dtype)


def fixture():
    torch.manual_seed(19)
    cfg = RepresentationConfigV18(feature_dim=8, state_dim=4, num_short_tokens=4,
        hidden_dim=8, num_heads=2, short_window=4, capacity_events=8)
    core = RepresentationMemoryV18(cfg).eval()
    with torch.no_grad():
        core.memory.fusion_projection.weight.normal_(std=.1)
    head = DiagnosticExpert()
    n = 13
    short = torch.randn(n, 4, 8)
    frames = torch.tensor([0] + list(range(7, 7 + 16 * (n - 1), 16)))
    ep = {"episode_id": 777, "frames": frames, "is_demo": torch.arange(n) < 2,
        "short": short, "moment": torch.randn_like(short), "state": torch.randn(n, 4),
        "features": [torch.cat((torch.randn(3, 8), short[j])) for j in range(n)],
        "attention_masks": [torch.tensor([True, False, True, True, True, True, True]) for _ in range(n)],
        "image_masks": [torch.tensor([True, False, True, False, False, False, False]) for _ in range(n)],
        "embodiment_id": 0, "targets": torch.randn(n - 1, 5, 10),
        "target_mask": torch.zeros(n - 1, 5, 10, dtype=torch.bool),
        "action_mask": torch.ones(n - 1, 2, dtype=torch.bool),
        "actions": torch.zeros(n - 1, 2, 10), "decision_mask": torch.arange(n - 1) >= 2}
    ep["target_mask"][..., :8] = True
    ep["target_mask"][:2] = False
    ep["action_mask"][:2] = False
    ep["action_mask"][10, 1] = False
    return head, core, ep


class DiagnoseV19MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_provenance_canonical_padding_fifo_and_read_before_write(self):
        _, core, ep = fixture()
        report = diagnostic.bank_provenance(ep, 10, core.config)
        self.assertEqual(report["visible_event_ids"], list(range(2, 10)))
        self.assertEqual(report["visible_token_count"], 32)
        self.assertEqual(report["events"][0]["source_frames"], [0, 0, 0, 0])
        self.assertEqual(report["events"][1]["source_frames"], [0, 0, 0, 7])
        self.assertEqual(report["events"][2]["source_frames"], [0, 0, 7, 23])
        self.assertEqual(report["events"][3]["source_frames"], [0, 7, 23, 39])
        self.assertEqual(report["events"][0]["deletion_at"], 8)
        self.assertEqual(report["events"][1]["deletion_at"], 9)
        self.assertIsNone(report["events"][2]["deletion_at"])
        self.assertEqual(report["events"][2]["insertion_at"], 2)
        self.assertEqual(report["events"][2]["current_position"], 0)
        self.assertEqual(report["events"][2]["token_positions"], [0, 1, 2, 3])
        self.assertEqual(report["events"][-1]["token_positions"], [28, 29, 30, 31])
        self.assertEqual(report["query"]["source_frames"], ep["frames"][7:11].tolist())
        self.assertIsNone(report["query"]["insertion_at"])
        self.assertTrue(report["events"][0]["is_demo"])
        self.assertFalse(report["events"][2]["is_demo"])
        # No actions/targets/future feature content are required for provenance.
        ep["targets"] = object()
        ep["short"][11:] = torch.nan
        self.assertEqual(report, diagnostic.bank_provenance(ep, 10, core.config))
        empty = diagnostic.bank_provenance(ep, 0, core.config)
        self.assertEqual(empty["visible_event_ids"], [])
        self.assertEqual(empty["query"]["source_frames"], [0] * 4)

    def test_moment_storage_has_endpoint_source_but_query_has_hamlet_window(self):
        _, core, ep = fixture()
        report = diagnostic.bank_provenance(ep, 4, replace(core.config, representation="moment"))
        self.assertEqual(report["events"][3]["source_frames"], [39])
        self.assertEqual(report["events"][3]["hamlet_source_frames"], [0, 7, 23, 39])
        self.assertEqual(report["query"]["source_frames"], [7, 23, 39, 55])

    def test_whole_event_drop_preserves_order_and_original_bank(self):
        bank = torch.arange(24).reshape(1, 12, 2)
        original = bank.clone()
        reduced = diagnostic.drop_event_positions(bank, [1], 4)
        torch.testing.assert_close(reduced, torch.cat((bank[:, :4], bank[:, 8:]), dim=1))
        reduced.zero_()
        self.assertTrue(torch.equal(bank, original))
        untouched = diagnostic.drop_event_positions(bank, [], 4)
        untouched.zero_()
        self.assertTrue(torch.equal(bank, original))
        self.assertEqual(diagnostic.drop_event_positions(bank, [0, 1, 2], 4).shape, (1, 0, 2))
        for positions in ([3], [-1], [True], [1, 1]):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                diagnostic.drop_event_positions(bank, positions, 4)
        with self.assertRaises(ValueError):
            diagnostic.drop_event_positions(bank[:, :11], [], 4)

    def test_partitions_and_random_sets_and_blocks_are_visible_paired_reproducible(self):
        ids = list(range(10, 21))
        rng_before, torch_before = random.getstate(), torch.get_rng_state().clone()
        specs = diagnostic.intervention_specs(ids, blocks=4, removal_seed=83)
        self.assertEqual(specs, diagnostic.intervention_specs(ids, blocks=4, removal_seed=83))
        self.assertEqual(random.getstate(), rng_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        blocks = [s for s in specs if s["removal_kind"] == "contiguous-visible-event-block"]
        random_sets = [s for s in specs if s["removal_kind"] == "same-size-random-event-set"]
        random_blocks = [s for s in specs if s["removal_kind"] == "same-size-random-contiguous-event-block"]
        self.assertEqual(len(specs), 14)
        self.assertEqual(len(random_blocks), 4)
        self.assertEqual([i for block in blocks for i in block["removed_event_ids"]], ids)
        for block, control in zip(blocks, random_sets):
            self.assertEqual(len(block["removed_event_ids"]), len(control["removed_event_ids"]))
            self.assertLessEqual(set(control["removed_event_ids"]), set(ids))
            self.assertEqual(control["removal_seed"], 83 + control["block_index"])
            # V2 adds an independent draw without changing v1 random sets.
            expected = sorted(random.Random(83 + control["block_index"]).sample(range(len(ids)),
                                                                                 len(block["removed_event_ids"])))
            self.assertEqual(control["removed_positions"], expected)
        self.assertTrue(any(b["removed_event_ids"] != r["removed_event_ids"] for b, r in zip(blocks, random_sets)))
        for block, control in zip(blocks, random_blocks):
            size = len(block["removed_event_ids"])
            self.assertEqual(len(control["removed_event_ids"]), size)
            self.assertLessEqual(set(control["removed_event_ids"]), set(ids))
            self.assertEqual(control["removal_seed"], 83 + 1000003 + control["block_index"])
            start = random.Random(control["removal_seed"]).randrange(len(ids) - size + 1)
            self.assertEqual(control["removed_positions"], list(range(start, start + size)))
            self.assertEqual(control["removed_event_ids"], ids[start:start + size])

    def test_random_blocks_use_every_valid_start_and_handle_empty_or_full_bank(self):
        # Mock the sampler to choose the final valid start. This independently
        # catches an off-by-one that would never remove the newest event.
        with patch.object(random.Random, "randrange", return_value=6) as draw:
            specs = diagnostic.intervention_specs(list(range(8)), blocks=4)
        self.assertEqual(draw.call_count, 4)
        for call in draw.call_args_list:
            self.assertEqual(call.args, (7,))
        for spec in specs:
            if spec["intervention"].startswith("random-block-"):
                self.assertEqual(spec["removed_event_ids"], [6, 7])
        for ids, blocks in (([], 4), ([9], 4), (list(range(8)), 1)):
            specs = diagnostic.intervention_specs(ids, blocks=blocks)
            fixed = {spec["block_index"]: spec for spec in specs
                     if spec["removal_kind"] == "contiguous-visible-event-block"}
            for spec in specs:
                if spec["intervention"].startswith("random-block-"):
                    self.assertEqual(len(spec["removed_event_ids"]),
                                     len(fixed[spec["block_index"]]["removed_event_ids"]))
                    self.assertLessEqual(set(spec["removed_event_ids"]), set(ids))

    def test_full_interventions_fix_seeds_inputs_masks_and_leave_models_unchanged(self):
        head, core, ep = fixture()
        initial_head = {key: value.clone() for key, value in head.state_dict().items()}
        initial_core = {key: value.clone() for key, value in core.state_dict().items()}
        initial_ep = {key: value.clone() for key, value in ep.items() if isinstance(value, torch.Tensor)}
        head._inference_gen = torch.Generator().manual_seed(901)
        inference_rng = head._inference_gen.get_state().clone()
        rng_before = torch.get_rng_state().clone()
        short_and_query = []
        read = core.read_from_bank

        def audited_read(short, query, bank, **kwargs):
            short_and_query.append((short.clone(), query.clone()))
            return read(short, query, bank, **kwargs)

        with patch.object(core, "replay", wraps=core.replay) as replay, \
             patch.object(core, "read_from_bank", side_effect=audited_read), \
             patch.object(diagnostic, "episode_flow_v19", wraps=diagnostic.episode_flow_v19) as flow, \
             patch.object(diagnostic, "generated_prefix_objective", wraps=diagnostic.generated_prefix_objective) as generation:
            result = diagnostic.interventions(head, core, ep, 10, seed=31,
                generation_seed=77, removal_seed=90, action_steps=2)
        self.assertEqual(replay.call_count, 1)
        self.assertEqual(flow.call_count, 14)
        self.assertEqual(generation.call_count, 14)
        for call in flow.call_args_list:
            self.assertIs(call.args[1], ep)
            self.assertEqual(call.kwargs["seed"], 31)
            self.assertEqual(call.kwargs["tail_weight"], 1.)
            self.assertFalse(call.kwargs["activation_checkpointing"])
        for call in generation.call_args_list:
            self.assertIs(call.args[1], ep)
            self.assertEqual(call.kwargs["seed"], 77)
        for short, query in short_and_query[1:]:
            self.assertTrue(torch.equal(short, short_and_query[0][0]))
            self.assertTrue(torch.equal(query, short_and_query[0][1]))
        rows = result["records"]
        self.assertEqual(len(rows), 14)
        self.assertEqual(result["version"], "v19_visible_fifo_event_interventions_v2")
        self.assertEqual(rows[0]["intervention"], "normal-bank")
        self.assertEqual(rows[1]["intervention"], "memory-off")
        self.assertEqual(rows[1]["read_event_ids"], [])
        for row in rows:
            self.assertEqual(row["target_valid_values"], 40)
            self.assertEqual(row["executed_prefix_valid_values"], 8)
            self.assertEqual(row["generated_prefix_valid_values"], 8)
            self.assertEqual(row["generated_executed_joint_valid_values"], 7)
            self.assertEqual(row["generated_executed_gripper_valid_values"], 1)
            self.assertAlmostEqual(row["generated_prefix_mse"],
                (7 * row["generated_executed_joint_mse"] + row["generated_executed_gripper_mse"]) / 8, places=5)
        self.assertTrue(any(row["generated_prefix_delta_from_normal_norm"] > 0 for row in rows[1:]))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertTrue(torch.equal(head._inference_gen.get_state(), inference_rng))
        for key, value in initial_head.items():
            self.assertTrue(torch.equal(value, head.state_dict()[key]))
        for key, value in initial_core.items():
            self.assertTrue(torch.equal(value, core.state_dict()[key]))
        for key, value in initial_ep.items():
            self.assertTrue(torch.equal(value, ep[key]))
        self.assertTrue(all(parameter.grad is None for parameter in [*head.parameters(), *core.parameters()]))

    def test_seed_validation_and_eval_requirement_precede_replay(self):
        head, core, ep = fixture()
        for options in ({"seed": True}, {"seed": -1}, {"generation_seed": None},
                        {"removal_seed": -1}, {"blocks": False}, {"action_steps": 0}):
            kwargs = dict(seed=3, generation_seed=8, action_steps=2)
            kwargs.update(options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                diagnostic.interventions(head, core, ep, 10, **kwargs)
        core.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            diagnostic.interventions(head, core, ep, 10, seed=3, generation_seed=8, action_steps=2)


if __name__ == "__main__":
    unittest.main()
