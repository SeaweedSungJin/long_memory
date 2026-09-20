"""CPU softmax/core-mask agreement and availability-aware attention statistics."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from run_scripts.robomme import audit_visual_attention_v11 as audit
from run_scripts.robomme import visual_patch_memory_v11 as core
from tests.test_replay_visual_patch_v11 import observations


class VisualAttentionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(831)
            memory = core.VisualPatchMemoryV11(core.VisualPatchConfig(feature_dim=8, hidden_dim=16)).eval().requires_grad_(False)
        source = observations(6)
        bank = audit.build_visual_patch_bank(memory, source, 5, camera_order=core.CAMERA_ORDER)
        current = memory.encode_observation(source["features"][5][None], source["image_masks"][5][None],
            source["attention_masks"][5][None], [80], [False], camera_order=core.CAMERA_ORDER)
        return memory, source, current, bank

    def test_exact_core_qkv_scale_mask_and_sdpa_weighted_value_parity(self):
        memory, _, current, bank = self.fixture()
        real = F.scaled_dot_product_attention
        with patch.object(core.F, "scaled_dot_product_attention", wraps=real) as sdpa:
            memory.read(current, bank)
            original = sdpa.call_args
            probabilities, checks = audit.attention_probabilities(memory, current, bank)
            observed = sdpa.call_args
        for left, right in zip(original.args, observed.args):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(original.kwargs["attn_mask"], observed.kwargs["attn_mask"]))
        self.assertEqual(original.kwargs["dropout_p"], 0.)
        self.assertIs(original.kwargs["is_causal"], False)
        self.assertNotIn("scale", original.kwargs)
        self.assertEqual(checks["scale"], .5)
        self.assertTrue(checks["sdpa_close"])
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(4, 162))

    def test_uniform_attention_mass_equals_availability_and_raw_age_boundaries(self):
        memory, _, current, bank = self.fixture()
        with torch.no_grad():
            memory.query_projection.weight.zero_()
            memory.key_projection.weight.zero_()
        # Re-encoding Q is unnecessary: K=0 gives uniform logits for any Q.
        probabilities, _ = audit.attention_probabilities(memory, current, bank)
        stats = audit.summarize_attention(memory, current, bank, probabilities)
        for key, availability in (("age_gt_64", .2), ("age_gt_48", .4), ("demo", .4),
                                  ("camera_front_view", .5), ("camera_wrist_view", .5)):
            group = stats["groups"][key]
            self.assertAlmostEqual(group["availability"], availability)
            self.assertAlmostEqual(group["mass"], availability, places=6)
            self.assertAlmostEqual(group["enrichment"], 1., places=5)
        self.assertAlmostEqual(stats["observation_entropy_mean"], math.log(5), places=5)
        self.assertAlmostEqual(stats["effective_observations_mean"], 5., places=5)
        self.assertAlmostEqual(stats["normalized_token_entropy_mean"], 1., places=5)

    def test_invalid_observation_padding_has_zero_mass_even_with_nan_tokens(self):
        memory, _, current, bank = self.fixture()
        valid = bank.valid.clone(); valid[:, 1] = False
        tokens = bank.tokens.clone(); tokens[:, 1] = float("nan")
        bank = replace(bank, valid=valid, tokens=tokens)
        probabilities, checks = audit.attention_probabilities(memory, current, bank)
        self.assertFalse(bool(probabilities[..., 162:324].any()))
        self.assertTrue(checks["sdpa_close"])
        stats = audit.summarize_attention(memory, current, bank, probabilities)
        self.assertEqual(stats["past_observations"], 4)
        self.assertEqual(stats["groups"]["demo"]["availability"], .25)

    def test_observation_only_prefix_access_never_touches_future_gt_or_state(self):
        memory, source, _, _ = self.fixture()
        class Column:
            def __init__(self, data): self.data = data
            def __getitem__(self, index):
                if type(index) is not int or not 0 <= index <= 3:
                    raise AssertionError("Future or broad access")
                return self.data[index]
        class Mapping(dict):
            def __getitem__(self, key):
                if key not in audit.OBSERVATION_KEYS:
                    raise AssertionError("Non-observation access")
                return super().__getitem__(key)
        guarded = Mapping({key: Column(source[key]) for key in audit.OBSERVATION_KEYS})
        guarded.update(targets=object(), actions=object(), state=object())
        result = audit.audit_observation(memory, guarded, 3)
        self.assertEqual(result["past_observations"], 3)
        self.assertEqual(result["past_frames"], [0, 16, 32])
        self.assertTrue(result["parity"]["sdpa_close"])


if __name__ == "__main__":
    unittest.main()
