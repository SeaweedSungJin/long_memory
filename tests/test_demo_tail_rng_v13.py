"""No-GPU tests for extra-ingest RNG isolation; not online action parity."""
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from run_scripts.robomme.demo_tail_rng_v13 import isolated_visual_ingest_rng


def sample(generator):
    return random.random(), np.random.random(), torch.rand(3), torch.rand(3, generator=generator)


class VisualIngestRngTests(unittest.TestCase):
    def test_cpu_global_and_episode_streams_restored_on_success_and_failure(self):
        for fail in (False, True):
            generator = torch.Generator().manual_seed(19)
            with isolated_visual_ingest_rng("cpu", generator):
                expected = sample(generator)
            try:
                with isolated_visual_ingest_rng("cpu", generator):
                    sample(generator)
                    if fail:
                        raise RuntimeError("failed frame")
            except RuntimeError:
                pass
            actual = sample(generator)
            self.assertEqual(actual[:2], expected[:2])
            self.assertTrue(torch.equal(actual[2], expected[2]))
            self.assertTrue(torch.equal(actual[3], expected[3]))

    def test_cpu_does_not_touch_cuda(self):
        with patch("torch.cuda.get_rng_state", side_effect=AssertionError("no CUDA")), \
             patch("torch.cuda.get_rng_state_all", side_effect=AssertionError("no other GPU")), \
             isolated_visual_ingest_rng("cpu"):
            torch.rand(1)

    def test_cuda_snapshots_only_explicit_device_not_all_devices(self):
        state = torch.tensor([3, 7], dtype=torch.uint8)
        with patch("torch.cuda.get_rng_state", return_value=state) as get, \
             patch("torch.cuda.set_rng_state") as put, \
             patch("torch.cuda.get_rng_state_all", side_effect=AssertionError("other GPU")):
            with isolated_visual_ingest_rng("cuda:1"):
                torch.rand(1)
        get.assert_called_once_with(torch.device("cuda:1"))
        self.assertTrue(torch.equal(put.call_args.args[0], state))
        self.assertEqual(put.call_args.args[1], torch.device("cuda:1"))

    def test_no_seed_reset_and_nested_isolation(self):
        with patch("torch.manual_seed", side_effect=AssertionError("do not reseed")):
            before = torch.get_rng_state().clone()
            with isolated_visual_ingest_rng("cpu"):
                torch.rand(4)
                nested = torch.get_rng_state().clone()
                with isolated_visual_ingest_rng("cpu"):
                    torch.rand(9)
                self.assertTrue(torch.equal(nested, torch.get_rng_state()))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_bad_device_generator_rejected(self):
        for device in ("cuda", "meta"):
            with self.assertRaises(ValueError):
                with isolated_visual_ingest_rng(device):
                    self.fail("Invalid device accepted")
        with self.assertRaises(TypeError):
            with isolated_visual_ingest_rng("cpu", object()):
                self.fail("Invalid generator accepted")
        with self.assertRaises(ValueError):
            with isolated_visual_ingest_rng("cuda:0", torch.Generator()):
                self.fail("Wrong-device generator accepted")


if __name__ == "__main__":
    unittest.main()
