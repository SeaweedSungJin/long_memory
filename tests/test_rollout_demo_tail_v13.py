"""No simulator/GPU: real original episode loop plus the additional V13 RPC."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from run_scripts.robomme import rollout_demo_tail_v13 as rollout
from tests.test_long_memory_rollout import FakeEnv, FakePolicy


class TailPolicy(FakePolicy):
    def __init__(self, bad=None):
        super().__init__(stride=16)
        self.rpc, self.bad = [], bad

    def call_endpoint(self, endpoint, data):
        self.rpc.append((endpoint, copy.deepcopy(data), len(self.calls), self.action_noise_draws))
        if self.bad == "exception":
            raise RuntimeError("injected RPC error")
        frames = data["frames"]
        response = {"session_id": data["session_id"], "episode_seed": data["episode_seed"],
            "n_demo": data["n_demo"], "ingested_observations": len(frames),
            "first_tail_frame": frames[0], "last_tail_frame": frames[-1],
            "canonical_observations": len(self.calls), "canonical_frame": self.calls[-1][1]["frame_index"],
            "parent_unchanged": True, "rng_preserved": True, "read_performed": False}
        if self.bad == "count":
            response["ingested_observations"] -= 1
        if self.bad == "bool":
            response["rng_preserved"] = 1
        return response


def config(**kwargs):
    return rollout.Config(task_id="PatternLock", model_config="unused", output_dir="unused",
        evaluation_id="synthetic", n_action_steps=16, max_episode_steps=40, **kwargs)


class DemoTailRolloutTests(unittest.TestCase):
    def test_single_rpc_only_after_canonical_primes_before_noise(self):
        policy, env, journal = TailPolicy(), FakeEnv(n_demo=48, finish=20), []
        row, _ = rollout.run_episode(policy, env, config(), 0, 42, journal.append)
        self.assertEqual(row["success"], 1)
        self.assertEqual(len(policy.rpc), 1)
        endpoint, data, calls, draws = policy.rpc[0]
        self.assertEqual(endpoint, rollout.ENDPOINT)
        self.assertEqual((calls, draws), (3, 0))
        self.assertEqual(data["frames"], list(range(33, 48)))
        self.assertEqual(set(data), {"session_id", "episode_seed", "n_demo", "frames", "images", "texts"})
        self.assertEqual(set(data["images"]), {"front_view", "wrist_view"})
        self.assertEqual(data["images"]["front_view"].shape, (15, 8, 8, 3))
        self.assertEqual(data["texts"], ["remember the red object"] * 15)
        self.assertEqual([o[1]["frame_index"] for o in policy.calls], [0, 16, 32, 48, 64])
        self.assertEqual(policy.action_noise_draws, 2)
        self.assertEqual(len([r for r in journal if r["kind"] == "policy_call"]), 5)
        self.assertEqual(len([r for r in journal if r["kind"] == "demo_tail_ingest"]), 1)
        self.assertTrue(env.closed)

    def test_no_demo_or_no_missing_demo_skips_rpc(self):
        for n_demo in (0, 1):
            with self.subTest(n_demo=n_demo):
                policy, env, journal = TailPolicy(), FakeEnv(n_demo=n_demo, finish=1), []
                rollout.run_episode(policy, env, config(), 0, 42, journal.append)
                self.assertEqual(policy.rpc, [])
                self.assertEqual(policy.action_noise_draws, 1)
                self.assertEqual(len([r for r in journal if r["kind"] == "demo_tail_skipped"]), 1)

    def test_rpc_failure_never_becomes_ordinary_failed_episode(self):
        for failure in ("exception", "count", "bool"):
            with self.subTest(failure=failure):
                policy, env, journal = TailPolicy(failure), FakeEnv(n_demo=48), []
                with self.assertRaises(RuntimeError):
                    rollout.run_episode(policy, env, config(), 0, 42, journal.append)
                self.assertEqual(env.actions, [])
                self.assertEqual(policy.action_noise_draws, 0)
                self.assertEqual(len(policy.resets), 2)
                self.assertTrue(env.closed)
                self.assertFalse(any(r["kind"] == "episode_complete" for r in journal))
                self.assertTrue(any(r["kind"] == "episode_error" for r in journal))

    def test_canonical_arm_is_original_episode_loop_without_rpc(self):
        policy, env = FakePolicy(stride=16), FakeEnv(n_demo=48, finish=20)
        with patch.object(rollout, "original_run_episode", wraps=rollout.original_run_episode) as original:
            rollout.run_episode(policy, env, config(include_tail=False), 0, 42, lambda _: None)
            self.assertIs(original.call_args.args[0], policy)
            self.assertIs(original.call_args.args[1], env)
        self.assertEqual(policy.action_noise_draws, 2)

    def test_tail_ingest_does_not_change_native_executed_commands(self):
        on, off = TailPolicy(), FakePolicy(stride=16)
        env_on, env_off = FakeEnv(n_demo=48, finish=20), FakeEnv(n_demo=48, finish=20)
        row_on, _ = rollout.run_episode(on, env_on, config(), 2, 100, lambda _: None)
        row_off, _ = rollout.run_episode(off, env_off, config(include_tail=False), 2, 100, lambda _: None)
        self.assertEqual(row_on, row_off)
        np.testing.assert_array_equal(np.stack(env_on.actions), np.stack(env_off.actions))
        self.assertEqual(on.action_noise_draws, off.action_noise_draws)
        for actual, reference in zip(on.calls, off.calls):
            for key in actual[0]:
                np.testing.assert_array_equal(actual[0][key], reference[0][key])
            for key in actual[1]:
                if key != "session_ids":
                    np.testing.assert_array_equal(actual[1][key], reference[1][key])

    def test_mismatched_stride_rejected_before_first_action(self):
        cfg = config()
        cfg.n_action_steps = 4
        policy, env = TailPolicy(), FakeEnv(n_demo=48)
        with self.assertRaisesRegex(ValueError, "stride16"):
            rollout.run_episode(policy, env, cfg, 0, 42, lambda _: None)
        self.assertEqual(env.actions, [])
        self.assertTrue(env.closed)

    def test_cli_explicit_arm(self):
        common = ["--task-id", "PatternLock", "--model-config", "unused", "--output-dir", "unused",
                  "--evaluation-id", "synthetic"]
        self.assertTrue(rollout.parse_args(common).include_tail)
        self.assertFalse(rollout.parse_args(common + ["--no-include-tail"]).include_tail)


if __name__ == "__main__":
    unittest.main()
