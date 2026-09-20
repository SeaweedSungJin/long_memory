"""CPU simulator-free tests for paired RoboMME rollout causality and journals."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gr00t.eval.sim.robomme.run_long_memory_rollout import (
    Config, _resolve_config, action_chunk, build_observation, check_manifest,
    demo_endpoints, episode_inference_seed, parse_args, read_results,
    run_episode, write_results,
)


def observation(n_frames=1, value=0):
    return {
        "front_rgb_list": [np.full((8, 8, 3), value + index, dtype=np.uint8) for index in range(n_frames)],
        "wrist_rgb_list": [np.full((8, 8, 3), value + index, dtype=np.uint8) for index in range(n_frames)],
        "joint_state_list": [np.arange(7, dtype=np.float32) + value for _ in range(n_frames)],
        "gripper_state_list": [np.array([0.1, 0.1], dtype=np.float32) for _ in range(n_frames)],
    }


class FakePolicy:
    def __init__(self, stride=4, fail_call=None):
        self.stride = stride
        self.calls = []
        self.resets = []
        self.action_noise_draws = 0
        self.fail_call = fail_call

    def reset(self, options):
        self.resets.append(copy.deepcopy(options))

    def get_action(self, obs, options):
        self.calls.append((copy.deepcopy(obs), copy.deepcopy(options)))
        if self.fail_call == len(self.calls):
            raise RuntimeError("injected policy failure")
        if options["prime_only"]:
            return {}, {"bank_fill": len(self.calls) - 1}
        self.action_noise_draws += 1
        values = np.arange((self.stride + 3) * 8, dtype=np.float32).reshape(-1, 8)
        values += self.action_noise_draws * 100
        return {"action.joint_position": values[None, :, :7], "action.gripper_close": values[None, :, 7:]}, {"bank_fill": 4}


class FakeEnv:
    def __init__(self, n_demo=0, finish=5, status="success", error_step=None):
        self.n_demo = n_demo
        self.finish = finish
        self.status = status
        self.error_step = error_step
        self.actions = []
        self.closed = False
        self.reset_kwargs = None

    def reset(self, **kwargs):
        self.reset_kwargs = kwargs
        return observation(self.n_demo + 1), {"task_goal": ["remember the red object"]}

    def step(self, action):
        self.actions.append(action.copy())
        action[:] = -999  # Wrappers must not mutate the remembered command.
        steps = len(self.actions)
        if self.error_step == steps:
            return None, 0, True, False, {"status": "error", "error_message": "injected simulator failure"}
        done = steps == self.finish
        return observation(value=steps), 0.25, done, False, {"status": self.status if done else "ongoing"}

    def close(self):
        self.closed = True


def config(**overrides):
    kwargs = dict(task_id="BinFill", model_config="unused", output_dir="unused",
                  evaluation_id="fake-identity", n_action_steps=4, max_episode_steps=20)
    kwargs.update(overrides)
    return Config(**kwargs)


class RolloutTests(unittest.TestCase):
    def test_demo_endpoints_exact_cache_schedule(self):
        from gr00t.long_memory.cache import decision_frames
        for n_demo in [0, 1, 3, 16, 17, 38, 48, 86]:
            with self.subTest(n_demo=n_demo):
                indices = demo_endpoints(n_demo, 16)
                cached = decision_frames(np.arange(n_demo + 19) < n_demo, 16)
                self.assertEqual(indices, cached[cached < n_demo].tolist())
                if n_demo:
                    cache = [indices[0]] * 4
                    for index in indices[1:] + [n_demo]:
                        cache = cache[1:] + [index]
                    self.assertEqual(cache, [max(0, n_demo - offset * 16) for offset in [3, 2, 1, 0]])
        self.assertEqual(demo_endpoints(86, 16), [0, 6, 22, 38, 54, 70])

    def test_seed_stable_and_model_independent(self):
        self.assertEqual(episode_inference_seed(6, "BinFill", 1), episode_inference_seed(6, "BinFill", 1))
        self.assertNotEqual(episode_inference_seed(6, "BinFill", 1), episode_inference_seed(6, "BinFill", 2))
        self.assertNotEqual(episode_inference_seed(6, "BinFill", 1), episode_inference_seed(6, "StopCube", 1))

    def test_demo_passive_then_only_executed_actions(self):
        policy, env, journal = FakePolicy(), FakeEnv(n_demo=10), []
        row, frames = run_episode(policy, env, config(), 0, 42, journal.append)
        opts = [call[1] for call in policy.calls]
        self.assertEqual([item["frame_index"] for item in opts], [0, 2, 6, 10, 14])
        self.assertEqual([item["passive"] for item in opts], [True, True, True, False, False])
        self.assertTrue(all(item["executed_actions"] is None for item in opts[:-1]))
        np.testing.assert_array_equal(opts[-1]["executed_actions"], np.stack(env.actions[:4]))
        self.assertEqual(opts[-1]["executed_actions"].shape, (4, 8))
        self.assertEqual([item["reset_memory"] for item in opts], [[True], [False], [False], [False], [False]])
        self.assertEqual(policy.action_noise_draws, 2)
        self.assertEqual(len(env.actions), 5)
        self.assertEqual(row["steps"], 5)
        self.assertEqual(row["success"], 1)
        self.assertEqual(frames, [])
        self.assertEqual(len(policy.resets), 2)
        self.assertEqual(policy.resets[0], policy.resets[1])
        self.assertTrue(env.closed)
        self.assertEqual(env.reset_kwargs, {})  # Never overwrite metadata scenario seed.
        self.assertEqual(journal[-1]["kind"], "episode_complete")

    def test_no_demo_first_call_resets_and_has_no_executed_actions(self):
        policy, env = FakePolicy(), FakeEnv(n_demo=0, finish=1)
        row, _ = run_episode(policy, env, config(), 1, 42, lambda _: None)
        self.assertEqual(len(policy.calls), 1)
        options = policy.calls[0][1]
        self.assertEqual(options["reset_memory"], [True])
        self.assertEqual(options["frame_index"], 0)
        self.assertFalse(options["passive"])
        self.assertIsNone(options["executed_actions"])
        self.assertEqual(row["success"], 1)

    def test_physical_step_limit_not_number_of_chunks(self):
        env = FakeEnv(finish=100)
        row, _ = run_episode(FakePolicy(), env, config(max_episode_steps=6), 0, 42, lambda _: None)
        self.assertEqual(len(env.actions), 6)
        self.assertEqual(row["steps"], 6)
        self.assertEqual(row["status"], "step_limit")
        self.assertEqual(row["success"], 0)

    def test_simulator_error_is_not_zero_success_result(self):
        env, policy, records = FakeEnv(error_step=2), FakePolicy(), []
        with self.assertRaisesRegex(RuntimeError, "simulator error"):
            run_episode(policy, env, config(), 0, 42, records.append)
        self.assertTrue(env.closed)
        self.assertEqual(len(policy.resets), 2)
        self.assertEqual(records[-1]["kind"], "episode_error")
        self.assertFalse(any(item["kind"] == "episode_complete" for item in records))

    def test_policy_error_cleans_both_states(self):
        env, policy = FakeEnv(), FakePolicy(fail_call=1)
        with self.assertRaisesRegex(RuntimeError, "injected policy"):
            run_episode(policy, env, config(), 0, 42, lambda _: None)
        self.assertTrue(env.closed)
        self.assertEqual(len(policy.resets), 2)

    def test_video_flag_controls_capture(self):
        _, frames = run_episode(FakePolicy(), FakeEnv(n_demo=3, finish=2), config(save_videos=True), 0, 42, lambda _: None)
        self.assertEqual(len(frames), 5)
        np.testing.assert_array_equal(frames[0][0, 0], [255, 0, 0])

    def test_stick_gripper_has_canonical_absent_value(self):
        policy, env = FakePolicy(), FakeEnv()
        run_episode(policy, env, config(task_id="PatternLock"), 0, 42, lambda _: None)
        self.assertTrue(all(action[-1] == -1 for action in env.actions))
        self.assertTrue(np.all(policy.calls[-1][1]["executed_actions"][:, -1] == -1))

    def test_action_shape_and_finite_guards(self):
        good = {"action.joint_position": np.zeros((1, 5, 7)), "action.gripper_close": np.zeros((1, 5, 1))}
        self.assertEqual(action_chunk(good, 4).shape, (5, 8))
        with self.assertRaisesRegex(ValueError, "stride requires"):
            action_chunk(good, 6)
        good["action.joint_position"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            action_chunk(good, 4)

    def test_observation_batch_shape(self):
        obs = build_observation(observation(), 0, "goal")
        self.assertEqual(obs["state.joint_position"].shape, (1, 1, 7))
        self.assertEqual(obs["video.front_view"].shape, (1, 1, 8, 8, 3))


class JournalTests(unittest.TestCase):
    def test_atomic_results_resume_without_video(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            identity = {"evaluation_id": "one", "seed": 6}
            check_manifest(out, identity)
            row, _ = run_episode(FakePolicy(), FakeEnv(finish=1), config(), 0, 42, lambda _: None)
            path = out / "simulation_results.csv"
            write_results(path, {0: row})
            self.assertEqual(read_results(path)[0]["success"], "1")
            self.assertFalse(list(out.glob("*.mp4")))
            check_manifest(out, identity)
            with self.assertRaisesRegex(RuntimeError, "identity differs"):
                check_manifest(out, {**identity, "evaluation_id": "two"})
            self.assertFalse(list(out.glob("*.tmp")))

    def test_results_without_identity_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "simulation_results.csv").write_text("episode_idx,success\n0,1\n")
            with self.assertRaisesRegex(RuntimeError, "no identity"):
                check_manifest(out, {"evaluation_id": "one"})

    def test_error_rows_cannot_become_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simulation_results.csv"
            row, _ = run_episode(FakePolicy(), FakeEnv(finish=1), config(), 0, 42, lambda _: None)
            row.update(success=0, status="error")
            write_results(path, {0: row})
            with self.assertRaisesRegex(RuntimeError, "incomplete/error"):
                read_results(path)

    def test_required_checkpoint_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"hamlet_mode": "finetune", "memory_stride": 16, "memory_window": 4}))
            with self.assertRaisesRegex(ValueError, "memory_stride"):
                _resolve_config(config(model_config=str(path), n_action_steps=4))
            model, digest = _resolve_config(config(model_config=str(path), n_action_steps=16))
            self.assertEqual(model["memory_window"], 4)
            self.assertEqual(len(digest), 64)

    def test_cli_save_video_flag(self):
        args = ["--task-id", "BinFill", "--model-config", "base", "--output-dir", "out", "--evaluation-id", "one"]
        self.assertFalse(parse_args(args).save_videos)
        self.assertTrue(parse_args(args + ["--save-videos"]).save_videos)


if __name__ == "__main__":
    unittest.main()
