"""CPU-only synthetic protocol tests; no actual policy/model/server/checkpoint."""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
import copy
import random
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from run_scripts.robomme import policy_demo_tail_ingest_v13 as policy
from run_scripts.robomme.visual_demo_tail_bank_v13 import VisualDemoTailMemoryV13, VisualDifferentialConfig
from tests.test_visual_patch_memory_v11 import inputs


class FakeHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vlln = torch.nn.Identity()
        self.model = torch.nn.Linear(8, 8)
        self._memory_cache = torch.ones(1, 4, 8)
        self._vision_cache = None
        self._inference_gen = None

    def process_backbone_output(self, *args, **kwargs):
        raise AssertionError("No short-memory call during synthetic image ingestion")

    def get_action(self, *args, **kwargs):
        raise AssertionError("No Action Expert during synthetic image ingestion")


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Identity()
        self.action_head = FakeHead()

    @property
    def device(self):
        return torch.device("cpu")


class FakeParent:
    """Only the existing serial/canonical protocol, not a replacement policy."""
    def _new_session(self, seed):
        return SimpleNamespace(generator=torch.Generator().manual_seed(seed), episode_seed=seed,
            bank=None, short_cache=torch.ones(1, 4, 8), raw_states={"joint": np.ones((1, 7), np.float32)},
            frame=None, passive=True, latent=torch.ones(1, 3, 8), observed=0,
            write_attempts=0, updates=0, keeps=0, demo_updates=0, demo_keeps=0,
            visual_bank=self.visual_memory.empty_bank(1), visual_demo_updates=0)

    def _visual_get_action(self, observation, options):
        sid, seed = options["session_ids"][0], options["episode_seed"]
        if options.get("reset_memory") == [True]:
            self.sessions.pop(sid, None)
        if sid not in self.sessions:
            self.sessions[sid] = self._new_session(seed)
        session = self.sessions[sid]
        frame, passive = options["frame_index"], options["passive"]
        current = self.visual_memory.encode_observation(*observation, [frame], [passive], camera_order=policy.CAMERA_ORDER)
        self.events.append((sid, "capture", frame))
        before = session.visual_bank
        result = current.features if passive else self.visual_memory.read(current, before, enabled=not self.visual_read_off)
        self.events.append((sid, "decoded", frame))
        if getattr(self, "fail_decode", False):
            raise RuntimeError("Synthetic decode failure")
        session.visual_bank = self.visual_memory.append(before, current)
        session.observed += 1
        session.updates += 1
        session.write_attempts += 1
        session.demo_updates += int(passive)
        session.visual_demo_updates += int(passive)
        session.frame, session.passive = frame, passive
        session.short_cache = current.features[:, -4:].clone()
        self.events.append((sid, "append_original", frame))
        assert session.observed == session.visual_bank.tokens.shape[1]
        noise = torch.zeros(1) if passive else torch.rand(1, generator=session.generator)
        return {"features": result, "noise": noise}, {"long_memory": {"observed": session.observed, "frame": frame},
            "visual_memory": {"bank_observations": session.observed, "demo_updates": session.visual_demo_updates}}

    @torch.inference_mode()
    def call(self, sid, frame, *, passive=False, seed=17, reset=False):
        if not self._visual_call_lock.acquire(blocking=False):
            raise RuntimeError("Serial parent action lock")
        try:
            return self._visual_get_action(inputs(seed=100 + frame, batch=1, dtype=torch.bfloat16),
                {"session_ids": [sid], "episode_seed": seed, "frame_index": frame,
                 "passive": passive, "prime_only": passive, "reset_memory": [reset]})
        finally:
            self._visual_call_lock.release()

    def reset(self, sid=None):
        if not self._visual_call_lock.acquire(blocking=False):
            raise RuntimeError("Serial parent reset lock")
        try:
            if sid is None:
                self.sessions.clear()
            else:
                self.sessions.pop(sid, None)
        finally:
            self._visual_call_lock.release()


class FakePolicy(policy.DemoTailPolicyMixinV13, FakeParent):
    def __init__(self, *, enabled=True, awake=False, off=False):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(381)
            self.visual_memory = VisualDemoTailMemoryV13(VisualDifferentialConfig(feature_dim=8, hidden_dim=16))
            if awake:
                with torch.no_grad():
                    self.visual_memory.output_projection.weight.normal_(std=.15)
            self.model = FakeModel().eval().requires_grad_(False)
        self.visual_memory.eval().requires_grad_(False)
        self.visual_camera_order = policy.CAMERA_ORDER
        self.visual_read_off = off
        self.processor = object()
        self.embodiment_tag = SimpleNamespace(value="new_embodiment")
        self.sessions = OrderedDict()
        self._visual_call_lock = threading.Lock()
        self.events = []
        self.configure_demo_tail_ingest(enabled=enabled)


def prime(actor, sid="a", n_demo=48, seed=17):
    for frame in policy._demo_frames(n_demo):
        actor.call(sid, frame, passive=True, seed=seed)
    return actor.sessions[sid]


def request(sid="a", n_demo=48, seed=17):
    last = policy._demo_frames(n_demo)[-1] if n_demo else -1
    frames = list(range(max(last + 1, n_demo - 15, 0), n_demo))
    rgb = np.stack([np.full((2, 3, 3), frame, np.uint8) for frame in frames]) if frames else np.empty((0, 2, 3, 3), np.uint8)
    return {"session_id": sid, "episode_seed": seed, "n_demo": n_demo, "frames": frames,
            "images": {camera: rgb.copy() for camera in policy.CAMERA_ORDER}, "texts": ["remember"] * len(frames)}


def fake_extract(backbone, vlln, processor, images, text, *, device, embodiment):
    assert set(images) == set(policy.CAMERA_ORDER) and text == "remember"
    frame = int(images[policy.CAMERA_ORDER[0]][0][0, 0, 0])
    values = torch.arange(2 * 81 * 8).reshape(2, 81, 8).float() / 100 + frame / 17
    return values.bfloat16(), {"synthetic": True}


def ingest(actor, payload=None, extractor=fake_extract):
    with patch.object(policy, "extract_image_features", side_effect=extractor):
        return actor.ingest_demo_tail(**(request() if payload is None else payload))


def rng_state(generator):
    return (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone(), generator.get_state().clone())


def assert_rng(case, first, second):
    case.assertEqual(first[0], second[0])
    case.assertEqual(first[1][0], second[1][0])
    case.assertTrue(np.array_equal(first[1][1], second[1][1]))
    case.assertEqual(first[1][2:], second[1][2:])
    case.assertTrue(torch.equal(first[2], second[2]))
    case.assertTrue(torch.equal(first[3], second[3]))


class DemoTailMixinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_explicit_configuration_and_real_v13_type_required(self):
        actor = FakePolicy()
        with self.assertRaisesRegex(ValueError, "once"):
            actor.configure_demo_tail_ingest(enabled=True)
        with self.assertRaisesRegex(ValueError, "explicit bool"):
            FakePolicy(enabled=1)
        del actor._demo_tail_ingest_enabled
        actor.visual_memory = torch.nn.Identity()
        with self.assertRaisesRegex(ValueError, "actual Visual"):
            actor.configure_demo_tail_ingest(enabled=True)

    def test_atomic_ingest_preserves_all_old_refs_values_rng_and_other_session(self):
        actor = FakePolicy()
        session = prime(actor)
        other = prime(actor, "b", 32, 29)
        snapshots = {key: policy._snapshot(value) for key, value in vars(session).items()}
        head_before = policy._snapshot(actor.model.action_head._memory_cache)
        before = rng_state(session.generator)
        calls = []
        def consuming(*args, **kwargs):
            random.random(); np.random.random(); torch.rand(2); torch.rand(2, generator=session.generator)
            calls.append(int(args[3][policy.CAMERA_ORDER[0]][0][0, 0, 0]))
            return fake_extract(*args, **kwargs)
        with patch.object(actor.visual_memory, "encode_observation", side_effect=AssertionError("No query surrogate")), \
                patch.object(actor.visual_memory.query_projection, "forward", side_effect=AssertionError("No query projection")):
            result = ingest(actor, extractor=consuming)
        self.assertEqual(calls, list(range(33, 48)))
        self.assertEqual(result["ingested_observations"], 15)
        self.assertFalse(result["read_performed"])
        self.assertEqual(session.visual_tail_bank.frames[0].tolist(), calls)
        self.assertEqual(list(actor.sessions), ["a", "b"])
        self.assertIs(actor.sessions["b"], other)
        for key, snapshot in snapshots.items():
            self.assertTrue(policy._unchanged(getattr(session, key), snapshot), key)
        self.assertTrue(policy._unchanged(actor.model.action_head._memory_cache, head_before))
        assert_rng(self, before, rng_state(session.generator))
        self.assertEqual(set(vars(session)) - set(snapshots), {"visual_tail_bank", "visual_tail_metadata"})

    def test_zero_and_visual_off_actions_noise_and_canonical_counts_exact(self):
        for awake, off in ((False, False), (True, True)):
            actor, reference = FakePolicy(awake=awake, off=off), FakePolicy(enabled=False, awake=awake, off=off)
            prime(actor); prime(reference)
            ingest(actor)
            actual, info = actor.call("a", 48)
            expected, baseline = reference.call("a", 48)
            for key in actual:
                self.assertTrue(torch.equal(actual[key], expected[key]), key)
            self.assertEqual(info["long_memory"], baseline["long_memory"])
            self.assertEqual(info["visual_memory"], baseline["visual_memory"])
            self.assertEqual(actor.sessions["a"].observed, 4)
            self.assertEqual(actor.sessions["a"].visual_bank.tokens.shape[1], 4)
            self.assertEqual(actor.sessions["a"].visual_tail_bank.tokens.shape[1], 15)
            self.assertEqual(info["demo_tail"]["read_enabled"], not off)
            self.assertEqual(info["demo_tail"]["effective_prior_observations"], 3 if off else 18)

    def test_warmed_read_changes_only_read_result_not_stored_current(self):
        actor, reference = FakePolicy(awake=True), FakePolicy(enabled=False, awake=True)
        prime(actor); prime(reference)
        ingest(actor)
        actual, _ = actor.call("a", 48)
        expected, _ = reference.call("a", 48)
        self.assertFalse(torch.equal(actual["features"], expected["features"]))
        for name in ("tokens", "content", "frames", "is_demo", "valid"):
            self.assertTrue(torch.equal(getattr(actor.sessions["a"].visual_bank, name),
                                        getattr(reference.sessions["a"].visual_bank, name)), name)
        self.assertEqual(actor.events[-3:], [("a", "capture", 48), ("a", "decoded", 48), ("a", "append_original", 48)])

    def test_two_sessions_read_merge_order_and_descriptor_restoration(self):
        actor = FakePolicy(awake=True)
        prime(actor); prime(actor, "b", 32, 29)
        ingest(actor); ingest(actor, request("b", 32, 29))
        actual_read = actor.visual_memory.read
        received = []
        def spy(current, bank=None, *, enabled=True):
            received.append(bank.frames[0].tolist())
            return actual_read(current, bank, enabled=enabled)
        actor.visual_memory.read = spy
        actor.call("a", 48)
        self.assertIs(actor.visual_memory.read, spy)
        actor.call("b", 32, seed=29)
        self.assertIs(actor.visual_memory.read, spy)
        self.assertEqual(received[0], [0, 16, 32] + list(range(33, 48)))
        self.assertEqual(received[1], [0, 16] + list(range(17, 32)))
        actor.call("a", 64)
        self.assertEqual(received[2], [0, 16, 32] + list(range(33, 49)))

    def test_duplicate_fails_invalidates_only_exact_session(self):
        actor = FakePolicy()
        original = prime(actor)
        other = prime(actor, "b", 32, 29)
        ingest(actor)
        with self.assertRaisesRegex(ValueError, "already"):
            ingest(actor)
        self.assertNotIn("a", actor.sessions)
        self.assertIs(actor.sessions["b"], other)
        self.assertEqual(original.visual_bank.tokens.shape[1], 3)

    def test_unknown_wrong_seed_and_busy_lock_cannot_evict_sessions(self):
        actor = FakePolicy()
        original = prime(actor)
        for payload in (request("unknown"), request(seed=999)):
            with self.assertRaisesRegex(ValueError, "Unknown session"):
                ingest(actor, payload)
            self.assertIs(actor.sessions["a"], original)
        actor._visual_call_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "serial"):
                ingest(actor)
            self.assertIs(actor.sessions["a"], original)
        finally:
            actor._visual_call_lock.release()

    def test_parent_numpy_integer_seed_contract_is_preserved(self):
        actor = FakePolicy(); prime(actor)
        result = ingest(actor, request(seed=np.int64(17)))
        self.assertIs(type(result["episode_seed"]), int)
        _, info = actor.call("a", 48, seed=np.int64(17))
        self.assertTrue(info["demo_tail"]["read_enabled"])

    def test_bounds_current_future_wrong_shapes_and_gt_rejected(self):
        changes = ({"frames": list(range(34, 49))}, {"frames": list(range(33, 47))},
                   {"frames": list(reversed(range(33, 48)))}, {"n_demo": 49},
                   {"images": {"front_view": np.zeros((15, 2, 3, 3), np.uint8)}},
                   {"actions": np.zeros((15, 8))}, {"texts": ["remember"]}, {"n_demo": 0})
        for change in changes:
            actor = FakePolicy(); prime(actor)
            other = prime(actor, "b", 32, 29)
            with self.assertRaises(ValueError):
                ingest(actor, {**request(), **change})
            self.assertNotIn("a", actor.sessions)
            self.assertIs(actor.sessions["b"], other)
            self.assertFalse(actor._visual_call_lock.locked())

    def test_short_demo_and_empty_client_skip_contract(self):
        actor = FakePolicy(); prime(actor, n_demo=3)
        result = ingest(actor, request(n_demo=3))
        self.assertEqual(result["ingested_observations"], 2)
        self.assertEqual(actor.call("a", 3)[1]["demo_tail"]["effective_prior_observations"], 3)
        actor = FakePolicy()
        actual, info = actor.call("a", 0, reset=True)
        self.assertFalse(info["demo_tail"]["read_enabled"])
        self.assertFalse(hasattr(actor.sessions["a"], "visual_tail_bank"))

    def test_error_rollback_old_refs_values_rng_head_methods_and_session(self):
        actor = FakePolicy(); session = prime(actor)
        other = prime(actor, "b", 32, 29)
        snapshots = {key: policy._snapshot(value) for key, value in vars(session).items()}
        head = actor.model.action_head
        head_snapshot = policy._snapshot(head._memory_cache)
        methods = dict(head.__dict__)
        before = rng_state(session.generator)
        def broken(*args, **kwargs):
            random.random(); np.random.random(); torch.rand(3); torch.rand(3, generator=session.generator)
            session.latent.add_(9); session.raw_states["joint"] += 7; session.observed += 4
            head._memory_cache.add_(2)
            raise RuntimeError("Synthetic extraction failure")
        with self.assertRaisesRegex(RuntimeError, "extraction failure"):
            ingest(actor, extractor=broken)
        self.assertNotIn("a", actor.sessions)
        self.assertIs(actor.sessions["b"], other)
        for key, snapshot in snapshots.items():
            self.assertTrue(policy._unchanged(getattr(session, key), snapshot), key)
        self.assertTrue(policy._unchanged(head._memory_cache, head_snapshot))
        self.assertEqual(set(methods), set(head.__dict__))
        assert_rng(self, before, rng_state(session.generator))
        self.assertFalse(actor._visual_call_lock.locked())

    def test_no_ae_or_short_memory_calls_and_no_commit_if_rng_restore_fails(self):
        for invoke in (lambda head: head.get_action(), lambda head: head.process_backbone_output(),
                       lambda head: head.model(torch.ones(1, 8))):
            actor = FakePolicy(); session = prime(actor)
            def forbidden(*args, **kwargs):
                invoke(actor.model.action_head)
            with self.assertRaisesRegex(RuntimeError, "cannot call"):
                ingest(actor, extractor=forbidden)
            self.assertFalse(hasattr(session, "visual_tail_bank"))
            self.assertNotIn("a", actor.sessions)
        actor = FakePolicy(); session = prime(actor)
        @contextmanager
        def failing_restore(*args):
            yield
            raise RuntimeError("Synthetic RNG restore failure")
        with patch.object(policy, "isolated_visual_ingest_rng", failing_restore):
            with self.assertRaisesRegex(RuntimeError, "RNG restore failure"):
                ingest(actor)
        self.assertFalse(hasattr(session, "visual_tail_bank"))
        self.assertNotIn("a", actor.sessions)

    def test_failure_restores_session_table_identity_membership_and_order(self):
        for replace_table, raises in ((False, False), (False, True), (True, True)):
            actor = FakePolicy(); session = prime(actor)
            b, c = prime(actor, "b", 32, 29), prime(actor, "c", 32, 41)
            original_table = actor.sessions
            def corrupted(*args, **kwargs):
                if replace_table:
                    actor.sessions = OrderedDict()
                else:
                    actor.sessions.pop("b", None)
                    actor.sessions.move_to_end("a")
                actor.sessions["spurious"] = object()
                if raises:
                    raise RuntimeError("Synthetic table mutation failure")
                return fake_extract(*args, **kwargs)
            with self.assertRaises((ValueError, RuntimeError)):
                ingest(actor, extractor=corrupted)
            self.assertIs(actor.sessions, original_table)
            self.assertEqual(list(actor.sessions), ["b", "c"])
            self.assertIs(actor.sessions["b"], b)
            self.assertIs(actor.sessions["c"], c)
            self.assertFalse(hasattr(session, "visual_tail_bank"))
            self.assertFalse(actor._visual_call_lock.locked())

    def test_input_mutation_rejected_and_original_rgb_restored(self):
        actor = FakePolicy(); session = prime(actor)
        payload = request()
        saved = {camera: policy._snapshot(rgb) for camera, rgb in payload["images"].items()}
        def mutated(*args, **kwargs):
            features = fake_extract(*args, **kwargs)
            args[3][policy.CAMERA_ORDER[0]][0][:] = 0
            return features
        with self.assertRaisesRegex(ValueError, "Raw RGB input changed"):
            ingest(actor, payload, extractor=mutated)
        self.assertTrue(all(policy._unchanged(payload["images"][camera], snapshot)
                            for camera, snapshot in saved.items()))
        self.assertFalse(hasattr(session, "visual_tail_bank"))
        self.assertNotIn("a", actor.sessions)

    def test_bad_tail_read_and_decode_failure_restore_read_method(self):
        for corrupt in (True, False):
            actor = FakePolicy(); session = prime(actor); other = prime(actor, "b", 32, 29)
            ingest(actor)
            self.assertNotIn("read", actor.visual_memory.__dict__)
            if corrupt:
                bad = session.visual_tail_bank.frames.clone(); bad[0, -1] = 48
                session.visual_tail_bank = replace(session.visual_tail_bank, frames=bad)
            else:
                actor.fail_decode = True
            with self.assertRaises((ValueError, RuntimeError)):
                actor.call("a", 48)
            self.assertNotIn("read", actor.visual_memory.__dict__)
            self.assertNotIn("a", actor.sessions)
            self.assertIs(actor.sessions["b"], other)

    def test_reset_clears_tail_and_hook_binds_new_session_not_old_reference(self):
        actor = FakePolicy(); old = prime(actor); ingest(actor)
        _, info = actor.call("a", 0, reset=True)
        self.assertIsNot(actor.sessions["a"], old)
        self.assertFalse(hasattr(actor.sessions["a"], "visual_tail_bank"))
        self.assertFalse(info["demo_tail"]["read_enabled"])
        actor.reset("a")
        self.assertNotIn("a", actor.sessions)
        prime(actor); ingest(actor); actor.reset()
        self.assertFalse(actor.sessions)

    def test_post_ingest_priming_and_wrong_first_execution_invalidates(self):
        for frame, passive in ((40, True), (47, False)):
            actor = FakePolicy(); prime(actor); ingest(actor)
            with self.assertRaises(ValueError):
                actor.call("a", frame, passive=passive)
            self.assertNotIn("a", actor.sessions)

    def test_enabled_nonempty_tail_missing_rpc_fails_closed_even_read_off(self):
        for off in (False, True):
            actor = FakePolicy(off=off); prime(actor)
            other = prime(actor, "b", 32, 29)
            with self.assertRaisesRegex(ValueError, "Missing required demo-tail ingest"):
                actor.call("a", 48)
            self.assertNotIn("a", actor.sessions)
            self.assertIs(actor.sessions["b"], other)
            self.assertNotIn("read", actor.visual_memory.__dict__)
        # Canonical-only arm, genuinely empty tail, and reset/no-demo actions
        # retain the unchanged parent's ordinary path without this RPC.
        actor = FakePolicy(enabled=False); prime(actor)
        self.assertFalse(actor.call("a", 48)[1]["demo_tail"]["read_enabled"])
        actor = FakePolicy(); prime(actor, n_demo=1)
        self.assertFalse(actor.call("a", 1)[1]["demo_tail"]["read_enabled"])
        actor = FakePolicy(); prime(actor)
        self.assertFalse(actor.call("a", 0, reset=True)[1]["demo_tail"]["read_enabled"])

    def test_first_execution_checks_exact_canonical_demo_prefix(self):
        actor = FakePolicy(); prime(actor)
        with self.assertRaisesRegex(ValueError, "canonical demo prime prefix"):
            actor.call("a", 49)
        self.assertNotIn("a", actor.sessions)

    def test_pure_merge_validation_and_inputs_unchanged(self):
        actor = FakePolicy(); session = prime(actor); ingest(actor)
        canonical, tail = session.visual_bank, session.visual_tail_bank
        snapshots = [policy._snapshot(value) for value in (canonical, tail)]
        kwargs = dict(current_frame=48, n_demo=48, hidden_dim=16, device="cpu")
        merged = policy.merge_demo_tail_banks(canonical, tail, **kwargs)
        self.assertEqual(merged.frames.shape, (1, 18))
        self.assertTrue(all(policy._unchanged(value, snap) for value, snap in zip((canonical, tail), snapshots)))
        for bad in (replace(tail, content=tail.content.double()),
                    replace(tail, tokens=torch.full_like(tail.tokens, float("nan"))),
                    replace(tail, valid=torch.zeros_like(tail.valid)),
                    replace(tail, is_demo=torch.zeros_like(tail.is_demo)),
                    replace(tail, frames=tail.frames.flip(1))):
            with self.assertRaises(ValueError):
                policy.merge_demo_tail_banks(canonical, bad, **kwargs)
        empty = actor.visual_memory.empty_bank(1)
        self.assertIs(policy.merge_demo_tail_banks(empty, empty, current_frame=0, n_demo=0,
                                                  hidden_dim=16, device="cpu"), empty)

    def test_unchanged_v12_v11_v7_method_chain_with_tiny_cpu_expert(self):
        # No production constructor, checkpoint, VLM or AE is loaded. Reuse
        # existing tiny fixtures but execute the actual inherited methods.
        from run_scripts.robomme.policy_visual_differential_v12 import VisualDifferentialV12Policy
        from tests import test_policy_visual_patch_v11 as legacy

        class SyntheticComposition(policy.DemoTailPolicyMixinV13, VisualDifferentialV12Policy):
            pass

        def create(*, enabled, awake, off):
            actor = SyntheticComposition.__new__(SyntheticComposition)
            actor.__dict__.update(legacy.parent_policy().__dict__)
            # The existing fixture's CPUModel is deliberately not nn.Module;
            # expose its actual only parameter-owning tiny head for freeze checks.
            actor.model.training = False
            actor.model.parameters = actor.model.action_head.parameters
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(381)
                actor.visual_memory = VisualDemoTailMemoryV13(
                    VisualDifferentialConfig(feature_dim=8, hidden_dim=16, time_scale=2.))
                if awake:
                    with torch.no_grad():
                        actor.visual_memory.output_projection.weight.normal_(std=.15)
            actor.visual_memory.eval().requires_grad_(False)
            actor.visual_camera_order = policy.CAMERA_ORDER
            actor.visual_read_off, actor.visual_read_mode = off, "differential"
            actor.checkpoint_variant, actor.checkpoint_step = "synthetic_mixin_test_not_checkpoint", 0
            actor.visual_weights_sha256 = actor.frozen_parent_checkpoint_sha256 = "synthetic"
            actor.stride = 16
            actor._visual_call_lock = threading.Lock()
            actor.configure_demo_tail_ingest(enabled=enabled)
            return actor

        for awake, off in ((False, False), (True, True), (True, False)):
            actor, baseline = [create(enabled=enabled, awake=awake, off=off) for enabled in (True, False)]
            for frame in (0, 16, 32):
                for instance in (actor, baseline):
                    legacy.VisualPatchPolicyTests.call(self, instance, marker=1 + frame, frame=frame, passive=True)
            original_bank = actor.sessions["A"].visual_bank
            ingest(actor, request("A"))
            self.assertIs(actor.sessions["A"].visual_bank, original_bank)
            actual, info = legacy.VisualPatchPolicyTests.call(self, actor, marker=4, frame=48)
            expected, reference = legacy.VisualPatchPolicyTests.call(self, baseline, marker=4, frame=48)
            self.assertEqual(info["long_memory"], reference["long_memory"])
            self.assertEqual(info["visual_memory"]["bank_observations"], 4)
            self.assertEqual(actor.sessions["A"].observed, 4)
            self.assertEqual(actor.sessions["A"].visual_bank.tokens.shape[1], 4)
            self.assertEqual(info["demo_tail"]["effective_prior_observations"], 3 if off else 18)
            for key in ("latent", "short_cache"):
                self.assertTrue(torch.equal(getattr(actor.sessions["A"], key), getattr(baseline.sessions["A"], key)))
            self.assertTrue(torch.equal(actor.sessions["A"].generator.get_state(), baseline.sessions["A"].generator.get_state()))
            a, b = actor.model.action_head, baseline.model.action_head
            image = legacy.CPUModel.backbone({"marker": 4})["image_mask"]
            self.assertTrue(torch.equal(a.last_features[~image], b.last_features[~image]))
            self.assertTrue(torch.equal(a.last_features[:, -4:], b.last_features[:, -4:]))
            self.assertFalse(torch.equal(a.last_features[:, -4:], a.last_processed[:, -4:]))
            if not awake or off:
                legacy.VisualPatchPolicyTests.assert_actions_equal(self, actual, expected)
            else:
                self.assertFalse(torch.equal(a.last_features[image], b.last_features[image]))
            for name in ("tokens", "content", "frames", "is_demo", "valid"):
                self.assertTrue(torch.equal(getattr(actor.sessions["A"].visual_bank, name),
                                            getattr(baseline.sessions["A"].visual_bank, name)))
            self.assertNotIn("read", actor.visual_memory.__dict__)
            self.assertFalse(actor.visual_memory.output_projection._forward_hooks)


if __name__ == "__main__":
    unittest.main()
