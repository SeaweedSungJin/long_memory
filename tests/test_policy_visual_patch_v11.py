"""CPU serial policy/server wiring with the real V7/visual memories and tiny AE.

No real VLM, Action Expert, simulator, sockets, or CUDA are used. The parent
constructor and V11 file loading are mocked; checkpoint validation has its own
tests. Here the inherited V7 action/session implementation remains live.
"""
import contextlib
import copy
from dataclasses import asdict
import io
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn

from gr00t.long_memory.cvom_v7 import CVOMV7
from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
from gr00t.long_memory.recurrent_v7 import MemoryV7Config, RecurrentMemoryV7
from run_scripts.robomme import policy_visual_patch_v11 as policy_module
from run_scripts.robomme import serve_visual_patch_v11 as server
from run_scripts.robomme.visual_patch_memory_v11 import CAMERA_ORDER, VisualPatchConfig, VisualPatchMemoryV11
from tests import test_long_memory_online_policy as fixture


VISUAL_CONFIG = VisualPatchConfig(feature_dim=8, hidden_dim=16, time_scale=2.)
PARENT_CONFIG = MemoryV7Config(feature_dim=8, state_dim=8, num_short_tokens=4,
    hidden_dim=16, num_heads=4, capacity=3, time_scale=2.)


class CPUHead(nn.Module, fixture._Head):
    def __init__(self):
        nn.Module.__init__(self)
        fixture._Head.__init__(self)
        self.probe = nn.Linear(8, 8, bias=False)
        self.original_features = []

    @staticmethod
    def vlln(features):
        return features * 1.3 + .2

    def process_backbone_output(self, backbone, action_inputs_B):
        self.process_calls += 1
        assert action_inputs_B == 1
        features = self.vlln(backbone["backbone_features"])
        moment = features[:, -4:]
        self._memory_cache = (moment.repeat(1, 3, 1) if self._memory_cache is None else
                              torch.cat((self._memory_cache[:, 4:], moment), dim=1))
        short = self._memory_cache.reshape(1, 3, 4, 8).mean(1)
        backbone["backbone_features"] = torch.cat((features[:, :-4], short), dim=1)
        self.last_processed = backbone["backbone_features"].clone()
        self.original_features.append(self.last_processed)
        return backbone

    def get_action_with_features(self, features, state_features, embodiment, backbone):
        result = fixture._Head.get_action_with_features(self, features, state_features, embodiment, backbone)
        result["action_pred"] = result["action_pred"].float() + self.probe(features.float()).mean() * .2
        return result


class CPUModel(fixture._Model):
    def __init__(self):
        self.action_head = CPUHead().eval().requires_grad_(False)

    @staticmethod
    def backbone(batch):
        length = 209 if int(batch["marker"]) % 2 else 214
        features = (torch.sin(torch.arange(length * 8).reshape(1, length, 8) / 13)
                    + batch["marker"] * .1).bfloat16()
        image = torch.zeros(1, length, dtype=torch.bool)
        start = length - 176
        image[:, start:start + 81] = True
        image[:, start + 88:start + 169] = True
        return {"backbone_features": features, "image_mask": image,
                "backbone_attention_mask": torch.ones_like(image)}


def parent_policy():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(731)
        policy = LongMemoryV7Policy.__new__(LongMemoryV7Policy)
        policy.__dict__.update(fixture._policy().__dict__)
        policy.model = CPUModel()
        policy.memory = RecurrentMemoryV7(PARENT_CONFIG)
        policy.cvom = CVOMV7(PARENT_CONFIG)
        with torch.no_grad():
            policy.memory.fusion_projection.weight.normal_(std=.4)
        policy.memory.eval().requires_grad_(False)
        policy.cvom.eval().requires_grad_(False)
        policy.n_q = 4
        policy.mode, policy.stage, policy.write_policy = "archive", 1, "append"
        policy.memory_off, policy.write_policy_override, policy.cvom_threshold = False, "checkpoint", .05
    return policy


def checkpoint_info(step=32):
    return {"step": step, "config": {"trainer_variant": "visual_patch_v11", "stage": 1,
        "mode": "visual_patch", "visual": asdict(VISUAL_CONFIG), "camera_order": list(CAMERA_ORDER)},
        "metadata": {"frozen_parent": {"path": "parent-checkpoint", "step": 1250,
            "files_sha256": {"checkpoint.json": "b" * 64}},
            "payload_sha256": {"visual.safetensors": "a" * 64}}}


class VisualPatchPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def policy(self, *, awake=True, off=False, step=32, parent=None):
        info, events = checkpoint_info(step), []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(911)
            visual = VisualPatchMemoryV11(VISUAL_CONFIG)
            if awake:
                with torch.no_grad():
                    visual.output_projection.weight.normal_(std=.05)
        state = copy.deepcopy(visual.state_dict())

        def preflight(base, checkpoint, expected_stage):
            self.assertEqual((base, checkpoint, expected_stage), ("base", "visual-checkpoint", 1))
            events.append("validated")
            return copy.deepcopy(info)

        def initialize(policy, base, checkpoint, *, device, strict, write_policy, memory_off):
            self.assertEqual(events, ["validated"])
            self.assertEqual((base, checkpoint, device, strict, write_policy, memory_off),
                             ("base", "parent-checkpoint", "cpu", True, "checkpoint", False))
            policy.__dict__.update((parent or parent_policy()).__dict__)
            events.append("parent_loaded")

        def load(checkpoint, module):
            self.assertEqual(checkpoint, "visual-checkpoint")
            self.assertEqual(events, ["validated", "parent_loaded"])
            module.load_state_dict(state)
            events.append("visual_loaded")
            return copy.deepcopy(info)

        with patch.object(policy_module, "checkpoint_info", side_effect=preflight), \
                patch.object(LongMemoryV7Policy, "__init__", initialize), \
                patch.object(policy_module, "load_checkpoint", side_effect=load):
            policy = policy_module.VisualPatchV11Policy("base", "visual-checkpoint", device="cpu", visual_read_off=off)
        self.assertEqual(events, ["validated", "parent_loaded", "visual_loaded"])
        return policy

    def call(self, policy, marker=1, **options):
        return policy.get_action(fixture._observation(marker=marker), fixture._options(**options))

    def assert_actions_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_preflight_invalid_variants_options_and_corruption_fail_before_parent(self):
        for change in ({"stage": 2}, {"mode": "archive"}, {"trainer_variant": "archive_projector_v10"}):
            info = checkpoint_info(); info["config"].update(change)
            with patch.object(policy_module, "checkpoint_info", return_value=info), \
                    patch.object(LongMemoryV7Policy, "__init__") as base, self.assertRaises(ValueError):
                policy_module.VisualPatchV11Policy("base", "bundle", device="cpu")
            base.assert_not_called()
        with patch.object(policy_module, "checkpoint_info", side_effect=ValueError("changed external parent")), \
                patch.object(LongMemoryV7Policy, "__init__") as base, self.assertRaisesRegex(ValueError, "external parent"):
            policy_module.VisualPatchV11Policy("base", "bundle", device="cpu")
        base.assert_not_called()
        for kwargs in ({"visual_read_off": "false"}, {"strict": 1}):
            with self.assertRaises(TypeError):
                policy_module.VisualPatchV11Policy("base", "bundle", **kwargs)
        with self.assertRaises(ValueError):
            policy_module.VisualPatchV11Policy("base", None)
        with self.assertRaises(ValueError):
            policy_module.VisualPatchV11Policy("base", "bundle", write_policy="update")

    def test_constructor_freezes_parent_and_visual_and_isolates_initialization_rng(self):
        before = torch.get_rng_state().clone()
        policy = self.policy()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual((policy.stage, policy.mode, policy.write_policy, policy.memory_off), (1, "archive", "append", False))
        for module in (policy.memory, policy.cvom, policy.model.action_head, policy.visual_memory):
            self.assertFalse(module.training)
            self.assertFalse(any(p.requires_grad for p in module.parameters()))
        reversed_parent = parent_policy()
        reversed_parent.modality_configs["video"].modality_keys.reverse()
        with self.assertRaisesRegex(ValueError, "camera order"):
            self.policy(parent=reversed_parent)

    def test_constructor_rejects_changed_identity_at_actual_load(self):
        info, changed = checkpoint_info(), checkpoint_info(step=64)

        def initialize(policy, *args, **kwargs):
            policy.__dict__.update(parent_policy().__dict__)

        with patch.object(policy_module, "checkpoint_info", return_value=info), \
                patch.object(LongMemoryV7Policy, "__init__", initialize), \
                patch.object(policy_module, "load_checkpoint", return_value=changed), \
                self.assertRaisesRegex(ValueError, "identity changed"):
            policy_module.VisualPatchV11Policy("base", "bundle", device="cpu")

    def test_zero_and_visual_off_match_parent_actions_archive_short_and_session_rng(self):
        parent, zero, off = parent_policy(), self.policy(awake=False, step=0), self.policy(off=True)
        global_rng = torch.get_rng_state().clone()
        initial_noise = torch.Generator().manual_seed(17).get_state()
        for index, (frame, passive) in enumerate(((0, True), (2, True), (4, False), (6, False))):
            controls = np.zeros((2, 8), np.float32) if index == 3 else None
            results = [self.call(policy, marker=index + 1, frame=frame, passive=passive, controls=controls)
                       for policy in (parent, zero, off)]
            for policy, (actions, info) in zip((zero, off), results[1:]):
                self.assert_actions_equal(actions, results[0][0])
                self.assertEqual(info["long_memory"], results[0][1]["long_memory"])
                a, b = parent.sessions["A"], policy.sessions["A"]
                self.assertTrue(torch.equal(a.latent, b.latent))
                self.assertTrue(torch.equal(a.short_cache, b.short_cache))
                self.assertTrue(torch.equal(a.generator.get_state(), b.generator.get_state()))
                if passive:
                    self.assertTrue(torch.equal(initial_noise, b.generator.get_state()))
                self.assertEqual(b.visual_bank.tokens.shape, (1, index + 1, 162, 16))
                self.assertEqual(info["visual_memory"]["image_delta_norm"], 0.)
                self.assertEqual(info["visual_memory"]["bank_tokens"], (index + 1) * 162)
                self.assertEqual(info["visual_memory"]["demo_updates"], min(index + 1, 2))
                self.assertEqual(info["visual_memory"]["read_enabled"], policy is zero and not passive)
            self.assertEqual(results[1][1]["checkpoint_step"], 0)
            self.assertEqual(results[2][1]["visual_weights_sha256"], "a" * 64)
            self.assertEqual(results[2][1]["frozen_parent_checkpoint_sha256"], "b" * 64)
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        self.assertEqual(off.model.action_head.denoiser_calls, 2)
        self.assertTrue(results[2][1]["long_memory"]["memory_read_enabled"])

    def test_warmed_read_changes_images_only_preserves_archive_tail_and_stores_original(self):
        on, off = self.policy(), self.policy(off=True)
        for policy in (on, off):
            self.call(policy, passive=True)
        before = on.sessions["A"].visual_bank.tokens.clone()
        with patch.object(on.visual_memory, "read", wraps=on.visual_memory.read) as reader, \
                patch.object(on.visual_memory, "append", wraps=on.visual_memory.append) as writer:
            actions_on, info = self.call(on, marker=2, frame=2)
        actions_off, _ = self.call(off, marker=2, frame=2)
        self.assertEqual(reader.call_args.args[1].tokens.shape[1], 1)
        self.assertEqual(writer.call_args.args[0].tokens.shape[1], 1)
        current = writer.call_args.args[1]
        head = on.model.action_head
        other = off.model.action_head
        image = CPUModel.backbone({"marker": 2})["image_mask"]
        self.assertTrue(torch.equal(head.last_features[~image], other.last_features[~image]))
        self.assertFalse(torch.equal(head.last_features[image], other.last_features[image]))
        self.assertTrue(torch.equal(head.last_features[:, -4:], other.last_features[:, -4:]))
        self.assertFalse(torch.equal(head.last_features[:, -4:], head.last_processed[:, -4:]))
        self.assertTrue(torch.equal(current.features, head.last_processed))
        self.assertTrue(torch.equal(on.sessions["A"].visual_bank.tokens[:, -1], current.encoded))
        self.assertTrue(torch.equal(on.sessions["A"].visual_bank.tokens[:, :1], before))
        self.assertTrue(torch.equal(on.sessions["A"].visual_bank.tokens, off.sessions["A"].visual_bank.tokens))
        self.assertGreater(info["visual_memory"]["image_delta_norm"], 0)
        self.assertGreater(info["visual_memory"]["image_changed_fraction"], 0)
        self.assertFalse(np.array_equal(actions_on["joint_position"], actions_off["joint_position"]))

    def test_passive_appends_without_visual_read_action_or_noise_and_append_follows_decode(self):
        policy = self.policy()
        events = []
        decode, append = policy.processor.decode_action, policy.visual_memory.append

        def decoded(*args):
            events.append("decode")
            return decode(*args)

        def appended(*args, **kwargs):
            events.append("append")
            return append(*args, **kwargs)

        with patch.object(policy.processor, "decode_action", side_effect=decoded), \
                patch.object(policy.visual_memory, "append", side_effect=appended), \
                patch.object(policy.visual_memory, "read", side_effect=AssertionError("No passive READ")):
            for frame in (0, 2):
                _, info = self.call(policy, frame=frame, passive=True)
                self.assertFalse(info["visual_memory"]["read_enabled"])
        self.assertEqual(events, ["decode", "append", "decode", "append"])
        self.assertEqual(policy.model.action_head.denoiser_calls, 0)
        self.assertTrue(torch.equal(policy.sessions["A"].generator.get_state(), torch.Generator().manual_seed(17).get_state()))

    def test_interleaved_sessions_and_resets_isolate_banks_rng_and_short_cache(self):
        isolated, mixed = self.policy(), self.policy()
        for policy in (isolated, mixed):
            self.call(policy, passive=True)
        first, _ = self.call(isolated, marker=2, frame=2)
        saved = mixed.sessions["A"].visual_bank.tokens.clone()
        self.call(mixed, marker=99, session="B", seed=83, passive=True)
        self.assertTrue(torch.equal(mixed.sessions["A"].visual_bank.tokens, saved))
        second, _ = self.call(mixed, marker=2, frame=2)
        self.assert_actions_equal(first, second)
        self.assertTrue(torch.equal(isolated.sessions["A"].generator.get_state(), mixed.sessions["A"].generator.get_state()))
        self.assertTrue(torch.equal(isolated.sessions["A"].short_cache, mixed.sessions["A"].short_cache))
        bank_b = mixed.sessions["B"].visual_bank
        self.assertEqual(mixed.reset({"session_ids": ["A"]}), {"cleared_sessions": 1})
        self.assertIs(mixed.sessions["B"].visual_bank, bank_b)
        _, info = self.call(mixed, passive=True)
        self.assertEqual(info["visual_memory"]["observed"], 1)
        self.assertEqual(mixed.reset(), {"cleared_sessions": 2})
        self.assertFalse(mixed.sessions)

    def test_forward_decode_append_failures_clear_session_and_restore_exact_methods(self):
        for failure in ("forward", "decode", "append", "interrupt"):
            policy = self.policy()
            head = policy.model.action_head
            self.call(policy, passive=True)
            # Preserve preexisting instance method overrides as well as class methods.
            head.process_backbone_output = head.process_backbone_output
            head.get_action_with_features = head.get_action_with_features
            saved = {name: head.__dict__[name] for name in ("process_backbone_output", "get_action_with_features")}
            with contextlib.ExitStack() as stack:
                if failure == "forward":
                    head.fail_prediction = True
                elif failure == "decode":
                    stack.enter_context(patch.object(policy.processor, "decode_action", side_effect=RuntimeError("decode failed")))
                elif failure == "interrupt":
                    stack.enter_context(patch.object(policy.processor, "decode_action", side_effect=KeyboardInterrupt()))
                else:
                    stack.enter_context(patch.object(policy.visual_memory, "append", side_effect=RuntimeError("append failed")))
                with self.assertRaises((RuntimeError, FloatingPointError, KeyboardInterrupt)):
                    self.call(policy, marker=2, frame=2)
            self.assertNotIn("A", policy.sessions)
            self.assertFalse(policy._visual_call_lock.locked())
            self.assertIsNone(head._memory_cache)
            self.assertIsNone(head._inference_gen)
            for name, value in saved.items():
                self.assertIs(head.__dict__[name], value)

    def test_success_restores_descriptors_and_reentrant_action_reset_are_rejected(self):
        policy = self.policy()
        head = policy.model.action_head
        names = ("process_backbone_output", "get_action_with_features")
        self.assertTrue(all(name not in head.__dict__ for name in names))
        original = head.get_action_with_features

        def reentrant(*args):
            with self.assertRaisesRegex(RuntimeError, "serial"):
                self.call(policy, session="B", seed=22)
            with self.assertRaisesRegex(RuntimeError, "overlap"):
                policy.reset()
            return original(*args)

        with patch.object(head, "get_action_with_features", side_effect=reentrant):
            self.call(policy)
        self.assertTrue(all(name not in head.__dict__ for name in names))
        self.assertEqual(list(policy.sessions), ["A"])
        self.assertFalse(policy._visual_call_lock.locked())

    def test_changed_nonshort_conditioning_is_rejected_and_no_visual_append_occurs(self):
        policy = self.policy()
        # Stronger direct parent READ corruption outside the tail is difficult
        # without rewriting V7. Instead alter the captured original clone at
        # encoding time, which the incoming-feature invariant must reject.
        encode = policy.visual_memory.encode_observation

        def corrupt(*args, **kwargs):
            current = encode(*args, **kwargs)
            current.features[:, 0] += 1
            return current

        with patch.object(policy.visual_memory, "encode_observation", side_effect=corrupt), \
                patch.object(policy.visual_memory, "append", wraps=policy.visual_memory.append) as writer:
            with self.assertRaisesRegex(ValueError, "outside"):
                self.call(policy)
        writer.assert_not_called()
        self.assertNotIn("A", policy.sessions)

    def test_changed_backbone_masks_are_rejected_before_expert_or_visual_write(self):
        policy = self.policy()
        head, captured = policy.model.action_head, {}
        process, state_encoder = head.process_backbone_output, head.state_encoder

        def remember_backbone(*args, **kwargs):
            backbone = process(*args, **kwargs)
            captured["backbone"] = backbone
            return backbone

        def alter_mask(*args):
            captured["backbone"]["image_mask"][:, 0] = True
            return state_encoder(*args)

        with patch.object(head, "process_backbone_output", side_effect=remember_backbone), \
                patch.object(head, "state_encoder", side_effect=alter_mask), \
                patch.object(policy.visual_memory, "append", wraps=policy.visual_memory.append) as writer:
            with self.assertRaisesRegex(ValueError, "masks changed"):
                self.call(policy)
        writer.assert_not_called()
        self.assertEqual(head.denoiser_calls, 0)
        self.assertNotIn("A", policy.sessions)

    def test_original_cadence_errors_and_lru_reset_leave_no_cross_session_visual_state(self):
        policy = self.policy()
        self.call(policy)
        with self.assertRaisesRegex(ValueError, "cadence"):
            self.call(policy, frame=3, controls=np.zeros((3, 8), np.float32))
        self.assertFalse(policy._visual_call_lock.locked())
        self.assertEqual(policy.sessions["A"].visual_bank.tokens.shape[1], 1)
        self.call(policy, seed=99, reset=True)
        self.assertEqual(policy.sessions["A"].visual_bank.tokens.shape[1], 1)
        policy.session_cap = 2
        self.call(policy, session="B", seed=21)
        self.call(policy, session="C", seed=22)
        self.assertEqual(list(policy.sessions), ["B", "C"])
        _, info = self.call(policy)
        self.assertEqual(info["visual_memory"]["observed"], 1)

    def test_cli_help_baseline_parent_visual_dispatch_and_invalid_flags(self):
        help_result = subprocess.run([sys.executable, "-B", "-S", server.__file__, "--help"],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--visual-read-off", help_result.stdout)
        parser = server.build_parser()
        with patch.object(server, "LongMemoryV7Policy") as parent, \
                patch.object(server, "VisualPatchV11Policy") as visual, \
                patch.object(server, "parent_reference") as validate:
            server.create_policy(parser.parse_args(["--base-model", "base", "--device", "cpu"]))
            parent.assert_called_once_with("base", device="cpu", write_policy="checkpoint", memory_off=False)
            visual.assert_not_called(); validate.assert_not_called()
            server.create_policy(parser.parse_args(["--base-model", "base", "--parent-checkpoint", "parent"]))
            validate.assert_called_once_with("base", "parent")
            parent.assert_called_with("base", "parent", device="cuda:0", write_policy="checkpoint", memory_off=False)
            server.create_policy(parser.parse_args(["--base-model", "base", "--memory-checkpoint", "visual", "--visual-read-off"]))
            visual.assert_called_once_with("base", "visual", device="cuda:0", write_policy="checkpoint", visual_read_off=True)
        for flags in (["--visual-read-off"], ["--parent-checkpoint", "parent", "--visual-read-off"],
                      ["--port", "0"], ["--port", "65536"], ["--host", ""]):
            with patch.object(server, "create_policy") as create, self.assertRaises(ValueError):
                server.main(["--base-model", "base", *flags])
            create.assert_not_called()

    def test_server_cleanup_without_real_transport(self):
        for error in (None, KeyboardInterrupt(), RuntimeError("transport failed")):
            policy, transport = MagicMock(), MagicMock()
            transport.run.side_effect = error
            with patch.object(server, "create_policy", return_value=policy), \
                    patch("gr00t.policy.gr00t_policy.Gr00tSimPolicyWrapper", return_value="wrapped"), \
                    patch("gr00t.policy.server_client.PolicyServer", return_value=transport), \
                    contextlib.redirect_stdout(io.StringIO()):
                if isinstance(error, RuntimeError):
                    with self.assertRaisesRegex(RuntimeError, "transport failed"):
                        server.main(["--base-model", "base"])
                else:
                    self.assertEqual(server.main(["--base-model", "base"]), 0)
            policy.reset.assert_called_once_with()
            transport.socket.close.assert_called_once_with(linger=0)
            transport.context.term.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
