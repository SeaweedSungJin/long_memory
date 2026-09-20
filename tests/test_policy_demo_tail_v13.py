"""Real V13 bundles/constructor/serial methods; only the large base is mocked."""
import contextlib
import io
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
from run_scripts.robomme import policy_demo_tail_v13 as policy_module
from run_scripts.robomme import serve_demo_tail_v13 as server
from tests import test_policy_visual_patch_v11 as legacy
from tests.test_policy_demo_tail_ingest_v13 import ingest, request
from tests import test_checkpoint_demo_tail_v13 as checkpoint_fixtures


def initialize_tiny_parent(actor, base, parent, **kwargs):
    actor.__dict__.update(legacy.parent_policy().__dict__)
    # Old fixture CPUModel is not an nn.Module. Expose the real tiny head's
    # parameter/freeze API without changing its original forward behavior.
    model = actor.model
    model.training = False
    model.parameters = model.action_head.parameters
    model.eval = lambda: model
    model.requires_grad_ = lambda flag: (model.action_head.requires_grad_(flag), model)[1]
    actor.stride = 16


class DemoTailPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.fixture = checkpoint_fixtures.DemoTailCheckpointTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    call = legacy.VisualPatchPolicyTests.call
    assert_actions_equal = legacy.VisualPatchPolicyTests.assert_actions_equal

    def actor(self, bundle, *, tail=True, off=False):
        with patch.object(LongMemoryV7Policy, "__init__", initialize_tiny_parent):
            return policy_module.DemoTailV13Policy(self.fixture.base, bundle, device="cpu",
                visual_read_off=off, expected_include_tail=tail)

    def test_real_distinct_checkpoint_constructor_zero_ingest_actions_and_cleanup(self):
        bundle = self.fixture.save()
        actor, off = self.actor(bundle), self.actor(bundle, off=True)
        for instance in (actor, off):
            self.assertEqual(instance.checkpoint_variant, "visual_demo_tail_v13")
            self.assertEqual(instance.checkpoint_step, 0)
            self.assertEqual(len(instance.visual_memory.state_dict()), 14)
            for name, value in self.fixture.visual.state_dict().items():
                self.assertTrue(torch.equal(value, instance.visual_memory.state_dict()[name]))
            for frame in (0, 16, 32):
                self.call(instance, marker=frame + 1, frame=frame, passive=True)
            ingest(instance, request("A"))
        actions, info = self.call(actor, marker=4, frame=48)
        reference, off_info = self.call(off, marker=4, frame=48)
        self.assert_actions_equal(actions, reference)
        self.assertEqual(info["long_memory"], off_info["long_memory"])
        self.assertEqual(info["demo_tail"]["effective_prior_observations"], 18)
        self.assertEqual(off_info["demo_tail"]["effective_prior_observations"], 3)
        self.assertEqual(info["checkpoint_variant"], "visual_demo_tail_v13")
        actor.reset()
        off.reset()
        self.assertFalse(actor.sessions)
        self.assertFalse(off.sessions)

    def test_real_bundle_wrong_arm_off_control_and_old_format_rejected_before_model(self):
        bundle = self.fixture.save()
        for kwargs in ({"expected_include_tail": False}, {"expected_include_tail": 1}):
            with patch.object(LongMemoryV7Policy, "__init__") as model:
                with self.assertRaises((TypeError, ValueError)):
                    policy_module.DemoTailV13Policy(self.fixture.base, bundle, **kwargs)
                model.assert_not_called()
        old = self.fixture.fixture.save(name="legacy")
        with patch.object(LongMemoryV7Policy, "__init__") as model:
            with self.assertRaisesRegex(ValueError, "visual_demo_tail_v13"):
                policy_module.DemoTailV13Policy(self.fixture.base, old)
            model.assert_not_called()

    def test_actual_canonical_bundle_does_not_accept_tail_off_control(self):
        f = self.fixture
        for section in (f.config, f.config["train"], f.metadata):
            section["include_tail"] = False
        from run_scripts.robomme.checkpoint_demo_tail_v13 import bind_visual_semantics
        from run_scripts.robomme.visual_demo_tail_bank_v13 import VisualDemoTailMemoryV13
        f.visual = bind_visual_semantics(VisualDemoTailMemoryV13(f.visual.config), include_tail=False)
        bundle = f.save(name="canonical")
        actor = self.actor(bundle, tail=False)
        self.assertFalse(actor._demo_tail_ingest_enabled)
        self.call(actor, frame=0)
        with patch.object(LongMemoryV7Policy, "__init__") as model:
            with self.assertRaisesRegex(ValueError, "tail-on checkpoint"):
                policy_module.DemoTailV13Policy(f.base, bundle, visual_read_off=True)
            model.assert_not_called()

    def test_required_ingest_cannot_silently_skip_on_off_role(self):
        bundle = self.fixture.save()
        for off in (True, False):
            actor = self.actor(bundle, off=off)
            for frame in (0, 16, 32):
                self.call(actor, frame=frame, passive=True)
            with self.assertRaisesRegex(ValueError, "ingest|tail|Tail"):
                self.call(actor, frame=48)
            self.assertFalse(actor.sessions)
            self.assertFalse(actor._visual_call_lock.locked())

    def test_server_help_has_no_heavy_import_and_strict_role_options(self):
        result = subprocess.run([sys.executable, "-B", "-S", server.__file__, "--help"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--expected-include-tail", result.stdout)
        with patch.object(policy_module, "DemoTailV13Policy") as create:
            args = server.build_parser().parse_args(["--base-model", "base", "--memory-checkpoint", "v13",
                "--no-expected-include-tail"])
            server.create_policy(args)
            create.assert_called_once_with("base", "v13", device="cuda:0", write_policy="checkpoint",
                visual_read_off=False, expected_include_tail=False)
        for flags in (["--port", "0"], ["--host", ""], ["--no-expected-include-tail", "--visual-read-off"]):
            with patch.object(server, "create_policy") as create, self.assertRaises(ValueError):
                server.main(["--base-model", "base", "--memory-checkpoint", "v13", *flags])
            create.assert_not_called()

    def test_server_registers_raw_rgb_endpoint_and_closes_on_error(self):
        for error in (None, KeyboardInterrupt(), RuntimeError("injected")):
            actor, transport = MagicMock(), MagicMock()
            transport.run.side_effect = error
            with patch.object(server, "create_policy", return_value=actor), \
                    patch("gr00t.policy.gr00t_policy.Gr00tSimPolicyWrapper", return_value="wrapped"), \
                    patch("gr00t.policy.server_client.PolicyServer", return_value=transport), \
                    contextlib.redirect_stdout(io.StringIO()):
                if isinstance(error, RuntimeError):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        server.main(["--base-model", "base", "--memory-checkpoint", "v13"])
                else:
                    self.assertEqual(server.main(["--base-model", "base", "--memory-checkpoint", "v13"]), 0)
            transport.register_endpoint.assert_called_once_with(server.ENDPOINT, actor.ingest_demo_tail)
            actor.reset.assert_called_once_with()
            transport.socket.close.assert_called_once_with(linger=0)
            transport.context.term.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
