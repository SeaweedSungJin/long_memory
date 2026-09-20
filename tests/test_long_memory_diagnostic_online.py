"""Small CPU regressions for read-only online interventions and safe evaluation."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from gr00t.long_memory.diagnostic_online import DiagnosticOnlineBank, temporal_content_permutation
from gr00t.long_memory.online_v3 import OnlineActionValueBank
from gr00t.long_memory.replay_v3 import event_inputs
from run_scripts.robomme import eval_long_memory_diagnostic as driver
from tests.test_long_memory_v3_core import model
from tests.test_long_memory_v3_online import advance, data_with_demo
from tests import test_long_memory_v4_evaluation as fixtures


class DiagnosticBankTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def test_native_full_and_interventions_never_change_writer_bank_on_identical_inputs(self):
        memory = model(capacity=8).eval().requires_grad_(False)
        data = data_with_demo()
        # Explicitly exercise an irregular final endpoint rather than frame//stride.
        data["frames"][-1] -= 1
        data["action_mask"][-1, 1] = False
        native = OnlineActionValueBank(memory, 2, "all")
        banks = {mode: DiagnosticOnlineBank(memory, 2, "all", memory_window=3, mode=mode)
                 for mode in ("full", "expert-only", "no-old", "shuffled-old", "fifo")}
        changed = {mode: False for mode in banks}
        for decision in range(9):
            expected, native_diag = advance(native, data, decision)
            for mode, bank in banks.items():
                fused, diag = advance(bank, data, decision)
                self.assertEqual(bank.bank_ids, native.bank_ids)
                self.assertEqual(diag["accepted_writes"], native_diag["accepted_writes"])
                self.assertEqual(diag["short_endpoint_frames"], data["frames"][max(0, decision-2):decision+1].tolist())
                boundary = int(data["frames"][max(0, decision-2)])
                old = [i for i in bank.bank_ids if int(data["frames"][i+1]) < boundary]
                self.assertEqual(diag["old_event_ids"], old)
                for event_id in native.bank_ids:
                    for field in native.encodings[event_id]:
                        self.assertTrue(torch.equal(bank.encodings[event_id][field], native.encodings[event_id][field]))
                if mode in ("full", "fifo") or (mode == "no-old" and not old) or (mode == "shuffled-old" and len(old) < 2):
                    self.assertTrue(torch.equal(fused, expected))
                if mode == "expert-only":
                    self.assertTrue(torch.equal(fused[0], data["short"][decision]))
                    self.assertEqual(diag["residual_norm"], 0)
                if mode == "no-old":
                    self.assertEqual(diag["read_event_ids"], [i for i in bank.bank_ids if i not in old])
                changed[mode] |= diag["intervention_changed_fused"]
                json.dumps(diag, allow_nan=False)
        self.assertTrue(changed["no-old"])
        self.assertTrue(changed["expert-only"])
        self.assertTrue(changed["shuffled-old"])
        self.assertFalse(changed["full"])

    def test_content_permutation_preserves_destination_time_and_recent_exactly(self):
        memory, data = model(capacity=8).eval().requires_grad_(False), data_with_demo()
        bank = DiagnosticOnlineBank(memory, 2, "all", memory_window=3, mode="full")
        for decision in range(9):
            advance(bank, data, decision)
        old = bank.last_intervention["old_event_ids"]
        self.assertGreaterEqual(len(old), 2)
        with torch.no_grad():
            keys, values, donors = temporal_content_permutation(memory, bank.encodings, bank.bank_ids, old)
            self.assertEqual(set(donors), set(old))
            self.assertEqual(set(donors.values()), set(old))
            for slot, event_id in enumerate(bank.bank_ids):
                dest = bank.encodings[event_id]
                if event_id not in old:
                    self.assertTrue(torch.equal(keys[0, slot], dest["keys"]))
                    self.assertTrue(torch.equal(values[0, slot], dest["values"]))
                    continue
                source = bank.encodings[donors[event_id]]
                self.assertNotEqual(donors[event_id], event_id)
                times = memory.time_features(dest["starts"][None], dest["ends"][None])
                expected_keys = memory.key(source["tokens"] + source["event"][None]) + memory.time_key(times)
                expected_values = memory.value(source["tokens"] + source["event"][None]) + memory.time_value(times)
                torch.testing.assert_close(keys[0, slot], expected_keys)
                torch.testing.assert_close(values[0, slot], expected_values)
            # Independent oracle: permute RAW content and run the whole encoder.
            raw = event_inputs(data, "cpu", 8)
            permuted = {}
            source_ids = [donors.get(i, i) for i in bank.bank_ids]
            for field, value in raw.items():
                indices = bank.bank_ids if field in ("start_frames", "end_frames") else source_ids
                permuted[field] = value[indices]
            reencoded = memory.encode_events(permuted)
            torch.testing.assert_close(keys[0], reencoded["keys"], atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(values[0], reencoded["values"], atol=3e-6, rtol=3e-5)

    def test_learned_writer_decisions_and_retained_encodings_also_unchanged(self):
        memory, data = model(capacity=3).eval().requires_grad_(False), data_with_demo()
        native = OnlineActionValueBank(memory, 2, "hard")
        banks = [DiagnosticOnlineBank(memory, 2, "hard", memory_window=2, mode=mode)
                 for mode in ("full", "expert-only", "no-old", "shuffled-old")]
        for decision in range(9):
            _, expected = advance(native, data, decision)
            for bank in banks:
                _, diag = advance(bank, data, decision)
                for name in ("storage_choice", "storage_options", "bank_event_ids", "learned_rejected", "learned_accepted"):
                    self.assertEqual(diag[name], expected[name])
                for event_id in native.bank_ids:
                    for field in native.encodings[event_id]:
                        self.assertTrue(torch.equal(bank.encodings[event_id][field], native.encodings[event_id][field]))

    def test_reset_drops_frame_history_and_strict_boundary_excludes_equal_end(self):
        memory, data = model(capacity=8).eval().requires_grad_(False), data_with_demo()
        bank = DiagnosticOnlineBank(memory, 2, "all", memory_window=3, mode="no-old")
        for decision in range(4):
            _, diag = advance(bank, data, decision)
        # At frame 6 the earliest short endpoint is frame 2; event 0 ends at 2.
        self.assertEqual(diag["old_boundary_frame"], 2)
        self.assertEqual(diag["old_event_ids"], [])
        bank.reset()
        self.assertEqual(list(bank.endpoint_frames), [])
        fused, diag = advance(bank, data, 0)
        self.assertTrue(torch.equal(fused[0], data["short"][0]))
        self.assertFalse(diag["intervention_applied"])


class DiagnosticDriverTests(unittest.TestCase):
    def setUp(self):
        # Reuse small synthetic, structurally valid v4 bundles, not real weights.
        self.fixture = fixtures.V4EvaluationTests()
        self.fixture.setUp()
        self.root, self.base, self.checkpoint = self.fixture.root, self.fixture.base, self.fixture.memory
        config_path = self.base / "config.json"
        config = json.loads(config_path.read_text())
        config["memory_window"] = 4
        config_path.write_text(json.dumps(config))
        for path in (self.fixture.reader, self.fixture.memory):
            self.fixture.update_info(path, lambda info: info["metadata"].update(
                base_model=fixtures.checkpoint_identity(self.base)))

    def tearDown(self):
        self.fixture.tearDown()

    def args(self, *extra):
        return driver.build_parser().parse_args(["--base-model", str(self.base),
            "--memory-checkpoint", str(self.checkpoint), "--output-dir", str(self.root / "output"), *extra])

    def identity(self, args):
        driver.validate_options(args)
        with patch.object(driver.v4, "benchmark_identity", return_value={}), contextlib.redirect_stdout(io.StringIO()):
            return driver.build_identity(args)

    def test_same_stage2_adapter_for_all_adapted_conditions_and_original_baseline(self):
        args = self.args()
        identity = self.identity(args)
        models = identity["models"]
        self.assertEqual(identity["trainer_variant"], driver.VARIANT)
        self.assertIsNone(models["baseline"]["memory_checkpoint"])
        for mode in driver.MODES[1:]:
            self.assertEqual(models[mode]["stage"], 2)
            self.assertEqual(models[mode]["memory_checkpoint"], str(self.checkpoint))
            self.assertEqual(models[mode]["expert_weights_sha256"], models["full"]["expert_weights_sha256"])
        self.assertEqual(models["fifo"]["write_policy"], "all")
        self.assertEqual(models["expert-only"]["write_policy"], "hard")
        self.assertIn("shadow", models["expert-only"]["description"])
        self.assertIn("gr00t/long_memory/diagnostic_online.py", identity["source_sha256"])
        command = driver.server_command(args, models["expert-only"], 5550)
        self.assertEqual(command[command.index("--mode")+1], "expert-only")
        self.assertNotIn("--expert-only", command)
        self.assertNotIn("--memory-checkpoint", driver.server_command(args, models["baseline"], 5550))
        self.assertFalse((self.root / "output").exists())

    def test_preflight_no_output_no_model_or_rollout(self):
        argv = ["--models", "baseline", "--base-model", str(self.base), "--preflight-only",
                "--output-dir", str(self.root / "readonly"), "--server-python", sys.executable]
        with patch.object(driver, "check_dependencies"), patch.object(driver.v4, "benchmark_identity", return_value={}), \
                patch.object(driver, "run_evaluation") as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(argv), 0)
        launch.assert_not_called()
        self.assertFalse((self.root / "readonly").exists())

    def test_wrong_stage_refused(self):
        args = self.args("--memory-checkpoint", str(self.fixture.reader))
        with self.assertRaises(ValueError):
            self.identity(args)

    def test_initial_output_lock_prevents_concurrent_bind_and_wrong_provenance_refused(self):
        args = self.args("--models", "baseline")
        identity = self.identity(args)
        output = self.root / "output"
        with driver.driver_lock(output):
            with self.assertRaisesRegex(RuntimeError, "Another"):
                driver.run_evaluation(args, identity, {})
        self.assertFalse(output.exists())
        driver.bind_manifest(output, identity)
        changed = {**identity, "evaluation_id": "changed"}
        with patch.object(driver.subprocess, "Popen") as launch, self.assertRaisesRegex(ValueError, "checksum"):
            driver.run_evaluation(args, changed, {})
        launch.assert_not_called()

    def test_report_only_rejects_legacy_manifest_and_adapted_baseline_contamination(self):
        args = self.args()
        identity = self.identity(args)
        output = self.root / "output"
        driver.bind_manifest(output, identity)
        target = output / "comparison_manifest.json"
        legacy = {**identity, "trainer_variant": "action_expert_v4"}
        target.write_text(json.dumps(legacy))
        with self.assertRaisesRegex(ValueError, "Not a diagnostic"):
            driver.main(["--report-only", "--output-dir", str(output)])
        identity["models"]["baseline"]["memory_checkpoint"] = str(self.checkpoint)
        target.write_text(json.dumps(identity))
        with self.assertRaisesRegex(ValueError, "baseline"):
            driver.main(["--report-only", "--output-dir", str(output)])
        self.assertFalse((output / "comparison_summary.json").exists())

    def test_report_manifest_cannot_mix_stage1_or_other_adapted_expert(self):
        identity = self.identity(self.args())
        identity["models"]["expert-only"]["stage"] = 1
        with self.assertRaisesRegex(ValueError, "Stage 2"):
            driver.validate_diagnostic_manifest(identity)
        identity["models"]["expert-only"]["stage"] = 2
        identity["models"]["expert-only"]["expert_weights_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "SAME Stage 2"):
            driver.validate_diagnostic_manifest(identity)

    def test_server_error_is_missing_not_a_failed_robot_episode(self):
        args = self.args("--models", "baseline", "--tasks", "BinFill", "--n-episodes", "1")
        identity = self.identity(args)
        with patch.object(driver.subprocess, "Popen"), patch.object(driver, "server_ready", side_effect=RuntimeError("fake failure")), \
                patch.object(driver, "stop_process"), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(driver.run_evaluation(args, identity, {}), 1)
        result = json.loads((self.root / "output/comparison_summary.json").read_text())
        task = result["models"]["baseline"]["tasks"]["BinFill"]
        self.assertIsNone(task["success_rate"])
        self.assertEqual(task["completed"], 0)

    def test_coverage_excludes_passive_and_incomplete_or_retry_sessions(self):
        folder = self.root / "coverage/full/BinFill"
        folder.mkdir(parents=True)
        (folder / "simulation_results.csv").write_text("episode_idx,episode_seed,success\n0,100,1\n1,101,0\n")
        records = []
        def call(sid, index, *, passive=False, old=2):
            return {"kind": "policy_call", "session_id": sid, "call": index, "passive": passive,
                "info": {"long_memory": {"diagnostic_mode": "full", "old_event_count": old,
                                          "intervention_applied": False, "intervention_changed_fused": False}}}
        records += [call("crashed", 0), call("ok", 0, passive=True), call("ok", 1), call("ok", 2, old=0),
                    {"kind": "episode_complete", "session_id": "ok", "episode_idx": 0, "episode_seed": 100, "success": True}]
        (folder / "memory_diagnostics.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records) + '{"torn":')
        totals = driver.completed_intervention_coverage(self.root / "coverage", "full", ["BinFill"], 2)
        self.assertEqual(totals["completed_sessions_with_diagnostics"], 1)
        self.assertEqual(totals["completed_sessions_missing_diagnostics"], 1)
        self.assertEqual(totals["executed_policy_calls"], 2)
        self.assertEqual(totals["calls_with_old"], 1)
        self.assertEqual(totals["old_coverage_fraction"], 0.5)
        self.assertEqual(totals["ignored_torn_final_lines"], 1)


if __name__ == "__main__":
    unittest.main()
