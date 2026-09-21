"""CPU regression for the deliberately narrow historical-native READ bridge."""
import copy
import unittest
from unittest.mock import patch
from run_scripts.robomme import compare_read_ablation_v19 as cmp
from run_scripts.robomme.feature_precision_v19 import feature_precision_contract
from tests.test_compare_feature_precision_v19 import fixtures, rehash


def pair():
    on, off = fixtures()
    off["models"]["memory-off"] = off["models"].pop("memory")
    off["models"]["memory-off"].update(memory_off=True, feature_precision="native",
        feature_precision_rules=feature_precision_contract("native"))
    return on, rehash(off)


class ReadAblationTests(unittest.TestCase):
    def test_reviewed_source_changes_only_and_no_manifest_mutation(self):
        on, off = pair()
        saved = copy.deepcopy((on, off))
        self.assertTrue(cmp.validate_pair(on, off)["changed"])
        self.assertEqual((on, off), saved)

    def test_reject_other_precision_or_actor_change(self):
        for update in ({"feature_precision": "cache-aligned",
                        "feature_precision_rules": feature_precision_contract("cache-aligned")},
                       {"memory_checkpoint": "/another/checkpoint"},
                       {"training_config": {"expert": "new_lora"}},
                       {"writer_checkpoint": "/new_writer"}):
            on, off = pair()
            off["models"]["memory-off"].update(update)
            with self.assertRaises(ValueError):
                cmp.validate_pair(on, rehash(off))

    def test_reject_seed_environment_and_unreviewed_source(self):
        for change in (lambda m: m["settings"].update(seed=7),
                       lambda m: m.update(benchmark={"changed": True}),
                       lambda m: m["source_sha256"].update({"gr00t/example.py": "bad"})):
            on, off = pair()
            change(off)
            with self.assertRaises(ValueError):
                cmp.validate_pair(on, rehash(off))

    def test_reject_off_flag_or_rules_missing(self):
        on, off = pair()
        off["models"]["memory-off"]["memory_off"] = False
        with self.assertRaises(ValueError):
            cmp.validate_pair(on, rehash(off))
        on, off = pair()
        del off["models"]["memory-off"]["feature_precision_rules"]
        with self.assertRaises(ValueError):
            cmp.validate_pair(on, rehash(off))

    def test_pairing_sign_both_cells_and_task_cancellation(self):
        rows = {t: {i: dict(success=0, episode_seed=i+5, scenario_seed=str(i+10),
                            task_instruction="same", status="fail", steps="10")
                    for i in range(10)} for t in cmp.TASKS}
        base, off, on = copy.deepcopy(rows), copy.deepcopy(rows), copy.deepcopy(rows)
        off["BinFill"][0]["success"] = 1
        on["MoveCube"][1]["success"] = 1
        off["MoveCube"][2]["success"] = on["MoveCube"][2]["success"] = 1
        with patch.object(cmp, "paired_macro_bootstrap", return_value=(-.01, .01)):
            result = cmp.summarize(base, off, on)
            self.assertEqual((result["wins"], result["losses"], result["both_success"], result["both_failure"]), (1,1,1,157))
            self.assertEqual(result["delta"], 0)
            self.assertEqual(len(result["flips"]), 2)
            self.assertGreater(result["tasks"]["MoveCube"]["delta"], 0)
            self.assertLess(result["tasks"]["BinFill"]["delta"], 0)
            on["MoveCube"][1]["episode_seed"] = 999
            with self.assertRaises(ValueError):
                cmp.summarize(base, off, on)


if __name__ == "__main__":
    unittest.main()
