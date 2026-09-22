#!/usr/bin/env python3
"""Read-only review gate. Never invokes training, even on positive evidence."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_scripts.robomme.compare_cvom_budget_noise import load_verified_run, validate_noise_pair
from run_scripts.robomme.train_cvom_admission import file_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args(argv)
    result = json.loads(args.comparison.read_text())
    proof = result["provenance"]
    for name, sha in proof["source_sha256"].items():
        if file_hash(ROOT/"run_scripts/robomme"/name) != sha:
            raise ValueError("Comparison analysis source changed; create a NEW comparison report")
    a, ap = load_verified_run(proof["noise2_dir"])
    b, bp = load_verified_run(proof["noise8_dir"])
    validate_noise_pair(a, b, ap, bp)
    if ([a["fingerprint"], b["fingerprint"]] != proof["protocol_fingerprints"] or
            [[p["result_sha256"] for p in packets] for packets in (ap, bp)] != proof["result_payload_hashes"]):
        raise ValueError("Compared evidence changed")
    # Recompute the decision from raw records; a hand-edited PASS cannot bypass
    # the gate by altering only the summary JSON.
    from run_scripts.robomme.compare_cvom_budget_noise import compare
    fresh = compare(a, b, ap, bp)
    if fresh["teacher_gate"] != result["teacher_gate"]:
        raise ValueError("Stored teacher decision does not match raw paired evidence")
    gate = fresh["teacher_gate"]
    for name in ("noise2", "noise8"):
        s = result[name]
        rank = s["stability"]["pairwise_repeat_pooled_episode_bootstrap"]
        gain = s["overall"]["generated_mse"]["episode_macro"]["gains"]["selected_a_vs_fifo"]
        print(f"{name}: rank repeat={rank['value']:.2%}; generated MSE gain vs FIFO={gain['mean']:+.9f}, CI={gain['ci95']}")
    print(json.dumps(gate, indent=2))
    print("No training was launched. This is offline cache-VAL evidence, NOT rollout success.")
    if not gate["eligible_for_manual_training_review"]:
        print("HOLD: writer retraining and reader/fusion adaptation are deferred; min_fill-only rollout remains an independent test.")
    else:
        print("Manual review only: preserve original actor as control; writer-only then a separate short reader/fusion arm. No automatic all pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
