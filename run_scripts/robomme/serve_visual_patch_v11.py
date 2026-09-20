#!/usr/bin/env python3
"""Serve standalone V11 visual memory, its unchanged parent, or original HAMLET.

--memory-checkpoint is a DISTINCT visual_patch_v11 bundle with a required
external archive-1250 parent. --visual-read-off bypasses only visual READ; both
APPEND streams and parent archive READ remain on. Step-zero bundles are labeled
initialization diagnostics, not trained models. This server runs serially.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--memory-checkpoint", help="Actual V11 visual-only checkpoint; external parent is mandatory")
    checkpoint.add_argument("--parent-checkpoint", help="Serve only this unchanged V7 archive-1250 parent")
    parser.add_argument("--visual-read-off", action="store_true", help="Bypass ONLY visual READ, retaining all APPEND writes")
    parser.add_argument("--write-policy", choices=("checkpoint",), default="checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    return parser


if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    build_parser().parse_args()  # Help does not import Torch or the model stack.

from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
from run_scripts.robomme.checkpoint_visual_patch_v11 import parent_reference
from run_scripts.robomme.policy_visual_patch_v11 import VisualPatchV11Policy


def validate_options(args):
    if args.memory_checkpoint is not None and args.parent_checkpoint is not None:
        raise ValueError("Select either a V11 checkpoint or the parent-only role")
    if args.visual_read_off and args.memory_checkpoint is None:
        raise ValueError("--visual-read-off requires --memory-checkpoint")
    if args.write_policy != "checkpoint":
        raise ValueError("V11 and parent WRITE must remain checkpoint APPEND")
    if type(args.port) is not int or not 1 <= args.port <= 65535:
        raise ValueError("Port must be an integer in [1, 65535]")
    if not isinstance(args.host, str) or not args.host.strip():
        raise ValueError("Host must be nonempty")


def create_policy(args):
    validate_options(args)
    if args.memory_checkpoint is not None:
        return VisualPatchV11Policy(args.base_model, args.memory_checkpoint, device=args.device,
                                   write_policy="checkpoint", visual_read_off=args.visual_read_off)
    if args.parent_checkpoint is not None:
        parent_reference(args.base_model, args.parent_checkpoint)
        return LongMemoryV7Policy(args.base_model, args.parent_checkpoint, device=args.device,
                                  write_policy="checkpoint", memory_off=False)
    return LongMemoryV7Policy(args.base_model, device=args.device, write_policy="checkpoint", memory_off=False)


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_options(args)
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    policy = create_policy(args)
    role = ("visual-off" if args.visual_read_off else "visual") if args.memory_checkpoint else (
        "parent" if args.parent_checkpoint else "baseline")
    print(f"[visual-patch-v11] role={role}; stage={policy.stage}; mode={policy.mode}; "
          f"write_policy={policy.write_policy}; checkpoint_step={getattr(policy, 'checkpoint_step', 'parent/original')}; "
          f"visual_step_zero_diagnostic={getattr(policy, 'checkpoint_step', None) == 0}; all weights frozen", flush=True)
    transport = None
    try:
        transport = PolicyServer(Gr00tSimPolicyWrapper(policy), host=args.host, port=args.port)
        transport.run()
    except KeyboardInterrupt:
        print("[visual-patch-v11] stopped", flush=True)
    finally:
        try:
            policy.reset()
        finally:
            if transport is not None:
                try:
                    transport.socket.close(linger=0)
                finally:
                    transport.context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
