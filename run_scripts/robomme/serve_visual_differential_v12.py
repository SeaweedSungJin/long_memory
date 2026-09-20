#!/usr/bin/env python3
"""Serve an actual V12 arm, its immutable archive1250 parent, or original HAMLET."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--memory-checkpoint")
    source.add_argument("--parent-checkpoint")
    parser.add_argument("--expected-read-mode", choices=("differential", "current_only"))
    parser.add_argument("--visual-read-off", action="store_true")
    parser.add_argument("--write-policy", choices=("checkpoint",), default="checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    return parser


if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    build_parser().parse_args()

from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
from run_scripts.robomme.checkpoint_visual_differential_v12 import parent_reference
from run_scripts.robomme.policy_visual_differential_v12 import VisualDifferentialV12Policy


def validate_options(args):
    if args.memory_checkpoint is not None and args.parent_checkpoint is not None:
        raise ValueError("Choose one real checkpoint")
    if (args.visual_read_off or args.expected_read_mode is not None) and args.memory_checkpoint is None:
        raise ValueError("Read-off/mode assertions require a V12 checkpoint")
    if args.visual_read_off and args.expected_read_mode == "current_only":
        raise ValueError("visual-off requires a differential checkpoint")
    if args.write_policy != "checkpoint" or type(args.port) is not int or not 1 <= args.port <= 65535:
        raise ValueError("Invalid checkpoint WRITE/port option")
    if not isinstance(args.host, str) or not args.host.strip():
        raise ValueError("Host must be nonempty")


def create_policy(args):
    validate_options(args)
    if args.memory_checkpoint is not None:
        return VisualDifferentialV12Policy(args.base_model, args.memory_checkpoint, device=args.device,
            write_policy="checkpoint", visual_read_off=args.visual_read_off, expected_read_mode=args.expected_read_mode)
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
    print(f"[visual-v12] mode={getattr(policy, 'visual_read_mode', 'parent/original')}; "
          f"read_off={args.visual_read_off}; step={getattr(policy, 'checkpoint_step', None)}; all weights frozen", flush=True)
    transport = None
    try:
        transport = PolicyServer(Gr00tSimPolicyWrapper(policy), host=args.host, port=args.port)
        transport.run()
    except KeyboardInterrupt:
        print("[visual-v12] stopped", flush=True)
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
