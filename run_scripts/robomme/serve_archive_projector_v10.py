#!/usr/bin/env python3
"""Serve original HAMLET or actual V10 archive/projector READ-on/off bundles.

Omit --memory-checkpoint for the unmodified original HAMLET. With a V10 bundle,
--archive-read-off disables only memory READ, not LoRA/projector adaptation or
APPEND writes. This launcher does not start an evaluator or save task results.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-checkpoint", help="Actual V10 Stage-1 archive/projector bundle; omit for baseline")
    parser.add_argument("--archive-read-off", action="store_true",
                        help="Keep the same adapted Expert and APPEND writes; bypass archive READ only")
    parser.add_argument("--write-policy", choices=("checkpoint",), default="checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    return parser


# --help must work without importing Torch or loading the model environment.
if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    build_parser().parse_args()

from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
from run_scripts.robomme.policy_archive_projector_v10 import ArchiveProjectorV10Policy


def validate_options(args):
    if args.archive_read_off and args.memory_checkpoint is None:
        raise ValueError("--archive-read-off requires --memory-checkpoint")
    if args.write_policy != "checkpoint":
        raise ValueError("V10 archive WRITE must remain checkpoint APPEND")
    if type(args.port) is not int or not 1 <= args.port <= 65535:
        raise ValueError("Port must be an integer in [1, 65535]")
    if not isinstance(args.host, str) or not args.host.strip():
        raise ValueError("Host must be nonempty")


def create_policy(args):
    validate_options(args)
    if args.memory_checkpoint is None:
        return LongMemoryV7Policy(args.base_model, device=args.device,
                                  write_policy="checkpoint", memory_off=False)
    return ArchiveProjectorV10Policy(args.base_model, args.memory_checkpoint, device=args.device,
        write_policy="checkpoint", archive_read_off=args.archive_read_off)


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_options(args)
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    policy = create_policy(args)
    role = "baseline" if args.memory_checkpoint is None else "archive-off" if args.archive_read_off else "archive"
    print(f"[archive-projector-v10] role={role}; stage={policy.stage}; mode={policy.mode}; "
          f"write_policy={policy.write_policy}; memory_off={policy.memory_off}; "
          f"checkpoint_step={getattr(policy, 'checkpoint_step', 'original')}; all inference weights frozen", flush=True)
    server = None
    try:
        server = PolicyServer(Gr00tSimPolicyWrapper(policy), host=args.host, port=args.port)
        server.run()
    except KeyboardInterrupt:
        print("[archive-projector-v10] stopped", flush=True)
    finally:
        try:
            policy.reset()
        finally:
            if server is not None:
                try:
                    server.socket.close(linger=0)
                finally:
                    server.context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
