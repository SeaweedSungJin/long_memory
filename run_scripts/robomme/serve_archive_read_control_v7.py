#!/usr/bin/env python3
"""Serve original HAMLET or paired READ-on/off from one V7 archive checkpoint.

The archive ablation retains its adapted Action Expert and APPENDs every
observed endpoint. Only the archive READ is disabled; native HAMLET short
memory, observation processing, session RNG and action decoding are unchanged.
This separate launcher leaves the original recurrent-only CLI contract intact.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-checkpoint", help="Trained V7 Stage-1 archive bundle; omit for original HAMLET")
    parser.add_argument("--write-policy", choices=("checkpoint",), default="checkpoint",
                        help="Archive WRITE remains APPEND in both arms")
    parser.add_argument("--archive-read-off", action="store_true",
                        help="Bypass archive READ only; preserve the same adapted AE and APPEND writes")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    return parser


# Keep the command-line help independent of Torch/model dependencies. Normal
# imports expose the policy class for CPU tests and the dedicated evaluator.
if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    build_parser().parse_args()

from gr00t.long_memory.checkpoint_v7 import v7_checkpoint_info
from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy


class ArchiveReadControlV7Policy(LongMemoryV7Policy):
    """Explicit Stage-1 archive extension of V7's existing READ bypass.

The inherited constructor intentionally permits its public ``memory_off``
option only for recurrent actors. Validate this archive-only experiment first,
then load the unchanged archive policy and select its existing READ gate. No
checkpoint configuration or inherited method is rewritten.
"""

    def __init__(self, base_model, memory_checkpoint, device="cuda:0", strict=True,
                 write_policy="checkpoint", archive_read_off=False):
        if type(archive_read_off) is not bool:
            raise TypeError("archive_read_off must be boolean")
        if write_policy != "checkpoint":
            raise ValueError("Archive READ controls require checkpoint APPEND writes")
        if memory_checkpoint is None:
            raise ValueError("Archive READ controls require a trained archive checkpoint")
        info = v7_checkpoint_info(base_model, memory_checkpoint, expected_stage=1)
        config = info["config"]
        if type(config.get("stage")) is not int or config["stage"] != 1 or config.get("mode") != "archive":
            raise ValueError("Archive READ controls require V7 Stage 1 mode=archive")
        if type(info.get("step")) is not int or info["step"] <= 0:
            raise ValueError("Archive READ controls require a trained nonzero checkpoint")
        super().__init__(base_model, memory_checkpoint, device=device, strict=strict,
                         write_policy="checkpoint", memory_off=False)
        if self.stage != 1 or self.mode != "archive" or self.write_policy != "append":
            raise RuntimeError("Loaded policy differs from the validated Stage-1 archive")
        self.memory_off = archive_read_off


def create_policy(args):
    """Keep baseline and archive roles explicit; reject an unbound ablation."""
    if args.memory_checkpoint is None:
        if args.archive_read_off:
            raise ValueError("--archive-read-off requires --memory-checkpoint")
        return LongMemoryV7Policy(args.base_model, device=args.device,
                                  write_policy=args.write_policy, memory_off=False)
    return ArchiveReadControlV7Policy(args.base_model, args.memory_checkpoint, device=args.device,
                                     write_policy=args.write_policy, archive_read_off=args.archive_read_off)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.archive_read_off and args.memory_checkpoint is None:
        raise ValueError("--archive-read-off requires --memory-checkpoint")
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    policy = create_policy(args)
    role = "baseline" if args.memory_checkpoint is None else "archive-off" if args.archive_read_off else "archive"
    print(f"[archive-read-control-v7] role={role}; stage={policy.stage}; mode={policy.mode}; "
          f"write_policy={policy.write_policy}; memory_off={policy.memory_off}; all inference weights frozen",
          flush=True)
    server = None
    try:
        server = PolicyServer(Gr00tSimPolicyWrapper(policy), host=args.host, port=args.port)
        server.run()
    except KeyboardInterrupt:
        print("[archive-read-control-v7] stopped", flush=True)
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
