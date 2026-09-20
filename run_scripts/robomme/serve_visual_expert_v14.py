#!/usr/bin/env python3
"""Serve a genuine V14 joint bundle or an explicitly original, unadapted baseline."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ENDPOINT = "ingest_demo_tail_v13"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-checkpoint", help="Actual visual_expert_v14 bundle; omit for original HAMLET")
    parser.add_argument("--expected-include-tail", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visual-read-off", action="store_true")
    parser.add_argument("--write-policy", choices=("checkpoint",), default="checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    return parser


def validate_options(args):
    if args.write_policy != "checkpoint" or not 1 <= args.port <= 65535:
        raise ValueError("Invalid unchanged WRITE policy/port")
    if not args.host.strip():
        raise ValueError("Host must be nonempty")
    if args.memory_checkpoint is None:
        if args.visual_read_off or args.expected_include_tail is not None:
            raise ValueError("Original baseline has no visual/tail/expert override")
    elif args.expected_include_tail is False:
        raise ValueError("V14 requires the joint include_tail=True bundle")


def create_policy(args):
    validate_options(args)
    if args.memory_checkpoint is None:
        from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
        return LongMemoryV7Policy(args.base_model, memory_checkpoint=None, device=args.device,
                                  write_policy="checkpoint", memory_off=False)
    from run_scripts.robomme.policy_visual_expert_v14 import VisualExpertV14Policy
    return VisualExpertV14Policy(args.base_model, args.memory_checkpoint, device=args.device,
        write_policy=args.write_policy, visual_read_off=args.visual_read_off,
        expected_include_tail=args.expected_include_tail)


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_options(args)
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer
    policy = create_policy(args)
    if args.memory_checkpoint is None:
        print("[visual-v14] original HAMLET baseline; no visual memory or expert adapters", flush=True)
    else:
        print(f"[visual-v14] include_tail=True; visual_read_off={policy.visual_read_off}; "
              f"saved expert retained; step={policy.checkpoint_step}; all weights frozen", flush=True)
    server = None
    try:
        server = PolicyServer(Gr00tSimPolicyWrapper(policy), host=args.host, port=args.port)
        if args.memory_checkpoint is not None:
            server.register_endpoint(ENDPOINT, policy.ingest_demo_tail)
        server.run()
    except KeyboardInterrupt:
        print("[visual-v14] stopped", flush=True)
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
