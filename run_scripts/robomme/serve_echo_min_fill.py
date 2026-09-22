#!/usr/bin/env python3
"""Serve the same ECHO weights with an explicitly recorded min-fill override."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_scripts.robomme.serve_echo_cvom import build_parser as echo_parser


def build_parser():
    parser = echo_parser()
    parser.description = __doc__
    parser.add_argument("--min-fill", type=int, choices=(4, 32), default=None,
                        help="Runtime override; omission preserves checkpoint configuration")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("Invalid TCP port")
    from run_scripts.robomme.policy_echo_min_fill import EchoMinFillPolicy
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer
    policy = EchoMinFillPolicy(args.base_model, args.checkpoint, device=args.device,
        memory_off=args.memory_off, fifo=args.fifo, min_fill=args.min_fill)
    print(f"[echo-min-fill] stage={policy.stage}; step={policy.checkpoint_step}; "
          f"write={policy.write_policy}; read_off={policy.memory_off}; "
          f"override={policy.min_fill_metadata}; checkpoint_sha256={policy.echo_checkpoint_sha256}; "
          f"sources={policy.min_fill_source_sha256}", flush=True)
    server = None
    try:
        server = PolicyServer(Gr00tSimPolicyWrapper(policy), host=args.host, port=args.port)
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        policy.reset()
        if server is not None:
            server.socket.close(linger=0)
            server.context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
