#!/usr/bin/env python3
"""Serve a genuine V18 bundle; baseline uses the unchanged reviewed V10 server."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--memory-off", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("Invalid TCP port")
    from run_scripts.robomme.policy_representation_v18 import RepresentationPolicyV18
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer
    policy = RepresentationPolicyV18(args.base_model, args.checkpoint, args.device, args.memory_off)
    print(f"[v18] representation={policy.representation.config.representation}; read_off={policy.memory_off}; step={policy.checkpoint_step}", flush=True)
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
