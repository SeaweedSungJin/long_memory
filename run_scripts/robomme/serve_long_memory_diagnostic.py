#!/usr/bin/env python3
"""Separate frozen diagnostic policy server; never modifies training weights."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-checkpoint")
    parser.add_argument("--mode", choices=("baseline", "full", "expert-only", "no-old", "shuffled-old", "fifo"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args(argv)
    from gr00t.long_memory.diagnostic_online import make_diagnostic_policy
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    policy = make_diagnostic_policy(args.base_model, args.memory_checkpoint, args.device, args.mode)
    print(f"[diagnostic-server] mode={args.mode}; stage={policy.stage}; frozen weights", flush=True)
    server = PolicyServer(Gr00tSimPolicyWrapper(policy), host=args.host, port=args.port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("[diagnostic-server] stopped", flush=True)
    finally:
        policy.reset()
        server.socket.close(linger=0)
        server.context.term()


if __name__ == "__main__":
    main()
