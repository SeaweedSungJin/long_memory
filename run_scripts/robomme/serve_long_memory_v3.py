#!/usr/bin/env python3
"""Serve frozen HAMLET or a new action-value v3 memory-only checkpoint."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-checkpoint", help="Omit for the unchanged frozen HAMLET baseline")
    parser.add_argument("--write-policy", choices=("checkpoint", "all"), default="checkpoint",
                        help="all is the explicit same-checkpoint FIFO writer ablation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args(argv)
    from gr00t.long_memory.online_policy_v3 import LongMemoryV3Policy
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    policy = LongMemoryV3Policy(args.base_model, args.memory_checkpoint, args.device,
                                write_policy=args.write_policy)
    print(f"[v3-server] stage={policy.stage}; writer={policy.write_policy}; "
          f"override={args.write_policy}; all weights frozen", flush=True)
    server = PolicyServer(Gr00tSimPolicyWrapper(policy), host=args.host, port=args.port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("[v3-server] stopped", flush=True)
    finally:
        policy.reset()
        server.socket.close(linger=0)
        server.context.term()


if __name__ == "__main__":
    main()
