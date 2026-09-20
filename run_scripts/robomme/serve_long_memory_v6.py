#!/usr/bin/env python3
"""Serve original HAMLET or the v6 temporal visual archive + adapted Expert."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-checkpoint", help="V6 complete bundle; omit for original HAMLET")
    parser.add_argument("--read-policy", choices=("checkpoint", "uniform"), default="checkpoint")
    parser.add_argument("--expert-only", action="store_true", help="Same adapted Expert, memory off at inference")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args(argv)
    from gr00t.long_memory.online_policy_v6 import LongMemoryV6Policy
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    policy = LongMemoryV6Policy(args.base_model, args.memory_checkpoint, args.device,
                               read_policy=args.read_policy, expert_only=args.expert_only)
    print(f"[v6-server] stage={policy.stage}; retrieval={policy.read_policy}; "
          f"expert_only={policy.expert_only}; all inference weights frozen", flush=True)
    server = PolicyServer(Gr00tSimPolicyWrapper(policy), host=args.host, port=args.port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("[v6-server] stopped", flush=True)
    finally:
        policy.reset()
        server.socket.close(linger=0)
        server.context.term()


if __name__ == "__main__":
    main()
