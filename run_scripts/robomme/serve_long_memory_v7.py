#!/usr/bin/env python3
"""Serve original HAMLET or a strict V7 recurrent actor/AE/CVOM bundle."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-checkpoint", help="V7 bundle; omit for original HAMLET")
    parser.add_argument("--write-policy", choices=("checkpoint", "update"), default="checkpoint")
    parser.add_argument("--memory-off", action="store_true", help="Same Stage-1 AE, READ bypass; WRITE continues")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args(argv)
    from gr00t.long_memory.online_policy_v7 import LongMemoryV7Policy
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    policy = LongMemoryV7Policy(args.base_model, args.memory_checkpoint, args.device,
                               write_policy=args.write_policy, memory_off=args.memory_off)
    print(f"[v7-server] stage={policy.stage}; mode={policy.mode}; write_policy={policy.write_policy}; "
          f"memory_off={policy.memory_off}; all inference weights frozen", flush=True)
    server = PolicyServer(Gr00tSimPolicyWrapper(policy), host=args.host, port=args.port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("[v7-server] stopped", flush=True)
    finally:
        policy.reset()
        server.socket.close(linger=0)
        server.context.term()


if __name__ == "__main__":
    main()
