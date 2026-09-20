#!/usr/bin/env python3
"""Serve original HAMLET or its trained episodic-memory adapter, without edits to either checkpoint."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-checkpoint", default=None, help="Omit for the original frozen HAMLET baseline")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()
    from gr00t.long_memory.online_policy import LongMemoryPolicy
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer
    policy = LongMemoryPolicy(args.base_model, args.memory_checkpoint, args.device)
    print(f"[eval-server] stage={policy.stage}, write_policy={policy.write_policy}; all weights frozen", flush=True)
    server = PolicyServer(Gr00tSimPolicyWrapper(policy), host=args.host, port=args.port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("[eval-server] stopped", flush=True)
    finally:
        policy.reset()
        server.socket.close(linger=0)
        server.context.term()


if __name__ == "__main__":
    main()
