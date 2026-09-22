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
    parser.add_argument("--writer-checkpoint")
    parser.add_argument("--cvom-admission", action="store_true",
                        help="Load a writer-only CVoM sidecar over the unchanged V19 parent")
    parser.add_argument("--cvom-fifo", action="store_true",
                        help="Validate the CVoM sidecar but bypass its controller for same-parent FIFO")
    parser.add_argument("--semantic-memory", action="store_true")
    parser.add_argument("--semantic-fifo", action="store_true",
                        help="Validate semantic manager weights but use same-capacity FIFO as its ablation")
    parser.add_argument("--feature-precision", choices=("native", "cache-aligned"), default="native",
                        help="Frozen feature extraction precision only; AE denoising is unchanged.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("Invalid TCP port")
    if args.semantic_fifo and not args.semantic_memory:
        raise ValueError("--semantic-fifo requires --semantic-memory")
    if args.semantic_memory and (args.writer_checkpoint is not None or args.feature_precision != "native"):
        raise ValueError("Semantic memory uses native precision and cannot also load a legacy writer")
    if args.cvom_fifo and not args.cvom_admission:
        raise ValueError("--cvom-fifo requires --cvom-admission")
    if args.cvom_admission and (args.writer_checkpoint is None or args.semantic_memory
                               or args.feature_precision != "native"):
        raise ValueError("CVoM admission requires its writer checkpoint, native precision, and no semantic manager")
    from run_scripts.robomme.policy_representation_v18 import RepresentationPolicyV18
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer
    policy = RepresentationPolicyV18(args.base_model, args.checkpoint, args.device, args.memory_off,
                                    writer_checkpoint=args.writer_checkpoint, feature_precision=args.feature_precision,
                                    semantic_memory=args.semantic_memory, semantic_fifo=args.semantic_fifo,
                                    cvom_admission=args.cvom_admission, cvom_fifo=args.cvom_fifo)
    print(f"[v18] representation={policy.representation.config.representation}; read_off={policy.memory_off}; "
          f"step={policy.checkpoint_step}; feature_precision={policy.feature_precision}; "
          f"feature_precision_rules={policy.feature_precision_rules}", flush=True)
    if args.semantic_memory:
        print(f"[semantic] stage={policy.stage}; write_policy={policy.write_policy}; "
              f"manifest_sha256={policy.semantic_manifest_sha256}; controller_sha256={policy.storage_manager_sha256}", flush=True)
    if args.cvom_admission:
        print(f"[cvom-admission] write_policy={policy.write_policy}; "
              f"parent={policy.cvom_parent_identity}; manifest_sha256={policy.cvom_manifest_sha256}; "
              f"writer_sha256={policy.cvom_writer_sha256}; sources={policy.cvom_source_sha256}", flush=True)
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
