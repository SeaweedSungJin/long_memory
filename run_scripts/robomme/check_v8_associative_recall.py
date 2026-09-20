#!/usr/bin/env python3
"""Bounded CPU associative-recall learning check, not RoboMME evaluation.

Each fresh episode contains four randomly ordered, unique one-hot keys with
independent random four-bit values. A later query contains a requested key and
zero value channels. Its supervised target is the value of the corresponding
past event. The target/index is never supplied to the memory or decoder.
"""

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from gr00t.long_memory.event_v8 import EventMemoryV8, MemoryV8Config
from gr00t.long_memory.monitoring import _atomic_json


KEYS, BITS, EVENTS, TOKENS = 8, 4, 4, 4
FEATURES = KEYS + BITS + 4


def sample_cases(batch, generator):
    # Independent random permutations prevent requested key from identifying
    # a fixed slot or age. Values are freshly sampled independently of keys.
    keys = torch.rand(batch, KEYS, generator=generator).argsort(-1)[:, :EVENTS]
    values = torch.randint(0, 2, (batch, EVENTS, BITS), generator=generator).float() * 2 - 1
    requested_event = torch.randint(EVENTS, (batch,), generator=generator)
    rows = torch.arange(batch)
    requested_key = keys[rows, requested_event]
    target = values[rows, requested_event]
    return dict(keys=keys, values=values, query_key=requested_key, target=target,
                requested_event=requested_event)


def observation_inputs(cases, values=None):
    keys = cases["keys"]
    batch = keys.shape[0]
    source = torch.zeros(batch, EVENTS, TOKENS, FEATURES)
    source[..., :KEYS] = F.one_hot(keys, KEYS).float()[:, :, None]
    source[..., KEYS:KEYS + BITS] = (cases["values"] if values is None else values)[:, :, None]
    short = torch.zeros(batch, TOKENS, FEATURES)
    short[..., :KEYS] = F.one_hot(cases["query_key"], KEYS).float()[:, None]
    # The requested value is absent from every current-observation channel.
    assert not bool(short[..., KEYS:KEYS + BITS].any())
    return source, short


def prediction(memory, decoder, cases, *, mode="event", values=None):
    source, short = observation_inputs(cases, values)
    batch = source.shape[0]
    state = torch.zeros(batch, memory.config.state_dim)
    frames = torch.arange(EVENTS).expand(batch, -1) * 16
    encoded = memory.encode(source.reshape(batch * EVENTS, TOKENS, FEATURES),
                            state[:, None].expand(-1, EVENTS, -1).reshape(batch * EVENTS, -1),
                            frames.reshape(-1), torch.zeros(batch * EVENTS, dtype=torch.bool))
    # Equivalent whole-event append order, without recurrent write dependence.
    bank = encoded.reshape(batch, EVENTS * TOKENS, memory.config.hidden_dim + 2)
    current = memory.encode(short, state, torch.full((batch,), EVENTS * 16),
                            torch.zeros(batch, dtype=torch.bool))
    fused, metrics = memory.read(short, current, bank, mode=mode)
    # Decode the memory contribution only. Scaling by its known bound avoids
    # requiring a decoder with very large weights merely because fusion is 0.1.
    scale = memory.config.residual_scale * short.square().mean(-1, keepdim=True).sqrt()
    contribution = ((fused - short) / scale.clamp_min(1e-12)).mean(1)
    return decoder(contribution), metrics


def scores(output, target):
    return dict(bit_accuracy=float(((output > 0) == (target > 0)).float().mean()),
                exact_vector_accuracy=float(((output > 0) == (target > 0)).all(-1).float().mean()),
                mse=float(F.mse_loss(output, target)))


@torch.no_grad()
def evaluate(memory, decoder, cases, wrong_values, batch_size):
    memory.eval()
    decoder.eval()
    outputs = {name: [] for name in ("correct_bank", "replaced_values", "no_memory")}
    for begin in range(0, len(cases["target"]), batch_size):
        end = begin + batch_size
        part = {name: value[begin:end] for name, value in cases.items()}
        for role in outputs:
            values = wrong_values[begin:end] if role == "replaced_values" else None
            mode = "none" if role == "no_memory" else "event"
            output, _ = prediction(memory, decoder, part, mode=mode, values=values)
            outputs[role].append(output)
    outputs = {name: torch.cat(values) for name, values in outputs.items()}
    metrics = {name: scores(output, cases["target"]) for name, output in outputs.items()}
    rows = torch.arange(len(cases["target"]))
    counterfactual_target = wrong_values[rows, cases["requested_event"]]
    # If replacement changes recall correctly, accuracy should remain high
    # against the replacement bank's answer while dropping against the original.
    metrics["replaced_values_against_replacement_target"] = scores(outputs["replaced_values"], counterfactual_target)
    return metrics, outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/verification/v8_associative_recall_20260916")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=105.)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-episodes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=8408)
    parser.add_argument("--learning-rate", type=float, default=.003)
    args = parser.parse_args(argv)
    if not 1 <= args.updates <= 1500 or not 0 < args.max_seconds <= 115:
        raise ValueError("Require 1..1500 updates and a training bound no greater than 115 seconds")
    if min(args.batch_size, args.validation_episodes) < 1 or args.seed < 0 or not 0 < args.learning_rate <= .1:
        raise ValueError("Invalid sample count, seed, or learning rate")
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to((ROOT / "runs/verification").resolve()) or output.exists():
        raise ValueError("Use a NEW directory under runs/verification; prior artifacts are preserved")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(args.seed)
    seeds = dict(initialization=args.seed, training_data=args.seed + 1,
                 validation_data=args.seed + 2, replacement_values=args.seed + 3)
    train_generator = torch.Generator().manual_seed(seeds["training_data"])
    validation = sample_cases(args.validation_episodes, torch.Generator().manual_seed(seeds["validation_data"]))
    wrong_values = torch.randint(0, 2, validation["values"].shape,
        generator=torch.Generator().manual_seed(seeds["replacement_values"])).float() * 2 - 1
    config = MemoryV8Config(feature_dim=FEATURES, state_dim=3, num_short_tokens=TOKENS,
                           hidden_dim=32, num_heads=4, capacity=8, source="moment")
    memory = EventMemoryV8(config)
    decoder = nn.Sequential(nn.Linear(FEATURES, 32), nn.SiLU(), nn.Linear(32, BITS))
    optimizer = torch.optim.AdamW(list(memory.parameters()) + list(decoder.parameters()),
                                  lr=args.learning_rate, weight_decay=0)
    source_paths = [Path(__file__).resolve(), ROOT / "gr00t/long_memory/event_v8.py"]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    protocol = dict(args=vars(args), seeds=seeds, memory=asdict(config), source_sha256=hashes,
                    torch_version=torch.__version__, device="cpu", num_threads=2, num_interop_threads=2,
                    task=dict(keys=KEYS, bits=BITS, events=EVENTS, target_values=[-1, 1]),
                    decoder="Linear(16,32), SiLU, Linear(32,4); memory contribution divided by fusion bound",
                    loss="MSE against four independent +/-1 target bits; no reconstruction auxiliary",
                    selection="Fixed update/time budget; held-out scores never select checkpoints or updates",
                    note="Synthetic associative recall only; not flow-loss or RoboMME success evidence")
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "config.json", protocol)
    torch.save(dict(cases=validation, replacement_values=wrong_values), output / "validation_cases.pt")
    initial, _ = evaluate(memory, decoder, validation, wrong_values, args.batch_size)
    start = time.monotonic()
    history = []
    for step in range(1, args.updates + 1):
        memory.train()
        decoder.train()
        cases = sample_cases(args.batch_size, train_generator)
        estimate, diagnostics = prediction(memory, decoder, cases)
        loss = F.mse_loss(estimate, cases["target"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Synthetic learning loss became nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(memory.parameters()) + list(decoder.parameters()), 5.)
        optimizer.step()
        elapsed = time.monotonic() - start
        if step == 1 or step % 100 == 0 or step == args.updates or elapsed >= args.max_seconds:
            entry = dict(step=step, elapsed_seconds=elapsed, train_mse=float(loss.detach()),
                         train_bit_accuracy=scores(estimate.detach(), cases["target"])["bit_accuracy"],
                         read_entropy=float(diagnostics["read_entropy"].detach()))
            history.append(entry)
            _atomic_json(output / "training_history.json", history)
            print(json.dumps(entry), flush=True)
        if elapsed >= args.max_seconds:
            break
    final, predictions = evaluate(memory, decoder, validation, wrong_values, args.batch_size)
    summary = dict(completed_updates=step, training_seconds=time.monotonic() - start,
                   stopped_by_time=step < args.updates, initial=initial, final=final,
                   queries=args.validation_episodes, bits_per_query=BITS,
                   note="Fresh held-out synthetic episodes; no robot actions, simulator, or production checkpoint used")
    _atomic_json(output / "summary.json", summary)
    torch.save(dict(memory=memory.state_dict(), decoder=decoder.state_dict(), optimizer=optimizer.state_dict(),
                    train_generator_state=train_generator.get_state(), completed_updates=step), output / "trained_state.pt")
    torch.save(predictions, output / "validation_predictions.pt")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
