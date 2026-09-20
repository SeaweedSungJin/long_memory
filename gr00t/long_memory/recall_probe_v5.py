"""Small, frozen-representation probes, not a robot policy or success metric.

Every representation uses the same sampled endpoints, episode split, optimizer
schedule and hidden width. Input dimensions/parameter counts differ and are
reported explicitly. Mean pooling discards spatial/temporal layout: a failed
probe is NOT proof that a representation contains no relevant information.
Past-image features here are cached contextual VL image tokens, not raw RGB
patches before HAMLET's backbone. No label selects a past event.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from .cache import EpisodeCache, _atomic_json, _atomic_torch_save
from .cache_reader_v3 import MappedEpisodes
from .core_v3 import ActionValueMemory, MemoryV3Config
from .diagnostic_interventions import old_event_ids
from .recall_data_v5 import RecallLabels, sha256_file
from .replay_v3 import encode_until, read_bank, replay_bank
from .safety_v5 import validate_output_scope


REPRESENTATIONS = ("current_short", "past_moment", "past_image", "past_event", "actual_read")


def _image_pool(ep, d):
    mask = ep["image_masks"][d] & ep["attention_masks"][d]
    # Original short memory is not an image patch, even if a stale mask marks it.
    mask = mask.clone()
    mask[-ep["short"].shape[1]:] = False
    values = ep["features"][d][mask].float()
    if not values.numel():
        return ep["short"][d].float().mean(0).new_zeros(ep["short"].shape[-1]), False
    return values.mean(0), True


@torch.no_grad()
def frozen_probe_vectors(memory, ep, decision, *, window, policy="hard", encoded=None):
    """Read only finished valid events older than the complete short window."""
    if decision < 0 or decision >= len(ep["decision_mask"]) or not bool(ep["decision_mask"][decision]):
        raise ValueError("Probe needs an active nonterminal decision")
    encoded = encode_until(memory, ep, decision) if encoded is None else encoded
    # Causal history: a GT action only belongs to a completed past event here.
    old = old_event_ids(ep, decision, [i for i in range(decision) if bool(ep["transition_valid"][i])], window)
    short = ep["short"][decision].float().mean(0)
    hidden = memory.config.hidden_dim
    moment = torch.stack([ep["moment"][i + 1].float().mean(0) for i in old]).mean(0) if old else torch.zeros_like(short)
    image_rows = [_image_pool(ep, i + 1) for i in old]
    usable_images = [value for value, valid in image_rows if valid]
    image = torch.stack(usable_images).mean(0) if usable_images else torch.zeros_like(short)
    event = encoded["event"][old].float().mean(0).cpu() if old else short.new_zeros(hidden)
    bank, _ = replay_bank(memory, ep, decision, policy, encoded=encoded)
    read = read_bank(memory, ep, decision, bank, encoded=encoded)["read"].float().reshape(-1, hidden).mean(0).cpu()
    vectors = {"current_short": short, "past_moment": moment, "past_image": image,
               "past_event": event, "actual_read": read}
    if any(value.ndim != 1 or not torch.isfinite(value).all() for value in vectors.values()):
        raise ValueError("Nonfinite or invalid frozen probe feature")
    return vectors, {"old_ids": old, "bank_ids": bank, "past_image_available": bool(usable_images),
                     "available_old": bool(old)}


def select_probe_plan(cache, labels, *, samples_per_split=512, per_episode=4, seed=42):
    """Episode-round-robin, seeded selection; no task success or GT target value used."""
    if samples_per_split <= 0 or per_episode <= 0 or seed < 0:
        raise ValueError("Positive sample counts and a nonnegative seed are required")
    plan = []
    for offset, split in enumerate(("train", "val")):
        rng = random.Random(seed + offset)
        eids = list(cache.manifest["splits"][split])
        rng.shuffle(eids)
        candidates = {}
        for eid in eids:
            rows = labels._rows(int(eid))
            # Unknown vocabulary targets remain in the plan and coverage report;
            # they are masked in metrics, never silently converted into classes.
            choices = [r["decision"] for r in rows if r["active"]]
            rng.shuffle(choices)
            candidates[eid] = choices[:per_episode]
        selected = []
        for round_index in range(per_episode):
            for eid in eids:
                if len(candidates[eid]) > round_index:
                    selected.append({"episode_id": int(eid), "decision": candidates[eid][round_index], "split": split})
                if len(selected) >= samples_per_split:
                    break
            if len(selected) >= samples_per_split:
                break
        if not selected:
            raise ValueError(f"No active {split} probe queries")
        plan.extend(selected)
    return plan


@torch.no_grad()
def extract_probe_dataset(cache, labels, memory, plan, *, policy="hard"):
    mapped = MappedEpisodes(cache)
    features = {name: [] for name in REPRESENTATIONS}
    output_rows = []
    # Group for I/O efficiency without changing the explicit paired query plan.
    ordered = sorted(enumerate(plan), key=lambda x: (x[1]["episode_id"], x[1]["decision"]))
    pending = {}
    for index, row in ordered:
        eid, decision = row["episode_id"], row["decision"]
        ep = mapped.fetch(eid)
        values, extra = frozen_probe_vectors(memory, ep, decision, window=labels.manifest["memory_window"], policy=policy)
        target = labels.get(eid, decision)
        if target["split"] != row["split"] or target["available_old"] != extra["available_old"]:
            raise ValueError("Probe plan/recall age or split alignment mismatch")
        pending[index] = (values, {**target, **extra})
    for index in range(len(plan)):
        values, row = pending[index]
        output_rows.append(row)
        for name in REPRESENTATIONS:
            features[name].append(values[name].detach().cpu())
    return {"features": {name: torch.stack(rows) for name, rows in features.items()}, "rows": output_rows}


class RecallProbe(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU())
        self.classifier = nn.Linear(hidden, num_classes)
        self.grounding = nn.Linear(hidden, 2)

    def forward(self, features):
        z = self.body(features)
        return self.classifier(z), self.grounding(z).sigmoid()


def train_standardization(features, train_mask):
    """Only the training split fits centering/scaling, including constant dims."""
    if not bool(train_mask.any()):
        raise ValueError("No training examples")
    mean = features[train_mask].mean(0)
    scale = features[train_mask].std(0, unbiased=False).clamp_min(1e-4)
    return (features - mean) / scale, mean, scale


def masked_probe_loss(logits, xy_pred, classes, xy, class_valid, xy_valid):
    zero = logits.sum() * 0.0 + xy_pred.sum() * 0.0
    cls = F.cross_entropy(logits[class_valid], classes[class_valid]) if bool(class_valid.any()) else zero
    coord = F.smooth_l1_loss(xy_pred[xy_valid], xy[xy_valid]) if bool(xy_valid.any()) else zero
    return cls + coord


@torch.no_grad()
def probe_metrics(logits, xy_pred, rows, *, split, old_only=False):
    selected = [i for i, row in enumerate(rows) if row["split"] == split and (not old_only or row["available_old"])]
    classes = [i for i in selected if rows[i]["class_valid"]]
    grounded = [i for i in selected if rows[i]["xy_valid"]]
    correct = sum(int(logits[i].argmax()) == rows[i]["class_id"] for i in classes)
    xy_error = [float((xy_pred[i] - torch.tensor(rows[i]["xy"])).abs().mean()) for i in grounded]
    return {"queries": len(selected), "class_targets": len(classes), "unknown_or_invalid": len(selected) - len(classes),
            "class_accuracy": correct / len(classes) if classes else None,
            "xy_targets": len(grounded), "xy_mae_normalized": sum(xy_error) / len(xy_error) if xy_error else None}


def fit_probes(dataset, num_classes, *, epochs=100, hidden=64, lr=1e-3, batch_size=64, seed=42):
    if epochs <= 0 or hidden <= 0 or lr <= 0 or batch_size <= 0 or num_classes <= 0:
        raise ValueError("Probe optimizer/model settings must be positive")
    rows = dataset["rows"]
    train = torch.tensor([r["split"] == "train" for r in rows])
    valid = torch.tensor([r["class_valid"] for r in rows])
    xy_valid = torch.tensor([r["xy_valid"] for r in rows])
    classes = torch.tensor([r["class_id"] for r in rows], dtype=torch.long)
    xy = torch.tensor([r["xy"] for r in rows], dtype=torch.float32)
    train_indices = torch.where(train & valid)[0]
    if not len(train_indices):
        raise ValueError("No valid train-only recall classes in probe plan")
    report, states = {}, {}
    for name, feature in dataset["features"].items():
        if not torch.isfinite(feature).all():
            raise ValueError("Probe features contain nonfinite values")
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        x, mean, scale = train_standardization(feature.float(), train)
        model = RecallProbe(x.shape[1], num_classes, hidden)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        history = []
        for epoch in range(epochs):
            model.train()
            perm = train_indices[torch.randperm(len(train_indices), generator=generator)]
            losses = []
            for begin in range(0, len(perm), batch_size):
                idx = perm[begin:begin + batch_size]
                logits, predicted = model(x[idx])
                loss = masked_probe_loss(logits, predicted, classes[idx], xy[idx], valid[idx], xy_valid[idx])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite probe loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                history.append({"epoch": epoch + 1, "train_loss": sum(losses) / len(losses)})
        model.eval()
        with torch.no_grad():
            logits, predicted = model(x)
        report[name] = {"input_dim": x.shape[1], "parameters": sum(p.numel() for p in model.parameters()),
                        "history": history,
                        "metrics": {split + ("/old_available" if old else "/all"): probe_metrics(logits, predicted, rows, split=split, old_only=old)
                                    for split in ("train", "val") for old in (False, True)}}
        states[name] = {"state_dict": model.state_dict(), "feature_mean": mean, "feature_scale": scale}
    return report, states


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Existing v4 memory+AE bundle; only memory is instantiated")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-split", type=int, default=512)
    parser.add_argument("--per-episode", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="Frozen memory extraction only; little probes fit on CPU")
    parser.add_argument("--writer-policy", choices=("all", "hard"), default="hard")
    args = parser.parse_args(argv)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite probe results: {output}")
    from safetensors.torch import load_file
    from .checkpoint_v4 import v4_checkpoint_info

    cache = EpisodeCache(args.cache_dir)
    labels = RecallLabels(args.labels_dir, cache)
    checkpoint = Path(args.checkpoint).resolve()
    validate_output_scope(output, cache.path, labels.root, checkpoint,
                          cache.manifest["dataset_path"], cache.manifest["model_path"])
    info = v4_checkpoint_info(cache.manifest["model_path"], checkpoint)
    if info["metadata"]["cache_fingerprint"] != cache.manifest["fingerprint"]:
        raise ValueError("Probe checkpoint/cache fingerprints differ")
    memory = ActionValueMemory(MemoryV3Config(**info["config"]["memory"]))
    memory.load_state_dict(load_file(str(checkpoint / "model.safetensors"), device="cpu"), strict=True)
    memory = memory.to(args.device).eval().requires_grad_(False)
    plan = select_probe_plan(cache, labels, samples_per_split=args.samples_per_split,
                             per_episode=args.per_episode, seed=args.seed)
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "plan.json", plan)
    print(f"[recall-probe] extracting frozen features for {len(plan)} paired endpoints", flush=True)
    dataset = extract_probe_dataset(cache, labels, memory, plan, policy=args.writer_policy)
    _atomic_torch_save(output / "probe_features.pt", dataset)
    report, states = fit_probes(dataset, labels.manifest["num_classes"], epochs=args.epochs,
                                hidden=args.hidden, lr=args.learning_rate, batch_size=args.batch_size, seed=args.seed)
    provenance = {"arguments": vars(args), "label_fingerprint": labels.manifest["fingerprint"],
                  "cache_fingerprint": cache.manifest["fingerprint"],
                  "checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json"),
                  "memory_sha256": sha256_file(checkpoint / "model.safetensors")}
    _atomic_json(output / "summary.json", {"provenance": provenance, "representations": report,
                 "limitations": ["Not robot success; subgoal is not necessarily memory-dependent.",
                     "Available-old is an age proxy, not evidence that a query requires that history.",
                     "Mean pooling discards layout/order; probe failure does not prove information absence.",
                     "Shared samples/hidden width/training protocol; parameter counts differ with input dimensions.",
                     "Cached historical image tokens are contextual VL features, not raw patches."]})
    _atomic_torch_save(output / "probe_models.pt", states)
    for name, result in report.items():
        print(name, json.dumps(result["metrics"]["val/all"]))
    print(f"[recall-probe] report: {output / 'summary.json'}")
    return 0
