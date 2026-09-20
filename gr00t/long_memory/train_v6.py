"""Isolated two-stage visual retrieval + read-time CVOM experiment.

Stage 1: ordered visual values, soft retrieval, direct AE bridges and AE LoRA.
Stage 2: identical frozen Stage-1 teacher, signed paired action-loss gains.
No human cue labels, future observations, learned eviction, or success labels.
See docs/LONG_MEMORY_V6_RETRIEVAL_CVOM.md for protocol and limitations.
"""
import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

import torch

from .cache import EpisodeCache
from .cache_reader_v3 import MappedEpisodes, validate_decision
from .checkpoint_v4 import file_sha256
from .checkpoint_v6 import (VARIANT, frozen_identity, load_checkpoint_v6,
                            save_checkpoint_v6, v6_checkpoint_info)
from .core_v6 import CANDIDATE_NAMES, VisualMemoryV6, VisualMemoryV6Config
from .expert_v4 import (LoRAConfig, adapter_disabled, expert_parameters,
                        install_expert_lora, set_expert_trainable)
from .expert_v6 import bridge_parameters, install_memory_bridge, v6_episode_flow_loss
from .hamlet import checkpoint_identity, load_frozen_hamlet, validate_cache_checkpoint
from .monitoring import RunLogger, _atomic_json
from .objectives_v6 import cvom_loss, paired_utility_targets, select_candidate
from .replay_v6 import prepare_candidates
from .train_v3 import _Tee, _grad_norm, _mean, _seed, runtime_identity


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=int, choices=(1, 2))
    p.add_argument('--cache-dir')
    p.add_argument('--output-dir', required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--init-checkpoint', help='Stage 2 requires a v6 Stage-1 memory reader')
    source.add_argument('--resume', help='Resume into a NEW output directory; max-steps is TOTAL updates')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-steps', type=int, default=2000)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--reader-mode', choices=('memory', 'none'), default='memory')
    p.add_argument('--reader-learning-rate', type=float, default=1e-4)
    p.add_argument('--bridge-learning-rate', type=float, default=1e-4)
    p.add_argument('--expert-learning-rate', type=float, default=1e-5)
    p.add_argument('--cvom-learning-rate', type=float, default=1e-4)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--visual-tokens', type=int, default=16)
    p.add_argument('--read-budget', type=int, default=16)
    p.add_argument('--max-archive-events', type=int, default=256)
    p.add_argument('--num-heads', type=int, default=4)
    p.add_argument('--temporal-layers', type=int, default=1)
    p.add_argument('--bridge-blocks', type=int, nargs='+', default=[7, 23])
    p.add_argument('--lora-rank', type=int, default=8)
    p.add_argument('--lora-alpha', type=float, default=16.)
    p.add_argument('--memory-dropout', type=float, default=.1)
    p.add_argument('--warmup-steps', type=int, default=50)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--max-grad-norm', type=float, default=1.)
    p.add_argument('--cvom-contexts', type=int, default=128)
    p.add_argument('--cvom-batch-size', type=int, default=4)
    p.add_argument('--context-refresh-steps', type=int, default=250)
    p.add_argument('--noise-samples', type=int, default=2)
    p.add_argument('--label-scale', type=float, default=.001)
    p.add_argument('--label-margin', type=float, default=1e-5)
    p.add_argument('--uncertainty-z', type=float, default=2.)
    p.add_argument('--ranking-weight', type=float, default=.1)
    p.add_argument('--fixed-policy', choices=('uniform',), default='uniform')
    p.add_argument('--cvom-threshold', type=float, default=.05,
                   help='Normalized gain improvement over fixed-policy, NOT a probability')
    p.add_argument('--val-samples', type=int, default=16)
    p.add_argument('--max-train-episodes', type=int, default=0)
    p.add_argument('--max-val-episodes', type=int, default=0)
    p.add_argument('--eval-steps', type=int, default=100)
    p.add_argument('--save-steps', type=int, default=250)
    p.add_argument('--log-steps', type=int, default=10)
    p.add_argument('--plot-steps', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--activation-checkpointing', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    return p


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    if args.resume:
        saved = json.loads((Path(args.resume) / 'checkpoint.json').read_text())['config']['train']
        explicit = {p._option_string_actions[x.split('=', 1)[0]].dest for x in argv
                    if x.split('=', 1)[0] in p._option_string_actions}
        for key, value in saved.items():
            if hasattr(args, key) and key not in explicit:
                setattr(args, key, value)
        args.init_checkpoint = None
    return args


def source_identity():
    """Hash the actual Python dependency closure, including dynamic model code."""
    root = Path(__file__).resolve().parents[2]
    pending = [Path(__file__).resolve(), root / 'run_scripts/robomme/train_long_memory_v6.py']
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                parent = path.parent
                for _ in range(node.level - 1):
                    parent = parent.parent
                target = parent / ((node.module or '').replace('.', '/') + '.py')
            elif node.module and node.module.startswith('gr00t.'):
                target = root / (node.module.replace('.', '/') + '.py')
            else:
                continue
            if target.is_file():
                pending.append(target.resolve())
    for directory in ('gr00t/model', 'gr00t/configs'):
        seen.update(path.resolve() for path in (root / directory).rglob('*.py'))
    return {str(path.relative_to(root)): file_sha256(path) for path in sorted(seen)}


def preflight(args):
    if args.stage not in (1, 2) or not args.cache_dir:
        raise ValueError('Provide --stage and --cache-dir, or inherit with --resume')
    if args.stage == 2 and not (args.init_checkpoint or args.resume):
        raise ValueError('Stage 2 needs --init-checkpoint from v6 Stage 1')
    if args.stage == 1 and args.init_checkpoint:
        raise ValueError('Stage 1 starts fresh on original HAMLET; use --resume to continue v6')
    for key in ('max_steps', 'grad_accum', 'cvom_contexts', 'cvom_batch_size', 'context_refresh_steps',
                'val_samples', 'eval_steps', 'save_steps', 'log_steps', 'plot_steps'):
        if getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive')
    for key in ('seed', 'warmup_steps', 'max_train_episodes', 'max_val_episodes'):
        if getattr(args, key) < 0:
            raise ValueError(f'{key} must be nonnegative')
    for key in ('reader_learning_rate', 'bridge_learning_rate', 'expert_learning_rate',
                'cvom_learning_rate', 'label_scale', 'max_grad_norm'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('weight_decay', 'label_margin', 'uncertainty_z', 'ranking_weight', 'cvom_threshold'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if args.noise_samples < 2 or not 0 <= args.memory_dropout < 1:
        raise ValueError('Require noise-samples >= 2 and memory-dropout in [0,1)')
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a NEW output directory; existing run is never overwritten: {output}')
    cache = EpisodeCache(args.cache_dir)
    manifest = cache.manifest
    validate_cache_checkpoint(manifest)
    base = str(Path(manifest['model_path']).resolve())
    base_cfg = json.loads((Path(base) / 'config.json').read_text())
    source = args.resume or args.init_checkpoint
    initial = v6_checkpoint_info(base, source) if source else None
    if initial:
        config = initial['config']
        if initial['metadata']['cache_fingerprint'] != manifest['fingerprint']:
            raise ValueError('Checkpoint and cache provenance differ')
        if args.resume:
            if config['stage'] != args.stage or args.max_steps <= initial['step']:
                raise ValueError('Resume must preserve stage and increase TOTAL max-steps')
            mutable = {'resume', 'init_checkpoint', 'output_dir', 'max_steps', 'preflight_only'}
            for key, value in vars(args).items():
                if key not in mutable and config['train'].get(key) != value:
                    raise ValueError(f'Exact resume option changed: {key}')
            if initial['metadata']['source_sha256'] != source_identity():
                raise ValueError('Training source changed; exact resume is not valid')
            if initial['metadata']['runtime'] != runtime_identity():
                raise ValueError('Runtime changed; exact resume is not valid')
        elif config['stage'] != 1 or config['reader_mode'] != 'memory':
            raise ValueError('CVOM requires a memory-enabled v6 Stage-1 teacher')
        memory_cfg = VisualMemoryV6Config(**config['memory'])
        expert_cfg = LoRAConfig(**config['expert'])
    else:
        memory_cfg = VisualMemoryV6Config(feature_dim=manifest['feature_dim'], state_dim=manifest['state_dim'],
            hidden_dim=args.hidden_dim, visual_tokens=args.visual_tokens, num_heads=args.num_heads,
            temporal_layers=args.temporal_layers, max_archive_events=args.max_archive_events,
            read_budget=args.read_budget, time_scale=float(base_cfg.get('memory_stride', 16)))
        expert_cfg = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha)
    if args.stage == 2 and args.reader_mode != 'memory':
        raise ValueError('Stage 2 cannot be an AE-only control')
    print(f'[v6 preflight] stage={args.stage} base={base}; cache={manifest["fingerprint"]}; '
          f'train/val episodes={len(manifest["splits"]["train"])}/{len(manifest["splits"]["val"])}', flush=True)
    return cache, base, memory_cfg, expert_cfg, initial


def make_plans(args, cache, episodes):
    """Uniform episode sampling, then uniform valid decision; no future targets used for selection."""
    plans = {'train': [], 'val': [], 'files': {}}
    records = {int(row['episode_id']): row for row in cache.manifest['episodes']}
    for split, limit in (('train', args.max_train_episodes), ('val', args.max_val_episodes)):
        ids = list(cache.manifest['splits'][split])
        if limit:
            ids = ids[:limit]
        for eid in ids:
            ep = episodes.fetch(eid)
            decisions = torch.nonzero(ep['decision_mask'], as_tuple=False).flatten().tolist()
            # Stage 2 needs some past evidence. Decision 0 cannot train a read critic.
            if args.stage == 2:
                decisions = [d for d in decisions if d > 0]
            if decisions:
                plans[split].append([int(eid), decisions])
            path = (cache.path / records[int(eid)]['path']).resolve()
            stat = path.stat()
            plans['files'][str(eid)] = [str(path), stat.st_size, stat.st_mtime_ns]
        if not plans[split]:
            raise ValueError(f'No usable decisions in {split}')
    rng = random.Random(args.seed + 701)
    plans['validation'] = [sample_context(plans['val'], rng) for _ in range(args.val_samples)]
    return plans


def sample_context(pool, rng=random):
    eid, decisions = rng.choice(pool)
    return [eid, rng.choice(decisions)]


def refresh_contexts(args, plans, step):
    # Independent RNG makes refresh plans insensitive to label-cache hits.
    block = max(step - 1, 0) // args.context_refresh_steps
    rng = random.Random(args.seed + 100003 * block + 991)
    return [sample_context(plans['train'], rng) for _ in range(args.cvom_contexts)]


def _paired_seed(args, eid, decision, draw, kind):
    value = json.dumps([args.seed, eid, decision, draw, kind]).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], 'little') % (2**31)


def flow(head, ep, decision, tokens=None, **kwargs):
    validate_decision(ep, decision)
    return v6_episode_flow_loss(head, ep, decision, tokens, **kwargs)


class FixedTeacherLabels:
    """Only signed same-query action-loss differences; cache is tied to frozen payloads."""
    def __init__(self, args, output, teacher_identity):
        self.args, self.output = args, Path(output) / 'labels'
        self.identity = teacher_identity

    @torch.no_grad()
    def get(self, head, ep, decision, prepared):
        eid = int(ep['episode_id'])
        candidate_ids = {name: pack['event_ids'] for name, pack in prepared['candidates'].items()}
        provenance = {'teacher': self.identity, 'episode': eid, 'decision': decision,
                      'candidate_ids': candidate_ids, 'noise_samples': self.args.noise_samples,
                      'seed': self.args.seed}
        key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        path = self.output / f'{key}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['provenance'] != provenance:
                raise ValueError('Counterfactual label provenance mismatch')
            draws = saved['loss_draws']
        else:
            draws = []
            for repeat in range(self.args.noise_samples):
                same_seed = _paired_seed(self.args, eid, decision, repeat, 'teacher')
                # Identical sets are the same intervention, not separate noisy labels.
                costs, row = {}, []
                for name in CANDIDATE_NAMES:
                    pack = prepared['candidates'][name]
                    ids = tuple(pack['event_ids'])
                    if ids not in costs:
                        costs[ids] = float(flow(head, ep, decision, pack['tokens'], seed=same_seed)['loss'])
                    row.append(costs[ids])
                draws.append(row)
            self.output.mkdir(parents=True, exist_ok=True)
            _atomic_json(path, {'provenance': provenance, 'loss_draws': draws})
        return paired_utility_targets(draws, scale=self.args.label_scale,
            margin=self.args.label_margin, uncertainty_z=self.args.uncertainty_z)


@torch.no_grad()
def validate(args, memory, head, episodes, plans):
    """Fresh held-out noise, paired across roles; action loss is NOT task success."""
    rows = []
    for eid, decision in plans['validation']:
        ep = episodes.fetch(eid)
        prepared = prepare_candidates(memory, ep, decision)
        packs = prepared['candidates']
        seed = _paired_seed(args, eid, decision, 0, 'heldout-validation')
        fixed = args.fixed_policy
        if args.stage == 2:
            predictions = memory.score_candidates(prepared['query'], packs)
            selected = select_candidate(predictions, args.cvom_threshold, fixed)
        else:
            selected = fixed if args.reader_mode == 'memory' else 'null'
        result = flow(head, ep, decision, packs[selected]['tokens'], seed=seed)
        fixed_loss = float(flow(head, ep, decision, packs[fixed]['tokens'], seed=seed)['loss'])
        no_memory = float(flow(head, ep, decision, seed=seed)['loss'])
        with adapter_disabled(head):
            baseline = float(flow(head, ep, decision, seed=seed)['loss'])
        loss = float(result['loss'])
        rows.append({'action_loss': loss, 'loss': loss, 'velocity_mae': float(result['velocity_mae']),
                     'baseline_action_loss': baseline, 'expert_only_action_loss': no_memory,
                     'fixed_action_loss': fixed_loss, 'memory_gain': no_memory - loss,
                     'cvom_gain_over_fixed': fixed_loss - loss,
                     'selected_null_rate': float(selected == 'null'),
                     'selection_change_rate': float(selected != fixed),
                     'archive_count': prepared['details']['archive_count'],
                     'retrieved_events': len(packs[selected]['event_ids'])})
    return _mean(rows)


def configure_training(args, memory, head):
    memory.eval()  # No dropout/random state in paired interventions.
    reader_on = args.stage == 1 and args.reader_mode == 'memory'
    for parameter in memory.reader_parameters():
        parameter.requires_grad_(reader_on)
    for parameter in memory.cvom_parameters():
        parameter.requires_grad_(args.stage == 2)
    set_expert_trainable(head, args.stage == 1)
    for parameter in bridge_parameters(head):
        parameter.requires_grad_(reader_on)
    if args.stage == 1:
        groups = [{'params': list(expert_parameters(head)), 'lr': args.expert_learning_rate}]
        if reader_on:
            groups += [{'params': list(memory.reader_parameters()), 'lr': args.reader_learning_rate},
                       {'params': list(bridge_parameters(head)), 'lr': args.bridge_learning_rate}]
    else:
        groups = [{'params': list(memory.cvom_parameters()), 'lr': args.cvom_learning_rate}]
    for group in groups:
        group['base_lr'] = group['lr']
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def run(args, checked):
    cache, base, memory_cfg, expert_cfg, initial = checked
    _seed(args.seed)
    output = Path(args.output_dir).resolve()
    episodes = MappedEpisodes(cache)
    plans = make_plans(args, cache, episodes)
    if args.resume and plans != initial['metadata']['plans']:
        raise ValueError('Selected cache files or data plan changed; exact resume rejected')
    model, processor = load_frozen_hamlet(base, args.device)
    head = model.action_head
    del model, processor  # Frozen VL features are already cached; only keep AE on GPU.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    targets = install_expert_lora(head, expert_cfg,
                                  targets=initial['config']['expert_targets'] if initial else None)
    bridge_config = install_memory_bridge(head, **initial['config']['bridge']) if initial else install_memory_bridge(
        head, hidden_dim=memory_cfg.hidden_dim, block_indices=args.bridge_blocks, num_heads=memory_cfg.num_heads)
    memory = VisualMemoryV6(memory_cfg).to(args.device)
    optimizer = configure_training(args, memory, head)
    if initial:
        load_checkpoint_v6(args.resume or args.init_checkpoint, memory, head,
                           optimizer=optimizer if args.resume else None)
    config = {'trainer_variant': VARIANT, 'stage': args.stage, 'reader_mode': args.reader_mode,
              'memory': asdict(memory_cfg), 'expert': asdict(expert_cfg), 'expert_targets': targets,
              'bridge': bridge_config, 'train': vars(args).copy()}
    metadata = {'base_model': checkpoint_identity(base), 'cache_fingerprint': cache.manifest['fingerprint'],
                'source_sha256': source_identity(), 'runtime': runtime_identity(), 'plans': plans}
    if args.stage == 2:
        frozen = frozen_identity(memory, head)
        metadata['frozen_identity'] = frozen
        metadata['stage1_parent'] = initial['metadata']['stage1_parent'] if args.resume else {
            'path': str(Path(args.init_checkpoint).resolve()),
            'checkpoint_sha256': file_sha256(Path(args.init_checkpoint) / 'checkpoint.json'),
            'frozen_identity': frozen}
    label_identity = {key: metadata[key] for key in ('cache_fingerprint', 'source_sha256')}
    label_identity['frozen_identity'] = metadata.get('frozen_identity')
    labels = FixedTeacherLabels(args, output, label_identity)
    logger = RunLogger(output)
    _atomic_json(output / 'config.json', config)
    start_step = initial['step'] if args.resume else 0
    best = float(initial['metadata']['best_val']) if args.resume else float('inf')
    best_reference = initial['metadata']['best_checkpoint'] if args.resume else None
    elapsed_before = initial['metadata'].get('elapsed_seconds', 0.) if args.resume else 0.
    started = time.monotonic()
    trainable = [parameter for group in optimizer.param_groups for parameter in group['params']]
    print(f'[v6] stage={args.stage}; trainable={sum(p.numel() for p in trainable):,}; '
          f'output={output}; original HAMLET is immutable', flush=True)

    def evaluate(step):
        metrics = validate(args, memory, head, episodes, plans)
        logger.log(step, 'val', metrics)
        print(f'[v6][val] step={step} action_loss={metrics["action_loss"]:.7f} '
              f'baseline={metrics["baseline_action_loss"]:.7f} '
              f'expert_only={metrics["expert_only_action_loss"]:.7f} '
              f'fixed={metrics["fixed_action_loss"]:.7f}', flush=True)
        return metrics['action_loss']

    def save(step, improved=False):
        nonlocal best_reference
        if improved:
            best_reference = {'path': str(output / f'checkpoint-{step:06d}'), 'step': step}
        metadata['best_checkpoint'] = best_reference
        if args.stage == 2 and frozen_identity(memory, head) != metadata['frozen_identity']:
            raise RuntimeError('Stage-2 fixed teacher changed unexpectedly')
        metadata['best_val'] = best
        metadata['elapsed_seconds'] = elapsed_before + time.monotonic() - started
        checkpoint = save_checkpoint_v6(output, step, memory, head, optimizer, config, metadata, best=improved)
        _atomic_json(output / 'last_checkpoint.json', {'path': checkpoint.name, 'step': step})
        print(f'[v6] saved {checkpoint}', flush=True)

    if not args.resume:
        best = evaluate(0)
        save(0, True)
        logger.plot()
    rows = []
    for step in range(start_step + 1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * min(1., step / max(args.warmup_steps, 1))
        if args.stage == 1:
            for _ in range(args.grad_accum):
                eid, decision = sample_context(plans['train'])
                ep = episodes.fetch(eid)
                prepared = prepare_candidates(memory, ep, decision) if args.reader_mode == 'memory' else None
                # Consume identical policy-choice RNG in the AE-only training control.
                drop = random.random() < args.memory_dropout
                candidate = random.choice(CANDIDATE_NAMES[:-1])
                name = 'null' if args.reader_mode == 'none' or drop else candidate
                tokens = prepared['candidates'][name]['tokens'] if prepared else None
                result = flow(head, ep, decision, tokens, activation_checkpointing=args.activation_checkpointing)
                if not torch.isfinite(result['loss']):
                    raise FloatingPointError('Nonfinite Stage-1 loss')
                (result['loss'] / args.grad_accum).backward()
                rows.append({'loss': float(result['loss'].detach()), 'action_loss': float(result['loss'].detach()),
                             'velocity_mae': float(result['velocity_mae'].detach()),
                             'null_rate': float(name == 'null'),
                             'retrieved_events': len(prepared['candidates'][name]['event_ids']) if prepared else 0})
        else:
            contexts = refresh_contexts(args, plans, step)
            for _ in range(args.cvom_batch_size):
                eid, decision = random.choice(contexts)
                ep = episodes.fetch(eid)
                with torch.no_grad():
                    prepared = prepare_candidates(memory, ep, decision)
                    target = labels.get(head, ep, decision, prepared)
                predicted = memory.score_candidates(prepared['query'], prepared['candidates'])
                result = cvom_loss(predicted, target, args.ranking_weight)
                if not torch.isfinite(result['loss']):
                    raise FloatingPointError('Nonfinite CVOM loss')
                (result['loss'] / args.cvom_batch_size).backward()
                rows.append({'loss': float(result['loss'].detach()), 'cvom_regression': float(result['regression'].detach()),
                             'cvom_ranking': float(result['ranking'].detach()),
                             'ranking_signal_fraction': result['signal_fraction'],
                             'target_gain_abs': float(target['raw_gain'].abs().mean()),
                             'distinct_candidate_sets': len({tuple(p['event_ids']) for p in prepared['candidates'].values()}),
                             'nonnull_set_diversity': len({tuple(prepared['candidates'][n]['event_ids']) for n in CANDIDATE_NAMES[:-1]}) / 3.})
        grad_norm = _grad_norm(trainable)
        if not math.isfinite(grad_norm):
            raise FloatingPointError('Nonfinite gradient; checkpoint not published')
        reader_grad = _grad_norm(list(memory.reader_parameters()))
        bridge_grad = _grad_norm(list(bridge_parameters(head)))
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        log_now = step % args.log_steps == 0 or step % args.save_steps == 0 or step % args.eval_steps == 0 or step == args.max_steps
        if log_now:
            metrics = _mean(rows)
            metrics.update(grad_norm=grad_norm, reader_grad_norm=reader_grad, bridge_grad_norm=bridge_grad,
                           learning_rate=optimizer.param_groups[0]['lr'],
                           elapsed_seconds=elapsed_before + time.monotonic() - started)
            logger.log(step, 'train', metrics)
            print(f'[v6][train] step={step} loss={metrics["loss"]:.7f} grad={grad_norm:.5f} '
                  f'reader_grad={reader_grad:.5f} bridge_grad={bridge_grad:.5f}', flush=True)
            rows = []
        improved = False
        if step % args.eval_steps == 0 or step == args.max_steps:
            loss = evaluate(step)
            if loss < best:
                best, improved = loss, True
        if improved or step % args.save_steps == 0 or step == args.max_steps:
            save(step, improved)
        if step % args.plot_steps == 0 or step == args.max_steps:
            logger.plot()
    # A resumed fork may not beat its parent; keep that parent explicitly as best.
    if args.resume and not (output / 'best_checkpoint.json').exists():
        _atomic_json(output / 'best_checkpoint.json', dict(best_reference,
                     note='No improvement over inherited best validation checkpoint'))
    print(f'[v6] complete. Best validation checkpoint: {output / "best_checkpoint.json"}', flush=True)
    return output


def main(argv=None):
    args = parse_args(argv)
    checked = preflight(args)
    if args.preflight_only:
        print('[v6 preflight] No model loaded, no outputs created, no training started.', flush=True)
        return
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / 'training.log').open('a', buffering=1) as log:
            with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
                try:
                    return run(args, checked)
                except BaseException:
                    traceback.print_exc()
                    raise


if __name__ == '__main__':
    main()
