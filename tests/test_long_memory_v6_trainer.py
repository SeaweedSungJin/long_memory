"""Independent bounded CPU trainer checks; no full-model training is launched."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gr00t.long_memory import train_v6 as train
from gr00t.long_memory.core_v6 import CANDIDATE_NAMES, VisualMemoryV6, VisualMemoryV6Config
from gr00t.long_memory.expert_v4 import expert_parameters
from gr00t.long_memory.expert_v6 import bridge_parameters
from gr00t.long_memory.objectives_v6 import cvom_loss
from gr00t.long_memory.replay_v6 import prepare_candidates
from test_long_memory_v6_expert import Head, installed


def args(**changes):
    result = train.parse_args(['--stage', '1', '--cache-dir', 'dummy', '--output-dir', 'unused'])
    for key, value in changes.items():
        setattr(result, key, value)
    return result


def memory():
    return VisualMemoryV6(VisualMemoryV6Config(feature_dim=8, state_dim=4, hidden_dim=8,
        visual_tokens=2, num_heads=2, max_archive_events=8, read_budget=2))


def episode(n=7):
    short = torch.randn(n, 2, 8)
    return {'episode_id': 12, 'frames': torch.arange(n) * 16, 'short': short,
            'features': [torch.cat((torch.randn(6, 8), short[i])) for i in range(n)],
            'state': torch.randn(n, 4), 'image_masks': [torch.ones(8, dtype=torch.bool) for _ in range(n)],
            'attention_masks': [torch.ones(8, dtype=torch.bool) for _ in range(n)],
            'is_demo': torch.arange(n) < 2}


def prepared():
    return {'query': torch.zeros(1, 8), 'candidates': {
        'uniform': {'event_ids': [0], 'tokens': torch.ones(1, 2, 8)},
        'relevant': {'event_ids': [0], 'tokens': torch.ones(1, 2, 8)},
        'hybrid': {'event_ids': [1], 'tokens': torch.full((1, 2, 8), 2.)},
        'null': {'event_ids': [], 'tokens': None}}}


class TeacherLabelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.calls = []

    def fake_flow(self, head, ep, decision, tokens, *, seed):
        self.calls.append((seed, float(tokens.mean()) if tokens is not None else 0.))
        # A local generator imitates flow's paired seeded randomness without
        # perturbing the training RNG. Two candidate sets share the same noise.
        noise = torch.rand((), generator=torch.Generator().manual_seed(seed))
        return {'loss': noise + 5 - (float(tokens.mean()) * .01 if tokens is not None else 0.)}

    def test_noise_is_paired_across_choices_and_duplicate_sets_reuse_loss(self):
        labeler = train.FixedTeacherLabels(args(noise_samples=3), self.directory.name, {'reader': 'same'})
        with patch.object(train, 'flow', side_effect=self.fake_flow):
            targets = labeler.get(None, {'episode_id': 5}, 7, prepared())
        self.assertEqual(len(self.calls), 3 * 3)  # Two distinct packs + null, three draws.
        for start in range(0, len(self.calls), 3):
            self.assertEqual(len({seed for seed, _ in self.calls[start:start + 3]}), 1)
        self.assertEqual(len({seed for seed, _ in self.calls}), 3)
        self.assertEqual(float(targets['raw_gain'][0]), float(targets['raw_gain'][1]))
        self.assertAlmostEqual(float(targets['raw_gain'][2]), .02, places=5)
        self.assertEqual(float(targets['raw_gain'][3]), 0.)

    def test_cache_hits_do_not_change_targets_or_global_rng(self):
        random.seed(48)
        torch.manual_seed(48)
        python_before, torch_before = random.getstate(), torch.get_rng_state().clone()
        labeler = train.FixedTeacherLabels(args(), self.directory.name, {'reader': 'same'})
        with patch.object(train, 'flow', side_effect=self.fake_flow):
            first = labeler.get(None, {'episode_id': 5}, 7, prepared())
        with patch.object(train, 'flow', side_effect=AssertionError('cache hit recomputed teacher')):
            second = labeler.get(None, {'episode_id': 5}, 7, prepared())
        torch.testing.assert_close(first['gain'], second['gain'])
        self.assertEqual(python_before, random.getstate())
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))

    def test_changed_teacher_seed_or_candidate_set_invalidates_cache(self):
        with patch.object(train, 'flow', side_effect=self.fake_flow):
            train.FixedTeacherLabels(args(), self.directory.name, {'reader': 'a'}).get(None, {'episode_id': 5}, 7, prepared())
            train.FixedTeacherLabels(args(), self.directory.name, {'reader': 'b'}).get(None, {'episode_id': 5}, 7, prepared())
            train.FixedTeacherLabels(args(seed=49), self.directory.name, {'reader': 'a'}).get(None, {'episode_id': 5}, 7, prepared())
            changed = prepared()
            changed['candidates']['hybrid']['event_ids'] = [2]
            train.FixedTeacherLabels(args(), self.directory.name, {'reader': 'a'}).get(None, {'episode_id': 5}, 7, changed)
        self.assertEqual(len(list((Path(self.directory.name) / 'labels').glob('*.json'))), 4)

    def test_cached_draws_recalibrate_without_teacher_recomputation(self):
        with patch.object(train, 'flow', side_effect=self.fake_flow):
            first = train.FixedTeacherLabels(args(label_scale=.001), self.directory.name, {}).get(None, {'episode_id': 5}, 7, prepared())
        with patch.object(train, 'flow', side_effect=AssertionError('raw draws can be reused')):
            second = train.FixedTeacherLabels(args(label_scale=.002), self.directory.name, {}).get(None, {'episode_id': 5}, 7, prepared())
        torch.testing.assert_close(first['gain'] / 2, second['gain'])
        torch.testing.assert_close(first['raw_gain'], second['raw_gain'])

    def test_tampered_label_provenance_rejected(self):
        labeler = train.FixedTeacherLabels(args(), self.directory.name, {})
        with patch.object(train, 'flow', side_effect=self.fake_flow):
            labeler.get(None, {'episode_id': 5}, 7, prepared())
        path = next((Path(self.directory.name) / 'labels').glob('*.json'))
        record = json.loads(path.read_text())
        record['provenance']['episode'] = 9
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            labeler.get(None, {'episode_id': 5}, 7, prepared())

    def test_heldout_noise_is_distinct_from_teacher_noise(self):
        settings = args()
        teacher = [train._paired_seed(settings, 7, 8, n, 'teacher') for n in range(5)]
        heldout = train._paired_seed(settings, 7, 8, 0, 'heldout-validation')
        self.assertNotIn(heldout, teacher)


class FreezeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(72)
        torch.set_num_threads(1)

    def test_stage1_memory_trains_only_reader_bridge_and_lora(self):
        mem, head = memory(), installed()
        optimizer = train.configure_training(args(), mem, head)
        expected = {id(p) for p in mem.reader_parameters()} | {id(p) for p in expert_parameters(head)} | {id(p) for p in bridge_parameters(head)}
        actual = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertEqual(expected, actual)
        self.assertTrue(all(p.requires_grad for group in optimizer.param_groups for p in group['params']))
        self.assertFalse(any(p.requires_grad for p in mem.cvom_parameters()))
        self.assertFalse(head.training)
        self.assertFalse(mem.training)
        self.assertTrue(all(not p.requires_grad for p in head.parameters() if id(p) not in expected))

    def test_stage1_ae_only_control_freezes_reader_cvom_and_bridge(self):
        mem, head = memory(), installed()
        optimizer = train.configure_training(args(reader_mode='none'), mem, head)
        self.assertEqual({id(p) for group in optimizer.param_groups for p in group['params']},
                         {id(p) for p in expert_parameters(head)})
        self.assertFalse(any(p.requires_grad for p in mem.parameters()))
        self.assertFalse(any(p.requires_grad for p in bridge_parameters(head)))

    def test_stage2_optimizer_updates_cvom_only_and_teacher_is_bitwise_frozen(self):
        mem, head = memory(), installed()
        optimizer = train.configure_training(args(stage=2), mem, head)
        teacher_before = {k: p.clone() for k, p in mem.state_dict().items() if not k.startswith('cvom.')}
        head_before = {k: p.clone() for k, p in head.state_dict().items()}
        critic_before = {k: p.clone() for k, p in mem.cvom.state_dict().items()}
        with torch.no_grad():
            packs = prepare_candidates(mem, episode(), 5)
        labels = train.paired_utility_targets([[1, 2, 3, 2]] * 2, scale=1.)
        prediction = mem.score_candidates(packs['query'], packs['candidates'])
        loss = cvom_loss(prediction, labels)['loss']
        loss.backward()
        optimizer.step()
        self.assertTrue(all(torch.equal(p, mem.state_dict()[k]) for k, p in teacher_before.items()))
        self.assertTrue(all(torch.equal(p, head.state_dict()[k]) for k, p in head_before.items()))
        self.assertTrue(any(not torch.equal(p, mem.cvom.state_dict()[k]) for k, p in critic_before.items()))
        self.assertFalse(any(p.requires_grad for p in head.parameters()))
        self.assertEqual({id(p) for group in optimizer.param_groups for p in group['params']},
                         {id(p) for p in mem.cvom_parameters()})


class PlanAndArgumentTests(unittest.TestCase):
    def test_actual_stage1_loop_matches_control_context_and_noise_sequence(self):
        """Run the real loop with tiny modules/flow, not a copied sampling expression."""
        def execute(mode, directory):
            settings = args(reader_mode=mode, device='cpu', output_dir=directory,
                max_steps=5, grad_accum=3, eval_steps=5, save_steps=5, plot_steps=5,
                log_steps=5, bridge_blocks=[0, 1], num_heads=2, hidden_dim=8)
            cfg = VisualMemoryV6Config(feature_dim=8, state_dim=4, hidden_dim=8,
                visual_tokens=2, num_heads=2, read_budget=2, max_archive_events=8)
            expert_cfg = train.LoRAConfig(rank=2, alpha=4.)
            data = {eid: dict(episode(), episode_id=eid) for eid in range(4)}
            plans = {'train': [[0, [2, 3, 4]], [1, [2, 3, 4]]],
                     'val': [[2, [2, 3]], [3, [2, 3]]], 'validation': [[2, 3]], 'files': {}}
            sequence, checkpoints = [], []

            def fake_flow(head, ep, decision, tokens=None, **unused):
                random_noise = torch.rand(())
                sequence.append((ep['episode_id'], decision, float(random_noise)))
                loss = sum(p.square().mean() for p in expert_parameters(head)) + random_noise
                if tokens is not None:
                    loss = loss + tokens.square().mean() * .001
                return {'loss': loss, 'velocity_mae': loss.detach()}

            def save(output, step, memory_module, head, optimizer, config, metadata, *, best):
                checkpoints.append((step, copy.deepcopy(metadata)))
                return Path(output) / f'checkpoint-{step:06d}'

            Path(directory).mkdir()
            logger = SimpleNamespace(log=lambda *args: None, plot=lambda: None)
            with ExitStack() as stack:
                stack.enter_context(patch.object(train, 'MappedEpisodes', return_value=SimpleNamespace(fetch=lambda eid: data[eid])))
                stack.enter_context(patch.object(train, 'make_plans', return_value=plans))
                stack.enter_context(patch.object(train, 'load_frozen_hamlet', side_effect=lambda *a: (SimpleNamespace(action_head=Head().eval().requires_grad_(False)), None)))
                stack.enter_context(patch.object(train, 'source_identity', return_value={'source': 'same'}))
                stack.enter_context(patch.object(train, 'runtime_identity', return_value={'runtime': 'same'}))
                stack.enter_context(patch.object(train, 'checkpoint_identity', return_value={'path': 'base'}))
                stack.enter_context(patch.object(train, 'RunLogger', return_value=logger))
                stack.enter_context(patch.object(train, 'save_checkpoint_v6', side_effect=save))
                stack.enter_context(patch.object(train, 'flow', side_effect=fake_flow))
                stack.enter_context(patch.object(train, 'validate', return_value={
                    'action_loss': 1., 'baseline_action_loss': 1., 'expert_only_action_loss': 1., 'fixed_action_loss': 1.}))
                stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
                stack.enter_context(redirect_stdout(io.StringIO()))
                train.run(settings, (SimpleNamespace(manifest={'fingerprint': 'cache'}), 'base', cfg, expert_cfg, None))
            self.assertEqual(checkpoints[-1][1]['best_checkpoint']['step'], 0)
            self.assertTrue(checkpoints[-1][1]['best_checkpoint']['path'].endswith('checkpoint-000000'))
            return sequence

        with tempfile.TemporaryDirectory() as directory:
            memory_sequence = execute('memory', str(Path(directory) / 'memory'))
            control_sequence = execute('none', str(Path(directory) / 'control'))
        self.assertEqual(len(memory_sequence), 15)
        self.assertEqual(memory_sequence, control_sequence)

    def test_fixed_validation_plans_do_not_consume_training_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for eid in range(4):
                (root / f'{eid}.pt').write_bytes(b'fixture')
                records.append({'episode_id': eid, 'path': f'{eid}.pt'})
            cache = SimpleNamespace(path=root, manifest={'episodes': records, 'splits': {'train': [0, 1], 'val': [2, 3]}})
            episodes = SimpleNamespace(fetch=lambda eid: {'decision_mask': torch.tensor([True, False, True, True])})
            before = random.getstate()
            memory_plans = train.make_plans(args(val_samples=9), cache, episodes)
            control_plans = train.make_plans(args(val_samples=9, reader_mode='none'), cache, episodes)
            self.assertEqual(memory_plans, control_plans)
            self.assertEqual(before, random.getstate())
            self.assertTrue(all(eid in (2, 3) for eid, _ in memory_plans['validation']))
            self.assertTrue(all(d in (0, 2, 3) for _, d in memory_plans['validation']))
            stage2 = train.make_plans(args(stage=2, val_samples=50), cache, episodes)
            self.assertTrue(all(d > 0 for _, decisions in stage2['train'] for d in decisions))
            self.assertTrue(all(d > 0 for _, d in stage2['validation']))

    def test_context_refresh_is_repeatable_and_independent_of_global_rng(self):
        settings = args(cvom_contexts=25, context_refresh_steps=10)
        plans = {'train': [[i, list(range(1, 12))] for i in range(12)]}
        first = train.refresh_contexts(settings, plans, 1)
        for _ in range(50):
            random.random()
            torch.rand(1)
        self.assertEqual(first, train.refresh_contexts(settings, plans, 10))
        self.assertNotEqual(first, train.refresh_contexts(settings, plans, 11))

    def test_resume_inherits_unspecified_options_but_preserves_explicit_ones(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            saved = vars(args(stage=2, cache_dir='saved-cache', max_steps=1000, grad_accum=7,
                              noise_samples=4, hidden_dim=64, reader_learning_rate=.0003))
            (path / 'checkpoint.json').write_text(json.dumps({'config': {'train': saved}}))
            result = train.parse_args(['--resume', str(path), '--output-dir', 'new-run', '--max-steps', '2000'])
            self.assertEqual(result.stage, 2)
            self.assertEqual(result.cache_dir, 'saved-cache')
            self.assertEqual(result.max_steps, 2000)
            self.assertEqual(result.grad_accum, 7)
            self.assertEqual(result.noise_samples, 4)
            self.assertEqual(result.output_dir, 'new-run')
            self.assertIsNone(result.init_checkpoint)

    def test_incompatible_init_and_resume_are_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train.parse_args(['--resume', 'a', '--init-checkpoint', 'b', '--output-dir', 'new'])

    def test_invalid_stage_and_label_settings_fail_before_loading_cache(self):
        for settings in (args(stage=2), args(noise_samples=1), args(memory_dropout=1),
                         args(label_scale=0), args(cvom_learning_rate=float('nan'))):
            with patch.object(train, 'EpisodeCache', side_effect=AssertionError('validation must run first')):
                with self.assertRaises(ValueError):
                    train.preflight(settings)


if __name__ == '__main__':
    unittest.main()
