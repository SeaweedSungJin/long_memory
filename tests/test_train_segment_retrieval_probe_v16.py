"""Temporary synthetic CPU runs only; no real dataset, policy or GPU execution."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from run_scripts.robomme import train_segment_retrieval_probe_v16 as trainer
from run_scripts.robomme import train_segment_retrieval_probe_v15 as old
from run_scripts.robomme.segment_probe_inputs_v16 import configure_scope, build_probe_inputs_v16
from run_scripts.robomme.segment_probe_inputs_v15 import remap_positive_frames
from tests import test_visual_demo_tail_bank_v13 as fixture


def examples():
    return [{"episode_id": eid, "decision": query, "split": split}
            for eid, split in ((7, "train"), (8, "train"), (9, "val"), (10, "val"))
            for query in (3, 4)]


class TinyInputs:
    def __init__(self, cache, sidecar, rows):
        self.examples = {(r['episode_id'], r['decision']): r for r in rows}
        self.observations = fixture.source()
        self.payload, self.record = fixture.sidecar()

    def get(self, model, row, permutation_seed=None):
        current, bank, info = build_probe_inputs_v16(model, self.observations, row['decision'],
            sidecar=self.payload, record=self.record, episode_id=7,
            cache_fingerprint=fixture.CACHE, sidecar_fingerprint=fixture.SIDECAR,
            permutation_seed=permutation_seed)
        # Labels enter only AFTER the new observation-only builder returns.
        positives = set(remap_positive_frames([0, 16], info))
        positive = torch.tensor([[frame in positives for frame in info['candidate_frames']]])
        return current, bank, positive, info


def target_rows():
    return [{**r, 'query_frame': 48 if r['decision'] == 3 else 64, 'n_demo': 48,
             'positive_frames': [0, 16], 'old_positive_frames': [0],
             'candidate_frames': [0, 16, 32, *range(33, 48)] + ([48] if r['decision'] == 4 else [])}
            for r in examples()]


class QuietLogger:
    def __init__(self, output):
        self.output = Path(output)

    def log(self, step, split, values):
        with (self.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': step, 'split': split, **values}) + '\n')

    def plot(self):
        pass


class V16TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.init = self.root / 'initial'
        self.init.mkdir()
        self.model = fixture.memory(awake=True)
        save_file({k: v.detach().contiguous() for k, v in self.model.state_dict().items()},
                  str(self.init / 'visual.safetensors'))
        self.input_file = self.root / 'protected.json'
        self.input_file.write_text('{}')

    def args(self, arm='qkp', output='run', steps=2):
        return trainer.build_parser().parse_args(['--arm', arm, '--targets', str(self.root/'targets.json'),
            '--cache-dir', str(self.root/'cache'), '--sidecar-dir', str(self.root/'sidecar'),
            '--init-checkpoint', str(self.init), '--output-dir', str(self.root/output),
            '--max-steps', str(steps), '--eval-steps', str(steps), '--batch-size', '2', '--device', 'cpu'])

    def context(self, args):
        plan = trainer.make_plan(target_rows(), args.max_steps, args.batch_size, args.seed)
        protocol = {'kind': trainer.KIND, 'arm': args.arm,
            'train': {k:v for k,v in vars(args).items() if k not in ('output_dir','preflight_only')},
            'selected_parameter_names': list(trainer.selected_names(args.arm)),
            'files_sha256': {str(self.input_file): trainer.file_hash(self.input_file),
                             str(self.init/'visual.safetensors'): trainer.file_hash(self.init/'visual.safetensors')},
            'plan_sha256': trainer.digest(plan), 'fixed_final_selection': True,
            'permutation_seed': args.seed + 1}
        return {'output': Path(args.output_dir), 'protocol': protocol, 'plan': plan, 'init': self.init,
                'initial': {'config': {'visual': asdict(self.model.config)}},
                'cache': None, 'sidecar': None, 'targets': {'examples': target_rows()},
                'selected': {'train':{7,8},'val':{9,10}}}

    def optimizer(self, model, arm):
        return torch.optim.AdamW(configure_scope(model, train_projection=arm=='qkp'), lr=1e-4, weight_decay=0.)

    def test_cli_scope_required_and_matched_defaults(self):
        for arm, count in (('qk',2),('qkp',3)):
            args = trainer.build_parser().parse_args(['--arm',arm,'--targets','t','--cache-dir','c',
                '--sidecar-dir','s','--init-checkpoint','i','--output-dir','o'])
            self.assertEqual((args.max_steps,args.batch_size,args.eval_steps,args.learning_rate,args.seed),
                             (256,4,64,1e-4,9151))
            self.assertEqual(len(trainer.selected_names(arm)),count)
            self.assertNotIn('image_projection.bias',trainer.selected_names(arm))
        with self.assertRaises(ValueError):
            trainer.selected_names('all')
        self.assertIs(trainer.make_plan, old.make_plan)
        self.assertIs(trainer.evaluate, old.evaluate)
        self.assertIs(trainer.update, old.update)
        self.assertIs(trainer.assess, old.assess)

    def test_both_actual_scopes_update_connected_only_and_keep_all_buffers(self):
        original = {name: value.detach().clone() for name,value in self.model.named_parameters()}
        for arm in trainer.ARMS:
            model=copy.deepcopy(self.model); optimizer=self.optimizer(model,arm)
            snapshot=trainer.frozen_snapshot(model,arm)
            inputs=TinyInputs(None,None,target_rows())
            for step in (1,2):
                metrics=trainer.update(model,optimizer,target_rows()[:2],inputs,1.)
                self.assertTrue(torch.isfinite(torch.tensor(metrics['loss'])))
                self.assertTrue(trainer.assert_scope(model,optimizer,arm,snapshot,step,require_gradients=True))
            changed={name for name,p in model.named_parameters() if not torch.equal(p,original[name])}
            self.assertEqual(changed,set(trainer.selected_names(arm)))
            self.assertEqual(len(optimizer.state),2 if arm=='qk' else 3)
            self.assertEqual(set(snapshot['buffers']),{'patch_cameras','patch_rows','patch_columns'})
            self.assertTrue(all(torch.equal(v,snapshot['buffers'][n]) for n,v in model.named_buffers()))
        self.assertTrue(all(torch.equal(p,original[n]) for n,p in self.model.named_parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_scope_rejects_buffer_bias_flags_optimizer_ownership_and_nonfinite(self):
        def setup():
            model=copy.deepcopy(self.model); opt=self.optimizer(model,'qkp')
            return model,opt,trainer.frozen_snapshot(model,'qkp')
        model,opt,snapshot=setup(); model.patch_rows[0]+=1
        with self.assertRaisesRegex(ValueError,'buffer'):
            trainer.assert_scope(model,opt,'qkp',snapshot,0)
        model,opt,snapshot=setup()
        with torch.no_grad(): model.image_projection.bias[0]+=1
        with self.assertRaisesRegex(ValueError,'Frozen parameter'):
            trainer.assert_scope(model,opt,'qkp',snapshot,0)
        model,opt,snapshot=setup(); model.value_projection.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError,'scope'):
            trainer.assert_scope(model,opt,'qkp',snapshot,0)
        model,opt,snapshot=setup(); opt.param_groups[0]['params'].reverse()
        with self.assertRaisesRegex(ValueError,'identity/order'):
            trainer.assert_scope(model,opt,'qkp',snapshot,0)
        model,opt,snapshot=setup(); opt.param_groups[0]['weight_decay']=.01
        with self.assertRaisesRegex(ValueError,'weight-decay'):
            trainer.assert_scope(model,opt,'qkp',snapshot,0)
        model,opt,snapshot=setup()
        with torch.no_grad(): model.query_projection.weight[0,0]=float('nan')
        with self.assertRaises(FloatingPointError):
            trainer.assert_scope(model,opt,'qkp',snapshot,0)

    def test_checkpoint_atomic_schema_finite_state_names_hashes_and_no_overwrite(self):
        args=self.args(); context=self.context(args); output=context['output']; output.mkdir()
        model=copy.deepcopy(self.model); opt=self.optimizer(model,args.arm)
        snapshot=trainer.frozen_snapshot(model,args.arm)
        first=trainer.save_checkpoint(output,0,model,opt,context['protocol'],snapshot)
        before=trainer.file_hash(first/'probe.safetensors')
        for step in (1,2):
            trainer.update(model,opt,target_rows()[:2],TinyInputs(None,None,target_rows()),1.)
        final=trainer.save_checkpoint(output,2,model,opt,context['protocol'],snapshot)
        header=json.loads((final/'checkpoint.json').read_text())
        state=torch.load(final/'optimizer.pt',map_location='cpu',weights_only=True)
        self.assertEqual(header['kind'],trainer.KIND)
        self.assertEqual(header['arm'],'qkp'); self.assertFalse(header['deployable_policy'])
        self.assertEqual(header['selected_parameter_names'],list(trainer.selected_names('qkp')))
        self.assertEqual(header['protocol_sha256'],trainer.digest(context['protocol']))
        self.assertEqual(header['payload_sha256'],trainer.file_hash(final/'probe.safetensors'))
        self.assertEqual(header['optimizer_sha256'],trainer.file_hash(final/'optimizer.pt'))
        self.assertEqual(header['frozen_buffers'],trainer.buffer_signatures(snapshot))
        self.assertEqual(len(state['optimizer']['state']),3)
        self.assertTrue(all(v['step'].item()==2 for v in state['optimizer']['state'].values()))
        self.assertEqual(trainer.file_hash(first/'probe.safetensors'),before)
        with self.assertRaises(FileExistsError):
            trainer.save_checkpoint(output,2,model,opt,context['protocol'],snapshot)
        self.assertEqual(trainer.file_hash(first/'probe.safetensors'),before)

    def test_checkpoint_rejects_bad_protocol_nonfinite_state_or_changed_source(self):
        args=self.args(); context=self.context(args); output=context['output']; output.mkdir()
        model=copy.deepcopy(self.model); opt=self.optimizer(model,'qkp'); snapshot=trainer.frozen_snapshot(model,'qkp')
        for mutate in (lambda p:p.update(kind='segment_retrieval_probe_v15'),
                       lambda p:p.update(arm='qk'),lambda p:p['train'].update(arm='qk'),
                       lambda p:p.update(selected_parameter_names=['image_projection.bias'])):
            bad=copy.deepcopy(context['protocol']); mutate(bad)
            with self.assertRaises(ValueError): trainer.save_checkpoint(output,0,model,opt,bad,snapshot)
        self.assertFalse(any(output.iterdir()))
        trainer.update(model,opt,target_rows()[:2],TinyInputs(None,None,target_rows()),1.)
        next(iter(opt.state.values()))['exp_avg'][0,0]=float('inf')
        with self.assertRaises(FloatingPointError):
            trainer.save_checkpoint(output,1,model,opt,context['protocol'],snapshot)
        self.assertFalse((output/'checkpoint-000001').exists())
        self.input_file.write_text('CHANGED')
        with self.assertRaisesRegex(ValueError,'Immutable'):
            trainer.save_checkpoint(output,1,model,opt,context['protocol'],snapshot)

    def test_failed_payload_publication_preserves_previous_checkpoint_and_evidence(self):
        args=self.args(); context=self.context(args); output=context['output']; output.mkdir()
        model=copy.deepcopy(self.model); opt=self.optimizer(model,'qkp'); snapshot=trainer.frozen_snapshot(model,'qkp')
        first=trainer.save_checkpoint(output,0,model,opt,context['protocol'],snapshot)
        digest=trainer.file_hash(first/'probe.safetensors')
        trainer.update(model,opt,target_rows()[:2],TinyInputs(None,None,target_rows()),1.)
        with patch.object(trainer.torch,'save',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                trainer.save_checkpoint(output,1,model,opt,context['protocol'],snapshot)
        self.assertFalse((output/'checkpoint-000001').exists())
        self.assertTrue(list(output.glob('.checkpoint-000001-*')))
        self.assertEqual(trainer.file_hash(first/'probe.safetensors'),digest)

    def test_two_arm_cpu_main_loops_same_initial_plan_validation_and_protected_parent(self):
        initial_hash=trainer.file_hash(self.init/'visual.safetensors')
        runs=[]
        for arm in trainer.ARMS:
            args=self.args(arm,output=arm); context=self.context(args)
            with patch.object(trainer,'Inputs',TinyInputs),patch.object(trainer,'RunLogger',QuietLogger):
                self.assertEqual(trainer.run(args,context),0)
            status=json.loads((context['output']/'status.json').read_text())
            self.assertEqual((status['status'],status['kind'],status['arm'],status['step'],status['processed_queries']),
                             ('complete',trainer.KIND,arm,2,4))
            self.assertFalse(status['policy_ready'])
            self.assertEqual(trainer.file_hash(self.init/'visual.safetensors'),initial_hash)
            runs.append(context['output'])
        for name in ('query_plan.json','validation-000000.json','checkpoint-000000/probe.safetensors'):
            self.assertEqual((runs[0]/name).read_bytes(),(runs[1]/name).read_bytes(),name)
        for arm,folder in zip(trainer.ARMS,runs):
            initial=load_file(str(folder/'checkpoint-000000/probe.safetensors'))
            final=load_file(str(folder/'checkpoint-000002/probe.safetensors'))
            self.assertEqual({n for n in initial if not torch.equal(initial[n],final[n])},set(trainer.selected_names(arm)))
        self.assertFalse(torch.cuda.is_initialized())

    def test_failure_status_is_explicit_and_does_not_publish_final(self):
        args=self.args(); context=self.context(args)
        with patch.object(trainer,'Inputs',TinyInputs),patch.object(trainer,'RunLogger',QuietLogger),\
                patch.object(trainer,'update',side_effect=RuntimeError('broken gradient')):
            with self.assertRaisesRegex(RuntimeError,'broken gradient'):
                trainer.run(args,context)
        status=json.loads((context['output']/'status.json').read_text())
        self.assertEqual(status['status'],'failed'); self.assertFalse(status['policy_ready'])
        self.assertEqual(status['integrity_error'],None)
        self.assertTrue((context['output']/'checkpoint-000000/checkpoint.json').exists())
        self.assertFalse((context['output']/'checkpoint-000002').exists())

    def test_preflight_main_never_allocates_model_gpu_or_output(self):
        args=self.args(); context=self.context(args)
        argv=['--arm','qkp','--targets','t','--cache-dir','c','--sidecar-dir','s',
              '--init-checkpoint','i','--output-dir',str(context['output']),'--preflight-only']
        with patch.object(trainer,'preflight',return_value=context),\
                patch.object(trainer,'run',side_effect=AssertionError('No run')),\
                patch.object(trainer,'VisualDemoTailMemoryV13',side_effect=AssertionError('No model')):
            self.assertEqual(trainer.main(argv),0)
        self.assertFalse(context['output'].exists())
        self.assertFalse(torch.cuda.is_initialized())

    def test_binding_guards_arm_independent_paths_splits_base_finalstep_and_tail(self):
        cache=SimpleNamespace(path=self.root/'cache',manifest={'fingerprint':'c','splits':{'train':[7,8],'val':[9,10]},
                              'model_path':str(self.root/'base')})
        sidecar=SimpleNamespace(path=self.root/'sidecar',manifest={'fingerprint':'s'})
        target={'identity':{'cache_dir':str(cache.path),'sidecar_dir':str(sidecar.path),
                            'cache_fingerprint':'c','sidecar_fingerprint':'s','original_splits':cache.manifest['splits'],
                            'base_model':str(self.root/'base')},'examples':target_rows()}
        initial={'step':512,'config':{'include_tail':True,'read_mode':'differential'},
                 'metadata':{'base_model':{'path':str(self.root/'base')},'cache_fingerprint':'c',
                             'sidecar':{'fingerprint':'s'}}}
        self.assertEqual(trainer.validate_bindings(target,cache,sidecar,initial),{'train':{7,8},'val':{9,10}})
        for mutate in (lambda t,i:t['identity'].update(cache_fingerprint='bad'),
                       lambda t,i:t['identity'].update(original_splits={'train':[9],'val':[7]}),
                       lambda t,i:t['identity'].update(base_model=str(self.root/'other')),
                       lambda t,i:t['examples'][0].update(split='test'),
                       lambda t,i:i.update(step=0),lambda t,i:i['config'].update(include_tail=False),
                       lambda t,i:i['config'].update(include_tail=1),lambda t,i:i['config'].update(read_mode='current_only'),
                       lambda t,i:i['metadata'].update(cache_fingerprint='bad')):
            bad_target,bad_initial=copy.deepcopy(target),copy.deepcopy(initial)
            mutate(bad_target,bad_initial)
            with self.assertRaises(ValueError): trainer.validate_bindings(bad_target,cache,sidecar,bad_initial)


if __name__=='__main__':
    unittest.main()
