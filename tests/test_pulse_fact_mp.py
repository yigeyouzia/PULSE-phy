"""MPIC checks: python -B -m unittest discover -s tests -p test_pulse_fact_mp.py -v."""

import ast
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from models import PULSE, PULSE_FACT, PULSE_FACT_MP
import test_pulse_fact as source_tests


ROOT = Path(__file__).resolve().parents[1]
SCOPES = {'ETTh1': (96, 720), 'ETTh2': (720,), 'PEMS04': (96,)}


def config(**overrides):
    values = dict(seq_len=24, pred_len=12, enc_in=7, d_model=8, inv_len=24,
                  patch_size=6, time_dim=1, dsa=1, dsb=1, ksize=3,
                  use_amp=False, use_multi_gpu=False, use_gpu=False, label_len=0,
                  model='PULSE_FACT_MP', data='ETTh1', features='M', output_attention=False,
                  add_noise=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def make(module=PULSE_FACT_MP, args=None):
    with redirect_stdout(io.StringIO()):
        return module.Model(args or config())


def batch(args, device='cpu'):
    return (torch.randn(2, args.seq_len, args.enc_in, device=device),
            torch.arange(2, device=device),
            torch.randn(2, args.seq_len, args.time_dim, device=device),
            torch.randn(2, args.pred_len, args.time_dim, device=device))


class MPICModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(2024)
        np.random.seed(2024)

    def test_projection_preserves_each_sample_variable_mean_and_fact_fluctuation(self):
        model = make()
        for dtype in (torch.float32, torch.float64):
            base = torch.randn(3, 12, 7, dtype=dtype) + torch.arange(21, dtype=dtype).reshape(3, 1, 7)
            fact = torch.randn_like(base) + 10
            result = model.preserve_base_prediction_mean(base, fact)
            torch.testing.assert_close(result.mean(1), base.mean(1))
            torch.testing.assert_close(result-result.mean(1, keepdim=True),
                                       fact-fact.mean(1, keepdim=True), rtol=1e-5, atol=3e-6)
            # A per-sample/variable constant in FACT cannot change the output.
            shift = torch.randn(3, 1, 7, dtype=dtype)
            torch.testing.assert_close(model.preserve_base_prediction_mean(base, fact+shift), result)
        with self.assertRaises(ValueError):
            model.preserve_base_prediction_mean(base[:, :-1], fact[:, :-1])

    def test_projection_routes_level_and_fluctuation_gradients(self):
        model = make()
        base = torch.randn(2, 12, 7, dtype=torch.double, requires_grad=True)
        fact = torch.randn_like(base, requires_grad=True)
        direction = torch.randn(2, 1, 7, dtype=torch.double)
        output = model.preserve_base_prediction_mean(base, fact)
        ((output.mean(1, keepdim=True))*direction).sum().backward()
        torch.testing.assert_close(base.grad, direction.expand_as(base)/12, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(fact.grad, torch.zeros_like(fact), rtol=0, atol=1e-15)
        base.grad = fact.grad = None
        direction = torch.randn_like(base)
        direction = direction-direction.mean(1, keepdim=True)
        (model.preserve_base_prediction_mean(base, fact)*direction).sum().backward()
        torch.testing.assert_close(base.grad, torch.zeros_like(base), rtol=0, atol=1e-15)
        torch.testing.assert_close(fact.grad, direction, rtol=1e-12, atol=1e-12)

    def test_original_parameters_and_methods_preserved_without_extra_parameters(self):
        torch.manual_seed(2024)
        source = make(PULSE_FACT)
        torch.manual_seed(2024)
        model = make()
        self.assertEqual(sum(v.numel() for v in source.parameters()), sum(v.numel() for v in model.parameters()))
        for name, tensor in source.state_dict().items():
            if torch.is_tensor(tensor):
                torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)
        def classes(path):
            return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)}
        old, current = classes(ROOT/'models/PULSE_FACT.py'), classes(ROOT/'models/PULSE_FACT_MP.py')
        self.assertEqual(ast.dump(old['ReVIN']), ast.dump(current['ReVIN']))
        for name in ('norm', 'phase_router', 'denorm'):
            methods = [{node.name: node for node in value['Model'].body if isinstance(node, ast.FunctionDef)}
                       for value in (old, current)]
            self.assertEqual(ast.dump(methods[0][name]), ast.dump(methods[1][name]))

    def test_fact_prediction_matches_source_eval_train_and_sam(self):
        for training, mixing in ((False, False), (True, False), (True, True)):
            with self.subTest(training=training, mixing=mixing):
                torch.manual_seed(2024)
                source = make(PULSE_FACT).train(training)
                torch.manual_seed(2024)
                model = make().train(training)
                inputs = batch(config())
                torch.manual_seed(41)
                np.random.seed(17)
                expected = source(*inputs, do_mixup=mixing)
                torch.manual_seed(41)
                np.random.seed(17)
                with patch.object(model, 'preserve_base_prediction_mean', wraps=model.preserve_base_prediction_mean) as projection:
                    actual = model(*inputs, do_mixup=mixing)
                base, fact = projection.call_args[0]
                torch.testing.assert_close(fact, expected[0] if training else expected, rtol=0, atol=0)
                output = actual[0] if training else actual
                torch.testing.assert_close(output.mean(1), base.mean(1))
                if training:
                    if mixing:
                        self.assertEqual(actual[1][0], expected[1][0])
                        torch.testing.assert_close(actual[1][1], expected[1][1], rtol=0, atol=0)
                    else:
                        self.assertIsNone(actual[1])
                for name, value in source.named_buffers():
                    torch.testing.assert_close(dict(model.named_buffers())[name], value, rtol=0, atol=0)

    def test_sam_once_and_shared_statistics_for_both_restored_predictions(self):
        model = make().train()
        x, cycle, xm, ym = batch(config())
        original, anchor, mean, var = model.norm(x, cycle, model.time_enc_x(xm))
        sigma = torch.sqrt(var+1e-5)
        captured = []
        hook = model.backbone[0].register_forward_pre_hook(lambda module, values: captured.append(values[0]))
        permutation = torch.tensor([1, 0])
        try:
            with patch('models.PULSE_FACT_MP.np.random.beta', return_value=0.3) as beta, \
                    patch('models.PULSE_FACT_MP.torch.randperm', return_value=permutation) as shuffle, \
                    patch.object(model, 'denorm', wraps=model.denorm) as restore:
                output, (lam, index) = model(x, cycle, xm, ym)
            self.assertEqual((beta.call_count, shuffle.call_count, len(captured), restore.call_count), (1,1,1,2))
            torch.testing.assert_close(captured[0].transpose(1,2), lam*original+(1-lam)*original[index])
            expected_var=(lam*sigma+(1-lam)*sigma[index]).square()-1e-5
            for call in restore.call_args_list:
                torch.testing.assert_close(call[0][1], anchor)
                torch.testing.assert_close(call[0][2], lam*mean+(1-lam)*mean[index])
                torch.testing.assert_close(call[0][3], expected_var)
                torch.testing.assert_close(call[0][4], model.time_enc_y(ym))
            self.assertEqual(output.shape, (2,12,7))
        finally:
            hook.remove()

    def test_identity_fact_recovers_same_weight_pulse_in_eval(self):
        torch.manual_seed(2024)
        pulse = make(PULSE).eval()
        torch.manual_seed(2024)
        model = make().eval()
        model.fact_block = nn.Identity()
        inputs = batch(config())
        with torch.no_grad():
            torch.testing.assert_close(model(*inputs), pulse(*inputs), rtol=1e-5, atol=1e-6)

    def test_shared_modules_called_once_or_twice_as_designed(self):
        model=make().train()
        modules={'temporal':model.backbone[0], 'head':model.backbone[2], 'fact':model.fact_block,
                 'sender':model.router_sender,'receiver':model.router_receiver,
                 'timestamp_x':model.time_enc_x,'timestamp_y':model.time_enc_y}
        counts={k:0 for k in modules}
        hooks=[]
        for name,module in modules.items():
            def counted(mod, inputs, output, name=name):
                counts[name]+=1
            hooks.append(module.register_forward_hook(counted))
        try:
            model(*batch(config()))
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(counts,dict(temporal=1,head=2,fact=1,sender=2,receiver=2,timestamp_x=1,timestamp_y=1))
        self.assertTrue(all(int(module.num_batches_tracked)==1 for module in model.modules() if isinstance(module,nn.BatchNorm2d)))

    def strict_step(self, device):
        previous=(torch.are_deterministic_algorithms_enabled(),torch.backends.cudnn.deterministic,
                  torch.backends.cudnn.benchmark)
        try:
            with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.deterministic=True
                torch.backends.cudnn.benchmark=False
                model=make().to(device).train()
                optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
                watched=model.fact_block.block.conv[0][1].conv1.weight
                before=watched.detach().clone()
                for _ in range(2):
                    optimizer.zero_grad()
                    prediction,(lam,index)=model(*batch(config(),device))
                    target=torch.randn_like(prediction)
                    target=lam*target+(1-lam)*target[index]
                    loss=(torch.fft.fft(prediction,dim=1)-torch.fft.fft(target,dim=1)).abs().mean()
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                    for parameter in (model.backbone[0].weight,watched,model.projection_3[0].weight):
                        self.assertGreater(parameter.grad.abs().sum().item(),0)
                    optimizer.step()
                self.assertFalse(torch.equal(watched,before))
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(previous[0])
            torch.backends.cudnn.deterministic=previous[1]
            torch.backends.cudnn.benchmark=previous[2]

    def test_strict_fft_two_optimizer_steps_cpu(self):
        self.strict_step('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable; run this check on the GPU server')
    def test_strict_fft_two_optimizer_steps_cuda(self):
        self.strict_step('cuda')

    def test_save_restore_config_guard_and_dedicated_prefix(self):
        model=make().eval();inputs=batch(config())
        memory=io.BytesIO();torch.save(model.state_dict(),memory);memory.seek(0)
        loaded=torch.load(memory,map_location='cpu',weights_only=False)
        restored=make().eval();restored.load_state_dict(loaded,strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(*inputs),model(*inputs),rtol=0,atol=0)
        with self.assertRaisesRegex(RuntimeError,'configuration differs'):
            make(args=config(pulse_fact_mp_dropout=0.2)).load_state_dict(loaded,strict=True)
        with self.assertRaisesRegex(RuntimeError,'configuration differs'):
            model.load_state_dict(make(PULSE_FACT).state_dict(),strict=True)
        legacy=make(args=config(d_ff=32,dropout=0.9,e_layers=5,pulse_fact_d_ff=16))
        self.assertEqual(legacy.get_extra_state(),make().get_extra_state())
        explicit=make(args=config(pulse_fact_mp_d_ff=32))
        self.assertEqual(explicit.fact_block.block.conv_ffn[1].out_channels,32)
        self.assertEqual(explicit.get_extra_state()['settings']['mean_axis'],1)

    def test_input_guards_small_grids_and_single_step_boundary(self):
        for channels in (1,2,4,7):
            args=config(enc_in=channels)
            model=make(args=args).eval()
            with torch.no_grad():
                self.assertEqual(model(*batch(args)).shape,(2,12,channels))
        args=config(pred_len=1,patch_size=1)
        model=make(args=args).eval()
        with patch.object(model,'preserve_base_prediction_mean',wraps=model.preserve_base_prediction_mean) as projected:
            output=model(*batch(args))
        torch.testing.assert_close(output,projected.call_args[0][0],rtol=0,atol=0)
        model=make()
        x,cycle,xm,ym=batch(config())
        with self.assertRaises(ValueError):
            model(x[:,:-1],cycle,xm,ym)
        for overrides in ({'use_amp':True},{'use_multi_gpu':True},{'label_len':6},{'patch_size':5}):
            with self.assertRaises(ValueError):
                make(args=config(**overrides))

    def test_registry_and_existing_validation_interface(self):
        from exp.exp_main import Exp_Main
        with redirect_stdout(io.StringIO()):
            experiment=Exp_Main(config())
        self.assertIsInstance(experiment.model,PULSE_FACT_MP.Model)
        x,cycle,xm,ym=batch(config())
        value=experiment.vali(object(),[(x,torch.randn(2,12,7),xm,ym,cycle)],nn.MSELoss())
        self.assertTrue(np.isfinite(value))

    def test_source_and_license_manifest(self):
        record=json.loads((ROOT/'notes/data/pulse_fact_mp_sources.json').read_text())
        for item in record['sources']:
            self.assertEqual(hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest(),item['sha256'])
        self.assertEqual(record['copied_methods'],['ReVIN','Model.norm','Model.phase_router','Model.denorm'])


class MPICScriptsTests(unittest.TestCase):
    def test_all_scripts_preserve_selected_source_parameters_paths_and_shapes(self):
        helper=source_tests.PULSEFACTScriptTests();parser=helper.fact_parser()
        for dataset,horizons in SCOPES.items():
            script=ROOT/'scripts/PULSE_FACT_MP'/(dataset+'.sh')
            subprocess.run(['bash','-n',str(script)],check=True)
            content=script.read_text()
            self.assertIn('set -e',content)
            self.assertNotIn('function ',content)
            self.assertNotIn('() {',content)
            self.assertEqual(content.count('python -u run.py'),len(horizons))
            with tempfile.TemporaryDirectory() as before,tempfile.TemporaryDirectory() as after:
                result,original=helper.fact_capture(ROOT/'scripts/PULSE_FACT'/(dataset+'.sh'),before)
                self.assertEqual(result.returncode,0)
                sources={parser.parse_args(c[2:]).pred_len:parser.parse_args(c[2:]) for c in original}
                result,commands=helper.fact_capture(script,after)
                self.assertEqual(result.returncode,0,result.stderr)
                args=[parser.parse_args(c[2:]) for c in commands]
                self.assertEqual(tuple(a.pred_len for a in args),horizons)
                reported=[Path(s[5:]).resolve() for s in result.stdout.splitlines() if s.startswith('Log: ')]
                expected=[]
                for current in args:
                    source=sources[current.pred_len]
                    for key in vars(source).keys()-{'model','model_id','checkpoints'}:
                        self.assertEqual(getattr(current,key),getattr(source,key),(dataset,key))
                    self.assertEqual(current.model,'PULSE_FACT_MP')
                    self.assertEqual(current.checkpoints,'./checkpoints/PULSE_FACT_MP/')
                    log=Path(after)/'logs/PULSE_FACT_MP'/dataset/f'PULSE_FACT_MP_96_{current.pred_len}.log'
                    expected.append(log.resolve())
                    self.assertEqual(log.read_text(),'stdout\nstderr\n')
                    current.use_gpu=False
                    model=make(args=current).eval()
                    with torch.no_grad():
                        output=model(torch.randn(1,96,current.enc_in),torch.zeros(1,dtype=torch.long),
                                     torch.randn(1,96,current.time_dim),torch.randn(1,current.pred_len,current.time_dim))
                    self.assertEqual(output.shape,(1,current.pred_len,current.enc_in))
                self.assertEqual(reported,expected)

    def test_failure_stops_and_repeat_overwrites_logs_with_distinct_checkpoint_ids(self):
        helper=source_tests.PULSEFACTScriptTests();parser=helper.fact_parser()
        for dataset in SCOPES:
            with tempfile.TemporaryDirectory() as directory:
                result,commands=helper.fact_capture(ROOT/'scripts/PULSE_FACT_MP'/(dataset+'.sh'),directory,exit_code=37)
                self.assertEqual(result.returncode,37)
                self.assertEqual(len(commands),1)
        with tempfile.TemporaryDirectory() as directory:
            script=ROOT/'scripts/PULSE_FACT_MP/ETTh1.sh'
            first,one=helper.fact_capture(script,directory)
            second,two=helper.fact_capture(script,directory,message='second')
            self.assertEqual((first.returncode,second.returncode),(0,0))
            self.assertEqual(len(two),4)
            self.assertTrue({parser.parse_args(c[2:]).model_id for c in one}.isdisjoint(
                            {parser.parse_args(c[2:]).model_id for c in two[2:]}))
            logs=list((Path(directory)/'logs').rglob('*.log'))
            self.assertEqual(len(logs),2)
            self.assertTrue(all(log.read_text()=='second\nstderr\n' for log in logs))

    @unittest.skipUnless(torch.cuda.is_available(),
                         'Legacy Exp_Main best-checkpoint restore requires CUDA; no CPU bypass is installed')
    def test_original_run_synthetic_cuda_training_restore_and_automatic_test(self):
        # A temporary project copy prevents run.py from writing any real output.
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            for folder in ('exp','models','layers','utils','data_provider'):
                shutil.copytree(ROOT/folder,work/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            shutil.copy2(ROOT/'run.py',work/'run.py')
            with (work/'smoke.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['date','a','b','c','d','e','f','OT'])
                from datetime import datetime,timedelta
                for index in range(240):
                    writer.writerow([(datetime(2020,1,1)+timedelta(hours=index)).isoformat()]+
                                     [np.sin(index/7+channel)+index*0.001 for channel in range(7)])
            command=[sys.executable,'-B','run.py','--is_training','1','--model','PULSE_FACT_MP',
                '--model_id','mp_smoke','--data','custom','--root_path',str(work),'--data_path','smoke.csv',
                '--seq_len','24','--pred_len','12','--enc_in','7','--time_dim','1','--label_len','0',
                '--patch_size','6','--d_model','8','--dsa','1','--dsb','1','--ksize','3',
                '--batch_size','8','--train_epochs','1','--patience','1','--learning_rate','0.001',
                '--num_workers','0','--checkpoints',str(work/'checkpoints/PULSE_FACT_MP'),
                '--freq','h','--rec_lambda','0','--auxi_lambda','1']
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            result=subprocess.run(command,cwd=work,env=env,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout[-3000:]+'\n'+result.stderr[-3000:])
            self.assertIn('>>>>>>>testing',result.stdout)
            self.assertIn('mse:',result.stdout)
            checkpoints=list((work/'checkpoints/PULSE_FACT_MP').glob('*/checkpoint.pth'))
            self.assertEqual(len(checkpoints),1)
            state=torch.load(checkpoints[0],map_location='cpu',weights_only=False)
            self.assertEqual(state['_extra_state']['model'],'PULSE_FACT_MP')


if __name__=='__main__':
    unittest.main()
