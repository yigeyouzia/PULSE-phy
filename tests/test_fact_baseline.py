"""FACT integration checks. Training/data/output writes stay in temporary dirs.

Run: python -B -m unittest discover -s tests -p test_fact_baseline.py -v
"""

import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from models.FACT import Model
from utils.fact_original.support import (add_fact_arguments, fact_model_config, fact_setting,
                                         sha256_file, validate_fact_arguments)


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / 'utils/fact_original'


def fact_parser():
    tree = ast.parse((ROOT / 'run.py').read_text())
    nodes = [n for n in tree.body if (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parser' for t in n.targets)
    ) or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
          and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name)
          and n.value.func.value.id == 'parser' and n.value.func.attr == 'add_argument')]
    namespace = {'argparse': argparse}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'run.py parser', 'exec'), namespace)
    add_fact_arguments(namespace['parser'])
    return namespace['parser']


def fact_args(**overrides):
    args = fact_parser().parse_args([
        '--is_training', '1', '--model', 'FACT', '--model_id', 'unit', '--data', 'ETTh1',
        '--seq_len', '24', '--pred_len', '12', '--enc_in', '7', '--time_dim', '1',
        '--rec_lambda', '1', '--auxi_lambda', '0', '--fact_d_model', '16',
        '--fact_d_ff', '32', '--fact_dilation', '1', '2', '--fact_dropout', '0',
        '--batch_size', '2', '--num_workers', '0'])
    vars(args).update(overrides)
    return args


class FACTBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_source_license_predictions_gradients_and_state_match(self):
        manifest = json.loads((SUPPORT / 'sources.json').read_text())
        for record in manifest['files']:
            self.assertEqual(sha256_file(ROOT / record['copied_file']), record['source_sha256'])
        source = ROOT.parent / 'FACT-main/models/FACT.py'
        if source.is_file():
            self.assertEqual(source.read_bytes(), (ROOT / 'models/FACT.py').read_bytes())
            spec = importlib.util.spec_from_file_location('fact_source_reference', str(source))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            config = fact_model_config(fact_args())
            original, copy = module.Model(config).train(), Model(config).train()
            copy.load_state_dict(original.state_dict())
            # Raw dataset inputs are not trainable. The source's in-place
            # normalization does not support differentiating the raw input.
            x = torch.randn(2, 24, 7)
            other = x.clone()
            expected, actual = original(x, None, None, None), copy(other, None, None, None)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected.square().mean().backward()
            actual.square().mean().backward()
            for (_, old), (_, new) in zip(original.named_parameters(), copy.named_parameters()):
                torch.testing.assert_close(new.grad, old.grad, rtol=0, atol=0)

    def test_legacy_defaults_hash_function_and_fact_config_identity(self):
        snapshot = json.loads((SUPPORT / 'legacy_entry_snapshot.json').read_text())
        args = fact_parser().parse_args(['--is_training', '1', '--model', 'PULSE',
                                         '--model_id', 'legacy_regression', '--data', 'ETTh1'])
        self.assertEqual(vars(args), snapshot['args'])
        tree = ast.parse((ROOT / 'run.py').read_text())
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gje_setting')
        self.assertEqual(hashlib.sha256(ast.dump(func).encode()).hexdigest(), snapshot['gje_setting_ast_sha256'])
        first = fact_args()
        second = fact_args(fact_core=0.3)
        self.assertNotEqual(fact_setting('same', first), fact_setting('same', second))
        self.assertEqual(fact_setting('legacy', args), 'legacy')
        validate_fact_arguments(first)
        for override in ({'fact_core': 0}, {'fact_d_model': 15}, {'fact_dilation': [0]},
                         {'enc_in': 4}, {'auxi_lambda': 1}, {'use_amp': True}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_fact_arguments(fact_args(**override))
        # FACT has no PULSE patch-divisibility constraint.
        validate_fact_arguments(fact_args(data='Gjemnessund', enc_in=10, time_dim=6,
                                         seq_len=25, pred_len=13, patch_size=24))

    def test_all_52_script_shapes_have_valid_native_model_outputs(self):
        sources = json.loads((SUPPORT / 'script_sources.json').read_text())
        for record in sources.values():
            for profile in record['calls']:
                args = fact_args(**profile)
                with self.subTest(data=args.data_path, horizon=args.pred_len):
                    validate_fact_arguments(args)
                    model = Model(fact_model_config(args)).eval()
                    with torch.no_grad():
                        prediction = model(torch.randn(1, args.seq_len, args.enc_in), None, None, None)
                    self.assertEqual(tuple(prediction.shape), (1, args.pred_len, args.enc_in))
                    self.assertTrue(torch.isfinite(prediction).all())

    def test_actual_default_width_bridge_gradient_and_restore(self):
        args = fact_args(data='Gjemnessund', enc_in=10, time_dim=6, seq_len=720, pred_len=96,
                         fact_d_model=512, fact_d_ff=2048, fact_dilation=[1, 2, 3, 2, 1])
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            config = fact_model_config(args)
            model = Model(config).train()
            x, target = torch.randn(2, 720, 10), torch.randn(2, 96, 10)
            loss = (model(x, None, None, None) - target).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            torch.optim.Adam(model.parameters(), lr=0.005).step()
            model.eval()
            with torch.no_grad():
                expected = model(x, None, None, None)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'checkpoint.pth'
                torch.save(model.state_dict(), path)
                restored = Model(config).eval()
                restored.load_state_dict(torch.load(str(path), map_location='cpu'))
            with torch.no_grad():
                torch.testing.assert_close(restored(x, None, None, None), expected, rtol=0, atol=0)
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_13_scripts_arguments_budgets_logs_and_failure_stop(self):
        sources = json.loads((SUPPORT / 'script_sources.json').read_text())
        parser = fact_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stub = root / 'python'
            stub.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                            'with open(os.environ["FACT_CAPTURE"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                            'print("stdout",flush=True)\nprint("stderr",file=sys.stderr,flush=True)\n'
                            'sys.exit(int(os.environ.get("FACT_EXIT","0")))\n')
            stub.chmod(0o755)
            capture = root / 'calls.jsonl'
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                       FACT_CAPTURE=str(capture), GJE_ROOT_PATH=str(root / 'H5 path with spaces'))
            for filename, source in sources.items():
                script = ROOT / 'scripts/FACT' / filename
                subprocess.run(['bash', '-n', str(script)], check=True)
                capture.unlink(missing_ok=True)
                result = subprocess.run(['bash', str(script)], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = [parser.parse_args(json.loads(line)[2:]) for line in capture.read_text().splitlines()]
                self.assertEqual(len(calls), 4)
                for new, old in zip(calls, source['calls']):
                    validate_fact_arguments(new)
                    for key, value in old.items():
                        if key != 'root_path':
                            self.assertEqual(getattr(new, key), value, (filename, key))
                    self.assertEqual(new.model, 'FACT')
                    self.assertEqual(new.fact_d_model, 512)
                    self.assertEqual(new.rec_lambda, 1)
                    self.assertEqual(new.auxi_lambda, 0)
                    self.assertEqual(new.checkpoints, './checkpoints/FACT/')
                logs = [Path(line[5:]) for line in result.stdout.splitlines() if line.startswith('Log: ')]
                self.assertEqual(len(logs), 4)
                self.assertTrue(all(p.read_text() == 'stdout\nstderr\n' for p in logs))
                for log, call in zip(logs, calls):
                    expected = root / 'logs/FACT' / source['dataset'] / 'FACT_{}_{}.log'.format(
                        call.seq_len, call.pred_len)
                    self.assertEqual(log.resolve(), expected.resolve())
                    self.assertTrue(call.model_id.startswith('FACT_' + source['dataset'] + '_'))
                if filename == 'Gjemnessund.sh':
                    self.assertEqual(calls[0].root_path, env['GJE_ROOT_PATH'])
                capture.unlink()
                failure_env = dict(env, FACT_EXIT='17')
                result = subprocess.run(['bash', str(script)], cwd=root, env=failure_env, capture_output=True)
                self.assertEqual(result.returncode, 17)
                self.assertEqual(len(capture.read_text().splitlines()), 1)

    def test_direct_experiment_native_mse_training_and_saved_data_guard(self):
        from exp.exp_fact import Exp_FACT
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Use Gje so its existing weighted scaler/window protocol is also checked.
            from test_gjemnessund import make_h5
            make_h5(root / 'gjemnessund.h5')
            args = fact_args(data='Gjemnessund', enc_in=10, time_dim=6, use_gpu=False,
                             root_path=str(root), data_path='gjemnessund.h5',
                             checkpoints=str(root / 'checkpoints'), gje_window_stride=6,
                             gje_max_batches=1, train_epochs=1, patience=1, learning_rate=0.005)
            previous = Path.cwd()
            try:
                os.chdir(root)
                with redirect_stdout(io.StringIO()):
                    experiment = Exp_FACT(args)
                    experiment.train('isolated')
                    automatic = experiment.test('isolated')
                    restored = Exp_FACT(args)
                    independent = restored.test('isolated', test=1)
                for key in ('mse', 'mae', 'acceleration_mse', 'macro_record_mse'):
                    self.assertEqual(automatic[key], independent[key])
                metadata = json.loads((root / 'checkpoints/isolated/fact_run.json').read_text())
                self.assertEqual(metadata['effective_model_config']['d_model'], 16)
                self.assertEqual(len(independent['per_channel']), 10)
                self.assertTrue(independent['debug'])
                # Same shape, different data must not silently load an old checkpoint.
                args.data_path = 'changed.h5'
                make_h5(root / 'changed.h5', future_shift=10)
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'data/scaler'):
                    restored.test('isolated', test=1)
            finally:
                os.chdir(previous)

    def test_original_entry_standard_and_bridge_train_and_independent_test(self):
        from test_gjemnessund import make_h5
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy2(str(ROOT / 'run.py'), str(root / 'run.py'))
            for folder in ('models', 'layers', 'data_provider', 'exp', 'utils'):
                shutil.copytree(str(ROOT / folder), str(root / folder), ignore=shutil.ignore_patterns('__pycache__'))
            data = root / 'data with spaces'
            data.mkdir()
            make_h5(data / 'gjemnessund.h5')
            frame = pd.DataFrame(np.random.default_rng(2024).normal(size=(220, 7)),
                                 columns=['a', 'b', 'c', 'd', 'e', 'f', 'OT'])
            frame.insert(0, 'date', pd.date_range('2020-01-01', periods=220, freq='h'))
            frame.to_csv(data / 'synthetic.csv', index=False)
            env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
            for dataset, file, channels, time_dim in [('custom', 'synthetic.csv', 7, 1),
                                                       ('Gjemnessund', 'gjemnessund.h5', 10, 6)]:
                checkpoint = root / 'checkpoints' / dataset
                common = [sys.executable, '-B', '-u', 'run.py', '--model', 'FACT',
                          '--model_id', dataset + '_synthetic_only', '--data', dataset,
                          '--root_path', str(data), '--data_path', file,
                          '--checkpoints', str(checkpoint), '--seq_len', '24', '--pred_len', '12',
                          '--enc_in', str(channels), '--time_dim', str(time_dim),
                          '--rec_lambda', '1', '--auxi_lambda', '0', '--fact_d_model', '16',
                          '--fact_d_ff', '32', '--fact_dilation', '1', '2', '--batch_size', '2',
                          '--num_workers', '0', '--train_epochs', '1', '--patience', '1',
                          '--gje_window_stride', '6', '--gje_max_batches', '1']
                for training in (1, 0):
                    result = subprocess.run(common + ['--is_training', str(training)], cwd=root,
                                            env=env, capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-3000:])
                    weights = list(checkpoint.glob('*/checkpoint.pth'))
                    self.assertEqual(len(weights), 1)
                    setting = weights[0].parent.name
                    report = json.loads((root / 'results' / setting / 'fact_metrics.json').read_text())
                    self.assertTrue(np.isfinite(report['mse']))
                    if training:
                        expected = report['mse']
                    else:
                        self.assertEqual(report['mse'], expected)


if __name__ == '__main__':
    unittest.main()
