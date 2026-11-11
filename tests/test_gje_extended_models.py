"""Bridge integration checks; all writes/training use temporary directories.

Run: python -B -m unittest discover -s tests -p test_gje_extended_models.py -v
Requires the existing requirements-gje.txt. Synthetic H5 scores are not results.
"""

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
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

import numpy as np
import torch

from models import PULSE_SCF, PULSE_FACT
from test_gjemnessund import make_h5
from utils.gje_model_support import GJE_SUPPORTED_MODELS, gje_extended_model_provenance


ROOT = Path(__file__).resolve().parents[1]
MODELS = {'PULSE_SCF': PULSE_SCF.Model, 'PULSE_FACT': PULSE_FACT.Model}


def bridge_config(pred_len=96):
    return SimpleNamespace(seq_len=720, pred_len=pred_len, enc_in=10,
                           time_dim=6, d_model=16, dsa=1, dsb=1,
                           inv_len=96, patch_size=24, ksize=5, label_len=0,
                           use_amp=False, use_multi_gpu=False)


class GjeExtendedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_entry_only_allows_verified_gje_models_and_preserves_guards(self):
        tree = ast.parse((ROOT / 'run.py').read_text())
        expected = ast.dump(ast.parse("args.data == 'Gjemnessund'", mode='eval').body)
        guard = next(node for node in tree.body if isinstance(node, ast.If)
                     and ast.dump(node.test) == expected)
        code = compile(ast.Module(body=[guard], type_ignores=[]), 'Gje guard', 'exec')
        valid = dict(data='Gjemnessund', features='M', enc_in=10, time_dim=6,
                     label_len=0, use_amp=False, do_predict=False, use_multi_gpu=False,
                     seq_len=720, pred_len=96, patch_size=24, inv_len=96,
                     dsa=1, dsb=1, gje_record_limit=0, gje_max_batches=0,
                     gje_window_stride=16, gje_cycle_len=96, gje_log_interval=100)

        def check(**overrides):
            args = SimpleNamespace(**dict(valid, **overrides))
            exec(code, {'args': args, 'parser': argparse.ArgumentParser()})

        self.assertEqual(GJE_SUPPORTED_MODELS[:3], ('PULSE', 'PULSE_SCF', 'PULSE_FACT'))
        self.assertEqual(GJE_SUPPORTED_MODELS[3:], ('PULSE_FACT_MODAL', 'PULSE_FACT_OSC'))
        for name in GJE_SUPPORTED_MODELS:
            check(model=name)
            for bad in ({'time_dim': 1}, {'features': 'MS'}, {'enc_in': 6},
                        {'use_amp': True}, {'use_multi_gpu': True},
                        {'label_len': 12}, {'patch_size': 25}, {'dsa': 11},
                        {'do_predict': True}, {'gje_window_stride': 0}):
                with self.subTest(name=name, bad=bad), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        check(model=name, **bad)
        for name in ('PULSE_SOFTS', 'PULSE_SCF_SIMPLE', 'PULSE_FACT_PC', 'iTransformer'):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                check(model=name)
        check(data='ETTh1', model='iTransformer', time_dim=1, enc_in=7)

    def test_four_bridge_horizons_fft_gradient_and_checkpoint_roundtrip(self):
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for name, constructor in MODELS.items():
                for horizon in (96, 192, 336, 720):
                    with self.subTest(model=name, horizon=horizon), redirect_stdout(io.StringIO()):
                        torch.manual_seed(2024)
                        np.random.seed(2024)
                        config = bridge_config(horizon)
                        model = constructor(config).train()
                        x = torch.randn(2, 720, 10)
                        cycle = torch.tensor([0, 95])
                        xm, ym = torch.randn(2, 720, 6), torch.randn(2, horizon, 6)
                        prediction, (lam, index) = model(x, cycle, xm, ym)
                        self.assertEqual(tuple(prediction.shape), (2, horizon, 10))
                        target = torch.randn_like(prediction)
                        target = lam * target + (1 - lam) * target[index]
                        loss = (torch.fft.fft(prediction, dim=1)
                                - torch.fft.fft(target, dim=1)).abs().mean()
                        loss.backward()
                        self.assertTrue(torch.isfinite(loss))
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(float(model.codebook.grad.abs().sum()), 0)
                        torch.optim.Adam(model.parameters(), lr=0.005).step()
                        model.eval()
                        with torch.no_grad():
                            expected = model(x, cycle, xm, ym)
                        with tempfile.TemporaryDirectory() as tmp:
                            path = Path(tmp) / 'checkpoint.pth'
                            torch.save(model.state_dict(), path)
                            restored = constructor(config).eval()
                            restored.load_state_dict(torch.load(str(path), map_location='cpu'))
                        with torch.no_grad():
                            torch.testing.assert_close(restored(x, cycle, xm, ym), expected,
                                                       rtol=0, atol=0)
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_metadata_records_actual_model_and_preserves_legacy_record(self):
        from exp.exp_main import Exp_Main
        for name, constructor in MODELS.items():
            with redirect_stdout(io.StringIO()):
                model = constructor(bridge_config())
            info = gje_extended_model_provenance(name, model)
            main = 'models/{}.py'.format(name)
            self.assertEqual(info['model_file'], main)
            self.assertEqual(info['model_sha256'], hashlib.sha256((ROOT / main).read_bytes()).hexdigest())
            self.assertEqual(info['effective_model_settings'], model.get_extra_state())
            namespace = 'pulse_scf' if name == 'PULSE_SCF' else 'pulse_fact'
            self.assertIn('layers/{}/vendor/fact_conv.py'.format(namespace), info['source_sha256'])
            json.dumps(info)

        # Exercise the exact experiment hook: old PULSE records keep their keys.
        with tempfile.TemporaryDirectory() as tmp:
            exp = Exp_Main.__new__(Exp_Main)
            exp.args = SimpleNamespace(model='PULSE', gje_record_limit=0, gje_max_batches=0)
            exp.device = torch.device('cpu')
            exp._save_gje_protocol(tmp, ())
            legacy = json.loads((Path(tmp) / 'gje_run.json').read_text())
            self.assertEqual(set(legacy), {'args', 'python', 'torch', 'numpy', 'device',
                                          'model_sha256', 'debug', 'loss', 'scheduler', 'test_policy'})
            self.assertEqual(legacy['model_sha256'], hashlib.sha256((ROOT / 'models/PULSE.py').read_bytes()).hexdigest())
            for name, constructor in MODELS.items():
                exp.args.model = name
                with redirect_stdout(io.StringIO()):
                    exp.model = constructor(bridge_config())
                exp._save_gje_protocol(tmp, ())
                extended = json.loads((Path(tmp) / 'gje_run.json').read_text())
                self.assertEqual(extended['model_name'], name)
                self.assertEqual(extended['effective_model_settings'], exp.model.get_extra_state())

    def test_scripts_match_baseline_configuration_paths_and_stop_on_failure(self):
        options = {node.args[0].value for node in ast.walk(ast.parse((ROOT / 'run.py').read_text()))
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == 'add_argument' and node.args}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            capture = root / 'capture.py'
            capture.write_text('import json, os, sys\n'
                               'with open(os.environ["CAPTURE"], "a") as f:\n'
                               '    f.write(json.dumps(sys.argv[1:]) + "\\n")\n'
                               'print("captured stdout")\n'
                               'print("captured stderr", file=sys.stderr)\n'
                               'sys.exit(int(os.environ.get("CAPTURE_EXIT", "0")))\n')
            stub = bin_dir / 'python'
            stub.write_text('#!/bin/bash\nexec "{}" "{}" "$@"\n'.format(sys.executable, capture))
            stub.chmod(0o755)
            env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
                       CAPTURE=str(root / 'calls.jsonl'), GJE_ROOT_PATH=str(root / 'data with spaces'))

            def run_script(script):
                Path(env['CAPTURE']).unlink(missing_ok=True)
                result = subprocess.run(['bash', str(script)], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=30)
                calls = [json.loads(line) for line in Path(env['CAPTURE']).read_text().splitlines()]
                return result, [dict(zip(call[2::2], call[3::2])) for call in calls]

            baseline_result, baseline_calls = run_script(ROOT / 'scripts/Gjemnessund.sh')
            self.assertEqual(baseline_result.returncode, 0, baseline_result.stderr)
            for name in MODELS:
                script = ROOT / 'scripts' / name / 'Gjemnessund.sh'
                subprocess.run(['bash', '-n', str(script)], check=True)
                result, calls = run_script(script)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(calls), 4)
                for call, base, horizon in zip(calls, baseline_calls, (96, 192, 336, 720)):
                    self.assertTrue(set(call) <= options)
                    self.assertEqual(call['--model'], name)
                    self.assertEqual(call['--root_path'], env['GJE_ROOT_PATH'])
                    self.assertEqual(call['--checkpoints'], './checkpoints/{}/'.format(name))
                    self.assertTrue(call['--model_id'].startswith(name + '_Gjemnessund_720_'))
                    self.assertEqual(call['--auxi_type'], 'complex')
                    for key, value in base.items():
                        if key not in ('--model_id', '--model', '--root_path'):
                            self.assertEqual(call[key], value, key)
                    log = root / 'logs' / name / 'Gjemnessund' / '{}_720_{}.log'.format(name, horizon)
                    self.assertIn('captured stdout', log.read_text())
                    self.assertIn('captured stderr', log.read_text())
                    self.assertIn(str(log), result.stdout)
                env['CAPTURE_EXIT'] = '17'
                result, calls = run_script(script)
                self.assertEqual(result.returncode, 17)
                self.assertEqual(len(calls), 1)
                del env['CAPTURE_EXIT']

    def test_original_entry_training_restore_and_independent_test_on_synthetic_h5(self):
        # Use an isolated source copy so even result.txt and caches never touch
        # real experiment artifacts. Include original PULSE as a regression run.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy2(str(ROOT / 'run.py'), str(root / 'run.py'))
            for directory in ('models', 'layers', 'exp', 'data_provider', 'utils'):
                shutil.copytree(str(ROOT / directory), str(root / directory),
                                ignore=shutil.ignore_patterns('__pycache__'))
            data = root / 'data with spaces'
            data.mkdir()
            make_h5(data / 'gjemnessund.h5')
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
            for name in ('PULSE', 'PULSE_SCF', 'PULSE_FACT'):
                with self.subTest(model=name):
                    checkpoint_root = root / 'checkpoints' / name
                    common = [sys.executable, '-B', '-u', 'run.py', '--model', name,
                              '--model_id', name + '_synthetic_only', '--data', 'Gjemnessund',
                              '--root_path', str(data), '--data_path', 'gjemnessund.h5',
                              '--checkpoints', str(checkpoint_root), '--features', 'M',
                              '--seq_len', '24', '--pred_len', '12', '--label_len', '0',
                              '--patch_size', '6', '--enc_in', '10', '--time_dim', '6',
                              '--dsa', '1', '--dsb', '1', '--ksize', '5', '--d_model', '16',
                              '--inv_len', '96', '--gje_cycle_len', '96',
                              '--gje_window_stride', '6', '--gje_max_batches', '1',
                              '--batch_size', '2', '--train_epochs', '1', '--patience', '1',
                              '--num_workers', '0', '--learning_rate', '0.005', '--itr', '1']
                    for training in (1, 0):
                        result = subprocess.run(common + ['--is_training', str(training)],
                                                cwd=root, env=env, capture_output=True,
                                                text=True, timeout=120)
                        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-4000:])
                        checkpoints = list(checkpoint_root.glob('*/checkpoint.pth'))
                        self.assertEqual(len(checkpoints), 1)
                        setting = checkpoints[0].parent.name
                        metrics = json.loads((root / 'results' / setting / 'gje_metrics.json').read_text())
                        self.assertTrue(metrics['debug'])
                        self.assertEqual(len(metrics['per_channel']), 10)
                        for key in ('mse', 'mae', 'acceleration_mse', 'acceleration_mae'):
                            self.assertTrue(np.isfinite(metrics[key]))
                        if training:
                            initial_metrics = metrics
                        else:
                            for key in ('mse', 'mae', 'acceleration_mse', 'acceleration_mae'):
                                self.assertEqual(metrics[key], initial_metrics[key])
                    metadata = json.loads((checkpoints[0].parent / 'gje_run.json').read_text())
                    main = 'models/{}.py'.format(name)
                    self.assertEqual(metadata['model_sha256'], hashlib.sha256((root / main).read_bytes()).hexdigest())
                    if name != 'PULSE':
                        self.assertEqual(metadata['model_file'], main)
                        self.assertIn('effective_model_settings', metadata)


if __name__ == '__main__':
    unittest.main()
