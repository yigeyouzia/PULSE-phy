"""Physics checks and isolated original-entry smoke runs, not benchmark results.

Run: python -B -m unittest discover -s tests -p test_gje_physics.py -v
Dependencies: requirements-gje.txt. All generated training outputs are temporary.
"""

from contextlib import redirect_stdout
import ast
import copy
import hashlib
import io
import json
import math
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

from models import PULSE_FACT, PULSE_FACT_MODAL, PULSE_FACT_OSC
from layers.gje_physics.modal_prior import GjeModalFrequencyPrior
from layers.gje_physics.stable_oscillator import GjeStableForcedOscillator
from layers.gje_physics.units import GjeAccelerationUnits, GJE_COLUMNS, STANDARD_GRAVITY
from utils.gje_model_support import gje_extended_model_provenance
from test_gjemnessund import make_h5


ROOT = Path(__file__).resolve().parents[1]
MODELS = {'PULSE_FACT_MODAL': PULSE_FACT_MODAL.Model, 'PULSE_FACT_OSC': PULSE_FACT_OSC.Model}


def calibration():
    return dict(protocol='per_record_50_25_25_train_scaler_trim_overlap_v1',
                columns=GJE_COLUMNS, sample_interval_seconds=0.5,
                scaler=dict(mean=[0.01 * (i + 1) for i in range(10)],
                            scale=[0.001 * (i + 1) for i in range(10)],
                            fit_scope='each record rows [0, train_end), once per raw row'))


def config(seq_len=720, pred_len=96):
    return SimpleNamespace(seq_len=seq_len, pred_len=pred_len, enc_in=10,
                           time_dim=6, d_model=16, dsa=1, dsb=1, inv_len=96,
                           patch_size=24 if seq_len == 720 else 6, ksize=5,
                           label_len=0, use_amp=False, use_multi_gpu=False,
                           data='Gjemnessund')


class GjePhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_standard_scaler_g_si_roundtrip_and_protocol_rejection(self):
        units = GjeAccelerationUnits()
        with self.assertRaises(RuntimeError):
            units.to_si(torch.zeros(2, 12, 6))
        info = calibration()
        units.bind_protocol(info)
        x = torch.randn(2, 24, 6)
        si = units.to_si(x)
        expected = (x * torch.tensor(info['scaler']['scale'][:6])
                    + torch.tensor(info['scaler']['mean'][:6])) * STANDARD_GRAVITY
        torch.testing.assert_close(si, expected)
        torch.testing.assert_close(units.from_si(si), x, atol=5e-6, rtol=1e-5)
        torch.testing.assert_close(units.si_residual_to_standardized(
            units.scale_g * STANDARD_GRAVITY), torch.ones(1, 1, 6))
        for mutation in ('sample_interval_seconds', 'columns', 'scale', 'fit_scope'):
            invalid = copy.deepcopy(info)
            if mutation == 'sample_interval_seconds':
                invalid[mutation] = 0.1
            elif mutation == 'columns':
                invalid[mutation] = list(reversed(invalid[mutation]))
            elif mutation == 'scale':
                invalid['scaler']['scale'][0] *= 2
            else:
                invalid['scaler']['fit_scope'] = 'train plus future rows'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                units.bind_protocol(invalid)
        other = GjeAccelerationUnits()
        other.load_state_dict(units.state_dict())
        torch.testing.assert_close(other.to_si(x), si)
        other.bind_protocol(info)

    def test_harmonic_extrapolation_uses_seconds_and_correct_phase(self):
        layer = GjeModalFrequencyPrior(720, 96, frequencies_hz=(0.176,),
                                       ridge=1e-8, initial_gain=0.1)
        past_time = torch.arange(-719, 1, dtype=torch.float64) * 0.5
        future_time = torch.arange(1, 97, dtype=torch.float64) * 0.5
        past = 0.04 * torch.cos(2 * math.pi * 0.176 * past_time + 0.7)
        future = 0.04 * torch.cos(2 * math.pi * 0.176 * future_time + 0.7)
        history = (past + 0.2)[None, :, None].repeat(2, 1, 6).float()
        result = layer(history)
        expected = (0.1 * (future - past.mean())).float()[None, :, None].expand_as(result)
        torch.testing.assert_close(result, expected, atol=1e-7, rtol=1e-4)
        result.square().mean().backward()
        self.assertGreater(float(layer.raw_gain.grad.abs().sum()), 0)
        for frequency in (0.0, 1.0, 1.4, float('nan')):
            with self.assertRaises(ValueError):
                GjeModalFrequencyPrior(720, 96, frequencies_hz=(frequency,))

    def test_exact_zoh_recurrence_stable_poles_and_causal_convolution(self):
        oscillator = GjeStableForcedOscillator(24, 96).double()
        z = torch.randn(2, 5, 2, dtype=torch.float64)
        forcing = torch.randn(2, 96, 5, dtype=torch.float64)
        acceleration, states = oscillator.simulate(z, forcing)
        A = oscillator.transition_powers(1)[:, 1]
        B = (torch.eye(2, dtype=A.dtype) - A)[:, :, 0]
        iterative = []
        for index in range(96):
            z = torch.einsum('kij,bkj->bki', A, z) + B[None] * forcing[:, index, :, None]
            iterative.append(z)
        torch.testing.assert_close(states, torch.stack(iterative, dim=1), atol=2e-13, rtol=2e-12)
        torch.testing.assert_close(acceleration, -states[..., 0]
                                   - 2 * oscillator.damping_ratios() * states[..., 1] + forcing)
        for offset in (-100.0, 0.0, 100.0):
            with torch.no_grad():
                oscillator.raw_damping.fill_(offset)
                oscillator.raw_frequency_offset.fill_(offset)
            eigenvalues = torch.linalg.eigvals(oscillator.transition_powers(1)[:, 1])
            self.assertTrue((eigenvalues.abs() < 1).all())
            self.assertTrue((oscillator.frequencies_hz() < 1).all())
            self.assertTrue((oscillator.damping_ratios() > 0).all())
        changed = forcing.clone()
        changed[:, 40:] += 3
        initial = torch.randn(2, 5, 2, dtype=torch.float64)
        before, _ = oscillator.simulate(initial, forcing)
        after, _ = oscillator.simulate(initial, changed)
        torch.testing.assert_close(before[:, :40], after[:, :40], atol=1e-12, rtol=1e-12)
        # Exact homogeneous dynamics decay over long horizons; forcing is NOT
        # claimed to monotonically decrease energy.
        final_power = oscillator.transition_powers(10000)[:, -1]
        self.assertLess(float(final_power.abs().max().detach()), 1e-10)

    def test_zero_gain_is_original_model_and_oscillator_has_no_acceleration_bypass(self):
        cfg = config(24, 12)
        with redirect_stdout(io.StringIO()):
            base = PULSE_FACT.Model(cfg).eval()
            modal = PULSE_FACT_MODAL.Model(cfg).eval()
            osc = PULSE_FACT_OSC.Model(cfg).eval()
        base_state = base.state_dict()
        for model in (modal, osc):
            state = model.state_dict()
            state.update({key: value for key, value in base_state.items() if key != '_extra_state'})
            # Calibration is required even before checkpoint serialization.
            model.bind_gje_protocol(calibration())
            state['acceleration_units._extra_state'] = model.acceleration_units.get_extra_state()
            model.load_state_dict(state)
        with torch.no_grad():
            modal.modal_prior.raw_gain.zero_()
        x, cycle = torch.randn(2, 24, 10), torch.tensor([0, 95])
        xm, ym = torch.randn(2, 24, 6), torch.randn(2, 12, 6)
        with torch.no_grad():
            reference, actual = base(x, cycle, xm, ym), modal(x, cycle, xm, ym)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        # Replacing the physical decoder with a known SI signal must replace
        # ALL six final acceleration outputs, regardless of backbone values.
        known_si = torch.full((2, 12, 6), 0.0123)
        with patch.object(osc.oscillator, 'forward', return_value=known_si) as decoder:
            with torch.no_grad():
                result = osc(x, cycle, xm, ym)
        decoder.assert_called_once()
        torch.testing.assert_close(result[:, :, :6], osc.acceleration_units.from_si(known_si), atol=0, rtol=0)
        torch.testing.assert_close(result[:, :, 6:], reference[:, :, 6:], atol=0, rtol=0)

    def test_sam_history_and_supervision_share_the_same_mixing(self):
        for constructor in MODELS.values():
            with redirect_stdout(io.StringIO()):
                model = constructor(config(24, 12)).train()
            model.bind_gje_protocol(calibration())
            x = torch.randn(2, 24, 10)
            cycle = torch.tensor([0, 95])
            xm, ym = torch.randn(2, 24, 6), torch.randn(2, 12, 6)
            original = model.acceleration_units.to_si
            with patch.object(model.acceleration_units, 'to_si', wraps=original) as conversion:
                _, (lam, index) = model(x, cycle, xm, ym)
            expected = lam * x + (1 - lam) * x[index]
            torch.testing.assert_close(conversion.call_args[0][0], expected[:, :, :6], atol=0, rtol=0)

    def _gradient_roundtrip(self, device):
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for name, constructor in MODELS.items():
                for horizon in (96, 192, 336, 720):
                    with self.subTest(model=name, horizon=horizon, device=device), redirect_stdout(io.StringIO()):
                        torch.manual_seed(2024)
                        np.random.seed(2024)
                        cfg = config(pred_len=horizon)
                        model = constructor(cfg).to(device).train()
                        model.bind_gje_protocol(calibration())
                        x = torch.randn(2, 720, 10, device=device)
                        cycle = torch.tensor([0, 95], device=device)
                        xm = torch.randn(2, 720, 6, device=device)
                        ym = torch.randn(2, horizon, 6, device=device)
                        prediction, (lam, index) = model(x, cycle, xm, ym)
                        self.assertEqual(tuple(prediction.shape), (2, horizon, 10))
                        target = torch.randn_like(prediction)
                        target = lam * target + (1 - lam) * target[index]
                        loss = (torch.fft.fft(prediction, dim=1) - torch.fft.fft(target, dim=1)).abs().mean()
                        loss.backward()
                        self.assertTrue(torch.isfinite(loss))
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(float(model.codebook.grad.abs().sum()), 0)
                        branch = model.modal_prior if name.endswith('MODAL') else model.oscillator
                        self.assertGreater(sum(float(p.grad.abs().sum()) for p in branch.parameters() if p.grad is not None), 0)
                        torch.optim.Adam(model.parameters(), lr=0.005).step()
                        model.eval()
                        with torch.no_grad():
                            expected = model(x, cycle, xm, ym)
                        with tempfile.TemporaryDirectory() as tmp:
                            file = Path(tmp) / 'checkpoint.pth'
                            torch.save(model.state_dict(), file)
                            restored = constructor(cfg).to(device).eval()
                            restored.load_state_dict(torch.load(file, map_location=device, weights_only=False))
                            with torch.no_grad():
                                actual = restored(x, cycle, xm, ym)
                            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                            restored.bind_gje_protocol(calibration())
                        provenance = gje_extended_model_provenance(name, model)
                        self.assertEqual(provenance['gje_physics_calibration']['acceleration_unit'], 'g')
                        main_file = 'models/' + name + '.py'
                        self.assertEqual(provenance['model_sha256'], hashlib.sha256((ROOT / main_file).read_bytes()).hexdigest())
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_four_horizons_fft_gradient_and_checkpoint_cpu(self):
        self._gradient_roundtrip('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable on this host')
    def test_four_horizons_fft_gradient_and_checkpoint_cuda(self):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        self._gradient_roundtrip('cuda')

    def test_scripts_are_direct_commands_valid_flags_fixed_paths_and_fail_fast(self):
        options = {node.args[0].value for node in ast.walk(ast.parse((ROOT / 'run.py').read_text()))
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == 'add_argument' and node.args}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / 'bin'
            binary.mkdir()
            capture = root / 'capture.py'
            capture.write_text('import json,os,sys\n'
                               'with open(os.environ["CAPTURE"],"a") as f:\n'
                               ' f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                               'print("stdout")\nprint("stderr",file=sys.stderr)\n'
                               'sys.exit(int(os.environ.get("CAPTURE_EXIT","0")))\n')
            stub = binary / 'python'
            stub.write_text('#!/bin/bash\nexec "{}" "{}" "$@"\n'.format(sys.executable, capture))
            stub.chmod(0o755)
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
                       CAPTURE=str(root / 'calls.jsonl'), GJE_ROOT_PATH=str(root / 'data with spaces'))
            for name in MODELS:
                script = ROOT / 'scripts' / name / 'Gjemnessund.sh'
                text = script.read_text()
                self.assertNotIn('function ', text)
                self.assertNotIn('for ', text)
                subprocess.run(['bash', '-n', str(script)], check=True)
                result = subprocess.run(['bash', str(script)], cwd=root, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = [json.loads(line)[2:] for line in Path(env['CAPTURE']).read_text().splitlines()]
                self.assertEqual(len(calls), 4)
                for call, horizon in zip(calls, (96, 192, 336, 720)):
                    args = dict(zip(call[::2], call[1::2]))
                    self.assertFalse(set(args) - options)
                    self.assertEqual(args['--model'], name)
                    self.assertEqual(args['--seq_len'], '720')
                    self.assertEqual(args['--pred_len'], str(horizon))
                    self.assertEqual(args['--root_path'], env['GJE_ROOT_PATH'])
                    self.assertEqual(args['--checkpoints'], './checkpoints/' + name + '/')
                    self.assertEqual(args['--auxi_lambda'], '1')
                    self.assertEqual(args['--rec_lambda'], '0')
                    log = root / 'logs' / name / 'Gjemnessund' / (name + '_720_{}.log'.format(horizon))
                    self.assertEqual(sorted(log.read_text().splitlines()), ['stderr', 'stdout'])
                Path(env['CAPTURE']).unlink()
                failure_env = dict(env, CAPTURE_EXIT='17')
                result = subprocess.run(['bash', str(script)], cwd=root, env=failure_env, capture_output=True)
                self.assertEqual(result.returncode, 17)
                self.assertEqual(len(Path(env['CAPTURE']).read_text().splitlines()), 1)
                Path(env['CAPTURE']).unlink()

    def test_original_entry_training_automatic_and_independent_test_in_temporary_project(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copy2(ROOT / 'run.py', root / 'run.py')
            for directory in ('models', 'layers', 'exp', 'data_provider', 'utils'):
                shutil.copytree(ROOT / directory, root / directory,
                                ignore=shutil.ignore_patterns('__pycache__'))
            data = root / 'data with spaces'
            data.mkdir()
            make_h5(data / 'gjemnessund.h5')
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
            for name in MODELS:
                with self.subTest(model=name):
                    checkpoints = root / 'checkpoints' / name
                    common = [sys.executable, '-B', '-u', 'run.py', '--model', name,
                              '--model_id', name + '_synthetic_only', '--data', 'Gjemnessund',
                              '--root_path', str(data), '--data_path', 'gjemnessund.h5',
                              '--checkpoints', str(checkpoints), '--features', 'M',
                              '--seq_len', '24', '--pred_len', '12', '--label_len', '0',
                              '--patch_size', '6', '--enc_in', '10', '--time_dim', '6',
                              '--dsa', '1', '--dsb', '1', '--ksize', '5', '--d_model', '16',
                              '--inv_len', '96', '--gje_cycle_len', '96',
                              '--gje_window_stride', '6', '--gje_max_batches', '1',
                              '--batch_size', '2', '--train_epochs', '1', '--patience', '1',
                              '--num_workers', '0', '--learning_rate', '0.005', '--itr', '1',
                              '--rec_lambda', '0', '--auxi_lambda', '1']
                    for training in (1, 0):
                        result = subprocess.run(common + ['--is_training', str(training)],
                                                cwd=root, env=env, capture_output=True,
                                                text=True, timeout=120)
                        self.assertEqual(result.returncode, 0, result.stdout[-2500:] + result.stderr[-3500:])
                        files = list(checkpoints.glob('*/checkpoint.pth'))
                        self.assertEqual(len(files), 1)
                        setting = files[0].parent.name
                        metrics = json.loads((root / 'results' / setting / 'gje_metrics.json').read_text())
                        self.assertTrue(metrics['debug'])
                        self.assertEqual(len(metrics['per_channel']), 10)
                        if training:
                            expected = metrics
                        else:
                            for key in ('mse', 'mae', 'acceleration_mse', 'acceleration_mae'):
                                self.assertEqual(expected[key], metrics[key])
                    metadata = json.loads((files[0].parent / 'gje_run.json').read_text())
                    self.assertEqual(metadata['model_name'], name)
                    self.assertEqual(metadata['gje_physics_calibration']['acceleration_unit'], 'g')
                    self.assertIsNotNone(metadata['gje_physics_calibration']['protocol_signature'])


if __name__ == '__main__':
    unittest.main()
