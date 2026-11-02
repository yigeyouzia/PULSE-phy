"""PULSE_FACT checks; run: python -B -m unittest discover -s tests -p test_pulse_fact.py -v."""

import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from layers.pulse_fact.fact_block import PULSEFACTBlock
from layers.pulse_fact.vendor.fact_conv import Conv_block2d
from models import PULSE, PULSE_FACT


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ETTh1.sh", "ETTh2.sh", "ETTm1.sh", "ETTm2.sh", "electricity.sh", "traffic.sh",
            "Weather.sh", "Solar.sh", "PEMS03.sh", "PEMS04.sh", "PEMS07.sh", "PEMS08.sh")


def fact_config(**overrides):
    values = dict(seq_len=24, pred_len=12, enc_in=7, d_model=8, inv_len=24,
                  patch_size=6, time_dim=1, dsa=1, dsb=1, ksize=3,
                  use_amp=False, use_multi_gpu=False, use_gpu=False, label_len=0,
                  model="PULSE_FACT", data="ETTh1", features="M", output_attention=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def fact_model(config=None):
    with redirect_stdout(io.StringIO()):
        return PULSE_FACT.Model(config or fact_config())


def fact_batch(config, device="cpu", batch_size=2):
    return (torch.randn(batch_size, config.seq_len, config.enc_in, device=device),
            torch.arange(batch_size, device=device),
            torch.randn(batch_size, config.seq_len, config.time_dim, device=device),
            torch.randn(batch_size, config.pred_len, config.time_dim, device=device))


class PULSEFACTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(2024)
        np.random.seed(2024)

    def test_source_block_forward_backward_and_running_stats_match(self):
        for channels in (7, 21):
            for training in (True, False):
                with self.subTest(channels=channels, training=training):
                    source = Conv_block2d(SimpleNamespace(
                        enc_in=channels, freq="n", num_kernels=2, d_ff=16, dropout=0.1),
                        dilation=2, d_model=8).double().train(training)
                    adapted = PULSEFACTBlock(channels, 8, 16, 2, 2, 0.1).double().train(training)
                    adapted.block.load_state_dict(source.state_dict())
                    x = torch.randn(2, channels, 8, dtype=torch.float64, requires_grad=True)
                    other = x.detach().clone().requires_grad_(True)
                    torch.manual_seed(42)
                    expected = source(x.transpose(1, 2)).transpose(1, 2)
                    torch.manual_seed(42)
                    actual = adapted(other)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    expected.square().mean().backward()
                    actual.square().mean().backward()
                    torch.testing.assert_close(other.grad, x.grad, rtol=1e-12, atol=1e-12)
                    for (name, old), (_, new) in zip(source.named_parameters(), adapted.block.named_parameters()):
                        torch.testing.assert_close(new.grad, old.grad, rtol=1e-12, atol=1e-12, msg=name)
                    for name, value in source.named_buffers():
                        torch.testing.assert_close(dict(adapted.block.named_buffers())[name], value)

    def test_small_square_and_large_variable_grids_have_finite_gradients(self):
        for channels in (1, 2, 4, 7, 9, 21, 321, 883):
            with self.subTest(channels=channels):
                block = PULSEFACTBlock(channels, hidden_dim=8, d_ff=16, dropout=0.0)
                x = torch.randn(2, channels, 8, requires_grad=True)
                # Independent cyclic-padding reference also covers cases where
                # the original source's single prefix cannot fill the grid.
                reference = x.transpose(1, 2)
                extra = block.grid_channels - channels
                if extra:
                    reference = torch.cat([reference] + [reference[:, :, i % channels:i % channels + 1]
                                                         for i in range(extra)], dim=-1)
                state = {k: v.clone() for k, v in block.block.state_dict().items()}
                expected = block.block(reference)[:, :, :channels].transpose(1, 2)
                block.block.load_state_dict(state)
                actual = block(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.shape, x.shape)
                actual.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in block.parameters()))

    def test_identity_fact_preserves_original_pulse_initialization_and_behavior(self):
        config = fact_config()
        torch.manual_seed(2024)
        baseline = PULSE.Model(config)
        torch.manual_seed(2024)
        adapted = fact_model(config)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(adapted.state_dict()[name], value, rtol=0, atol=0)
        adapted.fact_block = nn.Identity()
        batch = fact_batch(config)
        for training, mixing in ((False, True), (True, False), (True, True)):
            baseline.train(training)
            adapted.train(training)
            np.random.seed(17)
            torch.manual_seed(42)
            expected = baseline(*batch, do_mixup=mixing)
            np.random.seed(17)
            torch.manual_seed(42)
            actual = adapted(*batch, do_mixup=mixing)
            if training:
                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                if mixing:
                    self.assertEqual(actual[1][0], expected[1][0])
                    torch.testing.assert_close(actual[1][1], expected[1][1], rtol=0, atol=0)
                else:
                    self.assertIsNone(actual[1])
            else:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_sam_keeps_mixed_statistics_and_original_router_anchor(self):
        config = fact_config()
        model = fact_model(config).train()
        x, cycle, xm, ym = fact_batch(config)
        timestamp_x = model.time_enc_x(xm)
        original_input, anchor, mean, var = model.norm(x, cycle, timestamp_x)
        sigma = torch.sqrt(var + 1e-5)
        captured = []
        hook = model.backbone[0].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        with patch("models.PULSE_FACT.np.random.beta", return_value=0.3), \
                patch.object(model, "denorm", wraps=model.denorm) as denorm:
            prediction, (lam, index) = model(x, cycle, xm, ym)
        hook.remove()
        torch.testing.assert_close(captured[0].transpose(1, 2),
                                   lam * original_input + (1 - lam) * original_input[index])
        torch.testing.assert_close(denorm.call_args.args[1], anchor)
        torch.testing.assert_close(denorm.call_args.args[2], lam * mean + (1 - lam) * mean[index])
        expected_var = (lam * sigma + (1 - lam) * sigma[index]).square() - 1e-5
        torch.testing.assert_close(denorm.call_args.args[3], expected_var)
        torch.testing.assert_close(denorm.call_args.args[4], model.time_enc_y(ym))
        self.assertEqual(prediction.shape, (2, 12, 7))

    def test_model_fft_backward_all_parameters_and_eval_contract(self):
        for channels in (1, 7, 21):
            config = fact_config(enc_in=channels)
            model = fact_model(config).train()
            batch = fact_batch(config)
            prediction, (lam, index) = model(*batch)
            target = torch.randn_like(prediction)
            mixed_target = lam * target + (1 - lam) * target[index]
            loss = (torch.fft.fft(prediction, dim=1) - torch.fft.fft(mixed_target, dim=1)).abs().mean()
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            model.eval()
            with patch("models.PULSE_FACT.np.random.beta", side_effect=AssertionError), \
                    patch("torch.randperm", side_effect=AssertionError):
                first = model(*batch)
                second = model(*batch)
            self.assertEqual(first.shape, (2, 12, channels))
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_checkpoint_roundtrip_and_effective_configuration_guard(self):
        config = fact_config()
        model = fact_model(config).train()
        batch = fact_batch(config)
        model(*batch)[0].square().mean().backward()
        torch.optim.Adam(model.parameters(), lr=1e-3).step()
        model.eval()
        expected = model(*batch)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = fact_model(config).eval()
        state = torch.load(buffer, map_location="cpu")
        restored.load_state_dict(state)
        torch.testing.assert_close(restored(*batch), expected, rtol=0, atol=0)
        mismatched = fact_model(fact_config(pulse_fact_dropout=0.2))
        with self.assertRaisesRegex(RuntimeError, "configuration differs"):
            mismatched.load_state_dict(state)

    def test_settings_are_logged_independent_and_do_not_mutate_arguments(self):
        config = fact_config(d_ff=2048, dropout=0.0, pulse_fact_d_ff=32)
        before = vars(config).copy()
        output = io.StringIO()
        with redirect_stdout(output):
            model = PULSE_FACT.Model(config)
        self.assertEqual(vars(config), before)
        settings = json.loads(output.getvalue().split("PULSE_FACT effective settings: ")[1])
        self.assertEqual(settings, model.get_extra_state())
        self.assertEqual(model.fact_d_ff, 32)
        self.assertEqual(model.fact_hidden_dim, 512)
        self.assertEqual(model.fact_dropout, 0.1)
        self.assertEqual(model.d_model, 8)

    def test_invalid_configs_and_unsupported_training_paths_are_explicit(self):
        for change in ({"pulse_fact_d_ff": 0}, {"pulse_fact_dilation": 0},
                       {"pulse_fact_num_kernels": 0}, {"pulse_fact_dropout": 1.0},
                       {"label_len": 1}, {"patch_size": 7}, {"dsa": 8},
                       {"seq_len": 1}, {"use_amp": True}, {"use_multi_gpu": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                fact_model(fact_config(**change))

    def test_existing_experiment_registration_and_validation(self):
        from exp.exp_main import Exp_Main
        config = fact_config()
        experiment = Exp_Main.__new__(Exp_Main)
        experiment.args, experiment.device = config, torch.device("cpu")
        with redirect_stdout(io.StringIO()):
            experiment.model = experiment._build_model()
        self.assertIsInstance(experiment.model, PULSE_FACT.Model)
        x, cycle, xm, ym = fact_batch(config)
        target = torch.randn(2, 12, 7)
        dataset = torch.utils.data.TensorDataset(x, target, xm, ym, cycle)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        experiment.model.eval()
        with torch.no_grad():
            expected = (experiment.model(x, cycle, xm, ym) - target).square().mean().item()
        actual = experiment.vali(dataset, loader, nn.MSELoss())
        self.assertAlmostEqual(actual, expected, places=6)

    def fact_strict_optimizer_step(self, device):
        previous = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
                    torch.backends.cudnn.benchmark)
        try:
            with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                config = fact_config()
                model = fact_model(config).to(device).train()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                prediction, (lam, index) = model(*fact_batch(config, device))
                target = torch.randn_like(prediction)
                target = lam * target + (1 - lam) * target[index]
                (torch.fft.fft(prediction, dim=1) - torch.fft.fft(target, dim=1)).abs().mean().backward()
                weight = model.fact_block.block.conv[0][1].conv1.weight
                before = weight.detach().clone()
                optimizer.step()
                self.assertFalse(torch.equal(before, weight))
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(previous[0])
            torch.backends.cudnn.deterministic = previous[1]
            torch.backends.cudnn.benchmark = previous[2]

    def test_strict_deterministic_optimizer_step_cpu(self):
        self.fact_strict_optimizer_step("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; run on the training server")
    def test_strict_deterministic_optimizer_step_cuda(self):
        self.fact_strict_optimizer_step("cuda")

    def test_source_copy_and_license_hashes_match_manifest(self):
        vendor = ROOT / "layers/pulse_fact/vendor"
        manifest = json.loads((vendor / "sources.json").read_text())
        source = manifest["sources"][0]
        self.assertEqual(hashlib.sha256((vendor / source["copied_file"]).read_bytes()).hexdigest(),
                         source["copied_sha256"])
        license_record = manifest["licenses"][0]
        self.assertEqual(hashlib.sha256((vendor / license_record["copied_file"]).read_bytes()).hexdigest(),
                         license_record["sha256"])


class PULSEFACTScriptTests(unittest.TestCase):
    def fact_parser(self):
        tree = ast.parse((ROOT / "run.py").read_text())
        statements = [n for n in tree.body if (
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "parser" for t in n.targets)
        ) or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
              and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name)
              and n.value.func.value.id == "parser" and n.value.func.attr == "add_argument")]
        namespace = {"argparse": argparse}
        exec(compile(ast.Module(body=statements, type_ignores=[]), "run.py:parser", "exec"), namespace)
        return namespace["parser"]

    def fact_capture(self, script, directory, exit_code=0, message="stdout"):
        directory = Path(directory)
        (directory / "run.py").touch()
        stub = directory / "python"
        stub.write_text("#!" + sys.executable + "\nimport json,os,sys\n"
                        "with open(os.environ['FACT_CAPTURE'],'a') as out:\n"
                        " out.write(json.dumps(sys.argv[1:])+'\\n')\n"
                        "print(os.environ['FACT_MESSAGE'],flush=True)\n"
                        "print('stderr',file=sys.stderr,flush=True)\n"
                        "sys.exit(int(os.environ['FACT_EXIT']))\n")
        stub.chmod(0o755)
        capture = directory / "args.jsonl"
        env = dict(os.environ, PATH=str(directory) + os.pathsep + os.environ["PATH"],
                   FACT_CAPTURE=str(capture), FACT_EXIT=str(exit_code), FACT_MESSAGE=message)
        result = subprocess.run(["bash", str(script)], cwd=directory, env=env, text=True, capture_output=True)
        commands = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
        return result, commands

    def test_all_twelve_scripts_preserve_48_source_configs_and_output_layout(self):
        parser = self.fact_parser()
        for filename in DATASETS:
            with tempfile.TemporaryDirectory() as old_dir, tempfile.TemporaryDirectory() as new_dir:
                old_result, source = self.fact_capture(ROOT / "scripts" / filename, old_dir)
                self.assertEqual(old_result.returncode, 0)
                script = ROOT / "scripts/PULSE_FACT" / filename
                subprocess.run(["bash", "-n", str(script)], check=True)
                result, commands = self.fact_capture(script, new_dir)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(commands), 4)
                log_paths = [Path(line[5:]) for line in result.stdout.splitlines() if line.startswith("Log: ")]
                self.assertEqual(len(log_paths), 4)
                for original, current, log in zip(source, commands, log_paths):
                    old, new = parser.parse_args(original[2:]), parser.parse_args(current[2:])
                    for key in vars(old).keys() - {"model", "model_id", "checkpoints"}:
                        self.assertEqual(getattr(old, key), getattr(new, key), (filename, key))
                    self.assertEqual(new.model, "PULSE_FACT")
                    self.assertEqual(new.checkpoints, "./checkpoints/PULSE_FACT/")
                    dataset = old.model_id.rsplit("_", 2)[0]
                    expected_log = (Path(new_dir) / "logs/PULSE_FACT" / dataset /
                                    "PULSE_FACT_{}_{}.log".format(new.seq_len, new.pred_len))
                    self.assertEqual(log.resolve(), expected_log.resolve())
                    self.assertEqual(log.read_text(), "stdout\nstderr\n")

    def test_fixed_logs_overwrite_but_checkpoint_experiment_ids_change(self):
        parser = self.fact_parser()
        with tempfile.TemporaryDirectory() as directory:
            script = ROOT / "scripts/PULSE_FACT/ETTh1.sh"
            first, old = self.fact_capture(script, directory)
            second, all_commands = self.fact_capture(script, directory, message="second")
            self.assertEqual(first.returncode, 0)
            self.assertEqual(second.returncode, 0)
            self.assertEqual(len(all_commands), 8)
            self.assertTrue({parser.parse_args(c[2:]).model_id for c in old}.isdisjoint(
                {parser.parse_args(c[2:]).model_id for c in all_commands[4:]}))
            logs = list((Path(directory) / "logs").rglob("*.log"))
            self.assertEqual(len(logs), 4)
            self.assertTrue(all(p.read_text() == "second\nstderr\n" for p in logs))

    def test_launcher_covers_twelve_datasets_and_stops_on_failure(self):
        parser = self.fact_parser()
        launcher = ROOT / "scripts/run_PULSE_FACT.sh"
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        for exit_code in (0, 37):
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
                result, commands = self.fact_capture(launcher, directory, exit_code)
                self.assertEqual(result.returncode, exit_code, result.stderr)
                self.assertEqual(len(commands), 48 if exit_code == 0 else 1)
                if exit_code == 0:
                    args = [parser.parse_args(c[2:]) for c in commands]
                    self.assertEqual({a.model for a in args}, {"PULSE_FACT"})
                    self.assertEqual([a.data_path for a in args[::4]],
                                     ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv", "electricity.csv",
                                      "traffic.csv", "weather.csv", "solar_AL.txt", "PEMS03.npz", "PEMS04.npz",
                                      "PEMS07.npz", "PEMS08.npz"])


if __name__ == "__main__":
    unittest.main()
