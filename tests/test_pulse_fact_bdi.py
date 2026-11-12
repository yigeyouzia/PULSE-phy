"""BDI checks: python -B -m unittest discover -s tests -p test_pulse_fact_bdi.py -v."""

import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from layers.pulse_fact.fact_block import PULSEFACTBlock
from layers.pulse_fact_bdi.frequency import HistoricalResidualBandSplit
from layers.pulse_fact_bdi.interaction import PULSEFACTBDIBlock
from models import PULSE_FACT, PULSE_FACT_BDI
import test_pulse_fact as source_tests


ROOT = Path(__file__).resolve().parents[1]
SCOPES = {"PEMS03": (12, 24, 48, 96), "PEMS04": (12, 24, 48, 96),
          "ETTh1": (96, 192, 336, 720)}


def config(**changes):
    return source_tests.fact_config(model="PULSE_FACT_BDI", add_noise=False, **changes)


def make(args=None, module=PULSE_FACT_BDI):
    with redirect_stdout(io.StringIO()):
        return module.Model(args or config())


class BDIFrequencyTests(unittest.TestCase):
    def test_real_time_frequency_bands_reconstruct_and_separate_known_signals(self):
        for length in (25, 96):
            split = HistoricalResidualBandSplit(length).double()
            t = torch.arange(length, dtype=torch.float64)
            low = 3 + torch.cos(2 * torch.pi * t / length)
            high = 2 * torch.sin(2 * torch.pi * (length // 3) * t / length)
            signal = (low + high).reshape(1, length, 1).repeat(2, 1, 3)
            actual_low, actual_high = split(signal)
            torch.testing.assert_close(actual_low, low.reshape(1, length, 1).expand_as(signal),
                                       rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(actual_high, high.reshape(1, length, 1).expand_as(signal),
                                       rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(actual_low + actual_high, signal, rtol=1e-12, atol=1e-12)
        self.assertEqual(HistoricalResidualBandSplit(96).low_mask.sum().item(), 13)

    def test_gradients_reach_both_bands(self):
        split = HistoricalResidualBandSplit(9).double()
        x = torch.randn(1, 9, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(split, (x,), eps=1e-6, atol=1e-5))

    def test_invalid_cutoffs_shapes_and_precision(self):
        for ratio in (0, 1, -0.1, float("nan"), float("inf"), True, "0.25"):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                HistoricalResidualBandSplit(24, ratio)
        with self.assertRaises(ValueError):
            HistoricalResidualBandSplit(1)
        split = HistoricalResidualBandSplit(24)
        for x in (torch.randn(2, 23, 7), torch.randn(24, 7), torch.randn(2, 24, 7).half()):
            with self.assertRaises(ValueError):
                split(x)


class BDIModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(2024)
        np.random.seed(2024)

    def test_same_inputs_recover_original_fact_block_gradients_and_statistics(self):
        for channels in (1, 7, 21):
            old = PULSEFACTBlock(channels, 8, 16, 2, 2, 0.1).double().train()
            new = PULSEFACTBDIBlock(channels, 8, 16, 2, 0.1).double().train()
            new.load_state_dict(old.state_dict())
            x = torch.randn(2, channels, 8, dtype=torch.float64, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            torch.manual_seed(42)
            expected = old(x)
            torch.manual_seed(42)
            actual = new(y, y, y)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected.square().mean().backward()
            actual.square().mean().backward()
            torch.testing.assert_close(y.grad, x.grad, rtol=1e-10, atol=1e-12)
            for (name, before), (_, after) in zip(old.named_parameters(), new.named_parameters()):
                torch.testing.assert_close(after.grad, before.grad, rtol=1e-10, atol=1e-12, msg=name)
            for name, value in old.named_buffers():
                torch.testing.assert_close(dict(new.named_buffers())[name], value, rtol=0, atol=0)

    def test_band_routing_single_residual_and_variable_grid_edge_cases(self):
        class Zero(nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)
        for channels in (1, 2, 4, 7, 9, 307, 358, 883):
            block = PULSEFACTBDIBlock(channels, 8, 16, 2, 0.0)
            inputs = [torch.randn(2, channels, 8, requires_grad=True) for _ in range(3)]
            captured = []
            handles = [branch.register_forward_pre_hook(
                lambda module, args: captured.append(args[0].detach().clone()))
                for branch in block.block.conv]
            out = block(*inputs)
            for handle in handles:
                handle.remove()
            torch.testing.assert_close(captured[0], block.variable_grid(inputs[2]))
            torch.testing.assert_close(captured[1], block.variable_grid(inputs[1]))
            out.square().mean().backward()
            self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in inputs))
            block.block.conv = nn.ModuleList([Zero(), Zero()])
            block.block.conv_ffn = Zero()
            torch.testing.assert_close(block(*inputs), inputs[0], rtol=0, atol=0)

    def test_parameter_capacity_initialization_and_inherited_restoration_match_pf(self):
        torch.manual_seed(17)
        old = make(module=PULSE_FACT)
        torch.manual_seed(17)
        new = make()
        self.assertEqual(sum(p.numel() for p in old.parameters()), sum(p.numel() for p in new.parameters()))
        for name, value in old.named_parameters():
            torch.testing.assert_close(dict(new.named_parameters())[name], value, rtol=0, atol=0)
        def classes(path):
            return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)}
        original, current = classes(ROOT / "models/PULSE_FACT.py"), classes(ROOT / "models/PULSE_FACT_BDI.py")
        self.assertEqual(ast.dump(original["ReVIN"]), ast.dump(current["ReVIN"]))
        for name in ("phase_router", "denorm"):
            methods = [{n.name: n for n in c["Model"].body if isinstance(n, ast.FunctionDef)}
                       for c in (original, current)]
            self.assertEqual(ast.dump(methods[0][name]), ast.dump(methods[1][name]))
        args = config()
        x, cycle, xm, _ = source_tests.fact_batch(args)
        timestamp = old.time_enc_x(xm)
        expected = old.norm(x, cycle, timestamp)
        actual = new.norm_bdi(x, cycle, timestamp)
        for before, after in zip(expected, actual[:4]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(actual[0], actual[4] + actual[1], rtol=0, atol=0)

    def test_sam_pairs_main_residual_and_statistics_once_before_frequency_split(self):
        args = config()
        model = make(args).train()
        inputs = source_tests.fact_batch(args)
        x, cycle, xm, ym = inputs
        main, anchor, mean, var, residual = model.norm_bdi(x, cycle, model.time_enc_x(xm))
        permutation = torch.tensor([1, 0])
        captured = []
        hook = model.backbone[0].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        with patch("models.PULSE_FACT_BDI.np.random.beta", return_value=0.3) as beta, \
                patch("models.PULSE_FACT_BDI.torch.randperm", return_value=permutation) as shuffle, \
                patch.object(model.band_split, "forward", wraps=model.band_split.forward) as split, \
                patch.object(model, "denorm", wraps=model.denorm) as restore:
            prediction, (lam, index) = model(*inputs)
        hook.remove()
        self.assertEqual((beta.call_count, shuffle.call_count, split.call_count, restore.call_count), (1, 1, 1, 1))
        self.assertEqual(len(captured), 3)
        self.assertEqual(lam, 0.3)
        torch.testing.assert_close(index, permutation)
        torch.testing.assert_close(split.call_args.args[0], lam * residual + (1-lam) * residual[index])
        torch.testing.assert_close(captured[0].transpose(1, 2), lam * main + (1-lam) * main[index])
        torch.testing.assert_close(restore.call_args.args[1], anchor)
        torch.testing.assert_close(restore.call_args.args[2], lam * mean + (1-lam) * mean[index])
        sigma = torch.sqrt(var + 1e-5)
        torch.testing.assert_close(restore.call_args.args[3],
                                   (lam * sigma + (1-lam) * sigma[index]).square() - 1e-5)
        self.assertEqual(prediction.shape, (2, 12, 7))

    def strict_fft_optimizer_steps(self, device):
        previous = (torch.are_deterministic_algorithms_enabled(),
                    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        environment = patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        environment.start()
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            args = config()
            model = make(args).to(device).train()
            batch = source_tests.fact_batch(args, device=device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            watched = model.fact_block.block.conv[0][1].kernels[0].weight
            before = watched.detach().clone()
            for _ in range(2):
                optimizer.zero_grad()
                prediction, (lam, index) = model(*batch)
                target = torch.randn_like(prediction)
                target = lam * target + (1-lam) * target[index]
                loss = (torch.fft.fft(prediction, dim=1)-torch.fft.fft(target, dim=1)).abs().mean()
                loss.backward()
                for name, value in model.named_parameters():
                    self.assertIsNotNone(value.grad, name)
                    self.assertTrue(torch.isfinite(value.grad).all(), name)
                optimizer.step()
            self.assertGreater((watched-before).abs().max().item(), 0)
            model.eval()
            with torch.no_grad(), patch("models.PULSE_FACT_BDI.np.random.beta", side_effect=AssertionError), \
                    patch("models.PULSE_FACT_BDI.torch.randperm", side_effect=AssertionError):
                first = model(*batch)
                second = model(*batch)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        finally:
            torch.use_deterministic_algorithms(previous[0])
            torch.backends.cudnn.deterministic = previous[1]
            torch.backends.cudnn.benchmark = previous[2]
            environment.stop()

    def test_strict_fft_optimizer_steps_cpu(self):
        self.strict_fft_optimizer_steps("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; run on training server")
    def test_strict_fft_optimizer_steps_cuda(self):
        self.strict_fft_optimizer_steps("cuda")

    def test_checkpoint_roundtrip_and_configuration_guard(self):
        args = config()
        model = make(args).train()
        inputs = source_tests.fact_batch(args)
        model(*inputs)[0].square().mean().backward()
        torch.optim.Adam(model.parameters(), lr=1e-3).step()
        model.eval()
        expected = model(*inputs)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=False)
        restored = make(args).eval()
        restored.load_state_dict(state)
        torch.testing.assert_close(restored(*inputs), expected, rtol=0, atol=0)
        for changes in ({"pulse_fact_bdi_cutoff_ratio": 0.5}, {"pulse_fact_bdi_dropout": 0.2}):
            with self.assertRaisesRegex(RuntimeError, "configuration differs"):
                make(config(**changes)).load_state_dict(state)

    def test_independent_settings_and_invalid_configs(self):
        args = config(d_ff=2048, dropout=0, pulse_fact_d_ff=64, pulse_fact_bdi_d_ff=32)
        before = vars(args).copy()
        model = make(args)
        self.assertEqual(vars(args), before)
        self.assertEqual(model.fact_d_ff, 32)
        self.assertEqual(model.fact_dropout, 0.1)
        self.assertEqual(model.get_extra_state()["settings"]["low_dilation"], 2)
        for change in ({"pulse_fact_bdi_d_ff": 0}, {"pulse_fact_bdi_cutoff_ratio": 1},
                       {"pulse_fact_bdi_num_kernels": 0}, {"pulse_fact_bdi_dropout": 1},
                       {"use_amp": True}, {"use_multi_gpu": True}, {"label_len": 1},
                       {"patch_size": 7}, {"dsa": 8}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                make(config(**change))

    def test_existing_experiment_registration_and_validation(self):
        from exp.exp_main import Exp_Main
        args = config()
        experiment = Exp_Main.__new__(Exp_Main)
        experiment.args, experiment.device = args, torch.device("cpu")
        with redirect_stdout(io.StringIO()):
            experiment.model = experiment._build_model()
        self.assertIsInstance(experiment.model, PULSE_FACT_BDI.Model)
        x, cycle, xm, ym = source_tests.fact_batch(args)
        target = torch.randn(2, args.pred_len, args.enc_in)
        dataset = torch.utils.data.TensorDataset(x, target, xm, ym, cycle)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        self.assertTrue(np.isfinite(experiment.vali(dataset, loader, nn.MSELoss())))

    def test_source_hashes_and_protected_files(self):
        record = json.loads((ROOT / "notes/data/pulse_fact_bdi_sources.json").read_text())
        for item in record["sources"]:
            if not item.get("runtime_required", True) and not (ROOT / item["path"]).exists():
                continue  # Optional original ZIP provenance; vendored source is required.
            self.assertEqual(hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest(), item["sha256"])
        for item in record["protected_files"]:
            self.assertEqual(hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest(), item["sha256"])


class BDIScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_twelve_configs_preserve_reference_parameters_and_shapes(self):
        helper = source_tests.PULSEFACTScriptTests()
        parser = helper.fact_parser()
        for dataset, horizons in SCOPES.items():
            script = ROOT / "scripts/PULSE_FACT_BDI" / (dataset + ".sh")
            subprocess.run(["bash", "-n", str(script)], check=True)
            body = script.read_text()
            self.assertIn("set -e", body)
            self.assertNotIn("function ", body)
            self.assertNotIn("() {", body)
            self.assertEqual(body.count("python -u run.py"), 4)
            with tempfile.TemporaryDirectory() as before, tempfile.TemporaryDirectory() as after:
                result, original = helper.fact_capture(ROOT / "scripts/PULSE_FACT" / (dataset + ".sh"), before)
                self.assertEqual(result.returncode, 0)
                source = [parser.parse_args(c[2:]) for c in original]
                result, commands = helper.fact_capture(script, after)
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = [parser.parse_args(c[2:]) for c in commands]
                self.assertEqual(tuple(a.pred_len for a in actual), horizons)
                reported = [Path(line[5:]).resolve() for line in result.stdout.splitlines() if line.startswith("Log: ")]
                expected = []
                for old, new in zip(source, actual):
                    for key in vars(old).keys() - {"model", "model_id", "checkpoints"}:
                        self.assertEqual(getattr(new, key), getattr(old, key), (dataset, key))
                    self.assertEqual(new.model, "PULSE_FACT_BDI")
                    self.assertEqual(new.checkpoints, "./checkpoints/PULSE_FACT_BDI/")
                    log = Path(after) / "logs/PULSE_FACT_BDI" / dataset / ("PULSE_FACT_BDI_96_%d.log" % new.pred_len)
                    expected.append(log.resolve())
                    self.assertEqual(log.read_text(), "stdout\nstderr\n")
                    model = make(new).eval()
                    with torch.no_grad():
                        prediction = model(*source_tests.fact_batch(new, batch_size=1))
                    self.assertEqual(prediction.shape, (1, new.pred_len, new.enc_in))
                    self.assertTrue(torch.isfinite(prediction).all())
                self.assertEqual(reported, expected)

    def test_failure_stops_and_repeated_logs_overwrite_with_distinct_checkpoint_ids(self):
        helper = source_tests.PULSEFACTScriptTests()
        parser = helper.fact_parser()
        for dataset in SCOPES:
            with tempfile.TemporaryDirectory() as directory:
                result, commands = helper.fact_capture(ROOT / "scripts/PULSE_FACT_BDI" / (dataset + ".sh"),
                                                       directory, exit_code=37)
                self.assertEqual(result.returncode, 37)
                self.assertEqual(len(commands), 1)
        with tempfile.TemporaryDirectory() as directory:
            script = ROOT / "scripts/PULSE_FACT_BDI/PEMS04.sh"
            first, old = helper.fact_capture(script, directory)
            second, all_commands = helper.fact_capture(script, directory, message="second")
            self.assertEqual((first.returncode, second.returncode), (0, 0))
            self.assertEqual(len(all_commands), 8)
            self.assertTrue({parser.parse_args(c[2:]).model_id for c in old}.isdisjoint(
                {parser.parse_args(c[2:]).model_id for c in all_commands[4:]}))
            logs = list((Path(directory) / "logs").rglob("*.log"))
            self.assertEqual(len(logs), 4)
            self.assertTrue(all(path.read_text() == "second\nstderr\n" for path in logs))


if __name__ == "__main__":
    unittest.main()
