"""Run from repository root: python -B -m unittest discover -s tests -p test_pulse_scf.py -v."""

import argparse
import ast
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

import torch

from layers.pulse_scf.config import SCFConfig
from layers.pulse_scf.interaction import SCFSTAR, SCFFACTInteraction
from layers.pulse_scf.vendor.fact_conv import Conv_block2d
from layers.pulse_scf.vendor.softs_star import STAR
from models import PULSE, PULSE_SOFTS, PULSE_SCF_SIMPLE, PULSE_SCF


def scf_test_config(**overrides):
    values = dict(
        seq_len=24, pred_len=12, enc_in=7, d_model=16, inv_len=24,
        patch_size=6, time_dim=2, dsa=1, dsb=1, ksize=3,
        scf_d_model=8, scf_d_core=4, scf_d_ff=16, scf_e_layers=1,
        scf_dropout=0.0, scf_fact_dilation=2, scf_num_kernels=2,
        scf_gate_hidden=8, use_amp=False, use_multi_gpu=False,
        use_gpu=False, model="PULSE_SCF", data="ETTh1", label_len=0,
        features="M", output_attention=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def scf_test_batch(config, batch_size=2):
    return (
        torch.randn(batch_size, config.seq_len, config.enc_in),
        torch.arange(batch_size),
        torch.randn(batch_size, config.seq_len, config.time_dim),
        torch.randn(batch_size, config.pred_len, config.time_dim),
    )


class SCFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(2024)

    def test_star_preserves_source_training_and_inference(self):
        source, adapted = STAR(8, 4), SCFSTAR(8, 4)
        adapted.load_state_dict(source.state_dict())
        x = torch.randn(2, 7, 8)
        for training in (True, False):
            source.train(training)
            adapted.train(training)
            torch.manual_seed(42)
            expected, _ = source(x)
            torch.manual_seed(42)
            actual, core = adapted(x, return_core=True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(core.shape, (2, 1, 4))

    def test_fact_update_preserves_source_block(self):
        settings = SCFConfig.from_configs(scf_test_config())
        for channels in (7, 10, 21):
            config = SimpleNamespace(enc_in=channels, freq="n", num_kernels=2,
                                     d_ff=16, dropout=0.0)
            source = Conv_block2d(config, dilation=2, d_model=8)
            adapted = SCFFACTInteraction(channels, settings)
            adapted.load_state_dict(source.state_dict())
            for training in (True, False):
                source.train(training)
                adapted.train(training)
                x = torch.randn(2, channels, 8)
                expected = source(x.transpose(1, 2)).transpose(1, 2)
                actual = x + adapted(x)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_fact_padding_handles_small_and_square_channel_counts(self):
        settings = SCFConfig.from_configs(scf_test_config())
        for channels in (1, 2, 4, 9, 321):
            block = SCFFACTInteraction(channels, settings)
            x = torch.randn(1, channels, 8, requires_grad=True)
            delta = block(x)
            self.assertEqual(delta.shape, x.shape)
            delta.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_variants_forward_backward_with_pulse_mixup_contract(self):
        for module in (PULSE_SOFTS, PULSE_SCF_SIMPLE, PULSE_SCF):
            config = scf_test_config()
            model = module.Model(config).train()
            batch = scf_test_batch(config)
            prediction, (lam, indices) = model(*batch)
            target = torch.randn_like(prediction)
            target = lam * target + (1 - lam) * target[indices]
            # The original non-AMP default is the complex FFT MAE objective.
            loss = (torch.fft.fft(prediction, dim=1)
                    - torch.fft.fft(target, dim=1)).abs().mean()
            loss.backward()
            self.assertEqual(prediction.shape, (2, 12, 7))
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, (module.__name__, name))
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_sam_history_condition_matches_mixed_input_and_original_router(self):
        config = scf_test_config()
        model = PULSE_SCF.Model(config).train()
        x, cycle, xm, ym = scf_test_batch(config)
        anchor = model.codebook[(cycle[:, None] - torch.arange(24)[None]) % 24]
        anchor = anchor + model.time_enc_x(xm)
        residual = x - anchor
        mean = residual.mean(1, keepdim=True)
        sigma = torch.sqrt(residual.var(1, keepdim=True) + model.revin.eps)
        captured = {}

        def capture_backbone(module, args, kwargs):
            captured["input"] = args[0]
            captured["state"] = kwargs["scf_state"]

        hook = model.backbone.register_forward_pre_hook(capture_backbone, with_kwargs=True)
        with patch("models.PULSE_SCF.np.random.beta", return_value=0.3), \
                patch.object(model, "denorm", wraps=model.denorm) as denorm:
            _, (lam, index) = model(x, cycle, xm, ym)
        hook.remove()
        state = captured["state"]
        torch.testing.assert_close(state.anchor + state.residual, captured["input"])
        torch.testing.assert_close(state.anchor, lam * anchor + (1 - lam) * anchor[index])
        torch.testing.assert_close(state.mean, lam * mean + (1 - lam) * mean[index])
        torch.testing.assert_close(state.sigma, lam * sigma + (1 - lam) * sigma[index])
        torch.testing.assert_close(denorm.call_args.args[1], anchor)
        torch.testing.assert_close(denorm.call_args.args[4], model.time_enc_y(ym))

    def test_gate_uses_history_and_receives_condition_gradients(self):
        config = scf_test_config()
        model = PULSE_SCF.Model(config).eval()
        x, cycle, xm, ym = scf_test_batch(config)
        captured = []
        gate = model.backbone.encoder.attn_layers[0].attention.gate
        hook = gate.register_forward_hook(lambda module, args, out: captured.append(out))
        model(x, cycle, xm, ym)
        model(x, cycle, xm, ym + 100)
        hook.remove()
        torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
        self.assertEqual(captured[0].shape, (2, 7, 1))
        self.assertTrue(((captured[0] > 0) & (captured[0] < 1)).all())
        model(x, cycle, xm, ym).square().mean().backward()
        self.assertGreater(gate.anchor_encoder.weight.grad.abs().sum().item(), 0)
        self.assertGreater(gate.residual_encoder.weight.grad.abs().sum().item(), 0)

    def test_mixup_can_be_disabled_and_eval_never_samples(self):
        config = scf_test_config()
        model = PULSE_SCF.Model(config)
        batch = scf_test_batch(config)
        _, mixup = model(*batch, do_mixup=False)
        self.assertIsNone(mixup)
        model.eval()
        with patch("models.PULSE_SCF.np.random.beta", side_effect=AssertionError), \
                patch("torch.multinomial", side_effect=AssertionError):
            a = model(*batch)
            b = model(*batch, do_mixup=False)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_checkpoint_roundtrip_and_configuration_guard(self):
        config = scf_test_config()
        model = PULSE_SCF.Model(config).eval()
        batch = scf_test_batch(config)
        expected = model(*batch)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        state = torch.load(buffer, map_location="cpu", weights_only=True)
        restored = PULSE_SCF.Model(config).eval()
        restored.load_state_dict(state)
        torch.testing.assert_close(restored(*batch), expected, rtol=0, atol=0)
        mismatched = PULSE_SCF.Model(scf_test_config(scf_dropout=0.2))
        with self.assertRaisesRegex(RuntimeError, "configuration differs"):
            mismatched.load_state_dict(state)

    def test_scf_dimensions_are_independent_and_configs_not_mutated(self):
        config = scf_test_config()
        before = vars(config).copy()
        model = PULSE_SCF.Model(config)
        self.assertEqual(vars(config), before)
        self.assertEqual(model.d_model, 16)
        self.assertEqual(model.backbone.embedding.value_embedding.out_features, 8)

    def test_invalid_or_unsupported_configs_are_explicit(self):
        for override in ({"scf_d_model": 0}, {"scf_dropout": 1.0},
                         {"patch_size": 7}, {"dsa": 8},
                         {"use_amp": True}, {"use_multi_gpu": True}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                PULSE_SCF.Model(scf_test_config(**override))

    def test_original_experiment_registration_and_validation(self):
        from exp.exp_main import Exp_Main
        for name, module in (("PULSE", PULSE), ("PULSE_SOFTS", PULSE_SOFTS),
                             ("PULSE_SCF_SIMPLE", PULSE_SCF_SIMPLE),
                             ("PULSE_SCF", PULSE_SCF)):
            config = scf_test_config(model=name)
            experiment = Exp_Main.__new__(Exp_Main)
            experiment.args = config
            experiment.device = torch.device("cpu")
            experiment.model = experiment._build_model()
            self.assertIsInstance(experiment.model, module.Model)
            x, cycle, xm, ym = scf_test_batch(config)
            target = torch.randn(2, 12, 7)
            dataset = torch.utils.data.TensorDataset(x, target, xm, ym, cycle)
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            experiment.model.eval()
            with torch.no_grad():
                expected = (experiment.model(x, cycle, xm, ym) - target).square().mean().item()
            actual = experiment.vali(dataset, loader, torch.nn.MSELoss())
            self.assertAlmostEqual(actual, expected, places=6)


class SCFScriptTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    models = ("PULSE", "PULSE_SOFTS", "PULSE_SCF_SIMPLE", "PULSE_SCF")
    datasets = ("ETTh1.sh", "ETTh2.sh", "ETTm1.sh", "ETTm2.sh", "PEMS03.sh",
                "PEMS04.sh", "PEMS07.sh", "PEMS08.sh", "Solar.sh", "Weather.sh",
                "electricity.sh", "traffic.sh")

    def scf_run_script(self, script, directory, **environment):
        directory = Path(directory)
        (directory / "run.py").touch()
        stub = directory / "python"
        stub.write_text(
            "#!" + sys.executable + "\n"
            "import json,os,sys\n"
            "with open(os.environ['SCF_ARGV_CAPTURE'],'a') as out:\n"
            " out.write(json.dumps(sys.argv[1:])+'\\n')\n"
            "sys.exit(int(os.environ.get('SCF_STUB_EXIT','0')))\n"
        )
        stub.chmod(0o755)
        capture = directory / "arguments.jsonl"
        env = dict(os.environ, PATH=str(directory) + os.pathsep + os.environ["PATH"],
                   SCF_ARGV_CAPTURE=str(capture), RUN_TAG="argument_check")
        # Isolate the test from optional selections in the caller's shell.
        env.pop("MODEL_NAME", None)
        env.pop("PRED_LENS", None)
        env.update(environment)
        result = subprocess.run(["bash", str(script)], cwd=directory, env=env,
                                text=True, capture_output=True)
        commands = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
        return result, commands

    def scf_original_parser(self):
        tree = ast.parse((self.root / "run.py").read_text())
        statements = [node for node in tree.body if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "parser" for target in node.targets)
        ) or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "parser"
            and node.value.func.attr == "add_argument"
        )]
        namespace = {"argparse": argparse}
        exec(compile(ast.Module(body=statements, type_ignores=[]), "run.py:parser", "exec"), namespace)
        return namespace["parser"]

    def test_scripts_preserve_source_horizon_configs_and_existing_flags(self):
        parser = self.scf_original_parser()
        for filename in self.datasets:
            with tempfile.TemporaryDirectory() as old_dir:
                result, original = self.scf_run_script(self.root / "scripts" / filename, old_dir)
                self.assertEqual(result.returncode, 0, result.stderr)
            for model in self.models:
                with self.subTest(model=model, dataset=filename), tempfile.TemporaryDirectory() as new_dir:
                    script = self.root / "scripts" / model / filename
                    subprocess.run(["bash", "-n", str(script)], check=True)
                    result, commands = self.scf_run_script(script, new_dir)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(len(commands), len(original))
                    logs = [Path(line[len("Log: "):]) for line in result.stdout.splitlines()
                            if line.startswith("Log: ")]
                    self.assertEqual(len(logs), len(commands))
                    for old, new, log in zip(original, commands, logs):
                        self.assertEqual(new[:2], ["-u", "run.py"])
                        old_args = vars(parser.parse_args(old[2:]))
                        new_args = vars(parser.parse_args(new[2:]))
                        self.assertEqual(new_args["model"], model)
                        self.assertIn("argument_check", new_args["model_id"])
                        self.assertEqual(new_args["checkpoints"], "./checkpoints/" + model + "/")
                        for key in old_args.keys() - {"model", "model_id", "checkpoints"}:
                            self.assertEqual(new_args[key], old_args[key], (filename, key))
                        self.assertEqual(log.name, "{}_{}_{}.log".format(
                            model, new_args["seq_len"], new_args["pred_len"]))
                        self.assertTrue(log.is_absolute() and log.is_file())
                        self.assertIn("/logs/" + model + "/", str(log))

    def test_shared_script_model_controls_and_single_horizon(self):
        parser = self.scf_original_parser()
        for model in ("PULSE", "PULSE_SOFTS", "PULSE_SCF_SIMPLE", "PULSE_SCF"):
            with tempfile.TemporaryDirectory() as directory:
                result, commands = self.scf_run_script(
                    self.root / "scripts/PULSE_SCF/ETTh1.sh", directory,
                    MODEL_NAME=model, PRED_LENS="96",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(commands), 1)
                args = parser.parse_args(commands[0][2:])
                self.assertEqual(args.model, model)
                self.assertEqual(args.pred_len, 96)
                log_lines = [line[len("Log: "):] for line in result.stdout.splitlines()
                             if line.startswith("Log: ")]
                self.assertEqual(len(log_lines), 1)
                announced = Path(log_lines[0])
                self.assertTrue(announced.is_absolute())
                self.assertTrue(announced.is_file())
                expected = (Path(directory) / "logs" / model / "ETTh1" / "argument_check"
                            / (model + "_96_96.log"))
                self.assertEqual(announced.resolve(), expected.resolve())
                self.assertIn("Watch: tail -f ", result.stdout)

    def test_script_propagates_failure_and_prevents_log_overwrite(self):
        script = self.root / "scripts/PULSE_SCF/ETTh1.sh"
        with tempfile.TemporaryDirectory() as directory:
            result, commands = self.scf_run_script(
                script, directory, SCF_STUB_EXIT="37", PRED_LENS="96 192 336 720",
            )
            self.assertEqual(result.returncode, 37)
            self.assertEqual(len(commands), 1)
        with tempfile.TemporaryDirectory() as directory:
            result, commands = self.scf_run_script(script, directory)
            self.assertEqual(result.returncode, 0)
            repeated, repeated_commands = self.scf_run_script(script, directory)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(len(repeated_commands), len(commands))

    def test_scripts_reject_unknown_model_or_horizon(self):
        for selection in ({"MODEL_NAME": "unknown"}, {"PRED_LENS": "97"}):
            with tempfile.TemporaryDirectory() as directory:
                result, commands = self.scf_run_script(
                    self.root / "scripts/PULSE_SCF/Weather.sh", directory, **selection,
                )
                self.assertEqual(result.returncode, 2)
                self.assertEqual(commands, [])

    def test_pems_keeps_source_short_term_horizons(self):
        parser = self.scf_original_parser()
        script = self.root / "scripts/PULSE_SCF/PEMS03.sh"
        with tempfile.TemporaryDirectory() as directory:
            result, commands = self.scf_run_script(script, directory, PRED_LENS="12 48")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([parser.parse_args(c[2:]).pred_len for c in commands], [12, 48])
            logs = list((Path(directory) / "logs").rglob("*.log"))
            self.assertEqual({p.name for p in logs}, {"PULSE_SCF_96_12.log", "PULSE_SCF_96_48.log"})
        with tempfile.TemporaryDirectory() as directory:
            result, commands = self.scf_run_script(script, directory, PRED_LENS="192")
            self.assertEqual(result.returncode, 2)
            self.assertEqual(commands, [])


if __name__ == "__main__":
    unittest.main()
