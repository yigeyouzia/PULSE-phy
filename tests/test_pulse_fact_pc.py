"""PCSM checks: python -B -m unittest discover -s tests -p test_pulse_fact_pc.py -v."""

import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from layers.pulse_fact.fact_block import PULSEFACTBlock
from layers.pulse_fact_pc.interaction import PhaseWeightedFACTBlock
from layers.pulse_fact_pc.phase_condition import PhaseReferenceEncoder, PhaseScaleWeights, StaticScaleWeights
from models import PULSE_FACT, PULSE_FACT_PC, PULSE_FACT_STATIC
import test_pulse_fact as original_tests


ROOT = Path(__file__).resolve().parents[1]
MODELS = (PULSE_FACT_PC, PULSE_FACT_STATIC)
DATASETS = ("PEMS04", "PEMS03", "ETTh1")


def config(**overrides):
    values = dict(seq_len=24, pred_len=12, enc_in=7, d_model=8, inv_len=24,
        patch_size=6, time_dim=1, dsa=1, dsb=1, ksize=3, use_amp=False,
        use_multi_gpu=False, use_gpu=False, label_len=0, model="PULSE_FACT_PC",
        data="ETTh1", features="M", output_attention=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def make(module, args=None):
    with redirect_stdout(io.StringIO()):
        return module.Model(args or config())


def batch(args, device="cpu"):
    return (torch.randn(2, args.seq_len, args.enc_in, device=device),
        torch.arange(2, device=device),
        torch.randn(2, args.seq_len, args.time_dim, device=device),
        torch.randn(2, args.pred_len, args.time_dim, device=device))


class PCSMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(2024)
        np.random.seed(2024)

    def test_weighted_block_ones_preserve_source_outputs_gradients_and_bn(self):
        for channels in (1, 7, 21):
            for training in (False, True):
                with self.subTest(channels=channels, training=training):
                    old = PULSEFACTBlock(channels, 8, 16, 2, 2, 0.1).double().train(training)
                    new = PhaseWeightedFACTBlock(channels, 8, 16, 2, 2, 0.1).double().train(training)
                    new.block.load_state_dict(old.block.state_dict())
                    x = torch.randn(2, channels, 8, dtype=torch.double, requires_grad=True)
                    other = x.detach().clone().requires_grad_(True)
                    weights = torch.ones(2, 2, dtype=torch.double, requires_grad=True)
                    torch.manual_seed(50)
                    expected = old(x)
                    torch.manual_seed(50)
                    actual = new(other, weights)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    expected.square().mean().backward()
                    actual.square().mean().backward()
                    torch.testing.assert_close(other.grad, x.grad, rtol=1e-12, atol=1e-12)
                    self.assertTrue(torch.isfinite(weights.grad).all())
                    for (name, source), (_, adapted) in zip(old.named_parameters(), new.named_parameters()):
                        torch.testing.assert_close(adapted.grad, source.grad, rtol=1e-12, atol=1e-12, msg=name)
                    for name, value in old.named_buffers():
                        torch.testing.assert_close(dict(new.named_buffers())[name], value, rtol=0, atol=0)

    def test_small_grids_and_nonuniform_weight_gradients(self):
        for channels in (1, 2, 4, 7, 9, 307):
            with self.subTest(channels=channels):
                block = PhaseWeightedFACTBlock(channels, 8, 16, 2, 2, 0).eval()
                x = torch.randn(2, channels, 8, requires_grad=True)
                weights = torch.tensor([[1.5, 0.5], [0.25, 1.75]], requires_grad=True)
                y = block(x, weights)
                self.assertEqual(y.shape, x.shape)
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(torch.isfinite(weights.grad).all())
                self.assertFalse(block.last_branch_weights.requires_grad)
                self.assertFalse(any("last_branch_weights" in key for key in block.state_dict()))

    def test_weights_preserve_sum_and_static_weights_ignore_sample(self):
        dynamic = PhaseScaleWeights(4, 8, 2)
        contexts = torch.randn(5, 4)
        torch.testing.assert_close(dynamic(contexts), torch.ones(5, 2), rtol=0, atol=0)
        with torch.no_grad():
            dynamic.output.weight.normal_()
        weights = dynamic(contexts)
        self.assertTrue((weights >= 0).all())
        torch.testing.assert_close(weights.sum(-1), torch.full((5,), 2.0))
        self.assertGreater((weights[0] - weights[1]).abs().max().item(), 0)
        static = StaticScaleWeights(2)
        with torch.no_grad():
            static.logits.copy_(torch.tensor([1.0, -1.0]))
        weights = static(5)
        torch.testing.assert_close(weights, weights[:1].expand(5, -1), rtol=0, atol=0)
        torch.testing.assert_close(weights.sum(-1), torch.full((5,), 2.0))

    def test_reference_encoder_preserves_time_order_before_variable_pool(self):
        encoder = PhaseReferenceEncoder(4, 1)
        with torch.no_grad():
            encoder.temporal_projection.weight.zero_()
            encoder.temporal_projection.weight[0, 0] = 1
            encoder.temporal_projection.bias.zero_()
        anchor = torch.tensor([1., 0., -1., 0.]).view(1, 4, 1).expand(1, 4, 3)
        shifted = anchor.roll(1, dims=1)
        self.assertEqual(anchor.mean().item(), shifted.mean().item())
        self.assertNotEqual(encoder(anchor).item(), encoder(shifted).item())
        arbitrary = torch.randn(2, 4, 3)
        torch.testing.assert_close(encoder(arbitrary), encoder(arbitrary[:, :, [2, 0, 1]]))

    def test_new_models_initially_match_pulse_fact_with_and_without_sam(self):
        args = config()
        for module in MODELS:
            for training, mixing in ((False, True), (True, False), (True, True)):
                with self.subTest(module=module.__name__, training=training, mixing=mixing):
                    torch.manual_seed(2024)
                    source = make(PULSE_FACT, args).train(training)
                    torch.manual_seed(2024)
                    model = make(module, args).train(training)
                    state = model.state_dict()
                    for name, value in source.state_dict().items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(state[name], value, rtol=0, atol=0)
                    inputs = batch(args)
                    np.random.seed(17)
                    torch.manual_seed(41)
                    expected = source(*inputs, do_mixup=mixing)
                    np.random.seed(17)
                    torch.manual_seed(41)
                    actual = model(*inputs, do_mixup=mixing)
                    if training:
                        if mixing:
                            self.assertEqual(expected[1][0], actual[1][0])
                            torch.testing.assert_close(expected[1][1], actual[1][1])
                        else:
                            self.assertIsNone(actual[1])
                        expected, actual = expected[0], actual[0]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if training:
                        expected.square().mean().backward()
                        actual.square().mean().backward()
                        for name, parameter in source.named_parameters():
                            torch.testing.assert_close(dict(model.named_parameters())[name].grad,
                                parameter.grad, rtol=2e-5, atol=1e-6, msg=name)

    def test_extra_controller_initialization_preserves_original_cpu_rng_state(self):
        for module in MODELS:
            torch.manual_seed(2024)
            make(PULSE_FACT)
            expected = torch.random.get_rng_state()
            torch.manual_seed(2024)
            make(module)
            torch.testing.assert_close(torch.random.get_rng_state(), expected, rtol=0, atol=0)

    def test_condition_features_follow_the_same_sam_lambda_and_permutation(self):
        args = config()
        model = make(PULSE_FACT_PC, args).train()
        captured = {}
        def encoded(module, inputs, output):
            captured["encoded"] = output.detach().clone()
        def conditioned(module, inputs):
            captured["mixed"] = inputs[0].detach().clone()
        hooks = [model.phase_encoder.register_forward_hook(encoded),
                 model.phase_weights.register_forward_pre_hook(conditioned)]
        permutation = torch.tensor([1, 0])
        try:
            with patch("models.PULSE_FACT_PC.np.random.beta", return_value=0.3), \
                    patch("models.PULSE_FACT_PC.torch.randperm", return_value=permutation):
                _, (lam, index) = model(*batch(args))
            expected = lam * captured["encoded"] + (1 - lam) * captured["encoded"][index]
            torch.testing.assert_close(captured["mixed"], expected, rtol=0, atol=0)
        finally:
            for hook in hooks:
                hook.remove()

    def test_phase_condition_has_no_observed_residual_or_future_mark_input(self):
        model = make(PULSE_FACT_PC).eval()
        with torch.no_grad():
            model.phase_weights.output.weight.normal_(std=0.1)
            x, cycle, xm, ym = batch(config())
            model(x, cycle, xm, ym)
            expected = model.fact_block.last_branch_weights.clone()
            model(x + 10 * torch.randn_like(x), cycle, xm, ym + 10 * torch.randn_like(ym))
            torch.testing.assert_close(model.fact_block.last_branch_weights, expected, rtol=0, atol=0)

    def strict_step(self, module, device):
        old = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
               torch.backends.cudnn.benchmark)
        try:
            with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                args = config()
                model = make(module, args).to(device).train()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                controller = model.phase_weights.output if module is PULSE_FACT_PC else model.static_weights
                before = next(controller.parameters()).detach().clone()
                for _ in range(2):
                    optimizer.zero_grad()
                    prediction, (lam, index) = model(*batch(args, device))
                    target = torch.randn_like(prediction)
                    target = lam * target + (1 - lam) * target[index]
                    loss = (torch.fft.fft(prediction, dim=1) - torch.fft.fft(target, dim=1)).abs().mean()
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    if module is PULSE_FACT_PC and _ == 1:
                        self.assertGreater(model.phase_encoder.temporal_projection.weight.grad.abs().sum().item(), 0)
                    optimizer.step()
                self.assertFalse(torch.equal(before, next(controller.parameters())))
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(old[0])
            torch.backends.cudnn.deterministic = old[1]
            torch.backends.cudnn.benchmark = old[2]

    def test_strict_fft_backward_and_controller_optimizer_updates_cpu(self):
        for module in MODELS:
            with self.subTest(module=module.__name__):
                self.strict_step(module, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; run this check on the training server")
    def test_strict_fft_backward_and_controller_optimizer_updates_cuda(self):
        for module in MODELS:
            self.strict_step(module, "cuda")

    def test_checkpoint_roundtrip_and_configuration_model_guard(self):
        for module in MODELS:
            model = make(module).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model(*batch(config()))[0].square().mean().backward()
            optimizer.step()
            model.eval()
            inputs = batch(config())
            expected = model(*inputs)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            restored = make(module).eval()
            state = torch.load(buffer, map_location="cpu")
            restored.load_state_dict(state)
            torch.testing.assert_close(restored(*inputs), expected, rtol=0, atol=0)
            prefix = "pulse_fact_pc" if module is PULSE_FACT_PC else "pulse_fact_static"
            different = make(module, config(**{prefix + "_dropout": 0.2}))
            with self.assertRaisesRegex(RuntimeError, "configuration differs"):
                different.load_state_dict(state)
            other = make(PULSE_FACT_STATIC if module is PULSE_FACT_PC else PULSE_FACT_PC)
            with self.assertRaisesRegex(RuntimeError, "configuration differs"):
                other.load_state_dict(state)

    def test_settings_do_not_mutate_arguments_or_borrow_legacy_parameter_meanings(self):
        for module in MODELS:
            args = config(d_ff=2048, dropout=0, e_layers=9, n_heads=8, use_revin=0)
            before, output = vars(args).copy(), io.StringIO()
            with redirect_stdout(output):
                model = module.Model(args)
            self.assertEqual(vars(args), before)
            logged = json.loads(output.getvalue().split("effective settings: ")[1])
            self.assertEqual(logged, model.get_extra_state())
            self.assertEqual(model.fact_hidden_dim, 512)
            self.assertEqual(model.fact_d_ff, 256)
            self.assertEqual(model.fact_dropout, 0.1)
            self.assertEqual(logged["settings"]["depth"], 1)
            self.assertEqual(model.d_model, 8)

    def test_validation_errors_are_explicit(self):
        for module, prefix in ((PULSE_FACT_PC, "pulse_fact_pc"), (PULSE_FACT_STATIC, "pulse_fact_static")):
            for bad in ({prefix + "_d_ff": 0}, {prefix + "_dilation": 0},
                        {prefix + "_num_kernels": 0}, {prefix + "_dropout": 1},
                        {"patch_size": 7}, {"dsa": 8}, {"label_len": 1},
                        {"use_amp": True}, {"use_multi_gpu": True}):
                with self.subTest(module=module.__name__, bad=bad), self.assertRaises(ValueError):
                    make(module, config(**bad))
        for bad in ({"pulse_fact_pc_context_dim": 0}, {"pulse_fact_pc_controller_hidden": 0}):
            with self.assertRaises(ValueError):
                make(PULSE_FACT_PC, config(**bad))
        with self.assertRaises(ValueError):
            PhaseReferenceEncoder(4)(torch.randn(2, 5, 7))
        with self.assertRaises(ValueError):
            PhaseWeightedFACTBlock(7, 8)(torch.randn(2, 7, 8), torch.ones(2, 3))

    def test_original_experiment_registration_and_validation(self):
        from exp.exp_main import Exp_Main
        for module in MODELS:
            args = config(model=module.__name__.split(".")[-1])
            experiment = Exp_Main.__new__(Exp_Main)
            experiment.args, experiment.device = args, torch.device("cpu")
            with redirect_stdout(io.StringIO()):
                experiment.model = experiment._build_model()
            self.assertIsInstance(experiment.model, module.Model)
            x, cycle, xm, ym = batch(args)
            target = torch.randn(2, args.pred_len, args.enc_in)
            dataset = torch.utils.data.TensorDataset(x, target, xm, ym, cycle)
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            experiment.model.eval()
            with torch.no_grad():
                expected = (experiment.model(x, cycle, xm, ym) - target).square().mean().item()
            self.assertAlmostEqual(experiment.vali(dataset, loader, nn.MSELoss()), expected, places=6)

    def test_main_models_expand_source_methods_and_protected_source_hashes(self):
        def methods(path):
            tree = ast.parse(path.read_text())
            result = {}
            for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
                for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
                    result[(cls.name, fn.name)] = ast.dump(fn, include_attributes=False)
            return tree, result
        _, expected = methods(ROOT / "models/PULSE_FACT.py")
        for module in MODELS:
            tree, actual = methods(Path(module.__file__))
            model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model")
            self.assertEqual(ast.dump(model.bases[0]), "Attribute(value=Name(id='nn', ctx=Load()), attr='Module', ctx=Load())")
            for key in [("ReVIN", "__init__"), ("ReVIN", "forward"), ("ReVIN", "reverse"),
                        ("Model", "norm"), ("Model", "phase_router"), ("Model", "denorm")]:
                self.assertEqual(actual[key], expected[key], key)
        manifest = json.loads((ROOT / "layers/pulse_fact_pc/sources.json").read_text())
        for name, digest in manifest["protected_sources"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)


class PCSMScriptTests(unittest.TestCase):
    def test_six_single_horizon_scripts_match_source_configuration_and_outputs(self):
        helper = original_tests.PULSEFACTScriptTests()
        parser = helper.fact_parser()
        for module in MODELS:
            name = module.__name__.split(".")[-1]
            for dataset in DATASETS:
                script = ROOT / "scripts" / name / (dataset + ".sh")
                subprocess.run(["bash", "-n", str(script)], check=True)
                source_text = script.read_text()
                self.assertIn("set -e", source_text)
                self.assertNotIn("function ", source_text)
                self.assertEqual(source_text.count("python -u run.py"), 1)
                with tempfile.TemporaryDirectory() as before, tempfile.TemporaryDirectory() as after:
                    result, source = helper.fact_capture(ROOT / "scripts/PULSE_FACT" / (dataset + ".sh"), before)
                    self.assertEqual(result.returncode, 0)
                    original = next(parser.parse_args(c[2:]) for c in source if parser.parse_args(c[2:]).pred_len == 96)
                    result, commands = helper.fact_capture(script, after)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(len(commands), 1)
                    current = parser.parse_args(commands[0][2:])
                    for key in vars(original).keys() - {"model", "model_id", "checkpoints"}:
                        self.assertEqual(getattr(current, key), getattr(original, key), (name, dataset, key))
                    self.assertEqual(current.model, name)
                    self.assertEqual(current.checkpoints, "./checkpoints/{}/".format(name))
                    log = Path(after) / "logs" / name / dataset / (name + "_96_96.log")
                    reported = [Path(line[5:]) for line in result.stdout.splitlines() if line.startswith("Log: ")]
                    self.assertEqual([p.resolve() for p in reported], [log.resolve()])
                    self.assertEqual(log.read_text(), "stdout\nstderr\n")
                    args = current
                    args.use_gpu = False
                    torch.set_num_threads(2)
                    model = make(module, args).eval()
                    with torch.no_grad():
                        prediction = model(torch.randn(1, 96, args.enc_in), torch.zeros(1, dtype=torch.long),
                            torch.randn(1, 96, args.time_dim), torch.randn(1, 96, args.time_dim))
                    self.assertEqual(prediction.shape, (1, 96, args.enc_in))

    def test_failure_exit_status_and_fixed_log_distinct_experiment_ids(self):
        helper = original_tests.PULSEFACTScriptTests()
        parser = helper.fact_parser()
        for module in MODELS:
            name = module.__name__.split(".")[-1]
            script = ROOT / "scripts" / name / "ETTh1.sh"
            with tempfile.TemporaryDirectory() as directory:
                failed, captured = helper.fact_capture(script, directory, exit_code=37)
                self.assertEqual(failed.returncode, 37)
                self.assertEqual(len(captured), 1)
            with tempfile.TemporaryDirectory() as directory:
                first, one = helper.fact_capture(script, directory)
                second, two = helper.fact_capture(script, directory, message="second")
                self.assertEqual((first.returncode, second.returncode), (0, 0))
                self.assertNotEqual(parser.parse_args(one[0][2:]).model_id, parser.parse_args(two[1][2:]).model_id)
                logs = list((Path(directory) / "logs").rglob("*.log"))
                self.assertEqual(len(logs), 1)
                self.assertEqual(logs[0].read_text(), "second\nstderr\n")


if __name__ == "__main__":
    unittest.main()
