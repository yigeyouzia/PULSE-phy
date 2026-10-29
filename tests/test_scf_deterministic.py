"""Compatibility regression: python -B -m unittest discover -s tests -p test_scf_deterministic.py -v.

CUDA regression runs automatically when a CUDA device is available. Uses APIs
available in older PyTorch builds; the existing training entry is not modified.
"""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from layers.pulse_scf.deterministic import (
    scf_pad_channels_deterministic, scf_select_core_deterministic,
)
from models import PULSE_SOFTS, PULSE_SCF_SIMPLE, PULSE_SCF


class SCFDeterministicTests(unittest.TestCase):
    def setUp(self):
        self.previous_determinism = torch.are_deterministic_algorithms_enabled()
        self.previous_cudnn_determinism = torch.backends.cudnn.deterministic
        self.previous_benchmark = torch.backends.cudnn.benchmark
        self.environment = patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        self.environment.start()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(2024)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.previous_determinism)
        torch.backends.cudnn.deterministic = self.previous_cudnn_determinism
        torch.backends.cudnn.benchmark = self.previous_benchmark
        self.environment.stop()

    def test_core_forward_and_gradient_match_gather_reference(self):
        for channels in (1, 7, 21):
            source = torch.randn(2, channels, 8, dtype=torch.float64, requires_grad=True)
            adapted = source.detach().clone().requires_grad_(True)
            indices = torch.randint(channels, (2, 1, 8))
            expected = torch.gather(source, 1, indices)
            actual = scf_select_core_deterministic(adapted, indices)
            self.assertTrue(torch.equal(actual, expected))
            weights = torch.randn_like(expected)
            (expected * weights).sum().backward()
            (actual * weights).sum().backward()
            self.assertTrue(torch.equal(adapted.grad, source.grad))

    def test_padding_forward_and_gradient_match_index_select_reference(self):
        for channels in (1, 4, 7, 21):
            for target in (channels, channels + 2, 3 * channels + 1):
                source = torch.randn(2, 8, channels, dtype=torch.float64, requires_grad=True)
                adapted = source.detach().clone().requires_grad_(True)
                index = torch.arange(target - channels) % channels
                expected = torch.cat([source, source.index_select(-1, index)], dim=-1)
                actual = scf_pad_channels_deterministic(adapted, target)
                self.assertTrue(torch.equal(actual, expected))
                weights = torch.randn_like(expected)
                (expected * weights).sum().backward()
                (actual * weights).sum().backward()
                self.assertTrue(torch.allclose(adapted.grad, source.grad, rtol=1e-12, atol=1e-12))

    def test_compatibility_helpers_do_not_dispatch_scatter_or_index_add(self):
        features = torch.randn(2, 7, 8, requires_grad=True)
        indices = torch.randint(7, (2, 1, 8))
        with torch.autograd.profiler.profile() as trace:
            core = scf_select_core_deterministic(features, indices)
            padded = scf_pad_channels_deterministic(features.transpose(1, 2), 9)
            (core.square().sum() + padded.square().sum()).backward()
        operators = {event.name for event in trace.function_events}
        self.assertFalse(operators.intersection({
            "aten::gather", "aten::scatter_add_", "aten::index_select", "aten::index_add_",
        }), sorted(operators))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def scf_strict_training_step(self, device):
        config = SimpleNamespace(
            seq_len=24, pred_len=12, enc_in=7, d_model=8, inv_len=24,
            patch_size=6, time_dim=1, dsa=1, dsb=1, ksize=3,
            scf_d_model=8, scf_d_core=4, scf_d_ff=16,
            scf_e_layers=1, scf_dropout=0.0, scf_gate_hidden=8,
            use_amp=False, use_multi_gpu=False,
        )
        for module in (PULSE_SOFTS, PULSE_SCF_SIMPLE, PULSE_SCF):
            model = module.Model(config).to(device).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            batch = (
                torch.randn(2, 24, 7, device=device),
                torch.arange(2, device=device),
                torch.randn(2, 24, 1, device=device),
                torch.randn(2, 12, 1, device=device),
            )
            prediction, (lam, index) = model(*batch)
            target = torch.randn_like(prediction)
            target = lam * target + (1 - lam) * target[index]
            loss = (torch.fft.fft(prediction, dim=1)
                    - torch.fft.fft(target, dim=1)).abs().mean()
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, (module.__name__, name))
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            before = model.backbone.projection.weight.detach().clone()
            optimizer.step()
            self.assertFalse(torch.equal(before, model.backbone.projection.weight))
            self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_strict_deterministic_training_step_cpu(self):
        self.scf_strict_training_step("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; run this test on the server")
    def test_strict_deterministic_training_step_cuda(self):
        self.scf_strict_training_step("cuda")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
