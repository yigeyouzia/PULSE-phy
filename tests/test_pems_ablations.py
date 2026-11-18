"""Mechanism checks for the PEMS A00/A10/A01 experiment."""

import copy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from models import PULSE_FACT, PULSE_FACT_A00, PULSE_FACT_A01, PULSE_FACT_A10


class PEMSAblationTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(seq_len=24, pred_len=12, enc_in=7, time_dim=2,
                                      inv_len=24, patch_size=4, d_model=8, ksize=3, dsa=2, dsb=1)
        torch.manual_seed(24)
        self.inputs = (torch.randn(4, 24, 7), torch.arange(4),
                       torch.randn(4, 24, 2), torch.randn(4, 12, 2))

    def test_a10_matches_complete_anchor_path_with_same_per_sensor_mlp(self):
        torch.manual_seed(2024)
        reference = PULSE_FACT.Model(self.config)
        torch.manual_seed(2024)
        model = PULSE_FACT_A10.Model(self.config)
        reference.fact_block = copy.deepcopy(model.sensor_mlp)

        for training, mixing in ((False, False), (True, False), (True, True)):
            with self.subTest(training=training, mixing=mixing):
                def predict(module):
                    np.random.seed(11)
                    torch.manual_seed(11)
                    return module.train(training)(*self.inputs, do_mixup=mixing)

                torch.testing.assert_close(predict(model), predict(reference), rtol=0, atol=0)

    def test_anchor_free_paths_use_raw_window_statistics_and_ignore_time_inputs(self):
        x, cycle, past_marks, future_marks = self.inputs
        for variant in (PULSE_FACT_A00, PULSE_FACT_A01):
            with self.subTest(variant=variant.__name__):
                model = variant.Model(self.config).eval()
                mean = x.mean(dim=1, keepdim=True)
                sigma = (x.var(dim=1, keepdim=True) + model.revin.eps).sqrt()
                normalized = ((x - mean) / sigma).transpose(1, 2)
                block = model.sensor_mlp if variant is PULSE_FACT_A00 else model.fact_block
                hidden = model.backbone[1](model.backbone[0](normalized))
                expected = model.backbone[2](block(hidden)).transpose(1, 2) * sigma + mean
                with torch.no_grad():
                    actual = model(x, cycle + 7, past_marks + 3, future_marks - 2)
                    torch.testing.assert_close(actual, expected)
                self.assertFalse(hasattr(model, "codebook"))

    def test_capacity_sensor_independence_and_trainable_gradients(self):
        for variant in (PULSE_FACT_A00, PULSE_FACT_A10, PULSE_FACT_A01):
            with self.subTest(variant=variant.__name__):
                model = variant.Model(self.config)
                x, cycle, xm, ym = self.inputs
                if variant is PULSE_FACT_A01:
                    self.assertEqual(sum(p.numel() for p in model.fact_block.parameters()), 278272)
                else:
                    self.assertEqual(sum(p.numel() for p in model.sensor_mlp.parameters()), 278287)
                    with torch.no_grad():
                        model.eval()
                        forecast = model(x, cycle, xm, ym)
                        altered = x.clone()
                        altered[:, :, 0] *= -3
                        torch.testing.assert_close(model(altered, cycle, xm, ym)[:, :, 1:],
                                                   forecast[:, :, 1:], rtol=0, atol=0)
                forecast, (weight, order) = model.train()(*self.inputs)
                target = torch.randn_like(forecast)
                mixed_target = weight * target + (1 - weight) * target[order]
                (torch.fft.fft(forecast - mixed_target, dim=1).abs().mean()).backward()
                for name, param in model.named_parameters():
                    self.assertIsNotNone(param.grad, name)
                    self.assertTrue(torch.isfinite(param.grad).all(), name)
