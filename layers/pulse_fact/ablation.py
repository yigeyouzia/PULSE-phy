"""Shared per-sensor control and the statistic-aware mixup used by the PEMS ablations."""

import numpy as np
import torch
from torch import nn


class SensorResidualMLP(nn.Module):
    """Apply shared weights to each [512]-feature sensor vector independently."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(512, 271),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(271, 512),
        )

    def forward(self, hidden):
        return hidden + self.features(hidden)


def validate_forecast_config(configs):
    if configs.seq_len < 2 or configs.pred_len < 1 or configs.enc_in < 1:
        raise ValueError("Ablations require at least two history steps and positive horizon and sensor count")
    if getattr(configs, "label_len", 0) != 0:
        raise ValueError("Ablations use the PULSE forecasting protocol with label_len=0")
    if getattr(configs, "use_amp", False) or getattr(configs, "use_multi_gpu", False):
        raise ValueError("Ablations use full precision on one device")


def mix_input_and_stats(x, mean, var, eps):
    """Match PULSE_FACT.forward, including the permutation used by the trainer for Y."""
    weight = np.random.beta(0.15, 0.15)
    order = torch.randperm(x.size(0)).to(x.device)
    sigma = torch.sqrt(var + eps)
    mixed_sigma = weight * sigma + (1.0 - weight) * sigma[order]
    mixed_input = weight * x + (1.0 - weight) * x[order]
    mixed_mean = weight * mean + (1.0 - weight) * mean[order]
    return mixed_input, mixed_mean, mixed_sigma.square() - eps, (weight, order)
