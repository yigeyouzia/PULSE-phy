"""History-fitted harmonic extrapolation, a soft bridge frequency prior.

The basis axis is actual sampled time, never the PULSE/FACT hidden feature
axis. Published frequencies are approximate priors, not exact window labels.
"""

import math

import torch
from torch import nn

from .units import MODAL_FREQUENCIES_HZ, SAMPLE_INTERVAL_SECONDS


class GjeModalFrequencyPrior(nn.Module):
    def __init__(self, seq_len, pred_len, frequencies_hz=MODAL_FREQUENCIES_HZ,
                 ridge=1e-3, gain_limit=0.25, initial_gain=0.05):
        super().__init__()
        frequencies = torch.as_tensor(frequencies_hz, dtype=torch.float64)
        if frequencies.ndim != 1 or frequencies.numel() < 1 or not torch.isfinite(frequencies).all() or not ((frequencies > 0) & (frequencies < 1)).all():
            raise ValueError('Modal frequencies must be finite and strictly below the 1 Hz Nyquist limit')
        if seq_len < 2 or pred_len < 1 or not all(math.isfinite(v) for v in (ridge, gain_limit, initial_gain)) or ridge <= 0 or not (0 <= initial_gain < gain_limit):
            raise ValueError('Invalid modal prior length, ridge or gain bounds')
        self.gain_limit = float(gain_limit)
        self.ridge = float(ridge)
        self.register_buffer('frequencies_hz', frequencies.float())
        past_time = torch.arange(1 - seq_len, 1, dtype=torch.float64) * SAMPLE_INTERVAL_SECONDS
        future_time = torch.arange(1, pred_len + 1, dtype=torch.float64) * SAMPLE_INTERVAL_SECONDS
        past = self.basis(past_time, frequencies)
        future = self.basis(future_time, frequencies)
        # Center on the historical window; fitted/extrapolated oscillations do
        # not introduce a free DC degree of freedom into the modal branch.
        basis_mean = past.mean(dim=0, keepdim=True)
        past = past - basis_mean
        future = future - basis_mean
        gram = past.T @ past + ridge * seq_len * torch.eye(past.shape[1], dtype=past.dtype)
        projection = torch.linalg.solve(gram, past.T)  # [2K,L], computed once.
        self.register_buffer('history_projection', projection.float())
        self.register_buffer('future_basis', future.float())
        self.raw_gain = nn.Parameter(torch.full((1, 1, 6), math.atanh(initial_gain / gain_limit)))

    @staticmethod
    def basis(times, frequencies):
        phase = 2 * math.pi * times[:, None] * frequencies[None, :]
        return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)

    def forward(self, history_si):
        if history_si.ndim != 3 or history_si.shape[1] != self.history_projection.shape[1] or history_si.shape[2] != 6:
            raise ValueError('Modal prior expects historical SI acceleration [B,L,6]')
        centered = history_si - history_si.mean(dim=1, keepdim=True)
        coefficients = torch.matmul(self.history_projection, centered)  # [B,2K,6]
        extrapolation = torch.matmul(self.future_basis, coefficients)  # [B,H,6]
        return self.gain_limit * torch.tanh(self.raw_gain) * extrapolation
