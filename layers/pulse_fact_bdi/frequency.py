"""Complementary frequency bands on the observed historical time axis."""

import math

import torch
from torch import nn


class HistoricalResidualBandSplit(nn.Module):
    """[B,L,C] -> two real [B,L,C] residuals, without learned parameters.

    cutoff_ratio is relative to Nyquist, not a fraction of total FFT energy.
    No future targets, learned hidden-axis FFT, or physical-period assumption.
    """

    def __init__(self, seq_len, cutoff_ratio=0.25):
        super(HistoricalResidualBandSplit, self).__init__()
        if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 2:
            raise ValueError("BDI band split requires integer seq_len >= 2")
        if (isinstance(cutoff_ratio, bool) or not isinstance(cutoff_ratio, (int, float))
                or not math.isfinite(cutoff_ratio) or not 0 < cutoff_ratio < 1):
            raise ValueError("BDI cutoff_ratio must be finite and strictly between 0 and 1")
        self.seq_len = seq_len
        self.cutoff_ratio = float(cutoff_ratio)
        relative_frequency = 2.0 * torch.arange(seq_len // 2 + 1, dtype=torch.float64) / seq_len
        low_mask = (relative_frequency <= self.cutoff_ratio).float().view(1, -1, 1)
        self.register_buffer("low_mask", low_mask)

    def forward(self, residual):
        if residual.ndim != 3 or residual.shape[1] != self.seq_len:
            raise ValueError("BDI residual must have shape [B, seq_len, C]")
        if residual.dtype not in (torch.float32, torch.float64):
            raise ValueError("BDI historical FFT supports float32/float64; AMP is not supported")
        spectrum = torch.fft.rfft(residual, dim=1, norm="ortho")
        mask = self.low_mask.to(dtype=residual.dtype)
        low = torch.fft.irfft(spectrum * mask, n=self.seq_len, dim=1, norm="ortho")
        high = torch.fft.irfft(spectrum * (1.0 - mask), n=self.seq_len, dim=1, norm="ortho")
        return low, high
