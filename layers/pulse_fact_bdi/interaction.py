"""FACT components with separate residual-band inputs and one main residual.

Adapted from layers/pulse_fact/fact_block.py and the unmodified FACT selections
in layers/pulse_fact/vendor/fact_conv.py. See notes/data/pulse_fact_bdi_sources.json
and layers/pulse_fact/vendor/FACT_LICENSE. No source class is modified.
"""

from types import SimpleNamespace

import torch
from torch import nn

from layers.pulse_fact.vendor.fact_conv import Conv_block2d


class PULSEFACTBDIBlock(nn.Module):
    """Three [B,C,D] inputs -> one [B,C,D] output.

    The original dilation=1 branch sees high-band features; dilation=2 sees
    low-band features. Both update the reference-retaining main hidden path.
    Passing the same tensor three times recovers the existing PF block.
    """

    def __init__(self, enc_in, hidden_dim=512, d_ff=256, num_kernels=2, dropout=0.1):
        super(PULSEFACTBDIBlock, self).__init__()
        for name, value in (("enc_in", enc_in), ("hidden_dim", hidden_dim),
                            ("d_ff", d_ff), ("num_kernels", num_kernels)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("BDI {} must be a positive integer".format(name))
        if not 0 <= dropout < 1:
            raise ValueError("BDI dropout must be in [0, 1)")
        self.enc_in = enc_in
        self.hidden_dim = hidden_dim
        source_config = SimpleNamespace(enc_in=enc_in, freq="n", num_kernels=num_kernels,
                                        d_ff=d_ff, dropout=dropout)
        # Same construction/parameter ordering as the two-branch PF source.
        self.block = Conv_block2d(source_config, dilation=2, d_model=hidden_dim, d_ff=d_ff)
        self.grid_side = self.block.channel_patch
        self.grid_channels = self.grid_side * self.block.channel_patch2

    def variable_grid(self, hidden):
        if hidden.ndim != 3 or hidden.shape[1:] != (self.enc_in, self.hidden_dim):
            raise ValueError("BDI hidden inputs must have shape [B, enc_in, hidden_dim]")
        x = hidden.transpose(1, 2)  # [B,D,C]
        padding = self.grid_channels - self.enc_in
        if padding:
            if padding <= self.enc_in:
                prefix = x[:, :, :padding]
            else:
                repeats = (padding + self.enc_in - 1) // self.enc_in
                prefix = x.repeat(1, 1, repeats)[:, :, :padding]
            x = torch.cat([x, prefix], dim=-1)
        return x.reshape(hidden.shape[0], self.hidden_dim, self.grid_side, self.grid_side)

    def forward(self, hidden_main, hidden_low, hidden_high):
        if hidden_main.shape != hidden_low.shape or hidden_main.shape != hidden_high.shape:
            raise ValueError("BDI main/low/high inputs must have matching shapes")
        main_grid = self.variable_grid(hidden_main)
        low_grid = self.variable_grid(hidden_low)
        high_grid = self.variable_grid(hidden_high)

        # Keep source order (dilation=1 then 2) and one main residual only.
        high_update = self.block.conv[0](high_grid)
        low_update = self.block.conv[1](low_grid)
        mixed = main_grid + high_update + low_update
        output = mixed + self.block.conv_ffn(mixed)
        output = output.reshape(hidden_main.shape[0], self.hidden_dim, -1)
        return output[:, :, :self.enc_in].transpose(1, 2)
