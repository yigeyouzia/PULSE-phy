"""FACT variable-grid interaction between the original PULSE MLP projections."""

from types import SimpleNamespace

import torch
from torch import nn

from .vendor.fact_conv import Conv_block2d


class PULSEFACTBlock(nn.Module):
    """[B,C,D] -> FACT's [B,D,C] convention -> [B,C,D].

    Only real variables enter the grid. Conv_block2d owns its residual paths;
    this adapter returns the block output, without another residual update.
    """

    def __init__(self, enc_in, hidden_dim=512, d_ff=256, dilation=2,
                 num_kernels=2, dropout=0.1):
        super(PULSEFACTBlock, self).__init__()
        for name, value in (("enc_in", enc_in), ("hidden_dim", hidden_dim),
                            ("d_ff", d_ff), ("dilation", dilation),
                            ("num_kernels", num_kernels)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("PULSE_FACT {} must be a positive integer".format(name))
        if not 0 <= dropout < 1:
            raise ValueError("PULSE_FACT dropout must be in [0, 1)")

        self.enc_in = enc_in
        self.hidden_dim = hidden_dim
        # freq='n' prevents the source from reserving timestamp-token channels.
        source_config = SimpleNamespace(
            enc_in=enc_in, freq="n", num_kernels=num_kernels,
            d_ff=d_ff, dropout=dropout,
        )
        self.block = Conv_block2d(
            source_config, dilation=dilation, d_model=hidden_dim, d_ff=d_ff,
        )
        self.grid_channels = self.block.channel_patch * self.block.channel_patch2

    def forward(self, hidden):
        if hidden.ndim != 3 or hidden.shape[1:] != (self.enc_in, self.hidden_dim):
            raise ValueError("PULSE_FACT block expects [B, enc_in, hidden_dim]")

        x = hidden.transpose(1, 2)  # [B,D,C]
        padding = self.grid_channels - self.enc_in
        if padding:
            # A single prefix is insufficient for C=1/2/4. Cyclic repetition
            # extends the source padding without index_select/scatter backward.
            if padding <= self.enc_in:
                prefix = x[:, :, :padding]
            else:
                repeats = (padding + self.enc_in - 1) // self.enc_in
                prefix = x.repeat(1, 1, repeats)[:, :, :padding]
            x = torch.cat([x, prefix], dim=-1)

        # Already padded: the unmodified source adds a zero-length prefix,
        # builds its variable grid and applies its own convolutional residuals.
        output = self.block(x)
        return output[:, :, :self.enc_in].transpose(1, 2)
