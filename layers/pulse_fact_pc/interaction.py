"""FACT variable-grid adapter with explicit PCSM branch weights.

Source layout and cyclic padding: layers/pulse_fact/fact_block.py.
Unmodified FACT operators: layers/pulse_fact/vendor/fact_conv.py.
The new computation weights each parallel dilation update before the original
FFN/residual. Original layers and their forward methods are never modified.
"""

from types import SimpleNamespace

import torch
from torch import nn

from layers.pulse_fact.vendor.fact_conv import Conv_block2d
from .phase_condition import _positive_integer


class PhaseWeightedFACTBlock(nn.Module):
    """[B,C,D] + [B,m] weights -> [B,C,D], with one source block."""

    def __init__(self, enc_in, hidden_dim=512, d_ff=256, dilation=2,
                 num_kernels=2, dropout=0.1):
        super(PhaseWeightedFACTBlock, self).__init__()
        for name, value in (("enc_in", enc_in), ("hidden_dim", hidden_dim),
                            ("d_ff", d_ff), ("dilation", dilation),
                            ("num_kernels", num_kernels)):
            _positive_integer(name, value)
        if not 0 <= dropout < 1:
            raise ValueError("PCSM dropout must be in [0, 1)")
        self.enc_in = enc_in
        self.hidden_dim = hidden_dim
        self.num_branches = dilation
        source_config = SimpleNamespace(
            enc_in=enc_in, freq="n", num_kernels=num_kernels,
            d_ff=d_ff, dropout=dropout,
        )
        # Preserve constructor order and parameter names for seeded comparisons.
        self.block = Conv_block2d(
            source_config, dilation=dilation, d_model=hidden_dim, d_ff=d_ff,
        )
        self.grid_channels = self.block.channel_patch * self.block.channel_patch2
        # Analysis-only detached cache; not a parameter or checkpoint buffer.
        self.last_branch_weights = None

    def forward(self, hidden, branch_weights):
        if hidden.ndim != 3 or hidden.shape[1:] != (self.enc_in, self.hidden_dim):
            raise ValueError("PCSM FACT expects [B, enc_in, hidden_dim]")
        if branch_weights.shape != (hidden.shape[0], self.num_branches):
            raise ValueError("PCSM weights must have shape [B, num_branches]")
        if branch_weights.device != hidden.device or branch_weights.dtype != hidden.dtype:
            raise ValueError("PCSM weights must match hidden dtype and device")

        self.last_branch_weights = branch_weights.detach()
        x = hidden.transpose(1, 2)  # [B,D,C]
        padding = self.grid_channels - self.enc_in
        if padding:
            if padding <= self.enc_in:
                prefix = x[:, :, :padding]
            else:
                repeats = (padding + self.enc_in - 1) // self.enc_in
                prefix = x.repeat(1, 1, repeats)[:, :, :padding]
            x = torch.cat([x, prefix], dim=-1)
        grid = x.reshape(hidden.shape[0], self.hidden_dim,
                         self.block.channel_patch, self.block.channel_patch2)

        updated = grid
        for index, branch in enumerate(self.block.conv):
            weight = branch_weights[:, index].reshape(-1, 1, 1, 1)
            updated = updated + weight * branch(grid)
        updated = updated + self.block.conv_ffn(updated)
        output = updated.reshape(hidden.shape[0], self.hidden_dim, self.grid_channels)
        return output[:, :, :self.enc_in].transpose(1, 2)
