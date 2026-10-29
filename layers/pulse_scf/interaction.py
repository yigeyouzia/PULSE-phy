"""Reference-conditioned shared-core/deviation interaction, in feature space.

Source STAR/FACT computations are retained in vendor/. Adaptations below expose
the STAR core, handle small channel grids, and return a FACT update exactly once.
"""

from dataclasses import dataclass
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .deterministic import scf_pad_channels_deterministic, scf_select_core_deterministic
from .vendor.fact_conv import Conv_block2d
from .vendor.softs_star import STAR


@dataclass(frozen=True)
class SCFHistoryState:
    anchor: torch.Tensor       # [B, L, C], learned historical reference
    residual: torch.Tensor     # [B, L, C], affine-normalized historical residual
    mean: torch.Tensor         # [B, 1, C], historical residual mean
    sigma: torch.Tensor        # [B, 1, C], historical residual standard deviation


class SCFSTAR(STAR):
    """STAR with access to the same core used for redistribution.

    Sampling/weighting and the returned update match the source STAR. There is
    one pooling operation per call; the deviation branch reuses that exact core.
    """

    def forward(self, input, *args, return_core=False, **kwargs):
        batch, channels, _ = input.shape
        features = self.gen2(F.gelu(self.gen1(input)))
        weights = F.softmax(features, dim=1)
        if self.training:
            probabilities = weights.permute(0, 2, 1).reshape(-1, channels)
            indices = torch.multinomial(probabilities, 1)
            indices = indices.view(batch, -1, 1).permute(0, 2, 1)
            core = scf_select_core_deterministic(features, indices)
        else:
            core = torch.sum(features * weights, dim=1, keepdim=True)
        combined = torch.cat([input, core.repeat(1, channels, 1)], dim=-1)
        update = self.gen4(F.gelu(self.gen3(combined)))
        return update, core if return_core else None


class SCFFACTInteraction(Conv_block2d):
    """Adapt FACT's grid block to [B,C,d], returning F(D)-D.

    No time-marker tokens or spectral branch are added. The source grid and
    parallel dilation semantics are preserved. Repeating the prefix cyclically
    also handles C=1/4, where the source's single slice cannot fill its grid.
    """

    def __init__(self, channels, settings):
        config = SimpleNamespace(
            enc_in=channels, freq="n", num_kernels=settings.num_kernels,
            d_ff=settings.d_ff, dropout=settings.dropout,
        )
        super().__init__(
            config, dilation=settings.fact_dilation, d_model=settings.d_model,
            d_ff=settings.d_ff,
        )
        self.scf_channels = channels

    def forward(self, deviation):
        if (deviation.ndim != 3 or deviation.shape[1] != self.scf_channels
                or deviation.shape[2] != self.d_model):
            raise ValueError("SCF FACT expects [B, enc_in, scf_d_model]")
        x = deviation.transpose(1, 2)
        batch, dim, channels = x.shape
        x = scf_pad_channels_deterministic(x, self.channel_patch * self.channel_patch2)
        grid = x.reshape(batch, dim, self.channel_patch, self.channel_patch2)
        updated = grid
        for conv in self.conv:
            updated = updated + conv(grid)
        updated = updated + self.conv_ffn(updated)
        updated = updated.reshape(batch, dim, -1)[:, :, :channels]
        return updated.transpose(1, 2) - deviation


class SCFStateGate(nn.Module):
    """Historical reference/residual state -> [B,C,1] correction strength."""

    def __init__(self, seq_len, settings):
        super().__init__()
        hidden = settings.gate_hidden
        self.anchor_encoder = nn.Linear(seq_len, hidden)
        self.residual_encoder = nn.Linear(seq_len, hidden)
        self.hidden = nn.Linear(2 * hidden + 2, hidden)
        self.output = nn.Linear(hidden, 1)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.constant_(
            self.output.bias, math.log(settings.gate_init / (1 - settings.gate_init))
        )

    def forward(self, state):
        features = torch.cat([
            F.gelu(self.anchor_encoder(state.anchor.transpose(1, 2))),
            F.gelu(self.residual_encoder(state.residual.transpose(1, 2))),
            state.mean.transpose(1, 2),
            state.sigma.clamp_min(1e-5).log().transpose(1, 2),
        ], dim=-1)
        return torch.sigmoid(self.output(F.gelu(self.hidden(features))))


class SCFInteraction(nn.Module):
    """STAR update plus either a simple or a state-conditioned FACT update.

    EncoderLayer owns the outer residual. This module returns G + gate*DeltaF,
    without adding Z itself. The full variant applies FACT to Z - projection(c).
    """

    def __init__(self, channels, seq_len, settings, variant):
        super().__init__()
        if variant not in {"simple", "full"}:
            raise ValueError("SCF interaction variant must be simple or full")
        self.variant = variant
        self.star = SCFSTAR(settings.d_model, settings.d_core)
        self.fact = SCFFACTInteraction(channels, settings)
        if variant == "full":
            self.core_projection = nn.Linear(settings.d_core, settings.d_model)
            self.gate = SCFStateGate(seq_len, settings)
        else:
            self.gate_logit = nn.Parameter(torch.tensor(
                math.log(settings.gate_init / (1 - settings.gate_init))
            ))

    def forward(self, input, *args, scf_state=None, **kwargs):
        shared_update, core = self.star(input, return_core=True)
        if self.variant == "full":
            if scf_state is None:
                raise ValueError("Full SCF interaction requires historical state")
            deviation = input - self.core_projection(core)
            strength = self.gate(scf_state)
        else:
            deviation = input
            strength = torch.sigmoid(self.gate_logit)
        return shared_update + strength * self.fact(deviation), None
