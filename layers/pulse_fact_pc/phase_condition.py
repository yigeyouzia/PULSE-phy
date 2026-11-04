"""Historical-reference features and scale weights for PCSM.

PCSM: Phase-Conditioned Scale Mixing. No targets or future observations enter
these controllers. The model mixes encoded historical features with SAM.
"""

import torch
from torch import nn


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("{} must be a positive integer".format(name))


class PhaseReferenceEncoder(nn.Module):
    """Ordered [B,L,C] history -> shared per-variable encoder -> [B,context]."""

    def __init__(self, seq_len, context_dim=16):
        super(PhaseReferenceEncoder, self).__init__()
        _positive_integer("seq_len", seq_len)
        _positive_integer("context_dim", context_dim)
        self.seq_len = seq_len
        self.context_dim = context_dim
        self.temporal_projection = nn.Linear(seq_len, context_dim)
        self.activation = nn.GELU()

    def forward(self, phase_anchor):
        if phase_anchor.ndim != 3 or phase_anchor.shape[1] != self.seq_len or phase_anchor.shape[2] < 1:
            raise ValueError("PCSM phase anchor must have shape [B, seq_len, C]")
        features = self.temporal_projection(phase_anchor.transpose(1, 2))
        features = self.activation(features)  # [B,C,context_dim]
        return features.mean(dim=1)            # Pool variables, not history.


class PhaseScaleWeights(nn.Module):
    """Historical context -> nonnegative sample-wise weights summing to m."""

    def __init__(self, context_dim=16, hidden_dim=16, num_branches=2):
        super(PhaseScaleWeights, self).__init__()
        for name, value in (("context_dim", context_dim), ("hidden_dim", hidden_dim),
                            ("num_branches", num_branches)):
            _positive_integer(name, value)
        self.context_dim = context_dim
        self.num_branches = num_branches
        self.hidden = nn.Linear(context_dim, hidden_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, num_branches)
        # m*softmax(zeros) = ones: start from FACT's original branch sum.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, phase_context):
        if phase_context.ndim != 2 or phase_context.shape[1] != self.context_dim:
            raise ValueError("PCSM context must have shape [B, context_dim]")
        logits = self.output(self.activation(self.hidden(phase_context)))
        return self.num_branches * torch.softmax(logits, dim=-1)


class StaticScaleWeights(nn.Module):
    """Global learned branch proportions; no sample-dependent information."""

    def __init__(self, num_branches=2):
        super(StaticScaleWeights, self).__init__()
        _positive_integer("num_branches", num_branches)
        self.num_branches = num_branches
        self.logits = nn.Parameter(torch.zeros(num_branches))

    def forward(self, batch_size):
        _positive_integer("batch_size", batch_size)
        weights = self.num_branches * torch.softmax(self.logits, dim=-1)
        return weights.unsqueeze(0).expand(batch_size, -1)
