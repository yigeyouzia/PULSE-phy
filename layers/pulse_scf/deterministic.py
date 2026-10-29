"""Equivalent SCF indexing operations without scatter/index-add backward.

Older CUDA builds can reject gather/index_select gradients when the existing
entry enables strict deterministic algorithms. Keep that setting and replace
only the SCF selection/padding implementations; sampling and parameters stay
unchanged.
"""

import torch


def scf_select_core_deterministic(features, indices):
    """Select one channel per core coordinate: [B,C,d] -> [B,1,d].

    ``indices`` is the valid [B,1,d] LongTensor produced by STAR's unchanged
    multinomial sampling. A boolean mask preserves hard selection and its
    gradient without gather's scatter_add backward.
    """
    if features.ndim != 3 or indices.shape != (features.shape[0], 1, features.shape[2]):
        raise ValueError("SCF core selection expects features [B,C,d], indices [B,1,d]")
    channels = torch.arange(features.shape[1], device=features.device).view(1, -1, 1)
    selected = channels == indices
    return torch.where(selected, features, features.new_zeros(())).sum(1, keepdim=True)


def scf_pad_channels_deterministic(x, target_channels):
    """Cyclic-prefix padding of [B,d,C] without index_select's index_add gradient."""
    channels = x.shape[-1]
    if x.ndim != 3 or channels < 1 or target_channels < channels:
        raise ValueError("SCF padding expects [B,d,C] and target_channels >= C > 0")
    padding = target_channels - channels
    if padding == 0:
        return x
    if padding <= channels:
        prefix = x[:, :, :padding]
    else:
        repeats = (padding + channels - 1) // channels
        prefix = x.repeat(1, 1, repeats)[:, :, :padding]
    return torch.cat([x, prefix], dim=-1)
