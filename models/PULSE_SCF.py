"""PULSE with a shared-core/FACT backbone and historical state conditioning.

Reuses PULSE's norm, Router and denorm without editing the baseline. The public
forward contract matches PULSE's non-AMP, single-device training/evaluation path.
"""

import numpy as np
import torch

from models.PULSE import Model as PULSEModel
from layers.pulse_scf.backbone import SCFBackbone
from layers.pulse_scf.config import SCFConfig
from layers.pulse_scf.interaction import SCFHistoryState


class PULSESCFModel(PULSEModel):
    scf_variant = "full"

    def __init__(self, configs):
        settings = SCFConfig.from_configs(configs)
        if getattr(configs, "use_amp", False) or getattr(configs, "use_multi_gpu", False):
            raise ValueError("SCF currently supports the non-AMP, single-device SAM path")
        if (configs.seq_len < 2 or configs.pred_len < 1 or configs.enc_in < 1
                or configs.patch_size < 1 or configs.inv_len < 1):
            raise ValueError("SCF requires valid positive lengths and seq_len >= 2")
        if configs.seq_len % configs.patch_size or configs.pred_len % configs.patch_size:
            raise ValueError("PULSE patch_size must divide seq_len and pred_len")
        if (configs.dsa < 1 or configs.dsb < 1
                or min(configs.enc_in // configs.dsa, configs.enc_in // configs.dsb) < 1):
            raise ValueError("PULSE dsa/dsb would create an empty TimeEncoder dimension")
        if configs.d_model < 1 or configs.time_dim < 1 or configs.ksize < 1:
            raise ValueError("PULSE Router, timestamp and kernel dimensions must be positive")
        super().__init__(configs)
        self.scf_settings = settings
        self.backbone = SCFBackbone(
            self.seq_len, self.pred_len, self.enc_in, settings, self.scf_variant
        )

    def get_extra_state(self):
        # Persist effective settings even while run.py has no scf_* CLI flags.
        return {"scf_schema_version": 1, "variant": self.scf_variant,
                "settings": self.scf_settings.as_dict()}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("SCF checkpoint configuration differs from this model")

    def forward(self, x, cycle_index, batch_x_mark, batch_y_mark, do_mixup=True):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("SCF input must have shape [B, seq_len, enc_in]")
        if batch_x_mark.shape != (x.shape[0], self.seq_len, self.time_dim):
            raise ValueError("SCF historical timestamps have an unexpected shape")
        if batch_y_mark.shape != (x.shape[0], self.pred_len, self.time_dim):
            raise ValueError("SCF future timestamps require pred_len steps (label_len=0)")

        timestamp_x = self.time_enc_x(batch_x_mark)
        timestamp_y = self.time_enc_y(batch_y_mark)
        x_in, anchor, mean, var = self.norm(x, cycle_index, timestamp_x)
        residual = x_in - anchor
        sigma = torch.sqrt(var + self.revin.eps)
        mixup_params = None

        if self.training and do_mixup:
            lam = np.random.beta(0.15, 0.15)
            index = torch.randperm(x.shape[0], device=x.device)
            x_mixed = lam * x_in + (1 - lam) * x_in[index]
            mean_mixed = lam * mean + (1 - lam) * mean[index]
            sigma_mixed = lam * sigma + (1 - lam) * sigma[index]
            var_mixed = sigma_mixed.square() - self.revin.eps
            # The new condition describes the mixed backbone input. Preserve
            # the source Router's original (unmixed) anchor/time convention.
            condition_anchor = lam * anchor + (1 - lam) * anchor[index]
            condition_residual = lam * residual + (1 - lam) * residual[index]
            mixup_params = (lam, index)
        else:
            x_mixed, mean_mixed, sigma_mixed, var_mixed = x_in, mean, sigma, var
            condition_anchor, condition_residual = anchor, residual

        state = SCFHistoryState(
            condition_anchor, condition_residual, mean_mixed, sigma_mixed
        )
        intermediate = self.backbone(x_mixed, scf_state=state)
        prediction = self.denorm(intermediate, anchor, mean_mixed, var_mixed, timestamp_y)
        return (prediction, mixup_params) if self.training else prediction


Model = PULSESCFModel
