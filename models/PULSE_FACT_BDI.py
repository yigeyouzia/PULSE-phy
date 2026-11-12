"""Standalone PULSE with Band-Decoupled Interaction (BDI).

Expanded from models/PULSE_FACT.py. Keep the reference-retaining main path;
split SAM-aligned historical residuals along the real time axis, encode all
three paths with the same Linear/GELU, and route low/high features into the
original dilation=2/1 interaction updates. No new learned parameters or loss.
Provenance: notes/data/pulse_fact_bdi_sources.json.
"""

import json

import torch
import torch.nn as nn
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import numpy as np
from layers.pulse_fact_bdi.frequency import HistoricalResidualBandSplit
from layers.pulse_fact_bdi.interaction import PULSEFACTBDIBlock

class ReVIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ReVIN, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))

    def forward(self, x, return_stats=False):
        # x: [B, L, C]
        mean = torch.mean(x, dim=1, keepdim=True)  # [B, 1, C]
        var = torch.var(x, dim=1, keepdim=True)  # [B, 1, C]

        self.mean = mean
        self.var = var

        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        x_out = norm_x * self.weight + self.bias

        if return_stats:
            return x_out, (mean, var)
        return x_out

    def reverse(self, x_out, mean=None, var=None):
        if mean is None:
            mean = self.mean
        if var is None:
            var = self.var

        x_normalized = (x_out - self.bias) / self.weight
        x_reversed = x_normalized * torch.sqrt(var + self.eps) + mean
        return x_reversed


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.inv_len = configs.inv_len
        self.patch_size =configs.patch_size
        self.time_dim=configs.time_dim
        self.c_out=configs.enc_in
        self.dsa=configs.dsa
        self.dsb=configs.dsb
        self.ksize=configs.ksize

        if getattr(configs, "use_amp", False) or getattr(configs, "use_multi_gpu", False):
            raise ValueError("PULSE_FACT_BDI supports the non-AMP, single-device SAM path")
        if getattr(configs, "label_len", 0) != 0:
            raise ValueError("PULSE_FACT_BDI requires label_len=0 for the original SAM path")
        if self.seq_len < 2 or self.pred_len < 1 or self.enc_in < 1 or self.inv_len < 1:
            raise ValueError("PULSE_FACT_BDI requires seq_len >= 2 and positive prediction/channel/cycle sizes")
        if self.patch_size < 1 or self.seq_len % self.patch_size or self.pred_len % self.patch_size:
            raise ValueError("PULSE patch_size must divide seq_len and pred_len")
        if self.dsa < 1 or self.dsb < 1 or min(self.enc_in // self.dsa, self.enc_in // self.dsb) < 1:
            raise ValueError("PULSE dsa/dsb would create an empty TimeEncoder dimension")
        if self.d_model < 1 or self.time_dim < 1 or self.ksize < 1:
            raise ValueError("PULSE Router, timestamp and kernel dimensions must be positive")

        # Backbone
        self.backbone = nn.Sequential(
            nn.Linear(self.seq_len, 512),
            nn.GELU(),
            nn.Linear(512, self.pred_len),
        )
        self.router_sender = AttentionLayer(
            FullAttention(False, factor=5, attention_dropout=0.1, output_attention=False),
            self.d_model, 1
        )
        self.router_receiver = AttentionLayer(
            FullAttention(False, factor=5, attention_dropout=0.1, output_attention=False),
            self.d_model, 1
        )
        self.projection_1 = nn.Linear(self.seq_len // self.patch_size, self.d_model)
        self.projection_2 = nn.Linear(self.pred_len // self.patch_size, self.d_model)

        self.projection_3 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*2),
            nn.GELU(),
            nn.Linear(self.d_model*2, self.pred_len // self.patch_size)
        )

        self.revin = ReVIN(self.enc_in)

        self.codebook = nn.Parameter(torch.randn(self.inv_len, self.enc_in))
        nn.init.xavier_normal_(self.codebook)

        self.time_enc_x = nn.Sequential(
                                      nn.Linear(self.time_dim, self.c_out//self.dsa),
                                      nn.LayerNorm(self.c_out//self.dsa),
                                      nn.GELU(),
                                      nn.Linear(self.c_out//self.dsa, self.c_out//self.dsb),
                                      nn.LayerNorm(self.c_out//self.dsb),
                                      nn.GELU(),
                                      nn.Conv1d(in_channels=self.seq_len,out_channels=self.seq_len,kernel_size=self.ksize,padding='same'),
                                      nn.Linear(self.c_out//self.dsb, self.c_out),
                                      )
        self.time_enc_y = nn.Sequential(
                                      nn.Linear(self.time_dim, self.c_out//self.dsa),
                                      nn.LayerNorm(self.c_out//self.dsa),
                                      nn.GELU(),
                                      nn.Linear(self.c_out//self.dsa, self.c_out//self.dsb),
                                      nn.LayerNorm(self.c_out//self.dsb),
                                      nn.GELU(),
                                      nn.Conv1d(in_channels=self.pred_len,out_channels=self.pred_len,kernel_size=self.ksize,padding='same'),
                                      nn.Linear(self.c_out//self.dsb, self.c_out),
                                      )


        # Create FACT only after the original PULSE modules. Original weights
        # therefore match a freshly seeded PULSE model before any training.
        self.fact_hidden_dim = 512
        self.fact_d_ff = getattr(configs, "pulse_fact_bdi_d_ff", 256)
        self.fact_dilation = 2  # Fixed dilation=1/2 pair; the only first-version routing.
        self.fact_num_kernels = getattr(configs, "pulse_fact_bdi_num_kernels", 2)
        self.fact_dropout = getattr(configs, "pulse_fact_bdi_dropout", 0.1)
        self.fact_block = PULSEFACTBDIBlock(
            enc_in=self.enc_in,
            hidden_dim=self.fact_hidden_dim,
            d_ff=self.fact_d_ff,
            num_kernels=self.fact_num_kernels,
            dropout=self.fact_dropout,
        )
        self.bdi_cutoff_ratio = getattr(configs, "pulse_fact_bdi_cutoff_ratio", 0.25)
        self.band_split = HistoricalResidualBandSplit(self.seq_len, self.bdi_cutoff_ratio)
        print("PULSE_FACT_BDI effective settings: " + json.dumps(self.get_extra_state(), sort_keys=True))

    def get_extra_state(self):
        return {
            "pulse_fact_bdi_schema_version": 1,
            "model": "PULSE_FACT_BDI",
            "settings": {
                "seq_len": self.seq_len, "pred_len": self.pred_len,
                "enc_in": self.enc_in, "router_dim": self.d_model,
                "inv_len": self.inv_len, "patch_size": self.patch_size,
                "time_dim": self.time_dim, "dsa": self.dsa, "dsb": self.dsb,
                "ksize": self.ksize, "hidden_dim": self.fact_hidden_dim,
                "d_ff": self.fact_d_ff, "num_kernels": self.fact_num_kernels,
                "dropout": self.fact_dropout, "depth": 1,
                "cutoff_ratio_to_nyquist": float(self.bdi_cutoff_ratio),
                "low_bin_count": int(self.band_split.low_mask.sum().item()),
                "fft_axis": 1, "fft_norm": "ortho",
                "low_dilation": 2, "high_dilation": 1,
                "shared_temporal_encoder": True,
                "main_path": "normalized_residual_plus_reference",
                "band_input": "sam_mixed_normalized_historical_residual",
                "restoration_calls": 1,
            },
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("PULSE_FACT_BDI checkpoint configuration differs from this model")

    def norm_bdi(self, x, cycle_index, timestamp_x):

        gather_index = (cycle_index.view(-1, 1) - torch.arange(self.seq_len, device=x.device).view(1, -1)) % self.inv_len
        phase_anchor_x = self.codebook[gather_index]+timestamp_x

        residual_x = x - phase_anchor_x

        residual_x_norm, (mean, var) = self.revin(residual_x, return_stats=True)

        x_in = residual_x_norm + phase_anchor_x

        return x_in, phase_anchor_x, mean, var, residual_x_norm

    def phase_router(self, y_backbone, phase_anchor_x):


        b, t, c = y_backbone.size()

        feat_x = phase_anchor_x.permute(0, 2, 1)
        feat_y = y_backbone.permute(0, 2, 1)


        feat_x = feat_x.reshape(b * c, -1, self.patch_size)
        feat_y = feat_y.reshape(b * c, -1, self.patch_size)


        feat_x = self.projection_1(feat_x.permute(0, 2, 1))
        feat_y = self.projection_2(feat_y.permute(0, 2, 1))


        router_buffer, _ = self.router_sender(feat_x, feat_y, feat_y, attn_mask=None)
        router_receive, _ = self.router_receiver(feat_y, router_buffer, router_buffer, attn_mask=None)


        E = self.projection_3(router_receive).permute(0, 2, 1).reshape(b, c, -1).permute(0, 2, 1)

        E = E[:, -self.pred_len:, :]

        return E

    def denorm(self, y_backbone, phase_anchor_x, mean, var,timestamp_y):

        phase_anchor_y = self.phase_router(y_backbone, phase_anchor_x)+timestamp_y


        residual_y = y_backbone - phase_anchor_y

        residual_y_denorm = self.revin.reverse(residual_y, mean=mean, var=var)


        y_final = residual_y_denorm + phase_anchor_y

        return y_final

    def forward(self, x, cycle_index,batch_x_mark,batch_y_mark,do_mixup=True):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("PULSE_FACT_BDI input must have shape [B, seq_len, enc_in]")
        if batch_x_mark.shape != (x.shape[0], self.seq_len, self.time_dim):
            raise ValueError("PULSE_FACT_BDI historical timestamps have an unexpected shape")
        if batch_y_mark.shape != (x.shape[0], self.pred_len, self.time_dim):
            raise ValueError("PULSE_FACT_BDI future timestamps require pred_len steps")

        timestamp_x = self.time_enc_x(batch_x_mark)
        timestamp_y = self.time_enc_y(batch_y_mark)

        x_in, phase_anchor_x, mean, var, residual_norm = self.norm_bdi(
            x, cycle_index, timestamp_x)

        mixup_params = None
        if self.training and do_mixup:
            lam = np.random.beta(0.15, 0.15)
            batch_size = x.shape[0]
            index = torch.randperm(batch_size).to(x.device)

            x_in_mixed = lam * x_in + (1.0 - lam) * x_in[index, :]
            residual_mixed = lam * residual_norm + (1.0 - lam) * residual_norm[index, :]
            mean_mixed = lam * mean + (1.0 - lam) * mean[index, :]
            sigma = torch.sqrt(var + 1e-5)
            sigma_mixed = lam * sigma + (1.0 - lam) * sigma[index, :]
            var_mixed = sigma_mixed * sigma_mixed - 1e-5
            mixup_params = (lam, index)
        else:
            x_in_mixed = x_in
            residual_mixed = residual_norm
            mean_mixed = mean
            var_mixed = var

        # Split the observed time axis AFTER the same SAM pairing as the main path.
        residual_low, residual_high = self.band_split(residual_mixed)  # [B,L,C]

        # Three calls to ONE encoder; no second set of Linear/GELU parameters.
        hidden_main = self.backbone[0](x_in_mixed.permute(0, 2, 1))
        hidden_main = self.backbone[1](hidden_main)                    # [B,C,512]
        hidden_low = self.backbone[0](residual_low.permute(0, 2, 1))
        hidden_low = self.backbone[1](hidden_low)
        hidden_high = self.backbone[0](residual_high.permute(0, 2, 1))
        hidden_high = self.backbone[1](hidden_high)

        # Main residual once; low/high enter dilation=2/1 updates inside the block.
        hidden = self.fact_block(hidden_main, hidden_low, hidden_high)
        y_backbone = self.backbone[2](hidden).permute(0, 2, 1)          # [B,H,C]

        y_final = self.denorm(y_backbone,phase_anchor_x,mean_mixed,var_mixed,timestamp_y)

        if self.training:
            return y_final, mixup_params
        else:
            return y_final
