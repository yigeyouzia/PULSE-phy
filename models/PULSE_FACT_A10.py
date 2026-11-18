"""A10: PULSE_FACT's phase anchors and reconstruction with a per-sensor residual MLP."""

import torch
from torch import nn

from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.pulse_fact.ablation import SensorResidualMLP, mix_input_and_stats, validate_forecast_config
from models.PULSE_FACT import ReVIN


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        validate_forecast_config(configs)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.inv_len = configs.inv_len
        self.patch_size = configs.patch_size
        self.time_dim = configs.time_dim
        if self.patch_size < 1 or self.seq_len % self.patch_size or self.pred_len % self.patch_size:
            raise ValueError("patch_size must divide both seq_len and pred_len")
        if min(self.inv_len, self.time_dim, configs.d_model, configs.ksize, configs.dsa, configs.dsb) < 1:
            raise ValueError("Anchor dimensions and time encoder width divisors must be positive")
        first_width = self.enc_in // configs.dsa
        second_width = self.enc_in // configs.dsb
        if min(first_width, second_width) < 1:
            raise ValueError("Time encoder widths must be positive")

        # Initialize the shared modules in the same order as the complete model.
        self.backbone = nn.Sequential(nn.Linear(self.seq_len, 512), nn.GELU(), nn.Linear(512, self.pred_len))
        self.router_sender = AttentionLayer(
            FullAttention(False, factor=5, attention_dropout=0.1, output_attention=False), configs.d_model, 1,
        )
        self.router_receiver = AttentionLayer(
            FullAttention(False, factor=5, attention_dropout=0.1, output_attention=False), configs.d_model, 1,
        )
        self.projection_1 = nn.Linear(self.seq_len // self.patch_size, configs.d_model)
        self.projection_2 = nn.Linear(self.pred_len // self.patch_size, configs.d_model)
        self.projection_3 = nn.Sequential(
            nn.Linear(configs.d_model, 2 * configs.d_model), nn.GELU(),
            nn.Linear(2 * configs.d_model, self.pred_len // self.patch_size),
        )
        self.revin = ReVIN(self.enc_in)
        self.codebook = nn.Parameter(torch.randn(self.inv_len, self.enc_in))
        nn.init.xavier_normal_(self.codebook)
        self.time_enc_x = self._time_encoder(self.seq_len, first_width, second_width, configs.ksize)
        self.time_enc_y = self._time_encoder(self.pred_len, first_width, second_width, configs.ksize)
        self.sensor_mlp = SensorResidualMLP()

    def _time_encoder(self, length, first_width, second_width, kernel_size):
        return nn.Sequential(
            nn.Linear(self.time_dim, first_width), nn.LayerNorm(first_width), nn.GELU(),
            nn.Linear(first_width, second_width), nn.LayerNorm(second_width), nn.GELU(),
            nn.Conv1d(length, length, kernel_size=kernel_size, padding="same"),
            nn.Linear(second_width, self.enc_in),
        )

    def norm(self, x, cycle_index, timestamp_x):
        positions = cycle_index.reshape(-1, 1) - torch.arange(self.seq_len, device=x.device)
        past_anchor = self.codebook[positions % self.inv_len] + timestamp_x
        residual, (mean, var) = self.revin(x - past_anchor, return_stats=True)
        return residual + past_anchor, past_anchor, mean, var

    def phase_router(self, forecast, past_anchor):
        # One sequence per sensor. Tokens index offsets within a patch.
        batch, _, sensors = forecast.shape
        past = past_anchor.transpose(1, 2).reshape(batch * sensors, -1, self.patch_size)
        future = forecast.transpose(1, 2).reshape(batch * sensors, -1, self.patch_size)
        past = self.projection_1(past.transpose(1, 2))
        future = self.projection_2(future.transpose(1, 2))
        buffer, _ = self.router_sender(past, future, future, attn_mask=None)
        update, _ = self.router_receiver(future, buffer, buffer, attn_mask=None)
        offsets = self.projection_3(update).transpose(1, 2)
        return offsets.reshape(batch, sensors, self.pred_len).transpose(1, 2)

    def denorm(self, forecast, past_anchor, mean, var, timestamp_y):
        future_anchor = self.phase_router(forecast, past_anchor) + timestamp_y
        return self.revin.reverse(forecast - future_anchor, mean, var) + future_anchor

    def forward(self, x, cycle_index, batch_x_mark, batch_y_mark, do_mixup=True):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("Expected history with shape [batch, seq_len, enc_in]")
        if batch_x_mark.shape != (x.size(0), self.seq_len, self.time_dim):
            raise ValueError("Unexpected historical time mark shape")
        if batch_y_mark.shape != (x.size(0), self.pred_len, self.time_dim):
            raise ValueError("Unexpected future time mark shape")
        timestamp_x = self.time_enc_x(batch_x_mark)
        timestamp_y = self.time_enc_y(batch_y_mark)
        x_in, past_anchor, mean, var = self.norm(x, cycle_index, timestamp_x)
        mixup = None
        if self.training and do_mixup:
            x_in, mean, var, mixup = mix_input_and_stats(x_in, mean, var, self.revin.eps)

        hidden = self.backbone[1](self.backbone[0](x_in.transpose(1, 2)))  # [B,C,512]
        hidden = self.sensor_mlp(hidden)
        forecast = self.backbone[2](hidden).transpose(1, 2)
        forecast = self.denorm(forecast, past_anchor, mean, var, timestamp_y)
        return (forecast, mixup) if self.training else forecast
