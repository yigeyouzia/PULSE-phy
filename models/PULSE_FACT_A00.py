"""A00: paired RevIN on observations and a shared per-sensor residual MLP."""

from torch import nn

from layers.pulse_fact.ablation import SensorResidualMLP, mix_input_and_stats, validate_forecast_config
from models.PULSE_FACT import ReVIN


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        validate_forecast_config(configs)
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.backbone = nn.Sequential(nn.Linear(self.seq_len, 512), nn.GELU(), nn.Linear(512, configs.pred_len))
        self.revin = ReVIN(self.enc_in)
        self.sensor_mlp = SensorResidualMLP()

    def forward(self, x, cycle_index, batch_x_mark, batch_y_mark, do_mixup=True):
        # The anchor-free path consumes only the historical observations [B,L,C].
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("Expected history with shape [batch, seq_len, enc_in]")
        x_in, (mean, var) = self.revin(x, return_stats=True)
        mixup = None
        if self.training and do_mixup:
            x_in, mean, var, mixup = mix_input_and_stats(x_in, mean, var, self.revin.eps)

        hidden = self.backbone[1](self.backbone[0](x_in.transpose(1, 2)))  # [B,C,512]
        hidden = self.sensor_mlp(hidden)
        forecast = self.backbone[2](hidden).transpose(1, 2)  # [B,H,C]
        forecast = self.revin.reverse(forecast, mean, var)
        return (forecast, mixup) if self.training else forecast
