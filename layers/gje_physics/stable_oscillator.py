"""A nominal stable forced modal decoder, not an identified bridge PDE.

For each mode q'' + 2*zeta*omega*q' + omega^2*q = f. Dimensionless states
p=omega^2*q/a_ref and r=omega*q'/a_ref obey an exact ZOH discretization.
Frequencies and positive damping have explicit bounds. Six accelerations
are decoded from modal accelerations; no unconstrained output residual.
"""

import math

import torch
from torch import nn

from .units import MODAL_FREQUENCIES_HZ, SAMPLE_INTERVAL_SECONDS


class GjeStableForcedOscillator(nn.Module):
    def __init__(self, seq_len, pred_len, frequencies_hz=MODAL_FREQUENCIES_HZ,
                 frequency_relative_band=0.1, damping_min=0.002, damping_max=0.1,
                 initial_damping=0.02, force_controls=8, force_limit=2.0):
        super().__init__()
        frequencies = torch.as_tensor(frequencies_hz, dtype=torch.float32)
        if frequencies.ndim != 1 or frequencies.numel() < 1 or not torch.isfinite(frequencies).all():
            raise ValueError('Invalid oscillator modal frequency list')
        if not (0 <= frequency_relative_band < 1) or not ((frequencies > 0) & (frequencies * (1 + frequency_relative_band) < 1)).all():
            raise ValueError('The entire oscillator frequency band must be below Nyquist')
        if not (0 < damping_min < initial_damping < damping_max < 1):
            raise ValueError('Require strictly positive bounded underdamped oscillators')
        if seq_len < 2 or pred_len < 1 or force_controls < 2 or not math.isfinite(force_limit) or force_limit <= 0:
            raise ValueError('Invalid oscillator lengths/forcing budget')
        self.pred_len = pred_len
        self.history_steps = min(seq_len, 64)
        self.mode_count = frequencies.numel()
        self.frequency_relative_band = float(frequency_relative_band)
        self.damping_min, self.damping_max = float(damping_min), float(damping_max)
        self.force_controls = min(force_controls, pred_len)
        self.force_limit = float(force_limit)
        self.register_buffer('nominal_frequencies_hz', frequencies)
        self.raw_frequency_offset = nn.Parameter(torch.zeros(self.mode_count))
        initial_ratio = (initial_damping - damping_min) / (damping_max - damping_min)
        self.raw_damping = nn.Parameter(torch.full((self.mode_count,), math.log(initial_ratio / (1 - initial_ratio))))

        # Only observed history and the backbone's own forecast enter these
        # heads. The 8-knot forcing is intentionally lower dimensional than H.
        self.initial_state_head = nn.Linear(self.history_steps * 10, 2 * self.mode_count)
        self.force_head = nn.Linear(pred_len * 10, self.force_controls * self.mode_count)
        nn.init.normal_(self.initial_state_head.weight, std=0.001)
        nn.init.zeros_(self.initial_state_head.bias)
        nn.init.normal_(self.force_head.weight, std=0.001)
        nn.init.zeros_(self.force_head.bias)
        self.sensor_modes = nn.Parameter(torch.randn(6, self.mode_count) * 0.1)

        locations = torch.linspace(0, self.force_controls - 1, pred_len)
        knots = torch.arange(self.force_controls, dtype=torch.float32)
        interpolation = (1 - (locations[:, None] - knots[None, :]).abs()).clamp_min(0)
        self.register_buffer('force_interpolation', interpolation)

    def frequencies_hz(self):
        return self.nominal_frequencies_hz * (1 + self.frequency_relative_band * torch.tanh(self.raw_frequency_offset))

    def damping_ratios(self):
        return self.damping_min + (self.damping_max - self.damping_min) * torch.sigmoid(self.raw_damping)

    def transition_powers(self, steps):
        """A^n, n=0..steps. Exact matrix exponential for the two-state ODE."""
        omega = 2 * math.pi * self.frequencies_hz()
        damping = self.damping_ratios()
        beta = torch.sqrt(1 - damping.square())
        times = torch.arange(steps + 1, dtype=omega.dtype, device=omega.device) * SAMPLE_INTERVAL_SECONDS
        phase = omega[:, None] * beta[:, None] * times
        envelope = torch.exp(-damping[:, None] * omega[:, None] * times)
        sine, cosine = torch.sin(phase) / beta[:, None], torch.cos(phase)
        row_0 = torch.stack((cosine + damping[:, None] * sine, sine), dim=-1)
        row_1 = torch.stack((-sine, cosine - damping[:, None] * sine), dim=-1)
        return envelope[:, :, None, None] * torch.stack((row_0, row_1), dim=-2)

    def simulate(self, initial_state, forcing):
        """Return acceleration [B,H,K], states [B,H,K,2], for ZOH forcing.

        FFT evaluates the exact *linear*, causal convolution with zero padding;
        it does not imply a periodic future or use true future observations.
        """
        if forcing.shape[1:] != (self.pred_len, self.mode_count) or initial_state.shape != (forcing.shape[0], self.mode_count, 2):
            raise ValueError('Oscillator state/force shapes must be [B,K,2]/[B,H,K]')
        powers = self.transition_powers(self.pred_len)  # [K,H+1,2,2]
        identity = torch.eye(2, dtype=powers.dtype, device=powers.device)
        input_vector = (identity - powers[:, 1])[:, :, 0]  # B=(I-A)*[1,0].
        kernel = torch.einsum('ktij,kj->kti', powers[:, :-1], input_vector)
        free_state = torch.einsum('ktij,bkj->btki', powers[:, 1:], initial_state)
        fft_length = 1 << (2 * self.pred_len - 1).bit_length()
        force_fft = torch.fft.rfft(forcing.permute(0, 2, 1), n=fft_length, dim=-1)
        kernel_fft = torch.fft.rfft(kernel.permute(0, 2, 1), n=fft_length, dim=-1)
        driven_state = torch.fft.irfft(force_fft[:, :, None, :] * kernel_fft[None, :, :, :],
                                       n=fft_length, dim=-1)[..., :self.pred_len]
        states = free_state + driven_state.permute(0, 3, 1, 2)
        acceleration = -states[..., 0] - 2 * self.damping_ratios() * states[..., 1] + forcing
        return acceleration, states

    def forward(self, history_standardized, forecast_standardized, history_si):
        initial_state = 2 * torch.tanh(self.initial_state_head(
            history_standardized[:, -self.history_steps:].flatten(1)))
        initial_state = initial_state.reshape(-1, self.mode_count, 2)
        controls = self.force_limit * torch.tanh(self.force_head(forecast_standardized.flatten(1)))
        controls = controls.reshape(-1, self.force_controls, self.mode_count)
        forcing = torch.matmul(self.force_interpolation, controls)
        modal_acceleration, _ = self.simulate(initial_state, forcing)
        observation = self.sensor_modes / self.sensor_modes.square().sum(dim=1, keepdim=True).add(1e-8).sqrt()
        dynamic_acceleration = torch.matmul(modal_acceleration, observation.T)
        # The constant observation offset comes exclusively from the past; it
        # accounts for sensor/slow gravity bias, not a free time-varying bypass.
        offset = history_si.mean(dim=1, keepdim=True)
        acceleration_scale = (history_si - offset).square().mean(dim=(1, 2), keepdim=True).add(1e-10).sqrt()
        return offset + acceleration_scale * dynamic_acceleration
