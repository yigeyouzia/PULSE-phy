"""Gje acceleration calibration: StandardScaler <-> g <-> m/s^2.

The scaler is provided by the existing train-row-only dataset adapter.
No HDF5 read, fitting, or test-data statistics take place in a model.
"""

import hashlib
import json

import torch
from torch import nn


STANDARD_GRAVITY = 9.80665
SAMPLE_INTERVAL_SECONDS = 0.5
MODAL_FREQUENCIES_HZ = (0.090, 0.176, 0.220, 0.250, 0.720)
MODAL_SOURCE = '10.1016/j.jweia.2023.105583, Table 3, Full-scale column'
UNIT_SOURCE = '10.5281/zenodo.5979695, data_2Hz.h5, acceleration/G5 east|west/x|y|z: unit=g'
GJE_COLUMNS = (['accelerationG5 ' + side + axis for side in ('east', 'west') for axis in 'xyz']
               + ['windG5 wind' + name for name in ('T', 'U', 'angle', 'w')])


class GjeAccelerationUnits(nn.Module):
    """Persist calibration in checkpoints and reject incompatible re-binding."""

    def __init__(self):
        super().__init__()
        self.register_buffer('mean_g', torch.zeros(1, 1, 6))
        self.register_buffer('scale_g', torch.ones(1, 1, 6))
        self.protocol_signature = None

    def bind_protocol(self, info):
        if info.get('columns') != GJE_COLUMNS:
            raise ValueError('Physics models require the verified Gje ten-channel schema')
        if info.get('sample_interval_seconds') != SAMPLE_INTERVAL_SECONDS:
            raise ValueError('Gje physics models require the actual 2 Hz sampling interval')
        if info.get('protocol') != 'per_record_50_25_25_train_scaler_trim_overlap_v1':
            raise ValueError('Unverified Gje split/scaler protocol')
        scaler = info['scaler']
        if scaler.get('fit_scope') != 'each record rows [0, train_end), once per raw row':
            raise ValueError('The acceleration scaler must be fitted on training rows only')
        mean = torch.as_tensor(scaler['mean'], dtype=self.mean_g.dtype, device=self.mean_g.device)
        scale = torch.as_tensor(scaler['scale'], dtype=self.scale_g.dtype, device=self.scale_g.device)
        if mean.shape != (10,) or scale.shape != (10,) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError('Invalid Gje calibration')
        signature = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
        if self.protocol_signature is not None and self.protocol_signature != signature:
            raise ValueError('Gje calibration/protocol differs from the bound model/checkpoint')
        self.mean_g.copy_(mean[:6].view(1, 1, 6))
        self.scale_g.copy_(scale[:6].view(1, 1, 6))
        self.protocol_signature = signature

    def _require_bound(self):
        if self.protocol_signature is None:
            raise RuntimeError('Bind the existing training-set Gje scaler before physics forward')

    def to_si(self, standardized_acceleration):
        self._require_bound()
        return (standardized_acceleration * self.scale_g + self.mean_g) * STANDARD_GRAVITY

    def from_si(self, acceleration_si):
        self._require_bound()
        return (acceleration_si / STANDARD_GRAVITY - self.mean_g) / self.scale_g

    def si_residual_to_standardized(self, residual_si):
        self._require_bound()
        return residual_si / (self.scale_g * STANDARD_GRAVITY)

    def get_extra_state(self):
        return dict(schema_version=1, acceleration_unit='g', standard_gravity=STANDARD_GRAVITY,
                    sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
                    protocol_signature=self.protocol_signature, source=UNIT_SOURCE)

    def set_extra_state(self, state):
        expected = self.get_extra_state()
        expected['protocol_signature'] = state.get('protocol_signature')
        if expected != state or state['protocol_signature'] is None:
            raise RuntimeError('Invalid physics checkpoint calibration metadata')
        if self.protocol_signature is not None and self.protocol_signature != state['protocol_signature']:
            raise RuntimeError('Physics checkpoint uses a different dataset/scaler')
        self.protocol_signature = state['protocol_signature']
