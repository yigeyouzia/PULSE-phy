"""Load the four additional traffic benchmarks using FACT's chronological splits."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features


CHANNEL_COUNTS = {
    'PEMSD7.npy': 228,
    'PEMS-BAY.csv': 325,
    'traffic.npy': 963,
    'METR-LA.csv': 207,
}


class Dataset_FACTTraffic(Dataset):
    def __init__(self, root_path, flag, size, features, data_path,
                 target=None, timeenc=1, freq='5min', time_dim=1):
        self.seq_len, label_len, self.pred_len = size
        if flag not in ('train', 'val', 'test'):
            raise ValueError(f'Unsupported FACTTraffic split: {flag!r}')
        if features != 'M' or label_len != 0:
            raise ValueError('FACTTraffic requires features=M and label_len=0')
        if min(self.seq_len, self.pred_len, time_dim) < 1:
            raise ValueError('FACTTraffic window lengths and time_dim must be positive')

        path = Path(root_path) / data_path
        channels = CHANNEL_COUNTS.get(path.name)
        if channels is None:
            raise ValueError(f'Unsupported FACTTraffic data file: {path.name!r}')

        is_npy = path.suffix == '.npy'
        if is_npy:
            # Both released arrays store the sensor axis first.
            values = np.load(path, mmap_mode='r', allow_pickle=False).T
        else:
            if timeenc != 1:
                raise ValueError('FACTTraffic CSV files require embed=timeF')
            frame = pd.read_csv(path)
            if 'date' not in frame:
                raise ValueError(f'{path.name} is missing the date column')
            dates = pd.DatetimeIndex(pd.to_datetime(frame.pop('date'), errors='raise'))
            if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
                raise ValueError(f'{path.name} dates must be strictly increasing')
            # Preserve observed rows and sensor order, including the PEMS-BAY DST jump.
            values = frame.to_numpy(dtype=np.float64)

        if values.ndim != 2 or values.shape[1] != channels:
            raise ValueError(f'{path.name} must contain {channels} sensors; got {values.shape}')
        if values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
            raise ValueError(f'{path.name} must contain finite numeric sensor values')

        num_rows = len(values)
        # The author code floors all three NPY lengths separately. For CSV, the
        # validation segment is the remainder between training and test.
        train_end = int(num_rows * 0.7)
        test_start = num_rows - int(num_rows * 0.2)
        val_end = train_end + int(num_rows * 0.1) if is_npy else test_start
        split_bounds = {
            'train': (0, train_end),
            'val': (train_end - self.seq_len, val_end),
            'test': (test_start - self.seq_len, num_rows),
        }
        self.split_start, split_end = split_bounds[flag]
        self.window_count = split_end - self.split_start - self.seq_len - self.pred_len + 1
        if self.split_start < 0 or self.window_count < 1:
            raise ValueError(f'{path.name} {flag} split has no complete forecast window')

        self.scaler = StandardScaler().fit(values[:train_end])
        self.values = self.scaler.transform(values[self.split_start:split_end]).astype(np.float32)

        if is_npy:
            self.marks = np.zeros((len(self.values), time_dim), dtype=np.float32)
        else:
            marks = time_features(dates[self.split_start:split_end], freq=freq).T
            if time_dim > marks.shape[1]:
                raise ValueError(f'{freq!r} provides only {marks.shape[1]} time features')
            self.marks = marks[:, :time_dim].astype(np.float32)

    def __len__(self):
        return self.window_count

    def __getitem__(self, index):
        history_end = index + self.seq_len
        forecast_end = history_end + self.pred_len
        return (
            self.values[index:history_end],
            self.values[history_end:forecast_end],
            self.marks[index:history_end],
            self.marks[history_end:forecast_end],
            np.int64(self.split_start + history_end),
        )

    def inverse_transform(self, values):
        return self.scaler.inverse_transform(values)
