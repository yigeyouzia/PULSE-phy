"""Gjemnessund adapter. Original PULSE datasets do not import this module.

Protocol: per-record chronological 50/25/25, train-row-only StandardScaler,
no cross-record windows, and actual 0.5-second timestamps. H5 date payloads
are inspected with pickletools; no pickle deserialization is executed.
"""
import csv
import hashlib
import json
import os
import pickletools
import re
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.gje_timefeatures import gje_time_features

CHANNELS = ['accelerationG5 ' + side + axis for side in ('east', 'west')
            for axis in 'xyz'] + ['windG5 wind' + s for s in ('T', 'U', 'angle', 'w')]
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?$')
STEP_NS = 500_000_000
CACHE_VERSION = 2


def _text(x):
    return x.decode('utf-8') if isinstance(x, (bytes, np.bytes_)) else str(x)


def _blocks(group):
    return sorted(k[:-6] for k in group if re.fullmatch(r'block\d+_items', k))


def read_dates(group):
    n = len(group['axis1'])
    for stem in _blocks(group):
        names = [_text(x) for x in group[stem + '_items'][:]]
        if 'date' not in names:
            continue
        if names != ['date']:
            raise ValueError('Expected date in its own pandas fixed block.')
        ds = group[stem + '_values']
        if ds.dtype.kind == 'O':
            payload = ds[0]
            if not isinstance(payload, np.ndarray) or payload.dtype != np.uint8:
                raise ValueError('Unsupported date object block; see exploration notebook.')
            strings = [arg for op, arg, _ in pickletools.genops(payload.tobytes())
                       if op.name in {'BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE'}
                       and isinstance(arg, str) and DATE_RE.fullmatch(arg)]
        elif ds.dtype.kind in 'SU':
            strings = [_text(x) for x in ds[:].reshape(n, -1)[:, 0]]
        else:
            raise ValueError('Unsupported date dtype.')
        if len(strings) != n:
            raise ValueError('Date count differs from numeric row count.')
        normalized = [s.replace('T', ' ') if '.' in s else s.replace('T', ' ') + '.000000000' for s in strings]
        dates = pd.DatetimeIndex(pd.to_datetime(normalized, format='%Y-%m-%d %H:%M:%S.%f', errors='raise'))
        ns = dates.values.astype('datetime64[ns]').astype(np.int64)
        if len(ns) < 2 or not np.all(np.diff(ns) == STEP_NS):
            raise ValueError('Record is not strictly regular at 0.5 seconds.')
        if np.any(ns % STEP_NS):
            raise ValueError('Timestamp does not align with the 0.5-second grid.')
        return dates
    raise ValueError('Missing date column.')


def read_values(group):
    cols = [_text(x) for x in group['axis0'][:] if _text(x) != 'date']
    if cols != CHANNELS:
        raise ValueError('Unexpected Gje channel schema/order: ' + repr(cols))
    n = len(group['axis1'])
    values = {}
    for stem in _blocks(group):
        names = [_text(x) for x in group[stem + '_items'][:]]
        if 'date' in names:
            continue
        block = group[stem + '_values'][:]
        if block.shape != (n, len(names)) or block.dtype.kind not in 'fiu':
            raise ValueError('Unexpected numeric block shape/dtype.')
        values.update((name, block[:, j]) for j, name in enumerate(names))
    x = np.stack([values[c] for c in CHANNELS], axis=1).astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Record contains NaN/Inf; audit it before training (no silent imputation).')
    return x


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def prepare_gje(path, cache_dir=None, record_limit=0):
    """One scan shared by all horizons/splits; cache original numeric arrays.

    The data signature is a lightweight identity (size/mtime/header hash), not
    a full-file cryptographic content audit. Copies on another host rebuild it.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError('Place gjemnessund.h5 in all_datasets/: ' + str(path))
    stat = path.stat()
    with path.open('rb') as f:
        header_hash = hashlib.sha256(f.read(65536)).hexdigest()
    identity = dict(version=CACHE_VERSION, name=path.name, size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns, header_sha256=header_hash,
                    record_limit=int(record_limit), columns=CHANNELS)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(cache_dir).expanduser() if cache_dir else path.parent / '.gje_cache'
    folder = root / key
    info_path = folder / 'metadata.json'
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding='utf-8'))
        if info['identity'] == identity and all((folder / r['array']).is_file() for r in info['records']):
            return folder, info
    (folder / 'records').mkdir(parents=True, exist_ok=True)
    scaler, records, dropped_records = StandardScaler(), [], []
    previous_end = None
    with h5py.File(str(path), 'r') as f:
        keys = sorted(k for k in f if isinstance(f[k], h5py.Group)
                      and 'axis0' in f[k] and 'axis1' in f[k])
        if record_limit:
            keys = keys[:record_limit]
        if not keys:
            raise ValueError('No pandas fixed Gje records found.')
        print('Gje: checking %d records and fitting TRAIN rows only (first run).' % len(keys), flush=True)
        for i, name in enumerate(keys):
            try:
                dates, x = read_dates(f[name]), read_values(f[name])
            except Exception as e:
                raise ValueError('Gje record %s: %s' % (name, e)) from e
            start_ns, end_ns = int(dates[0].value), int(dates[-1].value)
            original_n = len(x)
            trim = max(0, (previous_end - start_ns) // STEP_NS + 1) if previous_end is not None else 0
            if trim >= original_n:
                dropped_records.append(dict(record_id=name, reason='fully overlapped by earlier record', rows=original_n))
                continue
            if trim:
                x = x[trim:]
                start_ns += trim * STEP_NS
                print('Gje overlap: trim %d prefix rows from %s' % (trim, name), flush=True)
            previous_end = end_ns
            n = len(x)
            a, b = n // 2, n - n // 4
            if a < 1:
                raise ValueError('Record has no training rows: ' + name)
            scaler.partial_fit(x[:a])
            filename = 'records/%06d.npy' % i
            tmp = folder / (filename + '.%d.tmp' % os.getpid())
            with tmp.open('wb') as out:
                np.save(out, x, allow_pickle=False)
            tmp.replace(folder / filename)
            records.append(dict(record_id=name, nrows=n, source_nrows=original_n,
                                source_start_row=int(trim), train_end=a, val_end=b,
                                start_ns=start_ns, end_ns=end_ns, array=filename))
            if (i + 1) % 50 == 0 or i + 1 == len(keys):
                print('Gje preprocessing: %d/%d records' % (i + 1, len(keys)), flush=True)
    info = dict(identity=identity, protocol='per_record_50_25_25_train_scaler_trim_overlap_v1',
                columns=CHANNELS, sample_interval_seconds=0.5, records=records,
                overlap_policy='chronological first record owns overlapping timestamps; trim later prefix',
                dropped_records=dropped_records,
                scaler=dict(mean=scaler.mean_.tolist(), var=scaler.var_.tolist(),
                            scale=scaler.scale_.tolist(), fit_rows=int(scaler.n_samples_seen_),
                            fit_scope='each record rows [0, train_end), once per raw row'))
    _write_json(info_path, info)
    return folder, info


@lru_cache(maxsize=32)
def _array(path):
    # mmap avoids holding three full copies for train/validation/test.
    return np.load(path, mmap_mode='r', allow_pickle=False)


class Dataset_Gjemnessund(Dataset):
    weighted_evaluation = True

    def __init__(self, root_path, data_path='gjemnessund.h5', flag='train',
                 size=None, features='M', time_dim=6, window_stride=16,
                 cycle_len=96, cache_dir=None, record_limit=0):
        if flag not in ('train', 'val', 'test'):
            raise ValueError('Gje supports train/val/test; unseen-future predict is not implemented.')
        if features != 'M' or time_dim != 6:
            raise ValueError('Initial Gje adapter requires features=M and time_dim=6 (calendar sin/cos).')
        self.seq_len, label_len, self.pred_len = size or (720, 0, 96)
        if label_len != 0 or self.seq_len < 2 or self.pred_len < 1 or window_stride < 1 or cycle_len < 1:
            raise ValueError('Require label_len=0, seq_len>=2, pred_len/stride/cycle_len>=1.')
        self.flag, self.window_stride, self.cycle_len = flag, window_stride, cycle_len
        self.folder, self.info = prepare_gje(Path(root_path) / data_path, cache_dir, record_limit)
        self.columns = self.info['columns']
        self.mean = np.asarray(self.info['scaler']['mean'])
        self.scale = np.asarray(self.info['scaler']['scale'])
        self.records = self.info['records']
        windows, skipped = [], []
        for i, record in enumerate(self.records):
            n, a, b = record['nrows'], record['train_end'], record['val_end']
            lo, hi = {'train': (0, a), 'val': (a, b), 'test': (b, n)}[flag]
            start = 0 if flag == 'train' else lo - self.seq_len
            last = hi - self.seq_len - self.pred_len
            if start < 0 or start > last:
                skipped.append(record['record_id'])
                continue
            windows.extend((i, s) for s in range(start, last + 1, window_stride))
        self.windows = np.asarray(windows, dtype=np.int64).reshape(-1, 2)
        self.skipped = skipped
        if not len(self.windows):
            raise ValueError('No valid %s windows; check T/H and record lengths.' % flag)
        index_name = 'windows_T%d_H%d_S%d_%s.npy' % (self.seq_len, self.pred_len, window_stride, flag)
        self.index_path = self.folder / index_name
        # Deterministic, compact manifest: [record index, absolute start row].
        if not self.index_path.exists():
            temp = self.index_path.with_name(index_name + '.%d.tmp' % os.getpid())
            with temp.open('wb') as f:
                np.save(f, self.windows, allow_pickle=False)
            temp.replace(self.index_path)
        self.origin_ns = pd.Timestamp('2013-01-01').value
        print('Gje %s: %d windows, %d records skipped; TRAIN scaler rows=%d' %
              (flag, len(self), len(skipped), self.info['scaler']['fit_rows']), flush=True)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        record_index, start = map(int, self.windows[index])
        record = self.records[record_index]
        end, stop = start + self.seq_len, start + self.seq_len + self.pred_len
        raw = _array(str(self.folder / record['array']))[start:stop]
        data = ((raw - self.mean) / self.scale).astype(np.float32)
        ticks = record['start_ns'] + np.arange(start, stop, dtype=np.int64) * STEP_NS
        marks = gje_time_features(pd.to_datetime(ticks))
        cycle = np.int64(((record['start_ns'] + end * STEP_NS - self.origin_ns) // STEP_NS) % self.cycle_len)
        return data[:self.seq_len], data[self.seq_len:], marks[:self.seq_len], marks[self.seq_len:], cycle

    def inverse_transform(self, data):
        return data * self.scale + self.mean

    def export_protocol(self, path):
        """Persist portable scaler + record map + exact window manifest for comparison."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        _write_json(path / 'gje_data_protocol.json', self.info)
        with (path / ('gje_%s_windows.npy' % self.flag)).open('wb') as f:
            np.save(f, self.windows, allow_pickle=False)
        _write_json(path / ('gje_%s_summary.json' % self.flag),
                    dict(split=self.flag, windows=len(self), seq_len=self.seq_len,
                         pred_len=self.pred_len, window_stride=self.window_stride,
                         cycle_len=self.cycle_len, cycle_origin='2013-01-01',
                         cycle_at='first target timestamp', time_features='day/week/year sin/cos',
                         skipped_records=self.skipped))
        with (path / 'gje_records.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.records[0]))
            writer.writeheader()
            writer.writerows(self.records)
