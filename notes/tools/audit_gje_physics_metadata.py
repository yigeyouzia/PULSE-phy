"""Read-only unit audit against the official 2 Hz Gjemnessund HDF5.

Requires requirements-gje.txt and requests (audit only, not model runtime).
Run from PULSE-phy: python notes/tools/audit_gje_physics_metadata.py \
    --local-h5 /path/to/gjemnessund.h5 --output /tmp/gje-unit-audit.json
Only selected official records are fetched with bounded HTTP range reads.
This is not a full-file MD5 verification or an audit of all local records.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import h5py
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


URL = 'https://zenodo.org/records/5979695/files/data_2Hz.h5?download=1'
SIZE = 542863860
RECORDS = ('NTNU-134-2013-11-27_10-06-12',
           'NTNU-134-2013-11-28_00-10-26',
           'NTNU-134-2013-11-30_21-06-58')


class OfficialRangeReader(io.RawIOBase):
    """Seekable H5 source with explicit download budget and disk block cache."""

    def __init__(self, cache):
        super().__init__()
        self.position = 0
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.mount('https://', HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.3)))
        self.blocks = {}
        self.fetched_bytes = 0
        self.network_requests = 0
        self.started = time.monotonic()
        self.block_size = 1024 * 1024

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else SIZE) + offset
        if self.position < 0:
            raise ValueError('Negative H5 seek')
        return self.position

    def _block(self, start):
        if start in self.blocks:
            return self.blocks[start]
        end = min(start + self.block_size, SIZE) - 1
        path = self.cache / '{}-{}.bin'.format(start, end)
        if path.is_file() and path.stat().st_size == end - start + 1:
            data = path.read_bytes()
        else:
            if self.network_requests >= 24 or time.monotonic() - self.started > 150:
                raise RuntimeError('Metadata audit exceeded bounded request/time budget')
            with self.session.get(URL + '&meta1m=' + str(start),
                                  headers={'Range': 'bytes={}-{}'.format(start, end)},
                                  timeout=(8, 25), stream=True) as response:
                response.raise_for_status()
                if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(
                        'bytes {}-{}/'.format(start, end)):
                    raise ValueError('Server did not honor the exact bounded Range request')
                data = response.raw.read(end - start + 1)
                if len(data) != end - start + 1:
                    raise ValueError('Short HTTP range response')
            path.write_bytes(data)
            self.network_requests += 1
            self.fetched_bytes += len(data)
        self.blocks[start] = data
        return data

    def read(self, n=-1):
        if n < 0:
            raise ValueError('Full-file download is disabled in this audit')
        n = min(n, max(0, SIZE - self.position))
        chunks = []
        while n:
            start = self.position // self.block_size * self.block_size
            offset = self.position - start
            part = self._block(start)[offset:offset + n]
            if not part:
                break
            chunks.append(part)
            n -= len(part)
            self.position += len(part)
        return b''.join(chunks)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def text(value):
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def attributes(group):
    return {key: text(value) for key, value in group.attrs.items()}


def array_hash(x):
    return hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest()


def audit(local_h5, cache, record_count=3):
    source = OfficialRangeReader(cache)
    records = []
    with h5py.File(source, 'r') as official, h5py.File(local_h5, 'r') as local:
        root = attributes(official)
        for name in RECORDS[:record_count]:
            # Exact lookups avoid walking/downloads of all 1856 upstream groups.
            upstream = official[name]
            local_id = name.replace('-', '_')
            repacked = local[local_id]
            columns = [text(v) for v in repacked['block0_items'][:]]
            local_values = repacked['block0_values'][:]
            channels = []
            for side in ('east', 'west'):
                sensor = upstream['acceleration']['G5 ' + side]
                for axis in 'xyz':
                    dataset = sensor[axis]
                    values = dataset[:].astype(np.float64)
                    column = 'accelerationG5 ' + side + axis
                    local_column = local_values[:, columns.index(column)].astype(np.float64)
                    if values.shape != local_column.shape:
                        raise ValueError('Upstream/local record lengths differ')
                    unit = text(dataset.attrs['unit'])
                    difference = float(np.max(np.abs(values - local_column)))
                    matched = bool(np.allclose(values, local_column, rtol=0, atol=1e-12))
                    if unit != 'g' or not matched:
                        raise ValueError('Unit/value mismatch: ' + name + '/' + column)
                    channels.append(dict(column=column, unit=unit, rows=len(values),
                                         max_abs_difference=difference, matched=matched,
                                         official_sha256=array_hash(values),
                                         local_sha256=array_hash(local_column),
                                         official_attributes=attributes(dataset),
                                         sensor_attributes=attributes(sensor)))
            wind = upstream['wind']['G5 wind']
            records.append(dict(official_id=name, local_id=local_id,
                                official_attributes=attributes(upstream), channels=channels,
                                wind_sensor_attributes=attributes(wind),
                                wind_components={key: attributes(wind[key])
                                                 for key in ('T', 'U', 'angle', 'w')}))
    stat = Path(local_h5).stat()
    return dict(schema_version=1, official_url=URL, official_size_bytes=SIZE,
                published_official_md5='9f9e25d26998afc023e77ca9980c2fab',
                full_official_md5_verified=False, local_file=str(Path(local_h5).resolve()),
                local_size_bytes=stat.st_size, local_mtime_ns=stat.st_mtime_ns,
                verified_acceleration_unit='g', standard_gravity_m_s2=9.80665,
                scope='Selected complete record acceleration arrays; not all records.',
                official_root_attributes=root, records=records,
                http_requests=source.network_requests, http_bytes=source.fetched_bytes,
                reused_or_downloaded_blocks=len(source.blocks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-h5', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache', default='/tmp/gje-official-physics-range-cache')
    parser.add_argument('--records', type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    report = audit(args.local_h5, args.cache, args.records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Verified g:', len(report['records']), 'records, six complete arrays per record.')
    print('New HTTP bytes:', report['http_bytes'], 'Report:', output)
