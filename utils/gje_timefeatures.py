"""Calendar features for the stored, timezone-naive 2 Hz Gje timestamps."""
import numpy as np
import pandas as pd


def gje_time_features(dates):
    """Return [N, 6]: sin/cos of day, week and actual calendar year."""
    dates = pd.DatetimeIndex(dates)
    if dates.tz is not None:
        raise ValueError('Gje timestamps must retain their original timezone-naive convention.')
    ns = dates.values.astype('datetime64[ns]').astype(np.int64)
    midnight = dates.normalize().values.astype('datetime64[ns]').astype(np.int64)
    fraction = (ns - midnight) / 86_400_000_000_000.0
    phases = (fraction, (dates.dayofweek.to_numpy() + fraction) / 7.0,
              (dates.dayofyear.to_numpy() - 1 + fraction) /
              np.where(dates.is_leap_year, 366.0, 365.0))
    return np.stack([fn(2 * np.pi * p) for p in phases
                     for fn in (np.sin, np.cos)], axis=1).astype(np.float32)
