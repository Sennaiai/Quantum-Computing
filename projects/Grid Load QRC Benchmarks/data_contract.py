"""Causal, UTC-ordered one-step forecasting from local settlement records."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
GRID = ('ENGLAND_WALES_DEMAND', 'EMBEDDED_WIND_GENERATION',
        'EMBEDDED_SOLAR_GENERATION', 'PUMP_STORAGE_PUMPING',
        'SCOTTISH_TRANSFER', 'NEMO_FLOW', 'ELECLINK_FLOW')
WINDOW = 8
SEEDS = (11, 29, 47)


def settlement_start(date, period):
    """Elapsed UTC half-hours from London midnight, including 46/50-period days."""
    day = datetime.strptime(str(date), '%Y-%m-%d').replace(tzinfo=ZoneInfo('Europe/London'))
    start = day.astimezone(timezone.utc)
    stop = (day + timedelta(days=1)).astimezone(timezone.utc)
    count = int((stop-start).total_seconds() // 1800)
    if isinstance(period, bool) or int(period) != period or not 1 <= period <= count:
        raise ValueError('Invalid settlement period for this local date')
    return start + timedelta(minutes=30*(int(period)-1))


def load_data(path=None, verify=True):
    path = Path(path) if path else ROOT/'data/neso_2025_jan_oct.csv'
    if verify:
        meta = json.loads((ROOT/'data/provenance.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('Dataset checksum mismatch')
    df = pd.read_csv(path)
    required = ['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', *GRID]
    if list(df.columns) != required or df[required].isna().any().any():
        raise ValueError('Unexpected schema or missing data; implicit imputation is forbidden')
    if not np.isfinite(df[list(GRID)].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite grid values')
    times = pd.DatetimeIndex([settlement_start(d, p) for d, p in
                             zip(df.SETTLEMENT_DATE, df.SETTLEMENT_PERIOD)])
    if not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('Settlement records must be unique and chronological')
    if len(times) < 1000 or not np.all(np.diff(times.asi8) == 1800*10**9):
        raise ValueError('A complete contiguous half-hour series is required')
    df.index = times
    return df


@dataclass
class ForecastData:
    windows: np.ndarray
    raw: np.ndarray
    y: np.ndarray
    times: pd.DatetimeIndex
    initial_train: int
    test_start: int
    scaler: StandardScaler
    persistence: np.ndarray
    daily: np.ndarray
    weekly: np.ndarray
    source_times: pd.DatetimeIndex

    def folds(self):
        edges = np.linspace(self.initial_train, self.test_start, 4, dtype=int)
        return [(np.arange(a), np.arange(a, b)) for a, b in zip(edges[:-1], edges[1:])]


def prepare(df):
    """All covariates stop at t-1. Calendar at target t is known in advance.

    Target t is observed after forecasting: rolling one-step evaluation, not
    recursive day-ahead prediction. Encoder scaler freezes at initial training.
    """
    values = df[list(GRID)].to_numpy(dtype=float)
    if len(values) < 1000 or not np.isfinite(values).all():
        raise ValueError('Insufficient finite observations')
    idx = np.arange(336, len(values))
    windows = np.stack([values[t-WINDOW:t] for t in idx])
    load_lags = np.stack([values[idx-lag, 0] for lag in range(1, 49)], axis=1)
    local = df.index[idx].tz_convert('Europe/London')
    clock = (local.hour.to_numpy()*60 + local.minute.to_numpy())/1440
    weekday = local.dayofweek.to_numpy()/7
    calendar = np.column_stack([np.sin(2*np.pi*clock), np.cos(2*np.pi*clock),
                                np.sin(2*np.pi*weekday), np.cos(2*np.pi*weekday)])
    raw = np.column_stack([windows.reshape(len(idx), -1), load_lags,
                           values[idx-336, 0], calendar])
    initial_train = int(.6*len(idx)); test_start = int(.8*len(idx))
    # Only rows available before the first validation target fit the encoder.
    scaler = StandardScaler().fit(values[:idx[initial_train]])
    encoded = scaler.transform(windows.reshape(-1, len(GRID))).reshape(windows.shape)
    return ForecastData(encoded, raw, values[idx, 0], df.index[idx], initial_train,
                        test_start, scaler, values[idx-1, 0], values[idx-48, 0],
                        values[idx-336, 0], df.index[idx-1])


def split_description(data):
    def span(a, b):
        return {'samples': int(b-a), 'first_target_utc': data.times[a].isoformat(),
                'last_target_utc': data.times[b-1].isoformat()}
    return {'initial_training': span(0, data.initial_train),
            'rolling_validation': span(data.initial_train, data.test_start),
            'heldout_test': span(data.test_start, len(data.y)),
            'folds': [dict(train=span(0, len(tr)), validation=span(va[0], va[-1]+1))
                      for tr, va in data.folds()],
            'task': 'rolling one-step 30-minute hindcast; previous outturn assumed available',
            'encoder_fit_last_exclusive_utc': data.times[data.initial_train].isoformat(),
            'window': WINDOW, 'raw_features': data.raw.shape[1]}
