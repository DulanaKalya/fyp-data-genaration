"""Loading, cleaning and scaling of 1 s slice traces (PLAN.md Phase 2)."""
import json
import os

import numpy as np
import pandas as pd

SLICES = ["embb", "urllc", "mmtc"]
MULTI_BASE = ["req_prb_mean", "req_prb_max", "active_ues", "thr_mbps", "cqi_mean", "granted_prb"]


DIRECTIONS = ("dl", "ul")


def dcol(slice_, metric="req_prb_mean", direction="dl"):
    """Column name of a per-slice metric: DL keeps the original names (embb_req_prb_mean),
    UL adds '_ul_' (embb_ul_req_prb_mean). active_ues is shared by both directions."""
    if direction not in DIRECTIONS:
        raise ValueError(direction)
    if direction == "dl" or metric == "active_ues":
        return f"{slice_}_{metric}"
    return f"{slice_}_ul_{metric}"


def target_col(slice_, direction="dl"):
    return dcol(slice_, "req_prb_mean", direction)


def slice_of(col):
    """'embb_ul_req_prb_mean' -> 'embb'."""
    return col.split("_")[0]


def feature_columns(features, slice_=None, direction="dl"):
    """uni: the slice's own demand only (PreNS). multi: KPIs of all slices (same direction).
    pair: the slice's demand in BOTH directions (target direction first) - E14 cross-direction test."""
    if features == "uni":
        return [target_col(slice_, direction)]
    if features == "pair":
        other = "ul" if direction == "dl" else "dl"
        return [target_col(slice_, direction), target_col(slice_, other)]
    if features == "multi":
        return [dcol(s, b, direction) for s in SLICES for b in MULTI_BASE]
    raise ValueError(features)


def has_direction(data_dir, direction):
    """True if the dataset in data_dir has columns for this direction (UL needs sim 2.0)."""
    if direction == "dl":
        return True
    m = load_manifest(data_dir)
    first = next(iter(m["traces"]))
    cols = pd.read_parquet(os.path.join(data_dir, f"trace_{first}.parquet"), columns=None).columns
    return target_col(SLICES[0], direction) in cols


def load_manifest(data_dir):
    with open(os.path.join(data_dir, "manifest.json")) as f:
        return json.load(f)


def load_split(data_dir, split, max_traces=None, manifest=None):
    manifest = manifest or load_manifest(data_dir)
    ids = manifest["splits"][split][:max_traces]
    return [pd.read_parquet(os.path.join(data_dir, f"trace_{t}.parquet")) for t in ids]


def clean(df, cols, median_filter=True, max_gap=2):
    """Interpolate gaps <= max_gap s, then a CAUSAL 3-point median (uses t-2..t only, no future leak)."""
    out = df.copy()
    for c in cols:
        x = out[c].astype(np.float64).interpolate(limit=max_gap, limit_area="inside")
        x = x.ffill().bfill()
        if median_filter:
            x = x.rolling(3, min_periods=1).median()
        out[c] = x.astype(np.float32)
    return out


class MinMaxScaler:
    """Per-column min-max to [0, 1]. Fit on TRAIN traces only."""

    def __init__(self, mins=None, maxs=None):
        self.mins, self.maxs = mins or {}, maxs or {}

    def fit(self, dfs, cols):
        for c in cols:
            self.mins[c] = float(min(d[c].min() for d in dfs))
            self.maxs[c] = float(max(d[c].max() for d in dfs))
        return self

    def transform(self, df, cols):
        lo = np.array([self.mins[c] for c in cols], np.float32)
        hi = np.array([self.maxs[c] for c in cols], np.float32)
        return (df[cols].to_numpy(np.float32) - lo) / np.maximum(hi - lo, 1e-9)

    def inverse(self, x, col):
        return x * (self.maxs[col] - self.mins[col]) + self.mins[col]

    def to_dict(self):
        return {"mins": self.mins, "maxs": self.maxs}

    @classmethod
    def from_dict(cls, d):
        return cls(dict(d["mins"]), dict(d["maxs"]))


TARGET_KINDS = ("point", "mean", "max")


def horizon_targets(y, t, horizons, kind="point"):
    """Target for anchor t (last observed second) and horizon h, shape (N, K); NaN past the end of y.
       point: y[t+h]                     (PreNS: demand at exactly t+h)
       mean : mean(y[t+1 .. t+h])        (average demand to reserve for the next h s)
       max  : max (y[t+1 .. t+h])        (peak demand in the next h s)"""
    y = np.asarray(y, np.float64)
    t = np.asarray(t)
    H = np.asarray(horizons)
    if kind == "point":
        idx = t[:, None] + H[None, :]
        return np.where(idx < len(y), y[np.minimum(idx, len(y) - 1)], np.nan)
    J = np.arange(1, H.max() + 1)
    idx = t[:, None] + J[None, :]
    path = np.where(idx < len(y), y[np.minimum(idx, len(y) - 1)], np.nan)     # (N, maxH)
    agg = np.cumsum(path, 1) / J[None, :] if kind == "mean" else np.maximum.accumulate(path, 1)
    return agg[:, H - 1]                                                       # NaN propagates past the end
