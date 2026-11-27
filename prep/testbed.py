"""Import testbed KPI logs (E2 KPM exports) into the simulator's dataset format.

The output folder (data/testbed/<name>/) has the same layout as data/raw/ — trace_<split>_NNN.parquet
plus manifest.json — so it works with train.py (fine-tuning, E7), evaluate.py, infer.py and the UI.

Steps
  1. read CSV/Parquet; parse the time column (epoch s, epoch ms or ISO text)
  2. long format (one row per time x slice) -> wide (<slice>_<metric> columns)
  3. rename / scale columns using the mapping YAML (configs/testbed_map.yaml)
  4. resample to 1 s (mean; max for *_max); gaps stay NaN (clean() fills gaps <= 2 s later)
  5. fill missing optional columns (see DERIVED) so 'uni' and 'multi' models both run
  6. cut into continuous pieces at gaps > --max-gap s, chop into --chunk-min chunks,
     and split the chunks chronologically into train / val / test (default 70 / 15 / 15)

Examples
    python -m prep.testbed --input templates/testbed_kpm_long_example.csv --map configs/testbed_map.yaml \\
                           --name example
    python -m prep.testbed --input my_wide_kpis.csv --name day1        # already uses our column names
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from .data import SLICES

METRICS = ["req_prb_mean", "req_prb_max", "req_prb_p95", "granted_prb", "unmet_prb", "active_ues",
           "thr_mbps", "cqi_mean", "buffer_kb", "latency_ms_p95"]
# uplink metrics use the same names with "ul_" (e.g. embb_ul_req_prb_mean; map ul_req_prb_mean / ul_granted_prb
# in the mapping YAML). active_ues is shared by both directions.
UL_METRICS = [m for m in METRICS if m != "active_ues"]
# metric -> how to fill it when the testbed does not report it (documented in manifest 'derived')
DERIVED = {
    "req_prb_max": "req_prb_mean", "req_prb_p95": "req_prb_mean",
    "granted_prb": "req_prb_mean", "unmet_prb": 0.0, "active_ues": np.nan, "thr_mbps": np.nan,
    "cqi_mean": np.nan, "buffer_kb": 0.0, "latency_ms_p95": np.nan,
}


def parse_time(s, unit="auto"):
    """Return seconds (float) from epoch s / epoch ms / ISO strings."""
    if unit == "iso" or (unit == "auto" and not pd.api.types.is_numeric_dtype(s)):
        return pd.to_datetime(s, utc=True).astype("int64").to_numpy() / 1e9
    v = s.astype(np.float64).to_numpy()
    if unit == "ms" or (unit == "auto" and np.nanmedian(v) > 1e11):
        v = v / 1e3
    return v


def to_wide(df, m):
    """Apply the mapping: returns DataFrame indexed by time (s) with <slice>_<metric> columns."""
    tcol = m.get("time", {}).get("column", "t")
    unit = m.get("time", {}).get("unit", "auto")
    if tcol not in df:
        raise ValueError(f"time column '{tcol}' not found; columns: {list(df.columns)[:20]}")
    df = df.copy()
    df["_t"] = parse_time(df[tcol], unit)
    scale = m.get("scale", {})
    cols = m.get("columns", {})
    if m.get("format", "wide") == "long":
        sc = m["slice_column"]
        names = {str(k): v for k, v in m.get("slice_values", {}).items()}
        df["_slice"] = df[sc].astype(str).map(lambda x: names.get(x, x.lower()))
        df = df[df["_slice"].isin(SLICES)]
        if df.empty:
            raise ValueError(f"no rows match slices {SLICES}; set slice_values in the mapping")
        out = {}
        for metric, src in cols.items():
            if src not in df:
                raise ValueError(f"mapped column '{src}' (for {metric}) not in file")
            p = df.pivot_table(index="_t", columns="_slice", values=src, aggfunc="mean")
            for s in p.columns:
                out[f"{s}_{metric}"] = p[s] * scale.get(metric, 1.0)
        return pd.DataFrame(out).sort_index()
    # wide: keys are full names (embb_req_prb_mean: source); unmapped columns already named ours pass through
    out = {}
    for full, src in cols.items():
        if src not in df:
            raise ValueError(f"mapped column '{src}' (for {full}) not in file")
        metric = full.split("_", 1)[1]
        out[full] = df[src].astype(np.float64) * scale.get(metric, scale.get(full, 1.0))
    for s in SLICES:
        for metric in METRICS + [f"ul_{m}" for m in UL_METRICS]:
            c = f"{s}_{metric}"
            if c not in out and c in df:
                out[c] = df[c].astype(np.float64)
    return pd.DataFrame(out, index=df["_t"].to_numpy()).sort_index()


def resample_1s(w):
    # nearest second relative to the first report: robust to report-time jitter (< 0.5 s)
    sec = np.rint(w.index.to_numpy() - w.index.min()).astype(np.int64)
    g = w.groupby(sec)
    r = g.mean()
    mx = [c for c in w.columns if c.endswith("_max")]
    if mx:
        r[mx] = g[mx].max()
    full = np.arange(sec.max() + 1)
    return r.reindex(full), float(w.index.min())


def fill_missing(r, total_prb):
    derived = {}
    for s in SLICES:
        if f"{s}_req_prb_mean" not in r:
            if f"{s}_granted_prb" in r:
                r[f"{s}_req_prb_mean"] = r[f"{s}_granted_prb"]
                derived[f"{s}_req_prb_mean"] = "granted_prb (no demand KPI: under-estimates when saturated)"
            else:
                r[f"{s}_req_prb_mean"] = 0.0
                derived[f"{s}_req_prb_mean"] = "0 (slice not in file)"
        for metric, src in DERIVED.items():
            c = f"{s}_{metric}"
            if c not in r:
                r[c] = r[f"{s}_{src}"] if isinstance(src, str) else src
                derived[c] = src if isinstance(src, str) else str(src)
    r["total_req_prb_mean"] = sum(r[f"{s}_req_prb_mean"] for s in SLICES)
    if any(f"{s}_ul_req_prb_mean" in r or f"{s}_ul_granted_prb" in r for s in SLICES):   # file has uplink
        for s in SLICES:
            if f"{s}_ul_req_prb_mean" not in r:
                if f"{s}_ul_granted_prb" in r:
                    r[f"{s}_ul_req_prb_mean"] = r[f"{s}_ul_granted_prb"]
                    derived[f"{s}_ul_req_prb_mean"] = "ul_granted_prb (no UL demand KPI: under-estimates when saturated)"
                else:
                    r[f"{s}_ul_req_prb_mean"] = 0.0
                    derived[f"{s}_ul_req_prb_mean"] = "0 (slice UL not in file)"
            for metric, src in DERIVED.items():
                if metric == "active_ues":
                    continue
                c = f"{s}_ul_{metric}"
                if c not in r:
                    r[c] = r[f"{s}_ul_{src}"] if isinstance(src, str) else src
                    derived[c] = f"ul_{src}" if isinstance(src, str) else str(src)
        r["total_ul_req_prb_mean"] = sum(r[f"{s}_ul_req_prb_mean"] for s in SLICES)
    bad = r[[f"{s}_req_prb_mean" for s in SLICES]].gt(total_prb * 1.5).any(axis=1).sum()
    return r, derived, int(bad)


def pieces(r, max_gap, min_len):
    """Continuous pieces: split where the demand columns are NaN for more than max_gap seconds."""
    nan = r[[f"{s}_req_prb_mean" for s in SLICES]].isna().all(axis=1).to_numpy()
    out, start, run = [], None, 0
    for i, isnan in enumerate(np.append(nan, True)):
        if not isnan:
            if start is None:
                start = i
            run = 0
            last = i
        elif start is not None:
            run += 1
            if run > max_gap or i == len(nan):
                if last - start + 1 >= min_len:
                    out.append((start, last + 1))
                start, run = None, 0
    return out


def import_testbed(inp, name, mapping=None, out_root="data/testbed", chunk_min=10, split=(0.7, 0.15, 0.15),
                   max_gap=30, min_len=120, total_prb=162, label_column=None):
    m = yaml.safe_load(open(mapping)) if mapping else {}
    df = pd.read_csv(inp) if inp.endswith(".csv") else pd.read_parquet(inp)
    if not m and "t" not in df:
        for c in ("timestamp", "time", "ts"):
            if c in df:
                m = {"time": {"column": c}}
                break
    w = to_wide(df, m)
    r, t0 = resample_1s(w)
    r, derived, n_bad = fill_missing(r, total_prb)
    lab = label_column or m.get("label_column")
    r["scenario"] = "testbed"
    if lab and lab in df:          # optional scenario labels from the experiment log -> per-second mode
        tm = m.get("time", {})
        sec = np.rint(parse_time(df[tm.get("column", "t")], tm.get("unit", "auto")) - t0).astype(np.int64)
        lab_s = df[lab].astype(str).groupby(sec).agg(lambda x: x.mode().iat[0]).reindex(r.index)
        r["scenario"] = lab_s.ffill().bfill().fillna("testbed").to_numpy()

    chunks = []
    step = int(chunk_min * 60)
    for a, b in pieces(r, max_gap, min_len):
        for c in range(a, b, step):
            e = min(b, c + step)
            if e - c >= min_len:
                chunks.append((c, e))
            elif chunks and chunks[-1][1] == c:          # merge a short tail into the previous chunk
                chunks[-1] = (chunks[-1][0], e)
    if len(chunks) < 3:
        raise ValueError(f"only {len(chunks)} usable chunk(s); need >= 3 (record longer or lower --chunk-min)")
    n = len(chunks)
    n_val = max(1, round(n * split[1])); n_test = max(1, round(n * split[2])); n_train = n - n_val - n_test
    names = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test      # chronological: no look-ahead
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    splits, traces, counters = {"train": [], "val": [], "test": []}, {}, {"train": 0, "val": 0, "test": 0}
    for (a, b), sp in zip(chunks, names):
        tid = f"{sp}_{counters[sp]:03d}"; counters[sp] += 1
        d = r.iloc[a:b].copy()
        gaps = int(d[f"{SLICES[0]}_req_prb_mean"].isna().sum())
        d.insert(0, "t", np.arange(len(d)))
        d.insert(1, "trace_id", tid)
        d.to_parquet(os.path.join(out_dir, f"trace_{tid}.parquet"), index=False)
        meta = {"trace_id": tid, "profile": "testbed", "source": os.path.basename(inp),
                "start_epoch_s": t0 + a, "seconds": len(d), "missing_s": gaps}
        json.dump(meta, open(os.path.join(out_dir, f"trace_{tid}.json"), "w"), indent=1)
        splits[sp].append(tid); traces[tid] = {"split": sp, "profile": "testbed"}
    secs = sum(b - a for a, b in chunks)
    manifest = {"sim_version": "testbed", "config": {
        "source": os.path.abspath(inp), "mapping": mapping, "imported": datetime.now().isoformat(timespec="seconds"),
        "trace_hours": round(secs / len(chunks) / 3600, 3), "out_dt_s": 1.0, "radio": {"total_prb": total_prb},
        "chunk_min": chunk_min, "split_fractions": list(split), "max_gap_s": max_gap,
        "derived": derived, "rows_over_1.5x_total_prb": n_bad},
        "splits": splits, "traces": traces}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    return out_dir, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV or Parquet KPI export")
    ap.add_argument("--name", required=True, help="output folder name under data/testbed/")
    ap.add_argument("--map", default=None, help="mapping YAML (see configs/testbed_map.yaml)")
    ap.add_argument("--out-root", default="data/testbed")
    ap.add_argument("--chunk-min", type=float, default=10, help="trace length in minutes")
    ap.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15], help="train val test fractions")
    ap.add_argument("--max-gap", type=int, default=30, help="cut traces at gaps longer than this (s)")
    ap.add_argument("--total-prb", type=int, default=162)
    a = ap.parse_args()
    out, m = import_testbed(a.input, a.name, a.map, a.out_root, a.chunk_min, tuple(a.split), a.max_gap,
                            total_prb=a.total_prb)
    sp = {k: len(v) for k, v in m["splits"].items()}
    print(f"Saved -> {out}   traces {sp}   ({m['config']['trace_hours'] * 60:.1f} min each on average)")
    if m["config"]["derived"]:
        print("Filled (not in the file):")
        for c, src in m["config"]["derived"].items():
            print(f"  {c:28s} <- {src}")
    if m["config"]["rows_over_1.5x_total_prb"]:
        print(f"WARNING: {m['config']['rows_over_1.5x_total_prb']} s with demand > 1.5 x total PRBs — check units")


if __name__ == "__main__":
    main()
