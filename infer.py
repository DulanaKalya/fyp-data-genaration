"""Run a trained model on a trace (or any 1 s KPI file) and emit A1 prediction packages.

Each package (one per slice x horizon x prediction time), as in the feasibility report P_i = (D̂, C, T):
  {"t_pred": 1712, "slice": "urllc", "direction": "dl", "horizon_s": 7, "target_time": 1719,
   "pred_prb": 23.4, "confidence": null, "run_id": "..."}
direction: "dl" or "ul" (DL and UL are separate PRB budgets; never add them).

Examples:
    python infer.py --run predp_uni_all_5-10-15_full --h 7 9 12 --trace test_000 --start 600 --seconds 60
    python infer.py --run predp_uni_all_5-10-15_full --h 5 10 15 --file my_testbed_kpis.csv --out a1.jsonl
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from predictor import Predictor


def load_input(a):
    if a.file:
        return pd.read_csv(a.file) if a.file.endswith(".csv") else pd.read_parquet(a.file)
    return pd.read_parquet(os.path.join(a.data, f"trace_{a.trace}.parquet"))


def packages(pred_obj, df, horizons, start=None, seconds=None, with_truth=False):
    """Generate A1 packages for prediction times t in [start, start+seconds).
    confidence = calibrated C in (0, 1] (see confidence.py), or null if the run is not calibrated."""
    from predictor import predict_with_confidence
    T = len(df)
    lo = max(pred_obj.L - 1, start if start is not None else T - 1)
    hi = T if seconds is None else min(T, lo + seconds)
    t, pred, truth, conf = predict_with_confidence(pred_obj, df, horizons, lo, hi)
    tcol = df["t"].to_numpy() if "t" in df else np.arange(T)
    out = []
    for i, ti in enumerate(t):
        for s in pred:
            for k, h in enumerate(horizons):
                p = {"t_pred": int(tcol[ti]), "slice": s, "direction": pred_obj.direction, "horizon_s": int(h),
                     "target_time": int(tcol[ti]) + int(h), "pred_prb": round(float(pred[s][i, k]), 2),
                     "confidence": None if conf is None else round(float(conf[s][i, k]), 3),
                     "run_id": pred_obj.name}
                if with_truth:
                    v = truth[s][i, k]
                    p["actual_prb"] = None if np.isnan(v) else round(float(v), 2)
                out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under runs/ or a path")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--h", type=int, nargs=3, required=True, help="H1 H2 H3")
    ap.add_argument("--trace", default="test_000")
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--file", default=None, help="CSV/Parquet with the same 1 s columns (e.g. testbed KPIs)")
    ap.add_argument("--start", type=int, default=None, help="first prediction second (default: last available)")
    ap.add_argument("--seconds", type=int, default=None, help="how many prediction times")
    ap.add_argument("--with-truth", action="store_true", help="add actual_prb for checking")
    ap.add_argument("--out", default=None, help="write JSON Lines here (default: print)")
    a = ap.parse_args()

    run_dir = a.run if os.path.isdir(a.run) else os.path.join(a.runs_dir, a.run)
    pred = Predictor(run_dir)
    hs = sorted(a.h)
    df = load_input(a)
    pk = packages(pred, df, hs, a.start, a.seconds, a.with_truth)
    if a.out:
        with open(a.out, "w") as f:
            for p in pk:
                f.write(json.dumps(p) + "\n")
        print(f"{len(pk)} packages -> {a.out}")
    else:
        for p in pk:
            print(json.dumps(p))


if __name__ == "__main__":
    main()
