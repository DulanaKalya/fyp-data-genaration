"""Load a trained run from the registry (runs/<id>/) and predict on 1 s traces.
Used by evaluate.py, infer.py and the UI."""
import json
import os

import numpy as np
import torch

from prep.data import clean, MinMaxScaler, horizon_targets, target_col
from models.predp import build_model


class Predictor:
    def __init__(self, run_dir, device=None):
        self.run_dir = run_dir
        self.cfg = json.load(open(os.path.join(run_dir, "config.json")))
        self.scaler = MinMaxScaler.from_dict(json.load(open(os.path.join(run_dir, "scaler.json"))))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.L = self.cfg["lookback"]
        self.model_h = self.cfg["model_horizons"]
        from confidence import Confidence
        self.conf = Confidence.load(run_dir)     # None until `python confidence.py --run ...` has been run
        self.models = {}
        for g, spec in self.cfg["groups"].items():
            ridx = [spec["features"].index(c) for c in spec["targets"]] \
                if self.cfg.get("target_mode", "level") == "residual" else None
            m = build_model(self.cfg["model"], len(spec["features"]),
                            len(spec["targets"]) * len(self.model_h), residual_idx=ridx,
                            hidden=self.cfg["hidden"], dropout=self.cfg.get("dropout", 0.0))
            sd = torch.load(os.path.join(run_dir, f"model_{g}.pt"), map_location=self.device, weights_only=True)
            m.load_state_dict(sd)
            self.models[g] = m.to(self.device).eval()

    @property
    def target(self):
        """What the model predicts: point = demand at t+h, mean / max = average / peak over (t, t+h]."""
        return self.cfg.get("target", "point")

    @property
    def direction(self):
        """dl (original runs) or ul."""
        return self.cfg.get("direction", "dl")

    @property
    def name(self):
        return self.cfg["run_id"]

    @property
    def slices(self):
        return [t.split("_")[0] for spec in self.cfg["groups"].values() for t in spec["targets"]]

    def check_horizons(self, horizons):
        missing = [h for h in horizons if h not in self.model_h]
        if missing:
            raise ValueError(f"run {self.name} was trained for horizons {self.model_h}; "
                             f"cannot serve {missing}. Retrain or pick an 'all'-mode run.")
        return [self.model_h.index(h) for h in horizons]

    @torch.no_grad()
    def predict_df(self, df, horizons, anchors=None, batch=8192):
        """Predict for every anchor t (last observed second) of one trace.
        Returns anchors (N,), pred {slice: (N, K)} and truth {slice: (N, K)} in PRBs
        (truth is NaN where t+h is past the end of the trace)."""
        kidx = self.check_horizons(horizons)
        T = len(df)
        t = np.arange(self.L - 1, T) if anchors is None else np.asarray(anchors)
        pred, truth = {}, {}
        hs = np.asarray(horizons)
        for g, spec in self.cfg["groups"].items():
            feats, tgts = spec["features"], spec["targets"]
            X = self.scaler.transform(clean(df, feats, median_filter=bool(self.cfg["median_filter"])), feats)
            Xt = torch.as_tensor(X, device=self.device)
            ar = torch.arange(-self.L + 1, 1, device=self.device)
            tt = torch.as_tensor(t, device=self.device)
            out = []
            for i in range(0, len(t), batch):
                xb = Xt[tt[i:i + batch, None] + ar[None, :]]
                out.append(self.models[g](xb).cpu())
            P = torch.cat(out).numpy().reshape(len(t), len(tgts), len(self.model_h))[:, :, kidx]
            raw = clean(df, tgts, median_filter=False)
            for j, col in enumerate(tgts):
                sl = col.split("_")[0]
                pred[sl] = np.maximum(self.scaler.inverse(P[:, j, :], col), 0.0)  # demand can't be negative
                truth[sl] = horizon_targets(raw[col].to_numpy(np.float64), t, hs, self.target)
        return t, pred, truth


def predict_with_confidence(P, df, horizons, lo, hi):
    """Predictions for anchors [lo, hi) plus confidence (needs recent history, so it predicts a bit earlier too).
    Returns t, pred, truth, conf (conf is None if the run has no calibration)."""
    T = len(df)
    lo = max(P.L - 1, lo)
    hi = min(T, hi)
    if P.conf is None:
        t, pred, truth = P.predict_df(df, horizons, anchors=np.arange(lo, hi))
        return t, pred, truth, None
    start = max(P.L - 1, lo - P.conf.window - max(horizons) - 2)
    t, pred, truth = P.predict_df(df, horizons, anchors=np.arange(start, hi))
    keep = t >= lo
    conf = {s: P.conf.compute(s, horizons, pred[s], truth_matured(df, s, t, horizons, P.target, P.direction))[keep]
            for s in pred}
    return t[keep], {s: v[keep] for s, v in pred.items()}, {s: v[keep] for s, v in truth.items()}, conf


def truth_matured(df, s, t, horizons, kind="point", direction="dl"):
    """Actual target for each anchor (matures at t+h for every kind); NaN when not yet observed."""
    return horizon_targets(df[target_col(s, direction)].to_numpy(np.float64), t, horizons, kind)


def list_runs(runs_dir="runs"):
    """Completed runs (have metrics.json), newest first."""
    out = []
    for root, dirs, files in os.walk(runs_dir):
        if "metrics.json" in files and "config.json" in files:
            out.append(root)
    return sorted(out, key=os.path.getmtime, reverse=True)
