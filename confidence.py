"""Prediction confidence v1 (PLAN.md 3.4) — PreNS-style, based on recent prediction error.

For slice s and horizon h at prediction time t:
    matured predictions = those made at t' <= t - h (their actual value y(t'+h) is already observed at t)
    r_h(t) = RMSE of the last W matured predictions                      (no future information)
    C_h(t) = 2 ** (-r_h(t) / tau_h)        tau_h = median of r_h on the VALIDATION split
So C = 0.5 means typical accuracy, C -> 1 the model has recently been very accurate, C -> 0 it is failing.
Before enough history exists (< min_n matured predictions) the validation-wide RMSE is used.

Usage:
    python confidence.py --run predp_uni_all_5-10-15_full        # fit tau on val, check on test, save
"""
import argparse
import json
import os

import numpy as np

WINDOW = 30
MIN_N = 5


def matured_rolling_rmse(pred, truth, H, window=WINDOW, min_n=MIN_N):
    """pred, truth: (N, K) for CONSECUTIVE anchors (1 s apart). Returns (N, K) rolling RMSE, NaN if too little history."""
    N, K = pred.shape
    out = np.full((N, K), np.nan)
    i = np.arange(N)
    for k, h in enumerate(H):
        e2 = (pred[:, k] - truth[:, k]) ** 2
        e2 = np.where(np.isnan(e2), 0.0, e2)       # only the last h rows can be NaN and they never mature
        cs = np.concatenate([[0.0], np.cumsum(e2)])
        hi = i - h + 1                               # matured anchors j satisfy j <= i - h  -> j < hi
        lo = np.maximum(0, hi - window)
        n = hi - lo
        ok = n >= min_n
        out[ok, k] = np.sqrt((cs[hi[ok]] - cs[lo[ok]]) / n[ok])
    return out


def to_confidence(rr, tau, fallback):
    """rr: (N, K) rolling RMSE; tau, fallback: (K,) arrays."""
    r = np.where(np.isnan(rr), fallback[None, :], rr)
    return np.power(2.0, -r / np.maximum(tau[None, :], 1e-9))


class Confidence:
    def __init__(self, spec):
        self.spec = spec
        self.window = spec["window"]
        self.min_n = spec["min_n"]

    @classmethod
    def load(cls, run_dir):
        f = os.path.join(run_dir, "confidence.json")
        return cls(json.load(open(f))) if os.path.exists(f) else None

    def compute(self, slice_, H, pred, truth):
        """pred/truth (N, K) for consecutive anchors and horizons H -> confidence (N, K) in (0, 1]."""
        sp = self.spec["slices"][slice_]
        tau = np.array([sp["tau"][str(h)] for h in H])
        fb = np.array([sp["fallback_rmse"][str(h)] for h in H])
        rr = matured_rolling_rmse(pred, truth, H, self.window, self.min_n)
        return to_confidence(rr, tau, fb)


def collect(P, dfs, H):
    """Run the predictor on every trace; returns per-slice lists of (pred, truth) arrays."""
    out = {}
    for d in dfs:
        _, pred, truth = P.predict_df(d, H)
        for s in pred:
            out.setdefault(s, []).append((pred[s], truth[s]))
    return out


def fit(P, val_dfs, window=WINDOW, min_n=MIN_N):
    H = P.model_h
    spec = {"method": "rolling_matured_rmse_v1", "window": window, "min_n": min_n,
            "formula": "C = 2^(-rolling_rmse / tau), tau = median rolling RMSE on validation",
            "horizons": H, "slices": {}}
    for s, pairs in collect(P, val_dfs, H).items():
        rrs = np.concatenate([matured_rolling_rmse(p, t, H, window, min_n) for p, t in pairs])
        err = np.concatenate([p - t for p, t in pairs])
        spec["slices"][s] = {
            "tau": {str(h): float(np.nanmedian(rrs[:, k])) for k, h in enumerate(H)},
            "fallback_rmse": {str(h): float(np.sqrt(np.nanmean(err[:, k] ** 2))) for k, h in enumerate(H)},
        }
    return spec


def calibration(P, conf, test_dfs, H_eval, n_bins=5):
    """On test: bin predictions by confidence and report the actual error in each bin."""
    from scipy.stats import spearmanr
    res = {}
    raw = {}
    for s, pairs in collect(P, test_dfs, P.model_h).items():
        idx = [P.model_h.index(h) for h in H_eval]
        C = np.concatenate([conf.compute(s, P.model_h, p, t)[:, idx] for p, t in pairs])
        AE = np.concatenate([np.abs(p - t)[:, idx] for p, t in pairs])
        res[s], raw[s] = {}, (C, AE)
        for k, h in enumerate(H_eval):
            c, ae = C[:, k], AE[:, k]
            ok = ~np.isnan(ae)
            c, ae = c[ok], ae[ok]
            edges = np.quantile(c, np.linspace(0, 1, n_bins + 1))
            b = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, n_bins - 1)
            bins = [{"c_lo": float(edges[j]), "c_hi": float(edges[j + 1]), "c_mean": float(c[b == j].mean()),
                     "mae": float(ae[b == j].mean()), "rmse": float(np.sqrt((ae[b == j] ** 2).mean())),
                     "n": int((b == j).sum())} for j in range(n_bins) if (b == j).any()]
            rho = spearmanr(c, ae).correlation
            res[s][str(h)] = {"bins": bins, "spearman_conf_vs_abs_error": float(rho),
                              "mean_conf": float(c.mean()),
                              "mae_high_conf": float(ae[c >= 0.5].mean()) if (c >= 0.5).any() else None,
                              "mae_low_conf": float(ae[c < 0.5].mean()) if (c < 0.5).any() else None,
                              "share_high_conf": float((c >= 0.5).mean())}
    return res, raw


def plot(res, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#f59e0b", "#8b5cf6", "#0891b2"]
    fig, axs = plt.subplots(1, len(res), figsize=(5 * len(res), 3.8))
    axs = np.atleast_1d(axs)
    for ax, (s, per) in zip(axs, res.items()):
        for i, (h, r) in enumerate(per.items()):
            ax.plot([b["c_mean"] for b in r["bins"]], [b["mae"] for b in r["bins"]], marker="o",
                    color=colors[i % 3], label=f"{h} s (ρ={r['spearman_conf_vs_abs_error']:.2f})")
        ax.set(title=s.upper(), xlabel="confidence C (bin mean)", ylabel="mean |error| (PRBs)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def calibrate_run(run_dir, data=None, H_eval=None, window=WINDOW, min_n=MIN_N, quiet=False):
    from predictor import Predictor
    from prep.data import load_split
    P = Predictor(run_dir, device="cpu")
    data = data or P.cfg["data"]
    spec = fit(P, load_split(data, "val"), window, min_n)
    conf = Confidence(spec)
    H_eval = H_eval or P.cfg["horizons_selected"]
    res, _ = calibration(P, conf, load_split(data, "test"), H_eval)
    spec["calibration_test"] = res
    with open(os.path.join(run_dir, "confidence.json"), "w") as f:
        json.dump(spec, f, indent=1)
    plot(res, os.path.join(run_dir, "confidence_calibration.png"),
         f"{P.name}: higher confidence should mean lower error (test split)")
    if not quiet:
        print(f"Confidence calibrated for {P.name} (window {window} s)")
        print(f"{'slice':6s} {'h':>3s} | {'tau':>6s} | {'ρ(C,|err|)':>10s} | MAE C>=0.5  MAE C<0.5 | share C>=0.5")
        for s, per in res.items():
            for h, r in per.items():
                print(f"{s:6s} {h:>3s} | {spec['slices'][s]['tau'][h]:6.2f} | {r['spearman_conf_vs_abs_error']:10.3f} | "
                      f"{(r['mae_high_conf'] or float('nan')):10.2f} {(r['mae_low_conf'] or float('nan')):10.2f} | "
                      f"{r['share_high_conf']:.2f}")
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--window", type=int, default=WINDOW)
    a = ap.parse_args()
    rd = a.run if os.path.isdir(a.run) else os.path.join(a.runs_dir, a.run)
    calibrate_run(rd, window=a.window)


if __name__ == "__main__":
    main()
