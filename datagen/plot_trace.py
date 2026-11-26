"""Plot requested vs allocated (granted) PRBs over time for one trace.

Usage:
    python -m datagen.plot_trace --trace test_000                      # full hour
    python -m datagen.plot_trace --trace test_000 --start 9 --minutes 5
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .check_dataset import shade, SC_COLORS
from .scenarios import SCENARIOS

SLICES = ["embb", "urllc", "mmtc"]
COLORS = {"embb": "#1f77b4", "urllc": "#d62728", "mmtc": "#2ca02c"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="test_000")
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--start", type=float, default=0.0, help="start minute")
    ap.add_argument("--minutes", type=float, default=None, help="window length (default: to end)")
    ap.add_argument("--total-prb", type=int, default=162)
    ap.add_argument("--out", default="reports/plots")
    a = ap.parse_args()

    d = pd.read_parquet(os.path.join(a.data, f"trace_{a.trace}.parquet"))
    t0 = a.start * 60
    t1 = t0 + a.minutes * 60 if a.minutes else d.t.max() + 1
    d = d[(d.t >= t0) & (d.t < t1)].reset_index(drop=True)
    tm = d.t / 60

    fig, axs = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    for ax, s in zip(axs, SLICES):
        shade(ax, d)
        req, gr = d[f"{s}_req_prb_mean"], d[f"{s}_granted_prb"]
        ax.plot(tm, req, color=COLORS[s], lw=1.1, label="requested PRBs")
        ax.plot(tm, gr, color="k", lw=0.9, ls="--", label="allocated (granted) PRBs")
        ax.fill_between(tm, gr, req, where=req > gr, color="orange", alpha=0.6, label="unmet demand")
        unmet_pct = 100 * (req - gr).clip(lower=0).sum() / max(req.sum(), 1e-9)
        ax.set_ylabel("PRBs")
        ax.set_title(f"{s.upper()}  — mean req {req.mean():.1f}, mean alloc {gr.mean():.1f}, "
                     f"unmet {unmet_pct:.1f}% of demand", fontsize=10, loc="left")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)

    ax = axs[3]
    shade(ax, d)
    ax.stackplot(tm, *[d[f"{s}_granted_prb"] for s in SLICES],
                 colors=[COLORS[s] for s in SLICES], alpha=0.75,
                 labels=[f"{s.upper()} allocated" for s in SLICES])
    ax.plot(tm, d["total_req_prb_mean"], color="purple", lw=1.1, label="total requested")
    ax.axhline(a.total_prb, color="k", ls=":", lw=1.2, label=f"capacity {a.total_prb} PRBs")
    ax.set_ylabel("PRBs"); ax.set_xlabel("time (min)")
    ax.set_title("All slices: stacked allocation vs total requested", fontsize=10, loc="left")
    ax.legend(loc="upper right", fontsize=8, ncol=2); ax.grid(alpha=0.25)

    handles = [plt.Rectangle((0, 0), 1, 1, color=SC_COLORS[s], alpha=0.5) for s in SCENARIOS]
    fig.legend(handles, SCENARIOS, loc="upper center", ncol=7, fontsize=9, title=f"Requested vs allocated PRBs — trace {a.trace}   |   background = scenario")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(a.out, exist_ok=True)
    suffix = f"_{a.start:g}m-{(t1 / 60):g}m" if (a.start or a.minutes) else "_full"
    path = os.path.join(a.out, f"req_vs_alloc_{a.trace}{suffix}.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
