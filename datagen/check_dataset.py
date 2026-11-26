"""Dataset validation (PLAN.md Phase 1.8): plots + pass/fail checks.

Usage: python -m datagen.check_dataset --data data/raw --out reports/dataset
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .scenarios import SCENARIOS

SLICES = ["embb", "urllc", "mmtc"]
COLORS = {"embb": "#1f77b4", "urllc": "#d62728", "mmtc": "#2ca02c"}
SC_COLORS = dict(zip(SCENARIOS, ["#ffffff", "#c7e9c0", "#fdd0a2", "#fcbba1",
                                 "#dadaeb", "#fb6a4a", "#9e9ac8"]))
HORIZONS = list(range(1, 16))


def load(data, split, manifest):
    ids = manifest["splits"].get(split, [])
    return [pd.read_parquet(os.path.join(data, f"trace_{t}.parquet")) for t in ids]


def acf(x, lags):
    x = x - x.mean()
    v = (x * x).mean()
    return np.array([(x[:-L] * x[L:]).mean() / v if v > 0 else 0.0 for L in lags])


def persistence_r2(dfs, col, h):
    yt, yp = [], []
    for d in dfs:
        x = d[col].to_numpy(np.float64)
        yt.append(x[h:]); yp.append(x[:-h])
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return 1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()


def shade(ax, d):
    sc = d["scenario"].astype(str).to_numpy()
    start = 0
    for i in range(1, len(sc) + 1):
        if i == len(sc) or sc[i] != sc[start]:
            ax.axvspan(d["t"].iloc[start] / 60, d["t"].iloc[i - 1] / 60 + 1 / 60,
                       color=SC_COLORS[sc[start]], alpha=0.5, lw=0)
            start = i


def plot_trace(d, path, minutes=None, total_prb=162):
    if minutes:
        d = d[d.t < minutes * 60]
    fig, axs = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    tm = d["t"] / 60
    for ax, s in zip(axs, SLICES):
        shade(ax, d)
        ax.plot(tm, d[f"{s}_req_prb_max"], color=COLORS[s], alpha=0.35, lw=0.7, label="requested (1 s max)")
        ax.plot(tm, d[f"{s}_req_prb_mean"], color=COLORS[s], lw=1.0, label="requested (1 s mean) — TARGET")
        ax.plot(tm, d[f"{s}_granted_prb"], color="k", lw=0.6, ls="--", label="granted")
        ax.set_ylabel(f"{s.upper()} PRBs"); ax.legend(loc="upper right", fontsize=8)
    ax = axs[3]; shade(ax, d)
    ax.plot(tm, d["total_req_prb_mean"], color="purple", lw=1.0, label="total requested")
    ax.axhline(total_prb, color="k", ls=":", label=f"{total_prb} PRB capacity")
    ax.set_ylabel("Total PRBs"); ax.set_xlabel("time (min)"); ax.legend(loc="upper right", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SC_COLORS[s], alpha=0.5) for s in SCENARIOS]
    fig.legend(handles, SCENARIOS, loc="upper center", ncol=7, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="reports/dataset")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    manifest = json.load(open(os.path.join(a.data, "manifest.json")))
    total_prb = manifest["config"]["radio"]["total_prb"]
    splits = {s: load(a.data, s, manifest) for s in manifest["splits"]}
    train, test = splits.get("train", []), splits.get("test", [])
    summary = {"hours": {s: len(v) * len(v[0]) / 3600 if v else 0 for s, v in splits.items()}}

    # --- statistics ---
    trn = pd.concat(train)
    summary["stats_train"] = {s: trn[f"{s}_req_prb_mean"].describe(percentiles=[.5, .95]).round(2).to_dict()
                              for s in SLICES}
    summary["scenario_share_train"] = trn["scenario"].value_counts(normalize=True).round(3).to_dict()
    over = (trn["total_req_prb_mean"] > total_prb)
    summary["frac_over_capacity_train"] = round(float(over.mean()), 3)
    summary["frac_over_capacity_by_scenario"] = (
        trn.groupby("scenario", observed=True)["total_req_prb_mean"]
        .apply(lambda x: float((x > total_prb).mean())).round(3).to_dict())

    # --- check 2: autocorrelation at 5..15 s ---
    lags = list(range(1, 31))
    acfs = {s: np.mean([acf(d[f"{s}_req_prb_mean"].to_numpy(np.float64), lags) for d in train], 0)
            for s in SLICES}
    summary["acf_train"] = {s: {str(L): round(float(v), 3) for L, v in zip(lags, acfs[s])} for s in SLICES}

    # --- check 3: persistence R2 per horizon on test ---
    pr2 = {s: [persistence_r2(test, f"{s}_req_prb_mean", h) for h in HORIZONS] for s in SLICES}
    summary["persistence_r2_test"] = {s: {str(h): round(float(v), 3) for h, v in zip(HORIZONS, pr2[s])}
                                      for s in SLICES}

    checks = {
        "acf_5_15_above_0.3": all(acfs[s][4:15].min() > 0.3 for s in SLICES),
        "persistence_r2_room_at_15s": all(pr2[s][14] < 0.85 for s in SLICES),
        "persistence_r2_nontrivial_at_5s": all(pr2[s][4] > 0.0 for s in SLICES),
        "contention_exceeds_capacity": summary["frac_over_capacity_by_scenario"].get("contention", 0) > 0.1,
    }
    summary["checks"] = checks

    # --- plots ---
    plot_trace(test[0], os.path.join(a.out, "trace_test_000_full.png"), total_prb=total_prb)
    plot_trace(test[0], os.path.join(a.out, "trace_test_000_first15min.png"), minutes=15, total_prb=total_prb)
    if splits.get("ood"):
        plot_trace(splits["ood"][0], os.path.join(a.out, "trace_ood_000_first15min.png"), minutes=15,
                   total_prb=total_prb)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for s in SLICES:
        axs[0].plot(lags, acfs[s], marker=".", color=COLORS[s], label=s)
        axs[1].plot(HORIZONS, pr2[s], marker="o", color=COLORS[s], label=s)
    for ax in axs:
        ax.axvspan(5, 15, color="grey", alpha=0.12); ax.grid(alpha=0.3); ax.legend()
    axs[0].set(title="Autocorrelation of requested PRBs (train)", xlabel="lag (s)", ylabel="ACF")
    axs[1].set(title="Persistence baseline R² (test)  ŷ(t+h)=y(t)", xlabel="horizon h (s)", ylabel="R²")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "acf_and_persistence.png"), dpi=110); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, s in zip(axs, SLICES):
        ax.hist(trn[f"{s}_req_prb_mean"], bins=80, color=COLORS[s])
        ax.set(title=f"{s.upper()} requested PRBs (train)", xlabel="PRBs")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "distributions.png"), dpi=110); plt.close(fig)

    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print("Hours per split:", {k: round(v, 1) for k, v in summary["hours"].items()})
    print("Over-capacity share (train):", summary["frac_over_capacity_train"])
    for s in SLICES:
        print(f"{s:6s} ACF@5/10/15 = {acfs[s][4]:.2f}/{acfs[s][9]:.2f}/{acfs[s][14]:.2f}   "
              f"persistence R2@5/10/15 = {pr2[s][4]:.2f}/{pr2[s][9]:.2f}/{pr2[s][14]:.2f}")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"Plots + summary.json -> {a.out}")


if __name__ == "__main__":
    main()
