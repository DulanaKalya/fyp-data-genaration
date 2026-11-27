"""Dataset validation (PLAN.md Phase 1.8): plots + pass/fail checks.

Usage: python -m datagen.check_dataset --data data/raw --out reports/dataset
       python -m datagen.check_dataset --data data/raw_v2 --out reports/dataset/raw_v2 --compare-dl data/raw

DL keys of summary.json are unchanged (the UI reads them). Datasets with uplink (sim 2.0) also get
summary["ul"] (same statistics for UL), summary["direction_stats"] and summary["dl_ul_link"].
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


def col(s, direction, metric="req_prb_mean"):
    return f"{s}_{metric}" if direction == "dl" else f"{s}_ul_{metric}"


def series_stats(trn, c, unmet_c):
    from scipy.stats import skew
    x = trn[c].to_numpy(np.float64)
    return {"mean": round(float(x.mean()), 2), "std": round(float(x.std()), 2),
            "cv": round(float(x.std() / max(x.mean(), 1e-9)), 3), "skew": round(float(skew(x)), 3),
            "p95": round(float(np.percentile(x, 95)), 2), "max": round(float(x.max()), 2),
            "zero_frac": round(float((x < 0.05).mean()), 4),
            "saturated_frac": round(float((trn[unmet_c] > 0.5).mean()), 4)}


def cross_corr(dfs, a, b, lags):
    """corr(a(t), b(t+L)) averaged over traces, for L in lags (L > 0: a leads b)."""
    out = []
    for L in lags:
        r = []
        for d in dfs:
            x, y = d[a].to_numpy(np.float64), d[b].to_numpy(np.float64)
            if L > 0:
                x, y = x[:-L], y[L:]
            elif L < 0:
                x, y = x[-L:], y[:L]
            if x.std() > 0 and y.std() > 0:
                r.append(np.corrcoef(x, y)[0, 1])
        out.append(float(np.mean(r)) if r else float("nan"))
    return out


def plot_trace(d, path, minutes=None, total_prb=162, direction="dl"):
    if minutes:
        d = d[d.t < minutes * 60]
    fig, axs = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    tm = d["t"] / 60
    for ax, s in zip(axs, SLICES):
        shade(ax, d)
        ax.plot(tm, d[col(s, direction, "req_prb_max")], color=COLORS[s], alpha=0.35, lw=0.7, label="requested (1 s max)")
        ax.plot(tm, d[col(s, direction)], color=COLORS[s], lw=1.0, label="requested (1 s mean) — TARGET")
        ax.plot(tm, d[col(s, direction, "granted_prb")], color="k", lw=0.6, ls="--", label="granted")
        ax.set_ylabel(f"{s.upper()} {direction.upper()} PRBs"); ax.legend(loc="upper right", fontsize=8)
    ax = axs[3]; shade(ax, d)
    tot = "total_req_prb_mean" if direction == "dl" else "total_ul_req_prb_mean"
    ax.plot(tm, d[tot], color="purple", lw=1.0, label=f"total requested ({direction.upper()})")
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
    ap.add_argument("--compare-dl", default=None, help="DL-only dataset whose DL columns must be identical")
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
    # --- uplink (sim 2.0): same statistics per direction + DL<->UL link per slice ---
    has_ul = all(col(s, "ul") in trn for s in SLICES)
    summary["directions"] = ["dl", "ul"] if has_ul else ["dl"]
    acfs_ul, pr2_ul = {}, {}
    if has_ul:
        acfs_ul = {s: np.mean([acf(d[col(s, "ul")].to_numpy(np.float64), lags) for d in train], 0) for s in SLICES}
        pr2_ul = {s: [persistence_r2(test, col(s, "ul"), h) for h in HORIZONS] for s in SLICES}
        over_ul = trn["total_ul_req_prb_mean"] > total_prb
        summary["ul"] = {
            "stats_train": {s: trn[col(s, "ul")].describe(percentiles=[.5, .95]).round(2).to_dict() for s in SLICES},
            "acf_train": {s: {str(L): round(float(v), 3) for L, v in zip(lags, acfs_ul[s])} for s in SLICES},
            "persistence_r2_test": {s: {str(h): round(float(v), 3) for h, v in zip(HORIZONS, pr2_ul[s])}
                                    for s in SLICES},
            "frac_over_capacity_train": round(float(over_ul.mean()), 3),
            "frac_over_capacity_by_scenario": trn.assign(o=over_ul).groupby("scenario", observed=True)["o"]
            .mean().round(3).to_dict()}
        summary["direction_stats"] = {s: {d: series_stats(trn, col(s, d), col(s, d, "unmet_prb")) for d in ("dl", "ul")}
                                      for s in SLICES}
        summary["direction_stats"]["ul_dl_mean_ratio"] = {
            s: round(float(trn[col(s, "ul")].mean() / max(trn[col(s, "dl")].mean(), 1e-9)), 3) for s in SLICES}
        xl = list(range(-15, 16))
        link = {}
        for s in SLICES:            # E14 step 1: does one direction tell us about the other's future?
            c = cross_corr(train, col(s, "dl"), col(s, "ul"), xl)
            link[s] = {"lags": xl, "corr_dl_t_ul_t_plus_L": [round(v, 3) for v in c],
                       "corr_lag0": round(c[15], 3),
                       "dl_leads_ul_at_5_10_15": [round(c[15 + h], 3) for h in (5, 10, 15)],
                       "ul_leads_dl_at_5_10_15": [round(c[15 - h], 3) for h in (5, 10, 15)]}
        summary["dl_ul_link"] = link
        checks.update({
            "ul_acf_5_15_above_0.3": all(acfs_ul[s][4:15].min() > 0.3 for s in SLICES),
            "ul_persistence_r2_room_at_15s": all(pr2_ul[s][14] < 0.85 for s in SLICES),
            "ul_persistence_r2_nontrivial_at_5s": all(pr2_ul[s][4] > 0.0 for s in SLICES),
            "ul_not_saturated_most_of_the_time": all(summary["direction_stats"][s]["ul"]["saturated_frac"] < 0.2
                                                     for s in SLICES),
        })
    if a.compare_dl:                # DL must be bit-identical to the DL-only dataset (same seeds)
        mref = json.load(open(os.path.join(a.compare_dl, "manifest.json")))
        same, n = True, 0
        for sp, ids in mref["splits"].items():
            for t in ids:
                f = os.path.join(a.data, f"trace_{t}.parquet")
                if not os.path.exists(f):
                    same = False; continue
                r, c = pd.read_parquet(os.path.join(a.compare_dl, f"trace_{t}.parquet")), pd.read_parquet(f)
                for k in r.columns:
                    eq = (r[k].astype(str) == c[k].astype(str)).all() if r[k].dtype.name in ("category", "object") \
                        else np.array_equal(r[k].to_numpy(), c[k].to_numpy())
                    same &= bool(eq)
                n += 1
        checks["dl_identical_to_reference"] = same
        summary["dl_reference"] = {"path": a.compare_dl, "traces_compared": n}
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

    if has_ul:
        plot_trace(test[0], os.path.join(a.out, "trace_test_000_first15min_ul.png"), minutes=15,
                   total_prb=total_prb, direction="ul")
        fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
        for s in SLICES:
            axs[0].plot(lags, acfs[s], color=COLORS[s], label=f"{s} DL")
            axs[0].plot(lags, acfs_ul[s], color=COLORS[s], ls="--", label=f"{s} UL")
            axs[1].plot(HORIZONS, pr2[s], marker="o", ms=3, color=COLORS[s], label=f"{s} DL")
            axs[1].plot(HORIZONS, pr2_ul[s], marker="s", ms=3, color=COLORS[s], ls="--", label=f"{s} UL")
            lk = summary["dl_ul_link"][s]
            axs[2].plot(lk["lags"], lk["corr_dl_t_ul_t_plus_L"], color=COLORS[s], label=s)
        for ax in axs[:2]:
            ax.axvspan(5, 15, color="grey", alpha=0.12)
        for ax in axs:
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        axs[0].set(title="Autocorrelation, DL (solid) vs UL (dashed)", xlabel="lag (s)", ylabel="ACF")
        axs[1].set(title="Persistence R² (test), DL vs UL", xlabel="horizon h (s)", ylabel="R²")
        axs[2].axvline(0, color="k", lw=0.6)
        axs[2].set(title="corr(DL(t), UL(t+L)) — L>0: DL leads UL", xlabel="L (s)", ylabel="correlation")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "dl_vs_ul.png"), dpi=110); plt.close(fig)
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
        for ax, s in zip(axs, SLICES):
            ax.hist(trn[col(s, "ul")], bins=80, color=COLORS[s], alpha=0.85)
            ax.set(title=f"{s.upper()} UL requested PRBs (train)", xlabel="PRBs")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "distributions_ul.png"), dpi=110); plt.close(fig)

    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print("Hours per split:", {k: round(v, 1) for k, v in summary["hours"].items()})
    print("Over-capacity share (train):", summary["frac_over_capacity_train"])
    for s in SLICES:
        print(f"{s:6s} ACF@5/10/15 = {acfs[s][4]:.2f}/{acfs[s][9]:.2f}/{acfs[s][14]:.2f}   "
              f"persistence R2@5/10/15 = {pr2[s][4]:.2f}/{pr2[s][9]:.2f}/{pr2[s][14]:.2f}")
    if has_ul:
        for s in SLICES:
            st, lk = summary["direction_stats"][s], summary["dl_ul_link"][s]
            print(f"{s:6s} UL ACF@5/10/15 = {acfs_ul[s][4]:.2f}/{acfs_ul[s][9]:.2f}/{acfs_ul[s][14]:.2f}   "
                  f"persistence R2@5/10/15 = {pr2_ul[s][4]:.2f}/{pr2_ul[s][9]:.2f}/{pr2_ul[s][14]:.2f}   "
                  f"UL/DL {summary['direction_stats']['ul_dl_mean_ratio'][s]:.2f}  CV DL/UL {st['dl']['cv']:.2f}/{st['ul']['cv']:.2f}  "
                  f"corr(DL,UL) lag0 {lk['corr_lag0']:.2f}")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"Plots + summary.json -> {a.out}")


if __name__ == "__main__":
    main()
