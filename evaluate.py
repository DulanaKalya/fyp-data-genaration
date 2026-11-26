"""Evaluate trained runs against baselines on the same test windows (PLAN.md Phase 3.3).

Examples:
    python evaluate.py --runs predp_uni_all_5-10-15_full
    python evaluate.py --runs predp_uni_all_5-10-15_full lstm_... gru_... --tag E1
    python evaluate.py --runs predp_uni_all_5-10-15_full --split ood --tag E6_ood
Outputs -> reports/eval/<tag>/ : table.md, results.json, rmse/r2 vs horizon, predictions plot, scenarios.
"""
import argparse
import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prep.data import SLICES, load_split, target_col, clean
from predictor import Predictor
from models.metrics import regression_metrics

COLORS = {"embb": "#1f77b4", "urllc": "#d62728", "mmtc": "#2ca02c"}
BASE_STYLE = {"persistence": dict(color="k", ls="--"), "ma10": dict(color="grey", ls=":"),
              "arima": dict(color="olive", ls="-.")}


def resolve_run(r, runs_dir):
    return r if os.path.isdir(r) else os.path.join(runs_dir, r)


def arima_forecasts(train_dfs, test_dfs, anchors, n, L, H, seed=0):
    """ARIMA(2,0,1) fitted per slice on train, applied with fixed params to the last L s of each sampled window."""
    from statsmodels.tsa.arima.model import ARIMA
    rng = np.random.default_rng(seed)
    pairs = [(i, t) for i, a in enumerate(anchors) for t in a]
    pick = sorted(rng.choice(len(pairs), size=min(n, len(pairs)), replace=False))
    out = {}
    for s in SLICES:
        col = target_col(s)
        y_fit = np.concatenate([d[col].to_numpy(np.float64) for d in train_dfs[:5]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(y_fit, order=(2, 0, 1)).fit()
            F = np.empty((len(pick), max(H)))
            for j, k in enumerate(pick):
                i, t = pairs[k]
                y = test_dfs[i][col].to_numpy(np.float64)
                F[j] = res.apply(y[t - L + 1:t + 1]).forecast(max(H))
        out[s] = np.maximum(F[:, np.asarray(H) - 1], 0)
    return pick, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--split", default="test", choices=["test", "val", "ood"])
    ap.add_argument("--horizons", type=int, nargs=3, default=None)
    ap.add_argument("--arima-n", type=int, default=2000, help="ARIMA windows (0 = skip)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="reports/eval")
    a = ap.parse_args()

    preds = [Predictor(resolve_run(r, a.runs_dir)) for r in a.runs]
    H_sel = a.horizons or preds[0].cfg["horizons_selected"]
    all15 = list(range(1, 16))
    H = all15 if all(set(all15) <= set(p.model_h) for p in preds) else sorted(set(H_sel))
    for h in H_sel:
        assert h in H
    tag = a.tag or (preds[0].name if len(preds) == 1 else f"compare_{len(preds)}runs")
    out = os.path.join(a.out, tag + ("" if a.split == "test" else f"_{a.split}"))
    os.makedirs(out, exist_ok=True)

    dfs = load_split(a.data, a.split)
    Lmax = max([p.L for p in preds] + [30])
    anchors = [np.arange(Lmax - 1, len(d) - 15) for d in dfs]

    # ---- collect predictions on common anchors ----
    methods = {p.name: {} for p in preds}
    methods.update({"persistence": {}, "ma10": {}})
    truth, scen = {s: [] for s in SLICES}, []
    for d, an in zip(dfs, anchors):
        scen.append(d["scenario"].astype(str).to_numpy()[an])
        for s in SLICES:
            y = clean(d, [target_col(s)], median_filter=False)[target_col(s)].to_numpy(np.float64)
            truth[s].append(y[an[:, None] + np.asarray(H)[None, :]])
            methods["persistence"].setdefault(s, []).append(np.repeat(y[an][:, None], len(H), 1))
            ma = pd.Series(y).rolling(10, min_periods=1).mean().to_numpy()
            methods["ma10"].setdefault(s, []).append(np.repeat(ma[an][:, None], len(H), 1))
        for p in preds:
            _, pr, _ = p.predict_df(d, H, anchors=an)
            for s, v in pr.items():
                methods[p.name].setdefault(s, []).append(v)
    truth = {s: np.concatenate(v) for s, v in truth.items()}
    scen = np.concatenate(scen)
    for m in methods:
        methods[m] = {s: np.concatenate(v) for s, v in methods[m].items()}

    arima_pick = None
    if a.arima_n > 0:
        t0 = time.time()
        train_dfs = load_split(a.data, "train", 5)
        arima_pick, ar = arima_forecasts(train_dfs, dfs, anchors, a.arima_n, 30, H)
        methods["arima"] = ar
        print(f"ARIMA on {len(arima_pick)} windows ({time.time() - t0:.0f}s)")

    # ---- metrics ----
    sc = preds[0].scaler
    res = {}
    for m, per in methods.items():
        res[m] = {}
        for s, P in per.items():
            Y = truth[s] if m != "arima" else truth[s][arima_pick]
            rng_prb = sc.maxs[target_col(s)] - sc.mins[target_col(s)] if target_col(s) in sc.maxs else 1.0
            res[m][s] = {}
            for k, h in enumerate(H):
                r = regression_metrics(Y[:, k], P[:, k])
                r["nrmse"] = r["rmse"] / rng_prb
                res[m][s][str(h)] = r
    # same-subset comparison: every method on exactly the ARIMA windows (fair vs ARIMA)
    subset = {}
    if arima_pick is not None:
        for m, per in methods.items():
            subset[m] = {}
            for s, P in per.items():
                Ps = P if m == "arima" else P[arima_pick]
                Ys = truth[s][arima_pick]
                subset[m][s] = {str(h): regression_metrics(Ys[:, k], Ps[:, k]) for k, h in enumerate(H)}
        for s in subset["arima"]:
            for h in subset["arima"][s]:
                res["arima"][s][h]["skill_vs_persistence"] = \
                    1 - subset["arima"][s][h]["rmse"] / subset["persistence"][s][h]["rmse"]
    for m in res:
        if m in ("persistence", "arima"):
            continue
        for s in res[m]:
            for h in res[m][s]:
                res[m][s][h]["skill_vs_persistence"] = 1 - res[m][s][h]["rmse"] / res["persistence"][s][h]["rmse"]

    # ---- per-scenario RMSE (first run vs persistence) at selected horizons ----
    scen_tab = {}
    p0 = preds[0].name
    for s in methods[p0]:
        for h in H_sel:
            k = H.index(h)
            for name in np.unique(scen):
                msk = scen == name
                rm = np.sqrt(np.mean((truth[s][msk, k] - methods[p0][s][msk, k]) ** 2))
                rp = np.sqrt(np.mean((truth[s][msk, k] - methods["persistence"][s][msk, k]) ** 2))
                scen_tab.setdefault(s, {}).setdefault(name, {})[str(h)] = {"model": rm, "persistence": rp}

    # ---- cost (from training) ----
    cost = {}
    for p in preds:
        c = json.load(open(os.path.join(p.run_dir, "metrics.json")))["cost"]
        cost[p.name] = {"model": p.cfg["model"], "params_per_slice_model": int(np.mean([v["params"] for v in c.values()])),
                        "train_min_total": round(sum(v["train_min"] for v in c.values()), 1),
                        "infer_ms_batch1": round(float(np.mean([v["infer_ms_batch1"] for v in c.values()])), 3),
                        "peak_gpu_mb": max((v["peak_gpu_mb"] or 0) for v in c.values())}

    json.dump({"split": a.split, "horizons": H, "horizons_selected": H_sel, "n_windows": int(len(scen)),
               "arima_windows": len(arima_pick) if arima_pick is not None else 0,
               "metrics": res, "arima_subset_metrics": subset, "scenario_rmse": scen_tab, "cost": cost},
              open(os.path.join(out, "results.json"), "w"), indent=1)

    # ---- markdown table ----
    lines = [f"# Evaluation `{tag}` — split `{a.split}`, {len(scen):,} windows"
             + (f" (ARIMA on a random {len(arima_pick)}-window subset)" if arima_pick is not None else ""), ""]
    for s in SLICES:
        hdr = "| method | " + " | ".join(f"RMSE@{h}s | MAE@{h}s | R²@{h}s | skill@{h}s" for h in H_sel) + " |"
        lines += [f"## {s.upper()} (requested PRBs)", "", hdr, "|" + "---|" * (1 + 4 * len(H_sel))]
        for m in res:
            if s not in res[m]:
                continue
            cells = []
            for h in H_sel:
                r = res[m][s][str(h)]
                sk = r.get("skill_vs_persistence")
                if m == "arima":
                    r = subset[m][s][str(h)]
                cells += [f"{r['rmse']:.2f}", f"{r['mae']:.2f}", f"{r['r2']:.3f}",
                          f"{100 * sk:+.1f}%" if sk is not None else "–"]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    if subset:
        lines += [f"## Same-subset comparison ({len(arima_pick)} random windows, all methods) — RMSE / R²", ""]
        for s in SLICES:
            lines += [f"**{s.upper()}**", "", "| method | " + " | ".join(f"RMSE@{h}s | R²@{h}s" for h in H_sel) + " |",
                      "|" + "---|" * (1 + 2 * len(H_sel))]
            for m in subset:
                if s in subset[m]:
                    lines.append(f"| {m} | " + " | ".join(
                        f"{subset[m][s][str(h)]['rmse']:.2f} | {subset[m][s][str(h)]['r2']:.3f}" for h in H_sel) + " |")
            lines.append("")
    lines += ["## Cost (from training)", "", "| run | model | params | train min | infer ms (batch 1) | peak GPU MB |",
              "|---|---|---|---|---|---|"]
    for n, c in cost.items():
        lines.append(f"| {n} | {c['model']} | {c['params_per_slice_model']:,} | {c['train_min_total']} | "
                     f"{c['infer_ms_batch1']} | {c['peak_gpu_mb']} |")
    lines += ["", f"## RMSE by scenario — `{p0}` vs persistence (PRBs)", ""]
    for s, tab in scen_tab.items():
        lines += [f"**{s.upper()}**", "", "| scenario | " + " | ".join(f"model@{h} | pers@{h}" for h in H_sel) + " |",
                  "|" + "---|" * (1 + 2 * len(H_sel))]
        for name, v in tab.items():
            lines.append(f"| {name} | " + " | ".join(f"{v[str(h)]['model']:.2f} | {v[str(h)]['persistence']:.2f}"
                                                     for h in H_sel) + " |")
        lines.append("")
    open(os.path.join(out, "table.md"), "w").write("\n".join(lines))

    # ---- plots: metric vs horizon ----
    for metric in ("rmse", "r2"):
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.3))
        for ax, s in zip(axs, SLICES):
            for m in res:
                if s not in res[m]:
                    continue
                ys = [res[m][s][str(h)][metric] for h in H]
                st = BASE_STYLE.get(m, {})
                ax.plot(H, ys, marker="o" if m not in BASE_STYLE else None, ms=3, label=m, **st)
            ax.axvspan(1, 15, color="grey", alpha=0.08)
            ax.set(title=f"{s.upper()}", xlabel="horizon (s)", ylabel=metric.upper() + (" (PRBs)" if metric == "rmse" else ""))
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"{metric}_vs_horizon.png"), dpi=110); plt.close(fig)

    # ---- plot: predictions vs actual (first run, first trace, 15 min) ----
    p = preds[0]
    d = dfs[0]
    h1, h3 = H_sel[0], H_sel[-1]
    t, pr, _ = p.predict_df(d, [h1, h3])
    fig, axs = plt.subplots(len(pr), 1, figsize=(16, 3.2 * len(pr)), sharex=True)
    axs = np.atleast_1d(axs)
    n = min(15 * 60, len(d))                     # testbed traces can be shorter than 15 min
    win = (t + h3) < n
    for ax, s in zip(axs, pr):
        y = d[target_col(s)].to_numpy()
        ax.plot(np.arange(n) / 60, y[:n], color=COLORS[s], lw=1.2, label="actual")
        ax.plot((t[win] + h1) / 60, pr[s][win, 0], color="orange", lw=0.9, label=f"predicted {h1} s earlier (made at t−{h1})")
        ax.plot((t[win] + h3) / 60, pr[s][win, 1], color="purple", lw=0.9, alpha=0.8, label=f"predicted {h3} s earlier (made at t−{h3})")
        ax.set_ylabel(f"{s.upper()} PRBs"); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
    axs[-1].set_xlabel("time (min)")
    fig.suptitle(f"{p.name} — {a.split} trace {d['trace_id'].iloc[0]} (each prediction is drawn at the time it is FOR; it was made h s earlier)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "predictions_vs_actual.png"), dpi=110); plt.close(fig)

    print("\n".join(lines[:4 + 6 * 3 + 20]))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
