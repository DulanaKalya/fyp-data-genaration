"""Compare our results with PreNS (Wu et al., IEEE TCCN 2026) Table III.

Reads an evaluate.py report (default reports/eval/E1_prens) and writes
reports/prens_comparison/{comparison.md, comparison.json, r2_comparison.png}.

Units: PreNS reports RMSE/MAE on normalized demand. Their numbers match z-score units (RMSE ≈ sqrt(1-R²)),
so ours are converted to z-units with the test-target std:  RMSE_z = sqrt(1 - R²),  MAE_z = MAE / std,
std = RMSE / sqrt(1 - R²). R² is unit-free and is the primary comparison.
"""
import argparse
import json
import os

import numpy as np

# PreNS Table III (prediction horizon 200 ms, 100 ms sampling, 500 ms input, OAI testbed, 3 slices)
PRENS = {
    "PreDP (attention Bi-LSTM)": {"key": "predp", "layers": 1, "params": 35844, "train_min": 10.6, "infer_ms": 0.467,
                                  "peak_gpu_mb": 31.09, "rmse": 0.6977, "mae": 0.5141, "r2": 0.5762},
    "UT-LSTM": {"key": "lstm", "layers": 2, "params": 51139, "train_min": 7.6, "infer_ms": 0.977,
                "peak_gpu_mb": 26.57, "rmse": 0.8677, "mae": 0.6261, "r2": 0.4523},
    "UT-GRU": {"key": "gru", "layers": 2, "params": 38403, "train_min": 4.6, "infer_ms": 0.550,
               "peak_gpu_mb": 20.39, "rmse": 0.8234, "mae": 0.5841, "r2": 0.5183},
    "UT-Transformer": {"key": "transformer", "layers": 2, "params": 283067, "train_min": 31.2, "infer_ms": 0.603,
                       "peak_gpu_mb": 54.07, "rmse": 1.0922, "mae": 0.8243, "r2": 0.2813},
    "ARIMA": {"key": "arima", "layers": 1, "params": 13443, "train_min": 2.1, "infer_ms": 2.015,
              "peak_gpu_mb": 18.07, "rmse": 2.2645, "mae": 1.5329, "r2": 0.1263},
}
SLICES = ["embb", "urllc", "mmtc"]


def zunits(m):
    r2 = m["r2"]
    rz = float(np.sqrt(max(1e-12, 1 - r2)))
    std = m["rmse"] / rz
    return {"rmse_z": rz, "mae_z": m["mae"] / std, "r2": r2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/eval/E1_prens")
    ap.add_argument("--out", default="reports/prens_comparison")
    a = ap.parse_args()
    R = json.load(open(os.path.join(a.report, "results.json")))
    os.makedirs(a.out, exist_ok=True)
    H = R["horizons_selected"]

    # map evaluate.py method names -> model keys
    ours = {}
    for name, per in R["metrics"].items():
        if name in ("persistence", "ma10"):
            key = name
        elif name == "arima":
            key = "arima"
            per = R["arima_subset_metrics"]["arima"]
        else:
            key = R["cost"][name]["model"]
        ours[key] = {"name": name, "per": per, "cost": R["cost"].get(name)}
    if "persistence" in ours and "arima_subset_metrics" in R:
        ours["persistence_arima_subset"] = {"name": "persistence", "per": R["arima_subset_metrics"]["persistence"]}

    rows = {}
    for key, o in ours.items():
        rows[key] = {}
        for h in H:
            zs = [zunits(o["per"][s][str(h)]) for s in SLICES if s in o["per"]]
            rows[key][h] = {k: float(np.mean([z[k] for z in zs])) for k in ("rmse_z", "mae_z", "r2")}
            rows[key][h]["per_slice_r2"] = {s: o["per"][s][str(h)]["r2"] for s in SLICES if s in o["per"]}

    # ---------------- markdown ----------------
    L = ["# Comparison with PreNS (Wu et al., IEEE TCCN 2026) — Table III", "",
         "| | **PreNS paper** | **This work** |", "|---|---|---|",
         "| Data | OAI 5G testbed (FlexRAN), 100 RBs | Testbed-matched simulator: n78, 60 MHz, 30 kHz SCS, **162 PRBs**, 3+3+3 UEs |",
         "| Sampling | 100 ms | **1 s** (100 ms simulated, aggregated) |",
         "| Input window | 5 steps = 500 ms | 30 steps = 30 s |",
         "| Prediction horizon | 200 ms | **5 s, 10 s, 15 s** |",
         f"| Test data | not stated | {R['n_windows']:,} windows (18 h, 18 unseen traces) |",
         "| Hardware | RTX 4090 + i9-13900K | RTX 4060 Laptop (shared by up to 6 jobs) |", "",
         "R² is unit-free and the fairest comparison. RMSE/MAE are in z-score (standardized) units; "
         "PreNS values fit that scale (RMSE ≈ √(1−R²)). Our metrics are the mean over the 3 slices.", ""]

    def cell(key, h, k):
        return f"{rows[key][h][k]:.3f}" if key in rows else "—"

    L += ["## Accuracy: PreNS (200 ms) vs this work (5 / 10 / 15 s)", "",
          "| Model | PreNS R² | Our R² @5s | @10s | @15s | PreNS RMSE | Our RMSE @5/10/15 s | PreNS MAE | Our MAE @5/10/15 s |",
          "|---|---|---|---|---|---|---|---|---|"]
    for label, p in PRENS.items():
        k = p["key"]
        if k in rows:
            r2s = [cell(k, h, "r2") for h in H]
            rm = " / ".join(cell(k, h, "rmse_z") for h in H)
            ma = " / ".join(cell(k, h, "mae_z") for h in H)
        else:
            r2s, rm, ma = ["(running)"] * 3, "(running)", "(running)"
        L.append(f"| {label} | {p['r2']:.4f} | {' | '.join(r2s)} | {p['rmse']:.4f} | {rm} | {p['mae']:.4f} | {ma} |")
    if "persistence" in rows:
        L.append("| Persistence ŷ(t+h)=y(t) *(not in PreNS)* | — | " + " | ".join(cell("persistence", h, "r2") for h in H)
                 + " | — | " + " / ".join(cell("persistence", h, "rmse_z") for h in H) + " | — | "
                 + " / ".join(cell("persistence", h, "mae_z") for h in H) + " |")
    if "arima" in rows:
        L.append(f"\nARIMA(2,0,1) uses a random {R['arima_windows']}-window subset; persistence on the same subset: "
                 "R² " + " / ".join(cell("persistence_arima_subset", h, "r2") for h in H) + ".")

    label_of = {p["key"]: lbl for lbl, p in PRENS.items()}
    # same-subset comparison (fair for ARIMA): every method on exactly the ARIMA windows
    sub = {}
    if "arima_subset_metrics" in R:
        for name, per in R["arima_subset_metrics"].items():
            key = name if name in ("persistence", "ma10", "arima") else R["cost"][name]["model"]
            sub[key] = {h: float(np.mean([per[s][str(h)]["r2"] for s in SLICES if s in per])) for h in H}
        L += ["", f"## Same-window comparison ({R['arima_windows']} random test windows, all methods) — mean R²", "",
              "| Model | " + " | ".join(f"{h} s" for h in H) + " |", "|---|" + "---|" * len(H)]
        for key in [p["key"] for p in PRENS.values()] + ["persistence", "ma10"]:
            if key in sub:
                best = {h: max(v[h] for v in sub.values()) for h in H}
                L.append(f"| {label_of.get(key, key)} | " + " | ".join(
                    (f"**{sub[key][h]:.3f}**" if abs(sub[key][h] - best[h]) < 1e-9 else f"{sub[key][h]:.3f}") for h in H) + " |")
    L += ["", "## Ranking of models", "",
          "| Rank | PreNS (R² at 200 ms) | This work, mean R² over 5/10/15 s (same windows) |", "|---|---|---|"]
    pr = sorted(PRENS.items(), key=lambda x: -x[1]["r2"])
    src = sub if sub else {k: {h: rows[k][h]["r2"] for h in H} for k in rows}
    ou = sorted([(k, np.mean([src[k][h] for h in H])) for k in src if k in
                 {p["key"] for p in PRENS.values()}], key=lambda x: -x[1])
    for i in range(len(pr)):
        o = f"{label_of[ou[i][0]]} ({ou[i][1]:.3f})" if i < len(ou) else "—"
        L.append(f"| {i + 1} | {pr[i][0]} ({pr[i][1]['r2']:.3f}) | {o} |")

    L += ["", "## Per-slice R² (this work)", "", "| Model | " + " | ".join(f"{s.upper()} @{h}s" for s in SLICES for h in H) + " |",
          "|---|" + "---|" * (len(SLICES) * len(H))]
    for key in [p["key"] for p in PRENS.values()] + ["persistence"]:
        if key in rows:
            L.append(f"| {label_of.get(key, key)} | " + " | ".join(
                f"{rows[key][h]['per_slice_r2'].get(s, float('nan')):.3f}" for s in SLICES for h in H) + " |")

    L += ["", "## Cost", "", "| Model | PreNS params | Our params (per slice) | PreNS train min | Our train min (3 slices) | "
          "PreNS infer ms | Our infer ms (batch 1) | PreNS peak GPU MB | Our peak GPU MB |", "|---|---|---|---|---|---|---|---|---|"]
    for label, p in PRENS.items():
        c = ours.get(p["key"], {}).get("cost")
        if c:
            L.append(f"| {label} | {p['params']:,} | {c['params_per_slice_model']:,} | {p['train_min']} | {c['train_min_total']} | "
                     f"{p['infer_ms']} | {c['infer_ms_batch1']} | {p['peak_gpu_mb']} | {c['peak_gpu_mb']} |")
        else:
            L.append(f"| {label} | {p['params']:,} | — | {p['train_min']} | — | {p['infer_ms']} | — | {p['peak_gpu_mb']} | — |")
    L += ["", "Our training times are on 84 h of data with 3–6 jobs sharing one laptop GPU, so they are not comparable to "
          "PreNS's single-job RTX 4090 times. Peak GPU memory includes the whole dataset kept on the GPU."]
    open(os.path.join(a.out, "comparison.md"), "w").write("\n".join(L))
    cost_by_key = {k: o["cost"] for k, o in ours.items() if o.get("cost")}
    json.dump({"prens": PRENS, "ours": rows, "same_window": {k: {str(h): v for h, v in d.items()} for k, d in sub.items()},
               "cost": cost_by_key, "horizons": H, "source_report": a.report,
               "n_windows": R["n_windows"], "arima_windows": R["arima_windows"],
               "runs": {k: o["name"] for k, o in ours.items()},
               "setup": {"prens": {"data": "OAI 5G testbed (FlexRAN), 100 RBs", "sampling": "100 ms",
                                   "input": "5 steps = 500 ms", "horizon": "200 ms", "hardware": "RTX 4090 + i9-13900K"},
                         "ours": {"data": "Testbed-matched simulator: n78, 60 MHz, 30 kHz SCS, 162 PRBs, 3+3+3 UEs",
                                  "sampling": "1 s (100 ms simulated, aggregated)", "input": "30 steps = 30 s",
                                  "horizon": " / ".join(f"{h} s" for h in H),
                                  "hardware": "RTX 4060 Laptop (shared by several jobs)"}}},
              open(os.path.join(a.out, "comparison.json"), "w"), indent=1)

    # ---------------- plot ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [l for l, p in PRENS.items()]
    x = np.arange(len(labels))
    w = 0.19
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.bar(x - 1.5 * w, [PRENS[l]["r2"] for l in labels], w, label="PreNS paper (200 ms)", color="#94a3b8")
    cols = ["#f59e0b", "#8b5cf6", "#0891b2"]
    for i, h in enumerate(H):
        vals = [rows[PRENS[l]["key"]][h]["r2"] if PRENS[l]["key"] in rows else np.nan for l in labels]
        ax.bar(x + (i - 0.5) * w, vals, w, label=f"this work, {h} s", color=cols[i])
    if "persistence" in rows:
        for i, h in enumerate(H):
            ax.axhline(rows["persistence"][h]["r2"], color=cols[i], ls=":", lw=1)
    ax.set_xticks(x, labels, fontsize=9); ax.set_ylabel("R² (mean over slices)"); ax.axhline(0, color="k", lw=0.6)
    ax.set_title("PreNS Table III vs this work (dotted = persistence baseline at each horizon)")
    ax.legend(fontsize=8, ncol=4); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "r2_comparison.png"), dpi=120); plt.close(fig)
    print("\n".join(L))
    print(f"\nSaved -> {a.out}")


if __name__ == "__main__":
    main()
