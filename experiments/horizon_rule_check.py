"""Reproduce the lab's horizon-selection rule (Stage 3) on our models, before handing them over.

Rule (as applied by the lab pipeline, 2026-11-27), per candidate horizon H, on the 18 val traces:
every slice/direction group must (1) beat persistence on pooled MAE, (2) beat persistence on the
p95 absolute error, and (3) win (lower MAE than persistence) on at least 10 of the 18 traces.

Example: final models, with the eMBB slice taken from other runs (e.g. median-loss E15 models):
    python experiments/horizon_rule_check.py --dl FINAL/predp_dl_point --ul E13/predp_ul_point_s0 \\
        --embb-dl E15/predp_embb_dl_l1_s0 --embb-ul E15/predp_embb_ul_l1_s0
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predictor import Predictor  # noqa: E402
from prep.data import load_split, target_col  # noqa: E402

H = [4, 5, 6, 7, 8, 9, 10, 12]
SLICES = ["embb", "urllc", "mmtc"]


def group_scores(P, s, direction, dfs, lookback_max):
    """Per trace: model and persistence predictions + truth for slice s (anchors shared by every run)."""
    out = []
    for df in dfs:
        an = np.arange(lookback_max - 1, len(df) - 15)
        _, pr, tr = P.predict_df(df, H, anchors=an)
        y = df[target_col(s, direction)].to_numpy(np.float64)
        out.append((pr[s], np.repeat(y[an][:, None], len(H), 1), tr[s]))
    return out


def evaluate_group(per_trace, min_wins):
    res = {}
    M, Pz, T = (np.concatenate([x[i] for x in per_trace]) for i in range(3))
    for k, h in enumerate(H):
        em, ep = np.abs(M[:, k] - T[:, k]), np.abs(Pz[:, k] - T[:, k])
        wins = int(sum(np.abs(x[0][:, k] - x[2][:, k]).mean() < np.abs(x[1][:, k] - x[2][:, k]).mean()
                       for x in per_trace))
        r = {"mae_ratio": float(em.mean() / ep.mean()), "p95_ratio": float(np.percentile(em, 95) / np.percentile(ep, 95)),
             "wins": wins, "n_traces": len(per_trace)}
        r["pass"] = r["mae_ratio"] < 1 and r["p95_ratio"] < 1 and wins >= min_wins
        res[str(h)] = r
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dl", default="FINAL/predp_dl_point")
    ap.add_argument("--ul", default="E13/predp_ul_point_s0")
    ap.add_argument("--embb-dl", default=None, help="run for the eMBB DL group (default: --dl)")
    ap.add_argument("--embb-ul", default=None, help="run for the eMBB UL group (default: --ul)")
    ap.add_argument("--data", default="data/raw_v2")
    ap.add_argument("--split", default="val")
    ap.add_argument("--min-wins", type=int, default=10)
    ap.add_argument("--out", default=None, help="write the result JSON here")
    a = ap.parse_args()

    runs = {("embb", "dl"): a.embb_dl or a.dl, ("embb", "ul"): a.embb_ul or a.ul}
    for s in ("urllc", "mmtc"):
        runs[(s, "dl")], runs[(s, "ul")] = a.dl, a.ul
    preds = {r: Predictor(os.path.join("runs", r)) for r in set(runs.values())}
    for (s, d), r in runs.items():
        if preds[r].direction != d or s not in preds[r].slices:
            raise SystemExit(f"{r} can't serve {s} {d.upper()} (direction {preds[r].direction}, slices {preds[r].slices})")
    lb = max(p.L for p in preds.values())
    dfs = load_split(a.data, a.split)
    table = {f"{s}_{d}": {"run": r, **evaluate_group(group_scores(preds[r], s, d, dfs, lb), a.min_wins)}
             for (s, d), r in sorted(runs.items())}
    passing = [h for h in map(str, H) if all(table[g][h]["pass"] for g in table)]

    print(f"Rule on {a.split} ({len(dfs)} traces): MAE ratio < 1, p95 ratio < 1, wins >= {a.min_wins}")
    print(f"{'group':10s} {'run':32s} " + " ".join(f"{h:>15s}" for h in map(str, H)))
    for g, t in table.items():
        cells = [f"{t[h]['mae_ratio']:.3f}/{t[h]['p95_ratio']:.2f}/{t[h]['wins']:2d}{'✓' if t[h]['pass'] else '✗'}"
                 for h in map(str, H)]
        print(f"{g:10s} {t['run'][:32]:32s} " + " ".join(f"{c:>15s}" for c in cells))
    print("cell = MAE ratio / p95 ratio / traces won   ✓ = passes")
    print(f"Horizons passing in all 6 groups: {passing or 'none'}")
    if a.out:
        json.dump({"rule": {"split": a.split, "min_wins": a.min_wins, "horizons": H}, "groups": table,
                   "passing_horizons": passing}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
