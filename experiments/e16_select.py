"""Apply the pre-registered E16 rule (reports/eval/E16_selection/RULE.md) to the eMBB quantile sweep.

For each direction and candidate q (seed 0 by default, or --seeds 0 1 2 for the confirmation):
  1. lab horizon rule on the eMBB group at 5-12 s (MAE, p95, >= 10/18 traces won vs persistence),
  2. shortfall = mean(max(actual - forecast, 0)) <= persistence at 5, 8 and 12 s.
Pick the lowest qualifying q per direction. Output: reports/eval/E16_selection/selection.{md,json}
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predictor import Predictor  # noqa: E402
from prep.data import load_split  # noqa: E402
from experiments.horizon_rule_check import H, evaluate_group, group_scores  # noqa: E402

RULE_H = ["5", "6", "7", "8", "9", "10", "12"]
SHORT_H = [5, 8, 12]
OUT = "reports/eval/E16_selection"


def shortfall(per_trace, h):
    k = H.index(h)
    M, Pz, T = (np.concatenate([x[i][:, k] for x in per_trace]) for i in range(3))
    return float(np.maximum(T - M, 0).mean()), float(np.maximum(T - Pz, 0).mean())


def check(run, d, dfs):
    P = Predictor(os.path.join("runs", run))
    pt = group_scores(P, "embb", d, dfs, 60)
    rule = evaluate_group(pt, 10)
    sf = {str(h): dict(zip(("model", "persistence"), shortfall(pt, h))) for h in SHORT_H}
    ok1 = all(rule[h]["pass"] for h in RULE_H)
    ok2 = all(v["model"] <= v["persistence"] for v in sf.values())
    return {"run": run, "rule_pass": ok1, "shortfall_ok": ok2, "qualifies": ok1 and ok2,
            "rule": {h: rule[h] for h in RULE_H}, "shortfall": sf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--qs", nargs="+", default=["q55", "q60", "q65"])
    a = ap.parse_args()
    rule_hash = hashlib.sha256(open(os.path.join(OUT, "RULE.md"), "rb").read()).hexdigest()
    reg = open(os.path.join(OUT, "RULE.sha256")).read().split()[0]
    if rule_hash != reg:
        raise SystemExit("RULE.md changed after registration")
    dfs = load_split("data/raw_v2", "val")
    res, L = {}, [f"# E16 selection (rule sha256 {rule_hash[:12]}…, seeds {a.seeds})", ""]
    for d in ("dl", "ul"):
        res[d] = {}
        L += [f"## eMBB {d.upper()}", "", "| q | seed | rule 5–12 s | MAE ratio 5 / 8 / 12 s | wins 5 / 8 / 12 | "
              "shortfall model vs persistence 5 / 8 / 12 s | qualifies |", "|---|---|---|---|---|---|---|"]
        for q in a.qs:
            for s in a.seeds:
                run = f"E16/predp_embb_{d}_{q}_s{s}"
                if not os.path.exists(os.path.join("runs", run, "metrics.json")):
                    continue
                r = check(run, d, dfs)
                res[d].setdefault(q, {})[str(s)] = r
                ru, sf = r["rule"], r["shortfall"]
                L.append(f"| 0.{q[1:]} | {s} | {'✅' if r['rule_pass'] else '❌'} | "
                         + " / ".join(f"{ru[h]['mae_ratio']:.3f}" for h in ("5", "8", "12")) + " | "
                         + " / ".join(str(ru[h]["wins"]) for h in ("5", "8", "12")) + " | "
                         + " / ".join(f"{sf[h]['model']:.2f} vs {sf[h]['persistence']:.2f}" for h in ("5", "8", "12"))
                         + f" | {'✅' if r['qualifies'] else '❌'} |")
        ok = [q for q in a.qs if q in res[d] and res[d][q] and all(v["qualifies"] for v in res[d][q].values())]
        res[d]["choice"] = ok[0] if ok else None
        L += ["", f"**Choice for {d.upper()}: " + (f"q = 0.{ok[0][1:]}**" if ok else "none qualifies → keep v2 + rApp headroom**"), ""]
    os.makedirs(OUT, exist_ok=True)
    tag = "selection" if a.seeds == [0] else "confirmation"
    open(os.path.join(OUT, f"{tag}.md"), "w").write("\n".join(L))
    json.dump(res, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
