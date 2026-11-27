"""E14 decision (PLAN_UL.md): per slice, 1 model (DL + UL history, 'pair') or 2 separate models ('uni')?

Rule, fixed before the results were seen. Choose 'pair' (1 model) for a slice only if ALL hold:
  A. it improves skill vs persistence by more than 2x the seed spread at some horizon (5/10/15 s)
     in at least one direction,
  B. it is not worse (by more than 2x the spread) at any horizon in the other direction,
  C. it is not worse (by more than 2x the spread) at any horizon on the OOD split, in either direction.
Otherwise keep 2 separate models (the default).

Input:  reports/eval/E14_{ul,dl}[_ood]/results.json   (3 seeds of uni and pair per direction)
Output: reports/eval/E14_decision/{decision.md, decision.json}
"""
import argparse
import json
import os

import numpy as np

SLICES = ["embb", "urllc", "mmtc"]
H = ["5", "10", "15"]
MIN_SPREAD = 0.002          # floor: skill differences below 0.2 % are noise whatever the seed std says


def load(tag):
    f = os.path.join("reports/eval", tag, "results.json")
    return json.load(open(f))["metrics"] if os.path.exists(f) else None


def stats(M, d, variant, s, h):
    runs = [k for k in M if k.startswith(f"E14/predp_{d}_{variant}_s")]
    v = np.array([M[k][s][h]["skill_vs_persistence"] for k in runs])
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def compare(M, d, s):
    """Per horizon: mean skill of uni and pair, gain (pair - uni) and the noise level 2 x spread."""
    out = {}
    for h in H:
        mu, su, nu = stats(M, d, "uni", s, h)
        mp, sp, npair = stats(M, d, "pair", s, h)
        noise = 2 * max(su, sp, MIN_SPREAD)
        out[h] = {"uni": round(mu, 4), "uni_std": round(su, 4), "pair": round(mp, 4), "pair_std": round(sp, 4),
                  "gain": round(mp - mu, 4), "noise": round(noise, 4), "seeds": [nu, npair],
                  "better": mp - mu > noise, "worse": mu - mp > noise}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/eval/E14_decision")
    a = ap.parse_args()
    test = {d: load(f"E14_{d}") for d in ("ul", "dl")}
    ood = {d: load(f"E14_{d}_ood") for d in ("ul", "dl")}
    if not all(test.values()):
        raise SystemExit("missing reports/eval/E14_ul or E14_dl - run the E14 evaluations first")

    res, L = {}, ["# E14 — one model (DL + UL history) or two separate models?", "",
                  "Skill = RMSE reduction vs persistence. `pair` = the slice's DL and UL history as input; "
                  "`uni` = own direction only. Mean ± std over 3 seeds. A difference counts only if it is "
                  "larger than 2 × the seed spread (min 0.2 %).", ""]
    for s in SLICES:
        r = {"test": {d: compare(test[d], d, s) for d in ("ul", "dl")},
             "ood": {d: compare(ood[d], d, s) for d in ("ul", "dl") if ood[d]}}
        A = {d: any(v["better"] for v in r["test"][d].values()) for d in ("ul", "dl")}
        worse_test = {d: any(v["worse"] for v in r["test"][d].values()) for d in ("ul", "dl")}
        worse_ood = any(v["worse"] for d in r["ood"] for v in r["ood"][d].values())
        cond_a = A["ul"] or A["dl"]
        cond_b = not any(worse_test[d] for d in ("ul", "dl") if not A[d])
        cond_c = not worse_ood
        choice = "1 model (pair)" if (cond_a and cond_b and cond_c) else "2 separate models"
        r.update({"A_gain_somewhere": cond_a, "B_other_direction_not_worse": cond_b, "C_ood_not_worse": cond_c,
                  "decision": choice})
        res[s] = r
        L += [f"## {s.upper()} → **{choice}**", "",
              f"A gain > noise: {'yes' if cond_a else 'no'} · B other direction not worse: {'yes' if cond_b else 'no'}"
              f" · C OOD not worse: {'yes' if cond_c else 'no'}", "",
              "| split | direction | h | uni skill | pair skill | gain | noise (2×spread) |", "|---|---|---|---|---|---|---|"]
        for split in ("test", "ood"):
            for d, per in r[split].items():
                for h, v in per.items():
                    mark = " ✅" if v["better"] else (" ❌" if v["worse"] else "")
                    L.append(f"| {split} | {d.upper()} | {h} s | {100 * v['uni']:+.1f}% ± {100 * v['uni_std']:.1f} | "
                             f"{100 * v['pair']:+.1f}% ± {100 * v['pair_std']:.1f} | {100 * v['gain']:+.1f}%{mark} | "
                             f"{100 * v['noise']:.1f}% |")
        L.append("")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "decision.md"), "w").write("\n".join(L))
    json.dump(res, open(os.path.join(a.out, "decision.json"), "w"), indent=1)
    print("\n".join(L))
    print(f"\nSaved -> {a.out}")


if __name__ == "__main__":
    main()
