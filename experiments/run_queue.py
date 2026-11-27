"""Run planned experiments (PLAN.md 3.5) as a parallel job queue. Skips runs that already finished.

Usage:
    python experiments/run_queue.py --exp E1 E2 --seeds 0 --parallel 2
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def jobs_for(exp, seed):
    J = []
    base = ["--horizons", "5", "10", "15", "--seed", str(seed)]
    if exp == "E1":       # PreDP vs neural baselines (same data, all-horizon mode)
        for m in ["predp", "lstm", "gru", "transformer"]:
            J.append((f"E1/{m}_uni_all_s{seed}", ["--model", m] + base))
    elif exp == "E2":     # one 'all' model vs dedicated 3-horizon models
        J.append((f"E2/predp_dedicated_5-10-15_s{seed}",
                  ["--horizon-mode", "dedicated", "--horizons", "5", "10", "15", "--seed", str(seed)]))
        J.append((f"E2/predp_dedicated_7-9-12_s{seed}",
                  ["--horizon-mode", "dedicated", "--horizons", "7", "9", "12", "--seed", str(seed)]))
    elif exp == "E3":     # uni vs multi features
        J.append((f"E3/predp_multi_all_s{seed}", ["--features", "multi"] + base))
    elif exp == "E4":     # lookback
        for L in (20, 60):
            J.append((f"E4/predp_L{L}_s{seed}", ["--lookback", str(L)] + base))
    elif exp == "E5":     # median filter off
        J.append((f"E5/predp_nofilter_s{seed}", ["--median-filter", "0"] + base))
    elif exp == "E8":     # learning curve (train hours)
        for n in (10, 25, 50):
            J.append((f"E8/predp_{n}h_s{seed}", ["--train-traces", str(n)] + base))
    elif exp == "E9":     # level-invariant residual target (fix for the E6 OOD weakness)
        J.append((f"E9/predp_residual_s{seed}", ["--target-mode", "residual"] + base))
    elif exp == "E10":    # reservation targets: average / peak demand over (t, t+h]
        J.append((f"E10/predp_mean_s{seed}", ["--target", "mean"] + base))
        J.append((f"E10/predp_max_s{seed}", ["--target", "max"] + base))
    elif exp == "E11":    # safe-side forecasts: 90th-percentile (pinball loss) of demand at t+h and of the peak
        J.append((f"E11/predp_q90_s{seed}", ["--loss", "quantile", "--quantile", "0.9"] + base))
        J.append((f"E11/predp_max_q90_s{seed}", ["--target", "max", "--loss", "quantile", "--quantile", "0.9"] + base))
    elif exp == "E12":    # final candidates: the winning settings combined (no median filter + 60 s lookback)
        best = ["--median-filter", "0", "--lookback", "60"]
        J.append((f"E12/predp_final_point_s{seed}", best + base))
        J.append((f"E12/predp_final_max_s{seed}", best + ["--target", "max"] + base))
    elif exp == "E13":    # uplink models (separate per slice and direction), final settings, data/raw_v2
        ul = ["--data", "data/raw_v2", "--direction", "ul", "--median-filter", "0", "--lookback", "60"] + base
        J.append((f"E13/predp_ul_point_s{seed}", ul))
        J.append((f"E13/predp_ul_max_s{seed}", ul + ["--target", "max"]))
        J.append((f"E13/predp_ul_q90_s{seed}", ul + ["--loss", "quantile", "--quantile", "0.9"]))
        J.append((f"E13/predp_ul_max_q90_s{seed}", ul + ["--target", "max", "--loss", "quantile", "--quantile", "0.9"]))
    elif exp == "E14":    # 1 vs 2 models: does the other direction's history help? (pair = DL + UL inputs)
        v2 = ["--data", "data/raw_v2", "--median-filter", "0", "--lookback", "60"] + base
        J.append((f"E14/predp_ul_uni_s{seed}", v2 + ["--direction", "ul"]))
        J.append((f"E14/predp_ul_pair_s{seed}", v2 + ["--direction", "ul", "--features", "pair"]))
        J.append((f"E14/predp_dl_uni_s{seed}", v2 + ["--direction", "dl"]))
        J.append((f"E14/predp_dl_pair_s{seed}", v2 + ["--direction", "dl", "--features", "pair"]))
    elif exp == "E15":    # eMBB with a median (MAE) loss: the lab horizon rule scores MAE / p95, MSE models learn the mean
        v2 = ["--data", "data/raw_v2", "--median-filter", "0", "--lookback", "60", "--slices", "embb",
              "--loss", "quantile", "--quantile", "0.5"] + base
        J.append((f"E15/predp_embb_dl_l1_s{seed}", v2 + ["--direction", "dl"]))
        J.append((f"E15/predp_embb_ul_l1_s{seed}", v2 + ["--direction", "ul"]))
    else:
        raise ValueError(exp)
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra args for train.py")
    a = ap.parse_args()

    todo = []
    for e in a.exp:
        for s in a.seeds:
            for name, args in jobs_for(e, s):
                rd = os.path.join(ROOT, "runs", name)
                if not (os.path.exists(os.path.join(rd, "metrics.json")) or os.path.islink(rd)):
                    todo.append((name, [sys.executable, "train.py", "--run-name", name] + args + a.extra))
    print(f"{len(todo)} jobs queued", flush=True)
    running = []
    while todo or running:
        while todo and len(running) < a.parallel:
            name, cmd = todo.pop(0)
            os.makedirs(os.path.join(ROOT, "runs", os.path.dirname(name)), exist_ok=True)
            log = open(os.path.join(ROOT, "runs", name + ".log"), "w")
            running.append((name, subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), time.time()))
            print(f"started  {name}", flush=True)
        for r in running[:]:
            if r[1].poll() is not None:
                print(f"finished {r[0]} rc={r[1].returncode} ({(time.time() - r[2]) / 60:.0f} min)", flush=True)
                running.remove(r)
        time.sleep(10)
    print("queue done")


if __name__ == "__main__":
    main()
