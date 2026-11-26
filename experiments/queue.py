"""Run planned experiments (PLAN.md 3.5) as a parallel job queue. Skips runs that already finished.

Usage:
    python experiments/queue.py --exp E1 E2 --seeds 0 --parallel 2
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
