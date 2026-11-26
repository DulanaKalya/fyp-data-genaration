"""Small PreDP grid search (PLAN.md 4-day schedule, Day 1 overnight).

hidden {32,64,128} x lr {5e-4,1e-3,2e-3}, one slice, 25 h train subset, 1 seed, runs in parallel.
Usage: python experiments/grid_search.py --slice urllc --parallel 3
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="urllc")
    ap.add_argument("--train-traces", type=int, default=25)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--lr", type=float, nargs="+", default=[5e-4, 1e-3, 2e-3])
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    a = ap.parse_args()

    out_dir = os.path.join(ROOT, "runs", "grid")
    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for h, lr in itertools.product(a.hidden, a.lr):
        name = f"grid/{a.slice}_h{h}_lr{lr:g}"
        if os.path.exists(os.path.join(ROOT, "runs", name, "metrics.json")):
            continue
        cmd = [sys.executable, "train.py", "--horizons", "5", "10", "15", "--slices", a.slice,
               "--hidden", str(h), "--lr", str(lr), "--train-traces", str(a.train_traces),
               "--max-epochs", str(a.max_epochs), "--patience", str(a.patience), "--run-name", name]
        jobs.append((name, cmd))

    running = []
    while jobs or running:
        while jobs and len(running) < a.parallel:
            name, cmd = jobs.pop(0)
            log = open(os.path.join(ROOT, "runs", name.replace("/", "_") + ".log"), "w")
            running.append((name, subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)))
            print("started", name, flush=True)
        for r in running[:]:
            if r[1].poll() is not None:
                print("finished", r[0], "rc", r[1].returncode, flush=True)
                running.remove(r)
        time.sleep(5)

    rows = []
    for d in sorted(os.listdir(out_dir)):
        mp = os.path.join(out_dir, d, "metrics.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp)); c = json.load(open(os.path.join(out_dir, d, "config.json")))
        sl = list(m["test"])[0]
        cost = list(m["cost"].values())[0]
        rows.append({"run": d, "hidden": c["hidden"], "lr": c["lr"], "val_loss": cost["best_val_loss"],
                     "epochs": cost["epochs"],
                     **{f"r2_{h}": m["test"][sl][str(h)]["model_prb"]["r2"] for h in (5, 10, 15)}})
    rows.sort(key=lambda r: r["val_loss"])
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"{'run':28s} {'val_loss':>9s} {'ep':>4s} {'R2@5':>6s} {'R2@10':>6s} {'R2@15':>6s}")
    for r in rows:
        print(f"{r['run']:28s} {r['val_loss']:9.5f} {r['epochs']:4d} {r['r2_5']:6.3f} {r['r2_10']:6.3f} {r['r2_15']:6.3f}")
    print("Best (by validation loss):", rows[0]["run"] if rows else None)


if __name__ == "__main__":
    main()
