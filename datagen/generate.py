"""Generate the full dataset.

Usage:
    python -m datagen.generate                       # all splits from configs/sim.yaml
    python -m datagen.generate --only train:2 --out data/sample
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import yaml

from .simulator import simulate_trace

SPLIT_OFFSET = {"train": 0, "val": 100_000, "test": 200_000, "ood": 300_000}


def _job(args):
    cfg, split, i, out = args
    tid = f"{split}_{i:03d}"
    seed = cfg["base_seed"] + SPLIT_OFFSET[split] + i
    df, meta = simulate_trace(cfg, cfg["split_profile"][split], seed, tid)
    meta["split"] = split
    df.to_parquet(os.path.join(out, f"trace_{tid}.parquet"), index=False)
    with open(os.path.join(out, f"trace_{tid}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return tid, split, seed, cfg["split_profile"][split]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sim.yaml")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--only", default=None, help="e.g. train:2,test:1 to generate a subset")
    ap.add_argument("--trace-hours", type=float, default=None, help="override trace length")
    ap.add_argument("--base-seed", type=int, default=None, help="override base seed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if a.trace_hours:
        cfg["trace_hours"] = a.trace_hours
    if a.base_seed is not None:
        cfg["base_seed"] = a.base_seed
    splits = cfg["splits"]
    if a.only:
        splits = {k: int(v) for k, v in (x.split(":") for x in a.only.split(","))}
    os.makedirs(a.out, exist_ok=True)

    jobs = [(cfg, s, i, a.out) for s, cnt in splits.items() for i in range(cnt)]
    t0 = time.time()
    manifest = {"sim_version": cfg["sim_version"], "config": cfg,
                "splits": {s: [] for s in splits}, "traces": {}}
    with ProcessPoolExecutor(a.workers) as ex:
        for done, (tid, split, seed, prof) in enumerate(ex.map(_job, jobs), 1):
            manifest["splits"][split].append(tid)
            manifest["traces"][tid] = {"split": split, "seed": seed, "profile": prof}
            if done % 10 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} traces  ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"Done: {len(jobs)} traces -> {a.out}  in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
