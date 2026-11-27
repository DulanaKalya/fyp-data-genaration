"""Compose one run from per-slice models of compatible runs (used for the E14 per-slice choice).

Example: URLLC from the 'pair' run (DL + UL history), eMBB and mMTC from the 'uni' run:
    python experiments/compose_run.py --name FINAL/predp_ul_final \\
        --slice embb=E13/predp_ul_point_s0 --slice urllc=E14/predp_ul_pair_s0 --slice mmtc=E13/predp_ul_point_s0

Every source must share the architecture, preprocessing, horizons, target, loss and direction. Each
slice keeps its own input list (config "groups"), the scalers are merged (columns shared by two sources
must have identical min/max: same training split), the test metrics are copied per slice, and
confidence is re-calibrated for the composed run. The run's "composed_from" records the sources.
"""
import argparse
import json
import os
import shutil
from datetime import datetime

SAME = ["model", "direction", "horizon_mode", "model_horizons", "lookback", "hidden", "dropout",
        "median_filter", "target_mode", "target", "loss", "quantile", "data"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="new run id under runs/")
    ap.add_argument("--slice", action="append", required=True, metavar="SLICE=RUN",
                    help="take this slice's model from RUN (repeat per slice)")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--data", default=None,
                    help="dataset of the composed run; allows sources trained on different but equivalent datasets "
                         "(e.g. data/raw and data/raw_v2, whose DL columns are bit-identical). The scaler check "
                         "still verifies that shared columns saw the same training data.")
    a = ap.parse_args()

    picks = dict(x.split("=", 1) for x in a.slice)
    # older runs were saved before these settings existed: missing keys mean the default
    defaults = {"direction": "dl", "target": "point", "loss": "mse", "quantile": None, "target_mode": "level"}
    cfgs = {s: {**defaults, **json.load(open(os.path.join(a.runs, r, "config.json")))} for s, r in picks.items()}
    ref = next(iter(cfgs.values()))
    same = [k for k in SAME if not (a.data and k == "data")]
    for s, c in cfgs.items():
        diff = [k for k in same if c.get(k) != ref.get(k)]
        if diff:
            raise SystemExit(f"{picks[s]} differs from the other sources in {diff}; models can't be combined")
        if s not in c["groups"]:
            raise SystemExit(f"{picks[s]} has no '{s}' model (groups: {list(c['groups'])})")

    out = os.path.join(a.runs, a.name)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; choose another --name")
    os.makedirs(out)
    mins, maxs, metrics, cost, logs = {}, {}, {}, {}, {}
    for s, r in picks.items():
        src = os.path.join(a.runs, r)
        shutil.copy2(os.path.join(src, f"model_{s}.pt"), os.path.join(out, f"model_{s}.pt"))
        sc = json.load(open(os.path.join(src, "scaler.json")))
        for col in cfgs[s]["groups"][s]["features"] + cfgs[s]["groups"][s]["targets"]:
            for dst, key in ((mins, "mins"), (maxs, "maxs")):
                v = sc[key][col]
                if col in dst and abs(dst[col] - v) > 1e-6:
                    raise SystemExit(f"scaler mismatch for {col} ({dst[col]} vs {v}); sources used different data")
                dst[col] = v
        m = json.load(open(os.path.join(src, "metrics.json")))
        metrics[s] = m["test"][s]
        cost[s] = m["cost"][s]
        tl = os.path.join(src, "train_log.json")
        if os.path.exists(tl):
            logs[s] = json.load(open(tl)).get(s)

    feats = {cfgs[s]["groups"][s]["features"] == [cfgs[s]["groups"][s]["targets"][0]] for s in picks}
    cfg = {k: ref[k] for k in ref if k not in ("run_id", "groups", "features", "seed", "created", "init_from")}
    if a.data:
        cfg["data"] = a.data
    cfg.update({"run_id": a.name, "features": "uni" if feats == {True} else "mixed",
                "groups": {s: cfgs[s]["groups"][s] for s in picks},
                "seed": {s: cfgs[s].get("seed") for s in picks},
                "composed_from": picks, "created": datetime.now().isoformat(timespec="seconds")})
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    json.dump({"mins": mins, "maxs": maxs}, open(os.path.join(out, "scaler.json"), "w"), indent=1)
    json.dump({"test": metrics, "cost": cost}, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    json.dump(logs, open(os.path.join(out, "train_log.json"), "w"))
    if not a.no_calibrate:
        from confidence import calibrate_run
        calibrate_run(out, data=cfg["data"], quiet=True)
    print(f"Composed {out}:")
    for s, r in picks.items():
        print(f"  {s:6s} <- {r}   inputs {cfgs[s]['groups'][s]['features']}")


if __name__ == "__main__":
    import sys
    # run from the repo root on sys.path, not experiments/ (its queue.py would shadow the stdlib module torch needs)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path = [x for x in sys.path if os.path.abspath(x or ".") != here]
    sys.path.insert(0, os.path.dirname(here))
    main()
