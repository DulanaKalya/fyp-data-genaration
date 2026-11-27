"""Train the PreDP Bi-LSTM on the 1 s simulator dataset.

Examples:
    python train.py --horizons 5 10 15                        # 'all' mode: model outputs 1..15 s
    python train.py --horizons 7 9 12 --horizon-mode dedicated
    python train.py --features multi --horizons 5 10 15
    python train.py --horizons 5 10 15 --train-traces 10 --max-epochs 3   # quick smoke test
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn

from prep.data import SLICES, load_manifest, load_split, clean, MinMaxScaler, feature_columns, target_col, has_direction
from prep.windows import WindowSet
from models.predp import build_model
from models.metrics import regression_metrics

MAX_H = 15
PROGRESS_LOGS = {}   # per-group epoch logs so far (for the live UI monitor)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--horizons", type=int, nargs=3, default=[5, 10, 15], help="H1 H2 H3 (1..15 s)")
    ap.add_argument("--horizon-mode", choices=["all", "dedicated"], default="all")
    ap.add_argument("--features", choices=["uni", "multi", "pair"], default="uni",
                    help="pair = DL + UL history of the slice (needs a sim 2.0 dataset)")
    ap.add_argument("--direction", choices=["dl", "ul"], default="dl",
                    help="dl = downlink requested PRBs (original) | ul = uplink (needs a sim 2.0 dataset, e.g. data/raw_v2)")
    ap.add_argument("--slices", nargs="+", default=SLICES, help="uni mode: which slice models to train")
    ap.add_argument("--model", default="predp", choices=["predp", "lstm", "gru", "transformer"])
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--median-filter", type=int, default=1)
    ap.add_argument("--target-mode", choices=["level", "residual"], default="level",
                    help="residual (E9): predict the change from the last value, inputs relative to it")
    ap.add_argument("--target", choices=["point", "mean", "max"], default="point",
                    help="point = demand at t+h (PreNS) | mean / max = average / peak demand over (t, t+h]")
    ap.add_argument("--loss", choices=["mse", "quantile"], default="mse")
    ap.add_argument("--quantile", type=float, default=0.9, help="q for --loss quantile (0.9 = safe side)")
    ap.add_argument("--train-traces", type=int, default=None, help="use only the first N train traces")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--progress-file", default=None, help="JSON progress for the UI")
    ap.add_argument("--init-from", default=None,
                    help="fine-tune (E7): start from this run's weights; reuses its scaler and model settings")
    a = ap.parse_args()
    if a.init_from:     # architecture / preprocessing must match the base run
        base = json.load(open(os.path.join(a.init_from, "config.json")))
        for k in ("model", "features", "horizon_mode", "lookback", "hidden", "dropout", "median_filter"):
            setattr(a, k, base[k])
        a.target_mode = base.get("target_mode", "level")
        a.direction = base.get("direction", "dl")
        a.target = base.get("target", "point")
        a.loss, a.quantile = base.get("loss", "mse"), base.get("quantile", 0.9)
        if base["horizon_mode"] == "dedicated":
            a.horizons = base["horizons_selected"]
        if a.features == "uni":
            a.slices = [s for s in a.slices if s in base["groups"]]
    h = sorted(a.horizons)
    if not (1 <= h[0] < h[1] < h[2] <= MAX_H):
        ap.error("horizons must satisfy 1 <= H1 < H2 < H3 <= 15")
    a.horizons = h
    return a


def prepare(dfs, feat_cols, tgt_cols, scaler, median_filter):
    Xs, Ys = [], []
    for d in dfs:
        Xs.append(scaler.transform(clean(d, feat_cols, median_filter=bool(median_filter)), feat_cols))
        Ys.append(scaler.transform(clean(d, tgt_cols, median_filter=False), tgt_cols))   # targets: raw
    return Xs, Ys


@torch.no_grad()
def predict(model, ws, bs=1024):
    model.eval()
    out = [model(xb).cpu() for xb, _ in ws.batches(bs)]
    return torch.cat(out).numpy()


LOSS = torch.nn.functional.mse_loss     # replaced by the training loss in train_group


@torch.no_grad()
def eval_loss(model, ws, bs=1024):
    model.eval()
    tot, n = 0.0, 0
    for xb, yb in ws.batches(bs):
        tot += (LOSS(model(xb), yb.flatten(1)) * yb.numel()).item(); n += yb.numel()
    return tot / n


def write_progress(path, **kw):
    if path:
        with open(path, "w") as f:
            json.dump(kw, f)


def train_group(name, feat_cols, tgt_cols, data, scaler, a, model_h, run_dir, progress):
    dev = a.device
    sets = {}
    for split in ("train", "val", "test"):
        Xs, Ys = prepare(data[split], feat_cols, tgt_cols, scaler, a.median_filter)
        sets[split] = WindowSet(Xs, Ys, a.lookback, model_h, device=dev, target=a.target)
    n_t, K = len(tgt_cols), len(model_h)
    torch.manual_seed(a.seed)
    ridx = [feat_cols.index(c) for c in tgt_cols] if a.target_mode == "residual" else None
    model = build_model(a.model, len(feat_cols), n_t * K, residual_idx=ridx,
                        hidden=a.hidden, dropout=a.dropout).to(dev)
    if a.init_from:
        model.load_state_dict(torch.load(os.path.join(a.init_from, f"model_{name}.pt"), map_location=dev,
                                         weights_only=True))
        print(f"[{name}] initialised from {a.init_from}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    global LOSS
    if a.loss == "quantile":       # pinball loss: over-prediction costs (1-q), under-prediction costs q
        q = a.quantile
        def lossf(p, y):
            e = y - p
            return torch.maximum(q * e, (q - 1) * e).mean()
    else:
        lossf = nn.MSELoss()
    LOSS = lossf
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    best, best_state, bad, log = float("inf"), None, 0, []
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    print(f"[{name}] train windows={len(sets['train']):,} val={len(sets['val']):,} "
          f"test={len(sets['test']):,} params={sum(p.numel() for p in model.parameters()):,}")
    for ep in range(1, a.max_epochs + 1):
        model.train(); te = time.time(); tl, nb = 0.0, 0
        for xb, yb in sets["train"].batches(a.batch_size, shuffle=True, generator=gen):
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb.flatten(1))
            loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        vl = eval_loss(model, sets["val"])
        log.append({"epoch": ep, "train_loss": tl / nb, "val_loss": vl, "sec": time.time() - te})
        improved = vl < best - 1e-7
        if improved:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"[{name}] ep {ep:3d} train {tl / nb:.5f} val {vl:.5f} {'*' if improved else ' '} "
              f"({time.time() - te:.1f}s)", flush=True)
        PROGRESS_LOGS[name] = log
        write_progress(progress, group=name, epoch=ep, max_epochs=a.max_epochs, log=log, logs=PROGRESS_LOGS,
                       status="training")
        if bad >= a.patience:
            break
    train_min = (time.time() - t0) / 60
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(run_dir, f"model_{name}.pt"))

    # ---- test metrics: model vs persistence, per target slice and horizon ----
    P = predict(model, sets["test"]).reshape(-1, n_t, K)
    ws = sets["test"]
    Yt = ws.targets(ws.idx).cpu().numpy()                                       # (N, n_t, K)
    Ylast = ws.Y[ws.idx].cpu().numpy()                                          # (N, n_t)
    metrics = {}
    for j, col in enumerate(tgt_cols):
        sl = col.split("_")[0]
        metrics[sl] = {}
        for k, h in enumerate(model_h):
            y, p, pp = Yt[:, j, k], P[:, j, k], Ylast[:, j]
            metrics[sl][str(h)] = {
                "model_norm": regression_metrics(y, p),
                "model_prb": regression_metrics(scaler.inverse(y, col), scaler.inverse(p, col)),
                "persistence_prb": regression_metrics(scaler.inverse(y, col), scaler.inverse(pp, col)),
            }
    # cost figures (PreNS Table III style)
    model.eval(); x1 = ws.get(torch.arange(1, device=dev))[0]
    with torch.no_grad():
        for _ in range(20): model(x1)
        if dev == "cuda": torch.cuda.synchronize()
        ti = time.time()
        for _ in range(200): model(x1)
        if dev == "cuda": torch.cuda.synchronize()
    cost = {"params": sum(p.numel() for p in model.parameters()),
            "train_min": round(train_min, 2), "epochs": len(log),
            "best_val_loss": best,
            "infer_ms_batch1": round((time.time() - ti) / 200 * 1e3, 3),
            "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if dev == "cuda" else None}
    return metrics, cost, log


def main():
    a = parse()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model_h = list(range(1, MAX_H + 1)) if a.horizon_mode == "all" else a.horizons
    run_id = a.run_name or datetime.now().strftime("%Y%m%d-%H%M%S") + \
        f"_{a.model}_{a.features}_{'' if a.direction == 'dl' else 'ul_'}{a.horizon_mode}_{'-'.join(map(str, a.horizons))}"
    run_dir = os.path.join(a.runs, run_id)
    os.makedirs(run_dir, exist_ok=True)
    progress = a.progress_file or os.path.join(run_dir, "progress.json")

    if not has_direction(a.data, a.direction):
        raise SystemExit(f"dataset {a.data} has no {a.direction.upper()} columns; use a sim 2.0 dataset such as data/raw_v2")
    manifest = load_manifest(a.data)
    data = {"train": load_split(a.data, "train", a.train_traces, manifest),
            "val": load_split(a.data, "val", None, manifest),
            "test": load_split(a.data, "test", None, manifest)}

    if a.features in ("uni", "pair"):
        if a.features == "pair" and not has_direction(a.data, "ul"):
            raise SystemExit("--features pair needs DL and UL columns (sim 2.0 dataset)")
        groups = {s: (feature_columns(a.features, s, a.direction), [target_col(s, a.direction)]) for s in a.slices}
    else:
        groups = {"joint": (feature_columns("multi", direction=a.direction), [target_col(s, a.direction) for s in SLICES])}
    all_cols = sorted({c for f, t in groups.values() for c in f + t})
    if a.init_from:      # keep the base scaler so the pretrained weights see the same input scale
        scaler = MinMaxScaler.from_dict(json.load(open(os.path.join(a.init_from, "scaler.json"))))
    else:
        scaler = MinMaxScaler().fit(data["train"], all_cols)

    config = {"run_id": run_id, "model": a.model, "features": a.features, "direction": a.direction, "groups":
              {g: {"features": f, "targets": t} for g, (f, t) in groups.items()},
              "horizon_mode": a.horizon_mode, "horizons_selected": a.horizons, "model_horizons": model_h,
              "lookback": a.lookback, "hidden": a.hidden, "dropout": a.dropout, "lr": a.lr,
              "batch_size": a.batch_size, "max_epochs": a.max_epochs, "patience": a.patience,
              "median_filter": a.median_filter, "target_mode": a.target_mode, "target": a.target,
              "loss": a.loss, "quantile": a.quantile if a.loss == "quantile" else None, "seed": a.seed, "data": a.data,
              "sim_version": manifest["sim_version"],
              "train_traces": len(data["train"]), "train_hours": sum(len(d) for d in data["train"]) / 3600,
              "init_from": a.init_from,
              "created": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    with open(os.path.join(run_dir, "scaler.json"), "w") as f:
        json.dump(scaler.to_dict(), f, indent=1)

    all_metrics, all_cost, all_logs = {}, {}, {}
    for g, (fc, tc) in groups.items():
        m, c, log = train_group(g, fc, tc, data, scaler, a, model_h, run_dir, progress)
        all_metrics.update(m); all_cost[g] = c; all_logs[g] = log
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"test": all_metrics, "cost": all_cost}, f, indent=1)
    with open(os.path.join(run_dir, "train_log.json"), "w") as f:
        json.dump(all_logs, f)
    try:   # confidence v1: fit tau on validation, check calibration on test
        from confidence import calibrate_run
        calibrate_run(run_dir, data=a.data, quiet=True)
        print("Confidence calibrated (confidence.json)")
    except Exception as e:  # noqa: BLE001
        print(f"Confidence calibration skipped: {e}")
    write_progress(progress, status="done", run_id=run_id)

    print(f"\nRun saved: {run_dir}")
    print(f"{'slice':6s} {'h':>3s} | {'RMSE(PRB)':>9s} {'MAE':>6s} {'R2':>6s} | persistence RMSE  R2")
    for sl, mh in all_metrics.items():
        for h in a.horizons:
            r = mh[str(h)]
            print(f"{sl:6s} {h:3d} | {r['model_prb']['rmse']:9.2f} {r['model_prb']['mae']:6.2f} "
                  f"{r['model_prb']['r2']:6.3f} | {r['persistence_prb']['rmse']:9.2f} "
                  f"{r['persistence_prb']['r2']:6.3f}")


if __name__ == "__main__":
    main()
