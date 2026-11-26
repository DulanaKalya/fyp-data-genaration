"""FastAPI backend for the React UI. Wraps datagen/, train.py, evaluate.py, predictor.py and infer.py.

Run from the project root:
    python3 -m uvicorn api.server:app --host 127.0.0.1 --port 8000
"""
import json
import math
import os
import re
import sys
import time
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from api.jobs import launch, list_jobs, pid_alive  # noqa: E402
from datagen.scenarios import SCENARIOS  # noqa: E402
from models.metrics import regression_metrics  # noqa: E402

SLICES = ["embb", "urllc", "mmtc"]
RUNS = os.path.join(ROOT, "runs")
DATA = os.path.join(ROOT, "data")
REPORTS = os.path.join(ROOT, "reports")
NAME_RE = re.compile(r"^[A-Za-z0-9_\-./]+$")

app = FastAPI(title="Predictive Slice Demand API")


# ---------------------------------------------------------------- helpers
def safe_join(base, rel):
    if not rel or not NAME_RE.match(rel) or ".." in rel.split("/"):
        raise HTTPException(400, f"invalid name: {rel!r}")
    path = os.path.realpath(os.path.join(base, rel))
    if not (path == os.path.realpath(base) or path.startswith(os.path.realpath(base) + os.sep)):
        raise HTTPException(400, "path outside allowed folder")
    return path


def data_dir(data):
    rel = data[len("data/"):] if data.startswith("data/") else data
    d = safe_join(DATA, rel)
    if not os.path.exists(os.path.join(d, "manifest.json")):
        raise HTTPException(404, f"dataset {data} not found")
    return d


def clean_json(x):
    """Replace NaN/inf with None recursively (JSON-safe)."""
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, dict):
        return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, (np.floating,)):
        return clean_json(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def r(v, nd=2):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd)


@lru_cache(maxsize=8)
def _manifest(d, mtime):
    return json.load(open(os.path.join(d, "manifest.json")))


def manifest(d):
    return _manifest(d, os.path.getmtime(os.path.join(d, "manifest.json")))


@lru_cache(maxsize=32)
def _trace(path, mtime):
    return pd.read_parquet(path)


def load_trace(data, tid):
    if data == "upload":
        path = safe_join(os.path.join(DATA, "_uploads"), tid + ".parquet")
    else:
        path = os.path.join(data_dir(data), f"trace_{safe_join(DATA, tid) and tid}.parquet")
    if not os.path.exists(path):
        raise HTTPException(404, f"trace {tid} not found")
    return _trace(path, os.path.getmtime(path))


def segments(df):
    sc = df["scenario"].astype(str).to_numpy()
    t = df["t"].to_numpy()
    out, start = [], 0
    for i in range(1, len(sc) + 1):
        if i == len(sc) or sc[i] != sc[start]:
            out.append({"start": int(t[start]), "end": int(t[i - 1]) + 1, "scenario": sc[start]})
            start = i
    return out


def run_dir(run):
    d = safe_join(RUNS, run)
    if not os.path.exists(os.path.join(d, "config.json")):
        raise HTTPException(404, f"run {run} not found")
    return d


@lru_cache(maxsize=8)
def _predictor(d, mtime):
    from predictor import Predictor
    return Predictor(d, device="cpu")   # keep the GPU for training


def predictor(run):
    d = run_dir(run)
    mf = os.path.join(d, "metrics.json")
    if not os.path.exists(mf):
        raise HTTPException(409, f"run {run} has not finished training")
    return _predictor(d, os.path.getmtime(mf))


def check_h(h):
    h = sorted(h)
    if len(h) != 3 or not (1 <= h[0] < h[1] < h[2] <= 15):
        raise HTTPException(400, "need three horizons 1 <= H1 < H2 < H3 <= 15")
    return h


# ---------------------------------------------------------------- overview
@app.get("/api/health")
def health():
    import torch
    return {"ok": True, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


@app.get("/api/overview")
def overview():
    return {"datasets": len(list_datasets()), "runs": len([x for x in list_runs() if x["status"] == "done"]),
            "running": len([x for x in list_runs() if x["status"] == "running"]),
            "reports": len(list_reports())}


# ---------------------------------------------------------------- datasets
@app.get("/api/datasets")
def list_datasets():
    out = []
    if os.path.isdir(DATA):
        names = []
        for name in sorted(os.listdir(DATA)):          # data/<name> and one level deeper (data/testbed/<name>)
            if name.startswith("_"):
                continue
            names.append(name)
            sub = os.path.join(DATA, name)
            if os.path.isdir(sub) and not os.path.exists(os.path.join(sub, "manifest.json")):
                names += [f"{name}/{n}" for n in sorted(os.listdir(sub))]
        for name in names:
            d = os.path.join(DATA, name)
            if os.path.exists(os.path.join(d, "manifest.json")):
                m = manifest(d)
                h = m["config"]["trace_hours"]
                out.append({"path": f"data/{name}", "sim_version": m["sim_version"], "trace_hours": h,
                            "total_prb": m["config"]["radio"]["total_prb"],
                            "splits": {s: len(v) for s, v in m["splits"].items()},
                            "hours": {s: len(v) * h for s, v in m["splits"].items()}})
    return out


def report_dir_for(data):
    name = data.rstrip("/").split("/")[-1]
    return os.path.join(REPORTS, "dataset") if name == "raw" else os.path.join(REPORTS, "dataset", name)


@app.get("/api/datasets/summary")
def dataset_summary(data: str = "data/raw"):
    data_dir(data)
    f = os.path.join(report_dir_for(data), "summary.json")
    if not os.path.exists(f):
        return {"available": False}
    s = json.load(open(f))
    s["available"] = True
    return clean_json(s)


@app.get("/api/datasets/traces")
def dataset_traces(data: str = "data/raw"):
    m = manifest(data_dir(data))
    return m["splits"]


@app.get("/api/trace")
def get_trace(data: str, id: str, start: int = 0, end: Optional[int] = None):
    df = load_trace(data, id)
    end = end or int(df.t.max()) + 1
    d = df[(df.t >= start) & (df.t < end)]
    out = {"t": d.t.astype(int).tolist(), "length": int(len(df)), "segments": segments(d),
           "total_req": [r(v) for v in d["total_req_prb_mean"]], "slices": {}}
    for s in SLICES:
        out["slices"][s] = {k: [r(v) for v in d[f"{s}_{c}"]] for k, c in
                            [("req_mean", "req_prb_mean"), ("req_max", "req_prb_max"), ("granted", "granted_prb"),
                             ("active_ues", "active_ues"), ("cqi", "cqi_mean"), ("latency_p95", "latency_ms_p95")]}
    meta_f = os.path.join(DATA, data[len("data/"):], f"trace_{id}.json") if data != "upload" else None
    if meta_f and os.path.exists(meta_f):
        out["params"] = json.load(open(meta_f))["params"]
    return out


class GenerateReq(BaseModel):
    out: str = "data/raw_new"
    trace_hours: float = Field(1.0, gt=0, le=24)
    seed: int = 20261126
    splits: Dict[str, int] = {"train": 84, "val": 18, "test": 18, "ood": 10}


@app.post("/api/datasets/generate")
def generate(req: GenerateReq):
    if not req.out.startswith("data/"):
        raise HTTPException(400, "output must be inside data/")
    name = req.out[len("data/"):].strip("/")
    safe_join(DATA, name)
    if name == "raw":
        raise HTTPException(400, "data/raw is used by the trained runs — choose another folder")
    if req.splits.get("train", 0) < 1 or req.splits.get("test", 0) < 1:
        raise HTTPException(400, "need at least 1 train and 1 test trace")
    only = ",".join(f"{k}:{int(v)}" for k, v in req.splits.items() if v > 0 and k in ("train", "val", "test", "ood"))
    rep = os.path.relpath(report_dir_for(req.out), ROOT)
    py = sys.executable
    cmd = ["bash", "-c", f"'{py}' -m datagen.generate --out 'data/{name}' --only {only} "
                         f"--trace-hours {req.trace_hours} --base-seed {int(req.seed)} && "
                         f"'{py}' -m datagen.check_dataset --data 'data/{name}' --out '{rep}'"]
    return launch("dataset", f"data/{name}", cmd, f"runs/_logs/gen_{name.replace('/', '_')}.log")


# ---------------------------------------------------------------- runs / training
def run_status(d):
    if os.path.exists(os.path.join(d, "metrics.json")):
        return "done"
    pf = os.path.join(d, "progress.json")
    for j in list_jobs("train", with_log=False):
        if os.path.realpath(os.path.join(RUNS, j["name"])) == os.path.realpath(d):
            return "running" if j["running"] else "failed"
    if os.path.exists(pf) and time.time() - os.path.getmtime(pf) < 300:
        return "running"
    return "failed"


@app.get("/api/runs")
def list_runs():
    out = []
    for root, dirs, files in os.walk(RUNS):
        dirs[:] = [x for x in dirs if not x.startswith("_")]
        if "config.json" in files:
            cfg = json.load(open(os.path.join(root, "config.json")))
            rel = os.path.relpath(root, RUNS)
            st = run_status(root)
            item = {"id": rel, "status": st, "model": cfg["model"], "features": cfg["features"],
                    "horizon_mode": cfg["horizon_mode"], "horizons_selected": cfg["horizons_selected"],
                    "model_horizons": cfg["model_horizons"], "lookback": cfg["lookback"],
                    "train_hours": cfg.get("train_hours"), "hidden": cfg["hidden"], "lr": cfg["lr"],
                    "slices": [t.split("_")[0] for g in cfg["groups"].values() for t in g["targets"]],
                    "created": cfg.get("created"), "data": cfg.get("data"), "mtime": os.path.getmtime(root),
                    "has_confidence": os.path.exists(os.path.join(root, "confidence.json"))}
            if st == "done":
                m = json.load(open(os.path.join(root, "metrics.json")))
                item["headline"] = {s: {str(h): r(m["test"][s][str(h)]["model_prb"]["r2"], 3)
                                        for h in cfg["horizons_selected"]} for s in m["test"]}
            out.append(item)
    return sorted(out, key=lambda x: x["mtime"], reverse=True)


@app.get("/api/runs/detail")
def run_detail(id: str):
    d = run_dir(id)
    out = {"id": id, "status": run_status(d), "config": json.load(open(os.path.join(d, "config.json")))}
    for name in ("metrics", "train_log", "progress", "confidence"):
        f = os.path.join(d, f"{name}.json")
        if os.path.exists(f):
            try:
                out[name] = json.load(open(f))
            except json.JSONDecodeError:
                pass
    if "train_log" not in out and "progress" in out:
        if "logs" in out["progress"]:
            out["train_log"] = out["progress"]["logs"]
        elif "log" in out["progress"]:
            out["train_log"] = {out["progress"].get("group", "?"): out["progress"]["log"]}
    log = os.path.join(RUNS, "_logs", id.replace("/", "_") + ".log")
    alt = os.path.join(RUNS, id + ".log")
    for f in (log, alt):
        if os.path.exists(f):
            out["log_tail"] = "".join(open(f, errors="replace").readlines()[-15:])
            break
    return clean_json(out)


class TrainReq(BaseModel):
    run_name: str
    data: str = "data/raw"
    horizons: List[int] = [5, 10, 15]
    horizon_mode: str = "all"
    model: str = "predp"
    features: str = "uni"
    train_traces: Optional[int] = None
    lookback: int = Field(30, ge=5, le=240)
    hidden: int = 64
    lr: float = 1e-3
    batch_size: int = 32
    max_epochs: int = Field(200, ge=1, le=2000)
    patience: int = Field(20, ge=1, le=500)
    median_filter: bool = True
    seed: int = 0


@app.post("/api/train")
def start_train(req: TrainReq):
    h = check_h(req.horizons)
    if not (1 <= h[0] and h[2] <= 15):
        raise HTTPException(400, "horizons must be within 1–15 s")
    if req.model not in ("predp", "lstm", "gru", "transformer") or req.features not in ("uni", "multi") \
            or req.horizon_mode not in ("all", "dedicated"):
        raise HTTPException(400, "invalid model/features/mode")
    safe_join(RUNS, req.run_name)
    data_dir(req.data)
    if run_status(os.path.join(RUNS, req.run_name)) == "running" and os.path.exists(os.path.join(RUNS, req.run_name)):
        raise HTTPException(409, "a run with this name is already training")
    cmd = [sys.executable, "train.py", "--data", req.data, "--run-name", req.run_name,
           "--horizons", *map(str, h), "--horizon-mode", req.horizon_mode, "--model", req.model,
           "--features", req.features, "--lookback", str(req.lookback), "--hidden", str(req.hidden),
           "--lr", str(req.lr), "--batch-size", str(req.batch_size), "--max-epochs", str(req.max_epochs),
           "--patience", str(req.patience), "--median-filter", str(int(req.median_filter)), "--seed", str(req.seed)]
    if req.train_traces:
        cmd += ["--train-traces", str(req.train_traces)]
    return launch("train", req.run_name, cmd, f"runs/_logs/{req.run_name.replace('/', '_')}.log")


class CalibReq(BaseModel):
    run: str


@app.post("/api/confidence/calibrate")
def calibrate(req: CalibReq):
    predictor(req.run)
    cmd = [sys.executable, "confidence.py", "--run", req.run]
    return launch("calibrate", req.run, cmd, f"runs/_logs/calib_{req.run.replace('/', '_')}.log")


@app.get("/api/jobs")
def jobs(kind: Optional[str] = None):
    return list_jobs(kind)


# ---------------------------------------------------------------- evaluation
class EvalReq(BaseModel):
    runs: List[str]
    split: str = "test"
    horizons: List[int] = [5, 10, 15]
    arima_n: int = Field(2000, ge=0, le=20000)
    tag: str


@app.post("/api/evaluate")
def start_eval(req: EvalReq):
    if not req.runs:
        raise HTTPException(400, "pick at least one run")
    for x in req.runs:
        predictor(x)
    h = check_h(req.horizons)
    if req.split not in ("test", "val", "ood"):
        raise HTTPException(400, "bad split")
    safe_join(os.path.join(REPORTS, "eval"), req.tag)
    cmd = [sys.executable, "evaluate.py", "--runs", *req.runs, "--split", req.split,
           "--horizons", *map(str, h), "--arima-n", str(req.arima_n), "--tag", req.tag]
    return launch("eval", req.tag, cmd, f"runs/_logs/eval_{req.tag.replace('/', '_')}.log",
                  meta={"out": req.tag + ("" if req.split == "test" else f"_{req.split}")})


# ---------------------------------------------------------------- PreNS comparison
PRENS_DIR = os.path.join(REPORTS, "prens_comparison")


@app.get("/api/prens")
def prens_comparison():
    f = os.path.join(PRENS_DIR, "comparison.json")
    if not os.path.exists(f):
        return {"available": False}
    out = json.load(open(f))
    out["available"] = True
    out["updated"] = os.path.getmtime(f)
    out["has_plot"] = os.path.exists(os.path.join(PRENS_DIR, "r2_comparison.png"))
    return clean_json(out)


@app.post("/api/prens/refresh")
def prens_refresh():
    """Re-evaluate every finished E1 model (PreDP/LSTM/GRU/Transformer, seed 0) and rebuild the comparison."""
    e1 = os.path.join(RUNS, "E1")
    runs = [f"E1/{d}" for d in sorted(os.listdir(e1)) if d.endswith("_s0")
            and os.path.exists(os.path.join(e1, d, "metrics.json"))] if os.path.isdir(e1) else []
    if not runs:
        raise HTTPException(409, "no finished E1 runs")
    py = sys.executable
    cmd = ["bash", "-c", f"'{py}' evaluate.py --runs {' '.join(runs)} --horizons 5 10 15 --arima-n 2000 --tag E1_prens"
                         f" && '{py}' experiments/prens_comparison.py"]
    return launch("prens", "prens_comparison", cmd, "runs/_logs/prens_comparison.log", meta={"runs": runs})


@app.get("/api/prens/plot")
def prens_plot():
    f = os.path.join(PRENS_DIR, "r2_comparison.png")
    if not os.path.exists(f):
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/reports")
def list_reports():
    base = os.path.join(REPORTS, "eval")
    if not os.path.isdir(base):
        return []
    out = []
    for d in os.listdir(base):
        f = os.path.join(base, d, "results.json")
        if os.path.exists(f):
            out.append({"tag": d, "mtime": os.path.getmtime(f)})
    return sorted(out, key=lambda x: x["mtime"], reverse=True)


@app.get("/api/reports/detail")
def report_detail(tag: str):
    d = safe_join(os.path.join(REPORTS, "eval"), tag)
    f = os.path.join(d, "results.json")
    if not os.path.exists(f):
        raise HTTPException(404, "report not found")
    res = json.load(open(f))
    res["tag"] = tag
    res["images"] = [x for x in sorted(os.listdir(d)) if x.endswith(".png")]
    return clean_json(res)


@app.get("/api/reports/image")
def report_image(tag: str, name: str):
    d = safe_join(os.path.join(REPORTS, "eval"), tag)
    f = safe_join(d, name)
    if not f.endswith(".png") or not os.path.exists(f):
        raise HTTPException(404)
    return FileResponse(f)


# ---------------------------------------------------------------- inference
@lru_cache(maxsize=16)
def _forecast(run, data, tid, horizons, mtime_key):
    from predictor import predict_with_confidence
    P = predictor(run)
    df = load_trace(data, tid)
    return predict_with_confidence(P, df, list(horizons), 0, len(df))


def forecast_cached(run, data, tid, H):
    d = run_dir(run)
    cf = os.path.join(d, "confidence.json")
    key = (os.path.getmtime(os.path.join(d, "metrics.json")), os.path.getmtime(cf) if os.path.exists(cf) else 0)
    return _forecast(run, data, tid, tuple(H), key)


@app.get("/api/infer/forecast")
def infer_forecast(run: str, data: str, id: str, h: List[int] = Query(...), start: int = 0,
                   end: Optional[int] = None):
    """Actual demand and predictions aligned at their TARGET time (t + h), for one window."""
    H = check_h(h)
    df = load_trace(data, id)
    t, pred, truth, conf = forecast_cached(run, data, id, H)
    T = len(df)
    end = min(end or T, T)
    times = np.arange(start, end)
    res = {"t": times.tolist(), "horizons": H, "segments": segments(df[(df.t >= start) & (df.t < end)]),
           "slices": {}, "metrics": {}}
    t0 = int(t[0])
    for s in pred:
        actual = df[f"{s}_req_prb_mean"].to_numpy()[start:end]
        series = {"actual": [r(v) for v in actual]}
        for k, hh in enumerate(H):
            anchor = times - hh
            ok = (anchor >= t0) & (anchor <= int(t[-1]))
            vals = np.full(len(times), np.nan)
            vals[ok] = pred[s][anchor[ok] - t0, k]
            series[f"h{hh}"] = [r(v) for v in vals]
            if conf is not None:
                cv = np.full(len(times), np.nan)
                cv[ok] = conf[s][anchor[ok] - t0, k]
                series[f"c{hh}"] = [r(v, 3) for v in cv]
        res["slices"][s] = series
        valid = ~np.isnan(truth[s]).any(1)
        res["metrics"][s] = {str(hh): {k2: r(v2, 3) for k2, v2 in
                                       regression_metrics(truth[s][valid, k], pred[s][valid, k]).items()}
                             for k, hh in enumerate(H)}
        y = df[f"{s}_req_prb_mean"].to_numpy(np.float64)
        res["metrics"][s]["persistence"] = {
            str(hh): r(regression_metrics(y[t[valid] + hh], y[t[valid]])["r2"], 3) for hh in H}
    res["length"] = T
    res["first_anchor"] = t0
    res["has_confidence"] = conf is not None
    return res


def conf_fn(run, H):
    """Confidence v1 hook (Day 4). Returns None until calibrated confidence is available."""
    return None


@app.get("/api/infer/replay")
def infer_replay(run: str, data: str, id: str, t: int, h: List[int] = Query(...), past: int = 60):
    H = check_h(h)
    P = predictor(run)
    df = load_trace(data, id)
    T = len(df)
    t = int(min(max(t, P.L - 1), T - 1))
    from infer import packages
    pk = packages(P, df, H, start=t, seconds=1, with_truth=True)
    out = {"t": t, "lookback": P.L, "horizons": H, "packages": pk, "slices": {}}
    lo = max(0, t - past + 1)
    for s in P.slices:
        y = df[f"{s}_req_prb_mean"].to_numpy()
        out["slices"][s] = {
            "past": [{"x": int(i - t), "y": r(y[i])} for i in range(lo, t + 1)],
            "future": [{"x": int(i - t), "y": r(y[i])} for i in range(t + 1, min(T, t + 17))],
            "forecast": [{"x": int(hh), "y": p["pred_prb"], "c": p["confidence"]}
                         for hh in H for p in pk if p["slice"] == s and p["horizon_s"] == hh]}
    sc = df["scenario"].astype(str).to_numpy()
    out["scenario"] = sc[t]
    return clean_json(out)


@app.get("/api/infer/export")
def infer_export(run: str, data: str, id: str, h: List[int] = Query(...), start: int = 0, seconds: int = 600,
                 with_truth: bool = False):
    H = check_h(h)
    P = predictor(run)
    df = load_trace(data, id)
    from infer import packages
    pk = packages(P, df, H, start=start, seconds=seconds, with_truth=with_truth)
    body = "\n".join(json.dumps(clean_json(p)) for p in pk) + "\n"
    fname = f"a1_packages_{'-'.join(map(str, H))}_{id}.jsonl"
    return PlainTextResponse(body, media_type="application/x-ndjson",
                             headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.post("/api/infer/upload")
async def infer_upload(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 200 * 2 ** 20:
        raise HTTPException(413, "file too large (200 MB max)")
    import io
    try:
        df = pd.read_csv(io.BytesIO(raw)) if file.filename.endswith(".csv") else pd.read_parquet(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"cannot read file: {e}")
    if "t" not in df:
        df["t"] = np.arange(len(df))
    if "scenario" not in df:
        df["scenario"] = "stable"
    need = [f"{s}_req_prb_mean" for s in SLICES]
    miss = [c for c in need if c not in df]
    if miss:
        raise HTTPException(400, f"missing columns: {miss}")
    base = re.sub(r"[^A-Za-z0-9_\-]", "_", os.path.splitext(file.filename)[0])[:40] or "upload"
    uid = f"{base}_{int(time.time())}"
    os.makedirs(os.path.join(DATA, "_uploads"), exist_ok=True)
    df.to_parquet(os.path.join(DATA, "_uploads", uid + ".parquet"), index=False)
    return {"data": "upload", "id": uid, "rows": len(df), "columns": list(df.columns)}


# ---------------------------------------------------------------- serve the built React app
UI_DIST = os.path.join(ROOT, "ui", "dist")
if os.path.isdir(os.path.join(UI_DIST, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(UI_DIST, "assets")), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    """Serve the React app; unknown paths fall back to index.html (client-side routing)."""
    if path.startswith("api/"):
        raise HTTPException(404)
    f = os.path.realpath(os.path.join(UI_DIST, path))
    if path and f.startswith(os.path.realpath(UI_DIST) + os.sep) and os.path.isfile(f):
        return FileResponse(f)
    index = os.path.join(UI_DIST, "index.html")
    if not os.path.exists(index):
        return PlainTextResponse("UI not built. Run: cd ui && npm install && npm run build", status_code=503)
    # never cache index.html: it names the current hashed JS bundle, which changes on every build
    return FileResponse(index, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
