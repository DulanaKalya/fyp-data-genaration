"""Testbed-matched 3-slice traffic simulator: 100 ms internal tick, 1 s output rows."""
import copy
import numpy as np
import pandas as pd

from .scenarios import SCENARIOS, IDX, scenario_timeline, build_intensity
from .channel import cqi_process, prb_capacity_bps
from .traffic import ue_activity, embb_ue, urllc_ues, mmtc_ue

SLICES = ["embb", "urllc", "mmtc"]


def resolve_profile(cfg, name):
    prof = cfg["profiles"][name]
    if "inherits" not in prof:
        return prof
    base = copy.deepcopy(resolve_profile(cfg, prof["inherits"]))
    for sec in ("per_trace", "events"):
        for k, v in prof.get(sec, {}).items():
            if isinstance(v, dict):
                base[sec].setdefault(k, {}).update(v)
            else:
                base[sec][k] = v
    return base


def sample_params(rng, prof):
    return {k: float(rng.uniform(*v)) for k, v in prof["per_trace"].items()}


def simulate_trace(cfg, profile, seed, trace_id):
    rng = np.random.default_rng(seed)
    prof = resolve_profile(cfg, profile)
    p, ev = sample_params(rng, prof), prof["events"]
    tick, radio = cfg["tick_s"], cfg["radio"]
    k = int(round(cfg["out_dt_s"] / tick))
    n_sec = int(round(cfg["trace_hours"] * 3600 / cfg["out_dt_s"]))
    n = n_sec * k

    labels, segs = scenario_timeline(rng, n, tick, p, ev)
    I = build_intensity(rng, segs, n, tick, p, ev)
    burst_mode = np.isin(labels, [IDX["burst"], IDX["sla_conflict"], IDX["contention"]])
    churn = labels == IDX["ue_churn"]
    n_ue = cfg["slices"]["n_ue"]

    req = np.zeros((3, n)); cap = np.zeros((3, n))
    active = np.zeros((3, n)); cqi_mean = np.zeros((3, n))
    for s, name in enumerate(SLICES):
        nu = n_ue[s]
        on = ue_activity(rng, n, tick, nu, churn, p)
        if name == "embb":
            bps = np.stack([embb_ue(rng, n, tick, I[s], p, ev) for _ in range(nu)])
        elif name == "urllc":
            on[:] = True                         # control devices stay attached
            bps = urllc_ues(rng, n, tick, nu, I[s], burst_mode, p, ev)
        else:
            bps = np.stack([mmtc_ue(rng, n, tick, I[s], p, ev) for _ in range(nu)])
        bps *= on
        cqi = cqi_process(rng, n, tick, nu, p[f"cqi_mean_{name}"], p, ev)
        c = prb_capacity_bps(cqi, radio)
        prb = np.minimum(bps / c, cfg["slices"]["ue_max_prb"][s])
        bps = prb * c
        if name == "urllc":
            prb *= p["urllc_headroom"]           # latency headroom
        req[s] = prb.sum(0)
        tot = bps.sum(0)
        cap[s] = np.where(tot > 0, tot / np.maximum(prb.sum(0), 1e-9), c.mean(0))
        active[s] = on.sum(0)
        cnt = on.sum(0)
        cqi_mean[s] = np.where(cnt > 0, (cqi * on).sum(0) / np.maximum(cnt, 1), cqi.mean(0))

    alloc = allocate_slices(req, cap, cfg, tick)

    # ---- aggregate 100 ms -> 1 s ----
    def blk(x):
        return x.reshape(x.shape[0], n_sec, k)

    R = blk(req)
    cols = {"t": np.arange(n_sec, dtype=np.int32),
            "trace_id": trace_id,
            "scenario": pd.Categorical([SCENARIOS[i] for i in labels[::k]], categories=SCENARIOS)}
    for s, name in enumerate(SLICES):
        cols[f"{name}_req_prb_mean"] = R[s].mean(1)
        cols[f"{name}_req_prb_max"] = R[s].max(1)
        cols[f"{name}_req_prb_p95"] = np.percentile(R[s], 95, axis=1)
        cols[f"{name}_granted_prb"] = blk(alloc["granted"])[s].mean(1)
        cols[f"{name}_unmet_prb"] = blk(alloc["unmet"])[s].mean(1)
        cols[f"{name}_active_ues"] = blk(active)[s][:, -1].astype(np.int8)
        cols[f"{name}_thr_mbps"] = (blk(alloc["granted"] * cap)[s].mean(1)) / 1e6
        cols[f"{name}_cqi_mean"] = blk(cqi_mean)[s].mean(1)
        cols[f"{name}_buffer_kb"] = blk(alloc["buffer_kb"])[s].mean(1)
        cols[f"{name}_latency_ms_p95"] = np.percentile(blk(alloc["latency_ms"])[s], 95, axis=1)
    df = pd.DataFrame(cols)
    df["total_req_prb_mean"] = sum(df[f"{s}_req_prb_mean"] for s in SLICES)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)

    meta = {"trace_id": trace_id, "profile": profile, "seed": int(seed),
            "sim_version": cfg["sim_version"], "params": p,
            "segments": [{"start_s": a * tick, "end_s": b * tick, "scenario": SCENARIOS[c]}
                         for a, b, c in segs]}
    return df, meta


def allocate_slices(req, cap, cfg, tick):
    from .allocator import allocate
    return allocate(req, cap, cfg, tick)
