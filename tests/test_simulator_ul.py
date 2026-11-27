"""Uplink simulator (PLAN_UL.md): DL must be unchanged by UL, and UL must be well-formed."""
import copy

import numpy as np
import yaml

from datagen.simulator import SLICES, simulate_trace

UL_METRICS = ["req_prb_mean", "req_prb_max", "req_prb_p95", "granted_prb", "unmet_prb",
              "thr_mbps", "cqi_mean", "buffer_kb", "latency_ms_p95"]


def _cfg(ul):
    cfg = yaml.safe_load(open("configs/sim.yaml"))
    cfg["trace_hours"] = 0.1                     # 6 min keeps the test fast
    cfg["ul"]["enabled"] = ul
    return cfg


def test_dl_identical_with_and_without_ul():
    for profile in ("in", "ood"):
        a, _ = simulate_trace(_cfg(False), profile, 123, "t")
        b, meta = simulate_trace(_cfg(True), profile, 123, "t")
        for c in a.columns:                      # every DL column bit-identical
            if a[c].dtype.name == "category" or a[c].dtype == object:
                assert (a[c].astype(str) == b[c].astype(str)).all(), c
            else:
                assert np.array_equal(a[c].to_numpy(), b[c].to_numpy()), c
        assert meta["directions"] == ["dl", "ul"]


def test_ul_columns_well_formed():
    cfg = _cfg(True)
    df, meta = simulate_trace(cfg, "in", 7, "t")
    max_prb = cfg["slices"]["max_prb"]
    total = sum(df[f"{s}_ul_granted_prb"] for s in SLICES)
    assert (total <= cfg["radio"]["total_prb"] + 1e-2).all()      # one UL PRB pool per slot
    for s in SLICES:
        for m in UL_METRICS:
            col = f"{s}_ul_{m}"
            assert col in df, col
            assert np.isfinite(df[col]).all() and (df[col] >= 0).all(), col
        assert (df[f"{s}_ul_req_prb_max"] >= df[f"{s}_ul_req_prb_mean"] - 1e-4).all()
        # granted can exceed the current request while backlog drains, but never the slice quota
        assert (df[f"{s}_ul_granted_prb"] <= max_prb[SLICES.index(s)] + 1e-3).all()
        assert (df[f"{s}_ul_cqi_mean"] <= df[f"{s}_cqi_mean"] + 1e-3).mean() > 0.95   # UL CQI below DL
    assert np.allclose(df["total_ul_req_prb_mean"], sum(df[f"{s}_ul_req_prb_mean"] for s in SLICES), atol=1e-3)
    assert meta["params_ul"] is not None


def test_ul_disabled_gives_dl_only_dataset():
    df, meta = simulate_trace(_cfg(False), "in", 7, "t")
    assert not [c for c in df.columns if "_ul_" in c]
    assert meta["directions"] == ["dl"] and meta["params_ul"] is None
