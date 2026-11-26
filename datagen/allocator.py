"""Simple slice-level PRB allocator. Only used to produce granted/unmet/latency features."""
import numpy as np


def allocate(req, cap_bps, cfg, tick):
    """
    req:     (3, n) requested PRBs per slice per tick (offered load)
    cap_bps: (3, n) demand-weighted bits/s per PRB for each slice
    Returns dict of (3, n) arrays: granted, unmet, buffer_kb, latency_ms.
    """
    sl, total = cfg["slices"], cfg["radio"]["total_prb"]
    mode, drain = cfg["allocator"]["mode"], cfg["allocator"]["drain_s"]
    w = np.asarray(sl["weight"], float)
    mn = np.asarray(sl["min_prb"], float)
    mx = np.asarray(sl["max_prb"], float)
    base_lat = np.asarray(sl["base_latency_ms"], float)
    n = req.shape[1]
    granted = np.empty((3, n)); unmet = np.empty((3, n))
    buf_kb = np.empty((3, n)); lat = np.empty((3, n))
    max_bl = cfg["allocator"]["max_backlog_s"] * mx
    backlog = np.zeros(3)                      # PRB-seconds of unserved demand
    for t in range(n):
        need = req[:, t] + backlog / drain
        if mode == "static":
            alloc = np.minimum(need, total / 3.0)
        else:
            alloc = np.minimum(need, mn)
            for _ in range(3):                 # weighted water-filling of the remainder
                rem = total - alloc.sum()
                room = np.minimum(need, mx) - alloc
                if rem <= 1e-6 or room.max() <= 1e-6:
                    break
                share = rem * w * room / max((w * room).sum(), 1e-9)
                alloc += np.minimum(share, room)
        g = alloc
        backlog = np.minimum(np.maximum(0.0, backlog + (req[:, t] - g) * tick), max_bl)
        granted[:, t] = g
        unmet[:, t] = np.maximum(0.0, need - g)
        buf_kb[:, t] = backlog * cap_bps[:, t] / 8e3
        lat[:, t] = base_lat + np.minimum(1000.0, 1e3 * backlog / np.maximum(g, 1e-3))
    return {"granted": granted, "unmet": unmet, "buffer_kb": buf_kb, "latency_ms": lat}
