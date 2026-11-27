"""Per-UE channel quality (CQI) and spectral efficiency -> PRB capacity."""
import numpy as np
from scipy.signal import lfilter

# 3GPP TS 38.214 Table 5.2.2.1-2 (CQI table 1), efficiency in bits per RE, CQI 1..15
SE_TABLE = np.array([0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141,
                     2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547])


def cqi_process(rng, n, tick, n_ue, mean, p, ev):
    """AR(1) CQI around a per-UE mean, with occasional deep drops (blockage)."""
    phi = np.exp(-tick / p["cqi_tau_s"])
    std = p["cqi_std"]
    eps = rng.normal(0, std * np.sqrt(1 - phi ** 2), (n_ue, n))
    mu = mean + rng.normal(0, 1.0, n_ue)
    x = lfilter([1.0], [1.0, -phi], eps, axis=1)
    x += mu[:, None]
    # blockage drops: half-sine dips
    rate = p["cqi_drops_per_min"] / 60.0 * tick
    for u in range(n_ue):
        for s in np.flatnonzero(rng.random(n) < rate):
            d = int(rng.uniform(*ev["cqi_drop_dur_s"]) / tick)
            e = min(s + d, n)
            depth = rng.uniform(*ev["cqi_drop_depth"])
            x[u, s:e] -= depth * np.sin(np.pi * np.arange(e - s) / max(d, 1))
    return np.clip(np.round(x), 1, 15).astype(np.int8)


def prb_capacity_bps(cqi, radio, direction="dl"):
    """Bits per second that one PRB carries at the given CQI (DL or UL slots of the TDD pattern)."""
    se = SE_TABLE[cqi.astype(int) - 1]
    if direction == "dl":
        return (radio["re_per_prb_slot"] * (1 - radio["overhead"]) * radio["slots_per_s"]
                * se * radio["dl_ratio"] * radio["se_scale"])
    return (radio["re_per_prb_slot"] * (1 - radio["ul_overhead"]) * radio["slots_per_s"]
            * se * radio["ul_ratio"] * radio["se_scale_ul"])


def ul_cqi_process(rng, dl_cqi, tick, p):
    """UL CQI from the same UE's DL CQI (shared path loss): an offset for the lower UE transmit power,
    an extra penalty at the cell edge (power-limited UEs) and slowly varying UL-specific noise."""
    n_ue, n = dl_cqi.shape
    phi = np.exp(-tick / 2.0)
    eps = rng.normal(0, p["ul_cqi_noise"] * np.sqrt(1 - phi ** 2), (n_ue, n))
    noise = lfilter([1.0], [1.0, -phi], eps, axis=1)
    dl = dl_cqi.astype(float)
    edge = 0.3 * np.maximum(0.0, 9.0 - dl)            # power limit bites at low CQI
    return np.clip(np.round(dl - p["ul_cqi_offset"] - edge + noise), 1, 15).astype(np.int8)
