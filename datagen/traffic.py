"""Per-UE offered traffic (bits/s) for eMBB, URLLC and mMTC at 100 ms resolution."""
import numpy as np
import pandas as pd


def ue_activity(rng, n, tick, n_ue, churn, p):
    """On/off state per UE; switching is much faster during ue_churn scenarios."""
    p_off = np.where(churn, tick / p["churn_mean_on_s"], tick / p["ue_mean_on_s"])
    p_on = np.where(churn, tick / p["churn_mean_off_s"], tick / p["ue_mean_off_s"])
    u = rng.random((n, n_ue))
    on = np.empty((n_ue, n), bool)
    state = rng.random(n_ue) < 0.9
    for t in range(n):
        flip = np.where(state, u[t] < p_off[t], u[t] < p_on[t])
        state = state ^ flip
        on[:, t] = state
    return on


def _poisson_starts(rng, rate_per_s, tick):
    return np.flatnonzero(rng.random(rate_per_s.shape[0]) < rate_per_s * tick)


def embb_ue(rng, n, tick, intensity, p, ev):
    """Adaptive-bitrate video (level follows load intensity) + bulk download sessions."""
    levels = np.asarray(ev["embb_levels_mbps"], float)
    rate = np.zeros(n)
    t = 0
    while t < n:                                 # per-dwell user preference offset,
        d = max(1, int(rng.exponential(p["embb_dwell_s"]) / tick))
        z = rng.normal(0, 0.7)                     # level follows load intensity directly
        idx = np.clip(np.round(intensity[t:t + d] * (len(levels) - 1) + z), 0, len(levels) - 1)
        rate[t:t + d] = levels[idx.astype(int)]
        t += d
    bulk_rate = p["embb_bulk_per_min"] / 60.0 * intensity
    for s in _poisson_starts(rng, bulk_rate, tick):
        d = int(rng.uniform(*ev["embb_bulk_dur_s"]) / tick)
        rate[s:s + d] += rng.uniform(*ev["embb_bulk_mbps"])
    rate *= np.maximum(1.0, intensity) ** 2      # surge (contention): more users/streams per UE
    sig = p["embb_noise"]
    rate *= rng.lognormal(-sig ** 2 / 2, sig, n)
    return rate * 1e6


def urllc_ues(rng, n, tick, n_ue, intensity, burst_mode, p, ev):
    """Low periodic control traffic + event bursts (more frequent in burst scenarios)."""
    out = np.zeros((n_ue, n))
    rate = np.where(burst_mode, p["urllc_burst_per_min_active"],
                    p["urllc_burst_per_min_idle"]) / 60.0
    cp = p["urllc_common_burst_p"]
    # sustained alarm mode: control loops switch to a high update rate for the whole event
    alarm = pd.Series(burst_mode.astype(float)).ewm(alpha=tick / p["urllc_alarm_ramp_s"]).mean().to_numpy()
    common = _poisson_starts(rng, rate * cp, tick)
    for u in range(n_ue):
        base = p["urllc_base_mbps"] * (0.5 + intensity) * rng.lognormal(-0.02, 0.2, n)
        r = base + p["urllc_alarm_mbps"] * rng.uniform(0.7, 1.3) * alarm * rng.lognormal(-0.02, 0.2, n)
        # periodic AGV-style control cycle (jittered period, active during "motion" phase)
        tt, ph = 0.0, rng.uniform(0, p["urllc_cycle_period_s"])
        lvl = p["urllc_cycle_mbps"] * rng.uniform(0.7, 1.3)
        while tt < n * tick:
            P = p["urllc_cycle_period_s"] * rng.uniform(0.9, 1.1)
            a0 = int((tt + ph) / tick); a1 = min(int((tt + ph + P * p["urllc_cycle_duty"]) / tick), n)
            if a1 > a0:
                r[a0:a1] += lvl * (0.5 + intensity[a0:a1])
            tt += P
        starts = np.union1d(_poisson_starts(rng, rate * (1 - cp), tick), common)
        for s in starts:
            d = max(1, int(rng.uniform(*ev["urllc_burst_dur_s"]) / tick))
            e = min(s + d, n)
            peak = rng.uniform(*ev["urllc_burst_mbps"])
            shape = rng.uniform(0.3, 1.3, e - s)
            shape[rng.random(e - s) < 0.2] *= 1.5     # sub-second spikes
            r[s:e] += peak * shape
        out[u] = r * 1e6
    return out


def mmtc_ue(rng, n, tick, intensity, p, ev):
    """Device group: active devices x report rate x payload, with synchronized report waves."""
    target = p["mmtc_dev_max"] * intensity
    n_dev = pd.Series(target).ewm(alpha=tick / p["mmtc_dev_tau_s"]).mean().to_numpy()
    n_dev *= rng.lognormal(-0.005, 0.1, n)
    reports = n_dev / p["mmtc_interval_s"]
    t = rng.uniform(0, p["mmtc_wave_period_s"])
    while t < n * tick:
        L = rng.uniform(*ev["mmtc_wave_len_s"])
        a, b = int(t / tick), min(int((t + L) / tick), n)
        if b > a:
            reports[a:b] += n_dev[a:b] * p["mmtc_wave_frac"] / L
        t += p["mmtc_wave_period_s"] * rng.uniform(0.85, 1.15)
    return reports * p["mmtc_payload_b"] * 8 * p["mmtc_overhead"]
