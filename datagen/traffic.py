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
    n_dev = n_dev * rng.lognormal(-0.005, 0.1, n)   # not in place: pandas 3 returns a read-only array
    reports = n_dev / p["mmtc_interval_s"]
    t = rng.uniform(0, p["mmtc_wave_period_s"])
    while t < n * tick:
        L = rng.uniform(*ev["mmtc_wave_len_s"])
        a, b = int(t / tick), min(int((t + L) / tick), n)
        if b > a:
            reports[a:b] += n_dev[a:b] * p["mmtc_wave_frac"] / L
        t += p["mmtc_wave_period_s"] * rng.uniform(0.85, 1.15)
    return reports * p["mmtc_payload_b"] * 8 * p["mmtc_overhead"]


# ------------------------------------------------------------------ uplink (PLAN_UL.md)
# Every UL generator takes its own RNG (rng_ul) so the DL draws are untouched.

def _alternating_sessions(rng, n, tick, on_s, off_s):
    """0/1 mask of sessions: on for U(on_s) seconds, off for U(off_s) seconds, random start state."""
    m = np.zeros(n)
    t, state = 0, rng.random() < 0.5
    while t < n:
        d = max(1, int(rng.uniform(*(on_s if state else off_s)) / tick))
        if state:
            m[t:t + d] = 1.0
        t += d
        state = not state
    return m


def embb_ul(rng, n, tick, dl_bps, on, p, ev):
    """eMBB UL per UE: TCP ACKs (fraction of the UE's DL rate) + camera streams uploaded by one UE.
    dl_bps, on: (n_ue, n). Returns (n_ue, n) bits/s."""
    n_ue = dl_bps.shape[0]
    out = dl_bps * p["embb_ul_ack_frac"]
    n_cam = int(np.round(p["embb_ul_cameras"]))
    up = rng.integers(n_ue)                          # the uploader UE (lab: one UE carries all AGV cameras)
    for _ in range(n_cam):
        sess = _alternating_sessions(rng, n, tick, ev["embb_ul_camera_on_s"], ev["embb_ul_camera_off_s"])
        rate = p["embb_ul_camera_mbps"] * rng.uniform(0.7, 1.3)
        sig = p["embb_ul_camera_noise"]
        out[up] += sess * rate * 1e6 * rng.lognormal(-sig ** 2 / 2, sig, n)
    return out * on


def urllc_ul(rng, n, tick, dl_bps, burst_mode, p):
    """URLLC UL per UE: acknowledgements of the DL commands (short lag) + alarm/status reports
    that ramp up during burst / contention / sla_conflict. Returns (n_ue, n) bits/s."""
    lag = int(np.round(p["urllc_ul_ack_lag_ticks"]))
    ack = np.roll(dl_bps, lag, axis=1) * p["urllc_ul_ack_ratio"]
    if lag:
        ack[:, :lag] = dl_bps[:, :1] * p["urllc_ul_ack_ratio"]
    alarm = pd.Series(burst_mode.astype(float)).ewm(alpha=tick / p["urllc_ul_alarm_ramp_s"]).mean().to_numpy()
    out = np.empty_like(dl_bps)
    for u in range(dl_bps.shape[0]):
        rep = p["urllc_ul_alarm_mbps"] * rng.uniform(0.7, 1.3) * alarm * rng.lognormal(-0.02, 0.2, n)
        out[u] = ack[u] * rng.lognormal(-0.005, 0.1, n) + rep * 1e6
    return out


def mmtc_ul(rng, n, tick, intensity, burst_mode, p):
    """mMTC UL for one UE: telemetry from several sources (records = position, speed, battery, status).
    Records per tick are Poisson; the rate follows load intensity and jumps during events. Returns (n,) bits/s."""
    n_src = max(1, int(np.round(p["mmtc_ul_sources"])))
    rate_hz = n_src * p["mmtc_ul_rate_hz"] * (0.5 + 0.5 * np.minimum(intensity, 1.5))
    rate_hz = rate_hz * np.where(burst_mode, p["mmtc_ul_event_mult"], 1.0)
    records = rng.poisson(rate_hz * tick)
    kb = p["mmtc_ul_record_kb"] * rng.lognormal(-0.02, 0.2, n)
    return records * kb * 8e3 * p["mmtc_ul_overhead"] / tick
