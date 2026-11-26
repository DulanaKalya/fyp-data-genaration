"""Scenario scheduler and per-slice load intensity envelopes."""
import numpy as np

SCENARIOS = ["stable", "ramp_up", "ramp_down", "burst", "ue_churn", "contention", "sla_conflict"]
IDX = {s: i for i, s in enumerate(SCENARIOS)}
EMBB, URLLC, MMTC = 0, 1, 2


def scenario_timeline(rng, n, tick, p, ev):
    """Markov chain over scenarios. Returns per-tick label array and segment list."""
    labels = np.empty(n, dtype=np.int8)
    segs = []
    t, prev = 0, None
    base_w = np.array([p[f"w_{s}"] for s in SCENARIOS], float)
    while t < n:
        w = base_w.copy()
        if prev is not None:
            w[prev] *= 0.3                       # discourage repeating the same scenario
            if prev in (IDX["contention"], IDX["sla_conflict"]):
                w[IDX["ramp_down"]] *= 3         # high-load periods usually relax afterwards
                w[IDX["stable"]] *= 2
                w[IDX["contention"]] *= 0.3
                w[IDX["sla_conflict"]] *= 0.3
        k = int(rng.choice(len(SCENARIOS), p=w / w.sum()))
        lo, hi = ev["seg_dur_s"][SCENARIOS[k]]
        d = max(1, int(round(rng.uniform(lo, hi) * p["seg_scale"] / tick)))
        b = min(t + d, n)
        labels[t:b] = k
        segs.append((t, b, k))
        t, prev = b, k
    return labels, segs


def _scurve(r):
    x = np.linspace(0.0, 1.0, r, endpoint=False) + 1.0 / max(r, 1)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def build_intensity(rng, segs, n, tick, p, ev):
    """Per-slice load intensity (normal range <= 0.8, surges up to 1.5) driven by the scenario timeline."""
    I = np.empty((3, n))
    cur = rng.uniform(0.2, 0.6, 3)
    for a, b, k in segs:
        name = SCENARIOS[k]
        target = cur.copy()
        ramp_s = rng.uniform(*ev["switch_s"])
        if name == "stable":
            target = cur + rng.normal(0, 0.05, 3)
        elif name in ("ramp_up", "ramp_down"):
            n_sl = rng.integers(1, 3)
            sl = rng.choice(3, size=n_sl, replace=False, p=[0.45, 0.2, 0.35])
            delta = rng.uniform(0.25, 0.6, n_sl)
            target[sl] = cur[sl] + (delta if name == "ramp_up" else -delta)
            ramp_s = rng.uniform(*ev["ramp_s"])
        elif name == "contention":
            target = rng.uniform(1.1, 1.4, 3)       # surge above normal operating range
            ramp_s = rng.uniform(*ev["contention_ramp_s"])
        elif name == "sla_conflict":
            target[EMBB] = rng.uniform(1.0, 1.3)
            target[URLLC] = max(cur[URLLC], rng.uniform(0.5, 0.8))
            ramp_s = rng.uniform(*ev["contention_ramp_s"])
        # burst / ue_churn keep the load level; their effect is in the traffic generators
        hi = 1.5 if name in ("contention", "sla_conflict") else 0.8
        target = np.clip(target, 0.05, hi)
        r = min(max(1, int(ramp_s / tick)), b - a)
        I[:, a:a + r] = cur[:, None] + (target - cur)[:, None] * _scurve(r)[None, :]
        I[:, a + r:b] = target[:, None]
        cur = target
    # slow compressed "daily" trend + small noise
    tt = np.arange(n) * tick
    phase = rng.uniform(0, 2 * np.pi, 3)[:, None]
    I *= 1.0 + p["trend_amp"] * np.sin(2 * np.pi * tt[None, :] / p["trend_period_s"] + phase)
    I += rng.normal(0, p["intensity_noise"], I.shape)
    return np.clip(I, 0.03, 1.5)
