# Plan: add uplink (UL) to the simulator

Status: **Phases 1–3 done, 2026-11-27** (simulator, `data/raw_v2` 130 h with checks in `reports/dataset/raw_v2/`, pipeline + UI with `--direction dl|ul`). Phases 4–5 to do.

Phase 3 notes:
- `train.py --direction dl|ul` (default dl; saved in `config.json`). Old runs read as DL.
- `--features pair` = the slice's DL + UL history, used for E14 (same information as a shared 2-output model).
- `evaluate.py` refuses to mix DL and UL runs. Its `--data` now defaults to the run's own dataset.
- A1 packages carry `"direction"`.
- API: `/api/trace?direction=`, runs and datasets report directions, UL uploads accepted.
- UI: DL/UL switch on Dataset, Direction + `pair` on Train, a UL badge and automatic dataset switch on Inference, mixed-direction guard on Evaluate.
- Experiment sets: `queue.py --exp E13` (UL point / peak / q90 / peak-q90) and `--exp E14` (uni vs pair, UL and DL). DL columns verified bit-identical to dataset v1; tests in `tests/test_simulator_ul.py`. UL parameters live in `per_trace_ul` / `events_ul` and use their own RNG stream.

Phase 1 check (18 test traces):

| Slice | UL / DL mean | UL CV | UL persistence R² @5/10/15 s | UL saturated |
|---|---|---|---|---|
| eMBB | 0.42 | 0.72 | 0.73 / 0.55 / 0.42 | 1.2% |
| URLLC | 1.17 | 1.06 | 0.70 / 0.41 / 0.19 | 2.7% |
| mMTC | 0.76 | 1.40 | 0.75 / 0.60 / 0.50 | 1.4% |

Phase 2 (full dataset `data/raw_v2`, 130 traces; all 9 checks pass, DL bit-identical to `data/raw`):

| Slice | UL/DL (train) | CV DL/UL | corr(DL,UL) lag 0 | DL(t)→UL(t+5/10/15) | UL persistence R² @5/10/15 |
|---|---|---|---|---|---|
| eMBB | 0.46 | 0.62 / 0.72 | 0.54 | 0.47 / 0.41 / 0.36 | 0.74 / 0.55 / 0.43 |
| URLLC | 0.92 | 0.84 / 1.15 | **0.93** | 0.72 / 0.53 / 0.42 | 0.70 / 0.41 / 0.20 |
| mMTC | 0.63 | 0.98 / 1.33 | 0.56 | 0.41 / 0.29 / 0.24 | 0.75 / 0.60 / 0.50 |

The DL↔UL cross-correlation is symmetric (shared scenario driver, no lead/lag). URLLC is strongly linked (partly by design: UL acks follow DL commands), so it's the slice to test with one 2-output model in E14.
Goal: generate UL PRB demand per slice next to the existing DL demand, and train separate UL models
(option 1: one model per slice and direction), without changing any DL result.

---

## Decisions (2026-11-27)

This is **pre-training** on a general simulator; the model is fine-tuned on the real application later. Lab facts set realistic **ranges**, not exact copies.

| Topic | Decision |
|---|---|
| **TDD split** | Lab default: 6 DL + 1 special (8 DL symbols) + 3 UL slots per 10 slots → about 65.7% DL, 30% UL, 4.3% guard. Set **`ul_ratio: 0.30`**. **Keep `dl_ratio: 0.7`** so the DL data and results stay unchanged; the ~6% DL difference is absorbed by `se_scale` calibration and fine-tuning. |
| **eMBB UL** | **Camera uploads**: 0–5 camera streams per trace over the eMBB UE (randomised count and bitrate), plus TCP ACKs |
| **mMTC UL** | **Telemetry**: several sources per UE (randomised 1–10, about 5 in the lab), each sending small combined records (position, speed, battery, status) at about 1–4 Hz (2 Hz in the lab) |
| **URLLC UL** | **Acknowledgements** for the DL commands (request → reply, short lag), plus alarm/status bursts |
| **UL quotas** | **Same quota ratios as DL** (the lab applies identical min/max/dedicated ratios to PRB_DL and PRB_UL). The UL allocator uses the DL `min_prb` / `max_prb` / `weight` values; only the airtime (`ul_ratio`) differs. |
| **Report target** | **C: both.** Point target (PreNS comparison) and peak target (own contribution), for DL and UL. |

---

## 0. Design principles

| Principle | How |
|---|---|
| **DL stays exactly the same** | UL uses its **own random stream** (`rng_ul = default_rng([seed, 1])`). DL columns stay bit-identical to the current dataset, so every DL model and result remains valid. |
| **DL and UL see the same events** | UL reuses the same scenario timeline and intensity curves. A burst, contention or UE churn affects both directions at the same time, as in a real cell. |
| **DL and UL are separate resources** | Separate capacity (TDD UL slots), separate allocator, separate columns. Never summed. |
| **Same output format** | 100 ms simulation, aggregated to 1 s (mean / max / p95), same trace and split structure. |

---

## 1. TDD slot split (`configs/sim.yaml → radio`)

The testbed's `N310.yml` sets no TDD pattern, so the srsRAN default for n78 / 30 kHz applies:
about 6 DL + 3 UL slots + 1 special slot per 5 ms.

```yaml
radio:
  dl_ratio: 0.7          # existing
  ul_ratio: 0.3          # new: share of time for UL
  ul_overhead: 0.18      # new: DMRS + PUCCH/SRS overhead (DL uses 0.14)
```

**To verify:** the TDD pattern on the running gNB (`tdd_ul_dl_cfg`). Set `dl_ratio` and `ul_ratio` from it.

---

## 2. UL channel (`datagen/channel.py`)

UEs transmit with limited power, so UL spectral efficiency is lower than DL.

| Item | Plan |
|---|---|
| UL CQI | Correlated with the UE's DL CQI (same path loss), minus a **UL offset** of about 1–3 CQI steps, plus extra noise |
| Power limit | Cell-edge UEs (low CQI) get an extra penalty, so UL MCS is capped |
| UL capacity per PRB | `168 × (1 − ul_overhead) × 2000 × SE_ul × ul_ratio × se_scale_ul` |
| Calibration | Separate `se_scale_ul`, set from an iperf3 UL test per slice |

New function: `prb_capacity_bps(cqi, radio, direction="dl")`. The DL call stays unchanged.

---

## 3. UL traffic per slice (`datagen/traffic.py`)

New functions, driven by the same intensity `I[s]` and `burst_mode` as DL:

| Slice | UL traffic model | Expected UL behaviour |
|---|---|---|
| **eMBB** | (a) TCP ACKs ≈ 2–5% of DL rate; (b) per-trace chance of an **uploader UE** (AGV camera, video call) at 2–10 Mbit/s; (c) occasional bulk uploads | Mostly low and smooth, plus upload sessions |
| **URLLC** | (a) **periodic AGV/robot status** (small packets, 10–100 ms cycle) → steady low rate; (b) **alarm/event reports** during `burst_mode`; (c) a **response for each DL command** (request–response coupling with a short lag) | Steady base plus event bursts, correlated with DL |
| **mMTC** | **Sensor uploads**: many small periodic reports (2–60 s period, random phase per sensor), occasional event-triggered batches | Mostly near zero with spikes, high relative variance: **the main mMTC direction** |

UL scheduling details (SR → BSR → grant) are much shorter than the 100 ms tick, so they're approximated by:
- a **minimum UL grant** of 1–2 PRBs when a UE has data (BSR granularity), which adds a floor and jitter to small flows
- the UL request = UE buffer ÷ UL capacity, as for DL

New `sim.yaml` ranges (per trace, randomised):

```yaml
per_trace:
  embb_ul_ack_frac:        [0.02, 0.05]   # TCP ACKs relative to DL rate
  embb_ul_cameras:         [0, 5]         # camera streams uploading over the eMBB UE (lab: up to 5 AGV cameras)
  embb_ul_camera_mbps:     [1, 4]         # per camera stream
  embb_ul_camera_on_s:     [30, 300]      # camera sessions start/stop
  urllc_ul_ack_ratio:      [0.3, 1.0]     # UL acknowledgement size relative to the DL command
  urllc_ul_ack_lag_ticks:  [0, 2]         # reply lag in 100 ms ticks
  urllc_ul_alarm_mbps:     [1.0, 3.0]     # alarm/status bursts in burst_mode
  mmtc_ul_sources:         [1, 10]        # telemetry sources per mMTC UE (lab: 5 AGVs)
  mmtc_ul_rate_hz:         [1, 4]         # records per second per source (lab: 2 Hz)
  mmtc_ul_record_kb:       [0.2, 2]       # combined record: position, speed, battery, status
  ul_cqi_offset:           [1, 3]         # UL CQI below DL (UE power limit)
```

The **OOD profile** gets wider UL ranges (longer alarm bursts, more uploaders) to test generalisation.

---

## 4. UL allocator (`datagen/allocator.py`)

- Reuse `allocate(req, cap_bps, cfg, tick)` with a **separate UL config**.
- New `slices.ul_min_prb`, `ul_max_prb` and `ul_weight`, defaulting to the DL values. The lab currently copies the DL budget to UL.
- Separate UL backlog and buffer, so unmet UL demand carries over.

---

## 5. Simulator and output (`datagen/simulator.py`)

In `simulate_trace`, after the DL loop:
1. Generate UL traffic, UL CQI and UL capacity per slice using `rng_ul`.
2. Compute UL requested PRBs (min grant, per-UE cap `ue_max_prb_ul`).
3. Run the UL allocator.
4. Aggregate to 1 s.

**New columns.** The DL names are unchanged for compatibility:

| Existing (DL) | New (UL) |
|---|---|
| `embb_req_prb_mean / max / p95` | `embb_ul_req_prb_mean / max / p95` |
| `embb_granted_prb`, `_unmet_prb` | `embb_ul_granted_prb`, `embb_ul_unmet_prb` |
| `embb_thr_mbps`, `_cqi_mean`, `_buffer_kb`, `_latency_ms_p95` | `embb_ul_thr_mbps`, `embb_ul_cqi_mean`, `embb_ul_buffer_kb`, `embb_ul_latency_ms_p95` |
| `total_req_prb_mean` | `total_ul_req_prb_mean` |

The same applies to `urllc_*` and `mmtc_*`, giving about 31 new columns (34 → ~65). Set `sim_version: "2.0"` and `directions: ["dl", "ul"]` in `manifest.json`.

---

## 6. Dataset checks (`datagen/check_dataset.py`)

Run each check **per direction**:

| Check | Pass condition |
|---|---|
| DL unchanged | DL columns identical to dataset v1 (same seeds) |
| UL/DL ratio per slice | eMBB UL ≪ DL; mMTC UL ≫ DL; URLLC in between |
| Variance, skew, zero fraction | Reported per slice and direction (expected: UL spikier, especially mMTC) |
| Autocorrelation 1–15 s | UL still above 0 → predictable |
| Persistence R² @1/5/10/15 s | Reported per direction: shows how hard UL is |
| Saturation | UL contention happens in contention and sla_conflict segments, not constantly |

Plots: UL vs DL requested PRBs per slice for one trace; distributions; ACF per direction.

---

## 7. Training pipeline (separate UL models)

| File | Change |
|---|---|
| `prep/data.py` | `target_col(slice, direction="dl")` → `<slice>_req_prb_mean` or `<slice>_ul_req_prb_mean`; `feature_columns(..., direction)` |
| `train.py` | `--direction dl|ul` (default `dl`), saved in `config.json`; run names like `predp_ul_...` |
| `predictor.py`, `confidence.py`, `evaluate.py` | Read the direction from the config. Confidence τ is fitted **per direction**. |
| `infer.py` + A1 package | Add `"direction": "dl" | "ul"` to each package |
| `experiments/queue.py` | New set **E13**: UL point, UL peak, UL q90 (for mMTC/URLLC) |
| `prep/testbed.py` | Map UL KPM columns (`RRU.PrbUsedUl`, `DRB.UEThpUl`) to `*_ul_*` |
| UI | **Direction** selector (DL / UL) on Train, Evaluate and Inference; direction badge on runs |

Optional later: **E14**, one model per slice with 2 outputs (DL and UL), compared against the separate models.

---

## 8. Experiments after the build

| ID | Question |
|---|---|
| E13a | UL point target: how does PreDP compare with persistence and ARIMA per slice? |
| E13b | UL peak target: is it more useful for spiky mMTC/URLLC UL? |
| E13c | UL q90 (safe side) for URLLC/mMTC: under-prediction rate and shortfall |
| E14 | Separate DL/UL models vs one 2-output model per slice |

Report UL with **skill vs persistence**, not only R², because UL and DL variances differ.

### Choosing 1 model (2 outputs) or 2 models (separate DL/UL) per slice — E14

1. **Check the link first (no training):** on the train split, per slice, measure the cross-correlation of DL(t) with UL(t+h) and of UL(t) with DL(t+h) for h = 1–15 s, and whether a linear UL forecast improves when DL history is added.
   - Weak link → 2 models. Strong link (expected for URLLC: DL command → UL reply) → test 1 model.
2. **Train both versions** with the same data, splits, horizons, input length and target, **3 seeds each**.
3. **Compare per slice and direction:** skill vs persistence at 5/10/15 s, R²/RMSE/MAE, under-prediction rate and shortfall (URLLC/mMTC UL), the **OOD test**, seed spread, and cost.
4. **Decision rule (fixed in advance).** Choose 1 model for a slice only if:
   - it improves skill by more than 2× the seed spread in at least one direction,
   - it's not worse in the other direction,
   - it's not worse on OOD.

   Otherwise keep **2 models** (the default). The choice can differ per slice.

Expected outcome (to confirm): eMBB → 2 models, URLLC → possibly 1 model, mMTC → 2 models.

---

## 9. Effort estimate

| Step | Time |
|---|---|
| 1–5 Simulator (channel, traffic, allocator, output) | 3–4 h |
| 6 Dataset checks + generation of 130 h | 1 h + about 10 min |
| 7 Pipeline + UI direction support | 2–3 h |
| 8 Training E13 (about 4 runs) + evaluation | about 2 h of GPU time |
| **Total** | **about 1–1.5 days** |

---

## 10. Open questions (decide before building)

1. **TDD pattern** on the running gNB: confirm the DL/UL slot split.
2. **Which UEs upload?** Do the AGVs or cameras send video UL (eMBB uploader), and how many sensors per mMTC UE?
3. **UL slice quotas:** same as DL for now, or separate values?
4. **UL targets:** point, peak, or q90 for each slice (likely peak or q90 for mMTC/URLLC).
