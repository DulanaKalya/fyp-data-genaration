# Plan: 1-second slice-demand dataset, multi-horizon attention Bi-LSTM, and a training/inference UI

Project: Predictive Network Slice Management for Private 5G (Group 17)
Reference: PreNS (Wu et al., IEEE TCCN 2026), the PreDP predictor
Data source: **a simulator that matches our testbed. The TRACTOR dataset is not used.**

---

## 0. Goals and fixed decisions

| Item | Decision |
|---|---|
| Sampling interval | **1 s** (PreNS used 100 ms). Our horizons are seconds long, so finer sampling is not needed. |
| Prediction horizons | Chosen in the UI: **3 values, H1 < H2 < H3, each from 5 to 15 s**, in whole seconds (e.g. 5/10/15 or 7/9/12) |
| Target | **Requested PRBs per slice** (the demand). Not granted PRBs, because those are capped by the controller. |
| Testbed being modelled | Band n78, 60 MHz, 30 kHz SCS, **162 PRBs**, 9 UEs: 3 eMBB, 3 URLLC, 3 mMTC |
| Model | PreNS PreDP: attention Bi-LSTM with 64 hidden units per direction, Adam lr 1e-3, MSE loss, batch 32 |
| Data source | Our own simulator (generalized dataset), then fine-tuning on live testbed traces later |
| Outputs to the xApp | For each slice and horizon: `{prediction, confidence, target_time = t_p + H}`. This is the A1 package from the feasibility report. |

### Main design choice: the dataset does not depend on the horizons
- The simulator writes **plain per-second time series**. No horizons are baked into the dataset.
- Input windows and targets are built **when training starts**, from the horizons chosen in the UI.
- The default model predicts **every horizon from 1 to 15 s** (15 outputs per slice). Choosing different horizons in the UI then only changes which outputs are read, so **no retraining is needed**.
- A "dedicated" mode trains a model with exactly 3 outputs, matching the paper. We will compare the two modes.

---

## Phase 1: Testbed-matched traffic simulator

### 1.1 How it works
- It runs internally on a **100 ms tick**, so bursts shorter than a second exist in the data.
- The output is **aggregated to 1 s** using mean, max and p95. Samples are not simply dropped (decimated).
- Every trace is reproducible from its `seed` and a `params.json` file.

```
scenario scheduler ─► per-UE traffic generators ─► per-UE channel (CQI → spectral efficiency)
        │                                                     │
        └──────────────► PRB demand per UE = bitrate / (PRB capacity × SE) ◄──┘
                                          │
                      slice sum ─► simple allocator (162 PRB) ─► granted PRBs, unmet demand, latency proxy
                                          │
                               100 ms → 1 s aggregation ─► Parquet trace + metadata
```

### 1.2 Converting traffic to PRBs
PRB capacity per second:

```
bits_per_PRB_per_s = RE_per_PRB_per_slot × slots_per_s × SE(CQI) × DL_ratio
                   = (12 × 14 × (1 − overhead)) × 2000 × SE × DL_ratio
PRB_req(UE)        = bitrate_bps / bits_per_PRB_per_s
```
- The defaults are 14% overhead, a TDD downlink ratio of about 0.7, and SE taken from the CQI table. At peak SE this gives about 1.1 Mbps per PRB, and about 180 Mbps for all 162 PRBs.
- **These numbers must be calibrated against the real testbed.** Run a short iperf3 test per slice and note the achieved Mbps and PRBs used. All values live in `configs/sim.yaml`.

### 1.3 Traffic models per slice (3 UEs each)

| Slice | Traffic model | What varies on a 5–15 s scale |
|---|---|---|
| **eMBB** (video, bulk transfer) | Video at adaptive bitrate levels {2, 5, 8, 15, 25} Mbps, plus iperf-style bulk sessions that switch ON and OFF | Bitrate switches, sessions lasting 5–60 s, ramps in quality |
| **URLLC** (control) | Periodic small packets at a low baseline (about 0.2–1 Mbps), plus event bursts of 2–10 s reaching 5–10 Mbps, with a latency headroom factor | When bursts start, how long they last, and clusters of bursts |
| **mMTC** (telemetry) | Each UE stands for a group of devices: active devices × publish rate × payload, plus synchronized reporting waves every 30–60 s | Devices joining and leaving, reporting waves |

**Channel:** each UE's CQI follows a slow random process (AR(1)) with occasional drops of 5–20 s. A lower CQI means more PRBs are needed for the same bitrate, so demand varies even when traffic is constant.

### 1.4 Scenario scheduler
A Markov chain moves between scenarios. Each scenario lasts 10–120 s, drawn at random. Every timestamp is labelled with its scenario.

| Scenario | Description |
|---|---|
| `stable` | Steady load with small noise |
| `ramp_up` / `ramp_down` | Linear or S-shaped change in load over 5–60 s |
| `burst` | Short spikes, mainly on URLLC |
| `ue_churn` | UEs switch on and off |
| `contention` | All three slices peak at the same time (total demand exceeds 162 PRBs) |
| `sla_conflict` | The URLLC burst arrives while eMBB is high |

**Domain randomization:** each trace draws its own regime lengths, burst sizes, scenario transition probabilities, noise levels and CQI dynamics. This stops the model from simply learning one fixed generator.

### 1.5 Allocator (used only to produce "granted" values)
- Two simple modes: a static 1/3 split, or proportional sharing with minimum and maximum PRBs per slice.
- It produces `granted_prb`, `unmet_prb`, `utilization` and a latency proxy based on queueing (buffer divided by served rate).
- These are logged features only. **The prediction target is still requested PRBs.**

### 1.6 Output format
One Parquet file per trace, named `data/raw/trace_<id>.parquet`, with one row per second.

| Column | Per slice (`embb_`, `urllc_`, `mmtc_`) |
|---|---|
| `t` (seconds), `trace_id`, `scenario` | – |
| `req_prb_mean`, `req_prb_max`, `req_prb_p95` | ✔ |
| `granted_prb`, `unmet_prb` | ✔ |
| `active_ues`, `thr_mbps`, `cqi_mean` | ✔ |
| `buffer_kb`, `latency_ms_p95` | ✔ |

Each trace also gets `data/raw/trace_<id>.json`, holding the seed, every parameter, the simulator version and the scenario timeline.

### 1.7 Dataset size and splits
- **120 traces of 1 hour each, giving about 432k seconds.** The total size is configurable.
- **Split by trace, never by shuffled windows:** 84 traces for training, 18 for validation and 18 for testing.
- **An extra out-of-distribution (OOD) test set:** 10 traces whose parameters lie outside the training ranges, such as longer bursts or a different CQI process. This checks whether the model generalizes beyond the generator.

### 1.8 Dataset checks (Phase 1 is done only when all pass)
1. Plots per slice and scenario look realistic, and the PRB ranges match the testbed calibration.
2. **Autocorrelation of `req_prb_mean` at lags 5–15 s is clearly above 0** for every slice. Otherwise there is nothing to predict at those horizons.
3. The **persistence baseline** (predict the last observed value) gives a non-trivial but imperfect R² at 5, 10 and 15 s. This shows there is room for the model to add value.
4. In `contention` scenarios, total demand exceeds 162 PRBs for part of the time.

---

## Phase 2: Preprocessing and windowing
PreNS steps, adapted to 1 s sampling:

| PreNS step | What we do |
|---|---|
| Unit-time smoothing (spreads each request over a 10–100 ms latency window) | **Dropped.** That window is smaller than one 1 s step. The aggregation in Phase 1 does this job. |
| Linear interpolation for gaps of up to 2 steps | Kept (gaps of up to 2 s). Matters mainly for real testbed traces. |
| Median filter, size 3 | Kept, but **configurable**, because it may smooth away real URLLC bursts at 1 s. Test it both on and off. |
| Min-max normalization per slice | Kept. **The scaler is fitted on the training traces only** and saved with the model. |

`prep/windows.py` provides `build_windows(df, lookback L, horizons, features, target)`:
- Input `X[t] = features[t−L+1 … t]`
- Output `Y[t] = target[t+h]` for each h in the chosen horizons
- **Windows never cross a trace boundary.**
- Default lookback **L = 30 s**, tuned over {20, 30, 60}. PreNS's look-back is 2.5× its horizon, so we use at least 2× the longest horizon.

Feature sets:
- **`uni`** (as in the paper): the slice's own `req_prb_mean` only, with one model per slice.
- **`multi`** (our extension): requested PRB mean and max, active UEs, throughput, CQI and granted PRBs for all slices. One joint model outputs 3 slices × K horizons.

---

## Phase 3: Models, training and evaluation

### 3.1 Models (`models/`)
| Model | Notes |
|---|---|
| **PreDP attention Bi-LSTM** | As in the paper: 1 Bi-LSTM layer with 64 units per direction, additive attention with softmax, and a fully connected layer with K outputs |
| UT-LSTM | 2-layer LSTM |
| UT-GRU | 2-layer GRU |
| UT-Transformer | Small encoder with 2 layers |
| ARIMA | statsmodels, fitted for each slice |
| **Persistence** | ŷ(t+h) = y(t). The main baseline. |
| Moving average | Mean of the last L values |

### 3.2 Training
- Hyperparameters as in PreNS: Adam with lr 1e-3, MSE loss, batch 32, 64 hidden units.
- **Early stopping** (at most 200 epochs, patience 20 on validation loss) instead of a fixed 600 epochs, because our dataset is much larger.
- **Two horizon modes:**
  - `all`: K = 15 outputs (horizons 1–15 s). This is the default.
  - `dedicated`: K = 3 outputs (the horizons chosen in the UI).
- Grid search over the PreNS ranges: hidden size {32, 64, 128}, lr [5e-4 … 5e-3], batch {16, 32, 64}, lookback {20, 30, 60}.
- **Repeat each experiment with 5 seeds** and report mean ± std, as PreNS does.

### 3.3 Evaluation (`evaluate.py`)
- **RMSE, MAE and R²** for each slice and **each horizon**, in normalized units and in PRBs.
- **Improvement over persistence** for each horizon: `1 − RMSE_model / RMSE_persistence`.
- Errors by scenario (`stable`, `burst`, `contention`, …) and on the OOD test set.
- A cost table like PreNS Table III: number of parameters, training time, inference time, GPU memory.
- A plot of how accuracy falls as the horizon grows from 5 to 15 s.

### 3.4 Confidence output (needed for the A1 package)
- **v1 (PreNS-style):** use the rolling RMSE over the last W predictions at each horizon. Map it to confidence with `C_h = exp(−rmse_h / τ_h)`, where τ_h is set on the validation data.
- v2 (optional): MC-dropout variance or a Gaussian output with NLL loss.
- Check calibration: when confidence is high, the measured error should be low.

### 3.5 Planned experiments
| # | Experiment | Question it answers |
|---|---|---|
| E1 | PreDP (`uni`, dedicated 5/10/15) compared with all baselines | Can we reproduce a PreNS-style comparison at 1 s sampling? |
| E2 | `all` (1–15 s) compared with `dedicated` on 5/10/15 and 7/9/12 | Is training once to support every horizon as accurate as a dedicated model? |
| E3 | `uni` compared with `multi` features | Do the extra KPIs help? |
| E4 | Lookback 20, 30 and 60 s | How much history is needed? |
| E5 | Median filter on and off | Does the filter remove real bursts? |
| E6 | In-distribution test compared with the OOD test | How well does the model generalize? |
| E7 (later) | Simulator only compared with simulator plus testbed fine-tuning | How large is the gap between simulated and real traffic? |

---

## Phase 4: Model registry
Each training run creates `runs/<run_id>/` containing:
```
model.pt        trained weights
config.json     horizons, horizon mode, lookback, features, model type, sim version, data split
scaler.json     min/max per feature (fitted on train only)
metrics.json    all evaluation results
train_log.csv   loss per epoch
```
Inference loads only from the registry, so the horizons and scaling always match the trained model.

---

## Phase 5: UI (React + FastAPI)

Backend `api/server.py` (FastAPI) wraps the Python code; frontend `ui/` (React + TypeScript, Vite, Tailwind, Recharts). Start with `./run_ui.sh` → http://127.0.0.1:8765.

| Page | Features |
|---|---|
| **1. Dataset** | Generate traces (number of traces, hours, seed, scenario mix, randomization ranges); preview a trace per slice with scenario shading; view dataset statistics, the autocorrelation check and the persistence check |
| **2. Train** | **H1 / H2 / H3 sliders (5–15 s, checked so H1 < H2 < H3)**; horizon mode `all` or `dedicated`; lookback; `uni` or `multi`; model type; hyperparameters; start training with live loss curves |
| **3. Evaluate** | Pick a run; metrics table per slice and horizon; comparison with baselines; predicted vs actual plots; errors by scenario |
| **4. Inference** | Pick a run and the H1/H2/H3 to use (any values from 1–15 if the run is `all`); replay a test trace second by second; plot the forecasts at t+H1, t+H2 and t+H3 against the actual values; **export A1 JSON packages** |
| *(optional)* Delay check | Load measured control-loop delay statistics, and warn if H1 < p95 delay + M_min |

Example A1 package:
```json
{"t_pred": 1712.0, "slice": "urllc", "horizon_s": 7,
 "target_time": 1719.0, "pred_prb": 23.4, "confidence": 0.82, "run_id": "..."}
```

---

## Phase 6 (later): Testbed fine-tuning
1. Collect live KPIs through E2 KPM with a **1000 ms reporting period**. Use the same column names as the simulator so the same code works.
2. Use the same preprocessing and windows, and fine-tune with a low learning rate. Optionally freeze the Bi-LSTM layer and train only the attention and output layers.
3. Report results three ways: simulator only, after fine-tuning, and evaluated on held-out testbed traces.

---

## Project layout
```
Data genaration/
├── PLAN.md
├── configs/        sim.yaml, train.yaml
├── datagen/        simulator.py, traffic.py, channel.py, scenarios.py, allocator.py, aggregate.py
├── prep/           clean.py, scaler.py, windows.py
├── models/         predp_bilstm_attn.py, baselines.py
├── train.py · evaluate.py · infer.py
├── api/            server.py (FastAPI), jobs.py
├── ui/             React app (src/pages: Overview, Dataset, Train, Evaluate, Inference)
├── run_ui.sh       start backend + built UI
├── data/raw/       trace_*.parquet + trace_*.json
├── runs/           model registry
└── document/       papers and reports
```
Tech stack: Python 3.10+, numpy, pandas, pyarrow, PyTorch, statsmodels, PyYAML, FastAPI, Uvicorn; Node 24 + React 19, Vite, Tailwind 4, Recharts.

---

## Order of work and done criteria

| Step | Deliverable | Done when |
|---|---|---|
| 1 | Simulator and calibration config | One trace generated and the plots look right |
| 2 | Full dataset (120 h + OOD set) | All Phase 1.8 checks pass |
| 3 | Preprocessing and `build_windows` | Unit tests pass, including the check that no window crosses a trace boundary |
| 4 | PreDP and training script (CLI) | Training on 5/10/15 converges and beats persistence at 5 s |
| 5 | Baselines and evaluation | E1 table produced |
| 6 | `all` vs `dedicated`, uni vs multi | Results for E2 and E3 |
| 7 | Registry and inference script | `infer.py --run X --h 7 9 12` writes A1 JSON |
| 8 | React + FastAPI UI | All 4 pages work end to end from the UI |
| 9 | Confidence output | Calibration plot produced |
| 10 | Testbed fine-tuning | Result for E7 |

---

## 4-day schedule (steps 1–9)
Hardware: RTX 4060 Laptop (8 GB), 20 CPU cores, 30 GB RAM. PyTorch must be switched to the CUDA build.
Training runs continue in the background and overnight while the next day's code is written.

| Day | Build | Runs in the background / overnight | Done when |
|---|---|---|---|
| **1** | CUDA PyTorch · simulator (estimated calibration) · generate 130 h · dataset checks · `build_windows` with tests · PreDP and `train.py` smoke test | Small grid search on the 25 h subset | Dataset checks pass; PreDP 5/10/15 beats persistence at 5 s |
| **2** | Baselines (LSTM, GRU, Transformer, ARIMA on a subset, persistence, moving average) · `evaluate.py` · registry · `infer.py` with A1 JSON | E1 (all models, 3 slices, 3 seeds) and E2 (all vs dedicated on 5/10/15 and 7/9/12) | E1 table produced; `infer.py --h 7 9 12` works |
| **3** | React + FastAPI UI: Dataset, Train, Evaluate and Inference pages | E3 (multi features), E4 (lookback 20/60), E5 (filter off), E8 (learning curve 10/25/50/100 h) | Full flow from the UI: generate → train → evaluate → infer |
| **4** | Confidence v1 (rolling RMSE) and calibration plot · confidence shown in UI and A1 export · E6 OOD evaluation · final results tables · README · buffer | – | Steps 1–9 complete |

Cuts compared with the full plan, to fit 4 days:
- **3 seeds instead of 5,** and 5 seeds only for the final PreDP configuration if time allows.
- **Small grid search:** hidden {32, 64, 128} × lr {5e-4, 1e-3, 2e-3}, run on one slice with 1 seed.
- **ARIMA on a subset** of test windows.
- **Confidence v1 only.** MC-dropout is left for later.
- **A simple Evaluate page** in the UI.
- **Default simulator calibration.** It gets updated when the iperf3 test on the testbed is done, and regenerating takes minutes.
- About 90 training runs in total. 3–4 run in parallel on the GPU.

Your review each day (about 1 h): Day 1, check that the simulated traces look realistic. Days 2–4, check the results and the UI.

---

## Open items
- Calibrate PRB capacity per Mbps and the CQI range from a short iperf3 test on the testbed.
- Confirm the per-slice minimum and maximum PRBs used by the existing xApp, for the allocator.
- Expect lower R² at 5–15 s than the R² of 0.576 PreNS reports at 200 ms. How accuracy falls with the horizon is itself a result to report.
