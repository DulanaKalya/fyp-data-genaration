# Comparison with PreNS (Wu et al., IEEE TCCN 2026) — Table III

| | **PreNS paper** | **This work** |
|---|---|---|
| Data | OAI 5G testbed (FlexRAN), 100 RBs | Testbed-matched simulator: n78, 60 MHz, 30 kHz SCS, **162 PRBs**, 3+3+3 UEs |
| Sampling | 100 ms | **1 s** (100 ms simulated, aggregated) |
| Input window | 5 steps = 500 ms | 30 steps = 30 s |
| Prediction horizon | 200 ms | **5 s, 10 s, 15 s** |
| Test data | not stated | 64,008 windows (18 h, 18 unseen traces) |
| Hardware | RTX 4090 + i9-13900K | RTX 4060 Laptop (shared by up to 6 jobs) |

R² is unit-free and the fairest comparison. RMSE/MAE are in z-score (standardized) units; PreNS values fit that scale (RMSE ≈ √(1−R²)). Our metrics are the mean over the 3 slices.

## Accuracy: PreNS (200 ms) vs this work (5 / 10 / 15 s)

| Model | PreNS R² | Our R² @5s | @10s | @15s | PreNS RMSE | Our RMSE @5/10/15 s | PreNS MAE | Our MAE @5/10/15 s |
|---|---|---|---|---|---|---|---|---|
| PreDP (attention Bi-LSTM) | 0.5762 | 0.692 | 0.545 | 0.465 | 0.6977 | 0.545 / 0.664 / 0.722 | 0.5141 | 0.355 / 0.443 / 0.494 |
| UT-LSTM | 0.4523 | 0.691 | 0.544 | 0.465 | 0.8677 | 0.547 / 0.665 / 0.722 | 0.6261 | 0.353 / 0.444 / 0.492 |
| UT-GRU | 0.5183 | 0.693 | 0.543 | 0.462 | 0.8234 | 0.545 / 0.666 / 0.724 | 0.5841 | 0.354 / 0.445 / 0.487 |
| UT-Transformer | 0.2813 | 0.693 | 0.546 | 0.463 | 1.0922 | 0.545 / 0.664 / 0.723 | 0.8243 | 0.354 / 0.444 / 0.492 |
| ARIMA | 0.1263 | 0.710 | 0.496 | 0.406 | 2.2645 | 0.531 / 0.702 / 0.763 | 1.5329 | 0.338 / 0.468 / 0.521 |
| Persistence ŷ(t+h)=y(t) *(not in PreNS)* | — | 0.666 | 0.416 | 0.274 | — | 0.566 / 0.748 / 0.835 | — | 0.351 / 0.485 / 0.552 |

ARIMA(2,0,1) uses a random 2000-window subset; persistence on the same subset: R² 0.681 / 0.397 / 0.240.

## Same-window comparison (2000 random test windows, all methods) — mean R²

| Model | 5 s | 10 s | 15 s |
|---|---|---|---|
| PreDP (attention Bi-LSTM) | 0.698 | **0.528** | **0.452** |
| UT-LSTM | 0.694 | 0.524 | 0.448 |
| UT-GRU | 0.699 | 0.521 | 0.445 |
| UT-Transformer | 0.698 | 0.526 | 0.449 |
| ARIMA | **0.710** | 0.496 | 0.406 |
| persistence | 0.681 | 0.397 | 0.240 |
| ma10 | 0.561 | 0.373 | 0.266 |

## Ranking of models

| Rank | PreNS (R² at 200 ms) | This work, mean R² over 5/10/15 s (same windows) |
|---|---|---|
| 1 | PreDP (attention Bi-LSTM) (0.576) | PreDP (attention Bi-LSTM) (0.559) |
| 2 | UT-GRU (0.518) | UT-Transformer (0.558) |
| 3 | UT-LSTM (0.452) | UT-GRU (0.555) |
| 4 | UT-Transformer (0.281) | UT-LSTM (0.555) |
| 5 | ARIMA (0.126) | ARIMA (0.537) |

## Per-slice R² (this work)

| Model | EMBB @5s | EMBB @10s | EMBB @15s | URLLC @5s | URLLC @10s | URLLC @15s | MMTC @5s | MMTC @10s | MMTC @15s |
|---|---|---|---|---|---|---|---|---|---|
| PreDP (attention Bi-LSTM) | 0.783 | 0.629 | 0.516 | 0.523 | 0.317 | 0.226 | 0.770 | 0.690 | 0.652 |
| UT-LSTM | 0.783 | 0.627 | 0.515 | 0.522 | 0.316 | 0.230 | 0.767 | 0.687 | 0.649 |
| UT-GRU | 0.783 | 0.630 | 0.516 | 0.527 | 0.312 | 0.222 | 0.768 | 0.686 | 0.648 |
| UT-Transformer | 0.783 | 0.628 | 0.513 | 0.528 | 0.318 | 0.226 | 0.767 | 0.691 | 0.650 |
| ARIMA | 0.810 | 0.627 | 0.524 | 0.570 | 0.268 | 0.161 | 0.750 | 0.591 | 0.534 |
| persistence | 0.794 | 0.600 | 0.442 | 0.471 | 0.057 | -0.144 | 0.734 | 0.591 | 0.524 |

## Cost

| Model | PreNS params | Our params (per slice) | PreNS train min | Our train min (3 slices) | PreNS infer ms | Our infer ms (batch 1) | PreNS peak GPU MB | Our peak GPU MB |
|---|---|---|---|---|---|---|---|---|
| PreDP (attention Bi-LSTM) | 35,844 | 44,559 | 10.6 | 40.3 | 0.467 | 0.13 | 31.09 | 1029.2 |
| UT-LSTM | 51,139 | 51,407 | 7.6 | 26.4 | 0.977 | 0.087 | 26.57 | 196.1 |
| UT-GRU | 38,403 | 38,799 | 4.6 | 25.4 | 0.55 | 0.074 | 20.39 | 547.9 |
| UT-Transformer | 283,067 | 84,431 | 31.2 | 56.0 | 0.603 | 0.267 | 54.07 | 236.1 |
| ARIMA | 13,443 | — | 2.1 | — | 2.015 | — | 18.07 | — |

Our training times are on 84 h of data with 3–6 jobs sharing one laptop GPU, so they are not comparable to PreNS's single-job RTX 4090 times. Peak GPU memory includes the whole dataset kept on the GPU.