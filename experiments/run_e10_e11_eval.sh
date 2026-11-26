#!/usr/bin/env bash
# Evaluate E10 (mean / max targets) and E11 (quantile) after their training queue (pid 1795320) ends.
cd "$(dirname "$0")/.."
while kill -0 1795320 2>/dev/null; do sleep 60; done
B=predp_uni_all_5-10-15_full
echo "[$(date '+%F %T')] evaluating E10/E11"
python3 evaluate.py --runs E10/predp_mean_s0 --split test --arima-n 1000 --tag E10_mean
python3 evaluate.py --runs E10/predp_max_s0  --split test --arima-n 1000 --tag E10_max
python3 evaluate.py --runs $B E11/predp_q90_s0 --split test --arima-n 0 --tag E11_q90
python3 evaluate.py --runs E10/predp_max_s0 E11/predp_max_q90_s0 --split test --arima-n 0 --tag E11_max_q90
echo "[$(date '+%F %T')] E10/E11 evaluations DONE"
