#!/usr/bin/env bash
# Evaluate the E12 final candidates once their training (queue pid 2145526) ends.
cd "$(dirname "$0")/.."
while kill -0 2145526 2>/dev/null; do sleep 60; done
B=predp_uni_all_5-10-15_full; F=E12/predp_final_point_s0
echo "[$(date '+%F %T')] evaluating E12"
python3 evaluate.py --runs $B $F --split test --arima-n 2000 --blend --tag E12_final_point
python3 evaluate.py --runs $B $F --horizons 1 2 3 --split test --arima-n 0 --tag E12_final_point_short
python3 evaluate.py --runs $B $F --split ood --arima-n 1000 --tag E12_final_point
python3 evaluate.py --runs E10/predp_max_s0 E12/predp_final_max_s0 --split test --arima-n 1000 --tag E12_final_max
echo "[$(date '+%F %T')] E12 evaluations DONE"
