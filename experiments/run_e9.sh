#!/usr/bin/env bash
# E9 (residual / level-invariant target): runs after run_remaining.sh finishes.
cd "$(dirname "$0")/.."
while kill -0 1353790 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] E9 start"
python3 experiments/queue.py --exp E9 --seeds 0 --parallel 1
python3 evaluate.py --runs predp_uni_all_5-10-15_full E9/predp_residual_s0 --split test --arima-n 0 --tag E9_residual
python3 evaluate.py --runs predp_uni_all_5-10-15_full E9/predp_residual_s0 --split ood  --arima-n 0 --tag E9_residual
echo "[$(date '+%F %T')] E9 DONE"
