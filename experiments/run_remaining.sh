#!/usr/bin/env bash
# Overnight pipeline for the remaining experiments (PLAN.md 3.5).
#   1. E6  OOD evaluation of the final model (no training)
#   2. E1 seed 1, one job at a time  -> clean, unshared-GPU cost numbers (train time, latency, memory)
#   3. E3, E4, E5, E8 (seed 0) + E1 seed 2, 3 jobs in parallel
#   4. Evaluate every new experiment against its reference run
# Progress: runs/_logs/remaining.log    (re-running skips finished runs)
set -u
cd "$(dirname "$0")/.."
PY=python3
BASE=predp_uni_all_5-10-15_full
stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "E6: OOD evaluation"
$PY evaluate.py --runs $BASE --split ood --arima-n 1000 --tag E6_ood

stamp "E1 seed 1 (sequential, clean cost)"
$PY experiments/run_queue.py --exp E1 --seeds 1 --parallel 1

stamp "E3 E4 E5 E8 seed 0 + E1 seed 2 (parallel 3)"
$PY experiments/run_queue.py --exp E3 E4 E5 E8 --seeds 0 --parallel 3
$PY experiments/run_queue.py --exp E1 --seeds 2 --parallel 3

stamp "Evaluations"
ev() { tag=$1; shift; $PY evaluate.py --runs $BASE "$@" --split test --arima-n 0 --tag "$tag"; }
ev E3_features      E3/predp_multi_all_s0
ev E4_lookback      E4/predp_L20_s0 E4/predp_L60_s0
ev E5_filter        E5/predp_nofilter_s0
ev E8_learning_curve E8/predp_10h_s0 E8/predp_25h_s0 E8/predp_50h_s0
ev E1_seeds         E1/lstm_uni_all_s0 E1/gru_uni_all_s0 E1/transformer_uni_all_s0 \
                    E1/predp_uni_all_s1 E1/lstm_uni_all_s1 E1/gru_uni_all_s1 E1/transformer_uni_all_s1 \
                    E1/predp_uni_all_s2 E1/lstm_uni_all_s2 E1/gru_uni_all_s2 E1/transformer_uni_all_s2

stamp "ALL DONE"
