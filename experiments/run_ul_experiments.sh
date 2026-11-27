#!/usr/bin/env bash
# Phase 4 of PLAN_UL.md: uplink models (E13) and 1-vs-2-model test (E14), then evaluations.
# Log: runs/_logs/ul_experiments.log   (re-running skips finished runs)
set -u
cd "$(dirname "$0")/.."
PY=python3
stamp() { echo "[$(date '+%F %T')] $*"; }

# Runs that would duplicate an existing run are links (queue.py skips linked run dirs):
#  - E14 UL uni seed 0 == E13 UL point seed 0 (same data, settings, seed)
#  - E14 DL uni seed 0 == E12 final DL model (same settings and seed; raw_v2 DL columns are bit-identical to raw)
mkdir -p runs/E14
[ -e runs/E14/predp_ul_uni_s0 ] || ln -s ../E13/predp_ul_point_s0 runs/E14/predp_ul_uni_s0
[ -e runs/E14/predp_dl_uni_s0 ] || ln -s ../E12/predp_final_point_s0 runs/E14/predp_dl_uni_s0

stamp "E13 + E14 seed 0 (parallel 3)"
$PY experiments/run_queue.py --exp E13 E14 --seeds 0 --parallel 3
stamp "E14 seeds 1, 2 (parallel 3)"
$PY experiments/run_queue.py --exp E14 --seeds 1 2 --parallel 3

stamp "Evaluations"
ev() { $PY evaluate.py "$@"; }
# E13: uplink models vs persistence / MA / ARIMA (point and peak targets are evaluated separately)
ev --runs E13/predp_ul_point_s0 E13/predp_ul_q90_s0 --data data/raw_v2 --split test --arima-n 1000 --tag E13_ul_point
ev --runs E13/predp_ul_max_s0 E13/predp_ul_max_q90_s0 --data data/raw_v2 --split test --arima-n 1000 --tag E13_ul_max
ev --runs E13/predp_ul_point_s0 --data data/raw_v2 --split test --horizons 1 2 3 --arima-n 0 --tag E13_ul_point_short
ev --runs E13/predp_ul_point_s0 --data data/raw_v2 --split ood --arima-n 500 --tag E13_ul_point
# E14: uni (own direction only) vs pair (DL + UL history), 3 seeds, test and OOD
for d in ul dl; do
  R="E14/predp_${d}_uni_s0 E14/predp_${d}_uni_s1 E14/predp_${d}_uni_s2 E14/predp_${d}_pair_s0 E14/predp_${d}_pair_s1 E14/predp_${d}_pair_s2"
  ev --runs $R --data data/raw_v2 --split test --arima-n 0 --tag E14_${d}
  ev --runs $R --data data/raw_v2 --split ood --arima-n 0 --tag E14_${d}
done
stamp "ALL DONE"
