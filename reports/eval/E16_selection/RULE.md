# E16 selection rule (written before any E16 model was trained)

Candidates: eMBB DL and eMBB UL models trained with a quantile (pinball) loss at q = 0.55, 0.60, 0.65.
Same settings as E15 otherwise (data/raw_v2, no median filter, 60 s lookback, all-horizon outputs).
Evaluated on the 18 val traces of data/raw_v2, separately for DL and UL.

A candidate q qualifies for a direction if BOTH hold:
1. **Lab horizon rule** (experiments/horizon_rule_check.py): at every horizon 5, 6, 7, 8, 9, 10, 12 s the eMBB group
   has pooled MAE < persistence, p95 |error| < persistence, and wins (lower MAE than persistence) on >= 10 of 18 traces.
2. **No extra under-reservation**: eMBB shortfall = mean(max(actual - forecast, 0)) is no larger than persistence's
   at 5, 8 and 12 s.

Choice per direction: the **lowest qualifying q** (closest to the median, so the best MAE).
Confirmation: the chosen q is retrained with seeds 1 and 2; both must also qualify.
If no q qualifies in a direction, keep v2 (q = 0.5) for that direction and handle under-reservation with
an rApp headroom policy instead (reported as such).
