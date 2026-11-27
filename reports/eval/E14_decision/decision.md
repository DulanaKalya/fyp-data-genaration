# E14 — one model (DL + UL history) or two separate models?

Skill = RMSE reduction vs persistence. `pair` = the slice's DL and UL history as input; `uni` = own direction only. Mean ± std over 3 seeds. A difference counts only if it is larger than 2 × the seed spread (min 0.2 %).

## EMBB → **2 separate models**

A gain > noise: no · B other direction not worse: yes · C OOD not worse: no

| split | direction | h | uni skill | pair skill | gain | noise (2×spread) |
|---|---|---|---|---|---|---|
| test | UL | 5 s | +5.7% ± 0.4 | +6.0% ± 0.2 | +0.3% | 0.9% |
| test | UL | 10 s | +9.3% ± 0.4 | +9.2% ± 0.2 | -0.1% | 0.9% |
| test | UL | 15 s | +10.5% ± 0.2 | +10.6% ± 0.1 | +0.1% | 0.5% |
| test | DL | 5 s | +3.5% ± 0.1 | +3.7% ± 0.2 | +0.2% | 0.4% |
| test | DL | 10 s | +6.2% ± 0.1 | +6.3% ± 0.1 | +0.1% | 0.4% |
| test | DL | 15 s | +8.3% ± 0.2 | +8.5% ± 0.1 | +0.2% | 0.4% |
| ood | UL | 5 s | +1.0% ± 2.0 | +1.6% ± 1.8 | +0.6% | 4.1% |
| ood | UL | 10 s | +1.8% ± 1.7 | +2.4% ± 2.3 | +0.6% | 4.5% |
| ood | UL | 15 s | +3.0% ± 0.1 | +3.1% ± 2.2 | +0.1% | 4.5% |
| ood | DL | 5 s | -0.9% ± 2.1 | -2.7% ± 1.4 | -1.8% | 4.2% |
| ood | DL | 10 s | -2.1% ± 1.3 | -5.4% ± 1.0 | -3.2% ❌ | 2.6% |
| ood | DL | 15 s | -3.1% ± 2.0 | -6.7% ± 1.4 | -3.6% | 4.0% |

## URLLC → **1 model (pair)**

A gain > noise: yes · B other direction not worse: yes · C OOD not worse: yes

| split | direction | h | uni skill | pair skill | gain | noise (2×spread) |
|---|---|---|---|---|---|---|
| test | UL | 5 s | +6.6% ± 0.2 | +7.8% ± 0.3 | +1.2% ✅ | 0.7% |
| test | UL | 10 s | +10.6% ± 0.2 | +11.6% ± 0.5 | +1.0% ✅ | 0.9% |
| test | UL | 15 s | +12.8% ± 0.3 | +13.7% ± 0.2 | +0.9% ✅ | 0.5% |
| test | DL | 5 s | +11.2% ± 0.2 | +15.1% ± 0.2 | +3.9% ✅ | 0.5% |
| test | DL | 10 s | +17.2% ± 0.2 | +21.2% ± 0.4 | +4.0% ✅ | 0.8% |
| test | DL | 15 s | +19.4% ± 0.2 | +22.1% ± 0.5 | +2.7% ✅ | 1.0% |
| ood | UL | 5 s | -6.8% ± 1.3 | -15.6% ± 8.3 | -8.8% | 16.6% |
| ood | UL | 10 s | -6.5% ± 2.1 | -17.0% ± 9.5 | -10.6% | 18.9% |
| ood | UL | 15 s | -2.6% ± 1.1 | -10.8% ± 8.2 | -8.2% | 16.4% |
| ood | DL | 5 s | -35.6% ± 0.7 | -33.8% ± 6.4 | +1.8% | 12.7% |
| ood | DL | 10 s | -30.8% ± 4.9 | -26.3% ± 3.4 | +4.5% | 9.8% |
| ood | DL | 15 s | -14.8% ± 3.3 | -14.3% ± 3.2 | +0.5% | 6.6% |

## MMTC → **2 separate models**

A gain > noise: no · B other direction not worse: yes · C OOD not worse: no

| split | direction | h | uni skill | pair skill | gain | noise (2×spread) |
|---|---|---|---|---|---|---|
| test | UL | 5 s | +5.6% ± 0.2 | +4.5% ± 1.3 | -1.0% | 2.7% |
| test | UL | 10 s | +8.4% ± 0.3 | +5.9% ± 3.1 | -2.5% | 6.1% |
| test | UL | 15 s | +9.8% ± 0.1 | +7.6% ± 2.3 | -2.2% | 4.7% |
| test | DL | 5 s | +10.2% ± 0.4 | +10.0% ± 0.5 | -0.3% | 0.9% |
| test | DL | 10 s | +14.7% ± 0.1 | +13.7% ± 0.5 | -1.0% | 1.1% |
| test | DL | 15 s | +15.5% ± 0.1 | +14.5% ± 0.5 | -1.0% | 1.1% |
| ood | UL | 5 s | -0.5% ± 0.1 | -7.5% ± 2.0 | -7.0% ❌ | 3.9% |
| ood | UL | 10 s | +1.3% ± 1.1 | -9.1% ± 6.6 | -10.5% | 13.2% |
| ood | UL | 15 s | +3.6% ± 1.5 | -5.4% ± 4.8 | -9.0% | 9.6% |
| ood | DL | 5 s | -0.4% ± 0.6 | -1.9% ± 3.4 | -1.5% | 6.7% |
| ood | DL | 10 s | +4.4% ± 0.2 | +0.0% ± 3.4 | -4.4% | 6.8% |
| ood | DL | 15 s | +9.2% ± 0.5 | +5.1% ± 2.6 | -4.1% | 5.2% |
