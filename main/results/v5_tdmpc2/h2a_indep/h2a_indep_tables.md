# TD-MPC2 E1 H2a의 독립 draw판 재측정 (DESIGN_main v5.2 9.1, 사후 정정: 동결 판정 옆에 싣고 대신하지 않는다)

동일성 검사: job 162개, episode 810개, 다른 job 0개, 빠진 job 0개

| 과제 | k | det OEI_indep [CI] | sh-iter OEI_indep [CI] | sh-iter OEI(같은 draw) [CI] | 성립 |
|---|---|---|---|---|---|
| cheetah-run | 5 | -6.48 [-7.14, -5.85] | +0.10 [-0.06, +0.26] | -2.48 [-2.67, -2.29] | True |
| mw-pick-place | 3 | -78.57 [-87.82, -69.35] | -13.66 [-16.80, -10.35] | -39.02 [-42.27, -35.79] | False |
| mw-push | 3 | -84.62 [-95.31, -74.16] | -4.56 [-6.27, -2.84] | -19.97 [-22.95, -17.51] | False |

| 칸 | 변형 | 재실행 | OEI | OEI_indep | sel_last | sel_indep | drift | OEI_fresh | 선택 elite가 정책 slot |
|---|---|---|---|---|---|---|---|---|---|
| cheetah-run|k5 | det | False | -6.48 | -6.48 | — | — | — | — | — |
| cheetah-run|k5 | ind | True | -6.24 | +0.19 | -6.40 | +0.03 | +0.16 | — | 0.08 |
| cheetah-run|k5 | shiter | True | -2.48 | +0.10 | -2.51 | +0.07 | +0.03 | — | 0.08 |
| cheetah-run|k5 | shstep | False | -6.91 | -6.91 | — | — | — | — | — |
| cheetah-run|k5 | shep | False | -6.73 | -6.73 | — | — | — | — | — |
| cheetah-run|k5 | detshift | True | -1.98 | +0.03 | -2.00 | +0.00 | +0.03 | — | 0.08 |
| mw-pick-place|k3 | det | False | -78.57 | -78.57 | — | — | — | — | — |
| mw-pick-place|k3 | ind | True | -70.96 | -7.63 | -65.31 | -1.98 | -5.64 | — | 0.04 |
| mw-pick-place|k3 | shiter | True | -39.02 | -13.66 | -27.02 | -1.65 | -12.01 | — | 0.07 |
| mw-pick-place|k3 | shstep | False | -65.44 | -65.44 | — | — | — | — | — |
| mw-pick-place|k3 | shep | False | -77.43 | -77.43 | — | — | — | — | — |
| mw-pick-place|k3 | detshift | True | -25.65 | -8.81 | -16.97 | -0.12 | -8.68 | — | 0.06 |
| mw-push|k3 | det | False | -84.62 | -84.62 | — | — | — | — | — |
| mw-push|k3 | ind | True | -54.56 | -3.71 | -50.26 | +0.58 | -4.30 | — | 0.04 |
| mw-push|k3 | shiter | True | -19.97 | -4.56 | -15.73 | -0.33 | -4.24 | — | 0.07 |
| mw-push|k3 | shstep | False | -73.42 | -73.42 | — | — | — | — | — |
| mw-push|k3 | shep | False | -94.73 | -94.73 | — | — | — | — | — |
| mw-push|k3 | detshift | True | -13.18 | -3.93 | -9.94 | -0.69 | -3.24 | — | 0.07 |
