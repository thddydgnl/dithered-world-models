## E11 (DINO-WM)

| 칸 | 변형 | n | 성공 | D %p [95% CI] | red [95% CI] | D_v − D_det %p [CI] | 1-step | 5-step | drift 비 |
|---|---|---|---|---|---|---|---|---|---|
| PushT k=4 | det | 100 | 69% | +4 [-7, 15] | — | — | 0.0598 | 0.3046 | nan |
| PushT k=4 | det+BC | 100 | 57% | +16 [5, 27] | -3.00 [-15.00, -0.42] | +12 [5, 20] | 0.0609 | 0.3843 | 1.19 |
| PushT k=4 | sh-plan | 100 | 69% | +4 [-3, 11] | 0.00 [-4.00, 1.50] | +0 [-11, 10] | 0.0657 | 0.0420 | 0.01 |
| PushT k=4 | det+shift-fixed | 100 | 71% | +2 [-3, 8] | 0.50 [-2.00, 2.00] | -2 [-13, 9] | 0.0322 | 0.0218 | 0.01 |
| PushT k=4 | sh-iter+b | 100 | 69% | +4 [-5, 13] | 0.00 [-5.00, 2.07] | +0 [-12, 12] | 0.0844 | 0.1147 | 0.02 |
| PushT k=4 | sh-iter | 100 | 75% | -2 [-8, 4] | 1.50 [0.00, 5.00] | -6 [-16, 4] | 0.0658 | 0.0428 | 0.00 |
| PushT k=4 | det+shift | 100 | 71% | +2 [-5, 9] | 0.50 [-2.00, 2.50] | -2 [-12, 8] | 0.0324 | 0.0220 | 0.01 |
| PushT k=4 | det+noise | 100 | 67% | +6 [-4, 16] | -0.50 [-4.58, 1.00] | +2 [-5, 9] | 0.0900 | 0.3239 | 1.00 |
| PointMaze k=3 | det | 100 | 70% | +9 [0, 18] | — | — | 0.3764 | 1.0764 | nan |
| PointMaze k=3 | det+BC | 100 | 80% | -1 [-8, 7] | 1.11 [0.25, 4.00] | -10 [-19, -1] | 0.3021 | 0.8278 | 0.23 |
| PointMaze k=3 | sh-plan | 100 | 83% | -4 [-11, 3] | 1.44 [0.77, 5.50] | -13 [-22, -5] | 0.4061 | 0.4775 | -0.15 |
| PointMaze k=3 | det+shift-fixed | 100 | 85% | -6 [-12, -1] | 1.67 [1.06, 6.50] | -15 [-23, -7] | 0.2028 | 0.2959 | -0.02 |
| PointMaze k=3 | sh-iter+b | 100 | 79% | +0 [-8, 8] | 1.00 [0.17, 3.67] | -9 [-18, 0] | 0.6487 | 0.8923 | 0.03 |
| PointMaze k=3 | sh-iter | 100 | 83% | -4 [-11, 3] | 1.44 [0.79, 5.50] | -13 [-21, -5] | 0.4070 | 0.4705 | -0.19 |
| PointMaze k=3 | det+shift | 100 | 85% | -6 [-13, 0] | 1.67 [0.95, 7.00] | -15 [-23, -7] | 0.2035 | 0.2987 | -0.11 |
| PointMaze k=3 | det+noise | 100 | 66% | +13 [3, 23] | -0.44 [-4.50, 0.40] | +4 [-5, 13] | 0.6200 | 1.1415 | 1.00 |
| Wall k=3 | det | 100 | 7% | +42 [31, 53] | — | — | 0.3408 | 3.1187 | nan |
| Wall k=3 | det+BC | 100 | 37% | +12 [1, 23] | 0.71 [0.50, 0.97] | -30 [-40, -20] | 0.1306 | 0.2638 | 0.15 |
| Wall k=3 | sh-plan | 100 | 53% | -4 [-14, 6] | 1.10 [0.88, 1.39] | -46 [-56, -36] | 0.2054 | 0.1490 | 0.05 |
| Wall k=3 | det+shift-fixed | 100 | 47% | +2 [-8, 12] | 0.95 [0.74, 1.23] | -40 [-51, -29] | 0.0927 | 0.0660 | 0.02 |
| Wall k=3 | sh-iter+b | 100 | 25% | +24 [12, 36] | 0.43 [0.24, 0.67] | -18 [-27, -10] | 0.4394 | 0.9686 | 0.40 |
| Wall k=3 | sh-iter | 100 | 49% | +0 [-10, 10] | 1.00 [0.78, 1.30] | -42 [-52, -32] | 0.2092 | 0.1484 | 0.04 |
| Wall k=3 | det+shift | 100 | 64% | -15 [-25, -4] | 1.36 [1.09, 1.74] | -57 [-67, -46] | 0.0929 | 0.0665 | 0.02 |
| Wall k=3 | det+noise | 100 | 10% | +39 [28, 49] | 0.07 [-0.07, 0.21] | -3 [-9, 3] | 0.4734 | 4.4752 | 1.29 |

- 손상 칸(Holm, α 0.05): [('wall', 3)]; 단위 읽기: **bias**; 칸별 {'wall': 'bias'}
- B2: True · B3(DINO-WM): True (det+noise가 고친 칸 0/1)
- PushT: Δ_bias -12 [-20, -5], Δ_pers +0 [-6, 6]; 보조 D(sh-plan)-D(sh-iter) +6 [-1, 13]; D(sh-iter+b)-D(sh-iter) +6 [-2, 14]; D(det+noise)-D(det) +2 [-5, 9]; D(det+shift)-D(sh-iter) +4 [-2, 10]
- PointMaze: Δ_bias +10 [1, 19], Δ_pers +0 [-6, 6]; 보조 D(sh-plan)-D(sh-iter) +0 [-7, 7]; D(sh-iter+b)-D(sh-iter) +4 [-2, 10]; D(det+noise)-D(det) +4 [-5, 13]; D(det+shift)-D(sh-iter) -2 [-8, 4]
- Wall: Δ_bias +30 [20, 40], Δ_pers +17 [7, 27]; 보조 D(sh-plan)-D(sh-iter) -4 [-12, 4]; D(sh-iter+b)-D(sh-iter) +24 [12, 36]; D(det+noise)-D(det) -3 [-9, 3]; D(det+shift)-D(sh-iter) -15 [-25, -5]

## Dg1

- 단위 평균 차 0.26190476190476186 → False
  - {'task': 'wall', 'k': 3, 'n': 8, 'rho_pr_D': 0.9047619047619048, 'rho_lat1_D': 0.6428571428571429, 'diff': 0.26190476190476186, 'aux_rho_lat1_last_D': 0.7142857142857144, 'aux_diff_last': 0.19047619047619035}

## E12

- PushT: k* main|det 4, main|detbc 5, main|shiter 2, pc|det 4, pc|detbc 4, pc|shiter 2, rc|det 4, rc|shiter 2; G {"main": {"G": 2, "det_side": 4, "shiter": 2}, "pc": {"G": 2, "det_side": 4, "shiter": 2}, "rc": {"G": 2, "det_side": 4, "shiter": 2}}; 최선 ('main|shiter', 2) two-stage cells (PushT): CI not adjusted for the interim look
- PointMaze: k* main|det 2, main|detbc 2, main|shiter 2, pc|det 2, pc|detbc 2, pc|shiter 2; G {"main": {"G": 0, "det_side": 2, "shiter": 2}, "pc": {"G": 0, "det_side": 2, "shiter": 2}}; 최선 ('main|det', 2) 
- Wall: k* main|det 4, main|detbc 4, main|shiter 2, pc|det 4, pc|detbc 4, pc|shiter 2; G {"main": {"G": 2, "det_side": 4, "shiter": 2}, "pc": {"G": 2, "det_side": 4, "shiter": 2}}; 최선 ('main|shiter', 2) 

## E16

- PointMaze: c* 2.0, none MPPI 81% / CEM 79% (자격 True); D(sh-plan) − D(sh-iter): MPPI +4 [-1, 10], CEM +0 [-7, 7]
- Wall: c* 1.0, none MPPI 63% / CEM 49% (자격 True); D(sh-plan) − D(sh-iter): MPPI -7 [-18, 4], CEM -4 [-12, 4]
