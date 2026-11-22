## [사후] PushT 3-bit E11 칸 (pod dw.6)

- none 73% (n 100); det 손상 실재(한쪽 p < 0.05, 이 칸 하나): True (p = 9.999e-05); 분류: resolution

| 변형 | 성공 | D %p [CI] | red [CI] | D_v − D_det %p [CI] | 1-step / 5-step | drift 비 | 덩어리 D CI |
|---|---|---|---|---|---|---|---|
| det | 38% | +35 [24, 45] | — | — | 0.3072 / 0.7390 | nan | [16, 58] |
| det+BC | 30% | +43 [33, 53] | -0.23 [-0.62, 0.03] | +8 [-1, 17] | 0.4239 / 0.9160 | 1.13 | [28, 60] |
| sh-plan | 70% | +3 [-4, 10] | 0.91 [0.71, 1.12] | -32 [-43, -21] | 0.3322 / 0.2078 | -0.04 | [-6, 13] |
| det+shift-fixed | 71% | +2 [-6, 10] | 0.94 [0.74, 1.20] | -33 [-43, -23] | 0.1476 / 0.0939 | -0.06 | [-5, 9] |
| sh-iter+b | 28% | +45 [34, 56] | -0.29 [-0.79, 0.05] | +10 [-2, 22] | 0.7439 / 1.2566 | -2.27 | [29, 60] |
| sh-iter | 75% | -2 [-9, 5] | 1.06 [0.85, 1.31] | -37 [-48, -26] | 0.3309 / 0.2109 | -0.00 | [-15, 10] |
| det+shift | 75% | -2 [-8, 4] | 1.06 [0.89, 1.27] | -37 [-48, -26] | 0.1484 / 0.0938 | -0.02 | [-10, 6] |
| det+noise | 32% | +41 [31, 51] | -0.17 [-0.37, -0.05] | +6 [2, 11] | 0.4008 / 0.7692 | 1.10 | [23, 64] |

- Δ_bias -8 [-17, 1], Δ_pers +4 [-3, 11]; 보조 D(sh-plan)-D(sh-iter) +5 [-3, 13]; D(sh-iter+b)-D(sh-iter) +47 [35, 59]; D(det+noise)-D(det) +6 [2, 11]; D(det+shift)-D(sh-iter) +0 [-7, 7]
- 덩어리 bootstrap(10덩어리): det D CI [16, 58], p 9.999e-05; Δ_bias [-21, 5], Δ_pers [-1, 9]; 덩어리 손상 True, 분류 resolution, 읽기 차이 False
- E1 같은 칸(eval6): none 83%, det 47%, D_det +36 [27, 45], red ind 0.97, sh-iter 0.92, sh-plan 0.89, det+shift 0.92 (A/main/results/E1/e1_dinowm_cells.csv (pusht, confirmatory, k=3))
- [사후 기술] DINO-WM 손상 칸(본 분석 Wall 3 + 이 칸): [{'task': 'wall', 'k': 3, 'cls': 'bias', 'detnoise_fixes': False}, {'task': 'pusht', 'k': 3, 'cls': 'resolution', 'detnoise_fixes': False}] → 읽기 mixed, det+noise가 고친 칸 0/2 (B2·B3·Dg1 판정은 바꾸지 않음)
