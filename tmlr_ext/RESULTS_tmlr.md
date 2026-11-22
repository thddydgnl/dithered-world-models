# A TMLR 추가 실험(E11~E17) — 종합 결과 (조정자)

- 작성: 2026-11-22T01:27Z(UTC, 한국 시간 10:27), 조정자. 네 묶음(td, mt, k, dw)의 단계 B 보고와 조정자 재계산을 모아 판정을 한곳에 적는다.
- 기준: [DESIGN_tmlr.md](DESIGN_tmlr.md) v1.0(sha256 `36d69ea8…0cc4`, 2026-11-21T17:59Z 고정)과 고정 뒤 변경 D1~D3(같은 문서 §11). 운영 기록은 [LOG.md](LOG.md), 진행판은 [STATUS.md](STATUS.md).
- 출처: [TD_tdmpc2/REPORT.md](TD_tdmpc2/REPORT.md), [MT_317m/REPORT.md](MT_317m/REPORT.md), [K_kernel/REPORT.md](K_kernel/REPORT.md), [DW_dinowm/REPORT.md](DW_dinowm/REPORT.md)(분석 출력 `results/analysis/tables_dw.md`·`analysis_dw.json`, 작업자의 독립 재계산 `recount_dw.json`). 조정자 재계산 스크립트와 출력은 [coordinator/](coordinator/).
- 이 문서는 **판정 기록**이다. 원고의 주장은 [CLAIMS_tmlr.md](CLAIMS_tmlr.md)에 있다(2026-11-22T01:35Z 사용자 승인).
- 표기: D는 같은 짝의 none 대비 하락(MW·DINO-WM 성공률 차, DMC 상대 return 하락). red = (D_det − D_v)/D_det, "고친다"는 red ≥ 0.5. CI는 짝 bootstrap 10,000회 95% 백분위(RandomState(0)).

## 0. 판정 표

| 가설 | 내용(설계) | 판정 | 핵심 수치 [95% CI] | 출처 | 조정자 재계산 |
|---|---|---|---|---|---|
| **B1** | MW는 지속 지배 | **성립** | MW 손상 8/8칸. 분류 지속 5(pick-place, door-open, peg-insert, window-open, sweep-into)·편향 3(push, drawer-open, button-press)·해상도 0 | td `judgements.B1` | 일치(00:00Z) |
| **B2** | DINO-WM은 지속 지배가 아님(편향 또는 해상도 지배) | **성립**(편향 지배) | 손상 칸은 Wall 3-bit 하나(D_det +42점 [31, 53]). PushT 4-bit(p 0.274)와 PointMaze 3-bit(p 0.038)는 3칸 Holm을 넘지 못함. Wall: det+BC red 0.71 [0.50, 0.97], 고정 무편향 sh-plan 1.10·det+shift-fixed 0.95도 고침. Δ_pers +17점 [7, 27]이지만 분류 순서상 편향이 앞섬 | dw `analysis_dw.json` `E11.B2` | 일치(01:27Z) |
| **B3** | det+noise는 어느 단위에서도 손상 칸의 과반을 못 고친다 | **불성립** | MW에서 det+noise가 8/8칸을 고침(red 0.52~0.92). DMC 1/4(finger), DINO-WM 0/1(Wall red 0.07) | td `judgements.B3_td`, dw `E11.B3_dinowm` | 일치 |
| **B4** | MW 손상 칸의 75% 이상에서 sh-iter red ≥ 0.5, 해로운 칸 없음 | **성립** | 7/8(window-open 0.29), 해로운 sh-iter 0 | td `judgements.B4` | 일치 |
| **Fr1** | PT에서 MW 과제 75% 이상 G ≥ 1 | **성립** | 7/8(G: push 2, pick-place 0, door-open 3, drawer-open 2, button-press 2, peg-insert 2, window-open 2, sweep-into 2) | td `judgements.Fr1` | 작업자 원자료 대조 |
| **Fr2** | PC에서 MW 과제 50% 이상 G ≥ 1 | **성립** | 7/8(G: push 0, pick-place 1, door-open 1, drawer-open 1, button-press 1, peg-insert 2, window-open 2, sweep-into 1) | td `judgements.Fr2` | 작업자 원자료 대조 |
| **Dg1** | 칸 안 D 순위를 ρ_p × σ_e/σ_J가 잠재 1-step 오차보다 Spearman으로 0.3 이상 더 잘 맞힌다(단위별 평균) | MW **성립**, DMC **성립**, DINO-WM **불성립** | MW 평균 차 0.971(8칸 모두 양수, 0.66~1.43), DMC 0.387(cheetah 1.000, finger 0.905, walker −0.238, quadruped −0.119), DINO-WM 0.262(손상 칸 Wall 하나: 0.905 − 0.643) | td `judgements.Dg1`, dw `Dg1` | DINO-WM P1 쪽 Spearman 0.905를 보고된 변형별 값으로 다시 셈 |
| **P1** | 317M: 6과제 이상 det 하락 실재, 그중 75% 이상 sh-iter red ≥ 0.5 | **성립** | 손상 8/8(D_det 0.122~0.978), sh-iter red ≥ 0.5 8/8(0.69~1.31) | mt `e14_judgement.json` | 일치(00:19Z) |
| **P2** | 317M 손상 과제 과반에서 D(sh-ep) − D(sh-iter) CI 하한 > 0 | **성립** | 7/8(차 0.24~0.82). 예외 push +0.033 [0.000, 0.078] | mt `e14_judgement.json` | 일치 |
| **It1** | DMC 반복 12에서 나은 dither가 3/4과제 이상 red ≥ 0.5, 해로운 과제 없음 | **성립** | red_best(12): cheetah 0.914, quadruped 0.758, finger 1.000, walker −0.095 | td `judgements.It1` | — |
| **It2** | 3과제 이상에서 red가 반복 6 < 12 < 24로 단조 증가 | **성립** | 단조 3: cheetah 0.745 → 0.914 → 1.087, walker −0.225 → −0.095 → 0.207, quadruped 0.587 → 0.758 → 0.840(finger는 아님) | td `judgements.It2` | — |
| **K1** | det+shift 정수 kernel 지연 오버헤드가 6조건 모두 중앙값 ≤ 5% | **불성립**(4/6) | 5M N 512 +5.30% [5.29, 5.33], 317M N 512 +7.23% [7.11, 7.27]. N ≥ 2048은 0.64~3.63%. sh-iter는 6조건 모두 ≤ 2.01% | k `analysis_lat/e17_lat_summary.json` | — |
| **K2** | 정수 kernel det+shift가 4과제 모두 fake-quant(E11)와 같은 방향(D(det) − D(det+shift)의 부호 같고 CI 겹침) | **성립**(4/4) | 가짜 / 정수: push +0.433 [0.322, 0.544] / +0.433 [0.333, 0.533], pick-place +0.267 [0.144, 0.389] / +0.300 [0.178, 0.411], cheetah +0.298 [0.268, 0.329] / +0.303 [0.270, 0.335], walker −0.134 [−0.178, −0.089] / −0.140 [−0.184, −0.096] | td `td_k2_e11_det_minus_detshift.csv`, k `e17_summary.json` | 조정자 결합 판정(00:14Z) |
| **K3** | 미리 바꾼 BF16의 D가 12과제 모두 CI에 0 포함 | **불성립**(9/12) | door-open +5.6점 [1.1, 11.1], cheetah +2.08% [0.46, 4.26], finger −1.42% [−4.00, −0.04](BF16이 더 좋음) | k `e17_summary.json` | — |
| E16 2×2(기술) | D(sh-ep) − D(sh-iter): 모델 × 계획기 | 기술 | TD-MPC2: 기본 MPPI push +48.9 [38.9, 60.0], pick-place +28.9 [15.6, 42.2], cheetah +36.6 [32.5, 40.7] / CEM식 +47.8 [36.7, 58.9], +42.2 [28.9, 55.6], +18.7 [16.5, 20.9]. DINO-WM: CEM(기본, E11) PointMaze +0 [−7, 7], Wall −4 [−12, 4] / MPPI식 PointMaze +4 [−1, 10], Wall −7 [−18, 4]. 모두 CI에 0 포함, 두 과제 모두 자격(none MPPI/CEM 1.03·1.29) | td `E16`, dw `E16` | DINO-WM 쪽 일치 |

- B3은 "어느 단위에서도"라서 MW 하나가 불성립이면 전체가 불성립이다. DINO-WM 결과와 관계없다.
- D2(편향 벡터 규칙)의 두 읽기((가) 모든 칸, (나) 미수렴 칸 제외)는 td·mt·dw 모두 같다. 미수렴 칸이 없었다(td 576칸, mt 8과제, dw 21칸; dw는 모두 α = 1에서 수렴).

## 1. 무엇을 보여 주나 (설계에 미리 적은 쓰임대로)

### 1.1 기전: 두 성분 — TD-MPC2는 반복 사이의 지속, DINO-WM은 평균 편향 (E11, E14, E16)

- **TD-MPC2 5M(E11)**: 손상 12/12칸(MW 8, DMC 4). 분류는 MW 지속 5·편향 3·해상도 0, DMC 지속 3·편향 1(finger)이다. 설계 §1.5의 첫 행 조건 가운데 MW 쪽(B1)이 성립했다.
  - 지속의 비용 Δ_pers = D(det+shift-fixed) − D(det+shift)는 **12칸 모두** CI 하한 > 0이다(MW +10.0~+40.0점, DMC +14.7~+46.7%). 편향으로 분류된 칸에도 지속의 비용이 크다(push +40.0 [28.9, 51.1]).
  - 원소별 오차 분포가 같고 반복 사이에 고정된 무작위 오차(sh-ep, det+shift-fixed)는 MW에서 거의 못 고친다(고친 칸 1/8, 0/8). 해상도 설명(b)은 MW에서 기각된다.
  - D(sh-ep) − D(sh-iter)는 12칸 모두 양수다(+10.0~+61.1, CI 하한 > 0). sh-ep는 button-press, peg-insert, sweep-into에서 det보다 해롭다.
  - 평균 편향만 뺀 det+BC는 MW 3/8, DMC 1/4칸에서 손상의 절반 이상을 없앤다. 새로 뽑는 dither에 det의 평균 편향을 되돌려 넣은 sh-iter+b도 MW 5/8칸을 고친다. 편향은 성분의 하나지만 주된 몫은 아니다.
- **317M(E14)**: 손상 8/8, P2 성립(지속의 비용 7/8). det+BC는 4/8과제(push, drawer-open, button-press, window-open)를 고친다. "기전이 크기에 걸쳐 성립한다"(설계 §4의 쓰임).
- **계획기 교차(E16, TD 쪽)**: TD-MPC2를 CEM식(elite 균등 가중, min_std 0.01, 정책 궤적 0)으로 바꿔도 지속의 비용이 3과제 모두 남는다. TD-MPC2에서 지속이 문제가 되는 것은 MPPI 가중 방식 때문이 아니다. DINO-WM은 반대로, 기본 CEM에서도 MPPI식으로 바꿔도 D(sh-plan) − D(sh-iter)의 CI가 0을 포함한다(지속의 비용 없음). **어느 성분이 지배하는지는 계획기가 아니라 모델 계열을 따른다.**
- **DINO-WM(E11, 새 분할 eval8)**: 손상 칸은 Wall 3-bit 하나다(D_det +42점, E1 탐색 칸 +50점의 복제; det 성공 7% 대 E1 11%). PushT 4-bit는 E1의 손상을 복제하지 못했고(+4점 [−7, 15], E1 +24점 [14, 34]), PointMaze 3-bit는 점 추정이 비슷하지만(+9점 [0, 18], E1 +11점) 3칸 Holm을 넘지 못했다. eval8 PushT 쌍은 궤적 10개(덩어리 10개)에서 나왔고, 덩어리 bootstrap으로도 읽기는 같다(D1).
  - Wall은 설계 §1.3 예측표의 (a) 평균 편향 줄과 모두 맞는다: det+BC가 고치고(0.71), 고정 무편향 오차도 고치며(sh-plan 1.10, det+shift-fixed 0.95), det의 평균 편향을 되돌려 넣은 sh-iter+b는 sh-iter보다 24점 나쁘고(0.43), 편향을 그대로 둔 det+noise는 못 고친다(0.07).
  - 그래서 B2가 성립한다(편향 지배). 다만 이 읽기는 손상 칸 하나에 기댄다. Δ_pers > 0(+17점)은 det+shift가 none보다 15점 나은 데서 온다(64% 대 49%; Wall none은 D3대로 E1보다 낮다).
- **이중 해리**: 같은 변형이 두 계열에서 반대로 움직인다. det+noise는 MW 8/8칸을 고치지만 Wall은 못 고친다. det+BC는 Wall을 고치지만 MW는 3/8칸뿐이다. 고정 무편향 오차(sh-ep·sh-plan, det+shift-fixed)는 Wall을 고치지만 MW는 0~1/8칸이다. 설계 §1.5의 첫 행("B1·B2 성립, DINO-WM 편향 지배 → 기전을 두 성분으로 쓴다")에 해당한다.

### 1.2 예측과 다른 것: 새 잡음만 더해도 MW는 고쳐진다 (B3 불성립)

- det+noise는 det 출력을 그대로 두고, 공유하는 새 균일 잡음을 매 forward 더한다. 이것이 MW 8/8칸을 고쳤다(red 0.52~0.92). 사전 예측("어떤 잡음도 손상을 줄이지 못한다")과 반대다.
- 다른 단위에서는 거의 고치지 못했다.
  - DMC는 1/4칸이다. 고친 칸은 편향으로 분류된 finger 하나이고, 지속으로 분류된 cheetah·walker·quadruped는 고치지 못했다.
  - DINO-WM Wall은 0/1칸이다(red 0.07).
  - 그래서 "지속이 지배하면 새 잡음으로 고쳐진다"고 일반화할 수 없다. 이 절의 첫 판(01:32Z)에 그렇게 쓴 문장은 DMC 결과와 어긋나서 지웠다(LOG 02:05Z).
- 기술 항목
  - 지속 몫 ρ_p(E13 마지막 반복 집합, 같은 E11 12칸): det+noise 0.05~0.23, sh-iter 0.00~0.12, det+shift 0.00~0.11, sh-iter+b 0.02~0.16. 고정 변형(det, det+BC, sh-ep, det+shift-fixed)은 모두 1.00이다(`TD_tdmpc2/results/analysis/td_e13_cells.csv`). [정정 02:55Z: 첫 판은 dither의 범위를 E13 전체 72행에서 가져와 칸 집합이 달랐다]
  - 기전 과제의 drift 비(변형 ÷ det): push 0.58, pick-place 0.53, cheetah 0.35.
- 같은 칸에서 dither는 더 많이 고친다.
  - DMC: det+shift 3/4칸, sh-iter 2/4칸
  - Wall: red sh-iter 1.00, det+shift 1.36
  - 전체 오차 분산은 det+noise가 sh-iter와 같은 Δ²/6이고, det+shift는 Δ²/12다.

### 1.3 처방과 비트 한계 (E12, E15)

- **E12**: MW에서 dither가 per-tensor(Fr1)와 per-channel(Fr2) 모두 7/8과제에서 1 bit 이상을 번다. 설계 §2.4의 쓰임은 "강한 범위 보정과 합쳐도 한 비트를 번다"를 처방 주장에 넣는 것이다.
  - MW 과제별 가장 낮은 k\*는 1 bit가 6과제(door-open, drawer-open, button-press, peg-insert, window-open, sweep-into, 모두 dither를 쓴 구성), 2 bit가 2과제(push: PC 네 반올림 모두, pick-place: PC det+shift)다.
  - DMC의 G는 PT 0·0·0·1, PC 1·0·0·1(cheetah, walker, quadruped, finger)이다. DMC에서는 기본 반복 수로 비트를 거의 벌지 못한다.
  - DINO-WM: PushT G = 2(PT·PC·RC 모두. k\*: det 4, det+BC 5(PT)·4(PC), sh-iter 2), Wall G = 2(PT·PC), PointMaze G = 0. rollout으로 고른 범위(RC, WM-Quant의 핵심 생각을 우리 코드로 구현)는 k\*를 바꾸지 않는다(k3 det D: RC +16, PT +32, PC +28점).
- **E15**: 반복을 12회로 늘리면 DMC 3/4과제에서 나은 dither가 손상의 절반 이상을 없애고(It1), red가 반복 수와 함께 단조로 커진다(It2, 3/4). 설계 §5의 쓰임은 DMC를 "기본 반복 수로는 새 잡음을 평균할 반복이 모자란 경계"로 쓰는 것이다.
  - 예외는 walker다. 반복 24에서도 red 0.21(det+shift)에 그친다. E1·E11에서 sh-iter가 해로웠던 칸이다.
  - 비용 비교(cheetah): det에 한 비트를 더한 구성(PT k6, 반복 6)은 D +8.2%에 결정당 30.2 ms, det+shift k5 + 반복 12는 +3.7%에 63.7 ms, 반복 24는 −3.6%에 123.9 ms다.

### 1.4 진단 (E13, Dg1)

- 사전 지정 비교 Dg1이 MW(0.971)와 DMC(0.387)에서 성립했다. 같은 비트 안에서 변형 사이의 손상 순위는 잠재 1-step 오차보다 "지속 몫 × 후보 사이 오차의 상대 크기"(ρ_p × σ_e/σ_J, 마지막 반복 후보 집합)가 훨씬 잘 맞힌다. 설계 §3.3의 쓰임은 "배치 전에 재야 할 것"으로 본문에 쓰는 것이다.
- 공통 held 상태에서 E1의 36칸을 다시 재면, det의 계획 시점 1-step 오차가 sh-iter보다 크지 않은 칸이 27/36이다. "잠재 1-step 오차로는 det의 손상이 보이지 않는다"는 문장은 공통 상태 기준으로도 유지된다.
- DMC 중 walker(−0.238)와 quadruped(−0.119)는 ρ_p × σ_e/σ_J가 1-step 오차보다 못 맞혔다.
- DINO-WM은 불성립이다(0.262 < 0.3, 손상 칸 Wall 하나). ρ_p × σ_e/σ_J의 순위 일치는 높지만(0.905), 편향이 지배하는 Wall에서는 잠재 1-step 오차도 det를 어느 정도 가른다(0.643). 진단 주장은 TD-MPC2(MPPI)로 한정한다.

### 1.5 비용과 정수 경로 (E17)

- 정수 kernel에서 sh-iter의 rollout 지연 오버헤드는 6조건 모두 ≤ 2.01%다(E8 L3과 같은 꼴).
- det+shift는 N ≥ 2048에서 0.64~3.63%지만 N 512에서 5.30%(5M)·7.23%(317M)라 K1이 불성립이다. 원인은 설계가 정한 offset 보정(층 호출마다 다음 층 가중치를 한 번 더 읽는 행렬–벡터 곱)이다. 설계 §7의 "K1·K2 성립이면 det+shift를 공동 처방으로 올린다"는 조건이 채워지지 않는다.
- K2는 성립했다. 정수 kernel에서도 det와 det+shift의 차가 fake-quant와 같은 크기로 재현된다(walker에서 det+shift가 det보다 나쁜 것까지 같다).
- 접은 per-channel 정수 구성(W8, 활성 scale을 다음 층에 접음)의 D는 E12의 fake-quant per-channel 칸과 16칸 중 13칸에서 CI 안에서 같다. 벗어난 칸은 push k2 det(정수가 +10.0점 [1.1, 18.9] 나쁨)와 push k2·k3 sh-iter(차 +3.3·+1.1점, CI 하한 0)다(`coordinator/e17_pc_vs_e12.txt`, 두 pod는 독립 표본).
- 미리 바꾼 BF16은 9/12과제에서 손상이 없고, door-open(+5.6점)과 cheetah(+2.1%)에 작은 손상이 있다(K3 불성립). finger는 BF16이 1.4% 낫다. "BF16을 폐루프로 재지 않았다"는 한계 문장은 "작지만 0이 아닌 손상이 2/12과제에 있다"로 바뀐다.

## 2. 묶음별 요약

| 묶음 | 실험 | pod(GPU) | episode | 실패 job | 지출 | 보고 |
|---|---|---|---|---|---|---|
| td | E11·E12·E13·E15·E16(TD) | td.1~td.6(A40) | 36,375 + E13 held 1,080 | 0 | $15.24 | [TD_tdmpc2/REPORT.md](TD_tdmpc2/REPORT.md) |
| mt | E14 | mt.1·mt.2(RTX 4090; A5000 품절) | 4,320 | 0 | $10.29 | [MT_317m/REPORT.md](MT_317m/REPORT.md) |
| k | E17 | k.1(A40) | 4,680 + 지연 block 240 | 0 | $1.67 | [K_kernel/REPORT.md](K_kernel/REPORT.md) |
| dw | E11·E12·E13·E16(DW) | dw.1~dw.5(RTX 4090; A5000 품절) | episode-run 9,140 | 0 | $23.75 | [DW_dinowm/REPORT.md](DW_dinowm/REPORT.md) |

## 3. 편차와 운영 사건

| # | 시각(UTC) | 무엇 | 정한 방법 | 영향 |
|---|---|---|---|---|
| D1 | 11-21 18:41Z | PushT eval8 표집 규칙(hard = held·smoke·eval6, soft = repro 최소 공유, 완전 중복 기각), 덩어리 단위 bootstrap을 보조로 | 판단 에이전트(Fable 5.2 max), eval8 결과 전 | PushT eval8을 설계 그대로 만들 수 없어서 정함 |
| D2 | 11-21 19:01Z | 편향 벡터 b의 반복을 α = 1 → ½ → ¼ 단계 감쇠로(각 단계 최대 60번, 잔차 < 1% Δ) | 판단 에이전트, 수렴한 b의 det+BC 결과 전 | 설계의 3 sweep이 모든 묶음에서 발산·진동. 바꾼 뒤 모든 칸 수렴 |
| D3 | 11-21 19:41Z | [사후] Wall eval8 점검 3 예외(none 49/100이 E1 61% ±10 밖; 같은 pod E1 쌍 재실행 62/100) | 판단 에이전트, Wall 양자화 변형 결과 전 | Wall은 기준선 셋(49%, 61%, 62%)을 함께 보고 |

- 운영 사건(판정 무관): RunPod REST 호스트 장애(11-21 18:20Z~, `rest.runpod.io` NXDOMAIN) → pod 조회·삭제를 GraphQL로 우회(`runpod_tmlr/rp_gql.py`). mt의 선택 진단이 GPU를 나눠 쓴 17분 동안 episode 속도 반감. dw.1의 뒤쪽 일을 같은 GPU 종류의 dw.5로 나눔(짝 규칙 유지, 첫 none job 비트 일치 확인). k의 det+shift 지연 구현을 smoke를 보고 v1 → v2로 한 번 고침(본 측정 전, 결과를 보고 고친 구현으로 기록). 결과 회수 rsync가 사라진 잠금 파일로 실패 → 잠금·임시 파일 제외.
- 설계 §8.2 추정 약 $42(4090 대체 시 약 $48)에 대해 실제 지출은 $50.95다. A5000 재고가 없어 mt·dw가 모두 RTX 4090이었고, D2 재계산과 확장 칸이 더해졌다.

## 4. 비용과 시간

- 지출(`runpod_tmlr/budget.py status`, 장부의 uptime × $/h × 1.03): td $15.24, mt $10.29, k $1.67, dw $23.75, **합 $50.95**(사용자 상한 $80, 감시 $77).
- 시간: 설계 고정 11-21 17:59Z → 첫 pod 18:28Z → 마지막 본 실행 끝 11-22 01:16Z(dw.2) → 모든 pod 삭제 01:23:58Z. 실패 job은 네 묶음 모두 0이다.

## 5. 조정자 재계산 기록

| 대상 | 방법 | 결과 | 파일 |
|---|---|---|---|
| TD E11(B1, B3 TD 몫, B4, DMC 읽기) | `td_episodes.csv`에서 D·red·Δ_pers의 짝 bootstrap과 Holm을 따로 계산 | 작업자 판정과 모두 같음 | `coordinator/verify_td_e11.py`, `.txt` |
| E14(P1, P2) | mt.1·mt.2의 job 파일에서 episode를 직접 읽어 계산(난수 상태 sha1 불일치 0, 한 pod 안의 짝) | 같음 | `coordinator/verify_mt_e14.py`, `.txt` |
| K2 | td E11 칸과 k 정수 칸의 Δ와 CI를 결합 | 성립(4/4) | LOG 00:14Z |
| E17 접은 per-channel 대 E12 | 두 pod 독립 bootstrap의 차 | 13/16칸 CI에 0 포함 | `coordinator/e17_pc_vs_e12.py`, `.txt` |
| DINO-WM(B2, B3 DW 몫, E12 k\*·G, E16) | 다섯 pod의 job JSON에서 직접 계산(작업자의 독립 재계산 recount_dw.py도 233항목 불일치 0) | 일치 | `coordinator/verify_dw.py`, `.txt` |

## 6. [사후] 보강 (편차 D4·D5·D6)

- 사용자 결정(2026-11-22T01:35Z)으로 DINO-WM PushT 3-bit E11 칸을 새 pod dw.6(RTX 4090)에서 더 돈다. none + 8변형 × eval8 100 episode이고 E13은 없다. 예상 약 4.5 h, 약 $4.2다.
- eval8에서 손상 칸이 Wall 하나뿐이었기 때문이다. 사전 지정 판정은 바꾸지 않고, 결과는 이 절에 [사후] 기술로 덧붙인다.

## 7. 재분석 R1~R5 (pod 없음, 2026-11-22T02:30Z)

- 판단 에이전트가 제안하고 사용자가 승인해 했다. 스크립트·출력·요약은 [coordinator/reanalysis/](coordinator/reanalysis/)의 `summary.txt`에 있다. 먼저 발표값을 모두 재현했다: TD E12 340칸, Fr1·Fr2 7/8, DINO-WM G, R5 대비 46개.
- **R1**(k\*·G bootstrap, 과제마다 같은 짝, 두 단계 칸은 층화)
  - P(Fr1 성립) 0.999, P(Fr2 성립) 1.000. sh-iter 단독 0.997·0.995, det+shift 단독 0.999·1.000이다.
  - 흔들리는 MW 칸: pick-place P(G≥1) PT 0.41·PC 0.66, window-open 0.74·0.87.
  - DINO-WM P(G≥1) / P(G≥2)
    - PushT: PT 0.999 / 0.985, PC 0.999 / 0.593, RC 0.977 / 0.805
    - Wall: PT 0.61 / 0.53, PC 0.84 / 0.67
    - PointMaze: PT 0.001, PC 0.14
- **R2**(τ 0.05~0.20)
  - 나은 dither 기준 Fr1 6~8/8, Fr2 5~7/8로 모든 τ에서 성립한다. sh-iter 단독은 문턱에 걸리는 τ가 있다.
  - D가 정확히 0.100인 MW 칸이 4개다. "D < 0.10"으로 바꾸면 Fr2가 6/8이다.
  - PushT PT G = 2는 모든 τ에서 유지된다.
- **R3**(방법별 G, MW PT/PC): 나은 쪽 7/8·7/8, sh-iter 단독 6/8·5/8, det+shift 단독 7/8·7/8.
- **R4** [사후](E1 + eval8 역분산 합산, 짝 아님)
  - PushT 4-bit: +15.0 [7.6, 22.4]. 이질적이다(Q p 0.008, I² 86%; 덩어리 분산이면 p 0.125).
  - PointMaze 3-bit: +10.1 [3.7, 16.5], I² 0%.
  - Wall 3-bit: +45.8 [38.0, 53.6], I² 1%.
- **R5**(지속 비용)
  - TD 5M: 12칸 모두 CI 하한 > 0이다(최소 door-open +3.3, window-open +4.4).
  - 317M: 7/8(push +3.3 [0.0, 7.8]만 예외).
  - DINO-WM D(sh-plan) − D(sh-iter)의 CI 상한: PushT +13(판정 밖 칸, 덩어리 CI는 [+1.7, +10.4]), PointMaze +7/+10, Wall +4/+4.
  - 두 계열의 칸별 크기는 겹친다.
- 이 결과로 CLAIMS v2.1에서 수치 문구를 정밀화했다(CLAIMS §5.2). 판정은 바뀌지 않았다.
- **X5(D6, 끝)**: MW det+noise 비트 곡선. td.7(A40)에서 02:14~03:37Z에 돌았다. 1,332 job(확장 36 포함), 실패 0, $0.88.
  - det 계열 대비 번 비트 G_noise ≥ 1: PT 4/8, PC 1/8(dither는 7/8·7/8)
  - k\*(det+noise) ≤ k\*(sh-iter): PT 3/8, PC 2/8
  - k\*(det+noise) ≤ 두 dither 모두: PT 2/8, PC 1/8
  - E11 1차 k 칸이 같은 기종의 다른 pod에서 비트 단위로 재현됐다.
  - 출처: [TD_tdmpc2/REPORT.md](TD_tdmpc2/REPORT.md) §11, `TD_tdmpc2/results/analysis_x5/`, 조정자 재계산 `coordinator/verify_x5.txt`(일치).
- **D4(PushT 3-bit)·X17(D5)**: dw.6에서 진행 중이다.
