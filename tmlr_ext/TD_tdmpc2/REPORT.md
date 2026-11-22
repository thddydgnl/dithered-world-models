# td 묶음(TD-MPC2 5M) 결과 보고 — E11·E12·E13·E15·E16(TD 절반)

- 작성: 2026-11-21T23:45Z, td 작업자. 단계 B(조정자 지시 23:3xZ). Write 도구가 보고서 파일을 막아 규칙대로 Bash heredoc으로 썼다.
- 기준: [../DESIGN_tmlr.md](../DESIGN_tmlr.md) v1.0과 편차 D2(b 반복 규칙, [../LOG.md](../LOG.md) 19:01:09Z, DESIGN_tmlr §11). 운영 계획 [PLAN.md](PLAN.md), 기록 [LOG.md](LOG.md).
- 분석: `code/analyze_td.py`(md5 `b5694de1033ddeb49ddb1caa62f626c2`; 본 실행 전 고정판 `d7597ad4…`에 조정자 요청의 K2 입력 출력만 더함, 판정 계산은 같음), `code/plan_td.py`(md5 `0d5a8a70b6046f46b4690a4dfbffd04f`). td.1에서 여섯 pod의 결과를 한 번에 분석했다(43초, 2026-11-21T23:37Z).
- 결과 파일(모두 `results/analysis/`): `td_summary.json`(판정·모든 칸), `td_e11_cells.csv`, `td_e11_contrasts.csv`, `td_k2_e11_det_minus_detshift.csv`, `td_e12_cells.csv`, `td_e13_cells.csv`, `td_episodes.csv`(episode 원자료 36,375행), `td_tables.md`(자동 표). 원자료는 `results/raw/td.1/`~`td.6/`(각 pod의 `out/` 전체).
- 표기: D는 none 대비 하락(MW 성공률 %p, DMC 상대 return %). CI는 짝 bootstrap 10,000회 95% 백분위(RandomState(0)). red = (D_det − D_v)/D_det(점 추정으로 판정). 따로 적지 않은 수치의 출처는 `results/analysis/td_summary.json`이고, 같은 값이 `td_tables.md`에 있다.

## 0. 판정 표

| 가설 | 판정 | 핵심 수치 [95% CI] | 출처(`td_summary.json`) |
|---|---|---|---|
| **B1**: MW는 지속 지배 | **성립** | MW 손상 칸 8/8. 분류: 지속 5(pick-place, door-open, peg-insert, window-open, sweep-into), 편향 3(push, drawer-open, button-press), 해상도 0 → 지속이 과반(5 > 4) | `judgements.B1`, `E11.units.mw` |
| B1, D2 두 읽기 | (가) = (나), 보류 아님 | bc_unconverged 0칸이라 (나)도 같은 읽기(지속) | `judgements.D2_two_readings` |
| **B3**(TD 몫): det+noise는 손상 칸의 과반을 못 고친다 | **MW 불성립**, DMC 성립 | MW: det+noise red ≥ 0.5가 8/8(red 0.52~0.92). DMC: 1/4(finger 0.52) | `judgements.B3_td` |
| **B4**: MW 손상 칸의 75% 이상에서 sh-iter red ≥ 0.5, 해로운 칸 없음 | **성립** | 7/8 = 0.875(못 미친 칸: window-open red 0.29). 해로운 sh-iter 칸 0/8(손상 칸만 읽기도 성립) | `judgements.B4` |
| DMC 읽기(기술) | 지속 지배 | 지속 3(cheetah, walker, quadruped), 편향 1(finger) | `judgements.DMC_reading` |
| **Fr1**: PT에서 MW 과제 75% 이상 G ≥ 1 | **성립** | G ≥ 1이 7/8(push 2, pick-place 0, door-open 3, drawer-open 2, button-press 2, peg-insert 2, window-open 2, sweep-into 2) | `judgements.Fr1` |
| **Fr2**: PC에서 MW 과제 50% 이상 G ≥ 1 | **성립** | G ≥ 1이 7/8(push 0, pick-place 1, door-open 1, drawer-open 1, button-press 1, peg-insert 2, window-open 2, sweep-into 1) | `judgements.Fr2` |
| Fr1·Fr2, D2 두 읽기 | (가) = (나), 보류 아님 | 미수렴 칸 없음 | `judgements.D2_two_readings` |
| **Dg1**(MW) | **성립** | 칸 안 Spearman 차의 평균 0.971(8칸 모두 양수, 0.66~1.43) | `judgements.Dg1.mw` |
| **Dg1**(DMC) | **성립** | 평균 0.387(cheetah 1.000, finger 0.905, walker −0.238, quadruped −0.119) | `judgements.Dg1.dmc` |
| **It1**: 반복 12에서 나은 dither가 3/4과제 이상 red ≥ 0.5, 해로운 과제 없음 | **성립** | red_best(12): cheetah 0.914, quadruped 0.758, finger 1.000, walker −0.095. 해로운 과제 없음 | `judgements.It1` |
| **It2**: 3과제 이상에서 red가 6 < 12 < 24로 단조 증가 | **성립** | 단조 3(cheetah 0.745 → 0.914 → 1.087, walker −0.225 → −0.095 → 0.207, quadruped 0.587 → 0.758 → 0.840). finger는 1.0001 → 0.9999 → 1.027로 아님 | `judgements.It2` |
| E16 TD 절반(기술) | 3과제 모두 자격 | D(sh-ep) − D(sh-iter): CEM식 push +47.8 [36.7, 58.9], pick-place +42.2 [28.9, 55.6], cheetah +18.7 [16.5, 20.9]; 기본(E11) +48.9 [38.9, 60.0], +28.9 [15.6, 42.2], +36.6 [32.5, 40.7] | `E16` |
| K2 입력(E17 결합용, 기술) | — | D(det) − D(det+shift): push +43.3 [32.2, 54.4], pick-place +26.7 [14.4, 38.9], cheetah +29.8 [26.8, 32.9], walker −13.4 [−17.8, −8.9] | `K2_E11_det_minus_detshift`, `td_k2_e11_det_minus_detshift.csv` |

- B3 전체 판정("어느 단위에서도")은 DINO-WM 단위와 합쳐 조정자가 정한다. TD의 MW 단위가 이미 불성립이므로 B3는 DINO-WM 결과와 관계없이 성립할 수 없다.
- B1은 성립하지만 MW 8칸 중 3칸은 det+BC가 det 손상의 절반 이상을 고친 칸(편향 분류)이다(§2).

## 1. 무엇을 돌렸나

- pod 여섯(A40 SECURE, $0.59/h), 과제 배치와 일정은 PLAN.md §4. 본 실행 driver 18:48~19:08Z 시작, 마지막 pod(td.1) 23:33Z 끝. job 실패 0.
- eval episode 36,375개(`n_episodes`): episode job 결과 7,275개, 모두 NVIDIA A40(`meta.gpus`), (과제, 체크포인트)마다 host 하나(`meta.pairing_one_host_per_task_ckpt` = true), seed = 100000 + episode가 모든 job에서 맞음(`meta.seed_mismatch` = 0). E13 held episode 1,080개(E13 job 216개, seed 101000~101029).
- 점검 3(새 seed의 none, 기본 계획기): 12과제 모두 E1 none에서 10점(DMC 10%) 안(`baseline_check3`). MW 0.967~1.000(E1 0.967~1.000), cheetah 900.9(900.8), walker 984.6(983.5), quadruped 971.7(975.5), finger 969.1(977.9).
- E12 적응 결정 32개(`meta.decisions`, 각 pod `out/driver/decisions.jsonl`): DMC 두 단계의 2단계 12개, 아래로 확장 8개, 위로 확장 12개. 분석이 다시 확인한 규칙 위반 0(`E12.tasks.*.rule_check`가 모두 빈 목록).
- 편향 벡터(D2): (과제, 체크포인트, 범위, k) 576칸 모두 수렴 — α = 1 34칸, α = ½ 540칸, α = ¼ 2칸, bc_unconverged 0, 최대 갱신 39, 최종 잔차(층 최악 mean_j |m_j|/Δ_j) 최대 0.00999(`judgements.D2_two_readings.bias_summary`; 칸별 기록 `raw/<tag>/jobs/bias/`). 따라서 §0.7 점검 2(D2: 최종 b에서)는 det+BC도 통과다.
- PT 범위를 같은 규약으로 다시 재면 E0 파일과 hi 차이가 E0 폭의 중앙값 0.47%, 최대 21%(lo는 차 0, Mish 최솟값). PC 범위의 단위별 평균 폭은 PT 폭의 0.08~0.70배(`meta.prep`).

## 2. E11 — 편향·해상도·지속 (PT, E1 1차 k, 90짝; `td_e11_cells.csv`, `td_e11_contrasts.csv`)

| 과제 | k | none | D_det [CI] | 분류 | det+BC red | sh-ep red | det+shift-fixed red | sh-iter+b red | sh-iter red | det+shift red | det+noise red | Δ_bias [CI] | Δ_pers [CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mw-push | 3 | 1.000 | +45.6 [35.6, 55.6] | 편향 | 0.63 | −0.15 | 0.07 | 0.88 | 0.93 | 0.95 | 0.80 | +28.9 [17.8, 41.1] | +40.0 [28.9, 51.1] |
| mw-pick-place | 3 | 1.000 | +38.9 [28.9, 48.9] | 지속 | 0.46 | −0.17 | 0.09 | 0.34 | 0.57 | 0.69 | 0.60 | +17.8 [4.4, 31.1] | +23.3 [11.1, 35.6] |
| mw-door-open | 4 | 1.000 | +27.8 [18.9, 37.8] | 지속 | 0.16 | 0.60 | 0.48 | 0.80 | 0.96 | 0.88 | 0.52 | +4.4 [−8.9, 17.8] | +11.1 [3.3, 20.0] |
| mw-drawer-open | 3 | 1.000 | +14.4 [7.8, 22.2] | 편향 | 0.85 | −0.23 | 0.31 | 0.92 | 1.00 | 1.00 | 0.92 | +12.2 [5.6, 20.0] | +10.0 [4.4, 16.7] |
| mw-button-press | 3 | 1.000 | +40.0 [30.0, 50.0] | 편향 | 0.94 | −0.53 | 0.22 | 0.25 | 1.00 | 1.00 | 0.86 | +37.8 [27.8, 48.9] | +31.1 [22.2, 41.1] |
| mw-peg-insert | 4 | 1.000 | +25.6 [16.7, 34.4] | 지속 | −0.35 | −0.78 | 0.26 | 0.78 | 1.00 | 1.00 | 0.65 | −8.9 [−21.1, 3.3] | +18.9 [11.1, 27.8] |
| mw-window-open | 4 | 1.000 | +18.9 [11.1, 27.8] | 지속 | 0.29 | −0.59 | −0.18 | 0.82 | 0.29 | 0.53 | 0.53 | +5.6 [−4.4, 15.6] | +13.3 [4.4, 22.2] |
| mw-sweep-into | 3 | 0.967 | +13.3 [4.4, 22.2] | 지속 | 0.00 | −2.00 | −1.00 | 0.42 | 1.17 | 1.25 | 0.67 | +0.0 [−11.1, 11.1] | +30.0 [21.1, 40.0] |
| cheetah-run | 5 | 900.9 | +40.0 [37.6, 42.7] | 지속 | 0.29 | −0.22 | −0.07 | 0.49 | 0.69 | 0.74 | 0.19 | +11.7 [8.4, 14.9] | +32.6 [29.0, 36.4] |
| walker-walk | 3 | 984.6 | +59.3 [55.1, 63.4] | 지속 | 0.03 | −0.50 | −0.47 | −0.38 | −0.30 | −0.23 | −0.39 | +1.8 [−2.3, 5.9] | +14.7 [12.3, 17.2] |
| quadruped-walk | 3 | 971.7 | +83.8 [77.8, 89.4] | 지속 | 0.44 | −0.06 | 0.03 | −0.01 | 0.29 | 0.59 | −0.01 | +36.7 [29.7, 43.7] | +46.7 [40.8, 52.4] |
| finger-turn-hard | 4 | 969.1 | +49.5 [39.1, 59.6] | 편향 | 0.77 | 0.31 | 0.52 | 0.73 | 1.00 | 0.98 | 0.52 | +38.1 [28.6, 47.9] | +23.2 [15.3, 31.7] |

- 손상 칸: 12칸 모두(단위 안 Holm, α 0.05; det의 단측 p는 sweep-into 0.0012, 나머지 0.0001). 분류 규칙은 PLAN §1.7(편향 → 지속(Δ_pers CI 하한 > 0) → 해상도 → 그 밖).
- 고친 칸 수(red ≥ 0.5, `E11.units.*.n_fix`): MW det+BC 3, sh-ep 1, det+shift-fixed 0, sh-iter+b 5, sh-iter 7, det+shift 8, det+noise 8 / DMC 각각 1, 0, 1, 1, 2, 3, 1.
- 해로운 칸(D_v − D_det CI 하한 > 5, `td_e11_cells.csv`의 harmful 열): sh-ep가 button-press +21.1 [7.8, 34.4], peg-insert +20.0 [7.8, 33.3], sweep-into +26.7 [13.3, 38.9]. walker는 det+BC를 뺀 6변형(sh-ep, det+shift-fixed, sh-iter+b, sh-iter, det+shift, det+noise)이 모두 해롭다(예: sh-iter +17.8 [13.3, 22.5], det+shift +13.4 [8.9, 17.8]; E1 walker k = 3과 같은 방향).
- 보조 대비(`td_e11_contrasts.csv`)
  - D(sh-ep) − D(sh-iter): 12칸 모두 양수(+10.0 ~ +61.1, CI 하한 모두 > 0)
  - D(det+noise) − D(det): MW 8칸 모두 음수(−8.9 ~ −36.7; CI가 0을 포함하는 칸은 window-open, sweep-into). DMC cheetah −7.6, finger −26.0, walker +23.3, quadruped +1.1
  - D(sh-iter+b) − D(sh-iter): −10.0(window-open) ~ +30.0(button-press)
  - D(det+shift) − D(sh-iter): MW −4.4 ~ +2.2(모두 CI가 0 포함), DMC cheetah −2.1 [−4.0, −0.4], walker −4.4 [−7.2, −1.6], quadruped −25.1 [−30.7, −19.3], finger +0.8 [−1.1, 3.5]
- 기전 과제의 drift 비(변형 ÷ det, `E11.cells.<task>.mech`)
  - mw-push: det+BC 0.36, sh-ep 0.82, det+shift-fixed 0.66, sh-iter+b 0.38, sh-iter 0.10, det+shift 0.07, det+noise 0.58
  - mw-pick-place: 0.42, 0.60, 0.57, 0.59, 0.20, 0.22, 0.53
  - cheetah-run: 1.02, 1.41, 1.18, −0.18, 0.02, −0.01, 0.35
- 변형별 1·3·5-step 잠재 오차는 E13 표(`td_e13_cells.csv`, 첫 집합)에 있다.

## 3. E12 — 비트 한계 (`td_e12_cells.csv`)

| 과제 | k\*(PT): det / det+BC / sh-iter / det+shift | G(PT) | k\*(PC): det / det+BC / sh-iter / det+shift | G(PC) | 가장 낮은 k\*와 구성 |
|---|---|---|---|---|---|
| mw-push | 5 / 5 / 3 / 3 | 2 | 2 / 2 / 2 / 2 | 0 | 2: PC 네 반올림 모두 |
| mw-pick-place | 5 / 4 / 4 / 4 | 0 | 4 / 3 / 3 / 2 | 1 | 2: PC det+shift |
| mw-door-open | 5 / 5 / 2 / 3 | 3 | 2 / 4 / 1 / 1 | 1 | 1: PC sh-iter·det+shift |
| mw-drawer-open | 4 / 3 / 1 / 1 | 2 | 2 / 2 / 1 / 1 | 1 | 1: PT·PC sh-iter·det+shift |
| mw-button-press | 4 / 3 / 1 / 1 | 2 | 2 / 2 / 1 / 1 | 1 | 1: PT·PC sh-iter·det+shift |
| mw-peg-insert | 5 / 6 / 3 / 3 | 2 | 3 / 4 / 1 / 1 | 2 | 1: PC sh-iter·det+shift |
| mw-window-open | 5 / 5 / 5 / 3 | 2 | 4 / 3 / 3 / 1 | 2 | 1: PC det+shift |
| mw-sweep-into | 4 / 4 / 2 / 2 | 2 | 2 / 2 / 1 / 1 | 1 | 1: PC sh-iter·det+shift |
| cheetah-run | 6 / 6 / 6 / 6 | 0 | 5 / 5 / 5 / 4 | 1 | 4: PC det+shift |
| walker-walk | 5 / 4 / 5 / 4 | 0 | 3 / 3 / 3 / 3 | 0 | 3: PC 네 반올림 모두 |
| quadruped-walk | 4 / 4 / 4 / 4 | 0 | 3 / 3 / 3 / 3 | 0 | 3: PC 네 반올림 모두 |
| finger-turn-hard | 5 / 5 / 4 / 4 | 1 | 4 / 4 / 3 / 3 | 1 | 3: PC sh-iter·det+shift |

- k\* = 9("k = 8까지도 D > τ")인 구성은 없다. 모든 구성에서 확장 규칙이 데이터와 맞게 적용됐다(경계 k의 D와 τ 비교, `rule_check` 빈 목록).
- DMC의 G: cheetah PT 0·PC 1, walker 0·0, quadruped 0·0, finger 1·1.
- 비트–하락 곡선 전체는 `td_tables.md`의 "E12 비트–하락 곡선"과 `td_e12_cells.csv`(n 열: DMC 45 또는 90). DMC 두 단계 칸의 CI는 중간 확인을 보정하지 않은 값이다.
- 눈에 띄는 비단조: finger PC det는 k = 2에서 +24.3, k = 3에서 +77.0, k = 4에서 −2.1(각 45짝). window-open PT sh-iter는 k = 2 +10.0, k = 4 +13.3(그래서 k\* = 5).

## 4. E15 — DMC × 반복 수 (none은 반복 수마다 따로; `E15.tasks`)

| 과제 | 반복 | 짝 | none return | D_det | sh-iter D / red | det+shift D / red |
|---|---|---|---|---|---|---|
| cheetah-run | 6 / 12 / 24 | 90 / 90 / 60 | 900.9 / 892.3 / 841.0 | +40.0 / +43.0 / +41.2 | +12.4 / 0.69 · +5.2 / 0.88 · +0.1 / 1.00 | +10.2 / 0.74 · +3.7 / 0.91 · −3.6 / 1.09 |
| walker-walk | 6 / 12 / 24 | 90 / 90 / 60 | 984.6 / 985.0 / 983.4 | +59.3 / +55.4 / +53.3 | +77.2 / −0.30 · +64.5 / −0.17 · +52.5 / 0.01 | +72.7 / −0.23 · +60.7 / −0.10 · +42.2 / 0.21 |
| quadruped-walk | 6 / 12 / 24 | 90 / 90 / 60 | 971.7 / 972.5 / 971.8 | +83.8 / +84.7 / +87.0 | +59.7 / 0.29 · +32.6 / 0.62 · +20.8 / 0.76 | +34.6 / 0.59 · +20.5 / 0.76 · +13.9 / 0.84 |
| finger-turn-hard | 6 / 12 / 24 | 90 / 90 / 60 | 969.1 / 977.5 / 961.7 | +49.5 / +50.2 / +56.6 | −0.0 / 1.00 · +2.2 / 0.96 · −0.8 / 1.01 | +0.8 / 0.98 · +0.0 / 1.00 · −1.5 / 1.03 |

- D_det의 CI: cheetah +40.0 [37.6, 42.7] / +43.0 [39.9, 46.0] / +41.2 [36.7, 45.4], walker +59.3 [55.1, 63.4] / +55.4 [51.1, 59.5] / +53.3 [47.9, 58.7], quadruped +83.8 [77.8, 89.4] / +84.7 [78.7, 90.4] / +87.0 [79.6, 93.2], finger +49.5 [39.1, 59.6] / +50.2 [39.9, 60.0] / +56.6 [43.6, 69.1].
- 결정당 계획 시간(대기열 끝 timing job, GPU 단독, 60결정 중앙값 ms, `meta.timing`): 반복 6 none 23.9~38.6, det 30.1~30.5, sh-iter 32.2~32.9, det+shift 33.4~34.1; 반복 12 none 45.2~45.9, det 56.9~58.0, sh-iter 59.3~63.2, det+shift 62.1~65.6; 반복 24 none 85.5~89.3, det 109.0~113.6, sh-iter 119.0~121.4, det+shift 123.9~126.1.
- return–시간 비교(`E15.return_time`, 기술): cheetah에서 "det + 한 비트 더"(PT k = 6, 반복 6)는 D +8.2에 30.2 ms, "det+shift k = 5 + 반복 12"는 D +3.7에 63.7 ms, 반복 24는 D −3.6에 123.9 ms다. walker는 det k = 4(반복 6)가 D +15.7이고, dither k = 3은 반복 24에서도 +42.2~+52.5다.
- 반복 6 walker의 det+shift는 해롭다(E11과 같은 칸). 반복 12·24에서는 해로운 칸이 없다.

## 5. E16 — TD-MPC2 CEM식 계획기 (temperature 0, min_std 0.01, num_pi_trajs 0; 90짝; `E16`)

| 과제 | none CEM / 기본 | 자격(≥ 60%) | det D | sh-iter D | sh-ep D | det+shift D | sh-ep − sh-iter, CEM [CI] | 기본(E11) [CI] |
|---|---|---|---|---|---|---|---|---|
| mw-push | 1.000 / 1.000 | 예 | +50.0 | +11.1 | +58.9 | +10.0 | +47.8 [36.7, 58.9] | +48.9 [38.9, 60.0] |
| mw-pick-place | 0.989 / 1.000 | 예 | +45.6 | +8.9 | +51.1 | +3.3 | +42.2 [28.9, 55.6] | +28.9 [15.6, 42.2] |
| cheetah-run | 830.4 / 900.9 | 예 | +57.1 | +48.9 | +67.6 | +46.4 | +18.7 [16.5, 20.9] | +36.6 [32.5, 40.7] |

- 2×2(모델 × 계획기)의 TD-MPC2 행: 두 계획기 모두 D(sh-ep) − D(sh-iter)의 CI 하한이 0보다 크다. DINO-WM 행은 dw 묶음 결과로 조정자가 합친다.
- temperature 0은 TD-MPC2 `_plan`의 `exp(temperature·(V − V_max))`가 모든 elite에서 1이 되어 가중이 정확히 1/64다(스모크에서 확인, PLAN §1.6).

## 6. E13 — held 상태 진단 (held seed 101000~101029, 과제당 90 episode × 8상태; `td_e13_cells.csv`)

- Dg1 칸별(`judgements.Dg1.*.per_cell`; ρ(P1, D) / ρ(P2, D) / 차)
  - MW: push 0.905 / 0.119 / 0.786, pick-place 0.810 / −0.071 / 0.881, door-open 0.857 / −0.190 / 1.048, drawer-open 0.988 / −0.096 / 1.084, button-press 0.874 / 0.216 / 0.659, peg-insert 0.922 / −0.503 / 1.425, window-open 0.687 / −0.325 / 1.012, sweep-into 0.838 / −0.036 / 0.874
  - DMC: cheetah 0.905 / −0.095 / 1.000, walker 0.405 / 0.643 / −0.238, quadruped 0.619 / 0.738 / −0.119, finger 0.690 / −0.214 / 0.905
- E1의 36칸을 공통 held 상태에서 다시 재면, det의 계획 시점 1-step 오차(err1h, 첫 집합)가 sh-iter보다 크지 않은 칸이 27/36이다(`E13.e1_h1_common_states`).
- 예(첫 집합 err1h, 마지막 집합 ρ_p·σ_e/σ_J): mw-push k3 det 0.566·481.7, det+BC 0.175·33.9, sh-iter 0.443·0.84, det+shift 0.267·0.47; cheetah k5 det 0.036·5.17, sh-iter 0.066·0.47; walker k3 det 0.330·65.7, sh-iter 0.960·2.27. 마지막 집합은 후보가 모여 σ_J가 작으므로 σ_e/σ_J가 크다(near-tie 영역).
- held none의 성공률(return, `held_none`): MW 0.989~1.000, cheetah 890.6, walker 987.3, quadruped 978.8, finger 978.2.

## 7. 편차와 운영 선택

- **D2(조정자·판단 에이전트, 19:01:09Z)**: §0.6의 b 반복(b ← b + m, 3 sweep)이 수렴하지 않아(스모크 64칸, 3 sweep 뒤 최대 1.9Δ; 진단 `results/raw/td.5/phaseA/bias_diag*.json`) α = 1 → ½ → ¼ 단계 규칙으로 바꿨다. 결정 전에는 det+BC·sh-iter+b·E13 job을 관문으로 막았고, 결정 뒤 driver를 다시 띄우고 관문을 열었다(19:05~19:08Z).
- 구 규칙 b로 돈 episode는 스모크(seed 109000, 6개)와 처리량 측정(seed 108000+)뿐이고 버렸다. 다만 스모크 6개의 return은 D2 전에 스모크 로그에서 보였다. 본 실행 자료와는 무관하다.
- 분석 코드는 본 실행 전에 고정했다(`b40f3d00…`). 그 뒤 D2 두 읽기와 bias 요약(`d7597ad4…`, 수렴한 b의 결과 전), 조정자 요청의 K2 입력 출력(`b5694de1…`, 판정 계산 그대로)만 더했다.
- 운영 선택(PLAN §1): E11 계측은 설계가 적은 셋만(drift, 선택 항, 독립 draw 재평가), E13은 episode당 8상태(E1 규칙), Dg1의 P1은 칸 평균의 곱, k\* 없음은 9로 둠(실제로는 없었음), It1의 해로움은 나은 변형 기준, It2는 엄격한 증가, B3(TD 몫)는 "고친 칸 ≤ 절반"이면 성립.
- 워커 8(MW 워커 10은 4,170 대 4,647 ep/h로 느림). td.1~td.4·td.6 driver는 처리량 측정이 끝나기 전에 워커 8로 시작했다(E9 근거; 결과는 워커 수와 무관).
- td.1은 장부에 `td.1-stray`와 `td.1` 두 줄로 있다(감시 스크립트가 먼저 기록). 같은 pod이고 지출은 한 번만 센다.

## 8. 수치 대조 (원자료에서 다시 셈)

`results/raw/td.*/jobs/episodes/ev_*.json.gz`를 순수 Python으로 다시 읽어 점 추정을 셌다. 모두 분석과 같다.
- E11 D_det: mw-push 0.4556, window-open 0.1889(sh-iter 0.1333 → red 0.294), cheetah 0.4004, walker 0.5933. D_det+shift: push 0.0222, cheetah 0.1022, walker 0.7271(K2: +0.433, +0.298, −0.134)
- E12: pick-place PT sh-iter k = 3 0.1667, peg-insert PT det+BC k = 6 0.0222, quadruped PT det+BC k = 3 0.4713, cheetah PC det+shift k = 4 0.0571(45짝), finger PC det k = 3 0.7699(45짝)
- E15: cheetah 반복 12의 D_det 0.4296(none 892.26). E16: cheetah CEM식 none 830.381
- 모든 짝에서 seed = 100000 + episode, 짝 수 90(DMC 두 단계 1단계 칸은 45)

## 9. 비용과 시간

- td 지출(`runpod_tmlr/budget.py status`, 23:40Z, 모두 gone): td.1 $2.99, td.2 $2.35, td.3 $2.35, td.4 $2.51, td.5 $2.68, td.6 $2.36 → **$15.24**(묶음 상한 $24의 64%, 80% 선 안)
- 시간: 단계 A 18:01~19:17Z(td.5 생성 18:28Z). 본 실행 driver 18:48~23:33Z. td.2·td.3·td.6은 22:37Z, td.4·td.5는 22:53Z에 조정자가 결과를 당긴 뒤 지웠다. td.1은 23:33Z에 끝나 분석을 돌린 뒤 23:40Z에 지웠다(`pod_down.sh` → GraphQL 종료, budget.py에서 gone, `rp_gql.py pods atm-`에 td pod 없음).

## 10. 파일

- 판정·칸: `results/analysis/td_summary.json`, `td_tables.md`, `td_e11_cells.csv`, `td_e11_contrasts.csv`, `td_e12_cells.csv`, `td_e13_cells.csv`
- K2 결합용: `results/analysis/td_k2_e11_det_minus_detshift.csv`; episode 원자료 `results/analysis/td_episodes.csv`
- 원자료: `results/raw/td.1/`~`td.6/`(`jobs/{episodes,e13,prep,bias,timing,calib}`, `driver/decisions.jsonl`, `check3_<tag>.json`; 단계 A 자료 `raw/td.5/phaseA/`), td.1 driver 로그 `results/logs_td.1/`
- 코드: `code/`(실제로 돈 코드 전부)

## 11. [사후] X5: MW det+noise 비트 곡선(D6)

- 설계: DESIGN_tmlr.md §11.1 D6(자료 전 고정, commit 1d73e10; 사용자 승인 "너의 추천대로 해줘"). 기술 보고이며 판정 문턱이 없다.
- 실행: pod td.7(A40 SECURE, `l9usfc4n7b7evv`). MW 8과제 × det+noise × {PT k1~5, PC k1~3} × 90짝과 none 90짝, E12의 위쪽 확장 규칙(가장 높은 k에서 D > τ = 0.10이면 k = 8까지). episode 6,660개, job 결과 1332개, 모두 NVIDIA A40, (과제, 체크포인트)마다 host 하나(True), seed 불일치 0. 적응 결정 2개(`raw/td.7/driver/decisions.jsonl`).
- 범위: PT는 E0 파일, PC는 td.5·td.6에서 다시 잰 파일 24개를 그대로 복사(md5 24/24 일치). 코드는 td 본 실행 판(qlib.py md5 `fd1ceb9c…`, det+noise 구현 포함), 대기열만 `plan_x5.py`.
- 점검: 처음 끝난 none 8 job이 E11(td.5)의 같은 job과 비트 단위로 같았다(아래 재현 표에 전체).
- 출처: `results/analysis_x5/x5_summary.json`, `x5_kstar.csv`, `x5_curve.csv`, `x5_tables.md`. E12의 k\*는 `results/analysis/td_e12_cells.csv`다. E12 값은 다른 pod(같은 A40)의 결과라 짝이 아니므로 k\* 비교는 기술이다.

### 11.1 k\*(det+noise)와 E12 k\*

| 과제 | 범위 | k\*(det+noise) | E12 k\*: det / det+BC / sh-iter / det+shift | G_noise | noise ≤ sh-iter | noise ≤ min(sh-iter, det+shift) |
|---|---|---|---|---|---|---|
| mw-push | PT | 5 | 5 / 5 / 3 / 3 | 0 | 아니오 | 아니오 |
| mw-pick-place | PT | 4 | 5 / 4 / 4 / 4 | 0 | 예 | 예 |
| mw-door-open | PT | 5 | 5 / 5 / 2 / 3 | 0 | 아니오 | 아니오 |
| mw-drawer-open | PT | 1 | 4 / 3 / 1 / 1 | 2 | 예 | 예 |
| mw-button-press | PT | 3 | 4 / 3 / 1 / 1 | 0 | 아니오 | 아니오 |
| mw-peg-insert | PT | 4 | 5 / 6 / 3 / 3 | 1 | 아니오 | 아니오 |
| mw-window-open | PT | 4 | 5 / 5 / 5 / 3 | 1 | 예 | 아니오 |
| mw-sweep-into | PT | 3 | 4 / 4 / 2 / 2 | 1 | 아니오 | 아니오 |
| mw-push | PC | 2 | 2 / 2 / 2 / 2 | 0 | 예 | 예 |
| mw-pick-place | PC | 3 | 4 / 3 / 3 / 2 | 0 | 예 | 아니오 |
| mw-door-open | PC | 2 | 2 / 4 / 1 / 1 | 0 | 아니오 | 아니오 |
| mw-drawer-open | PC | 2 | 2 / 2 / 1 / 1 | 0 | 아니오 | 아니오 |
| mw-button-press | PC | 2 | 2 / 2 / 1 / 1 | 0 | 아니오 | 아니오 |
| mw-peg-insert | PC | 2 | 3 / 4 / 1 / 1 | 1 | 아니오 | 아니오 |
| mw-window-open | PC | 5 | 4 / 3 / 3 / 1 | -2 | 아니오 | 아니오 |
| mw-sweep-into | PC | 2 | 2 / 2 / 1 / 1 | 0 | 아니오 | 아니오 |

| 범위 | 과제 | k\*(noise) ≤ k\*(sh-iter) | k\*(noise) ≤ min(k\*(sh-iter), k\*(det+shift)) | G_noise ≥ 1 |
|---|---|---|---|---|
| PT | 8 | 3 | 2 | 4 |
| PC | 8 | 2 | 1 | 1 |

- k\* = 9는 k = 8까지도 D > τ라는 뜻이다(E12와 같은 정의).

### 11.2 det+noise 비트–하락 곡선 (D [95% CI], 90짝)

| 과제 | PT k: D [CI] | PC k: D [CI] |
|---|---|---|
| mw-push | 1: +32.2 [23.3, 42.2]; 2: +24.4 [15.6, 33.3]; 3: +8.9 [3.3, 15.6]; 4: +11.1 [5.6, 17.8]; 5: +0.0 [0.0, 0.0] | 1: +28.9 [20.0, 38.9]; 2: +3.3 [0.0, 7.8]; 3: +0.0 [0.0, 0.0] |
| mw-pick-place | 1: +68.9 [58.9, 78.9]; 2: +44.4 [34.4, 54.4]; 3: +15.6 [8.9, 23.3]; 4: +10.0 [4.4, 16.7]; 5: +2.2 [0.0, 5.6] | 1: +56.7 [46.7, 66.7]; 2: +17.8 [10.0, 25.6]; 3: +7.8 [2.2, 13.3] |
| mw-door-open | 1: +12.2 [5.6, 18.9]; 2: +14.4 [7.8, 22.2]; 3: +22.2 [14.4, 31.1]; 4: +13.3 [6.7, 21.1]; 5: +7.8 [3.3, 13.3] | 1: +36.7 [26.7, 46.7]; 2: +7.8 [3.3, 13.3]; 3: +8.9 [3.3, 15.6] |
| mw-drawer-open | 1: +8.9 [3.3, 15.6]; 2: +10.0 [4.4, 16.7]; 3: +1.1 [0.0, 3.3]; 4: +0.0 [0.0, 0.0]; 5: +0.0 [0.0, 0.0] | 1: +13.3 [6.7, 21.1]; 2: +0.0 [0.0, 0.0]; 3: +0.0 [0.0, 0.0] |
| mw-button-press | 1: +31.1 [22.2, 41.1]; 2: +41.1 [31.1, 51.1]; 3: +5.6 [1.1, 11.1]; 4: +0.0 [0.0, 0.0]; 5: +0.0 [0.0, 0.0] | 1: +30.0 [21.1, 40.0]; 2: +0.0 [0.0, 0.0]; 3: +0.0 [0.0, 0.0] |
| mw-peg-insert | 1: +68.9 [58.9, 78.9]; 2: +46.7 [36.7, 56.7]; 3: +32.2 [22.2, 42.2]; 4: +8.9 [3.3, 15.6]; 5: +2.2 [0.0, 5.6] | 1: +28.9 [20.0, 37.8]; 2: +3.3 [0.0, 7.8]; 3: +1.1 [0.0, 3.3] |
| mw-window-open | 1: +28.9 [20.0, 38.9]; 2: +17.8 [10.0, 25.6]; 3: +20.0 [12.2, 28.9]; 4: +8.9 [3.3, 15.6]; 5: +4.4 [1.1, 8.9] | 1: +13.3 [6.7, 21.1]; 2: +13.3 [6.7, 21.1]; 3: +17.8 [10.0, 25.6]; 4: +14.4 [7.8, 22.2]; 5: +0.0 [0.0, 0.0] |
| mw-sweep-into | 1: +13.3 [4.4, 22.2]; 2: +42.2 [32.2, 53.3]; 3: +4.4 [-1.1, 11.1]; 4: -3.3 [-7.8, 0.0]; 5: -3.3 [-7.8, 0.0] | 1: +13.3 [4.4, 22.2]; 2: -3.3 [-7.8, 0.0]; 3: -1.1 [-4.4, 2.2] |

### 11.3 E11 재현 점검 (PT 1차 k의 det+noise와 none, 같은 seed, 같은 기종의 다른 pod)

| 과제 | k | D_det+noise X5 [CI] | E11 [CI] | 같은 episode: none | det+noise |
|---|---|---|---|---|---|
| mw-push | 3 | +8.9 [3.3, 15.6] | +8.9 [3.3, 15.6] | 90/90 | 90/90 |
| mw-pick-place | 3 | +15.6 [8.9, 23.3] | +15.6 [8.9, 23.3] | 90/90 | 90/90 |
| mw-door-open | 4 | +13.3 [6.7, 21.1] | +13.3 [6.7, 21.1] | 90/90 | 90/90 |
| mw-drawer-open | 3 | +1.1 [0.0, 3.3] | +1.1 [0.0, 3.3] | 90/90 | 90/90 |
| mw-button-press | 3 | +5.6 [1.1, 11.1] | +5.6 [1.1, 11.1] | 90/90 | 90/90 |
| mw-peg-insert | 4 | +8.9 [3.3, 15.6] | +8.9 [3.3, 15.6] | 90/90 | 90/90 |
| mw-window-open | 4 | +8.9 [3.3, 15.6] | +8.9 [3.3, 15.6] | 90/90 | 90/90 |
| mw-sweep-into | 3 | +4.4 [-1.1, 11.1] | +4.4 [-1.1, 11.1] | 90/90 | 90/90 |

### 11.4 읽기 (기술)

- det+noise는 E11 1차 k에서 MW 8칸을 모두 고쳤지만(같은 칸이 여기서 비트 단위로 재현됨), 비트를 내리면 dither만큼 버티지 못한다. PT에서 k\*(det+noise)가 k\*(sh-iter) 이하인 과제는 3/8(pick-place, drawer-open, window-open), sh-iter와 det+shift 둘 다 이하인 과제는 2/8(pick-place, drawer-open)이다. PC에서는 각각 2/8(push, pick-place), 1/8(push)이다.
- det 계열 대비 번 비트 G_noise ≥ 1은 PT 4/8(drawer-open 2, peg-insert·window-open·sweep-into 1), PC 1/8(peg-insert)이다. E12의 dither G(PT 7/8, PC 7/8)보다 적다.
- 곡선이 단조가 아닌 곳이 많다: PT에서 door-open(k = 3에서 +22.2), button-press(k = 2에서 +41.1), sweep-into(k = 2에서 +42.2)는 한 비트 아래보다 나쁘다. window-open PC는 k = 1~4에서 +13.3~+17.8로 τ 위에 머물다 k = 5에서 0이 되어 확장 규칙으로 k = 4·5가 더해졌다(결정 2개, `raw/td.7/driver/decisions.jsonl`).
- E12의 k\*는 다른 pod(td.5·td.6)의 값이라 짝이 아니다. 다만 같은 기종에서 none 720 episode와 1차 k det+noise 720 episode가 모두 E11과 비트 단위로 같았으므로(11.3), 다른 칸도 같은 코드·seed·기종이면 같은 결과가 나왔을 것으로 본다(확인한 것은 이 칸들뿐).

### 11.5 수치 대조, 비용, 파일

- 원자료(`results/raw/td.7/jobs/episodes/`)에서 점 추정 8개를 순수 Python으로 다시 셌다. 모두 분석과 같다: push PT k3 0.0889, sweep-into PT k2 0.4222, window-open PC k4 0.1444·k5 0.0000, pick-place PC k1 0.5667, door-open PT k3 0.2222, drawer-open PT k1 0.0889, peg-insert PT k4 0.0889(모두 90짝, seed = 100000 + episode).
- 실행: driver 2026-11-22T02:14:54Z ~ 03:36:50Z(4,916초, job 1,332개 = static 1,296 + 확장 36, 실패 0). 분석 03:37Z(pod 위). 회수: 원격 1,365파일 = 로컬 1,365파일, rsync dry-run 남은 전송 0. `pod_down.sh td.7` 03:38Z, budget.py에서 gone.
- 비용: td.7 $0.88(budget.py 03:38Z). td 묶음 합 $16.12(상한 $24).
- 파일: `results/analysis_x5/x5_summary.json`(곡선·k\*·과제 수·재현), `x5_kstar.csv`, `x5_curve.csv`, `x5_tables.md`, `input_td_e12_cells.csv`(E12 입력 사본), `code_md5.txt`; 원자료 `results/raw/td.7/`; driver 로그 `results/logs_td.7/`; 코드 `code/plan_x5.py`(6217b554…), `code/analyze_x5.py`(203b9813…), `code/x5_section.py`(이 절의 표 생성).

## 12. [사후] X6: 정책 기준선(D7)

- 설계: DESIGN_tmlr.md §11 D7·§11.1 "D7 — X6"(자료 전 고정, commit 24a59a00; 사용자 승인 "돌려줘"). 기술 보고이며 판정 문턱이 없다.
- 실행: pod td.8(A40 SECURE, `rqvrq4u986pqia`). E11의 12과제 × 90짝(체크포인트 1~3 × seed 100000~100029) × {none(기본 MPPI, FP32), policy(TD-MPC2의 `mpc=False` 경로: 정책 π의 평균 행동 `info["mean"]`, eval_mode, 계획 없음)}. episode 2,160개, job 결과 432개(none 216 + policy 216, 실패 0), 모두 NVIDIA A40, (과제, 체크포인트)마다 host 하나(True), seed 불일치 0.
- 코드: td 본 실행 판(qlib.py `fd1ceb9c…`, plan_td.py `0d5a8a70…`, driver.py `05ff35f3…`, tdjobs.py `22607b64…`)에 `runner.py`의 `mpc=False` 선택만 더했다(`3782a3c5…`). 대기열은 `plan_x6.py`(`962cd9c8…`)다.
- 점검 (1): td.8의 none이 E11(td.1~td.6)의 none과 12과제 1,080/1,080 episode에서 비트 단위로 같다(return·성공). 시작 때 처음 끝난 none 8 job으로 먼저 확인했고, 끝난 뒤 전체를 다시 맞췄다.
- 점검 (2): policy 경로에서 dynamics·reward 층(dyn0·dyn1·rew0·rew1)의 forward 호출이 12과제 모두 0이다(과제마다 체크포인트 1, seed 100000의 한 episode, MW 100 step·DMC 500 step). 같은 훅으로 센 MPPI 3 step은 dyn0 60·dyn1 60·rew0 54·rew1 54회다. `TDMPC2.act` 원문도 남겼다(`if self.cfg.mpc:`이면 `plan`, 아니면 `self.model.pi(z, task)`, eval_mode에서 `info["mean"]`).
- 비교 규칙(자료 전에 정함): E12에서 dither 구성(PT·PC × sh-iter·det+shift) 가운데 k\*가 가장 낮은 것(같으면 k\*에서의 D가 작은 것, 그다음 PT·PC, sh-iter·det+shift 순)의 D와 CI를 쓴다. 두 CI가 겹치지 않으면 "낫다"(dither CI 상한 < policy CI 하한) 또는 "못하다"(dither CI 하한 > policy CI 상한), 겹치면 "같다"로 센다. E12 값은 다른 pod(td.1~td.6, 같은 A40·코드·seed)의 결과라 td.8과 짝이 아니다.
- 출처: `results/analysis_x6/x6_summary.json`, `x6_cells.csv`, `x6_tables.md`, `check_policy_calls.json`. E12 입력은 `results/analysis/td_e12_cells.csv`(사본 `results/analysis_x6/input_td_e12_cells.csv`, md5 `3fd6f967…` 같음).

### 12.1 과제별 policy와 E12의 가장 낮은 dither 구성 (D [95% CI], none 대비)

| 과제 | none | policy | D_policy [CI] | E12 가장 낮은 dither 구성(k\*) | D [CI] (짝) | dither 대 policy | 참고: 모든 반올림 중 가장 낮은 구성 |
|---|---|---|---|---|---|---|---|
| mw-push | 1.000 | 0.922 | +7.8 [2.2, 13.3] | PC sh-iter k = 2 | +0.0 [0.0, 0.0] (90) | 낫다 | PC sh-iter k = 2: +0.0 |
| mw-pick-place | 1.000 | 0.767 | +23.3 [14.4, 32.2] | PC det+shift k = 2 | +10.0 [4.4, 16.7] (90) | 같다 | PC det+shift k = 2: +10.0 |
| mw-door-open | 1.000 | 0.989 | +1.1 [0.0, 3.3] | PC det+shift k = 1 | +4.4 [1.1, 8.9] (90) | 같다 | PC det+shift k = 1: +4.4 |
| mw-drawer-open | 1.000 | 1.000 | +0.0 [0.0, 0.0] | PC sh-iter k = 1 | +1.1 [0.0, 3.3] (90) | 같다 | PC sh-iter k = 1: +1.1 |
| mw-button-press | 1.000 | 1.000 | +0.0 [0.0, 0.0] | PC sh-iter k = 1 | +0.0 [0.0, 0.0] (90) | 같다 | PC sh-iter k = 1: +0.0 |
| mw-peg-insert | 1.000 | 0.989 | +1.1 [0.0, 3.3] | PC sh-iter k = 1 | +3.3 [0.0, 7.8] (90) | 같다 | PC sh-iter k = 1: +3.3 |
| mw-window-open | 1.000 | 0.933 | +6.7 [2.2, 12.2] | PC det+shift k = 1 | +7.8 [2.2, 13.3] (90) | 같다 | PC det+shift k = 1: +7.8 |
| mw-sweep-into | 0.967 | 0.978 | -1.1 [-5.6, 3.3] | PC det+shift k = 1 | +4.4 [-2.2, 11.1] (90) | 같다 | PC det+shift k = 1: +4.4 |
| cheetah-run | 900.9 | 419.0 | +53.5 [49.4, 57.5] | PC det+shift k = 4 | +5.7 [4.5, 7.0] (45) | 낫다 | PC det+shift k = 4: +5.7 |
| walker-walk | 984.6 | 430.1 | +56.3 [52.3, 60.3] | PC det+shift k = 3 | +4.2 [3.0, 5.5] (45) | 낫다 | PC det+shift k = 3: +4.2 |
| quadruped-walk | 971.7 | 838.8 | +13.7 [11.8, 15.5] | PC det+shift k = 3 | +1.9 [1.0, 2.8] (45) | 낫다 | PC det k = 3: +1.6 |
| finger-turn-hard | 969.1 | 908.1 | +6.3 [0.9, 12.1] | PC det+shift k = 3 | +1.0 [-6.0, 7.5] (45) | 같다 | PC det+shift k = 3: +1.0 |

- 과제 수(dither 대 policy): MW 낫다 1 · 같다 7 · 못하다 0, DMC 낫다 3 · 같다 1 · 못하다 0.
- D는 MW 성공률 %p, DMC 상대 return %다(none 대비). none은 td.8의 none(E11과 같음)이다.
- 이 표에 놓은 E12 DMC 칸은 모두 n = 45다(두 단계 칸; CI는 중간 확인을 보정하지 않은 값, §3). policy와 none은 모든 과제에서 90짝이다.

### 12.2 읽기 (기술)

- 계획 없는 정책(FP32 π)은 MW에서 none에 가깝다: drawer-open·button-press 1.000(D 0), door-open·peg-insert 0.989(+1.1%p), sweep-into 0.978(none 0.967보다 1.1%p 높음), window-open 0.933(+6.7), push 0.922(+7.8), pick-place 0.767(+23.3). D_policy의 CI 하한이 0보다 큰 MW 과제는 push·pick-place·window-open 3개다.
- DMC에서는 정책만으로 return이 크게 준다: cheetah-run 900.9 → 419.0(+53.5%), walker-walk 984.6 → 430.1(+56.3%), quadruped-walk 971.7 → 838.8(+13.7%), finger-turn-hard 969.1 → 908.1(+6.3%). 네 과제 모두 CI 하한이 0보다 크다.
- E12의 가장 낮은 dither 구성(모두 PC; MW k = 1~2, DMC k = 3~4)과 견주면: DMC는 cheetah·walker·quadruped 3과제에서 dither 구성이 낫고(예: cheetah +5.7 [4.5, 7.0] 대 +53.5 [49.4, 57.5]), finger는 같다(+1.0 [−6.0, 7.5] 대 +6.3 [0.9, 12.1]). MW는 push 하나만 낫고(PC sh-iter k = 2 +0.0 [0.0, 0.0] 대 +7.8 [2.2, 13.3]), 나머지 7과제는 CI가 겹친다. 그 7과제 가운데 pick-place는 dither 쪽 점 추정이 작고(+10.0 대 +23.3; dither 상한 16.7 > policy 하한 14.4라 겹침), door-open·drawer-open·peg-insert·window-open·sweep-into 5과제는 policy 쪽 점 추정이 작으며, button-press는 둘 다 0이다. "못하다"는 0과제다.
- 비교의 범위: dither 구성은 dynamics·reward 은닉층(dyn0·dyn1·rew0·rew1)만 k비트로 둔 MPPI이고 encoder·π·Q는 FP32다. policy는 같은 FP32 encoder·π를 계획 없이 쓴다. 따라서 이 표는 "저비트 world model로 계획하기"와 "world model 없이 정책만 쓰기"의 비교이며, 정책 자체의 양자화는 재지 않았다.
- 추가 확인(자료 전 규칙 밖, 기술): 같은 k\*에 묶인 다른 dither 구성으로 바꿔 보면 DMC 네 과제의 판정은 그대로다(walker PC sh-iter +5.3 [3.8, 6.9]·quadruped PC sh-iter +2.2 [1.5, 3.1]도 낫다; finger PC sh-iter k = 3(90짝) +2.1 [−1.5, 6.1]도 같다). MW는 push에서 PC det+shift k = 2(+3.3 [0.0, 7.8])를 쓰면 "같다"가 되고, drawer-open에서 PT sh-iter k = 1(+4.4 [1.1, 8.9])을 쓰면 "못하다"가 된다(policy +0.0 [0.0, 0.0]). 나머지 MW 묶음은 모두 "같다"다. 그러므로 MW의 "낫다 1"과 "못하다 0"은 묶음 규칙에 기대는 값이다(묶음은 `x6_summary.json`의 `ties`, 값은 `td_e12_cells.csv`).

### 12.3 수치 대조, 비용, 파일

- 원자료(`results/raw/td.8/jobs/episodes/`)에서 점 추정 8개를 numpy 없이 순수 Python으로 다시 셌다. 모두 분석과 같다(D_policy; none·policy 평균도 같음, 90짝, seed = 100000 + episode): push 0.0778, pick-place 0.2333, peg-insert 0.0111, sweep-into −0.0111, cheetah 0.5350, walker 0.5632, quadruped 0.1367, finger 0.0630. none의 E11 비트 동일성도 원래 td.1~td.6 원자료에서 door-open·walker-walk 두 과제를 다시 셌다(90/90, 90/90).
- 실행: driver 2026-11-22T09:30:00Z ~ 09:56:07Z(1,567초, job 432개, 실패 0). 분석 09:56Z(pod 위). 회수: 원격 out 441파일 = 로컬 441파일, logs 8 = 8, rsync dry-run 남은 전송 0. `pod_down.sh td.8` 09:58:53Z(REST 실패 → GraphQL로 삭제), budget.py에서 gone, `rp_gql.py pods atm-` 빈 목록.
- 비용: td.8 $0.33(budget.py 09:59Z). td 묶음 합 $16.45(상한 $24).
- 파일: `results/analysis_x6/x6_summary.json`(과제별 none·policy·D_policy·CI, E12 구성과 묶음, 판정, E11 동일성), `x6_cells.csv`, `x6_tables.md`(`x6_section.py` 출력), `check_policy_calls.json`, `input_td_e12_cells.csv`(E12 입력 사본), `code_md5.txt`; 원자료 `results/raw/td.8/`; driver 로그 `results/logs_td.8/`; 코드 `code/plan_x6.py`(962cd9c8…), `code/analyze_x6.py`(2eed6d41…), `code/check_policy_calls.py`(d41edace…), `code/x6_section.py`(37bc3cdf…, 12.1 표 생성), `code/runner.py`(3782a3c5…, `mpc=False` 선택).
