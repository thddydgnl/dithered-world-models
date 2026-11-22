## 12. [사후] X6: 정책 기준선(D7)

- 설계: DESIGN_tmlr.md §11.1 D7(자료 전 고정, commit 24a59a00; 사용자 승인 "돌려줘"). 기술 보고이며 판정 문턱이 없다.
- 실행: pod td.8(A40 SECURE). E11의 12과제 × 90짝(체크포인트 1~3 × seed 100000~100029) × {none(기본 MPPI, FP32), policy(TD-MPC2의 `mpc=False` 경로: 정책 π의 평균 행동, eval_mode)}. episode 2,160개, job 결과 432개, 모두 NVIDIA A40, (과제, 체크포인트)마다 host 하나(True), seed 불일치 0.
- 점검: td.8의 none이 E11의 none과 1080/1080 episode에서 비트 단위로 같다 (return·성공). policy 경로는 dynamics·reward 층(dyn0·dyn1·rew0·rew1)의 호출 수가 모든 과제에서 0이다(`check_policy_calls.json`).
- 비교 규칙(자료 전에 정함): E12에서 dither 구성(PT·PC × sh-iter·det+shift) 가운데 k\*가 가장 낮은 것(같으면 k\*에서의 D가 작은 것, 그다음 PT·PC, sh-iter·det+shift 순)의 D와 CI를 쓴다. 두 CI가 겹치지 않으면 "낫다"(dither CI 상한 < policy CI 하한) 또는 "못하다", 겹치면 "같다"로 센다. E12 값은 다른 pod(같은 A40, 같은 seed)의 결과라 짝이 아니다.
- 출처: `results/analysis_x6/x6_summary.json`, `x6_cells.csv`, E12 입력 `results/analysis/td_e12_cells.csv`.

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

