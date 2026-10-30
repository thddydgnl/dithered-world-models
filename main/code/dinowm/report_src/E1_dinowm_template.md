# E1 DINO-WM 결과 보고 (A 논문: 주 벤치마크, DINO-WM 계열)

- 작성: 2026-10-22, DINO-WM 작업자. E1 마지막 칸은 18:56Z에 끝났다
- 기준: [../DESIGN_main.md](../DESIGN_main.md)
  - §4(v2): DINO-WM 칸·변형·분할
  - §5(v3): OEI 확증 지표 `oei_ref_mf`, PushT eval6 쌍
  - §6(v4): 두 계열 공통 판정 계산
  - §9(v5.2): E1 뒤에 더한 사전등록 (b1)·(b2)·(b3)
- 코드: [../code/dinowm/](../code/dinowm/)
  - `analyze_e1.py`: 10-21T23:10Z에 PushT 결과를 보기 전에 썼고, v4가 공통 규칙으로 채택했다
  - `analyze_e1_v52.py`, `drift_ref.py`: v5.2 고정 뒤, DINO-WM 분해를 계산하기 전에 썼다
  - `e1_tables.py`, `e1_finalize.sh`
- 결과: [../results/E1/](../results/E1/)
  - job JSON: `E1_dinowm_wp/`, `E1_dinowm_pusht/`
  - `e1_dinowm_summary.json`·`.txt`, `e1_dinowm_cells.csv`, `e1_dinowm_tables.md`
  - `e1_dinowm_v52.json`, `e1_dinowm_v52_tables.md`, `drift_ref/`, `mf_same/`, `it0_cost_std.json`
- 작업 로그: [../worklogs/E1_dinowm.md](../worklogs/E1_dinowm.md)

## 0. 요약: 판정

| 가설 | 단위 | 판정 | 핵심 수치 |
|---|---|---|---|
| **H1** | DINO-WM | **성립** | det 하락이 실재하는 자격 칸 5개: PushT k=3·4·5, PointMaze k=2·3. 5칸 모두에서 det의 계획 시점 1-step 오차가 sh-iter 이하다 |
| **H3a** | DINO-WM(3과제 9칸) | **성립** | 자격 5칸 중 4칸(80%)에서 red_sh-iter ≥ 0.5다(PushT k=5만 0.20). 해로운 칸은 없다. 하한이 가장 큰 칸은 Wall k=4의 D_sh-iter − D_det +15%p [4, 26]이다 |
| **H2a** | PushT k=4 | **성립(경계)** | det oei_ref_mf −0.297 [−0.350, −0.248]. sh-iter +0.0048 [−0.000008, +0.0099]이고, CI가 0을 아주 조금 포함한다 |
| **H2b** | DINO-WM(45점) | **불성립** | 과제 평균 Spearman 차 **−0.144** < 0.15(PushT −0.556, Wall +0.312, PointMaze −0.189). 보조 계산 셋도 모두 음수다 |
| (b1) H2a_indep | PushT | 성립 | 동결 정의 oei_ref_mf가 곧 독립 draw판이라 H2a와 같다 |
| (b2) H2a′ | PushT k=4 | **성립** | drift_det −0.269 [−0.318, −0.224], r_sh-iter −0.013 [−0.032, +0.004]. **양성 대조 r_sh-plan +0.011 [−0.009, +0.031]**는 예상(CI 하한 > 0.5)과 다르다 |
| 탐색 | Wall k=3 | (판정 밖) | det 하락 +50%p [39, 61]. dither 4종의 red는 0.92~1.08이다 |

- **go/no-go 입력(§6.1)**: DINO-WM H3a가 성립한다. TD-MPC2 Meta-World H3a도 성립했으므로(E1_tdmpc2 보고) 규칙상 **go**다.
- **경계에 가까운 점**(모두 사전등록 규칙대로 판정했고, 다시 계산해도 같은 값이다)
  1. H3a 비율 4/5 = 80%는 문턱 75%를 한 칸 차이로 넘는다. 자격 칸이 하나 더 있고 그 칸이 red < 0.5였다면 4/6 = 67%로 불성립이다.
  2. PushT k=5는 det 하락이 +5%p [1, 10]로 작다. 한쪽 bootstrap p = 0.0076으로 Holm 문턱 0.025를 통과해 자격이 됐지만, McNemar p는 0.0625다. 이 칸의 red_sh-iter가 0.20으로 H3a의 유일한 실패 칸이다.
  3. 해 점검: Wall k=4에서 D_sh-iter − D_det의 CI 하한이 +4%p로, 문턱 +5%p에 1%p 모자란다. 이 칸은 det가 none보다 14%p 나은 칸이고(det 75%, none 61%), dither 4종은 모두 none 수준(58~67%)으로 돌아간다. 즉 "det가 none보다 나은 칸에서 dither가 그 차이를 없앤다"에 해당하며, 규칙상 해로운 칸은 아니다.
  4. H2a의 sh-iter OEI CI 하한은 −8×10⁻⁶이다.

{{STATIC}}

## 3. det 하락과 자격 칸 (과제별 Holm, 한쪽 α = 0.025)

{{T:표 1. det 하락과 자격 칸 (확증 칸)}}

- **Holm 세부**(과제마다 확증 k 3칸)
  - PushT: p = 1.0e−4(k=3), 1.0e−4(k=4), 0.0076(k=5). 기준 0.0083 / 0.0125 / 0.025를 모두 통과해 3칸 모두 실재다. k=3과 k=4의 p는 bootstrap 최솟값 1/(B+1)에서 같다
  - PointMaze: p = 0.0006(k=2), 0.0088(k=3), 0.567(k=4)
    - k=2(기준 0.0083)와 k=3(기준 0.0125)은 실재다
    - k=4는 아니다(기준 0.025)
  - Wall: 가장 작은 p가 0.278(k=5)이라 실재 칸이 없다
    - k=4는 det가 none보다 **나은** 칸이다: −14%p [−24, −4], McNemar p = 0.0125
- 보정 없는 CI 하한 > 0 기준으로 봐도 자격 칸은 같은 5칸이다
- 1차 k와 비교(held 선택, §4.1)
  - PushT: 1차 k = 4다. 이웃 k=3(+36%p)과 k=5(+5%p)도 자격이 됐다
  - Wall: 1차 k = 5(held에서 20%p)인데, eval에서는 +3%p로 실재하지 않았다
  - PointMaze: held에서 20%p에 이르는 k가 없어 대체 규칙으로 1차 k = 3을 정했다. eval에서는 k=2와 k=3이 자격이다

## 4. H1: 성립

{{T:표 2. H1: 자격 칸의 계획 시점 1-step 오차 (det vs sh-iter)}}

- det의 1-step 오차는 자격 5칸 모두에서 sh-iter보다 작다. 짝 CI 상한도 모두 0보다 작다
- 그런데도 손상은 det에서만 크다(§9 표 4). 1-step 오차로는 det 손상이 보이지 않는다는 진단(v5.2 §9.1의 진단 기여)과 맞는다
- 3·5-step에서는 det 오차가 dither보다 크다
  - PushT k=4 5-step: det 0.250, sh-iter 0.042
  - PointMaze k=3: det 1.083, sh-iter 0.483
  - det 오차는 horizon을 따라 쌓이고(체계적 편향), dither 오차는 쌓이지 않는다
- 보조(데이터셋형 입력 오차)도 같은 방향이다. 자격 5칸 중 PointMaze k=2를 뺀 4칸에서 det ≤ sh-iter다(k=2는 0.700 대 0.694)

## 5. H3a: 성립

{{T:표 3. H3a: red_sh-iter (자격 칸)와 해 점검 (모든 확증 칸)}}

- red_sh-iter ≥ 0.5인 칸은 4/5 = 80%다
  - PushT k=3: 0.92
  - PushT k=4: 1.00
  - PointMaze k=2: 0.69
  - PointMaze k=3: 1.09
- PushT k=5의 red는 0.20 [−2.00, 0.86]이다. det 하락 자체가 5%p라 비율의 CI가 매우 넓다
  - 같은 칸의 다른 dither: ind 0.60, sh-plan 0.80, det+shift 1.00
  - 손상 크기로 보면 sh-iter도 +4%p [1, 8]로 det와 같은 수준이다
- 해 점검: CI 하한이 +5%p를 넘는 확증 칸은 없다. 가장 큰 하한은 Wall k=4의 +4%p다(§0 경계 3)
- 판정 단위는 DINO-WM 9칸이다. 탐색 칸 Wall k=3은 넣지 않았다. 넣어도 red가 0.92~1.08이라 판정은 바뀌지 않는다

## 6. H2a: PushT 성립(경계)

{{T:표 7. H2a (PushT 1차 k = 4)}}

- H2a 전체 판정은 4과제 중 3과제 이상이어야 성립이다
  - TD-MPC2 3과제(mw-push, mw-pick-place, cheetah-run)는 모두 불성립이었다(E1_tdmpc2 §6)
  - 그래서 PushT가 성립해도 H2a는 불성립이다
- PushT 성립은 sh-iter CI 하한 −8×10⁻⁶에 기대고 있다. sh-iter의 점추정은 +0.0048이다
  - 부호는 이용이 아니라 반대쪽이다
  - 크기는 det의 1.6%다
- 보조(§9 표 6)
  - oei_ref_argmin: det −0.305, sh-iter −0.003으로 같은 그림이다
  - 시뮬레이터 참 비용 기준 oei_true_mf: det −0.195, sh-iter +0.077이다. dither 4종이 모두 +0.07~+0.08이다(참 비용 기준의 보정식 오차가 섞인 값이다)

## 7. H2b: 불성립

{{T:표 8. H2b (확증 칸, 점 = (과제, k, 변형))}}

- 과제 안 Spearman 차의 평균이 −0.144다. Wall만 양수(+0.312)다
- PushT에서는 −OEI와 손상의 상관이 음수(−0.244)다
  - det의 OEI가 k=3에서 **양수**(+0.217)인데, 이 칸의 손상이 가장 크다(+36%p)
  - k=4에서는 음수(−0.297, 손상 +24%p)이고, k=5에서는 양수(+0.030, 손상 +5%p)다(§10)
- 섞은 원값(−0.420), OEI 정규화(−0.367), 데이터셋형 오차(과제 평균 −0.172, 섞은 값 −0.359) 모두 불성립 방향이다
- TD-MPC2 E1(과제 평균 0.079)과 함께, v5.2가 진단 주장 "OEI가 1-step 오차보다 손상을 잘 예측한다"를 뺀 결정과 맞는다

## 8. (v5.2) E1 뒤에 더한 사전등록: (b1)·(b2)와 분해 기술, (b3)

{{V52DEF}}

### 8.1 판정 (PushT k=4)
- **(b1) H2a_indep**: 성립. §6의 H2a와 같은 계산이다
- **(b2) H2a′**: 성립
  - 조건 1: drift_det −0.269 [−0.318, −0.224], 상한 < 0
  - 조건 2: r_sh-iter −0.013 [−0.032, +0.004], 상한 < 0.5. 버린 재표본은 0개다
- **양성 대조**(판정 밖): r_sh-plan = +0.011 [−0.009, +0.031]
  - 예상(CI 하한 > 0.5)과 다르다
  - 계획 동안 고정된 확률 표(sh-plan)도 drift가 거의 없다(−0.003 [−0.008, +0.002])
  - 같은 칸의 det drift는 −0.269다
- **기술 보고**
  - r_ind −0.009 [−0.019, −0.0001], r_det+shift +0.003 [−0.009, +0.013]
  - sel_last는 같은 draw판(elite)이 −0.002~−0.006(확률적 변형), −0.028(det)이다
  - 독립 draw판은 −0.000~+0.003(확률적 변형), −0.028(det)이다
- H2a′ 전체(4과제 중 3과제)는 TD-MPC2 E2 base 자료로 정해진다. PushT는 그중 한 과제다
- 표본 밖 검정이라는 점(TD-MPC2 자료로만 세운 가설)에서 PushT 성립은 의미가 있다. 다만 아래 (b3)이 해석을 좁힌다

### 8.2 (b3) 영역 효과 대조 (기술 보고)
- PushT k=4: drift_ref −0.177 [−0.221, −0.137]이다. det drift −0.269 가운데 약 2/3에 해당한다
  - drift_ref는 비양자화 계획기의 첫·마지막 모집단을 det 모델로 평가한 값이다
  - 비양자화 계획기가 가는 영역에서 이미 det 오차가 유리하다는 뜻이다
- det 계획기 자신의 몫(drift_det − drift_ref)은 −0.092 [−0.108, −0.076]이다. CI가 0과 떨어져 있다
- 다른 칸
  - PushT k=3: drift_ref +0.448 [+0.390, +0.512], det − ref −0.206 [−0.235, −0.178]
  - PushT k=5: drift_ref +0.035, det − ref −0.003
  - Wall k=4: drift_ref +0.165, det − ref −0.035
  - Wall k=5: −0.109, det − ref −0.014
  - Wall k=3(탐색): −1.223, det − ref −0.640
  - PointMaze k=2·3: +0.017·+0.008, det − ref −0.028·−0.013
- 모든 칸에서 drift_det − drift_ref는 0 또는 음수다

### 8.3 분해 표 (모든 E1 칸, 기술 보고)

{{V:decomp}}

{{V:ref}}

- r의 "—"는 재표본을 모두 버린 칸이다. det drift가 0보다 크거나 0 근처면(Wall k=4, PushT k=3·5, PointMaze k=4) 비가 정의되지 않는다
- 항등식 oei_ref_mf = sel_indep + drift: 모든 기록에서 오차 0이다
- 같은 draw mf판은 sh-iter·det+shift를 마지막 반복의 표로 다시 평가한 값이다
  - 점검: it = opt_steps의 draw로 다시 잰 값이 저장된 J_v_mf와 모든 episode에서 비트 단위로 같았다
  - drift_ref 점검(none 재계산, det 첫 반복)도 모두 0이었다

## 9. 칸별 표

### 9.1 확증 칸의 모든 변형 (표 4)

{{T:표 4. 확증 칸의 모든 변형}}

### 9.2 탐색 칸 Wall k=3 (표 5, H1·H3a에 넣지 않음)

{{T:표 5. 탐색 칸 (Wall k=3, H1·H3a에 넣지 않음)}}

### 9.3 OEI (표 6, 확증 지표 oei_ref_mf, 음수 = 이용)

{{T:표 6. OEI (확증 지표 oei_ref_mf, 음수 = 이용)}}

## 10. 관찰 (사후, 판정에 쓰지 않음)
- **det OEI의 부호는 손상을 따라가지 않는다**
  - 음수 OEI(이용)이면서 손상이 실재하는 칸은 PushT k=4(−0.297, +24%p)와 PointMaze k=2·3(−0.038·−0.029, +16·+11%p)이다
  - 손상이 가장 큰 PushT k=3은 OEI +0.217(drift +0.243)이다
  - Wall k=5는 OEI −0.151인데 손상이 없다(+3%p)
  - Wall k=4는 OEI +0.171이고 det가 none보다 낫다
  - (b3)로 보면 PushT k=3의 양수 OEI는 영역 효과에서 온다
    - 비양자화 계획기가 가는 영역 자체에서 det 오차가 크게 불리하다(drift_ref +0.448)
    - det 계획기는 그보다 −0.206만큼 유리한 쪽으로 간다
  - 영역 효과를 빼면 det 계획기 자신의 몫(drift_det − drift_ref)은 모든 E1 칸에서 0 이하다
  - 그래서 OEI(전체)의 부호는 "이용"과 "영역"이 섞인 값이고, 칸 사이 손상과 단조로 이어지지 않는다(H2b 불성립과 맞음)
- **오차 크기**
  - det+shift의 계획 시점 오차는 모든 horizon에서 다른 dither의 약 절반이다(예: PushT k=4 1-step 0.031 대 0.063)
  - 손상 회복은 다른 dither와 같다
- **PointMaze k=2**
  - 모든 변형의 상대 1-step 오차가 2.5 안팎이다. 양자화 오차가 예측 분산보다 크다
  - 그런데도 dither는 손상의 69~100%를 없앤다
- **계획 편차**(첫 행동 RMS / nonealt seed 바닥)
  - 자격 칸에서 det가 dither보다 크다. 예: PushT k=3 0.71 대 0.58~0.63, PointMaze k=2 0.90 대 0.82~0.85
  - 다만 none 대비 편차가 seed 바닥보다 작은 값(< 1)이라, 계획 편차만으로는 칸을 가르기 어렵다
- **none과 nonealt**: 세 과제 모두 성공률 차가 seed 잡음 범위 안이다(McNemar p 0.29~1)

## 11. 주장 수준의 결과와 조정자 결정이 필요한 것
- **처방(H3a)**: DINO-WM에서 성립한다. TD-MPC2 Meta-World와 함께 §6.1의 go 조건을 채운다
  - 경계가 가깝다: 80% 대 75%, Wall k=4 하한 4%p 대 5%p
  - 논문 문장은 "자격 5칸 중 4칸에서 det 손상의 50% 이상을 없앤다(3칸은 90% 이상). 손상이 5%p로 작은 PushT k=5에서는 효과가 불분명하다" 수준이 맞다
- **진단(H1)**: 두 계열 모두 성립한다. 1-step 오차는 손상을 보지 못하고, 3·5-step 오차는 det를 가른다
- **기전**(v5.2로 좁힌 "지속 오차 이용")
  - PushT H2a′는 표본 밖에서 성립한다
  - 다만 두 가지가 해석을 좁힌다
    1. sh-plan 양성 대조가 실패했다. DINO-WM에서는 "지속되기만 하면 이용된다"가 아니다. det의 체계적 편향(입력에 따라 같은 방향)이 이용되는 것으로 보인다
    2. (b3)에서 PushT k=4 det drift의 약 2/3가 영역 효과다
  - 영역 효과를 뺀 det 계획기 자신의 몫은 모든 칸에서 0 이하다(PushT k=3 −0.206, k=4 −0.092)
  - 기전 문장은 미룬다(조정자, 10-22 19:35Z). 최종 문장은 E1~E3를 모두 본 뒤 A 최종 보고에서 사용자에게 제안한다
    - E2(TD-MPC2 b2·b3, 새 seed)와 E3 표 크기 sweep이 이 문장을 바꿀 두 가지를 직접 잰다. 하나는 작은 N_eff에서 지속이 이용되는지, 다른 하나는 drift 중 영역 효과의 몫이다
    - 그때까지 sh-plan 양성 대조와 영역 효과는 기술 관찰로만 적는다
- **H2a·H2b**: 두 계열 모두 불성립이다. 사전등록 판정 그대로 보고한다
- **E5 자격**: E1에서 det 하락이 실재한 3~4-bit 확증 칸은 PushT k=3·4와 PointMaze k=3이다
  - E5 held·eval manifest는 이 3칸으로 만든다
  - PointMaze PTQ 보정은 GPU 1에서 한다

## 12. 편차와 운영 메모
- **사전등록 계산에서 벗어난 것은 없다.** 판정은 모두 v4·v5.2 규칙 그대로다
- 운영
  - **PushT E1 중 GPU 0 공유**: 07:39~08:05Z E5 스모크, 09:10~09:14Z N64·E5 스모크
    - 그동안 PushT runner가 느려졌다
    - 계획은 결정적이고 E·GPU에 불변이라(E0 §2) 결과에는 영향이 없다
  - **GPU 1 공유**: B가 ~11:40Z까지 같이 썼다. PointMaze job의 GPU-h는 공유 때문에 부풀었다
    - 14:13~15:03Z에는 내 drift_ref·mf_same(Wall·PointMaze)이 k=5와 같이 돌았다
  - **PushT 칸 배정**: k=4·k=3은 GPU 0, k=5는 GPU 1이다. k=3·k=5는 claim으로 미리 배정했다
- 정의 세부(계산 전에 정함)
  - H2b 보조 정규화의 none 첫 반복 비용 std는 무작위 후보 299개로 잰다(행 0 = mu 제외, OEI 기준과 같음)
  - v5.2 같은 draw판 sel_last는 두 가지다
    - elite판: 모든 변형
    - mf판: sh-iter·det+shift만. ind는 정의되지 않는다
- 서버 runner의 오래된 lock 파일 2개(JSON이 이미 있음)는 무해하다
- 규칙 위반 메모: 09:00Z쯤 it0_cost_std.py 시험 출력을 서버 /tmp에 쓰고 지웠다(내가 만든 파일 하나)
  - ~/pmx 밖에서 지운 것이라 "~/pmx 밖에서 지우지 않는다" 규칙에 어긋난다. 영향은 없다
  - 이후 임시 파일은 ~/pmx 안에만 쓴다

## 13. 비용과 시간
- 실행 시간: 2026-10-21T20:01Z ~ 10-22T18:56Z(약 23시간)
- GPU-h(job의 GPU 부분 벽시계, 공유로 느려진 시간 포함)
  - Wall 4.9(2,200 ep)
  - PointMaze 12.4(1,700 ep, B와 공유)
  - PushT 18.3(1,700 ep)
  - 합계 약 35.6. E0의 단독 추정은 약 32였다
- v5.2 GPU 계산(GPU 1, 공유): drift_ref PushT 3 k·Wall 4 k·PointMaze 3 k와 mf_same을 합쳐 약 1.2 GPU-h

## 14. 파일
- 보고: 이 파일
- 결과: `results/E1/`
  - 요약: `e1_dinowm_summary.json`·`.txt`
  - 칸별 CSV: `e1_dinowm_cells.csv`
  - 표: `e1_dinowm_tables.md`, `e1_dinowm_v52_tables.md`
  - v5.2: `e1_dinowm_v52.json`
  - 정규화값: `it0_cost_std.json`
  - (b3)·같은 draw mf: `drift_ref/`·`mf_same/`
  - job JSON
- 서버
  - 모집단 배열: `~/pmx/A/dinowm_main/pops/E1_dinowm_wp/`, `.../E1_dinowm_pusht/`
  - 로그: `~/pmx/A/logs/E1_dinowm*.log`, `v52_drift_ref.log`
- 다시 만들기: `bash code/dinowm/e1_finalize.sh`(서버 CPU 한 번 + Mac)

## 부록 A. 자료 점검과 실행 요약

{{T:자료 점검}}

{{T:표 0. 실행 요약}}
