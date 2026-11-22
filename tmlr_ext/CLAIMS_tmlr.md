# ② TMLR 주장 (v2, 사용자 승인)

- 판 기록
  - v1: 2026-11-22T01:30Z 작성, 01:35Z 사용자 승인(§5)
  - v2: 02:15Z. 독립 판단([JUDGE_write_vs_more_2026-11-22.md](JUDGE_write_vs_more_2026-11-22.md) §3)이 지적한 문구 7건을 고쳤다. 사용자 승인은 02:07Z다(§5.1)
  - v2.1: 02:40Z. 재분석 R1~R5(`coordinator/reanalysis/`)의 결과로 수치 문구를 정밀화했다(§5.2). 판정과 주장은 바뀌지 않았다
  - v2.3: 11:23Z. 문장·문체 검토(Opus 5.5 max)가 제안한 한 줄 주장의 문구 다듬기 4건을 넣었다. 뜻은 그대로이고 사용자 승인이 있었다(§5.5)
  - v2.2: 10:40Z. 정책 기준선 X6 결과에 맞춰 한 줄 주장의 비트 문장에 단서를 달았다. 사용자 승인이 있었다(§5.4)
- 판정 전체는 [RESULTS_tmlr.md](RESULTS_tmlr.md)에 있다.
- 주장 변경은 사용자 승인으로만 한다(DESIGN_tmlr §8.6). 이 문구로 TMLR 원고를 쓰고, 원고를 고친 뒤 arXiv v2를 올린다.
- 바탕
  - arXiv v1(2611.11780, 2026-11-09)
  - RSS 초안 v2의 주장 C1~C6([../rss_ext/CLAIMS_rss.md](../rss_ext/CLAIMS_rss.md), 11-01 승인)

## 0. 무엇이 바뀌나

| | arXiv v1·RSS 초안 | TMLR |
|---|---|---|
| 기전 | "결과를 정하는 것은 오차가 반복 사이에 지속되고 후보가 공유하는지다"(일반 문장) | 대조 실험으로 **두 성분**이 갈린다.<br>(1) 고정이든 새로 뽑든 무편향 무작위 반올림이면 없어지는 체계 오차. DINO-WM의 주된 몫이다.<br>(2) 반복 사이에 지속되어 계획기가 이용하는 오차장. TD-MPC2 5M·317M의 주된 몫이다.<br>고정 무편향 오차와 새 잡음이 두 계열에서 반대로 움직인다(이중 해리) |
| 무엇이 성분을 정하나 | 계열 차이는 설명만 함(표 크기 가설) | 계획기의 갱신 규칙을 서로 바꿔도 양상이 모델 계열을 따랐다(E16). 원인은 가르지 않았다 |
| 처방의 이유 | 공유하고 반복마다 새로 뽑는 dither가 지속을 없앤다 | 반복마다 새로 뽑는 무편향 dither가 두 성분을 함께 없앤다. det 위에 새 잡음만 더하면(det+noise) MW는 8/8칸 고치지만 DMC는 1/4, Wall은 0/1이다 |
| 처방의 범위 | 같은 비트에서 손상을 없앤다 | 더해서 **비트를 번다**.<br>- MW 7/8과제(sh-iter와 det+shift 중 나은 쪽 기준, det+shift 단독도 같음). sh-iter 단독은 PT 6/8, PC 5/8<br>- PushT 1~2 bit(PT 2), Wall 0~2 bit(경계값), PointMaze 0<br>- DMC는 대부분 0 |
| DMC | 기전으로 설명되는 경계 | **반복 수의 경계**다. 반복 12에서 3/4과제가 고쳐지고, 반복이 늘수록 단조로 나아진다(It1, It2). walker는 끝까지 실패한다 |
| 진단 | 잠재 1-step 오차로는 안 보인다 | 같은 비트 안의 손상 순위는 ρ_p × σ_e/σ_J와 맞는다(Dg1: MW 0.97, DMC 0.39, DINO-WM 불성립). 이 값은 기전과 맞는 요약 통계로 쓰고, 배치 전 예측 도구로는 쓰지 않는다 |
| 크기 | 317M에서 같은 실패와 처방(E10) | 새 seed 복제(P1 8/8)와 기전 복제(P2 7/8) |
| 비용 | 공유 dither 오버헤드 −0.2~+2.1%(E8) | 정수 kernel에서 sh-iter는 6조건 모두 ≤ 2.0%다. det+shift는 N 512에서 5~7%라 공동 처방이 아니다(K1 불성립) |
| BF16 | 폐루프로 재지 않음(한계) | 미리 바꾼 BF16은 9/12과제에서 손상이 없고, 2/12과제에 작은 손상이 있다(K3 불성립) |

## 1. 제목과 한 줄 주장

- **제목은 그대로 둔다**: *Planners Exploit Deterministic Error: Dithered Low-Bit World Models*.
  - 설계 §1.5의 쓰임 표에서 B1·B2가 모두 성립한 경우의 처리다("기전을 두 성분으로 쓴다. 제목은 유지한다").
  - 새로움은 PLAN_tmlr §4.9의 ③·④(결정적 반올림 오차의 이용, 반복 사이 지속)의 범위로 쓴다. 일반 명제 ①로 읽히지 않게 한다.
- 한 줄 주장(영문 초안, v2):
  > At a fixed bit width, deterministic rounding error damages sampling-based planning through two error components that controlled variants separate: a systematic error that any unbiased randomized rounding removes, whether fixed or redrawn, and an input-determined error field that persists across planner iterations and that the planner exploits. The persistent part dominates in TD-MPC2 at 5M and 317M parameters but not in DINO-WM, and this pattern followed the model family when we swapped the planners' update rules. Dither shared by all candidates and redrawn at every planner iteration repairs most damaged settings and lowers the smallest usable bit width by at least one bit in most Meta-World tasks, even on per-channel grids. On Meta-World the policy alone, without planning, nearly matches the unquantized planner, so there the gain means that the low-bit planner does not fall below the policy; on DMC the low-bit dithered planners still beat the policy. Its simplest form adds at most about 2% rollout latency in an integer kernel.

## 2. 주장 목록

**C1 현상과 복제**
- 주장: 같은 비트에서 결정적 반올림(det)은 반복하는 계획기 안에서 큰 계획 손상을 만든다. 새 seed와 새 평가 분할에서도 다시 나온다.
- 근거
  - E11(새 seed 100000~100029): TD-MPC2 5M 손상 12/12칸. MW D_det +13.3~+45.6점, DMC +40.0~+83.8%
  - E14: 317M 8/8과제(+12.2~+97.8점)
  - DINO-WM eval8: Wall 3-bit +42점 [31, 53]. E1 탐색 칸(+50점)의 복제다
  - [사후] DINO-WM eval8 PushT 3-bit(D4): +35점 [24, 45], 덩어리 CI [16, 58]. E1 같은 칸(+36점 [27, 45])의 복제다
- 한정
  - DINO-WM PushT 4-bit는 새 분할에서 det 손상이 재현되지 않았다(+4점 [−7, 15], E1 +24점 [14, 34]).
    - eval8 PushT 쌍은 궤적 10개에서 나와 덩어리 CI가 [−19, 29]로 넓다(D1). 그래서 미재현 자체도 약한 증거로 쓴다.
    - 원인은 분할 차이다(X17, D5, [사후]). E1의 평가 쌍(eval6)을 RTX 4090에서 다시 돌리니 E1과 같은 손상이 나왔다. none 83%(E1 83%), det 62%(E1 59%), D_det +21점 [12, 30](E1 +24점 [14, 34]), sh-iter red 1.00이다. 미리 적은 판정 규칙 (i)에 해당한다. GPU·수치 차이로는 설명되지 않는다.
    - [사후] E1과 eval8을 합치면(R4, 역분산 가중, 짝이 아님) PushT는 두 분할이 이질적이다(Q p 0.008, I² 86%; eval8에 덩어리 분산을 쓰면 p 0.125). 그래서 합산하지 않고 나란히 싣는다. PointMaze 3-bit의 합산 +10점 [4, 16]과 Wall 3-bit의 합산 +46점 [38, 54]는 기술로 덧붙일 수 있다.
  - PointMaze 3-bit는 점 추정이 비슷하지만(+9점, E1 +11점) 3칸 Holm을 넘지 못했다.
  - 원고에는 E1과 eval8을 나란히 싣는다. 사전 지정 칸 중 eval8 손상 칸은 Wall 하나였다. [사후] PushT 3-bit 칸(D4)이 둘째 손상 칸이다.

**C2 두 성분과 이중 해리 (E11, E14)**
- 주장: det 오차에는 계획을 해치는 성분이 둘 있다.
  - (1) 체계 오차: 고정이든 새로 뽑든 무편향 무작위 반올림이면 없어진다.
  - (2) 오차장: 반복 사이에 지속되어 계획기가 이용한다.
  - TD-MPC2에서는 (2)가 지배하고, DINO-WM에서는 (2)가 지배하지 않는다.
- 근거
  - **TD-MPC2(지속 지배, B1 성립)**
    - 지속의 비용 Δ_pers = D(det+shift-fixed) − D(det+shift)는 손상 12칸 모두 CI 하한 > 0이다(MW +10.0~+40.0점, DMC +14.7~+46.7%).
    - 원소별 오차 분포를 같게 두고 반복 사이에 고정한 무작위 오차(sh-ep, det+shift-fixed)는 MW를 1/8, 0/8칸만 고친다. 해상도 설명은 기각된다.
    - 317M(P2 성립): 지속의 비용이 7/8과제다(D(sh-ep) − D(sh-iter) +0.24~+0.82).
    - 칸 분류(MW 지속 5·편향 3, DMC 지속 3·편향 1)는 문턱에 걸린 칸이 있다(pick-place det+BC red 0.457 [0.125, 0.686]). 본문에서는 분류 개수보다 Δ_pers와 "고정 대 새로 뽑음" 대비를 근거로 쓴다.
  - **DINO-WM(지속 지배가 아님, B2 성립)**
    - 고정 무편향 오차가 두 손상 칸 모두를 고친다. Wall은 sh-plan red 1.10, det+shift-fixed 0.95다. [사후] PushT 3-bit은 0.91, 0.94다. E1 PushT의 sh-plan도 k3 0.89, k4 1.04다.
    - 평균 편향 보정(det+BC)의 효과는 과제마다 다르다.
      - Wall: 고친다(0.71).
      - PushT: det보다 나쁘다. [사후] 3-bit E11 칸 red −0.23(det+BC 성공 30%, det 38%), E12 3-bit +46점 대 +32점(각 50 episode), 4-bit +16점 대 +4점(E11).
    - det의 평균 편향을 되돌려 넣은 sh-iter+b와 det+noise는 두 손상 칸 모두 못 고친다. Wall은 0.43, 0.07이고, [사후] PushT 3-bit은 −0.29, −0.17이다(PushT에서 sh-iter+b는 sh-iter보다 +47점 [35, 59] 나쁘다).
    - 두 칸의 분류는 편향(Wall)과 해상도(PushT 3-bit: det+BC는 못 고치고, det+shift-fixed는 고치고, Δ_pers +4 [−3, 11])로 갈린다. 합친 읽기는 [사후] 혼합이고, 어느 칸도 지속 지배가 아니다.
  - **이중 해리**: 고정 무편향 오차와 새 잡음이 두 계열에서 반대로 움직인다. det+BC는 이 근거에서 뺀다.
    - 새 잡음(det+noise)은 MW 8/8칸을 고치지만 DINO-WM 손상 칸은 0/2다(Wall 0.07, [사후] PushT 3-bit −0.17).
    - 고정 무편향 오차는 DINO-WM 손상 칸 2/2를 고치지만 MW는 0~1/8칸만 고친다.
    - MW에서 "편향"으로 분류된 3칸도 이 대비에서는 지속 쪽이다. det+shift-fixed red는 0.07·0.31·0.22, det+noise red는 0.80·0.92·0.86이다.
- 한정
  - DINO-WM의 eval8 손상 칸은 사전 지정 칸 하나(Wall)와 [사후] 칸 하나(PushT 3-bit)다. Wall은 none 49%에 det k4가 none보다 22점 높아 기준선도 불안정하다.
  - TD-MPC2에서도 평균 편향은 부분 성분이다. det+BC가 고친 칸은 MW 3/8, DMC 1/4, 317M 4/8이다.
  - 편향 벡터 b는 held 자료로 구했고 단계 감쇠 반복으로 수렴시켰다(D2).
  - [사후] PushT에서는 D2 규칙으로 구한 b가 해롭다. b를 빼도(det+BC) 더해도(sh-iter+b) det만큼 나쁘고, b를 넣으면 계획 시점 오차가 커진다(1/5-step: det 0.31/0.74 → det+BC 0.42/0.92). PushT PT k3의 전파 5-step 오차도 b로 줄지 않았다(0.93 → 0.95). 그래서 PushT에서 det+BC가 못 고친 것을 "평균 편향이 없다"의 증거로 읽지 않는다. held 자료로 구한 b가 계획 상태에 맞지 않았을 수 있다(DW_dinowm/REPORT.md §9.2).

**C3 계획기 갱신 규칙을 바꿔도 지속의 비용은 모델 계열을 따랐다 (E16)**
- 주장: 계획기의 갱신 규칙을 서로 바꿔도, 지속의 비용이 있는지 없는지는 모델 계열을 따랐다. 원인은 가르지 않았다.
- 근거
  - TD-MPC2를 CEM식으로 바꿔도(elite 균등 가중, 정책 궤적 0) D(sh-ep) − D(sh-iter)가 3/3과제에서 남는다. CEM식은 push +47.8, pick-place +42.2점, cheetah +18.7%이고, 기본 MPPI는 +48.9, +28.9, +36.6이다.
  - DINO-WM을 MPPI식으로 바꿔도 D(sh-plan) − D(sh-iter)는 0과 구별되지 않는다(n 100).
    - CEM: PointMaze +0 [−7, 7], Wall −4 [−12, 4]
    - MPPI식: PointMaze +4 [−1, 10], Wall −7 [−18, 4]
    - [사후] PushT 3-bit(CEM): +5 [−3, 13]
    - CI 상한은 +4~+10점이다.
    - 판정 밖 칸 PushT 4-bit(det 손상이 실재하지 않은 칸)는 +6 [−1, 13]이고, 덩어리 CI는 [+1.7, +10.4]로 0을 넘는다. DINO-WM 전체로 쓸 때는 이 칸을 밝힌다(R5).
    - 계열 차이는 "TD-MPC2는 CI 하한 > 0, DINO-WM은 CI에 0 포함"으로 쓴다. 칸별 크기는 겹치므로 "DINO-WM 쪽이 더 작다"고 칸마다 쓰지 않는다(DINO-WM 상한보다 TD-MPC2 하한이 큰 칸은 MW 5M 5~7/8, 317M 6~7/8, R5).
  - 바꾼 계획기는 모두 자격을 얻었다. none 성공률(return)이 기본의 60% 이상이다.
- 한정
  - 기술 결과다(판정 문턱 없음). PushT는 빠졌다.
  - 두 계열은 모델 구조, 폐루프·개루프, horizon, 목적 함수가 함께 다르다. 그래서 "모델이 정한다"는 인과로 쓰지 않는다. E3의 표 크기 결과(H4a)는 가설로만 든다.
  - Wall의 Δ_pers +17점 [7, 27]은 det+shift가 none보다 15점 높아서 생긴 값이다. 본문에 밝힌다.
  - TD-MPC2 CEM식 cheetah에서는 sh-iter도 거의 고치지 못한다(D +48.9%, det +57.1%, red 0.14). 본문에 밝힌다.

**C4 처방: 공유하고 반복마다 새로 뽑는 dither가 손상 대부분을 없애고 비트를 번다 (B4, P1, E12, E15)**
- 주장: 모든 후보가 공유하고 반복마다 새로 뽑는 dither는 학습 없이 손상 대부분을 없앤다. 범위를 per-channel로 잘 잡은 양자화기 위에서도 MW 대부분의 과제에서 쓸 수 있는 최소 비트를 1 bit 이상 낮춘다.
- 근거
  - B4 성립(E1 H3a의 독립 seed 복제): MW 7/8칸에서 sh-iter red ≥ 0.5이고, 해로운 칸이 없다.
  - P1 성립: 317M 8/8(red 0.69~1.31). DINO-WM Wall: red 1.00.
  - Fr1·Fr2 성립
    - sh-iter와 det+shift 중 나은 쪽 기준으로 MW 7/8과제에서 PT·PC 모두 G ≥ 1이다.
    - sh-iter 단독은 PT 6/8, PC 5/8이다(조정자 재계산, `TD_tdmpc2/results/analysis/td_e12_cells.csv`). det+shift 단독은 7/8·7/8이다(R3).
    - 짝 bootstrap에서 Fr1·Fr2가 성립할 확률은 세 기준(나은 쪽, sh-iter 단독, det+shift 단독) 모두 0.994 이상이다(R1, 최솟값은 sh-iter 단독 Fr2의 0.9947). 문턱 τ를 0.05~0.20으로 바꿔도 나은 쪽 기준으로 Fr1 6~8/8, Fr2 5~7/8로 성립한다(R2).
    - 흔들리는 칸: pick-place(P(G≥1) PT 0.41, PC 0.66), window-open(PT 0.74, PC 0.87). Fr2의 pick-place는 D가 정확히 0.100인 경계 칸 하나에 걸려 있다. τ를 "D < 0.10"으로 바꾸면 Fr2가 6/8이 되지만 여전히 성립한다(R2).
    - MW 6과제에는 1 bit에서 손상 10점 이하인 구성이 있다. E12의 네 반올림(det, det+BC, sh-iter, det+shift) 가운데서는 모두 dither 구성이다. [사후] X5의 det+noise도 drawer-open per-tensor에서 k\* = 1이다(2 bit의 D가 정확히 10.0점인 경계값에 기댐; `TD_tdmpc2/results/analysis_x5/x5_summary.json`).
  - DINO-WM의 G
    - PushT: 1 bit 이상은 세 범위 모두 튼튼하다(P(G≥1) PT 0.999, PC 0.999, RC 0.977). 2 bit는 PT만 튼튼하다(P(G≥2) PT 0.985, PC 0.593, RC 0.805; R1). 그래서 "PT 2 bit, PC·RC 1~2 bit"로 쓴다
    - Wall 0~2. 경계값이다: det+BC k3 +12점 [1, 23]이 문턱 10점을 2점 넘어 k\*(det+BC) = 4가 됐다.
    - PointMaze 0
  - rollout으로 고른 범위(RC, WM-Quant의 핵심 생각)는 k\*를 바꾸지 않는다. PushT에서는 그 위에서도 dither가 1~2 bit를 번다.
  - **새 잡음과 dither의 비교**(B3, 사전 예측과 반대)
    - det+noise가 고친 칸은 MW 8/8(red 0.52~0.92), DMC 1/4(편향으로 분류된 finger만), Wall 0/1(0.07)이다.
    - det+noise의 지속 몫 ρ_p(마지막 반복 집합, E11 12칸)는 0.05~0.23이다. 고정 변형(1.00)보다 훨씬 낮지만, 같은 칸의 새로 뽑는 dither(sh-iter 0.00~0.11, det+shift 0.00~0.10)보다는 높다. 그 차이가 고치는 칸 수의 차이와 관련 있는지는 가르지 않았다.
    - 같은 칸에서 dither는 더 많이 고친다(DMC det+shift 3/4, sh-iter 2/4; Wall 1.00, 1.36). 왜 그런지는 가르지 않았다.
    - 비트 축(X5, D6, [사후] 기술): 비트를 내리면 det+noise는 dither만큼 버티지 못한다.
      - det 계열 대비 번 비트 G_noise ≥ 1은 PT 4/8, PC 1/8이다. dither는 7/8·7/8이다.
      - k\*(det+noise) ≤ k\*(sh-iter)인 과제는 PT 3/8, PC 2/8이다.
      - det+noise의 곡선은 단조가 아닌 곳이 많다(PT door-open k3 +22.2, button-press k2 +41.1, sweep-into k2 +42.2).
      - E11 1차 k 칸은 같은 기종의 다른 pod에서 비트 단위로 재현됐다(none·det+noise 각 720 episode).
      - 출처: `TD_tdmpc2/results/analysis_x5/`, 조정자 재계산 `coordinator/verify_x5.txt`.
- 한정
  - DMC는 반복 수의 경계다. 기본 반복 6에서 sh-iter가 고친 칸은 2/4다.
    - 반복 12에서는 3/4과제 red ≥ 0.5다(It1).
    - 3/4과제에서 red가 반복과 함께 단조로 커진다(It2).
    - walker는 반복 24에서도 red 0.21이다. 기본 반복에서는 sh-iter가 해롭다.
  - DMC의 G는 대부분 0이다(PT 0·0·0·1, PC 1·0·0·1).
  - 반복을 늘리면 결정당 계획 시간이 늘어난다(cheetah 반복 6 → 24에 30 → 124 ms).
  - "지연 2%"(C7)는 sh-iter의 값이다. "7/8"과 한 방법의 수치처럼 묶어 쓰지 않는다.
  - [사후] 정책 기준선(X6, D7): 계획 없이 FP32 정책 π만 쓰면 MW는 none에 가깝다. 하락은 drawer-open·button-press 0, door-open·peg-insert +1.1점, sweep-into −1.1점, window-open +6.7점, push +7.8점, pick-place +23.3점이다.
    - 그래서 MW의 낮은 비트 결과("1 bit에서도 쓸 수 있다")는 계획기가 정책 수준을 망치지 않는다는 뜻에 가깝다. 가장 낮은 dither 구성 대 정책은 낫다 1·같다 7·못하다 0이다. 다만 이 개수는 같은 k*에 묶인 구성을 고르는 규칙에 따라 달라진다.
    - DMC에서는 정책만 쓰면 return이 크게 준다(cheetah +53.5%, walker +56.3%, quadruped +13.7%, finger +6.3%). 낮은 비트 dither 계획기(PC k = 3~4)는 3/4과제에서 정책보다 낫고 finger에서는 같다.
    - 출처: `TD_tdmpc2/REPORT.md` §12, `TD_tdmpc2/results/analysis_x6/`.

**C5 진단 (E13): 기전과 맞는 요약 통계**
- 주장: TD-MPC2에서는 같은 비트 안의 반올림 방식 사이 손상 차이가 잠재 1-step 예측 오차로 보이지 않는다. 계획기가 보는 후보 return 오차의 지속 몫과 상대 크기(ρ_p × σ_e/σ_J)와는 맞는다.
- 근거
  - Dg1 사전 지정 비교: 칸 안 Spearman의 평균 차가 MW 0.971(8칸 모두 양수), DMC 0.387이다.
  - 공통 held 상태에서 det의 1-step 오차가 sh-iter 이하인 칸이 TD-MPC2 E1 36칸 전체 중 27칸이다(손상 여부와 무관한 분모). arXiv v1의 29/36은 두 계열의 손상 칸 기준이라 분모가 다르다. 본문에 둘을 구별해 쓴다.
- 한정
  - ρ_p는 고정 변형에서 1.00, 새로 뽑는 변형에서 −0.00~0.24라 조작한 변수(지속)를 거의 그대로 담는다. 그래서 "배치 전에 재면 손상을 예측한다"는 도구로 쓰지 않고, 기전과 맞는 요약 통계로 쓴다.
  - DINO-WM은 불성립이다(0.262; Wall 한 칸에서 0.905 − 0.643).
  - DMC의 walker(−0.24)·quadruped(−0.12)에서는 1-step 오차보다 못 맞혔다.

**C6 크기 (E14)**
- 주장: 317M 다중 과제 모델에서도 같은 실패, 처방, 지속의 비용이 새 seed로 다시 나온다.
- 근거: P1 성립(8/8), P2 성립(7/8).
- 한정
  - 체크포인트는 하나이고 과제는 MW뿐이다.
  - GPU가 E10과 같은 RTX 4090이라 복제는 seed 쪽만이다.
  - sh-ep는 sweep-into에서 det보다 해롭다(r = 2.61).

**C7 비용과 정수 경로 (E17)**
- 주장: 가장 단순한 처방(sh-iter)은 정수 kernel에서 rollout 지연을 2% 남짓 더할 뿐이고, 실제 정수 경로에서도 fake-quant와 같은 결과를 낸다.
- 근거
  - sh-iter 지연 오버헤드는 6조건 모두 ≤ 2.01%다(5M·317M × 후보 512·2048·8192).
  - K2 성립: 정수 det − det+shift가 4/4과제에서 fake-quant와 같은 부호이고 CI가 겹친다.
  - 접은 per-channel 정수 구성이 fake-quant per-channel과 13/16칸에서 CI 안에서 같다.
- 한정
  - TD-MPC2와 A40 한 종류에서만 쟀다.
  - det+shift의 오버헤드는 N 512에서 5.3%(5M)·7.2%(317M)다(K1 불성립). 그래서 det+shift는 기전 대조이자 처방의 한 형태로만 쓰고, 공동 처방으로 올리지 않는다.
  - 미리 바꾼 BF16에는 작은 손상이 있다(door-open +5.6점, cheetah +2.1%; K3 불성립).
  - 속도 주장은 RSS C4 수정판(11-01) 그대로다. A40에서는 미리 바꾼 BF16이 더 빠르다.

## 3. 쓰지 않을 주장 (CLAIMS_rss §3에 더함)

- "결과를 정하는 것은 지속이다"를 두 계열에 일반 문장으로 쓰는 것. DINO-WM은 지속 지배가 아니고(B2), 지속의 비용이 0과 구별되지 않는다(E16).
- "DINO-WM은 평균 편향이 지배한다". PushT에서는 det+BC가 det보다 나쁘다.
- "어느 성분이 지배하는지는 모델이 정한다"(인과). E16은 두 계열을 계획기만 바꿔 비교했을 뿐이다.
- "어떤 잡음도 det 손상을 줄이지 못한다", "무작위화만 있으면 된다", "지속 몫이 낮으면 고쳐진다". det+noise는 MW 8/8칸을 고치지만 DMC 1/4, Wall 0/1이다.
- "해상도 손실이 det 손상의 원인이다"(TD-MPC2). MW에서 고정 무작위 문턱이 0~1/8칸만 고쳤다.
- "bias correction이 처방이다". 고친 칸은 MW 3/8, DMC 1/4, 317M 4/8이고, PushT에서는 det보다 나쁘다.
- "sh-iter 하나로 MW 7/8과제에서 비트를 번다". 단독으로는 PT 6/8, PC 5/8이다.
- "Wall에서도 dither가 2 bit를 번다". 경계값이라 0~2로 쓴다.
- "DINO-WM PushT 4-bit의 det 손상이 새 분할에서도 재현된다". eval8에서는 +4점이었다.
- "ρ_p × σ_e/σ_J를 배치 전에 재면 손상을 예측할 수 있다". 이 값은 조작 변수를 거의 그대로 담고, DINO-WM에서는 불성립이다.
- "det+shift를 지연 비용 없이 쓸 수 있다". K1이 불성립이다.
- "미리 바꾼 BF16은 폐루프 손상이 없다". K3이 불성립이다.
- "기본 반복 수에서 DMC도 처방이 통한다". E11 DMC의 sh-iter는 2/4이고, walker는 반복 24에서도 실패한다.

## 4. 결정 항목 (v1에서 정함)

| # | 정할 것 | 추천(v1) |
|---|---|---|
| T1 | C1~C7 주장 묶음 | 승인 |
| T2 | 제목 유지 | 유지(설계 §1.5의 처리) |
| T3 | det+shift의 위치 | 기전 대조이자 처방의 한 형태로 쓴다. 공동 처방으로 올리지 않는다(K1 불성립) |
| T4 | DINO-WM 보강: PushT 3-bit E11 칸을 새 pod에서 [사후]로 더 돌릴지 | 돌린다(약 4.5 h, 약 $4.2) |
| T5 | arXiv v2 | 원고를 고친 뒤 12월 초에 올린다(PLAN §4.9) |
| T6 | E18(LeWM, 제3 계열) | T4와 함께 정한다(PLAN §4.7의 조건) |

## 5. 승인 기록 (v1, 2026-11-22T01:35Z, 한국 시간 10:35)

- 사용자 답(원문): "너의 추천대로 해주고". §4의 추천을 그대로 승인했다.
  - T1: C1~C7 승인
  - T2: 제목 유지
  - T3: det+shift는 기전 대조이자 처방의 한 형태로 쓴다. 공동 처방으로 올리지 않는다
  - T4: PushT 3-bit E11 칸을 [사후] 보강으로 돌린다(편차 D4, DESIGN_tmlr §11)
    - 결과가 나오면 C1·C2의 DINO-WM 문장(손상 칸 수, 근거)을 그 결과에 맞춰 고친다
    - 사전 지정 판정(B2·B3·Dg1)은 바꾸지 않는다
  - T5: arXiv v2는 원고를 고친 뒤 12월 초에 올린다
  - T6: 추천은 "T4와 함께 정한다"였다. 조정자 판단으로 지금은 E18을 하지 않는다. PLAN §4.7대로 두 계열로 한정해 쓰면 TMLR 기준에 지장이 없다. PushT 3-bit 결과를 본 뒤 필요하면 다시 제안한다

### 5.1 v2 문구 수정 승인 기록 (2026-11-22T02:07Z, 한국 시간 11:07)

- 배경: 사용자 요청으로 판단 에이전트(Claude Fable 5.2, 추론 노력 max)가 "추가 실험을 더 하고 쓸지, 지금 진행 중인 것이 끝나면 바로 쓸지"를 판단했다([JUDGE_write_vs_more_2026-11-22.md](JUDGE_write_vs_more_2026-11-22.md)).
  - 판정: (c) 지금 집필을 시작하고, 작은 실험(X5, 선택 X17)을 병렬로 돈다.
  - 함께 문구 7건과 재분석 5건(R1~R5)을 제안했다.
  - 조정자가 지적의 수치를 원자료로 확인했다(LOG 02:00Z).
- 사용자 답(원문): "너의 추천대로 해줘". 조정자 추천은 X17을 포함해 판정 (c)대로 진행하는 것이었다.
- 고친 것
  1. C2의 DINO-WM 문장: "편향 지배" → "지속 지배가 아니다. 고정 무편향 오차가 고치고, det+BC는 Wall만 고친다(PushT에서는 det보다 나쁨)"
  2. 이중 해리의 근거: det+noise와 고정 무편향 변형으로만 세우고 det+BC는 뺐다. 분류 개수보다 Δ_pers와 고정 대 새로 뽑음 대비를 근거로 쓴다
  3. C3: "모델이 정한다"(인과) → "갱신 규칙을 바꿔도 양상이 계열을 따랐다. 원인은 가르지 않았다". DINO-WM 쪽은 "0과 구별되지 않음(CI 상한 +4~+10점)"으로 쓴다. Wall Δ_pers와 CEM식 cheetah 값을 밝힌다
  4. C4의 "7/8": sh-iter와 det+shift 중 나은 쪽 기준임을 밝힌다(sh-iter 단독 PT 6/8, PC 5/8). "지연 2%"는 sh-iter의 값이므로 묶지 않는다
  5. Wall G = 2 → 0~2(경계값)
  6. C4의 det+noise 설명: 인과 설명("편향이 지배하는 곳에서는 통하지 않는다")을 지우고 수치 비교로 바꿨다. v1의 설명은 DMC 결과(지속 칸 셋을 det+noise가 못 고치고 편향 칸 finger만 고침)와 어긋났다
  7. C5: "진단"을 "기전과 맞는 요약 통계"로 바꿨다(ρ_p가 조작 변수를 담음). 27/36과 29/36의 분모 차이를 밝힌다
- 한 줄 주장과 §0·§3을 이 수정에 맞췄다. B1·B2·B3·B4·Fr1·Fr2·Dg1 등 사전 지정 판정은 바뀌지 않았다.
- 남은 덧붙임: D4(PushT 3-bit)·X17·X5의 결과가 나오면 이 절 뒤에 반영 기록을 더한다.

### 5.2 v2.1 수치 문구 정밀화 (2026-11-22T02:40Z, 재분석 R1~R5 반영)

- 재분석 R1~R5는 판단 에이전트가 제안했고 사용자가 승인한 일이다(§5.1). 결과는 `coordinator/reanalysis/summary.txt`에 있고, 발표된 판정을 뒤집는 것은 없었다.
- 그 결과에 맞춰 수치 문구만 정밀화했다. 주장의 범위는 넓히지 않았다.
  1. C4: PushT의 G를 "2(PT·PC·RC 모두)"에서 "PT 2 bit, PC·RC 1~2 bit"로 바꿨다(P(G≥2) PT 0.985, PC 0.593, RC 0.805)
  2. C4: Fr1·Fr2의 bootstrap 성립 확률(≥ 0.995)과 τ 민감도를 덧붙였다. det+shift 단독 7/8·7/8과, Fr2 pick-place의 경계 칸(D = 0.100)도 덧붙였다
  3. C3: 판정 밖 칸 PushT 4-bit의 D(sh-plan) − D(sh-iter) +6 [−1, 13](덩어리 CI는 0 초과)을 밝혔다. 계열 차이는 "TD 하한 > 0, DINO-WM CI에 0 포함"으로 쓰고, 칸별 크기는 겹친다고 적었다
  4. C1: [사후] E1과 eval8의 합산 결과를 덧붙였다. PushT는 이질적이라 합산하지 않고 나란히 싣는다
  5. [정정, 02:55Z] C4의 ρ_p 비교를 같은 칸 기준으로 바로잡았다. v2는 det+noise(E11 12칸)를 dither 전체 72행의 범위와 견주어 "비슷하다"고 썼다. 같은 E11 칸에서는 dither가 0.00~0.11로 더 낮다(07:45Z에 반올림을 다시 고침: 최댓값 0.1149, 0.1047). 원고 작업자가 찾았고, 조정자가 원자료로 확인했다
  6. [정정, 02:55Z] PointMaze 합산 구간을 [4, 17]에서 [4, 16]으로, Fr 성립 확률을 "0.995 이상"에서 "0.994 이상"으로 고쳤다. 반올림을 두 번 해서 생긴 차이다
- Wall "0~2 bit", PointMaze 0, DMC "대부분 0"은 R1·R2와 맞아 그대로 둔다.

### 5.3 [사후] 결과 반영 기록

- 2026-11-22T03:50Z X5(D6): C4의 "새 잡음과 dither의 비교"에 비트 축 결과를 넣었다. det+noise는 G_noise ≥ 1이 PT 4/8·PC 1/8이고, dither는 7/8·7/8이다. 주장의 범위는 그대로다(X5는 C4를 받치려고 돌렸다).
- 2026-11-22T06:05Z D4([사후] PushT 3-bit E11 칸, dw.6 eval8 100 episode): C1·C2·C3의 DINO-WM 문장을 그 결과에 맞췄다. §5 T4에서 미리 승인된 반영이다.
  - 손상이 실재한다: D_det +35점 [24, 45]. E1 +36점의 복제다.
  - 분류는 해상도다: det+BC red −0.23, det+shift-fixed 0.94, Δ_pers +4 [−3, 11].
  - 고친 변형은 sh-iter 1.06, det+shift 1.06, sh-plan 0.91이다. det+noise(−0.17)와 sh-iter+b(−0.29)는 못 고쳤다.
  - DINO-WM 손상 칸은 둘이 되었고, 둘 다 지속 지배가 아니다. 이중 해리도 두 칸으로 선다(det+noise 0/2, 고정 무편향 2/2).
  - 사전 지정 판정 B2·B3·Dg1은 바꾸지 않았다.
  - 조정자 재계산: `coordinator/verify_d4.txt`(일치).
- 2026-11-22T07:20Z X17(D5, [사후] PushT 4-bit E1 분할 재실행, dw.6 RTX 4090): C1의 PushT 4-bit 미재현 문장에 결과를 넣었다.
  - E1 쌍에서는 4090에서도 손상이 그대로 나왔다(D_det +21점 [12, 30], E1 +24점 [14, 34]).
  - D5 판정은 (i)이다: 미재현은 eval8 분할의 차이로 본다.
  - 조정자 재계산: `coordinator/verify_x17.txt`.
- 2026-11-22T07:25Z dw 작업자 최종 보고(D4·X17, dw.6 07:15:10Z 삭제)를 반영했다.
  - C2 한정에 PushT의 b 문제를 덧붙였다. D2 규칙의 b는 PushT에서 해롭다.
  - 합친 DINO-WM 읽기([사후] mixed)는 사전 지정 B2(Wall 한 칸, 성립)와 따로 적는다. 주장 문구는 v2.1대로 "지속 지배가 아니다"다.
- 2026-11-22T08:45Z 그림 작업자가 찾은 충돌을 반영했다. C4의 "1 bit 구성은 모두 dither"를 "E12의 네 반올림 가운데서는"으로 한정했고, X5의 det+noise 예외(drawer-open PT, 경계값)를 밝혔다.
- 2026-11-22T10:02Z X6(D7, 정책 기준선): C4 한정에 결과를 넣었다. MW에서는 정책만으로도 none에 가까워 낮은 비트 결과의 의미를 좁혀 쓴다. DMC에서는 낮은 비트 dither 계획기가 정책보다 낫다(3/4). 조정자 재계산(td.8 원자료의 D_policy 12개)이 작업자 값과 같다.

### 5.4 v2.2 초록 단서 승인 기록 (2026-11-22T10:40Z)

- 배경: 정책 기준선(X6)을 보니 MW에서는 계획 없는 FP32 정책만으로도 계획기와 거의 같다. 그래서 "MW 대부분 과제에서 1 bit 이상 번다"는 초록 문장이 계획의 이득처럼 읽힐 수 있다(원고 작업자 제기).
- 사용자 답: "단서 추가 (Recommended)". 조정자가 예로 든 문구를 바탕으로 한다.
- 고친 것: 한 줄 주장(§1)과 원고 초록·결론의 비트 문장 뒤에 넣었다. "On Meta-World the unquantized policy alone nearly matches the planner, so there the gain means that low-bit planning stops hurting; on DMC the low-bit dithered planners still beat the policy."
- 같은 결정에서 B 원고의 LLM 각주는 그대로 두기로 했다(사용자 답 "B는 그대로").

### 5.5 v2.3 문구 다듬기 승인 기록 (2026-11-22T11:23Z)

- 배경: 문장·문체 독립 검토(`REVIEW_opus_prose_2026-11-22.md` §4)가 승인된 초록 문구에서 문법·정확성 문제 넷을 짚었다. 뜻은 바뀌지 않는다.
- 사용자 답: "4건 모두 반영 (Recommended)".
- 고친 것(한 줄 주장 §1과 원고 초록·결론)
  1. "in two ways that controlled variants separate: a systematic error … and an … error field" → "through two error components that controlled variants separate". ways와 오차 명사가 맞지 않았다.
  2. "usable bit width" → "smallest usable bit width". 본문 3.4의 정의 용어다.
  3. "nearly matches the planner" → "nearly matches the unquantized planner".
  4. "low-bit planning stops hurting" → "the low-bit planner does not fall below the policy". 구어를 피하고 비교 대상을 밝혔다.
- 같은 결정에서 부록 K의 "Design documents, logs and stage reports are in Korean" 문장은 PDF에서 빼고 보충자료 README에만 두기로 했다(사용자 답 "PDF에서 빼고 README에만"). 이중맹검 단서를 줄이기 위해서다.
- 원고 작업자가 같은 뜻에서 "unquantized"가 두 번 겹치지 않게 "the policy alone, without planning"으로 썼다. 한 줄 주장도 원고와 같게 맞췄다(정책은 처음부터 FP32다).
