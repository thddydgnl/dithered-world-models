# ② TMLR 주장 재구성안 (사용자 승인용)

- 작성: 2026-11-22T01:30Z(UTC, 한국 시간 10:30), 조정자. E11~E17이 모두 끝나고 판정 수치를 원자료에서 다시 계산한 뒤에 썼다. 판정 전체는 [RESULTS_tmlr.md](RESULTS_tmlr.md)에 있다.
- 이 문서는 제안으로 썼고, 2026-11-22T01:35Z에 사용자가 승인했다(§5). 주장 변경은 사용자 승인으로만 한다(DESIGN_tmlr §8.6). 승인된 문구로 TMLR 원고를 쓰고, 승인되는 대로 arXiv v2를 올린다(PLAN_tmlr §4.9).
- 바탕은 둘이다.
  - arXiv v1(2611.11780, 2026-11-09)
  - RSS 초안 v2의 주장 C1~C6([../rss_ext/CLAIMS_rss.md](../rss_ext/CLAIMS_rss.md), 11-01 승인)

## 0. 무엇이 바뀌나

| | arXiv v1·RSS 초안 | TMLR 제안 |
|---|---|---|
| 기전 | "결과를 정하는 것은 오차가 반복 사이에 지속되고 후보가 공유하는지다"(일반 문장) | **두 성분**이 있다. 반복 사이에 지속되는 오차장은 TD-MPC2(5M·317M)에서 지배한다. 평균 편향은 DINO-WM에서 지배하고 TD-MPC2에서는 부분 몫이다. 같은 변형이 두 계열에서 반대로 움직인다(이중 해리) |
| 무엇이 성분을 정하나 | 계열 차이는 설명만 함(표 크기 가설) | 계획기를 서로 바꿔도 양상이 그대로다. 정하는 것은 **모델**이다(E16) |
| 처방의 이유 | 공유하고 반복마다 새로 뽑는 dither가 지속을 없앤다 | dither는 **두 성분을 다 없앤다**. 반복마다 새 잡음만 더하는 것(det+noise)은 지속만 없애므로 MW에서만 통하고, 편향이 지배하는 Wall과 DMC에서는 통하지 않는다 |
| 처방의 범위 | 같은 비트에서 손상을 없앤다 | 더해서 **비트를 번다**. MW 7/8과제에서 per-tensor·per-channel 모두 1 bit 이상이고, DINO-WM PushT·Wall에서는 2 bit다 |
| DMC | 기전으로 설명되는 경계 | **반복 수의 경계**다. 반복 12에서 3/4과제가 고쳐지고, 반복이 늘수록 단조로 나아진다(It1, It2). walker는 끝까지 실패한다 |
| 진단 | 잠재 1-step 오차로는 안 보인다 | 더해서 **무엇을 재야 하는지**: TD-MPC2에서는 ρ_p × σ_e/σ_J가 같은 비트 안의 손상 순위를 맞힌다(Dg1: MW 0.97, DMC 0.39). DINO-WM에서는 불성립이다 |
| 크기 | 317M에서 같은 실패와 처방(E10) | 새 seed 복제(P1 8/8)와 기전 복제(P2 7/8) |
| 비용 | 공유 dither 오버헤드 −0.2~+2.1%(E8) | 정수 kernel에서 sh-iter는 6조건 모두 ≤ 2.0%다. det+shift는 N 512에서 5~7%라 공동 처방이 아니다(K1 불성립) |
| BF16 | 폐루프로 재지 않음(한계) | 미리 바꾼 BF16은 9/12과제에서 손상이 없고, 2/12과제에 작은 손상이 있다(K3 불성립) |

## 1. 제목과 한 줄 주장

- **제목은 그대로 둔다**: *Planners Exploit Deterministic Error: Dithered Low-Bit World Models*.
  - 설계 §1.5의 쓰임 표에서 "B1·B2 성립, DINO-WM 편향 지배"인 경우의 처리다("기전을 두 성분으로 쓴다. 제목은 유지한다").
  - 새로움은 PLAN_tmlr §4.9의 ③·④(결정적 반올림 오차의 이용, 반복 사이 지속)의 범위로 쓴다. 일반 명제 ①로 읽히지 않게 한다.
- 한 줄 주장(영문 초안):
  > At a fixed bit width, deterministic rounding error damages sampling-based planning through two components: a mean bias that accumulates along rollouts, and an input-determined error field that persists across planner iterations and that the planner exploits. Which component dominates is set by the world model, not by the planner. Dither shared by all candidates and redrawn at every planner iteration removes both, lowers the usable bit width by at least one bit even on per-channel grids, and adds about 2% latency in an integer kernel.

## 2. 주장 목록

**C1 현상과 복제**
- 주장: 같은 비트에서 결정적 반올림(det)은 반복하는 계획기 안에서 큰 계획 손상을 만든다. 새 seed와 새 평가 분할에서도 다시 나온다.
- 근거
  - E11(새 seed 100000~100029): TD-MPC2 5M 손상 12/12칸. MW D_det +13.3~+45.6점, DMC +40.0~+83.8%
  - E14: 317M 8/8과제(+12.2~+97.8점)
  - DINO-WM eval8: Wall 3-bit +42점 [31, 53]. E1 탐색 칸(+50점)의 복제다
- 한정
  - DINO-WM PushT 4-bit는 새 분할에서 det 손상이 재현되지 않았다(+4점 [−7, 15], E1 +24점 [14, 34]).
  - PointMaze 3-bit는 점 추정이 비슷하지만(+9점, E1 +11점) 3칸 Holm을 넘지 못했다.
  - 그래서 eval8에서 DINO-WM 손상 칸은 하나다.
  - eval8 PushT 쌍은 궤적 10개에서 나왔다(D1). 덩어리 bootstrap으로도 손상이 아니다.
  - 원고에는 E1과 eval8을 나란히 싣는다.

**C2 두 성분과 이중 해리 (E11, E14)**
- 주장: det 오차에는 계획을 해치는 성분이 둘 있다. 반복 사이에 지속되어 계획기가 이용하는 오차장과, rollout을 따라 쌓이는 평균 편향이다. TD-MPC2에서는 앞의 것이, DINO-WM에서는 뒤의 것이 지배한다.
- 근거
  - **TD-MPC2(지속 지배, B1 성립)**
    - MW 손상 8칸은 지속 지배 5칸, 편향 3칸, 해상도 0칸이다. DMC는 지속 3칸, 편향 1칸이다.
    - 지속의 비용 Δ_pers = D(det+shift-fixed) − D(det+shift)는 손상 12칸 모두 CI 하한 > 0이다(MW +10.0~+40.0점, DMC +14.7~+46.7%).
    - 원소별 오차 분포를 같게 두고 반복 사이에 고정한 무작위 오차(sh-ep, det+shift-fixed)는 MW를 1/8, 0/8칸만 고친다. 해상도 설명은 기각된다.
    - 317M(P2 성립): 지속의 비용이 7/8과제다(D(sh-ep) − D(sh-iter) +0.24~+0.82).
  - **DINO-WM(편향 지배, B2 성립)**
    - 손상 칸 Wall 3-bit에서 det+BC가 손상의 71%를 없앤다.
    - 고정 무편향 오차도 고친다(sh-plan red 1.10, det+shift-fixed 0.95).
    - det의 평균 편향을 되돌려 넣은 sh-iter+b(0.43)와 편향을 그대로 둔 det+noise(0.07)는 못 고친다.
  - **이중 해리**: 같은 변형이 두 계열에서 반대로 움직인다.
    - det+noise는 MW 8/8칸을 고치지만 Wall은 못 고친다.
    - det+BC는 Wall을 고치지만 MW는 3/8칸뿐이다.
    - 고정 무편향 오차는 Wall을 고치지만 MW는 0~1/8칸이다.
- 한정
  - DINO-WM의 읽기는 손상 칸 하나(Wall)에 기댄다.
  - TD-MPC2에서도 편향은 부분 성분이다. det+BC가 고친 칸은 MW 3/8, DMC 1/4, 317M 4/8이다.
  - 편향 벡터 b는 held 자료로 구했고 단계 감쇠 반복으로 수렴시켰다(D2).

**C3 어느 성분이 지배하는지는 계획기가 아니라 모델이 정한다 (E16)**
- 주장: 계획기를 서로 바꿔도 지속의 비용은 모델 계열을 따른다.
- 근거
  - TD-MPC2를 CEM식으로 바꿔도(elite 균등 가중, 정책 궤적 0) D(sh-ep) − D(sh-iter)가 3/3과제에서 남는다(push +47.8, pick-place +42.2점, cheetah +18.7%; 기본 MPPI +48.9, +28.9, +36.6).
  - DINO-WM을 MPPI식으로 바꿔도 D(sh-plan) − D(sh-iter)가 0과 구별되지 않는다(PointMaze +4 [−1, 10], Wall −7 [−18, 4]; 기본 CEM +0, −4).
  - 바꾼 계획기는 모두 자격을 얻었다. none 성공률(return)이 기본의 60% 이상이다.
- 한정
  - 기술 결과다(판정 문턱 없음).
  - 왜 모델이 정하는지(표 크기, 층당 단위 수, 잠재 구조)는 이 실험으로 가르지 않았다. 원고에는 E3의 표 크기 결과(H4a)와 함께 가설로 쓴다.

**C4 처방: 공유하고 반복마다 새로 뽑는 dither가 두 성분을 다 없애고 비트를 번다 (B4, P1, E12, E15)**
- 주장: 모든 후보가 공유하고 반복마다 새로 뽑는 dither(sh-iter)는 학습 없이 두 성분을 함께 없앤다. 범위를 per-channel로 잘 잡은 양자화기 위에서도 쓸 수 있는 최소 비트를 1 bit 이상 낮춘다.
- 근거
  - B4 성립(E1 H3a의 독립 seed 복제): MW 7/8칸에서 sh-iter red ≥ 0.5이고, 해로운 칸이 없다.
  - P1 성립: 317M 8/8(red 0.69~1.31).
  - DINO-WM Wall: red 1.00.
  - Fr1·Fr2 성립: MW 7/8과제에서 PT·PC 모두 G ≥ 1이다. MW 6과제에는 1 bit에서 손상 10점 이하인 구성이 있다(모두 dither 구성).
  - DINO-WM: PushT G = 2(PT·PC·RC), Wall G = 2(PT·PC), PointMaze G = 0.
  - rollout으로 고른 범위(RC, WM-Quant의 핵심 생각)는 k\*를 바꾸지 않는다. dither는 그 위에서도 2 bit를 번다.
  - **새 잡음이 아니라 dither여야 하는 이유**(B3, 사전 예측과 반대)
    - det+noise는 지속만 없애고 평균 편향은 남긴다.
    - 그래서 지속이 지배하는 MW에서는 8/8칸을 고친다(red 0.52~0.92).
    - 편향이 지배하는 Wall(0.07)과 DMC(1/4)에서는 못 고친다.
    - dither는 무편향이면서 반복마다 새로 뽑히므로 두 경우 모두 고친다.
- 한정
  - DMC는 반복 수의 경계다. 기본 반복 6에서 sh-iter가 고친 칸은 2/4다.
    - 반복 12에서는 3/4과제 red ≥ 0.5다(It1).
    - 3/4과제에서 red가 반복과 함께 단조로 커진다(It2).
    - walker는 반복 24에서도 red 0.21이다. 기본 반복에서는 sh-iter가 해롭다.
  - DMC의 G는 대부분 0이다(PT 0·0·0·1, PC 1·0·0·1).
  - 반복을 늘리면 결정당 계획 시간이 늘어난다(cheetah 반복 6 → 24에 30 → 124 ms).

**C5 진단 (E13)**
- 주장: TD-MPC2에서는 같은 비트 안의 반올림 방식 사이 손상 차이가 잠재 1-step 예측 오차로 보이지 않는다. 계획기가 보는 후보 return 오차의 지속 몫과 상대 크기(ρ_p × σ_e/σ_J)로는 보인다.
- 근거
  - Dg1 사전 지정 비교: 칸 안 Spearman의 평균 차가 MW 0.971(8칸 모두 양수), DMC 0.387이다.
  - 공통 held 상태에서 det의 1-step 오차가 sh-iter 이하인 칸이 E1 36칸 중 27칸이다.
- 한정
  - DINO-WM은 불성립이다(0.262; Wall 한 칸에서 0.905 − 0.643). 편향이 지배하는 곳에서는 1-step 오차도 det를 어느 정도 가른다.
  - DMC의 walker(−0.24)·quadruped(−0.12)에서는 1-step 오차보다 못 맞혔다.
  - 진단 주장은 TD-MPC2(MPPI)로 한정한다.

**C6 크기 (E14)**
- 주장: 317M 다중 과제 모델에서도 같은 실패, 처방, 지속의 비용이 새 seed로 다시 나온다.
- 근거: P1 성립(8/8), P2 성립(7/8).
- 한정
  - 체크포인트는 하나다.
  - GPU가 E10과 같은 RTX 4090이라 복제는 seed 쪽만이다.
  - sh-ep는 sweep-into에서 det보다 해롭다(r = 2.61).

**C7 비용과 정수 경로 (E17)**
- 주장: 처방(sh-iter)은 정수 kernel에서 rollout 지연을 2% 남짓 더할 뿐이고, 실제 정수 경로에서도 fake-quant와 같은 결과를 낸다.
- 근거
  - sh-iter 지연 오버헤드는 6조건 모두 ≤ 2.01%다(5M·317M × 후보 512·2048·8192).
  - K2 성립: 정수 det − det+shift가 4/4과제에서 fake-quant와 같은 부호이고 CI가 겹친다.
  - 접은 per-channel 정수 구성이 fake-quant per-channel과 13/16칸에서 CI 안에서 같다.
- 한정
  - det+shift의 오버헤드는 N 512에서 5.3%(5M)·7.2%(317M)다(K1 불성립). 그래서 det+shift는 기전 대조이자 처방의 한 형태로만 쓰고, 공동 처방으로 올리지 않는다.
  - 미리 바꾼 BF16에는 작은 손상이 있다(door-open +5.6점, cheetah +2.1%; K3 불성립).
  - 속도 주장은 RSS C4 수정판(11-01) 그대로다. A40에서는 미리 바꾼 BF16이 더 빠르다.

## 3. 쓰지 않을 주장 (CLAIMS_rss §3에 더함)

- "결과를 정하는 것은 지속이다"를 두 계열에 일반 문장으로 쓰는 것. DINO-WM은 편향 지배이고(B2), 지속의 비용이 없다(E16).
- "어떤 잡음도 det 손상을 줄이지 못한다", "무작위화만 있으면 된다". B3 결과를 보면 det+noise는 MW 8/8칸을 고치지만 Wall과 DMC에서는 통하지 않는다.
- "해상도 손실이 det 손상의 원인이다". MW에서 고정 무작위 문턱이 0~1/8칸만 고쳤다.
- "bias correction이 처방이다". 고친 칸은 MW 3/8, DMC 1/4, 317M 4/8이다.
- "DINO-WM PushT 4-bit의 det 손상이 새 분할에서도 재현된다". eval8에서는 +4점이었다.
- "ρ_p × σ_e/σ_J 진단이 모든 계열에서 1-step 오차보다 낫다". DINO-WM에서는 불성립이다.
- "det+shift를 지연 비용 없이 쓸 수 있다". K1이 불성립이다.
- "미리 바꾼 BF16은 폐루프 손상이 없다". K3이 불성립이다.
- "기본 반복 수에서 DMC도 처방이 통한다". E11 DMC의 sh-iter는 2/4이고, walker는 반복 24에서도 실패한다.

## 4. 결정이 필요한 것

| # | 정할 것 | 추천 |
|---|---|---|
| T1 | C1~C7 주장 묶음 | 승인. 특히 C2·C3(두 성분과 모델 의존), C4(B3 결과를 처방의 이유로 쓰는 것), C5의 한정 |
| T2 | 제목 유지 | 유지(설계 §1.5의 처리) |
| T3 | det+shift의 위치 | 기전 대조이자 처방의 한 형태로 쓴다. 공동 처방으로 올리지 않는다(K1 불성립) |
| T4 | DINO-WM 보강: PushT 3-bit E11 칸을 새 pod에서 [사후]로 더 돌릴지 | 돌린다. 약 4.5 h, 약 $4.2다. 돌리면 DINO-WM 손상 칸이 둘이 된다 |
| T5 | arXiv v2 | 주장 승인 뒤 원고를 고쳐 12월 초에 올린다. 새 결과의 공개 날짜를 남긴다(PLAN §4.9) |
| T6 | E18(LeWM, 제3 계열) | T4와 함께 정한다. C3("모델이 정한다")을 세 번째 계열로 넓힐 수 있는 유일한 실험이다. PLAN §4.7의 조건(스모크 재현, 일정, 예산)을 따른다 |

## 5. 승인 기록 (2026-11-22T01:35Z, 한국 시간 10:35)

- 사용자 답(원문): "너의 추천대로 해주고". §4의 추천을 그대로 승인했다.
  - T1: C1~C7 승인
  - T2: 제목 유지
  - T3: det+shift는 기전 대조이자 처방의 한 형태로 쓴다. 공동 처방으로 올리지 않는다
  - T4: PushT 3-bit E11 칸을 [사후] 보강으로 돌린다(편차 D4, DESIGN_tmlr §11)
    - 결과가 나오면 C1·C2의 DINO-WM 문장(손상 칸 수, 편향 지배의 근거)을 그 결과에 맞춰 고친다
    - 사전 지정 판정(B2·B3·Dg1)은 바꾸지 않는다
  - T5: arXiv v2는 원고를 고친 뒤 12월 초에 올린다
  - T6: 추천은 "T4와 함께 정한다"였다. 조정자 판단으로 지금은 E18을 하지 않는다. PLAN §4.7대로 두 계열로 한정해 쓰면 TMLR 기준에 지장이 없다. PushT 3-bit 결과를 본 뒤 필요하면 다시 제안한다
- 이 승인으로 원고의 주장은 C1~C7이다. PushT 3-bit 결과를 반영한 문구 수정은 이 절에 덧붙여 기록한다.
