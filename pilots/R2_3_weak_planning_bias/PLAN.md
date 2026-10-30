# R2-3 Weak activations and planning: bias or information? — 사전등록 계획 (PLAN)

- 작성: 2026-10-21T04:39Z(본 실행 시작 04:39:26Z 직전). **본 실행(main run) 전에** 작성·고정했다.
- 스모크(mw-push, 2 episode, α ∈ {0.1, 0.2})는 파이프라인 확인용으로 먼저 돌렸다. 그 수치는 §7에 그대로 적는다.
- 상위 문서: [../IDEAS_ROUND2.md](../../../archive/selection_process/IDEAS_ROUND2.md) R2-3 · 선행 파일럿: [../R1_1_neartie_planning/REPORT.md](../R1_1_neartie_planning/REPORT.md) · 규칙: [../runpod_infra/PILOT_RULES.md](../../../runpod_infra/PILOT_RULES.md)
- pod `pai-r2n3`(RTX A5000, $0.27/h, CPU 할당 7.65코어), 04:32Z 생성. 예산 상한 $3, 목표 $2.4 이내

## 1. 질문 (IDEAS_ROUND2.md 요약)

**R1-1의 사후 결과.** TD-MPC2에서 1-step 예측 오차를 맞춘 무작위 제거와 비교하면, 약한 활성 제거(0 대입)가 계획을 더 망가뜨렸다. 20칸 중 12칸에서 90% CI가 0을 벗어났다.

**경쟁하는 두 설명**
- **(i) 편향:** Mish의 약한 활성(음수 꼬리 포함)을 0으로 만들면 평균이 이동한다. 그 편향이 롤아웃과 MPPI 반복에서 쌓인다.
- **(ii) 정보:** 약한 활성의 표본별 요동이 계획에 필요한 미세 구분을 담는다.

## 2. 설정 (R1-1과 같음)

| 항목 | 값 |
|---|---|
| 코드 | R1-1 코드를 `code/`에 복사해 수정했다. 공식 tdmpc2 repo(같은 commit 계열), venv `/root/venv_mw`(torch 2.7.1, dm-control 1.0.16, mujoco 3.1.2, Meta-World@04be337 + mujoco-py) |
| 모델 | 공식 단일 과제 5M 체크포인트 `<task>-1.pt`. mpc=True, compile=False, `model.eval()` |
| 과제 | mw-push, mw-pick-place, cheetah-run, walker-walk |
| α | {0.1, 0.2, 0.3} → 12칸(과제 × α) |
| MPPI | 기본값(horizon 3, iteration 6, 표본 512, elite 64, policy 궤적 24, temperature 0.5) |
| episode | 조건마다 10개. seed 1000~1009를 모든 조건이 공유하는 짝 설계 |
| calibration | 비제거 episode 10개. 0~4로 m_j와 통계를 구하고, 5~9를 held-out 예측 세트로 쓴다(후보 시퀀스 6만 개) |
| 제거 대상 | dynamics MLP 은닉 2층 + reward MLP 은닉 2층의 Mish 출력(층당 512 unit). encoder, Q, policy는 그대로 둔다 |

## 3. 조건 정의

- **약한 집합:** 행(입력 표본 1개)마다 W = {j : |a_j| < α·m_j}.
- **m_j:** 계획 시점 입력에서 |a_j|의 99.9 백분위(R1-1과 같은 방식).
- **통계(층·unit·α별):** calibration split의 **비제거** 계획 시점 활성에서 구한다.
  - μ_j = E[a_j | j ∈ W]
  - δ_j = −E[a_j·1(j ∈ W)]
  - ā_j = E[a_j]

| 조건 | 정의 | 무엇을 없애나 |
|---|---|---|
| **A** | W의 값을 0으로(R1-1 weak) | 요동과 평균 |
| **B** | W의 값을 μ_j로 | 요동만(평균 보존) |
| **C** | 모든 활성에 δ_j를 더함 | 없음. A의 평균 이동만 넣는다 |
| **D1z** | 무작위 unit을 0으로. 1-step 오차를 A에 맞춤 | R1-1 rand_pred 재현 |
| **D3z** | 무작위 unit을 0으로. 3-step 오차를 A에 맞춤 | |
| **D1m** | 무작위 unit을 ā_j로(평균 보정). 1-step 오차를 B에 맞춤 | **판정에 쓰는 D** |
| **D3m** | 무작위 unit을 ā_j로. 3-step 오차를 B에 맞춤 | |

**무작위 대조의 세부**
- 선택 방식: 행마다 W의 크기 c를 센다. 각 unit을 확률 min(1, r·c/n)로 독립 선택한다. 별도 CUDA generator를 쓰므로 planner의 난수열은 모든 조건에서 같다.
- **r_dyn 맞춤:** dynamics 층의 r을 기하 이분 탐색한다(고정 mask seed, 22회). 대상은 held-out 세트에서 대상 조건의 err1(1-step latent) 또는 err3(3-step latent rollout)이다.
- **r_rew 맞춤:** 모든 D에서 reward 층의 r은 1-step reward 오차를 대상 조건(z계열은 A, m계열은 B)에 맞춘다. reward는 1-step head라서다.
- **매칭 확인:** 맞춘 r로 다른 mask seed를 써서 오차를 다시 재고 보고한다.

**설계 선택(코드 작성 시점, 스모크 전에 정함).**
- 판정 문구의 "1-step 오차를 맞춘 무작위(D)"는 **D1m**으로 해석한다.
- 이유: B는 평균을 보존한다. 비교 대상도 평균을 보존하는 무작위여야 편향이 섞이지 않고 "약한 집합의 요동 vs 무작위 요동"만 비교된다.
- 0 대입 무작위 D1z/D3z는 R1-1 재현과 A의 비교 대상으로만 쓴다.

## 4. 측정

### 4.1 예측
held-out 계획 시점 입력에서 원래 모델 출력 대비 정규화 오차를 잰다.
- err1: 1-step latent
- err3: 3-step rollout latent
- errr: reward
- 맥락: 실제 다음 관측 encoder 대비 오차

### 4.2 계획
- 상대 하락 = (R0 − R) / R0. R0는 비제거 10 episode 평균이다.
- DMC는 return, MW는 마지막 step 성공률(R1-1과 같음)이다.
- 보조: MW의 return 기준 하락.

### 4.3 결정 수준
- 상태: 비제거 calibration rollout에서 과제당 300개
- **한 번 평가의 순위 보존:** 실제 계획의 첫 iteration과 마지막 iteration 512개 후보 집합을 고정한다. 원래 모델과 제거 모델의 `_estimate_value`를 같은 seed로 비교한다.
  - 지표: Spearman, elite Jaccard, elite regret, MPPI 가중치 겹침
- **최종 계획 행동 편차:** 상태 150개에서 저장한 obs와 직전 mean으로 `_plan`을 다시 돌린다. 최종 mean 첫 step의 RMS 편차를 seed noise 바닥(원래 모델, 다른 seed)과 비교한다.

## 5. 사전등록 판정 기준 (IDEAS_ROUND2.md에서 그대로 옮김)

- 판정 A (편향 설명): 두 조건이 모두 성립한다.
  - B의 계획 하락이 1-step 오차를 맞춘 무작위(D)보다 유의하게 크지 않은 칸이 80% 이상이다.
  - C가 A의 계획 하락의 70% 이상을 재현하는 칸이 과반이다.
- 판정 I (정보 설명): B가 D보다 계획을 더 망가뜨리는 칸(90% CI가 0을 벗어남)이 12칸 중 6칸 이상이고, C는 A 하락의 50% 미만이다.
- 둘 다 아니면 혼합으로 판정한다. 그 경우 각 성분의 몫을 보고한다.
- 논문 방향은 판정에 따라 갈린다.
  - A: 평균을 보존하는 활성 희소화는 계획에서도 공짜다. 원문 지지이자 on-robot 월드모델 가속 처방이다.
  - I: 약한 활성은 예측에는 공짜지만 계획에는 필요하다. 원문 주장의 제어 영역 경계다.

### 5.1 실행 전에 채운 세부 규칙 (주어진 기준값은 바꾸지 않았다)

1. **칸과 지표:** 칸은 (과제, α) 12개다. 하락 지표는 §4.2다(DMC는 return, MW는 성공률).
2. **"D":** D1m이다(§3).
3. **유의성:** 같은 seed로 짝지은 episode별 차이 (하락_B − 하락_D1m)에 paired bootstrap 90% CI(4000회)를 매긴다.
   - "유의하게 크지 않다" = CI 하한 ≤ 0
   - "더 망가뜨린다(CI가 0을 벗어남)" = CI 하한 > 0
   - MW에서 두 조건이 모두 0% 또는 100%라 차이가 전부 0이면 CI는 [0, 0]이고, "유의하게 크지 않다"에 들어간다.
4. **판정 A 조건 1:** 12칸 중 10칸 이상(80% = 9.6 → 10).
5. **판정 A 조건 2**
   - "C가 A 하락의 70% 이상을 재현" = 하락_C ≥ 0.7 × 하락_A.
   - 대상 칸: A의 손상이 의미 있는 칸만 센다. 하락_A ≥ 0.10이고, A 하락(기준선 대비 episode 짝 차이)의 90% CI 하한 > 0인 칸이다.
   - 과반: 대상 칸 수의 절반 초과.
   - 대상 칸이 0개면 조건 2는 성립하지 않은 것으로 본다.
6. **판정 I 조건 2:** 대상 칸에서 Σ하락_C / Σ하락_A < 0.5(합산 비율). 칸별 비율의 중앙값도 함께 보고한다.
7. **판정 순서:** A와 I가 동시에 성립할 수는 없다. A 조건 1(10칸 이상 "크지 않음")과 I 조건 1(6칸 이상 "더 망가뜨림")은 함께 참일 수 없기 때문이다. 둘 다 아니면 "혼합"이다.
8. **혼합일 때 보고할 몫:** 대상 칸별 하락_C/하락_A(편향 몫), 하락_B/하락_A(요동 제거 몫)와 합산 비율.
9. **보조 비교(판정 밖)**
   - A vs D1z: R1-1 재현
   - A vs D3z: 3-step 오차까지 맞추면 A의 초과 손상이 설명되는가
   - B vs D3m
   - MW return 기준 판정
   - 결정 수준 지표

## 6. 실행 순서와 우선순위

- 순서: calib(과제당 10 episode) → prep → grid(880 episode) → decision → analyze
- **job 순서(조정자 지시 우선순위):**
  1. 기준선
  2. MW 2과제 + cheetah의 α=0.2
  3. 같은 과제의 α=0.1, 0.3
  4. walker
- 시간이나 예산이 모자라면 뒤에서부터 자른다. 자른 사실은 REPORT에 적는다.
- CUDA MPS를 쓰고 워커는 7개다.

## 7. 스모크에서 본 것 (투명성)

스모크는 mw-push, calibration 1 episode(본 실행과 m_j가 다름), α ∈ {0.1, 0.2}, 조건마다 2 episode였다. 파이프라인이 끝까지 돌았다(calib → prep → grid 30 job → decision 20 상태).

- **예측 쪽(α=0.2)**
  - err1: A 0.114, B 0.197, C 0.081
  - err3: A 0.418, B 0.759, C 0.628
  - B의 예측 오차가 A보다 컸다.
  - 무작위 대조 매칭: D1z err1 0.114, D1m 0.196, D3z err3 0.417, D3m 0.756
- **계획(α=0.2, 2 episode)**
  - 성공: A 1/2, B 0/2, C 0/2, D 네 종류 모두 2/2
  - α=0.1에서는 모두 2/2

이 수치를 본 뒤 바꾼 판정 규칙은 없다. §5.1의 해석(D = D1m, 대상 칸 정의)은 판정 문구와 설계에서 정했다.
