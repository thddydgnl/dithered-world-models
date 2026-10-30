# R2-7 사전등록 계획: The near-tie regime of iterative latent planning

- 작성: 2026-10-21 06:25Z(pilot worker `r2n7`, R2-1 작업자가 이어서 진행). 본 실행 전
- pod: `pai-r2n7`, NVIDIA A40 48GB, $0.59/h, 생성 06:21Z
- 비용: 상한 $3, 목표 $2.4 이하
- 이 문서를 쓰기 전에 본 것: R2-1 결과 전체. 특히 사후 분석의 CEM 모집단 순위 하락이다.
  - PushT 0.906 → 0.710 → 0.529, Wall 0.918 → 0.555 → 0.510
  - R2-1 평가 쌍(seed 2) 50 episode에서 얻었다.
- 이번에 쓰는 시작·목표 쌍(seed 3, 4)과 planner seed는 아직 아무 결과도 보지 않았다.

## 1. 질문과 가설

CEM이 반복되면 후보 집단 안에서 latent 비용과 참 비용의 순위 상관이 떨어진다. 이것이 후보가 수렴하며 참 비용 범위가 줄어드는 범위 축소(range restriction)로 저절로 생기는 것인가? 아니면 월드모델 오차가 후보 간에 구조화되어 해상도가 무너지는 "near-tie 영역"인가? 그렇다면 해상도를 의식한 반복 일정이 계획을 돕는가?

- H1 (해상도 붕괴): 반복이 진행되면 후보의 참 비용 spread가 모델 비용 오차의 크기 아래로 내려간다. 해상도 비 R = sd(모델 비용 오차) / sd(참 비용)이 1을 넘는다.
  - 범위 축소만으로 설명되지 않는다는 근거도 필요하다. 0단계에서 맞춘 고정 분산 잡음을 참 비용에 더한 oracle과 비교해, 같은 참 비용 spread에서 모델의 후반 Spearman이 더 낮아야 한다.
- H2 (성공 예측): 마지막 단계의 해상도가 episode 성공을 예측한다.
- H3 (해상도 인지 계획): 참값 없이 계산되는 해상도 대리 지표로 반복 일정을 바꾸면, 같은 계산량에서 성공률이 오른다.
- H4 (원문 연결, 권장): 평균 대체 약한 활성 제거가 후반 단계의 순위를 초반보다 2배 이상 더 망가뜨린다. 후보 간 일관된 고정 무작위 대조(D-fix)는 그렇지 않다.

## 2. 플랫폼 (R2-1에서 검증한 것 재사용)

- DINO-WM commit 0a9492f. 체크포인트 `pusht`, `wall_single`. 인코더는 frozen DINOv2 ViT-S/14로 patch 196개 × 384다.
- 계획 비용은 DINO-WM `objective_fn_last`로, 시각 MSE + α·proprio 임베딩 MSE(α=1)다.
- CEM: DINO-WM `CEMPlanner`와 같은 의미
  - 300 samples, top-30, var_scale 1, 첫 후보 = mu
  - opt_steps는 PushT 30, Wall 10
  - goal_H 5. 시작·목표는 PushT `dset`, Wall `random_state`로 과제 설정을 따른다.
- open-loop(= MPC 1회 반복)로 실행한다. 설정의 MPC max_iter=null은 끝나지 않는다(R2-1 편차 5와 같음).
- 월드모델은 bf16 autocast, 배치 구현으로 돌린다. 여러 episode를 한 forward로 처리하고, 시작 이미지는 한 번만 encode한다.
  - R2-1 검증: TF32 대비 비용 Spearman ≥ 0.9997, top-30 30/30, episode별 구현 대비 성공 여부 4/4 일치
- 참 비용: 같은 시작 상태에서 시뮬레이터로 재생한다(렌더링 없는 빠른 재생, R2-1에서 env.rollout과 차이 0).
  - 1차는 margin = 평가기 성공 정의의 연속형이다. PushT는 max(‖Δ[agent_xy, block_xy]‖/20, Δangle/(π/9)), Wall은 ‖Δxy‖/4.5다.
  - 2차는 `state_dist`다.
- 편차: PointMaze는 R2-1과 같은 이유(mujoco-py)로 제외한다. Wall 데이터는 obs 없이 메타 파일만 푼다(풀면 59GB). Wall 쌍은 환경이 렌더링하므로 obs가 필요 없다.

## 3. 분할과 seed

| 분할 | 시작·목표 쌍 seed | episode | planner seed | 용도 |
|---|---|---|---|---|
| held(보류) | 3 | 과제당 20 | 1000+e | 일정 규칙과 문턱 선택, S4 보정 |
| eval(평가) | 4 | 과제당 50 | 2000+e | S3 판정(같은 seed의 기본 CEM 대비), S4 측정 |

- R2-1이 쓴 seed 1, 2와 겹치지 않는 새 seed다.
- 기본 CEM은 두 분할 모두에서 돌린다. 이때 단계별 모집단을 기록한다.
  - PushT step {0, 4, 9, 14, 19, 24, 29}, Wall {0, 2, 4, 6, 9}
  - 후보 300개의 행동, J, 시각 항 V, proprio 항 Q
- 기록한 모집단은 모두 시뮬레이터로 재생해 참 비용을 얻는다.
- S1과 S2에는 선택 과정이 없으므로 기본 CEM episode 전체(과제당 70)를 쓴다.

## 4. 측정 정의

### 4.1 단계별 순위·해상도 (S1)
- Spearman_model(s): episode마다 모집단 안에서 Spearman(J, y)를 구하고, episode 평균을 낸다. y는 margin이다.
- **해상도 비 R(s)**
  - 정의: R(s) = sd_pop(ε) / sd_pop(y)이고 ε = a_e + b_e·J − y다.
  - a_e, b_e는 그 episode의 0단계 모집단에서 y를 J에 OLS로 맞춘 값이다. 같은 episode의 모든 단계에 고정한다.
  - 요약: episode 중앙값
- **고정 잡음 oracle(1차)**
  - episode마다 σ0을 정한다. 0단계 모집단에서 E[Spearman(y + σ0·ξ, y)]가 모델의 0단계 Spearman과 같아지는 값이다.
    - ξ ~ N(0,1), 고정 난수 64회, 이분 탐색
    - Spearman을 맞추므로 J의 단조 변환에 영향을 받지 않는다.
  - 같은 σ0으로 각 단계 모집단의 oracle Spearman = E[Spearman(y_s + σ0 ξ, y_s)]를 계산한다.
  - 같은 모집단의 y를 쓰므로 참 비용 spread는 자동으로 맞춰진다.
  - Δ(s) = oracle − 모델(episode 평균, episode bootstrap 95% CI)
- 2차 oracle: σ0 = 0단계 OLS 잔차의 sd. 2차 R: 과제 공통 보정(전체 episode 0단계 pooled OLS)

### 4.2 참값 없는 해상도 대리 지표 (매 opt step 계산, 추가 forward 없음)
- **R_boot(1차 대리 지표)**
  - patch bootstrap을 20회 한다(고정 다항 가중치).
  - 각 후보 비용의 bootstrap 값에서 모집단 평균을 뺀다.
  - 그 sd의 후보 중앙값을 모집단 비용 sd로 나눈다.
  - 정의: R_boot = median_i sd_b(J_i^(b) − J̄^(b)) / sd_i(J_i)
- 2차(보고만): bootstrap 순위 안정도(bootstrap 비용과 J의 Spearman 평균), ρ_VQ = Spearman(V, Q), 비용 spread sd(J)

### 4.3 성공 예측 (S2)
- 과제별 기본 CEM 70 episode
- 마지막 단계 R(참값 사용, 1차)과 마지막 단계 R_boot(참값 없음, 1차)로 실패를 예측하는 AUROC를 구한다. 점수가 높을수록 실패로 본다.
- episode bootstrap CI를 붙인다. 2차 대리 지표 AUROC는 보고만 한다.

### 4.4 해상도 인지 일정 (S3)
- 문턱 τ_q: 과제별로 held 기본 CEM의 중간 단계(PushT step 14, Wall step 4) R_boot 분포의 q 분위수다.
  - 규칙(q)은 과제 공통이고, 수치는 과제별 보류 데이터에서 정한다.
- 후보 규칙. 모두 참값을 쓰지 않고, 계산량은 기본 CEM 이하다.
  - **ES(q)** 조기 정지: step t ≥ 1에서 처음으로 R_boot_t > τ_q이면 그 단계의 elite 갱신 후 mu를 반환하고 멈춘다. q ∈ {0.25, 0.5, 0.75}
    - 기본 CEM 실행에서 단계별 mu를 모두 재생해 두므로 추가 계산 없이 평가된다.
  - **FV(0.5)** 분산 고정: 처음 R_boot_t > τ_0.5인 단계부터 sigma를 그 단계 값으로 고정한다. mu는 계속 갱신한다. 계산량은 같다.
  - **AK(0.5)** elite 확대: 처음 R_boot_t > τ_0.5인 단계부터 top-k를 30에서 90으로 늘린다. 계산량은 같다.
- 선택(held, 같은 seed의 기본 CEM 대비): 다음 순서로 가장 좋은 규칙 하나를 고른다.
  1. 과제 평균 Δ성공률(%p)이 가장 큰 것
  2. 동률이면 과제 평균 Δmargin(작을수록 좋음)
  3. 그래도 동률이면 계산량이 적은 것
  - 선택된 규칙은 held에서 기본보다 나쁘더라도 평가에 쓴다.
- 평가(eval 50 episode, 같은 seed): 성공률 차이, McNemar exact p, 최종 margin 짝 비교

### 4.5 약한 활성 제거 (S4, 권장)
- 대상: predictor MLP 6개 층의 GELU 출력(층당 2048 unit)
- 보정: held 기본 CEM의 기록 모집단을 다시 계산하며 GELU 출력 통계를 모은다.
  - m_j = |a_j|의 99.9 분위수
  - μ_j = E[a_j | |a_j| < α m_j]
  - ā_j = E[a_j]
- **weak-meanrep(α=0.2)**: 행(후보 × 토큰)마다 |a_j| < α·m_j인 값을 μ_j로 바꾼다.
- **D-fix**
  - 층마다 고정된 무작위 unit 집합 U를 정한다. 모든 행·후보에 같은 U를 쓴다.
  - U의 값을 ā_j로 바꾼다.
  - |U| 맞춤: held 모집단에서 예측 최종 latent(시각+proprio)의 평균 제곱 변화가 weak-meanrep과 같아지도록, 고정 순열의 앞쪽 비율을 이분 탐색한다.
- 측정
  - eval 기본 CEM 50 episode의 0단계(초반)와 마지막 단계(후반) 모집단을 쓴다.
  - 손상 = ρ(J_base, y) − ρ(J_abl, y)를 구하고 episode 평균을 낸다.
  - 비 = 후반 손상 / 초반 손상
  - 초반 손상 평균이 0 이하이면 비를 정의할 수 없으므로 S4 실패로 둔다.
- 2차: 순위 교란 1 − Spearman(J_abl, J_base). α = 0.1, 0.3은 시간이 남으면 보고한다.

## 5. 사전등록 판정 기준 (IDEAS_ROUND2.md R2-7에서 복사)

- S1: 과제 하나 이상에서 마지막 단계 R > 1이고, 같은 참 비용 spread의 고정 잡음 oracle보다 모델의 Spearman이 0.1 이상 낮다.
- S2: 과제 하나 이상에서 마지막 단계 R(또는 대리 지표)이 episode 실패를 AUROC 0.65 이상으로 예측한다.
- S3: 사전에 고른 해상도 인지 일정 하나가 과제 하나 이상에서 성공률을 5%p 이상 올리고, 다른 과제를 3%p 넘게 떨어뜨리지 않는다(같은 seed 50 episode, McNemar 보고).
- S4 (권장): 약한 제거의 후반/초반 순위 손상 비가 2 이상이고 D-fix의 같은 비보다 크다.
- 판정
  - S1과 (S2 또는 S3)를 만족하면 있음
  - S1만 만족하면 조건부
  - S1이 실패하면(범위 축소로 충분히 설명되면) 없음

### 열린 부분을 채운 값
- S1의 "마지막 단계 R > 1"은 episode 중앙값 R이다. "0.1 이상 낮다"는 1차 oracle 기준 episode 평균 Δ(마지막) ≥ 0.1이다.
- S2의 "R(또는 대리 지표)"는 사전 지정한 두 개로 한정한다: 마지막 단계 R과 마지막 단계 R_boot. 둘 중 하나가 AUROC ≥ 0.65이면 통과다. 2차 대리 지표는 판정에 쓰지 않는다.
- S3는 4.4의 선택 규칙으로 고른 하나만 판정한다.
- S4는 α=0.2 하나로 판정한다.

## 6. 통제와 주의
- 범위 축소 대조: 1차 oracle과 2차 oracle을 모두 보고한다.
- 일정 규칙 선택(held)과 판정(eval)은 서로 다른 쌍과 seed로 분리한다. 같은 seed로 짝 비교하고 McNemar를 보고한다.
- S4의 무작위 대조는 후보 간 일관된 고정 집합이다. R2-3에서 후보마다 독립 mask를 쓰면 후보 간 평균으로 상쇄된다는 것을 보았기 때문이다.
- 결과는 과제마다 공개 체크포인트 1개에 대한 것이다.

## 7. 예산·시간 규칙
- 측정된 속도(A40, bf16): PushT 1 episode 약 30s, Wall 약 4.5s(GPU를 혼자 쓸 때)
- 예상 계산량
  - held: PushT 기본·FV·AK 60 episode(30분), Wall 60 episode(5분)
  - eval: 기본 50 + 선택 규칙 50(ES면 0)
  - PushT 최대 50분
- S4: eval 기본이 끝나고 pod 생성 후 2h15m 이내, 지출 $1.8 이하일 때만 한다.
- 스모크: held 2 episode로 기본·FV·AK, 재생, 분석까지 확인한다.
- 모든 결과는 episode마다 jsonl과 npz로 저장한다. 결과는 주기적으로 회수한다.

## 8. 변경 기록
- (없음)
