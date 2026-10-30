# R1-1 Planning lives in the near-ties — 사전등록 계획 (PLAN)

- 작성: 2026-10-21T03:15Z. **본 실행(main run) 전에** 작성·고정했다. 스모크 테스트는 파이프라인 동작과 속도 확인에만 썼다.
- 상위 문서: [../IDEAS_ROUND1.md](../../../archive/selection_process/IDEAS_ROUND1.md) R1-1 · 규칙: [../runpod_infra/PILOT_RULES.md](../../../runpod_infra/PILOT_RULES.md)
- pod `pai-r1n1` (RTX A5000, $0.27/h, CPU 할당 약 7.6코어), 생성 03:01Z. 예산 상한 $4, 목표 $3.2 이내

## 1. 질문과 가설 (IDEAS_ROUND1.md 그대로)

원문 *Neural Networks Are Already Clean*은 약한 활성을 지워도 손실이 거의 없고, 바뀌는 답은 이미 near-tie였던 것뿐이라고 한다. MPC 계획은 비슷한 후보 수백 개의 예측 return을 비교해 고르는 일이므로, 계획 자체가 near-tie 문제다.

- **H1 (해리):** 예측 오차가 거의 늘지 않는 약한 활성 제거가 계획 성능은 크게 떨어뜨린다. 비교 대상은 **예측 오차를 같게 맞춘 무작위 제거**다.
- **H2 (near-tie 국소화):** 행동 선택이 바뀌는 곳은 후보 간 가치 차이가 작은 결정 시점에 몰려 있다.
- **H0 (원문 그대로):** 약한 활성 제거는 예측과 계획 모두에 거의 공짜다.

## 2. 설정

| 항목 | 값 |
|---|---|
| 코드 | 공식 `nicklashansen/tdmpc2` commit `332f19e`(2026-10-20). hydra 앱 없이 `parse_cfg`로 cfg 생성(`hydra.utils.get_original_cwd` 패치) |
| 모델 | 공식 단일 과제 체크포인트 `<task>-1.pt`(5M). obs=state, model_size=5, compile=False, mpc=True. `TDMPC2.__init__`이 `model.eval()`을 부르므로 공식 평가와 같다 |
| 과제 | walker-walk, cheetah-run(DMControl, 500 결정 step, action repeat 2), **mw-push, mw-pick-place**(Meta-World v2 goal-observable, 100 결정 step) |
| MPPI | 기본값 그대로: horizon 3, iterations 6, num_samples 512, num_elites 64, num_pi_trajs 24, temperature 0.5, min_std 0.05, max_std 2. `act(obs, t0, eval_mode=True)`(공식 eval과 같이 elite 중 Gumbel-softmax 샘플) |
| 할인율 | DMC 0.99, MW 0.95(repo 휴리스틱) |
| 환경 | 별도 venv `/root/venv_mw`: torch 2.7.1, tensordict 0.8.3, dm-control 1.0.16, mujoco 3.1.2, numpy 1.24.4 + Meta-World 블록(gym 0.21.0, cython<3, setuptools 65.5.0, wheel 0.38.0, mujoco-py 2.1.2.14 + MuJoCo 2.1.0, Metaworld@04be337) |

**Meta-World 설치 기록(본 실행 전).** 20분 시간 상자 안(약 12분)에 성공했으므로 대체 과제(finger-turn-hard, quadruped-walk)는 쓰지 않는다. 스모크에서 mw-push, mw-pick-place 모두 2/2 성공(return 1650~1810).
- uv가 gym 0.21.0의 옛 메타데이터(`opencv-python>=3.`)를 거부해 pip 23.3.1로 설치했다.
- mujoco210 압축 해제에 `--no-same-owner`가 필요했다.
- 호환 패치 1개: repo의 `envs/metaworld.py`가 `env.max_episode_steps = env._max_episode_steps`를 대입하는데, 이 버전의 `Timeout.max_episode_steps`는 setter가 없는 property라 예외가 난다. 같은 값을 넣는 setter를 붙였다(동작 변화 없음).

## 3. 제거(ablation) 정의

- **대상:** 월드모델 dynamics MLP 은닉 2층(`dyn0`, `dyn1`)과 reward MLP 은닉 2층(`rew0`, `rew1`)의 출력 Mish(LayerNorm(Linear(x))), 층당 512 unit. forward hook으로 출력을 바꾼다. dynamics 출력층(SimNorm), encoder, Q-ensemble, policy prior는 그대로 둔다. Q 제거 조건은 하지 않는다.
- **기준값 m_j:** 비제거 calibration 에피소드 0~4의 MPPI 계획 입력 분포에서 구한 |a_j|의 99.9 백분위.
  - 매 결정 step, 6 iteration마다 512 후보 중 무작위 16개를 기록한다.
  - 기록한 후보 시퀀스를 원래 모델로 horizon 3까지 rollout해 계획 중 MLP가 실제로 보는 입력 (z_h, a_h), h=0,1,2를 재현한다.
  - DMC는 과제당 72만 행이다.
- **weak(α):** |a_j| < α·m_j인 항목을 0으로 만든다. α ∈ {0.05, 0.1, 0.2, 0.3, 0.5}.
- **rand_count(α):** 행(입력 표본 1개)마다 weak가 지울 개수 c를 센다. 각 unit을 확률 c/n으로 독립적으로 0으로 만든다. 행별 기대 개수가 일치하고 unit은 균일 무작위다.
- **rand_pred(α) (핵심 대조):** 확률을 r·c/n으로 바꾼다.
  - r은 dynamics 층(r_dyn)과 reward 층(r_rew)에서 따로 기하 이분 탐색한다(18회, 고정 mask seed, 같은 seed면 mask가 r에 대해 포함 관계).
  - 맞추는 대상은 held-out 계획 시점 세트의 1-step 정규화 latent 오차(dynamics)와 reward 정규화 오차(reward)다. 둘 다 weak(α)와 같아지게 한다.
  - 다른 mask seed로 다시 잰 오차도 기록한다(맞춤의 일반화 확인).
- **strong(α) (정상 작동 점검):** 행마다 |a_j|/m_j가 가장 큰 c개를 0으로 만든다.
- **난수:** 무작위 mask는 별도 CUDA generator를 쓴다. 그래서 planner의 난수열은 모든 조건에서 같다.
- **0 제거를 쓰는 이유:** 원문 주장 자체가 "약한 활성을 0으로 만들어도 된다"이므로 평균 제거 대신 0 제거를 쓴다. 조건 간 비교는 같은 seed의 짝 비교로 한다.

## 4. 측정

### 4.1 예측 (held-out 세트)
held-out 세트는 calibration 에피소드 5~9의 계획 시점 후보 시퀀스 중 6만 개다.

- err1 = E‖d_abl(z_h,a_h) − d(z_h,a_h)‖² / E‖d − E[d]‖²
  - h = 0,1,2를 합친다. z_h는 원래 모델 rollout이고, d는 원래 모델의 다음 latent(SimNorm 출력)다.
- err3: 루트 latent에서 기록된 행동열로 3-step rollout한 latent의 오차(같은 방식으로 정규화)
- errr = E(r̂_abl − r̂)² / Var(r̂): two-hot을 복원한 scalar reward
- 제거 비율: dyn0·dyn1·rew0·rew1의 전체 항목 중 0이 된 비율(같은 held-out 입력). 계획 에피소드 중 실제 제거 비율도 따로 기록한다.
- 맥락(보조): 실제 다음 관측의 encoder 출력 대비 1-step 일관성 오차와 실제 reward 대비 오차. 원래 모델과 각 조건에서 잰다.

### 4.2 계획
- 과제·조건마다 10 episode를 돌린다.
  - strong과 rand_count는 판정에 쓰지 않는 대조라서 5 episode만 돌린다(속도 확인 후 정한 축소, §6).
- seed 1000~1009는 모든 조건에 공통이다. 환경을 그 seed로 새로 만들므로 같은 초기 상태와 같은 planner noise에서 시작한다.
- 지표: DMC는 return. MW는 마지막 step `info['success']` 비율(공식 eval 규약)이고, return도 함께 보고한다.
- 상대 하락 drop = (R0 − R) / R0. R0는 비제거(none) 10 episode 평균이다.
  - MW는 성공률을 기준으로 쓴다.
  - R0 = 0인 과제는 S2에서 제외하고 return 기준을 보조로 보고한다.

### 4.3 결정 수준
- **상태:** 비제거 calibration 에피소드마다 50개 결정 상태(DMC 10 step 간격, MW 2 step 간격)를 뽑아 과제당 500개를 쓴다.
- **후보 집합:** 각 상태에서 실제 계획이 만든 512개 후보 집합을 고정해서 쓴다.
  - 1차(S3 판정용)는 **첫 MPPI iteration**의 후보 집합이다(policy prior 24개 + 넓은 Gaussian 표본 488개).
  - 2차는 마지막 iteration의 후보 집합이다. 이 집합의 elite가 실행 행동을 정한다.
  - 처음에는 마지막 iteration을 1차로 정했다. 스모크 후 바꾼 이유는 §6.1에 있다.
- **가치 계산:** 후보마다 `agent._estimate_value`로 가치를 구한다(H-step reward + policy 행동의 할인 terminal Q).
  - 원래 모델과 제거 모델에 같은 torch seed를 써서 Q-head 부분집합과 policy noise를 같게 한다.
- **지표**
  - Spearman 순위상관, argmax 일치율, elite(top-64) Jaccard, 정규화 가치 오차
  - 원래 가치의 top1 − top2 gap 사분위별 argmax 뒤집힘 비율(Q1 = gap 최소). 사분위는 과제 안에서 나눈다.
- **대조**
  - seednoise: 원래 모델을 다른 seed로 돌린 것. 가치 추정기 자체의 무작위성을 잰다.
  - rand_pred, rand_count, strong
- **전체 계획 재실행:** 상태 250개에서 저장한 obs와 직전 plan mean으로 `_plan`을 같은 seed로 다시 돌린다.
  - 실행 행동과 최종 mean 첫 step의 RMS 편차를 잰다.
  - 이 편차를 seed noise(같은 모델, 다른 seed)와 비교한다.

## 5. 사전등록 판정 기준 (IDEAS_ROUND1.md에서 그대로 옮김)

- S1 (예측 쪽 원문 재현): α ≥ 0.1인 어떤 α에서 활성의 50% 이상을 지우고도 1-step 정규화 예측 오차 ≤ 0.10
- S2 (해리): S1을 만족하는 가장 큰 α에서, 약한 제거의 상대 return 하락이 예측 오차를 맞춘 무작위 제거보다 2배 이상 크다(과제 2개 이상). 또는 예측 오차 ≤ 0.10인데 상대 return 하락 ≥ 20%
- S3 (near-tie 국소화): gap 최하위 사분위의 argmax 뒤집힘 비율이 최상위 사분위의 2배 이상
- 판정
  - S1과 (S2 또는 S3)를 만족하면 있음
  - S1만 만족하고 계획도 멀쩡하면 "있음(다른 방향)": 제어 월드모델도 깨끗하다는 결과이며 희소화 가속 논문 후보
  - S1이 실패하면(약한 활성이 예측에도 필요한 dial형) 그 사실을 보고하고 없음/조건부를 판단

### 5.1 실행 전에 채운 세부 규칙 (주어진 기준값은 바꾸지 않았다)

1. **S1**
   - 과제별로 판정한다.
   - "활성의 50%"는 dyn0·dyn1·rew0·rew1 전체 항목 중 weak가 0으로 만든 비율이다(held-out 계획 시점 입력).
   - "1-step 정규화 예측 오차"는 dynamics의 err1이다.
   - α*(과제별)는 S1을 만족하는 가장 큰 α다.
   - 전체 S1은 4과제 중 2개 이상에서 성립할 때 충족이다. S2가 과제 2개 이상을 요구하기 때문이다.
2. **S2**
   - 첫 가지: 과제별 α*에서 drop_weak ≥ 2 × max(drop_rand_pred, 0)이고 drop_weak ≥ 0.05여야 한다.
     - 0.05는 최소 효과 하한이다. 두 값이 모두 노이즈 수준일 때 "2배"가 저절로 성립하는 것을 막는다.
     - 하한 없이 판정한 결과도 함께 보고한다.
     - 성립 과제가 2개 이상이어야 한다.
   - 둘째 가지: α와 관계없이 err1_weak ≤ 0.10이면서 drop_weak ≥ 0.20인 과제가 2개 이상이어야 한다. 첫 가지의 "과제 2개 이상"을 보수적으로 같이 적용했다. 1과제만 성립한 경우도 보고한다.
   - MW에서 "return 하락"은 성공률 하락이다(IDEAS: "Meta-World는 성공률").
3. **S3**
   - 1차 후보 집합(첫 iteration, §6.1)에서 과제별 α*(없으면 α = 0.1)의 weak 제거 argmax 뒤집힘을 쓴다. 마지막 iteration 집합의 같은 판정은 2차로 보고한다.
   - 4과제를 합쳐 Q1 뒤집힘 비율 ≥ 2 × Q4 뒤집힘 비율이고, Q1 뒤집힘이 10건 이상이면 충족이다.
   - 과제별 결과도 보고한다.
4. **"계획도 멀쩡"**: S1을 만족한 모든 과제에서 α*의 drop_weak ≤ 0.10이다.
5. **표에 없는 경우**
   - S1 성립, S2·S3 불성립, 계획 하락 > 10%이면 "없음"이다. 계획 손실이 예측 손실로 설명된다는 뜻이다.
   - S1 실패 시: S2′이 2과제 이상에서 성립하거나 S2 둘째 가지가 성립하면 "조건부 있음", 아니면 "없음"이다.
     - S2′: err1 ≤ 0.10인 가장 큰 α에서 S2 첫 가지 규칙을 적용한 것이다.
6. **사전등록 보조 분석(판정에 넣지 않음)**
   - D1 (S3 특이성): weak의 Q1 뒤집힘 비율이 seednoise와 rand_pred의 Q1보다 큰가. S3는 어떤 작은 교란에도 성립할 수 있는 약한 기준이라 이 점검을 함께 본다.
   - D2: 전체 계획 재실행 편차를 weak, rand_pred, seednoise 사이에서 비교한다.
   - D3: 실제 관측 대비 일관성 오차의 변화.
   - D4: 2차 후보 집합(마지막 iteration)의 같은 지표. 공통 이동을 뺀 정규화 가치 오차(value_nerr_centered)도 함께 본다.

### 5.2 PILOT_RULES 대조군 항목과의 관계
- 이 파일럿은 뉴런 선택성을 주장하지 않는다. 그래서 회전 기저 null은 핵심 대조가 아니다.
- 핵심 null은 예측 오차를 맞춘 무작위 제거와 개수를 맞춘 무작위 제거다.
- 추가 null은 가치 추정기의 seed noise다.
- probe와 입력 복사 대조는 해당하지 않는다.
- 모든 비교는 같은 seed의 짝 비교로 한다.

## 6. 실행 계획과 축소 규칙

### 6.1 스모크 후 바꾼 것 (본 실행 전, 03:20Z)

스모크(walker-walk와 mw-push, 각 2 episode, α ∈ {0.1, 0.3})의 수치를 보고 바꾼 것을 투명하게 적는다. 판정 기준값은 그대로다.

1. **S3의 1차 후보 집합: 마지막 iteration → 첫 iteration**
   - 마지막 iteration 집합은 후보가 평균 근처에 몰려 있어 후보 간 가치 분산이 매우 작다.
   - 그래서 argmax가 포화되었다. 모델을 바꾸지 않고 seed만 바꾼 seednoise 대조도 argmax 뒤집힘이 70~100%였다(사분위 무관). 같은 seed를 쓴 weak α=0.1도 거의 100%였다.
   - 이 집합에서는 "near-tie 국소화"를 판별할 수 없다. 측정 타당성 문제다.
   - 첫 iteration 집합에서는 seednoise 뒤집힘이 gap에 따라 달라졌다(Q1 0.9 → Q4 0.3, n=40).
   - 마지막 iteration 결과는 2차로 그대로 보고한다.
2. **추가 지표:** 공통 이동을 뺀 정규화 가치 오차(value_nerr_centered).
3. **속도와 그에 따른 조정**
   - CUDA MPS를 켰다. 7개 프로세스 처리량이 94 → 131 step/s로 늘었다.
   - 판정에 쓰지 않는 strong과 rand_count는 5 episode로 줄였다.
4. **스모크에서 이미 본 수치**(본 실행에서 다시 잰다)
   - walker-walk α=0.1: 제거 비율 0.35, err1 0.062
   - walker-walk α=0.3: 제거 비율 0.71, err1 0.40
   - 이 값들은 S1 세부 규칙을 정한 뒤에 보았다. 규칙은 바꾸지 않았다.

### 6.2 본 실행 순서

- 스모크(03:08Z~): walker-walk 2 episode, α ∈ {0.1, 0.3}, 전체 파이프라인. mw-push도 같이 확인했다.
  - 스모크에서 확인한 속도: 프로세스 7개 동시 실행 시 DMC episode당 40~58초.
  - 이에 따라 strong과 rand_count는 5 episode로 줄였다(위 4.2).
- 본 실행: calib(과제당 10 episode) → prep → grid → decision(500 상태, 전체 계획 재실행 250 상태) → analyze
- 시간이 부족하면 이 순서로 줄인다.
  1. strong을 α ∈ {0.1, 0.3}만 돌린다.
  2. 결정 수준 상태를 500 → 300, 전체 계획 재실행을 250 → 100으로 줄인다.
  3. rand_pred의 α = 0.05를 뺀다.
  - weak, rand_pred(α ≥ 0.1), none은 10 episode를 유지한다.
- 목표: 05:30Z 이전에 결과 회수와 pod 삭제. 예상 비용은 2.5h × $0.27 ≈ $0.7이다.

## 7. 사전등록 수정 1 — 조정자 요청 (2026-10-21T03:21Z, Q 조건 실행 전)

**작성 시점에 본 것**
- 본 실행의 WM 범위 예측 쪽 수치(prep, 4과제)
- walker-walk ep0 계획 episode 약 15개

**작성 시점에 보지 않은 것**
- Q 제거 조건은 아직 아무것도 돌리지 않았다.
- 새 결정 지표도 아직 계산하지 않았다.
- 따라서 아래 항목은 Q 조건과 새 지표에 대해 사전등록이다.
- 원래 판정(S1~S3, WM 범위)은 바꾸지 않는다.

### 7.1 제거 범위(scope) 3가지
| scope | 제거 대상 층 |
|---|---|
| WM(원래, 1차 판정) | dyn0, dyn1, rew0, rew1 |
| Q | Q-ensemble 5개 멤버의 은닉 2층(멤버마다 NormedLinear(Linear→LayerNorm→Mish) 2개, 512 unit) |
| ALL | WM + Q |

- **Q hook 구현:** 공식 Ensemble은 `torch.vmap`으로 멤버를 한 번에 계산한다. hook에서 멤버별 m_j를 쓰려고, Q 범위를 다루는 프로세스에서는 Ensemble forward를 같은 파라미터로 멤버를 하나씩 도는 loop로 바꾼다(수치는 같다, float 반올림 수준).
- **Q의 m_j:** WM과 같은 방식으로 구한다. 계획 중 Q는 horizon 끝 latent z_3와 policy 행동 π(z_3)를 입력으로 받는다. calibration split의 후보 시퀀스를 원래 모델로 rollout해 z_3를 만들고, π는 고정 seed로 샘플한다. 그 입력에서 멤버별·unit별 |a|의 99.9 백분위를 쓴다.
- **Q 예측 오차 errq:** held-out 계획 시점 (z_3, π(z_3))에서 멤버별 two-hot 복원 Q의 E(Q_abl − Q)² / Var(Q)를 구하고 5개 멤버를 평균한다.
- **rand_pred의 r_q:** errq를 weak와 맞추도록 따로 이분 탐색한다. ALL 범위의 rand_pred는 (r_dyn, r_rew, r_q)를 함께 쓴다.
- **조건:** Q와 ALL 범위 각각 weak, rand_pred × α ∈ {0.1, 0.2, 0.3} × 10 episode(seed 1000~1009, 비제거 기준선은 WM과 공통)
  - α 0.05와 0.5는 시간 때문에 계획 실험에서 뺀다. 예측 쪽은 5개 α 모두 잰다.
  - rand_count와 strong은 Q와 ALL 범위에서 예측 쪽만 잰다.
- **범위별 판정(추가 보고, 원래 판정과 별도)**
  - S1_scope: α ≥ 0.1인 어떤 α에서 그 범위 층의 제거 비율 ≥ 50%이고 그 범위의 예측 오차 ≤ 0.10
    - Q 범위의 예측 오차는 errq다.
    - ALL 범위는 err1 ≤ 0.10과 errq ≤ 0.10을 둘 다 요구한다.
  - S2_scope: §5.1의 S2 규칙과 같다. α*_scope는 {0.1, 0.2, 0.3} 중 S1_scope를 만족하는 가장 큰 α다.
  - S3_scope: §5.1의 S3 규칙과 같다(첫 iteration 집합).

### 7.2 결정 수준 지표 2개 추가 (모든 범위, 두 후보 집합)
1. **elite regret** = V_orig(원래 모델의 최고 후보) − V_orig(제거 모델이 고른 argmax 후보)
   - 원래 모델 가치로 잰다. 상태별 원래 가치의 후보 간 표준편차로 나눈 값도 보고한다.
   - 대조는 seednoise와 rand_pred다.
2. **MPPI elite 가중치 분포 변화**
   - 각 모델이 자기 가치로 고른 top-64 elite에 MPPI 가중치 softmax(temperature × (V − max V))를 매긴다. temperature는 0.5다.
   - 그 가중치의 엔트로피 H를 원래 모델과 제거 모델에서 각각 구하고, 차이 ΔH = H_abl − H_orig를 본다.
   - 보조 지표로 두 가중치 분포의 겹침 Σ_i min(w0_i, w1_i)도 본다(512 후보 위, elite 밖은 0).
- 이 지표는 판정에 넣지 않는 보조 지표다. weak와 rand_pred를 같은 예측 오차에서 비교하는 데 쓴다.

### 7.3 시간 순서
1. WM grid
2. decision(모든 범위)
3. analyze(WM)
4. Q·ALL grid
5. analyze(전체)

시간이 모자라면 Q·ALL grid의 episode를 10 → 5로 줄이고 그 사실을 적는다.
