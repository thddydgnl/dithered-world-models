# R2-8b Common-random-number(CRN) dither — 사전등록 계획 (PLAN)

- 작성: 2026-10-21T07:33Z, **본 실행(grid) 전**.
  - 이미 실행한 것: pod 설치, 비양자화 calibration rollout(seed 1000~1009), prep(양자화 범위 확인, 예측 오차).
  - 스모크 결과는 본 실행 전에 §8에 덧붙인다.
- 상위: [../IDEAS_ROUND2.md](../../../archive/selection_process/IDEAS_ROUND2.md) R2-8b
- 선행: R2-8 [../R2_8_dithered_world_models/REPORT.md](../R2_8_dithered_world_models/REPORT.md), R2-3b [../R2_3_weak_planning_bias/followup_3b/REPORT_3b.md](../R2_3_weak_planning_bias/followup_3b/REPORT_3b.md)
- 규칙: [../runpod_infra/PILOT_RULES.md](../../../runpod_infra/PILOT_RULES.md). 이 pilot의 판정 기준 출처는 IDEAS_ROUND2.md다.
- pod `pai-r2n8b`
  - RTX A5000을 확보하지 못해 **RTX 3090($0.50/h)**을 썼다. id `4v97c6d7vijrmz`, 07:18:33Z 생성
  - CPU 할당은 cgroup 기준 31코어다
  - 상한 $1.5, 목표 $1.2 이내

## 1. 배경
- R2-8의 사전등록 판정은 없음이었다. 그러나 대조로 넣은 공통 dither(sh)는 결정적 반올림(det)과 독립 dither(ind)보다 덜 해로웠다.
  - sh는 그 forward의 모든 후보가 같은 잡음을 공유하고, 매 forward 새로 뽑는다.
  - mw-push k=3의 성공률: det 30%, ind 50%, sh 90%(10 episode)
- 사후 해석: 해로운 오차는 두 종류다.
  - (i) 시간적으로 지속되는 편향
  - (ii) 후보마다 독립인 잡음
- 후보 간에는 같고 시간적으로 새로운 잡음은 Monte Carlo의 공통 난수(CRN)와 같은 구조다.
- 이 계획은 공유 범위를 바꿔 그 사후 해석을 사전등록 검정한다.

## 2. 가설 (IDEAS_ROUND2.md 그대로)
- **H1:** 공통·새 잡음 dither(sh-fwd, sh-step)가 같은 비트 수의 결정적 반올림보다 계획 손상을 크게 줄인다.
- **H2:** 공통 dither가 독립 dither보다 덜 해롭다(CRN 효과).
- **H3:** 공통이지만 episode 내내 고정된 잡음(sh-ep)은 지속 편향이 되어 결정적 반올림만큼 해롭다. 즉 시간적 새로움이 필요하다.

## 3. 설정

### 3.1 R2-8과 같은 것
- TD-MPC2 공식 5M 단일 과제 체크포인트
- MPPI 기본값: 후보 512개(그중 policy prior 24개), elite 64개, iteration 6회, horizon 3, temperature 0.5, eval_mode
- 양자화 대상: dynamics·reward MLP의 은닉 활성 4개 층(Mish 이후)
- 층별 균일 k-bit 양자화. 간격 Δ = (hi − lo)/(2^k − 1), [lo, hi] 밖은 잘라낸다.
- **양자화 범위는 R2-8의 층별 [lo, hi] 값을 그대로 쓴다**(`inputs/r28/prep_<task>.json`). 그래서 k-bit 격자가 R2-8과 같다.
  - 이 pod의 calibration에서 같은 절차로 다시 잰 범위는 비교용으로만 기록한다.
- 코드: R2-8의 `nt_lib`, calib, decision을 복사했다. 양자화 hook에 변형 2개를 더했고, grid·prep·analyze는 R2-8b용으로 고쳤다.

### 3.2 칸
과제와 비트 수는 IDEAS 그대로다.
- mw-push k=3
- mw-pick-place k=3
- cheetah-run k=4
- walker-walk k=4

### 3.3 반올림 변형
x는 [lo, hi]로 자른 뒤 격자 단위로 바꾼 값이다. det만 round(x)이고, 나머지는 모두 확률적 반올림 floor(x + u), u ~ U[0, 1)이다. 변형끼리는 u를 공유하는 범위만 다르다.

| 변형 | 후보 간 | 시간 | u를 뽑는 방식 |
|---|---|---|---|
| det | 같은 입력이면 같은 오차 | 결정적 | 없음. round(x) |
| ind | 독립 | 매 forward 새로 | 항목(후보 × unit)마다 독립. R2-8의 (b) |
| sh-fwd | 공유 | 매 forward 새로 | unit마다 하나, 그 forward의 모든 행(후보)이 공유. R2-8의 (c) `sh`와 같은 코드 |
| sh-step | 공유 | 제어 step 안에서 고정, step마다 새로 | 층마다 (horizon 위치 h × unit) 표 U를 제어 step 시작에 뽑는다. 그 step의 MPPI iteration 6회와 policy prior rollout이 horizon 위치 h에서 같은 U[h]를 쓴다 |
| sh-ep | 공유 | episode 내내 고정 | sh-step과 같은 표를 episode 시작에 한 번만 뽑는다 |

- **표를 horizon 위치마다 따로 두는 이유:** sh-fwd에서는 horizon step마다 forward가 따로라서 잡음이 서로 독립이다. 표를 (h × unit)으로 두면 변형끼리 한 가지씩만 다르다.
  - sh-step은 sh-fwd와 "iteration 사이의 새로움"만 다르다.
  - sh-ep은 sh-step과 "제어 step 사이의 새로움"만 다르다.
- **horizon 위치를 정하는 방법:** 행 수별 forward 호출 횟수를 3으로 나눈 나머지를 쓴다.
  - MPPI 호출(512행)은 iteration마다 h = 0, 1, 2 순서다.
  - policy prior rollout(24행)은 dynamics만 h = 0, 1 순서로 부른다.
  - 스모크에서 제어 step당 호출 수를 확인한다. 기대값은 512행이 층마다 18회, 24행이 dynamics 층마다 2회다.
  - 표 갱신 횟수도 확인한다. 기대값은 sh-step이 step 수 + 1, sh-ep이 1이다.
- **오차 분포:** ind, sh-fwd, sh-step, sh-ep는 항목별 오차의 주변분포가 같다.
  - 모두 평균 0인 확률적 반올림이고, 평균 MSE는 Δ²/6이다(det는 Δ²/12).
  - 차이는 후보 간 상관과 시간 상관뿐이다.
  - sh-ep는 unit마다 u가 고정되므로, 그 episode 동안 unit별 지속 편향(평균 (u − ½)Δ)을 가진다.
- **난수:** 무작위 요소는 별도 CUDA generator를 쓴다. seed는 7 × episode seed + 13이다. 그래서 planner의 난수열은 모든 조건에서 같다.

### 3.4 episode와 seed
- **본 실행 seed는 2000~2019**다. 조건마다 20 episode이고, 모든 조건과 기준선이 같은 seed로 짝을 이룬다.
  - **R2-8의 seed 1000~1009와 겹치지 않게 했다.** R2-8의 사후 결과를 만든 episode를 확증에 다시 쓰지 않기 위해서다.
  - 스모크 seed는 9000~9001이다.
- **기준선:** 양자화하지 않은 같은 seed의 episode를 이 pod에서 돌린다. GPU가 다르면 episode가 비트 단위로 같지 않기 때문이다(R2-3b).
- **합계:** 4칸 × (기준선 1 + 변형 5) × 20 = **480 episode**
- **calibration:** 과제마다 비양자화 10 episode(seed 1000~1009, R2-8과 같다)
  - 0~4: 범위 재측정(비교용)
  - 5~9: held-out 6만 시퀀스(예측 오차)
  - 10 episode 전체: 결정 상태(계획 편차)

## 4. 측정
- **계획 하락(1차)**
  - 같은 seed 기준선의 평균 R0 대비 상대 하락 (R0 − R)/R0이다.
  - DMC는 return, MW는 마지막 step의 성공(성공률)을 쓴다. R0 = 1이면 MW 하락은 %p와 같다.
  - MW return 하락은 보조로 보고한다.
- **짝 bootstrap 90% CI**
  - episode(seed) 단위로 재표집한다(4000회, percentile 5~95).
  - 기준선 대비 하락은 episode마다 (기준선_i − 조건_i)/R0를 계산해 평균의 CI를 구한다.
  - 조건 쌍 차이(예: D_sh-fwd − D_ind)는 episode마다 하락 차를 계산해 평균의 CI를 구한다.
- **하락 감소율:** red_v = (D_det − D_v)/D_det. 같은 episode 재표집에서 비율을 계산해 CI를 구한다.
- **예측 오차:** held-out 계획 시점 입력에서 원 모델 대비 정규화 오차를 잰다. dither seed 3개의 평균이다.
  - err1: 1-step latent
  - err3: 3-step 열린 rollout
  - errr: 1-step reward
- **최종 계획 편차**
  - calibration 결정 상태 150개에서 저장된 obs와 이전 mean으로 `_plan`을 같은 torch seed로 다시 실행한다.
  - 최종 MPPI mean 첫 step의 RMS 편차를 seed noise 바닥으로 나눈다. seed noise 바닥은 같은 비양자화 모델에 seed만 다르게 준 값이다.
- **보조(결정 수준):** 150 상태, 첫 iteration의 고정 후보 512개에서 잰다.
  - CVB(elite 가중 평균 첫 행동의 이동 / seed 바닥)
  - Spearman 순위 상관, argmax flip
- **측정의 한계(미리 적음)**
  - 가치 평가 한 번에서는 sh-fwd, sh-step, sh-ep가 분포상 같다. 셋 다 horizon 위치마다 독립인 공유 잡음이기 때문이다. 그래서 결정 수준 지표는 det, ind, 공유 변형 셋의 세 무리만 가른다.
  - 제어 step 하나의 계획 재실행에서는 sh-step과 sh-ep가 분포상 같다. 그래서 최종 계획 편차는 sh-fwd와 (sh-step, sh-ep)만 가른다.
  - sh-step과 sh-ep의 차이는 closed-loop episode에서만 드러난다.

## 5. 사전등록 판정 기준 (IDEAS_ROUND2.md 그대로)
- **S1:** sh-fwd 또는 sh-step이 det 대비 하락을 50% 이상 줄이는 과제가 2개 이상이다(90% CI 보고).
- **S2:** sh-fwd가 ind보다 유의하게 덜 해로운 과제가 4개 중 3개 이상이다(짝 bootstrap 90% CI가 0 미만).
- **S3:** sh-ep가 det보다 유의하게 덜 해롭지 않은 과제가 2개 이상이다(시간적 새로움의 필요성).
- **판정**
  - S1과 S2를 만족하면 있음(CRN dither 처방)
  - S3까지 만족하면 강한 있음(기전: 지속 편향 + 독립 잡음 두 성분)
  - 그 외는 조건부 또는 없음

### 5.1 실행 전에 채운 세부 규칙
1. **공통**
   - 하락은 §4의 1차 지표다(MW 성공률, DMC return).
   - "유의"는 episode 짝 bootstrap 90% CI가 0을 포함하지 않는다는 뜻이다.
   - "det의 하락이 실재한다"는 기준선 대비 det 하락의 짝 bootstrap 90% CI 하한 > 0이라는 뜻이다.
2. **S1**
   - 과제마다 v ∈ {sh-fwd, sh-step}에 대해 red_v = (D_det − D_v)/D_det를 점추정으로 계산한다.
   - det의 하락이 실재하고 max(red_sh-fwd, red_sh-step) ≥ 0.5이면 그 과제는 성립이다. 2과제 이상이면 S1 성립이다.
   - det의 하락이 실재하지 않는 과제는 나눌 하락이 없으므로 불성립으로 센다.
   - red의 bootstrap 90% CI와 (D_det − D_v)의 짝 CI를 함께 보고한다.
   - 엄격 보조판(판정에 쓰지 않음): red ≥ 0.5이면서 (D_det − D_v)의 CI 하한 > 0
3. **S2**
   - (D_sh-fwd − D_ind)의 짝 bootstrap 90% CI 상한 < 0이면 그 과제는 성립이다.
   - 4과제 중 3개 이상이면 S2 성립이다.
4. **S3**
   - det의 하락이 실재하고, (D_sh-ep − D_det)의 짝 CI 상한 ≥ 0이면 그 과제는 성립이다. 후자는 sh-ep가 det보다 유의하게 덜 해롭지 않다는 뜻이다.
   - 2과제 이상이면 S3 성립이다.
   - det의 하락이 실재하지 않으면 "유의하게 덜 해롭지 않음"이 자명하게 성립하므로 세지 않는다(보수적 채움). 이 조건 없이 문구 그대로 센 개수도 보고한다.
   - 보조(판정에 쓰지 않음)
     - sh-ep가 sh-fwd보다 유의하게 해로운 과제 수(D_sh-ep − D_sh-fwd의 CI 하한 > 0)
     - sh-step 대 sh-fwd, sh-ep 대 sh-step
5. **판정표의 "그 외"**
   - S1과 S2 중 하나만 성립: 조건부 있음(S3 결과는 기전 정보로 함께 보고)
   - S1과 S2 모두 불성립: 없음(S3가 성립해도)
6. **MW 기준선 성공률이 1보다 낮을 때:** MW 하락은 R0 대비 상대값이다. S1의 비율은 R0와 무관하고, S2·S3의 부호도 R0와 무관하다.

## 6. 참고 예측 (R2-8, 10 episode, seed 1000~1009)

| 칸 | det | ind | sh(= sh-fwd) | 예상 |
|---|---|---|---|---|
| mw-push k=3 | 70%p | 50 | 10 | S1·S2 성립 |
| mw-pick-place k=3 | 60%p | 40 | 30 | S1 경계(감소율 50%), S2 불확실(−10%p, 유의하지 않음) |
| cheetah-run k=4 | 52.9% | 48.1 | 35.0 | S1 불성립(34%), S2 성립(−13.1%p, 유의) |
| walker-walk k=4 | 15.5% | 29.3 | 7.5 | S1 경계(52%)이고 det 하락이 실재하는지가 관건, S2 성립(−21.8%p, 유의) |

sh-step과 sh-ep는 R2-8에 없던 변형이라 예측값이 없다.

## 7. 실행 순서와 예산
1. calibration(40 episode)과 prep(이미 실행)
2. **스모크:** mw-push k=3, 기준선과 5변형 × 2 episode(seed 9000~9001). forward 호출 패턴과 표 갱신을 기록하고, decision을 10 상태에서 돌린다.
3. **본 실행:** grid 480 episode(워커 16개, CUDA MPS) → decision(과제마다 150 상태, 5변형) → analyze_r28b
4. 결과를 회수하고 pod를 삭제한다.
- 예상 시간: grid 15~25분, pod 가동 1.3시간 이내(약 $0.7)

## 8. 스모크 결과 (본 실행 전에 기록, 07:38Z)
- **calibration·prep(07:29~07:31Z)**
  - 이 pod에서 다시 잰 양자화 범위가 R2-8과 소수 셋째 자리까지 같았다. det의 예측 오차도 R2-8과 같았다(mw-push k=3 err1 0.8241).
  - 이 3090이 R2-8의 A40과 같은 수치를 낸 것으로 보인다. 그래서 seed 1000~1009를 다시 쓰면 R2-8 episode를 그대로 재현했을 가능성이 크다. 본 실행 seed를 2000~2019로 둔 결정(§3.4)이 확증에 필요했다.
  - 공유 변형의 예측 오차는 dither 표 한 번의 추출에 따라 크게 흔들린다(seed 간 sd: err3 최대 0.26, errr 최대 1.1). 그래서 보고용 오차는 본 실행과 동시에 dither seed 20개로 다시 잰다. 판정에는 쓰지 않는다.
- **스모크(mw-push k=3, seed 9000~9001, 12 episode, 07:35Z)**: 끝까지 돌았고 구현이 의도대로 동작했다.
  - 제어 step당 forward 호출 수: 모든 변형, 모든 step에서 한 가지 패턴만 나왔다. 512행은 층마다 18회, 24행(policy prior)은 dynamics 층마다 2회로 기대와 같다.
  - 표 갱신 횟수: sh-step은 101회(100 step + 1), sh-ep은 1회
  - 표 서명(층 합): sh-step은 step마다 바뀌고, sh-ep은 episode 안에서 같고 episode끼리는 다르다.
  - decision(10 상태): sh-step과 sh-ep의 값이 같았다(§4의 예상대로 한 step 재실행에서는 같은 표). 5변형 모두 CSV가 나왔다.
  - 속도: MW episode 6~7초(워커 12개 동시)
- **스모크 결과는 기능 점검에만 쓰고 해석하지 않는다.** 본 실행 seed와 겹치지 않는다.
- **본 실행 설정:** 워커 20개. CPU 31코어라서 R2-8의 7개보다 늘렸다. 결과에는 영향이 없고 속도만 다르다.
