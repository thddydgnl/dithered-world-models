# R2-13 사전등록 계획: CRN dither for a pixel planning world model (DINO-WM CEM)

- 작성: 2026-10-21 08:35Z(pilot worker `r2n13`). k 선택 결과(§6)는 k 스모크 뒤, 본 실행 전에 채운다.
- pod: `pai-r2n13`, **NVIDIA GeForce RTX 4090 24GB, $0.89/h**, 생성 08:31Z(자동 삭제 12:31Z)
- 비용: 상한 $2, 목표 $1.6 이하(약 1시간 48분 가동)
- 이 문서를 쓰기 전에 본 것
  - R2-7 결과: 같은 eval 분할(pair seed 4)의 기본 CEM 성공률 PushT 74%, Wall 48%(A40). 후반 CEM 모집단의 순위 상관 0.53(PushT), 0.51(Wall)
  - R2-8b 결과: MPPI(TD-MPC2)에서 sh-fwd가 det 손상의 82~92%(MW) 제거, sh-step·sh-ep는 det 이하
  - 이 파일럿의 양자화 결과(오차, 성공률)는 아직 아무것도 보지 않았다.

## 1. 질문과 가설 (IDEAS_ROUND2.md R2-13에서 그대로 옮김)

- H1: DINO-WM 예측기 MLP 은닉 활성(GELU 이후)을 k-bit로 결정적 양자화하면 CEM 성공률이 떨어진다. 반복마다 새로 뽑는 공통 dither(sh-iter)는 그 손상의 절반 이상을 없앤다.
- H2: 공통이지만 한 계획 내내 고정한 dither(sh-fixed)는 결정적 반올림보다 낫지 않다(새로움의 필요성).
- H3: 후보 독립 dither(ind)는 sh-iter보다 해롭다.
- H4: 1-step 예측 오차는 변형 간 손상 차이를 설명하지 못한다.

## 2. 플랫폼 (R2-1·R2-7 코드 재사용, 복사본은 `code/r21lib.py`)

- DINO-WM 공개 체크포인트 `pusht`, `wall_single`. 인코더는 frozen DINOv2 ViT-S/14(patch 196 × 384)
- 예측기: ViT 6 block, MLP 은닉 2048(GELU). PushT num_hist 3, Wall 1
- 계획 비용: DINO-WM `objective_fn_last` = 시각 MSE + α·proprio 임베딩 MSE(α=1)
- CEM: DINO-WM `CEMPlanner`와 같은 의미. 300 samples, top-30, var_scale 1, 첫 후보 = mu, goal_H 5
  - opt_steps: **PushT 30, Wall 10**(과제 설정값. Wall의 sh-fixed는 10 iteration 내내 고정)
  - open-loop(= MPC 1회 반복). 설정의 MPC max_iter=null은 끝나지 않는다(R2-1 편차 5와 같음)
- 월드모델은 bf16 autocast, 배치 구현(R2-7 `rollout_batched`와 같은 계산에 양자화 hook만 추가)
- 참 비용: 같은 시작 상태에서 시뮬레이터 재생(R2-1의 빠른 재생). 성공은 평가기 `eval_state`
  - 연속 지표 margin(R2-7과 같음): PushT max(‖Δ[agent_xy, block_xy]‖/20, Δangle/(π/9)), Wall ‖Δxy‖/4.5

## 3. 분할과 seed (R2-7과 같은 정의)

| 분할 | 시작·목표 쌍 seed | episode | planner seed | 용도 |
|---|---|---|---|---|
| held | 3 | 과제당 20 | 1000+e | k 선택 스모크 |
| eval | 4 | 과제당 50 | 2000+e | 본 실행(판정) |

- 모든 변형은 같은 episode, 같은 planner seed(CPU generator)로 돈다. 후보 표본 난수열이 변형 간에 같으므로 짝 비교다.
- dither 난수는 별도 CUDA generator에서 나온다. 계획기의 난수열을 건드리지 않는다.
- 모든 변형은 이 pod 하나에서, 과제마다 같은 배치 크기 E로 돈다(bf16 배치 수치 차이 통제). `none`도 이 pod에서 다시 돈다.
- 본 실행은 episode 블록(E개) 단위로 5변형을 차례로 돈다. 중간에 멈춰도 모든 변형이 같은 episode를 갖는다.

## 4. 양자화 정의

- 위치: 예측기 6 block 모두의 MLP 은닉 활성(GELU 출력, `blk[1].net[2]`). attention, 인코더, proprio·action 인코더는 건드리지 않는다.
- 격자: 층별(per-tensor) k-bit 균일 격자 [lo, hi], 간격 Δ = (hi − lo)/(2^k − 1). x = (clamp(a, lo, hi) − lo)/Δ
- 층별 scale 보정(데이터셋 latent)
  - train 분할 궤적 64개 구간을 고른다. 데이터셋 상태에서 데이터셋 행동으로 시뮬레이터가 렌더링한 프레임(frameskip 5)을 encode한다.
    - 데이터셋의 저장 영상은 풀지 않았다(디스크, R2-1·R2-7과 같음). 렌더러는 계획에서 쓰는 것과 같다.
  - num_hist 프레임 창에 예측기를 bf16으로 한 번 돌려 GELU 출력을 모은다.
  - lo = 최솟값, hi = 99.99 분위수(R2-8b와 같은 규칙)
- 반올림 변형 5개. 같은 격자를 쓰고 잡음 공유 범위만 다르다.
  - **none**: 양자화 없음
  - **det**: 결정적 반올림 round(x)
  - **ind**: 확률적 반올림 floor(x + u). u는 항목(후보 × token × unit)마다 독립이고 매 forward 새로 뽑는다.
  - **sh-iter**: u는 한 episode의 300후보가 공유한다(token·unit마다는 독립). 매 forward 호출마다 새로 뽑는다. 즉 CEM iteration마다 새롭다(한 iteration 안의 horizon step 5개도 서로 다른 u).
  - **sh-fixed**: u는 300후보가 공유한다. (층, horizon step)마다 표 하나를 계획 시작 때 뽑고 그 계획의 모든 iteration(PushT 30, Wall 10)에서 그대로 쓴다.
- 한 번의 forward에서 ind·sh-iter·sh-fixed의 항목별 오차 분포는 같다. 차이는 후보 간 공유(CRN)와 iteration 간 지속뿐이다.

## 5. 측정

1. **CEM 성공률**(1차): eval 50 episode, 변형별. 하락 D_v = 성공률(none) − 성공률(v) (%p)
2. margin(2차): 최종 계획의 연속 지표. 짝 Wilcoxon과 중앙값 보고
3. **1-step 예측 오차**(양자화하지 않은 예측 대비, 계획 비용 단위 = 시각 MSE + α·proprio MSE)
   - 1차(S4용) ds_err1: valid 분할 데이터셋 창 64개(num_hist 프레임)에서 다음 프레임 예측 한 번. 확률적 변형은 dither seed 3개 평균
   - 2차: ds_err5(데이터셋 행동 5-step open-loop 마지막 프레임), plan_err1·plan_err5(held 시작 프레임 + CEM 0단계 분포의 후보 300개)
   - 모두 양자화하지 않은 예측의 분산으로 나눈 상대값도 보고
4. **마지막 iteration 후보 순위 상관**: episode마다 마지막 iteration의 후보 300개를 시뮬레이터로 재생해 Spearman(J_v, margin)을 구하고 episode 평균을 낸다.
   - 같은 모집단에서 양자화하지 않은 비용 J_none의 Spearman도 계산한다(모집단 차이와 비용 오차를 분리).
   - 0단계 모집단은 모든 변형에서 같으므로(같은 generator, mu=0, sigma=1) none에서 한 번 재생해 0단계 Spearman(J_v, margin)을 같이 보고한다.
5. 통계
   - 짝 성공 비교: McNemar exact(불일치 쌍의 이항검정, 양측)
   - 하락 차와 감소율 r = (D_det − D_sh-iter)/D_det: episode 짝 bootstrap(10,000회) 95% CI
   - 성공률: Wilson 95% CI

## 6. k 선택 (본 실행 전에 결과를 여기에 적는다)

- 규칙(스모크 전에 정함)
  1. det의 데이터셋 1-step 오차를 k ∈ {2, 3, 4, 5, 6, 8}에서 잰다. 상대 오차가 너무 작거나(det가 무해할 것) 너무 큰(완전 붕괴) k를 빼고 후보 2~3개를 고른다.
  2. held 분할 20 episode에서 none과 후보 k의 det를 CEM으로 돌린다(같은 seed).
  3. 과제마다 held 하락이 **20%p 이상인 가장 큰 k**를 고른다. 20%p에 이르는 k가 없으면 15%p 이상인 가장 큰 k, 그것도 없으면 시험한 가장 작은 k를 고른다.
     - 15%p 기준보다 여유를 둔 이유: held 20 episode로 고르면 eval에서 평균으로의 회귀가 생긴다.
  4. 과제마다 k를 따로 정할 수 있다(판정 기준이 과제·비트 수 칸 단위).
- 선택 결과
  - 1단계(08:37Z, det의 상대 오차; ds = 데이터셋 1-step, plan = 계획 시점 5-step)

    | k | PushT ds_err1 | PushT plan_err5 | Wall ds_err1 | Wall plan_err5 |
    |---|---|---|---|---|
    | 2 | 0.263 | 3.24 | 0.132 | 8.70 |
    | 3 | 0.033 | 0.51 | 0.057 | 3.09 |
    | 4 | 0.0056 | 0.22 | 0.0091 | 0.15 |
    | 5 | 0.0010 | 0.014 | 0.0020 | 0.052 |
    | 6, 8 | ≤ 0.0004 | ≤ 0.015 | ≤ 0.0004 | ≤ 0.004 |

    - k=2는 계획 시점 오차가 예측 분산의 3~9배라 붕괴로 보고 뺐다. k ≥ 5는 상대 오차가 1~5% 이하라 무해할 것으로 보고 뺐다. **후보는 k ∈ {3, 4}**
  - 기능 스모크(held 앞 2~5 episode, 5변형, k=4)를 먼저 돌렸다. 판정에 쓰지 않는다.
    - PushT 2 episode: none 2/2, det 0/2, ind·sh-iter·sh-fixed 2/2
    - Wall 5 episode: none 2/5, det 4/5, ind 2/5, sh-iter 2/5, sh-fixed 3/5
  - 2단계 **Wall**(held 20 episode, 08:44Z): none 11/20(55%), det k=3 3/20(15%, **−40%p**), det k=4 15/20(75%, +20%p)
    - → 규칙에 따라 **Wall k = 3**(20%p 이상 하락하는 가장 큰 k)
  - 2단계 **PushT**: 큰 후보부터 시험한다(규칙이 "가장 큰 k"이므로 같은 결론). none과 det k=4를 held 20 episode에서 먼저 돌린다.
    - det k=4의 하락이 20%p 이상이면 k=4로 정하고 k=3은 돌리지 않는다. 아니면 det k=3을 같은 episode에서 돌려 규칙을 적용한다.
    - PushT 결과(held 20 episode, 09:05Z): none 17/20(85%), det k=4 8/20(40%, **−45%p**)
    - → 규칙에 따라 **PushT k = 4**. k=3은 돌리지 않았다(k=4가 이미 20%p 이상).
    - 기록 순서: 규칙 적용 09:05:10Z → PushT 본 실행 시작 09:05:29Z → 이 줄 기입 09:06Z. PushT 본 실행 결과는 아직 하나도 나오지 않은 시점이다.
  - **최종: PushT k = 4, Wall k = 3.** 배치 크기 E: PushT 2, Wall 4(본 실행의 모든 변형에 같음)

## 7. 사전등록 판정 기준 (IDEAS_ROUND2.md R2-13에서 그대로 옮김)

- S1: det가 기준 대비 15%p 이상 떨어뜨리는 과제·비트 수에서, sh-iter가 그 하락을 50% 이상 줄인다(과제 하나 이상). 다른 과제를 기준 대비 5%p 넘게 떨어뜨리지 않는다.
- S2: sh-fixed가 det보다 유의하게 낫지 않은 과제가 하나 이상이다.
- S3 (권장): sh-iter가 ind보다 하락이 작은 과제가 하나 이상이다.
- S4 (권장): 변형들의 1-step 오차 차이가 5% 이내인데 하락이 2배 이상 갈리는 사례가 하나 이상이다.
- 판정
  - S1과 S2를 만족하면 있음: A축 처방이 두 번째 계획기와 픽셀 월드모델에서도 성립
  - S1만 만족하면 조건부
  - S1이 실패하면 없음

### 7.1 기준을 채운 세부 규칙 (주어진 문턱은 바꾸지 않음)

1. **해당 과제(qualifying):** 본 실행 eval 50 episode에서 D_det ≥ 15%p인 과제. k는 §6에서 과제마다 하나다.
2. **S1**
   - 해당 과제 하나 이상에서 r = (D_det − D_sh-iter)/D_det ≥ 0.5(점추정)
   - 그리고 나머지 과제 각각에서 다음을 만족한다.
     - 해당하지 않는 과제(D_det < 15%p)이면 D_sh-iter ≤ 5%p(문구 그대로)
     - 그 과제도 해당 과제이면 D_sh-iter ≤ D_det(처방이 det보다 더 해치지 않음)
     - 이유: 두 과제가 모두 해당하는 경우, 문구를 그대로 적용하면 두 번째 과제에서 det 하락의 절반 이상을 줄여도 5%p를 넘으면 실패가 된다. 이것은 기준의 취지(처방이 다른 과제를 해치지 않음)와 맞지 않는다.
   - 보조 보고: 문구 그대로의 엄격판(다른 과제 D_sh-iter ≤ 5%p 무조건)과 r의 bootstrap 95% CI
   - 해당 과제가 없으면(두 과제 모두 D_det < 15%p) S1은 성립할 수 없다. 규칙상 "없음"이지만 "k 선택 실패로 검정 불가"를 함께 적는다.
3. **S2**
   - "유의하게 낫다" = McNemar exact 양측 p < 0.05이고 sh-fixed 쪽 성공이 많다.
   - 해당 과제 중 sh-fixed가 det보다 유의하게 낫지 않은 과제가 하나 이상이면 성립
   - det가 무해한 과제에서는 "낫지 않음"이 자명하므로 세지 않는다(보수적 채움). 문구 그대로(모든 과제) 센 결과도 보고
   - 보조: sh-fixed의 감소율 r_fixed와 sh-iter 대비 McNemar
4. **S3:** D_sh-iter < D_ind(점추정)인 과제가 하나 이상이면 성립. McNemar p를 같이 보고
5. **S4**
   - 같은 과제의 두 변형 a, b에서 |e_a − e_b| / min(e_a, e_b) ≤ 0.05(e = ds_err1, 1차)
   - 그리고 max(D_a, D_b) ≥ 2 · max(min(D_a, D_b), 0)이고 max(D_a, D_b) ≥ 10%p
     - 10%p 하한은 잡음 수준의 하락 쌍(예: 2%p 대 0%p)을 세지 않으려고 둔 값이다.
   - 이런 쌍이 하나 이상이면 성립. plan_err1로 센 결과도 보고
6. **판정:** 위 문구대로(S1·S2 → 있음, S1만 → 조건부, S1 실패 → 없음). S3·S4는 기전 정보로 같이 보고

## 8. 편차와 운영 규칙

- Wall opt_steps는 10이다(과제 설정값). sh-fixed는 "그 계획의 모든 iteration"(PushT 30, Wall 10) 동안 고정이다.
- 데이터셋 latent는 저장 영상 대신 데이터셋 상태·행동을 시뮬레이터로 렌더링해 만든다(위 §4).
- 본 실행 이후 시간·비용이 모자라면 episode 블록 경계에서 멈추고, 끝난 episode(모든 변형 동일)로 판정한다. 이 경우 REPORT에 episode 수를 밝힌다. 판정 문턱은 바꾸지 않는다.
- 순위 상관용 재생은 CPU에서 본 실행과 동시에 돈다. 시간이 모자라면 재생한 episode만으로 보고한다(판정과 무관한 보조 지표).
- 사후 분석은 "사후 분석"으로 표시한다.
