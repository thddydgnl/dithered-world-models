# R2-14 Iteration dose-response — 사전등록 계획 (PLAN)

- 작성: 2026-10-21T08:50Z, **본 실행(grid, 08:54:07Z 시작) 전**. 스모크 결과는 본 실행 전에 §8에 덧붙인다.
- 상위: [../IDEAS_ROUND2.md](../../../archive/selection_process/IDEAS_ROUND2.md) R2-14 · 선행: R2-8b [../R2_8b_crn_dither/REPORT.md](../R2_8b_crn_dither/REPORT.md) · 배경: [../literature/NOVELTY_R2.md](../../../literature/NOVELTY_R2.md) 1.6절 1번
- 규칙: [../runpod_infra/PILOT_RULES.md](../../../runpod_infra/PILOT_RULES.md). 판정 기준 출처는 IDEAS_ROUND2.md다.
- pod `pai-r2n14`
  - **RTX A5000($0.27/h)**, id `duxs8ymw2ktpst`, 08:46:43Z 생성
  - CPU 할당은 cgroup 기준 10.2코어다
  - 상한 $1.2, 목표 $1.0 이내

## 1. 배경
- R2-8b에서 해로운 오차는 "최적화기 반복 동안 고정된 오차"였다.
  - 공유 잡음을 한 제어 step의 MPPI iteration 6회 동안 고정하면(sh-step), 매 forward 새로 뽑는 경우(sh-fwd)보다 4/4 과제에서 유의하게 해로웠다.
  - 예: mw-push k=3 성공률 하락이 sh-step 65%p, sh-fwd 5%p였다.
- 노벨티 재조사는 이 주장이 고전 원리(PEGASUS의 고정 scenario 과적합, CRN 확률 근사, optimizer's curse)와 구별되려면 **반복 수에 따른 용량-반응**을 보여야 한다고 지적했다.

## 2. 가설 (IDEAS_ROUND2.md 그대로)
- **H1:** 결정적 반올림(det)과 반복 내 고정 공유 잡음(sh-step)의 손상은 MPPI 반복 수가 늘수록 커진다. 최적화기가 고정 오차를 더 깊이 이용하기 때문이다.
- **H2:** 반복마다 새로운 공유 잡음(sh-fwd)의 손상은 반복 수에 거의 무관하다.
- **H3:** 비양자화 기준선은 반복 수가 늘면 좋아지거나 그대로다. 손상 증가가 단순히 "많이 최적화하면 나빠짐" 때문이 아님을 보인다.

## 3. 설정
- **R2-8b와 같은 것**
  - TD-MPC2 공식 5M 단일 과제 체크포인트
  - 양자화: dynamics·reward MLP 은닉 활성 4개 층, 층별 균일 k-bit, R2-8의 층별 범위 그대로
  - 변형 정의와 코드: det, sh-step, sh-fwd(R2-8b `nt_lib`)
  - dither 전용 CUDA generator. seed는 7 × episode seed + 13이다.
- **칸:** mw-push k=3, cheetah-run k=4
- **MPPI 반복 수 n ∈ {1, 3, 6, 12}**
  - `agent.cfg.iterations`만 바꾼다. `TDMPC2._plan`이 매 호출마다 이 값을 읽는다.
  - 다른 MPPI 설정은 기본값 그대로다: 후보 512개(policy prior 24개), elite 64개, horizon 3, temperature 0.5, std 범위, 이전 계획 warm start.
- **조건:** 과제마다 n 4개 × {none, det, sh-step, sh-fwd} = 16조건
- **episode:** 조건마다 15 episode, 합계 2 × 16 × 15 = **480 episode**
- **seed**
  - 본 실행은 **3000~3014**다. R2-8(1000~1009), R2-8b(2000~2019), 스모크(9000~9001)와 겹치지 않는 새 seed다.
  - 모든 n과 조건에서 같은 15개 seed를 쓴다. 그래서 변형 간에도, 반복 수 간에도 짝을 이룬다.
  - 스모크 seed는 9100이다.
- **calibration:** 비양자화, 기본 6 iteration, seed 1000~1009, 과제마다 10 episode
  - plandev의 결정 상태와 prep(범위 파일 작성, 예측 오차)에만 쓴다. 시험 episode가 아니다.
- **n=1에서 sh-step과 sh-fwd의 관계:** n=1이면 sh-step의 표는 MPPI iteration 한 번에서만 쓰인다. 두 변형은 policy prior rollout(24개 제안)이 표를 공유하는지 여부만 다르다. 그래서 n=1에서 두 변형의 차는 거의 0으로 예상된다.

## 4. 측정
- **하락(1차):** 반복 수 n에서 episode마다 d_i(n) = (기준_i(n) − 조건_i(n)) / R0(n)를 계산한다.
  - 기준은 **같은 n, 같은 seed의 비양자화 planner**이고, R0(n)은 그 평균이다.
  - MW는 마지막 step 성공(1차)과 return(보조)을 쓰고, cheetah는 return을 쓴다.
- **bootstrap:** 4000회, percentile 5~95. episode를 n과 조건에 걸쳐 함께 재표집한다. 그래서 반복 수 간 차이도 seed로 짝지어진다.
- **증가량:** Δ_v = D_v(12) − D_v(1)과 그 짝 CI
- **sh-step − sh-fwd:** n마다 짝 CI를 구하고, 1→12에서 그 차가 얼마나 늘었는지 본다.
- **기준선 R0(n)(H3)**와 짝 변화 기준(12) − 기준(1)
- **최종 계획 편차(보조)**
  - calibration 결정 상태 150개에서 n마다 한 step 계획을 다시 실행한다(같은 torch seed).
  - 최종 MPPI mean 첫 step의 RMS 편차를 같은 n의 seed noise 바닥으로 나눈다.
- **wall-clock**
  - (i) 단일 프로세스 측정(1차): grid가 끝나 GPU가 비었을 때, 과제 × n × 조건마다 제어 step 10회로 예열한 뒤 60회의 `agent.act` 시간을 잰다(cuda synchronize).
  - (ii) grid 안의 episode당 계획 시간: 워커 10개가 동시에 도는 상태라 참고용이다.

## 5. 사전등록 판정 기준 (IDEAS_ROUND2.md 그대로)
- **S1:** 과제 2개 모두에서, det의 하락(기준 대비)이 반복 1 → 12에서 단조 증가하거나, 반복 12의 하락이 반복 1의 2배 이상이다.
- **S2:** 같은 범위에서 sh-fwd 하락 증가는 det 하락 증가의 1/3 이하다.
- **S3 (권장):** sh-step이 det와 같은 방향의 용량-반응을 보인다.
- **판정**
  - S1과 S2를 만족하면 있음: 반복이 고정 오차를 증폭한다는 기전의 용량-반응 증거
  - S1만 만족하면 조건부

### 5.1 실행 전에 채운 세부 규칙
1. **"기준 대비 하락"**은 같은 n의 비양자화 기준선 대비 하락이다(seed 짝). 판정은 평균 하락의 점추정으로 한다.
2. **S1(과제마다)**
   - 다음 둘 중 하나를 만족하면 된다.
     - (a) 단조: D(1) ≤ D(3) ≤ D(6) ≤ D(12)이고 D(12) > D(1)
     - (b) 2배: D(12) > 0이고 D(12) ≥ 2 · max(D(1), 0)
   - 채움 조건: n=12에서 det의 하락이 실재해야 한다(CI 하한 > 0). 하락이 거의 0인 상태에서 잡음 수준의 증가로 통과하는 것을 막기 위해서다. 이 조건 없이 문구 그대로 센 결과도 보고한다.
   - 과제 2개 모두 성립하면 S1 성립이다.
3. **S2(과제마다):** Δ_det = D_det(12) − D_det(1) > 0이고 Δ_sh-fwd ≤ Δ_det / 3이면 성립이다(점추정).
   - S1과 같은 범위이므로 과제 2개 모두 성립해야 S2 성립이다.
   - 두 증가량의 CI도 보고한다.
4. **S3(과제마다):** Δ_det > 0이고 Δ_sh-step > 0이면 성립이다. 과제 2개 모두 성립해야 S3 성립이다. n별 sh-step − sh-fwd도 보고한다.
5. **판정:** S1과 S2 → 있음, S1만 → 조건부 있음, S1 실패 → 없음. S3는 함께 보고한다.
6. **H3 점검(판정 기준 아님):** R0(n)과 기준(12) − 기준(1)을 보고한다. 기준선이 n과 함께 나빠지면 혼입 요인으로 적는다.

## 6. 참고 예측과 우려
- **R2-8b(n=6, seed 2000~2019)**
  - mw-push: det 65%p, sh-step 65, sh-fwd 5
  - cheetah: det 52.4%, sh-step 77.2, sh-fwd 47.1
- **우려(미리 적음): warm start가 제어 step 사이에서 계획을 이어 준다.**
  - 그래서 det처럼 episode 내내 같은 오차는 n=1에서도 step을 거치며 누적해서 이용될 수 있다.
  - 이 경우 det의 하락은 n=1에서 이미 크고 용량-반응이 평평할 수 있다. 그렇게 되면 S1은 실패한다.
  - sh-step은 step마다 새로 뽑으므로 이 누적이 없다. 그래서 sh-step의 용량-반응이 반복 내 이용을 더 깨끗하게 보여 줄 것으로 본다(S3, n별 sh-step − sh-fwd).

## 7. 실행 순서와 예산
1. 설치, calibration(20 episode), prep(범위 파일)
2. **스모크:** mw-push, n ∈ {1, 12}, none·det·sh-step·sh-fwd × 1 episode(seed 9100)
   - forward 호출 패턴을 확인한다. 512행 호출이 층마다 3n회여야 한다.
   - `cfg.iterations` 반영과 표 갱신 횟수를 확인한다.
   - plandev 5 상태, timing 5 step, analyze까지 돌린다.
3. **본 실행:** grid 480 episode(워커 10개, MPS) → plandev(150 상태 × n 4개) → timing(단일 프로세스) → analyze_r214
4. 결과를 회수하고 pod를 삭제한다.
- 예상: grid 20~25분, pod 가동 약 1시간($0.3)

## 8. 스모크 결과 (본 실행 전에 기록, 08:53Z)
- **설치·calibration·prep(08:47~08:52Z)**
  - 비양자화 mw-push seed 1000의 return은 1770.1이다. R2-8b의 3090(1769.3)과 다르므로 이 A5000은 수치가 다르다. 모든 비교는 이 pod 안에서 한다.
  - 이 pod에서 다시 잰 범위는 R2-8과 1~3% 다르다(예: cheetah dyn0 hi 9.43 vs 9.17). 양자화는 계획대로 R2-8의 범위를 쓴다.
- **스모크(mw-push k=3, seed 9100, n ∈ {1, 12}, 8 episode, 08:52Z)**: 끝까지 돌았고 구현이 의도대로 동작했다.
  - `cfg.iterations` 반영: 기록값이 설정값과 같았다(1, 12).
  - 제어 step당 forward 호출 수: 모든 step에서 한 가지 패턴이다.
    - n=1: 512행 호출이 층마다 3회
    - n=12: 512행 호출이 층마다 36회
    - policy prior(24행): dynamics 층마다 2회
  - 표 갱신 횟수: sh-step은 101회(100 step + 1)
  - plandev(5 상태), timing(5 step), analyze까지 돌았다.
- **스모크 결과는 기능 점검에만 쓰고 해석하지 않는다.**
- **속도(단일 프로세스, 제어 step당 계획 시간)**
  - n=1: 7~8ms
  - n=12: 비양자화 46ms, det 57ms, sh 변형 68~69ms. 양자화 hook의 부담이 n에 비례해 늘어난다.
- **본 실행 설정:** 워커 10개(cgroup 10.2코어)
