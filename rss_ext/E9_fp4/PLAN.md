# E9 4-bit 부동소수점 형식(NVFP4·MXFP4) — 실행 계획 (본 실행 전)

- 작성: 2026-10-31T09:38Z, E9 작업자(마지막 수정 09:39:36Z = 워커 수와 처리량 숫자 채움). e9.1 본 실행은 그 직후 09:39:39Z에 시작했다. 작성 시점에 본 실행(eval seed 80000~80029, held seed 81000~81029)의 결과는 하나도 없었다.
- 기준: [../DESIGN_ext.md](../DESIGN_ext.md) §0(공통), §2(E9), §4. 가설·지표·문턱·칸·seed는 설계 그대로이고, 이 문서는 운영 세부만 정한다.
- 코드: [code/](code/) (E1 `qlib.py`·`runner.py` 복사본 + FP4 추가), 기록: [LOG.md](LOG.md)

## 1. 가설과 판정 (DESIGN_ext §2.5 그대로)

- **F1(FP4 det도 계획을 해친다)**: 형식마다 하락이 실재하는 칸(그 형식의 12과제 안에서 Holm)이 3칸 이상이면 성립
- **F2(sh-iter 확률적 반올림이 고친다)**: F1에서 하락이 실재한 칸(손상 칸)으로 Meta-World와 DMC를 따로 판정한다(두 형식은 합친다).
  - 손상 칸의 75% 이상에서 red ≥ 0.5이고 해로운 칸이 없으면 성립
  - 주장은 E1 H3a처럼 Meta-World 판정으로 한다.
- **F3(1-step 오차로는 손상이 안 보인다)**: 손상 칸 중 det의 pt_err1h ≤ sh-iter의 pt_err1h인 칸이 하나 이상이면 성립. 칸 수를 보고한다.
- 기술 항목: 형식별 손상 크기와 E1 균일 k = 4 det 손상 비교, NVFP4 대 MXFP4
- 칸 = 과제 12 × 형식 2 = 24칸. 과제마다 none 하나, 칸마다 det와 sh-iter. 변형마다 90짝(체크포인트 1·2·3 × seed 80000~80029)

## 2. 구현 선택

### 2.1 FP4 산술 (설계 §2.3, `code/fp4lib.py`)
- 행 = 후보 하나(양자화 층 출력 (N, 512)의 한 행), 블록 = 은닉 차원을 따라 이어진 16(NVFP4)·32(MXFP4) 단위. 512는 두 블록 크기로 나누어떨어진다.
- 양자화 위치는 E1과 같다(dyn0, dyn1, rew0, rew1의 Mish 출력, 계획의 모든 dynamics·reward 호출, 정책 사전 rollout 포함). 가중치·encoder·Q·π는 fp32다.
- E2M1: 크기 {0, 0.5, 1, 1.5, 2, 3, 4, 6}, |x_s| > 6이면 6으로 자른다.
- **NVFP4**
  - g = hi/(6·448). hi는 E1 범위 상한(`main/results/E0_tdmpc2/jobs/ranges/ranges_<과제>_s<체크포인트>.json.gz`의 `quant[층][1]`, 과제·체크포인트·층별)
  - s_b = E4M3_RNE(amax_b/(6g))·g
  - **"E4M3 범위로 자르고"의 읽기**: 반올림 전에 amax_b/(6g)를 양의 E4M3 범위 [2^-9, 448]로 자른다. 위쪽은 RNE 뒤 자르기와 결과가 같다(448 초과는 448이 된다). 아래쪽 자르기는 scale이 0이 되는 것을 막는다. 실제 활성(mw-push 4층)에서 아래쪽에 걸린 블록은 0개였다.
  - amax_b = 0이면 s_b = g다.
- **MXFP4**: s_b = 2^(⌊log2 amax_b⌋ − 2)(지수 비트에서 정확히 읽음), amax_b = 0이면 1이다. amax_b < 2^-124는 2^-126으로 자르는데, Mish 출력에서는 생기지 않는다.
- x_s = x/s_b(IEEE 나눗셈), 출력 = q·s_b
- **det**: E2M1 RNE. 그 binade의 간격(0.5, 1, 2) 단위 반올림(동률은 짝수 배수 = 짝수 가수)과 같고, 정확하다.
- **sh-iter**: q = lo + (hi − lo)·1[u < (|x_s| − lo)/(hi − lo)], 부호를 다시 붙인다. |x_s| ≥ 6이면 ±6이다.
  - u는 (층, forward 호출)마다 (1, 512) 한 줄이다. 모든 후보가 공유하고, forward마다 새로 뽑는다.
  - E1 shiter와 같은 개별 CUDA generator(`Quantizer.gen`, episode마다 seed 7·seed + 13)에서 같은 모양으로 뽑는다. 계획기의 전역 RNG는 건드리지 않는다.
- 블록 scale은 두 변형 모두 결정적이다. 두 변형은 원소 반올림만 다르다.
- fp32 규약: g = fp32(hi/2688), 6g = fp32(6·g), r = fl(amax_b/6g)(IEEE), E4M3 반올림은 정확, s_b = fl(q_e4m3·g)

### 2.2 구현과 검증
- 두 구현이 있다. `fake_quant_ref`는 PyTorch 연산(참조)이고, `fake_quant_triton`은 같은 산술의 Triton 융합 kernel(층 호출당 kernel 1개)이다.
- **선택 규칙**(스모크 전에 정함): 단위 점검에서 Triton이 참조와 비트 단위로 같고, 같은 seed의 episode가 같으면 Triton을 쓴다. 아니면 참조를 쓴다.
- **결과**: Triton을 쓴다(§3).
  - 단 하나의 차이는 0의 부호다. 0으로 반올림된 음수 항목에서 참조는 −0.0, Triton은 +0.0을 낸다.
  - 부호 있는 0을 정규화하면 56경우 모두 비트 단위로 같다. 스모크 episode 8쌍의 return도 두 구현에서 같다.
- E1 코드 경로: hook의 FP4 분기는 균일 격자 코드보다 앞에서 반환한다. 손대지 않은 E1 `qlib.py`와 E9 `qlib.py`가 같은 pod에서 none·det·shiter·shstep episode를 비트 단위로 같게 낸다.

### 2.3 episode 실행 (E1과 같은 짝 구조)
- `qlib.run_episode`를 그대로 쓴다. 환경·random·numpy·torch는 episode seed로, dither generator는 7·seed + 13으로 정한다. 그래서 변형 사이에 계획기 RNG가 같다.
- eval episode에는 계측을 켜지 않는다(OEI, 계획 재실행, episode별 예측 오차 없음). PlanProbe는 closed loop를 바꾸지 않으므로(E1에서 확인) 계측 유무는 짝 구조에 영향이 없다.
- job = (과제, 체크포인트, 변형, 5 episode). 결과는 job마다 `jobs/episodes/eval_<과제>_s<c>_<변형>_cXX.json.gz`이고, 이미 있으면 건너뛴다.
  - 순서는 chunk 우선이다(모든 칸이 첫 chunk를 받은 뒤 둘째 chunk). chunk 안에서는 DMC를 먼저 둔다.
- **한 과제의 모든 episode(none + 4변형, 체크포인트 3개)는 한 pod(같은 GPU 종류)에서 돈다.** GPU가 다르면 같은 seed라도 수치가 비트 단위로 같지 않기 때문이다(E0 §4).

### 2.4 pt_err1h (E1 H1과 같은 정의와 코드 경로)
- held 입력은 none episode에서 만든다. 체크포인트 1·2·3 × seed 81000~81029, 과제당 90 episode다.
- 각 episode
  - OEI 단계(E1과 같은 간격: MW 4 step마다 25상태, DMC 10 step마다 50상태)에서 PlanProbe가 첫 MPPI 반복의 후보 집합(512개)을 보관한다.
  - 그중 고르게 떨어진 8상태로 `qlib.plan_inputs(K = 5)`를 만든다.
  - 변형마다 `qlib.pred_errors(h1 = horizon 3)`을 잰다. sh-iter는 dither seed 3개(9,000,001 + 1000·seed + j) 평균이다. 이것이 E1 episode별 `perr_plan`과 같은 계산이다.
- 칸 값 pt_err1h = 90 held episode의 `perr_plan.err1h` 평균. 데이터셋형(`perr_traj`, 같은 episode의 궤적 창)은 보조로 함께 기록한다.
- E1과 다른 점: E1은 각 변형 자신의 eval episode 후보에서 쟀다. E9는 설계 §2.4대로 held none episode의 후보로 잰다. 그래서 det와 sh-iter가 같은 입력에서 비교된다.

### 2.5 통계 (DESIGN_ext §0.3, `code/analyze_e9.py`, E1 `analyze_e1.py`의 함수 복사)
- D: Meta-World는 성공률 차(none − v), DMC는 상대 return 하락이다. DMC는 재표본마다 비율을 다시 계산한다.
- 짝 bootstrap 10,000회, `RandomState(0)`. 한 칸의 모든 통계에 같은 재표본 행렬을 쓰고, 95% 백분위 CI를 낸다.
- **하락 실재**
  - 한쪽 p = (#{D* ≤ 0} + 1)/(B + 1)(DESIGN_main v4 §6.1 규약)
  - 형식마다 12과제에 Holm(α = 0.05, 설계 §0.3)을 걸고, D_det > 0을 함께 요구한다(E1과 같음).
  - 문자 그대로의 비율 #{D* ≤ 0}/B로 판정이 달라지는 칸이 있으면 함께 보고한다.
- red = (D_det − D_sh-iter)/D_det(점 추정으로 판정). CI는 D_det* > 0인 재표본에서 구한다(E1과 같음).
- 해로운 칸: D_sh-iter − D_det의 95% CI 하한 > 0.05
- 모든 문턱 비교에 허용 오차 1e-9를 쓴다(red ≥ 0.5 − 1e-9, 비율 ≥ 0.75 − 1e-9, det ≤ sh-iter + 1e-9, 해로움은 하한 > 0.05 + 1e-9).
- **F2의 "해로운 칸이 없으면" 읽기**
  - 주 판정은 E1 H3a 규약을 따른다(설계가 "E1 H3a처럼"이라고 적음). 단위의 모든 칸(MW 16칸, DMC 8칸)에서 해로운 칸이 없어야 한다.
  - "손상 칸 안에서만"으로 읽은 판정도 함께 보고한다.
- 손상 칸이 없는 단위의 F2와, 손상 칸이 하나도 없을 때의 F3은 "판정 불가"로 적는다.
- 기술 항목
  - E1 균일 k = 4 det 손상(`main/results/E1_tdmpc2/e1_tdmpc2_cells.csv`, seed 20000~20029)과 과제별로 나란히 놓는다. seed가 달라 짝이 아니므로 기술로만 쓴다.
  - NVFP4 − MXFP4의 D_det·D_sh-iter 짝 차(같은 짝, 같은 재표본)와 CI
  - Meta-World McNemar 정확 p(보조)
- 분석은 pod에서 돌린다(맥북 계산 금지). 산출물: `results/e9_summary.json`(판정), `e9_cells.csv`(24칸), `e9_episodes.csv`(episode별 원자료), `e9_held_errors.csv`, `e9_tables.md`

## 3. 스모크 결과 (e9.1, A40, 2026-10-31T09:13~09:39Z)
- 설치: E1 고정판 그대로(torch 2.7.1+cu126, triton 3.3.1, tensordict 0.8.3, dm-control 1.0.16, mujoco 3.1.2, numpy 1.24.4, gym 0.21.0, Metaworld@04be337, mujoco-py 2.1.2.14 CPU builder), tdmpc2 332f19e. 체크포인트 36개의 SHA256이 E0와 같다.
- 단위 점검 7종(`results/smoke/test_fp4.json`)이 모두 PASS다.
  1. E2M1 RNE: 6,094값이 brute force와 같다. 중점은 0.25→0, 0.75→1, 1.25→1, 1.75→2, 2.5→2, 3.5→4, 5→4다. sh-iter 식도 brute force와 같다.
  2. E4M3 RNE: 20,501값이 torch `float8_e4m3fn` 변환과 brute force 모두와 같다.
  3. 블록 scale: NVFP4 8,192블록이 fp32 brute force와, MXFP4 4,096블록이 2^(⌊log2 amax⌋ − 2)와 같다. amax = 0이면 각각 g와 1이다.
  4. 비편향: 20,000회 평균의 z 최대 3.40(NVFP4)·3.20(MXFP4), 평균 |z| 0.78·0.76. 포화 항목은 정확히 ±6·s_b다.
  5. Triton = 참조: 56경우(무작위 5척도, Mish, 0·이상치, 행 24·1·4096, 실제 4층 활성) 모두 부호 있는 0 정규화 뒤 비트 단위로 같다.
  6. hook
     - 같은 입력의 512후보는 같은 출력을 낸다(u 공유). sh-iter는 forward마다 출력이 바뀌고 det는 그대로다.
     - 다시 seed하면 재현된다. generator는 층 호출마다 (1, 512) 한 번만 나아간다.
     - 전역 CPU·CUDA RNG는 바뀌지 않는다. encoder·Q·π에는 hook이 걸리지 않는다.
  7. 블록 경계: 한 항목을 바꾸면 그 행의 그 블록만 바뀐다(j = 0, 15, 16, 31, 32, 255, 256, 511).
- 실제 활성(mw-push s1 4층, 무작위 입력 후보)의 det 상대 MSE: NVFP4 0.008~0.013, MXFP4 0.023~0.026. 포화 비율은 NVFP4 3.3~4.6%(블록 scale이 내림될 때 최댓값 항목), MXFP4 1.5~1.8%다.
- E0 저장 episode 재현(seed 12000): 비트 단위로는 다르다. none은 가깝고(1785.18 대 1784.28, 871.44 대 861.54), det는 갈라졌다. E0 §4가 적은 GPU 의존(A40 대 A5000)이다. 같은 pod에서 E1·E9 qlib는 비트 단위로 같다(§2.2).
- FP4 스모크 episode(seed 89000)
  - 두 구현의 return이 8쌍 모두 같다.
  - mw-push: nvfp4 det 1808.0, sh-iter 1804.5, mxfp4 det 1794.8, sh-iter 1803.4(모두 성공)
  - cheetah-run: nvfp4 det 822.7, sh-iter 823.9, mxfp4 det 723.2, sh-iter 762.4
  - 호출 패턴은 step마다 dyn0·dyn1 ×512 18회, ×24 2회, rew0·rew1 ×512 18회다(E1과 같음).
- heldperr 스모크(seed 89001)
  - 상태 시점은 MW [0, 12, 28, 40, 56, 68, 84, 96], DMC [0, 70, …, 490]이다.
  - mw-push err1h: nvfp4 det 0.0081, sh-iter 0.0171, mxfp4 det 0.0216, sh-iter 0.0396
  - cheetah err1h: 0.0097, 0.0178, 0.0197, 0.0336
- 처리량(A40, 5변형 섞음, 워커당 1 episode job, 고정 시간)

| 묶음 | 워커 6 | 워커 8 | 워커 12 |
|---|---|---|---|
| MW(mw-push, pick-place) | 5,694 ep/h | 5,995 ep/h | 4,443 ep/h |
| DMC(cheetah, walker) | 1,155 ep/h | 1,163 ep/h | — |

  - pod의 CPU 할당이 7.65코어라 워커 12에서는 오히려 느려진다(CPU 제한). DMC는 워커 6과 8이 같고(GPU 포화), MW는 워커 8이 가장 빠르다. **본 실행은 워커 8**로 돈다.
  - episode당 시간은 워커 8 기준 MW none 4.4초·FP4 4.8~4.9초, DMC none 22.0초·FP4 24.0~24.9초다. Triton 경로의 FP4 비용은 none의 1.1배다.

## 4. 실행 순서와 pod 분할
- pod 둘(같은 GPU 목록)
  - **e9.1**(A40, 이미 떠 있음): Meta-World 8과제 + cheetah-run
  - **e9.2**(A40, 09:37Z 생성): walker-walk, quadruped-walk, finger-turn-hard
- 과제는 pod 사이에 나누지 않는다(§2.3).
- 각 pod의 manifest(`make_manifest_e9.py main --tasks …`): eval job(chunk 우선, DMC 먼저) 다음에 heldperr job
  - e9.1: eval 9과제 × 3 × 5변형 × 6 chunk = 810 job(4,050 episode), heldperr 162 job(810 episode)
  - e9.2: eval 270 job(1,350 episode), heldperr 54 job(270 episode)
- e9.2는 설치 뒤 `test_fp4.py`를 다시 돌린다(같은 GPU라도 kernel 점검). 통과하면 바로 본 실행이다.
- 1시간 이내 간격으로 진행·오류를 LOG.md에 적고, 결과를 맥북으로 당겨 온다.
- pod 4시간 자동 종료(e9.1 13:08Z, e9.2 13:36Z) 전에 끝나도록 계획했다. runner는 이어 돌릴 수 있다.
- 끝나면 한 pod로 결과를 모아 `analyze_e9.py`를 돌리고, 모두 당겨 온 뒤 두 pod를 지운다.

## 5. 비용 추정 (상한 $7, 80% = $5.60)
- 처리량은 워커 8 기준이다. held episode는 eval과 같은 비용 × 1.2로 잡았다(계측과 예측 오차 계산).

| pod | 일 | 추정 시간 |
|---|---|---|
| e9.1 | MW eval 3,600 ep / 5,995 = 0.60h, MW held 720 × 1.2 / 5,995 = 0.14h, cheetah eval 450 / 1,163 = 0.39h, held 90 × 1.2 / 1,163 = 0.09h | **1.22h** |
| e9.2 | DMC eval 1,350 / 1,163 = 1.16h, held 270 × 1.2 / 1,163 = 0.28h | **1.44h** |

- 비용
  - e9.1: 09:09Z부터 본 실행 끝(약 11:00Z)과 분석·회수(약 0.2h)까지 약 2.1h × $0.59 × 1.03 ≈ $1.28
  - e9.2: 09:37Z부터 설치(0.1h)·본 실행(1.44h)·회수까지 약 1.8h × $0.61 ≈ $1.09
  - **합 약 $2.4**(상한의 34%)
  - 처리량이 추정의 절반이어도 약 $4.0으로 80% 안이다.
- 지출이 $5.0을 넘을 것 같으면 멈추고 조정자에게 알린다.

## 6. 미리 적어 두는 편차와 위험
- E0 저장 episode가 이 pod에서 비트 단위로 재현되지 않는다(GPU 차이, E0 §4). E9의 판정은 모두 E9 안의 짝 비교라 영향이 없다. E1 k = 4 비교는 seed와 GPU가 달라 기술로만 쓴다.
- pt_err1h 입력: 설계 §2.4대로 held none episode의 후보로 잰다(§2.4의 E1과 다른 점).
- F2 해로운 칸의 범위: §2.5처럼 두 읽기를 모두 보고하고, 주 판정은 E1 H3a 규약으로 한다.
