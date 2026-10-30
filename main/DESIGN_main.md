# A 본 실험 사전등록 (DESIGN_main.md) — v1

- 고정: 2026-10-21T18:14Z (조정자). v1이 고정하는 것
  - A 전체의 가설·지표·판정 규칙
  - TD-MPC2 쪽의 과제, 비트 수, seed와 조작적 정의
- DINO-WM 쪽 세부(과제별 k, start·goal 분할, 처리량에 따른 episode 묶음)는 DINO-WM E1 전에 v2로 **덧붙인다**. v1의 내용은 바꾸지 않는다.
- 근거
  - 설계: [EXPERIMENT_DESIGN.md](../EXPERIMENT_DESIGN.md). 아래 1~3절은 그 문서 3장, 4.2~4.3절, 6.0~6.8절을 고정 시점 그대로 옮긴 것이다.
  - E0 보고: [reports/E0_tdmpc2.md](reports/E0_tdmpc2.md)
- 이 문서를 고정한 뒤 바꾼 것은 모두 사후 분석이나 편차로 표시한다.

---

## 0. E0에서 정한 것 (TD-MPC2)

### 0.1 실행 장소와 환경
- 서버 GPU 0(RTX A5000 24GB), CUDA MPS 작업자 16개
- 한 venv(`~/pmx/envs/a-mw`)로 Meta-World와 DMC를 함께 돈다.
  - Meta-World의 mujoco-py는 사용자 공간 Mesa 24.3.4(micromamba env `gl24`)로 빌드했다.
  - 파일럿 return(mw-push seed 1000 = 1770.13)이 RunPod A5000 파일럿과 같다. 그래서 RunPod 대체 경로는 쓰지 않는다.
- 같은 장비 원칙: TD-MPC2의 모든 E1 칸은 GPU 0 한 장에서 돈다.

### 0.2 과제와 비트 수 (held 선택 결과, `results/E0_tdmpc2/ktable.json`)
- held는 seed 12000~12019, 체크포인트 seed 1·2·3에서 돌렸다. Meta-World는 체크포인트마다 20 episode(과제당 60), DMC는 10 episode(과제당 30)다.
- 설계 문구 "held 20 episode"는 **체크포인트 3개 합**으로 해석해 고정한다. E1 조건과 같고 표본이 더 많기 때문이다.
- Meta-World 8과제 모두 기준 성공률 ≥ 98%로 자격이 있다.
- 1차 k가 3인 과제의 k−1 = 2 칸도 규칙대로 넣는다.
- mw-drawer-open은 20%p와 15%p에 이르는 k가 없다. 그래서 R2-13 대체 규칙(시험한 가장 작은 k)을 따른다.
- mw-sweep-into는 k = 3 하락이 정확히 20.0%p로 문턱에 걸린다(규칙상 자격).

| 과제 | 1차 k | E1 칸 | 규칙 |
|---|---|---|---|
| mw-button-press-topdown | 3 | 2, 3, 4 | 20%p 규칙 |
| mw-door-open | 4 | 3, 4, 5 | 20%p 규칙 |
| mw-drawer-open | 3 | 2, 3, 4 | 대체 규칙(시험한 가장 작은 k) |
| mw-peg-insert-side | 4 | 3, 4, 5 | 20%p 규칙 |
| mw-pick-place | 3 | 2, 3, 4 | 20%p 규칙 |
| mw-push | 3 | 2, 3, 4 | 20%p 규칙 |
| mw-sweep-into | 3 | 2, 3, 4 | 20%p 규칙 |
| mw-window-open | 4 | 3, 4, 5 | 20%p 규칙 |
| cheetah-run | 5 | 4, 5, 6 | 20%p 규칙 |
| finger-turn-hard | 4 | 3, 4, 5 | 20%p 규칙 |
| quadruped-walk | 3 | 2, 3, 4 | 20%p 규칙 |
| walker-walk | 3 | 2, 3, 4 | 20%p 규칙 |

### 0.3 seed와 분할 (모든 체크포인트 seed에 같은 episode seed)

| 분할 | episode seed | 용도 |
|---|---|---|
| calib | 11000~11009 | ep 0~4: 층별 범위(min, 99.99 백분위). ep 5~9: held 예측 오차 집합과 고정 결정 상태 |
| held | 12000~12019 | 과제·k 선택(끝남) |
| **eval** | **20000~20029** | **E1 판정 전용. E0에서 한 번도 쓰지 않았다** |
| smoke | 19000~19009 | 스모크(해석하지 않음) |

- 칸마다 30 episode × 체크포인트 3개 = 90 짝 episode다. 모든 변형이 같은 episode seed와 planner 난수열을 쓴다.
- dither는 별도 CUDA generator(episode seed × 7 + 13)를 쓴다.

### 0.4 조작적 정의 (설계에 없던 세부)
1. **det+shift**
   - unit마다 offset o ~ U[−½, ½)(격자 단위), 모든 후보가 공유, forward 호출마다 새로 뽑는다(sh-iter와 같은 일정).
   - q = round(x − o)를 [0, 2^k − 1]로 자르고, 값은 lo + Δ(q + o)다.
2. **TD-MPC2 OEI**
   - 비용 오차 e = V₀ − V_q다. V₀는 같은 RNG 상태(같은 Q 쌍, 같은 policy 잡음)에서 잰 비양자화 가치다.
   - OEI = e(실행된 elite: 마지막 반복에서 Gumbel로 뽑힌 것) − 첫 반복 표본 후보 488개(policy prior 24개 제외)의 평균 e
   - 보조로 `oei_w`, `oei_top1`, `oei_first`를 함께 기록한다.
   - 측정 간격: Meta-World는 4 step마다, DMC는 10 step마다다.
3. **계획 시점 5-step 오차**: 후보의 3개 행동 뒤에 비양자화 policy prior의 평균 행동 2개를 붙인다(모든 변형에 같은 행동열). 데이터셋형 입력은 실제 실행 행동을 쓴다.
4. **예측 오차**
   - R2-8b 정의(비양자화 예측 분산으로 나눈 상대 MSE)를 쓴다.
   - 확률적 변형은 dither seed 3개의 평균이다.
   - 계획 편차는 칸마다 decision job(고정 상태 150개)과 episode 안 계획 재실행(Meta-World 20 step, DMC 50 step마다)으로 잰다.
5. **보정 범위**: (과제, 체크포인트)마다 calib ep 0~4에서 R2-8 절차로 다시 잡는다. 그래서 같은 k라도 파일럿과 격자가 다르다.
6. **오차 구조 기술자**: `rho_iter`(연속 두 반복의 후보 간 비용 오차 상관), `common_frac`(모든 후보에 공통인 오차 비중)

### 0.5 실행
- runner: `main/code/tdmpc2/run_e1.sh`(manifest 4,320 job, 다시 시작하면 끝난 job은 건너뜀)
- 로그: `~/pmx/A/logs/E1_tdmpc2.log`
- E1 처리량은 Meta-World 약 3,700, DMC 약 860 episode/시간이고, GPU 0 단독으로 약 12시간이다.

---

## 1. 연구 질문과 가설 (설계 3장 그대로)


| RQ | 질문 | 가설(사전등록) |
|---|---|---|
| RQ1 현상 | 결정적 저비트 반올림은 1-step 예측 오차가 말해 주는 것보다 계획을 더 망가뜨리는가? | H1: 두 월드모델 계열 × 두 계획기 모두에서, det가 기준 대비 20%p 이상(DMC는 return 20% 이상) 떨어뜨리는 비트 수가 있다. 그 비트 수에서 det의 1-step 오차는 확률적 변형보다 크지 않다 |
| RQ2 기전 | 손상은 계획기가 결정적 오차를 이용하기 때문인가? | H2a: det에서 OEI(4.3절)가 유의하게 음수이고, 확률적 변형에서는 0 근처다. H2b: 조건 전체에서 OEI가 1-step·k-step 오차보다 손상을 더 잘 예측한다(Spearman 차 ≥ 0.15). H2c: \|OEI\|는 후보 수가 많을수록 커진다(optimizer's curse의 용량-반응) |
| RQ3 처방 | 같은 비트 수에서 dither가 손상을 없애는가? 강한 PTQ 위에서도 그런가? | H3a: 자격 칸(det 하락이 실재하는 과제 × 비트 수 × 모델)의 75% 이상에서, 반복마다 새로 뽑는 공유 dither(sh-iter)가 det 손상의 50% 이상을 없앤다. 어떤 칸에서도 det보다 5%p 넘게 나쁘지 않다. H3b: 최선의 PTQ 기준선 위에 dither를 더하면 3~4-bit에서 추가 이득이 있다 |
| RQ4 구조 | 공유·새로움은 언제 필요한가? | H4a: 고정 dither(sh-fixed)의 손상은 dither 표 크기 N_eff가 커질수록 줄어든다(단조, 사전 예측식 4.4절). H4b: 후보 공유(CRN)의 이득은 near-tie 밀도가 높은 과제에서 크다 |
| RQ5 배치 | 실제 정수 커널에서도 이득이 남는가? | H5: dither 추가 지연은 10% 이하다. "dither + 낮은 비트"가 "det + 높은 비트"를 지연–성공률 Pareto에서 지배하는 설정이 하나 이상 있다 |
| RQ6 일반성(선택) | 다른 결정적 오차원(활성 희소화, 수치 정밀도)에도 같은 원리가 적용되나? | H6: 결정적 활성 희소화(R1-1의 약한 활성 제거)를 기대값을 맞춘 확률적 희소화로 바꾸면 계획 손상이 준다 |

---

---

## 2. 변형과 지표 (설계 4.2~4.3절 그대로)

### 4.2 오차 구조의 두 축
- **후보 간 공유**: 같은 forward의 모든 후보가 u를 공유하는가(shared), 후보마다 독립인가(ind)
- **반복 간 지속**: u를 매 forward 새로 뽑는가(fresh), 한 계획(제어 step)의 반복 동안 고정하는가(plan-fixed), episode 내내 고정하는가(episode-fixed)
- 변형 표
  - 주 처방은 **sh-iter**(= R2-8b의 sh-fwd)다.
  - ind, sh-iter, sh-step, sh-ep는 항목별 오차의 주변분포가 같고, 후보 간·시간 상관만 다르다(R2-8b §3.3 정의를 그대로 쓴다).

| 변형 | 후보 간 | 반복 간 | 비고 |
|---|---|---|---|
| none | — | — | 비양자화 기준 |
| det | 입력이 같으면 같은 오차 | 결정적(완전 지속) | RTN |
| ind | 독립 | 매 forward 새로 | |
| **sh-iter** | 공유 | 매 forward 새로 | **주 처방** |
| sh-step | 공유 | 계획 하나 안에서 고정 | (horizon 위치 × unit) 표 |
| sh-ep | 공유 | episode 내내 고정 | |
| det+shift | 입력이 같으면 같은 오차 | 반복마다 공유 격자 이동 | 결정적 반올림 + 반복마다 바뀌는 공유 offset(무작위 격자). "지속 편향만 깨도 충분한가"를 가르는 대조 |

### 4.3 지표
- **1차 지표**
  - 과제 성공률. Meta-World와 DINO-WM 과제는 마지막 step 성공, DMC는 episode return
  - 같은 seed의 비양자화 기준선 대비 하락 D_v
  - 손상 제거율 red_v = (D_det − D_v)/D_det
- **계획기 이용 지수(OEI)**: 한 계획에서 계획기가 최종적으로 고른 후보(또는 elite 평균)의 비용 오차 e(â)에서, 같은 계획 첫 분포의 무작위 후보 비용 오차의 평균을 뺀 값이다. 음수면 계획기가 오차가 유리한 후보를 골랐다는 뜻이다.
  - 비용 오차의 기준
    - DINO-WM: 시뮬레이터 재생으로 얻은 참 비용
    - TD-MPC2: 비양자화 모델의 가치 추정치. 참 return은 closed-loop로만 잰다.
  - 파일럿 값: DINO-WM PushT의 det에서 무작위 후보 +0.30, 고른 후보 −0.09(R2-13)
- **예측 오차**
  - 1-step·k-step(k=3, 5) 상대 오차. 데이터셋 입력과 계획 시점 입력(첫 반복의 후보) 두 가지로 잰다.
  - 확률적 변형은 dither seed 3개의 평균이다.
- **계획 편차**: 같은 상태와 seed로 계획을 다시 돌렸을 때 최종 계획 첫 행동의 RMS 편차를 seed-noise 바닥으로 나눈 값(R2-8b §4)
- **결정 수준(보조)**: 고정 후보 집합에서의 Spearman 순위 상관, argmax flip, gap 사분위별 flip, 마지막 반복 순위 상관(DINO-WM은 참 비용 기준, R2-7·R2-13)
- **오차 구조 기술자**
  - 후보 간 오차 상관 ρ_cand
  - 반복 간 지속 ρ_iter
  - 유효 dither 표 크기 N_eff: 비용 하나가 합산하는 독립 offset 수
  - near-tie 밀도: 1·2위 gap / 후보 std의 분포

---

## 3. 본 실험 프로토콜과 판정 (설계 6.0~6.8절 그대로)

### 6.0 공통 프로토콜 (사전등록 대상)

**모델·계획기**

| 모델 | 체크포인트 | 계획기(기본값 고정) | 과제 |
|---|---|---|---|
| TD-MPC2 | 공식 5M 단일 과제, seed 1·2·3(HF `nicklashansen/tdmpc2`) | MPPI: 후보 512(policy prior 24 포함), elite 64, 반복 6, horizon 3, temperature 0.5 | Meta-World 8개(기준 성공률 ≥ 80%인 과제를 사전 선정 규칙으로 고름. 후보: push, pick-place, door-open, drawer-open, button-press-topdown, peg-insert-side, window-open, sweep-into), DMC 4개(cheetah-run, walker-walk, quadruped-walk, finger-turn-hard) |
| DINO-WM | 공식 PushT · Wall · PointMaze(OSF) | CEM: 후보 300, top-30, opt_steps(PushT 30, Wall 10, PointMaze 설정값), goal_H 5, open-loop | PushT, Wall, PointMaze(서버에서 mujoco-py가 E0 안에 설치되지 않으면 제외하고 사전에 기록) |
| V-JEPA 2-AC(선택) | 공식 `vjepa2_ac_vit_giant` | 공식 CEM(표본 800, 반복 10, horizon 1) 오프라인 | DROID-100 창(실험 6) |

**양자화 위치**
- TD-MPC2: dynamics·reward MLP 은닉 활성 4개 층(Mish 이후)
- DINO-WM: 예측기 MLP 은닉 활성(GELU 이후, 6블록)
- 범위는 보정 데이터로 정하고, 보정 데이터와 평가 seed는 겹치지 않게 한다.
- 가중치 양자화는 실험 5에서 따로 본다.

**비트 수 선택 규칙(R2-13 규칙 그대로)**
- 과제마다 held 20 episode에서 det가 20%p 이상(DMC는 return 20% 이상) 떨어뜨리는 가장 큰 k를 1차 칸으로 둔다.
- k−1과 k+1을 2차 칸으로 둔다.
- k ∈ {3, 4, 5, 6}이다.

**episode·seed**
- TD-MPC2: 칸마다 30 episode × 체크포인트 seed 3개
- DINO-WM: 칸마다 100 episode(eval 시작·목표 쌍)
- 모든 변형이 같은 episode seed와 계획기 난수열로 짝을 이룬다.
- 확률적 변형의 난수는 별도 generator를 쓴다(R2-8b §3.3).

**실행 장소: 대학 GPU 서버**
- 모든 실험을 대학 GPU 서버에서 돈다. 맥북에서 Tailscale과 Windows 중계를 거쳐 접속한다(ssh 별칭 `kiis-mvf-gpu`).
- 장비: Ubuntu 22.04 컨테이너, RTX A5000 24GB × 2(드라이버 560, CUDA 12.6), 64스레드, RAM 503GiB. CPU와 디스크는 다른 사용자와 공유한다.
- 제약: Docker, sudo, conda가 없고 OpenGL·EGL·OSMesa 라이브러리도 없다. 환경은 사용자 공간의 uv venv로 만든다.
- **mujoco-py 점검(E0 첫 작업)**
  - TD-MPC2의 Meta-World와 DINO-WM PointMaze는 mujoco-py를 쓴다. 렌더링을 하지 않아도 빌드에 OpenGL 라이브러리(OSMesa·GLEW)가 필요하다(파일럿 `setup_mw.sh`).
  - 사용자 공간 설치를 먼저 시도한다: micromamba로 conda-forge Mesa·GLEW, MuJoCo 2.1 바이너리, pip `patchelf`
  - Meta-World가 끝내 안 되면 그 칸만 RunPod에서 돌리고 사전등록에 적는다. PointMaze는 제외한다.
- DMC는 렌더링을 끄고(`MUJOCO_GL=disable`) 돈다.
- DINO-WM PushT·Wall은 MuJoCo와 OpenGL 없이 돈다(파일럿 설치에 MuJoCo가 없었다). 서버의 기존 venv에서 pygame·pymunk·opencv가 동작하는 것을 확인했다(2026-10-21).
- 작업은 작업 목록 기반으로 돌린다. 작업마다 결과 파일을 쓰고, 이미 있으면 건너뛰어 중단 뒤에도 이어서 돈다. 결과는 맥북으로 수시로 rsync한다.

**같은 장비 원칙**
- GPU 모델이 다르면 같은 seed도 episode가 비트 단위로 같지 않다(R2-3b). 그래서 한 칸의 모든 변형과 기준선을 같은 GPU에서 돈다.
- 서버의 두 GPU는 같은 모델이지만, 한 칸을 두 GPU에 나누지 않는다. 칸마다 GPU를 배정하고 기록한다.

**보정·평가 분리**
- 범위 보정, k 선택, 이론 예측식의 상수는 held 분할에서 정한다.
- 판정은 eval 분할로만 한다.

### 6.1 실험 1 — 주 벤치마크 (RQ1, RQ3)
- **격자**: 모델 2종 × 과제(TD-MPC2 12, DINO-WM 3) × 비트 수 3(1차 k와 ±1) × 변형 7(4.2절 표)
- **지표**: 1차 지표, red, 1·3·5-step 오차, 계획 편차
- **판정(H1, H3a)**
  - H1: 두 모델 계열 모두에서 det 하락이 실재하는(짝 bootstrap 95% CI 하한 > 0) 칸이 하나 이상 있다. 그 칸에서 det의 계획 시점 1-step 오차 ≤ sh-iter의 1-step 오차다.
  - H3a: 자격 칸 중 75% 이상에서 red_sh-iter ≥ 0.5다. 어떤 칸에서도 D_sh-iter − D_det의 95% CI 하한이 +5%p를 넘지 않는다.
  - 과제별 Holm 보정을 하고, 모델 계열마다 따로 보고한다.
- **예상**
  - MW에서는 큰 이득(80~90% 제거)
  - DMC에서는 작은 이득(10~30%)
  - DINO-WM에서는 큰 이득(R2-13 수준)
- **go/no-go(E1 결과로 판단)**
  - H3a가 TD-MPC2 MW와 DINO-WM 둘 다에서 성립하면 go다.
  - 하나만 성립하면 주장을 그 계열로 좁히고, 대신 실험 3(구조)을 강화한다.
  - 둘 다 실패하면 진단 논문(OEI와 1-step 지표의 실패)으로 전환한다.

### 6.2 실험 2 — 기전: 계획기의 오차 이용 (RQ2)
- **측정**: 모든 칸에서 OEI를 기록한다. DINO-WM은 참 비용 기준, TD-MPC2는 비양자화 가치 기준이다.
- **용량-반응(사전등록)**: 과제 4개(mw-push, mw-pick-place, cheetah, PushT)에서 다음을 바꾼다.
  - 후보 수 N ∈ {64, 128, 256, 512, 1024}
  - elite 비율 ∈ {2%, 12.5%, 25%}
  - MPPI temperature ∈ {0.1, 0.5, 2}
  - 예측: det의 |OEI|와 손상은 N과 함께 커지고(극값 통계, 4.4절), sh-iter에서는 평탄하다.
- **반복 수(R2-14의 확장)**
  - n ∈ {1, 3, 6, 12}(MPPI), {5, 10, 30}(CEM)
  - warm start를 끈 대조를 함께 돈다. 한 step 안에서의 이용과 step 사이 누적을 가르기 위해서다(R2-14 다음 단계 1).
- **판정(H2)**
  - H2a: det의 OEI 95% CI 상한 < 0이고, sh-iter의 OEI CI가 0을 포함한다(과제 4개 중 3개 이상).
  - H2b: 실험 1 전체 칸에서 Spearman(OEI, 손상) − max(Spearman(1-step, 손상), Spearman(3-step, 손상)) ≥ 0.15
  - H2c: N 증가에 따른 det 손상과 |OEI|의 Spearman이 0.8 이상이다(과제 4개 중 3개 이상).

### 6.3 실험 3 — 오차 구조와 dither 표 크기 (RQ4)
- **2축 대조**
  - 두 모델에서 공유(공유 / 독립) × 지속(매 forward / 계획 고정 / episode 고정)의 6칸을 같은 비트 수로 돈다.
  - 4.2절의 ind, sh-iter, sh-step, sh-ep에 "독립 + 계획 고정"을 더한다.
- **표 크기 sweep(핵심)**: 같은 모델 안에서 고정 dither의 독립 offset 수를 줄여 간다.
  - DINO-WM: token × unit(기본, 층당 40만~120만) → unit당 하나(2048) → 블록 group 단위(128) → 층당 하나(1)
  - TD-MPC2: horizon × unit(기본) → unit당 하나(512) → group(32) → 층당 하나(1). 반대로 (후보 × horizon × unit)으로 키우는 방향도 둔다.
  - 예측: 고정 dither 손상과 후보 간 오차 변동은 N_eff가 커질수록 단조 감소한다. 4.4절의 예측식으로 맞춘 기울기가 두 모델에서 같은 형태다.
- **near-tie 밀도와 CRN 이득**
  - 과제마다 near-tie 밀도(gap / std 중앙값)를 잰다.
  - 그것이 (D_ind − D_sh-iter)를 예측하는지 본다(H4b). DMC의 작은 효과를 설명하는 후보 변수다.
- **판정(H4)**
  - H4a: 두 모델 모두에서 N_eff에 대한 sh-fixed 손상의 Spearman이 −0.8 이하다.
  - H4b: 과제 전체에서 Spearman(near-tie 밀도, CRN 이득) ≥ 0.5

### 6.4 실험 4 — 이론과 모의 (4.4절)
- 2차 목적함수(차원 2~12)와 TD-MPC2식 MPPI·DINO-WM식 CEM을 그대로 구현한다. 오차 모형 4가지 × N × 반복 수 × elite 비율 격자에서 다음을 잰다.
  - 최종 평균의 편향
  - regret
  - OEI
- 유도한 예측식과 비교한다.
- 실험 2·3의 측정값을 같은 축에 겹쳐 그린다.
- CPU만으로 충분하다.

### 6.5 실험 5 — 강한 기준선과 결합 (RQ3, H3b)
- **활성 PTQ 기준선**
  - per-channel scale
  - 범위 보정: percentile 대 MSE 최소화
  - SmoothQuant식 활성→가중치 이전
  - 혼합 정밀도(마지막 층 8-bit 유지)
- **가중치 양자화**: W8 / W4 per-channel RTN, AWQ식 weight-only
  - 가중치의 확률적 반올림은 원리상 고정된다(sh-fixed와 같음). 그래서 "SR 사본 K개를 반복마다 돌려 쓰기"(K ∈ {1, 2, 4})를 탐색 조건으로 둔다.
- **QAT**: 월드모델 데이터로 fake-quant 미세조정(짧은 예산). QAT가 문제를 없애면, dither의 가치는 "학습 없는 PTQ 처방"으로 한정해 보고한다.
- **판정(H3b)**: 최선 기준선 + sh-iter가 최선 기준선 + det보다 낫다. 3~4-bit 자격 칸의 과반에서 짝 CI 하한 > 0이어야 한다.

### 6.6 실험 6 — 다른 결정적 오차원 (RQ6, 선택)
- **6a 결정적 활성 희소화**
  - R1-1의 약한 활성 제거(|a| < α·m_j → 0)와, 기대값을 맞춘 확률적 희소화를 비교한다.
  - 확률적 희소화는 각 항목을 확률 p_j로 유지하고 1/p_j로 재스케일한다. 같은 희소율이다.
  - 원문 에세이와의 연결을 닫는 실험이다.
- **6b 수치 정밀도(V-JEPA 2-AC)**
  - 공식 CEM(표본 800, 반복 10)을 DROID 창 200개에서 오프라인으로 돈다.
  - bf16 / fp16 / fp32에 따른 최종 행동 편차와 실행 행동 대비 regret을 잰다.
  - 정밀도 잡음이 "후보 독립·반복 간 지속"(입력이 같으면 같은 반올림) 구조인지 측정한다.
  - "입력 미세 dither"(반복마다 공유 미소 잡음)로 결정 안정성이 오르는지 탐색한다.
  - 서버 A5000(24GB)에서는 후보 800개를 나눠 forward한다(파일럿 R2-5는 A40 48GB). 나누는 크기는 모든 정밀도 조건에서 같게 고정한다.

### 6.7 실험 7 — 실제 커널과 배치 Pareto (RQ5)
- TD-MPC2 MLP를 int8 활성(torch `_int_mm` 또는 torchao dynamic int8)과 int4(Triton 커스텀 커널, 4-bit 활성 pack)로 구현한다.
- dither는 층·forward마다 난수 벡터 하나를 더하는 비용이고, 후보 수와 무관하다.
- 장비: 서버의 RTX A5000(Ampere라 int8 텐서 코어와 torchao int4 커널을 쓸 수 있다). 연구실에 Jetson Orin이 있으면 추가한다.
- 서버 CPU는 다른 사용자와 공유한다. 그래서 지연은 조건 순서를 섞어 반복 측정하고 중앙값으로 보고하며, 측정 때의 호스트 부하를 함께 기록한다.
- 격자: {fp32, bf16, int8 det, int8 dither, int4 det, int4 dither} × MPPI 반복 {3, 6, 12} × 과제 4개
- 측정: 제어 step 지연과 성공률. Pareto 곡선을 그린다.
- **판정(H5)**
  - dither 추가 지연(중앙값) ≤ 10%
  - Pareto 지배 사례가 하나 이상

### 6.8 통계와 사전등록
- 모든 하락과 차이에 episode 단위 짝 bootstrap(10,000회) 95% CI를 붙인다. 성공률 쌍 비교에는 McNemar를 쓴다.
- 1차 가설군(H1, H3a)은 과제 단위 Holm–Bonferroni로 보정한다. 2차(H2, H4, H5)는 보정 없이 탐색으로 표시한다.
- **검정력**
  - 성공률 짝 비교에서 불일치 비율 30%, 효과 20%p를 가정하면, 30 episode × 체크포인트 3개(= 90 짝)로 검정력이 0.9 이상이다.
  - DINO-WM은 100 episode다.
  - DMC return은 R2-8b의 분산으로 다시 계산해 사전등록 문서에 적는다.
- **사전등록 문서**: `DESIGN_main.md`(이 문서의 3·6장을 옮겨 고정)를 실험 1 본 실행 전에 만든다. 체크포인트 seed, 과제 목록, k 선택 결과, episode seed 범위를 기록한다.
- **보고 원칙**
  - 판정 실패 칸도 모두 보고한다.
  - 사후 분석은 표시한다.
  - 평균만이 아니라 칸별 그림을 부록에 넣는다.

---

## 4. DINO-WM 세부 — v2 (2026-10-21T20:01Z 덧붙임; 1~3절과 0절은 바꾸지 않음)

- 근거: [reports/E0_dinowm.md](reports/E0_dinowm.md). R2-13 재현이 맞았다.
  - PushT k=4: det 하락 +18%p [4, 32], sh-iter 감소율 1.00
  - Wall k=3: +36%p [20, 52], sh-iter 감소율 1.00
  - OEI 파일럿 값 +0.30 / −0.09를 +0.299 / −0.103으로 재현했다.

### 4.1 과제와 비트 수 (held = pair seed 3, 쌍 0~19, 20 episode)

| 과제 | none | det 하락 k3 / k4 / k5 / k6 | 1차 k | E1 확증 칸 | 비고 |
|---|---|---|---|---|---|
| PushT | 85% | −50 / −40 / +5 / 0 %p | 4 | 3, 4, 5 | 20%p 규칙, 파일럿과 같음 |
| Wall | 55% | −55 / +15 / −20 / −5 %p | 5 | 4, 5, 6 | 20%p 규칙 문자 그대로(k=5 하락이 정확히 20%p, p = 0.22, 비단조). **k = 3을 탐색 칸으로 덧붙인다**(파일럿의 k, 확증 판정 H1·H3a에는 넣지 않고 따로 보고) |
| PointMaze | 85% | −15 / −5 / −5 / +5 %p | 3 | 2, 3, 4 | 20%p에 이르는 k가 없어 R2-13 대체 규칙(15%p 이상인 가장 큰 k)을 따름. TD-MPC2 drawer-open과 같은 처리 |

- PointMaze는 mujoco-py 2.1.2.14 + OSMesa로 동작하므로 과제 목록에 남긴다.

### 4.2 변형
- none은 과제마다 한 번(k와 무관), nonealt(planner seed + 500)는 계획 편차의 seed 바닥으로 한 번 더 돈다.
- det, ind, sh-iter, det+shift를 돈다. det+shift는 0.4절 정의 그대로다.
- **sh-plan**: DINO-WM은 open-loop라 episode당 계획이 한 번이다. 그래서 sh-step(계획 안 고정)과 sh-ep(episode 내내 고정)가 정의상 같은 조건이다. 하나(코드 이름 `shstep`)만 돌리고, 결과를 sh-step과 sh-ep 양쪽 자리에 쓴다(E3의 2축 대조에서도 같은 칸).

### 4.3 분할과 seed (`results/E0_dinowm/splits_<task>.json`)
- eval은 pair seed 5, 쌍 0~99다.
  - env seed 5001~5100, planner seed 3000~3099, nonealt 3500~3599, dither salt `eval5`
  - E0에서는 이 쌍으로 계획을 한 번도 돌리지 않았다.
- **Wall·PointMaze의 쌍**은 seed별 무작위 상태라 held·파일럿 쌍과 겹치지 않는다.
- **PushT의 eval 쌍**은 val 궤적 21개에서 잘라 held·파일럿 구간과 프레임이 겹친다(eval 100쌍 중 held 31, 파일럿 eval 70). 처리 방법은 사용자 확인 뒤 v3에서 고정한다. **PushT E1은 그 전에 시작하지 않는다.**
- 배치 크기 E는 PushT 1, Wall 2, PointMaze 1이다. GPU 0과 1, 그리고 E에 관계없이 계획이 비트 단위로 같다. 그래서 칸을 GPU별로 나눠도 되고, job마다 GPU를 기록한다.

### 4.4 OEI 기준
- 두 기준의 입력을 모두 저장한다.
  - 비양자화 모델 비용 기준(`oei_ref_*`, 파일럿 값의 정의)
  - 시뮬레이터 참 비용 기준(`oei_true_*`)
- 어느 쪽을 확증 지표로 쓸지는 사용자 확인 뒤 E2 분석 전에 v3에서 고정한다. 그 전에는 E1의 OEI 수치를 보지 않는다.


## 5. v3 덧붙임 (2026-10-21T22:54Z, 사용자 승인; PushT E1과 E1 OEI 분석 전)
- **DINO-WM OEI 확증 지표**
  - 비양자화 모델 비용 기준(`oei_ref_*`)으로 정한다. 파일럿 값의 정의와 같고, TD-MPC2(비양자화 가치 기준)와도 같다. 양자화 오차만 분리한다.
  - 시뮬레이터 참 비용 기준(`oei_true_*`)은 보조로 보고한다.
  - 설계 문서 4.3절의 문구를 이에 맞게 정정했다.
- **PushT eval 쌍 다시 뽑기**
  - held 쌍(pair seed 3, 0~19)과 프레임이 하나도 겹치지 않도록 eval 쌍 100개를 새로 뽑는다.
  - 파일럿 eval(pair seed 4)과의 겹침은 가능한 한 줄이고 겹침 통계를 기록한다.
  - 새 쌍은 E1에서 처음 쓰며, 그 전에는 계획을 돌리지 않는다.
  - Wall·PointMaze는 바뀌지 않는다(애초에 겹침 없음).


## 6. v4 덧붙임 (2026-10-22T06:55Z, 조정자; TD-MPC2 E1 결과와 두 계열의 E1 OEI 결과를 보기 전)
- 1~5절은 바꾸지 않는다. 6.1절 판정 문구를 두 계열에 같은 방식으로 계산하도록 세부를 고정한다.
- DINO-WM 작업자가 2026-10-21T23:10Z에 PushT 결과 전에 정한 계산 규칙(`code/dinowm/analyze_e1.py`)을 기준으로 삼고, TD-MPC2에도 그대로 적용한다.

### 6.1 E1 판정 계산 (두 계열 공통)
- **칸**: (계열, 과제, k). 확증 칸은 0.2절과 4.1절 표다. Wall k = 3 탐색 칸은 H1·H3a에 넣지 않고 따로 보고한다.
- **짝**: TD-MPC2는 (체크포인트 seed, episode seed) 90개, DINO-WM은 eval 쌍 100개다.
- **하락 D_v**
  - Meta-World와 DINO-WM: 성공률 차(%p)
  - DMC: 상대 return 하락 (R̄_none − R̄_v) / R̄_none. bootstrap에서는 짝을 다시 뽑은 뒤 비율을 다시 계산한다.
- **det 하락 실재**
  - 짝 bootstrap(10,000회) 한쪽 p(D_det ≤ 0) = (#{D* ≤ 0} + 1) / (B + 1)
  - 과제마다 그 과제의 확증 k 칸 3개에 Holm–Bonferroni를 건다. α = 0.025(한쪽)로, 두쪽 95% CI와 같은 수준이다.
  - 보조로 보정 없는 95% CI 하한 > 0과 McNemar 정확 p(성공률)를 함께 보고한다.
- **red_v** = (D_det − D_v) / D_det는 자격 칸에서만 계산한다. CI는 같은 bootstrap 표본에서 비율로 구한다.
- **H1**(계열마다): det 하락이 실재하는 확증 칸 가운데, det의 계획 시점 1-step 오차가 sh-iter의 것보다 크지 않은 칸이 하나 이상 있다.
  - 1-step 오차는 칸의 episode 평균이다. 첫 반복 후보에 대한 R2-8b 상대 오차이고, 확률적 변형은 dither seed 3개의 평균이다.
  - 데이터셋형 입력의 오차는 보조다.
- **H3a**: 자격 칸 가운데 red_sh-iter ≥ 0.5인 비율이 75% 이상이다. 그리고 어느 확증 칸에서도 D_sh-iter − D_det의 95% CI 하한이 +5%p(DMC는 상대 return 0.05)를 넘지 않는다.
  - 판정 단위는 TD-MPC2 Meta-World(8과제 24칸), TD-MPC2 DMC(4과제 12칸), DINO-WM(3과제 9칸)이다. TD-MPC2 전체도 함께 보고한다.
  - go/no-go(6.1절)는 TD-MPC2 Meta-World와 DINO-WM의 H3a로 판단한다.

### 6.2 E1 자료로 계산하는 H2a·H2b (실험 2의 일부)
- **OEI 확증 지표**
  - TD-MPC2: `oei`(0.4절, 실행된 elite − 첫 반복 표본 후보 488개 평균)
  - DINO-WM: `oei_ref_mf`(5절)
  - 칸 값은 측정한 모든 계획의 episode 평균을 짝에 걸쳐 평균한 것이다. CI는 짝 bootstrap이다.
  - 두 계열 모두 음수가 이용(모델이 유리하게 틀린 후보를 고름)이다.
- **H2a**: 실험 2의 과제 4개(mw-push, mw-pick-place, cheetah-run, PushT)에서 1차 k 칸(기본 계획기 설정)으로 판정한다.
  - 과제마다 det OEI의 95% CI 상한 < 0이고, sh-iter OEI의 95% CI가 0을 포함하면 그 과제는 성립이다.
  - 4개 중 3개 이상이면 H2a 성립이다.
  - 실험 2의 sweep에서 기본 설정을 다시 돌리면 재현으로 따로 보고한다.
- **H2b**(계열마다, 확증 칸 전체)
  - 점: (과제, k, 변형). 변형은 none을 뺀 전부다(TD-MPC2 6개, DINO-WM 5개).
  - 예측 변수: −OEI(이용 크기), 계획 시점 1-step 오차, 계획 시점 3-step 오차. 결과 변수: 손상 D_v
  - **주 계산**: 과제마다 그 과제의 점들로 Spearman 차 ρ(−OEI, D) − max(ρ(1-step, D), ρ(3-step, D))를 구하고, 과제 평균이 0.15 이상이면 성립이다.
    - 이렇게 정한 이유: TD-MPC2의 OEI는 가치 단위라 과제마다 척도가 다르다. 과제를 섞어 순위를 매기면 상관이 과제 차이를 주로 잰다.
  - **보조**: 설계 문구 그대로 과제를 섞은 Spearman(원값), 그리고 OEI를 과제의 none 변형 첫 반복 후보 가치(비용) 표준편차로 나눈 값으로 섞은 Spearman. 데이터셋형 오차로 바꾼 계산도 함께 보고한다.
- H2c(후보 수 용량-반응)와 실험 2의 sweep 세부는 sweep 시작 전에 v5로 고정한다.


## 7. v5 덧붙임 (2026-10-22T09:00Z, 조정자; E2·E3·E5·E7 본 실행 전)
- 근거: 두 작업자의 초안
  - [reports/v5_draft_tdmpc2.md](reports/v5_draft_tdmpc2.md)
  - [reports/E2E3E5_dinowm_proposal.md](reports/E2E3E5_dinowm_proposal.md)
- 1~6절은 바꾸지 않는다.
- 사용자 결정이 필요한 두 가지는 v5.1로 따로 고정한다. 그 전에는 해당 분석을 하지 않는다.
  - E2 덧붙임 층 (b): 독립 draw판 OEI 판정, drift 기반 H2a′·H2c′, 영역 효과 대조
  - TD-MPC2 E1 H2a 3칸의 독립 draw판 재측정
- E2의 실행 목록은 사용자 답과 무관하다(모든 지표를 기록). 층 (b)는 분석만 바뀐다.

### 7.0 공통
- **통계**: v4 §6.1과 같다(짝 bootstrap 10,000회, RandomState(0), 백분위 95% CI, DMC는 재표본마다 비율 재계산).
- **episode 수**: 모든 실험·설정에서 프로토콜(6.0절) 그대로다.
  - TD-MPC2: 90짝(체크포인트 3 × 30)
  - DINO-WM: 100 episode
  - 초안의 20 ep(60짝) 설정은 모두 30 ep(90짝)으로 올린다.
- **OEI 기록**: 모든 E2·E3 episode에 두 판과 분해를 함께 기록한다.
  - TD-MPC2: `oei`(동결, 같은 draw), `oei_indep`(독립 draw 재평가), sel_last, sel_indep, drift. 표 변형은 e_fresh, 정책 slot 비중도 기록한다.
  - DINO-WM: `oei_ref_mf`(동결 = 독립 draw판), `oei_ref_argmin`, last_pop_dJ, it0_rand_dJ, last_elite_dJ, mf_dJ
  - det에서는 두 판이 정의상 같다(결정적 오차).
- **같은 장비**
  - TD-MPC2: 실험(manifest) 하나를 한 GPU에서 끝까지 돈다. E1과 같은 GPU 0을 쓴다.
  - DINO-WM: 칸 하나를 한 GPU에서 돈다. 두 GPU의 결과가 비트 단위로 같다는 것을 확인했으므로, 칸을 GPU 사이에 나눠 배정해도 된다.
- **계산 보류**
  - v5.1 전에는 DINO-WM E1의 OEI 분해(sel_last, drift)를 계산하지 않는다.
  - H4b와 DMC 설명의 상관은 E3 자료가 다 나온 뒤 한 번만 계산한다.

### 7.1 E2 (기전)
**TD-MPC2**
- 칸: mw-push k=3, mw-pick-place k=3, cheetah-run k=5(1차 k). seed는 E2 분할 40000~40029(처음 씀)이고, 체크포인트는 1~3이다.
- OFAT 16설정(기본값 둘레에서 한 번에 한 요인). 모든 설정이 같은 seed라 설정 사이에도 짝이 맞는다.

| 설정 | 후보 N | elite | 정책 prior | temperature | 반복 | warm start |
|---|---|---|---|---|---|---|
| base | 512 | 64 | 24 | 0.5 | 6 | 켬 |
| N64 / N128 / N256 / N1024 | N | N/8(비율 12.5%) | round(24N/512) = 3 / 6 / 12 / 48 | 0.5 | 6 | 켬 |
| el2 / el25 | 512 | 10 / 128 | 24 | 0.5 | 6 | 켬 |
| T01 / T2 | 512 | 64 | 24 | 0.1 / 2.0 | 6 | 켬 |
| it1 / it3 / it12 | 512 | 64 | 24 | 0.5 | 1 / 3 / 12 | 켬 |
| nows1 / nows3 / nows6 / nows12 | 512 | 64 | 24 | 0.5 | 1 / 3 / 6 / 12 | 끔(매 step `t0=True`) |

- 변형: 모든 설정에 none, det, sh-iter, sh-step. base에는 ind, sh-ep, det+shift를 더한다. 각 설정의 none이 그 설정의 손상 기준이다.
- 계측: E1 간격(OEI는 MW 4 step, DMC 10 step마다, 계획 재실행은 20 / 50 step마다)이다. episode별 예측 오차는 끈다.
- N = 512는 E2 seed로 다시 돈다. E1 기본값 자료는 재현으로 따로 보고한다.

**DINO-WM**
- 칸: PushT k=4. 변형은 none, det, sh-iter이다. E1의 eval6 쌍과 planner seed를 쓰고, 점마다 100 episode다.
- 후보 수 N ∈ {64, 128, 256, 512, 1024}에 E1의 N = 300을 여섯 번째 점으로 더한다.
  - top-k는 elite 비율 10%로 고정한다: 6, 13, 26, 51, 102
  - N = 1024는 `max_rows = 300`으로 나눠 돈다(계획이 비트 단위로 같음을 확인).
- elite 비율 {2%, 12.5%, 25%}(N = 300, top-k 6 / 38 / 75)에 E1의 10%를 더한다.
- opt_steps {5, 10}에 E1의 30을 더한다.
- temperature와 warm start 끈 대조는 해당 없다. CEM에는 temperature가 없고, open-loop라 episode당 계획이 한 번이며 매번 mu = 0, σ = 1에서 시작한다.

**판정: 층 (a), 원래 사전등록**
- **H2a 재현**: TD-MPC2 base(E2 seed)에서 동결 정의 `oei`로 v4 6.2 규칙을 다시 계산하고, 재현으로 보고한다. PushT는 E1 값이다.
- **H2c**: 과제마다 그 과제의 N 수준 전체에 걸쳐 판정한다(TD-MPC2 5수준, DINO-WM은 E1의 300을 포함한 6수준).
  - ρ_s(N, |OEI_det|) ≥ 0.8이고 ρ_s(N, D_det) ≥ 0.8이면 그 과제는 성립이다. D_det(N)은 같은 N의 none 대비 하락이다.
  - 과제 4개(mw-push, mw-pick-place, cheetah-run, PushT) 중 3개 이상이면 H2c 성립이다.
  - 설계 문장 "|OEI|는 후보 수가 많을수록 커진다"와 예측 "det의 |OEI|와 손상은 N과 함께 커진다"가 모두 N에 대한 단조성이라 이렇게 읽는다.
  - 보조
    - N에 걸친 ρ_s(D_det, |OEI_det|)
    - sh-iter가 평탄한지: ρ_s(N, |OEI_sh-iter|), ρ_s(N, D_sh-iter)
    - ρ의 bootstrap 분포(설정 사이 짝을 함께 다시 뽑음)
- **elite 비율, temperature, 반복 수, warm start**: 판정 문턱이 없다. 설정별 D, OEI 두 판과 성분을 CI와 함께 기술 보고한다.
  - warm start를 켠 반복 수 곡선과 끈 곡선을 함께 그린다(R2-14 확장).
- **층 (b)**: v5.1

### 7.2 E3 (구조와 표 크기)
**TD-MPC2**
- **2축 6칸**: 12과제의 1차 k. 6칸은 아래와 같고, none과 det을 함께 돈다.
  - 후보 공유: sh-iter, sh-step, sh-ep
  - 후보 독립: ind, indstep(새로), indep(새로)
  - 설계 6.3은 "6칸"이라 하면서 새 칸으로 "독립 + 계획 고정"만 적었다. 그래서 여섯 번째 칸을 "독립 + episode 고정"(indep)으로 채운다.
  - seed는 E3 분할 50000~50029(90짝)다.
- **indstep·indep 정의**
  - 층마다 (horizon, 후보 slot, unit) 표를 후보 slot 번호에 붙인다.
  - indstep은 제어 step마다, indep은 episode마다 고정한다.
- **사전 예상**(실행 전 명시)
  - indstep ≈ ind ≈ indep. 손상, 독립 재평가 OEI, drift에서 구별되지 않을 것으로 본다.
  - 이유: MPPI는 반복마다 표본 후보를 새로 뽑으므로 slot 번호에 행동열 정보가 없다.
  - 예외는 정책 prior slot 24개다. 한 제어 step 안에서 같은 행동열이 유지되지만, 실행 elite의 약 1%다.
- **표 크기 sweep**
  - 과제: mw-push k=3, mw-pick-place k=3, cheetah-run k=5, walker-walk k=3
  - walker는 E1에서 sh-iter가 해로웠던 칸이라 넣었다. 결과를 본 뒤 고른 과제임을 밝힌다.

| 이름 | 층당 표 크기 | N_eff(비용 하나가 합산하는 독립 offset, 4층) |
|---|---|---|
| shstep_l | 1 | 4 |
| shstep_g32 | 32 | 128 |
| shstep_u | 512 | 2,048 |
| shstep(기본, horizon × unit) | 1,536 | 6,144 |

  - indstep(후보 × horizon × unit)은 4.3절 정의로 N_eff가 기본과 같다(6,144). 그래서 H4a 축에 넣지 않고, 후보 축 대조(sh-step 대 indstep)로 따로 보고한다.
  - 같은 거칠기의 새로 뽑기 대조 shiter_l, shiter_g32를 보조로 돈다. 지속의 비용 = D(shstep_g) − D(shiter_g)
  - 새 변형마다 decision job(E1과 같은 고정 상태)으로 rho_iter, rho_iter_perm, common_frac를 잰다.

**DINO-WM**
- **2축 4칸**(open-loop라 계획 고정 = episode 고정): sh-iter, sh-plan, ind는 E1 칸을 쓴다. indplan만 새로 돈다.
  - 칸: 세 과제의 1차 k(PushT k=4, Wall k=5, PointMaze k=3)
  - 덧붙여 PushT k=3과 Wall k=3(탐색)도 돈다.
- **indplan 정의**
  - 후보 slot × token × unit 표를 층·horizon step마다 계획 시작에 한 번 뽑는다.
  - 그 계획의 모든 CEM 반복에서 같은 slot 번호에 쓴다.
- **사전 예상**: indplan ≈ ind. 지속은 slot 0(현재 평균)에만 남고 top-30 평균에서는 작다.
- **표 크기 sweep**
  - 수준: L0 shstep(E1, token × 2048) > L1 shunit(2048) > L2 shgroup(128) > L3 shlayer(1)
  - L+ indplan은 따로 보고한다.
  - 과제: 세 과제의 1차 k(PushT k=4, Wall k=5, PointMaze k=3). Wall k=3은 탐색이다.

**판정**
- **H4a**
  - 과제마다 공유 표 4수준에서 Spearman(N_eff, D_sh-fixed) ≤ −0.8이면 그 과제는 성립이다. 4점이라 이웃 자리바꿈 한 번까지 허용한다.
  - E1 1차 k 칸에서 det 하락이 실재한(v4 6.1) sweep 과제만 쓴다.
  - 계열마다 그 과제의 과반(TD-MPC2 4개 중 3개, DINO-WM 3개 중 2개)이 성립하면 그 계열은 성립이다.
  - 두 계열 모두 성립하면 H4a 성립이다(설계 "두 모델 모두에서").
  - 보조: 과제별 Spearman 중앙값
- **H4b**
  - near-tie 밀도 = −(마지막 반복 후보 집합의 비양자화 1·2위 gap / 후보 std의 중앙값)
    - TD-MPC2: E1 decision job의 고정 상태 450개
    - DINO-WM: E1 none 변형의 마지막 반복 모집단
  - **부호**: E1 보고표의 값은 gap 중앙값이라 near-tie가 많을수록 작다. 밀도는 그 음수다.
  - CRN 이득 = 1차 k에서 D_ind − D_sh-iter
    - TD-MPC2: E3 seed
    - DINO-WM: E1 칸
  - 확증: TD-MPC2 12과제에서 Spearman(밀도, CRN 이득) ≥ 0.5
  - 보조: DINO-WM 3과제를 섞은 15과제, 첫 반복 집합
  - 이렇게 정한 이유: DINO-WM은 3과제뿐이라 계열 안 검정이 안 되고, 두 계획기의 밀도 정의가 다르다.
- **DMC 설명**(설계 §7 대응의 조작적 정의)
  - 자료: E3 seed의 2축 칸(12과제 1차 k). E3 seed에서 det 하락이 실재한 과제(한쪽 bootstrap p < 0.025)만 쓰고, 빠진 과제는 적는다.
  - 결과 변수: 과제마다 red_sh-iter
  - 설명 변수 1(확증): S_det = drift_det / oei_det
    - det 칸 평균이고, oei_det의 CI가 0을 포함하면 그 과제를 뺀다.
    - 뜻: 실행 elite에서 det가 얻은 유리한 오차 가운데 반복 사이에 지속되며 쌓인 몫이다.
  - 설명 변수 2: H4b의 near-tie 밀도
  - 규칙
    - Spearman(S_det, red_sh-iter) ≥ 0.5이면 "설명됨"이다.
    - 아니고 Spearman(밀도, red_sh-iter) ≥ 0.5이면 "부분 설명"이다.
    - 둘 다 아니면 "설명 안 됨"이다.
  - 설명 여부와 무관하게 E1의 DMC H3a 불성립은 그대로다. 설명 결과는 그 한계를 어떻게 서술하는지에 쓴다.
  - 처방 주장을 좁히는 것은 사용자 결정 사항이다(2026-10-22 질문).

### 7.3 E5 (강한 기준선과 결합, H3b)
**TD-MPC2**
- **칸**: E1에서 det 하락이 실재한 3~4-bit 칸 19개
  - MW 12칸: button-press 3, door-open 3·4, drawer-open 3, peg-insert 3·4, pick-place 3, push 3·4, sweep-into 3, window-open 3·4
  - DMC 7칸: finger 3·4, quadruped 3·4, walker 3·4, cheetah 4
- **seed**: 기준선 선택은 held 12000~12019, 평가는 E5 분할 60000~60029(90짝), QAT 자료는 65000~65019
- **활성 PTQ 기준선 4종**(같은 k, 같은 4층)
  1. per-channel 범위: unit마다 [min, 99.99 백분위]
  2. MSE 최소 범위: hi = α × 99.99 백분위, α ∈ {0.5, …, 1.1} 중 보정 활성의 k-bit det MSE 최소
  3. SmoothQuant식 이전: α 0.5, 다음 Linear에 접는다.
  4. 혼합 정밀도: dyn1·rew1을 8-bit로 둔다.
- **최선 기준선 B**: 칸마다 held에서 D_det(B)가 가장 작은 것을 고른다. 같으면 목록의 앞쪽이다. dither는 B의 격자 위에 sh-iter 일정으로 얹는다.
- **가중치 양자화**(탐색, 활성은 fp)
  - 과제: sweep 과제 4개, 90짝
  - 변형: w8rtn, w4rtn, w4awq, w4sr1·2·4(SR 사본 K개를 반복 번호 mod K로 돌림)
- **QAT**
  - 칸: 자격 3~4-bit 칸 중 과제의 1차 k 11칸 × 체크포인트 3개
  - 방법: det fake-quant + STE, dynamics·reward MLP만, consistency + reward 손실, 5,000 step, batch 256, lr 1e-4, grad clip 20
  - 평가: E5 seed에서 QAT+det, QAT+sh-iter를 none과 비교한다(90짝).
  - QAT+det 손상의 CI 상한이 5%p(DMC 0.05) 아래면, dither의 가치는 "학습 없는 PTQ 처방"으로 한정해 보고한다(설계 6.5).

**DINO-WM**: 초안의 전체판을 돈다(약 16 GPU-시간).
- 활성 PTQ 4종
  - per-channel(보정 창 256개)
  - percentile {99.9, 99.99, 99.999} 대 MSE 최소
  - SmoothQuant α 0.5
  - 마지막 블록 8-bit
- 최선 기준선: (과제, k)마다 held 20 ep의 det 성공률로 고른다.
- 평가: eval 100 ep에서 B + det 대 B + sh-iter. 칸은 E1에서 det 하락이 실재한 3~4-bit 확증 칸이다(Wall k=3 탐색 칸은 뺀다).
- 가중치(PushT k=4, 100 ep, 활성은 fp): W8·W4 RTN, AWQ식 W4, W4 SR K ∈ {1, 2, 4}
- QAT: PushT k=4, 약 2 GPU-시간. 평가는 none·det·sh-iter, 100 ep다.

**H3b**: 계열마다 판정한다. 자격 3~4-bit 칸의 과반에서 D(B+det) − D(B+sh-iter)의 95% CI 하한이 0보다 크면 성립이다(TD-MPC2는 19칸 중 10칸 이상).

### 7.4 E7 (실제 커널과 지연, TD-MPC2)
- **구현**
  - 층마다 융합 Triton kernel 하나: int8 dot(int32 누적) → 역양자화 → bias → LayerNorm → Mish → 다음 층 입력 재양자화(det 또는 dither)
  - dither는 층·forward마다 공유 u 벡터 하나다(sh-iter 일정).
  - int4 활성은 int8 그릇이 기본이다. packed int4는 목표이고, 준비되지 않으면 그릇판으로 보고하고 편차로 적는다.
  - torchao는 선택 기준선이다. 쓸 때는 별도 venv(`~/pmx/envs/a-e7`, a-mw 복제)에 torch 2.7.1과 맞는 판을 고정해 설치하고, a-mw는 바꾸지 않는다.
- **지연 규약**
  - 배타 GPU(다른 프로세스와 MPS 없음), 단일 프로세스, 제어 step을 CUDA event로 잰다.
  - warmup 50 step 뒤 200 step 블록을 순서를 섞어 5회전 잰다.
  - 중앙값과 IQR을 보고하고, 호스트 부하와 GPU clock을 기록한다.
  - 확증은 eager다. CUDA graph는 보조다.
- **격자**: {fp32, bf16, int8 det, int8 dither, int4 det, int4 dither} × 반복 {3, 6, 12} × 과제 4개(mw-push, mw-pick-place, cheetah-run, walker-walk)
  - 성공률은 실제 kernel로 90짝(seed 70000~)에서 잰다.
- **H5**: (int8 dither − int8 det) / int8 det의 지연 중앙값이 10% 이하(int4도 같음)이고, Pareto 지배 사례가 하나 이상이다.

### 7.5 일정 (서버)
- **E1을 먼저 끝낸다**
  - GPU 0: PushT k=4 → k=3, 약 19:30Z까지
  - GPU 1: B(약 12Z까지) + PointMaze(약 14Z까지) → PushT k=5(약 19~20Z)
  - 그때까지 GPU 0에는 PushT 옆에서 E7 kernel 개발·정확성 시험(단일 프로세스, 간헐)과 몇 분짜리 E5 통계만 돌린다. PushT가 go/no-go의 임계 경로다.
- **E1 뒤 GPU 0(TD-MPC2, MPS)**: (승인 시 H2a 재측정, 약 0.5시간) → E2 → E3 → E5(QAT 포함) → E7 성공률
  - E7 지연 측정은 배타 창에서 한다.
  - 다 끝나면 GPU 0도 남은 DINO-WM 칸을 맡는다.
- **E1 뒤 GPU 1(DINO-WM)**: E2 N sweep → E3 → E5 → E2 elite·opt_steps
- **예상**: TD-MPC2 약 30 GPU-시간, DINO-WM 약 55 GPU-시간. 끝은 10-24 오후(UTC) 무렵이다.
- **예산 편차**: 설계 6.9절의 E2~E7 추정은 약 36 GPU-시간이었다. 늘어난 이유는 셋이다. 서버 비용은 0이고 RunPod는 쓰지 않는다.
  - A5000이 4090보다 느리다.
  - 모든 설정을 프로토콜 episode 수로 돈다.
  - E3를 모든 과제의 1차 k로 넓혔다.


## 8. v5.1 덧붙임 (2026-10-22T09:30Z, 조정자; E2·E3·E5 자료가 하나도 없을 때)
- 7절의 실행 세부 가운데 세 가지를 고친다. 가설과 문턱은 바꾸지 않는다.
- 근거: E5 통계, [reports/E4_theory.md](reports/E4_theory.md)의 7.2·7.3절. E4는 E1 결과 파일을 읽지 않았다.
- 사용자 결정이 필요한 두 가지(E2 층 (b), TD-MPC2 H2a 재측정)는 이름을 v5.2로 바꿔 미룬다.

### 8.1 E5 TD-MPC2 MSE clip α 격자
- 7.3절 격자 {0.5, …, 1.1}는 k=3에서 144경우 중 102경우가 아래 끝 0.5에 걸렸다. 그래서 MSE 최소 기준선이 실제로 MSE 최소가 아니었다.
- 고친 격자는 {0.10, 0.15, …, 0.45} ∪ {0.5, 0.6, …, 1.1}이다. 다시 잰 결과 아래 끝(0.10)과 위 끝(1.1)에 걸린 경우는 288경우 중 하나도 없다.

### 8.2 E2 후보 수 sweep: H2c 확증은 elite 수를 고정한 sweep으로 한다
- **이유**
  - 설계 4.4절의 H2c 예측식은 극값 통계다(가우시안이면 ∝ √(2 log N)). 고르는 후보 수가 정해져 있을 때의 optimizer's curse다.
  - 7.1절은 두 작업자 초안대로 elite 비율을 고정했다. 그러면 선택 강도가 N과 무관하게 일정하다.
  - E4 유도와 사전 선언 모의(5.7절)에 따르면, 비율을 고정했을 때 |OEI|는 N 64 → 1024에서 0.99~1.40배밖에 오르지 않는다. 손상은 오히려 N과 함께 준다.
  - elite 수를 고정하면 |OEI|가 1.7~5.4배로 오르고, 지속 오차에서는 손상도 N과 함께 커진다.
  - 비율 고정은 설계의 예측과 다른 질문을 잰다. 그래서 확증 sweep을 설계 문장에 맞게 바로잡는다.
- **확증 sweep**: N만 바꾸는 OFAT다.
  - TD-MPC2: elite 64, 정책 prior 24로 고정한다. 설정 이름은 N64k / N128k / N256k / N1024k이고, base가 N = 512다.
  - DINO-WM: top-30으로 고정한다. E1의 N = 300이 여섯 번째 점이다.
  - N = 64의 TD-MPC2는 64개가 모두 elite라 절단 선택이 없다(softmax 가중과 최종 Gumbel 선택만 남는다). 용량-반응의 아래 끝으로 그대로 쓴다.
- **보조**
  - TD-MPC2: 7.1절의 비율 고정 설정(N64 / N128 / N256 / N1024)을 그대로 돌리고 기술 보고한다(약 2 GPU-시간 추가).
  - DINO-WM: 비율 고정 sweep은 돌리지 않는다(약 19 GPU-시간). 선택 강도의 효과는 N = 300의 elite 비율 sweep이 본다.
- H2c의 판정 문구와 문턱(7.1절)은 그대로이고, 확증 sweep에 적용한다.
- 보고에는 OEI 분해를 함께 싣는다. 이론상 N 의존은 sel_last(선택 연산자)에서 오고 drift는 1차로 평탄하다(E4 7.2절).

### 8.3 E3 표 크기 sweep에 var_cand 측정을 더한다
- 수준마다 고정 표 오차의 후보 간 분산(`var_cand`, 첫 반복 후보 비용 오차의 분산)을 함께 잰다. 두 계열 모두다.
- 손상의 보조 예측 변수로만 보고하고, H4a 판정은 바꾸지 않는다.
- 이유: E4 유도(2.6절)에 따르면 1/N_eff 법칙은 "일관 축"에서만 성립한다. 손상의 직접 예측 변수는 그 수준의 후보 간 오차 분산이다.


## 9. v5.2 덧붙임 (2026-10-22T09:40Z, 사용자 승인; E2 자료와 DINO-WM E1 OEI 분해를 보기 전)
- 사용자가 조정자의 보완 추천안 1과 추천안 2를 그대로 승인했다(MAIN_STATUS 2026-10-22 질문).
- 1~8절의 사전등록 판정은 그대로 남는다. TD-MPC2 E1의 H2a·H2b 불성립은 고정 정의대로 기록하고 보고한다.

### 9.1 기전·진단 주장 (추천안 1)
- **기전 주장을 좁힌다**: "계획기가 반복에 걸쳐 지속되는 오차를 이용한다."
  - 같은 평가 안의 winner's curse(sel_last)는 새로 뽑는 잡음에도 생긴다. 그래서 기전의 증거로 쓰지 않는다.
- **진단 주장을 뺀다**: "OEI가 1-step 오차보다 손상을 더 잘 예측한다"는 쓰지 않는다(H2b 불성립).
  - 진단 기여는 H1(1-step 오차가 손상을 보지 못함)과 계획 편차로 한다.
- **TD-MPC2 E1 H2a 3칸의 독립 draw판 재측정**(사후 정정으로 표시)
  - 칸: mw-push k=3, mw-pick-place k=3, cheetah-run k=5. 새로 뽑는 변형 3개(ind, sh-iter, det+shift)를 E1과 같은 seed·job으로 다시 돈다. 810 episode다.
  - 재실행 episode는 기존 필드 전부가 E1과 비트 단위로 같아야 한다. 같지 않으면 그 칸은 보고만 하고 판정하지 않는다.
  - `oei_indep`로 H2a 규칙(v4 6.2)을 다시 계산한다. det, sh-step, sh-ep는 구성상 oei_indep = oei이고, none은 e = 0이다.
  - 결과는 "E1 뒤 정정한 측정"으로 동결 판정 옆에 함께 싣는다. 동결 판정을 대신하지 않는다.
- **E2 덧붙임 층 (b)**: E1 결과 뒤에 더한 사전등록이라고 표시한다. 층 (a)(7.1·8.2절)를 대신하지 않는다.
  - **자료**
    - TD-MPC2는 E2 seed(40000~)만 쓴다.
    - DINO-WM PushT는 E1 k=4 칸을 쓴다. 이 가설은 TD-MPC2 자료만 보고 세웠고 DINO-WM 분해는 이 고정 전까지 계산하지 않았으므로, 표본 밖 검정이다.
  - **(b1) H2a_indep**
    - 과제마다 det `oei_indep`의 95% CI 상한 < 0이고, sh-iter `oei_indep`의 95% CI가 0을 포함하면 그 과제는 성립이다.
    - 과제 4개(mw-push, mw-pick-place, cheetah-run, PushT) 중 3개 이상이면 성립이다.
    - PushT는 동결 정의 `oei_ref_mf`가 곧 이 판이다.
  - **(b2) H2a′(지속 오차 이용)**
    - 과제마다 두 조건이 모두 맞으면 그 과제는 성립이다. 과제 4개 중 3개 이상이면 H2a′ 성립이다.
      1. drift_det의 95% CI 상한 < 0
      2. r = drift_sh-iter / drift_det의 95% CI 상한 < 0.5. 칸 평균의 비를 재표본마다 다시 계산하고, drift_det* ≥ 0인 재표본은 버린다.
    - drift 정의
      - TD-MPC2: l_e_rand − f_e_rand. E2 base에서 잰다.
      - DINO-WM: last_pop_dJ − it0_rand_dJ
    - 양성 대조(보조, 판정 밖): 지속되는 확률 표의 r은 CI 하한 > 0.5로 예상한다. TD-MPC2는 sh-step·sh-ep, DINO-WM은 sh-plan이다.
    - 기술 보고: ind, det+shift의 r. 같은 draw판과 독립 draw판의 sel_last
  - **(b3) 영역 효과 대조**(기술 보고, 판정 밖)
    - 같은 상태에서 비양자화 계획기의 첫·마지막 모집단을 양자화 모델로 평가해 drift_ref를 얻는다.
    - TD-MPC2는 E2 base에만 둔다. DINO-WM은 E1 none 변형 모집단으로 오프라인 계산할 수 있으면 한다.
    - 구현과 스모크가 E2 시작 전에 끝나지 않으면 하지 않고 편차로 적는다.
  - **H2c′는 두지 않는다.** E4 유도에 따르면 drift는 N에 1차로 무관하다. 대신 8.2절 확증 sweep에서 N에 따른 sel_last·drift를 기술 보고한다.

### 9.2 DMC (추천안 2)
- 설계 7장 위험 표의 대응을 그대로 적용한다. 규칙은 7.2절 "DMC 설명"이다.
- E3 판정이 "설명 안 됨"이면 처방 주장을 TD-MPC2 Meta-World와 DINO-WM으로 좁힌다. DMC는 해로운 칸(walker k=3)을 포함해 한계로 보고한다.
- "설명됨"이나 "부분 설명"이어도 DMC의 H3a 불성립은 그대로다. 처방 주장은 Meta-World와 DINO-WM에 대해 하고, DMC 결과는 그 설명 변수(지속 편향 비중, near-tie 밀도)로 서술한다.


## 10. v5.3 덧붙임 (2026-10-24T02:38Z, 사용자 승인; E5·E7 TD-MPC2 결과를 본 뒤)
- 1~9절의 사전등록 판정은 그대로 남는다.
- 아래 10.2·10.3의 두 점검은 E5·E7 결과를 본 뒤 정한 **사후 점검**이다. 어떤 판정도 바꾸거나 대신하지 않고, 보고에 "사후"로 표시한다.
- 실행 세부는 아래에 고정한 뒤 돌린다.

### 10.1 TD-MPC2 QAT의 보고 방식
- v5 §7.3 QAT 규칙의 결과는 그대로 보고한다. 조건에 해당하는 칸은 0/11이다.
- 단서를 함께 단다.
  - 사전등록한 QAT 설정(5,000 step, batch 256, lr 1e-4, QAT 자료 20 episode)에서는 11칸 모두 QAT+det의 손상이 B+det보다 컸다. 그래서 이 QAT는 약한 기준선이다.
  - 그러므로 "QAT로는 손상이 사라지지 않는다"는 결론은 내리지 않는다.
- dither 처방은 학습 없는 처방으로 서술한다.
- DINO-WM QAT(E5_dinowm_qat)도 결과가 같은 양상(QAT+det가 B+det보다 나쁨)이면 같은 단서를 단다. DINO-WM은 QAT+none이 이미 실행 목록에 있다.

### 10.2 사후 점검 1: TD-MPC2 QAT+none
- 목적: QAT 학습이 양자화와 무관하게 모델 자체를 나쁘게 만들었는지 확인한다.
- 내용
  - E5 QAT 가중치(11칸 × 체크포인트 3개)를 활성 양자화 없이(none) 평가한다.
  - seed는 E5 평가 분할 60000~60029(90짝)다.
  - 원래 가중치의 E5 none과 짝을 맞춰 D(QAT+none)와 짝 bootstrap CI를 보고한다.
- 문턱 없이 기술로만 보고한다.
- MPS를 써도 된다.

### 10.3 사후 점검 2: E7 지연 재측정
- 목적: 첫 측정은 block 지연이 두 모드였다(17%의 block이 약 35% 느림). 그래서 회전별 오버헤드가 −30 ~ +38%로 흔들렸다. 같은 비트의 det·dither 지연 차를 더 안정적으로 잰다.
- 내용
  - 첫 측정과 같은 kernel·과제·반복 수 격자(96조건)를 쓴다. 규약도 같다: 배타 GPU(MPS 없음), 단일 프로세스, CUDA event, eager, warmup 50 + 200 step block.
  - 회전은 10회로 늘린다.
  - 회전마다 같은 (과제, 반복 수, kernel 비트)의 det block과 dither block을 바로 이어 잰다. 짝 안의 순서는 무작위로 정하고, 짝들의 순서도 회전마다 섞는다.
  - 짝마다 오버헤드 비 (dither − det)/det를 구한다. 조건마다 짝 오버헤드의 중앙값과 bootstrap 95% CI를 보고한다.
  - 첫 측정의 사전 정의 계산(합친 step 중앙값의 비)도 같은 자료로 함께 낸다.
  - block마다 SM clock, 호스트 부하, 외부 프로세스 수를 기록한다.
- H5 판정은 첫 측정의 사전 정의 계산 그대로다. 재측정은 기술로 보고한다.

### 10.4 실행 순서
- 두 점검 모두 DINO-WM v5 실행이 끝난 뒤 서버에서 돈다.
- QAT+none은 GPU 0의 DINO-WM 사슬이 끝나 GPU 0이 비면 시작한다.
- 지연 재측정은 MPS를 멈춘 뒤, 두 DINO-WM 사슬이 모두 끝나 GPU가 배타일 때 한다.
