# A TMLR 추가 실험 설계 (E11~E17) — v1.0 (고정판)

- 작성: 2026-11-22(KST), 조정자. 초안 v0.1을 사용자가 2026-11-21T17:52Z 무렵(한국 시간 11-22 02:52) 승인했다(10절). 이 판을 고정하고 sha256과 시각을 [LOG.md](LOG.md)에 적는다. 어떤 자료도 이 고정 전에 생기지 않았다.
- 배경
  - 2026-11-22 사용자 결정: A를 TMLR에 낸다. 추천 범위(E11~E17)로 설계 초안을 쓴다. **실험은 모두 RunPod에서 병렬로** 돌리고 비용을 조금 줄인다.
  - 왜 이 실험들인가는 [PLAN_tmlr.md](PLAN_tmlr.md) 2.2절(W1~W16)과 3절에 있다. 이 문서는 무엇을 어떻게 재고 어떻게 판정하는지를 정한다.
- 기준 문서: [../main/DESIGN_main.md](../main/DESIGN_main.md) v5.3(통계·판정 규약), [../rss_ext/DESIGN_ext.md](../rss_ext/DESIGN_ext.md)(RunPod 실행 규약), [../main/reports/FINAL_A.md](../main/reports/FINAL_A.md)
- 원칙
  - 가설, 지표, 문턱, 칸, seed는 이 문서가 정한다. 작업자는 운영 세부(스모크 결과, 처리량, 워커 수, pod 재배치)만 실험 폴더의 `PLAN.md`에 본 실행 전에 적는다.
  - 결과를 본 뒤 정한 분석은 [사후]로 표시한다. 주장을 바꾸는 일은 사용자 승인으로만 한다(`CLAIMS_tmlr.md`).
  - 문서는 한국어, 코드와 주석은 영어로 쓴다.
- 9절의 결정 항목은 사용자가 10절대로 정했다.

---

## 0. 공통

### 0.1 장비와 비용

- **RunPod만 쓴다**(사용자 지시). 도구는 `runpod_rss/`를 복사한 `runpod_tmlr/`(pod 접두어 `atm`)이다. 다른 접두어의 pod는 건드리지 않는다.
- 가격과 재고(2026-11-22 조회, `runpodctl gpu list`)

| GPU | secure $/h | community $/h | 재고 | 이 설계에서의 쓰임 |
|---|---|---|---|---|
| RTX A5000 24GB | 0.27 | 0.16 | Low | 317M, DINO-WM PointMaze·Wall. 본 실험(E1~E7)의 서버와 같은 기종 |
| A40 48GB | 0.59 | — | High | TD-MPC2 5M(E9에서 처리량을 잰 기종), 지연 측정(E8·E8c와 같은 기종) |
| RTX 4090 24GB | 0.89 | 0.34 | Low | DINO-WM PushT(가장 무거운 칸), A5000이 없을 때의 대체 |
| RTX 3090 24GB | 0.50 | 0.22 | Low | 대체 |

- 처리량의 근거(지난 실행의 측정)
  - TD-MPC2 5M, A40, MPS 워커 8: Meta-World 5,995 ep/h, DMC 1,163 ep/h(E9 PLAN §3). pod의 CPU 할당은 7.65코어였다.
  - TD-MPC2 317M, RTX 4090, 워커 1: episode 8.0초(E10 LOG). A5000은 그 약 2배로 잡는다.
  - DINO-WM, episode당 초(E0 보고 §4): A5000 단독 PushT 30~38, PointMaze 10~16, Wall 4~5. RTX 4090은 PushT 14~17, Wall 2~2.5.
- **비용 상한: $80(사용자 결정, 10절). 감시 스크립트는 추정 지출 $77에서 모든 `atm` pod를 지운다.** 예상 지출은 약 $42다(8절). 계정 잔액은 2026-11-21T17:56Z에 $83.42였다.
- 태그별 상한(`runpod_tmlr/caps.json`): `td`(TD-MPC2 5M) $24, `mt`(317M) $12, `k`(정수 kernel) $6, `dw`(DINO-WM) $35. 합 $77.
- pod는 SECURE 클라우드에서만 만든다(10절).
- pod는 생성 뒤 정해진 시간에 자동 종료된다(`TERM_H`): td 8, k 8, mt 9, dw 10시간. 맥북이 멈춰 감시가 끊겨도 지출이 상한을 넘지 않게 고른 값이다. 대기열이 끝난 pod는 결과를 당겨 온 직후 지운다. 자동 종료 전에 끝나지 않으면 같은 GPU 종류의 새 pod에서 이어 돈다.
- 맥북에서는 계산하지 않는다. 분석도 pod에서 돌리고 결과 파일만 당겨 온다.

### 0.2 소프트웨어 고정

- TD-MPC2(E1·E7·E9와 같음): torch 2.7.1(cu126)과 딸린 Triton, tensordict 0.8.3, dm-control 1.0.16, mujoco 3.1.2, numpy 1.24.4, gymnasium 0.29.1, gym 0.21.0, Metaworld@04be337, mujoco-py 2.1.2.14(CPU 빌더), tdmpc2 commit `332f19e`. 설치는 E9의 `setup_e9.sh`(약 3분)를 쓴다.
- DINO-WM(E1과 같음): torch 2.4.1(cu124), numpy < 2, gym 0.23.1, hydra-core 1.3.2, pymunk 6.2.1, dino_wm commit `0a9492f`, DINOv2 ViT-S/14. PointMaze는 MuJoCo 2.1.0 + mujoco-py 2.1.2.14 + d4rl@89141a6. 설치는 파일럿 `setup13.sh`와 `setup_pointmaze.sh`를 합친다. 체크포인트와 데이터는 OSF에서 받고 sha256을 E0 기록과 대조한다.
- 317M: E10의 `setup_pod_e10.sh`, 체크포인트 `mt80-317M.pt`(sha256 `38efd60b…99cc`).

### 0.3 통계 (DESIGN_main v4 §6.1, DESIGN_ext §0.3과 같음)

- D(손상): 같은 짝의 none 대비 하락. Meta-World와 DINO-WM은 성공률 차(비율), DMC는 상대 return 하락 1 − ΣR_v / ΣR_none(재표본마다 비율).
- 짝: TD-MPC2는 같은 (체크포인트, episode seed), DINO-WM은 같은 (시작·목표 쌍, 계획기 seed). 짝 bootstrap 10,000회, 95% 백분위 CI, `numpy.random.RandomState(0)`.
- 손상 실재: 단측 bootstrap p < 0.05, 실험마다 정한 칸 집합 안에서 Holm.
- red = (D_det − D_v)/D_det(점 추정으로 판정, CI 보고). "고친다"는 red ≥ 0.5, "못 고친다"는 red < 0.5다.
- 해로움: D_v − D_det의 CI 하한 > 0.05.
- 문턱 비교의 허용 오차 1e-9.
- **짝은 한 pod 안에서만 맺는다.** 한 (과제, 체크포인트)의 none과 모든 변형은 한 pod(한 GPU)에서 돈다. DINO-WM은 한 (과제, k) 칸의 none과 모든 변형이 한 pod에서 돈다. GPU가 다르면 같은 seed라도 수치가 비트 단위로 같지 않기 때문이다(E9 PLAN §2.3).

### 0.4 seed, 새 평가 분할, 재사용 규칙

- TD-MPC2 5M: eval episode seed **100000~100029**(체크포인트 1·2·3 → 90짝). 이 구간을 E11·E12·E15·E16이 함께 쓴다. held(E13 입력) seed 101000~101029.
- 317M: eval seed **102000~102089**(체크포인트 1개 → 90짝), 보정용 held seed 103000~103009.
- DINO-WM: 새 평가 분할 **eval8**. pair seed 8, 쌍 100개, 계획기 seed 5000 + e, dither salt `eval8`. PushT는 E1의 eval6처럼 제약 표집으로 held·smoke·repro·eval6의 프레임과 겹치지 않게 뽑는다. 지금까지 쓴 pair seed는 1, 2, 3, 4, 5, 6, 7, 11이다.
- 모든 seed 구간은 E1~E10과 겹치지 않는다. 그래서 E1·E10과 같은 조건의 칸은 독립 seed 복제가 된다.
- **재사용 규칙**(같은 pod, 같은 seed라서 그대로 겹친다. 비용 절감의 중심이다)
  - R1: E12의 칸 가운데 E11에 이미 있는 칸(per-tensor 범위, 1차 k의 det·det+BC·sh-iter·det+shift)은 다시 돌리지 않는다.
  - R2: E15의 반복 6회 칸은 E11의 DMC 칸(none·det·sh-iter·det+shift)이다.
  - R3: E16의 기본 계획기 기준은 E11의 칸이다.
  - R4: E14는 E10의 범위와 1차 k를 그대로 쓴다(k 선택을 다시 하지 않는다).
  - R5: none은 (과제, 체크포인트, 계획기 설정)마다 pod에서 한 번만 돌리고 그 pod의 모든 실험이 함께 쓴다.

### 0.5 변형의 정의

격자: 층별 범위 [lo, hi], nl = 2^k − 1, Δ = (hi − lo)/nl, x = (clip(a, lo, hi) − lo)/Δ. W는 층의 단위 수다.

| 변형 | 출력 | 난수 | 상태 |
|---|---|---|---|
| det | lo + Δ·clamp(round(x), 0, nl) | — | 기존 |
| sh-iter | lo + Δ·clamp(⌊x + u⌋, 0, nl) | u ~ U[0,1), (1, W). 후보 공유, forward마다 새로 | 기존 |
| sh-ep (DINO-WM은 sh-plan) | sh-iter와 같음 | u를 (층, horizon 위치)마다 한 줄, episode(계획)마다 한 번 | 기존 |
| ind | sh-iter와 같음 | u를 항목마다 독립, forward마다 새로 | 기존 |
| det+shift | lo + Δ·(clamp(round(x − o), 0, nl) + o) | o ~ U[−½, ½), (1, W). 후보 공유, forward마다 새로 | 기존 |
| **det+shift-fixed** | det+shift와 같음 | o를 (층, horizon 위치)마다 한 줄, episode(계획)마다 한 번(sh-ep와 같은 일정) | 새로 |
| **det+BC** | lo + Δ·clamp(round(x), 0, nl) − b | b는 0.6절의 편향 벡터(층별, 단위별, 상수) | 새로 |
| **sh-iter+b** | lo + Δ·clamp(⌊x + u⌋, 0, nl) + b | u는 sh-iter와 같음. b는 같은 (범위, k)의 det+BC의 b | 새로 |
| **det+noise** | lo + Δ·(clamp(round(x), 0, nl) + n) | n = u − ½, u는 sh-iter와 같은 일정 | 새로 |

- 새 변형의 뜻
  - det+BC: det의 오차에서 단위별 평균만 뺀다. 입력에 따라 정해지는 나머지 오차는 그대로 남는다.
  - det+shift-fixed: det+shift와 원소별 오차 분포가 같고, 반복 사이의 지속만 다르다.
  - sh-iter+b: 새로 뽑는 dither에 det의 평균 편향만 되돌려 넣는다.
  - det+noise: det의 오차는 그대로 두고 새 잡음만 더한다. 전체 오차 분산은 sh-iter와 같다(Δ²/6).
- 난수는 E1과 같이 episode마다 seed한 개별 CUDA generator에서 뽑는다. 계획기의 난수열은 변형 사이에 같다.
- DINO-WM은 예측기 MLP의 GELU 출력 6층에 같은 정의를 쓴다. u와 o의 모양과 일정은 기존 `shiter`·`shstep`·`detshift`와 같고, b는 (층, 단위)별 벡터를 token과 후보가 함께 쓴다.
- 양자화 위치는 E1과 같다. TD-MPC2는 dyn0·dyn1·rew0·rew1의 Mish 출력, 가중치·encoder·Q·π는 FP32다(E17만 정수 kernel).

### 0.6 범위와 편향 벡터

- **per-tensor 범위(PT)**: 저장된 E0 범위 파일을 그대로 쓴다(`main/results/E0_tdmpc2/jobs/ranges/`, `main/results/E0_dinowm/ranges/ranges_main_*.json`). E1·E9와 같은 격자다.
- **per-channel 범위(PC)**: 단위별 [min_j, 99.99 백분위_j]. E5의 `perchan`(TD-MPC2)·`pc`(DINO-WM) 정의다. E5의 통계 파일은 서버에만 있으므로 pod에서 다시 계산한다.
  - TD-MPC2: E0와 같은 규약의 보정 episode 0~4를 pod에서 다시 기록하고, 계획 시점 입력 12,000행 × horizon 3 step의 활성에서 구한다.
  - DINO-WM: 보정 창 256개.
  - 점검: 같은 자료로 다시 잰 per-tensor 범위와 E0 범위의 차를 기록한다.
- **편향 벡터 b**(det+BC, sh-iter+b): (과제, 체크포인트, 범위 종류, k)마다 하나. 보정 자료의 후보 rollout에서 구한다.
  1. b = 0에서 시작한다.
  2. 지금의 b로 보정한 det 양자화 모델과 양자화하지 않은 모델을 같은 시작 상태·같은 행동열로 굴린다(TD-MPC2 3 step, DINO-WM 5 step).
  3. 층마다, 모든 행과 모든 step에 걸친 (보정된 양자화 출력 − 비양자화 출력)의 단위별 평균을 b에 더한다.
  4. 2~3을 세 번 한다. 번마다 남은 평균을 기록한다.
  - 이렇게 하면 앞 층과 앞 step의 양자화가 만든 평균 이동까지 보정된다. det+BC를 약한 기준선으로 두지 않기 위해서다.
- **317M**: 범위와 1차 k는 E10의 저장값이다. b는 과제마다 held none 10 episode에서 구한다.

### 0.7 본 실행 전 점검 (하나라도 실패하면 멈추고 조정자에게 알린다)

1. 코드 경로: 손대지 않은 E1 `qlib.py`와 새 `qlib.py`가 같은 pod에서 none·det·sh-iter·sh-ep·det+shift episode를 비트 단위로 같게 낸다(E9와 같은 점검).
2. 새 변형의 단위 점검
   - det+BC: 세 번째 sweep 뒤 남은 단위별 평균의 절댓값 평균이 Δ의 1% 미만
   - det+shift-fixed: 한 episode 안에서 표가 바뀌지 않고 episode마다 바뀐다. 원소별 오차의 평균·분산이 det+shift와 같다(상대 차 2% 안)
   - sh-iter+b: 원소별 오차의 평균이 b, 분산이 sh-iter와 같다
   - det+noise: 더한 잡음의 분산이 Δ²/12(상대 차 2% 안), det 부분은 det와 비트 단위로 같다
   - 모든 새 변형에서 계획기의 전역 난수가 바뀌지 않는다
3. 기준 성능: 새 seed의 none 성공률(return)이 E1의 none에서 10점(10%) 안이다. DINO-WM은 새 분할 eval8에서 본다.
4. 처리량: pod 종류마다 5~10분 재서 워커 수를 정하고, 8절의 시간·비용 추정을 고친다. 고친 추정 총액이 $57을 넘으면 본 실행 전에 조정자가 사용자에게 알린다.

### 0.8 산출물

- `tmlr_ext/<실험>/`: `PLAN.md`(본 실행 전), `code/`(돌린 코드 전부), `results/`(JSON·CSV·작은 PNG), `LOG.md`, `REPORT.md`(한국어)
- 판정 수치는 결과 파일에서 다시 계산할 수 있어야 한다. 조정자가 원자료에서 다시 계산해 작업자 보고와 대조한다.

---

## 1. E11 — 편향·해상도·지속을 가르는 대조

### 1.1 물음

det 손상은 (a) 단위별 평균 편향이 rollout을 따라 쌓여서, (b) 격자 간격보다 작은 변화가 결정적 반올림에서 사라져서, (c) 입력에 따라 정해지는 오차장이 계획기 반복 사이에 남아 계획기가 이용해서 생기는가.

### 1.2 설정

- TD-MPC2 5M: 12과제 × E1의 1차 k(push·pick-place·drawer-open·button-press·sweep-into·quadruped·walker 3, door-open·peg-insert·window-open·finger 4, cheetah 5) × per-tensor 범위.
  - 변형 8개: det, det+BC, sh-ep, det+shift-fixed, sh-iter+b, sh-iter, det+shift, det+noise. 그리고 none.
  - 90짝(체크포인트 3 × seed 100000~100029). 9,720 episode.
  - 기전 과제 mw-push, mw-pick-place, cheetah-run에는 E2와 같은 계측(PlanProbe: drift, 선택 항, 독립 draw 재평가)을 켠다.
- DINO-WM: PushT 4-bit, PointMaze 3-bit, Wall 3-bit × per-tensor 범위(main) × 같은 변형 8개(sh-ep 자리는 sh-plan) + none, 분할 eval8의 100 episode. 2,700 episode.
  - 계획기는 E1과 같다(CEM, 후보 300, top-30, horizon 5, 반복 PushT 30·Wall 10·PointMaze 10).
  - episode마다 E1과 같은 진단(1·3·5-step 오차, OEI 입력)을 기록한다.

### 1.3 설명별 예측 (판정의 틀)

| 설명 | det+BC | 고정 무편향(sh-ep·sh-plan, det+shift-fixed) | sh-iter+b | sh-iter, det+shift |
|---|---|---|---|---|
| (a) 평균 편향 | 고친다 | 고친다 | 해롭다 | 고친다 |
| (b) 해상도 손실 | 못 고친다 | 고친다 | 고친다 | 고친다 |
| (c) 지속 이용 | 못 고친다 | 못 고친다 | 고친다 | 고친다 |

### 1.4 판정

- 손상 칸: det 하락이 실재하는 칸(단위 안에서 Holm). 단위는 Meta-World(8칸), DMC(4칸), DINO-WM(3칸)이다.
- 1차 대비
  - Δ_bias = D(det) − D(det+BC)
  - Δ_pers = D(det+shift-fixed) − D(det+shift)
- 단위별 읽기(손상 칸의 과반 기준)
  - **편향 지배**: det+BC가 고친다
  - **지속 지배**: det+BC가 못 고치고, Δ_pers의 CI 하한 > 0
  - **해상도 지배**: det+BC가 못 고치고, det+shift-fixed가 고치며, Δ_pers의 CI가 0을 포함
  - 그 밖: **혼합**
- 가설
  - **B1**: Meta-World는 지속 지배다.
  - **B2**: DINO-WM은 지속 지배가 아니다(편향 지배나 해상도 지배).
  - **B3**: det+noise는 어느 단위에서도 손상 칸의 과반을 고치지 못한다.
  - **B4(E1 H3a의 독립 복제)**: Meta-World 손상 칸의 75% 이상에서 sh-iter의 red ≥ 0.5이고 해로운 칸이 없다.
- 기술 항목
  - 보조 대비 D(sh-ep) − D(sh-iter), D(sh-iter+b) − D(sh-iter), D(det+noise) − D(det)
  - 기전 과제의 drift 비(변형 ÷ det)
  - 변형별 1·3·5-step 잠재 오차(E13의 정의)
  - det+shift 대 sh-iter(D의 짝 차), DMC 단위의 읽기, Wall 3-bit의 E1 탐색 칸 복제 여부

### 1.5 결과의 쓰임 (미리 적음)

| 결과 | 원고 |
|---|---|
| B1·B2 성립, DINO-WM 편향 지배 | 기전을 두 성분으로 쓴다: rollout을 따라 쌓이는 평균 편향(bias correction이나 어떤 dither로도 없어짐)과 반복 사이에 지속되는 오차장(새로 뽑아야 없어짐) |
| B1·B2 성립, DINO-WM 해상도 지배 | 첫 성분을 "결정적 반올림이 지우는 격자 아래의 정보"로 쓴다. bias correction은 처방이 아니라고 쓴다 |
| B1 불성립(Meta-World가 편향 지배) | "계획기 이용" 주장을 양성 대조(고정된 무편향 오차도 해롭다)의 범위로 좁히고, 처방에 bias correction을 넣는다. 제목을 다시 정한다 |
| B2 불성립(DINO-WM도 지속 지배) | 지금의 일반 문장을 두 계열에 쓴다 |
| B4 불성립 | E1의 처방 수치와 함께 복제 실패를 그대로 보고하고 처방 주장을 줄인다 |

---

## 2. E12 — 비트 한계

### 2.1 물음

범위를 잘 잡은 양자화기 위에서도 dither가 "쓸 수 있는 최소 비트"를 낮추는가.

### 2.2 설정

- 범위: PT(per-tensor), PC(per-channel). DINO-WM PushT에는 RC(rollout으로 고른 범위)를 더한다.
  - RC: 층별 상한을 백분위 격자 {99, 99.9, 99.99, 99.999, max}에서 좌표 하강 두 번으로 골라, held 쌍의 5-step 잠재 상대 오차(det, 그 k)를 최소로 한다. WM-Quant의 rollout 전체에 맞춘 범위 보정이라는 생각을 우리 코드로 다시 구현한 것이다. 그쪽의 다른 구성요소(encoder 정밀도, 제어 경로, 가중치 양자화)는 넣지 않는다.
- 반올림: det, det+BC, sh-iter, det+shift(DINO-WM은 det, det+BC, sh-iter. RC는 det, sh-iter).
- 처음 도는 k 격자

| 단위 | PT | PC |
|---|---|---|
| Meta-World 8과제 | 2, 3, 4, 5 | 1, 2, 3 |
| DMC 4과제 | 3, 4, 5 | 2, 3, 4 |
| PushT | 2, 3, 4 | 2, 3, 4 (RC는 3, 4) |
| PointMaze, Wall | 2, 3 | 2, 3 |

- **확장 규칙**: (과제, 범위, 반올림)마다 격자의 가장 낮은 k에서 D ≤ τ면 한 비트씩 내려 더 돈다(k = 1까지). 가장 높은 k에서 D > τ면 한 비트씩 올려 더 돈다(k = 8까지). τ는 0.10(성공률 10점, DMC 10%)이다.
- **두 단계 표본**(비싼 칸에만): DMC 칸은 45짝(seed 100000~100014), PushT 칸은 50 episode를 먼저 돈다. 그 D의 95% CI가 τ를 포함하면 나머지(45짝, 50 episode)를 더 돈다. Meta-World·PointMaze·Wall은 처음부터 전부 돈다. 칸의 D는 그 칸이 가진 모든 짝으로 계산한다.
- E11과 겹치는 칸은 다시 돌리지 않는다(R1).

### 2.3 지표와 판정

- k\*(범위, 반올림) = D ≤ τ인 가장 작은 k. 그보다 큰 모든 잰 k에서도 D ≤ τ여야 한다.
- 번 비트 G(범위) = min(k\*(det), k\*(det+BC)) − min(k\*(sh-iter), k\*(det+shift)).
- **Fr1**: PT에서 Meta-World 과제의 75% 이상이 G ≥ 1이다.
- **Fr2**: PC에서 Meta-World 과제의 50% 이상이 G ≥ 1이다.
- 기술 항목: 과제별 전체 최선 구성(가장 낮은 k\*와 그 범위·반올림), DMC·DINO-WM의 G, RC 대 PT·PC, 비트–성공률 곡선. 두 단계 표본을 쓴 칸의 CI는 중간 확인을 보정하지 않은 값이라고 밝힌다.

### 2.4 결과의 쓰임

- Fr2 성립: "강한 범위 보정과 합쳐도 한 비트를 번다"를 처방 주장에 넣는다.
- Fr2 불성립, Fr1 성립: 처방을 "per-tensor 활성 격자(정수 추론의 표준)에서의 무학습 처방"으로 한정한다.
- 둘 다 불성립: 처방 문장을 내리고 dither를 기전을 드러내는 도구로 쓴다.

---

## 3. E13 — held 상태 진단 (오프라인)

### 3.1 물음

같은 비트 안에서 반올림 방식 사이의 손상 차이를 어떤 오프라인 오차가 보는가.

### 3.2 설정

- 입력: 각 pod에서 none으로 돈 held episode(TD-MPC2 seed 101000~101029, 과제당 90 episode; DINO-WM은 held 쌍 20개와 eval8의 none 실행)의 계획 시점 후보 집합. 모든 변형이 같은 입력을 쓴다.
  - 첫 반복 후보 집합과, none 계획기의 마지막 반복 후보 집합(near-tie 영역) 둘 다 쓴다.
- 대상: E11의 모든 칸과 변형, E12의 1차 k 주변 칸. E1의 36칸 × 7변형도 같은 held 상태에서 E1 격자 그대로 다시 계산한다(환경 rollout 없음). 그러면 E1의 오차 비교가 변형별 자기 상태가 아니라 공통 상태 기준이 된다.
- 지표
  - 잠재 오차 1·3·5-step(E1 정의: 오차 제곱합 ÷ 비양자화 예측의 분산)
  - 보상 오차: |r_q − r_0|을 그 후보 집합의 보상 표준편차로 나눈 값의 중앙값. E1의 `errr`(평균 제곱 ÷ 분산)는 분산이 작은 상태에서 폭주하므로 쓰지 않는다
  - 후보 return 오차 e(a): 후보 공통 몫, 후보 사이 표준편차 σ_e, 후보 return 표준편차 σ_J에 대한 비 σ_e/σ_J
  - 지속 몫 ρ_p: 같은 후보 집합을 계획기의 연속한 두 반복에 해당하는 난수 상태로 두 번 평가했을 때 e(a)의 상관
  - 순위 보존: 양자화·비양자화 return의 Spearman, elite(TD-MPC2 64개, DINO-WM 30개)의 겹침

### 3.3 판정

- 기술 보고가 중심이다.
- **Dg1**(사전 지정 비교 하나): E11의 손상 칸 안에서 변형 8개의 D 순위를 "ρ_p × σ_e/σ_J"(마지막 반복 후보 기준)와 잠재 1-step 오차가 각각 얼마나 맞히는지 칸 안 Spearman으로 재고, 단위별 평균 차가 0.3 이상이면 성립이다.
- 쓰임: 성립하면 "배치 전에 재야 할 것"으로 본문에 쓴다. 불성립이면 표로만 싣고 진단 주장을 하지 않는다.

---

## 4. E14 — 317M의 기전 대조

- **물음**: 폭 4096의 다중 과제 모델에서도 고정 dither가 det만큼 해로운가.
- **설정**: mt80-317M, E10의 8과제, E10의 1차 k와 범위. 변형 none, det, sh-iter, sh-ep, det+shift, det+BC. 90 episode(seed 102000~102089). 4,320 episode.
- **보고**: 과제별 r = D(sh-ep)/D(det)와 CI, det+BC·det+shift의 red.
- **가설**
  - **P1(E10의 독립 복제)**: 8과제 중 6과제 이상에서 det 하락이 실재하고, 그 과제의 75% 이상에서 sh-iter의 red ≥ 0.5다.
  - **P2**: 손상 과제의 과반에서 D(sh-ep) − D(sh-iter)의 CI 하한 > 0이다(지속의 비용이 큰 모델에서도 있다).
- **쓰임**: P2 성립이면 기전이 크기에 걸쳐 성립한다고 쓴다. 불성립이면 "단위가 많을수록 고정 오차가 평균된다"는 설명(표 크기, H4a)의 근거로 쓴다.
- E10 결과는 다시 쓰지 않는다. 새 seed와 다른 GPU에서의 복제가 목적의 일부다.

---

## 5. E15 — DMC × 반복 수

- **물음**: DMC에서 MPPI 반복을 늘리면 dither가 det 손상의 절반 이상을 없애는가.
- **설정**: DMC 4과제 × 1차 k × 반복 {6, 12, 24} × {none, det, sh-iter, det+shift}. 반복 6은 E11의 칸이다(R2). 반복 12는 90짝, 반복 24는 60짝(seed 100000~100019)이다. none은 반복 수마다 따로 돈다. 다른 계획기 설정은 기본값이다.
- **가설**
  - **It1**: 반복 12에서 sh-iter와 det+shift 중 나은 쪽이 4과제 중 3과제 이상에서 red ≥ 0.5이고, 해로운 과제가 없다.
  - **It2**: 3과제 이상에서 그 red가 반복 6 < 12 < 24로 단조 증가한다.
- 기술 항목: 반복 수별 none의 return과 결정당 계획 시간, "det + 높은 비트" 대 "dither + 낮은 비트 + 더 많은 반복"의 return–시간 비교(E12의 값을 함께 씀).
- **쓰임**: It1 성립이면 DMC를 "기본 반복 수로는 새 잡음을 평균할 반복이 모자란 경계"로 쓴다. 불성립이면 DMC는 한계로 남는다.

---

## 6. E16 — 계획기 교차

- **물음**: "지속이 문제가 되는가"의 계열 차이는 모델에서 오는가 계획기에서 오는가.
- **TD-MPC2를 CEM식으로**: temperature 0(elite 균등 가중), min_std 0.01, 정책 궤적 0. 그 밖은 기본값. mw-push, mw-pick-place, cheetah-run × 1차 k × {none, det, sh-iter, sh-ep, det+shift} × 90짝.
- **DINO-WM을 MPPI식으로**: top-30의 평균 대신 softmax(−J/λ) 가중 평균과 가중 표준편차, 표준편차 바닥은 처음 값의 5%. λ는 held 쌍 20개에서 none 성공률이 가장 높은 값을 {0.25, 0.5, 1, 2} × (첫 반복 elite 비용의 표준편차)에서 고른다(동률이면 큰 값). PointMaze 3-bit, Wall 3-bit × {none, det, sh-iter, sh-plan, det+shift} × 100 episode.
- **자격**: 바꾼 계획기의 none 성공률(return)이 기본 계획기의 60% 이상인 과제만 판정 표에 넣는다.
- **보고**(기술, 판정 문턱 없음): D(sh-ep) − D(sh-iter)의 부호와 CI를 모델 × 계획기 2×2로 싣는다. 기본 계획기 쪽 값은 E11이다(R3).

---

## 7. E17 — 정수 kernel과 BF16 폐루프

- **폐루프 det+shift**: E7의 융합 정수 kernel(가중치 int8, 활성 코드는 int8 그릇에 담은 k bit)에 det+shift를 넣는다. offset은 층·forward마다 뽑고, 다음 층에서는 W·(Δ·o)를 bias에 더한다(호출당 행렬–벡터 곱 하나). mw-push, mw-pick-place, cheetah-run, walker-walk × 각 과제의 1차 k × {FP32, det, sh-iter, det+shift} × 90짝, 반복 6. k를 E11과 같게 두어 fake-quant와 바로 비교한다.
- **지연**: E8과 같은 규약(A40 한 대 배타, CUDA graph로 잡은 world-model rollout, 짝 block 10회전). 모델 크기 5M·317M × 후보 512·2048·8192. det 대 det+shift, det 대 sh-iter의 짝 오버헤드.
- **미리 바꾼 BF16의 폐루프**: 가중치를 한 번 BF16으로 바꾼 TD-MPC2 5M, 12과제 × 90짝.
- **per-channel scale을 접은 정수 구성**: per-channel 활성 scale Δ_j와 lo_j를 다음 층 가중치의 열과 bias에 접은 뒤 가중치를 출력 채널별 int8로 양자화한다. 위 4과제 × k {2, 3} × {det, sh-iter} × 90짝.
- **가설**
  - **K1**: det+shift의 rollout 지연 오버헤드가 모든 조건에서 중앙값 5% 이하다.
  - **K2**: 정수 kernel의 det+shift가 4과제 모두에서 fake-quant(E11)와 같은 방향이다(D(det) − D(det+shift)의 부호가 같고 CI가 겹친다).
  - **K3**: 미리 바꾼 BF16의 D가 12과제 모두에서 CI에 0을 포함한다(손상 없음).
- 기술 항목: 접은 per-channel 구성의 D와 E12의 fake-quant per-channel 칸의 차.
- **쓰임**: K1·K2 성립이면 det+shift를 공동 처방으로 올릴 근거가 된다(올릴지는 사용자 결정). K3은 "BF16을 폐루프로 재지 않았다"는 한계 문장을 대신한다.

---

## 8. 실행 계획 (RunPod 병렬)

### 8.1 pod 배치 (처리량 점검 뒤 `PLAN.md`에서 다시 맞춘다)

| pod | GPU(우선 → 대체) | 맡는 일 | 추정 시간 |
|---|---|---|---|
| td.1~td.4 | A40 | DMC 과제 하나씩(cheetah, walker, quadruped, finger): E11, E12, E15. td.1은 E16의 cheetah도 | 각 3.7~4.2 h |
| td.5, td.6 | A40 | Meta-World 4과제씩: E11, E12. td.5는 E16의 push·pick-place도 | 각 2.5~3 h |
| mt.1~mt.4 | RTX A5000 → RTX 4090(이때는 pod 2개) | 317M 2과제씩: E14 | 각 5.3 h(4090이면 5 h × 2대) |
| k.1 | A40 | E17(지연을 먼저, 배타로) | 4 h |
| dw.1 | RTX 4090 → RTX A5000 | PushT 4-bit: E11, E12의 4-bit 칸 | 5.7 h |
| dw.2 | RTX 4090 → RTX A5000 | PushT 3-bit·2-bit: E12(none 포함), 확장 칸 | 5.5 h |
| dw.3 | RTX A5000 → RTX 4090 | PointMaze 3-bit: E11, E12, E16 | 6.5 h(4090이면 3.3 h) |
| dw.4 | RTX A5000 → RTX 4090 | PointMaze 2-bit(E12), Wall 전부(E11, E12, E16) | 5.5 h |

- pod 15개가 한꺼번에 돈다. 가장 긴 줄이 약 6.5시간이라, 스모크와 설치를 합쳐 **하루 안에 실행이 끝난다.**
- RTX 4090을 받지 못하면 PushT 칸을 RTX A5000 pod 넷으로 나눈다. 칸 단위로 나누고 pod마다 none을 따로 돈다(A5000에서는 PushT 한 칸의 E11이 약 9시간이다).
- A5000과 4090이 모두 없어 A40만 쓸 수 있으면 예상 총액이 상한에 가까워진다. 그때는 pod를 띄우기 전에 사용자에게 알린다.
- E13은 각 pod가 자기 과제의 episode를 끝낸 뒤 이어서 돈다(TD-MPC2 pod당 약 0.3 h, DINO-WM pod당 약 0.5 h).
- 한 (과제, 체크포인트)와 한 DINO-WM 칸은 pod 사이에 나누지 않는다(0.3절).
- 순서: 첫 pod(td.5, dw.4, mt.1)에서 0.7절 점검 → 통과하면 나머지 pod 생성 → 본 실행. 대기열은 job마다 결과 파일을 쓰고, 이미 있으면 건너뛴다. 싼 칸과 1차 판정에 쓰이는 칸(E11)을 먼저 돈다.
- 조정자는 1시간마다 `runpod_tmlr/check_all.sh`로 점검하고 부분 결과를 당겨 온다. 감시 스크립트(`budget.py watch 300`)가 꺼져 있으면 다시 켠다.

### 8.2 예상 비용

| 묶음 | 계산 | 추정 |
|---|---|---|
| TD-MPC2 5M(E11, E12, E15, E16, E13) | A40 약 24 pod-h | $14.5 |
| 317M(E14) | A5000 약 21 pod-h | $5.9 |
| 정수 kernel(E17) | A40 약 4 pod-h | $2.4 |
| DINO-WM PushT(E11, E12) | RTX 4090 약 13 pod-h | $12.2 |
| DINO-WM PointMaze·Wall(E11, E12, E16, E13) | A5000 약 16 pod-h | $4.4 |
| 설치·스모크·유휴 | pod 15개 × 약 0.35 h | $2.9 |
| **합** | | **약 $42** |

- 재고 때문에 A5000 대신 RTX 4090(secure)을 쓰게 되면 317M이 약 +$3, PointMaze·Wall이 약 +$3이다. 그 경우 약 $48이다.
- 상한 $80 안의 여유는 확장 규칙의 추가 칸, 재실행, A40 대체에 쓴다.

### 8.3 비용을 줄인 장치

| 장치 | 줄어든 양(어림) |
|---|---|
| seed 구간을 함께 써서 E11의 칸을 E12·E15·E16이 다시 쓰는 규칙(R1~R3) | 약 6,500 episode, $3 |
| E14의 변형을 8개에서 6개로, 범위와 1차 k는 E10 값을 그대로 씀(R4) | $3 |
| E12의 k 격자를 알려진 한계 둘레로 좁히고 확장 규칙으로만 넓힘 | $3 |
| DMC·PushT 칸의 두 단계 표본, 반복 24회 칸의 60짝 | $3 |
| E11 PushT를 1차 k 한 칸으로, E16의 DINO-WM 쪽을 싼 과제(PointMaze, Wall)로 | $5 |
| 일에 맞는 GPU: 317M과 PointMaze·Wall을 RTX 4090 대신 A5000으로 | $4 |

- 처음 추정(전부 RunPod이면 $60~65)에서 약 $20를 줄였다. 항목끼리 겹치는 부분이 있어 어림값이다.
- 주장에 쓰이는 칸의 짝 수는 줄이지 않았다. E11과 1차 판정에 쓰이는 칸은 90짝·100 episode 그대로다.

### 8.4 멈추는 조건

- 0.7절 점검 실패
- 태그별 지출이 상한의 80%에 이르렀는데 그 태그의 본 실행을 끝내지 못할 것 같을 때
- 설계대로 잴 수 없을 때(예: 새 분할에서 none 성공률이 기준 밖, 바꾼 계획기의 none이 자격 미달)
- 멈추면 작업자는 선택지를 조정자에게 보고한다. 조정자는 8.6절의 절차로 정한다.

### 8.5 작업 분담과 단계

- 작업자 넷이 묶음 하나씩 맡는다. 공통 규칙은 [WORKER_RULES.md](WORKER_RULES.md)다.

| 묶음 | 폴더 | 실험 | pod |
|---|---|---|---|
| td | `TD_tdmpc2/` | E11·E12·E13·E15의 TD-MPC2 몫, E16의 TD-MPC2 몫 | td.1~td.6 |
| dw | `DW_dinowm/` | E11·E12·E13의 DINO-WM 몫, E16의 DINO-WM 몫 | dw.1~dw.4 |
| mt | `MT_317m/` | E14 | mt.1~mt.4 |
| k | `K_kernel/` | E17 | k.1 |

- 단계
  - A: 구현 → 첫 pod에서 0.7절 점검과 처리량 → 운영 계획(`PLAN.md`) → 나머지 pod 생성과 본 실행 시작 → 조정자에게 보고
  - 실행 중: 조정자가 30분마다 점검하고(`runpod_tmlr/check_all.sh`), 결과를 당겨 오며(`pull_all.sh`), 멈춘 대기열을 다시 띄운다
  - B: pod에서 분석 → `REPORT.md` → 마지막 회수 → pod 삭제
- 조정자는 판정 수치를 원자료에서 다시 계산하고, 묶음을 합쳐야 하는 판정(B3, E16의 2×2, E17의 K2)을 맡는다.

### 8.6 밤사이 판단 절차 (사용자 지시, 10절)

- 운영 수정(버그, 재실행, GPU 대체, pod 재배치, 워커 수)은 조정자가 정하고 LOG.md에 적는다.
- 설계를 바꿔야 하는 일은 조정자가 판단 요청서(무엇이 문제인지, 근거 파일, 선택지, 각 선택지가 주장에 미치는 영향)를 써서 **Claude Fable 5.2(추론 노력 max) 판단 에이전트**에게 맡긴다. 그 에이전트를 쓸 수 없으면 조정자가 판단한다.
- 판단 결과는 LOG.md에 [편차]로, 이 문서에는 끝에 덧붙이는 변경 기록으로 적는다(고정된 본문은 고치지 않는다). 판단은 결과를 보기 전의 정보로만 한다. 이미 본 결과에 기대는 변경이면 그 사실과 [사후] 표시를 함께 적는다.
- 주장을 정하는 것(`CLAIMS_tmlr.md`)은 사용자 승인으로만 한다.

---

## 9. 결정 항목 (사용자가 10절대로 정함)

| # | 정할 것 | 결정 |
|---|---|---|
| S1 | 비용 상한 | $80(감시 $77). 처음 추천은 $60이었고, 넘을 때를 묻는 질문에 사용자가 $80을 골랐다 |
| S2 | 클라우드 종류 | SECURE |
| S3 | pod 수 | 15개 동시 |
| S4 | E12의 문턱 τ | 10점(DMC 10%) |
| S5 | E11의 "고친다" 기준 | red ≥ 0.5 |
| S6 | 시작 시점 | 승인 즉시 |

- 이 문서를 고정한 뒤의 변경은 [LOG.md](LOG.md)에 시각과 이유를 적고, 8.6절의 절차를 따른다.

## 10. 승인 기록 (2026-11-21T17:52Z 무렵, 한국 시간 11-22 02:52)

- 사용자 지시: "추천대로 진행하고 실행, 완료까지 확인해줘. 실험은 30분마다 한번씩 잘 하고 있는지 확인하고 오류가 있으면 고쳐서 다시 실행까지 해줘."
- 밤사이 결정에 대해 미리 받은 답
  - 비용이 상한을 넘을 것 같을 때: "상한을 $80으로"
  - 설계대로 할 수 없을 때: "설계 변경도 너가 판단해서 해줘. 판단이 필요한 거는 fable 5.2 max를 이용해서 판단해줘. 만약 안 되면 너가 판단해" → 8.6절
  - 자료와 git: pod에 공개 체크포인트·데이터를 받는다. 끝나면 설계·코드·결과를 공개 저장소에 push한다.
  - 알림: 완료와 중대한 문제만 휴대폰으로 알린다.
- 조정자가 고정 전에 바꾼 운영 값(가설·문턱·칸은 그대로): 상한에 맞춘 태그별 상한(td $24, mt $12, k $6, dw $35)과 묶음별 자동 종료 시간(td 8, k 8, mt 9, dw 10시간), 8.5·8.6절.

---

## 11. 고정 뒤 변경 기록 (§8.6, 본문은 고치지 않는다)

| # | 시각(UTC) | 바뀐 곳 | 내용 | 정한 방법 |
|---|---|---|---|---|
| D1 | 2026-11-21T18:41:18Z | §0.4 PushT eval8 | hard = held·smoke·eval6, soft = repro(공유 0 우선, 없으면 5,000번 중 최소 공유), 완전 중복 기각. 겹침·몰림 통계, PushT 칸의 덩어리 단위 bootstrap CI를 보조로 보고. 점검 3이 창 밖이면 eval6 쌍 0~49의 none을 E1 41/50과 비교(36~46이면 진행). 자세한 내용은 LOG.md | 판단 에이전트(Fable 5.2, max), eval8 결과 전 |
| D2 | 2026-11-21T19:01:09Z | §0.6 4단계, §0.7 점검 2 | b ← b + α·m을 모든 층 잔차 < 1% Δ까지(갱신 최대 60), α = 1 → ½ → ¼ 순서로 단계마다 b = 0에서. 못 미치면 최소 잔차 b + bc_unconverged 표시, 판정 두 가지 보고. 점검 2는 최종 b에서. 자세한 내용은 LOG.md | 판단 에이전트(Fable 5.2, max), 수렴한 b의 det+BC 결과 전 |
| D3 | 2026-11-21T19:41:09Z | §0.7 점검 3 (Wall) | [사후] Wall eval8 none 49/100이 창 밖이지만 같은 pod의 E1 쌍 재실행 62/100(E1 61)로 파이프라인 정상 → §0.4 그대로 진행, 판정 규칙 불변, 기준선 셋을 함께 보고. 이후 DINO-WM 점검 3이 창 밖이면 E1 분할 none 재실행이 E1 ±10점 안일 때 진행 | 판단 에이전트(Fable 5.2, max), Wall 양자화 변형 결과 전 |
| D4 | 2026-11-22T01:36Z | §1.2 DINO-WM 칸 | [사후] PushT 3-bit E11 칸(8변형 + none, eval8 100 episode, E13 없음)을 새 pod dw.6에서 보강으로 더 돈다. eval8에서 손상 칸이 Wall 하나뿐이었기 때문이다. 사전 지정 판정(B2·B3·Dg1)은 바꾸지 않고, 이 칸의 읽기와 두 칸을 합친 단위 읽기는 [사후] 기술로 따로 보고한다 | 사용자 결정(조정자 추천, eval8 결과를 본 뒤) |
