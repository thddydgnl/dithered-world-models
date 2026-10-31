# A RSS 2027 확장 실험 설계 (E8–E10) — 사전등록

- 작성·고정: 2026-10-31(UTC), 조정자. 어느 실험의 자료도 생기기 전에 고정한다. 고정 시각과 sha256은 [LOG.md](LOG.md)에 적는다.
- 배경
  - 사용자가 10-31에 A를 RSS 2027 Stage 1(12-04 AoE)에 내기로 정했다.
  - 같은 날 약점 보강 추천 ①~④ 가운데 실험인 ②③④를 RunPod에서 돌리라고 지시했다(예산 $30). ①(주장 재구성)은 이 결과를 받은 뒤 사용자 승인으로 정한다.
  - 약점: E7에서 5M 모델·후보 512·A5000의 정수 kernel이 fp32보다 약 18% 느렸다. W8A8은 손상이 없고 손상은 4-bit 이하 활성에서만 생긴다. 그래서 "빨라지지도 않는데 왜 저비트인가"라는 물음에 답이 약하다.
- 기준 문서: [../main/DESIGN_main.md](../main/DESIGN_main.md) v5.3(통계·판정 규약), [../main/reports/FINAL_A.md](../main/reports/FINAL_A.md)
- 원칙
  - 가설, 지표, 문턱, 칸은 이 문서가 정한다. 작업자는 운영 세부(스모크 결과, 구현 선택과 그 규칙, 실행 순서, block 길이)만 각 실험 폴더의 `PLAN.md`에 본 실행 전에 적는다.
  - 결과를 본 뒤 정한 분석은 [사후]로 표시한다. 주장을 바꾸는 일은 사용자 승인으로만 한다.
  - 문서는 한국어, 코드와 주석은 영어로 쓴다.

---

## 0. 공통

### 0.1 장비와 비용
- RunPod만 쓴다(사용자 지시). 도구는 `runpod_rss/`(pod 접두어 `arx`)이고, 감시 스크립트가 추정 지출 $28에서 모든 pod를 지운다(사용자 상한 $30).
- 실험별 상한(`runpod_rss/caps.json`): E8 $3, E8b $2, E9 $7, E10 $13. 합 $25이고, 남은 $3은 조정자가 필요할 때 배정한다.
- pod는 SECURE 클라우드에서 만들고, 생성 4시간 뒤 자동 종료된다. E10은 조정자 허락으로 최대 8시간까지 쓴다.
- 결과를 받은 뒤 pod를 지우고 `budget.py status`에서 `gone`을 확인한다.

### 0.2 소프트웨어 고정 (E1·E7과 같음)
- torch 2.7.1(cu126)과 딸린 Triton, tensordict 0.8.3, dm-control 1.0.16, mujoco 3.1.2, numpy 1.24.4, gymnasium 0.29.1, hydra-core 1.3.2, omegaconf 2.3.0
- Meta-World: gym 0.21.0, Metaworld@04be337, mujoco-py 2.1.2.14, MuJoCo 2.1.0
- tdmpc2 commit `332f19e`
- 예외: E8b(Blackwell)만 CUDA 12.8 이미지와 그 이미지의 torch를 쓴다.

### 0.3 통계 (DESIGN_main v4 §6.1과 같음)
- D(하락): 같은 짝의 none 대비. Meta-World는 성공률 차(비율), DMC는 상대 return 하락 1 − ΣR_v / ΣR_none(재표본마다 비율).
- 짝: 같은 (체크포인트, episode seed). 짝 bootstrap 10,000회, 95% 백분위 CI, `numpy.random.RandomState(0)`.
- 하락 실재: 단측 bootstrap p(재표본 D ≤ 0의 비율) < 0.05. 실험마다 정한 칸 집합 안에서 Holm으로 보정한다.
- red(손상 감소율) = (D_det − D_v) / D_det. 판정은 점 추정, CI는 보고한다.
- 해로운 칸: D_v − D_det의 CI 하한 > 0.05(Meta-World 5%p, DMC 상대 5%).
- 문턱 비교에는 허용 오차 1e-9를 쓴다.

### 0.4 seed
- E8: 무작위 가중치 seed 0, 측정 순서 numpy seed 0
- E9: eval episode 80000~80029(체크포인트 1·2·3 → 90짝), held(1-step 오차 입력) 81000~81029
- E10: held 91000~91029, eval 90000~90089(체크포인트 1개 → 90짝)

### 0.5 산출물
- `rss_ext/<실험>/`: `PLAN.md`(본 실행 전), `code/`(실제로 돌린 코드 전부), `results/`(JSON·CSV·작은 PNG, checkpoint와 데이터 없음), `LOG.md`(pod, GPU, 오류와 수정), `REPORT.md`(한국어)
- 판정 수치는 결과 파일에서 다시 계산할 수 있어야 한다.

---

## 1. E8 지연 영역 지도 (추천 ②)

### 1.1 물음
저비트 정수 kernel이 fp32·bf16보다 빨라지는 (모델 폭, 후보 수) 영역이 있는가. 그 영역에서 공유 dither(sh-iter)의 추가 지연은 얼마인가.

### 1.2 장비
- NVIDIA A40(Ampere GA102, E7의 RTX A5000과 같은 칩 계열. 10-31 기준 RunPod에 A5000 재고가 없다). 대체는 RTX A6000.
- pod 하나, GPU 배타, MPS 없음, 측정은 단일 프로세스

### 1.3 조건
- 모델: TD-MPC2 구조, model_size ∈ {5, 19, 48, 317}
  - mlp_dim 512 / 1024 / 1792 / 4096, latent 512 / 768 / 768 / 1376, num_q 5 / 5 / 5 / 8(공식 `MODEL_SIZE`)
  - 단일 과제 설정(task embedding 없음), 관측 39차원, 행동 4차원(mw-push 차원)
  - 무작위 초기화(seed 0). 지연은 가중치 값과 무관하다. 양자화 범위는 무작위 입력으로 잡은 값을 쓴다.
- 계획기: MPPI 기본값(반복 6, horizon 3, elite 64, 정책 궤적 24, temperature 0.5), 후보 수 N ∈ {512, 2048, 8192}
- 칸 = 크기 4 × N 3 = 12칸
- 정밀도 경로. 양자화 대상은 E1·E7과 같은 dynamics·reward MLP 은닉 활성 4층이다. encoder·Q·π는 fp32다.
  - fp32: TF32 끔(TD-MPC2 기본, E7의 fp32와 같음)
  - tf32: matmul TF32 허용
  - bf16: dynamics·reward MLP autocast(E7과 같음)
  - int-det, int-sh-iter: 층 1·2 가중치 W8(출력 채널별 RTN) + 4-bit 활성 코드(int8 그릇). E7의 int4 그릇판과 같은 수치 경로다. 층 0은 fp32 GEMM이다(입력 [z, a]는 양자화하지 않음).
- 정수 GEMM 구현
  - 크기와 무관한 일반 경로(출력 열을 타일로 나누는 GEMM)를 새로 짠다. E7의 융합 kernel은 행 전체 LayerNorm을 한 block에서 하므로 폭 4096에 그대로 쓸 수 없다.
  - 후보 (a) Triton int8 타일 GEMM + 역양자화 epilogue, (b) `torch._int_mm`(cuBLASLt) + Triton 행 kernel(역양자화·bias·LayerNorm·Mish·양자화)
  - 작업자는 **int-det 지연만으로** 크기마다 빠른 쪽을 고른다. 그 규칙과 결과를 본 측정 전에 PLAN.md에 적는다.
  - 5M에서는 E7의 융합 kernel(`e7lib`)도 함께 잰다(기술).
- dither 난수: sh-iter 일정(층·forward마다 u 벡터 하나, 모든 후보 공유). seed는 device tensor로 넘기고 forward마다 device에서 늘린다. 그래서 CUDA graph 안에서도 forward마다 새 난수가 나온다.
- 실행 방식
  - **graph-WM(주)**: 한 제어 step의 world-model rollout을 CUDA graph 하나로 잡아 재생한 시간. rollout은 반복 6 × horizon 3 × (dynamics + reward)를 배치 N으로, 그리고 정책 궤적 rollout(배치 24, dynamics 2회)을 포함한다.
  - eager-WM(보조): 같은 rollout을 eager로 잰 시간
  - eager-step(보조, E7과 같은 규약): TD-MPC2 agent의 계획 step 전체를 eager로 잰 시간. N = 512에서만 잰다(크기 4개).

### 1.4 측정 규약
- 조건마다 warmup 30회 뒤 block 단위로 CUDA event로 잰다.
  - block 길이는 약 2초 이상이 되게 조건마다 정한다(최소 10회). PLAN.md에 적는다.
  - block 값은 block 안 회당 시간의 중앙값이다.
- int-det와 int-sh-iter는 같은 (크기, N, 방식)의 두 block을 바로 이어 잰 짝으로 잰다. 짝 안의 순서는 무작위다. fp32·tf32·bf16은 단일 block이다.
- 10회전이다. 회전마다 단위 순서를 새로 섞는다(numpy seed 0).
- block마다 SM clock, GPU 온도, 호스트 1분 load를 기록한다.
- 계산
  - 조건 지연 T = 그 조건 모든 block 값의 중앙값. 비율 = T의 비
  - 짝 오버헤드 = (sh-iter block − det block) / det block. 칸 값은 짝 10개의 중앙값이고, CI는 짝을 다시 뽑는 bootstrap 10,000회다.

### 1.5 가설과 판정 (graph-WM 기준)
- **L1(저비트가 빨라지는 영역이 있다)**: T(int-det) / T(fp32) ≤ 0.80인 칸이 12칸 중 하나 이상이면 성립
- **L2(가장 빠른 부동소수점보다도 빠르다)**: T(int-det) / min(T(tf32), T(bf16)) ≤ 0.90인 칸이 하나 이상이면 성립
- **L3(dither는 공짜다)**: 12칸 모두 짝 오버헤드 중앙값 ≤ 5%이고, CI 상한 ≤ 10%인 칸이 10칸 이상이면 성립
- 기술 항목
  - 크기별 교차점: int-det가 fp32, 그리고 min(tf32, bf16)보다 빨라지는 가장 작은 N
  - eager-WM과 graph-WM의 차
  - eager-step의 비율(Q·π가 fp32라 이득이 줄어드는 정도)
  - (5M, N = 512)에서 E7 재현 여부, E7 융합 kernel 대 새 일반 경로

### 1.6 E8b (선택, 기술, 시간 상자 1.5시간, 상한 $2)
- E8 주 측정이 끝나고 예산이 남을 때만 한다.
- Blackwell GPU(RTX 5090 또는 RTX PRO 4500 Blackwell)와 CUDA 12.8 이미지(`runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`)
- 설치된 PyTorch가 지원하는 경로로만 잰다. 새 kernel은 짜지 않는다.
  - 폭 4096 dynamics·reward 층 모양(M = N ∈ {512, 2048, 8192})에서 BF16, FP8(E4M3, `torch._scaled_mm`), NVFP4(지원되면) GEMM 지연
  - FP4 활성 양자화 연산에서 det 대 sh-iter 확률적 반올림의 추가 시간
- 지원되지 않으면 그 사실과 오류만 기록하고 끝낸다. 판정은 없다.

---

## 2. E9 4-bit 부동소수점 형식 (추천 ③)

### 2.1 물음
가속기가 실제로 쓰는 블록 스케일 4-bit 부동소수점(NVFP4, MXFP4)에서도 결정적 반올림이 계획을 해치는가. 그리고 공유·매 forward 새 확률적 반올림(sh-iter)이 이를 고치는가.

### 2.2 설정 (E1과 같음)
- TD-MPC2 공식 단일 과제 5M 체크포인트, seed 1·2·3
- 계획기 기본값(MPPI, 후보 512, elite 64, 반복 6, horizon 3)
- 과제: E1의 12과제
  - Meta-World 8: push, pick-place, door-open, drawer-open, button-press-topdown, peg-insert-side, window-open, sweep-into
  - DMC 4: cheetah-run, walker-walk, quadruped-walk, finger-turn-hard
- 양자화 위치: E1과 같은 4층(dyn0, dyn1, rew0, rew1의 Mish 출력). 가중치는 fp32다.

### 2.3 형식 (행 = 후보 하나, 블록 = 은닉 차원을 따라 이어진 단위)
- 원소 형식 E2M1: 표현값은 {0, 0.5, 1, 1.5, 2, 3, 4, 6}과 그 음수다. 크기가 6을 넘으면 ±6으로 자른다.
- **NVFP4**: 블록 크기 16
  - 층별 전역 scale g = hi / (6 · 448). hi는 E1이 쓴 층별 범위의 상한이다(`main/results/E0_tdmpc2/jobs/ranges/ranges_<과제>_s<체크포인트>.json.gz`의 `quant`, 99.99 백분위).
  - 블록 scale s_b = E4M3_RNE(amax_b / (6g)) · g. E4M3 범위로 자르고, amax_b = 0이면 s_b = g다.
- **MXFP4**(OCP MX v1.0): 블록 크기 32. s_b = 2^(⌊log2 amax_b⌋ − 2)(E8M0). amax_b = 0이면 s_b = 1이다.
- 두 형식 모두 x_s = x / s_b, 출력 x̂ = q · s_b
- 변형
  - det: x_s를 E2M1 격자의 가장 가까운 값으로 반올림한다(동률은 짝수 가수 쪽, RNE).
  - sh-iter: x_s를 감싸는 이웃 격자값 lo ≤ |x_s| < hi 사이에서 확률적으로 반올림한다. q = lo + (hi − lo) · 1[u < (|x_s| − lo) / (hi − lo)]이고 부호를 다시 붙인다.
    - |x_s| ≥ 6이면 q = ±6이다(det와 같은 포화).
    - u ~ U[0, 1)는 (층, 단위)마다 하나다. 모든 후보가 공유하고 forward마다 새로 뽑는다(E1 sh-iter 일정, 개별 CUDA generator, episode마다 seed).
  - 블록 scale은 두 변형 모두 그 행의 값으로 결정적으로 정한다. 두 변형은 원소 반올림만 다르다.
- 칸 = 과제 12 × 형식 2 = 24칸. 과제마다 none이 하나 있고, 칸마다 det와 sh-iter가 있다.

### 2.4 episode와 1-step 오차
- 칸의 변형마다 90짝(체크포인트 3 × episode 30, seed 80000~80029). none도 같은 90 episode다.
- 1-step 오차: E1 H1과 같은 정의(pt_err1h: 계획 시점 후보 집합의 1-step 상대 오차)와 같은 코드 경로(`qlib.pred_errors`, `plan_inputs`)를 쓴다. held 입력은 seed 81000~81029의 none episode에서 만든다.

### 2.5 가설과 판정
- **F1(FP4 det도 계획을 해친다)**: 형식마다 하락이 실재하는 칸(그 형식의 12과제 안에서 Holm)이 3칸 이상이면 성립
- **F2(sh-iter 확률적 반올림이 고친다)**: F1에서 하락이 실재한 칸(손상 칸)으로 Meta-World와 DMC를 따로 판정한다(두 형식은 합친다).
  - 손상 칸의 75% 이상에서 red ≥ 0.5이고 해로운 칸이 없으면 성립
  - 주장은 E1 H3a처럼 Meta-World 판정으로 한다.
- **F3(1-step 오차로는 손상이 안 보인다)**: 손상 칸 중 det의 pt_err1h ≤ sh-iter의 pt_err1h인 칸이 하나 이상이면 성립. 칸 수를 보고한다.
- 기술 항목: 형식별 손상 크기와 E1 균일 k = 4 det 손상 비교, NVFP4 대 MXFP4

### 2.6 결과의 쓰임 (미리 적음)
- F1과 F2가 성립하면: "가속기의 4-bit 형식에서도 같은 실패와 같은 처방"을 본문에 쓴다.
- F1이 불성립하면(블록 스케일이 손상 자체를 없앰): 처방의 실용 범위를 균일 정수 격자로 좁히고, 이 결과를 경계로 본문에 쓴다.

---

## 3. E10 큰 다중 과제 모델 mt80-317M (추천 ④)

### 3.1 물음
은닉 폭 4096의 큰 world model에서도 det 손상과 sh-iter 처방이 그대로인가. 이 크기는 E8에서 정수 kernel이 이득일 것으로 예상하는 크기다.

### 3.2 설정
- 체크포인트: TD-MPC2 공식 다중 과제 mt80-317M(tdmpc2.com/models, 체크포인트 하나)
- 공식 코드는 다중 과제 체크포인트의 단일 과제 평가를 지원하지 않는다. 그래서 학습 때의 다중 과제 wrapper와 같은 관측 padding, 행동 mask, task embedding으로 한 과제를 평가하는 wrapper를 쓴다. 스모크에서 none 성공률로 확인한다.
- 계획기: 기본값(E1과 같음)
- 양자화: E1과 같은 4층(dyn0, dyn1, rew0, rew1의 Mish 출력)과 층별 균일 k-bit 격자
  - 범위 [min, 99.99 백분위]는 과제마다 held none episode의 계획 forward에서 잡는다(E0와 같은 방식).
  - 가중치는 fp32다. 변형은 none, det, sh-iter다.
- GPU: RTX 4090(대체 L40S, A40). 과제를 나눠 pod 둘까지 쓴다.

### 3.3 과제 자격
- 후보는 E1의 Meta-World 8과제다.
- held none 30 episode(seed 91000~91029)의 성공률이 0.7 이상이면 자격이 있다.
- 자격 과제가 4개 미만이면 아래 목록 순서로 더한다. 자격 과제가 6개가 되거나 목록이 끝나면 멈춘다.
  - mw-reach, mw-button-press, mw-door-close, mw-drawer-close, mw-faucet-open, mw-faucet-close, mw-plate-slide, mw-coffee-button, mw-handle-press, mw-window-close

### 3.4 비트 수 선택 (E1의 held 20%p 규칙)
- 자격 과제마다 held det를 k = 5, 4, 3, 2 순서로 30 episode씩(seed 91000~91029) 돌린다.
- none 대비 하락이 처음 20%p 이상이 되는 k(그런 k 중 가장 큰 값)를 1차 k로 정하고 멈춘다.
- k = 2까지 20%p에 이르지 않으면 그 과제는 "2비트 이상에서 손상 없음"으로 보고하고 G2에서 뺀다.

### 3.5 episode와 1-step 오차
- 1차 k에서 none, det, sh-iter를 각 90 episode(seed 90000~90089, 짝) 돌린다.
- 1-step 오차: E1 H1과 같은 정의(pt_err1h). held 입력은 seed 91000~91029의 none episode에서 만든다.

### 3.6 가설과 판정
- **G1(큰 모델에서도 손상이 생긴다)**: 자격 과제 중 1차 k의 eval에서 하락이 실재하는(과제들 사이 Holm) 과제가 3개 이상이면 성립
- **G2(처방)**: 손상 과제의 75% 이상에서 red ≥ 0.5이고 해로운 과제가 없으면 성립
- **G3(1-step 오차)**: 손상 과제 중 det의 pt_err1h ≤ sh-iter인 과제가 하나 이상이면 성립
- 기술 항목: 1차 k와 5M(E1)의 k 비교, 손상 크기, none 성공률, 결정 한 번의 계획 지연(fp32, 그 GPU)

### 3.7 결과의 쓰임 (미리 적음)
- G1과 G2가 성립하면: "317M에서도 같은 실패와 처방"을 본문 표에 넣는다. E8과 합쳐 "정수 kernel이 이득인 크기에서도"라고 쓴다.
- G1이 불성립하면: 큰 모델은 같은 비트에서 더 견딘다는 경계로 쓴다(비트 수 의존과 함께).

---

## 4. 진행과 중단
- 세 실험은 서로 독립이라 병렬로 돈다. 각 작업자는 스모크 → PLAN.md → 본 실행 → 분석 → REPORT.md 순서로 한다.
- 상한의 80%에 이르렀는데 본 실행을 끝내지 못할 것 같으면 멈추고 조정자에게 알린다. 여유분 $3의 배정은 조정자가 정한다.
- 설계대로 잴 수 없으면(예: mt80 wrapper가 none 성공률을 재현하지 못함) 멈추고 선택지를 조정자에게 보고한다.
- 1시간 넘게 도는 작업은 조정자가 1시간마다 `runpod_rss/check_all.sh`로 점검한다.
