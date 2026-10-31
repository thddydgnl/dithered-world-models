# E8c 추가 지연 측정 — 실행 계획 (본 측정 전 작성)

- 작성: 2026-10-31T16:40Z, E8c 작업자. 본 측정 자료가 생기기 전이다(스모크와 보정 측정, 코드 경로 확인용 quick 실행만 있음).
- 기준
  - [../DESIGN_E8c.md](../DESIGN_E8c.md)(sha256 `2e9174b3…`)
  - [../DESIGN_ext.md](../DESIGN_ext.md) §0·§1·§4
  - [../E8_latency/PLAN.md](../E8_latency/PLAN.md)(E8 규약)
  - 가설(L2c, S1~S3), 칸(크기 4 × N 3), 경로, 문턱, 기술 항목, 결과의 쓰임은 설계 그대로이고 여기서 바꾸지 않는다.
- 이 문서는 운영 세부만 정한다: 구현 선택과 그 규칙, graph 메모리 전략, `randperm` 처리, 기능 점검 방법과 결과, block 길이와 실행 순서, 비용 추정.
- **운영 변경**(조정자 결정, 본 측정 자료 전, LOG 16:04Z): 스모크 전 추정이 4시간·$3을 넘었다. 그래서 상한을 $3 → $4.5로, pod 수명을 4 → 6시간으로 바꿨다. 설계(가설, 칸, 10회전, warmup 30, block 규칙)는 그대로다. 스모크 뒤 추정 총지출이 $3.6을 넘으면 본 측정 전에 멈춘다(§9).
- 코드: `code/`
  - `e8lib.py`: E8 파일 그대로(md5 `7dd7b30d…`). 정수 경로, row kernel, dither, fp32·tf32·bf16 경로, 모델, 입력, 범위, 측정 도구
  - `e8clib.py`: E8c 추가분(bf16cast, int 층 0 TF32 고정, int-det-L0tf32, graph-step, 단위, graph 생성)
  - `e8c_smoke.py`, `e8c_main.py`, `e8c_analyze.py`, `dev_q_tf32.py`, `setup_pod.sh`(E8과 같음), `run.sh`
  - `e7dev.py`: e8lib가 import할 수 있게 둔 E8 파일. 쓰지 않음
- 입력: `results/e8_inputs/`(E8의 `gemm_cfg.json`, `ranges.json`, `selection.json`, `e8_summary.json`)
- 스모크 산출물: `results/smoke/`(env, ranges, ranges_vs_e8, randperm, checks, calib), 로그는 `results/logs/`

## 1. 장비와 소프트웨어

- pod `arx-e8c` (`oti534analhzml`)
  - **NVIDIA A40** 1장(GA102, 84 SM, 46 GB, 전력 상한 300 W, 최대 SM clock 1,740 MHz), SECURE, $0.59/h
  - driver 595.91.07, 96 vCPU(호스트 공유), 생성 16:16:25Z, 자동 종료 22:15:52Z
- E8 pod(`30cyekyjii6pzs`)와 GPU 종류·driver가 같지만 다른 장비다. 그래서 E8 수치와 한 표에 섞지 않는다(설계 §2). E8 대 E8c는 기술 표로만 나란히 둔다.
- 측정은 단일 프로세스, GPU 배타(pod에 GPU 하나), MPS 없음
- 소프트웨어는 E8과 같다: torch 2.7.1+cu126, triton 3.3.1, tensordict 0.8.3, numpy 1.24.4, hydra-core 1.3.2, omegaconf 2.3.0, matplotlib 3.10.9, nvidia-ml-py 13.615.71, tdmpc2 `332f19e4`

## 2. 모델, 입력, 양자화 범위 (E8 그대로)

- E8 `e8lib.py`를 고치지 않고 import한다. 다음이 모두 E8과 같다.
  - 모델: 공식 `TDMPC2(cfg)`, `torch.manual_seed(0)`, 단일 과제, 관측 39, 행동 4, 크기별 차원(공식 `MODEL_SIZE`)
  - 양자화 범위: 무작위 입력, seed 0
  - graph-WM 입력과 eager-step 관측 열(N(0,1) 39차원 200개, seed 0). graph-step도 이 관측 열을 쓴다.
- 범위: 스모크와 본 측정에서 다시 계산해 E8 `ranges.json`과 같은지 assert한다. 스모크에서 네 크기 모두 같았다.
- 정수 GEMM
  - 네 크기 모두 E8이 고른 경로 (a)(Triton int8 타일 GEMM)를 쓴다.
  - 타일 설정은 E8 `gemm_cfg.json` 그대로다(다시 조율하지 않음).
  - row kernel 배치(tile 1개), dither(u를 (층, forward)마다 한 kernel 앞에서 한 번, device counter)도 E8 그대로다.
- 지연 측정은 E8처럼 공식 초기화를 그대로 쓴다. reward head와 Q head가 0이라 모든 후보의 가치가 0이지만, 계산량은 같다. 기능 점검만 두 head를 다시 무작위로 채운 agent를 쓴다(§6).

## 3. 정밀도 경로 구현

- **fp32, tf32, bf16**: e8lib의 `WMfp32`, `WMtf32`, `WMbf16` 그대로다(E8 PLAN §3).
  - tf32: WM 호출 안에서만 TF32를 허용한다.
  - bf16: `torch.autocast(bfloat16, cache_enabled=False)`이고, 호출마다 가중치를 bf16으로 바꾼다.
- **bf16cast(새)**: 측정 전에 `copy.deepcopy(model._dynamics / _reward).to(torch.bfloat16)` 복사본을 한 번 만든다(가중치, bias, LayerNorm affine 모두 bf16).
  - 호출: `cat([z, a]).to(bf16)` → bf16 MLP → `.float()`. autocast는 쓰지 않는다.
  - MLP 안의 연산: Linear는 cuBLAS bf16 GEMM이고, LayerNorm·Mish·SimNorm은 bf16 텐서에 PyTorch 기본 kernel을 쓴다(내부 누적은 fp32).
- **int-det / int-shiter**: E8 정수 경로(`e8lib.WMint`, GEMM (a))에 `WMintX`를 씌운다. MLP 호출 동안 `allow_tf32 = False`를 둔다.
  - 정수 MLP forward 안의 cuBLAS 호출은 층 0의 `F.linear` 하나뿐이다(층 1·2는 Triton int8 kernel).
  - 그래서 이것은 "층 0은 fp32 GEMM"이라는 E8 정의를 rest-tf32에서도 지키는 장치다. rest-fp32에서는 아무것도 바꾸지 않는다(전역 값이 이미 False).
- **int-det-L0tf32(새, graph-WM만)**: 같은 클래스에서 MLP 호출 동안 `allow_tf32 = True`를 둔다. 그래서 층 0 GEMM만 TF32가 된다.
- **rest-tf32(graph-step만)**: 캡처하는 동안 전역 `allow_tf32 = True`.
  - WM 호출은 경로마다 자기 값을 둔다: tf32는 True(전부 TF32), bf16 autocast·bf16cast는 bf16 GEMM, int는 층 0을 False로 둔다.
  - 캡처 뒤 전역을 False로 되돌린다.
- TF32 설정은 host에서 GEMM을 고를 때 읽힌다. 그래서 graph에는 캡처 때의 선택이 그대로 들어간다(스모크에서 rest-tf32가 결과를 바꿈을 확인).

## 4. 실행 방식 구현

- **graph-WM**: E8 그대로(`e8lib.GraphRunner`: side stream에서 warmup 3회 뒤 캡처, graph마다 private pool, 같은 rollout).
- **graph-step**: `e8clib.GraphStep`
  - 캡처 전
    - WM 경로를 agent의 model에 설치한다(`e8lib.install`, E8 eager-step과 같은 방식).
    - `cfg.num_samples = N`으로 두고, rest의 TF32를 설정한다.
    - side stream에서 eager `_plan(static_obs, t0=False, eval_mode=True)`를 3회 부른 뒤 `torch.cuda.graph`로 한 번 캡처한다.
  - 캡처 뒤 모두 되돌린다: `cfg.num_samples` 512, TF32 끔, 설치 해제.
  - 공식 `_plan` 코드를 그대로 캡처한다(고친 연산 없음). 잡히는 것:
    - encoder
    - 정책 궤적 24개(π 3회, dynamics 2회)
    - MPPI 6반복: 표본, WM 3 step × (reward + dynamics), π, Q(avg, 무작위 2개), top-k 64, 갱신
    - Gumbel 행동 선택, `_prev_mean` 갱신
  - **한 회** = 고정 관측 열에서 다음 관측을 static 입력 버퍼로 copy + graph 재생이다. 관측 열은 host pageable 메모리에 있고, copy는 `TDMPC2.act`처럼 `non_blocking=True`다. CUDA event 쌍이 copy와 재생을 함께 감싼다.
    - 행동은 device의 static 출력에 남고 host로 읽지 않는다(설계 정의: `_plan` 재생 + 관측 copy). eager-step의 `act`에는 `.cpu()`가 있다(§10).
    - block마다 관측 순번을 0부터 다시 시작한다.
  - 난수
    - MPPI 표본(`torch.randn`), π 잡음(`randn_like`), Gumbel(`exponential_`), Q 부분집합(`torch.randperm`)은 모두 기본 CUDA generator를 쓴다.
    - PyTorch의 graph-safe generator라 재생마다 offset이 이어지고 새 난수가 나온다.
    - **`torch.randperm`은 그대로 캡처된다**(스모크 §6). 그래서 설계 §3의 "graph 호환 연산으로 바꾼다"는 해당하지 않고, 바꾼 연산이 없다.
  - dither: E8의 device counter식 그대로다. 재생마다 dynamics 20회, reward 18회 forward의 u가 새로 나온다.
  - t0 = False이므로 `_prev_mean`(agent buffer, pool 밖)이 재생과 block 사이에 이어진다(warm start, 계산량과 무관).
- **eager-step**: E8 그대로다. `TDMPC2.act(obs, t0=(block 첫 호출), eval_mode=True)`, N 512, rest-fp32이고, 경로는 `install`로 바꿔 끼운다.

## 5. graph 메모리 전략

- **graph-WM 84개**: E8처럼 graph마다 private pool이다.
- **graph-step 132개**: 한 공유 pool(`torch.cuda.graph_pool_handle()`)에 모두 캡처한다.
  - private pool로는 다 둘 수 없다. 317M N 8192 graph-step 하나의 중간 텐서(Q 8개 × 8192 × 4096 fp32 = 1 GiB짜리 여러 개)가 4 GiB를 넘어서, 그 칸 11개만으로 약 50 GB가 된다.
  - 안전 근거
    - 재생이 읽는 텐서는 pool 밖에 있거나(가중치, static 관측, `_prev_mean`, dither 상태와 u 버퍼), 같은 재생 안에서 먼저 쓴 것이다.
    - 그래서 어느 graph가 먼저 돌았든 같은 값을 계산한다.
    - 다른 graph의 재생이 덮어쓸 수 있는 것은 각 graph의 static 출력(행동 4개)뿐이다. 측정 block은 이를 읽지 않는다.
  - 스모크에서 확인: 크기·N마다 11개를 한 공유 pool에 잡고 무작위 순서로 재생해도 eager와 비트 단위로 같았다(§6).
  - 캡처 순서는 큰 것부터(317M N 8192 → 5M N 512)다. warmup은 side stream 하나를 같이 쓴다. 생성 뒤 `empty_cache`를 부른다.
- 보정 때(본 측정과 같은 생성): graph-WM 26초, graph-step 113초. allocated 4.0 GiB, reserved 21.0 GiB, 최대 allocated 9.1 GiB(A40 46 GB 안).
- 그래서 설계 §4의 "크기별로 graph를 만들고 지운다"는 필요 없다. 모든 graph를 회전 0 전에 한 번 만들고, 회전 안에서 200단위를 E8처럼 전부 섞는다.

## 6. 기능 점검 방법과 결과 (`results/smoke/`, 모두 통과)

- **기능 점검 agent**
  - 같은 모델(seed 0)에서 0으로 초기화된 reward head와 Q head만 다시 무작위로 채운다(trunc normal 0.02, E8 reward 점검과 같은 규칙).
  - 공식 초기화 그대로면 모든 후보의 가치가 0이라, WM·Q 계산이 행동에 드러나지 않기 때문이다.
- **범위**: 네 크기 모두 E8과 같다(`ranges_vs_e8.json`).
- **randperm**(`randperm.json`)
  - `torch.randperm(5 / 8)[:2]`를 graph로 잡고 2,000회 재생했다. 순서쌍이 20/20개, 56/56개 모두 나왔다.
  - χ²는 14.3(자유도 19), 49.4(자유도 55)로 p 0.001 문턱 43.8, 93.2 아래다. 연속 재생의 95~99%에서 쌍이 바뀐다.
  - `WorldModel.Q(…, 'avg')` graph = eager(같은 seed, 차 0)이고, 재생끼리 다르다(5M, 317M).
- **bf16cast**(`checks.json`, 크기 4개): 매개변수가 모두 bf16이고 출력은 fp32다. fp32 대비 상대 L2는 아래 표와 같다. graph-WM 재생 = eager(차 0).

  | 비교 | bf16cast | bf16 autocast |
  |---|---|---|
  | dynamics 단일 forward | 0.50~0.52% | 0.40~0.42% |
  | reward 단일 forward | 0.67~0.89% | 0.53~0.69% |
  | rollout z | 0.68~0.71% | 0.56~0.59% |

  - 문턱: bf16cast의 fp32 대비, bf16 대비 차가 모두 bf16 autocast의 fp32 대비 차의 3배 + 1e-3 이하. 모두 통과했다.
- **int-det-L0tf32**
  - 층 0 GEMM에서 TF32가 실제로 쓰인다(fp32 대비 상대 L2 2.9~3.0e-4).
  - int-det 대비 층 0 코드 불일치는 0.04~0.05%(최대 1단계), 층 1은 0.6~1.4%다.
  - int-det는 전역 TF32를 켜도 출력이 같다(층 0 고정 확인). graph-WM 재생 = eager(차 0).
- **graph-step** 66건(크기 4 × N 512, 5M은 N 2048·8192 추가, 각 11경로, 한 공유 pool, 무작위 순서)
  - 같은 CUDA seed, `_prev_mean`, 관측, dither 상태에서 eager `_plan`과 graph 재생의 행동과 최종 MPPI mean이 **비트 단위로 같다**(문턱 1e-4, 실제 차 0). dither counter도 같다.
  - seed를 다시 두지 않은 재생은 달라진다(새 MPPI·π·Gumbel·Q 부분집합 난수).
  - int-shiter는 같은 seed에서도 재생마다 달라진다(새 u). counter는 재생마다 dynamics +20, reward +18이다. det·float 경로는 같은 seed에서 비트 단위로 같다.
  - rest-tf32는 같은 WM 경로의 rest-fp32와 결과가 다르다(TF32가 실제로 쓰임).
- **quick 실행**(`e8c_main.py --quick --sizes 5`, 1회전, n 3, warmup 2, 수치는 쓰지 않음): 본 측정 코드 경로와 `e8c_analyze.py`(표, CSV, 그림)를 확인했다(LOG).

## 7. 본 측정 규약과 실행 순서 (`code/e8c_main.py`)

- **단위**: 한 회전에 200단위, 240 block이다.

  | 방식 | 칸 | 단일 block | int 짝 | 단위 / block |
  |---|---|---|---|---|
  | graph-WM | 12 | fp32, tf32, bf16, bf16cast, int-det-L0tf32 | det·sh-iter | 72 / 84 |
  | graph-step rest-fp32 | 12 | fp32, tf32, bf16, bf16cast | det·sh-iter | 60 / 72 |
  | graph-step rest-tf32 | 12 | tf32, bf16, bf16cast | det·sh-iter | 48 / 60 |
  | eager-step (N 512, rest-fp32) | 크기 4 | fp32, tf32, bf16, bf16cast | det·sh-iter | 20 / 24 |

- **block**: E8과 같다.
  - warmup 30회 뒤 n회를 잰다. 매회 CUDA event 쌍으로 재고 device sync를 한다. block 값은 n회의 중앙값이다.
  - **block 길이**: n = max(10, ⌈2,000 ms / 1회 시간⌉). 짝은 det의 1회 시간으로 정한 n을 두 block에 같이 쓴다.
  - 1회 시간과 조건별 n은 스모크 보정(`results/smoke/calib.json`의 `t_rep`, `n_reps`)에서 가져온다.
  - n의 범위: graph-WM 10~1,263, graph-step rest-fp32 10~316, rest-tf32 10~365, eager-step 13~93
  - n = 10인 무거운 칸: graph-step 317M N 2048·8192, 48M N 8192. 1회 0.25~2.3 s이고, block 하나가 40회 재생이다.
- **순서**
  - 회전마다 단위 순서를 numpy `default_rng(0)` 순열로 새로 섞는다. 짝 안의 순서는 같은 rng의 `random() < 0.5`로 정한다.
  - 10회전이고, 회전마다 결과를 다시 쓴다(`results/main/e8c_main.json`, 회당 시간은 `e8c_reps.npz`).
- **block마다**
  - dither 상태를 base = 7·(70000 + 회전) + 13으로 다시 둔다(E8).
  - graph-step과 eager-step은 관측 순번을 0으로 되돌린다.
- **기록**(E8과 같은 항목 + rest): 회전, 단위 위치, 단위 안 위치, 회당 시간 전부(npz), block 통계, 시작·끝 시각, 호스트 1·5·15분 load, block 동안 호스트 CPU 사용률, block 시작·중간·끝의 GPU 상태(SM·메모리 clock, 온도, 전력, clock event 이유 비트, 보이는 compute 프로세스 수)
- block 뒤에 `allow_tf32`가 False인지, agent의 `cfg.num_samples`가 512인지 assert한다.
- 회전 0 전에 모든 graph를 만든다(약 2.3분, §5).

## 8. 분석 (`code/e8c_analyze.py`, 본 측정 전에 작성)

- **사전등록(설계 §4–5)**
  - T = 조건의 block 값 10개의 중앙값. 비율은 T의 비다.
  - 짝 오버헤드 = 짝 10개 (sh-iter − det)/det의 중앙값이다. CI는 짝 bootstrap 10,000회, 칸마다 `RandomState(0)`, 백분위 95%다.
  - min(float) = min(T(tf32), T(bf16), T(bf16cast))(같은 칸, 같은 방식). 문턱 비교의 허용 오차는 1e-9다.
  - L2c(graph-WM): T(int-det)/min(float) ≤ 0.90인 칸이 12칸 중 1개 이상
  - S1(graph-step, rest-fp32): T(int-det)/T(fp32) ≤ 0.80인 칸이 1개 이상
  - S2(graph-step, rest-fp32): T(int-det)/min(float) ≤ 0.90인 칸이 1개 이상
  - S3(graph-step, rest-fp32): 12칸 모두 짝 오버헤드 중앙값 ≤ 5%이고, CI 상한 ≤ 10%인 칸이 10개 이상
  - 판정 수치와 함께 칸 범위(최소·최대 비율, 최소 칸)와 통과 칸 수를 낸다.
- **기술(설계 §5)**
  - 같은 pod에서 E8 L1~L3 재계산(graph-WM, E8 정의 그대로: L2는 min(tf32, bf16)). E8 값(다른 pod)과 나란히 둔다(`--e8`).
  - 칸별 T(bf16cast)/T(bf16), T(int-det)/T(bf16cast)(모든 방식). graph-WM의 T(int-det-L0tf32)/T(int-det)
  - graph-step 대 eager-step(N 512, 경로별 시간비, 두 방식의 int-det/fp32, int-det/min(float))
  - 같은 경로에서 T(graph-WM)/T(graph-step): 계획 step에서 WM rollout의 몫(rest-fp32, rest-tf32)
  - rest-tf32 graph-step의 T(int-det)/min(float), T(int-det)/T(tf32), 짝 오버헤드[CI]
  - 크기별 교차점: graph-step rest-fp32에서 int-det가 fp32, min(float)보다 빨라지는 가장 작은 N(비 < 1). graph-WM과 rest-tf32도 같이 낸다.
  - 비율의 회전 bootstrap 95% CI(10회전을 함께 다시 뽑음, RandomState(0), 10,000회). 판정에는 쓰지 않는다.
  - 측정 상태: SM clock, 온도, 전력, 전력 상한 이유 비트의 비율, 호스트 load, CPU 사용률, 자기 조건 T보다 15% 넘게 느린 block의 비율(방식별)
  - 그림: `e8c_ratio_graph.png`, `e8c_ratio_gstep.png`(위 rest-fp32, 아래 rest-tf32, 기준은 rest-fp32 fp32), `e8c_overhead.png`, `e8c_wm_share.png`, `e8c_latency_gstep.png`
- 분석은 pod에서 `run.sh analyze`로 돌린다(E8과 같음). 결과를 본 뒤 정하는 분석은 [사후]로 표시한다.

## 9. 비용과 시간 추정

- 보정(`calib.json`) 기준 회전당 1,845초다.

  | 방식 | 회전당 |
  |---|---|
  | graph-WM | 343초 |
  | graph-step rest-fp32 | 831초 |
  | graph-step rest-tf32 | 583초 |
  | eager-step | 87초 |

  - 10회전에 graph 생성 139초와 모델 준비 약 60초를 더하면 **약 311분(5.18 h)**이다.
  - E8에서 같은 방식의 추정(644초)과 실제(637~640초)는 1% 안에서 맞았다.
- 일정: 본 측정 시작 약 16:45Z → 끝 약 21:56Z. 분석·당겨 오기·삭제에 약 10분 → 약 22:06Z. pod 자동 종료는 22:15:52Z다. 여유는 약 10분으로 작다. 그래서 회전마다 결과를 맥북으로 당겨 둔다.
- 지출(budget.py 방식 = 가동 시간 × $0.59 × 1.03)
  - pod 생성 16:16:25Z부터 약 22:06Z까지 5.83 h → **약 $3.54**
  - 상한 $4.5의 79%이고, 중단선 $3.6 아래다. 그래서 본 측정을 시작한다.
- 위험
  - 실제 시간이 추정보다 3% 넘게 길어지면 마지막 회전이 pod 수명을 넘을 수 있다. 그때는 끝난 회전까지만 남는다.
  - 21:55Z에 회전 9가 끝나지 않을 것으로 보이면 당겨 온 결과(9회전 이상)로 분석하고, 조정자에게 보고한다(판정은 설계의 10회전 규칙을 바꾸지 않고, 모자란 사실을 그대로 적는다).

## 10. 설계 대비 편차·해석 (본 측정 전에 밝힘)

1. **운영 변경**: 상한 $3 → $4.5, pod 수명 4 → 6시간(조정자 결정, 본 측정 자료 전). 설계 내용은 그대로다.
2. **bf16cast의 정규화 연산 정밀도**: LayerNorm·Mish·SimNorm도 bf16 텐서에서 계산된다(설계의 "매개변수 모두 bf16, autocast 없음"을 따른 결과). autocast(bf16)는 LayerNorm·softmax를 fp32로 한다. 이 차이가 bf16cast의 시간과 수치(fp32 대비 0.5~0.9%, autocast는 0.4~0.7%)에 들어 있다.
3. **int 경로의 층 0 TF32 고정**: E8 정의(층 0은 fp32)를 rest-tf32에서도 지키려는 장치다. rest-fp32와 graph-WM에서는 E8과 같은 kernel이다.
4. **graph-step 시간에 행동의 host 읽기가 없다**(설계 정의). eager-step(`act`)에는 `.cpu()`(4개 값 D2H와 동기화)가 있다. graph-step 대 eager-step 비교는 이 차이를 포함한다.
5. **기능 점검 agent**는 reward·Q head를 다시 채운 복사본이다. 지연 측정은 공식 초기화(두 head 0, 모든 가치 0)를 그대로 쓴다(E8 eager-step과 같음).
6. **graph-step 공유 memory pool**(§5): 측정되는 계산은 같다. 다른 graph가 static 출력만 덮어쓸 수 있다.
7. **`torch.randperm`은 바꾸지 않았다**(캡처됨, §6).
8. **rest-tf32의 효과**(스모크 보정, 기술): 317M N 8192에서 rest를 TF32로 해도 graph-step이 6% 정도만 빨라진다(N 512에서는 약 30%).
   - 본 측정 전에 `dev_q_tf32.py`로 eager 분해를 쟀다(317M, TF32/fp32 시간비).

     | 연산 | N 512 | N 8192 |
     |---|---|---|
     | Q(avg, 8개 ensemble, 공식 vmap 구현) | 0.61 | 0.85 |
     | π | 0.48 | 0.42 |
     | 같은 크기의 단순 mm | 0.35 | 0.30 |

   - profiler: N 8192 Q의 TF32 GEMM(cutlass s1688 256×64) 2회가 116 ms로, Q 전체 184 ms의 63%다.
   - 즉 큰 N에서 rest의 대부분인 vmap Q ensemble의 batched GEMM이 이 PyTorch 판에서 TF32 이득을 거의 못 받는다. 측정 정의(캡처 동안 전역 TF32 허용)는 바꾸지 않고, 이 사실을 기술로만 보고한다.
