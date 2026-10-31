# E8c 추가 지연 측정 (D3: 미리 바꾼 BF16, 계획 step 전체 CUDA graph) — 사전등록

- 작성·고정: 2026-10-31T16:00Z(UTC, 한국 시간 11-01), 조정자. E8c 자료가 생기기 전에 고정한다. sha256은 [LOG.md](LOG.md)에 적는다.
- 배경
  - 사용자가 [CLAIMS_rss.md](CLAIMS_rss.md) §5를 "추천대로 승인"했다. D3는 원고와 함께 하는 추가 측정 두 가지다.
  - (i) 가중치를 미리 bf16으로 바꾼 BF16 기준선. E8의 BF16은 E7 정의대로 autocast라서 호출마다 가중치를 바꾼다.
  - (ii) 계획 step 전체를 CUDA graph로 잡은 지연. E8은 계획 step 전체를 eager로만 쟀다(317M에서 fp32 대비 0.74배).
  - 둘 다 C4(배치와 속도)에 리뷰어가 물을 질문이다.
- 기준: [DESIGN_ext.md](DESIGN_ext.md) §0(장비, 소프트웨어, 통계), §1(E8), §4(진행과 중단). 이 문서는 바꾸거나 더하는 것만 적는다. 나머지는 E8과 같다.
- E8의 판정(L1~L3)은 바꾸지 않는다. E8c는 E8의 범위를 넓히는 부록이다.

## 1. 물음
1. 미리 바꾼 BF16(bf16-cast)까지 기준선에 넣어도, 정수 world-model rollout이 가장 빠른 부동소수점보다 빠른 칸이 있는가(graph-WM).
2. 계획 step 전체를 CUDA graph 하나로 잡으면 정수 world model의 이득이 계획 step에 얼마나 남는가. 그때 공유 dither의 추가 지연은 얼마인가(graph-step).

## 2. 장비와 소프트웨어
- E8과 같다: NVIDIA A40(대체 RTX A6000), SECURE, torch 2.7.1(cu126)과 딸린 Triton, tdmpc2 `332f19e`.
- E8 코드(`E8_latency/code/`)를 복사해 고쳐 쓴다. 정수 경로(Triton int8 타일 GEMM, row kernel, device seed dither)와 크기별 GEMM 설정은 E8 그대로다.
- 새 pod 하나(`arx-e8c`)에서 모든 조건을 다시 잰다. pod가 다르므로 E8 수치와 한 표에 섞지 않는다.

## 3. 조건
- 칸: 크기 {5, 19, 48, 317} × 후보 수 N {512, 2048, 8192} = 12칸(E8과 같음). 모델 차원, 무작위 가중치(seed 0), 양자화 범위 잡는 법, MPPI 기본값은 E8 그대로다.
- WM = dynamics·reward MLP. 나머지 = encoder·π·Q, 그리고 MPPI 연산(표본, top-k, 갱신, 행동 선택)이다.
- WM 정밀도 경로
  - fp32, tf32, bf16(autocast), int-det, int-sh-iter: E8 PLAN §3 그대로
  - **bf16-cast(새)**: dynamics·reward MLP의 매개변수(가중치, bias, LayerNorm)를 측정 전에 한 번 bf16으로 바꾼 복사본이다. 입력을 bf16으로 바꿔 넣고, 출력(다음 잠재, reward logits)을 fp32로 돌린다. autocast는 쓰지 않는다.
  - **int-det-L0tf32(새, 기술, graph-WM만)**: int-det와 같고 층 0의 fp32 GEMM에만 TF32를 허용한다. 정수 경로에서 층 0(양자화하지 않는 입력 [z, a]의 GEMM)이 차지하는 몫을 보려는 것이다. 폐루프 성공률을 잰 수치 경로가 아니므로 판정과 주장에 쓰지 않는다.
- 나머지 모듈의 정밀도
  - **rest-fp32(주)**: 모든 경로에서 fp32, TF32 끔(E7, E8 eager-step과 같음)
  - **rest-tf32(기술, graph-step만)**: 모든 경로에서 나머지 모듈에 TF32를 허용한다. 이때 경로는 tf32(전부 TF32), bf16, bf16-cast, int-det, int-sh-iter 다섯이다(fp32 경로 없음).
- 실행 방식
  - **graph-WM**: E8과 같다(한 제어 step의 world-model rollout을 CUDA graph 하나로 재생). 경로 7개: fp32, tf32, bf16, bf16-cast, int-det-L0tf32, int-det·int-sh-iter 짝
  - **graph-step(주)**: `TDMPC2._plan`(t0 = False, eval_mode = True) 한 번을 CUDA graph 하나로 잡아 재생한 시간이다. encoder, π 궤적, MPPI 6반복(표본, WM rollout, Q, top-k, 갱신), 행동 선택을 모두 포함한다. 관측은 static 입력 버퍼에 넣는다. 재생마다 E8 eager-step의 고정 관측 열(N(0,1) 39차원 200개)에서 다음 관측을 copy하고, 이 copy도 시간에 넣는다. 경로: rest-fp32 6개(fp32, tf32, bf16, bf16-cast, int-det·int-sh-iter 짝), rest-tf32 5개(tf32, bf16, bf16-cast, int-det·int-sh-iter 짝)
  - **eager-step(기술)**: E8 eager-step과 같은 규약(`TDMPC2.act`), N 512만, rest-fp32 경로 6개. 같은 pod에서 graph-step과 비교하려는 것이다.
- 난수
  - MPPI 표본은 PyTorch 기본 CUDA generator를 쓴다. graph 재생마다 새 난수가 나와야 한다.
  - Q의 `return_type='avg'` 안 `torch.randperm`처럼 캡처되지 않는 연산이 있으면, 같은 분포를 내는 graph 호환 연산으로 바꾸고 PLAN에 적는다. 모든 경로에 똑같이 적용한다.
  - dither seed는 E8과 같은 device counter식이다(재생마다 새 u).
- 기능 점검(스모크, 본 측정 전): graph-step 재생이 eager `_plan`과 같은 계산을 하는지 PLAN에 적은 방법으로 확인한다(예: 같은 난수 입력에서 행동, mean, std 일치). 재생마다 MPPI 난수와 dither u가 바뀌는지도 확인한다.

## 4. 측정 규약
- E8(DESIGN_ext §1.4, E8 PLAN §6)과 같다.
  - warmup 30회, block 길이 ≥ 약 2초(최소 10회), block 값 = block 안 회당 시간의 중앙값
  - int-det·int-sh-iter는 같은 (크기, N, 방식, rest)의 두 block을 바로 이어 잰 짝이고, 짝 안 순서는 무작위다. 나머지는 단일 block이다.
  - 10회전, 회전마다 단위 순서를 numpy seed 0으로 새로 섞는다.
  - block마다 SM clock, 온도, 전력, clock event 이유, 호스트 load를 기록한다.
- graph를 모두 메모리에 둘 수 없으면 크기별로 graph를 만들고 지운다. 이때 회전 안에서 크기 순서를 섞고, 크기 안에서 단위 순서를 섞는다. 이 규칙을 PLAN에 적는다.
- 계산: T = 조건 block 값 10개의 중앙값, 비율 = T의 비. 짝 오버헤드 = 짝 10개 (sh-iter − det)/det의 중앙값, CI는 짝 bootstrap 10,000회(RandomState(0), 백분위 95%). 문턱 비교의 허용 오차 1e-9.
- min(float) = min(T(tf32), T(bf16), T(bf16-cast)). E8의 min(fast)에 bf16-cast를 더한 것이다.

## 5. 가설과 판정 [사전등록]
- **L2c(graph-WM)**: T(int-det) / min(float) ≤ 0.90인 칸이 12칸 중 하나 이상이면 성립
- **S1(graph-step, rest-fp32)**: T(int-det) / T(fp32) ≤ 0.80인 칸이 12칸 중 하나 이상이면 성립
- **S2(graph-step, rest-fp32)**: T(int-det) / min(float) ≤ 0.90인 칸이 하나 이상이면 성립
- **S3(graph-step, rest-fp32)**: 12칸 모두 짝 오버헤드 중앙값 ≤ 5%이고, CI 상한 ≤ 10%인 칸이 10칸 이상이면 성립
- 기술 항목(판정 없음)
  - 같은 pod에서 E8의 L1~L3 다시 계산(재현)
  - 칸별 T(bf16-cast)/T(bf16), T(int-det)/T(bf16-cast)
  - T(int-det-L0tf32)/T(int-det)
  - graph-step 대 eager-step(N 512), 같은 경로에서 graph-WM T / graph-step T(계획 step에서 WM rollout의 몫)
  - rest-tf32 graph-step의 T(int-det)/min(float)와 짝 오버헤드
  - 크기별 교차점(graph-step에서 int-det가 fp32, min(float)보다 빨라지는 가장 작은 N)

## 6. 결과의 쓰임 (미리 적음, C4 문구)
- L2c 성립: C4의 "TF32·BF16보다 최대 1.48배"를 "미리 바꾼 BF16을 포함한 가장 빠른 부동소수점보다 빠르다(k/12칸, 최대 x배)"로 고쳐 쓴다. 불성립: 그 부분을 빼고 "미리 바꾼 BF16과는 비슷하다"를 수치와 함께 쓰고, 사용자에게 보고한다.
- S1·S2 성립: "계획 step 전체를 CUDA graph로 잡아도 fp32보다 x~y배, 가장 빠른 부동소수점보다 최대 z배 빠르다"를 C4에 더한다. 불성립: 지금 문구("이득은 world-model rollout에 한정되고 계획 step 전체의 이득은 작다")를 E8c 수치로 유지한다.
- S3 성립: "공유 dither는 계획 step 전체에 ≤ w%를 더한다"를 더한다. 불성립: 수치 그대로 쓴다.
- rest-tf32, int-det-L0tf32, eager-step은 기술 문장으로만 쓴다.
- E8과 E8c의 같은 조건 수치가 다르면 둘 다 보고한다(다른 pod). 원고의 주 수치는 E8(graph-WM)과 E8c(graph-step, bf16-cast)로 나눠 출처를 밝힌다.

## 7. 산출물, 비용, 진행
- 폴더 `rss_ext/E8c_latency/`: `PLAN.md`(본 측정 전), `code/`, `results/`, `LOG.md`, `REPORT.md`(DESIGN_ext §0.5와 같음)
- 상한: `runpod_rss/caps.json`의 `e8c` $3. pod는 생성 4시간 뒤 자동 종료된다.
- 상한의 80%($2.40)에 이르렀는데 본 측정을 끝내지 못할 것 같으면 멈추고 조정자에게 알린다.
- 설계대로 잴 수 없으면(예: `_plan`을 graph로 잡을 수 없음) 멈추고 선택지를 조정자에게 보고한다.
