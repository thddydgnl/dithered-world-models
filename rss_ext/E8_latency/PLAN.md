# E8 지연 영역 지도 — 실행 계획 (본 측정 전 작성)

- 작성: 2026-10-31T09:50Z, E8 작업자(본 측정 시작 09:51Z 전에 완성). 본 측정 자료가 생기기 전이다(스모크와 짧은 보정 측정만 있음).
- 기준: [../DESIGN_ext.md](../DESIGN_ext.md) §0·§1·§4. 가설(L1–L3), 지표, 문턱, 칸(크기 4 × N 3), seed는 설계 그대로이고 여기서는 바꾸지 않는다.
- 이 문서는 운영 세부만 정한다: 스모크 결과, 구현 선택과 그 규칙, 실행 순서, block 길이, 비용 추정.
- 코드: `code/` (e8lib.py, e7dev.py, e8_smoke.py, e8_main.py, e8_analyze.py, dev_rowtiles.py, setup_pod.sh, run.sh)
- 스모크 산출물: `results/smoke/` (env, ranges, gemm_cfg, gemm_tune, checks, selection, calib, rowbench, 로그), `results/dev/rowtiles.json`

## 1. 장비와 소프트웨어

- pod `arx-e8` (`30cyekyjii6pzs`), **NVIDIA A40** 1장(GA102, 84 SM, 46 GB, 전력 상한 300 W, 최대 SM clock 1,740 MHz), SECURE, $0.59/h, driver 595.91.07, 96 vCPU(호스트 공유)
- 측정은 단일 프로세스, GPU 배타(이 pod에 GPU 하나), MPS 없음
- torch 2.7.1+cu126(PyPI), triton 3.3.1, tensordict 0.8.3, numpy 1.24.4, hydra-core 1.3.2, omegaconf 2.3.0, matplotlib 3.10.9, nvidia-ml-py 13.615.71(block별 GPU 상태), tdmpc2 `332f19e4`
- 환경(Meta-World·DMC)은 설치하지 않았다. 모델은 환경 없이 만든다.

## 2. 모델, 입력, 양자화 범위

- 공식 `TDMPC2(cfg)`를 `torch.manual_seed(0)` 뒤에 만든다. 단일 과제(`task=mw-push` 이름만 사용, task embedding 없음), 관측 39, 행동 4, `episode_length = 100`(할인율에만 쓰임, 지연과 무관)
- 크기별 차원(공식 `MODEL_SIZE`): 5M mlp 512·latent 512·num_q 5, 19M 1024·768·5, 48M 1792·768·5, 317M 4096·1376·8
- 공식 초기화는 reward head 가중치를 0으로 둔다. 지연 측정에는 그대로 쓴다. 정확성 점검의 reward 경로는 head만 다시 무작위로 채운 복사본(trunc normal 0.02)을 쓴다(head GEMM을 실제로 확인하려고).
- 양자화 범위: 무작위 입력(관측 ~ N(0,1) 2,048개 → 공식 encoder, 행동 ~ U[−1,1])에서 dyn0·dyn1·rew0·rew1 Mish 출력의 [최솟값, 99.99 백분위]. 값은 `results/smoke/ranges.json`(모두 약 [−0.309, 3.5~4.0]). 본 측정도 같은 seed로 다시 계산하고 스모크 값과 같은지 assert한다.
- WM rollout 입력: z0 = encoder(무작위 관측 1개)를 N번 반복(`TDMPC2._plan`과 같음), 행동은 U[−1,1] 고정 텐서(seed 0)
- eager-step 관측: N(0,1) 39차원 관측 200개 고정 열을 순환(E7은 보정 episode의 관측 열을 썼다. 여기는 환경이 없다)

## 3. 정밀도 경로 구현

- **fp32**: 공식 모듈 그대로, TF32 끔
- **tf32**: dynamics·reward 호출 안에서만 `allow_tf32 = True`(호출 뒤 되돌림). encoder·Q·π는 fp32다(설계 §1.3 "encoder·Q·π는 fp32"). graph-WM·eager-WM에는 영향이 없고 eager-step에서만 범위가 의미 있다.
- **bf16**: dynamics·reward를 `torch.autocast(bfloat16, cache_enabled=False)` 안에서 부르고 출력을 fp32로(E7과 같음). `cache_enabled=False`는 CUDA graph 캡처를 위한 것이다. 호출마다 autocast 구역이 따로라 캐시가 원래 쓰이지 않으므로 하는 일은 같다. 가중치는 호출마다 bf16으로 변환된다(E7의 bf16 정의).
- **int-det / int-sh-iter**(일반 경로, 크기와 무관): E7 수치 경로
  - 층 1·2 가중치: 대칭 출력 채널별 int8 RTN(W8). 활성: 층별 [lo, hi] 격자의 4-bit 코드(0..15)를 int8 그릇에
  - 층 0: `F.linear`(cuBLAS fp32) → row kernel(LayerNorm → Mish → 4-bit 코드)
  - 층 1: int8 GEMM → 역양자화(+bias) → row kernel(LN → Mish → 4-bit 코드)
  - 층 2: int8 GEMM → 역양자화(+bias) → dynamics는 row kernel(LN → SimNorm), reward는 logits 101개
  - GEMM 후보 (a): Triton 타일 int8 GEMM(가중치 K 연속 저장, grouped 순서, int32 누적, 역양자화 epilogue, fp32 출력). 타일 설정은 (M, N, K)마다 후보 17개 중 `triton.testing.do_bench`가 가장 빠른 것(int-det GEMM만으로 잼, `results/smoke/gemm_cfg.json`)
  - GEMM 후보 (b): `torch._int_mm`(cuBLASLt, int32 출력, reward head는 N을 104로 0-padding) + 역양자화를 row kernel 첫 단계로
  - row kernel 배치(고정, det·sh-iter 같음): BLOCK_W = 폭의 다음 2의 거듭제곱, program 하나가 4,096원소 tile 하나(행 max(1, 4096/BLOCK_W)개), 8 warp
- **dither(sh-iter)**: q = floor(x + u), u = Philox(seed, 열). 한 forward의 모든 행(후보)이 같은 u를 쓴다.
  - seed(f, l) = (base·1000003 + f·8 + salt + l) & 0x7FFFFFFF. [base, f]는 MLP마다 device int64 텐서, f는 그 MLP의 forward 수, salt는 dynamics 0·reward 4, l은 층 0·1
  - u는 (층, forward)마다 **한 번만, 한 kernel 앞에서** 그린다. u 버퍼 (2, 폭)에 담는다.
    - 층 0 row kernel: u0을 읽고, program들이 나눠서 u1 = Philox(seed(f, 1))을 그린다.
    - 층 1 row kernel: u1을 읽고, program 0이 f를 1 늘린다.
    - head kernel: program들이 나눠서 다음 forward의 u0 = Philox(seed(f+1, 0))을 그린다. head는 dynamics LN/SimNorm row kernel, 또는 경로 (a)의 reward head GEMM이다.
    - `new_episode(base)`는 [base, 0]으로 두고 u0을 그린다.
  - 같은 위치를 같은 kernel이 읽고 쓰는 일이 없다(stream 순서가 갈라 놓음). 그래서 eager와 CUDA graph 재생 모두 forward마다 새 u가 나온다.
  - det는 counter와 버퍼가 없다. block마다 base = 7·(70000 + 회전) + 13으로 다시 둔다(E7과 같음).
- **E7 융합 kernel(5M, 기술)**: `code/e7dev.py`는 `main/code/tdmpc2/e7lib.py`의 int8 그릇판을 복사한 것이다.
  - 바꾼 것은 seed뿐이다. host 인자 대신 위와 같은 device [base, f]를 읽는다(층 0 kernel은 Y load 뒤, 융합 층 1 kernel은 K 루프 앞).
  - head kernel의 program (0,0)이 시작할 때 f를 늘린다.
  - 나머지(block 크기, 512폭 행 전체 융합, K×N 가중치, dynamics head의 PyTorch LN·SimNorm)는 E7 그대로다. u는 각 program이 그린다(E7 방식).

## 4. 스모크 결과 (`results/smoke/`, v2 = 최종 구현)

### 4.1 정확성 (모두 통과)
- 비교 기준: 같은 격자의 fp32 fake-quant(W8 가중치를 역양자화해 씀, TF32 끔). 층마다 teacher forcing(앞 층의 정수 코드를 기준에 넣음)으로 비교한다.
- 허용 범위
  - 코드가 다른 비율 ≤ 1e-3이고 차는 최대 1단계
  - head 출력 |차| ≤ 1e-4 × max(1, |기준|max)
- 크기마다 경로 (a)·(b) × det·sh-iter × dyn·rew(M 512, dyn은 M 24도), 5M은 E7 kernel까지 확인했다.

| 크기 | forward 점검 | 층 0 코드 불일치(최대) | 층 1 코드 불일치(최대) | head 상대 오차(최대) | end-to-end 상대 L2(참고) |
|---|---|---|---|---|---|
| 5M | 24 | 0 | 3.8e-6 (1단계) | 6.6e-7 | 6.1e-4 |
| 19M | 16 | 1.9e-6 (1단계) | 3.8e-6 (1단계) | 6.3e-7 | 2.1e-3 |
| 48M | 16 | 0 | 1.1e-6 (1단계) | 5.4e-7 | 3.0e-4 |
| 317M | 16 | 4.8e-7 (1단계) | 1.4e-6 (1단계) | 6.6e-7 | 1.3e-3 |

- **sh-iter**: 모든 크기·경로에서 통과했다.
  - forward마다 counter가 1 늘고, 각 forward가 그 forward의 기준 u와 맞는다.
  - 같은 입력의 다음 forward에서 코드가 층 0 약 29~31%, 층 1 약 43~44% 바뀐다(새 u).
  - 같은 행만 있는 입력에서는 모든 행의 코드와 출력이 같다(후보 사이에 u 공유).
  - (a)와 (b)는 같은 상태에서 코드와 출력이 비트 단위로 같다.
- **graph 재생**
  - MLP forward 하나를 잡은 graph: 재생마다 counter +1, 코드 약 28~30% 바뀜, 각 재생이 기준과 맞음
  - rollout 전체 graph: 재생마다 dynamics counter +20(정책 rollout 2 + MPPI 18), reward +18. sh-iter 재생끼리는 출력이 다르고, det 재생끼리는 비트 단위로 같다. det graph와 eager의 출력 차는 0이다.
- fp32·tf32·bf16 rollout도 graph와 eager 출력이 같다(차 0).
- det 대 sh-iter: 층 0 코드의 약 21~22%가 다르다(dither가 실제로 들어감).

### 4.2 GEMM 조율 (`gemm_tune.json`, int-det GEMM만)
- Triton 최적 설정이 `torch._int_mm`보다 44개 모양 중 대부분에서 빠르거나 비슷하다(0.51~1.08배). reward head(N 101)와 M 24에서 특히 빠르다.
- 가장 큰 모양(8192×4096×4096)에서는 Triton 212 TOPS, cuBLASLt가 2% 빠르다. A40 int8 이론값은 299 TOPS(1,740 MHz)다.

### 4.3 구현 변경 이력 (본 측정 전, 결과를 보고 고른 구현 세부임을 밝혀 둠)
1. **dither 생성 위치**: 처음 구현은 E7처럼 모든 program이 seed를 읽고 u 전체를 그렸다. 이때 sh-iter가 graph-WM에서 5M·317M(N 512) 모두 약 +5%였다(4회 × 300/60회 측정). 원인은 317M에서 row kernel마다 program 256개가 4,096개씩 다시 그리는 Philox 계산, 그리고 5M에서 seed load → Philox 사슬이다. 그래서 위 §3의 "한 kernel 앞에서 한 번 그리기"로 바꿨다(같은 측정에서 +0.7~1.7%).
2. **row kernel 배치**: 처음에는 program을 약 256개로 맞추려고 program당 tile을 늘렸다(Philox 상각용). `dev_rowtiles.py`로 tile 수를 쓸어 보니 이 규칙은 det에도 나빴다(M 8192·W 4096: 32 tile 439 µs 대 1 tile 292 µs). 그래서 **det 시간으로 tile 1개를 골랐다**(8개 모양 모두에서 det 최솟값의 4% 안).
   - 대가: 폭 4096에서는 program마다 u(16 KB)를 다시 읽는다. 그래서 row kernel 수준의 sh-iter 추가 시간이 tile 1개일 때 커진다(M 8192·W 4096: det 292 대 dither 327 µs. tile 2개면 304 대 318 µs).
   - 이 선택은 int-det에 유리하고 L3에는 보수적이다. 보고서에 적는다.
3. 위 두 변경 뒤 smoke의 checks·select·calib·rowbench를 다시 돌렸다(v2). 판정에 쓰는 것은 v2다. v1은 pod의 `results/smoke_v1/`(LOG 참고)에만 둔다.

## 5. GEMM 경로 선택 (규칙과 결과; int-det만 봄)

- **규칙**: 크기마다 int-det의 graph-WM 지연으로 고른다.
  - {a, b} × N {512, 2048, 8192}를 3회전, 무작위 순서(numpy default_rng(1))로 잰다.
  - block = warmup 30 + max(10, ⌈1 s / 1회 시간⌉)회. T = block 중앙값 3개의 중앙값
  - N 3개에 걸친 T의 기하평균이 작은 경로를 고른다.
- **결과**(`results/smoke/selection.json`, ms): 네 크기 모두 **(a) Triton**이다.

| 크기 | N 512 a / b | N 2048 a / b | N 8192 a / b | 기하평균 a / b | 선택 |
|---|---|---|---|---|---|
| 5M | 2.228 / 2.314 | 5.868 / 6.039 | 20.39 / 20.53 | 6.435 / 6.595 (b가 2.5% 느림) | a |
| 19M | 3.854 / 4.042 | 12.86 / 13.28 | 46.65 / 47.20 | 13.22 / 13.63 (3.1%) | a |
| 48M | 5.988 / 6.377 | 21.30 / 22.49 | 75.15 / 76.79 | 21.24 / 22.25 (4.8%) | a |
| 317M | 22.34 / 24.42 | 74.78 / 77.84 | 283.0 / 288.6 | 77.90 / 81.86 (5.1%) | a |

## 6. 본 측정 규약과 실행 순서 (`code/e8_main.py`)

- 단위
  - 한 회전에 119단위, 154 block이다.
  - graph-WM: 12칸 × (fp32·tf32·bf16 단일 block 3 + int 짝 1), 5M은 E7 짝 3개를 더한다.
  - eager-WM: graph-WM과 같다.
  - eager-step: N 512 × 크기 4 × (단일 3 + int 짝 1)에 5M E7 짝 1개를 더한다.
- **graph-WM**: rollout 전체를 CUDA graph 하나로 잡아 재생한다. rollout은 정책 궤적(배치 24, dynamics 2회), 그리고 반복 6 × horizon 3 × (reward → dynamics)를 배치 N으로 한 것이다. 66개 graph를 미리 만들어 둔다(약 22초, 할당 3.6 GiB).
- **eager-WM**: 같은 rollout을 eager로 돈다.
- **eager-step**: `TDMPC2.act(obs, t0=block 첫 호출, eval_mode=True)`. N 512, MPPI 기본값(반복 6, horizon 3, elite 64, 정책 궤적 24, temperature 0.5)이고, dynamics·reward 메서드를 정밀도 경로로 바꿔 끼운다(E7과 같음).
- **block**
  - warmup 30회 뒤 n회를 잰다. 매회 CUDA event 쌍으로 재고 device sync를 한다. block 값은 n회의 중앙값이다.
  - **block 길이**: n = max(10, ⌈2,000 ms / 1회 시간⌉). 짝은 det의 1회 시간으로 정한 n을 두 block에 같이 쓴다. 1회 시간은 smoke 보정(`results/smoke/calib.json`의 `t_rep`)이고, 조건별 n은 같은 파일의 `n_reps`다.
  - n의 범위: graph 10~899, eager 10~297, step 13~84. 317M N 8192는 10회(1회 0.28~0.90 s)다.
- **순서**
  - 회전마다 단위 순서를 numpy `default_rng(0)` 순열로 새로 섞는다. 짝 안의 순서는 같은 rng의 `random() < 0.5`로 정한다.
  - 10회전이고, 회전마다 결과를 다시 쓴다(`results/main/e8_main.json`, 회당 시간은 `e8_reps.npz`).
- **block마다 기록**: 회전, 단위 위치, 단위 안 위치, 회당 시간 전부(npz), block 통계, 시작·끝 시각, 호스트 1·5·15분 load, block 동안 호스트 CPU 사용률, block 시작·중간·끝의 GPU 상태(SM·메모리 clock, 온도, 전력, clock event 이유 비트, 보이는 compute 프로세스 수).
- `allow_tf32`가 block 뒤 False인지 assert한다.

## 7. 분석 (`code/e8_analyze.py`, 본 측정 전에 작성)

- **사전 등록(설계 §1.4–1.5, graph-WM)**
  - T = 조건의 block 값 10개의 중앙값. 비율은 T의 비다.
  - 짝 오버헤드 = 짝 10개 (sh-iter − det)/det의 중앙값이다. CI는 짝 bootstrap 10,000회, 칸마다 `RandomState(0)`, 백분위 95%다.
  - 문턱 비교의 허용 오차는 1e-9다.
  - L1: T(int-det)/T(fp32) ≤ 0.80인 칸이 1개 이상
  - L2: T(int-det)/min(T(tf32), T(bf16)) ≤ 0.90인 칸이 1개 이상
  - L3: 12칸 모두 짝 오버헤드 중앙값 ≤ 5%이고, CI 상한 ≤ 10%인 칸이 10개 이상
- **기술(설계 §1.5 + 이 계획)**
  - 크기별 교차점: int-det가 fp32보다 빨라지는 가장 작은 N, min(tf32, bf16)보다 빨라지는 가장 작은 N
  - eager-WM 대 graph-WM(비율과 절대 시간), eager-step 비율
  - (5M, N 512) E7 재현: eager-step int-det/fp32를 새 경로·E7 kernel 모두로 구해 E7(A5000) +17.7%와 비교. E7 kernel 대 새 경로를 모든 방식·N에서 비교
  - 비율의 회전 bootstrap 95% CI(10회전을 함께 다시 뽑음, RandomState(0), 10,000회). 판정에는 쓰지 않는다.
  - 측정 상태: SM clock, 온도, 전력, 전력 상한 이유 비트의 비율, 호스트 load, CPU 사용률, 자기 조건 T보다 15% 넘게 느린 block의 비율(E7의 두 모드 점검)
  - 그림: `e8_ratio_graph.png`(크기별 패널, x = N, y = fp32 대비 비, 선 = 정밀도), `e8_ratio_eager.png`, `e8_latency_graph.png`(절대 ms), `e8_overhead.png`(짝 오버헤드와 CI)
- 결과를 본 뒤 정하는 분석은 [사후]로 표시한다.

## 8. 비용과 시간 추정

- 지금까지(09:49Z): e8 $0.36(pod 생성 09:14Z부터)
- 본 측정 추정: 회전당 644초 × 10 + graph 생성과 모델 준비 약 1분 = **약 108분**($1.10)
- 분석·당겨 오기·삭제: 약 20분($0.20)
- **총 추정 약 $1.7**: 상한 $3의 57%, 80%선 $2.40 아래다. pod 자동 종료(13:13Z) 전에 끝난다(본 측정 종료 예상 약 11:45Z).
- E8b(선택)는 E8이 끝난 뒤 지출을 보고 정한다(태그 e8b, 상한 $2, 1.5시간).

## 9. 설계 대비 편차·해석 (본 측정 전에 밝힘)

1. tf32의 범위를 dynamics·reward 호출로 한정했다. 설계 §1.3의 "encoder·Q·π는 fp32"를 따른 것이고, eager-step에만 영향이 있다.
2. bf16 autocast는 `cache_enabled=False`다(graph 캡처용, 하는 일은 같음).
3. sh-iter 난수는 E7과 같은 Philox(tl.rand)·같은 일정이다. 다만 seed 식은 device counter식이고, u는 한 kernel 앞에서 한 번 그린다(§3). 값은 E7의 난수와 다르다(E7도 E1과 달랐음).
4. E7 kernel은 device seed로 바꾼 복사본이다(graph 재생에 필요). 나머지는 E7 그대로다.
5. 지연 측정은 공식 초기화(reward head 0)를 그대로 쓴다. 정확성 점검의 reward만 head를 다시 무작위로 채운 복사본을 썼다.
6. eager-step 관측은 무작위 고정 열이다(환경 없음). 계획 계산량은 관측 값과 무관하다.
