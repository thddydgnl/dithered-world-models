# ① RSS 2027 주장 재구성안 (사용자 승인용)

- 작성: 2026-10-31(UTC), 조정자. E8~E10이 모두 끝나고 판정 수치를 원자료에서 다시 계산한 뒤에 썼다([LOG.md](LOG.md)).
- 이 문서는 **제안**이다. 주장 변경은 사용자 승인으로만 하고, 승인된 문구로 RSS Stage 1 원고(5쪽)를 쓴다.
- 근거: [../main/reports/FINAL_A.md](../main/reports/FINAL_A.md)(E0~E7), [E8_latency/REPORT.md](E8_latency/REPORT.md), [E9_fp4/REPORT.md](E9_fp4/REPORT.md), [E10_mt317m/REPORT.md](E10_mt317m/REPORT.md)

## 0. 무엇이 바뀌나

| | JKAIA 포스터(10-25) | RSS 제안 |
|---|---|---|
| 속도 | "no speed-up over fp32"(한계) | CUDA graph 실행에서 정수 world model rollout이 fp32보다 1.9~3.3배, TF32·BF16보다 최대 1.5배 빠르다(E8). E7의 느림은 eager 실행의 kernel 호출 비용 때문이었다 |
| 크기 | 5M 단일 과제 모델 | 317M 다중 과제 모델에서도 같은 실패와 처방(E10, 8/8과제) |
| 형식의 경계 | 없음 | 블록 스케일 FP4는 손상을 크게 줄인다(E9). 실패는 공유 scale의 거친 격자에서 생긴다 |
| 진단 문장 | "one-step error no larger than dither's" | **잠재(dynamics) 1-step 오차**로 한정. 보상 1-step 오차는 det를 가른다([사후] 관찰) |

## 1. 제목과 한 줄 주장
- 제목은 그대로 둔다: *Planners Exploit Deterministic Error: Dithered Low-Bit World Models* (FINAL_A [결정 9]).
- 한 줄 주장(영문 초안): "At the same bit width, the structure of rounding error, not its size, decides whether a low-bit world model plans well: iterating planners exploit deterministic rounding error that persists across iterations, and dither shared by all candidates and redrawn at every iteration removes most of the damage at no measurable latency cost."

## 2. 주장 목록

**C1 현상과 진단 (문구 한정)**
- 주장: 같은 비트에서 det는 잠재 1-step 예측 오차로는 드러나지 않는 계획 손상을 만든다.
- 근거: 손상 칸에서 det의 잠재 1-step 오차가 sh-iter 이하인 경우가 E1 29/36, E10 8/8(det 0.58~6.36 대 sh-iter 1.06~9.53), E9 6/6(0.48~0.63배)이다. det 하락은 최대 100%p다.
- 한정
  - "1-step 오차"는 **잠재(dynamics) 오차**로 쓴다.
  - [사후] 보상 1-step 오차는 det가 더 크다: E1 Meta-World 19/19칸, DMC 7/12칸, E10 8/8과제(det 4.9~10.1 대 sh-iter 0.005~1.36). det 오차는 단위 사이에 상관되어 보상 같은 스칼라 출력에서 쌓이고, dither 오차는 단위마다 독립이라 상쇄된다는 설명이 기전과 맞는다. 다만 이 설명은 검정하지 않았다. 원고에는 관찰과 Stage 2 가설로만 쓴다.
  - 3·5-step 오차가 det를 가르는 점(FINAL_A 4.1)도 그대로 밝힌다.

**C2 처방**
- 주장: 모든 후보가 공유하고 반복마다 새로 뽑는 dither(sh-iter)가, 같은 비트에서 학습 없이 det 손상 대부분을 없앤다.
- 근거
  - E1: Meta-World 19/19칸에서 손상의 절반 이상 제거(중앙값 95%), DINO-WM 4/5칸
  - E10: mt80-317M Meta-World 8/8과제(red 0.75~1.04). 8과제 모두 sh-iter가 det보다 유의하게 덜 해롭다.
  - E7: 실제 int4 kernel에서도 같은 방향과 크기
- 범위: 반복하는 계획기. Meta-World, DINO-WM, 317M 다중 과제 모델. DMC는 기전으로 설명되는 경계다(S_det, ρ 0.83). E9의 DMC FP4 칸도 같은 양상이다(F2 DMC 2/5).

**C3 기전 (변경 없음)**
- 계획기는 반복 사이에 지속되는 오차를 이용한다. 근거는 H2a′ 4/4, sh-step·sh-ep 양성 대조(70~130% 회복), H2c 3/4, det+shift다(FINAL_A 4.1의 한정 그대로).

**C4 배치와 속도 (새 주장)**
- **11-01 E8c 뒤 §9(E1)에서 다시 정했다. 지금의 C4는 §9의 문장이다. 아래는 E8 기준의 원래 문구다.**
- 주장: 정수 저비트 world model은 실제로 빠르고, 공유 dither는 지연을 거의 더하지 않는다.
- 근거(E8, A40, CUDA graph로 재생한 world-model rollout, 5M~317M × 후보 512~8192)
  - W8A4 정수 경로가 12/12칸에서 fp32보다 1.9~3.3배 빠르다.
  - 10/12칸에서 TF32·BF16 중 빠른 쪽보다 1.11배 이상 빠르고, 최대 1.48배다.
  - 공유 dither의 짝 오버헤드는 −0.2~+2.1%다.
- 한정
  - eager 실행의 작은 모델에서는 정수 kernel이 느리다(E7 재현, 5M에서 +11~28%). 이것은 kernel 호출 비용 때문이다.
  - 계획 step 전체(eager, Q·π fp32)의 이득은 작다. 317M에서 fp32 대비 0.74배, 가장 빠른 부동소수점 대비 0.95배다.
  - BF16 기준선은 E7 정의대로 autocast(호출마다 가중치 변환)다.
  - 4-bit 활성은 int8 그릇에 담아 int8 dot을 쓰므로 8-bit보다 빠르지 않다.
- 기술(E8b, RTX 5090): 폭 4096 층에서 NVFP4 GEMM은 BF16보다 약 5배, int8은 약 3배 빠르다.

**C5 실패가 생기는 범위 (새 주장, 경계)**
- 주장: 실패는 공유 scale을 쓰는 거친 격자에서 생긴다. 미세한 블록 스케일은 오차 크기를 줄여 손상을 대부분 없애고, dither는 오차의 지속을 없앤다. 두 처방은 서로 다른 축이다.
- 근거
  - E7: 8-bit(W8A8)는 손상이 없다.
  - E9 NVFP4(블록 16): 손상은 DMC 2과제뿐이고 Meta-World는 0/8이다.
  - E9 MXFP4(블록 32, 2의 거듭제곱 scale): 4/12과제에서 손상이 생겼다(cheetah +31.3%).
  - 두 FP4 모두 E1 균일 4-bit보다 손상이 훨씬 작다(cheetah 71.6% → 9.9%·31.3%).
  - 손상이 남은 곳에서는 det가 잠재 1-step 오차는 더 작은데 더 해롭다(E9 F3 6/6).
- 한정: E9는 TD-MPC2 5M에서만 쟀다(DINO-WM 없음). F2 Meta-World는 손상 칸이 하나뿐이라 처방 근거로 쓰지 않는다.

**C6 크기 (새 주장)**
- 주장: 317M 다중 과제 모델에서도 같은 실패와 처방이 나온다. 이 크기는 정수 kernel 이득이 가장 큰 크기다(E8: fp32 대비 0.31~0.35배).
- 근거: E10 G1 8/8, G2 8/8, G3 8/8
- 한정: 317M의 1차 k(3~5)는 5M(3~4)과 비슷하다. "큰 모델이 더 견딘다"고 쓰지 않는다. 체크포인트는 하나다(seed 하나).

## 3. 쓰지 않을 주장 (FINAL_A 4.3에 더함)
- "가속기의 FP4 형식에서도 같은 실패가 생긴다"(E9 NVFP4 F1 불성립)
- "저비트로 계획 step 전체가 2~3배 빨라진다"(2~3배는 world-model rollout의 값이다. 계획 step 전체는 CUDA graph 하나로 잡아도 FP32보다 1.28~1.49배 빠를 뿐이다(E8c). E8의 eager 측정에서는 FP32 시간의 0.74배였다)
- "4-bit 활성이 8-bit보다 빠르다"(이 GPU에서 같은 int8 dot)
- "큰 모델은 반올림에 더 강하다"(E10 1차 k가 5M과 비슷)
- "어떤 1-step 오차로도 손상이 보이지 않는다"(보상 오차는 det를 가른다)
- "정수 W8A4 world model이 BF16보다 빠르다"(A40에서 미리 바꾼 BF16이 rollout과 계획 step 전체 모두에서 더 빠르다, E8c. 11-01 §9에서 더함)
- "world model 연산이 계획 step 비용을 지배한다"(dynamics·reward는 FP32 계획 step의 40~52%다, E8c. 11-01 §9에서 더함)

## 4. 동기 문단 (영문 초안)
> Sampling-based planners such as MPPI and CEM query a learned world model thousands of times per control step, so the world model's arithmetic dominates planning cost. Low-bit integer arithmetic cuts that cost: replayed as a CUDA graph, the planning rollouts of a W8A4 TD-MPC2 world model run 1.9–3.3× faster than FP32 and up to 1.5× faster than TF32/BF16 on an Ampere GPU, from 5M to 317M parameters. Yet at aggressive bit widths planning collapses, by up to 100 points of success, while in most affected settings the world model's latent one-step error is no larger than that of a stochastic rounding scheme that plans well. We show that what matters is the structure of the rounding error, not its size.

## 5. 결정이 필요한 것

| # | 정할 것 | 추천 |
|---|---|---|
| D1 | C1~C6 주장 묶음 | 승인. 특히 C4(속도), C5(FP4 경계), C6(317M) |
| D2 | C1을 잠재 1-step 오차로 한정하고, 보상 오차 관찰은 Stage 2 가설로 쓰기 | 예 |
| D3 | 원고와 병행할 추가 측정(선택, A40 약 2시간, 약 $1~2): (i) 가중치를 미리 bf16으로 바꾼 BF16 기준선, (ii) 계획 step 전체의 CUDA graph 지연 | (i)·(ii) 모두. C4의 리뷰어 질문 두 개를 막는다. 예산이 $19 남아 있다 |
| D4 | 이중맹검 대응: 공개 CV에 같은 제목과 "RSS 2027"이 함께 있다 | 심사 중 CV에서 "RSS 2027" 표기를 뺀다. 제목은 그대로 둔다 |
| D5 | Stage 2(Extension Charter)에 미리 적을 확장 | 보상 오차 진단의 사전등록 검정, DINO-WM 실제 정수 kernel, 계획 step 전체 graph, 엣지 장비(Jetson) 측정 |

## 6. Stage 1 원고 구성 제안 (5쪽 + 참고문헌 1쪽)
1. 서론: 동기 문단(4절), 기여 넷(C1·C2·C3, C4~C6)
2. 설정: 양자화 위치, 변형(det, sh-iter, sh-step, det+shift), 계획기, 내부 사전등록
3. 현상과 처방: 표 1(사전등록 판정 전체: E1~E10), 그림 1(손상 칸 det 대 sh-iter, 5M과 317M)
4. 기전: drift와 양성 대조, 반복 수, 후보 수(그림 2)
5. 배치: E8 지연 지도(그림 3), FP4 경계(E9), 317M
6. 한계와 Stage 2 계획: DMC, PTQ 결합(H3b), 보상 오차 가설, 계획 step 전체

## 7. 승인 기록 (2026-10-31T15:50Z, 한국 시간 11-01)
- 사용자 답: "추천대로 승인". §5의 추천을 모두 받아들였다.
- D1: C1~C6을 원고의 주장 묶음으로 쓴다. §1의 제목과 한 줄 주장, §3의 쓰지 않을 주장도 함께 정해졌다.
- D2: C1은 잠재(dynamics) 1-step 오차로 한정한다. 보상 1-step 오차 관찰은 [사후] 관찰과 Stage 2 가설로만 쓴다.
- D3: (i) 미리 바꾼 BF16 기준선과 (ii) 계획 step 전체의 CUDA graph 지연을 E8c로 잰다. 설계는 [DESIGN_E8c.md](DESIGN_E8c.md)(자료 전 고정)이다. C4의 최종 수치와 문구는 E8c 결과와 그 설계 §6의 규칙으로 정한다.
- D4: 심사 기간에 공개 CV에서 "RSS 2027" 표기를 뺀다. 제목은 그대로 둔다. CV를 고치면 공개 홈페이지·GitHub 프로필·CV PDF에 바로 반영되고, 지원서용 PDF에서도 빠진다. 그래서 제출 직전에 하도록 원고 제출 점검표에 넣는다(`paper/rss2027/README.md`).
- D5: Stage 2(Extension Charter)에 미리 적을 확장은 보상 오차 진단의 사전등록 검정, DINO-WM 실제 정수 kernel, 계획 step 전체 graph(E8c가 일부를 미리 함), 엣지 장비(Jetson) 측정이다.

## 8. E8c 결과에 따른 C4 수정안 (사용자 승인용, 2026-10-31T21:52Z, 한국 시간 11-01 06:52)

- 근거: [E8c_latency/REPORT.md](E8c_latency/REPORT.md). 판정은 조정자가 원 block 2,400개에서 다시 계산했다([LOG.md](LOG.md) 21:50Z 항목).
- 모든 수치는 NVIDIA A40 한 장, pod 하나에서 잰 것이다. 출처는 둘로 나뉜다.
  - rollout과 FP32의 비교는 E8에서 왔다.
  - 계획 step 전체와 미리 바꾼 BF16의 비교는 E8c에서 왔다(설계 §6).

### 8.1 결과 요약

| 가설 | 판정 | 수치 |
|---|---|---|
| L2c: 정수 rollout이 가장 빠른 부동소수점(미리 바꾼 BF16 포함)의 0.9배 이하인 칸이 있다 | 불성립 0/12 | 정수 rollout이 미리 바꾼 BF16보다 1.07~1.52배 오래 걸린다. 12칸 모두 미리 바꾼 BF16이 가장 빠르다 |
| S1: 계획 step 전체(CUDA graph 하나)가 FP32의 0.8배 이하 | 성립 12/12 | FP32보다 1.28~1.49배 빠르다 |
| S2: 계획 step 전체가 가장 빠른 부동소수점의 0.9배 이하 | 불성립 0/12 | 미리 바꾼 BF16보다 1.01~1.12배 오래 걸린다 |
| S3: 계획 step 전체에서 dither 오버헤드 ≤ 5% | 성립 | 칸별 중앙값은 최대 0.66%, CI 상한은 최대 0.76%다 |

- 기술 항목(판정 없음)
  - 미리 바꾸면 BF16 rollout이 autocast보다 1.5~2.1배 빨라진다.
  - FP32 계획 step에서 dynamics·reward의 몫은 40~52%다.
  - 층 0(양자화하지 않는 FP32 GEMM)만 TF32로 바꾸면 정수 rollout이 미리 바꾼 BF16의 0.72~0.96배가 된다. 다만 이것은 결과를 본 뒤 계산한 값이고, 이 경로는 폐루프로 재지 않았다.
  - 미리 바꾼 BF16도 폐루프 성공률은 재지 않았다. LayerNorm 등까지 bf16으로 계산하며, fp32 대비 출력 차는 0.5~0.9%다.

### 8.2 결정이 필요한 것

| # | 정할 것 | 추천 |
|---|---|---|
| E1 | C4 재정의 | 새 C4: "정수 경로에서 공유 dither는 지연을 거의 더하지 않는다(rollout ≤ 2.1%, 계획 step 전체 ≤ 0.7%). 정수 경로는 FP32보다 빠르다(rollout 1.9~3.3배, step 1.28~1.49배). 그러나 A40에서는 미리 바꾼 BF16보다 느리다(1.07~1.52배, 1.01~1.12배)." 원래 C4의 "정수 저비트 world model은 실제로 빠르다"는 이렇게 범위를 좁힌다 |
| E2 | L2c 문구 | 설계 §6에 미리 적은 "미리 바꾼 BF16과 비슷하다"는 쓰지 않는다. 대신 수치를 그대로 쓴다. 층 0 관찰은 "폐루프로 재지 않음"을 밝히고 기술 문장 하나로만 쓴다 |
| E3 | S1만 성립한 경우(§6에 분기 없음) | S1 수치와 S2 불성립을 한 문장에 같이 쓴다. "계획 step 전체의 이득은 rollout보다 작다"는 기존 취지는 유지한다(1.9~3.3배 → 1.28~1.49배) |
| E4 | 서론 동기 문장 | "world model 연산이 step 비용을 지배한다"를 "망 평가가 지배한다"로 고친다. E8c에서 dynamics·reward는 FP32 step의 40~52%이기 때문이다. 속도 수치는 초록과 6절에만 둔다(쪽수도 줄어듦) |
| E5 | Stage 2 charter(D5 변경) | "계획 step 전체 graph" 항목은 E8c로 끝났다. 이 항목을 "BF16과의 차 줄이기"로 바꾼다: 층 0을 FP32에서 옮긴 정수 경로와 미리 바꾼 BF16을 폐루프로 평가한다. Jetson 항목은 미리 바꾼 BF16을 기준선으로 쓴다 |

- §3 "쓰지 않을 주장"에 다음 둘을 더한다.
  - "정수 W8A4 world model이 BF16보다 빠르다"
  - "world model 연산이 계획 step 비용을 지배한다"

### 8.3 원고 문구 (영문 초안, 승인되면 넣음)

- 초록
  > Captured as CUDA graphs on an Ampere GPU, a W8A4 integer TD-MPC2 world model runs its planning rollouts 1.9–3.3× and the whole planning step 1.3–1.5× faster than FP32, though not faster than BF16 with pre-converted weights; shared dither adds at most 2.1% latency.
- 서론
  > Network evaluations therefore dominate the cost of each control step, and low-bit arithmetic is a direct way to cut it (Sec. VI).
- 기여 목록
  > Shared dither adds at most 2.1% to the planning rollouts of an integer TD-MPC2 world model and under 1% to the whole planning step, which the integer path makes 1.3–1.5× faster than FP32 but not faster than BF16 with pre-converted weights; in TD-MPC2, block-scaled FP4 formats mark the boundary of the failure.
- 6절: "1.19–1.48× faster than the faster of TF32 and BF16"와 eager 계획 step 문장을 다음으로 바꾼다.
  > BF16 is a stronger baseline when its weights are converted once in advance instead of at every call (autocast): it is then the fastest float format in all 12 cells, and our integer rollout takes 1.07–1.52× its time. Much of the gap is the first layer, which we keep in FP32; with TF32 there, the integer rollout takes 0.72–0.96× the time of pre-converted BF16, but we have not evaluated that path in closed loop. Because the dynamics and reward networks are only 40–52% of an FP32 planning step, the whole step (encoder, policy, Q-ensemble and MPPI captured as one CUDA graph) gains less: 1.28–1.49× over FP32, but 1.01–1.12× slower than with pre-converted BF16; shared dither adds at most 0.7% to it.
- 표 1: 두 줄을 더한다.
  - L2c: fails, 0/12 (1.07–1.52×)
  - S1–S3: S1 holds 12/12; S2 fails 0/12 (1.01–1.12×); S3 holds (max 0.7%)
- 한계: "the latency map from one GPU type" 뒤에 다음을 더한다.
  > on which BF16 with pre-converted weights, not evaluated in closed loop, is faster than our integer path
- 그림 3
  - (a)는 그대로 둔다(E8 rollout, FP32 대비).
  - (b)는 E8c 계획 step 전체로 바꾼다. 크기별로 FP32 대비 속도 향상을 보이고, 정수 W8A4와 미리 바꾼 BF16 두 계열을 그린다. 막대는 N 범위다.
  - 캡션에 두 pod라는 것을 밝힌다.
- 쪽수: 늘어나는 것은 약 8줄이다. 관련 연구와 FP4 문단의 표현을 줄여 6쪽을 지킨다. 주장은 빼지 않는다.

## 9. C4 수정 승인 기록 (2026-11-01T02:25Z, 한국 시간 11-01 11:25)
- 사용자 답: "추천대로 승인". §8.2의 추천 E1~E5와 §3에 더할 두 항목을 모두 받아들였다.
- E1: 지금의 C4는 다음 문장이다. "정수 경로에서 공유 dither는 지연을 거의 더하지 않는다(rollout ≤ 2.1%, 계획 step 전체 ≤ 0.7%). 정수 경로는 FP32보다 빠르다(rollout 1.9~3.3배, step 1.28~1.49배). 그러나 A40에서는 미리 바꾼 BF16보다 느리다(1.07~1.52배, 1.01~1.12배)." §2의 원래 C4 문구는 이 문장으로 대신한다.
- E2: L2c는 수치로 쓰고 "비슷하다"는 쓰지 않는다. 층 0 관찰은 "폐루프로 재지 않음"을 밝힌 기술 문장 하나로만 쓴다.
  - 조정자 메모: 층 0을 TF32로 바꾼 정수 rollout과 미리 바꾼 BF16의 비(0.72~0.96)는 결과를 본 뒤 고른 비교다. 설계 §5의 기술 항목은 T(int-det-L0tf32)/T(int-det)뿐이었다. 그래서 원고의 규칙("Analyses chosen after seeing data are labeled post hoc")대로 그 문장에 post hoc를 붙인다. 주장을 바꾸는 것은 아니다.
- E3: S1의 수치와 S2 불성립을 한 문장에 쓴다.
- E4: 서론의 동기 문장을 "망 평가가 step 비용을 지배한다"로 고친다. 속도 수치는 초록과 6절에만 둔다.
- E5: Stage 2 charter의 "계획 step 전체" 항목을 "BF16과의 차 줄이기"로 바꾼다. Jetson 항목은 미리 바꾼 BF16을 기준선으로 쓴다. §7의 D5 목록은 이렇게 고쳐 읽는다.
- 원고에는 §8.3의 문구를 넣는다(`paper/rss2027/main.tex`). 쪽수를 맞추려고 고친 표현은 커밋 메시지와 README에 적는다.
- 조정자 메모(원고 반영 때, 2026-11-01T02:36Z): rollout의 dither 오버헤드 상한을 2.1%에서 2.3%로 고쳤다.
  - E8c의 rollout 재현에서 오버헤드 최대가 2.29%였다(E8은 2.07%). §8.3 초안은 E8 값만 썼다.
  - 설계 §6("E8과 E8c의 같은 조건 수치가 다르면 둘 다 보고한다")에 따라 6절과 표 1에는 두 값을 모두 적었다. 초록과 기여에는 두 측정을 모두 덮는 상한 2.3%를 썼다.
  - 주장("공유 dither는 지연을 거의 더하지 않는다")은 그대로다. 위 E1 문장의 "rollout ≤ 2.1%"는 "≤ 2.3%(두 측정)"로 읽는다.
