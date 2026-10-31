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
- "저비트로 계획 step 전체가 2~3배 빨라진다"(graph로 잰 것은 world-model rollout뿐이다. 계획 step 전체는 eager에서 0.74배다)
- "4-bit 활성이 8-bit보다 빠르다"(이 GPU에서 같은 int8 dot)
- "큰 모델은 반올림에 더 강하다"(E10 1차 k가 5M과 비슷)
- "어떤 1-step 오차로도 손상이 보이지 않는다"(보상 오차는 det를 가른다)

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
