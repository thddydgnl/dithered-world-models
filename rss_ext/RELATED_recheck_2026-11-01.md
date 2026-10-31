# 논문 A 관련 연구 재점검 (2026-11-01)

- 목적: RSS 2027 Stage 1 확장 초록(마감 2026-12-04)을 쓰기 전에, 논문 A의 주장과 겹치는 연구를 다시 찾는다.
- 범위: 2026-10-15 이후 새로 나온 연구, 그리고 기존 목록 8편이 빠뜨린 2025–2026년 연구.
- 확인 방식
  - arXiv ID는 모두 `arxiv.org/abs/<ID>` 페이지를 열어 제목, 저자, v1 날짜를 확인했다. 열지 않은 ID는 적지 않았다.
  - ICLR 2027 투고(OpenReview, 익명)는 OpenReview API로 제목, 투고일, 공개일(2026-10-03), 초록을 확인했다. arXiv판은 찾지 못했다. ID 앞에 `OR`을 붙였다.
  - OpenReview 본문 PDF는 사이트의 브라우저 확인(challenge) 화면 때문에 열지 못했다. 그래서 OpenReview 항목은 **초록만 보고 판단**했다. 위협이 높은 두 편은 사람이 직접 읽어야 한다(5절).
  - arXiv 항목 중 핵심 6편(SteerQuant, Q-WAM, QuantWAMs, LatentQuant, QCRL, QbD-VLA)과 2609.32922, 2609.39185, 2610.01889, 2608.06564, 2610.12910, 2602.11882, 2604.12374는 본문에서 키워드("stochastic rounding", "dither", "MPPI", "CEM", "CUDA graph", "NVFP4" 등)를 직접 확인했다.
- 검색 방법은 부록에 적었다.

## 0. 요약

- 표에 넣은 새 관련 연구는 **67편**이다. arXiv 44편, OpenReview(ICLR 2027 투고) 23편이다.
  - 핵심 표(2.1) 23편: 위협 높음 2, 중간 15, 낮음 6(반드시 인용할 것)
  - 참고 표(2.2) 44편: 모두 위협 낮음
  - 2026-10-15 이후 arXiv v1은 6편뿐이다(AGQ, QbD-VLA, VLA-HWBench, Events-Not-Pixels, OpTTA, Best-of-N WAM). 위협이 큰 새 연구는 주로 2026-10-03에 공개된 ICLR 2027 투고에서 나왔다.
- 위협 높음
  - **What Must Quantization Preserve in Latent World Models?** (OR 3VSa1Kx7HD): LeWM·DINO-WM·PLDM을 W4로 양자화한 동시 연구다. 보정(Module-MSE)만으로 FP 수준의 계획 성공을 되찾는다. "저비트에서 계획이 무너진다"는 우리 동기 문장을 약하게 만든다.
  - **When Can a World Model Decide? Certifying Elite Decisions in Sampling-Based Planning** (OR cVCKr4bBU2): CEM 반복이 진행될수록 후보 차이가 모델 해상도 아래로 몰린다는 것을 LeWM에서 쟀다(인증 비율 0.67→0.29). 공백 4(near-tie 밀도)를 일부 닫는다. "반복하는 계획기가 모델 오차에 걸린다"는 C3의 서사도 일반 모델 오차 판으로 먼저 냈다.
- 공백 판정
  - 공백 1(world model·VLA 추론의 dither 처방): **열림.** 단, world model 밖에서는 추론 시 SR·dither가 이미 쓰인다(Nemotron 3 Super, 2609.39185, 2610.01889).
  - 공백 2(결정적 압축 오차의 착취를 직접 측정): **열림.** 단, 일반 모델 오차의 선택 편향 측정은 생겼다. 공백 문구를 "반올림 오차"로 좁혀야 한다.
  - 공백 3(같은 오차 분포에서 공유·지속만 바꾸는 통제): **열림.**
  - 공백 4(near-tie 밀도와 정밀도 flip): **부분적으로 닫힘.**
- 기존 8편: 2026-10-15 이후 새 arXiv 버전은 없다. Min et al.과 DRPE는 ICLR 2027에 투고되었다(초록 취지 동일). DRPE는 v2(2026-09-30)가 있다. PreDE·VLAQuantBench 그룹은 후속 QbD-VLA(2610.38661)를 냈다.

## 1. 기존 목록 8편의 새 버전과 후속

| arXiv ID | 첫 저자 | 현재 버전 | 새 버전·후속·인용 | 비고 |
|---|---|---|---|---|
| [2610.05550](https://arxiv.org/abs/2610.05550) | Min et al. | v1(2026-10-04)만 | ICLR 2027 투고 [OR h0B8U1nn68](https://openreview.net/forum?id=h0B8U1nn68)(투고 09-19, 공개 10-03). 초록은 arXiv와 같은 취지 | 초록에 양자화, 반복 간 지속, dither, near-tie가 없다. Semantic Scholar 인용은 아직 0건(색인 지연) |
| [2602.02110](https://arxiv.org/abs/2602.02110) | Fu et al. | v1(2026-02-02)만 | Semantic Scholar 인용: LatentQuant(2610.03959), Ranganath et al.(2602.11882), LLM 논문 1편 | Fu et al. 자체의 ICLR 2027 투고는 찾지 못했다 |
| [2609.19441](https://arxiv.org/abs/2609.19441) | Xu et al.(PreDE) | v1(2026-09-16)만 | 같은 그룹(Jiuyi Xu, Meida Chen)의 후속 QbD-VLA(2610.38661, 10-28) | 후속도 반올림은 RTN·무보정이다 |
| [2607.10362](https://arxiv.org/abs/2607.10362) | You et al. | v1(2026-07-11)만 | Semantic Scholar 인용 9건(Min et al. 포함). 제목 기준으로 양자화를 다룬 것은 없다 | |
| [2609.32322](https://arxiv.org/abs/2609.32322) | Wang, Fan, Huang(DRPE) | **v2(2026-09-30)** | ICLR 2027 투고 [OR hlfQI5hrMQ](https://openreview.net/forum?id=hlfQI5hrMQ). 인용: Events, Not Pixels(2610.29435) | BibTeX는 v2 기준으로 |
| [2610.00921](https://arxiv.org/abs/2610.00921) | Obst & Stolzenburg | v1(2026-10-01)만 | 없음 | |
| [2609.25376](https://arxiv.org/abs/2609.25376) | Xu et al.(VLAQuantBench) | v1(2026-09-21)만 | 후속 QbD-VLA(같은 1저자). AGQ(2610.33074)가 인용 | |
| [2602.03782](https://arxiv.org/abs/2602.03782) | Xu et al.(QVLA) | v1(2026-02-03)만, ICLR 2026 | 2026-09~10월 VLA·WAM 양자화 논문 다수가 인용(2.2) | |

## 2. 관련 논문 표

열 설명
- 닿는 곳: C1 현상, C2 dither 처방, C3 지속 기전, C4 지연, C5 FP4 경계, C6 317M, 공백 1–4
- 구분: [신규]는 2026-10-15 이후, [ICLR27]은 2026-10-03 공개된 ICLR 2027 투고, [누락]은 그 전의 2025–2026년 연구로 기존 8편 목록에 없던 것이다. [누락] 중 프로젝트의 NOVELTY_R1·R2에 이미 있던 것은 따로 적었다.

### 2.1 핵심 표 (위협 높음·중간, 그리고 반드시 인용할 낮음)

| # | ID | 제목 | 저자 | v1 | 보인 것 (한 줄) | 닿는 곳 | 위협 | 인용·차별화 |
|---|---|---|---|---|---|---|---|---|
| 1 | [OR 3VSa1Kx7HD](https://openreview.net/forum?id=3VSa1Kx7HD) [ICLR27] | What Must Quantization Preserve in Latent World Models? | Anonymous | 투고 2026-09-16, 공개 10-03 | LeWM·DINO-WM·PLDM, 4과제, 대칭 per-channel W4. 양자화 표현 오차가 goal distance와 후보 순위로 퍼지는 경로를 분석한다. 보정 목표 Module-MSE로 평균 성공 79.33%(FP 79.83%)를 얻고 "planning-decision fidelity"가 가장 좋다. "naive W4의 심한 저하는 피할 수 있다"와 "표현 보존 관점의 첫 연구"를 주장한다 | C1, C2(경쟁 처방), 공백 2·4(일부) | **높음** | 동시 연구로 반드시 인용한다. 차이: (i) 우리는 3–5bit **활성**의 반올림 규칙, 그들은 **가중치** W4의 보정 목표 (ii) 우리 처방은 학습·보정이 없다 (iii) 계획기 착취 기전. 동기 문장에 "round-to-nearest, 보정 없음"을 붙인다. H3b(최선 PTQ 위의 sh-iter: TD-MPC2 10/19 성립, DINO-WM 1/3 실패)를 보완 관계의 근거로 정직하게 쓴다. 본문 미확인: "decision fidelity"가 결정 flip을 재는지 읽어야 한다 |
| 2 | [OR cVCKr4bBU2](https://openreview.net/forum?id=cVCKr4bBU2) [ICLR27] | When Can a World Model Decide? Certifying Elite Decisions in Sampling-Based Planning | Anonymous | 투고 2026-09-19, 공개 10-03 | rollout 오차 bound로 elite 판정을 인증한다. 후보가 elite 경계에서 오차의 2배보다 멀면 판정이 맞다. LeWM Push-T에서 CEM 탐색이 진행될수록 인증 비율이 0.67→0.29로 떨어진다("후보 차이가 모델이 구분할 수 있는 수준 아래로 수렴"). 오차가 서로 독립인 validator 모델로 경계 근처만 다시 채점해 CEM·iCEM·MPPI 성공을 9–14점 올린다 | 공백 4(near-tie 밀도), C3(반복 계획기 서사), C2(독립 오차가 처방) | **높음** | 공백 4 문구를 좁힌다(3절). C3는 "일반 모델 오차의 해상도 한계"(그들)와 "반복 사이에 지속되는 반올림 오차"(우리)로 구분한다. 그들은 독립 오차를 추가 모델로 얻고, 우리는 오차를 반복마다 새로 뽑아 추가 모델 없이(≤2%) 얻는다. Stage 2에서 그들의 인증 비율을 det와 sh-iter로 재면 좋은 연결고리가 된다 |
| 3 | [OR srXR8yWw5A](https://openreview.net/forum?id=srXR8yWw5A) [ICLR27, 철회] | Correcting the Optimizer's Curse in World-Model Planning | Anonymous | 투고 2026-09-17 (철회 투고) | 후보별 예측 오차의 평균이 0이어도 최대 예측 return은 낙관적으로 치우치고, 그 편향은 후보 수와 함께 커진다. 보류 문맥으로 보정한 하한(LCB)이 양수일 때만 기준 행동을 바꾼다. visual world model을 포함한 실험 | 공백 2 | 중간 | 공백 2를 "결정적 **압축(반올림)** 오차"와 "반올림 규칙에 따른 선택 후보의 비용 오차 편향"으로 한정한다. 철회 투고라 인용은 선택이다. Smith & Winkler(2006)와 묶어 "일반 모델 오차의 선택 편향은 알려져 있다"로 처리한다 |
| 4 | [OR eTu4mzKor3](https://openreview.net/forum?id=eTu4mzKor3) [ICLR27] | Resolution Collapse in Latent-Space CEM Planning | Anonymous | 투고 2026-09-19, 공개 10-03 | JEPA world model 4환경에서, 예측·실제 후보 품질의 상관이 CEM이 최적화하는 마지막 frame에서 정확히 우연 수준으로 무너진다. horizon을 늘리면 붕괴 지점도 같이 움직인다. 해상 가능한 frame으로 순위를 매기면 3–15점 오르고, 같은 분산의 잡음 대조를 둔다 | C3, 공백 2(일반형) | 중간 | "계획기가 자기 최적화 목표에서 모델 오차에 걸린다"는 일반 서사는 이 논문이 먼저 보였다. 우리는 같은 오차 분포에서 공유·지속만 바꾸는 통제와 반올림 오차 처방으로 차별화한다. 프로젝트의 예비 축 E(NOVELTY_R2 4.2)와도 겹친다 |
| 5 | [OR 1vmM17t1R8](https://openreview.net/forum?id=1vmM17t1R8) [ICLR27] | Not All Evaluations Are Equal: Where Test-Time Compute Matters in World-Model Planning | Anonymous | 투고 2026-09-18, 공개 10-03 | CEM의 결정은 초기 round에서 정해진다. 초기 round를 실행 중 적합한 선형 연산자(결정적 근사)로 바꾸면 성공이 떨어지지만, 같은 크기의 score 잡음에는 그런 단계 의존이 없다. 후반 round는 근사해도 된다(계획 12% 단축) | C3, C4 | 중간 | "결정적 근사 오차는 해롭고 같은 크기의 잡음은 그렇지 않다"는 관찰이 C3와 같은 방향이다. 인용하고, 우리는 반올림 오차의 지속성을 직접 조작한 통제라는 점을 앞세운다. "dither가 초기 반복에서만 필요한가"는 리뷰어 질문이 될 수 있으니 Stage 2 질문으로 적어 둔다 |
| 6 | [OR 5alhlhb7zd](https://openreview.net/forum?id=5alhlhb7zd) [ICLR27] | What Rollout Error Magnitude Misses in Latent World-Model Planning | Anonymous | 투고 2026-09-19, 공개 10-03 | 오차 크기보다 goal 변위 방향으로의 부호 있는 투영이 결정에 중요하다. PushT 1차 계획 성공 69/96→80/96 | C1(한 줄 주장 문구) | 중간 | 한 줄 주장의 "error structure, not size" 문구가 이 논문, Min et al., DRPE와 겹친다. 우리 "구조"는 후보 간 공유와 반복 간 지속이라고 명시한다(4절) |
| 7 | [2609.39185](https://arxiv.org/abs/2609.39185) [누락] | Low-Discrepancy Dither for Quantized Recurrent State Caches | Snigdha Chandan Khilar | 2026-09-30 | Mamba·hybrid LM의 저정밀 재귀 state. 운영 시스템은 SR을 쓴다. 난수가 필요 없는 결정적 golden-ratio Weyl dither가 SR보다 FP에 더 가깝다. RTN은 작은 갱신을 버려 긴 생성에서 오차가 계속 커진다. 누적 오차 크기를 맞추면 지속 교란과 매번 새 잡음의 피해는 비슷하다(감도비 0.55–1.09) | 공백 1(일반 "추론 dither"), 공백 3(지속성 대조의 LLM 판), C3(det+shift) | 중간 | 인용 필수. 계획기와 후보 축이 없다. 그들의 대조에서는 같은 오차 크기에서 지속이 더 해롭지 않았다. 우리는 1-step 오차가 크지 않은데도 계획기가 지속을 착취한다. 이 대비가 오히려 C3를 강화한다. Weyl dither는 우리 det+shift(호출마다 격자 이동)와 같은 계열로 소개한다 |
| 8 | [2604.12374](https://arxiv.org/abs/2604.12374) [누락] | Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning | NVIDIA (저자 547명) | 2026-04-14 | 추론 시 Mamba SSM cache를 FP16으로 내리면 RTNE 편향이 재귀에서 일관되게 쌓여 답이 장황해졌다. cache를 내리기 전에 SR을 적용해 해결했다 | 공백 1 문구 | 중간(문구) | "추론 시 SR이나 dither를 처방으로 쓴 연구가 없다"는 일반 문장은 거짓이 된다. 한 문장으로 인용하고, 우리 범위를 world model과 표본 기반 계획으로 한정한다 |
| 9 | [2610.01889](https://arxiv.org/abs/2610.01889) [누락] | Stochastic Rounding in Low-Precision Transformer Inference: A Variable-Precision Emulation Study of a Small GPT-2 | Yohan Chatelain, Pablo de Oliveira Castro | 2026-10-01 (v2 10-08) | 형식은 고정하고 연산 위치별 반올림 규칙만 바꾼다. DistilGPT-2(significand 6bit)에서 MLP는 SR이 유리하다(perplexity 1.15배 대 RN 2.21배). LM head에서는 SR의 분산 때문에 RN이 유리하다 | 공백 1(일반), C1, C2 | 중간 | "같은 형식에서 반올림 규칙만 바꾼 통제"의 LLM 선례로 인용한다. head에서 SR의 분산이 해롭다는 결과는, 독립 dither가 해로울 수 있고 후보 공유가 결정 단계의 분산을 없앤다는 우리 결과와 연결된다 |
| 10 | [2608.06564](https://arxiv.org/abs/2608.06564) [누락] | Which Decisions Low-Bit Quantization Breaks, and How to Predict Them | Zekun Wu, Swati Dhiman, Adriano Koshiyama | 2026-08-06 (v5 10-05) | LLM 16종을 2–4bit로 양자화하면 결정 margin이 m'≈cm+b로 줄어든다. 이 관계로 flip 비율을 예측한다(중앙 오차 1.0%p). 3bit에서 가중치 SR(한 번 뽑아 고정)은 RTN보다 훨씬 나빴다(Qwen3-4B 거부 87–100% 상실 대 13%) | 공백 4(LLM 판 margin·flip), C1, C3 | 중간 | 공백 4는 "로봇 world model"로 한정해야 한다. 고정된 SR 추첨이 RTN보다 나쁘다는 결과는 우리 sh-ep(고정 dither) 양성 대조와 같은 방향이라 C3 근거로 인용할 수 있다 |
| 11 | [2609.32922](https://arxiv.org/abs/2609.32922) [누락] | Precision As You Need: Stochastic Computing Is a Dense Adaptive Quantizer | Haoran Jin, Kangqi Zhang, Jirong Yang et al. | 2026-09-26 (NeurIPS 2026) | 확률 계산(SC) 행렬곱을 GPU에서 모사한다. DINO-WM(wall_single)을 CEM-MPC로 계획하면 8·7bit 수준에서 FP와 비슷하다. SC 오차는 seed가 고정된 Sobol stream으로 원소마다 고정된다 | 공백 1(가장 가까운 사례) | 중간 | 공백 1을 닫지 않는다. 오차가 고정되어 있고, 7–8bit이며, det 손상의 처방이 아니다. 다만 "world model 계획에 확률적 산술을 쓴 연구가 없다"고는 쓰지 않는다. "고정된 확률 오차는 결정적 오차와 같다"는 C3 논리로 설명할 수 있다 |
| 12 | [OR AhG6FcJBHt](https://openreview.net/forum?id=AhG6FcJBHt) [ICLR27] | LoopQuant: Loop-Aware Post-Training Quantization for Vision-Language-Action Models | Anonymous | 투고 2026-09-07, 공개 10-03 | VLA W4A8 PTQ. 정적 4bit 가중치 오차가 매 denoising step에 다시 나타나고, 연속 요청을 거쳐 쌓인다. 보정과 비균일 codebook, Triton 융합 kernel로 π0.5를 BF16보다 1.88배 빠르게 돌린다 | C3(반복 루프의 지속 오차), C4 | 중간 | "양자화 오차가 반복 루프에서 지속된다"는 관찰 자체는 새롭지 않다고 쓴다(이 논문, STEPQuant, Nemotron, DyQ-VLA). 우리 기여는 그 지속을 계획기가 후보 선택으로 착취한다는 것과 dither 처방이다 |
| 13 | [OR hdBfu8lIC7](https://openreview.net/forum?id=hdBfu8lIC7) [ICLR27] | WAMPTQ: Beyond Weight Quantization via Task-Utility-Guided Precision Allocation for Efficient World Model Deployment | Anonymous | 투고 2026-09-11, 공개 10-03 | WAM 4계열 15과제. 과제 효용으로 4/8bit 가중치 지도를 고르고 kernel을 융합한다. 최대 품질 하락 2.22%p, 종단 속도 중앙값 2.36배, 최대 메모리 −71.5% | C4 | 중간 | "저비트 world model이 실제로 빠르다"는 일반 주장은 선점되었다. C4는 "TD-MPC2 world-model rollout, CUDA graph 재생, A40, W8A4 정수"로 구체화하고 "first"를 쓰지 않는다 |
| 14 | [2609.39056](https://arxiv.org/abs/2609.39056) [누락] | SteerQuant: Steering Quantization Error with Action-Guided Scaling in World-Action Models | Yunhan Wang, Haodong Wang, Zhiming Liu et al. | 2026-09-30 | WAM을 W4A8·W4A4로 양자화한다. 스트림별 오차가 행동에 주는 영향 지도로 scaling을 정하고, 4bit 엔진 Rudder를 만든다. LIBERO 평균 성공이 FP와 0.8%p 이내, BF16 대비 denoising 최대 2.23배(CUDA Graph로 측정), 실제 양팔 로봇 종단 1.35배 | C4, C1(같은 크기의 오차도 행동 영향은 다르다) | 중간 | C4의 관련 연구로 인용한다. 정수 kernel과 CUDA graph 측정은 이미 표준이다. 계획기가 없는 생성형 정책이라 C1–C3과 직접 겹치지 않는다 |
| 15 | [2610.03959](https://arxiv.org/abs/2610.03959) [누락] | LatentQuant: Preserving the Policy-Facing Latent Contract under NVFP4 VAE Quantization | Ziye Deng, Lufang Chen, Shuyu Feng et al. | 2026-10-02 | WAM의 video VAE를 NVFP4 W4A4로 바꾼다. 직접 NVFP4(PTQ)는 LIBERO 94.38%(FP32 95.50%)를 지킨다. 공동 QAT는 재구성 점수가 FP와 거의 같은데 성공이 10.5%로 무너진다("지속 latent drift"). B300에서 VAE 1.17–1.26배 | C5, C1 | 중간 | C5의 보조 근거다. 블록 스케일 NVFP4 PTQ가 로봇 정책을 크게 해치지 않았다. "재구성 지표가 제어를 예측하지 못한다"는 C1 문구 군집에 넣는다 |
| 16 | [2610.15675](https://arxiv.org/abs/2610.15675) [누락] | Recovering Low-Bit Failures in Vision-Language-Action Policies via Quantization-Conditioned Reinforcement Learning | Seongmin Park, Jaewon Lee, Yujun Heo et al. | 2026-10-11 | VLA 5종, 실제 시도 4,050회. QAT도 장기 과제와 분포 이동에서 FP와 격차가 남는다. π0.5 NVFP4 W4A4는 시각 이동 조건에서 QAT 87.0%→QCRL 100.7%(FP 대비). Jetson Thor에서 NVFP4/FP8 1.73배 | C5(범위), C4 | 중간 | C5를 TD-MPC2 5M 결과로 한정하는 근거다. "블록 스케일 FP4면 로봇 모델은 안전하다"로 일반화하지 않는다 |
| 17 | [2602.11882](https://arxiv.org/abs/2602.11882) [누락, NOVELTY_R1에 있음] | Where Bits Matter in World Model Planning: A Paired Mixed-Bit Study for Efficient Spatial Reasoning | Suraj Ranganath, Anish Patnaik, Vaishak Menon | 2026-02-12 | DINO-WM Wall, 가중치만 대칭 per-output-channel INT. 8·6bit는 FP16과 비슷하고, 3bit는 무너지고, 4bit는 bit 배분에 민감하다(encoder 정밀도 보존이 유리) | C1(DINO-WM 저비트 붕괴) | 중간 | 기존 8편 목록에 없었다. Fu et al.과 함께 "DINO-WM 저비트 계획 붕괴"의 선행으로 인용한다. 반올림 규칙, 활성 양자화, 기전은 다루지 않는다 |
| 18 | [2605.04732](https://arxiv.org/abs/2605.04732) [누락] | Using Common Random Numbers for Simulation-based Planning with Rollouts | Sandarbh Yadav, Frederic J Maliakkal, Harshad Khadilkar et al. | 2026-05-06 | rollout 계획에서 CRN이 상대 효용의 분산을 줄이는 간단한 처방을 증명한다(연금 지급 과제, Ludo의 UCT) | C2(후보 간 공유) | 낮음 | C2의 "후보 간 공유 = CRN"을 소개할 때 고전(Kleinman et al. 1999 등)과 함께 최근 계획 사례로 인용한다. 학습된 모델의 오차, 양자화, 반복마다 새 추첨은 다루지 않는다 |
| 19 | [OR aTtR1G9Xnv](https://openreview.net/forum?id=aTtR1G9Xnv) [ICLR27] | CoSR: Memory-Efficient FP8 Training for Multitask Reinforcement Learning | Anonymous | 투고 2026-09-14, 공개 10-03 | FP8 상주 actor–critic 학습. 결정적 writeback은 작은 갱신을 지우고, 독립 SR은 오차가 쌓인다. 그래서 연산자를 고려해 반올림 오차의 상관을 맞추는 결합 SR을 제안한다. DMC·Meta-World, 학습 메모리 −74% | C2(용어·벤치마크 겹침) | 낮음 | 같은 벤치마크에서 SR을 쓰지만 **학습**용이다. "추론 시 계획을 지키는 dither"와 구분하는 한 문장을 둔다 |
| 20 | [OR 7YiYBzr2MV](https://openreview.net/forum?id=7YiYBzr2MV) [ICLR27] | VSAD: Valuation-Sensitivity Attacks and Robust Defense for Latent World-Model Planning | Anonymous | 투고 2026-09-18, 공개 10-03 | 계획 과정이 국소 가치 오차를 키운다. return이 거짓으로 높은 궤적은 작은 관측 교란에 민감하다. 작은 latent 교란으로 불량 궤적을 elite에 넣을 수 있다. 방어는 여러 교란 관측에서 다시 평가한 보수적 점수다 | 공백 2(일반형), C2 | 낮음 | 선택 편향의 적대적 판이다. 그들의 방어는 후보마다 여러 번 다시 평가하므로 비싸다. 우리 dither는 층·호출마다 공유 난수 벡터 하나(≤2%)라는 점으로 대비한다 |
| 21 | [2610.12910](https://arxiv.org/abs/2610.12910) [누락, NOVELTY_R2에 있음] | Sampling-based Kinodynamic Planning with World Models | Yitian Gao, Hanlan Yang, Zachary Kingston | 2026-10-09 | DINO-WM에 RRT·KPIECE를 쓴다. 정밀도 표: CEM 성공 fp32 45/50, tf32 44/50, fp16 46/50이고, fp16이 fp32보다 3.4배 빠르다 | C4, 공백 4(집계 수준) | 낮음 | "fp16이면 충분하지 않나?"라는 질문에 대비한다. 우리 기준선은 TF32와 BF16을 포함한다. 집계 성공률만 보고하므로 공백 4를 닫지 않는다 |
| 22 | [2507.01823](https://arxiv.org/abs/2507.01823) [누락] / [2501.05329](https://arxiv.org/abs/2501.05329) [누락] | TD-MPC-Opt: Distilling Model-Based Multi-Task Reinforcement Learning Agents / Knowledge Transfer in Model-Based Reinforcement Learning Agents for Efficient Multi-Task Learning | Dmytro Kuzmenko, Nadiya Shvai (두 편 같음) | 2025-07-02 / 2025-01-09 (AAMAS 2025 확장 초록) | 317M 다중 과제 TD-MPC2를 1M으로 증류하고(MT30), FP16 PTQ로 크기를 절반으로 줄였다 | C6 | 낮음 | 317M 체크포인트 압축의 선행으로 인용한다. "317M을 처음 압축했다"고 쓰지 않는다(4절) |

### 2.2 참고 표 (위협 낮음)

**(a) VLA·WAM 양자화.** 모두 처방이 보정, 채널 보호, 혼합 정밀도, QAT·RL 회복이다. SR이나 dither를 쓴 것은 없다(공백 1의 근거).

| ID | 제목 | 저자 | v1 | 보인 것 | 닿는 곳 |
|---|---|---|---|---|---|
| [2609.33269](https://arxiv.org/abs/2609.33269) [누락] | Q-WAM: 4-Bit Quantization of World Action Models with Action-Subspace Protection | Arash Akbari, Arman Akbari, Jingwu Luo et al. | 2026-09-27 | WAM W4A4. 반올림 오차가 모든 denoising step을 거쳐 행동에 주는 영향(AOG)으로 민감 부분공간을 16bit로 보호. RoboTwin에서 16bit와 1.1%p 이내. CUDA graph로 시간 측정 | C4, 공백 1 |
| [2607.28405](https://arxiv.org/abs/2607.28405) [누락] | QuantWAMs: Calibrating at the Right Granularity for World Action Models | Jiacheng Zhou, Jinfan Lv, Ruixuan Li et al. | 2026-07-30 | WAM W4A4 PTQ, closed-loop rollout 감사. FP16과 0.2–0.7%p 차이, block 1.4–1.6배 | C4, 공백 1 |
| [2610.38661](https://arxiv.org/abs/2610.38661) [신규] | Quantize by Doing: Task-Guided Mixed-Precision Quantization of Vision-Language-Action Models | Jiuyi Xu, Xiao Hu, Meida Chen et al. | 2026-10-28 | VLAQuantBench·PreDE 그룹의 후속. 짧은 closed-loop 성공률로 bit를 배분한다. 반올림은 RTN·무보정이라고 명시. Franka 실험에서 π0.5는 실패 | 공백 1 |
| [2610.33074](https://arxiv.org/abs/2610.33074) [신규] | AGQ: Action-Guided Ultra-Low-Bit Quantization for Vision-Language-Action Models | Xuhong Qiu, Zhuoyu Wu, Qiawei Zheng et al. | 2026-10-21 | 가중치만 초저비트. LIBERO W3 92.7%, W2.5 70.5% | 공백 1 |
| [2610.08444](https://arxiv.org/abs/2610.08444) [누락] | ActTune: Action-Aware Precision and GPU Operating-Point Adaptation for Energy-Efficient Vision-Language-Action Inference | Zou Qingyun, Bin Gao, Wenju Zhao et al. | 2026-10-06 | 호출마다 정밀도와 GPU 동작점을 고른다. BF16 대비 최대 2.02배, 성공당 에너지 −76.8% | C4 |
| [2610.05745](https://arxiv.org/abs/2610.05745) [누락] | Beyond In-Distribution Preservation: Recovering Generalization in Quantized VLAs via Vulnerability-Oriented Tuning | Shen Ruan, Wenchang Gao, Jin Wang et al. | 2026-10-05 | 양자화된 정책의 행동 차이가 소수의 rollout 상태에 몰린다. 취약 상태만 골라 증류 | 공백 4 주변 |
| [2610.02666](https://arxiv.org/abs/2610.02666) [누락] | CHASE-VLA: Post-Training Quantization Framework for Vision-Language-Action Models with Chunk-Aware Scale Estimation | Jin Hyun, Jung Gyu Min, Gyuhyun Jung et al. | 2026-10-02 (ACCV 2026) | 반복 호출되는 action expert를 W4A4로. denoising 단계와 이전 chunk로 scale을 조정. π0.5 97.3% | 공백 1 |
| [2609.24433](https://arxiv.org/abs/2609.24433) [누락] | FoldQuantVLA: Native Low-Bit Quantization of Vision-Language-Action Models via Consistent Folding | Hung T. Ho, Khanh D. Nguyen, Quang D. Nguyen et al. | 2026-09-21 | TensorRT W4A4 실행. Orin 1.20–1.33배, 데스크톱 1.25–1.52배 | C4 |
| [2610.34717](https://arxiv.org/abs/2610.34717) [신규] | VLA-HWBench: Where the Time Goes When VLA Models Run on Edge Hardware | Borui Xu, Hongshi Tan, Wenqi Jiang et al. | 2026-10-26 | VLA 18종의 7단계 지연 분해. 4bit 가중치 kernel 융합이 torch.compile보다 2.6–2.9배 빠르다 | C4 |
| [2603.07904](https://arxiv.org/abs/2603.07904) [누락] | DyQ-VLA: Temporal-Dynamic-Aware Quantization for Embodied Vision-Language-Action Models | Zihao Zheng, Hangyu Cao, Chibang Tao et al. | 2026-03-09 (v3 10-04) | 양자화는 이동 오차의 분산을 키우고, 회전 오차는 중심이 실행 step마다 누적 이동한다(편향 drift). 누적 잔차로 보정 | C3 주변(편향 대 분산) |
| [2604.11572](https://arxiv.org/abs/2604.11572) [누락, NOVELTY_R2에 있음] | DA-PTQ: Drift-Aware Post-Training Quantization for Efficient Vision-Language-Action Models | Siyuan Xu, Tianshi Wang, Fengling Li et al. | 2026-04-13 | 시간 누적 오차가 kinematic drift를 만든다 | C3 주변 |
| [2605.28803](https://arxiv.org/abs/2605.28803) [누락] | HoloQ-VLA: Uniform W4A4 Quantization of Vision-Language-Action Models | Xinyu Wang, Mingze Li, Sicheng Lyu et al. | 2026-05-27 | 회전·Hadamard·step별 scaling으로 action head까지 균일 W4A4 | 공백 1 |
| [2605.24011](https://arxiv.org/abs/2605.24011) [누락] | ActQuant: Sub-4-bit Action-Guided Quantization for Vision-Language-Action Models | Arash Akbari, Arman Akbari, Masih Eskandar et al. | 2026-05-19 | 행동 기여로 bit 배분, 3bit 이하 가중치, C/C++ 저비트 실행기 | 공백 1, C4 |
| [2602.20309](https://arxiv.org/abs/2602.20309) [누락] | QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models | Jingxuan Zhang, Yunta Hsieh, Zhongwei Wan et al. | 2026-02-23 (CVPR 2026) | VLA PTQ, DiT action head 정수화 | 공백 1 |
| [2606.19565](https://arxiv.org/abs/2606.19565) [누락] | Mix-QVLA: Task-Evidence-Aware Mixed-Precision Quantization of Vision-Language-Action Models | Navin Ranjan, Andreas Savakis | 2026-06-17 | 혼합 정밀도, 1.52배 | 공백 1 |
| [2606.28529](https://arxiv.org/abs/2606.28529) [누락] | The Speedup Paradox: Rethinking Inference Speed-Quality Trade-off in Embodied Tasks | Yujin Wang, Junli Chen, Yixuan Li et al. | 2026-06-26 (v3 10-11) | 손실 경량화(양자화 등)가 동적 과제에서는 성공을 오히려 높이기도 한다 | C4 해석 주의 |
| [2609.14146](https://arxiv.org/abs/2609.14146) [누락] | When Faster VLA Deployment Changes Closed-Loop Behavior: Task Success-Latency Analysis of SmolVLA Across PyTorch and ONNX Variants | Rafiqul Islam | 2026-09-12 | "INT8" ONNX 산출물이 실제로는 FP32 그래프였다. 지연 주장에는 산출물 감사가 필요하다 | C4 보고 방식 |

**(b) world model 계획의 계산량·지연** (C4 맥락)

| ID | 제목 | 저자 | v1 | 보인 것 | 닿는 곳 |
|---|---|---|---|---|---|
| [2610.08350](https://arxiv.org/abs/2610.08350) [누락] | How Much Planning Is Enough? Reducing Search and Computation in World-Model Planning | Changbai Li, Sirui Li, Yichen Yang et al. | 2026-10-06 | 예산을 인증해 줄이고, 반복 사이에 관측·goal 표현을 캐시한다. Full 예산의 행동과 일치하지 않아도 성능이 유지된다. ICLR 2027에도 투고(OR 3UkdMRuupk) | C4, 공백 4 주변 |
| [2609.33218](https://arxiv.org/abs/2609.33218) [누락] | Scope-WM: Scoped Computation for Efficient Visual World Models | Chunzheng Li, Zesheng Jia, Hongda Zhang et al. | 2026-09-27 | Push-T 계획 시간 14.3%, 최대 메모리 18.1%로 감소 | C4 |
| [OR 1ymrzbtiNR](https://openreview.net/forum?id=1ymrzbtiNR) [ICLR27] | How Many Samples Should a Planner Draw? The Answer Depends on a Convention Nobody Reports | Anonymous | 투고 2026-09-18 | elite 규칙(비율 대 개수)에 따라 표본 수 효과의 부호가 뒤집힌다. TD-MPC2 14과제에서 반복 수가 계산을 return으로 바꾸는 더 믿을 만한 방법이다 | H2c(후보 수)·반복 수 실험 보고 시 elite 규칙을 명시할 것 |
| [2601.17231](https://arxiv.org/abs/2601.17231) [누락] | Real-Time, Energy-Efficient, Sampling-Based Optimal Control via FPGA Acceleration | Tanmay Desai, Brian Plancher, R. Iris Bahar | 2026-01-23 | FPGA MPPI, embedded GPU보다 3.1배 빠름. 학습 모델이 아니다 | C4 맥락 |
| [2601.12089](https://arxiv.org/abs/2601.12089) [누락] | Domain-specific Hardware Acceleration for Model Predictive Path Integral Control | Erwan Tanguy-Legac, Tommaso Belvedere, Gianluca Corsini et al. | 2026-01-17 | MPPI 전용 가속기(시뮬레이션) | C4 맥락 |

**(c) "오차 크기 ≠ 결정" 군집** (C1 문구가 붐비는 정도를 보여 준다. 양자화는 다루지 않는다)

| ID | 제목 | 저자 | v1 | 보인 것 | 닿는 곳 |
|---|---|---|---|---|---|
| [2610.29435](https://arxiv.org/abs/2610.29435) [신규] | Events, Not Pixels: Where World Models Get the Return Wrong | Haoyue Liu, Xiaoying Tang | 2026-10-21 | 보상·종료가 일어나는 사건 전이만 실제로 바꿔도 return 오차의 70%가 사라진다. MSE가 큰 전이를 바꾸는 것은 무작위보다 못하다 | C1(우리 [사후] 보상 1-step 관찰과 연결 가능) |
| [2610.35636](https://arxiv.org/abs/2610.35636) [신규] | Adapt for Planning, Not just Prediction: Optimistic Test-Time World Model Adaptation | Akshay Mete, Dileep Kalathil, P. R. Kumar | 2026-10-26 (CoRL 2026 WS) | 예측 손실만으로 적응하면 계획이 좋아지지 않는다. 계획 비용 항을 넣어 +13.5점 | C1 |
| [OR HHiFzgMMJm](https://openreview.net/forum?id=HHiFzgMMJm) [ICLR27] | Feasible Actions, Misleading Rollouts: Diagnosing Latent World Model Planning | Anonymous | 투고 2026-09-19 | V-JEPA2-AC/RoboCasa와 LeWM/PushT에서 예측으로 고른 행동의 실제 비용 regret이 19.2%·6.5% | 공백 4(실제 로봇 WM 쪽은 regret만 재고 near-tie·flip은 없음) |
| [OR 5IBvYYspZv](https://openreview.net/forum?id=5IBvYYspZv) [ICLR27] | What Must A World Model Distinguish for Planning? | Anonymous | 투고 2026-09-18 | 필요한 해상도는 질의, 후보 집합, 계획기에 따라 달라진다 | C1 |
| [OR bFpC2Q9vIb](https://openreview.net/forum?id=bFpC2Q9vIb) [ICLR27] | Which Predictions Suffice for a World-Model Planner's Decision? Certifying Reward and Value Evidence by Re-Planning | Anonymous | 투고 2026-09-19 | TD-MPC2 결정에 충분한 보상·가치 증거를 재실행으로 인증 | C1 |
| [OR 7v3WeiyTb0](https://openreview.net/forum?id=7v3WeiyTb0) [ICLR27] | Trust Before Use: When Do World-Model Quality Metrics Predict Control? | Anonymous | 투고 2026-09-19 | 예측 손실과 영상 품질이 제어를 오도한다. DMC·Meta-World | C1 |
| [OR R97tGtRJ4s](https://openreview.net/forum?id=R97tGtRJ4s) [ICLR27] | Candidate-Dependent Margins in Latent World-Model Evaluation | Anonymous | 투고 2026-09-19 | 후보 집합이 모델 간 비교 margin을 바꾼다(Push-T, DINO-WM 재현) | C1 |
| [OR 12X9kLLOs5](https://openreview.net/forum?id=12X9kLLOs5) [ICLR27] | Diagnosing and Repairing World-Model Errors for Planning | Anonymous | 투고 2026-09-19 | 평균 오차가 작아도 상대 return 오차가 후보 순위를 뒤집는다 | C1 |
| [OR sEge4TbIx4](https://openreview.net/forum?id=sEge4TbIx4) [ICLR27] | Calibrated Candidate Replacement under Imperfect World Models via Baseline-Relative Learning | Anonymous | 투고 2026-08-22 | TD-MPC2·DreamerV3, DMC·Meta-World·ManiSkill2에서 후보 교체를 보정된 하한으로 결정 | 공백 2 주변 |

**(d) 선택 편향·모델 착취**

| ID | 제목 | 저자 | v1 | 보인 것 | 닿는 곳 |
|---|---|---|---|---|---|
| [2610.37196](https://arxiv.org/abs/2610.37196) [신규] | How Much Can Best-of-N Help a World Action Model? Measuring Selectable Headroom and Realizing It in Closed Loop | Haojia Gao, Yue Cao, Yumin Liu et al. | 2026-10-27 | CRN과 보류 oracle로 optimizer's curse를 걷어 내니, 표준 추정이 선택 가능한 이득을 1.27–1.5배 이상 과대평가했다 | 공백 2 주변(결과 무작위성의 선택 편향), C2(CRN) |
| [2607.14180](https://arxiv.org/abs/2607.14180) [누락] | RENEW: Towards Learning World Models and Repairing Model Exploitation from Preferences | Logan Mondal Bhamidipaty, Mykel Kochenderfer, Subramanian Ramamoorthy | 2026-07-15 | 오프라인 MBRL에서 정책의 모델 착취를 인간 선호로 수리 | C3 배경 |

**(e) FP4 형식** (C5 보조)

| ID | 제목 | 저자 | v1 | 보인 것 | 닿는 곳 |
|---|---|---|---|---|---|
| [2605.20402](https://arxiv.org/abs/2605.20402) [누락] | Decomposing MXFP4 quantization error for LLM reinforcement learning: reducible bias, recoverable deadzone, and an irreducible floor | Xiaocan Li, Shiliang Wu, Zheng Shen | 2026-05-19 | MXFP4 오차 = 2의 거듭제곱 scale 편향 + deadzone + 격자 잡음 | C5(MXFP4 손상의 설명) |
| [2603.08747](https://arxiv.org/abs/2603.08747) [누락] | Diagnosing FP4 inference: a layer-wise and block-wise sensitivity analysis of NVFP4 and MXFP4 | Musa Cim, Burak Topcu, Mahmut Taylan Kandemir | 2026-03-05 | MXFP4가 특히 초기 block에서 민감하다(Qwen2.5) | C5 |
| [OR oGWaGHIPWm](https://openreview.net/forum?id=oGWaGHIPWm) [ICLR27] | ThunderWorld: Efficient World Model Generation with Selective FP4 Refinement via Native GEMMs | Anonymous | 투고 2026-09-15 | Cosmos 계열 world model 생성을 W4A4 FP4로, 잔차 FP4 곱으로 보정. Jetson AGX Thor | C5 주변(생성형, 계획 아님) |

**(f) 반올림 이론·재귀 state·기타**

| ID | 제목 | 저자 | v1 | 보인 것 | 닿는 곳 |
|---|---|---|---|---|---|
| [2609.38169](https://arxiv.org/abs/2609.38169) [누락] | STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization | Bingchen Yao, Haobo Xu, Haokun Lin et al. | 2026-09-29 | 긴 기억 state의 양자화 오차가 여러 decoding step에 지속된다. 처방은 정밀도 배분 | C3("지속" 관찰은 새롭지 않음) |
| [2609.05641](https://arxiv.org/abs/2609.05641) [누락] | A Nuclear-Norm Lower Bound for Dithered Scalar Quantization of Matrix Products | Piyush Sao, Narasinga Miniskar, Pedro Valero-Lara et al. | 2026-09-04 | subtractive dither 아래 행렬곱 오차의 하한 | C2 이론 배경 |
| [2609.31641](https://arxiv.org/abs/2609.31641) [누락] | Product-Aware Deterministic Rounding for Quantized Matrix Multiplication | Piyush Sao, Narasinga Miniskar, Pedro Valero-Lara et al. | 2026-09-11 | 곱을 고려한 결정적 반올림, dither 정규화 오차 0.010 대 RTN 0.899 | C2 이론 배경 |
| [OR FMo2cupFU2](https://openreview.net/forum?id=FMo2cupFU2) [ICLR27] | Freeze or Dither: Rounding-Aware Precision Floors for Low-Precision Training | Anonymous | 투고 2026-09-14 | RN은 격자 간격보다 작은 갱신을 얼리고, 잡음이 있으면 dither처럼 풀린다(학습 이론) | 용어("dither") 배경 |
| [OR TI7DwVcwBN](https://openreview.net/forum?id=TI7DwVcwBN) [ICLR27] | Nonlinear Error Cancellation at Fixed ReLU Rounding Endpoints | Anonymous | 투고 2026-09-19 | 원소별 SR 주변분포를 지키면서 뉴런 안의 상관으로 오차를 상쇄할 수 있다(이론) | C2(상관된 반올림) 이론 배경 |
| [OR ixriOFrLES](https://openreview.net/forum?id=ixriOFrLES) [ICLR27] | Same Noise, Better Gradients: Policy Optimization with Common Random Numbers | Anonymous | 투고 2026-09-12 | 정책경사에서 전이 잡음을 공유하는 CRN control variate | C2 배경 |
| [OR Za7lxmQR73](https://openreview.net/forum?id=Za7lxmQR73) [ICLR27] | Adaptive Precision Allocation Can Starve Itself: Consequence-Aware Allocation for Bounded World Models | Anonymous | 투고 2026-09-17 | DreamerV3 표현 채널의 bit 예산(4 bits/coordinate) 배분. 산술 양자화가 아니라 표현 rate다 | 낮음(용어 혼동 주의) |
| [2609.26425](https://arxiv.org/abs/2609.26425) [누락] | QuantWM: Temporally Consistent 2-Bit KV Cache Quantization for Video World Models | Jiaqi Zhao, Xiaobin Hu, Bo Yin et al. | 2026-09-22 | 영상 world model KV cache 2bit. Key 재구성 오차가 작은데 출력 손상은 더 크다 | C1 주변(생성형, 계획 아님) |

## 3. 공백 판정

**공백 1. world model(또는 VLA) 저비트 추론의 처방으로 dither나 SR을 쓴 연구가 없다 → 열림 (범위를 world model·VLA로 한정할 때)**
- world model·WAM·VLA 양자화 논문 약 25편(2.1, 2.2a)의 처방은 보정, 채널 보호, 혼합 정밀도, QAT, RL 회복이다.
- 본문을 직접 검색한 6편(SteerQuant, Q-WAM, QuantWAMs, LatentQuant, QCRL, QbD-VLA)에는 "stochastic rounding"도 "dither"도 나오지 않는다. QbD-VLA는 "round-to-nearest without calibration"이라고 명시한다.
- 가장 가까운 사례는 2609.32922다. DINO-WM 계획에 확률 계산을 쓰지만 오차가 seed로 고정되고, 7–8bit이며, det 손상의 처방이 아니다.
- 주의: world model 밖에서는 추론 시 SR·dither가 이미 쓰인다.
  - Nemotron 3 Super(2604.12374)는 운영 모델에서 Mamba cache에 SR을 쓴다.
  - 2609.39185는 재귀 state cache에 결정적 Weyl dither를 제안한다.
  - 2610.01889는 GPT-2 추론에서 위치별 SR을 분석한다.
  - 따라서 "추론 시 SR·dither를 처음 썼다"는 문장은 쓸 수 없다.

**공백 2. 계획기가 결정적 압축 오차를 착취한다는 직접 측정(선택된 후보의 비용 오차 편향)이 없다 → 열림 (압축 오차형). 일반 모델 오차형은 일부 선점됨**
- 결정적 반올림 오차에서 선택 후보의 비용 오차 편향을 재고 det와 dither를 비교한 연구는 찾지 못했다.
- 일반 모델 오차의 선택 편향은 측정되었다.
  - Correcting the Optimizer's Curse in World-Model Planning(OR srXR8yWw5A, 철회 투고): 후보 수와 함께 커지는 낙관 편향
  - VSAD(OR 7YiYBzr2MV): return이 거짓으로 높은 궤적이 elite에 들어감
  - Resolution Collapse(OR eTu4mzKor3): CEM이 최적화하는 frame에서 순위 상관이 우연 수준으로 무너짐
  - Best-of-N WAM(2610.37196): 결과 무작위성의 optimizer's curse를 CRN으로 정량화
- 그래서 공백 문구는 "반올림(압축) 오차"와 "det 대 dither"로 좁혀야 한다.

**공백 3. 원소별 오차 분포를 고정하고 (a) 후보 간 공유와 (b) 반복 간 지속만 바꾸는 통제가 없다 → 열림**
- 계획에서 두 축을 함께 조작한 통제는 찾지 못했다.
- 가까운 것
  - 2609.39185: LLM 재귀 state에서 매번 새 잡음과 고정 방향의 지속 교란을 비교했다. 지속 축만 다루고, 누적 오차를 맞추면 피해가 비슷했다. 우리 결과와 대비되는 좋은 인용이다.
  - OR 1vmM17t1R8: 초기 CEM round의 결정적 근사와 같은 크기의 score 잡음을 비교했다. 오차 분포를 고정한 통제는 아니다.
  - OR cVCKr4bBU2(EliteCert): 독립 오차를 가진 validator 모델을 쓴다. 공유 축과 비슷하지만 통제 실험은 아니다.
  - 2610.01889: 형식을 고정하고 반올림 규칙만 바꿨다. 공유·지속 축은 없다.

**공백 4. 실제 로봇 world model에서 결정 near-tie 밀도와 정밀도 flip을 잰 보고가 없다 → 부분적으로 닫힘**
- near-tie 밀도
  - EliteCert(OR cVCKr4bBU2)가 LeWM Push-T에서 CEM 반복에 따른 인증 비율 0.67→0.29를 쟀다. 후보가 elite 경계에서 모델 오차 2배 안으로 몰리는 비율, 즉 모델 오차 단위의 near-tie 밀도다.
- 정밀도 flip
  - Module-MSE 투고(OR 3VSa1Kx7HD)가 LeWM·DINO-WM·PLDM W4에서 "planning-decision fidelity"를 보고한다. 초록 수준이라 정의는 확인하지 못했다.
  - LLM에서는 2608.06564가 결정 margin으로 flip 비율을 예측한다.
  - 2610.12910은 DINO-WM의 fp32·tf32·fp16 집계 성공률만 보고한다.
- 남은 것: 실제 로봇 world model(V-JEPA 2-AC 등)에서 정밀도·반올림 규칙에 따른 결정 flip 비율, 그리고 정밀도 단위의 near-tie 밀도는 찾지 못했다. Feasible Actions(OR HHiFzgMMJm)는 V-JEPA2-AC에서 regret만 쟀다.

## 4. 새 논문 때문에 위험해진 문구

| # | 위험한 문구·암시 | 근거 | 대안(영문 초안) |
|---|---|---|---|
| 1 | 추론 시 SR·dither를 처음 썼다는 암시, 또는 "No prior work uses stochastic rounding/dither at inference" | 2604.12374(Nemotron 3 Super 운영), 2609.39185, 2610.01889 | "To our knowledge, dither has not been used to protect sampling-based planning with low-bit world models. In LLM serving, stochastic rounding is applied to recurrent state caches to stop error accumulation [NVIDIA 2026; Khilar 2026]; here it serves a different purpose: denying the planner a persistent error to exploit." |
| 2 | 한 줄 주장 "the structure of rounding error, not its size, decides…"가 일반 원리처럼 읽힘 | Min et al., DRPE, You et al., OR 5alhlhb7zd, 2610.29435, OR 7v3WeiyTb0 | "At the same bit width, whether rounding error is shared across candidates and redrawn across planner iterations, not its size, decides whether a low-bit world model plans well." 그리고 "error size ≠ decision quality"는 알려진 사실로 인용한다 |
| 3 | 동기 문장 "at aggressive bit widths planning collapses" | OR 3VSa1Kx7HD(보정된 W4가 FP 수준), 2602.11882(bit 배분이 결과를 바꿈) | "With round-to-nearest activation quantization and no calibration, planning collapses at 3–4 bits …" H3b(PTQ 위의 sh-iter)를 함께 적는다 |
| 4 | C3를 "반복하는 계획기는 모델 오차를 착취한다"는 일반 발견으로 제시 | OR cVCKr4bBU2, OR eTu4mzKor3, OR 1vmM17t1R8, Obst & Stolzenburg | "Iterating planners are known to drive candidates to where the model cannot resolve them [EliteCert; Resolution Collapse]. We isolate which property of rounding error they exploit: its persistence across iterations." |
| 5 | 공백 2를 "planner가 모델 오차를 착취한다는 측정이 없다"로 일반화 | OR srXR8yWw5A(철회), OR 7YiYBzr2MV, 2610.37196 | "No prior work measures the selection bias that deterministic rounding induces in the chosen candidate's cost, or shows that redrawn dither removes it." |
| 6 | 공백 4 "near-tie 밀도를 잰 보고가 없다" | OR cVCKr4bBU2(LeWM에서 반복별 인증 비율) | "Precision-induced decision flips have not been measured in real-robot world models." Module-MSE 본문을 읽고 그들이 flip을 쟀는지 확인한 뒤에 쓴다 |
| 7 | C4를 "저비트 world model이 실제로 빠르다"는 새 사실처럼 제시하거나 "first"를 붙임 | OR hdBfu8lIC7(2.36배), 2609.39056(2.23배, CUDA Graph), OR AhG6FcJBHt(1.88배), 2610.34717(2.6–2.9배), 2610.15675(1.73배), 2609.24433 | 범위를 구체적으로 쓴다: "planning rollouts of TD-MPC2 world models (5M–317M), replayed as CUDA graphs on an A40". "first"를 쓰지 않는다. fp16 3.4배(2610.12910)에 대비해 BF16 기준선이 같은 속도 계열임을 밝힌다(E8c의 미리 변환한 BF16 기준선이 이 질문에 답한다) |
| 8 | C5를 "블록 스케일 FP4는 손상을 피한다"로 일반화 | 2610.15675(π0.5 NVFP4 W4A4에 분포 이동 격차), 2610.03959(QAT 붕괴) | "In TD-MPC2 (5M), NVFP4 removed most of the damage; MXFP4 did not (4/12 tasks)." |
| 9 | C6에서 "317M TD-MPC2를 처음 압축했다"는 암시 | 2507.01823, 2501.05329(증류 + FP16 PTQ) | "To our knowledge, the first study of 3–5-bit activation quantization on the 317M multi-task checkpoint." |
| 10 | "SR(dither)이 round-to-nearest보다 낫다"는 일반 문장 | 2608.06564(3bit 가중치 SR이 RTN보다 훨씬 나쁨), 2610.01889(LM head에서는 RN이 나음), 2609.39185(결정적 Weyl dither가 SR보다 나음) | "shared across candidates and redrawn every iteration"이라는 한정어를 항상 붙인다. 독립 dither가 더 나쁠 수 있다는 우리 결과(walker)도 함께 적는다 |
| 11 | "양자화 오차가 반복 사이에 지속된다"는 관찰 자체를 기여로 제시 | OR AhG6FcJBHt, 2609.38169, 2604.12374, 2603.07904 | 지속 자체는 알려진 사실로 두고, 기여를 "계획기가 후보 선택으로 그 지속을 착취한다는 것과 그 통제·처방"으로 둔다 |

## 5. 다음 할 일

1. OpenReview 본문을 사람이 직접 읽는다. 자동 수집은 사이트의 확인 화면 때문에 막혔다. 우선순위는 다음 순서다.
   - OR 3VSa1Kx7HD: "planning-decision fidelity"의 정의, 활성 양자화 여부, 반올림 규칙
   - OR cVCKr4bBU2: 인증 비율을 반복별로 잰 방법, 독립 오차 validator의 설계
   - OR 1vmM17t1R8, OR eTu4mzKor3: 결정적 근사 대 잡음 대조의 설계
2. 관련 연구 단락을 네 묶음으로 쓴다.
   - world model 양자화: Fu, Ranganath, PreDE, Module-MSE 투고, WAM 계열
   - 오차 크기 ≠ 결정: Min, DRPE, You, EliteCert, Resolution Collapse, Rollout Error Magnitude
   - 반올림 규칙과 dither: Gupta 2015, Nemotron 3 Super, Khilar 2026, Chatelain 2026, Wu 2026
   - CRN과 optimizer's curse: Smith & Winkler 2006, Kleinman et al. 1999, Yadav et al. 2026
3. 제출 직전(12월 초)에 다시 확인한다. ICLR 2027 토론 기간에 위 투고들의 수정본이나 arXiv판이 올라올 수 있다.

## 부록: 검색 방법

- arXiv API(`export.arxiv.org/api/query`): 질의 약 40개, 2025-01-01 이후, 일부는 `submittedDate` 2026-10-15~11-01로 제한했다. 주요 질의는 다음과 같다.
  - world model × (quantization, low-bit, 4-bit, int4, int8, stochastic rounding, dither)
  - (MPPI, CEM, sampling-based MPC, model predictive control) × quantization
  - (TD-MPC2, DINO-WM, Dreamer, PlaNet, PETS) × (quantization, precision, compression)
  - VLA·world action model × (quantization, low-bit), (NVFP4, MXFP4, FP4) × (robot, policy, world model, VLA)
  - stochastic rounding × (inference, planning, control, RL), dither × (cs.LG, cs.RO, cs.AR)
  - optimizer's curse, winner's curse, selection bias, model exploitation × (planning, world model)
  - near-tie, decision flip, precision flip, nondeterminism × (robot, policy, world model)
  - world model × (latency, CUDA graph, real-time) × (planning, MPC)
- Semantic Scholar 인용 API: Fu, PreDE, You, DRPE, Obst, VLAQuantBench, QVLA, Ranganath, Min의 인용 논문. Min과 PreDE는 인용 0건(색인 지연)이었다.
- OpenReview API(`api2.openreview.net/notes/search`, group ICLR.cc/2027/Conference): 질의 약 30개. 예: quantized world model, world model quantization, stochastic rounding, dither quantization, deterministic rounding, rounding error planning, elite boundary, optimizer's curse, TD-MPC2, MPPI quantization, VLA quantization closed-loop, FP4 robot policy, common random numbers planning. 보조 색인으로 GitHub `hexmSeeU/iclr-2027-world-models`(ICLR 2027 world model 투고 133편, 비망라)도 훑었다.
- WebSearch: "stochastic rounding/dither × quantized world model planning", "low-bit TD-MPC2 MPPI", "NVFP4 world model robot", Min et al. 후속. arXiv·OpenReview에서 찾은 것 외에 새로 나온 것은 없었다.
- 한계
  - Google Scholar는 직접 질의하지 않았다.
  - OpenReview 본문은 읽지 못했다(초록만).
  - Semantic Scholar 색인이 늦어 10월 말 인용은 빠졌을 수 있다.
  - 키워드 검색이라 다른 용어(예: "randomized quantization", "noise injection")만 쓴 논문은 빠졌을 수 있다.
