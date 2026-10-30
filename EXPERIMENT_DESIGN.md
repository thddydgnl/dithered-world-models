# A 논문 실험설계안 — Planners Exploit Deterministic Error: 저비트 월드모델을 위한 Dither

- 작성: 2026-10-21 · 상태: 본 실험 설계 v1(사전등록 초안)
- **2026-10-22 진행 상황**: TD-MPC2 E1에서 dither 처방(H3a)은 Meta-World에서 성립했다. DMC와 기전(H2a)·진단(H2b) 가설은 성립하지 않았다. 사용자 승인으로 다음을 바꿨다. 상세: [main/DESIGN_main.md](main/DESIGN_main.md) §9
  - 기전 주장을 "반복에 걸쳐 지속되는 오차의 이용"으로 좁혔다.
  - "OEI가 1-step 오차보다 손상을 더 잘 예측한다"는 진단 주장을 뺐다.
  - DMC는 7장 대응을 적용한다. 처방 주장은 Meta-World와 DINO-WM에 대해 하고, DMC는 E3의 설명 결과와 함께 한계로 보고한다.
  - 아래 0장의 주장 문장과 기여 1번은 이 범위로 읽는다.
- 근거
  - 파일럿 8개([pilots/](pilots/), 합계 약 $4.6)
  - 노벨티 재조사 [../literature/NOVELTY_R2.md](../literature/NOVELTY_R2.md) §1
  - 아이디어 선정 과정 [../archive/selection_process/FINAL_IDEAS.md](../archive/selection_process/FINAL_IDEAS.md)
- 원칙: 본 실행 전에 이 문서의 가설·지표·판정 규칙을 `DESIGN_main.md`로 고정한다. 그 뒤에 바꾼 것은 사후 분석으로 표시한다.

---

## 0. 한눈에 보기

**문제**
- on-robot 배치를 위해 월드모델을 저비트로 압축하면 표본 기반 계획기(MPPI·CEM)의 성공률이 크게 떨어진다.
- 그런데 압축의 품질을 재는 표준 지표인 1-step 예측 오차로는 이 손상이 보이지 않는다.
  - 같은 비트 수에서 1-step 오차가 가장 작았던 결정적 반올림(det)만 성공률을 잃었다.
  - DINO-WM PushT에서 det의 1-step 오차는 0.0056으로 확률적 변형들(0.0066~0.0070)보다 작았다. 그런데 성공률은 78%에서 52%로 떨어졌다.

**주장(한 문장)**
결정적 저비트 반올림은 계획기가 이용할 수 있는 체계적 오차장을 만든다. 반복 계획기는 그 오차가 유리하게 작용하는 후보를 골라 계획이 무너진다(optimizer's curse). 같은 비트 수에서 반올림을 확률적으로 바꾸는 것(dither)만으로 오차가 무편향 잡음이 되고, 계획 손상 대부분이 사라진다.

**기여(예정)**
1. **현상과 진단**
   - 결정적 압축 오차를 계획기가 이용한다는 것을 직접 잰다. 지표는 *계획기 이용 지수*(OEI)로, 계획기가 고른 후보의 비용 오차가 무작위 후보보다 얼마나 쏠리는지를 본다.
   - 1-step 예측 오차가 이것을 보지 못한다는 것을 보인다.
2. **처방**: 학습 없이 코드 한 줄로 바꾸는 *dithered low-bit world model*. 두 월드모델 계열 × 두 계획기에서 검증한다.
3. **오차 구조 분석**
   - 오차가 후보 간에 공유되는지와 최적화 반복 간에 지속되는지를 분리하는 2축 대조
   - 고정 dither가 무해해지는 조건(dither 표의 크기)
   - SA 대 SAA 관점의 이론
4. **배치 지침**
   - 실제 정수 커널에서 지연–성공률 Pareto
   - 압축 평가 프로토콜: k-step 오차와 함께 OEI와 계획 편차를 보고할 것

**지금까지의 증거(요약, 자세히는 5장)**

| 설정 | 결정적 반올림 | Dither | 사전등록 판정 |
|---|---|---|---|
| TD-MPC2 + MPPI, mw-push, 3-bit 활성 | 성공률 하락 65%p | 5%p(손상 92% 제거) | R2-8b 조건부 있음 |
| TD-MPC2 + MPPI, mw-pick-place, 3-bit | 55%p | 10%p(82% 제거) | 〃 |
| TD-MPC2 + MPPI, DMC cheetah·walker, 4-bit | return 하락 | 10~29% 제거 | 〃 |
| DINO-WM + CEM, PushT, 4-bit | 78% → 52% | 80% | R2-13 조건부 있음 |
| DINO-WM + CEM, Wall, 3-bit | 48% → 8% | 58% | 〃 |

**남은 질문**
- 공유와 새로움이 왜 TD-MPC2에서는 필요하고 DINO-WM에서는 필요 없었나
- DMC 효과는 왜 작나
- 강한 PTQ 기준선 위에서도 이득이 남나
- 실제 정수 커널에서 지연은 어떤가

**계산 자원**
- 전부 대학 GPU 서버(RTX A5000 24GB × 2)에서 돌린다(6.0절). 약 92 GPU-시간이고, RunPod로 환산하면 약 $62다.
- E1(주 벤치마크) 결과로 go/no-go를 판단한다(6.1절).

**투고**
- 1순위: ICML 2027(마감 추정 2027-01-22) 또는 RSS 2027(1월 말 예상)
- 대안: IROS 2027 / RA-L(2027-03-01)
- 병행: arXiv와 ICRA 2027 워크숍
- 날짜는 모두 투고 직전에 공식 공지로 다시 확인한다(8장).

---

## 1. 문제와 동기

### 1.1 월드모델을 로봇에 올리려면 압축이 필요하다
- 학습된 월드모델로 계획하는 MPC는 결정 한 번에 수백~수천 번 월드모델을 호출한다.
  - TD-MPC2 MPPI: 후보 512개 × 반복 6회 × horizon 3
  - DINO-WM CEM: 후보 300개 × 반복 30회 × horizon 5
- 저비트 추론(int8/int4 활성·가중치)은 지연과 전력을 줄이는 가장 직접적인 방법이다.
- 사용자의 DriveWAM 연구에서도 거대 world-action model의 지연이 핵심 병목이었다.
- 그런데 월드모델 양자화는 계획을 망가뜨린다(Fu et al. 2026, arXiv 2602.02110). 그 정도는 비트 수나 예측 오차로 설명되지 않는다(PreDE, arXiv 2609.19441).

### 1.2 계획은 near-tie에서 이루어진다
- 표본 기반 계획기는 비슷한 후보들 가운데서 고른다. 그래서 결정의 대부분이 near-tie다.
  - V-JEPA 2-AC(DROID)에서 1·2위 후보의 energy 차이 중앙값은 후보 표준편차의 0.04배였다.
  - 추론 정밀도(fp16 / fp32 / bf16)만 바꿔도 1위 후보가 10~23% 바뀌었다(R2-5).
  - DINO-WM CEM 후반 반복의 후보 집단은 모델 비용으로 보면 near-tie였다. 크기를 맞춘 일관 교란이 후반 순위만 0.53에서 0.12로 무너뜨렸다(R2-7).
- 이 영역에서는 오차의 *크기*보다 *구조*가 결정을 정한다.
  - 같은 1-step 오차를 가진 네 가지 확률적 반올림의 손상이 mw-push에서 5%p부터 85%p까지 갈렸다(R2-8b).

### 1.3 원문 에세이와의 연결
- *Neural Networks Are Already Clean*(Cammarata 2026)은 약한 활성을 지워도 바뀌는 답은 near-tie뿐이라고 주장한다.
- 우리 파일럿에서 성립한 부분
  - 같은 예측 변화를 내려면 약한 항목을 무작위 항목의 3.3~13.6배 지워야 했다(R2-5).
  - 계획에서도 약한 활성은 특별하지 않았다. 크기를 맞춘 후보 일관 무작위 교란과 손상이 같았다(R2-3b 8/8칸, R2-5 비 0.85~1.06).
- 우리 파일럿에서 깨진 부분: "near-tie는 중요하지 않다"는 전제다. 계획은 near-tie에서 이루어지므로, 결정적이고 체계적인 작은 오차가 결과를 정한다.
- 이 연결은 논문에서 동기 단락 하나로만 쓴다. 본론은 압축과 계획이다.

---

## 2. 선행연구와 차별점

### 2.1 가장 가까운 연구

| 연구 | 보인 것 | 우리와의 차이 |
|---|---|---|
| **Min et al. 2026, *When Low Prediction Error Misleads Planning*** (arXiv 2610.05550, 가장 위협적) | 후보 300개의 끝점 오차를 후보 풀 중심 오차와 행동 상대 응답으로 분해했다. 오차 크기와 결정 효과는 별개라는 결론이다 | 후보 축만 oracle 수리로 다룬다. 최적화 반복 간 지속성, 실제 오차원(압축·정밀도), 처방이 없다 |
| Fu et al. 2026, *An Empirical Study of World Model Quantization* (arXiv 2602.02110) | DINO-WM PTQ에서 공격적 비트일 때 계획 목표와 과제 성공이 어긋난다 | 반올림 방식(결정적·확률적)과 오차 구조를 다루지 않는다. 기전과 처방이 없다 |
| PreDE (arXiv 2609.19441) | world action model 5개 × PTQ 설정 28개에서 오프라인 행동 편차로 closed-loop 저하를 예측한다 | 진단 지표 연구다. 반올림 방식 처방이 없다 |
| You et al. 2026 (arXiv 2607.10362), DRPE (arXiv 2609.32322) | 데이터 평균 예측 오차는 계획 품질을 추적하지 못한다. 결정 관련 오차는 추적한다 | 우리는 그것이 왜 그런지(계획기의 오차 이용)와 어떻게 고치는지(dither)를 보인다 |
| 고전: Smith & Winkler 2006(optimizer's curse), Ng & Jordan 2000(PEGASUS), Strens & Moore 2002, Kleinman et al. 1999(CRN-SA) | 추정치를 최적화해 고르면 고른 대안이 과대평가된다. 고정 scenario에 최적화하면 과적합한다. 비교 안에서는 난수를 공유하라 | 원리는 고전이다. 대상(학습된 월드모델의 압축 오차), 축의 분리, 효과 크기가 새롭다 |
| 수치해석: stochastic rounding(Gupta et al. 2015 등) | 결정적 반올림의 편향은 반복 알고리즘에서 쌓이고, 무편향 SR은 쌓이지 않는다 | 계획기의 선택(argmin·elite)이 만드는 *선택 편향*은 다루지 않는다 |
| 모델 착취: Boney et al. 2019, Ha & Schmidhuber 2018, Obst & Stolzenburg (arXiv 2610.00921) | 계획기는 모델 오차를 이용한다. World Models는 temperature로 꿈에 잡음을 넣어 착취를 막았다 | 학습 오차가 아니라 *압축 오차*의 구조, 그리고 같은 비트 수 안의 처방 |
| VLA 양자화(VLAQuantBench 2609.25376, QVLA 2602.03782) | closed-loop에서의 시간적 오차 누적 | 처방이 비트 배분과 보정에 그친다. 확률적 반올림이 없다 |

### 2.2 남은 공백 (NOVELTY_R2 §1.3)
1. 저비트 월드모델(또는 VLA) 추론의 처방으로 dither나 확률적 반올림을 쓴 연구가 없다.
2. 최적화기가 결정적 압축 오차를 이용한다는 직접 측정이 없다. 고른 후보의 비용 오차 쏠림을 잰 연구가 없다는 뜻이다.
3. 항목별 오차 분포를 고정한 채 (a) 후보 간 공유와 (b) 반복 간 지속만 바꾸는 대조가 없다.
4. 실제 로봇 월드모델에서 결정 near-tie의 밀도와 정밀도 flip을 잰 보고가 없다.

### 2.3 서술 원칙
- 서론 첫 단락에서 고전 원리(CRN, SAA의 optimizer's curse, stochastic rounding)를 먼저 인정한다. 기여는 "학습된 월드모델의 압축 오차와 MPPI·CEM으로 옮겨 오차 구조를 분리하고, 처방을 두 계열에서 실증한 것"으로 정의한다.
- Min et al.과의 관계는 그림 하나로 정리한다. 그들의 후보 축에 우리가 반복 축과 실제 오차원을 더한다.
- **쓰지 않을 주장**
  - "k-step 오차는 틀린 지표다." 조건 전체로 보면 1-step 오차와 손상의 Spearman은 0.70이다(R2-8, 156조건). 대신 "같은 1-step 오차 수준 안에서는 구조가 정한다"고 쓴다.
  - "반복이 고정 오차를 증폭한다." closed-loop로 지지되지 않았다(R2-14). 대신 "반복은 새 잡음만 평균한다"고 쓴다.
  - "후보 간 공유 + 반복마다 새로"가 항상 필요하다. TD-MPC2에서만 성립했다(R2-8b 대 R2-13).
  - "dither는 일반적으로 좋다." 후보 독립 dither가 결정적 반올림보다 나쁜 칸이 있었다(R2-8 walker).

---

## 3. 연구 질문과 가설 (사전등록 대상)

| RQ | 질문 | 가설(사전등록) |
|---|---|---|
| RQ1 현상 | 결정적 저비트 반올림은 1-step 예측 오차가 말해 주는 것보다 계획을 더 망가뜨리는가? | H1: 두 월드모델 계열 × 두 계획기 모두에서, det가 기준 대비 20%p 이상(DMC는 return 20% 이상) 떨어뜨리는 비트 수가 있다. 그 비트 수에서 det의 1-step 오차는 확률적 변형보다 크지 않다 |
| RQ2 기전 | 손상은 계획기가 결정적 오차를 이용하기 때문인가? | H2a: det에서 OEI(4.3절)가 유의하게 음수이고, 확률적 변형에서는 0 근처다. H2b: 조건 전체에서 OEI가 1-step·k-step 오차보다 손상을 더 잘 예측한다(Spearman 차 ≥ 0.15). H2c: \|OEI\|는 후보 수가 많을수록 커진다(optimizer's curse의 용량-반응) |
| RQ3 처방 | 같은 비트 수에서 dither가 손상을 없애는가? 강한 PTQ 위에서도 그런가? | H3a: 자격 칸(det 하락이 실재하는 과제 × 비트 수 × 모델)의 75% 이상에서, 반복마다 새로 뽑는 공유 dither(sh-iter)가 det 손상의 50% 이상을 없앤다. 어떤 칸에서도 det보다 5%p 넘게 나쁘지 않다. H3b: 최선의 PTQ 기준선 위에 dither를 더하면 3~4-bit에서 추가 이득이 있다 |
| RQ4 구조 | 공유·새로움은 언제 필요한가? | H4a: 고정 dither(sh-fixed)의 손상은 dither 표 크기 N_eff가 커질수록 줄어든다(단조, 사전 예측식 4.4절). H4b: 후보 공유(CRN)의 이득은 near-tie 밀도가 높은 과제에서 크다 |
| RQ5 배치 | 실제 정수 커널에서도 이득이 남는가? | H5: dither 추가 지연은 10% 이하다. "dither + 낮은 비트"가 "det + 높은 비트"를 지연–성공률 Pareto에서 지배하는 설정이 하나 이상 있다 |
| RQ6 일반성(선택) | 다른 결정적 오차원(활성 희소화, 수치 정밀도)에도 같은 원리가 적용되나? | H6: 결정적 활성 희소화(R1-1의 약한 활성 제거)를 기대값을 맞춘 확률적 희소화로 바꾸면 계획 손상이 준다 |

---

## 4. 방법

### 4.1 설정과 표기
- 월드모델 f_θ는 latent dynamics와 비용(또는 reward·value)으로 이루어진다. 계획기는 후보 행동열 {a_i}의 비용 Ĵ(a_i)를 비교한다.
  - MPPI: softmax 가중
  - CEM: elite 평균
- 압축 연산자 Q: 층 ℓ의 은닉 활성 x를 k-bit 균일 격자로 보낸다.
  - 범위 [lo_ℓ, hi_ℓ]는 보정 데이터로 정한다.
  - Δ = (hi − lo)/(2^k − 1)이고, 범위 밖은 잘라낸다.
- 반올림 방식(파일럿과 같은 정의)
  - **det(RTN)**: Q(x) = Δ·round(x/Δ)
  - **확률적(SR)**: Q(x) = Δ·⌊x/Δ + u⌋, u ~ U[0, 1). E[Q(x)] = x이고 평균 MSE는 Δ²/6이다(det는 Δ²/12).
  - 확률적 변형은 u를 공유하는 범위만 다르다(4.2절).

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
    - DINO-WM: 비양자화 모델의 비용(확증 지표, 파일럿 값 +0.30 / −0.09의 정의). 시뮬레이터 재생으로 얻은 참 비용 기준은 보조로 보고한다(2026-10-22 정정, 사용자 승인: 참 비용은 모델 비용과 단위가 달라 선형 변환이 필요하고 양자화가 없어도 0이 아니다)
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

### 4.4 이론 (부록으로 쓸 것)
- **모형**
  - 후보 a의 참 비용을 J(a), 모델 비용을 Ĵ(a) = J(a) + e(a)로 둔다.
  - 계획기는 이산 후보 집합에서 argmin(또는 softmax / elite 평균)을 고르고, 분포를 다시 맞춰 반복한다.
- **세 가지 오차 모형에서 유도할 것**
  1. **지속 오차장** e(a) = b(a)(결정적 함수, det에 해당)
     - 반복은 J + b의 최솟값으로 수렴한다(SAA와 같다).
     - 고른 후보의 기대 오차 E[b(â)] < E[b(a)]다. 이것이 optimizer's curse이고, OEI < 0의 예측이다.
     - 후보 수 N이 늘면 |E[b(â)]|가 극값 통계를 따라 커진다(예: b가 가우시안이면 ∝ √(2 log N)). H2c의 예측식이다.
  2. **반복마다 새로운 독립 잡음** e_t(a) ~ i.i.d.(ind에 해당)
     - 한 반복의 선택은 섞인다(gap ≲ 잡음 크기일 때).
     - 하지만 평균 갱신에 들어가는 잡음은 반복에 걸쳐 평균된다(SA형). 남는 비용은 분산이고, 1/(elite 수 × 반복 수)에 비례해 준다.
  3. **공유 잡음**(sh-iter에 해당)
     - 후보 간 차이에서 상관 성분이 상쇄된다(CRN의 분산 감소).
     - 반복마다 새로우므로 지속 편향도 남지 않는다.
  4. **고정 공유 dither**(sh-step / sh-ep)
     - 오차장이 고정되므로 원리상 1과 같다.
     - 하지만 비용 하나가 N_eff개의 독립 offset을 합산하면, 후보 간에 이용 가능한 오차 변동의 크기가 N_eff와 함께 줄어든다.
     - 예측식 후보: Var_cand[b] ∝ 1/N_eff. 이것이 TD-MPC2(표 = horizon × 512 unit)와 DINO-WM(token × 2048 unit, 층당 40만~120만 값)의 차이를 설명하는지 실험 3에서 검정한다.
- **검증**
  - 2차 목적함수 수치 모의(후보 수 · 반복 수 · elite 비율 · 잡음 구조 격자)
  - 측정한 OEI·계획 편차와의 일치

---

## 5. 지금까지의 증거 (파일럿, 모두 사전등록)

| 파일럿 | 설정 | 판정 | 핵심 수치 |
|---|---|---|---|
| [R1-1](pilots/R1_1_neartie_planning/REPORT.md) | TD-MPC2 MPPI, MW 2 + DMC 2, 약한 활성 제거 | 있음(다른 방향) | MW α=0.1에서 활성의 51~54%를 지워도 1-step 오차 0.024~0.029, 성공 10/10. 사후: 1-step 오차를 맞춘 무작위 제거보다 약한 제거가 계획을 더 망가뜨림(20칸 중 12칸). mw-push α=0.2에서 성공률 하락 70%p 대 10%p |
| [R2-3](pilots/R2_3_weak_planning_bias/REPORT.md) | 같은 설정, 0 대입 / 평균 대체 / 편향만 / 무작위 | 혼합 | 평균을 보존해도 0 대입만큼 해로움(합산 1.02). **독립 mask 무작위 대조는 후보 간 상쇄로 손상을 과소평가**한다는 confound 발견 |
| [R2-3b](pilots/R2_3_weak_planning_bias/followup_3b/REPORT_3b.md) | 후보 일관 고정 무작위 제거(D-fix) | **일관성** | D-fix가 약한 제거만큼 해로움(8/8칸, 합산 111%). 독립 무작위보다 6/8칸 더 해로움. 약한 활성 고유 효과 0/8. 최종 계획 편차는 D-fix 0.54~0.64, seed 바닥 0.07~0.18 |
| [R2-5](pilots/R2_5_vjepa_ac_nearties/REPORT.md) | V-JEPA 2-AC, DROID 160창 × 후보 64, 오프라인 순위 | 없음(S1 경계) | 1·2위 gap 중앙값 = 후보 std의 0.040배. 실행 행동의 1위 비율 1.25%. 정밀도만으로 1위 flip(fp16 기준 fp32 10%, bf16 22.8%). 약한 활성 = 일관 무작위(비 0.85~1.06) |
| [R2-7](pilots/R2_7_neartie_regime/REPORT.md) | DINO-WM CEM, 후보 300개 전부 재생 | 없음 | PushT의 후반 순위 하락은 범위 축소로 설명됨(모델 0.533, oracle 0.534). Wall은 구조적 붕괴(0.51 vs oracle 0.81). 일관 교란이 후반 순위만 0.53 → 0.12로 무너뜨림 |
| [R2-8](pilots/R2_8_dithered_world_models/REPORT.md) | TD-MPC2 3~5-bit 활성, det / ind / sh / sub | 없음(S3 반대 방향) | 후보 독립 dither는 det 손상을 29~41%만 줄임(walker는 악화). 공유 dither가 최선(mw-push 성공 30% → 90%). 156조건에서 손상과의 Spearman: 1-step 0.695, 일관 가치 편향 0.741 |
| **[R2-8b](pilots/R2_8b_crn_dither/REPORT.md)** | 공유 범위 5종, 20 ep × 4과제, 새 seed | **조건부 있음** | sh-fwd(= sh-iter)가 det 손상을 mw-push 92%(65 → 5%p), pick-place 82%(55 → 10%p) 제거. cheetah 10%, walker 29%. sh-step은 det만큼 해롭고, sh-ep는 DMC에서 det보다 나쁨(+28.3, +40.1). 확률적 변형 4개의 1-step 오차는 5% 이내로 같은데 하락은 5~85%p로 갈림 |
| **[R2-13](pilots/R2_13_crn_dither_dinowm/REPORT.md)** | DINO-WM + CEM, PushT k=4 · Wall k=3, 50 ep × 5변형 | **조건부 있음** | PushT 성공률: none 78%, det 52%, sh-iter 80%. Wall: 48%, 8%, 58%. McNemar p = 0.001, 1e-5. det의 1-step 오차가 가장 작음(0.0056 대 0.0066~0.0070). 5-step 오차는 det를 7~31배로 가름. OEI: 무작위 후보 +0.30, 고른 후보 −0.09. **sh-fixed와 ind도 똑같이 잘 됨**(공유·새로움 불필요) |
| [R2-14](pilots/R2_14_iteration_dose_response/REPORT.md) | MPPI 반복 1·3·6·12 | 없음(반대 방향) | det 손상은 반복과 무관. sh-fwd 손상은 반복과 함께 감소(cheetah 85 → 27%, mw-push 71 → 0%p). cheetah sh-step − sh-fwd 차는 n=1에서 −0.8, n=12에서 48.4. 실용: sh-fwd n=6(35.5 ms/step, return 571)이 det n=12(61.4 ms/step, 352)보다 빠르고 좋음 |

**확립된 것**
- 같은 비트 수에서 det만 계획을 크게 망가뜨린다. dither가 손상을 없앤다(2계열 × 2계획기).
- 1-step 오차는 이 차이를 보지 못한다.
- DINO-WM에서 계획기가 det 오차를 이용한다는 직접 측정이 있다(OEI).

**미확립(본 실험의 대상)**
- 공유·새로움이 필요한 조건(TD-MPC2와 DINO-WM의 차이)
- DMC의 작은 효과
- 강한 PTQ 위의 이득
- 실제 커널
- 과제·체크포인트 규모(현재 과제 2~4개, 체크포인트 seed 1개)

---

## 6. 본 실험 계획

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

### 6.9 계산 예산과 실행 순서

**처리량 실측(파일럿)**
- TD-MPC2: RTX 3090 + CUDA MPS 작업자 20개로 시간당 약 640 episode(R2-8b)
- DINO-WM: RTX 4090으로 시간당 약 200 episode(5변형 교대 + 재생 포함, R2-13)
- 서버에서는 A5000 두 장을 동시에 쓴다. A5000은 4090보다 느려 DINO-WM 쪽 GPU-시간이 늘 수 있으므로 E0에서 처리량을 다시 잰다.

| 실험 | 규모 | GPU 시간 | RunPod 환산(참고) |
|---|---|---|---|
| E0 재현·보정·k 선택 | R2-8b·R2-13 주요 칸 재현, held 보정 | 6 | $4 |
| E1 주 벤치마크 | TD-MPC2 약 12,000 ep, DINO-WM 약 6,300 ep | 50 | $33 |
| E2 기전 sweep | 과제 4 × 후보 수 5 × elite·temperature, 반복 수 | 10 | $6 |
| E3 구조·표 크기 | 2모델 × 6칸 + 표 크기 4~5단계 | 10 | $7 |
| E4 모의 | CPU | 0 | $0 |
| E5 기준선·QAT | PTQ 5종 + QAT 학습 | 8 | $6 |
| E6 선택 | 희소화 + V-JEPA 2-AC | 5 | $4 |
| E7 커널 | 커널 구현 + 측정 | 3 | $2 |
| **합계** | | **약 92** | **약 $62** |

- 전부 대학 GPU 서버에서 돌리므로 RunPod 비용은 들지 않는다. 오른쪽 열은 참고용 환산이다. 예외는 6.0절의 Meta-World 대체 경로(약 $10~15)뿐이다.

**실행 순서**(앞 단계의 결과가 다음 단계의 입력이다)

| 순서 | 할 일 | 산출물 |
|---|---|---|
| 1 | E0: 서버 환경(uv venv, mujoco-py 점검)과 처리량 측정, 주요 칸 재현, 과제·k 선정, `DESIGN_main.md` 고정 | 환경 점검표, 재현표, 사전등록 문서 |
| 2 | E1 | 주 결과표, **go/no-go** |
| 3 | E2, E4 모의 | 기전 그림, 이론 초안 |
| 4 | E3 | 구조·표 크기 그림 |
| 5 | E5 | 기준선 표 |
| 6 | E6·E7 | 일반성·Pareto |
| 7 | 집필, 그림, 부록 | arXiv, 투고본 |

- 자원이 모자라면 우선순위는 E1 > E2 > E3 > E5 > E7 > E6이다.

### 6.10 재사용할 코드

| 용도 | 위치 |
|---|---|
| TD-MPC2 설치(Meta-World 포함), 양자화 hook, 변형 5종, grid, 분석 | [pilots/R2_8b_crn_dither/code/](pilots/R2_8b_crn_dither/code/): `nt_lib.py`, `grid.py`, `calib.py`, `prep_r28b.py`, `analyze_r28b.py`, `setup_pod.sh`, `setup_mw.sh` |
| 반복 수 sweep, 계획 편차, 지연 측정 | [pilots/R2_14_iteration_dose_response/code/](pilots/R2_14_iteration_dose_response/code/): `plandev_r214.py`, `timing_r214.py` |
| 후보 일관 대조(D-fix), 결정 수준 지표 | [pilots/R2_3_weak_planning_bias/followup_3b/code/](pilots/R2_3_weak_planning_bias/followup_3b/code/): `decision.py` |
| 약한 활성 제거(실험 6a) | [pilots/R1_1_neartie_planning/code/](pilots/R1_1_neartie_planning/code/) |
| DINO-WM 설치, bf16 배치 CEM, 양자화 hook, 재생(참 비용), OEI | [pilots/R2_13_crn_dither_dinowm/code/](pilots/R2_13_crn_dither_dinowm/code/): `qlib13.py`, `cem13.py`, `replay13.py`, `analyze13.py`, `setup13.sh` |
| 순위 상관 oracle, 범위 축소 대조 | [pilots/R2_7_neartie_regime/code/](pilots/R2_7_neartie_regime/code/) |
| V-JEPA 2-AC + DROID(실험 6b) | [pilots/R2_5_vjepa_ac_nearties/code/](pilots/R2_5_vjepa_ac_nearties/code/) |
| 서버 실행 도구(새로 만듦) | ssh 별칭 `kiis-mvf-gpu`로 접속한다. 코드와 결과는 rsync로 옮기고, 작업은 tmux에서 작업 목록 기반으로 돌려 중단 뒤에도 이어 간다. 파일럿 설치 스크립트(`setup_pod.sh`, `setup_mw.sh`, `setup13.sh`)는 RunPod용(root·apt)이라 사용자 공간 uv·micromamba 설치로 고친다 |
| RunPod(대체 경로만) | [../runpod_infra/](../runpod_infra/). Meta-World 대체 경로가 필요할 때만 쓴다. 쓸 때는 `budget.py`의 START_BALANCE·LIMIT·HARD_DEADLINE과 장부, `common.sh`의 pod 접두어를 새로 정한다 |

---

## 7. 리스크와 대응

| 리스크 | 현재 근거 | 대응 |
|---|---|---|
| DMC에서 효과가 작다 | cheetah 10%, walker 29% 제거(R2-8b) | near-tie 밀도와 지속 편향 비중으로 설명(실험 3). 설명이 안 되면 주장을 조작 과제와 DINO-WM 중심으로 좁히고 DMC는 한계로 보고 |
| 공유·새로움 결과가 모델마다 다르다 | TD-MPC2는 필요, DINO-WM은 불필요 | 주 주장에서 빼고 표 크기 법칙(실험 3)으로 통합. 법칙이 맞지 않으면 "모델 의존적"으로 정직하게 보고 |
| Min et al.과 겹친다는 리뷰 | 같은 커뮤니티, 2026년 10월 공개 | 후보 축 + 반복 축 + 실제 오차원 + 처방을 한 그림으로 대비 |
| "고전 원리의 재발견"이라는 리뷰 | CRN, SAA, SR은 고전 | 서론에서 먼저 인정. 이론 부록과 효과 크기(같은 비트 수에서 mw-push 하락 65 → 5%p, Wall 성공률 8 → 58%)로 답함 |
| fake-quant와 실제 커널의 차이 | 파일럿은 모두 fake-quant(지연 20~41% 증가) | 실험 7. 실제 커널에서 이득이 사라지면 한계로 명시 |
| QAT가 문제를 없앤다 | 미측정 | 그래도 PTQ·학습 없는 처방의 가치가 남는다. 결합 결과를 함께 보고 |
| 큰 월드모델로 일반화되지 않음 | V-JEPA 2-AC는 오프라인만 | 실험 6b로 근거를 보완하고 한계에 명시. TD-MPC2 다중 과제 48M·317M 체크포인트로 크기 축을 추가할 수 있음 |
| GPU 간 비결정성 | R2-3b에서 비트 단위로 다름 | 같은 장비 원칙(6.0) |
| 서버에서 mujoco-py가 빌드되지 않음 | 서버에 OpenGL 라이브러리와 sudo가 없음 | E0 첫 작업으로 사용자 공간 Mesa·GLEW 설치를 시도. 안 되면 Meta-World 칸만 RunPod(약 $10~15), PointMaze는 제외 |
| 서버 공유 자원 | CPU·디스크를 다른 사용자와 공유(디스크 86% 사용) | 큰 데이터는 쓰고 지움. 결과는 맥북으로 수시 동기화. 지연 측정(E7)은 반복해 중앙값으로 |
| 서버 사용 중단 | 대여를 연장해 가며 씀 | 작업 목록 기반 재개. 환경 설치를 스크립트 하나로 재현 |

---

## 8. 논문 구성과 투고 전략

**제목 후보**
- *Planners Exploit Deterministic Error: Dithered Low-Bit World Models for Model-Predictive Control*
- *Same Bits, Different Plans: Error Structure Decides Planning with Quantized World Models*
- *Don't Let the Planner See the Grid: Stochastic Rounding for Compressed World Models*

**구성(8쪽 + 부록)**
1. 서론
   - 압축은 계획을 망가뜨리고, 예측 오차는 그것을 보지 못한다.
   - 그림 1: 같은 비트 수에서 det와 dither의 성공률, 그리고 det의 더 작은 1-step 오차
2. 관련 연구: 2.1절 표의 요약
3. 문제 설정과 오차 구조
   - 4.1~4.2절
   - 이론 요약(4.4절)
4. 진단: OEI. 그림 2: 무작위 후보 대 고른 후보의 비용 오차, 후보 수에 따른 용량-반응
5. 처방: dithered low-bit world models. 표 1: 2모델 × 15과제 × 비트 수
6. 언제 공유와 새로움이 필요한가
   - 그림 3: 2축 대조
   - 그림 4: 표 크기 법칙과 이론 곡선
7. 배치
   - 그림 5: 지연–성공률 Pareto
   - 표 2: PTQ 기준선 + dither
8. 한계(DMC 효과, 큰 월드모델, 실로봇)
- 부록: 파일럿 사전등록 기록, 칸별 결과, 이론 유도, 결정 수준 지표

**투고 경로**
- 2026-10-21 확인. 2027년 일정은 아직 공식 공지가 없는 것이 많아, 투고 직전에 다시 확인한다.

| 경로 | 투고처 | 일정 | 조건 |
|---|---|---|---|
| 1순위 | ICML 2027 | 마감 추정 2027-01-22(초록 1월 16일 무렵, 제3자 추정) | E1~E5 완료, go 기준 통과 |
| 1순위 대안 | RSS 2027(아테네, 7/6–11, 제3자 목록 기준) | 마감 1월 말 예상(2026년은 1/30) | 같음. 로봇 쪽 강조 시 |
| 대안 | IROS 2027 / RA-L | 2027-03-01(ECM 계획서에서 확인한 일정) | 결과가 조작 과제 중심으로 좁혀질 때 |
| 대안 | CoRL 2027, NeurIPS 2027 | 5월 말 예상(CoRL 2026은 5/28) | 실험 7이나 실로봇을 보강할 때 |
| 병행 | arXiv(12월 말) + ICRA 2027 워크숍(서울, 5/24–28, 워크숍 마감은 보통 2~4월) | | 대학원 지원서와 인턴 지원에 링크 |

**지원서 서사 연결**
- DriveWAM(거대 world-action model의 지연 문제)에서 출발해 "압축된 월드모델로 계획할 때 무엇이 깨지는가"를 원리와 처방으로 답한 연구로 소개한다.

---

## 9. 참고문헌 (원문 확인 수준은 NOVELTY_R2 표기를 따름)

- Cammarata 2026, *Neural Networks Are Already Clean*, Goodfire. https://already-clean.goodfire.com/
- Min et al. 2026, *When Low Prediction Error Misleads Planning*, arXiv 2610.05550
- Fu et al. 2026, *An Empirical Study of World Model Quantization*, arXiv 2602.02110
- Xu et al. 2026, *Predict Before You Deploy* (PreDE), arXiv 2609.19441
- You et al. 2026, *A Control Theory of Predictability in Latent World Models*, arXiv 2607.10362
- Wang, Fan, Huang 2026, *Not All Errors Matter* (DRPE), arXiv 2609.32322
- Obst & Stolzenburg 2026, *In CEM, a World Model Is Also a Proposal Mechanism*, arXiv 2610.00921
- Smith & Winkler 2006, *The Optimizer's Curse*, Management Science
- Ng & Jordan 2000, PEGASUS, UAI · Strens & Moore 2002, JMLR · Kleinman, Spall, Naiman 1999, Management Science
- Gupta et al. 2015, *Deep Learning with Limited Numerical Precision*, ICML
- Boney et al. 2019, *Regularizing Trajectory Optimization with Denoising Autoencoders*, NeurIPS · Ha & Schmidhuber 2018, *World Models*
- Hansen et al. 2024, TD-MPC2, ICLR · Zhou et al. 2025, DINO-WM · Assran et al. 2025, V-JEPA 2(arXiv 2506.09985)
