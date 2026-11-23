# 인용 관행 검토 (Fable 5.2 max, 2026-11-23)

- 요청(사용자, 11-23): "지금 논문이 너무 정확하게 다른 논문 인용 등을 하려고 하는 건 아닌지, 너무 과한 건 아닌지"
- 검토자: Fable 5.2, effort max. 독립 검토로, 조정자의 판단은 넘기지 않았다. 검토자는 `paper/tmlr/main.tex`·`main.pdf`·`refs.bib`·`REFERENCES_CHECK.md`를 읽었다. 인용된 논문 내용은 원문 대신 확인 기록과 계획 문서(PLAN 4.8)에 기댔다.
- 대상: 원고 커밋 39790748.
- 조정자 확인(11-23): 검토자가 사실 문제로 짚은 다섯 곳을 원문과 대조했고, 다섯 곳 모두 맞다.
  1. WM-Quant 초록(arXiv 2611.21882)은 "approaching full-precision performance at W4A4"다. 기록된 LeWM PushT W4A4 수치는 72.7~82.7 대 FP 98이다. 그래서 본문의 "close to full precision on LeWM"은 강하다.
  2. 부록 C는 "re-implements the range calibration of concurrent work"이고, 본문 2절은 "a simplified form of their range calibration"이다. 서로 다르다.
  3. 부록 G의 E5에는 "SmoothQuant-style smoothing"과 DINO-WM 가중치의 "AWQ-like scheme"이 있다. 그런데 한계 절은 "we did not test … outlier-aware scaling (SmoothQuant, AWQ)"라고 쓴다. 서로 어긋난다.
  4. 서론의 "scenario-fixed policy search"가 Strens & Moore도 함께 인용하는데, Strens & Moore는 scenario를 바꾸는 쪽이다.
  5. "in line with our fixed-dither controls"는 기전이 다른 가중치 고정 draw 결과를 우리 대조군에 잇는다.
- 원고는 이 검토로 아직 고치지 않았다.

---

## 검토자 보고서 (원문 그대로)

파일은 고치지 않았습니다. 줄 수는 원고 사본을 scratchpad에서 tectonic으로 다시 조판해 쟀고, 기준 조판은 원본과 같습니다(43쪽, 본문 12쪽 99%). "줄"은 PDF의 10pt 본문 줄(한 쪽 약 54줄), ¶는 문단입니다.

인용된 논문의 원문은 열지 않았습니다. 그 내용은 `REFERENCES_CHECK.md`와 `PLAN_tmlr.md` 4.8의 기록에 기댔으므로, 아래 고침 문구는 넣기 전에 원문과 대조해야 합니다.

### 1. 판정

**부분적으로 과합니다. 과한 곳은 서론 ¶4와 §2 ¶2–¶3이고, 나머지는 정상입니다.**

| 위치 | 본문 줄 | 인용 키 | 비고 |
|---|---|---|---|
| 초록 | 20 | 0 | |
| 서론 | 52 | 16 | ¶1 5, ¶2 2, ¶4 9. 10편이 §2에서 같은 내용으로 다시 나옴 |
| §2 관련 연구 | 51 | 43 (42편) | 22문장에 `\cite` 34개. ¶1 13, ¶2 4, ¶3 10, ¶4 16 |
| §3 설정 | 65 | 9 | 모델, benchmark, 방법의 출처 |
| §4–§7 결과 | 198 | 6 | §4는 0 |
| §8 한계 | 13 | 5 | |

- **정상인 것**
  - 분량: §2는 본문 글줄 412줄의 12%, 약 1쪽입니다.
  - 문헌 수: 본문 61편입니다.
  - 인용 묶음: 4개 이상 묶음은 논문 전체에 1곳(5개)뿐입니다.
  - 새로움 주장: "first"는 없고 "To our knowledge"는 1회입니다.
  - 결과 절, 초록, 캡션, 결론에는 문제가 없습니다.
- **과한 것**
  - §2 ¶2–¶3이 심사를 거치지 않은 2026년 논문을 한 편씩, 원문의 조건까지 붙여 요약합니다.
  - 서론이 §2와 같은 문장을 되풀이합니다.
  - §2 ¶1–¶3의 26편이 모두 2026년 것이고, 그중 23편이 미심사입니다(arXiv 16, 익명 투고 7).
- **심사자가 읽는 방식**: 거절 사유는 아니고 minor comment 수준입니다. 인상은 "성실하지만 방어적이고, 남의 논문 설명이 길다"입니다.
- **원인**: `REFERENCES_CHECK.md` 3.3에서 보이듯, 문장이 원문과 맞을 때까지 한정어를 더했습니다("PushT 예외", "설정을 밝힌다", "기댓값이 평균인 합의"). 검증 자체는 장점입니다. 다만 정확성을 지키는 더 싼 방법은 덜 말하는 것입니다.

### 2. 중요도순 지적

#### 1) "Concurrent work" 문단이 동시 연구치고 3–4배 깁니다 (§2 ¶2, `main.tex` L133, 13줄)

- **인용**: "Higher precision for early encoder activations, a floating-point action-conditioning path and activation ranges learned per rollout step bring planning … (W4A4) close to full precision on LeWM (Maes et al., 2026) and, with rotation-based quantizers, on DINO-WM Wall, though not on PushT."
- **왜 문제인가**
  - 논증에 쓰이는 것은 셋뿐입니다: 동시 연구라는 것, 그쪽은 보정이고 이쪽은 반올림 규칙이라는 것, rollout-chosen 범위가 그쪽 보정의 단순형이라는 것.
  - 길게 설명할수록 "비교했어야 할 baseline"으로 보입니다. WM-Quant는 본문 인용 3회, §6의 "(Section 2)" 되짚기, 부록 C까지 다섯 군데에 나옵니다.
  - 세부가 곧 위험입니다. 저자 기록에는 LeWM PushT W4A4가 73–83 대 FP 98로 적혀 있는데, 본문은 "close to full precision on LeWM"입니다. 재확인이 필요하고, 과제 이름을 빼면 이 문제가 사라집니다.
  - LeWM과 Fast-WAM 인용은 남의 실험 설정을 가리키려고만 들어간 2차 인용입니다.
  - "corresponds most closely to our systematic component"는 실험하지 않은 대응입니다(저자 기록에도 "우리 해석").
- **고침** (13 → 9줄, −4)

> **Concurrent work.** Hammer et al. (2026) find that quantization error in JEPA-style world models compounds over predictor steps until action rankings become near-random, and their calibration, which includes activation ranges learned per rollout step, brings planning with 4-bit weights and activations close to full precision on some tasks; our rollout-chosen ranges (Section 6) are a simplified form of that range calibration, with one range per layer for all rollout steps. In a quantized world action model, an injected error of equal energy damages the predicted future more when held fixed across the denoising steps than when redrawn at every step (Guo et al., 2026), an analogue of our persistence axis without a planner. The works above calibrate, allocate bits or train; none randomizes rounding at inference or separates how the error is shared from how long it persists.

- "until action rankings become near-random"까지 빼면 −5입니다.
- "concurrent"의 경계가 저자 v1 날짜(11-09)에 맞춰져 있어 심사자에게는 기준이 보이지 않습니다(11-06 논문은 선행, 11-16 논문은 동시). 심사자는 자기 기준으로 읽습니다. 제 기억으로 ICLR은 미출판·최근 4개월, NeurIPS는 마감 전 두 달을 동시 연구로 봅니다.

#### 2) §2 ¶3은 미심사 9편의 한 줄 요약 목록이고, 이 논문과의 관계 문장이 없습니다 (L135, 14줄)

- **인용**: "certified elite decisions become rarer as CEM iterates (Anonymous, 2026a), predicted candidate quality falls to chance at the frame that CEM optimizes (Anonymous, 2026f), …"
- **왜 문제인가**
  - 이 문단은 논문과 개념이 가장 가까운 연구들입니다. 그런데 네 문단 가운데 유일하게 "그래서 우리는 무엇이 다른가"가 없습니다.
  - "예측 오차와 제어 성능이 어긋난다"는 2020년 무렵 심사를 거쳐 확립된 논점(objective mismatch)인데, 2026년 preprint만 인용합니다. RL 심사자가 알아볼 만한 빈자리입니다.
- **고침** (14 → 10줄, −4. Lambert 인용을 넣지 않아도 같습니다)

> **Prediction error versus decisions.** Prediction error averaged over data need not track control performance (Lambert et al., 2020); decision-relevant error measures track planning quality better (You et al., 2026; Wang et al., 2026a), and Min et al. (2026) split end-point error across a candidate pool into a pool-common and an action-relative part. Iterating planners have been found to concentrate their candidates in regions where the model cannot rank them (Anonymous, 2026a;f); replacing early CEM rounds with a deterministic approximation hurts while score noise of the same size does not (Anonymous, 2026b), and harder MPPI search against a learned reward model lowers the true return while its held-out accuracy stays high (Bakshi & Bose, 2026). In concurrent diagnostics, rounding a predictor's weights to nearest at low bit widths turns most PushT failures into scoring rather than search failures (Jia et al., 2026). These studies diagnose where a planner fails; we ask which property of the error it exploits, at a fixed per-entry error distribution.

- 뺀 것: Anonymous 2026d 절, 2026a·f의 개별 결과, "a Goodhart effect" 표현, 8-bit 공동 설계 문장(Zhu et al.). Zhu et al.을 남기려면 ¶1 묶음에 키만 넣으면 됩니다.
- Lambert et al., "Objective Mismatch in Model-based Reinforcement Learning"(arXiv:2002.04523)은 검색으로 존재만 확인했습니다. 서지는 저자 절차로 확인해야 합니다.

#### 3) 서론이 §2를 되풀이하고, 첫 두 문단에 익명 인용이 3개 있습니다 (L99, L102, L106)

- **왜 문제인가**: 서론의 논지가 심사 중인 익명 논문에 기대는 것처럼 보입니다. "common random numbers"는 이웃한 두 문장에 두 번 나옵니다. NVIDIA 모델 보고서는 곁가지 사실 하나 때문에 인용됩니다.
- **고침** (−1, −1, −3 = −5)
  - L99: "…do not explain (Fu et al., 2026; Ranganath et al., 2026)**, unless a calibration objective preserves what planning needs (Anonymous, 2026e)**." 굵은 부분을 삭제합니다(§2와 한계 절에 이미 있습니다).
  - L102: "Iterating planners have been reported to concentrate … (Anonymous, 2026a;f); we isolate…"를 "Planners are known to exploit model error (Boney et al., 2019); we isolate which property of rounding error they exploit."로 바꿉니다.
  - L106: "These ideas draw on classical work: … (NVIDIA, 2026; Khilar, 2026)."를 "These ideas are classical (Section 2): the optimizer's curse (Smith & Winkler, 2006), scenario-based policy search (Ng & Jordan, 2000; Strens & Moore, 2002) and unbiased stochastic rounding (Gupta et al., 2015; Croci et al., 2022)."로 바꿉니다.
  - Kleywegt는 실제로 쓰이는 §5.1에 남습니다. "scenario-fixed"는 "scenario-based"가 맞습니다(Strens & Moore는 scenario를 바꾸는 쪽).

#### 4) 위치 설정 문장은 수가 많지 않지만, 가장 중요한 하나가 너무 넓습니다 (줄 절약 0)

- **"none changes the rounding rule"** (L133)
  - 양자화 심사자는 학습된 반올림을 떠올립니다. 원고 스스로 한계 절에서 AdaRound를 "learned rounding"이라 부릅니다.
  - `refs.bib` 주석에 따르면 LoopQ는 GPTQ를 비교하고, 저자 기록에 따르면 LAQ는 "2차 반올림"을 씁니다.
  - "none randomizes rounding at inference"로 좁힙니다(1번 고침에 반영).
- **"in line with our fixed-dither controls"** (L137): 가중치를 한 번 뽑아 고정한 결과이고 반복 사이의 선택과는 기전이 다릅니다. 삭제를 권합니다.
- **부록 J 마지막 항목** "That we are the first…; we state novelty only to our knowledge": 서론의 "To our knowledge"와 겹치는 이중 방어입니다.
- 나머지는 필요합니다(3절).

#### 5) 익명 7편과 2026년 preprint 의존 (줄 절약 0)

- **사실**: 본문 61편 중 31편이 2026년 것이고 27편이 미심사입니다. 익명 7편은 본문에 12번 나옵니다. 61편 중 20편은 첫 저자 표기가 다른 문헌과 겹칩니다(Anonymous 7, Xu 3, Wang·Zhou·Hansen·Nagel·NVIDIA 각 2).
- **심사자가 읽는 방식**
  - 형식은 문제가 없습니다. URL이 붙은 "Submitted to ICLR 2027"은 OpenReview 표준 서지와 같으니 URL은 유지하세요.
  - 7편은 많습니다(보통 0–2편). 근거로 삼기 어렵고, 이중맹검에서는 주제가 겹치는 익명 인용 묶음을 자기 인용으로 의심하는 심사자도 있습니다.
  - 실명으로 바뀌면 인용 문자열 12곳의 길이가 달라집니다. 99%에서는 13쪽으로 넘칠 수 있습니다.
- **고침**
  - 서론에서 빼고(3번), 한 편을 한 번씩만 인용합니다. 인접 분야 1–2편(WAMPTQ, 2026d)은 빼도 감점이 없습니다.
  - L131의 "quantization error has been reported to recur across their denoising loops (Anonymous, 2026c)"는 지속 개념의 선행 인정이라 남길 가치가 있습니다. 다만 `REFERENCES_CHECK.md` 5.4에 11-01 이후 재확인하지 못했다고 적혀 있습니다. 확인하지 못하면 빼야 합니다.

#### 6) 원문 한정어를 그대로 옮긴 곳 (§2 ¶4 L137, §7 L302)

- **고침** (¶4는 따로 재면 −1, −2이고 함께 넣으면 16 → 14줄. §7은 −1)
  - "reach a consensus whose expected value is the average"를 "…makes quantized average consensus, which fails under deterministic quantization, converge (Aysal et al., 2008)"로 바꿉니다.
  - 언어 모델 세 편 문장(3.4줄)을 "Comparisons of rounding rules for low-precision language-model inference give mixed results (Chatelain & de Oliveira Castro, 2026; Khilar, 2026; Wu et al., 2026)."로 바꿉니다.
  - §7: "…on a Jetson AGX Orin that evaluate 300 CEM candidates 1.36–2.13× faster than FP16"을 "…custom W4A4 integer kernels that are 1.36–2.13× faster than PyTorch FP16 on a Jetson AGX Orin"으로 바꿉니다. 후보 수는 필요 없습니다. 반대로 기준선은 필요합니다. 저자 기록상 기본 PyTorch FP16이고, 이 논문 스스로 사전 변환 BF16이 정수 kernel을 이긴다고 보고하기 때문입니다.

#### 7) 그 밖의 불일치 (본문 줄 0, 선택 항목은 각 −1)

- **부록 C**(`appendix/app_C.tex` L32, PDF 22쪽): "this re-implements the range calibration of concurrent work"인데 본문 §2는 "a simplified form"입니다. `REFERENCES_CHECK.md` 3.3의 수정이 부록에는 반영되지 않았습니다.
- **부록 G**: "SmoothQuant-style smoothing", "AWQ-like scheme"에 인용이 없습니다. 한편 한계 절은 "we did not test … outlier-aware scaling (Xiao et al., 2023; Lin et al., 2024)"라고 씁니다. 무엇을 시험하지 않았는지 저자가 정리해야 합니다.
- **서지**: 2026년 항목만 Title Case입니다(이중 중괄호). "v2" 표기는 Wang et al. 2026a 하나에만 있습니다. `obst2026cem`은 인용되지 않습니다.
- **선택 절약**
  - §3.3의 "(mt80; prior work distilled and quantized it to FP16, …)" 괄호: −1
  - "which is essentially unavoidable on large policy sets (Bhamidipaty et al., 2026)": −1
  - ¶1의 "fused low-bit kernels" 절과 PreDe 절 정리: −1

### 3. 그대로 둘 것

- **§2 ¶4의 고전 계보**(공통 난수, PEGASUS와 Strens & Moore, optimizer's curse, Hansen 2009, Goschin 2013, Roberts·Schuchman, Kieburtz·Aysal, Gupta·Croci). 아는 심사자가 꺼낼 "이건 고전 아닌가"를 먼저 인정합니다. 전부 심사를 거친 문헌이고, 빼면 감점입니다.
- **경계 문장 셋**: "To our knowledge, such dither has not been used to protect sampling-based planning"(1회), ¶4 끝의 "These results concern error that accumulates within one computation; we study…", ¶2 끝의 "The works above…"(좁힌 형태).
- **"our rollout-chosen ranges are a simplified form of their range calibration"**. 동시 연구의 아이디어를 시험했다는 공개이고, §6이 이 문장을 가리킵니다.
- **한계 절의 미시험 baseline 인용**(AdaRound, SmoothQuant, AWQ, 보정 목적함수). "왜 안 했나"를 미리 막습니다.
- **§3과 결과 절의 출처 인용**: TD-MPC2, DINO-WM, CEM, MPPI, Meta-World, DMC, Holm, bias correction, NVFP4·MXFP4, Triton, 그리고 개념을 쓰는 자리의 optimizer's curse와 SAA.
- **서론의 Fu et al.과 Ranganath et al.**, 그리고 동시 연구 네 편 각각의 인용 1회. 저자가 심사자일 수 있습니다.
- **묶음 인용**. 줄을 쓰는 것은 키(약 20자)가 아니라 키마다 붙인 문장입니다. 주변 문헌은 키를 남기고 문장을 지우세요.

### 4. 비는 본문 줄 수

| 수정 | 줄 |
|---|---|
| 1) 동시 연구 문단 | 13 → 9 (−4) |
| 2) §2 ¶3 | 14 → 10 (−4) |
| 3) 서론 세 곳 | −5 |
| 6) §2 ¶4 두 곳 | 16 → 14 (−2) |
| 6) §7 Orin 문장 | −1 |
| **권장 합계 (함께 조판해 잼)** | **−16줄. 본문 끝 12쪽 99% → 72%** |
| 선택 항목까지 | −20줄. 12쪽 64% |
| 1)과 2)만 | −8줄. 12쪽 74% |

- §2는 51 → 41줄, 서론은 52 → 47줄이 됩니다. 참고문헌은 4편 줄고(LeWM, Fast-WAM, NVIDIA Nemotron, Anonymous 2026d) 1편 늡니다(Lambert). Zhu et al.을 ¶1 묶음에 키로 옮기지 않으면 한 편 더 줄어듭니다.
- 작은 수정 하나로는 `build.sh`의 "99%" 표시가 움직이지 않습니다. 줄이 5쪽의 세로 여백으로 흡수되기 때문입니다(서론만 고친 −5줄은 그대로 99%). 1)과 2)를 함께 넣은 −8줄에서 74%로 내려갑니다.
- 비는 줄은 채우기보다 여유로 두는 편이 낫습니다(익명 인용이 실명으로 바뀔 때, 심사 요청을 반영할 때).

### 파일

- 권장 수정을 모두 넣은 시험 사본(원본과 L99, L102, L106, L133, L135, L137, L302만 다름)은 세션 임시 폴더(scratchpad/review/v_F_rec/)에 있었다. 이 사본의 `refs.bib`에 붙인 Lambert 항목은 줄 수를 재려고 넣은 자리표시다.

---

## 반영 결과 (2026-11-23, 사용자: "권장 수정 모두 반영해줘")

- 반영한 것: 1)–6)의 권장 수정 전부와 7)의 부록 C·G 정합. 바꾼 문장마다 원문이나 기록과 대조했다(WM-Quant·Lambert는 원문 페이지, 나머지는 `REFERENCES_CHECK.md` 기록).
- 부록 J의 "That we are the first…" 항목도 지웠다(4번 지적).
- 검토안과 다르게 한 것
  - 8-bit 공동 설계(Zhu et al.)는 ¶1 묶음으로 옮기지 않고 본문에서 뺐다. 동시 연구이고 ¶1의 주어와 맞지 않는다.
  - 선택 항목(§3.3 괄호, Bhamidipaty 절, ¶1 정리)은 넣지 않았다.
- 결과: 본문 끝이 12쪽 99%에서 74%로 내려갔다(검토 추정 72%). strict 빌드, 보충자료(식별 문자열 0), 대역본(11문단 재번역), 보관 사본이 모두 통과했다.
- 기록: `paper/tmlr/REFERENCES_CHECK.md` 6절.
