# 참고문헌 확인 기록 (TMLR 원고, 논문 A)

- 작성: 2026-11-22, 골격 v0. 3절(원 출처 확인)은 같은 날 통합 때 더했다. 대상: [`refs.bib`](refs.bib) 52항목.
- 합친 방법
  - `paper/rss2027/refs.bib`(40항목, 11-01 확인, [`../rss2027/REFERENCES_CHECK.md`](../rss2027/REFERENCES_CHECK.md))를 그대로 가져왔다. 이 파일은 `rss_ext/related_2026.bib`의 23항목을 글자 그대로 담고 있다(diff로 확인). 그래서 두 파일을 합친 결과가 곧 RSS판 bib이다.
  - DINO-WM은 arXiv 2024판 대신 ICML 2025(PMLR 267)판으로 바꿨다. B의 확인 기록(`B_near_miss_blindness/paper/tmlr/REFERENCES_CHECK.md` §1)을 따랐다.
  - 새 항목 12개를 끝에 더했다(아래). 11-22에 원 출처로 확인했다(3절).
- 원고에는 인용한 항목만 나온다(bibtex).

## 1. 새 항목

| key | 무엇 | 근거 | 할 일 |
|---|---|---|---|
| `wmquant2026` | WM-Quant, arXiv:2611.21882 (11-16), 동시 연구 | PLAN 4.8(초록은 11-22 직접 확인, 본문 수치는 에이전트가 읽음) | 제목·저자·판 확인. 본문 7절의 Orin 속도 수치(2.12×·1.63×와 평균 1.71×가 PLAN 안에서 다름)를 원문으로 확인 |
| `searchscoring2026` | Search or Scoring?, arXiv:2611.25347 (11-19), 동시 연구 | PLAN 4.8 | 제목·저자, "예측기 가중치 6~3 bit 결정적 반올림" 문장 확인 |
| `laq2026` | LAQ, arXiv:2611.25614 (11-19), 동시 연구 | PLAN 4.8 | 제목·저자, "재사용 오차 2~3배" 문장 확인 |
| `codesign2026` | 계획기·하드웨어 공동 설계, arXiv:2611.22483 (11-17) | PLAN 4.8 | 제목·저자, "W8A8 손상 없음" 확인 |
| `goodhart2026planner` | Goodhart in the Planner, arXiv:2611.09670 (11-06, 우리 v1 전) | PLAN 4.8·4.9 | 제목·저자, 인용 문장 확인 |
| `loopq2026` | LoopQ, arXiv:2611.05093 | PLAN 4.8 | 제목·저자·날짜. ICLR 2027 투고 LoopQuant(`anon2026loopquant`)의 arXiv판인지 확인 |
| `ng2000pegasus` | PEGASUS (UAI 2000) | `literature/NOVELTY_R2.md` | 쪽수 확인 |
| `strens2002paired` | Strens & Moore, JMLR 2002 | `literature/NOVELTY_R2.md` | 권·쪽 확인 |
| `goschin2013cem` | CEM optimizes for quantiles (ICML 2013) | 기억(로컬 기록 없음) | 전체 확인 |
| `aysal2008consensus` | dithered quantization consensus (IEEE TSP 2008) | 기억 | 전체 확인 |
| `kieburtz1977limit` | randomized quantization과 limit cycle (IEEE TCAS 1977) | 기억 | 전체 확인 |
| `hansen2009uncertainty` | 잡음 있는 진화 전략의 재평가 (IEEE TEC 2009) | 기억 | 전체 확인 |

## 2. 투고 전 할 일

- 위 12항목을 원 출처(arXiv abs 페이지, 출판 페이지)에서 확인하고 `note`를 지운다. 확인하지 못하면 인용을 뺀다. → 11-22 완료(3절).
- 본문의 `\todo{verify}` 네 곳(2절 동시 연구 문장 셋, 7절 Orin 속도)을 원문과 대조한다. → 11-22 완료(3절).
- ICLR 2027 익명 투고 7편(`anon2026*`)의 상태를 12-16 결정 뒤에 다시 본다(철회·공개·arXiv판에 따라 인용 형식이 바뀐다).
- 11-19 이후 arXiv 제출분을 다시 검색한다(PLAN 4.8 마지막 줄).

## 3. 원 출처 확인 (2026-11-22)

- 서브에이전트가 arXiv API·abs 페이지(제목, 저자, 판 이력)와 HTML 본문, PMLR·JMLR 페이지로 확인했다. IEEE 세 편은 출판사가 Crossref에 넣은 기록과 저자 사본(Aysal: McGill 연구실, Hansen: HAL)으로 확인했다. PEGASUS의 쪽수는 arXiv에 올라온 UAI 2000 논문집 쪽의 머리글(406~415)에서 읽었다.
- DBLP(봇 확인 페이지), IEEE Xplore(빈 응답), OpenReview(challenge 페이지)는 열리지 않아 쓰지 않았다.
- 결과: 12항목 모두 확인하고 `note = {TODO verify}`를 지웠다. 키는 그대로다. 항목마다 위 주석에 출처와 날짜를 적었다. 따로 컴파일해 tmlr.bst 경고가 없음을 확인했다.

| key | 출처 | 판정 | 바뀐 점 |
|---|---|---|---|
| `wmquant2026` | arXiv 2611.21882 v1 (11-16) | 고침 | 자리표시 → "WM-Quant: Rollout-Aware Post-Training Quantization for JEPA-Style World Models", 저자 5명 |
| `searchscoring2026` | arXiv 2611.25347 v1 (11-19) | 고침 | "Search or Scoring? Diagnosing Why Visual World-Model Planners Fail", 저자 13명(앞 3명 + others) |
| `laq2026` | arXiv 2611.25614 v1 (11-19) | 고침 | "Don't Lose the Future: Latent-Aware 4-Bit Quantization of World Action Models", 저자 6명 |
| `codesign2026` | arXiv 2611.22483 v1 (11-17) | 고침 | "Manufacturing Differential Sparsity: Planner-Hardware Co-Design for Efficient World Model Planning", 저자 5명 |
| `goodhart2026planner` | arXiv 2611.09670 v3 (v1 11-06) | 고침 | "Goodhart in the Planner: Sampling-Based Control Exploits Rewards Learned from Human Preferences", 저자 2명. v1부터 제목·저자 같음 |
| `loopq2026` | arXiv 2611.05093 v1 (11-03) | 고침 | "LoopQ: Fair, Closed-Loop Evaluation of Quantized Vision-Language-Action Models", 저자 3명. LoopQuant 투고(`anon2026loopquant`)의 arXiv판이 아니다 |
| `ng2000pegasus` | arXiv 1301.3878 (UAI 2000 쪽 영상) | 확인 | 그대로(406–415) |
| `strens2002paired` | jmlr.org/papers/v3/strens02a | 확인 | 그대로(3(Dec):921–950) |
| `goschin2013cem` | proceedings.mlr.press/v28/goschin13 | 고침 | 저자 "Littman, Michael"(PMLR 표기), publisher 추가. 28(3)의 호수는 tmlr.bst가 volume과 함께 못 찍어 뺌 |
| `aysal2008consensus` | Crossref 10.1109/TSP.2008.927071, 저자 사본 | 고침 | "Tuncer Can" → "Tuncer C."(출처 표기). 56(10):4905–4918 맞음 |
| `kieburtz1977limit` | Crossref 10.1109/TCS.1977.1084345 | 고침 | 기억으로 넣은 이름을 출처의 머리글자(R., V., K.)로. 24(6):291–299 맞음. 초록 못 읽음 |
| `hansen2009uncertainty` | Crossref 10.1109/TEVC.2008.924423, HAL inria-00276216 | 확인 | 그대로(13(1):180–197) |

### 3.1 선행과 동시 연구 (기준: 우리 v1, 2026-11-09 02:37 UTC)

| 연구 | v1 (UTC) | 구분 |
|---|---|---|
| LoopQ | 11-03 15:51 | 선행 |
| Goodhart in the Planner | 11-06 16:22 (v2 11-12, v3 11-17) | 선행. 초록의 "학습한 동역학 모델에서도 같다"는 v2부터 |
| WM-Quant | 11-16 20:51 | 동시 |
| 계획기·하드웨어 공동 설계(Candelta) | 11-17 09:40 | 동시 |
| Search or Scoring? | 11-19 01:38 | 동시 |
| LAQ | 11-19 06:59 | 동시 |

### 3.2 원문으로 확인한 본문 내용

- **WM-Quant**
  - 기전: 예측기 step마다 오차가 쌓여 행동 순위가 무작위에 가까워진다(초록, 3.3절, 그림 4). 편향과 지속을 나누지는 않는다.
  - 구성요소 셋: encoder 앞 세 블록의 활성 8-bit, 제어 경로 FP 유지, rollout step마다 따로 둔 활성 범위를 encoder 범위와 함께 Adam·STE로 학습(식 9).
  - 제거 실험(LeWM PushT W3A3, 표 5): 제어 경로를 빼면 −31.6점, rollout-aware scaling −19.6점, encoder 정밀도 −3.2점(CI가 0 포함), 범위를 step끼리 공유하면 −6.8점(CI가 0 포함).
  - W4A4: LeWM PushT 72.7~82.7 대 FP 98(2회째 계획). DINO-WM Wall은 DuQuant 계열만 92~98 대 94. DINO-WM PushT는 최고 42 대 90.
  - Jetson AGX Orin(표 6, 4.2절): 300 후보의 CEM 비용 평가 1회(시작·목표 encoding, 5-step rollout, 비용 포함), 자체 W4A4 정수 CUDA kernel 대 기본 PyTorch FP16. DINO-WM RTN 2.12×(WM-Quant 적용 2.09×), LeWM RTN 1.63×(1.58×), 전체 1.36~2.13×. 평균 1.71×(2모델 × 5방법, WM-Quant 없이), 적용하면 1.70×. PLAN 4.5(2.12·1.63)와 4.8(1.71)은 둘 다 맞고 가리키는 값이 다르다.
  - CEM만 쓰고 MPPI·확률적 반올림·dither는 없다. 코드 링크도 없다.
- **우리 RC 범위**(`DW_dinowm/code/calib_tmlr.py` select_rc): 층마다 범위 하나를 백분위 격자에서 좌표 하강으로 고르고, 모든 step이 같이 쓴다. WM-Quant의 step별 학습 범위를 줄인 형태다.
- **LAQ**: Fast-WAM을 W4A4KV4(회전 + 2차 반올림, 행동 경로 보호)로 둔 상태에서, 같은 에너지의 주입 오차를 denoising step 내내 고정하면 매 step 새로 뽑을 때보다 미래 잠재 오차(reader metric 가중 Z_M)가 커진다. 관측 token 3.1×, 관측 frame cache 2.2×, 텍스트 cache 2.1×(3.2절 관찰 2, 표 2). 실제 반올림 오차가 아니라 주입 오차이고, 지표는 과제 성공이 아니다.
- **Search or Scoring?**: 예측기의 모든 linear 층을 b bit(출력 행마다 대칭, round-to-nearest)로 양자화한다. b = 6, 5, 4, 3이고 encoder·목표·비용은 그대로다(부록 S19). 가중치 양자화다. 개발 과제에서 DINO-WM은 6 bit부터, JEPA-WM은 4 bit부터 실패 대부분이 채점 실패가 된다(7절, 표 2).
- **Candelta**: 예측기 가중치는 채널별, 활성은 텐서별 정적 scale의 INT8이다(w8a8s, LayerNorm·attention은 bf16). 계획 품질이 떨어지지 않는다(FP32 대비 합산 성공률 +1.1점, 3.2절, 그림 4a).
- **Goodhart in the Planner**: 인간 선호로 학습한 보상 모델에 맞춰 MPPI 탐색을 세게 할수록 실제 return이 최대 76% 줄고, held-out 판정 정확도는 0.75~0.97이다. 실제 시뮬레이터로 계획하므로 손상은 보상 모델 탓이다.
- **LoopQ**: 같은 checkpoint·층·엔진에서 방법만 바꾸는 폐루프 벤치마크다. 4 bit에서 RTN·AWQ·GPTQ가 VLA 전용 방법과 같거나 낫다. H100·Thor에서는 저비트가 16 bit보다 빠르지 않고, L4 int8만 4~14% 빠르다. denoising 루프에서 오차가 되풀이된다는 주장은 없다. 이름이 같은 다른 논문("LoopQ: quantization for looped language models", arXiv 2605.16343)과 섞지 않는다.
- **Strens & Moore**: 고정 scenario에서 생기는 'overfitting'(훈련 scenario 쪽 편향)을 재고, 탐색 중 scenario를 바꾸는 방법과 짝 비교로 줄인다(초록, 3.1절). "공유하고 반복마다 새로 뽑는다"의 고전적 선례에 가깝다.
- **Hansen et al.**: 모집단 일부를 다시 평가해 순위 변화로 불확실성을 잰다. frozen noise도 다루도록 재평가 전에 해를 조금 흔든다(HAL 초안 III절).
- **Aysal et al.**: 결정적 균일 양자화로는 합의에 이르지 못한다. dither를 쓰면 거의 확실히 한 양자화 값에서 합의하고, 그 기댓값이 평균이다(초록).
- **Goschin et al.**: 잡음 있는 평가에서 CEM은 기댓값이 아니라 분위수를 최적화한다(PMLR 초록).
- **Kieburtz et al.**: 제목만 확인했다.

### 3.3 본문 수정 제안 (2026-11-22)

| 위치 | 고칠 점 |
|---|---|
| 2절 1문단 | LoopQ를 "denoising 루프" 인용에서 빼고 폐루프 벤치마크로 옮긴다. WM-Quant: DINO-WM PushT 예외, 회복을 범위 보정만의 효과로 쓰지 않기, "systematic 성분"은 우리 해석으로, RC는 "simplified form". LAQ: 설정을 밝힌다 |
| 2절 2문단 | Goodhart(선행)를 "Concurrent" 문장에서 뗀다. "imperfect model"을 학습한 보상 모델로 좁힌다 |
| 2절 3문단 | Aysal "평균으로 수렴" → "기댓값이 평균인 합의". (선택) Kieburtz "controls", Hansen 구체화. (권장) Strens & Moore 선례 추가 |
| 6절 | RC "re-implementation" → "simplified form" |
| 7절 | Orin: 1.36~2.13×, 평균 약 1.7×, 300 후보 CEM 평가 1회, 기본 PyTorch FP16 대비 |

- 원고 반영(11-22 통합): 3.3절의 제안 가운데 2절의 LoopQ·WM-Quant·LAQ·Goodhart·Search or Scoring?·공동 설계 문장, Aysal·Kieburtz 문구, Strens & Moore 선례 문장, 6.1절의 "simplified form", 7절의 Orin 수치를 넣었다. Hansen 구체화(선택)는 본문 12쪽에 맞추느라 넣지 않았다.

### 3.4 남은 일

- `kieburtz1977limit`의 저자 이름 전체와 초록은 IEEE Xplore가 열리는 곳에서 확인한다.
- 고전 6편을 DBLP로 한 번 더 대조한다.
- `anon2026loopquant`의 "denoising step마다 오차가 다시 나타난다"는 11-01 기록에 기댄다. OpenReview가 열리면 다시 본다.
- 확인하지 않은 후보: "Imperfect world models are exploitable"(arXiv 2605.15960, Goodhart 논문이 인용), QuantWAMs(arXiv 2607.28405, WM-Quant가 "오차가 denoising step을 따라 쌓인다"로 인용).

## 4. 추가 항목 (2026-11-22)

- 개정 중인 `main.tex`가 인용할 11항목을 `refs.bib` 끝("second batch" 묶음)에 더했다. 키는 요청받은 그대로다. 이로써 63항목이다(맨 위의 "52항목"은 3절까지 기준). 항목마다 위 주석에 출처와 확인 날짜를 적었다. `main.tex`는 건드리지 않았다.
- 서브에이전트가 출판 쪽 페이지(CVF open access, PMLR, MLSys 논문집과 그 bib, TMLR 논문 목록과 그 bib, NeurIPS 2026 virtual 페이지), 출판사가 Crossref에 넣은 기록(IEEE·ACM), arXiv API·abs 페이지(판 이력), PDF 첫 쪽(PMLR·MLSys 논문, DINOv2 arXiv v2, Triton 저자 사본)으로 확인했다. Holm은 MathSciNet(MR Lookup, MR538597)과 zbMATH(Zbl 0402.62058)로 확인했다.
- DBLP(세 미러 모두 봇 확인 페이지), ACM DL(봇 확인, 403), JSTOR("Client Challenge"), OpenReview API(challenge)는 열리지 않아 쓰지 않았다.
- 판정: 확인 = 요청 때 받은 이름이 출처와 같다. 보충 = 받은 이름이 줄인 꼴이라 출처의 전체 제목으로 채웠다. 찾음 = 논문을 출처에서 찾아 정했다.
- 따로 컴파일(tmlr.sty·tmlr.bst)해, 11항목을 인용한 문서와 `\nocite{*}`(63항목) 모두 BibTeX 경고가 없고 키 중복도 없음을 확인했다.

| key | 출처 | 판정 | 바뀐 점 |
|---|---|---|---|
| `nagel2019dfq` | CVF open access ICCV 2019 페이지, Crossref 10.1109/ICCV.2019.00141 | 확인 | ICCV 2019, pp. 1325–1334(두 출처 같음). 저자 4명 |
| `nagel2020adaround` | proceedings.mlr.press/v119/nagel20a | 확인 | PMLR 119:7197–7206. 저자 5명. 페이지 메타데이터는 "Van Baalen", PDF는 "van Baalen"이라 PDF를 따름 |
| `xiao2023smoothquant` | proceedings.mlr.press/v202/xiao23c | 보충 | 제목 "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models". PMLR 202:38087–38099, 저자 6명 |
| `lin2024awq` | proceedings.mlsys.org MLSys 2024 초록 페이지와 bib, 논문 PDF | 보충 | 제목 "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration". MLSys 6:87–100, 저자 10명. arXiv 최신판(v6)에는 "On-Device"가 없어 MLSys판 제목을 씀 |
| `bengio2013ste` | arXiv 1308.3432 v1 (2013-08-15, 판 하나) | 확인 | 그대로. 학회판 없음 |
| `holm1979` | MathSciNet MR538597, zbMATH Zbl 0402.62058 | 확인 | Scand. J. Statist. 6(2):65–70, 1979. 호수 2는 MathSciNet에서만 확인(zbMATH 기록에는 호수가 없다). JSTOR는 못 엶 |
| `oquab2024dinov2` | jmlr.org/tmlr/papers 목록과 bib(OpenReview a68SUt6zFt), arXiv 2304.07193 v2 PDF | 확인 | TMLR 2024(1월). PDF 머리글 "Published in Transactions on Machine Learning Research (01/2024)". 저자 26명이라 앞 3명 + others(`rouhani2023mx`과 같은 방식) |
| `fu2020d4rl` | arXiv 2004.07219 v4 (2021-02-06) | 확인 | 학회판 없음. 지금 제목은 v2(2020-04-20)부터이고 v1은 "Datasets for Data-Driven Reinforcement Learning". 연도는 2020(저자 README의 인용과 같음) |
| `tillet2019triton` | Crossref 10.1145/3315508.3329973(ACM), 저자 사본(Harvard) | 확인 | MAPL '19(PLDI '19의 workshop), pp. 10–19, ACM. Crossref 제목은 콜론 뒤가 소문자, PDF는 title case. tmlr.bst가 제목을 소문자로 바꾸므로 출력은 같다. ACM DL은 못 엶 |
| `maes2026lewm` | le-wm README → arXiv 2603.19312 v3, NeurIPS 2026 poster 150076 | 찾음 | "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels", 저자 5명. NeurIPS 2026 채택(포스터 12-11, Paris)이라 `@inproceedings`(NeurIPS 2026). 논문집이 아직 없어 권·쪽 없음 |
| `fastwam2026` | LAQ 참고문헌 [36] → arXiv 2603.16666 v2, NeurIPS 2026 poster 149922 | 찾음 | "Fast-WAM: Do World Action Models Need Test-time Future Imagination?", 저자 4명. LAQ는 arXiv판을 인용했지만 NeurIPS 2026에 채택돼(포스터 12-09, Sydney) NeurIPS 2026으로 적음. 권·쪽 없음 |

### 4.1 선행과 동시 연구 (기준: 우리 v1, 2026-11-09 02:37 UTC)

| 연구 | v1 (UTC) | 구분 |
|---|---|---|
| LeWM | 03-13 19:48 (v2 03-24, v3 06-03, 세 판 모두 제목·저자 같음) | 선행 |
| Fast-WAM | 03-17 15:33 (v2 03-23, 두 판 모두 제목·저자 같음) | 선행 |

### 4.2 원문으로 확인한 내용 (본문에 쓸 때 볼 점)

- **Fast-WAM의 구조와 LAQ의 설명**
  - Fast-WAM 초록과 본문(arXiv v2): 학습 때는 영상 co-training을 하고, 시험 때는 미래 예측을 건너뛴다. 미래를 denoising하는 것은 비교용 변형인 Fast-WAM-Joint(영상·행동 함께)와 Fast-WAM-IDM(영상 먼저, 행동 나중)이다.
  - LAQ 5.1절은 자기가 양자화한 Fast-WAM을 "미래 잠재를 먼저 denoising하고, 행동 chunk가 key–value cache로 그것을 참조한다"로 설명하고 변형 이름은 쓰지 않는다. 이 설명은 IDM 변형에 가깝다.
  - 본문에서 Fast-WAM이 미래를 denoising한다고 쓰려면 "LAQ가 양자화한 설정"으로 한정한다.
- **Fast-WAM의 판 차이**: NeurIPS 초록은 110 ms·6배 넘게 빠름, arXiv v2 초록은 190 ms·4배 넘게 빠름이다. 수치를 인용하면 어느 판인지 밝힌다.
- **LeWM**: 손실 두 항(다음 embedding 예측, 잠재의 Gaussian 분포 정규화), 약 15M 매개변수, GPU 한 장에서 몇 시간 학습, foundation-model 기반 world model보다 계획이 최대 48배 빠름(arXiv v3과 NeurIPS 초록이 같다).
- **DINOv2 저자 표기**: TMLR 목록·bib은 "Mido Assran", "Daniel HAZIZA", "Herve Jegou", 논문 PDF는 "Mahmoud Assran", "Daniel Haziza", "Hervé Jegou"다. bib에는 앞 3명만 넣어 영향이 없다. 전체 목록이 필요하면 PDF 표기를 따른다.

### 4.3 남은 일

- NeurIPS 2026 논문집이 나오면(학회 12월) `maes2026lewm`·`fastwam2026`을 논문집 기록과 대조한다.
- `holm1979`의 호수(2)와 제목 표기는 JSTOR가 열리는 곳에서 한 번 더 본다. `tillet2019triton`은 ACM DL에서 한 번 더 본다.
- OpenReview가 열리면 DINOv2 TMLR판 PDF와 NeurIPS 2026 두 편의 camera-ready를 본다.
- 3.4의 DBLP 대조에 이 절의 고전 9편도 넣는다.
