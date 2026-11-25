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
- 11-19 이후 arXiv 제출분을 다시 검색한다(PLAN 4.8 마지막 줄). → 11-23에 11-19 19:00 UTC 제출분까지 봤다(5.5절). 나머지는 11-24 10:00 KST 뒤에 다시 본다.

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

- `kieburtz1977limit`의 저자 이름 전체와 초록은 IEEE Xplore가 열리는 곳에서 확인한다. → 11-23 확인(5.2절). 출판사 기록도 머리글자만 있다.
- 고전 6편을 DBLP로 한 번 더 대조한다. → 11-23 완료(5.1절).
- `anon2026loopquant`의 "denoising step마다 오차가 다시 나타난다"는 11-01 기록에 기댄다. OpenReview가 열리면 다시 본다. → 11-23에도 열지 못했다(5.4절). arXiv판도 없다.
- 확인하지 않은 후보: "Imperfect world models are exploitable"(arXiv 2605.15960, Goodhart 논문이 인용), QuantWAMs(arXiv 2607.28405, WM-Quant가 "오차가 denoising step을 따라 쌓인다"로 인용). → 11-23 확인하고 둘 다 인용했다(5.3절).

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
- `holm1979`의 호수(2)와 제목 표기는 JSTOR가 열리는 곳에서 한 번 더 본다. `tillet2019triton`은 ACM DL에서 한 번 더 본다. → 11-23 둘 다 확인(5.2절).
- OpenReview가 열리면 DINOv2 TMLR판 PDF와 NeurIPS 2026 두 편의 camera-ready를 본다.
- 3.4의 DBLP 대조에 이 절의 고전 9편도 넣는다. → 11-23 완료(5.1절).

## 5. 남은 확인 (2026-11-23)

- 방법
  - DBLP API, IEEE Xplore, JSTOR, ACM DL은 앱 안 브라우저(실제 브라우저)로 열렸다. 11-22에 막혔던 곳들이다.
  - OpenReview는 사람이 직접 통과해야 하는 확인 화면이 떠서 열지 않았다. 봇 확인은 대신 통과하지 않는다.
  - arXiv는 서브에이전트가 API와 abs 페이지로 봤다. 두 후보는 abs 페이지 메타데이터로 한 번 더 확인했다.
- `refs.bib`는 이로써 65항목이다(4절의 63 + 5.3절의 2).

### 5.1 DBLP 대조 (고전 15편: 3절의 6편 + 4절의 9편)

| key | DBLP | 결과 |
|---|---|---|
| `ng2000pegasus` | conf/uai/NgJ00, 406–415 | 같음 |
| `strens2002paired` | JMLR 3:921–950 | 같음 |
| `goschin2013cem` | ICML 2013, 1193–1201 | 같음. DBLP는 "Michael L. Littman"이다. bib은 PMLR 표기 "Michael Littman"을 그대로 둔다 |
| `aysal2008consensus` | IEEE TSP 56(10-1):4905–4918 | 같음. "10-1"은 10호 1부다 |
| `kieburtz1977limit` | 없음 | IEEE Xplore로 확인(5.2절) |
| `hansen2009uncertainty` | IEEE TEC 13(1):180–197 | 같음 |
| `nagel2019dfq` | ICCV 2019, 1325–1334 | 같음 |
| `nagel2020adaround` | ICML 2020, 7197–7206 | 같음 |
| `xiao2023smoothquant` | ICML 2023, 38087–38099 | 고침: "Mickael" → "Mickaël" Seznec |
| `lin2024awq` | MLSys 2024, 저자 10명 | 같음. DBLP에는 쪽이 없다 |
| `bengio2013ste` | CoRR abs/1308.3432 | 같음 |
| `holm1979` | 없음 | JSTOR로 확인(5.2절) |
| `oquab2024dinov2` | TMLR 2024 | 같음 |
| `fu2020d4rl` | CoRR abs/2004.07219 | 같음 |
| `tillet2019triton` | MAPL@PLDI 2019, 10–19 | 같음 |

### 5.2 출판사 페이지

- `kieburtz1977limit`: IEEE Xplore 문서 1084345.
  - 저자: 출판사 기록도 머리글자뿐이다(R. Kieburtz, V. Lawrence, K. Mina; Bell Labs). 그래서 bib의 머리글자를 그대로 둔다.
  - 서지: 24(6):291–299, 1977년 6월.
  - 초록: multiplier 출력을 무작위로 다시 양자화해 limit cycle을 줄이거나 없앤다. 본문 문장 "Randomized quantization controls limit cycles in recursive digital filters"와 맞는다. 3.2절의 "제목만 확인"은 이것으로 바뀐다.
- `holm1979`: JSTOR stable/4615733의 메타데이터는 "Scandinavian Journal of Statistics, Vol. 6, No. 2, 1979"이고 제목도 같다. 호수 2를 확인했다.
- `tillet2019triton`: ACM DL에서 저자 3명(Philippe Tillet, H. T. Kung, David Cox), MAPL 2019, 10–19쪽이 bib과 같다.

### 5.3 후보 두 편 → 인용 추가

- `bhamidipaty2026exploitable`
  - 서지: "Imperfect World Models are Exploitable", 저자 5명, v1 2026-05-15(선행), NeurIPS 2026 poster.
  - 반영: 2절 "Exploitation and rounding rules" 첫 문장에 "which is essentially unavoidable on large policy sets"로 넣었다. 초록의 표현이다.
  - 한정: 존재 정리다. 계획기가 그 역전을 실제로 찾는다는 내용은 없다.
- `quantwams2026`
  - 서지: "QuantWAMs: Calibrating at the Right Granularity for World Action Models", 저자 7명, v1 2026-07-30(선행).
  - 반영: 2절 1문단의 "calibrated and benchmarked in closed loop"에 넣었다.
  - 한정: WM-Quant는 이 논문을 "denoising step을 따라 오차가 쌓인다"로 요약하지만, QuantWAMs 본문에는 그런 주장이 없다. 그 근거로는 쓰지 않는다.
- 두 문장을 더해도 본문은 그대로 12쪽의 99%에서 끝난다. `build.sh --strict`도 통과했다.

### 5.4 ICLR 2027 익명 투고 7편

- arXiv판: 7편 모두 없다(arXiv API 제목·초록 검색, 웹 검색).
  - 이름만 겹치는 다른 논문: LoopQuant for Looped Transformers(2610.16368), LoopQ 두 편(2605.16343, 2611.05093).
- 일정(iclr.cc/Conferences/2027/Dates): 원고 마감 2026-09-25 AoE, 최종 결정 2026-12-16. 7편은 마감 뒤 공개되었으므로 우리 v1보다 앞선다.
- OpenReview는 확인 화면 때문에 열지 못했다. `anon2026loopquant`의 "denoising 루프에서 오차가 되풀이된다"는 여전히 11-01 기록에 기댄다.

### 5.5 11-19 이후 arXiv

- 볼 수 있던 범위: 11-19 19:00 UTC 제출분까지(11-20 목록).
- 꼭 인용할 새 논문은 없다. 본문 분량(12쪽의 99%) 때문에 동시 연구는 넣지 않았다.
  - 중간
    - 2611.24234: VLA의 flow step별 가중치 정밀도. 11-18, 동시.
    - 2611.12127 "Lost in Rounding": 11-09 09:42 UTC로 우리 v1보다 약 7시간 뒤, 동시. bf16 RTN 재귀 상태와 SR을 다룬다.
  - 낮음: 2611.25319, 2611.25415, 2611.26406, 2611.26339, 2611.24470, 2611.24229, 2611.10316(WMPA, 선행).
- `obst2026cem`(본문 인용 없음)은 v2(11-14)에서 제목과 저자가 바뀌었다. 인용하게 되면 고친다.

### 5.6 남은 일

- 11-24 10:00 KST 뒤: 11-20~11-23 arXiv 제출분을 검색한다.
- 12-16 뒤
  - 익명 7편의 상태를 본다. 채택이면 실명으로 바꾸고, 철회 여부도 본다.
  - `anon2026loopquant` 문장을 OpenReview에서 확인한다. 사람이 확인 화면을 한 번 통과해야 한다.
- 12월 NeurIPS 2026 논문집이 나오면 `bhamidipaty2026exploitable`를 대조한다. papers.nips.cc의 2026 쪽은 11-23 기준 404다. `maes2026lewm`·`fastwam2026`은 6절에서 본문 인용이 빠졌다.
- DINOv2 TMLR판 PDF의 저자 표기는 더 볼 필요가 없다. DBLP TMLR 기록으로 앞 3명을 확인했고 bib에는 그 3명만 있다.

## 6. 인용 관행 검토 반영 (2026-11-23)

- 근거: Fable 5.2 max 독립 검토(`tmlr_ext/REVIEW_fable_citations_2026-11-23.md`). 판정은 "부분적으로 과함, minor comment 수준"이었다. 사용자가 권장 수정을 모두 반영하라고 했다.
- **본문 인용에서 빠진 것**: 키는 `refs.bib`에 남아 있지만 인쇄되지 않는다.
  - `maes2026lewm`, `fastwam2026`: 동시 연구 문단의 2차 인용
  - `nvidia2026nemotron`: 서론의 곁가지 사실
  - `anon2026magnitude`
  - `codesign2026`: 8-bit 공동 설계 문장. ¶1로 옮기지 않았다. 동시 연구이고, ¶1의 주어(VLA·WAM)와 맞지 않는다.
- **더한 것**: `lambert2020mismatch`. L4DC 2020, PMLR 120:761–770이고, DBLP와 PMLR 페이지로 확인했다. 2절 3문단의 "prediction error averaged over data need not track control performance" 근거다.
- **익명 인용**: 7편에서 6편이 되었다. 본문의 "Anonymous" 인용은 12회에서 6회가 되었고, 서론에는 하나도 없다.
- **원문 대조로 고친 사실 표현**
  - WM-Quant: "close to full precision on LeWM"을 "close to full precision on some tasks"로 바꿨다(초록 "approaching full-precision performance at W4A4").
  - 부록 C: "re-implements"를 "a simplified form"으로 바꿨다(본문과 같게).
  - 한계 절: "outlier-aware scaling을 시험하지 않았다"를 "단순화한 형태로만 시험했다(부록 G)"로 바꿨다.
  - 부록 G: smoothing과 AWQ 비슷한 방식에 인용을 붙였다.
  - 서론: "scenario-fixed"를 "scenario-based"로 바꿨다(Strens & Moore는 scenario를 바꾼다).
  - "none changes the rounding rule"을 "none randomizes rounding at inference"로 바꿨다.
  - 언어 모델 세 편: "in line with our fixed-dither controls"를 지우고 한 문장("give mixed results")으로 줄였다.
  - Aysal: "converge"로 줄였다.
  - Orin 문장: 후보 수를 빼고 기준선(PyTorch FP16)을 밝혔다.
  - 부록 J: "That we are the first…" 항목을 지웠다(서론의 "To our knowledge"와 겹침).
- **결과**: 본문은 12쪽의 74%(11.74쪽)에서 끝난다. 전에는 99%였다. `build.sh --strict`가 통과했다(등록 문구 376/376, 문구 검사 54/54).

## 7. 투고 직전 OpenReview 확인 (2026-11-26)

- 사용자가 앱 안 브라우저에서 OpenReview에 직접 로그인했다. 그 뒤 익명 투고 7편의 공개 기록을 API(api2.openreview.net/notes)로 읽었다.
- `anon2026loopquant`: 초록에 "static four-bit weights cause reconstruction errors to recur at every denoising step and compound across consecutive requests"가 있다. 본문 "quantization error has been reported to recur across their denoising loops"와 맞는다. 3.4·5.6절의 남은 일이 이것으로 끝났다.
- `anon2026evaluations`: 초록은 "approximating early rounds … harms the search more than score noise that moves it nearly as far"다. 본문의 "score noise of the same size does not"는 원문보다 강해서 "hurts more than score noise that moves the search about as far"로 고쳤다.
- `anon2026wamptq`: ICLR 2027 Withdrawn Submission으로 바뀌었다. 본문 인용에서 뺐다. "fused low-bit kernels"는 Wang et al. 2026b와 LoopQuant가 뒷받침한다.
- `anon2026elitecert`: 제목이 "World-Model Errors at the Elite Cut-off: Selective Re-scoring for Sampling-Based Planning"으로 바뀌었다. bib를 고쳤다. 내용(후보가 수렴할수록 예측 비용의 순위 상관이 떨어짐)은 본문 인용과 맞는다.
- `anon2026resolution`, `anon2026modulemse`: 초록이 본문 인용과 맞는다. `anon2026magnitude`는 6절에서 본문 인용이 빠졌다.
- 남은 익명 인용은 5편이다. 12-16 ICLR 2027 결정 뒤 상태를 다시 본다.

## 8. 투고 당일 arXiv 확인 (2026-11-26)

- 범위: v1이 2026-11-19 19:00 ~ 11-24 18:59 UTC인 전 분야 6,269편(arXiv API)을 키워드로 거르고 직접 읽었다. 목록 페이지 공지 11-23~25분도 봤다. 11-24 19:00 UTC 이후 제출분은 아직 공지 전이다.
- 핵심 주장과 직접 겹치는 논문은 없다. 이 기간의 양자화 논문(2611.30195, 2611.30178)도 추론 때 반올림을 무작위화하지 않는다. 본문의 "none randomizes rounding at inference …"는 그대로 유효하다.
- 동시 연구 가운데 인용할 만한 것(필수 아님; 본문이 12쪽을 꽉 채워서 투고본에는 넣지 않았다. 심사 중 수정 때 검토한다)
  - 2611.27124 "A Plan or an Adversarial Example? A Round-Trip Check for Latent World Models" (11-20): 양자화하지 않은 DINO-WM·LeWM·JEPA-WM에서 CEM/MPPI 반복을 늘리면 예측 비용은 내려가지만 이미 푼 문제를 다시 잃는다. 역동역학 일관성 검사로 고친다.
  - 2611.30195 PTQ4WAM (11-23): 층별 비트와 실행 경로를 과제 점수로 고른다. 결정적 반올림이다. 철회된 WAMPTQ의 arXiv판일 수 있다(확인 못 함).
  - 2611.28919 (11-21): 후보 순서를 바꾸는 비용 오차가 선택 개선을 예측하고, 독립 체크포인트 합의로 이를 평균한다.
- 우리 v1(11-09)보다 앞선 연구로 새로 나온 것: Agrawal et al. 2512.09929, 2610.37891 "When More Planning Hurts", ACID 2607.02403, "Mind the Gap" 2607.12547. 초록으로 확인했다. 모두 계획과 모델 오차의 일반 문제이고, 양자화나 반올림 규칙은 다루지 않는다. 본문의 주장과 겹치지 않아 투고본에 넣지 않았다. 심사에서 물으면 objective mismatch 문단에 묶어 인용할 수 있다.
