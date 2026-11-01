# 논문 A 투고 원고

| 폴더 | 대상 | 양식 | 상태 (2026-10-24) |
|---|---|---|---|
| [`jkaia2026/`](jkaia2026/) | 2026 한국인공지능학회 추계학술대회(JKAIA 2026), **General (Poster) 트랙** | 학회 LaTeX 템플릿 `kaia.cls`, 2쪽 | 완성본 v1. 영문 제출본 `main.tex`/`main.pdf`, 한국어 검토본 `main_ko.tex`/`main_ko.pdf`, EasyChair 국문 초록 `abstract_ko.txt` |
| [`rss2027/`](rss2027/) | RSS 2027 Stage 1 extended abstract (마감 2026-12-04 AoE) | 본문 5쪽 + 참고문헌 1쪽, 이중맹검 | 초안 v2(11-01, E8c 반영, C4 수정안 승인). 주장 C1~C6 승인(11-01). 상태와 제출 점검표는 [`rss2027/README.md`](rss2027/README.md), 보강 실험 E8~E10·E8c는 [`../rss_ext/`](../rss_ext/) |
| ~~(예정) `tmlr/`~~ | ~~TMLR~~ | — | 10-31에 RSS로 바꿈(아래 5절) |
| (예정) ICLR 2027 워크숍 | 워크숍 목록 공개 11-29 | — | 아직 없음. non-archival이면 RSS와 함께 낼 수 있음 |

```bash
cd "A_dithered_world_models/paper/jkaia2026" && python3 build/make_fig.py && tectonic -X compile main.tex && tectonic -X compile main_ko.tex
```

---

## 1. JKAIA 2026 규정 (2026-10-24에 학회 홈페이지에서 직접 확인)

- 출처: [논문제출 안내](https://aiassociation.kr/Conference/ConferenceView.asp?AC=0&CODE=CC20260901&CpPage=377), [논문 모집(CFP 이미지)](https://aiassociation.kr/Conference/ConferenceView.asp?AC=0&CODE=CC20260901&CpPage=376), 학회소식 게시판
- 일정
  - **논문 제출 마감 2026-10-25(일)**, 심사결과 통보 11-09(월). CFP 이미지에 마감 시각은 적혀 있지 않다.
  - 10-24 23시(KST) 기준 게시판에 연장 공지는 없다. 2024년에는 마감 직전에 연장 공지가 올라왔지만, 연장을 기대하지 않고 10-25 안에 낸다.
  - 학술대회: 2026-11-19(목)–20(금), 서울대학교 호암교수회관 베리타스홀
- 제출: EasyChair <https://easychair.org/conferences/?conf=jkaia2026>(계정 필요). 트랙은 **General (Poster)**를 고른다.
- 언어: **논문은 영문**으로 쓴다. EasyChair 입력란의 **제목은 영문, 초록은 국문**이다.
- 심사는 single-blind다. 그래서 PDF에 저자를 적는다.
- 포스터 트랙
  - 학회 LaTeX 템플릿을 쓴다. 초록 서술은 선택 사항이다.
  - 템플릿을 1쪽이나 2쪽으로 늘려 쓴다.
  - 제목, 저자, 소속, 초록(선택)을 넣은 두 장 분량의 요약본 PDF를 낸다.
- 템플릿: 포스터 트랙 링크는 `UploadData/Editor/Conference/202610/4C732AD210FF48878CC64322E9E5470A.zip`(91,635 bytes, Last-Modified 2026-10-01)이다. DriveWAM 포스터(2026-10-15)가 받은 것과 같은 파일이라, 그 폴더의 `kaia.cls`를 복사해 썼다. 일반 트랙 템플릿(202310 ZIP)은 다른 파일이다.
- 저작권
  - 양도 여부는 선택이다. **양도하지 않는다**(TMLR과 ICLR 워크숍 투고를 위해).
  - 학회 안내: 저작권 양도 여부와 별도로, 논문이 승인되는 시점에 다른 학회나 저널에 저작권이 양도돼 있으면 안 된다(승인 통지나 발표 준비 같은 가확정도 포함).
  - 상금과 JARA 출판은 일반(Full Paper) 트랙만 해당한다. 포스터 논문은 논문집에 실리지 않고 국제학회에 낼 수 있다(CFP 이미지 4번 항목, DriveWAM README에 기록).

## 2. 제출할 것

| EasyChair 항목 | 내용 |
|---|---|
| Track | General (Poster) |
| Title (영문) | Planners Exploit Deterministic Error: Dithered Low-Bit World Models |
| Authors | YongHwi Song (충북대학교 정보통신공학부, thddydgnl1937@gmail.com), Keon Myung Lee (충북대학교 소프트웨어학부, kmlee@cbnu.ac.kr) |
| Abstract (국문) | [`jkaia2026/abstract_ko.txt`](jkaia2026/abstract_ko.txt)를 그대로 붙여 넣는다 |
| Keywords | World models; model-predictive control; quantization; stochastic rounding; robot planning |
| Paper | [`jkaia2026/main.pdf`](jkaia2026/main.pdf) (2쪽) |
| 저작권 양도 | 선택하지 않음 |

- 저자 표기는 DriveWAM 포스터와 CV에 맞춰 "YongHwi Song"으로 했다. 이건명 교수님 소속과 이메일은 JEV 논문(KIIS 2026 추계)과 같다.

**제출 전 확인**
- [ ] `main_ko.pdf`를 읽고 고칠 부분을 정한 뒤 영문 `main.tex`에 반영한다(두 파일의 문단 순서는 1:1로 같다).
- [ ] 이건명 교수님께 공저자 등재를 확인받는다.
- [x] 10-25 오전에 서버의 남은 결과를 확인한다(아래 4절). 지연 재측정이 첫 측정과 다르면 지연 문장만 고친다.
  - 10-25 확인: 남은 결과가 모두 나왔다. 지연 재측정의 짝 오버헤드 중앙값이 36조건 모두 ±3.2% 안이라, 아래 4절 기준으로는 지연 문장을 고치지 않는다.
- [x] 고친 뒤 `main.pdf`가 2쪽인지 확인한다(`python3 -c "import pypdf; print(len(pypdf.PdfReader('main.pdf').pages))"`).

## 3. 수치 출처 (2026-10-24, 결과 파일에서 직접 다시 계산)

| 원고의 수치 | 출처 |
|---|---|
| 최대 100%p 하락(mw-button-press k=2) | `main/results/E1_tdmpc2/e1_tdmpc2_cells.csv` (D_det 1.000) |
| 손상 칸 36개 중 29개에서 det 1-step 오차 ≤ sh-iter | 같은 파일의 `pt_err1h`(Meta-World 14/19, DMC 10/12), `main/results/E1/e1_dinowm_cells.csv`의 `rel_err1`(5/5) |
| mw-pick-place k=3: 0.22 대 0.39, 37.8 대 7.8%p | `e1_tdmpc2_cells.csv` |
| Meta-World 19/19칸 red ≥ 0.5, 중앙값 95% | `e1_tdmpc2_cells.csv` (red 중앙값 0.947) |
| DINO-WM 4/5칸 | `e1_dinowm_cells.csv` (red 0.92, 1.00, 0.20, 0.69, 1.09) |
| drift 비 ≤ 0.24, 지속 dither 70–130% | `main/results/E2_tdmpc2/e2_tdmpc2_summary.json`의 `layer_b`(r_sh-iter 0.10, 0.24, 0.03; r_sh-step 0.70, 0.81, 1.30; r_sh-ep 0.97, 0.87, 1.25), `main/results/E1/e1_dinowm_v52.json`(PushT r −0.01) |
| 반복 12회: sh-iter 0–11%, det 43–47% | `main/results/E2_tdmpc2/e2_tdmpc2_cells.csv` (설정 it1·it3·base·it12) |
| H2c 3/4, H4a 5/6 | `E2_tdmpc2/e2_tdmpc2_summary.json`의 `H2c`, `main/reports/E2_dinowm.md`(PushT 불성립), `E3_tdmpc2/e3_tdmpc2_summary.json`·`E3_dinowm/e3_dinowm_summary.json`의 `H4a` |
| H3b 10/19, 1/3 | `E5_tdmpc2/e5_tdmpc2_summary.json`, `E5_dinowm/e5_dinowm_summary.json` |
| 지연 오버헤드 최대 6.9%(표 1의 H5), 정수 kernel이 fp32보다 약 18% 느림 | `E7_tdmpc2/e7_tdmpc2_summary.json`의 `H5_overhead`(int8 3.8%, int4 5.8%, packed int4 6.9%), `latency_rel_fp32`(중앙값 17.5–18.2%) |
| 두 측정 모두 중앙값 10% 미만, 짝 재측정 −3.2 ~ +2.7% | `main/reports/posthoc_tdmpc2.md` §0(재측정의 사전 정의 계산 최대 int8 9.1%, int4 3.0%, packed 5.7%) |
| DMC 설명 Spearman 0.83 | `E3_tdmpc2/e3_tdmpc2_final.json`의 `DMC_explanation` (0.827) |
| 지속 몫 DMC 0.33–0.52, Meta-World 0.54–0.73 | 같은 파일 `DMC_explanation.rows`의 `S_det` (walker 0.330, quadruped 0.373, cheetah 0.444, finger 0.522; Meta-World 0.539–0.733) |
| DMC 12칸 중 5칸에서 손상 절반 이상 제거 | `e1_tdmpc2_cells.csv` (DMC red ≥ 0.5인 칸 5/12) |
| walker k=3 78.9% 대 56.9% | `e1_tdmpc2_cells.csv` |
| 모의 92만 2천 회, 기울기 0.92–1.05 | `main/reports/E4_theory.md` 요약 |

- 보고서와 다른 점 하나: `main/reports/E1_tdmpc2.md` §5.2는 Meta-World에서 red ≥ 0.9인 칸을 "19칸 중 14칸"이라고 적었다. 실제로는 13칸이다. mw-sweep-into k=3의 red가 0.897인데 표에 0.90으로 반올림돼 있다. 그래서 원고에는 반올림에 흔들리지 않는 중앙값(95%)을 썼다.
- 원고에 쓴 주장은 2026-10-24 사용자가 승인한 범위(추천 1~5)다.
  1. 기전: 지속 오차 이용(v5.2 문구)과 후보 수 용량-반응(H2c)
  2. 처방: "반복하는 계획기"로 한정
  3. 처방 범위: Meta-World와 DINO-WM. DMC는 기전이 설명하는 경계로 서술한다(10-24 사용자 승인, 아래 참고)
  4. 강한 PTQ와의 결합: 계열별로 보고하고, 학습 없는 처방으로 서술한다(QAT 결론 없음)
  5. 지연: 같은 비트에서 오버헤드 10% 이하, 낮은 비트의 속도 이득 없음
- 10-24 수정(사용자 승인)
  - DMC를 "한계"에서 본문 "Where the fix is weak" 문단으로 옮겼다. DMC는 det의 유리한 오차 가운데 반복 사이에 지속되는 몫(S_det)이 가장 작고, dither 효과도 가장 작다. 근거는 1단계 뒤, E3 자료 전에 사전등록한 DMC 설명 규칙이다(v5 §7.2, ρ ≥ 0.5이면 "설명됨", 결과 0.83). 처방 주장은 Meta-World와 DINO-WM으로 둔다(v5.2 §9.2). 표 1에 이 규칙의 행을 더했고, H3a의 DMC 불성립은 그대로 적었다.
  - 사전등록은 "internal pre-registration"(단계마다 시각을 기록한 설계 문서로 고정)이라고 썼다. OSF 같은 외부 등록이 아니고, 이 폴더는 git 저장소도 아니다. TMLR 때는 설계 문서를 보충자료로 공개한다.
  - 2쪽에 맞추려고 참고문헌을 한 문단으로 쓰고, 인용 번호는 본문에 직접 적었다(`\cite` 없음). 참고문헌을 더하거나 순서를 바꾸면 번호를 손으로 맞춘다.
- 10-25 수정(사용자 승인, 결과와 다시 대조한 뒤)
  - 초록의 "and overrates the plan it selects"를 뺐다. 국문 검토본 초록의 "자신이 고른 계획을 과대평가한다"와 EasyChair 국문 초록(`abstract_ko.txt`)의 같은 구절도 뺐다.
    - 이유: DINO-WM에서 손상이 가장 큰 PushT k=3의 det OEI가 +0.217 [0.165, 0.274]로 양수다. PushT k=5도 +0.030이다(`main/results/E1/e1_dinowm_cells.csv`).
    - 그래서 손상 칸 5개 중 2개에서 "고른 계획을 과대평가한다"가 성립하지 않는다.
    - 앞 구절(유리하게 보는 후보 쪽으로 이동)은 기전 과제 4개 모두에서 성립한다(H2a′). 다만 PushT k=3·k=5에서는 원래 drift도 양수라서, 10-25 수정 2에서 이 구절의 범위를 기전 과제로 한정했다.
  - "never worse than det in their affected cells"를 "in any affected Meta-World or DINO-WM cell"로 바꿨다. 국문 검토본의 "두 계열의 손상 칸"도 "Meta-World와 DINO-WM의 손상 칸"으로 바꿨다.
    - 이유: "두 계열"로 읽으면 DMC walker k=3(TD-MPC2 계열, sh-iter가 해로움)과 충돌한다.
  - 두 PDF를 다시 만들었고 모두 2쪽이다. 그림(`build/make_fig.py`)은 다시 만들지 않았다(본문만 바뀜).
- 10-25 수정 2(사용자 승인, 위 수정을 검토한 뒤)
  - 기전 구절의 범위를 사전등록 검정 범위로 한정했다.
    - 초록: "in the four tasks where we tested the mechanism, an iterating planner drifts toward candidates this field happens to favor". 국문 검토본과 `abstract_ko.txt`는 "기전을 검정한 네 과제에서 … 이동했다".
    - 기전 문단: "in all four mechanism tasks at their primary bit-widths"(국문: "각 과제의 1차 비트 수").
    - 이유: '과대평가' 구절을 뺀 근거가 이 구절에도 해당한다. PushT k=3·k=5에서 det drift가 +0.243 [0.191, 0.299], +0.032 [0.026, 0.038]로 양수다.
    - 원인은 영역 효과다. 양자화하지 않은 계획기의 모집단을 양자화 모델로 재면 drift_ref가 +0.448, +0.035다. 그래서 기준 대비 값(det − ref)은 손상 칸 5개 모두 음수다(−0.206, −0.092, −0.003, −0.028, −0.013). 이 대비는 기술 분석(v5.2 §9.1 (b3))이라 초록에는 쓰지 않았다.
    - 출처: `main/results/E1/e1_dinowm_v52.json`의 `cells.<칸>.vars.det.drift`, `cells.<칸>.vars.drift_ref`
  - 지연 문장을 "두 측정 모두 중앙값 10% 미만"으로 고쳤다.
    - 한계 절: "the median latency overhead of dither stayed below 10% in both measurements (paired re-measurement: −3.2% to +2.7%)". 초록: "at under 10% median extra latency". 국문도 같다.
    - 이유: "at most 6.9%"는 첫 측정 값이다. 재측정에서 같은 사전 정의 계산은 int8 최대 9.1%이고, 짝 CI 상한은 16/36조건에서 10%를 넘는다(`main/reports/posthoc_tdmpc2.md`).
    - 표 1의 H5 행(max 6.9%)은 사전등록 판정(첫 측정)이라 그대로 뒀다.
  - 2쪽을 지키려고 영문에서 덜어 낸 것(의미는 같다)
    - 초록 첫 문장의 "such as MPPI and CEM"(같은 초록의 "TD-MPC2 (MPPI …)", "DINO-WM (CEM …)"에 남아 있다)
    - 그림 캡션 축약
    - "Where the fix is weak" 마지막 문장 축약
  - 두 PDF 모두 2쪽이다.

- 10-25 수정 3(사용자 지시, `../workspace/JKAIA_POSTER_REVIEW_2026-10-25.md`의 필수 항목 A1~A3)
  - A1: Setup의 "such cells are called affected" 뒤에 "(31 TD-MPC2 and 5 DINO-WM cells)"를 넣고, 다음 문장을 "In 29 of these 36 affected cells"로 바꿨다. "36 task–bit cells"(TD-MPC2 전체)와 "29 of 36 affected cells"(손상 칸 합계)가 같은 36으로 읽히는 문제를 푼다. 국문: "손상 칸이라 부른다(TD-MPC2 31칸, DINO-WM 5칸)", "이 손상 칸 36개 중 29개".
  - A2: Other limitations의 "10 of 19 TD-MPC2 cells"를 "10 of the 19 affected 3--4-bit TD-MPC2 cells"로 바꿨다(E5의 19칸은 E1의 Meta-World 손상 칸 19개와 다른 집합). 국문: "3--4비트 손상 칸 19개 중 10개".
  - A3: 초록의 지연 구절을 "; in a fused integer kernel (TD-MPC2) the median extra latency was under 10%"로 바꿨다. 지연은 TD-MPC2 융합 kernel에서만 쟀는데 DINO-WM 문장에 붙어 있었다. `main_ko.tex` 초록과 `abstract_ko.txt`는 "TD-MPC2의 융합 정수 kernel에서"로 맞췄다.
  - 2쪽을 지키려고 참고문헌 앞 `\vspace{3pt}`를 `1pt`로 줄였다(본문 글은 더 줄이지 않았다). 두 PDF 모두 2쪽이다.
  - 권장·선택 항목(A4~A9)은 반영하지 않았다. 검토 보고서에 교체 문장이 있다.

- 10-25 수정 4(사용자 지시, 검토 보고서의 권장 항목 C1·A4~A6와 국문 초록 재작성)
  - C1: 초록 "In internally pre-registered experiments"(B 포스터와 표기를 맞춤).
  - A4: "with one iteration sh-iter is no better than det"(반복 1회에서 sh-iter 손상이 det보다 크므로 "as harmful as"는 약한 표현이었다). 국문 "det보다 낫지 않지만".
  - A5: "where less of det's favorable error persists across iterations (the persistent share, drift over OEI)"로 지속 몫을 정의했다. 국문 "반복 사이에 지속되는 몫(지속 몫, 이동을 OEI로 나눈 값)".
  - A6: "as the number of independent offsets it draws (its table size) grows". 국문 "독립 offset 수(표 크기)".
  - 국문 초록(`abstract_ko.txt`)을 다시 썼고 `main_ko.tex` 초록을 같은 글로 맞췄다(A9 해소). 바꾼 것: "MPPI·CEM"의 가운뎃점과 "%p", "%"를 없앴다(템플릿의 기호 금지 규칙, B 국문 초록과 같은 처리: "100퍼센트포인트", "95퍼센트", "10퍼센트"), "내부에서 사전등록한 실험", "유리하게 평가하는 후보", "반복하는 계획기를 필요로 하며". 수치와 주장은 그대로다. 영문 초록의 "%"는 그대로 뒀다(선택 항목 C3).
  - 2쪽을 지키려고 mdframed 안쪽 위아래 여백과 skip을 4pt → 2pt로 줄였다(B와 같음). 본문은 줄이지 않았다. 두 PDF 모두 2쪽이다.
  - 반영하지 않은 것: A7(1차 근사 표현), A8(Wall 손상 칸 0개 언급), C2(참고문헌 제목), C3(영문 초록의 %).

- 10-25 수정 5(사용자 지시: 기계 번역·생성문처럼 읽히는 문장 정리. 수치와 주장은 바꾸지 않음)
  - 초록: "what matters is the structure of the rounding error rather than its size", "drifts toward the candidates that this error favors", "Replacing it with dither (stochastic rounding) shared by all candidates and redrawn at every planner iteration removes most of this drift, at the same bit-width and without training", "The fix requires an iterating planner".
  - Problem Setup: 끝의 수사 의문문을 "We ask why det hurts planning more than its one-step error suggests, and whether changing only the rounding repairs it."로. "the most direct way".
  - Novelty: "These principles (…) are classical; their application …, the two-axis separation and the effect size are new", "failed hypotheses are reported".
  - Algorithms: "(sh-iter, the proposed fix)", "the official defaults", "has zero expected drift"("exactly" 삭제, A7도 함께 해소).
  - Experiments: "Damage tracks the drift:", 단락 제목 "Limits of the fix."(국문 "처방의 한계"), "If dither works by removing persistent exploitation, it should help less where less of det's favorable error persists (the persistent share, drift over OEI). That share is …, and on DMC … is harmful on walker", "confirms that this share predicts", "the low-bit kernels gave no speed-up".
  - 2쪽 유지: 그림 폭 1.0 → 0.90, `\maketitle` 아래 `-28pt` → `-36pt`. 두 PDF 모두 2쪽.
- 10-25 수정 6(사용자 승인, 수정 3~5를 검토한 뒤)
  - 필수 1: "Damage tracks the drift:"를 "The benefit needs iterations:"로 바꿨다. 국문은 "손상도 이를 따른다"를 "dither의 이득은 반복이 있어야 생긴다"로 바꿨다.
    - 이유: det의 drift는 반복 1 → 12회에서 0 → −60으로 커지는데(mw-push), det 손상은 0.33~0.47로 거의 그대로다. 반복 1회에서는 drift가 정의상 0인데도 손상이 43~57%다(`main/results/E2_tdmpc2/e2_tdmpc2_cells.csv`).
  - 필수 2: 초록의 "Replacing it with dither"를 "Replacing round-to-nearest with dither"로 바꿨다. 바로 앞 명사가 "this error"라 it이 모호했다.
  - 필수 3: 영문 초록의 "%" 두 개를 "95 percent", "10 percent"로 바꿨다(템플릿 원문 규칙: 제목과 초록에 기호·특수문자를 쓰지 않는다). 이제 영문 초록에 기호가 없다.
  - 다듬기 1(레이아웃 복원)
    - mdframed 여백과 skip을 4pt로, `\maketitle` 아래를 `-28pt`로, 그림 폭을 `\linewidth`로, 참고문헌 앞을 `3pt`로 되돌렸다.
    - 대신 2쪽을 지키려고 세 군데를 줄였다.
      - Other limitations 끝의 한 단어 줄("so low bits gave no speed-up")
      - 알고리즘 절의 det+shift 정의 문장: 이 2쪽 원고의 결과·표 어디에도 det+shift가 다시 나오지 않는다. 국문 검토본도 같이 뺐다.
      - 그림 높이 1.6 → 1.45인치(`build/make_fig.py`). 폭은 그대로라 그림 글자 크기는 그대로다.
  - 다듬기 2(국문 번역투): "내부에서 사전등록한 실험을 … 수행한 결과"를 "TD-MPC2(…)와 DINO-WM(…)으로 내부 사전등록한 실험에서"로, "반복하는 계획기를 필요로 하며"를 "반복하는 계획기가 필요하고"로 바꿨다(`abstract_ko.txt`와 `main_ko.tex` 초록, 두 글은 글자까지 같다).
  - 두 PDF 모두 2쪽이다.
- 10-25 수정 7(사용자 요청: 한계 부분이 AI가 쓴 글처럼 보임)
  - "Other limitations" 문단(국문 "그 밖의 한계")을 없앴다. 그 문단의 사실은 이렇게 옮겼다.
    - 지연: 표 1 H5 행을 "holds weakly (max 6.9%); no speed-up over fp32"로 바꿨다(국문: "약하게 성립(최대 6.9%); fp32보다 빠르지 않음"). 초록의 "median extra latency under 10 percent"는 두 측정 모두에 맞으므로 그대로 뒀다. 짝 재측정 수치(−3.2 ~ +2.7%)는 본문에서 빠졌다.
    - 강한 PTQ와의 결합: 표 1 H3b 행(TD-MPC2 10/19 성립, DINO-WM 1/3 불성립)에 이미 있다. "PushT는 최선 PTQ만으로 손상 대부분이 사라진다"는 설명은 뺐다.
  - "Limits of the fix" 문단(국문 "처방의 한계")은 사실만 남겨 "DMC." 문단으로 바꿨다. "If dither works by …, it should … and it does" 같은 문장 틀을 없애고 지속 몫 정의, DMC 0.33–0.52 대 Meta-World 0.54–0.73, Spearman 0.83(1단계 뒤 고정, 새 seed), DMC 5/12칸과 walker 3비트 결과를 그대로 적었다.
  - 자리가 생겨 그림 높이를 1.6인치로 되돌렸다. 두 PDF 모두 2쪽이다.

## 4. 10-24 기준으로 아직 나오지 않았던 결과 (모두 기술·사후 항목, 판정은 바뀌지 않음; 10-25에 모두 나옴)

- 10-25 결과
  - E2 EO: 기술 결과이고 원고 수치에는 영향이 없다(`main/reports/E2_dinowm.md` §7).
  - QAT+none, 지연 재측정: `main/reports/posthoc_tdmpc2.md`
    - 지연 짝 오버헤드 중앙값은 36조건 모두 −3.2 ~ +2.7%다.
    - 다만 CI 상한은 16/36조건에서 10%를 넘는다.
    - 4절 기준(짝 오버헤드 > 10%)에는 해당하지 않는다. 하지만 "at most 6.9%"가 첫 측정 값이라, 10-25 수정 2에서 지연 문장을 "두 측정 모두 중앙값 10% 미만"으로 고쳤다.
- 10-24에 적은 내용

- DINO-WM E2 elite 비율·반복 수 sweep(`E2_dinowm_EO`): 10-24 13:51Z에 118/150, 23:45 KST 무렵 끝날 예정
- 사후 점검(DESIGN_main v5.3 §10): TD-MPC2 QAT+none, E7 짝 모드 지연 재측정. 10-25 02시 KST 무렵 끝날 예정
- 원고에 영향을 줄 수 있는 것은 **E7 지연 재측정**뿐이다.
  - 같은 비트의 det·dither 짝 오버헤드가 어떤 조건에서 10%를 넘으면, 초록의 "less than 10% extra latency"와 "Other limitations" 절의 "at most 6.9%"를 "pre-registered median ≤ 10%; a paired re-measurement found …"처럼 고친다.
  - H5 판정은 첫 측정의 사전 정의 계산 그대로다(v5.3 §10.3).

## 5. 다른 투고와의 관계

- **10-31: 사용자가 A를 RSS 2027 Stage 1(extended abstract, 12-04 AoE)에 내기로 정했다.** 같은 날 보강 실험 E8~E10을 RunPod에서 돌렸다([`../rss_ext/`](../rss_ext/)). 원고는 그 결과와 주장 재구성안(`../rss_ext/CLAIMS_rss.md`) 승인 뒤 쓴다.
  - RSS 2027 CFP(10-31 공식 사이트 재확인): Stage 1은 본문 5쪽 + 참고문헌 1쪽, 가설·방법론을 심사, 메타리뷰가 요구 평가를 적고 그것이 Extension Charter가 된다(02-26 결정). 최종 8쪽은 04-16. 이중맹검이고 preprint는 허용된다. 동시 투고는 금지다.
  - 그래서 RSS 심사 중에는 TMLR에 낼 수 없다. Stage 1에서 떨어지면(02-05) TMLR을 대안으로 쓴다.
- (이전 기록) 10-24 이전 계획은 A → RSS 2027 stage 1(12-04)이었다. 10-24에 사용자가 A의 투고처로 JKAIA, TMLR, ICLR 2027 워크숍을 말했다. RSS와 TMLR은 동시 투고가 안 되므로, TMLR을 준비할 때 다시 확인한다.
- TMLR은 병행 심사 중인 archival 학회와 글, 그림, 결과를 공유하는 것을 금지하고, non-archival 워크숍과 arXiv는 허용한다. JKAIA 포스터는 논문집에 실리지 않고 저작권도 양도하지 않으므로 충돌하지 않는다. TMLR 제출 때 이 사실을 밝힐 수 있게 CFP 안내를 보관한다.
- ICLR 2027 워크숍은 non-archival인 곳을 골라야 TMLR과 겹치지 않는다.
