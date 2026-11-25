# TMLR 모의 심사 (Fable 5.2 max, 2026-11-25)

- 요청(사용자, 11-25): "최종적으로 이걸 내면 리뷰어가 accept을 줄까, 피어리뷰를 몇 점 줄까를 면밀하게 분석해주고, 이유도 받아서 수정할 수 있게 해줘"
- 검토자: Fable 5.2, effort max. 역할은 TMLR action editor 한 명과 리뷰어 세 명(R1 양자화, R2 MBRL·계획, R3 방법론·통계)이다.
  - 독립 검토다. 이전 검토 보고서(`REVIEW_*.md`)와 `PLAN_tmlr.md`는 읽지 않게 했다.
  - 프롬프트는 `PAPER_REVIEW_PROMPTS.md`의 검증 1을 바탕으로 했다.
- 대상: 원고 커밋 1f5208d7. 본문은 12쪽의 74%에서 끝난다.
- 조정자 확인(11-25): 판정에 영향이 큰 지적 네 개를 원자료와 대조했다. 넷 모두 맞다.
  1. 보충자료에 공개 저장소의 커밋 해시가 남아 있다.
     - 위치: `rss_ext/LOG.md:27`, `tmlr_ext/LOG.md:13,14,60,263-265`, `TD_tdmpc2/REPORT.md:158,234`, `code/analyze_x5.py:1`, `x5_section.py:23` 등.
     - "우리 v1(11-09)"(`tmlr_ext/LOG.md:172`)와 원고 작업 기록(164-165행 등)도 남아 있다.
     - 다른 투고처 이력도 남아 있다(`main/reports/FINAL_A.md` 9.1절의 RSS·ICML 등, `rss_ext/DESIGN_ext.md` 1·5행의 RSS 2027). README가 "other venues … were replaced"라고 한 것과 다르다.
  2. 본문 4.2절의 "0.00–0.24 for every redrawn dither"는 TD-MPC2 값이다. DINO-WM 새 분할의 sh-iter·det+shift ρp는 0.17–0.46이다(`analysis_dw.json` E11 cells e13 last).
  3. DMC에서 정책 단독의 D는 13.7–56.3%다(`x6_summary.json`). 그래서 τ=10% 안에 드는 저비트 계획기는 반올림 규칙과 상관없이 정책을 이긴다. 초록 문장은 사실이지만 dither 덕분처럼 읽힌다.
  4. 부록 A D1의 "the split used in E0 to reproduce the published DINO-WM results"는 틀렸다. `main/reports/E0_dinowm.md`의 재현은 파일럿 R2-13의 재현이다.
- 아직 확인하지 않은 것
  - DINO-WM 원 논문의 평가 프로토콜(재계획 MPC)과 공개 수치(90/96/98%). F3 문장을 넣기 전에 원 논문에서 확인한다.
  - F9의 구간 기준 개수(18, 6, 11). 저자 스크립트(RandomState(0))로 다시 계산해야 한다.
- 원고와 보충자료는 이 검토로 아직 고치지 않았다.

---

## 검토자 보고서 (원문 그대로)

# TMLR 모의 심사 결과 — "Planners Exploit Deterministic Error: Dithered Low-Bit World Models"

가장 가능성 높은 결정은 **accept with minor revision**입니다. 다만 투고 전에 보충자료 정리와 본문 문구 4곳은 꼭 고쳐야 합니다. 프로젝트 파일은 수정하지 않았고 git도 쓰지 않았습니다. 점수와 확률은 제 판단이고, 수치 대조는 파일에서 직접 확인한 것입니다.

## (0) 판정 표

| 리뷰어 | 권고(현 원고) | Claims & Evidence | Audience | 1–10 | 확신 |
|---|---|---|---|---|---|
| R1 양자화·효율 추론 | Leaning Accept | Yes (문장 3곳 수정 조건) | Yes | 6 | 4 |
| R2 MBRL·표본 기반 계획 | Leaning Reject (문구 수정 시 Leaning Accept) | No (현 상태) | Yes | 5 | 4 |
| R3 실험 방법론·통계 | Leaning Accept | Yes (조건부) | Yes | 7 | 4 |

| AE 결정 확률(의견) | accept as is | minor revision | reject |
|---|---|---|---|
| 지금 그대로 제출 | 10% | 55% | 35% |
| 아래 must·should 반영 후 | 15% | 62% | 23% |

결정을 가를 쟁점 3개:
1. **DINO-WM 쪽 증거가 얇은데 계열 대비 문장은 세다.** 새 분할의 손상 칸은 2개(하나는 post hoc)이고 Wall 기준선이 불안정하며, 프로토콜이 공개 DINO-WM 결과와 다릅니다.
2. **처방의 실질 의미.** Meta-World는 정책만으로 계획기와 거의 같고, DMC의 비트 이득은 0–1입니다. 초록의 DMC 문장은 dither의 공처럼 읽히지만 det에도 성립합니다.
3. **통제의 해석과 본문 밀도.** det+noise가 "설명 못 한 이상 결과"로 남아 있고, 1-step 오차 결과는 사실상 구성상 예상되는 것이며, 본문이 매우 빽빽합니다("clear evidence" 기준).

심사 전 위험은 따로 하나 있습니다. 보충자료 로그에 원고 제작 과정, 다른 투고처, 프리프린트 날짜, 공개 저장소의 커밋 해시가 남아 있습니다((3) 참조).

## (1) 리뷰 3편

### R1 — 양자화·효율 추론 (Leaning Accept, 6/10, 확신 4)

**기여 요약.** 고정 비트·고정 격자에서 활성 반올림 규칙만 바꿔 계획 성능을 비교합니다. 원소별 오차 분포를 같게 두고 공유·지속만 바꾼 변형(Table 1)으로 원인을 가르고, 비트 한계(k*), 정수 커널 지연, FP4를 보고합니다.

**강점**
- sh-iter / sh-step / sh-ep, det+shift / det+shift-fixed가 같은 오차 분포에서 지속만 다릅니다. 깔끔한 통제입니다.
- §7이 불리한 결과를 그대로 적습니다: W8A4가 미리 변환한 BF16보다 느리고, det+shift 오버헤드는 5.30%·7.23%입니다.
- 정수 커널이 fake-quant를 재현합니다(4/4, per-channel 13/16).
- §6 첫 문단과 FP4 결과(NVFP4 2/12)를 숨기지 않습니다.

**약점**
1. **기본 격자가 약합니다(§3.2, p.4).** Mish 뒤 꼬리가 긴 활성에 per-tensor [min, 99.99%]를 쓰고 0이 격자점이 아닙니다. 보충자료 범위 파일 기준으로 push ckpt 1의 dyn0는 3 bit 간격이 약 1.26인데 평균 활성은 0.13, p99.9는 3.9입니다. 대부분의 활성이 최저 수준으로 떨어진다고 추정됩니다(비율은 재지 않았습니다). 초록의 "up to 100 points"는 이 격자의 2 bit 칸입니다. 본문도 "The best post-training quantizer we tested, mostly per-channel ranges, already removes most of det's damage at 3–4 bits"(§6, p.9)라고 인정합니다.
2. **1-step 오차 결과는 구성상 예상됩니다(§4.2, p.6; 초록).** Table 1이 SR의 분산을 Δ²/6, det를 Δ²/12로 적고 있으니 MSE는 SR이 큰 것이 당연합니다. 그 설명이 없어 발견처럼 읽힙니다.
3. **det+noise는 깨끗한 통제가 아닙니다(§5.2 p.8, §6 p.10).** `A/tmlr_ext/TD_tdmpc2/code/qlib.py:371-373`을 보면 잡음을 양자화 출력에 더하므로 다음 양자화 층과 다음 rollout 단계의 반올림을 다시 무작위화합니다. 논문 자체 수치가 이를 뒷받침합니다: ρp가 TD-MPC2에서 0.05–0.23, DINO-WM에서 0.89–0.95입니다(Table 25의 Wall 0.89). 그런데 본문은 "we did not test whether this difference relates to which cells are repaired, nor did we identify why dither repairs more"라고만 씁니다. (코드와 ρp에서 끌어낸 추론이고, 직접 시험한 것은 아닙니다.)
4. **"unbiased"는 draw에 대한 평균입니다.** 고정 draw 변형(sh-ep, det+shift-fixed)은 에피소드 안에서 unit별 조건부 평균 오차가 남습니다. "which rules out a resolution explanation there"(§5.2, p.8)는 다소 셉니다.
5. **범위가 좁습니다.** 양자화한 것은 dynamics·reward MLP의 은닉 활성 4개뿐이고 encoder·Q·policy는 FP32입니다. 이 부분은 FP32 step의 40–52%입니다(§7).
6. QAT 기준선은 양자화 없이도 망가져 정보가 없습니다(App. G).
7. 기여 4의 "at most 2.01%"와 Table 3 L3의 "max 2.1% (second run 2.3%)"가 어긋나 보입니다.

**요청 변경**
- 필수: 기본 격자의 성격을 명시(§3.2). 분산 설명 한 문장(§4.2). det+noise 재해석(§5.2, §6).
- 권장: 조건부 편향 단서. 2.01% 문구 통일. 초록에 "(at 2 bits)". 선택 실험 X-b, X-c((5) 참조).

### R2 — MBRL·표본 기반 계획 (현 원고 Leaning Reject, 5/10, 확신 4)

**기여 요약.** 반복하는 계획기가 반복 사이에 고정된 반올림 오차장을 이용한다는 것을 TD-MPC2에서 개입 실험으로 보이고, DINO-WM에서는 그 성분이 지배적이지 않다고 보고합니다.

**강점**
- Fig. 3b: 반복 1회에서는 sh-iter와 det가 같고 12회에서 0–11 대 43–47입니다. 반복을 통해 이득이 생긴다는 직접 증거입니다.
- 제어 step 안에서만 고정해도 손상 대부분이 돌아옵니다(sh-step 34.9 대 sh-iter 4.0).
- 317M 복제와 정책 단독 기준선(X6)을 초록에까지 넣었습니다.
- 이론 부록이 틀린 예측을 열거합니다.

**약점**
1. **초록 "on DMC the low-bit dithered planners still beat the policy"는 오독을 부릅니다.** Table 22에서 DMC의 G는 PT 0·0·0·1, PC 1·0·0·1입니다. E12 칸을 다시 보면 det도 자기 k*에서 정책을 이깁니다:

   | 과제 (PC) | det의 D | 정책의 D |
   |---|---|---|
   | walker, 3 bit | 6.3% [5.0, 7.7] | 56.3% |
   | quadruped, 3 bit | 1.6% [0.8, 2.6] | 13.7% |
   | cheetah, 5 bit | 8.4% [5.3, 12.1] | 53.5% |

   finger는 구간이 겹칩니다. dither 고유의 결과가 아닙니다.
2. **Meta-World는 계획이 필요 없는 벤치입니다.** 정책 단독 D가 7/8과제에서 7.8 이하입니다(Table 34). 계획이 도움 되는 유일한 과제 pick-place(정책 +23.3)에서도 최저 비트 dither는 정책과 "same"입니다. 계획이 중요한 DMC에서는 기본 설정에서 2/4만 고쳐지고, walker는 새 seed에서도 다시 해롭습니다(77.2% 대 59.3%, Table 26).
3. **DINO-WM 프로토콜 설명이 없습니다(§3.3).** 모든 과제를 open-loop CEM으로 돌려 무양자화 성공률이 83/61/85%(Table 8), 새 분할 73/49/79%(Table 31)입니다. 공개 DINO-WM은 재계획하는 MPC로 0.90/0.96/0.98입니다(DINO-WM arXiv HTML 판에서 오늘 확인). 차이가 설명되지 않았습니다.
4. **App. A D1(p.19)의 문구가 내부 기록과 다릅니다.** "the split used in E0 to reproduce the published DINO-WM results"라고 썼지만 `A/main/reports/E0_dinowm.md`의 재현은 자기 파일럿(R2-13) 재현입니다.
5. **Wall은 신뢰하기 어려운 시험대입니다.** 기준선이 49–62%이고 4 bit det가 무양자화보다 14–22점 높습니다(§5.2). 그런데 이중 해리의 DINO-WM 쪽과 계획기 교차에서 정보가 있는 유일한 칸입니다(MPPI식 PointMaze는 D_det +10 [1, 19]뿐이고 PushT는 안 돌림).
6. **초록 "this pattern followed the model family when we swapped the planners' update rules"가 증거보다 셉니다.** TD-MPC2 3과제, DINO-WM 2과제입니다. CEM식 교차는 세 가지(균등 가중, 정책 궤적 제거, std 바닥)를 한꺼번에 바꿨습니다.
7. **§5.1 "The derivation's selection and drift terms matched 922,368 simulated MPPI and CEM runs."** Table 14에서 지속장의 drift가 음수인 칸은 CEM 83%, MPPI 64%입니다. "matched"는 과합니다.
8. 제목이 일반 명제입니다(아래 제목 한 줄).

**요청 변경**
- 필수: 초록·§6의 DMC 문장. DINO-WM 프로토콜 문장과 D1 정정. 교차 문장에 범위 명시.
- 권장: MPC 프로토콜 실험 X-a. "matched" 수정. E16에서 정책 궤적 없이도 push·pick-place가 고쳐진다(red 0.78, 0.80)는 점을 명시.

### R3 — 실험 방법론·통계 (Leaning Accept, 7/10, 확신 4)

**기여 요약.** 짝 지은 평가, 새 seed 복제, 실패한 검정까지 모두 싣는 기록, 사후 표시를 갖춘 대규모 통제 실험입니다.

**강점**
- Table 3–4와 App. J가 실패한 가설(H2a, H2b, H4b, B3, K1, K3 등)을 빠짐없이 싣습니다.
- 보충자료만으로 숫자가 재생성됩니다. 직접 실행해 numbers.tex 578개 값이 바이트 단위로 같고 `--strict`가 통과했습니다.
- episode 표로 E11·E14를 독립 재계산할 수 있었고 모두 일치했습니다.
- R1·R2 재분석, 체크포인트별 표(Table 21), 구간 미보정(two-stage) 공개가 있습니다.

**약점**
1. **§4.2(p.6)의 수치 범위가 틀립니다.** "Since ρp is 1.00 for every fixed variant and 0.00–0.24 for every redrawn dither"는 "cells re-run on new seeds and splits" 문맥인데, 0.00–0.24는 TD-MPC2 파일만의 값입니다. DINO-WM 새 분할에서는 sh-iter·det+shift의 ρp가 0.17–0.46이고, Table 25의 Wall도 0.34/0.31입니다.
2. **"repairs"가 점추정 기준입니다.** 예: window-open 4 bit는 red 0.53 [0.00, 0.82]인데 "19/19"에 들어갑니다. 제 재계산으로는 red 구간 하한이 0.5 이상인 칸이 E1 MW 18/19, 새 seed MW 6/8입니다(pick-place [0.30, 0.77], window-open [−0.42, 0.67]).
3. **퇴화 구간.** [0.0, 0.0], [1.00, 1.00], "+100 [100, 100]"(Tables 7, 9, 15, 23, 27)은 불확실성을 과소 표시합니다.
4. **계열 대비가 일부 "0을 포함하느냐"에 기댑니다.** 본문도 "The families differ in whether the interval excludes zero, but cell by cell the sizes overlap"(§5.3, p.9)라고 인정하지만 초록은 그대로입니다.
5. **체크포인트 조건부 추론입니다**(5M 3개, 317M 1개, DINO-WM 과제당 1개). 제 계산으로 Δpers는 11/12칸에서 세 체크포인트 모두 양수입니다(peg-insert 한 체크포인트만 0.0). 이 사실을 실으면 좋습니다.
6. α(0.025/0.05)와 Holm 묶음이 단계마다 다릅니다(§3.4).
7. H4a는 표 크기 4점의 Spearman 문턱 −0.8이 10⁻⁹ 허용오차로 갈렸습니다. "holds (borderline)"보다 "inconclusive"가 맞습니다.
8. 사전 고정은 자기 기록입니다. 설계 문서가 한국어이고, 편집본은 해시가 재현되지 않으며, D1–D3은 LLM 에이전트가 결정했습니다. 투명성 기록으로는 좋지만 확증 근거로는 약합니다.
9. 결과 표가 본문에 하나도 없습니다(Table 1만 본문).

**요청 변경**
- 필수: ρp 범위 정정. 구간 기준 개수 병기. 퇴화 구간 단서.
- 권장: 체크포인트 일관성 문장. α·Holm 강건성 한 줄. H4a 표현.

## (2) AE 메타 리뷰

- **합의.** 세 리뷰어 모두 TD-MPC2의 지속성 개입(Fig. 3b, Δpers 12/12, 317M 7/8)은 설득력 있고 독자가 있다고 봅니다. 주장은 대체로 스스로 한정돼 있습니다.
- **이견.** R2는 초록의 DMC·교차 문장과 DINO-WM 프로토콜 누락 때문에 현 상태로는 Claims에 "No"입니다. R1·R3은 문구 수정으로 충분하다고 봅니다.
- **결정.** 새 실험 없이 문구로 닫을 수 있는 틈이 대부분이라 accept with minor revision이 가장 유력합니다.
- **AE가 조건으로 달 것.** 초록 두 문장 수정, DINO-WM 프로토콜과 D1 정정, det+noise·ρp 서술 정리, 본문 밀도 완화.
- **reject로 가는 길.** DINO-WM 증거가 "convincing"하지 않다고 두 명이 볼 때. 밀도 때문에 "clear"가 아니라고 볼 때. 보충자료 로그를 읽은 심사자가 LLM 관여 범위와 첫 쪽 각주의 일치를 문제 삼을 때.
- **제목.** "Planners Exploit Deterministic Error"는 일반 명제로 읽히는데 본문은 TD-MPC2로 한정합니다. R2가 지적할 수 있으나 위험은 낮음~중간입니다.

## (3) 데스크 리젝 점검

| 항목 | 결과 | 근거 |
|---|---|---|
| stylefile | 문제 없음(공식본 대조는 못 함) | `\usepackage{tmlr}` 옵션 없음. tmlr.sty SHA-256 `816214ff…0002`는 B 사본과 같음. 내려받기를 하지 않아 공식 저장소와 바이트 대조는 못 했습니다. 투고 직전 최신본과 대조하세요. |
| 쪽수 | 통과 | 본문이 12쪽의 74%에서 끝남(직접 측정). 전체 43쪽. Broader Impact 포함. |
| PDF 익명성 | 통과 | 메타데이터에 저자 없음. 감사·저장소 링크 없음. 미정의 참조·TODO 0. |
| 보충자료 크기·형식 | 통과 | zip 6.8 MB(한도 100 MB), 858항목, macOS 메타 없음. |
| 보충자료 이름·경로·호스트 | 통과 | 이름·이메일·`/Users`·`/home`·호스트명 없음. gz 116개 내부까지 확인(host는 `[server]`). 비밀키 패턴 없음. |
| 보충자료 잔여 단서 | **고칠 것** | 아래 목록 |
| 필수 문구 | 통과, 확인 필요 1 | Broader Impact 있음. TMLR 공개 지침(오늘 확인)에는 LLM 고지 의무나 첫 쪽 위치 규정을 찾지 못했습니다. 정책은 "LLMs may be used as general-purpose assistive tools"이고 저자 전적 책임입니다. OpenReview 양식의 문항은 확인하지 못했습니다. |
| 재현 | 통과 | zip을 scratch에 풀어 README 명령 3개 실행: numbers.tex 바이트 동일, `--strict` 통과. 익명 소스 tectonic 빌드는 43쪽이고 main.pdf와 본문 텍스트가 같습니다. |

**보충자료에 남은 것.** README는 "other venues of this work and the preprint identifier were replaced … lines … were emptied"라고 하지만 실제로는 남아 있습니다.
- **다른 투고처·일정:** `supp/main/reports/FINAL_A.md:888-895`, `supp/rss_ext/DESIGN_ext.md:1,5`, `supp/rss_ext/LOG.md:57`, `supp/EXPERIMENT_DESIGN.md:62-64, 497-502`.
- **프리프린트 날짜:** `supp/tmlr_ext/LOG.md:172` "우리 v1(11-09)".
- **자기 저장소 커밋 해시:**
  - `supp/rss_ext/LOG.md:27`의 34ab689.
  - `supp/tmlr_ext/LOG.md:13,14,60,263,264`의 0e0a61d, 3dad73e, 9e1c9923.
  - `supp/tmlr_ext/TD_tdmpc2/` 아래 `code/{analyze,plan}_x5.py`, `code/{analyze,plan}_x6.py`, `code/x5_section.py`, `code/x6_section.py`, `REPORT.md:158,234`, `LOG.md:55,68`, `results/analysis_x6/x6_tables.md:3`의 1d73e10, 24a59a00.
  - 저장소가 공개라면(메모리 기록 기준, 이번에 확인하지 않음) 검색 경로가 됩니다.
- **원고 제작 과정:** `supp/tmlr_ext/LOG.md:157-269`. 예: 164행 "원고 작업자가 main.tex에 합치고, 12쪽에 맞추고, 빌드한다", 165–168행 "AI티 나는 문장도 검증받아 고친다", 61·85행 "사용자 답(원문): 너의 추천대로…". 실험 검증과 무관하고, 첫 쪽 각주의 "drafting and editing portions of the manuscript", "verified all reported numbers"와 나란히 읽힙니다.
- **기타:** 한국어 문서 108개(README에 고지됨, 규정 위반은 아님), 폴더명 `rss_ext`와 `runpod_rss`, 자매 과제 "B" 언급(`supp/tmlr_ext/LOG.md:96`).

## (4) 수치 대조

모두 일치했습니다. 서술 범위가 어긋난 곳은 두 군데입니다.

| # | 본문 수치 | 확인한 파일 | 결과 |
|---|---|---|---|
| 1 | 초록 "up to 100 points" | `A/main/results/E1_tdmpc2/e1_tdmpc2_cells.csv` (button-press k=2, D=1.0) | 일치 |
| 2 | 서론 29/36, 그중 21 | 같은 파일 + `A/main/results/E1/e1_dinowm_cells.csv` (TD 24/31 중 17, DW 5/5 중 4) | 일치 |
| 3 | §4.1 19/19, 중앙값 95%, 4.0 / 50.1 / 44.6 | 같은 파일 | 일치 |
| 4 | 새 seed 12/12 손상, 13.3–45.6점, 40.0–83.8% | `A/tmlr_ext/TD_tdmpc2/results/analysis/td_episodes.csv`, 자체 bootstrap + Holm | 일치 |
| 5 | Δpers 하한 > 0이 12/12; sh-ep 1/8, det+shift-fixed 0/8; det+noise 8/8 | 같은 파일, 자체 bootstrap | 일치 |
| 6 | 317M 8/8, 지속 비용 7/8, push +3.3 [0.0, 7.8], sweep-into 2.61 | `A/tmlr_ext/MT_317m/results/analysis/e14_episodes.csv`, 자체 bootstrap | 일치 |
| 7 | 2.01%, 5.30%, 7.23% | `A/tmlr_ext/K_kernel/results/raw/k.1/analysis_lat/e17_lat_summary.json` | 일치 |
| 8 | NVFP4 2/12, MXFP4 4/12 | `A/rss_ext/E9_fp4/results/e9_cells.csv` | 일치 |
| 9 | G ≥ 1이 7/8·7/8, push k* 5→3 | `A/tmlr_ext/TD_tdmpc2/results/analysis/td_summary.json` | 일치 |
| 10 | 정책 D ≤ 7.8 (7/8), pick-place +23.3 [14.4, 32.2], DMC 6.3–56.3%, 3/4 | `A/tmlr_ext/TD_tdmpc2/results/analysis_x6/x6_summary.json` | 일치 |
| 11 | E16 +47.8 / +42.2 / +18.7, cheetah red 0.14; E15 Table 26 | td_episodes.csv (ptag cem, it12, it24) | 일치 |
| 12 | PushT 4 bit +4 [−7, 15], Holm p 0.274 / 0.038 / 1e-4 | `A/tmlr_ext/DW_dinowm/results/analysis/analysis_dw.json` | 일치 |
| 13 | §4.2 ρp "0.00–0.24 for every redrawn dither" | 같은 파일 + Table 25 | **범위 불일치** (DINO-WM 0.17–0.46) |
| 14 | 기여 4 "at most 2.01%" | Table 3 L3 "max 2.1% (second run 2.3%)" | 값은 맞으나 다른 실행과 어긋나 보임 |

## (5) 통합 수정 목록 (결정에 미칠 효과 순)

본문에 새 숫자를 넣으면 `\nv{}` 키를 만들거나 `build/design_literals.txt`에 등록해야 `--strict`가 통과합니다. F6–F9의 추가분(약 9행)은 F11의 삭제로 상쇄해 12쪽을 지키세요.

**F1 [must] 보충자료 정리 — AE, 전원. 실험 불필요.**
- 위치: (3)의 목록.
- 변경:
  - `supp/tmlr_ext/LOG.md`에서 실험과 무관한 원고 제작·push 항목(59–60, 93–96, 157–269행)과 172행을 뺍니다.
  - 커밋 해시는 `[commit]`으로 바꿉니다.
  - 투고처·일정 줄은 README에 적은 대로 비웁니다.
  - README의 Anonymization 문단을 실제 처리와 맞춥니다.
- 각주(main.tex:100)는 저자 결정 사항이지만, 로그가 보여 주는 관여 범위와 글자 그대로 맞아야 합니다. 고지를 줄이라는 뜻이 아니라 사실과 맞추라는 뜻입니다.
- 효과: 심사 전 중단과 신뢰 문제를 막습니다.

**F2 [must] 초록·§6의 DMC 문장 — R2. 실험 불필요.**
- main.tex:87
  - old: `on DMC the low-bit dithered planners still beat the policy.`
  - new: `on DMC, where the policy alone loses up to 56\% of return, low-bit planners beat the policy under either rounding rule and dither gains at most one bit.`
- main.tex:270
  - old: `and the lowest-bit dither planner (per channel, 3–4 bits) beats the policy in 3 of 4 tasks.`
  - new: `the lowest-bit dither planner (per channel, 3–4 bits) beats the policy in 3 of 4 tasks, and so does round-to-nearest at its own smallest usable bit width (per channel, 3–5 bits).`
- det 쪽 3/4는 제가 `td_e12_cells.csv`와 `x6_summary.json`으로 본 값입니다. 저자 스크립트로 다시 확인하세요.
- 효과: R2의 Claims No → Yes, 5 → 6.

**F3 [must] DINO-WM 프로토콜과 D1 — R2. 실험 불필요.**
- main.tex:176, "…10 otherwise)." 뒤에 추가:
  - new: `We plan once per episode without replanning, whereas the published DINO-WM results use receding-horizon replanning; our unquantized success (83\%, 61\% and 85\% on PushT, Wall and PointMaze) is therefore below the published 90\%, 96\% and 98\%.`
  - 공개 수치와 프로토콜은 저자가 원 논문으로 재확인하세요.
- `A/paper/tmlr/appendix/app_A.tex:92`
  - old: `the split used in E0 to reproduce the published DINO-WM results`
  - new: `the split used in E0 to reproduce our pilot results`
- 효과: R2의 Claims, 쟁점 1.

**F4 [must] 계획기 교차 문장 — R2, R3. 실험 불필요.**
- main.tex:87
  - old: `and this pattern followed the model family when we swapped the planners' update rules.`
  - new: `and the pattern did not change when we swapped the planners' update rules on the tasks where we tested this.`
- main.tex:109
  - old: `The pattern held when we swapped the planners' update rules.`
  - new: `The pattern held when we swapped the planners' update rules (three TD-MPC2 tasks; Wall and PointMaze for DINO-WM, without PushT).`
- 승인된 한 줄 주장이 바뀌므로 저자 승인이 필요합니다.
- 효과: 쟁점 1.

**F5 [must] ρp 범위 — R3. 실험 불필요.**
- main.tex:210
  - old: `Since $\rho_p$ is 1.00 for every fixed variant and 0.00–0.24 for every redrawn dither, the statistic nearly restates the manipulated variable`
  - new: `Since $\rho_p$ is 1.00 for every fixed variant and, in TD-MPC2, 0.00–0.24 for every redrawn dither (0.17–0.46 in DINO-WM), the statistic nearly restates the manipulated variable in TD-MPC2`
- 효과: 사실 오류 제거.

**F6 [should] det+noise 재해석 — R1. 실험 불필요(선택 X-b).**
- main.tex:238
  - old: `The variant det+noise keeps det's rounding and adds fresh shared noise; we predicted …`
  - new: `The variant det+noise keeps det's rounding and adds fresh shared noise to each quantized output. In TD-MPC2 this noise enters the next quantized layer and the next rollout step, so it can randomize later rounding decisions. We had predicted …`
- main.tex:279
  - old: `we did not test whether this difference relates to which cells are repaired, nor did we identify why dither repairs more.`
  - new: `In TD-MPC2 the low share means that det+noise no longer keeps det's error pattern, whereas in DINO-WM its share stays at 0.89–0.95 (Table 25); it repairs the Meta-World cells and neither DINO-WM cell, as does sh-iter+b (5 of 8 and 0 of 2). We did not test this reading directly.`
- 0.89–0.95는 analysis_dw.json의 세 칸 값입니다. Table 25에는 Wall만 있으니 표에 PushT·PointMaze를 추가하거나 "0.89 on Wall"로 쓰세요.
- 효과: "설명 못 한 이상 결과"가 두 성분 서술의 근거로 바뀝니다. R1 6 → 7, 쟁점 3.

**F7 [should] 분산 설명 — R1, R3. 실험 불필요.**
- main.tex:208, "(Figure 2c)." 뒤에 추가:
  - new: `This is what the error variances of Table 1 predict ($\Delta^2/6$ against $\Delta^2/12$): a squared error measures the size of the error, not whether it stays the same across planner iterations.`

**F8 [should] 기본 격자 명시 — R1. 작은 재측정 필요.**
- main.tex:151, "with no error-minimizing calibration" 뒤에 추가:
  - new: `This default grid is deliberately plain: zero is not a grid point, and at 3 bits its step is several times the typical activation, so round-to-nearest maps most activations to the lowest level; Section 6 repeats the comparison on per-channel ranges.`
- "most"를 쓰려면 최저 구간 비율을 보정 활성에서 한 번 재야 합니다. 대략 0.1 GPU-시간 정도로 봅니다.
- 효과: R1 약점 1을 선제 차단합니다.

**F9 [should] 구간 기준 개수와 단서 — R3. 실험 불필요.**
- main.tex:195, "(a median of 95\%)" 뒤: `; the interval of red lies above 0.5 in 18`
- main.tex:197, "(all but window-open" 뒤: `; the interval of red lies above 0.5 in 6 of the 8`
- main.tex:180에 추가: `A bootstrap interval collapses to a point when all pairs agree and then understates the uncertainty.`
- main.tex:240, "in all 12 damaged cells (…)" 뒤: `and is positive in every checkpoint in 11 of them`
- main.tex:281에 walker의 새 seed 값(77.2% 대 59.3%)을 적습니다.
- 18, 6, 11은 제 bootstrap 값입니다. 저자의 RandomState(0)로 다시 계산하세요.
- 효과: R3 7 → Accept 쪽.

**F10 [should] "matched" — R2, R3. 실험 불필요.**
- main.tex:227
  - old: `The derivation's selection and drift terms matched 922,368 simulated MPPI and CEM runs.`
  - new: `In 922,368 simulated MPPI and CEM runs, the derivation's selection term and depth of exploitation agreed with the simulation (regression slopes 0.92–1.05), and the drift under a persistent field was negative in 83\% (CEM) and 64\% (MPPI) of the settings (Table 14).`

**F11 [should] 밀도 — 전원. 실험 불필요.**
- main.tex:210의 Dg1 문단을 한 문장으로 줄이고 나머지는 App. I로 옮깁니다.
- main.tex:242의 Wall 기준선 세부(마지막 네 문장)를 App. I로 옮깁니다.
- 여유가 있으면 Table 19를 §5.2 본문으로 올리고 중복 수치를 지웁니다.
- 제 측정으로 본문 평균 문장 길이는 30단어를 넘고, §4–7에는 괄호가 137개 있습니다.

**F12 [should] 투고 직전 점검.**
- TMLR stylefile 최신본과 대조합니다.
- OpenReview 양식의 LLM 문항을 확인합니다.
- "Submitted to ICLR 2027" 익명 인용 6편의 상태를 갱신합니다.

**F13 [could] 작은 문구.**
- main.tex:111: `at most 2.01\%` → `about 2\% (at most 2.3\% in any of our runs)`
- 초록 "up to 100 points of success" 뒤에 `(at 2 bits)`
- Table 3의 H4a: `holds (borderline)` → `inconclusive`
- §3.4에 "모든 칸을 한 묶음으로 Holm 0.025를 적용해도 손상 칸이 같다"는 강건성 한 줄(E1은 재계산 필요)
- App. A에 고정 시각이 자기 기록임을 한 문장
- main.tex:240의 `rules out` → `argues against`

**F14 [could] 리뷰어가 고집할 수 있는 실험 (GPU-시간은 제 어림).**
- **X-a (R2):** DINO-WM을 공개 프로토콜인 재계획 MPC로 Wall·PointMaze에서 none / det / sh-iter / sh-plan / det+shift-fixed × 100쌍. 4090 기준 약 7 GPU-시간, PushT 3 bit까지 넣으면 약 12 더. E0 보고의 episode당 시간에 재계획 5회를 가정했습니다. 쟁점 1에 가장 직접적입니다.
- **X-b (R1):** 잡음을 예측 return에만 더하는 통제를 MW 8칸에서. 1 GPU-시간 미만. F6의 추론을 증거로 바꿉니다.
- **X-c (R1, R3):** 317M에서 det+shift-fixed와 det+noise(Fig. 4의 "not run"). 약 4 GPU-시간.
- **X-d (R2):** 계획이 중요한 과제를 더(DMC 4과제 추가, PC 3–5 bit, det / dither / 정책). 약 5–10 GPU-시간.

## (6) 바꾸지 말아야 할 것

- 같은 오차 분포에서 지속만 바꾼 통제(Table 1의 sh-iter / sh-step / sh-ep, det+shift / det+shift-fixed)와 Fig. 3b의 반복 수 sweep. 논문의 핵심 증거입니다.
- App. B의 전체 검정표, App. J의 "What Did Not Hold"와 "Claims we do not make". R3이 가장 높이 삽니다.
- 새 seed·새 분할 복제와 317M 복제, PushT 4 bit 미재현의 공개, post hoc 표시.
- 초록·결론의 정책 단독 단서와 §6 첫 문단(강한 PTQ 위에서는 이득이 경계선), NVFP4 2/12.
- §7의 불리한 지연 결과(BF16 우위, K1·K3 불성립)와 Limitations 절.
- §5.3의 "cell by cell the sizes overlap", "we do not claim that the model causes the pattern" 같은 자기 한정 문장.
- `numbers.tex` 파이프라인과 보충자료의 재생성 명령, TD-MPC2 episode 표. 실제로 돌아갑니다.
- Fig. 2, 4, 5는 그대로 읽힙니다. 빌드도 깨끗합니다(미정의 참조·overfull 0).
