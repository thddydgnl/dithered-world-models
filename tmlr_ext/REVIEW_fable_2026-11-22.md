<!-- 검증자(Claude Fable 5.2, 추론 노력 max)의 최종 보고 전문. 2026-11-22 사용자 요청(검증 항목 포함, LOG 08:00Z)으로 조정자가 맡겼고, 조정자가 그대로 옮겼다. 원고는 로컬에만 있다. -->

# 논문 A TMLR 원고 심사 보고 (읽기 전용)

- 대상: `/Users/song-yonghwi/Documents/physical ai - Neural Networks/A_dithered_world_models/paper/tmlr/`의 `main.tex`, `appendix/app_A.tex`~`app_K.tex`, `main.pdf`(39쪽). 아래 `main.tex:NN`은 이 폴더 기준 줄 번호다.
- 기준 문서: `/Users/song-yonghwi/Documents/physical ai - Neural Networks/A_dithered_world_models/tmlr_ext/`의 `CLAIMS_tmlr.md` v2.1, `RESULTS_tmlr.md`, 묶음 `REPORT.md`, `PLAN_tmlr.md` §2.1·§5.
- 원고·git·클라우드는 건드리지 않았고 빌드도 돌리지 않았다. 임시 파일은 scratchpad에만 있다.
- 표본 대조: `main/results/E1_tdmpc2/e1_tdmpc2_cells.csv`에서 E1 평균 손상(44.6/29.2/34.9/50.1/4.0/4.0), 24/31, 22/31, 중앙값 95%를 다시 셌고 모두 원고와 같다. 그 밖의 수치는 대조하지 않았다.

## (0) 판정

| 항목 | 내용 |
|---|---|
| 예상 결과(지금 그대로) | 소폭 수정 뒤 채택(minor revision). TMLR에는 major revision 판정이 없어 "크게 고쳐야 한다"는 평은 거절로 간다(PLAN §2.1) |
| 내 추정(의견) | 그대로 내면 채택 55~60%, 거절 40~45%. 아래 must를 반영하면 70% 이상 |
| 신뢰도 | 중간 |
| 위험 1 | 전달 밀도. 변형 9개, 단계 라벨 10종이 본문에 41번, "post hoc" 16번. 본문 평균 문장 31단어, 50단어 이상이 약 30문장. 6.1절은 35줄짜리 한 문단 |
| 위험 2 | DINO-WM 쪽 근거가 얇다(손상 칸 2개, 하나는 사후, Wall 기준선 불안정). 그런데 초록·서론·결론은 계열 수준 문장으로 쓴다 |
| 위험 3 | 심사자가 원고의 표만으로 반박할 수 있는 머리 문장이 있다(M1, M2, M5) |

- 기준 1(주장–근거)
  - TD-MPC2의 지속 주장은 양성 대조, Δ_pers 12/12, 반복 수 용량-반응, 317M 7/8로 탄탄하다.
  - 부정 결과와 한계를 숨기지 않은 점은 TMLR에서 강점이다.
  - 약한 곳은 DINO-WM 일반화와 M1·M2·M5의 문장이다.
- 기준 2(관심·전달)
  - 관심은 충분하다(계획용 world model 양자화, 학습 없는 처방).
  - 전달이 가장 큰 약점이다.
- desk reject를 부를 수 있는 것
  - 보충자료 익명성(M4). 아직 만들어지지 않았다.
  - 보충자료 100MB 초과. 원자료 합이 약 272MB다(`main` 73, `rss_ext` 21, `tmlr_ext` 178).
  - style 파일은 B 사본과 sha256이 같고 PDF 메타데이터도 비어 있어 문제없다.
- 부정 심사를 부를 수 있는 것
  - 정의되지 않은 핵심 양(M3)
  - 정책만 쓴 기준선의 부재(S6)
  - "strong quantizers"라는 말과 표준 PTQ 인용 0건(S7)
  - 검증할 수 없는 사전 고정 문구의 반복(S13)
  - LLM 각주와 부록 A의 "worker" 표현의 어긋남(S14)

## (1) 고칠 것

### Must

- **M1. 초록 둘째 문장과 서론 둘째 문단이 표 7과 어긋난다.**
  - 100점 칸(button-press-topdown 2-bit)은 det 오차 2.02, sh-iter 0.94로 det가 더 크다. 그림 2c의 오른쪽 위 점이다.
  - 오차가 더 작은 칸의 최대 손상은 94.4점(pick-place 2-bit)이다.
  - 29칸 중 sh-iter가 손상의 절반 이상을 없앤 칸은 21칸뿐이다(TD-MPC2 17, DINO-WM 4, 내 재계산). walker 3칸 등 8칸은 "plans well"이 아니다.
  - `main.tex:86`: `With round-to-nearest activation quantization, planning can lose up to \nv{e1.maxdet} points of success while the world model's latent one-step error stays no larger than under a dither that plans well.` → `With round-to-nearest activation quantization, planning loses up to \nv{e1.maxdet} points of success, and in most affected settings the world model's latent one-step error is no larger than under stochastic rounding (dither).`
  - `main.tex:100`: `…points of success, while in \nv{e1.h1.n} of \nv{e1.h1.of} affected cells (task $\times$ bit width) the latent one-step error of the world model is no larger than under a stochastic rounding scheme that plans well (Section~\ref{sec:break}).` → `…points of success. In \nv{e1.h1.n} of \nv{e1.h1.of} damaged cells (task $\times$ bit width) the latent one-step error of the world model is no larger than under shared dither, and in 21 of these \nv{e1.h1.n} cells that dither removes at least half of the damage (Section~\ref{sec:break}).` (21은 새 `\nv` key로)
- **M2. E10·E9의 잠재 오차는 "자기 계획 상태"가 아니라 비양자화 계획기의 held 상태에서 잰 값이다**(`rss_ext/E10_mt317m/REPORT.md` 63·159행, `app_C.tex:57`). PLAN W12가 이미 경고한 점이다.
  - `main.tex:198`: `(c) \dtm's extra damage against the ratio of the two latent one-step errors, on each variant's own planning states, for the cells of (a); in the shaded region…` → `(c) \dtm's extra damage, $D_{\dtm}-D_{\shiter}$, against the ratio of the two latent one-step errors for the cells of (a). The E1 errors are measured on each variant's own planning states and the E10 errors on held-out states of the unquantized planner. In the shaded region…`
  - `main.tex:204`: `At 317M, \dtm's latent error is…` → `At 317M, on held-out states of the unquantized planner, \dtm's latent error is…`
  - `app_D.tex:3`: `The latent one-step error is measured on each variant's own planning states, as in Section~\ref{sec:offline}; Appendix~\ref{app:new} repeats it on common held-out states.` → `In E1 the latent one-step error is measured on each variant's own planning states, as in Section~\ref{sec:offline}, and in E9 and E10 on held-out states of the unquantized planner. Appendix~\ref{app:new} repeats the E1 measurement on common held-out states.`
- **M3. 핵심 양 넷이 정의되지 않았거나 부호가 틀렸다.**
  - e(a)의 부호. 설계의 정의는 비용 오차 e = V₀ − V_q다(`main/DESIGN_main.md` 63행). 본문 정의대로 읽으면 "Negative drift … favors"와 표 11의 음수가 거꾸로다.
    - `main.tex:180`: `Let $e(a)$ be the error of the quantized model's predicted return for candidate $a$ relative to the unquantized model.` → `Let $e(a)$ be the cost error of candidate $a$: the return that the unquantized model predicts for $a$ minus the return that the quantized model predicts (for DINO-WM, the quantized minus the unquantized planning cost), so that $e(a)<0$ when quantization makes $a$ look better.`
    - `app_E.tex:3`의 첫 정의도 같게 고친다.
  - "latent one-step error". 식이 본문과 부록 어디에도 없다(0.22 대 0.39, 표 7의 2.99가 무엇의 비인지 알 수 없다).
    - `main.tex:180` 끝에 추가: `The latent $n$-step error of a variant is the relative error of its $n$-step latent prediction against the unquantized model's on the first-iteration candidates, averaged over the episodes of a cell and, for randomized variants, over three draws (Appendix~\ref{app:details}).`
    - 정규화 식은 `app_C.tex`에 넣는다. 나는 코드의 정규화까지 확인하지 않았다.
  - "persistent share"가 두 양을 가리킨다. 3.5절의 ρ_p(det에서 1.00)와 6.3절·그림 6의 S = drift/OEI(det에서 0.33~0.73)다. S는 정의가 없다.
    - `main.tex:180`에 추가: `For \dtm we also report the drift share $S=\mathrm{drift}/\mathrm{OEI}$.`
    - `main.tex:280`: `The persistent share of \dtm's favorable error is lower on DMC (…) than on Meta-World (…), and across the 12 TD-MPC2 tasks it predicts the damage removed (Spearman…` → `The drift share $S$ of \dtm is lower on DMC (\nv{e3.share.dmc}) than on Meta-World (\nv{e3.share.mw}), and across the 12 TD-MPC2 tasks it is rank-correlated with the damage that \shiter removes (Spearman…`
    - `main.tex:285`: `(a) Persistent share of \dtm's favorable error ($S$, E3)` → `(a) Drift share $S=\mathrm{drift}/\mathrm{OEI}$ of \dtm (E3)`
    - 그림 6의 x축 이름(`build/make_figs.py`의 `fig_iterations`)과 표 3의 DMC rule 행(`build/make_tables.py`)도 같이 바꾼다.
  - 유의수준이 없다.
    - `main.tex:176`: `if the \dtm drop is real under a one-sided paired bootstrap ($10^4$ resamples) with Holm correction over…` → `if \dtm's drop is greater than zero in a one-sided paired bootstrap test ($10^4$ resamples, $\alpha=0.05$, Holm correction over…`
- **M4. 보충자료와 부록 K의 서술이 맞지 않는다.**
  - `build/verify_numbers.py` 31행이 `main.tex`를 읽는데, 그 파일은 저자 블록과 저장소 URL이 주석으로 남은 비익명 파일이다(파일 머리말에 "Never put it in the supplement"). 450행은 `LOG.md` 세 개를 읽는다.
  - `OUTLINE.md` §4 기록으로는 결과 파일 6개에 `/home/…` 경로가 남아 있다.
  - 설계 문서·로그·보고서는 한국어인데 부록 K에 그 말이 없다.
  - 할 일: 익명 사본을 만들어 빈 폴더에서 세 명령을 실제로 돌려 본다. 압축 크기가 100MB 이하인지 확인한다.
  - `app_K.tex:8`의 `…without a GPU.` 뒤에 추가: `The supplement includes an anonymized copy of the manuscript source, which \texttt{verify\_numbers.py} reads. Design documents, logs and stage reports are in Korean; a README maps every table and figure to its result files.`
- **M5. 5.2절과 기여 2의 두 문장.**
  - sh-ep의 오차 분산은 Δ²/6으로 det의 두 배다(표 1). "det-like size"가 틀렸다.
  - `main.tex:232`: `Random errors with \dtm-like size but held fixed across iterations (\shep, \detshiftfix) repair only … Meta-World cells, which rules out a resolution explanation, and $D_{\shep}-D_{\shiter}$ is positive in every cell (…).` → `Unbiased random errors held fixed across iterations (\shep, with twice the error variance of \dtm, and \detshiftfix, with the same variance) repair only \nv{e11.mw.nfix.shep} and \nv{e11.mw.nfix.detshiftfix} of the \nv{e11.mw.ndam} Meta-World cells. A randomized threshold alone therefore does not repair Meta-World, which rules out a resolution explanation there. $D_{\shep}-D_{\shiter}$ is positive in all 12 cells (\nv{e11.shepminusshiter} points or \%).`
  - 317M은 7/8이다. `main.tex:107`: `costs success in every damaged cell, so…` → `costs success in all \nv{e11.pers.npos} damaged cells at 5M and in \nv{e14.p2.n} of \nv{e14.ndam} tasks at 317M, so…`
- **M6. H2a′의 PushT 몫은 새 자료가 아니라 E1 4-bit 칸으로 판정했다**(`main/DESIGN_main.md` 639행, `build/make_numbers.py` 186행).
  - `app_A.tex:30`: `these were tested on the new seeds of E2 and E3 (marked…` → `these were tested on the new seeds of E2 and E3 and, for PushT, on the 4-bit cell of E1, whose drift had not been computed when the rule was fixed (marked…`
  - `app_B.tex:4`: `tested on data collected afterwards;` → `tested on data collected afterwards (for PushT, on E1 episodes not yet analyzed for that quantity);`
  - `app_J.tex:5`: `tested on new seeds (H2a$'$)` → `tested on new seeds for TD-MPC2 and on the E1 cell for PushT (H2a$'$)`
- **M7. 과제 이름.** 실제 ID는 `mw-button-press-topdown`, `mw-peg-insert-side`다. 원고의 "mw-button-press"는 Meta-World의 다른 과제 이름이다.
  - `main.tex:172`에 추가: `The Meta-World tasks are push, pick-place, door-open, drawer-open, button-press-topdown, peg-insert-side, window-open and sweep-into; we write button-press and peg-insert.`
  - `main.tex:191`(`mw-button-press`, `mw-peg-insert-side`), `:204`(`mw-pick-place`), `:116`(`mw-push`)과 그림 1·3의 판 제목을 표와 같은 짧은 이름으로 맞춘다.
- **M8. 대조군의 예측을 결과보다 먼저 쓴다.** sh-iter+b와 det+noise는 표 1과 서론 한 구절뿐이고, det+noise의 예측은 6.2절에 가서야 나온다. 설계 §1.3의 예측표가 원고에 없다.
  - `main.tex:230`: `E11 tells them apart with the variants of Table~\ref{tab:variants}: a mean bias is removed by \detbc and by any unbiased variant; a resolution loss is removed by any randomized threshold, fixed or redrawn; a persistent field is removed only by redrawing.` → `We call the first two together the systematic error. E11 tells the three apart with the variants of Table~\ref{tab:variants}. If the mean bias causes the damage, \detbc and every unbiased variant repair it, and putting the bias back into fresh dither (\shiterb) undoes the repair. If the resolution loss causes it, every randomized threshold repairs it, fixed (\shep, \detshiftfix) or redrawn, and \detbc does not. If the persistent pattern causes it, only the redrawn variants (\shiter, \detshift) repair it. \detnoise keeps \dtm's rounding and adds fresh shared noise; we predicted that it would repair at most half of the damaged cells in every benchmark.`
  - 본문이 약 3줄 는다. 줄일 곳은 S1과 C3에 적었다. 12쪽을 넘어도 Long 트랙(심사 4주)이 될 뿐이다(PLAN §5.1).

### Should

- **S1. 6절 구조.**
  - 같은 비트의 처방 결과(기여 3의 앞 절반)가 4.1절에만 있다. 6절 첫 줄에 추가: `At the same bit width, \shiter repairs most damaged cells (Section~\ref{sec:glance}, Figure~\ref{fig:damage}). This section asks whether the repair remains on top of better ranges and where it stops.`
  - `main.tex:265`를 넷으로 나눈다: `\textbf{Meta-World.}`, `\textbf{Robustness of the counts.}`, `\textbf{DINO-WM.}`, `\textbf{DMC.}`
  - pick-place·window-open의 bootstrap 확률과 strict 문턱은 표 28·29에 있으니 본문에서 뺀다(약 3줄).
- **S2. 5.3절 재배치와 "공유"의 근거.**
  - `main.tex:247` 문단(반복·후보 수·표 크기)은 TD-MPC2 지속 근거라 5.1절 끝으로 옮긴다. 5.3절은 "Planner swap"만 둔다.
  - 초록의 "shared by all candidates"를 받치는 본문 근거가 4.1절의 숫자 하나(ind 29.2 대 sh-iter 4.0)뿐이다. 5.1절에 추가: `Sharing matters in TD-MPC2: $D_{\ind}-D_{\shiter}$ has a lower bound above zero in \nv{e.e3.crn.n} of \nv{e.e3.ntasks} tasks on new seeds, while in DINO-WM unshared dither plans as well as shared dither (Appendix~\ref{app:mech}).`
  - `main.tex:249` 끝의 `the table-size result above is one hypothesis.`는 무엇의 가설인지 없다. → `One hypothesis is the table size: DINO-WM's fixed dither has one offset per token and unit, and only coarser tables were exploited there.`
- **S3. 단계 라벨과 "mechanism tasks".** `main.tex:172`에 한 문장으로 열쇠를 넣는다: `Stages are numbered E1 (main benchmark), E2--E3 (mechanism sweeps), E5 (post-training baselines), E10 (317M), E11 (component controls on new seeds), E12 (bit sweep), E13 (offline errors on common states), E14 (317M, new seeds), E15 (DMC iterations), E16 (planner swap) and E17 (integer kernel). We call push, pick-place, cheetah-run and PushT at their primary bit widths the mechanism tasks.`
- **S4. Wall에서는 양자화가 기준선보다 낫다.** det 4-bit가 none보다 새 분할에서 22점 [12, 33], E1에서 14점 [4, 24] 높다(`DW_dinowm/REPORT.md` 136·152행). 본문은 "baseline is unstable"만 쓴다.
  - `main.tex:234` 끝에 추가: `On Wall, \dtm at 4 bits also succeeds more often than the unquantized planner (by 22 points on the new split and 14 in E1), so the unquantized planner is not an upper bound there and $\red$ can exceed 1.` (수치는 새 key로)
- **S5. "cost of persistence"가 두 대비를 가리키고, 둘 다 episode 단위 고정이다.**
  - `main.tex:180`: `the cost of persistence at an equal per-entry error distribution.` → `the cost of keeping a draw for an episode at an equal per-entry error distribution. Where \detshiftfix was not run we use the same contrast for dither, $D_{\shep}-D_{\shiter}$ ($D_{\shplan}-D_{\shiter}$ in DINO-WM), and call both the cost of persistence.`
  - "반복 사이"의 지속만 가르는 변형은 sh-step이다. 5.2절에 추가: `Keeping the draw only within a control step (\shstep) already costs most of this: in E1 it loses \nv{e1.mw.mean.shstep} points against \nv{e1.mw.mean.shiter} for \shiter.`
- **S6. 정책만 쓴 기준선(의견, 작은 추가 실험).**
  - per-channel 1-bit에서 dither의 손상이 8점 이하인 과제가 6개다. TD-MPC2에는 양자화하지 않은 정책 궤적 24개와 Q가 있어서, 심사자는 "usable"이 "계획기가 해를 끼치지 않는다"는 뜻인지 물을 것이다.
  - 12과제의 계획 없는 정책 성공률을 표 21 옆에 싣기를 권한다.
  - 이미 있는 반증도 본문에 쓴다. 정책 궤적을 뺀 CEM식에서 push는 det +50.0, sh-iter +11.1, pick-place는 +45.6, +8.9다(`TD_tdmpc2/REPORT.md` 111~113행). 표 23에 D_det·D_sh-iter 열을 더한다.
  - DINO-WM MPPI식의 det 손상(Wall +54, PointMaze +10)도 표 23에 넣는다. 지금은 "지속 비용 없음"이 손상이 없어서인지 알 수 없다.
- **S7. "strong"과 인용(의견).** per-channel 백분위 범위와 bias correction을 "strong quantizers"라 부르면 PTQ 심사자가 반발한다.
  - 6.1절 제목: `Strong quantizers and…` → `Better ranges and…`
  - `main.tex:315`: `On top of strong post-training quantization, the bit gain is smaller…` → `On top of per-channel ranges the bit gain is smaller…; we did not test learned rounding or learned step sizes`
  - `main.tex:147`의 `it is the standard bias correction of post-training quantization`에 인용이 없다.
  - 넣을 문헌: Nagel et al. 2019(bias correction), Nagel et al. 2020(AdaRound), Xiao et al. 2023(SmoothQuant), Lin et al. 2024(AWQ), Bengio et al. 2013(STE), Holm 1979, Oquab et al. 2024(DINOv2), Fu et al. 2020(D4RL), Tillet et al. 2019(Triton). LeWM과 Fast-WAM도 인용이 없다. 항목은 기억에서 적었으니 `REFERENCES_CHECK.md` 방식으로 원 출처와 대조한다.
- **S8. 범위를 넘는 문장들.** (5)의 O1~O12.
- **S9. 그림.** (3)의 F1~F6.
- **S10. 보상 오차.** E13에서 다시 정의한 1-step 보상 오차는 det가 9/12칸에서 더 크다(표 24). CLAIMS_rss §3은 "어떤 1-step 오차로도 안 보인다"를 금한다.
  - `main.tex:204` 끝에 추가: `The one-step reward error, redefined on these states, is larger for \dtm in \nv{e.e13.rew.n} of \nv{e.e13.ncells} cells (Appendix~\ref{app:mech}).`
- **S11. 체크포인트.** 90짝은 체크포인트 3개 × seed 30개인데 짝 단위로만 재표본한다.
  - `main.tex:176`에 추가: `The bootstrap resamples pairs, so intervals are conditional on the checkpoints.`
  - `main.tex:172`: `independent replications` → `replications on independent episodes with the same checkpoints`
  - 체크포인트별 D_det와 red 표를 부록 I에 더하기를 권한다.
- **S12. 표 21.** Wall G = 2, PushT PC G = 2가 단서 없이 실려 있다. CLAIMS §3은 "Wall 2 bit"를 금한다. `build/make_tables.py`에서 두 행에 각주를 단다: `borderline; bootstrap in Table~\ref{tab:r1}`
- **S13. 사전 고정 문구.**
  - `main.tex` 머리말 주석은 "one sentence in Sec. 3.4 + App. A/B"라고 정했는데 서론에 한 문장이 더 있다. `main.tex:111`: `The analysis plan of every stage was fixed before its data (Appendix~\ref{app:plan}), and the tests that did not hold are reported (…).` → `Appendix~\ref{app:plan} records the analysis plan of every stage, and Appendices~\ref{app:tests} and~\ref{app:neg} report the tests that did not hold.`
  - 해시는 저자 로그에만 있고 보충자료의 사본은 해시가 달라 심사자가 확인할 수 없다. 본문의 "pre-specified" 4회와 "fixed before" 류 4회는 3.4절 한 번으로 줄이기를 권한다(의견).
  - 5.1절의 drift 검정은 E1 뒤에 세운 것이다. `main.tex:219` 첫 문장 뒤에 추가: `We formulated this drift test after E1, when the original OEI tests had failed (Section~\ref{sec:theory}).`
- **S14. LLM 각주와 "worker"(저자가 정할 일).**
  - `app_A.tex:49`, `:57`, `:58`, `:66`의 "The E7 worker saw…", "The E1 DINO-WM worker deleted its own test files…", "The theory worker…", "delegated to the workers"는 누가 worker인지 묻게 만든다.
  - 보충자료의 로그에는 worker, coordinator, 판단 에이전트가 D1~D3을 정한 기록이 있다. 각주의 "The authors designed the study, executed and managed the experiments"와 어긋나 보일 수 있다.
  - 각주는 사용자 문구라 고칠 글은 제안하지 않는다. 다만 각주, 부록 A, 보충자료가 같은 사실을 말해야 한다. worker를 한 번 정의하거나, 자료에 닿지 않은 운영 사건(항목 14, 16, 22, 24, 25, 28)은 로그로 넘기는 방법이 있다.
- **S15. 표 1 캡션.** DINO-WM의 draw는 (token, unit)마다 한 값이다(`main/code/dinowm/qlib.py` 11·19행). `draws have one entry per unit of a layer` → `draws have one entry per unit of a layer (per token and unit in DINO-WM)`

### Could

- C1. 부록 A의 편차 28개와 부록 J의 "Claims we do not make" 14개는 내부 점검표처럼 읽힌다. 독자가 실제로 오해할 5~6개만 남긴다(의견).
- C2. 부록의 표가 절 제목보다 먼저 나온다(표 7이 부록 D 제목 앞, 표 3이 부록 C 안). `placeins`의 `\FloatBarrier`를 절마다 넣는다. 표 20은 `\resizebox`로 글자가 매우 작다.
- C3. 12쪽을 지키려면 줄일 곳
  - `main.tex:193`의 합산(R4) 문장 → 부록 I(이미 있음)
  - `main.tex:129`의 WM-Quant·LAQ 서술 압축
  - `main.tex:302`의 Orin 세부
  - `main.tex:180`의 Δ_bias(본문에서 쓰이지 않음)
- C4. 표 2의 해시 앞 12자리는 공개 저장소의 LOG와 그대로 맞는 검색어다. 익명판에서는 해시 열을 보충자료로 넘기는 것을 고려한다(의견, 위험 낮음).
- C5. LLM 각주가 TD-MPC2 인용 문장에 붙어 있다. 번호 없는 각주나 문단 끝이 덜 어색하다.
- C6. 겹친 괄호: `main.tex:172`의 `in~\citep{kuzmenko2025tdmpcopt}`를 `\citet`로, `(Yu et al., 2019) (success rate)` 꼴도 정리한다.
- C7. `REFERENCES_CHECK.md` §3.4에 남은 후보 둘(arXiv 2605.15960 "Imperfect world models are exploitable", 2607.28405)과 LoopQuant 문장을 투고 전에 확인한다.
- C8. 그림 5 캡션에 추가: `Points missing from a curve were not run.`

## (2) 논리 흐름

- 큰 순서(현상 → 무엇을 이용하나 → 처방과 한계 → 비용)는 맞고, 주장 순서로 쓴 점도 좋다. 걸리는 곳은 다섯이다.
  1. 처방의 같은 비트 결과가 4.1절에 있고 6절은 비트 한계로 시작한다(S1).
  2. 초록의 "두 성분"과 5.2절의 "bias, resolution, persistence" 셋 사이의 대응 문장이 없다. systematic = bias + resolution이라고 한 번 써야 한다(M8).
  3. 5.3절은 네 주제를 묶었고, 5.1절은 E2 결과인데 단계 이름이 없다(S2, S13).
  4. det+noise는 5.2절에서 결과가 먼저 나오고 6.2절에서 "Contrary to our prediction"이 나온다(M8).
  5. 6.1절 한 문단에 E5, E12, R1, R2, X5, 세 DINO-WM 과제, DMC가 다 들어 있다(S1).
- 대조군 소개
  - det+BC: 충분하다(3.2절).
  - sh-ep/sh-plan: 양성 대조로 5.1절에서 소개된다. 다만 결과가 4.1절에 먼저 나온다.
  - det+shift-fixed: 3.5절의 Δ_pers로 소개된다.
  - sh-iter+b, det+noise: 부족하다(M8).
  - ind: 왜 있는지 본문에 없다(S2).
- 계열 사이 전환: 5.2절의 문단 머리(`TD-MPC2: persistence.` / `DINO-WM: not persistence.`)는 좋다. 4.1·4.2·6.1절은 한 문단 안에서 계열이 섞인다.
- 사후 칸과 E1~E10: 표시는 정직하고 일관된다. 다만 DINO-WM 네 칸의 지위를 글로만 따라가기 어렵다.
  - Wall 3-bit는 E1 탐색 칸이자 새 분할의 사전 지정 칸이다.
  - PushT 3-bit는 E1 확증 칸이자 새 분할의 사후 칸이다.
  - PushT 4-bit는 미재현, PointMaze는 Holm 미통과다.
  - 표 30을 4행으로 줄여 4.1절에 올리고 `main.tex:193`의 수치 문장을 덜어내면 줄 수도 준다.
- `main.tex:234`는 "resolution on PushT"라고 읽은 뒤 곧바로 det+BC의 실패는 편향의 반증이 아니라고 쓴다. 결론을 문단 앞에 둔다: `Neither cell is dominated by persistence; on PushT we cannot tell whether the mean bias or the resolution loss is at work.`

## (3) 그림과 표

- 본문은 표 1개와 그림 7개다. 결과 수치는 모두 문장 안에 있다. 그림 2(손상 대 처방), 그림 4(대조), 그림 5(비트)가 핵심을 보여 준다. 고칠 곳은 다음이다.
- **F1 그림 5**(`fig_bitlimit`, 23판, 한 판 폭 약 0.65 in, 글자 5.8~6.4pt). 인쇄 크기에서 읽기 어렵다.
  - k\* 표식이 자료 축 안의 y ≈ −30에 있다. Wall 판에서는 det 4-bit의 실제 값(−22)과 붙어 한 덩어리로 보인다.
  - 제안: k\* 띠를 축 밖으로 내린다. 본문에는 Meta-World per-tensor 줄과 DINO-WM 줄만 두고 per-channel 줄은 부록으로 보내거나, 과제별 k\* 아령 그림 한 장으로 바꾼다.
- **F2 그림 2.** (c) 캡션은 M2. 축 "(%)"에 MW·DINO-WM의 점과 DMC의 return %가 섞인다(`damage (points or %)`). (c) 판 제목 "error size hides it"은 (8) 참조.
- **F3 그림 4.** 캡션은 "by each control"인데 sh-iter+b가 없다. `main.tex:241`: `by each control in every damaged cell` → `by each control except \shiterb (Table~\ref{tab:components}) in every damaged cell`. 열 머리에 원인별 예측을 한 줄로 넣으면 그림만으로 읽힌다. 행 이름은 5.6pt다.
- **F4 그림 3.** "bars show 95% bootstrap intervals"는 `error bars show`로 고친다. 범례의 det(주황)와 기준 점선(회색)이 다르다. (b)에 단계 이름(E2)이 없다.
  - (b) push에서 sh-step은 반복 1 → 12에 56 → 18로 내려간다. 본문에 설명이 없어 심사자가 물을 수 있다(S5와 같은 논점).
- **F5 그림 7(a).** FP32 대비 2~3배만 보인다. 정직한 비교(미리 바꾼 BF16이 더 빠름)는 (b)와 본문에만 있다. (a)에 E8c의 BF16 선을 더하거나 (a)를 뺀다.
- **F6 그림 6(a).** 축 이름과 S 정의(M3).
- 중복: 그림 1(b)는 그림 5의 push per-tensor 판과 같은 자료다(성공률로 다시 그림). 맛보기로 둘 만하다. 표 1과 그림 1(a)는 서로 보완한다.

## (4) 재현성

- 좋은 점: 수치가 모두 `\nv` 매크로이고, 재계산 스크립트, seed 구간, 상류 커밋, 독립 재계산이 있다.
- 지금 서술대로는 되지 않는 것: M4.
- 스크립트는 `paper/tmlr/build/`에서 상대 경로로 `main/results`, `rss_ext`, `tmlr_ext`를 읽는다(`build/make_numbers.py` 22~27행). 보충자료가 이 폴더 구조를 그대로 가져야 하는데 부록 K는 구조를 말로만 적었다.
- 빠진 실험 세부
  - 잠재 오차의 식, e의 부호, S, α(M3)
  - 정확한 과제 ID(M7)
  - 칸 분류 규칙(편향 → 지속 → 해상도 → 혼합, 설계 §1.4). 표 20 캡션에 한 문장으로 넣는다.
  - per-tensor 범위에 쓴 E0 episode 수
  - DINO-WM의 성공 기준과 실행 방식(계획 한 번, horizon 5)
  - DINO-WM의 draw 모양(S15)
  - 표 24 보상 오차의 척도. 후보 보상 표준편차의 수십 배(예: 115.4)인 이유를 한 줄로 설명한다.
  - 분석 스크립트의 Python·NumPy·SciPy 판(SciPy의 ±0.8 문제를 스스로 적었으므로)
- 해시 기록(부록 A)은 저자 로그에만 있다. 독립 증거로 읽히지 않는다는 점을 감안해 쓴다.

## (5) 과장 목록 (CLAIMS v2.1 대비)

- M1, M2, M5에서 다룬 문장은 다시 적지 않는다.
- **O1** `main.tex:102`, `:236` "in the two families". 승인 문구와 같지만 근거는 Meta-World 대 DINO-WM 2칸이다. DMC는 어느 쪽도 아니다(새 잡음 1/4, 고정 무편향 0~1/4).
  - `:236`: `…move in opposite directions in the two families.` → `…move in opposite directions on Meta-World and on DINO-WM; DMC follows neither side (Section~\ref{sec:noise}).`
  - `:102`: `in the two families, and swapping` → `on TD-MPC2's Meta-World tasks and on DINO-WM, and swapping`
- **O2** `main.tex:104` `where its role is different: to deny the planner a persistent error to exploit.` 지속을 계획 일반의 역할로 쓴다(CLAIMS §3 첫 항). → `where redrawing has a second effect in TD-MPC2: the planner finds no error that persists across its iterations.`
- **O3** `main.tex:109` `and the integer path reproduces the fake-quantization results.` 근거는 K2의 한 대비 4과제와 13/16칸이다. → `and the kernel reproduces the fake-quantization difference between \dtm and \detshift in all \nv{e17.k2.n} tasks tested.`
- **O4** `main.tex:191` `so the larger model is not more robust to rounding.` 체크포인트 하나다. → `so we see no sign that the larger model is more robust to rounding.`
- **O5** `main.tex:204` 첫 문장. 7/36칸에서는 det 오차가 더 크다. → `Within a bit width, the latent one-step error does not reveal \dtm's damage in most cells.`
- **O6** `main.tex:234` 사후 칸에 "pre-specified reading"을 쓴다. → `Applying the pre-specified classification rule gives bias on Wall and resolution on PushT`
- **O7** `main.tex:247`
  - `The fix needs an iterating planner.` MPPI 근거뿐이다. DINO-WM은 반복 5~30 모두에서 통했다(부록 F (iv)). → `In TD-MPC2 the fix needs an iterating planner.`
  - `\dtm's exploitation $|\mathrm{OEI}|$ and damage grow` 5.4절은 OEI를 이용의 척도로 쓰지 않는다고 한다. → `\dtm's $|\mathrm{OEI}|$ and its damage grow`
- **O8** `main.tex:280` `On DMC the fix is thus bounded by the number of planner iterations.` walker는 설명되지 않는다. → `On three of the four DMC tasks the fix is thus limited by the number of planner iterations; walker-walk is not repaired at any number we tested.`
- **O9** 초록 끝과 `main.tex:315`의 "helps less". CLAIMS C4 한정은 walker에서 "해롭다"다(E1 +22.1 [18.0, 26.3], E11 +17.8 [13.3, 22.5]).
  - `:315`: `The prescription assumes an iterating planner: with one iteration it does nothing, and on DMC it needs more iterations than the default.` → `The prescription assumes an iterating planner. With one iteration it does nothing, on DMC it needs more iterations than the default, and on walker-walk at 3 bits it is worse than round-to-nearest at the default six iterations.`
- **O10** `main.tex:320` 결론
  - `the second dominates in TD-MPC2 and the first in DINO-WM.` → `the second dominates in TD-MPC2 but not in DINO-WM, where a fixed unbiased error repaired both damaged cells.`
  - `…, and costs about \nv{e17.shiter.overhead.max0}\% of rollout latency. Since the latent one-step error does not show the damage, quantized planners need closed-loop validation.` 한정어가 빠졌고, 부록 J의 "wrong measure in general" 비주장과 어긋난다. → `…. Its simplest form adds at most about \nv{e17.shiter.overhead.max0}\% to rollout latency in our integer kernel. At a given bit width the latent one-step error did not separate the rounding rules that plan well from those that do not, so we recommend validating quantized planners in closed loop.`
- **O11** `main.tex:129` `In our terms the compounding error it describes is the systematic component` 다른 논문에 대한 해석이다. → `The compounding error it describes corresponds most closely to our systematic component`
- **O12** `main.tex:98`, `:100`, `:129`의 "are known to". 근거가 preprint와 익명 투고다. → `have been reported to`. Xu et al. 2026a는 서론에서는 "offline error do not explain"의 근거로, 2절에서는 "offline action deviation predicts"로 인용돼 서로 어긋나 보인다.
- 제목(한 줄): "Planners Exploit…"은 DINO-WM에서 지속 이용이 보이지 않았고, "Low-Bit World Models"는 대부분의 실험이 은닉 활성 4~6개 텐서만 양자화했다는 점에서 넓게 읽힐 위험이 있다. 초록과 범위 문장이 막아 주므로 유지해도 된다(의견).

## (6) 용어와 표기

- **affected / damaged**: 같은 뜻인데 본문에 6번 / 16번 섞인다. "affected"는 부록 D에서만 정의된다. `damaged` 하나로 통일한다(`main.tex:100`, `:191`, `:198`, `:204`, 표 7·8·18의 캡션).
- **dither**: `main.tex:147` "All dither variants … the same per-entry error distribution"은 det+shift(Δ²/12)에 맞지 않는다. 그런데 3.4·6.1절은 det+shift를 "dither form"으로 센다. → `The dither variants \ind, \shiter, \shstep and \shep use … distribution; \detshift and \detshiftfix (subtractive dither: round-to-nearest on a randomly shifted grid) share another.`
- **det+shift**: 이름이 결정적 변형처럼 읽힌다. 처음 나올 때 위 괄호 설명을 붙인다.
- **cost of persistence**: 두 대비(S5). **persistent share**: 두 양(M3).
- **τ**: 6.1절에서 처음 나온다. `main.tex:176`: `with $D\le10$ points (10\% on DMC)` → `with $D\le\tau$, $\tau=10$ points (10\% on DMC),`
- **G**: DINO-WM에는 det+shift가 없다. 같은 문장에 `(the DINO-WM sweep has no \detshift arm, so there the second minimum is $k^*_{\shiter}$)`를 붙인다.
- **J**: 3.1절은 return(MPPI 가중 exp(β(J − J_max))), 부록 C·F는 cost(w ∝ exp(−J/λ))다. 부록 첫머리에 `In this appendix $J$ denotes a cost.`를 넣는다.
- **red**: 그림 캡션에서 색 이름으로 읽힌다(`main.tex:241` "Damage removed, red, with 95% intervals", 그림 6 범례 "red = 0.5"). → `Fraction of the damage removed ($\red$), with 95\% intervals`
- **unit**: 뉴런(`per-unit bias`)과 검정 단위(표 3 "A unit is TD-MPC2 Meta-World…")로 두 뜻이다. 뒤쪽을 `benchmark`로 바꾼다.
- **family**: 두 모델 계열, 표 18의 네 묶음, "the det family"로 세 뜻이다.
- **table**: `main.tex:219`의 "the full-size fixed table"은 정의(5.3절)보다 먼저 나온다. → `the fixed dither with one offset per token and unit`
- **samples / candidates**(`main.tex:172`), **decision / control step**: 하나로 통일한다.
- **단위**: `app_G.tex:23`은 한 문장에 비율(0.42~0.93)과 점(−1.1~2.2)이 섞인다. "real"(drop is real, not real)은 `significant`나 M3의 문구로 바꾼다.

## (7) 첫 쪽

- 초록과 서론 첫 문단만 읽은 판단
  - 문제와 질문은 분명하다(`We ask a narrower question…`이 좋다).
  - 주된 결과는 초록 3~5문장(승인 문구)에 있다. 다만 dither, error field, per-channel grids, NVFP4가 설명 없이 나온다.
  - "왜 중요한가"는 첫 문장에 암시만 있다.
  - 둘째 문장은 M1대로 사실과 다르다.
- 제안(승인된 3~5문장은 그대로 둔다)
  - `main.tex:86` 첫 두 문장 → `Sampling-based planners such as MPPI and CEM query a learned world model thousands of times per decision, so running them on low-bit hardware requires quantizing that model. We find that the rounding rule alone changes whether planning works. With round-to-nearest activation quantization, planning loses up to \nv{e1.maxdet} points of success, and in most affected settings the world model's latent one-step error is no larger than under stochastic rounding (dither).`
  - 마지막 문장: `It helps less where the planner iterates little, as on DMC with the default number of iterations, and in TD-MPC2 block-scaled NVFP4 largely avoids the failure.` → `It helps less where the planner iterates little, as on DMC with the default number of iterations, where it can also hurt. With block-scaled NVFP4 activations, round-to-nearest damages \nv{e9.nvfp4.ndam} of 12 TD-MPC2 tasks.`
  - `main.tex:100` 첫 문장: `It matters a great deal.` → `It does.`

## (8) AI처럼 읽히는 문장

- **전체 통계(본문)**: em dash 0개, 콜론 62개(5절에 18개), 세미콜론 86개, 괄호 약 205쌍. 문제는 기호보다 문장 길이와 괄호 속 수치다.
- **(a) 강조·전환어**: "Notably", "Crucially", "Importantly", "It is worth noting", "not only … but also", "Moreover", "Furthermore"는 본문과 부록에 한 건도 없다(grep). 가까운 것만 적는다.
  - a1 `main.tex:100` `It matters a great deal.` → `It does.`
  - a2 `main.tex:219` `and \shep is even more harmful than \dtm on sweep-into ($D_{\shep}/D_{\dtm}=\nv{e14.sweep.r}$).` → `and on sweep-into \shep loses \nv{e14.sweep.r} times as many points as \dtm.`
  - a3 `main.tex:204` `Across bit widths, one-step error and damage do rank together; within a bit width, errors over more steps increasingly separate \dtm.` → `Across bit widths, one-step error and damage rank together. Within a bit width, errors over more steps separate \dtm in more cells.`
  - a4 `main.tex:206` `agrees much better with` → `agrees better with` (크기는 뒤의 수치가 말한다)
  - a5 "we therefore"가 본문에 3번(`:206`, `:232`, `:300`), "therefore"가 7번이다. `:300` `; we therefore treat \detshift as a mechanism control and one form of the fix, not as a co-equal prescription.` → `. For this reason we use \detshift as a mechanism control and as one form of the fix, and we recommend \shiter.`
  - a6 강조용 only·already
    - `:292` `damaged only \nv{e9.nvfp4.ndam} of 12 tasks` → `damaged \nv{e9.nvfp4.ndam} of 12 tasks`
    - `:280` `repairs only` → `repairs`
    - `:107` `already repairs it` → `repairs it`
- **(b) 비유·구호형**
  - b1 `main.tex:104` `redrawing denies the planner an error it can exploit across iterations` → `redrawing leaves no error that stays the same across iterations`
  - b2 `main.tex:104` `The ingredients are classical: the optimizer's curse…` → `These ideas are classical. They are the optimizer's curse…`
  - b3 `main.tex:100`, `:131` `drive their candidates to where the model can no longer resolve them` → `concentrate their candidates in regions where the model cannot rank them`
  - b4 `main.tex:219` `The control survives scale: at 317M, the cost of persistence…` → `The result is the same at 317M. The cost of persistence…`
  - b5 `main.tex:247` `a single offset per layer collapsed planning (\nv{e3.wall.singleoffset} points)` → `a single offset per layer cost \nv{e3.wall.singleoffset} points`
  - b6 `main.tex:253` `A persistent error field instead pulls the population toward the minimizer of the smoothed model cost,` → `With a persistent error field the population instead converges to the minimizer of the smoothed model cost,`
  - b7 `main.tex:276` `\detnoise does not hold up as well as dither: it gains…` → `\detnoise gains fewer bits than dither. It gains…`
  - b8 `main.tex:292` `Block-scaled FP4 formats mark a boundary of the failure in TD-MPC2: with activations in NVFP4…` → `With block-scaled FP4 activations, \dtm causes little damage in TD-MPC2. With activations in NVFP4…`
  - b9 `main.tex:302` `Low-bit arithmetic pays off on hardware where it is much faster than 16-bit floating point: an NVFP4 GEMM…` → `On other hardware low-bit arithmetic is much faster than 16-bit floating point. An NVFP4 GEMM…`
  - b10 `main.tex:323` `Low-bit world models can fail silently: offline prediction error may look fine while closed-loop planning collapses.` → `A low-bit world model can have a small offline prediction error and still plan badly in closed loop.`
  - b11 제목류
    - 4.2절 `What offline error sees and what it misses` → `Offline errors and planning damage`
    - 그림 2(c) 판 제목 `error size hides it` → `extra damage against error ratio`
    - 그림 3(a) 판 제목 `persistence brings the drift back` → `drift kept by each variant` (PushT에서는 되살아나지 않았으니 사실과도 맞지 않는다)
  - b12 `app_D.tex:53` `Where damage remained, the signature was the same:` → `Where damage remained, the pattern matched the uniform grid:` / `Block scaling shrinks the error and dither removes its persistence, two different axes;` → `Block scaling reduces the size of the error, and dither changes whether it persists;`
  - b13 `app_E.tex:61` `The theory predicts the direction (Appendix~\ref{app:theory}); the data do not show it.` → `The theory predicts a positive correlation (Appendix~\ref{app:theory}), which the data do not show.`
  - b14 `app_F.tex:23` `MPPI stops at its floor $\sigma_{\min}$, while CEM keeps collapsing.` → `MPPI's population width stops at its floor $\sigma_{\min}$, while CEM's keeps shrinking.` / `app_F.tex:27` `rounding toys` → `toy rounding models`
  - b15 `app_G.tex:21` `, as if the model had adapted to deterministic rounding.` → `. Training may have adapted the model to deterministic rounding; we did not test this.`
  - b16 "X, not Y" 틀이 본문에 5번, 부록에 3번 되풀이된다.
    - `main.tex:249` `The difference lies in the sign of the bounds, not in the size of every cell:` → `The families differ in whether the interval excludes zero. The cell-by-cell sizes overlap:`
    - `main.tex:253` `and why our mechanism claims concern the drift and the controlled contrasts, not the OEI.` → `, and our mechanism claims rest on the drift and on the controlled contrasts.`
    - `main.tex:193` `so the split, not the hardware, explains the difference.` → `so the hardware does not explain the difference, which leaves the split.`
- **(c) 구두점·형식**
  - c1 `main.tex:193` 한 문단에 괄호 14쌍. `Pooling the two splits post hoc (inverse-variance weights; the splits are unpaired and ran on different GPU types) gives … so we do not pool them.` → `Appendix~\ref{app:new} pools the two splits for PointMaze and Wall and explains why we do not pool PushT.`
  - c2 `main.tex:265` `Which tasks gain a bit is less stable: pick-place (bootstrap probability of $G\ge1$: … per tensor, … per channel) and window-open (…, …); pick-place's per-channel gain rests on one cell whose damage equals the threshold, and with a strict threshold ($D<\tau$) the per-channel count is \nv{r2.fr2.strict}.` → `Which tasks gain a bit is less stable. For pick-place the bootstrap probability of $G\ge1$ is \nv{r1.pickplace.pt} per tensor and \nv{r1.pickplace.pc} per channel, and for window-open it is \nv{r1.windowopen.pt} and \nv{r1.windowopen.pc}. With a strict threshold, $D<\tau$, the per-channel count is \nv{r2.fr2.strict}.`
  - c3 `main.tex:129` 78단어 한 문장(`Concurrent work proposes a post-training quantization framework … whole rollout`) → `Concurrent work proposes a post-training quantization framework for JEPA-style world models~\mbox{\citep{wmquant2026}}. It reports that quantization error compounds with each predictor step until action rankings become near-random. It approaches full-precision planning at W4A4 on LeWM and, with rotation-based quantizers, on DINO-WM Wall, though not on DINO-WM PushT. Its components are higher precision for early encoder activations, a floating-point action-conditioning path, and activation ranges learned per rollout step against the error of the whole rollout.`
  - c4 `main.tex:204` 겹친 괄호 `…\nv{e.e13.err3.n} over three steps (and in \nv{e1.td.held.err3} of the \nv{e1.td.naff} damaged E1 cells) and \nv{e.e13.err5.n} over five steps, and in DINO-WM…` → `…\nv{e.e13.err3.n} over three steps and \nv{e.e13.err5.n} over five steps. Over three steps the count is \nv{e1.td.held.err3} of the \nv{e1.td.naff} damaged E1 cells. In DINO-WM…`
  - c5 `main.tex:198` 그림 2 캡션이 92단어 한 문장이다. (a), (b), (c)마다 문장을 끊는다.
  - c6 `main.tex:302`의 85단어 문장(NVFP4 GEMM과 Orin)은 b9처럼 끊고 Orin 세부를 `1.36--2.13$\times$ faster than FP16`까지만 남긴다.
