<!-- 검토자(Claude Opus 5.5, 추론 노력 max)의 최종 보고 전문. 2026-11-22 사용자 요청("이번엔 opus 5.5 max로 전체적인 문장, 논문으로써 자연스러운 문장구조, 논문으로써 자연스러운 문체 등을 중점으로 독립적으로 전체 원고를 확인해줘")으로 조정자가 맡겼고, 조정자가 그대로 옮겼다. -->

# 원고 A(TMLR) 문장·문체 독립 검토

`main.tex` 본문과 캡션, 부록 A–K, 표 캡션을 읽었습니다. 파일은 고치지 않았습니다. 인용한 원문 168곳은 모두 스크립트로 해당 행에 실제로 있는지 대조했습니다. 각 문단의 마지막 줄이 얼마나 남는지는 PDF 좌표로 쟀고, 제안마다 그 폭 안에 드는지 확인했습니다. 길이 표기는 렌더링된 글자 수 차이이고, ±5자 이내는 '같음'으로 적었습니다.

## (0) 총평

**문장 품질.** 문장은 대체로 문법에 맞습니다. 주장 범위를 지키는 단서와 post hoc 표시가 일관되고, 기계 문체 흔적(em dash, 과장어, 구호)도 거의 없습니다(본문 em dash 0개). 가장 큰 약점은 밀도입니다. 결과 수치, 구간, 조건, 예외를 한 문장에 몰아 넣어서, 독자는 수치를 다 읽은 뒤에야 문장의 요점을 알게 됩니다. 둘째 약점은 해석을 담는 핵심 구절의 번역투와 비유입니다. 셋째는 지시어와 생략에서 오는 모호함입니다. 초록과 서론은 정확하고 질문도 선명합니다. 다만 첫 문장의 오독, 정의 전 사용, 붙여 놓은 듯한 끝 문장 때문에 다듬어진 TMLR 논문의 첫인상에는 조금 못 미칩니다.

**자주 나온 문제 다섯 가지**
1. **과밀 문장.** 본문 227문장의 평균이 29.4단어입니다. 45단어 이상이 45문장(그중 50단어 이상 31, 60단어 이상 12)이고, 괄호 묶음이 셋 이상인 문장이 25개입니다. 예: 262행(63단어, 삽입구 세 겹), 298행(77단어), 131행 Fast-WAM 문장(80단어).
2. **지시어·생략의 모호함(본문 약 17곳).**
   - 109행 "repairs it": it이 persistence로 읽힙니다.
   - 221행 "The result is the same at 317M": 바로 앞의 PushT 무결과를 가리키는 것처럼 읽힙니다.
   - 264행 "seven of these tasks": 앞 문장의 여섯 과제로 읽힙니다.
   - 그 밖에 253행 "both tasks there", 279행 "why they differ", 149행 "share another" 등.
3. **정의 전 사용과 내부 라벨(본문 약 12곳, 부록 약 9곳).**
   - 그림 1 캡션의 "E12"는 3쪽에 나오는데 정의는 5쪽에 있습니다.
   - 223행 "exactly at the threshold": 문턱 −0.8을 본문에서 말하지 않습니다.
   - 그 밖에 "does not pass Holm", "planned classification rule", "within-bit".
   - 약어: W4A4·W8A4가 첫 사용에서 풀리지 않았고, MPPI·CEM·DMC는 끝까지 풀리지 않습니다.
   - 부록: §11, coordinator, smoke, reproduction split, MPS, VLA.
4. **번역투·비유·구어, 그리고 완곡 표현의 과부족(약 18곳).**
   - 번역투·비유·구어: "the main share", "a partial component", "tells the three apart", "move in opposite directions / follows neither side", "brings back", "keeps the policy's level", "which leaves the split", "planning works", "stops hurting", "does nothing". 비교에 쓴 "against"도 약 12회입니다.
   - 완곡 과잉: "at most about".
   - 완곡 결핍: "much faster", "only redrawing removes".
5. **표기 불일치.**
   - 단위: points / success points / points of success, % of return / relative return.
   - red가 분수로도 백분율로도 쓰입니다.
   - 부호 붙은 값에 동사 loses를 씁니다("loses +42 [31, 53] points").
   - 음수 범위가 "−0.00–0.24"처럼 찍힙니다.
   - DMC 과제 약칭, "dither"가 가리키는 범위.
   - 기호 중복: α에 세 가지 뜻, K, ℓ, H, n.
   - 문장이 소문자 변형명으로 시작하는 곳이 11곳입니다(102, 110, 149×2, 193, 221, 236×2, 240, 296×2).

**우선순위**
1. 오독을 낳는 곳: (1)의 11·31·38·40·44, 부록 A2·A3, (5)의 1·2. 길이는 거의 변하지 않습니다.
2. 초록과 서론((4)).
3. 4–7장의 과밀 문장(항목 25–47).
4. 용어·단위 통일((3)). 부호, 범위, "922k" 같은 매크로 출력은 `make_numbers.py`에서 한꺼번에 고치는 편이 낫습니다.
5. 부록.

**쪽수 예산.** Broader Impact 끝 줄이 이미 12쪽 바닥에 있습니다. 아래 제안을 모두 적용해도 각 문단은 마지막 줄의 남은 폭 안에 들어갑니다. 게다가 항목 12·29·41·43에서 한 줄씩 줄어 3–4줄이 남습니다. 여유가 거의 없는 곳은 3.1(0자), 5.3(약 7자), 한계(약 8자), 기여 1(약 12자), 초록(약 18자)입니다. 그림 위치가 바뀔 수 있으니 적용한 뒤 빌드해서 쪽수를 확인하십시오.

## (1) 문장 단위 수정 (읽는 순서)

### 초록·서론 (붙여 넣을 전체 글은 (4))
1. `main.tex:87` [구어] `whether planning works.` → `whether planning succeeds.` (같음 +3)
2. `87` [문법, 결론 316행과 꼴이 다름] `adds at most about \nv{e17.shiter.overhead.max0}\% rollout latency` → `... \% to rollout latency`. 같음 +3이고, 승인 문장 안에서 전치사만 바꾼 것입니다.
3. `87` [지시어 It] `It helps less where ..., as on DMC with the default number of iterations` → `Dither helps less where ..., as on DMC at the default number of iterations` (같음 +2)
4. `87` [해석 없이 붙은 끝 문장] `round-to-nearest damages \nv{e9.nvfp4.ndam} of 12` → `round-to-nearest damages only \nv{e9.nvfp4.ndam} of 12`. 같음 +5이고, 초록 전체 +13자로 남은 폭 약 18자 안입니다.
5. `99` [명사 나열로 첫 문장 오독] `evaluates its world model candidates $\times$ iterations $\times$ horizon times` → `calls its world model (candidates $\times$ iterations $\times$ horizon) times` (같음 −2)
6. `99` [unless가 수식하는 범위] `Quantized world models have been reported to break planning` → `Quantized world models can break planning` (짧음 −18)
7. `102` [정의 전 사용, 구정보→신정보 순서, 문두 소문자 \dtm, "It does."]
   - "shared dither"가 dither 정의보다 세 문장 앞에 나옵니다.
   - 순서를 정의 → 결과 → 선행 연구로 바꿉니다((4)). 김 +19, 남은 폭 61자 안.
8. `104` 두 곳(문단 합계 +37, 남은 폭 55자 안)
   - [목적어 불명] `put it back,` → `add it back to fresh dither,`
   - [번역투, 주어 어색, 시제] `The first is the main share in DINO-WM ...; swapping the planners' update rules kept this pattern, and we do not identify its cause.` → `The first accounts for most of the damage in DINO-WM and the second in TD-MPC2 at 5M and 317M parameters; the pattern held when we swapped the planners' update rules, and we did not identify its cause.`
9. `106` 세 곳
   - [범주 오류] `These ideas are classical. They are the optimizer's curse` → `These ideas draw on classical work: the optimizer's curse`
   - [어색한 전치사구] `which language-model serving also uses at inference against error accumulating in` → `which language-model inference also uses to keep error from accumulating in`
   - [지시어] `To our knowledge it has not been used` → `To our knowledge, such dither has not been used`. it이 stochastic rounding 전반을 뜻했다면 그 말로 바꾸십시오. 합계 +4.
10. `108` [지시어] `It reappears on new seeds` → `The damage reappears on new seeds` (김 +8; 이 항목 끝 줄의 남은 폭 약 12자)
11. `109` [it이 persistence로 읽힘] 두 곳(김 +14)
    - `repairs it.` → `repairs the damage.`
    - `Swapping the planners' update rules kept this pattern.` → `The pattern held when we swapped the planners' update rules.`
12. `111` [줄 절약] `to the world-model rollouts, and` → `to rollout latency, and`, 그리고 `tasks tested.` → `tasks.`. 짧음 −16이고, 마지막 줄 "tested."가 없어져 한 줄이 줄어듭니다.
13. `113` [부정확: 부록 B는 모든 검정을 실음] `and Appendices~\ref{app:tests} and~\ref{app:neg} report the tests that did not hold.` → `Appendix~\ref{app:tests} lists every planned test, and Appendix~\ref{app:neg} what did not hold.` (김 +12)

### 본문
14. `118` 그림 1 [정의 전 라벨] `(E12, 90 paired episodes per point)` → `(Section~\ref{sec:bits}; 90 paired episodes per point)` (같음 +6)
15. `131` [and 연쇄] `..., and offline action deviation predicts the closed-loop loss of quantized world action models~\citep{xu2026prede}.` → `...; for quantized world action models, offline action deviation predicts the closed-loop loss~\citep{xu2026prede}.` (같음 −2)
16. `131` [콜론 뒤 65단어] `after training~\mbox{\citep{wmquant2026}}: quantization error compounds ... near-random, and higher precision` → `after training and finds that quantization error compounds ... near-random~\citep{wmquant2026}. Higher precision` (김 +10)
17. `131` [80단어, 삽입구 중첩] 다음으로 바꿉니다(짧음 −42).
    `Another concurrent study~\citep{laq2026} quantizes world action models to W4A4KV4 and weights each rounding error by the number of denoising steps that read it. In its Fast-WAM setting~\citep{fastwam2026}, an injected error of fixed energy damages the predicted future two to three times more when frozen across the denoising steps than when redrawn at every step, an analogue of our persistence axis without a planner.`
18. `133` [서론 102행 "have been reported"와 완곡 정도를 맞춤] `Iterating planners concentrate` → `Iterating planners have been found to concentrate` (김 +19)
19. `135` 세 곳
    - [논문 제목 "World Models"가 일반명사로 읽힘] `World Models injected ... exploit them~\citep{ha2018worldmodels}, and` → `\citet{ha2018worldmodels} injected ... exploit them, and` (−14)
    - [this] `reduces this` → `reduces the overfitting` (+11)
    - [네 항목을 한 문장에 나열] `from the signal~\citep{roberts1962picture, schuchman1964dither}, randomized quantization controls` → `... the signal~\citep{...}. Randomized quantization controls` (같음)
20. `145` (남은 폭 0자) 두 곳
    - ["for I iterations" 위치] `At every control step a planner samples $N$ ... to the best ones, for $I$ iterations, and then executes` → `For $I$ iterations per control step, a planner samples $N$ ... to the best ones; it then executes` (−6)
    - [끝에 매달린 동격구] `network, $N\times I\times H$ calls per control step.` → `network ($N\times I\times H$ calls per control step).` (+1)
21. `149` 두 곳(+21, 남은 폭 44자 안)
    - [생략 과다] `share another, with the error variance of \dtm.` → `share another distribution, with the error variance of \dtm.`
    - [같은 난수로 오독] `use the same unbiased draw` → `use the same kind of unbiased draw`
22. `174` [관계절 모호: 저자가 증류본을 쓴 것처럼 읽힘] `(mt80), which \citet{kuzmenko2025tdmpcopt} distilled and quantized to FP16, on` → `(mt80; distilled and quantized to FP16 in prior work~\citep{kuzmenko2025tdmpcopt}) on` (김 +8)
23. `178` 세 곳(합계 +14)
    - [유의성 누락] `drop is greater than zero` → `drop is significantly greater than zero`
    - [정의문 구조] `the smallest $k$ with $D\le\tau$, $\tau=10$ points (10\% on DMC), at that and every larger tested $k$,` → `the smallest $k$ such that $D\le\tau$ ($\tau=10$ points; 10\% on DMC) at $k$ and at every larger tested $k$,`
    - [단위] `(relative return on DMC)` → `(\% of return on DMC)`
24. `182` 두 곳(합계 +11)
    - [정의 과밀] 다음으로 바꿉니다. `The cost of persistence, the extra damage from keeping a draw for an episode at an equal per-entry error distribution, is $\dpers=D_{\detshiftfix}-D_{\detshift}$ (E11) or, where \detshiftfix was not run, $D_{\shep}-D_{\shiter}$ ($D_{\shplan}-D_{\shiter}$ in DINO-WM).`
    - [동격구가 세 항목으로 읽힘] `the persistent share $\rho_p$, the correlation of ... iterations, and the spread of $e(a)$ ... returns, $\sigma_e/\sigma_J$.` → `the persistent share $\rho_p$ (the correlation of ... consecutive iterations) and the relative spread $\sigma_e/\sigma_J$ (the spread of $e(a)$ across candidates relative to that of the predicted returns).`
25. `193` 두 곳(+21)
    - [문두 소문자] `\dtm costs up to` → `Round-to-nearest costs up to`
    - [its가 sh-iter로 읽힘] `; its primary bit widths` → `; the model's primary bit widths`
26. `195` [부호 붙은 값에 loses] 다음으로 바꿉니다(같음).
    `On the new DINO-WM split, \dtm's damage is \nv{dw.wall3.D.det} points on Wall at 3 bits, replicating the exploratory E1 cell (\nv{e1.wall3.D.det}), and \nv{d4.D.det} points on PushT at 3 bits, a cell we added post hoc, against \nv{d4.e1.D.det} in E1.`
27. `195` 네 곳(문단 −13)
    - [which의 지시 대상] `, which is weak evidence because its new pairs come from` → `, but this is weak evidence: its new pairs come from only`
    - [구어적 생략] `so the hardware does not explain the difference, which leaves the split.` → `so the split, not the hardware, explains the difference.`
    - [검정 이름을 동사처럼 씀] `but does not pass Holm over the three cells.` → `but is not significant after Holm correction over the three cells.`
    - [줄 보전] `and explains why we do not pool PushT.` → `but not for PushT.`
28. `200` 그림 2 [비율 방향] `the ratio of the two latent one-step errors` → `the ratio of \dtm's to \shiter's latent one-step error` (+10)
29. `206` 문단 −19로, 마지막 줄 "(Appendix E)."가 없어져 한 줄이 줄어듭니다.
    - [대조 표지 없음] `Across bit widths, one-step error` → `Across bit widths, by contrast, one-step error`
    - [연어 오류, 같은 부록 참조가 두 번] `for \shiter; Appendix~\ref{app:mech}).` → `for \shiter).`
    - 이어서 `The one-step reward error, redefined on these states, is larger for \dtm in ... cells (Appendix~\ref{app:mech}).` → `A redefined one-step reward error (Appendix~\ref{app:mech}) is larger for \dtm in \nv{e.e13.rew.n} of \nv{e.e13.ncells} cells.`
30. `208` 두 곳(+13)
    - [비표준 표현] `the within-bit ranking of the variants' damage agrees` → `the ranking of the variants' damage within a bit width agrees`
    - [DINO-WM 판정이 안 보임] `against a planned threshold of 0.3, and \nv{dg1.dw} in DINO-WM.` → `above the planned threshold of 0.3, but only \nv{dg1.dw} in DINO-WM.`
31. `221` 두 곳(−16)
    - [구어] `brings back ... in the three TD-MPC2 tasks` → `restores ... in the three TD-MPC2 mechanism tasks`
    - [지시 오독] `The result is the same at 317M. The cost of persistence` → `At 317M, the cost of persistence`
32. `223` 네 곳(+39, 남은 폭 69자 안)
    - [단위: 그림 3b 축은 "points or %"] `its damage is \nv{e2.it12.shiter}\% while \dtm's stays at \nv{e2.it12.det}\% and \shstep's at \nv{e2.it12.shstep}\%` → `its damage is \nv{e2.it12.shiter} points or \% while \dtm's stays at \nv{e2.it12.det} and \shstep's at \nv{e2.it12.shstep}`
    - [수 표기] `of four tasks` → `of 4 tasks`
    - [문턱 미정의] `exactly at the threshold)` → `exactly at the planned Spearman cut-off of $-0.8$)`
    - [모호] `no contrast of sharing or persistence separated the damage` → `no contrast of sharing or persistence had an interval excluding zero`
33. `225` [주어와 동사 사이 34단어] 두 곳을 바꿉니다(+10).
    - `(Appendix~\ref{app:theory}), whose selection and drift terms matched \nv{e4.nruns} simulated MPPI and CEM runs, shows that` → `(Appendix~\ref{app:theory}) shows that`
    - `approximation~\citep{kleywegt2002saa}. It also predicted` → `approximation~\citep{kleywegt2002saa}. The derivation's selection and drift terms matched \nv{e4.nruns} simulated MPPI and CEM runs. It also predicted`
34. `236` 두 곳(+10)
    - [격식] `E11 tells the three apart` → `E11 distinguishes the three`
    - [초록의 용어와 통일] `every randomized threshold repairs it` → `every unbiased randomized rounding repairs it`
35. `238` 세 곳(+11)
    - [respectively 누락, 환유] `Meta-World cells. A randomized threshold alone therefore does not repair Meta-World,` → `Meta-World cells, respectively. Fixed randomized rounding therefore does not repair these cells,`
    - [번역투] `The mean bias is a partial component:` → `The mean bias accounts for part of the damage:`
    - [보전] `and on the fixed-versus-redrawn contrast rather than on the class counts.` → `and the fixed-versus-redrawn contrast, not on the class counts.`
36. `240` 네 곳(+26)
    - [respectively] `at 3 and 4 bits in E12 and E11` → `at 3 bits in E12 and at 4 bits in E11`
    - [규칙 미설명] `so the planned classification rule gives bias on Wall and resolution on PushT` → `so the planned per-cell rule labels Wall bias-dominated and PushT resolution-dominated`
    - `is no evidence against` → `is not evidence against`
    - [59단어] `than the unquantized model, and \dtm at 4 bits also beats` → `than the unquantized model. Round-to-nearest at 4 bits also beats`
37. `242` 세 곳(−10)
    - [비유] `thus move in opposite directions on Meta-World and on DINO-WM, and DMC follows neither side:` → `thus have opposite effects on Meta-World and DINO-WM, and DMC fits neither pattern:`
    - [동사 생략] `and fresh noise on top of \dtm the reverse, all` → `whereas fresh noise added to \dtm repairs all`
    - `sit on the persistence side` → `fall on the persistence side`
38. `253` (남은 폭 약 7자, 여섯 곳을 함께 바꾸면 +3)
    - [기계 과제는 넷인데 "three"] `in all three mechanism tasks` → `in all three TD-MPC2 mechanism tasks`
    - [선행사 없음] `although \dtm damages both tasks there: ... on PointMaze and Wall with either planner` → `although \dtm damages PointMaze and Wall under it: ... on both tasks with either planner`
    - [잘못된 인과 so] `Every swapped planner kept at least 60\% of ..., so the pattern` → `With every swapped planner keeping at least 60\% of ..., the pattern`
    - [it] `causes it.` → `causes the pattern.`
    - [보전] `Only the PushT cell at 4 bits, where \dtm did not damage planning on the new split and which therefore stayed out of the planned readings, gives` → `Only the PushT cell at 4 bits, which stayed out of the planned readings because \dtm did not damage it on the new split, gives`
    - `is the table size:` → `is table size:`
39. `262` [63단어, 삽입구 중첩] 같음 +5.
    `(Section~\ref{sec:glance}), but on top of the best post-training quantization of E5, mostly per-channel ranges, which already removes ... points), its extra gain was` → `(Section~\ref{sec:glance}). The best post-training quantizer of E5, mostly per-channel ranges, already removes most of \dtm's damage at 3--4 bits (Meta-World $D$ of \nv{e5.pc.mw.range} points), and the extra gain of \shiter on top of it was`
40. `264` 세 곳(+31)
    - [it] `a rounding rule of E12 ..., and in all of them it is a dither` → `some rounding rule of E12 ..., and in all of them that rule is a dither`
    - [these tasks가 여섯 과제로 읽힘] `on seven of these tasks` → `on seven of the eight Meta-World tasks`
    - [어색] `mean mainly that the quantized planner keeps the policy's level.` → `mainly mean that the quantized planner does not fall below the policy.`
41. `268` [줄 절약, robustly가 supports와 중복] `but two bits robustly only per tensor` → `but two bits only per tensor`. 짧음 −9이고, 마지막 줄 "hoc)."가 없어져 한 줄이 줄어듭니다.
42. `270` [과제 수와 한 과제의 손상을 맞비교] `\nv{e9.mxfp4.ndam}, against \nv{e1.cheetah4.D.det}\% of return lost on cheetah-run with a uniform 4-bit grid;` → `\nv{e9.mxfp4.ndam}; on cheetah-run \dtm lost \nv{e9.nvfp4.cheetah}\% and \nv{e9.mxfp4.cheetah}\% of return, against \nv{e1.cheetah4.D.det}\% with a uniform 4-bit grid;`. 김 +20이고, 부록 D에 이미 있는 매크로를 씁니다.
43. `275` 그림 5 ["points"가 단위와 점 두 뜻] `points missing from a curve were not run.` → `missing markers: not run.`. −16이고 캡션이 한 줄 줄어듭니다. 그림 8 캡션(`app_I.tex:262`)도 같게 고칩니다.
44. `279` 세 곳(+32)
    - [where] `PushT), where dither repairs more (` → `PushT); dither repairs more in both (`
    - [they] `we did not test whether this relates to the repaired cells and did not identify why they differ.` → `we did not test whether this difference relates to which cells are repaired, nor did we identify why dither repairs more.`
    - [67단어 나열] `for dither, its $k^*$ ... tasks, and its damage is often` → `for dither, and its $k^*$ ... tasks. Its damage is also often`
45. `281` [모호] `at any number we tested` → `at any iteration count we tested` (+9)
46. `296` [56단어, "in both"가 모호] `worse than \dtm in both, and per-channel ranges, folded` → `worse than \dtm in both implementations. Per-channel ranges, folded` (+12)
47. `298` [77단어] `on another A40, but BF16 with the weights converted once in advance is the fastest` → `on another A40. BF16 with the weights converted once in advance, however, is the fastest` (+6)
48. `311` (남은 폭 약 8자, 세 곳 합계 −12)
    - [nor 구문] `, nor combine dither` → `, or combine dither`
    - [세미콜론 네 개 연쇄] `Our quantization-aware-training baseline was weak; ... and we have` → `Our quantization-aware-training baseline was weak, and weights stay in floating point except in the integer kernel. Latency comes from one GPU type and TD-MPC2 only, the FP4 results are fake quantization of the 5M models, and we have`
    - [주어 어색] `but which Meta-World tasks gain a bit does not.` → `but the per-task results do not.`
49. `316` [쉼표 위치] `Because, in most damaged cells, the latent` → `Because in most damaged cells the latent` (−2)

**공통 사항**
- 문장이 소문자 변형명으로 시작하는 11곳은 여유가 있는 문단에서 `The variant \detnoise ...`처럼 시작하거나 문장을 재배열하십시오.
- 첫 쪽 LLM 사용 각주에서는 문법 오류를 찾지 못했습니다. "LLM" 약어가 정의 없이 쓰였다는 점만 참고하십시오.

### 부록 (자주 나온 문제와 가장 나쁜 문장)
- A1 `app_A.tex:4` [세미콜론 연쇄, "v1's hash is the one logged" 모호] → `In the supplement, the copies of the main design for v2--v5.3 no longer reproduce their logged hashes because one line naming a server was removed; the copies of the extension, E8c and E11--E17 designs reproduce theirs. The hash of v1 was logged at its freeze, and that of v5.1 only by its first eight hexadecimal digits, in two worker logs.`
- A2 `app_A.tex:52` [쉼표 누락으로 오독] `on the new split D4--D6 after` → `on the new split, D4--D6 after`
- A3 `app_A.tex:54` [시제] `the tolerance fix described after the list does.` → `... did.`
- A4 `app_A.tex:58` [문두 소문자, 어순] → `\item No tested bit width gave the 20-point drop on drawer-open or PointMaze, so the pilot's fallback rule (the smallest tested width) was used. On sweep-into, the drop at 3 bits lay exactly at the threshold.`
- A5 `app_A.tex:86` [exactly 위치] → `SciPy returns Spearman correlations whose exact value is $\pm0.8$ as $\pm0.7999999999999999$.`
- A6 `app_A.tex:88` ["TF32 with pre-converted BF16"으로 오독] → `The comparison with pre-converted BF16 of an integer rollout whose first layer runs in TF32 was chosen ...`
- A7 `app_A.tex:93` [조각문] `Decided before any ...; episodes` → `This rule was fixed before any \detbc result with a converged $b$, and episodes`
- A8 `app_A.tex:92·94`: "smoke"와 "reproduction split"이 정의되지 않았습니다. `the pipeline was sound` → `the pipeline was not at fault`.
- C1 `app_C.tex:61` `16 MPS processes` → `16 processes under NVIDIA's Multi-Process Service (MPS)`. 같은 파일 31행은 `reuses these files` → `reuses these ranges`.
- D1 `app_D.tex:3` [det+shift는 분포가 다름] `to \shiter and \detshift, which stay near` → `to \shiter, which, like \detshift, stays near`. 같은 파일 53행은 `VLA policies` → `vision-language-action policies`.
- E1 `app_E.tex:3` `Mainly the variants that keep their error across iterations drift:` → `Drift comes mainly from the variants that keep their error across iterations:`. 같은 파일 21행은 `the difference is the planner's own share` → `a variant's drift minus this reference is the planner's own share`.
- F1 `app_F.tex:5` [콜론 뒤 예시 나열] `$W(\hat J+c\mathbf 1)=W(\hat J)$: argmin, ..., or the softmax ...;` → `..., as are argmin, the top-$K$ mean of CEM and the softmax over the top $K$ of MPPI;`
- F2 `app_F.tex:11` [목적어 없음] `Stein's identity gives $\sum` → `Stein's identity gives the selection term as $\sum`, 그리고 `in proportion to the near ties` → `in proportion to the density of near ties`.
- F3 `app_F.tex:30` 캡션 `Slopes are of measured on predicted values` → `Slopes are from regressions of measured on predicted values`. 49행은 `did not average fresh noise` → `did not average out fresh noise`.
- G1 `app_G.tex:23` [8-bit가 두 번 나와 모순처럼 읽힘] → `8-bit weights cause no damage (\nv{g.w.td.w8} points or \% of return), while 4-bit round-to-nearest weights alone lose \nv{e5.weights.w4.pts} (four TD-MPC2 tasks).` 같은 문단의 K(사본 수)는 top-K의 K와 겹칩니다((3) 참조).
- H1 `app_H.tex:3` [완곡 중첩] `at least about two seconds` → `about two seconds or more`
- I1 `app_I.tex:6` [D·X 라벨 혼용, X6 누락] `the post hoc runs D4, X5 and X17.` → `the post hoc cell D4 and runs X5, X6 and X17.` 같은 문단의 `The post hoc runs repeated their overlapping cells exactly:` → `Where a post hoc run overlapped an earlier cell, it reproduced that cell exactly:`
- I2 `app_I.tex:241` [본문 5.3과 다른 표현] `differ in the sign of the bounds` → `differ in whether the interval excludes zero`
- J1 `app_J.tex:11` [unit이 뉴런과 충돌] `in any unit;` → `in any benchmark;`
- J2 `app_J.tex:14` [4/6이 실패 수로 읽힘, 전문어] `(K1: \nv{e17.k1.n} of` → `(K1 held in only \nv{e17.k1.n} of`. 그리고 `the Pareto cases of H5 were weak.` → `H5's cases where dither with fewer bits Pareto-dominates \dtm with more were weak (push only).`

## (2) 문단·절 재구성 (쪽수 예산 고려)

1. **서론 2문단:** 순서를 det·dither 정의 → 결과 → 선행 연구 연결로 바꿉니다((4)). +19자로 남은 폭 안입니다.
2. **서론 4문단:** 마지막 절 "where redrawing has a second effect ..."는 두 문장 앞의 "redrawing leaves no error ..."와 내용이 겹칩니다. 줄이 더 필요하면 이 절을 지우십시오. 마지막 줄이 "are:" 넉 자라서 한 줄이 줄고, 새로움 문장 자체는 남습니다.
3. **관련 연구 첫 문단(약 20줄):** 동시대 연구 두 편을 run-in 문단 "Concurrent work."로 떼어 내면 읽기 쉬워집니다. 한 줄이 늘므로 항목 12·29·41에서 줄인 줄을 쓰십시오.
4. **4.2:** "Across bit widths ..."를 첫 문장 앞의 양보절로 옮길 수 있습니다(항목 29의 대안, 같은 길이). 그러면 주제문이 대비를 직접 말합니다.
5. **5.1 둘째 문단:** 주제문("the fix needs an iterating planner")이 첫 두 문장만 덮습니다. 후보 수, 표 크기, 공유 결과 앞에 run-in "Candidates, table size and sharing."을 두십시오. 한 줄이 늘므로 절약한 줄을 씁니다.
6. **5.2 run-in 제목:** "TD-MPC2: persistence dominates." / "DINO-WM: persistence does not dominate."처럼 결론을 말하는 꼴로 바꾸면 좋습니다. 늘어나는 길이는 남은 폭 안입니다.
7. **6장 순서:** "Number formats, training and weights"가 비트 한계와 새 잡음 비교 사이에서 흐름을 끊습니다. 6장 끝(DMC 문단 뒤)으로 옮기면 줄 변화는 없고, 그림 5 위치만 확인하면 됩니다.
8. **7장 둘째 문단:** 주제 다섯 개(FP32 대비, BF16 대비, 계획 단계 전체, BF16 손실, 다른 하드웨어)가 한 문단에 있습니다. "Pre-converted BF16 is nearly lossless ..."부터 새 문단으로 나누면 한 줄이 늘어납니다.
9. **더 줄여야 할 때:**
   - 4.1의 PushT 4-bit 상세 두 문장을 부록 I로 보내고 한 문장만 남기면 약 2줄이 줄어듭니다.
   - 서론 범위 문장의 "weights stay in floating point except in the integer kernel"은 한계 절과 겹칩니다.

## (3) 일관성 표

| 항목 | 변형과 위치 | 권장 |
|---|---|---|
| 성공률 단위 | points of success(87, 102), success points(178, 그림 2·5, 부록 D·I), points(대부분) | 178에서 "percentage points of success (points)"로 한 번 정의하고 이후 points |
| DMC 단위 | relative return(178), % of return(18곳), points or %(238, 부록 G) | % of return. 섞일 때는 "points or % of return" |
| red | 분수(240의 0.71·1.10 등, 표 19·21) / 백분율(193의 "median 95%", "75–104%") | 기호 red는 분수로만 쓰고, 백분율은 "removes N% of the damage"로만 |
| 부호 붙은 값 | loses +42 [31, 53](195), loses +12 [1, 23](268), +23.3 [14.4, 32.2] on pick-place(264) | loses·costs 뒤에는 부호 없이 쓰고, 부호 붙은 값은 "D = +42 [31, 53]" 꼴 |
| 음수가 낀 범위 | −0.00–0.24(208), −1.1–2.2(부록 G) / 0.93 to 0.95, −5.8 to 6.9%(표 13) | 음수가 끼면 "to"로 쓰고 "−0.00"은 "0.00"으로. 매크로에서 고침 |
| 개수 표기 | 3 of four(223), seven of these(264), two of the three(부록 E) | 비교하는 개수는 숫자로 |
| DMC 과제명 | cheetah-run/cheetah, walker-walk/walker, quadruped-walk/quadruped, finger-turn-hard/finger | 174의 "(we write ...)"에 DMC 약칭도 추가 |
| dither의 범위 | 102 "Dither, i.e. stochastic rounding"; 6장 "dither"에 det+shift 포함; 부록 D:3 | 102를 "e.g."로 바꾸고(같은 길이), 149에서 det+shift를 subtractive dither로 이미 정의함 |
| 지속 축 용어 | kept(표 1, 그림 1), fixed, held fixed, persistent, frozen(131) / redrawn, fresh, new every call | 뽑기는 fixed/redrawn, 오차장은 persistent, fresh noise는 det+noise에만 |
| 무작위 문턱 | randomized threshold(236, 238), random thresholds(부록 J) / unbiased randomized rounding(87, 104, 316) | randomized rounding |
| "separate" | 성분 분리(87, 104), 오차 구별(206), 구간(223) | 성분 분리에만 쓰고, 나머지는 distinguish 또는 "had an interval excluding zero" |
| "share" | drift share, persistent share, own share(부록 E ×5), main share(104), 동사 share | 정의된 두 지표에만 share. own share는 own part로 |
| 신뢰 하한 | lower confidence bound(221, 238) / lower bound(12곳) | lower bound |
| world action model | world action models(131 ×2) / world-action models(131) | 하나로 통일 |
| 약어 미정의 | MPPI, CEM, DMC, W4A4·W8A4·W8A8·W4A4KV4(131), VLA(부록 D), MPS(부록 C·K), RTN/SR(부록 G), QAT(부록) | 첫 사용에서 풀어 씀 |
| 기호 중복 | α: Holm 수준(178)·감쇠(부록 A·C)·비용 가중(부록 C:35); K: top-K(145)·사본 수(부록 G); ℓ: 층·상관 길이(부록 F:13·19); H: horizon·Hessian; 반복 수 I(145)·T(부록 F); n: n-step·2^k−1·n/N·pairs | 감쇠 η, 사본 수 M, 상관 길이 λ, 반복 수 I로 바꾸고 Hessian은 별도 기호 |
| x의 단위 | 102 "Δ⌊x/Δ+u⌋"(실제 단위) / 표 1 "x in grid units" | 한쪽에 맞춤 |
| 단계 라벨 | 178 "E1–E7"(E4·E6·E7은 174의 목록에 없음), 부록 C "E0" | 174 목록에 추가하거나 "the main study (E1–E7)" |
| 사후 실행명 | 부록 I:6 "runs D4, X5 and X17"(X6 누락) | cell D4, runs X5·X6·X17 |
| 내부 문서 참조 | §11, §11.1(표 2, 부록 A:90), v5.2(부록 E:21), coordinator(부록 K) | "Section 11 of the E11–E17 design document"로 쓰고 용어를 정의 |
| 철자·하이픈 | re-run(8)/rerun(3), "1- and 3-step"(표 13)/one-step | re-run, one-step |
| 큰 수 | 922k(225, 매크로) | about 922,000 |

## (4) 초록·서론

**평가.** 초록은 7/10입니다. 내용은 정확하고 경계 조건까지 들어 있지만, 구어 두 곳, 지시어 It, 해석 없이 덧붙인 끝 문장이 걸립니다. 서론은 6.5/10입니다. 질문의 틀과 기여 목록은 좋지만, 첫 문장 오독, 정의 전 사용, "main share", "These ideas are classical. They are ..."가 매끄러움을 깎습니다.

**초록** (+13자, 남은 폭 약 18자 안)
```
Sampling-based planners such as MPPI and CEM query a learned world model thousands of times per control step, so running them on low-bit hardware requires quantizing that model. We find that the rounding rule alone changes whether planning succeeds. With round-to-nearest activation quantization, planning loses up to \nv{e1.maxdet} points of success, and in most damaged settings the world model's latent one-step error is no larger than under stochastic rounding (dither). At a fixed bit width, deterministic rounding error damages sampling-based planning in two ways that controlled variants separate: a systematic error that any unbiased randomized rounding removes, whether fixed or redrawn, and an input-determined error field that persists across planner iterations and that the planner exploits. The persistent part dominates in TD-MPC2 at 5M and 317M parameters but not in DINO-WM, and this pattern followed the model family when we swapped the planners' update rules. Dither shared by all candidates and redrawn at every planner iteration repairs most damaged settings and lowers the usable bit width by at least one bit in most Meta-World tasks, even on per-channel grids. On Meta-World the unquantized policy alone nearly matches the planner, so there the gain means that low-bit planning stops hurting; on DMC the low-bit dithered planners still beat the policy. The simplest form of this dither adds at most about \nv{e17.shiter.overhead.max0}\% to rollout latency in an integer kernel. Dither helps less where the planner iterates little, as on DMC at the default number of iterations, where it can also hurt. With block-scaled NVFP4 activations, round-to-nearest damages only \nv{e9.nvfp4.ndam} of 12 TD-MPC2 tasks.
```
선택안(같은 길이): 비용 문장을 맨 끝으로 옮기면 경계(DMC, NVFP4) 다음에 실용적 결론으로 끝납니다.

**승인 문구 자체의 문제(고쳐 쓰지 않았고, 바꾸려면 재승인이 필요합니다)**
- "damages ... in two ways ...: a systematic error ... and an input-determined error field"는 ways와 뒤의 오차 명사가 맞지 않습니다. 예: "through two error components that controlled variants separate".
- "lowers the usable bit width"는 3.4에서 정의한 용어인 "the smallest usable bit width"로 쓰는 편이 맞습니다.
- "nearly matches the planner"는 "the unquantized planner"로 써야 정확합니다.
- "low-bit planning stops hurting"은 구어이고, 무엇에 비해 손해가 없는지가 모호합니다. 264행 "does not fall below the policy"와 표현을 맞추기를 권합니다.

**서론 1문단** (−20자)
```
A sampling-based planner calls its world model (candidates $\times$ iterations $\times$ horizon) times per control step: $512\times6\times3$ forward calls for the MPPI planner of TD-MPC2~\citep{hansen2024tdmpc2} and $300\times30\times5$ for the CEM planner of DINO-WM on PushT~\citep{zhou2025dinowm}. Moving such planners onto hardware whose fast arithmetic is low-bit, such as integer accelerators, FP4 tensor cores or memory-bound embedded devices, requires quantizing the world model. Quantized world models can break planning in ways that bit width and offline prediction error do not explain~\citep{fu2026wmquant, ranganath2026bits}, unless a calibration objective preserves what planning needs~\citep{anon2026modulemse}. We ask a narrower question: at a fixed bit width and grid, does the rounding rule matter, and if so, what does the planner exploit?
```
**서론 2문단** (+19자)
```
The rounding rule matters. Round-to-nearest (\dtm) is a fixed function of its input, so the error it adds to the predicted return of an action sequence is the same in every planner iteration. Dither, i.e.\ stochastic rounding $\Delta\lfloor x/\Delta+u\rfloor$ with $u\sim U[0,1)$, is unbiased and can be redrawn. With \dtm activation quantization on per-tensor percentile ranges, planning loses up to \nv{e1.maxdet} points of success. In \nv{e1.h1.n} of \nv{e1.h1.of} damaged cells (task $\times$ bit width), the latent one-step error of the world model is no larger than under shared dither, and in \nv{e1.h1.fix} of these \nv{e1.h1.n} cells that dither removes at least half of the damage (Section~\ref{sec:break}). Iterating planners have been reported to concentrate their candidates in regions where the model cannot rank them~\citep{anon2026elitecert, anon2026resolution}; we isolate which property of rounding error they exploit.
```
**서론 3문단** (+37자)
```
We build rounding variants that keep the per-entry error distribution and change only how the random draw is shared across candidates and how long it is kept, together with variants that remove the mean bias of \dtm, add it back to fresh dither, or add fresh noise to \dtm (Table~\ref{tab:variants}, Figure~\ref{fig:schematic}). They separate a systematic error, which any unbiased randomized rounding removes, whether its draw is fixed or redrawn, from an input-determined error field that persists across planner iterations and that the planner exploits. The first accounts for most of the damage in DINO-WM and the second in TD-MPC2 at 5M and 317M parameters; the pattern held when we swapped the planners' update rules, and we did not identify its cause.
```
**서론 4문단:** 항목 9의 세 곳만 바꿉니다(+4자).

**기여 목록**
- 1번: 항목 10을 적용합니다.
- 2번 전체:
```
\item \textbf{Two components} (Section~\ref{sec:mech}). In TD-MPC2, holding an unbiased error fixed across iterations costs success in all \nv{e11.pers.npos} damaged cells at 5M and in \nv{e14.p2.n} of \nv{e14.ndam} tasks at 317M, so the planner exploits persistence; on both damaged DINO-WM cells (Wall, and PushT at 3 bits post hoc), a fixed unbiased error repairs the damage. The pattern held when we swapped the planners' update rules.
```
- 4번: 항목 12를 적용합니다.
- 범위 문장: 항목 13을 적용합니다.

## (5) 내용 문제로 보이는 것 (목록만)

1. `app_F.tex:49`의 `(Section~\ref{sec:swap})`은 반복 수에 따른 결과를 가리키므로 `sec:persist`(5.1, 그림 3b)가 맞아 보입니다.
2. `main.tex:149`는 편향 벡터 추정 절차를 부록 G로 안내하지만, 절차는 부록 C("Bias vector")와 부록 A의 D2에 있습니다.
3. Holm 보정 범위의 서술이 세 곳에서 다릅니다: 178행 "over the cells of a task or stage", 부록 B:4 "within the cells of a benchmark", 부록 D:3 "over the bit widths of a task".
4. 부록 D:3은 "share one per-entry error distribution"이라 하면서 det+shift를 포함합니다. 표 1에서 det+shift의 분산은 Δ²/12이고 나머지는 Δ²/6입니다.
5. 262행과 부록 J:8의 "absent in DINO-WM (1 of 3)"에서, 한 칸은 하한이 0보다 큽니다. "absent"는 과장입니다.
6. 268행 "Ranges chosen to minimize ... do not change $k^*$"에서 누구의 k*인지가 빠져 있습니다. 그림 5 캡션으로 보면 rollout-chosen 범위에는 det만 있습니다.
7. 정책 단서(v2.2)가 세 가지로 표현됩니다: "stops hurting"(87, 316), "keeps the policy's level"(264), "stays above acting without a world model"(부록 I:284). 그리고 기여 3번에는 이 단서가 없습니다.
8. 결론(316행)의 "only redrawing removes"는 C2보다 넓은 주장입니다("only"가 없는 꼴을 권합니다).
9. 99행에서 "memory-bound embedded devices"가 "hardware whose fast arithmetic is low-bit"의 예로 들어가 있습니다. 동기 문장의 범주가 맞지 않습니다.
10. 298행의 "On other hardware low-bit arithmetic is much faster ..."는 GEMM 측정 하나와 동시대 보고 하나로 뒷받침하기에 일반화가 큽니다. 완곡하게 하려면 "can be much faster"인데, 이는 주장 강도를 바꾸는 것이므로 저자 판단으로 남깁니다.
11. 145행 "MPPI weights candidates"와 표 3 "MPPI weights the elites"가 다릅니다. TD-MPC2의 MPPI는 엘리트만 가중합니다.
12. 108행 기여 1번 문제 두 가지.
    - "PushT at 4 bits replicated only on the original evaluation pairs": 같은 쌍을 다시 돌린 것은 재현이지 복제가 아닙니다.
    - "new seeds and splits"를 TD-MPC2 칸에도 붙였는데, TD-MPC2에는 새 분할이 없습니다.
13. 부록 I:223의 "another GPU type ..., so the two splits are unpaired"는 논리가 어긋납니다. 짝이 아닌 이유는 평가 쌍이 달라서이지 GPU 때문이 아닙니다.
14. 부록 K:4 "No job of E11–E17 failed"와 부록 A:100의 "out-of-memory failures"가 어긋나 보입니다.
15. 부록 K:6 "design documents ..., two of them redacted"와 부록 A:4(본 설계 v2–v5.3 사본의 해시 불일치)의 개수가 맞지 않습니다.
16. 178행 "harmful if the 95% interval ... lies above 5 points"는 구간 전체(하한 > 5)를 뜻하는지 밝혀야 합니다.
17. 부록 K:8 "Design documents, logs and stage reports are in Korean"은 이중 맹검에서 저자를 추정할 단서가 될 수 있으니 저자가 판단하십시오.
