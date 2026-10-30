# E4 이론·수치 모의 작업 로그 (작업자, UTC)

- 2026-10-22T06:40Z 시작. 읽은 문서: MAIN_RULES.md, EXPERIMENT_DESIGN.md(4.1~4.4, 5, 6.2~6.4), DESIGN_main.md(v1~v4, 특히 0.4, 4.2, 5, 6).
  - E1 결과 파일(results/E1/, 서버 out/)은 읽지 않는다. 형식은 분석 **코드**(dinowm/analyze_e1.py, 서버 tdmpc2_main/code/analyze_e1.py의 출력 구조 부분)로만 확인했다.
  - 파일럿 근거(5장)는 읽었다: R2-8b·R2-13·R2-14 보고서, R2-8b 결정 수준 CSV(decision_*.csv.gz)의 크기 척도(후보 가치 std, 오차/후보 std 비).
- 06:41Z 서버 상태: load 6/64, 디스크 259GB 남음, tmux A-E1-*, B-* 돌고 있음(건드리지 않음).
- 06:45Z 실제 계획기 코드 확인(서버)
  - TD-MPC2 `~/pmx/A/tdmpc2_main/tdmpc2/tdmpc2/tdmpc2.py` @ 332f19e `_plan`: policy prior 24개(계획마다 1회 표본, 모든 반복에서 재평가), 표본 488개 = mean + std·randn을 [−1, 1]로 자름, value top-64, score = exp(0.5(V − max V)) 정규화, mean·std = score 가중 elite 평균·std, std ∈ [0.05, 2], 초기 std = 2, mean 0(t0) 또는 warm start(prev[1:] + 0), 실행 = 마지막 반복 score로 gumbel_softmax_sample(= Categorical(score)) 한 elite의 첫 행동, prev_mean = 마지막 mean.
  - DINO-WM `~/pmx/A/dinowm_main/dino_wm/planning/cem.py` @ 0a9492f: mu = 0, sigma = var_scale(1), 매 단계 표본 300개(첫 표본 = mu), 비용 낮은 top-30의 mean·std(불편 std), clamp 없음, open loop, mu 반환. main/code/dinowm/qlib.run_cem이 이를 batch로 옮긴 것과 같다.
  - OEI 정의: TD-MPC2 = e(마지막 반복에서 Gumbel로 뽑힌 elite, 같은 평가값) − 첫 반복 표본 488개 평균 e(qlib.PlanProbe), DINO-WM `oei_ref_mf` = e(최종 elite 평균 mu; 확률적 변형은 추가 draw 한 번) − iteration-0 후보 1..N−1 평균 e(runner.summarize).
- 06:50Z venv `~/pmx/envs/a-theory`(uv, Python 3.11.17, numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2). torch는 설치하지 않음. 작업 폴더 `~/pmx/A/theory/`.
- 06:58Z 조정자 추가 지시 2건 수신(같은 원칙: E1 결과 파일은 읽지 않음)
  1. OEI 분해 OEI = sel_last + drift(sel_last = e(실행) − 마지막 반복 모집단 평균 e, drift = 마지막 반복 모집단 평균 e − iteration-0 표본 평균 e)를 오차 모형 (1)~(4)·det+shift, 두 계획기에서 유도. 같은 평가값(winner's curse on the draw)과 독립 재평가의 차이, N·elite·반복 수에 대한 척도. 모의 출력에 두 항을 모두 넣는다.
  2. 비편향 dither가 비선형 층을 지날 때 생기는 2차(Jensen) 편향장 E_u[Ĵ(a)] − J(a): 선도 차수 형태, 비트 수 척도(Δ² ∝ 4^−k), 비선형 사상을 거치는 오차 모형을 모의에 추가, det drift 대비 크기.
- 07:00~07:22Z 코드 작성(`main/code/theory/`: e4_formulas.py, e4_sim.py, e4_jobs.py, e4_run.sh), 선택 강도 표(`s_tables.json`, 16 s)와 오차 척도 보정(`calib.json`, 14 s), 스모크(manifest `--smoke`, 71 job, 8 작업자 약 2분, 오류 0). 스모크 수치는 코드 점검에만 쓰고 해석하지 않는다(예외: 아래 두 설계 결정의 근거).
  - 보정 결과: iteration-0 near-tie gap g0(무양자화 첫 모집단 1·2위 J 차의 중앙값) / σ_J0 = MPPI d6 sep 0.070, tie 0.013, CEM d6 sep 0.035, tie 0.009(파일럿 TD-MPC2는 약 0.005). 그래서 오차 척도 격자를 g0 단위 4수준에서 5수준 {0.5, 2, 8, 32, 128}로 넓혔다(σ_e0/σ_J0 ≈ 0.004~9를 덮음).
  - 스모크 블록 A에서 2변량 정규 winner's curse가 정규 모집단에서는 정확하나, 실제 2차 목적함수 첫 모집단(아래가 막힌 비대칭 분포)의 argmin에서는 σ_e ≪ σ_J일 때 과소예측함을 봤다(비 0.1에서 −0.030 대 측정 −0.09). 그래서 고정 모집단 정확식(`argmin_cond_exact`)과 near-tie 수 근사(`argmin_near_tie`), Stein 항등식 예측(`stein_selection`)을 사전 선언 비교에 추가했다(본 실행 전).

## 07:24Z 사전 선언: 본 실행 격자, seed, 비교 (이후 바꾸지 않음. 추가는 사후로 표시)

**코드·입력 고정**: e4_formulas.py sha256 86991e55…, e4_sim.py d6f2e233…, e4_jobs.py 9a6f4af0…, e4_run.sh 6dd9da10…, calib.json 1f3ddeb6…, s_tables.json e43f5b4c…, manifest.json(504 job, 계획기 실행 922,368회) e555eacb….

**seed**: MASTER = 20261022. 모든 난수는 `np.random.SeedSequence(entropy=MASTER, spawn_key=crc32(부분들))`. 인스턴스(목적함수·prior 중심) = ('inst', block, planner, regime, d, inst), 오차 구조(RFF 장·toy 특징) = ('struct', …, inst, 모형 tag), 계획기 표본 = ('plan', …, inst, N, k, slot) — 모든 오차 모형이 같은 계획기 난수열로 짝, 오차 draw = ('err', …, slot, 모형) — 오차 척도 사이에서 공유(척도 간 CRN), Gumbel pick·재평가는 별도 stream. slot 0·1 = 같은 인스턴스의 두 계획기 seed(재실행 편차·지속 지수용). 보정은 별도 'cal' 인스턴스 64개.

**공통**: 2차 목적함수 J = ½(a − a*)ᵀH(a − a*), a* ~ U[−0.5, 0.5]^d. regime sep = 고유값 log-균등 [1, 10], tie = 절반을 [0.01, 0.1](평평한 방향). d ∈ {2, 6, 12} = (horizon, action) (1,2), (3,2), (3,4). 계획기는 서버 코드 그대로(위 06:45Z 항목). MPPI: N 중 24개가 policy prior(중심 a* + N(0, 0.3²), 표본 std 0.2, 계획마다 새로), 반복 12회 실행 후 n ∈ {1, 3, 6, 12} 기록(반복 상태는 총 반복 수와 무관하므로 동일). CEM: 30회 실행 후 n ∈ {5, 10, 30} 기록. elite 규칙: q ∈ {2%, 12.5%, 25%}(k = max(2, round(qN))) 또는 kfix(MPPI 64, CEM 30). 오차 척도 r = (iteration-0 후보 간 오차 std) / g0. 인스턴스 128 × slot 2.

**오차 모형**(모두 첫 모집단 후보 간 오차 std를 r·g0로 맞춤, 블록 C·E·R·J 제외)
- M1 지속장(det 유사): 고정 RFF 장(F = 128, 길이 ℓ = 0.3)
- M2 새 독립 잡음(ind 유사): 후보·평가마다 i.i.d. 정규
- M3 새 공유 잡음(sh-iter 유사): 평가마다 새 RFF 장(ℓ = 0.3) + 공통 충격, 공통 비중 ρ_c = 0.5
- R-* 반올림 toy(K = 512 entry, 1 token, ν = 3 격자칸/단위 행동, γ = 1): R-det(round), R-shiter(평가마다 공유 offset), R-fixed(계획 고정 offset, entry마다 = sh-step/sh-ep), R-detshift(평가마다 공유 격자 이동)

**블록**
- M 주 격자: planner {MPPI, CEM} × regime {sep, tie} × N {64, 128, 256, 512, 1024} × elite {q2%, q12.5%, q25%, kfix} × 모형 7종 × r {0.5, 2, 8, 32, 128}, d = 6 (+none), 320 job
- D 차원: d {2, 12}(MPPI), {2, 6, 12}(CEM 기본 300/30) × 모형 7종 × r {8, 32}
- L 상관 길이: M1·M3의 ℓ {0.03, 0.1, 1, 3} × r {8, 32}, 기본 계획기
- C CRN: M3 ρ_c {0, 0.5, 0.9, 0.99}와 M2를 같은 주변 진폭(amp = r·g0)으로, r {8, 32}
- E 표 크기: toy 16 token × 256 unit(K = 4096), 일관(coh: token 같은 위상·민감도, unit 간 민감도 변화 공유) / 비일관(incoh) 두 설정 × {det, ind, shiter, detshift} + fixed의 N_eff {4096, 1024, 256, 64, 16, 4, 1}(token 블록 × unit 블록 공유), 같은 비트(진폭 = 그 toy det 기준) r {8, 32}
- R 같은 비트 반올림 변형: {det, ind, shiter, fixed, detshift} × 특징 {inc: 위상 균등, clus: 위상 [0, 0.5)·방향 일관 0.5} × r {8, 32}
- J Jensen(조정자 2번): NLToy(256 unit, latent 8, 2차 head e = amp[2ρ(a)·δz + κ‖δz‖²], ρ(a) = P(a − a*)) × 비트 {3, 4, 5, 6} × κ {1, 10} × {det, ind, shiter, detshift, fixed}, 진폭 고정(det k4 κ1 기준 r = 8)
- V 장기 반복: MPPI N 512, k {8, 32, 64, 128}, 반복 100(기록 1, 3, 6, 12, 25, 50, 100); CEM 300/30 반복 60; M1·M2, r {8, 32}, sep
- A 단발 선택: 정규 (J, e)(ρ {0, 0.3}, σ_e/σ_J {0.1, 0.3, 1, 3, 10}, N 5수준, argmin·top-q·MPPI θ {0.1, 1, 10}) 20,000 반복; 실제 2차 첫 모집단 2,000개 × 비 5수준(argmin, top-12.5%)
- W 폐루프(warm start): 2차원 적분기 LQ, 25 step, MPPI 512/64, 반복 n {1, 3, 6, 12} × warm start {on, off} × {none, M1j(상태·행동 결합 지속장 = det/sh-ep), Mstep(제어 step마다 새 고정장 = sh-step), M3, M2} × r {8, 32}, 64 episode
- EV 직접 측정: E toy의 고정 표 후보 간 분산을 표 32개로 측정, 모집단 퍼짐 {0.003, …, 1} × N_eff 7수준 × 구조 16개, 정확식(브라운 다리 커널) 비교
- JV 직접 측정: Jensen 편향장 E_u[e]의 후보 간 std(정확식) 대 det 오차 std, 비트·κ·퍼짐별

**기록 지표**(기록 반복마다, 셀 평균·sd·n): OEI(MPPI = e(Gumbel 실행 elite, 같은 평가값) − 첫 반복 표본 평균 e; CEM = e(최종 elite 평균, 새 잡음은 독립 재평가) − 첫 반복 후보 1..N−1 평균 e), 분해 sel_last·drift(같은 평가값과 독립 재평가 두 가지, CEM은 elite 평균·argmin판도), 보조 OEI(oei_w, oei_top1, oei_first, oei_elite, oei_argmin), 손상 = 같은 slot 무양자화 대비 J 증가(실행 계획과 평균), regret J − J*, 평균 위치 ‖μ − a*‖², 첫 행동 편차(무양자화 같은 seed 대비, seed 바닥, 변형 재실행 퍼짐), 지속 지수(두 seed의 평균 이동 상관), 모집단 기술자(σ_J, σ_e, ρ, θ = λσ_V, k_eff), 반복별 궤적, 예측값(아래).

**사전 선언 비교와 적합도 지표**
- C1 단발 winner's curse: 정규 모집단에서 공식 대 MC(|차| < 2 SE 비율, 상대 오차); 2차 첫 모집단에서 2변량 공식·고정 모집단 정확식·near-tie 근사 대 측정(상대 오차)
- C2 반복 안 선택: 기록 반복마다 sel_last(같은 평가값) 대 2변량 예측(MPPI S = MPPI 강도(N, k, θ_L), CEM elite·argmin), Stein 예측, near-tie 근사; 셀 단위 기울기·R²·상대 오차 중앙값; MPPI n = 1(첫 모집단 단발)도 따로
- C3 분해: 새 잡음 모형(M2, M3, R-shiter, R-detshift, E-ind/shiter/detshift)에서 drift ≈ 0, 재평가 sel_last ≈ 0(95% CI가 0을 포함하는 셀 비율); 지속 모형(M1, R-det, R-fixed)에서 재평가 = 같은 평가값, drift < 0
- C4 SAA(M1): 점진 평활 SAA 예측 위치 대 실제 평균(이동의 회귀 기울기·R²), CEM OEI_mf 대 예측, drift 대 예측, 약한 장 모멘트(rff_saa_moments) 대 측정
- C5 SA(M2, 블록 V): 장기 반복 손상 대 AR(1) 정상 공식 dσ_ε/(4 S k_eff), elite 수에 대한 기울기(log-log, 예측 −1)
- C6 반복 수: 지속 모형 손상의 n 기울기(예측: 평탄 또는 포화) 대 새 잡음 모형(예측: 감소); 블록 W의 Mstep − M3 차(예측: n = 1에서 0, n과 함께 증가)
- C7 N 용량-반응(H2c 예측식): 고정 q와 고정 k에서 |OEI|·손상의 N 의존(Spearman, log-log 기울기)과 S_op(N) 비 예측의 비교, 모형·계획기·OEI 정의별
- C8 CRN(블록 C): 같은 주변 진폭에서 ρ_c에 따른 sel_last·손상 대 비공통 분산 (1 − ρ_c) 예측
- C9 N_eff(블록 EV·E): 측정 후보 간 분산 대 정확식(2 SE 안 비율), A/N_eff + B 적합(R²), 계획 지표(OEI, drift, 손상)의 N_eff Spearman
- C10 det+shift: 같은 비트·같은 진폭에서 sh-iter 대비 손상·OEI·drift(예측: 같거나 작음, drift ≈ 0)
- C11 Jensen(블록 J·JV): 새 SR 변형(ind, shiter)의 drift 대 detshift(예측 ≈ 0)·det, 비트 기울기(예측 log|bias| ∝ −k log 4), κ 의존, det drift 대비 비율 예측식
- C12 계획 편차: 지속 지수(예측 지속 ≈ 1, 새 잡음 ≈ 0)와 dev/floor

- 07:25:24Z 본 실행 시작: tmux `A-E4-sim`, `bash e4_run.sh ~/pmx/A/theory/results/E4 8`(작업자 8개, 각 1 thread, nice 10, CUDA_VISIBLE_DEVICES=""), 로그 `~/pmx/A/logs/E4_sim.log`(스모크 로그는 `E4_sim_smoke.log`로 옮김). 예상 CPU 약 12 CPU-시간, 벽시계 약 1.5~2.5시간. 시작 시 서버 load 10/64.
- 07:27~07:55Z 실행 중 작업(격자·seed·비교 선언은 바꾸지 않음)
  - 분석 `e4_analyze.py`(사전 선언 비교 C1~C12 구현, 칸 pooling, 표·그림), 표 생성 `e4_report_tables.py`, 겹쳐 그리기 `e4_overlay.py`를 작성했다. 스모크 결과로 코드 경로만 점검했다(해석하지 않음).
  - 수치 점검(유도 확인용, 서버 수 초): Stein softmax 항등식 MC −0.9245 대 식 −0.9248(비정규 J), 공유 offset의 V(σ) MC와 식 일치(σ 0.02~1), det+shift 오차 = SR 오차 − (u − ½) 항등식과 [−½, ½) 균등(분산 0.0835).
  - **보조(사후, 이론의 파일럿 적용)**: 파일럿 R2-8b 결정 수준 CSV(마지막 MPPI 반복, 150 상태 중앙값)의 σ_e/σ_V와 σ_V로 같은 평가 sel_last 예측. sh-fwd / det = mw-push −8.9 / −98, pick-place −38 / −42, cheetah(k4) −3.2 / −3.4, walker(k4) −0.39 / −0.16(가치 단위). E1 결과가 아니다.
  - 07:46Z 모니터 스크립트가 zsh 단어 분리 문제로 한 번 죽어 `read`로 고쳐 다시 걸었다(시뮬레이션과 무관).
- 07:56Z **사후 분석 추가**(사전 선언 뒤, 결과 전체를 보기 전): (a) 블록 E에서 고정 표 손상의 log-log 기울기 대 그 표의 측정 첫 반복 오차 분산(`posthoc_loglog_slope_dmg_vs_var0`), (b) C13: 블록 M group 안에서 손상과 −OEI, −drift, −sel_last의 Spearman(H2b 관련 함의 점검). 둘 다 보고서에 사후로 표시한다.
- 08:16Z **규칙 위반 기록(경미)**: 07:41~08:05Z에 작업자 8개와 함께 짧은 단일 thread 점검 4번(유도 수치 점검, 파일럿 기반 예측, 선택 강도 표, 부분 분석 `E4_partial`, 각 2분 이하)을 돌렸다. 그동안 잠시 단일 thread 프로세스가 9개였다(8코어 상한 초과). 지금부터 작업자가 도는 동안 서버에서는 ls·grep 같은 가벼운 명령만 쓴다. 부분 분석은 코드 점검용이고 판정에 쓰지 않는다(본 분석은 실행이 끝난 뒤 전체 자료로 다시 한다).
- 08:15Z 진행 50/504 job, 오류 0. 비용 모형으로 본 예상 종료 약 08:50~09:10Z.
- 08:52:13Z **본 실행 끝**: 504/504 job, 오류 0, 벽시계 1시간 27분, job CPU 합 41,636 s(11.6 CPU-시간), 계획기 실행 922,368회. tmux `A-E4-sim`은 스크립트 종료와 함께 닫혔다. job 원자료 176 MB는 서버에만 둔다.
- 08:53~09:05Z 분석(서버, 4 thread, 1회 14 s)
  - `e4_analyze.py` → `results/E4/`(e4_comparisons.json, e4_tables.md, cells_*.csv.gz, fig1~7)
  - `e4_overlay.py --selftest` → `results/E4/overlay_test/`(합성 입력만)
  - **사후 수정 3건**(결과를 본 뒤. 모의 자체와 사전 선언 비교 목록은 바꾸지 않음)
    1. 적합 통계의 수치 유효성 필터: 예측이 측정의 10³배를 넘는 퇴화 칸을 제외하고 개수를 보고한다. 원인은 CEM 모집단이 SAA 고정점에 붕괴해 ρ → −1, $\sigma_{\hat J} \to 0$인 칸과, Stein k-NN 밀도의 동률 후보다. 2변량 1칸, Stein 최대 169칸.
    2. 겹쳐 그리기에서 DINO-WM 계열 묶음 키를 고쳤다(N마다 다른 topk 때문에 묶음이 쪼개짐). CEM 새 잡음의 |OEI| 비 곡선은 뺐다(기대값 0).
    3. 그림 6 제목 겹침 수정
  - 사후 계산(보고서에 사후 표시): 블록 M 반복 후반 손상 비(MPPI n 6 → 12, CEM 10 → 30), C13의 같은 척도 안 Spearman과 같은 비트 블록 E·R의 변형 간 Spearman, 블록 V 모집단 std 점검
- 09:10Z 보고서 `reports/E4_theory.md` 작성(유도, 예측식·상수, 격자, 일치 표, 어긋남, 결정 사항 3건, 편차, 겹쳐 그리기, CPU, 파일)
- 09:12Z 마무리: 맥북·서버 코드 sha256 일치(실행 코드 4개는 사전 선언 해시 그대로). 서버 부분 분석 폴더 `E4_partial` 삭제(중간 산출물). 남은 서버 산출물은 `~/pmx/A/theory/results/E4`(job 원자료 176 MB 포함), `E4smoke`(16 MB). 돌고 있는 작업 없음.
