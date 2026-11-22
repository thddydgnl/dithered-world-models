# td 묶음(TD-MPC2 5M) — 실행 계획 (본 실행 전)

- 작성: 2026-11-21T18:50~19:20Z, td 작업자. 운영 선택과 분석 규칙(§1)은 본 실행 driver 시작(18:48Z) 전에 정했고, 그때 본 실행 결과는 td.5의 none(기본 계획기) 360 episode뿐이었다(§0.7 점검 3의 입력). 점검 결과·처리량·비용(§2~§5)과 편차 D2의 반영은 그 뒤 실행 중에 채웠다. det+BC·sh-iter+b·E13의 결과는 D2 결정(19:01Z)과 관문 개방(19:05~19:08Z) 전에는 하나도 없었다.
- 기준: [../DESIGN_tmlr.md](../DESIGN_tmlr.md) v1.0(sha256 `36d69ea8…0cc4`)의 §0, §1(E11), §2(E12), §3(E13), §5(E15), §6(E16), §8. 가설·지표·문턱·칸·seed는 설계 그대로이고, 이 문서는 운영 세부만 정한다. 공통 규칙은 [../WORKER_RULES.md](../WORKER_RULES.md).
- 코드: [code/](code/), 기록: [LOG.md](LOG.md), 결과: `results/raw/<tag>/`(조정자가 30분마다 `pull_all.sh`로 당겨 옴)

## 1. 구현 선택

### 1.1 기준 코드와 바꾼 곳
- E9 `rss_ext/E9_fp4/code/` 전체를 복사했다(설치 `setup_td.sh` = `setup_e9.sh`와 같은 고정판, pod 배치만 `/workspace/atm/{code,out,big,logs}`). E1 `calib_lib.py`·`e5lib.py`를 그대로 더했다.
- `qlib.py`: 새 변형 넷을 hook의 기존 분기 **뒤에** 더했다. 기존 분기는 한 글자도 바꾸지 않았고, 표 변형·확률적 변형 목록(tuple)만 넓혔다. 점검 1(§2)로 비트 단위 동일을 확인했다.

| 변형(코드 이름) | 출력 | 난수 |
|---|---|---|
| det+BC (`detbc`) | lo + Δ·clamp(round(x), 0, nl) − b | 없음. b는 `Quantizer.bias[층]`((W,) 벡터) |
| det+shift-fixed (`detshiftfix`) | lo + Δ·(clamp(round(x − o), 0, nl) + o), o = u − ½ | u는 (층, horizon 위치)마다 한 줄. episode 시작(`set`)에서 한 번 뽑고 `begin_step`은 그대로 둔다(sh-ep 표와 같은 일정, 같은 개별 generator) |
| sh-iter+b (`shiterb`) | lo + Δ·clamp(⌊x + u⌋, 0, nl) + b | u는 sh-iter와 같다((1, W), forward마다, 후보 공유). b는 같은 (범위, k)의 det+BC b |
| det+noise (`detnoise`) | lo + Δ·(clamp(round(x), 0, nl) + n), n = u − ½ | u는 sh-iter와 같은 일정 |

- 범위
  - PT: E0 범위 파일 그대로(`code/inputs/ranges/`, E1·E9와 같은 격자).
  - PC: pod에서 보정 episode 0~4(seed 11000~11004, E0 규약 `record_calib_episode`)를 다시 기록하고, E0의 12,000행 부분표본(numpy `default_rng(1)`) × horizon 3 step의 활성 36,000행에서 단위별 [min_j, 4번째로 큰 값(= 99.99 백분위, E5 `perchan`)], hi_j ≥ lo_j + 1e-6.
  - 같은 자료로 다시 잰 per-tensor 범위와 E0 파일의 차를 prep 결과(`out/jobs/prep/`)에 적는다(스모크 mw-push s1: hi 차는 E0 폭의 0.6~1.7%, lo 차 0).
- 편향 벡터 b: (과제, 체크포인트, 범위 종류, k = 1~8)마다 별도 job `bias`, **편차 D2의 규칙**(`runpod_tmlr/bias_rule_D2.json`, 각 pod `/workspace/atm/bias_rule.json`): α = 1 → ½ → ¼ 단계, 단계마다 b = 0에서, pass마다 m(보정된 det 출력 − 비양자화 출력의 단위별 평균, 12,000행 × 3 step, 모든 층 동시)과 r = mean_j |m_j|/Δ_j를 재고 모든 층 r < 1%인 첫 pass의 b로 확정, 아니면 b ← b + α·m(단계마다 갱신 최대 60). 세 단계 모두 못 미치면 단계 마지막 b 중 최악 층 r이 가장 작은 b + `bc_unconverged`. 관문(`/workspace/atm/GATE_bias`)은 결정 전까지 닫혀 있었다. 칸마다 단계·갱신 수·수렴·표본 수, pass별 층별 평균·최대 |m_j|/Δ_j(PC는 mean|m|/mean Δ도), 최종 b와 m₀(`out/prep/<task>_s<c>_bias.npz`), mean_j |b_j|/Δ_j, b = 0과 최종 b의 층별 MSE(/Δ² 포함)와 전파 잠재 오차를 남긴다.

### 1.2 seed, job, 재사용
- eval seed 100000~100029 × 체크포인트 1·2·3(90짝), held 101000~101029(E13), 보정 11000~11004, 스모크 109000+, 처리량 108000+(해석하지 않음).
- job = (과제, 체크포인트, 칸, 계획기 태그) × episode 5개(chunk c = episode 5c~5c+4). chunk 0~2가 DMC 두 단계 표본의 첫 45짝(seed 100000~100014), chunk 0~3이 E15 반복 24의 60짝이다.
- 칸 = (변형, 범위 종류, k). 계획기 태그: `base`(기본), `it12`·`it24`(E15), `cem`(E16).
- 재사용: R1(E12의 PT 1차 k 칸 = E11 칸), R2(E15 반복 6 = E11 DMC 칸), R3(E16 기본 계획기 = E11), R5(none은 (과제, 체크포인트, 계획기 태그)마다 한 번). 모든 칸의 none과 변형이 같은 pod에서 돈다(짝 규칙).
- 결과: job마다 `out/jobs/<type>/<id>.json.gz`(임시 파일 → rename), 있으면 건너뛴다. 보정 파일과 lock은 `big/`.

### 1.3 계측
- E11의 기전 과제(mw-push, mw-pick-place, cheetah-run)는 none 포함 모든 칸에 E2의 PlanProbe 계측을 켠다: OEI 상태 MW 4 step·DMC 10 step마다(drift, sel_last, oei 등)와 독립 draw 재평가(oei_indep, sel_indep). 설계가 적은 셋(drift, 선택 항, 독립 draw 재평가)만 켜고, E2의 계획 재실행(pd_stride)과 영역 효과 기준 계획(region_ref)은 켜지 않는다. PlanProbe는 closed loop를 바꾸지 않는다(E1·E2 확인; 스모크에서도 필드 확인).
- 그 밖의 칸은 계측 없음.

### 1.4 E12 적응 규칙 (driver가 스스로 적용, `plan_td.adaptive_jobs`)
- 두 단계 표본: DMC E12 칸(R1 칸 제외)은 45짝(chunk 0~2)을 먼저 돌고, 그 45짝의 D의 95% 짝 bootstrap CI(10,000회, RandomState(0))가 τ = 0.10을 포함하면(lo ≤ τ ≤ hi, 허용 오차 1e-9) 나머지 45짝을 더 돈다.
- 확장: (과제, 범위, 반올림)마다 잰 k 가운데 가장 낮은 k의 칸이 끝났고 D ≤ τ면 k − 1을 더하고(k = 1까지), 가장 높은 k의 칸이 끝났고 D > τ면 k + 1을 더한다(k = 8까지). 칸의 D는 그 칸이 가진 모든 짝으로 계산한다. "끝남" = MW 90짝, DMC는 두 단계 결정까지 끝난 상태.
- 결정마다 `out/driver/decisions.jsonl`에 남는다. driver를 다시 띄우면 결과에서 같은 결정을 다시 만든다(결정은 결과의 함수).

### 1.5 E13 (held 상태 진단)
- held none episode(기본 계획기, 계측 없음)를 돌리는 동안 통과 기록기가 OEI 상태(MW 4 step·DMC 10 step마다)의 첫 반복과 마지막 반복 후보 집합을 보관하고, E1 규칙대로 고르게 떨어진 8상태를 쓴다(MW t = 0, 12, 28, 40, 56, 68, 84, 96; DMC 0, 70, …, 490).
- 칸: E11 8칸, E1 3칸 × 6변형(+ none, 오차 0이라 계산 생략), E12의 1차 k 주변(PT k1 ± 1의 det+BC, PC k1 − 1~k1 + 1 × det·det+BC·sh-iter·det+shift). 중복을 빼면 과제당 36칸.
- 지표(두 집합 모두)
  - 잠재 err1·err3·err5·err1h와 E1 errr: `qlib.pred_errors` on `qlib.plan_inputs`(K = 5) — E1 `perr_plan`과 같은 계산
  - rew_med: 비양자화 rollout의 잠재 z_0,j(j < 3)에서 1-step 보상의 |r_q − r_0| / (그 상태·step의 후보 512개 r_0 표준편차), (상태, step, 후보, dither seed)를 모두 모은 중앙값
  - 후보 return 오차 e = J_0 − J_q(같은 torch seed = 같은 Q 쌍·정책 잡음의 가치 추정): 공통 몫(평균, common_frac = ē²/mean(e²)), σ_e, σ_J, σ_e/σ_J — 표본 후보 488개(정책 prior 24개 제외, E1과 같음)
  - ρ_p: 같은 집합을 연속 두 번 평가(양자화기 상태가 이어짐: 새 draw는 진행, 표·det는 그대로; E1 rho_iter)했을 때 e의 상관(표본 후보)
  - 순위 보존: Spearman(J_q, J_0)(후보 전체), elite 64개 겹침
  - 확률적 변형은 E1 dither seed 3개(9,000,001 + 1000·seed + j) 평균
- Dg1의 예측 변수: P1 = (마지막 집합 ρ_p의 칸 평균) × (마지막 집합 σ_e/σ_J의 칸 평균), P2 = 첫 집합 err1h(E1 H1의 1-step 오차). 칸 값 = held episode별 평균(8상태)의 평균.

### 1.6 E15·E16
- E15: `set_planner({"iterations": 12 | 24})`, none은 반복 수마다 따로 돈다. 결정당 계획 시간은 대기열 끝에 혼자 도는 `timing` job(GPU에 다른 일이 없을 때, 한 프로세스, CUDA 동기화, 반복 6·12·24 × none·det·sh-iter·det+shift, 60 결정)으로 잰다. episode의 secs_loop(8 워커 경합)도 함께 남긴다.
- E16: TD-MPC2 332f19e `_plan`은 `score = exp(temperature·(V − V_max))`, `score /= score.sum()`이다. temperature로 나누는 곳이 없으므로 **temperature 0이면 모든 elite의 score가 exp(0) = 1, 정규화 뒤 정확히 1/64**다(0으로 나누기 없음, 코드 수정 없음). 정책 궤적은 `num_pi_trajs > 0`일 때만 만든다. 그래서 `set_planner({"temperature": 0.0, "min_std": 0.01, "num_pi_trajs": 0})` override만으로 한다. 스모크에서 마지막 Gumbel 추출에 넘어가는 가중이 모두 0.015625임을 확인했다. 최종 행동은 기본 코드대로 elite 중 (균등) Gumbel 추출이다.

### 1.7 분석 규칙의 세부 (`code/analyze_td.py`, 본 실행 전 작성)
- 통계: DESIGN_tmlr §0.3. 손상 = 단위 안 Holm(α 0.05, 단측 p = (#{D* ≤ 0} + 1)/(B + 1))과 D_det > 0. red CI는 D_det* > 0인 재표본. 모든 문턱 비교에 1e-9.
- E11 칸 분류(손상 칸마다): 편향(det+BC red ≥ 0.5) → 아니면 지속(Δ_pers CI 하한 > 0) → 아니면 해상도(det+shift-fixed red ≥ 0.5이고 Δ_pers CI가 0 포함) → 아니면 그 밖. 단위 읽기 = 손상 칸의 과반(절반 초과)이 같은 분류면 그 분류, 아니면 혼합, 손상 칸이 없으면 판정 불가. 이 순서 판정은 설계의 세 조건(서로 배타)과 같다.
- B3(TD 몫): det+noise가 고친(red ≥ 0.5) 손상 칸 수 ≤ 손상 칸의 절반이면 성립. B4의 "해로운 칸 없음"은 E1 H3a 규약(MW 8칸 전체)으로 판정하고 손상 칸만 읽은 판정도 함께 낸다.
- E12 k\*: 잰 k 가운데 D ≤ τ이고 그보다 큰 모든 잰 k에서도 D ≤ τ인 가장 작은 k. 그런 k가 없으면(k = 8까지도 D > τ) k\* = 9("> 8")로 두고 G를 그대로 계산한다(G는 det 계열과 dither 계열이 모두 9이면 0). 분석은 확장 규칙이 제대로 적용됐는지도 다시 확인한다.
- E15: 반복 n에서 best = red가 큰 쪽(sh-iter, det+shift; D_det ≤ 0이면 red 없음). It1의 "해로운 과제"는 best 변형이 해로운 과제(D_best − D_det CI 하한 > 0.05). It2의 단조 증가는 red_best(6) < red_best(12) < red_best(24)(엄격, red가 없으면 단조 아님).
- E16: 자격 = CEM식 none 평균 ≥ 0.6 × 기본 none 평균(MW 성공률, DMC return).
- 산출물: `td_summary.json`(판정과 모든 칸), `td_e11_cells.csv`, `td_e12_cells.csv`, `td_e13_cells.csv`, `td_episodes.csv`, `td_tables.md`.

## 2. §0.7 점검 결과 (td.5, A40)

1. **코드 경로: 통과.** 손대지 않은 E1 `qlib.py`(md5 7934e1b2…)와 새 `qlib.py`가 같은 pod에서 mw-push k = 3과 cheetah-run k = 5의 none·det·sh-iter·sh-ep·det+shift 10 episode를 return, 성공, n_refresh, 행동열·보상열 sha256까지 같게 냈다(`smoke_out/e1paths_*.txt`). runner job 경로도 같은 return이다.
2. **새 변형 단위 점검: det+BC 외 통과, det+BC는 실패(결정 대기)**
   - det+shift-fixed: 표가 episode 안에서 고정, episode마다 바뀜, seed로 재현. 원소별 오차의 평균 차 ≤ 0.003 sd, 분산 상대 차 0.1~1.1%(128표 대 128 draw; mw-push PT k3·PC k2, cheetah PT k5)
   - sh-iter+b: 같은 seed에서 (출력 − sh-iter 출력) = b(최대 오차 4.5e-7), 오차 평균 차 = b, 분산 상대 차 < 1e-8
   - det+noise: 더한 잡음 분산 / (Δ²/12) = 1.00005, det 부호가 det와 비트 단위로 같다
   - 모든 새 변형: hook 호출과 계획 한 step 뒤 CPU·CUDA 전역 난수 상태가 none과 같다
   - E16: temperature 0의 elite 가중 = 1/64(정확), num_pi_trajs 0에서 정책 prior 호출 없음, `set_planner(None)`이 기본값으로 되돌린다
   - **det+BC: 설계 규칙(b ← b + m, 3 sweep)은 수렴하지 않았고(아래), 편차 D2의 규칙으로 본 실행 576칸이 모두 수렴했다 → 최종 b에서 통과**(D2: 점검 2는 최종 b에서 읽는다). α = 1 34칸, α = ½ 540칸, α = ¼ 2칸, bc_unconverged 0칸, 최대 갱신 39, 최종 잔차 최대 0.0100(< 1%) (`<pod>:/workspace/atm/out/jobs/bias/*.json.gz`). 설계 규칙의 진단: 스모크 4과제 × PT·PC × k 1~8 = 64칸에서 3번째 sweep 뒤 층 최악 평균 |m_j|/Δ_j는 최대 1.90(cheetah PC k6), 거의 모든 칸에서 sweep마다 커진다. 진단(`smoke_out/bias_diag.json`, `bias_diag2.json`)
     - Jacobi(설계대로) 6 sweep: 발산 또는 정체
     - Gauss–Seidel(층 순서대로 갱신): 진동, 6 sweep 뒤에도 1% 미만이 드묾
     - 같은 입력의 국소 반올림 편향(전파 없음): 2~6%에서 멈춤
     - **감쇠 η = ½: 64칸 모두 3~14 sweep에 모든 층 1% 미만**
   - 조정자 지시(18:4xZ)로 결정 전까지 det+BC·sh-iter+b episode와 E13(b가 필요)을 관문으로 막고 나머지를 진행했다. 결정 D2(19:01Z) 뒤 6 pod에 규칙을 쓰고 관문을 열었다(19:05~19:08Z). 구 규칙 b로 돈 스모크·처리량 episode는 버렸다(LOG).
3. **기준 성능(none, 새 seed 100000~100029, 기본 계획기)**: 기준은 E1 none(`main/results/E1_tdmpc2/e1_tdmpc2_summary.json`의 baseline; `e1_tdmpc2_cells.csv`에는 none 행이 없고 none 평균 = 변형 행의 mean + D(MW) 또는 mean/(1 − D)(DMC)이며 모든 행에서 같은 값이 나온다, `code/inputs/e1_none_baseline.json`). 문턱: MW 10점, DMC E1 return의 10%. 90짝이 다 있어야 판정한다(`code/check_baseline.py`).
   - td.5(driver 전, 18:39~18:44Z): mw-push 1.000(E1 0.989), mw-pick-place 1.000(0.989), mw-door-open 1.000(0.967), mw-drawer-open 1.000(1.000) → 통과
   - td.6, td.1~td.4: driver의 첫 job(none)으로 쟀다(18:53~18:57Z): button-press 1.000(E1 1.000), peg-insert 1.000(1.000), window-open 1.000(1.000), sweep-into 0.967(1.000), cheetah 900.9(900.8), walker 984.6(983.5), quadruped 971.7(975.5), finger 969.1(977.9) → **12과제 모두 통과**(`<pod>:/workspace/atm/out/check3_<tag>.json`).
4. **처리량: §3. 워커 8로 정함.** 고친 td 추정은 약 $13.5(§5)로 $57과 거리가 멀다.

## 3. 처리량과 워커 수
- td.5(A40, CPU 할당 7.65코어) 고정 시간 측정(처리량 seed 108000+, 1-episode job, 본 실행의 변형 섞음; `td.5:/workspace/atm/bench/*/rate.json`)

| 묶음 | 워커 8 | 워커 10 |
|---|---|---|
| MW(mw-push 계측 + mw-pick-place) | 4,647 ep/h(episode당 5.8~6.5초) | 4,170 ep/h(8.0~9.4초) |
| DMC(cheetah 계측 + walker, 반복 12·24 포함) | 657 ep/h(1-episode job; episode당 반복 6: 30.5~33.8초, 12: 48.5~49.1, 24: 93.9~94.8) | — |

- 워커 10은 CPU 할당을 넘어 느리다(E9의 워커 12와 같은 결론). **워커 8**로 정했다. 같은 기종·같은 CPU 할당에서 E9가 워커 8을 최적으로 쟀으므로, 측정이 끝나기 전에 td.1~td.4·td.6 driver를 워커 8로 먼저 띄웠다(결과는 워커 수와 무관).
- 본 실행(5-episode job, driver 로그 18:57Z): MW td.6 5,372 ep/h; DMC td.1 1,011(cheetah, E11 계측), td.2 1,063, td.3 1,095, td.4 1,083 ep/h.
- 반복 수에 따른 결정당 계획 시간(스모크 timing, 8 워커 경합 중이라 참고): 반복 6 none 56 ms, 12 99.5 ms, 24 169.5 ms. 본 측정은 대기열 끝의 timing job이 한다.

## 4. pod 배치와 예상 종료 시각 (UTC)

| pod | id | 생성 | 자동 종료 | 과제 | 실험 | static job / episode | 예상 끝 |
|---|---|---|---|---|---|---|---|
| td.1 | pvfh339jrgqzr0 | 18:44:38 | 02:43:55 | cheetah-run | E11, E12, E15, E16, E13 | 592 / 2,850 | 약 22:10~22:45 |
| td.2 | sbzf2oyp9xjwpb | 18:44:57 | 02:44:38 | walker-walk | E11, E12, E15, E13 | 502 / 2,400 | 약 22:10~22:40 |
| td.3 | ddtsi386w98i6k | 18:45:35 | 02:44:58 | quadruped-walk | E11, E12, E15, E13 | 502 / 2,400 | 약 22:05~22:40 |
| td.4 | hqnnacntu4arrv | 18:46:16 | 02:45:36 | finger-turn-hard | E11, E12, E15, E13 | 502 / 2,400 | 약 22:00~22:40 |
| td.5 | hzcu21hf6vx1ka | 18:28:53 | 02:28:28 | push, pick-place, door-open, drawer-open | E11, E12, E16(push·pick-place), E13 | 2,712 / 13,140 | 약 22:00~22:55 |
| td.6 | tds0okvepphzoi | 18:43:55 | 02:43:13 | button-press, peg-insert, window-open, sweep-into | E11, E12, E13 | 2,532 / 12,240 | 약 21:45~22:40 |

- 예상 끝은 본 실행 처리량에 E15 반복 비용(12회 ×1.6, 24회 ×3.1), E12 적응 칸(DMC 두 단계·확장 +15~20%, MW 확장 +5%), E13 계산(held episode당 약 55초, MW pod에서 약 40분)을 더한 값이다. 관문은 19:05~19:08Z에 열려 기다린 job이 없다. MW pod는 E13 때문에 표의 값보다 약 20분 늦어질 수 있다.
- 모든 pod가 자동 종료 전에 4시간 이상 여유가 있다. 같은 (과제, 체크포인트)는 한 pod에서만 돈다.

## 5. 비용 추정
- 예상 pod 시간: td.1 약 3.95 h, td.2~td.4 약 3.5 h씩, td.5 약 3.3 h(스모크·진단·측정 0.6 h 포함), td.6 약 2.6 h → 약 20.4 pod-h × $0.59 × 1.03 ≈ **$12.4**, 회수·분석 여유를 더해 **약 $13.5**(묶음 상한 $24의 56%, 80% 선 $19.2 안).
- 19:16Z 지출 추정: td $2.07(budget.py; pod 여섯 합). 범위의 위쪽은 driver의 ETA(19:16Z, 적응 칸 제외), 아래쪽은 처리량 계산이다.
- 처리량이 추정보다 20% 낮아도 약 $16으로 80% 선 안이다. 관문이 3시간 넘게 닫혀 있으면 DMC pod 넷이 기다리는 동안 시간당 약 $2.4가 더 든다.

## 6. 편차와 결정
- **D2(조정자, 판단 에이전트, 19:01Z)**: det+BC의 b 반복 규칙(§1.1). 분석은 B1과 E12 k\*·G를 (가) 모든 칸, (나) bc_unconverged 칸의 det+BC·sh-iter+b를 결측으로 둔 것으로 함께 내고, 갈리면 "보류"로 적는다(`analyze_td.py`에 반영, 수렴한 b의 det+BC 결과 전). 본 실행에서 미수렴 칸이 0이라 두 읽기는 같다.
- 운영 선택(설계 범위 안): E11 계측은 설계가 적은 셋만(계획 재실행·영역 기준 계획 없음), E13은 episode당 8상태(E1 규칙), E15 결정당 시간은 대기열 끝 timing job, td.1~td.4·td.6 driver를 처리량 측정이 끝나기 전에 워커 8로 시작(E9 근거).

## 7. 분석 코드 (본 실행 전 고정)
- `code/analyze_td.py` md5 `d7597ad46583b53f31c2d5406e3cf10c`(D2 두 읽기와 bias 요약을 더한 최종판; 첫 dry run판은 `b40f3d00…`), `code/plan_td.py` md5 `0d5a8a70b6046f46b4690a4dfbffd04f`(분석이 칸 이름·k 격자·τ를 여기서 읽는다)
- dry run 2(19:10~19:16Z, 최종판): `make_fake.py`(md5 `28adb0b1461ef0b2737f89a5e6262902`, bias 결과를 D2 형식으로, 미수렴 표시 3칸 포함)의 가짜 자료로 `analyze_td.py`가 두 읽기까지 끝까지 돌았다(`td.5:/workspace/atm/fake2_analysis/`).
- dry run 1: td.5에서 `make_fake.py`(md5 `fa7c69211a20deb5b5fe72ff84669384`)가 6 pod 배치 전체의 가짜 결과(episode job 7,995개, 적응 결정 84개, E13 held episode 1,080개, timing, prep, bias)를 만들고 `analyze_td.py`가 끝까지 돌았다(35초, 2026-11-21T18:45Z; `td.5:/workspace/atm/fake_analysis/`). 숫자는 가짜이며 해석하지 않았다.
- 돌리는 코드(19:16Z 기준 6 pod 모두 같음): `qlib.py` `fd1ceb9cfe7a65ab9c49398770f1345c`, `runner.py` `50025f47652429bbd9362d50cb234284`, `tdjobs.py` `22607b6414f54053d49a918431e2edd3`(D2 bias job; 그 전 prep·episode는 `09344d26…`판으로 돌았고 두 판의 차이는 bias job뿐), `driver.py` `0ad53e56a2e182c39b6d8cb8455e3b74`, `plan_td.py` `0d5a8a70…`.

## 8. 운영
- 상태 한 줄: `bash /workspace/atm/code/status.sh`
- driver 다시 띄우기(여러 번 불러도 안전): `bash /workspace/atm/code/restart.sh`
- driver 설정: `/workspace/atm/driver_conf.json`(tag, 과제, 워커 수)
- 로그: `/workspace/atm/logs/driver.log`(job마다 한 줄), 적응 결정 `out/driver/decisions.jsonl`, 오류 `out/errors/<id>.txt`
