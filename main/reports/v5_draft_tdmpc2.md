# DESIGN_main v5 초안 — TD-MPC2 부분 (E2·E3·E5·E7, GPU 시간, 일정)

- 작성: 2026-10-22 08:20Z, TD-MPC2 작업자. **초안이다. 고정은 조정자가 한다.**
- 상태
  - E2·E3·E5·E7의 본 실행은 하나도 돌리지 않았다. E2 seed(40000~), E3 seed(50000~), E5 seed(60000~), E7 seed(70000~)는 아직 한 번도 쓰지 않았다.
  - 오늘 GPU 0에서 돌린 것: 스모크 분할 점검, 짧은 처리량 측정, 커널 탐침뿐이다. 모두 MPS 없이 프로세스 2개 이하로 PushT 옆에서 돌렸다(9절).
  - H4b와 near-tie × 이득 상관은 계산하지 않았다. E1 분해 성분과 near-tie 값은 보이지만, 상관은 v5 고정 전에 계산하지 않는다.
- 사용자 답과의 관계
  - E2는 두 층으로 썼다. **층 (a)**는 설계 문구대로의 측정이라 사용자 답과 무관하게 돈다. **층 (b)**는 사용자가 받거나 거절하는 덧붙임이다.
  - 어느 답이든 E2 실행 내용(manifest)은 같다. 모든 E2 episode가 층 (b)의 양까지 기록하므로, 답에 따라 바뀌는 것은 판정 문구뿐이다.

---

## 1. 독립 draw OEI (조정자 2026-10-22 요청)

### 1.1 두 계열의 양 대응

| 양 | TD-MPC2 필드 | DINO-WM 필드 |
|---|---|---|
| 실행 계획의 오차, 같은 draw | `e_chosen`(마지막 반복에서 Gumbel로 뽑힌 elite) | `last_argmin_dJ`(가장 가까운 짝) |
| 실행 계획의 오차, 독립 draw(영속 오차장은 유지) | `e_indep` | `mf_dJ`(계획 뒤 forward 1회) |
| 첫 반복 무작위 후보 평균 | `f_e_rand`(정책 slot 24개 제외) | `it0_rand_dJ`(row 0 = 평균 제외) |
| 마지막 반복 모집단 평균 | `l_e_rand`(정책 slot 제외) | `last_pop_dJ`(전부) |
| OEI, 같은 draw | `oei`(v1 0.4 동결 정의) | `oei_ref_argmin` |
| OEI, 독립 draw | `oei_indep` | `oei_ref_mf`(v3 동결 정의) |
| 마지막 반복 선택, 같은 draw | `sel_last` = e_chosen − l_e_rand | last_argmin_dJ − last_pop_dJ |
| 마지막 반복 선택, 독립 draw | `sel_indep` = e_indep − l_e_rand | mf_dJ − last_pop_dJ |
| 모집단 이동 | `drift` = l_e_rand − f_e_rand | last_pop_dJ − it0_rand_dJ |
| 표까지 다시 뽑은 오차(TD-MPC2만) | `e_fresh`, `oei_fresh`, `sel_fresh` | — |
| 정책 prior slot 비중(TD-MPC2만) | `chosen_pi`, `elite_pi`, `elite_pi_w` | — |

- 항등식(계획마다 정확히): oei = sel_last + drift, oei_indep = sel_indep + drift
- 남는 차이
  - TD-MPC2가 실행하는 것은 계획 중에 평가된 후보(뽑힌 elite)다. DINO-WM이 실행하는 것은 계획 중에 평가되지 않은 점(elite 평균)이다.
  - 그래서 TD-MPC2에서는 같은 draw판과 독립 draw판이 draw만 다르다. DINO-WM의 argmin판과 mf판은 draw와 점이 함께 다르다.
- `e_fresh`
  - 표 변형(sh-step, sh-ep, E3 표 변형)에서만 기록한다. 고정 표까지 독립 generator로 다시 뽑아 평가한 값이다.
  - fresh 변형에서는 정의상 e_indep와 같고, det에서는 e_chosen과 같아서 기록하지 않는다.
  - 쓰임: oei − oei_fresh는 "표 하나의 실현값을 이용한 몫"이다. oei_fresh는 dither 평균 오차장 E_u[e(a)]를 이용한 몫이다(조정자의 비선형 전파 가설, 2.4절).

### 1.2 구현
- `qlib.PlanProbe.independent_reeval(seed, fresh_tables=False)`
  - 마지막 반복의 후보 집합을 양자화 모델로 다시 평가한다. RNG 상태는 계획기 자신의 마지막 반복 호출과 같다(같은 Q 쌍, 같은 policy 잡음).
  - dither는 별도 `torch.Generator`로 뽑는다. seed는 e_indep가 3,000,017 + 101·(episode seed) + t, e_fresh가 그 값 + 1,000,003이다.
  - 끝나면 양자화기의 generator, 표, 호출 카운터와 CPU·CUDA RNG를 모두 되돌린다.
- 비용: OEI step마다 가치 평가 1회(계획의 1/6)다. MW(4 step마다)는 약 +4%, DMC(10 step마다)는 약 +1.7%다. 표 변형은 e_fresh에 1회를 더 쓴다.

### 1.3 검증 (smoke 분할, GPU 0, MPS 없음, 07:48~08:04Z)
1. **runner 수준 동일성**(`compare_runs.py`)
   - E0 스모크 13 job, 14 episode를 지금 코드에 재평가를 켜고 다시 돌렸다(`v5/identity/manifest.jsonl`).
     - E1 변형 7개: mw-push k=3, 호출 기록 포함
     - cheetah det k=4와 sh-iter s2 k=5, door-open det+shift s3(2 ep), walker det k=2
     - 계획기 override(N128 it3), warm start 끔
   - 저장된 E0 결과(E1 시대 코드, MPS)와 **저장된 모든 필드가 비트 단위로 같다(13/13)**: return, success, 표 갱신 수, 호출 기록, 표 signature, 단계별 OEI·결정 기록, episode 평균, 계획 재실행 기록, 예측 오차.
   - MPS 유무도 수치를 바꾸지 않는다는 뜻이다.
   - runner의 기록 키 처리를 바꾼 뒤(4번 항목) 다시 돌려도 13/13 같았다.
2. **프로세스 안 동일성**(`test_v5_paths.py`, 변형 12개, 결과 ALL_OK)
   - probe 끔 / 켬 / 켬+재평가에서 return, 행동열 hash, 표 갱신 수, 호출 기록이 같다. 기존 단계별 필드의 차이는 0개다.
   - det, sh-step, sh-ep, indstep, indep, shstep_u/g32/l에서 e_indep == e_chosen이 정확히 성립한다(indep_same = 1). ind, sh-iter, det+shift, shiter_l에서는 다르다(0).
   - 표 변형에서만 e_fresh가 있고, e_chosen과 다르다(fresh_same = 0).
   - sel_last + drift − oei와 sel_indep + drift − oei_indep의 최대 절댓값은 0.0이다.
   - E0 스모크 det return 116.36395911313593을 재현했다.
3. **decision job**(rho_iter_perm 추가 뒤)
   - E0 스모크 decision(E1 변형 6개)은 275개 필드, smoke2 decision(E3 변형 7개)은 316개 필드가 비트 단위로 같다.
   - 새 필드는 `rho_iter_perm` 하나다(1.5절).
4. runner의 `columnar`·`means`가 행 키의 합집합을 쓰도록 바꿨다. 그래서 변형에 따라 있는 필드(e_fresh, rho_iter_perm)가 빠지지 않는다. 한 job 안의 행이 같은 키를 갖는 기존 기록은 바뀌지 않는다(1번 재확인).

### 1.4 H2a 재실행 (사후 보정, 사용자 승인 필요; 준비만 함)
- **목적**: TD-MPC2의 세 H2a 칸에서 동결 정의(같은 draw)와 함께 독립 draw판을 잰다. 칸은 mw-push k=3, mw-pick-place k=3, cheetah-run k=5다. 동결 정의의 H2a 결과(불성립)는 그대로 기록에 남는다.
- **manifest**(`~/pmx/A/tdmpc2_main/v5/manifests/`)
  - `h2a_indep_min.jsonl`: ind, sh-iter, det+shift. 162 job, 810 episode
  - `h2a_indep_all.jsonl`: 7변형. 378 job, 1,890 episode
  - spec은 **E1 결과 파일에 저장된 spec 그대로**다. job id, seed 20000~20029, 체크포인트 1~3, E0 범위, OEI·계획 재실행 간격이 같다. 예측 오차만 끄고(closed loop와 무관) 독립 재평가를 켰다.
  - 출력은 별도 폴더 `~/pmx/A/tdmpc2_main/out_h2a_indep/`다. job id가 같으므로 E1 폴더와 섞지 않는다.
- **실행**(승인 뒤): `bash code/run_v5.sh h2a_indep ../v5/manifests/h2a_indep_min.jsonl ~/pmx/A/tdmpc2_main/out_h2a_indep 0 nomps 2` (tmux `A-v5-h2a_indep`)
- **분석**: `analyze_h2a_indep.py`
  1. 재실행 episode마다 E1 episode와 비트 동일성을 확인한다. 기존 필드 전부를 보고, 예측 오차만 뺀다.
  2. oei 대신 oei_indep로 H2a를 계산한다. 나머지 규칙은 E1과 같다: 같은 90짝, 같은 재표본(RandomState(0), 10,000회), det CI 상한 < 0이고 sh-iter CI가 0을 포함.
  3. 변형별 oei, oei_indep, sel_last, sel_indep, drift, (oei_fresh), 정책 slot 비중과 CI를 기술한다.
  - smoke 동일성 출력으로 코드 경로를 점검했다. 수치는 해석하지 않는다.
- **필요한 변형**
  - H2a는 det와 sh-iter만 쓴다.
  - det, sh-step, sh-ep는 구성상 oei_indep ≡ oei다(1.3절에서 확인). 그래서 E1 값을 그대로 쓴다. none은 e ≡ 0이다.
  - 최소는 sh-iter(270 ep)다. **권장은 fresh 3개(810 ep)**로, 분해표가 fresh 변형 전부를 덮는다.
  - 7변형판에는 sh-step·sh-ep의 e_fresh와 전 변형 동일성 확인이 덧붙는다.
- **GPU 시간**: 5절 표
- **같은 장비**
  - E1은 GPU 0(MPS)에서 돌았다. 비트 동일성을 확인하려면 GPU 0이 맞고, PushT 옆에서 MPS 없이 2 worker로 지금 돌릴 수 있다.
  - GPU 1에서 돌리려면 먼저 `v5/identity/manifest.jsonl`(2.5분)로 GPU 1의 동일성을 확인한다.

### 1.5 ind + 계획 고정(indstep)과 ind + episode 고정(indep)의 사전 예상
- **구조**: 표를 후보 slot 번호에 붙인다. 층마다 (horizon, slot, unit) 표이고, indstep은 제어 step마다, indep은 episode마다 고정이다.
- **표본 slot(512개 중 488개)**
  - TDMPC2._plan은 매 반복 `actions[:, 24:] = clamp(mean + std·ε)`를 새로 뽑는다. 평균 자체를 담는 slot은 없다(DINO-WM의 row 0과 다름).
  - 그래서 slot 번호는 행동열 정보를 갖지 않고, DINO-WM 작업자의 추론이 그대로 맞다. 운 좋은 u_j를 가진 slot은 반복마다 elite에 들 수 있다. 하지만 그 slot의 행동은 매번 새 표본이라, 평균 갱신에 ind와 같은 잡음만 더하고 특정 행동 영역으로 끌고 가지 못한다.
- **정책 prior slot(0~23)**
  - 제어 step마다 MPPI 루프 전에 한 번 계산되어, 6반복 내내 같은 행동열로 남는다.
  - indstep에서는 행동열과 표가 함께 고정되므로, 이 24개 후보의 오차가 반복 간에 지속된다. det와 같은 구조지만 후보끼리는 독립이다. DINO-WM slot 0의 짝이다.
  - indep에서는 step 사이에도 표가 남는다. 다만 정책 행동열은 상태와 함께 바뀐다.
- **정책 slot 경로의 크기**(`pi_share.py`; 비양자화 계획기, smoke seed 19005, 체크포인트 1, 과제 12개 × 1 episode, 2 step마다)
  - 정책 slot은 마지막 반복 elite 64개 중 **0~2.4%**다. 후보 비중은 4.7%다.
  - MPPI 가중치로는 0~2.5%다.
  - 실행 elite가 정책 slot인 step은 **0~4%**(12과제 평균 약 1%)다.
- **사전 예상**(실행 전 기록)
  - indstep ≈ ind, indep ≈ ind다. 손상, 표를 다시 뽑은 OEI(`oei_fresh`), drift가 모두 그렇다.
  - 정책 slot을 통한 det 같은 성분은 작다.
  - 양자화가 정책 비중을 바꿀 수 있으므로, E2·E3 모든 episode에 `chosen_pi`, `elite_pi`, `elite_pi_w`를 기록해 이 예상을 확인한다.
- **지표 주의**
  1. indstep·indep의 e_indep는 slot 표를 유지한다(DINO-WM 규칙: 계획 고정 오차장은 유지). 그래서 e_chosen과 같다.
     - TD-MPC2에서는 이 값이 **선택된 slot의 winner's curse를 포함**한다. DINO-WM mf는 선택되지 않는 slot 0에서 잰다.
     - 그래서 ind와는 `oei_fresh`(indstep) 대 `oei_indep`(ind)로 비교한다.
  2. 고정 후보 집합의 rho_iter(decision job)는 slot 표에서 구성상 1이다(같은 후보, 같은 slot).
     - decision job에 `rho_iter_perm`을 더했다. 두 번째 평가에서 표본 후보만 slot을 무작위로 섞는다(정책 slot은 그대로). 그래서 slot이 아니라 행동열에 붙은 지속을 잰다.
     - 다른 변형은 오차가 slot에 의존하지 않으므로 분포상 rho_iter와 같다.
     - 6개 smoke 상태에서 코드 경로를 점검했다(해석하지 않음).

---

## 2. E2 (실험 2, 기전) — TD-MPC2 운영 세부

### 2.1 공통
- **과제와 칸**: mw-push k=3, mw-pick-place k=3, cheetah-run k=5(1차 k)다. PushT는 DINO-WM 계열이 맡는다.
- **seed**: E2 분할 40000~40029(처음 씀), 체크포인트 1~3, E0 범위(main)
  - 모든 설정이 같은 seed를 쓴다. 그래서 설정 사이에도 짝이 맞는다.
- **설계: OFAT**(기본값 둘레에서 한 번에 한 요인씩, 16설정)

| 설정 | 후보 N | elite | 정책 prior | temperature | 반복 | warm start | episode/체크포인트 |
|---|---|---|---|---|---|---|---|
| base | 512 | 64 | 24 | 0.5 | 6 | 켬 | 30 |
| N64 / N128 / N256 / N1024 | N | N/8 | round(24N/512) = 3 / 6 / 12 / 48 | 0.5 | 6 | 켬 | 30 |
| el2 / el25 | 512 | 10 / 128 | 24 | 0.5 | 6 | 켬 | 20 |
| T01 / T2 | 512 | 64 | 24 | 0.1 / 2.0 | 6 | 켬 | 20 |
| it1 / it3 / it12 | 512 | 64 | 24 | 0.5 | 1 / 3 / 12 | 켬 | 20 |
| nows1 / nows3 / nows6 / nows12 | 512 | 64 | 24 | 0.5 | 1 / 3 / 6 / 12 | 끔 | 20 |

- **OFAT로 한 이유**
  - 설계 6.2는 요인마다 수준 목록만 준다. 완전 격자(5 × 3 × 3 × 4 × 2 = 360설정)는 GPU 시간이 20배를 넘는다.
  - 판정이 있는 것은 N(H2c)과 기본 설정(H2a)뿐이다. 나머지는 기술 보고다.
- **N sweep: elite는 개수가 아니라 비율(12.5%)로 고정한다**
  - elite 64개로 고정하면 N = 64에서 선택이 없어진다(모든 후보가 elite).
  - 정책 prior도 비율(24/512 = 4.7%)로 맞춘다. 24개로 고정하면 N = 64에서 후보의 37.5%가 정책 궤적이 되어 계획기 성격이 바뀐다. 작은 N에서는 3·6·12개가 된다.
- **elite 비율**: 2% → 10개, 25% → 128개(N = 512)
- **warm start 끄기**
  - 매 step `t0=True`로 계획한다. 평균은 0, std는 max_std에서 시작한다.
  - 기본 TD-MPC2는 첫 step에서만 그렇게 하고, 이후에는 `_prev_mean`을 한 칸 밀어 쓴다.
- **N = 512**
  - E2 seed로 다시 돈다(base).
  - E1 기본값 자료(seed 20000~)는 별도 재현으로 보고한다(v4 6.2 "재현으로 따로 보고").
- **변형**
  - 모든 설정: none, det, sh-iter, sh-step
  - base에만 더함: ind, sh-ep, det+shift
  - 각 설정의 none이 그 설정의 손상 기준이다.
- **계측**: E1 간격을 쓴다. OEI는 MW 4 step, DMC 10 step마다, 계획 재실행은 20 / 50 step마다다. episode별 예측 오차는 끈다.
  - 모든 episode에 기록하는 것: oei(동결), oei_indep, (oei_fresh), sel_last, sel_indep, (sel_fresh), drift, 정책 slot 비중
- **규모**: 2,826 job, 14,130 episode(MW 9,420, DMC 4,710)
  - base 1,890, N 각 수준 1,080, 나머지 설정 각 720
  - manifest `v5/manifests/e2.jsonl`(`make_manifest.py e2`)
- **통계**: v4 6.1과 같다.
  - 칸 값 = 짝에 걸친 episode 평균의 평균
  - 짝 bootstrap 10,000회, RandomState(0), 백분위 95% CI
  - DMC 하락은 재표본마다 비율로 다시 계산한다.

### 2.2 층 (a): 원래 측정 (사용자 답과 무관)
- **H2a 재현**(base, E2 seed, 동결 정의 `oei`)
  - v4 6.2 그대로다: det OEI CI 상한 < 0이고 sh-iter OEI CI가 0을 포함
  - 과제 4개 중 3개 이상이면 성립이다. 재현으로 보고한다.
- **H2c**(설계 6.2: "N 증가에 따른 det 손상과 |OEI|의 Spearman이 0.8 이상이다(과제 4개 중 3개 이상)")
  - **문구가 두 가지로 읽힌다**
    - (i) 각각을 N에 대해: ρ_s(N, |OEI_det|) ≥ 0.8 **그리고** ρ_s(N, D_det) ≥ 0.8(N 5수준)
    - (ii) 둘 사이: N 5수준에 걸친 ρ_s(D_det, |OEI_det|) ≥ 0.8
  - **권장: (i)를 확증, (ii)를 보조로 둔다.**
    - 가설 H2c 문장("|OEI|는 후보 수가 많을수록 커진다")과 예측 문장("det의 |OEI|와 손상은 N과 함께 커지고")이 둘 다 N에 대한 단조성이다.
    - (ii)는 둘 다 N과 무관하게 흔들려도 성립할 수 있다.
  - D_det(N)은 같은 N의 none 대비 하락이다(MW %p, DMC 상대 return).
  - |OEI_det(N)|은 칸 평균의 절댓값이다.
  - 5수준 Spearman 0.8은 이웃한 자리바꿈 두 번까지 허용한다(한 번이면 0.9).
  - 보조: 설계의 "sh-iter에서는 평탄하다"를 보려고 ρ_s(N, |OEI_sh-iter|)와 ρ_s(N, D_sh-iter)를 보고한다. ρ의 bootstrap 분포도 보고한다. 설정 사이 seed가 같으므로 짝을 함께 다시 뽑는다.
- **elite 비율, temperature, 반복 수, warm start**: 설계에 판정 문턱이 없다. 설정별 D, OEI 성분과 CI를 기술 보고한다.
  - R2-14 확장: warm start를 켠 반복 수 곡선과 끈 곡선을 함께 그린다. 한 step 안의 이용과 step 사이 누적을 가르기 위해서다.

### 2.3 층 (b): 덧붙임 (사용자가 받거나 거절)
- **위치**
  - E1 결과 뒤에 더하는 사전등록이라고 표시한다.
  - E2 seed로만 검정하고, E1 자료는 쓰지 않는다.
  - 층 (a)를 대신하지 않는다. 동결 H2a 불성립은 그대로 남는다.
- **(b1) 독립 draw판 OEI**(두 계열 공통; 조정자 제안)
  - 판정: H2a_indep. det `oei_indep` CI 상한 < 0이고 sh-iter `oei_indep` CI가 0을 포함
  - 과제 4개 중 3개 이상이면 성립이다. DINO-WM은 v3 동결 정의 `oei_ref_mf`가 곧 이 판이다.
  - det는 구성상 oei와 같다.
- **(b2) 지속 오차 이용 가설(drift 기반)**
  - **H2a′**(base, 1차 k칸): 과제마다 다음 둘이 모두 성립하면 그 과제는 성립이다. 과제 4개 중 3개 이상이면 H2a′ 성립이다. PushT는 DINO-WM의 last_pop_dJ − it0_rand_dJ로 같은 규칙을 쓴다.
    1. drift_det의 95% CI 상한 < 0
    2. r = drift_sh-iter / drift_det의 95% CI 상한 < 0.5
       - 칸 평균의 비다. 재표본마다 다시 계산하고, drift_det* ≥ 0인 재표본은 버린다(analyze.boot와 같은 처리).
       - 뜻: sh-iter가 det의 지속 이동을 절반 넘게 없앤다. 문턱 0.5는 H3a의 red ≥ 0.5와 맞췄다.
       - sh-iter drift가 양수(영역 효과)면 r < 0이 되어 조건을 만족한다.
    - 양성 대조(보조, 판정 밖): r_sh-step, r_sh-ep의 CI 하한 > 0.5. 지속되는 확률 표는 det처럼 이용된다.
    - 기술: ind, det+shift의 r. 같은 draw판과 독립 draw판의 sel_last도 함께 보고한다.
      - 예상: sel_last는 모든 변형에서 음수다(같은 draw의 winner's curse). sel_indep는 fresh 변형에서 0 근처다.
  - **H2c′**: ρ_s(N, −drift_det) ≥ 0.8(N 5수준)이고, 과제 4개 중 3개 이상이면 성립이다.
    - 보조: ρ_s(N, −drift_sh-iter)(예상: 평탄), ρ_s(N, −sel_last_det)(같은 draw 몫의 극값 증가)
- **(b3) 영역 효과 대조**(선택; 구현 안 함)
  - drift에는 이용 말고도 "모집단이 E_u[e]가 다른 영역으로 가는 효과"가 섞인다. E1 사후 점검에서 pick-place k=2 sh-iter drift가 +30.6이었다.
  - 대조 방법: OEI step마다 같은 상태, 같은 RNG로 비양자화 계획기를 한 번 더 돌린다. 그 첫·마지막 모집단을 양자화 모델로 평가해 drift_ref를 얻는다. drift − drift_ref가 양자화 오차가 계획을 끈 몫이다.
  - 비용: 계획 1회 + 평가 2회가 OEI step마다 든다. MW 약 +25%, DMC 약 +10%다.
  - H2a′를 영역 효과에 강하게 만들고 싶을 때만 권한다. 받으면 E2 시작 전에 구현하고 스모크한다(약 1시간).
- **덧붙일 값싼 지표**(조정자 질문): 실행 elite의 독립 draw 재평가 `e_indep`
  - 비용: MW +4%, DMC +1.7%
  - det에서 현재 e와 정확히 같다(1.3절). 이미 구현해서 모든 E2·E3 episode에 기록한다.

### 2.4 조정자의 비선형 전파 가설(E4 이론 작업자에게) — 기존 자료로 본 빠른 점검
- 가설: 비선형 전파 때문에 E_u[V_q] ≠ V₀다. dither가 무편향이어도 지속 편향장이 남는다.
- 점검(E1 decision 자료, 새 GPU job 없음)
  - fresh 변형의 rho_iter가 0이 아니다. 같은 고정 후보를 독립 draw 두 번으로 평가한 오차 상관이다.
  - sh-iter 중앙값 0.06, 최대 0.51(pick-place k=2), pick-place k=3은 0.30이다.
  - 그래서 후보에 따라 다른 평균 오차 E_u[e(a)]가 남는다.
- 다만 36칸에서 rho_iter와 −drift의 Spearman은 0.21로 약하다. pick-place k=2 sh-iter drift가 +30.6이라 drift에는 영역 효과가 섞여 있다.
- E2에서는 이 장을 직접 볼 수 있다.
  - fresh 변형의 oei_indep와 표 변형의 oei_fresh는 실행 elite에서 E_u[e]의 이용만 남긴다(같은 draw 몫이 없음).
  - 음수면 지속 편향장을 이용한다는 신호다.

---

## 3. E3 (실험 3, 구조와 표 크기) — TD-MPC2 운영 세부

### 3.1 2축 6칸
- **6칸**: 같은 비트 수(과제의 1차 k), 12과제

| | 매 forward 새로 | 계획(제어 step) 고정 | episode 고정 |
|---|---|---|---|
| 후보 공유 | sh-iter | sh-step | sh-ep |
| 후보 독립 | ind | **indstep(새로)** | **indep(새로)** |

- 설계 6.3은 "독립 + 계획 고정"만 이름으로 적는다. 하지만 "6칸"이라고 했으므로 "독립 + episode 고정"(indep)으로 마지막 칸을 채웠다. 비용은 E3 2축 예산의 1/8이다.
- 문자 그대로를 원하면 indep를 빼면 된다.
- none과 det도 함께 돈다. none은 손상 기준이고, det는 red의 분모다.
- **seed**: E3 분할 50000~50029, 30 episode × 체크포인트 3개 = 90짝
- **계측**: E1 계측(OEI, 계획 재실행, episode별 예측 오차)에 독립 재평가를 더한다(e_indep, e_fresh, 정책 slot 비중).
- **사전 예상**: 1.5절(indstep ≈ ind ≈ indep)

### 3.2 표 크기 sweep (sh-fixed = 계획 고정 공유 표)
- **과제**: mw-push, mw-pick-place, cheetah-run, walker-walk. 1차 k다. walker k=3은 E1에서 sh-iter가 해로웠던 칸이다.
- **수준**: 층마다 독립 offset 수. 4개 양자화 층은 모두 폭 512다.

| 이름 | 코드 | 층당 표 크기 | N_eff(비용 하나가 합산하는 독립 offset 수, 4층) |
|---|---|---|---|
| 층당 하나 | shstep_l | 1 | 4 |
| group 32 | shstep_g32 | 32 | 128 |
| unit당 하나 | shstep_u | 512 | 2,048 |
| horizon × unit(기본) | shstep | 3 × 512 = 1,536 | 6,144 |
| 후보 × horizon × unit(키우는 방향) | indstep | 512 × 1,536 = 786,432 | 6,144 |

- **주의: 설계 4.3의 N_eff 정의로는 키우는 방향의 N_eff가 기본과 같다**
  - 늘어난 offset은 후보 사이에 퍼져 있고, 비용 하나가 합산하는 수는 그대로다.
  - 그래서 H4a 확증 계산은 공유 표 4수준(l, g32, u, H×u)으로 하기를 권한다. indstep은 같은 그림에 따로 표시한다. 후보 축 대조(sh-step 대 indstep)로 읽는다.
  - 대안: 표의 전체 항목 수로 5수준에 순서를 매긴다(설계의 "표 크기" 문자 그대로).
- **같은 거칠기의 새로 뽑기 대조**(덧붙임): shiter_l, shiter_g32. 매 forward 새로 뽑는다.
  - 거친 표는 unit 사이 오차를 상관시켜 오차 크기 자체를 바꾼다. smoke2 코드 경로 점검에서 이것이 보였다(수치는 해석하지 않음).
  - 지속의 비용 = D(shstep_g) − D(shiter_g)를 거칠기마다 보조로 보고한다.
- **H4a 조작적 정의(권장)**
  - 과제마다 공유 표 4수준에 대해 Spearman(N_eff, D_sh-fixed) ≤ −0.8을 본다. 4점이므로 이웃 자리바꿈 한 번까지 허용한다.
  - TD-MPC2 계열은 sweep 과제 4개 중 3개 이상이면 성립이다.
  - 보조: 과제별 Spearman의 중앙값, 과제마다 D를 정규화해 섞은 Spearman
  - DINO-WM은 자기 수준(token × unit → unit → group 128 → 층당 1)으로 같은 규칙을 쓴다.
- **decision job**: 새 변형마다 고정 결정 상태(E1과 같은 calib ep 5~9, 3 × 150 상태)에서 rho_iter, rho_iter_perm(slot 변형), common_frac를 잰다.

### 3.3 H4b (near-tie 밀도와 CRN 이득)
- **CRN 이득**: 과제마다 E3 seed의 2축 칸(1차 k)에서 D_ind − D_sh-iter
- **near-tie 밀도**
  - E1 decision job에서 이미 계산한 값을 쓴다. 3 × 150 고정 상태(calib ep 5~9)에서 비양자화 top1–top2 가치 gap / 후보 가치 std의 중앙값이다.
  - **마지막 반복 집합을 확증**으로 둔다(조정자 의견에 동의). 실행 elite가 뽑히는 곳이고, 같은 draw 선택(sel_last)이 일어나는 곳이다. 첫 반복 집합은 보조다.
- **부호 주의**
  - E1 분석이 보고한 값은 gap의 중앙값이다. 값이 클수록 near-tie가 **적다**.
  - 그래서 H4b의 "near-tie 밀도가 높은 과제에서 이득이 크다"는 Spearman(중앙값 gap/std, CRN 이득) ≤ −0.5와 같다. 밀도를 −(중앙값 gap/std)로 두면 ≥ 0.5다.
  - 이 부호를 v5에 명시해야 한다.
- **단위**: TD-MPC2 12과제를 확증으로 둔다. DINO-WM 3과제를 섞은 15과제는 보조다.
  - 단위 문제: MW·DINO-WM은 %p, DMC는 상대 return이다. 이득을 과제 단위로 순위만 쓰는 Spearman이라 그대로 섞는다.
- **v5 고정 전에는 계산하지 않는다**(지시대로). E1의 D_ind − D_sh-iter도 있지만 상관은 계산하지 않았다.

### 3.4 DMC 설명 계획 (설계 §7 대응; 사전등록 안)
- **질문**: E1 DMC 양상을 near-tie 밀도와 지속 편향 비중이 설명하는가? 양상은 H3a DMC 불성립(5/12)과 walker k=3의 sh-iter 해(+22.1% [18.0, 26.3])다.
- **자료**: E3 seed의 2축 칸(1차 k, 12과제). walker k=3이 들어 있으므로 해로운 칸이 새 seed로 다시 재진다.
- **결과 변수**: 과제마다 red_sh-iter(E3)
  - E3 seed에서 det 하락이 실재하는 과제만 쓴다. 기준은 한쪽 bootstrap p < 0.025다. 빠진 과제는 적는다.
- **설명 변수 1(확증): det의 지속 편향 비중 S_det = drift_det / oei_det**
  - E3 seed det 칸의 평균이다.
  - 뜻: 실행 elite에서 det가 얻은 유리한 오차 가운데, 마지막 반복 전에 이미 모집단에 쌓여 있던 몫이다(반복 사이에 지속된 오차의 이용).
  - oei_det의 CI가 0을 포함하면 S를 정의하지 않고, 그 과제를 적고 뺀다.
  - **이것을 고른 이유**
    - closed loop 계획 안에서 잰다.
    - 단위가 없어 과제 사이에 비교된다.
    - det에서 정의되고, 좁힌 기전 주장("반복 사이에 지속되는 오차의 이용")과 같은 양이다.
  - **다른 두 후보를 확증으로 쓰지 않는 이유**
    - common_frac는 모든 후보에 공통인 오차 비중이다. 오차가 얼마나 해롭지 않은지를 재지, 지속을 재지 않는다.
    - rho_iter는 det, sh-step, sh-ep에서 구성상 1이라 det로는 과제 순위를 못 매긴다.
- **설명 변수 2(이차)**: near-tie 밀도. −(마지막 반복 집합 중앙값 gap/std)이고, 첫 반복은 보조다.
- **보조**
  - sh-iter에 남는 지속: rho_iter(sh-iter, 고정 상태)와 drift_sh-iter/drift_det(계획 안)
  - common_frac(det, 첫 반복)
- **규칙(제안)**
  - Spearman(S_det, red_sh-iter) ≥ 0.5(포함된 과제 전체)이면 "설명됨"이다.
  - 그렇지 않고 Spearman(near-tie 밀도, red_sh-iter) ≥ 0.5이면 "부분 설명"이다.
  - 둘 다 아니면 설계 §7대로 처방 주장을 MW와 DINO-WM으로 좁히고, DMC는 한계로 보고한다.
  - 예상 방향: S_det가 크면 sh-iter가 없앨 지속 이용이 많아 red가 크다. near-tie가 많으면 det의 결정적 오차가 순위를 더 뒤집고, 공유 잡음은 순위를 보존해 red가 크다.
  - 기술 점검: DMC 과제의 S_det 중앙값이 MW보다 작은가? 해로운 칸(walker k=3)의 S_det가 낮은 쪽인가? S_det가 작으면 det 손상이 주로 지속 이용이 아니다. 그러면 dither는 오차 분산(D²/6 대 det의 D²/12)만 늘려 해로울 수 있다.
- **v5 고정 전에는 어떤 상관도 계산하지 않는다.** E1의 성분은 이미 보이지만, 판정은 E3 seed로만 한다.

### 3.5 규모
- 2,124 job: episode job 2,088개, decision job 36개
- episode 10,440개(MW 6,660, DMC 3,780)
  - 2축: 12과제 × 8변형 × 90 = 8,640
  - sweep 덧붙임: 4과제 × 5변형 × 90 = 1,800
- manifest `v5/manifests/e3.jsonl`(`make_manifest.py e3 --ktable ../out/ktable.json`)

---

## 4. E5 (강한 기준선과 결합, H3b) — TD-MPC2 운영 세부

### 4.1 칸과 seed
- **H3b 칸**: E1에서 det 하락이 실재한(Holm) 3~4-bit 칸 19개
  - MW 12칸: button-press 3, door-open 3·4, drawer-open 3, peg-insert 3·4, pick-place 3, push 3·4, sweep-into 3, window-open 3·4
  - DMC 7칸: finger 3·4, quadruped 3·4, walker 3·4, cheetah 4
- **seed**: E5 평가 60000~60029(90짝), QAT 학습 자료 65000~65019, 기준선 선택은 held 12000~12019(E0와 같은 분량)

### 4.2 활성 PTQ 기준선 (같은 k, 같은 4층)
1. **per-channel 범위**: 같은 보정 활성(calib ep 0~4, 12,000행)에서 unit마다 [min, 99.99 백분위]를 잡는다. Quantizer가 (폭,) 텐서 범위를 이미 받는다.
2. **MSE 최소 범위**: 층마다 hi = α × (99.99 백분위)로 둔다. α ∈ {0.5, 0.6, …, 1.1} 가운데 보정 활성의 k-bit det 양자화 MSE가 가장 작은 값을 고른다(lo = min).
3. **SmoothQuant식 이전**
   - 양자화할 활성 a_j(Mish 출력)를 s_j = max|a_j|^0.5 / max|W_:,j|^0.5로 나눈다. s_j는 다음 Linear의 열 j에 곱해 접는다(fp32에서 정확).
   - dyn1·rew1의 다음 층은 출력 head다.
4. **혼합 정밀도**: 각 MLP의 마지막 은닉층(dyn1, rew1)은 8-bit로 두고, dyn0·rew0만 k-bit다.
- **최선 기준선 B 선택(칸마다, 사전등록)**
  - held 분할에서 det 손상 D_det(B)가 가장 작은 기준선을 고른다. none은 E0 held 결과를 쓴다.
  - 같으면 위 목록의 앞쪽을 고른다.
  - dither는 B가 정한 격자 위에 얹는다. 확률적 반올림이고, sh-iter 일정이다.
- **H3b**
  - E5 seed에서 칸마다 D(B+det) − D(B+sh-iter)를 짝 bootstrap으로 잰다.
  - 95% CI 하한 > 0인 칸이 19개 중 과반(10칸 이상)이면 성립이다.
  - B+det와 E1 기본 det의 차이, B+sh-iter의 red도 함께 보고한다.

### 4.3 가중치 양자화 (탐색; 활성은 양자화하지 않음)
- 대상은 같은 4층(dyn0, dyn1, rew0, rew1)의 Linear 가중치이고, 출력 head는 fp32로 둔다.
- 변형(DINO-WM `ptq.py`와 같은 이름)
  - w8rtn, w4rtn: 출력 채널별 대칭 RTN
  - w4awq: 입력 채널 scale = (보정 활성 평균 |a_j|)^0.5를 곱한 뒤 RTN, 다시 접음
  - w4sr1, w4sr2, w4sr4: episode마다 W의 확률적 반올림 K개, MPPI 반복 번호 mod K번째를 씀. K = 1은 sh-fixed의 짝이다.
- 과제는 sweep 과제 4개, 60짝(20 × 3)이다. none은 E5 평가 seed를 같이 쓴다.

### 4.4 QAT (짧은 예산)
- **칸**: 3~4-bit 자격 칸 가운데 과제의 1차 k 11칸(MW 8, DMC 3: finger 4, quadruped 3, walker 3). 체크포인트 3개, 합계 33회 학습
- **방법**
  - 4층 은닉 활성에 det fake-quant + STE를 건다.
  - dynamics·reward MLP만 미세조정한다. 손실은 TD-MPC2의 consistency(ρ 할인)와 reward cross-entropy다.
  - encoder, Q, policy는 고정한다.
- **설정**
  - 자료: 비양자화 agent episode 20개(seed 65000~)와 저장된 calib ep 0~9
  - 5,000 step, batch 256, horizon 3, Adam lr 1e-4, grad clip 20(TD-MPC2 값)
- **평가**: E5 seed에서 QAT+det, QAT+sh-iter를 none과 비교한다(90짝).
  - QAT+det 손상의 CI 상한이 5%p(DMC 0.05) 아래면, dither의 가치는 "학습 없는 PTQ 처방"으로 한정해 보고한다(설계 6.5).

---

## 5. E7 (실제 커널과 지연) — TD-MPC2 운영 세부

### 5.1 오늘 탐침 결과 (`e7_feasibility.py`; PushT와 공유한 GPU 0, 수초, 지시용 수치)
- 환경: `torch._int_mm`가 A5000(torch 2.7.1)에서 TD-MPC2 은닉층 모양(512 × 520 × 512 등)으로 돈다. 정수 기준과 정확히 같다. Triton 3.3.1을 import할 수 있다.
- **같은 모양 GEMM 한 번의 시간**: int8 `_int_mm`는 모양과 무관하게 약 48~52 µs라 고정 비용으로 보인다.
  - 512행(후보 수)에서는 fp32(약 50 µs)와 같다. 1024행에서는 fp32보다 빠르다(51 대 94 µs).
  - 모든 모양에서 bf16(7~24 µs)보다 느리다.
- **eager 양자화 연산**: dither(rand + add + floor + clamp + cast) 24~46 µs, det(round + clamp + cast) 7~24 µs
- **뜻**
  - TD-MPC2 크기(폭 512, 후보 512)에서는 계획 지연이 kernel 실행 횟수에 묶인다. GEMM 정밀도를 낮춰도 빨라지지 않을 수 있다.
  - 그래서 E7은 융합 Triton kernel과 CUDA graph가 필요하다.
  - 그래도 빨라지지 않으면 H5의 Pareto 부분은 성립하기 어렵다. 설계 §7대로 한계로 적는다.

### 5.2 구현
- **int8 W8A8**
  - Triton kernel 하나로 다음을 묶는다: int8 `tl.dot`(int32 누적) → 역양자화(scale, zero-point 보정 Σq_w z_a) → bias → LayerNorm → Mish → 다음 층 입력의 재양자화(det 또는 dither).
  - 층마다 kernel 하나다. 가중치는 출력 채널별 대칭 int8이다. 활성은 지금 격자 [lo, hi]의 비대칭 per-tensor다.
- **dither**: 층·forward마다 공유 u 벡터 하나(sh-iter 일정)를 kernel 안에서 열 단위로 읽는다. 후보 수와 무관하다.
- **int4 활성**
  - 기본은 4-bit 수준(0~15)을 int8 그릇에 담는 방식이다. 수치는 4-bit와 같지만 속도는 int8과 같다.
  - packed int4(1바이트에 2개, kernel 안에서 풀기)는 목표로 둔다. Ampere에서 Triton에는 int4 tensor-core 경로가 없다. 그래서 packed가 빨라지는 것은 메모리 대역 몫뿐이다. 준비되지 않으면 int8 그릇으로 보고하고, 이를 편차로 적는다.
  - torchao dynamic int8은 venv에 없다. 설치(PyPI 내려받기)는 승인이 필요하다.
- **기준 정밀도**: fp32(TD-MPC2 기본), bf16(autocast)

### 5.3 지연 측정 규약
- 배타적 GPU 창이 필요하다. 그 GPU에 다른 프로세스가 없어야 하고, MPS도 끈다. 단일 프로세스로 잰다.
- 제어 step(agent.act) 하나를 CUDA event로 잰다.
- 조건마다 warmup 50 step 뒤 200 step 블록을 잰다. 블록 순서를 섞어 5회전 한다.
- 조건별 중앙값과 IQR을 보고한다.
- 측정 때 호스트 부하(loadavg, CPU 사용률)와 GPU clock을 기록한다.
- eager와 CUDA graph(가능하면)를 따로 보고한다. 확증은 eager로 둔다. 배치 경로이고 dither RNG를 graph에 넣는 위험이 없다. graph는 보조다.

### 5.4 격자와 판정
- 격자: {fp32, bf16, int8 det, int8 dither, int4 det, int4 dither} × 반복 {3, 6, 12} × 과제 4개(mw-push, mw-pick-place, cheetah-run, walker-walk)
- 성공률은 실제 kernel로 따로 잰다. 정수 누적은 fake-quant와 수치가 조금 달라 episode가 같지 않다.
  - MW는 60짝(20 × 3), DMC는 30짝(10 × 3)이다. seed는 70000~다.
- H5: dither 추가 지연 = (int8 dither − int8 det) / int8 det 중앙값 ≤ 10%(int4도 같음), 그리고 Pareto 지배 사례가 하나 이상

---

## 6. GPU 시간 (TD-MPC2)

- **근거 처리량**: E0·E1 실측, episode/시간, 바쁜 시간 기준

| 조건 | MW | DMC |
|---|---|---|
| MPS 16 worker, E1 계측 | 3,716 | 860 |
| MPS 16 worker, 계측 끔 | 4,800 | 929 |
| MPS 없이 16 worker, E1 계측 | 2,095 | 488 |
| MPS 없이 2 worker, PushT와 GPU 0 공유(오늘 07:21~07:31Z) | 960 | 267 |

- **설정별 비용 배수**(단일 프로세스 step 시간, 오늘 측정, 기본 = 1)
  - N64 0.64, N128 0.67, N256 0.71~0.79, N1024 1.39~1.45
  - it1 0.23~0.25, it3 0.54, it12 1.90~1.93
  - elite·temperature 1.00
  - warm start 끔은 같은 반복 수와 같다.
  - 작은 N의 배수는 MPS 포화 상태에서 이보다 작을 것이라 보수적이다.

| 작업 | episode | MPS, GPU 한 장 전부 | MPS 없이 16 worker, GPU 한 장 전부 | MPS 없이 2 worker, PushT 옆 |
|---|---|---|---|---|
| H2a 재실행: sh-iter만 / fresh 3개 / 7변형 | 270 / 810 / 1,890 | 0.15 / 0.46 / 1.1 h | 0.27 / 0.81 / 1.9 h | 0.52 / 1.6 / 3.7 h |
| E2 | 14,130 | 6.5~7.4 h | 13.1 h | 25.5 h |
| E3 | 10,440 + decision job 36개 | 6.2 h(계측이 가벼우면 5.5; decision job은 몇 분) | 10.9 h | 21 h |
| E5: held 선택 / 평가 / 가중치 / QAT(자료·학습·평가) | 3,720 / 4,500 / 1,440 / 2,640 | 1.5 / 2.7 / 1.0 / 3.0 h → **8.2 h** | 약 13 h | 하지 않음 |
| E7: kernel 개발 / 지연(배타) / 성공률 | — / — / 3,240 | 1 / 1.2 / 1.9 h → **약 4 h** | 약 4.5 h | kernel 개발만 |
| **합계** | | **약 26 h** | **약 43 h** | |

- E2의 범위(6.5~7.4)는 계측 비용이 E1 계측과 계측 끔 사이에 있다는 뜻이다. E2는 예측 오차를 끄므로 아래쪽에 가깝다.
- QAT 학습(33회 × 약 3분 = 1.7 h)은 단일 프로세스 GPU 시간이다. MPS와 무관하고 PushT 옆에서도 돈다.
- 설계 6.9 예산(두 계열 합)은 E2 10, E3 10, E5 8, E7 3 h다. TD-MPC2 몫만으로 E5와 E7이 예산 수준이다. 줄이는 안은 다음과 같다.
  - E5 held 선택의 MW를 체크포인트당 10 ep로: −0.3 h
  - 가중치 탐색을 2과제로: −0.5 h
  - E7 성공률을 반복 6에서만: −1.3 h

---

## 7. 일정 제안

- **전제**(조정자의 GPU 상황)
  - GPU 0: DINO-WM PushT E1(MPS 없음)이 약 17~18Z까지
  - GPU 1: B E4/E5가 약 12Z까지, DINO-WM PointMaze가 약 13:30Z까지, 그 뒤 PushT k=5일 가능성
  - MPS와 non-MPS 프로세스는 한 GPU를 나눠 쓰지 못한다. `run_v5.sh`는 그런 GPU에서 MPS 시작을 거부한다.
- **지금 ~ 첫 빈 GPU: GPU 0, PushT 옆, MPS 없이 프로세스 2개 이하**
  1. H2a 재실행(승인 시): fresh 3개 1.6 h. E1과 같은 GPU라 비트 동일성 확인이 바로 된다.
  2. E5 준비: per-channel, MSE 범위, SmoothQuant, AWQ 통계(분 단위). QAT 경로와 E5 변형의 스모크(smoke 분할). v5 고정 뒤에는 QAT 자료 수집(660 ep, 2 worker로 약 1.2 h)과 QAT 학습(33회, 약 1.7 h)
  3. E7: Triton kernel 개발과 정확성 시험(단일 프로세스). 지연 측정은 하지 않는다.
  - 합계 4~5 h 분량이라 13:30Z 전후까지 찬다. PushT를 얼마나 늦추는지는 DINO-WM 작업자에게 확인이 필요하다.
- **A안: GPU 1이 약 13:30Z에 TD-MPC2에 넘어오는 경우**
  - 13:30Z~: GPU 1에 자체 MPS daemon(`pipe_g1`)을 띄우고 16 worker로 E2를 돈다(약 7 h, ~20:30Z). 이어서 E3(약 6.2 h, ~02:45Z)
  - 18Z~: GPU 0이 비면 MPS로 E5 episode(held 선택 → 평가 → 가중치 → QAT 평가, 약 6.5 h, ~00:30Z), 이어서 E7 성공률(1.9 h)
  - E7 지연(배타 1.2 h)은 실험 사이 빈 창(예: GPU 1의 E2와 E3 사이)에 둔다.
  - 다른 안: 18Z에 E3를 GPU 0으로 옮겨 E2와 나란히 돌리면 E3가 ~00:15Z에 끝난다. 그 뒤 E5는 먼저 빈 GPU에서 돈다.
  - 끝: 내일 03~04Z
- **B안: GPU 1을 PushT k=5가 쓰고, 첫 빈 GPU가 GPU 0(~18Z)인 경우**
  - 18Z~01:30Z E2, ~07:45Z E3, ~14:15Z E5, ~17:30Z E7(내일). GPU 1이 중간에 비면 E3나 E5를 그쪽으로 넘긴다.
- **같은 장비 원칙(설계 6.0)**
  - manifest 하나(실험 하나)를 한 GPU에서 끝까지 돈다. 그래서 한 칸의 모든 변형과 기준선이 같은 GPU다. job마다 `env.cvd`로 기록된다.
  - E2와 E3를 서로 다른 GPU에서 돌리는 것은 괜찮다(칸이 겹치지 않음).
  - GPU 1을 처음 쓸 때는 `v5/identity/manifest.jsonl`(2.5분)로 GPU 1이 GPU 0과 같은 수치를 내는지 먼저 기록한다. 원칙상 필요하지는 않지만, 두 GPU 결과를 함께 해석할 때 근거가 된다.

---

## 8. 편차와 결정이 필요한 것

| 번호 | 내용 | 누가 |
|---|---|---|
| 1 | H2a 재실행(사후 보정): fresh 3개(권장) / 7변형 / sh-iter만, GPU 0에서 2 worker | 사용자 승인 |
| 2 | E2 층 (b): (b1) 독립 draw OEI 판정, (b2) H2a′·H2c′(문턱 0.5, 0.8), (b3) 영역 효과 대조(구현 필요) | 사용자 |
| 3 | H2c 문구: (i) 각각 N에 대해(권장) 대 (ii) 둘 사이 | 조정자 |
| 4 | E2 OFAT 16설정, N sweep을 elite 비율·정책 비율 고정으로, 기본 외 설정 20 ep/체크포인트 | 조정자 |
| 5 | E3 6번째 칸 indep(설계는 "독립 + 계획 고정"만 적음) | 조정자 |
| 6 | E3 표 sweep에 같은 거칠기의 새로 뽑기 대조(shiter_l, shiter_g32) 추가 | 조정자 |
| 7 | H4a: 설계 정의로는 후보 × horizon × unit의 N_eff가 기본과 같다 → 확증은 공유 4수준(권장) 대 표 항목 수 5수준 | 조정자 |
| 8 | H4b 부호: E1이 보고한 "near-tie 밀도"는 gap 중앙값이라 near-tie가 많을수록 작다 → 부호를 뒤집어 판정 | 조정자 |
| 9 | DMC 설명: 확증 변수 S_det = drift_det / oei_det, near-tie는 마지막 반복 집합, 문턱 0.5 | 조정자 |
| 10 | 새 지표: e_fresh, 정책 slot 비중, rho_iter_perm(판정에 쓰지 않는 진단) | 기록 |
| 11 | E7: `_int_mm`가 이 크기에서 fp32보다 빠르지 않음 → 융합 Triton kernel이 필요, int4는 int8 그릇이 기본(packed는 목표). torchao 설치는 승인 필요 | 조정자, 사용자 |
| 12 | runner 기록 키 합집합(결과 동일 재확인), env.sh·mps.sh의 선택적 A_GPU(기본값 그대로), `run_v5.sh` | 기록 |

---

## 9. 오늘 돌린 것 (모두 GPU 0, MPS 없음, 프로세스 2개 이하, 본 실험 seed 없음)

| 시각(Z) | 내용 | 결과 |
|---|---|---|
| 07:21~07:31 | 2 worker 처리량(bench seed 18000~, DMC 360초, MW 240초) | DMC 267, MW 960 ep/h |
| 07:48~07:51 | 동일성 재실행(E0 smoke 13 job, 2 worker) | 13/13 비트 동일 |
| 07:51~07:58 | `test_v5_paths.py`, `pi_share.py`(smoke seed, 프로세스 2개) | ALL_OK; 정책 slot 비중 |
| 08:01~08:04 | 다시 점검(runner 키 변경 뒤 동일성 + decision job 2개) | 13/13, 275/316 필드 동일 |
| 08:05 | `e7_feasibility.py`(수초, 1 프로세스) | 5.1절 |
| 08:16~08:18 | `run_v5.sh` 시험(동일성 manifest, MPS 없이 2 worker) | 13/13 비트 동일, env.mps 없음 |

- 파일
  - 코드: `main/code/tdmpc2/`의 qlib.py, calib_lib.py, runner.py, make_manifest.py, compare_runs.py, analyze_h2a_indep.py, pi_share.py, e7_feasibility.py, test_v5_paths.py, run_v5.sh, env.sh, mps.sh
  - 결과: `main/results/v5_tdmpc2/`
  - 서버: `~/pmx/A/tdmpc2_main/v5/`(manifest, 점검 출력)
