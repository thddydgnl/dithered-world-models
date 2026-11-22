# DW_dinowm 운영 계획 (DINO-WM: E11·E12·E13·E16의 DINO-WM 몫)

- 작성: 2026-11-21(UTC), dw 작업자. 본 실행 전에 썼다. 결과를 본 뒤 고친 곳은 [사후]로 표시한다
- 기준: [../DESIGN_tmlr.md](../DESIGN_tmlr.md) v1.0(sha256 `36d69ea8…0cc4`) + §11 변경 D1, [../WORKER_RULES.md](../WORKER_RULES.md)
- 가설·지표·문턱·칸·seed는 설계 그대로다. 여기에는 운영 선택만 적는다
- 기록: [LOG.md](LOG.md) · 코드: [code/](code/) · 결과: `results/raw/<tag>/`(조정자가 `pull_all.sh`로 당겨 온다)

## 1. 맡은 칸

| 실험 | 과제 · k · 범위 | 변형 | episode | pod |
|---|---|---|---|---|
| E11 | PushT 4-bit PT | none + 8 | eval8 100 | dw.1 |
| E11 | PointMaze 3-bit PT | none + 8 | eval8 100 | dw.3 |
| E11 | Wall 3-bit PT | none + 8 | eval8 100 | dw.4 |
| E12 | PushT PT·PC {2,3,4}, RC {3,4} | det, det+BC, sh-iter(RC는 det, sh-iter) | 두 단계 50 + 50 | k=4는 dw.1, k=2·3은 dw.2 |
| E12 | PointMaze PT·PC {2,3} | det, det+BC, sh-iter | 100 | k=3은 dw.3, k=2는 dw.4 |
| E12 | Wall PT·PC {2,3} | det, det+BC, sh-iter | 100 | dw.4 |
| E12 확장 | 위로는 가장 높은 k의 pod, 아래로는 가장 낮은 k의 pod | 같음 | 같음 | PushT 위 dw.1·아래 dw.2, PointMaze 위 dw.3·아래 dw.4, Wall dw.4 |
| E16 | PointMaze 3-bit, Wall 3-bit, MPPI식 | none + det, sh-iter, sh-plan, det+shift | eval8 100 + λ 선택 held 20 × 4 | dw.3, dw.4 |
| E13 | E11 칸 × 8변형, E12의 k1 ± 1 칸 | 오프라인 | held 20 + eval8 none 100 상태 | dw.1(PushT), dw.3(PointMaze), dw.4(Wall) |

- 변형 코드 이름: det, detbc(det+BC), shstep(sh-plan, E1과 같음), detshiftfix(det+shift-fixed), shiterb(sh-iter+b), shiter, detshift(det+shift), detnoise(det+noise)
- R1(E11에 있는 E12 칸은 다시 돌리지 않는다)과 R5(none은 pod·계획기마다 한 번)를 driver가 job id로 지킨다

## 2. 구현 선택

### 2.1 코드 경로
- E1 `qlib.py`의 함수는 하나도 고치지 않았다. 새 변형은 하위 클래스 `QuantizerT`와 새 kernel(`_qdetbc`, `_qshb`, `_qdetnoise`)에 있다. hook은 E1 변형과 none을 손대지 않은 E1 hook에 그대로 넘긴다
- `run_cem`에는 `mppi_c=None` 인자 하나만 더했다. None이면 E1 CEM과 같다(점검 1에서 비트 단위로 확인)
- 난수
  - 새 변형의 표는 모두 episode·호출마다 seed한 CUDA generator에서 나온다: `seed_of(salt, 변형 이름, episode, …)`. 기본 CPU·CUDA generator와 계획기의 CPU generator는 건드리지 않는다(점검 2-B)
  - det+shift-fixed의 offset 표: `seed_of(salt, 'detshiftfix', e, plan_idx, h, layer)`. 표 모양은 det+shift와 같고(token × unit, 후보 공유), 일정은 sh-plan과 같다
  - sh-iter+b와 det+noise의 u: `seed_of(salt, 변형, e, it, h, layer)`. sh-iter와 일정이 같지만 draw는 독립이다. E1에서도 변형마다 seed 이름이 달랐다
  - det+noise: 출력 = lo + Δ·(round(y) + u − ½). 더한 뒤 clamp하지 않는다(설계 식 그대로)
- 계획기, 수치, 배치 크기 E(PushT 1, Wall 2, PointMaze 1), job 크기(PushT·PointMaze 10, Wall 20 episode)는 E1과 같다. E11 job은 진단을 켜고(E1과 같은 1·3·5-step 오차, OEI 입력, 모집단 참 비용 재생), E12·E16 job은 진단을 끈다(설계가 요구하지 않음, 계획 결과는 같다)

### 2.2 분할 eval8
- Wall·PointMaze: `L.sample_pairs`, pair seed 8, 100쌍. env seed 8001~8100, planner seed 5000~5099, salt `eval8`
- PushT: D1(§11). 제약 표집기 그대로 쓰고, hard = held·smoke·eval6(저장된 100구간), soft = repro로 둔다
  - 표집 전 점검: 실제 길이로 센 빈 시작점 수(held+smoke 1,294, +repro 485)를 대조하고, 복제본이 `splits_pusht_v3.json`과 같은지 본다
  - 표집 뒤 기록과 위반 점검은 `out/splits/eval8_pusht.json`에 남긴다. 위반이 있으면 driver가 PushT 계획을 막는다
  - dw.1과 dw.2가 만든 쌍의 sha256을 대조한다

### 2.3 범위와 편향 벡터 (`calib_tmlr.py`)
- 보정 창: train 256개, 창 seed 11. E5 `pc`와 같은 자료이고, 앞 64개는 E0 보정 창이다
- PC: E5 `ptq.ranges_perchannel`(99.99 백분위, 256창)과 같다. hi ≤ lo인 단위는 lo + 1e-6으로 막는다(있으면 파일에 기록)
- PT 재측정 점검(§0.6): 256창 중 앞 64창으로 E0 절차를 다시 해 저장된 E0 범위와 비교한다(`ptcheck_<task>.json`)
- b(§0.6)
  - 보정 시작 상태는 보정 창 256개의 첫 프레임이다. 각 시작 상태에서 후보 32개를 굴린다. 후보는 N(0,1)이고, 첫 행 = 0(CEM 첫 반복 분포)이다. seed는 `seed_of('bcal', task, j)`다. 5-step rollout을 쓴다
  - 평균은 행·token·step 전체에서 잰 단위별 평균이다. 보정한 det 모델과 비양자화 모델을 따로 굴린 두 rollout의 평균 차를 쓴다
  - **반복 규칙은 조정자 결정 대기 중이다**(317M과 dw에서 설계 규칙 b ← b + m, 3 sweep이 1%에 닿지 않음; LOG 18:5x). 결정 전에는 설계 규칙으로 계산해 기록한다. det+BC·sh-iter+b job은 driver 관문으로 막는다
  - 정해지면 그 규칙으로 다시 계산한다. sweep마다 다음을 기록한다: 층별 잔차(Δ 대비), 짝지은 원소별 MSE(Δ² 대비, 시작점 16개), 잠재 1·3·5-step 상대 오차
- 소유 pod: 과제마다 보정 파일 한 벌을 한 pod가 만들고 다른 pod에 복사한다(PushT는 dw.1 → dw.2, PointMaze는 dw.4 → dw.3, Wall은 dw.4). 그래서 같은 과제의 모든 k 칸이 같은 PC 범위와 같은 RC 범위를 쓴다. 확장 k의 b와 RC는 그 칸을 도는 pod가 계산한다
- RC(PushT, §2.2)
  - 층별 hi 후보는 256창의 {p99, p99.9, p99.99, p99.999, max}이고, lo는 min이다
  - p99.99에서 시작해 층 0~5를 두 번 돈다. 엄격히 나아질 때만 옮긴다
  - 목적은 held 20쌍의 첫 반복 후보 300개(planner seed 1000 + e)에서 잰 5-step 잠재 상대 오차의 평균이다(det, 그 k)

### 2.4 E12 규칙의 자동화 (`driver.py`)
- 두 단계(PushT): 먼저 0~49를 돈다. 짝 bootstrap 95% CI(10,000, RandomState(0))가 τ = 0.10을 포함하면(허용 오차 1e-9) 50~99를 더 돈다. 판단은 `out/state/decisions.json`에 남긴다
- 확장: (과제, 범위, 반올림)마다 격자 경계 칸의 최종 D로 정한다. 아래로는 D ≤ τ인 동안 k = 1까지, 위로는 D > τ인 동안 k = 8까지 간다. 확장 칸도 두 단계를 따른다
- 우선순위: 보정 → eval8 none → 점검 3 → E11 → E12 격자 1단계 → 2단계 → 확장 → E16 → held none → E13

### 2.5 점검 3 관문
- eval8 none(100)이 끝나면 E1 none(PushT 0.83, Wall 0.61, PointMaze 0.85)에서 10점 안인지 본다. 밖이면 그 과제의 계획을 막는다
- PushT 예외(D1): eval6 쌍 0~49의 none(planner seed 4000 + e)을 돌린다. 쌍은 E1 cache json과 같은지 확인하고 다시 만든다. 36~46/50(E1 41/50)이면 진행하고 LOG에 편차로 적는다

### 2.6 E16 MPPI식 계획기
- 갱신: top-30 elite에서 w = softmax(−(J − min J)/λ), μ = Σ w a, σ = √(Σ w (a − μ)²)를 쓰고, σ 바닥은 0.05(처음 σ = 1의 5%)다
- λ = c × (그 episode 첫 반복 elite 30개 비용의 표준편차, 불편)이고, 그 계획 안에서 고정한다
- c는 held 20쌍 none 성공률이 가장 높은 값을 {0.25, 0.5, 1, 2}에서 고른다(동률이면 큰 c). `out/state/e16_lambda_<task>.json`에 남긴다
- 자격(none MPPI ≥ 0.6 × none CEM)은 같은 pod의 E11 none과 비교한다

### 2.7 E13 (`e13_diag.py`)
- 상태: 그 pod의 none(CEM) held 20쌍과 eval8 100 episode다. 집합은 첫 반복 모집단과 none 계획기의 마지막 반복 모집단이다
- 보상 오차: DINO-WM에는 보상이 없다. 그래서 계획 비용 J(마지막 step 잠재 거리)를 return 자리에 쓴다. median_a |J_q − J_0| / std_a J_0
- ρ_p: 같은 후보 집합을 it = i0와 i0 + 1의 dither 상태로 두 번 평가해 e(a)의 Pearson 상관을 낸다. det, det+BC, sh-plan, det+shift-fixed는 두 상태의 오차장이 같으므로 1로 둔다(정의상, 점검 2에서 표 고정 확인)
- draw: 새로 뽑는 변형은 2개(A, B)이고 잠재 오차는 그 평균이다. 고정 변형은 1개다
- 대상: E11 칸 × 8변형, E12 칸 중 k1 − 1, k1, k1 + 1(PushT 3~5, PointMaze·Wall 2~4)이다. 실제로 돈 칸이나 처음 격자의 칸만 넣는다
- 순서: 상태마다 기준 rollout을 한 번 하고 모든 구성을 잰다. 상태 파일 하나씩 쓰고 이어 돌 수 있다

### 2.8 분석의 운영 정의 (`analyze_tmlr.py`, 본 실행 전에 씀)
- 손상 칸: 한쪽 bootstrap p < 0.05, DINO-WM 3칸 안 Holm
- 칸 분류(손상 칸마다)
  - 편향: det+BC red ≥ 0.5
  - 지속: 편향이 아니고 Δ_pers CI 하한 > 0
  - 해상도: 편향이 아니고, det+shift-fixed red ≥ 0.5이며 Δ_pers CI가 0을 포함
  - 그 밖: 혼합
  - 단위 읽기 = 손상 칸 과반의 분류(없으면 혼합)
- 가설 판정
  - B2: 단위 읽기가 편향 또는 해상도일 때만 성립
  - B3(DINO-WM 몫): det+noise가 고친 손상 칸이 절반 이하
- Dg1: 칸마다 8변형에 대해 D와의 Spearman을 잰다. 하나는 ρ_p × σ_e/σ_J(마지막 반복 집합, E13 모든 상태 평균)이고, 다른 하나는 잠재 1-step(첫 반복 집합, E1 정의)이다. 두 Spearman의 차를 손상 칸에서 평균한다. 0.3 이상이면 성립이다. 마지막 반복 집합의 1-step 판도 보조로 적는다
- E12 G = min(k\*(det), k\*(det+BC)) − k\*(sh-iter)이다(DINO-WM의 E12에는 det+shift가 없다). RC는 k\*(det) − k\*(sh-iter)다
- D1 보조: PushT eval8 칸의 D, Δ_bias, Δ_pers에 겹침 연결 덩어리 단위 bootstrap CI를 붙이고, 읽기가 바뀌면 표시한다

## 3. 본 실행 전 점검 (§0.7) — 결과 파일은 `results/raw/<tag>/checks/`

| 점검 | 결과 | 숫자 | 파일 |
|---|---|---|---|
| 1. 코드 경로 | **통과**(Wall, PointMaze: dw.4 / PushT: dw.2, 모두 RTX 4090) | 손대지 않은 E1 qlib(md5 `f667047f…`)과 새 qlib을 smoke 0~1에서 비교했다. 7변형(none, det, sh-iter, sh-plan, det+shift, ind, sh-ep)의 계획 평균, 반복별 최저 비용, 첫·마지막 모집단, 오차 draw, J_v/J_none이 모두 비트 단위로 같다. 새 코드를 다시 돌려도 같다 | `check1_codepath_{wall,pointmaze,pusht}.json` |
| 2. 새 변형 | **통과**(Wall, PointMaze) | det+BC = det − b(최대 차 1.2e-7). det+shift-fixed는 계획 안에서 표가 고정되고 step·계획·episode마다 바뀌며, 오차 분산이 det+shift와 상대 차 0.02%다. sh-iter+b는 단위별 평균 − b의 최대 z가 2.7, 분산 상대 차 0.1%다. det+noise는 det 부호가 비트 단위로 같고 잡음 분산이 1/12와 상대 차 0.05%다. 계획기·전역 난수는 그대로다. 최종 b 잔차(D2, "최종 b에서")는 8개 모두 < 1%다 | `check2_variants_{wall,pointmaze}.json` |
| 2. b 수렴(D2) | 지금까지 14개 모두 α = 1에서 수렴 | Wall 갱신 6~12번(최악 r 0.0016~0.0067), PointMaze 6~8번(0.0061~0.0099), PushT main k2·k3·k4 9·5·4번, pc k2·k3·k4 9·5·4번 | `calib/bias_*.json`, `checks/bias_resid.jsonl` |
| 3. eval8 none | PointMaze **통과**(0.79 vs E1 0.85; dw.3·dw.4 같음). PushT **통과**(dw.2: 0.73 vs 0.83, 차 −0.10으로 창 경계, 허용 오차 1e-9 안). Wall **창 밖**(0.49 vs 0.61, −12점) → **D3**(조정자, [사후]): eval8 그대로 진행. 실패 기록은 history에 남겼다 | Wall 진단: 같은 pod에서 E1 eval 쌍을 돌리니 62/100(E1 61/100, episode 일치 85%)이다. 쌍 모집단의 차이다. 앞으로 창 밖이면 E1 분할 none을 다시 돌려 ±10점 안이면 진행한다(D3 규칙, driver) | `check3_*.json`, `diag/wall/` |
| 4. 처리량 | 아래 §4 | | `state/timing.jsonl` |
| D1 분할 점검 | **통과** | 빈 시작점 held+smoke 1,294, +repro 485(일치), 복제본 4개 일치. eval8 PushT는 held·smoke·eval6 공유 0, repro는 95쌍과 겹쳐 평균 8.2·최대 13이고 같은 구간은 0이다. 10궤적에 덩어리 10개다 | `checks/eval8_pusht_precheck.json`, `splits/eval8_pusht.json` |
| 범위 재측정(§0.6) | 기록 | PT(앞 64창 vs E0 main): PushT·PointMaze는 정확히 같다. Wall은 hi가 최대 0.031 다르다(bf16 한 칸, GPU 종류 차) | `calib/ptcheck_*.json` |
| RC(PushT) | 기록 | k=3: [p99.999, p99.9, p99.99, p99.99, p99, p99.99], held 5-step 오차 0.687 → 0.300. k=4: [p99.9, p99.999, p99.9, p99.99, p99.9, p99.9], 0.199 → 0.022 | `calib/ranges_rc_k{3,4}_pusht.json` |
| eval8 쌍 일치 | 확인 | PushT sha256 `14f2be60…`이 dw.1과 dw.2에서 같다. PointMaze `e557614b…`가 dw.3과 dw.4에서 같다 | `splits/eval8_*.json` |

## 4. 처리량, 워커 수, 배치, 예상 종료, 비용

- 장비: 네 pod 모두 **RTX 4090 SECURE $0.89/h**다. 2026-11-21 18:39~19:01Z에 A5000 SECURE 재고가 없었다. 순서는 설계대로(dw.3·dw.4 A5000 → 4090, dw.1·dw.2 4090 → A5000)이고, 짝 규칙상 한 칸은 한 pod 안에 있다
- 처리량(측정, s/episode, driver 1개, E1 배치 크기; 진단 = E1 진단 켬)

| 과제 | none(진단) | det(진단) | det+BC(진단) | 확률적·이동 변형(진단) | 진단 끔(E12·E16) | 근거 |
|---|---|---|---|---|---|---|
| PushT | 14.4 | 약 16.3(E0 비율로 추정) | 약 16.3 | 약 18.0(E0 비율) | 약 14.6 | dw.2 none 60 ep |
| PointMaze | 4.97 | 6.41 | 6.37 | 7.41 | 약 5.0 | dw.3 |
| Wall | 1.86 | 약 2.4(E0 비율) | 약 2.4 | 약 2.8 | 약 1.9 | dw.4 |

- 4090은 E0의 A5000보다 약 2.0~2.1배 빠르다(PointMaze none 10.1 → 4.97, PushT none 30.2 → 14.4)
- 워커 수: pod마다 driver 1개(GPU 사용률 85~100%). 같은 GPU에 둘째 프로세스를 돌리면 첫째가 느려졌다(Wall 진단과 PointMaze none이 겹친 구간에서 none 4.97 → 7.56 s/ep)

| pod | 생성(UTC) | 자동 종료 | 맡은 일 | 남은 일 추정 | 예상 종료 |
|---|---|---|---|---|---|
| dw.1 | 18:46:31Z | 04:46:06Z | PushT 4-bit E11, E12 k=4(PC, RC), 위 확장, E13 PushT, PushT 보정 소유 | 보정 0.55 h + none 0.4 + E11 3.9 + E12 1.5 + 확장 0.6 + E13 1.0 = 약 8.0 h(19:16Z부터) | **약 03:20Z**(여유 약 1.4 h) |
| dw.2 | 19:01:38Z | 05:01:06Z | PushT E12 k=2·3(PT, PC, RC k3), 아래 확장 | none 0.4 + E12 1단계 2.8 + 2단계 약 1.2 + 확장 약 0.7 = 약 5.1 h(19:22Z부터) | **약 00:30Z** |
| dw.3 | 19:01:05Z | 04:59:38Z | PointMaze 3-bit E11, E12 PC k=3, 위 확장, E16 PointMaze, E13 PointMaze | none 0.14 + E11 1.6 + E12 0.4 + 확장 0.3 + E16 0.8 + E13 0.8 = 약 4.1 h(19:12Z부터) | **약 23:20Z** |
| dw.4 | 18:42:56Z | 04:41:26Z | Wall 전부(관문 대기), PointMaze E12 k=2, 아래 확장, Wall·PointMaze 보정 소유 | PointMaze 약 1.3 h + Wall 약 1.8 h(결정 뒤) | Wall 결정이 1시간 안에 오면 **약 22:40Z** |

- 비용 추정(4090 $0.89/h × 1.03, 설치·점검 포함)
  - dw.1 약 8.7 h, $8.0
  - dw.2 약 5.7 h, $5.2
  - dw.3 약 4.5 h, $4.1
  - dw.4 약 4.3 h, $3.9
  - **합 약 $21**. 묶음 상한 $35, 그 80%는 $28이다. 설계 추정 약 $17보다 높은 것은 PointMaze·Wall도 4090이라서다(시간은 줄었다). 19:25Z까지 지출은 $1.98이다
  - 끝난 pod는 결과를 당긴 뒤 바로 지워야 한다(§0.1). 단계 B의 분석은 마지막까지 남는 pod(dw.1)에서 모든 pod의 `out/`을 모아 돌린다
- 남은 위험
  - 확장 칸이 예상보다 많을 수 있다(특히 PushT 위로 k = 6 이상). dw.1이 04:46Z 전에 못 끝나면 E13 PushT를 같은 GPU 종류의 dw.1b에서 잇는다
  - Wall 결정이 늦어지면 dw.4가 놀 수 있다. PointMaze 일이 약 1.3 h 남아 있다

## 5. 분석 코드 (본 실행 전에 만듦, 합성 자료로 끝까지 돌림)

- `code/analyze_tmlr.py` md5 `d27d66d64003b0a8cff159fd3324ed8c`
  - B2, B3(DINO-WM), §1.4 기술 항목, E12 k\*·G와 RC 대 PT·PC, Dg1, E16, D1 덩어리 bootstrap, D2 (가)/(나) 두 판
  - 의존: `podplan.py` md5 `be98f26e…`
- E13 계산: `code/e13_diag.py` md5 `22c2716a038cb1575346761cfeb16e2d`
- dry run: 2026-11-21T19:0xZ, dw.4 CPU. `--fake`로 실제 파일 배치의 합성 자료를 만들어 돌렸다(job JSON, e13 상태, 분할, λ). 6.5초에 끝까지 돌았고, 표·JSON·PNG가 나왔다. D2 (나) 판은 그 뒤에 넣었다(같은 함수에 결측 처리만 더함)
- 단계 B: 모든 pod의 `out/`을 한 pod에 모아 `python analyze_tmlr.py <raw>`로 돌린다. 판정 수치는 job JSON에서 다시 계산할 수 있다

## 6. 운영 변경: dw.1 일을 dw.5로 나눔 (2026-11-21T20:00Z, 조정자 요청, 단계 A 보고 뒤)

- 이유: 처음 배치에서 dw.1이 약 03:20Z로 가장 늦었다(나머지는 22:40Z~00:30Z). 사용자는 빠른 완료를 원한다
- **dw.5**: RTX 4090 SECURE $0.89/h, pod `r1mikehh6t9o13`, 19:56:41Z 생성, 자동 종료 05:55:55Z
- 옮긴 일: E12 PushT k=4의 PC 칸(det, det+BC, sh-iter)과 RC 칸(det, sh-iter), 그 칸들의 위쪽 확장, E13 PushT 구성 대부분('rest')
  - dw.1에서 돈 적이 없는 칸들이다. 옮기기 전 dw.1은 eval8 none 중이었다
- 짝: dw.5는 자기 PushT eval8 none(100)과 held none(20)을 따로 돈다. 옮긴 칸의 D는 dw.5의 none으로 잰다. 점검 3도 dw.5에서 따로 한다
- dw.1에 남긴 일
  - E11 PushT k=4 전부(none + 8변형)
  - PT의 위쪽 확장(k=5…). 경계 칸이 E11 칸이라 그 결과가 dw.1에 있다
  - E13 중 PT k > 4 구성('main_ext'): 그 확장 칸의 b가 dw.1에만 있다
- E13 나눔
  - dw.5: E11 칸 × 8변형, PT k=3, PC k=3·4(·5), RC k=3·4(·5). dw.5 자신의 none 상태를 쓴다
  - dw.1: PT k=5 이상
  - 같은 구성을 두 pod가 계산하지 않는다. 분석(`e13_summary`)은 과제마다 모든 pod의 상태 파일을 (분할, episode)로 합친다
  - 두 pod는 같은 쌍·seed·코드·GPU 종류다. dw.1과 dw.5의 none 계획이 비트 단위로 같은지 대조해 기록한다
- 보정 파일: dw.1의 PushT 보정(PC, 백분위, PT 점검, b 6개, RC k3·k4)을 dw.5에 복사했다. 확장 k의 b와 RC는 그 칸을 도는 pod가 계산한다
- 코드: `podplan.py`의 dw.1·dw.5 계획과 `e13_part`, `driver.e13_configs`의 나눔 필터, `analyze_tmlr.e13_summary`의 pod 간 병합. 다른 pod의 계획은 그대로다. 모든 pod의 driver를 정상 재시작 플래그로 새 코드에 올렸다
- 새 예상
  - dw.1: none + E11 약 4.3 h, PT 확장 0.3~0.8 h, held none 0.1 h, E13 PT k5 약 0.1 h. 19:52Z부터 **약 00:30~01:00Z**
  - dw.5: 설치 약 0.3 h, none 0.4, PC·RC k4 1단계 1.0, 2단계 약 0.5, 확장 0~0.4, held none 0.1, E13 약 1.2 h. 19:57Z부터 **약 00:00~00:30Z**
  - 비용은 거의 같다: dw.1이 약 2.5 h 줄고 dw.5가 약 4.0 h 는다. 합 약 $22
