# DINO-WM E2·E3·E5 운영 세부 제안 (DESIGN_main v5 초안용)

- 작성: 2026-10-22T07:20Z, DINO-WM 작업자. 본 실행은 하지 않았다(smoke 분할 스모크만)
- 근거: 설계 §6.2·6.3·6.5, DESIGN_main v1~v4, E0·E1 측정
- 공통 전제(E1과 같음): 같은 eval 쌍과 planner seed(PushT `eval6` 4000+e, Wall·PointMaze `eval` 3000+e), 같은 범위(`main`), 진단 포함(OEI 확증 = `oei_ref_mf`), E=1(PushT·PointMaze)·2(Wall). E1 칸을 기준점으로 다시 쓴다(짝 비교)
- 비용은 **A5000 단독 실측**(GPU 1 벤치, 진단 포함): PushT k=4에서 none 30.2 · det 34.2 · sh-iter 37.8 s/episode. 아래 배율은 오늘 smoke에서 같은 조건으로 잰 비율이다

## 0. 스모크로 확인한 것 (smoke 분할, PushT k=4, 1 episode씩, GPU 0)

| 항목 | 결과 |
|---|---|
| 새 코드 = E1 코드 | E1 스모크 job을 새 코드로 다시 돌리면 mu_final·iteration별 비용·오차·OEI가 비트 단위로 같다(GPU 1 → GPU 0) |
| 후보 나눠 돌리기(`planner.max_rows`) | N=300을 150씩 나눠도 det·sh-iter 계획이 비트 단위로 같다. N=1024를 300씩 나누면 최대 메모리 6.3GB(안 나누면 약 17GB 예상) |
| N에 따른 시간 | N=64: 0.22배, N=1024: 3.55배(N=300 대비). 거의 N에 비례 |
| opt_steps 10 | 0.46배(진단 forward의 고정 부담 포함) |
| 새 E3 변형 | `indplan` 1.10배·최대 7.1GB, `shunit`·`shgroup`·`shlayer` 1.0배. 모두 정상 동작(hook 호출 990) |
| 작은 차이 | 최종 계획 평균의 비용(한 행 forward)만 E=1과 E=2 사이에서 0.4% 다르다(OEI_mf −0.924 대 −0.926). E1 안에서는 E가 과제마다 고정이라 문제없음 |

## 1. E2 — 기전: 계획기의 오차 이용 (PushT)

### 1.1 격자
- 과제·칸: PushT, E1 1차 칸 **k=4**. 변형 **none, det, sh-iter**(설계 예측 "det의 |OEI|와 손상은 커지고 sh-iter는 평탄"에 필요한 최소)
- **후보 수 N ∈ {64, 128, 256, 512, 1024}** + E1의 N=300을 그대로 6번째 점으로 쓴다(추가 비용 0)
  - top-k는 **elite 비율을 기본값 10%로 고정**: top-k = 6, 13, 26, 51, 102(E1 300 → 30)
  - 이유: 설계가 N과 elite 비율을 따로 된 요인으로 적었다. 개수를 30으로 고정하면 N과 선택 비율(47% → 2.9%)이 함께 바뀐다
  - 선택 강도는 아래 elite 비율 sweep에서 따로 본다
  - N=1024는 `max_rows=300`으로 나눠 돈다(수치 동일 확인)
- **elite 비율 ∈ {2%, 12.5%, 25%}**(N=300: top-k 6, 38, 75) + E1의 10%(30)
- **opt_steps ∈ {5, 10}** + E1의 30
- MPPI temperature는 CEM에 없으므로 DINO-WM에는 해당 없음
- **warm start 끈 대조는 해당 없음**
  - DINO-WM은 open-loop라 episode당 계획이 한 번이고, 매번 mu=0, σ=1에서 시작한다. 이미 "warm start 없음"이 기본이다
  - CEM 반복 사이의 재적합은 알고리즘 자체라 끌 수 없다
  - 그래서 E2의 DINO-WM 쪽은 "한 계획 안의 이용"만 잰다. step 사이 누적은 TD-MPC2 쪽에서만 본다
  - closed-loop MPC로 바꾸면 비교가 생기지만 계획기 규약이 바뀐다(설계 밖, 권하지 않음)

### 1.2 비용 (단독 A5000, episode당 none + det + sh-iter)

| 점 | episode당 | 100 ep | 50 ep |
|---|---|---|---|
| N=64 | 22 s | 0.6 h | 0.3 h |
| N=128 | 46 s | 1.3 h | 0.6 h |
| N=256 | 88 s | 2.4 h | 1.2 h |
| N=512 | 178 s | 4.9 h | 2.5 h |
| N=1024 | 362 s | 10.1 h | 5.0 h |
| **N sweep 합계** | | **19.3 h** | **9.7 h** |
| elite 3점 (N=300) | 307 s | 8.5 h | 4.3 h |
| opt_steps 5·10 | 78 s | 2.2 h | 1.1 h |
| **전체** | | **30 h** | **15 h** |

### 1.3 검정력 (H2c: N 6점에서 Spearman(det 손상, |OEI|) ≥ 0.8)
- 몬테카를로로 대략 계산했다(사후 분석이 아님, 계획용). 가정: 손상과 |OEI|가 log N에 따라 선형으로 커지고, 점마다 짝 episode n개로 추정한다. N=300 점은 E1의 100 ep로 고정했다

| 참 효과(N=64 → 1024) | 25 ep | 50 ep | 100 ep |
|---|---|---|---|
| 강함: 손상 10 → 35%p, \|OEI\| 0.15 → 0.60 | 0.27 | 0.46 | 0.68 |
| 중간: 15 → 30%p, 0.25 → 0.50 | 0.11 | 0.17 | 0.30 |
| 약함: 18 → 25%p, 0.35 → 0.45 | 0.04 | 0.05 | 0.06 |

- 손상 추정의 표집 잡음(100 ep에서 짝 SE 약 5%p)이 점 사이 차이와 비슷하다. 그래서 episode를 줄이면 검정력이 크게 떨어진다. |OEI|는 정밀하다

### 1.4 선택지와 권고
- A. 모든 점 100 ep: 30 GPU-h. H2c 검정력이 가장 높다
- **B(권고)**: N sweep은 100 ep(H2c 판정 대상), elite·opt_steps는 탐색이라 50 ep → **약 25 GPU-h**
- C. 예산이 빠듯하면: N=1024만 50 ep, elite·opt_steps 50 ep → 약 20 GPU-h. N=1024 점의 잡음이 커진다
- D. 더 줄이면(N sweep 50 ep) 약 15 GPU-h이지만 H2c 검정력이 강한 효과에서도 0.46이다. 권하지 않는다
- 설계 §6.9의 E2 예산(두 계열 합 10 GPU-h)으로는 PushT 한 과제도 100 ep로 못 돈다. 예산을 늘리거나 C·D로 줄이는 결정이 필요하다

## 2. E3 — 오차 구조와 dither 표 크기

### 2.1 2축 대조 (open-loop DINO-WM)
- 지속 축의 "계획 고정"과 "episode 고정"이 같은 조건이다(episode당 계획 한 번). 그래서 6칸이 4칸으로 줄어든다

| | 매 forward 새로 | 계획(= episode) 고정 |
|---|---|---|
| 후보 공유 | sh-iter (E1) | sh-plan (E1 `shstep`) |
| 후보 독립 | ind (E1) | **ind-plan (새 변형 `indplan`)** |

- `indplan` 정의
  - 후보 칸(slot) × token × unit마다 독립인 u 표를 층·horizon step마다 계획 시작에 한 번 뽑아, 그 계획의 모든 CEM 반복에서 같은 칸 번호에 쓴다
  - CEM은 반복마다 후보를 새로 뽑으므로, 지속은 행동열이 아니라 칸 번호에 붙는다(칸 0 = 현재 평균만 계속 같은 행동). TD-MPC2 쪽 정의와 맞출 필요가 있다
- 새로 돌 것은 `indplan` 칸뿐이다. 나머지 셋은 E1 칸을 짝으로 다시 쓴다
- 비트 수: PushT k=4(1차)와 k=3(잡음이 커서 공유 효과가 보일 수 있는 쪽, 파일럿 R2-13 §5-2), Wall k=3(탐색)
- 비용: PushT 2칸 × 100 ep × 41.6 s = 2.3 h, Wall k=3 0.2 h

### 2.2 고정 dither 표 크기 sweep (핵심, H4a)
- sh-plan의 고정 표를 층·horizon step마다 줄여 간다. 비용 하나가 합산하는 독립 offset 수 N_eff가 바뀐다

| 수준 | 코드 | 층·step당 독립 값 | 비고 |
|---|---|---|---|
| L+ (더 큰 쪽) | `indplan` | 후보 300 × token × 2048 | 후보마다 독립(2.1절과 같은 칸) |
| L0 (기본) | `shstep` | token(196~588) × 2048 | E1 칸 |
| L1 | `shunit` | 2048(unit당 하나, token 공유) | |
| L2 | `shgroup` | 128(unit 16개씩 묶음) | |
| L3 | `shlayer` | 1(층·step당 하나) | |

- H4a는 L3 < L2 < L1 < L0 네 점에서 잰다(공유 표끼리라 같은 축). L+는 함께 보고한다
- 스모크에서 계획 시점 1-step 오차가 표가 작을수록 커졌다(sh-plan 0.07 대 shunit·shgroup·shlayer 0.10~0.14, 1 episode). 오차가 token 사이에서 상쇄되지 않기 때문으로 보인다
- 과제·k: PushT k=4(det 손상이 있는 1차 칸), Wall k=3(탐색, 손상이 큼)
- 비용: PushT 3변형 × 100 ep × 37.8 s = 3.2 h, Wall k=3 0.5 h

### 2.3 near-tie 밀도와 CRN 이득 (H4b)
- 새 실행 없이 E1 자료(서버 `pops/`의 모집단 배열)로 잰다
  - near-tie 밀도: none 변형 마지막 반복 모집단의 (1·2위 모델 비용 차 ÷ 후보 비용 std)의 중앙값
  - CRN 이득: 1차 k에서 D_ind − D_sh-iter
- 과제 3개를 TD-MPC2 과제와 합쳐 Spearman을 잰다

### 2.4 E3 합계: 약 6.2 GPU-h

## 3. E5 — 강한 기준선과 결합

### 3.1 활성 PTQ 기준선
- 위치는 E1과 같다(예측기 MLP 은닉, GELU 뒤 6층)
1. **per-channel 범위**: unit마다 [min, 99.99%]. 보정 창을 64개에서 256개로 늘려 unit별 꼬리를 안정시킨다
2. **범위 보정 방식**: percentile(99.9, 99.99[현재], 99.999) 대 **MSE 최소화**(층마다 det 양자화 MSE가 최소인 hi를 찾음)
3. **SmoothQuant식 이전**: GELU 출력 채널을 s_j = max|x_j|^α / max|W_j|^(1−α)(α = 0.5)로 나누고, 다음 Linear(2048 → 404)의 입력 가중치에 곱한다. 가중치는 fp로 두므로 사실상 활성의 채널별 격자 조정이다
4. **혼합 정밀도**: 마지막 블록의 MLP 은닉만 8-bit, 나머지 k-bit
- 절차
  - held 20 ep에서 det 성공률로 과제·k마다 최선 기준선을 고른다(eval은 쓰지 않음)
  - eval 100 ep에서 **최선 기준선 + det 대 + sh-iter**를 비교한다(H3b: 3~4-bit 자격 칸 과반에서 짝 CI 하한 > 0)
- 칸: E1에서 자격(det 하락 실재)이 확인된 3~4-bit 칸. 예상은 PushT k ∈ {3, 4}, PointMaze k=3이 자격일 때 PointMaze k=3. Wall k=3은 탐색

### 3.2 가중치 양자화 (활성은 fp)
- 대상: 예측기 Linear 전부(6블록 × attention qkv·out, MLP fc1·fc2)
1. W8·W4 per-channel RTN
2. AWQ식 W4: 보정 활성으로 입력 채널 scale을 찾은 뒤 RTN
3. **W4 SR 사본 K ∈ {1, 2, 4}**: 독립 확률적 반올림 사본 K개를 CEM 반복마다 돌려 쓴다(반복 i는 사본 i mod K). K=1은 고정(sh-plan과 같은 구조)

### 3.3 QAT (짧은 예산)
- PushT만 한다(서버에 PushT 학습 영상이 있음. Wall·PointMaze obs는 풀지 않아 59GB·30GB가 더 필요)
- 예측기를 k=4 fake-quant(det, STE)로 DINO-WM 학습 손실을 써서 약 2 GPU-h 미세조정한다
- eval: QAT 모델의 none, det, sh-iter
- QAT가 손상을 없애면 dither의 가치는 "학습 없는 PTQ 처방"으로 한정해 보고한다(설계 그대로)

### 3.4 비용 (단독)

| 항목 | GPU-h |
|---|---|
| held 기준선 선택(4종 × det × 20 ep × PushT 2칸) | 1.5 |
| eval 최선 기준선 × {det, sh-iter} × 100 ep × PushT 2칸 | 4.0 |
| 가중치 6조건 × 100 ep(PushT) | 5.2 |
| QAT 학습 + eval 3조건 × 100 ep | 4.8 |
| PointMaze k=3(자격이면) | 1.0 |
| **합계** | **약 16** |

- **최소판(약 8~9 GPU-h)이 빼는 것**: W8 RTN, AWQ식 W4, SR 사본 K=2, QAT 전부(학습 + eval), PointMaze k=3
  - 남기는 것: 활성 PTQ 4종 held 선택 → eval 최선 기준선 + det/sh-iter(PushT k=3·4), W4 RTN, W4 SR K ∈ {1, 4}
- **구현과 스모크 완료(07:40~08:05Z, smoke 분할, GPU 0)**
  - `ptq.py`: 보정(train 창 256개, 2분) → 범위 파일 p999, p99999, p9999w256, pc, smooth, mse_k3, mse_k4와 AWQ 통계. 256창 99.99% 범위는 E1 'main'(64창)과 bf16 한 칸 안에서 같다
  - `qlib.py`: per-channel 격자, 혼합 정밀도(`qopts.last_bits`), 가중치 변형(`w8rtn`, `w4rtn`, `w4awq`, `w4sr1/2/4`; 비양자화 기준 forward는 자동으로 fp 가중치), QAT용 straight-through
  - `qat.py`: 20 step 스모크(bs 8) 동작, 체크포인트 저장 → runner `ckpt`로 평가
  - 계획 스모크 15 job 모두 정상. E1 코드와 비트 일치(identity job), QAT job 뒤 원래 예측기 복원 확인. 1 episode 계획 시점 1-step 오차(det k=4): main 0.094, mse_k4 0.015, p999 0.020, pc 0.027, smooth 0.063, last8 0.083. 가중치: W8 0.002, W4 RTN 0.59, W4 AWQ 0.40, W4 SR K=1 2.8, K=4 1.9(기능 점검일 뿐, 해석하지 않음)
  - 남은 것: E5 manifest(held 선택 → eval)는 v5 고정 뒤 만든다

## 4. GPU 일정 제안

- **제약: MPS와 같은 GPU를 못 쓴다.** 비-MPS DINO-WM 프로세스가 있는 GPU에는 새 MPS client가 붙지 못한다(오늘 06:08Z Error 205). 그래서 TD-MPC2의 MPS sweep과 DINO-WM은 서로 다른 GPU를 써야 한다
- 현재 E1 PushT 예상
  - GPU 0: k=4 칸 + none·nonealt가 약 14Z에 끝나고, 이어 k=3이 약 19Z에 끝난다
  - GPU 1: B 작업 뒤 PointMaze가 끝나면(조정자 예상 약 13:30Z) g1 driver가 k=5를 맡아 약 19Z에 끝난다
- DINO-WM E2/E3/E5의 가장 이른 시작은 약 19Z다
  - TD-MPC2가 MPS로 한 GPU 전체를 쓰면, DINO-WM은 다른 GPU에서 순서대로 돈다: E2 N sweep → E3 → E2 elite·opt_steps → E5
  - 권고안 합계: E2 25 + E3 6 + E5 16 = 약 47 GPU-h, GPU 한 장으로 약 2일(B와 나누면 1.3~2배)
- 두 GPU의 결과가 비트 단위로 같으므로 (과제, 설정) 칸을 GPU 사이에 나눠도 된다. TD-MPC2 sweep이 끝나는 대로 나머지 DINO-WM 칸을 그 GPU에 얹을 수 있다
- 서두를 때의 순서(설계 §6.9의 우선순위 E2 > E3 > E5): E2 N sweep(19 h) → E3(6 h) → E5 최소판(9 h) → E2 elite·opt_steps

## 5. 구현 상태
- 코드: `main/code/dinowm/`
  - `qlib.py`: `indplan`, `shunit`, `shgroup`, `shlayer` 변형과 `rollout_chunks`(후보 나누기)
  - `runner.py`: job별 `planner`(num_samples, topk, opt_steps, max_rows)
  - `make_manifest.py custom --planner ... --ptag ...`로 칸을 만든다
- E1 변형과 E1 계산은 바뀌지 않았다(스모크 비트 일치)
- 스모크 결과: 서버 `results/E2E3_smoke/`, manifest `manifests/E2E3_smoke.json`
- E5 코드: `ptq.py`, `qat.py`, qlib·runner의 E5 옵션(§3.4). 스모크 결과: 서버 `results/E5_smoke/`, `qat/pusht_qatsmoke/`
- indplan 예상(v5 문장 후보): CEM이 반복마다 후보를 전부 새로 뽑으므로, 칸 0(현재 평균)을 빼면 칸에 고정한 dither는 새로 뽑는 독립 잡음과 같다. ind와 성공률·OEI(drift ≈ 0)에서 구별되지 않을 것으로 예상한다
