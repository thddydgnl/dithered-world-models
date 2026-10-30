# A 최종 종합 보고 — Planners Exploit Deterministic Error: Dithered Low-Bit World Models

- 작성: 2026-10-25 (UTC). 작업자: A 최종 보고 담당. 맥북에서 문서와 결과 파일만 읽었다. 새 실험·학습·시뮬레이션은 없고, 서버와 RunPod에는 접근하지 않았다.
- 기준: [DESIGN_main.md](../DESIGN_main.md) v1 → v5.3(§0~§10). 설계는 [EXPERIMENT_DESIGN.md](../../EXPERIMENT_DESIGN.md), 기록은 [LOG.md](../LOG.md)와 프로젝트 루트 `MAIN_STATUS.md`다.
  - 주장 조정은 두 번 있었다. v5.2 §9(10-22 09:50Z, 사용자 승인)는 기전 주장을 좁히고 OEI 진단 주장을 뺐으며, DMC에는 설계 7장 대응을 적용했다. v5.3 §10(10-24 02:38Z, 사용자 승인)은 QAT 보고 방식과 사후 점검 둘을 정했다.
- 근거 보고(모두 `reports/`): [E0_tdmpc2](E0_tdmpc2.md), [E0_dinowm](E0_dinowm.md), [E1_tdmpc2](E1_tdmpc2.md), [E1_dinowm](E1_dinowm.md), [E4_theory](E4_theory.md), [E2_tdmpc2](E2_tdmpc2.md), [E2_dinowm](E2_dinowm.md), [E3_tdmpc2](E3_tdmpc2.md), [E3_dinowm](E3_dinowm.md), [E5_tdmpc2](E5_tdmpc2.md), [E5_dinowm](E5_dinowm.md), [E7_tdmpc2](E7_tdmpc2.md), [posthoc_tdmpc2](posthoc_tdmpc2.md)
  - 수치 뒤 괄호가 출처 보고와 절이다.
  - 줄임: E0T·E0D = E0_tdmpc2·E0_dinowm, E1T·E1D, E2T·E2D, E3T·E3D, E5T·E5D, E7T = E7_tdmpc2, PH = posthoc_tdmpc2, E4 = E4_theory, DM = DESIGN_main, ED = EXPERIMENT_DESIGN
  - "이 보고 계산"이라고 적은 값만 이 보고가 결과 JSON이나 기존 수치로 셈한 것이다(맥북에서 몇 초 걸린 순수 파이썬 읽기).
- 판정을 정한 핵심 수치는 결과 JSON에서 직접 확인했다. 확인한 파일: `E1_tdmpc2/e1_tdmpc2_summary.json`, `E1/e1_dinowm_summary.json`, `E1/e1_dinowm_v52.json`, `v5_tdmpc2/h2a_indep/h2a_indep_summary.json`, `E2_tdmpc2/family_points_tdmpc2.json`, `E2_tdmpc2/e2_tdmpc2_summary.json`, `E2_dinowm/family_points_dinowm.json`, `E3_tdmpc2/e3_tdmpc2_summary.json`, `E3_tdmpc2/e3_tdmpc2_final.json`, `E3_tdmpc2/h4b_points_tdmpc2.json`, `E2_dinowm/h4b_points_dinowm.json`, `E3_dinowm/family_points_e3_dinowm.json`, `E5_tdmpc2/family_points_e5_tdmpc2.json`, `E5_dinowm/family_points_e5_dinowm.json`, `E7_tdmpc2/family_points_e7_tdmpc2.json`, `posthoc_tdmpc2/e7_posthoc_latency_summary.json`, `posthoc_tdmpc2/posthoc_qatnone_summary.json`
  - 판정 수치는 모두 보고와 같았다. 보조 15과제 H4b ρ도 다시 계산해 LOG와 같았다(−0.32 / +0.21).
  - 판정에 쓰이지 않은 기술 수치 하나가 보고와 달랐다(Meta-World red ≥ 0.9 칸 수 14 대 13). [불일치 D1](#10-불일치-목록)에 적었다.
- 판정은 조정자가 확정한 계열 합산 판정 그대로 쓴다(LOG 10-22 19:40Z, 10-23 16:10Z·22:40Z, 10-24 12:29Z). 이 보고는 판정, 가설, 문턱을 바꾸지 않는다.
- **출처 분류 표시**
  - [사전등록]: 설계 v1의 가설·문턱과, 해당 자료가 생기기 전에 정한 조작적 정의(v4·v5·v5.1)
  - [E1 뒤 사전등록]: E1 결과를 본 뒤 v5·v5.2로 더했지만, 해당 자료를 계산하기 전에 고정한 것(H2a_indep, H2a′, DMC 설명 규칙)
  - [사후]: 결과를 본 뒤 정한 분석(v5.3 사후 점검, 각 보고의 사후 절)
  - [기술]: 문턱 없이 미리 정한 보고 항목
- **7절이 사용자가 정할 주장 문구다.** 4절의 [결정 n]이 7절의 항목 n이다.
- **먼저 알릴 것**: `A_dithered_world_models/paper/README.md`(맥북 수정 시각 10-24 14:55Z)는 사용자가 10-24에 JKAIA 2026 포스터 원고용으로 "추천 1~5"를 승인했다고 적었다. 투고처도 JKAIA·TMLR·ICLR 2027 워크숍으로 바꿨다고 적었다. LOG와 MAIN_STATUS에는 이 기록이 없다. 그래서 7절에서는 항목마다 그 원고의 현재 문구를 함께 적었다([불일치 D2](#10-불일치-목록)).

---

## 1. 한눈에

같은 비트 수에서 결정적 반올림(det, round-to-nearest)은 1-step 예측 오차로는 보이지 않는 계획 손상을 만든다. 모든 후보가 공유하고 반복마다 새로 뽑는 dither(sh-iter)로 바꾸면 학습 없이 그 손상 대부분이 사라진다. 두 계열 모두 손상 칸 대부분에서 det의 1-step 오차가 sh-iter보다 크지 않았다(H1 성립, 손상 칸 36개 중 29개). sh-iter는 TD-MPC2 Meta-World 손상 칸 19개 모두와 DINO-WM 5칸 중 4칸에서 det 손상의 절반 이상을 없앴다(H3a 성립, go). 기전은 "det 오차가 최적화 반복 내내 남고, 계획기가 모집단을 그 오차가 유리한 쪽으로 옮긴다(drift)"로 좁혀졌다. 근거는 H2a′ 4/4, H2c 3/4, H4a 두 계열 성립이다. 다만 H2a′는 E1 뒤 사전등록이고, H4a는 경계 성립이다. 처음 설계(ED §0)는 다음 네 가지를 주장하려 했다. (1) OEI(계획기 이용 지수)로 오차 이용을 직접 재고, OEI가 1-step 오차보다 손상을 잘 예측한다는 진단, (2) 두 계열 × 두 계획기 전부에서의 처방, (3) 강한 PTQ 위의 추가 이득, (4) 지연–성공률 Pareto. 결과에 따라 네 곳을 좁혔다. 첫째, OEI 가설 H2a·H2b가 불성립했다. 그래서 진단 주장을 빼고 기전을 "반복에 걸쳐 지속되는 오차의 이용"으로 좁혔다(v5.2, 사용자 승인). 둘째, DMC에서 H3a가 불성립했고 walker 3비트에서는 dither가 해로웠다. 그래서 처방 주장은 Meta-World와 DINO-WM에 대해 한다. DMC는 det의 유리한 오차 가운데 반복 사이에 지속되는 몫(S_det)이 작다는 사전등록 설명("설명됨", ρ 0.83)으로 서술한다. 셋째, 강한 PTQ 위의 추가 이득은 TD-MPC2에서만 문턱 경계로 성립했다(10/19). DINO-WM에서는 최선 PTQ가 손상을 거의 없애 불성립했다(1/3). 넷째, 실제 정수 kernel에서 dither의 추가 지연은 측정 잡음 수준이었다(사전 정의 계산 최대 6.9%). 그러나 정수 kernel이 fp32보다 약 18% 느려, 낮은 비트로 얻는 배치 이득은 보이지 못했다. 또 MPPI 반복이 1회면 sh-iter도 det만큼 해로웠다. 그래서 처방을 "반복하는 계획기"로 한정하는 것이 자료와 맞는다. 남은 일은 7절의 주장 문구 13개를 정하고 집필하는 것이다.

---

## 2. 범위와 경과

### 2.1 모델, 계획기, 칸

| 계열 | 체크포인트 | 계획기(기본값 고정) | 확증 칸 | 짝 | 장소 |
|---|---|---|---|---|---|
| TD-MPC2 | 공식 5M 단일 과제, seed 1·2·3(HF `nicklashansen/tdmpc2`) | MPPI: 후보 512(정책 prior 24 포함), elite 64, 반복 6, horizon 3, temperature 0.5 | Meta-World 8과제 × k 3개 = 24칸, DMC 4과제 × 3 = 12칸, 합 36칸 | 90짝(체크포인트 3 × 30 ep) | 서버 GPU 0, CUDA MPS 작업자 16개 |
| DINO-WM | 공식 PushT·Wall·PointMaze(OSF) | open-loop CEM: 후보 300, top-30, horizon 5, opt_steps PushT 30·Wall 10·PointMaze 10 | PushT {3, 4, 5}, Wall {4, 5, 6}, PointMaze {2, 3, 4}, 합 9칸. 탐색 칸 Wall k=3 | 100 ep(eval 쌍) | 서버 GPU 1(일부 칸 GPU 0) |

- 출처: DM §0·§3 6.0·§4, E1T §1, E1D §1
- **양자화 위치**: TD-MPC2는 dynamics·reward MLP 은닉 활성 4층(Mish 뒤), DINO-WM은 예측기 MLP GELU 출력 6층이다. 층마다 균일 격자 [min, 99.99 백분위]를 보정 분할에서 잡았다(E0T §9, E1D §1).
- **비트 수 선택**(held 20%p 규칙, DM §0.2·§4.1)
  - 1차 k: push·pick-place·drawer-open·button-press·sweep-into·quadruped·walker 3, door-open·peg-insert·window-open·finger 4, cheetah 5, PushT 4, Wall 5, PointMaze 3
  - drawer-open과 PointMaze는 20%p에 이르는 k가 없어 R2-13 대체 규칙을 썼다. sweep-into는 k=3 하락이 정확히 20.0%p였다. Wall은 규칙 문자 그대로 k=5(정확히 20%p, p = 0.22)를 골랐고, k=3을 탐색 칸으로 덧붙였다.
- **변형**(DM §2 4.2, §0.4, §4.2): none, det, ind(후보마다 독립·매 forward 새로), **sh-iter**(후보 공유·매 forward 새로, 주 처방), sh-step(계획 안 고정), sh-ep(episode 고정), det+shift(공유 격자 offset을 매 forward 옮김)
  - DINO-WM은 open-loop라 sh-step과 sh-ep가 같은 조건이다. 하나(sh-plan)만 돌렸다.
- **E0 재현**
  - TD-MPC2 R2-8b(40 ep): det 하락 mw-push 62.5%p(파일럿 65), pick-place 57.5%p(55), cheetah 53.1%(52.4). sh-iter 손상 제거 96%(92), 61%(82, CI 겹침), 20%(10)(E0T §4)
  - DINO-WM R2-13(50 ep): PushT k=4 det 하락 +18%p [4, 32], sh-iter 감소율 1.00. Wall k=3 +36%p [20, 52], 1.00. 파일럿 OEI 값 +0.30 / −0.09를 +0.299 / −0.103으로 재현(E0D §3)
- **mujoco-py**: 사용자 공간 Mesa 24.3.4(micromamba)로 빌드했다. Meta-World와 PointMaze를 서버에서 돌렸고, RunPod 대체 경로는 쓰지 않았다(E0T §3, E0D §6).

### 2.2 단계

| 단계 | 내용 | 판정 대상 | 보고 |
|---|---|---|---|
| E0 | 환경, 재현, k 선택, 처리량 | — | E0T, E0D |
| E1 | 주 벤치마크. TD-MPC2 36칸 × 7변형 × 90짝(20,520 ep, 4,320 job), DINO-WM 9칸 + 탐색 1칸 × 6변형 × 100 ep | H1, H3a, H2a, H2b, go/no-go | E1T, E1D |
| E1 추가(v5.2) | TD-MPC2 H2a 3칸 독립 draw 재측정(810 ep, E1과 비트 단위 동일). DINO-WM (b1)·(b2)·(b3) | H2a_indep(사후 정정), PushT의 H2a′ | E1T §15, E1D §8 |
| E4 | 이론 유도와 모의(CPU, 사전 선언 격자 504 job, 계획기 실행 922,368회). E1 결과 파일은 읽지 않았다 | — | E4 |
| E2 | TD-MPC2 OFAT 20설정(4,482 job, 22,410 ep). DINO-WM PushT k=4의 N 6점, elite 비율 4점, opt_steps 3점 | H2a 재현, H2c, (b1), (b2), (b3) 기술 | E2T, E2D |
| E3 | 2축 대조(TD-MPC2 12과제 × 8변형, DINO-WM 2 × 2), 고정 dither 표 크기 sweep(TD-MPC2 4과제, DINO-WM 3과제 + 탐색) | H4a, H4b, DMC 설명 | E3T, E3D |
| E5 | PTQ 기준선(TD-MPC2 4종, DINO-WM 7종) 중 최선 B 위의 det·sh-iter, 가중치 양자화(탐색), QAT | H3b, QAT 보고 규칙 | E5T, E5D |
| E7 | TD-MPC2 융합 Triton int8·int4 kernel. 성공률 1,296 job, 지연 96조건 × 5회전 | H5 | E7T |
| 사후(v5.3) | TD-MPC2 QAT+none 198 job, 짝 모드 지연 재측정(96조건 × 10회전) | 판정 없음(기술) | PH |
| E6(선택) | 하지 않음 | H6 | — |

### 2.3 장비

- 대학 서버 `kiis-mvf-gpu`(Tailscale과 Windows 중계): Ubuntu 22.04 컨테이너, RTX A5000 24GB × 2(드라이버 560.35.05, CUDA 12.6), 64스레드. sudo·Docker·conda·EGL이 없어 사용자 공간 uv venv를 썼다(E0T §2, E0D §2).
  - TD-MPC2: venv `a-mw`, torch 2.7.1+cu126, tdmpc2 commit `332f19e`
  - DINO-WM: venv `a-dinowm`·`a-dinowm-pm`, torch 2.4.1+cu124, dino_wm commit `0a9492f`
- **같은 장비 원칙**: TD-MPC2의 모든 실행은 GPU 0에서 했다. DINO-WM은 칸 하나를 한 GPU에서만 돌렸다. 두 GPU의 DINO-WM 계획은 비트 단위로 같다(E0D §2).
- **MPS**: TD-MPC2는 MPS 작업자 16개로 1.6~1.8배 빨랐다. MPS 유무로 episode 수치는 바뀌지 않는다(E0T §2·§5·§8).
- 맥북은 파일 편집, 결과 읽기, 몇 초짜리 점검에만 썼다. **A는 RunPod를 쓰지 않았다.**

### 2.4 경과 (UTC)

| 시각 | 일 |
|---|---|
| 10-21 13:00 | 본 실험 시작 |
| 13:14~18:09 | E0 TD-MPC2: 환경, mujoco-py, R2-8b 재현, held k 선택, 처리량(E0T) |
| 13:08~19:58 | E0 DINO-WM: 환경, R2-13 재현, k 선택, PointMaze 동작 확인(E0D) |
| 18:14 | DM v1 고정. 같은 시각 E1 TD-MPC2 시작(GPU 0, MPS 16) |
| 19:11 | 칸 단위 job 95개가 GPU 메모리 부족으로 실패. 실행 순서를 고쳤다(2.5절 사고 1) |
| 20:01 | DM v2(DINO-WM 세부). E1 Wall·PointMaze 시작(GPU 1) |
| 22:54 | DM v3(사용자 승인): DINO-WM OEI 확증 지표는 비양자화 모델 비용 기준, PushT eval 쌍 다시 뽑기 |
| 10-22 06:08 | TD-MPC2 E1 episode 4,104개 완료. 칸 단위 job 95개가 MPS Error 205로 실패(사고 2). PushT E1 시작 |
| 06:55 | DM v4: 두 계열 공통 판정 계산(TD-MPC2 E1 결과 전) |
| 07:25~08:52 | E4 모의 본 실행(CPU 8개) |
| 07:40 | E1 TD-MPC2 보고: H3a Meta-World 성립, DMC 불성립, H2a·H2b 불성립 |
| 09:00 / 09:30 | DM v5(E2·E3·E5·E7 세부) / v5.1(MSE α 격자, H2c는 elite 수 고정 sweep, var_cand) |
| 09:50 | DM v5.2 사용자 승인: 기전 주장 좁힘, OEI 진단 주장 뺌, H2a 독립 draw 재측정, E2 층 (b), DMC는 설계 7장 대응 |
| 13:15 / 18:56 | E1 Wall·PointMaze 완료 / E1 PushT 완료 |
| 19:04~19:33 | DINO-WM v5 사슬 시작(GPU 1, E2 N sweep). TD-MPC2 H2a 독립 draw 재측정(19:05~19:33) |
| 19:33 | E2 TD-MPC2 시작 |
| 19:40 | E1 DINO-WM 보고. **go/no-go = go** |
| 10-23 06:18 | E2 TD-MPC2 완료. MPS Error 205가 다시 나 GPU 0 queue가 멈췄다(사고 3) |
| 08:33~08:35 | MPS daemon 재시작, E3 TD-MPC2 시작 |
| 13:55 | E2 DINO-WM N sweep 완료, E3 DINO-WM 시작 |
| 14:51 | E3 TD-MPC2 완료, E5 TD-MPC2 시작 |
| 15:14 무렵 | 맥북 재부팅. Tailscale과 RunPod 감시 스크립트가 꺼졌다. 15:27에 점검을 재개했다(사고 4) |
| 15:55 / 16:05 | E2 DINO-WM N 보고 / E2·E3 TD-MPC2 보고 |
| 16:10 | 계열 합산: H2c 성립, H2a′ 성립, H2a 재현·H2a_indep 불성립, H4b 불성립, DMC는 S_det로 서술 |
| 22:11 / 22:22 | E5 TD-MPC2 완료(E7 성공률 시작) / E3 DINO-WM 완료(E5 DINO-WM 시작) |
| 22:27 / 22:39 / 22:40 | E5 TD-MPC2 보고(H3b 10/19) / E3 DINO-WM 보고 / H4a 계열 합산(두 계열 성립) |
| 10-24 01:29~02:15 | E7 성공률 완료, MPS를 멈추고 배타 GPU 0에서 지연 측정 |
| 02:17~02:30 | GPU 0에 DINO-WM 두 번째 사슬. E7 보고(H5 성립) |
| 02:38 | DM v5.3 사용자 승인: QAT 보고 방식, 사후 점검 둘 |
| 02:54 | 사용자 요청으로 정기 점검을 멈췄다 |
| 11:51 / 12:29 | E5 DINO-WM 실행 완료 / 보고(H3b 1/3) |
| 14:24~14:46 | 사후 점검 1: TD-MPC2 QAT+none |
| 14:41 | DINO-WM v5 실행 완료(E2 elite·opt_steps). A 본 실험 실행이 모두 끝났다 |
| 14:51~16:26 | 사후 점검 2: 짝 모드 지연 재측정. A의 서버 GPU 작업이 모두 끝났다 |
| 16:38 / 16:39 | E2 DINO-WM elite·opt_steps 보고 / 사후 점검 보고. 단계 분석이 모두 끝났다 |
| 16:42 | 이 보고를 맡겼다 |

- 출처: LOG.md, MAIN_STATUS 작업 표·점검 기록, 각 보고의 실행 절. 문서마다 시각이 조금 다른 곳은 [불일치 D3·D4](#10-불일치-목록)에 적었다.

### 2.5 주요 사고와 수정

모두 "결과·판정 영향 없음"으로 기록됐다.

| # | 사고 | 영향 | 수정 | 출처 |
|---|---|---|---|---|
| 1 | E1 TD-MPC2 칸 단위 job(prederr·decision) 95개가 GPU 메모리 부족으로 실패했다(10-21 19:11Z). MPS 작업자 16개가 큰 held 후보 집합을 동시에 올렸다 | episode job은 정상(911개 완료). 잃은 결과 없음 | `run_e1_split.sh`: episode는 16개씩, 칸 단위 job은 그 뒤 3개씩, 마지막에 2개씩 | LOG 19:11Z |
| 2 | **MPS Error 205, 1차**(10-22 06:08Z). PushT E1 runner가 MPS 아닌 프로세스로 GPU 0에 올라간 뒤, 새 MPS client가 CUDA 초기화에 실패했다. 칸 단위 job 95개 실패 | episode 4,104개는 그 전에 끝남. 손실 없음 | MPS를 멈추고 MPS 없이 작업자 2개로 재실행(06:20~06:45Z). 이후 "queue가 MPS를 돌리는 동안 GPU 0에 MPS가 아닌 프로세스를 올리지 않는다" | LOG 06:19Z·08:50Z |
| 3 | **MPS Error 205, 2차**(10-23 06:18Z). E2가 끝난 뒤 MPS server가 새 client를 받지 못해, queue가 E3 시작 전 점검에서 멈췄다. 외부 프로세스는 없었고 /dev/shm도 정상이었다 | **GPU 0이 약 2.3시간 놀았다.** 결과 영향 없음 | E3를 띄우기 전에 MPS daemon을 다시 켰다(`mps.sh stop 0` → `start 0`). 이후 단계 전환마다 client가 정상으로 붙는지 확인했다 | LOG 08:35Z, E3T §1 |
| 4 | **맥북 쪽 연결 끊김**. 10-23 08:35Z 점검 때 Tailscale이 꺼져 있었다. 15:14Z 무렵 재부팅으로 Tailscale과 RunPod 예산 감시 스크립트가 꺼졌다 | 서버 작업은 계속 돌았다. 점검만 늦어졌다 | 사용자가 Tailscale을 켜고, 조정자가 감시 스크립트를 다시 켰다(pod 없음, 누적 $20.27) | LOG 08:35Z·15:27Z, MAIN_STATUS 점검 기록 |
| 5 | E0 중 B의 진단 작업 3개가 A의 GPU 0에서 돌았다(10-21 13:48~14:17Z, 14:40~14:42Z) | 처리량 측정 창만 오염 | 그 창을 처리량 근거에서 뺐다. B 스크립트에 GPU 가드를 더했다 | E0T §10.3, MAIN_STATUS 14:45 |
| 6 | E0 DINO-WM이 GPU 0에서 한 번 OOM(13:52Z, TD-MPC2 14.7GB와 겹침) | 그 job만 결과 없이 재실행 | 배치 E를 줄였다 | E0D §10.3 |
| 7 | 정기 점검을 두 번 멈췄다(사용자 요청). 10-23 02:15~15:27Z(08:35Z에 한 번 점검), 10-24 02:54Z 이후(요청 때만 점검) | 사고 3의 발견이 약 2.3시간 늦어졌다 | 대기열은 job마다 결과 파일을 써서 재개할 수 있었다 | LOG 10-23 08:35Z·15:27Z, 10-24 02:54Z, MAIN_STATUS 점검 기록 02:15 |
| 8 | ρ 문턱 비교의 부동소수점: scipy가 정확히 −0.8·0.8인 Spearman을 ±0.7999999999999999로 낸다 | 고치지 않았다면 H4a가 두 계열 모두 불성립으로 셌을 값이다 | 결과 수치를 보기 전에 두 계열 모두 허용 오차 1e-9로 비교하도록 고쳤다 | E2T §3, E3T §3, E3D §8, LOG 10-23 16:05Z·22:39Z |

- 그 밖의 편차 28가지는 8.1절에 모았다.

---

## 3. 사전등록 판정 전체 표

- **가설군**(ED §6.8)
  - 1차: H1, H3a. det 하락의 실재를 과제 단위 Holm으로 정한다.
  - 2차: H2, H4, H5. 보정 없는 탐색이다.
  - H3b는 어느 쪽에도 적혀 있지 않다. v5 §7.3의 칸별 95% CI 규칙대로 보정 없이 판정했다(조정자 결정, LOG 10-23 22:27Z).
- **집계 기준**
  - H1: 두 계열 모두. H3a: 단위별(TD-MPC2 Meta-World, TD-MPC2 DMC, DINO-WM). go/no-go는 Meta-World와 DINO-WM의 H3a로 정한다.
  - H2a·H2c·H2a_indep·H2a′: 기전 과제 4개(mw-push, mw-pick-place, cheetah-run, PushT) 중 3개 이상
  - H2b: 계열마다. H4a: 계열마다 자격 과제의 과반, 그리고 두 계열 모두. H4b: TD-MPC2 12과제. H3b: 계열마다. H5: TD-MPC2(E7)
- **문턱 운용**: ρ 문턱(H2c, H4a, H4b, DMC 설명)은 점추정으로 판정했다. bootstrap은 보조다. 문턱 비교에는 허용 오차 1e-9를 썼다(2.5절 사고 8).

| 가설 | 출처 분류 | 계열·단위 | 판정 | 핵심 수치 [95% CI] | 근거 |
|---|---|---|---|---|---|
| **H1**(1차) | [사전등록] v1, v4 계산 | TD-MPC2 Meta-World | 성립 | det 하락 실재(과제별 Holm) 19/24칸. 그중 14칸에서 det의 계획 시점 1-step 오차 ≤ sh-iter. 예: pick-place k=3에서 오차는 0.223 대 0.388인데 하락은 37.8 대 7.8%p | E1T §3·§4 |
| | | TD-MPC2 DMC | 성립 | 자격 12/12칸, 그중 10칸 | E1T §4 |
| | | DINO-WM | 성립 | 자격 5/9칸(PushT k=3·4·5, PointMaze k=2·3). 5칸 모두 det 오차 < sh-iter(짝 CI 상한 < 0) | E1D §3·§4 |
| | | 전체 | **성립(두 계열)** | 손상 칸 36개 중 29개(14 + 10 + 5, 이 보고 계산) | — |
| **H3a**(1차) | [사전등록] v1, v4 | TD-MPC2 Meta-World(24칸) | **성립** | 자격 19칸 모두 red_sh-iter ≥ 0.5(19/19). 해로운 칸 없음. red 중앙값 0.95(이 보고 계산, JSON 0.947). 가장 약한 칸은 window-open k=4 0.53 [0.00, 0.82] | E1T §5 |
| | | TD-MPC2 DMC(12칸) | 불성립 | 5/12(42%). **walker k=3에서 sh-iter가 det보다 해롭다**: D_sh-iter − D_det +22.1% [18.0, 26.3] | E1T §5 |
| | | TD-MPC2 전체(36칸) | 불성립 | 24/31(77%)로 비율은 넘지만 walker k=3 해로운 칸 | E1T §5 |
| | | DINO-WM(9칸) | **성립** | 4/5(80%, 문턱 75%). PushT k=5만 0.20 [−2.00, 0.86]. 해로운 칸 없음. 하한이 가장 큰 칸은 Wall k=4의 +4%p(문턱 +5%p) | E1D §5 |
| go/no-go | [사전등록] 설계 6.1 | Meta-World + DINO-WM | **go** | 두 단위 모두 H3a 성립 | LOG 10-22 19:40Z |
| **H2a**(동결 OEI) | [사전등록] v1, v4 §6.2 | TD-MPC2 3과제(E1) | 불성립 0/3 | det OEI의 CI 상한 < 0은 셋 다 맞다. 그러나 sh-iter OEI도 유의하게 음수다: push −20.0 [−22.9, −17.5], pick-place −39.0 [−42.3, −35.8], cheetah −2.48 [−2.67, −2.29] | E1T §6 |
| | | PushT k=4(E1) | 성립(경계) | det −0.297 [−0.350, −0.248], sh-iter +0.0048 [−0.000008, +0.0099] | E1D §6 |
| | | 전체 | **불성립** | 1/4 | LOG 10-22 19:40Z |
| H2a 재현(E2 새 seed) | [사전등록] v5 §7.1 | TD-MPC2 3과제 | 불성립 0/3 | sh-iter −18.4 [−20.8, −16.2], −37.5 [−40.9, −34.3], −2.61 [−2.85, −2.41] | E2T §4.1 |
| | | 전체(PushT는 E1 값) | 불성립 | 1/4 | LOG 10-23 16:10Z |
| H2a 독립 draw 재측정(E1 칸) | [사후] 사후 정정(v5.2 §9.1). 동결 판정을 대신하지 않음 | TD-MPC2 3과제 | 1/3 | 재실행 810 ep가 E1과 비트 단위로 같다. cheetah sh-iter +0.10 [−0.06, +0.26]만 성립. pick-place −13.66 [−16.80, −10.35], push −4.56 [−6.27, −2.84] | E1T §15 |
| | | PushT를 더함 | 불성립 | 2/4 | E1T §15 |
| (b1) H2a_indep | [E1 뒤 사전등록] v5.2 §9.1, E2 seed | TD-MPC2 3과제 | 1/3 | cheetah −0.02 [−0.17, +0.13]만 성립. push −5.48 [−7.42, −3.59], pick-place −11.17 [−14.93, −7.41] | E2T §6.1 |
| | | PushT(E1, 표본 밖) | 성립 | 동결 정의가 곧 독립 draw판이다(H2a와 같은 계산) | E1D §8.1 |
| | | 전체 | **불성립** | 2/4 | LOG 10-23 16:10Z |
| **(b2) H2a′**(지속 오차 이용) | [E1 뒤 사전등록] v5.2 §9.1 | TD-MPC2 3과제(E2 seed) | 3/3 | drift_det −53.3 [−60.0, −46.8], −46.2 [−52.4, −40.2], −2.53 [−2.77, −2.29]. r = drift_sh-iter / drift_det: 0.097 [0.065, 0.133], 0.242 [0.171, 0.318], 0.030 [−0.023, 0.082] | E2T §6.2 |
| | | PushT k=4(E1, 표본 밖) | 성립 | drift_det −0.269 [−0.318, −0.224], r −0.013 [−0.032, +0.004] | E1D §8.1 |
| | | 전체 | **성립** | 4/4 | LOG 10-23 16:10Z |
| H2a′ 양성 대조(판정 밖) | [E1 뒤 사전등록] 보조 | TD-MPC2 | 예상대로 | r_sh-step 0.70 / 0.81 / 1.30, r_sh-ep 0.97 / 0.87 / 1.25(모두 CI 하한 > 0.5) | E2T §6.2 |
| | | PushT | **예상과 다름** | r_sh-plan +0.011 [−0.009, +0.031](예상: CI 하한 > 0.5) | E1D §8.1 |
| **H2b** | [사전등록] v1, v4 §6.2 | TD-MPC2(216점) | 불성립 | 과제 안 Spearman 차의 평균 0.079 < 0.15. 보조(과제를 섞은 원값) −0.214 | E1T §7 |
| | | DINO-WM(45점) | 불성립 | −0.144(PushT −0.556, Wall +0.312, PointMaze −0.189) | E1D §7 |
| | | 전체 | **불성립** | 두 계열 모두 | — |
| **H2c** | [사전등록] v1 가설·문턱, v5 §7.1 + v5.1 §8.2(elite 수 고정 sweep, E2 자료 전) | TD-MPC2 3과제 | 3/3 | ρ_s(N, \|OEI_det\|) / ρ_s(N, D_det): push 1.00 / 1.00, pick-place 0.90 / 0.90, cheetah 1.00 / 1.00 | E2T §4.2 |
| | | PushT k=4(6점) | 불성립(경계) | +0.943 / **+0.771**. D_det가 N = 300·512·1024에서 +24·+23·+22%p로 평탄하다. 결합 bootstrap에서 P(ρ_D ≥ 0.8) = 0.55 | E2D §0·§4 |
| | | 전체 | **성립** | 3/4 | LOG 10-23 16:10Z |
| **H4a** | [사전등록] v1, v5 §7.2(공유 표 4수준, 이웃 자리바꿈 한 번 허용, 자격, 과반) | TD-MPC2(자격 4과제) | 성립(경계) 3/4 | Spearman(N_eff, D_sh-fixed): push −0.80, pick-place −0.80, cheetah −1.00, walker +0.40 | E3T §5.1 |
| | | DINO-WM(자격 2과제) | 성립(경계) 2/2 | PushT k=4 −0.80, PointMaze k=3 −1.00. Wall k=5는 E1 하락이 실재하지 않아 비자격(기술, −0.20) | E3D §3 |
| | | 전체 | **성립(두 계열)** | 성립 5과제 중 3과제가 정확히 −0.8 | LOG 10-23 22:40Z |
| **H4b** | [사전등록] v1, v5 §7.2(확증 = TD-MPC2 12과제, 한 번 계산) | TD-MPC2 12과제 | **불성립** | Spearman(near-tie 밀도(마지막 반복), CRN 이득) = +0.03. 보조(첫 반복 집합) −0.35 | E3T §6 |
| H4b 보조 15과제 | [사전등록 보조] | + DINO-WM 3과제 | (판정 밖) | 마지막 반복 −0.32, 첫 반복 +0.21(이 보고가 JSON으로 다시 계산: −0.3235, +0.2055) | LOG 10-23 16:10Z |
| **DMC 설명** | [E1 뒤 사전등록] v5 §7.2, v5.2 §9.2(한 번 계산) | TD-MPC2 12과제(E3 seed) | **"설명됨"** | Spearman(S_det, red_sh-iter) = +0.83(0.827). Spearman(밀도, red) = −0.27. S_det: DMC 0.33~0.52, Meta-World 0.54~0.73. 이 결과와 무관하게 DMC H3a 불성립은 그대로다 | E3T §7 |
| **H3b** | [사전등록] v1, v5 §7.3 | TD-MPC2(19칸) | 성립(경계) | 정확히 10/19(Meta-World 6/12, DMC 4/7). pick-place k=3·drawer-open k=3은 CI 하한이 정확히 0이라 불통과. walker k=4 하한 −0.0002 불통과, quadruped k=4 하한 +0.0008 통과 | E5T §5 |
| | | DINO-WM(3칸) | **불성립** 1/3 | PointMaze k=3 +12%p [4, 20]만 통과. PushT k=3 −2 [−6, 2], k=4 +2 [−4, 8]. PushT에서는 최선 PTQ만으로 D(B+det)가 +3·+4%p다 | E5D §5 |
| QAT 보고 규칙 | [사전등록] 설계 §6.5, v5 §7.3 | TD-MPC2(11칸) | 조건 안 걸림(0/11) | D(QAT+det) 0.29~0.85, CI 상한은 모두 0.38 이상. v5.3 §10.1 단서: 11칸 모두 QAT+det가 B+det보다 나빠 약한 기준선이다 | E5T §6 |
| | | DINO-WM(PushT k=4) | 조건 안 걸림 | D(QAT+det) +6%p [−1, 13], 상한 13 ≥ 5. QAT+det − B+det = +2 [−5, +10]. 조정자 판단: TD-MPC2와 같은 단서는 달지 않는다 | E5D §6, LOG 10-24 12:29Z |
| **H5** | [사전등록] v1, v5 §7.4 | TD-MPC2(E7) | **성립**(네 읽기 모두) | ① 12조건 모두 dither 오버헤드 ≤ 10%: int8 −5.3~+3.8%, int4 그릇판 −2.7~+5.8%, packed int4 −5.8~+6.9%. ② Pareto 지배는 mw-push 사례뿐이다(kernel마다 4개). 같은 반복 수 사례는 성공이 같고 지연 차 3~5%가 5회전 모두에서 유지되지 않는다 | E7T §4 |
| 사후 1: QAT+none | [사후] v5.3 §10.2 | TD-MPC2(11칸) | (기술) | D(QAT+none) 0.20~0.82, 11/11칸에서 CI 하한 > 0 | PH §4 |
| 사후 2: 지연 재측정 | [사후] v5.3 §10.3 | TD-MPC2(36조건) | (기술) | 짝 오버헤드 중앙값 −3.2~+2.7%(모두 ±3.2% 안). CI는 35/36조건에서 0을 포함한다. CI 상한 > 10%가 16/36(최대 35.5%) | PH §3 |
| H6 | [사전등록] v1(선택) | — | 하지 않음 | E6 생략 | MAIN_STATUS |

- **판정 밖 탐색 칸**: Wall k=3(파일럿의 k). det 하락 +50%p [39, 61], dither 4종의 red 0.92~1.08이다(E1D §9.2). H1·H3a에 넣지 않았고, 넣어도 판정은 같다(E1D §5).
- **다중 비교**
  - Holm은 1차 가설군의 자격 칸을 정하는 데 쓰였다. DINO-WM은 보정 없는 CI 하한 > 0 기준으로도 자격 칸이 같다(E1D §3). TD-MPC2에서는 door-open k=5와 sweep-into k=4가 보정 전 p < 0.05였지만 Holm에서 떨어졌다(E1T §3).
  - H2·H4·H5의 성립은 보정 없는 탐색 결과로 표시한다. H3b 칸별 CI도 보정하지 않았다.
- **표 읽는 법**: "(경계)"는 규칙의 문턱에 정확히 걸렸거나 한 칸·한 순위 차이로 판정이 갈린 경우다. 표시 방식은 [결정 7]이다.

---

## 4. 최종 주장 (DM v5.2 §9·v5.3 §10 범위)

- 아래 주장은 3절 판정이 받치는 범위 안에서 썼다. [결정 n]이 붙은 곳은 7절의 추천안 문구를 기준으로 썼고, 사용자 결정에 따라 바뀐다.

### 4.1 논문이 하는 주장

1. **현상과 진단: 같은 비트 수에서 det는 1-step 오차로는 보이지 않는 계획 손상을 만든다.** [결정 10]
   - 근거: H1이 두 계열에서 성립했다. 손상 칸 36개 중 29개에서 det의 1-step 오차가 sh-iter 이하다(3절). det 하락은 최대 100%p다(button-press k=2, E1T 부록 A).
   - 같은 비트에서 확률적 변형들의 항목별 오차 분포는 같은데 손상은 크게 갈린다. E1 Meta-World 자격 칸 평균 하락은 det 44.6, ind 29.2, sh-iter 4.0, sh-step 34.9, sh-ep 50.1, det+shift 4.0%p다(E1T §9).
   - 한정: 3·5-step 오차는 det를 가른다(PushT k=4의 5-step: det 0.250 대 sh-iter 0.042, PointMaze k=3: 1.083 대 0.483; E1D §4). 비트 수가 다른 칸을 섞으면 1·3-step 오차도 손상과 순위가 잘 맞는다(TD-MPC2 과제 안 ρ(1-step, D) 0.47~0.94, E1T §7). 그래서 "같은 비트 안에서 반올림 방식 사이의 차이"로 한정한다(설계 2.3절 원칙).
2. **처방: 같은 비트 수에서 학습 없이, 모든 후보가 공유하고 반복마다 새로 뽑는 dither(sh-iter)가 det 손상 대부분을 없앤다.** [결정 2·3]
   - 근거: H3a가 Meta-World 19/19(red 중앙값 0.95), DINO-WM 4/5에서 성립했고, 두 단위 모두 해로운 칸이 없다(3절). E3 새 seed에서도 Meta-World red가 0.75~1.22다(E3T §4.1). E7 실제 int4 kernel에서 E1 fake-quant의 방향과 크기가 재현됐다(반복 6의 cheetah: det 0.703, dither 0.538 대 E1 0.716, 0.518; E7T §5).
   - 한정: 반복하는 계획기에 한정한다 [결정 2]. 처방의 과제 범위는 Meta-World와 DINO-WM이다 [결정 3]. det 손상 자체가 작은 칸(PushT k=5, +5%p)에서는 효과가 불분명하다(E1D §5).
3. **기전: 계획기는 반복에 걸쳐 지속되는 오차를 이용한다.** [결정 1]
   - 근거
     - H2a′ 4/4: det drift는 4과제 모두 음수이고, sh-iter drift는 det의 24% 이하다.
     - TD-MPC2 양성 대조: 같은 공유 dither를 계획 동안(sh-step) 또는 episode 동안(sh-ep) 고정하면 drift의 70~130%가 돌아온다(E2T §6.2).
     - H2c 3/4: elite 수를 고정하면 후보 수와 함께 det의 \|OEI\|와 손상이 커진다.
     - E3: 같은 거칠기에서 지속의 비용 D(shstep_g) − D(shiter_g)가 12/12에서 0보다 크다(E3T §5.3).
     - det drift가 반복 수와 함께 커진다: TD-MPC2 push 0 → −46.0 → −53.3 → −59.6(E2T §5), PushT −0.212 → −0.229 → −0.269(E2D §7.3).
     - 결정적 반올림을 유지한 채 격자만 반복마다 옮겨도(det+shift) sh-iter와 같은 효과가 난다(E1T §9) [결정 12].
   - 한정
     - 사전등록 OEI 가설(H2a·H2b)은 불성립이다(4.2절). H2a′는 E1 뒤 사전등록이다.
     - drift에는 영역 효과가 섞인다. PushT k=4 det drift의 약 2/3가 영역 효과다(E1D §8.2). TD-MPC2에서는 영역 효과가 반대 부호다(E2T §6.3).
     - 지속만으로는 이용되지 않는다. DINO-WM의 전체 크기 고정 표(sh-plan)는 drift를 만들지 않았고, 표를 거칠게 해야 이용됐다(E3D §3.4·§7).
     - Meta-World의 sh-iter에도 작은 drift가 남는다(E2T §6.3).
4. **오차 구조: 고정 dither는 표가 작을수록(거칠수록) 해롭다.** [결정 11·7]
   - 근거: H4a가 두 계열에서 성립했다(경계). 지속의 비용은 표가 거칠수록 크다(push: 층당 1 / group 32 / 기본에서 +0.62 / +0.34 / +0.26, E3T §5.3). DINO-WM 탐색 칸 Wall k=3은 층당 offset 하나(L3)에서만 +37%p로 무너졌다(E3D §6).
   - 한정
     - 성립 5과제 중 3과제가 정확히 문턱이다. walker는 반대(+0.40)다.
     - 공유·새로움의 효과가 계열마다 다르다. TD-MPC2는 공유 이득 9/12, 지속 비용 10/12·11/12이고, DINO-WM 전체 크기 표에서는 2 × 2 대비가 모두 CI에 0을 포함한다(E3T §4.2, E3D §4.2).
     - 설계 6.3절이 예측한 1/N_eff 정량 법칙과 "두 모델에서 같은 기울기"는 보이지 못했다. TD-MPC2의 후보 간 오차 분산(var_cand)은 표 크기를 따르지 않았다(E3T §5.2).
5. **처방이 약한 곳(DMC)은 기전 변수로 설명된다.** [결정 3]
   - 근거: DMC 설명 규칙이 "설명됨"이다(ρ 0.83, E1 뒤 사전등록, 새 seed). S_det가 가장 작은 두 과제 walker(0.33)와 quadruped(0.37)가 sh-iter의 효과가 없거나 해로운 과제다(E3T §7). [사후] 계열 안에서도 ρ가 양수다(Meta-World 0.57, DMC 1.00; E3T §8).
   - 한정: DMC H3a 불성립과 walker k=3의 해로움은 그대로다. 상관 결과이고, DMC는 4과제다.
6. **강한 PTQ와의 결합은 계열마다 다르고, QAT에 대해서는 결론을 내리지 않는다.** [결정 4·6]
   - TD-MPC2 10/19(경계), DINO-WM 1/3이다. 최선 PTQ(대부분 per-channel 범위)가 det 손상 대부분을 없앤다. Meta-World D(B+det)는 0.000~0.167이고, quadruped k=3은 0.745 → 0.008이다(E5T §5).
   - QAT: TD-MPC2의 사전등록 QAT는 약한 기준선이다(QAT+none 11/11 손상). DINO-WM QAT+det는 최선 PTQ와 구별되지 않는다.
7. **배치: 같은 비트에서 dither의 추가 지연은 측정 잡음 수준이지만, 이 구현에서는 낮은 비트의 지연 이득이 없다.** [결정 5]
   - H5가 성립했다(사전 정의 계산 최대 6.9%). 사후 짝 재측정의 중앙값은 ±3.2% 안이다. 정수 kernel은 fp32보다 17.5~18.2% 느리고(E7T §6), int4와 int8의 지연 차가 없다.
8. **이론(부록): E4의 유도와 모의가 측정의 정성적 모양을 미리 맞혔다**(5.11절).
   - 예: TD-MPC2 정의의 OEI는 새로 뽑는 잡음에서도 음수일 것이다(E4 §7.1, E1 결과를 읽기 전) → E1·E2에서 확인했다. N 의존은 선택 연산자에서 온다(E4 §7.2) → elite 수를 고정할 때만 용량-반응이 나왔다.

### 4.2 숨기지 않을 부정 결과 [결정 8]

- **OEI 가설 둘이 불성립이다.** H2a는 동결 정의로 1/4, 독립 draw로 2/4다. H2b는 0.079와 −0.144다. "OEI가 1-step 오차보다 손상을 잘 예측한다"는 진단 주장은 E2 자료 전에 뺐다(v5.2).
- **H4b 불성립**(+0.03). CRN 이득 자체는 TD-MPC2 9/12과제에서 CI가 0보다 크지만, near-tie 밀도를 따르지 않는다.
- **DMC H3a 불성립**과 walker k=3의 해로움(+22.1%)
- **H3b**: DINO-WM은 불성립(1/3)이고, TD-MPC2는 문턱 경계(10/19)다.
- **H2c의 PushT 몫 불성립**(경계, 손상 조건)
- **DINO-WM sh-plan 양성 대조**가 예상과 달랐다(r +0.011).
- **Wall**: 확증 칸 셋 모두 det 하락이 실재하지 않았다. k=4에서는 det가 none보다 14%p 낫다. Wall은 확증 판정에 기여하지 않았다.
- **E3 사전 예상** "indstep ≈ ind ≈ indep"가 손상에서 3~4/12과제 어긋났다. slot 표가 더 해로웠고 원인은 확인하지 않았다(E3T §4.3).
- **적은 반복**: 반복 1회면 sh-iter가 det만큼 해롭다. 실제 kernel의 반복 3에서는 DMC에서 dither가 해롭다.
- **배치**: 낮은 비트의 지연 이득이 없고, Pareto 사례는 약하다.
- **가중치 SR**(탐색)은 계열마다 방향이 반대다.

### 4.3 쓰지 않을 주장

- "OEI가 1-step 오차보다 손상을 더 잘 예측한다"(H2b, v5.2)
- "확률적 변형에서는 계획기가 오차를 이용하지 않는다(OEI ≈ 0)". 같은 평가 안의 winner's curse(sel_last)는 새로 뽑는 잡음에도 생긴다(v5.2 §9.1).
- "1-step(또는 k-step) 오차는 틀린 지표다"(설계 2.3절)
- "dither는 일반적으로 좋다", 또는 "모든 과제와 계획기 설정에서 좋다"(walker k=3, 반복 1회, DMC 반복 3회)
- "DMC에서도 처방이 성립한다"(H3a 불성립)
- "지속되는 오차는 모두 이용된다"(DINO-WM sh-plan과 전체 크기 고정 표)
- "det 계획기는 모든 칸에서 자신이 고른 계획을 과대평가한다". PushT k=3에서는 det drift가 +0.243, OEI가 +0.217로 반대 방향이다. 영역 효과를 뺀 몫만 음수다(E1D §8.2·§10) [결정 1].
- "반복이 고정 오차의 손상을 키운다"(일반 명제). DINO-WM에서는 opt_steps와 함께 det 손상이 커졌다(+7 → +24%p). 그러나 none 성공률도 48 → 83%로 바뀌었고, TD-MPC2에서는 반복 수와 거의 무관했다(0.27~0.50; E2T §5, E2D §7.3). 계열별 기술로만 쓴다([불일치 D8](#10-불일치-목록)).
- "강한 PTQ 위에서도 dither가 일반적으로 이득이다"(DINO-WM 불성립)
- "QAT로는 손상이 사라지지 않는다", "QAT가 dither를 대체한다"(v5.3 §10.1)
- "낮은 비트 + dither가 지연–성공률 Pareto에서 높은 비트 + det를 이긴다"(사례가 약하고, 정수 kernel이 fp32보다 느림)
- "near-tie가 많은 과제에서 공유(CRN) 이득이 크다"(H4b)
- "1/N_eff 법칙", "두 계열에서 같은 기울기"(정량 형태를 보이지 못함)
- "공유와 새로움이 언제나 필요하다"(DINO-WM 2축 대조)
- "최초"(노벨티는 집필 전에 다시 확인한다. ED §2.2는 "찾지 못했다" 수준이다)

---

## 5. 탐색·서술 결과 (판정 밖)

표시 규칙
- **[기술]**: 결과 전에 보고 항목으로 정했지만 판정이 없는 것
- **[보조]**: 사전등록된 보조 통계
- **[탐색]**: 설계가 탐색으로 지정한 것
- **[사후]**: 결과를 본 뒤 정한 분석

### 5.1 [사후→기술] OEI 분해: OEI = sel_last + drift
- 정의(항등식, E1T §10.1, E4 §2.9)
  - sel_last = e(실행 후보) − ē(마지막 반복 모집단). 마지막 평가 한 번 안의 선택이다.
  - drift = ē(마지막 반복 모집단) − ē(첫 반복 표본 후보). 반복을 거치며 모집단이 모델이 낙관하는 쪽으로 옮겨 간 양이다.
  - E1 TD-MPC2에서는 사후 분석이었고, E2부터는 두 계열 모두 기술 항목으로 기록했다(v5 §7.0).
- **TD-MPC2 E1**(사후, E1T §10.1)
  - det drift: push −49.8, pick-place −50.6, cheetah −2.90
  - sh-step·sh-ep drift: Meta-World −35 ~ −48, cheetah −3.2 ~ −3.4
  - 새로 뽑는 변형(ind, sh-iter, det+shift)의 drift: Meta-World −3 ~ −12, cheetah 0 근처
  - sh-iter의 음수 OEI는 대부분 sel_last다(−15.7, −27.0, −2.51).
- **같은 draw판 대 독립 draw판 sel_last**(기술, E2T §6.2)
  - ind −53.3 → −0.27, −66.7 → −1.84, −6.21 → +0.02
  - sh-iter −13.2 → −0.30, −26.3 → +0.03, −2.54 → +0.06
  - det+shift −10.7 → −0.76, −17.6 → −0.50, −2.02 → +0.03
  - 같은 평가 안의 winner's curse는 독립 draw로 다시 재면 사라진다.
- **DINO-WM**(E1D §8.3, E2D §3.2): det OEI의 약 90%가 drift다(N = 300: −0.269 / −0.297). PushT k=4의 확률적 변형 OEI는 +0.000 ~ +0.005다.
- [사후] −drift를 H2b의 예측 변수로 바꿔도 과제 안 차의 평균은 −0.265다(E1T §10.1). 손상 예측력 문제는 성분 선택으로 풀리지 않는다.
- E4의 결과 전 예측과 같다. TD-MPC2 정의의 OEI는 새 잡음에서도 음수, DINO-WM `oei_ref_mf`는 0이다. 모의에서도 새 잡음 3,720칸의 94.4%에서 같은 평가 sel_last의 CI가 0보다 작았다(E4 §7.1).

### 5.2 [기술] 영역 효과 대조 (b3)
- drift_ref는 비양자화 계획기가 지나간 첫·마지막 모집단을 양자화 모델로 평가한 drift다. drift − drift_ref가 그 계획기 자신의 몫이다(v5.2 §9.1).
- **TD-MPC2**(E2 base, E2T §6.3)
  - det drift_ref는 +9.29 [+1.54, +17.06], +3.94 [−2.10, +9.80], +0.88 [+0.80, +0.96]이다. det drift(−53.3, −46.2, −2.53)와 **반대 부호**다.
  - 그래서 det 자신의 몫(−62.6, −50.2, −3.41, 모두 CI < 0)이 drift보다 크다. sh-step·sh-ep도 같은 모양이다.
  - sh-iter의 작은 Meta-World drift
    - push: −5.18 중 −4.16이 영역 효과다. 남는 몫은 −1.02 [−3.96, +1.48]이다.
    - pick-place: −11.20 중 −6.07이 영역 효과다. 남는 몫은 −5.13 [−8.59, −1.74]이다.
    - cheetah: 0 근처다.
  - 남는 몫의 원인(비선형 전파 편향이 후보, E4 §2.8)은 가르지 않았다.
- **DINO-WM**(E1 none 모집단으로 오프라인 계산, E1D §8.2)
  - PushT k=4: drift_ref −0.177 [−0.221, −0.137]로 det drift −0.269의 약 2/3다. det 자신의 몫은 −0.092 [−0.108, −0.076]이다.
  - PushT k=3: drift_ref +0.448이 det drift를 +0.243으로 만들었다. 자신의 몫은 −0.206이다.
  - **det 계획기 자신의 몫은 E1 10칸 모두 0 이하다.**
- 그래서 DINO-WM에서 OEI 전체의 부호는 칸 사이 손상과 단조로 이어지지 않는다. 손상이 가장 큰 PushT k=3의 OEI는 +0.217이다(E1D §10).

### 5.3 [기술] 반복 수
- **TD-MPC2 E2 반복 곡선**(warm start 켬, D; E2T §5. 반복 1회의 CI는 `e2_tdmpc2_summary.json`의 `iteration_curves`에서 이 보고가 확인)

| 과제 | 변형 | 반복 1 | 3 | 6 | 12 |
|---|---|---|---|---|---|
| mw-push | det | 0.433 [0.333, 0.533] | 0.333 | 0.389 | 0.467 |
| | sh-iter | 0.567 [0.456, 0.678] | 0.144 | 0.011 | 0.000 |
| mw-pick-place | det | 0.478 [0.367, 0.589] | 0.333 | 0.400 | 0.444 |
| | sh-iter | 0.500 [0.389, 0.611] | 0.211 | 0.056 | 0.111 |
| cheetah-run | det | 0.498 [0.451, 0.539] | 0.414 | 0.409 | 0.428 |
| | sh-iter | 0.556 [0.517, 0.591] | 0.284 | 0.154 | 0.046 |

  - 반복 1회면 drift가 정의상 0이다. 이때 sh-iter의 손상은 점추정으로 det와 같거나 크다(CI 겹침).
  - warm start를 꺼도 곡선 모양이 같다. sh-iter의 이득은 한 제어 step 안의 반복에서 생긴다.
  - push의 drift: det 0 → −46.0 → −53.3 → −59.6, sh-iter 0 → −2.2 → −5.2 → −6.1
- **DINO-WM opt_steps**(N 300, top-30; E2D §7.3)
  - D_det +7 / +17 / +24%p(5 / 10 / 30회), D_sh-iter −2 / +3 / 0, 이득 +9 [3, 16] / +14 / +24
  - none 성공률도 48 / 75 / 83%로 바뀐다. none 대비 상대 손상은 15 / 23 / 29%다.
  - det drift −0.212 / −0.229 / −0.269
  - [사후] 짝 대비(opt_steps 5 − 30): D_det −17 [−28, −6], 이득 −15 [−25, −5](E2D §7.4)
- **두 계열 비교**(E2D §7.5): sh-iter 이득과 det drift는 두 계열 모두 반복과 함께 커진다. 경로는 다르다. TD-MPC2는 sh-iter 손상이 줄고, DINO-WM은 det 손상이 늘어난다.
- **E7 실제 kernel**(int4, det − dither; E7T §5)
  - cheetah −0.041 [−0.068, −0.015] / +0.165 / +0.417(반복 3 / 6 / 12)
  - walker −0.049 [−0.083, −0.016] / +0.056 / +0.077
  - mw-push +0.144 / +0.111 / +0.111
  - 반복 3의 DMC에서는 dither가 해롭다.
- **E4**(C6, E4 §5.6): MPPI 모의에서 새 잡음 손상은 반복과 함께 준다(n 6 → 12 비 0.75~0.91). 지속 오차는 평탄하다. 폐루프 toy의 (sh-step형 − sh-iter형) 손상 차는 n = 1에서 0, n = 12에서 +0.22~+0.39다. CEM toy에서는 붕괴 때문에 새 잡음도 평균되지 않았지만, 실제 DINO-WM에서 sh-iter는 모든 opt_steps에서 none 수준이었다.

### 5.4 [기술] 선택 강도: elite, temperature, 후보 수
- **TD-MPC2**(E2T §5): 선택이 강할수록(elite 10, temperature 2) det의 \|OEI\|와 손상이 대체로 커진다(push, cheetah). sh-iter는 모든 설정에서 det보다 손상이 작다(가장 큰 값은 pick-place T2의 0.222).
- **DINO-WM elite 비율**(E2D §7.2): D_det +19 / +24 / +20 / +17%p(top-6 / 30 / 38 / 75)로 단조 관계가 없다. 이득은 네 비율 모두 CI 하한이 0보다 크다. top-6에서는 det OEI의 거의 전부가 drift다(−0.309 / −0.310, sel_indep −0.001).
- **N에 따른 분해**
  - TD-MPC2 det(E2T §4.2): Meta-World에서 sel_last가 커지고(push −24.6 → −46.1, pick-place −20.1 → −30.6) drift는 거의 평탄하다(−52.9 → −49.1, −41.3 → −46.1). cheetah는 두 성분이 모두 커진다.
  - DINO-WM det(E2D §5): 선택 성분은 2.1~2.5배, drift는 1.1배(−0.250 → −0.275)다. 그래서 \|OEI\|는 1.15배만 오른다.
  - E4의 예측(N 의존은 선택 연산자에서, drift는 1차로 평탄; E4 §7.2)과 같은 모양이다.
- **비율 고정 sweep**(TD-MPC2 보조, E2T §4.2): \|OEI_det\|는 오르지만 D_det는 거의 평탄하다(bootstrap 중앙값 ρ 0.3 / 0.0 / 0.6). v5.1 §8.2 수정의 근거(E4 예측)와 맞다.

### 5.5 [기술] 2축 대조
- **TD-MPC2**(E3 seed, 12과제; E3T §4.2). 짝 차의 CI가 0을 넘는 과제 수
  - ind − sh-iter 9/12: 매 forward 새로 뽑을 때의 공유(CRN) 이득
  - sh-step − sh-iter 10/12, sh-ep − sh-iter 11/12: 공유 안에서 지속의 비용
  - sh-ep − sh-step 9/12: 지속이 길수록 해롭다
  - 후보 독립 slot 표(indstep·indep)는 지속해도 drift가 거의 생기지 않는다.
- **사전 예상 "indstep ≈ ind ≈ indep"**(v5 §7.2): drift는 대체로 맞았다. 손상은 3~4/12과제에서 어긋났다(slot 표가 더 해로움: button-press +0.311, peg-insert +0.122, sweep-into +0.167). 정책 prior slot의 비중은 비슷해 원인이 아니다(E3T §4.3).
- **DINO-WM**(E3D §4.2): indplan ≈ ind로 사전 예상과 맞았다. 2 × 2의 다른 대비도 모두 CI가 0을 포함한다. 전체 크기 표에서는 공유도 지속도 손상을 가르지 못한다.
- **E1 sh-ep**: sh-ep가 det보다 해롭다(Meta-World 평균 50.1 대 44.6%p, DMC 57.3 대 50.6%, E1T §9). 파일럿 R2-8b와 같은 방향이다. E4 C9(비일관 축의 고정 표가 det보다 2~15배 해로울 수 있음)가 기전 후보다(E4 §5.9).

### 5.6 [보조·기술] 표 크기 sweep 세부와 var_cand
- **TD-MPC2**(E3T §5)
  - push의 D: N_eff 4 / 128 / 2,048 / 6,144에서 0.767 / 0.411 / 0.422 / 0.278. 감소의 대부분은 가장 거친 표(N_eff 4)와 나머지의 차이다.
  - var_cand는 N_eff를 따라 변하지 않는다: ρ(N_eff, var_cand) −0.40, +0.20, −0.40, +1.00, ρ(var_cand, D) +0.20, −0.40, +0.40, +0.40
  - 층당 offset 하나(shstep_l)의 오차는 대부분 후보 공통이다(common_frac 0.75, push). 그런데도 손상이 가장 크다.
  - 지속의 비용 D(shstep_g) − D(shiter_g)는 12/12에서 CI가 0보다 크고, 표가 거칠수록 크다(E3T §5.3).
- **DINO-WM**(E3D §3)
  - var_cand는 네 sweep 모두 표가 작을수록 커진다(ρ −1.00). ρ(var_cand, D)는 PushT +0.80, PointMaze +1.00이다.
  - L0(전체 크기 = sh-plan)의 drift는 거의 0이다. L3에서는 PushT −0.059(det의 약 1/5), Wall k=3 −10.1이다.
  - [사후] D(L3) − D(L0): PushT +9 [+2, +16], PointMaze +12 [+4, +20]. 이웃 수준 사이 차는 모두 CI가 0을 포함한다(E3D §5).
- **E4**(§2.6, C9): 1/N_eff 법칙은 일관 축에서만 성립하고, 정확식은 측정과 맞는다(비 1.03, R² 0.985). [사후 해석] DINO-WM sweep은 일관 축에 가깝고 TD-MPC2 sweep은 그렇지 않은 쪽으로 보인다.

### 5.7 [탐색] 가중치 양자화 (활성은 fp)
- **TD-MPC2**(sweep 4과제, E5T §7)
  - w8rtn은 손상이 없다(−0.011 ~ +0.022). w4rtn은 0.42~0.93이다.
  - SR 사본 K를 반복마다 돌리면 네 과제 모두 K = 1 > 2 > 4로 손상이 준다.
  - w4rtn − w4sr4: push +0.500, pick-place +0.611, cheetah +0.191(SR이 나음), walker −0.260(RTN이 나음)
  - AWQ식은 Meta-World·walker에서 RTN보다 낫고, cheetah에서 나쁘다.
- **DINO-WM**(PushT k=4, E5D §7)
  - w8rtn +3 [−1, 8]. w4rtn +55, AWQ식 +71, SR K = 1·2·4는 +76·+68·+71%p다.
  - SR이 RTN보다 모두 13~21%p 나쁘고, K에 따라 단조로 줄지 않는다.
- 방향이 계열마다 반대다. 활성 dither의 결과를 가중치 양자화로 일반화하지 않는다.

### 5.8 [기술] det+shift(결정적 반올림 + 반복마다 옮기는 공유 격자)
- E1 평균 하락: Meta-World 4.0%p(sh-iter 4.0), DMC 30.9%(sh-iter 36.5)(E1T §9). DINO-WM red: PushT 0.92 / 0.96 / 1.00, PointMaze 0.94 / 1.27(E1D §9.1)
- 계획 시점 오차는 다른 dither의 약 절반이다(MSE Δ²/12 대 Δ²/6).
- drift 비 r_det+shift: 0.06, 0.15, −0.02(E2T §6.2), PushT +0.003(E1D §8.1)
- E4 예측(같은 비트에서 det+shift ≤ sh-iter)과 맞다. 모의에서 유의하게 나쁜 칸은 0/32였다(E4 §5.10).
- E5(PTQ 위)와 E7(실제 kernel)에서는 재지 않았다. [결정 12]

### 5.9 [기술] 계획 편차와 common_frac
- TD-MPC2 Meta-World 고정 상태 계획 편차(seed 바닥 배수): det 3.83, sh-step·sh-ep 3.70, ind 2.44, sh-iter 2.65, det+shift 2.55(E1T §9). R2-8b와 같은 모양이다.
- DINO-WM에서도 det가 dither보다 크다. 그러나 none 대비 편차가 seed 바닥보다 작아(< 1) 칸을 가르기 어렵다(E1D §10).
- common_frac(Meta-World 평균): det 0.53, 공유 변형 0.35~0.40, ind 0.16(E1T §9)

### 5.10 [기술] near-tie 밀도
- 마지막 반복 후보 집합의 1·2위 gap / 후보 std 중앙값: TD-MPC2 0.040~0.124(첫 반복 0.000~0.056; E1T §8), DINO-WM 0.012~0.057(E2D §6)
- CRN 이득은 TD-MPC2 9/12과제에서 CI가 0보다 크지만, 밀도와 순위가 맞지 않는다(H4b). DINO-WM 3과제는 CI가 0을 포함한다.
- [사후] 밀도와 red의 관계는 계열마다 부호가 반대다(Meta-World −0.81, DMC +0.80; E3T §8).

### 5.11 E4 이론과의 대조

| E4 예측(결과 전) | 실험 | 대조 | 출처 |
|---|---|---|---|
| 같은 평가 sel_last는 새 잡음에서도 음수다 → TD-MPC2 H2a 실패 | E1·E2에서 sh-iter OEI가 유의하게 음수이고, 독립 draw로 재면 사라진다 | 맞음 | E4 §7.1, E1T §15, E2T §6.2 |
| 새 잡음의 drift 기대값은 0이다 | cheetah·PushT sh-iter drift는 0 근처다. Meta-World −5 ~ −11은 대부분 또는 절반이 영역 효과다 | 대체로 맞음 | E2T §6.3 |
| 지속장의 drift는 N에 1차로 무관하다 | Meta-World det drift가 평탄하고, PushT는 1.1배다 | 맞음 | E2T §4.2, E2D §5 |
| 고정 elite 수에서 \|OEI\| 1.7~5.4배, 고정 비율에서는 약하고 손상이 준다 | TD-MPC2 H2c 3/3, 비율 고정에서 D 평탄. PushT \|OEI\|는 1.15배(drift 지배)이고 D는 N 300 뒤 평탄 | TD-MPC2 맞음, PushT 부분 | E4 §5.7, E2T §4.2, E2D §4 |
| 1/N_eff는 일관 축에서만 성립하고, 손상의 직접 예측 변수는 var_cand다 | DINO-WM var_cand 단조, ρ(var_cand, D) 높음. TD-MPC2 var_cand는 N_eff를 따르지 않음 | 부분 | E4 §7.3, E3T §5.2, E3D §3.3 |
| 비일관 축의 고정 표는 det보다 해로울 수 있다 | sh-ep가 det보다 나쁘고, walker의 고정 표는 모두 det보다 해롭다 | 방향 맞음(기술) | E4 §5.9, E1T §9, E3T §5.1 |
| 같은 비트에서 det+shift ≤ sh-iter | Meta-World 같음, DMC 더 작음 | 맞음 | E4 §5.10, E1T §9 |
| MPPI 새 잡음 손상은 반복과 함께 줄고, 반복 1회에서는 지속형과 차가 0이다 | TD-MPC2 E2 반복 곡선 | 맞음 | E4 §5.6, E2T §5 |
| CEM에서는 반복이 새 잡음을 평균하지 못한다(toy) | DINO-WM sh-iter는 opt_steps 5~30 모두 none 수준 | 어긋남(toy의 잡음 척도 차로 보임, 사후 해석) | E4 §6.1, E2D §7.3 |
| CRN 이득은 near-tie에서 커진다(H4b 방향 지지) | H4b 불성립 | 어긋남 | E4 §7.4, E3T §6 |
| 지속 지수는 지속 오차와 새 오차를 가르지 못한다. drift가 낫다 | 실험도 drift를 기전 지표로 썼다 | 권고 반영 | E4 §6.1 |

### 5.12 [탐색] Wall
- Wall 확증 칸 k=4·5·6은 det 하락이 실재하지 않았다(−14, +3, −3%p). k=4에서는 det가 none보다 14%p 낫고, dither 4종은 none 수준으로 돌아간다(E1D §3·§5).
- 탐색 칸 k=3: det +50%p, dither red 0.92~1.08(E1D §9.2). 표 크기 sweep에서는 L3만 +37%p [26, 48]다(E3D §6).

### 5.13 [기술·사후] QAT 세부
- **TD-MPC2**
  - QAT+sh-iter가 QAT+det보다 9/11칸에서 손상이 작다(E5T §6).
  - QAT+none은 11/11칸에서 손상이 크다(PH §4.1).
  - [사후] 세 기준을 모두 만족하는 6칸(door, peg, pick-place, push, window, walker): 양자화 없이도 나쁘고, det에 적응한 흔적이 없고, det 성능도 원래 모델보다 나쁘다(PH §4.2).
- **DINO-WM**(PushT k=4, E5D §6)
  - QAT+det가 E1 det보다 18%p [8, 28] 낫다. QAT 모델 안에서 det가 none보다 21%p [12, 30] 낫다. det 양자화에 맞춘 모델로 보인다.
  - QAT+sh-iter는 QAT+det보다 16%p [7, 25] 나쁘다.
- QAT 모델 위 dither의 효과가 계열마다 반대다.

### 5.14 [사후] 지연 측정의 두 모드
- 첫 측정(E7T §7): block의 17%가 약 +35% 느렸다. 회전별 오버헤드가 −30 ~ +38%로 흔들렸다. 회전 중앙값으로 읽으면 int4 그릇판 cheetah 반복 6이 +30.1%다.
- 재측정(PH §3)
  - block의 19.2%가 약 +33% 느렸다(fp32·bf16도 17.5%). 절반가량이 SM clock 하락과 같이 나타났다.
  - 두 block이 모두 빠른 짝 241개의 오버헤드 중앙값은 −0.04%이고, 92%가 ±5% 안이다.
  - 첫 측정에서 흔들린 칸은 −1.0% [−15.0, +0.8]이다.

---

## 6. 한계

- **DMC**
  - H3a가 불성립했고, walker k=3에서는 dither가 해로웠다(+22.1% [18.0, 26.3]).
  - S_det 설명은 E1 뒤 사전등록이고 상관 결과다. 12과제 ρ 0.83에는 Meta-World와 DMC를 가르는 몫이 섞여 있다. [사후] 계열 안 ρ도 양수지만(Meta-World 0.57, DMC 1.00), DMC는 4과제라 순위의 해상도가 낮다(E3T §8).
  - 최선 PTQ 격자 위에서는 walker k=3의 해로움이 사라졌다(−0.003 [−0.019, +0.011]). quadruped k=3에서는 dither가 조금 해로웠다(−0.012 [−0.018, −0.006]; E5T §5).
- **H3b의 계열 차이**
  - TD-MPC2는 문턱 경계(10/19), DINO-WM은 1/3이다. 최선 PTQ가 손상 대부분을 없애 dither가 더할 여지가 작다.
  - PushT k=4의 최선 기준선은 동률(pc·p999·mse가 held 17/20)이라 목록 순서로 pc를 골랐다. 다른 기준선은 평가하지 않았다(E5D §9).
- **약한 QAT**
  - TD-MPC2의 사전등록 QAT(5,000 step, batch 256, lr 1e-4, 자료 20 episode)는 모델 자체를 손상시켰다(QAT+none 11/11).
  - DINO-WM QAT는 한 칸(PushT k=4)이고, 120분 예산(10,083 step)에서 멈췄다.
  - 어느 계열에서도 QAT에 대해 결론을 내리지 않는다.
- **지연**
  - 측정 환경의 두 모드 때문에, 조건별 95% 상한이 10% 아래임은 보이지 못했다(16/36조건이 넘음).
  - 정수 kernel이 fp32보다 약 18% 느리다. int4 그릇판과 packed int4는 int8과 같은 int8 dot을 써서 지연이 같다.
  - 조건이 좁다: 후보 512개의 작은 MLP 한 종류, A5000 한 장, eager만 쟀다. torchao, CUDA graph, Jetson은 하지 않았다. 지연은 체크포인트 1로만 쟀다.
  - Pareto 사례는 mw-push 하나이고 약하다.
- **Wall과 DINO-WM 과제 수**: Wall 확증 칸에서 det 하락이 실재하지 않았다. 그래서 DINO-WM의 확증 결과(H1·H3a 자격 5칸, H4a 자격 2과제, H3b 3칸)는 사실상 PushT와 PointMaze 두 과제에 기댄다. 기전 검정(H2)의 DINO-WM 몫은 PushT 한 과제다.
- **경계 판정**: H4a(3/5과제가 정확히 −0.8, 허용 오차가 없었다면 불성립), H3b TD-MPC2(정확히 10/19), H2c PushT(0.771), H3a DINO-WM(80% 대 문턱 75%, 해 점검 하한 +4 대 +5%p), H2a PushT(sh-iter CI 하한 −8×10⁻⁶). 한 칸이나 한 순위로 판정이 바뀐다.
- **적은 점 수**: H2·H4 판정은 과제 4개, 점 4~6개의 Spearman이다. 4점 Spearman은 0.2 간격의 값만 갖고, 6점은 0.829 다음이 0.771이다.
- **모델과 과제의 범위**: TD-MPC2 5M 단일 과제 체크포인트와 공식 DINO-WM 세 모델뿐이다. 큰 월드모델(V-JEPA 2-AC, E6)과 다중 과제 체크포인트는 하지 않았다. 모두 시뮬레이션이고 실로봇이 없다.
- **fake-quant와 실제 kernel**: E1~E5는 fake-quant다. 실제 정수 kernel(E7)은 TD-MPC2 4과제에서만 쟀고, DINO-WM에는 실제 kernel 결과가 없다.
- **양자화 범위**: 주 결과는 활성 양자화다. 가중치 양자화는 탐색이고, SR의 방향이 계열마다 반대다.
- **기전 측정의 해석**
  - drift에는 영역 효과가 섞인다(PushT 약 2/3, TD-MPC2는 반대 부호). 영역 효과 대조(b3)는 기술이고 판정이 아니다.
  - Meta-World sh-iter에 남는 drift(pick-place −5.13)의 원인은 가르지 않았다.
  - DINO-WM sh-plan 양성 대조가 실패했다.
- **사전등록의 성격**
  - 내부 사전등록이다. 단계마다 시각과 sha256을 LOG에 남긴 설계 문서이고, 외부 등록이 아니다.
  - E1 결과 뒤 v5·v5.2로 더한 것이 있다: H2a′, H2a_indep, DMC 설명 규칙, E3 sweep에 walker를 넣은 것(E1에서 해로웠던 칸, DM §7.2가 밝힘).
  - 판정 계산의 허용 오차 수정은 결과 전이었지만 H4a 판정을 바꾼다.
  - E7 작업자는 코드 대조를 마치기 전에 지연 요약을 보았다(판정 영향 없음, 보고에서 공개; E7T 머리).
- **H4b 입력의 일부가 정의 고정 전에 공개돼 있었다**: near-tie 밀도 값은 v5가 H4b 정의를 고정하기 전에 E1 보고(10-22 07:40Z)에 실렸다. CRN 이득은 E3 seed로 처음 계산했다. 판정이 불성립이라 결론에는 영향이 없다.
- **seed 분할이 단계마다 다르다**(E1 20000~, E2 40000~, E3 50000~, E5 60000~, E7 70000~). 같은 칸의 값이 단계마다 조금 다르지만, 결론은 같은 방향으로 되풀이됐다(예: Meta-World red는 E1 0.53~1.09, E3 0.75~1.22).

---

## 7. 결정이 필요한 주장 문구

- 항목마다 **정할 것 → 근거(수치와 출처) → 추천안(논문 문장 초안, 한국어와 영문) → 선택지와 장단점 → JKAIA 원고의 현재 문구** 순서로 적었다.
- 1~9는 조정자가 지정한 항목이고, 10~13은 근거를 읽고 이 보고가 더한 항목이다.
- JKAIA 원고(`A_dithered_world_models/paper/jkaia2026/main.tex`)와 그 README는 10-24에 사용자가 "추천 1~5"를 승인했다고 적었다. 그 승인이 최종 논문(TMLR 등)에도 그대로 적용되는지 확인이 필요하다([불일치 D2](#10-불일치-목록)).
- 어느 항목도 판정을 바꾸지 않는다. 문장의 범위만 정한다.

### [결정 1] 기전 문장

- **정할 것**: v5.2의 기전 문장 "계획기가 반복에 걸쳐 지속되는 오차를 이용한다"를 그대로 쓸지, 한정을 붙일지, H2c를 이을지. 선택지 A/B/C는 E2T §8의 것이다.
- **근거**
  - 받치는 것
    - H2a′ 4/4: det drift < 0, r_sh-iter ≤ 0.24(3절)
    - TD-MPC2 양성 대조: r_sh-step·sh-ep 0.70~1.30
    - H2c 3/4(elite 수 고정)
    - E3 지속 비용 12/12(E3T §5.3)
    - 반복 곡선: det drift가 반복과 함께 커지고(두 계열), sh-iter 손상은 반복과 함께 사라진다(TD-MPC2)
    - det+shift가 sh-iter와 같은 효과(5.8절)
  - 좁히는 것
    1. **영역 효과**: PushT k=4 det drift의 약 2/3가 영역 효과다(drift_ref −0.177 / drift −0.269). det 자신의 몫은 −0.092다(E1D §8.2). TD-MPC2에서는 drift_ref가 det drift와 반대 부호라(+9.3, +3.9, +0.88) 자신의 몫이 drift보다 크다(E2T §6.3). 자신의 몫은 잰 모든 칸에서 0 이하다(기술).
    2. **지속만으로는 부족하다**: DINO-WM sh-plan 양성 대조가 r +0.011로 예상(> 0.5)과 달랐다. E3에서 전체 크기 표(L0)의 drift는 거의 0이고, 거친 표 L3에서는 PushT −0.059, Wall k=3 −10.1이다(E3D §3.4). 이용되는 것은 후보 사이에서 충분히 거칠게 변하는 지속 오차다. TD-MPC2에서는 전체 크기 계획 고정 표도 이용됐다(r_sh-step 0.70~1.30).
    3. **Meta-World의 sh-iter에도 작은 drift가 남는다**(−5.2, −11.2). 일부는 영역 효과이고, pick-place에서는 −5.13이 남는다. 그래서 H2a_indep가 2/4로 불성립이다.
    4. **H2c**: Meta-World의 N 의존은 주로 sel_last(같은 결정적 오차장 안의 선택)에서 온다. PushT는 손상 조건에서 경계 불성립이다(0.771).
    5. **"고른 계획을 과대평가한다"는 모든 칸에서 맞지 않다**: PushT k=3에서는 det drift가 +0.243, OEI가 +0.217이다(영역 효과 +0.448 때문, E1D §10).
- **추천안**: **(C) + (B)의 한정을 본문에 둔다.** 제목과 초록에는 v5.2 문장과 H2c를 쓰고, 기전 절 본문에 영역 효과와 "지속만으로는 부족하다"를 쓴다.
  - 초록 문장
    - "결정적 반올림은 최적화 반복 내내 남는 오차장을 만들고, 반복 계획기는 모집단을 그 오차장이 유리하게 보는 후보 쪽으로 옮긴다. 같은 비트 수에서 반복마다 새로 뽑는 공유 dither는 이 이동을 없앤다."
    - (영문) "Deterministic rounding leaves an error field that persists across optimizer iterations, and an iterating planner moves its population toward the candidates this field favors; shared dither redrawn at every iteration removes this drift at the same bit-width."
  - 본문 근거 문장
    - "기전 과제 4개 모두에서 det의 이동(drift)은 음수였고, sh-iter의 이동은 det의 24% 이하였다(E1 뒤 사전등록). 같은 dither를 계획 동안 고정하면 TD-MPC2에서 이동의 70~130%가 돌아왔다. elite 수를 고정하면 후보 수가 늘수록 det의 이용과 손상이 함께 커졌다(4과제 중 3과제)."
    - (영문) "Under det the population drifted toward favorable error in all four mechanism tasks, while sh-iter kept at most 24% of that drift (registered after stage 1). Freezing the same dither for a whole plan brought back 70–130% of the drift on TD-MPC2, and with a fixed number of elites both exploitation and damage grew with the candidate count (3 of 4 tasks)."
  - 본문 한정 문장
    - "이 이동에는 계획기가 가는 영역 자체의 오차도 섞인다. PushT에서는 det 이동의 약 2/3가 비양자화 계획기도 겪는 영역 효과였고, TD-MPC2에서는 영역 효과의 부호가 반대였다. 영역 효과를 빼도 det 계획기 자신의 몫은 잰 모든 칸에서 0 이하였다. 또 지속만으로는 충분하지 않다. DINO-WM에서 크기가 큰 고정 dither 표는 이용되지 않았고, 표를 거칠게 할수록 이동과 손상이 함께 커졌다."
    - (영문) "Part of this drift is a region effect that an unquantized planner visiting the same candidates also sees (about two thirds on PushT; opposite in sign on TD-MPC2), yet the planner's own share is non-positive in every cell we measured. Persistence alone is not sufficient: full-size fixed dither tables were not exploited on DINO-WM, whereas coarser tables were, with damage growing alongside the drift."
- **선택지**
  - (A) v5.2 문구 그대로
    - 장점: 승인된 문구이고 짧다. 사전등록 검정 H2a′(4/4)와 정확히 맞는다.
    - 단점: 영역 효과, sh-plan 실패, Meta-World sh-iter의 잔여 drift를 숨긴다. 리뷰에서 반례(PushT k=3, sh-plan)로 반박당하기 쉽다.
  - (B) (A)에 한정을 첫 문장부터 붙인다
    - 장점: 가장 정직하다.
    - 단점: 핵심 메시지가 흐려진다. 한정의 근거(영역 효과 대조)는 기술 결과라 판정 근거가 아니다.
  - (C) (A)에 H2c를 잇는다(JKAIA 원고 방식)
    - 장점: 용량-반응이 optimizer's curse 해석을 강하게 한다.
    - 단점: PushT가 경계 불성립이다. Meta-World의 N 의존은 주로 sel_last라 "이동" 기전과 같은 성분이 아니다(det의 sel_last도 결정적 오차의 이용이라 모순은 아니다).
- **JKAIA 원고**: (C) 방식이다. 초록은 "an iterating planner drifts toward the candidates this field happens to favor and overrates the plan it selects"다.
  - "overrates the plan it selects"는 PushT k=3처럼 det drift가 양수인 칸에서는 맞지 않는다. 상대적 표현("모집단을 유리한 쪽으로 옮긴다")으로 바꾸기를 권한다.
  - 영역 효과와 sh-plan 한정은 원고에 없다(2쪽 분량 때문으로 보인다).

### [결정 2] 처방 범위: "반복하는 계획기"

- **정할 것**: 처방 주장에 계획기 반복 조건을 붙일지, 붙인다면 어떻게 붙일지.
- **근거**
  - TD-MPC2 E2(5.3절 표): 반복 1회에서 sh-iter 0.567 / 0.500 / 0.556 대 det 0.433 / 0.478 / 0.498(CI 겹침). 반복 12회에서 0.000 / 0.111 / 0.046 대 0.467 / 0.444 / 0.428. warm start를 꺼도 같다(E2T §5).
  - E7 실제 kernel 반복 3회: DMC에서 dither가 해롭다(cheetah −0.041 [−0.068, −0.015], walker −0.049 [−0.083, −0.016]). mw-push는 반복 3회에서도 이득이다(+0.144; E7T §5).
  - DINO-WM: 잰 가장 적은 반복(5회)에서도 이득이 +9%p [3, 16]이다. 1~4회는 재지 않았다(E2D §7.3).
  - E1 기본 설정(MPPI 6회, CEM 30·10회)이 H3a의 근거다.
- **추천안**: 제목·초록에는 "반복하는 계획기"로 쓰고, 본문에 잰 범위를 밝힌다.
  - "이 처방은 계획기가 여러 번 반복한다는 것을 전제한다. MPPI 반복이 1회면 sh-iter도 det만큼 해로웠고(세 과제의 손상 0.50~0.57 대 0.43~0.50), 반복 12회에서는 sh-iter의 손상이 0.00~0.11로 줄었지만 det는 0.43~0.47에 머물렀다. 실제 정수 kernel에서 MPPI 반복이 3회면 DMC 두 과제에서 dither가 det보다 조금 해로웠다. CEM은 잰 가장 적은 반복(5회)에서도 이득이 남았다."
  - (영문) "The fix assumes an iterating planner. With a single MPPI iteration, sh-iter was as harmful as det (damage 0.50–0.57 vs. 0.43–0.50 on three tasks); at 12 iterations its damage fell to 0.00–0.11 while det stayed at 0.43–0.47. With three MPPI iterations in the integer kernel, dither was slightly worse than det on both DMC tasks, whereas CEM kept a gain at the fewest iterations we tried (five)."
- **선택지**
  - (a) 조건 없이 "같은 비트의 처방"으로 쓴다
    - 장점: 단순하다.
    - 단점: 반복 1회 결과와 E7 반복 3회 DMC 결과가 반례가 된다.
  - (b) 수치 문턱을 주장에 넣는다(예: "MPPI 반복 6회 이상")
    - 장점: 정확하다.
    - 단점: 잰 과제가 적어(E2 3과제, E7 4과제) 문턱을 일반화하기 어렵다. DINO-WM은 1~4회를 재지 않았다.
- **JKAIA 원고**: "It needs an iterating planner"(초록). 추천과 같은 방향이다. 수치 범위는 그림 1(b)로 보인다.

### [결정 3] 처방 주장의 과제 범위와 DMC 서술

- **정할 것**: v5.2 §9.2가 정한 처리(처방은 Meta-World·DINO-WM, DMC는 S_det로 서술)를 어떤 문장으로, 어느 절에 쓸지.
- **근거**
  - DMC H3a 5/12, walker k=3 +22.1% [18.0, 26.3](E1T §5)
  - DMC 설명 "설명됨", ρ 0.83(E1 뒤 사전등록, 새 seed, 12과제 모두 포함; E3T §7)
  - S_det: DMC 0.33~0.52, Meta-World 0.54~0.73(이 보고가 `e3_tdmpc2_final.json`에서 확인)
  - [사후] 계열 안 ρ: Meta-World 0.57, DMC 1.00(E3T §8)
  - 최선 PTQ 격자 위에서 walker k=3의 해로움은 사라졌다(E5T §5).
  - 반복 6·12회에서는 DMC에서도 dither 이득이 있다(E7T §5).
- **추천안**: 처방 주장은 Meta-World·DINO-WM에 대해 한다. DMC는 본문의 "처방이 약한 곳" 문단으로 쓴다. H3a 표에는 DMC 불성립을 그대로 둔다.
  - "처방 주장은 Meta-World와 DINO-WM에 대해 한다. DMC에서는 sh-iter가 12칸 중 5칸에서만 손상의 절반 이상을 없앴고, walker 3비트에서는 det보다 해로웠다(상대 return 손실 78.9% 대 56.9%). det의 유리한 오차 가운데 반복 사이에 지속되는 몫은 DMC에서 0.33~0.52로 Meta-World(0.54~0.73)보다 작다. E1 뒤에 사전등록하고 새 seed로 잰 TD-MPC2 12과제에서, 이 몫이 dither의 손상 제거율을 예측했다(Spearman 0.83)."
  - (영문) "We make the prescriptive claim for Meta-World and DINO-WM. On DMC, sh-iter removed at least half of the damage in only 5 of 12 cells and hurt on walker at 3 bits (78.9% vs. 56.9% of return lost). The share of det's favorable error that persists across iterations is 0.33–0.52 on DMC versus 0.54–0.73 on Meta-World, and a test registered after stage 1 and run on new seeds finds that this share predicts dither's damage reduction across the 12 TD-MPC2 tasks (Spearman 0.83)."
  - 부록에 [사후] 계열 안 ρ를 사후로 적는다. 12과제 ρ에 계열 차이가 섞인다는 점을 밝히기 위해서다.
- **선택지**
  - (a) v5.2 §9.2 문자 그대로 "한계" 절에만 둔다
    - 장점: 가장 보수적이다.
    - 단점: 기전을 지지하는 사전등록 검정(DMC 설명)을 본문에서 버린다.
  - (b) DMC를 처방 범위에 넣고 "효과가 작다"로 쓴다
    - 장점: 과제 범위가 넓어 보인다.
    - 단점: v5.2 §9.2와 H3a 불성립(해로운 칸)에 어긋난다. 권하지 않는다.
- **JKAIA 원고**: 추천과 같다. paper/README는 10-24 사용자 승인으로 DMC를 "한계"에서 본문 "Where the fix is weak" 문단으로 옮겼다고 적었다.

### [결정 4] H3b 계열별 결과의 서술

- **정할 것**: "강한 PTQ 위의 추가 이득"을 어떻게 쓸지.
- **근거**
  - TD-MPC2 10/19(Meta-World 6/12, DMC 4/7)로 문턱과 같다. 하한이 정확히 0이라 불통과인 칸이 둘(pick-place k=3, drawer-open k=3), +0.0008로 통과한 칸이 하나(quadruped k=4)다(E5T §5).
  - 최선 B는 per-channel 범위가 17/19칸이다. Meta-World D(B+det)는 0.000~0.167이다.
  - DINO-WM 1/3: PointMaze k=3만 +12 [4, 20]으로 통과했다. PushT는 최선 PTQ만으로 D(B+det)가 +3·+4%p다(E1 det 대비 −33·−20%p; E5D §5).
- **추천안**
  - "강한 PTQ 기준선은 그것만으로 det 손상 대부분을 없앴다. 칸마다 고른 최선 기준선 위에 dither를 더하면 TD-MPC2에서는 19칸 중 10칸(사전등록 문턱과 같은 수), DINO-WM에서는 3칸 중 1칸(PointMaze)에서 추가 이득이 있었다. DINO-WM PushT에서는 기준선만으로 남은 손상이 3~4%p였다."
  - (영문) "Strong PTQ baselines removed most of the det damage on their own. Adding dither on top of the best baseline per cell still helped in 10 of 19 TD-MPC2 cells (exactly the pre-registered threshold) but in only 1 of 3 DINO-WM cells; on PushT the baseline alone left 3–4 points of damage."
  - 덧붙일 수 있는 점: dither는 PTQ 격자 선택과 직교해 최선 격자 위에 그대로 얹을 수 있다(E5가 그렇게 쟀다).
- **선택지**
  - (a) "강한 PTQ 위에서도 이득이 남는다"로 TD-MPC2 성립을 앞세운다
    - 장점: 처방의 실용 가치가 커 보인다.
    - 단점: DINO-WM 불성립과 TD-MPC2의 경계를 가린다.
  - (b) H3b를 부정 결과로만 보고하고, dither를 "같은 격자에서 RTN의 대안"으로만 둔다
    - 장점: 가장 보수적이다.
    - 단점: TD-MPC2 10/19(DMC 4/7 포함)의 통과를 버린다.
- **JKAIA 원고**: 추천과 같은 방향이다("Other limitations": 10/19 대 1/3, PushT는 기준선만으로 손상 대부분이 사라짐).

### [결정 5] RQ5·H5 범위

- **정할 것**: 배치 주장을 "같은 비트에서 dither 추가 지연은 무시할 수준"으로 한정할지, 사후 재측정을 어떻게 쓸지, 낮은 비트 배치 이득이 없다는 것을 어떻게 쓸지.
- **근거**
  - H5 성립: 12조건 모두 오버헤드 ≤ 10%(최대 int8 +3.8%, int4 그릇판 +5.8%, packed +6.9%; E7T §4)
  - Pareto: mw-push 사례뿐이다. 같은 반복 수 사례는 성공이 같고 지연 차 3~5%가 회전마다 유지되지 않는다. 다른 반복 수 사례는 지연 이득이 반복 수에서 오고, 성공 차 CI가 0을 포함하거나 하한이 0이다(E7T §4.2).
  - 정수 kernel은 fp32보다 17.5~18.2% 느리고, int4와 int8의 지연 차가 없다(E7T §6).
  - [사후] 짝 재측정: 중앙값 −3.2~+2.7%, CI는 35/36이 0을 포함하고, 상한 > 10%가 16/36이다. 이것은 dither가 아니라 측정 환경의 두 모드 때문이다(fp32·bf16도 느린 block을 가짐; PH §3).
- **추천안**
  - "같은 비트의 융합 정수 kernel에서 dither의 추가 지연은 측정 잡음 수준이었다(사전등록 계산으로 12조건 모두 10% 이하, 최대 6.9%; 사후 짝 재측정의 조건별 중앙값 −3.2~+2.7%). 다만 이 구현의 정수 kernel은 fp32보다 약 18% 느려, 낮은 비트로 지연을 줄이는 배치 이득은 얻지 못했다."
  - (영문) "In a fused integer kernel, dither added latency within measurement noise at the same bit-width (pre-registered computation: at most 10% in all 12 conditions, maximum 6.9%; a post-hoc paired re-measurement found per-condition medians of −3.2% to +2.7%). Our integer kernels were, however, about 18% slower than fp32, so lower bit-widths gave no latency benefit."
  - H5 표에는 "성립(Pareto 사례는 약함)"으로 적고, 본문에서는 Pareto 지배를 주장하지 않는다. 두 모드와 CI는 부록에 둔다.
- **선택지**
  - (a) "추가 지연 10% 미만"만 쓴다(JKAIA 원고 방식)
    - 장점: 짧고 사전 정의 판정과 같다.
    - 단점: "미만"이 조건별 상한으로 읽힐 수 있는데, 재측정 CI 상한은 16/36에서 10%를 넘는다. 정수 kernel이 느리다는 점을 같은 문단에 쓰지 않으면 배치 이득으로 오해될 수 있다.
  - (b) RQ5를 한계로만 보고한다
    - 장점: 가장 보수적이다.
    - 단점: "dither는 지연 비용이 거의 없다"는 쓸모 있는 결과를 버린다.
- **JKAIA 원고**: 초록 "at less than 10% extra latency", 본문 "at most 6.9% ... integer kernels were about 18% slower than fp32, so low bits gave no speedup". 본문은 추천과 같다. 초록에 "median"이나 "pre-registered"를 넣으면 더 정확하다.

### [결정 6] QAT 단서 문구

- **정할 것**: 두 계열의 QAT 결과를 어떤 문구로 쓸지.
- **근거**
  - TD-MPC2
    - 규칙 조건 0/11
    - QAT+det가 11칸 모두 B+det보다 나쁘다(CI 겹치지 않음). 7/11칸에서는 E1의 일반 det보다도 나쁘다(E5T §6).
    - [사후] QAT+none 11/11 손상(D 0.20~0.82)
    - [사후] 세 기준을 모두 만족하는 6칸(door, peg, pick-place, push, window, walker)에서는 "학습이 모델 자체를 손상했다"는 서술이 맞는다(PH §4.2).
  - DINO-WM(PushT k=4)
    - QAT+det +6 [−1, 13] ≈ B+det +4(짝 차 +2 [−5, +10])
    - QAT 모델 안에서 det가 none보다 21%p 낫다. det에 맞춘 모델로 보인다.
    - QAT+sh-iter는 QAT+det보다 16%p 나쁘다(E5D §6).
  - 조정자 판단(LOG 10-24 12:29Z): DINO-WM에는 같은 단서를 달지 않는다.
- **추천안**
  - "QAT에 대해서는 결론을 내리지 않는다. TD-MPC2에서 사전등록한 짧은 QAT(5,000 step, 자료 20 episode)는 양자화 없이 써도 11칸 모두에서 원래 모델보다 나빠(사후 점검) 약한 기준선이었다. DINO-WM PushT에서는 QAT+det가 최선 PTQ와 구별되지 않았다(+6 대 +4%p). 그래서 dither는 학습 없는 처방으로 제시한다."
  - (영문) "We draw no conclusion about QAT. The short pre-registered QAT for TD-MPC2 (5,000 steps on 20 episodes) degraded the model even without quantization in all 11 cells (post-hoc check), making it a weak baseline, and on DINO-WM PushT QAT with det was indistinguishable from the best PTQ (+6 vs. +4 points). We therefore present dither as a training-free fix."
  - 부록: QAT 모델 위 dither의 효과는 계열마다 반대였다(TD-MPC2 9/11칸 이득, DINO-WM −16%p).
- **선택지**
  - (a) 두 계열 모두에 "약한 기준선" 단서를 단다
    - 장점: 단순하다.
    - 단점: DINO-WM 자료(QAT+det ≈ B+det, det 적응)와 맞지 않고 조정자 판단과 다르다.
  - (b) QAT를 본문에서 빼고 부록에만 둔다
    - 장점: 본문이 짧아진다.
    - 단점: 리뷰에서 QAT 비교 질문이 반드시 나온다. v5.3 §10.1이 보고 방식을 정했다.
- **JKAIA 원고**: QAT를 표에 싣지 않았고, 처방을 "training-free"로 썼다. 추천과 충돌하지 않는다.

### [결정 7] 경계 판정의 표시 방식

- **정할 것**: 문턱에 걸린 판정을 표와 본문에서 어떻게 표시할지.
- **근거**(3절)

| 판정 | 경계의 내용 | 보조 정보 |
|---|---|---|
| H4a 성립 | 성립 5과제 중 3과제(push, pick-place, PushT)가 정확히 −0.8(4점에서 이웃 자리바꿈 한 번, 규칙이 허용). 허용 오차가 없었다면 두 계열 모두 불성립 | bootstrap에서 ρ ≤ −0.8 비율: push 0.98, pick-place 0.83, PushT 0.70(E3T §5.1, E3D §3.1) |
| H3b TD-MPC2 성립 | 정확히 10/19. 하한이 정확히 0인 두 칸은 불통과, +0.0008인 한 칸은 통과 | — (E5T §5) |
| H2c PushT 불성립 | ρ(N, D) 0.771(6점에서 0.8 바로 아래 값). D가 N ≥ 300에서 평탄 | 결합 bootstrap P(ρ ≥ 0.8) = 0.55(E2D §4) |
| H3a DINO-WM 성립 | 4/5 = 80%(문턱 75%). 해 점검 하한 최대 +4%p(문턱 +5%p) | 자격 칸이 하나 더 있고 실패했다면 67%(E1D §0) |
| H2a PushT 성립 | sh-iter CI 하한 −8×10⁻⁶ | 계열 판정(1/4)에는 영향 없음(E1D §6) |

- **추천안**: 사전등록 표에 "성립†" / "불성립†"을 붙이고, 각주에 여유 폭을 적는다. 본문에 "판정 규칙과 경계" 한 단락을 두고, bootstrap 보조 값은 부록에 둔다.
  - 각주: "† 경계: 사전등록 규칙의 문턱에 정확히 걸렸다. H4a는 성립 5과제 중 3과제가 ρ = −0.8이다(4점에서 이웃 자리바꿈 한 번으로, 규칙이 명시적으로 허용한다). 부동소수점 비교의 허용 오차(1e-9)는 두 계열 모두 결과를 보기 전에 넣었다. H3b(TD-MPC2)는 19칸 중 정확히 10칸이고, CI 하한이 정확히 0인 두 칸은 불통과로 셌다. H2c(PushT)는 ρ(N, D) = 0.771로, 6점에서 0.8 바로 아래 값이다."
  - (영문) "† Borderline: the frozen threshold was met exactly. For H4a, three of the five passing tasks have ρ = −0.8, a single neighbor swap among four points that the rule explicitly allowed; a 1e-9 tolerance for floating-point comparison was added in both families before the results were read. H3b (TD-MPC2) passed in exactly 10 of 19 cells, counting two cells with a CI lower bound of exactly zero as failures. H2c (PushT) has ρ(N, D) = 0.771, the value just below 0.8 for six points."
- **선택지**
  - (a) 표에만 짧게 표시한다
    - 장점: 간결하다.
    - 단점: 독자가 놓치고, 허용 오차 수정이 H4a를 바꾼 사실이 드러나지 않는다.
  - (b) 경계 판정을 "판정 불가"로 낮춘다
    - 장점: 보수적으로 보인다.
    - 단점: 동결한 규칙을 결과 뒤에 바꾸는 것이다. 권하지 않는다.
- **JKAIA 원고**: 표에 "5/6 tasks (three exactly −0.8)"만 있다. H3b와 H2c의 경계는 표시되지 않았다.

### [결정 8] 부정 결과의 서술

- **정할 것**: H2a·H2b·H4b와, OEI 진단 주장을 뺀 이유를 어디에 어떻게 쓸지.
- **근거**
  - H2a: 동결 1/4, 독립 draw 2/4. TD-MPC2 OEI 정의가 같은 평가 안의 선택(winner's curse)을 포함해, 새로 뽑는 잡음에서도 음수가 된다(E4 §7.1이 예측). 독립 draw로 다시 재도 Meta-World에서 작은 drift가 남는다.
  - H2b: 0.079와 −0.144. 과제 안 18점이 비트 수 셋에 걸쳐 있어 1·3-step 오차도 손상과 순위가 맞는다(E1T §10.2). DINO-WM에서는 OEI 부호에 영역 효과가 섞인다.
  - H4b: +0.03. CRN 이득은 있지만(9/12) 밀도를 따르지 않는다.
  - v5.2(10-22 09:50Z, 사용자 승인)가 E2 자료 전에 진단 주장을 뺐다.
- **추천안**: 본문에 "사전등록했지만 성립하지 않은 것" 단락을 두고, 상세는 부록에 둔다.
  - "사전등록한 OEI 가설 둘은 성립하지 않았다. H2a(확률적 변형의 OEI가 0 근처)는 TD-MPC2의 OEI 정의가 같은 평가 안의 선택(winner's curse)을 포함해, 새로 뽑는 잡음에서도 음수가 되었기 때문에 실패했다. 이 실패는 이론이 미리 예측했다. 독립 draw로 다시 재도 Meta-World에서 작은 이동이 남아 4과제 중 2과제였다. H2b(OEI가 1·3-step 오차보다 손상을 잘 예측)는 두 계열 모두 불성립이었다. 그래서 진단 주장은 E2 자료 전에 뺐고, 기전 검정은 이동(drift) 기반 H2a′로 좁혀 E1 뒤에 사전등록했다. near-tie 밀도가 공유 이득을 예측한다는 H4b도 불성립이었다."
  - (영문) "Two pre-registered OEI hypotheses failed. H2a (OEI near zero for stochastic variants) failed because the TD-MPC2 OEI, as defined, includes selection within the final evaluation, a winner's curse that is negative even for freshly drawn noise, as our theory predicted; re-measured with an independent draw, a small drift remained on Meta-World and the test passed in only 2 of 4 tasks. H2b (OEI predicts damage better than one- and three-step errors) failed in both families. We therefore dropped the diagnostic claim before the stage-2 data and narrowed the mechanism test to the drift-based H2a′, registered after stage 1. H4b (near-tie density predicts the gain from sharing) also failed."
- **선택지**
  - (a) 부록에만 둔다
    - 장점: 본문이 짧다.
    - 단점: 설계 6.8절("판정 실패 칸도 모두 보고")에 어긋나고, 리뷰어가 숨김으로 읽을 수 있다.
  - (b) OEI 분해(sel_last + drift)를 "진단 도구" 기여로 앞세운다
    - 장점: 측정 기여가 산다.
    - 단점: 분해는 E1 사후에서 나왔고, 손상 예측력은 없다(−drift로 바꿔도 −0.265). "측정 도구"로만 쓴다.
- **JKAIA 원고**: 표 마지막 줄 "H2a, H2b; H4b — fail (claims narrowed)". "새로운 점 (1)"에서 OEI를 "이용의 직접 측정"으로 쓰고 손상 예측력은 주장하지 않는다. 추천과 같다.

### [결정 9] 논문 제목과 한 줄 주장

- **정할 것**: 좁혀진 범위에 맞는 제목과 한 줄 주장
- **근거**
  - 제목의 주장 "Planners Exploit Deterministic Error"는 H2a′(4/4)와 H1이 받친다.
  - 좁혀진 곳은 반복하는 계획기, Meta-World·DINO-WM, 같은 비트 수다.
  - README의 A 한 줄 주장("반복 계획기가 그것을 이용한다. 같은 비트 수에서 dither로 바꾸면 손상 대부분이 사라진다")에는 과제 범위가 없다.
- **추천안**: 제목은 그대로 둔다(JKAIA와 같음). 한 줄 주장을 좁힌 범위로 바꾼다.
  - "같은 비트 수에서 결정적 반올림은 반복 계획기가 이용하는 지속 오차장을 만든다. 반복마다 새로 뽑는 공유 dither로 바꾸면 Meta-World와 DINO-WM에서 그 손상 대부분이 학습 없이 사라진다."
  - (영문) "At the same bit-width, deterministic rounding creates a persistent error field that iterating planners exploit; replacing it with shared dither redrawn at every iteration removes most of the damage on Meta-World and DINO-WM without training."
- **선택지**
  - (b) *Iterating Planners Exploit Persistent Quantization Error: Dithered Low-Bit World Models*
    - 장점: 반복과 지속이라는 범위가 제목에 들어간다.
    - 단점: JKAIA 포스터 제목과 달라진다(비공식 출판이라 문제는 없다). 길다.
  - (c) *Same Bits, Different Plans: Error Structure Decides Planning with Quantized World Models*(설계 후보)
    - 장점: 가장 강한 결과(H1·H3a)를 앞세운다.
    - 단점: 기전 메시지가 빠진다. "error structure"가 불성립한 H4b와 2축 계열 차이를 떠올리게 한다.

### [결정 10] 진단(H1) 문구와 k-step 오차 (이 보고가 더함)

- **정할 것**: v5.2가 남긴 진단 기여("H1과 계획 편차")를 어떤 범위로 쓸지
- **근거**
  - H1이 두 계열에서 성립했다(29/36).
  - 3·5-step 오차는 det를 가른다(5-step: PushT k=4 det 0.250 대 sh-iter 0.042, Wall k=3 det 3.45 대 sh-iter 0.15; E1D §4·§9.2).
  - 비트 수를 섞으면 1·3-step 오차도 손상과 순위가 맞는다(E1T §7).
  - 계획 편차: TD-MPC2에서는 det가 크다(3.83 대 sh-iter 2.65). DINO-WM에서는 none 대비 편차가 seed 바닥보다 작아 칸을 가르기 어렵다(E1D §10).
  - 설계 2.3절은 "k-step 오차는 틀린 지표다"를 쓰지 않을 주장으로 정했다.
- **추천안**
  - "같은 비트 수에서 1-step 예측 오차는 반올림 방식 사이의 손상 차이를 보지 못한다. 손상 칸 36개 중 29개에서 det의 1-step 오차가 sh-iter 이하였지만, 손상은 det에서만 컸다. 오차가 horizon을 따라 쌓이는지(3·5-step 오차)는 det를 가른다."
  - (영문) "At a fixed bit-width, one-step prediction error does not see the damage difference between rounding schemes: in 29 of 36 affected cells det's one-step error was no larger than sh-iter's, yet only det caused large damage. Whether the error accumulates over the horizon (three- and five-step error) does separate det."
- **선택지**
  - (a) "예측 오차는 계획 손상을 예측하지 못한다"(일반 명제)
    - 장점: 메시지가 강하다.
    - 단점: 비트 수를 섞으면 맞지 않고, 설계 2.3절이 금한 문장이다.
  - (b) 계획 편차를 진단 지표로 함께 제안한다
    - 장점: v5.2가 진단 기여로 적은 항목이다.
    - 단점: DINO-WM에서는 칸을 가르지 못한다. TD-MPC2 보조 지표로만 쓴다.
- **JKAIA 원고**: "its one-step error was no larger than dither's in 29 of 36 affected cells". 같은 방향이다. multi-step 문장은 없다.

### [결정 11] 2축 대조와 표 크기 결과의 서술: 공유·새로움의 계열 차이 (이 보고가 더함)

- **정할 것**: 설계 7장 위험 표의 대응("공유·새로움 결과가 모델마다 다르면 표 크기 법칙으로 통합한다. 법칙이 맞지 않으면 '모델 의존적'으로 정직하게 보고한다") 가운데 어느 쪽으로 쓸지
- **근거**
  - TD-MPC2: 공유 이득 9/12, 지속 비용 10/12·11/12(E3T §4.2)
  - DINO-WM: 전체 크기 표의 2 × 2 대비가 모두 CI에 0을 포함한다(E3D §4.2).
  - H4a 두 계열 성립(경계)
  - DINO-WM var_cand는 단조이고 ρ(var_cand, D) 0.80·1.00이다. TD-MPC2 var_cand는 평탄하다. walker는 +0.40이다.
  - E4: 1/N_eff 법칙은 일관 축에서만 성립한다.
- **추천안**: 정성적으로 통합하되, 모델 의존성과 정량 법칙이 없음을 밝힌다.
  - "공유와 새로움이 얼마나 필요한지는 고정된 오차장이 후보 사이에서 얼마나 거칠게 변하는지에 달려 있었다. 두 계열 모두 고정 dither 표를 거칠게 할수록 손상이 커졌다(6과제 중 5과제, 3과제는 문턱 경계). DINO-WM의 전체 크기 표는 이용되지 않을 만큼 고와서 공유도 새로움도 손상을 가르지 못했고, TD-MPC2에서는 둘 다 필요했다. 손상을 표 크기의 정량 법칙으로 맞추지는 못했다."
  - (영문) "How much sharing and freshness matter depended on how coarsely the fixed error field varies across candidates. In both families, coarser fixed dither tables did more damage (5 of 6 tasks, three exactly at the threshold). DINO-WM's full-size tables were fine enough not to be exploited, so neither sharing nor freshness changed the damage there, whereas TD-MPC2 needed both. We could not fit the damage with a quantitative table-size law."
- **선택지**
  - (a) "표 크기 법칙"으로 통합한다(설계 대응의 강한 판)
    - 장점: 계열 차이를 한 원리로 설명한다.
    - 단점: H4a가 경계이고, TD-MPC2 var_cand가 맞지 않으며, 정량 법칙이 없다.
  - (b) "모델 의존적"으로만 보고한다
    - 장점: 가장 보수적이다.
    - 단점: H4a 성립(두 계열)과 DINO-WM 거친 표 결과를 쓰지 못한다.
- **JKAIA 원고**: "the damage of fixed dither shrinks as its table grows (5 of 6 tasks)". (a)에 가깝고, 계열 차이는 서술하지 않았다.

### [결정 12] det+shift의 서술 (이 보고가 더함)

- **정할 것**: det+shift를 대안 처방으로 쓸지, 기전 대조로 쓸지
- **근거**
  - E1 평균 하락: Meta-World 4.0%p(sh-iter와 같음), DMC 30.9%(sh-iter 36.5)
  - drift 비가 sh-iter와 같은 수준이다(0.06, 0.15, −0.02).
  - 오차 분산이 절반이다(Δ²/12).
  - E4가 미리 예측했다(det+shift ≤ sh-iter).
  - 설계 4.2절이 "지속 편향만 깨도 충분한가"를 가르는 대조로 정했다.
  - 처방(H3a)은 sh-iter로 정의됐고, det+shift는 E5·E7에서 재지 않았다(5.8절).
- **추천안**: 기전 절에서 대조로 쓴다. 처방은 sh-iter로 둔다.
  - "결정적 반올림을 그대로 두고 공유 격자만 반복마다 옮겨도(det+shift) sh-iter와 같은 손상 제거가 나왔다. 필요한 것은 확률적 반올림 자체가 아니라 반복 사이의 지속을 깨는 것이다."
  - (영문) "Keeping deterministic rounding but shifting a shared grid at every iteration (det+shift) removed as much damage as sh-iter: what matters is breaking persistence across iterations, not stochastic rounding as such."
- **선택지**
  - (a) 부록 표에만 둔다
    - 장점: 처방 메시지가 단순하다.
    - 단점: 기전을 가장 깔끔하게 보이는 대조를 버린다.
  - (b) det+shift를 공동 처방으로 제시한다
    - 장점: 오차 분산이 절반이고, 정수 kernel에서 offset을 bias로 접을 수 있다(E0T §7.2).
    - 단점: 사전등록 처방이 아니고, 강한 PTQ 위와 실제 kernel에서는 재지 않았다.
- **JKAIA 원고**: 변형 정의에만 나오고 결과 서술은 없다.

### [결정 13] 사전등록의 표기 (이 보고가 더함)

- **정할 것**: 논문에서 사전등록을 어떻게 적을지
- **근거**
  - 내부 문서 v1~v5.3이다. LOG에 고정 시각과 sha256이 있다(v5.1만 없음, [불일치 D5](#10-불일치-목록)).
  - 지금 DESIGN_main.md의 sha256이 LOG의 v5.3 기록과 같다(이 보고 계산).
  - E1 뒤에 더한 것(v5·v5.2: H2a′, H2a_indep, DMC 설명, E3 sweep의 walker)과 사후 점검(v5.3)이 있다.
  - 결과 전 코드 수정 가운데 허용 오차 수정이 H4a 판정을 바꿨다.
- **추천안**: "내부 사전등록"으로 쓰고, 사전등록 표에 시점 열(v1 / E1 뒤 / 사후)을 둔다. DESIGN_main v1~v5.3과 LOG의 시각·sha256을 보충자료로 공개한다.
  - "가설, 문턱, 분석 코드는 단계마다 실행 전에 문서로 고정했고, 시각과 sha256을 기록했다(내부 사전등록이며 외부 등록은 아니다). E1 결과를 본 뒤 더한 검정(H2a′, DMC 설명)과 사후 점검은 표에 따로 표시했다."
  - (영문) "Hypotheses, thresholds and analysis code were frozen in documents before each stage, with timestamps and SHA-256 hashes (an internal pre-registration, not an external one). Tests added after seeing stage-1 results (H2a′, the DMC rule) and post-hoc checks are marked separately in the table."
- **선택지**
  - (a) "pre-registered"만 쓴다
    - 장점: 짧다.
    - 단점: 외부 등록(OSF 등)으로 오해될 수 있다.
  - (b) "사전등록"이라는 말을 피하고 "미리 정한 판정 규칙"으로 쓴다
    - 장점: 오해가 없다.
    - 단점: 시각·해시 기록이라는 강점을 덜 드러낸다.
- **JKAIA 원고**: "Internal pre-registration ... timestamped documents"이고, 표에 "added after stage 1"을 표시했다. 추천과 같다.

### 7.14 결정 요약

| # | 정할 것 | 추천 | 다른 선택지 | JKAIA 원고 |
|---|---|---|---|---|
| 1 | 기전 문장 | (C) v5.2 문구 + H2c, 본문에 영역 효과·"지속만으로는 부족" 한정 | (A) 그대로, (B) 첫 문장부터 한정 | (C). "overrates the plan it selects"는 고칠 것을 권함 |
| 2 | 처방 범위 | "반복하는 계획기", 본문에 잰 범위 | (a) 조건 없음, (b) 반복 수 문턱 | 같음 |
| 3 | 과제 범위·DMC | Meta-World·DINO-WM. DMC는 본문 "처방이 약한 곳" 문단(S_det, ρ 0.83) | (a) 한계 절에만, (b) DMC 포함(비권장) | 같음(10-24 승인 기록) |
| 4 | H3b | 계열별: TD-MPC2 10/19, DINO-WM 1/3, 강한 PTQ가 손상 대부분을 없앰 | (a) "강한 PTQ 위에서도 이득", (b) 부정 결과로만 | 같은 방향 |
| 5 | RQ5·H5 | 같은 비트 오버헤드는 잡음 수준(최대 6.9%, 재측정 중앙값 ±3.2%), 낮은 비트 지연 이득 없음 | (a) "10% 미만"만, (b) 한계로만 | 본문 같음, 초록에 "median" 권함 |
| 6 | QAT | 결론 없음. TD-MPC2 약한 기준선, DINO-WM은 최선 PTQ와 비슷. 학습 없는 처방 | (a) 두 계열 같은 단서, (b) 부록으로 | 충돌 없음 |
| 7 | 경계 표시 | 표에 †, 각주에 여유 폭, 본문 한 단락 | (a) 표에만, (b) "판정 불가"(비권장) | H4a만 표시 |
| 8 | 부정 결과 | 본문 단락(이유 포함) + 부록 | (a) 부록만, (b) OEI 분해를 진단 기여로 | 같음 |
| 9 | 제목·한 줄 주장 | 제목 유지, 한 줄 주장에 범위 추가 | (b) "Iterating Planners Exploit Persistent…", (c) "Same Bits, Different Plans…" | 같은 제목 |
| 10 | 진단 문구 | "같은 비트에서 1-step 오차는 반올림 방식 사이의 차를 못 본다. multi-step은 가른다" | (a) 일반 명제, (b) 계획 편차 포함 | 같은 방향 |
| 11 | 2축·표 크기 | 정성적 통합 + 모델 의존 + 정량 법칙 없음 | (a) 표 크기 법칙, (b) 모델 의존만 | (a)에 가까움 |
| 12 | det+shift | 기전 대조로 본문에 | (a) 부록만, (b) 공동 처방 | 서술 없음 |
| 13 | 사전등록 표기 | "내부 사전등록" + 시점 열 + 보충자료 공개 | (a) "pre-registered"만, (b) 표현 회피 | 같음 |

---

## 8. 편차와 사고 요약, 비용과 시간

### 8.1 편차 (사고는 2.5절)

| # | 내용 | 결과·판정 영향 | 기록 위치 |
|---|---|---|---|
| 1 | 설계 문구 "held 20 episode"를 체크포인트 3개 합으로 해석했다(Meta-World 과제당 60, DMC 30) | 없음(결과 전) | DM §0.2 |
| 2 | 1차 k = 3인 과제의 k−1 = 2 칸을 규칙대로 넣었다(설계 후보 범위 {3..6} 밖) | 없음 | DM §0.2 |
| 3 | 20%p에 이르는 k가 없는 drawer-open과 PointMaze에 R2-13 대체 규칙을 썼다. sweep-into는 k=3 하락이 정확히 20.0%p였다 | 없음(drawer-open k=3은 eval에서 하락 실재) | DM §0.2·§4.1 |
| 4 | Wall: 규칙 문자 그대로 k=5를 골랐다(held에서 정확히 20%p, p = 0.22). eval에서는 확증 칸 셋 모두 det 하락이 실재하지 않았다. k=3을 탐색 칸으로 덧붙였다 | Wall이 확증 판정에 기여하지 않음(6절) | DM §4.1, E1D §3 |
| 5 | DINO-WM은 open-loop라 sh-step = sh-ep이므로 sh-plan 하나만 돌렸다 | 없음 | DM §4.2 |
| 6 | PushT eval 쌍을 held와 프레임이 겹치지 않게 다시 뽑았다(v3, 사용자 승인) | 없음(PushT E1 전) | DM §5 |
| 7 | DINO-WM OEI 확증 지표를 비양자화 모델 비용 기준으로 정했다. 설계 4.3절의 "참 비용" 문구를 정정했다(v3, 사용자 승인) | 없음(OEI 분석 전) | DM §5 |
| 8 | det+shift의 조작적 정의(unit별 offset, 후보 공유, forward마다 새로) | 없음 | DM §0.4 |
| 9 | 두 계열의 OEI 구현이 달랐다(TD-MPC2는 같은 draw, DINO-WM은 독립 draw). E2부터 두 판을 모두 기록했고, TD-MPC2 H2a 3칸은 독립 draw로 재측정했다(사후 정정, 동결 판정은 그대로) | 동결 판정 불변 | LOG 10-22 08:00Z, DM §9.1 |
| 10 | H2c 확증 sweep을 elite 수 고정으로 바로잡았다. 비율 고정은 TD-MPC2 보조로만 돌렸다(v5.1, E2 자료 전) | 판정 정의의 결과 전 수정 | DM §8.2 |
| 11 | E5 MSE clip α 격자를 넓혔다(k=3에서 102/144가 아래 끝에 걸림. v5.1, E5 결과 전) | 기준선을 약하게 두지 않으려는 보정 | DM §8.1 |
| 12 | E3에 var_cand 측정을 더했다(v5.1, 판정 밖 보조) | 없음 | DM §8.3 |
| 13 | E3 TD-MPC2의 여섯째 칸을 indep으로 채웠다. walker를 sweep 과제에 넣었다(E1에서 해로웠던 칸, 결과 뒤 선택임을 밝힘) | walker는 H4a TD-MPC2 몫의 불성립 과제 | DM §7.2 |
| 14 | 모든 설정을 프로토콜 episode 수로 돌렸다. 그래서 설계 6.9절의 E2 예산을 넘었다(DINO-WM E2만 약 30 GPU-시간) | 비용만(서버) | DM §7.0, LOG 10-22 07:50Z |
| 15 | E7: torchao 기준선 생략, CUDA graph 미구현, int4 성공률은 int8 그릇판, packed int4는 지연에만 사용 | 기술 범위 축소(6절) | LOG 10-22 09:05Z, E7T §1 |
| 16 | E7 작업자가 코드 대조를 마치기 전에 지연 요약을 보았다(공개). 코드 수정은 설계 문구와 지시에 근거한 것뿐이다 | 판정 영향 없음 | E7T 머리, LOG 10-24 02:30Z |
| 17 | ρ 문턱 비교에 허용 오차 1e-9, bootstrap ρ 동순위를 평균 순위로, `--final` 한 번 보장(결과 전) | **허용 오차는 H4a 판정을 바꾼다**(2.5절 사고 8) | E2T §3, E3T §3, E3D §8 |
| 18 | H3b를 칸별 95% CI로 보정 없이 판정했다(조정자 결정) | — | LOG 10-23 22:27Z |
| 19 | DINO-WM H4a 자격: Wall k=5를 빼고, "과반"을 절반 초과로 읽었다(2개 중 2개) | 다른 읽기로도 결론이 같다(E3D §0) | LOG 10-23 22:23Z |
| 20 | DINO-WM QAT에는 "약한 기준선" 단서를 달지 않았다(조정자 판단, v5.3 §10.1 적용) | 서술만 | LOG 10-24 12:29Z |
| 21 | DINO-WM QAT를 120분 예산에서 멈췄다(10,083 step). 사전등록 문구 "약 2 GPU-시간" 그대로다 | 없음 | E5D §10 |
| 22 | 실행 순서 조정: GPU 0에 DINO-WM 두 번째 사슬을 띄우고, QAT 단계는 GPU 1로 몰았다 | 없음(칸의 GPU 간 배정은 v5 §7.0이 허용) | LOG 10-24 02:17Z·02:25Z |
| 23 | E2 DINO-WM: sh-iter의 같은 draw mf판 sel_last는 GPU 재평가가 필요해 재지 않고 elite판으로 보고했다 | 기술 항목 하나 | LOG 10-23 15:55Z, E2D §9 |
| 24 | E1 DINO-WM 작업자가 서버 /tmp의 자기 시험 파일을 지웠다(`~/pmx` 밖 삭제 금지 규칙 위반) | 없음 | E1D §12 |
| 25 | E4 작업자가 몇 분 동안 CPU 프로세스 9개를 돌렸다(상한 8) | 없음 | LOG 10-22 09:15Z |
| 26 | E6(선택)를 하지 않았다 | H6 증거 없음 | MAIN_STATUS |
| 27 | 사후 점검 두 가지(QAT+none, 짝 모드 지연 재측정)를 v5.3으로 더했다(사용자 승인, 기술) | 판정 불변 | DM §10 |
| 28 | 정기 점검을 사용자 요청으로 두 번 멈췄다 | 사고 3의 발견 지연(2.5절 사고 7) | LOG |

### 8.2 서버 GPU 시간

| 단계 | TD-MPC2(GPU 0) | DINO-WM | 출처 |
|---|---|---|---|
| E0 | 약 5 h(MPS, DINO-WM과 공유) | 약 7 h(GPU 0 약 3 h 공유, GPU 1 약 4 h) | E0T §11, E0D §11 |
| E1 | 약 12.3 h(episode 11.9 h + 칸 단위 재실행 25분) | 35.6 GPU-h + v5.2 계산 약 1.2 | E1T §13, E1D §13 |
| E4 | (CPU 11.6 CPU-시간, GPU 없음) | — | E4 §10 |
| H2a 재측정 | 약 0.5 h | — | E1T §15 |
| E2 | 10.75 h | N 18.8 + elite·opt_steps 10.1 GPU-h | E2T §10, E2D §10 |
| E3 | 6.3 h | 8.4 GPU-h | E3T §11, E3D §9 |
| E5 | 약 7.3 h | 약 18.1 GPU-h | E5T §11, E5D §11 |
| E7 | 성공률 약 3.3 h + 지연 46분 | — | E7T §10 |
| 사후 | QAT+none 22분 + 지연 95분 | — | PH §6 |
| **합(이 보고 계산)** | **약 48 h** | **약 99 GPU-h** | 두 계열 합 **약 147 GPU-시간** |

- **이 값은 점유 시간이다.** TD-MPC2는 GPU 0 한 장을 MPS 작업자 16개가 쓴 벽시계이고, DINO-WM은 job의 GPU 부분 합이다(GPU 공유로 느려진 시간 포함). 단독 GPU-시간과 같지 않다.
- **설계 대비**(이 보고 계산)
  - 설계 6.9절의 총 추정은 약 92 GPU-시간이다(E6 5 포함). 실제로 돈 단계 기준 추정 87 대비 약 1.7배다.
  - E0+E1: 설계 56(E0 6 + E1 50) 대 실제 약 61
  - E2~E7 + 사후: 설계 약 36(E6 포함) → v5 추정 약 85(TD-MPC2 약 30, DINO-WM 약 55) 대 실제 약 86(TD-MPC2 약 31, DINO-WM 약 55)
  - 늘어난 이유(DM §7.5): A5000이 4090보다 느리다(같은 일에 약 2배, E0D §4.2). 모든 설정을 프로토콜 episode 수로 돌렸다. E3를 모든 과제의 1차 k로 넓혔다.
- 서버 비용은 0이다.

### 8.3 시간

- 10-21 13:00Z 시작 → 10-22 19:40Z go 판정(약 30.7시간) → 10-24 16:26Z 서버 GPU 작업 끝(약 75.4시간) → 16:39Z 단계 분석 끝(약 75.7시간). 모두 이 보고 계산이다.
- v5 §7.5의 예상("끝은 10-24 오후(UTC) 무렵")과 맞았다. 사고 3으로 GPU 0이 약 2.3시간 놀았다.
- 맥북에서는 파일 편집, 결과 읽기, 짧은 점검만 했다(각 보고).

### 8.4 RunPod

- **A는 RunPod를 쓰지 않았다($0).** 설계의 Meta-World 대체 경로(약 $10~15)도 쓰지 않았다(E0T §3).
- 공유 계정의 RunPod 누적 $20.27은 모두 B의 M2다(MAIN_STATUS 점검 기록, FINAL_B §7.1). A 기간 중 켜진 pod는 없었다.
- 예산 감시 스크립트가 두 번 꺼져 있었다(10-23 08:35Z, 15:27Z에 발견해 다시 켬, pod 없음; LOG).

---

## 9. 논문 구성 제안, 남은 일(집필)

### 9.1 투고 경로와 일정

- **루트 README(10-22 기준)와 조정자 지시**: arXiv 12월 말, ICML·RSS 2027
  - ED §8 표: ICML 2027(마감 추정 2027-01-22), RSS 2027(1월 말 예상), 대안 IROS 2027·RA-L(2027-03-01), CoRL·NeurIPS 2027(5월 말 예상), 병행 arXiv + ICRA 2027 워크숍
- **paper/README(10-24)**
  - JKAIA 2026 General(Poster) 트랙: 제출 마감 10-25, 2쪽, 영문 원고 v1 완성, 저작권 양도 안 함
  - 그다음 TMLR, ICLR 2027 워크숍(목록 11-29 공개, non-archival인 곳을 고름)
  - 그 전 계획은 RSS 2027 stage 1(12-04)이었다고 적었다. RSS와 TMLR은 동시 투고가 안 된다.
- **처리**: 두 계획이 다르다([불일치 D2](#10-불일치-목록)). 아래 구성은 두 경로 모두에 맞게 썼다. 사용자 확인이 필요하다. 날짜는 투고 직전에 공식 공지로 다시 확인한다(ED §8).
- 분량: TMLR처럼 분량이 넉넉하면 아래 구성을 그대로 쓴다. 워크숍(4~8쪽)이면 5~7절을 묶고, 8절은 짧은 단락과 부록으로 돌린다.

### 9.2 절과 그림·표 후보

그림 경로는 `A_dithered_world_models/` 기준이다.

| 절 | 내용 | 그림·표(기존 파일) | 수치 출처 |
|---|---|---|---|
| 1 서론 | 압축과 계획, 1-step 오차가 못 보는 손상, 고전 원리(CRN, SAA, SR)를 먼저 인정(ED §2.3) | 그림 1 후보: 손상 칸 전체의 det 대 sh-iter 손상(`paper/jkaia2026/figures/fig_main.pdf`의 (a)) | E1T §5, E1D §5 |
| 2 관련 연구 | ED §2.1 표와 Min et al.과의 대비 그림(ED §2.3). 2026-10-21 이후 논문 재조사 | — | ED §2 |
| 3 설정 | 양자화 위치, 변형 7종(2축), OEI = sel_last + drift, 계획기 설정 | 표: 변형과 두 축(DM §2 4.2) | DM §0·§2 |
| 4 현상과 처방 | H1, H3a(Meta-World·DINO-WM), 사전등록 표 | 표 1: 사전등록 판정(이 보고 3절을 줄여 시점 열을 둠). 부록: 칸별 H3a 표(E1T §5.2, E1D §9.1) | E1T, E1D |
| 5 기전 | H2a′와 양성 대조, det+shift, 반복 수, H2c, 영역 효과 | `main/results/E2_tdmpc2/e2_iteration_curves.png`(JKAIA 그림 1(b)의 원자료), `main/results/E2_tdmpc2/e2_h2c_nsweep.png`. 새로 그릴 것: drift 비 막대(E2T §6.2, E1D §8.1), DINO-WM opt_steps 곡선(`family_points_dinowm.json`) | E2T, E2D, E1D §8 |
| 6 오차 구조 | 2축 대조, 고정 표 크기 sweep(H4a), 이론 곡선 | `main/results/E3_tdmpc2/e3_table_sweep.png`, `main/results/E4/fig6_neff.png`. 새로 그릴 것: DINO-WM sweep(`family_points_e3_dinowm.json`) | E3T, E3D, E4 §5.9 |
| 7 처방이 약한 곳 | DMC와 S_det | 새로 그릴 것: S_det 대 red 산점도(`main/results/E3_tdmpc2/e3_tdmpc2_final.json`) | E3T §7 |
| 8 배치 | H3b, QAT, 가중치(탐색), E7 성공률·지연 | `main/results/E5_tdmpc2/e5_h3b_cells.png`, `main/results/E7_tdmpc2/e7_pareto.png`, `main/results/posthoc_tdmpc2/posthoc_latency_pairs.png` | E5T, E5D, E7T, PH |
| 9 한계와 부정 결과 | 이 보고 6절과 4.2절 | — | — |
| 부록 | 사전등록 이력과 sha256, 이론 유도(E4 2장)와 모의, 칸별 표, OEI 분해, 2축 표, 가중치, QAT, 지연 두 모드, 재현 정보 | `main/results/E4/fig1_singleshot.png`~`fig7_jensen.png`, `main/results/E1_tdmpc2/e1_tdmpc2_posthoc_oei_decomposition.md`, `main/results/posthoc_tdmpc2/posthoc_qatnone.png`, `main/results/posthoc_tdmpc2/posthoc_latency_modes.png` | 각 보고 |

- 아직 그림이 없는 것: drift 비 막대, DINO-WM opt_steps·elite 곡선, DINO-WM 표 크기 sweep, S_det 대 red 산점도, DINO-WM H3b 칸. 모두 기존 JSON으로 그릴 수 있다(새 계산 없음).

### 9.3 남은 일

1. 7절의 결정 13개를 사용자와 정한다. JKAIA 원고에 기록된 10-24 승인 범위와 맞춘다(D2).
2. **집필 전에 arXiv 경쟁 연구를 다시 찾는다.**
   - 대상: Min et al. 2026(arXiv 2610.05550, 가장 위협적)의 후속, Fu et al.(2602.02110)과 PreDE(2609.19441)의 후속, VLA 양자화(VLAQuantBench 2609.25376, QVLA 2602.03782)
   - 2026-10-21 이후 "stochastic rounding / dither + world model / MPC / planning" 계열 논문
   - ED §2.2의 공백 1~4가 아직 비어 있는지 확인한다. "최초"는 쓰지 않는다(4.3절).
3. 9.2절의 새 그림을 기존 JSON으로 그린다(새 계산 없음).
4. **사전등록 부록**을 쓴다.
   - DM v1~v5.3의 고정 시각과 sha256(LOG). v5.1은 해시가 없다(D5).
   - E1 뒤 추가(v5·v5.2)와 사후(v5.3)를 구분한다.
   - 8.1절 편차 목록을 싣고, 허용 오차 수정이 H4a를 바꾼 점을 밝힌다.
5. 다중 비교 단락을 쓴다(3절 끝의 내용).
6. **재현 부록**을 쓴다.
   - 코드: `main/code/tdmpc2`, `main/code/dinowm`, `main/code/theory`
   - 환경: torch 2.7.1+cu126·2.4.1+cu124, tdmpc2 `332f19e`, dino_wm `0a9492f`, 사용자 공간 Mesa 24.3.4
   - seed 분할: DM §0.3·§4.3·§7, 장비(A5000, MPS)
7. 불일치 목록을 반영한다: D1(13칸), 시각 표기 통일(D3·D4).
8. 이론 부록: E4의 유도, §6.1의 어긋남 10개, 실험과의 대조(5.11절)
9. JKAIA 원고(마감 10-25): 이 보고가 확인한 두 곳(D7)을 사용자가 판단한다. 이 보고는 원고를 고치지 않았다.
10. 사용자 확인 뒤 루트 README의 A 한 줄 주장과 투고 계획을 갱신한다([결정 9], D2). 이 보고는 README를 고치지 않았다.
11. 투고처와 일정을 다시 확인한다.
12. 비용은 "RunPod $0, 서버 GPU 약 147 GPU-시간(점유 기준)"으로 쓴다(8.2절).

---

## 10. 불일치 목록

| # | 내용 | 문서 | 이 보고의 처리 |
|---|---|---|---|
| D1 | E1T §5.2는 Meta-World에서 red_sh-iter ≥ 0.9인 칸을 "19칸 중 14칸"이라고 적었다. JSON으로 세면 13칸이다. sweep-into k=3의 red 0.8966이 표에 0.90으로 반올림됐다. paper/README §3도 같은 점을 적었다 | E1T §5.2, `E1_tdmpc2/e1_tdmpc2_summary.json`(이 보고 확인), paper/README §3 | 13칸으로 쓴다. 판정(red ≥ 0.5, 19/19)과는 무관하다 |
| D2 | `paper/README.md`(10-24 14:55Z 수정)는 두 가지를 적었다. (1) 사용자가 10-24에 JKAIA 원고의 주장 범위 "추천 1~5"를 승인했다: 기전(v5.2 + H2c), "반복하는 계획기", Meta-World·DINO-WM과 DMC 본문 문단, PTQ 계열별·QAT 결론 없음, 지연. (2) 투고처를 JKAIA·TMLR·ICLR 2027 워크숍으로 바꿨다(이전 계획은 RSS 2027 stage 1, 12-04). LOG와 MAIN_STATUS에는 이 기록이 없다. 루트 README와 조정자 지시는 "arXiv 12월 말, ICML·RSS 2027"이다 | paper/README §3·§5, LOG, MAIN_STATUS, README | 7절 각 항목에 JKAIA 원고 문구를 함께 적었고, 9.1절에 두 계획을 함께 적었다. **사용자 확인이 필요하다** |
| D3 | DM v1 고정 시각이 DM 머리·LOG는 18:14Z, MAIN_STATUS는 18:12Z다. v2는 DM·LOG 20:01Z, MAIN_STATUS 20:00Z다 | DM, LOG, MAIN_STATUS | 2.4절은 DM·LOG 시각을 따랐다. 판정 영향 없음 |
| D4 | v5.2 시각이 DM §9 제목·E1D §8은 09:40Z, LOG·MAIN_STATUS는 09:50Z(사용자 승인과 고정)다 | DM §9, E1D §8, LOG, MAIN_STATUS | 2.4절은 LOG의 09:50Z(승인과 sha256 기록 시각)를 썼다. 둘 다 E2 자료와 DINO-WM 분해 전이라 영향 없음 |
| D5 | LOG에는 v1·v2·v3·v4·v5·v5.2·v5.3의 sha256이 있고 v5.1만 없다. 지금 DESIGN_main.md의 sha256은 v5.3 기록(`eee46e0a…`)과 같다(이 보고 계산) | LOG 10-22 09:30Z | 사전등록 부록에 "v5.1은 별도 해시 없음, v5.2 해시가 v5.1 내용을 포함"으로 적기를 권한다(9.3절 4) |
| D6 | 파일럿 수가 다르다. ED 머리는 "파일럿 8개", ED §5 표는 9줄, 루트 README는 "파일럿 9개"다 | ED 머리·§5, README | R2-3b를 R2-3의 후속으로 세면 8개다. 논문에는 9개(표의 줄 수)로 쓰고 R2-3b가 후속임을 밝히기를 권한다 |
| D7 | JKAIA 원고 문구 두 곳. (1) 초록·본문의 "overrates the plan it selects"는 det drift가 양수인 칸(PushT k=3, +0.243)에서 맞지 않는다. (2) 한국어 검토본의 "두 계열의 손상 칸 어디에서도 det보다 나쁘지 않다"는 TD-MPC2 DMC walker k=3(sh-iter가 해로움)과 충돌할 수 있다. 영문 "never worse than det in their affected cells"는 Meta-World와 DINO-WM을 가리켜 맞다 | `paper/jkaia2026/main.tex`, `main_ko.tex`, E1D §8.2, E1T §5 | 이 보고는 원고를 고치지 않았다. [결정 1]과 9.3절 9에 적었다 |
| D8 | ED §2.3은 "반복이 고정 오차를 증폭한다"를 쓰지 않을 주장으로 정했다(R2-14: closed-loop로 지지되지 않음). E2 TD-MPC2는 이를 재현했지만(det 손상이 반복 수와 거의 무관), E2 DINO-WM은 opt_steps와 함께 det 손상이 커졌다(+7 → +24%p, none 성공률도 48 → 83%로 바뀜) | ED §2.3, E2T §5, E2D §7.3 | 일반 명제로 쓰지 않고 계열별 기술로 쓴다(4.3절) |
