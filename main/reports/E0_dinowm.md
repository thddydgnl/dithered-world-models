# E0 DINO-WM 단계 보고 (A 논문: dithered low-bit world model)

- 작성: 2026-10-21, E0 DINO-WM 작업자
- 기준 문서: [../../EXPERIMENT_DESIGN.md](../../EXPERIMENT_DESIGN.md) §4.2·4.3·6.0~6.3·6.9·6.10, [../../../MAIN_RULES.md](../../workspace/MAIN_RULES.md)
- 작업 로그: [../worklogs/E0_dinowm.md](../worklogs/E0_dinowm.md) · 코드: [../code/dinowm/](../code/dinowm/) · 결과: [../results/E0_dinowm/](../results/E0_dinowm/)
- 장소: 대학 서버 `kiis-mvf-gpu`(RTX A5000 24GB × 2, 드라이버 560.35.05, CUDA 12.6). RunPod는 쓰지 않았다(비용 $0)
  - 13:10~16:28Z는 **GPU 0**(TD-MPC2 작업자의 MPS 작업과 공유. 13:48~14:17Z, 14:40~14:42Z에는 B의 작은 작업도 있었음)
  - 15:53Z 조정자 재배정 뒤로는 **GPU 1**(단독). 칸 하나((과제, k))는 한 GPU 안에서만 돌렸고, job JSON마다 `cuda_visible`(GPU 번호)과 `cuda_device_order`를 기록했다

## 0. 요약

| 항목 | 결과 |
|---|---|
| 환경 | uv venv 2개(`a-dinowm`, PointMaze용 `a-dinowm-pm`). torch 2.4.1+cu124 등 파일럿 pod와 같은 판. dino_wm은 파일럿 commit `0a9492f`. 범위 보정 값이 파일럿과 같다(PushT 완전 일치, Wall 2층에서 bf16 한 칸) |
| 재현(R2-13 주요 칸, 50 ep, GPU 1) | **두 과제 모두 파일럿과 일치.** PushT k=4: none 74%(파일럿 78%), det 56%(52%), sh-iter 74%(80%), ind 74%(82%), sh-fixed 84%(82%); det 하락 +18%p [4, 32](+26), sh-iter 감소율 1.00(1.08). Wall k=3: none 48%(48%), det 12%(8%), sh-iter 48%(58%), ind 54%(52%), sh-fixed 54%(56%); det 하락 +36%p [20, 52](+40), sh-iter 감소율 1.00(1.25). 서버 대 파일럿 성공률 차는 어느 변형도 유의하지 않다. OEI 파일럿 값(det 무작위 후보 +0.30, 고른 후보 −0.09)은 +0.299, −0.103으로 재현 |
| 처리량(GPU 1 단독) | s/episode: PushT none 30.2 · det 34.2 · 무작위 변형 37.8, Wall 3.6 · 4.6 · 5.4, PointMaze 10.1 · 13.3 · 16.1(진단 포함, E와 무관). TD-MPC2 MPS와 같은 GPU를 쓰면 **2.2배 느리다**(PushT none 65.6, det 75.3; Wall 8.1, 10.2) |
| E1 GPU-시간 | 6,000 episode-run(과제 3 × [none + nonealt + k 3개 × 6변형] × 100). **단독 약 32 GPU-시간**(PushT 20.3, Wall 2.8, PointMaze 8.4), **TD-MPC2와 공유하면 약 69시간**. 4배 느려진 원인은 대부분 GPU 공유(2.2배)와 A5000이 4090보다 같은 일에 약 2배 느린 것이다. 설계 가정(200 ep/h, 31.5시간)과 단독 추정은 거의 같다 |
| k 선택(held 20) | **PushT k=4 → 칸 {3,4,5}**(파일럿과 같음). **Wall은 규칙 문자 그대로 k=5 → 칸 {4,5,6}**(k=5 하락이 정확히 20%p, p=0.22, k에 대해 비단조). **PointMaze는 20%p에 이르는 k가 없다**(k3 −15%p가 최대). 결정 필요(§10) |
| PointMaze | **동작함**(mujoco-py 2.1.2.14 + OSMesa, TD-MPC2 작업자의 MuJoCo 2.1·gl24를 읽기 전용으로 사용). 제외하지 않았다. 스모크·held k 선택 완료 |
| E1 runner | 7변형 + nonealt, 진단(1/3/5-step 오차, 계획 편차 입력, OEI 입력과 시뮬레이터 참 비용) 포함, job마다 JSON 1개, 재개 가능. 스모크 통과. **E1은 시작하지 않았다**. manifest 초안 `code/dinowm/manifests/E1_dinowm.json` |

## 1. 무엇을 돌렸나

| 실행 | 분할 | 내용 | GPU | 결과 |
|---|---|---|---|---|
| 범위 보정 | train 창 64개 | 층별 [min, 99.99%] (R2-13 절차) | 0(PushT·Wall), 1(PointMaze) | `results/E0_dinowm/ranges/` |
| 스모크 | smoke(pair seed 7) | 7변형 + nonealt, PushT·Wall 2~4 ep, PointMaze 1 ep | 0, 1 | `runs/E0_smoke*` |
| 배치·메모리 점검 | smoke | E=1·2(Wall 2·4)의 수치 동일성, 메모리 | 0 | `runs/E0_memtest` |
| **k 선택** | held(pair seed 3, 20 ep) | none + det k = 3, 4, 5, 6 (+ PointMaze k=2 참고) | PushT·Wall 0, PointMaze 1 | `runs/E0_ksel*`, `ktable.csv` |
| **재현** | repro(= R2-13 eval 50 ep) | 파일럿 5변형, 파일럿 k·범위·seed·dither 표 | 1 | `runs/E0_repro` |
| **처리량 벤치** | smoke | PushT·Wall·PointMaze × none·det·sh-iter × E 2종 | 1(단독) | `runs/E0_bench_g1` |
| 예측 오차 | valid 창 64개 + held 시작 8개 | k = 2~7 × 6변형, 1/3/5-step | 1 | `err_<task>_main.json` |

- eval 분할(E1 판정용, pair seed 5)은 쌍을 만들어 기록만 했고, 그 위에서 계획을 한 번도 돌리지 않았다

## 2. 환경과 수치 설정

- 서버 작업 폴더 `~/pmx/A/dinowm_main/`(코드 `code/` = 맥북 `main/code/dinowm/`과 같음, 결과 `results/`, 모집단 배열 `pops/`, 쌍 캐시 `pairs/`)
- `setup_server.sh`(파일럿 setup13.sh의 사용자 공간판)
  - CPython 3.11.17(uv 관리, `~/pmx/envs/uv-python`), venv `~/pmx/envs/a-dinowm`
  - torch 2.4.1+cu124(cuDNN 9.1.0), torchvision 0.19.1, triton 3.0.0, numpy 1.26.4, scipy 1.17.1, gym 0.23.1, pymunk 6.2.1, hydra-core 1.3.2, omegaconf 2.3.1, opencv-python-headless 4.11.0.86, pygame 2.6.1, shapely 2.2.0, decord 0.6.0, einops 0.8.2, accelerate 1.15.0, setuptools 70.3.0(triton import에 필요)
  - DINOv2 ViT-S/14: torch hub(`TORCH_HOME=~/pmx/torch`), dino_wm `0a9492fa`(2025-03-24, 파일럿과 같은 commit)
  - 전체 목록: 서버 `~/pmx/A/dinowm_main/pip_freeze.txt`, 맥북 `results/E0_dinowm/env/`
- `setup_pointmaze.sh`: venv `~/pmx/envs/a-dinowm-pm` = 위와 같은 패키지 + mujoco-py 2.1.2.14, cython 0.29.37, setuptools 65.5.0, d4rl@89141a6(--no-deps), h5py. 두 venv의 차이는 이 추가 패키지와 setuptools 판뿐이다
- 데이터(읽기 전용): 체크포인트 `~/pmx/data/dinowm/outputs/{pusht,wall_single,point_maze}`, Wall 메타 `~/pmx/data/dinowm/wall/wall_single/*.pth`, PushT `~/pmx/data/pusht_noise`(train 18,685 / val 21 궤적, dino_wm `PushTDataset`이 기대하는 구조와 같음), PointMaze `~/pmx/data/dinowm/point_maze/*.pth`(OSF vr5gy, obs 30GB는 풀지 않음; 프레임은 env가 렌더링)
- **수치 설정(R2-13과 같음)**: 예측기 forward 한 번에 E × 300 후보 전부를 bf16 autocast로 배치 처리(r27lib.rollout_batched와 같은 계산), TF32 matmul, SDPA attention(원래 구현과 같은 수식, frame-causal mask), 양자화 연산은 kernel 안에서 fp32로 하고 bf16으로 되돌림(torch.compile), 후보 표집은 episode별 CPU generator, open-loop CEM(300 후보, top-30, var_scale 1, 첫 후보 = mu, goal_H 5, opt_steps PushT 30 · Wall 10 · PointMaze 10)
- **배치 크기 E는 수치를 바꾸지 않는다**: E=1과 2(Wall 2와 4)에서 mu_final, iteration별 최저 비용, OEI 입력이 비트 단위로 같다(ind만 커널 내부 난수열이 텐서 모양에 따라 달라짐). 그래서 E는 메모리에 맞춰 정했다(PushT 1, Wall 2, PointMaze 1; 재현은 파일럿 E와 결과가 같다)
- **두 GPU는 같은 결과를 낸다**: GPU 0과 GPU 1에서 같은 episode의 계획이 비트 단위로 같다(PushT·Wall 스모크 episode, E 교차 포함). 계획 시점 오차의 정규화 분산만 1e-7 상대 수준으로 다르다(float32 합산 순서)

## 3. 재현: R2-13 주요 칸

- 설정: R2-13 eval 분할 그대로(pair seed 4의 쌍 0~49, planner seed 2000+e), 파일럿 k(PushT 4, Wall 3), **파일럿 범위 파일**, 파일럿 dither seed 공식(sh-iter·sh-fixed 표가 파일럿과 같은 난수), GPU 1. 다른 점은 GPU(4090 → A5000)와 E(PushT 2 → 1, Wall 4 → 2; 결과에 영향 없음)뿐이다
- 비교: `analyze.py repro`(파일럿 `cem13_<task>_eval.jsonl`과 episode 단위로 짝)

### 3.1 Wall (k=3, 50 episode, GPU 1)

| 변형 | 서버 성공 | 파일럿 성공 | 하락 D [95% CI] (파일럿) | none 대비 McNemar | episode 일치율 | margin Spearman(서버, 파일럿) |
|---|---|---|---|---|---|---|
| none | 24/50 (48%) | 24/50 (48%) | — | — | 84% | 0.81 |
| det | 6/50 (12%) | 4/50 (8%) | **+36%p** [20, 52] (+40 [26, 54]) | 21/3, p = 2.8e-4 (21/1, 1.1e-5) | 96% | 0.87 |
| ind | 27/50 (54%) | 26/50 (52%) | −6 [−22, 10] (−4) | 6/9, p = 0.61 | 74% | 0.58 |
| sh-iter | 24/50 (48%) | 29/50 (58%) | 0 [−16, 16] (−10) | 8/8, p = 1 | 74% | 0.64 |
| sh-fixed | 27/50 (54%) | 28/50 (56%) | −6 [−20, 8] (−8) | 5/8, p = 0.58 | 78% | 0.54 |

- det 하락 감소율: sh-iter **1.00** [0.62, 1.62](파일럿 1.25), sh-fixed 1.17 [0.81, 1.82](1.20), ind 1.17 [0.80, 1.91](1.10). sh-iter 대 det McNemar 21/3, p = 2.8e-4(파일럿 25/0)
- 사후 진단(파일럿 §2.3 아래 표와 같은 정의)이 소수 둘째 자리까지 맞는다

| 변형 | 비용 오차 ΔJ: 0단계 / 마지막 모집단 / 고른 후보(서버) | 파일럿 | 후보별 ΔJ sd ÷ 비용 sd: 0단계 / 마지막 (서버, 파일럿) | ρ(J_v, J_none) 0단계 (서버, 파일럿) | ρ 마지막(J_v), 차 Δρ (서버) | 파일럿 |
|---|---|---|---|---|---|---|
| det | +2.365 / +0.318 / +0.037 | +2.360 / +0.320 / +0.034 | 3.36 / 2.44, 3.36 / 2.10 | 0.691, 0.692 | 0.161, −0.331 | 0.168, −0.296 |
| ind | +0.123 / +0.054 / −0.034 | +0.122 / +0.054 / −0.001 | 0.28 / 0.41, 0.27 / 0.39 | 0.965, 0.966 | 0.503 | 0.529 |
| sh-iter | +0.121 / +0.057 / +0.016 | +0.116 / +0.058 / +0.016 | 0.20 / 0.39, 0.20 / 0.36 | 0.984, 0.984 | 0.509 | 0.538 |
| sh-fixed | +0.124 / +0.046 / +0.001 | +0.121 / +0.050 / −0.003 | 0.20 / 0.38, 0.20 / 0.42 | 0.983, 0.983 | 0.498 | 0.502 |

- 계획 시점 상대 오차(episode별 iteration-0 후보, 1-step / 5-step): det 0.355 / 4.01, 확률적 3종 0.207~0.210 / 0.151~0.153(파일럿: held 시작 8개 기준 det 0.340 / 2.99, 확률적 0.195~0.204 / 0.141~0.143)
- episode 단위 결과는 비트 단위로 같지 않다(GPU가 달라 bf16 연산 결과가 달라지고, CEM이 그 차이를 키운다). 그래도 성공 여부 일치율이 74~96%이고 모든 변형의 성공률 차가 McNemar로 유의하지 않다(서버 대 파일럿, 같은 episode 짝)

### 3.2 PushT (k=4, 50 episode, GPU 1)

| 변형 | 서버 성공 | 파일럿 성공 | 하락 D [95% CI] (파일럿) | none 대비 McNemar (파일럿) | episode 일치율 | margin 중앙값 서버 (파일럿) |
|---|---|---|---|---|---|---|
| none | 37/50 (74%) | 39/50 (78%) | — | — | 96% | 0.511 (0.453) |
| det | 28/50 (56%) | 26/50 (52%) | **+18%p** [4, 32] (+26 [14, 40]) | 12/3, p = 0.035 (14/1, 0.001) | 92% | 0.928 (0.920) |
| ind | 37/50 (74%) | 41/50 (82%) | 0 [−6, 6] (−4) | 1/1, p = 1 | 92% | 0.538 (0.483) |
| sh-iter | 37/50 (74%) | 40/50 (80%) | 0 [−6, 6] (−2) | 1/1, p = 1 | 90% | 0.468 (0.476) |
| sh-fixed | 42/50 (84%) | 41/50 (82%) | −10 [−20, −2] (−4) | 0/5, p = 0.06 | 94% | 0.556 (0.463) |

- det 하락 감소율: sh-iter **1.00** [0.70, 1.67](파일럿 1.08 [0.75, 1.60]), ind 1.00 [0.70, 1.67](1.15), sh-fixed 1.56 [1.11, 4.0](1.15). sh-iter 대 det McNemar 11/2, p = 0.022(파일럿 15/1, 5.2e-4)
- 파일럿 판정 기준 S1(det 하락 ≥ 15%p인 과제에서 sh-iter가 하락의 50% 이상을 줄임)은 이 재현에서도 PushT·Wall 두 과제 모두 성립한다
- 서버 대 파일럿(같은 episode 짝)의 성공률 차는 어느 변형도 유의하지 않다(McNemar p ≥ 0.13). PushT det 하락이 파일럿보다 8%p 작지만 CI가 겹친다

| 변형 | ΔJ: 0단계 / 마지막 모집단 / 고른 후보 (서버) | 파일럿 | 후보별 ΔJ sd ÷ 비용 sd: 0단계 / 마지막 (서버, 파일럿) | ρ(J_v, J_none) 0단계 (서버, 파일럿) | ρ 마지막(J_v / J_none), Δρ [CI] (서버) | 파일럿 |
|---|---|---|---|---|---|---|
| det | **+0.299 / −0.054 / −0.103** | +0.299 / −0.051 / −0.087 | 0.259 / 0.839, 0.258 / 0.833 | 0.982, 0.982 | 0.357 / 0.692, −0.334 [−0.43, −0.24] | 0.382 / 0.678, −0.295 |
| ind | +0.009 / +0.008 / 0.000 | +0.009 / +0.009 / +0.002 | 0.068 / 0.298, 0.067 / 0.295 | 0.997, 0.997 | 0.506 / 0.533, −0.027 | 0.526 / 0.553 |
| sh-iter | +0.014 / +0.008 / +0.005 | +0.011 / +0.009 / +0.004 | 0.056 / 0.273, 0.054 / 0.295 | 0.998, 0.998 | 0.539 / 0.536, +0.003 | 0.531 / 0.579 |
| sh-fixed | +0.009 / +0.004 / +0.002 | +0.010 / +0.005 / +0.003 | 0.055 / 0.293, 0.057 / 0.274 | 0.998, 0.998 | 0.529 / 0.542, −0.012 | 0.532 / 0.580 |

- 설계 §4.3의 OEI 파일럿 값(det 무작위 후보 +0.30, 고른 후보 −0.09)이 재현된다: +0.299, −0.103. 이 정의로 잰 OEI(고른 후보 − 무작위 후보)는 det −0.40, 확률적 3종 −0.006 ~ −0.009
- 계획 시점 상대 오차(1-step / 5-step): det 0.058 / 0.306, 확률적 3종 0.062~0.063 / 0.039~0.040(파일럿: held 시작 기준 det 0.065 / 0.320, 확률적 0.067~0.071 / 0.038~0.040). PushT에서 det의 1-step 오차가 확률적 변형보다 작은 역전도 재현된다

### 3.3 재현 판단
- **두 과제 모두 파일럿과 맞는다.** 기준선, det 손상의 크기·유의성, dither의 손상 제거, 그리고 비용 오차 진단(OEI, 순위 상관)이 파일럿 CI 안에 있다. 진단 수치는 소수 둘째~셋째 자리까지 같다
- 다른 점: 계획이 비트 단위로 같지는 않다(GPU가 다름). 성공 여부 일치율은 PushT 90~96%, Wall 74~96%(Wall의 확률적 변형은 CEM 후반이 불안정해 episode 단위 결과가 민감하다. R2-7 참고)

## 4. 처리량과 E1 GPU-시간

### 4.1 속도 설정 점검(조정자 요청 1)
- 위 §2의 R2-13 설정이 모두 켜져 있다. 후보별 Python 반복은 없다(episode별 비용 계산·정렬만, 파일럿과 같음). 변형은 episode 묶음(job) 단위로 교대하고, 참 비용 재생은 CPU pool(8 worker)이 GPU와 겹쳐 처리한다(job 끝의 재생 대기 0~2초)
- 파일럿에 없던 비용은 진단 forward뿐이다. 무작위 변형은 iteration-0 기준 rollout 1회 + 오차용 추가 dither draw 2회 + 마지막 모집단 비양자화 비용 1회(PushT 30 iteration 대비 +13%, Wall·PointMaze 10 iteration 대비 +40%), det는 +2회, none·nonealt는 추가 없음

### 4.2 측정(s/episode, 진단 포함)

| 조건 | PushT none / det / 무작위 | Wall none / det / 무작위 | PointMaze none / det / 무작위 | 근거 |
|---|---|---|---|---|
| **GPU 1 단독**(15:53~16:05Z) | 30.2 / 34.2 / 37.8 | 3.6 / 4.6 / 5.4 | 10.1 / 13.3 / 16.1 | `E0_bench_g1`(E 두 가지, 속도 같음) |
| GPU 0, TD-MPC2 MPS 16 client와 공유(B 부하 구간 제외) | 65.6 / 75.3 / — | 8.1 / 10.2 / — | — | `E0_ksel` 중 깨끗한 job |
| 공유/단독 비 | 2.17 / 2.20 | 2.25 / 2.22 | (같은 비로 가정) | |
| 참고: 파일럿 RTX 4090 | 14.4 / 16.2~16.7 / 16.0~16.7 | 1.9~2.5(스모크 E=5) | — | R2-13 LOG |

- A5000 단독은 같은 일에 4090의 약 2.0~2.1배 시간이 든다(PushT none 30.2 대 14.4)
- 조정자가 본 74.8s/ep(PushT det, E=1)는 TD-MPC2 MPS와 같은 GPU를 시분할한 값이다. **느려진 원인은 대부분 GPU 공유(2.2배)이고, 나머지는 A5000 대 4090(약 2배)이다.** 설정 문제는 없었다

### 4.3 E1 추정(조정자 요청 3)

- E1 규모: 과제마다 100 episode × [none 1 + nonealt 1 + k 3개 × 6변형(det, ind, sh-iter, sh-step, sh-ep, det+shift)] = 2,000 episode-run, 과제 3개 합계 **6,000**(none은 k와 무관하므로 과제당 한 번. 설계 표의 "약 6,300"은 none을 k마다 센 값)

| 과제 | none·nonealt 200 | det 300 | 무작위 1,500 | 단독 합계 | TD-MPC2와 공유 시(×2.2) |
|---|---|---|---|---|---|
| PushT | 1.7 h | 2.9 h | 15.8 h | **20.3 h** | 약 45 h |
| Wall | 0.2 h | 0.4 h | 2.3 h | **2.8 h** | 약 6 h |
| PointMaze | 0.6 h | 1.1 h | 6.7 h | **8.4 h** | 약 18 h |
| 합계 | | | | **31.5 h**(≈ 190 ep/h) | **약 69 h** |

- runner 시작(모델 적재 약 1분/과제)과 재생 대기는 무시할 만하다(+1~2%). 재현 실행(GPU 1 단독, 진단 포함)의 실측 속도도 같다: PushT none 30.4, det 33.9, 무작위 37.3 s/ep, Wall none 5.3(일부 구간에 내 참고 실행과 겹침) s/ep
- **GPU 배치별 벽시계**(칸 = (과제, k)는 한 GPU 안에서)
  - GPU 1 하나만(단독): 약 **32시간**
  - A5000 두 장을 다 쓰면: 과제 단위로 나누면(GPU A: PushT 20.3시간, GPU B: Wall + PointMaze 11.2시간) **약 20시간**. k 단위로 나누면(GPU A: PushT none·nonealt + k=3·4 + Wall 16.9시간, GPU B: PushT k=5 + PointMaze 14.6시간; PushT none은 lock으로 한 번만 돈다) **약 17시간**. 두 GPU의 계획이 비트 단위로 같으므로(§2) none을 다른 GPU의 k 칸과 짝지어도 수치는 같다
  - TD-MPC2나 다른 작업과 같은 GPU를 시분할하면 그 GPU의 시간이 최대 약 2.2배(무거운 MPS 작업과 겹칠 때의 실측)
- 설계 §6.9의 E1 DINO-WM 가정(6,300 ep, 200 ep/h, 약 31.5시간)과 단독 추정이 거의 같다

## 5. 비트 수 선택(held, 설계 §6.0)

### 5.1 규칙과 실행
- 규칙(설계 §6.0, "R2-13 규칙 그대로"): 과제마다 held 20 episode에서 det가 none 대비 20%p 이상 떨어뜨리는 k ∈ {3, 4, 5, 6} 중 가장 큰 k를 1차 칸, k−1과 k+1을 2차 칸으로 둔다
- held = pair seed 3의 쌍 0~19(planner seed 1000~1019, 파일럿 held와 같음). 범위는 서버 보정값(`ranges_main_<task>.json`). none과 det k=3~6을 같은 episode·seed로 돌림. 진단 포함
- PushT·Wall은 GPU 0(모든 k와 none이 GPU 0), PointMaze는 GPU 1(전부 GPU 1)

### 5.2 결과 (`results/E0_dinowm/ktable.csv`)

| 과제 | none | det k=3 | det k=4 | det k=5 | det k=6 | 규칙 적용 |
|---|---|---|---|---|---|---|
| PushT | 17/20 (85%) | 7/20, **−50%p** [−70, −30], p=0.002 | 9/20, **−40%p** [−65, −15], p=0.02 | 18/20, +5%p | 17/20, 0 | **k=4, 칸 {3,4,5}** |
| Wall | 11/20 (55%) | 0/20, **−55%p** [−75, −35], p=0.001 | 14/20, +15%p | 7/20, **−20%p** [−40, 0], p=0.22 | 10/20, −5%p | 문자 그대로 **k=5, 칸 {4,5,6}** |
| PointMaze | 17/20 (85%) | 14/20, −15%p [−30, 0], p=0.25 | 16/20, −5%p | 16/20, −5%p | 18/20, +5%p | **20%p에 이르는 k 없음** |

- 대괄호는 짝 bootstrap 95% CI(하락을 음수로 표기), p는 McNemar exact
- det의 계획 시점 상대 오차(1-step / 5-step): PushT k3 0.284 / 0.691, k4 0.056 / 0.275, k5 0.008 / 0.016; Wall k3 0.347 / 2.40, k4 0.040 / 0.138, k5 0.011 / 0.046; PointMaze k3 0.330 / 1.07, k4 0.045 / 0.307, k5 0.012 / 0.247
- 파일럿 held와 비교: PushT none 17/20·det4 8/20(이번 9/20), Wall none 11/20·det3 3/20(이번 0/20)·det4 15/20(이번 14/20). 서로 잘 맞는다
- 참고(규칙 밖): PointMaze det k=2는 12/20(−25%p, p=0.125)이지만 1-step 상대 오차가 2.4로 모델이 사실상 무너진 상태다

### 5.3 해석과 결정이 필요한 점
- **PushT**: 규칙대로 k=4, E1 칸 {3, 4, 5}
- **Wall**: 규칙을 문자 그대로 적용하면 k=5다. 그러나
  - k=5의 하락은 정확히 문턱(4/20)이고 유의하지 않다(McNemar 5/1, p=0.22). k=4는 오히려 +15%p, k=6은 −5%p로 k에 대해 단조가 아니다. 20 episode의 표집 잡음(하락 차의 표준오차 약 10~15%p) 안이다
  - 유의한 det 손상은 k=3뿐이다(0/20). 파일럿도 Wall k=3을 골랐다(파일럿 규칙은 오차 기준으로 k ≥ 5를 미리 뺐다)
  - 그대로 두면 Wall E1 칸 {4,5,6}에서는 det 손상이 작거나 없을 가능성이 크다. H3a의 "자격 칸"이 Wall에서 거의 나오지 않을 수 있다
- **PointMaze**: 20%p 하락하는 k가 {3,…,6}에 없다. 설계 §6.0에는 이때의 규칙이 없다. R2-13 PLAN §6의 대체 규칙(15%p 이상인 가장 큰 k, 없으면 시험한 가장 작은 k)을 쓰면 k=3, 칸 {2, 3, 4}다. PointMaze는 det 양자화에 강하다(k=3에서 1-step 오차 0.33인데도 −15%p)
- 위 둘은 가설·규칙을 건드리는 결정이므로 고치지 않고 조정자에게 넘긴다. manifest 초안은 PushT {3,4,5}, Wall {4,5,6}(규칙 그대로), PointMaze {2,3,4}(R2-13 대체 규칙)로 만들어 두었다. 바꾸려면 §7.4의 명령 한 줄로 다시 만든다

### 5.4 예측 오차(칸마다 한 번, `results/E0_dinowm/pred_errors.csv`, `err_<task>_main.json`)

비양자화 예측 대비 상대 오차(비용 단위 오차 ÷ 비양자화 예측의 분산). 데이터셋 입력 = valid 창 64개(1-step은 num_hist 창 다음 프레임, 3·5-step은 첫 프레임에서 데이터셋 행동으로 open-loop), 계획 시점 = held 시작 8개 × N(0,1) 후보 300개. 무작위 변형은 dither seed 3개 평균. 표는 E1 칸과 파일럿 k만 보인다(전체는 k = 2~7)

| 과제 k | 입력 | det 1 / 3 / 5-step | 확률적 4종(ind·sh-iter·sh-step·sh-ep) 1 / 5-step | det+shift 1 / 5-step |
|---|---|---|---|---|
| PushT 3 | 데이터셋 | 0.033 / 0.094 / 0.181 | 0.033~0.035 / 0.062~0.070 | 0.015 / 0.030 |
| PushT 4 | 데이터셋 | **0.0056** / 0.037 / 0.112 | **0.0065~0.0069** / 0.013~0.017 | 0.0034 / 0.008 |
| PushT 4 | 계획 시점 | 0.065 / 0.178 / 0.320 | 0.066~0.067 / 0.038 | 0.033 / 0.020 |
| PushT 5 | 데이터셋 | 0.0009 / 0.003 / 0.008 | 0.0016 / 0.004~0.006 | 0.0009 / 0.003 |
| Wall 3 | 데이터셋 | 0.056 / 0.287 / 1.19 | 0.031~0.033 / 0.037~0.049 | 0.014 / 0.017 |
| Wall 4 | 데이터셋 | 0.0094 / 0.026 / 0.038 | 0.0059~0.0063 / 0.007~0.008 | 0.0030 / 0.004 |
| Wall 5 | 데이터셋 | 0.0020 / 0.007 / 0.013 | 0.0015~0.0016 / 0.0018~0.0019 | 0.0008 / 0.001 |
| Wall 6 | 데이터셋 | 0.0004 / 0.0007 / 0.0011 | 0.0005 / 0.0006~0.0007 | 0.0003 / 0.0005 |
| PointMaze 2 | 데이터셋 | 0.70 / 1.34 / 1.78 | 0.69~0.70 / 1.24~1.27 | 0.33 / 0.84 |
| PointMaze 3 | 데이터셋 | 0.102 / 0.705 / 1.00 | 0.109~0.110 / 0.38~0.46 | 0.054 / 0.28 |
| PointMaze 4 | 데이터셋 | 0.0125 / 0.079 / 0.280 | 0.023 / 0.13~0.16 | 0.012 / 0.11 |

- 파일럿과 같다: PushT k=4 데이터셋 1-step det 0.0056 대 확률적 0.0066~0.0070, 계획 시점 det 0.0652·5-step 0.320; Wall k=3 det 0.0565·5-step 1.136, 확률적 0.031·0.037~0.041
- 같은 비트 수에서 det의 데이터셋 1-step 오차는 PushT(k ≥ 3)와 PointMaze(k ≥ 3)에서 확률적 변형보다 작거나 같고, Wall에서는 1.3~1.8배 크다. 5-step에서 det만 크게 벌어지는 파일럿의 모양은 세 과제 모두에서 나온다(PushT k=4 7~9배, Wall k=3 24~32배, PointMaze k=3 2.2~2.6배). det+shift는 MSE가 Δ²/12라 확률적 변형의 약 절반이다

## 6. PointMaze 상태: 사용 가능(제외하지 않음)

- TD-MPC2 작업자의 mujoco-py 빌드(13:33Z 성공)를 확인하고 같은 방식으로 별도 venv에 설치했다. MuJoCo 2.1 바이너리(`~/pmx/envs/mujoco210`)와 conda-forge mesalib 24.3.4(OSMesa, `~/pmx/envs/micromamba/envs/gl24`)는 읽기 전용으로 공유한다
- 문제와 해결: mujoco-py의 첫 import 빌드가 `patchelf`를 못 찾음 → venv bin을 PATH에 추가. env의 빠른 재생이 env.rollout과 9e-8 다름 → env가 float32 상태를 내므로 같게 맞춰 비트 단위 일치
- 동작 확인: OSMesa 렌더링(224×224), 데이터셋 2,000 궤적(train 1,800 / valid 200), 시작·목표 표집(random_state), 범위 보정(창 렌더링 10초), 7변형 스모크, held k 선택 100 run + k=2 참고 20 run
- 비용 함수는 계획 설정 그대로 α=0(시각 MSE만), opt_steps 10, 성공 = 목표와 xy 거리 < 0.5
- 데이터셋 다운로드: OSF `point_maze.zip`(718MB, `~/pmx/data/dinowm/pm_dl/`)에서 `states.pth`, `actions.pth`, `seq_lengths.pth`만 풀었다(obs 30GB 제외). zip은 지워도 된다

## 7. E1 runner (E2·E3·E5에서도 재사용)

### 7.1 구성 파일 (`main/code/dinowm/`)

| 파일 | 역할 |
|---|---|
| `wmlib.py` | DINO-WM 적재, 시작·목표 쌍(DINO-WM prepare_targets), 시뮬레이터 빠른 재생과 평가기(파일럿 r21lib의 서버판 + PointMaze 분기) |
| `qlib.py` | 양자화 hook(7변형 + 파일럿 sh-fixed), 배치 CEM(R2-13과 같은 계산) + 진단, 데이터셋 창, 분할 정의, 쌍 캐시 |
| `runner.py` | manifest 기반 실행기. job마다 JSON 1개(`results/<name>/<task>/<id>.json`), 있으면 건너뜀, job별 lock(죽은 pid의 lock은 인수), 재생 pool |
| `make_manifest.py` | manifest 생성(`repro`, `ksel`, `smoke`, `e1`, `custom`) |
| `calib.py` | 범위 보정, 데이터셋 입력·계획 시점 입력의 k·변형별 1/3/5-step 오차 |
| `analyze.py` | 칸 통계(`cells`), 재현 비교(`repro`), k 선택(`ksel`), 처리량(`throughput`), CSV |
| `status.py` | manifest 진행(끝난 job 수) |
| `run.sh`, `run_pm.sh` | 실행기. `DINOWM_GPU`를 **반드시** 명시해야 하고 `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DINOWM_GPU`로 고정 |
| `run_queue.sh`, `E1_queue.sh`, `E0_queue*.sh` | 순차 대기열(오류 시 5분 뒤 재시도, 끝난 job은 건너뜀) |
| `setup_server.sh`, `setup_pointmaze.sh` | 사용자 공간 설치 |
| `test_quant.py`, `test_pointmaze.py`, `pairs_info.py`, `gpu_occupancy.sh` | 커널 성질 점검, PointMaze 점검, 분할 기록, GPU 사용 기록 |

### 7.2 변형 (설계 §4.2; 같은 층별 k-bit 격자 [lo, hi], Δ = (hi − lo)/(2^k − 1), y = (clip(x) − lo)/Δ)

| 코드 이름 | 정의 | 파일럿과의 관계 |
|---|---|---|
| none | 양자화 없음 | |
| det | round(y) | R2-13 det와 같음 |
| ind | floor(y + u), u는 항목(후보 × token × unit)마다 독립, forward마다 새로 | R2-13 ind와 같음 |
| shiter | floor(y + u), u는 (token × unit) 표 하나를 한 episode의 후보 300개가 공유, **예측기 forward 호출마다 새로**(CEM 반복 × horizon step) | R2-13 sh-iter와 같은 seed 공식 |
| shstep | 층·horizon step마다 (token × unit) 표를 계획(제어 step) 시작에 뽑아 그 계획의 모든 CEM 반복이 공유 | R2-13 sh-fixed와 같은 구조(seed만 다름) |
| shep | 같은 표를 episode 시작에 한 번 | |
| detshift | 결정적 반올림을 이동한 격자에서: clamp(round(y − o), 0, 2^k−1) + o, o ~ U[−½, ½)는 sh-iter의 u와 같은 표 모양·일정(token × unit, 후보 공유, forward마다 새로). 준위 수 2^k | TD-MPC2 판(unit별 o)과 같은 구성 |
| (shfixed) | R2-13 sh-fixed의 seed 공식 그대로(재현 전용) | |

- `test_quant.py`로 확인(compile판 = eager판): det MSE Δ²/12와 지속 편향, 확률적 4종 MSE Δ²/6, sh-step·sh-ep·sh-fixed는 계획 안에서 고정, det+shift는 MSE Δ²/12·무편향·forward마다 새로·같은 입력 같은 출력
- dither 난수는 seed_of(salt, 변형, episode, …)로 만든 CUDA generator(ind는 begin_plan에서 기본 CUDA generator 재설정)에서 나오고, 계획기의 CPU 난수열은 모든 변형에서 같다(짝 비교). iteration-0 모집단이 변형 사이에 같음을 assert한다

### 7.3 episode별 기록(진단은 계획 난수를 바꾸지 않는다)

- `success`, `state_dist`(평가기 거리), `margin`(PushT max(‖Δpos‖/20, Δθ/(π/9)), Wall ‖Δxy‖/4.5, PointMaze ‖Δxy‖/0.5), `mu_final`(계획 평균 5 × 10), iteration별 최저 비용, 묶음 시작·끝 시각, GPU
- **계획 시점 1/3/5-step 오차**(`err`): 같은 episode의 iteration-0 후보 300개에서 비양자화 예측 대비 비용 단위 오차(시각 MSE + α·proprio MSE)의 후보 평균과, 비양자화 예측의 후보 간 분산으로 나눈 상대값. 무작위 변형은 계획에 쓴 draw + 추가 draw 2개 = **dither seed 3개 평균**
- **계획 편차 입력**: `mu_final`. 분석에서 같은 seed의 none과의 첫 행동 RMS를 nonealt(다른 planner seed의 none)로 잰 seed 바닥으로 나눈다(R2-8b §4 정의의 open-loop판)
- **OEI 입력**(`oei` 요약 + 서버 `pops/`의 배열): iteration-0 모집단(모든 변형에서 같음)과 마지막 모집단의 J_v(변형 비용)와 J_none(비양자화 비용), 최종 계획 평균의 J_v·J_none, 그리고 **두 모집단 후보 300개 전부의 시뮬레이터 참 비용**(margin, state_dist, success; CPU 재생)
  - 요약값: `oei_ref_argmin`·`_elite`·`_mf` = (고른 후보의 J_v − J_none) − (iteration-0 무작위 후보 299개의 평균 J_v − J_none). R2-13 사후 진단과 같은 정의다
  - `oei_true_*`(잠정): 참 margin을 같은 episode의 iteration-0 모집단에서 J_none에 선형 맞춤해 비용 단위로 바꾼 오차. none에서도 0이 아니어서(선형 보정의 한계) 정의 결정이 필요하다(§10)
  - 순위 지표: 마지막 모집단의 Spearman(J_v, 참 margin)과 같은 모집단의 Spearman(J_none, 참 margin)(R2-13과 같음), iteration-0의 같은 값, 고정 후보 집합의 Spearman(J_v, J_none). argmax flip과 gap 사분위별 flip은 저장된 배열로 E2에서 계산할 수 있다
- 데이터셋 입력 1/3/5-step 오차는 칸마다 한 번 `calib.py`로(valid 창 64개, dither seed 3개)

### 7.4 사용법 (정확한 명령)

```bash
A="/Users/song-yonghwi/Documents/physical ai - Neural Networks/A_dithered_world_models"
# (1) manifest (맥북 또는 서버). k는 조정자가 DESIGN_main.md에서 고정한 값으로
python3 "$A/main/code/dinowm/make_manifest.py" e1 --tasks pusht,wall,pointmaze \
  --k pusht=3,4,5 --k wall=4,5,6 --k pointmaze=2,3,4 --E pusht=1,wall=2,pointmaze=1 --chunk pusht=10,wall=20,pointmaze=10
# (2) 코드 동기화
rsync -a --exclude __pycache__ "$A/main/code/dinowm/" kiis-mvf-gpu:pmx/A/dinowm_main/code/
# (3) 시작: GPU는 조정자가 정한 번호를 명시(모든 칸이 그 GPU에서 돈다)
ssh kiis-mvf-gpu 'tmux new-session -d -s A-E1-dinowm "DINOWM_GPU=1 bash ~/pmx/A/dinowm_main/code/E1_queue.sh >> ~/pmx/A/logs/E1_dinowm.log 2>&1"'
#     두 GPU(과제 단위): 세션 2개, 예) "DINOWM_GPU=1 TASKS=pusht bash ..." 와 "DINOWM_GPU=0 TASKS='wall pointmaze' bash ..."
#     두 GPU(k 단위, KS는 한 번 부를 때 모든 과제에 같이 걸리므로 과제별로 나눠 부른다): 세션 2개, 예)
#       GPU 1: "DINOWM_GPU=1 TASKS=pusht KS=3,4 bash E1_queue.sh; DINOWM_GPU=1 TASKS=wall bash E1_queue.sh"
#       GPU 0: "DINOWM_GPU=0 TASKS=pusht KS=5 bash E1_queue.sh; DINOWM_GPU=0 TASKS=pointmaze bash E1_queue.sh"
#       (none/nonealt는 양쪽 runner에 포함되고 job lock으로 한 번만 돈다. 로그 파일·세션 이름은 따로)
# (4) 진행
ssh kiis-mvf-gpu 'python3 ~/pmx/A/dinowm_main/code/status.py ~/pmx/A/dinowm_main/code/manifests/E1_dinowm.json'
# (5) 멈추기/이어 돌리기: tmux kill-session -t A-E1-dinowm → 같은 시작 명령(끝난 job은 건너뜀, 죽은 job의 lock은 인수)
# (6) 결과 회수와 분석(맥북)
rsync -a --exclude '*.lock' kiis-mvf-gpu:pmx/A/dinowm_main/results/E1_dinowm "$A/main/results/E1/"
python3 "$A/main/code/dinowm/analyze.py" cells "$A/main/results/E1/E1_dinowm" --out "$A/main/results/E1/dinowm_cells.json" --csv "$A/main/results/E1/dinowm_cells.csv"
```

- E2·E3·E5 재사용: `make_manifest.py custom --name <N> --split eval --vars ... --k task=...`로 칸을 만든다
  - 후보 수·elite 수·CEM 반복 수(E2): job의 `planner`(num_samples, topk, opt_steps)로 job마다 바꾼다. 예) `--planner num_samples=64,topk=8,opt_steps=30 --ptag N64`(job id에 붙음). 기본값을 쓰면 E1과 계산이 같다. smoke 1 episode로 확인(반복 3회, hook 호출 120 = 3 × 5 × 6 + 30)
  - 새 변형(E3 표 크기 sweep 등)은 `qlib.Quantizer._hook`과 `_table_seed`/`_shared_u`에, 새 범위 규칙(E5)은 `ranges` id(`results/ranges/ranges_<id>_<task>.json`)로 넣는다
- 두 GPU로 나눠 돌리기: runner `--ks`(양자화 칸의 k만), `E1_queue.sh`의 `TASKS`·`KS`(§7.4 예). none/nonealt는 job lock으로 한 번만 돈다
- 디스크: E1 6,000 run의 모집단 배열은 약 0.4GB(서버 `pops/`, episode-run당 약 65KB), job JSON은 약 25MB(episode-run당 약 4KB)

## 8. 스모크와 구현 점검

- 진단 불변성: iteration-0 모집단 미리보기는 복제한 generator로 만들고, 추가 dither draw 뒤 begin_plan이 난수를 다시 설정한다. none·det·shared 변형의 결과가 진단과 무관하다(E·GPU 교차 비트 일치로 확인)
- hook 호출 수: PushT det 930 = 30 반복 × 5 step × 6층 + 최종 계획 평가 30, 무작위 990(+오차 draw 60). Wall·PointMaze det 330, 무작위 390
- 정의상 값: none의 OEI = 0, 계획 시점 오차 없음. nonealt의 iteration-0 모집단은 none과 다르다(seed가 달라서)
- 스모크·벤치 결과는 smoke 분할(pair seed 7, R2-1 스모크 쌍)에서만 냈고 해석하지 않는다

## 9. 분할·seed·episode 범위 (`results/E0_dinowm/splits_<task>.json`)

| 분할 | 시작·목표 쌍 | episode 번호 | env seed | planner seed | dither salt | 용도 |
|---|---|---|---|---|---|---|
| held | pair seed 3 | 0~19 | 3001~3020 | 1000~1019 | `held` | 범위 확인, k 선택(파일럿 held와 같음) |
| repro | pair seed 4 | 0~49 | 4001~4050 | 2000~2049 | `eval`(파일럿과 같음) | R2-13 재현 전용 |
| **eval** | **pair seed 5** | **0~99** | **5001~5100** | **3000~3099**, nonealt **3500~3599** | `eval5` | **E1 판정 전용. E0에서는 계획을 돌리지 않음** |
| smoke | pair seed 7 | 0~3 | 7001~7004 | 9000~9003 | `smoke` | 스모크·벤치 |

- 범위 보정은 train 분할 창(창 seed 11), 데이터셋 예측 오차는 valid 분할 창(창 seed 12)이다. 둘 다 시작·목표 쌍과 독립이다
- 파일럿이 쓴 pair seed는 1, 2, 3, 4, 7, 11이다. eval의 5는 새 seed다
- **PushT 쌍 겹침(구조적)**: PushT 쌍은 val 궤적 21개에서 자른 26프레임 구간이다. 그래서 분할 사이에 구간이 겹친다: held–eval 겹침 42쌍(eval 쌍 31개 해당), 파일럿 eval(repro)–E1 eval 145쌍(eval 쌍 70개 해당, 시작 프레임까지 같은 쌍 2개). Wall·PointMaze 쌍은 seed별 무작위 상태라 겹치지 않는다

## 10. 편차와 결정이 필요한 것

### 10.1 결정 필요(조정자)
1. **Wall의 k**: 규칙 문자 그대로는 k=5(칸 {4,5,6}). 근거와 대안은 §5.3. 규칙을 바꾸는 결정이라 내가 정하지 않았다
2. **PointMaze의 k**: 설계 규칙으로는 정해지지 않는다. R2-13 대체 규칙을 쓰면 k=3(칸 {2,3,4})
3. **sh-step과 sh-ep가 open-loop DINO-WM에서 같은 분포다**: episode당 계획이 한 번이므로 "계획 고정"과 "episode 고정"이 같다. 지금은 둘 다 정의대로 구현했고 seed만 다르다(독립 반복으로 쓸 수 있다). 하나를 빼면 E1이 900 run(약 4.9 GPU-시간) 준다. E3의 2축 대조에서도 DINO-WM의 이 두 칸은 같은 조건이다
4. **DINO-WM OEI의 참 비용 기준**(설계 §4.3: "DINO-WM은 시뮬레이터 재생으로 얻은 참 비용"): 모델 비용(latent MSE)과 참 비용(상태 거리)은 단위가 달라 변환이 필요하다. 선형 보정판(`oei_true_*`)은 none에서도 0이 아니다. 파일럿 값(+0.30 / −0.09)은 비양자화 모델 비용 기준(`oei_ref_*`)이다. 두 기준의 입력을 모두 저장했으므로 E2 전에 정의만 고르면 된다
5. **E1 GPU 배정**: GPU 1 단독이면 약 32시간, 두 GPU를 쓰면 약 17~20시간, 다른 무거운 작업과 같은 GPU를 시분할하면 그 GPU의 시간이 최대 약 2.2배(§4.3). 두 GPU의 결과가 비트 단위로 같아 칸을 GPU별로 나눠도 수치는 같다(§2)
6. **PushT 분할 겹침**(§9): E1 eval 쌍 100개 중 70개가 파일럿 eval 구간과, 31개가 held 구간과 프레임을 나눠 갖는다(DINO-WM PushT 평가 방식상 val 궤적 21개에서 자르기 때문). 그대로 둘지 기록만 할지 결정 필요

### 10.2 설계에 없던 구현 선택(DESIGN_main.md에 고정할 항목)
1. **det+shift**: §7.2. 설계 문구("반복마다 바뀌는 공유 offset")를 TD-MPC2 작업자 구현과 같은 구성으로 정했다. 처음에는 층마다 스칼라 offset 하나로 구현했다가(Wall k=3에서 한 draw의 5-step 오차가 43배로 튐) 스모크 단계에서 바꿨다. 그 결과는 어디에도 쓰지 않았다
2. **none은 과제당 한 번**(k와 무관), **nonealt**(planner seed + 500)를 계획 편차의 seed 바닥으로 과제당 한 번 더 돈다
3. **계획 편차**: open-loop라 "같은 상태 재계획"이 곧 episode의 계획이다. 최종 계획 평균의 첫 행동(정규화 행동 10차원) RMS를 nonealt 바닥으로 나눈다
4. **계획 시점 오차**: 같은 episode의 iteration-0 후보 300개(첫 행 = mu = 0 포함)
5. **범위**: E1은 서버 보정값(`main`), 재현만 파일럿 값. PushT는 두 값이 같고 Wall은 2층에서 bf16 한 칸 다르다
6. **E**: PushT 1, Wall 2, PointMaze 1(수치와 무관, 메모리 기준)

### 10.3 그 밖의 편차와 운영 기록
- 처리량 측정 창: 13:48~14:17Z, 14:40~14:42Z(B 부하)에 걸친 시간은 쓰지 않았다. 단독 수치는 GPU 1 벤치, 공유 수치는 그 밖의 GPU 0 job에서 냈다
- GPU 0에서 OOM 1회(13:52Z, TD-MPC2 14.7GB + 내 PushT E=2): E를 줄여 해결. 그 job은 결과 없이 다시 돌림
- uv venv에 setuptools가 없어 torch.compile(triton)이 처음에 실패: setuptools 설치로 해결
- 재현의 PushT·Wall은 GPU 1, k 선택의 PushT·Wall은 GPU 0이다(칸마다 한 GPU)

## 11. 비용과 시간

- RunPod $0
- 서버 GPU 사용: GPU 0 13:24~16:29Z(약 3시간, 13:34Z부터 TD-MPC2 MPS와 시분할), GPU 1 15:53~19:56Z(약 4시간, 단독. 16:34~16:44Z만 내 참고 실행 2개가 겹침). GPU 0에는 16:29Z 뒤로 아무것도 돌리지 않았다(조정자 지시: TD-MPC2 E1 독점)
- 소요: 13:08Z 지시 → 19:58Z 결과 회수(약 7시간). 대부분 GPU 대기(PushT 재현 2.45시간, PushT k 선택 2.1시간)
- 디스크: `~/pmx/A/dinowm_main` 111MB, venv 2개(uv 캐시 hardlink), PointMaze 메타 9MB와 zip 0.7GB(`~/pmx/data/dinowm/pm_dl/`, 풀어 둔 뒤라 지워도 됨). 여유 디스크 293GB
- **아직 도는 작업: 없음**(내 tmux 세션·GPU 프로세스 없음. E1은 시작하지 않았다)
