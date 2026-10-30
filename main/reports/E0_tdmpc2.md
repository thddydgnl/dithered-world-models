# E0 TD-MPC2 단계 보고 (A 논문: dithered low-bit world model)

- 작성: 2026-10-21, E0 TD-MPC2 작업자
- 기준 문서: [../../EXPERIMENT_DESIGN.md](../../EXPERIMENT_DESIGN.md) §4.2·4.3·6.0~6.3·6.9·6.10, [../../../MAIN_RULES.md](../../workspace/MAIN_RULES.md)
- 작업 로그: [../worklogs/E0_tdmpc2.md](../worklogs/E0_tdmpc2.md) · 코드: [../code/tdmpc2/](../code/tdmpc2/) · 결과: [../results/E0_tdmpc2/](../results/E0_tdmpc2/)
- 장소: 대학 서버 `kiis-mvf-gpu`, **GPU 0만**(RTX A5000 24GB, 드라이버 560.35.05, CUDA 12.6)
  - GPU 0은 A의 DINO-WM 작업자와 같이 썼다. 13:48~14:17Z, 14:40~14:42Z에는 B의 작은 job도 잘못 올라가 있었다(조정자 알림). 수치에는 영향이 없고, 처리량은 그 창 밖에서 따로 쟀다(§5)
  - RunPod는 쓰지 않았다(비용 $0). 지금 돌고 있는 내 job은 없다. MPS 데몬도 껐다

## 0. 요약

| 항목 | 결과 |
|---|---|
| DMC 환경 | uv venv, 파일럿과 같은 패키지 버전(71개 전부). `MUJOCO_GL=disable`로 동작 |
| **mujoco-py** | **성공**(2차 시도). conda-forge 최신 Mesa에는 OSMesa가 없어 1차 실패 → `mesalib=24.3.4`로 사용자 공간 빌드. Meta-World episode가 RunPod A5000 파일럿과 같은 return(mw-push seed 1000: 1770.13)을 낸다. **RunPod 대체 경로 불필요** |
| 재현(R2-8b, 40 ep) | det 하락 mw-push 62.5%p(파일럿 65), pick-place 57.5%p(55), cheetah 53.1%(52.4). sh-iter 손상 제거 96%(92), 61%(82, CI 겹침), 20%(10). 공유 범위 순서 sh-iter < sh-step ≤ sh-ep 재현 |
| 처리량·E1 추정 | MPS 워커 8개에서 포화, MPS가 없을 때보다 1.6~1.8배 빠름. E1 계측 포함 DMC 860 ep/h, MW 3,716 ep/h → **E1 TD-MPC2 약 12 GPU-시간**(MW 13,680 + DMC 6,840 episode; DINO-WM이 GPU 0을 무겁게 같이 쓰면 약 24시간) |
| 과제·k 선택 | MW 8과제 모두 자격(기준 ≥ 98%). 1차 k: push 3, pick-place 3, door-open 4, drawer-open 3(대체 규칙), button-press 3, peg-insert 4, window-open 4, sweep-into 3(문턱에 걸림), cheetah 5, finger 4, quadruped 3, walker 3 |
| E1 runner | 7변형, 계측(OEI·결정 지표·계획 편차·1/3/5-step 오차·오차 지속도) 포함, 재개 가능. 스모크와 비트 일치 점검 통과. **E1은 돌리지 않았다** |
| 결정 필요 | held 크기 해석, 1차 k = 3일 때 k = 2 칸(MW에서는 처방 효과가 가장 큰 칸, DMC에서는 둘 다 붕괴), drawer-open의 대체 규칙(§10.2) |

## 1. 무엇을 돌렸나

| 단계 | 내용 | 규모 | 시간(UTC) |
|---|---|---|---|
| 설치 | `setup_server.sh`(DMC venv, tdmpc2, 체크포인트 36개), `setup_mw_server.sh`(Meta-World venv, mujoco-py) | | 13:14~13:33 |
| 구현 점검 | `test_equiv.py`: 파일럿 nt_lib 대 qlib 비트 일치, 계측 불변성(수정 뒤 재확인), `test_cache.py` | 66 episode(smoke seed) | 13:35~15:02 |
| 재현 | R2-8b 칸 3개, 체크포인트 1, R2-8 범위, seed 2000~2039 | 600 episode | 13:38~14:29 |
| 보정 | 12과제 × 체크포인트 3 × 10 episode(비양자화) + 층별 범위 | 360 episode | 13:43~14:29 |
| 스모크 | episode job 13개(7변형, E2용 override 경로, E0 범위의 E1 모양 job), prederr·decision job | 16 episode + 3 job | 14:08~15:21 |
| held 선택 | none + det k=3~6, MW 20 / DMC 10 episode × 체크포인트 3 | 3,000 episode | 14:29~16:38 |
| k=2 보조 | 1차 k = 3인 7과제의 det·sh-iter k=2(정보용) | 720 episode | 16:38~16:53 |
| 처리량 | `run_bench.sh` 6설정(고정 시간) | 약 1,900 episode | 16:53~18:08 |
| 칸별 job 시간 | E1의 prederr·decision job 4개(별도 폴더, 지운 뒤 E1에는 남기지 않음) | 4 job | 18:08~18:09 |

## 2. 환경 (정확한 버전)

- **Python**: uv 관리 CPython 3.11.17(`~/pmx/envs/python`). `~/.local/bin/python3.11`은 다른 폴더(DriveWAM)를 가리키는 링크라 쓰지 않았다
- **venv `~/pmx/envs/a-mw`(모든 실행에 사용)**: torch 2.7.1+cu126, tensordict 0.8.3, dm-control 1.0.16, mujoco 3.1.2, numpy 1.24.4, gymnasium 0.29.1, hydra-core 1.3.2, omegaconf 2.3.0, scipy 1.15.3, pandas 2.3.3, matplotlib 3.10.9, glfw 2.11.0 + Meta-World 쪽 gym 0.21.0, Metaworld@04be337a, mujoco-py 2.1.2.14, cython 0.29.37, setuptools 65.5.0, wheel 0.38.0, pip 23.3.1, patchelf 0.19.1.0, cffi 2.1.1, fasteners 0.20, imageio 2.38.1. 전체 목록은 서버 `~/pmx/A/tdmpc2_main/freeze_a-mw.txt`
- **venv `~/pmx/envs/a-tdmpc2`(DMC 전용, 파일럿 `setup_pod.sh`와 같은 71개 패키지)**: a-mw와 DMC episode가 비트 단위로 같아서(cheetah seed 2000 return 877.2399995140731) 실행은 a-mw 하나로 통일했다
- **tdmpc2**: commit `332f19e4aba4bffe2dc736f45ea7f7bdc585903a`(파일럿 R2-8b와 같음), `~/pmx/A/tdmpc2_main/tdmpc2`
- **체크포인트**: HF `nicklashansen/tdmpc2`의 단일 과제 5M, 12과제 × seed 1·2·3 = 36개(각 31MB), `~/pmx/A/tdmpc2_main/ckpt/<task>-<seed>.pt`, sha256은 같은 폴더 `SHA256SUMS`
- **MuJoCo 2.1.0 바이너리**: `~/pmx/envs/mujoco210`(tarball sha256 a436ca2f…d012), `MUJOCO_PY_MUJOCO_PATH`
- **micromamba 2.9.0**: `~/pmx/envs/micromamba`, env `gl24` = conda-forge mesalib 24.3.4, libllvm19 19.1.7, libglvnd/libgl/libgl-devel 1.7.0, glew 2.3.0
- **실행 환경 변수**(`code/tdmpc2/env.sh`): `CUDA_VISIBLE_DEVICES=0`, `MUJOCO_GL=disable`, `OMP_NUM_THREADS=1`, `LD_LIBRARY_PATH=mujoco210/bin:gl24/lib`, `CUDA_MPS_PIPE_DIRECTORY=~/pmx/A/mps/pipe`
- **MPS**: 자체 데몬(`code/tdmpc2/mps.sh start|stop|status`), pipe·log `~/pmx/A/mps/`, GPU 0만. MPS를 써도 episode 수치는 같다(MPS 없이 돈 test_equiv와 MPS로 돈 스모크의 return이 비트 단위로 같음)
- 디스크: venv a-mw 5.8GB(uv 캐시 hardlink), micromamba 0.45GB, 체크포인트 1.1GB, E0 산출물 1.3GB. 남은 공간 293GB(60GB 문턱보다 훨씬 큼)

## 3. mujoco-py 점검 결과: **성공**

1. **MuJoCo 2.1 바이너리**: GitHub release 리다이렉트가 멈춰서 새 주소(google-deepmind)로 직접 받았다. `libmujoco210.so`와 `libglewosmesa.so`는 시스템 libc/libstdc++만 필요하다
2. **1차 시도(실패)**: micromamba로 conda-forge `mesalib glew`(최신 26.2.5)를 깔았다. **Mesa 25.1부터 OSMesa가 빠져** `libOSMesa`와 `GL/osmesa.h`가 없다. mujoco-py의 CPU 빌더(`osmesashim.c`)가 컴파일에서 멈췄다(`~/pmx/A/logs/E0_tdmpc2_mwsetup_try1.log`)
3. **2차 시도(성공)**: env `gl24` = `mesalib=24.3.4`(OSMesa가 있는 마지막 24.x) + `glew` + `libgl-devel`(GL/gl.h, libGL.so). 빌드 때 `CPATH=gl24/include`, `LIBRARY_PATH=gl24/lib`, 실행 때 `LD_LIBRARY_PATH=mujoco210/bin:gl24/lib`. `import mujoco_py`가 cymj 확장(CPU/OSMesa 빌더, `-lmujoco210 -lglewosmesa -lOSMesa -lGL`)을 빌드하고 동작한다. 렌더링은 쓰지 않는다
4. **끝까지 확인**: TD-MPC2 체크포인트를 실은 Meta-World episode
   - mw-push seed 1000(비양자화): return **1770.13**, 성공. R2-14 RunPod A5000의 1770.1과 같다(3090은 1769.3)
   - mw-pick-place seed 2000: 1712.35, 성공. 보정 360 episode, 재현 480 MW episode, held 실행이 모두 이 환경에서 돌았다
- 결론: **Meta-World 칸을 RunPod로 옮길 필요가 없다.** 사전 계획된 대체 경로(비용)는 쓰지 않는다. DINO-WM PointMaze도 같은 mujoco-py 빌드(`a-mw`의 방법)를 쓸 수 있을 것이다(확인은 DINO-WM 작업자 몫)

## 4. 재현: R2-8b 주요 칸

- 설정: 체크포인트 seed 1, **R2-8 범위 그대로**(파일럿과 같은 k-bit 격자), MPPI 기본값, 계측 없음, MPS 워커 16
- episode: seed 2000~2039(앞 20개는 파일럿 seed와 같다), 칸·변형마다 40 episode, 합계 600 episode(MW 480, DMC 120)
- GPU가 다르면(3090 대 A5000) 같은 seed라도 episode가 비트 단위로 같지 않다. 그래서 비교는 분포 수준이다
- 하락: MW는 성공률 차(%p), DMC는 return 상대 하락(%). 짝 bootstrap 10,000회

| 칸 | 변형 | 하락, seed 2000~2019 (90% CI) | 하락, seed 2000~2039 (90% CI) | 파일럿 R2-8b(3090, 20 ep) |
|---|---|---|---|---|
| mw-push k=3 | det | 60.0 [40, 75] | **62.5** [50, 75] | 65 [45, 80] |
| | ind | 40.0 [20, 60] | 37.5 [25, 50] | 15 [5, 30] |
| | **sh-iter** | 5.0 [0, 15] | **2.5** [0, 7.5] | 5 [0, 15] |
| | sh-step | 50.0 [30, 70] | 55.0 [42.5, 67.5] | 65 [45, 80] |
| | sh-ep | 70.0 [55, 85] | 75.0 [62.5, 85] | 85 [70, 95] |
| | **감소율 red(sh-iter)** | 92% [77, 100] | **96%** [88, 100] | 92% [75, 100] |
| mw-pick-place k=3 | det | 60.0 [40, 80] | **57.5** [45, 70] | 55 [35, 75] |
| | ind | 35.0 [20, 55] | 42.5 [30, 55] | 35 [20, 50] |
| | **sh-iter** | 25.0 [10, 40] | **22.5** [12.5, 32.5] | 10 [0, 20] |
| | sh-step | 40.0 [25, 60] | 42.5 [30, 55] | 40 [20, 60] |
| | sh-ep | 55.0 [35, 75] | 55.0 [42.5, 67.5] | 75 [60, 90] |
| | **red(sh-iter)** | 58% [27, 83] | **61%** [38, 79] | 82% [56, 100] |
| cheetah-run k=4 | det | 53.6 [50.2, 57.2] | **53.1** [50.4, 56.0] | 52.4 [48.6, 56.5] |
| | **sh-iter** | 42.4 [36.1, 49.2] | **42.6** [38.2, 47.3] | 47.1 [40.7, 54.0] |
| | **red(sh-iter)** | 21% [7, 33] | **20%** [11, 28] | 10% [−6, 24] |

- 기준선(같은 seed, 이 서버): mw-push 40/40 성공(return 1780), pick-place 40/40(1690), cheetah return 857.8. 파일럿은 20/20(1786), 20/20(1682), 864.6
- MW return 하락(보조, 40 ep): mw-push det 55.1%, sh-iter 2.5%. pick-place det 38.8%, sh-iter 18.8%
- **판정(재현 일치)**
  - **det 하락은 세 칸 모두 파일럿과 거의 같다**(62.5 대 65%p, 57.5 대 55%p, 53.1 대 52.4%).
  - **sh-iter의 손상 제거도 재현됐다**: mw-push 96%(파일럿 92%), cheetah 20%(10%). pick-place는 61% [38, 79]로 파일럿 82% [56, 100]보다 작지만 CI가 겹친다. 세 칸 모두 sh-iter의 하락이 det보다 작다(CI가 겹치지 않음)
  - 공유 범위의 순서 sh-iter < sh-step ≤ sh-ep가 MW 두 과제에서 그대로 나왔다
  - 다른 점: mw-push의 ind(37.5 대 15%p). R2-8(50%p)과 R2-8b(15%p) 사이에서도 크게 흔들린 값이라 seed·GPU 잡음 범위로 본다
- 결과 파일: `results/E0_tdmpc2/repro_summary.json`, `repro_table.md`, job 결과 `results/E0_tdmpc2/jobs/episodes/repro_*.json.gz`
- 소요 시간 주의: 이 실행은 B의 job이 GPU 0에 잘못 올라간 창(13:48~14:17Z)과 겹친다. 수치에는 영향이 없고, 시간은 처리량 근거로 쓰지 않는다

## 5. 처리량과 E1 GPU 시간 추정

### 5.1 측정 방법
- `run_bench.sh`(16:53~18:08Z, B 오염 창 이후): E1과 같은 계측을 켠 job 묶음을 정해진 시간 동안 돌리고 완료 시각에서 속도를 잰다
  - DMC: cheetah-run k=4 det·sh-iter, job당 1 episode, 420초
  - MW: mw-push·pick-place k=3 det·sh-iter, job당 2 episode, 300초
  - 설정: MPS + 워커 8·16·24, 계측 끔(MPS 16), MPS 없이 16, 그리고 최종 E1 계측 밀도(MPS 16, `mps16b`)
- 같은 시간대 GPU 0의 DINO-WM 작업자는 주로 CPU 단계(재생)였다. GPU 0 메모리 평균 5.7GB(대부분 내 워커), 사용률 평균 96%. 그래서 이 값은 **"DINO-WM이 GPU를 거의 쓰지 않을 때"**의 속도다
- 비교 기준(경합 상태): held 실행(14:42~16:38Z, 계측 없음, MPS 16)에서는 DINO-WM이 GPU 0을 6~13GB, 100%로 쓰고 있었고 DMC 127초/ep/워커(약 450 ep/h), MW 25초/ep/워커(약 2,300 ep/h)였다. 같은 조건의 bench(계측 없음, 929·4,800 ep/h)의 약 절반이다

### 5.2 결과 (episode/시간; 괄호는 워커당 episode 소요 시간)

| 설정 | DMC(cheetah, 500 step) | MW(push·pick-place, 100 step) |
|---|---|---|
| MPS, 워커 8, 계측(초기 밀도) | 778 (37초) | 3,200 (9초) |
| MPS, 워커 16, 계측(초기 밀도) | 811 (71초) | 3,200 (18초) |
| MPS, 워커 24, 계측(초기 밀도) | 807 (107초) | 3,200 (27초) |
| **MPS, 워커 16, E1 계측(최종 밀도)** | **860** (67초) | **3,716** (15.5초) |
| MPS, 워커 16, 계측 없음 | 929 (62초) | 4,800 (12초) |
| MPS 없이, 워커 16, 계측(초기 밀도) | 488 (118초) | 2,095 (27.5초) |

- **MPS 몫의 GPU가 워커 8개에서 이미 포화**한다. 워커를 늘려도 처리량은 그대로이고 episode당 시간만 길어진다. 워커당 GPU 메모리는 약 386MB(16개면 약 6.2GB)
- **MPS가 1.6~1.8배 빠르다**(MPS 없이 16 프로세스 488·2,095 대 811·3,200)
- **계측 부담**: 처음 밀도(OEI 4/10 step, 계획 재실행 10/25 step, 계획 시점 예측 오차를 모든 OEI 상태에서)에서 MW +50%, DMC +15%였다. 그래서 계획 재실행을 MW 20·DMC 50 step마다로, episode별 계획 시점 오차를 OEI 상태 중 8개로 줄였다(최종 밀도). 최종 부담은 MW +29%, DMC +8%다. 칸(90 episode)당 OEI 상태 MW 2,250·DMC 4,500, 계획 편차 상태 MW 450·DMC 900, 고정 상태 계획 편차 150(decision job)이 남는다
- 결과 파일: `results/E0_tdmpc2/throughput.json`, 로그 `results/E0_tdmpc2/logs/E0_bench_*.log`, GPU 표본 `E0_bench_gpu.csv`

### 5.3 E1 TD-MPC2 GPU 시간 추정 (서버 GPU 0)
- E1 규모(현재 `ktable.json`, 12과제 × 칸 3 × 체크포인트 3): episode job 4,104개 = **MW 13,680 episode + DMC 6,840 episode**, 칸별 job 216개(prederr 108 + decision 108)
- **DINO-WM이 GPU를 거의 쓰지 않을 때(bench 조건)**: MW 13,680/3,716 = 3.7시간 + DMC 6,840/860 = 8.0시간 + 칸별 job 216개 약 0.3시간(시간 측정: prederr 10~11초, decision 47~49초, 워커 4개 동시) → **약 12시간**
- **DINO-WM이 GPU 0을 계속 무겁게 쓸 때(held 조건, 속도 약 절반)**: **약 24~26시간**
- 설계 §6.9의 E1 50 GPU-시간(두 모델 합) 안에 든다. 다만 두 A 작업자가 GPU 0을 동시에 쓰면 서로 늦어지므로, 순서를 나누거나 한쪽을 GPU 1로 옮기는 것이 빠르다(조정자 결정)
- 줄이는 선택지(효과 추정): DMC k=2 칸 제외 −1,080 DMC episode(−1.3시간, §10.2 결정 2-b), 계측 끔 −2시간 정도(권장하지 않음: OEI 기록이 E2의 재료)
- 권장 워커 수: 16(8과 거의 같은 처리량, GPU 메모리 6GB). DINO-WM이 메모리를 많이 쓰면 8로 줄여도 처리량 손실이 거의 없다

## 6. 과제와 비트 수 선택(held, 설계 §6.0)

### 6.1 규칙과 실행
- 규칙(설계 §6.0, 사전등록 그대로)
  - Meta-World: 기준(none) 성공률 ≥ 80%인 후보 과제만 남긴다(목표 8개)
  - 모든 과제: held episode에서 det가 기준 대비 20%p 이상(DMC는 return 20% 이상) 떨어뜨리는 **가장 큰 k ∈ {3, 4, 5, 6}**을 1차 k로, k−1과 k+1을 2차 칸으로 둔다
  - 20%p에 이르는 k가 없을 때: 설계가 "R2-13 규칙 그대로"라고 했으므로 R2-13의 대체 규칙(15%p 이상인 가장 큰 k, 그것도 없으면 시험한 가장 작은 k)을 적용하고 표시한다
- 실행: held seed 12000~12019, 체크포인트 seed 1·2·3, none과 det(k = 3, 4, 5, 6). MW는 체크포인트마다 20 episode(과제당 60), DMC는 체크포인트마다 10 episode(과제당 30). 범위는 E0 보정 범위(§9)
- 하락: MW는 성공률 차(%p), DMC는 return 상대 하락. 짝 bootstrap 90% CI. 판정은 점추정으로(R2-13과 같음)

- 실행 결과: 600 job 전부 OK(3,000 episode, 14:29~16:38Z)

### 6.2 결과: 과제 목록과 k 표

**Meta-World 기준 성공률(held, 체크포인트 1/2/3)**: 8과제 모두 자격(≥ 80%). push·pick-place·door-open·drawer-open·button-press-topdown·peg-insert-side·window-open 100%(60/60), sweep-into 98%(95/100/100). → **MW 8과제 전부 E1에 들어간다**(목표 8 충족)

**det 하락(held, 체크포인트 3개 합; MW %p n=60, DMC return % n=30; [90% CI])**

| 과제 | det k=3 | k=4 | k=5 | k=6 | **1차 k** | **E1 칸** | 비고 |
|---|---|---|---|---|---|---|---|
| mw-push | 52 [42, 62] | 13 [7, 20] | 0 | 0 | **3** | 2, 3, 4 | 체크포인트 1만 보면 4(k=4: 30%p) |
| mw-pick-place | 33 [23, 43] | 8 [3, 15] | 0 | 0 | **3** | 2, 3, 4 | |
| mw-door-open | 18 [10, 27] | 30 [20, 40] | 2 | 2 | **4** | 3, 4, 5 | k=4가 k=3보다 더 해롭다(비단조) |
| mw-drawer-open | 12 [5, 18] | 0 | 0 | 0 | **3** | 2, 3, 4 | **대체 규칙**: 20%p·15%p에 이르는 k가 없어 시험한 가장 작은 k. 체크포인트 2에서만 하락(0/35/0%p) |
| mw-button-press-topdown | 38 [28, 48] | 0 | 0 | 0 | **3** | 2, 3, 4 | 체크포인트 간 차이 큼(90/10/15%p) |
| mw-peg-insert-side | 53 [43, 63] | 28 [18, 38] | 3 | 2 | **4** | 3, 4, 5 | 체크포인트 1의 범위가 넓다(dyn0 hi 35.7) |
| mw-window-open | 38 [28, 48] | 32 [22, 42] | 5 | 2 | **4** | 3, 4, 5 | |
| mw-sweep-into | 20 [12, 30] | −2 | −2 | −2 | **3** | 2, 3, 4 | **문턱 위에 정확히 걸림**(12/60). 20 ep 문자 그대로 읽으면 5%p → 대체 규칙으로 가도 1차 k는 3 |
| cheetah-run | 93 [90, 95] | 73 [69, 76] | 46 [41, 51] | 13 [8, 20] | **5** | 4, 5, 6 | 체크포인트 1만 보면 6(k=6: 27%) |
| finger-turn-hard | 63 [50, 75] | 39 [26, 54] | 0 | 0 | **4** | 3, 4, 5 | |
| quadruped-walk | 69 [59, 79] | 6 [5, 8] | 1 | 0 | **3** | 2, 3, 4 | |
| walker-walk | 56 [52, 61] | 18 [15, 21] | 3 | 0 | **3** | 2, 3, 4 | k=4가 문턱 근처(17.8%) |

- DMC 기준 return(held): cheetah 903.5, finger 976.5, quadruped 982.3, walker 989.5
- **세 가지 읽기의 비교**(설계 문구 "held 20 episode"): 위 표(체크포인트마다 MW 20·DMC 10 ep), "과제당 20 ep 문자 그대로"(체크포인트 1·2·3에서 7·7·6 ep), "체크포인트 1만 20 ep". 1차 k가 갈리는 과제는 셋이다
  - mw-push: 3 / 3 / **4**, cheetah-run: 5 / 5 / **6**, mw-sweep-into: 3 / (자격 k 없음 → 대체 규칙 3) / 3, mw-drawer-open: 대체 3 / 대체 3 / 대체 3
- `results/E0_tdmpc2/ktable.json`(위 표의 E1 칸), `held_selection.json`(체크포인트별 하락, 세 읽기), `held_table.md`

### 6.3 k−1 = 2 칸(보조 정보, 선택 규칙과 무관)

- 1차 k가 3인 7과제는 규칙대로면 2차 칸에 k = 2가 들어간다. k = 2는 설계의 후보 범위 {3, 4, 5, 6} 밖이라 넣을지 판단 자료로 det와 sh-iter를 같은 held episode에서 쟀다(16:38~16:53Z, 144 job)

| 과제 | det 하락 k=2 | sh-iter 하락 k=2 | 성공률(또는 return) det / sh-iter |
|---|---|---|---|
| mw-button-press-topdown | 98 [95, 100] | **0** [0, 0] | 2% / 100% |
| mw-drawer-open | 72 [62, 82] | **2** [0, 5] | 28% / 98% |
| mw-pick-place | 93 [88, 98] | 23 [15, 33] | 7% / 77% |
| mw-push | 95 [90, 98] | 20 [12, 28] | 5% / 80% |
| mw-sweep-into | 73 [63, 83] | **5** [2, 10] | 25% / 93% |
| quadruped-walk | 79 [73, 86] | 76 [68, 83] | return 202 / 236 |
| walker-walk | 93 [92, 94] | 91 [90, 92] | return 68 / 86 |

- MW의 k=2 칸은 "붕괴 칸"이 아니다. det는 거의 전부 실패하지만 sh-iter는 손상의 75~100%를 없앤다. 처방 효과가 가장 크게 보이는 칸이다
- DMC의 k=2 칸은 det와 sh-iter가 모두 붕괴한다(return 7~24%만 남음). 정보가 적다
- 특히 drawer-open과 sweep-into는 k ≥ 3에서 det 하락이 작거나 문턱에 걸리지만 k = 2에서는 det 하락이 크다(72, 73%p)
- 이 표는 판단 자료일 뿐 선택 규칙에는 쓰지 않았다(held seed, 판정용 eval seed와 겹치지 않음)

## 7. E1용 runner (E2·E3·E5에서도 재사용)

### 7.1 구성 파일 (`main/code/tdmpc2/`)

| 파일 | 역할 |
|---|---|
| `qlib.py` | 공용 라이브러리(파일럿 nt_lib의 확장). 양자화 hook(7변형), 계측(PlanProbe, 계획 재실행), episode 루프, 예측 오차 함수 |
| `calib_lib.py` | 보정 rollout 기록, 층별 범위, held 집합 예측 오차, 고정 상태 결정 분석(R2-8b decision) |
| `runner.py` | manifest 기반 실행기. job마다 결과 파일 1개(`out/jobs/<type>/<id>.json.gz`), 있으면 건너뜀, lock 파일로 여러 runner 동시 실행 가능, 실패는 `out/errors/<id>.txt` |
| `make_manifest.py` | manifest 생성(`calib`, `held`, `heldx`, `repro`, `bench`, `e1`, `smoke`) |
| `status.py` | 진행 상황과 남은 시간 대략 추정 |
| `analyze_e0.py` | 재현표, 처리량, held 선택(`ktable.json`) |
| `run_e1.sh`, `run_held.sh`, `run_post_held.sh`, `run_bench.sh` | 서버 실행 스크립트(tmux에서). `run_e1.sh`는 E1 전용(E0에서 실행하지 않음) |
| `env.sh`, `mps.sh` | 실행 환경, MPS 데몬 |
| `setup_server.sh`, `setup_mw_server.sh` | 사용자 공간 설치(재현용) |
| `test_equiv.py`, `test_cache.py` | 파일럿 일치·계측 불변성 점검, 워커 캐시 점검 |

### 7.2 변형 (설계 §4.2; 모두 같은 k-bit 격자 [lo, hi], Δ = (hi − lo)/(2^k − 1), x = (clip(a) − lo)/Δ)

| 이름(코드) | 정의 | 파일럿과의 관계 |
|---|---|---|
| none | 양자화 없음 | |
| det | round(x) | R2-8b det와 비트 단위로 같음 |
| ind | floor(x + u), u는 항목(후보 × unit)마다 독립, forward마다 새로 | R2-8b ind와 같음 |
| shiter | floor(x + u), u는 forward마다 unit별 벡터 하나를 모든 후보가 공유 | R2-8b sh-fwd와 같음 |
| shstep | 층마다 (horizon 위치 × unit) 표를 제어 step 시작에 뽑아 그 step의 MPPI 반복 6회와 policy prior rollout이 공유 | R2-8b sh-step과 같음 |
| shep | shstep의 표를 episode 시작에 한 번만 뽑음 | R2-8b sh-ep와 같음 |
| detshift | 결정적 반올림을 무작위로 이동한 격자에서: q = round(x − o), 값 = lo + Δ(q + o), o ~ U[−½, ½)는 unit마다, 모든 후보가 공유, forward마다 새로(shiter와 같은 일정). 준위 수 2^k로 같음 | 새 변형(설계 §4.2의 det+shift) |

- 무작위 값은 모두 별도 CUDA generator(episode seed × 7 + 13)에서 나온다. planner의 난수열은 모든 변형에서 같다(짝 비교)
- det+shift의 성질: 한 forward 안에서는 입력이 같으면 오차가 같고(결정적), 격자가 forward마다 움직여 반복 사이의 지속 편향이 없다. o에 대해 평균하면 무편향이고 MSE는 det와 같은 Δ²/12다(확률적 반올림은 Δ²/6). 실제 정수 커널에서는 o·Δ를 다음 층 bias로 접어 넣을 수 있다

### 7.3 job 종류와 기록 항목

- **episodes**(칸 하나의 episode 묶음): 과제, 체크포인트 seed, k, 변형, episode 목록, planner override, 계측 설정
  - episode마다: return, 마지막 step 성공(`success`), 한 번이라도 성공(`success_any`), step 수, 표 갱신 횟수, 소요 시간
  - **OEI와 결정 지표**(MW 4 step마다 = episode당 25 상태, DMC 10 step마다 = 50 상태): 첫 반복과 마지막 반복의 후보 512개 전부를 같은 RNG 상태(같은 Q 쌍, 같은 policy 잡음)에서 비양자화 모델로 다시 평가한다. 비용 오차 e = Ĵ − J(비용 = −가치, 즉 e = V₀ − V_q)
    - `oei` = e(실행된 elite: Gumbel로 뽑힌 마지막 반복 elite) − 첫 반복 무작위 후보(policy prior 24개 제외 488개)의 평균 e. 음수면 계획기가 오차가 유리한 후보를 골랐다는 뜻
    - `oei_w`(elite 가중 평균), `oei_top1`(마지막 반복 argmax), `oei_first`(첫 반복 elite 가중, 한 번의 선택만의 쏠림), 정규화용 `f_sd_v0`(첫 반복 후보 가치의 std)
    - 고정 후보 집합 결정 지표(첫/마지막 반복): Spearman(V_q, V₀), argmax flip, top-64 Jaccard, gap(1·2위 차/std), regret, 그리고 가치 추정기 자체의 seed 잡음 바닥(`f_sn_spear`, `f_sn_flip`). step별 값이 그대로 남으므로 gap 사분위별 flip을 E2에서 다시 계산할 수 있다
  - **계획 편차**(MW 20 step마다 = 5 상태, DMC 50 step마다 = 10 상태; §5.2의 비용 때문에 처음 밀도 10/25에서 줄임): 같은 관측, 같은 이전 mean, 같은 RNG 상태에서 비양자화 planner를 다시 돌린 결과와 최종 MPPI mean 첫 step의 RMS 편차(`dev_mean`), 실행 행동 편차(`dev_act`), 다른 seed의 비양자화 planner로 잰 바닥(`sn_mean`, `sn_act`)
  - **episode별 예측 오차**(episode 끝, 확률적 변형은 dither seed 3개 평균): 실행 궤적 입력(`perr_traj`: 모든 창에서 err1·err3·err5, 1-step reward 오차, 참 미래 관측 인코딩 대비 true1·3·5와 비양자화 기준 true0)과 계획 시점 입력(`perr_plan`: OEI 측정 상태 중 고르게 8개의 첫 반복 후보 512개)
  - 모든 계측은 RNG 상태를 저장·복원해 closed loop를 바꾸지 않는다(§8에서 7변형 모두 비트 일치 확인)
- **prederr**(칸 하나: 과제 × 체크포인트 × k, 6변형): 보정 episode 5~9의 held 집합에서 계획 시점 입력(후보 시퀀스 최대 60,000개, R2-8b prep와 같은 방법)과 궤적 입력의 1/3/5-step 상대 오차, dither seed 3개 평균. `err1h`·`err3`·`errr`가 R2-8b의 err1·err3·errr와 같은 정의다
- **decision**(칸 하나): 보정 episode 5~9의 결정 상태 150개에서 R2-8b decision 분석(고정 후보 집합 지표 + 한 step 계획 재실행 편차/seed 바닥). 여기에 오차 구조 기술자 두 개를 더했다(설계 §4.3, E3가 E1 자료를 다시 쓸 수 있게): `rho_iter`(같은 후보를 연속 두 반복으로 평가했을 때 비용 오차의 후보 간 상관; det·sh-step·sh-ep = 1, 새로 뽑는 변형은 0.05~0.2)와 `common_frac`(모든 후보에 공통인 오차 비중)
  - 시간 측정용으로 돌린 mw-push s1 k=3 decision의 고정 상태 계획 편차(seed 바닥 대비): det 3.19, sh-step 3.20, ind 2.34, sh-iter 2.36, det+shift 2.04. R2-8b(3.31, 3.29, 2.28, 2.41)와 거의 같다
- **calib**, **ranges**: §9

### 7.4 사용법 (정확한 명령)

```bash
# 맥북 → 서버 코드 동기화
rsync -a "/Users/song-yonghwi/Documents/physical ai - Neural Networks/A_dithered_world_models/main/code/tdmpc2/" kiis-mvf-gpu:pmx/A/tdmpc2_main/code/
# E1 시작 (조정자가 DESIGN_main.md를 고정한 뒤). manifest는 out/ktable.json으로 한 번 만들어진다
ssh kiis-mvf-gpu 'tmux new-session -d -s A-E1-tdmpc2 "WORKERS=16 bash ~/pmx/A/tdmpc2_main/code/run_e1.sh"'
# 진행 상황 (로그: ~/pmx/A/logs/E1_tdmpc2.log)
ssh kiis-mvf-gpu 'source ~/pmx/A/tdmpc2_main/code/env.sh; cd ~/pmx/A/tdmpc2_main/code; $PY status.py --manifest ../out/manifests/e1.jsonl --out ../out'
# 멈추기와 이어 돌리기: tmux kill-session -t A-E1-tdmpc2 → 같은 시작 명령(끝난 job은 건너뜀)
# 결과를 맥북으로
rsync -a kiis-mvf-gpu:pmx/A/tdmpc2_main/out/jobs/ "/Users/song-yonghwi/Documents/physical ai - Neural Networks/A_dithered_world_models/main/results/E1_tdmpc2/jobs/"
# MPS 데몬 상태/중지 (E1이 끝난 뒤)
ssh kiis-mvf-gpu 'bash ~/pmx/A/tdmpc2_main/code/mps.sh status'   # stop
```

- manifest를 직접 만들 때: `$PY make_manifest.py e1 --ktable ../out/ktable.json --ckpts 1,2,3 --eps 30 --chunk 5 --out ../out/manifests/e1.jsonl`
- **E2·E3·E5 재사용**: job의 `planner`에 `num_samples`, `num_elites`, `temperature`, `iterations`, `num_pi_trajs`를 넣으면 그 job에만 적용된다(캐시된 agent는 job마다 기본값으로 되돌린다). `warm_start: false`는 매 step mean 0에서 시작한다. 새 변형은 `qlib.Quantizer._hook`에, 새 범위 규칙(E5)은 `scales` 이름으로 추가한다. id의 `ptag`로 칸을 구분한다(예: `N128it3`, `nows`; 스모크에서 동작 확인)

## 8. 스모크 테스트와 구현 점검

- **파일럿 일치**: mw-push k=3, R2-8 범위, smoke seed 19000~19001에서 qlib의 det·ind·sh-iter·sh-step·sh-ep episode가 파일럿 nt_lib와 return과 행동열 해시까지 같다(10/10)
- **계측 불변성**: 7변형 × 2 seed에서 계측(OEI 25 상태, 계획 재실행 10 상태, 예측 오차)을 켜도 episode가 비트 단위로 같다(14/14)
- **정의상 값**: none에서 OEI = 0, 계획 편차 = 0, Spearman = 1
- **MPS 수치 동일**: MPS 없이(test_equiv)와 MPS로(runner 스모크) 같은 seed의 return이 같다
- **forward 호출 패턴**: 제어 step마다 512행 호출이 층마다 18회, policy prior 24행이 dyn 층마다 2회(파일럿 PLAN §8과 같음). 표 갱신: sh-step 101회(100 step + 1), sh-ep 1회
- **runner job 종류**: episodes 10개(7변형, cheetah, planner override N=128·반복 3, warm start 끔), prederr 1개, decision 1개 모두 OK. 결과 파일 구조 확인
- **E1 모양 job**(E0 보정 범위, 체크포인트 2·3, smoke seed): cheetah s2 k=5 sh-iter, door-open s3 k=4 det+shift(2 ep), walker s1 k=2 det 모두 OK. 실행 elite 기록(e_chosen) 누락 0
- **워커 캐시**(`test_cache.py`): 한 프로세스에서 (과제, 체크포인트)를 4번 바꿔 agent 내보내기가 2번 일어나도 기록이 빠지지 않고, 다시 실은 agent의 episode가 처음과 비트 단위로 같다
- **칸별 job**: E1의 prederr·decision job 4개(cheetah s1 k=5, mw-push s1 k=3)를 별도 폴더에서 돌려 시간과 출력을 확인했다(결과는 E1 폴더에 남기지 않았다)
- 스모크 결과는 기능 점검에만 쓰고 해석하지 않는다(smoke seed는 다른 분할과 겹치지 않음)

## 9. seed와 episode 범위 (모든 체크포인트 seed에 같은 episode seed)

| 분할 | episode seed | 용도 |
|---|---|---|
| calib | 11000~11009 | 비양자화 보정 rollout. ep 0~4 → 층별 범위(min, 99.99 백분위, 12,000행; R2-8 절차), ep 5~9 → held 예측 오차 집합과 고정 결정 상태 |
| held | 12000~12019 | 과제·k 선택(MW 20, DMC 10 episode/체크포인트) |
| eval | 20000~20029 | E1 판정 전용. E0에서는 한 번도 쓰지 않았다 |
| smoke | 19000~19009 | 스모크(해석하지 않음) |
| bench | 18000~ | 처리량 |
| repro | 2000~2039 | R2-8b 재현(파일럿 seed) |

- 범위는 (과제, 체크포인트)마다 따로 보정했다(`out/scales/<task>_s<ckpt>.json`). 평가 seed와 겹치지 않는다
- 각 episode: 환경을 그 seed로 새로 만들고 random·numpy·torch를 같은 seed로 맞춘다. dither generator는 seed × 7 + 13

## 10. 편차와 결정이 필요한 것

### 10.1 설계에 없던 구현 선택 (DESIGN_main.md에 고정할 항목)
1. **det+shift의 정확한 정의**(설계 §4.2는 "결정적 반올림 + 반복마다 바뀌는 공유 offset(무작위 격자)"만 적음): unit마다 offset o ~ U[−½, ½)(격자 단위), 모든 후보가 공유, **forward 호출마다 새로**(sh-iter와 같은 일정; 한 MPPI 반복은 horizon 3개의 forward로 이루어지므로 "반복마다"보다 촘촘함), q = round(x − o)를 [0, 2^k − 1]로 자르고 값은 lo + Δ(q + o). 이것은 "후보 공유·매 forward 새로"인 subtractive dither와 같다(오차가 입력에 대한 결정적 톱니 함수이고 o에 대해 무편향, MSE Δ²/12). 층마다 스칼라 하나만 이동하는 판은 E3 후보로 남긴다
2. **TD-MPC2 OEI의 조작적 정의**(설계 §4.3): 비용 = −가치, 비용 오차 e = V₀ − V_q(V₀는 같은 RNG 상태, 즉 같은 Q 쌍·policy 잡음으로 잰 비양자화 가치). "고른 후보" = 실제 실행된 elite(마지막 반복에서 Gumbel로 뽑힌 것), "무작위 후보" = 첫 반복의 표본 후보 488개(policy prior 제안 24개 제외)의 평균. 보조로 elite 가중(`oei_w`), argmax(`oei_top1`), 첫 반복 안의 선택만(`oei_first`), 마지막 반복 모집단 평균(`l_e_rand`)을 함께 남겼다. 과제 간 비교용 정규화 값(`f_sd_v0`)도 있다
3. **계획 시점 5-step 오차**: TD-MPC2 후보는 horizon 3이라 4~5번째 행동이 없다. 후보의 3개 행동 뒤에 비양자화 rollout을 따라 policy prior의 평균 행동 2개를 붙인 고정 행동열을 쓴다(모든 변형에 같은 행동열). 데이터셋형 입력(실행 궤적)의 5-step 오차는 실제 실행 행동을 쓴다
4. **예측 오차 정규화**: R2-8b와 같다(비양자화 예측의 분산으로 나눈 상대 MSE; `err1h`·`err3`·`errr`가 R2-8b의 err1·err3·errr). 확률적 변형은 dither seed 3개 평균(설계 그대로)
5. **보정 범위**: (과제, 체크포인트)마다 새 보정 seed(11000~11004)에서 R2-8 절차(층별 min과 99.99 백분위)로 다시 잡았다. 그래서 같은 k라도 파일럿과 격자가 다르다(예: cheetah s1 dyn0 hi 11.7 대 R2-8 9.4). 재현(§4)만 R2-8 범위를 썼다
6. **계측 간격**(E1 비용과의 절충, §5.2): OEI·결정 지표는 MW 4 step·DMC 10 step마다, 계획 재실행은 MW 20·DMC 50 step마다, episode별 계획 시점 오차는 OEI 상태 중 8개. 칸(90 episode)당 MW 2,250·DMC 4,500 OEI 상태, MW 450·DMC 900 계획 편차 상태다. 설계 §4.3의 계획 편차(R2-8b 정의, 고정 상태 150개)는 칸마다 decision job으로 따로 잰다. 계측은 RNG 상태를 저장·복원해 closed loop를 바꾸지 않는다
7. **held 예측 오차 집합과 결정 상태**: 보정 episode 5~9(범위를 잡은 0~4와 겹치지 않음). DMC 60,000 / MW 48,000 계획 시점 시퀀스, 궤적 창 DMC 2,480 / MW 480개

### 10.2 설계 문구 해석과 결정이 필요한 것 (조정자 확인 필요)
1. **held 크기**: "held 20 episode"를 체크포인트 seed마다 MW 20, DMC 10 episode로 돌렸다(과제당 MW 60, DMC 30). MW 성공률은 이진이라 20 episode로는 표준오차가 약 10%p라서 늘렸다. DMC는 return 하락의 분산이 작고 GPU 0이 공유라 시간을 아꼈다. 문자 그대로 읽기(과제당 20 ep = 7/7/6)와 체크포인트 1만 쓴 결과도 계산했다(§6.2). **1차 k가 갈리는 과제: mw-push(3 대 체크포인트 1만 4), cheetah-run(5 대 6)**. 어느 읽기를 고정할지 정해야 한다(추천: 체크포인트 3개 합, 현재 `ktable.json`)
2. **1차 k = 3일 때 k−1 = 2를 넣을지**(7과제): §6.3에 따르면 MW의 k=2 칸은 처방 효과가 가장 크게 보이는 칸이고(det 72~98%p 대 sh-iter 0~23%p), DMC의 k=2 칸은 두 변형 모두 붕괴한다. 선택지
   - (a) 규칙 그대로 k−1 = 2를 모든 1차-3 과제에 넣는다(현재 `ktable.json`; E1 episode 20,520개 중 k=2가 MW 2,700, DMC 1,080)
   - (b) MW만 k=2를 넣고 DMC(quadruped, walker)는 {3, 4}만 둔다(DMC 1,080 episode 절약)
   - (c) k ∈ {3..6} 밖은 모두 뺀다(1차-3 과제는 {3, 4}만)
3. **20%p에 이르는 k가 없는 과제**: mw-drawer-open(k ≥ 3에서 최대 11.7%p). R2-13 대체 규칙으로 1차 k = 3(칸 2, 3, 4)을 넣었다. 이 과제는 k = 3에서 "det 하락이 실재하는 칸"이 아닐 가능성이 크다. mw-sweep-into는 k = 3 하락이 정확히 20.0%p(12/60)로 문턱 위다. 두 과제 모두 k = 2에서는 det 하락이 크다(72, 73%p). 선택지: 대체 규칙대로 유지 / 과제 제외(MW 6개) / 이 두 과제만 1차 k를 2로(규칙 밖)
4. walker-walk의 k = 4 하락은 17.8% [15.0, 20.9]로 문턱 바로 아래다(1차 k = 3). 참고로만 적는다

### 10.3 그 밖의 편차
- **처리량 측정 창**: 조정자 알림대로 13:48~14:17Z, 14:40~14:42Z에는 B의 작은 job이 GPU 0에 있었다. 이 창에 걸친 재현·보정·held 초반의 소요 시간은 처리량 근거에서 뺐고, §5는 그 뒤 따로 잰 값만 쓴다
- **버그 수정 1건**(qlib.PlanProbe, 15:00Z): 워커 캐시에서 agent를 내보낼 때 마지막 elite 추첨 hook이 풀릴 수 있었다. 지금까지의 결과에는 영향이 없다(재현·held는 계측 없음, 스모크는 내보내기 없음). 수정 뒤 test_equiv 28 episode가 수정 전과 비트 단위로 같고, test_cache로 내보내기 뒤 기록을 확인했다
- **MPS 공정성**: MPS 없이 16 프로세스를 띄우면 GPU 시분할에서 DINO-WM 쪽 몫을 더 가져갈 수 있지만, 측정해 보니 MPS가 오히려 1.6~1.8배 빨랐다. E1은 MPS로 돌린다(지시대로)
- E1 manifest 초안(`results/E0_tdmpc2/manifests/e1_draft.jsonl`)은 만들기만 했다. `run_e1.sh`는 `out/manifests/e1.jsonl`이 없으면 그때의 `out/ktable.json`으로 새로 만든다. 그래서 조정자가 §10.2 결정에 따라 `ktable.json`의 `cells`를 고친 뒤 시작하면 된다(예: 결정 2-b면 quadruped·walker의 `cells`를 [3, 4]로)

## 11. 비용과 시간

- **RunPod**: 쓰지 않음, **$0**. pod 없음
- **서버 GPU 0 사용**(모두 MPS, GPU 1은 쓰지 않음)
  - 재현 51분(워커 16), 보정 46분(워커 8, 재현과 겹침), held 2.14시간(워커 16), k=2 보조 15분, 처리량 측정 75분, 점검·스모크 약 20분 → 약 5시간(설계 E0 예산 6 GPU-시간 안, DINO-WM과 공유)
- **디스크**: 산출물 `~/pmx/A/tdmpc2_main/out` 1.3GB(대부분 보정 npz, E1 prederr·decision job이 쓴다), 체크포인트 1.1GB, venv·micromamba 약 6.3GB. 남은 공간 293GB. 처리량 측정 job 결과(13MB)는 지웠고 로그만 남겼다
- **맥북으로 옮긴 결과**: `results/E0_tdmpc2/`(7MB): job 결과 817개(`jobs/`), 요약 JSON·CSV·MD, 범위(`scales/`), manifest, 로그, 패키지 목록, 체크포인트 sha256
- **지금 돌고 있는 작업: 없음.** 내 tmux 세션(A-E0-*)은 모두 끝났고, MPS 데몬은 18:09Z에 껐다

## 12. 파일

- 코드: `main/code/tdmpc2/`(§7.1 표)
- 결과: `main/results/E0_tdmpc2/`
  - `repro_summary.json`, `repro_table.md`, `repro_drops.csv`(§4)
  - `throughput.json`, `e1_estimate.json`(§5)
  - `ktable.json`, `held_selection.json`, `held_table.md`, `held_det_drops.csv`(§6)
  - `scales/<task>_s<ckpt>.json`(층별 범위 36개), `manifests/`, `logs/`, `jobs/`(episode·prederr·decision·calib·ranges job 결과)
  - `freeze_a-mw.txt`, `freeze_a-tdmpc2.txt`, `SHA256SUMS`, `out_test_*.jsonl`(구현 점검)
- 서버: 작업 폴더 `~/pmx/A/tdmpc2_main/`(code, tdmpc2, ckpt, inputs/r28, out), 로그 `~/pmx/A/logs/E0_*`
