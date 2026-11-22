# E17 정수 kernel과 BF16 폐루프 — 실행 계획 (본 실행 전)

- 작성: 2026-11-21T18:50Z~19:22Z, k 작업자. 본 실행(지연 본 측정과 폐루프 main manifest)의 결과는 이 문서를 쓰는 동안 하나도 없다. 미리 돈 것은 §0.7 점검용 단계뿐이다(§4·§5.1: 점검 1·2, 지연 smoke·quick, 처리량 bench, 보정 episode, FP32 none).
- 기준: [../DESIGN_tmlr.md](../DESIGN_tmlr.md) v1.0(sha256 `36d69ea8…0cc4`) §0, §7, §8. 가설·지표·문턱·칸·seed는 설계 그대로이고, 이 문서는 운영 세부만 정한다([../WORKER_RULES.md](../WORKER_RULES.md)).
- 코드: [code/](code/), 기록: [LOG.md](LOG.md). pod 결과의 회수 위치: `results/raw/k.1/`(조정자의 `pull_all.sh`).

## 1. 판정 (설계 §7 그대로, 계산 규칙만 구체화)

| 가설 | 판정 규칙 | 코드 |
|---|---|---|
| **K1** | det+shift의 rollout 지연 오버헤드가 6조건(5M·317M × 후보 512·2048·8192) **모두에서 중앙값 ≤ 5%**. 오버헤드 = 짝 block의 (det+shift − det)/det, 조건마다 짝 10개(10회전)의 중앙값. CI는 짝 bootstrap 10,000회(RandomState(0), 백분위 95%, E8 규약)로 보고만 한다. 허용 오차 1e-9 | `code/e17_lat_analyze.py` |
| **K2** | 4과제(mw-push k3, mw-pick-place k3, cheetah-run k5, walker-walk k3) 모두에서 정수 kernel의 Δ_int = D(det) − D(det+shift)와 E11 fake-quant의 Δ_fq가 **부호가 같고 95% CI가 겹친다**. 부호는 \|Δ\| ≤ 1e-9이면 0으로 본다. Δ_fq는 td 작업자의 E11 칸(그쪽 pod, 그쪽 none)으로 계산한다. 이쪽은 Δ_int와 CI를 낸다. td의 episode 자료가 단계 B에 있으면 비교까지 하고, 아니면 조정자가 합친다(설계 §8.5) | `code/analyze_e17.py` |
| **K3** | 미리 바꾼 BF16(bf16p)의 D가 12과제 **모두에서 95% CI에 0을 포함**한다 | `code/analyze_e17.py` |
| 기술 | 접은 per-channel 정수 구성(4과제 × k {2, 3} × {det, sh-iter})의 D와 CI. td의 E12 fake-quant per-channel 칸이 있으면 D_int − D_fq와, 두 pod를 독립 표본으로 본 CI(이쪽 RandomState(0), 그쪽 RandomState(1); pod가 다르면 짝이 아니다, 설계 §0.3) | `code/analyze_e17.py` |

- D(설계 §0.3): 같은 pod의 FP32 none 대비. Meta-World는 마지막 step 성공률 차(none − v, E1의 `success`), DMC는 1 − ΣR_v/ΣR_none(재표본마다 비율). 짝 = (체크포인트, seed), 과제마다 none과 모든 칸에 공통인 짝으로 재표본 행렬 하나(10,000 × n, RandomState(0))를 모든 통계에 쓴다. Δ = D(det) − D(v)는 같은 재표본에서 구한다.
- 함께 내는 기술 수치: 단측 p(D ≤ 0), red, D_v − D_det의 CI(해로움 기준 0.05), sh-iter 오버헤드(E8 L3와 같은 꼴), 절대 ms 차, 측정 상태.

## 2. pod

| 태그 | GPU | $/h | 생성 | 자동 종료 | 맡은 일 |
|---|---|---|---|---|---|
| k.1 (`atm-k.1`, `0fwb5dot0ow6m2`) | NVIDIA A40 (SECURE) | 0.59 | 2026-11-21T18:31:09Z | 2026-11-22T02:30:38Z | E17 전부(지연 → 폐루프) |

- cgroup CPU 할당 7.65코어(E9와 같음), RAM 503 GB, GPU 46 GB, 전력 상한 300 W, 최대 SM clock 1,740 MHz, driver 595.112.07.
- 소프트웨어(설계 §0.2, E9 `setup_e9.sh` 그대로 + nvidia-ml-py): torch 2.7.1+cu126, triton 3.3.1, tensordict 0.8.3, numpy 1.24.4, dm-control 1.0.16, mujoco 3.1.2, gym 0.21.0, Metaworld@04be337, mujoco-py 2.1.2.14, tdmpc2 `332f19e4`. 체크포인트 36개 SHA256이 E0와 같다.
- 짝 규칙(§0.3): 모든 과제의 none과 모든 변형이 이 pod 하나에서 돈다. 이어 돌 일이 생기면 같은 A40 pod(k.1b)에서 이어 돈다.

## 3. 구현 선택

### 3.1 지연(K1): E8 규약, 정수 det·sh-iter·det+shift

- 측정 방식은 E8의 graph-WM(주 방식) 그대로다(`code/e17_lat.py`).
  - 한 제어 step의 world-model rollout 전체(정책 사전 rollout: 배치 24, dynamics 2회 + MPPI 6반복 × horizon 3 × (reward, dynamics), 배치 N)를 CUDA graph 하나로 잡아 재생한다.
  - 모델: E8과 같은 공식 TD-MPC2 5M·317M, `torch.manual_seed(0)` 무작위 초기화(mw-push 차원), E8의 무작위 입력 범위. 이 pod에서 다시 잰 범위가 E8 pod의 값과 같다(차 0).
  - GEMM 경로 'a'(E8이 모든 크기에서 고른 Triton int8 GEMM), 타일 설정은 E8의 A40 조율표(`code/inputs/gemm_cfg_e8.json`)를 그대로 쓴다. 활성 코드는 E8처럼 4-bit(int8 그릇)다. 그릇 안의 코드 폭은 산술 비용을 바꾸지 않는다.
  - det와 sh-iter는 E8의 `e8lib.py`(md5 `7dd7b30d…`, 바꾸지 않음) 그대로다.
- **det+shift**(`code/e17lat.py`): 같은 층·가중치·GEMM·launch 설정에서 양자화 row kernel 두 개(층 0, 층 1)만 바꾼다.
  - offset: o = u − ½. u는 E8 sh-iter가 같은 호출에서 그릴 Philox 줄이다(층·forward마다 한 줄, 모든 후보가 공유, 한 kernel 앞에서 그림). 즉 o ~ U[−½, ½), (1, W).
  - 코드: clamp(rint(x − o), 0, nl), 2^k단계 그대로. 역양자화 값 lo + Δ(q + o).
  - **offset 보정**(설계 "W·(Δ·o)를 bias에 더함, 호출당 행렬–벡터 곱 하나"): c_eff[n] = c[n] + α[n]·Σ_k Wq[n, k]·o[k](α = Δ·s는 다음 GEMM의 역양자화 scale, Wq는 다음 층의 int8 가중치)를 양자화 row kernel 안에서 계산한다. 별도 launch는 없다. 다음 GEMM(E8 `gemm_a`, 그대로)이 c_eff를 bias로 읽는다.
  - GEMV 분할(v2): program당 출력 8개, K block ≤ 1,024, grid = max(row program 수, ⌈GN/8⌉). row tile이 없는 program은 GEMV만 한다.
  - GEMV는 호출마다 다음 층의 int8 가중치를 한 번 더 읽는다. 후보 수와 무관하다(317M 층 1: 16.8 MB/호출).
- **구현 변경 이력(결과를 보고 고른 세부, E8 PLAN §4.3처럼 밝힘)**
  - v1(smoke만): GEMV가 row kernel의 program을 그대로 썼다(program당 출력 = next_pow2(GN/program 수)).
  - smoke v1의 보정 시간(quick_rep, 판정 아님)이 det+shift에서 5M N 512 +5.8%, 317M N 512 +9.3%였다. rowbench(`results/raw/k.1/latency/smoke_v1/rowbench.json`)로 보니 원인은 병렬 분할이었다.
    - 정책 rollout(24행)에서 GEMV program이 2~3개(5M, +13.9 µs/호출)·16개(317M, +73 µs)뿐이었다.
    - 8,192행에서는 program당 출력 1개의 1차원 내적이었다.
  - v2로 바꾼 뒤(18:44Z) 같은 rowbench: 24행 +1.4 µs(5M)·+32.9 µs(317M), 317M 512행 층 1 GEMV +27 µs(16.8 MB, 약 620 GB/s로 A40 대역폭 696 GB/s에 가깝다).
  - 본 측정은 v2다(`e17lat.py` md5 `e34c0ab2…`). v1 결과는 `latency/smoke_v1/`, `quick_v1/`에 남겼다.
  - 더 줄이려면 보정을 GEMM의 K 루프에 넣어 가중치를 다시 읽지 않는 구조가 필요하다. 이것은 설계의 "호출당 행렬–벡터 곱 하나"와 다른 구현이라 하지 않는다(§8 결정 항목).
- 정확성(smoke v2, `latency/smoke/checks.json`, ALL_PASS)
  - 기준: 같은 W8 가중치·같은 offset의 fp32 fake-quant det+shift, 층마다 teacher forcing
  - 코드 불일치: 5M 0, 317M 4.8e-7(1단계)
  - head 최대 차 ≤ 1.2e-6, c_eff 상대 차 ≤ 5.6e-8
  - forward마다 counter +1·새 offset(코드 26~31% 바뀜), 같은 행이면 같은 출력
  - MLP graph 재생 3회가 각자 기준과 맞는다. rollout graph 재생마다 counter dynamics +20·reward +18이다. det 재생은 비트 단위로 같고 eager와도 같다.
- 단위와 순서: 회전마다 (크기, N)별 짝 두 개 (det, det+shift)·(det, sh-iter), 12단위 24 block. 회전마다 numpy `default_rng(0)` 순열로 섞고 짝 안의 순서는 같은 rng의 `random() < 0.5`. 10회전.
- block(E8): warmup 30회 뒤 n회, 매회 CUDA event 쌍 + device sync, block 값 = 중앙값. n = max(10, ⌈2,000 ms / t_rep(det)⌉)로 짝의 두 block이 같은 n을 쓴다.
  - smoke v2 보정의 n: 5M 881·337·97, 317M 90·27·10(N 512·2048·8192)
  - 회전당 약 103초, 10회전 약 17.2분 + graph 18개 생성
- block마다 dither 상태를 base 7·(70000 + 회전) + 13으로 다시 둔다(E8). block마다 회당 시간 전부(npz), 시작·중간·끝 GPU 상태(clock, 온도, 전력, clock event 이유, 보이는 compute 프로세스), 호스트 load·CPU 사용률을 기록한다.
- **배타**: `run_lat.sh`가 runner가 살아 있으면 거부하고, MPS daemon을 멈추고, MPS 변수 없이 단일 프로세스로 돈다. `e17_lat.py main`은 시작 때 GPU에 다른 compute 프로세스가 보이면 거부한다(이 pod에서는 nvidia-smi가 프로세스를 보여 줌을 smoke에서 확인).
- [참고, 판정 아님] smoke v2 보정의 det 대비 시간(quick_rep): det+shift 5M +4.6%·+1.2%·+0.3%, 317M +6.7%·+3.2%·+3.4%; sh-iter 5M +0.6%·+0.5%·−0.2%, 317M +1.5%·+3.0%·+2.2%. 판정은 본 측정의 짝 계산으로만 한다.

### 3.2 폐루프 정수 kernel(K2): E7 융합 kernel + det+shift

- `code/e7lib.py`는 main `e7lib.py`에 정밀도만 더한 것이다. E7의 kernel·class·정밀도 분기는 한 줄도 바꾸지 않았다(점검 1).
- **intk det, intk sh-iter**: E7의 `IntMLP`와 kernel을 그대로 쓴다. E7 kernel은 k를 원래 실행 인자로 받았고, E7 class만 k = 8·4로 제한했다. k-bit 코드(0..2^k − 1)를 int8 그릇에 담는다(zp 0).
  - 층 0: fp32 cuBLAS + 융합 LN/Mish/양자화
  - 층 1: 융합 int8 GEMM + LN/Mish/양자화
  - 층 2: int8 GEMM + 역양자화(dynamics는 PyTorch LN·SimNorm)
  - 가중치는 층 1·2만 W8 출력 채널별 RTN이다.
  - 범위는 E0 per-tensor(`inputs/ranges/`, E1·E11과 같은 격자)다.
- **intk det+shift**(`IntMLP2`, v2 kernel)
  - offset: 층·forward마다 o = Philox(seed, 열) − ½. seed는 E7 sh-iter와 같은 식이라, 같은 호출에서 sh-iter가 쓸 u에서 ½을 뺀 값이다(qlib에서 det+shift의 o가 sh-iter의 u − ½인 것과 같은 성질).
  - 코드: clamp(rint(x − o), 0, nl)
  - 보정: 다음 층 bias c_eff = c + α ⊙ (Wq·o). 층 호출마다 `torch.addmv`를 한 번 부른다.
  - 양자화 kernel과 보정이 같은 o 버퍼를 읽는다.
  - 폐루프에서는 지연이 판정에 들지 않으므로 eager `addmv`를 쓴다.
- seed: episode마다 환경·random·numpy·torch를 episode seed로, kernel의 Philox base를 7·seed + 13으로 둔다(E1·E7). 계획기 난수열은 변형 사이에 같다. 정수 경로의 난수는 전역 난수를 쓰지 않는다(점검 2에서 확인).
- E11 fake-quant와 다른 점: 층 1·2 가중치가 W8이고, dither·offset 난수가 Philox다(fake-quant는 qlib의 개별 torch generator). 그래서 K2는 값이 아니라 방향(Δ의 부호와 CI 겹침)을 비교한다(설계 그대로).

### 3.3 미리 바꾼 BF16(K3)

- E8c의 bf16cast와 같다: job마다 dynamics·reward MLP 복사본의 모든 매개변수(가중치, bias, LayerNorm affine)를 한 번 bf16으로 바꾼다. 입력 [z, a]는 bf16, 출력은 fp32이고 autocast는 없다(LayerNorm·Mish·SimNorm도 bf16).
- 원래 module의 forward hook(Quantizer)은 복사본에서 뗀다. encoder·Q·π는 fp32다.

### 3.4 접은 per-channel 정수 구성(기술 항목)

- 범위: 설계 §0.6대로 pod에서 다시 구한다.
  - 보정 episode: E0 규약(seed 11000~11004, 체크포인트 1~3, 4과제 → 60 episode, `calib_lib.record_calib_episode`)
  - 행: 계획 시점 입력 12,000행 × horizon 3 step(E0 규약, `e5lib.collect`)
  - 단위별 [min_j, 99.99 백분위_j], hi_j ≥ lo_j + 1e-6(e5lib `perchan`)
  - 같은 행으로 다시 잰 per-tensor 범위와 E0 범위의 차를 `pcstats` 결과에 기록한다.
- 접기: W' = W·diag(Δ), bias' = b + W·lo(fp32 W). W'는 출력 채널별 int8 RTN이다.
- 코드: 단위마다 자기 격자의 det rint(x_j) 또는 sh-iter ⌊x_j + u_j⌋(u는 E7 일정)
- v2 kernel이 열별 lo·hi·1/Δ를 읽는다.

### 3.5 FP32 none

- 손대지 않은 모델(qlib none, `e7` 키 없음)이다. td 작업자의 none과 같은 코드 경로다.
- 과제마다 한 번 돌리고 부분 2~4가 함께 쓴다(R5).

## 4. §0.7 점검 결과

1. **코드 경로(통과, `checks/paths.json`)**
   - 손대지 않은 E7 `e7lib.py`(`code/inputs/e7_reference/`, md5 `38ab3cc8…`)와 새 `e7lib.py`가 같은 pod·같은 seed(109000)에서 episode를 비트 단위로 같게 냈다.
     - mw-push s1: int8 det 1793.8138, int8 sh-iter 1796.3145, int4 det 1773.3420, int4 sh-iter 1787.1777
     - mw-push s1 intk k = 3(기준 IntMLP를 같은 k로): det 1744.8678, sh-iter 1765.6379
     - cheetah-run s1 intk k = 5: det 86.2208, sh-iter 740.4875
   - v2 kernel의 det·sh-iter 반올림 코드가 E7 kernel과 비트 단위로 같다.
   - qlib.py·calib_lib.py·e5lib.py는 E1 원본 그대로다(md5 같음).
2. **새 변형의 단위 점검(통과)**
   - det+shift(`checks/shift.json`)
     - offset 1,024,000개: 평균 −0.00037(SE 0.00029), 분산/(1/12) − 1 = −0.21%, 범위 [−0.4999993, 0.4999987], 20칸 χ² 26.2(자유도 19)
     - 접은 정수 det+shift = 같은 W8·offset의 fake-quant det+shift: 코드 불일치 0, head 최대 차 ≤ 1.9e-5(척도 67), c_eff 상대 차 ≤ 6.5e-8
     - 원소별 오차 평균/Δ ≤ 6.2e-4, 분산/(Δ²/12) − 1은 −0.8% ~ +0.3%
     - 행 공유와 forward마다 새 offset을 확인했고, 전역 CPU·CUDA 난수가 바뀌지 않는다.
     - 폐루프 smoke는 같은 seed 반복이 같다. fp32 episode는 정수 job 앞뒤로 비트 단위로 같다.
   - per-channel 접기(`checks/pc.json`, 통과, pcstats 뒤 19:15–19:17Z): mw-push·cheetah-run s1 × k {2, 3} × {det, sh-iter} × dyn·rew 16경우
     - 기준: 같은 접힌 W8 가중치의 fake-quant per-channel 계산(층마다 teacher forcing)
     - 코드 불일치 ≤ 3.8e-6(15경우는 0), head 최대 차 ≤ 7.6e-6, 끝까지(teacher forcing 없음) 상대 L2 ≤ 1.1e-3
     - [기술] fp32 가중치 fake-quant per-channel(E12의 구성) 대비 상대 L2 3~11%: 접은 W' = W·diag(Δ)를 출력 채널별 W8로 양자화한 영향이다.
     - 폐루프 smoke(seed 109002, k = 3): mw-push det 1699.52(성공)·sh-iter 1785.68(성공), cheetah det 318.16·sh-iter 633.71. fp32 episode가 앞뒤로 비트 단위로 같다.
     - pcstats 12개(36,000행): 다시 잰 per-tensor 하한은 E0와 같다(차 0). 상한 차는 −2.52(cheetah s1 dyn0) ~ +1.00(cheetah s3 dyn1), 그 밖은 \|차\| ≤ 0.36이다. 보정 episode가 E0(A5000)와 비트 단위로 같지 않아 99.99 백분위 꼬리가 달라진 것이다(설계 §0.6의 기록 항목). 정수 PT 경로는 E0 범위를, PC 경로는 이 pod의 범위를 쓴다.
   - bf16p(`checks/bf16p.json`): fp32 대비 상대 L2 dyn 0.95%·rew 0.63%(mw-push). 폐루프 smoke(성공 1799.09 대 fp32 1800.05)와 복원을 확인했다.
   - 지연 경로의 det+shift: §3.1
3. **기준 성능(통과)**: 새 seed의 FP32 none 90 episode(seed 100000~100029 × 체크포인트 1~3) 대 E1 none(`analysis_pre/e17_summary.json`). 12과제 모두 10점(10%) 안이다.

| 과제 | none(이 pod) | E1 none | 차 |
|---|---|---|---|
| mw-push | 1.000 | 0.989 | +1.1점 |
| mw-pick-place | 1.000 | 0.989 | +1.1점 |
| mw-door-open | 1.000 | 0.967 | +3.3점 |
| mw-drawer-open | 1.000 | 1.000 | 0 |
| mw-button-press-topdown | 1.000 | 1.000 | 0 |
| mw-peg-insert-side | 1.000 | 1.000 | 0 |
| mw-window-open | 1.000 | 1.000 | 0 |
| mw-sweep-into | 0.967 | 1.000 | −3.3점 |
| cheetah-run | 900.9 | 900.8 | +0.02% |
| walker-walk | 984.6 | 983.5 | +0.12% |
| quadruped-walk | 971.7 | 975.5 | −0.40% |
| finger-turn-hard | 969.1 | 977.9 | −0.89% |

4. **처리량**(MPS 워커 8. E9가 같은 A40 종류·같은 CPU 할당 7.65코어에서 정한 수이고, 워커 수를 다시 쓸어 보지는 않았다)
   - pre 단계(bench 60, calib 60, none 1,080 episode)는 1,682초에 끝났다. 작업 시간 합 / 8 = 1,667초라 워커 8개가 거의 쉬지 않았다. GPU 사용률은 100%였고 전력 상한 298 W에 걸려 있었다.
   - episode당 초(`analysis_pre/throughput.json`, 워커 8 동시)

| 묶음 | fp32 none | int det | int sh-iter | int det+shift | bf16p |
|---|---|---|---|---|---|
| MW | 4.69 | 5.01 (1.07배) | 4.80 (1.02배) | 5.51 (1.17배) | 4.69 (1.00배) |
| DMC | 23.0 | 23.3 (1.01배) | 23.2 (1.01배) | 26.1 (1.14배) | 22.2 (0.96배) |

   - det+shift가 느린 것은 폐루프 경로가 층 호출마다 offset 생성 kernel과 eager `addmv`를 더 띄우기 때문이다(§3.2). 지연 판정은 §3.1의 graph 경로로 따로 한다.
   - per-channel 칸은 같은 반올림의 int 칸과 같다고 잡았다(같은 kernel 구조).
   - 남은 main 3,600 episode: 작업 47,428초 / 8 = **1.65시간**

## 5. 실행 순서와 배치

### 5.1 이미 돈 것(본 실행 전, §4 점검용)

- 점검 1·2(단일 프로세스, 18:35–18:40Z)
- 지연 smoke v1·v2와 quick(배타, 수치 무효)
- pre driver(18:47Z~, MPS 워커 8, `manifests/pre.jsonl`)
  - 처리량 bench: seed 108000+, 분석하지 않음, 60 episode
  - 보정 episode 60개
  - FP32 none 12과제 × 90(점검 3에 필요)
- pcstats 12개

### 5.2 본 실행

1. 지연 본 측정(배타, `run_lat.sh main`): pre driver가 끝난 뒤 MPS를 멈추고 단일 프로세스로 돈다. 약 18분이다.
2. 폐루프 main driver(`restart.sh main`)
   - calib·pcstats(이미 끝나 건너뜀) 다음에 `manifests/main.jsonl`을 돈다. 936 job, 4,680 episode 중 none 216 job은 pre에서 끝났다.
   - job = (과제, 체크포인트, 정밀도, k, 변형, 5 episode)이고 결과 파일은 `out/jobs/episodes/<id>.json.gz`다. 임시 파일 뒤 rename하며, 결과가 있으면 건너뛴다.
   - 순서: chunk 우선(모든 칸이 chunk 0을 받은 뒤 chunk 1), chunk 안에서 DMC 먼저(E9)
   - 실패한 job은 manifest마다 최대 3번 돈다(첫 실행 + 재시도 2번). `out/attempts/<id>.n`이 시작 횟수를 센다.
3. 표준 배치: `/workspace/atm/{code,out,big,logs}`
   - `status.sh`는 한 줄을 0.1초에 낸다.
   - `restart.sh`는 살아 있으면 아무것도 하지 않는다(확인함).
   - `out/`에는 job 결과, 점검, 지연, manifest, 분석만 둔다. 보정 npz는 `big/`에 둔다.

| 단계 | 내용 | episode·block | 예상 |
|---|---|---|---|
| pre(진행 중) | bench, calib, FP32 none, pcstats | 1,200 ep | 18:47Z → 약 19:20Z |
| 지연 본 측정 | 6조건 × 짝 2 × 10회전 | 240 block | 약 18분 |
| main | 부분 2 1,080, 부분 3 1,080, 부분 4 1,440 episode | 3,600 ep | 지연 뒤 약 19:42Z → 약 21:30Z(1.65 h + 꼬리) |
| 분석(단계 B) | pod에서 `analyze_e17.py`, `e17_lat_analyze.py` | | 수 분 |

## 6. 비용 추정 (묶음 상한 $6, 80% = $4.8)

| 항목 | 시간 | 비용(× $0.59 × 1.03) |
|---|---|---|
| 지금까지(18:31:09Z ~ 19:20Z: 설치, 점검, 지연 smoke, pre) | 0.82 h | $0.50 |
| 지연 본 측정 | 0.32 h | $0.19 |
| main | 1.70 h | $1.03 |
| 단계 B 분석·회수와 대기 | 약 0.25 h | $0.15 |
| **합** | **약 3.1 h** | **약 $1.9**(상한의 31%) |

- 처리량이 추정의 70%여도 약 $2.4다. 80%선 $4.8과 자동 종료(02:30:38Z) 모두 여유가 있다.
- main이 끝나는 대로(약 21:30Z) 단계 B를 시작하면 대기 비용이 준다($0.61/h).

## 7. 분석 코드 (본 실행 전에 만들어 끝까지 돌려 봄)

| 파일 | md5 | 건조 실행 |
|---|---|---|
| `analyze_e17.py` | `9f4d00816183d6c0b7c0076b43b2919f` | main manifest 936 job의 가짜 결과(`fake_e17.py`)로 끝까지 돌렸다(18:52Z). fake-quant CSV 없음, 일반 형식, td 작업자의 `td_episodes.csv` 형식 세 경우다. 두 형식의 K2·per-channel 비교가 같다. 수치는 무효이고 보지 않는다 |
| `e17_lat_analyze.py` | `f578ab10a829290bc6bdfe3535767759` | 지연 quick(1회전, n 3) 출력으로 끝까지. 수치는 무효 |
| `fake_e17.py` | `be05a3a94e08886d2bae3010fbad4569` | 건조 실행용 |
| `bench_e17.py` | `d4ad42f064ed1a2f9c3b17bec377c2e1` | 처리량 요약 |

- 본 실행 코드의 md5(맥북 `code/`와 pod `/workspace/atm/code/`가 같음)
  - 커널·실행: `e7lib.py` d2484e13…, `e17lat.py` e34c0ab2…, `e17_lat.py` 6b904afc…, `runner.py` 0aa93724…, `make_manifest_e17.py` c537de89…
  - 대기열: `driver.sh` 31804e6a…, `restart.sh` b173ab3d…, `status.sh` 62101fd3…, `qstat.py` b05752e1…, `run_lat.sh` 49d26cf4…
  - 점검: `e17_checks.py` c7f308b0…
  - E1·E8 원본 그대로: `e8lib.py` 7dd7b30d…, `qlib.py` 7934e1b2…, `calib_lib.py` 6f2c4b96…, `e5lib.py` 6583b0c3…

## 8. 편차·운영 선택(미리 밝힘)과 조정자가 정할 것

1. **det+shift 지연 구현**(§3.1): 설계의 "호출당 행렬–벡터 곱 하나"를 양자화 row kernel에 융합한 GEMV로 구현했다. smoke를 보고 GEMV 분할을 한 번 고쳤다(v1 → v2).
   - GEMV는 다음 층 가중치를 한 번 더 읽는다. 그래서 317M·N 512처럼 계산이 작은 조건에서는 오버헤드가 남을 수 있다(smoke 지시값 +6.7%).
   - 판정은 본 측정 그대로 한다.
   - 보정을 GEMM 안에 넣는 구현(가중치 재독 없음)은 하지 않는다. 필요하면 조정자가 기술 항목으로 따로 정한다.
2. **K2의 fake-quant 쪽**: td 작업자의 E11 칸(mw-push k3, mw-pick-place k3, cheetah k5, walker k3의 none·det·det+shift, seed 100000~100029 × 체크포인트 1~3)이 필요하다.
   - `analyze_e17.py --fq_csv`는 episode별 CSV(task, ckpt, seed, variant, k, success, ret[, range])를 받는다.
   - 형식이 다르면 단계 B에서 맞추거나 조정자가 합친다.
3. 정수 경로의 난수는 Philox이고 fake-quant와 다르다(E7과 같음). 가중치는 W8이다. 그래서 같은 seed라도 episode는 짝이 아니다(pod도 다르다).
4. 지연은 E8처럼 무작위 초기화 모델과 4-bit 코드로 잰다. 폐루프 k(3·5)와 코드 폭이 다르지만 산술 비용은 같다.
5. FP32 none 1,080 episode와 보정 episode는 점검 3과 per-channel 범위 때문에 PLAN보다 먼저 돌렸다. none은 점검 3 말고는 보지 않았다.
6. pre 단계의 bench episode(seed 108000+)와 smoke episode(109000+)는 분석하지 않는다.

## 9. 위험

- 지연 본 측정 동안 다른 일을 돌리지 않는다. 호스트는 다른 사용자와 같이 쓰므로 load가 높다(18:49Z 1분 load 109). graph 방식은 E8에서 호스트 load에 거의 영향을 받지 않았다.
- pod 자동 종료(02:30:38Z) 전에 main이 끝나지 않으면 같은 A40 종류의 k.1b에서 이어 돈다. pcstats는 job 결과(`out/`)에 범위가 있으므로 맥북 사본에서 되살린다.
- REST API 장애(18:20Z~): pod 확인과 삭제는 GraphQL 도구(`rp_gql.py`)로만 한다.
