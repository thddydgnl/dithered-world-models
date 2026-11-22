# E17 정수 kernel과 BF16 폐루프 — 결과 보고

- 작성: 2026-11-21T21:20Z, k 작업자
- 기준
  - [../DESIGN_tmlr.md](../DESIGN_tmlr.md) v1.0 §0, §7, §8(sha256 `36d69ea8…0cc4`)
  - [PLAN.md](PLAN.md)(본 실행 전 완성, 19:18:33Z, sha256 `c814e588…5677`)
  - [LOG.md](LOG.md)
- 코드: [code/](code/)
- 결과: 모두 `results/raw/k.1/` 아래다(pod `/workspace/atm/out/` 그대로, 원격 md5 2,186개와 같음). 아래 경로는 이 폴더 기준이다.
  - 분석: `analysis/`(e17_summary.json, e17_cells.csv, e17_episodes.csv, e17_tables.md), `analysis_lat/`(e17_lat_summary.json, e17_lat_cells.csv, e17_lat_tables.md)
  - 원자료: `jobs/episodes/eval_*.json.gz`(936 job), `latency/main/e17_lat_main.json`·`e17_lat_reps.npz`
  - 점검: `checks/`, `latency/smoke/`, `latency/smoke_v1/`, `analysis_pre/`
  - pod 로그: `results/logs_k.1/`
- 표기
  - D: 같은 pod FP32 none 대비 손상. MW는 성공률 차(점), DMC는 1 − ΣR_v/ΣR_none(%)
  - CI: 짝 bootstrap 10,000회 95%(RandomState(0))
  - 지연 오버헤드: 짝 block의 (변형 − det)/det, 조건마다 10짝의 중앙값
  - 출처 표시: **[사전등록]**(설계 §7·PLAN §1의 판정), **[기술]**(설계·PLAN에 미리 적은 기술 항목), **[사후]**(결과를 본 뒤 정함)

## 1. 판정 [사전등록]

| 가설 | 판정 | 핵심 수치 [95% CI] |
|---|---|---|
| **K1** det+shift의 rollout 지연 오버헤드가 6조건 모두에서 중앙값 ≤ 5% | **불성립**(4/6조건) | 불통과: **5M N 512 +5.30% [+5.29, +5.33]**, **317M N 512 +7.23% [+7.11, +7.27]**<br>통과: 5M N 2048 +1.00% [+0.90, +1.25], N 8192 +0.64% [+0.58, +0.72]; 317M N 2048 +3.63% [+3.53, +3.72], N 8192 +2.95% [+2.75, +3.04] |
| **K2** 정수 det+shift가 4과제 모두에서 E11 fake-quant와 같은 방향 | **조정자가 td E11과 합쳐 판정** | 정수 쪽 Δ_int = D(det) − D(det+shift):<br>mw-push k3 +43.3점 [+33.3, +53.3]<br>mw-pick-place k3 +30.0점 [+17.8, +41.1]<br>cheetah-run k5 +30.3% [+27.0, +33.5]<br>walker-walk k3 **−14.0%** [−18.4, −9.6] |
| **K3** 미리 바꾼 BF16의 D가 12과제 모두에서 CI에 0을 포함 | **불성립**(9/12과제) | CI가 0을 포함하지 않은 과제:<br>mw-door-open +5.6점 [+1.1, +11.1]<br>cheetah-run +2.08% [+0.46, +4.26]<br>finger-turn-hard −1.42% [−4.00, −0.04](BF16이 더 좋음) |
| 기술: 접은 per-channel 정수 구성 | — | §5. E12 fake-quant per-channel과의 차는 조정자가 td 결과와 합친다 |

- 출처
  - K1: `analysis_lat/e17_lat_summary.json`(`K1`, `cells`)
  - K2·K3: `analysis/e17_summary.json`(`K2.integer_side`, `K3`, `tasks`)
- 판정 계산은 본 실행 전에 만들어 건조 실행한 `e17_lat_analyze.py`(md5 `f578ab10…`)와 `analyze_e17.py`(md5 `9f4d0081…`) 그대로다(`analysis/analysis_code_md5.txt`). 둘 다 pod에서 돌렸다.
- 다시 세기: 맥북에서 job 파일과 지연 JSON으로 핵심 수치 17개(D 11개, 지연 오버헤드 중앙값 6개)를 순수 Python으로 다시 계산했다(LOG 21:15Z). 모두 분석 출력과 같다.

## 2. K1: 지연 (graph-WM, E8 규약, A40 배타) [사전등록 수치]

- 측정: 19:18:46–19:36:11Z, 10회전 × 24 block(짝 12개), block 240개, 짝 120개(조건·변형마다 10개)
- 원자료: `latency/main/e17_lat_main.json`. 시작 때 GPU에 다른 compute 프로세스가 없었다.

| 크기 | N | int-det ms | int-sh-iter ms | int-det+shift ms | **det+shift 오버헤드 [CI]** | sh-iter 오버헤드 [CI] | det+shift 짝 차(ms, 중앙값) |
|---|---|---|---|---|---|---|---|
| 5M | 512 | 2.232 | 2.256 | 2.351 | **+5.30% [+5.29, +5.33]** | +1.04% [+1.00, +1.06] | +0.118 |
| 5M | 2048 | 6.011 | 6.035 | 6.072 | +1.00% [+0.90, +1.25] | +0.59% [+0.34, +0.81] | +0.060 |
| 5M | 8192 | 20.998 | 21.081 | 21.134 | +0.64% [+0.58, +0.72] | +0.39% [+0.29, +0.43] | +0.135 |
| 317M | 512 | 22.475 | 22.926 | 24.103 | **+7.23% [+7.11, +7.27]** | +2.01% [+1.61, +2.15] | +1.625 |
| 317M | 2048 | 75.580 | 76.850 | 78.321 | +3.63% [+3.53, +3.72] | +1.69% [+1.66, +1.79] | +2.740 |
| 317M | 8192 | 286.470 | 291.302 | 294.914 | +2.95% [+2.75, +3.04] | +1.67% [+1.59, +1.80] | +8.435 |

- 출처: `analysis_lat/e17_lat_cells.csv`, `e17_lat_tables.md`
- 짝 오버헤드는 회전 사이에 매우 고르다. 317M N 512의 10짝은 +6.81% ~ +7.35%, 5M N 512는 +5.25% ~ +5.35%다. CI 상한은 6조건 모두 10% 아래다(최대 +7.27%).
- **sh-iter**(E8 L3와 같은 꼴) [기술]: 6조건 모두 중앙값 ≤ 5%(최대 +2.01%), CI 상한 최대 +2.15%. E8(다른 pod)의 sh-iter 오버헤드는 같은 조건에서 +0.36% ~ +2.05%였다.
- **E8 pod와 비교**(같은 조건, 다른 pod) [기술]: T(int-det)/T_E8(int-det) = 1.007(5M N 512) ~ 1.016(317M N 512). 출처: `analysis_lat/e17_lat_summary.json`의 `cells.*.e8_pod_reference`, 원본 `rss_ext/E8_latency/results/analysis/e8_cells.csv`
- **오버헤드의 구성** [사후, 기술; smoke v2 rowbench `latency/smoke/rowbench.json`으로 어림]
  - det+shift는 sh-iter와 같은 offset 생성 비용에 더해 층 호출마다 다음 층 int8 가중치를 한 번 더 읽는 GEMV가 있다(설계의 "호출당 행렬–벡터 곱 하나").
  - 317M N 512의 det+shift − sh-iter 짝 차는 1.625 − 0.453 = 1.17 ms다. rowbench의 GEMV 증분으로 어림하면 step당 약 1.21 ms다(MPPI dynamics 18회 × (27.0 + 8.5) µs + reward 18회 × 27.0 µs + 정책 rollout 2회 × 약 41 µs). 두 값이 맞는다.
  - 이 GEMV는 후보 수와 무관하다(317M 층 1에서 16.8 MB, 약 620 GB/s). 그래서 계산이 작은 N 512에서 몫이 가장 크다.
  - N 8192에서는 det+shift − sh-iter 차가 3.66 ms로 더 컸다. GEMV program이 행 program과 겹쳐 있는 탓이다(rowbench M 8192 +93 µs/호출).
  - 5M N 512의 det+shift 짝 차는 +0.118 ms(sh-iter +0.023 ms)다. 이 µs 규모에서는 rowbench 잡음이 커서 나누지 않았다.
- **구현 변경(결과를 보고 고친 구현, PLAN §3.1)**
  - 본 측정은 GEMV 분할 v2로 했다(program당 출력 8개, grid = max(행 program, ⌈GN/8⌉)).
  - v1은 smoke만 돌렸다. 정책 rollout(24행)에서 GEMV program이 2~3개뿐이었고, smoke 지시값이 5M N 512 +5.8%, 317M N 512 +9.3%였다. 이것을 보고 본 측정 전에 고쳤다(18:44Z).
  - 조정자 결정(19:41Z)대로 보정을 GEMM 안에 넣는 다른 구현은 하지 않았다(설계의 구현이 아님).
- 측정 상태 [기술]: SM clock 1,275~1,740 MHz(중앙값 1,575), 온도 38~66 ℃, 전력 최대 303 W, 읽은 값의 83%가 전력 상한 이유(0x4)이고 열 이유는 0%였다. 호스트 1분 load 1.7~7.5, block CPU 사용률 1.5~10.6%. 자기 조건 T보다 15% 넘게 느린 block은 0%다.

## 3. K2: 정수 kernel 쪽 칸 (조정자가 td E11과 합쳐 판정)

- 4과제 × {FP32 none, int det, int sh-iter, int det+shift} × 90짝, MPPI 반복 6(기본)
- 정수 경로: E7 융합 kernel + E0 per-tensor 격자, W8 가중치. det+shift는 층마다 offset 보정 `addmv`를 쓴다(PLAN §3.2).

| 과제 | k | none | D(det) [CI] | D(sh-iter) [CI] | D(det+shift) [CI] | **Δ_int = D(det) − D(det+shift) [CI]** | D(det) − D(sh-iter) [CI] | red sh-iter / det+shift |
|---|---|---|---|---|---|---|---|---|
| mw-push | 3 | 1.000 | +45.6 [+35.6, +55.6] | +4.4 [+1.1, +8.9] | +2.2 [+0.0, +5.6] | **+43.3 [+33.3, +53.3]** | +41.1 [+30.0, +52.2] | 0.90 / 0.95 |
| mw-pick-place | 3 | 1.000 | +37.8 [+27.8, +47.8] | +11.1 [+5.6, +17.8] | +7.8 [+2.2, +13.3] | **+30.0 [+17.8, +41.1]** | +26.7 [+14.4, +38.9] | 0.71 / 0.79 |
| cheetah-run | 5 | 900.9 | +42.1% [+39.6, +44.8] | +11.4% [+10.2, +12.7] | +11.8% [+10.0, +14.0] | **+30.3% [+27.0, +33.5]** | +30.8% [+27.7, +33.8] | 0.73 / 0.72 |
| walker-walk | 3 | 984.6 | +57.1% [+53.4, +60.8] | +78.5% [+76.6, +80.3] | +71.1% [+69.0, +73.1] | **−14.0% [−18.4, −9.6]** | −21.4% [−25.6, −17.2] | −0.37 / −0.25 |

- 출처: `analysis/e17_summary.json`(`K2.integer_side`, `tasks.<과제>.cells`), `analysis/e17_cells.csv`. episode별 원자료는 `analysis/e17_episodes.csv`(task, ckpt, prec, k, variant, seed, success, ret, …)다.
- 조정자가 td E11과 합칠 때 쓸 값은 Δ_int와 그 CI다. td 쪽 `td_episodes.csv`는 `analyze_e17.py --fq_csv`가 그대로 받는다(가짜 자료로 확인, PLAN §7).
- [기술] walker-walk k3에서는 sh-iter와 det+shift 둘 다 det보다 해롭다. D_v − D_det의 CI 하한이 sh-iter +17.2%, det+shift +9.6%로 0.05를 넘는다.
  - E1에서도 walker-walk k3는 sh-iter가 해로운 칸이었다(`main/results/E1_tdmpc2/e1_tdmpc2_tables.md`).
  - 다른 세 과제에서는 det+shift와 sh-iter가 det 손상을 71~95% 줄인다(red 0.706 ~ 0.951).
- 정수 경로와 fake-quant의 차이(PLAN §3.2, §8): 정수 경로는 W8 가중치이고 난수가 Philox다. pod도 다르므로 episode끼리는 짝이 아니다. K2는 설계대로 방향(Δ의 부호와 CI 겹침)만 비교한다.

## 4. K3: 미리 바꾼 BF16 (12과제 × 90짝) [사전등록 수치]

| 과제 | none | bf16p | D [CI] | CI에 0 포함 | 비고 |
|---|---|---|---|---|---|
| mw-push | 1.000 | 1.000 | 0.0 [0.0, 0.0] | 예 | |
| mw-pick-place | 1.000 | 1.000 | 0.0 [0.0, 0.0] | 예 | |
| mw-door-open | 1.000 | 0.944 | **+5.6 [+1.1, +11.1]** | **아니오** | none만 성공 5쌍, bf16p만 성공 0쌍; 단측 p 0.0067 |
| mw-drawer-open | 1.000 | 1.000 | 0.0 [0.0, 0.0] | 예 | |
| mw-button-press-topdown | 1.000 | 1.000 | 0.0 [0.0, 0.0] | 예 | |
| mw-peg-insert-side | 1.000 | 1.000 | 0.0 [0.0, 0.0] | 예 | |
| mw-window-open | 1.000 | 0.967 | +3.3 [0.0, +7.8] | 예(하한 0) | none만 성공 3쌍 |
| mw-sweep-into | 0.967 | 0.989 | −2.2 [−5.6, 0.0] | 예(상한 0) | bf16p만 성공 2쌍 |
| cheetah-run | 900.9 | 882.2 | **+2.08% [+0.46, +4.26]** | **아니오** | 단측 p 0.0018 |
| walker-walk | 984.6 | 984.7 | −0.01% [−0.05, +0.04] | 예 | |
| quadruped-walk | 971.7 | 972.3 | −0.07% [−0.29, +0.15] | 예 | |
| finger-turn-hard | 969.1 | 982.9 | **−1.42% [−4.00, −0.04]** | **아니오** | BF16이 none보다 좋다(CI 상한 −0.035%) |

- 출처: `analysis/e17_summary.json`(`K3`, `tasks.<과제>.cells.bf16p`), `analysis/e17_tables.md`
- 규칙(CI가 0을 포함)으로 9/12과제라 K3은 불성립이다.
  - 손상 방향으로 0을 벗어난 과제는 mw-door-open(5.6점)과 cheetah-run(2.1%) 둘이다.
  - finger-turn-hard는 개선 방향으로 벗어났다.
- 구현은 E8c의 bf16cast와 같다: 모든 매개변수를 bf16으로 바꾸고 autocast를 쓰지 않으므로 LayerNorm·Mish·SimNorm도 bf16이다. 단위 점검에서 fp32 대비 상대 L2는 dyn 0.95%, rew 0.63%였다(`checks/bf16p.json`).

## 5. 접은 per-channel 정수 구성 (기술 항목) [기술]

| 과제 | k | det D [CI] | sh-iter D [CI] | D(det) − D(sh-iter) [CI] | red(sh-iter) |
|---|---|---|---|---|---|
| mw-push | 2 | +16.7 [+10.0, +24.4] | +3.3 [0.0, +7.8] | +13.3 [+4.4, +22.2] | 0.80 |
| mw-push | 3 | +3.3 [0.0, +7.8] | +1.1 [0.0, +3.3] | +2.2 [−2.2, +6.7] | 0.67 |
| mw-pick-place | 2 | +31.1 [+22.2, +41.1] | +18.9 [+11.1, +27.8] | +12.2 [0.0, +24.4] | 0.39 |
| mw-pick-place | 3 | +11.1 [+5.6, +17.8] | +4.4 [+1.1, +8.9] | +6.7 [−1.1, +15.6] | 0.60 |
| cheetah-run | 2 | +83.2% [+81.0, +85.3] | +80.3% [+78.9, +81.7] | +2.9% [+0.1, +5.6] | 0.04 |
| cheetah-run | 3 | +68.4% [+63.6, +72.8] | +38.9% [+34.5, +43.4] | +29.5% [+22.4, +36.1] | 0.43 |
| walker-walk | 2 | +37.5% [+33.1, +41.8] | +68.0% [+65.3, +70.8] | −30.6% [−35.9, −25.2] | −0.82(해로움) |
| walker-walk | 3 | +7.4% [+6.0, +8.9] | +5.3% [+4.3, +6.4] | +2.1% [+0.2, +4.0] | 0.28 |

- 출처: `analysis/e17_summary.json`(`pc_descriptive`, `tasks.<과제>.cells.intpc*`), `analysis/e17_tables.md`
- 범위는 이 pod의 보정 episode(E0 규약, seed 11000~11004)로 다시 구한 단위별 [min_j, 99.99 백분위_j]다(`jobs/pcstats/`). 다음 층에 접은 뒤 W8로 양자화했다.
  - 접힌 W8의 영향: fp32 가중치 fake-quant per-channel 대비 단일 forward 상대 L2 3~11%(`checks/pc.json`)
- [사후, 기술] 같은 pod의 per-tensor 정수 칸과 비교하면, k3 det 손상이 per-channel에서 크게 준다: mw-push +45.6 → +3.3점, mw-pick-place +37.8 → +11.1점. walker-walk는 PT k3 det +57.1%에서 PC k3 det +7.4%로 준다. E5의 "per-channel 범위만으로 3~4 bit det 손상이 대부분 사라진다"와 같은 방향이다.
- E12 fake-quant per-channel 칸(td)과의 차 D_int − D_fq는 조정자가 td 결과와 합친다(조정자 결정 19:41Z). 계산 경로는 `analyze_e17.py --fq_csv`에 있고, 두 pod는 독립 표본으로 다룬다.

## 6. §0.7 점검 (자세한 수치는 PLAN §4)

| 점검 | 결과 | 출처 |
|---|---|---|
| 1 코드 경로 | 통과. 손대지 않은 E7 `e7lib.py`와 새 판이 같은 pod·seed에서 episode 8쌍 비트 단위로 같음. v2 kernel의 det·sh-iter 코드가 E7과 같음 | `checks/paths.json` |
| 2 det+shift | 통과. offset 평균 −0.00037, 분산이 1/12와 −0.21% 차이. fake-quant와 코드 불일치 0, c_eff 상대 차 ≤ 6.5e-8. 오차 분산이 Δ²/12와 −0.8~+0.3% 차이. 전역 난수 불변 | `checks/shift.json`, `latency/smoke/checks.json` |
| 2 per-channel 접기 | 통과. 코드 불일치 ≤ 3.8e-6, head ≤ 7.6e-6, 끝까지 상대 L2 ≤ 1.1e-3 | `checks/pc.json` |
| 3 FP32 none | 통과. 12과제 모두 E1 none과 MW −3.3~+3.3점, DMC −0.89~+0.12% 차이 | `analysis/e17_tables.md`(점검 3 표), `analysis_pre/` |
| 4 처리량 | MPS 워커 8. pre 단계 기준 남은 main 1.65 h로 추정했고, 실제로는 1.30 h | `analysis_pre/throughput.json`, `analysis/throughput_all.json` |

- 무결성: 52칸 × 90 episode = 4,680 episode다. 모든 칸에서 seed 100000~100029 × 체크포인트 1~3이 빠짐없다. host 하나·GPU A40 하나, 문제 0. 실패한 job은 0이고 재시도도 0이다(`analysis/e17_summary.json`의 `integrity`).

## 7. 결과의 쓰임 (설계 §7에 미리 적은 분기, 문구 결정은 조정자·사용자)

- 설계: "K1·K2 성립이면 det+shift를 공동 처방으로 올릴 근거가 된다(올릴지는 사용자 결정)."
  - K1이 불성립이므로 K2 결과와 상관없이 이 조건은 채워지지 않는다.
  - 사실: det+shift의 정수 kernel 지연 오버헤드는 N ≥ 2048에서 0.6~3.6%다. 그러나 N 512에서는 5.3%(5M)·7.2%(317M)다. 원인은 설계의 offset 보정(층 호출마다 다음 층 가중치를 한 번 더 읽는 행렬–벡터 곱)이다.
  - sh-iter는 같은 측정에서 모든 조건 ≤ 2.0%다.
- 설계: "K3은 'BF16을 폐루프로 재지 않았다'는 한계 문장을 대신한다."
  - 폐루프로 잰 결과: 12과제 중 9과제는 손상이 없다(CI에 0 포함). mw-door-open 5.6점·cheetah-run 2.1%는 작은 손상이고, finger-turn-hard는 1.4% 개선이다.
  - 한계 문장을 "손상 없음"으로 바꾸는 근거로는 쓸 수 없다. 이 수치로 문장을 고치는 것은 조정자·사용자가 정한다.

## 8. 편차

1. **det+shift 지연 구현 v1 → v2**(결과를 보고 고친 구현). 본 측정 전 18:44Z에 고쳤다. PLAN §3.1, LOG 18:41Z
   - v1은 융합 GEMV가 row kernel의 program 분할을 그대로 써서 정책 rollout(24행)의 병렬성이 낮았다.
   - smoke 지시값(det+shift 5M N 512 +5.8%, 317M N 512 +9.3%)을 보고 분할만 바꿨다(program당 출력 8개).
   - v1 결과는 `latency/smoke_v1/`, `quick_v1/`에 있다.
   - 보정을 GEMM에 넣는 구현은 조정자 결정으로 하지 않았다.
2. **pod 생성**: 18:20Z부터 REST API(`rest.runpod.io`) 장애가 있었다.
   - 첫 생성은 재고 오류로 실패했다. `rp_gql.py`로 남은 pod가 없음을 확인한 뒤 두 번째에 잡혔다.
   - pod 확인과 삭제는 GraphQL로만 했다(조정자 알림 18:28Z).
3. **본 실행 전에 돈 것**: FP32 none 1,080 episode와 보정 episode 60개는 점검 3과 per-channel 범위에 필요해 PLAN 전에 돌렸다. none은 점검 3 말고는 PLAN 전에 보지 않았다. pre의 bench·smoke episode(seed 108000+, 109000+)는 분석하지 않았다.
4. **K2와 E12 비교**: 조정자 결정(19:41Z)대로 td CSV를 기다리지 않았다. 정수 쪽 수치와 episode CSV만 남겼다.
5. 운영
   - `qstat.py`의 eta 창을 driver 시작부터로 고쳤다(18:49Z).
   - `analyze_e17.py`가 td CSV 형식도 받게 했다. 본 실행 전에 고쳤고 건조 실행 뒤 md5를 PLAN에 적었다.
   - `code/`에서 돌리지 않는 기준 사본을 지웠다(18:58Z).
   - main이 지연 프로세스 종료 뒤 자동으로 시작되게 사슬을 걸었다(19:21Z).
6. 지연은 E8처럼 무작위 초기화 모델과 4-bit 코드로 잰다(PLAN §8.4). 폐루프 k(3·5)와 코드 폭이 다르지만 산술 비용은 같다.
7. [사후] 표시 항목(§2 오버헤드 구성 어림, §5 PT·PC 비교)은 결과를 본 뒤 결과 파일만 읽어 계산했다.

## 9. 비용·GPU·시간

| 항목 | GPU | pod | 시간(UTC) | 비용(budget.py 추정) | 묶음 상한 |
|---|---|---|---|---|---|
| E17 | NVIDIA A40 (SECURE, $0.59/h) | `atm-k.1` `0fwb5dot0ow6m2` | 18:31:09 – 21:16:08 (2.75 h) | **$1.67**(gone) | $6 (28%) |

- 내역
  - setup: 18:31:54–18:34Z, 147초
  - 점검 1·2: 18:35–18:40Z
  - 지연 smoke v1·v2와 quick: 18:40–18:46Z
  - pre(bench, calib, FP32 none, pcstats): 18:47:10–19:15:21Z
  - PLAN 완성: 19:18:33Z
  - 지연 본 측정: 19:18:46–19:36:11Z, 1,036초
  - main: 19:36:25–20:54:43Z, 4,697초, 720 job
  - 조정자 점검 대기: 20:54–21:14Z
  - 단계 B 분석: 21:14:31Z
  - 회수와 md5 대조: 21:15Z
  - 삭제: 21:16:08Z
- 결과를 당기고 원격 md5와 대조한 뒤 `pod_down.sh k.1`로 지웠다. REST가 실패해 GraphQL로 지웠다. `budget.py status`에서 k.1은 **gone**이고, `rp_gql.py pods atm-k` 목록은 비었다.
