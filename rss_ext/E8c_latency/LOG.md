# E8c 추가 지연 측정 작업 기록

- 기준: [../DESIGN_E8c.md](../DESIGN_E8c.md) (sha256 `2e9174b3…`, 10-31 고정), [../DESIGN_ext.md](../DESIGN_ext.md) §0, §1, §4
- 도구: `runpod_rss/` (pod 태그 `e8c`, 상한 $3 → 조정자 결정으로 $4.5, 스모크 뒤 추정 지출 중단선 $3.6)
- 시각은 모두 UTC이고, 이 작업자의 맥북 시계(`date -u`) 기준이다.

## 시간순

- 2026-10-31T15:53Z 작업 시작. 읽은 것: DESIGN_E8c(sha256 확인, `rss_ext/LOG.md`의 값과 같음), DESIGN_ext §0·§1·§4, E8 PLAN·REPORT·LOG·code, `runpod_rss` README·스크립트, PILOT_RULES §2–§4. tdmpc2 `332f19e`의 소스 9개(`tdmpc2.py`, `world_model.py`, `layers.py`, `math.py` 등)를 scratchpad로 받아 `_plan`의 캡처 가능성을 읽음.
- 2026-10-31T15:54Z `budget.py status`: 실행 중인 arx pod 없음(지출 추정 $10.79).
- 2026-10-31T16:00Z 시간 추정(스모크 전, E8 calib의 graph-WM 시간과 FLOP 비로): graph-step에서는 π·Q(317M은 8개 ensemble)가 rest 정밀도로 남는다. 그래서 317M N 8192 재생 1회가 fp32 약 2.1 s, int-det 약 1.5 s다. 설계의 warmup 30 + 최소 10회를 지키면 회전당 약 1,600 s, 10회전 약 4.5 h, 총 약 $3.2다. 상한 $3과 pod 수명 4 h를 넘으므로 pod를 만들기 전에 조정자에게 알림.
- 2026-10-31T16:04Z 조정자 결정 수신(조정자 기록은 `rss_ext/LOG.md`의 "16:40Z" 항목): 선택지 (A). 설계(가설, 칸, 10회전, warmup 30, block 규칙)는 그대로다. e8c 상한 $4.5(`caps.json` 확인), pod는 `TERM_H=6`으로 만든다. 스모크 뒤 실측 추정 총지출이 $3.6을 넘으면 본 측정 전에 멈추고 보고한다. 본 측정 자료가 생기기 전의 운영 변경이다.
- 2026-10-31T16:05Z `E8c_latency/` 생성. E8 `code/e8lib.py`·`e7dev.py`·`setup_pod.sh`를 복사함(e8lib md5 `7dd7b30d…`, E8과 같음; 정수 경로·row kernel·dither·fp32/tf32/bf16 정의·모델 차원·seed·범위 추정은 이 파일을 고치지 않고 그대로 import한다). E8 입력 `results/smoke/{gemm_cfg,ranges,selection}.json`을 `results/e8_inputs/`에 복사함(md5 `086e701b…`, `7de92843…`, `ed4e862c…`).
- 2026-10-31T16:15Z 코드 작성(맥북에서는 `py_compile`만 함): `code/e8clib.py`(bf16cast, int 층 0 TF32 고정, int-det-L0tf32, graph-step, 단위 목록, graph 생성), `code/e8c_smoke.py`(env, ranges, randperm, 기능 점검, 보정), `code/e8c_main.py`, `code/e8c_analyze.py`(본 측정 전 작성), `code/run.sh`. E8 `results/analysis/e8_summary.json`을 `results/e8_inputs/`에 복사(E8 대 E8c 기술 비교용).
- 2026-10-31T16:16:25Z pod `arx-e8c` 생성: id `oti534analhzml`, **NVIDIA A40**(첫 후보로 바로 잡힘), $0.59/h, SECURE, `TERM_H=6` → 자동 종료 22:15:52Z. driver 595.91.07, 96 vCPU, 503 GB RAM, GPU 46 GB, 전력 상한 300 W, 최대 SM clock 1,740 MHz(E8 pod와 같은 사양).
- 2026-10-31T16:16:49Z setup 시작(`code/setup_pod.sh`, E8과 같은 파일, `(setsid nohup … &)`).
- 2026-10-31T16:18Z setup 끝(약 1분): torch 2.7.1+cu126, triton 3.3.1, tensordict 0.8.3, numpy 1.24.4, hydra-core 1.3.2, omegaconf 2.3.0, matplotlib 3.10.9, nvidia-ml-py 13.615.71, tdmpc2 `332f19e4`(E8과 같은 판). 코드·입력을 pod로 보내고 md5 확인(e8lib `7dd7b30d…`, e8clib `470333f7…`, e8c_smoke `913569e4…`, e8c_main `709be017…`, e8c_analyze `83028e30…`).
- 2026-10-31T16:18:38–16:19:55Z smoke(env, ranges, randperm, checks; 75초, `results/smoke/`). 모두 통과.
  - 범위: 네 크기 모두 E8 `ranges.json`과 같음(1e-6 상대).
  - `torch.randperm`(Q `return_type='avg'`)은 CUDA graph로 캡처된다. 재생 2,000회에서 순서쌍 20/20개(num_q 5)와 56/56개(num_q 8)가 나왔다. χ² 14.3(자유도 19), 49.4(자유도 55)로 p 0.001 문턱 43.8, 93.2 아래다. 연속 재생의 95~99%에서 쌍이 바뀐다. Q(avg) graph = eager(차 0), 재생끼리 다름. 그래서 **바꿀 필요가 없고 원래 연산을 그대로 쓴다**.
  - 기능 점검용 agent: 같은 모델(seed 0)에서 0으로 초기화된 reward head와 Q head만 다시 무작위로 채움(E8 reward 점검과 같은 규칙). 공식 초기화 그대로면 모든 후보의 가치가 0이라 WM·Q 계산이 행동에 드러나지 않기 때문이다. 지연 측정은 E8처럼 공식 초기화 그대로다.
  - bf16cast: 매개변수 모두 bf16, 출력 fp32. 단일 forward 상대 L2(fp32 대비) dynamics 0.50~0.52%, reward 0.67~0.89%. bf16 autocast는 0.40~0.42%, 0.53~0.69%다. rollout z 0.68~0.71%(autocast 0.56~0.59%). graph-WM 재생 = eager(차 0).
  - int-det-L0tf32: 층 0 GEMM에서 TF32가 실제로 쓰인다(fp32 대비 상대 L2 2.9~3.0e-4). 층 0 코드 불일치 0.04~0.05%(최대 1단계), 층 1 0.6~1.4%. int-det는 전역 TF32를 켜도 출력이 같다(층 0 고정 확인). graph 재생 = eager(차 0).
  - graph-step 66건 통과(크기 4 × N 512, 5M은 N 2048·8192 추가, 각 11경로, 한 공유 pool, 무작위 순서): 같은 CUDA seed·`_prev_mean`·관측·dither 상태에서 eager `_plan`과 graph 재생의 행동·최종 MPPI mean이 **비트 단위로 같다**(차 0). seed를 다시 두지 않은 재생은 달라진다(새 MPPI 난수). int-shiter는 같은 seed에서도 재생마다 달라진다(새 u). dither counter는 재생마다 dynamics +20, reward +18이다. det·float는 같은 seed에서 비트 단위로 같다. rest-tf32는 같은 경로의 rest-fp32와 결과가 다르다(TF32가 실제로 쓰임).
- 2026-10-31T16:20:27Z smoke calib 시작(모든 조건을 본 측정처럼 만들고 회당 시간 측정).
- 2026-10-31T16:27:23Z smoke calib 끝(414초). graph 생성: graph-WM 84개 26초, graph-step 132개 113초(한 공유 pool), allocated 4.0 GiB, reserved 21.0 GiB, 최대 allocated 9.1 GiB. 회전당 추정 1,845초(graph-WM 343, graph-step rest-fp32 831, rest-tf32 583, eager-step 87). 10회전 + 생성 약 311분(5.18 h). 총지출 추정 약 $3.5로 중단선 $3.6 아래라 본 측정을 진행하기로 함(PLAN §9).
  - [기술] 보정에서 317M N 8192 graph-step은 rest-fp32 1.67~2.30 s, rest-tf32 1.48~1.66 s다. rest를 TF32로 해도 6% 정도만 빨라진다.
- 2026-10-31T16:29:40–16:30:01Z quick 실행(`e8c_main.py --quick --sizes 5`, 1회전, n 3, warmup 2; 수치는 쓰지 않음): 5M graph 54개 생성, 50단위 60 block, JSON·npz 저장 정상. 이어서 `run.sh analyze-quick`로 `e8c_analyze.py` 확인(표·CSV·그림 5장 정상, 칸이 3개라 판정은 "불성립"으로 나옴; 의도대로 12칸이 아니면 성립으로 세지 않음).
- 2026-10-31T16:30–16:31Z 개발용 점검 `code/dev_q_tf32.py`(317M, eager, TF32/fp32 시간비): N 512에서 Q(avg) 0.61, π 0.48, 단순 mm 0.35. N 8192에서 Q(avg) 0.85, Q(all) 1.00, π 0.42, 단순 mm 0.30. profiler(N 8192, TF32): Q의 cutlass s1688 TF32 GEMM 2회가 116 ms로 Q 184 ms의 63%다. 큰 N에서 rest-tf32 이득이 작은 것은 공식 vmap Q ensemble의 batched GEMM 때문이다(PLAN §10.8, 기술).
- 2026-10-31T16:31Z PLAN.md 완성(머리말의 "16:40Z"는 쓰는 동안 잡은 예상 시각이고, 실제 완성은 본 측정 시작 직전이다). sha256 `cb0acabf297ad75b4d8269fef199712a74dad17368a4f596e3a97649092b2cce`(본 측정 결과를 보기 전).
- 2026-10-31T16:31:48Z **본 측정 시작**(`run.sh main`, 로그 `/workspace/logs/main.log`, 결과 `/workspace/results/main/`). pod 코드와 맥북 코드의 md5가 같음(e8c_main `709be017…`, e8c_analyze `83028e30…`, e8clib `470333f7…`, e8lib `7dd7b30d…`). PLAN의 예상(16:45Z 시작)보다 13분 이르다. 종료 예상 약 21:43Z(pod 자동 종료 22:15:52Z).
- 2026-10-31T16:33Z `e8c_analyze.py`의 그림 하나(`e8c_overhead.png`)에서 범례 위치만 옮김(10% 선과 겹침; quick 출력에서 봄, 본 측정 결과를 보기 전). 계산은 바꾸지 않음. 새 md5 `fc651842…`로 pod에 다시 보냄.
- 2026-10-31T16:34:12Z 본 측정 graph 생성 끝(graph-WM 84개 24초, graph-step 132개 112초, allocated 4.0 GiB, reserved 21.0 GiB).
- 2026-10-31T17:04:49Z 회전 0 완료(1,837초, 추정 1,845초와 같음). 240 block, 짝 40개. 맥북으로 당겨 둠. 자료 품질 점검(판정 수치는 보지 않음): block 끝 SM clock 1,245~1,740 MHz(중앙값 1,575), 최고 온도 74℃, 최대 전력 316 W, clock event 이유 {0, 0x4(전력 상한)}, 호스트 1분 load 6.3~40.0(중앙값 9.4), 보이는 compute 프로세스는 이 측정 하나. block 안 p90/중앙값은 graph-WM 최대 1.014, graph-step 최대 1.056, eager-step 최대 1.079다. 종료 예상 약 21:40Z.
- 2026-10-31T17:35:27Z 회전 1 완료(1,838초). 맥북으로 당겨 둠(조정자 요청: 2~3회전마다 당김). 종료 예상 약 21:40Z.
- 2026-10-31T18:06:08Z 회전 2 완료(1,841초).
- 2026-10-31T18:36:46Z 회전 3 완료(1,838초). 맥북으로 당겨 둠(4회전, 960 block). 종료 예상 약 21:41Z.
- 2026-10-31T19:07:25Z 회전 4 완료(1,839초).
- 2026-10-31T19:38:05Z 회전 5 완료(1,840초). 맥북으로 당겨 둠(6회전, 1,440 block). 종료 예상 약 21:41Z.
- 2026-10-31T20:08:43Z 회전 6 완료(1,838초).
- 2026-10-31T20:39:21Z 회전 7 완료(1,838초). 맥북으로 당겨 둠(8회전, 1,920 block). 종료 예상 약 21:41Z.
- 2026-10-31T21:09:59Z 회전 8 완료(1,838초). 맥북으로 당겨 둠(9회전, 2,160 block). 종료 예상 약 21:41Z.
- 2026-10-31T21:40:37Z **본 측정 완료**: 10회전, 2,400 block, 짝 400개, 18,529초(회전당 1,837~1,841초, 추정 1,845초). 결과 `results/main/e8c_main.json`, `e8c_reps.npz`.
- 2026-10-31T21:40Z 분석(`run.sh analyze`, pod, 수 초): `results/analysis/`(e8c_summary.json, e8c_cells.csv, e8c_tables.md, 그림 5장). 사전등록 계산 그대로(`e8c_analyze.py` md5 `fc651842…`).
- 2026-10-31T21:41Z 결과·분석·로그·smoke(freeze 포함)를 맥북으로 당김. md5 확인: e8c_main.json `a2a67415…`, e8c_reps.npz `f8a1f142…`, e8c_summary.json `25e78bd7…`, e8c_cells.csv `2c31750e…`(pod와 같음). quick 결과는 수치가 무효라 두지 않음(E8과 같음).
- 2026-10-31T21:41:38Z `pod_down.sh e8c`. `budget.py status`: e8c **gone**, $3.29(상한 $4.5의 73%).
- 2026-10-31T21:50Z REPORT.md 작성. Write 도구가 보고서 파일 쓰기를 거부해서 Bash heredoc으로 씀(조정자 지시대로). 판정 수치는 맥북에서 원 block 목록으로 따로 다시 계산해 분석 결과와 같음을 확인함: L2c 0/12(1.067~1.516), S1 12/12(0.671~0.780), S2 0/12(1.013~1.123), S3 중앙값 −0.09~+0.66%, CI 상한 최대 +0.76%.
- 판정 요약: **L2c 불성립, S1 성립, S2 불성립, S3 성립**. E8 L1~L3은 같은 pod에서 그대로 재현(L1 12/12, L2 10/12, L3 성립). 미리 바꾼 BF16(bf16cast)이 graph-WM·graph-step의 모든 칸에서 가장 빠른 부동소수점이었다. 설계 §6의 L2c 불성립 문구("비슷하다")가 자료(int-det가 1.07~1.52배 느림)와 맞지 않아 조정자 결정이 필요함(REPORT §3).
