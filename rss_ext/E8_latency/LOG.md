# E8 지연 영역 지도 작업 기록

- 기준: [../DESIGN_ext.md](../DESIGN_ext.md) §0, §1, §4 (sha256 `c660a10d…`, 10-31T08:46Z 고정)
- 도구: `runpod_rss/` (pod 태그 `e8`, 상한 $3, 계획 상한 80% = $2.40)

## 시간순

- 2026-10-31T08:50Z 작업 시작. 설계·규칙·참고 코드(e7lib, e7_latency(_paired), qlib, E7·posthoc 보고) 읽음. tdmpc2 commit 332f19e의 소스 9개 파일을 scratchpad로 받아 구조 확인(MODEL_SIZE, parse_cfg, WorldModel, layers, TDMPC2._plan).
- 2026-10-31T09:07Z budget: 우리 쪽 pod는 `e10.1`(다른 작업자)만 RUNNING. 건드리지 않음.
- 2026-10-31T09:13Z 코드 작성: `code/e8lib.py`(일반 정수 경로 a/b, row kernel, WM rollout, 시간 측정), `code/e7dev.py`(E7 융합 kernel + device seed), `code/e8_smoke.py`, `code/e8_main.py`, `code/setup_pod.sh`, `code/run.sh`. 맥북에서는 py_compile만 함.
- 2026-10-31T09:13:59Z pod `arx-e8` 생성: id `30cyekyjii6pzs`, **NVIDIA A40**(첫 후보로 바로 잡힘), $0.59/h, SECURE, 자동 종료 13:13Z. driver 595.91.07, 96 vCPU, 503 GB RAM, GPU 46 GB, 전력 상한 300 W, 최대 SM clock 1,740 MHz.
- 2026-10-31T09:15Z setup(`code/setup_pod.sh`): torch 2.7.1+cu126, triton 3.3.1, tensordict 0.8.3, numpy 1.24.4, hydra-core 1.3.2, omegaconf 2.3.0, matplotlib 3.10.9, nvidia-ml-py 13.615.71, tdmpc2 `332f19e4`. 목록은 `results/smoke/../freeze.txt`(pod) → 당겨 옴.
- 2026-10-31T09:20Z 기능 점검(5M): 모든 경로가 돌아감. 짧은 graph-WM 측정(5M, N 512): fp32 4.14 ms, int-det 2.32 ms, int-sh-iter 2.47 ms(+6.5%).
- 2026-10-31T09:22Z **dither 구현 수정 1**(본 측정 전, 구현 세부): 처음 구현(row kernel의 모든 program이 device seed를 읽고 u 전체를 Philox로 그림)은 sh-iter가 5M·317M(N 512) 모두 약 +5%였다. 원인은 (i) 317M에서 row kernel 하나당 program 256개가 각자 4,096개 Philox를 다시 그림, (ii) 5M에서 seed load → Philox 사슬 지연. 그래서 **u를 (층, forward)마다 한 번만, 한 kernel 앞에서 그리도록** 바꿨다: 층 0 row kernel의 program들이 층 1의 u를 나눠 그리고, head kernel이 다음 forward의 층 0 u를 그리며, 층 1 row kernel의 program 0이 counter를 늘린다. 의미(층·forward마다 새 u 하나, 모든 후보 공유, device counter)는 같다. 재측정: 5M +0.8~1.7%, 317M +0.7~0.9%.
- 2026-10-31T09:26–09:34Z smoke v1(469초): 정확성 점검 모두 통과, GEMM 경로 선택은 네 크기 모두 (a), 본 측정 추정 108분. pod 쪽 원격 bash 하나가 ssh 채널을 붙잡아(`do_wait`) 로컬 작업이 끝나지 않음 → 그 bash(pid 3082)만 kill. 측정 프로세스는 setsid라 영향 없음. 이후 실행은 `(setsid nohup … &)` 꼴로 띄움.
- 2026-10-31T09:40Z **row kernel 배치 수정 2**(본 측정 전): rowbench에서 dither row kernel이 M 8192·W 512에서 det보다 7~12% 느림. `code/dev_rowtiles.py`로 program당 tile 수(TILES)를 쓸어 보니, 처음 규칙(program 약 256개가 되게 TILES를 키움; Philox 상각용이었음)이 det에도 나빴다(M 8192·W 4096: TILES 32 439 µs 대 TILES 1 292 µs). **det 시간 기준으로 TILES = 1로 고정**(8개 모양 모두에서 det 최솟값의 4% 안). 결과 `results/dev/rowtiles.json`. 바뀐 배치로 smoke의 checks·select·calib·rowbench를 다시 돌림(v1 결과는 pod `results/smoke_v1/`에 보관).
- 2026-10-31T09:46Z smoke v2(344초): 정확성 점검 모두 통과(크기마다 경로 a·b × det·sh-iter, 5M E7 kernel, MLP·rollout graph 재생), 선택은 다시 네 크기 모두 (a), 본 측정 추정 회전당 644초(약 108분).
- 2026-10-31T09:47Z `e8_main.py --quick`(1회전, n 3, warmup 2; 수치는 쓰지 않음)으로 본 측정 코드 경로와 `e8_analyze.py`·그림 확인: 154 block, 짝 35개, JSON·npz·표·그림 정상. graph 66개 생성 21초, 할당 3.6 GiB. 무거운 칸에서 GPU가 전력 상한(clock event 이유 비트 0x4)에 걸려 SM clock이 1,635~1,725 MHz로 내려감을 확인(block마다 기록됨).
- 2026-10-31T09:50Z PLAN.md 작성 완료(본 측정 전).
- 2026-10-31T09:51:14Z **본 측정 시작**(`run.sh main`, 로그 `/workspace/logs/main.log`, 결과 `/workspace/results/main/`). 종료 예상 약 11:40Z. pod 코드와 맥북 코드의 md5가 같음(e8lib.py `7dd7b30d…`, e8_main.py `6a452450…`).
- 2026-10-31T09:52Z PLAN.md sha256 `5f5cca2043f33bf2d80f6c836e56db8753ff62e936e971e2bd4038c46e6eaf06`(본 측정 결과를 보기 전).
- 2026-10-31T10:02Z 회전 0 완료(638초, 추정과 같음). 154 block, 짝 35개. block 끝 SM clock 1,305~1,740 MHz(중앙값 1,665), clock event 이유 {0, 0x4(전력 상한)}, 호스트 1분 load 10.3~16.1. 종료 예상 약 11:38Z. (판정 수치는 10회전이 끝난 뒤 분석 스크립트로만 계산한다.)
- 2026-10-31T10:05Z E8b 스크립트 `code/e8b_bench.py` 작성(E8 주 측정이 끝난 뒤, 지출이 상한 안이면 실행). 이미지의 PyTorch 연산만 씀: bf16·int8·fp8(텐서/행 scale)·NVFP4·MXFP4 GEMM(`torch._scaled_mm`), FP4 활성 fake-quant det 대 sh-iter(PyTorch 연산, eager와 CUDA graph).
- 2026-10-31T10:55Z 회전 0–4 완료(회전당 637~640초). 자료 품질 점검(판정 수치는 보지 않음): graph 방식 block은 안정적이다. block 사이 흔들림이 큰 것은 eager 방식뿐이다(호스트 의존; 최대 15%). 원격 대기 ssh 한 번이 exit 255로 끊겼지만 측정은 계속 돌았다.
- 2026-10-31T11:20Z 회전 0–7 완료. 종료 예상 약 11:38Z.
- 2026-10-31T11:38:04Z **본 측정 완료**: 10회전, 1,540 block, 짝 350개, 6,409초(회전당 637~640초). 결과 `results/main/e8_main.json`, `e8_reps.npz`. 회전 3·4·6에 호스트 1분 load가 최대 493까지 튀었다(77 block > 50). graph block 값은 영향이 없었고(≤ +0.2%), eager·step block은 최대 +15%였다.
- 2026-10-31T11:38Z 분석(`run.sh analyze`, pod, 수 초): `results/analysis/`(e8_summary.json, e8_cells.csv, e8_tables.md, 그림 4장). 사전등록 계산 그대로.
- 2026-10-31T11:38:56Z **E8b** pod `arx-e8b` 생성(`sqx2hkthmr49lx`, RTX 5090, $1.19/h, 이미지 runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404). E8 지출이 $1.46(상한 $3 안)이라 조건을 만족했다.
  - 이미지 Python에 numpy가 없어 스크립트를 표준 라이브러리(statistics, random)로 바꿨다.
  - 첫 실행은 FP4 양자화 graph 단계에서 PyTorch 2.9 generator 상태 오류("Offset increment outside graph capture")로 멈췄다. 그래서 eager의 RNG 사용을 모두 캡처보다 앞에 두도록 순서를 바꿨다.
  - 둘째 실행(11:43–11:49Z): GEMM과 eager 양자화는 완료. graph 양자화는 함수 안의 `torch.tensor(E2M1)`(host→device 복사)가 캡처를 깨뜨려 실패했다.
  - grid 텐서를 캡처 전에 만들어 두도록 고친 뒤 graph 양자화만 다시 돌림(11:52–11:53Z, `e8b_quant_graph.json`).
  - MXFP4 GEMM은 torch 2.9.1의 `_scaled_mm`이 지원하지 않는다(block 1×32 e8m0 scale은 float8, 즉 MXFP8만 됨). 오류를 기록했다.
- 2026-10-31T11:54:30Z `pod_down.sh e8b`(결과를 당겨 온 뒤). 11:56:17Z `pod_down.sh e8`(smoke_v1까지 당겨 옴. quick 결과는 수치 무효라 두지 않음). `budget.py status`: e8 `gone` $1.64, e8b `gone` $0.32.
- 2026-10-31T12:15Z REPORT.md 작성(판정 L1·L2·L3 모두 성립, E8b 절 포함). pod에서 돌린 개발용 점검 스크립트 두 개(기능 점검, dither 오버헤드 비교)를 `code/dev_t0_functional.py`, `code/dev_t1_overhead.py`로 남김. 지출 합계 $1.96(E8 $1.64, E8b $0.32), 두 pod 모두 gone.
