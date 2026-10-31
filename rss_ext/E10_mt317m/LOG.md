# E10 mt80-317M 작업 기록 (시간순, UTC)

- 2026-10-31T08:49Z 작업 시작. DESIGN_ext.md §0·§3·§4, MAIN_RULES.md, PILOT_RULES.md §1–§4, runpod_rss/README.md, 참고 코드(qlib.py, runner.py, calib_lib.py, analyze_e1.py)와 E0·E1 보고를 읽었다. budget.py status: ours_est $0.00, 실행 중인 arx pod 없음.
- 2026-10-31T09:00Z tdmpc2 commit 332f19e 원본(envs/__init__.py, wrappers/multitask.py, common/parser.py, common/__init__.py, tdmpc2.py, world_model.py, layers.py, envs/metaworld.py, dmcontrol.py)을 GitHub raw로 읽음(맥북에서 읽기만). 요점
  - mt80: TASK_SET['mt80'] 80과제(DMC 30 + Meta-World 50), task_dim 96, 317M 설정 enc/mlp 4096, latent 1376, enc 층 5, num_q 8.
  - 다중 과제 env = 각 과제의 단일 과제 env(make_env)를 MultitaskWrapper로 묶음: 관측은 끝에 0을 붙여 최대 차원으로, 행동은 그 과제 차원까지만 잘라 env.step.
  - TDMPC2: act(task=idx) → task embedding을 encoder·dynamics·reward·pi·Q 입력에 붙임, _plan에서 행동 mask, 할인율은 과제별 episode 길이에서.
  - 과제 index(mt80): push 63, pick-place 56, door-open 41, drawer-open 44, button-press-topdown 32, peg-insert-side 53, window-open 73, sweep-into 71.
  - README가 가리키는 학습 때 task 순서(commit 7ec6bc8)의 common/__init__.py와 332f19e의 것이 같다(diff 없음).
- 2026-10-31T09:00Z 체크포인트: HF nicklashansen/tdmpc2 `multitask/mt80-317M.pt`, 2,014,203,797 B, LFS sha256 `38efd60ba341bff89282898201483e38c81030a6b716f152de7f82f5567b99cc`(pod에서 받아 대조할 값).
- 2026-10-31T09:00:48Z pod `e10.1` 생성(TERM_H=8, 종료 17:00Z): RTX 4090 24GB, 드라이버 595.91.07, vCPU 96, RAM 251GB, $0.89/h. `code/setup_pod_e10.sh`로 설치(약 2분): torch 2.7.1+cu126 등 E1 고정판, gym 0.21.0, Metaworld@04be337, mujoco-py 2.1.2.14(CPU/OSMesa 빌더, `MUJOCO_PY_FORCE_CPU=1`), tdmpc2 332f19e. 체크포인트 sha256 `38efd60b…99cc` 일치(HF LFS oid와 같음).
  - 작은 문제: `rsh.sh e10.1 'cd … && setsid nohup … &'`가 ssh를 붙잡아 Bash 호출이 시간 초과로 백그라운드로 넘어감(설치 자체는 정상). 이후 `cd &&` 없이 `setsid nohup bash -c "…" > log 2>&1 < /dev/null &`로 띄운다.
- 09:05Z `mt_dims.py`: 공식 `make_env(task=mt80)`으로 80개 env 생성(11.6초) → `mt80_dims.json`. 최대 관측 39(= MW), 최대 행동 6, MW mask [1,1,1,1,0,0], 할인 0.95. 체크포인트는 구 API(`_Qs.params.N`), `_action_masks` 없음, metadata(task 목록이 TASK_SET['mt80'] 순서와 같음, date 2023-08-13). agent 매개변수 317,383,129. hook 층 dyn0/rew0 = NormedLinear(1478→4096, Mish), dyn1/rew1 = (4096→4096, Mish).
- 09:07Z `test_wrapper.py`(mw-push, mw-door-open, seed 99000): 공식 MultitaskWrapper 경로와 mtlib 경로의 행동열이 비트 단위로 같음, return·성공 같음, 두 agent state_dict·할인율 같음 → **wrapper 확인 통과**.
- 09:08~09:12Z `smoke_e10.py`(seed 99000~99009): none 성공 push 10/10, door-open 9/10, pick-place 10/10. episode 7.65초, 결정당 73.2~73.6 ms(단일 프로세스, fp32·TF32 끔, GPU 사용률 94%). 기록기 켜도 episode 비트 동일. det·sh-iter k=3 호출 패턴(층마다 512행 18회, dyn 24행 2회), nq 7,600. 범위·pt_err1h 경로 동작. 산출물 `results/smoke/`.
- 09:13:58Z pod `e10.2` 생성(TERM_H=8, 종료 17:13Z): RTX 4090, 드라이버 595.91.07, vCPU 128, RAM 503GB. 같은 설치 스크립트로 설치(약 2분).
- 09:16Z PLAN.md 작성(본 실행 전). 예상 비용 약 $6.5~7.3, 두 pod 병렬 3.5~4시간.
- 09:16:39Z **본 실행 시작**(`code/run_e10.sh`, driver 작업자 2개): e10.1 = push, pick-place, door-open, drawer-open / e10.2 = button-press-topdown, peg-insert-side, window-open, sweep-into(e10.2는 먼저 `mt_dims.py`로 dims JSON 생성). 로그 `/workspace/logs/driver.log`, 상태 `/workspace/e10/state.json`.
- 09:18Z 작업자 2개의 처리량: 결정당 153~160 ms(두 프로세스가 GPU를 나눠 씀) → episode당 실효 7.8~8.0초(held는 기록기 포함). 단일 프로세스 스모크 7.65초와 거의 같다(차이 2~4%, 기록기의 작은 동기 복사 포함). eval 단계(기록기 없음)에서 다시 비교해 1개가 확실히 빠르면 되돌린다(PLAN §5).
- 09:37Z 진행: 8과제 held none 끝, **8과제 모두 자격**(held none 성공률 push 1.000, pick-place 0.967, door-open 0.867, drawer-open 0.700(= 문턱, 21/30), button-press-topdown 1.000, peg-insert-side 1.000, window-open 1.000, sweep-into 1.000) → 대체 목록 불필요. 범위: rew0 상한이 28~29로 넓다(5M은 7~8), dyn1 상한은 약 1.8. k 선택 진행 중. mw-push는 k=5에서 det 하락 0.233(≥ 0.20) → 1차 k = 5. 오류 없음. 지출 추정(E10) $0.94.
- 10:08Z 진행(1시간 점검): 오류 없음. 1차 k(held 하락): push 5(0.233), pick-place 3(k5 0.067, k4 0.100, k3 0.967), door-open 3(0.033, 0.067, 0.800), button-press-topdown 3(0.000, 0.000, 0.367), peg-insert-side 3(0.067, 0.000, 1.000). drawer-open(k5 0.133)·window-open(k5 0.067)은 k=4 진행, sweep-into는 k=5 진행. push eval 시작. 결과 job·범위·로그를 `results/raw_e10.{1,2}/`로 당김.
- 10:18Z 작업자 2개 대 1개 비교(eval, 기록기 없음): 2개일 때 결정당 161~168 ms(실효 80~84 ms), 단일 프로세스 스모크 det 76.5·sh-iter 77.4 ms → 2개가 4~8% 느리다. PLAN §5 규칙대로 작업자 1개로 되돌리기로 함.
- 10:20Z 되돌리는 중 운영 실수 2건(결과에는 영향 없음, GPU 유휴 약 10분씩 ≈ $0.3)
  1. 재시작 스크립트의 `pkill -f e10_driver.py`가 자기 자신(bash -c 명령줄에 같은 문자열)을 죽여 새 driver가 뜨지 않음.
  2. 부모가 죽은 ProcessPool 작업자 2개가 큐에 남은 job을 끝낸 뒤(결과 파일은 정상 기록: eval push det/shiter c3·c4) 큐에서 영원히 대기 → GPU 0%를 확인하고 kill.
- 10:32Z 두 pod에서 driver를 작업자 1개로 다시 시작(`run_e10.sh <tasks> 1`). 끝난 job은 건너뜀. e10.1 첫 job: eval push det c5, 결정당 76.7 ms, episode 8.0초. (참고: `run_e10.sh`의 "driver exit $?"는 `$(date)` 뒤라 항상 0으로 찍히는 로그 버그. 실행에는 영향 없음.)
- 11:02Z 진행(1시간 점검): 오류 없음. **k 선택 끝(8과제 모두 1차 k 있음)**: push 5, pick-place 3, door-open 3, drawer-open 4(k5 0.133, k4 0.267), button-press-topdown 3, peg-insert-side 3, window-open 3(k5 0.067, k4 0.167, k3 0.600), sweep-into 3(k5 0.000, k4 0.000, k3 0.200 = 문턱, 허용 오차 규칙으로 해당). eval 진행 중(작업자 1개, episode 약 8.0초). 지출 추정(E10) $3.46. 예상 종료 e10.1 약 12:45Z, e10.2 약 13:00Z. 결과를 `results/raw_e10.{1,2}/`로 당김. (10:52Z e10.2로의 ssh가 원격 sleep 중 한 번 끊김(exit 255); pod와 driver는 정상.)
- 12:02Z 진행(1시간 점검): 오류 없음. det·sh-iter eval은 e10.1 4과제 모두 끝, e10.2는 sweep-into만 남음. perr job 7개 끝(sweep-into 남음). 이제 none eval(과제당 90)을 돌리는 중. 예상 종료 e10.1 약 12:45Z, e10.2 약 13:00Z. 지출 추정(E10) $5.34(e10.1 $2.77, e10.2 $2.57).
- 12:42:57Z e10.1 driver 끝(DRIVER_DONE, 실패 0). job 수 held 24, ksel 54, eval 108, perr 4, ranges 4(맥북 사본과 pod 수 일치). `results/raw_e10.1/`에 job·범위·state·로그·dims·패키지 목록을 당긴 뒤 12:43Z `pod_down.sh e10.1` → budget.py에서 **gone**($3.40). held 기록 npz(120MB, 데이터)는 pod와 함께 지움.
- 12:54:12Z e10.2 driver 끝(DRIVER_DONE, 실패 0). 두 pod 합계 job 406개(held 48, ksel 126, eval 216, perr 8, ranges 8), episode 3,030개(held 240, k 선택 630, eval 2,160). 모든 job이 RTX 4090. det·sh-iter episode의 nq는 모두 7,600, 모든 episode 100 step. 두 pod의 dims JSON·패키지 목록 동일.
- 12:54Z 분석: e10.1 결과를 e10.2의 `/workspace/merged`에 올려 `analyze_e10.py --w1_since 2026-10-31T10:30:00Z`(지연 요약은 작업자 1개 시기의 eval episode만)로 판정 계산 → `results/analysis/`(e10_summary.json, e10_judgement.json, e10_qualification.csv, e10_ksel.csv, e10_eval.csv, e10_episodes.csv, e10_perr_episodes.csv). 판정: G1 성립(8/8 과제 하락 실재), G2 성립(red ≥ 0.5 8/8, 해로운 과제 0), G3 성립(8/8에서 det pt_err1h < sh-iter).
- 12:55:34Z `pod_down.sh e10.2` → budget.py에서 **gone**($3.39). **E10 pod 모두 gone.** E10 지출 추정 합계 $6.79(e10.1 $3.40 + e10.2 $3.39; 상한 $13의 52%).
- 13:05Z REPORT.md 작성(판정 G1·G2·G3 모두 성립, [사후] 항목 2개 표시). 맥북 `code/__pycache__`(로컬 py_compile 부산물) 삭제. 최종 budget.py: e10.1·e10.2 모두 gone, E10 합계 추정 $6.79.
