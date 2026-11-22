# E14 실행 계획 — mt80-317M 기전 대조 (본 실행 전 작성)

- 작성: 2026-11-21T18:45Z, mt 작업자. 기준: [../DESIGN_tmlr.md](../DESIGN_tmlr.md) v1.0 §0·§4 E14·§8 (sha256 `36d69ea8…0cc4`, 고정값과 같음), [../WORKER_RULES.md](../WORKER_RULES.md).
- 순서에 관한 사실: mt.1의 본 실행 driver는 18:33:31Z에 시작했고, 이 문서는 그 뒤에 마무리했다. 운영 선택(job 구성, 우선순위, 관문, 재시도, 분석 규칙)은 driver 시작 전에 코드로 고정돼 있었다. 그 뒤 코드 변경은 하나다. 18:39Z에 bclib의 규칙 파일 인자를 더했고, 기본값에서는 계산이 같다. 이 문서를 쓸 때 본 결과는 held seed의 none 성공과 b 잔차, 처리량 확인 중 driver 로그에서 본 door-open none 첫 세 job의 성공 수(5/10, 9/10, 10/10, 18:46Z)뿐이다. 그것으로 정한 것은 없다.
- 가설(P1, P2), 지표, 문턱, 칸, seed는 설계 그대로다. 이 문서는 운영 세부(구현 선택, §0.7 점검 결과, 처리량, pod 배치, 비용)만 적는다.
- 기록: [LOG.md](LOG.md) · 코드: [code/](code/) (pod의 `/workspace/atm/code/`와 같음, md5는 §7)

## 1. 칸과 seed (설계 그대로)

| 과제 | mt80 index | 1차 k (E10, R4) | 범위 (E10 per-tensor [min, 99.99 백분위]) |
|---|---|---|---|
| mw-push | 63 | 5 | `code/inputs/ranges/ranges_mw-push.json.gz` |
| mw-pick-place | 56 | 3 | 같은 폴더 |
| mw-door-open | 41 | 3 | 같은 폴더 |
| mw-drawer-open | 44 | 4 | 같은 폴더 |
| mw-button-press-topdown | 32 | 3 | 같은 폴더 |
| mw-peg-insert-side | 53 | 3 | 같은 폴더 |
| mw-window-open | 73 | 3 | 같은 폴더 |
| mw-sweep-into | 71 | 3 | 같은 폴더 |

- 1차 k는 `rss_ext/E10_mt317m/results/analysis/e10_ksel.csv`와 대조했다(driver가 시작할 때 다시 assert). 범위 파일은 E10 job 결과(`raw_e10.{1,2}/jobs/ranges/`)를 그대로 복사했고 `scales/<task>.json`과 같다.
- 변형: none, det, sh-iter(`shiter`), sh-ep(`shep`), det+shift(`detshift`), det+BC(`detbc`).
- eval seed 102000~102089(체크포인트 1개 → 변형마다 90짝). held seed 103000~103009(det+BC의 b). 스모크 seed 108000~108001(점검용, 분석하지 않음).
- episode: eval 8 × 6 × 90 = 4,320, held 8 × 10 = 80.

## 2. 구현 (`code/`)

- **qlib.py** = E10 판(sha256 `1edda3b2…ca59`) + det+BC 변형 `detbc` 하나. 바뀐 곳은 4군데뿐이다(`diff code/e10_ref/qlib.py code/qlib.py`).
  1. 이름: `E14_VARIANTS = ("detbc",)`, 별칭 `det+BC`, `canon_variant`가 기존 검사 뒤에 이 이름을 받는다.
  2. `Quantizer.bias = {}`(job 단위로 넣는 층별 (4096,) 벡터, qrange와 같은 방식)
  3. hook의 맨 끝 분기(기존 분기 모두 뒤, `else` 앞): `lo + Δ·clamp(round(x), 0, nl) − b[층]`. 앞부분은 det와 같은 연산 순서라 b = 0이면 det와 비트 단위로 같다(점검 2a). 난수를 뽑지 않는다.
  4. 문서 문자열
- **mtlib.py, mt_dims.py**: E10 판 그대로(md5 같음). episode 규약도 E10(= E1)이다. 매 episode env를 seed로 새로 만들고, `seed_all(seed)`, dither generator는 7·seed + 13, `eval_mode=True`, 성공 = 마지막 step의 `info['success']`.
- **bclib.py**(§0.6의 b 절차)
  - 자료: 그 과제의 held none 10 episode(seed 103000~103009)를 E10의 `HeldRecorder`로 기록한 행. step마다 root latent와 MPPI 반복마다 무작위 후보 16개다.
  - **기록한 행 전부(10 × 100 step × 6 반복 × 16 = 96,000행)를 쓴다.** E0·E10의 12,000행 부분표본은 백분위 범위를 위한 규칙이었다. b는 평균이라 행이 많을수록 잡음만 줄고 비용은 거의 없다(운영 선택).
  - 굴림: 같은 시작 latent·같은 행동열로 3 step. step마다 보상 머리 다음 dynamics(TDMPC2._estimate_value와 같은 순서), task embedding 포함.
  - 보정된 양자화 출력은 양자화 hook **뒤에** 등록한 두 번째 forward hook(OutTap)으로 읽는다. 그래서 값은 폐루프에서 다음 층이 받는 값 그대로다.
  - 층마다 (보정 출력 − 비양자화 출력)의 단위별 평균(모든 행·모든 step, float64 누적)을 b에 더한다. 이것을 세 번 한다.
  - 네 번째 패스는 마지막 b로 남은 평균만 잰다(= "세 번째 sweep 뒤 남은 평균", 점검 2의 판정값). 번마다 단위별 평균 |m|/Δ, 최대, 부호 평균, MSE/Δ²를 기록한다.
  - b와 sweep별 평균 벡터는 `out/bias/bias_<task>_k<k>.npz`, 보고는 `out/jobs/bias/bias_<task>_k<k>.json.gz`에 둔다.
- **mt14_jobs.py / mt14_driver.py**(E10 job·driver를 고친 재개 가능 대기열)
  - job: held 5 episode × 2, bias 1, eval 10 episode × 9(변형마다). 과제당 57 job, 550 episode.
  - job마다 결과 파일 하나를 원자적으로 쓰고(임시 → rename), 결과가 있으면 건너뛴다.
  - episode 기록: seed, success, success_any, return, steps, nq, n_refresh, 결정당 지연, 행동열 sha1, **episode 끝의 전역 torch CPU·CUDA 난수 상태 sha1**(변형 사이 같아야 함, 분석에서 전수 점검), 초
  - 우선순위: held → bias → none → det → sh-iter → sh-ep → det+shift → det+BC. 같은 순위 안에서는 과제 순서, 그다음 chunk 순서다. P1에 쓰는 칸(none, det, sh-iter)이 모든 과제에서 먼저 끝난다.
  - **관문(설계 §0.7의 점검 가운데 본 실행 자료로만 할 수 있는 것; 실패하면 그 과제의 해당 job만 멈추고 status에 `HOLD`로 보인다)**
    - 점검 3: 과제의 none 90 episode가 끝나고 |none − E10 none| ≤ 0.10(+1e-9)이어야 그 과제의 변형 job이 시작된다. 아니면 조정자 결정(`out/gate/release_check3_<task>` 파일)까지 멈춘다. 결과는 `out/gate/check3_<task>.json`이다.
    - 점검 2(det+BC): 과제의 bias job의 잔차(세 번째 sweep 뒤)가 모든 층에서 Δ의 1% 미만이어야 det+BC eval이 시작된다. 아니면 `out/gate/release_bc_<task>`까지 멈춘다.
  - 실패: 실패한 job은 두 번까지 다시 넣는다(시도 수 = `out/errors/<id>.<n>.txt` 개수). 깨진 worker pool은 다시 만든다(실행당 5번까지).
  - pod당 driver 하나(`logs/driver.lock`의 flock), pid는 `logs/driver.pid`, 상태는 `out/status.json`(60초마다 갱신)과 `out/state.json`(과제별)이다.
- **status.sh**: WORKER_RULES §3의 한 줄 형식이다. `done`은 job 단위(과제당 57)다. note에 HOLD·SLOW·실패 수, 최근 episode 초, 진행 중 job을 적는다.
- **restart.sh**: driver가 살아 있으면 아무것도 하지 않는다. 죽었으면 그 process group(setsid, pgid = 옛 driver pid)에 남은 우리 python 프로세스(spawn worker)만 끄고 새로 띄운다. `pkill -f`를 쓰지 않는다(E10 교훈). `--force`는 살아 있어도(멈춤 등) 끄고 다시 띄운다. 띄우기는 `setsid nohup … &`(앞에 `cd … &&` 없음)이다.
- **점검·보조**: check_paths.py(점검 1), check_new.py(점검 2), run_jobs.py(점검 단계에서 본 실행 job을 직접 실행), run_checks.sh, make_fake_e14.py(분석 시험용 가짜 자료), setup_pod_mt.sh(E10 설치와 같은 고정판, 경로만 `/workspace/atm`), env_mt.sh

## 3. §0.7 점검 결과 (mt.1, RTX 4090, 2026-11-21T18:25~18:33Z)

1. **코드 경로: 통과.** E10의 손대지 않은 qlib.py·mtlib.py(`code/e10_ref/`)와 E14 판을 같은 pod에서 두 프로세스로 돌렸다(seed 108000; door-open k=3, drawer-open k=4 × none·det·sh-iter·sh-ep·det+shift).
   - 10쌍 모두 행동열이 float32 바이트 단위로 같다. return 같음, nq 같음(7,600 / none 0), sh-ep n_refresh 같음(1), episode 끝의 전역 난수 상태 같음
   - 근거: `results/raw/mt.1/smoke/check1.txt`(pod `out/smoke/`). pod에서 만든 mt80 dims JSON도 E10 사본과 같다.
2. **새 변형(det+BC)과 sh-ep·det+shift: 일부 실패(det+BC의 b).** 근거: `out/smoke/check2.json`(seed 108001, door-open k=3)
   - 통과
     - (a) det+BC(b = 0)가 det와 비트 단위로 같다.
     - (b) 계획기의 전역 난수 불변: none, det, det+BC(b = 0과 실제 b), sh-iter, sh-ep, det+shift의 episode 끝 torch CPU·CUDA 난수 상태가 모두 같다.
     - (c) sh-ep: 한 episode 100 step 동안 표(층마다 (3, 4096))가 바이트 단위로 그대로이고 n_refresh = 1, 다른 episode에서는 새 표. 호출 패턴은 제어 step마다 512행 × 18, 24행 × 2(dyn)로 E10과 같다.
     - (d) det+BC(실제 b)의 nq 7,600. det+shift는 점검 1에서 E10과 비트 단위로 같다.
   - **실패(설계 규칙): det+BC 잔차.** 세 번째 sweep 뒤 단위별 남은 평균의 절댓값 평균 ÷ Δ가 dyn0 0.220, dyn1 1.264, rew0 0.043, rew1 0.166이다(문턱 0.01). 반복이 수렴하지 않는다. 자세한 수치와 진단은 §8.
   - **D2 뒤(19:20Z): 통과.** 최종 b에서 판정한다. 8과제 모두 수렴했고 모든 층 r < 0.01, 최악 층은 0.0063~0.0099다.
     - α = ½로 수렴: door-open 12번, push 11, pick-place 11, button-press 13, peg-insert 13, sweep-into 10
     - α = ¼로 수렴: drawer-open 21번, window-open 21번
     - α = 1은 8과제 모두 60번 안에 못 미쳤다. `bc_unconverged`는 없다.
3. **기준 성능(none 대 E10 none): 본 실행 자료로 판정한다.** 각 과제 none 90 episode(본 실행 job)가 끝나면 driver 관문이 판정한다. 문턱은 |none − E10 none| ≤ 0.10이다.
   - E10 none: push 1.000, pick-place 0.978, door-open 0.833, drawer-open 0.789, 나머지 4과제 1.000
   - 실패한 과제는 변형 job을 멈추고 조정자에게 알린다. 결과: `out/gate/check3_<task>.json`
   - E10 기준값은 E10 summary의 정확한 값(`code/inputs/e10_eval_none_exact.json`)이다. csv는 유효숫자 4자리라 경계 사례(예: 62/90)가 뒤집힐 수 있어 18:53Z에 바꿨다.
   - **통과(지금까지 3과제)**: door-open 0.856(77/90) 대 E10 0.833(+0.022), drawer-open 0.811(73/90) 대 0.789(+0.022), button-press-topdown 1.000 대 1.000. 나머지 5과제는 각 pod의 none이 끝나는 대로 관문이 판정한다.
   - 문턱 근처의 위험: door-open·drawer-open(none 0.79~0.83)은 90 대 90 비교에서 표본 잡음만으로도 10점을 넘을 확률이 각 약 7~10%다.
4. **처리량: 워커 1.** RTX 4090에서 단일 프로세스, GPU 단독으로 쟀다.
   - 결정당 지연(점검 2 episode, 중앙값): none 70.2 ms, det 73.6, det+BC 73.9, sh-iter 74.3, sh-ep 74.1, det+shift 75.0 ms
   - episode: held(none + 기록기) 7.4초, 본 실행 첫 job 7.4~7.5초. GPU 사용률 92~95%다.
   - 같은 기종에서 E10은 작업자 2개가 1개보다 4~8% 느렸다. 그래서 pod당 워커 1개로 둔다. 본 실행 첫 10분의 값은 §4에 덧붙인다.

## 4. 처리량과 시간 추정

- episode 초(추정): none 7.3, det 7.7, sh-iter 7.7, sh-ep 7.7, det+shift 7.8, det+BC 7.7(결정당 지연 × 100 + env 등 약 0.3초). 변형 평균 약 7.65초다.
- pod당(4과제): eval 4 × 540 × 7.65초 + held 40 × 7.4초 + bias 4 × 10~40초 ≈ **4.7시간**
- 처리 순서 때문에 det+BC(pod당 360 episode, 약 0.77시간)가 마지막이다. det+BC를 빼면 pod당 약 3.9시간이다.

## 5. pod 배치와 예상 종료 시각 (A5000 재고 없음 → 설계 §8.1의 대체, 조정자 승인 18:27Z)

| pod | GPU, $/h | 생성 | 자동 종료 | 과제 | 본 실행 시작 | 예상 끝(det+BC 포함) |
|---|---|---|---|---|---|---|
| mt.1 | RTX 4090, $0.89 | 18:21:23Z | 2026-11-22T03:21:23Z | door-open(k3), drawer-open(k4), push(k5), pick-place(k3) | 18:33:31Z | 약 23:15Z |
| mt.2 | RTX 4090, $0.89 | 18:39:52Z(ledger 18:40:10Z) | 2026-11-22T03:39:51Z | button-press-topdown(k3), peg-insert-side(k3), window-open(k3), sweep-into(k3) | 18:44:13Z | 약 23:40Z |

- mt.2의 4090은 조금 느리다(held none 결정당 74.5 ms, mt.1은 71.4 ms; episode 7.8초 대 7.4초). 예상 끝은 status ETA다.
- 한 과제의 held·bias·모든 eval은 한 pod에서 돈다(짝 규칙). 자동 종료 전에 끝나지 않으면 같은 4090 종류의 새 pod(mt.1b 등)에서 `out/`을 옮겨 이어 돈다.
- det+BC eval은 b 결정이 날 때까지 관문이 막는다(§8). 결정 전에 다른 job이 다 끝나면 driver는 HOLD 상태로 기다린다(60초마다 다시 봄).

## 6. 비용 추정 (budget.py 규약, $/h × 1.03)

- 본 실행 끝까지: mt.1 약 4.9 pod-h($4.5) + mt.2 약 4.8 pod-h($4.4) = **약 $8.9**
- 끝난 pod를 결과를 당긴 직후 지우고, 분석은 늦게 끝나는 pod 하나에서 돌린다(조정자 지시, 18:41Z). 유휴·분석을 더하면 **약 $9.1~9.4**다.
- mt 묶음 상한은 $15(18:41Z, $12에서 올림)이고 80% 선은 $12다. 추정은 그 안이다. det+BC를 빼면 약 $7.5다.
- 지금까지(18:41Z) 약 $0.4

## 7. 분석 코드와 코드 목록

- `code/analyze_e14.py` md5 `5ac8af6599d3ccc4b522d30d17a17e0e`(19:05Z, D2 반영). 본 실행 자료가 생기기 전에 썼다.
  - 19:05Z: D2에 맞춰 bias 표, det+BC 기술 항목 두 가지, `bc_unconverged` 열을 더했다. 그 판(표시 과제가 있는 가짜 자료)으로 다시 끝까지 돌렸다(`out/smoke/fake_analysis3.txt`). P1·P2 규칙은 그대로다.
  - 첫 판은 md5 `9fd6eb7f…`였다. 18:53Z에 바뀐 것은 점검 3의 E10 기준값을 csv(유효숫자 4자리) 대신 E10 summary의 정확한 값에서 읽는 것 하나다. 판정 규칙은 그대로이고, 바꾼 뒤 가짜 자료로 다시 돌렸다.
  - mt.1에서 가짜 자료(8과제 × 6변형 × 90, 두 pod 배치)와 det+BC 하나를 뺀 불완전 자료로 끝까지 돌렸다(`out/smoke/fake_analysis*.txt`). 불완전 자료에서는 P1·P2 판정이 None이다.
  - 판정 규칙
    - E10과 같은 짝 bootstrap 10,000회, `RandomState(0)` 색인 하나, 단측 p < 0.05, 8과제 Holm(엄격한 <), D_det > 0
    - P1: 손상 과제 ≥ 6, 그 가운데 red(sh-iter) ≥ 0.5인 비율 ≥ 0.75
    - P2: 손상 과제의 과반(2·n > n_손상)에서 D(sh-ep) − D(sh-iter)의 CI 하한 > 0
    - 기술 항목: r = D(sh-ep)/D(det)와 CI, 변형별 red·해로움(CI 하한 > 0.05), 점검 3 표, b sweep 표, 무결성(nq 7,600, 100 step, sh-ep n_refresh 1, 변형 사이 전역 난수 상태 같음)
- 분석은 단계 B에서 pod 위에서 돈다(맥북에서 돌리지 않는다).
- 코드 md5(맥북 = mt.1 = mt.2, 19:05Z 파일 묶음 md5 `f149fb1e…`)

| 파일 | md5 |
|---|---|
| qlib.py | bc85f7355091e344fde90f845a262be2 (sha256 92f2777d…3876) |
| mtlib.py = e10_ref/mtlib.py | a684dd843c4a0bf569081dd8ff52987f |
| e10_ref/qlib.py (E10 그대로) | 7934e1b2e2224797b354c613ba2e8a63 (sha256 1edda3b2…ca59) |
| bclib.py | f09165fa2aedf3bc230f99cb4d4b9664 |
| inputs/bias_rule.json (= bias_rule_D2.json) | 2a9e8529453b388eaa09065338dca1cd |
| mt14_jobs.py | 9aa6bee8c96af70af03b1fd5176a56c8 |
| mt14_driver.py | 615b8aae0dde2041c3adcd0311278dd4 |
| analyze_e14.py | 5ac8af6599d3ccc4b522d30d17a17e0e |
| inputs/e10_eval_none_exact.json | 79e14740f332c9769fbca0b8428e4bdb |
| status.sh / restart.sh | 2dc5c25ab10ab9d9fb0815f652fe6fae / ee934340060bdb4e8c9e21e731f9534c |

## 8. det+BC의 b 절차: §0.7 점검 2 실패 → [편차 D2] (19:01Z 결정, 19:06Z 적용)

- 설계 절차(b ← b + m, 3 sweep)가 317M에서 수렴하지 않는다. door-open k=3, 3 sweep 뒤 잔차 ÷ Δ: dyn0 0.220, dyn1 1.264, rew0 0.043, rew1 0.166(문턱 0.01).
- 진단(`out/smoke/diag_bias_door_k{3,4,5}.json`, 본 실행에 쓰지 않음)

| 규칙 | k | sweep | dyn0 | dyn1 | rew0 | rew1 |
|---|---|---|---|---|---|---|
| 설계(α = 1) | 3 | 3 | 0.220 | 1.264 | 0.043 | 0.166 |
| 설계(α = 1) | 3 | 10 | 0.276 | 1.426 | 0.047 | 0.177 |
| 감쇠 α = ½ | 3 | 12 | 0.0017 | 0.0074 | 0.0004 | 0.0016 |
| 감쇠 α = ¼ | 3 | 16 | 0.0057 | 0.026 | 0.0014 | 0.024 |
| 국소(층 자신의 반올림 편향만) | 3 | 6 | 0.0082 | 0.037 | 0.0061 | 0.011 |
| 설계(α = 1) | 4 | 3 | 0.414 | 1.993 | 0.089 | 0.139 |
| 감쇠 α = ½ | 4 | 8 | 0.0090 | 0.045 | 0.0033 | 0.017 |
| 설계(α = 1) | 5 | 3 | 0.246 | 1.447 | 0.085 | 0.127 |

- 감쇠 ½가 수렴할 때 전파 오차의 MSE도 크게 준다(k=3, dyn1 1.75 → 0.235 Δ², rew1 2.49 → 0.026 Δ²). 설계 규칙은 반대로 커진다(dyn1 → 4.17 Δ²).
- **결정 D2**(판단 에이전트 Fable 5.2 max, 2026-11-21T19:01:09Z; DESIGN_tmlr.md §11, tmlr_ext/LOG.md; 규칙 파일 `runpod_tmlr/bias_rule_D2.json` = `code/inputs/bias_rule.json`)
  - α = 1 → ½ → ¼ 순서로 하고, 단계마다 b = 0에서 시작한다.
  - pass마다 m과 층별 r = mean_j|m_j|/Δ_j를 잰다. 모든 층 r < 0.01인 첫 pass의 b로 확정한다.
  - 아니면 b ← b + α·m을 한다. 단계당 갱신은 최대 60번이고, 못 미치면 다음 단계로 간다.
  - 세 단계 모두 못 미친 과제는 최악 층 r이 가장 작은 단계의 마지막 b를 쓰고 `bc_unconverged`로 표시한다. det+BC는 그대로 돌리고 조정자에게 알린다.
  - 자료(held 10 episode 96,000행 × 3 step)와 모든 층 동시 갱신은 그대로다. 점검 2는 최종 b에서 판정한다. α 목록과 상한은 결과를 보고 바꾸지 않는다.
- 구 규칙 b(3 sweep): `out/bias_frozen_rule/`로 옮겨 기록으로만 둔다. 그 b로 돈 episode는 스모크 1개뿐이고 쓰지 않는다. 본 실행 det+BC는 D2 b로만 돈다.
- 구현: `bclib.bias_staged`. driver는 D2 결과면 관문을 연다(미수렴이면 `FLAG bc_unconverged:<task>`). 분석은 det+BC 기술 항목을 (가) 모든 과제, (나) 표시 과제 제외로 낸다.
- 기록(과제마다, `out/jobs/bias/bias_<task>_k<k>.json.gz`, npz)
  - 수렴 단계, 갱신 횟수
  - pass별 층별 평균·최대 |m_j|/Δ_j, MSE/Δ², 전파 잠재 오차(step 1~3)
  - 최종 b와 m₀, 층별 mean_j|b_j|/Δ_j, b = 0과 최종 b의 MSE/Δ²
- 권장 진단(실행에 쓰지 않음): `diag_bias_split.py`. held 5 episode로 구한 b를 나머지 5 episode에서 잰 잔차다. 결과는 `out/diag/`.

## 9. 설계와 다른 점

- GPU: RTX A5000 대신 RTX 4090 두 대(설계 §8.1의 대체 경로, 조정자 승인). 그래서 E10(4090)과 GPU 종류가 같다. "다른 GPU에서의 복제"(§4 쓰임)는 이번에는 성립하지 않고, 새 seed에서의 복제만 남는다.
- det+BC의 b 절차: [편차 D2](위 §8). 설계 §0.6 4단계와 §0.7 점검 2의 읽기가 바뀌었다.
- 나머지(가설, 지표, 문턱, 칸, seed, 변형 정의, 통계)는 설계 그대로다. b에 기록 행 전부(96,000행)를 쓰는 것은 설계가 열어 둔 운영 선택이다(§2).
