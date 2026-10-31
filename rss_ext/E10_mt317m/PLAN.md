# E10 실행 계획 (본 실행 전에 작성: 2026-10-31T09:16Z)

- 기준: [../DESIGN_ext.md](../DESIGN_ext.md) §0·§3·§4. sha256 `c660a10dc72ba044903996c0d8342cd90bc1941beeac9d654cbd00d1b050850c`로 고정판과 같음을 확인했다.
- 가설(G1–G3), 지표, 문턱, 과제 규칙, seed는 설계 그대로다. 이 문서는 운영 세부(스모크 결과, 구현 선택과 그 규칙, 실행 순서, 비용)만 적는다.
- 코드: [code/](code/). 스모크 산출물: [results/smoke/](results/smoke/)(해석하지 않는다).

## 1. wrapper 확인 결과: 공식 다중 과제 경로와 비트 단위로 같다

공식 코드(commit 332f19e)는 다중 과제 체크포인트를 `task=mt80` 평가(80개 env를 `MultitaskWrapper`로 묶음)로만 돌린다. `code/mtlib.py`는 한 과제의 env만 만들고 같은 입력을 준다.
- agent: `parse_cfg(task=mt80, model_size=317)` + 공식 `make_env(mt80)`이 cfg에 채우는 값(과제별 관측 차원, 행동 차원, episode 길이)
  - 이 값은 `code/mt_dims.py`가 공식 경로로 한 번 만들어 `mt80_dims.json`에 저장한다. 작업자 프로세스는 80개 env를 만들지 않는다.
- env: 그 과제의 단일 과제 env(`make_env`, 공식 경로가 구성원 env를 만드는 방식과 같음). 관측은 끝에 0을 붙여 최대 차원으로, 행동은 그 과제 차원까지만 잘라 `env.step`(`MultitaskWrapper._pad_obs`, `step`과 같음)
- 계획: `agent.act(obs, t0, eval_mode=True, task=<mt80 index>)`. task embedding이 encoder·dynamics·reward·π·Q 입력에 붙고, `_plan` 안에서 행동 mask가 걸린다.

**점검(`code/test_wrapper.py`, `results/smoke/test_wrapper.json`)**: mw-push와 mw-door-open, seed 99000
- 공식 경로(80개 env, `env.reset(task_idx)`)와 mtlib 경로의 episode가 **행동열 비트 단위로 같다**. return(1799.529…, 1637.560…)과 성공도 같다.
- 두 agent의 state_dict가 모든 키에서 같다. 할인율 벡터도 같다.
- 마스크된 행동 차원(5·6번째)은 항상 0이다.

**체크포인트와 설정(`results/smoke/mt80_ckpt_report.json`, `mt80_dims.json`)**
- HF `nicklashansen/tdmpc2/multitask/mt80-317M.pt`, sha256 `38efd60b…99cc`(HF LFS oid와 같음). pod의 `/workspace/ckpt/mt80-317M.pt`
- 구 API 형식이다(`_Qs.params.N`). 공식 `api_model_conversion`이 변환한다.
  - `_action_masks`가 체크포인트에 없다. 그래서 mask는 cfg의 과제별 행동 차원(공식 경로가 채우는 값)에서 만들어진다.
- metadata: task mt80, 과제 목록이 `TASK_SET['mt80']`과 같은 순서, model_size 317, task_dim 96, num_q 8, date 2023-08-13. 학습 매개변수 317,383,129개
- mt80의 최대 관측 차원은 39(= Meta-World)라서 Meta-World 과제에는 padding이 없다.
- 최대 행동 차원은 6이다. Meta-World mask는 [1,1,1,1,0,0]이고, 할인율은 0.95(episode 100)다.
- 과제 index: push 63, pick-place 56, door-open 41, drawer-open 44, button-press-topdown 32, peg-insert-side 53, window-open 73, sweep-into 71
- 참고: 공식 `evaluate.py`는 다중 과제 평가에서 `eval_mode`를 넘기지 않는다(탐색 잡음 포함). 이 실험은 E1처럼 `eval_mode=True`다(설계 "계획기 기본값(E1과 같음)").

## 2. 양자화 hook 확인
- hook 층은 qlib 그대로다. 모두 NormedLinear(Linear → LayerNorm → Mish)이고, 입력 1478 = latent 1376 + 행동 6 + task embedding 96이다.
  - dyn0 = `_dynamics[0]`(1478 → 4096), dyn1 = `_dynamics[1]`(4096 → 4096)
  - rew0 = `_reward[0]`(1478 → 4096), rew1 = `_reward[1]`(4096 → 4096)
- 호출 패턴(det·sh-iter, call log)
  - 제어 step마다 512행 호출이 층마다 18회, 정책 prior 24행 호출이 dyn0·dyn1에 2회다. E0와 같다.
  - episode당 양자화 호출은 7,600회다(본 실행에서도 episode마다 `nq`로 기록해 점검).
- 기록기(`HeldRecorder`)를 켜도 episode가 비트 단위로 같다(행동열 동일). 기록기는 numpy 난수만 쓴다.

## 3. 스모크 none 성공률과 결정당 지연 (RTX 4090, seed 99000~99009, 해석하지 않음)

| 과제 | none 성공 | episode 시간 | 결정당 계획 지연(중앙값) |
|---|---|---|---|
| mw-push | 10/10 | 7.63초 | 73.2 ms |
| mw-door-open | 9/10 | 7.69초 | 73.4 ms |
| mw-pick-place | 10/10 | 7.67초 | 73.6 ms |

- 공식 다중 과제 결과와의 비교
  - 저장소에는 과제별 수치가 없다. 논문은 80과제 정규화 점수만 보고한다.
  - 그래서 wrapper 확인의 주 근거는 §1의 "공식 경로와 비트 일치"이고, none 성공률 0.9~1.0은 보조 근거다. 크게 어긋나지 않는다.
- 결정당 지연(기술 항목 "결정 한 번의 계획 지연(fp32, 그 GPU)")
  - fp32(TF32 꺼짐, `allow_tf32=False` 확인), 단일 프로세스, GPU에 다른 작업 없음, episode 첫 결정 제외
  - **RTX 4090에서 73.2~73.6 ms**, GPU 사용률 94%, GPU 메모리 최대 4.9GB(점검 중 agent 1개)
  - det·sh-iter(k=3)는 76.5·77.4 ms다. hook 부담은 약 5%다.
- 경로 점검
  - det·sh-iter(k=3, 스모크 기록 1개로 잡은 범위) episode가 돈다.
  - 범위 계산과 pt_err1h 경로(8 상태 × 512 = 4,096 시퀀스, 0.67초)도 돈다.

## 4. 운영 세부 (구현 선택과 규칙; 결과를 보기 전에 고정)
1. **episode 규약 = qlib.run_episode(E1)**
   - 매 episode env를 seed로 새로 만든다. 이어서 `seed_all(seed)`를 하고, dither generator는 seed × 7 + 13으로 둔다.
   - `eval_mode=True`, 성공 = 마지막 step의 `info['success']`(E1의 `success`)
   - OEI·계획 재실행 계측은 하지 않는다(E10 설계에 없음). episode마다 결정당 지연과 `nq`를 기록한다.
2. **held none 30 episode(91000~91029)의 쓰임**: 설계대로 이 30 episode가 자격(성공률 ≥ 0.7, 곧 21/30 이상), 보정, k 선택의 none 기준, pt_err1h 입력에 모두 쓰인다.
3. **보정(E0 방식, `mtlib.quant_ranges_mt` = calib_lib.quant_ranges_from_calib + task index)**
   - 30 episode의 모든 step에서 MPPI 반복마다 무작위 후보 16개와 root latent를 기록한다(`numpy.default_rng(episode 번호)`).
   - 30 × 100 × 6 × 16 = 288,000행에서 12,000행을 뽑는다(`default_rng(1)`).
   - 그 행들의 horizon 3 forward에서 층별(per-tensor) [min, 99.99 백분위]를 잡는다.
4. **k 선택**
   - det를 held 30 episode에서 돌린다. 하락 = held none 성공률 − held det 성공률(같은 seed 30개)
   - k = 5, 4, 3, 2 순서로 처음 하락 ≥ 0.20(허용 오차 1e-9; 30 episode에서 6개 차이)인 k가 1차 k다.
   - k = 2까지 없으면 "2비트 이상에서 손상 없음"이다. 그 과제는 eval을 돌리지 않고 G2에서 뺀다.
5. **pt_err1h(E1 H1 정의, 입력만 held none)**
   - held none episode마다 E1 perr 상태 8개의 첫 MPPI 반복 후보 512개(정책 prior 24개 포함)를 쓴다.
     - E1 perr 상태 = Meta-World OEI 상태 t % 4 = 0 가운데 고르게 뽑은 8개 → t = 0, 12, 28, 40, 56, 68, 84, 96
   - `plan_inputs_mt`(K = 5)를 쓴다. horizon 밖 2 step은 비양자화 rollout 위의 정책 평균 행동이다.
   - `pred_errors_mt`(h1 = 3)
     - err1h = 비양자화 궤적의 horizon 위치 j < 3마다의 1-step 잠재 오차를 비양자화 예측 분산으로 나눈 값
     - sh-iter는 dither seed 3개(9,000,001 + 1000·seed + j) 평균
   - 과제 값은 30 episode 평균이다. det와 sh-iter는 같은 입력을 쓴다. 범위는 그 과제의 보정 범위, k는 1차 k다.
   - 위 함수들은 qlib.pred_errors·plan_inputs의 복사본이다. 바뀐 것은 task index를 넘기는 것뿐이다.
6. **eval**
   - none(k와 무관)은 과제의 자격이 정해지면 바로 시작한다(빈 GPU 시간을 채우는 가장 낮은 우선순위).
   - 그 과제가 나중에 "2비트 이상 손상 없음"이 되면 none 90 episode는 판정에 쓰이지 않는다(낭비 약 11 GPU-분).
   - det·sh-iter는 1차 k에서 90 episode(90000~90089)다.
7. **판정 계산(DESIGN_ext §0.3 그대로; `code/analyze_e10.py`, 가짜 자료로만 점검함)**
   - 짝 = eval episode seed 90개. 짝 bootstrap 10,000회를 하고, `RandomState(0)` 인덱스 행렬 하나를 모든 통계·과제에 쓴다(E1 관례). 백분위 95% CI
   - 하락 실재
     - p = 재표본 D* ≤ 0의 비율 < 0.05, Holm(엄격한 <)으로 보정한다.
     - **Holm 집합 = 1차 k가 있어 eval을 돈 자격 과제**
     - E1처럼 D_det > 0도 요구한다(p < 0.05면 사실상 자동). E1식 p((# + 1)/(B + 1))도 보고한다.
   - red = (D_det − D_sh)/D_det(점추정으로 판정). CI는 D_det* > 0인 재표본에서 구한다.
   - 해로운 과제: D_sh − D_det의 CI 하한 > 0.05
     - G2의 "해로운 과제가 없으면"은 **eval 과제 전체**에서 본다(E1 H3a 관례: 단위 안 모든 확증 칸).
     - 손상 과제만 본 판도 함께 보고한다.
   - 손상 과제가 0개면 G2·G3는 "판정 불가"로 적는다.
   - 허용 오차 1e-9의 적용
     - "≥ a"는 x ≥ a − 1e-9, "≤ a"는 x ≤ a + 1e-9
     - "> a"는 x > a + 1e-9, "< a"는 x < a − 1e-9
   - 기술 항목
     - 1차 k와 5M(E1) k 비교: E1 1차 k는 push 3, pick-place 3, door-open 4, drawer-open 3, button-press-topdown 3, peg-insert-side 4, window-open 4, sweep-into 3
     - 손상 크기, none 성공률, 결정당 지연(§3)
     - pt_err1h det − sh-iter의 held episode 짝 bootstrap CI
     - err3·err5·errr, McNemar
8. **대체 목록**: 두 pod의 자격 결과를 합친다. 자격 과제가 4개 미만이면 설계 목록 순서로 과제를 더한다(한 pod의 driver에 `--tasks`로 추가). 자격 과제가 6개가 되거나 목록이 끝나면 멈춘다.
9. **같은 GPU 종류 안의 짝**
   - 한 과제의 모든 episode(held, k 선택, eval)는 한 pod에서 돈다.
   - 두 pod 모두 RTX 4090(드라이버 595.91.07)이다.

## 5. 실행 순서와 pod 분할
- `code/e10_driver.py`(재개 가능한 단계 기계, 결정은 결과 파일에서만 다시 계산)
  - 작업자 프로세스 2개가 같은 GPU를 쓴다(1개일 때 GPU 사용률이 94%라서 남는 6%와 CPU 단계를 겹친다).
  - 처리량이 1개일 때보다 낮으면 1개로 되돌린다.
- job: held 6 × 5 episode, k마다 ksel 6 × 5 episode, 변형마다 eval 9 × 10 episode, ranges 1, perr 1
- 우선순위: held·ranges → ksel → det·sh-iter eval·perr → none eval
- pod 분할(과제 단위)
  - **e10.1**: mw-push, mw-pick-place, mw-door-open, mw-drawer-open
  - **e10.2**: mw-button-press-topdown, mw-peg-insert-side, mw-window-open, mw-sweep-into
- 결과(job JSON, 범위 JSON, state.json, 로그)는 1시간 이내 간격으로 맥북 `results/raw_e10.N/`으로 당겨 온다.
  - held 기록 npz(과제당 약 40MB)는 pod에만 둔다.
- 끝나면 pod에서 `analyze_e10.py`를 돌리고 결과를 당긴 뒤 pod를 지운다.

## 6. 비용 추정 (RTX 4090 $0.89/h, budget.py의 ×1.03 포함)
- 처리량: 7.65초/episode(GPU 포화). 작업자 2개도 거의 같다고 본다(보수적).
- 과제당 episode
  - held 30 + k 선택 30 × m(m = 시험한 k 개수 1~4) + eval 270
  - m = 3이면 390 episode, 약 50분이다.
- pod당 4과제
  - m = 2~3이면 약 3.0~3.3시간, 모든 과제가 m = 4이면 약 3.6시간이다.
  - 여기에 설치·점검·분석·대기 약 0.4시간을 더한다.
- **합계 추정: 약 $6.5(m = 2~3) ~ $7.3(m = 4)**
  - 지금까지(스모크 포함) 약 $0.35다. 상한 $13의 80% = $10.4 안이다.
  - 대체 과제가 필요하면 과제당 약 +$0.7이다.
- 시간: 두 pod 병렬로 약 3.5~4시간(09:17Z 시작 → 12:50~13:20Z 끝). TERM_H = 8(종료 17:00Z, 17:13Z)
- 추정 지출이 $10.4에 다가가는데 끝나지 않을 것 같으면 멈추고 조정자에게 보고한다.

## 7. 설계와 다른 점
- 없음. 위 §4는 설계가 열어 둔 운영 세부의 고정이다.
- 결과를 본 뒤 정하는 분석은 REPORT.md에 [사후]로 표시한다.
