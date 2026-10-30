# E2·E3 TD-MPC2 분석 작업 로그 (작업자, UTC, 서버 시계 기준)

- 15:30Z경(10-23) 조정자 지시 수신: TD-MPC2 E2(4,482 job, 06:18Z 끝)와 E3(2,124 job, 14:51Z 끝)의 자료 점검, 분석 스크립트 대조, 본 분석, 단계 보고. 사용자 승인 있음
- 15:3xZ 읽은 것: MAIN_RULES.md 전체, DESIGN_main.md 전체(v1~v5.2, 고치지 않음), worklogs/E1_tdmpc2.md, reports/E1_tdmpc2.md(§0~§15, 부록 A), main/LOG.md 10-22 이후
- 15:36Z 서버 접속 확인(ssh kiis-mvf-gpu, BatchMode). load 37/64, 디스크 여유 266G. 결과 파일 수: out_e2 episode 4,482, out_e3 episode 2,088 + decision 36. 두 폴더 모두 errors 폴더 없음, locks 비어 있음
- 15:37Z 코드 md5 대조: analyze_e1/e2/e3, v5_common, test_v5_analyses, analyze_h2a_indep, qlib, runner, make_manifest, calib_lib, check_v5_manifests, status 모두 맥북 = 서버
- 15:38Z 서버 python: `~/pmx/envs/a-mw/bin/python`(3.11, numpy 1.24.4, scipy 1.15.3, matplotlib 3.10.9). 모든 python은 `CUDA_VISIBLE_DEVICES="" OMP/MKL/OPENBLAS=4 nice -n 10`으로 돌리고 env.sh는 source하지 않는다. 내 임시 폴더는 `~/pmx/A/tdmpc2_main/work_e2e3/`
- 15:38Z **판정 문턱 비교의 부동소수점 문제 확인**(가짜 순위로만, 자료 아님): scipy spearmanr는 4점에서 이웃 자리바꿈 한 번을 −0.7999999999999999로, 5점에서 Σd² = 4를 0.7999999999999999로 낸다. 그러면 코드의 `r <= -0.8`, `r >= 0.8`가 거짓이 된다. DESIGN v5 §7.2 H4a는 "4점이라 이웃 자리바꿈 한 번까지 허용한다"(= −0.8 포함)이고 H2c·H4b·DMC 설명도 "≥"이므로, 규칙대로 계산하려면 문턱 비교에 작은 허용 오차가 필요하다(아래 코드 수정)
- 15:40Z 새 점검 스크립트 `code/tdmpc2/integrity_e2e3.py` 작성(결과 수치는 출력하지 않음). 설정·변형·k·계측 기대값은 make_manifest가 아니라 DESIGN v5 §7.1 표, v5.1 §8.2, v5 §7.2에서 옮겨 적었다. 맥북 py_compile 뒤 서버 code/로 복사(md5 같음)
- 15:41~15:42Z 자료 점검(서버 CPU, 2 프로세스, 약 40초). 결과 `work_e2e3/integrity_e2.json`, `integrity_e3.json`, 둘 다 **ALL_OK**
  - E2: manifest 4,482 job = 결과 4,482개, 모든 결과의 spec이 manifest 항목과 같다. 22,410 episode = 747조건 × 30(ep 0~29), seed = 40000 + ep. 3과제 × 20설정 그룹이 모두 있고 설정별 변형 집합이 설계와 같다(base 7, 나머지 4). planner 실효값과 warm start가 v5 §7.1 표·v5.1 §8.2와 같다. 계측은 MW 4/20 step, DMC 10/50 step, 독립 재평가 켬, region_ref는 base에만
  - E3: manifest 2,124 job(episode 2,088 + decision 36) = 결과 2,124개, spec 같음. 10,440 episode = 348조건 × 30, seed = 50000 + ep. 12과제 1차 k, 8변형, sweep 4과제에 5변형 추가. decision 36개 모두 상태 150개, 변형 집합이 설계와 같다. episode별 예측 오차 누락 0
  - 두 단계 모두 모든 job이 같은 host, NVIDIA RTX A5000, CUDA_VISIBLE_DEVICES=0, GPU 0 MPS pipe, torch 2.7.1+cu126(같은 장비 원칙 충족). errors 파일 0, lock 0, OEI 필드 누락·비유한값 0, episode당 OEI step 수는 MW 25, DMC 50으로 일정
  - 독립 재평가의 기계적 항등식: indep_same은 det·표 변형 1.0, 새로 뽑는 변형 0.0이고, fresh_same(표를 다시 뽑은 재평가)은 0.0이다
  - 실행 시간대(결과 파일 기준): E2 10-22 19:33:08Z ~ 10-23 06:17:53Z, E3 10-23 08:33:52Z ~ 14:51:36Z
- 15:42~15:47Z **분석 스크립트와 DESIGN 규칙 대조**(E2·E3 결과 수치를 읽기 전). 대조한 규칙: v4 §6.1·§6.2(통계, H2a), v5 §7.0·§7.1·§7.2, v5.1 §8.2·§8.3, v5.2 §9.1·§9.2. qlib의 정의도 확인했다(oei = e_chosen − f_e_rand, sel_last = e_chosen − l_e_rand, drift = l_e_rand − f_e_rand, oei_indep·sel_indep는 독립 draw 재평가, drift_ref는 비양자화 계획기 모집단의 양자화 평가)
  - 규칙대로인 것: 짝 bootstrap(RandomState(0), 10,000회, 정렬한 (체크포인트, episode) 짝, 백분위 CI, DMC 비율 재계산), 설정별 none 기준 D, H2a 재현(base, 동결 oei), H2c 확증 sweep 수준(N64k, N128k, N256k, base, N1024k)과 비율 고정 보조, 설정 사이 짝을 함께 다시 뽑는 bootstrap, (b1) oei_indep 규칙, (b2) drift_det CI 상한 < 0과 r CI 상한 < 0.5(재표본마다 비, drift_det* ≥ 0 버림), 양성 대조 sh-step·sh-ep r CI 하한 > 0.5(판정 밖), (b3) 기술, E3 2축 칸·red·D_v − D_det, H4a(E1 Holm 자격 sweep 과제, 4수준, 과반), var_cand·shiter_l/g32·지속 비용, H4b(밀도 = −E1 마지막 반복 gap 중앙값, 이득 = D_ind − D_sh-iter, E3 seed), DMC 설명(E3 seed p < 0.025, oei_det CI가 0을 포함하면 제외, S_det = drift_det/oei_det, 0.5 문턱, 판정 순서), `--final`은 E3 episode job이 모두 있어야 돈다
  - **규칙과 다르게 계산하던 곳(고침)**
    1. 문턱 비교의 부동소수점(15:38Z 항목): `r <= -0.8`, `r >= 0.8`, `r >= 0.5`를 `v5_common.le/ge`(허용 오차 1e-9)로 바꿨다. Spearman은 12점 이하에서 거친 유리수 격자 위의 값만 가지므로, 1e-9는 문턱 반대편에 있는 값을 통과시키지 않는다. H4a(이웃 자리바꿈 한 번 허용), H2c(≥ 0.8), H4b(≥ 0.5), DMC 설명(≥ 0.5)에 적용
    2. bootstrap ρ 분포의 동순위 처리: `spearman_rows`가 동순위를 위치 순서로 끊고 있었다(성공률 하락은 재표본에서 자주 같아진다). scipy spearmanr와 같은 평균 순위로 바꿨다(`rankdata`). 보조 통계(H2c ρ 분포, H4a ρ 분포)에만 쓰이고 판정에는 쓰이지 않는다
  - **기능 추가(규칙 변경 아님)**: H2c 수준별 OEI_det·D_det CI와 모든 ρ의 bootstrap 분포(보조 ρ 포함), 반복 수 곡선의 CI, 계열 합산용 `family_points_tdmpc2.json`, 셀 CSV. E3: CRN 이득의 짝 CI, `h4b_points_tdmpc2.json`, 2축 짝 대조(사전 예상 indstep ≈ ind ≈ indep을 D·oei_indep·drift로; 공유 축과 지속 축), sweep 수준별 CI, decision job 체크포인트 평균, 셀 CSV
  - **`--final` 한 번 보장**: `e3_tdmpc2_final.json`이 있으면 `--final`을 거부한다. `--final` 없이 다시 돌리면 그 파일을 읽어 표에 넣고 다시 계산하지 않는다(v5 §7.0 "한 번만")
- 15:47Z 맥북 문법 검사(ast.parse, 바이트코드를 쓰지 않음; 그 전에 한 번 py_compile을 썼다가 맥북 `code/tdmpc2/__pycache__`의 analyze_e2·v5_common 옛 pyc 두 개를 지웠다. 재생성되는 캐시다). 서버 code/에 복사, md5 같음. E5·E7 분석은 v5_common에서 바뀐 함수를 쓰지 않는다(grep 확인)
- 15:47Z 단위 점검 `work_e2e3/test_fix.py`(가짜 수치): spearman_rows·rows2가 동순위가 많은 행에서 scipy와 1e-12 안에서 같다. 정확히 −0.8·0.8은 통과, −0.6·0.7·NaN은 실패
- 15:48Z 가짜 자료 dry run(`test_v5_analyses.py`로 e2·e3 manifest마다 가짜 결과를 `work_e2e3/fake_e2`, `fake_e3`에 만듦): analyze_e2 5초, analyze_e3 `--final` 4초, 모든 표·JSON이 나온다. 두 번째 `--final`은 거부됨(rc 1). `--final` 없는 재실행은 저장된 final을 읽는다(15:50Z, 고친 판으로 다시 확인)
- 15:49:28~15:49:42Z **본 분석 실행**(서버 CPU 2 프로세스, nice 10, 4스레드, CUDA_VISIBLE_DEVICES=""). 고친 판 md5: analyze_e2 8d54a408…, analyze_e3 7cce8832…, v5_common ee92bb3f…
  - `analyze_e2.py --res ../out_e2 --outdir ../analysis_e2`: 22,410 episode, 불완전 그룹 0. H2a 재현 0/3 성립, H2c 3/3 성립, (b1) 1/3(cheetah-run), (b2) 3/3
  - `analyze_e3.py ... --outdir ../analysis_e3 --final`(**한 번만**, computed_at 15:49:35Z): 빠진 job 0, 10,440 episode, 12과제 모두 90짝. H4a TD-MPC2 몫 3/4 성립, H4b 불성립(ρ = 0.03), DMC 설명 "explained"(ρ(S_det, red) = 0.83, 12과제)
  - **H4a의 mw-push와 mw-pick-place는 Spearman이 정확히 −0.8(이웃 자리바꿈 한 번, N_eff 128과 2,048 순서가 뒤바뀜)이다.** scipy 값은 −0.7999999999999999이다. 고치기 전 코드였다면 두 과제가 불성립이 되어 TD-MPC2 몫이 1/4(불성립)이 된다. 15:38Z에 자료를 보기 전 DESIGN v5 §7.2 문구("4점이라 이웃 자리바꿈 한 번까지 허용")대로 고친 판으로 계산했다. 판정이 이 경계에 달려 있다는 점을 보고에 적는다
- 15:50Z 결과 확인. 결과 폴더 `analysis_e2/`(summary·cells CSV·tables MD·family_points), `analysis_e3/`(summary·final·h4b_points·cells CSV·tables MD)
- 15:50Z 맥북으로 옮김(JSON·CSV만): `results/E2_tdmpc2/`(e2_tdmpc2_summary.json, e2_tdmpc2_cells.csv, family_points_tdmpc2.json, integrity_e2.json), `results/E3_tdmpc2/`(e3_tdmpc2_summary.json, e3_tdmpc2_final.json, h4b_points_tdmpc2.json, e3_tdmpc2_cells.csv, integrity_e3.json). 표 MD는 서버에만 둔다
- 15:51~15:53Z 그림(기술): `code/tdmpc2/plot_e2e3_tdmpc2.py`(summary JSON만 읽음, 통계 계산 없음, dataviz 기준 팔레트 앞 세 색, 검증기 통과). 서버에서 그려 `results/E2_tdmpc2/e2_iteration_curves.png`, `e2_h2c_nsweep.png`, `results/E3_tdmpc2/e3_table_sweep.png`로 옮김(각 75~130KB). 눈으로 확인함(겹침 없음)
- 15:55Z **사후 분석**(final 결과를 읽은 뒤 정함, 판정에 쓰지 않음): `code/tdmpc2/posthoc_e3_dmc_within_family.py`가 저장된 DMC 설명 행을 읽어 계열 안 Spearman을 낸다(다시 계산하지 않음). ρ(S_det, red): 12과제 0.83(사전등록 값), Meta-World 8과제 0.57, DMC 4과제 1.00. ρ(밀도, red): −0.27, −0.81, 0.80. 결과 `results/E3_tdmpc2/e3_tdmpc2_posthoc_dmc_within_family.json`
- 15:56Z 파일 sha256: e3_tdmpc2_final.json 7984244b…, h4b_points_tdmpc2.json 9f088743…, family_points_tdmpc2.json c99c6faa…
- 15:56~16:03Z 보고서 작성: `reports/E2_tdmpc2.md`, `reports/E3_tdmpc2.md`. 숫자는 summary·final JSON에서 다시 뽑아 옮겼다(처음 쓴 문장 몇 곳을 JSON과 대조해 고침: 지속 비용 개수 16 → 12, sh-step 반복 곡선 서술, 계획 편차 최댓값 서술, drift_ref CI가 0을 포함하는 칸 표시)
- 16:03Z 서버 정리: 내가 만든 임시 파일만 지웠다(`work_e2e3/fake_e2`, `fake_e3`, `fake_analysis_e2`, `fake_analysis_e3`, `mplcfg`, `test_fix.py`). 단위 점검 스크립트는 `code/tdmpc2/test_e2e3_fixes.py`로 맥북·서버 code/에 남겼다. `work_e2e3/`에는 점검 JSON과 실행 로그만 남는다
- 16:04Z 최종 md5(맥북 = 서버): analyze_e2 8d54a408…, analyze_e3 7cce8832…, v5_common ee92bb3f…, integrity_e2e3 0bdd87d4…, plot_e2e3_tdmpc2 164f4e09…, posthoc_e3_dmc_within_family 042e0d07…, test_e2e3_fixes b329a6d7…
- 16:04Z 돌고 있는 내 작업 없음(ps 확인). tmux `A-v5-queue-g0`(E5), `A-v5-dinowm-g1`(DINO-WM E3)은 건드리지 않았다. GPU는 쓰지 않았다. MAIN_STATUS.md, main/LOG.md, DESIGN_main.md는 고치지 않았다
