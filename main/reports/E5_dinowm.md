# E5 DINO-WM 결과 보고 (A 논문: 실험 5 강한 기준선과 결합, H3b, DINO-WM 계열)

- 작성: 2026-10-24(UTC), DINO-WM E5 분석 작업자(E2·E3 분석에 이어서 맡음). E5 실행은 10-24T11:51Z에 끝났다
- 기준: [../DESIGN_main.md](../DESIGN_main.md)
  - v5 §7.3: DINO-WM 기준선 4종, 최선 기준선 선택, 평가 칸, 가중치, QAT, H3b. TD-MPC2 QAT 규칙(CI 상한 5%p)
  - 설계 §6.5(QAT 보고 규칙), §6.8(통계), v4 §6.1(짝 bootstrap)
  - v5.3 §10.1(10-24T02:38Z 고정, 사용자 승인): QAT 보고 방식. DINO-WM QAT도 QAT+det가 B+det보다 나쁜 양상이면 TD-MPC2와 같은 단서를 단다
- 코드: [../code/dinowm/](../code/dinowm/)
  - `analyze_e5.py`·`e5_select.py`: E5 결과 전에 만들었다. 10-24T12:22Z, E5 수치를 계산하기 전에 출력만 보탰다(§3)
  - `check_e5.py`(자료 점검), `e5_posthoc_dinowm.py`(사후 기술)
- 결과: [../results/E5_dinowm/](../results/E5_dinowm/) · 작업 로그: [../worklogs/E5_dinowm.md](../worklogs/E5_dinowm.md)
- 표기
  - D: 같은 쌍·planner seed의 E1 none(원래 모델, 비양자화) 대비 성공률 하락(%p)이다. E5 평가는 E1 eval 쌍과 planner seed를 그대로 쓴다
  - CI는 짝 bootstrap 10,000회(RandomState(0)) 백분위 95%다
  - 출처 표시: **사전등록**(v5 §7.3, 설계 §6.5), **v5.3 사전등록**(§10.1: TD-MPC2 E5 결과 뒤, DINO-WM QAT 평가 전에 고정), **기술**(문턱 없음), **탐색**(설계가 탐색으로 지정), **사후**(결과를 본 뒤 고른 계산)

## 0. 요약: 판정 (DINO-WM 몫)

| 항목 | 출처 | DINO-WM 판정 | 핵심 수치 |
|---|---|---|---|
| **H3b** | 사전등록 v5 §7.3 | **불성립: 3칸 중 1칸 통과(필요 2칸)** | PointMaze k=3(B = last8)만 D(B+det) − D(B+sh-iter) = +12%p [4, 20]로 통과다. PushT k=3(B = p999)은 −2 [−6, 2], PushT k=4(B = pc)는 +2 [−4, 8]이다. PushT에서는 최선 PTQ가 det 손상을 거의 다 없앴다(D(B+det) +3, +4%p) |
| **QAT 보고 조건** | 사전등록 설계 §6.5, v5 §7.3 | **조건이 걸리지 않음** | D(QAT+det) = +6%p [−1, 13]. CI 상한 13%p ≥ 5%p다 |
| QAT 양상(v5.3 §10.1) | v5.3 사전등록 | **조정자 판단 필요(§9)** | D(QAT+det) − D(B+det) = +2%p [−5, +10]. 점추정은 QAT+det가 조금 나쁘지만 짝 CI가 0을 포함한다. TD-MPC2(11칸 모두 CI가 겹치지 않게 나쁨)처럼 뚜렷하지 않다 |
| QAT+none | 기술 | — | D = +27%p [19, 36]. QAT 예측기를 비양자화로 쓰면 원래 모델보다 크게 나쁘다. QAT 모델 안에서는 det가 none보다 21%p 낫다 |
| QAT+sh-iter | 기술 | — | D = +22%p [14, 30]. det(+6)보다 16%p [7, 25] 나쁘다. det fake-quant로 학습한 모델 위의 dither는 해롭다 |
| 가중치 양자화 | 탐색(v5 §7.3) | — | W8 RTN은 손상이 없다(+3 [−1, 8]). W4는 RTN +55, AWQ식 +71, SR K = 1·2·4가 +76·+68·+71%p다. SR은 RTN보다 모두 나쁘고, K에 따라 단조로 줄지 않는다. TD-MPC2(K가 클수록 손상이 줄고, K = 4가 3과제에서 RTN보다 나음)와 방향이 다르다 |
| 기준선 선택 | 사전등록 규칙, 재계산으로 확인 | 규칙대로 | PushT k=3 p999, PushT k=4 pc(pc·p999·mse가 17/20으로 같아 목록 앞쪽), PointMaze k=3 last8. B = main인 칸은 없다 |

## 1. 무엇을 돌렸나
- **manifest 4개**(서버 `~/pmx/A/dinowm_main/code/manifests/`)
  - E5_dinowm_held 42 job: 3칸(PushT k=3·4, PointMaze k=3) × 기준선 7개의 det, held 쌍 0~19(pair seed 3, planner seed 1000 + ep)
  - E5_dinowm_eval 60 job: 3칸 × B+det·B+sh-iter, eval 쌍 0~99(PushT eval6·planner seed 4000 + ep, PointMaze eval·3000 + ep). `e5_select.py`가 썼다
  - E5_dinowm_w 60 job: PushT eval6, 가중치 변형 6개 × 100 ep, 활성은 fp
  - E5_dinowm_qat 30 job: PushT eval6 k=4, QAT 모델(`qat_k4`)의 none·det·sh-iter × 100 ep
- **기준선 7개**(v5 §7.3 목록 순서, `make_manifest.E5B`)
  1. pc: per-channel 범위(보정 창 256개)
  2. p999: 99.9 백분위
  3. main: 99.99 백분위(E1 격자)
  4. p99999: 99.999 백분위
  5. mse: MSE 최소 범위
  6. smooth: SmoothQuant식 α 0.5
  7. last8: 마지막 블록 8-bit, 나머지는 E1 격자
- **QAT 학습**(`qat.py`, v5 §7.3 "PushT k=4, 약 2 GPU-시간")
  - det fake-quant + STE, E1 위치와 범위(main), DINO-WM 잠재 손실, AdamW lr 5e-5, batch 32
  - 예산 120분에서 멈췄다: 10,083 step, final
  - 손실은 처음 10개 기록 평균 0.0598에서 마지막 10개 평균 0.0556으로 줄었다
- **실행**(GPU 1 사슬 `A-v5-dinowm-g1`, GPU 0 사슬 `A-v5-dinowm-g0`)
  - GPU 1: held 3칸(10-23T22:22Z ~ 10-24T01:31Z) → e5_select(01:31:50Z) → eval PushT k=3·PointMaze k=3 → 가중치 w4awq·w4sr2·w8rtn → QAT 학습(07:01~09:01Z) → QAT 평가(~11:51Z)
  - GPU 0: eval PushT k=4(02:17~04:16Z) → 가중치 w4rtn·w4sr1·w4sr4(~06:58Z)
  - 칸 하나는 한 GPU에서 끝까지 돌았다. 실패 0
- **분석**: 서버 CPU 약 3초(`CUDA_VISIBLE_DEVICES=""`, nice 10, 스레드 4). 자료 점검 0.8초

## 2. 자료 점검 (결과 수치를 보기 전, `check_e5.py`)

결과: **불일치 0**(`results/E5_dinowm/check_e5.json`).
- **manifest와 결과가 1:1이다**
  - held 42, eval 60, w 60, qat 30 job이 모두 결과 JSON을 가진다
  - 저장된 job이 manifest 항목과 같고, 남는 파일은 없다
- **쌍과 seed**
  - held: split held(pair seed 3, salt held), ep 0~19, planner seed 1000 + ep
  - eval·w·qat: PushT eval6(pair seed 6, salt eval6, 4000 + ep), PointMaze eval(pair seed 5, salt eval5, 3000 + ep), ep 0~99
  - 칸·변형마다 episode가 한 번씩 다 있다(held 20, 나머지 100)
- **meta**: ranges·qopts·ckpt가 job과 같다. planner 기본값(후보 300, top-30), opt_steps PushT 30·PointMaze 10, bf16·TF32·SDPA, PCI_BUS_ID
- **장비**: 칸마다 GPU 하나다
  - held 3칸, eval PushT k=3·PointMaze k=3, w4awq·w4sr2·w8rtn, QAT는 GPU 1
  - eval PushT k=4, w4rtn·w4sr1·w4sr4는 GPU 0
  - hook 호출 수가 (칸, 변형)마다 일정하다. 가중치 변형은 활성 hook이 0이다(활성 fp)
- **선택 재계산**: held det 성공 수를 기준선별로 다시 세고 규칙(가장 높음, 같으면 목록의 앞쪽)을 적용했다
  - 세 칸 모두 selection.json과 같다(§4)
  - eval job의 baseline·ranges·qopts가 고른 B와 같다: pc → ranges pc, p999 → p999, last8 → main + last_bits 8
- **B = main인 칸은 없다.** 그래서 "B = main이면 E1 det·sh-iter 칸 재사용" 경로는 쓰이지 않았다
- **D 기준(E1 none) 연결**
  - eval·w·qat의 모든 episode에서 planner seed와 E가 E1 none과 같다
  - 첫 반복 비양자화 비용 배열은 eval·w에서 E1 none과 비트 단위로 같다
  - qat는 다르다(최대 차 0.995). QAT 예측기 자체가 비양자화 기준이라 예상된 차이다
- **QAT 학습 로그**(`qat/pusht_qat_k4/log.json`): final = True, 10,083 step, 120.04분, 손실 기록 201개 모두 유한

## 3. 분석 코드 대조와 수정 (결과 수치를 읽기 전)

**규칙대로 계산하던 것**
- H3b 칸 = selection.json의 세 칸이다. E1에서 det 하락이 실재한 3~4-bit 확증 칸이고, Wall k=3 탐색 칸은 없다
- D = E1 none 대비 성공률 하락(같은 쌍·seed)이다
- 차 = D(B+det) − D(B+sh-iter) = S(B+sh-iter) − S(B+det), 짝 bootstrap 백분위 CI(같은 재표본)
- 통과 = CI 하한 > 0, 성립 = 통과 칸이 절반 초과(3칸이면 2칸)
- QAT 조건: D(QAT+det)의 CI 상한 < 0.05
- 선택(`e5_select.py`): held 20 ep det 성공이 가장 높은 기준선, 같으면 목록의 앞쪽. 동률 규칙은 v5 §7.3 TD-MPC2 문구를 따른 것이고, 조정자가 10-22T19:35Z에 승인했다(E1 작업 로그)

**규칙과 다르게 계산하던 곳**: 없다.

**고친 것**(10-24T12:22Z, 판정 계산은 그대로)
- 문턱 비교에 1e-9 허용 오차를 썼다(조정자 지시, E2·E3와 같음)
  - 엄격 부등호(CI 하한 > 0, CI 상한 < 0.05)에서는 문턱과 1e-9 안의 값을 문턱과 같다고 보고 떨어뜨린다
  - 일반 비교 결과도 함께 저장했고, 이번 자료에서 둘은 같다
- split 거르기와 칸 완전성 표시
- H3b 칸: McNemar 정확 p(보조, 설계 §6.8), red와 CI(기술), held 선택 행과 최고점 동률
- QAT: 규칙 표시와 짝 대비(기술)
  - QAT+det − B+det: v5.3 §10.1 양상 점검
  - QAT+det − E1 det
  - QAT+det − QAT+sh-iter
  - QAT+none 대 E1 none
  - QAT 모델 자신의 none 대비 손상
  - 학습 로그 요약
- 가중치: TD-MPC2와 같은 짝 대비 다섯(w4rtn − w4sr1·w4sr2·w4sr4·w4awq, w8rtn − w4rtn)과 SR 단계 대비 둘(w4sr1 − w4sr2, w4sr2 − w4sr4)
- `--family` 출력

**점검**: B = main 칸을 넣은 가짜 자료로 dry run(2.4초) 뒤 실제 자료에 한 번 돌렸다.

## 4. 기준선 선택 (사전등록 규칙, held 20 ep det 성공 수)

| 칸 | B | pc | p999 | main(E1 격자) | p99999 | mse | smooth | last8 |
|---|---|---|---|---|---|---|---|---|
| PushT k=3 | p999 | 9 | **17** | 7 | 1 | 15 | 0 | 9 |
| PushT k=4 | pc | **17** | 17 | 9 | 13 | 17 | 14 | 9 |
| PointMaze k=3 | last8 | 14 | 14 | 14 | 12 | 16 | 16 | **18** |

- PushT k=4는 pc·p999·mse가 17/20으로 같아, 목록의 앞쪽인 pc를 골랐다
- 세 칸 모두 E1 격자(main)는 최선이 아니었다
- 참고(선택 규칙 밖): held 계획 시점 1-step 오차는 PushT k=4에서 mse 0.015, p999 0.019, pc 0.022 순이다

## 5. H3b (사전등록 v5 §7.3): DINO-WM 몫 불성립

| 칸 | B | D(B+det) %p [CI] | D(B+sh-iter) %p [CI] | 차 D(B+det) − D(B+sh-iter) %p [CI] | McNemar p(보조) | red [CI](기술) | 판정 |
|---|---|---|---|---|---|---|---|
| PushT k=3 | p999 | +3 [−2, 8] | +5 [0, 10] | −2 [−6, 2] | 0.625 | −0.67 [−4.00, 0.50] | — |
| PushT k=4 | pc | +4 [−2, 10] | +2 [0, 5] | +2 [−4, 8] | 0.754 | 0.50 [−1.33, 1.00] | — |
| PointMaze k=3 | last8 | +11 [2, 20] | −1 [−8, 6] | **+12 [4, 20]** | 0.012 | 1.09 [0.50, 3.00] | **통과** |

- **판정**: CI 하한 > 0인 칸이 1/3이다. 규칙(과반, 2칸 이상)에 못 미쳐 **DINO-WM 몫 불성립**이다
  - H3b는 계열마다 판정한다(v5 §7.3). TD-MPC2 몫은 10/19로 성립했다(E5_tdmpc2 보고)
- 짝 수는 세 칸 모두 100이다. 성공률(none / B+det / B+sh-iter)
  - PushT k=3: 83 / 80 / 78%
  - PushT k=4: 83 / 79 / 81%
  - PointMaze k=3: 85 / 74 / 86%
- **PushT에서는 dither가 더할 여지가 없다.** 최선 PTQ만으로 det 손상이 거의 사라진다(기술, 같은 쌍의 E1 칸과 짝 대비, `e5_posthoc_dinowm.py`)

| 칸 | B | E1 det D | D(B+det) − D(E1 det) %p [CI] | E1 sh-iter D | D(B+sh-iter) − D(E1 sh-iter) %p [CI] |
|---|---|---|---|---|---|
| PushT k=3 | p999 | +36 | −33 [−44, −22] | +3 | +2 [−4, +8] |
| PushT k=4 | pc | +24 | −20 [−30, −10] | +0 | +2 [−2, +6] |
| PointMaze k=3 | last8 | +11 | +0 [−8, +8] | −1 | +0 [−6, +6] |

- **PointMaze k=3**: last8은 held에서 18/20으로 골랐지만, eval에서는 E1 격자 det와 손상이 같다(+11, 불일치 쌍 9 대 9). 그래서 det 손상이 남고, sh-iter가 그것을 없앤다(red 1.09)
- 이 표의 짝 대비는 결과를 본 뒤 고른 기술이다(사후)

## 6. QAT (PushT k=4, 100 ep)

| 조건 | 성공 | D(E1 none 대비) %p [CI] | 출처 |
|---|---|---|---|
| E1 det(main 격자, 학습 없음) | 59% | +24 [14, 34] | 참고 |
| B+det(pc, 학습 없음, §5) | 79% | +4 [−2, 10] | 참고 |
| QAT+none | 56% | +27 [19, 36] | 기술 |
| **QAT+det** | 77% | **+6 [−1, 13]** | 사전등록 규칙의 대상 |
| QAT+sh-iter | 61% | +22 [14, 30] | 기술 |

- **규칙 판정(설계 §6.5, v5 §7.3)**: D(QAT+det)의 CI 상한이 13%p로 5%p 이상이다. 그래서 "dither의 가치를 학습 없는 PTQ 처방으로 한정해 보고한다"는 조건은 **걸리지 않는다**
  - 규칙 문구 확인: v5 §7.3의 DINO-WM 항목은 QAT 칸과 평가만 적고 수치 문턱을 다시 쓰지 않았다
  - 설계 §6.5의 규칙("QAT가 문제를 없애면 …")은 두 계열 공통이다. 그 조작적 정의는 v5 §7.3 TD-MPC2 항목의 "QAT+det 손상의 CI 상한이 5%p 아래"뿐이다
  - DINO-WM은 성공률 과제라 5%p를 그대로 썼다(미리 쓴 코드와 같다). D는 원래 모델의 none 대비다(TD-MPC2와 같음)
- **v5.3 §10.1 양상 점검**(같은 쌍의 짝 대비)
  - D(QAT+det) − D(B+det) = **+2%p [−5, +10]**, McNemar p 0.79(불일치 쌍 8 대 6)
  - 점추정은 QAT+det가 B+det보다 2%p 나쁘다. 하지만 짝 CI가 0을 포함해 둘을 가를 수 없다
  - TD-MPC2에서는 11칸 모두 QAT+det가 B+det보다 크게 나빴고 CI가 겹치지 않았다. DINO-WM은 그런 뚜렷한 양상이 아니다
  - 단서를 달지는 조정자가 정할 일이다(§9)
- **QAT는 학습한 격자(main)의 det 손상을 크게 줄였다**: D(QAT+det) − D(E1 det) = −18%p [−28, −8](불일치 쌍 24 대 6)
- **QAT+none(기술)**: 원래 모델의 none보다 27%p [19, 36] 나쁘다. 불일치 쌍이 27 대 0이다
  - QAT 모델 안에서는 det가 none보다 21%p [12, 30] 낫다(불일치 쌍 24 대 3)
  - 해석: QAT 예측기가 det로 양자화된 활성에 맞춰졌다. 그래서 비양자화로 쓰면 분포가 어긋난다. "학습이 모델 자체를 망쳤다"기보다 "det 양자화에 맞춘 모델"로 보는 것이 자료와 맞는다(기술)
- **QAT+sh-iter(기술)**: QAT+det보다 16%p [7, 25] 나쁘다. det fake-quant로 학습한 모델 위에 dither를 얹으면 해롭다. QAT 모델 자신의 none보다는 5%p [0, 10] 낫다(CI 경계 0)

## 7. 가중치 양자화 (탐색, 활성은 fp, 기술; PushT k=4 설정, 100 ep)

| 조건 | 성공 | D %p [CI] |
|---|---|---|
| w8rtn | 80% | +3 [−1, 8] |
| w4rtn | 28% | +55 [44, 66] |
| w4awq | 12% | +71 [62, 80] |
| w4sr1 | 7% | +76 [67, 84] |
| w4sr2 | 15% | +68 [58, 77] |
| w4sr4 | 12% | +71 [62, 80] |

| 대비(D_a − D_b) | 차 %p [CI] | McNemar p |
|---|---|---|
| w4rtn − w4sr1 | −21 [−31, −11] | 0.0002 |
| w4rtn − w4sr2 | −13 [−24, −1] | 0.047 |
| w4rtn − w4sr4 | −16 [−27, −5] | 0.007 |
| w4rtn − w4awq | −16 [−26, −6] | 0.004 |
| w8rtn − w4rtn | −52 [−63, −41] | 4e−14 |
| w4sr1 − w4sr2 | +8 [−1, +17] | 0.12 |
| w4sr2 − w4sr4 | −3 [−13, +7] | 0.69 |

- w4sr\<K\>는 확률적 반올림 사본 K개를 CEM 반복 i에 사본 i mod K로 돌려 쓴다. K = 1은 계획(= episode) 내내 고정된 사본 하나다
- W8 RTN은 손상이 없다(TD-MPC2와 같다)
- W4에서는 RTN이 가장 낫다. SR 세 가지와 AWQ식이 모두 RTN보다 13~21%p 나쁘고, CI가 0을 넘지 않는다
- **SR 사본 수 K의 방향**
  - K = 1 → 2는 점추정으로 8%p 낫다(CI가 0을 포함). K = 2 → 4는 차이가 없다(−3 [−13, +7])
  - TD-MPC2에서는 네 과제 모두 K = 1 > 2 > 4로 손상이 줄었고, K = 4가 세 과제에서 RTN보다 나았다
  - DINO-WM PushT에서는 K에 따른 단조 감소가 보이지 않고, SR이 RTN보다 나은 경우가 없다
- AWQ식은 TD-MPC2의 Meta-World·walker와 달리 RTN보다 나쁘다(cheetah와 같은 방향)

## 8. 계열 합산용 값 (조정자)
- [../results/E5_dinowm/family_points_e5_dinowm.json](../results/E5_dinowm/family_points_e5_dinowm.json)
  - H3b 칸별: B, held 행과 동률, D(B+det)·D(B+sh-iter)와 CI, 차와 CI, 통과, red, McNemar
  - H3b: 3칸 중 1칸 통과, 불성립
  - QAT: 조건별 D와 CI, 규칙 표시(false), 짝 대비, 학습 기록
  - 가중치: 조건별 D와 CI, 대비
- DINO-WM 몫: H3b 1/3 불성립, QAT 조건 걸리지 않음(CI 상한 13%p)

## 9. 결정이 필요한 것 (조정자)
1. **v5.3 §10.1 단서를 DINO-WM QAT에 달지**
   - 자료: D(QAT+det) +6 [−1, 13], D(B+det) +4 [−2, 10], 짝 차 +2 [−5, +10]
   - 선택지 A(문구의 점추정으로 읽음): QAT+det가 B+det보다 2%p 나쁘므로 "같은 양상"으로 보고 TD-MPC2와 같은 단서를 단다("이 QAT는 약한 기준선이라 'QAT로는 손상이 사라지지 않는다'는 결론을 내리지 않는다")
   - 선택지 B(짝 CI로 읽음): 차이가 구별되지 않아 같은 양상이 아니라고 본다. 단서 대신 "DINO-WM QAT+det는 최선 PTQ+det와 비슷하고, 학습한 격자의 det 손상을 줄였다(−18%p)"고 기술한다
   - 어느 쪽이든 규칙 판정(조건이 걸리지 않음)은 같다
   - 함께 볼 기술 사실: QAT+none +27%p, QAT+sh-iter가 QAT+det보다 16%p 나쁨. dither를 QAT 모델에 얹지 않고 "학습 없는 처방"으로 서술하는 것은 두 선택지 모두와 맞는다
2. **H3b 계열 결과의 서술**
   - TD-MPC2 몫은 10/19로 성립했고, DINO-WM 몫은 1/3으로 불성립이다
   - DINO-WM 불성립의 직접 원인은 PushT에서 최선 PTQ(p999, pc)가 det 손상을 33·20%p 줄여 남은 손상이 3~4%p뿐인 것이다
   - "강한 PTQ 위의 추가 이득"을 어떻게 쓸지는 사용자 결정이다
3. **PushT k=4 기준선 동률**: pc·p999·mse가 held 17/20으로 같아 목록 순서로 pc를 골랐다. 규칙대로이고, 다른 기준선의 eval은 돌리지 않았다(정보로만 적음)

## 10. 편차
- 분석 코드의 판정 계산은 바꾸지 않았다. 출력만 더했다(§3). 맥북 `code/dinowm/`와 서버 `~/pmx/A/dinowm_main/code/`의 md5가 같다
  - analyze_e5 85b9cb02…, check_e5 eca5f20d…, e5_posthoc_dinowm fbc8bf61…
- §5의 E1 대비 표와 §6의 QAT 해석은 기술 또는 사후다. 판정에 쓰지 않았다
- QAT 학습은 사전등록 문구("약 2 GPU-시간") 그대로 120분 예산에서 멈췄다(10,083 step)

## 11. 비용과 시간
- RunPod: 쓰지 않음($0)
- 서버 GPU(job GPU 부분 합, 칸마다 한 장)
  - held: PushT k=3 1.32, PushT k=4 1.32, PointMaze k=3 0.49 GPU-h(GPU 1)
  - eval: PushT k=3 1.98(GPU 1), PushT k=4 1.98(GPU 0), PointMaze k=3 0.78(GPU 1)
  - 가중치: 6변형 각 0.90(GPU 0·1에 셋씩)
  - QAT 학습 2.0(GPU 1, 120분), QAT 평가 2.82(GPU 1)
  - 합계 약 18.1 GPU-h. 벽시계 10-23T22:22Z ~ 10-24T11:51Z(두 GPU)
- 분석: 서버 CPU 수 초. 맥북에서는 문법 검사와 결과 읽기만 했다

## 12. 파일
- `results/E5_dinowm/e5_dinowm_summary.json`: 전체 결과(H3b 칸, QAT, 가중치와 대비, QAT 학습 요약)
- `results/E5_dinowm/family_points_e5_dinowm.json`: 계열 합산용
- `results/E5_dinowm/e5_dinowm_posthoc.json`: §5 E1 대비, QAT 자신의 none 대비(사후)
- `results/E5_dinowm/check_e5.json`: 자료 점검(선택 재계산, E1 연결, QAT 로그)
- `results/E5_dinowm/qat_pusht_k4_log.json`: QAT 학습 로그 사본
- `results/E5_dinowm/E5_dinowm_{held,eval,w,qat}/`: job JSON 192개와 selection.json
- 서버만: `~/pmx/A/dinowm_main/analysis/E5_dinowm/`(자동 생성 표 `.md`, 출력 `.txt`), QAT 가중치 `~/pmx/A/dinowm_main/qat/pusht_qat_k4/`
