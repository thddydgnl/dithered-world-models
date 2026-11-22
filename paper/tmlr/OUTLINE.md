# TMLR 원고 구성 — Planners Exploit Deterministic Error (논문 A)

- 작성: 2026-11-22, 골격 v0. 같은 날 통합판(그림 1~7, 부록 A~K, 참고문헌 확인)으로 고쳤고, 이어서 검증 보고([`../../tmlr_ext/REVIEW_fable_2026-11-22.md`](../../tmlr_ext/REVIEW_fable_2026-11-22.md))를 반영했다(5절). 원고: [`main.tex`](main.tex). 이 문서는 절마다 무엇을 근거로 쓰는지와 남은 일을 적는다.
- 근거
  - 주장: [`../../tmlr_ext/CLAIMS_tmlr.md`](../../tmlr_ext/CLAIMS_tmlr.md) v2.1(§5.2까지). 이보다 넓게 쓰지 않고, §3의 문장은 쓰지 않는다.
  - 구성: [`../../tmlr_ext/PLAN_tmlr.md`](../../tmlr_ext/PLAN_tmlr.md) §4.3(본문), §4.4(부록), §4.6(RSS 문구 수정), §5(양식)
  - 결과: [`../../tmlr_ext/RESULTS_tmlr.md`](../../tmlr_ext/RESULTS_tmlr.md)(§7 재분석 R1~R5), 묶음 보고 4개, `main/reports/FINAL_A.md`
- 양식: `tmlr.sty`·`tmlr.bst`·`fancyhdr.sty`·`math_commands.tex`는 B 폴더 사본과 바이트 단위로 같다(sha256 앞 8자리 `816214ff`·`306fd454`·`3d292254`·`90473c4d`). 고치지 않는다.

## 1. 절별 근거와 상태

| 절 | 주장 | 그림·표 | 상태 |
|---|---|---|---|
| 초록 | 한 줄 주장(§1) + C1·C4·C7 수치, 경계(DMC 반복 수, FP4) | — | 초안 완료(212단어) |
| 1 서론 | C1~C7의 틀. 동기는 "저비트 산술 장비"(PLAN 4.5), 새로움은 ③·④(PLAN 4.9) | 그림 1 | 초안 완료. LLM 각주는 사용자 문구(손대지 않음) |
| 2 관련 연구 | 동시 연구 넷(WM-Quant, Search or Scoring?, LAQ, 공동 설계)과 선행(LoopQ, Goodhart in the Planner, 고전) | — | 초안 완료. 세부는 11-22 원 출처로 확인([`REFERENCES_CHECK.md`](REFERENCES_CHECK.md) 3절) |
| 3 설정 | 변형 정의(표 1, 오차 분산 열 포함), 단계·칸(표 2, 부록 C), 지표·통계, OEI·Δ_bias·Δ_pers·ρ_p | 표 1 | 초안 완료 |
| 4 det가 계획을 망친다 | C1(복제 포함, R4 [사후] 합산, [사후] D4·X17), C5(요약 통계, E13 여러 step 오차) | 그림 2; 표 3(생성, [사후] PushT 3-bit 열)은 부록 I | 초안 완료 |
| 5 계획기가 이용하는 것 | C2(두 성분, 이중 해리, [사후] D4와 PushT의 b 단서), C3(계획기 교차, R5 겹침), C6(317M) | 그림 3·4; 표 4(생성, [사후] PushT 3-bit 열)는 부록 I | 초안 완료 |
| 6 처방과 한계 | C4(비트 한계, R1·R2 견고성, 잡음 대 dither와 [사후] X5 비트 축·drawer-open 예외, DMC 반복 수), FP4 요약 | 그림 5·6 | 초안 완료 |
| 7 처방의 비용 | C7(정수 kernel, 지연, 미리 바꾼 BF16) | 그림 7 | 초안 완료. TF32 첫 층 [사후] 수치는 부록 H로 옮김 |
| 8 한계, 9 결론, Broader Impact | CLAIMS 각 항의 "한정" | — | 초안 완료 |
| 부록 A~K | PLAN 4.4 | 본문은 `appendix/app_A.tex`~`app_K.tex`(`\input`), 표는 대부분 `build/make_tables.py`가 생성 | 초안 완료. K의 저장소 URL은 `\ifarxiv`에만 |

## 2. PLAN과 다르게 한 것 (확인 필요)

- 표 3은 "계열별 변형 평균 손상"(E1·E10 본 단계와 E11·E14·eval8 복제를 나란히)이다. 성분 대비(고친 칸 수)는 표 4로 5.2절에 따로 두었다.
- PLAN 4.3의 지시대로 6.4절(FP4, QAT, 가중치)의 세부를 부록 D·G로 보내고 본문에는 요약 두 문장만 남겼다.
- PLAN 4.1의 "독자가 얻는 것 (3) 배치 전에 재야 할 오차"는 쓰지 않았다. CLAIMS C5가 ρ_p × σ_e/σ_J를 "기전과 맞는 요약 통계"로만 쓰도록 정했기 때문이다.
- 그림 1~7은 모두 `build/make_figs.py`가 TMLR 폭(6.5 in)으로 그리고 본문에 `\linewidth`로 넣는다.
- 본문 12쪽에 맞추느라 표 2(단계·칸)를 부록 C로, 표 3·4를 부록 I로 옮겼다(11-22 조정자 결정). 그림 캡션에서 본문과 겹치는 수치를 덜었다.

## 2.1 검증 반영으로 바뀐 구성 (2026-11-22)

- 5.4절(이론 요약)은 5.1절 끝 문단으로 합쳤다. 6장은 소절 없이 굵은 제목 문단으로 이어진다(Meta-World / 견고성 / DINO-WM·DMC / 새 잡음 대 dither / DMC와 반복 수 / 수 형식·학습·가중치; 수 형식 문단은 문체 검토 (2)-7에 따라 맨 끝으로 옮김). 12쪽을 맞추느라 6.1~6.3 제목을 없앴다. 옛 5.3절 첫 문단(반복·후보 수·표 크기)은 5.1절로 옮기고 5.3절은 계획기 교차만 둔다(S2).
- 6장 첫 문단이 같은 비트의 처방 결과와 E5를 잇는다(S1).
- 초록과 결론에 X6 단서(Meta-World에서는 정책만으로 계획기에 가깝다)를 넣었다. 사용자 승인, CLAIMS v2.2 §5.4.
- 그림 5의 Meta-World per-channel 줄은 부록 I의 새 그림(`fig_bitlimit_pc`)으로 옮겼다(F1). k\* 표시는 각 판 아래 띠로 옮겼다.
- 부록 A의 판 표는 익명판에서 해시 열을 빼고 arXiv판(`\ifarxiv`)에만 둔다(C4). D7(X6) 행을 더했다.
- 부록 I: E11 표를 두 표로 나눴고(C2), 체크포인트별 표(S11), X6 표(S6), E16 표의 교차 계획기 D 열(S6)을 더했다.

## 2.2 문장·문체 검토 반영 (2026-11-22, `tmlr_ext/REVIEW_opus_prose_2026-11-22.md`)

- 초록·결론·본문의 정책 단서는 CLAIMS v2.3 §5.5의 승인 문구로 통일했다("two error components that controlled variants separate", "lowers the smallest usable bit width", "nearly matches the unquantized planner", "the low-bit planner does not fall below the policy").
- 기호: 감쇠 η, 사본 수 M, 상관 길이 λ, 반복 수 I, Hessian 𝓗(부록 F). 약어(MPPI, CEM, DMC, W4A4·W8A4·W8A8·W4A4KV4, VLA, MPS, RTN/SR, QAT)는 첫 사용에서 풀어 썼다.
- 수 형식: 음수가 낀 범위는 "a to b"(`rng()`), E4 실행 수는 922{,}368. 변형 매크로로 시작하던 문장은 "The variant …" 꼴로 바꿨다(본문·부록).
- 5.2절 굵은 제목은 "persistence dominates / does not dominate"로, 그림 7 캡션과 한계 절은 줄였다. 부록 K의 "Design documents, logs and stage reports are in Korean" 문장은 PDF에서 뺐다(보충자료 README에는 남음).
- 재구성 (2)-3: 관련 연구에서 동시대 연구 두 편을 run-in 문단 "Concurrent work."로 떼었다(맺는 문장은 "The works above …"로 두 문단을 함께 가리킴). (2)-8: 7장 둘째 문단을 "Pre-converted BF16 is nearly lossless …"부터 나눴다.
- 참고문헌은 `\clearpage` 뒤 새 쪽에서 시작한다. 본문 마지막 쪽이 쪽 바닥까지 늘어나지 않고, `build.sh`가 본문 끝 위치를 그대로 잰다.

## 3. 수치 체계

- 본문의 결과 수치는 모두 `\nv{key}`다. `build/make_numbers.py`가 결과 파일에서 계산해 `numbers.tex`에 쓴다(값마다 출처 주석).
- `build/verify_numbers.py`
  1. `numbers.tex`가 결과 파일과 맞는지(다시 계산해 바이트 비교)
  2. 쓰인 key가 모두 정의됐는지
  3. "모두", "어느 칸에서도" 같은 문구가 자료와 맞는지(53개)
  4. 본문과 부록(`\input`을 펼쳐서)에 남은 숫자가 모두 등록됐는지(설계 상수, 정의, 칸 이름; 출처 포함). 등록 문구는 스크립트 안의 표와 `build/design_literals.txt`에 있다. 등록되지 않은 숫자는 UNCHECKED로 나온다
  5. 보고서에서 옮긴 값(10개: GPU 시간, CPU 시간, 비용, E4 실행 수; 결과 파일로 다시 계산하지 못함)을 따로 나열하고, 값이 출처 문서 본문에 있는지 대조한다
- 생성 표(`tables/`)는 스크립트가 만든 값이라 다시 검사하지 않는다(B와 같음).
- 빌드: `bash build/build.sh`(초안), `bash build/build.sh --strict`(투고: UNCHECKED·보고서 값·`\todo`가 하나라도 있으면 멈춘다).

## 4. 남은 일

- 상태(11-22 문체 검토와 재구성 (2)-3·8 반영 뒤): 본문(Broader Impact 포함)이 12쪽의 93%에서 끝난다(11.93쪽, 12쪽 아래에 약 48pt, 네 줄 남음). (2)-5(5.1절 run-in 제목)는 넣으면 0.7줄만 남아 넣지 않았다(조정자 조건: 한 줄 이상 남을 때만). 참고문헌은 13쪽에서 시작한다. 전체 43쪽. `\todo` 0개, `bash build/build.sh --strict` 통과(문구 점검 53/53, 등록 문구 380/380, 보고서 값 10/10). 
- 투고 전 할 일(조판, 11-22 조정자 결정): 쪽 바닥 맞춤(flushbottom) 때문에 절 제목이 다음 쪽으로 넘어가며 쪽 안 간격이 늘어난 곳을 투고 직전 마지막 조판에서 맞춘다(11-22 기준 4쪽의 3.3절 제목, 11쪽의 8장 제목). 내용 수정이 끝난 뒤에 한 번만 한다.
- 참고문헌: Kieburtz 1977의 저자 이름 전체와 초록, 고전 6편의 DBLP 대조, LoopQuant 투고(OpenReview) 재확인([`REFERENCES_CHECK.md`](REFERENCES_CHECK.md) 3.4절). ICLR 2027 익명 투고 7편은 12-16 결정 뒤에 다시 본다.
- 보충자료 익명화: 결과 JSON/CSV 6개에 홈 경로(`/home/...`)가 남아 있다(부록 작업자 보고).
- 보관 폴더 사본(`build/sync_papers_copy.sh --create`)과 arXiv v2(`build/make_arxiv.sh`)는 사용자 승인 뒤에 돌린다.
- 보충자료(다른 작업자의 `build/make_supplement.sh`)에 들어가야 하는 것: 익명화한 원고 소스 사본, 한국어 문서 안내와 표·그림별 결과 파일 README(부록 K, M4), 설계 문서 판의 해시 목록(부록 A, C4).
