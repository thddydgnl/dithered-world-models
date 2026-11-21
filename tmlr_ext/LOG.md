# A TMLR 추가 실험 — 조정자 기록 (UTC)

- 2026-11-21T17:52Z 무렵 사용자 승인: DESIGN_tmlr.md 초안 v0.1을 추천대로 진행, 30분마다 점검, 오류는 고쳐서 다시 실행. 밤사이 결정은 DESIGN_tmlr.md §10대로(상한 $80, 설계 변경은 Fable 5.2 max 판단, 끝나면 push, 완료·중대 문제만 알림).
- 17:56Z `runpod_tmlr/` 생성(`runpod_rss/`에서 상태 파일 빼고 복사). 접두어 atm, LIMIT $77, START_BALANCE $83.42, HARD_DEADLINE 2026-11-25T00:00Z, caps td $24 / mt $12 / k $6 / dw $35, TERM_H 기본 8. 새 도구 check_all.sh(30분 점검, 읽기 전용), pull_all.sh(결과 회수). 맥북은 AC 전원, 잠자기 방지(Amphetamine, Claude keep-awake).
- 2026-11-21T17:59:00Z **DESIGN_tmlr.md v1.0 고정**: sha256 `36d69ea82ef90a50e8622f03b5dc7fadadd92b227121cc0015cf4174235d0cc4`. 고정 전에 어떤 E11~E17 자료도 없다. 고정 전 운영 변경(가설·문턱·칸 그대로): 상한에 맞춘 태그별 상한과 묶음별 TERM_H, §8.5 작업 분담, §8.6 판단 절차, §9·§10 결정 기록.
- 2026-11-21T17:59:00Z WORKER_RULES.md 작성(작업자 공통 규칙: pod 표준 배치, status.sh·restart.sh, 짝 규칙, 단계 A·B, 보고 형식).
