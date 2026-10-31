# A RSS 확장 실험 (E8–E10) 시간순 기록

- 2026-10-31T08:46Z 사용자 지시(10-31): "우선 4까지 포함해서 runpod에서 실험 진행해줘, 예산은 30달러. 1~4가 끝나면 논문 작성을 시작해 Stage 1에 제출."
- 2026-10-31T08:46Z RunPod 도구 `runpod_rss/` 생성(`runpod_main` 복사): 접두어 `arx`, START_BALANCE $98.23(08:37Z, 실행 중인 pod 없음), 감시 한도 $28(사용자 상한 $30), HARD_DEADLINE 11-15. 실험별 상한 E8 $3, E8b $2, E9 $7, E10 $13. `pod_up.sh`에 `--min-cuda-version`(기본 12.6) 추가, `budget.py`의 상한 키를 `<실험>.<n>` 태그에 맞게 고침. 감시 스크립트 `budget.py watch 300` 시작(08:43Z).
- 2026-10-31T08:46Z RunPod 재고(08:40Z): RTX A5000 없음. A40 $0.59/h(High), RTX 4090 $0.89/h(Medium), RTX A6000 $0.59/h(Low), L40S $1.09/h(Low), RTX 5090 $1.19/h(Medium), RTX PRO 4500 Blackwell $0.72/h(Medium). 그래서 E8은 A40(E7 A5000과 같은 GA102 계열)에서 잰다.
- 2026-10-31T08:46Z **설계 고정**: [DESIGN_ext.md](DESIGN_ext.md) sha256 `c660a10dc72ba044903996c0d8342cd90bc1941beeac9d654cbd00d1b050850c`. 어느 실험의 자료도 생기기 전이다.
