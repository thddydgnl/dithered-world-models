# TMLR 추가 실험 작업 규칙 (작업자는 시작 전에 끝까지 읽는다)

- 프로젝트 루트(공백 포함, 항상 따옴표): `/Users/song-yonghwi/Documents/physical ai - Neural Networks`
- 논문 폴더: `A_dithered_world_models/` (이하 `A/`). 이 문서와 설계는 `A/tmlr_ext/`에 있다.
- 기준 설계: [DESIGN_tmlr.md](DESIGN_tmlr.md) v1.0(고정판, sha256은 [LOG.md](LOG.md)). 통계 규약은 `A/main/DESIGN_main.md` v5.3과 `A/rss_ext/DESIGN_ext.md` §0.3을 따른다.

## 0. 역할

- 조정자(메인 세션)가 묶음(td, dw, mt, k) 하나씩을 작업자에게 맡긴다. 작업자는 맡은 묶음을 구현하고, RunPod에서 돌리고, 분석하고, 보고한다.
- **가설, 지표, 문턱, 칸, seed는 바꾸지 않는다.** 설계대로 할 수 없거나 설계를 바꿔야 할 것 같으면 추측해서 바꾸지 말고, 거기서 멈추고 조정자에게 보고한다(9절). 판단은 조정자가 한다.
- 운영 선택(구현 방법, 워커 수, 실행 순서, block 길이, pod 배치의 세부, 재시도)은 작업자가 정하고 `PLAN.md`와 `LOG.md`에 적는다.
- 문서(PLAN, LOG, REPORT)는 한국어, 코드와 주석은 영어로 쓴다. 시각은 UTC(`2026-11-21T18:00Z`)로 적는다.

## 1. 하지 않는 것

- 맥북에서 계산하지 않는다. 맥북에서는 파일 편집, `python3 -m py_compile` 같은 문법 검사, 몇 초 안에 끝나는 순수 파이썬 점검, 결과 읽기만 한다. 모델·데이터 다운로드도 맥북에서 하지 않는다.
- 대학 서버(`kiis-mvf-gpu`)를 쓰지 않는다. 이번 단계는 전부 RunPod다.
- **git commit과 push를 하지 않는다.** 조정자가 한다.
- 다른 묶음의 pod, 다른 접두어의 pod(`arx-*`, `pmx-*`, `pai-*`, `nnc-*`, `wmc-*`, `bx7-*`)는 절대 건드리지 않는다.
- 사용자의 파일을 지우지 않는다. 지워도 되는 것은 자기 pod 안의 파일과 자기가 만든 임시 파일뿐이다.
- 다른 작업자의 폴더(`A/tmlr_ext/<다른 묶음>/`)를 고치지 않는다. 읽는 것은 괜찮다.

## 2. RunPod 사용법 (`runpod_tmlr/`)

- 쓰는 스크립트: `pod_up.sh`, `rsh.sh`, `sync.sh`, `pod_down.sh`, `budget.py`. 사용법은 `runpod_tmlr/README.md`다.
  - 생성: `TERM_H=<h> bash "<루트>/runpod_tmlr/pod_up.sh" <tag> "<GPU1>|<GPU2>"`. GPU 이름은 `NVIDIA A40`, `NVIDIA RTX A5000`, `NVIDIA GeForce RTX 4090`처럼 runpodctl의 gpuId를 쓴다.
  - SECURE만 쓴다(스크립트가 이미 그렇다). `TERM_H`: td 8, k 8, mt 9, dw 10.
  - 태그: td.1~td.6, dw.1~dw.4, mt.1~mt.4, k.1. 자동 종료 등으로 같은 일을 새 pod에서 이어 가면 td.3b처럼 글자를 붙인다. 새 pod는 **같은 GPU 종류**여야 한다(짝 규칙, 4절).
  - 끝나면 결과를 당겨 온 뒤 `pod_down.sh <tag>`로 지우고 `python3 "<루트>/runpod_tmlr/budget.py" status`에서 `gone`을 확인한다.
- 지난 실행에서 배운 것
  - 원격에서 오래 도는 일은 `bash rsh.sh <tag> 'setsid nohup bash -c "<명령>" > /workspace/atm/logs/<이름>.log 2>&1 < /dev/null &'`로 띄운다. 앞에 `cd … &&`를 붙이면 ssh가 붙잡혀 Bash 호출이 시간 초과된다(E10).
  - `pkill -f <패턴>`의 패턴이 자기 명령줄과 겹치면 자기 자신을 죽인다(E10). pid 파일로 끈다.
  - `sync.sh`에는 pod의 rsync가 필요하다(`apt-get install -y rsync`).
  - ssh·rsync가 멈출 수 있다. 맥북 쪽 명령은 `perl -e 'alarm 600; exec @ARGV' …`로 감싼다.
  - Bash에서 `cd`를 쓰면 세션의 작업 폴더가 바뀐다. 절대 경로를 쓴다.
  - 기다릴 때 맥북에서 `sleep`을 쓰지 않는다. 원격 명령이 끝나기를 기다리거나 Monitor 도구(until 반복)를 쓴다.
  - A40의 CPU 할당은 약 7.65코어였다. TD-MPC2는 MPS 워커 8개가 가장 빨랐다(E9).
- 예산: `budget.py status`로 묶음별 지출을 본다. 묶음 상한은 td $24, mt $12, k $6, dw $35이고 감시 스크립트가 넘으면 그 묶음의 pod를 지운다. 본 실행이 끝나기 전에 상한의 80%에 닿을 것 같으면 멈추고 보고한다.

## 3. pod 안의 표준 배치 (조정자가 30분마다 이것을 읽는다)

- `/workspace/atm/code/`: 돌리는 코드 전부(맥북의 `A/tmlr_ext/<GROUP>/code/`와 같아야 한다)
- `/workspace/atm/out/`: **작은 결과 파일만**(job별 JSON·CSV, 분석 결과, 작은 PNG). 조정자가 30분마다 `runpod_tmlr/pull_all.sh`로 `A/tmlr_ext/<GROUP>/results/raw/<tag>/`에 당겨 온다. 파일 하나 20MB 미만, pod당 합 300MB 미만.
- `/workspace/atm/big/`: 큰 중간 파일(후보 집합, 보정 활성 등). 당겨 오지 않는다.
- `/workspace/atm/logs/`: 로그
- `/workspace/atm/code/status.sh`: **한 줄**을 5초 안에 출력한다.
  `tag=<tag> phase=<setup|smoke|main|analysis|done> done=<a>/<b> failed=<c> driver=<alive|dead|finished> eta=<UTC 또는 -> updated=<UTC> note=<짧은 메모>`
- `/workspace/atm/code/restart.sh`: 대기열 driver를 다시 띄운다. 여러 번 불러도 안전해야 한다(이미 살아 있으면 아무것도 하지 않음, 끝난 job은 건너뜀, 실패한 job은 2번까지 다시 넣음).
- driver는 이어 돌 수 있어야 한다: job마다 결과 파일 하나, 원자적 쓰기(임시 파일 → rename), 결과가 있으면 건너뜀. 적응 규칙(E12 확장, 두 단계 표본, E16의 λ 선택)도 driver가 스스로 적용한다.

## 4. 짝 규칙 (DESIGN_tmlr.md §0.3)

- 한 (과제, 체크포인트)의 none과 모든 변형은 한 pod에서 돈다. DINO-WM은 한 (과제, k) 칸과 그 none이 한 pod에서 돈다.
- 이어 돌 새 pod는 같은 GPU 종류여야 한다. 다른 종류로 옮겨야 하면 그 칸들은 none부터 새 pod에서 다시 돈다.
- 실행 전 0.7절 점검 1(코드 경로 불변)을 첫 pod에서 반드시 통과한다.

## 5. 기록

- `A/tmlr_ext/<GROUP>/LOG.md`: 시간순 기록(pod, GPU, 오류와 수정, 지출)
- `A/tmlr_ext/<GROUP>/PLAN.md`: **본 실행 전에** 쓴다. 구현 선택, 0.7절 점검 결과, 처리량과 워커 수, pod별 배치와 예상 종료 시각, 비용 추정, 분석 코드의 md5(분석 코드는 본 실행 전에 만들어 가짜 자료나 스모크 자료로 끝까지 돌려 본다)
- `A/tmlr_ext/<GROUP>/REPORT.md`: 분석 뒤에 쓴다. 맨 앞에 판정 표(가설, 판정, 핵심 수치와 CI), 그다음 칸별 표, 편차, 비용과 시간. 모든 수치에 출처 파일 경로를 붙인다.
- `A/tmlr_ext/<GROUP>/code/`: 실제로 돌린 코드 전부. 결과 요약은 `A/tmlr_ext/<GROUP>/results/`.
- Write 도구가 실패하면 Bash heredoc으로 쓴다.

## 6. 단계

- **단계 A**(처음 맡을 때)
  1. 설계와 기준 코드를 읽고, `A/tmlr_ext/<GROUP>/code/`에 구현한다.
  2. 첫 pod를 띄워 설치하고 DESIGN_tmlr.md §0.7의 점검 1~4를 돈다.
  3. 분석 코드를 만들어 스모크 자료로 끝까지 돌린다.
  4. `PLAN.md`를 쓴다.
  5. 나머지 pod를 띄우고 모든 pod에서 본 실행 driver를 시작한다. 각 pod의 첫 job 몇 개가 정상으로 끝나는지, `status.sh`와 `restart.sh`가 동작하는지 확인한다.
  6. 조정자에게 보고하고 끝낸다(8절의 형식). 본 실행이 끝나기를 기다리지 않는다.
- **실행 중**: 조정자가 30분마다 점검하고 결과를 당겨 온다. 오류가 있으면 조정자가 메시지로 알린다. 그때 고쳐서 다시 돌리고 짧게 보고한다.
- **단계 B**(조정자가 메시지로 시작시킬 때): pod에서 분석을 돌리고(맥북에서 돌리지 않는다), 결과를 당겨 와 `REPORT.md`를 쓰고, 모든 pod를 지우고, 보고한다.

## 7. 수치 점검

- 판정 수치는 결과 파일에서 다시 계산할 수 있어야 한다. 보고 전에 핵심 수치 몇 개를 원자료에서 다시 세어 대조한다.
- 조정자도 원자료에서 다시 계산해 대조한다. 둘이 다르면 원자료가 기준이다.

## 8. 보고 형식 (40줄 안팎)

- pod: 태그, GPU, $/h, 생성 시각, 자동 종료 시각, 맡은 일
- 점검(§0.7 1~4): 통과·실패와 숫자
- 처리량과 예상 종료 시각(UTC), 지금까지의 지출
- status·restart·pull 명령이 동작하는지
- 편차와 운영 선택
- 조정자가 정해야 할 것(선택지와 근거)

## 9. 멈추고 보고하는 경우

- §0.7 점검이 실패했고, 고치려면 설계를 바꿔야 할 때
- 설계에 적힌 대로 잴 수 없을 때(예: 새 분할의 none 성공률이 기준 밖, 바뀐 계획기의 none이 자격 미달은 설계가 이미 처리하므로 멈추지 않는다)
- 예산이 묶음 상한의 80%를 넘을 것 같을 때
- 이때 보고에 무엇이 문제인지, 근거(파일 경로와 숫자), 선택지 둘 이상과 각각이 주장에 미치는 영향을 적는다. 조정자가 정해 메시지로 알린다.
