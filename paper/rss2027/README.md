# RSS 2027 Stage 1 확장 초록 (A: Planners Exploit Deterministic Error)

- 마감: **2026-12-04 AoE**, OpenReview(`roboticsfoundation.org/RSS/2027/Conference`)
- 분량: 본문 5쪽 + 참고문헌 1쪽. Stage 1에는 부록·보충자료·영상을 받지 않는다(6쪽 밖은 심사하지 않음).
- 주장: [../../rss_ext/CLAIMS_rss.md](../../rss_ext/CLAIMS_rss.md)의 C1~C6. 사용자가 2026-11-01(한국 시간)에 "추천대로 승인"했다(§7 기록).
- 공식 CFP 확인: 2026-11-01(roboticsconference.org/information/cfp)
  - 이중맹검: 이름·소속·감사의 글을 쓰지 않고, 외부 링크(익명 링크 포함)도 넣지 않는다. 파일 metadata를 익명으로 한다. 자기 이전 연구는 남의 연구처럼 인용한다.
  - PDF는 **Type 1 글꼴만** 담아야 한다. 쪽 번호와 "draft" 표시는 넣지 않는다.
  - LLM 사용: 허용된다. 공개는 의무가 아니지만 "highly encouraged"다. **환각 인용은 desk reject다**(자동 검사).
  - 동시 투고 금지. 예외는 proceedings가 없는 발표와 arXiv 같은 preprint다. preprint는 심사자에게 신원이 드러날 위험이 있다.

## 지금 상태 (초안 v1, 2026-11-01)

| 항목 | 상태 |
|---|---|
| 본문 `main.tex` | 초안 v1. tectonic으로 5쪽(본문 약 4쪽 + 참고문헌)이다. 본문은 약 1쪽 남는다 |
| 그림 3개 | `build/make_figs.py`가 결과 파일에서 그린다. 글자를 path로 그려서 그림 PDF에는 글꼴이 없다 |
| 표 1 | 사전등록 가설 전체(E1~E10)의 판정 |
| 수치 점검 | `build/verify_numbers.py`가 본문의 수치 44개를 결과 파일에서 다시 계산해 대조한다. 44/44 통과 |
| E8c(D3) | A40에서 측정 중이다. 끝나면 6절 문단, 그림 3, 표 1에 넣는다(설계 `rss_ext/DESIGN_E8c.md` §6의 규칙) |
| 참고문헌 | 2026년 arXiv 8편은 저자·제목을 arXiv 페이지에서 확인한 뒤 확정한다(`rss_ext/RELATED_recheck_2026-11-01.md`). 그 전까지 `refs.bib`에 TODO로 남아 있다 |
| 공식 템플릿 | 아직 받지 않았다. 이 초안은 RSS 템플릿의 바탕인 IEEEtran conference 형식이다. 제출본은 공식 템플릿(LaTeX, pdfLaTeX)으로 옮겨 컴파일한다 |

## 빌드

```bash
cd "/Users/song-yonghwi/Documents/physical ai - Neural Networks/A_dithered_world_models/paper/rss2027" && python3 build/make_figs.py && python3 build/verify_numbers.py && tectonic -X compile main.tex
```

- tectonic(XeTeX)에서는 `times`가 매핑되지 않는다. 그래서 초안은 Times와 글자 폭이 같은 TeX Gyre Termes(OpenType)를 쓴다. 쪽수는 pdfLaTeX + Times와 거의 같다.
- **제출본은 pdfLaTeX로 만든다**(Overleaf 또는 TeX Live). `\usepackage{times}` 분기가 자동으로 쓰인다. 맥북에는 pdflatex가 없다.

## 제출 전 점검표

1. **공식 템플릿**
   - `paper-template-latex.tar.gz`(roboticsconference.org/docs/, 115 KB, 2026-10-15 수정)를 받아서, 이 원고의 본문을 그 안으로 옮긴다.
   - 내려받기는 사용자 허락을 받고 한다. Overleaf의 RSS 템플릿을 써도 된다.
   - 서식(글꼴, 여백, 간격)은 바꾸지 않는다. 바꾸면 desk reject다.
2. **글꼴**: pdfLaTeX로 만든 PDF를 Acrobat의 문서 속성 → 글꼴에서 연다. 모두 Type 1(또는 Type 1C)인지 확인한다. 그림 PDF에는 글꼴이 없다.
3. **분량**: 본문 5쪽 + 참고문헌 1쪽. 참고문헌이 5쪽 안에서 시작해도 된다.
4. **익명**
   - PDF metadata에 저자가 없는지 확인한다(`hypersetup`이 비워 둠).
   - 코드·GitHub 링크를 넣지 않는다.
   - JKAIA 포스터(proceedings에 실리지 않고 저작권을 양도하지 않음)는 인용하지 않는다.
5. **CV의 "RSS 2027" 표기(D4)**
   - 제출 직전에 `~/Documents/대학원 지원/CV/cv.yaml`의 A 항목 두 줄을 고친다. 바꿀 줄은 `note`의 "to be submitted to … RSS 2027"과 `venue_short`의 "RSS 2027 in prep."이다.
   - 이 저장소에 push하면 공개 홈페이지·GitHub 프로필·CV PDF가 바로 바뀐다. 지원서용 PDF에서도 빠진다. push는 사용자가 정한다.
6. **LLM 사용 공개**: 의무는 아니지만 권장된다. 어디에(OpenReview 양식 또는 각주) 무엇을 쓸지 사용자가 정한다.
7. **preprint**: arXiv에 올리면 허용되지만 심사자에게 신원이 드러날 수 있다. 올릴지 사용자가 정한다.
8. **OpenReview**: 공저자(이건명 교수)를 포함한 모든 저자의 OpenReview 프로필이 필요하다.
9. **인용**: 모든 항목을 원 출처(arXiv 페이지, 출판사)에서 확인했는지 `REFERENCES_CHECK.md`로 확인한다.

## 파일

- `main.tex`, `refs.bib`: 원고
- `figures/`: 그림 PDF(본문용), PNG(미리보기)
- `build/make_figs.py`: 그림. 원자료는 `main/results/`(E1, E2, E3)와 `rss_ext/`(E8, E10)
- `build/verify_numbers.py`: 본문 수치 대조
