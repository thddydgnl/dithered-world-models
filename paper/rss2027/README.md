# RSS 2027 Stage 1 확장 초록 (A: Planners Exploit Deterministic Error)

- 마감: **2026-12-04 AoE**, OpenReview(`roboticsfoundation.org/RSS/2027/Conference`)
- 분량: 본문 5쪽 + 참고문헌 1쪽. Stage 1에는 부록·보충자료·영상을 받지 않는다(6쪽 밖은 심사하지 않음).
- 주장: [../../rss_ext/CLAIMS_rss.md](../../rss_ext/CLAIMS_rss.md)의 C1~C6. 사용자가 2026-11-01(한국 시간)에 "추천대로 승인"했다(§7 기록).
- 공식 CFP 확인: 2026-11-01(roboticsconference.org/information/cfp)
  - 이중맹검: 이름·소속·감사의 글을 쓰지 않고, 외부 링크(익명 링크 포함)도 넣지 않는다. 파일 metadata를 익명으로 한다. 자기 이전 연구는 남의 연구처럼 인용한다.
  - PDF는 **Type 1 글꼴만** 담아야 한다. 쪽 번호와 "draft" 표시는 넣지 않는다.
  - LLM 사용: 허용된다. 공개는 의무가 아니지만 "highly encouraged"다. **환각 인용은 desk reject다**(자동 검사).
  - 동시 투고 금지. 예외는 proceedings가 없는 발표와 arXiv 같은 preprint다. preprint는 심사자에게 신원이 드러날 위험이 있다.

## 지금 상태 (초안 v2, 2026-11-01: E8c 반영)

| 항목 | 상태 |
|---|---|
| 본문 `main.tex` | 초안 v2. tectonic으로 정확히 6쪽이다. 본문은 5쪽 오른쪽 단 위에서 끝나고, 참고문헌이 6쪽을 채운다(6쪽 오른쪽 단에 약 48pt, 참고문헌 약 5줄이 남음). "사전등록"이라는 말은 쓰지 않는다. 대신 "분석 계획을 자료 전에 고정했다"고 쓴다(10-31 사용자 결정) |
| 그림 3개 | `build/make_figs.py`가 결과 파일에서 그린다. 글자를 path로 그려서 그림 PDF에는 글꼴이 없다. 그림 3은 (a) E8 world-model rollout, (b) E8c 계획 step 전체(정수 W8A4와 미리 바꾼 BF16; 점은 N의 중앙값, 막대는 N 범위)다 |
| 표 1 | 자료 전에 고정한 가설 전체(E1~E10, E8c의 L2c·S1~S3 포함)의 판정 |
| 수치 점검 | `build/verify_numbers.py`가 본문의 수치 문구 62개를 대조한다. 그중 18개(E8 일부, E8c 전부)는 결과 파일에서 다시 계산한 값이 본문 숫자와 글자 그대로 같은지 자동으로 본다. 62/62 통과 |
| E8c(D3) | 끝났다(11-01). L2c·S2 불성립, S1·S3 성립. C4 수정안 E1~E5를 사용자가 승인했다(`rss_ext/CLAIMS_rss.md` §9). 초록, 서론, 기여, 표 1, 6절, 그림 3(b), 한계, Stage 2에 넣었다. rollout의 dither 상한은 두 측정을 모두 덮는 2.3%다(E8 2.07%, E8c 2.29%) |
| 참고문헌 | 40편 확정(11-01). 2025–2026년 23편은 경쟁 연구 재조사(`rss_ext/RELATED_recheck_2026-11-01.md`)와 항목별 확인(`rss_ext/related_2026.bib`)에서 왔다. 확인 기록은 [REFERENCES_CHECK.md](REFERENCES_CHECK.md)에 있다. E8c를 넣었을 때 7쪽으로 넘쳐서 표현을 줄였다(아래 '쪽수 맞춤 기록') |
| 공식 템플릿 | 11-01 사용자 허락을 받아 받았다(`paper-template-latex.tar.gz`, 115 KB). 내용은 예전 RSS 템플릿 그대로다(IEEEtran.cls V1.7a, `times`, `natbib`+`plainnat`, `hyperref`, `\pdfinfo`). 그 `IEEEtran.cls`를 이 폴더에 두었고, `main.tex`는 템플릿 구조를 따른다. 남은 일은 pdfLaTeX로 컴파일하는 것이다 |

## 빌드

```bash
cd "/Users/song-yonghwi/Documents/physical ai - Neural Networks/A_dithered_world_models/paper/rss2027" && python3 build/make_figs.py && python3 build/verify_numbers.py && tectonic -X compile main.tex
```

- tectonic(XeTeX)에서는 `times`가 매핑되지 않는다. 그래서 초안은 Times와 글자 폭이 같은 TeX Gyre Termes(OpenType)를 쓴다. 쪽수는 pdfLaTeX + Times와 거의 같다.
- **제출본은 pdfLaTeX로 만든다**(Overleaf 또는 TeX Live). `\usepackage{times}` 분기가 자동으로 쓰인다. 맥북에는 pdflatex가 없다.

## 제출 전 점검표

1. **공식 템플릿**
   - 받아서 반영했다(11-01). Overleaf에 이 폴더를 올리고 pdfLaTeX로 컴파일한다(`main.tex`, `refs.bib`, `IEEEtran.cls`, `figures/*.pdf`).
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
9. **인용**: `REFERENCES_CHECK.md`의 남은 일을 끝낸다. OpenReview 7편을 브라우저로 확인하고, 위협이 큰 두 편(3VSa1Kx7HD, cVCKr4bBU2)은 본문을 읽는다. 12월 초에 새 버전이 나왔는지 다시 본다.

## 파일

- `main.tex`, `refs.bib`: 원고
- `figures/`: 그림 PDF(본문용), PNG(미리보기)
- `build/make_figs.py`: 그림. 원자료는 `main/results/`(E1, E2, E3)와 `rss_ext/`(E8, E8c, E10)
- `build/verify_numbers.py`: 본문 수치 대조

## 쪽수 맞춤 기록 (11-01, E8c 반영 때)

E8c 문구를 넣으니 참고문헌이 7쪽으로 약 11줄 넘쳤다. 그래서 아래 표현을 줄였다. 주장과 수치는 빼지 않았다.

- 서론 끝 문단: 분석 계획의 자세한 설명을 한 문장으로 줄였다. 자세한 내용은 3절 "Analysis plan"과 표 1 캡션에 있다.
- 관련 연구 1문단: "후처리 양자화 위에서 dither 이득이 경계적"이라는 반복을 뺐다. 이 내용은 한계 문단과 표 1(H3b)에 있다.
- 3절 "Analysis plan": 단계 목록과 "kernel tiling" 예를 줄였다.
- 4·5절: det+shift의 정의 반복(3절에 정의 있음)을 뺐다.
- 6절: rollout 측정 설명의 괄호(반복 수, horizon)는 그림 3 캡션으로 옮겼다. FP4 문단의 도입 문장과 결론 문장을 줄였다.
- 그림 3 캡션, Stage 2 1·3번을 짧게 고쳤다.
