#!/bin/bash
# Copy the built manuscripts into the author's paper archive (~/Documents/research/papers) and log the versions:
#   main.pdf (anonymous RSS version)  -> Dithered_Low-Bit_World_Models/RSS2027/
#   arxiv.pdf + upload bundle + form sheet -> Dithered_Low-Bit_World_Models/arXiv/   (when arxiv.pdf exists)
# Run after every rebuild (build/build.sh does):   bash build/sync_papers_copy.sh ["short note for the logs"]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RSS="$(cd "$HERE/.." && pwd)"
SRC="$RSS/main.pdf"
ARCHIVE="$HOME/Documents/research/papers/Dithered_Low-Bit_World_Models"
DEST="$ARCHIVE/RSS2027"
NAME="Dithered_Low-Bit_World_Models_RSS2027_원고.pdf"
DEST_ARXIV="$ARCHIVE/arXiv"
NAME_ARXIV="Dithered_Low-Bit_World_Models_arXiv_원고.pdf"
NOTE="${1:-}"
# files that determine the PDF; the dirty flag looks only at these
PDF_SOURCES=(main.tex arxiv.tex refs.bib IEEEtran.cls figures build/make_figs.py build/make_arxiv.py)

[ -f "$SRC" ] || { echo "missing $SRC -- build first: tectonic -X compile main.tex"; exit 1; }
mkdir -p "$DEST"
cp "$SRC" "$DEST/$NAME"

REV="$(git -C "$RSS" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$RSS" status --porcelain -- "${PDF_SOURCES[@]}" 2>/dev/null)" ]; then REV="$REV+uncommitted"; fi
pages() { python3 -c "
import sys
try:
    import pymupdf
    print(len(pymupdf.open(sys.argv[1])))
except Exception:
    try:
        from pypdf import PdfReader
        print(len(PdfReader(sys.argv[1]).pages))
    except Exception:
        print('?')
" "$1"; }
PAGES="$(pages "$SRC")"

LOG="$DEST/원고_갱신기록.txt"
if [ ! -s "$LOG" ]; then
  {
    echo "Dithered Low-Bit World Models: RSS 2027 Stage 1 원고 사본 갱신 기록 (탭 구분)"
    echo "원본: A_dithered_world_models/paper/rss2027/main.pdf -- build/sync_papers_copy.sh가 복사하고 아래 줄을 덧붙인다"
    echo "초안 PDF는 tectonic(TeX Gyre Termes)으로 만든다. 제출본은 pdfLaTeX(Times, Type 1 글꼴)로 따로 만든다"
    echo
    printf '%s\t%s\t%s\t%s\n' "갱신 시각" "git 커밋" "쪽수" "메모"
  } > "$LOG"
fi
printf '%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M')" "$REV" "$PAGES" "$NOTE" >> "$LOG"
echo "copied $SRC -> $DEST/$NAME ($PAGES pages, $REV)"

# arXiv version (author block, venue-neutral wording) and its upload bundle
SRC_ARXIV="$RSS/arxiv.pdf"
if [ -f "$SRC_ARXIV" ]; then
  mkdir -p "$DEST_ARXIV"
  cp "$SRC_ARXIV" "$DEST_ARXIV/$NAME_ARXIV"
  [ -f "$RSS/arxiv_source.tar.gz" ] && cp "$RSS/arxiv_source.tar.gz" "$DEST_ARXIV/Dithered_Low-Bit_World_Models_arXiv_source.tar.gz"
  [ -f "$RSS/arxiv_metadata.txt" ] && cp "$RSS/arxiv_metadata.txt" "$DEST_ARXIV/arXiv_제출정보.txt"
  PAGES_ARXIV="$(pages "$SRC_ARXIV")"
  LOG_ARXIV="$DEST_ARXIV/원고_갱신기록.txt"
  if [ ! -s "$LOG_ARXIV" ]; then
    {
      echo "Dithered Low-Bit World Models: arXiv 원고 사본 갱신 기록 (탭 구분)"
      echo "원본: A_dithered_world_models/paper/rss2027/arxiv.pdf (main.tex + 저자 블록) -- build/sync_papers_copy.sh가 복사하고 아래 줄을 덧붙인다"
      echo "같이 두는 것: 올릴 소스 묶음(_arXiv_source.tar.gz, arXiv가 pdfLaTeX로 컴파일)과 제출 양식 내용(arXiv_제출정보.txt)"
      echo "이 PDF는 tectonic(TeX Gyre Termes) 빌드다. arXiv가 만드는 PDF(Times)와 쪽수가 한 줄 차이로 다를 수 있다"
      echo
      printf '%s\t%s\t%s\t%s\n' "갱신 시각" "git 커밋" "쪽수" "메모"
    } > "$LOG_ARXIV"
  fi
  printf '%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M')" "$REV" "$PAGES_ARXIV" "$NOTE" >> "$LOG_ARXIV"
  echo "copied $SRC_ARXIV -> $DEST_ARXIV/$NAME_ARXIV ($PAGES_ARXIV pages, $REV)"
fi
