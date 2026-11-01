#!/bin/bash
# Copy the built manuscript PDF into the author's paper archive (~/Documents/research/papers) and log the version.
# Run after every rebuild of main.pdf:   bash build/sync_papers_copy.sh ["short note for the log"]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RSS="$(cd "$HERE/.." && pwd)"
SRC="$RSS/main.pdf"
DEST="$HOME/Documents/research/papers/Dithered_Low-Bit_World_Models/RSS2027"
NAME="Dithered_Low-Bit_World_Models_RSS2027_원고.pdf"
NOTE="${1:-}"
# files that determine the PDF; the dirty flag looks only at these
PDF_SOURCES=(main.tex refs.bib IEEEtran.cls figures build/make_figs.py)

[ -f "$SRC" ] || { echo "missing $SRC -- build first: tectonic -X compile main.tex"; exit 1; }
mkdir -p "$DEST"
cp "$SRC" "$DEST/$NAME"

REV="$(git -C "$RSS" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$RSS" status --porcelain -- "${PDF_SOURCES[@]}" 2>/dev/null)" ]; then REV="$REV+uncommitted"; fi
PAGES="$(python3 -c "
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
" "$SRC")"

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
