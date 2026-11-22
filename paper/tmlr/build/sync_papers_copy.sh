#!/bin/bash
# Copy the built TMLR manuscript into the author's paper archive (~/Documents/research/papers) and log the version.
#   main.pdf (anonymous TMLR version) -> Dithered_Low-Bit_World_Models/TMLR/Dithered_Low-Bit_World_Models_TMLR_원고.pdf
# PLAN_tmlr.md 5.2: the TMLR archive folder is created only after the user approves it, so the first run needs --create.
# Usage: bash build/sync_papers_copy.sh ["short note for the log"]            (folder must exist)
#        bash build/sync_papers_copy.sh --create ["short note for the log"]   (first time, after the user's OK)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TMLR="$(cd "$HERE/.." && pwd)"
SRC="$TMLR/main.pdf"
DEST="$HOME/Documents/research/papers/Dithered_Low-Bit_World_Models/TMLR"
NAME="Dithered_Low-Bit_World_Models_TMLR_원고.pdf"
CREATE=0
if [ "${1:-}" = "--create" ]; then CREATE=1; shift; fi
NOTE="${1:-}"
# files that determine the PDF; the dirty flag looks only at these
PDF_SOURCES=(main.tex numbers.tex refs.bib math_commands.tex tmlr.sty tmlr.bst fancyhdr.sty figures tables
             build/make_numbers.py build/make_figs.py build/make_tables.py)

[ -f "$SRC" ] || { echo "missing $SRC -- build first: bash build/build.sh"; exit 1; }
if [ ! -d "$DEST" ] && [ "$CREATE" -ne 1 ]; then
  echo "$DEST does not exist. Ask the user first (PLAN_tmlr.md 5.2), then run with --create."; exit 1
fi
mkdir -p "$DEST"
cp "$SRC" "$DEST/$NAME"

REV="$(git -C "$TMLR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$TMLR" status --porcelain -- "${PDF_SOURCES[@]}" 2>/dev/null)" ]; then REV="$REV+uncommitted"; fi
PAGES="$(python3 -I -c "
import sys
try:
    import pymupdf
    print(len(pymupdf.open(sys.argv[1])))
except Exception:
    print('?')
" "$SRC")"

LOG="$DEST/원고_갱신기록.txt"
if [ ! -s "$LOG" ]; then
  {
    echo "Dithered Low-Bit World Models: TMLR 원고 사본 갱신 기록 (탭 구분)"
    echo "원본: A_dithered_world_models/paper/tmlr/main.pdf -- build/sync_papers_copy.sh가 복사하고 아래 줄을 덧붙인다"
    echo
    printf '%s\t%s\t%s\t%s\n' "갱신 시각" "git 커밋" "쪽수" "메모"
  } > "$LOG"
fi
printf '%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M')" "$REV" "$PAGES" "$NOTE" >> "$LOG"
echo "copied $SRC -> $DEST/$NAME ($PAGES pages, $REV)"
