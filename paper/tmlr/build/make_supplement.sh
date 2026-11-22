#!/usr/bin/env bash
# make_supplement.sh -- anonymized supplementary material of paper A's TMLR submission, tested in an empty folder.
#
# Usage (from paper/tmlr):  bash build/make_supplement.sh [--no-compile] [--keep-work] [--src DIR] [--out DIR]
#   --no-compile skip typesetting the original and the anonymized manuscript (tectonic) to compare their text
#   --keep-work  keep the work folder (snapshot, traces, logs) under $TMPDIR
#   --src DIR    paper-A folder to package (default: the one that holds this script)
#   --out DIR    output folder (default: <src>/paper/tmlr/supplement, git-ignored)
# Output: <out>/supplementary_material.zip (the upload), <out>/supplementary_material/ (the same files, unpacked),
#         <out>/BUILD_REPORT.md (sizes, anonymization record, scan and test results), <out>/logs/
# Steps
#   1 snapshot  copy the paper-A folder (no .git, no other papers) to a work folder; the originals are only read,
#               and edits made by others while this runs cannot mix two versions into the package
#   2 baseline  run make_numbers, make_tables, verify_numbers --strict and make_figs on the snapshot, recording the
#               files each reads; re-run make_tables and make_figs recording which result files each output used
#   3 assemble  copy what the scripts read plus the code, design documents, logs and stage reports, anonymizing the
#               copies (rules and package contents: build/supplement_tool.py)
#   4 scan      search every file of the package (text, .gz, binary, file names) for identifying strings
#   5 zip       TMLR limit 100 MB
#   6 test      unzip into an empty folder and run the three commands and make_figs there, unchanged, in a clean
#               environment (empty HOME, no inherited variables)
#   7 compare   numbers.tex, tables and figures of the test against the baseline, the shipped copies and the project;
#               typeset the original and the anonymized manuscript and compare the text of the two PDFs
# Exit status: 0 ready; 1 not ready (identifying strings left, a failed test, a difference, or a zip over 100 MB; the
#              package is still written for inspection); 2 the scripts already fail on the unmodified snapshot.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$HERE/supplement_tool.py"
SRC="$(cd "$HERE/../../.." && pwd)"
OUT=""
COMPILE=1
KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --src) SRC="$(cd "$2" && pwd)"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --no-compile) COMPILE=0; shift ;;
    --keep-work) KEEP=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || OUT="$SRC/paper/tmlr/supplement"
mkdir -p "$OUT" && OUT="$(cd "$OUT" && pwd)"
PY="${PYTHON:-python3}"
NAME=supplementary_material
LIMIT=100000000
WORK="$(mktemp -d "${TMPDIR:-/tmp}/a_supplement.XXXXXX")"
cleanup() { if [ "$KEEP" = 1 ]; then echo "work folder kept: $WORK"; else rm -rf "$WORK"; fi; }
trap cleanup EXIT
mkdir -p "$WORK/trace" "$WORK/logs" "$WORK/pkg" "$WORK/baseline" "$WORK/shipped" "$WORK/mpl"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
publish_logs() { rm -rf "$OUT/logs"; cp -R "$WORK/logs" "$OUT/logs"; }

# ---------------------------------------------------------------------------------------------------- 1 snapshot
log "1/7 snapshot of $SRC"
rsync -a --exclude='.git/' --exclude='.DS_Store' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='/paper/tmlr/supplement/' --exclude='/paper/tmlr/arxiv/' \
  --include='/paper/' --include='/paper/tmlr/' --exclude='/paper/*' \
  --exclude='/pilots/' --exclude='/workspace/' "$SRC/" "$WORK/snapshot/" || { echo "snapshot failed" >&2; exit 2; }
GIT="$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo none)"
[ -n "$(git -C "$SRC" status --porcelain 2>/dev/null | head -1)" ] && GIT="$GIT + uncommitted changes"
S="$WORK/snapshot/paper/tmlr"
cp "$S/numbers.tex" "$WORK/shipped/" && cp -R "$S/tables" "$S/figures" "$WORK/shipped/" 2>/dev/null

# ---------------------------------------------------------------------------------------------------- 2 baseline
log "2/7 baseline: the four build scripts on the unmodified snapshot"
BASE=""
for spec in "trace make_numbers make_numbers" "trace make_tables make_tables" \
            "trace verify_numbers verify_numbers --strict" "trace make_figs make_figs" \
            "provenance prov_make_tables make_tables" "provenance prov_make_figs make_figs"; do
  set -- $spec
  mode=$1; tag=$2; script=$3; shift 3
  if [ "$tag" = prov_make_tables ]; then   # outputs of the plain runs, to check the traced re-runs against
    cp "$S/numbers.tex" "$WORK/baseline/" && cp -R "$S/tables" "$S/figures" "$WORK/baseline/"
  fi
  ( cd "$S" && MPLCONFIGDIR="$WORK/mpl" "$PY" "$TOOL" "$mode" "$WORK/trace/$tag.json" "$WORK/snapshot" \
      "build/$script.py" "$@" ) > "$WORK/logs/baseline_$tag.txt" 2>&1
  rc=$?
  BASE="$BASE $tag:$rc"
  if [ $rc -ne 0 ]; then
    echo "The build scripts fail on an unmodified copy of the project ($script.py $*, exit $rc), so there is" >&2
    echo "nothing valid to package. Fix the project first. Last lines:" >&2
    tail -12 "$WORK/logs/baseline_$tag.txt" >&2
    rm -rf "$OUT/last_failed_build" && cp -R "$WORK/logs" "$OUT/last_failed_build"
    echo "Logs: $OUT/last_failed_build/ (a package built earlier, if any, was left as it was)" >&2
    exit 2
  fi
done
"$PY" "$TOOL" compare "$S" "traced re-run vs plain run=$WORK/baseline:must" > "$WORK/logs/provenance_check.txt" 2>&1 \
  || log "warning: the traced re-runs changed an output; the table/figure map of the README may be wrong"
"$PY" "$TOOL" compare "$WORK/baseline" "shipped copies vs regenerated=$WORK/shipped:must" \
  > "$WORK/logs/shipped_vs_regenerated.txt" 2>&1
SHIPPED=$?

# ---------------------------------------------------------------------------------------------------- 3 assemble
log "3/7 assemble and anonymize"
"$PY" "$TOOL" assemble "$WORK/snapshot" "$WORK" "$WORK/pkg/$NAME" "$SRC" > "$WORK/logs/assemble.txt" 2>&1 \
  || { cat "$WORK/logs/assemble.txt" >&2; publish_logs; exit 1; }
cat "$WORK/logs/assemble.txt"

# ---------------------------------------------------------------------------------------------------- 4 scan
log "4/7 scan for identifying strings"
"$PY" "$TOOL" scan "$WORK/pkg/$NAME" "$WORK/scan.json" "$WORK/snapshot" > "$WORK/logs/scan.txt" 2>&1
SCAN=$?
head -12 "$WORK/logs/scan.txt"

# ---------------------------------------------------------------------------------------------------- 5 zip
log "5/7 zip"
( cd "$WORK/pkg" && zip -q -r -9 -X "$WORK/$NAME.zip" "$NAME" ) || { echo "zip failed" >&2; exit 1; }
ZIPB=$(wc -c < "$WORK/$NAME.zip" | tr -d ' ')
UNZB=$(du -sk "$WORK/pkg/$NAME" | cut -f1)
log "    $NAME.zip: $ZIPB bytes ($(( UNZB / 1024 )) MB unpacked)"

# ---------------------------------------------------------------------------------------------------- 6 test
log "6/7 clean-folder test"
T="$WORK/clean_test"
mkdir -p "$T/home"
( cd "$T" && unzip -q "$WORK/$NAME.zip" )
TT="$T/$NAME/paper/tmlr"
( cd "$TT" && env -i PATH="$PATH" HOME="$T/home" MPLCONFIGDIR="$T/home/mpl" TMPDIR="$T/home" sh -c \
  'python3 build/make_numbers.py && python3 build/make_tables.py && python3 build/verify_numbers.py --strict' ) \
  > "$WORK/logs/test_commands.txt" 2>&1
TEST=$?
( cd "$TT" && env -i PATH="$PATH" HOME="$T/home" MPLCONFIGDIR="$T/home/mpl" TMPDIR="$T/home" sh -c \
  'python3 build/make_figs.py' ) > "$WORK/logs/test_make_figs.txt" 2>&1
FIGS=$?
grep -E '^(wrote .*numbers|numbers.tex:|wording|registered|--strict)' "$WORK/logs/test_commands.txt" | sed 's/^/    /'
[ $TEST -eq 0 ] || tail -15 "$WORK/logs/test_commands.txt"
[ $FIGS -eq 0 ] || tail -8 "$WORK/logs/test_make_figs.txt"

# ---------------------------------------------------------------------------------------------------- 7 compare
log "7/7 compare outputs"
"$PY" "$TOOL" compare "$TT" "test vs baseline=$WORK/baseline:must" "test vs shipped copies=$WORK/pkg/$NAME/paper/tmlr:must" \
  "test vs project folder now (information)=$SRC/paper/tmlr" > "$WORK/logs/compare.txt" 2>&1
CMP=$?
sed 's/^/    /' "$WORK/logs/compare.txt" | head -40
TEX=""
if [ "$COMPILE" = 1 ]; then
  if command -v tectonic >/dev/null 2>&1; then
    log "    typesetting the original and the anonymized manuscript"
    mkdir -p "$WORK/tex/orig" "$WORK/tex/anon"
    rsync -a "$S/" "$WORK/tex/orig/" && rsync -a "$TT/" "$WORK/tex/anon/"
    ( cd "$WORK/tex/orig" && tectonic -X compile main.tex ) > "$WORK/logs/tex_orig.txt" 2>&1
    ( cd "$WORK/tex/anon" && tectonic -X compile main.tex ) > "$WORK/logs/tex_anon.txt" 2>&1
    if "$PY" "$TOOL" pdftext "$WORK/tex/orig/main.pdf" "$WORK/tex/anon/main.pdf" > "$WORK/logs/tex_compare.txt" 2>&1
    then TEX="identical"; else TEX="DIFFERENT"; fi
    sed 's/^/    /' "$WORK/logs/tex_compare.txt" | head -12
  else
    TEX="not checked (no tectonic)"
  fi
fi

# ---------------------------------------------------------------------------------------------------- publish
RESULT=READY
[ $SCAN -eq 0 ] || RESULT="NOT READY (identifying strings left)"
[ $TEST -eq 0 ] && [ $FIGS -eq 0 ] || RESULT="NOT READY (clean-folder test failed)"
[ $CMP -eq 0 ] || RESULT="NOT READY (outputs differ)"
[ $SHIPPED -eq 0 ] || RESULT="NOT READY (the project's numbers.tex, tables or figures are not current: rebuild the paper)"
[ "$ZIPB" -le $LIMIT ] || RESULT="NOT READY (zip over 100 MB)"
case "$TEX" in DIFFERENT) RESULT="NOT READY (the anonymized manuscript typesets differently)" ;; esac
rm -rf "$OUT/$NAME" "$OUT/$NAME.zip" "$OUT/last_failed_build"
cp "$WORK/$NAME.zip" "$OUT/" && cp -R "$WORK/pkg/$NAME" "$OUT/$NAME"
cp "$WORK/scan.json" "$WORK/logs/"
publish_logs
"$PY" "$TOOL" report "$WORK" "$OUT" "date=$(date '+%Y-%m-%d %H:%M %Z')" "src=$SRC" "git=$GIT" "result=$RESULT" \
  "zip_bytes=$ZIPB" "baseline=${BASE# }" "test=$TEST" "figs=$FIGS" "compare=$CMP" "tex=$TEX"
log "$RESULT: $OUT/$NAME.zip ($(( ZIPB / 1024 )) kB); report: $OUT/BUILD_REPORT.md"
case "$RESULT" in READY) exit 0 ;; *) exit 1 ;; esac
